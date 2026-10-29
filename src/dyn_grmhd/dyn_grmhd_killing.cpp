//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyn_grmhd_killing.cpp
//! \brief <mhd> energy_variable = killing_t1 | killing_ta (R-154/R-155 item 1, user-approved
//! 2026-10-29; research/notes/KILLING_ENERGY_NOTE.md): the energy equation in the form of the
//! BH-frame Killing energy (HARM-type),
//!   T = sqrt(gamma) (alpha0 E - beta0^k S_k - w D),  w = 1 (killing_t1) or alpha0 (killing_ta),
//! with the static trumpet background's weights (alpha0, beta0^i = b(r) x^i) about the hole.
//!
//! The tau slot keeps holding sqrt(gamma) tau. Since T is, cell by cell, a fixed linear
//! combination of the densitized D, S_k and tau (the background is stationary: BH frame,
//! <cfc> freeze_puncture_mass), evolving tau with the update implied by T's conservative update
//! is evolving T: every RK combination, C2P, floor, dual-energy resync, cold recovery, zone
//! action, prolongation and restriction that changes (D, S, tau) changes T by the same linear
//! map, so none of them needs a separate T <-> tau conversion. Two places change:
//!   (1) KillingFaceFluxes, at the end of CalcFluxes (after FOFC, before the fine/coarse flux
//!       correction): the tau slot's face flux becomes T's,
//!         F_T = alpha0_f (F_tau + F_D) - beta0_f^k F_Sk - w_f F_D,
//!       with the weights at the face (pointwise in the face coordinates: a ghost's face and its
//!       owner's are bitwise equal). The HLLE/LLF flux of T is exactly this combination (the
//!       solvers are linear in the states and fluxes at fixed wave speeds), FOFC's first-order
//!       fluxes are included, and the flux correction then restricts F_T itself, so T is
//!       conserved across refinement levels.
//!   (2) AddKillingTerms, after AddCoordTermsEOS (which applied tau's Valencia sources): the tau
//!       update is replaced by
//!         dtau/dt = (dT/dt + beta0_c^k dS_k/dt + w_c dD/dt)/alpha0_c - dD/dt,
//!         dT/dt = -div F_T + sqrt(gamma)(S_K + S_alpha + S_E + S_beta [- D V^j d_j alpha0]),
//!       dS_k/dt = -div F_S + the Valencia momentum source, dD/dt = -div F_D, the weights at the
//!       cell centre. The sources (KILLING_ENERGY_NOTE Sec. 2; the last term for killing_ta) are
//!       first order in the residual delta alpha = alpha - alpha0, delta beta = beta - beta0,
//!       h_ij = (psi^4 - psi0^4) delta_ij, K_ij - K0_ij (K0_ij = psi0^-2 Ahat0_ij), whose
//!       derivatives are the Dx<NGHOST> of the residual itself (kill_res, ghosts included):
//!         S_K     = alpha S^ij [alpha0 (K_ij - K0_ij) - 1/2 Lie_beta0 h_ij],
//!         S_alpha = -alpha0 S^j d_j delta alpha + delta alpha S^j d_j alpha0
//!                   (= -alpha0^2 S^j d_j(delta alpha/alpha0), without the division),
//!         S_E     = E (beta0^k d_k delta alpha - delta beta^k d_k alpha0),
//!         S_beta  = -S_k (beta0^j d_j delta beta^k - delta beta^j d_j beta0^k).
//! Cells in the excision floor mask (reset by the C2P) and cells with alpha0 < 1e-3 (inside the
//! trumpet throat) keep the Valencia update. The default (valencia) runs none of this.

#include <math.h>

#include <cstdio>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "cfc/cfc.hpp"
#include "cfc/cfc_puncture.hpp"
#include "utils/finite_diff.hpp"
#include "dyn_grmhd.hpp"
#include "metric_faces.hpp"

#include "eos/primitive_solver_hyd.hpp"
#include "eos/primitive-solver/idealgas.hpp"
#include "eos/primitive-solver/eos_compose.hpp"
#include "eos/primitive-solver/eos_hybrid.hpp"
#include "eos/primitive-solver/piecewise_polytrope.hpp"
#include "eos/primitive-solver/reset_floor.hpp"

namespace dyngr {

namespace {

//! the static trumpet background at x (relative to the hole) through the table: alpha0,
//! beta0^i = b x^i, psi0, K0_ij = psi0^-2 Ahat0_ij (packed xx, yy, zz, xy, xz, yz), dalpha0/dr,
//! b and db/dr (metric_faces.hpp TrumpetTableBackground, cfc_puncture.hpp TrumpetBackground)
KOKKOS_INLINE_FUNCTION
void KillingBackground(const DvceArray2D<Real> &d, const Real h, const Real ih, const int n,
                       const Real m_bh, const Real x1, const Real x2, const Real x3,
                       Real &alpha0, Real beta0[3], Real &psi0, Real K0[6], Real &dal0,
                       Real &b, Real &db) {
  const Real r = sqrt(x1*x1 + x2*x2 + x3*x3 + 1.0e-30);
  const Real rho = TrumpetTableRho(d, h, ih, n, r/m_bh);
  Real A0[6], a2, dpsi0;
  cfc::TrumpetBackground(m_bh, x1, x2, x3, m_bh*rho, &psi0, &alpha0, beta0, A0, &a2,
                         &dpsi0, &dal0);
  const Real ip2 = 1.0/(psi0*psi0);
  for (int q = 0; q < 6; ++q) K0[q] = A0[q]*ip2;
  b = 0.75*sqrt(3.0)/(m_bh*rho*rho*rho);
  db = -3.0*b*alpha0/r;
}

//! the weights: the trumpet background, or (killing_weights = boost) the flat comoving frame's
//! d_t': alpha0 = 1, beta0 = xidot (constant), psi0 = 1, K0 = 0, no gradients
KOKKOS_INLINE_FUNCTION
void KWeights(const bool boost, const Real fv0, const Real fv1, const Real fv2,
              const DvceArray2D<Real> &d, const Real h, const Real ih, const int n,
              const Real m_bh, const Real x1, const Real x2, const Real x3,
              Real &alpha0, Real beta0[3], Real &psi0, Real K0[6], Real &dal0,
              Real &b, Real &db) {
  if (boost) {
    alpha0 = 1.0; psi0 = 1.0; dal0 = 0.0; b = 0.0; db = 0.0;
    beta0[0] = fv0; beta0[1] = fv1; beta0[2] = fv2;
    for (int q = 0; q < 6; ++q) K0[q] = 0.0;
    return;
  }
  KillingBackground(d, h, ih, n, m_bh, x1, x2, x3, alpha0, beta0, psi0, K0, dal0, b, db);
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void DynGRMHD::KillingFaceFluxes
//! \brief the tau slot's face flux -> F_T = alpha0_f (F_tau + F_D) - beta0_f^k F_Sk - w_f F_D on
//! every face the update reads (the interior faces, the block's boundary faces included)
void DynGRMHD::KillingFaceFluxes() {
  auto *pcfc = pmy_pack->pcfc;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmb = pmy_pack->nmb_thispack;
  auto &size = pmy_pack->pmb->mb_size;
  const Real cx = pcfc->bg_xbh_[0], cy = pcfc->bg_xbh_[1], cz = pcfc->bg_xbh_[2];
  const Real m_bh = pcfc->puncture_mass_;
  const bool ta = (energy_variable == EnergyVariable::killing_ta);
  const bool boost = kill_boost;
  const Real fv0 = pmy_pack->padm->frame_vel_u[0], fv1 = pmy_pack->padm->frame_vel_u[1];
  const Real fv2 = pmy_pack->padm->frame_vel_u[2];
  auto &td = ttab.d;
  const Real th = ttab.h, tih = ttab.ih;
  const int nt = ttab.n;
  for (int dir = 0; dir < 3; ++dir) {
    auto &f = (dir == 0) ? pmy_pack->pmhd->uflx.x1f :
              ((dir == 1) ? pmy_pack->pmhd->uflx.x2f : pmy_pack->pmhd->uflx.x3f);
    const int si = (dir == 0), sj = (dir == 1), sk = (dir == 2);
    par_for("kill_faces", DevExeSpace(), 0, nmb-1, ks, ke + sk, js, je + sj, is, ie + si,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      const auto &sz = size.d_view(m);
      const Real x = (si ? LeftEdgeX(i-is, nx1, sz.x1min, sz.x1max)
                         : CellCenterX(i-is, nx1, sz.x1min, sz.x1max)) - cx;
      const Real y = (sj ? LeftEdgeX(j-js, nx2, sz.x2min, sz.x2max)
                         : CellCenterX(j-js, nx2, sz.x2min, sz.x2max)) - cy;
      const Real z = (sk ? LeftEdgeX(k-ks, nx3, sz.x3min, sz.x3max)
                         : CellCenterX(k-ks, nx3, sz.x3min, sz.x3max)) - cz;
      Real al0, b0[3], psi0, K0[6], dal0, b, db;
      KWeights(boost, fv0, fv1, fv2, td, th, tih, nt, m_bh, x, y, z, al0, b0, psi0, K0, dal0,
               b, db);
      const Real fD = f(m, IDN, k, j, i);
      const Real w = ta ? al0 : 1.0;
      f(m, IEN, k, j, i) = al0*(f(m, IEN, k, j, i) + fD)
                           - (b0[0]*f(m, IM1, k, j, i) + b0[1]*f(m, IM2, k, j, i)
                              + b0[2]*f(m, IM3, k, j, i))
                           - w*fD;
    });
  }
  if (kill_probe_cycles > 0) KillingFaceProbe();
}

//----------------------------------------------------------------------------------------
//! n void DynGRMHD::KillingFaceProbe
//! rief <dyn_grmhd> killing_probe_cycles / face_probe_box (the ghost/owner probe of the
//! design rule): every face KillingFaceFluxes wrote inside the box, in every MeshBlock that
//! wrote it, one record per stage: KPROBE cycle stage dir gid i j k level x y z F_D F_S1 F_S2
//! F_S3 F_T (%.17g; T's flux after the transform), to kprobe_rank<NNNNN>.txt. The two copies
//! of a face on a same-level block boundary (each block computes its own boundary faces) must
//! agree bitwise (scripts/c147_kprobe_check.py).
void DynGRMHD::KillingFaceProbe() {
  const int cyc = pmy_pack->pmesh->ncycle;
  if (kill_probe_c0 < 0) kill_probe_c0 = cyc;
  if (cyc >= kill_probe_c0 + kill_probe_cycles) return;
  const int stage = ++kill_probe_stage;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmb = pmy_pack->nmb_thispack, gid0 = pmy_pack->gids;
  auto &size = pmy_pack->pmb->mb_size;
  auto &lev = pmy_pack->pmb->mb_lev;
  Kokkos::Array<Real, 6> pb;
  for (int q = 0; q < 6; ++q) pb[q] = fm_probe_box[q];
  const int nrec = 15, cap = 2000000;
  DvceArray2D<Real> buf("kprobe_buf", cap, nrec);
  DvceArray1D<int> cnt("kprobe_cnt", 1);
  for (int dir = 0; dir < 3; ++dir) {
    auto &f = (dir == 0) ? pmy_pack->pmhd->uflx.x1f :
              ((dir == 1) ? pmy_pack->pmhd->uflx.x2f : pmy_pack->pmhd->uflx.x3f);
    const int si = (dir == 0), sj = (dir == 1), sk = (dir == 2);
    par_for("kill_probe", DevExeSpace(), 0, nmb-1, ks, ke + sk, js, je + sj, is, ie + si,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      const auto &sz = size.d_view(m);
      const Real x = si ? LeftEdgeX(i-is, nx1, sz.x1min, sz.x1max)
                        : CellCenterX(i-is, nx1, sz.x1min, sz.x1max);
      const Real y = sj ? LeftEdgeX(j-js, nx2, sz.x2min, sz.x2max)
                        : CellCenterX(j-js, nx2, sz.x2min, sz.x2max);
      const Real z = sk ? LeftEdgeX(k-ks, nx3, sz.x3min, sz.x3max)
                        : CellCenterX(k-ks, nx3, sz.x3min, sz.x3max);
      if (!(x > pb[0] && x < pb[1] && y > pb[2] && y < pb[3] && z > pb[4] && z < pb[5])) {
        return;
      }
      const int r = Kokkos::atomic_fetch_add(&cnt(0), 1);
      if (r >= cap) return;
      Real v[15] = {static_cast<Real>(dir), static_cast<Real>(gid0 + m),
                    static_cast<Real>(i), static_cast<Real>(j), static_cast<Real>(k),
                    static_cast<Real>(lev.d_view(m)), x, y, z,
                    f(m, IDN, k, j, i), f(m, IM1, k, j, i), f(m, IM2, k, j, i),
                    f(m, IM3, k, j, i), f(m, IEN, k, j, i), 0.0};
      for (int q = 0; q < 15; ++q) buf(r, q) = v[q];
    });
  }
  auto hc = Kokkos::create_mirror_view_and_copy(HostMemSpace(), cnt);
  const int nr = (hc(0) < cap) ? hc(0) : cap;
  auto hb = Kokkos::create_mirror_view_and_copy(HostMemSpace(), buf);
  char fn[64];
  snprintf(fn, sizeof(fn), "kprobe_rank%05d.txt", global_variable::my_rank);
  FILE *fp = fopen(fn, "a");
  if (fp == nullptr) return;
  for (int r = 0; r < nr; ++r) {
    fprintf(fp, "KPROBE %d %d", cyc, stage);
    for (int q = 0; q < 6; ++q) fprintf(fp, " %d", static_cast<int>(hb(r, q)));
    for (int q = 6; q < 14; ++q) fprintf(fp, " %.17g", hb(r, q));
    fprintf(fp, "\n");
  }
  if (hc(0) > cap) fprintf(fp, "KPROBE_OVERFLOW %d %d %d\n", cyc, stage, hc(0));
  fclose(fp);
}

//----------------------------------------------------------------------------------------
//! \fn void DynGRMHDPS::AddKillingTerms
//! \brief see the file header: rhs(IEN) += dt [dtau/dt (from T) - (-div F_T + sqrt(gamma)
//! tau's Valencia source)], the latter being what MHD::RKUpdate and AddCoordTermsEOS applied
template<class EOSPolicy, class ErrorPolicy>
template<int NGHOST>
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddKillingTerms(const DvceArray5D<Real> &prim,
    const DvceArray5D<Real> &bcc, const Real dt, DvceArray5D<Real> &rhs) {
  auto *pcfc = pmy_pack->pcfc;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmb = pmy_pack->nmb_thispack;
  auto &size = pmy_pack->pmb->mb_size;
  auto &adm = pmy_pack->padm->adm;
  auto &eos_ = eos.ps.GetEOS();
  const int nhyd = pmy_pack->pmhd->nmhd;
  const int nscal = pmy_pack->pmhd->nscalars;
  const Real mb = eos.ps.GetEOS().GetBaryonMass();
  const Real cx = pcfc->bg_xbh_[0], cy = pcfc->bg_xbh_[1], cz = pcfc->bg_xbh_[2];
  const Real m_bh = pcfc->puncture_mass_;
  const bool ta = (energy_variable == EnergyVariable::killing_ta);
  const bool boost = kill_boost;
  const Real fv0 = pmy_pack->padm->frame_vel_u[0], fv1 = pmy_pack->padm->frame_vel_u[1];
  const Real fv2 = pmy_pack->padm->frame_vel_u[2];
  const bool exc = pmy_pack->pcoord->coord_data.bh_excise;
  auto &xfloor = pmy_pack->pcoord->excision_floor;
  auto &td = ttab.d;
  const Real th = ttab.h, tih = ttab.ih;
  const int nt = ttab.n;
  auto &res = kill_res;
  auto &fx1_ = pmy_pack->pmhd->uflx.x1f;
  auto &fx2_ = pmy_pack->pmhd->uflx.x2f;
  auto &fx3_ = pmy_pack->pmhd->uflx.x3f;
  const int imap[3][3] = {
    {S11, S12, S13},
    {S12, S22, S23},
    {S13, S23, S33}
  };

  // the residual at every cell, ghosts included (the adm metric is exchanged there)
  par_for("kill_residual", DevExeSpace(), 0, nmb-1, 0, res.extent_int(2)-1,
          0, res.extent_int(3)-1, 0, res.extent_int(4)-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const auto &sz = size.d_view(m);
    const Real x = CellCenterX(i-is, nx1, sz.x1min, sz.x1max) - cx;
    const Real y = CellCenterX(j-js, nx2, sz.x2min, sz.x2max) - cy;
    const Real z = CellCenterX(k-ks, nx3, sz.x3min, sz.x3max) - cz;
    Real al0, b0[3], psi0, K0[6], dal0, b, db;
    KWeights(boost, fv0, fv1, fv2, td, th, tih, nt, m_bh, x, y, z, al0, b0, psi0, K0, dal0,
             b, db);
    const Real p2 = psi0*psi0;
    res(m, 0, k, j, i) = adm.alpha(m, k, j, i) - al0;
    for (int a = 0; a < 3; ++a) res(m, 1 + a, k, j, i) = adm.beta_u(m, a, k, j, i) - b0[a];
    res(m, 4, k, j, i) = adm.psi4(m, k, j, i) - p2*p2;
  });

  par_for("kill_src", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    if (exc && xfloor(m, k, j, i)) return;
    const auto &sz = size.d_view(m);
    const Real xr[3] = {CellCenterX(i-is, nx1, sz.x1min, sz.x1max) - cx,
                        CellCenterX(j-js, nx2, sz.x2min, sz.x2max) - cy,
                        CellCenterX(k-ks, nx3, sz.x3min, sz.x3max) - cz};
    Real al0, b0[3], psi0, K0[6], dal0, bb, db;
    KWeights(boost, fv0, fv1, fv2, td, th, tih, nt, m_bh, xr[0], xr[1], xr[2], al0, b0, psi0,
             K0, dal0, bb, db);
    if (!(al0 >= 1.0e-3)) return;
    const Real r = sqrt(xr[0]*xr[0] + xr[1]*xr[1] + xr[2]*xr[2] + 1.0e-30);
    Real dal0_d[3], dB0[3][3];   // d_a alpha0, d_a beta0^b
    for (int a = 0; a < 3; ++a) {
      dal0_d[a] = dal0*xr[a]/r;
      for (int c = 0; c < 3; ++c) dB0[a][c] = (a == c ? bb : 0.0) + db*xr[a]*xr[c]/r;
    }
    const int kmap[3][3] = {{0, 3, 4}, {3, 1, 5}, {4, 5, 2}};

    // the metric and its derivatives (as AddCoordTermsEOS)
    Real g3d[NSPMETRIC] = {adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
                           adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
                           adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i)};
    const Real alpha = adm.alpha(m, k, j, i);
    Real detg = adm::SpatialDet(g3d[S11], g3d[S12], g3d[S13],
                                g3d[S22], g3d[S23], g3d[S33]);
    Real vol = sqrt(detg);
    Real g3u[NSPMETRIC] = {0.};
    adm::SpatialInv(1.0/detg, g3d[S11], g3d[S12], g3d[S13], g3d[S22], g3d[S23], g3d[S33],
                    &g3u[S11], &g3u[S12], &g3u[S13], &g3u[S22], &g3u[S23], &g3u[S33]);
    Real idx[] = {1./sz.dx1, 1./sz.dx2, 1./sz.dx3};
    Real dalpha_d[3], dbeta_du[3][3], dg_ddd[3][3][3];
    Real dda[3], ddb[3][3], dH[3];   // d_a delta alpha, d_a delta beta^c, d_a (psi^4 - psi0^4)
    for (int a = 0; a < 3; a++) {
      dalpha_d[a] = Dx<NGHOST>(a, idx, adm.alpha, m, k, j, i);
      dda[a] = Dx<NGHOST>(a, idx, res, m, 0, k, j, i);
      dH[a] = Dx<NGHOST>(a, idx, res, m, 4, k, j, i);
      for (int c = 0; c < 3; c++) {
        dbeta_du[a][c] = Dx<NGHOST>(a, idx, adm.beta_u, m, c, k, j, i);
        ddb[a][c] = Dx<NGHOST>(a, idx, res, m, 1 + c, k, j, i);
        for (int e = 0; e < 3; ++e) {
          dg_ddd[a][c][e] = Dx<NGHOST>(a, idx, adm.g_dd, m, c, e, k, j, i);
        }
      }
    }

    // the fluid (as AddCoordTermsEOS)
    Real prim_pt[NPRIM] = {0.0};
    prim_pt[PRH] = prim(m, IDN, k, j, i)/mb;
    prim_pt[PVX] = prim(m, IVX, k, j, i);
    prim_pt[PVY] = prim(m, IVY, k, j, i);
    prim_pt[PVZ] = prim(m, IVZ, k, j, i);
    for (int s = 0; s < nscal; s++) {
      prim_pt[PYF + s] = prim(m, nhyd + s, k, j, i);
    }
    prim_pt[PPR] = prim(m, IPR, k, j, i);
    prim_pt[PTM] = eos_.GetTemperatureFromP(prim_pt[PRH], prim_pt[PPR], &prim_pt[PYF]);
    Real H =
      prim(m, IDN, k, j, i)*eos_.GetEnthalpy(prim_pt[PRH], prim_pt[PTM], &prim_pt[PYF]);
    Real usq = Primitive::SquareVector(&prim_pt[PVX], g3d);
    Real const Wsq = 1.0 + usq;
    Real const W = sqrt(Wsq);
    Real B_u[NMAG] = {bcc(m, IBX, k, j, i)/vol,
                      bcc(m, IBY, k, j, i)/vol,
                      bcc(m, IBZ, k, j, i)/vol};
    Real Bv = 0.0;
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3; b++) {
        Bv += adm.g_dd(m,a,b,k,j,i)*prim_pt[PVX + a]*B_u[b];
      }
    }
    Real Bsq = Primitive::SquareVector(B_u, g3d);
    Bv = Bv/W;
    Real bsq = Bv*Bv + Bsq/Wsq;
    Real E = (H*Wsq + Bsq) - prim_pt[PPR] - 0.5*bsq;
    Real S_d[3] = {0.0};
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3; b++) {
        S_d[a] += ((H*Wsq + Bsq)*prim_pt[PVX + b]/W - Bv*B_u[b])*g3d[imap[a][b]];
      }
    }
    Real S_uu[3][3];
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b <= a; b++) {
        S_uu[a][b] = (H + Bsq/Wsq)*prim_pt[PVX + a]*prim_pt[PVX + b]
                      - B_u[a]*B_u[b]/Wsq
                      - Bv*(B_u[a]*prim_pt[PVX + b] + B_u[b]*prim_pt[PVX + a])/W
                      + (prim_pt[PPR] + 0.5*bsq)*g3u[imap[a][b]];
        S_uu[b][a] = S_uu[a][b];
      }
    }

    // tau's Valencia source (applied by AddCoordTermsEOS) and the momentum's
    Real src_tau = 0.0, src_S[3];
    for (int a = 0; a < 3; a++) {
      src_S[a] = -E*dalpha_d[a];
      for (int b = 0; b < 3; b++) {
        src_tau += alpha*adm.vK_dd(m, a, b, k, j, i)*S_uu[a][b]
                   - g3u[imap[a][b]]*S_d[a]*dalpha_d[b];
        src_S[a] += S_d[b]*dbeta_du[a][b];
        for (int c = 0; c < 3; c++) {
          src_S[a] += 0.5*alpha*S_uu[b][c]*dg_ddd[a][b][c];
        }
      }
    }

    // T's source, first order in the residual (KILLING_ENERGY_NOTE Sec. 2)
    const Real dal = res(m, 0, k, j, i), Hr = res(m, 4, k, j, i);
    const Real dbe[3] = {res(m, 1, k, j, i), res(m, 2, k, j, i), res(m, 3, k, j, i)};
    Real S_u[3];
    for (int a = 0; a < 3; a++) {
      S_u[a] = 0.0;
      for (int b = 0; b < 3; b++) S_u[a] += g3u[imap[a][b]]*S_d[b];
    }
    Real b0dH = 0.0;
    for (int c = 0; c < 3; c++) b0dH += b0[c]*dH[c];
    Real sK = 0.0;
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3; b++) {
        const Real lie_h = (a == b ? b0dH : 0.0) + Hr*(dB0[a][b] + dB0[b][a]);
        sK += S_uu[a][b]*(al0*(adm.vK_dd(m, a, b, k, j, i) - K0[kmap[a][b]]) - 0.5*lie_h);
      }
    }
    sK *= alpha;
    Real sA = 0.0, sE = 0.0, sB = 0.0;
    for (int a = 0; a < 3; a++) {
      sA += S_u[a]*(dal*dal0_d[a] - al0*dda[a]);
      sE += b0[a]*dda[a] - dbe[a]*dal0_d[a];
      Real lie_b = 0.0;   // (Lie_beta0 delta beta)^a
      for (int c = 0; c < 3; c++) lie_b += b0[c]*ddb[c][a] - dbe[c]*dB0[c][a];
      sB -= S_d[a]*lie_b;
    }
    sE *= E;
    Real sT = sK + sA + sE + sB;
    if (ta) {   // -D V^j d_j alpha0, V^j = alpha v^j - beta^j
      const Real D = prim(m, IDN, k, j, i)*W;
      for (int a = 0; a < 3; a++) {
        sT -= D*(alpha*prim_pt[PVX + a]/W - adm.beta_u(m, a, k, j, i))*dal0_d[a];
      }
    }

    // the flux divergences (densitized, flux-corrected: the update's own)
    auto divf = [&](int n) {
      return (fx1_(m, n, k, j, i+1) - fx1_(m, n, k, j, i))/sz.dx1
           + (fx2_(m, n, k, j+1, i) - fx2_(m, n, k, j, i))/sz.dx2
           + (fx3_(m, n, k+1, j, i) - fx3_(m, n, k, j, i))/sz.dx3;
    };
    const Real dT = -divf(IEN) + vol*sT;
    const Real dD = -divf(IDN);
    Real b0dS = 0.0;
    for (int a = 0; a < 3; a++) b0dS += b0[a]*(-divf(IM1 + a) + vol*src_S[a]);
    const Real w = ta ? al0 : 1.0;
    const Real dtau = (dT + b0dS + w*dD)/al0 - dD;
    const Real dtau_applied = -divf(IEN) + vol*src_tau;
    rhs(m, IEN, k, j, i) += dt*(dtau - dtau_applied);
  });
}

// instantiations (as AddCoordTermsEOS)
#define INSTANTIATE_KILLING_TERMS(EOSPolicy, ErrorPolicy) \
template void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddKillingTerms<2>( \
    const DvceArray5D<Real> &, const DvceArray5D<Real> &, const Real, DvceArray5D<Real> &); \
template void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddKillingTerms<3>( \
    const DvceArray5D<Real> &, const DvceArray5D<Real> &, const Real, DvceArray5D<Real> &); \
template void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddKillingTerms<4>( \
    const DvceArray5D<Real> &, const DvceArray5D<Real> &, const Real, DvceArray5D<Real> &);

INSTANTIATE_KILLING_TERMS(Primitive::IdealGas, Primitive::ResetFloor)
INSTANTIATE_KILLING_TERMS(Primitive::PiecewisePolytrope, Primitive::ResetFloor)
INSTANTIATE_KILLING_TERMS(Primitive::EOSCompOSE<Primitive::NormalLogs>, Primitive::ResetFloor)
INSTANTIATE_KILLING_TERMS(Primitive::EOSCompOSE<Primitive::NQTLogs>, Primitive::ResetFloor)
INSTANTIATE_KILLING_TERMS(Primitive::EOSHybrid<Primitive::NormalLogs>, Primitive::ResetFloor)
INSTANTIATE_KILLING_TERMS(Primitive::EOSHybrid<Primitive::NQTLogs>, Primitive::ResetFloor)

}  // namespace dyngr
