//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyn_grmhd_taubudget.cpp
//! \brief <mhd> tau_budget_every (R-184 items 1-4, user-approved 2026-10-30): the per-cell tau
//! budget of a Killing formulation against Valencia's tau update from the same state.
//! Diagnostic only: separate kernels before and after AddKillingTerms (DynGRMHD::AddCoordTerms),
//! none of which writes the solution.
//!
//! Per cell and stage (rates; "Valencia" = -div F_tau^V + sqrt(g) Sigma_tau, the update the tau
//! slot would get without the Killing rebuild):
//!   Delta = dtau_K - dtau_V = (u_after - u_before)/dt - div F_T + div F_tau^V,
//! with u the tau slot around AddKillingTerms (which adds dt (dtau_K - (-div F_T + sqrt(g)
//! Sigma_tau)), so Sigma_tau cancels), F_T the update's own face flux and F_tau^V the copy of
//! the Valencia face flux taken in KillingFaceFluxes before its transform. The copy is not
//! flux-corrected: at a face where its transform differs from F_T beyond round-off (a coarse
//! face the fine/coarse correction replaced), F_tau^V is inverted from the corrected fluxes,
//! F_tau^V = (F_T + b_f.F_S + w_f F_D)/a_f - F_D, and the cell counted (nfb).
//!   killing_full (item 1): A = -1/2 sqrt(g) S^ij V_ij, V_ij = gamma-dot_ij + 2 alpha K_ij
//!     - (Lie_beta' gamma)_ij, with the code's gamma-dot (the residual chord plus the analytic
//!     background rate, as AddKillingFullTerms; R-186: the rate also on a stage without a chord),
//!     adm.vK_dd, and
//!     Dx<NGHOST> of adm.beta_u, adm.g_dd. The background part V^0 is the trumpet's own
//!     (psi0, alpha0, beta0 + xidot, K0 = psi0^-2 Ahat0 from the table about bg_xbh_; its
//!     gamma-dot 4 psi0^3 xidot.grad psi0 where the code applies the rate; Dx<NGHOST> of the
//!     table's psi0^4, beta0 at the stencil's cell centres), V^res = V - V^0; each split into
//!     the trace part -1/6 sqrt(g) (gamma_ij S^ij)(gamma^ij V_ij) and the rest. B = Delta - A.
//!   comoving killing_t1 (item 2): the frame-advection pairing at the T level,
//!     M_T = S_motion - S_motion,disc, S_motion = sqrt(g) s_W (kill_scr 0), S_motion,disc =
//!     sum_a [(w_f - w_c) F^xi]_{f-}^{f+} / dx_a over the T weights (alpha0 on sqrt(g) E,
//!     -beta0^k on sqrt(g) S_k), F^xi = -xidot^a (U_c + U_nb)/2 the central frame advection
//!     of the two adjacent cells' (sqrt(g) E, sqrt(g) S_k) (the source's own fluid values; the
//!     Riemann solver's reconstructed states are not kept), entering tau as M_T/alpha0_c;
//!     R = Delta - M_T/alpha0_c.
//!   item 4: |Delta|, its pre-division form alpha0 |Delta| (killing_full: alpha), <alpha0>.
//!   valencia (item 3): D and D eps only.
//! Bins (one KTB line per stage, rank 0, sums x dV over the interior cells and the ranks):
//!   star: rho/rho_c >= 0.5, [0.1, 0.5), [0.01, 0.1), [1e-3, 0.01) x (|x - x_c| <= 0.8 R*,
//!     > 0.8 R*), x_c the rho dV centroid of rho >= 0.01 rho_c (this stage);
//!   shells: r_iso about the hole between 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 13, 16, 20.
//! The regrid flag: a MeshBlock created or deleted since the previous reporting cycle.

#include <math.h>

#include <cstdio>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
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

// tb_c components: 0-15 reduced per bin (x dV), 16 the bin, 17-20 A (Pre), 21 the snapshot
constexpr int kTbBin = 16, kTbA = 17, kTbSnap = 21;

//! the trumpet about the hole through the table, as dyn_grmhd_killing.cpp KillingBackground
//! (the same calls: t1's weights bitwise), plus dpsi0/dr
KOKKOS_INLINE_FUNCTION
void TbBackground(const DvceArray2D<Real> &d, const Real h, const Real ih, const int n,
                  const Real m_bh, const Real x1, const Real x2, const Real x3,
                  Real &alpha0, Real beta0[3], Real &psi0, Real K0[6], Real &dpsi0) {
  const Real r = sqrt(x1*x1 + x2*x2 + x3*x3 + 1.0e-30);
  const Real rho = TrumpetTableRho(d, h, ih, n, r/m_bh);
  Real A0[6], a2, dal0;
  cfc::TrumpetBackground(m_bh, x1, x2, x3, m_bh*rho, &psi0, &alpha0, beta0, A0, &a2,
                         &dpsi0, &dal0);
  const Real ip2 = 1.0/(psi0*psi0);
  for (int q = 0; q < 6; ++q) K0[q] = A0[q]*ip2;
}

//! the fluid at a cell, as AddCoordTermsEOS / AddKillingFullTerms: E, S_i, S^ij, W, eps rho
template<class EOS>
KOKKOS_INLINE_FUNCTION
void TbFluid(const EOS &eos_, const Real mb, const int nhyd, const int nscal,
             const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc,
             const Real g3d[NSPMETRIC], const Real g3u[NSPMETRIC], const Real vol,
             const int m, const int k, const int j, const int i,
             Real &E, Real S_d[3], Real S_uu[3][3], Real &W, Real &ein) {
  const int imap[3][3] = {{S11, S12, S13}, {S12, S22, S23}, {S13, S23, S33}};
  Real prim_pt[NPRIM] = {0.0};
  prim_pt[PRH] = prim(m, IDN, k, j, i)/mb;
  prim_pt[PVX] = prim(m, IVX, k, j, i);
  prim_pt[PVY] = prim(m, IVY, k, j, i);
  prim_pt[PVZ] = prim(m, IVZ, k, j, i);
  for (int s = 0; s < nscal; s++) prim_pt[PYF + s] = prim(m, nhyd + s, k, j, i);
  prim_pt[PPR] = prim(m, IPR, k, j, i);
  prim_pt[PTM] = eos_.GetTemperatureFromP(prim_pt[PRH], prim_pt[PPR], &prim_pt[PYF]);
  const Real H =
    prim(m, IDN, k, j, i)*eos_.GetEnthalpy(prim_pt[PRH], prim_pt[PTM], &prim_pt[PYF]);
  const Real usq = Primitive::SquareVector(&prim_pt[PVX], g3d);
  const Real Wsq = 1.0 + usq;
  W = sqrt(Wsq);
  Real B_u[NMAG] = {bcc(m, IBX, k, j, i)/vol, bcc(m, IBY, k, j, i)/vol,
                    bcc(m, IBZ, k, j, i)/vol};
  Real Bv = 0.0;
  for (int a = 0; a < 3; a++) {
    for (int b = 0; b < 3; b++) Bv += g3d[imap[a][b]]*prim_pt[PVX + a]*B_u[b];
  }
  const Real Bsq = Primitive::SquareVector(B_u, g3d);
  Bv = Bv/W;
  const Real bsq = Bv*Bv + Bsq/Wsq;
  E = (H*Wsq + Bsq) - prim_pt[PPR] - 0.5*bsq;
  for (int a = 0; a < 3; a++) {
    S_d[a] = 0.0;
    for (int b = 0; b < 3; b++) {
      S_d[a] += ((H*Wsq + Bsq)*prim_pt[PVX + b]/W - Bv*B_u[b])*g3d[imap[a][b]];
    }
  }
  for (int a = 0; a < 3; a++) {
    for (int b = 0; b <= a; b++) {
      S_uu[a][b] = (H + Bsq/Wsq)*prim_pt[PVX + a]*prim_pt[PVX + b]
                    - B_u[a]*B_u[b]/Wsq
                    - Bv*(B_u[a]*prim_pt[PVX + b] + B_u[b]*prim_pt[PVX + a])/W
                    + (prim_pt[PPR] + 0.5*bsq)*g3u[imap[a][b]];
      S_uu[b][a] = S_uu[a][b];
    }
  }
  ein = eos_.GetEnergy(prim_pt[PRH], prim_pt[PTM], &prim_pt[PYF]) - prim(m, IDN, k, j, i);
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn bool DynGRMHD::TauBudgetOn
//! \brief this cycle reports: every tau_budget_every cycles, the step's start time in
//! [tau_budget_tmin, tau_budget_tmax]
bool DynGRMHD::TauBudgetOn() const {
  if (tb_every <= 0) return false;
  const auto *pm = pmy_pack->pmesh;
  return (pm->ncycle % tb_every == 0) && pm->time >= tb_tmin && pm->time <= tb_tmax;
}

//----------------------------------------------------------------------------------------
//! \fn void DynGRMHD::TauBudgetFaceCopy
//! \brief the tau slot's face flux (Valencia's, after FOFC), before KillingFaceFluxes
//! transforms it into F_T, on the faces the transform writes
void DynGRMHD::TauBudgetFaceCopy() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pmy_pack->nmb_thispack;
  auto &fv = tb_fv;
  for (int dir = 0; dir < 3; ++dir) {
    auto &f = (dir == 0) ? pmy_pack->pmhd->uflx.x1f :
              ((dir == 1) ? pmy_pack->pmhd->uflx.x2f : pmy_pack->pmhd->uflx.x3f);
    const int si = (dir == 0), sj = (dir == 1), sk = (dir == 2);
    par_for("tb_face_copy", DevExeSpace(), 0, nmb-1, ks, ke + sk, js, je + sj, is, ie + si,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      fv(m, dir, k-ks, j-js, i-is) = f(m, IEN, k, j, i);
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void DynGRMHDPS::TauBudgetPre
//! \brief before AddKillingTerms: the tau slot (snapshot); killing_full: A's four parts;
//! comoving t1: sqrt(g) E, sqrt(g) S_k at the cells and one ghost layer
template<class EOSPolicy, class ErrorPolicy>
template<int NGHOST>
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::TauBudgetPre(const DvceArray5D<Real> &prim,
    const DvceArray5D<Real> &bcc, const DvceArray5D<Real> &rhs) {
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
  auto &tc = tb_c;
  if (energy_variable == EnergyVariable::valencia) return;
  par_for("tb_snap", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    tc(m, kTbSnap, k-ks, j-js, i-is) = rhs(m, IEN, k, j, i);
  });

  if (kill_comov) {   // the frame pairing's cell values (the T source's own fluid)
    auto &tu = tb_u;
    par_for("tb_u", DevExeSpace(), 0, nmb-1, ks-1, ke+1, js-1, je+1, is-1, ie+1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real g3d[NSPMETRIC] = {adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
                             adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
                             adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i)};
      const Real detg = adm::SpatialDet(g3d[S11], g3d[S12], g3d[S13],
                                        g3d[S22], g3d[S23], g3d[S33]);
      const Real vol = sqrt(detg);
      Real g3u[NSPMETRIC];
      adm::SpatialInv(1.0/detg, g3d[S11], g3d[S12], g3d[S13], g3d[S22], g3d[S23], g3d[S33],
                      &g3u[S11], &g3u[S12], &g3u[S13], &g3u[S22], &g3u[S23], &g3u[S33]);
      Real E, S_d[3], S_uu[3][3], W, ein;
      TbFluid(eos_, mb, nhyd, nscal, prim, bcc, g3d, g3u, vol, m, k, j, i, E, S_d, S_uu, W,
              ein);
      tu(m, 0, k, j, i) = vol*E;
      for (int a = 0; a < 3; ++a) tu(m, 1 + a, k, j, i) = vol*S_d[a];
    });
  }
  if (energy_variable != EnergyVariable::killing_full) return;

  // killing_full: the inputs of AddKillingFullTerms' gamma-dot (dyn_grmhd_killing.cpp), copied
  auto *pcfc = pmy_pack->pcfc;
  auto &pv = kf_prev;
  const Real dtm = kf_tstage - kf_tprev;
  const bool dtg = kf_dtg && kf_have_prev && dtm != 0.0;
  const Real idtm = dtg ? 1.0/dtm : 0.0;
  auto &psi0 = pcfc->u_psi0;
  auto &ap0 = pcfc->u_alpha0_psi0;
  const bool rsc = pcfc->rs_on_ && pcfc->rs_f_ != 1.0;
  const Real rsf = pcfc->rs_f_;
  auto &dps = pcfc->delta_psi;
  const Real gx[3] = {pmy_pack->padm->frame_vel_u[0], pmy_pack->padm->frame_vel_u[1],
                      pmy_pack->padm->frame_vel_u[2]};
  const bool gauge = (gx[0] != 0.0) || (gx[1] != 0.0) || (gx[2] != 0.0);
  const bool bgm = kf_bgmove;
  const bool rate = kf_dtg && (bgm || kf_xdd[0] != 0.0 || kf_xdd[1] != 0.0 || kf_xdd[2] != 0.0);
  const Real cx = pcfc->bg_xbh_[0], cy = pcfc->bg_xbh_[1], cz = pcfc->bg_xbh_[2];
  const Real m_bh = pcfc->puncture_mass_;
  auto &td = ttab.d;
  const Real th = ttab.h, tih = ttab.ih;
  const int nt = ttab.n;
  const int imap[3][3] = {{S11, S12, S13}, {S12, S22, S23}, {S13, S23, S33}};
  const int kmap[3][3] = {{0, 3, 4}, {3, 1, 5}, {4, 5, 2}};

  par_for("tb_kf_A", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const auto &sz = size.d_view(m);
    Real g3d[NSPMETRIC] = {adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
                           adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
                           adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i)};
    const Real alpha = adm.alpha(m, k, j, i);
    const Real beta[3] = {adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
                          adm.beta_u(m, 2, k, j, i)};
    const Real detg = adm::SpatialDet(g3d[S11], g3d[S12], g3d[S13],
                                      g3d[S22], g3d[S23], g3d[S33]);
    const Real vol = sqrt(detg);
    Real g3u[NSPMETRIC];
    adm::SpatialInv(1.0/detg, g3d[S11], g3d[S12], g3d[S13], g3d[S22], g3d[S23], g3d[S33],
                    &g3u[S11], &g3u[S12], &g3u[S13], &g3u[S22], &g3u[S23], &g3u[S33]);
    Real idx[] = {1./sz.dx1, 1./sz.dx2, 1./sz.dx3};
    Real dbeta_du[3][3], dg_ddd[3][3][3];
    for (int a = 0; a < 3; a++) {
      for (int c = 0; c < 3; c++) {
        dbeta_du[a][c] = Dx<NGHOST>(a, idx, adm.beta_u, m, c, k, j, i);
        for (int e = 0; e < 3; ++e) {
          dg_ddd[a][c][e] = Dx<NGHOST>(a, idx, adm.g_dd, m, c, e, k, j, i);
        }
      }
    }
    Real E, S_d[3], S_uu[3][3], W, ein;
    TbFluid(eos_, mb, nhyd, nscal, prim, bcc, g3d, g3u, vol, m, k, j, i, E, S_d, S_uu, W,
            ein);

    // the code's gamma-dot = (residual chord + background rate) delta_ij, as the source kernel
    const Real p4 = adm.psi4(m, k, j, i);
    const Real xr[3] = {CellCenterX(i-is, nx1, sz.x1min, sz.x1max) - cx,
                        CellCenterX(j-js, nx2, sz.x2min, sz.x2max) - cy,
                        CellCenterX(k-ks, nx3, sz.x3min, sz.x3max) - cz};
    Real al0, b0c[3], ps0, K0[6], dpsi0;
    TbBackground(td, th, tih, nt, m_bh, xr[0], xr[1], xr[2], al0, b0c, ps0, K0, dpsi0);
    const Real r = sqrt(xr[0]*xr[0] + xr[1]*xr[1] + xr[2]*xr[2] + 1.0e-30);
    const Real vx = (gx[0]*xr[0] + gx[1]*xr[1] + gx[2]*xr[2])/r;
    Real gdr = 0.0, gdb = 0.0, gd0 = 0.0;
    if (dtg) {
      Real p4n = p4, p4p = pv(m, 4, k, j, i);
      if (bgm) {
        auto q4 = [&](const Real dp) {
          const Real ps = rsc ? psi0(m, 0, k, j, i) + rsf*dp : dp + psi0(m, 0, k, j, i);
          return ps*ps*ps*ps;
        };
        p4n = q4(dps(m, 0, k, j, i));
        p4p = q4(pv(m, 5, k, j, i));
      }
      gdr = (p4n - p4p)*idtm;
    }
    if (rate && bgm) {   // R-186 item 1: with or without the chord, as the source kernel
      Real pt, apt, b0t[3], dps0t, dal0t, bb0, db0;
      TrumpetTableBackground(td, th, tih, nt, m_bh, xr[0], xr[1], xr[2], pt, apt, b0t,
                             dps0t, dal0t, bb0, db0);
      const Real psd = vx*dps0t;
      const Real ps = sqrt(sqrt(p4));
      gdb = 4.0*ps*ps*ps*psd;
      gd0 = 4.0*ps0*ps0*ps0*vx*dpsi0;
    }

    // the background's Lie term from Dx<NGHOST> of the table at the stencil's cell centres
    auto bgat = [&](const int kk, const int jj, const int ii, Real b[3]) {
      Real a_, p_, K_[6], dp_;
      TbBackground(td, th, tih, nt, m_bh, CellCenterX(ii-is, nx1, sz.x1min, sz.x1max) - cx,
                   CellCenterX(jj-js, nx2, sz.x2min, sz.x2max) - cy,
                   CellCenterX(kk-ks, nx3, sz.x3min, sz.x3max) - cz, a_, b, p_, K_, dp_);
      return (p_*p_)*(p_*p_);
    };
    auto q4f = [&](const int, const int kk, const int jj, const int ii) {
      Real b[3];
      return bgat(kk, jj, ii, b);
    };
    auto b0f = [&](const int, const int c, const int kk, const int jj, const int ii) {
      Real b[3];
      bgat(kk, jj, ii, b);
      return b[c];
    };
    Real dq40[3], db0[3][3];
    for (int a = 0; a < 3; a++) {
      dq40[a] = Dx<NGHOST>(a, idx, q4f, m, k, j, i);
      for (int c = 0; c < 3; c++) db0[a][c] = Dx<NGHOST>(a, idx, b0f, m, c, k, j, i);
    }
    const Real q40 = (ps0*ps0)*(ps0*ps0);
    Real bp0[3];
    for (int a = 0; a < 3; ++a) bp0[a] = b0c[a] + (gauge ? gx[a] : 0.0);

    // V and V^0; A = -1/2 vol S^ij V_ij, the trace part -1/6 vol (g_ij S^ij)(g^ij V_ij)
    Real sv = 0.0, sv0 = 0.0, trv = 0.0, trv0 = 0.0, trs = 0.0;
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3; b++) {
        Real lie = 0.0;
        for (int c = 0; c < 3; c++) {
          lie += beta[c]*dg_ddd[c][a][b] + g3d[imap[c][b]]*dbeta_du[a][c]
                 + g3d[imap[a][c]]*dbeta_du[b][c];
        }
        Real lie0 = q40*(db0[a][b] + db0[b][a]);
        if (a == b) {
          for (int c = 0; c < 3; c++) lie0 += bp0[c]*dq40[c];
        }
        const Real v = (a == b ? gdr + gdb : 0.0) + 2.0*alpha*adm.vK_dd(m, a, b, k, j, i) - lie;
        const Real v0 = (a == b ? gd0 : 0.0) + 2.0*al0*K0[kmap[a][b]] - lie0;
        sv += S_uu[a][b]*v;
        sv0 += S_uu[a][b]*v0;
        trv += g3u[imap[a][b]]*v;
        trv0 += g3u[imap[a][b]]*v0;
        trs += g3d[imap[a][b]]*S_uu[a][b];
      }
    }
    const Real a0 = -0.5*vol*sv0, a0tr = -0.5*vol*trs*trv0/3.0;
    const Real at = -0.5*vol*sv, attr = -0.5*vol*trs*trv/3.0;
    tc(m, kTbA, k-ks, j-js, i-is) = a0tr;
    tc(m, kTbA + 1, k-ks, j-js, i-is) = a0 - a0tr;
    tc(m, kTbA + 2, k-ks, j-js, i-is) = attr - a0tr;
    tc(m, kTbA + 3, k-ks, j-js, i-is) = (at - attr) - (a0 - a0tr);
  });
}

//----------------------------------------------------------------------------------------
//! \fn void DynGRMHDPS::TauBudgetPost
//! \brief after AddKillingTerms: the per-cell budget into tb_c, summed per bin, one KTB line
template<class EOSPolicy, class ErrorPolicy>
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::TauBudgetPost(const DvceArray5D<Real> &prim,
    const Real dt, const DvceArray5D<Real> &rhs) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmb = pmy_pack->nmb_thispack, nkji = nx3*nx2*nx1, nji = nx2*nx1;
  auto &size = pmy_pack->pmb->mb_size;
  auto &adm = pmy_pack->padm->adm;
  auto &eos_ = eos.ps.GetEOS();
  const int nscal = pmy_pack->pmhd->nscalars;
  const int nhyd = pmy_pack->pmhd->nmhd;
  const Real mb = eos.ps.GetEOS().GetBaryonMass();
  auto &tc = tb_c;
  auto *pcfc = pmy_pack->pcfc;
  const bool pun = (pcfc != nullptr) && (pcfc->puncture_mass_ > 0.0);
  const Real cx = pun ? pcfc->bg_xbh_[0] : 0.0, cy = pun ? pcfc->bg_xbh_[1] : 0.0;
  const Real cz = pun ? pcfc->bg_xbh_[2] : 0.0;
  const Real m_bh = pun ? pcfc->puncture_mass_ : 0.0;
  const bool val = (energy_variable == EnergyVariable::valencia);
  const bool kf = (energy_variable == EnergyVariable::killing_full);
  const bool comov = kill_comov;
  const Real rhoc = tb_rhoc, rs2 = 0.64*tb_rstar*tb_rstar;
  const bool shells = tb_shells;
  const int nb = shells ? kTbShellBins : kTbStarBins;
  const Real se[kTbShellBins + 1] = {1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 13.0,
                                     16.0, 20.0};
  const Real xi[3] = {pmy_pack->padm->frame_vel_u[0], pmy_pack->padm->frame_vel_u[1],
                      pmy_pack->padm->frame_vel_u[2]};
  const Real idt = (dt != 0.0) ? 1.0/dt : 0.0;
  auto &td = ttab.d;
  const Real th = ttab.h, tih = ttab.ih;
  const int nt = ttab.n;
  auto &fv = tb_fv;
  auto &fx1_ = pmy_pack->pmhd->uflx.x1f;
  auto &fx2_ = pmy_pack->pmhd->uflx.x2f;
  auto &fx3_ = pmy_pack->pmhd->uflx.x3f;
  auto &tu = tb_u;
  auto &scr = kill_scr;

  // the star bins' centre: the rho dV centroid of rho >= 0.01 rho_c
  Real xc[4] = {0.0, 0.0, 0.0, 0.0};
  if (!shells) {
    Real s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    Kokkos::parallel_reduce("tb_centre", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*nkji),
    KOKKOS_LAMBDA(const int idx, Real &a0, Real &a1, Real &a2, Real &a3) {
      const int m = idx/nkji;
      const int k = (idx - m*nkji)/nji + ks;
      const int j = (idx - m*nkji - (k - ks)*nji)/nx1 + js;
      const int i = idx - m*nkji - (k - ks)*nji - (j - js)*nx1 + is;
      const auto &sz = size.d_view(m);
      const Real rho = prim(m, IDN, k, j, i);
      if (!(rho >= 0.01*rhoc)) return;
      const Real w = rho*sz.dx1*sz.dx2*sz.dx3;
      a0 += w;
      a1 += w*CellCenterX(i-is, nx1, sz.x1min, sz.x1max);
      a2 += w*CellCenterX(j-js, nx2, sz.x2min, sz.x2max);
      a3 += w*CellCenterX(k-ks, nx3, sz.x3min, sz.x3max);
    }, Kokkos::Sum<Real>(s0), Kokkos::Sum<Real>(s1), Kokkos::Sum<Real>(s2),
       Kokkos::Sum<Real>(s3));
    Real v[4] = {s0, s1, s2, s3};
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, v, 4, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
    if (v[0] > 0.0) {
      for (int a = 0; a < 3; ++a) xc[a] = v[1 + a]/v[0];
    }
  }
  const Real xs0 = xc[0], xs1 = xc[1], xs2 = xc[2];

  par_for("tb_post", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const auto &sz = size.d_view(m);
    const int kk = k-ks, jj = j-js, ii = i-is;
    const Real dv = sz.dx1*sz.dx2*sz.dx3;
    const Real x = CellCenterX(i-is, nx1, sz.x1min, sz.x1max);
    const Real y = CellCenterX(j-js, nx2, sz.x2min, sz.x2max);
    const Real z = CellCenterX(k-ks, nx3, sz.x3min, sz.x3max);
    const Real rho = prim(m, IDN, k, j, i);
    // the bin (-1: none)
    int bin = -1;
    if (shells) {
      const Real r = sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz));
      for (int q = 0; q < kTbShellBins; ++q) {
        if (r >= se[q] && r < se[q + 1]) bin = q;
      }
    } else {
      const Real f = rho/rhoc;
      const int band = (f >= 0.5) ? 0 : ((f >= 0.1) ? 1 : ((f >= 0.01) ? 2 :
                       ((f >= 1.0e-3) ? 3 : -1)));
      const Real d2 = (x-xs0)*(x-xs0) + (y-xs1)*(y-xs1) + (z-xs2)*(z-xs2);
      if (band >= 0) bin = 2*band + ((d2 > rs2) ? 1 : 0);
    }
    tc(m, kTbBin, kk, jj, ii) = static_cast<Real>(bin);
    for (int q = 0; q < kTbNr; ++q) tc(m, q, kk, jj, ii) = 0.0;
    if (bin < 0) return;

    Real g3d[NSPMETRIC] = {adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
                           adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
                           adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i)};
    const Real detg = adm::SpatialDet(g3d[S11], g3d[S12], g3d[S13],
                                      g3d[S22], g3d[S23], g3d[S33]);
    const Real vol = sqrt(detg);
    Real prim_pt[NPRIM] = {0.0};
    prim_pt[PRH] = rho/mb;
    prim_pt[PVX] = prim(m, IVX, k, j, i);
    prim_pt[PVY] = prim(m, IVY, k, j, i);
    prim_pt[PVZ] = prim(m, IVZ, k, j, i);
    for (int s = 0; s < nscal; s++) prim_pt[PYF + s] = prim(m, nhyd + s, k, j, i);
    prim_pt[PPR] = prim(m, IPR, k, j, i);
    prim_pt[PTM] = eos_.GetTemperatureFromP(prim_pt[PRH], prim_pt[PPR], &prim_pt[PYF]);
    const Real W = sqrt(1.0 + Primitive::SquareVector(&prim_pt[PVX], g3d));
    const Real ein = eos_.GetEnergy(prim_pt[PRH], prim_pt[PTM], &prim_pt[PYF]) - rho;
    tc(m, 0, kk, jj, ii) = dv;
    tc(m, 1, kk, jj, ii) = vol*W*rho*dv;
    tc(m, 2, kk, jj, ii) = vol*W*ein*dv;
    if (val) return;

    // the weights at the cell and at a face (killing_full: the average of the two cells, as
    // KillingFaceFluxes; t1: the trumpet at the face, KWeights' calls)
    Real alc, bc[3];
    if (kf) {
      alc = adm.alpha(m, k, j, i);
      for (int a = 0; a < 3; ++a) bc[a] = adm.beta_u(m, a, k, j, i);
    } else {
      Real p_, K_[6], dp_;
      TbBackground(td, th, tih, nt, m_bh, x - cx, y - cy, z - cz, alc, bc, p_, K_, dp_);
    }
    auto wface = [&](const int dir, const int k1, const int j1, const int i1, Real &al,
                     Real b[3]) {
      if (kf) {
        const int k0 = k1 - (dir == 2), j0 = j1 - (dir == 1), i0 = i1 - (dir == 0);
        al = (adm.alpha(m, k1, j1, i1) + adm.alpha(m, k0, j0, i0))*0.5;
        for (int a = 0; a < 3; ++a) {
          b[a] = (adm.beta_u(m, a, k1, j1, i1) + adm.beta_u(m, a, k0, j0, i0))*0.5;
        }
        return;
      }
      const Real fx = ((dir == 0) ? LeftEdgeX(i1-is, nx1, sz.x1min, sz.x1max) : x) - cx;
      const Real fy = ((dir == 1) ? LeftEdgeX(j1-js, nx2, sz.x2min, sz.x2max) : y) - cy;
      const Real fz = ((dir == 2) ? LeftEdgeX(k1-ks, nx3, sz.x3min, sz.x3max) : z) - cz;
      Real p_, K_[6], dp_;
      TbBackground(td, th, tih, nt, m_bh, fx, fy, fz, al, b, p_, K_, dp_);
    };
    const Real wd = kf ? 0.0 : 1.0;   // T's -w F_D (killing_t1: w = 1)
    int nfb = 0;
    Real divT = 0.0, divV = 0.0, sdisc = 0.0;
    for (int dir = 0; dir < 3; ++dir) {
      const auto &f = (dir == 0) ? fx1_ : ((dir == 1) ? fx2_ : fx3_);
      const Real id = 1.0/((dir == 0) ? sz.dx1 : ((dir == 1) ? sz.dx2 : sz.dx3));
      for (int s = 0; s < 2; ++s) {   // s = 0: the lower face, 1: the upper
        const int k1 = k + s*(dir == 2), j1 = j + s*(dir == 1), i1 = i + s*(dir == 0);
        const Real sg = s ? 1.0 : -1.0;
        Real al, b[3];
        wface(dir, k1, j1, i1, al, b);
        const Real fT = f(m, IEN, k1, j1, i1), fD = f(m, IDN, k1, j1, i1);
        const Real bS = b[0]*f(m, IM1, k1, j1, i1) + b[1]*f(m, IM2, k1, j1, i1)
                        + b[2]*f(m, IM3, k1, j1, i1);
        Real fV = fv(m, dir, k1-ks, j1-js, i1-is);
        const Real tr = al*(fV + fD) - bS - wd*fD;
        const Real sc = fabs(al*(fV + fD)) + fabs(bS) + fabs(wd*fD) + 1.0e-300;
        if (fabs(tr - fT) > 1.0e-10*sc && al >= 1.0e-3) {   // flux-corrected: invert
          fV = (fT + bS + wd*fD)/al - fD;
          nfb = 1;
        }
        divT += sg*fT*id;
        divV += sg*fV*id;
        if (comov) {   // the central frame advection against the T weights' face - cell
          // the face's two cells: (kl, jl, il) below it, (k1, j1, i1) above it
          const int kl = k1 - (dir == 2), jl = j1 - (dir == 1), il = i1 - (dir == 0);
          const Real fe = -xi[dir]*0.5*(tu(m, 0, kl, jl, il) + tu(m, 0, k1, j1, i1));
          Real p = (al - alc)*fe;
          for (int c = 0; c < 3; ++c) {
            const Real fs = -xi[dir]*0.5*(tu(m, 1 + c, kl, jl, il) + tu(m, 1 + c, k1, j1, i1));
            p -= (b[c] - bc[c])*fs;
          }
          sdisc += sg*p*id;
        }
      }
    }
    const Real d = (rhs(m, IEN, k, j, i) - tc(m, kTbSnap, kk, jj, ii))*idt - divT + divV;
    const Real w = kf ? adm.alpha(m, k, j, i) : alc;
    tc(m, 3, kk, jj, ii) = d*dv;
    tc(m, 4, kk, jj, ii) = fabs(d)*dv;
    tc(m, 5, kk, jj, ii) = w*fabs(d)*dv;
    tc(m, 6, kk, jj, ii) = w*dv;
    tc(m, 15, kk, jj, ii) = static_cast<Real>(nfb);
    if (kf) {
      Real at = 0.0;
      for (int q = 0; q < 4; ++q) {
        tc(m, 7 + q, kk, jj, ii) = tc(m, kTbA + q, kk, jj, ii)*dv;
        at += tc(m, kTbA + q, kk, jj, ii);
      }
      tc(m, 11, kk, jj, ii) = (d - at)*dv;
      return;
    }
    // t1: M_T = S_motion - S_motion,disc (0 in the BH frame and inside alpha0 < 1e-3, where
    // the source kernel keeps Valencia), into tau as M_T/alpha0
    Real mt = 0.0, sm = 0.0;
    if (comov && alc >= 1.0e-3) {
      sm = scr(m, 0, kk, jj, ii)/dv;
      mt = sm - sdisc;
      tc(m, 12, kk, jj, ii) = sm/alc*dv;
      tc(m, 13, kk, jj, ii) = sdisc/alc*dv;
      tc(m, 14, kk, jj, ii) = mt/alc*dv;
    }
    tc(m, 11, kk, jj, ii) = (d - ((comov && alc >= 1.0e-3) ? mt/alc : 0.0))*dv;
  });

  // per bin sums (components 0-2 for valencia, 0-15 otherwise)
  const int nr = val ? 3 : kTbNr;
  Real out[kTbShellBins*kTbNr] = {0.0};
  for (int b = 0; b < nb; ++b) {
    const Real bb = static_cast<Real>(b);
    for (int c = 0; c < nr; ++c) {
      Real s = 0.0;
      Kokkos::parallel_reduce("tb_bin", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*nkji),
      KOKKOS_LAMBDA(const int idx, Real &acc) {
        const int m = idx/nkji;
        const int k = (idx - m*nkji)/nji;
        const int j = (idx - m*nkji - k*nji)/nx1;
        const int i = idx - m*nkji - k*nji - j*nx1;
        if (tc(m, kTbBin, k, j, i) == bb) acc += tc(m, c, k, j, i);
      }, Kokkos::Sum<Real>(s));
      out[b*kTbNr + c] = s;
    }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, out, nb*kTbNr, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (tb_stage == 1) {
    auto *pmr = pmy_pack->pmesh->pmr;
    const int key = (pmr != nullptr) ? pmr->nmb_created + pmr->nmb_deleted : 0;
    tb_regrid = (tb_key >= 0) && (key != tb_key);
    tb_key = key;
  }
  if (global_variable::my_rank != 0) return;
  const char *mode = val ? "valencia" : (kf ? "killing_full" : (comov ? "t1_comoving" : "t1"));
  std::printf("KTB cycle=%d stage=%d time=%.12g dt=%.10e regrid=%d mode=%s bins=%s "
              "xc=%.8e,%.8e,%.8e nb=%d", pmy_pack->pmesh->ncycle, tb_stage,
              pmy_pack->pmesh->time, dt, static_cast<int>(tb_regrid), mode,
              shells ? "shells" : "star", xs0, xs1, xs2, nb);
  for (int b = 0; b < nb; ++b) {
    std::printf(" b%d=", b);
    for (int c = 0; c < kTbNr; ++c) std::printf(c ? ",%.10e" : "%.10e", out[b*kTbNr + c]);
  }
  std::printf("\n");
  std::fflush(stdout);
}

// instantiations (as AddKillingTerms)
#define INSTANTIATE_TAU_BUDGET(EOSPolicy, ErrorPolicy) \
template void DynGRMHDPS<EOSPolicy, ErrorPolicy>::TauBudgetPre<2>( \
    const DvceArray5D<Real> &, const DvceArray5D<Real> &, const DvceArray5D<Real> &); \
template void DynGRMHDPS<EOSPolicy, ErrorPolicy>::TauBudgetPre<3>( \
    const DvceArray5D<Real> &, const DvceArray5D<Real> &, const DvceArray5D<Real> &); \
template void DynGRMHDPS<EOSPolicy, ErrorPolicy>::TauBudgetPre<4>( \
    const DvceArray5D<Real> &, const DvceArray5D<Real> &, const DvceArray5D<Real> &); \
template void DynGRMHDPS<EOSPolicy, ErrorPolicy>::TauBudgetPost( \
    const DvceArray5D<Real> &, const Real, const DvceArray5D<Real> &);

INSTANTIATE_TAU_BUDGET(Primitive::IdealGas, Primitive::ResetFloor)
INSTANTIATE_TAU_BUDGET(Primitive::PiecewisePolytrope, Primitive::ResetFloor)
INSTANTIATE_TAU_BUDGET(Primitive::EOSCompOSE<Primitive::NormalLogs>, Primitive::ResetFloor)
INSTANTIATE_TAU_BUDGET(Primitive::EOSCompOSE<Primitive::NQTLogs>, Primitive::ResetFloor)
INSTANTIATE_TAU_BUDGET(Primitive::EOSHybrid<Primitive::NormalLogs>, Primitive::ResetFloor)
INSTANTIATE_TAU_BUDGET(Primitive::EOSHybrid<Primitive::NQTLogs>, Primitive::ResetFloor)

}  // namespace dyngr
