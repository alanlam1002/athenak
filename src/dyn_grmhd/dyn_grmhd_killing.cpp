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
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "driver/driver.hpp"
#include "eos/mass_closure.hpp"
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
  if (TauBudgetOn()) TauBudgetFaceCopy();   // R-184: Valencia's tau flux, before the transform
  if (energy_variable == EnergyVariable::killing_full) {
    // R-165 item 4: the weights at the face are the Riemann solver's face metric (cell
    // metric_terms: adm::Face?Metric, the average of the two cells, same order), so F_T is
    // the HLLE flux of T at the solver's own face metric and a ghost's face equals its owner's
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
    const int ks = indcs.ks, ke = indcs.ke;
    const int nmb = pmy_pack->nmb_thispack;
    auto &adm = pmy_pack->padm->adm;
    for (int dir = 0; dir < 3; ++dir) {
      auto &f = (dir == 0) ? pmy_pack->pmhd->uflx.x1f :
                ((dir == 1) ? pmy_pack->pmhd->uflx.x2f : pmy_pack->pmhd->uflx.x3f);
      const int si = (dir == 0), sj = (dir == 1), sk = (dir == 2);
      par_for("kill_full_faces", DevExeSpace(), 0, nmb-1, ks, ke + sk, js, je + sj, is, ie + si,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        const int km = k - sk, jm = j - sj, im = i - si;
        const Real al = (adm.alpha(m, k, j, i) + adm.alpha(m, km, jm, im))*0.5;
        Real b[3];
        for (int a = 0; a < 3; ++a) {
          b[a] = (adm.beta_u(m, a, k, j, i) + adm.beta_u(m, a, km, jm, im))*0.5;
        }
        f(m, IEN, k, j, i) = al*(f(m, IEN, k, j, i) + f(m, IDN, k, j, i))
                             - (b[0]*f(m, IM1, k, j, i) + b[1]*f(m, IM2, k, j, i)
                                + b[2]*f(m, IM3, k, j, i));
      });
    }
    if (kill_probe_cycles > 0) KillingFaceProbe();
    return;
  }
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
  // comoving (R-175): the weights exclude xidot; without a puncture they are (1, 0), the boost
  // form at rest
  const bool boost = kill_boost || kill_flat;
  const Real fv0 = kill_boost ? pmy_pack->padm->frame_vel_u[0] : 0.0;
  const Real fv1 = kill_boost ? pmy_pack->padm->frame_vel_u[1] : 0.0;
  const Real fv2 = kill_boost ? pmy_pack->padm->frame_vel_u[2] : 0.0;
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
//! \fn void DynGRMHD::KillingFullStage
//! \brief killing_full, at the stage's flux step (its metric solved, u0 still the stage's
//! input): the stage time (arm ii's d_t'), the validity of kf_prev, and KCLOSE's weights item.
//! KCLOSE evaluates T with each stage's own weights, so its RK identity mixes weights:
//!   stage 1: the gap M_1(u0) - M_prev(u0) since the last stage (outside the stages);
//!   stage s > 1: gam0 [M_s - M_{s-1}](u0) + gam1 [M_s(u1) - M_1(u1)] (inside its Y).
//! Both are booked as "weights" (mass_closure.cpp), the latter out of "source". Each is summed
//! per cell as a difference (R-177 item 1: two sums of T, then subtracted, left KCLOSE an
//! R ~ eps |T|), the u1 part telescoped stage by stage: [M_s - M_1](u1) = sum over q <= s of
//! [M_q - M_{q-1}](u1) (kf_m1u1, from kf_prev = the previous stage's weights).
void DynGRMHD::KillingFullStage(Driver *pdriver, int stage) {
  // the time of the metric this stage's fluxes use: solved from the previous stage's state,
  // t + tau_{s-1} dt with tau_s = gam0 tau_{s-1} + beta (as CFC::StageTime, cfc.cpp:2945)
  Real tau = 0.0;
  for (int q = 0; q < stage - 1; ++q) tau = pdriver->gam0[q]*tau + pdriver->beta[q];
  kf_tstage = pmy_pack->pmesh->time + tau*pmy_pack->pmesh->dt;
  kill_stage = stage;
  auto *pmr = pmy_pack->pmesh->pmr;
  const int key[3] = {(pmr != nullptr) ? pmr->nmb_created : 0,
                      (pmr != nullptr) ? pmr->nmb_deleted : 0, pmy_pack->nmb_thispack};
  if (kf_rst_loaded) {   // kf_prev came with the restart, in this pack's MeshBlock order
    for (int q = 0; q < 3; ++q) kf_key[q] = key[q];
    kf_rst_loaded = false;
  }
  if (key[0] != kf_key[0] || key[1] != kf_key[1] || key[2] != kf_key[2]) kf_have_prev = false;
  if (kf_dtg && stage == 1) {
    // xi-ddot in d_t' beta' (R-174 (c)): the analytic frame's frame_accel; the controller's
    // xidot is constant over a step and jumps at its end (CFC::FrameControlStep), so its own
    // update, (xidot_new - xidot_old)/dt, is spread over the step it starts (integral exact).
    // It enters the T source only: s_T's beta-dot part cancels the explicit S_k d_t' beta^k
    // term of the tau update identically (AddKillingFullTerms)
    auto *pcfc = pmy_pack->pcfc;
    const Real *fv = pmy_pack->padm->frame_vel_u;
    for (int a = 0; a < 3; ++a) kf_xdd[a] = 0.0;
    if (pcfc != nullptr && pcfc->FrameShiftEnabled()) {
      if (pcfc->FrameController()) {
        const Real dt = pmy_pack->pmesh->dt;
        for (int a = 0; a < 3; ++a) {
          kf_xdd[a] = (kf_have_xd && dt > 0.0) ? (fv[a] - kf_xdstep[a])/dt : 0.0;
        }
      } else {
        for (int a = 0; a < 3; ++a) kf_xdd[a] = pcfc->frame_accel_[a];
      }
    }
    for (int a = 0; a < 3; ++a) kf_xdstep[a] = fv[a];
    kf_have_xd = true;
  }
  if (pmclose == nullptr) return;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmb = pmy_pack->nmb_thispack, nkji = nx3*nx2*nx1, nji = nx2*nx1;
  auto &size = pmy_pack->pmb->mb_size;
  auto &adm = pmy_pack->padm->adm;
  auto &u0 = pmy_pack->pmhd->u0;
  auto &u1 = pmy_pack->pmhd->u1;
  auto &pv = kf_prev;
  const bool hp = kf_have_prev, s1 = (stage == 1);
  Real a = 0.0, c = 0.0;
  Kokkos::parallel_reduce("kill_full_w", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*nkji),
  KOKKOS_LAMBDA(const int idx, Real &sa, Real &sc) {
    const int m = idx/nkji;
    const int k = (idx - m*nkji)/nji + ks;
    const int j = (idx - m*nkji - (k - ks)*nji)/nx1 + js;
    const int i = idx - m*nkji - (k - ks)*nji - (j - js)*nx1 + is;
    const auto &sz = size.d_view(m);
    const Real dv = sz.dx1*sz.dx2*sz.dx3;
    auto tw = [&](const DvceArray5D<Real> &u, const Real al, const Real b0, const Real b1,
                  const Real b2) {
      return al*(u(m, IEN, k, j, i) + u(m, IDN, k, j, i))
             - (b0*u(m, IM1, k, j, i) + b1*u(m, IM2, k, j, i) + b2*u(m, IM3, k, j, i));
    };
    const Real al = adm.alpha(m, k, j, i);
    const Real b0 = adm.beta_u(m, 0, k, j, i), b1 = adm.beta_u(m, 1, k, j, i);
    const Real b2 = adm.beta_u(m, 2, k, j, i);
    if (!hp) return;
    const Real p0 = pv(m, 0, k, j, i), p1 = pv(m, 1, k, j, i), p2 = pv(m, 2, k, j, i);
    const Real p3 = pv(m, 3, k, j, i);
    sa += (tw(u0, al, b0, b1, b2) - tw(u0, p0, p1, p2, p3))*dv;
    if (!s1) sc += (tw(u1, al, b0, b1, b2) - tw(u1, p0, p1, p2, p3))*dv;
  }, Kokkos::Sum<Real>(a), Kokkos::Sum<Real>(c));
  Real v[2] = {a, c};
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, v, 2, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  Real g = 0.0;
  if (s1) {
    if (hp) g = v[0];
    kf_m1u1 = 0.0;
  } else if (hp) {
    kf_m1u1 += v[1];
    g = pdriver->gam0[stage-1]*v[0] + pdriver->gam1[stage-1]*kf_m1u1;
  }
  pmclose->KillWeights(stage, g);
}

//----------------------------------------------------------------------------------------
//! \fn void DynGRMHD::KillingT1Stage
//! \brief killing_weights = comoving (R-175), at the stage's flux step (u0 still its input):
//! the weights moved with the hole since the last stage (CFC::bg_xbh_), so KCLOSE's T, which
//! evaluates them at each stage's centre, gains the gap booked as "weights", exactly as
//! KillingFullStage's (stage 1: since the last stage; stage s > 1: gam0 [M_s - M_{s-1}](u0) +
//! gam1 [M_s(u1) - M_1(u1)]). The weights are analytic in x - x_BH, so the previous stage's
//! centre suffices. Without a puncture the weights are (1, 0): no gap.
void DynGRMHD::KillingT1Stage(Driver *pdriver, int stage) {
  kill_stage = stage;
  if (pmclose == nullptr || kill_flat) return;
  auto *pcfc = pmy_pack->pcfc;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmb = pmy_pack->nmb_thispack, nkji = nx3*nx2*nx1, nji = nx2*nx1;
  auto &size = pmy_pack->pmb->mb_size;
  auto &u0 = pmy_pack->pmhd->u0;
  auto &u1 = pmy_pack->pmhd->u1;
  const Real m_bh = pcfc->puncture_mass_;
  const Real c0 = pcfc->bg_xbh_[0], c1 = pcfc->bg_xbh_[1], c2 = pcfc->bg_xbh_[2];
  const Real p0 = kill_xprev[0], p1 = kill_xprev[1], p2 = kill_xprev[2];
  auto &td = ttab.d;
  const Real th = ttab.h, tih = ttab.ih;
  const int nt = ttab.n;
  const bool hp = kill_have_xprev, s1 = (stage == 1);
  const bool ta = (energy_variable == EnergyVariable::killing_ta);
  Real a = 0.0, b = 0.0, cc = 0.0;
  Kokkos::parallel_reduce("kill_t1_w", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*nkji),
  KOKKOS_LAMBDA(const int idx, Real &sa, Real &sb, Real &sc) {
    const int m = idx/nkji;
    const int k = (idx - m*nkji)/nji + ks;
    const int j = (idx - m*nkji - (k - ks)*nji)/nx1 + js;
    const int i = idx - m*nkji - (k - ks)*nji - (j - js)*nx1 + is;
    const auto &sz = size.d_view(m);
    const Real dv = sz.dx1*sz.dx2*sz.dx3;
    const Real x = CellCenterX(i-is, nx1, sz.x1min, sz.x1max);
    const Real y = CellCenterX(j-js, nx2, sz.x2min, sz.x2max);
    const Real z = CellCenterX(k-ks, nx3, sz.x3min, sz.x3max);
    // T = alpha0 (tau + D) - beta0^k S_k - w D (w = 1 killing_t1, alpha0 killing_ta), as
    // MassClosure::KSrc
    auto tw = [&](const DvceArray5D<Real> &u, const Real cx, const Real cy, const Real cz) {
      Real psi0, ap0, b0[3], dpsi0, dal0, bb, db;
      TrumpetTableBackground(td, th, tih, nt, m_bh, x - cx, y - cy, z - cz, psi0, ap0, b0,
                             dpsi0, dal0, bb, db);
      const Real al0 = ap0/psi0;
      const Real d = u(m, IDN, k, j, i);
      return al0*(u(m, IEN, k, j, i) + d)
             - (b0[0]*u(m, IM1, k, j, i) + b0[1]*u(m, IM2, k, j, i) + b0[2]*u(m, IM3, k, j, i))
             - (ta ? al0 : 1.0)*d;
    };
    sa += tw(u0, c0, c1, c2)*dv;
    if (hp) sb += tw(u0, p0, p1, p2)*dv;
    if (!s1) sc += tw(u1, c0, c1, c2)*dv;
  }, Kokkos::Sum<Real>(a), Kokkos::Sum<Real>(b), Kokkos::Sum<Real>(cc));
  Real v[3] = {a, b, cc};
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, v, 3, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  Real g = 0.0;
  if (s1) {
    if (hp) g = v[0] - v[1];
    kill_m1u1 = v[0];
  } else if (hp) {
    g = pdriver->gam0[stage-1]*(v[0] - v[1]) + pdriver->gam1[stage-1]*(v[2] - kill_m1u1);
  }
  pmclose->KillWeights(stage, g);
  kill_xprev[0] = c0; kill_xprev[1] = c1; kill_xprev[2] = c2;
  kill_have_xprev = true;
}

//----------------------------------------------------------------------------------------
//! \fn void DynGRMHD::ReportKillingStage
//! \brief after a killing source kernel (dt = the stage's beta dt): kill_scr's components
//! summed over the interior cells and the ranks. killing_full: KFSPLIT (every
//! killing_full_split_every cycles, every stage): the s_T integrals sum sqrt(g) s_T dV,
//! background (analytic rate, xi-ddot) and residual (the differenced chord) parts, over all
//! cells and over rho >= killing_full_split_rho. Comoving t1: KCLOSE's "motion" item, the
//! stage's applied beta_dt sum sqrt(g) s_W dV.
void DynGRMHD::ReportKillingStage(int stage, Real dt) {
  auto &scr = kill_scr;
  const int nc = scr.extent_int(1);
  const int nmb = pmy_pack->nmb_thispack;
  const int n3 = scr.extent_int(2), n2 = scr.extent_int(3), n1 = scr.extent_int(4);
  const int nkji = n3*n2*n1, nji = n2*n1;
  // components 0-11, 15-22 summed; 12-14 (the residual rates' |.|) maximized
  constexpr int nv = 23;
  Real v[nv] = {}, vx[3] = {0.0, 0.0, 0.0};
  for (int c = 0; c < nc && c < nv; ++c) {
    Real s = 0.0;
    if (c >= 12 && c <= 14) {
      Kokkos::parallel_reduce("kill_scr_max", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*nkji),
      KOKKOS_LAMBDA(const int idx, Real &acc) {
        const int m = idx/nkji;
        const int k = (idx - m*nkji)/nji;
        const int j = (idx - m*nkji - k*nji)/n1;
        const int i = idx - m*nkji - k*nji - j*n1;
        acc = fmax(acc, scr(m, c, k, j, i));
      }, Kokkos::Max<Real>(s));
      vx[c - 12] = s;
      continue;
    }
    Kokkos::parallel_reduce("kill_scr_sum", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*nkji),
    KOKKOS_LAMBDA(const int idx, Real &acc) {
      const int m = idx/nkji;
      const int k = (idx - m*nkji)/nji;
      const int j = (idx - m*nkji - k*nji)/n1;
      const int i = idx - m*nkji - k*nji - j*n1;
      acc += scr(m, c, k, j, i);
    }, Kokkos::Sum<Real>(s));
    v[c] = s;
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, v, nv, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  if (nc > 12) MPI_Allreduce(MPI_IN_PLACE, vx, 3, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  if (kill_comov) {
    if (pmclose != nullptr) pmclose->KillMotion(stage, dt*v[0]);
    return;
  }
  const int cyc = pmy_pack->pmesh->ncycle;
  if (kf_split_every <= 0 || cyc % kf_split_every != 0) return;
  if (global_variable::my_rank != 0) return;
  // R-177 item 2: the residual's d_t over the masked (dense) set, rms (coordinate-volume
  // weighted) and max, and the fraction of its cells whose residual chord is nonzero
  const Real iv = (v[11] > 0.0) ? 1.0/v[11] : 0.0;
  const Real *fv = pmy_pack->padm->frame_vel_u;
  std::printf("KFSPLIT cycle=%d stage=%d tstage=%.12g dtm=%.10e beta_dt=%.10e bg_all=%.10e "
              "res_all=%.10e bg_dense=%.10e res_dense=%.10e bgchord_all=%.10e "
              "bgchord_dense=%.10e l1_chord_minus_bg=%.10e l1_bg=%.10e have_prev=%d "
              "rms_dpsi=%.6e rms_dap=%.6e rms_dbres=%.6e max_dpsi=%.6e max_dap=%.6e "
              "max_dbres=%.6e nz_frac=%.6e n_dense=%.0f vdS_src=%.10e vdS_flux=%.10e "
              "bg_tau=%.10e val_work=%.10e Deps_dense=%.12e D_dense=%.12e "
              "xidot=%.12e,%.12e,%.12e xidd=%.6e,%.6e,%.6e\n", cyc,
              stage, kf_tstage, kf_tstage - kf_tprev, dt, v[0], v[1], v[2], v[3], v[4], v[5], v[6],
              v[7], static_cast<int>(kf_have_prev), sqrt(v[8]*iv), sqrt(v[9]*iv),
              sqrt(v[10]*iv), vx[0], vx[1], vx[2], (v[16] > 0.0) ? v[15]/v[16] : 0.0, v[16],
              v[17], v[18], v[19], v[20], v[21], v[22], fv[0], fv[1], fv[2], kf_xdd[0],
              kf_xdd[1], kf_xdd[2]);
  std::fflush(stdout);
}

//----------------------------------------------------------------------------------------
//! \fn DynGRMHD::SaveKillingScalars / LoadKillingScalars
//! \brief killing_full arm (ii)'s stage bookkeeping in the restart's parameter copy (R-174 (b);
//! kf_prev itself is the restart_state group dyngr.kfprev): absent = the feature-off state
void DynGRMHD::SaveKillingScalars(ParameterInput *pin) {
  if (!KfRst()) return;
  pin->SetReal("mhd", "kf_tprev", kf_tprev);
  pin->SetBoolean("mhd", "kf_have_prev", kf_have_prev);
  pin->SetBoolean("mhd", "kf_have_xd", kf_have_xd);
  for (int a = 0; a < 3; ++a) {
    pin->SetReal("mhd", "kf_xdstep" + std::to_string(a + 1), kf_xdstep[a]);
  }
}

void DynGRMHD::LoadKillingScalars(ParameterInput *pin) {
  // absent: a fresh start, or a restart written without arm (ii) (no dyngr.kfbres group: CFC's
  // u_beta comes back zero, LoadRestartState); kf_pre_rst marks the latter there
  kf_pre_rst = !pin->DoesParameterExist("mhd", "kf_tprev");
  if (kf_pre_rst) return;
  kf_tprev = pin->GetReal("mhd", "kf_tprev");
  kf_have_prev = pin->GetBoolean("mhd", "kf_have_prev");
  kf_have_xd = pin->GetBoolean("mhd", "kf_have_xd");
  for (int a = 0; a < 3; ++a) {
    kf_xdstep[a] = pin->GetReal("mhd", "kf_xdstep" + std::to_string(a + 1));
  }
}

//----------------------------------------------------------------------------------------
//! \fn void DynGRMHDPS::AddKillingFullTerms
//! \brief killing_full (R-165 item 4; Papadopoulos & Font's Killing form with the full
//! comoving metric): T = alpha (tau + D) - beta'^k S_k (densitized), its face flux F_T
//! (KillingFaceFluxes) and source s_T = -1/2 alpha sqrt(gamma) T^{mu nu} d_t' g_{mu nu}. The tau
//! update becomes
//!   dtau/dt = (dT/dt - (tau + D) d_t' alpha + S_k d_t' beta^k + beta^k dS_k/dt)/alpha - dD/dt,
//! dT/dt = -div F_T + s_T, dS_k/dt and dD/dt the Valencia updates, (alpha, beta) at the cell.
//! No weight gradient appears at the cell: the only metric derivatives are the Valencia
//! momentum source's (Sigma, in dS), so the scheme's T change is -div F_T + s_T to round-off.
//! Arm (i) (killing_full_dtg = false): every d_t' term = 0. Arm (ii): d_t' of alpha, beta^i,
//! psi^4 from the previous stage, 0 on the run's first stage and after a regrid. Then
//! kf_prev <- this stage's metric and CFC residual.
//! R-174: the chord is taken between this stage's metric and the previous stage's CFC residual
//! (delta psi, delta(alpha psi), beta_res) assembled on THIS stage's background and xidot, as
//! CFC::AssembleADM assembles it: only the residual is differenced (stage-time difference
//! kf_tstage - kf_tprev, SSP-RK3's +dt, -dt/2, +dt/2, a step's first stage against the previous
//! step's last). With a fixed background the previous metric is rebuilt bitwise, i.e. the R-165
//! arm (ii). The background's own rate is analytic: comoving (<cfc> puncture_moving) x_BH moves
//! with -xidot, d_t' g0 = xidot^k d_k g0 from the trumpet table (psi0, alpha0 psi0, beta0 = b x),
//! plus xi-ddot in beta' (KillingFullStage), all at linear order: gamma-dot = 4 psi^3 psi0-dot,
//! alpha-dot = (v0-dot - alpha psi0-dot)/psi (0 where alpha is floored). Linear in the rates, the
//! alpha-dot and beta-dot parts of s_T cancel the explicit -E alpha-dot + S_k beta-dot terms of
//! the tau update identically, so only psi-dot reaches tau (KILLING_ENERGY_NOTE; argued).
template<class EOSPolicy, class ErrorPolicy>
template<int NGHOST>
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddKillingFullTerms(const DvceArray5D<Real> &prim,
    const DvceArray5D<Real> &bcc, const Real dt, DvceArray5D<Real> &rhs) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pmy_pack->nmb_thispack;
  auto &size = pmy_pack->pmb->mb_size;
  auto &adm = pmy_pack->padm->adm;
  auto &eos_ = eos.ps.GetEOS();
  const int nhyd = pmy_pack->pmhd->nmhd;
  const int nscal = pmy_pack->pmhd->nscalars;
  const Real mb = eos.ps.GetEOS().GetBaryonMass();
  auto &pv = kf_prev;
  const Real dtm = kf_tstage - kf_tprev;
  const bool dtg = kf_dtg && kf_have_prev && dtm != 0.0;
  const Real idtm = dtg ? 1.0/dtm : 0.0;
  // R-174: this stage's background and xidot (CFC::AssembleADM's), the analytic rate's inputs
  auto *pcfc = pmy_pack->pcfc;
  auto &psi0 = pcfc->u_psi0;
  auto &ap0 = pcfc->u_alpha0_psi0;
  auto &b0 = pcfc->beta0_u;
  const Real afl = pcfc->alpha_floor_;
  // a constant <cfc> residual_scale f != 1: CFC::AssembleADM's scaled assembly (background + f
  // residual, cfc_reconstruct.cpp AssembleLapseShiftKScaled); f = 0, background only
  const bool rsc = pcfc->rs_on_ && pcfc->rs_f_ != 1.0;
  const Real rsf = pcfc->rs_f_;
  auto &dps = pcfc->delta_psi;
  auto &dap = pcfc->delta_alpha_psi;
  auto &bres = pcfc->beta_u;
  const Real gx[3] = {pmy_pack->padm->frame_vel_u[0], pmy_pack->padm->frame_vel_u[1],
                      pmy_pack->padm->frame_vel_u[2]};
  const bool gauge = (gx[0] != 0.0) || (gx[1] != 0.0) || (gx[2] != 0.0);
  const bool bgm = kf_bgmove;
  const Real xdd[3] = {kf_xdd[0], kf_xdd[1], kf_xdd[2]};
  // R-186 item 1: the background's analytic rate and xi-ddot need no previous stage, so they are
  // kept where the residual chord is not (the first stage after a regrid or a run's start; C-148:
  // dropping them there put -1/2 sqrt(g) S^ij gamma0-dot_ij's absence into B1K's tau)
  const bool rate = kf_dtg && (bgm || xdd[0] != 0.0 || xdd[1] != 0.0 || xdd[2] != 0.0);
  const Real cx = pcfc->bg_xbh_[0], cy = pcfc->bg_xbh_[1], cz = pcfc->bg_xbh_[2];
  const Real m_bh = pcfc->puncture_mass_;
  auto &td = ttab.d;
  const Real th = ttab.h, tih = ttab.ih;
  const int nt = ttab.n;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const bool split = (kf_split_every > 0);
  const bool fz = kf_fz;   // R-190 (KF_z)
  const bool fcfc = kf_fcfc;   // R-191 (KF_CFC)
  const Real srho = kf_split_rho, srmin2 = kf_split_rmin*kf_split_rmin;
  auto &scr = kill_scr;
  auto &fx1_ = pmy_pack->pmhd->uflx.x1f;
  auto &fx2_ = pmy_pack->pmhd->uflx.x2f;
  auto &fx3_ = pmy_pack->pmhd->uflx.x3f;
  const int imap[3][3] = {
    {S11, S12, S13},
    {S12, S22, S23},
    {S13, S23, S33}
  };

  par_for("kill_full_src", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const auto &sz = size.d_view(m);
    // the metric and its derivatives (as AddCoordTermsEOS)
    Real g3d[NSPMETRIC] = {adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
                           adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
                           adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i)};
    const Real alpha = adm.alpha(m, k, j, i);
    const Real beta[3] = {adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
                          adm.beta_u(m, 2, k, j, i)};
    Real detg = adm::SpatialDet(g3d[S11], g3d[S12], g3d[S13],
                                g3d[S22], g3d[S23], g3d[S33]);
    Real vol = sqrt(detg);
    Real g3u[NSPMETRIC] = {0.};
    adm::SpatialInv(1.0/detg, g3d[S11], g3d[S12], g3d[S13], g3d[S22], g3d[S23], g3d[S33],
                    &g3u[S11], &g3u[S12], &g3u[S13], &g3u[S22], &g3u[S23], &g3u[S33]);
    Real idx[] = {1./sz.dx1, 1./sz.dx2, 1./sz.dx3};
    Real dalpha_d[3], dbeta_du[3][3], dg_ddd[3][3][3];
    for (int a = 0; a < 3; a++) {
      dalpha_d[a] = Dx<NGHOST>(a, idx, adm.alpha, m, k, j, i);
      for (int c = 0; c < 3; c++) {
        dbeta_du[a][c] = Dx<NGHOST>(a, idx, adm.beta_u, m, c, k, j, i);
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

    // tau's Valencia source (applied by AddCoordTermsEOS, or AddCoordTermsFM) and the momentum's
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

    // arm (ii): d_t' alpha, beta^i, psi^4 (gamma_ij = psi^4 delta_ij) and
    // s_T = -1/2 alpha vol T^{mu nu} d_t' g_{mu nu}, with T^00 = E/alpha^2,
    // T^0i = (S^i - E beta^i/alpha)/alpha, T^ij = E beta^i beta^j/alpha^2
    // - (S^i beta^j + S^j beta^i)/alpha + S^ij; g_00 = -alpha^2 + psi^4 beta.beta,
    // g_0i = psi^4 beta^i, g_ij = psi^4 delta_ij
    Real sT = 0.0, dal = 0.0, dbe[3] = {0.0, 0.0, 0.0}, sTb = 0.0, sTc = 0.0;
    // KFSPLIT (R-177 item 2): the residual's chord rates, d_t dpsi, d_t d(alpha psi), |d_t
    // beta_res|_coord (moving background only), and whether the chord is nonzero
    Real rrp = 0.0, rra = 0.0, rrb = 0.0;
    bool rnz = false;
    // KFSPLIT (R-177 addendum): the background rate's net tau deposit, (sTb - E dalb + S.bd)/alpha
    Real bgt = 0.0;
    const Real p4 = adm.psi4(m, k, j, i);
    Real S_u[3];
    for (int a = 0; a < 3; a++) {
      S_u[a] = 0.0;
      for (int b = 0; b < 3; b++) S_u[a] += g3u[imap[a][b]]*S_d[b];
    }
    const Real ia = 1.0/alpha;
    if (dtg) {
      // the chord's ends (an, bn, p4n: this stage; alp, bp, p4p: the previous one). A fixed
      // background: the stored metric (kf_prev 0-4) against this stage's, the R-165 arm (ii) (the
      // full metric's change is the residual's). A moving background (R-174 (c)): both ends
      // assembled from the CFC residual on THIS stage's background and xidot (as CFC::AssembleADM:
      // background + (f) residual, the alpha floor), so the chord is the residual's change only.
      // Both ends use the same arithmetic: the assembly kernels' own psi^4 differs from it by an
      // ulp under fast-fp codegen (c154: 172992 cells of 1 ulp on the CPU build), which would put
      // epsilon/dt noise into the chord.
      Real an = alpha, bn[3] = {beta[0], beta[1], beta[2]}, p4n = p4;
      Real alp = pv(m, 0, k, j, i), p4p = pv(m, 4, k, j, i);
      Real bp[3] = {pv(m, 1, k, j, i), pv(m, 2, k, j, i), pv(m, 3, k, j, i)};
      if (bgm) {
        auto asm_ = [&](const Real dp, const Real dv, const Real *dr, Real &al, Real *bt,
                        Real &q4) {
          const Real ps = rsc ? psi0(m, 0, k, j, i) + rsf*dp : dp + psi0(m, 0, k, j, i);
          q4 = ps*ps*ps*ps;
          const Real av = (rsc ? ap0(m, 0, k, j, i) + rsf*dv : dv + ap0(m, 0, k, j, i))/ps;
          al = (av < afl) ? afl : av;
          for (int a = 0; a < 3; ++a) {
            bt[a] = rsc ? b0(m, a, k, j, i) + rsf*dr[a] : dr[a] + b0(m, a, k, j, i);
            if (gauge) bt[a] += gx[a];
          }
        };
        const Real rn[3] = {bres(m, 0, k, j, i), bres(m, 1, k, j, i), bres(m, 2, k, j, i)};
        const Real rp[3] = {pv(m, 7, k, j, i), pv(m, 8, k, j, i), pv(m, 9, k, j, i)};
        asm_(dps(m, 0, k, j, i), dap(m, 0, k, j, i), rn, an, bn, p4n);
        asm_(pv(m, 5, k, j, i), pv(m, 6, k, j, i), rp, alp, bp, p4p);
        if (split) {
          rrp = (dps(m, 0, k, j, i) - pv(m, 5, k, j, i))*idtm;
          rra = (dap(m, 0, k, j, i) - pv(m, 6, k, j, i))*idtm;
          Real b2 = 0.0;
          for (int a = 0; a < 3; ++a) b2 += (rn[a] - rp[a])*(rn[a] - rp[a]);
          rrb = sqrt(b2)*fabs(idtm);
          rnz = (dps(m, 0, k, j, i) != pv(m, 5, k, j, i)) ||
                (dap(m, 0, k, j, i) != pv(m, 6, k, j, i)) || (b2 != 0.0);
        }
      }
      dal = (an - alp)*idtm;
      for (int a = 0; a < 3; ++a) dbe[a] = (bn[a] - bp[a])*idtm;
      const Real dp4 = (p4n - p4p)*idtm;
      const Real bbn = bn[0]*bn[0] + bn[1]*bn[1] + bn[2]*bn[2];
      const Real bbp = bp[0]*bp[0] + bp[1]*bp[1] + bp[2]*bp[2];
      const Real dg00 = -(an*an - alp*alp)*idtm + (p4n*bbn - p4p*bbp)*idtm;
      Real dg0[3];
      for (int a = 0; a < 3; ++a) dg0[a] = (p4n*bn[a] - p4p*bp[a])*idtm;
      Real q = 0.5*E*ia*ia*dg00;
      for (int a = 0; a < 3; ++a) {
        q += (S_u[a] - E*beta[a]*ia)*ia*dg0[a];
        // 1/2 T^ij d_t g_ij = 1/2 dp4 T^aa (conformally flat)
        q += 0.5*dp4*(E*beta[a]*beta[a]*ia*ia - 2.0*S_u[a]*beta[a]*ia + S_uu[a][a]);
      }
      sT = -alpha*vol*q;
      if (split) {
        // KFSPLIT's check of the analytic rate (diagnostic): the R-165 chord of the FULL metric
        // (kf_prev 0-4, which differences the background) minus the residual chord above is the
        // background's chord effect; it must match sTb to O(dt), and be 0 with a fixed background
        const Real alq = pv(m, 0, k, j, i), p4q = pv(m, 4, k, j, i);
        const Real bq[3] = {pv(m, 1, k, j, i), pv(m, 2, k, j, i), pv(m, 3, k, j, i)};
        const Real bbq = bq[0]*bq[0] + bq[1]*bq[1] + bq[2]*bq[2];
        const Real bb = beta[0]*beta[0] + beta[1]*beta[1] + beta[2]*beta[2];
        const Real dg00q = -(alpha*alpha - alq*alq)*idtm + (p4*bb - p4q*bbq)*idtm;
        const Real dp4q = (p4 - p4q)*idtm;
        Real qq = 0.5*E*ia*ia*dg00q;
        for (int a = 0; a < 3; ++a) {
          qq += (S_u[a] - E*beta[a]*ia)*ia*(p4*beta[a] - p4q*bq[a])*idtm;
          qq += 0.5*dp4q*(E*beta[a]*beta[a]*ia*ia - 2.0*S_u[a]*beta[a]*ia + S_uu[a][a]);
        }
        sTc = -alpha*vol*qq - sT;
      }
    }
    if (rate) {
      // the background's analytic rate and xi-ddot, linear in the rates
      Real psd = 0.0, vd = 0.0, bd[3] = {xdd[0], xdd[1], xdd[2]};
      if (bgm) {
        const Real xr[3] = {CellCenterX(i-is, nx1, sz.x1min, sz.x1max) - cx,
                            CellCenterX(j-js, nx2, sz.x2min, sz.x2max) - cy,
                            CellCenterX(k-ks, nx3, sz.x3min, sz.x3max) - cz};
        Real pt, apt, b0t[3], dpsi0, dal0, bb0, db0;
        TrumpetTableBackground(td, th, tih, nt, m_bh, xr[0], xr[1], xr[2], pt, apt, b0t,
                               dpsi0, dal0, bb0, db0);
        const Real r = sqrt(xr[0]*xr[0] + xr[1]*xr[1] + xr[2]*xr[2] + 1.0e-30);
        // xidot^k d_k f(r) = (xidot . x/r) f'(r); d_k beta0^i = b delta_ki + b' x^k x^i/r
        const Real vx = (gx[0]*xr[0] + gx[1]*xr[1] + gx[2]*xr[2])/r;
        psd = vx*dpsi0;
        vd = vx*(dal0*pt + (apt/pt)*dpsi0);
        for (int a = 0; a < 3; ++a) bd[a] += bb0*gx[a] + db0*vx*xr[a];
      }
      const Real ps = sqrt(sqrt(p4));
      const Real dalb = (alpha > afl) ? (vd - alpha*psd)/ps : 0.0;
      const Real dp4b = 4.0*ps*ps*ps*psd;
      Real bdb = 0.0;
      for (int a = 0; a < 3; ++a) bdb += beta[a]*bd[a];
      const Real bbc = beta[0]*beta[0] + beta[1]*beta[1] + beta[2]*beta[2];
      const Real dg00b = -2.0*alpha*dalb + dp4b*bbc + 2.0*p4*bdb;
      Real qb = 0.5*E*ia*ia*dg00b;
      for (int a = 0; a < 3; ++a) {
        qb += (S_u[a] - E*beta[a]*ia)*ia*(dp4b*beta[a] + p4*bd[a]);
        qb += 0.5*dp4b*(E*beta[a]*beta[a]*ia*ia - 2.0*S_u[a]*beta[a]*ia + S_uu[a][a]);
      }
      sTb = -alpha*vol*qb;
      Real bS = 0.0;
      for (int a = 0; a < 3; ++a) bS += S_d[a]*bd[a];
      bgt = (sTb - vol*E*dalb + vol*bS)/alpha;
      dal += dalb;
      for (int a = 0; a < 3; ++a) dbe[a] += bd[a];
    }
    auto divf = [&](int n) {
      return (fx1_(m, n, k, j, i+1) - fx1_(m, n, k, j, i))/sz.dx1
           + (fx2_(m, n, k, j+1, i) - fx2_(m, n, k, j, i))/sz.dx2
           + (fx3_(m, n, k+1, j, i) - fx3_(m, n, k, j, i))/sz.dx3;
    };
    if (split) {   // KFSPLIT (R-174 (e)): vol s_T dV, background and residual parts
      const Real dv = sz.dx1*sz.dx2*sz.dx3;
      // the masked set: rho >= killing_full_split_rho and |x - x_BH| >= killing_full_split_rmin
      const Real xb = CellCenterX(i-is, nx1, sz.x1min, sz.x1max) - cx;
      const Real yb = CellCenterX(j-js, nx2, sz.x2min, sz.x2max) - cy;
      const Real zb = CellCenterX(k-ks, nx3, sz.x3min, sz.x3max) - cz;
      const bool dn = (prim(m, IDN, k, j, i) >= srho) && (xb*xb + yb*yb + zb*zb >= srmin2);
      scr(m, 6, k-ks, j-js, i-is) = dn ? fabs(sTc - sTb)*dv : 0.0;   // the rate check, L1
      scr(m, 7, k-ks, j-js, i-is) = dn ? fabs(sTb)*dv : 0.0;
      scr(m, 0, k-ks, j-js, i-is) = sTb*dv;
      scr(m, 1, k-ks, j-js, i-is) = sT*dv;
      scr(m, 2, k-ks, j-js, i-is) = dn ? sTb*dv : 0.0;
      scr(m, 3, k-ks, j-js, i-is) = dn ? sT*dv : 0.0;
      scr(m, 4, k-ks, j-js, i-is) = sTc*dv;
      scr(m, 5, k-ks, j-js, i-is) = dn ? sTc*dv : 0.0;
      // the residual's rates over the masked set: squares x dV and dV (rms), |.| (max), counts
      scr(m, 8, k-ks, j-js, i-is) = dn ? rrp*rrp*dv : 0.0;
      scr(m, 9, k-ks, j-js, i-is) = dn ? rra*rra*dv : 0.0;
      scr(m, 10, k-ks, j-js, i-is) = dn ? rrb*rrb*dv : 0.0;
      scr(m, 11, k-ks, j-js, i-is) = dn ? dv : 0.0;
      scr(m, 12, k-ks, j-js, i-is) = dn ? fabs(rrp) : 0.0;
      scr(m, 13, k-ks, j-js, i-is) = dn ? fabs(rra) : 0.0;
      scr(m, 14, k-ks, j-js, i-is) = dn ? rrb : 0.0;
      scr(m, 15, k-ks, j-js, i-is) = (dn && rnz) ? 1.0 : 0.0;
      scr(m, 16, k-ks, j-js, i-is) = dn ? 1.0 : 0.0;
      // R-177 addendum (the hole's work against its force), on the masked set: v^k dS_k/dt
      // (Eulerian v^k = u^k/W) from the Valencia momentum source and from the fluxes; the
      // background rate's net tau deposit; Valencia's own work, -sqrt(g) S^k d_k alpha; D eps; D
      Real vS = 0.0, vF = 0.0, vw = 0.0;
      for (int a = 0; a < 3; ++a) {
        const Real va = prim_pt[PVX + a]/W;
        vS += va*vol*src_S[a];
        vF -= va*divf(IM1 + a);
        for (int b = 0; b < 3; ++b) vw -= g3u[imap[a][b]]*S_d[a]*dalpha_d[b];
      }
      const Real ein = eos_.GetEnergy(prim_pt[PRH], prim_pt[PTM], &prim_pt[PYF])
                       - prim(m, IDN, k, j, i);
      scr(m, 17, k-ks, j-js, i-is) = dn ? vS*dv : 0.0;
      scr(m, 18, k-ks, j-js, i-is) = dn ? vF*dv : 0.0;
      scr(m, 19, k-ks, j-js, i-is) = dn ? bgt*dv : 0.0;
      scr(m, 20, k-ks, j-js, i-is) = dn ? vol*vw*dv : 0.0;
      scr(m, 21, k-ks, j-js, i-is) = dn ? vol*W*ein*dv : 0.0;
      scr(m, 22, k-ks, j-js, i-is) = dn ? vol*W*prim(m, IDN, k, j, i)*dv : 0.0;
    }
    sT += sTb;
    if (fcfc) {   // R-191 (KF_CFC): -sqrt(g) (alpha/3) S D_k beta'^k, S = gamma_ij S^ij,
      // D_k beta'^k = d_k beta'^k + beta'^k 1/2 gamma^ij d_k gamma_ij (= d_k(psi^6 beta'^k)/psi^6)
      Real trS = 0.0, divb = 0.0;
      for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) trS += g3d[imap[a][b]]*S_uu[a][b];
      }
      for (int c = 0; c < 3; c++) {
        Real dln = 0.0;   // 1/2 gamma^ij d_c gamma_ij
        for (int a = 0; a < 3; a++) {
          for (int b = 0; b < 3; b++) dln += 0.5*g3u[imap[a][b]]*dg_ddd[c][a][b];
        }
        divb += dbeta_du[c][c] + beta[c]*dln;
      }
      sT += -vol*(alpha/3.0)*trS*divb;
    } else if (fz) {   // R-190 (KF_z): sqrt(g) 1/2 alpha S^ij (2 alpha K_ij - (Lie_beta' gamma)_ij)
      Real sz = 0.0;
      for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
          Real lie = 0.0;   // beta^c d_c g_ab + g_cb d_a beta^c + g_ac d_b beta^c
          for (int c = 0; c < 3; c++) {
            lie += beta[c]*dg_ddd[c][a][b] + g3d[imap[c][b]]*dbeta_du[a][c]
                   + g3d[imap[a][c]]*dbeta_du[b][c];
          }
          sz += S_uu[a][b]*(2.0*alpha*adm.vK_dd(m, a, b, k, j, i) - lie);
        }
      }
      sT += vol*0.5*alpha*sz;
    }

    const Real dT = -divf(IEN) + sT;
    const Real dD = -divf(IDN);
    Real bdS = 0.0, Sdb = 0.0;
    for (int a = 0; a < 3; a++) {
      bdS += beta[a]*(-divf(IM1 + a) + vol*src_S[a]);
      Sdb += vol*S_d[a]*dbe[a];
    }
    const Real dtau = (dT - vol*E*dal + Sdb + bdS)/alpha - dD;
    const Real dtau_applied = -divf(IEN) + vol*src_tau;
    rhs(m, IEN, k, j, i) += dt*(dtau - dtau_applied);
  });

  // this stage's metric -> kf_prev (every cell, ghosts included)
  par_for("kill_full_prev", DevExeSpace(), 0, nmb-1, 0, pv.extent_int(2)-1,
          0, pv.extent_int(3)-1, 0, pv.extent_int(4)-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    pv(m, 0, k, j, i) = adm.alpha(m, k, j, i);
    for (int a = 0; a < 3; ++a) pv(m, 1 + a, k, j, i) = adm.beta_u(m, a, k, j, i);
    pv(m, 4, k, j, i) = adm.psi4(m, k, j, i);
  });
  if (kf_dtg) {   // R-174 (b): this stage's CFC residual (interior: the source reads only it)
    par_for("kill_full_prev_res", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      pv(m, 5, k, j, i) = dps(m, 0, k, j, i);
      pv(m, 6, k, j, i) = dap(m, 0, k, j, i);
      for (int a = 0; a < 3; ++a) pv(m, 7 + a, k, j, i) = bres(m, a, k, j, i);
    });
  }
  if (split) ReportKillingStage(kill_stage, dt);
  kf_tprev = kf_tstage;
  // after a restart without arm (ii)'s groups, the first stage's u_beta is the zero fill, not
  // a solve: that stage's store is no chord end (the next stage has none, as after a regrid)
  kf_have_prev = !kf_bres_stale;
  kf_bres_stale = false;
  auto *pmr = pmy_pack->pmesh->pmr;
  kf_key[0] = (pmr != nullptr) ? pmr->nmb_created : 0;
  kf_key[1] = (pmr != nullptr) ? pmr->nmb_deleted : 0;
  kf_key[2] = nmb;
}

//----------------------------------------------------------------------------------------
//! \fn void DynGRMHDPS::AddKillingTerms
//! \brief see the file header: rhs(IEN) += dt [dtau/dt (from T) - (-div F_T + sqrt(gamma)
//! tau's Valencia source)], the latter being what MHD::RKUpdate and AddCoordTermsEOS applied
template<class EOSPolicy, class ErrorPolicy>
template<int NGHOST>
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddKillingTerms(const DvceArray5D<Real> &prim,
    const DvceArray5D<Real> &bcc, const Real dt, DvceArray5D<Real> &rhs) {
  if (energy_variable == EnergyVariable::killing_full) {
    AddKillingFullTerms<NGHOST>(prim, bcc, dt, rhs);
    return;
  }
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
  // apply: killing_t1/ta; otherwise (valencia + killing_source_diag) the diagnostic only
  const bool apply = (energy_variable != EnergyVariable::valencia);
  const bool ta = apply ? (energy_variable == EnergyVariable::killing_ta) : kill_diag_ta;
  // comoving (R-175): the weights exclude xidot ((1, 0) without a puncture); xidot itself
  // (fr*) leaves the residual shift, delta beta = beta' - xidot - beta_BH, and drives the
  // frozen weights' rate
  const bool boost = kill_boost || kill_flat;
  const Real fv0 = kill_boost ? pmy_pack->padm->frame_vel_u[0] : 0.0;
  const Real fv1 = kill_boost ? pmy_pack->padm->frame_vel_u[1] : 0.0;
  const Real fv2 = kill_boost ? pmy_pack->padm->frame_vel_u[2] : 0.0;
  const bool comov = kill_comov;
  const Real fr0 = comov ? pmy_pack->padm->frame_vel_u[0] : 0.0;
  const Real fr1 = comov ? pmy_pack->padm->frame_vel_u[1] : 0.0;
  const Real fr2 = comov ? pmy_pack->padm->frame_vel_u[2] : 0.0;
  auto &scr = kill_scr;
  const bool dg = kill_diag;
  auto &kd = kdiag;
  const Real kd_lr0 = log(kKdR0), kd_ilr = static_cast<Real>(kKdBins - 2)/log(kKdR1/kKdR0);
  const bool exc = pmy_pack->pcoord->coord_data.bh_excise;
  auto &xfloor = pmy_pack->pcoord->excision_floor;
  auto &td = ttab.d;
  const Real th = ttab.h, tih = ttab.ih;
  const int nt = ttab.n;
  auto &res = kill_res;
  // metric_terms != cell (R-171): the applied Valencia sources are AddCoordTermsFM's, so the
  // metric derivatives below are its (dyn_grmhd_metric.cpp:440-485), not Dx of adm
  const bool fmode = (metric_terms != MetricTerms::cell);
  const bool fwb = (metric_terms == MetricTerms::wellbalanced);
  const bool mfd = kill_mom_fd;   // R-178: AddCoordTermsFM's momentum derivatives
  auto &fm1 = fm_x1;
  auto &fm2 = fm_x2;
  auto &fm3 = fm_x3;
  auto &fres = fm_res;
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
    const Real fr[3] = {fr0, fr1, fr2};
    for (int a = 0; a < 3; ++a) {
      res(m, 1 + a, k, j, i) = adm.beta_u(m, a, k, j, i) - b0[a];
      if (comov) res(m, 1 + a, k, j, i) -= fr[a];
    }
    res(m, 4, k, j, i) = adm.psi4(m, k, j, i) - p2*p2;
  });

  par_for("kill_src", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // excision-floor cells (reset by the C2P) get the rebuild too: RKUpdate gave their tau
    // slot the T face flux, so without it their T change is not -div F_T, and KCLOSE booked
    // that as a "source" cancelled by the reset's "excision" item (C-136: the non-converging
    // pair of R-155 3(a)); the diagnostic below counts evolved cells only
    const bool xcell = exc && xfloor(m, k, j, i);
    const auto &sz = size.d_view(m);
    const Real xr[3] = {CellCenterX(i-is, nx1, sz.x1min, sz.x1max) - cx,
                        CellCenterX(j-js, nx2, sz.x2min, sz.x2max) - cy,
                        CellCenterX(k-ks, nx3, sz.x3min, sz.x3max) - cz};
    Real al0, b0[3], psi0, K0[6], dal0, bb, db;
    KWeights(boost, fv0, fv1, fv2, td, th, tih, nt, m_bh, xr[0], xr[1], xr[2], al0, b0, psi0,
             K0, dal0, bb, db);
    const Real r = sqrt(xr[0]*xr[0] + xr[1]*xr[1] + xr[2]*xr[2] + 1.0e-30);
    // killing_source_diag: the r_iso bin and the cell-time weight
    int kb = 0;
    if (r >= kKdR1) {
      kb = kKdBins - 1;
    } else if (r >= kKdR0) {
      kb = 1 + static_cast<int>((log(r) - kd_lr0)*kd_ilr);
      kb = (kb < 1) ? 1 : ((kb > kKdBins - 2) ? kKdBins - 2 : kb);
    }
    const Real kwgt = dt*sz.dx1*sz.dx2*sz.dx3;
    if (comov) scr(m, 0, k-ks, j-js, i-is) = 0.0;
    if (!(al0 >= 1.0e-3)) {
      if (dg && !xcell) Kokkos::atomic_add(&kd(kb, 6), kwgt);
      return;
    }
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
    Real dam[3], dbm[3][3], dgm[3][3][3];   // the momentum rows' (AddCoordTermsFM's)
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

    if (fmode) {   // AddCoordTermsFM's derivatives (killing_t1/ta need a 3D mesh)
      for (int a = 0; a < 3; a++) {
        for (int c = 0; c < 3; c++) {
          for (int e = 0; e < 3; ++e) dg_ddd[a][c][e] = 0.0;
        }
      }
      if (fwb) {
        for (int a = 0; a < 3; a++) {
          const auto &fm = (a == 0) ? fm1 : ((a == 1) ? fm2 : fm3);
          const int ip = i + (a == 0), jp = j + (a == 1), kp = k + (a == 2);
          dalpha_d[a] = (fm(m, kFmAlpha, kp, jp, ip) - fm(m, kFmAlpha, k, j, i))*idx[a];
          for (int b = 0; b < 3; b++) {
            dbeta_du[a][b] = (fm(m, kFmBeta + b, kp, jp, ip) - fm(m, kFmBeta + b, k, j, i))*
                             idx[a];
          }
          const Real pp = fm(m, kFmPsi, kp, jp, ip), pm = fm(m, kFmPsi, k, j, i);
          const Real dg = ((pp*pp)*(pp*pp) - (pm*pm)*(pm*pm))*idx[a];
          for (int b = 0; b < 3; b++) dg_ddd[a][b][b] = dg;
        }
      } else {
        Real psi0f, ap0f, b0f[3], dpsi0f, dal0f, bbf, dbf;
        TrumpetTableBackground(td, th, tih, nt, m_bh, xr[0], xr[1], xr[2], psi0f, ap0f, b0f,
                               dpsi0f, dal0f, bbf, dbf);
        const Real rf = sqrt(xr[0]*xr[0] + xr[1]*xr[1] + xr[2]*xr[2] + 1.0e-30);
        const Real xh[3] = {xr[0]/rf, xr[1]/rf, xr[2]/rf};
        const Real al0f = ap0f/psi0f;
        const Real pf = sqrt(sqrt(adm.psi4(m,k,j,i)));
        const Real p3 = pf*pf*pf;
        for (int a = 0; a < 3; a++) {
          const Real dpsi = dpsi0f*xh[a] + Dx<NGHOST>(a, idx, fres, m, 0, k, j, i);
          const Real dap = (dal0f*psi0f + al0f*dpsi0f)*xh[a] +
                           Dx<NGHOST>(a, idx, fres, m, 1, k, j, i);
          dalpha_d[a] = (dap - alpha*dpsi)/pf;
          for (int b = 0; b < 3; b++) {
            dbeta_du[a][b] = ((a == b) ? bbf : 0.0) + dbf*xh[a]*xr[b] +
                             Dx<NGHOST>(a, idx, fres, m, 2 + b, k, j, i);
            dg_ddd[a][b][b] = 4.0*p3*dpsi;
          }
        }
        if (mfd) {   // as AddCoordTermsFM (dyn_grmhd_metric.cpp), the same arithmetic
          for (int a = 0; a < 3; a++) {
            for (int b = 0; b < 3; b++) {
              for (int c = 0; c < 3; c++) dgm[a][b][c] = 0.0;
            }
          }
          const int ic[3] = {i - is, j - js, k - ks}, nc[3] = {nx1, nx2, nx3};
          const Real lo[3] = {sz.x1min, sz.x2min, sz.x3min};
          const Real hi[3] = {sz.x1max, sz.x2max, sz.x3max};
          const Real c0[3] = {cx, cy, cz};
          const Real p03 = psi0f*psi0f*psi0f;
          for (int a = 0; a < 3; a++) {
            Real xm[3] = {xr[0], xr[1], xr[2]}, xp[3] = {xr[0], xr[1], xr[2]};
            xm[a] = LeftEdgeX(ic[a], nc[a], lo[a], hi[a]) - c0[a];
            xp[a] = LeftEdgeX(ic[a] + 1, nc[a], lo[a], hi[a]) - c0[a];
            Real fa, fb[3], fg;
            TrumpetFaceDiff(td, th, tih, nt, m_bh, xm, xp, idx[a], fa, fb, fg);
            dam[a] = dalpha_d[a] - dal0f*xh[a] + fa;
            for (int b = 0; b < 3; b++) {
              dbm[a][b] = fb[b] + Dx<NGHOST>(a, idx, fres, m, 2 + b, k, j, i);
              dgm[a][b][b] = dg_ddd[a][b][b] - 4.0*p03*dpsi0f*xh[a] + fg;
            }
          }
        }
      }
    }
    if (!mfd) {
      for (int a = 0; a < 3; a++) {
        dam[a] = dalpha_d[a];
        for (int b = 0; b < 3; b++) {
          dbm[a][b] = dbeta_du[a][b];
          for (int c = 0; c < 3; c++) dgm[a][b][c] = dg_ddd[a][b][c];
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

    // tau's Valencia source (applied by AddCoordTermsEOS, or AddCoordTermsFM) and the momentum's
    Real src_tau = 0.0, src_S[3];
    for (int a = 0; a < 3; a++) {
      src_S[a] = -E*dam[a];
      for (int b = 0; b < 3; b++) {
        src_tau += alpha*adm.vK_dd(m, a, b, k, j, i)*S_uu[a][b]
                   - g3u[imap[a][b]]*S_d[a]*dalpha_d[b];
        src_S[a] += S_d[b]*dbm[a][b];
        for (int c = 0; c < 3; c++) {
          src_S[a] += 0.5*alpha*S_uu[b][c]*dgm[a][b][c];
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
    Real sTa = 0.0;
    if (ta) {   // -D V^j d_j alpha0, V^j = alpha v^j - beta^j
      const Real D = prim(m, IDN, k, j, i)*W;
      for (int a = 0; a < 3; a++) {
        sTa -= D*(alpha*prim_pt[PVX + a]/W - adm.beta_u(m, a, k, j, i))*dal0_d[a];
      }
    }
    Real sW = 0.0;
    if (comov) {   // -(d_t' G) q = -xidot^j (E d_j alpha0 - S_k d_j beta_BH^k) (NOTE Sec. 5)
      const Real fr[3] = {fr0, fr1, fr2};
      // killing_ta (R-188 item 5): D's weight -alpha0 moves too, -(d_t' G_D) D = +xidot^j D
      // d_j alpha0, i.e. E -> E - D
      const Real Ew = ta ? E - prim(m, IDN, k, j, i)*W : E;
      for (int a = 0; a < 3; a++) {
        Real sdb = 0.0;
        for (int c = 0; c < 3; c++) sdb += S_d[c]*dB0[a][c];
        sW -= fr[a]*(Ew*dal0_d[a] - sdb);
      }
      scr(m, 0, k-ks, j-js, i-is) = vol*sW*sz.dx1*sz.dx2*sz.dx3;
    }
    const Real sT = sK + sA + sE + sB + sTa + sW;

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
    if (apply) rhs(m, IEN, k, j, i) += dt*(dtau - dtau_applied);
    if (!dg || xcell) return;

    // killing_source_diag: this cell's effective T source, dT + div F_T. Applied: sqrt(g) s_T.
    // Valencia: from the update the RK step applied, with F_T formed at the cell's six faces
    // from the Valencia fluxes (the weights at the face, as KillingFaceFluxes).
    Real eff = vol*sT;
    if (!apply) {
      auto fT = [&](const DvceArray5D<Real> &fl, const int kk, const int jj, const int ii,
                    const Real fx, const Real fy, const Real fz) {
        Real a0, bf[3], p0, Kf[6], d0, b_, db_;
        KWeights(false, 0.0, 0.0, 0.0, td, th, tih, nt, m_bh, fx, fy, fz, a0, bf, p0, Kf,
                 d0, b_, db_);
        const Real fD = fl(m, IDN, kk, jj, ii);
        return a0*(fl(m, IEN, kk, jj, ii) + fD) - (bf[0]*fl(m, IM1, kk, jj, ii)
               + bf[1]*fl(m, IM2, kk, jj, ii) + bf[2]*fl(m, IM3, kk, jj, ii))
               - (ta ? a0 : 1.0)*fD;
      };
      const Real xl = LeftEdgeX(i-is, nx1, sz.x1min, sz.x1max) - cx;
      const Real xh = LeftEdgeX(i+1-is, nx1, sz.x1min, sz.x1max) - cx;
      const Real yl = LeftEdgeX(j-js, nx2, sz.x2min, sz.x2max) - cy;
      const Real yh = LeftEdgeX(j+1-js, nx2, sz.x2min, sz.x2max) - cy;
      const Real zl = LeftEdgeX(k-ks, nx3, sz.x3min, sz.x3max) - cz;
      const Real zh = LeftEdgeX(k+1-ks, nx3, sz.x3min, sz.x3max) - cz;
      const Real divT =
          (fT(fx1_, k, j, i+1, xh, xr[1], xr[2]) - fT(fx1_, k, j, i, xl, xr[1], xr[2]))/sz.dx1
        + (fT(fx2_, k, j+1, i, xr[0], yh, xr[2]) - fT(fx2_, k, j, i, xr[0], yl, xr[2]))/sz.dx2
        + (fT(fx3_, k+1, j, i, xr[0], xr[1], zh) - fT(fx3_, k, j, i, xr[0], xr[1], zl))/sz.dx3;
      eff = al0*(dtau_applied + dD) - b0dS - w*dD + divT;
    }
    const Real sv[6] = {eff, vol*sK, vol*sA, vol*sE, vol*sB, vol*sTa};
    for (int q = 0; q < 6; ++q) Kokkos::atomic_add(&kd(kb, q), kwgt*sv[q]);
    Kokkos::atomic_add(&kd(kb, 7), kwgt*fabs(dal));
    Kokkos::atomic_add(&kd(kb, 13), kwgt);
    Kokkos::atomic_max(&kd(kb, 8), fabs(dal));
    Kokkos::atomic_max(&kd(kb, 9), fabs(dal/al0));
    Kokkos::atomic_max(&kd(kb, 10), sqrt(dbe[0]*dbe[0] + dbe[1]*dbe[1] + dbe[2]*dbe[2]));
    Kokkos::atomic_max(&kd(kb, 11), fabs(Hr));
    Kokkos::atomic_max(&kd(kb, 12), sqrt(dda[0]*dda[0] + dda[1]*dda[1] + dda[2]*dda[2]));
  });
  if (comov) ReportKillingStage(kill_stage, dt);   // KCLOSE item "motion"
  if (kill_diag) {
    const int cyc = pmy_pack->pmesh->ncycle;
    if (cyc != kill_diag_last && cyc % kill_diag_every == 0) {
      kill_diag_last = cyc;
      ReportKillingDiag();
    }
  }
}

//----------------------------------------------------------------------------------------
//! n void DynGRMHD::ReportKillingDiag
//! rief <mhd> killing_source_diag: one KDIAG line per non-empty r_iso bin (rank 0), the sums
//! cumulative since the run's start (this process: a restart starts them again)
void DynGRMHD::ReportKillingDiag() {
  auto h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), kdiag);
  std::vector<Real> s(kKdBins*kKdComp);
  for (int b = 0; b < kKdBins; ++b) {
    for (int q = 0; q < kKdComp; ++q) s[b*kKdComp + q] = h(b, q);
  }
#if MPI_PARALLEL_ENABLED
  std::vector<Real> mx(s);
  MPI_Allreduce(MPI_IN_PLACE, s.data(), kKdBins*kKdComp, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, mx.data(), kKdBins*kKdComp, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
  for (int b = 0; b < kKdBins; ++b) {
    for (int q = 8; q < 13; ++q) s[b*kKdComp + q] = mx[b*kKdComp + q];
  }
#endif
  if (global_variable::my_rank != 0) return;
  const char *nm[kKdComp] = {"eff", "sK", "sA", "sE", "sB", "sTa", "skip_dtdV", "dal_dtdV",
                             "max_dal", "max_dal_rel", "max_dbeta", "max_dpsi4", "max_ddal",
                             "dtdV"};
  const Real q = pow(kKdR1/kKdR0, 1.0/static_cast<Real>(kKdBins - 2));
  for (int b = 0; b < kKdBins; ++b) {
    if (s[b*kKdComp + 13] == 0.0 && s[b*kKdComp + 6] == 0.0) continue;
    const Real rlo = (b == 0) ? 0.0 : kKdR0*pow(q, b - 1);
    const Real rhi = (b == kKdBins - 1) ? 1.0e30 : kKdR0*pow(q, b);
    std::printf("KDIAG cycle=%d time=%.10g bin=%d rlo=%.6g rhi=%.6g", pmy_pack->pmesh->ncycle,
                pmy_pack->pmesh->time, b, rlo, rhi);
    for (int c = 0; c < kKdComp; ++c) std::printf(" %s=%.10e", nm[c], s[b*kKdComp + c]);
    std::printf("\n");
  }
  std::fflush(stdout);
}

// instantiations (as AddCoordTermsEOS)
#define INSTANTIATE_KILLING_TERMS(EOSPolicy, ErrorPolicy) \
template void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddKillingTerms<2>( \
    const DvceArray5D<Real> &, const DvceArray5D<Real> &, const Real, DvceArray5D<Real> &); \
template void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddKillingTerms<3>( \
    const DvceArray5D<Real> &, const DvceArray5D<Real> &, const Real, DvceArray5D<Real> &); \
template void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddKillingTerms<4>( \
    const DvceArray5D<Real> &, const DvceArray5D<Real> &, const Real, DvceArray5D<Real> &); \
template void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddKillingFullTerms<2>( \
    const DvceArray5D<Real> &, const DvceArray5D<Real> &, const Real, DvceArray5D<Real> &); \
template void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddKillingFullTerms<3>( \
    const DvceArray5D<Real> &, const DvceArray5D<Real> &, const Real, DvceArray5D<Real> &); \
template void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddKillingFullTerms<4>( \
    const DvceArray5D<Real> &, const DvceArray5D<Real> &, const Real, DvceArray5D<Real> &);

INSTANTIATE_KILLING_TERMS(Primitive::IdealGas, Primitive::ResetFloor)
INSTANTIATE_KILLING_TERMS(Primitive::PiecewisePolytrope, Primitive::ResetFloor)
INSTANTIATE_KILLING_TERMS(Primitive::EOSCompOSE<Primitive::NormalLogs>, Primitive::ResetFloor)
INSTANTIATE_KILLING_TERMS(Primitive::EOSCompOSE<Primitive::NQTLogs>, Primitive::ResetFloor)
INSTANTIATE_KILLING_TERMS(Primitive::EOSHybrid<Primitive::NormalLogs>, Primitive::ResetFloor)
INSTANTIATE_KILLING_TERMS(Primitive::EOSHybrid<Primitive::NQTLogs>, Primitive::ResetFloor)

}  // namespace dyngr
