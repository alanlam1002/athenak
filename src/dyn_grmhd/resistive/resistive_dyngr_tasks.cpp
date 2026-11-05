//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file resistive_dyngr_tasks.cpp
//! \brief EOS-dependent part of resistive GRMHD on the dyngr path (07_scoping_v2 §3.3):
//!        task queue, LF fluxes of the total (fluid + EM) conserved variables and of the
//!        charge, the implicit Ohmic step (cell Newton + face closed form, D2/D5) and
//!        the final resistive c2p (D4). Explicitly instantiated for all dyngr policies.

#include <cmath>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "coordinates/adm.hpp"
#include "mhd/mhd.hpp"
#include "tasklist/numerical_relativity.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "dyn_grmhd/resistive/resistive_dyngr.hpp"
#include "dyn_grmhd/resistive/resistive_mesh_util.hpp"
#include "dyn_grmhd/resistive/resistive_em.hpp"
#include "dyn_grmhd/resistive/resistive_edge.hpp"
#include "dyn_grmhd/resistive/resistive_implicit.hpp"

#include "eos/primitive_solver_hyd.hpp"
#include "eos/primitive-solver/idealgas.hpp"
#include "eos/primitive-solver/eos_compose.hpp"
#include "eos/primitive-solver/eos_hybrid.hpp"
#include "eos/primitive-solver/piecewise_polytrope.hpp"
#include "eos/primitive-solver/reset_floor.hpp"

namespace dyngr {

namespace {

// Number of reconstructed hydro primitives: rho, u^1..u^3, p, then scalars.
constexpr int kNW = 5 + MAX_SPECIES;

// One side of a face: total conserved state U (fluid + EM, undensitised) and the
// physical flux F along direction d (undensitised, including lapse and shift).
template <class PSolver>
KOKKOS_INLINE_FUNCTION
void SideState(const PSolver &ps, const Real g_in[NSPMETRIC], const Real sg,
               const Real alpha, const Real beta[3], const int d, const Real mb,
               const Real w[kNW], const int nscal, const Real em[6],
               Real U[NCONS], Real F[NCONS]) {
  Real prim[NPRIM];
  for (int n = 0; n < NPRIM; ++n) prim[n] = 0.0;
  prim[PRH] = w[0]/mb;
  prim[PVX] = w[1];
  prim[PVY] = w[2];
  prim[PVZ] = w[3];
  prim[PPR] = w[4];
  for (int n = 0; n < nscal; ++n) prim[PYF + n] = w[5 + n];
  prim[PTM] = ps.GetEOS().GetTemperatureFromP(prim[PRH], prim[PPR], &prim[PYF]);
  ps.GetEOS().ApplyPrimitiveFloor(prim[PRH], &prim[PVX], prim[PPR], prim[PTM],
                                  &prim[PYF]);
  Real g[NSPMETRIC];
  for (int n = 0; n < NSPMETRIC; ++n) g[n] = g_in[n];
  Real bz[NMAG] = {0.0, 0.0, 0.0};
  for (int n = 0; n < NCONS; ++n) U[n] = 0.0;
  ps.PrimToCon(prim, U, bz, g);

  const Real u[3] = {prim[PVX], prim[PVY], prim[PVZ]};
  const Real W = sqrt(1.0 + resistive::Dot(g, u, u));
  const Real vn = u[d]/W;
  const Real vt = alpha*vn - beta[d];
  F[CDN] = U[CDN]*vt;
  F[CSX] = U[CSX]*vt;
  F[CSY] = U[CSY]*vt;
  F[CSZ] = U[CSZ]*vt;
  F[CTA] = U[CTA]*vt;
  for (int n = 0; n < nscal; ++n) F[CYD + n] = U[CYD + n]*vt;
  F[CSX + d] += alpha*prim[PPR];
  F[CTA] += alpha*prim[PPR]*vn;

  const Real E[3] = {em[0], em[1], em[2]};
  const Real B[3] = {em[3], em[4], em[5]};
  Real Ed[3], Bd[3], s_em[3], e_em, poy[3];
  resistive::Lower(g, E, Ed);
  resistive::Lower(g, B, Bd);
  resistive::EMParts(g, sg, E, B, s_em, e_em);
  resistive::CrossUp(sg, Ed, Bd, poy);
  for (int a = 0; a < 3; ++a) {
    F[CSX + a] += alpha*(-E[d]*Ed[a] - B[d]*Bd[a] + ((a == d) ? e_em : 0.0))
                - beta[d]*s_em[a];
    U[CSX + a] += s_em[a];
  }
  F[CTA] += alpha*poy[d] - beta[d]*e_em;
  U[CTA] += e_em;
}

KOKKOS_INLINE_FUNCTION
Real CellCharge(const DvceFaceFld4D<Real> &ef, const int m, const int k, const int j,
                const int i, const Real idx1, const Real idx2, const Real idx3,
                const bool multi_d, const bool three_d) {
  Real q = (ef.x1f(m, k, j, i+1) - ef.x1f(m, k, j, i))*idx1;
  if (multi_d) q += (ef.x2f(m, k, j+1, i) - ef.x2f(m, k, j, i))*idx2;
  if (three_d) q += (ef.x3f(m, k+1, j, i) - ef.x3f(m, k, j, i))*idx3;
  return q;
}

KOKKOS_INLINE_FUNCTION
Real FaceAvg(const DvceFaceFld4D<Real> &f, const int d, const int m, const int k,
             const int j, const int i) {
  if (d == 0) return 0.5*(f.x1f(m, k, j, i) + f.x1f(m, k, j, i+1));
  if (d == 1) return 0.5*(f.x2f(m, k, j, i) + f.x2f(m, k, j+1, i));
  return 0.5*(f.x3f(m, k, j, i) + f.x3f(m, k+1, j, i));
}

}  // namespace

//----------------------------------------------------------------------------------------
//! Resistive task list (replaces the ideal list; branch in QueueDynGRMHDTasks).

template<class EOSPolicy, class ErrorPolicy>
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::QueueResistiveTasks() {
  using namespace mhd;     // NOLINT(build/namespaces)
  using namespace numrel;  // NOLINT(build/namespaces)
  using resistive::ResistiveDynGR;
  using Self = DynGRMHDPS<EOSPolicy, ErrorPolicy>;
  MHD *pmhd = pmy_pack->pmhd;
  NumericalRelativity *pnr = pmy_pack->pnr;
  ResistiveDynGR *pr = presist;

  pnr->QueueTask(&MHD::InitRecv, pmhd, MHD_Recv, "MHD_Recv", Task_Start);
  pnr->QueueTask(&ResistiveDynGR::InitRecv, pr, MHD_RRecv, "MHD_RRecv", Task_Start);

  // stage-1 pre-implicit solves of the ImEx scheme (no-ops for stage > 1)
  pnr->QueueTask(&MHD::CopyCons, pmhd, MHD_CopyU, "MHD_CopyU", Task_Run);
  pnr->QueueTask(&ResistiveDynGR::CopyE, pr, MHD_RCopyE, "MHD_RCopyE", Task_Run);
  pnr->QueueTask(&Self::ResistivePre1, this, MHD_RPre1, "MHD_RPre1", Task_Run,
                 {MHD_CopyU, MHD_RCopyE});
  pnr->QueueTask(&ResistiveDynGR::SendEP, pr, MHD_RSendEP, "MHD_RSendEP", Task_Run,
                 {MHD_RPre1});
  pnr->QueueTask(&ResistiveDynGR::RecvEP, pr, MHD_RRecvEP, "MHD_RRecvEP", Task_Run,
                 {MHD_RSendEP});
  pnr->QueueTask(&ResistiveDynGR::ApplyBCsEP, pr, MHD_RBCsEP, "MHD_RBCsEP", Task_Run,
                 {MHD_RRecvEP});
  pnr->QueueTask(&Self::ResistivePre2, this, MHD_RPre2, "MHD_RPre2", Task_Run,
                 {MHD_RBCsEP});
  pnr->QueueTask(&ResistiveDynGR::SendEP2, pr, MHD_RSendEP2, "MHD_RSendEP2", Task_Run,
                 {MHD_RPre2});
  pnr->QueueTask(&ResistiveDynGR::RecvEP2, pr, MHD_RRecvEP2, "MHD_RRecvEP2", Task_Run,
                 {MHD_RSendEP2});
  pnr->QueueTask(&ResistiveDynGR::ApplyBCsEP, pr, MHD_RBCsEP2, "MHD_RBCsEP2", Task_Run,
                 {MHD_RRecvEP2});
  pnr->QueueTask(&Self::ResistiveC2PPre, this, MHD_RC2PPre, "MHD_RC2PPre", Task_Run,
                 {MHD_RBCsEP2});

  // explicit stage
  pnr->QueueTask(&Self::ResistiveFluxes, this, MHD_Flux, "MHD_Flux", Task_Run,
                 {MHD_RC2PPre});
  pnr->QueueTask(&ResistiveDynGR::EdgeSolve, pr, MHD_REdge, "MHD_REdge", Task_Run,
                 {MHD_RC2PPre});
  pnr->QueueTask(&MHD::SendFlux, pmhd, MHD_SendFlux, "MHD_SendFlux", Task_Run,
                 {MHD_Flux});
  pnr->QueueTask(&MHD::RecvFlux, pmhd, MHD_RecvFlux, "MHD_RecvFlux", Task_Run,
                 {MHD_SendFlux});
  pnr->QueueTask(&MHD::SendE, pmhd, MHD_SendE, "MHD_SendE", Task_Run, {MHD_REdge});
  pnr->QueueTask(&MHD::RecvE, pmhd, MHD_RecvE, "MHD_RecvE", Task_Run, {MHD_SendE});
  pnr->QueueTask(&ResistiveDynGR::SendH, pr, MHD_RSendH, "MHD_RSendH", Task_Run,
                 {MHD_REdge});
  pnr->QueueTask(&ResistiveDynGR::RecvH, pr, MHD_RRecvH, "MHD_RRecvH", Task_Run,
                 {MHD_RSendH});
  pnr->QueueTask(&MHD::RKUpdate, pmhd, MHD_ExplRK, "MHD_ExplRK", Task_Run,
                 {MHD_RecvFlux});
  pnr->QueueTask(&MHD::CT, pmhd, MHD_CT, "MHD_CT", Task_Run, {MHD_RecvE, MHD_RecvFlux});
  pnr->QueueTask(&ResistiveDynGR::CTE, pr, MHD_RCTE, "MHD_RCTE", Task_Run,
                 {MHD_RRecvH, MHD_RecvFlux});

  // boundary exchange of U, B, E~
  pnr->QueueTask(&MHD::RestrictU, pmhd, MHD_RestU, "MHD_RestU", Task_Run, {MHD_ExplRK});
  pnr->QueueTask(&MHD::SendU, pmhd, MHD_SendU, "MHD_SendU", Task_Run, {MHD_RestU});
  pnr->QueueTask(&MHD::RecvU, pmhd, MHD_RecvU, "MHD_RecvU", Task_Run, {MHD_SendU});
  pnr->QueueTask(&MHD::RestrictB, pmhd, MHD_RestB, "MHD_RestB", Task_Run, {MHD_CT});
  pnr->QueueTask(&MHD::SendB, pmhd, MHD_SendB, "MHD_SendB", Task_Run, {MHD_RestB});
  pnr->QueueTask(&MHD::RecvB, pmhd, MHD_RecvB, "MHD_RecvB", Task_Run, {MHD_SendB});
  pnr->QueueTask(&ResistiveDynGR::SendE, pr, MHD_RSendE, "MHD_RSendE", Task_Run,
                 {MHD_RCTE});
  pnr->QueueTask(&ResistiveDynGR::RecvE, pr, MHD_RRecvE, "MHD_RRecvE", Task_Run,
                 {MHD_RSendE});
  pnr->QueueTask(&MHD::Prolongate, pmhd, MHD_Prolong, "MHD_Prolong", Task_Run,
                 {MHD_RecvB, MHD_RecvU, MHD_RRecvE});
  pnr->QueueTask(&MHD::ApplyPhysicalBCs, pmhd, MHD_BCS, "MHD_BCS", Task_Run,
                 {MHD_Prolong});
  pnr->QueueTask(&ResistiveDynGR::ApplyBCsE, pr, MHD_RBCsE, "MHD_RBCsE", Task_Run,
                 {MHD_Prolong});

  // implicit Ohmic step, E~ exchange, final c2p
  pnr->QueueTask(&Self::ResistiveImplicit, this, MHD_RImpl, "MHD_RImpl", Task_Run,
                 {MHD_BCS, MHD_RBCsE});
  pnr->QueueTask(&ResistiveDynGR::SendEI, pr, MHD_RSendEI, "MHD_RSendEI", Task_Run,
                 {MHD_RImpl});
  pnr->QueueTask(&ResistiveDynGR::RecvEI, pr, MHD_RRecvEI, "MHD_RRecvEI", Task_Run,
                 {MHD_RSendEI});
  pnr->QueueTask(&ResistiveDynGR::ApplyBCsE, pr, MHD_RBCsEI, "MHD_RBCsEI", Task_Run,
                 {MHD_RRecvEI});
  pnr->QueueTask(&Self::ConToPrim, this, MHD_C2P, "MHD_C2P", Task_Run, {MHD_RBCsEI});
  pnr->QueueTask(&MHD::NewTimeStep, pmhd, MHD_Newdt, "MHD_Newdt", Task_Run, {MHD_C2P});

  pnr->QueueTask(&MHD::ClearSend, pmhd, MHD_ClearS, "MHD_ClearS", Task_End);
  pnr->QueueTask(&MHD::ClearRecv, pmhd, MHD_ClearR, "MHD_ClearR", Task_End);
  pnr->QueueTask(&ResistiveDynGR::ClearSend, pr, MHD_RClearS, "MHD_RClearS", Task_End);
  pnr->QueueTask(&ResistiveDynGR::ClearRecv, pr, MHD_RClearR, "MHD_RClearR", Task_End);
}

//----------------------------------------------------------------------------------------
template<class EOSPolicy, class ErrorPolicy>
TaskStatus DynGRMHDPS<EOSPolicy, ErrorPolicy>::ResistivePre1(Driver *d, int stage) {
  if (stage == 1) ResistiveImplicitSolve(d, -1);
  return TaskStatus::complete;
}

template<class EOSPolicy, class ErrorPolicy>
TaskStatus DynGRMHDPS<EOSPolicy, ErrorPolicy>::ResistivePre2(Driver *d, int stage) {
  if (stage == 1) ResistiveImplicitSolve(d, 0);
  return TaskStatus::complete;
}

template<class EOSPolicy, class ErrorPolicy>
TaskStatus DynGRMHDPS<EOSPolicy, ErrorPolicy>::ResistiveC2PPre(Driver *d, int stage) {
  if (stage == 1) ResistiveC2P();
  return TaskStatus::complete;
}

template<class EOSPolicy, class ErrorPolicy>
TaskStatus DynGRMHDPS<EOSPolicy, ErrorPolicy>::ResistiveImplicit(Driver *d, int stage) {
  ResistiveImplicitSolve(d, stage);
  return TaskStatus::complete;
}

template<class EOSPolicy, class ErrorPolicy>
TaskStatus DynGRMHDPS<EOSPolicy, ErrorPolicy>::ResistiveConToPrim(Driver *d, int stage) {
  if (stage == 0) {
    // driver initialisation (after the pgen and the U/B exchange): sync E~ ghosts
    presist->CheckConstantMetric();
    presist->ExchangeEBlocking();
  }
  ResistiveC2P();
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! LF fluxes of (D, S_i, tau, D Y) with the EM stress included, and the convective
//! charge current qflx on every face used by CTE (07 §2.2, §2.3).

template<class EOSPolicy, class ErrorPolicy>
TaskStatus DynGRMHDPS<EOSPolicy, ErrorPolicy>::ResistiveFluxes(Driver *pdriver,
                                                               int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  const int act[3] = {1, multi_d ? 1 : 0, three_d ? 1 : 0};
  const int nmhd = pmy_pack->pmhd->nmhd;
  const int nscal = pmy_pack->pmhd->nscalars;
  auto &ps_ = eos.ps;
  const Real mb = ps_.GetEOS().GetBaryonMass();
  auto adm = pmy_pack->padm->adm;
  auto w0_ = pmy_pack->pmhd->w0;
  auto u0_ = pmy_pack->pmhd->u0;
  auto emc_ = presist->emc;
  DvceFaceFld4D<Real> bf = pmy_pack->pmhd->b0;
  DvceFaceFld4D<Real> ef = presist->e0;
  DvceFaceFld4D<Real> qf = presist->qflx;
  auto uflx = pmy_pack->pmhd->uflx;
  auto &mbsize = pmy_pack->pmb->mb_size;
  const bool rusanov = (presist->params.charge_flux == resistive::ChargeFlux::rusanov);

  for (int d = 0; d < 3; ++d) {
    if (!act[d]) continue;
    const int il = is, iu = (d == 0) ? ie+1 : ie;
    const int jl = js, ju = (d == 1) ? je+1 : je;
    const int kl = ks, ku = (d == 2) ? ke+1 : ke;
    par_for("resist_flux", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const int cR[3] = {i, j, k};
      int cL[3] = {i, j, k};
      cL[d] -= 1;
      int cLL[3] = {cL[0], cL[1], cL[2]}, cRR[3] = {i, j, k};
      cLL[d] -= 1;
      cRR[d] += 1;

      resistive::PointMetric pm;
      resistive::ZeroMetric(pm);
      resistive::AddMetric(adm, m, cL[2], cL[1], cL[0], 0.5, pm);
      resistive::AddMetric(adm, m, cR[2], cR[1], cR[0], 0.5, pm);
      const Real sg = resistive::SqrtDet(pm.g);
      Real gu[NSPMETRIC];
      resistive::MetricInverse(pm.g, gu);

      // PLM (van Leer) to the face
      Real wl[kNW], wr[kNW], eml[6], emr[6];
      const int nw = 5 + nscal;
      for (int n = 0; n < nw; ++n) {
        const int iv = (n < 5) ? n : nmhd + (n - 5);
        const Real qll = w0_(m, iv, cLL[2], cLL[1], cLL[0]);
        const Real ql = w0_(m, iv, cL[2], cL[1], cL[0]);
        const Real qr = w0_(m, iv, cR[2], cR[1], cR[0]);
        const Real qrr = w0_(m, iv, cRR[2], cRR[1], cRR[0]);
        wl[n] = ql + 0.5*resistive::VanLeerSlope(ql - qll, qr - ql);
        wr[n] = qr - 0.5*resistive::VanLeerSlope(qr - ql, qrr - qr);
      }
      for (int n = 0; n < 6; ++n) {
        const Real qll = emc_(m, n, cLL[2], cLL[1], cLL[0]);
        const Real ql = emc_(m, n, cL[2], cL[1], cL[0]);
        const Real qr = emc_(m, n, cR[2], cR[1], cR[0]);
        const Real qrr = emc_(m, n, cRR[2], cRR[1], cRR[0]);
        eml[n] = ql + 0.5*resistive::VanLeerSlope(ql - qll, qr - ql);
        emr[n] = qr - 0.5*resistive::VanLeerSlope(qr - ql, qrr - qr);
      }
      // face-normal E and B are the staggered face values
      Real en, bn;
      if (d == 0) {
        en = ef.x1f(m, k, j, i); bn = bf.x1f(m, k, j, i);
      } else if (d == 1) {
        en = ef.x2f(m, k, j, i); bn = bf.x2f(m, k, j, i);
      } else {
        en = ef.x3f(m, k, j, i); bn = bf.x3f(m, k, j, i);
      }
      eml[d] = emr[d] = en/sg;
      eml[3+d] = emr[3+d] = bn/sg;

      Real UL[NCONS], FL[NCONS], UR[NCONS], FR[NCONS];
      SideState(ps_, pm.g, sg, pm.alpha, pm.beta, d, mb, wl, nscal, eml, UL, FL);
      SideState(ps_, pm.g, sg, pm.alpha, pm.beta, d, mb, wr, nscal, emr, UR, FR);
      const Real a = fabs(pm.beta[d]) + pm.alpha*sqrt(resistive::InverseDiag(gu, d));

      Real flx[NCONS];
      for (int n = 0; n < NCONS; ++n) {
        flx[n] = 0.5*sg*((FL[n] + FR[n]) - a*(UR[n] - UL[n]));
      }
      auto &fx = (d == 0) ? uflx.x1f : ((d == 1) ? uflx.x2f : uflx.x3f);
      fx(m, IDN, k, j, i) = flx[CDN];
      fx(m, IM1, k, j, i) = flx[CSX];
      fx(m, IM2, k, j, i) = flx[CSY];
      fx(m, IM3, k, j, i) = flx[CSZ];
      fx(m, IEN, k, j, i) = flx[CTA];
      for (int n = 0; n < nscal; ++n) fx(m, nmhd + n, k, j, i) = flx[CYD + n];

      // convective charge current
      const Real idx1 = 1.0/mbsize.d_view(m).dx1;
      const Real idx2 = 1.0/mbsize.d_view(m).dx2;
      const Real idx3 = 1.0/mbsize.d_view(m).dx3;
      const Real qL = CellCharge(ef, m, cL[2], cL[1], cL[0], idx1, idx2, idx3,
                                 multi_d, three_d);
      const Real qR = CellCharge(ef, m, cR[2], cR[1], cR[0], idx1, idx2, idx3,
                                 multi_d, three_d);
      Real qflux;
      if (rusanov) {
        Real ul[3], ur[3];
        for (int c = 0; c < 3; ++c) {
          ul[c] = w0_(m, IVX + c, cL[2], cL[1], cL[0]);
          ur[c] = w0_(m, IVX + c, cR[2], cR[1], cR[0]);
        }
        const Real vl = pm.alpha*ul[d]/sqrt(1.0 + resistive::Dot(pm.g, ul, ul))
                      - pm.beta[d];
        const Real vr = pm.alpha*ur[d]/sqrt(1.0 + resistive::Dot(pm.g, ur, ur))
                      - pm.beta[d];
        qflux = resistive::ChargeFluxRusanov(qL, vl, qR, vr, a);
      } else {
        qflux = resistive::ChargeFluxUpwind(flx[CDN], qL,
                                            u0_(m, IDN, cL[2], cL[1], cL[0]),
                                            qR, u0_(m, IDN, cR[2], cR[1], cR[0]));
      }
      if (d == 0) {
        qf.x1f(m, k, j, i) = qflux;
      } else if (d == 1) {
        qf.x2f(m, k, j, i) = qflux;
      } else {
        qf.x3f(m, k, j, i) = qflux;
      }
    });
  }

  // charge current through faces normal to inactive directions: cell value
  for (int d = 1; d < 3; ++d) {
    if (act[d]) continue;
    const int il = is, iu = ie;
    const int jl = js, ju = (d == 1) ? js+1 : je;
    const int kl = ks, ku = (d == 2) ? ks+1 : ke;
    par_for("resist_qflx_inactive", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      int c[3] = {i, j, k};
      c[d] = (d == 1) ? js : ks;
      resistive::PointMetric pm;
      resistive::LoadMetric(adm, m, c[2], c[1], c[0], pm);
      const Real idx1 = 1.0/mbsize.d_view(m).dx1;
      const Real idx2 = 1.0/mbsize.d_view(m).dx2;
      const Real idx3 = 1.0/mbsize.d_view(m).dx3;
      const Real q = CellCharge(ef, m, c[2], c[1], c[0], idx1, idx2, idx3,
                                multi_d, three_d);
      Real u[3];
      for (int a = 0; a < 3; ++a) u[a] = w0_(m, IVX + a, c[2], c[1], c[0]);
      const Real W = sqrt(1.0 + resistive::Dot(pm.g, u, u));
      const Real v = pm.alpha*u[d]/W - pm.beta[d];
      if (d == 1) {
        qf.x2f(m, k, j, i) = q*v;
      } else {
        qf.x3f(m, k, j, i) = q*v;
      }
    });
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! Implicit Ohmic step for ImEx stage istage = estage + 2 (IonNeutral::ImpRKUpdate form):
//! add the a_twid sources, then (estage < nexp) solve the cell problem for u^i by
//! Newton-Broyden (07 D5), update every face E~ by the closed form with the adjacent-cell
//! u (D2) and store R = (E~new - E~old)/(a_impl dt) in Driver::impl_src_e.

template<class EOSPolicy, class ErrorPolicy>
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::ResistiveImplicitSolve(Driver *pdriver,
                                                                int estage) {
  const int istage = estage + 2;
  presist->AddImexSources(pdriver, istage);
  if (estage >= pdriver->nexp_stages) return;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*indcs.ng) : 1;
  const int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*indcs.ng) : 1;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  const int nmhd = pmy_pack->pmhd->nmhd;
  const int nscal = pmy_pack->pmhd->nscalars;
  auto &ps_ = eos.ps;
  const Real mb = ps_.GetEOS().GetBaryonMass();
  auto adm = pmy_pack->padm->adm;
  auto w0_ = pmy_pack->pmhd->w0;
  auto u0_ = pmy_pack->pmhd->u0;
  auto temp_ = temperature;
  auto cw = presist->cwork;
  DvceFaceFld4D<Real> bf = pmy_pack->pmhd->b0;
  DvceFaceFld4D<Real> ef = presist->e0;
  const resistive::Conductivity cond = presist->cond;
  const resistive::NewtonStats nstat = presist->newton;
  const Real tol = presist->params.newton_tol;
  const int maxit = presist->params.newton_maxit;
  const bool fallback =
      (presist->params.newton_fallback == resistive::NewtonFallback::lagged_v);
  const Real adt = pdriver->a_impl*(pmy_pack->pmesh->dt);

  par_for("resist_impl_cell", DevExeSpace(), 0, nmb1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    resistive::PointMetric pm;
    resistive::LoadMetric(adm, m, k, j, i, pm);
    const Real sg = resistive::SqrtDet(pm.g);
    const Real isg = 1.0/sg;
    Real gu[NSPMETRIC];
    resistive::MetricInverse(pm.g, gu);
    Real cons[NCONS];
    for (int n = 0; n < NCONS; ++n) cons[n] = 0.0;
    cons[CDN] = u0_(m, IDN, k, j, i)*isg;
    cons[CSX] = u0_(m, IM1, k, j, i)*isg;
    cons[CSY] = u0_(m, IM2, k, j, i)*isg;
    cons[CSZ] = u0_(m, IM3, k, j, i)*isg;
    cons[CTA] = u0_(m, IEN, k, j, i)*isg;
    for (int n = 0; n < nscal; ++n) cons[CYD + n] = u0_(m, nmhd + n, k, j, i)*isg;
    Real B[3], R[3], ug[3];
    for (int a = 0; a < 3; ++a) {
      B[a] = FaceAvg(bf, a, m, k, j, i)*isg;
      R[a] = FaceAvg(ef, a, m, k, j, i)*isg;
      ug[a] = w0_(m, IVX + a, k, j, i);
    }
    const Real sdt = cond(w0_(m, IDN, k, j, i)).par*pm.alpha*adt;
    Real prim[NPRIM], E[3];
    resistive::CellSolveResult res = resistive::ResistiveCellSolve(
        ps_, pm.g, gu, sdt, cons, B, R, ug, tol, maxit, fallback, prim, E);
    nstat.Record(res.iterations, res.fallback, res.failed);
    Real u[3] = {ug[0], ug[1], ug[2]};
    if (!res.failed) {
      for (int a = 0; a < 3; ++a) u[a] = prim[PVX + a];
      w0_(m, IDN, k, j, i) = prim[PRH]*mb;
      w0_(m, IVX, k, j, i) = prim[PVX];
      w0_(m, IVY, k, j, i) = prim[PVY];
      w0_(m, IVZ, k, j, i) = prim[PVZ];
      w0_(m, IPR, k, j, i) = prim[PPR];
      for (int n = 0; n < nscal; ++n) w0_(m, nmhd + n, k, j, i) = prim[PYF + n];
      temp_(m, 0, k, j, i) = prim[PTM];
    }
    for (int a = 0; a < 3; ++a) {
      cw(m, resistive::kWU + a, k, j, i) = u[a];
      cw(m, resistive::kWR + a, k, j, i) = R[a];
      cw(m, resistive::kWB + a, k, j, i) = B[a];
    }
    cw(m, resistive::kWS, k, j, i) = sdt;
  });

  auto rs = pdriver->impl_src_e;
  const int s = istage - 1;
  const int nn[3] = {n1, n2, n3};
  for (int d = 0; d < 3; ++d) {
    const int iu = (d == 0) ? n1 : n1-1;
    const int ju = (d == 1) ? n2 : n2-1;
    const int ku = (d == 2) ? n3 : n3-1;
    par_for("resist_impl_face", DevExeSpace(), 0, nmb1, 0, ku, 0, ju, 0, iu,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      int cL[3] = {i, j, k}, cR[3] = {i, j, k};
      cL[d] = (cL[d] - 1 < 0) ? 0 : cL[d] - 1;
      cR[d] = (cR[d] > nn[d] - 1) ? nn[d] - 1 : cR[d];
      resistive::PointMetric pm;
      resistive::ZeroMetric(pm);
      resistive::AddMetric(adm, m, cL[2], cL[1], cL[0], 0.5, pm);
      resistive::AddMetric(adm, m, cR[2], cR[1], cR[0], 0.5, pm);
      const Real sg = resistive::SqrtDet(pm.g);
      Real ul[3], ur[3], B[3], R[3];
      for (int a = 0; a < 3; ++a) {
        ul[a] = cw(m, resistive::kWU + a, cL[2], cL[1], cL[0]);
        ur[a] = cw(m, resistive::kWU + a, cR[2], cR[1], cR[0]);
        B[a] = 0.5*(cw(m, resistive::kWB + a, cL[2], cL[1], cL[0]) +
                    cw(m, resistive::kWB + a, cR[2], cR[1], cR[0]));
        R[a] = 0.5*(cw(m, resistive::kWR + a, cL[2], cL[1], cL[0]) +
                    cw(m, resistive::kWR + a, cR[2], cR[1], cR[0]));
      }
      const Real sdt = 0.5*(cw(m, resistive::kWS, cL[2], cL[1], cL[0]) +
                            cw(m, resistive::kWS, cR[2], cR[1], cR[0]));
      auto &ex = (d == 0) ? ef.x1f : ((d == 1) ? ef.x2f : ef.x3f);
      auto &bx = (d == 0) ? bf.x1f : ((d == 1) ? bf.x2f : bf.x3f);
      const Real eold = ex(m, k, j, i);
      R[d] = eold/sg;
      B[d] = bx(m, k, j, i)/sg;
      Real E[3];
      resistive::OhmicFaceE(pm.g, sg, sdt, ul, ur, B, R, E);
      const Real enew = sg*E[d];
      rs(s, m, d, k, j, i) = (enew - eold)/adt;
      ex(m, k, j, i) = enew;
    });
  }
}

//----------------------------------------------------------------------------------------
//! Final resistive c2p on all cells (07 D4): E = face average of E~/sqrt(gamma), remove
//! the EM part from (S_i, tau), hydro c2p with B = 0. Sets w0, temperature, bcc0 and emc.

template<class EOSPolicy, class ErrorPolicy>
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::ResistiveC2P() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*indcs.ng) : 1;
  const int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*indcs.ng) : 1;
  const int nmb = pmy_pack->nmb_thispack;
  const int nmhd = pmy_pack->pmhd->nmhd;
  const int nscal = pmy_pack->pmhd->nscalars;
  auto &ps_ = eos.ps;
  const Real mb = ps_.GetEOS().GetBaryonMass();
  auto adm = pmy_pack->padm->adm;
  auto w0_ = pmy_pack->pmhd->w0;
  auto u0_ = pmy_pack->pmhd->u0;
  auto bcc_ = pmy_pack->pmhd->bcc0;
  auto temp_ = temperature;
  auto emc_ = presist->emc;
  DvceFaceFld4D<Real> bf = pmy_pack->pmhd->b0;
  DvceFaceFld4D<Real> ef = presist->e0;
  const int nkji = n3*n2*n1, nji = n2*n1;

  int nerr = 0;
  Kokkos::parallel_reduce("resist_c2p", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*nkji),
  KOKKOS_LAMBDA(const int idx, int &sumerr) {
    const int m = idx/nkji;
    int r = idx - m*nkji;
    const int k = r/nji;
    r -= k*nji;
    const int j = r/n1;
    const int i = r - j*n1;
    resistive::PointMetric pm;
    resistive::LoadMetric(adm, m, k, j, i, pm);
    const Real sg = resistive::SqrtDet(pm.g);
    const Real isg = 1.0/sg;
    Real gu[NSPMETRIC];
    resistive::MetricInverse(pm.g, gu);
    Real E[3], B[3], bcc[3];
    for (int a = 0; a < 3; ++a) {
      bcc[a] = FaceAvg(bf, a, m, k, j, i);
      B[a] = bcc[a]*isg;
      E[a] = FaceAvg(ef, a, m, k, j, i)*isg;
    }
    Real s_em[3], e_em;
    resistive::EMParts(pm.g, sg, E, B, s_em, e_em);
    Real cons[NCONS], prim[NPRIM];
    for (int n = 0; n < NCONS; ++n) cons[n] = 0.0;
    for (int n = 0; n < NPRIM; ++n) prim[n] = 0.0;
    cons[CDN] = u0_(m, IDN, k, j, i)*isg;
    cons[CSX] = u0_(m, IM1, k, j, i)*isg - s_em[0];
    cons[CSY] = u0_(m, IM2, k, j, i)*isg - s_em[1];
    cons[CSZ] = u0_(m, IM3, k, j, i)*isg - s_em[2];
    cons[CTA] = u0_(m, IEN, k, j, i)*isg - e_em;
    for (int n = 0; n < nscal; ++n) cons[CYD + n] = u0_(m, nmhd + n, k, j, i)*isg;
    Real bz[NMAG] = {0.0, 0.0, 0.0};
    Real g[NSPMETRIC];
    for (int n = 0; n < NSPMETRIC; ++n) g[n] = pm.g[n];
    Primitive::SolverResult res = ps_.ConToPrim(prim, cons, bz, g, gu);
    if (res.error != Primitive::Error::SUCCESS) sumerr++;
    w0_(m, IDN, k, j, i) = prim[PRH]*mb;
    w0_(m, IVX, k, j, i) = prim[PVX];
    w0_(m, IVY, k, j, i) = prim[PVY];
    w0_(m, IVZ, k, j, i) = prim[PVZ];
    w0_(m, IPR, k, j, i) = prim[PPR];
    for (int n = 0; n < nscal; ++n) w0_(m, nmhd + n, k, j, i) = prim[PYF + n];
    temp_(m, 0, k, j, i) = prim[PTM];
    for (int a = 0; a < 3; ++a) {
      bcc_(m, a, k, j, i) = bcc[a];
      emc_(m, resistive::kEX + a, k, j, i) = E[a];
      emc_(m, resistive::kBX + a, k, j, i) = B[a];
    }
    if (res.cons_floor || res.cons_adjusted) {
      u0_(m, IDN, k, j, i) = cons[CDN]*sg;
      u0_(m, IM1, k, j, i) = (cons[CSX] + s_em[0])*sg;
      u0_(m, IM2, k, j, i) = (cons[CSY] + s_em[1])*sg;
      u0_(m, IM3, k, j, i) = (cons[CSZ] + s_em[2])*sg;
      u0_(m, IEN, k, j, i) = (cons[CTA] + e_em)*sg;
      for (int n = 0; n < nscal; ++n) u0_(m, nmhd + n, k, j, i) = cons[CYD + n]*sg;
    }
  }, Kokkos::Sum<int>(nerr));
  if (nerr > 0) {
    std::cout << "### WARNING: resistive c2p failed in " << nerr << " cells on rank "
              << global_variable::my_rank << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! Initial prim -> cons: fluid part from PrimitiveSolver (B = 0) plus the EM energy and
//! momentum of the face fields b0 and E~ (the pgen sets w0, b0 and presist->e0).

template<class EOSPolicy, class ErrorPolicy>
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::ResistivePrimToCon(int il, int iu, int jl,
                                                            int ju, int kl, int ku) {
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  const int nmhd = pmy_pack->pmhd->nmhd;
  const int nscal = pmy_pack->pmhd->nscalars;
  auto &ps_ = eos.ps;
  const Real mb = ps_.GetEOS().GetBaryonMass();
  auto adm = pmy_pack->padm->adm;
  auto w0_ = pmy_pack->pmhd->w0;
  auto u0_ = pmy_pack->pmhd->u0;
  auto bcc_ = pmy_pack->pmhd->bcc0;
  auto temp_ = temperature;
  auto emc_ = presist->emc;
  DvceFaceFld4D<Real> bf = pmy_pack->pmhd->b0;
  DvceFaceFld4D<Real> ef = presist->e0;
  par_for("resist_p2c", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    resistive::PointMetric pm;
    resistive::LoadMetric(adm, m, k, j, i, pm);
    const Real sg = resistive::SqrtDet(pm.g);
    Real prim[NPRIM], cons[NCONS];
    for (int n = 0; n < NPRIM; ++n) prim[n] = 0.0;
    for (int n = 0; n < NCONS; ++n) cons[n] = 0.0;
    prim[PRH] = w0_(m, IDN, k, j, i)/mb;
    prim[PVX] = w0_(m, IVX, k, j, i);
    prim[PVY] = w0_(m, IVY, k, j, i);
    prim[PVZ] = w0_(m, IVZ, k, j, i);
    prim[PPR] = w0_(m, IPR, k, j, i);
    for (int n = 0; n < nscal; ++n) prim[PYF + n] = w0_(m, nmhd + n, k, j, i);
    prim[PTM] = ps_.GetEOS().GetTemperatureFromP(prim[PRH], prim[PPR], &prim[PYF]);
    ps_.GetEOS().ApplyPrimitiveFloor(prim[PRH], &prim[PVX], prim[PPR], prim[PTM],
                                     &prim[PYF]);
    Real g[NSPMETRIC];
    for (int n = 0; n < NSPMETRIC; ++n) g[n] = pm.g[n];
    Real bz[NMAG] = {0.0, 0.0, 0.0};
    ps_.PrimToCon(prim, cons, bz, g);
    Real E[3], B[3], bcc[3];
    for (int a = 0; a < 3; ++a) {
      bcc[a] = FaceAvg(bf, a, m, k, j, i);
      B[a] = bcc[a]/sg;
      E[a] = FaceAvg(ef, a, m, k, j, i)/sg;
    }
    Real dim[3], dien;
    resistive::EMIncrementDyn(pm.g, sg, E, B, dim, dien);
    u0_(m, IDN, k, j, i) = sg*cons[CDN];
    u0_(m, IM1, k, j, i) = sg*cons[CSX] + dim[0];
    u0_(m, IM2, k, j, i) = sg*cons[CSY] + dim[1];
    u0_(m, IM3, k, j, i) = sg*cons[CSZ] + dim[2];
    u0_(m, IEN, k, j, i) = sg*cons[CTA] + dien;
    for (int n = 0; n < nscal; ++n) u0_(m, nmhd + n, k, j, i) = sg*cons[CYD + n];
    temp_(m, 0, k, j, i) = prim[PTM];
    for (int a = 0; a < 3; ++a) {
      bcc_(m, a, k, j, i) = bcc[a];
      emc_(m, resistive::kEX + a, k, j, i) = E[a];
      emc_(m, resistive::kBX + a, k, j, i) = B[a];
    }
  });
}

#define INSTANTIATE_RESISTIVE(EOSPolicy, ErrorPolicy) \
template void DynGRMHDPS<EOSPolicy, ErrorPolicy>::QueueResistiveTasks(); \
template TaskStatus DynGRMHDPS<EOSPolicy, ErrorPolicy>::ResistiveFluxes(Driver*, int); \
template TaskStatus DynGRMHDPS<EOSPolicy, ErrorPolicy>::ResistivePre1(Driver*, int); \
template TaskStatus DynGRMHDPS<EOSPolicy, ErrorPolicy>::ResistivePre2(Driver*, int); \
template TaskStatus DynGRMHDPS<EOSPolicy, ErrorPolicy>::ResistiveC2PPre(Driver*, int); \
template TaskStatus DynGRMHDPS<EOSPolicy, ErrorPolicy>::ResistiveImplicit(Driver*, int); \
template TaskStatus \
    DynGRMHDPS<EOSPolicy, ErrorPolicy>::ResistiveConToPrim(Driver*, int); \
template void DynGRMHDPS<EOSPolicy, ErrorPolicy>::ResistiveImplicitSolve(Driver*, int); \
template void DynGRMHDPS<EOSPolicy, ErrorPolicy>::ResistiveC2P(); \
template void DynGRMHDPS<EOSPolicy, ErrorPolicy>::ResistivePrimToCon(int, int, int, int, \
                                                                     int, int);

INSTANTIATE_RESISTIVE(Primitive::IdealGas, Primitive::ResetFloor)
INSTANTIATE_RESISTIVE(Primitive::PiecewisePolytrope, Primitive::ResetFloor)
INSTANTIATE_RESISTIVE(Primitive::EOSCompOSE<Primitive::NormalLogs>, Primitive::ResetFloor)
INSTANTIATE_RESISTIVE(Primitive::EOSCompOSE<Primitive::NQTLogs>, Primitive::ResetFloor)
INSTANTIATE_RESISTIVE(Primitive::EOSHybrid<Primitive::NormalLogs>, Primitive::ResetFloor)
INSTANTIATE_RESISTIVE(Primitive::EOSHybrid<Primitive::NQTLogs>, Primitive::ResetFloor)

}  // namespace dyngr
