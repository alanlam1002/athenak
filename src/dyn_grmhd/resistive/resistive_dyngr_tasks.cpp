//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file resistive_dyngr_tasks.cpp
//! \brief EOS-dependent part of resistive GRMHD on the dyngr path (07_scoping_v2 §3.3):
//!        task queue, MHLLC/LF fluxes of the total (fluid + EM) conserved variables
//!        and of the charge, the implicit Ohmic step (cell Newton + face closed form,
//!        D2/D5) and the final resistive c2p (D4). Explicitly instantiated for all
//!        dyngr policies.

#include <cmath>
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "mhd/mhd.hpp"
#include "tasklist/numerical_relativity.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "dyn_grmhd/resistive/resistive_dyngr.hpp"
#include "dyn_grmhd/resistive/resistive_mesh_util.hpp"
#include "dyn_grmhd/resistive/resistive_em.hpp"
#include "dyn_grmhd/resistive/resistive_edge.hpp"
#include "dyn_grmhd/resistive/resistive_implicit.hpp"
#include "dyn_grmhd/resistive/resistive_riemann.hpp"
#include "dyn_grmhd/resistive/resistive_probe.hpp"

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
void SidePrim(const PSolver &ps, const Real mb, const Real w[kNW], const int nscal,
              Real prim[NPRIM]) {
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
}

template <class PSolver>
KOKKOS_INLINE_FUNCTION
void SideState(const PSolver &ps, const Real g_in[NSPMETRIC], const Real sg,
               const Real alpha, const Real beta[3], const int d, const Real mb,
               const Real w[kNW], const int nscal, const Real em[6],
               Real U[NCONS], Real F[NCONS]) {
  Real prim[NPRIM];
  SidePrim(ps, mb, w, nscal, prim);
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
//! Test-only probe dumps (07 v2.9 §3.2.4(e); resistive_probe.hpp).

namespace {
template <class F>
void ProbeFace(resistive::ProbeFile *f, const char *nm, const F &v) {
  const std::string s(nm);
  f->Add((s + "1").c_str(), v.x1f);
  f->Add((s + "2").c_str(), v.x2f);
  f->Add((s + "3").c_str(), v.x3f);
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
  pnr->QueueTask(&ResistiveDynGR::SyncSendEP, pr, MHD_RSyncSEP, "MHD_RSyncSEP",
                 Task_Run, {MHD_RPre1});
  pnr->QueueTask(&ResistiveDynGR::SyncRecvEP, pr, MHD_RSyncREP, "MHD_RSyncREP",
                 Task_Run, {MHD_RSyncSEP});
  pnr->QueueTask(&ResistiveDynGR::SendEP, pr, MHD_RSendEP, "MHD_RSendEP", Task_Run,
                 {MHD_RSyncREP});
  pnr->QueueTask(&ResistiveDynGR::RecvEP, pr, MHD_RRecvEP, "MHD_RRecvEP", Task_Run,
                 {MHD_RSendEP});
  pnr->QueueTask(&ResistiveDynGR::ApplyBCsEP, pr, MHD_RBCsEP, "MHD_RBCsEP", Task_Run,
                 {MHD_RRecvEP});
  pnr->QueueTask(&Self::ResistivePre2, this, MHD_RPre2, "MHD_RPre2", Task_Run,
                 {MHD_RBCsEP});
  pnr->QueueTask(&ResistiveDynGR::SyncSendEP2, pr, MHD_RSyncSEP2, "MHD_RSyncSEP2",
                 Task_Run, {MHD_RPre2});
  pnr->QueueTask(&ResistiveDynGR::SyncRecvEP2, pr, MHD_RSyncREP2, "MHD_RSyncREP2",
                 Task_Run, {MHD_RSyncSEP2});
  pnr->QueueTask(&ResistiveDynGR::SendEP2, pr, MHD_RSendEP2, "MHD_RSendEP2", Task_Run,
                 {MHD_RSyncREP2});
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
  pnr->QueueTask(&ResistiveDynGR::SyncSendE, pr, MHD_RSyncSE, "MHD_RSyncSE",
                 Task_Run, {MHD_RCTE});
  pnr->QueueTask(&ResistiveDynGR::SyncRecvE, pr, MHD_RSyncRE, "MHD_RSyncRE",
                 Task_Run, {MHD_RSyncSE});
  pnr->QueueTask(&ResistiveDynGR::SendE, pr, MHD_RSendE, "MHD_RSendE", Task_Run,
                 {MHD_RSyncRE});
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
  pnr->QueueTask(&ResistiveDynGR::SyncSendEI, pr, MHD_RSyncSEI, "MHD_RSyncSEI",
                 Task_Run, {MHD_RImpl});
  pnr->QueueTask(&ResistiveDynGR::SyncRecvEI, pr, MHD_RSyncREI, "MHD_RSyncREI",
                 Task_Run, {MHD_RSyncSEI});
  pnr->QueueTask(&ResistiveDynGR::SendEI, pr, MHD_RSendEI, "MHD_RSendEI", Task_Run,
                 {MHD_RSyncREI});
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
  if (stage == 1) ResistiveC2P(false, 0);
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
  ResistiveC2P(stage > 0, (stage > 0) ? stage : -1);
  if (presist->Probing()) {
    resistive::ProbeFile f(presist->probe_base, pmy_pack, stage, "C");
    f.Add("u0", pmy_pack->pmhd->u0);
    f.Add("w0", pmy_pack->pmhd->w0);
    f.Add("temp", temperature);
    f.Add("emc", presist->emc);
    f.Add("cw", presist->cwork);
    if (presist->params.Offset()) f.Add("doff", presist->doff);
  }
  if (stage == d->nexp_stages) {
    presist->MaybeWriteHist(pmy_pack->pmesh->time + pmy_pack->pmesh->dt,
                            pmy_pack->pmesh->ncycle + 1);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! Tetrad MHLLC (default) or LF fluxes of (D, S_i, tau, D Y) with the EM stress included,
//! and the convective
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
  const bool mhllc = (presist->params.rsolver == resistive::RSolver::mhllc);
  if (presist->Probing()) {
    resistive::ProbeFile f(presist->probe_base, pmy_pack, stage, "F");
    f.Add("u0", u0_);
    f.Add("w0", w0_);
    f.Add("temp", temperature);
    f.Add("emc", emc_);
    ProbeFace(&f, "e", ef);
    ProbeFace(&f, "b", bf);
  }

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

      const Real a = fabs(pm.beta[d]) + pm.alpha*sqrt(resistive::InverseDiag(gu, d));
      Real flx[NCONS];
      if (mhllc) {
        Real priml[NPRIM], primr[NPRIM];
        SidePrim(ps_, mb, wl, nscal, priml);
        SidePrim(ps_, mb, wr, nscal, primr);
        resistive::MHLLCFace(ps_, pm.g, gu, sg, pm.alpha, pm.beta, d, priml, primr,
                             nscal, eml, emr, flx);
      } else {
        Real UL[NCONS], FL[NCONS], UR[NCONS], FR[NCONS];
        SideState(ps_, pm.g, sg, pm.alpha, pm.beta, d, mb, wl, nscal, eml, UL, FL);
        SideState(ps_, pm.g, sg, pm.alpha, pm.beta, d, mb, wr, nscal, emr, UR, FR);
        for (int n = 0; n < NCONS; ++n) {
          flx[n] = 0.5*sg*((FL[n] + FR[n]) - a*(UR[n] - UL[n]));
        }
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
  const bool probe = presist->Probing();
  if (probe) {
    auto *pmhd = pmy_pack->pmhd;
    resistive::ProbeFile f(presist->probe_base, pmy_pack, istage, "Ia");
    f.Add("u0", pmhd->u0);
    f.Add("w0", pmhd->w0);
    ProbeFace(&f, "e", presist->e0);
    ProbeFace(&f, "b", pmhd->b0);
    ProbeFace(&f, "f", pmhd->uflx);
    ProbeFace(&f, "q", presist->qflx);
    f.Add("ee1", pmhd->efld.x1e);
    f.Add("ee2", pmhd->efld.x2e);
    f.Add("ee3", pmhd->efld.x3e);
    f.Add("pedge", presist->pedge);
  }
  presist->AddImexSources(pdriver, istage);
  if (estage >= pdriver->nexp_stages) return;
  if (probe) {
    resistive::ProbeFile f(presist->probe_base, pmy_pack, istage, "Ib");
    f.Add("u0", pmy_pack->pmhd->u0);
    ProbeFace(&f, "e", presist->e0);
  }

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
  resistive::CellSolveOptions opt;
  opt.form = (presist->params.implicit_residual == resistive::ImplicitResidual::momentum)
             ? resistive::kResidualMomentum : resistive::kResidualC2P;
  // 07 v2.8 §3.2.3 FA + blend (default since 07 v2.9)
  opt.blend = (presist->params.newton_fallback == resistive::NewtonFallback::blend);
  opt.floor_accept = presist->params.floor_accept;
  opt.floor_accept_tol = presist->params.floor_accept_tol;
  const bool proto = presist->params.Prototype();
  // 07 v2.10 §3.2.5: R = E_split = FaceAvg(E~)/sqrt(gamma) + dE_c in the cell solve
  const bool doffon = presist->params.Offset();
  const bool d4fail = doffon || (presist->params.d4 == resistive::D4Mode::resplit);
  auto doff_ = presist->doff;
  const resistive::ProtoLog plog = presist->plog;
  const int ncyc = pmy_pack->pmesh->ncycle;
  const bool dorfail = (ncyc < presist->params.fail_reason_cycles);
  const resistive::RFailCount rfail = presist->rfail;
  const bool rcen = presist->rfail_center_set;
  const Real rc0 = presist->rfail_center[0], rc1 = presist->rfail_center[1];
  const Real rc2 = presist->rfail_center[2];
  auto &mbsize = pmy_pack->pmb->mb_size;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  static_assert(resistive::kRFailReasons == resistive::kNReasons, "rfail reasons");
  const resistive::SigmaMHist hist = presist->hist;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, ks = indcs.ks;
  const int je = (indcs.nx2 > 1) ? indcs.je : js;
  const int ke = (indcs.nx3 > 1) ? indcs.ke : ks;
  auto pst = presist->pst;

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
    // R: face average (also the face solves' input via kWR); Rs: the cell solve's
    Real B[3], R[3], Rs[3], ug[3];
    for (int a = 0; a < 3; ++a) {
      B[a] = FaceAvg(bf, a, m, k, j, i)*isg;
      R[a] = FaceAvg(ef, a, m, k, j, i)*isg;
      Rs[a] = doffon ? R[a] + doff_(m, a, k, j, i) : R[a];
      ug[a] = w0_(m, IVX + a, k, j, i);
    }
    const Real sdt = cond(w0_(m, IDN, k, j, i)).par*pm.alpha*adt;
    Real prim[NPRIM], E[3];
    resistive::CellSolveResult res = resistive::ResistiveCellSolve(
        ps_, pm.g, gu, sdt, cons, B, Rs, ug, tol, maxit, fallback, prim, E, opt);
    nstat.Record(res.iterations, res.fallback, res.failed);
    if (probe) {
      // 07 v2.9 §3.2.4(e) items 2 and 4: ideal c2p W on the pre-solve totals
      Real cil[NCONS], pil[NPRIM], gd[NSPMETRIC], gi[NSPMETRIC];
      for (int n = 0; n < NCONS; ++n) cil[n] = cons[n];
      for (int n = 0; n < NPRIM; ++n) pil[n] = 0.0;
      for (int n = 0; n < NSPMETRIC; ++n) {
        gd[n] = pm.g[n];
        gi[n] = gu[n];
      }
      const Primitive::SolverResult ri = resistive::detail::IdealC2P(ps_, pil, cil, B,
                                                                     gd, gi);
      const Real uil[3] = {pil[PVX], pil[PVY], pil[PVZ]};
      const Real usl[3] = {prim[PVX], prim[PVY], prim[PVZ]};
      const int fl = (res.converged ? 1 : 0) | (res.fallback ? 2 : 0) |
                     (res.failed ? 4 : 0) | (res.floor ? 8 : 0) | (res.retry ? 16 : 0) |
                     (res.prestep_inadmissible ? 32 : 0) | (res.continued ? 64 : 0) |
                     (res.fa ? 128 : 0) | (res.fa_reject ? 256 : 0) |
                     (res.blend ? 512 : 0) | (res.il ? 1024 : 0) |
                     (res.il_floor ? 2048 : 0) | (res.keep ? 4096 : 0);
      pst(m, resistive::kPReason, k, j, i) = static_cast<Real>(res.reason);
      pst(m, resistive::kPFlags, k, j, i) = static_cast<Real>(fl);
      pst(m, resistive::kPIter, k, j, i) = static_cast<Real>(res.iterations);
      pst(m, resistive::kPWIL, k, j, i) = sqrt(1.0 + resistive::Dot(pm.g, uil, uil));
      pst(m, resistive::kPILErr, k, j, i) = static_cast<Real>(static_cast<int>(ri.error));
      pst(m, resistive::kPWSol, k, j, i) = res.failed ? -1.0
                                         : sqrt(1.0 + resistive::Dot(pm.g, usl, usl));
      for (int a = 0; a < 3; ++a) pst(m, resistive::kPE + a, k, j, i) = E[a];
    }
    if (i >= is && i <= ie && j >= js && j <= je && k >= ks && k <= ke) {
      // b^2/(rho h) from the solved state, or from the pre-solve w0, T and E~ if failed
      Real hp[NPRIM], eh[3];
      for (int n = 0; n < NPRIM; ++n) hp[n] = prim[n];
      for (int a = 0; a < 3; ++a) eh[a] = E[a];
      if (res.failed) {
        hp[PRH] = w0_(m, IDN, k, j, i)/mb;
        hp[PTM] = temp_(m, 0, k, j, i);
        for (int n = 0; n < nscal; ++n) hp[PYF + n] = w0_(m, nmhd + n, k, j, i);
        for (int a = 0; a < 3; ++a) {
          hp[PVX + a] = ug[a];
          eh[a] = Rs[a];
        }
      }
      const Real rho = hp[PRH]*mb;
      const Real rhoh = rho*ps_.GetEOS().GetEnthalpy(hp[PRH], hp[PTM], &hp[PYF]);
      const int bs = resistive::SigmaMHist::Bin(resistive::Dot(pm.g, B, B), rho);
      const int bm = resistive::SigmaMHist::Bin(
          resistive::ComovingB2(pm.g, sg, &hp[PVX], eh, B), rhoh);
      hist.Add(bs, bm, resistive::kHCells);
      if (res.converged) hist.Add(bs, bm, resistive::kHConverged);
      if (res.retry) hist.Add(bs, bm, resistive::kHRetry);
      if (res.fallback) hist.Add(bs, bm, resistive::kHFallback);
      if (res.prestep_inadmissible) hist.Add(bs, bm, resistive::kHPrestep);
      if (res.floor) hist.Add(bs, bm, resistive::kHFloor);
      if (res.continued) hist.Add(bs, bm, resistive::kHContinued);
      if (res.fa) hist.Add(bs, bm, resistive::kHFA);
      if (res.fa_reject) hist.Add(bs, bm, resistive::kHFAReject);
      if (res.blend) hist.Add(bs, bm, resistive::kHBlend);
      if (res.il) hist.Add(bs, bm, resistive::kHIL);
      if (res.il_floor) hist.Add(bs, bm, resistive::kHILFloor);
      if (res.keep) hist.Add(bs, bm, resistive::kHKeep);
      if (res.fa_deficit >= 0.0) plog.Add(resistive::kLogFADeficit, bm, res.fa_deficit);
      if (dorfail && res.reason != resistive::kReasonNone) {
        int rb = 0;
        if (rcen) {
          const Real x = CellCenterX(i - is, nx1, mbsize.d_view(m).x1min,
                                     mbsize.d_view(m).x1max) - rc0;
          const Real y = (nx2 > 1) ? CellCenterX(j - js, nx2, mbsize.d_view(m).x2min,
                                                  mbsize.d_view(m).x2max) - rc1 : 0.0;
          const Real z = (nx3 > 1) ? CellCenterX(k - ks, nx3, mbsize.d_view(m).x3min,
                                                  mbsize.d_view(m).x3max) - rc2 : 0.0;
          const Real rr = sqrt(x*x + y*y + z*z)/mbsize.d_view(m).dx1;
          rb = (rr < static_cast<Real>(resistive::kRFailRBins - 1))
               ? static_cast<int>(rr) : resistive::kRFailRBins - 1;
        }
        rfail.Add(res.reason, rb);
      }
    }
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
    if (proto) {
      // flag + cell E for ResistiveC2P (07 v2.8 §3.2.3(d)): FA and blend/IL cells
      const bool flg = res.fa || res.blend || res.il;
      cw(m, resistive::kWF, k, j, i) = static_cast<Real>(
          resistive::kFlagSolved + (flg ? resistive::kFlagFallback : 0) +
          ((d4fail && res.failed) ? resistive::kFlagFailed : 0));
      for (int a = 0; a < 3; ++a) {
        cw(m, resistive::kWE + a, k, j, i) = res.failed ? Rs[a] : E[a];
      }
    }
  });
  if (dorfail) presist->WriteRFail(ncyc + 1, istage);

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
      // each side keeps its own sigma (07 §3.2 step 2 with sigma(rho), Step 1b)
      const Real sdtl = cw(m, resistive::kWS, cL[2], cL[1], cL[0]);
      const Real sdtr = cw(m, resistive::kWS, cR[2], cR[1], cR[0]);
      auto &ex = (d == 0) ? ef.x1f : ((d == 1) ? ef.x2f : ef.x3f);
      auto &bx = (d == 0) ? bf.x1f : ((d == 1) ? bf.x2f : bf.x3f);
      const Real eold = ex(m, k, j, i);
      R[d] = eold/sg;
      B[d] = bx(m, k, j, i)/sg;
      Real E[3];
      resistive::OhmicFaceE(pm.g, sg, sdtl, sdtr, ul, ur, B, R, E);
      const Real enew = sg*E[d];
      rs(s, m, d, k, j, i) = (enew - eold)/adt;
      ex(m, k, j, i) = enew;
    });
  }
  if (probe) {
    resistive::ProbeFile f(presist->probe_base, pmy_pack, istage, "Ic");
    f.Add("pst", presist->pst);
    f.Add("cw", cw);
    f.Add("w0", w0_);
    f.Add("temp", temp_);
    ProbeFace(&f, "e", ef);
  }
}

//----------------------------------------------------------------------------------------
//! Final resistive c2p on all cells (07 D4): E = face average of E~/sqrt(gamma), remove
//! the EM part from (S_i, tau), hydro c2p with B = 0. Sets w0, temperature, bcc0 and emc.

template<class EOSPolicy, class ErrorPolicy>
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::ResistiveC2P(const bool record,
                                                     const int pstage) {
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
  const resistive::SigmaMHist hist = presist->hist;
  // 07 v2.8 §3.2.3(d) prototype: flagged cells (and every solved cell under
  // resistive_d4 = keep) keep their implicit-step fluid state; eps_D4 is logged
  const bool proto = presist->params.Prototype();
  const bool d4keep = (presist->params.d4 == resistive::D4Mode::keep);
  // 07 v2.10 §3.2.5: the fluid split uses E_split = FaceAvg(E~)/sqrt(gamma) + dE_c, with
  // dE_c = E_solve - FaceAvg(E~)/sqrt(gamma) set here after each cell solve (0 where the
  // solve hard-failed). offset_emc also puts E_split into emc. resplit (comparison arm):
  // after a solve, u0 += EM(E_face) - EM(E_solve) at fixed fluid.
  const bool doffon = presist->params.Offset();
  const bool emcsplit = (presist->params.d4 == resistive::D4Mode::offset_emc);
  const bool resplit = (presist->params.d4 == resistive::D4Mode::resplit);
  auto doff_ = presist->doff;
  Kokkos::View<Real, DevMemSpace> rde;
  if (resplit) rde = Kokkos::View<Real, DevMemSpace>("resist_resplit_de");
  auto &mbsize = pmy_pack->pmb->mb_size;
  const bool multi_d = (indcs.nx2 > 1), three_d = (indcs.nx3 > 1);
  const resistive::ProtoLog plog = presist->plog;
  auto cw = presist->cwork;
  const Real sigmax = presist->params.sigma_max;
  const bool ceiling = (sigmax < 0.1*std::numeric_limits<float>::max());
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, ks = indcs.ks;
  const int je = (indcs.nx2 > 1) ? indcs.je : js;
  const int ke = (indcs.nx3 > 1) ? indcs.ke : ks;

  if (doffon || resplit) {
    // set dE_c after a cell solve (0 where it hard-failed), or re-split u0 (resplit)
    const int nmb1 = nmb - 1;
    par_for("resist_d4split", DevExeSpace(), 0, nmb1, 0, n3-1, 0, n2-1, 0, n1-1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const int flag = static_cast<int>(cw(m, resistive::kWF, k, j, i));
      if ((flag & resistive::kFlagSolved) == 0) return;
      const bool sfailed = (flag & resistive::kFlagFailed) != 0;
      resistive::PointMetric pm;
      resistive::LoadMetric(adm, m, k, j, i, pm);
      const Real sg = resistive::SqrtDet(pm.g);
      const Real isg = 1.0/sg;
      Real Ef[3], Es[3], B[3];
      for (int a = 0; a < 3; ++a) {
        B[a] = FaceAvg(bf, a, m, k, j, i)*isg;
        Ef[a] = FaceAvg(ef, a, m, k, j, i)*isg;
        Es[a] = cw(m, resistive::kWE + a, k, j, i);
      }
      if (doffon) {
        for (int a = 0; a < 3; ++a) doff_(m, a, k, j, i) = sfailed ? 0.0 : Es[a] - Ef[a];
        return;
      }
      if (sfailed) return;
      Real ss_em[3], es_em, sf_em[3], ef_em;
      resistive::EMParts(pm.g, sg, Es, B, ss_em, es_em);
      resistive::EMParts(pm.g, sg, Ef, B, sf_em, ef_em);
      u0_(m, IM1, k, j, i) += (sf_em[0] - ss_em[0])*sg;
      u0_(m, IM2, k, j, i) += (sf_em[1] - ss_em[1])*sg;
      u0_(m, IM3, k, j, i) += (sf_em[2] - ss_em[2])*sg;
      u0_(m, IEN, k, j, i) += (ef_em - es_em)*sg;
      if (i >= is && i <= ie && j >= js && j <= je && k >= ks && k <= ke) {
        Real dv = mbsize.d_view(m).dx1;
        if (multi_d) dv *= mbsize.d_view(m).dx2;
        if (three_d) dv *= mbsize.d_view(m).dx3;
        Kokkos::atomic_add(&rde(), (ef_em - es_em)*sg*dv);
      }
    });
  }

  if (presist->Probing() && pstage >= 0) {
    // 07 v2.10 §3.2.5: c2p input (totals, solve flags and E, E~, dE_c) to log the
    // admissibility of the fluid split with E_split and with FaceAvg(E~)
    resistive::ProbeFile f(presist->probe_base, pmy_pack, (pstage > 0) ? pstage : 1,
                           (pstage > 0) ? "P" : "Ppre");
    f.Add("u0", u0_);
    f.Add("cw", cw);
    ProbeFace(&f, "e", ef);
    ProbeFace(&f, "b", bf);
    if (doffon) f.Add("doff", doff_);
  }

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
    if (doffon) {
      for (int a = 0; a < 3; ++a) E[a] += doff_(m, a, k, j, i);
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
    int flag = 0;
    if (proto) {
      flag = static_cast<int>(cw(m, resistive::kWF, k, j, i));
      cw(m, resistive::kWF, k, j, i) = 0.0;
    }
    const bool solved = (flag & resistive::kFlagSolved) != 0;
    const bool flagged = (flag & resistive::kFlagFallback) != 0;
    const bool keepst = solved && (flagged || d4keep);
    Primitive::SolverResult res;
    if (keepst) {
      // the implicit step's state: w0/temperature as the solve left them
      res.error = Primitive::Error::SUCCESS;
      res.iterations = 0;
      res.cons_floor = res.prim_floor = res.cons_adjusted = false;
      prim[PRH] = w0_(m, IDN, k, j, i)/mb;
      prim[PVX] = w0_(m, IVX, k, j, i);
      prim[PVY] = w0_(m, IVY, k, j, i);
      prim[PVZ] = w0_(m, IVZ, k, j, i);
      prim[PPR] = w0_(m, IPR, k, j, i);
      prim[PTM] = temp_(m, 0, k, j, i);
      for (int n = 0; n < nscal; ++n) prim[PYF + n] = w0_(m, nmhd + n, k, j, i);
    } else {
      res = ps_.ConToPrim(prim, cons, bz, g, gu);
    }
    if (res.error != Primitive::Error::SUCCESS) sumerr++;
    // sigma_M ceiling (decision 10 A): add fluid mass at fixed u, T, Y; E~, B~ untouched
    const Real b2 = resistive::Dot(pm.g, B, B);
    bool ceil_on = false;
    if (ceiling && prim[PRH]*mb < b2/sigmax) {
      ceil_on = true;
      Real Y[MAX_SPECIES] = {0.0};
      for (int n = 0; n < nscal; ++n) Y[n] = prim[PYF + n];
      prim[PRH] = b2/(sigmax*mb);
      prim[PPR] = ps_.GetEOS().GetPressure(prim[PRH], prim[PTM], Y);
      ps_.PrimToCon(prim, cons, bz, g);
      res.cons_adjusted = true;
    }
    if (record && i >= is && i <= ie && j >= js && j <= je && k >= ks && k <= ke) {
      const Real rho = prim[PRH]*mb;
      const Real rhoh = rho*ps_.GetEOS().GetEnthalpy(prim[PRH], prim[PTM], &prim[PYF]);
      const int bs = resistive::SigmaMHist::Bin(b2, rho);
      const int bm = resistive::SigmaMHist::Bin(
          resistive::ComovingB2(pm.g, sg, &prim[PVX], E, B), rhoh);
      if (ceil_on) hist.Add(bs, bm, resistive::kHCeiling);
      if (resistive::Dot(pm.g, E, E) > b2) hist.Add(bs, bm, resistive::kHEgtB);
      if (solved) {
        Real de[3];
        for (int a = 0; a < 3; ++a) {
          // eps_D4 is against the face average (E - dE_c under offset|offset_emc)
          de[a] = (doffon ? E[a] - doff_(m, a, k, j, i) : E[a])
                  - cw(m, resistive::kWE + a, k, j, i);
        }
        const Real eps = sqrt(fmax(resistive::Dot(pm.g, de, de), 0.0))/sqrt(b2);
        plog.Add(flagged ? resistive::kLogEpsFlagged : resistive::kLogEpsUnflagged, bm,
                 eps);
      }
      if (doffon) {
        Real dc[3];
        for (int a = 0; a < 3; ++a) dc[a] = doff_(m, a, k, j, i);
        plog.Add(resistive::kLogDoff, bm,
                 sqrt(fmax(resistive::Dot(pm.g, dc, dc), 0.0))/sqrt(b2));
      }
    }
    w0_(m, IDN, k, j, i) = prim[PRH]*mb;
    w0_(m, IVX, k, j, i) = prim[PVX];
    w0_(m, IVY, k, j, i) = prim[PVY];
    w0_(m, IVZ, k, j, i) = prim[PVZ];
    w0_(m, IPR, k, j, i) = prim[PPR];
    for (int n = 0; n < nscal; ++n) w0_(m, nmhd + n, k, j, i) = prim[PYF + n];
    temp_(m, 0, k, j, i) = prim[PTM];
    for (int a = 0; a < 3; ++a) {
      bcc_(m, a, k, j, i) = bcc[a];
      emc_(m, resistive::kEX + a, k, j, i) =
          (doffon && !emcsplit) ? E[a] - doff_(m, a, k, j, i) : E[a];
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
  if (resplit) {
    Real h = 0.0;
    Kokkos::deep_copy(h, rde);
    presist->resplit_de += h;
  }
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
template void DynGRMHDPS<EOSPolicy, ErrorPolicy>::ResistiveC2P(bool, int); \
template void DynGRMHDPS<EOSPolicy, ErrorPolicy>::ResistivePrimToCon(int, int, int, int, \
                                                                     int, int);

INSTANTIATE_RESISTIVE(Primitive::IdealGas, Primitive::ResetFloor)
INSTANTIATE_RESISTIVE(Primitive::PiecewisePolytrope, Primitive::ResetFloor)
INSTANTIATE_RESISTIVE(Primitive::EOSCompOSE<Primitive::NormalLogs>, Primitive::ResetFloor)
INSTANTIATE_RESISTIVE(Primitive::EOSCompOSE<Primitive::NQTLogs>, Primitive::ResetFloor)
INSTANTIATE_RESISTIVE(Primitive::EOSHybrid<Primitive::NormalLogs>, Primitive::ResetFloor)
INSTANTIATE_RESISTIVE(Primitive::EOSHybrid<Primitive::NQTLogs>, Primitive::ResetFloor)

}  // namespace dyngr
