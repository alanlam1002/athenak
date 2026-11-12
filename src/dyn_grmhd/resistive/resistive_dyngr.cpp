//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file resistive_dyngr.cpp
//! \brief EOS-independent part of resistive GRMHD on the dyngr path: storage, E~
//!        boundary exchange, edge (2D Riemann) solve for e and h, CT update of E~ and
//!        the ImEx bookkeeping of the stiff Ohmic source (07_scoping_v2 §2.2, §3.3).

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "driver/driver.hpp"
#include "bvals/bvals.hpp"
#include "coordinates/adm.hpp"
#include "mhd/mhd.hpp"
#include "z4c/z4c.hpp"
#include "dyn_grmhd/resistive/resistive_dyngr.hpp"
#include "dyn_grmhd/resistive/resistive_mesh_util.hpp"
#include "dyn_grmhd/resistive/resistive_em.hpp"
#include "dyn_grmhd/resistive/resistive_edge.hpp"
#include "dyn_grmhd/resistive/resistive_probe.hpp"

namespace resistive {

namespace {
void Fatal(const int line, const std::string &msg) {
  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << line << std::endl
            << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

KOKKOS_INLINE_FUNCTION
Real FaceVal(const DvceFaceFld4D<Real> &f, const int d, const int m, const int c[3]) {
  if (d == 0) return f.x1f(m, c[2], c[1], c[0]);
  if (d == 1) return f.x2f(m, c[2], c[1], c[0]);
  return f.x3f(m, c[2], c[1], c[0]);
}
}  // namespace

//----------------------------------------------------------------------------------------
ResistiveDynGR::ResistiveDynGR(MeshBlockPack *pp, ParameterInput *pin) :
    params(pp->pmhd->resist_params),
    cond(pp->pmhd->resist_params),
    e0("resist_e0", 1, 1, 1, 1),
    e1("resist_e1", 1, 1, 1, 1),
    coarse_e0("resist_ce0", 1, 1, 1, 1),
    qflx("resist_qflx", 1, 1, 1, 1),
    hfld("resist_h", 1, 1, 1, 1),
    emc("resist_emc", 1, 1, 1, 1, 1),
    cwork("resist_cwork", 1, 1, 1, 1, 1),
    doff("resist_doff", 1, 1, 1, 1, 1),
    esrc("resist_esrc", 1, 1, 1, 1),
    pbval_e(nullptr), pbval_ei(nullptr), pbval_ep(nullptr), pbval_ep2(nullptr),
    psync{nullptr, nullptr, nullptr, nullptr},
    e0_alias("resist_e0a", 1, 1, 1, 1, 1),
    pmy_pack(pp),
    multilevel(pp->pmesh->multilevel) {
  // Step 1a/2 scope (07 §4): static refinement, static metric, imex2/imex3, ng >= 2
  if (pp->pmesh->adaptive) {
    Fatal(__LINE__, "<mhd> resistive = true: AMR (refine/derefine of E~) is not "
          "supported yet; static mesh refinement is");
  }
  if (multilevel && pin->DoesParameterExist("mesh_refinement", "prolong_primitives") &&
      pin->GetBoolean("mesh_refinement", "prolong_primitives")) {
    Fatal(__LINE__, "<mhd> resistive = true: <mesh_refinement> prolong_primitives "
          "uses the ideal c2p and is not supported");
  }
  if (pp->pz4c != nullptr || pp->padm == nullptr) {
    Fatal(__LINE__, "<mhd> resistive = true: only a fixed <adm> background is supported "
          "(no <z4c>)");
  }
  std::string integ = pin->GetString("time", "integrator");
  if (integ != "imex2" && integ != "imex3") {
    Fatal(__LINE__, "<mhd> resistive = true: integrator '" + integ + "' not supported; "
          "use imex2 or imex3 (imex2+ has zero-weight implicit stages that need an "
          "explicit Ohmic source, see proposals/)");
  }
  auto &indcs = pp->pmesh->mb_indcs;
  if (indcs.ng < 2) {
    Fatal(__LINE__, "<mhd> resistive = true requires <mesh> nghost >= 2");
  }
  if (pin->DoesBlockExist("mhd_srcterms")) {
    Fatal(__LINE__, "<mhd> resistive = true: <mhd_srcterms> is not supported yet");
  }

  int nmb = std::max((pp->nmb_thispack), (pp->pmesh->nmb_maxperrank));
  int n1 = indcs.nx1 + 2*indcs.ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*indcs.ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*indcs.ng) : 1;
  Kokkos::realloc(e0.x1f, nmb, n3, n2, n1+1);
  Kokkos::realloc(e0.x2f, nmb, n3, n2+1, n1);
  Kokkos::realloc(e0.x3f, nmb, n3+1, n2, n1);
  Kokkos::realloc(e1.x1f, nmb, n3, n2, n1+1);
  Kokkos::realloc(e1.x2f, nmb, n3, n2+1, n1);
  Kokkos::realloc(e1.x3f, nmb, n3+1, n2, n1);
  Kokkos::realloc(qflx.x1f, nmb, n3, n2, n1+1);
  Kokkos::realloc(qflx.x2f, nmb, n3, n2+1, n1);
  Kokkos::realloc(qflx.x3f, nmb, n3+1, n2, n1);
  Kokkos::realloc(hfld.x1e, nmb, n3+1, n2+1, n1);
  Kokkos::realloc(hfld.x2e, nmb, n3+1, n2, n1+1);
  Kokkos::realloc(hfld.x3e, nmb, n3, n2+1, n1+1);
  Kokkos::realloc(emc, nmb, kNEMC, n3, n2, n1);
  Kokkos::realloc(cwork, nmb, kNWORK, n3, n2, n1);
  // dE_c = 0 at the pgen (no solve yet); a restart overwrites it
  if (params.Offset()) Kokkos::realloc(doff, nmb, 3, n3, n2, n1);
  probe_cycles = ProbeCycles(pin);
  probe_base = pin->GetString("job", "basename");
  if (probe_cycles > 0) {
    Kokkos::realloc(pst, nmb, kNPST, n3, n2, n1);
    Kokkos::realloc(pedge, nmb, kNPEDGE, n3+1, n2+1, n1+1);
  }
  if (params.freeze_background) {
    if (pp->pmesh->adaptive) {
      std::cout << "### FATAL ERROR: <mhd> resistive_freeze_background is not "
                << "supported with adaptive mesh refinement" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    Kokkos::realloc(esrc.x1f, nmb, n3, n2, n1+1);
    Kokkos::realloc(esrc.x2f, nmb, n3, n2+1, n1);
    Kokkos::realloc(esrc.x3f, nmb, n3+1, n2, n1);
  }

  // K5-run histogram interval: resistive_hist_dt, else the dt of the first hst output
  hist_dt = params.hist_dt;
  for (int n = 1; n < 100 && !(hist_dt > 0.0); ++n) {
    const std::string blk = "output" + std::to_string(n);
    if (pin->DoesBlockExist(blk) && pin->DoesParameterExist(blk, "file_type") &&
        pin->GetString(blk, "file_type") == "hst" &&
        pin->DoesParameterExist(blk, "dt")) {
      hist_dt = pin->GetReal(blk, "dt");
    }
  }
  hist_fname = pin->GetString("job", "basename") + ".rhist.txt";
  rlog_fname = pin->GetString("job", "basename") + ".rlog.txt";
  rfail_fname = pin->GetString("job", "basename") + ".rfail.txt";
  esrc_fname = pin->GetString("job", "basename") + ".esrc.r"
               + std::to_string(global_variable::my_rank);
  hist_next = pp->pmesh->time + hist_dt;

  pbval_e = new MeshBoundaryValuesFC(pp, pin);
  pbval_e->InitializeBuffers(3);
  pbval_ei = new MeshBoundaryValuesFC(pp, pin);
  pbval_ei->InitializeBuffers(3);
  pbval_ep = new MeshBoundaryValuesFC(pp, pin);
  pbval_ep->InitializeBuffers(3);
  pbval_ep2 = new MeshBoundaryValuesFC(pp, pin);
  pbval_ep2->InitializeBuffers(3);

  if (multilevel) {
    int nc1 = indcs.cnx1 + 2*indcs.ng;
    int nc2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*indcs.ng) : 1;
    int nc3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*indcs.ng) : 1;
    Kokkos::realloc(coarse_e0.x1f, nmb, nc3, nc2, nc1+1);
    Kokkos::realloc(coarse_e0.x2f, nmb, nc3, nc2+1, nc1);
    Kokkos::realloc(coarse_e0.x3f, nmb, nc3+1, nc2, nc1);
    // e0 is LayoutRight (nmb,n3,n2,n1[+1]): the same memory read as (nmb,1,n3,n2,n1[+1])
    e0_alias.x1f = DvceArray5D<Real>(e0.x1f.data(), nmb, 1, n3, n2, n1+1);
    e0_alias.x2f = DvceArray5D<Real>(e0.x2f.data(), nmb, 1, n3, n2+1, n1);
    e0_alias.x3f = DvceArray5D<Real>(e0.x3f.data(), nmb, 1, n3+1, n2, n1);
    for (int r = 0; r < kNSync; ++r) {
      psync[r] = new MeshBoundaryValuesCC(pp, pin, false);
      psync[r]->InitializeBuffers(1);
    }
  }
}

ResistiveDynGR::~ResistiveDynGR() {
  delete pbval_e;
  delete pbval_ei;
  delete pbval_ep;
  delete pbval_ep2;
  for (int r = 0; r < kNSync; ++r) delete psync[r];
}

//----------------------------------------------------------------------------------------
//! Coordinate source terms are not added on the resistive path (Step 1a/2), so the ADM
//! metric must be uniform. Checked once at start-up on all cells including ghosts.
bool ResistiveDynGR::Probing() const {
  return probe_cycles > 0 && pmy_pack->pmesh->ncycle < probe_cycles;
}

void ResistiveDynGR::CheckConstantMetric() const {
  if (pmy_pack->padm->is_dynamic) {
    Fatal(__LINE__, "<mhd> resistive = true: a time-dependent ADM metric is not "
          "supported yet (Step 3)");
  }
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int n1 = indcs.nx1 + 2*indcs.ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*indcs.ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*indcs.ng) : 1;
  int nmb = pmy_pack->nmb_thispack;
  auto adm = pmy_pack->padm->adm;
  Real maxdev = 0.0;
  Kokkos::parallel_reduce("resist_metric_check",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*n3*n2*n1),
  KOKKOS_LAMBDA(const int idx, Real &md) {
    int m = idx/(n3*n2*n1);
    int r = idx - m*(n3*n2*n1);
    int k = r/(n2*n1);
    r -= k*(n2*n1);
    int j = r/n1;
    int i = r - j*n1;
    PointMetric p, q;
    LoadMetric(adm, m, k, j, i, p);
    LoadMetric(adm, 0, 0, 0, 0, q);
    Real d = fabs(p.alpha - q.alpha);
    for (int n = 0; n < NSPMETRIC; ++n) d = fmax(d, fabs(p.g[n] - q.g[n]));
    for (int a = 0; a < 3; ++a) d = fmax(d, fabs(p.beta[a] - q.beta[a]));
    md = fmax(md, d);
  }, Kokkos::Max<Real>(maxdev));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &maxdev, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  if (maxdev > 0.0) {
    Fatal(__LINE__, "<mhd> resistive = true: the ADM metric must be spatially uniform "
          "(coordinate source terms are not implemented on the resistive path yet)");
  }
}

//----------------------------------------------------------------------------------------
// Boundary communication tasks. pbval_e carries E~ after the explicit update and the
// same-level sync of h*, pbval_ei the post-implicit E~, pbval_ep/ep2 the two stage-1
// pre-implicit solves.

TaskStatus ResistiveDynGR::InitRecv(Driver *d, int stage) {
  TaskStatus ts = pbval_e->InitRecv(3);
  if (ts != TaskStatus::complete) return ts;
  ts = pbval_e->InitFluxRecv(3);
  if (ts != TaskStatus::complete) return ts;
  ts = pbval_ei->InitRecv(3);
  if (ts != TaskStatus::complete) return ts;
  if (stage == 1) {
    ts = pbval_ep->InitRecv(3);
    if (ts != TaskStatus::complete) return ts;
    ts = pbval_ep2->InitRecv(3);
    if (ts != TaskStatus::complete) return ts;
  }
  if (multilevel) {
    for (int r = (stage == 1)? 0 : kSyncE; r < kNSync; ++r) {
      ts = psync[r]->InitFluxRecv(1);
      if (ts != TaskStatus::complete) return ts;
    }
  }
  return ts;
}

TaskStatus ResistiveDynGR::ClearSend(Driver *d, int stage) {
  TaskStatus ts = pbval_e->ClearSend();
  if (ts != TaskStatus::complete) return ts;
  ts = pbval_e->ClearFluxSend();
  if (ts != TaskStatus::complete) return ts;
  ts = pbval_ei->ClearSend();
  if (ts != TaskStatus::complete) return ts;
  if (stage == 1) {
    ts = pbval_ep->ClearSend();
    if (ts != TaskStatus::complete) return ts;
    ts = pbval_ep2->ClearSend();
    if (ts != TaskStatus::complete) return ts;
  }
  if (multilevel) {
    for (int r = (stage == 1)? 0 : kSyncE; r < kNSync; ++r) {
      ts = psync[r]->ClearFluxSend();
      if (ts != TaskStatus::complete) return ts;
    }
  }
  return ts;
}

TaskStatus ResistiveDynGR::ClearRecv(Driver *d, int stage) {
  TaskStatus ts = pbval_e->ClearRecv();
  if (ts != TaskStatus::complete) return ts;
  ts = pbval_e->ClearFluxRecv();
  if (ts != TaskStatus::complete) return ts;
  ts = pbval_ei->ClearRecv();
  if (ts != TaskStatus::complete) return ts;
  if (stage == 1) {
    ts = pbval_ep->ClearRecv();
    if (ts != TaskStatus::complete) return ts;
    ts = pbval_ep2->ClearRecv();
    if (ts != TaskStatus::complete) return ts;
  }
  if (multilevel) {
    for (int r = (stage == 1)? 0 : kSyncE; r < kNSync; ++r) {
      ts = psync[r]->ClearFluxRecv();
      if (ts != TaskStatus::complete) return ts;
    }
  }
  return ts;
}

TaskStatus ResistiveDynGR::CopyE(Driver *d, int stage) {
  if (stage == 1) {
    Kokkos::deep_copy(DevExeSpace(), e1.x1f, e0.x1f);
    Kokkos::deep_copy(DevExeSpace(), e1.x2f, e0.x2f);
    Kokkos::deep_copy(DevExeSpace(), e1.x3f, e0.x3f);
  }
  return TaskStatus::complete;
}

TaskStatus ResistiveDynGR::SendH(Driver *d, int stage) {
  return pbval_e->PackAndSendFluxFC(hfld);
}
TaskStatus ResistiveDynGR::RecvH(Driver *d, int stage) {
  return pbval_e->RecvAndUnpackFluxFC(hfld);
}
TaskStatus ResistiveDynGR::SendE(Driver *d, int stage) {
  RestrictE();
  return pbval_e->PackAndSendFC(e0, coarse_e0);
}
TaskStatus ResistiveDynGR::RecvE(Driver *d, int stage) {
  return pbval_e->RecvAndUnpackFC(e0, coarse_e0);
}
TaskStatus ResistiveDynGR::SendEI(Driver *d, int stage) {
  RestrictE();
  return pbval_ei->PackAndSendFC(e0, coarse_e0);
}
TaskStatus ResistiveDynGR::RecvEI(Driver *d, int stage) {
  return pbval_ei->RecvAndUnpackFC(e0, coarse_e0);
}
TaskStatus ResistiveDynGR::SendEP(Driver *d, int stage) {
  if (stage != 1) return TaskStatus::complete;
  RestrictE();
  return pbval_ep->PackAndSendFC(e0, coarse_e0);
}
TaskStatus ResistiveDynGR::RecvEP(Driver *d, int stage) {
  if (stage != 1) return TaskStatus::complete;
  return pbval_ep->RecvAndUnpackFC(e0, coarse_e0);
}
TaskStatus ResistiveDynGR::SendEP2(Driver *d, int stage) {
  if (stage != 1) return TaskStatus::complete;
  RestrictE();
  return pbval_ep2->PackAndSendFC(e0, coarse_e0);
}
TaskStatus ResistiveDynGR::RecvEP2(Driver *d, int stage) {
  if (stage != 1) return TaskStatus::complete;
  return pbval_ep2->RecvAndUnpackFC(e0, coarse_e0);
}

// SMR: fine->coarse sync of the shared faces after each E~ update (before the ghost
// exchange). The CC flux-correction routines overwrite the coarse face with the area
// average of the finer faces, which is the required operation (07 D3).
TaskStatus ResistiveDynGR::SyncSend(int r) {
  if (!multilevel) return TaskStatus::complete;
  return psync[r]->PackAndSendFluxCC(e0_alias);
}
TaskStatus ResistiveDynGR::SyncRecv(int r) {
  if (!multilevel) return TaskStatus::complete;
  return psync[r]->RecvAndUnpackFluxCC(e0_alias);
}
TaskStatus ResistiveDynGR::SyncSendEP(Driver *d, int stage) {
  return (stage == 1)? SyncSend(kSyncEP) : TaskStatus::complete;
}
TaskStatus ResistiveDynGR::SyncRecvEP(Driver *d, int stage) {
  return (stage == 1)? SyncRecv(kSyncEP) : TaskStatus::complete;
}
TaskStatus ResistiveDynGR::SyncSendEP2(Driver *d, int stage) {
  return (stage == 1)? SyncSend(kSyncEP2) : TaskStatus::complete;
}
TaskStatus ResistiveDynGR::SyncRecvEP2(Driver *d, int stage) {
  return (stage == 1)? SyncRecv(kSyncEP2) : TaskStatus::complete;
}
TaskStatus ResistiveDynGR::SyncSendE(Driver *d, int stage) { return SyncSend(kSyncE); }
TaskStatus ResistiveDynGR::SyncRecvE(Driver *d, int stage) { return SyncRecv(kSyncE); }
TaskStatus ResistiveDynGR::SyncSendEI(Driver *d, int stage) { return SyncSend(kSyncEI); }
TaskStatus ResistiveDynGR::SyncRecvEI(Driver *d, int stage) { return SyncRecv(kSyncEI); }

void ResistiveDynGR::RestrictE() {
  if (multilevel) pmy_pack->pmesh->pmr->RestrictFC(e0, coarse_e0);
}

// Fill fine ghost faces at fine/coarse boundaries from the received coarse E~ (the B~
// operators: every fine cell gets the coarse divergence, i.e. q~ piecewise constant).
void ResistiveDynGR::ProlongE() {
  if (!multilevel) return;
  pbval_e->FillCoarseInBndryFC(e0, coarse_e0);
  if (!(pmy_pack->pmesh->strictly_periodic)) {
    MeshBoundaryValues::BFieldBCsCoarse(pmy_pack, pbval_e->b_in, coarse_e0);
  }
  pbval_e->ProlongateFC(e0, coarse_e0);
}

// Physical BCs for E~ use the face-field helpers (outflow copies faces; periodic is
// handled by the exchange). [VERIFY] reflecting walls flip the normal component, which
// is the conducting-wall condition for B but not for E; only periodic/outflow are tested.
TaskStatus ResistiveDynGR::ApplyBCsE(Driver *d, int stage) {
  ProlongE();
  if (pmy_pack->pmesh->strictly_periodic) return TaskStatus::complete;
  MeshBoundaryValues::BFieldBCs(pmy_pack, pbval_e->b_in, e0);
  return TaskStatus::complete;
}
TaskStatus ResistiveDynGR::ApplyBCsEP(Driver *d, int stage) {
  if (stage != 1) return TaskStatus::complete;
  return ApplyBCsE(d, stage);
}

void ResistiveDynGR::ExchangeEBlocking() {
  if (multilevel) {
    psync[kSyncE]->InitFluxRecv(1);
    psync[kSyncE]->PackAndSendFluxCC(e0_alias);
    TaskStatus ts;
    do {
      ts = psync[kSyncE]->RecvAndUnpackFluxCC(e0_alias);
    } while (ts != TaskStatus::complete);
    psync[kSyncE]->ClearFluxSend();
    psync[kSyncE]->ClearFluxRecv();
  }
  RestrictE();
  pbval_e->InitRecv(3);
  pbval_e->PackAndSendFC(e0, coarse_e0);
  pbval_e->ClearSend();
  pbval_e->ClearRecv();
  TaskStatus ts = pbval_e->RecvAndUnpackFC(e0, coarse_e0);
  if (ts != TaskStatus::complete) {
    Fatal(__LINE__, "resistive: blocking E~ exchange did not complete");
  }
  ApplyBCsE(nullptr, 0);
}

//----------------------------------------------------------------------------------------
//! EdgeSolve: upwind edge values of e (Faraday, into pmhd->efld) and h (Ampere, into
//! hfld) from four PLM-reconstructed corner states of the cell E, B (07 §2.2).
//! Inactive directions use the same cell with zero slope and zero normal-field jumps.

TaskStatus ResistiveDynGR::EdgeSolve(Driver *d, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  const int act0 = 1;
  const int act1 = pmy_pack->pmesh->multi_d ? 1 : 0;
  const int act2 = pmy_pack->pmesh->three_d ? 1 : 0;
  const int lo0 = is, lo1 = js, lo2 = ks;
  auto emc_ = emc;
  auto adm = pmy_pack->padm->adm;
  DvceFaceFld4D<Real> bf = pmy_pack->pmhd->b0;
  DvceFaceFld4D<Real> ef = e0;
  auto efld = pmy_pack->pmhd->efld;
  auto hf = hfld;
  const bool probe = Probing();
  auto pe = pedge;

  for (int kk = 0; kk < 3; ++kk) {
    int il = is, iu = ie+1, jl = js, ju = je+1, kl = ks, ku = ke+1;
    if (kk == 0) iu = ie;
    if (kk == 1) ju = je;
    if (kk == 2) ku = ke;
    par_for("resist_edge", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const int act[3] = {act0, act1, act2};
      const int lo[3] = {lo0, lo1, lo2};
      const int a = (kk + 1)%3, b = (kk + 2)%3;
      const int idx[3] = {i, j, k};

      // metric: average of the (up to) four cells sharing the edge
      PointMetric pm;
      ZeroMetric(pm);
      Real E4[2][2][3], B4[2][2][3], E40[2][2][3], B40[2][2][3];
      for (int sa = 0; sa < 2; ++sa) {
        for (int sb = 0; sb < 2; ++sb) {
          int c[3];
          c[kk] = idx[kk];
          c[a] = act[a] ? idx[a] - 1 + sa : lo[a];
          c[b] = act[b] ? idx[b] - 1 + sb : lo[b];
          AddMetric(adm, m, c[2], c[1], c[0], 0.25, pm);
          const Real wa = (sa == 0) ? 0.5 : -0.5;
          const Real wb = (sb == 0) ? 0.5 : -0.5;
          for (int n = 0; n < kNEMC; ++n) {
            const Real q = emc_(m, n, c[2], c[1], c[0]);
            Real v = q;
            if (act[a]) {
              int cm[3] = {c[0], c[1], c[2]}, cp[3] = {c[0], c[1], c[2]};
              cm[a] -= 1; cp[a] += 1;
              v += wa*VanLeerSlope(q - emc_(m, n, cm[2], cm[1], cm[0]),
                                   emc_(m, n, cp[2], cp[1], cp[0]) - q);
            }
            if (act[b]) {
              int cm[3] = {c[0], c[1], c[2]}, cp[3] = {c[0], c[1], c[2]};
              cm[b] -= 1; cp[b] += 1;
              v += wb*VanLeerSlope(q - emc_(m, n, cm[2], cm[1], cm[0]),
                                   emc_(m, n, cp[2], cp[1], cp[0]) - q);
            }
            if (n < 3) {
              E4[sa][sb][n] = v;
              E40[sa][sb][n] = q;
            } else {
              B4[sa][sb][n-3] = v;
              B40[sa][sb][n-3] = q;
            }
          }
        }
      }
      const Real sg = SqrtDet(pm.g);
      Real gu[NSPMETRIC];
      MetricInverse(pm.g, gu);
      Real fe[2][2], fh[2][2];
      for (int sa = 0; sa < 2; ++sa) {
        for (int sb = 0; sb < 2; ++sb) {
          Real Ed[3], Bd[3];
          Lower(pm.g, E4[sa][sb], Ed);
          Lower(pm.g, B4[sa][sb], Bd);
          fe[sa][sb] = EdgeEMF(pm.alpha, pm.beta[a], pm.beta[b], Ed[kk],
                               sg*B4[sa][sb][a], sg*B4[sa][sb][b]);
          fh[sa][sb] = EdgeAmpere(pm.alpha, pm.beta[a], pm.beta[b], Bd[kk],
                                  sg*E4[sa][sb][a], sg*E4[sa][sb][b]);
        }
      }

      // normal-field jumps: a-face field across b, b-face field across a
      Real jb_a = 0.0, jb_b = 0.0, je_a = 0.0, je_b = 0.0;
      if (act[b]) {
        int cl[3] = {idx[0], idx[1], idx[2]}, cr[3] = {idx[0], idx[1], idx[2]};
        cl[b] = idx[b] - 1;
        int cll[3] = {cl[0], cl[1], cl[2]}, crr[3] = {cr[0], cr[1], cr[2]};
        cll[b] -= 1; crr[b] += 1;
        Real ul = FaceVal(bf, a, m, cl), ur = FaceVal(bf, a, m, cr);
        Real sl = VanLeerSlope(ul - FaceVal(bf, a, m, cll), ur - ul);
        Real sr = VanLeerSlope(ur - ul, FaceVal(bf, a, m, crr) - ur);
        jb_a = (ur - 0.5*sr) - (ul + 0.5*sl);
        ul = FaceVal(ef, a, m, cl); ur = FaceVal(ef, a, m, cr);
        sl = VanLeerSlope(ul - FaceVal(ef, a, m, cll), ur - ul);
        sr = VanLeerSlope(ur - ul, FaceVal(ef, a, m, crr) - ur);
        je_a = (ur - 0.5*sr) - (ul + 0.5*sl);
      }
      if (act[a]) {
        int cl[3] = {idx[0], idx[1], idx[2]}, cr[3] = {idx[0], idx[1], idx[2]};
        cl[a] = idx[a] - 1;
        int cll[3] = {cl[0], cl[1], cl[2]}, crr[3] = {cr[0], cr[1], cr[2]};
        cll[a] -= 1; crr[a] += 1;
        Real ul = FaceVal(bf, b, m, cl), ur = FaceVal(bf, b, m, cr);
        Real sl = VanLeerSlope(ul - FaceVal(bf, b, m, cll), ur - ul);
        Real sr = VanLeerSlope(ur - ul, FaceVal(bf, b, m, crr) - ur);
        jb_b = (ur - 0.5*sr) - (ul + 0.5*sl);
        ul = FaceVal(ef, b, m, cl); ur = FaceVal(ef, b, m, cr);
        sl = VanLeerSlope(ul - FaceVal(ef, b, m, cll), ur - ul);
        sr = VanLeerSlope(ur - ul, FaceVal(ef, b, m, crr) - ur);
        je_b = (ur - 0.5*sr) - (ul + 0.5*sl);
      }
      Real aap, aam, abp, abm;
      LightSpeeds(pm.alpha, pm.beta[a], InverseDiag(gu, a), aap, aam);
      LightSpeeds(pm.alpha, pm.beta[b], InverseDiag(gu, b), abp, abm);
      const Real ev = EdgeUpwind(fe, jb_a, jb_b, aap, aam, abp, abm);
      const Real hv = EdgeUpwind(fh, je_a, je_b, aap, aam, abp, abm);
      if (probe) {
        // 07 v2.9 §3.2.4(e) item 6: unsloped average, sloped average, full value
        Real fe0[2][2];
        for (int sa = 0; sa < 2; ++sa) {
          for (int sb = 0; sb < 2; ++sb) {
            Real Ed[3];
            Lower(pm.g, E40[sa][sb], Ed);
            fe0[sa][sb] = EdgeEMF(pm.alpha, pm.beta[a], pm.beta[b], Ed[kk],
                                  sg*B40[sa][sb][a], sg*B40[sa][sb][b]);
          }
        }
        pe(m, 3*kk + kPEdgeAvg, k, j, i) = EdgeUpwind(fe0, 0.0, 0.0, aap, aam, abp, abm);
        pe(m, 3*kk + kPEdgeSlope, k, j, i) = EdgeUpwind(fe, 0.0, 0.0, aap, aam, abp, abm);
        pe(m, 3*kk + kPEdgeFull, k, j, i) = ev;
      }
      if (kk == 0) {
        efld.x1e(m, k, j, i) = ev;
        hf.x1e(m, k, j, i) = hv;
      } else if (kk == 1) {
        efld.x2e(m, k, j, i) = ev;
        hf.x2e(m, k, j, i) = hv;
      } else {
        efld.x3e(m, k, j, i) = ev;
        hf.x3e(m, k, j, i) = hv;
      }
    });
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! CTE: E~ = gam0 E~ + gam1 E~(n) - beta dt [curl h* + qflx]  (same stencil and sign as
//! MHD::CT with e -> h). All three components are updated in every dimensionality
//! because the convective charge current drives E~^1 even in 1D.

TaskStatus ResistiveDynGR::CTE(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  Real gam0 = pdriver->gam0[stage-1];
  Real gam1 = pdriver->gam1[stage-1];
  Real beta_dt = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);
  bool multi_d = pmy_pack->pmesh->multi_d;
  bool three_d = pmy_pack->pmesh->three_d;
  auto h1 = hfld.x1e;
  auto h2 = hfld.x2e;
  auto h3 = hfld.x3e;
  auto &mbsize = pmy_pack->pmb->mb_size;

  // freeze_background: record E~_pre = gam0 E~ + gam1 E~(n) before the first update
  const bool frz = params.freeze_background;
  if (frz && !esrc_set && (pmy_pack->pmesh->ncycle != 0 || stage != 1)) {
    FreezeFile(false);  // restart: esrc recorded by the cycle-0 run
  }
  const bool capture = frz && !esrc_set;
  if (capture) FreezeBackground(gam0, gam1, beta_dt, 0);

  auto ex1 = e0.x1f, ex1o = e1.x1f, q1 = qflx.x1f;
  par_for("CTE-e1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real v = gam0*ex1(m,k,j,i) + gam1*ex1o(m,k,j,i);
    if (multi_d) {
      v -= beta_dt*(h3(m,k,j+1,i) - h3(m,k,j,i))/mbsize.d_view(m).dx2;
    }
    if (three_d) {
      v += beta_dt*(h2(m,k+1,j,i) - h2(m,k,j,i))/mbsize.d_view(m).dx3;
    }
    ex1(m,k,j,i) = v - beta_dt*q1(m,k,j,i);
  });

  auto ex2 = e0.x2f, ex2o = e1.x2f, q2 = qflx.x2f;
  par_for("CTE-e2", DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real v = gam0*ex2(m,k,j,i) + gam1*ex2o(m,k,j,i);
    v += beta_dt*(h3(m,k,j,i+1) - h3(m,k,j,i))/mbsize.d_view(m).dx1;
    if (three_d) {
      v -= beta_dt*(h1(m,k+1,j,i) - h1(m,k,j,i))/mbsize.d_view(m).dx3;
    }
    ex2(m,k,j,i) = v - beta_dt*q2(m,k,j,i);
  });

  auto ex3 = e0.x3f, ex3o = e1.x3f, q3 = qflx.x3f;
  par_for("CTE-e3", DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real v = gam0*ex3(m,k,j,i) + gam1*ex3o(m,k,j,i);
    v -= beta_dt*(h2(m,k,j,i+1) - h2(m,k,j,i))/mbsize.d_view(m).dx1;
    if (multi_d) {
      v += beta_dt*(h1(m,k,j+1,i) - h1(m,k,j,i))/mbsize.d_view(m).dx2;
    }
    ex3(m,k,j,i) = v - beta_dt*q3(m,k,j,i);
  });
  if (frz) {
    if (capture) {
      FreezeBackground(gam0, gam1, beta_dt, 1);
      FreezeFile(true);
    }
    FreezeBackground(gam0, gam1, beta_dt, 2);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! resistive_freeze_background (test only): eta-compensation of the initial state. The
//! discrete explicit E~ RHS of the initial data, -[curl h* + qflx](t=0), is recorded once
//! and subtracted in every later stage, so that the initial equilibrium is an exact
//! discrete steady state (its resistive and numerical diffusion are cancelled) while
//! perturbations evolve normally. mode 0: esrc = E~_pre; 1: esrc = -(E~ - esrc)/(beta
//! dt); 2: E~ += beta dt esrc.

void ResistiveDynGR::FreezeBackground(Real gam0, Real gam1, Real beta_dt, int mode) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  for (int d = 0; d < 3; ++d) {
    auto ex = (d == 0) ? e0.x1f : ((d == 1) ? e0.x2f : e0.x3f);
    auto eo = (d == 0) ? e1.x1f : ((d == 1) ? e1.x2f : e1.x3f);
    auto es = (d == 0) ? esrc.x1f : ((d == 1) ? esrc.x2f : esrc.x3f);
    par_for("resist_freeze", DevExeSpace(), 0, nmb1, ks, ke + (d == 2), js,
            je + (d == 1), is, ie + (d == 0),
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      if (mode == 0) {
        es(m,k,j,i) = gam0*ex(m,k,j,i) + gam1*eo(m,k,j,i);
      } else if (mode == 1) {
        es(m,k,j,i) = -(ex(m,k,j,i) - es(m,k,j,i))/beta_dt;
      } else {
        ex(m,k,j,i) += beta_dt*es(m,k,j,i);
      }
    });
  }
  if (mode == 1) esrc_set = true;
}

//----------------------------------------------------------------------------------------
//! esrc sidecar, one file per rank: magic, nmb, the MeshBlock gids, then x1f/x2f/x3f raw.
//! A restart must use the same MeshBlock decomposition and rank count as the cycle-0 run
//! that wrote it (checked).

void ResistiveDynGR::FreezeFile(bool write) {
  const int magic = 0x45535243;
  int nmb = pmy_pack->nmb_thispack;
  auto &gid = pmy_pack->pmb->mb_gid.h_view;
  auto fail = [&](const char *msg) {
    std::cout << "### FATAL ERROR: <mhd> resistive_freeze_background " << msg << " '"
              << esrc_fname << "' (a restart needs the sidecar written by the cycle-0 "
              << "run with the same decomposition)" << std::endl;
    std::exit(EXIT_FAILURE);
  };
  FILE *fp = std::fopen(esrc_fname.c_str(), write ? "wb" : "rb");
  if (fp == nullptr) fail(write ? "cannot write" : "cannot read");
  bool ok = true;
  if (write) {
    ok = std::fwrite(&magic, sizeof(int), 1, fp) == 1 &&
         std::fwrite(&nmb, sizeof(int), 1, fp) == 1;
    for (int m = 0; m < nmb && ok; ++m) {
      int g = gid(m);
      ok = std::fwrite(&g, sizeof(int), 1, fp) == 1;
    }
  } else {
    int hdr[2] = {0, 0};
    ok = std::fread(hdr, sizeof(int), 2, fp) == 2 && hdr[0] == magic && hdr[1] == nmb;
    for (int m = 0; m < nmb && ok; ++m) {
      int g = -1;
      ok = std::fread(&g, sizeof(int), 1, fp) == 1 && g == gid(m);
    }
  }
  for (int d = 0; d < 3 && ok; ++d) {
    auto es = (d == 0) ? esrc.x1f : ((d == 1) ? esrc.x2f : esrc.x3f);
    auto h = Kokkos::create_mirror_view(es);
    size_t n = h.size();
    if (write) {
      Kokkos::deep_copy(h, es);
      ok = std::fwrite(h.data(), sizeof(Real), n, fp) == n;
    } else {
      ok = std::fread(h.data(), sizeof(Real), n, fp) == n;
      if (ok) Kokkos::deep_copy(es, h);
    }
  }
  if (!write && ok && std::fgetc(fp) != EOF) ok = false;
  std::fclose(fp);
  if (!ok) fail(write ? "write error on" : "mismatched or truncated");
  if (!write) esrc_set = true;
}

//----------------------------------------------------------------------------------------
//! E~ += sum_{s <= istage-2} a_twid[istage-2][s] dt R_s on all faces (IonNeutral form).

void ResistiveDynGR::AddImexSources(Driver *pdriver, int istage) {
  if (istage <= 1) return;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int n1 = indcs.nx1 + 2*indcs.ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*indcs.ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*indcs.ng) : 1;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  Real adt[4] = {0.0, 0.0, 0.0, 0.0};
  for (int s = 0; s <= istage-2; ++s) {
    adt[s] = pdriver->a_twid[istage-2][s]*(pmy_pack->pmesh->dt);
  }
  const int ns = istage - 1;
  auto rs = pdriver->impl_src_e;
  auto ex1 = e0.x1f, ex2 = e0.x2f, ex3 = e0.x3f;
  par_for("resist_imex_add", DevExeSpace(), 0, nmb1, 0, n3, 0, n2, 0, n1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    for (int s = 0; s < ns; ++s) {
      const Real w = adt[s];
      if (w == 0.0) continue;
      if (k < n3 && j < n2) ex1(m,k,j,i) += w*rs(s,m,0,k,j,i);
      if (k < n3 && i < n1) ex2(m,k,j,i) += w*rs(s,m,1,k,j,i);
      if (j < n2 && i < n1) ex3(m,k,j,i) += w*rs(s,m,2,k,j,i);
    }
  });
}

//----------------------------------------------------------------------------------------
//! K5-run histogram output: per binning (sigM = B^2/rho, magn = b^2/(rho h)), one row
//! per decade with the counts accumulated since the previous row (cells of the implicit
//! solve, converged, C-retry, fallback (any non-Newton exit), pre-step inadmissible,
//! sigma_M-ceiling activations, floored, E^2 > B^2, converged through the sigma dt
//! continuation; 07 v2.8 §3.2.3(e): FA accepted/rejected, blend, pure IL, IL floored,
//! keep). With FA/blend or d4 keep on (FA + blend is the default since 07 v2.9), also
//! the <basename>.rlog.txt histograms (ProtoLog).

void ResistiveDynGR::MaybeWriteHist(const Real t_end, const int cycle) {
  if (!(hist_dt > 0.0) || t_end < hist_next*(1.0 - 1.0e-12)) return;
  while (hist_next <= t_end*(1.0 + 1.0e-12)) hist_next += hist_dt;
  int64_t h[kHistBinnings][kHistBins][kHNCounters];
  hist.Read(h);
  hist.Reset();
  if (params.Prototype()) MaybeWriteRLog(t_end, cycle);
  if (global_variable::my_rank != 0) return;
  // a new run overwrites, a restart appends (header only if the file is new)
  FILE *fp = std::fopen(hist_fname.c_str(), hist_header ? "a" : "w");
  if (fp == nullptr) return;
  std::fseek(fp, 0, SEEK_END);
  if (std::ftell(fp) == 0) {
    std::fprintf(fp, "# binning time cycle bin log10_lo cells converged retry fallback "
                     "prestep_inadmissible ceiling floor e_gt_b continued fa fa_reject "
                     "blend il il_floor keep\n");
  }
  hist_header = true;
  const char *label[kHistBinnings] = {"sigM", "magn"};
  for (int s = 0; s < kHistBinnings; ++s) {
    for (int b = 0; b < kHistBins; ++b) {
      std::fprintf(fp, "%s %.10e %d %d %d", label[s], t_end, cycle, b,
                   (b == 0) ? -99 : b - 3);
      for (int c = 0; c < kHNCounters; ++c) std::fprintf(fp, " %" PRId64, h[s][b][c]);
      std::fprintf(fp, "\n");
    }
  }
  std::fclose(fp);
}

//! rlog rows: "set time cycle magn_bin max", then the kLogBins counts (bin 0: < 1e-12,
//! bin 1 + q: [1e-12 10^(q/4), 1e-12 10^((q+1)/4)), last: >= 1e2 or NaN); max is the
//! exact maximum of the set over all magn bins in the interval.
void ResistiveDynGR::MaybeWriteRLog(const Real t_end, const int cycle) {
  int64_t h[kNLogSets][kHistBins][kLogBins];
  Real mx[kNLogSets][kHistBins];
  plog.Read(h, mx);
  plog.Reset();
  Real rde = resplit_de;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &rde, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (global_variable::my_rank != 0) return;
  FILE *fp = std::fopen(rlog_fname.c_str(), rlog_header ? "a" : "w");
  if (fp == nullptr) return;
  std::fseek(fp, 0, SEEK_END);
  if (std::ftell(fp) == 0) {
    std::fprintf(fp, "# set time cycle magn_bin max counts[%d] (bin 0 < 1e-12, bin 1+q "
                     "from 1e-12*10^(q/4), bin %d >= 1e2 or NaN)\n", kLogBins,
                 kLogBins - 1);
  }
  rlog_header = true;
  const char *label[kNLogSets] = {"eps_d4", "eps_d4_flagged", "fa_deficit", "doff"};
  const int nsets = params.Offset() ? kNLogSets : kLogDoff;
  for (int s = 0; s < nsets; ++s) {
    for (int b = 0; b < kHistBins; ++b) {
      std::fprintf(fp, "%s %.10e %d %d %.6e", label[s], t_end, cycle, b, mx[s][b]);
      for (int c = 0; c < kLogBins; ++c) std::fprintf(fp, " %" PRId64, h[s][b][c]);
      std::fprintf(fp, "\n");
    }
  }
  // resistive_d4 = resplit: cumulative volume-integrated EM energy change (since the
  // start of this run segment)
  if (params.d4 == D4Mode::resplit) {
    std::fprintf(fp, "resplit_de %.10e %d -1 %.10e\n", t_end, cycle, rde);
  }
  std::fclose(fp);
}

//----------------------------------------------------------------------------------------
//! rfail rows: "cycle stage reason rbin count" (nonzero counts only; cycle 1-based, stage
//! = ImEx istage, reason = FailReason, rbin = floor(r/dx) from rfail_center or 0).

void ResistiveDynGR::WriteRFail(const int cycle, const int stage) {
  std::vector<int64_t> c(kRFailReasons*kRFailRBins);
  rfail.Read(c.data());
  rfail.Reset();
  if (global_variable::my_rank != 0) return;
  FILE *fp = std::fopen(rfail_fname.c_str(), rfail_header ? "a" : "w");
  if (fp == nullptr) return;
  std::fseek(fp, 0, SEEK_END);
  if (std::ftell(fp) == 0) {
    std::fprintf(fp, "# cycle stage reason rbin count (reason: 1 prestep_inadmissible "
                     "2 all_floored 3 c2p_error 4 no_descent 5 singular_jacobian 6 maxit "
                     "7 continuation_stalled; rbin: r/dx from (%g, %g, %g)%s)\n",
                 rfail_center[0], rfail_center[1], rfail_center[2],
                 rfail_center_set ? "" : ", no centre: single bin 0");
  }
  rfail_header = true;
  for (int r = 0; r < kRFailReasons; ++r) {
    for (int b = 0; b < kRFailRBins; ++b) {
      const int64_t n = c[r*kRFailRBins + b];
      if (n > 0) std::fprintf(fp, "%d %d %d %d %" PRId64 "\n", cycle, stage, r, b, n);
    }
  }
  std::fclose(fp);
}

//----------------------------------------------------------------------------------------
//! Restart state: hist_dt, hist_next, Newton counters, pending histogram counts.

std::size_t ResistiveDynGR::CheckRestartHeader(const char *hdr) {
  uint32_t ver, size;
  std::memcpy(&ver, hdr + 8, sizeof(uint32_t));
  std::memcpy(&size, hdr + 8 + sizeof(uint32_t), sizeof(uint32_t));
  if (std::memcmp(hdr, kRestartMagic, 8) != 0) {
    Fatal(__LINE__, "resistive restart block has no format tag: the restart file was "
          "written by a binary older than the R1 tag (07 v2.6) or without "
          "<mhd> resistive = true; it cannot be read by this binary.");
  }
  if ((ver != 2 && ver != kRestartVersion) || size != RestartStateSize(ver)) {
    Fatal(__LINE__, "resistive restart block format mismatch: file version " +
          std::to_string(ver) + " (" + std::to_string(size) + " bytes), this binary "
          "reads version 2 (" + std::to_string(RestartStateSize(2)) + " bytes) or " +
          std::to_string(kRestartVersion) + " (" +
          std::to_string(RestartStateSize()) + " bytes).");
  }
  return size;
}

void ResistiveDynGR::PackRestartState(char *buf) const {
  int64_t ns[kNewtonNCounters];
  int64_t h[kHistBinnings][kHistBins][kHNCounters];
  newton.Read(ns);
  hist.Read(h);
  const uint32_t ver = kRestartVersion;
  const uint32_t size = static_cast<uint32_t>(RestartStateSize());
  std::memcpy(buf, kRestartMagic, 8);
  std::memcpy(buf + 8, &ver, sizeof(uint32_t));
  std::memcpy(buf + 8 + sizeof(uint32_t), &size, sizeof(uint32_t));
  buf += RestartHeaderSize();
  std::size_t o = 0;
  std::memcpy(buf + o, &hist_dt, sizeof(Real)); o += sizeof(Real);
  std::memcpy(buf + o, &hist_next, sizeof(Real)); o += sizeof(Real);
  std::memcpy(buf + o, ns, sizeof(ns)); o += sizeof(ns);
  std::memcpy(buf + o, h, sizeof(h)); o += sizeof(h);
  const uint32_t has_doff[2] = {params.Offset() ? 1u : 0u, 0u};
  std::memcpy(buf + o, has_doff, sizeof(has_doff));
}

void ResistiveDynGR::UnpackRestartState(const char *buf, const std::size_t size) {
  int64_t ns[kNewtonNCounters];
  int64_t h[kHistBinnings][kHistBins][kHNCounters];
  Real dt_rst, next_rst;
  std::size_t o = 0;
  std::memcpy(&dt_rst, buf + o, sizeof(Real)); o += sizeof(Real);
  std::memcpy(&next_rst, buf + o, sizeof(Real)); o += sizeof(Real);
  std::memcpy(ns, buf + o, sizeof(ns)); o += sizeof(ns);
  std::memcpy(h, buf + o, sizeof(h)); o += sizeof(h);
  uint32_t has_doff = 0;
  if (size >= o + 2*sizeof(uint32_t)) std::memcpy(&has_doff, buf + o, sizeof(uint32_t));
  if ((has_doff != 0) != params.Offset()) {
    Fatal(__LINE__, std::string("resistive restart: the file ") +
          (has_doff ? "holds" : "has no") + " persistent cell offset dE_c but this run "
          "has <mhd> resistive_d4 " + (params.Offset() ? "= offset|offset_emc" :
          "!= offset|offset_emc") + " (07 v2.10 §3.2.5(e)); restart with the key of "
          "the run that wrote the file.");
  }
  newton.Load(ns);
  hist.Load(h);
  if (dt_rst == hist_dt) hist_next = next_rst;
  hist_header = true;  // continue the run's rhist file (and rlog/rfail, if present)
  rlog_header = true;
  rfail_header = true;
}

}  // namespace resistive
