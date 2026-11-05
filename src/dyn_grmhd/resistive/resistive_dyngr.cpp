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
#include <cstdlib>
#include <iostream>
#include <string>

#include "athena.hpp"
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
      Real E4[2][2][3], B4[2][2][3];
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
            } else {
              B4[sa][sb][n-3] = v;
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
  const bool capture = frz && !esrc_set;
  if (capture) {
    if (pmy_pack->pmesh->ncycle != 0 || stage != 1) {
      std::cout << "### FATAL ERROR: <mhd> resistive_freeze_background must start at "
                << "cycle 0 (restarts are not supported)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    FreezeBackground(gam0, gam1, beta_dt, 0);
  }

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
    if (capture) FreezeBackground(gam0, gam1, beta_dt, 1);
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

}  // namespace resistive
