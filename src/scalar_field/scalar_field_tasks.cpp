//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file scalar_field_tasks.cpp
//! \brief functions that control scalar-field tasks in the NumericalRelativity task list
//!
//! Phase 0 task graph (start -> run -> end), no matter/Z4c back-reaction dependencies
//! yet: InitRecv -> CopyU -> CalcRHS -> ExpRKUpdate -> RestrictU -> SendU -> RecvU ->
//! ApplyPhysicalBCs -> Prolongate -> ClearSend -> ClearRecv. See DEVELOPMENT_NOTES.md for
//! the additional tasks (RescaleTmunu, ImpRKUpdate, ...) planned for later phases.

#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "scalar_field/scalar_field.hpp"
#include "tasklist/numerical_relativity.hpp"

namespace scalarfield {

//----------------------------------------------------------------------------------------
//! \fn  void ScalarField::QueueScalarFieldTasks
//! \brief queue ScalarField tasks into NumericalRelativity
void ScalarField::QueueScalarFieldTasks() {
  using namespace numrel;  // NOLINT(build/namespaces)
  NumericalRelativity *pnr = pmy_pack->pnr;
  auto &indcs = pmy_pack->pmesh->mb_indcs;

  // Start task list
  pnr->QueueTask(&ScalarField::InitRecv, this, SF_Recv, "SF_Recv", Task_Start);

  // Run task list
  pnr->QueueTask(&ScalarField::CopyU, this, SF_CopyU, "SF_CopyU", Task_Run);
  switch (indcs.ng) {
    case 2:
      pnr->QueueTask(&ScalarField::CalcRHS<2>, this, SF_CalcRHS, "SF_CalcRHS",
                     Task_Run, {SF_CopyU});
      break;
    case 3:
      pnr->QueueTask(&ScalarField::CalcRHS<3>, this, SF_CalcRHS, "SF_CalcRHS",
                     Task_Run, {SF_CopyU});
      break;
    case 4:
      pnr->QueueTask(&ScalarField::CalcRHS<4>, this, SF_CalcRHS, "SF_CalcRHS",
                     Task_Run, {SF_CopyU});
      break;
  }
  pnr->QueueTask(&ScalarField::ExpRKUpdate, this, SF_ExplRK, "SF_ExplRK",
                 Task_Run, {SF_CalcRHS});
  pnr->QueueTask(&ScalarField::RestrictU, this, SF_RestU, "SF_RestU",
                 Task_Run, {SF_ExplRK});
  pnr->QueueTask(&ScalarField::SendU, this, SF_SendU, "SF_SendU",
                 Task_Run, {SF_RestU});
  pnr->QueueTask(&ScalarField::RecvU, this, SF_RecvU, "SF_RecvU",
                 Task_Run, {SF_SendU});
  pnr->QueueTask(&ScalarField::ApplyPhysicalBCs, this, SF_BCS, "SF_BCS",
                 Task_Run, {SF_RecvU});
  pnr->QueueTask(&ScalarField::Prolongate, this, SF_Prolong, "SF_Prolong",
                 Task_Run, {SF_BCS});

  // End task list
  pnr->QueueTask(&ScalarField::ClearSend, this, SF_ClearS, "SF_ClearS", Task_End);
  pnr->QueueTask(&ScalarField::ClearRecv, this, SF_ClearR, "SF_ClearR",
                 Task_End, {SF_ClearS});
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ScalarField::InitRecv
//! \brief post non-blocking receives and initialize boundary receive status flags

TaskStatus ScalarField::InitRecv(Driver *pdrive, int stage) {
  return pbval_u->InitRecv(nscalarfield);
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ScalarField::ClearRecv
//! \brief waits for all MPI receives to complete

TaskStatus ScalarField::ClearRecv(Driver *pdrive, int stage) {
  return pbval_u->ClearRecv();
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ScalarField::ClearSend
//! \brief waits for all MPI sends to complete

TaskStatus ScalarField::ClearSend(Driver *pdrive, int stage) {
  return pbval_u->ClearSend();
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ScalarField::CopyU
//! \brief copy u0 --> u1 in first stage

TaskStatus ScalarField::CopyU(Driver *pdrive, int stage) {
  auto integrator = pdrive->integrator;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nvar = nscalarfield;
  auto &u0 = pmy_pack->pscalarfield->u0;
  auto &u1 = pmy_pack->pscalarfield->u1;

  if (integrator == "rk4") {
    Real &delta = pdrive->delta[stage-1];
    if (stage == 1) {
      Kokkos::deep_copy(DevExeSpace(), u1, u0);
    } else {
      par_for("SFCopyCons", DevExeSpace(), 0, nmb1, 0, nvar-1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
        u1(m,n,k,j,i) += delta*u0(m,n,k,j,i);
      });
    }
  } else {
    if (stage == 1) {
      Kokkos::deep_copy(DevExeSpace(), u1, u0);
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ScalarField::SendU
//! \brief sends cell-centered scalar-field variables

TaskStatus ScalarField::SendU(Driver *pdrive, int stage) {
  return pbval_u->PackAndSendCC(u0, coarse_u0);
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ScalarField::RecvU
//! \brief receives cell-centered scalar-field variables

TaskStatus ScalarField::RecvU(Driver *pdrive, int stage) {
  return pbval_u->RecvAndUnpackCC(u0, coarse_u0);
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ScalarField::RestrictU
//! \brief restrict solution onto coarse mesh (SMR/AMR only)

TaskStatus ScalarField::RestrictU(Driver *pdrive, int stage) {
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(u0, coarse_u0, true);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ScalarField::Prolongate
//! \brief prolongate solution at fine/coarse boundaries (SMR/AMR only)

TaskStatus ScalarField::Prolongate(Driver *pdrive, int stage) {
  if (pmy_pack->pmesh->multilevel) {
    pbval_u->ProlongateCC(u0, coarse_u0, true);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ScalarField::ApplyPhysicalBCs
//! \brief
//!
//! Phase 0: there is no dedicated ScalarField boundary-value formula yet (Phase 4 will
//! add a Yukawa-falloff Sommerfeld condition for the massive field, see
//! DEVELOPMENT_NOTES.md). Ghost zones are filled by inter-MeshBlock communication above;
//! only the generic user-BC hook runs here, matching the convention used by every other
//! physics module (Z4c, MHD, ...).

TaskStatus ScalarField::ApplyPhysicalBCs(Driver *pdrive, int stage) {
  if (!(pmy_pack->pmesh->strictly_periodic)) {
    if (pmy_pack->pmesh->pgen->user_bcs) {
      (pmy_pack->pmesh->pgen->user_bcs_func)(pmy_pack->pmesh);
    }
  }
  return TaskStatus::complete;
}

} // namespace scalarfield
