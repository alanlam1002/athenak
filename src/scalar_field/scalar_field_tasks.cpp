//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file scalar_field_tasks.cpp
//! \brief functions that control scalar-field tasks in the NumericalRelativity task list
//!
//! Task graph (start -> run -> end): InitRecv -> CopyU -> RescaleTmunu -> CalcRHS ->
//! SomBC -> ExpRKUpdate -> RestrictU -> SendU -> RecvU -> Prolongate -> ApplyPhysicalBCs
//! -> ClearSend -> ClearRecv. Prolongate/ApplyPhysicalBCs run in that order (matching
//! Z4c's own Z4c_Prolong -> Z4c_BCS ordering) because each is now split into a
//! coarse-array step and a fine-array step: Prolongate first fills the coarse array's
//! own physical-boundary ghost zones (ScalarFieldBCsCoarse) so the prolongation stencil
//! reads valid data there, then prolongates; ApplyPhysicalBCs then fills the fine
//! array's physical-boundary ghost zones (ScalarFieldBCs) so that corner ghost zones
//! between a coarse neighbor and a physical boundary -- freshly overwritten by
//! prolongation -- end up correct rather than stale. RescaleTmunu (Phase 3) rescales
//! the fluid's Tmunu by 1/A(sphi); SomBC (Phase 4) applies the Yukawa Sommerfeld outer
//! BC. The mass term
//! itself is explicit (added directly in CalcRHS, see scalar_field_calcrhs.cpp) -- no
//! implicit/IMEX update was needed, see PLAN.md's "Mass-term treatment" section.

#include <string>

#include <math.h>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "z4c/tmunu.hpp"
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
  // Rescales the fluid's Jordan-frame Tmunu by 1/A(sphi) (a no-op, internally, when no
  // fluid/Tmunu module exists -- see RescaleTmunu below). Must run before anything that
  // reads Tmunu for a geometry-equation source: both CalcRHS's below, and (via the
  // optional-dependency mechanism) Z4c_CalcRHS.
  pnr->QueueTask(&ScalarField::RescaleTmunu, this, SF_RescaleT, "SF_RescaleT",
                 Task_Run, {}, {MHD_SetTmunu});
  switch (indcs.ng) {
    case 2:
      pnr->QueueTask(&ScalarField::CalcRHS<2>, this, SF_CalcRHS, "SF_CalcRHS",
                     Task_Run, {SF_CopyU, SF_RescaleT});
      break;
    case 3:
      pnr->QueueTask(&ScalarField::CalcRHS<3>, this, SF_CalcRHS, "SF_CalcRHS",
                     Task_Run, {SF_CopyU, SF_RescaleT});
      break;
    case 4:
      pnr->QueueTask(&ScalarField::CalcRHS<4>, this, SF_CalcRHS, "SF_CalcRHS",
                     Task_Run, {SF_CopyU, SF_RescaleT});
      break;
  }
  // Yukawa Sommerfeld outer BC (Phase 4): overwrites rhs.sphi/rhs.vpi at outflow-type
  // faces, mirroring Z4c_SomBC's placement between CalcRHS and ExplRK exactly.
  pnr->QueueTask(&ScalarField::ScalarFieldBoundaryRHS, this, SF_SomBC, "SF_SomBC",
                 Task_Run, {SF_CalcRHS});
  // SF_ExplRK must not overwrite the scalar field's u0 before Z4c_CalcRHS has read it
  // (Phase 2 back-reaction reads sf.sphi/sf.vpi directly) -- both sectors are RK-evolved
  // and this is not the "recompute fresh each stage" case MHD_SetTmunu is, so ordinary
  // task-order luck is not enough; see DEVELOPMENT_NOTES.md for the full reasoning.
  pnr->QueueTask(&ScalarField::ExpRKUpdate, this, SF_ExplRK, "SF_ExplRK",
                 Task_Run, {SF_SomBC}, {Z4c_CalcRHS});
  pnr->QueueTask(&ScalarField::RestrictU, this, SF_RestU, "SF_RestU",
                 Task_Run, {SF_ExplRK});
  pnr->QueueTask(&ScalarField::SendU, this, SF_SendU, "SF_SendU",
                 Task_Run, {SF_RestU});
  pnr->QueueTask(&ScalarField::RecvU, this, SF_RecvU, "SF_RecvU",
                 Task_Run, {SF_SendU});
  pnr->QueueTask(&ScalarField::Prolongate, this, SF_Prolong, "SF_Prolong",
                 Task_Run, {SF_RecvU});
  pnr->QueueTask(&ScalarField::ApplyPhysicalBCs, this, SF_BCS, "SF_BCS",
                 Task_Run, {SF_Prolong});

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
//! \fn  TaskStatus ScalarField::RescaleTmunu
//! \brief rescale the fluid's Jordan-frame stress-energy (E, S_d, S_dd) by 1/A(sphi) so
//! that Z4c_CalcRHS and ScalarField::CalcRHS read the Einstein-frame Tmunu, matching the
//! convention in ~/SACRA_2D/SACRA_MPI/bssn_st.f90 (its `tabfac = 1/A(sphi)` factor,
//! applied uniformly to tnn/txx/... before they're used in any geometry- or scalar-RHS
//! formula). No-op (and no-op safe -- `pmy_pack->ptmunu` stays untouched) when no
//! fluid/Tmunu module exists. A(sphi) = exp(0.5*beta0*sphi^2); beta0 is hardcoded to 1
//! here to match SACRA, same as the rest of the Phase 2/3 back-reaction terms.

TaskStatus ScalarField::RescaleTmunu(Driver *pdrive, int stage) {
  if (pmy_pack->ptmunu == nullptr) {
    return TaskStatus::complete;
  }
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nmb = pmy_pack->nmb_thispack;

  auto &tmunu = pmy_pack->ptmunu->tmunu;
  auto &sf = pmy_pack->pscalarfield->sf;

  par_for("SF RescaleTmunu", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real oo_asphi = exp(-0.5*SQR(sf.sphi(m,k,j,i)));
    tmunu.E(m,k,j,i) *= oo_asphi;
    for (int a = 0; a < 3; ++a) {
      tmunu.S_d(m,a,k,j,i) *= oo_asphi;
      for (int b = a; b < 3; ++b) {
        tmunu.S_dd(m,a,b,k,j,i) *= oo_asphi;
      }
    }
  });
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
  if (pmy_pack->pmesh->multilevel) {  // only prolongate with SMR/AMR
    // Step 1: apply physical BCs to the coarse array, so the prolongation stencil
    //         reads valid data in coarse ghost zones that sit at a physical boundary.
    if (!(pmy_pack->pmesh->strictly_periodic)) {
      pbval_u->ScalarFieldBCsCoarse(pmy_pack, pbval_u->u_in, coarse_u0);
    }

    // Step 2: prolongate fine ghost zones from the coarse array.
    pbval_u->ProlongateCC(u0, coarse_u0, true);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ScalarField::ApplyPhysicalBCs
//! \brief
//!
//! Fills sphi/Pi ghost zones at true physical (domain-edge) boundaries via
//! pbval_u->ScalarFieldBCs (scalar_field_bcs.cpp), mirroring Z4c::ApplyPhysicalBCs'
//! pbval_u->Z4cBCs call. This is a *separate* mechanism from the inter-MeshBlock
//! communication above (SendU/RecvU only exchange data with real neighbor MeshBlocks,
//! never touching true domain-edge ghost cells) -- without it, sphi/Pi ghost zones at
//! any reflect/outflow/diode/vacuum/inflow face stay frozen at their problem-generator
//! initial values for the entire run. Unlike Z4cBCs, there is no odd-parity sign flip
//! anywhere in ScalarFieldBCs: sphi and Pi are true 3-scalars (even parity under every
//! reflection), unlike Z4c's tensor components. (SF_SomBC/ScalarFieldBoundaryRHS in
//! scalar_field_Sbc.cpp is unrelated: that's an RHS/source-term correction applied
//! during CalcRHS, not a ghost-zone fill, and doesn't substitute for this.)

TaskStatus ScalarField::ApplyPhysicalBCs(Driver *pdrive, int stage) {
  if (!(pmy_pack->pmesh->strictly_periodic)) {
    // Step 3: apply physical BCs to the fine array. This is called *after*
    //         prolongation, so that the corner ghost zones between a coarse neighbor
    //         and a physical boundary read valid data.
    pbval_u->ScalarFieldBCs((pmy_pack), (pbval_u->u_in), u0);
    if (pmy_pack->pmesh->pgen->user_bcs) {
      (pmy_pack->pmesh->pgen->user_bcs_func)(pmy_pack->pmesh);
    }
  }
  return TaskStatus::complete;
}

} // namespace scalarfield
