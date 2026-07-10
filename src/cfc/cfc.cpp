//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cfc.cpp
//! \brief implementation of the CFC class

#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "parameter_input.hpp"
#include "coordinates/adm.hpp"
#include "mhd/mhd.hpp"
#include "driver/driver.hpp"
#include "tasklist/numerical_relativity.hpp"
#include "cfc.hpp"
#include "cfc_reconstruct.hpp"

namespace cfc {

//----------------------------------------------------------------------------------------
//! \fn CFC::CFC(MeshBlockPack *pmbp, ParameterInput *pin)
//! \brief CFC constructor: allocates intermediate fields and the 6 multigrid solvers.

CFC::CFC(MeshBlockPack *pmbp, ParameterInput *pin) :
    pmy_pack(pmbp),
    u_x("cfc_u_x", 1, 1, 1, 1, 1),
    u_beta("cfc_u_beta", 1, 1, 1, 1, 1),
    u_adual("cfc_u_adual", 1, 1, 1, 1, 1),
    a_sq("cfc_a_sq", 1, 1, 1, 1, 1),
    psi("cfc_psi", 1, 1, 1, 1, 1),
    alpha_psi("cfc_alpha_psi", 1, 1, 1, 1, 1),
    u_tilde("cfc_u_tilde", 1, 1, 1, 1, 1),
    u_stilde("cfc_u_stilde", 1, 1, 1, 1, 1),
    s_tilde("cfc_s_tilde", 1, 1, 1, 1, 1),
    u_p_x("cfc_u_p_x", 1, 1, 1, 1, 1),
    eta_x("cfc_eta_x", 1, 1, 1, 1, 1),
    u_p_beta("cfc_u_p_beta", 1, 1, 1, 1, 1),
    eta_beta("cfc_eta_beta", 1, 1, 1, 1, 1),
    pmgd_px(nullptr),
    pmgd_etax(nullptr),
    pmgd_pbeta(nullptr),
    pmgd_etabeta(nullptr),
    pmgd_psi(nullptr),
    pmgd_alpha(nullptr) {
  // TODO(cfc): require pmbp->padm != nullptr and pmbp->ptmunu != nullptr (fatal error
  // otherwise, mirroring the z4c||adm + mhd check in meshblock_pack.cpp); size all
  // intermediate DvceArray5D storage fields (u_x, u_beta, u_adual, a_sq, psi,
  // alpha_psi, u_tilde, u_stilde, s_tilde, u_p_x, eta_x, u_p_beta, eta_beta) to
  // (nmb, ncomponents, ncells3, ncells2, ncells1); then wire the AthenaTensor views
  // into their backing storage, mirroring adm::ADM::ADM(...):
  //   x_u.InitWithShallowSlice(u_x, 0, 2);
  //   beta_u.InitWithShallowSlice(u_beta, 0, 2);
  //   a_dd.InitWithShallowSlice(u_adual, 0, 5);
  //   s_tilde_d.InitWithShallowSlice(u_stilde, 0, 2);
  //   p_x.InitWithShallowSlice(u_p_x, 0, 2);
  //   p_beta.InitWithShallowSlice(u_p_beta, 0, 2);
  // (eta_x, eta_beta are genuine scalars and stay plain DvceArray5D<Real> -- no
  // AthenaTensor view needed.)
  // construct pmgd_px/pmgd_etax/pmgd_pbeta/pmgd_etabeta/pmgd_psi/pmgd_alpha.
}

//----------------------------------------------------------------------------------------
//! \fn CFC::~CFC()

CFC::~CFC() {
  delete pmgd_px;
  delete pmgd_etax;
  delete pmgd_pbeta;
  delete pmgd_etabeta;
  delete pmgd_psi;
  delete pmgd_alpha;
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::QueueCFCTasks()
//! \brief queues CFC's tasks into the shared NumericalRelativity task graph, interleaved
//! with dyn_grmhd's hydro/con2prim tasks (see cfc.hpp's QueueCFCTasks doc comment for
//! the full dependency rationale).

void CFC::QueueCFCTasks() {
  using namespace numrel;  // NOLINT(build/namespaces)
  NumericalRelativity *pnr = pmy_pack->pnr;

  pnr->QueueTask(&CFC::SolveVecXTask, this, CFC_SolveVecX, "CFC_SolveVecX",
                 Task_Run, {MHD_AddSrc});
  pnr->QueueTask(&CFC::SolvePsiTask, this, CFC_SolvePsi, "CFC_SolvePsi",
                 Task_Run, {CFC_SolveVecX});
  pnr->QueueTask(&CFC::RescaleSrcTask, this, CFC_RescaleSrc, "CFC_RescaleSrc",
                 Task_Run, {MHD_C2P});
  pnr->QueueTask(&CFC::SolveLapseTask, this, CFC_SolveLapse, "CFC_SolveLapse",
                 Task_Run, {CFC_RescaleSrc});
  pnr->QueueTask(&CFC::SolveShiftTask, this, CFC_SolveShift, "CFC_SolveShift",
                 Task_Run, {CFC_SolveLapse});
  pnr->QueueTask(&CFC::AssembleFinalTask, this, CFC_AssembleFinal, "CFC_AssembleFinal",
                 Task_Run, {CFC_SolveShift});
  return;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus CFC::SolveVecXTask(Driver *pdriver, int stage)
//! \brief steps 1-2: X^i vector potential (eq. 72) and Adual^ij/Ahat^2 (eq. 76). Runs
//! after MHD_AddSrc, i.e. once this stage's hydro flux+source update has produced the
//! conserved state AssembleVectorSource reads from.

TaskStatus CFC::SolveVecXTask(Driver *pdriver, int stage) {
  SolveVectorPotential(pdriver, stage);
  ComputeADual();
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus CFC::SolvePsiTask(Driver *pdriver, int stage)
//! \brief step 3: psi (nonlinear), then the early psi4/g_dd write MHD_C2P depends on.

TaskStatus CFC::SolvePsiTask(Driver *pdriver, int stage) {
  SolveConformalFactor(pdriver, stage);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus CFC::RescaleSrcTask(Driver *pdriver, int stage)
//! \brief step 4: rebuild S-tilde from the primitives MHD_C2P (this task's dependency)
//! just recovered -- no con2prim call here, see RescaleMatterSources.

TaskStatus CFC::RescaleSrcTask(Driver *pdriver, int stage) {
  RescaleMatterSources(pdriver, stage);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus CFC::SolveLapseTask(Driver *pdriver, int stage)
//! \brief step 5: alpha*psi (nonlinear).

TaskStatus CFC::SolveLapseTask(Driver *pdriver, int stage) {
  SolveLapse(pdriver, stage);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus CFC::SolveShiftTask(Driver *pdriver, int stage)
//! \brief step 6: beta^i.

TaskStatus CFC::SolveShiftTask(Driver *pdriver, int stage) {
  SolveShift(pdriver, stage);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus CFC::AssembleFinalTask(Driver *pdriver, int stage)
//! \brief final step: vK_dd/alpha/beta_u -> padm->u_adm.

TaskStatus CFC::AssembleFinalTask(Driver *pdriver, int stage) {
  AssembleADM();
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::AssembleVectorSource(...)

void CFC::AssembleVectorSource(AthenaTensor<Real, TensorSymm::NONE, 3, 1> &p_src,
                               DvceArray5D<Real> &eta_src, bool for_shift) {
  // TODO(cfc): if (!for_shift): build S-tilde_i = psi^6 * S_i directly from
  // pmy_pack->pmhd->u0's momentum components (IM1..IM3) divided by sqrt(detg) of the
  // *current* (not-yet-updated-by-this-solve) padm->adm.g_dd -- mirrors
  // dyn_grmhd.cpp's DynGRMHD::SetTmunu (S_d(m,a,...) = cons(IM1+a,...)*ivol), but
  // computed inline here rather than read from ptmunu (which may not even be
  // populated -- see cfc.hpp's u_tilde comment); then p_src = 8*pi*S-tilde_i
  // (Gmunu eq. 72 rhs).
  // if (for_shift): p_src = 16*pi*alpha*psi^-6*S-tilde_i + 2*Adual^ij*D_j(alpha*psi^-6)
  // (Gmunu eq. 75 rhs), reusing s_tilde_d already built for the X^i solve.
  // In both cases: eta_src = -p_src_i * x^i (Shibata eq. 3.11).
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::SolveVectorPotential(Driver *pdriver, int stage)

void CFC::SolveVectorPotential(Driver *pdriver, int stage) {
  // TODO(cfc): AssembleVectorSource(p_src, eta_src, /*for_shift=*/false), built
  // directly from pmy_pack->pmhd->u0 (post MHD_AddSrc, this task's dependency -- see
  // QueueCFCTasks), to build both right-hand sides. Then, in order (P_i first, since
  // P_x/P_y/P_z/eta are all independent of each
  // other but eta's source was built from the same S_i used for P_i):
  //   1. pmgd_px->LoadPoissonSource(p_src); pmgd_px->Solve(pdriver, stage);
  //      pmgd_px->RetrieveSolution(p_x);
  //   2. pmgd_etax->LoadPoissonSource(eta_src); pmgd_etax->Solve(pdriver, stage);
  //      pmgd_etax->RetrieveSolution(eta_x);
  // Finally cfc::ReconstructVectorFromPotentials(pmy_pack, p_x, eta_x, x_u).
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::ComputeADual()

void CFC::ComputeADual() {
  // TODO(cfc): cfc::ComputeADualFromX(pmy_pack, x_u, a_dd); then contract a_dd with
  // itself (flat metric) into a_sq.
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::SolveConformalFactor(Driver *pdriver, int stage)

void CFC::SolveConformalFactor(Driver *pdriver, int stage) {
  // TODO(cfc): pmgd_psi->LoadMatterSource(u_tilde), pmgd_psi->LoadNonlinearCoefficient
  // (a_sq), pmgd_psi->Solve(pdriver, stage), pmgd_psi->RetrieveSolution(...) into psi
  // (adding back the +1 offset from the delta_psi convention). Then
  // cfc::AssembleConformalMetric(pmy_pack, psi) to write psi4/g_dd into
  // pmy_pack->padm->u_adm -- MHD_C2P (the single con2prim shared with dyn_grmhd,
  // queued to depend on CFC_SolvePsi -- see QueueCFCTasks/dyn_grmhd.cpp) reads
  // padm->adm.g_dd directly (PrimitiveSolverHydro::ConsToPrim), so this write cannot
  // be deferred to the final AssembleADM() step.
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::RescaleMatterSources(Driver *pdriver, int stage)

void CFC::RescaleMatterSources(Driver *pdriver, int stage) {
  // TODO(cfc): No con2prim call here -- RescaleSrcTask depends on MHD_C2P (dyn_grmhd's
  // own per-stage con2prim, queued to depend on CFC_SolvePsi so it inverts against
  // this stage's new g_dd -- see QueueCFCTasks/dyn_grmhd.cpp's MHD_C2P task), so
  // pmy_pack->pmhd->w0 (density, pressure, velocity) is already fresh by the time
  // this task runs. This is the fix for the double con2prim call: CFC used to run
  // its own ConToPrim here, duplicating dyn_grmhd's; now there is exactly one
  // con2prim per stage, shared by both.
  //
  // u_tilde (psi^6 U) and s_tilde_d (psi^6 S_i) do NOT need primitives at all: build
  // them directly from pmy_pack->pmhd->u0 (D, S_i, tau -- see AssembleVectorSource),
  // the same way steps 1/3 already do -- no con2prim involved either way.
  //
  // s_tilde (trace of S_ij, needed by the lapse equation in step 5) is different: it
  // needs primitives (velocity, pressure), which is exactly why this task waits for
  // MHD_C2P. Mirror dyn_grmhd.cpp's DynGRMHD::SetTmunu's S_dd formula (lines
  // 464-468), computing s_tilde = psi^6 * (rho*h*W^2*v^2 + 3*P) directly from the
  // fresh w0/g_dd this task now has (no ptmunu involved).
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::SolveLapse(Driver *pdriver, int stage)

void CFC::SolveLapse(Driver *pdriver, int stage) {
  // TODO(cfc): pmgd_alpha->LoadMatterSource(u_tilde + 2*s_tilde),
  // pmgd_alpha->LoadKnownFields(psi, a_sq), pmgd_alpha->Solve(pdriver, stage),
  // pmgd_alpha->RetrieveSolution(...) into alpha_psi (adding back the +1 offset).
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::SolveShift(Driver *pdriver, int stage)

void CFC::SolveShift(Driver *pdriver, int stage) {
  // TODO(cfc): AssembleVectorSource(p_src, eta_src, /*for_shift=*/true) using
  // alpha_psi, psi, a_dd, s_tilde_d (Gmunu eq. 75 rhs) to build both right-hand
  // sides. Then, in order (P_i first, same reasoning as SolveVectorPotential):
  //   1. pmgd_pbeta->LoadPoissonSource(p_src); pmgd_pbeta->Solve(pdriver, stage);
  //      pmgd_pbeta->RetrieveSolution(p_beta);
  //   2. pmgd_etabeta->LoadPoissonSource(eta_src); pmgd_etabeta->Solve(pdriver, stage);
  //      pmgd_etabeta->RetrieveSolution(eta_beta);
  // Finally cfc::ReconstructVectorFromPotentials(pmy_pack, p_beta, eta_beta, beta_u).
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::AssembleADM()

void CFC::AssembleADM() {
  // TODO(cfc): cfc::AssembleLapseShiftK(pmy_pack, psi, alpha_psi, a_dd, beta_u);
  return;
}

}  // namespace cfc
