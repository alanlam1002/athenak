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
#include "z4c/tmunu.hpp"
#include "driver/driver.hpp"
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
//! \fn void CFC::Solve(Driver *pdriver, int stage)
//! \brief runs the 6-step XCFC solve (Gmunu sec. 2.6) and writes the result into
//! pmy_pack->padm->u_adm.

void CFC::Solve(Driver *pdriver, int stage) {
  // Step 1: X^i (Gmunu eq. 72)
  SolveVectorPotential(pdriver, stage);
  // Step 2: Adual^ij, Ahat^2 (Gmunu eq. 76)
  ComputeADual();
  // Step 3: psi (Gmunu eq. 73, nonlinear); also writes psi4/g_dd into padm->u_adm
  SolveConformalFactor(pdriver, stage);
  // Step 4: recover primitives via con2prim (now that psi/g_dd is known) and build
  // the trace matter source S-tilde needed by the lapse equation
  RescaleMatterSources(pdriver, stage);
  // Step 5: alpha*psi (Gmunu eq. 74, nonlinear)
  SolveLapse(pdriver, stage);
  // Step 6: beta^i (Gmunu eq. 75)
  SolveShift(pdriver, stage);
  // Final: assemble psi4, g_dd, vK_dd, alpha, beta_u into padm->u_adm
  AssembleADM();
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::AssembleVectorSource(...)

void CFC::AssembleVectorSource(AthenaTensor<Real, TensorSymm::NONE, 3, 1> &p_src,
                               DvceArray5D<Real> &eta_src, bool for_shift) {
  // TODO(cfc): if (!for_shift): p_src = 8*pi*S-tilde_i (Gmunu eq. 72 rhs);
  // if (for_shift): p_src = 16*pi*alpha*psi^-6*S-tilde_i + 2*Adual^ij*D_j(alpha*psi^-6)
  // (Gmunu eq. 75 rhs); then eta_src = -p_src_i * x^i (Shibata eq. 3.11) in both cases.
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::SolveVectorPotential(Driver *pdriver, int stage)

void CFC::SolveVectorPotential(Driver *pdriver, int stage) {
  // TODO(cfc): AssembleVectorSource(p_src, eta_src, /*for_shift=*/false) using
  // S-tilde_i (from pmy_pack->ptmunu->tmunu.S_d) to build both right-hand sides.
  // Then, in order (P_i first, since P_x/P_y/P_z/eta are all independent of each
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
  // pmy_pack->padm->u_adm -- RescaleMatterSources()'s con2prim call right after this
  // reads padm->adm.g_dd directly (PrimitiveSolverHydro::ConsToPrim), so this write
  // cannot be deferred to the final AssembleADM() step.
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::RescaleMatterSources(Driver *pdriver, int stage)

void CFC::RescaleMatterSources(Driver *pdriver, int stage) {
  // TODO(cfc): pmy_pack->pdyngr->ConToPrim(pdriver, stage) -- the same virtual,
  // EOS-policy-agnostic entry point dyn_grmhd's own MHD_C2P task uses (see
  // dyn_grmhd.hpp's DynGRMHD::ConToPrim / dyn_grmhd.cpp's
  // DynGRMHDPS<...>::ConToPrim), which internally calls
  // eos.ConsToPrim(pmhd->u0, pmhd->b0, pmhd->bcc0, pmhd->w0, temperature, ...) and
  // reads pmy_pack->padm->adm.g_dd to do the inversion -- valid now that
  // SolveConformalFactor() has already written psi4/g_dd for this stage. This fills
  // pmy_pack->pmhd->w0 with density/pressure/velocity.
  //
  // u_tilde (psi^6 U) and s_tilde_d (psi^6 S_i) do NOT need primitives: per
  // dyn_grmhd.cpp's DynGRMHD::SetTmunu (lines 461/463), ptmunu->tmunu.E and
  // .S_d are algebraically exact functions of the conserved state alone
  // (E = (tau+D)/sqrt(gamma), S_i = cons_momentum_i/sqrt(gamma)), so u_tilde/
  // s_tilde_d can be built directly from ptmunu->tmunu.E/.S_d (or equivalently
  // straight from pmy_pack->pmhd->u0) multiplied by the newly-solved psi^6, same
  // as steps 1/3 already do -- no con2prim involved.
  //
  // s_tilde (trace of S_ij, needed by the lapse equation in step 5) is different:
  // SetTmunu's tmunu.S_dd (dyn_grmhd.cpp lines 464-468) is built from prim/w0
  // (velocity, pressure) evaluated with whatever g_dd was current when SetTmunu
  // last ran -- i.e. the *previous* stage's psi, not the one just solved in step 3.
  // So ptmunu->tmunu.S_dd's trace is stale here and must NOT be reused directly.
  // Instead, recompute the trace here using the freshly recovered w0 (density,
  // pressure, velocity) and the new g_dd: s_tilde = psi^6 * (rho*h*W^2*v^2 + 3*P),
  // mirroring SetTmunu's own S_dd formula but evaluated post-con2prim, post-new-psi.
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
  // TODO(cfc): cfc::AssembleADMFromCFC(pmy_pack, psi, alpha_psi, a_dd, beta_u);
  return;
}

}  // namespace cfc
