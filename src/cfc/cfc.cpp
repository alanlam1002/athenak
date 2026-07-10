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
//! \brief CFC constructor: allocates intermediate fields and the 4 multigrid solvers.

CFC::CFC(MeshBlockPack *pmbp, ParameterInput *pin) :
    pmy_pack(pmbp),
    x_u("cfc_x_u", 1, 1, 1, 1, 1),
    a_dd("cfc_a_dd", 1, 1, 1, 1, 1),
    a_sq("cfc_a_sq", 1, 1, 1, 1, 1),
    psi("cfc_psi", 1, 1, 1, 1, 1),
    alpha_psi("cfc_alpha_psi", 1, 1, 1, 1, 1),
    beta_u("cfc_beta_u", 1, 1, 1, 1, 1),
    u_tilde("cfc_u_tilde", 1, 1, 1, 1, 1),
    s_tilde_d("cfc_s_tilde_d", 1, 1, 1, 1, 1),
    s_tilde("cfc_s_tilde", 1, 1, 1, 1, 1),
    pmgd_vecx(nullptr),
    pmgd_vecbeta(nullptr),
    pmgd_psi(nullptr),
    pmgd_alpha(nullptr) {
  // TODO(cfc): require pmbp->padm != nullptr and pmbp->ptmunu != nullptr (fatal error
  // otherwise, mirroring the z4c||adm + mhd check in meshblock_pack.cpp); size all
  // intermediate DvceArray5D fields to (nmb, ncomponents, ncells3, ncells2, ncells1);
  // construct pmgd_vecx/pmgd_vecbeta/pmgd_psi/pmgd_alpha.
}

//----------------------------------------------------------------------------------------
//! \fn CFC::~CFC()

CFC::~CFC() {
  delete pmgd_vecx;
  delete pmgd_vecbeta;
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
  // Step 3: psi (Gmunu eq. 73, nonlinear)
  SolveConformalFactor(pdriver, stage);
  // Step 4: rescale Ũ, S-tilde, S-tilde_i with the new psi
  RescaleMatterSources();
  // Step 5: alpha*psi (Gmunu eq. 74, nonlinear)
  SolveLapse(pdriver, stage);
  // Step 6: beta^i (Gmunu eq. 75)
  SolveShift(pdriver, stage);
  // Final: assemble psi4, g_dd, vK_dd, alpha, beta_u into padm->u_adm
  AssembleADM();
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::SolveVectorPotential(Driver *pdriver, int stage)

void CFC::SolveVectorPotential(Driver *pdriver, int stage) {
  // TODO(cfc): AssembleVectorSource(src, /*for_shift=*/false) using S-tilde_i (from
  // pmy_pack->ptmunu->tmunu.S_d), pmgd_vecx->LoadPoissonSource(src),
  // pmgd_vecx->Solve(pdriver, stage), pmgd_vecx->RetrieveSolution(...), then
  // ReconstructVectorFromPotentials(...) into x_u.
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
  // (adding back the +1 offset from the delta_psi convention).
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::RescaleMatterSources()

void CFC::RescaleMatterSources() {
  // TODO(cfc): u_tilde = psi^6 * pmy_pack->ptmunu->tmunu.E, s_tilde_d = psi^6 *
  // tmunu.S_d, s_tilde = psi^6 * (flat-metric trace of tmunu.S_dd). No con2prim call
  // needed -- Tmunu's E/S_d/S_dd are already the physical (psi-independent)
  // projections.
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
  // TODO(cfc): AssembleVectorSource(src, /*for_shift=*/true) using alpha_psi, psi,
  // a_dd, s_tilde_d (Gmunu eq. 75 rhs), pmgd_vecbeta->LoadPoissonSource(src),
  // pmgd_vecbeta->Solve(pdriver, stage), pmgd_vecbeta->RetrieveSolution(...), then
  // cfc::ReconstructVectorFromPotentials(...) into beta_u.
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::AssembleADM()

void CFC::AssembleADM() {
  // TODO(cfc): cfc::AssembleADMFromCFC(pmy_pack, psi, alpha_psi, a_dd, beta_u);
  return;
}

}  // namespace cfc
