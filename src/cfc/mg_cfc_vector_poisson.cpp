//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mg_cfc_vector_poisson.cpp
//! \brief implementation of MGCFCVectorPoisson[Driver]

#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "multigrid/multigrid.hpp"
#include "mg_cfc_vector_poisson.hpp"

//----------------------------------------------------------------------------------------
//! \fn MGCFCVectorPoisson::MGCFCVectorPoisson(...)

MGCFCVectorPoisson::MGCFCVectorPoisson(MultigridDriver *pmd, MeshBlockPack *pmbp,
                                       int nghost, bool on_host)
    : Multigrid(pmd, pmbp, nghost, on_host) {
}

MGCFCVectorPoisson::~MGCFCVectorPoisson() {
}

void MGCFCVectorPoisson::SmoothPack(int color) {
  // TODO(cfc): call the generic templated Smooth<CFCVectorPoissonStencil> helper on
  // all 3 components (P_x, P_y, P_z), mirroring MGGravity::SmoothPack.
  return;
}

void MGCFCVectorPoisson::CalculateDefectPack() {
  // TODO(cfc): call the generic templated CalculateDefect<CFCVectorPoissonStencil>.
  return;
}

void MGCFCVectorPoisson::CalculateFASRHSPack() {
  // TODO(cfc): call the generic templated CalculateFASRHS<CFCVectorPoissonStencil>.
  return;
}


//----------------------------------------------------------------------------------------
//! \fn MGCFCVectorPoissonDriver::MGCFCVectorPoissonDriver(...)
//! \brief constructs the root + meshblock-level Multigrid hierarchies with nvar_ = 3
//! and Dirichlet-zero (mg_zerofixed) boundary conditions on all non-periodic faces
//! (P_i inherits the zero falloff of X^i|_rmax = 0, Gmunu eq. 80 / beta^i|_rmax = 0,
//! eq. 79).

MGCFCVectorPoissonDriver::MGCFCVectorPoissonDriver(MeshBlockPack *pmbp,
                                                   ParameterInput *pin)
    : MultigridDriver(pmbp, 3) {
  // TODO(cfc): set mg_mesh_bcs_[f] = BoundaryFlag::mg_zerofixed for all non-periodic
  // faces; allocate mgroot_/mglevels_ as new MGCFCVectorPoisson(...); allocate
  // mglevels_->pbval boundary buffers, mirroring MGGravityDriver::MGGravityDriver.
}

MGCFCVectorPoissonDriver::~MGCFCVectorPoissonDriver() {
  // TODO(cfc): delete mgroot_, mglevels_.
}

void MGCFCVectorPoissonDriver::Solve(Driver *pdriver, int stage, Real dt) {
  // TODO(cfc): SetupMultigrid(...) + SolveFMG(pdriver) or SolveMG(pdriver), assuming
  // LoadPoissonSource() was already called for this cycle.
  return;
}

void MGCFCVectorPoissonDriver::LoadPoissonSource(
    const AthenaTensor<Real, TensorSymm::NONE, 3, 1> &p_src) {
  // TODO(cfc): mglevels_->LoadSource(..., /*ns=*/0, /*ngh=*/..., /*fac=*/1.0), reading
  // the 3 components out of p_src's underlying storage.
  return;
}

void MGCFCVectorPoissonDriver::RetrieveSolution(
    AthenaTensor<Real, TensorSymm::NONE, 3, 1> &p_dst) {
  // TODO(cfc): mglevels_->RetrieveResult(..., /*ns=*/0, /*ngh=*/...), writing into
  // p_dst's underlying storage.
  return;
}

void MGCFCVectorPoissonDriver::SmoothOctet(MGOctet &oct, int rlev, int color) {
  // TODO(cfc): host-side octet analogue of MGCFCVectorPoisson::SmoothPack.
  return;
}

void MGCFCVectorPoissonDriver::CalculateDefectOctet(MGOctet &oct, int rlev) {
  // TODO(cfc): host-side octet analogue of MGCFCVectorPoisson::CalculateDefectPack.
  return;
}

void MGCFCVectorPoissonDriver::CalculateFASRHSOctet(MGOctet &oct, int rlev) {
  // TODO(cfc): host-side octet analogue of MGCFCVectorPoisson::CalculateFASRHSPack.
  return;
}

void MGCFCVectorPoissonDriver::ProlongateOctetBoundariesFluxCons(MGOctet &oct,
     std::vector<Real> &cbuf, const std::vector<bool> &ncoarse) {
  // TODO(cfc): flux-conservative face prolongation for the 3-component Laplacian
  // (same structure as MGGravityDriver::ProlongateOctetBoundariesFluxCons, but
  // looped over 3 channels instead of 1); or fall back to the base-class default
  // (MultigridDriver::ProlongateOctetBoundariesFluxCons) if exact flux conservation
  // is not required for these auxiliary potentials.
  return;
}
