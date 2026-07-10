//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mg_cfc_scalar_poisson.cpp
//! \brief implementation of MGCFCScalarPoisson[Driver]

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "multigrid/multigrid.hpp"
#include "mg_cfc_vector_poisson.hpp"  // reuses CFCVectorPoissonStencil (nvar-agnostic)
#include "mg_cfc_scalar_poisson.hpp"

//----------------------------------------------------------------------------------------
//! \fn MGCFCScalarPoisson::MGCFCScalarPoisson(...)

MGCFCScalarPoisson::MGCFCScalarPoisson(MultigridDriver *pmd, MeshBlockPack *pmbp,
                                       int nghost, bool on_host)
    : Multigrid(pmd, pmbp, nghost, on_host) {
}

MGCFCScalarPoisson::~MGCFCScalarPoisson() {
}

void MGCFCScalarPoisson::SmoothPack(int color) {
  // TODO(cfc): call the generic templated Smooth<CFCVectorPoissonStencil> helper
  // (nvar_ = 1), mirroring MGGravity::SmoothPack.
  return;
}

void MGCFCScalarPoisson::CalculateDefectPack() {
  // TODO(cfc): call the generic templated CalculateDefect<CFCVectorPoissonStencil>.
  return;
}

void MGCFCScalarPoisson::CalculateFASRHSPack() {
  // TODO(cfc): call the generic templated CalculateFASRHS<CFCVectorPoissonStencil>.
  return;
}


//----------------------------------------------------------------------------------------
//! \fn MGCFCScalarPoissonDriver::MGCFCScalarPoissonDriver(...)
//! \brief constructs the root + meshblock-level Multigrid hierarchies with nvar_ = 1
//! and Dirichlet-zero (mg_zerofixed) boundary conditions on all non-periodic faces.

MGCFCScalarPoissonDriver::MGCFCScalarPoissonDriver(MeshBlockPack *pmbp,
                                                   ParameterInput *pin)
    : MultigridDriver(pmbp, 1) {
  // TODO(cfc): set mg_mesh_bcs_[f] = BoundaryFlag::mg_zerofixed for all non-periodic
  // faces; allocate mgroot_/mglevels_ as new MGCFCScalarPoisson(...); allocate
  // mglevels_->pbval boundary buffers, mirroring MGGravityDriver::MGGravityDriver.
}

MGCFCScalarPoissonDriver::~MGCFCScalarPoissonDriver() {
  // TODO(cfc): delete mgroot_, mglevels_.
}

void MGCFCScalarPoissonDriver::Solve(Driver *pdriver, int stage, Real dt) {
  // TODO(cfc): SetupMultigrid(...) + SolveFMG(pdriver) or SolveMG(pdriver), assuming
  // LoadPoissonSource() was already called for this cycle.
  return;
}

void MGCFCScalarPoissonDriver::LoadPoissonSource(const DvceArray5D<Real> &eta_src) {
  // TODO(cfc): mglevels_->LoadSource(eta_src, /*ns=*/0, /*ngh=*/..., /*fac=*/1.0).
  return;
}

void MGCFCScalarPoissonDriver::RetrieveSolution(DvceArray5D<Real> &eta_dst) {
  // TODO(cfc): mglevels_->RetrieveResult(eta_dst, /*ns=*/0, /*ngh=*/...).
  return;
}

void MGCFCScalarPoissonDriver::SmoothOctet(MGOctet &oct, int rlev, int color) {
  // TODO(cfc): host-side octet analogue of MGCFCScalarPoisson::SmoothPack.
  return;
}

void MGCFCScalarPoissonDriver::CalculateDefectOctet(MGOctet &oct, int rlev) {
  // TODO(cfc): host-side octet analogue of MGCFCScalarPoisson::CalculateDefectPack.
  return;
}

void MGCFCScalarPoissonDriver::CalculateFASRHSOctet(MGOctet &oct, int rlev) {
  // TODO(cfc): host-side octet analogue of MGCFCScalarPoisson::CalculateFASRHSPack.
  return;
}
