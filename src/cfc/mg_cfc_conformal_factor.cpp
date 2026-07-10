//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mg_cfc_conformal_factor.cpp
//! \brief implementation of MGCFCConformalFactor[Driver]

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "multigrid/multigrid.hpp"
#include "mg_cfc_conformal_factor.hpp"

//----------------------------------------------------------------------------------------
//! \fn MGCFCConformalFactor::MGCFCConformalFactor(...)

MGCFCConformalFactor::MGCFCConformalFactor(MultigridDriver *pmd, MeshBlockPack *pmbp,
                                           int nghost, bool on_host)
    : Multigrid(pmd, pmbp, nghost, on_host) {
}

MGCFCConformalFactor::~MGCFCConformalFactor() {
}

void MGCFCConformalFactor::SmoothPack(int color) {
  // TODO(cfc): Newton-Gauss-Seidel point relaxation for
  // Delta(delta_psi+1) + 2 pi Ũ (delta_psi+1)^-1 + (1/8) Ahat^2 (delta_psi+1)^-7 = 0,
  // i.e. u_new = u_old - Residual(u_old)/dResidual_du(u_old), red-black colored.
  return;
}

void MGCFCConformalFactor::CalculateDefectPack() {
  // TODO(cfc): evaluate the nonlinear residual of eq. 73 at the current delta_psi.
  return;
}

void MGCFCConformalFactor::CalculateFASRHSPack() {
  // TODO(cfc): FAS coarse-grid right-hand-side correction (nonlinear operator
  // evaluated at the restricted solution), mirroring MGGravity::CalculateFASRHSPack
  // but with the nonlinear source terms included.
  return;
}


//----------------------------------------------------------------------------------------
//! \fn MGCFCConformalFactorDriver::MGCFCConformalFactorDriver(...)
//! \brief nvar_ = 1, ncoeff_ = 1 (carries Ahat^2); mg_multipole boundary conditions
//! (Gmunu eq. 77, isolated/asymptotically-flat falloff).

MGCFCConformalFactorDriver::MGCFCConformalFactorDriver(MeshBlockPack *pmbp,
                                                       ParameterInput *pin)
    : MultigridDriver(pmbp, 1) {
  // TODO(cfc): set ncoeff_ = 1; set mg_mesh_bcs_[f] = BoundaryFlag::mg_multipole for
  // all non-periodic faces; configure multipole order (mporder_) via
  // AllocateMultipoleCoefficients(); allocate mgroot_/mglevels_ as
  // new MGCFCConformalFactor(...), mirroring MGGravityDriver::MGGravityDriver.
}

MGCFCConformalFactorDriver::~MGCFCConformalFactorDriver() {
  // TODO(cfc): delete mgroot_, mglevels_.
}

void MGCFCConformalFactorDriver::Solve(Driver *pdriver, int stage, Real dt) {
  // TODO(cfc): CalculateCenterOfMass()/CalculateMultipoleCoefficients() for the
  // isolated BC, then SetupMultigrid(...) + SolveFMG(pdriver) or SolveMG(pdriver),
  // assuming LoadMatterSource()/LoadNonlinearCoefficient() were already called.
  return;
}

void MGCFCConformalFactorDriver::LoadMatterSource(const DvceArray5D<Real> &u_tilde) {
  // TODO(cfc): mglevels_->LoadSource(u_tilde, /*ns=*/0, /*ngh=*/..., /*fac=*/-2.0*M_PI).
  return;
}

void MGCFCConformalFactorDriver::LoadNonlinearCoefficient(
    const DvceArray5D<Real> &a_sq) {
  // TODO(cfc): mglevels_->LoadCoefficients(a_sq, /*ngh=*/...).
  return;
}

void MGCFCConformalFactorDriver::RetrieveSolution(DvceArray5D<Real> &dst) {
  // TODO(cfc): mglevels_->RetrieveResult(dst, /*ns=*/0, /*ngh=*/...).
  return;
}

void MGCFCConformalFactorDriver::SmoothOctet(MGOctet &oct, int rlev, int color) {
  // TODO(cfc): host-side Newton-Gauss-Seidel octet analogue of
  // MGCFCConformalFactor::SmoothPack.
  return;
}

void MGCFCConformalFactorDriver::CalculateDefectOctet(MGOctet &oct, int rlev) {
  // TODO(cfc): host-side octet analogue of MGCFCConformalFactor::CalculateDefectPack.
  return;
}

void MGCFCConformalFactorDriver::CalculateFASRHSOctet(MGOctet &oct, int rlev) {
  // TODO(cfc): host-side octet analogue of MGCFCConformalFactor::CalculateFASRHSPack.
  return;
}
