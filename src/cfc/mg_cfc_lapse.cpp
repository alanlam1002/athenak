//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mg_cfc_lapse.cpp
//! \brief implementation of MGCFCLapse[Driver]

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "multigrid/multigrid.hpp"
#include "mg_cfc_lapse.hpp"

//----------------------------------------------------------------------------------------
//! \fn MGCFCLapse::MGCFCLapse(...)

MGCFCLapse::MGCFCLapse(MultigridDriver *pmd, MeshBlockPack *pmbp, int nghost,
                       bool on_host)
    : Multigrid(pmd, pmbp, nghost, on_host) {
}

MGCFCLapse::~MGCFCLapse() {
}

void MGCFCLapse::SmoothPack(int color) {
  // TODO(cfc): Newton-Gauss-Seidel point relaxation for
  // Delta(delta_ap+1) - (delta_ap+1)[2 pi(Ũ+2S̃)psi^-2 + (7/8)Ahat^2 psi^-8] = 0,
  // i.e. u_new = u_old - Residual(u_old)/dResidual_du(u_old), red-black colored.
  return;
}

void MGCFCLapse::CalculateDefectPack() {
  // TODO(cfc): evaluate the nonlinear residual of eq. 74 at the current delta_(alpha
  // psi).
  return;
}

void MGCFCLapse::CalculateFASRHSPack() {
  // TODO(cfc): FAS coarse-grid right-hand-side correction (nonlinear operator
  // evaluated at the restricted solution), mirroring MGCFCConformalFactor's version.
  return;
}


//----------------------------------------------------------------------------------------
//! \fn MGCFCLapseDriver::MGCFCLapseDriver(...)
//! \brief nvar_ = 1, ncoeff_ = 2 (carries psi, Ahat^2); mg_multipole boundary
//! conditions (Gmunu eq. 78, isolated/asymptotically-flat falloff).

MGCFCLapseDriver::MGCFCLapseDriver(MeshBlockPack *pmbp, ParameterInput *pin)
    : MultigridDriver(pmbp, 1) {
  // TODO(cfc): set ncoeff_ = 2; set mg_mesh_bcs_[f] = BoundaryFlag::mg_multipole for
  // all non-periodic faces; configure multipole order (mporder_) via
  // AllocateMultipoleCoefficients(); allocate mgroot_/mglevels_ as
  // new MGCFCLapse(...), mirroring MGGravityDriver::MGGravityDriver.
}

MGCFCLapseDriver::~MGCFCLapseDriver() {
  // TODO(cfc): delete mgroot_, mglevels_.
}

void MGCFCLapseDriver::Solve(Driver *pdriver, int stage, Real dt) {
  // TODO(cfc): CalculateMultipoleCoefficients() for the isolated BC, then
  // SetupMultigrid(...) + SolveFMG(pdriver) or SolveMG(pdriver), assuming
  // LoadMatterSource()/LoadKnownFields() were already called.
  return;
}

void MGCFCLapseDriver::LoadMatterSource(const DvceArray5D<Real> &u_plus_2s_tilde) {
  // TODO(cfc): mglevels_->LoadSource(u_plus_2s_tilde, /*ns=*/0, /*ngh=*/...,
  // /*fac=*/2.0*M_PI).
  return;
}

void MGCFCLapseDriver::LoadKnownFields(const DvceArray5D<Real> &psi,
                                       const DvceArray5D<Real> &a_sq) {
  // TODO(cfc): mglevels_->LoadCoefficients(..., /*ngh=*/...) for both psi and Ahat^2
  // channels.
  return;
}

void MGCFCLapseDriver::RetrieveSolution(DvceArray5D<Real> &dst) {
  // TODO(cfc): mglevels_->RetrieveResult(dst, /*ns=*/0, /*ngh=*/...).
  return;
}

void MGCFCLapseDriver::SmoothOctet(MGOctet &oct, int rlev, int color) {
  // TODO(cfc): host-side Newton-Gauss-Seidel octet analogue of
  // MGCFCLapse::SmoothPack.
  return;
}

void MGCFCLapseDriver::CalculateDefectOctet(MGOctet &oct, int rlev) {
  // TODO(cfc): host-side octet analogue of MGCFCLapse::CalculateDefectPack.
  return;
}

void MGCFCLapseDriver::CalculateFASRHSOctet(MGOctet &oct, int rlev) {
  // TODO(cfc): host-side octet analogue of MGCFCLapse::CalculateFASRHSPack.
  return;
}
