//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file smooth_pulse_convergence_test.cpp
//! \brief Unit test for Task F3: formal convergence-order verification. A smooth
//! Gaussian density pulse (uniform background pressure/velocity, so the pulse is an
//! EXACT entropy-mode perturbation -- passively advected at the background velocity
//! with NO shape change, to full nonlinear accuracy, since it carries no pressure or
//! velocity perturbation to source any dynamics) is advected by a uniform z-velocity in
//! cylindrical_axisym (R,z) -- the required 2D curvilinear layout -- on a domain
//! periodic in z. After exactly one z-period the exact solution is IDENTICAL to the
//! t=0 IC (same reused-pgen_final_func-plus-OutputErrors() mechanism as
//! ct_field_loop_test.cpp/geom_equilibrium_test.cpp). Run at several resolutions (via
//! the python test wrapper, which reads the RMS-L1 column of consecutive -errs.dat
//! rows), the ratio of consecutive errors should approach 2^p for a scheme with design
//! order p (PLM: p=2) -- this is a genuine END-TO-END integration check (reconstruction
//! + flux divergence + time integration together), complementing but not duplicating
//! Tasks B6/B7's exact-reconstruction unit tests (which check the reconstruction
//! formula alone, not the full evolved solution).

#include <cmath>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/meshblock_pack.hpp"
#include "coordinates/mesh_geometry.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"

namespace {
//----------------------------------------------------------------------------------------
//! \fn FillSmoothPulse()
//! \brief writes the analytic pulse state into whichever conserved-variable register
//! is passed (u0 at t=0, u1 for the end-of-run reference solution).
void FillSmoothPulse(MeshBlockPack *pmbp, ParameterInput *pin,
                     DvceArray5D<Real> &u_target) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie, &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int nmb1 = pmbp->nmb_thispack - 1;
  auto &geom = pmbp->pgeom->geom_data;

  Real d0    = pin->GetOrAddReal("problem", "dens", 1.0);
  Real p0    = pin->GetOrAddReal("problem", "pgas", 1.0);
  Real w0    = pin->GetOrAddReal("problem", "vz", 1.0);
  Real amp   = pin->GetOrAddReal("problem", "amp", 0.5);
  Real sig   = pin->GetOrAddReal("problem", "sigma", 0.1);
  Real R0    = pin->GetOrAddReal("problem", "R0", 1.5);
  Real z0    = pin->GetOrAddReal("problem", "z0", 0.0);
  Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;

  par_for("smooth_pulse_ic", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real R = geom.x1v(m,i);
    Real z = geom.x2v(m,j);
    Real dist2 = SQR(R-R0) + SQR(z-z0);
    Real d = d0 + amp*d0*std::exp(-0.5*dist2/SQR(sig));
    u_target(m,IDN,k,j,i) = d;
    u_target(m,IM1,k,j,i) = 0.0;
    u_target(m,IM2,k,j,i) = d*w0;
    u_target(m,IM3,k,j,i) = 0.0;
    u_target(m,IEN,k,j,i) = p0/gm1 + 0.5*d*w0*w0;
  });
}

//----------------------------------------------------------------------------------------
//! \fn SmoothPulseErrors()
//! \brief pgen_final_func: fills u1 with the (unchanged after one period) analytic
//! solution and calls the generic OutputErrors() to report RMS-L1/Linfty.
void SmoothPulseErrors(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  FillSmoothPulse(pmbp, pin, pmbp->phydro->u1);
  pm->pgen->OutputErrors(pin, pm);
}
} // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::SmoothPulseConvergenceTest()

void ProblemGenerator::SmoothPulseConvergenceTest(ParameterInput *pin,
                                                   const bool restart) {
  if (restart) return;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "SmoothPulseConvergenceTest requires a <hydro> block"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  FillSmoothPulse(pmbp, pin, pmbp->phydro->u0);
  auto &indcs = pmy_mesh_->mb_indcs;
  pmbp->phydro->peos->ConsToPrim(pmbp->phydro->u0, pmbp->phydro->w0, false,
                                  indcs.is, indcs.ie, indcs.js, indcs.je,
                                  indcs.ks, indcs.ke);
  pgen_final_func = SmoothPulseErrors;
  return;
}
