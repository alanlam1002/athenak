//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file geom_equilibrium_test.cpp
//! \brief Unit test for Task C1/C2: a purely rotating (centrifugal-pressure-balance)
//! equilibrium in cylindrical_axisym (R,z) or spherical_polar (1D radial) coordinates,
//! with uniform density rho0 and CONSTANT rotation speed v0 in the phi slot appropriate
//! to the coordinate system (IM3 for both axisym and spherical -- see the coordinate
//! systems' handedness notes), and pressure profile P(x1) = P0 + rho0*v0^2*ln(x1/x1_0)
//! solving dP/dx1 = rho0*v0^2/x1 -- the Newtonian centrifugal-balance ODE, independent
//! of the m_coord Jacobian power, so the SAME analytic profile is the equilibrium for
//! both systems. No gravity, no z/theta dependence (v_z=v_theta=0 everywhere).
//!
//! This is the well-balancedness gate the v2 plan's Correction C2 exists for: with the
//! WRONG (1/x1v) source coefficient instead of the correct Delta-A/Delta-V form, this
//! profile would NOT stay static (it would drift at O(dR/R) even at t=0, not just
//! accumulate truncation error over time) -- see DEVELOPMENT.md Task C1/C2 log.
//!
//! Reuses the established LinearWave-style pattern (see src/pgen/tests/linear_wave.cpp):
//! sets the analytic solution into u0 at t=0 (register 0), and via pgen_final_func, sets
//! the SAME analytic solution (since it doesn't evolve with time, being an equilibrium)
//! into u1 (register 1) at the end of the run, then calls the generic OutputErrors() to
//! report the L1/RMS deviation -- which should shrink with resolution (truncation-error
//! rate), not just be small at any one resolution, since this checks a genuine dynamical
//! balance, not an exact discrete identity like the B4 conservation tests.

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
//! \fn FillGeomEquilibrium()
//! \brief writes the analytic rotating-equilibrium conserved state directly into
//! whichever register (u0 at t=0, u1 for the end-of-run reference solution) is passed.
void FillGeomEquilibrium(MeshBlockPack *pmbp, ParameterInput *pin, Mesh *pm,
                          DvceArray5D<Real> &u_target) {
  auto &indcs = pm->mb_indcs;
  int nmb1 = pmbp->nmb_thispack - 1;
  auto &geom = pmbp->pgeom->geom_data;
  Real rho0 = pin->GetReal("problem", "dens");
  Real p0   = pin->GetReal("problem", "pgas");
  Real x1_0 = pin->GetReal("problem", "r0");
  Real v0   = pin->GetReal("problem", "vphi0");
  Real gm1  = pmbp->phydro->peos->eos_data.gamma - 1.0;
  // both cylindrical_axisym and spherical_polar carry phi in the IM3 slot (see each
  // system's handedness/layout doc comment); this test never targets general cylindrical
  bool phi_in_im3 = true;

  int &is = indcs.is, &ie = indcs.ie, &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  par_for("geom_equil_ic", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x1 = geom.x1v(m, i);
    Real pgas = p0 + rho0*v0*v0*log(x1/x1_0);
    u_target(m,IDN,k,j,i) = rho0;
    u_target(m,IM1,k,j,i) = 0.0;
    u_target(m,IM2,k,j,i) = phi_in_im3 ? 0.0 : rho0*v0;
    u_target(m,IM3,k,j,i) = phi_in_im3 ? rho0*v0 : 0.0;
    u_target(m,IEN,k,j,i) = pgas/gm1 + 0.5*rho0*v0*v0;
  });
}

//----------------------------------------------------------------------------------------
//! \fn GeomEquilibriumErrors()
//! \brief pgen_final_func: writes the (unchanged) analytic solution into u1, then calls
//! the generic error-output function to report the deviation of the evolved u0 from it.
void GeomEquilibriumErrors(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  FillGeomEquilibrium(pmbp, pin, pm, pmbp->phydro->u1);
  pm->pgen->OutputErrors(pin, pm);
}
} // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::GeomEquilibriumTest()

void ProblemGenerator::GeomEquilibriumTest(ParameterInput *pin, const bool restart) {
  if (restart) return;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "GeomEquilibriumTest requires a <hydro> block" << std::endl;
    exit(EXIT_FAILURE);
  }
  FillGeomEquilibrium(pmbp, pin, pmy_mesh_, pmbp->phydro->u0);
  auto &indcs = pmy_mesh_->mb_indcs;
  pmbp->phydro->peos->ConsToPrim(pmbp->phydro->u0, pmbp->phydro->w0, false,
                                  indcs.is, indcs.ie, indcs.js, indcs.je,
                                  indcs.ks, indcs.ke);
  pgen_final_func = GeomEquilibriumErrors;
  return;
}
