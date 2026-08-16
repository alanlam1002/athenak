//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sr_geom_equilibrium_test.cpp
//! \brief Unit test for Task G1: SR geometric source terms (src/coordinates/
//! geometric_srcterms.hpp's IsSR branch) preserve the SAME rotating centrifugal-
//! pressure-balance equilibrium as geom_equilibrium_test.cpp (uniform density, constant
//! rotation speed v0, log-pressure profile solving dP/dR = rho*v0^2/R) at SMALL v0 --
//! the v/c ->0 cross-check the v2 plan calls out explicitly, since the SR momentum-flux
//! generalization (rho -> rho*h, v^i -> u^i = Gamma*v^i) reduces to the Newtonian
//! centrifugal term to leading order in v0 (relativistic corrections enter at
//! O(v0^2/c^2) relative, i.e. they are a higher-order correction ON TOP of the same
//! leading-order balance, not a different leading-order physics) -- see
//! DEVELOPMENT.md's Task G1 log for the expansion. Uses PrimToCons() (not a hand-rolled
//! conserved-energy formula) for both the t=0 IC and the reference solution at
//! t_final, since AthenaK's SR conserved-energy formula (unlike the Newtonian
//! P/(gamma-1)+KE form geom_equilibrium_test.cpp uses directly) is nontrivial and
//! already implemented/tested in the EOS classes -- reusing it here avoids duplicating
//! or subtly getting the SR energy formula wrong in a NEW test written to verify a
//! DIFFERENT piece of physics.

#include <cmath>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/meshblock_pack.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/mesh_geometry.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"

namespace {
//----------------------------------------------------------------------------------------
//! \fn FillSRGeomEquilibrium()
//! \brief writes the analytic SR rotating-equilibrium PRIMITIVE state (d, u^R=0,
//! u^phi=Gamma*v0, u^z=0, e) into w_target, then converts to the conserved register
//! u_target via the existing (SR-aware) PrimToCons().
void FillSRGeomEquilibrium(MeshBlockPack *pmbp, ParameterInput *pin,
                            DvceArray5D<Real> &w_target, DvceArray5D<Real> &u_target) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie, &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int nmb1 = pmbp->nmb_thispack - 1;
  auto &geom = pmbp->pgeom->geom_data;

  Real rho0 = pin->GetReal("problem", "dens");
  Real p0   = pin->GetReal("problem", "pgas");
  Real x1_0 = pin->GetReal("problem", "r0");
  Real v0   = pin->GetReal("problem", "vphi0");
  Real gm1  = pmbp->phydro->peos->eos_data.gamma - 1.0;
  Real Gamma = 1.0/std::sqrt(1.0 - v0*v0);

  par_for("sr_geom_equil_ic", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x1 = geom.x1v(m,i);
    // same leading-order log profile as the Newtonian case (see file docstring)
    Real pgas = p0 + rho0*v0*v0*std::log(x1/x1_0);
    w_target(m,IDN,k,j,i) = rho0;
    w_target(m,IVX,k,j,i) = 0.0;
    w_target(m,IVY,k,j,i) = 0.0;
    // u^phi: phi is carried in IM3/IVZ for axisym/spherical
    w_target(m,IVZ,k,j,i) = Gamma*v0;
    w_target(m,IEN,k,j,i) = pgas/gm1;  // internal energy density
  });

  auto &peos = pmbp->phydro->peos;
  peos->PrimToCons(w_target, u_target, is, ie, js, je, ks, ke);
}

//----------------------------------------------------------------------------------------
//! \fn SRGeomEquilibriumErrors()
//! \brief pgen_final_func: builds the (time-invariant) reference solution in a
//! scratch primitive array, converts to u1 via PrimToCons(), and calls OutputErrors().
void SRGeomEquilibriumErrors(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int nmb = pmbp->nmb_thispack;
  int ncells1 = indcs.nx1 + 2*indcs.ng;
  int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*indcs.ng) : 1;
  int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*indcs.ng) : 1;
  DvceArray5D<Real> w_scratch("w_scratch", nmb, 5, ncells3, ncells2, ncells1);
  FillSRGeomEquilibrium(pmbp, pin, w_scratch, pmbp->phydro->u1);
  pm->pgen->OutputErrors(pin, pm);
}
} // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::SRGeomEquilibriumTest()

void ProblemGenerator::SRGeomEquilibriumTest(ParameterInput *pin, const bool restart) {
  if (restart) return;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "SRGeomEquilibriumTest requires a <hydro> block"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!pmbp->pcoord->is_special_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "SRGeomEquilibriumTest requires <coord>/special_rel=true"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  FillSRGeomEquilibrium(pmbp, pin, pmbp->phydro->w0, pmbp->phydro->u0);
  pgen_final_func = SRGeomEquilibriumErrors;
  return;
}
