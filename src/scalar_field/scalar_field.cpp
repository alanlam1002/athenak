//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file scalar_field.cpp
//! \brief implementation of ScalarField class constructor and destructor

#include <math.h>

#include <string>

#include <Kokkos_Core.hpp>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "scalar_field/scalar_field.hpp"

namespace scalarfield {

char const * const ScalarField::ScalarField_names[ScalarField::nscalarfield] = {
  "sf_sphi",
  "sf_pi",
};

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

ScalarField::ScalarField(MeshBlockPack *ppack, ParameterInput *pin) :
  u0("u0 sf",1,1,1,1,1),
  u1("u1 sf",1,1,1,1,1),
  u_rhs("u_rhs sf",1,1,1,1,1),
  coarse_u0("coarse u0 sf",1,1,1,1,1),
  pmy_pack(ppack) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));
  {
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    Kokkos::Profiling::pushRegion("ScalarField tensor fields");
    Kokkos::realloc(u0,    nmb, (nscalarfield), ncells3, ncells2, ncells1);
    Kokkos::realloc(u1,    nmb, (nscalarfield), ncells3, ncells2, ncells1);
    Kokkos::realloc(u_rhs, nmb, (nscalarfield), ncells3, ncells2, ncells1);

    sf.sphi.InitWithShallowSlice(u0, I_SF_SPHI);
    sf.vpi.InitWithShallowSlice (u0, I_SF_PI);

    rhs.sphi.InitWithShallowSlice(u_rhs, I_SF_SPHI);
    rhs.vpi.InitWithShallowSlice (u_rhs, I_SF_PI);

    opt.omega_c = pin->GetOrAddReal("scalarfield", "omega_c", 12.0);
    opt.beta0   = pin->GetOrAddReal("scalarfield", "beta0", 1.0);
    opt.mass2   = pin->GetOrAddReal("scalarfield", "mass2", 0.0);
    opt.sphi0   = pin->GetOrAddReal("scalarfield", "sphi0", 0.0);
    opt.diss    = pin->GetOrAddReal("scalarfield", "diss", 0.0);
    opt.newton_tol     = pin->GetOrAddReal("scalarfield", "newton_tol", 1.0e-10);
    opt.newton_maxiter = pin->GetOrAddInteger("scalarfield", "newton_maxiter", 20);

    diss = opt.diss*pow(2., -2.*indcs.ng)*(indcs.ng % 2 == 0 ? -1. : 1.);
    Kokkos::Profiling::popRegion();
  }

  // allocate memory for solution on coarse mesh, needed for SMR/AMR
  if (ppack->pmesh->multilevel) {
    int nccells1 = indcs.cnx1 + 2*(indcs.ng);
    int nccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
    int nccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(coarse_u0, nmb, (nscalarfield), nccells3, nccells2, nccells1);
  }

  // Allocate boundary buffers for cell-centered variables. The scalar field is
  // differentiated twice (D^2 sphi) and feeds directly into the Z4c RHS once the
  // back-reaction is enabled (Phase 2), so -- like Z4c's own fields -- it needs
  // 4th-order-safe prolongation/restriction across refinement boundaries; hence
  // the "is_z4c=true" buffer treatment here, even though this class is a separate
  // module from z4c::Z4c.
  Kokkos::Profiling::pushRegion("ScalarField buffers");
  pbval_u = new MeshBoundaryValuesCC(ppack, pin, true);
  pbval_u->InitializeBuffers(nscalarfield);
  Kokkos::Profiling::popRegion();
}

//----------------------------------------------------------------------------------------
// destructor

ScalarField::~ScalarField() {
  delete pbval_u;
}

} // namespace scalarfield
