//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file origin_conservation_test.cpp
//! \brief Unit test for Task E2: at the coordinate origin (x1min=0, coord=cylindrical_
//! axisym or spherical_polar), checks three things the plan calls out explicitly:
//! (1) geom.Area1 at the innermost face (is) is EXACTLY zero -- the algebraic reason a
//!     reflecting origin cannot leak mass/momentum flux regardless of the interior
//!     solution;
//! (2) the (pre-existing, UNMODIFIED) reflect BC in src/bvals/physics/hydro_bcs.cpp
//!     mirrors ghost-zone data with a sign flip on EXACTLY the radial (IVX) component
//!     and identity on the tangential ones, using the SAME mirror-index convention
//!     (ghost is-g <-> active is+g-1) as Task B6's ghost-GEOMETRY mirror -- verified
//!     here by direct inspection of w0 in the ghost zone after boundary conditions have
//!     run, for a manufactured field with a nonzero, non-symmetric velocity of every
//!     component;
//! (3) no NaN/Inf appears anywhere in the domain after evolving.
//! (Total-mass-conserved-to-roundoff at the origin is already exercised by Task B4's
//! cylindrical_axisym/spherical mass-conservation tests, both of which already use
//! x1min=0 with ix1_bc=reflect -- not re-implemented here to avoid duplicating that
//! check under a different name.)

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
constexpr Real kTol = 1.0e-13;

//----------------------------------------------------------------------------------------
//! \fn OriginConservationCheck()
//! \brief pgen_final_func: runs the three checks described in the file docstring.
void OriginConservationCheck(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie, &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke, &ng = indcs.ng;
  auto &geom = pmbp->pgeom->geom_data;
  auto &w0 = pmbp->phydro->w0;

  auto a1i_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.a1i);
  bool failed = false;

  // (1) Area1(is) == 0 exactly at the origin (m=0 suffices; all MeshBlocks share the
  // same x1min in this single-level test).
  if (a1i_h(0, is) != 0.0) {
    std::cout << "Origin Conservation Test FAILED: Area1 factor at is is "
              << a1i_h(0, is) << ", expected exactly 0" << std::endl;
    failed = true;
  }

  // (2) ghost-zone sign flip: IVX (radial) negated, IVY/IVZ (tangential) unchanged,
  // comparing ghost cell (is-g) to its mirror-index active partner (is+g-1) -- the
  // SAME convention MirrorReflectingGhostGeometry uses for xv/xf (Task B6) and
  // MirrorReflectingGhostPpmCoeffs uses for the PPM coefficients (Task B7).
  auto w0_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), w0);
  int m = 0, k = ks, j = js;
  for (int g = 1; g <= ng; ++g) {
    int ghost_i = is - g;
    int mirror_i = is + g - 1;
    Real vr_ghost = w0_h(m,IVX,k,j,ghost_i), vr_mirror = w0_h(m,IVX,k,j,mirror_i);
    Real vt2_ghost = w0_h(m,IVY,k,j,ghost_i), vt2_mirror = w0_h(m,IVY,k,j,mirror_i);
    Real vt3_ghost = w0_h(m,IVZ,k,j,ghost_i), vt3_mirror = w0_h(m,IVZ,k,j,mirror_i);
    Real d_ghost = w0_h(m,IDN,k,j,ghost_i), d_mirror = w0_h(m,IDN,k,j,mirror_i);
    if (std::abs(vr_ghost + vr_mirror) > kTol*std::fmax(std::abs(vr_mirror), 1.0)) {
      std::cout << "Origin Conservation Test FAILED: ghost radial velocity at g=" << g
                << " is " << vr_ghost << ", expected -" << vr_mirror << std::endl;
      failed = true;
    }
    if (std::abs(vt2_ghost - vt2_mirror) > kTol*std::fmax(std::abs(vt2_mirror), 1.0)) {
      std::cout << "Origin Conservation Test FAILED: ghost tangential-2 velocity at g="
                << g << " is " << vt2_ghost << ", expected " << vt2_mirror << std::endl;
      failed = true;
    }
    if (std::abs(vt3_ghost - vt3_mirror) > kTol*std::fmax(std::abs(vt3_mirror), 1.0)) {
      std::cout << "Origin Conservation Test FAILED: ghost tangential-3 velocity at g="
                << g << " is " << vt3_ghost << ", expected " << vt3_mirror << std::endl;
      failed = true;
    }
    if (std::abs(d_ghost - d_mirror) > kTol*std::fmax(std::abs(d_mirror), 1.0)) {
      std::cout << "Origin Conservation Test FAILED: ghost density at g=" << g
                << " is " << d_ghost << ", expected " << d_mirror << std::endl;
      failed = true;
    }
  }

  // (3) no NaN/Inf anywhere in the active domain.
  for (int kk = ks; kk <= ke && !failed; ++kk) {
    for (int jj = js; jj <= je && !failed; ++jj) {
      for (int ii = is; ii <= ie; ++ii) {
        for (int n = 0; n < 5; ++n) {
          Real val = w0_h(0,n,kk,jj,ii);
          if (!std::isfinite(val)) {
            std::cout << "Origin Conservation Test FAILED: non-finite value "
                      << val << " at n=" << n << " k=" << kk << " j=" << jj
                      << " i=" << ii << std::endl;
            failed = true;
          }
        }
      }
    }
  }

  if (failed) {
    std::cout << "Origin Conservation Test FAILED (see above)" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout << "Origin Conservation Test Passed" << std::endl;
}
} // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::OriginConservationTest()

void ProblemGenerator::OriginConservationTest(ParameterInput *pin, const bool restart) {
  if (restart) return;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "OriginConservationTest requires a <hydro> block"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie, &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  auto &geom = pmbp->pgeom->geom_data;
  auto &u0 = pmbp->phydro->u0;
  auto &w0 = pmbp->phydro->w0;
  EOS_Data &eos = pmbp->phydro->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;

  Real d0 = pin->GetOrAddReal("problem", "dens", 1.0);
  Real p0 = pin->GetOrAddReal("problem", "pgas", 1.0);
  Real vr0 = pin->GetOrAddReal("problem", "vr0", 0.2);
  Real vt0 = pin->GetOrAddReal("problem", "vt0", 0.15);
  Real vp0 = pin->GetOrAddReal("problem", "vp0", 0.1);

  int nmb1 = pmbp->nmb_thispack - 1;
  // manufactured smooth, nonzero, non-symmetric velocity field (all 3 components
  // nonzero and mutually distinct so the sign-flip/identity checks above are not
  // masked by an accidental symmetry).
  par_for("origin_cons_ic", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x1 = geom.x1v(m,i);
    w0(m,IDN,k,j,i) = d0*(1.0 + 0.1*std::sin(x1));
    w0(m,IVX,k,j,i) = vr0*x1;
    w0(m,IVY,k,j,i) = vt0;
    w0(m,IVZ,k,j,i) = vp0;
    w0(m,IEN,k,j,i) = p0;
  });
  pmbp->phydro->peos->PrimToCons(w0, u0, is, ie, js, je, ks, ke);

  pgen_final_func = OriginConservationCheck;
  return;
}
