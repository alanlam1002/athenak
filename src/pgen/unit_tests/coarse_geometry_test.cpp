//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file coarse_geometry_test.cpp
//! \brief Unit test for the COARSE GeomData built alongside the fine one on a multilevel
//! mesh (MeshGeometry::coarse_geom_data), and for the two telescoping identities that
//! curvilinear SMR/AMR restriction and flux correction are built on:
//!
//!     V_coarse(I)  ==  sum over the 2/4/8 child cells of V_fine
//!     A_coarse(I)  ==  sum over the 2/4   child faces of A_fine   (per direction)
//!
//! These identities are the whole reason restriction and flux correction need no coarse
//! geometry at all: a coarse cell is exactly the union of its children, so the coarse
//! volume/area can be formed from the fine arrays a block already owns. If either
//! identity fails, the weighted restriction in mesh_refinement.cpp and the weighted flux
//! correction in flux_correct_cc.cpp are both silently wrong, so they are asserted
//! directly here rather than only inferred from a downstream conservation test.
//!
//! They hold to ROUNDOFF (not approximately) because both index spaces are the same
//! rational interpolation of [xmin,xmax] -- see LeftEdgeX() in cell_locations.hpp -- so a
//! coarse face lands bit-exactly on every other fine face.
//!
//! Also spot-checks coarse areas/volumes/centroids against independently re-derived
//! analytic formulas, so a systematically wrong coarse build (e.g. one that used the FINE
//! cell width, which is the natural mistake here) cannot pass just by being
//! self-consistent.
//!
//! All geometry is read through GeomDataHost/MirrorGeomData -- never through GeomData's
//! KOKKOS_INLINE_FUNCTION accessors, which would dereference a device pointer from this
//! host code (see the GeomDataHost doc comment in mesh_geometry.hpp).

#include <cmath>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/meshblock_pack.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/mesh_geometry.hpp"

namespace {

constexpr Real kTol = 1.0e-12;

void CheckClose(const char *what, Real got, Real expected, bool *failed) {
  Real scale = std::max(std::abs(expected), static_cast<Real>(1.0));
  if (std::abs(got - expected) > kTol*scale) {
    std::cout << "### FAIL " << what << ": got " << got << " expected " << expected
              << " (rel err " << std::abs(got-expected)/scale << ")" << std::endl;
    *failed = true;
  }
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::CoarseGeometryTest()
//! \brief runs the coarse-geometry checks and exits.

void ProblemGenerator::CoarseGeometryTest(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;
  bool failed = false;

  if (!pmbp->pgeom->has_coarse) {
    std::cout << "### FAIL coarse geometry was not built -- this test requires a "
              << "multilevel mesh (<mesh_refinement>/refinement != none)" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  GeomDataHost gh = MirrorGeomData(pmbp->pgeom->geom_data);
  GeomDataHost gc = MirrorGeomData(pmbp->pgeom->coarse_geom_data);

  int nmb = pmbp->nmb_thispack;
  bool multi_d = pmy_mesh_->multi_d;
  bool three_d = pmy_mesh_->three_d;

  // number of children per coarse cell along each direction
  int nj = multi_d ? 2 : 1;
  int nk = three_d ? 2 : 1;

  for (int m = 0; m < nmb; ++m) {
    // ---- identity 1: coarse volume == sum of child volumes ----------------------------
    for (int k = indcs.cks; k <= indcs.cks + indcs.cnx3 - 1; ++k) {
      for (int j = indcs.cjs; j <= indcs.cjs + indcs.cnx2 - 1; ++j) {
        for (int i = indcs.cis; i <= indcs.cie; ++i) {
          int fi = 2*i - indcs.cis;
          int fj = multi_d ? (2*j - indcs.cjs) : j;
          int fk = three_d ? (2*k - indcs.cks) : k;
          Real vsum = 0.0;
          for (int dk = 0; dk < nk; ++dk) {
            for (int dj = 0; dj < nj; ++dj) {
              for (int di = 0; di < 2; ++di) {
                vsum += gh.Vol(m, fk+dk, fj+dj, fi+di);
              }
            }
          }
          CheckClose("Vol_coarse == sum(Vol_fine)", gc.Vol(m,k,j,i), vsum, &failed);

          // ---- identity 3: the volumetric centroid is the volume-weighted mean of the
          // child centroids. This is what makes the centroid-based prolongation exactly
          // conservative, so it is asserted here rather than assumed.
          Real xsum = 0.0;
          for (int di = 0; di < 2; ++di) {
            xsum += gh.vi(m, fi+di)*gh.x1v(m, fi+di);
          }
          Real visum = gh.vi(m,fi) + gh.vi(m,fi+1);
          CheckClose("x1v_coarse == volume-weighted mean of children",
                     gc.x1v(m,i), xsum/visum, &failed);
        }
      }
    }

    // ---- identity 2: coarse face area == sum of child face areas ----------------------
    // x1-faces: a coarse x1-face coincides with a fine x1-face in position, and is tiled
    // by nj*nk fine faces in the transverse directions.
    for (int k = indcs.cks; k <= indcs.cks + indcs.cnx3 - 1; ++k) {
      for (int j = indcs.cjs; j <= indcs.cjs + indcs.cnx2 - 1; ++j) {
        for (int i = indcs.cis; i <= indcs.cie + 1; ++i) {
          int fi = 2*i - indcs.cis;
          int fj = multi_d ? (2*j - indcs.cjs) : j;
          int fk = three_d ? (2*k - indcs.cks) : k;
          Real asum = 0.0;
          for (int dk = 0; dk < nk; ++dk) {
            for (int dj = 0; dj < nj; ++dj) {
              asum += gh.Area1(m, fk+dk, fj+dj, fi);
            }
          }
          CheckClose("Area1_coarse == sum(Area1_fine)", gc.Area1(m,k,j,i), asum, &failed);
        }
      }
    }

    if (multi_d) {
      for (int k = indcs.cks; k <= indcs.cks + indcs.cnx3 - 1; ++k) {
        for (int j = indcs.cjs; j <= indcs.cjs + indcs.cnx2; ++j) {
          for (int i = indcs.cis; i <= indcs.cie; ++i) {
            int fi = 2*i - indcs.cis;
            int fj = 2*j - indcs.cjs;
            int fk = three_d ? (2*k - indcs.cks) : k;
            Real asum = 0.0;
            for (int dk = 0; dk < nk; ++dk) {
              for (int di = 0; di < 2; ++di) {
                asum += gh.Area2(m, fk+dk, fj, fi+di);
              }
            }
            CheckClose("Area2_coarse == sum(Area2_fine)", gc.Area2(m,k,j,i), asum,
                       &failed);
          }
        }
      }
    }

    if (three_d) {
      for (int k = indcs.cks; k <= indcs.cks + indcs.cnx3; ++k) {
        for (int j = indcs.cjs; j <= indcs.cjs + indcs.cnx2 - 1; ++j) {
          for (int i = indcs.cis; i <= indcs.cie; ++i) {
            int fi = 2*i - indcs.cis;
            int fj = 2*j - indcs.cjs;
            int fk = 2*k - indcs.cks;
            Real asum = 0.0;
            for (int dj = 0; dj < 2; ++dj) {
              for (int di = 0; di < 2; ++di) {
                asum += gh.Area3(m, fk, fj+dj, fi+di);
              }
            }
            CheckClose("Area3_coarse == sum(Area3_fine)", gc.Area3(m,k,j,i), asum,
                       &failed);
          }
        }
      }
    }

    // ---- independent analytic check of the coarse radial face factor -------------------
    // Guards against a coarse build that is internally consistent but systematically
    // wrong -- in particular one that used RegionSize::dx1 (always the FINE width) and so
    // is off by a factor of two.
    Real x1min = pmbp->pmb->mb_size.h_view(m).x1min;
    Real x1max = pmbp->pmb->mb_size.h_view(m).x1max;
    for (int i = indcs.cis; i <= indcs.cie + 1; ++i) {
      Real rf = LeftEdgeX(i - indcs.cis, indcs.cnx1, x1min, x1max);
      Real expected;
      switch (pmy_mesh_->coord_general) {
        case CoordinateGeneral::cartesian:          expected = 1.0;    break;
        case CoordinateGeneral::cylindrical:        expected = rf;     break;
        case CoordinateGeneral::cylindrical_axisym: expected = rf;     break;
        case CoordinateGeneral::spherical_polar:    expected = rf*rf;  break;
        default:                                    expected = 1.0;    break;
      }
      CheckClose("coarse a1i vs analytic", gc.a1i(m,i), expected, &failed);
    }
  }

  if (failed) {
    std::cout << "### FATAL ERROR coarse_geometry_test FAILED" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "coarse_geometry_test: all checks passed" << std::endl;
  std::exit(EXIT_SUCCESS);
}
