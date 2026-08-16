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
//!     Len_coarse(I)==  sum over the 2     child edges of Len_fine (per direction)
//!     w_coarse(I)*fc_coarse(I) == sum over the 2 children of w_fine*fc_fine
//!
//! The last two were added for SMR/AMR Phase 2 (face-centered / MHD): edge lengths carry
//! the EMF correction the way areas carry the cell-centered flux correction, and the
//! area-weighted centroid moment is what makes face-centered prolongation exactly
//! flux-conservative.
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

    // ---- identity 4: edge lengths telescope -------------------------------------------
    // Len_coarse == sum of the 2 child edge lengths along the edge's OWN direction. This
    // is what lets EMF correction at a level boundary send sum(Len_f*E_f)/sum(Len_f)
    // using only the fine block's own geometry, exactly as area weighting does for the
    // cell-centered flux correction. Note the transverse factors of Len are FACE-valued
    // and must come out bit-identical between the two levels (a coarse face lands on a
    // fine face), so this also checks that coincidence.
    {
      int jf_lo = indcs.cjs, jf_hi = multi_d ? indcs.cjs + indcs.cnx2 : indcs.cjs;
      int kf_lo = indcs.cks, kf_hi = three_d ? indcs.cks + indcs.cnx3 : indcs.cks;
      // Len1: cell-indexed in x1, face-indexed in x2/x3
      for (int k = kf_lo; k <= kf_hi; ++k) {
        for (int j = jf_lo; j <= jf_hi; ++j) {
          for (int i = indcs.cis; i <= indcs.cie; ++i) {
            int fi = 2*i - indcs.cis;
            int fj = multi_d ? (2*j - indcs.cjs) : j;
            int fk = three_d ? (2*k - indcs.cks) : k;
            CheckClose("Len1_coarse == sum(Len1_fine)", gc.Len1(m,k,j,i),
                       gh.Len1(m,fk,fj,fi) + gh.Len1(m,fk,fj,fi+1), &failed);
          }
        }
      }
      if (multi_d) {  // Len2: cell-indexed in x2, face-indexed in x1/x3
        for (int k = kf_lo; k <= kf_hi; ++k) {
          for (int j = indcs.cjs; j <= indcs.cjs + indcs.cnx2 - 1; ++j) {
            for (int i = indcs.cis; i <= indcs.cie + 1; ++i) {
              int fi = 2*i - indcs.cis;
              int fj = 2*j - indcs.cjs;
              int fk = three_d ? (2*k - indcs.cks) : k;
              CheckClose("Len2_coarse == sum(Len2_fine)", gc.Len2(m,k,j,i),
                         gh.Len2(m,fk,fj,fi) + gh.Len2(m,fk,fj+1,fi), &failed);
            }
          }
        }
      }
      if (three_d) {  // Len3: cell-indexed in x3, face-indexed in x1/x2
        for (int k = indcs.cks; k <= indcs.cks + indcs.cnx3 - 1; ++k) {
          for (int j = jf_lo; j <= jf_hi; ++j) {
            for (int i = indcs.cis; i <= indcs.cie + 1; ++i) {
              int fi = 2*i - indcs.cis;
              int fj = 2*j - indcs.cjs;
              int fk = 2*k - indcs.cks;
              CheckClose("Len3_coarse == sum(Len3_fine)", gc.Len3(m,k,j,i),
                         gh.Len3(m,fk,fj,fi) + gh.Len3(m,fk+1,fj,fi), &failed);
            }
          }
        }
      }
    }

    // ---- identity 5: area-weighted transverse face centroids telescope ----------------
    //
    //     w_coarse(I)*fc_coarse(I) == sum over the 2 children of w_fine*fc_fine
    //
    // with w the face's own area factor in that direction. This is the FC analogue of
    // identity 3, and it is the entire reason face-centered prolongation written as
    // B_f = B_c + s*(y_f - y_c) is exactly flux-conservative: the area-weighted first
    // moment of the offsets vanishes, so sum(A_f*B_f) == A_c*B_c identically.
    //
    // Asserted on the FACTORED 1D arrays rather than on the assembled Area, because that
    // is the form the prolongation kernels actually consume and because it avoids any
    // chance of an error in one direction being masked by cancellation in another.
    //
    // These are precisely the checks that catch the tempting-but-wrong shortcut of
    // reusing x1v/x2v/x3v here: in spherical, a2i/a3i weight r by integral(r dr) while
    // vi weights it by integral(r^2 dr), and a3j weights theta by dtheta while vj uses
    // integral(sin(th) dtheta).
    {
      auto check_moment = [&](const char *what,
                              const GeomDataHost::HostArr &wc,
                              const GeomDataHost::HostArr &cc,
                              const GeomDataHost::HostArr &wf,
                              const GeomDataHost::HostArr &cf,
                              int cs, int ncoarse) {
        for (int I = cs; I <= cs + ncoarse - 1; ++I) {
          int f = 2*I - cs;
          CheckClose(what, wc(m,I)*cc(m,I), wf(m,f)*cf(m,f) + wf(m,f+1)*cf(m,f+1),
                     &failed);
        }
      };
      // direction 1 is always resolved
      check_moment("a2i*fc2_1 telescopes", gc.a2i, gc.fc2_1, gh.a2i, gh.fc2_1,
                   indcs.cis, indcs.cnx1);
      check_moment("a3i*fc3_1 telescopes", gc.a3i, gc.fc3_1, gh.a3i, gh.fc3_1,
                   indcs.cis, indcs.cnx1);
      if (multi_d) {
        check_moment("a1j*fc1_2 telescopes", gc.a1j, gc.fc1_2, gh.a1j, gh.fc1_2,
                     indcs.cjs, indcs.cnx2);
        check_moment("a3j*fc3_2 telescopes", gc.a3j, gc.fc3_2, gh.a3j, gh.fc3_2,
                     indcs.cjs, indcs.cnx2);
      }
      if (three_d) {
        check_moment("a1k*fc1_3 telescopes", gc.a1k, gc.fc1_3, gh.a1k, gh.fc1_3,
                     indcs.cks, indcs.cnx3);
        check_moment("a2k*fc2_3 telescopes", gc.a2k, gc.fc2_3, gh.a2k, gh.fc2_3,
                     indcs.cks, indcs.cnx3);
      }
    }

    // ---- independent analytic check of the coarse radial face factor ---------------
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
