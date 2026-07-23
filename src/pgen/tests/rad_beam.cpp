//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_beam.cpp
//  \brief Beam test for radiation.  Also checks orthonormality of tetrad

// C++ headers
#include <algorithm>  // min, max
#include <iostream>   // endl
#include <limits>     // numeric_limits
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <utility>    // pair
#include <vector>

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "mesh/mesh.hpp"
#include "radiation/radiation.hpp"
#include "radiation/radiation_tetrad.hpp"
#include "pgen/pgen.hpp"

// Prototypes for user-defined BCs
void ZeroIntensity(Mesh *pm);
void CrossingBeamBoundary(Mesh *pm);

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::RadiationBeam(ParameterInput *pin)
//! \brief Checks tetrad is orthonormal.  Beam is introduced as rad_srcterm, so nothing
//! need be done here

void ProblemGenerator::RadiationBeam(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  // User boundary function
  user_bcs_func = ZeroIntensity;

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int nmb1 = (pmbp->nmb_thispack-1);
  int nang1 = (pmbp->prad->prgeo->nangles-1);
  auto &size = pmbp->pmb->mb_size;
  auto &flat = pmbp->pcoord->coord_data.is_minkowski;
  auto &spin = pmbp->pcoord->coord_data.bh_spin;
  auto &use_excise = pmbp->pcoord->coord_data.bh_excise;
  auto &excision_floor_ = pmbp->pcoord->excision_floor;

  auto &tet_c_ = pmbp->prad->tet_c;
  par_for("check_tetrad",DevExeSpace(),0,nmb1,0,nang1,0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    bool excised = false;
    if (use_excise) {
      if (excision_floor_(m,k,j,i)) {
        excised = true;
      }
    }

    if (!(excised)) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1v,x2v,x3v,flat,spin,glower,gupper);

      // Compute eta_alpha beta = g_mu nu e^mu_alpha e^nu_beta
      Real test_eta[4][4] = {0.0};
      for (int alpha=0; alpha<4; ++alpha) {
        for (int beta=0; beta<4; ++beta) {
          test_eta[alpha][beta] = 0.0;
          for (int mu=0; mu<4; ++mu) {
            for (int nu=0; nu<4; ++nu) {
              test_eta[alpha][beta] += (glower[mu][nu]*
                                        tet_c_(m,alpha,mu,k,j,i)*tet_c_(m,beta,nu,k,j,i));
            }
          }
        }
      }

      // Check for orthonormality
      for (int alpha=0; alpha<4; ++alpha) {
        for (int beta=0; beta<4; ++beta) {
          Real comp = 1.0;
          if   (alpha != beta) comp =  0.0;
          else if (alpha == 0) comp = -1.0;
          if (fabs(test_eta[alpha][beta] - comp) > 1.0e-13) {
            Kokkos::abort("Tetrad is not orthonormal!\n");
          }
        }
      }
    }
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ZeroIntensity
//! \brief Sets boundary condition on surfaces of computational domain

void ZeroIntensity(Mesh *pm) {
  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  auto &mb_bcs = pm->pmb_pack->pmb->mb_bcs;

  // Determine if radiation is enabled
  bool is_radiation_enabled_ = (pm->pmb_pack->prad != nullptr) ? true : false;
  DvceArray5D<Real> i0_; int nang1;
  if (is_radiation_enabled_) {
    i0_ = pm->pmb_pack->prad->i0;
    nang1 = pm->pmb_pack->prad->prgeo->nangles - 1;
  }
  int nmb = pm->pmb_pack->nmb_thispack;

  // X1-Boundary
  if (is_radiation_enabled_) {
    // Set X1-BCs on i0 if Meshblock face is at the edge of computational domain
    par_for("noinflow_rad_x1", DevExeSpace(),0,(nmb-1),0,nang1,0,(n3-1),0,(n2-1),
    KOKKOS_LAMBDA(int m, int n, int k, int j) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          i0_(m,n,k,j,is-i-1) = 0.0;
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          i0_(m,n,k,j,ie+i+1) = 0.0;
        }
      }
    });
  }

  // X2-Boundary
  if (is_radiation_enabled_) {
    // Set X2-BCs on i0 if Meshblock face is at the edge of computational domain
    par_for("noinflow_rad_x2", DevExeSpace(),0,(nmb-1),0,nang1,0,(n3-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int n, int k, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          i0_(m,n,k,js-j-1,i) = 0.0;
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          i0_(m,n,k,je+j+1,i) = 0.0;
        }
      }
    });
  }

  // x3-Boundary
  if (is_radiation_enabled_) {
    // Set x3-BCs on i0 if Meshblock face is at the edge of computational domain
    par_for("noinflow_rad_x3", DevExeSpace(),0,(nmb-1),0,nang1,0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int n, int j, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          i0_(m,n,ks-k-1,j,i) = 0.0;
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          i0_(m,n,ke+k+1,j,i) = 0.0;
        }
      }
    });
  }

  return;
}


//========================================================================================
// [Stage 12] Colliding-beams test (arXiv:2302.04283 Section 3.1), ported from
// ~/athenak_IAS's src/pgen/tests/rad_beam.cpp (RadiationCrossingBeams and its
// helpers), stripped of the dyn_radiation (pmbp->pdynrad) branches -- this
// module has no dyn_radiation submodule, only the static-background
// pmbp->prad path is needed. Two noninteracting, optically-thin beams cross
// in flat spacetime; this angular-grid discretization can represent both
// simultaneously (a genuinely bimodal radiation field at the crossing
// point), which the M1 module's own colliding-beams pgen
// (RadiationM1CrossingBeams, below) exists specifically to show it cannot.
//========================================================================================

namespace {

struct CrossingBeamData {
  bool enabled = false;
  Real amp = 1.0;
  Real sigma = 0.055;
  Real flux_fraction = 0.995;
  Real x0 = 0.12;
  Real y_lower = 0.15;
  Real y_upper = 0.85;
  // [Stage 12] radius of the continuously-re-emitting source disk around each of
  // (x0,y_lower)/(x0,y_upper), matching the paper's own "radius of 1/10" (arXiv:
  // 2302.04283 Section 3.1). Needed because this test's sources sit strictly
  // *inside* the domain (x0 > x1min) -- unlike ~/athenak_IAS's own as-ported
  // mechanism, which only ever (a) fills the whole downstream ray pattern once,
  // at t=0, as an initial condition, and (b) refreshes the x1min *ghost* zone
  // every step (CrossingBeamBoundary) -- (b) is a geometric no-op here: a ghost
  // cell at x<x1min<x0 is always "behind" the source (CrossingBeamProfile's
  // along<0 check), so nothing is ever actually re-injected there. Confirmed by
  // direct measurement: without a genuine continuous interior source, the t=0
  // pattern simply free-streams out through the outflow boundaries with nothing
  // replacing it (domain-integrated R^tt fell from 1648 at t=0 to 1.3e-3 by
  // t=2.5 in an early diagnostic run) -- not remotely a steady state. The fix:
  // FillCrossingBeams's per-step (boundaries_only=true) call now also
  // re-asserts the analytic profile within source_radius of either source
  // point, every step -- a genuine continuous point source, not just an IC.
  Real source_radius = 0.1;
  Real lower_profile_qx = 1.0;
  Real lower_profile_qy = 0.0;
  Real upper_profile_qx = 1.0;
  Real upper_profile_qy = 0.0;
  DvceArray2D<Real> *angular_weights = nullptr;
};

CrossingBeamData crossing_beams;

bool SolveLinear4(Real a[4][5], Real x[4]) {
  for (int col=0; col<4; ++col) {
    int pivot = col;
    Real max_abs = fabs(a[col][col]);
    for (int row=col+1; row<4; ++row) {
      const Real value = fabs(a[row][col]);
      if (value > max_abs) {
        max_abs = value;
        pivot = row;
      }
    }
    if (max_abs < 1.0e-14) {
      return false;
    }
    if (pivot != col) {
      for (int c=col; c<5; ++c) {
        std::swap(a[col][c], a[pivot][c]);
      }
    }
    const Real inv_pivot = 1.0/a[col][col];
    for (int c=col; c<5; ++c) {
      a[col][c] *= inv_pivot;
    }
    for (int row=0; row<4; ++row) {
      if (row == col) {
        continue;
      }
      const Real factor = a[row][col];
      for (int c=col; c<5; ++c) {
        a[row][c] -= factor*a[col][c];
      }
    }
  }
  for (int row=0; row<4; ++row) {
    x[row] = a[row][4];
  }
  return true;
}

bool SolveLinear3(Real a[3][4], Real x[3]) {
  for (int col=0; col<3; ++col) {
    int pivot = col;
    Real max_abs = fabs(a[col][col]);
    for (int row=col+1; row<3; ++row) {
      const Real value = fabs(a[row][col]);
      if (value > max_abs) {
        max_abs = value;
        pivot = row;
      }
    }
    if (max_abs < 1.0e-14) {
      return false;
    }
    if (pivot != col) {
      for (int c=col; c<4; ++c) {
        std::swap(a[col][c], a[pivot][c]);
      }
    }
    const Real inv_pivot = 1.0/a[col][col];
    for (int c=col; c<4; ++c) {
      a[col][c] *= inv_pivot;
    }
    for (int row=0; row<3; ++row) {
      if (row == col) {
        continue;
      }
      const Real factor = a[row][col];
      for (int c=col; c<4; ++c) {
        a[row][c] -= factor*a[col][c];
      }
    }
  }
  for (int row=0; row<3; ++row) {
    x[row] = a[row][3];
  }
  return true;
}

// Positive all-angle maximum-entropy projection of a requested beam direction onto the
// geodesic angular grid: exact injected zeroth moment (weights sum to 1) and exact
// first moment along the requested direction, up to the realizable flux factor of the
// finite angular grid (flux_fraction scales how close to that realizable maximum to
// target). Populates weights(beam, 0:nangles-1).
template <typename NhView, typename SolidAngleView, typename WeightView>
void SetAllAngleMomentWeights(NhView nh_c, SolidAngleView solid_angles,
                              WeightView weights, const int beam, const int nangles,
                              const Real qx_in, const Real qy_in, const Real qz_in,
                              const Real flux_fraction, const char *label) {
  for (int n=0; n<nangles; ++n) {
    weights(beam,n) = 0.0;
  }

  const Real qnorm = sqrt(SQR(qx_in) + SQR(qy_in) + SQR(qz_in));
  if (qnorm <= 0.0) {
    throw std::runtime_error(std::string(label) + " has a zero beam direction");
  }
  const Real qx = qx_in/qnorm;
  const Real qy = qy_in/qnorm;
  const Real qz = qz_in/qnorm;

  std::vector<std::pair<Real, int>> ranked;
  ranked.reserve(nangles);
  for (int n=0; n<nangles; ++n) {
    const Real dot = nh_c.h_view(n,1)*qx + nh_c.h_view(n,2)*qy
                   + nh_c.h_view(n,3)*qz;
    ranked.emplace_back(dot, n);
  }
  std::sort(ranked.begin(), ranked.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });

  Real best_r = -1.0;
  const int ncand = std::min(nangles, 80);
  for (int ia=0; ia<ncand; ++ia) {
    const int aidx = ranked[ia].second;
    for (int ib=ia+1; ib<ncand; ++ib) {
      const int bidx = ranked[ib].second;
      for (int ic=ib+1; ic<ncand; ++ic) {
        const int cidx = ranked[ic].second;
        Real mat[4][5] = {
          {nh_c.h_view(aidx,1), nh_c.h_view(bidx,1), nh_c.h_view(cidx,1), -qx, 0.0},
          {nh_c.h_view(aidx,2), nh_c.h_view(bidx,2), nh_c.h_view(cidx,2), -qy, 0.0},
          {nh_c.h_view(aidx,3), nh_c.h_view(bidx,3), nh_c.h_view(cidx,3), -qz, 0.0},
          {1.0,                 1.0,                 1.0,                 0.0, 1.0},
        };
        Real sol[4];
        if (!(SolveLinear4(mat, sol))) {
          continue;
        }
        const Real min_lam = std::min(sol[0], std::min(sol[1], sol[2]));
        const Real r = sol[3];
        if (min_lam >= -1.0e-10 && r > best_r && r <= 1.0 + 1.0e-10) {
          best_r = r;
        }
      }
    }
  }
  if (best_r <= 0.0) {
    throw std::runtime_error(std::string(label) + " could not find a realizable "
                             "projected angular flux");
  }

  const Real frac = fmin(0.999999, fmax(0.0, flux_fraction));
  const Real target_flux = frac*best_r;
  const Real target[3] = {target_flux*qx, target_flux*qy, target_flux*qz};
  Real lambda[3] = {0.0, 0.0, 0.0};
  bool converged = false;
  for (int iter=0; iter<80; ++iter) {
    Real max_arg = -std::numeric_limits<Real>::max();
    for (int n=0; n<nangles; ++n) {
      const Real arg = lambda[0]*nh_c.h_view(n,1) +
                       lambda[1]*nh_c.h_view(n,2) +
                       lambda[2]*nh_c.h_view(n,3);
      max_arg = fmax(max_arg, arg);
    }

    Real z = 0.0;
    Real moment[3] = {0.0, 0.0, 0.0};
    Real second[3][3] = {
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0}
    };
    for (int n=0; n<nangles; ++n) {
      const Real prior = solid_angles.h_view(n)/(4.0*M_PI);
      const Real arg = lambda[0]*nh_c.h_view(n,1) +
                       lambda[1]*nh_c.h_view(n,2) +
                       lambda[2]*nh_c.h_view(n,3);
      const Real unorm = prior*exp(arg - max_arg);
      z += unorm;
      const Real dir[3] = {nh_c.h_view(n,1), nh_c.h_view(n,2), nh_c.h_view(n,3)};
      for (int a=0; a<3; ++a) {
        moment[a] += unorm*dir[a];
        for (int b=0; b<3; ++b) {
          second[a][b] += unorm*dir[a]*dir[b];
        }
      }
    }
    for (int a=0; a<3; ++a) {
      moment[a] /= z;
      for (int b=0; b<3; ++b) {
        second[a][b] /= z;
      }
    }

    const Real residual[3] = {target[0] - moment[0],
                              target[1] - moment[1],
                              target[2] - moment[2]};
    const Real res_norm = sqrt(SQR(residual[0]) + SQR(residual[1]) +
                               SQR(residual[2]));
    if (res_norm < 1.0e-12) {
      converged = true;
      break;
    }

    Real mat[3][4];
    for (int a=0; a<3; ++a) {
      for (int b=0; b<3; ++b) {
        mat[a][b] = second[a][b] - moment[a]*moment[b];
      }
      mat[a][a] += 1.0e-14;
      mat[a][3] = residual[a];
    }
    Real delta[3];
    if (!(SolveLinear3(mat, delta))) {
      break;
    }
    Real max_delta = fmax(fabs(delta[0]), fmax(fabs(delta[1]), fabs(delta[2])));
    const Real step = (max_delta > 8.0) ? (8.0/max_delta) : 1.0;
    for (int a=0; a<3; ++a) {
      lambda[a] += step*delta[a];
    }
  }
  if (!(converged)) {
    throw std::runtime_error(std::string(label) + " all-angle moment projection "
                             "did not converge");
  }

  Real max_arg = -std::numeric_limits<Real>::max();
  for (int n=0; n<nangles; ++n) {
    const Real arg = lambda[0]*nh_c.h_view(n,1) +
                     lambda[1]*nh_c.h_view(n,2) +
                     lambda[2]*nh_c.h_view(n,3);
    max_arg = fmax(max_arg, arg);
  }
  Real z = 0.0;
  for (int n=0; n<nangles; ++n) {
    const Real prior = solid_angles.h_view(n)/(4.0*M_PI);
    const Real arg = lambda[0]*nh_c.h_view(n,1) +
                     lambda[1]*nh_c.h_view(n,2) +
                     lambda[2]*nh_c.h_view(n,3);
    weights(beam,n) = prior*exp(arg - max_arg);
    z += weights(beam,n);
  }
  Real sum_w = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
  for (int n=0; n<nangles; ++n) {
    weights(beam,n) /= z;
    const Real w = weights(beam,n);
    sum_w += w;
    mx += w*nh_c.h_view(n,1);
    my += w*nh_c.h_view(n,2);
    mz += w*nh_c.h_view(n,3);
  }
  const Real moment_err = sqrt(SQR(mx - target[0]) + SQR(my - target[1]) +
                               SQR(mz - target[2]));
  if (fabs(sum_w - 1.0) > 1.0e-11 || moment_err > 1.0e-10) {
    throw std::runtime_error(std::string(label) + " all-angle projection failed "
                             "moment check");
  }
}

KOKKOS_INLINE_FUNCTION
Real CrossingBeamProfile(const Real x, const Real y,
                         const Real x0, const Real y0,
                         const Real qx, const Real qy,
                         const Real sigma, const Real amp) {
  const Real dx = x - x0;
  const Real dy = y - y0;
  const Real along = dx*qx + dy*qy;
  if (along < 0.0) {
    return 0.0;
  }
  const Real perp = -dx*qy + dy*qx;
  return amp*exp(-0.5*SQR(perp/sigma));
}

// [Stage 12] Only the pmbp->prad branch is ported -- this module has no
// dyn_radiation submodule (pmbp->pdynrad), unlike ~/athenak_IAS.
void FillCrossingBeams(Mesh *pm, const bool boundaries_only) {
  if (!(crossing_beams.enabled)) {
    return;
  }
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;  int &ie = indcs.ie;
  int &js = indcs.js;  int &je = indcs.je;
  int &ks = indcs.ks;  int &ke = indcs.ke;
  int nmb1 = pmbp->nmb_thispack - 1;

  if (pmbp->prad == nullptr) {
    throw std::runtime_error("rad_crossing_beams requires <radiation>");
  }
  int nang1 = pmbp->prad->prgeo->nangles - 1;
  auto i0 = pmbp->prad->i0;
  auto nh_c = pmbp->prad->nh_c;
  auto solid_angles = pmbp->prad->prgeo->solid_angles;
  auto tet_c = pmbp->prad->tet_c;
  auto tetcov_c = pmbp->prad->tetcov_c;

  auto &size = pmbp->pmb->mb_size;
  auto &mb_bcs = pmbp->pmb->mb_bcs;
  const Real amp = crossing_beams.amp;
  const Real sigma = crossing_beams.sigma;
  const Real x0 = crossing_beams.x0;
  const Real y_lower = crossing_beams.y_lower;
  const Real y_upper = crossing_beams.y_upper;
  const Real source_radius = crossing_beams.source_radius;
  if (crossing_beams.angular_weights == nullptr) {
    throw std::runtime_error("crossing-beam angular weights were not initialized");
  }
  auto angular_weights = *(crossing_beams.angular_weights);
  const Real lower_profile_qx = crossing_beams.lower_profile_qx;
  const Real lower_profile_qy = crossing_beams.lower_profile_qy;
  const Real upper_profile_qx = crossing_beams.upper_profile_qx;
  const Real upper_profile_qy = crossing_beams.upper_profile_qy;

  par_for("crossing_beams_fill",DevExeSpace(),0,nmb1,0,nang1,0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    const Real x = CellCenterX(i-is, indcs.nx1,
                               size.d_view(m).x1min, size.d_view(m).x1max);
    const Real y = CellCenterX(j-js, indcs.nx2,
                               size.d_view(m).x2min, size.d_view(m).x2max);

    bool fill_cell = !(boundaries_only);
    if (boundaries_only) {
      // [Stage 12] genuine continuous point source: re-assert the analytic
      // profile every step within source_radius of either source point (see
      // CrossingBeamData::source_radius above for why this is required --
      // the ghost-zone-only conditions below are a no-op for this geometry,
      // kept for consistency with the original ported mechanism).
      bool near_source =
          (SQR(x - x0) + SQR(y - y_lower) <= SQR(source_radius)) ||
          (SQR(x - x0) + SQR(y - y_upper) <= SQR(source_radius));
      fill_cell = near_source ||
          ((i < is && mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::user) ||
           (i > ie && mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::user) ||
           (j < js && mb_bcs.d_view(m, BoundaryFace::inner_x2) == BoundaryFlag::user) ||
           (j > je && mb_bcs.d_view(m, BoundaryFace::outer_x2) == BoundaryFlag::user) ||
           (k < ks && mb_bcs.d_view(m, BoundaryFace::inner_x3) == BoundaryFlag::user) ||
           (k > ke && mb_bcs.d_view(m, BoundaryFace::outer_x3) == BoundaryFlag::user));
    }
    if (!(fill_cell)) {
      return;
    }

    Real intensity = 0.0;
    intensity += CrossingBeamProfile(x, y, x0, y_lower,
                                     lower_profile_qx, lower_profile_qy,
                                     sigma, amp)*angular_weights(0,n)/
                 solid_angles.d_view(n);
    intensity += CrossingBeamProfile(x, y, x0, y_upper,
                                     upper_profile_qx, upper_profile_qy,
                                     sigma, amp)*angular_weights(1,n)/
                 solid_angles.d_view(n);

    Real n_0 = 0.0;
    for (int d=0; d<4; ++d) {
      n_0 += tetcov_c(m,d,0,k,j,i)*nh_c.d_view(n,d);
    }
    Real norm = tet_c(m,0,0,k,j,i)*n_0;
    i0(m,n,k,j,i) = norm*intensity;
  });
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn CrossingBeamBoundary
//! \brief Fills physical ghost zones with the analytic crossing-beam inflow/outflow
//! state.

void CrossingBeamBoundary(Mesh *pm) {
  FillCrossingBeams(pm, true);
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::RadiationCrossingBeams(ParameterInput *pin)
//! \brief Two noninteracting beams crossing in flat spacetime.  The initialized
//! one-sided Gaussian beam profiles use the requested physical axes, and the
//! angular weights are a positive all-angle maximum-entropy projection with exact
//! injected zeroth moment and exact first moment along the requested beam direction,
//! up to the realizable flux factor of the finite angular grid.
//! [Stage 12] Ported from ~/athenak_IAS, stripped of the dyn_radiation branch (see
//! FillCrossingBeams above).

void ProblemGenerator::RadiationCrossingBeams(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  user_bcs_func = CrossingBeamBoundary;

  if (pmbp->prad == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "rad_crossing_beams requires a <radiation> block" << std::endl;
    exit(EXIT_FAILURE);
  }
  int nangles = pmbp->prad->prgeo->nangles;
  auto nh_c = pmbp->prad->nh_c;
  auto solid_angles = pmbp->prad->prgeo->solid_angles;

  crossing_beams.enabled = true;
  crossing_beams.amp = pin->GetOrAddReal("problem", "beam_amp", 1.0);
  crossing_beams.sigma = pin->GetOrAddReal("problem", "beam_sigma", 0.055);
  crossing_beams.flux_fraction =
      pin->GetOrAddReal("problem", "beam_flux_fraction", 0.995);
  crossing_beams.x0 = pin->GetOrAddReal("problem", "beam_x0", 0.12);
  crossing_beams.y_lower = pin->GetOrAddReal("problem", "beam_y_lower", 0.15);
  crossing_beams.y_upper = pin->GetOrAddReal("problem", "beam_y_upper", 0.85);
  crossing_beams.source_radius =
      pin->GetOrAddReal("problem", "beam_source_radius", 0.1);
  const Real x_cross = pin->GetOrAddReal("problem", "beam_x_cross", 0.75);
  const Real y_cross = pin->GetOrAddReal("problem", "beam_y_cross", 0.5);

  Real lower_tx = x_cross - crossing_beams.x0;
  Real lower_ty = y_cross - crossing_beams.y_lower;
  Real upper_tx = x_cross - crossing_beams.x0;
  Real upper_ty = y_cross - crossing_beams.y_upper;
  Real lower_norm = sqrt(SQR(lower_tx) + SQR(lower_ty));
  Real upper_norm = sqrt(SQR(upper_tx) + SQR(upper_ty));
  if (lower_norm <= 0.0 || upper_norm <= 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "rad_crossing_beams requires nonzero beam directions" << std::endl;
    exit(EXIT_FAILURE);
  }
  lower_tx /= lower_norm;
  lower_ty /= lower_norm;
  upper_tx /= upper_norm;
  upper_ty /= upper_norm;

  if (crossing_beams.angular_weights == nullptr) {
    crossing_beams.angular_weights = new DvceArray2D<Real>();
  }
  Kokkos::realloc(*(crossing_beams.angular_weights), 2, nangles);
  auto h_weights = Kokkos::create_mirror_view(*(crossing_beams.angular_weights));
  SetAllAngleMomentWeights(nh_c, solid_angles, h_weights,
                           0, nangles, lower_tx, lower_ty, 0.0,
                           crossing_beams.flux_fraction, "rad_crossing_beams");
  SetAllAngleMomentWeights(nh_c, solid_angles, h_weights,
                           1, nangles, upper_tx, upper_ty, 0.0,
                           crossing_beams.flux_fraction, "rad_crossing_beams");
  Kokkos::deep_copy(*(crossing_beams.angular_weights), h_weights);
  crossing_beams.lower_profile_qx = lower_tx;
  crossing_beams.lower_profile_qy = lower_ty;
  crossing_beams.upper_profile_qx = upper_tx;
  crossing_beams.upper_profile_qy = upper_ty;

  if (!(restart)) {
    FillCrossingBeams(pmy_mesh_, false);
  }
}
