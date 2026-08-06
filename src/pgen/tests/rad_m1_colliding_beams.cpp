//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_m1_colliding_beams.cpp
//! \brief grey M1 colliding-beams test (arXiv:2302.04283 Section 3.1)
//!
//! Two noninteracting beams cross in flat spacetime, injected near the left edge of
//! the domain at central angles +/-beam_angle relative to +x (matching the geometry
//! of the DO module's ported RadiationCrossingBeams, rad_beam.cpp). M1 tracks only a
//! single (E, F_d) pair per cell -- it has no angular resolution at all -- so wherever
//! the two beams' paths overlap, the solver can only represent their vector-summed
//! flux, which looks like one beam pointing in the averaged direction. This is
//! precisely the failure mode the paper's Section 3.1 describes ("Two beams crossing
//! in vacuum will merge into a single beam... when using M1"); this pgen exists to
//! measure that merging directly, not just cite it.
//!
//! Simplification (documented, not hidden): the paper's sources are small circular
//! regions strictly inside the domain (radius 1/10 at x=2/15). M1 has no existing
//! interior volumetric point-source mechanism; rather than add a new, more invasive
//! one, this reuses the same boundary-ghost-cell injection pattern already
//! established and validated by the 2D beam tests (ApplyBeamSources2D,
//! radiation_m1_beams.cpp) -- both beam sources sit at the domain's left edge
//! (ix1_bc=outflow) instead of at x=2/15 in the interior. This does not change the
//! qualitative physics being tested (whether the two beams merge downstream of
//! their sources) -- only the sources' exact x-position.

#include <cmath>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "radiation_m1/radiation_m1.hpp"
#include "radiation_m1/radiation_m1_helpers.hpp"

namespace {

struct M1CollidingBeamData {
  bool enabled = false;
  Real y_lower = 2.0 / 15.0;
  Real y_upper = 13.0 / 15.0;
  Real band_halfwidth = 0.05;
  // Heap-allocated, deliberately never freed (same idiom as ~/athenak_IAS's
  // crossing_beams.angular_weights, rad_beam.cpp): these are namespace-scope
  // globals, so a plain (non-pointer) Kokkos View member would be destructed
  // at static-storage-duration cleanup time, which runs *after*
  // Kokkos::finalize() -- confirmed to trigger a "Kokkos allocation is being
  // deallocated after Kokkos::finalize was called" warning/backtrace at exit
  // when tried as plain members. A `new`'d pointer's pointee is never
  // destructed at all, sidestepping the ordering problem entirely.
  DvceArray1D<Real> *lower_vals = nullptr;  // size 4: E, Fx, Fy, Fz
  DvceArray1D<Real> *upper_vals = nullptr;  // size 4: E, Fx, Fy, Fz
};

M1CollidingBeamData m1_colliding_beams;

}  // namespace

// Injects the two colliding-beam states into the ix1_bc=outflow ghost zones, in
// their respective (non-overlapping) y-bands -- same mechanism/convention as
// ApplyBeamSources2D (radiation_m1_beams.cpp), generalized to two simultaneous beams.
void ApplyM1CollidingBeamSources2D(Mesh *pmesh) {
  if (!(m1_colliding_beams.enabled)) {
    return;
  }
  auto &indcs = pmesh->mb_indcs;
  int &is = indcs.is;
  int &js = indcs.js;

  int nmb1 = pmesh->pmb_pack->nmb_thispack - 1;
  auto nvars_ = pmesh->pmb_pack->pradm1->nvars;
  auto &nspecies_ = pmesh->pmb_pack->pradm1->nspecies;
  auto &mb_bcs = pmesh->pmb_pack->pmb->mb_bcs;
  auto &size = pmesh->pmb_pack->pmb->mb_size;

  int &ng = indcs.ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * ng) : 1;

  auto &u0_ = pmesh->pmb_pack->pradm1->u0;
  auto lower_vals_ = *(m1_colliding_beams.lower_vals);
  auto upper_vals_ = *(m1_colliding_beams.upper_vals);
  Real y_lower_ = m1_colliding_beams.y_lower;
  Real y_upper_ = m1_colliding_beams.y_upper;
  Real hw_ = m1_colliding_beams.band_halfwidth;

  par_for(
      "radiation_m1_colliding_beams_populate_2d", DevExeSpace(), 0, nmb1, 0, nvars_-2, 0,
      (n3 - 1), 0, (n2 - 1), KOKKOS_LAMBDA(int m, int n, int k, int j) {
        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        int nx2 = indcs.nx2;
        Real x2 = CellCenterX(j - js, nx2, x2min, x2max);

        switch (mb_bcs.d_view(m, BoundaryFace::inner_x1)) {
          case BoundaryFlag::outflow:
            if (fabs(x2 - y_lower_) <= hw_) {
              for (int i = 0; i < ng; ++i) {
                for (int nuidx = 0; nuidx < nspecies_; nuidx++) {
                  u0_(m, radiationm1::CombinedIdx(nuidx, n, nvars_), k, j, is - i - 1) =
                      lower_vals_(n);
                }
              }
            } else if (fabs(x2 - y_upper_) <= hw_) {
              for (int i = 0; i < ng; ++i) {
                for (int nuidx = 0; nuidx < nspecies_; nuidx++) {
                  u0_(m, radiationm1::CombinedIdx(nuidx, n, nvars_), k, j, is - i - 1) =
                      upper_vals_(n);
                }
              }
            }
            break;
          default:
            break;
        }
      });
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::RadiationM1CrossingBeams(ParameterInput *pin)
//! \brief Sets initial/boundary data for the grey M1 colliding-beams test

void ProblemGenerator::RadiationM1CrossingBeams(ParameterInput *pin,
                                                const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (pmbp->pradm1 == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "The colliding-beams problem generator can only be run with "
                 "radiation-m1, but no <radiation_m1> block in input file"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!(pmbp->pmesh->two_d)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "The colliding-beams problem generator requires a 2D mesh"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is;
  int &ie = indcs.ie;
  int &js = indcs.js;
  int &je = indcs.je;
  int isg = is - indcs.ng;
  int ieg = ie + indcs.ng;
  int jsg = js - indcs.ng;
  int jeg = je + indcs.ng;
  int nmb = pmbp->nmb_thispack;
  adm::ADM::ADM_vars &adm = pmbp->padm->adm;
  DvceArray5D<Real> w0_ = pmbp->pradm1->w0;

  // flat, static background -- matches the DO-side crossing-beams test (flat
  // spacetime), and the existing M1 beam tests' minkowski branch
  par_for(
      "pgen_colliding_beams_metric", DevExeSpace(), 0, nmb - 1, 0, 0, jsg, jeg, isg,
      ieg, KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        for (int a = 0; a < 3; ++a)
          for (int b = a; b < 3; ++b) {
            adm.g_dd(m, a, b, k, j, i) = (a == b ? 1. : 0.);
          }
        adm.psi4(m, k, j, i) = 1.;
        adm.alpha(m, k, j, i) = 1.;
        w0_(m, IVX, k, j, i) = 0.;
        w0_(m, IVY, k, j, i) = 0.;
        w0_(m, IVZ, k, j, i) = 0.;
      });

  Real beam_E = pin->GetOrAddReal("problem", "beam_E", 1.0);
  Real beam_angle = pin->GetOrAddReal("problem", "beam_angle", M_PI / 6.0);
  m1_colliding_beams.y_lower =
      pin->GetOrAddReal("problem", "beam_y_lower", 2.0 / 15.0);
  m1_colliding_beams.y_upper =
      pin->GetOrAddReal("problem", "beam_y_upper", 13.0 / 15.0);
  m1_colliding_beams.band_halfwidth =
      pin->GetOrAddReal("problem", "beam_band_halfwidth", 0.05);

  // lower-source beam points up-and-right (+beam_angle above +x); upper-source beam
  // points down-and-right (-beam_angle below +x) -- the two converge and cross in
  // the domain interior, matching the DO-side geometry (RadiationCrossingBeams).
  AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_uu{};
  g_uu(0, 0) = -1;
  g_uu(1, 1) = g_uu(2, 2) = g_uu(3, 3) = 1;

  if (m1_colliding_beams.lower_vals == nullptr) {
    m1_colliding_beams.lower_vals = new DvceArray1D<Real>();
  }
  if (m1_colliding_beams.upper_vals == nullptr) {
    m1_colliding_beams.upper_vals = new DvceArray1D<Real>();
  }
  Kokkos::realloc(*(m1_colliding_beams.lower_vals), 4);
  Kokkos::realloc(*(m1_colliding_beams.upper_vals), 4);
  HostArray1D<Real> lower_host, upper_host;
  Kokkos::realloc(lower_host, 4);
  Kokkos::realloc(upper_host, 4);

  {
    AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
    Real Fx = beam_E * cos(beam_angle);
    Real Fy = beam_E * sin(beam_angle);
    pack_F_d(0, 0, 0, Fx, Fy, 0., F_d);
    Real E = beam_E;
    apply_floor(g_uu, E, F_d, pmbp->pradm1->params);
    lower_host(M1_E_IDX) = E;
    lower_host(M1_FX_IDX) = F_d(1);
    lower_host(M1_FY_IDX) = F_d(2);
    lower_host(M1_FZ_IDX) = F_d(3);
  }
  {
    AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
    Real Fx = beam_E * cos(-beam_angle);
    Real Fy = beam_E * sin(-beam_angle);
    pack_F_d(0, 0, 0, Fx, Fy, 0., F_d);
    Real E = beam_E;
    apply_floor(g_uu, E, F_d, pmbp->pradm1->params);
    upper_host(M1_E_IDX) = E;
    upper_host(M1_FX_IDX) = F_d(1);
    upper_host(M1_FY_IDX) = F_d(2);
    upper_host(M1_FZ_IDX) = F_d(3);
  }
  Kokkos::deep_copy(*(m1_colliding_beams.lower_vals), lower_host);
  Kokkos::deep_copy(*(m1_colliding_beams.upper_vals), upper_host);
  m1_colliding_beams.enabled = true;

  user_bcs = true;
  user_bcs_func = ApplyM1CollidingBeamSources2D;

  if (restart) {
    return;
  }
  return;
}
