//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mesh_geometry.cpp
//! \brief implementation of MeshGeometry constructor. Dispatches (once, host-side, at
//! construction/regrid time -- never inside a hot per-cell loop, see mesh_geometry.hpp)
//! to exactly one geometry factory function based on Mesh::coord_general.

#include <cmath>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/meshblock_pack.hpp"
#include "mesh_geometry.hpp"

// geometry factory functions, one per coordinate system (declared here, defined in
// geometry_<system>.cpp -- Task A2 implements cartesian, Task B1 cylindrical; Task
// B2/B3 add cylindrical_axisym/spherical_polar)
void BuildCartesianGeometry(ParameterInput *pin, MeshBlockPack *ppack, GeomData &geom);
void BuildCylindricalGeometry(ParameterInput *pin, MeshBlockPack *ppack, GeomData &geom);
void BuildCylindricalAxisymGeometry(ParameterInput *pin, MeshBlockPack *ppack,
                                     GeomData &geom);
void BuildSphericalGeometry(ParameterInput *pin, MeshBlockPack *ppack, GeomData &geom);

namespace {
//----------------------------------------------------------------------------------------
//! \fn MirrorReflectingGhostGeometry()
//! \brief Task B6 fix: PLM reads ghost-zone centroid (xv) and face (xf) positions for the
//! one/two ghost cells adjacent to the active domain. Those positions are filled by the
//! geometry factories via the SAME coordinate-formula linear extrapolation used for
//! active cells -- correct and even naturally mirror-symmetric at a coordinate
//! SINGULARITY like r=0 (verified in the B1-B3 logs), but NOT mirror-symmetric at a
//! generic reflecting wall away from any such symmetry point (e.g. an outer reflecting
//! boundary at R=2.0): linear extrapolation just continues OUTWARD there, while the
//! reflecting BC mirrors the physics DATA in the ghost zone. That mismatch between
//! "ghost geometry continuing outward" and "ghost data mirrored inward" breaks the
//! exact-zero-flux-at-a-reflecting-wall property PLM's reconstruction depends on,
//! causing a small (not roundoff) mass leak -- discovered via Task B4's conservation
//! tests failing at the ~1e-6 relative level after this task's PLM change, traced to
//! non-origin reflecting walls specifically (see DEVELOPMENT.md Task B6 log for the
//! full derivation). Fix: overwrite ghost-zone xv/xf with the TRUE mirror of the
//! corresponding active-cell geometry, for any (per-MeshBlock, via mb_bcs -- so this is
//! correct for multi-block decompositions too) face whose BC is reflect.
void MirrorReflectingGhostGeometry(MeshBlockPack *ppack, DvceArray2D<Real> &xv,
                                    DvceArray2D<Real> &xf, int is, int ie, int ng,
                                    BoundaryFace inner_face, BoundaryFace outer_face) {
  int nmb = ppack->nmb_thispack;
  auto &mb_bcs = ppack->pmb->mb_bcs;
  auto xv_h = Kokkos::create_mirror_view(xv);
  auto xf_h = Kokkos::create_mirror_view(xf);
  Kokkos::deep_copy(xv_h, xv);
  Kokkos::deep_copy(xf_h, xf);
  bool changed = false;
  for (int m = 0; m < nmb; ++m) {
    if (mb_bcs.h_view(m, inner_face) == BoundaryFlag::reflect) {
      Real wall = xf_h(m, is);
      for (int g = 1; g <= ng; ++g) {
        int ghost_i = is - g;
        int mirror_i = is + g - 1;
        xv_h(m, ghost_i) = 2.0*wall - xv_h(m, mirror_i);
        xf_h(m, ghost_i) = 2.0*wall - xf_h(m, mirror_i + 1);
      }
      changed = true;
    }
    if (mb_bcs.h_view(m, outer_face) == BoundaryFlag::reflect) {
      Real wall = xf_h(m, ie + 1);
      for (int g = 1; g <= ng; ++g) {
        int ghost_i = ie + g;
        int mirror_i = ie - g + 1;
        xv_h(m, ghost_i) = 2.0*wall - xv_h(m, mirror_i);
        xf_h(m, ghost_i + 1) = 2.0*wall - xf_h(m, mirror_i);
      }
      changed = true;
    }
  }
  if (changed) {
    Kokkos::deep_copy(xv, xv_h);
    Kokkos::deep_copy(xf, xf_h);
  }
}

//----------------------------------------------------------------------------------------
//! \fn MirrorReflectingGhostPpmCoeffs()
//! \brief Task B7 analogue of MirrorReflectingGhostGeometry, for the x1 PPM4/PPMX
//! interpolation weights (ppm_c1i..c4i, face-indexed) and overshoot ratios (ppm_hpi/hmi,
//! cell-indexed). Verified algebraically (see DEVELOPMENT.md Task B7 log) that the
//! analytic Mignone eq. B.9/B.14 formulas, evaluated at the SIGNED local io=r/dr
//! (not an index offset from `is`), automatically satisfy c1(io)=c4(-io) and
//! c2(io)=c3(-io) -- i.e. the correct "mirror + reverse stencil order" relationship --
//! for free at a genuine coordinate singularity (r=0, io_wall=0), needing no explicit
//! fix there. That symmetry is specific to io_wall=0 and does NOT hold at a generic
//! reflecting wall away from the origin (e.g. an outer boundary, or an inner boundary of
//! an annulus at r0>0), for exactly the same underlying reason ghost xv/xf needed an
//! explicit fix in Task B6: the analytic formula continues smoothly past a non-origin
//! wall rather than mirroring. Fix: for such walls, overwrite ghost-face coefficients
//! with the REVERSED-order coefficients of the mirror-partner active face (c1<->c4,
//! c2<->c3), and ghost-cell ratios by SWAPPING hp<->hm with the mirror-partner active
//! cell (h_plus/h_minus are defined via a left/right asymmetry that swaps under
//! reflection).
void MirrorReflectingGhostPpmCoeffs(MeshBlockPack *ppack,
                                     DvceArray2D<Real> &c1, DvceArray2D<Real> &c2,
                                     DvceArray2D<Real> &c3, DvceArray2D<Real> &c4,
                                     DvceArray2D<Real> &hp, DvceArray2D<Real> &hm,
                                     int is, int ie, int ng,
                                     BoundaryFace inner_face, BoundaryFace outer_face) {
  int nmb = ppack->nmb_thispack;
  auto &mb_bcs = ppack->pmb->mb_bcs;
  auto c1_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), c1);
  auto c2_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), c2);
  auto c3_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), c3);
  auto c4_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), c4);
  auto hp_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), hp);
  auto hm_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), hm);
  bool changed = false;
  for (int m = 0; m < nmb; ++m) {
    if (mb_bcs.h_view(m, inner_face) == BoundaryFlag::reflect) {
      for (int g = 1; g <= ng; ++g) {
        int ghost_face = is - g;      // ghost face index
        int mirror_face = is + g;     // mirror-partner active face index
        c1_h(m, ghost_face) = c4_h(m, mirror_face);
        c2_h(m, ghost_face) = c3_h(m, mirror_face);
        c3_h(m, ghost_face) = c2_h(m, mirror_face);
        c4_h(m, ghost_face) = c1_h(m, mirror_face);
        int ghost_cell = is - g;
        int mirror_cell = is + g - 1;
        hp_h(m, ghost_cell) = hm_h(m, mirror_cell);
        hm_h(m, ghost_cell) = hp_h(m, mirror_cell);
      }
      changed = true;
    }
    if (mb_bcs.h_view(m, outer_face) == BoundaryFlag::reflect) {
      for (int g = 1; g <= ng; ++g) {
        int ghost_face = ie + 1 + g;
        int mirror_face = ie + 1 - g;
        c1_h(m, ghost_face) = c4_h(m, mirror_face);
        c2_h(m, ghost_face) = c3_h(m, mirror_face);
        c3_h(m, ghost_face) = c2_h(m, mirror_face);
        c4_h(m, ghost_face) = c1_h(m, mirror_face);
        int ghost_cell = ie + g;
        int mirror_cell = ie - g + 1;
        hp_h(m, ghost_cell) = hm_h(m, mirror_cell);
        hm_h(m, ghost_cell) = hp_h(m, mirror_cell);
      }
      changed = true;
    }
  }
  if (changed) {
    Kokkos::deep_copy(c1, c1_h);
    Kokkos::deep_copy(c2, c2_h);
    Kokkos::deep_copy(c3, c3_h);
    Kokkos::deep_copy(c4, c4_h);
    Kokkos::deep_copy(hp, hp_h);
    Kokkos::deep_copy(hm, hm_h);
  }
}
//----------------------------------------------------------------------------------------
//! \fn BuildPlmFactors()
//! \brief fills one direction's precomputed PLM limiter factors from that direction's
//! centroid (xv) and face (xf) positions. Deliberately generic: it derives everything
//! from positions via MakePlmCoeff(), so it is written ONCE here rather than duplicated
//! in each of the four coordinate factories, and any new coordinate system gets correct
//! PLM factors for free just by filling xv/xf.
//!
//! MUST run after MirrorReflectingGhostGeometry(), which adjusts xv/xf in ghost zones at
//! reflecting walls -- the factors have to be derived from the corrected positions.

//! Returns true if the direction turned out to be uniformly spaced (see
//! GeomData::plm_uniform1/2/3), in which case the stored factors are snapped to the
//! exact uniform constants first.

bool BuildPlmFactors(DvceArray3D<Real> &arr, const DvceArray2D<Real> &xv,
                     const DvceArray2D<Real> &xf, const std::string &label) {
  int nmb = xv.extent_int(0);
  int n = xv.extent_int(1);
  arr = DvceArray3D<Real>(label, nmb, NPLM_COEF, n);
  auto arr_h = Kokkos::create_mirror_view(arr);
  auto xv_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), xv);
  auto xf_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), xf);
  for (int m = 0; m < nmb; ++m) {
    for (int idx = 0; idx < n; ++idx) {
      // the outermost cells have no i-1/i+1 centroid stencil (and a 1-cell direction has
      // no stencil at all); they are never reconstructed, so use the uniform factors
      PlmCoeff c = (idx >= 1 && idx <= n-2)
          ? MakePlmCoeff(xv_h(m,idx-1), xv_h(m,idx), xv_h(m,idx+1),
                         xf_h(m,idx), xf_h(m,idx+1))
          : UniformPlmCoeff();
      arr_h(m,IPLM_PF,idx) = c.pF;
      arr_h(m,IPLM_PB,idx) = c.pB;
      arr_h(m,IPLM_CF,idx) = c.cf;
      arr_h(m,IPLM_CB,idx) = c.cb;
      arr_h(m,IPLM_CC,idx) = c.cc;
      arr_h(m,IPLM_PP,idx) = c.pP;
      arr_h(m,IPLM_PM,idx) = c.pM;
    }
  }

  // Is this direction uniformly spaced? Compared with a relative tolerance rather than
  // by equality: on a uniform grid dx1f and dx1v_fwd are mathematically equal but are
  // computed from different subtractions, so pF can come out an ulp away from 1.0.
  // Only the interior is examined -- the two end cells are uniform by construction and
  // would otherwise mask a genuinely non-uniform direction.
  PlmCoeff u = UniformPlmCoeff();
  const Real uval[NPLM_COEF] = {u.pF, u.pB, u.cf, u.cb, u.cc, u.pP, u.pM};
  bool uniform = true;
  for (int m = 0; m < nmb && uniform; ++m) {
    for (int idx = 1; idx <= n-2 && uniform; ++idx) {
      for (int c = 0; c < NPLM_COEF; ++c) {
        if (std::abs(arr_h(m,c,idx) - uval[c]) > 1.0e-12*std::abs(uval[c])) {
          uniform = false;
          break;
        }
      }
    }
  }
  // Snap to the exact constants so the flag and the data agree bit for bit. This makes
  // the uniform branch in PLMGeom() reproduce upstream's arithmetic EXACTLY, so a
  // Cartesian run is bitwise identical to a pre-curvilinear build rather than merely
  // close -- which is also what makes that branch safe to take at all.
  if (uniform) {
    for (int m = 0; m < nmb; ++m) {
      for (int idx = 0; idx < n; ++idx) {
        for (int c = 0; c < NPLM_COEF; ++c) { arr_h(m,c,idx) = uval[c]; }
      }
    }
  }

  Kokkos::deep_copy(arr, arr_h);
  return uniform;
}

} // namespace

//----------------------------------------------------------------------------------------
// constructor

MeshGeometry::MeshGeometry(ParameterInput *pin, MeshBlockPack *ppack) :
    pmy_pack(ppack) {
  CoordinateGeneral coord_general = ppack->pmesh->coord_general;
  switch (coord_general) {
    case CoordinateGeneral::cartesian:
      BuildCartesianGeometry(pin, ppack, geom_data);
      break;
    case CoordinateGeneral::cylindrical:
      BuildCylindricalGeometry(pin, ppack, geom_data);
      break;
    case CoordinateGeneral::cylindrical_axisym:
      BuildCylindricalAxisymGeometry(pin, ppack, geom_data);
      break;
    case CoordinateGeneral::spherical_polar:
      BuildSphericalGeometry(pin, ppack, geom_data);
      break;
  }

  // Task B6 fix (see MirrorReflectingGhostGeometry doc comment above): must run after
  // the coordinate-specific factory above has filled x1v/xf1/x2v/xf2/x3v/xf3 for every
  // system, including Cartesian -- reflecting-wall ghost geometry needs this fix there
  // too in principle, though for Cartesian the correction is exactly zero (uniform
  // spacing already has mirror-symmetric ghost geometry under linear extrapolation), so
  // this is a genuine no-op for Cartesian and does not affect any existing behavior.
  auto &indcs = ppack->pmesh->mb_indcs;
  MirrorReflectingGhostGeometry(ppack, geom_data.x1v, geom_data.xf1,
                                 indcs.is, indcs.ie, indcs.ng,
                                 BoundaryFace::inner_x1, BoundaryFace::outer_x1);
  if (ppack->pmesh->multi_d) {
    MirrorReflectingGhostGeometry(ppack, geom_data.x2v, geom_data.xf2,
                                   indcs.js, indcs.je, indcs.ng,
                                   BoundaryFace::inner_x2, BoundaryFace::outer_x2);
  }
  if (ppack->pmesh->three_d) {
    MirrorReflectingGhostGeometry(ppack, geom_data.x3v, geom_data.xf3,
                                   indcs.ks, indcs.ke, indcs.ng,
                                   BoundaryFace::inner_x3, BoundaryFace::outer_x3);
  }

  // Task B7 fix (see MirrorReflectingGhostPpmCoeffs doc comment above): x1-only, since
  // the non-uniform PPM generalization itself is scoped to x1 only (x2/x3 keep the old
  // hardcoded uniform PPM4/PPMX formula, which needs no such fix).
  MirrorReflectingGhostPpmCoeffs(ppack, geom_data.ppm_c1i, geom_data.ppm_c2i,
                                  geom_data.ppm_c3i, geom_data.ppm_c4i,
                                  geom_data.ppm_hpi, geom_data.ppm_hmi,
                                  indcs.is, indcs.ie, indcs.ng,
                                  BoundaryFace::inner_x1, BoundaryFace::outer_x1);

  // Precomputed PLM limiter factors. Built here, after BOTH the coordinate factory and
  // the reflecting-ghost corrections above, since they are derived purely from the final
  // centroid/face positions. Built for all three directions unconditionally: a
  // 1-cell direction just gets the uniform factors, and the reconstruction kernel never
  // reads a direction it does not reconstruct.
  // Is the x1 PPM coefficient set the flat/uniform one? Same tolerance-then-snap
  // approach as BuildPlmFactors (see there); snapping makes the fast path bitwise
  // identical to upstream PPM4/PPMX rather than merely equivalent.
  {
    const Real c1 = -1.0/12.0, c2 = 7.0/12.0, hp = 2.0;
    auto c1_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom_data.ppm_c1i);
    auto c2_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom_data.ppm_c2i);
    auto c3_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom_data.ppm_c3i);
    auto c4_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom_data.ppm_c4i);
    auto hp_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom_data.ppm_hpi);
    auto hm_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom_data.ppm_hmi);
    int nmb = c1_h.extent_int(0);
    int nf = c1_h.extent_int(1);
    int nc = hp_h.extent_int(1);
    bool uni = true;
    auto near = [](Real a, Real b) { return std::abs(a-b) <= 1.0e-12*std::abs(b); };
    for (int m = 0; m < nmb && uni; ++m) {
      for (int i = 0; i < nf && uni; ++i) {
        if (!near(c1_h(m,i),c1) || !near(c2_h(m,i),c2) ||
            !near(c3_h(m,i),c2) || !near(c4_h(m,i),c1)) { uni = false; }
      }
      for (int i = 0; i < nc && uni; ++i) {
        if (!near(hp_h(m,i),hp) || !near(hm_h(m,i),hp)) { uni = false; }
      }
    }
    if (uni) {
      for (int m = 0; m < nmb; ++m) {
        for (int i = 0; i < nf; ++i) {
          c1_h(m,i) = c1; c2_h(m,i) = c2; c3_h(m,i) = c2; c4_h(m,i) = c1;
        }
        for (int i = 0; i < nc; ++i) { hp_h(m,i) = hp; hm_h(m,i) = hp; }
      }
      Kokkos::deep_copy(geom_data.ppm_c1i, c1_h);
      Kokkos::deep_copy(geom_data.ppm_c2i, c2_h);
      Kokkos::deep_copy(geom_data.ppm_c3i, c3_h);
      Kokkos::deep_copy(geom_data.ppm_c4i, c4_h);
      Kokkos::deep_copy(geom_data.ppm_hpi, hp_h);
      Kokkos::deep_copy(geom_data.ppm_hmi, hm_h);
    }
    geom_data.ppm_uniform1 = uni;
  }

  geom_data.plm_uniform1 =
      BuildPlmFactors(geom_data.plm_c1, geom_data.x1v, geom_data.xf1, "geom.plm_c1");
  geom_data.plm_uniform2 =
      BuildPlmFactors(geom_data.plm_c2, geom_data.x2v, geom_data.xf2, "geom.plm_c2");
  geom_data.plm_uniform3 =
      BuildPlmFactors(geom_data.plm_c3, geom_data.x3v, geom_data.xf3, "geom.plm_c3");
}
