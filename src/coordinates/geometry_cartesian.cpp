//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file geometry_cartesian.cpp
//! \brief Cartesian geometry factory: fills GeomData with the values that make the
//! generic Area1/2/3, Vol, Len1/2/3 accessors (see mesh_geometry.hpp) reduce exactly to
//! today's uniform-Cartesian arithmetic (dx2*dx3 for Area1, dx1*dx2*dx3 for Vol, etc.).
//! This is the reference/template implementation for all future geometry factories
//! (see Task B1/B2/B3 for the cylindrical/spherical versions): a single O(N) host-side
//! fill, run once at MeshBlockPack construction (and at every SMR regrid), never inside
//! a hot per-cell kernel.

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/meshblock_pack.hpp"
#include "cell_locations.hpp"
#include "mesh_geometry.hpp"

namespace {
// (nx>1) ? nx+2*ng : 1 -- matches the ncells2/ncells3 convention used throughout the
// code (e.g. src/hydro/hydro.cpp:115-116) for directions that may be inactive (size 1).
int NCells(int nx, int ng) { return (nx > 1) ? (nx + 2*ng) : 1; }

// Allocates a DvceArray2D<Real> of shape (nmb, n), fills it via a host mirror using the
// given host-callable functor value(m,idx), and deep_copies to device. This is the one
// piece of setup-time (not hot-loop) boilerplate shared by every factory function.
template <typename F>
DvceArray2D<Real> BuildFactor(const std::string &label, int nmb, int n, F value) {
  DvceArray2D<Real> arr(label, nmb, n);
  auto arr_h = Kokkos::create_mirror_view(arr);
  for (int m = 0; m < nmb; ++m) {
    for (int idx = 0; idx < n; ++idx) {
      arr_h(m, idx) = value(m, idx);
    }
  }
  Kokkos::deep_copy(arr, arr_h);
  return arr;
}
} // namespace

void BuildCartesianGeometry(ParameterInput *pin, MeshBlockPack *ppack, GeomData &geom) {
  auto &indcs = ppack->pmesh->mb_indcs;
  int nmb = ppack->nmb_thispack;
  int ng = indcs.ng;
  int ncells1 = indcs.nx1 + 2*ng;
  int ncells2 = NCells(indcs.nx2, ng);
  int ncells3 = NCells(indcs.nx3, ng);

  auto &size = ppack->pmb->mb_size;  // DualArray1D<RegionSize>, host view already synced

  // "cell" (width-type) factors: constant = dx_d for every index, per MeshBlock
  auto dx1_of = [&](int m, int) { return size.h_view(m).dx1; };
  auto dx2_of = [&](int m, int) { return size.h_view(m).dx2; };
  auto dx3_of = [&](int m, int) { return size.h_view(m).dx3; };
  // "face" (face-valued) factors: constant = 1 for Cartesian (own-direction area/edge
  // factor has no metric weighting when the direction is flat)
  auto one_of = [&](int, int) { return static_cast<Real>(1.0); };
  // volumetric-centroid position: for Cartesian this is the ordinary CellCenterX() used
  // everywhere else in the code (arithmetic midpoint); ncells_d>1 branches use the real
  // nx_d, the ncells_d==1 (inactive-direction) branch uses n=1 so CellCenterX(0,1,..)
  // returns the domain midpoint -- a harmless placeholder since no reconstruction or
  // area/volume weighting ever varies with that direction's position.
  auto x1v_of = [&](int m, int i) {
    return CellCenterX(i - indcs.is, indcs.nx1, size.h_view(m).x1min,
                       size.h_view(m).x1max);
  };
  auto x2v_of = [&](int m, int j) {
    int n2 = (indcs.nx2 > 1) ? indcs.nx2 : 1;
    int j0 = (indcs.nx2 > 1) ? (j - indcs.js) : 0;
    return CellCenterX(j0, n2, size.h_view(m).x2min, size.h_view(m).x2max);
  };
  auto x3v_of = [&](int m, int k) {
    int n3 = (indcs.nx3 > 1) ? indcs.nx3 : 1;
    int k0 = (indcs.nx3 > 1) ? (k - indcs.ks) : 0;
    return CellCenterX(k0, n3, size.h_view(m).x3min, size.h_view(m).x3max);
  };
  auto zero_of = [&](int, int) { return static_cast<Real>(0.0); };
  // face positions (Task B6), same LeftEdgeX() formula used everywhere else; ncells_d==1
  // (inactive-direction) branches use n=1, matching the x*v_of placeholder convention.
  auto xf1_of = [&](int m, int i) {
    return LeftEdgeX(i - indcs.is, indcs.nx1, size.h_view(m).x1min, size.h_view(m).x1max);
  };
  auto xf2_of = [&](int m, int j) {
    int n2 = (indcs.nx2 > 1) ? indcs.nx2 : 1;
    return LeftEdgeX(j - indcs.js, n2, size.h_view(m).x2min, size.h_view(m).x2max);
  };
  auto xf3_of = [&](int m, int k) {
    int n3 = (indcs.nx3 > 1) ? indcs.nx3 : 1;
    return LeftEdgeX(k - indcs.ks, n3, size.h_view(m).x3min, size.h_view(m).x3max);
  };

  // Area1 = a1i(i)*a1j(j)*a1k(k) = 1 * dx2 * dx3
  geom.a1i = BuildFactor("geom.a1i", nmb, ncells1+1, one_of);
  geom.a1j = BuildFactor("geom.a1j", nmb, ncells2,   dx2_of);
  geom.a1k = BuildFactor("geom.a1k", nmb, ncells3,   dx3_of);
  // Area2 = a2i(i)*a2j(j)*a2k(k) = dx1 * 1 * dx3
  geom.a2i = BuildFactor("geom.a2i", nmb, ncells1,   dx1_of);
  geom.a2j = BuildFactor("geom.a2j", nmb, ncells2+1, one_of);
  geom.a2k = BuildFactor("geom.a2k", nmb, ncells3,   dx3_of);
  // Area3 = a3i(i)*a3j(j)*a3k(k) = dx1 * dx2 * 1
  geom.a3i = BuildFactor("geom.a3i", nmb, ncells1,   dx1_of);
  geom.a3j = BuildFactor("geom.a3j", nmb, ncells2,   dx2_of);
  geom.a3k = BuildFactor("geom.a3k", nmb, ncells3+1, one_of);
  // Vol = dx1 * dx2 * dx3
  geom.vi = BuildFactor("geom.vi", nmb, ncells1, dx1_of);
  geom.vj = BuildFactor("geom.vj", nmb, ncells2, dx2_of);
  geom.vk = BuildFactor("geom.vk", nmb, ncells3, dx3_of);
  // Len1 = dx1 * 1 * 1;  Len2 = 1 * dx2 * 1;  Len3 = 1 * 1 * dx3
  geom.l1i = BuildFactor("geom.l1i", nmb, ncells1,   dx1_of);
  geom.l1j = BuildFactor("geom.l1j", nmb, ncells2+1, one_of);
  geom.l1k = BuildFactor("geom.l1k", nmb, ncells3+1, one_of);
  geom.l2i = BuildFactor("geom.l2i", nmb, ncells1+1, one_of);
  geom.l2j = BuildFactor("geom.l2j", nmb, ncells2,   dx2_of);
  geom.l2k = BuildFactor("geom.l2k", nmb, ncells3+1, one_of);
  geom.l3i = BuildFactor("geom.l3i", nmb, ncells1+1, one_of);
  geom.l3j = BuildFactor("geom.l3j", nmb, ncells2+1, one_of);
  geom.l3k = BuildFactor("geom.l3k", nmb, ncells3,   dx3_of);
  // volumetric centroids
  geom.x1v = BuildFactor("geom.x1v", nmb, ncells1, x1v_of);
  geom.x2v = BuildFactor("geom.x2v", nmb, ncells2, x2v_of);
  geom.x3v = BuildFactor("geom.x3v", nmb, ncells3, x3v_of);
  // face positions
  geom.xf1 = BuildFactor("geom.xf1", nmb, ncells1+1, xf1_of);
  geom.xf2 = BuildFactor("geom.xf2", nmb, ncells2+1, xf2_of);
  geom.xf3 = BuildFactor("geom.xf3", nmb, ncells3+1, xf3_of);
  // geometric source-term coefficients: zero for cartesian (no curvature)
  geom.src1 = BuildFactor("geom.src1", nmb, ncells1, zero_of);
  geom.src2 = BuildFactor("geom.src2", nmb, ncells1, zero_of);
  // CFL-purpose cell widths (Task B5): CenterWidth2=dx2, CenterWidth3=dx3, both flat
  geom.cw2i = BuildFactor("geom.cw2i", nmb, ncells1, one_of);
  geom.cw2j = BuildFactor("geom.cw2j", nmb, ncells2, dx2_of);
  geom.cw3i = BuildFactor("geom.cw3i", nmb, ncells1, one_of);
  geom.cw3j = BuildFactor("geom.cw3j", nmb, ncells2, one_of);
  geom.cw3k = BuildFactor("geom.cw3k", nmb, ncells3, dx3_of);
  // PPM4/PPMX x1 interpolation weights (Task B7): flat/uniform-Cartesian values
  // (Mignone 2014 eq. B.4) everywhere; overshoot ratios = 2.0 (original CW/CS limiters)
  auto ppm_c1_of = [&](int, int) { return static_cast<Real>(-1.0/12.0); };
  auto ppm_c2_of = [&](int, int) { return static_cast<Real>(7.0/12.0); };
  auto ppm_hp_of = [&](int, int) { return static_cast<Real>(2.0); };
  geom.ppm_c1i = BuildFactor("geom.ppm_c1i", nmb, ncells1+1, ppm_c1_of);
  geom.ppm_c2i = BuildFactor("geom.ppm_c2i", nmb, ncells1+1, ppm_c2_of);
  geom.ppm_c3i = BuildFactor("geom.ppm_c3i", nmb, ncells1+1, ppm_c2_of);
  geom.ppm_c4i = BuildFactor("geom.ppm_c4i", nmb, ncells1+1, ppm_c1_of);
  geom.ppm_hpi = BuildFactor("geom.ppm_hpi", nmb, ncells1, ppm_hp_of);
  geom.ppm_hmi = BuildFactor("geom.ppm_hmi", nmb, ncells1, ppm_hp_of);
}
