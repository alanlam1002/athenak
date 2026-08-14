//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file geometry_cylindrical.cpp
//! \brief Cylindrical (x1,x2,x3)=(R,phi,z) geometry factory, right-handed, general 3D.
//! Formulas ported (math only) from old Athena++'s src/coordinates/cylindrical.cpp,
//! verified against that reference before being written here (see DEVELOPMENT.md Task
//! B1 log): Face1Area=R_f*dphi*dz, Face2Area=dR*dz (no R dependence -- inherited from
//! the flat/Cartesian base case, not overridden in the reference), Face3Area=
//! 0.5*(R_f,+^2-R_f,-^2)*dphi, CellVolume=0.5*(R_f,+^2-R_f,-^2)*dphi*dz, Edge1Length=dR,
//! Edge2Length=R_f*dphi, Edge3Length=dz, volumetric centroid x1v=(2/3)*(R_f,+^3-R_f,-^3)/
//! (R_f,+^2-R_f,-^2) (Mignone 2014 eq. 17), geometric source coefficients src1=dR/Vol_i,
//! src2=dR/((R_f,-+R_f,+)*Vol_i) (Delta-A/Delta-V ratios, NOT 1/x1v -- see v2 plan
//! Correction C2 for why this distinction matters for well-balancedness).
//!
//! Ghost-zone note: R_f(i) is computed via the same LeftEdgeX() linear-extrapolation
//! formula used for i in the active range, with NO explicit mirroring/reflection for
//! ghost indices below R=0. This exactly matches old Athena++'s cylindrical.cpp (which
//! also has no ghost-zone special-casing) and is safe here because every "weighted"
//! (R-dependent-squared) factor -- a3i, vi, src1, src2 -- is only ever read at ACTIVE
//! cell indices (is..ie) by the flux-divergence/CT kernels (Task A3/A4/D1), never at
//! ghost indices; only the volumetric centroid x1v is read into the ghost region (by
//! reconstruction, Task B6), and the raw-signed formula naturally gives the physically
//! correct (negative) centroid there, which is exactly what's needed for a correct
//! gradient across the R=0 reflecting boundary.

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/meshblock_pack.hpp"
#include "cell_locations.hpp"
#include "mesh_geometry.hpp"

namespace {
int NCells(int nx, int ng) { return (nx > 1) ? (nx + 2*ng) : 1; }

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

void BuildCylindricalGeometry(ParameterInput *pin, MeshBlockPack *ppack, GeomData &geom) {
  auto &indcs = ppack->pmesh->mb_indcs;
  int nmb = ppack->nmb_thispack;
  int ng = indcs.ng;
  int ncells1 = indcs.nx1 + 2*ng;
  int ncells2 = NCells(indcs.nx2, ng);
  int ncells3 = NCells(indcs.nx3, ng);
  int is = indcs.is;

  auto &size = ppack->pmb->mb_size;

  // raw (possibly-negative-in-ghost-zone) radial face position -- see file docstring
  auto rf = [&](int m, int i) {
    return LeftEdgeX(i - is, indcs.nx1, size.h_view(m).x1min, size.h_view(m).x1max);
  };
  auto dx2_of = [&](int m, int) { return size.h_view(m).dx2; };
  auto dx3_of = [&](int m, int) { return size.h_view(m).dx3; };
  auto one_of = [&](int, int) { return static_cast<Real>(1.0); };

  // R_f(i): face factor for Area1/Len2 (own-direction of Area1, transverse of Len2)
  auto Rf_of = [&](int m, int i) { return rf(m, i); };
  // dR(i) = R_f(i+1)-R_f(i): plain cell width, for Area2/Len1
  auto dR_of = [&](int m, int i) { return rf(m, i+1) - rf(m, i); };
  // 0.5*(R_f(i+1)^2 - R_f(i)^2): weighted cell integral, for Area3/Vol
  auto Rmom_of = [&](int m, int i) {
    Real rm = rf(m, i), rp = rf(m, i+1);
    return 0.5*(rp*rp - rm*rm);
  };
  auto x1v_of = [&](int m, int i) {
    Real rm = rf(m, i), rp = rf(m, i+1);
    Real rp2 = rp*rp, rm2 = rm*rm;
    return (2.0/3.0)*(rp2*rp - rm2*rm)/(rp2 - rm2);
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
  // face positions (Task B6); xf1 = Rf_of (already computed above, R-direction face)
  auto xf2_of = [&](int m, int j) {
    int n2 = (indcs.nx2 > 1) ? indcs.nx2 : 1;
    return LeftEdgeX(j - indcs.js, n2, size.h_view(m).x2min, size.h_view(m).x2max);
  };
  auto xf3_of = [&](int m, int k) {
    int n3 = (indcs.nx3 > 1) ? indcs.nx3 : 1;
    return LeftEdgeX(k - indcs.ks, n3, size.h_view(m).x3min, size.h_view(m).x3max);
  };
  // geometric source coefficients (Delta-A/Delta-V form, Task C1 will consume these)
  auto src1_of = [&](int m, int i) { return dR_of(m, i) / Rmom_of(m, i); };
  auto src2_of = [&](int m, int i) {
    Real rm = rf(m, i), rp = rf(m, i+1);
    return dR_of(m, i) / ((rm + rp) * Rmom_of(m, i));
  };

  // Area1 = R_f(i) * dphi(j) * dz(k)
  geom.a1i = BuildFactor("geom.a1i", nmb, ncells1+1, Rf_of);
  geom.a1j = BuildFactor("geom.a1j", nmb, ncells2,   dx2_of);
  geom.a1k = BuildFactor("geom.a1k", nmb, ncells3,   dx3_of);
  // Area2 = dR(i) * 1 * dz(k)  (no phi dependence, inherited flat formula)
  geom.a2i = BuildFactor("geom.a2i", nmb, ncells1,   dR_of);
  geom.a2j = BuildFactor("geom.a2j", nmb, ncells2+1, one_of);
  geom.a2k = BuildFactor("geom.a2k", nmb, ncells3,   dx3_of);
  // Area3 = 0.5*(R_f,+^2-R_f,-^2)(i) * dphi(j) * 1
  geom.a3i = BuildFactor("geom.a3i", nmb, ncells1,   Rmom_of);
  geom.a3j = BuildFactor("geom.a3j", nmb, ncells2,   dx2_of);
  geom.a3k = BuildFactor("geom.a3k", nmb, ncells3+1, one_of);
  // Vol = 0.5*(R_f,+^2-R_f,-^2)(i) * dphi(j) * dz(k)
  geom.vi = BuildFactor("geom.vi", nmb, ncells1, Rmom_of);
  geom.vj = BuildFactor("geom.vj", nmb, ncells2, dx2_of);
  geom.vk = BuildFactor("geom.vk", nmb, ncells3, dx3_of);
  // Len1 = dR(i) * 1 * 1
  geom.l1i = BuildFactor("geom.l1i", nmb, ncells1,   dR_of);
  geom.l1j = BuildFactor("geom.l1j", nmb, ncells2+1, one_of);
  geom.l1k = BuildFactor("geom.l1k", nmb, ncells3+1, one_of);
  // Len2 = R_f(i) * dphi(j) * 1
  geom.l2i = BuildFactor("geom.l2i", nmb, ncells1+1, Rf_of);
  geom.l2j = BuildFactor("geom.l2j", nmb, ncells2,   dx2_of);
  geom.l2k = BuildFactor("geom.l2k", nmb, ncells3+1, one_of);
  // Len3 = 1 * 1 * dz(k)
  geom.l3i = BuildFactor("geom.l3i", nmb, ncells1+1, one_of);
  geom.l3j = BuildFactor("geom.l3j", nmb, ncells2+1, one_of);
  geom.l3k = BuildFactor("geom.l3k", nmb, ncells3,   dx3_of);
  // volumetric centroids
  geom.x1v = BuildFactor("geom.x1v", nmb, ncells1, x1v_of);
  geom.x2v = BuildFactor("geom.x2v", nmb, ncells2, x2v_of);
  geom.x3v = BuildFactor("geom.x3v", nmb, ncells3, x3v_of);
  // face positions
  geom.xf1 = BuildFactor("geom.xf1", nmb, ncells1+1, Rf_of);
  geom.xf2 = BuildFactor("geom.xf2", nmb, ncells2+1, xf2_of);
  geom.xf3 = BuildFactor("geom.xf3", nmb, ncells3+1, xf3_of);
  // geometric source-term coefficients
  geom.src1 = BuildFactor("geom.src1", nmb, ncells1, src1_of);
  geom.src2 = BuildFactor("geom.src2", nmb, ncells1, src2_of);
  // CFL-purpose cell widths (Task B5): CenterWidth2 = R_v(i)*dphi(j) (angular, weighted
  // by the radial CENTROID x1v, matching old Athena++'s CenterWidth2 exactly -- NOT the
  // face value used by Len2/Edge2Length); CenterWidth3 = dz(k), flat (no R dependence).
  geom.cw2i = BuildFactor("geom.cw2i", nmb, ncells1, x1v_of);
  geom.cw2j = BuildFactor("geom.cw2j", nmb, ncells2, dx2_of);
  geom.cw3i = BuildFactor("geom.cw3i", nmb, ncells1, one_of);
  geom.cw3j = BuildFactor("geom.cw3j", nmb, ncells2, one_of);
  geom.cw3k = BuildFactor("geom.cw3k", nmb, ncells3, dx3_of);

  // PPM4/PPMX x1 interpolation weights (Task B7): Mignone (2014) eq. B.9, m_coord=1
  // (cylindrical Jacobian ~ R^1). io = xf1(face)/dR is the dimensionless LOCAL radius in
  // units of the (uniform -- AthenaK has no mesh-stretching feature at all, so dR is
  // exactly constant) grid spacing, evaluated at the SIGNED face position -- NOT
  // old Athena++'s io=abs(face_index-is), which implicitly assumes x1min=0 exactly.
  // Using the actual position instead of an index offset is what makes this correct for
  // annulus domains (x1min>0) too, not just origin-touching ones: verified by checking
  // the io-->infinity limit reduces exactly to the flat (-1/12,7/12,7/12,-1/12) weights
  // (see DEVELOPMENT.md Task B7 log for the algebraic check), which would NOT hold if io
  // were measured from an arbitrary index rather than the true r=0. Ghost-zone values
  // near a REFLECTING wall (including r=0 itself) are fixed up separately by
  // MirrorReflectingGhostGeometry's PPM-coefficient extension (mesh_geometry.cpp) --
  // this formula only needs to give a well-defined, smoothly-continued value everywhere,
  // matching the same division of labor established for x1v/xf1 in Task B6.
  auto ppm_io_of = [&](int m, int i) { return rf(m, i) / size.h_view(m).dx1; };
  auto ppm_c1_of = [&](int m, int i) {
    Real io = ppm_io_of(m, i);
    Real io2 = io*io, io3 = io2*io;
    Real delta = 120.0*io2*io2 - 360.0*io2 + 96.0;
    return -(2.0*io - 3.0)*(5.0*io3 + 8.0*io2 - 3.0*io - 4.0)/delta;
  };
  auto ppm_c2_of = [&](int m, int i) {
    Real io = ppm_io_of(m, i);
    Real io2 = io*io, io3 = io2*io;
    Real delta = 120.0*io2*io2 - 360.0*io2 + 96.0;
    return (2.0*io - 1.0)*(35.0*io3 + 24.0*io2 - 93.0*io - 60.0)/delta;
  };
  auto ppm_c3_of = [&](int m, int i) {
    Real io = ppm_io_of(m, i);
    Real io2 = io*io, io3 = io2*io;
    Real delta = 120.0*io2*io2 - 360.0*io2 + 96.0;
    return (2.0*io + 1.0)*(35.0*io3 - 24.0*io2 - 93.0*io + 60.0)/delta;
  };
  auto ppm_c4_of = [&](int m, int i) {
    Real io = ppm_io_of(m, i);
    Real io2 = io*io, io3 = io2*io;
    Real delta = 120.0*io2*io2 - 360.0*io2 + 96.0;
    return -(2.0*io + 3.0)*(5.0*io3 - 8.0*io2 - 3.0*io + 4.0)/delta;
  };
  // Mignone eq. 48 overshoot-limiter ratios, cylindrical (using |x1v|, the CENTROID --
  // matches old Athena++'s explicit choice to take the absolute value here since the
  // ghost region can have a formally-negative centroid, and this ratio needs a physical
  // positive radius).
  auto ppm_hp_of = [&](int m, int i) {
    Real xv = std::abs(x1v_of(m, i));
    Real dxi = dR_of(m, i);
    Real h_plus = 3.0 + dxi/(2.0*xv);
    Real h_minus = 3.0 - dxi/(2.0*xv);
    return (h_plus + 1.0)/(h_minus - 1.0);
  };
  auto ppm_hm_of = [&](int m, int i) {
    Real xv = std::abs(x1v_of(m, i));
    Real dxi = dR_of(m, i);
    Real h_plus = 3.0 + dxi/(2.0*xv);
    Real h_minus = 3.0 - dxi/(2.0*xv);
    return (h_minus + 1.0)/(h_plus - 1.0);
  };
  geom.ppm_c1i = BuildFactor("geom.ppm_c1i", nmb, ncells1+1, ppm_c1_of);
  geom.ppm_c2i = BuildFactor("geom.ppm_c2i", nmb, ncells1+1, ppm_c2_of);
  geom.ppm_c3i = BuildFactor("geom.ppm_c3i", nmb, ncells1+1, ppm_c3_of);
  geom.ppm_c4i = BuildFactor("geom.ppm_c4i", nmb, ncells1+1, ppm_c4_of);
  geom.ppm_hpi = BuildFactor("geom.ppm_hpi", nmb, ncells1, ppm_hp_of);
  geom.ppm_hmi = BuildFactor("geom.ppm_hmi", nmb, ncells1, ppm_hm_of);
}
