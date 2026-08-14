//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file geometry_cylindrical_axisym.cpp
//! \brief Axisymmetric cylindrical (x1,x2)=(R,z) geometry factory. x3 is unused
//! (Mesh::ValidateCoordGeneral() enforces nx3=1); phi is carried as a non-grid
//! rotational component with a "per unit radian" (Delta-phi=1) convention -- see the
//! v2 plan's "Corrections to v1" C1 for why this system exists at all (AthenaK's
//! nested one_d/two_d/three_d/multi_d flags cannot represent nx2=1,nx3>1, so "general
//! cylindrical with phi dropped" is impossible; this is a genuinely separate 2-real-
//! dimension reduction, not a relabeling of geometry_cylindrical.cpp).
//!
//! IMPORTANT handedness note (see plan C1): AthenaK's generic CT/curl machinery assumes
//! a right-handed (x1,x2,x3) system (x1 x/ x2 = x3 direction, standard curl formulas in
//! mhd_ct.cpp). Physically, R-hat x z-hat = -phi-hat (since the standard right-handed
//! cylindrical order is R,phi,z with R-hat x phi-hat = z-hat, i.e. z-hat x R-hat =
//! phi-hat, so R-hat x z-hat = -phi-hat). This means AthenaK's generic "x3"/"e3"/IM3/IB3
//! slot, for this coordinate system, corresponds to MINUS the physical phi-component
//! (v3_AthenaK = -v_phi, B3_AthenaK = -B_phi), not +phi as one might assume. This file
//! (geometry only -- areas/volumes/edge-lengths are orientation-independent positive
//! scalars) is unaffected by the sign choice; it becomes load-bearing starting at Task
//! C1 (geometric source terms) and Task D1 (CT), and is verified by Task D3's PHYSICAL
//! induction tests (field-loop advection), not by the topological D2 div(B)=0 check,
//! which cannot detect a sign error (see plan C6).
//!
//! Formulas (re-derived from physical first principles for the actual (R,z) meaning of
//! x1,x2 here -- NOT a mechanical relabeling of geometry_cylindrical.cpp's (R,phi,z)
//! formulas, since x2's physical meaning changed from phi to z; see DEVELOPMENT.md Task
//! B2 log for the full derivation):
//!   Area1 (R-face, normal=R)   = R_f(i) * dz(j) * 1            [flux through R=const]
//!   Area2 (z-face, normal=z)   = 0.5*(R_f,+^2-R_f,-^2)(i) * 1  [flux through z=const]
//!   Area3 (phi-face, virtual)  = dR(i) * dz(j)                 [flat R-z cross-section]
//!   Vol                        = 0.5*(R_f,+^2-R_f,-^2)(i) * dz(j) * 1
//!   Edge1 (R-direction)        = dR(i)                         [same as general cyl.]
//!   Edge2 (z-direction)        = dz(j)                         [flat, no R dependence]
//!   Edge3 (circumferential)    = R_f(i) * 1                    [no z dependence]
//!   x1v = (2/3)*(R_f,+^3-R_f,-^3)/(R_f,+^2-R_f,-^2)             [Mignone 2014 eq. 17]
//!   src1, src2: identical formulas to geometry_cylindrical.cpp (the geometric-source
//!   physics for the centrifugal/pressure terms is the same whether phi is resolved or
//!   virtual -- Task C1 consumes these).

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

void BuildCylindricalAxisymGeometry(ParameterInput *pin, MeshBlockPack *ppack,
                                     GeomData &geom) {
  auto &indcs = ppack->pmesh->mb_indcs;
  int nmb = ppack->nmb_thispack;
  int ng = indcs.ng;
  int ncells1 = indcs.nx1 + 2*ng;
  int ncells2 = NCells(indcs.nx2, ng);
  int ncells3 = NCells(indcs.nx3, ng);  // always 1 (nx3=1 enforced for this system)
  int is = indcs.is;

  auto &size = ppack->pmb->mb_size;

  // raw (possibly-negative-in-ghost-zone) radial face position -- see the identical
  // note in geometry_cylindrical.cpp: only ever read at active indices except for x1v.
  auto rf = [&](int m, int i) {
    return LeftEdgeX(i - is, indcs.nx1, size.h_view(m).x1min, size.h_view(m).x1max);
  };
  auto dz_of = [&](int m, int) { return size.h_view(m).dx2; };  // x2 = z here
  auto one_of = [&](int, int) { return static_cast<Real>(1.0); };

  auto Rf_of = [&](int m, int i) { return rf(m, i); };
  auto dR_of = [&](int m, int i) { return rf(m, i+1) - rf(m, i); };
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
  // face positions (Task B6); xf1 = Rf_of (R-direction face, above). xf3 is never read
  // by PLM (three_d is always false, nx3=1 enforced) but filled for consistency.
  auto xf2_of = [&](int m, int j) {
    int n2 = (indcs.nx2 > 1) ? indcs.nx2 : 1;
    return LeftEdgeX(j - indcs.js, n2, size.h_view(m).x2min, size.h_view(m).x2max);
  };
  auto xf3_of = [&](int m, int k) {
    int n3 = (indcs.nx3 > 1) ? indcs.nx3 : 1;
    return LeftEdgeX(k - indcs.ks, n3, size.h_view(m).x3min, size.h_view(m).x3max);
  };
  auto src1_of = [&](int m, int i) { return dR_of(m, i) / Rmom_of(m, i); };
  auto src2_of = [&](int m, int i) {
    Real rm = rf(m, i), rp = rf(m, i+1);
    return dR_of(m, i) / ((rm + rp) * Rmom_of(m, i));
  };

  // Area1 (R-face) = R_f(i) * dz(j) * 1
  geom.a1i = BuildFactor("geom.a1i", nmb, ncells1+1, Rf_of);
  geom.a1j = BuildFactor("geom.a1j", nmb, ncells2,   dz_of);
  geom.a1k = BuildFactor("geom.a1k", nmb, ncells3,   one_of);
  // Area2 (z-face) = 0.5*(R_f,+^2-R_f,-^2)(i) * 1 * 1
  geom.a2i = BuildFactor("geom.a2i", nmb, ncells1,   Rmom_of);
  geom.a2j = BuildFactor("geom.a2j", nmb, ncells2+1, one_of);
  geom.a2k = BuildFactor("geom.a2k", nmb, ncells3,   one_of);
  // Area3 (phi-face, virtual) = dR(i) * dz(j) * 1  (flat R-z cross-section)
  geom.a3i = BuildFactor("geom.a3i", nmb, ncells1,   dR_of);
  geom.a3j = BuildFactor("geom.a3j", nmb, ncells2,   dz_of);
  geom.a3k = BuildFactor("geom.a3k", nmb, ncells3+1, one_of);
  // Vol = 0.5*(R_f,+^2-R_f,-^2)(i) * dz(j) * 1
  geom.vi = BuildFactor("geom.vi", nmb, ncells1, Rmom_of);
  geom.vj = BuildFactor("geom.vj", nmb, ncells2, dz_of);
  geom.vk = BuildFactor("geom.vk", nmb, ncells3, one_of);
  // Len1 (R-edge) = dR(i) * 1 * 1
  geom.l1i = BuildFactor("geom.l1i", nmb, ncells1,   dR_of);
  geom.l1j = BuildFactor("geom.l1j", nmb, ncells2+1, one_of);
  geom.l1k = BuildFactor("geom.l1k", nmb, ncells3+1, one_of);
  // Len2 (z-edge) = 1 * dz(j) * 1
  geom.l2i = BuildFactor("geom.l2i", nmb, ncells1+1, one_of);
  geom.l2j = BuildFactor("geom.l2j", nmb, ncells2,   dz_of);
  geom.l2k = BuildFactor("geom.l2k", nmb, ncells3+1, one_of);
  // Len3 (circumferential edge, virtual) = R_f(i) * 1 * 1
  geom.l3i = BuildFactor("geom.l3i", nmb, ncells1+1, Rf_of);
  geom.l3j = BuildFactor("geom.l3j", nmb, ncells2+1, one_of);
  geom.l3k = BuildFactor("geom.l3k", nmb, ncells3,   one_of);
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
  // CFL-purpose cell widths (Task B5): CenterWidth2 = dz(j), flat -- x2 is z here, NOT
  // phi, so unlike geometry_cylindrical.cpp's CenterWidth2 this needs NO R-weighting
  // (moving along z at fixed R doesn't stretch/scale). CenterWidth3 is never read
  // (three_d is always false for this system, nx3=1 enforced), filled with the same
  // trivial placeholder convention used elsewhere in this file for unused x3 slots.
  geom.cw2i = BuildFactor("geom.cw2i", nmb, ncells1, one_of);
  geom.cw2j = BuildFactor("geom.cw2j", nmb, ncells2, dz_of);
  geom.cw3i = BuildFactor("geom.cw3i", nmb, ncells1, one_of);
  geom.cw3j = BuildFactor("geom.cw3j", nmb, ncells2, one_of);
  geom.cw3k = BuildFactor("geom.cw3k", nmb, ncells3, one_of);

  // PPM4/PPMX x1 (R) interpolation weights (Task B7): same m_coord=1 (Mignone eq. B.9)
  // formula as geometry_cylindrical.cpp -- the R-direction Jacobian is the same R^1
  // power law regardless of whether phi is resolved or virtual. See that file for the
  // full derivation/rationale (io = signed local radius in units of dR, not an index
  // offset from is; ghost-zone reflecting-wall fixup handled separately).
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
