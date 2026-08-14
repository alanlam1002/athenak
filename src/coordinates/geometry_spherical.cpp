//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file geometry_spherical.cpp
//! \brief Spherical-polar (x1,x2,x3)=(r,theta,phi) geometry factory, right-handed,
//! general 3D. Formulas ported (math only) from old Athena++'s
//! src/coordinates/spherical_polar.cpp, verified against that reference line-by-line
//! before being written here (see DEVELOPMENT.md Task B3 log):
//!   Area1 = r_f^2 * |cos(theta_j)-cos(theta_j+1)| * dphi(k)
//!   Area2 = 0.5*(r_f,+^2-r_f,-^2)(i) * sin(theta_f,j) * dphi(k)
//!   Area3 = 0.5*(r_f,+^2-r_f,-^2)(i) * dtheta(j)
//!   Vol   = (1/3)*(r_f,+^3-r_f,-^3)(i) * |cos(theta_j)-cos(theta_j+1)| * dphi(k)
//!   Edge1 = dr(i)                                    [inherited flat formula]
//!   Edge2 = r_f(i) * dtheta(j)
//!   Edge3 = r_f(i) * sin(theta_f,j) * dphi(k)
//!   x1v = 0.75*(r_f,+^4-r_f,-^4)/(r_f,+^3-r_f,-^3)     [Mignone 2014 eq. 17]
//!   x2v = [(sin th_f,+ - th_f,+ cos th_f,+) - (sin th_f,- - th_f,- cos th_f,-)]
//!         / (cos th_f,- - cos th_f,+)                 [nx2>1 case]; simple midpoint
//!         when nx2==1 (matches the reference's own nx2==1 special case, which is what
//!         the required 1D-radial layout always uses)
//!   src1, src2 = the i-indexed (radial) geometric source coefficients, Delta-A/Delta-V
//!   form, consumed by Task C2. The theta-momentum term additionally needs a
//!   j-indexed coefficient (old code's coord_src1_j_/coord_src3_j_, both numerically
//!   identical: (sin th_f,+ - sin th_f,-)/|cos th_j - cos th_j+1|) -- that field is
//!   added to GeomData when Task C2 is implemented, not needed for pure geometry.
//!
//! Regularity at r=0 (x1min=0): a1i(is)=r_f(is)^2=0 exactly, and x1v's innermost cell
//! reduces to a finite value (0.75*r_+^4/r_+^3 = 0.75*r_+, no 0/0) -- confirmed the same
//! way as geometry_cylindrical.cpp; no special origin handling is needed in this file
//! (Task E1/E2 handle the boundary-condition side). No explicit ghost-zone mirroring is
//! used here either, for the same reason given in geometry_cylindrical.cpp: the
//! r-weighted quantities (a2i/a3i/vi/src1/src2) are only ever read at active indices.

#include <cmath>

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

void BuildSphericalGeometry(ParameterInput *pin, MeshBlockPack *ppack, GeomData &geom) {
  auto &indcs = ppack->pmesh->mb_indcs;
  int nmb = ppack->nmb_thispack;
  int ng = indcs.ng;
  int ncells1 = indcs.nx1 + 2*ng;
  int ncells2 = NCells(indcs.nx2, ng);
  int ncells3 = NCells(indcs.nx3, ng);
  int is = indcs.is, js = indcs.js;

  auto &size = ppack->pmb->mb_size;

  // raw radial face position (see geometry_cylindrical.cpp for the ghost-zone rationale)
  auto rf = [&](int m, int i) {
    return LeftEdgeX(i - is, indcs.nx1, size.h_view(m).x1min, size.h_view(m).x1max);
  };
  // raw polar-angle face position (n=1 case: LeftEdgeX linearly interpolates between
  // x2min at j=js and x2max at j=js+1, same semantics as the nx2>1 case)
  auto thf = [&](int m, int j) {
    int n2 = (indcs.nx2 > 1) ? indcs.nx2 : 1;
    return LeftEdgeX(j - js, n2, size.h_view(m).x2min, size.h_view(m).x2max);
  };
  auto dphi_of = [&](int m, int) { return size.h_view(m).dx3; };
  auto one_of = [&](int, int) { return static_cast<Real>(1.0); };

  auto rf_of = [&](int m, int i) { return rf(m, i); };
  auto dr_of = [&](int m, int i) { return rf(m, i+1) - rf(m, i); };
  auto r2mom_of = [&](int m, int i) {  // 0.5*(r_f,+^2 - r_f,-^2)
    Real rm = rf(m, i), rp = rf(m, i+1);
    return 0.5*(rp*rp - rm*rm);
  };
  auto r3mom_of = [&](int m, int i) {  // (1/3)*(r_f,+^3 - r_f,-^3)
    Real rm = rf(m, i), rp = rf(m, i+1);
    return (1.0/3.0)*(rp*rp*rp - rm*rm*rm);
  };
  auto x1v_of = [&](int m, int i) {
    Real rm = rf(m, i), rp = rf(m, i+1);
    Real rm2 = rm*rm, rp2 = rp*rp;
    return 0.75*(rp2*rp2 - rm2*rm2)/(rp2*rp - rm2*rm);
  };

  // theta-direction: face-valued cos/sin/dtheta, and the |cos(th_j)-cos(th_j+1)| /
  // sin(theta_face) quantities used throughout (matches old code's coord_area1_j_ /
  // coord_area2_j_ exactly, including the abs() -- see spherical_polar.cpp:198-224)
  auto dtheta_of = [&](int m, int j) { return thf(m, j+1) - thf(m, j); };
  auto dcostheta_of = [&](int m, int j) {  // |cos(th_j)-cos(th_j+1)|
    return std::abs(std::cos(thf(m, j)) - std::cos(thf(m, j+1)));
  };
  auto sintheta_face_of = [&](int m, int j) { return std::abs(std::sin(thf(m, j))); };

  auto x2v_of = [&](int m, int j) {
    if (indcs.nx2 == 1) {
      return 0.5*(thf(m, j+1) + thf(m, j));
    }
    Real tm = thf(m, j), tp = thf(m, j+1);
    Real num = (std::sin(tp) - tp*std::cos(tp)) - (std::sin(tm) - tm*std::cos(tm));
    Real den = std::cos(tm) - std::cos(tp);
    return num/den;
  };
  auto x3v_of = [&](int m, int k) {
    int n3 = (indcs.nx3 > 1) ? indcs.nx3 : 1;
    int k0 = (indcs.nx3 > 1) ? (k - indcs.ks) : 0;
    return CellCenterX(k0, n3, size.h_view(m).x3min, size.h_view(m).x3max);
  };
  // face positions (Task B6); xf1 = rf_of (above), xf2 = thf (theta face, already
  // computed above and reused directly, no new lambda needed)
  auto xf3_of = [&](int m, int k) {
    int n3 = (indcs.nx3 > 1) ? indcs.nx3 : 1;
    return LeftEdgeX(k - indcs.ks, n3, size.h_view(m).x3min, size.h_view(m).x3max);
  };

  auto src1_of = [&](int m, int i) { return r2mom_of(m, i) / r3mom_of(m, i); };
  auto src2_of = [&](int m, int i) {
    Real rm = rf(m, i), rp = rf(m, i+1);
    return dr_of(m, i) / ((rm + rp) * r3mom_of(m, i));
  };

  // Area1 = r_f(i)^2 * |cos th_j - cos th_j+1|(j) * dphi(k)
  geom.a1i = BuildFactor("geom.a1i", nmb, ncells1+1,
                          [&](int m, int i) { Real r = rf_of(m, i); return r*r; });
  geom.a1j = BuildFactor("geom.a1j", nmb, ncells2, dcostheta_of);
  geom.a1k = BuildFactor("geom.a1k", nmb, ncells3, dphi_of);
  // Area2 = 0.5*(r_f,+^2-r_f,-^2)(i) * sin(theta_f)(j) * dphi(k)
  geom.a2i = BuildFactor("geom.a2i", nmb, ncells1,   r2mom_of);
  geom.a2j = BuildFactor("geom.a2j", nmb, ncells2+1, sintheta_face_of);
  geom.a2k = BuildFactor("geom.a2k", nmb, ncells3,   dphi_of);
  // Area3 = 0.5*(r_f,+^2-r_f,-^2)(i) * dtheta(j) * 1
  geom.a3i = BuildFactor("geom.a3i", nmb, ncells1,   r2mom_of);
  geom.a3j = BuildFactor("geom.a3j", nmb, ncells2,   dtheta_of);
  geom.a3k = BuildFactor("geom.a3k", nmb, ncells3+1, one_of);
  // Vol = (1/3)*(r_f,+^3-r_f,-^3)(i) * |cos th_j - cos th_j+1|(j) * dphi(k)
  geom.vi = BuildFactor("geom.vi", nmb, ncells1, r3mom_of);
  geom.vj = BuildFactor("geom.vj", nmb, ncells2, dcostheta_of);
  geom.vk = BuildFactor("geom.vk", nmb, ncells3, dphi_of);
  // Len1 = dr(i) * 1 * 1
  geom.l1i = BuildFactor("geom.l1i", nmb, ncells1,   dr_of);
  geom.l1j = BuildFactor("geom.l1j", nmb, ncells2+1, one_of);
  geom.l1k = BuildFactor("geom.l1k", nmb, ncells3+1, one_of);
  // Len2 = r_f(i) * dtheta(j) * 1
  geom.l2i = BuildFactor("geom.l2i", nmb, ncells1+1, rf_of);
  geom.l2j = BuildFactor("geom.l2j", nmb, ncells2,   dtheta_of);
  geom.l2k = BuildFactor("geom.l2k", nmb, ncells3+1, one_of);
  // Len3 = r_f(i) * sin(theta_f)(j) * dphi(k)
  geom.l3i = BuildFactor("geom.l3i", nmb, ncells1+1, rf_of);
  geom.l3j = BuildFactor("geom.l3j", nmb, ncells2+1, sintheta_face_of);
  geom.l3k = BuildFactor("geom.l3k", nmb, ncells3,   dphi_of);
  // volumetric centroids
  geom.x1v = BuildFactor("geom.x1v", nmb, ncells1, x1v_of);
  geom.x2v = BuildFactor("geom.x2v", nmb, ncells2, x2v_of);
  geom.x3v = BuildFactor("geom.x3v", nmb, ncells3, x3v_of);
  // face positions
  geom.xf1 = BuildFactor("geom.xf1", nmb, ncells1+1, rf_of);
  geom.xf2 = BuildFactor("geom.xf2", nmb, ncells2+1, thf);
  geom.xf3 = BuildFactor("geom.xf3", nmb, ncells3+1, xf3_of);
  // geometric source-term coefficients (radial part; theta part below, Task C2)
  geom.src1 = BuildFactor("geom.src1", nmb, ncells1, src1_of);
  geom.src2 = BuildFactor("geom.src2", nmb, ncells1, src2_of);
  // theta-momentum geometric source coefficients (Task C2): src1_j = (sp-sm)/|cm-cp|
  // (old Athena++'s coord_src1_j_ == coord_src3_j_); src2_j = (sp-sm)/((sm+sp)*|cm-cp|)
  // (coord_src2_j_). For the required 1D-radial layout (nx2=1, x2min=0, x2max=pi),
  // sm=sin(0)=0 and sp=sin(pi)=0 exactly, so src1_j=0 identically -- this is what makes
  // the theta-momentum centrifugal term automatically "inert" for that layout with no
  // special-casing (see geometric_srcterms.cpp).
  auto src1_j_of = [&](int m, int j) {
    Real sm = sintheta_face_of(m, j), sp = sintheta_face_of(m, j+1);
    return (sp - sm) / dcostheta_of(m, j);
  };
  auto src2_j_of = [&](int m, int j) {
    Real sm = sintheta_face_of(m, j), sp = sintheta_face_of(m, j+1);
    return (sp - sm) / ((sm + sp) * dcostheta_of(m, j));
  };
  geom.src1_j = BuildFactor("geom.src1_j", nmb, ncells2, src1_j_of);
  geom.src2_j = BuildFactor("geom.src2_j", nmb, ncells2, src2_j_of);
  // CFL-purpose cell widths (Task B5): CenterWidth2 = r_v(i)*dtheta(j) (both theta and
  // phi directions are angular here, unlike cylindrical -- see old Athena++'s
  // spherical_polar.cpp:319-335, CenterWidth2/3, ported directly); CenterWidth3 =
  // r_v(i)*sin(theta_v(j))*dphi(k), using the CENTROID sin(theta), not the face value
  // sintheta_face_of used by Len3/Area2 above.
  geom.cw2i = BuildFactor("geom.cw2i", nmb, ncells1, x1v_of);
  geom.cw2j = BuildFactor("geom.cw2j", nmb, ncells2, dtheta_of);
  geom.cw3i = BuildFactor("geom.cw3i", nmb, ncells1, x1v_of);
  geom.cw3j = BuildFactor("geom.cw3j", nmb, ncells2,
                          [&](int m, int j) { return std::abs(std::sin(x2v_of(m, j))); });
  geom.cw3k = BuildFactor("geom.cw3k", nmb, ncells3, dphi_of);

  // PPM4/PPMX x1 (r) interpolation weights (Task B7): Mignone (2014) eq. B.14, m_coord=2
  // (spherical radial Jacobian ~ r^2). Same io=signed-local-radius-in-units-of-dr
  // rationale as geometry_cylindrical.cpp (see that file's comment for the full
  // derivation) -- theta/phi directions are explicitly NOT generalized here (see
  // mesh_geometry.hpp's GeomData doc comment on ppm_c1i..c4i for why).
  auto ppm_io_of = [&](int m, int i) { return rf(m, i) / size.h_view(m).dx1; };
  auto ppm_c1_of = [&](int m, int i) {
    Real io = ppm_io_of(m, i);
    Real io2 = io*io, io3 = io2*io, io4 = io2*io2, io5 = io4*io, io6 = io4*io2;
    Real delta = 36.0*(15.0*io4*io4 - 85.0*io6 + 150.0*io4 - 60.0*io2 + 16.0);
    return -(3.0*io2 - 9.0*io + 7.0)*(15.0*io6 + 48.0*io5 + 23.0*io4
                                       - 48.0*io3 - 30.0*io2 + 16.0*io + 12.0)/delta;
  };
  auto ppm_c2_of = [&](int m, int i) {
    Real io = ppm_io_of(m, i);
    Real io2 = io*io, io3 = io2*io, io4 = io2*io2, io5 = io4*io, io6 = io4*io2;
    Real delta = 36.0*(15.0*io4*io4 - 85.0*io6 + 150.0*io4 - 60.0*io2 + 16.0);
    return (3.0*io2 - 3.0*io + 1.0)*(105.0*io6 + 144.0*io5 - 487.0*io4
                                      - 720.0*io3 + 510.0*io2 + 1008.0*io + 372.0)/delta;
  };
  auto ppm_c3_of = [&](int m, int i) {
    Real io = ppm_io_of(m, i);
    Real io2 = io*io, io3 = io2*io, io4 = io2*io2, io5 = io4*io, io6 = io4*io2;
    Real delta = 36.0*(15.0*io4*io4 - 85.0*io6 + 150.0*io4 - 60.0*io2 + 16.0);
    return (3.0*io2 + 3.0*io + 1.0)*(105.0*io6 - 144.0*io5 - 487.0*io4
                                      + 720.0*io3 + 510.0*io2 - 1008.0*io + 372.0)/delta;
  };
  auto ppm_c4_of = [&](int m, int i) {
    Real io = ppm_io_of(m, i);
    Real io2 = io*io, io3 = io2*io, io4 = io2*io2, io5 = io4*io, io6 = io4*io2;
    Real delta = 36.0*(15.0*io4*io4 - 85.0*io6 + 150.0*io4 - 60.0*io2 + 16.0);
    return -(3.0*io2 + 9.0*io + 7.0)*(15.0*io6 - 48.0*io5 + 23.0*io4
                                       + 48.0*io3 - 30.0*io2 - 16.0*io + 12.0)/delta;
  };
  // Mignone eq. 48 overshoot ratios, spherical (uses |x1v|, same rationale as cylindrical)
  auto ppm_hp_of = [&](int m, int i) {
    Real xv = std::abs(x1v_of(m, i));
    Real dxi = dr_of(m, i);
    Real denom = 20.0*SQR(xv) + SQR(dxi);
    Real h_plus = 3.0 + (2.0*dxi*(10.0*xv + dxi))/denom;
    Real h_minus = 3.0 + (2.0*dxi*(-10.0*xv + dxi))/denom;
    return (h_plus + 1.0)/(h_minus - 1.0);
  };
  auto ppm_hm_of = [&](int m, int i) {
    Real xv = std::abs(x1v_of(m, i));
    Real dxi = dr_of(m, i);
    Real denom = 20.0*SQR(xv) + SQR(dxi);
    Real h_plus = 3.0 + (2.0*dxi*(10.0*xv + dxi))/denom;
    Real h_minus = 3.0 + (2.0*dxi*(-10.0*xv + dxi))/denom;
    return (h_minus + 1.0)/(h_plus - 1.0);
  };
  geom.ppm_c1i = BuildFactor("geom.ppm_c1i", nmb, ncells1+1, ppm_c1_of);
  geom.ppm_c2i = BuildFactor("geom.ppm_c2i", nmb, ncells1+1, ppm_c2_of);
  geom.ppm_c3i = BuildFactor("geom.ppm_c3i", nmb, ncells1+1, ppm_c3_of);
  geom.ppm_c4i = BuildFactor("geom.ppm_c4i", nmb, ncells1+1, ppm_c4_of);
  geom.ppm_hpi = BuildFactor("geom.ppm_hpi", nmb, ncells1, ppm_hp_of);
  geom.ppm_hmi = BuildFactor("geom.ppm_hmi", nmb, ncells1, ppm_hm_of);
}
