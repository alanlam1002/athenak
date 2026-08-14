//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file geometry_curvilinear_test.cpp
//! \brief Unit test for Task B4: verifies MeshGeometry/GeomData's cylindrical,
//! cylindrical_axisym, and spherical geometry factories against INDEPENDENTLY
//! re-derived analytic formulas (written fresh here, not copied from
//! geometry_{cylindrical,cylindrical_axisym,spherical}.cpp), on a small
//! hand-computable grid. Dispatches on Mesh::coord_general so the same pgen name
//! drives all three systems' tests via three separate input files.

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

bool CheckClose(const std::string &label, Real got, Real expected, bool *failed) {
  Real err = std::abs(got - expected);
  Real scale = std::max(std::abs(expected), Real(1.0));
  if (err > kTol*scale) {
    std::cout << "Geometry Curvilinear Test FAILED: " << label
              << " got=" << got << " expected=" << expected
              << " err=" << err << std::endl;
    *failed = true;
    return false;
  }
  return true;
}

int NCells(int nx, int ng) { return (nx > 1) ? (nx + 2*ng) : 1; }

//----------------------------------------------------------------------------------------
// Cylindrical (R,phi,z): independently re-derived reference formulas.
void CheckCylindrical(MeshBlockPack *pmbp, RegionIndcs &indcs, bool *failed) {
  int nmb = pmbp->nmb_thispack;
  int ng = indcs.ng;
  int ncells1 = indcs.nx1 + 2*ng;
  int ncells2 = NCells(indcs.nx2, ng);
  int ncells3 = NCells(indcs.nx3, ng);
  int is = indcs.is;
  auto &size = pmbp->pmb->mb_size;
  auto &geom = pmbp->pgeom->geom_data;

  auto a1i_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.a1i);
  auto a2i_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.a2i);
  auto a3i_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.a3i);
  auto vi_h  = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.vi);
  auto l1i_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.l1i);
  auto l2i_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.l2i);
  auto x1v_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.x1v);
  auto src1_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.src1);
  auto src2_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.src2);

  for (int m = 0; m < nmb; ++m) {
    Real x1min = size.h_view(m).x1min, x1max = size.h_view(m).x1max;
    Real dx2 = size.h_view(m).dx2, dx3 = size.h_view(m).dx3;
    for (int i = 0; i < ncells1; ++i) {
      // fresh re-derivation: R_f at faces i and i+1 (active cells only: i>=is)
      Real Rm = LeftEdgeX(i - is, indcs.nx1, x1min, x1max);
      Real Rp = LeftEdgeX(i + 1 - is, indcs.nx1, x1min, x1max);
      Real dR = Rp - Rm;
      Real Rmom = 0.5*(Rp*Rp - Rm*Rm);
      CheckClose("cyl a2i", a2i_h(m,i), dR, failed);
      CheckClose("cyl a3i", a3i_h(m,i), Rmom, failed);
      CheckClose("cyl vi",  vi_h(m,i),  Rmom, failed);
      CheckClose("cyl l1i", l1i_h(m,i), dR, failed);
      if (i >= is && i <= indcs.ie) {
        Real x1v_expected = (2.0/3.0)*(Rp*Rp*Rp - Rm*Rm*Rm)/(Rp*Rp - Rm*Rm);
        CheckClose("cyl x1v", x1v_h(m,i), x1v_expected, failed);
        Real src1_expected = dR/Rmom;
        Real src2_expected = dR/((Rm+Rp)*Rmom);
        CheckClose("cyl src1", src1_h(m,i), src1_expected, failed);
        CheckClose("cyl src2", src2_h(m,i), src2_expected, failed);
      }
    }
    for (int i = 0; i < ncells1+1; ++i) {
      Real Rf = LeftEdgeX(i - is, indcs.nx1, x1min, x1max);
      CheckClose("cyl a1i", a1i_h(m,i), Rf, failed);
      CheckClose("cyl l2i", l2i_h(m,i), Rf, failed);
    }
    // spot-check a full Area/Vol at one active cell
    int i_chk = indcs.is + 1, j_chk = indcs.js, k_chk = indcs.ks;
    Real Rf1 = LeftEdgeX(i_chk - is, indcs.nx1, x1min, x1max);
    Real area1_expected = Rf1 * dx2 * dx3;
    Real area1_got = geom.Area1(m, k_chk, j_chk, i_chk);
    CheckClose("cyl Area1 full", area1_got, area1_expected, failed);

    // Task B5: CenterWidth2 = R_v(i)*dphi (angular, R-weighted at the CENTROID, not the
    // face); CenterWidth3 = dz (flat, no R dependence)
    Real Rm1 = LeftEdgeX(i_chk - is, indcs.nx1, x1min, x1max);
    Real Rp1 = LeftEdgeX(i_chk + 1 - is, indcs.nx1, x1min, x1max);
    Real Rv1 = (2.0/3.0)*(Rp1*Rp1*Rp1 - Rm1*Rm1*Rm1)/(Rp1*Rp1 - Rm1*Rm1);
    CheckClose("cyl CenterWidth2", geom.CenterWidth2(m, k_chk, j_chk, i_chk),
               Rv1*dx2, failed);
    CheckClose("cyl CenterWidth3", geom.CenterWidth3(m, k_chk, j_chk, i_chk),
               dx3, failed);
  }
}

//----------------------------------------------------------------------------------------
// Cylindrical axisymmetric (R,z): independently re-derived reference formulas.
void CheckCylindricalAxisym(MeshBlockPack *pmbp, RegionIndcs &indcs, bool *failed) {
  int nmb = pmbp->nmb_thispack;
  int ng = indcs.ng;
  int ncells1 = indcs.nx1 + 2*ng;
  int is = indcs.is;
  auto &size = pmbp->pmb->mb_size;
  auto &geom = pmbp->pgeom->geom_data;

  auto a1i_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.a1i);
  auto a2i_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.a2i);
  auto a3i_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.a3i);
  auto vi_h  = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.vi);
  auto l3i_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.l3i);
  auto x1v_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.x1v);

  for (int m = 0; m < nmb; ++m) {
    Real x1min = size.h_view(m).x1min, x1max = size.h_view(m).x1max;
    Real dx2 = size.h_view(m).dx2;  // x2 = z here
    for (int i = 0; i < ncells1; ++i) {
      Real Rm = LeftEdgeX(i - is, indcs.nx1, x1min, x1max);
      Real Rp = LeftEdgeX(i + 1 - is, indcs.nx1, x1min, x1max);
      Real dR = Rp - Rm;
      Real Rmom = 0.5*(Rp*Rp - Rm*Rm);
      CheckClose("axisym a2i", a2i_h(m,i), Rmom, failed);  // Area2's own i-factor
      CheckClose("axisym a3i", a3i_h(m,i), dR, failed);    // Area3's i-factor = dR
      CheckClose("axisym vi",  vi_h(m,i),  Rmom, failed);
      if (i >= is && i <= indcs.ie) {
        Real x1v_expected = (2.0/3.0)*(Rp*Rp*Rp - Rm*Rm*Rm)/(Rp*Rp - Rm*Rm);
        CheckClose("axisym x1v", x1v_h(m,i), x1v_expected, failed);
      }
    }
    for (int i = 0; i < ncells1+1; ++i) {
      Real Rf = LeftEdgeX(i - is, indcs.nx1, x1min, x1max);
      CheckClose("axisym a1i", a1i_h(m,i), Rf, failed);
      CheckClose("axisym l3i", l3i_h(m,i), Rf, failed);
    }
    // spot-check full Area1 (R-face) and Area3 (virtual phi-face, flat R-z rectangle)
    int i_chk = indcs.is + 1, j_chk = indcs.js, k_chk = indcs.ks;
    Real Rf1 = LeftEdgeX(i_chk - is, indcs.nx1, x1min, x1max);
    CheckClose("axisym Area1 full", geom.Area1(m, k_chk, j_chk, i_chk), Rf1*dx2, failed);
    Real Rm_c = LeftEdgeX(i_chk - is, indcs.nx1, x1min, x1max);
    Real Rp_c = LeftEdgeX(i_chk + 1 - is, indcs.nx1, x1min, x1max);
    CheckClose("axisym Area3 full", geom.Area3(m, k_chk, j_chk, i_chk),
               (Rp_c-Rm_c)*dx2, failed);
  }
}

//----------------------------------------------------------------------------------------
// Spherical polar (r,theta,phi): independently re-derived reference formulas.
void CheckSpherical(MeshBlockPack *pmbp, RegionIndcs &indcs, bool *failed) {
  int nmb = pmbp->nmb_thispack;
  int ng = indcs.ng;
  int ncells1 = indcs.nx1 + 2*ng;
  int is = indcs.is;
  auto &size = pmbp->pmb->mb_size;
  auto &geom = pmbp->pgeom->geom_data;

  auto a1i_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.a1i);
  auto a2i_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.a2i);
  auto vi_h  = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.vi);
  auto x1v_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.x1v);
  auto src1_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.src1);
  auto src2_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.src2);

  for (int m = 0; m < nmb; ++m) {
    Real x1min = size.h_view(m).x1min, x1max = size.h_view(m).x1max;
    Real dx3 = size.h_view(m).dx3;
    for (int i = 0; i < ncells1; ++i) {
      Real rm = LeftEdgeX(i - is, indcs.nx1, x1min, x1max);
      Real rp = LeftEdgeX(i + 1 - is, indcs.nx1, x1min, x1max);
      Real r2mom = 0.5*(rp*rp - rm*rm);
      Real r3mom = (1.0/3.0)*(rp*rp*rp - rm*rm*rm);
      CheckClose("sph a2i", a2i_h(m,i), r2mom, failed);
      CheckClose("sph vi",  vi_h(m,i),  r3mom, failed);
      if (i >= is && i <= indcs.ie) {
        Real rm2 = rm*rm, rp2 = rp*rp;
        Real x1v_expected = 0.75*(rp2*rp2 - rm2*rm2)/(rp2*rp - rm2*rm);
        CheckClose("sph x1v", x1v_h(m,i), x1v_expected, failed);
        Real src1_expected = r2mom/r3mom;
        Real src2_expected = (rp-rm)/((rm+rp)*r3mom);
        CheckClose("sph src1", src1_h(m,i), src1_expected, failed);
        CheckClose("sph src2", src2_h(m,i), src2_expected, failed);
      }
    }
    for (int i = 0; i < ncells1+1; ++i) {
      Real rf = LeftEdgeX(i - is, indcs.nx1, x1min, x1max);
      CheckClose("sph a1i", a1i_h(m,i), rf*rf, failed);
    }
    // spot-check full Area1/Vol for the required 1D (nx2=nx3=1, full solid angle) case
    if (indcs.nx2 == 1 && indcs.nx3 == 1) {
      int i_chk = indcs.is + 1, j_chk = indcs.js, k_chk = indcs.ks;
      Real rf1 = LeftEdgeX(i_chk - is, indcs.nx1, x1min, x1max);
      Real th_m = size.h_view(m).x2min, th_p = size.h_view(m).x2max;
      Real dcos = std::abs(std::cos(th_m) - std::cos(th_p));
      Real area1_expected = rf1*rf1 * dcos * dx3;
      CheckClose("sph Area1 full", geom.Area1(m, k_chk, j_chk, i_chk),
                 area1_expected, failed);
      // if x2 spans [0,pi] and x3 spans [0,2pi], this must equal a full sphere shell:
      // Area1 = 4*pi*r_f^2 (the standard result, a strong end-to-end sanity check)
      if (std::abs(th_m - 0.0) < 1e-10 && std::abs(th_p - M_PI) < 1e-10 &&
          std::abs(dx3 - 2.0*M_PI) < 1e-8) {
        CheckClose("sph Area1 4pi*r^2", geom.Area1(m, k_chk, j_chk, i_chk),
                   4.0*M_PI*rf1*rf1, failed);
      }
    }
    // Task B5: CenterWidth2 = r_v(i)*dtheta (angular), CenterWidth3 =
    // r_v(i)*sin(theta_v(j))*dphi (angular, theta-dependent too) -- both directions are
    // angular in spherical, unlike cylindrical where only x2=phi is. Uses indcs.js/ks
    // directly so this holds for both the required 1D (nx2=nx3=1) and general 2D/3D
    // cases without needing the nx2==1 gate above.
    {
      int i_chk = indcs.is + 1, j_chk = indcs.js, k_chk = indcs.ks;
      Real rm2 = LeftEdgeX(i_chk - is, indcs.nx1, x1min, x1max);
      Real rp2 = LeftEdgeX(i_chk + 1 - is, indcs.nx1, x1min, x1max);
      Real rm2sq = rm2*rm2, rp2sq = rp2*rp2;
      Real rv2 = 0.75*(rp2sq*rp2sq - rm2sq*rm2sq)/(rp2sq*rp2 - rm2sq*rm2);
      Real th_m2 = size.h_view(m).x2min, th_p2 = size.h_view(m).x2max;
      int n2 = (indcs.nx2 > 1) ? indcs.nx2 : 1;
      Real thf_m = LeftEdgeX(j_chk - indcs.js, n2, th_m2, th_p2);
      Real thf_p = LeftEdgeX(j_chk + 1 - indcs.js, n2, th_m2, th_p2);
      Real dtheta = thf_p - thf_m;
      Real thetav;
      if (indcs.nx2 == 1) {
        thetav = 0.5*(thf_m + thf_p);
      } else {
        Real num = (std::sin(thf_p) - thf_p*std::cos(thf_p)) -
                   (std::sin(thf_m) - thf_m*std::cos(thf_m));
        Real den = std::cos(thf_m) - std::cos(thf_p);
        thetav = num/den;
      }
      Real dphi = size.h_view(m).dx3;
      CheckClose("sph CenterWidth2", geom.CenterWidth2(m, k_chk, j_chk, i_chk),
                 rv2*dtheta, failed);
      CheckClose("sph CenterWidth3", geom.CenterWidth3(m, k_chk, j_chk, i_chk),
                 rv2*std::abs(std::sin(thetav))*dphi, failed);
    }
  }
}
} // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::GeometryCurvilinearTest()
//! \brief dispatches to the appropriate independently-derived-formula check based on
//! Mesh::coord_general.

void ProblemGenerator::GeometryCurvilinearTest(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;
  bool failed = false;

  switch (pmy_mesh_->coord_general) {
    case CoordinateGeneral::cylindrical:
      CheckCylindrical(pmbp, indcs, &failed);
      break;
    case CoordinateGeneral::cylindrical_axisym:
      CheckCylindricalAxisym(pmbp, indcs, &failed);
      break;
    case CoordinateGeneral::spherical_polar:
      CheckSpherical(pmbp, indcs, &failed);
      break;
    case CoordinateGeneral::cartesian:
      std::cout << "Geometry Curvilinear Test: <mesh>/coord=cartesian is not a "
                << "curvilinear system; use geometry_cartesian_test instead."
                << std::endl;
      exit(EXIT_FAILURE);
      break;
  }

  if (failed) {
    std::cout << "Geometry Curvilinear Test FAILED (see above)" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout << "Geometry Curvilinear Test Passed" << std::endl;
  return;
}
