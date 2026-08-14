//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file recon_exact_cubic_test.cpp
//! \brief Unit test for Task B7: verifies that the generalized (Mignone 2014 eq. B.4/
//! B.9/B.14) PPM4/PPMX 4-point interpolation weights in src/reconstruct/ppm.hpp exactly
//! reconstruct the TRUE face point value of a manufactured field that is an exact CUBIC
//! polynomial in x1 -- to roundoff -- given the correct metric-WEIGHTED cell average as
//! input (unweighted for Cartesian, R-weighted for cylindrical/axisym, R^2-weighted for
//! spherical, matching each system's actual finite-volume cell average). This is the
//! direct PPM analogue of Task B6's recon_exact_gradient_test (which checks PLM is exact
//! for a linear field): a 4-point Lagrange-type interpolant built from EXACT
//! weighted-moment cell averages is exact for any cubic, so this test manufactures a
//! monotonic cubic (chosen with no interior extremum, so neither PPM4's simple clamp nor
//! PPMX's extremum-preserving branch activates and both reduce to the pure unlimited
//! interpolation) and calls PPM4()/PPMX() directly with the hand-computed exact cell
//! averages and the actual GeomData coefficients, on whichever coordinate system the run
//! is configured for.

#include <cmath>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/meshblock_pack.hpp"
#include "coordinates/mesh_geometry.hpp"
#include "reconstruct/ppm.hpp"

namespace {
constexpr Real kTol = 1.0e-10;

// weighted average of R^n over [Rm,Rp] with weight R^m_coord (m_coord=0,1,2 for
// cartesian/cylindrical/spherical): ratio of the (n+m_coord)-th and m_coord-th weighted
// moments, i.e. \int_{Rm}^{Rp} R^n R^{m_coord} dR / \int_{Rm}^{Rp} R^{m_coord} dR.
Real WeightedMonomialAvg(int m_coord, int n, Real Rm, Real Rp) {
  int p = n + m_coord + 1;
  int q = m_coord + 1;
  Real num = (std::pow(Rp, p) - std::pow(Rm, p))/p;
  Real den = (std::pow(Rp, q) - std::pow(Rm, q))/q;
  return num/den;
}

// exact weighted cell average of c0 + c1*R + c2*R^2 + c3*R^3 over [Rm,Rp]
Real CubicCellAverage(int m_coord, const Real c[4], Real Rm, Real Rp) {
  Real avg = 0.0;
  for (int n = 0; n < 4; ++n) {
    avg += c[n]*WeightedMonomialAvg(m_coord, n, Rm, Rp);
  }
  return avg;
}

Real CubicEval(const Real c[4], Real R) {
  return c[0] + c[1]*R + c[2]*R*R + c[3]*R*R*R;
}
} // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::ReconExactCubicTest()

void ProblemGenerator::ReconExactCubicTest(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;
  auto &geom = pmbp->pgeom->geom_data;

  int m_coord;
  switch (pmy_mesh_->coord_general) {
    case CoordinateGeneral::cartesian:          m_coord = 0; break;
    case CoordinateGeneral::cylindrical:        m_coord = 1; break;
    case CoordinateGeneral::cylindrical_axisym: m_coord = 1; break;
    case CoordinateGeneral::spherical_polar:    m_coord = 2; break;
    default: m_coord = 0; break;
  }

  auto xf1_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.xf1);
  auto c1_h  = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.ppm_c1i);
  auto c2_h  = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.ppm_c2i);
  auto c3_h  = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.ppm_c3i);
  auto c4_h  = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.ppm_c4i);
  auto hp_h  = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.ppm_hpi);
  auto hm_h  = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.ppm_hmi);

  // manufactured cubic, chosen strictly monotonic (derivative b+2cR+3dR^2 > 0) over any
  // domain with x1min>=0.5 as used by the coordinates-suite input files for this test,
  // so neither PPM4's simple clamp nor PPMX's extremum branch can activate.
  const Real c[4] = {1.7, 2.3, 0.11, 0.04};

  bool failed = false;
  int m = 0;
  for (int i = indcs.is + 2; i <= indcs.ie - 2; ++i) {
    Real q_im2 = CubicCellAverage(m_coord, c, xf1_h(m,i-2), xf1_h(m,i-1));
    Real q_im1 = CubicCellAverage(m_coord, c, xf1_h(m,i-1), xf1_h(m,i));
    Real q_i   = CubicCellAverage(m_coord, c, xf1_h(m,i),   xf1_h(m,i+1));
    Real q_ip1 = CubicCellAverage(m_coord, c, xf1_h(m,i+1), xf1_h(m,i+2));
    Real q_ip2 = CubicCellAverage(m_coord, c, xf1_h(m,i+2), xf1_h(m,i+3));

    Real expected_left_face  = CubicEval(c, xf1_h(m,i));    // qr at face i
    Real expected_right_face = CubicEval(c, xf1_h(m,i+1));  // ql at face i+1
    Real scale = std::max(std::abs(expected_right_face), Real(1.0));

    Real ql4, qr4;
    PPM4(q_im2, q_im1, q_i, q_ip1, q_ip2,
         c1_h(m,i),   c2_h(m,i),   c3_h(m,i),   c4_h(m,i),
         c1_h(m,i+1), c2_h(m,i+1), c3_h(m,i+1), c4_h(m,i+1),
         hp_h(m,i), hm_h(m,i), ql4, qr4);

    Real qlx, qrx;
    PPMX(q_im2, q_im1, q_i, q_ip1, q_ip2,
         c1_h(m,i),   c2_h(m,i),   c3_h(m,i),   c4_h(m,i),
         c1_h(m,i+1), c2_h(m,i+1), c3_h(m,i+1), c4_h(m,i+1),
         hp_h(m,i), hm_h(m,i), qlx, qrx);

    Real err4_r = std::abs(ql4 - expected_right_face);
    Real err4_l = std::abs(qr4 - expected_left_face);
    Real errx_r = std::abs(qlx - expected_right_face);
    Real errx_l = std::abs(qrx - expected_left_face);

    if (err4_r > kTol*scale || err4_l > kTol*scale) {
      std::cout << "Recon Exact Cubic Test FAILED (PPM4) at i=" << i
                << ": ql=" << ql4 << " expected=" << expected_right_face
                << " (err=" << err4_r << "); qr=" << qr4
                << " expected=" << expected_left_face << " (err=" << err4_l << ")"
                << std::endl;
      failed = true;
    }
    if (errx_r > kTol*scale || errx_l > kTol*scale) {
      std::cout << "Recon Exact Cubic Test FAILED (PPMX) at i=" << i
                << ": ql=" << qlx << " expected=" << expected_right_face
                << " (err=" << errx_r << "); qr=" << qrx
                << " expected=" << expected_left_face << " (err=" << errx_l << ")"
                << std::endl;
      failed = true;
    }
  }

  if (failed) {
    std::cout << "Recon Exact Cubic Test FAILED (see above)" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout << "Recon Exact Cubic Test Passed" << std::endl;
  return;
}
