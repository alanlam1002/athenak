//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file recon_exact_gradient_test.cpp
//! \brief Unit test for Task B6: verifies that the generalized (Mignone 2014) PLM
//! limiter in src/reconstruct/plm.hpp exactly reconstructs a field that is an exact
//! linear function of the stored volumetric centroid, evaluated at the TRUE face
//! position -- to roundoff -- regardless of grid non-uniformity/curvilinearity. Calls
//! PLM() directly with hand-computed positions from the actual GeomData arrays (not a
//! synthetic grid), on whichever coordinate system the run is configured for, so this
//! single pgen covers Cartesian (regression: must stay exact, as it always was) and any
//! curvilinear system by simply changing the input file's <mesh>/coord.

#include <cmath>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/meshblock_pack.hpp"
#include "coordinates/mesh_geometry.hpp"
#include "reconstruct/recon_geom.hpp"

namespace {
constexpr Real kTol = 1.0e-12;
}

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::ReconExactGradientTest()
//! \brief for a manufactured field q(i) = a + b*x1v(i) (exact linear function of the
//! STORED centroid), checks that PLM's reconstructed ql/qr at the faces of several
//! interior cells equal a + b*xf1(face) to roundoff.

void ProblemGenerator::ReconExactGradientTest(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;
  auto &geom = pmbp->pgeom->geom_data;

  auto x1v_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.x1v);
  auto xf1_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.xf1);

  const Real a = 3.7, b = -1.3;  // arbitrary manufactured linear coefficients

  bool failed = false;
  int m = 0;  // single-MeshBlock test input is sufficient to exercise the formula
  for (int i = indcs.is + 1; i <= indcs.ie - 1; ++i) {
    Real q_im1 = a + b*x1v_h(m, i-1);
    Real q_i   = a + b*x1v_h(m, i);
    Real q_ip1 = a + b*x1v_h(m, i+1);
    Real x_im1 = x1v_h(m, i-1), x_i = x1v_h(m, i), x_ip1 = x1v_h(m, i+1);
    Real xf_i = xf1_h(m, i), xf_ip1 = xf1_h(m, i+1);

    Real ql_ip1, qr_i;
    // Drive the production limiter through the same factors the geometry builder
    // precomputes (MakePlmCoeff is the single shared source of that formula), so this
    // test cannot silently diverge from what the reconstruction kernel actually runs.
    PlmCoeff pc = MakePlmCoeff(x_im1, x_i, x_ip1, xf_i, xf_ip1);
    PLMGeom(q_im1, q_i, q_ip1, pc, ql_ip1, qr_i);

    Real expected_right_face = a + b*xf_ip1;  // ql at face i+1
    Real expected_left_face  = a + b*xf_i;    // qr at face i

    Real err_r = std::abs(ql_ip1 - expected_right_face);
    Real err_l = std::abs(qr_i - expected_left_face);
    Real scale = std::max(std::abs(expected_right_face), Real(1.0));
    if (err_r > kTol*scale || err_l > kTol*scale) {
      std::cout << "Recon Exact Gradient Test FAILED at i=" << i
                << ": ql_ip1=" << ql_ip1 << " expected=" << expected_right_face
                << " (err=" << err_r << "); qr_i=" << qr_i
                << " expected=" << expected_left_face << " (err=" << err_l << ")"
                << std::endl;
      failed = true;
    }
  }

  if (failed) {
    std::cout << "Recon Exact Gradient Test FAILED (see above)" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout << "Recon Exact Gradient Test Passed" << std::endl;
  return;
}
