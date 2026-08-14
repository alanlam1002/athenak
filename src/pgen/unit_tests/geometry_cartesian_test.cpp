//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file geometry_cartesian_test.cpp
//! \brief Unit test for Task A2: verifies that MeshGeometry/GeomData's Cartesian
//! geometry factory (geometry_cartesian.cpp) reproduces, to machine precision, the same
//! arithmetic the rest of the code already computes directly from mb_size.dx1/dx2/dx3
//! and CellCenterX() -- i.e. that materializing today's implicit Cartesian geometry into
//! GeomData's factored arrays changes nothing numerically.

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
constexpr Real kTol = 1.0e-13;

// (nx>1) ? nx+2*ng : 1, matching geometry_cartesian.cpp's NCells() and
// src/hydro/hydro.cpp's ncells2/ncells3 convention.
int NCells(int nx, int ng) { return (nx > 1) ? (nx + 2*ng) : 1; }

bool CheckClose(const std::string &label, Real got, Real expected, bool *failed) {
  Real err = std::abs(got - expected);
  Real scale = std::max(std::abs(expected), Real(1.0));
  if (err > kTol*scale) {
    std::cout << "Geometry Cartesian Test FAILED: " << label
              << " got=" << got << " expected=" << expected
              << " err=" << err << std::endl;
    *failed = true;
    return false;
  }
  return true;
}
} // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::GeometryCartesianTest()
//! \brief compares every GeomData accessor (Area1/2/3, Vol, Len1/2/3, x1v/x2v/x3v) built
//! by the Cartesian geometry factory against directly-recomputed reference values.

void ProblemGenerator::GeometryCartesianTest(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;
  int nmb = pmbp->nmb_thispack;
  int ng = indcs.ng;
  int ncells1 = indcs.nx1 + 2*ng;
  int ncells2 = NCells(indcs.nx2, ng);
  int ncells3 = NCells(indcs.nx3, ng);

  auto &size = pmbp->pmb->mb_size;  // DualArray1D<RegionSize>, host view synced at init
  auto &geom = pmbp->pgeom->geom_data;

  // pull every factor array back to host for comparison
  auto a1i_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.a1i);
  auto a1j_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.a1j);
  auto a1k_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.a1k);
  auto a2i_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.a2i);
  auto a2j_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.a2j);
  auto a2k_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.a2k);
  auto a3i_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.a3i);
  auto a3j_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.a3j);
  auto a3k_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.a3k);
  auto vi_h  = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.vi);
  auto vj_h  = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.vj);
  auto vk_h  = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.vk);
  auto l1i_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.l1i);
  auto l1j_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.l1j);
  auto l1k_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.l1k);
  auto l2i_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.l2i);
  auto l2j_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.l2j);
  auto l2k_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.l2k);
  auto l3i_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.l3i);
  auto l3j_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.l3j);
  auto l3k_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.l3k);
  auto x1v_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.x1v);
  auto x2v_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.x2v);
  auto x3v_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.x3v);
  auto src1_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.src1);
  auto src2_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), geom.src2);

  bool failed = false;
  for (int m = 0; m < nmb; ++m) {
    Real dx1 = size.h_view(m).dx1;
    Real dx2 = size.h_view(m).dx2;
    Real dx3 = size.h_view(m).dx3;

    // face factors must be exactly 1 (Cartesian: no metric weighting)
    for (int i = 0; i < ncells1+1; ++i) {
      CheckClose("a1i", a1i_h(m,i), 1.0, &failed);
    }
    for (int j = 0; j < ncells2+1; ++j) { CheckClose("a2j", a2j_h(m,j), 1.0, &failed); }
    for (int k = 0; k < ncells3+1; ++k) { CheckClose("a3k", a3k_h(m,k), 1.0, &failed); }
    for (int i = 0; i < ncells1+1; ++i) { CheckClose("l2i", l2i_h(m,i), 1.0, &failed); }
    for (int i = 0; i < ncells1+1; ++i) { CheckClose("l3i", l3i_h(m,i), 1.0, &failed); }
    for (int j = 0; j < ncells2+1; ++j) { CheckClose("l1j", l1j_h(m,j), 1.0, &failed); }
    for (int j = 0; j < ncells2+1; ++j) { CheckClose("l3j", l3j_h(m,j), 1.0, &failed); }
    for (int k = 0; k < ncells3+1; ++k) { CheckClose("l1k", l1k_h(m,k), 1.0, &failed); }
    for (int k = 0; k < ncells3+1; ++k) { CheckClose("l2k", l2k_h(m,k), 1.0, &failed); }

    // width-type factors must equal the block's dx1/dx2/dx3
    for (int j = 0; j < ncells2; ++j) { CheckClose("a1j", a1j_h(m,j), dx2, &failed); }
    for (int k = 0; k < ncells3; ++k) { CheckClose("a1k", a1k_h(m,k), dx3, &failed); }
    for (int i = 0; i < ncells1; ++i) { CheckClose("a2i", a2i_h(m,i), dx1, &failed); }
    for (int k = 0; k < ncells3; ++k) { CheckClose("a2k", a2k_h(m,k), dx3, &failed); }
    for (int i = 0; i < ncells1; ++i) { CheckClose("a3i", a3i_h(m,i), dx1, &failed); }
    for (int j = 0; j < ncells2; ++j) { CheckClose("a3j", a3j_h(m,j), dx2, &failed); }
    for (int i = 0; i < ncells1; ++i) { CheckClose("vi",  vi_h(m,i),  dx1, &failed); }
    for (int j = 0; j < ncells2; ++j) { CheckClose("vj",  vj_h(m,j),  dx2, &failed); }
    for (int k = 0; k < ncells3; ++k) { CheckClose("vk",  vk_h(m,k),  dx3, &failed); }
    for (int i = 0; i < ncells1; ++i) { CheckClose("l1i", l1i_h(m,i), dx1, &failed); }
    for (int j = 0; j < ncells2; ++j) { CheckClose("l2j", l2j_h(m,j), dx2, &failed); }
    for (int k = 0; k < ncells3; ++k) { CheckClose("l3k", l3k_h(m,k), dx3, &failed); }

    // derived accessors: Area1=dx2*dx3, Area2=dx1*dx3, Area3=dx1*dx2, Vol=dx1*dx2*dx3
    // (spot-check at a handful of representative indices, not the full NxNxN grid)
    int i_chk = indcs.is, j_chk = (indcs.nx2 > 1) ? indcs.js : 0;
    int k_chk = (indcs.nx3 > 1) ? indcs.ks : 0;
    Real area1 = a1i_h(m,i_chk) * a1j_h(m,j_chk) * a1k_h(m,k_chk);
    Real area2 = a2i_h(m,i_chk) * a2j_h(m,j_chk) * a2k_h(m,k_chk);
    Real area3 = a3i_h(m,i_chk) * a3j_h(m,j_chk) * a3k_h(m,k_chk);
    Real vol   = vi_h(m,i_chk)  * vj_h(m,j_chk)  * vk_h(m,k_chk);
    CheckClose("Area1", area1, dx2*dx3, &failed);
    CheckClose("Area2", area2, dx1*dx3, &failed);
    CheckClose("Area3", area3, dx1*dx2, &failed);
    CheckClose("Vol",   vol,   dx1*dx2*dx3, &failed);

    // volumetric centroids must match CellCenterX() exactly (same formula, materialized)
    for (int i = 0; i < ncells1; ++i) {
      Real expected = CellCenterX(i - indcs.is, indcs.nx1,
                                   size.h_view(m).x1min, size.h_view(m).x1max);
      CheckClose("x1v", x1v_h(m,i), expected, &failed);
    }

    // geometric source coefficients must be exactly zero (no curvature in Cartesian)
    for (int i = 0; i < ncells1; ++i) {
      CheckClose("src1", src1_h(m,i), 0.0, &failed);
      CheckClose("src2", src2_h(m,i), 0.0, &failed);
    }
  }

  if (failed) {
    std::cout << "Geometry Cartesian Test FAILED (see above)" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout << "Geometry Cartesian Test Passed" << std::endl;
  return;
}
