#ifndef Z4C_Z4C_POS_HPP_
#define Z4C_Z4C_POS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_pos.hpp
//! \brief The one place Z4c turns cell indices into a Cartesian position (D-014, R3).
//!
//! Z4c grid points are the arithmetic cell midpoints (CellCenterX), never the geometry
//! object's x1v/x2v/x3v (hydro volume centroids, which differ in curvilinear coords).
//!   none      (cartesian 3D)          : (x1, x2, x3), exactly as before
//!   axisym    (cylindrical_axisym 2D) : (R, 0, z) with R = x1, z = x2
//!   spherical (spherical_polar 1D)    : (r, 0, 0) with r = x1   (M4)
//! Indices may lie outside the active range (ghosts, stencil offsets); CellCenterX
//! extrapolates linearly, as it always has.
//!
//! NOTE: this returns the Cartesian POSITION. Tensor COMPONENTS in u0/u_cart follow
//! the mesh axes, so on the Cartoon mesh their order is (x, z, y): index 1 is z, index
//! 2 is y.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "z4c/z4c.hpp"

namespace z4c {

KOKKOS_INLINE_FUNCTION
void Z4cCartesianPos(const Z4c::CartoonMode mode, const RegionSize &s,
                     const RegionIndcs &indcs, const int k, const int j, const int i,
                     Real &x, Real &y, Real &z) {
  if (mode == Z4c::CartoonMode::axisym) {
    x = CellCenterX(i - indcs.is, indcs.nx1, s.x1min, s.x1max);
    y = 0.0;
    z = CellCenterX(j - indcs.js, indcs.nx2, s.x2min, s.x2max);
  } else if (mode == Z4c::CartoonMode::spherical) {
    x = CellCenterX(i - indcs.is, indcs.nx1, s.x1min, s.x1max);
    y = 0.0;
    z = 0.0;
  } else {
    x = CellCenterX(i - indcs.is, indcs.nx1, s.x1min, s.x1max);
    y = CellCenterX(j - indcs.js, indcs.nx2, s.x2min, s.x2max);
    z = CellCenterX(k - indcs.ks, indcs.nx3, s.x3min, s.x3max);
  }
}

} // namespace z4c
#endif // Z4C_Z4C_POS_HPP_
