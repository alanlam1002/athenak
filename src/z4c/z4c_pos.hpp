#ifndef Z4C_Z4C_POS_HPP_
#define Z4C_Z4C_POS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_pos.hpp
//! \brief The one place Z4c turns cell indices into a position (D-014 R3, D-017).
//!
//! Z4c grid points are the arithmetic cell midpoints (CellCenterX), never the geometry
//! object's x1v/x2v/x3v (hydro volume centroids, which differ in curvilinear coords).
//!
//! The position is returned in MESH component order (x1, x2, x3), the same order as the
//! tensor components in u0/u_cart, so a site can use it directly as a vector:
//!   none      (cartesian 3D)          : (x1, x2, x3), exactly as before
//!   axisym    (cylindrical_axisym 2D) : (R, z, 0), R = x1, z = x2, on the x3 = 0 plane
//!   spherical (spherical_polar 1D)    : (r, 0, 0), r = x1 (M4)
//! Frame (D-017): physical (x, y, z) = mesh (x1, -x3, x2); symmetry axis = mesh x2.
//! This is a proper rotation, so formulas written in mesh components (including ones
//! with a Levi-Civita symbol) need no sign changes.
//! Indices may lie outside the active range (ghosts, stencil offsets); CellCenterX
//! extrapolates linearly, as it always has.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "z4c/z4c.hpp"

namespace z4c {

KOKKOS_INLINE_FUNCTION
void Z4cCartesianPos(const Z4c::CartoonMode mode, const RegionSize &s,
                     const RegionIndcs &indcs, const int k, const int j, const int i,
                     Real &x1, Real &x2, Real &x3) {
  x1 = CellCenterX(i - indcs.is, indcs.nx1, s.x1min, s.x1max);
  if (mode == Z4c::CartoonMode::axisym) {
    x2 = CellCenterX(j - indcs.js, indcs.nx2, s.x2min, s.x2max);
    x3 = 0.0;
  } else if (mode == Z4c::CartoonMode::spherical) {
    x2 = 0.0;
    x3 = 0.0;
  } else {
    x2 = CellCenterX(j - indcs.js, indcs.nx2, s.x2min, s.x2max);
    x3 = CellCenterX(k - indcs.ks, indcs.nx3, s.x3min, s.x3max);
  }
}

} // namespace z4c
#endif // Z4C_Z4C_POS_HPP_
