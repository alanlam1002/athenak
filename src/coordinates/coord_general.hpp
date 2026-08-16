#ifndef COORDINATES_COORD_GENERAL_HPP_
#define COORDINATES_COORD_GENERAL_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file coord_general.hpp
//! \brief Grid-geometry (curvilinear) coordinate-system selector.
//!
//! Deliberately kept OUT of mesh.hpp so that mesh.hpp's diff against upstream/main stays
//! to a single #include plus one data member -- the enum, its documentation, and the
//! input parsing all live here instead. See coord_general.cpp for the parser and for
//! Mesh::ValidateCoordGeneral().

// Forward declaration (this header must not pull in parameter_input.hpp or mesh.hpp)
class ParameterInput;

//----------------------------------------------------------------------------------------
//! \enum CoordinateGeneral
//! \brief selects the grid-geometry (curvilinear) coordinate system used by the Mesh.
//! This is independent of, and must not be confused with, the relativistic metric
//! properties held by the <coord> input block and the Coordinates class (GR/SR on a
//! Cartesian computational background) -- CoordinateGeneral instead controls the
//! face-area/cell-volume/edge-length geometry used by MeshGeometry (see
//! src/coordinates/mesh_geometry.hpp).
//!
//! Layouts (see DEVELOPMENT.md for the full rationale):
//!   cartesian           : x1,x2,x3 = x,y,z                    (1D/2D/3D)
//!   cylindrical         : x1,x2,x3 = R,phi,z (right-handed)   (2D (R,phi)/3D)
//!   cylindrical_axisym  : x1,x2 = R,z; x3 unused (nx3=1 required); phi carried
//!                         as a non-grid rotational component (left-handed R,z,phi).
//!                         This -- not "cylindrical with nx2=1" -- is how an
//!                         axisymmetric (R,z) grid is represented, because AthenaK's
//!                         one_d/two_d/three_d/multi_d flags are strictly nested
//!                         (nx3>1 forces multi_d, and nx2<4 && multi_d is fatal), so
//!                         nx2=1,nx3>1 cannot be built.
//!   spherical_polar     : x1,x2,x3 = r,theta,phi (right-handed) (1D (r)/2D/3D)

enum class CoordinateGeneral {cartesian, cylindrical, cylindrical_axisym,
                             spherical_polar};

//----------------------------------------------------------------------------------------
//! \fn ParseCoordGeneral()
//! \brief reads <mesh>/coord and returns the corresponding enumerator (defaults to
//! cartesian). Fatal error on an unrecognized string. Defined in coord_general.cpp.

CoordinateGeneral ParseCoordGeneral(ParameterInput *pin);

#endif  // COORDINATES_COORD_GENERAL_HPP_
