//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file coord_general.cpp
//! \brief Parsing and validation of the grid-geometry (curvilinear) coordinate system.
//!
//! Both functions here logically belong to Mesh, but are kept in this file rather than in
//! mesh.cpp so that mesh.cpp's diff against upstream/main stays to three lines (one
//! #include, one constructor initializer, one ValidateCoordGeneral() call). All of the
//! coordinate-system-specific input checking lives here instead. See
//! coordinates/coord_general.hpp for the CoordinateGeneral enum and its layout docs.

#include <cmath>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coord_general.hpp"

//----------------------------------------------------------------------------------------
//! \fn CoordinateGeneral ParseCoordGeneral()
//! \brief reads <mesh>/coord and returns the corresponding enumerator. Defaults to
//! cartesian, so every pre-existing input file keeps working unchanged.

CoordinateGeneral ParseCoordGeneral(ParameterInput *pin) {
  std::string coord_str = pin->GetOrAddString("mesh", "coord", "cartesian");
  if (coord_str == "cartesian") {
    return CoordinateGeneral::cartesian;
  } else if (coord_str == "cylindrical") {
    return CoordinateGeneral::cylindrical;
  } else if (coord_str == "cylindrical_axisym") {
    return CoordinateGeneral::cylindrical_axisym;
  } else if (coord_str == "spherical_polar") {
    return CoordinateGeneral::spherical_polar;
  }
  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
      << "<mesh>/coord = '" << coord_str << "' not recognized. Valid options are "
      << "'cartesian', 'cylindrical', 'cylindrical_axisym', 'spherical_polar'"
      << std::endl;
  std::exit(EXIT_FAILURE);
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::ValidateCoordGeneral()
//! \brief error checks mesh_size/mesh_indcs/multilevel against coord_general. Called
//! from the constructor (using values parsed from the input file) and again from
//! BuildTreeFromRestart() (using mesh_size/mesh_indcs as overwritten by the restart
//! file's binary header) -- see the declaration in mesh.hpp for why the second call
//! is needed.

void Mesh::ValidateCoordGeneral() {
  // x1 is always the radial-like direction (r or R) for all three curvilinear systems,
  // and must be non-negative since face areas/volumes are not defined (and reflect BCs
  // are required, see Task E1) for x1min < 0.
  if (coord_general == CoordinateGeneral::cylindrical ||
      coord_general == CoordinateGeneral::cylindrical_axisym ||
      coord_general == CoordinateGeneral::spherical_polar) {
    if (mesh_size.x1min < 0.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
                << __LINE__ << std::endl
          << "Input x1min must be >= 0 for coord=cylindrical, cylindrical_axisym, or "
          << "spherical_polar: x1min=" << mesh_size.x1min << std::endl;
      std::exit(EXIT_FAILURE);
    }
    // Task E1: x1min==0 is the coordinate SINGULARITY (r=0), not an ordinary domain
    // boundary. outflow/periodic there would be silently unphysical (e.g. outflow lets
    // mass/momentum "leak" through a point that isn't actually a boundary at all); the
    // existing reflect BC already flips only the normal (x1) component and leaves
    // tangential components untouched (verified in src/bvals/physics/hydro_bcs.cpp --
    // no new BC code needed), which is exactly the correct physical statement at r=0
    // for any of the three curvilinear systems, so require it explicitly.
    if (mesh_size.x1min == 0.0 &&
        mesh_bcs[BoundaryFace::inner_x1] != BoundaryFlag::reflect) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
                << __LINE__ << std::endl
          << "Input x1min=0 is the coordinate origin/singularity for coord=cylindrical, "
          << "cylindrical_axisym, or spherical_polar, and requires ix1_bc=reflect (not "
          << "outflow/periodic/inflow, which are unphysical there): ix1_bc="
          << GetBoundaryString(mesh_bcs[BoundaryFace::inner_x1]) << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  // for spherical_polar, x2 is the polar angle theta and must lie within [0,pi]
  if (coord_general == CoordinateGeneral::spherical_polar && multi_d) {
    if (mesh_size.x2min < 0.0 || mesh_size.x2max > M_PI) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
                << __LINE__ << std::endl
          << "Input x2min/x2max must satisfy 0 <= x2min < x2max <= pi for "
          << "coord=spherical_polar: x2min=" << mesh_size.x2min
          << " x2max=" << mesh_size.x2max << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  // cylindrical_axisym represents (R,z) as x1,x2 with x3 unused (phi is a non-grid
  // rotational component); a third grid dimension is meaningless for it.
  if (coord_general == CoordinateGeneral::cylindrical_axisym && mesh_indcs.nx3 != 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "coord=cylindrical_axisym requires nx3=1 (x1,x2 represent R,z; x3 is unused "
        << "since phi is carried as a non-grid rotational component): nx3="
        << mesh_indcs.nx3 << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // SMR/AMR restriction/prolongation (mesh_refinement.cpp) currently does unweighted
  // 1/4, 1/8 averaging, which silently breaks conservation at level boundaries for
  // curvilinear volumes/areas. Curvilinear + multilevel is out of scope until that is
  // fixed (see DEVELOPMENT.md "Deferred" section), so guard against it explicitly
  // rather than allowing a silent conservation violation.
  if (multilevel && coord_general != CoordinateGeneral::cartesian) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "SMR/AMR (mesh_refinement/refinement != none) is not yet supported for "
        << "coord != cartesian: curvilinear restriction/prolongation is not "
        << "implemented and would silently break conservation at level boundaries."
        << std::endl;
    std::exit(EXIT_FAILURE);
  }
}
