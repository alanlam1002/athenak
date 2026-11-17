//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cartgrid.cpp
//! \brief writes data on a Cartesian sub-grid in binary format

#include <sys/stat.h>  // mkdir

#include <cstdio> // snprintf
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "outputs.hpp"
#include "parameter_input.hpp"
#include "utils/cart_grid.hpp"

CartesianGridOutput::CartesianGridOutput(ParameterInput *pin, Mesh *pm,
                                         OutputParameters op)
    : BaseTypeOutput(pin, pm, op) {
  Real center[3], extent[3];
  int numpoints[3];
  bool is_cheb;

  mkdir("cart", 0755);

  center[0] = pin->GetOrAddReal(op.block_name, "center_x", 0.0);
  center[1] = pin->GetOrAddReal(op.block_name, "center_y", 0.0);
  center[2] = pin->GetOrAddReal(op.block_name, "center_z", 0.0);
  extent[0] = pin->GetOrAddReal(op.block_name, "extent_x", 2.0);
  extent[1] = pin->GetOrAddReal(op.block_name, "extent_y", 2.0);
  extent[2] = pin->GetOrAddReal(op.block_name, "extent_z", 2.0);
  numpoints[0] = pin->GetOrAddInteger(op.block_name, "numpoints_x", 32);
  numpoints[1] = pin->GetOrAddInteger(op.block_name, "numpoints_y", 32);
  numpoints[2] = pin->GetOrAddInteger(op.block_name, "numpoints_z", 32);
  is_cheb = pin->GetOrAddBoolean(op.block_name, "chebyshev", false);

  pcart = new CartesianGrid(pm->pmb_pack, center, extent, numpoints, is_cheb);

  // K-013 (wip/cartplane): an optional tilted grid (plane_e1_*, plane_e2_*: both vectors required;
  // default off = the axis-aligned grid) and nearest-cell sampling (interp = nearest; default
  // lagrange, as before). Read with DoesParameterExist so a deck without them is unchanged.
  const char *ax[3] = {"x", "y", "z"};
  bool has1 = false, has2 = false;
  for (int d = 0; d < 3; ++d) {
    has1 = has1 || pin->DoesParameterExist(op.block_name, std::string("plane_e1_") + ax[d]);
    has2 = has2 || pin->DoesParameterExist(op.block_name, std::string("plane_e2_") + ax[d]);
  }
  if (has1 != has2) {
    std::cout << "### FATAL ERROR in CartesianGridOutput: <" << op.block_name << "> needs both "
              << "plane_e1_* and plane_e2_* for a tilted grid" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (has1) {
    Real a1[3], a2[3];
    for (int d = 0; d < 3; ++d) {
      a1[d] = pin->DoesParameterExist(op.block_name, std::string("plane_e1_") + ax[d]) ?
              pin->GetReal(op.block_name, std::string("plane_e1_") + ax[d]) : 0.0;
      a2[d] = pin->DoesParameterExist(op.block_name, std::string("plane_e2_") + ax[d]) ?
              pin->GetReal(op.block_name, std::string("plane_e2_") + ax[d]) : 0.0;
    }
    pcart->SetPlane(a1, a2);
  }
  if (pin->DoesParameterExist(op.block_name, "interp")) {
    std::string im = pin->GetString(op.block_name, "interp");
    if (im == "nearest") {
      pcart->SetNearest(true);
    } else if (im != "lagrange") {
      std::cout << "### FATAL ERROR in CartesianGridOutput: interp = " << im
                << " (lagrange | nearest)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  // the grid's geometry, once, beside the data (the binary MetaData is unchanged)
  if (global_variable::my_rank == 0 && (pcart->tilted || pcart->nearest)) {
    char fn[BUFSIZ];
    std::snprintf(fn, BUFSIZ, "cart/%s.%s.plane.txt", op.file_basename.c_str(),
                  op.file_id.c_str());
    std::ofstream f(fn);
    f.precision(17);
    f << "# K-013 cart grid geometry: point(i,j,k) = center + u e1 + v e2 + w e3,\n"
      << "# u,v,w uniform (or Chebyshev) over [-extent, extent]; data written k, j, i\n";
    f << "center " << center[0] << " " << center[1] << " " << center[2] << "\n";
    f << "extent " << extent[0] << " " << extent[1] << " " << extent[2] << "\n";
    f << "numpoints " << numpoints[0] << " " << numpoints[1] << " " << numpoints[2] << "\n";
    f << "chebyshev " << (is_cheb ? 1 : 0) << "\n";
    f << "e1 " << pcart->e1[0] << " " << pcart->e1[1] << " " << pcart->e1[2] << "\n";
    f << "e2 " << pcart->e2[0] << " " << pcart->e2[1] << " " << pcart->e2[2] << "\n";
    f << "e3 " << pcart->e3[0] << " " << pcart->e3[1] << " " << pcart->e3[2] << "\n";
    f << "interp " << (pcart->nearest ? "nearest" : "lagrange") << "\n";
  }

  for (int d = 0; d < 3; ++d) {
    md.center[d] = center[d];
    md.extent[d] = extent[d];
    md.numpoints[d] = numpoints[d];
  }
  md.is_cheb = is_cheb;
}

CartesianGridOutput::~CartesianGridOutput() { delete pcart; }

void CartesianGridOutput::LoadOutputData(Mesh *pm) {
  // If AMR is enabled we need to reset the CartesianGrid
  if (pm->adaptive) {
    pcart->SetInterpolationIndices();
    pcart->SetInterpolationWeights();
  }

  int nout_vars = outvars.size();
  Kokkos::realloc(outarray, nout_vars, 1, md.numpoints[0], md.numpoints[1],
                  md.numpoints[2]);

  // Calculate derived variables, if required
  if (out_params.contains_derived) {
    ComputeDerivedVariable(out_params.variable, pm);
  }

  for (int n = 0; n < nout_vars; ++n) {
    pcart->InterpolateToGrid(outvars[n].data_index, *(outvars[n].data_ptr));
    auto v_slice =
        Kokkos::subview(outarray, n, 0, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    Kokkos::deep_copy(v_slice, pcart->interp_vals.h_view);
  }
#if MPI_PARALLEL_ENABLED
  // Note that InterpolateToGrid will set zero values for points not owned by
  // current rank
  int count = nout_vars * md.numpoints[0] * md.numpoints[1] * md.numpoints[2];
  if (0 == global_variable::my_rank) {
    MPI_Reduce(MPI_IN_PLACE, outarray.data(), count, MPI_ATHENA_REAL, MPI_SUM,
               0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(outarray.data(), outarray.data(), count, MPI_ATHENA_REAL,
               MPI_SUM, 0, MPI_COMM_WORLD);
  }
#endif
}

void CartesianGridOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
#if MPI_PARALLEL_ENABLED
  if (0 == global_variable::my_rank) {
#endif

    // Assemble filename
    char fname[BUFSIZ];
    std::snprintf(fname, BUFSIZ, "cart/%s.%s.%05d.bin",
                  out_params.file_basename.c_str(), out_params.file_id.c_str(),
                  out_params.file_number);

    // Open file
    std::ofstream ofile(fname, std::ios::binary);

    // Write metadata
    md.cycle = pm->ncycle;
    md.time = pm->time;
    md.noutvars = outvars.size();
    ofile.write(reinterpret_cast<char *>(&md), sizeof(MetaData));

    // Write list of variables
    {
      std::stringstream msg;
      for (int n = 0; n < md.noutvars - 1; ++n) {
        msg << outvars[n].label << " ";
      }
      msg << outvars[md.noutvars - 1].label;
      std::string smsg = msg.str();
      int len = smsg.size();
      ofile.write(reinterpret_cast<char *>(&len), sizeof(int));
      ofile.write(smsg.c_str(), len);
    }

    // Write actual data
    for (int n = 0; n < md.noutvars; ++n) {
      for (int k = 0; k < md.numpoints[2]; ++k) {
        for (int j = 0; j < md.numpoints[1]; ++j) {
          for (int i = 0; i < md.numpoints[0]; ++i) {
            // Note that we are accessing the array with the convention of
            // CartesianGrid which is opposite from the one used in the rest of
            // the code, but we write the output as k, j, i
            float var = outarray(n, 0, i, j, k);
            ofile.write(reinterpret_cast<char *>(&var), sizeof(float));
          }
        }
      }
    }

#if MPI_PARALLEL_ENABLED
  }
#endif

  // increment counters
  out_params.file_number++;
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetInteger(out_params.block_name, "file_number", out_params.file_number);
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);
}
