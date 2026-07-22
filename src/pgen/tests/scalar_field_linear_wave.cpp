//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file scalar_field_linear_wave.cpp
//! \brief Phase-1 decoupling-limit sanity test for the scalar-tensor scalar field:
//! a massless scalar plane wave propagating at the speed of light on a fixed, flat
//! (Minkowski) Z4c background. There is no back-reaction on the metric yet (Phase 1),
//! so a flat background is an exact static solution of vacuum Z4c and stays flat for
//! all time -- any error measured here is purely in the scalar sector's own evolution.
//!
//! The initial data is a monochromatic wave
//!   sphi(x,0) = amp*sin(2*pi*k.x),           Pi(x,0) = 2*pi*|k|*amp*cos(2*pi*k.x)
//! with k chosen (as in z4c_linear_wave.cpp) so that exactly one wavelength fits across
//! the periodic domain. Since the wave travels at unit speed on the flat background, it
//! returns to its initial pattern after exactly one period t = 1/|k|; the input tlim is
//! interpreted as a number of periods and rescaled accordingly, mirroring
//! z4c_linear_wave.cpp's convention exactly.

#include <cmath>     // sqrt(), sin(), cos()
#include <cstdio>    // fopen(), fprintf(), freopen()
#include <iostream>  // endl
#include <string>    // c_str()

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "scalar_field/scalar_field.hpp"
#include "driver/driver.hpp"
#include "pgen/pgen.hpp"

// function to compute errors in solution at end of run
void ScalarFieldLinearWaveErrors(ParameterInput *pin, Mesh *pm);

namespace {
// global variable to control computation of initial conditions versus errors
bool sf_set_initial_conditions = true;
} // end anonymous namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::ScalarFieldLinearWave()
//! \brief Sets initial conditions for the scalar-field linear (plane) wave test

void ProblemGenerator::ScalarFieldLinearWave(ParameterInput *pin, const bool restart) {
  pgen_final_func = ScalarFieldLinearWaveErrors;

  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "ScalarField linear wave test requires a <z4c> block in input file"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->pscalarfield == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "ScalarField linear wave test requires a <scalarfield> block in input "
              << "file" << std::endl;
    exit(EXIT_FAILURE);
  }

  // capture variables for the kernel
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is;
  int &ie = indcs.ie;
  int &js = indcs.js;
  int &je = indcs.je;
  int &ks = indcs.ks;
  int &ke = indcs.ke;
  auto &pz4c = pmbp->pz4c;
  auto &psf = pmbp->pscalarfield;

  // Code below will automatically calculate wavevector along grid diagonal, imposing the
  // conditions of periodicity and exactly one wavelength along each grid direction
  Real x1size = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
  Real x2size = pmy_mesh_->mesh_size.x2max - pmy_mesh_->mesh_size.x2min;
  Real x3size = pmy_mesh_->mesh_size.x3max - pmy_mesh_->mesh_size.x3min;

  // Wave amplitude
  Real amp = pin->GetOrAddReal("problem", "amp", 1.0e-8);

  // Initialize wavevector
  Real kx1 = pin->GetOrAddReal("problem", "kx1", 1. / x1size);
  Real kx2 = pin->GetOrAddReal("problem", "kx2", 1. / x2size);
  Real kx3 = pin->GetOrAddReal("problem", "kx3", 1. / x3size);

  // Wavevector length (= 1/wavelength for the chosen normalization of the phase below)
  Real knorm = sqrt(SQR(kx1) + SQR(kx2) + SQR(kx3));
  Real lambda = 1./knorm;

  // set new time limit in ParameterInput (to be read by Driver constructor) based on the
  // (unit) wave speed of a massless field. Input tlim is interpreted as number of wave
  // periods for evolution.
  if (sf_set_initial_conditions) {
    Real tlim = pin->GetReal("time", "tlim");
    pin->SetReal("time", "tlim", tlim*lambda);
  }

  // compute solution in the u1 register when recomputing the analytic reference; for
  // the actual initial conditions, write directly into u0.
  auto &zu = (sf_set_initial_conditions) ? pz4c->u0 : pz4c->u1;
  auto &su = (sf_set_initial_conditions) ? psf->u0 : psf->u1;

  par_for("pgen_sf_linwave", DevExeSpace(), 0, (pmbp->nmb_thispack - 1),
      ks, ke, js, je, is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;

        int nx1 = indcs.nx1;
        int nx2 = indcs.nx2;
        int nx3 = indcs.nx3;

        Real x1v = CellCenterX(i - is, nx1, x1min, x1max);
        Real x2v = CellCenterX(j - js, nx2, x2min, x2max);
        Real x3v = CellCenterX(k - ks, nx3, x3min, x3max);
        Real phase = 2. * M_PI * (kx1*x1v + kx2*x2v + kx3*x3v);

        // massless plane wave on a flat background: sphi = amp*sin(2*pi*(k.x - |k|*t)),
        // Pi := -alpha^-1 dt(sphi) = 2*pi*|k|*amp*cos(2*pi*(k.x - |k|*t)) at t=0
        su(m, psf->I_SF_SPHI, k,j,i) = amp*sin(phase);
        su(m, psf->I_SF_PI,   k,j,i) = 2.*M_PI*knorm*amp*cos(phase);

        // flat (Minkowski) Z4c background -- an exact static solution of vacuum Z4c,
        // so it provides a fixed spacetime for the Phase-1 scalar (no back-reaction).
        zu(m,pz4c->I_Z4C_GXX,k,j,i) = 1.;
        zu(m,pz4c->I_Z4C_GXY,k,j,i) = 0.;
        zu(m,pz4c->I_Z4C_GXZ,k,j,i) = 0.;
        zu(m,pz4c->I_Z4C_GYY,k,j,i) = 1.;
        zu(m,pz4c->I_Z4C_GYZ,k,j,i) = 0.;
        zu(m,pz4c->I_Z4C_GZZ,k,j,i) = 1.;

        zu(m,pz4c->I_Z4C_AXX,k,j,i) = 0.;
        zu(m,pz4c->I_Z4C_AXY,k,j,i) = 0.;
        zu(m,pz4c->I_Z4C_AXZ,k,j,i) = 0.;
        zu(m,pz4c->I_Z4C_AYY,k,j,i) = 0.;
        zu(m,pz4c->I_Z4C_AYZ,k,j,i) = 0.;
        zu(m,pz4c->I_Z4C_AZZ,k,j,i) = 0.;

        zu(m,pz4c->I_Z4C_ALPHA,k,j,i) = 1.;
        zu(m,pz4c->I_Z4C_CHI,k,j,i) = 1.;
        zu(m,pz4c->I_Z4C_KHAT,k,j,i) = 0.;
        zu(m,pz4c->I_Z4C_THETA,k,j,i) = 0.;

        zu(m,pz4c->I_Z4C_GAMX,k,j,i) = 0.;
        zu(m,pz4c->I_Z4C_GAMY,k,j,i) = 0.;
        zu(m,pz4c->I_Z4C_GAMZ,k,j,i) = 0.;

        zu(m,pz4c->I_Z4C_BETAX,k,j,i) = 0.;
        zu(m,pz4c->I_Z4C_BETAY,k,j,i) = 0.;
        zu(m,pz4c->I_Z4C_BETAZ,k,j,i) = 0.;
      });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ScalarFieldLinearWaveErrors()
//! \brief Computes L1/Linfty errors in (sphi, Pi) at the end of the run, comparing the
//! evolved solution (u0) against the analytic initial-data pattern recomputed at t=0
//! (u1) -- valid because the wave returns to its initial shape after exactly one period.

void ScalarFieldLinearWaveErrors(ParameterInput *pin, Mesh *pm) {
  // calculate reference solution by calling pgen again.
  sf_set_initial_conditions = false;
  pm->pgen->ScalarFieldLinearWave(pin, false);

  Real l1_err[2] = {0.0, 0.0};
  Real linfty_err = 0.0;
  int nvars = 0;

  // capture class variables for kernel
  auto &indcs = pm->mb_indcs;
  int &nx1 = indcs.nx1;
  int &nx2 = indcs.nx2;
  int &nx3 = indcs.nx3;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &size = pmbp->pmb->mb_size;

  // compute errors
  if (pmbp->pscalarfield != nullptr) {
    nvars = 2; // sphi, Pi
    auto &psf = pmbp->pscalarfield;
    auto &u0_ = psf->u0;
    auto &u1_ = psf->u1;

    const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
    const int nkji = nx3*nx2*nx1;
    const int nji  = nx2*nx1;
    array_sum::GlobalSum sum_this_mb;
    Kokkos::parallel_reduce("SFLW-err",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum, Real &max_err) {
      // compute m,k,j,i indices of thread
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;

      array_sum::GlobalSum evars;
      evars.the_array[0] = vol*fabs(u0_(m,psf->I_SF_SPHI,k,j,i)
                                  - u1_(m,psf->I_SF_SPHI,k,j,i));
      max_err = fmax(max_err, evars.the_array[0]);
      evars.the_array[1] = vol*fabs(u0_(m,psf->I_SF_PI,k,j,i)
                                  - u1_(m,psf->I_SF_PI,k,j,i));
      max_err = fmax(max_err, evars.the_array[1]);

      // fill rest of the_array with zeros, if narray < NREDUCTION_VARIABLES
      for (int n=nvars; n<NREDUCTION_VARIABLES; ++n) {
        evars.the_array[n] = 0.0;
      }

      // sum into parallel reduce
      mb_sum += evars;
    }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb), Kokkos::Max<Real>(linfty_err));

    // store data into l1_err array
    for (int n=0; n<nvars; ++n) {
      l1_err[n] = sum_this_mb.the_array[n];
    }
  }

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &l1_err, nvars, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &linfty_err, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif

  // normalize errors by number of cells
  Real vol=  (pmbp->pmesh->mesh_size.x1max - pmbp->pmesh->mesh_size.x1min)
            *(pmbp->pmesh->mesh_size.x2max - pmbp->pmesh->mesh_size.x2min)
            *(pmbp->pmesh->mesh_size.x3max - pmbp->pmesh->mesh_size.x3min);
  for (int i=0; i<nvars; ++i) l1_err[i] = l1_err[i]/vol;
  linfty_err /= vol;

  // compute rms error
  Real rms_err = 0.0;
  for (int i=0; i<nvars; ++i) {
    rms_err += SQR(l1_err[i]);
  }
  rms_err = std::sqrt(rms_err);

  // root process opens output file and writes out errors
  if (global_variable::my_rank == 0) {
    std::string fname;
    fname.assign(pin->GetString("job","basename"));
    fname.append("-errs.dat");
    FILE *pfile;

    // The file exists -- reopen the file in append mode
    if ((pfile = std::fopen(fname.c_str(), "r")) != nullptr) {
      if ((pfile = std::freopen(fname.c_str(), "a", pfile)) == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Error output file could not be opened" <<std::endl;
        std::exit(EXIT_FAILURE);
      }

    // The file does not exist -- open the file in write mode and add headers
    } else {
      if ((pfile = std::fopen(fname.c_str(), "w")) == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Error output file could not be opened" <<std::endl;
        std::exit(EXIT_FAILURE);
      }
      std::fprintf(pfile, "# Nx1   Nx2   Nx3   Ncycle    RMS-L1    L-infty   ");
      std::fprintf(pfile, "sphi_L1   pi_L1    \n");
    }

    // write errors
    std::fprintf(pfile, "%04d", pmbp->pmesh->mesh_indcs.nx1);
    std::fprintf(pfile, "  %04d", pmbp->pmesh->mesh_indcs.nx2);
    std::fprintf(pfile, "  %04d", pmbp->pmesh->mesh_indcs.nx3);
    std::fprintf(pfile, "  %05d  %e %e", pmbp->pmesh->ncycle, rms_err, linfty_err);
    for (int i=0; i<nvars; ++i) {
      std::fprintf(pfile, "  %e", l1_err[i]);
    }
    std::fprintf(pfile, "\n");
    std::fclose(pfile);
  }

  return;
}
