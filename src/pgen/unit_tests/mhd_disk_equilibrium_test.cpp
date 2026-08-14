//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_disk_equilibrium_test.cpp
//! \brief Unit test for Task F2: axisymmetric (R,z) magnetized rotating-disk
//! equilibrium -- the integration-level check on Task C1's phi-component sign
//! conventions AND the MHD extension of the centrifugal/pressure force balance.
//!
//! Extends geom_equilibrium_test.cpp's pure-hydro rotating equilibrium (uniform
//! density, constant rotation speed v0, log-pressure profile) by adding a spatially
//! UNIFORM toroidal field B_phi=B0 (carried in the IM3/IB3 slot, per axisym's
//! handedness convention -- same slot as v_phi). For B_R=B_z=0, the geometric
//! source term's m_pp (Task C1: `rho*vphi^2 + P + 0.5*(BR^2-Bphi^2+Bz^2)`) and the
//! radial momentum flux (`rho*vR^2 + P + 0.5*(Bphi^2+Bz^2-BR^2)`, standard MHD stress)
//! combine (worked out by hand, see DEVELOPMENT.md Task F2 log) to give the continuum
//! equilibrium condition:
//!   d(P)/dR = (rho*v0^2 - B0^2)/R
//! i.e. the SAME log profile as the pure-hydro case with rho0*v0^2 replaced by
//! (rho0*v0^2 - B0^2) -- the magnetic tension (hoop stress) of a uniform-strength
//! toroidal field acts like a NEGATIVE centrifugal term, exactly as expected physically
//! (a purely toroidal field of uniform strength pulls inward, requiring an
//! outward-increasing pressure to balance, opposite to rotation's outward centrifugal
//! push). Uses the same reused-pgen_final_func-plus-OutputErrors() mechanism as
//! geom_equilibrium_test.cpp (and, for MHD, ALSO checks div(B) stays at roundoff
//! throughout via the same check as ct_divb_test.cpp, since a uniform Bphi with
//! BR=Bz=0 is trivially divergence-free and must stay exactly so).

#include <cmath>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/meshblock_pack.hpp"
#include "coordinates/mesh_geometry.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"

namespace {
constexpr Real kDivBRelTol = 1.0e-10;

//----------------------------------------------------------------------------------------
//! \fn FillDiskEquilibrium()
//! \brief writes the analytic magnetized rotating-equilibrium state into whichever
//! (u,b) register pair is passed.
void FillDiskEquilibrium(MeshBlockPack *pmbp, ParameterInput *pin,
                         DvceArray5D<Real> &u_target, DvceFaceFld4D<Real> &b_target) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie, &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int nmb1 = pmbp->nmb_thispack - 1;
  auto &geom = pmbp->pgeom->geom_data;

  Real rho0 = pin->GetReal("problem", "dens");
  Real p0   = pin->GetReal("problem", "pgas");
  Real x1_0 = pin->GetReal("problem", "r0");
  Real v0   = pin->GetReal("problem", "vphi0");
  Real b0   = pin->GetReal("problem", "bphi0");
  Real gm1  = pmbp->pmhd->peos->eos_data.gamma - 1.0;

  auto bx1f = b_target.x1f;
  auto bx2f = b_target.x2f;
  auto bx3f = b_target.x3f;

  par_for("mhd_disk_equil_ic", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x1 = geom.x1v(m,i);
    Real pgas = p0 + (rho0*v0*v0 - b0*b0)*std::log(x1/x1_0);
    u_target(m,IDN,k,j,i) = rho0;
    u_target(m,IM1,k,j,i) = 0.0;
    u_target(m,IM2,k,j,i) = 0.0;
    u_target(m,IM3,k,j,i) = rho0*v0;
    u_target(m,IEN,k,j,i) = pgas/gm1 + 0.5*rho0*v0*v0 + 0.5*b0*b0;

    bx1f(m,k,j,i) = 0.0;
    bx2f(m,k,j,i) = 0.0;
    bx3f(m,k,j,i) = b0;
    if (i==ie) { bx1f(m,k,j,i+1) = 0.0; }
    if (j==je) { bx2f(m,k,j+1,i) = 0.0; }
    if (k==ke) { bx3f(m,k+1,j,i) = b0; }
  });
}

//----------------------------------------------------------------------------------------
//! \fn DiskEquilibriumErrors()
//! \brief pgen_final_func: fills u1/b1 with the (time-invariant) analytic solution,
//! diffs via OutputErrors(), and separately checks div(B) is still at roundoff.
void DiskEquilibriumErrors(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  FillDiskEquilibrium(pmbp, pin, pmbp->pmhd->u1, pmbp->pmhd->b1);
  pm->pgen->OutputErrors(pin, pm);

  auto &indcs = pm->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie, &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  auto &geom = pmbp->pgeom->geom_data;
  auto &b0 = pmbp->pmhd->b0;
  int nmb = pmbp->nmb_thispack;
  int nx1 = ie-is+1, nx2 = je-js+1, nx3 = ke-ks+1;
  int nkji = nx3*nx2*nx1, nji = nx2*nx1, nmkji = nmb*nkji;

  Real max_divb = 0.0, max_scale = 0.0;
  Kokkos::parallel_reduce("disk_divb_check", Kokkos::RangePolicy<>(DevExeSpace(), 0,
                           nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &mx, Real &mscale) {
    int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    Real divb = (geom.Area1(m,k,j,i+1)*b0.x1f(m,k,j,i+1)
                 - geom.Area1(m,k,j,i)*b0.x1f(m,k,j,i))
              + (geom.Area2(m,k,j+1,i)*b0.x2f(m,k,j+1,i)
                 - geom.Area2(m,k,j,i)*b0.x2f(m,k,j,i))
              + (geom.Area3(m,k+1,j,i)*b0.x3f(m,k+1,j,i)
                 - geom.Area3(m,k,j,i)*b0.x3f(m,k,j,i));
    mx = fmax(mx, fabs(divb));
    Real scale = geom.Area3(m,k,j,i)*fabs(b0.x3f(m,k,j,i));
    mscale = fmax(mscale, scale);
  }, Kokkos::Max<Real>(max_divb), Kokkos::Max<Real>(max_scale));

  Real rel = max_divb/std::fmax(max_scale, 1.0e-300);
  if (rel > kDivBRelTol) {
    std::cout << "MHD Disk Equilibrium Test FAILED: div(B) grew to relative " << rel
              << " (tolerance " << kDivBRelTol << ")" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout << "MHD Disk Equilibrium div(B) check passed (rel=" << rel << ")"
            << std::endl;
}
} // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::MHDDiskEquilibriumTest()

void ProblemGenerator::MHDDiskEquilibriumTest(ParameterInput *pin, const bool restart) {
  if (restart) return;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MHDDiskEquilibriumTest requires a <mhd> block"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  FillDiskEquilibrium(pmbp, pin, pmbp->pmhd->u0, pmbp->pmhd->b0);
  pgen_final_func = DiskEquilibriumErrors;
  return;
}
