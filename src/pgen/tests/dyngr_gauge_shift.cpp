//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyngr_gauge_shift.cpp
//! \brief Unit tests for the comoving (uniform-shift) gauge in dyn_grmhd: flat space,
//! alpha = 1, gamma_ij = delta_ij, K_ij = 0, and a spatially uniform shift
//! beta^i = <problem> gauge_xdot1/2/3. Under x' = x - X(t) only the shift changes,
//! beta -> beta + Xdot, so every run here is the same physics as gauge_xdot = 0 seen
//! from a translating grid: the coordinate velocity is V = v - Xdot
//! (src/cfc/DEVELOPMENT.md item 68). Periodic boxes. <problem> type:
//!   uniform      (T0a) uniform rho0, p0, Eulerian v: must stay uniform to round-off;
//!   entropy_wave (T0b) rho = rho0 (1 + amp sin(2 pi x1/L1)), uniform p0 and v along x1:
//!                an exact solution advected at V1 = v1 - Xdot1;
//!   field_loop   (T0c) 2D, A_z = b0 (R - r) for r < R (Gardiner & Stone 2005), uniform
//!                rho0, p0, v: B advected at V; div B must stay at round-off (CT).
//! The pgen writes no error file: compare outputs offline (scripts/gauge_shift_tests.py).

#include <cmath>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/eos.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "pgen/pgen.hpp"

namespace {
Real gshift[3];
void SetADMVariablesFlatShift(MeshBlockPack *pmbp);
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::DynGRGaugeShift()

void ProblemGenerator::DynGRGaugeShift(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  for (int a = 0; a < 3; ++a) {
    gshift[a] = pin->GetOrAddReal("problem", "gauge_xdot" + std::to_string(a + 1), 0.0);
  }
  if (pmbp->padm == nullptr || pmbp->pmhd == nullptr ||
      !pmbp->pcoord->is_dynamical_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "dyngr_gauge_shift needs <mhd> with dynamical GR (<adm> block)"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  pmbp->padm->SetADMVariables = &SetADMVariablesFlatShift;
  for (int a = 0; a < 3; ++a) { pmbp->padm->gauge_xdot[a] = gshift[a]; }
  if (restart) return;

  std::string type = pin->GetString("problem", "type");
  int itype;
  if (type == "uniform") {
    itype = 0;
  } else if (type == "entropy_wave") {
    itype = 1;
  } else if (type == "field_loop") {
    itype = 2;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "unknown <problem> type = " << type << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const Real rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  const Real p0 = pin->GetOrAddReal("problem", "p0", 1.0);
  const Real v1 = pin->GetOrAddReal("problem", "v1", 0.0);
  const Real v2 = pin->GetOrAddReal("problem", "v2", 0.0);
  const Real v3 = pin->GetOrAddReal("problem", "v3", 0.0);
  const Real amp = pin->GetOrAddReal("problem", "amp", 0.1);
  const Real b0 = pin->GetOrAddReal("problem", "b0", 1.0e-3);
  const Real rloop = pin->GetOrAddReal("problem", "rloop", 0.3);
  const Real x1m = pmy_mesh_->mesh_size.x1min;
  const Real lx1 = pmy_mesh_->mesh_size.x1max - x1m;
  const Real x1c = 0.5*(pmy_mesh_->mesh_size.x1max + pmy_mesh_->mesh_size.x1min);
  const Real x2c = 0.5*(pmy_mesh_->mesh_size.x2max + pmy_mesh_->mesh_size.x2min);
  const Real lorentz = 1.0/std::sqrt(1.0 - (v1*v1 + v2*v2 + v3*v3));

  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  auto &w0 = pmbp->pmhd->w0;
  auto &bf = pmbp->pmhd->b0;
  auto &bcc0 = pmbp->pmhd->bcc0;
  int nmb = pmbp->nmb_thispack;

  par_for("pgen_gauge_shift", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min, &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min, &x2max = size.d_view(m).x2max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
    Real rho = rho0;
    if (itype == 1) { rho = rho0*(1.0 + amp*sin(2.0*M_PI*(x1v - x1m)/lx1)); }
    w0(m,IDN,k,j,i) = rho;
    w0(m,IVX,k,j,i) = lorentz*v1;      // dyn_grmhd primitives carry W v^i
    w0(m,IVY,k,j,i) = lorentz*v2;
    w0(m,IVZ,k,j,i) = lorentz*v3;
    w0(m,IPR,k,j,i) = p0;

    // face fields from the corner vector potential A_z (zero unless field_loop)
    Real dx1 = size.d_view(m).dx1, dx2 = size.d_view(m).dx2;
    auto az = [=](Real x, Real y) {
      Real r = sqrt((x - x1c)*(x - x1c) + (y - x2c)*(y - x2c));
      return (itype == 2 && r < rloop) ? b0*(rloop - r) : 0.0;
    };
    Real x1f = LeftEdgeX(i-is, indcs.nx1, x1min, x1max);
    Real x2f = LeftEdgeX(j-js, indcs.nx2, x2min, x2max);
    bf.x1f(m,k,j,i) = (az(x1f, x2f + dx2) - az(x1f, x2f))/dx2;
    bf.x2f(m,k,j,i) = -(az(x1f + dx1, x2f) - az(x1f, x2f))/dx1;
    bf.x3f(m,k,j,i) = 0.0;
    if (i == ie) {
      bf.x1f(m,k,j,i+1) = (az(x1f + dx1, x2f + dx2) - az(x1f + dx1, x2f))/dx2;
    }
    if (j == je) {
      bf.x2f(m,k,j+1,i) = -(az(x1f + dx1, x2f + dx2) - az(x1f, x2f + dx2))/dx1;
    }
    if (k == ke) { bf.x3f(m,k+1,j,i) = 0.0; }
  });
  par_for("pgen_gauge_shift_bcc", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    bcc0(m,IBX,k,j,i) = 0.5*(bf.x1f(m,k,j,i) + bf.x1f(m,k,j,i+1));
    bcc0(m,IBY,k,j,i) = 0.5*(bf.x2f(m,k,j,i) + bf.x2f(m,k,j+1,i));
    bcc0(m,IBZ,k,j,i) = 0.5*(bf.x3f(m,k,j,i) + bf.x3f(m,k+1,j,i));
  });

  pmbp->padm->SetADMVariables(pmbp);
  pmbp->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);
  return;
}

namespace {

//! Flat metric with a uniform shift, over every cell (ghosts included).
void SetADMVariablesFlatShift(MeshBlockPack *pmbp) {
  auto &adm = pmbp->padm->adm;
  auto &u_adm = pmbp->padm->u_adm;
  int nmb = pmbp->nmb_thispack;
  int n1 = u_adm.extent_int(4), n2 = u_adm.extent_int(3), n3 = u_adm.extent_int(2);
  const Real s0 = gshift[0], s1 = gshift[1], s2 = gshift[2];
  par_for("adm_flat_shift", DevExeSpace(), 0, nmb-1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    adm.alpha(m,k,j,i) = 1.0;
    adm.beta_u(m,0,k,j,i) = s0;
    adm.beta_u(m,1,k,j,i) = s1;
    adm.beta_u(m,2,k,j,i) = s2;
    adm.psi4(m,k,j,i) = 1.0;
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        adm.g_dd(m,a,b,k,j,i) = (a == b) ? 1.0 : 0.0;
        adm.vK_dd(m,a,b,k,j,i) = 0.0;
      }
    }
  });
}

}  // namespace
