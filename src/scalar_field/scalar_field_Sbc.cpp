//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file scalar_field_Sbc.cpp
//! \brief Yukawa-consistent Sommerfeld outer boundary condition for the scalar field.
//!
//! Mirrors z4c/z4c_Sbc.cpp's structure exactly (same pseudoradial-vector RHS-replacement
//! pattern, applied only at outflow/diode/vacuum/user faces), but with the massive-field
//! generalization of the falloff ODE:
//!
//!   rhs(f) = -(f - f_inf)/r - m*(f - f_inf) - s^i*di(f - f_inf)
//!
//! where m = sqrt(mass2) and f_inf is the field's asymptotic value: sphi0 for sphi, 0 for
//! Pi (a static/quasi-static configuration has dt(sphi)->0 at infinity, so Pi->0 there).
//! Reduces exactly to Z4c's massless Sommerfeld form when mass2=0 (di(f_inf)=0 always,
//! since f_inf is a constant). Cross-checked against
//! ~/SACRA_2D/SACRA_MPI/boundary.f90's own outer-boundary treatment: it extrapolates
//! toward `asym` (sphi00 for sphi, 0 for Pi) with an exponential rate `exp_drop = pmass`
//! (i.e. m, not m^2) for both components -- the same (f_inf, m) pairing used here, even
//! though SACRA implements it via buffer-zone extrapolation rather than an RHS-replacement
//! ODE (AthenaK's existing Z4c convention, used here for consistency with this codebase).

#include <math.h>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "scalar_field/scalar_field.hpp"
#include "coordinates/cell_locations.hpp"

namespace scalarfield {

//----------------------------------------------------------------------------------------
//! \fn void ScalarFieldSommerfeld
//! \brief apply the Yukawa Sommerfeld BC at a single boundary point
KOKKOS_INLINE_FUNCTION
static void ScalarFieldSommerfeld(const ScalarField::ScalarField_vars& sf,
    const ScalarField::ScalarField_vars& rhs, Real sphi0, Real sf_mass,
    const RegionIndcs &indcs, const DualArray1D<RegionSize> &size,
    const int m, const int k, const int j, const int i) {
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dsphi_d;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dpi_d;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> s_u;

  Real idx[] = {1./size.d_view(m).dx1, 1./size.d_view(m).dx2, 1./size.d_view(m).dx3};

  // We force all derivatives to be calculated at second-order, matching Z4cSommerfeld
  // (found necessary for stability there; same finite-difference stencil as boundary
  // points can't reach far enough for a higher-order stencil).
  for (int a = 0; a < 3; a++) {
    dsphi_d(a) = Dx<2>(a, idx, sf.sphi, m, k, j, i);
    dpi_d(a)   = Dx<2>(a, idx, sf.vpi,  m, k, j, i);
  }

  Real &x1min = size.d_view(m).x1min;
  Real &x1max = size.d_view(m).x1max;
  Real &x2min = size.d_view(m).x2min;
  Real &x2max = size.d_view(m).x2max;
  Real &x3min = size.d_view(m).x3min;
  Real &x3max = size.d_view(m).x3max;

  Real x1v = CellCenterX(i-indcs.is, indcs.nx1, x1min, x1max);
  Real x2v = CellCenterX(j-indcs.js, indcs.nx2, x2min, x2max);
  Real x3v = CellCenterX(k-indcs.ks, indcs.nx3, x3min, x3max);

  Real r = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
  s_u(0) = x1v/r;
  s_u(1) = x2v/r;
  s_u(2) = x3v/r;

  Real dsphi = sf.sphi(m,k,j,i) - sphi0;
  Real dpi   = sf.vpi(m,k,j,i);  // Pi's asymptotic value is 0

  rhs.sphi(m,k,j,i) = -dsphi/r - sf_mass*dsphi;
  rhs.vpi(m,k,j,i)  = -dpi/r   - sf_mass*dpi;
  for (int a = 0; a < 3; a++) {
    rhs.sphi(m,k,j,i) -= s_u(a) * dsphi_d(a);
    rhs.vpi(m,k,j,i)  -= s_u(a) * dpi_d(a);
  }
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus ScalarField::ScalarFieldBoundaryRHS
//! \brief applies the Yukawa Sommerfeld BC on outflow-type boundaries; a no-op
//! (rhs unchanged) elsewhere. Structured identically to Z4c::Z4cBoundaryRHS.
TaskStatus ScalarField::ScalarFieldBoundaryRHS(Driver *pdriver, int stage) {
  auto &pm = pmy_pack->pmesh;
  auto &mb_bcs = pmy_pack->pmb->mb_bcs;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;

  int nmb = pmy_pack->nmb_thispack;
  int is = indcs.is;
  int ie = indcs.ie;
  int js = indcs.js;
  int je = indcs.je;
  int ks = indcs.ks;
  int ke = indcs.ke;

  auto &sf_ = sf;
  auto &rhs_ = rhs;
  Real sphi0 = opt.sphi0;
  Real sf_mass = sqrt(opt.mass2);
  bool &user_Sbc = opt.user_Sbc;

  if (pm->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::outflow
      || pm->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::diode
      || pm->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::vacuum
      || pm->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::user
      || pm->mesh_bcs[BoundaryFace::outer_x1] == BoundaryFlag::outflow
      || pm->mesh_bcs[BoundaryFace::outer_x1] == BoundaryFlag::diode
      || pm->mesh_bcs[BoundaryFace::outer_x1] == BoundaryFlag::vacuum
      || pm->mesh_bcs[BoundaryFace::outer_x1] == BoundaryFlag::user) {
    par_for("sfrhs_bc_x1", DevExeSpace(), 0, (nmb-1), ks, ke, js, je,
    KOKKOS_LAMBDA(int m, int k, int j) {
      switch (mb_bcs.d_view(m,BoundaryFace::inner_x1)) {
        case BoundaryFlag::diode:
        case BoundaryFlag::vacuum:
        case BoundaryFlag::outflow:
            ScalarFieldSommerfeld(sf_, rhs_, sphi0, sf_mass, indcs, size, m, k, j, is);
          break;
        case BoundaryFlag::user:
            if (user_Sbc) {
              ScalarFieldSommerfeld(sf_, rhs_, sphi0, sf_mass, indcs, size, m, k, j, is);
            }
          break;
        default:
          break;
      }
      switch (mb_bcs.d_view(m,BoundaryFace::outer_x1)) {
        case BoundaryFlag::diode:
        case BoundaryFlag::vacuum:
        case BoundaryFlag::outflow:
            ScalarFieldSommerfeld(sf_, rhs_, sphi0, sf_mass, indcs, size, m, k, j, ie);
          break;
        case BoundaryFlag::user:
            if (user_Sbc) {
              ScalarFieldSommerfeld(sf_, rhs_, sphi0, sf_mass, indcs, size, m, k, j, ie);
            }
          break;
        default:
          break;
      }
    });
  }
  if (pm->mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::outflow
      || pm->mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::diode
      || pm->mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::vacuum
      || pm->mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::user
      || pm->mesh_bcs[BoundaryFace::outer_x2] == BoundaryFlag::outflow
      || pm->mesh_bcs[BoundaryFace::outer_x2] == BoundaryFlag::diode
      || pm->mesh_bcs[BoundaryFace::outer_x2] == BoundaryFlag::vacuum
      || pm->mesh_bcs[BoundaryFace::outer_x2] == BoundaryFlag::user) {
    par_for("sfrhs_bc_x2", DevExeSpace(), 0, (nmb-1), ks, ke, is, ie,
    KOKKOS_LAMBDA(int m, int k, int i) {
      switch (mb_bcs.d_view(m,BoundaryFace::inner_x2)) {
        case BoundaryFlag::diode:
        case BoundaryFlag::vacuum:
        case BoundaryFlag::outflow:
            ScalarFieldSommerfeld(sf_, rhs_, sphi0, sf_mass, indcs, size, m, k, js, i);
          break;
        case BoundaryFlag::user:
            if (user_Sbc) {
              ScalarFieldSommerfeld(sf_, rhs_, sphi0, sf_mass, indcs, size, m, k, js, i);
            }
          break;
        default:
          break;
      }
      switch (mb_bcs.d_view(m,BoundaryFace::outer_x2)) {
        case BoundaryFlag::diode:
        case BoundaryFlag::vacuum:
        case BoundaryFlag::outflow:
            ScalarFieldSommerfeld(sf_, rhs_, sphi0, sf_mass, indcs, size, m, k, je, i);
          break;
        case BoundaryFlag::user:
            if (user_Sbc) {
              ScalarFieldSommerfeld(sf_, rhs_, sphi0, sf_mass, indcs, size, m, k, je, i);
            }
          break;
        default:
          break;
      }
    });
  }
  if (pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::outflow
      || pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::diode
      || pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::vacuum
      || pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::user
      || pm->mesh_bcs[BoundaryFace::outer_x3] == BoundaryFlag::outflow
      || pm->mesh_bcs[BoundaryFace::outer_x3] == BoundaryFlag::diode
      || pm->mesh_bcs[BoundaryFace::outer_x3] == BoundaryFlag::vacuum
      || pm->mesh_bcs[BoundaryFace::outer_x3] == BoundaryFlag::user) {
    par_for("sfrhs_bc_x3", DevExeSpace(), 0, (nmb-1), js, je, is, ie,
    KOKKOS_LAMBDA(int m, int j, int i) {
      switch (mb_bcs.d_view(m,BoundaryFace::inner_x3)) {
        case BoundaryFlag::diode:
        case BoundaryFlag::vacuum:
        case BoundaryFlag::outflow:
            ScalarFieldSommerfeld(sf_, rhs_, sphi0, sf_mass, indcs, size, m, ks, j, i);
          break;
        case BoundaryFlag::user:
            if (user_Sbc) {
              ScalarFieldSommerfeld(sf_, rhs_, sphi0, sf_mass, indcs, size, m, ks, j, i);
            }
          break;
        default:
          break;
      }
      switch (mb_bcs.d_view(m,BoundaryFace::outer_x3)) {
        case BoundaryFlag::diode:
        case BoundaryFlag::vacuum:
        case BoundaryFlag::outflow:
            ScalarFieldSommerfeld(sf_, rhs_, sphi0, sf_mass, indcs, size, m, ke, j, i);
          break;
        case BoundaryFlag::user:
            if (user_Sbc) {
              ScalarFieldSommerfeld(sf_, rhs_, sphi0, sf_mass, indcs, size, m, ke, j, i);
            }
          break;
        default:
          break;
      }
    });
  }

  return TaskStatus::complete;
}

} // namespace scalarfield
