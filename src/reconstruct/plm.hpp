#ifndef RECONSTRUCT_PLM_HPP_
#define RECONSTRUCT_PLM_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file plm.hpp
//! \brief  piecewise linear reconstruction implemented as inline functions
//! Works for both uniform and non-uniform mesh spacing, including all curvilinear
//! coordinate systems (Task B6). Formula ported (math only) from old Athena++'s
//! src/reconstruct/plm.cpp (verified against that reference before being written here,
//! see DEVELOPMENT.md Task B6 log): the generalized van Leer limiter with the Mignone
//! (2014) eq. 33/37 correction for non-uniform/curvilinear centroid spacing. Reduces
//! exactly to the simplified uniform-Cartesian van Leer formula this file previously
//! contained when x_ip1-x_i == x_i-x_im1 == dx1f and xf_i/xf_ip1 sit exactly halfway
//! between centroids.

#include <math.h>
#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \fn PLM()
//! \brief Reconstructs linear slope in cell i to compute ql(i+1) and qr(i). Works for
//! reconstruction in any dimension by passing in the appropriate q_im1, q_i, and q_ip1,
//! together with the corresponding CENTROID positions x_im1, x_i, x_ip1 (NOT necessarily
//! evenly spaced -- e.g. cylindrical/spherical radial centroids) and the FACE positions
//! xf_i, xf_ip1 bounding cell i (needed because the centroid-to-face offset is generally
//! NOT half the cell width for a non-uniform/curvilinear grid).
//!
//! Exactness property (verified algebraically, see DEVELOPMENT.md): if q is an exactly
//! linear function of the stored centroid x (q_im1/i/ip1 = a + b*x_im1/i/ip1 for any a,b),
//! this reconstructs the SAME linear function evaluated exactly at the true face
//! positions xf_i/xf_ip1, regardless of how non-uniform the spacing is.

KOKKOS_INLINE_FUNCTION
void PLM(const Real &q_im1, const Real &q_i, const Real &q_ip1,
         const Real &x_im1, const Real &x_i, const Real &x_ip1,
         const Real &xf_i, const Real &xf_ip1,
         Real &ql_ip1, Real &qr_i) {
  // compute L/R differences
  Real dwl = (q_i - q_im1);
  Real dwr = (q_ip1 - q_i);

  Real dx1f = xf_ip1 - xf_i;         // plain width of cell i
  Real dx1v_fwd = x_ip1 - x_i;       // forward centroid spacing ("dx1v(i)")
  Real dx1v_bwd = x_i - x_im1;       // backward centroid spacing ("dx1v(i-1)")

  // Mignone (2014) eq. 33/37 generalized van Leer limiter for non-uniform spacing
  Real dqF = dwr*dx1f/dx1v_fwd;
  Real dqB = dwl*dx1f/dx1v_bwd;
  Real dq2 = dqF*dqB;

  Real cf = dx1v_fwd/(xf_ip1 - x_i);
  Real cb = dx1v_bwd/(x_i - xf_i);

  Real dqm = dq2*(cf*dqB + cb*dqF)/(SQR(dqB) + SQR(dqF) + dq2*(cf + cb - 2.0));
  if (dq2 <= 0.0) dqm = 0.0;

  // compute ql_(i+1/2) and qr_(i-1/2) using limited slope and the true face offsets
  ql_ip1 = q_i + ((xf_ip1 - x_i)/dx1f)*dqm;
  qr_i   = q_i - ((x_i - xf_i)/dx1f)*dqm;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn PLM() [uniform-spacing overload]
//! \brief Simplified van Leer limiter for callers with no spatial position/geometry
//! information at all -- e.g. src/radiation/radiation_fluxes.cpp, which reconstructs
//! intensities across ANGULAR bins on a geodesic grid, not physical space. This overload
//! is unrelated to Task B6's curvilinear-coordinate work (angular bins are not spatial
//! coordinates, so no GeomData involvement is applicable), and is kept unchanged from
//! this file's pre-B6 version specifically so that caller is left untouched.

KOKKOS_INLINE_FUNCTION
void PLM(const Real &q_im1, const Real &q_i, const Real &q_ip1,
         Real &ql_ip1, Real &qr_i) {
  Real dql = (q_i - q_im1);
  Real dqr = (q_ip1 - q_i);
  Real dq2 = dql*dqr;
  Real dqm = dq2/(dql + dqr);
  if (dq2 <= 0.0) dqm = 0.0;
  ql_ip1 = q_i + dqm;
  qr_i   = q_i - dqm;
  return;
}
#endif // RECONSTRUCT_PLM_HPP_
