#ifndef DYN_GRMHD_RESISTIVE_RESISTIVE_EM_HPP_
#define DYN_GRMHD_RESISTIVE_RESISTIVE_EM_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file resistive_em.hpp
//! \brief Device-inline 3+1 electromagnetic helpers for resistive GRMHD.
//!
//! Conventions (07_scoping_v2 §2; 02_derivation §4):
//!  * E^i, B^i are Eulerian (normal-observer) contravariant fields, undensitised.
//!    AthenaK fixed-metric code B is B_Eul/alpha; dyngr B is B_Eul.
//!  * eps_ijk = sqrt(gamma)[ijk], eps^ijk = [ijk]/sqrt(gamma).
//!  * dyngr      : IDN = sg D, IM_i = sg S_i, IEN = sg tau          (sg = sqrt(gamma))
//!  * fixed GR   : IDN = sg D, IM_i = sg S_i, IEN = -alpha sg calE + beta^i IM_i + IDN
//!    with calE = tau + D  (= T^t_t + D densitised by sqrt(-g) = alpha sg).
//!  * EM parts   : S^EM_i = eps_ijk E^j B^k,  calE^EM = (E^2 + B^2)/2.
//! Metric arrays use the PrimitiveSolver ordering g3d[S11,S12,S13,S22,S23,S33].

#include <cmath>

#include "athena.hpp"
#include "eos/primitive-solver/ps_types.hpp"

namespace resistive {

KOKKOS_INLINE_FUNCTION
void Lower(const Real g[NSPMETRIC], const Real vu[3], Real vd[3]) {
  vd[0] = g[S11]*vu[0] + g[S12]*vu[1] + g[S13]*vu[2];
  vd[1] = g[S12]*vu[0] + g[S22]*vu[1] + g[S23]*vu[2];
  vd[2] = g[S13]*vu[0] + g[S23]*vu[1] + g[S33]*vu[2];
}

KOKKOS_INLINE_FUNCTION
Real Dot(const Real g[NSPMETRIC], const Real a[3], const Real b[3]) {
  Real ad[3];
  Lower(g, a, ad);
  return ad[0]*b[0] + ad[1]*b[1] + ad[2]*b[2];
}

KOKKOS_INLINE_FUNCTION
Real SqrtDet(const Real g[NSPMETRIC]) {
  const Real det = g[S11]*g[S22]*g[S33] + 2.0*g[S12]*g[S13]*g[S23]
                 - (g[S11]*g[S23]*g[S23] + g[S12]*g[S12]*g[S33] + g[S13]*g[S13]*g[S22]);
  return sqrt(det);
}

// c_i = eps_ijk a^j b^k  (contravariant inputs, covariant output)
KOKKOS_INLINE_FUNCTION
void CrossLow(const Real sg, const Real a[3], const Real b[3], Real c[3]) {
  c[0] = sg*(a[1]*b[2] - a[2]*b[1]);
  c[1] = sg*(a[2]*b[0] - a[0]*b[2]);
  c[2] = sg*(a[0]*b[1] - a[1]*b[0]);
}

// c^i = eps^ijk a_j b_k  (covariant inputs, contravariant output)
KOKKOS_INLINE_FUNCTION
void CrossUp(const Real sg, const Real a[3], const Real b[3], Real c[3]) {
  const Real isg = 1.0/sg;
  c[0] = isg*(a[1]*b[2] - a[2]*b[1]);
  c[1] = isg*(a[2]*b[0] - a[0]*b[2]);
  c[2] = isg*(a[0]*b[1] - a[1]*b[0]);
}

// Ideal-MHD electric field E^i = -eps^ijk v_j B_k (v^i Eulerian 3-velocity).
KOKKOS_INLINE_FUNCTION
void IdealE(const Real g[NSPMETRIC], const Real sg, const Real v[3], const Real B[3],
            Real E[3]) {
  Real vd[3], Bd[3];
  Lower(g, v, vd);
  Lower(g, B, Bd);
  CrossUp(sg, vd, Bd, E);
  E[0] = -E[0]; E[1] = -E[1]; E[2] = -E[2];
}

// Comoving b^2 (b^mu = -*F^{mu nu} u_nu) from the Eulerian E^i, B^i and u^i = W v^i:
// b^2 = W^2 [(B - v x E)^2 - (v.B)^2], (v x E)^i = eps^ijk v_j E_k. Ideal limit:
// B^2/W^2 + (v.B)^2.
KOKKOS_INLINE_FUNCTION
Real ComovingB2(const Real g[NSPMETRIC], const Real sg, const Real u[3], const Real E[3],
                const Real B[3]) {
  const Real W = sqrt(1.0 + Dot(g, u, u));
  const Real v[3] = {u[0]/W, u[1]/W, u[2]/W};
  Real vd[3], Ed[3], vxe[3];
  Lower(g, v, vd);
  Lower(g, E, Ed);
  CrossUp(sg, vd, Ed, vxe);
  const Real d[3] = {B[0] - vxe[0], B[1] - vxe[1], B[2] - vxe[2]};
  const Real vb = vd[0]*B[0] + vd[1]*B[1] + vd[2]*B[2];
  return W*W*(Dot(g, d, d) - vb*vb);
}

// Undensitised EM contributions: S^EM_i = eps_ijk E^j B^k, calE^EM = (E^2+B^2)/2.
KOKKOS_INLINE_FUNCTION
void EMParts(const Real g[NSPMETRIC], const Real sg, const Real E[3], const Real B[3],
             Real s_em[3], Real &e_em) {
  CrossLow(sg, E, B, s_em);
  e_em = 0.5*(Dot(g, E, E) + Dot(g, B, B));
}

// Densitised increments to (IM_i, IEN) from adding the EM field, dyngr convention.
KOKKOS_INLINE_FUNCTION
void EMIncrementDyn(const Real g[NSPMETRIC], const Real sg, const Real E[3],
                    const Real B[3], Real dim[3], Real &dien) {
  Real s_em[3], e_em;
  EMParts(g, sg, E, B, s_em, e_em);
  for (int a = 0; a < 3; ++a) dim[a] = sg*s_em[a];
  dien = sg*e_em;
}

// Same for the fixed-metric (T^t_t + D) convention.
KOKKOS_INLINE_FUNCTION
void EMIncrementFixed(const Real g[NSPMETRIC], const Real sg, const Real alpha,
                      const Real beta[3], const Real E[3], const Real B[3],
                      Real dim[3], Real &dien) {
  Real s_em[3], e_em;
  EMParts(g, sg, E, B, s_em, e_em);
  for (int a = 0; a < 3; ++a) dim[a] = sg*s_em[a];
  dien = -alpha*sg*e_em + beta[0]*dim[0] + beta[1]*dim[1] + beta[2]*dim[2];
}

// Conversions between the two densitised energy variables (IDN, IM shared).
KOKKOS_INLINE_FUNCTION
Real FixedIENFromDynIEN(const Real ien_dyn, const Real idn, const Real im[3],
                        const Real alpha, const Real beta[3]) {
  return -alpha*(ien_dyn + idn) + beta[0]*im[0] + beta[1]*im[1] + beta[2]*im[2] + idn;
}

KOKKOS_INLINE_FUNCTION
Real DynIENFromFixedIEN(const Real ien_fix, const Real idn, const Real im[3],
                        const Real alpha, const Real beta[3]) {
  return -(ien_fix - idn - (beta[0]*im[0] + beta[1]*im[1] + beta[2]*im[2]))/alpha - idn;
}

// Full densitised prim->cons (dyngr convention) for a fluid with rest-mass density rho,
// pressure p, specific enthalpy h, normal-frame u^i = W v^i, plus Eulerian E, B.
KOKKOS_INLINE_FUNCTION
void PrimToConsDyn(const Real g[NSPMETRIC], const Real rho, const Real p, const Real h,
                   const Real wv[3], const Real E[3], const Real B[3],
                   Real &idn, Real im[3], Real &ien) {
  const Real sg = SqrtDet(g);
  const Real W = sqrt(1.0 + Dot(g, wv, wv));
  Real wvd[3];
  Lower(g, wv, wvd);
  Real dim[3], dien;
  EMIncrementDyn(g, sg, E, B, dim, dien);
  const Real D = rho*W;
  idn = sg*D;
  for (int a = 0; a < 3; ++a) im[a] = sg*rho*h*W*wvd[a] + dim[a];
  ien = sg*(rho*h*W*W - p - D) + dien;
}

}  // namespace resistive

#endif  // DYN_GRMHD_RESISTIVE_RESISTIVE_EM_HPP_
