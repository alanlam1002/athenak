#ifndef DYN_GRMHD_RESISTIVE_RESISTIVE_EDGE_HPP_
#define DYN_GRMHD_RESISTIVE_RESISTIVE_EDGE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file resistive_edge.hpp
//! \brief Edge (2D Riemann) solver and face charge flux for resistive GRMHD
//!        (07_scoping_v2 §2.2, §2.3; Mignone et al. 2019 Eqs. 46-50).
//!
//! The edge is along k; (a, b) are the cyclic transverse directions (k=3: a=1, b=2).
//! With the AthenaK CT convention dB~^a/dt = -d_b e_k + ..., the edge fluxes are
//!     e_k = alpha E_k + beta^a B~^b - beta^b B~^a      (Faraday, E_k covariant)
//!     h_k = -alpha B_k + beta^a E~^b - beta^b E~^a     (Ampere, B_k covariant)
//! with B~ = sqrt(gamma) B and E~ = sqrt(gamma) E, and Ẽ is advanced by the same CT
//! kernel with e -> h. The upwind edge value from four states (Londrillo & Del Zanna
//! 2004 HLL form; Mignone 2019 Eqs. 46-47 for unit light speeds) is
//!     f* = sum_{s} w_s f_s - phi_b (U^a_R - U^a_L)|_b + phi_a (U^b_R - U^b_L)|_a
//! where U = B~ for f = e and U = E~ for f = h, w_s are the HLL weights and
//! phi = a+ a- / (a+ + a-), with light-cone speeds lambda± = -beta^n ± alpha sqrt(g^nn).

#include <cmath>

#include "athena.hpp"

namespace resistive {

// a+ = max(0, lambda+), a- = max(0, -lambda-) for light along direction n.
KOKKOS_INLINE_FUNCTION
void LightSpeeds(const Real alpha, const Real beta_n, const Real gup_nn,
                 Real &ap, Real &am) {
  const Real c = alpha*sqrt(gup_nn);
  ap = fmax(0.0, -beta_n + c);
  am = fmax(0.0, beta_n + c);
}

// f[ia][ib]: edge states, index 0 = L (minus side), 1 = R (plus side) along a and b.
// ua_jump_b = U^a(R in b) - U^a(L in b), the a-face normal field across b.
// ub_jump_a = U^b(R in a) - U^b(L in a), the b-face normal field across a.
KOKKOS_INLINE_FUNCTION
Real EdgeUpwind(const Real f[2][2], const Real ua_jump_b, const Real ub_jump_a,
                const Real aap, const Real aam, const Real abp, const Real abm) {
  const Real ia = 1.0/(aap + aam);
  const Real ib = 1.0/(abp + abm);
  const Real favg = (aap*(abp*f[0][0] + abm*f[0][1]) +
                     aam*(abp*f[1][0] + abm*f[1][1]))*ia*ib;
  return favg - abp*abm*ib*ua_jump_b + aap*aam*ia*ub_jump_a;
}

// Edge-flux states. Ek_cov / Bk_cov are covariant undensitised components along k;
// (Ua, Ub) are the densitised transverse fields (B~ for e, E~ for h).
KOKKOS_INLINE_FUNCTION
Real EdgeEMF(const Real alpha, const Real beta_a, const Real beta_b, const Real Ek_cov,
             const Real bt_a, const Real bt_b) {
  return alpha*Ek_cov + beta_a*bt_b - beta_b*bt_a;
}

KOKKOS_INLINE_FUNCTION
Real EdgeAmpere(const Real alpha, const Real beta_a, const Real beta_b,
                const Real Bk_cov, const Real et_a, const Real et_b) {
  return -alpha*Bk_cov + beta_a*et_b - beta_b*et_a;
}

// Explicit convective charge flux through a face, upwinded on the mass flux:
//   (q~ (alpha v^n - beta^n))_f ~ (q~/D~)_upwind * F_IDN       (07 §2.3)
KOKKOS_INLINE_FUNCTION
Real ChargeFluxUpwind(const Real f_idn, const Real ql, const Real dl, const Real qr,
                      const Real dr) {
  return (f_idn >= 0.0) ? f_idn*ql/dl : f_idn*qr/dr;
}

// Rusanov alternative (Mignone 2019 Eq. 49): w = alpha v^n - beta^n, a = max |lambda|.
KOKKOS_INLINE_FUNCTION
Real ChargeFluxRusanov(const Real ql, const Real wl, const Real qr, const Real wr,
                       const Real a) {
  return 0.5*(ql*wl + qr*wr) - 0.5*a*(qr - ql);
}

}  // namespace resistive

#endif  // DYN_GRMHD_RESISTIVE_RESISTIVE_EDGE_HPP_
