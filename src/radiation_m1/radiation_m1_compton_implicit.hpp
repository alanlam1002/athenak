#ifndef RADIATION_M1_COMPTON_IMPLICIT_HPP
#define RADIATION_M1_COMPTON_IMPLICIT_HPP
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_compton_implicit.hpp
//  \brief closed-form quartic pre-solve for T_gas/J self-consistency under
//  Compton exchange, ported from the discrete-ordinate radiation module's
//  FourthPolyRoot (src/radiation/radiation_source.cpp) -- see DEVELOPMENT.md's
//  "Compton implementation" section for the derivation.

#include "athena.hpp"

namespace radiationm1 {

//----------------------------------------------------------------------------------------
//! \fn bool radiationm1::FourthPolyRoot
//  \brief Exact solution for a fourth order polynomial of the form
//  coef4 * x^4 + x + tconst = 0, ported verbatim from the discrete-ordinate
//  radiation module (src/radiation/radiation_source.cpp:396-435).
KOKKOS_INLINE_FUNCTION
bool FourthPolyRoot(const Real coef4, const Real tconst, Real &root) {
  // Calculate real root of z^3 - 4*tconst/coef4 * z - 1/coef4^2 = 0
  Real ccubic = tconst * tconst * tconst;
  Real delta1 = 0.25 - 64.0 * ccubic * coef4 / 27.0;
  if (delta1 < 0.0) {
    return false;
  }
  delta1 = Kokkos::sqrt(delta1);
  if (delta1 < 0.5) {
    return false;
  }
  Real zroot;
  if (delta1 > 1.0e11) {  // to avoid small number cancellation
    zroot = Kokkos::pow(delta1, -2.0 / 3.0) / 3.0;
  } else {
    zroot = Kokkos::pow(0.5 + delta1, 1.0 / 3.0) - Kokkos::pow(-0.5 + delta1, 1.0 / 3.0);
  }
  if (zroot < 0.0) {
    return false;
  }
  zroot *= Kokkos::pow(coef4, -2.0 / 3.0);

  // Calculate quartic root using cubic root
  Real rcoef = Kokkos::sqrt(zroot);
  Real delta2 = -zroot + 2.0 / (coef4 * rcoef);
  if (delta2 < 0.0) {
    return false;
  }
  delta2 = Kokkos::sqrt(delta2);
  root = 0.5 * (delta2 - rcoef);
  if (root < 0.0) {
    return false;
  }
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn bool radiationm1::SolveComptonQuartic
//  \brief Jointly solves for a self-consistent (T_new, J_new) pair under
//  backward-Euler matter/radiation exchange, linearizing the exchange rate
//  at the "star" (pre-coupling) state so the result reduces to a depressed
//  quartic in T_new -- mirrors the discrete-ordinate module's
//  RadFluidCoupling Compton solve (radiation_source.cpp:290-330), but built
//  from M1's own combined Compton+Planck rate law
//  (radiation_m1_calc_opacities_photons.cpp:
//  sigma_tot = kappa_s*rho*4*T*inv_t_electron + rho*kappa_p, driving J
//  toward arad*T^4). Despite the name/file history, this now covers the
//  Planck-only channel too (kappa_s=0): the two channels share the same
//  LTE target and their rates simply add, so a single joint solve handles
//  either or both. Assumes Primitive::IdealGas (T=P/rho, e_int=rho*T/gm1);
//  safe since EOSCompOSE hard-errors upstream at parse time whenever this
//  path could be reached.
//
//  volform (sqrt of the spatial metric determinant) is carried through so
//  the radiation-side emission term and the Jstar/J_new bookkeeping stay
//  exact in curved space; the gas-side proper-time factor folded into dtau
//  (dtau = beta_dt*alpha/w_lorentz) is an estimate-only approximation away
//  from the static, flat-space case (see DEVELOPMENT.md).
//
//  Returns false (T_new/J_new left untouched) if no valid positive real
//  root exists -- the caller should then fall back to the existing
//  frozen-opacity path for that cell.
KOKKOS_INLINE_FUNCTION
bool SolveComptonQuartic(const Real rho, const Real T_star, const Real Jstar,
                         const Real gm1, const Real kappa_s, const Real kappa_p,
                         const Real arad, const Real inv_t_electron,
                         const Real dtau, const Real volform, Real &T_new,
                         Real &J_new) {
  if (!(rho > 0.0) || !(T_star > 0.0) || !(gm1 > 0.0)) {
    return false;
  }

  // Combined rate: Compton (T-dependent) + Planck (rho*kappa_p, no
  // additional T-dependence in the rate itself -- both channels drive J
  // toward the same arad*T^4 target, so their rates simply add).
  const Real sigma_c_star = kappa_s * rho * 4.0 * T_star * inv_t_electron;
  const Real sigma_tot_star = sigma_c_star + rho * kappa_p;
  const Real rho_over_gm1 = rho / gm1;
  const Real one_plus_dtau_sigma = 1.0 + dtau * sigma_tot_star;

  // Etot_star = Jstar + (rho/gm1)*T_star, conserved by (i)+(rho/gm1)*(ii)
  const Real Etot_star = Jstar + rho_over_gm1 * T_star;

  const Real denom = rho_over_gm1 * one_plus_dtau_sigma;
  if (!(denom > 0.0)) {
    return false;
  }

  // No matter coupling at all (kappa_s=kappa_p=0): coef4=0 degenerates the
  // quartic to a linear equation (T_new=-tconst); FourthPolyRoot's
  // cubic-resolvent algebra divides by coef4 and would blow up here, so
  // solve the linear case directly instead of calling it.
  if (sigma_tot_star <= 0.0) {
    T_new = T_star;
    J_new = Jstar;
    return true;
  }

  const Real coef4 = dtau * sigma_tot_star * arad * volform / denom;
  const Real tconst = (Jstar - one_plus_dtau_sigma * Etot_star) / denom;

  Real root{};
  if (!FourthPolyRoot(coef4, tconst, root)) {
    return false;
  }
  if (!(root > 0.0) || !Kokkos::isfinite(root)) {
    return false;
  }

  T_new = root;
  J_new = Etot_star - rho_over_gm1 * T_new;
  if (!(J_new >= 0.0) || !Kokkos::isfinite(J_new)) {
    return false;
  }
  return true;
}

}  // namespace radiationm1
#endif  // RADIATION_M1_COMPTON_IMPLICIT_HPP
