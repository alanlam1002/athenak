#ifndef DYN_GRMHD_RESISTIVE_RESISTIVE_PARAMS_HPP_
#define DYN_GRMHD_RESISTIVE_RESISTIVE_PARAMS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file resistive_params.hpp
//! \brief Input parameters for resistive GRMHD (staggered face-centred E, dyngr path).
//!
//! Keys (all in the <mhd> block, read only when <mhd>/resistive = true):
//!   resistive              = true|false   (default false; ideal GRMHD when false)
//!   resistive_sigma        = <Real >= 0>  uniform conductivity (required for uniform)
//!   resistive_sigma_model  = uniform|density
//!   resistive_sigma_in, resistive_sigma_out, resistive_rho_transition,
//!   resistive_sigma_width  (density model, see conductivity.hpp)
//!   resistive_rsolver      = mhllc|lf     (tetrad MHLLC, default; or light-speed LF)
//!   resistive_charge_flux  = upwind|rusanov
//!   resistive_newton_fallback = lagged_v|fail|blend (blend: 07 v2.8 §3.2.3(c)
//!                            ideal-limit blend; test-only prototype, like the four
//!                            keys below)
//!   resistive_floor_accept = false      (07 v2.8 §3.2.3(b) floored-root acceptance)
//!   resistive_floor_accept_tol = 1e-3   (FA condition 2, discarded energy/(tau + D))
//!   resistive_d4           = average|keep (keep: no hydro c2p in flagged cells,
//!                            §3.2.3(d))
//!   resistive_fail_reason_cycles = 0    (log failure reasons for cycles < N to
//!                            <base>.rfail.txt)
//!   resistive_newton_tol   = 1e-11
//!   resistive_newton_maxit = 30
//!   resistive_allow_incomplete = false  (developer switch while steps are unfinished)
//!   resistive_freeze_background = false (test only: subtract the initial discrete E~
//!                                        RHS every stage, i.e. eta-compensation of the
//!                                        initial equilibrium; no restart, no AMR)
//!   resistive_implicit_residual = c2p|momentum (decision 10 B: g or Mignone Eq. 56 f)
//!   sigma_max              = FLT_MAX (decision 10 A: rho >= B^2/sigma_max after c2p)
//!   resistive_hist_dt      = -1 (K5-run histogram interval; <= 0: hst output dt)

#include <string>

#include "athena.hpp"

class ParameterInput;

namespace resistive {

enum class ChargeFlux {upwind, rusanov};
enum class RSolver {mhllc, lf};
enum class SigmaModel {uniform, density};
enum class NewtonFallback {lagged_v, fail, blend};
enum class D4Mode {average, keep};
enum class ImplicitResidual {c2p, momentum};

struct ResistiveParams {
  bool enabled = false;
  bool allow_incomplete = false;
  SigmaModel sigma_model = SigmaModel::uniform;
  Real sigma = 0.0;
  Real sigma_in = 0.0, sigma_out = 0.0, rho_transition = 0.0, sigma_width = 0.0;
  RSolver rsolver = RSolver::mhllc;
  ChargeFlux charge_flux = ChargeFlux::upwind;
  NewtonFallback newton_fallback = NewtonFallback::lagged_v;
  Real newton_tol = 1.0e-11;
  int newton_maxit = 30;
  bool freeze_background = false;
  ImplicitResidual implicit_residual = ImplicitResidual::c2p;
  Real sigma_max = 0.0;   // <mhd> sigma_max: fluid density floor rho >= B^2/sigma_max
  Real hist_dt = -1.0;    // K5-run histogram output interval
  // 07 v2.8 §3.2.3 test-only prototype keys
  bool floor_accept = false;
  Real floor_accept_tol = 1.0e-3;
  D4Mode d4 = D4Mode::average;
  int fail_reason_cycles = 0;
  // any prototype key on: ResistiveC2P consumes the per-cell flags (D4 diagnostics/keep)
  bool Prototype() const {
    return newton_fallback == NewtonFallback::blend || floor_accept || d4 == D4Mode::keep;
  }
};

// True only if <block>/resistive exists and is true. Never adds keys to pin.
bool ResistiveEnabled(ParameterInput *pin, const std::string &block = "mhd");

// Parses and validates all keys; calls std::exit on invalid input. When resistive is
// off the returned struct has enabled=false and pin is not modified.
ResistiveParams ParseResistiveParams(ParameterInput *pin,
                                     const std::string &block = "mhd");

// Checks that the run configuration supports resistive GRMHD (dyngr path, ImEx
// integrator). Calls std::exit otherwise.
void CheckResistiveConfiguration(const ResistiveParams &rp,
                                 bool is_dynamical_relativistic,
                                 const std::string &integrator);

}  // namespace resistive

#endif  // DYN_GRMHD_RESISTIVE_RESISTIVE_PARAMS_HPP_
