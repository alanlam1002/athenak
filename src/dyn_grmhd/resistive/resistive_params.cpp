//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file resistive_params.cpp
//! \brief Parsing and validation of resistive GRMHD input keys.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "dyn_grmhd/resistive/resistive_params.hpp"

namespace resistive {

namespace {

[[noreturn]] void Fatal(int line, const std::string &msg) {
  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << line << std::endl
            << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

const std::vector<std::string> &ResistiveKeys() {
  static const std::vector<std::string> keys = {
    "resistive_sigma", "resistive_sigma_model", "resistive_sigma_in",
    "resistive_sigma_out", "resistive_rho_transition", "resistive_sigma_width",
    "resistive_charge_flux", "resistive_newton_fallback", "resistive_newton_tol",
    "resistive_newton_maxit", "resistive_allow_incomplete"};
  return keys;
}

Real RequirePositive(ParameterInput *pin, const std::string &block,
                     const std::string &key, int line) {
  if (!pin->DoesParameterExist(block, key)) {
    Fatal(line, "<" + block + "> " + key + " is required for this sigma model");
  }
  Real v = pin->GetReal(block, key);
  if (!std::isfinite(v) || v <= 0.0) {
    Fatal(line, "<" + block + "> " + key + " must be finite and > 0");
  }
  return v;
}

}  // namespace

bool ResistiveEnabled(ParameterInput *pin, const std::string &block) {
  return pin->DoesParameterExist(block, "resistive") &&
         pin->GetBoolean(block, "resistive");
}

ResistiveParams ParseResistiveParams(ParameterInput *pin, const std::string &block) {
  ResistiveParams rp;
  if (!ResistiveEnabled(pin, block)) {
    for (const auto &k : ResistiveKeys()) {
      if (pin->DoesParameterExist(block, k)) {
        std::cout << "### WARNING: <" << block << "> " << k << " is ignored because "
                  << "resistive is not enabled" << std::endl;
      }
    }
    return rp;
  }
  rp.enabled = true;
  rp.allow_incomplete = pin->GetOrAddBoolean(block, "resistive_allow_incomplete", false);

  std::string model = pin->GetOrAddString(block, "resistive_sigma_model", "uniform");
  if (model == "uniform") {
    rp.sigma_model = SigmaModel::uniform;
    if (!pin->DoesParameterExist(block, "resistive_sigma")) {
      Fatal(__LINE__, "<" + block + "> resistive_sigma is required when resistive = true "
            "and resistive_sigma_model = uniform");
    }
    rp.sigma = pin->GetReal(block, "resistive_sigma");
    if (!std::isfinite(rp.sigma) || rp.sigma < 0.0) {
      Fatal(__LINE__, "<" + block + "> resistive_sigma must be finite and >= 0");
    }
  } else if (model == "density") {
    rp.sigma_model = SigmaModel::density;
    rp.sigma_in = RequirePositive(pin, block, "resistive_sigma_in", __LINE__);
    rp.sigma_out = RequirePositive(pin, block, "resistive_sigma_out", __LINE__);
    rp.rho_transition = RequirePositive(pin, block, "resistive_rho_transition", __LINE__);
    rp.sigma_width = RequirePositive(pin, block, "resistive_sigma_width", __LINE__);
  } else {
    Fatal(__LINE__, "<" + block + "> resistive_sigma_model = '" + model +
          "' not recognised; use uniform|density");
  }

  std::string cflux = pin->GetOrAddString(block, "resistive_charge_flux", "upwind");
  if (cflux == "upwind") {
    rp.charge_flux = ChargeFlux::upwind;
  } else if (cflux == "rusanov") {
    rp.charge_flux = ChargeFlux::rusanov;
  } else {
    Fatal(__LINE__, "<" + block + "> resistive_charge_flux = '" + cflux +
          "' not recognised; use upwind|rusanov");
  }

  std::string fb = pin->GetOrAddString(block, "resistive_newton_fallback", "lagged_v");
  if (fb == "lagged_v") {
    rp.newton_fallback = NewtonFallback::lagged_v;
  } else if (fb == "fail") {
    rp.newton_fallback = NewtonFallback::fail;
  } else {
    Fatal(__LINE__, "<" + block + "> resistive_newton_fallback = '" + fb +
          "' not recognised; use lagged_v|fail");
  }

  rp.newton_tol = pin->GetOrAddReal(block, "resistive_newton_tol", 1.0e-11);
  if (!std::isfinite(rp.newton_tol) || rp.newton_tol <= 0.0 || rp.newton_tol >= 1.0) {
    Fatal(__LINE__, "<" + block + "> resistive_newton_tol must be in (0,1)");
  }
  rp.newton_maxit = pin->GetOrAddInteger(block, "resistive_newton_maxit", 30);
  if (rp.newton_maxit < 1) {
    Fatal(__LINE__, "<" + block + "> resistive_newton_maxit must be >= 1");
  }
  return rp;
}

void CheckResistiveConfiguration(const ResistiveParams &rp,
                                 bool is_dynamical_relativistic,
                                 const std::string &integrator) {
  if (!rp.enabled) return;
  if (!is_dynamical_relativistic) {
    Fatal(__LINE__, "<mhd> resistive = true is implemented only on the dynamical-GR "
          "(dyngr) path; add an <adm> or <z4c> block (flat space: <coord> minkowski)");
  }
  if (integrator != "imex2" && integrator != "imex3" && integrator != "imex2+") {
    Fatal(__LINE__, "<mhd> resistive = true requires <time> integrator = "
          "imex2|imex3|imex2+ (got '" + integrator + "')");
  }
  if (!rp.allow_incomplete) {
    Fatal(__LINE__, "<mhd> resistive = true: the resistive update is not implemented yet "
          "(plan step 1a). Set resistive_allow_incomplete = true only for development.");
  }
  std::cout << "### WARNING: resistive_allow_incomplete = true; resistive GRMHD is "
            << "INCOMPLETE and results are not physical." << std::endl;
}

}  // namespace resistive
