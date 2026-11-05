#ifndef DYN_GRMHD_RESISTIVE_CONDUCTIVITY_HPP_
#define DYN_GRMHD_RESISTIVE_CONDUCTIVITY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file conductivity.hpp
//! \brief Device-callable isotropic conductivity closure sigma(rho) for resistive GRMHD.
//!
//! uniform : sigma = sigma0
//! density : log sigma = log sigma_out + s (log sigma_in - log sigma_out),
//!           s = 1/2 [1 + tanh((log10 rho - log10 rho_t)/width)]
//!           (smooth, monotone in rho; sigma_in for rho >> rho_t, sigma_out for
//!           rho << rho_t). Functional form [VERIFY] against Palenzuela et al. 2013
//!           (PRD 88, 043011); see notes proposal on the sigma(rho) form.
//!
//! The functor returns parallel and perpendicular conductivities (flag F1); for the
//! isotropic phase-1 closure they are equal.

#include <cmath>

#include "athena.hpp"
#include "dyn_grmhd/resistive/resistive_params.hpp"

namespace resistive {

struct SigmaPair {
  Real par;
  Real perp;
};

class Conductivity {
 public:
  Conductivity() = default;
  explicit Conductivity(const ResistiveParams &rp) {
    density_model_ = (rp.sigma_model == SigmaModel::density);
    sigma0_ = rp.sigma;
    if (density_model_) {
      log_in_ = std::log(rp.sigma_in);
      log_out_ = std::log(rp.sigma_out);
      log10_rhot_ = std::log10(rp.rho_transition);
      inv_width_ = 1.0/rp.sigma_width;
      sigma_out_ = rp.sigma_out;
    }
  }

  // rho: rest-mass density; extra state (p, B^2) reserved for later closures.
  KOKKOS_INLINE_FUNCTION
  SigmaPair operator()(const Real rho) const {
    if (!density_model_) return SigmaPair{sigma0_, sigma0_};
    if (!(rho > 0.0)) return SigmaPair{sigma_out_, sigma_out_};
    const Real s = 0.5*(1.0 + tanh((log10(rho) - log10_rhot_)*inv_width_));
    const Real sig = exp(log_out_ + s*(log_in_ - log_out_));
    return SigmaPair{sig, sig};
  }

  KOKKOS_INLINE_FUNCTION bool IsDensityModel() const { return density_model_; }

 private:
  bool density_model_ = false;
  Real sigma0_ = 0.0;
  Real log_in_ = 0.0, log_out_ = 0.0, log10_rhot_ = 0.0, inv_width_ = 1.0;
  Real sigma_out_ = 0.0;
};

}  // namespace resistive

#endif  // DYN_GRMHD_RESISTIVE_CONDUCTIVITY_HPP_
