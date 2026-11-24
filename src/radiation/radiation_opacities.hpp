#ifndef RADIATION_RADIATION_OPACITIES_HPP_
#define RADIATION_RADIATION_OPACITIES_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_opacities.hpp
//! \brief implements functions for computing opacities

#include <math.h>

#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! bhstar WP2 opacity level A (<radiation>/opacity_model = saha_kramers; numerics-009
//! branch 3, DEC-20261119-numerics-opacity-levels). Constants (cgs) as in the reference
//! implementation docs/numerics/wp2/opacity_levelA.py of the bhstar mailbox repository.
namespace saha_kramers {
constexpr Real k_b    = 1.380649e-16;     // Boltzmann constant [erg/K]
constexpr Real m_e    = 9.1093837e-28;    // electron mass [g]
constexpr Real h_pl   = 6.62607015e-27;   // Planck constant [erg s]
constexpr Real m_h    = 1.6735575e-24;    // hydrogen atom mass [g]
constexpr Real ev     = 1.602176634e-12;  // electron volt [erg]
constexpr Real chi_h  = 13.598434;        // hydrogen ionisation energy [eV]
constexpr Real sig_t  = 6.6524587e-25;    // Thomson cross section [cm^2]
constexpr Real x_mass = 0.75;             // hydrogen mass fraction X
constexpr Real k0     = 7.6e22;           // Kramers coefficient [cm^5 g^-2 K^3.5]
constexpr Real planck_over_ross = 37.0;   // free-free kappa_P / kappa_R

//! \fn Real HydrogenIonFraction
//! \brief hydrogen-only Saha ionisation fraction x_H(rho, T), cgs input:
//!   x^2/(1-x) = S,  S = (2 pi m_e k T/h^2)^1.5 exp(-chi_H/kT) / n_H,  n_H = X rho/m_H,
//!   x = 2/(1 + sqrt(1 + 4/S))   (cancellation-free root of x^2 + S x - S = 0).
//! Returns 0 where S underflows (T below ~200 K), so that x^2 T^-3.5 is never 0 * inf.
//! A NaN temperature or density propagates (no silent repair).
KOKKOS_INLINE_FUNCTION
Real HydrogenIonFraction(const Real rho_cgs, const Real t_cgs) {
  if (t_cgs <= 0.0) return 0.0;
  const Real kt = k_b*t_cgs;
  Real s = pow(2.0*M_PI*m_e*kt/(h_pl*h_pl), 1.5)*exp(-chi_h*ev/kt)/(x_mass*rho_cgs/m_h);
  if (s < 1.0e-300) return 0.0;
  return 2.0/(1.0 + sqrt(1.0 + 4.0/s));
}
}  // namespace saha_kramers

//----------------------------------------------------------------------------------------
//! \fn void OpacityFunction
//! \brief sets sigma_a, sigma_s, sigma_p in the comoving frame

KOKKOS_INLINE_FUNCTION
void OpacityFunction(// density and density scale
                     const Real dens, const Real density_scale,
                     // temperature and temperature scale
                     const Real temp, const Real temperature_scale,
                     // length scale, adiabatic index minus one, mean molecular weight
                     const Real length_scale, const Real gm1, const Real mu,
                     // power law opacities
                     const bool pow_opacity,
                     // bhstar opacity level A (Saha-Thomson + x_H^2 Kramers), cgs units
                     const bool saha_kramers_opacity,
                     const Real rosseland_coef, const Real planck_minus_rosseland_coef,
                     // spatially and temporally constant opacities
                     const Real k_a, const Real k_s, const Real k_p,
                     // output sigma
                     Real& sigma_a, Real& sigma_s, Real& sigma_p) {
  if (saha_kramers_opacity) {
    // level A: kappa_s = (sigma_T/m_H) X x_H; kappa_aR = K0 rho T^-3.5 x_H^2;
    // kappa_P = 37 kappa_aR. AthenaK convention: sigma_a = rho kappa_aR (Rosseland
    // absorption), sigma_p = rho (kappa_P - kappa_aR) (Planck minus Rosseland).
    const Real rho_cgs = dens*density_scale;
    const Real t_cgs = temp*temperature_scale;
    const Real x_h = saha_kramers::HydrogenIonFraction(rho_cgs, t_cgs);
    const Real k_s_cgs = saha_kramers::sig_t/saha_kramers::m_h*saha_kramers::x_mass*x_h;
    Real k_ar_cgs = 0.0;
    if (x_h > 0.0) k_ar_cgs = saha_kramers::k0*rho_cgs*pow(t_cgs, -3.5)*x_h*x_h;
    sigma_a = dens*k_ar_cgs*density_scale*length_scale;
    sigma_p = dens*(saha_kramers::planck_over_ross - 1.0)*k_ar_cgs*density_scale*length_scale;
    sigma_s = dens*k_s_cgs*density_scale*length_scale;
  } else if (pow_opacity) {  // power law opacity (accounting for diff b/w Ross & Planck)
    Real power_law = (dens*density_scale)*pow(gm1*mu/(temp*temperature_scale), 3.5);
    Real k_a_r = rosseland_coef * power_law;
    Real k_a_p = planck_minus_rosseland_coef * power_law;
    sigma_a = dens*k_a_r*density_scale*length_scale;
    sigma_p = dens*k_a_p*density_scale*length_scale;
    sigma_s = dens*k_s  *density_scale*length_scale;
  } else {  // spatially and temporally constant opacity
    sigma_a = dens*k_a*density_scale*length_scale;
    sigma_p = dens*k_p*density_scale*length_scale;
    sigma_s = dens*k_s*density_scale*length_scale;
  }
  return;
}

#endif // RADIATION_RADIATION_OPACITIES_HPP_
