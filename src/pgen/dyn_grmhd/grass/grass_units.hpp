#ifndef PGEN_DYN_GRMHD_RNS_GRASS_UNITS_HPP_
#define PGEN_DYN_GRMHD_RNS_GRASS_UNITS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file grass_units.hpp
//  \brief Unit conversion for GRASS (RNS-family rotating-NS equilibrium code) initial
//  data: GRASS-internal (KAPPA/KSCALE-scaled RNS units) -> cgs -> AthenaK's own
//  geometrized G=c=Msun=1 units, in two hops.
//
//  Hop 1 (GRASS-internal -> cgs), per GRASS's exporter_mod.f90/para_panel.f90:
//    C = 2.99792458e10 cm/s, G = 6.67408e-8 cgs, MB = 1.6749286e-24 g
//    KAPPA = 1e-15*C^2/G, KSCALE = KAPPA*G/C^4 (= 1e-15/C^2)
//    r_cgs = r_internal*sqrt(KAPPA), Omega_cgs = Omega_internal*C/sqrt(KAPPA)
//    e_cgs = e_internal/(C^2*KSCALE), p_cgs = p_internal/KSCALE
//  GRASS's own n0_at_e() (eos_mod.f90) takes its argument already in GRASS-internal
//  units, so grass_eos_table.hpp feeds the restart's raw `energy` field to N0FromE()
//  with no pre-conversion.
//
//  Hop 2 (cgs -> AthenaK code units) via Primitive::UnitSystem
//  (eos/primitive-solver/unit_system.hpp). Angular velocity is a rate (1/time): divide
//  by TimeConversion, don't multiply.

#include <cmath>

#include "athena.hpp"
#include "eos/primitive-solver/unit_system.hpp"

namespace grass {

//----------------------------------------------------------------------------------------
//! \struct GrassUnits
//  \brief Precomputed hop-1 (GRASS-internal->cgs) and hop-2 (cgs->AthenaK code units)
//  conversion factors. Constructed once and passed around by value (cheap, all Reals).

struct GrassUnits {
  // ---- Hop 1: GRASS-internal -> cgs -----------------------------------------------
  static constexpr Real kGrassC  = 2.99792458e10;   // cm/s
  static constexpr Real kGrassG  = 6.67408e-8;      // cm^3 g^-1 s^-2
  static constexpr Real kGrassMB = 1.6749286e-24;   // g (mean baryon mass, GRASS's own)
  static constexpr Real kKappa   = 1.0e-15 * kGrassC * kGrassC / kGrassG;
  static constexpr Real kKscale  = 1.0e-15 / (kGrassC * kGrassC);  // = KAPPA*G/C^4

  // ---- Hop 2: cgs -> AthenaK code (geometrized G=c=Msun=1) ------------------------
  Primitive::UnitSystem cgs;
  Primitive::UnitSystem code;

  GrassUnits() : cgs(Primitive::MakeCGS()), code(Primitive::MakeGeometricSolar()) {}

  // -- Hop 1 helpers (GRASS-internal -> cgs) ----------------------------------------
  Real LengthGrassToCgs(Real r_internal) const {
    return r_internal * Kokkos::sqrt(kKappa);
  }
  Real RateGrassToCgs(Real omega_internal) const {
    return omega_internal * kGrassC / Kokkos::sqrt(kKappa);
  }
  Real EnergyDensityGrassToCgs(Real e_internal) const {
    return e_internal / (kGrassC * kGrassC * kKscale);
  }
  Real PressureGrassToCgs(Real p_internal) const {
    return p_internal / kKscale;
  }

  // -- Hop 2 helpers (cgs -> AthenaK code units) ------------------------------------
  Real LengthCgsToCode(Real r_cgs) const {
    return r_cgs * cgs.LengthConversion(code);
  }
  Real MassDensityCgsToCode(Real rho_cgs) const {
    return rho_cgs * cgs.MassDensityConversion(code);
  }
  Real EnergyDensityCgsToCode(Real e_cgs) const {
    return e_cgs * cgs.EnergyDensityConversion(code);
  }
  Real PressureCgsToCode(Real p_cgs) const {
    // Pressure has the same dimensions as energy density (erg/cm^3); UnitSystem has
    // no separate pressure-conversion method.
    return p_cgs * cgs.EnergyDensityConversion(code);
  }
  Real RateCgsToCode(Real omega_cgs) const {
    return omega_cgs / cgs.TimeConversion(code);  // rate = 1/time, see file header
  }

  // -- Convenience: full GRASS-internal -> AthenaK code, one hop each ---------------
  Real LengthToCode(Real r_internal) const {
    return LengthCgsToCode(LengthGrassToCgs(r_internal));
  }
  Real RateToCode(Real omega_internal) const {
    return RateCgsToCode(RateGrassToCgs(omega_internal));
  }
  Real EnergyDensityToCode(Real e_internal) const {
    return EnergyDensityCgsToCode(EnergyDensityGrassToCgs(e_internal));
  }
  Real PressureToCode(Real p_internal) const {
    return PressureCgsToCode(PressureGrassToCgs(p_internal));
  }
  // Rest-mass density arrives already in cgs (g/cm^3) from GrassEosTable::N0FromE()
  // (which multiplies by GRASS's own baryon mass kGrassMB) -- only hop 2 needed.
  Real RestMassDensityCgsToCode(Real rho0_cgs) const {
    return MassDensityCgsToCode(rho0_cgs);
  }
};

}  // namespace grass

#endif  // PGEN_DYN_GRMHD_RNS_GRASS_UNITS_HPP_
