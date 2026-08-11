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
//  geometrized G=c=Msun=1 units. Two explicit hops, both confirmed against GRASS's own
//  source this session (not re-derived from scratch):
//
//  Hop 1 (GRASS-internal -> cgs), confirmed against
//  /u/tlam/GRASS/src/tool/exporter_mod.f90 and /u/tlam/GRASS/src/para_panel.f90:
//    C = 2.99792458e10 cm/s, G = 6.67408e-8 cgs, MB = 1.6749286e-24 g
//    KAPPA = 1e-15 * C^2 / G
//    KSCALE = KAPPA * G / C^4  (simplifies to 1e-15/C^2, independent of G)
//    r_cgs[cm]      = r_internal * sqrt(KAPPA)
//    Omega_cgs[1/s] = Omega_internal * C/sqrt(KAPPA)   (applies to both omg and ww)
//    e_cgs[g/cm^3]  = e_internal / (C^2 * KSCALE)
//    p_cgs[erg/cm^3] = p_internal / KSCALE
//  n0_at_e() in GRASS's own eos_mod.f90 takes its argument already in GRASS-internal
//  units (its own table loader bakes the C^2*KSCALE/KSCALE factors in when the table
//  file is read) -- grass_eos_table.hpp mirrors that convention exactly, so the restart
//  file's raw `energy` field is fed to N0FromE() with NO pre-conversion.
//
//  Hop 2 (cgs -> AthenaK code units), via Primitive::UnitSystem (eos/primitive-solver/
//  unit_system.hpp): mass density / energy density / length use
//  MassDensityConversion/EnergyDensityConversion/LengthConversion directly. Angular
//  velocity is a RATE (1/time), which transforms with the RECIPROCAL of
//  TimeConversion (TimeConversion converts a duration; verified by dimensional
//  analysis against the textbook GM_sun/c^3 ~ 4.925 microsecond timescale -- do not
//  multiply by TimeConversion for a rate, that is backwards).

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
    return r_internal * std::sqrt(kKappa);
  }
  Real RateGrassToCgs(Real omega_internal) const {
    return omega_internal * kGrassC / std::sqrt(kKappa);
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
    // Pressure has the same dimensions as energy density (erg/cm^3), so the same
    // conversion method applies -- UnitSystem has no separate "pressure density"
    // method distinct from EnergyDensityConversion for this purpose.
    return p_cgs * cgs.EnergyDensityConversion(code);
  }
  Real RateCgsToCode(Real omega_cgs) const {
    // Rate = 1/time transforms with the RECIPROCAL of TimeConversion (which converts
    // a duration, not a rate) -- see file header note.
    return omega_cgs / cgs.TimeConversion(code);
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
