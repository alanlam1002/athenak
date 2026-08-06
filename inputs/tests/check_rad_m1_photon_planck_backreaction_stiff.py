#!/usr/bin/env python
"""Planck-only (compton=false) radiation-pressure backreaction check for the
grey photon M1 single zone test, at a deliberately STIFF cfl_number
(dtau*sigma_p = O(10)), with matter_implicit=true and theta_limiter=false.

This is Stage 6's generalization check: the matter_implicit quartic
pre-solve (radiation_m1_compton_implicit.hpp::SolveComptonQuartic) was
extended to solve with the COMBINED Planck+Compton rate, not just Compton,
so it should handle a pure-Planck (kappa_s=0, kappa_p>0) backreaction test
just as well as the Compton stiff tests
(check_rad_m1_photon_compton_backreaction_stiff.py) -- without needing
theta_limiter, since the quartic's own (T_new, J_new) pair is
energy-conserving by construction regardless of which channel(s) produced
the rate.

Not wired into the tst/ pytest harness yet -- run by hand after executing
AthenaK with inputs/tests/rad_m1_photon_planck_backreaction_stiff.athinput,
e.g.:

    ./src/athena -i ../inputs/tests/rad_m1_photon_planck_backreaction_stiff.athinput
    python ../inputs/tests/check_rad_m1_photon_planck_backreaction_stiff.py

Run from the build directory (so the default "tab/" output directory is
found next to it), or pass --tab-dir explicitly.

Same two-check structure as check_rad_m1_photon_compton_backreaction_singlezone.py:
(1) total energy E_tot = E + e_int is conserved, (2) (E, T_gas) converge to
the independently-solved joint equilibrium
a_rad*T_final^4 + rho*T_final/(gamma-1) = E_tot(0). Same physical
calibration (rho, T0, gamma, arad via <units>) as the Compton stiff tests,
so it targets the exact same equilibrium.
"""
import argparse
import os
import sys

from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(__file__))
from m1_tab_utils import load_series  # noqa: E402

# Physical constants (cgs), must match src/units/units.hpp
K_BOLTZMANN_CGS = 1.3806488e-16
GRAV_CONSTANT_CGS = 6.67408e-8
SPEED_OF_LIGHT_CGS = 2.99792458e10
RAD_CONSTANT_CGS = 7.56573325e-15
ATOMIC_MASS_UNIT_CGS = 1.660538921e-24
MSUN_CGS = 1.98841586e+33

# Must match inputs/tests/rad_m1_photon_planck_backreaction_stiff.athinput
BHMASS_MSUN = 1.0
DENSITY_CGS = 1.0
MU = 1.0
RHO = 1.0
TEMP0 = 1.5e-3
GAMMA = 2.0
BASENAME = "photon_planck_backreaction_stiff"

# Replicate units::Units' GR (bhmass_msun, density_cgs) -> MLT scale derivation.
_length_cgs = GRAV_CONSTANT_CGS * (BHMASS_MSUN * MSUN_CGS) / SPEED_OF_LIGHT_CGS**2
_mass_cgs = DENSITY_CGS * _length_cgs**3
_time_cgs = _length_cgs / SPEED_OF_LIGHT_CGS
_velocity_cgs = _length_cgs / _time_cgs
_energy_cgs = _mass_cgs * _velocity_cgs**2
_pressure_cgs = _energy_cgs / _length_cgs**3
_temperature_cgs = _velocity_cgs**2 * MU * ATOMIC_MASS_UNIT_CGS / K_BOLTZMANN_CGS

ARAD = RAD_CONSTANT_CGS * _temperature_cgs**4 / _pressure_cgs


def e_int(rho, T):
    return rho * T / (GAMMA - 1.0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tab-dir", default="tab", help="directory with .tab output")
    parser.add_argument("--conservation-tol", type=float, default=2e-5,
                        help="relative tolerance on E_tot(t) vs E_tot(0)")
    parser.add_argument("--equilibrium-tol", type=float, default=5e-2,
                        help="relative tolerance on late-time E, T vs the "
                             "independently-solved equilibrium")
    args = parser.parse_args(argv)

    t, E = load_series(args.tab_dir, BASENAME, "rad_m1_E", "E:0")
    _, dens = load_series(args.tab_dir, BASENAME, "mhd_w", "dens")
    _, press = load_series(args.tab_dir, BASENAME, "mhd_w", "press")

    T = press / dens
    eint = e_int(dens, T)
    Etot = E + eint
    Etot0 = Etot[0]

    print("a_rad (code units) = {:.6e}".format(ARAD))
    print("initial: E={:.6e}, T={:.6e}, e_int={:.6e}, E_tot(0)={:.6e}".format(
        E[0], T[0], eint[0], Etot0))

    def residual(Tf):
        return e_int(RHO, Tf) + ARAD * Tf**4 - Etot0

    T_final_expected = brentq(residual, 1e-12, TEMP0)
    E_final_expected = ARAD * T_final_expected**4
    print("independently-solved equilibrium: T_final={:.6e}, E_final={:.6e}".format(
        T_final_expected, E_final_expected))

    print("\n   t          E(t)          T(t)        E_tot(t)     cons_rel_err")
    max_cons_err = 0.0
    for ti, Ei, Ti, Etoti in zip(t, E, T, Etot):
        cons_rel_err = abs(Etoti - Etot0) / abs(Etot0)
        max_cons_err = max(max_cons_err, cons_rel_err)
        print("{:10.4f}  {:12.6e}  {:10.6e}  {:12.6e}  {:10.3e}".format(
            ti, Ei, Ti, Etoti, cons_rel_err))

    final_E_rel_err = abs(E[-1] - E_final_expected) / E_final_expected
    final_T_rel_err = abs(T[-1] - T_final_expected) / T_final_expected

    print("\nmax relative error of E_tot(t) vs E_tot(0) (conservation) = {:.3e}".format(
        max_cons_err))
    print("final E = {:.6e}, target E_final = {:.6e}, relative error = {:.3e}".format(
        E[-1], E_final_expected, final_E_rel_err))
    print("final T = {:.6e}, target T_final = {:.6e}, relative error = {:.3e}".format(
        T[-1], T_final_expected, final_T_rel_err))

    ok = (max_cons_err < args.conservation_tol and
          final_E_rel_err < args.equilibrium_tol and
          final_T_rel_err < args.equilibrium_tol)
    if ok:
        print("\nPASS: energy conserved and (E, T_gas) converged to the correct "
              "Planck-only backreaction joint equilibrium, at stiff cfl and "
              "theta_limiter=false -- matter_implicit's generalization to "
              "the Planck channel works as designed")
    else:
        print("\nFAIL: either energy was not conserved or (E, T_gas) did not "
              "converge to the correct joint equilibrium")
        return False
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
