#!/usr/bin/env python
"""Standalone Compton-relaxation check for the grey photon M1 single zone test.

Not wired into the tst/ pytest harness yet -- run by hand after executing
AthenaK with inputs/tests/rad_m1_photon_compton_singlezone.athinput, e.g.:

    ./src/athena -i ../inputs/tests/rad_m1_photon_compton_singlezone.athinput
    python ../inputs/tests/check_rad_m1_photon_compton_singlezone.py

Run from the build directory (so the default "tab/" output directory is
found next to it), or pass --tab-dir explicitly.

This is the Compton-only analogue of check_rad_m1_photon_singlezone.py:
kappa_p = kappa_a = 0 (true absorption/emission off), so the only channel
driving E toward equilibrium is the Compton term added in
CalcOpacityPhotons_IdealGas_ (radiation_m1_calc_opacities_photons.cpp),
which folds the grey/Kompaneets-zeroth-order rate
    kappa_s * 4*k_B*T_gas/(m_e*c^2)
into eta_1/abs_1 as an extra effective Planck-like channel. Since
S_a = (eta - kabs*J)*u_a - ... is linear in J (and F=0 in this uniform,
static setup, so J=E), this reduces to exactly the same relaxation form as
the pure-absorption test:

    E(t) = J_eq + (E0 - J_eq) * exp(-abs_1 * t),   J_eq = a_rad * T_gas^4,
    abs_1 = kappa_s * rho * 4*k_B*T_gas/(m_e*c^2)

a_rad and k_B*T_gas/(m_e*c^2) are recomputed here from the <units> block in
the athinput (bhmass_msun, density_cgs, mu), replicating src/units/units.cpp,
since RadiationM1 derives a_rad from those rather than taking it directly
from the input file when a <units> block is present.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from m1_tab_utils import load_series  # noqa: E402

# Physical constants (cgs), must match src/units/units.hpp
K_BOLTZMANN_CGS = 1.3806488e-16
GRAV_CONSTANT_CGS = 6.67408e-8
SPEED_OF_LIGHT_CGS = 2.99792458e10
RAD_CONSTANT_CGS = 7.56573325e-15
ELECTRON_REST_MASS_ENERGY_CGS = 5.93e9  # Kelvin
ATOMIC_MASS_UNIT_CGS = 1.660538921e-24
MSUN_CGS = 1.98841586e+33

# Must match inputs/tests/rad_m1_photon_compton_singlezone.athinput
BHMASS_MSUN = 1.0
DENSITY_CGS = 1.0
MU = 1.0
RHO = 1.0
TEMP = 1.5e-3
KAPPA_S = 1.0
BASENAME = "photon_compton_singlezone"

# Replicate units::Units' GR (bhmass_msun, density_cgs) -> MLT scale derivation.
_length_cgs = GRAV_CONSTANT_CGS * (BHMASS_MSUN * MSUN_CGS) / SPEED_OF_LIGHT_CGS**2
_mass_cgs = DENSITY_CGS * _length_cgs**3
_time_cgs = _length_cgs / SPEED_OF_LIGHT_CGS
_velocity_cgs = _length_cgs / _time_cgs
_energy_cgs = _mass_cgs * _velocity_cgs**2
_pressure_cgs = _energy_cgs / _length_cgs**3
_temperature_cgs = _velocity_cgs**2 * MU * ATOMIC_MASS_UNIT_CGS / K_BOLTZMANN_CGS

ARAD = RAD_CONSTANT_CGS * _temperature_cgs**4 / _pressure_cgs
INV_T_ELECTRON = _temperature_cgs / ELECTRON_REST_MASS_ENERGY_CGS

J_EQ = ARAD * TEMP**4
ABS_1_EXPECTED = KAPPA_S * RHO * 4.0 * TEMP * INV_T_ELECTRON


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tab-dir", default="tab", help="directory with .tab output")
    parser.add_argument("--tol", type=float, default=1e-3,
                        help="relative tolerance on late-time E vs J_eq")
    args = parser.parse_args()

    t, E = load_series(args.tab_dir, BASENAME, "rad_m1_E", "E:0")
    _, abs_1 = load_series(args.tab_dir, BASENAME, "rad_m1_abs_1", "abs_1:0")

    print("a_rad (code units) = {:.6e}".format(ARAD))
    print("J_eq = a_rad * T^4 = {:.6e}".format(J_EQ))
    print("expected abs_1 = kappa_s * rho * 4*T/inv_t_electron = {:.6e}".format(
        ABS_1_EXPECTED))
    print("measured abs_1 (mean over run) = {:.6e}".format(np.mean(abs_1)))

    E0 = E[0]
    analytic = J_EQ + (E0 - J_EQ) * np.exp(-ABS_1_EXPECTED * t)
    rel_err = np.abs(E - analytic) / np.maximum(np.abs(analytic), 1e-300)

    print("\n   t          E(t)          analytic       rel_err")
    for ti, Ei, Ai, ri in zip(t, E, analytic, rel_err):
        print("{:10.4f}  {:12.6e}  {:12.6e}  {:10.3e}".format(ti, Ei, Ai, ri))

    final_rel_err = abs(E[-1] - J_EQ) / J_EQ
    print("\nfinal E = {:.6e}, target J_eq = {:.6e}, relative error = {:.3e}".format(
        E[-1], J_EQ, final_rel_err))

    if final_rel_err < args.tol:
        print("PASS: late-time E converged to a_rad*T^4 (via Compton) within "
              "tolerance {:.1e}".format(args.tol))
    else:
        print("FAIL: late-time E did NOT converge to a_rad*T^4 within tolerance "
              "{:.1e}".format(args.tol))
        sys.exit(1)


if __name__ == "__main__":
    main()
