#!/usr/bin/env python
"""Radiation-pressure backreaction check for the grey photon M1 single zone
test.

Not wired into the tst/ pytest harness yet -- run by hand after executing
AthenaK with inputs/tests/rad_m1_photon_backreaction_singlezone.athinput,
e.g.:

    ./src/athena -i ../inputs/tests/rad_m1_photon_backreaction_singlezone.athinput
    python ../inputs/tests/check_rad_m1_photon_backreaction_singlezone.py

Run from the build directory (so the default "tab/" output directory is
found next to it), or pass --tab-dir explicitly.

Unlike rad_m1_photon_singlezone.athinput (backreact=false, E chases a FIXED
external T), this test has backreact=true: radiation_m1_update.cpp:672-680
subtracts the same matter-exchange increment (DrEFN) used to update the M1
fields from the MHD conserved energy-momentum, so E and T_gas jointly relax
to whatever equilibrium is set by total energy conservation.

Primitive::IdealGas (src/eos/primitive-solver/idealgas.hpp, code units,
mb=1): Pressure(n,T) = n*T, so T = P/n = P/rho; internal energy density
e_int = n*T/(gamma-1) = rho*T/(gamma-1) (Energy(n,T) = n*(mb+T/(gamma-1))
minus the rest-mass piece n*mb).

Two checks:
  1. Energy conservation (cheap, shape-independent): E_tot(t) = E(t) +
     e_int(t) should stay at its initial value throughout the run.
  2. Correct joint equilibrium (physics correctness, not just bookkeeping):
     independently solve rho*T_final/(gamma-1) + arad*T_final^4 = E_tot(0)
     for T_final (scipy.optimize.brentq), giving E_final = arad*T_final^4,
     and compare against the simulation's own late-time E/T.
"""
import argparse
import os
import sys

from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(__file__))
from m1_tab_utils import load_series  # noqa: E402

# Must match inputs/tests/rad_m1_photon_backreaction_singlezone.athinput
RHO = 1.0
TEMP0 = 1.0
GAMMA = 2.0
ARAD = 1.0
BASENAME = "photon_backreaction_singlezone"


def e_int(rho, T):
    return rho * T / (GAMMA - 1.0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tab-dir", default="tab", help="directory with .tab output")
    parser.add_argument("--conservation-tol", type=float, default=1e-5,
                        help="relative tolerance on E_tot(t) vs E_tot(0)")
    parser.add_argument("--equilibrium-tol", type=float, default=1e-3,
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

    print("initial: E={:.6e}, T={:.6e}, e_int={:.6e}, E_tot(0)={:.6e}".format(
        E[0], T[0], eint[0], Etot0))

    # Independently solve for the joint equilibrium: rho*T/(gamma-1) +
    # arad*T^4 = E_tot(0). The system can only cool (T_final <= T0) since it
    # starts with E << e_int (radiation starts at the floor); bracket [0,T0].
    def residual(Tf):
        return e_int(RHO, Tf) + ARAD * Tf**4 - Etot0

    T_final_expected = brentq(residual, 1e-10, TEMP0)
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
              "joint equilibrium")
    else:
        print("\nFAIL: either energy was not conserved or (E, T_gas) did not "
              "converge to the correct joint equilibrium")
        return False
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
