#!/usr/bin/env python
"""Frozen-vs-implicit Compton opacity comparison at a deliberately STIFF
cfl_number (dtau*sigma_c = O(10)), for the grey photon M1 single-zone
Compton+backreaction test.

Not wired into the tst/ pytest harness yet -- run by hand after executing
AthenaK with all three of
inputs/tests/rad_m1_photon_compton_backreaction_stiff_frozen.athinput,
..._stiff_implicit.athinput, and ..._stiff_implicit_nolimiter.athinput, e.g.:

    ./src/athena -i ../inputs/tests/rad_m1_photon_compton_backreaction_stiff_frozen.athinput
    mv tab tab_frozen
    ./src/athena -i ../inputs/tests/rad_m1_photon_compton_backreaction_stiff_implicit.athinput
    mv tab tab_implicit
    ./src/athena -i ../inputs/tests/rad_m1_photon_compton_backreaction_stiff_implicit_nolimiter.athinput
    mv tab tab_nolimiter
    python ../inputs/tests/check_rad_m1_photon_compton_backreaction_stiff.py \
        --frozen-tab-dir tab_frozen --implicit-tab-dir tab_implicit \
        --nolimiter-tab-dir tab_nolimiter

All three athinputs share identical physics (rho, temp, kappa_s, <units>)
with rad_m1_photon_compton_backreaction_singlezone.athinput -- only
cfl_number (mild vs stiff), nlim/tlim, matter_implicit, and theta_limiter
differ -- so they relax toward the exact same independently-solved joint
equilibrium (a_rad*T_final^4 + rho*T_final/(gamma-1) = E_tot(0)); see that
test's check script and DEVELOPMENT.md's Stage 5/6 sections for the
derivation.

This script does NOT require the frozen-opacity run to reach equilibrium --
demonstrating that it does *not* (within this stiff run's few, large steps)
is the entire point of the matter_implicit feature. Pass criteria:
  1. Energy conservation (E_tot(t) == E_tot(0)) holds for ALL runs provided
     -- this is a hard invariant of the backreaction bookkeeping regardless
     of the per-step opacity linearization's accuracy (radiation_m1_update.cpp's
     DrEFN subtraction is exact by construction), so a violation on any
     side is a real bug, not just reduced accuracy.
  2. The implicit run's (and, if provided, the nolimiter run's) final
     (E, T) match the independently-solved equilibrium within
     --equilibrium-tol.
  3. The implicit run's final error is smaller than the frozen run's final
     error by at least --improvement-factor -- the quantitative statement
     of "the quartic pre-solve is more accurate than the frozen approach
     at this stiffness", not just "implicit happens to be close".
  4. If --nolimiter-tab-dir is given (Stage 6's proof point): that run's
     final error must ALSO clear --improvement-factor against the frozen
     run, on its own, with theta_limiter=false -- demonstrating the
     solver-level J_new correction (radiation_m1_update.cpp) suffices by
     itself, without the theta_limiter/source_limiter workaround Stage 5
     needed.
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

# Must match both rad_m1_photon_compton_backreaction_stiff_{frozen,implicit}.athinput
BHMASS_MSUN = 1.0
DENSITY_CGS = 1.0
MU = 1.0
RHO = 1.0
TEMP0 = 1.5e-3
GAMMA = 2.0

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


def load_run(tab_dir, basename):
    t, E = load_series(tab_dir, basename, "rad_m1_E", "E:0")
    _, dens = load_series(tab_dir, basename, "mhd_w", "dens")
    _, press = load_series(tab_dir, basename, "mhd_w", "press")
    T = press / dens
    Etot = E + e_int(dens, T)
    return t, E, T, Etot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-tab-dir", default="tab_frozen")
    parser.add_argument("--implicit-tab-dir", default="tab_implicit")
    parser.add_argument("--nolimiter-tab-dir", default=None,
                        help="optional: tab/ dir for "
                             "..._stiff_implicit_nolimiter.athinput "
                             "(matter_implicit=true, theta_limiter=false) -- "
                             "Stage 6's proof that the solver-level fix "
                             "suffices without the theta_limiter workaround")
    parser.add_argument("--conservation-tol", type=float, default=2e-5,
                        help="relative tolerance on E_tot(t) vs E_tot(0), each run")
    parser.add_argument("--equilibrium-tol", type=float, default=5e-2,
                        help="relative tolerance on the implicit run's final "
                             "E, T vs the independently-solved equilibrium")
    parser.add_argument("--improvement-factor", type=float, default=10.0,
                        help="required ratio of frozen final error to "
                             "implicit final error (T, the more sensitive "
                             "variable near this stiff equilibrium)")
    args = parser.parse_args(argv)

    t_f, E_f, T_f, Etot_f = load_run(args.frozen_tab_dir,
                                      "photon_compton_backreaction_stiff_frozen")
    t_i, E_i, T_i, Etot_i = load_run(args.implicit_tab_dir,
                                      "photon_compton_backreaction_stiff_implicit")
    nolimiter_run = None
    if args.nolimiter_tab_dir is not None:
        nolimiter_run = load_run(
            args.nolimiter_tab_dir,
            "photon_compton_backreaction_stiff_implicit_nolimiter")

    Etot0 = e_int(RHO, TEMP0) + 1e-30  # E(0) is at the radiation floor

    def residual(Tf):
        return e_int(RHO, Tf) + ARAD * Tf**4 - Etot0

    T_final_expected = brentq(residual, 1e-12, TEMP0)
    E_final_expected = ARAD * T_final_expected**4
    print("a_rad (code units) = {:.6e}".format(ARAD))
    print("independently-solved equilibrium: T_final={:.6e}, E_final={:.6e}".format(
        T_final_expected, E_final_expected))

    ok = True

    print("\n=== frozen (matter_implicit=false) ===")
    print("   t          E(t)          T(t)        cons_rel_err")
    max_cons_err_f = 0.0
    for ti, Ei, Ti, Etoti in zip(t_f, E_f, T_f, Etot_f):
        cons_rel_err = abs(Etoti - Etot0) / abs(Etot0)
        max_cons_err_f = max(max_cons_err_f, cons_rel_err)
        print("{:10.4f}  {:12.6e}  {:10.6e}  {:10.3e}".format(ti, Ei, Ti, cons_rel_err))
    final_T_rel_err_f = abs(T_f[-1] - T_final_expected) / T_final_expected
    print("max conservation error = {:.3e}, final T rel error = {:.3e}".format(
        max_cons_err_f, final_T_rel_err_f))
    if max_cons_err_f >= args.conservation_tol:
        print("FAIL: frozen run did not conserve energy (this IS a bug, "
              "independent of opacity accuracy)")
        ok = False

    print("\n=== implicit (matter_implicit=true, theta_limiter=true) ===")
    print("   t          E(t)          T(t)        cons_rel_err")
    max_cons_err_i = 0.0
    for ti, Ei, Ti, Etoti in zip(t_i, E_i, T_i, Etot_i):
        cons_rel_err = abs(Etoti - Etot0) / abs(Etot0)
        max_cons_err_i = max(max_cons_err_i, cons_rel_err)
        print("{:10.4f}  {:12.6e}  {:10.6e}  {:10.3e}".format(ti, Ei, Ti, cons_rel_err))
    final_E_rel_err_i = abs(E_i[-1] - E_final_expected) / E_final_expected
    final_T_rel_err_i = abs(T_i[-1] - T_final_expected) / T_final_expected
    print("max conservation error = {:.3e}".format(max_cons_err_i))
    print("final E = {:.6e}, target = {:.6e}, rel err = {:.3e}".format(
        E_i[-1], E_final_expected, final_E_rel_err_i))
    print("final T = {:.6e}, target = {:.6e}, rel err = {:.3e}".format(
        T_i[-1], T_final_expected, final_T_rel_err_i))

    if max_cons_err_i >= args.conservation_tol:
        print("FAIL: implicit run did not conserve energy")
        ok = False
    if final_E_rel_err_i >= args.equilibrium_tol or final_T_rel_err_i >= args.equilibrium_tol:
        print("FAIL: implicit run did not converge to the correct equilibrium")
        ok = False

    final_T_rel_err_n = None
    if nolimiter_run is not None:
        t_n, E_n, T_n, Etot_n = nolimiter_run
        print("\n=== nolimiter (matter_implicit=true, theta_limiter=false) ===")
        print("   t          E(t)          T(t)        cons_rel_err")
        max_cons_err_n = 0.0
        for ti, Ei, Ti, Etoti in zip(t_n, E_n, T_n, Etot_n):
            cons_rel_err = abs(Etoti - Etot0) / abs(Etot0)
            max_cons_err_n = max(max_cons_err_n, cons_rel_err)
            print("{:10.4f}  {:12.6e}  {:10.6e}  {:10.3e}".format(ti, Ei, Ti, cons_rel_err))
        final_E_rel_err_n = abs(E_n[-1] - E_final_expected) / E_final_expected
        final_T_rel_err_n = abs(T_n[-1] - T_final_expected) / T_final_expected
        print("max conservation error = {:.3e}".format(max_cons_err_n))
        print("final E = {:.6e}, target = {:.6e}, rel err = {:.3e}".format(
            E_n[-1], E_final_expected, final_E_rel_err_n))
        print("final T = {:.6e}, target = {:.6e}, rel err = {:.3e}".format(
            T_n[-1], T_final_expected, final_T_rel_err_n))

        if max_cons_err_n >= args.conservation_tol:
            print("FAIL: nolimiter run did not conserve energy")
            ok = False
        if (final_E_rel_err_n >= args.equilibrium_tol or
                final_T_rel_err_n >= args.equilibrium_tol):
            print("FAIL: nolimiter run did not converge to the correct "
                  "equilibrium -- the solver-level fix alone is not "
                  "sufficient without theta_limiter")
            ok = False

    print("\n=== comparison ===")
    improvement = final_T_rel_err_f / max(final_T_rel_err_i, 1e-300)
    print("frozen final T rel err            = {:.3e}".format(final_T_rel_err_f))
    print("implicit final T rel err          = {:.3e}".format(final_T_rel_err_i))
    print("improvement factor (frozen/implicit) = {:.3e}, required >= {:.1f}".format(
        improvement, args.improvement_factor))
    if improvement < args.improvement_factor:
        print("FAIL: matter_implicit=true was not meaningfully more accurate "
              "than matter_implicit=false at this stiffness")
        ok = False

    if final_T_rel_err_n is not None:
        improvement_n = final_T_rel_err_f / max(final_T_rel_err_n, 1e-300)
        print("nolimiter final T rel err         = {:.3e}".format(final_T_rel_err_n))
        print("improvement factor (frozen/nolimiter) = {:.3e}, required >= {:.1f}".format(
            improvement_n, args.improvement_factor))
        if improvement_n < args.improvement_factor:
            print("FAIL: matter_implicit=true, theta_limiter=false was not "
                  "meaningfully more accurate than the frozen path -- the "
                  "solver-level fix alone (Stage 6) is not sufficient "
                  "without theta_limiter")
            ok = False

    if ok:
        msg = ("\nPASS: all runs conserve energy; matter_implicit=true tracks "
               "the true equilibrium; matter_implicit=false is measurably "
               "worse at this stiffness, as expected")
        if final_T_rel_err_n is not None:
            msg += ("; the nolimiter run confirms the solver-level fix "
                    "(Stage 6) suffices without theta_limiter")
        print(msg)
    else:
        print("\nFAIL: see above")
        return False
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
