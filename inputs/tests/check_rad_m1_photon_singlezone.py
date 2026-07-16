#!/usr/bin/env python
"""Standalone LTE-relaxation check for the grey photon M1 single zone test.

Not wired into the tst/ pytest harness yet -- run by hand after executing
AthenaK with inputs/tests/rad_m1_photon_singlezone.athinput, e.g.:

    ./src/athena -i ../inputs/tests/rad_m1_photon_singlezone.athinput
    python ../inputs/tests/check_rad_m1_photon_singlezone.py

Run from the build directory (so the default "tab/" output directory is
found next to it), or pass --tab-dir explicitly.

Compares the M1-evolved radiation energy density E(t) against the analytic
solution of dE/dt = eta_1 - abs_1*E (flat space, no spatial gradients since
the box is uniform/periodic):

    E(t) = J_eq + (E0 - J_eq) * exp(-abs_1 * t),   J_eq = a_rad * T^4

using the same <photons>/<problem> parameters as the athinput. Update the
constants below if you change the athinput.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "vis", "python"))
import athena_read  # noqa: E402

# Must match inputs/tests/rad_m1_photon_singlezone.athinput
RHO = 1.0
TEMP = 1.0
KAPPA_P = 10.0
KAPPA_A = 0.0
ARAD = 1.0
BASENAME = "photon_singlezone"

J_EQ = ARAD * TEMP**4
ABS_1_EXPECTED = KAPPA_P * RHO


def load_series(tab_dir, file_id, column):
    pattern = os.path.join(tab_dir, "{}.{}.*.tab".format(BASENAME, file_id))
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit("No files matched {} -- run AthenaK first".format(pattern))
    times, values = [], []
    for fname in files:
        data = athena_read.tab(fname)
        times.append(data["time"])
        # spatially uniform box: average over the (periodic, identical) cells
        values.append(np.mean(data[column]))
    order = np.argsort(times)
    return np.array(times)[order], np.array(values)[order]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tab-dir", default="tab", help="directory with .tab output")
    parser.add_argument("--tol", type=float, default=1e-3,
                        help="relative tolerance on late-time E vs J_eq")
    args = parser.parse_args()

    t, E = load_series(args.tab_dir, "rad_m1_E", "E:0")
    _, abs_1 = load_series(args.tab_dir, "rad_m1_abs_1", "abs_1:0")

    print("J_eq = a_rad * T^4 = {:.6e}".format(J_EQ))
    print("expected abs_1 = kappa_p * rho = {:.6e}".format(ABS_1_EXPECTED))
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
        print("PASS: late-time E converged to a_rad*T^4 within tolerance {:.1e}".format(
            args.tol))
    else:
        print("FAIL: late-time E did NOT converge to a_rad*T^4 within tolerance "
              "{:.1e}".format(args.tol))
        sys.exit(1)


if __name__ == "__main__":
    main()
