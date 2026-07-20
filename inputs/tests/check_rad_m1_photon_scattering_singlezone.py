#!/usr/bin/env python
"""Standalone scattering-only null-test check for the grey photon M1 single
zone test.

Not wired into the tst/ pytest harness yet -- run by hand after executing
AthenaK with inputs/tests/rad_m1_photon_scattering_singlezone.athinput, e.g.:

    ./src/athena -i ../inputs/tests/rad_m1_photon_scattering_singlezone.athinput
    python ../inputs/tests/check_rad_m1_photon_scattering_singlezone.py

Run from the build directory (so the default "tab/" output directory is
found next to it), or pass --tab-dir explicitly.

kappa_p = kappa_a = 0 and compton = false, so there is no channel exchanging
energy between matter and radiation -- only elastic scattering (kappa_s > 0),
which conserves photon energy in this formalism (it damps flux via the
kscat*H_a term in calc_rad_sources, radiation_m1_helpers.hpp, but does not
appear in the (eta - kabs*J) emission/absorption term at all). In this
static, homogeneous, zero-flux setup that term is identically zero too, so
E(t) should stay pinned at its initial (floor) value for the whole run --
NOT drift toward a_rad*T^4 the way the companion LTE/Compton tests do. This
guards against kscat ever leaking into the emission-driving term.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from m1_tab_utils import load_series  # noqa: E402

BASENAME = "photon_scattering_singlezone"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tab-dir", default="tab", help="directory with .tab output")
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="relative tolerance on E(t) staying at E(0)")
    args = parser.parse_args()

    t, E = load_series(args.tab_dir, BASENAME, "rad_m1_E", "E:0")

    E0 = E[0]
    rel_dev = np.abs(E - E0) / E0

    print("E(0) = {:.6e} (should equal rad_E_floor)".format(E0))
    print("\n   t          E(t)         rel_dev_from_E0")
    for ti, Ei, ri in zip(t, E, rel_dev):
        print("{:10.4f}  {:12.6e}  {:10.3e}".format(ti, Ei, ri))

    max_rel_dev = np.max(rel_dev)
    print("\nmax relative deviation of E(t) from E(0) over the whole run = {:.3e}".format(
        max_rel_dev))

    if max_rel_dev < args.tol:
        print("PASS: E stayed pinned at its floor value within tolerance {:.1e} "
              "-- scattering alone does not drive LTE equilibration".format(args.tol))
    else:
        print("FAIL: E drifted away from its floor value by more than tolerance "
              "{:.1e} -- kscat may be leaking into the emission/absorption "
              "term".format(args.tol))
        sys.exit(1)


if __name__ == "__main__":
    main()
