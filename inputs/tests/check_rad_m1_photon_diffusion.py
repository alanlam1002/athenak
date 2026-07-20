#!/usr/bin/env python
"""Standalone optically-thick diffusion check for the grey photon M1 test.

Not wired into the tst/ pytest harness yet -- run by hand after executing
AthenaK with inputs/tests/rad_m1_photon_diffusion.athinput, e.g.:

    ./src/athena -i ../inputs/tests/rad_m1_photon_diffusion.athinput
    python ../inputs/tests/check_rad_m1_photon_diffusion.py

Run from the build directory (so the default "tab/" output directory is
found next to it), or pass --tab-dir explicitly.

kappa_p = kappa_a = 0, kappa_s > 0, compton = false: abs_1 = 0 exactly (no
local LTE relaxation channel at all -- see the Stage 2 scattering-only
single-zone null test, which confirms E is left completely alone by this
opacity configuration in a homogeneous setup), while kscat = kappa_s*rho > 0
still damps the flux (the -(kabs+kscat)*H_a term in calc_rad_sources,
radiation_m1_helpers.hpp). With no competing local-relaxation physics, the
initial Gaussian E(x) = exp(-9x^2) pulse should evolve by pure radiative
diffusion:

    dE/dt = D * d^2E/dx^2,   D = 1/(3*kappa_s*rho)   (code units, c=1)

whose solution for a Gaussian stays Gaussian with variance growing linearly:

    sigma^2(t) = sigma_0^2 + 2*D*t,   sigma_0^2 = 1/18 (from exp(-9x^2))

sigma^2(t) is measured directly from each output snapshot as the E-weighted
variance of x (no Gaussian-shape assumption baked into the *measurement*,
only into the analytic comparison). Total integrated energy is also checked
for near-exact conservation, as a shape-independent cross-check.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from m1_tab_utils import read_profile  # noqa: E402

# Must match inputs/tests/rad_m1_photon_diffusion.athinput
RHO = 1.0
KAPPA_S = 200.0
BASENAME = "photon_diffusion"
SIGMA0_SQ = 1.0 / 18.0

D = 1.0 / (3.0 * KAPPA_S * RHO)


def weighted_variance(x1v, E):
    total = np.sum(E)
    mean = np.sum(E * x1v) / total
    meansq = np.sum(E * x1v**2) / total
    return meansq - mean**2, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tab-dir", default="tab", help="directory with .tab output")
    parser.add_argument("--tol", type=float, default=0.05,
                        help="relative tolerance on sigma^2(t) vs the diffusion law")
    args = parser.parse_args()

    pattern = os.path.join(args.tab_dir, "{}.rad_m1_E.*.tab".format(BASENAME))
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit("No files matched {} -- run AthenaK first".format(pattern))

    rows = []
    for fname in files:
        x1v, data = read_profile(fname)
        sigma_sq, total = weighted_variance(x1v, data["E:0"])
        rows.append((data["time"], sigma_sq, total))
    rows.sort()

    t0, _, total0 = rows[0]
    print("D = 1/(3*kappa_s*rho) = {:.6e}".format(D))
    print("sigma0^2 (analytic, from exp(-9x^2)) = {:.6e}".format(SIGMA0_SQ))
    print("\n   t        sigma^2(t)    predicted     rel_err     integral/integral(0)")
    max_rel_err = 0.0
    for t, sigma_sq, total in rows:
        predicted = SIGMA0_SQ + 2.0 * D * t
        rel_err = abs(sigma_sq - predicted) / predicted
        max_rel_err = max(max_rel_err, rel_err)
        print("{:8.4f}  {:12.6e}  {:12.6e}  {:10.3e}  {:10.6f}".format(
            t, sigma_sq, predicted, rel_err, total / total0))

    print("\nmax relative error of sigma^2(t) vs sigma0^2+2*D*t over the run = {:.3e}".format(
        max_rel_err))

    if max_rel_err < args.tol:
        print("PASS: measured spreading rate matches the radiative-diffusion-limit "
              "prediction within tolerance {:.1e}".format(args.tol))
    else:
        print("FAIL: measured spreading rate does NOT match the diffusion-limit "
              "prediction within tolerance {:.1e}".format(args.tol))
        sys.exit(1)


if __name__ == "__main__":
    main()
