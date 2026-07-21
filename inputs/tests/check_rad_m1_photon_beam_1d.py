#!/usr/bin/env python
"""Free-streaming beam check for the grey photon M1 test.

Not wired into the tst/ pytest harness yet -- run by hand after executing
AthenaK with inputs/tests/rad_m1_photon_beam_1d.athinput, e.g.:

    ./src/athena -i ../inputs/tests/rad_m1_photon_beam_1d.athinput
    python ../inputs/tests/check_rad_m1_photon_beam_1d.py

Run from the build directory (so the default "tab/" output directory is
found next to it), or pass --tab-dir explicitly.

kappa_a = kappa_p = 0, kappa_s > 0 (small), compton = false: there is no local
LTE emission/absorption channel at all (see the Stage 1 scattering-only
single-zone null test, which confirms scattering alone conserves E exactly in
a homogeneous setting), only the flux-damping -(kabs+kscat)*H_a term
(calc_rad_sources, radiation_m1_helpers.hpp). A beam of E=1, F_x=E (causality-
saturated) is continuously injected at x=0 (radiation_m1_beams.cpp) and
should free-stream at exactly c=1, arriving at x=c*t, with:

  - E in the illuminated region (x < c*t, away from the front's numerical
    smearing) staying close to its injected value of 1 -- this is the classic
    M1 failure mode this test exists to catch: an over-damped or spuriously
    decaying/growing implicit source solve at near-zero opacity.
  - F_x/E staying close to its injected value (causality-saturated, close to
    1) rather than decaying toward 0 -- i.e. the beam should stay "beam-like"
    (chi->1), not get artificially isotropized.
  - The front position (measured as the half-max crossing of E) tracking
    x_front ~ c*t = t, to within the front's inherent numerical smearing width
    (the donor-cell reconstruction used here is first-order and diffusive by
    design -- this test is not checking front sharpness, only that the front
    isn't systematically lagging/leading beyond that smearing, which would
    indicate sub- or super-luminal propagation).
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from m1_tab_utils import read_profile  # noqa: E402

BASENAME = "photon_beam_1d"
INJECTED_E = 1.0

# How far behind the front to sample the "plateau" (in x), to stay clear of
# the front's numerical smearing (comparable to a handful of cells at
# nx1=400 over x1max=8, i.e. dx=0.02 -- the front smears over roughly 30-40
# cells in practice, so stay at least ~1 unit behind it).
PLATEAU_MARGIN = 1.5


def load_series(tab_dir, file_id, column):
    pattern = os.path.join(tab_dir, "{}.{}.*.tab".format(BASENAME, file_id))
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit("No files matched {} -- run AthenaK first".format(pattern))
    rows = []
    for fname in files:
        x1v, data = read_profile(fname)
        rows.append((data["time"], x1v, data[column]))
    rows.sort(key=lambda r: r[0])
    return rows


def front_position(x1v, E, threshold):
    """First (smallest-x) location where E crosses below threshold, scanning
    from large x inward -- i.e. the half-max point of the illuminated front."""
    above = E >= threshold
    if not above.any():
        return None
    last_above = np.max(np.where(above)[0])
    if last_above == len(x1v) - 1:
        return x1v[last_above]
    # linear interpolation between the last cell above and the first below
    x0, x1 = x1v[last_above], x1v[last_above + 1]
    e0, e1 = E[last_above], E[last_above + 1]
    return x0 + (threshold - e0) * (x1 - x0) / (e1 - e0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tab-dir", default="tab", help="directory with .tab output")
    parser.add_argument("--front-tol", type=float, default=0.5,
                        help="absolute tolerance on front position x_front - c*t")
    parser.add_argument("--plateau-tol", type=float, default=0.02,
                        help="relative tolerance on plateau E vs the injected value")
    parser.add_argument("--fe-ratio-tol", type=float, default=0.02,
                        help="absolute tolerance on F_x/E drift from its injected value")
    args = parser.parse_args()

    e_rows = load_series(args.tab_dir, "rad_m1_E", "E:0")
    f_rows = load_series(args.tab_dir, "rad_m1_F", "Fx:0")

    print("\n   t     x_front    c*t     front_err   plateau_E   rel_err    Fx/E    fe_err")
    max_front_err = 0.0
    max_plateau_rel_err = 0.0
    max_fe_err = 0.0
    injected_fe_ratio = None
    for (t, x1v, E), (tf, _, Fx) in zip(e_rows, f_rows):
        assert abs(t - tf) < 1e-9, (t, tf)
        if t < 2 * PLATEAU_MARGIN:
            # not enough illuminated plateau yet to measure meaningfully
            continue
        xfront = front_position(x1v, E, 0.5 * INJECTED_E)
        front_err = xfront - t
        max_front_err = max(max_front_err, abs(front_err))

        plateau_mask = x1v < (t - PLATEAU_MARGIN)
        plateau_E = np.mean(E[plateau_mask])
        plateau_rel_err = abs(plateau_E - INJECTED_E) / INJECTED_E
        max_plateau_rel_err = max(max_plateau_rel_err, plateau_rel_err)

        plateau_Fx = np.mean(Fx[plateau_mask])
        fe_ratio = plateau_Fx / plateau_E
        if injected_fe_ratio is None:
            # reference F_x/E, from the earliest usable plateau measurement
            # (beam has had the least time to be affected by scattering)
            injected_fe_ratio = fe_ratio
        fe_err = abs(fe_ratio - injected_fe_ratio)
        max_fe_err = max(max_fe_err, fe_err)

        print("{:6.2f}  {:8.4f}  {:6.2f}  {:10.3e}  {:10.6f}  {:9.3e}  {:7.4f}  {:8.3e}".format(
            t, xfront, t, front_err, plateau_E, plateau_rel_err, fe_ratio, fe_err))

    print("\nreference (earliest-measured) plateau F_x/E = {:.6f}".format(injected_fe_ratio))
    print("max |x_front - c*t|                = {:.3e} (tol {:.1e})".format(
        max_front_err, args.front_tol))
    print("max relative error of plateau E     = {:.3e} (tol {:.1e})".format(
        max_plateau_rel_err, args.plateau_tol))
    print("max drift of plateau F_x/E          = {:.3e} (tol {:.1e})".format(
        max_fe_err, args.fe_ratio_tol))

    ok = (max_front_err < args.front_tol and
          max_plateau_rel_err < args.plateau_tol and
          max_fe_err < args.fe_ratio_tol)
    if ok:
        print("\nPASS: beam propagates at c with no spurious damping/growth or "
              "isotropization")
    else:
        print("\nFAIL: beam does not free-stream cleanly within tolerance")
        sys.exit(1)


if __name__ == "__main__":
    main()
