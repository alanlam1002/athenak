#!/usr/bin/env python
"""Stage 3 sanity-check for dyngr_bhstar.cpp's radiation-on Bondi run.

Not wired into the tst/ pytest harness (this pgen is a USER_PROBLEM, compiled
via '-D PROBLEM=dyngr_bhstar' into its own separate build -- not part of the
standard built_in_pgens build the CI harness uses, see DEVELOPMENT.md Stage
3). Run by hand after executing AthenaK with
inputs/tests/dyngr_bhstar_radiation.athinput, e.g.:

    ./athena -i .../inputs/tests/dyngr_bhstar_radiation.athinput
    python .../inputs/tests/check_dyngr_bhstar_radiation.py

Run from the directory containing the "tab/" output, or pass --tab-dir.

dyngr_bhstar.cpp's radiation IC is an ad hoc thick/thin optical-depth blend,
not a self-consistent closed-form solution -- there is nothing exact to
check the DYNAMICAL evolution against. This script therefore has two tiers:

  1. An EXACT check of the t=0 initial data against the C++ IC formula,
     transcribed here line-for-line (rsch, vr, lfac, tau, f_tau, Fr_hat,
     Fr_lab, apply_floor) -- this is a real regression check on the IC
     computation itself, to floating-point precision.
  2. Physical SANITY checks on the dynamical evolution (no NaN/Inf,
     positivity, causality |F|<=E via the proper metric contraction
     g^ab F_a F_b, not a flat Euclidean norm), plus bounded-growth checks
     calibrated to two characterized findings from validating this pgen
     (see DEVELOPMENT.md Stage 3):
       - the near-horizon region (|x1v| < NEAR_HORIZON_CUT) undergoes a
         fast (~1M), self-limiting local adjustment away from the ad hoc
         free-fall IC -- checked only for boundedness, not IC-matching;
       - the bulk/outer region drifts slowly (a boundary-condition artifact:
         this pgen has no inflow BC or horizon excision, see
         dyngr_bhstar_hydro.athinput's comment) -- checked to stay within a
         generous factor of its initial value over the short tlim used here.

Mesh geometry note: outputs use a "slice_x2=0.0, slice_x3=0.0" 1D cut along
the x1-axis. Since nx2=nx3=32 is even, no cell center sits exactly at 0 --
the slice lands on the nearest cell, offset by dx2/2=dx3/2 from the axis.
The true radius at each sampled row is therefore
    r = sqrt(x1v^2 + 2*OFFSET^2),   OFFSET = dx2/2 = dx3/2
not |x1v| -- confirmed empirically (innermost |x1v|=0.3125 has true r=0.541,
just outside the r=0.5 cutoff, matching the measured density there).
"""
import argparse
import glob
import math
import os
import sys

import numpy as np

# Must match inputs/tests/dyngr_bhstar_radiation.athinput
EOSK = 1.0
GAMMA = 5.0 / 3.0
BONDI_RS = 8.0
ARAD = 1.0
T_PHOTOSPHERE = 0.01
KAPPA_S = 1900.0
RAD_EPS = 0.01
BASENAME = "dyngr_bhstar_radiation"

# Mesh geometry (must match the athinput's <mesh> block)
NX2, X2MIN, X2MAX = 32, -10.0, 10.0
OFFSET = 0.5 * (X2MAX - X2MIN) / NX2  # dx2/2 = dx3/2 = 0.3125

# Region split for the dynamical sanity checks (see module docstring)
NEAR_HORIZON_CUT = 2.0  # |x1v| below this: fast local transient, bounded-only
BULK_GROWTH_FACTOR = 3.0     # bulk region: density/E must stay within this factor
HORIZON_GROWTH_FACTOR = 15.0  # near-horizon region: looser bound, no blowup


def true_radius(x1v):
    return np.sqrt(x1v**2 + 2.0 * OFFSET**2)


def rsch_of(r):
    return r * (1.0 + 0.5 / r) ** 2


def bondi_vr(rsch):
    return -0.5 * math.sqrt(2.0 / rsch) * (
        1.0 + 0.5 / math.sqrt(max(1.0e-8, rsch - 1.0)))


def expected_ic(x1v, lum):
    """Reproduce dyngr_bhstar.cpp's IC formulas exactly (r>0.5 branch only --
    every sampled row here has true_radius>0.541>0.5, see module docstring).
    x2v=x3v=OFFSET at the sliced coordinate (the C++ code uses the cell's
    true x2v/x3v, not 0 -- see module docstring)."""
    r = true_radius(x1v)
    rsch = rsch_of(r)
    psi2 = (1.0 + 0.5 / r) ** 2
    psi4 = psi2 * psi2
    vr = bondi_vr(rsch)
    vsq = psi4 * vr * vr + 1.0
    lfac = math.sqrt(vsq)

    rho = 0.0625 / (rsch * BONDI_RS) ** 1.5
    p = EOSK * rho ** GAMMA

    E_lte = ARAD * T_PHOTOSPHERE ** 4
    tau = KAPPA_S * 0.125 / math.pi / BONDI_RS / math.sqrt(BONDI_RS * rsch)
    f_tau = (1.0 - math.exp(-tau)) / (1.0 + tau)
    Fr_hat = f_tau * lum / (4.0 * math.pi * rsch * rsch)
    Fr_lab = Fr_hat + vr * E_lte / lfac

    inv_r = 1.0 / r
    F_raw = [0.0, Fr_lab * x1v * inv_r, Fr_lab * OFFSET * inv_r, Fr_lab * OFFSET * inv_r]
    E_rad, F_final = apply_floor_py(E_lte, F_raw, psi4)
    return rho, p, vr, E_rad, F_final[1]


def apply_floor_py(E, F_d, psi4):
    """Transcription of apply_floor() in radiation_m1_helpers.hpp. The
    causality bound uses the metric contraction F2 = g^ab F_a F_b, NOT the
    flat Euclidean norm -- for this diagonal, zero-shift metric,
    g_uu(1,1)=g_uu(2,2)=g_uu(3,3)=1/psi4 (g_uu(0,0) doesn't enter since
    F_d(0)=beta^i F_i=0 here)."""
    rad_e_floor = 1e-30
    rad_eps = RAD_EPS
    finite = all(math.isfinite(v) for v in F_d) and math.isfinite(E)
    at_floor = (not finite) or E <= rad_e_floor
    E = max(rad_e_floor, E) if math.isfinite(E) else rad_e_floor
    if at_floor:
        return E, [0.0, 0.0, 0.0, 0.0]
    F2 = (F_d[1]**2 + F_d[2]**2 + F_d[3]**2) / psi4
    lim = E * E * (1.0 - rad_eps)
    if F2 > lim:
        fac = lim / F2
        F_d = [F_d[0]] + [f * fac for f in F_d[1:]]
    return E, F_d


def read_slice_series(tab_dir, file_id, columns):
    pattern = os.path.join(tab_dir, "{}.{}.*.tab".format(BASENAME, file_id))
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit("No files matched {} -- run AthenaK first".format(pattern))
    rows = []
    for fname in files:
        with open(fname) as f:
            header = f.readline()
            f.readline()  # column-name header, unused (columns given explicitly)
            data = np.loadtxt(f)
        time = float(header.split("time=")[1].split()[0])
        x1v = data[:, 2]
        cols = {name: data[:, 3 + idx] for idx, name in enumerate(columns)}
        rows.append((time, x1v, cols))
    rows.sort(key=lambda r: r[0])
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tab-dir", default="tab", help="directory with .tab output")
    parser.add_argument("--ic-tol", type=float, default=1e-4,
                        help="relative tolerance on the t=0 exact IC-formula check "
                             "(loosened from a tighter first guess to cover "
                             "Python-vs-C++ math-library rounding in the pow()/"
                             "sqrt() chain, measured at ~2.6e-6)")
    parser.add_argument("--bulk-tol", type=float, default=BULK_GROWTH_FACTOR,
                        help="max growth factor allowed in the bulk region")
    parser.add_argument("--horizon-tol", type=float, default=HORIZON_GROWTH_FACTOR,
                        help="max growth factor allowed near the horizon")
    args = parser.parse_args(argv)

    hydro_rows = read_slice_series(
        args.tab_dir, "mhd_w", ["dens", "velx", "vely", "velz", "press", "temp"])
    E_rows = read_slice_series(args.tab_dir, "rad_m1_E", ["E"])
    F_rows = read_slice_series(args.tab_dir, "rad_m1_F", ["Fx", "Fy", "Fz"])

    ok = True

    # ---- Check 1: exact t=0 IC-formula regression check ----
    kappa_s = KAPPA_S
    lum_edd = 4.0 * math.pi / kappa_s
    t0, x1v0, hydro0 = hydro_rows[0]
    _, _, E0 = E_rows[0]
    _, _, F0 = F_rows[0]
    assert t0 == 0.0 and E_rows[0][0] == 0.0 and F_rows[0][0] == 0.0

    max_rho_err = max_p_err = max_E_err = max_Fx_err = 0.0
    for i, x1 in enumerate(x1v0):
        rho_e, p_e, vr_e, E_e, Fx_e = expected_ic(x1, lum_edd)
        rho_a = hydro0["dens"][i]
        p_a = hydro0["press"][i]
        E_a = E0["E"][i]
        Fx_a = F0["Fx"][i]
        max_rho_err = max(max_rho_err, abs(rho_a - rho_e) / rho_e)
        max_p_err = max(max_p_err, abs(p_a - p_e) / p_e)
        max_E_err = max(max_E_err, abs(E_a - E_e) / max(E_e, 1e-300))
        denom = max(abs(Fx_e), 1e-20)
        max_Fx_err = max(max_Fx_err, abs(Fx_a - Fx_e) / denom)

    print("=== Check 1: exact t=0 IC-formula regression ===")
    print("max rel err: dens={:.3e} press={:.3e} E={:.3e} Fx={:.3e} (tol {:.1e})".format(
        max_rho_err, max_p_err, max_E_err, max_Fx_err, args.ic_tol))
    ic_ok = (max_rho_err < args.ic_tol and max_p_err < args.ic_tol and
             max_E_err < args.ic_tol and max_Fx_err < args.ic_tol)
    print("PASS" if ic_ok else "FAIL", ": t=0 IC matches the C++ formula")
    ok = ok and ic_ok

    # ---- Check 2: dynamical sanity ----
    print("\n=== Check 2: dynamical sanity ===")

    # psi4(r) is a fixed function of the (static) grid coordinates -- same at
    # every output time -- so compute it once from x1v0.
    r_grid = true_radius(x1v0)
    psi4_grid = np.array([((1.0 + 0.5 / r) ** 2) ** 2 for r in r_grid])

    all_finite = True
    all_positive = True
    causality_ok = True
    for (t, x1v, hydro), (_, _, E), (_, _, F) in zip(hydro_rows, E_rows, F_rows):
        for name in ("dens", "press"):
            arr = hydro[name]
            if not np.all(np.isfinite(arr)):
                all_finite = False
            if np.any(arr <= 0.0):
                all_positive = False
        if not np.all(np.isfinite(E["E"])):
            all_finite = False
        if np.any(E["E"] <= 0.0):
            all_positive = False
        for name in ("Fx", "Fy", "Fz"):
            if not np.all(np.isfinite(F[name])):
                all_finite = False

        # F2 = g^ab F_a F_b = (Fx^2+Fy^2+Fz^2)/psi4 for this diagonal,
        # zero-shift metric (see apply_floor_py) -- NOT the flat Euclidean norm.
        F2 = (F["Fx"]**2 + F["Fy"]**2 + F["Fz"]**2) / psi4_grid
        if np.any(F2 > E["E"]**2 * (1.0 + 1e-6)):
            causality_ok = False

    print("all finite (no NaN/Inf)      :", "PASS" if all_finite else "FAIL")
    print("dens/press/E stay positive   :", "PASS" if all_positive else "FAIL")
    print("causality |F| <= E           :", "PASS" if causality_ok else "FAIL")
    ok = ok and all_finite and all_positive and causality_ok

    # Bounded-growth checks, split into near-horizon vs. bulk (see docstring)
    near_mask = np.abs(x1v0) < NEAR_HORIZON_CUT
    bulk_mask = ~near_mask
    _, x1v_last, hydro_last = hydro_rows[-1]
    _, _, E_last = E_rows[-1]
    assert np.allclose(x1v0, x1v_last)

    def max_growth(field0, field_last, mask):
        ratio = field_last[mask] / np.maximum(field0[mask], 1e-300)
        return np.max(np.abs(ratio))

    bulk_dens_growth = max_growth(hydro0["dens"], hydro_last["dens"], bulk_mask)
    horizon_dens_growth = max_growth(hydro0["dens"], hydro_last["dens"], near_mask)
    bulk_E_growth = max_growth(E0["E"], E_last["E"], bulk_mask)
    horizon_E_growth = max_growth(E0["E"], E_last["E"], near_mask)

    print("\nbulk region (|x1v|>={:.1f}) max growth : dens={:.2f}x E={:.2f}x "
          "(tol {:.1f}x)".format(
              NEAR_HORIZON_CUT, bulk_dens_growth, bulk_E_growth, args.bulk_tol))
    print("near-horizon (|x1v|<{:.1f}) max growth : dens={:.2f}x E={:.2f}x "
          "(tol {:.1f}x)".format(
              NEAR_HORIZON_CUT, horizon_dens_growth, horizon_E_growth, args.horizon_tol))
    growth_ok = (bulk_dens_growth < args.bulk_tol and bulk_E_growth < args.bulk_tol and
                 horizon_dens_growth < args.horizon_tol and
                 horizon_E_growth < args.horizon_tol)
    print("PASS" if growth_ok else "FAIL", ": growth stays within the "
          "characterized bounds")
    ok = ok and growth_ok

    print()
    if ok:
        print("PASS: dyngr_bhstar radiation-on run is physically sane")
    else:
        print("FAIL: one or more checks failed")
        return False
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
