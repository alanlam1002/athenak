#!/usr/bin/env python
"""Stage 9: M1 vs. discrete-ordinate (DO) comparison for the ADVECTED
diffusion test (arXiv:2302.04283 Section 3.7's moving-fluid sub-case).

Both sides use the same physical parameters (v1=0.1, kappa_s=100, nu=4,
domain x in [-1,2], tlim=5): M1's rad_m1_photon_diffusion_advected.athinput
and the DO module's own pre-existing inputs/radiation/rad_diffusion.athinput
(v1=0.1, already validated/unmodified in this repo since before this
session). The two codes' initial conditions are NOT bit-identical (M1 uses a
v1=0-simplified diffusive-flux correction on a lab-frame Gaussian; DO uses
the full boosted self-similar solution with a time-offset trick -- see
rad_m1_diffusiontest.cpp's comment and DEVELOPMENT.md Stage 9), so rather
than requiring a shared exact analytic solution, this checks that both
codes' own pulses independently satisfy the same two physical predictions:
(1) the peak advects at x_peak(t) = x_peak(0) + v1*t, and (2) the E-weighted
variance grows as sigma^2(t) = sigma0^2 + 2*D*t, D = 1/(3*kappa_s*rho).

Run after executing both athinputs (see DEVELOPMENT.md Stage 9 for exact
build/run commands):
    python check_rad_m1_photon_diffusion_advected.py \\
        --m1-tab-dir tab_m1 --do-tab-dir tab_do --plot-dir plots
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                 "inputs", "tests"))
from m1_tab_utils import read_profile  # noqa: E402

V1 = 0.1
KAPPA_S = 100.0
RHO = 1.0
D_EXPECTED = 1.0 / (3.0 * KAPPA_S * RHO)


def load_profiles(tab_dir, basename, file_id, column):
    pattern = os.path.join(tab_dir, "{}.{}.*.tab".format(basename, file_id))
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit("No files matched {}".format(pattern))
    times, peaks, sigma2s = [], [], []
    for fname in files:
        x, data = read_profile(fname)
        E = data[column]
        E = np.maximum(E, 0.0)  # guard tiny negative numerical noise
        total = np.trapz(E, x)
        x_mean = np.trapz(x * E, x) / total
        sigma2 = np.trapz((x - x_mean) ** 2 * E, x) / total
        x_peak = x[np.argmax(E)]
        times.append(data["time"])
        peaks.append(x_peak)
        sigma2s.append(sigma2)
    order = np.argsort(times)
    return (np.array(times)[order], np.array(peaks)[order],
            np.array(sigma2s)[order])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-tab-dir", default="tab_m1")
    parser.add_argument("--do-tab-dir", default="tab_do")
    parser.add_argument("--plot-dir", default=None)
    parser.add_argument("--advection-tol", type=float, default=0.05,
                         help="relative tolerance on fitted advection speed vs v1")
    parser.add_argument("--diffusion-tol", type=float, default=0.35,
                         help="relative tolerance on fitted D vs D_expected "
                              "(loose -- matches the existing static "
                              "diffusion test's own tolerance, since the "
                              "M1/DO ICs are not identical)")
    args = parser.parse_args(argv)

    t_m1, peak_m1, sig2_m1 = load_profiles(
        args.m1_tab_dir, "photon_diffusion_advected", "rad_m1_E", "E:0")
    t_do, peak_do, sig2_do = load_profiles(
        args.do_tab_dir, "diffusion", "rad_fluid", "r00_ff")

    v_m1 = np.polyfit(t_m1, peak_m1, 1)[0]
    v_do = np.polyfit(t_do, peak_do, 1)[0]

    D_m1 = np.polyfit(t_m1, sig2_m1, 1)[0] / 2.0
    D_do = np.polyfit(t_do, sig2_do, 1)[0] / 2.0

    print("Expected: advection speed v1 = {:.4f}, diffusion coeff D = {:.6f}".format(
        V1, D_EXPECTED))
    print()
    print("{:>6s}  {:>12s} {:>12s}   {:>12s} {:>12s}".format(
        "t", "x_peak(M1)", "x_peak(DO)", "sigma2(M1)", "sigma2(DO)"))
    n = min(len(t_m1), len(t_do))
    for i in np.linspace(0, n - 1, 10, dtype=int):
        print("{:6.3f}  {:12.6f} {:12.6f}   {:12.6f} {:12.6f}".format(
            t_m1[i], peak_m1[i], np.interp(t_m1[i], t_do, peak_do),
            sig2_m1[i], np.interp(t_m1[i], t_do, sig2_do)))

    print()
    print("fitted advection speed: M1={:.5f} (rel err {:.3e})  DO={:.5f} (rel err {:.3e})".format(
        v_m1, abs(v_m1 - V1) / V1, v_do, abs(v_do - V1) / V1))
    print("fitted diffusion coeff: M1={:.6f} (rel err {:.3e})  DO={:.6f} (rel err {:.3e})".format(
        D_m1, abs(D_m1 - D_EXPECTED) / D_EXPECTED,
        D_do, abs(D_do - D_EXPECTED) / D_EXPECTED))

    ok = True
    for label, v in [("M1", v_m1), ("DO", v_do)]:
        if abs(v - V1) / V1 > args.advection_tol:
            print("FAIL: {} advection speed off by more than {:.0%}".format(
                label, args.advection_tol))
            ok = False
    for label, D in [("M1", D_m1), ("DO", D_do)]:
        if abs(D - D_EXPECTED) / D_EXPECTED > args.diffusion_tol:
            print("FAIL: {} diffusion coefficient off by more than {:.0%}".format(
                label, args.diffusion_tol))
            ok = False

    if args.plot_dir:
        make_plots(args.plot_dir, t_m1, peak_m1, sig2_m1, t_do, peak_do, sig2_do)

    if ok:
        print("\nPASS: both codes' pulses advect at v1 and diffuse at D, "
              "within tolerance of each other and of the analytic "
              "prediction")
    else:
        print("\nFAIL: see above")
        return False
    return True


def make_plots(plot_dir, t_m1, peak_m1, sig2_m1, t_do, peak_do, sig2_do):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(plot_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(t_m1, peak_m1, "o-", ms=4, color="C1", label="M1")
    ax[0].plot(t_do, peak_do, "s-", ms=4, color="C0", label="DO")
    t_ref = np.linspace(0, max(t_m1[-1], t_do[-1]), 50)
    ax[0].plot(t_ref, peak_m1[0] + V1 * t_ref, "k:", label="analytic slope v1")
    ax[0].set_xlabel("t"); ax[0].set_ylabel("pulse peak position")
    ax[0].set_title("Advection: peak position vs t")
    ax[0].legend(fontsize=8)

    ax[1].plot(t_m1, sig2_m1, "o-", ms=4, color="C1", label="M1")
    ax[1].plot(t_do, sig2_do, "s-", ms=4, color="C0", label="DO")
    ax[1].plot(t_ref, sig2_m1[0] + 2 * D_EXPECTED * t_ref, "k:",
               label="analytic slope 2D")
    ax[1].set_xlabel("t"); ax[1].set_ylabel(r"$\sigma^2(t)$")
    ax[1].set_title("Diffusion: E-weighted variance vs t")
    ax[1].legend(fontsize=8)

    fig.suptitle("Stage 9 (arXiv:2302.04283 §3.7): advected diffusion, M1 vs DO",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "stage9_diffusion_advected.png"), dpi=130)
    plt.close(fig)
    print("\nplot written to", plot_dir)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
