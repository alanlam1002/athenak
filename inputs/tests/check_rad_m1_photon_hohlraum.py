#!/usr/bin/env python
"""1D plane-parallel hohlraum check (arXiv:2302.04283 Section 3.4, Stage 11).

Compares the DO module's rad_coord output (hohlraum_1d.athinput, built via
-DPROBLEM=rad_hohlraum) against the M1 module's rad_m1_E/rad_m1_F output
(rad_m1_photon_hohlraum_{minerbo,eddington}.athinput), and both against the
paper's closed-form solution (their Eq. 61):

    R^tt = (1/2)(1 - x/t),   R^tx = (1/4)(1 - x^2/t^2),
    R^xx = (1/6)(1 - x^3/t^3)   for x < t (all vanish for x > t)

DO's "rad_coord" output gives R^{alpha beta} directly in the coordinate
frame (derived_variables.cpp) -- r00=R^tt, r01=R^tx, r11=R^xx, no further
processing needed.

M1 has no direct Rxx output. But for this test the fluid is static, flat,
gr_sources=false (alpha=1, beta=0, v=0), so the Eulerian/coordinate frame
IS the fluid frame here: M1's own evolved (E, F_x) already equal R^tt, R^tx
directly. R^xx is recovered from M1's closure relation for a purely 1D
field aligned with F: P^xx = chi(xi)*E, with xi=|F_x|/E and chi(xi) given
by the same closure_fun() formula the solver itself uses
(radiation_m1_closure.hpp) -- reproduced here rather than read from the
code to keep this an independent check.

Usage (run from each run directory, after executing AthenaK):
    python check_rad_m1_photon_hohlraum.py \\
        --do-tab-dir /path/to/do/tab --m1-minerbo-tab-dir /path/to/m1_minerbo/tab \\
        --m1-eddington-tab-dir /path/to/m1_eddington/tab --plot-dir /path/to/plots
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from m1_tab_utils import read_profile  # noqa: E402

DO_BASENAME = "hohlraum_1d"
M1_BASENAME = {"minerbo": "photon_hohlraum_minerbo", "eddington": "photon_hohlraum_eddington"}


def closure_chi(xi, closure):
    if closure == "eddington":
        return 1.0 / 3.0 * np.ones_like(xi)
    if closure == "minerbo":
        return 1.0 / 3.0 + xi**2 * (6.0 - 2.0 * xi + 6.0 * xi**2) / 15.0
    raise ValueError(closure)


def analytic(x, t):
    rtt = np.where(x < t, 0.5 * (1.0 - x / t), 0.0)
    rtx = np.where(x < t, 0.25 * (1.0 - (x / t) ** 2), 0.0)
    rxx = np.where(x < t, (1.0 / 6.0) * (1.0 - (x / t) ** 3), 0.0)
    return rtt, rtx, rxx


def latest_tab(tab_dir, basename, file_id):
    pattern = os.path.join(tab_dir, "{}.{}.*.tab".format(basename, file_id))
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit("No files matched {} -- run AthenaK first".format(pattern))
    return files[-1]


def find_col(data, prefix):
    for name in data:
        if name == prefix or name.startswith(prefix + ":"):
            return data[name]
    raise KeyError("no column matching {!r} in {}".format(prefix, list(data)))


def load_do(tab_dir):
    fname = latest_tab(tab_dir, DO_BASENAME, "rad_coord")
    x1v, data = read_profile(fname)
    return data["time"], x1v, find_col(data, "r00"), find_col(data, "r01"), find_col(data, "r11")


def load_m1(tab_dir, closure):
    basename = M1_BASENAME[closure]
    fname_e = latest_tab(tab_dir, basename, "rad_m1_E")
    fname_f = latest_tab(tab_dir, basename, "rad_m1_F")
    x1v_e, data_e = read_profile(fname_e)
    x1v_f, data_f = read_profile(fname_f)
    assert abs(data_e["time"] - data_f["time"]) < 1e-9
    assert np.allclose(x1v_e, x1v_f)
    E = find_col(data_e, "E")
    Fx = find_col(data_f, "Fx")
    xi = np.abs(Fx) / np.maximum(E, 1e-300)
    chi = closure_chi(xi, closure)
    Rxx = chi * E
    return data_e["time"], x1v_e, E, Fx, Rxx


def l1_error(x1v, field, field_exact):
    # trapezoidal L1 norm over x in [0,1], matching the paper's eps_ab (Eq 62b)
    return np.trapz(np.abs(field - field_exact), x1v)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--do-tab-dir", required=True)
    parser.add_argument("--m1-minerbo-tab-dir", required=True)
    parser.add_argument("--m1-eddington-tab-dir", required=True)
    parser.add_argument("--plot-dir", default=None)
    args = parser.parse_args(argv)

    t_do, x_do, rtt_do, rtx_do, rxx_do = load_do(args.do_tab_dir)
    t_mn, x_mn, e_mn, fx_mn, rxx_mn = load_m1(args.m1_minerbo_tab_dir, "minerbo")
    t_ed, x_ed, e_ed, fx_ed, rxx_ed = load_m1(args.m1_eddington_tab_dir, "eddington")

    print("DO snapshot t = {:.4f}, M1 minerbo t = {:.4f}, M1 eddington t = {:.4f}"
          .format(t_do, t_mn, t_ed))

    rtt_an_do, rtx_an_do, rxx_an_do = analytic(x_do, t_do)
    rtt_an_mn, rtx_an_mn, rxx_an_mn = analytic(x_mn, t_mn)
    rtt_an_ed, rtx_an_ed, rxx_an_ed = analytic(x_ed, t_ed)

    print("\n{:>10s} {:>12s} {:>12s} {:>12s}".format("field", "DO", "M1-minerbo", "M1-eddington"))
    eps_do = l1_error(x_do, rtt_do, rtt_an_do) + l1_error(x_do, rtx_do, rtx_an_do) + \
        l1_error(x_do, rxx_do, rxx_an_do)
    eps_mn = l1_error(x_mn, e_mn, rtt_an_mn) + l1_error(x_mn, fx_mn, rtx_an_mn) + \
        l1_error(x_mn, rxx_mn, rxx_an_mn)
    eps_ed = l1_error(x_ed, e_ed, rtt_an_ed) + l1_error(x_ed, fx_ed, rtx_an_ed) + \
        l1_error(x_ed, rxx_ed, rxx_an_ed)
    print("{:>10s} {:12.5e} {:12.5e} {:12.5e}".format("eps (sum |R-R_exact| dx, tt+tx+xx)",
                                                        eps_do, eps_mn, eps_ed))

    if args.plot_dir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(args.plot_dir, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
        labels = ["R^tt", "R^tx", "R^xx"]
        do_fields = [rtt_do, rtx_do, rxx_do]
        mn_fields = [e_mn, fx_mn, rxx_mn]
        ed_fields = [e_ed, fx_ed, rxx_ed]
        an_do = [rtt_an_do, rtx_an_do, rxx_an_do]
        for ax, label, do_f, mn_f, ed_f, an_f in zip(
                axes, labels, do_fields, mn_fields, ed_fields, an_do):
            ax.plot(x_do, an_f, "k-", lw=2, label="exact")
            ax.plot(x_do, do_f, "C0o", ms=3, label="DO")
            ax.plot(x_mn, mn_f, "C1-", lw=1.5, label="M1 minerbo")
            ax.plot(x_ed, ed_f, "C2--", lw=1.5, label="M1 eddington")
            ax.set_xlabel("x")
            ax.set_title(label)
            ax.legend(fontsize=8)
        fig.suptitle("1D hohlraum at t={:.3f} (paper Fig. 11, arXiv:2302.04283)".format(t_do))
        fig.tight_layout()
        outpng = os.path.join(args.plot_dir, "stage11_hohlraum_1d.png")
        fig.savefig(outpng, dpi=150)
        print("\nplot written to {}".format(outpng))

    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
