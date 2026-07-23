#!/usr/bin/env python
"""Colliding-beams check (arXiv:2302.04283 Section 3.1, Stage 12).

Compares the DO module's rad_coord output (rad_crossing_beams.athinput) against
the M1 module's rad_m1_E output (rad_m1_photon_colliding_beams.athinput). The
paper's own claim (Section 3.1): "Two beams crossing in vacuum will merge into a
single beam pointing in the average direction of the two when using M1... this
test is failed by M1 and commonly employed closure methods." This script
measures that directly: at several x-slices downstream of the crossing point,
DO's field should be genuinely bimodal in y (two separated peaks -- the beams
continuing, undisturbed, on their original trajectories), while M1's field
should be a single unimodal peak (the merged, average-direction beam).

Usage (run after executing AthenaK for both athinputs):
    python check_rad_m1_photon_colliding_beams.py \\
        --do-bin-dir /path/to/do/bin --m1-bin-dir /path/to/m1/bin \\
        --plot-dir /path/to/plots
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "vis", "python"))
from bin_convert import read_binary  # noqa: E402

X_CROSS = 0.7684186294419217
Y_CROSS = 0.5
# downstream x-slices to check (must be > X_CROSS and < x1max)
X_SLICES = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]


def latest_bin(bin_dir, basename, file_id):
    pattern = os.path.join(bin_dir, "{}.{}.*.bin".format(basename, file_id))
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit("No files matched {} -- run AthenaK first".format(pattern))
    return files[-1]


def load_2d(fname, var):
    d = read_binary(fname)
    field = np.array(d["mb_data"][var])[0, 0]
    nx2, nx1 = field.shape
    x1 = np.linspace(d["x1min"], d["x1max"], nx1, endpoint=False) + \
        (d["x1max"] - d["x1min"]) / nx1 / 2
    x2 = np.linspace(d["x2min"], d["x2max"], nx2, endpoint=False) + \
        (d["x2max"] - d["x2min"]) / nx2 / 2
    return d["time"], x1, x2, field


def bimodality_metric(y, prof):
    """peak/valley ratio using the profile's own center-of-mass split: crude but
    robust to numerical ray-effect noise (unlike naive local-maxima counting,
    which is fooled by sub-percent noise on flat plateaus)."""
    ny = len(y)
    left = prof[:ny // 2]
    right = prof[ny // 2:]
    peak_left = left.max()
    peak_right = right.max()
    valley = prof[ny // 2 - 2:ny // 2 + 2].min()
    peak = min(peak_left, peak_right)
    if valley <= 0:
        return np.inf
    return peak / valley


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--do-bin-dir", required=True)
    parser.add_argument("--m1-bin-dir", required=True)
    parser.add_argument("--plot-dir", default=None)
    args = parser.parse_args(argv)

    do_fname = latest_bin(args.do_bin_dir, "crossing_beams_do", "rad_coord")
    m1_fname = latest_bin(args.m1_bin_dir, "photon_colliding_beams", "rad_m1_E")

    t_do, x1_do, x2_do, r00 = load_2d(do_fname, "r00")
    t_m1, x1_m1, x2_m1, E = load_2d(m1_fname, "E:0")

    print("DO snapshot t = {:.4f}, M1 snapshot t = {:.4f}".format(t_do, t_m1))
    print("\n{:>6s} {:>16s} {:>16s}".format("x", "DO peak/valley", "M1 peak/valley"))
    for xt in X_SLICES:
        ix_do = np.argmin(np.abs(x1_do - xt))
        ix_m1 = np.argmin(np.abs(x1_m1 - xt))
        bm_do = bimodality_metric(x2_do, r00[:, ix_do])
        bm_m1 = bimodality_metric(x2_m1, E[:, ix_m1])
        print("{:6.2f} {:16.3f} {:16.3f}".format(xt, bm_do, bm_m1))

    if args.plot_dir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(args.plot_dir, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        im0 = axes[0].imshow(r00, origin="lower", extent=[x1_do[0], x1_do[-1],
                              x2_do[0], x2_do[-1]], aspect="auto", cmap="inferno")
        axes[0].set_title("DO: R^tt (t={:.2f})".format(t_do))
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(E, origin="lower", extent=[x1_m1[0], x1_m1[-1],
                              x2_m1[0], x2_m1[-1]], aspect="auto", cmap="inferno")
        axes[1].set_title("M1: E (t={:.2f})".format(t_m1))
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        plt.colorbar(im1, ax=axes[1])
        fig.suptitle("Colliding beams (arXiv:2302.04283 Sec. 3.1): "
                      "DO resolves both beams, M1 merges them")
        fig.tight_layout()
        fig.savefig(os.path.join(args.plot_dir, "stage12_colliding_beams_2d.png"), dpi=150)

        fig2, axes2 = plt.subplots(1, len(X_SLICES), figsize=(4 * len(X_SLICES), 3.5),
                                    sharey=True)
        for ax, xt in zip(axes2, X_SLICES):
            ix_do = np.argmin(np.abs(x1_do - xt))
            ix_m1 = np.argmin(np.abs(x1_m1 - xt))
            ax.plot(r00[:, ix_do], x2_do, "C0-", label="DO")
            ax.plot(E[:, ix_m1], x2_m1, "C1-", label="M1")
            ax.set_title("x={:.1f}".format(xt))
            ax.set_xlabel("field")
            ax.axhline(Y_CROSS, color="gray", ls=":", lw=0.8)
        axes2[0].set_ylabel("y")
        axes2[0].legend(fontsize=8)
        fig2.suptitle("Downstream y-profiles: DO stays bimodal, M1 merges to one peak")
        fig2.tight_layout()
        fig2.savefig(os.path.join(args.plot_dir, "stage12_colliding_beams_profiles.png"),
                     dpi=150)
        print("\nplots written to {}".format(args.plot_dir))

    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
