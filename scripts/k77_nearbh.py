#!/usr/bin/env python3
"""Near-BH gas: corrected trumpet fill (K77 = Q1 restarted at t = 121.5 with the corrected
Ahat0) vs Q1 (old fill), with N5 (Q1 repeated, same binary and fill) as the run-to-run floor.

Per prim_xy dump, t = 122-160: max rho/rho_c and the orbital-plane slice mass in the annuli
r = 2-3, 3-6 and 6-10 M around the BH (at the origin, BH frame); and the excision-accreted mass
since t = 121.5 from ensemble.out (the tally restarts at 0 on a restart).

    k77_nearbh.py [--out DIR]
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tde_t11_surface_diag as T                  # noqa: E402,F401 (sets up bc)
from tde_ensemble_plots import bc                 # noqa: E402

R = T.RUN
B = 'cfc_tde_wd_imbh_parabolic_tracker'
RHOC = 3.24e-4
RUNS = {'Q1 (old fill)': ('Q1_V5L4_fix_t300', 'C3'), 'N5 (Q1 repeat)': ('N5_Q1_repeat_t150', '0.5'),
        'K77 (corrected fill)': ('K77_q1_rst5', 'C0')}
ANN = ((2, 3), (3, 6), (6, 10))
T0, T1 = 121.5, 160.0


def accreted(d):
    rows, t = [], None
    for line in open(f'{R}/{d}/ensemble.out'):
        m = re.search(r'time=([-+.\deE]+)', line)
        if m:
            t = float(m.group(1))
        m = re.search(r'r=3\.0+e\+01 M_res=.*M_accreted=([-+.\deE]+)', line)
        if m and t is not None:
            rows.append((t, float(m.group(1))))
    a = np.array(rows)
    a = a[(a[:, 0] >= T0 - 0.6) & (a[:, 0] <= T1 + 0.1)]
    return a[:, 0], a[:, 1] - np.interp(T0, a[:, 0], a[:, 1])


def annuli(f):
    q = bc.read_binary(f)
    g, nx = np.asarray(q['mb_geometry']), q['nx1_out_mb']
    rho = np.asarray(q['mb_data']['dens'])[:, 0]
    mx, ms = np.zeros(len(ANN)), np.zeros(len(ANN))
    for m in range(len(g)):
        if g[m, 0] > 10 or g[m, 1] < -10 or g[m, 2] > 10 or g[m, 3] < -10:
            continue                                  # block entirely outside |x|, |y| <= 10
        h = (g[m, 1] - g[m, 0]) / nx
        x = g[m, 0] + (np.arange(nx) + .5) * h
        y = g[m, 2] + (np.arange(nx) + .5) * h
        r = np.hypot(*np.meshgrid(x, y))
        for n, (a, b) in enumerate(ANN):
            s = (r >= a) & (r < b)
            if s.any():
                mx[n] = max(mx[n], rho[m][s].max())
                ms[n] += (rho[m][s] * h * h).sum()
    return q['time'], mx / RHOC, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{R}/plots/phase2_m1')
    a = ap.parse_args()
    rows = []
    fig, ax = plt.subplots(1, 4, figsize=(22, 4.8), constrained_layout=True)
    S = {}
    for lab, (d, col) in RUNS.items():
        fs = [f for f in sorted(glob.glob(f'{R}/{d}/bin/{B}.prim_xy.*.bin'))]
        res = [annuli(f) for f in fs]
        res = [r for r in res if T0 - 0.01 <= r[0] <= T1 + 0.01]
        t = np.array([r[0] for r in res]); mx = np.array([r[1] for r in res]); ms = np.array([r[2] for r in res])
        ta, acc = accreted(d)
        S[lab] = (t, mx, ms, ta, acc)
        for n, (lo, hi) in enumerate(ANN):
            ax[n].semilogy(t, mx[:, n], 'o-', ms=3, color=col, label=lab)
        ax[3].semilogy(ta, np.maximum(acc, 1e-20), '-', color=col, label=lab)
        pick = [125, 130, 140, 150, 160]
        rows.append(f'{lab}: ' + '; '.join(
            f't={p}: max rho/rhoc ' + '/'.join(f'{np.interp(p, t, mx[:, n]):.1e}' for n in range(3))
            + f', accreted {np.interp(p, ta, acc):.2e}' for p in pick if t.max() >= p - 0.5))
    for n, (lo, hi) in enumerate(ANN):
        ax[n].set_title(f'max ρ/ρ_c, r = {lo}-{hi} M (z = 0)')
    ax[3].set_title('excision-accreted mass since t = 121.5')
    for x in ax:
        x.set_xlabel('t [M]'); x.grid(alpha=0.3); x.legend(fontsize=8)
    fig.suptitle('Near-BH gas: corrected trumpet fill (K77) vs Q1 (old fill) and N5 (Q1 repeat, run-to-run floor)')
    fig.savefig(f'{a.out}/k77_nearbh.png', dpi=80)
    with open(f'{a.out}/k77_nearbh.txt', 'w') as f:
        f.write('\n'.join(rows) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
