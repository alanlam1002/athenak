#!/usr/bin/env python3
"""C-098 (R-103 E2): shock-free heating, coord vs covariant divergence gate, deterministic pairs.

Runs c097_e2/{R6,R7}{coord,cov} (fixed binary 30c52c1d, (A) on; R6: X11 deck from R6 00005, t 97.1 -> 102.5;
R7: production deck from R7 00005, t 101.5 -> 107; only <mhd> dual_energy_div differs within a pair). Per run:
  - energy-branch fraction of the dense cells, whole grid: 1 - de-nentropy/de-ndense (user.hst, the last full C2P
    of each history row; dyngr_tov.cpp history columns, primitive_solver_hyd.hpp de_ndense/de_nent);
  - on the z = 0 slice (dediag_xy, primitive_solver_hyd.hpp l. 208-219: channel 0 branch, 2 final shock flag):
    the flagged fraction of dense cells (branch > 0) and the energy-branch fraction (branch 4 or 5);
  - from each restart (c051_s2.rst_pass / physics, r_BH >= 3): the dense mean K/K0 and M3 (K/K0 - 1 > 0.05 mass
    fraction) against t, and their drift over the window (the shock-free answer is ~ 0).

    c098_e2.py [--out FILE]
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c051_s2 as S                                # noqa: E402
from tde_ensemble_plots import bc                  # noqa: E402

E = f'{S.C}/c097_e2'
BASE = 'cfc_tde_wd_imbh_parabolic_tracker'
RUNS = ('R6coord', 'R6cov', 'R7coord', 'R7cov')


def run_rows(d):
    u = np.loadtxt(glob.glob(f'{d}/{BASE}.user.hst')[0])
    _, i = np.unique(u[:, 0], return_index=True); u = u[i]
    ok = u[:, 15] > 0
    fen = 1.0 - u[ok, 16] / u[ok, 15]
    sl = []
    for fn in sorted(glob.glob(f'{d}/bin/{BASE}.dediag_xy.*.bin')):
        q = bc.read_binary(fn)
        br = np.asarray(q['mb_data']['de_branch'])[:, 0]
        sh = np.asarray(q['mb_data']['de_shock'])[:, 0]
        dn = br > 0
        sl.append((q['time'], dn.sum(), (sh[dn] > 0.5).mean() if dn.any() else np.nan,
                   np.isin(br[dn], (4, 5)).mean() if dn.any() else np.nan))
    ph = []
    for fn in sorted(glob.glob(f'{d}/rst/*.rst')):
        z = S.rst_pass(fn)
        p = S.physics(z)
        ph.append((p['t'], p['K'], p['M3']))
    return u[ok, 0], fen, np.array(sl), np.array(ph)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out')
    a = ap.parse_args()
    out = ['C-098 E2: shock-free windows, coord vs covariant (deterministic pairs, fixed binary 30c52c1d, (A) on)']
    R = {r: run_rows(f'{E}/{r}') for r in RUNS}
    for r in RUNS:
        t, fen, sl, ph = R[r]
        out.append(f'== {r}: hst rows {len(t)}, t {t[0]:.2f}-{t[-1]:.2f}; energy-branch fraction (grid) mean {fen.mean():.4f} '
                   f'(min {fen.min():.4f}, max {fen.max():.4f})')
        out.append('   slice z=0:  t      n_dense  flagged  energy-branch')
        for row in sl:
            out.append(f'             {row[0]:6.2f} {int(row[1]):7d}  {row[2]:.4f}   {row[3]:.4f}')
        out.append('   restarts:   t      K/K0 (dense mean)   M3')
        for row in ph:
            out.append(f'             {row[0]:6.2f}  {row[1]:.8f}        {row[2]:.6f}')
        if len(ph) > 1:
            out.append(f'   drift over the window: d(K/K0) {ph[-1, 1] - ph[0, 1]:+.3e}  dM3 {ph[-1, 2] - ph[0, 2]:+.3e} '
                       f'(t {ph[0, 0]:.2f} -> {ph[-1, 0]:.2f})')
    for s in ('R6', 'R7'):
        c, v = R[s + 'coord'], R[s + 'cov']
        n = min(len(c[3]), len(v[3]))
        if n:
            out.append(f'-- {s} covariant - coord at the last common restart (t {v[3][n - 1, 0]:.2f}): '
                       f'K/K0 {v[3][n - 1, 1] - c[3][n - 1, 1]:+.3e}, M3 {v[3][n - 1, 2] - c[3][n - 1, 2]:+.3e}; '
                       f'mean flagged (slice) coord {np.nanmean(c[2][:, 2]):.4f} cov {np.nanmean(v[2][:, 2]):.4f}; '
                       f'mean energy-branch (grid) coord {c[1].mean():.4f} cov {v[1].mean():.4f}')
    txt = '\n'.join(out)
    print(txt)
    if a.out:
        open(a.out, 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
