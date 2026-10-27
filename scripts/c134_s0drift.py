#!/usr/bin/env python3
"""c134 (R-147 item 4b): the pilots' approach drift with dual energy on. Over S0 (t <= t_p - 50), from the 3D
primitive dumps (every 12 M), c122_b1.dump's dense-band statistics: the mass-weighted <K/K0 - 1> over
rho >= 0.01 and >= 0.1 rho_c, the warm/cold (+-5%) mass fractions, rho_max/rho_c. Plotted against the time since
the start and against t - t_p, with B.1's B1N_cell (DE off) and B1D_cell (DE on) drift (c122_b1_dense.csv,
rp10 comoving from r0 = 30, t 42 -> 120) for scale.
    c134_s0drift.py OUT LABEL=RUNDIR:TP [...] [--b1 c122_b1_dense.csv] [--nproc 8]
"""
import argparse
import glob
import os
import re
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c122_b1 as B1   # noqa: E402

KEYS = ['t', 'd01_mean', 'd01_warm5', 'd01_cold5', 'd1_mean', 'd1_warm5', 'd1_cold5', 'rhomax', 'K_at_rhomax']
COL = {'P18_r30': 'C0', 'P18_r50': 'C1', 'P18_r82': 'C2', 'P10_r82': 'C3'}


def par(d, k):
    return float(re.search(rf'(?m)^{k}\s*=\s*([-+.\deE]+)', open(f'{d}/parfile.par').read()).group(1))


def one(arg):
    f, rhoc, k0 = arg
    try:
        return B1.dump(f, rhoc, k0)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--b1', default=None)
    ap.add_argument('--nproc', type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for spec in a.runs:
        lab, rest = spec.split('=', 1)
        d, tp = rest.split(':'); tp = float(tp)
        rhoc, k0 = par(d, 'rhoc'), par(d, 'kappa')
        fs = sorted(glob.glob(f'{d}/bin/{B1.B}.prim_3d.*.bin'))
        with Pool(a.nproc) as pool:
            res = [o for o in pool.map(one, [(f, rhoc, k0) for f in fs]) if o is not None]
        res = sorted((o for o in res if o['t'] <= tp - 50.0 + 1e-6), key=lambda o: o['t'])
        rows += [(lab, tp, o) for o in res]
        print(lab, len(res), 'dumps', flush=True)
    with open(f'{a.out}/c134_s0drift.csv', 'w') as fo:
        fo.write('pilot,t_p,' + ','.join(KEYS) + '\n')
        for lab, tp, o in rows:
            fo.write(f'{lab},{tp:g},' + ','.join(f'{o[k]:.6g}' for k in KEYS) + '\n')
    b1 = {}
    if a.b1:
        z = np.genfromtxt(a.b1, delimiter=',', names=True, dtype=None, encoding=None)
        for arm in ('B1N_cell', 'B1D_cell'):
            m = z['arm'] == arm
            if m.any():
                b1[arm] = z[m]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for col, (q, tit) in enumerate([('d01_mean', r'$\langle K/K_0-1\rangle$, $\rho \geq 0.01\rho_c$'),
                                    ('d01_cold5', r'cold fraction ($K/K_0-1 < -5\%$), $\rho \geq 0.01\rho_c$'),
                                    ('rhomax', r'$\rho_{max}/\rho_c$ (3D dumps)')]):
        for row, xlab in enumerate(['time since the start [M]', r'$t - t_p$ [M]']):
            ax = axs[row, col]
            for lab in dict.fromkeys(r[0] for r in rows):
                R = [(tp, o) for l, tp, o in rows if l == lab]
                t = np.array([o['t'] for _, o in R]); y = np.array([o[q] for _, o in R])
                ax.plot(t - (0.0 if row == 0 else R[0][0]), y, color=COL.get(lab, '0.5'), marker='.', label=lab)
            if row == 0:
                for arm, ls in (('B1N_cell', '--'), ('B1D_cell', ':')):
                    if arm in b1:
                        ax.plot(b1[arm]['t'], b1[arm][q], 'k', ls=ls, label=f'{arm} (rp10, r0 30)')
            ax.set_title(tit); ax.set_xlabel(xlab)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle('c134 S0 (to t_p - 50), dual energy on: dense-band adiabat drift (3D dumps every 12 M)')
    fig.tight_layout(); fig.savefig(f'{a.out}/c134_s0drift.png', dpi=100)


if __name__ == '__main__':
    main()
