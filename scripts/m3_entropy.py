#!/usr/bin/env python3
"""M3 (research R-034): mass-weighted entropy-error fractions from restart files (3D).

Over the dense gas (rho > 0.1 rho_c, rho_c = <problem> rhoc), the mass fractions with
K/K0 - 1 > +5%, > +10% (warm) and < -5%, < -10% (cold), K = P/rho^Gamma, K0 = kappa.
Replaces min K as the convergence metric (R-033 0b).

    m3_entropy.py LABEL=RUNDIR [LABEL=RUNDIR ...] [--rcut 0]
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tde_dmde as T   # noqa: E402

GAM = 5.0 / 3.0


def par(text, key):
    m = re.search(rf'^{key}\s*=\s*([-+.\deE]+)', text, re.M)
    return float(m.group(1))


def fractions(fn, rcut):
    c = T.cells(fn, rcut)
    rhoc, k0 = par(c['par'], 'rhoc'), par(c['par'], 'kappa')
    d = c['rho'] > 0.1 * rhoc
    m = c['dM'][d]; x = c['P'][d] / c['rho'][d]**GAM / k0 - 1.0
    M = m.sum()
    f = lambda sel: m[sel].sum() / M if M > 0 else np.nan
    return c['t'], M, f(x > 0.05), f(x > 0.10), f(x < -0.05), f(x < -0.10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--rcut', type=float, default=0.0)
    ap.add_argument('--every', type=int, default=1)
    a = ap.parse_args()
    for spec in a.runs:
        lab, d = spec.split('=', 1)
        for fn in sorted(glob.glob(f'{d}/rst/*.rst'))[::a.every]:
            t, M, w5, w10, c5, c10 = fractions(fn, a.rcut)
            print(f'{lab:10s} t={t:6.1f} dense mass {M:.4e}  warm >5% {w5:.4f} >10% {w10:.4f}  '
                  f'cold <-5% {c5:.4f} <-10% {c10:.4f}', flush=True)


if __name__ == '__main__':
    main()
