#!/usr/bin/env python3
"""c151 (R-170 item 2(i), user-approved 2026-10-30): the Hill-bound remnant's entropy profile in its
enclosed-mass coordinate, from full-3D restarts (tde_dmde.cells, rcut 3; remnant.bound_mass for the centre
x_c; the Hill radius r_H = r_bh (M_rem/3M)^(1/3) of c142_remnant.py).
  Cells with |x - x_c| < r_H, sorted by |x - x_c| (spherical shells about x_c); m = the cumulative mass from the
  centre, in M* (MSTAR as c142); K/K0 = (P/rho^Gamma)/kappa per cell. Per bin of m (fixed edges in M*, so the
  profiles of different times compare at the same m): the mass-weighted <K/K0>, and the bin's mean radius.
  Also, per pair of consecutive restarts (t0 < t1, M1 = M_Hill(t1) < M0), the change of the Hill-mean K split as
      Kbar1 - Kbar0 = [<K1>_[0,M1] - <K0>_[0,M1]]   (heating/cooling at fixed m)
                    + [<K0>_[0,M1] - <K0>_[0,M0]]   (removing the shells M1 < m < M0 at t0's K(m))
    c151_km.py OUT LABEL=RST:TP [...] [--nproc 3]   -> OUT/c151_km.csv (per restart and bin), OUT/c151_split.csv"""
import argparse
import csv
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import remnant as RM   # noqa: E402
import tde_dmde as T   # noqa: E402

GAM = 5.0 / 3.0
K0 = 1.2346626297e-2
MSTAR = 7.47e-5
MBH = 1.0
MEDGES = np.round(np.arange(0.0, 0.5001, 0.01), 4)


def one(arg):
    lab, fn, tp = arg
    c = T.cells(fn, rcut=3.0)
    Mb, xc, _, _ = RM.bound_mass(c)
    X = np.stack([c['x'], c['y'], c['z']])
    r = np.linalg.norm(X - xc[:, None], axis=0)
    rbh = float(np.linalg.norm(xc - np.asarray(c['xbh'], dtype=float)))
    rH = rbh * (Mb / (3.0 * MBH))**(1.0 / 3.0)
    s = r < rH
    o = np.argsort(r[s])
    dM = c['dM'][s][o] / MSTAR
    k = (c['P'][s][o] / c['rho'][s][o]**GAM) / K0
    rr = r[s][o]
    m = np.cumsum(dM) - 0.5 * dM          # the cell's own mid-mass
    return dict(label=lab, t=c['t'], tp=tp, MH=float(dM.sum()), rH=rH, m=m, dM=dM, k=k, r=rr)


def binned(d):
    out = []
    for a, b in zip(MEDGES[:-1], MEDGES[1:]):
        w = (d['m'] >= a) & (d['m'] < b)
        if not w.any():
            continue
        out.append(dict(label=d['label'], t_minus_tp=d['t'] - d['tp'], m_lo=a, m_hi=b,
                        mass=d['dM'][w].sum(), K=np.average(d['k'][w], weights=d['dM'][w]),
                        r_mean=np.average(d['r'][w], weights=d['dM'][w]),
                        full=bool(d['MH'] >= b)))
    return out


def kmean(d, mmax):
    w = d['m'] < mmax
    return np.average(d['k'][w], weights=d['dM'][w])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--nproc', type=int, default=3)
    a = ap.parse_args()
    args = []
    for spec in a.runs:
        lab, rest = spec.split('=', 1)
        fn, tp = rest.rsplit(':', 1)
        args.append((lab, fn, float(tp)))
    with Pool(a.nproc) as p:
        D = sorted(p.map(one, args), key=lambda d: d['t'])
    os.makedirs(a.out, exist_ok=True)
    rows = [r for d in D for r in binned(d)]
    with open(f'{a.out}/c151_km.csv', 'w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    sp = []
    for d0, d1 in zip(D[:-1], D[1:]):
        M0, M1 = d0['MH'], d1['MH']
        heat = kmean(d1, M1) - kmean(d0, M1)
        strip = kmean(d0, M1) - kmean(d0, M0)
        sp.append(dict(t0=d0['t'] - d0['tp'], t1=d1['t'] - d1['tp'], M0=M0, M1=M1, Kbar0=kmean(d0, M0),
                       Kbar1=kmean(d1, M1), heat_fixed_m=heat, strip=strip))
        print(sp[-1])
    with open(f'{a.out}/c151_split.csv', 'w') as f:
        w = csv.DictWriter(f, fieldnames=list(sp[0]))
        w.writeheader()
        w.writerows(sp)


if __name__ == '__main__':
    main()
