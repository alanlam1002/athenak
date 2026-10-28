#!/usr/bin/env python3
"""c138 (R-150 item 3, user-approved 2026-10-28): the remnant mass-loss audit on full-3D restarts
(tde_dmde.cells, rcut 3; remnant.bound_mass for the self-bound set, its centre x_c).
  (a) the Hill sphere r_H = r (M_rem/3M)^(1/3) (r = |x_c - x_BH|, M = 1, M_rem = remnant.py's bound mass):
      the mass inside r_H, remnant.py's bound mass inside / outside r_H;
  (b) the rest-mass flux through shells |x - x_c| = R (R/r_H = 0.5, 0.75, 1, 1.5, 2; width max(0.1 R, 2 dx_min)):
      F = sum dM (V - V_c).r_hat / w with the coordinate velocity V^i = alpha v^i - beta'^i (beta' the stored
      shift, so V is the grid velocity in either gauge) and V_c the bound set's mass-weighted V; outward (F+)
      and net; the outward flux's split into the L1 cone (r_hat.n > cos 30 deg, n = the unit vector to the hole),
      the L2 cone (r_hat.n < -cos 30 deg; together 13.4% of the solid angle) and the polar caps (|r_hat_z| >
      cos 30 deg; 13.4%), the rest the equatorial band;
  (c) the remnant's entropy profile: the mass-weighted K/K0 - 1 (K0 = kappa) of the bound set in bins of the
      enclosed bound mass m/M_rem (sorted by |x - x_c|) and the warm (> +5%) / cold (< -5%) fractions.
    c138_audit.py OUT LABEL=RST:TP [...] [--nproc 2]
-> OUT/c138_audit.csv (one row per restart), OUT/c138_profile.csv (per restart and mass bin)"""
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
SHELLS = (0.5, 0.75, 1.0, 1.5, 2.0)
MBINS = np.linspace(0.0, 1.0, 11)
C30 = np.cos(np.radians(30.0))


def audit(arg):
    lab, fn, tp = arg
    c = T.cells(fn, rcut=3.0)
    Mb, xc, _, sel = RM.bound_mass(c)
    X = np.stack([c['x'], c['y'], c['z']])
    v = np.stack([c['vx'], c['vy'], c['vz']])
    beta = np.stack([c['betax'], c['betay'], c['betaz']])
    V = c['alpha'] * v - beta
    dM = c['dM']
    Vc = (V * dM * sel).sum(1) / (dM * sel).sum()
    d = X - xc[:, None]
    r = np.linalg.norm(d, axis=0)
    rhat = d / np.maximum(r, 1e-12)
    xbh = np.asarray(c['xbh'], dtype=float)
    rbh = float(np.linalg.norm(xc - xbh))
    n = (xbh - xc) / rbh
    rH = rbh * (Mb / (3.0 * MBH))**(1.0 / 3.0)
    dxmin = float(np.cbrt(c['dV'].min()))
    row = dict(label=lab, t=c['t'], t_minus_tp=c['t'] - tp, M_rem=Mb / MSTAR, r_bh=rbh, r_H=rH,
               M_inHill=dM[r < rH].sum() / MSTAR, Mb_inHill=dM[sel & (r < rH)].sum() / MSTAR,
               Mb_outHill=dM[sel & (r >= rH)].sum() / MSTAR,
               r_b90=float(np.sort(r[sel])[int(0.9 * sel.sum())]) if sel.any() else np.nan,
               Vc=float(np.linalg.norm(Vc)), dx_min=dxmin)
    vr = ((V - Vc[:, None]) * rhat).sum(0)
    cn = (rhat * n[:, None]).sum(0)
    for s in SHELLS:
        R = s * rH
        w = max(0.1 * R, 2.0 * dxmin)
        sh = np.abs(r - R) < 0.5 * w
        f = dM[sh] * vr[sh] / w
        fo = np.where(f > 0, f, 0.0)
        tot = fo.sum()
        k = f'{s:g}'
        row[f'Fout_{k}'] = tot / MSTAR
        row[f'Fnet_{k}'] = f.sum() / MSTAR
        row[f'L1_{k}'] = fo[cn[sh] > C30].sum() / tot if tot > 0 else np.nan
        row[f'L2_{k}'] = fo[cn[sh] < -C30].sum() / tot if tot > 0 else np.nan
        row[f'pole_{k}'] = fo[np.abs(rhat[2][sh]) > C30].sum() / tot if tot > 0 else np.nan
    k = c['P'][sel] / c['rho'][sel]**GAM / K0 - 1.0
    m = dM[sel]
    o = np.argsort(r[sel])
    mf = np.cumsum(m[o]) / m.sum()
    prof = []
    for a, b in zip(MBINS[:-1], MBINS[1:]):
        j = o[(mf > a) & (mf <= b)]
        if len(j) == 0:
            continue
        prof.append(dict(label=lab, t_minus_tp=c['t'] - tp, m_lo=a, m_hi=b, r_mid=float(np.median(r[sel][j])),
                         K_mean=np.average(k[j], weights=m[j]), warm5=m[j][k[j] > 0.05].sum() / m[j].sum(),
                         cold5=m[j][k[j] < -0.05].sum() / m[j].sum(),
                         rho_mid=float(np.median(c['rho'][sel][j]))))
    row['K_rem'] = np.average(k, weights=m)
    print(row, flush=True)
    return row, prof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--nproc', type=int, default=2)
    a = ap.parse_args()
    args = []
    for s in a.runs:
        lab, rest = s.split('=', 1)
        fn, tp = rest.rsplit(':', 1)
        args.append((lab, fn, float(tp)))
    with Pool(a.nproc) as p:
        res = p.map(audit, args)
    os.makedirs(a.out, exist_ok=True)
    rows = [r for r, _ in res]
    with open(f'{a.out}/c138_audit.csv', 'w') as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    prof = [q for _, p in res for q in p]
    with open(f'{a.out}/c138_profile.csv', 'w') as f:
        w = csv.DictWriter(f, list(prof[0].keys()))
        w.writeheader()
        w.writerows(prof)


if __name__ == '__main__':
    main()
