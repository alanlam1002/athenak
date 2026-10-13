#!/usr/bin/env python3
"""C-044 (R-044 §4): the item 6/7 reruns with the metric solved every RK stage (si = 0),
against their BH-frame baselines at matched restart times.

  R6 (comoving rp10, L4, si = 0)  vs  RQ1 (Q1 deck, corrected fill, BH frame), t = 195.2, 200
  R7 (comoving rp18, L4, si = 0)  vs  Q3  (rp18 L4, BH frame, si = 1),         t = 300

Per restart (full 3D, scripts/tde_dmde.py cells; comoving: BH-frame Killing energy, r_BH from
x_BH(t)): debris bound fraction (E < 0 and Bernoulli), mass outside r_BH = 3, the
self-bound remnant (scripts/remnant.py) with its BH distance, its Eulerian speed |v_c|
(NOT the grid speed: the controller's is the history's fc_vg columns) and entropy K/K0 - 1.
Per prim_xy dump: dense mean K/K0 (rho > 0.1 rho0, z = 0) and K at rho_max.

    c044_checks.py [--out DIR]
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tde_t11_surface_diag as T                  # noqa: E402
import tde_dmde as D                              # noqa: E402
import remnant as RM                              # noqa: E402
import rst_tree as RT                             # noqa: E402

C = T.RUN
B = 'cfc_tde_wd_imbh_parabolic_tracker'
MSTAR = 7.47e-5
K0 = 1.2346626297e-2
DE = 0.677 / 16.4**2
PAIRS = (('R6', 'R6_rp10_comov_L4_fix,R6x_rp10_comov_L4_t300', 'RQ1', 'RQ1_bh_L4_t300_from_De', (195.2, 300.0)),
         ('R7', 'R7_rp18_comov_L4_fix', 'Q3', 'Q3_P2_rp18_L4', (300.0,)))
KSEG = {'RQ1': ('D_e_Q1deck_kfix_t80', 'RQ1_bh_L4_t300_from_De'),
        'R6': ('R6_rp10_comov_L4_fix', 'R6x_rp10_comov_L4_t300')}   # BH-frame Q1 = D_e + RQ1


def rst_at(d, t):
    fs = sorted(f for s in d.split(',') for f in glob.glob(f'{C}/{s}/rst/*.rst'))   # d: segment[,segment]
    ts = np.array([RT.read_tree(f)['time'] for f in fs])
    i = int(np.argmin(np.abs(ts - t)))
    return fs[i] if abs(ts[i] - t) < 0.5 else None


def state(fn):
    c = D.cells(fn, rcut=3.0)
    E, EB, dM = c['E'], c['EB'], c['dM']
    Mb, xc, vc, sel = RM.bound_mass(c)
    xbh = np.asarray(c['xbh']) if c.get('xbh') is not None else np.zeros(3)
    k = c['P'][sel] / c['rho'][sel]**RM.GAM / K0 - 1
    m = c['dM'][sel]
    return (f't={c["t"]:6.1f}  M(r>3)/M* {dM.sum() / MSTAR:.4f}  bound {dM[E < 0].sum() / dM.sum():.4f} '
            f'(Bernoulli {dM[EB < 0].sum() / dM.sum():.4f})  |E|<0.25dE {dM[np.abs(E / DE) < 0.25].sum() / dM.sum():.4f} | '
            f'remnant {Mb / MSTAR:.4f} M* at r_BH {np.linalg.norm((xc - xbh)[:2]):.1f}, Eulerian |v_c| '
            f'{np.linalg.norm(vc):.3f}, K/K0-1 mean {np.average(k, weights=m):+.4f}, '
            f'warm>5% {m[k > 0.05].sum() / m.sum():.3f}')


def kseries(lab, d):
    best = {}
    for seg in KSEG.get(lab, (d,)):
        T.RUNS[f'C44_{seg}'] = (f'{C}/{seg}', B, 10, 'k', '-')
        s = T.series(f'C44_{seg}', f'{C}/plots/dual_energy')
        for i, t in enumerate(s['t']):
            best[round(float(t), 3)] = (s['Kw01'][i], s['Kmax'][i])
    t = np.array(sorted(best))
    v = np.array([best[x] for x in t])
    return t, v[:, 0], v[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{C}/plots/phase2_m1')
    a = ap.parse_args()
    rows = []
    for la, da, lb, db, times in PAIRS:
        rows.append(f'== {la} ({da}) vs {lb} ({db})')
        for lab, d in ((la, da), (lb, db)):
            t, kw, km = kseries(lab, d)
            pick = [p for p in (60, 70, 80, 100, 120, 140, 160, 180, 200, 250, 300) if t.max() >= p - 0.5]
            rows.append(f'  {lab:4s} dense K/K0 | K at rho_max: ' + '; '.join(
                f't={p}: {np.interp(p, t, kw):.3f} | {np.interp(p, t, km):.3f}' for p in pick))
            print(rows[-1], flush=True)
        for tt in times:
            for lab, d in ((la, da), (lb, db)):
                f = rst_at(d, tt)
                rows.append(f'  {lab:4s} ' + (state(f) if f else f'no restart at t = {tt}'))
                print(rows[-1], flush=True)
    with open(f'{a.out}/c044_checks.txt', 'w') as f:
        f.write('\n'.join(rows) + '\n')


if __name__ == '__main__':
    main()
