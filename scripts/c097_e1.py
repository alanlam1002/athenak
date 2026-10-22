#!/usr/bin/env python3
"""C-097 (R-103 E1): false-flag census of the dual-energy shock flag, coord vs covariant divergence gate.

On fixed-binary (>= 30c52c1d) restarts, dense cells (D > dual_energy_rho_switch) of the z = 0 layer:
  theta = W (D_i v^i + v^i d_i ln alpha) + (W/alpha)(d_t + V^i d_i) ln W - W K   (c050_theta.thetas, 'dt1'),
  theta_cont = -(W/alpha)(d_t + V^i d_i) ln rho   (c050_theta.theta_cont; independent, from continuity),
  the flag computed BOTH ways on the same state by the C-049 3D replica of de_shock_flag (c049_divflag.block:
  vcurv sensor, div <= 0 gate with d_i v^i (coord) or + v^i d_i ln sqrt(gamma) (covariant), weak/dilation),
  so each restart gives both forms' flags whatever the run's own setting (the run's own form is printed).
Per run and phase (pre-periapsis / periapsis / post; t_p from --tp): among FLAGGED dense cells of each form,
the fraction with theta > 0 (expanding: false flags) and theta < 0; the same restricted to cells where theta's
sign is robust: sign(theta) == sign(theta_cont) and |theta| > the restart's noise level, taken as C-050's
median |theta - theta_cont| over the unflagged bulk star (rho > 0.1 rho_max, sensor ratio <= 1).

    c097_e1.py LABEL:RUNDIR:TP [...] [--first N] [--phase-half 12] [--nproc N] [--out FILE]
"""
import argparse
import glob
import os
import re
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tde_dmde as D                              # noqa: E402
import c049_divflag as F                          # noqa: E402
import c050_theta as T                            # noqa: E402

BASE = 'cfc_tde_wd_imbh_parabolic_tracker'


def one(job):
    lab, fn = job
    t = D.header(fn)
    p = F.deck(t['par'])
    d = os.path.dirname(os.path.dirname(fn))
    fs = sorted(glob.glob(f'{d}/bin/{BASE}.prim_xy.*.bin'))
    nums = np.array([int(f.split('.')[-2]) for f in fs])
    tt = t['time']
    nb = int(nums[nums <= tt + 1e-9].max()) if (nums <= tt + 1e-9).any() else int(nums.min())
    na = nb + 1 if nb + 1 <= nums.max() else nb - 1
    pr = tuple(sorted((nb, na)))
    S = {n: T.slice_lnW(d, n) for n in pr}
    mm = np.memmap(fn, dtype=np.float64, mode='r', offset=t['data_off'], shape=(t['nmb'], t['dsize'] // 8))
    x1min, x2min, x3min, x1max, x2max, x3max = t['size'][:6]
    w = np.array([x1max - x1min, x2max - x2min, x3max - x3min]); lo = np.array([x1min, x2min, x3min])
    lev = t['ll'][:, 3]
    cols = {k: [] for k in ('rho', 'W', 'alpha', 'K', 'divc', 'cov', 'lap', 'advW', 'advR', 'ratio', 'dt1', 'rt1',
                            'fc', 'fv')}
    for b in range(t['nmb']):
        ext = w / 2.0**lev[b]
        bmin = lo + t['ll'][b, :3] * ext
        if abs(bmin[2]) > 1e-9:
            continue
        o = T.block(mm[b], 1, ext[0] / 16, p)
        if o is None:
            continue
        f = F.block(mm[b], 1, ext[0] / 16, p)
        dn = o['dense']
        key = (round(bmin[0], 6), round(bmin[1], 6), round(bmin[0] + ext[0], 6), round(bmin[1] + ext[1], 6))
        if all(key in S[n][0] for n in pr):
            dts = [(S[pr[1]][0][key][q] - S[pr[0]][0][key][q]) / (S[pr[1]][1] - S[pr[0]][1]) for q in (0, 1)]
        else:
            dts = [np.full((16, 16), np.nan)] * 2
        for kk in cols:
            if kk == 'dt1':
                cols[kk].append(dts[0][dn])
            elif kk == 'rt1':
                cols[kk].append(dts[1][dn])
            elif kk == 'fc':
                cols[kk].append(f['f_coord'][0][dn])
            elif kk == 'fv':
                cols[kk].append(f['f_cov'][0][dn])
            else:
                cols[kk].append(o[kk][dn])
    c = {k: np.concatenate(v) if v else np.zeros(0) for k, v in cols.items()}
    m = re.search(r'^\s*dual_energy_div\s*=\s*(\w+)', t['par'], re.M)
    form = m.group(1) if m else 'coord (default)'
    return lab, tt, form, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', help='LABEL:RUNDIR:T_PERIAPSIS')
    ap.add_argument('--first', type=int, default=0, help='skip the first N restarts of each run')
    ap.add_argument('--phase-half', type=float, default=12.0)
    ap.add_argument('--nproc', type=int, default=8)
    ap.add_argument('--out')
    a = ap.parse_args()
    jobs, tps = [], {}
    for spec in a.runs:
        lab, d, tp = spec.split(':')
        tps[lab] = float(tp)
        jobs += [(lab, f) for f in sorted(glob.glob(f'{d}/rst/*.rst'))[a.first:]]
    with Pool(a.nproc) as pool:
        res = pool.map(one, jobs)
    rows = ['C-097 E1: flagged dense cells (z = 0 layer), both gate forms computed on the same state (C-049 replica); '
            'theta (C-050, dt1) and theta_cont; robust = sign(theta) == sign(theta_cont) and |theta| > noise.']
    agg = {}
    for lab, tt, form, c in res:
        good = np.isfinite(c['dt1'])
        th, _ = T.thetas(c, 'dt1')
        tcn = T.theta_cont(c)
        bulk = good & (c['rho'] > 0.1 * c['rho'].max()) & (c['ratio'] <= 1)
        noise = np.median(np.abs(th[bulk] - tcn[bulk])) if bulk.any() else np.inf
        rob = good & ((th <= 0) == (tcn <= 0)) & (np.abs(th) > noise)
        tp = tps[lab]
        ph = 'pre' if tt < tp - a.phase_half else ('peri' if tt <= tp + a.phase_half else 'post')
        rows.append(f'{lab} t {tt:7.2f} [{ph}] run form {form}: dense {good.sum()} (no slice {(~good).sum()}), noise '
                    f'{noise:.2e}, robust {rob.sum()}; flagged coord {(c["fc"] & good).sum()} cov {(c["fv"] & good).sum()}')
        for fm, fl in (('coord', c['fc']), ('covariant', c['fv'])):
            z = agg.setdefault((lab, ph, fm), np.zeros(6))
            m = fl & good
            z += [m.sum(), (m & (th > 0)).sum(), (m & (th < 0)).sum(),
                  (m & rob).sum(), (m & rob & (th > 0)).sum(), (m & rob & (th < 0)).sum()]
    rows.append('')
    rows.append('run  phase  form       flagged  theta>0  theta<0 | robust  theta>0  theta<0   (fractions of flagged)')
    for (lab, ph, fm), z in sorted(agg.items(), key=lambda kv: (kv[0][0], ['pre', 'peri', 'post'].index(kv[0][1]), kv[0][2])):
        n, nr = max(z[0], 1), max(z[3], 1)
        rows.append(f'{lab:5s} {ph:5s} {fm:10s} {int(z[0]):7d}  {z[1] / n:.3f}    {z[2] / n:.3f}  | {int(z[3]):6d}  '
                    f'{z[4] / nr:.3f}    {z[5] / nr:.3f}')
    txt = '\n'.join(rows)
    print(txt)
    if a.out:
        open(a.out, 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
