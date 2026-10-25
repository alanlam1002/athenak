#!/usr/bin/env python3
"""D2 (R-140 item 2, c123): the +-64 vs +-128 box check. Two arms of the same rp10 comoving production deck
(c110_2x2/CD and c123_box/P128 = box x2, root dx x2, +1 level), compared over their common window [T0, TMAX]:
  cost:   cycles, mean dt, wall per M (out.log elapsed), node-h per M (NODES nodes), the PHASE_TIMERS split
          (mg / hydro_etc / regrid / io per cycle) and nmb; MeshBlocks per physical level (3D dumps' block sizes)
  orbit:  rho_max (user hst), star-BH separation (z = 0 density maximum minus x_BH, prim_xy), alpha_min
  metric: alpha and psi = (gxx)^(1/4) at the star (z = 0 density maximum) and at points 3, 5, 10, 20 M from
          x_BH (adm_xy, nearest cell of the finest block containing the point); the CFC ADM-mass surface
          integrals M_res at r = 20, 30, 40 (out.log, per cycle); MCLOSE/ECLOSE closure lines
    c123_box.py OUTDIR A=RUNDIR B=RUNDIR [--t0 2 --tmax 80 --nodes 8]
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c110_analysis as A          # noqa: E402
from tde_ensemble_plots import bc  # noqa: E402

B = A.B
OFFS = (3.0, 5.0, 10.0, 20.0)


def logs(d):
    """per cycle (t, wall) and ADM M_res(r); PHASE_TIMERS; closure lines"""
    t = None
    T, W, adm, pt, clos = [], [], {}, [], []
    w0 = 0.0
    for l in open(f'{d}/out.log', errors='replace'):
        g = re.match(r'elapsed=(\S+) cycle=(\d+) time=(\S+)', l)
        if g:
            w = float(g.group(1))
            if W and w < W[-1] - w0:
                w0 = W[-1]
            t = float(g.group(3))
            T.append(t); W.append(w + w0)
            continue
        g = re.match(r'### CFC ADM mass: r=(\S+) M_res=(\S+)', l)
        if g and t is not None:
            adm.setdefault(float(g.group(1)), []).append((t, float(g.group(2))))
            continue
        if l.startswith('PHASE_TIMERS cycle='):
            pt.append(dict(kv.split('=') for kv in l.split()[1:]))
            continue
        if re.match(r'(MCLOSE|ECLOSE|SCLOSE[123]) cycles=', l):
            clos.append(l.strip())
    return np.array(T), np.array(W), {r: np.array(v) for r, v in adm.items()}, pt, clos


def levels(fn):
    p = bc.read_binary(fn)
    g = np.asarray(p['mb_geometry'])
    w = g[:, 1] - g[:, 0]
    return float(p['time']), {round(float(x), 6): int(n) for x, n in zip(*np.unique(w, return_counts=True))}


def sample(p, pts):
    """nearest-cell values of alpha and psi at the (x, y) points, from the finest block containing each"""
    g = np.asarray(p['mb_geometry'])
    nx = p['nx1_out_mb']
    al = np.asarray(p['mb_data']['adm_alpha'])[:, 0]
    gx = np.asarray(p['mb_data']['adm_gxx'])[:, 0]
    out = []
    for x, y in pts:
        m = np.where((g[:, 0] <= x) & (x < g[:, 1]) & (g[:, 2] <= y) & (y < g[:, 3]))[0]
        if len(m) == 0:
            out.append((np.nan, np.nan)); continue
        m = m[np.argmin(g[m, 1] - g[m, 0])]
        h = (g[m, 1] - g[m, 0]) / nx
        i = min(int((x - g[m, 0]) / h), nx - 1); j = min(int((y - g[m, 2]) / h), nx - 1)
        out.append((al[m, j, i], gx[m, j, i]**0.25))
    return out


def metric(arg):
    d, fa, fp = arg
    try:
        p = bc.read_binary(fa)
        t, x, y, _ = A.dump(fp)
    except Exception:
        return None
    bh = A.bh_xy(d, np.array([t]))[0]
    pts = [(x, y)] + [(bh[0] + r, bh[1]) for r in OFFS]
    return (t, *[v for s in sample(p, pts) for v in s])


def arm(lab, d, nproc):
    s = A.series(lab, d, nproc)
    T, W, adm, pt, clos = logs(d)
    fa = sorted(glob.glob(f'{d}/bin/{B}.adm_xy.*.bin'))
    jobs = [(d, f, f.replace('adm_xy', 'prim_xy')) for f in fa if os.path.exists(f.replace('adm_xy', 'prim_xy'))]
    with Pool(nproc) as pool:
        M = [r for r in pool.map(metric, jobs) if r is not None]
        L = pool.map(levels, sorted(glob.glob(f'{d}/bin/{B}.prim_3d.*.bin')))
    return dict(s=s, T=T, W=W, adm=adm, pt=pt, clos=clos, M=np.array(sorted(M)), L=sorted(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('runs', nargs=2)
    ap.add_argument('--t0', type=float, default=2.0)
    ap.add_argument('--tmax', type=float, default=80.0)
    ap.add_argument('--nodes', type=int, default=8)
    ap.add_argument('--nproc', type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = {lab: arm(lab, d, a.nproc) for lab, d in (r.split('=', 1) for r in a.runs)}
    labs = list(R)
    tmax = min(a.tmax, min(r['T'].max() for r in R.values()))
    rows = [f'# window t = {a.t0} -> {tmax:.3f}', 'quantity,' + ','.join(labs) + ',ratio_B/A']

    def add(name, v):
        rows.append(f'{name},' + ','.join(f'{x:.6g}' for x in v) + f',{v[1] / v[0]:.4g}')
    # cost
    c, w, dt, nmb, ph = [], [], [], [], {k: [] for k in ('mg', 'hydro_etc', 'regrid', 'io')}
    for r in R.values():
        k = (r['T'] >= a.t0) & (r['T'] <= tmax)
        c.append(k.sum()); w.append(np.interp(tmax, r['T'], r['W']) - np.interp(a.t0, r['T'], r['W']))
        dt.append((tmax - a.t0) / k.sum())
        P = [p for p in r['pt'] if a.t0 <= float(p['time']) <= tmax]
        p0, p1 = P[0], P[-1]
        ncyc = float(p1['cycles']) - float(p0['cycles'])
        for q in ph:
            ph[q].append((float(p1[q]) - float(p0[q])) / ncyc)
        nmb.append(np.mean([float(p['nmb']) for p in P]))
    add('cycles', c); add('mean_dt', dt); add('wall_s', w)
    add('wall_s_per_M', [x / (tmax - a.t0) for x in w])
    add('node_h_per_M', [x / (tmax - a.t0) * a.nodes / 3600 for x in w])
    add('wall_s_per_cycle', [x / y for x, y in zip(w, c)])
    for q in ph:
        add(f'{q}_s_per_cycle', ph[q])
    add('nmb_mean', nmb)
    # orbit and metric at matched times
    for tt in np.arange(10.0, tmax + 1e-9, 10.0):
        v = [np.interp(tt, r['s']['t_u'], r['s']['rhomax']) / A.RHOC for r in R.values()]; add(f'rhomax_t{tt:g}', v)
        v = [np.interp(tt, r['s']['t_dump'], r['s']['sep']) for r in R.values()]; add(f'sep_t{tt:g}', v)
        v = [np.interp(tt, r['s']['t_u'], r['s']['amin']) for r in R.values()]; add(f'alpha_min_t{tt:g}', v)
        for j, nm in enumerate(['star'] + [f'bh+{x:g}' for x in OFFS]):
            for q, o in (('alpha', 1), ('psi', 2)):
                v = [np.interp(tt, r['M'][:, 0], r['M'][:, 1 + 2 * j + (o - 1)]) for r in R.values()]
                add(f'{q}_{nm}_t{tt:g}', v)
        for rr in (20.0, 30.0, 40.0):
            v = [np.interp(tt, r['adm'][rr][:, 0], r['adm'][rr][:, 1]) for r in R.values()]
            add(f'ADM_Mres_r{rr:g}_t{tt:g}', v)
    open(f'{a.out}/c123_compare.csv', 'w').write('\n'.join(rows) + '\n')
    # blocks per level
    lr = ['label,t,' + 'dx_block_width:count ...']
    for lab, r in R.items():
        for t, d in r['L']:
            if t <= tmax + 1e-6:
                lr.append(f'{lab},{t:g},' + ' '.join(f'{k:g}:{n}' for k, n in sorted(d.items(), reverse=True)))
    open(f'{a.out}/c123_levels.csv', 'w').write('\n'.join(lr) + '\n')
    open(f'{a.out}/c123_closure.txt', 'w').write('\n'.join(f'{lab}: {l}' for lab, r in R.items() for l in r['clos']) + '\n')
    # figure
    fig, ax = plt.subplots(2, 3, figsize=(14, 7.5))
    for lab, r in R.items():
        s = r['s']
        ax[0, 0].plot(s['t_u'], s['rhomax'] / A.RHOC, label=lab)
        ax[0, 1].plot(s['t_dump'], s['sep'], label=lab)
        ax[0, 2].plot(s['t_u'], s['amin'], label=lab)
        ax[1, 0].plot(r['M'][:, 0], r['M'][:, 1], label=f'{lab} star')
        for j, x in enumerate(OFFS):
            ax[1, 1].plot(r['M'][:, 0], r['M'][:, 3 + 2 * j], label=f'{lab} bh+{x:g}')
        for rr in (20.0, 30.0, 40.0):
            ax[1, 2].plot(r['adm'][rr][:, 0], r['adm'][rr][:, 1] / 7.47e-5, label=f'{lab} r={rr:g}')
    for x, tl in zip(ax.flat, ['rho_max/rho_c', 'star-BH separation [M] (coord)', 'alpha_min',
                               'alpha at the star', 'alpha at x_BH + (r, 0)', 'ADM M_res / M*']):
        x.set_title(tl); x.set_xlim(0, tmax); x.legend(fontsize=6); x.set_xlabel('t [M]')
    fig.tight_layout(); fig.savefig(f'{a.out}/c123_compare.png', dpi=90)
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
