#!/usr/bin/env python3
"""R-127 Q5 (c114): where alpha_min sits. Per adm_xy dump (z = 0 slice): the minimum lapse over the
slice's cells, its position, its distance from the BH (puncture_x - frame_disp from the user hst),
the dynamic excision radius r_f at that time (EXCDYN lines of out.log) and the user-hst alpha-min.
Also the minimum lapse over cells with r > r_f (the evolved region).
    c114_alpha_min.py OUT.csv LABEL=RUNDIR [...] [--tmin T]
"""
import argparse
import glob
import re
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tde_ensemble_plots import bc            # noqa: E402
from c110_analysis import par_get, hst, bh_xy, B   # noqa: E402


def excdyn(d):
    T, R = [], []
    for l in open(f'{d}/out.log', errors='replace'):
        g = re.match(r'EXCDYN cycle=\d+ time=(\S+) r_f=(\S+)', l)
        if g:
            T.append(float(g.group(1))); R.append(float(g.group(2)))
    T, R = np.array(T), np.array(R)
    o = np.argsort(T, kind='stable')
    return T[o], R[o]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--tmin', type=float, default=150.0)
    a = ap.parse_args()
    rows = ['label,t,alpha_min_slice,x,y,r_from_bh,r_f,alpha_min_hst,alpha_min_r_gt_rf,r_of_that,dx_at_min']
    for spec in a.runs:
        lab, d = spec.split('=', 1)
        te, rf = excdyn(d)
        n, uh = hst(d, 'user')
        for fn in sorted(glob.glob(f'{d}/bin/{B}.adm_xy.*.bin')):
            p = bc.read_binary(fn)
            t = p['time']
            if t < a.tmin:
                continue
            g = np.asarray(p['mb_geometry'])
            nx, ny = p['nx1_out_mb'], p['nx2_out_mb']
            al = np.asarray(p['mb_data']['adm_alpha'])[:, 0]
            bh = bh_xy(d, np.array([t]))[0]
            xs = g[:, 0, None] + (np.arange(nx)[None, :] + 0.5) * ((g[:, 1] - g[:, 0]) / nx)[:, None]
            ys = g[:, 2, None] + (np.arange(ny)[None, :] + 0.5) * ((g[:, 3] - g[:, 2]) / ny)[:, None]
            X = np.broadcast_to(xs[:, None, :], al.shape)
            Y = np.broadcast_to(ys[:, :, None], al.shape)
            r = np.hypot(X - bh[0], Y - bh[1])
            m = np.unravel_index(np.argmin(al), al.shape)
            r_f = np.interp(t, te, rf) if len(te) else np.nan
            ev = r > r_f
            m2 = np.unravel_index(np.argmin(np.where(ev, al, np.inf)), al.shape)
            ah = np.interp(t, uh[:, 0], uh[:, n.index('alpha-min')])
            rows.append(f'{lab},{t:.2f},{al[m]:.5f},{X[m]:.4f},{Y[m]:.4f},{r[m]:.4f},{r_f:.4f},{ah:.5f},'
                        f'{al[m2]:.5f},{r[m2]:.4f},{(g[m[0], 1] - g[m[0], 0]) / nx:.4f}')
    open(a.out, 'w').write('\n'.join(rows) + '\n')
    print('\n'.join(rows[:3]), '...', len(rows) - 1)


if __name__ == '__main__':
    main()
