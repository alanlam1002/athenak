#!/usr/bin/env python3
"""C-044 (R-044 §4 items 3, 4, 7 and M3): restart and slice time series of the item 6/7 reruns
against their BH-frame baselines.

  R6 (comoving rp10, si = 0)  vs  Q1c = D_e (t <= 80) + RQ1 (corrected fill, BH frame)
  R7 (comoving rp18, si = 0)  vs  Q3 (rp18, BH frame, si = 1, old fill)

Per restart (tde_dmde.cells, frame-aware; r_BH < 3 excluded): M3 warm > 5% and > 10% of the
dense mass (rho > 0.1 rho_c), self-bound remnant / M* (remnant.py), bound fraction (E < 0),
mass-weighted BH-frame Killing E = -u_t - 1 and L_z about x_BH (as t6_item6.orbit).
Per prim_xy dump: star-BH coordinate distance |x(rho_max) - x_BH(t)| (x_BH from the user
history in the comoving gauge; the origin-centred puncture otherwise), its minimum and time.

    c044_series.py [--nproc 16] [--out DIR]
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tde_dmde as D                              # noqa: E402
import remnant as RM                              # noqa: E402
from tde_ensemble_plots import bc                 # noqa: E402
from anim_global_local import bh_track            # noqa: E402

C = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc'
B = 'cfc_tde_wd_imbh_parabolic_tracker'
RHOC, K0, MSTAR, GAM = 3.24e-4, 1.2346626297e-2, 7.47e-5, 5.0 / 3.0
RUNS = {'R6': ('R6_rp10_comov_L4_fix',), 'Q1c': ('D_e_Q1deck_kfix_t80', 'RQ1_bh_L4_t300_from_De'),
        'R7': ('R7_rp18_comov_L4_fix',), 'Q3': ('Q3_P2_rp18_L4',)}


def per_rst(f):
    c = D.cells(f, rcut=3.0)
    dense = c['rho'] > 0.1 * RHOC
    m = c['dM'][dense]
    x = c['P'][dense] / c['rho'][dense]**GAM / K0 - 1
    dM, xb = c['dM'], c['xbh']
    L = (c['x'] - xb[0]) * c['uy'] - (c['y'] - xb[1]) * c['ux']
    return (c['t'], m[x > 0.05].sum() / m.sum(), m[x > 0.10].sum() / m.sum(),
            RM.bound_mass(c)[0] / MSTAR, dM[c['E'] < 0].sum() / dM.sum(),
            (c['E'] * dM).sum() / dM.sum(), (L * dM).sum() / dM.sum())


def per_dump(f):
    q = bc.read_binary(f)
    g = np.asarray(q['mb_geometry']); nx = q['nx1_out_mb']
    rho = np.asarray(q['mb_data']['dens'])[:, 0]
    m, j, i = np.unravel_index(np.argmax(rho), rho.shape)
    h = (g[m, 1] - g[m, 0]) / nx
    return q['time'], g[m, 0] + (i + .5) * h, g[m, 2] + (j + .5) * h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nproc', type=int, default=16)
    ap.add_argument('--out', default=f'{C}/plots/phase2_m1')
    a = ap.parse_args()
    rst, dmp = {}, {}
    for lab, segs in RUNS.items():
        byt = {}
        for s in segs:                              # later segments win on duplicate times
            for f in sorted(glob.glob(f'{C}/{s}/rst/*.rst')):
                byt[os.path.basename(f) + s] = f
        rst[lab] = sorted(byt.values())
        dmp[lab] = [f for s in segs for f in sorted(glob.glob(f'{C}/{s}/bin/{B}.prim_xy.*.bin'))]
    jobs = [f for v in rst.values() for f in v]
    with Pool(a.nproc) as p:
        R = dict(zip(jobs, p.map(per_rst, jobs)))
        Dd = {lab: p.map(per_dump, v) for lab, v in dmp.items()}
    rows = []
    for lab, segs in RUNS.items():
        a_ = np.array(sorted({round(r[0], 2): r for r in (R[f] for f in rst[lab])}.values()))
        np.savetxt(f'{a.out}/c044_series_{lab}.txt', a_, fmt='%.6g',
                   header='t, M3 warm>5%, warm>10%, remnant/M*, bound fraction, <E>, <L_z>')
        tr = [bh_track(f'{C}/{s}') for s in segs]
        th = np.concatenate([t for t, _ in tr]); xh = np.concatenate([x for _, x in tr])
        o = np.argsort(th, kind='stable'); th, xh = th[o], xh[o]
        dd = np.array(sorted(Dd[lab]))
        r = np.hypot(dd[:, 1] - np.interp(dd[:, 0], th, xh[:, 0]), dd[:, 2] - np.interp(dd[:, 0], th, xh[:, 1]))
        k = int(np.argmin(r))
        rows.append(f'== {lab} ({", ".join(segs)}): star-BH coordinate distance min {r[k]:.2f} M at t = {dd[k, 0]:.1f}')
        for row in a_:
            rows.append(f'  t={row[0]:6.1f}  M3 warm>5% {row[1]:.3f} (>10% {row[2]:.3f})  remnant {row[3]:.4f} M*  '
                        f'bound {row[4]:.4f}  <E> {row[5]:+.4e}  <L> {row[6]:.4f}')
    with open(f'{a.out}/c044_series.txt', 'w') as f:
        f.write('\n'.join(rows) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
