#!/usr/bin/env python3
"""H (research R-030/R-031): which cell population sets dt in a TDE run.

From the z = 0 slices (prim_xy + adm_xy, same MeshBlocks), per cell and direction a:
    rate_a = (|alpha v^a - beta^a| + alpha c_s sqrt(gamma^aa)) / dx_a,
with v^a = u^a/W, W = sqrt(1 + gamma_ij u^i u^j), gamma^aa ~ 1/gamma_aa (conformally flat),
c_s^2 = Gamma P/(rho h). An approximation to NewTimeStep's eigenvalues (no relativistic
velocity composition), good enough to identify the limiting population. dt ~ cfl / max rate.

For the slice maximum and the top-N cells: distance to the BH (origin) and to the star (the
density maximum), rho / rho_0, rho / dfloor, |V| and c_s, alpha, |beta|. Populations:
  near-BH      r_BH < 4 M
  star/stream  rho > 1e-3 rho_0
  bow/env      otherwise, within 4 R* of the star (shocked near-floor gas around the star)
  atmosphere   otherwise

    tde_dt_cells.py [label ...] [--times ...] [--top N]
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tde_t11_surface_diag as T          # noqa: E402
from tde_ensemble_plots import bc          # noqa: E402

GAM = 5.0 / 3.0
RUN = T.RUN
DIRS = {'V4F': f'{RUN}/de_V4F_rp16_L4', 'V4': f'{RUN}/de_V4_rp16_L4'}
DFLOOR = 1.0e-24


def dump(d, kind, t):
    fs = sorted(glob.glob(f'{d}/bin/*{kind}*.bin'))
    return min(fs, key=lambda f: abs(int(f.split('.')[-2]) - t)) if fs else None


def rates(d, t):
    p = bc.read_binary(dump(d, 'prim_xy', t))
    q = bc.read_binary(dump(d, 'adm_xy', t))
    md = {k: np.asarray(v)[:, 0] for k, v in p['mb_data'].items()}
    ad = {k: np.asarray(v)[:, 0] for k, v in q['mb_data'].items()}
    g = np.asarray(p['mb_geometry'])
    nx = p['nx1_out_mb']
    rho, P = md['dens'], md['press']
    u = [md['velx'], md['vely'], md['velz']]
    gd = [ad['adm_gxx'], ad['adm_gyy'], ad['adm_gzz']]
    W = np.sqrt(1 + sum(gd[a] * u[a]**2 for a in range(3)))
    alp = ad['adm_alpha']
    beta = [ad['adm_betax'], ad['adm_betay'], ad['adm_betaz']]
    cs = np.sqrt(np.clip(GAM * P / (rho + GAM / (GAM - 1) * P), 0, None))
    dx = ((g[:, 1] - g[:, 0]) / nx)[:, None, None]
    rate = np.zeros_like(rho)
    for a in range(3):
        ra = (np.abs(alp * u[a] / W - beta[a]) + alp * cs / np.sqrt(gd[a])) / dx
        rate = np.maximum(rate, ra)
    xc = g[:, 0, None, None] + (np.arange(nx)[None, None, :] + .5) * dx
    yc = g[:, 2, None, None] + (np.arange(nx)[None, :, None] + .5) * dx
    xc, yc = np.broadcast_to(xc, rho.shape), np.broadcast_to(yc, rho.shape)
    m = np.unravel_index(np.argmax(rho), rho.shape)
    V = np.sqrt(sum(gd[a] * (u[a] / W)**2 for a in range(3)))
    bmag = np.sqrt(sum(gd[a] * beta[a]**2 for a in range(3)))
    return dict(t=p['time'], rate=rate, rho=rho, cs=cs, V=V, alp=alp, bmag=bmag,
                rbh=np.hypot(xc, yc), rst=np.hypot(xc - xc[m], yc - yc[m]) / T.RSTAR,
                dx=np.broadcast_to(dx, rho.shape), sep=np.hypot(xc[m], yc[m]))


def classify(s, idx):
    if s['rbh'][idx] < 4.0:
        return 'near-BH'
    if s['rho'][idx] > 1e-3 * T.RHO0:
        return 'star/stream'
    if s['rst'][idx] < 4.0:
        return 'bow/env'
    return 'atmosphere'


def dt_series(d):
    t, dt = [], []
    with open(f'{d}/ensemble.out') as f:
        for line in f:
            mm = re.search(r'cycle=(\d+) time=([\d.e+-]+) dt=([\d.e+-]+)', line)
            if mm:
                t.append(float(mm.group(2))); dt.append(float(mm.group(3)))
    return np.array(t), np.array(dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('labels', nargs='*', default=['V4F'])
    ap.add_argument('--times', default='1,20,40,60,80,100,110,120,125,130,140,150,160')
    ap.add_argument('--top', type=int, default=50)
    ap.add_argument('--out', default=f'{RUN}/plots/dual_energy')
    a = ap.parse_args()
    rows = []
    for lab in a.labels:
        d = DIRS[lab]
        for t in [float(x) for x in a.times.split(',')]:
            if dump(d, 'prim_xy', t) is None:
                continue
            s = rates(d, t)
            flat = np.argsort(s['rate'], axis=None)[::-1][:a.top]
            idx = np.unravel_index(flat[0], s['rate'].shape)
            cats = [classify(s, np.unravel_index(f, s['rate'].shape)) for f in flat]
            frac = {c: cats.count(c) / len(cats) for c in ('near-BH', 'star/stream', 'bow/env', 'atmosphere')}
            rows.append(
                f"{lab} t={s['t']:6.1f} sep {s['sep']:5.1f}  dt~{0.4 / s['rate'][idx]:.4f}  max cell: "
                f"{classify(s, idx):11s} r_BH {s['rbh'][idx]:5.2f} r_*/R* {s['rst'][idx]:5.2f} "
                f"rho/rho0 {s['rho'][idx] / T.RHO0:8.1e} rho/dfloor {s['rho'][idx] / DFLOOR:8.1e} "
                f"|V| {s['V'][idx]:.3f} c_s {s['cs'][idx]:.3f} alpha {s['alp'][idx]:.2f} "
                f"|beta| {s['bmag'][idx]:.3f} dx {s['dx'][idx]:.3f} | top{a.top}: "
                + ' '.join(f'{k} {v:.2f}' for k, v in frac.items() if v > 0))
    print('\n'.join(rows))
    os.makedirs(a.out, exist_ok=True)
    with open(f'{a.out}/H_dt_cells.txt', 'w') as f:
        f.write('\n'.join(rows) + '\n')
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5), constrained_layout=True)
    for lab, c in (('V4F', '#1b9e77'), ('V4', '#e7298a')):
        if os.path.exists(f'{DIRS[lab]}/ensemble.out'):
            t, dt = dt_series(DIRS[lab])
            ax.semilogy(t, dt, color=c, lw=0.8, label=f'{lab} ({len(t)} cycles to t={t[-1]:.0f})')
    ax.set_xlabel('t [M]'); ax.set_ylabel('dt'); ax.grid(alpha=0.3); ax.legend()
    ax.set_title('rp16 L4: time step (H)')
    fig.savefig(f'{a.out}/H_dt_history.png', dpi=110)


if __name__ == '__main__':
    main()
