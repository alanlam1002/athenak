#!/usr/bin/env python3
"""Comoving-gauge test pair (research R-021, comoving_gauge.pdf §10; src/cfc/DEVELOPMENT.md
item 68): isolated WD, L4, si=1, no BH.

    run         IC Eulerian v   gauge Xdot      coordinate velocity   bulk KE in tau
    static G0   0               0               0                     no   (±32 box, same deck)
    boosted     V0              0               V0                    yes
    G2          V0              V0              ~0                    yes
    G3          0               -V0             V0                    no
    G3b         0               -a t            a t (accelerating)    no

Per dump (z=0 slices; tde_t11_surface_diag.measure): rho_max, K = P/rho^(5/3) at the max,
min and rho-weighted K over rho > 0.1 rho_0, puffed slice-mass fraction beyond R_*. Plus
the star's CoM mapped to the physical frame, x = x' + X(t) (X(t) from the user.hst gauge
columns), the history rest mass, and star-centred density profiles at t = 40 and 100.
Velocities in the dumps are Eulerian (gauge-invariant), so puff speeds compare directly.

    gauge_g2g3_plots.py [--out DIR]
"""
import argparse
import glob
import os
import sys
import types

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

sys.modules.setdefault('h5py', types.ModuleType('h5py'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tde_t11_surface_diag as T          # noqa: E402
from orbit_invariants import gauge_from_hst   # noqa: E402

RUN = T.RUN
ISO = T.ISO
T.RUNS.update({
    'G0':  (f'{RUN}/wd_gauge_G0_L4_si1', ISO, None, '#000000', '-'),
    'G2':  (f'{RUN}/wd_gauge_G2_L4_si1', ISO, None, '#d95f02', '-'),
    'G3':  (f'{RUN}/wd_gauge_G3_L4_si1', ISO, None, '#7570b3', '-'),
    'G3b': (f'{RUN}/wd_gauge_G3b_L4_si1', ISO, None, '#e7298a', '--'),
})
LABELS = ['G0', 'iso_static', 'iso_boost', 'G2', 'G3', 'G3b']
NAMES = {'G0': 'G0 static (box ±32)', 'iso_static': 'old static (box ±16, t≤40)', 'iso_boost': 'boosted (v=V0, Xdot=0)',
         'G2': 'G2 (v=V0, Xdot=V0)', 'G3': 'G3 (v=0, Xdot=-V0)', 'G3b': 'G3b (v=0, Xdot=-at)'}


def style(lab):
    _, _, _, c, ls = T.RUNS[lab]
    return dict(color=c, ls=ls, lw=1.5, label=NAMES[lab])


def physical_track(lab):
    d, base, _, _, _ = T.RUNS[lab]
    g = gauge_from_hst(d, base)
    rows = []
    for pf in sorted(glob.glob(f'{d}/bin/{base}.prim_xy.*.bin')):
        t, c = T.flatten(pf)
        s = c['rho'] > 0.5 * c['rho'].max()
        w = c['rho'][s]
        x, y = (w * c['x'][s]).sum() / w.sum(), (w * c['y'][s]).sum() / w.sum()
        if g is not None:
            X = g[1](t)
            x, y = x + X[0], y + X[1]
        rows.append((t, x, y))
    return np.array(rows)


def hst_cols(lab, names):
    d, base, _, _, _ = T.RUNS[lab]
    out = []
    for kind in ('user', 'mhd'):
        f = f'{d}/{base}.{kind}.hst'
        if not os.path.exists(f):
            continue
        lab2col = {}
        for line in open(f):
            if line.startswith('#') and '[1]=' in line:
                for tok in line[1:].split():
                    if '=' in tok:
                        i, n = tok.split('=', 1)
                        lab2col[n] = int(i.strip('[]')) - 1
                break
        h = np.loadtxt(f, comments='#', ndmin=2)
        for n in names:
            if n in lab2col:
                out.append((n, h[:, 0], h[:, lab2col[n]]))
    return {n: (t, v) for n, t, v in out}


def profile(lab, t):
    from tde_ensemble_plots import dump_at
    pf = dump_at(lab, t)
    if pf is None:
        return None
    tt, c = T.flatten(pf)
    i = np.argmax(c['rho'])
    r = np.hypot(c['x'] - c['x'][i], c['y'] - c['y'][i])
    edges = np.arange(0, 1.6, 1 / 32)
    w, _ = np.histogram(r, edges, weights=c['rho'] * c['dA'])
    a, _ = np.histogram(r, edges, weights=c['dA'])
    return 0.5 * (edges[1:] + edges[:-1]), np.where(a > 0, w / np.maximum(a, 1e-30), np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{RUN}/plots/gauge_g2g3')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    S, P = {}, {}
    for lab in LABELS:
        if glob.glob(f'{T.RUNS[lab][0]}/bin/*.prim_xy.*.bin'):
            S[lab] = T.series(lab, a.out)
            P[lab] = physical_track(lab)

    fig, ax = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    for lab, s in S.items():
        h = hst_cols(lab, ['rho-max', 'mass'])
        if 'rho-max' in h:
            t, v = h['rho-max']; ax[0, 0].plot(t, v / v[0], **style(lab))
        ax[0, 1].plot(s['t'], s['Kmax'], **style(lab))
        ax[0, 2].semilogy(s['t'], np.maximum(s['Kmin01'], 1e-20), **style(lab))
        ax[1, 0].semilogy(s['t'], np.maximum(s['puff'], 1e-6), **style(lab))
        if 'mass' in h:
            t, v = h['mass']; ax[1, 1].plot(t, v / v[0] - 1, **style(lab))
        p = P[lab]
        ax[1, 2].plot(p[:, 0], np.hypot(p[:, 1] - p[0, 1], p[:, 2] - p[0, 2]), **style(lab))
    ttl = [r'$\rho_{max}/\rho_{max}(0)$ (hst, 3D)', r'$K/K_0$ at the density max',
           r'min $K/K_0$ over $\rho>0.1\rho_0$', r'slice-mass fraction beyond $R_*$',
           'rest mass M(t)/M(0) - 1 (hst)', 'physical displacement of the core CoM [M]']
    for x, tl in zip(ax.flat, ttl):
        x.set_title(tl); x.set_xlabel('t [M]'); x.grid(alpha=0.3)
    ax[1, 2].plot([0, 100], [0, 0.2447 * 100], 'k:', lw=0.8, label='V0 t (boosted/G2 expected)')
    ax[0, 0].legend(fontsize=8); ax[1, 2].legend(fontsize=8)
    fig.suptitle('Comoving-gauge test pair (R-021 §10): same physics in different gauges')
    fig.savefig(f'{a.out}/gauge_test_summary.png', dpi=110); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for j, t in enumerate((40, 100)):
        for lab in S:
            pr = profile(lab, t)
            if pr is not None:
                ax[j].semilogy(pr[0], pr[1] / T.RHO0, **style(lab))
        ax[j].set_title(f'star-centred density profile (slice, azimuthal mean), t={t}')
        ax[j].set_xlabel('r from density max [M]'); ax[j].axvline(T.RSTAR, color='0.6', lw=0.8)
        ax[j].set_ylim(1e-8, 2); ax[j].grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.savefig(f'{a.out}/gauge_test_profiles.png', dpi=110); plt.close(fig)

    with open(f'{a.out}/gauge_test_table.txt', 'w') as f:
        f.write('label t rho_max/rho0 Kmax/K0 Kmin01/K0 Kw01/K0 puff puff_v | phys CoM disp\n')
        for lab, s in S.items():
            p = P[lab]
            for t in (40, 100):
                k = np.argmin(abs(s['t'] - t))
                if abs(s['t'][k] - t) > 1:
                    continue
                q = np.argmin(abs(p[:, 0] - t))
                disp = np.hypot(p[q, 1] - p[0, 1], p[q, 2] - p[0, 2])
                f.write(f"{lab:10s} {s['t'][k]:6.1f} {s['rhomax'][k]:.4f} {s['Kmax'][k]:.4f} "
                        f"{s['Kmin01'][k]:.4f} {s['Kw01'][k]:.4f} {s['puff'][k]:.3e} "
                        f"{s['puff_v'][k]:.2e} | {disp:.3f}\n")
    print(open(f'{a.out}/gauge_test_table.txt').read())


if __name__ == '__main__':
    main()
