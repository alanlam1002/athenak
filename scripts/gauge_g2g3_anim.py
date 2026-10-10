#!/usr/bin/env python3
"""Snapshots and animation for the comoving-gauge test pair (G0, boosted, G2, G3, G3b;
src/cfc/DEVELOPMENT.md item 68, scripts/gauge_g2g3_plots.py for the time series).

  gauge_snapshots_density.png  star-centred log rho (z=0), runs x times
  gauge_snapshots_entropy.png  star-centred K/K_0 = P rho^(-5/3) / kappa, same layout
  gauge_grid_view.png          the whole +-32 box at t = 0, 50, 100: where each star sits
                               on the grid (grid coordinates x')
  gauge_anim.gif               all five runs side by side, density (top) and K (bottom)

Star-centred = centred on the density maximum of each dump, so all runs compare in the
star's frame regardless of gauge. Needs numpy + matplotlib + Pillow.

    gauge_g2g3_anim.py [--out DIR] [--every N]
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                   # noqa: E402
from matplotlib.colors import LogNorm              # noqa: E402
from matplotlib import animation                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gauge_g2g3_plots as G                      # noqa: E402  (registers G0/G2/G3/G3b)
import tde_t11_surface_diag as T                   # noqa: E402
from tde_ensemble_plots import dump_at, draw       # noqa: E402
import bin_convert as bc                           # noqa: E402

G.T.RUNS['E1b'] = (f'{G.RUN}/E1b_G2c_atmframe', G.ISO, None, '#66a61e', ':')
for _lab, _d, _c in (('V2', 'de_V2_static_L4', '#1b9e77'),
                     ('V3', 'de_V3_boost_contract_L4', '#d95f02'),
                     ('V3c', 'de_V3c_boost_contract_noDE_L4', '#7570b3'),
                     ('G2cDE', 'de_G_G2c_DE_L4', '#e7298a'),
                     ('V3F', 'de_V3F_boost_contract_L4', '#1b9e77'),
                     ('V3FJ', 'de_V3FJ_boost_contract_L4', '#e7298a'),
                     ('GJ', 'de_GJ_G2c_DE_L4', '#e6ab02')):
    G.T.RUNS[_lab] = (f'{G.RUN}/{_d}', G.ISO, None, _c, '--')
RUNS = ['G0', 'iso_boost', 'G2', 'G3', 'G3b']
TITLE = {'G2c': 'G2c (contracted ID, ξ̇=V0)', 'E1b': 'E1b (G2c + comoving atm, t≤40)',
         'G0': 'G0 static', 'iso_boost': 'boosted (v=V0, ξ̇=0)', 'G2': 'G2 (v=V0, ξ̇=V0)',
         'G3': 'G3 (v=0, ξ̇=−V0)', 'G3b': 'G3b (v=0, ξ̇=−at)',
         'V2': 'V2 static, DE on', 'V3c': 'V3c boosted, contracted, DE off',
         'V3': 'V3 boosted, contracted, DE on', 'G2cDE': 'G (G2c + DE)',
         'V3F': 'V3F boosted, DE with F', 'V3FJ': 'V3FJ boosted, F + J',
         'GJ': 'GJ comoving, F + J'}
NR = LogNorm(vmin=T.RHO0 * 1e-7, vmax=T.RHO0 * 1.2)
NK = LogNorm(vmin=0.3, vmax=3.0)
CR = plt.get_cmap('inferno').copy(); CR.set_bad('k')
CK = plt.get_cmap('coolwarm').copy(); CK.set_bad('0.15')


def blocks(lab, t):
    pf = dump_at(lab, t)
    if pf is None:
        return None
    p = bc.read_binary(pf)
    g = np.asarray(p['mb_geometry'])
    d = np.asarray(p['mb_data']['dens'])[:, 0]
    P = np.asarray(p['mb_data']['press'])[:, 0]
    K = np.where(d > 1e-4 * T.RHO0, P / d**T.GAM / T.K0, np.nan)
    m, j, i = np.unravel_index(np.argmax(d), d.shape)
    nx, ny = p['nx1_out_mb'], p['nx2_out_mb']
    x0 = g[m, 0] + (i + .5) * (g[m, 1] - g[m, 0]) / nx
    y0 = g[m, 2] + (j + .5) * (g[m, 3] - g[m, 2]) / ny
    bd = [(g[k, 0], g[k, 1], g[k, 2], g[k, 3], d[k]) for k in range(len(g))]
    bk = [(g[k, 0], g[k, 1], g[k, 2], g[k, 3], K[k]) for k in range(len(g))]
    return p['time'], x0, y0, bd, bk, d.max() / T.RHO0


def panel(ax, b, field, half, outline=False):
    t, x0, y0, bd, bk, _ = b
    view = ((x0 - half, x0 + half), (y0 - half, y0 + half))
    if field == 'rho':
        draw(ax, bd, view, NR, CR, outline=outline)
    else:
        draw(ax, bk, view, NK, CK, outline=outline)
    ax.add_patch(plt.Circle((x0, y0), T.RSTAR, fill=False, color='c', lw=0.6, ls='--'))
    ax.set_xticks([]); ax.set_yticks([])


def snapshots(out, field, times=(0, 20, 40, 60, 80, 100), half=1.4, tag=''):
    fig, ax = plt.subplots(len(RUNS), len(times), figsize=(2.4 * len(times), 2.5 * len(RUNS)),
                           constrained_layout=True)
    for r, lab in enumerate(RUNS):
        for c, t in enumerate(times):
            b = blocks(lab, t)
            a = ax[r, c]
            if b is None:
                a.axis('off'); continue
            panel(a, b, field, half)
            if r == 0:
                a.set_title(f't = {t}', fontsize=10)
            if c == 0:
                a.set_ylabel(TITLE[lab], fontsize=9)
    sm = plt.cm.ScalarMappable(norm=NR if field == 'rho' else NK, cmap=CR if field == 'rho' else CK)
    fig.colorbar(sm, ax=ax, shrink=0.6,
                 label=(r'$\rho$ (z=0)' if field == 'rho' else r'$K/K_0$ ($\rho>10^{-4}\rho_0$)')
                 + r';  star-centred $\pm$%.1f M, dashed = $R_*$' % half)
    name = (f'gauge{tag}_snapshots_density.png' if field == 'rho'
            else f'gauge{tag}_snapshots_entropy.png')
    fig.savefig(f'{out}/{name}', dpi=100); plt.close(fig)


def grid_view(out, times=(0, 50, 100), tag=''):
    fig, ax = plt.subplots(len(times), len(RUNS), figsize=(3.2 * len(RUNS), 3.2 * len(times)),
                           constrained_layout=True)
    for c, lab in enumerate(RUNS):
        for r, t in enumerate(times):
            b = blocks(lab, t)
            a = ax[r, c]
            if b is None:
                a.axis('off'); continue
            draw(a, b[3], ((-32, 32), (-32, 32)), NR, CR, outline=True)
            a.plot(10, -8, 'c+', ms=8)
            a.set_xticks([-30, 0, 30]); a.set_yticks([-30, 0, 30]); a.tick_params(labelsize=7)
            if r == 0:
                a.set_title(TITLE[lab], fontsize=10)
            if c == 0:
                a.set_ylabel(f't = {t}  (grid coords)', fontsize=9)
    fig.suptitle("Where the star sits on the grid (x' = x − ξ). Cyan + = start (10, −8); "
                 'white lines = MeshBlocks', fontsize=11)
    fig.savefig(f'{out}/gauge{tag}_grid_view.png', dpi=90); plt.close(fig)


def anim(out, every=1, half=1.4, tag='', tmax=100):
    times = list(range(0, tmax + 1, every))
    frames = {lab: [blocks(lab, t) for t in times] for lab in RUNS}
    fig, ax = plt.subplots(2, len(RUNS), figsize=(3.0 * len(RUNS), 6.4), constrained_layout=True)
    fig.colorbar(plt.cm.ScalarMappable(norm=NR, cmap=CR), ax=ax[0], shrink=0.8, label=r'$\rho$')
    fig.colorbar(plt.cm.ScalarMappable(norm=NK, cmap=CK), ax=ax[1], shrink=0.8, label=r'$K/K_0$')

    def update(n):
        for c, lab in enumerate(RUNS):
            b = frames[lab][n]
            for r, field in enumerate(('rho', 'K')):
                a = ax[r, c]
                a.clear()
                if b is None:
                    a.axis('off'); continue
                panel(a, b, field, half)
            if b is not None:
                ax[0, c].set_title(f'{TITLE[lab]}\n' + rf'$\rho_{{max}}/\rho_0$={b[5]:.3f}',
                                   fontsize=9)
        fig.suptitle(f'Comoving-gauge test pair, star-centred ±{half} M,  t = {times[n]}',
                     fontsize=11)
        return []

    an = animation.FuncAnimation(fig, update, frames=len(times), blit=False)
    an.save(f'{out}/gauge{tag}_anim.gif', writer=animation.PillowWriter(fps=8), dpi=70)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{T.RUN}/plots/gauge_g2g3')
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--runs', default=','.join(RUNS), help='comma-separated labels')
    ap.add_argument('--tag', default='', help='suffix for output names, e.g. _G2c')
    ap.add_argument('--tmax', type=int, default=100)
    ap.add_argument('--no-grid', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    RUNS[:] = a.runs.split(',')
    times = tuple(t for t in (0, 20, 40, 60, 80, 100) if t <= a.tmax)
    snapshots(a.out, 'rho', times, tag=a.tag); print('density snapshots', flush=True)
    snapshots(a.out, 'K', times, tag=a.tag); print('entropy snapshots', flush=True)
    if not a.no_grid:
        grid_view(a.out, tag=a.tag); print('grid view', flush=True)
    anim(a.out, a.every, tag=a.tag, tmax=a.tmax); print('animation', flush=True)


if __name__ == '__main__':
    main()
