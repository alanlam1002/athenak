#!/usr/bin/env python3
"""Phase 2 M1 figures (src/cfc/DEVELOPMENT.md item 75): the vacuum trumpet moving on the grid.

  m1_residuals.png   test 1a: max|d alpha|, max|d psi4|/psi4, max|d beta| against the analytic
                     trumpet (m1_vacuum.measure, BH-centred, 1.5 <= r_BH <= 8, alpha > 0.3)
                     and the lapse-minimum offset, vs t; static, xidot = 0.25 (xd025b+c) and
                     xidot = 0.45 (xd045b+c+d), restart segments chained
  m1_lapse_anim.gif  alpha (z = 0) of the xidot = 0.45 run with MeshBlock outlines (the BH
                     box follows x_BH(t) = puncture_x - xi(t), marked +)
  m1_dt.png          test 1b: mean dt over 5 < t <= 20 (mhd.hst) vs BH-box level, with and
                     without dt_skip_excised, BH-frame references, and max|d psi4|/psi4 at t = 20
                     (L4 with skip is the 1a run xd045b)

    m1_plots.py [--out DIR] [--every N] [--no-anim | --anim-only]
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                   # noqa: E402
from matplotlib import animation                   # noqa: E402
from matplotlib.colors import Normalize            # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m1_vacuum as V                              # noqa: E402
from tde_ensemble_plots import bc, draw            # noqa: E402

C = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc'
CHAINS = {'static': (['M1_1a_static'], '#444444'),
          'ξ̇ = 0.25': (['M1_1a_xd025b', 'M1_1a_xd025c'], '#1b9e77'),
          'ξ̇ = 0.45': (['M1_1a_xd045b', 'M1_1a_xd045c', 'M1_1a_xd045d'], '#d95f02')}
BH_FRAME_DT = (0.044, 0.062)      # BH frame with the star, no skip / skip (item 75)
RMAX = 8.0


def setup(d):
    txt = open(f'{C}/{d}/parfile.par').read()
    mov = re.search(r'^puncture_moving\s*=\s*true', txt, re.M) is not None
    return V.par(txt, 'cfc', 'puncture_x1'), V.par(txt, 'cfc', 'frame_vel1'), mov


def chain_files(dirs):
    seen = {}
    for d in dirs:                                  # later segments win on overlap
        for f in sorted(glob.glob(f'{C}/{d}/bin/*adm_xy*.bin')):
            seen[int(f.split('.')[-2])] = (d, f)
    return [seen[k] for k in sorted(seen)]


def residuals(out, every):
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.3), constrained_layout=True)
    names = ('max|Δα|', 'max|Δψ⁴|/ψ⁴', 'max|β − β_P − ξ̇|', 'lapse-min offset from x_BH')
    for lab, (dirs, col) in CHAINS.items():
        rows = []
        for d, f in chain_files(dirs)[::every]:
            rows.append(V.measure(f, *setup(d), rmax=RMAX))
        r = np.array(rows)
        print(f'{lab}: {len(r)} dumps t={r[0, 0]:.0f}-{r[-1, 0]:.0f}  dalpha {r[:, 2].min():.3e}-{r[:, 2].max():.3e}'
              f'  dpsi4 {r[:, 3].min():.3e}-{r[:, 3].max():.3e}  dbeta {r[:, 4].min():.3e}-{r[:, 4].max():.3e}',
              flush=True)
        for k in range(4):
            ax[k].plot(r[:, 0], r[:, 2 + k], '-', color=col, lw=1.2, label=lab)
        for d in dirs[1:]:                          # restart points
            t0 = int(chain_files([d])[0][1].split('.')[-2])
            for k in range(4):
                ax[k].axvline(t0, color=col, ls=':', lw=0.8)
    for k, a in enumerate(ax):
        a.set_title(names[k], fontsize=10); a.set_xlabel('t [M]'); a.grid(alpha=0.3)
        if k < 3:
            a.set_ylim(0, None)
    ax[0].legend(fontsize=9, title='dotted = restart')
    fig.suptitle('M1 test 1a: vacuum trumpet vs the analytic static trumpet in BH-centred coordinates '
                 f'(1.5 ≤ r_BH ≤ {RMAX:g}, α > 0.3, BH box L4)', fontsize=11)
    fig.savefig(f'{out}/m1_residuals.png', dpi=100); plt.close(fig)


def lapse_anim(out, step=2, half=56.0):
    dirs = CHAINS['ξ̇ = 0.45'][0]
    files = [(d, f) for d, f in chain_files(dirs) if int(f.split('.')[-2]) % step == 0]
    x0, v, _ = setup(dirs[0])
    norm = Normalize(0.0, 1.0); cmap = plt.get_cmap('viridis')
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.6), constrained_layout=True)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.85, label='α (z = 0)')

    def update(n):
        p = bc.read_binary(files[n][1])
        t = p['time']; xb = x0 - v * t
        g = np.asarray(p['mb_geometry']); al = np.asarray(p['mb_data']['adm_alpha'])[:, 0]
        bl = [(g[k, 0], g[k, 1], g[k, 2], g[k, 3], al[k]) for k in range(len(g))]
        for a, view, ttl in ((ax[0], ((-half, half), (-half, half)), 'grid, ±%g M' % half),
                             (ax[1], ((xb - 6, xb + 6), (-6, 6)), 'BH-centred ±6 M')):
            a.clear()
            draw(a, bl, view, norm, cmap, outline=False)
            a.add_collection(LineCollection(
                [[(b[0], b[2]), (b[1], b[2]), (b[1], b[3]), (b[0], b[3]), (b[0], b[2])] for b in bl],
                colors='k', linewidths=0.3, alpha=0.6))
            a.plot(xb, 0.0, '+', color='r', ms=12, mew=1.5)
            a.set_title(ttl, fontsize=10)
        fig.suptitle(f'M1: moving puncture, ξ̇ = {v:g}, x_BH(t) = {x0:g} − ξ̇t (red +);  t = {t:.0f},'
                     f' x_BH = {xb:+.1f}', fontsize=11)
        return []

    an = animation.FuncAnimation(fig, update, frames=len(files), blit=False)
    an.save(f'{out}/m1_lapse_anim.gif', writer=animation.PillowWriter(fps=8), dpi=65)
    plt.close(fig)


def dt_bars(out):
    lev = ['L4', 'L3', 'L2']
    dt = {s: [] for s in ('false', 'true')}
    err = []
    run = lambda L, s: 'M1_1a_xd045b' if (L, s) == ('L4', 'true') else f'M1_1b_{L}_skip{s}'
    for L in lev:
        for s in dt:
            h = np.loadtxt(glob.glob(f'{C}/{run(L, s)}/*.mhd.hst')[0])
            sel = (h[:, 0] > 5.0) & (h[:, 0] <= 20.0)
            dt[s].append(h[sel, 1].mean())
        d = run(L, 'true')
        f = sorted(glob.glob(f'{C}/{d}/bin/*adm_xy.00020.bin'))[-1]
        err.append(V.measure(f, *setup(d), rmax=RMAX)[3])
    print('dt no skip', np.round(dt['false'], 4), 'skip', np.round(dt['true'], 4),
          'dpsi4', ['%.2e' % e for e in err], flush=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    x = np.arange(len(lev)); w = 0.36
    b0 = ax.bar(x - w / 2, dt['false'], w, color='#7570b3', label='moving BH, no skip')
    b1 = ax.bar(x + w / 2, dt['true'], w, color='#1b9e77', label='moving BH, dt_skip_excised')
    ax.axhline(BH_FRAME_DT[0], color='#7570b3', ls='--', lw=1, label='BH frame + star, no skip (0.044)')
    ax.axhline(BH_FRAME_DT[1], color='#1b9e77', ls='--', lw=1, label='BH frame + star, skip (0.062)')
    for bb in (b0, b1):
        ax.bar_label(bb, fmt='%.3f', fontsize=8)
    for i, e in enumerate(err):
        ax.text(x[i], max(dt['true'][i], dt['false'][i]) * 1.12,
                f'max|Δψ⁴|/ψ⁴ {e:.1e}\n({e / err[0]:.1f}× L4)', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f'BH box {L}' for L in lev])
    ax.set_ylabel('mean dt, 5 < t ≤ 20 [M]'); ax.set_ylim(0, max(dt['true']) * 1.4)
    ax.legend(fontsize=8, loc='upper left'); ax.grid(axis='y', alpha=0.3)
    ax.set_title('M1 test 1b: timestep vs BH-box refinement (vacuum, ξ̇ = 0.45)', fontsize=11)
    fig.savefig(f'{out}/m1_dt.png', dpi=100); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{C}/plots/phase2_m1')
    ap.add_argument('--every', type=int, default=2)
    ap.add_argument('--no-anim', action='store_true')
    ap.add_argument('--anim-only', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if not a.anim_only:
        dt_bars(a.out); print('dt', flush=True)
        residuals(a.out, a.every); print('residuals', flush=True)
    if not a.no_anim:
        lapse_anim(a.out); print('animation', flush=True)


if __name__ == '__main__':
    main()
