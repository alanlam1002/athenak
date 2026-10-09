#!/usr/bin/env python3
"""Plots for the R-028 tests that are not star snapshots (src/cfc/DEVELOPMENT.md 69/70):

  de_V1_sod.png         1D relativistic Sod, dual energy off vs on: rho, P, v, K
  de_V1_wave.png        high-Mach entropy wave: P/P0 - 1 and K/K0 - 1, off vs on
  dt_history.png        dt(t) and cycles(t) for G0, boosted, G2, G2c, E1b
  cfl_maps_t30.png      what sets dt: local CFL rate (|V'| + c_s)/dx and c_s near the star
                        at t = 30 for G0, G2c, E1b (V' = v - xidot, grid velocity)
  cfl_anim.gif          the same c_s / CFL-rate maps for t = 0..40
  E2_nondeterminism.png two identical runs: dt per cycle and relative difference

    de_e_diagnostic_plots.py [--out DIR] [--no-anim]
"""
import argparse
import glob
import os
import re
import sys
import types

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                   # noqa: E402
from matplotlib.colors import LogNorm              # noqa: E402
from matplotlib import animation                   # noqa: E402

sys.modules.setdefault('h5py', types.ModuleType('h5py'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'vis', 'python'))
sys.path.insert(0, HERE)
import bin_convert as bc                           # noqa: E402
from tde_ensemble_plots import draw                # noqa: E402

RUN = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc'
GAM = 5.0 / 3.0
V0 = np.array([-0.19262227, 0.15093084, 0.0])
RUNS3D = {   # label: (dir, frame velocity xidot, colour)
    'G0': ('wd_gauge_G0_L4_si1', np.zeros(3), 'k'),
    'boosted': ('wd_isolated_boost_L4_si1_ens1', np.zeros(3), '#1b9e77'),
    'G2': ('wd_gauge_G2_L4_si1', V0, '#d95f02'),
    'G2c': ('wd_gauge_G2c_L4_si1', V0, '#a6761d'),
    'E1b': ('E1b_G2c_atmframe', V0, '#66a61e'),
}


def tab(d, n):
    fn = sorted(glob.glob(f'{d}/tab/*.tab'))[n]
    hdr = [l for l in open(fn) if l.startswith('#')][-1].split()[1:]
    a = np.loadtxt(fn, ndmin=2)
    if a.shape[1] < len(hdr):
        hdr = [h for h in hdr if h not in ('j', 'x2v', 'k', 'x3v')]
    return {h: a[:, i] for i, h in enumerate(hdr)}


def fig_v1(out):
    D = f'{RUN}/dual_energy_tests'
    off, on = tab(f'{D}/V1_sod_off', 1), tab(f'{D}/V1_sod_on', 1)
    x = off['x1v']
    fig, ax = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for s, lab, st in ((off, 'dual energy off', dict(color='k', lw=1.5)),
                       (on, 'dual energy on', dict(color='#d95f02', lw=1.2, ls='--', marker='.', ms=3))):
        ax[0, 0].plot(x, s['dens'], label=lab, **st)
        ax[0, 1].plot(x, s['press'], label=lab, **st)
        ax[1, 0].plot(x, s['velx'] / np.sqrt(1 + s['velx']**2), label=lab, **st)
        ax[1, 1].plot(x, s['press'] / s['dens']**GAM, label=lab, **st)
    for a, t in zip(ax.flat, (r'$\rho$', 'P', r'$v^x$', r'$K = P/\rho^{5/3}$')):
        a.set_title(t); a.set_xlabel('x'); a.grid(alpha=0.3)
    ax[0, 0].legend()
    ax[1, 1].set_yscale('log')
    fig.suptitle('V1: relativistic Sod, Γ=5/3, ppmx/HLLE/RK3/FOFC, 256 cells, t=0.4 '
                 '(shock x≈0.83, contact x≈0.67)')
    fig.savefig(f'{out}/de_V1_sod.png', dpi=110); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for r, lab, c in (('V1_wave_hiM_off', 'off', 'k'), ('V1_wave_hiM_on', 'on', '#d95f02')):
        a, b = tab(f'{D}/{r}', 0), tab(f'{D}/{r}', 1)
        x = b['x1v']
        ax[0].plot(x, b['press'] / a['press'] - 1, color=c, label=f'dual energy {lab}')
        Ka, Kb = a['press'] / a['dens']**GAM, b['press'] / b['dens']**GAM
        ax[1].plot(x, Kb / Ka - 1, color=c, label=f'dual energy {lab}')
    ax[0].set_title(r'$P/P_0 - 1$ after 2 periods'); ax[1].set_title(r'$K/K_0 - 1$ after 2 periods')
    for a in ax:
        a.set_xlabel('x'); a.grid(alpha=0.3); a.legend()
    fig.suptitle('High-Mach entropy wave (v=0.25, p0=1e-6, grid Mach ~200, 128 cells): '
                 'contact noise of the entropy tracer')
    fig.savefig(f'{out}/de_V1_wave.png', dpi=110); plt.close(fig)


def dt_series(d):
    rows = []
    for line in open(f'{RUN}/{d}/ensemble.out'):
        m = re.match(r'elapsed=(\S+) cycle=(\d+) time=(\S+) dt=(\S+)', line)
        if m:
            rows.append((float(m.group(3)), int(m.group(2)), float(m.group(4))))
    return np.array(rows)


def fig_dt(out):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for lab, (d, _, c) in RUNS3D.items():
        if not os.path.exists(f'{RUN}/{d}/ensemble.out'):
            continue
        a = dt_series(d)
        ax[0].semilogy(a[:, 0], a[:, 2], color=c, lw=1, label=lab)
        ax[1].plot(a[:, 0], a[:, 1], color=c, lw=1.5, label=f'{lab} ({a[-1, 1]} cycles to t={a[-1, 0]:.0f})')
    ax[0].set_title('dt(t)'); ax[1].set_title('cycles(t)')
    for a in ax:
        a.set_xlabel('t [M]'); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle('Timestep: a moving star costs ~6x the cycles of a static one, in either gauge '
                 '(E1b: comoving atmosphere does not help)')
    fig.savefig(f'{out}/dt_history.png', dpi=110); plt.close(fig)


def cfl_fields(lab, t):
    d, xd, _ = RUNS3D[lab]
    pfs = sorted(glob.glob(f'{RUN}/{d}/bin/*prim_xy*.bin'))
    pf = min(pfs, key=lambda f: abs(int(f.split('.')[-2]) - t))
    p = bc.read_binary(pf)
    g = np.asarray(p['mb_geometry']); nx = p['nx1_out_mb']
    md = {k: np.asarray(v)[:, 0] for k, v in p['mb_data'].items()}
    u = [md['velx'], md['vely'], md['velz']]
    W = np.sqrt(1 + u[0]**2 + u[1]**2 + u[2]**2)
    V = np.sqrt(sum((u[a] / W - xd[a])**2 for a in range(3)))
    cs = np.sqrt(GAM * md['press'] / (md['dens'] + GAM / (GAM - 1) * md['press']))
    dx = ((g[:, 1] - g[:, 0]) / nx)[:, None, None]
    rate = (V + cs) / dx
    m, j, i = np.unravel_index(np.argmax(rate), rate.shape)
    x0 = g[m, 0] + (i + .5) * dx[m, 0, 0]; y0 = g[m, 2] + (j + .5) * dx[m, 0, 0]
    dm, jm, im = np.unravel_index(np.argmax(md['dens']), md['dens'].shape)
    cx = g[dm, 0] + (im + .5) * dx[dm, 0, 0]; cy = g[dm, 2] + (jm + .5) * dx[dm, 0, 0]
    blk = lambda f: [(g[k, 0], g[k, 1], g[k, 2], g[k, 3], f[k]) for k in range(len(g))]
    return p['time'], (cx, cy), (x0, y0), blk(cs), blk(rate), blk(md['dens'])


NCS = LogNorm(vmin=1e-4, vmax=0.3)
NRT = LogNorm(vmin=1e-2, vmax=10)


def draw_cfl(axc, axr, lab, t, half=3.0):
    tt, (cx, cy), (x0, y0), bcs, brt, bd = cfl_fields(lab, t)
    view = ((cx - half, cx + half), (cy - half, cy + half))
    draw(axc, bcs, view, NCS, 'magma', outline=False)
    draw(axr, brt, view, NRT, 'viridis', outline=False)
    for a in (axc, axr):
        a.plot(x0, y0, 'c*', ms=12, mec='w')
        a.add_patch(plt.Circle((cx, cy), 0.677, fill=False, color='w', lw=0.6, ls='--'))
        a.set_xticks([]); a.set_yticks([])
    axc.set_title(f'{lab} t={tt:.0f}: sound speed c_s', fontsize=9)
    axr.set_title(f'{lab}: CFL rate (|V\'|+c_s)/dx', fontsize=9)


def fig_cfl(out, labs=('G0', 'G2c', 'E1b'), t=30):
    fig, ax = plt.subplots(2, len(labs), figsize=(4.4 * len(labs), 8.4), constrained_layout=True)
    for c, lab in enumerate(labs):
        draw_cfl(ax[0, c], ax[1, c], lab, t)
    fig.colorbar(plt.cm.ScalarMappable(norm=NCS, cmap='magma'), ax=ax[0], label='c_s')
    fig.colorbar(plt.cm.ScalarMappable(norm=NRT, cmap='viridis'), ax=ax[1],
                 label="(|V'| + c_s)/dx   [dt ≈ 0.4 / max]")
    fig.suptitle('What sets dt (z=0, ±3 M around the star; ★ = slice maximum; dashed = R_*). '
                 'Moving stars heat the near-floor gas around them.')
    fig.savefig(f'{out}/cfl_maps_t30.png', dpi=100); plt.close(fig)


def anim_cfl(out, labs=('G0', 'G2c', 'E1b'), tmax=40):
    fig, ax = plt.subplots(2, len(labs), figsize=(4.2 * len(labs), 8), constrained_layout=True)
    fig.colorbar(plt.cm.ScalarMappable(norm=NCS, cmap='magma'), ax=ax[0], label='c_s')
    fig.colorbar(plt.cm.ScalarMappable(norm=NRT, cmap='viridis'), ax=ax[1], label="(|V'|+c_s)/dx")

    def update(n):
        for c, lab in enumerate(labs):
            ax[0, c].clear(); ax[1, c].clear()
            draw_cfl(ax[0, c], ax[1, c], lab, n)
        fig.suptitle(f'Sound speed and CFL rate around the star, t = {n}')
        return []
    an = animation.FuncAnimation(fig, update, frames=tmax + 1, blit=False)
    an.save(f'{out}/cfl_anim.gif', writer=animation.PillowWriter(fps=6), dpi=65)
    plt.close(fig)


def fig_e2(out):
    a, b = dt_series('E2_nondet_a'), dt_series('E2_nondet_b')
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    ax[0].plot(a[:, 1], a[:, 2], 'ko-', label='run a'); ax[0].plot(b[:, 1], b[:, 2], 'r.--', label='run b')
    ax[0].set_xlabel('cycle'); ax[0].set_ylabel('dt'); ax[0].legend(); ax[0].grid(alpha=0.3)
    rel = np.abs(a[:, 2] / b[:, 2] - 1)
    ax[1].semilogy(a[:, 1], np.maximum(rel, 1e-17), 'bo-')
    ax[1].set_xlabel('cycle'); ax[1].set_ylabel('|dt_a/dt_b − 1|'); ax[1].grid(alpha=0.3)
    ax[1].set_title('identical at cycles 0-1, 1e-3 from cycle 2')
    fig.suptitle('E2: two identical GPU runs (static WD, same binary and deck)')
    fig.savefig(f'{out}/E2_nondeterminism.png', dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{RUN}/plots/dual_energy')
    ap.add_argument('--no-anim', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_v1(a.out); print('V1', flush=True)
    fig_dt(a.out); print('dt', flush=True)
    fig_cfl(a.out); print('cfl maps', flush=True)
    fig_e2(a.out); print('E2', flush=True)
    if not a.no_anim:
        anim_cfl(a.out); print('cfl animation', flush=True)


if __name__ == '__main__':
    main()
