#!/usr/bin/env python3
"""c253 (R-253): figures from c253_dmde.py and c253_stream.py outputs in DIR.
  c253_dmde.png      the debris dM/dE (Hill sphere removed) per orbit, rungs overlaid (light -> dark = 1/16 -> 1/64)
  c253_census.png    the stream's mass-weighted cell size and the share at the rung's finest dx vs t - t_p
    c253_plots.py DIR"""
import sys, os, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument('dir'); ap.add_argument('--rad', type=float, default=30.0); a = ap.parse_args()
D = a.dir
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
RES = {16: '#86b6ef', 32: '#2a78d6', 64: '#173f73'}
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.8})
ORB = {'P18': ([('P18_r82', 16), ('L32_P18', 32), ('L64_P18', 64)], ['P18_r82_late']),
       'P10': ([('P10_r82', 16), ('L32_P10', 32), ('L64_P10', 64)], ['P10_r82_late']),
       'PD07': ([('PD07', 16), ('PD07_L32', 32)], [])}
rd = lambda f: pd.read_csv(f) if os.path.exists(f) and os.path.getsize(f) > 0 else None

# ---- 1. dM/dE
fig, axs = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
for x, orb in zip(axs, ORB):
    f = f'{D}/dmde_{orb}.npz'
    if not os.path.exists(f):
        continue
    z = np.load(f); e = z['edges']; mid = 0.5*(e[1:] + e[:-1])
    for k in sorted(k for k in z.files if k.endswith('_debris')):
        res = int(k.split('_')[1]); tt = k.split('_')[2]
        x.step(mid, np.maximum(z[k], 1e-6), where='mid', color=RES[res], label=f'dx 1/{res}, t = {tt[1:]}',
               ls='-' if k == sorted(kk for kk in z.files if kk.endswith('_debris') and f'_{res}_' in kk)[0] else ':')
    x.set_yscale('log'); x.set_xlim(-3, 3); x.set_ylim(1e-4, 10); x.axvline(0, color=MUTED, lw=0.8, ls=':')
    x.set_title(f'{orb}: debris dM/dE (Hill sphere removed)', loc='left', fontsize=10)
    x.set_xlabel('E / ΔE'); x.legend(frameon=False, fontsize=7)
axs[0].set_ylabel('(1/M) dM/d(E/ΔE)')
fig.tight_layout(); fig.savefig(f'{D}/c253_dmde.png', dpi=130); plt.close(fig)

# ---- 2. census
fig, axs = plt.subplots(2, 3, figsize=(13, 6.5), sharex='col')
for c, orb in enumerate(ORB):
    runs, late = ORB[orb]
    for tag, res in runs:
        d = rd(f'{D}/{tag}_census.csv')
        if d is None:
            continue
        if res == 16:
            for l in late:
                dl = rd(f'{D}/{l}_census.csv')
                if dl is not None:
                    d = pd.concat([d, dl]).sort_values('t')
        axs[0, c].semilogy(d.t_tp, d.dx_mw, color=RES[res], label=f'dx 1/{res}')
        col = f'f_dx{res}'
        if col in d:
            axs[1, c].plot(d.t_tp, d[col], color=RES[res], label=f'dx 1/{res}')
    axs[0, c].set_title(orb, loc='left', fontsize=10)
    axs[1, c].set_xlabel('t − t_p  [M]')
    for x in axs[:, c]:
        x.axvline(0, color=MUTED, lw=0.8, ls=':')
axs[0, 0].set_ylabel('stream ⟨dx⟩ (mass-weighted) [M]')
axs[1, 0].set_ylabel('stream mass share at the rung\'s finest dx')
axs[0, 0].legend(frameon=False, fontsize=8)
fig.suptitle('Stream (1e-6 ≤ ρ/ρ_c < 0.01, outside the Hill sphere): refinement census per rung', x=0.01, ha='left', fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f'{D}/c253_census.png', dpi=130); plt.close(fig)

print('wrote', D)
