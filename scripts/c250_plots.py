#!/usr/bin/env python3
"""c250 diagnosis figures (C-214; R-249 item 2): energy-only (DE off) vs D-ent (production) on the P18 approach, t = 0 -> 300,
with the earlier surface arms for context (C-209 DX_mode = legacy dual energy; C-212 DX_fixQ = the qtr resync gate).
  c250_core.png     core and inner shells <K/K0> vs t: r < 0.2 R*, 0.2-0.5 R*, 0.5-0.8 R* (rho >= 0.01 rho_c), c250_corek.py;
                    the 2-node production P18_r82 (c247_coldcore) as the reference.
  c250_surface.png  (a) M(d > 1.5 R*)/M*, (b) v_r of that gas (c206_outflow.py), (c) <K/K0> in the 1.0-1.5 R* shell,
                    (d) D-ent's branch-code mass shares in that shell (c250_shell_codes.py).
    c250_plots.py OUTDIR"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc'
A, X, O = f'{R}/c250_deoff/ana', f'{R}/c247_dxfix/ana', f'{R}/c206_outflow'
out = sys.argv[1]
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.8})
# categorical slots in fixed order, one per arm across both figures
ARM = {'DE_on': ('D-ent (production), DE_on', '#2a78d6'), 'DE_off': ('energy only, DE_off', '#eb6834'),
       'DX_mode': ('legacy dual energy, DX_mode', '#1baf7a'), 'DX_fixQ': ('D-ent + qtr resync gate, DX_fixQ', '#eda100')}
REF = dict(color=MUTED, ls='--', lw=1.2)


def corek(a):
    return pd.read_csv(f'{A}/corek_{a}.txt', sep=r'\s+', comment='#', names=['t', 'core', 's25', 's58'])


# ---- figure 1: core and inner shells
P = pd.read_csv(f'{R}/c247_coldcore/P18.csv')
P = P[P.t <= 301]
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)
for x, col, pcol, ttl in zip(ax, ('core', 's25', 's58'), ('core_KP', 's25_KP', 's58_KP'),
                             ('(a) core, r < 0.2 R*', '(b) 0.2-0.5 R*', '(c) 0.5-0.8 R*')):
    x.plot(P.t, P[pcol], label='production P18_r82 (2 nodes)', **REF)
    for a in ('DE_on', 'DE_off'):
        d = corek(a); x.plot(d.t, d[col], color=ARM[a][1], label=ARM[a][0])
    x.axhline(1.0, color=MUTED, lw=0.6)
    x.set_title(ttl, loc='left', fontsize=10); x.set_xlabel('t  [M]')
ax[0].set_ylabel('⟨K/K0⟩, mass-weighted')
h, l = ax[0].get_legend_handles_labels()
fig.suptitle('Energy-only vs D-ent on the P18 approach (dx 1/16): the core cools identically', x=0.01, y=0.995,
             ha='left', fontsize=11)
fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.86))
fig.savefig(f'{out}/c250_core.png', dpi=130); plt.close(fig)

# ---- figure 2: the surface
of = {'DE_on': f'{A}/outflow_DE_on.csv', 'DE_off': f'{A}/outflow_DE_off.csv', 'DX_mode': f'{O}/DX_mode.csv',
      'DX_fixQ': f'{X}/outflow_DX_fixQ.csv'}
sh = {'DE_on': f'{A}/shell_DE_on.csv', 'DE_off': f'{A}/shell_DE_off.csv', 'DX_mode': f'{X}/shell_DX_mode.csv',
      'DX_fixQ': f'{X}/shell_DX_fixQ.csv'}
fig, ax = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True)
ax = list(ax.flat)
for a, f in of.items():
    if not os.path.exists(f):
        continue
    d = pd.read_csv(f).sort_values('t'); lab, c = ARM[a]
    s = d['M_gt1.5'] > 0
    ax[0].semilogy(d.t[s], d['M_gt1.5'][s], color=c, label=lab, marker='o', ms=3)
    if 'vr15' in d:
        v = d.dropna(subset=['vr15'])
        ax[1].plot(v.t, v.vr15, color=c, label=lab, marker='o', ms=3)
for a, f in sh.items():
    d = pd.read_csv(f); d = d[d.shell == 'S2'].sort_values('t'); lab, c = ARM[a]
    ax[2].semilogy(d.t, d.K, color=c, label=lab, marker='o', ms=3)
d = pd.read_csv(sh['DE_on']); d = d[d.shell == 'S2'].sort_values('t')
CODE = [('c11', '11 tracer (q_tr < η1)', INK, '-'), ('c12', '12 tracer by the safeguard', MUTED, '--'),
        ('c14', '14 energy + two-way resync', INK, ':'), ('c15', '15 energy, tracer passive', MUTED, '-.')]
for k, lab, c, ls in CODE:
    ax[3].plot(d.t, d[k], color=c, ls=ls, label=lab)
ax[1].axhline(0, color=MUTED, lw=0.6)
ax[0].set_title('(a) mass beyond 1.5 R*,  M(d > 1.5 R*)/M*', loc='left', fontsize=10)
ax[1].set_title('(b) its radial velocity relative to the star', loc='left', fontsize=10)
ax[2].set_title('(c) ⟨K/K0⟩ in the surface shell 1.0-1.5 R*', loc='left', fontsize=10)
ax[3].set_title('(d) D-ent (DE_on): branch-code mass shares, 1.0-1.5 R*', loc='left', fontsize=10)
ax[3].set_ylim(0, 1); ax[3].legend(frameon=False, fontsize=8, loc='center right')
for x in ax[2:]:
    x.set_xlabel('t  [M]')
h, l = ax[0].get_legend_handles_labels()
fig.suptitle('The pre-pericentre wind is D-ent\'s conditioning logic at the surface (P18, dx 1/16)', x=0.01, y=0.995,
             ha='left', fontsize=11)
fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 0.965), ncol=4, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(f'{out}/c250_surface.png', dpi=130); plt.close(fig)
print('wrote', out)
