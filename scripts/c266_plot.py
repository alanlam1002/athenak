#!/usr/bin/env python3
"""c266 (R-261): figures from c266_hidden.py's CSVs. Columns = class (core, outer, remnant, stream); rows:
(a) the hidden budget / the class's tracer thermal energy (H_hot + H_cold over E_th; solid: net, dotted: hot, dashed: cold);
(b) dken = K_en/K_tr - 1 over the class's 11 + 12 cells: the median (line) and the 25-75% band;
(c) the hidden energy carried out of the tracer branches per interval (11/12 -> 14, 15, 0), over the class's E_th at the
    interval's start (signed; markers per destination).
    c266_plot.py OUT.png LABEL=PREFIX:COLOR[:LS] [...]"""
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402

INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.4})
CLS = ['core', 'outer', 'remnant', 'stream']
MK = {'14': 'o', '15': 's', '0': '^'}
runs = []
for s in sys.argv[2:]:
    lab, rest = s.split('=', 1)
    p = rest.split(':')
    runs.append((lab, p[0], p[1], p[2] if len(p) > 2 else '-'))
fig, ax = plt.subplots(3, 4, figsize=(15, 9.5), sharex=True)
for lab, pre, col, ls in runs:
    b = pd.read_csv(f'{pre}_budget.csv'); q = pd.read_csv(f'{pre}_quant.csv')
    try:
        tr = pd.read_csv(f'{pre}_trans.csv')
    except FileNotFoundError:
        tr = None
    for j, c in enumerate(CLS):
        r = b[b.cls == c].sort_values('t_tp')
        ok = r.Eth_tr > 0
        ax[0, j].plot(r.t_tp[ok], (r.H_hot + r.H_cold)[ok] / r.Eth_tr[ok], color=col, ls=ls, label=lab)
        ax[0, j].plot(r.t_tp[ok], r.H_hot[ok] / r.Eth_tr[ok], color=col, ls=':', lw=0.9)
        ax[0, j].plot(r.t_tp[ok], r.H_cold[ok] / r.Eth_tr[ok], color=col, ls='--', lw=0.9)
        s = q[(q.cls == c) & (q.M_tr > 0)].sort_values('t_tp')
        ax[1, j].plot(s.t_tp, s.q50, color=col, ls=ls)
        ax[1, j].fill_between(s.t_tp, s.q25, s.q75, color=col, alpha=0.18, lw=0)
        if tr is not None and len(tr):
            t2 = tr[(tr.cls == c) & tr.frm.isin([11, 12]) & tr.to.astype(str).isin(['14', '15', '0'])]
            e0 = r.set_index('t').Eth_tr
            for dst, g in t2.groupby(t2.to.astype(str)):
                h = g.groupby('t0')[['H_hot', 'H_cold']].sum()
                den = e0.reindex(h.index).values
                x = h.index.values + 0.0 - (r.t - r.t_tp).iloc[0]
                with np.errstate(invalid='ignore', divide='ignore'):
                    ax[2, j].plot(x, (h.H_hot + h.H_cold).values / den, color=col, ls='none', marker=MK[dst], ms=3.2,
                                  mec='white', mew=0.4, alpha=0.85)
for j, c in enumerate(CLS):
    ax[0, j].set_title(f'{c}', fontsize=10, loc='left')
    ax[0, j].set_yscale('symlog', linthresh=0.01)
    ax[1, j].set_yscale('symlog', linthresh=0.01)
    ax[2, j].set_yscale('symlog', linthresh=1e-4)
    for i in range(3):
        ax[i, j].axvline(0, color=MUTED, ls=':', lw=0.8)
        ax[i, j].axhline(0, color=MUTED, lw=0.6)
    ax[2, j].set_xlabel('t − t_p  [M]')
ax[0, 0].set_ylabel('(a) ΣΔE_hid / E_th (11+12)\nnet —, hot ··, cold - -')
ax[1, 0].set_ylabel('(b) K_en/K_tr − 1 (11+12)\nmedian, 25–75%')
ax[2, 0].set_ylabel('(c) ΔE_hid carried out per\ninterval / E_th  (○ →14, □ →15, △ →0)')
ax[0, 0].legend(fontsize=8, frameon=False)
fig.suptitle('R-261: the hidden energy of D-ent tracer cells (11, 12), per class', x=0.01, ha='left', fontsize=11)
fig.tight_layout()
fig.savefig(sys.argv[1], dpi=125)
print(sys.argv[1])
