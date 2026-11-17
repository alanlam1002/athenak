#!/usr/bin/env python3
"""c265 (user 2026-11-16): figure from c265_bernoulli.py's CSV (one run). (a) bound masses: Newtonian bound_mass, the
self-consistent Hill mass, the metric Bernoulli sets (h = 1 and h; free and rotating frame); (b) the core's Bernoulli
constant against the saddle, (<ln h + Psi_corr> - Psi_s) / (Psi_s - Psi_c) over the innermost 50% and the 50-90% shell of
the Newtonian set; (c) the Newtonian set's mass share bound if at rest (fstatic: with h, fstatic_h1: without).
    c265_bernoulli_plot.py CSV OUT.png [TITLE]"""
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd              # noqa: E402

INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.6})
d = pd.read_csv(sys.argv[1]).sort_values('t_minus_tp')
x = d.t_minus_tp
fig, ax = plt.subplots(1, 3, figsize=(13, 4.0))
kw = dict(marker='o', ms=4, mec='white', mew=0.7)
ax[0].plot(x, d.M_N, color='#52514e', ls='--', label='Newtonian bound set (remnant.py)', **kw)
ax[0].plot(x, d.M_Hsc, color='#86b6ef', label='self-consistent Hill mass M_Hsc', **kw)
ax[0].plot(x, d.free_h1_M, color='#173f73', label='Bernoulli h = 1, free-fall frame', **kw)
ax[0].plot(x, d.rot_h1_M, color='#173f73', ls=':', label='Bernoulli h = 1, rotating frame', **kw)
ax[0].plot(x, d.free_h_M, color='#c2410c', label='Bernoulli with h, free-fall frame', **kw)
ax[0].set_ylabel('M / M*')
ax[0].legend(fontsize=7.5, frameon=False)
ax[1].plot(x, d.free_h_dhyd / d.free_h_well, color='#c2410c', label='innermost 50%', **kw)
ax[1].plot(x, d.free_h_dhyd_out / d.free_h_well, color='#c2410c', ls='--', label='50-90% shell', **kw)
ax[1].set_yscale('log' if (d.free_h_dhyd > 0).all() and (d.free_h_dhyd_out > 0).all() else 'symlog')
ax[1].axhline(1, color=MUTED, lw=0.8, ls=':')
ax[1].legend(fontsize=7.5, frameon=False)
ax[2].plot(x, d.free_h_fstatic, color='#c2410c', label='with h', **kw)
ax[2].plot(x, d.free_h_fstatic_h1, color='#173f73', label='h = 1', **kw)
ax[2].set_ylim(0, 1.02)
ax[2].legend(fontsize=7.5, frameon=False)
for a, t in zip(ax, ['(a) bound mass', '(b) (⟨ln h + Ψ_corr⟩ − Ψ_s) / well depth', '(c) share of the Newtonian set bound at rest']):
    a.set_title(t, fontsize=9, loc='left')
    a.set_xlabel('t − t_p  [M]')
fig.suptitle(sys.argv[3] if len(sys.argv) > 3 else 'metric Bernoulli criterion', x=0.01, ha='left', fontsize=10.5)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=130)
print(sys.argv[2])
