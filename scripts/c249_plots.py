#!/usr/bin/env python3
"""c249 (user 2026-11-08): analysis plots for the new runs (P18 production, PD07 beta 0.7, PD07_L32 beta 0.7 dx 1/32,
PD04 beta 0.4), all against t - t_p:
  c249_overview.png: (a) M(d > 1.5 R*) [c206_outflow]  (b) M_Hill(0) - M_Hill [c142_remnant]  (c) rho_max/rho_c with the frame
                     controller's frame_track_rho = 0.1 rho_c  (d) core K/K0 [c160_census, d >= 0.1 rho_c core]
  c249_PD04_dmde.png: PD04 lost mass (d > 1.5 R*), (1/M_lost) dM/d((E - E_rem)/dE) at four times [c247_dmde_lost]
    c249_plots.py OUTDIR"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc'
A, P4 = f'{R}/c249_ana', f'{R}/c207_pd/ana_PD04'
out = sys.argv[1]
# categorical slots 1-4 (dataviz reference palette, light), fixed per run
RUNS = [('P18', 'P18 (β = 1, r_p 18)', 491.1, '#2a78d6', A, 'P18'),
        ('PD07', 'PD07 (β = 0.7)', 508.7, '#eb6834', A, 'PD07'),
        ('PD07_L32', 'PD07_L32 (β = 0.7, dx 1/32)', 508.7, '#1baf7a', A, 'PD07_L32'),
        ('PD04', 'PD04 (β = 0.4)', 533.5, '#eda100', P4, 'PD04')]
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 2})


def rd(f):
    return pd.read_csv(f) if os.path.exists(f) and os.path.getsize(f) > 0 else None


fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
(a, b), (c, d) = ax
for key, lab, tp, col, D, tag in RUNS:
    o = rd(f'{D}/outflow_{tag}.csv')
    if o is not None:
        s = o['M_gt1.5'] > 0
        a.semilogy(o.t[s] - tp, o['M_gt1.5'][s], color=col, label=lab)
    r = rd(f'{D}/c142_B.csv' if key == 'PD04' else f'{D}/c142_{tag}/c142_B.csv')
    if r is not None:
        r = r.sort_values('t'); y = r.M_Hill.iloc[0] - r.M_Hill
        b.semilogy(r.t - tp, y.clip(lower=1e-6), color=col, marker='o', ms=4, label=lab)
    cz = rd(f'{D}/census_{tag}.csv')
    if cz is not None:
        c.semilogy(cz.t_minus_tp, cz.rhomax, color=col, label=lab)
        d.plot(cz.t_minus_tp, cz.core_K, color=col, label=lab)
c.axhline(0.1, color=MUTED, lw=1, ls='--'); c.text(c.get_xlim()[0], 0.105, ' frame_track_rho = 0.1 ρ_c', color=MUTED, fontsize=8, va='bottom')
a.set_title('(a) outflow  M(d > 1.5 R*) / M*', loc='left', fontsize=10)
b.set_title('(b) mass lost from the Hill sphere  [M_Hill(0) − M_Hill(t)] / M*', loc='left', fontsize=10)
c.set_title('(c) densest cell  ρ_max / ρ_c', loc='left', fontsize=10)
d.set_title('(d) core  ⟨K/K0⟩  (ρ ≥ 0.1 ρ_c)', loc='left', fontsize=10)
for x in ax.flat:
    x.axvline(0, color=MUTED, lw=0.8, ls=':')
for x in (c, d):
    x.set_xlabel('t − t_p  [M]')
h, l = a.get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=4, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(f'{out}/c249_overview.png', dpi=130); plt.close(fig)

z = np.load(f'{P4}/dmde_lost_PD04.npz')
e = z['edges']; mid = 0.5*(e[1:] + e[:-1])
# sequential single hue (blue ramp), light -> dark with time
seq = ['#9cc3f0', '#5f9ee6', '#2a78d6', '#174e94']
fig, x = plt.subplots(figsize=(7, 4.2))
i = 0
while f't_{i}' in z:
    x.step(mid, z[f'H_{i}'], where='mid', color=seq[i % 4],
           label=f"t − t_p = {float(z[f't_{i}']) - 533.5:+.0f}   M_lost = {float(z[f'M_lost_{i}']):.1e} M*")
    i += 1
x.axvline(0, color=MUTED, lw=0.8, ls=':')
x.set_xlim(-1.5, 1.5); x.set_xlabel('(E − E_rem) / ΔE,   ΔE = R*/r_t²'); x.set_ylabel('(1/M_lost) dM/d(E/ΔE)')
x.set_title('PD04 (β = 0.4): energy of the gas beyond 1.5 R*', loc='left', fontsize=10)
x.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(f'{out}/c249_PD04_dmde.png', dpi=130); plt.close(fig)
print('wrote', out)
