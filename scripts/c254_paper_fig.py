#!/usr/bin/env python3
"""R-254 item 4: the candidate Paper I fallback figure (the fallback rate to pericentre = the mass supply, not accretion).
(a) the three orbits at their latest converged hydro time: P10 (full) = the frozen-in ballistic rate from dM/dE with Schwarzschild
    radial periods (c254_fallback.py b1, dx 1/16 at C2, t_p + 250); P18 and PD07 (partials) = test-particle returns to 3 r_p in the
    hole's field plus the remnant (Hill-softened point mass; c254tp_integrate.py) from t_p + 500 (P18 dx 1/16, PD07 dx 1/32).
    Guides t^-5/3 and t^-9/4.
(b) resolution: P10 b1 at t_p + 50 (1/16, 1/32, 1/64) and PD07's test-particle returns at t_p + 500 (1/16, 1/32).
Units: M_BH = 1e4 M_sun (1 M = 0.0493 s); Mdot_Edd = L_Edd/(eta c^2), eta = 0.1 (2.22e-4 M_sun/yr).
    c254_paper_fig.py OUTDIR"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc'
out = sys.argv[1]
TS = 0.0493                      # s per M
MU = 6.41e12                     # M_sun/yr per (M_BH per M)
EDD = 2.22e-4                    # M_sun/yr, eta = 0.1
MS = 7.47e-5                     # M* in M_BH
TP = {'P10': 459.8, 'P18': 491.1, 'PD07': 508.7}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.8})
COL = {'P10': '#2a78d6', 'P18': '#eb6834', 'PD07': '#1baf7a'}
RES = {'1/16': '#86b6ef', '1/32': '#2a78d6', '1/64': '#173f73'}
cv = pd.read_csv(f'{R}/c254_fallback/c254_curves.csv')
tp = pd.read_csv(f'{R}/c254_testparticle/c254tp_return.csv')


def edd(m_code):                 # Mdot in M_BH per M -> Mdot / Mdot_Edd
    return m_code*MU/EDD


def ballistic(col, orb):         # c254 b1: t is the return time from t_p (in M)
    s = cv[col] > 0
    return cv.t_M[s]*TS, edd(cv[col][s])


def particles(lab):              # c254tp: Mdot in M* per M vs t - t_p
    d = tp[tp.run == lab]; s = d.Mdot_Mstar_per_M > 0
    return d.t_minus_tp_M[s]*TS, edd(d.Mdot_Mstar_per_M[s]*MS)


fig, ax = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
x, y = ballistic('P10_16_t710_b1', 'P10'); ax[0].loglog(x, y, color=COL['P10'], label='P10 (β 1.64, full): ballistic, dx 1/16, t_p+250')
x, y = particles('P18 1/16, + remnant (Hill soft.)'); ax[0].loglog(x, y, color=COL['P18'], label='P18 (β 0.91, partial): test particles + remnant, dx 1/16, t_p+500')
x, y = particles('PD07 1/32, + remnant (Hill soft.)'); ax[0].loglog(x, y, color=COL['PD07'], label='PD07 (β 0.7, partial): test particles + remnant, dx 1/32, t_p+500')
t0 = np.array([3e3, 3e4]); a0 = 2e7*(t0/3e3)**(-5/3)
ax[0].loglog(t0, a0, color=MUTED, ls='--', lw=1.1); ax[0].text(3.2e4, a0[-1], 't$^{-5/3}$', color=MUTED, fontsize=8, va='center')
t1 = np.array([3e3, 3e4]); a1 = 3e6*(t1/3e3)**(-9/4)
ax[0].loglog(t1, a1, color=MUTED, ls=':', lw=1.1); ax[0].text(3.2e4, a1[-1], 't$^{-9/4}$', color=MUTED, fontsize=8, va='center')
ax[0].set_title('(a) fallback to pericentre, the three orbits', loc='left', fontsize=10)
ax[0].legend(frameon=False, fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=1)
for c, r in (('P10_16_t510_b1', '1/16'), ('P10_32_t510_b1', '1/32'), ('P10_64_t506_b1', '1/64')):
    x, y = ballistic(c, 'P10'); ax[1].loglog(x, y, color=RES[r], label=f'P10 ballistic, dx {r}, t_p+50')
for lab, r in (('PD07 1/16, + remnant (Hill soft.)', '1/16'), ('PD07 1/32, + remnant (Hill soft.)', '1/32')):
    x, y = particles(lab); ax[1].loglog(x, y, color=RES[r], ls='--', label=f'PD07 test particles, dx {r}, t_p+500')
ax[1].set_title('(b) resolution', loc='left', fontsize=10)
ax[1].legend(frameon=False, fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2)
for a in ax:
    a.set_xlabel('t − t_p  [s]   (M_BH = 1e4 M$_\\odot$)'); a.set_xlim(1e2, 5e4); a.set_ylim(1e2, 1e8)
ax[0].set_ylabel('Ṁ$_{\\rm fb}$ / Ṁ$_{\\rm Edd}$  (η = 0.1)')
fig.suptitle('Predicted fallback rate to pericentre (mass supply, not accretion)', x=0.01, y=0.995, ha='left', fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(f'{out}/c254_paper_fallback.png', dpi=140); fig.savefig(f'{out}/c254_paper_fallback.pdf')
print('wrote', out)
