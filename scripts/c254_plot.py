#!/usr/bin/env python3
"""c254: the predicted fallback rate (Schwarzschild radial period from t_p, c254_fallback.py's b1) per orbit and resolution, with
the 1/16 run at its latest time dashed and t^-5/3, t^-9/4 guides; physical axes for M_BH = 1e4 M_sun.   c254_plot.py DIR"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
d = sys.argv[1]; z = np.load(f'{d}/c254_curves.npz'); t = z['t']
TU, MU = 0.049255, 6.407e12; EDD = 2.22e-4
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e4e3df', 'axes.edgecolor': '#52514e'})
RES = {'16': '#86b6ef', '32': '#2a78d6', '64': '#173f73'}
COL = [('P10 (β 1.64)', [('P10_16_t510', '16', '-'), ('P10_32_t510', '32', '-'), ('P10_64_t506', '64', '-'), ('P10_16_t910', '16', '--')]),
       ('P18 (β 0.91)', [('P18_16_t541', '16', '-'), ('P18_32_t541', '32', '-'), ('P18_64_t541', '64', '-'), ('P18_16_t991', '16', '--')]),
       ('PD07 (β 0.7)', [('PD07_16_t1009', '16', '-'), ('PD07_32_t1009', '32', '-'), ('PD07_16_t572', '16', ':')])]
fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
for x, (ttl, runs) in zip(ax, COL):
    for lab, r, ls in runs:
        y = z[f'{lab}_b1']; s = y > 0
        x.loglog(t[s]*TU, y[s]*MU/EDD, color=RES[r], ls=ls, lw=1.6, label=f'dx 1/{r}, from t_p+{lab.split("_t")[1]} '.replace(f'+{lab.split("_t")[1]}', '') + f'(dump t={lab.split("_t")[1]})')
    t0 = 3e4; y0 = z[f'{runs[0][0]}_b1'][np.argmin(abs(t - t0))]*MU/EDD
    tt = np.logspace(np.log10(5e4), 6.5, 10)
    x.loglog(tt*TU, y0*(tt/t0)**(-5/3), color='#52514e', lw=0.8, ls='-.', label='t^-5/3')
    x.loglog(tt*TU, y0*(tt/t0)**(-9/4), color='#52514e', lw=0.8, ls=(0, (1, 1)), label='t^-9/4')
    x.set_title(ttl, loc='left'); x.set_xlabel('t − t_p  [s]  (M_BH = 1e4 M_⊙)'); x.set_xlim(200, 2e5); x.set_ylim(1e3, 5e7)
    x.legend(frameon=False, fontsize=7, loc='lower left')
ax[0].set_ylabel('predicted fallback Ṁ_fb / Ṁ_Edd  (η = 0.1)')
fig.suptitle('Frozen-in ballistic fallback to pericentre (Schwarzschild radial periods of the bound debris)', x=0.01, ha='left')
fig.tight_layout(); fig.savefig(f'{d}/c254_fallback.png', dpi=130)
