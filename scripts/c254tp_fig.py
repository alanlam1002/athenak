#!/usr/bin/env python3
"""c254 (R-254 items 2-3): the test-particle return-rate figure for the partials. Each curve = the full mass-weighted sample for
t < 1.2 t_cut joined to the gap-window sample (E/dE in [-0.45, 0.1], c254tp_integrate.py --ewin) beyond it, where every return
comes from the window (t_cut = the full run's 1%-of-peak time, from summary_*.csv). Log bins, 16 per decade.
    c254tp_fig.py OUT.png OUT.csv"""
import sys, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/c254_testparticle'
MS = 7.47e-5
out, outc = sys.argv[1], sys.argv[2]
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.8})
tcut = {}
for f in ('summary_500.csv', 'summary_res.csv'):
    for r in csv.DictReader(open(f'{D}/{f}')):
        tcut[r['run']] = float(r['t_cut'])
# (label, full npz, gap npz, t_cut key, colour, linestyle)
RUNS = [('PD07 1/16, ballistic', 'PD07_500_none', 'gPD07_500_none', 'PD07_none', '#1baf7a', ':'),
        ('PD07 1/16, + remnant (Hill soft.)', 'PD07_500_hill', 'gPD07_500_hill', 'PD07_hill', '#1baf7a', '-'),
        ('PD07 1/32, + remnant (Hill soft.)', 'PD07_L32_500_hill', 'gPD07_L32_500_hill', 'PD07L32_hill', '#0e6b4a', '--'),
        ('P18 1/16, ballistic', 'P18_500_none', 'gP18_500_none', 'P18_none', '#eb6834', ':'),
        ('P18 1/16, + remnant (Hill soft.)', 'P18_500_hill', 'gP18_500_hill', 'P18_hill', '#eb6834', '-')]
fig, ax = plt.subplots(figsize=(7.6, 4.8))
rows = []
for lab, ff, fg, key, c, ls in RUNS:
    tc_ = tcut[key]
    def rate(fn, sel):
        R = np.load(f'{D}/{fn}.npz'); tp, W = float(R['tp']), float(R['W'])
        tr = R['t_ev'][R['status'] == 1] - tp
        e = np.logspace(3.5, np.log10(10*float(R['tmb_ref'])), 16*2 + 1)
        e = np.logspace(np.log10(5e3), np.log10(10*float(R['tmb_ref'])), int(16*np.log10(10*float(R['tmb_ref'])/5e3)) + 1)
        n, _ = np.histogram(tr, bins=e); t = np.sqrt(e[1:]*e[:-1])
        md = n*W/np.diff(e)/MS
        k = sel(t) & (n >= 5)
        return t[k], md[k]
    t1, m1 = rate(ff, lambda t: t < 1.2*tc_)
    t2, m2 = rate(fg, lambda t: t >= 1.2*tc_)
    t, m = np.r_[t1, t2], np.r_[m1, m2]
    ax.loglog(t, m, color=c, ls=ls, label=lab)
    rows += [(lab, a, b) for a, b in zip(t, m)]
tt = np.array([5e4, 1.7e5])
ax.loglog(tt, 2e-7*(tt/5e4)**(-5/3), color=MUTED, lw=1, ls='-.', label='t^-5/3')
ax.loglog(tt, 2e-7*(tt/5e4)**(-9/4), color=MUTED, lw=1, ls=(0, (1, 2)), label='t^-9/4')
ax.set_xlabel('t − t_p  [M]   (t_mb = 1.76e4 M = 870 s at 1e4 M_sun)'); ax.set_ylabel('return rate to 3 r_p  [M*/M]')
ax.set_title('Test-particle return of the partials\' debris from t_p + 500 (Schwarzschild + remnant)', loc='left', fontsize=10)
ax.set_ylim(1e-11, 5e-5); ax.legend(frameon=False, fontsize=7.5, loc='lower left')
fig.tight_layout(); fig.savefig(out, dpi=140)
with open(outc, 'w') as fo:
    fo.write('run,t_minus_tp_M,Mdot_Mstar_per_M\n')
    for r in rows:
        fo.write(f'"{r[0]}",{r[1]:.6g},{r[2]:.6g}\n')
print('wrote', out, outc)
