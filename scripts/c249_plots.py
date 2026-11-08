#!/usr/bin/env python3
"""c249 (user 2026-11-08): analysis figures for every run on the current production setting (proj/tde 682699f4,
binary 5a4ff8ff; D-ent conditioning, kmax 1000, rho_switch 3.24e-10, q_delta 0.05), all against t - t_p.
  c249_orbits.png  the orbit series at dx 1/16: P10 (r_p 10), P18 (r_p 18), PD07 (beta 0.7), PD04 (beta 0.4)
  c249_ladder.png  the resolution ladder, one column per orbit (P18, P10, PD07), dx 1/16 -> 1/32 -> 1/64 (+ D64s)
  Rows / panels: (a) M(d > 1.5 R*) [c206_outflow]  (b) M_Hill(0) - M_Hill [c142_remnant]
                 (c) rho_max/rho_c with frame_track_rho = 0.1 rho_c  (d) core <K/K0>, rho >= 0.1 rho_c [c160_census]
  c249_PD04_dmde.png  PD04 lost mass (d > 1.5 R*), (1/M_lost) dM/d((E - E_rem)/dE) at four times [c247_dmde_lost]
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
TP = {'P10': 459.8, 'P18': 491.1, 'PD07': 508.7, 'PD04': 533.5}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.8})


def rd(f):
    try:
        return pd.read_csv(f) if os.path.exists(f) and os.path.getsize(f) > 0 else None
    except Exception:   # noqa: a half-written CSV
        return None


def data(tag):
    D = P4 if tag == 'PD04' else A
    return (rd(f'{D}/outflow_{tag}.csv'), rd(f'{D}/c142_B.csv' if tag == 'PD04' else f'{D}/c142_{tag}/c142_B.csv'),
            rd(f'{D}/census_{tag}.csv'))


def draw(axs, tag, tp, kw):
    o, r, cz = data(tag)
    if o is not None and 'M_gt1.5' in o:
        s = o['M_gt1.5'] > 0
        axs[0].semilogy(o.t[s] - tp, o['M_gt1.5'][s], **kw)
    if r is not None and len(r):
        r = r.sort_values('t'); y = (r.M_Hill.iloc[0] - r.M_Hill).clip(lower=1e-7)
        axs[1].semilogy(r.t - tp, y, marker='o', ms=3, **kw)
    if cz is not None and len(cz):
        axs[2].semilogy(cz.t_minus_tp, cz.rhomax, **kw)
        axs[3].plot(cz.t_minus_tp, cz.core_K, **kw)


ROWS = ['outflow  M(d > 1.5 R*) / M*', 'Hill-sphere loss  [M_Hill(0) − M_Hill] / M*', 'densest cell  ρ_max / ρ_c',
        'core ⟨K/K0⟩  (ρ ≥ 0.1 ρ_c)']


def finish(axs):
    axs[2].axhline(0.1, color=MUTED, lw=1, ls='--')
    axs[2].annotate('frame_track_rho', (0.01, 0.1), xycoords=('axes fraction', 'data'), color=MUTED, fontsize=7,
                    va='bottom')
    for x in axs:
        x.axvline(0, color=MUTED, lw=0.8, ls=':')


# ---- 1. orbit series at dx 1/16 (categorical slots 1-4, fixed per run)
ORB = [('P10', 'P10 (r_p 10, β 1.64)', '#2a78d6'), ('P18', 'P18 (r_p 18, β 0.91)', '#eb6834'),
       ('PD07', 'PD07 (r_p 23.4, β 0.7)', '#1baf7a'), ('PD04', 'PD04 (r_p 41, β 0.4)', '#eda100')]
fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
axs = list(ax.flat)
for tag, lab, col in ORB:
    draw(list(axs), tag, TP[tag], dict(color=col, label=lab))
finish(list(axs))
for x, t, l in zip(axs, ROWS, 'abcd'):
    x.set_title(f'({l}) {t}', loc='left', fontsize=10)
for x in ax[1]:
    x.set_xlabel('t − t_p  [M]')
h, l = axs[0].get_legend_handles_labels()
fig.suptitle('Production setting, orbit series at dx 1/16', x=0.01, y=0.995, ha='left', fontsize=11)
fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 0.965), ncol=4, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(f'{out}/c249_orbits.png', dpi=130); plt.close(fig)

# ---- 2. resolution ladder: columns = orbit, colour = resolution (one hue, light -> dark)
RES = {'1/16': '#86b6ef', '1/32': '#2a78d6', '1/64': '#173f73'}
COLS = [('P18 (r_p 18)', 'P18', [('P18', '1/16'), ('L32_P18', '1/32'), ('L64_P18', '1/64'), ('D64s', 'D64s')]),
        ('P10 (r_p 10)', 'P10', [('P10', '1/16'), ('L32_P10', '1/32'), ('L64_P10', '1/64')]),
        ('PD07 (β 0.7)', 'PD07', [('PD07', '1/16'), ('PD07_L32', '1/32')])]
fig, ax = plt.subplots(4, 3, figsize=(13, 11), sharex='col')
for c, (title, orb, runs) in enumerate(COLS):
    col = ax[:, c]
    for tag, res in runs:
        kw = dict(color=RES['1/64'], ls=':', label='D64s (1/64, t ≤ 20)') if res == 'D64s' else \
            dict(color=RES[res], label=f'dx {res}')
        draw(list(col), tag, TP[orb], kw)
    finish(list(col))
    col[0].set_title(title, fontsize=10, loc='left')
    col[-1].set_xlabel('t − t_p  [M]')
    if c == 0:
        for x, t in zip(col, ROWS):
            x.set_ylabel(t, fontsize=8)
hs, ls = [], []
for x in ax[0]:
    for h, l in zip(*x.get_legend_handles_labels()):
        if l not in ls:
            hs.append(h); ls.append(l)
fig.suptitle('Production setting, resolution ladder (finest dx on the star)', x=0.01, y=0.995, ha='left', fontsize=11)
fig.legend(hs, ls, loc='upper center', bbox_to_anchor=(0.5, 0.975), ncol=4, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.945))
fig.savefig(f'{out}/c249_ladder.png', dpi=120); plt.close(fig)

# ---- 3. PD04 lost-mass energy distribution
z = np.load(f'{P4}/dmde_lost_PD04.npz')
e = z['edges']; mid = 0.5*(e[1:] + e[:-1])
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
