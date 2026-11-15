#!/usr/bin/env python3
"""c264 (user 2026-11-15): L32T (relative tracker, remnant at dx 1/32) against P18T (relative tracker, dx 1/16) and
P18_r82 (grid frame, dx 1/16), r_p = 18, all against t - t_p (t_p = 491.1). Offline, read-only.
  history lines (user.hst): rho-max [3] / rho_c, fc_mass [14] / M*; L32T is prefixed by its parent L32_P18_r82 (t <= 541.1)
      and P18T by its parent P18_r82 (t <= 611.1)
  markers (c258_remnant_cmp.py rows, full-3D restarts): M_rem, M_Hsc, K_in50, K_Hill, R50, R90, the DE code 12 + 13 mass
      share inside r_H (c192's codes: 12 safeguard, 13 tracer after a floored/failed energy C2P or tau <= 0)
  Colour = finest dx on the remnant, line style = frame (solid tracker, dashed grid frame).
    c264_l32t_plots.py OUTDIR C258CSV [C258CSV ...]"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402

R = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc'
HST = 'cfc_tde_wd_imbh_parabolic_tracker.user.hst'
TP, RC, MSTAR = 491.1, 3.24e-4, 7.47e-5
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.6})
# label, c258 label prefixes, history pieces [(run dir, t_max)], colour, line style
RUNS = [('P18_r82 (grid frame, dx 1/16)', ('P18r82_',), [(f'{R}/c203_chain/P18_r82', 1e9)], '#86b6ef', '--'),
        ('P18T (tracker, dx 1/16)', ('P18T_',), [(f'{R}/c203_chain/P18_r82', 611.1), (f'{R}/c252_trial/P18T', 1e9)],
         '#86b6ef', '-'),
        ('L32T (tracker, dx 1/32)', ('L32P18_', 'L32T_'),
         [(f'{R}/c203_chain/L32_P18_r82', 541.1), (f'{R}/c259_L32T/L32T', 1e9)], '#173f73', '-')]


def hst(pieces):
    out, t0 = [], -1e9
    for d, tmax in pieces:
        a = np.loadtxt(f'{d}/{HST}', comments='#', usecols=(0, 2, 13))
        a = a[(a[:, 0] > t0) & (a[:, 0] <= tmax + 1e-9)]
        out.append(a)
        t0 = tmax
    a = np.concatenate(out)
    _, i = np.unique(a[::-1, 0], return_index=True)   # restarts repeat times: keep the last write
    return a[::-1][i]


def main():
    od = sys.argv[1]
    tab = pd.concat([pd.read_csv(f) for f in sys.argv[2:]], ignore_index=True)
    tab['de_12_13'] = tab.get('de_12', 0) + tab.get('de_13', 0)
    fig, ax = plt.subplots(2, 3, figsize=(13, 7.4), sharex=True)
    ax = ax.ravel()
    for lab, pre, pieces, col, ls in RUNS:
        h = hst(pieces)
        x = h[:, 0] - TP
        ax[0].semilogy(x, h[:, 1] / RC, color=col, ls=ls, lw=1.2, label=lab)
        trk = h[:, 0] > (pieces[0][1] if len(pieces) > 1 else 1e9)   # fc_mass only where the tracker is on
        if trk.any():
            ax[1].plot(x[trk], h[trk, 2] / MSTAR, color=col, ls=':', lw=1.0)
        r = tab[tab.label.str.startswith(pre)].sort_values('t_minus_tp')
        if r.empty:
            continue
        kw = dict(color=col, ls=ls, marker='o', ms=4.5, mec='white', mew=0.8)
        ax[1].plot(r.t_minus_tp, r.M_rem, **kw)
        ax[2].plot(r.t_minus_tp, r.M_Hsc, **kw)
        ax[3].plot(r.t_minus_tp, r.K_in50, **kw)
        ax[3].plot(r.t_minus_tp, r.K_Hill, **dict(kw, marker='s', ms=3.5, lw=0.9, alpha=0.7))
        ax[4].plot(r.t_minus_tp, r.R50, **kw)
        ax[4].plot(r.t_minus_tp, r.R90, **dict(kw, marker='s', ms=3.5, lw=0.9, alpha=0.7))
        ax[5].plot(r.t_minus_tp, 100 * r.de_12_13, **kw)
    titles = ['(a) ρ_max / ρ_c  (history)', '(b) M_rem / M*  (bound set, ●);  tracker fc_mass (dotted)',
              '(c) self-consistent Hill mass M_Hsc / M*', '(d) ⟨K/K0⟩: innermost 50% (●), inside r_H (■)',
              '(e) R50 (●), R90 (■) of the bound set  [M]', '(f) DE codes 12 + 13, mass share inside r_H  [%]']
    for a, t in zip(ax, titles):
        a.set_title(t, fontsize=9, loc='left')
        a.axvline(0, color=MUTED, ls=':', lw=0.8)
        a.set_xlim(-120, 1020)
    for a in ax[3:]:
        a.set_xlabel('t − t_p  [M]')
    ax[3].axhline(1, color=MUTED, ls='--', lw=0.8)
    hs, ls_ = ax[0].get_legend_handles_labels()
    fig.suptitle('L32T vs P18T vs P18_r82 (r_p = 18, t_p = 491.1): remnant to t_p + 1000', x=0.01, y=0.995,
                 ha='left', fontsize=11)
    fig.legend(hs, ls_, loc='upper center', bbox_to_anchor=(0.5, 0.968), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    os.makedirs(od, exist_ok=True)
    fig.savefig(f'{od}/c264_L32T_remnant.png', dpi=130)
    plt.close(fig)
    print(f'{od}/c264_L32T_remnant.png')


if __name__ == '__main__':
    main()
