#!/usr/bin/env python3
"""c177 companion figure (user request 2026-11-03): B.1 Bb10 (kcfc_bg, DE off) vs Bd10 (kcfc_bg + DE, production settings),
with Bc10 (kcfc, C-167) as the kcfc reference, by debris class, plus Bd10's dual-energy branch maps.
Inputs: census.csv of c169_b1_ana.py dumps (Bb10, Bd10; and c161_b1's for Bc10) and de_bands.csv of c177_de_bands.py (Bd10).
Panels (time in M, periapsis t_p = 135.9 dashed):
  (a) core rho >= 0.1 rho_c: rho dV-weighted K/K0 - 1 [%]      (b) outer layers 0.01-0.1 rho_c (from the d01/d1 sums) [%]
  (c) Hill remnant: K/K0 - 1 [%]                                (d) stream (1e-6-0.01 rho_c outside r_H): K/K0 - 1 [%] (log)
  (e) masses: core, remnant, stream [M*]                        (f) Bd10: shock-flagged mass share per class
  (g) Bd10: energy-branch (code 4) and guard (code 2) shares     (h) Bd10: K_en/K_tr - 1 in flagged dense cells, median and 5-95%
  (i) the DE branch shares of the core over time (stacked: entropy 1 / guard 2 / energy 4 / other)
    c177_b1kbg_classes.py OUT.png CENSUS_BbBd.csv CENSUS_Bc.csv DE_BANDS.csv"""
import csv
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

TP, MS = 135.9, 7.5e-5
ARMS = [('Bc10', 'Bc10 (kcfc, DE off)', '#1baf7a', ':'), ('Bb10', 'Bb10 (kcfc_bg, DE off)', '#eb6834', '-'),
        ('Bd10', 'Bd10 (kcfc_bg + DE)', '#8a3ffc', '-')]
CLS = [('core', 'core >= 0.1 rho_c', '#2a78d6'), ('outer', 'outer 0.01-0.1 rho_c', '#eb6834'),
       ('rem', 'Hill remnant', '#1baf7a'), ('stream', 'stream', '#8a8f98')]


def census(paths):
    d = {}
    for p in paths:
        for r in csv.DictReader(open(p)):
            d.setdefault(r['arm'], []).append({k: (float(v) if k != 'arm' else v) for k, v in r.items()})
    return {k: sorted(v, key=lambda x: x['t']) for k, v in d.items()}


def main():
    out, cbd, cbc, deb = sys.argv[1:5]
    C = census([cbd, cbc])
    D = {}
    for r in csv.DictReader(open(deb)):
        D.setdefault(r['cls'], []).append({k: (float(v) if k != 'cls' else v) for k, v in r.items()})
    fig, ax = plt.subplots(3, 3, figsize=(17, 13))
    ax = ax.ravel()
    for a in ax:
        a.axvline(TP, color='0.5', ls='--', lw=1)
        a.grid(alpha=0.25)
    for arm, lab, col, ls in ARMS:
        if arm not in C:
            continue
        R = C[arm]; t = np.array([r['t'] for r in R])
        core = np.array([100*r['d1_K'] if r['d1_M'] > 0 else np.nan for r in R])
        outer = np.array([100*(r['d01_M']*r['d01_K'] - (r['d1_M']*r['d1_K'] if r['d1_M'] > 0 else 0.0))
                          / max(r['d01_M'] - r['d1_M'], 1e-300) for r in R])
        ax[0].plot(t, core, color=col, ls=ls, lw=2, label=lab)
        ax[1].plot(t, outer, color=col, ls=ls, lw=2, label=lab)
        ax[2].plot(t, [100*r['rem_K'] for r in R], color=col, ls=ls, lw=2, label=lab)
        ax[3].plot(t, [100*r['st_K'] for r in R], color=col, ls=ls, lw=2, label=lab)
        for q, mk in (('d1_M', '-'), ('rem_M', '--'), ('st_M', ':')):
            ax[4].plot(t, [r[q]/MS for r in R], color=col, ls=mk, lw=2 if mk == '-' else 1.5,
                       label=f'{arm} ' + {'d1_M': 'core', 'rem_M': 'remnant', 'st_M': 'stream'}[q])
    for c, lab, col in CLS:
        R = [r for r in D.get(c, []) if r['M'] > 0]
        if not R:
            continue
        t = np.array([r['t'] for r in R])
        ax[5].plot(t, [100*r['flagged'] for r in R], 'o-', color=col, lw=2, ms=4, label=lab)
        ax[6].plot(t, [100*r['br4'] for r in R], 'o-', color=col, lw=2, ms=4, label=f'{lab}: energy (4)')
        ax[6].plot(t, [100*r['br2'] for r in R], 'x--', color=col, lw=1.2, ms=5, label=f'{lab}: guard (2)')
        if c in ('core', 'outer'):
            m = np.array([r['dk50'] for r in R]); lo = np.array([r['dk05'] for r in R]); hi = np.array([r['dk95'] for r in R])
            ok = np.isfinite(m)
            ax[7].plot(t[ok], m[ok], 'o-', color=col, lw=2, ms=4, label=f'{lab}: median')
            ax[7].fill_between(t[ok], lo[ok], hi[ok], color=col, alpha=0.18, label=f'{lab}: 5-95%')
    ax[7].axhline(-1e-3, color='k', lw=1, ls='-.', label='guard tol -1e-3')
    R = [r for r in D.get('core', []) if r['M'] > 0]
    if R:
        t = np.array([r['t'] for r in R])
        b1 = np.array([r['br1'] for r in R]); b2 = np.array([r['br2'] for r in R]); b4 = np.array([r['br4'] for r in R])
        oth = np.clip(1.0 - b1 - b2 - b4, 0, 1)
        ax[8].stackplot(t, 100*b1, 100*b2, 100*b4, 100*oth, colors=['#2a78d6', '#1baf7a', '#eb6834', '#c3c7cf'],
                        labels=['entropy, unflagged (1)', 'entropy by guard (2)', 'energy, flagged (4)', 'other (0/3/5)'])
    ax[3].set_yscale('log')
    ax[4].set_yscale('log')
    ax[7].set_yscale('symlog', linthresh=1e-5)
    titles = ['(a) core >= 0.1 rho_c: <K/K0 - 1> [%]', '(b) outer layers 0.01-0.1 rho_c: <K/K0 - 1> [%]',
              '(c) Hill remnant: <K/K0 - 1> [%]', '(d) stream: <K/K0 - 1> [%]', '(e) mass [M*]: core / remnant / stream',
              '(f) Bd10: shock-flagged mass share [%]', '(g) Bd10: energy-branch (4) and guard (2) mass shares [%]',
              '(h) Bd10: K_en/K_tr - 1 in flagged dense cells', '(i) Bd10 core: branch mass shares [%]']
    for a, tt in zip(ax, titles):
        a.set_title(tt, fontsize=10)
        a.set_xlabel('t [M]')
        a.legend(fontsize=7, loc='best')
    fig.suptitle('B.1 by debris class: Bb10 (kcfc_bg, DE off) vs Bd10 (kcfc_bg + DE, production settings), Bc10 (kcfc) dotted; '
                 'DE branch maps from the 3D DE dumps every 12 M; dashed grey = periapsis', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=95)
    print('wrote', out)


if __name__ == '__main__':
    main()
