#!/usr/bin/env python3
"""c157 (user request 2026-10-30): analysis figure for the B.1 arms B1V (Valencia + analytic_bg) and B1T (comoving
killing_t1 + analytic_bg), both from B1N_cell's t = 41.78 restart, with B1N_cell (Valencia + cell) as reference.
Panels (dense bands as c122_b1.py: rho >= 0.01 / 0.1 rho_c, mass weights rho dV_coord):
  (a) <K/K0 - 1>; (b) warm fraction (K/K0 - 1 > +5%); (c) cold fraction (< -5%)   -- c154_b1_dense.csv (C-143)
  (d) rho_max/rho_c from the user history (every row)
  (e) cold dense mass fraction (rho >= rho_c, K/K0 < 0.5) of the dense mass        -- c155_coldspot.csv (C-144)
  (f) K/K0 percentiles 1 and 50 of the dense gas                                   -- c155_coldspot.csv
    c157_b1vt_plots.py OUT.png DENSE.csv COLDSPOT.csv RUNROOT"""
import csv
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

ARMS = [('B1N_cell', 'c122_b1/B1N_cell', 'B1N_cell (Valencia, cell)', '#2a78d6'),
        ('B1V', 'c154_kdtg/B1V', 'B1V (Valencia, analytic_bg)', '#eb6834'),
        ('B1T', 'c154_kdtg/B1T', 'B1T (comoving t1, analytic_bg)', '#1baf7a')]
RHOC = 3.24e-4
T_RST = 41.78


def rows(path):
    out = {}
    for r in csv.DictReader(open(path)):
        out.setdefault(r['arm'], []).append(r)
    return out


def col(rs, k):
    v = [(float(r['t']), float(r[k]) if r[k] not in ('', 'nan') else np.nan) for r in rs]
    v.sort()
    return np.array([a for a, _ in v]), np.array([b for _, b in v])


def hst(d):
    f = f'{d}/cfc_tde_wd_imbh_parabolic_tracker.user.hst'
    a = np.loadtxt(f)
    return a[:, 0], a[:, 2]


def main():
    out, dense, cold, root = sys.argv[1:5]
    D, C = rows(dense), rows(cold)
    fig, ax = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    ax = ax.ravel()
    for arm, rel, lab, c in ARMS:
        if arm in D:
            for k, ls, nm in (('d01', '-', r'$\rho\geq0.01\rho_c$'), ('d1', '--', r'$\rho\geq0.1\rho_c$')):
                t, y = col(D[arm], k + '_mean')
                ax[0].plot(t, y, ls, color=c, lw=1.6, label=lab if ls == '-' else None)
                t, y = col(D[arm], k + '_warm5')
                ax[1].plot(t, y, ls, color=c, lw=1.6)
                t, y = col(D[arm], k + '_cold5')
                ax[2].plot(t, y, ls, color=c, lw=1.6)
        th, rm = hst(f'{root}/{rel}')
        m = th >= (0.0 if arm == 'B1N_cell' else T_RST)
        ax[3].plot(th[m], rm[m]/RHOC, color=c, lw=1.4)
        if arm in C:
            t, y = col(C[arm], 'cold_mass_frac')
            ax[4].plot(t, 100*y, 'o-', color=c, lw=1.6, ms=4)
            for k, ls in (('K_p1', ':'), ('K_p50', '-')):
                t, y = col(C[arm], k)
                ax[5].plot(t, y, ls, color=c, lw=1.6)
    ax[0].plot([], [], 'k-', lw=1.2, label=r'$\rho\geq0.01\rho_c$')
    ax[0].plot([], [], 'k--', lw=1.2, label=r'$\rho\geq0.1\rho_c$')
    ax[5].plot([], [], 'k-', lw=1.2, label='median')
    ax[5].plot([], [], 'k:', lw=1.2, label='1st percentile')
    titles = [r'(a) dense $\langle K/K_0-1\rangle$', r'(b) warm fraction ($K/K_0-1>+5\%$)',
              r'(c) cold fraction ($K/K_0-1<-5\%$)', r'(d) $\rho_{\max}/\rho_c$ (history)',
              r'(e) cold dense mass [% of $\rho\geq0.01\rho_c$]' '\n' r'($\rho\geq\rho_c$, $K/K_0<0.5$)',
              r'(f) dense-gas $K/K_0$ percentiles']
    for a, ttl in zip(ax, titles):
        a.set_title(ttl, fontsize=10, loc='left')
        a.set_xlabel('t [M]')
        a.grid(alpha=0.25, lw=0.5)
        a.axvline(T_RST, color='0.6', lw=0.8, ls=':')
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
    ax[0].axhline(0, color='0.5', lw=0.6)
    ax[5].set_yscale('log')
    ax[0].legend(fontsize=8, frameon=False, loc='upper left')
    ax[5].legend(fontsize=8, frameon=False, loc='lower left')
    fig.suptitle('B.1 bed (rp10, L4, DE off): B1V and B1T from B1N_cell at t = 41.78 (dotted line); '
                 'periapsis t_p = 135.9', fontsize=11)
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == '__main__':
    main()
