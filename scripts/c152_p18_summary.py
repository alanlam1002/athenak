#!/usr/bin/env python3
"""c152: P18_r82 summary figure, t = 0 -> t_p + 1000 (whole run).
Top row, from the histories (every cycle): rho_max/rho_c (user.hst rho-max; rho_c = 1e10 x the deck's
atmosphere_zone_rho, which is 1e-10 rho_c), the grid rest mass change M_grid/M_grid(0) - 1 (mhd.hst mass) and dt.
Rows with a non-finite mass or rho_max < 1e-20 rho_c are dropped: P18_r82 has one, the first row written after
the t_p + 800 embed restart (t = 1291.22: mass NaN, rho_max = 1e-24); the next row is normal.
Bottom row, from the remnant series (c142_remnant.py on the restarts, results/c149/P18_B_to_tp1000.csv):
the bound remnant masses M_Hill, M_Hsc, M_rem in M_*, their mean K = P/rho^Gamma in K0, and the remnant-BH
distance r_BH. t_p = 491.1 (C-138).
    c152_p18_summary.py RUNDIR B.csv OUT.png"""
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt               # noqa: E402
import numpy as np                            # noqa: E402

TP, MSTAR = 491.1, 7.47e-5
C = ['#2a78d6', '#eb6834', '#1baf7a']          # dataviz categorical slots 1-3, fixed order
INK, MUTED = '#2b2b2a', '#8a8984'


def hst(f):
    h = open(f).readlines()[1]
    cols = {m.group(2): int(m.group(1)) - 1 for m in re.finditer(r'\[\s*(\d+)\]=(\S+)', h)}
    return np.loadtxt(f), cols


def style(a, xl, yl):
    a.set_xlabel(xl, color=INK); a.set_ylabel(yl, color=INK)
    a.grid(True, color='#e4e3df', lw=0.6); a.set_axisbelow(True)
    for s in ('top', 'right'):
        a.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        a.spines[s].set_color(MUTED)
    a.tick_params(colors=MUTED, labelcolor=INK)


def main():
    d, bcsv, out = sys.argv[1:4]
    par = open(f'{d}/parfile.par').read()
    rhoc = 1e10*float(re.search(r'(?m)^atmosphere_zone_rho\s*=\s*([-+.\deE]+)', par).group(1))
    u, cu = hst(f'{d}/cfc_tde_wd_imbh_parabolic_tracker.user.hst')
    m, cm = hst(f'{d}/cfc_tde_wd_imbh_parabolic_tracker.mhd.hst')
    n = min(len(u), len(m))
    ok = np.isfinite(m[:n, cm['mass']]) & (u[:n, cu['rho-max']] > 1e-20*rhoc)
    print('dropped rows at t =', u[:n][~ok, 0])
    u, m = u[:n][ok], m[:n][ok]
    B = np.genfromtxt(bcsv, delimiter=',', names=True, dtype=None, encoding=None)
    x = B['t_minus_tp']

    fig, ax = plt.subplots(2, 3, figsize=(15, 8.2), constrained_layout=True)
    a = ax[0, 0]
    a.semilogy(u[:, 0], u[:, cu['rho-max']]/rhoc, color=C[0], lw=1.2)
    style(a, 't [M]', r'$\rho_{max}/\rho_c$'); a.set_title('peak density', color=INK, loc='left')
    a = ax[0, 1]
    a.plot(m[:, 0], m[:, cm['mass']]/m[0, cm['mass']] - 1.0, color=C[0], lw=1.2)
    style(a, 't [M]', r'$M_{grid}/M_{grid}(0) - 1$'); a.set_title('rest mass on the grid', color=INK, loc='left')
    a.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    a = ax[0, 2]
    a.plot(u[:, 0], u[:, cu['dt']] if 'dt' in cu else m[:, 1], color=C[0], lw=1.0)
    style(a, 't [M]', r'$\Delta t$ [M]'); a.set_title('time step', color=INK, loc='left')
    for a in ax[0]:
        a.axvline(TP, color=MUTED, lw=0.8, ls='--')
        a.annotate(r'$t_p$', (TP, 1), xycoords=('data', 'axes fraction'), xytext=(3, -12),
                   textcoords='offset points', color=MUTED, fontsize=9)

    a = ax[1, 0]
    for k, (col, lab) in enumerate((('M_Hill', 'Hill sphere'), ('M_Hsc', 'scaled Hill'), ('M_rem', 'bound remnant'))):
        a.plot(x, B[col], '-o', color=C[k], lw=2, ms=4, label=lab)
        a.annotate(lab, (x[-1], B[col][-1]), xytext=(4, 0), textcoords='offset points', va='center',
                   color=INK, fontsize=8)
    style(a, r'$t - t_p$ [M]', r'$M/M_*$'); a.set_title('remnant mass', color=INK, loc='left')
    a.legend(frameon=False, fontsize=8, labelcolor=INK); a.set_xlim(right=x[-1]*1.22)
    a = ax[1, 1]
    for k, (col, lab) in enumerate((('K_Hill', 'Hill sphere'), ('K_rem', 'bound remnant'))):
        a.plot(x, B[col], '-o', color=C[[0, 2][k]], lw=2, ms=4, label=lab)
    style(a, r'$t - t_p$ [M]', r'mass-weighted $\langle K/K_0 - 1\rangle$ (c142)')
    a.set_title('remnant entropy', color=INK, loc='left'); a.legend(frameon=False, fontsize=8, labelcolor=INK)
    a = ax[1, 2]
    a.plot(x, B['r_bh'], '-o', color=C[0], lw=2, ms=4)
    style(a, r'$t - t_p$ [M]', r'$r_{BH}$ [M]'); a.set_title('remnant–BH distance', color=INK, loc='left')
    fig.suptitle(f'P18_r82 (r_p = 18, L4, comoving), t = 0 → {u[-1, 0]:.0f} M', color=INK)
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == '__main__':
    main()
