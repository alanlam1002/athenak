#!/usr/bin/env python3
"""c155 (R-177 item 2 + addenda): the B1K8 / B1K10 figure.
  (a) dense <K/K0 - 1> (rho >= 0.01 rho_c) vs t: B1K8, B1K10 with B1N_cell, B1V, B1T (C-143's c154_b1_dense.csv);
  (b) the residual's d_t over the dense band: rms and max of d_t dpsi, per stage (KFSPLIT);
  (c) |s_T,res| / |s_T,bg| over the dense band, per stage;
  (d) addendum 1: the dense rates bg tau deposit, v.dS (Valencia source), v.dS (fluxes), Valencia's own work,
      against d(D eps)/dt (step differences, smoothed over 20 steps);
  (e) addendum 2: per-step Delta(D eps) against |Delta xidot|_coord.
    c155_b1k_plots.py OUT.png ANA_DIR B1K_DENSE.csv C154_DENSE.csv"""
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

COL = {'B1N_cell': '#2a78d6', 'B1V': '#eb6834', 'B1T': '#1baf7a', 'B1K8': '#6250d6', 'B1K10': '#e87ba4'}


def main():
    out, ana, kd, cd = sys.argv[1:5]
    dense = pd.concat([pd.read_csv(cd), pd.read_csv(kd)])
    fig, ax = plt.subplots(1, 5, figsize=(22, 4.6), constrained_layout=True)
    for arm in ['B1N_cell', 'B1V', 'B1T', 'B1K8', 'B1K10']:
        d = dense[dense.arm == arm].sort_values('t')
        ax[0].plot(d.t, d.d01_mean, 'o-', ms=3, lw=1.5, color=COL[arm], label=arm)
    for arm in ['B1K8', 'B1K10']:
        s = pd.read_csv(f'{ana}/{arm}_stage.csv')
        p = pd.read_csv(f'{ana}/{arm}_step.csv')
        c = COL[arm]
        ax[1].semilogy(s.tstage, s.max_dpsi.where(s.max_dpsi > 0), '.', ms=1.5, color=c, alpha=0.5)
        ax[1].semilogy(s.tstage, s.rms_dpsi.where(s.rms_dpsi > 0), '-', lw=0.6, color=c, label=f'{arm} rms (line) / max (dots)')
        r = (s.res_dense.abs()/s.bg_dense.abs()).where(s.bg_dense != 0)
        ax[2].semilogy(s.tstage, r, '.', ms=1.5, color=c, label=arm)
        if arm == 'B1K10':
            sm = lambda x: pd.Series(x).rolling(60, center=True, min_periods=10).mean()   # noqa: E731
            for k, ls, nm in (('bg_tau', '-', 'bg tau deposit'), ('vdS_src', '--', 'v.dS (source)'),
                              ('vdS_flux', ':', 'v.dS (fluxes)'), ('val_work', '-.', 'Valencia -S.grad(alpha)')):
                ax[3].plot(s.tstage, sm(s[k]), ls, lw=1.3, color='0.25', label=nm)
            ax[3].plot(p.t, pd.Series(p.dDeps/p.dt).rolling(20, center=True, min_periods=5).mean(), '-', lw=1.6,
                       color=c, label='d(D eps)/dt (B1K10)')
        ax[4].plot(p.dxidot_abs, p.dDeps, '.', ms=2, color=c, alpha=0.6,
                   label=f'{arm}: r = {np.corrcoef(p.dxidot_abs, p.dDeps)[0, 1]:.2f}')
    ttl = [r'(a) dense $\langle K/K_0-1\rangle$ ($\rho\geq0.01\rho_c$)',
           r'(b) residual $\partial_t\,\delta\psi$, dense band [1/M]',
           r'(c) $|s_{T,\rm res}|/|s_{T,\rm bg}|$, dense band',
           '(d) dense-band rates [per M], 60-stage means',
           r'(e) per step: $\Delta(D\epsilon)$ vs $|\Delta\dot\xi|_{\rm coord}$']
    for a, t in zip(ax, ttl):
        a.set_title(t, fontsize=10, loc='left')
        a.grid(alpha=0.25, lw=0.5)
        for sp in ('top', 'right'):
            a.spines[sp].set_visible(False)
        a.legend(fontsize=7.5, frameon=False)
    for a in ax[:4]:
        a.set_xlabel('t [M]')
    ax[4].set_xlabel(r'$|\Delta\dot\xi|$')
    ax[0].axhline(0, color='0.5', lw=0.6)
    ax[3].axhline(0, color='0.5', lw=0.6)
    fig.suptitle('B1K8 (mg 1e-8) and B1K10 (mg 1e-10): killing_full arm (ii) + analytic background on the B.1 bed, '
                 'from B1N_cell at t = 41.78', fontsize=11)
    fig.savefig(out, dpi=100)
    print(out)


if __name__ == '__main__':
    main()
