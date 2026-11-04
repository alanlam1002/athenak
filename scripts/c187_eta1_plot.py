#!/usr/bin/env python3
"""c187 (R-223 item 1(a)) figure from c187_eta1_calib.py's CSVs (energy-only B.1: Bb10 to t = 300, Bbr to 150).
  (a) q = e_int/tau of the stored (energy) state, mass median and 1-99% per class vs t (Bb10): what a switch on the energy
      solution would read                                   (b) the same on the initial adiabat, q_ad (the flow's conditioning)
  (c) energy-only K/K0 - 1 per q_ad bin, before t_p (median, 5-95%; Bb10)   (d) the same after t_p
  dotted: candidate eta1 = 1.5e-4, eta2 = 1e-3 (C-187)
    c187_eta1_plot.py OUT.png PREFIX"""
import csv, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
TP, E1, E2 = 135.9, 1.5e-4, 1.0e-3
CLS = [('core', 'core >= 0.1 rho_c', '#2a78d6'), ('outer', 'outer 0.01-0.1 rho_c', '#eb6834'),
       ('rem', 'Hill remnant', '#1baf7a'), ('stream', 'stream', '#8a8f98')]


def main():
    out, pre = sys.argv[1:3]
    D = [r for r in csv.DictReader(open(pre + '_dumps.csv')) if r['arm'] == 'Bb10' and r.get('q50')]
    B = [r for r in csv.DictReader(open(pre + '_bins.csv')) if r['arm'] == 'Bb10']
    fig, ax = plt.subplots(2, 2, figsize=(14, 10)); ax = ax.ravel()
    for c, lab, col in CLS:
        R = sorted([r for r in D if r['cls'] == c], key=lambda r: float(r['t']))
        t = np.array([float(r['t']) for r in R])
        for k, (p, a) in enumerate((('q', ax[0]), ('qa', ax[1]))):
            m = np.array([float(r[p + '50']) for r in R]); lo = np.array([float(r[p + '01']) for r in R])
            hi = np.array([float(r[p + '99']) for r in R])
            a.plot(t, m, color=col, lw=2, label=f'{lab}: median')
            a.fill_between(t, np.maximum(lo, 1e-9), hi, color=col, alpha=0.15)
        for ph, a in (('pre', ax[2]), ('post', ax[3])):
            S = [r for r in B if r['cls'] == c and r['phase'] == ph and float(r['M']) > 1e-7]
            if not S:
                continue
            x = np.array([10**(0.5*(float(r['lo']) + float(r['hi']))) for r in S])
            a.plot(x, [float(r['K50']) for r in S], 'o-', color=col, lw=2, ms=5, label=f'{lab}: median')
            a.fill_between(x, [float(r['K05']) for r in S], [float(r['K95']) for r in S], color=col, alpha=0.15)
    for a in ax[:2]:
        a.set_yscale('log'); a.set_ylim(1e-7, 1e-2); a.axvline(TP, color='0.5', ls='--', lw=1); a.set_xlabel('t [M]')
    for a in ax[2:]:
        a.set_xscale('log'); a.set_yscale('symlog', linthresh=0.01); a.axhline(0, color='k', lw=0.8)
        a.set_xlabel('q_ad (bin centre)')
    for a in ax:
        for e, nm in ((E1, 'eta1'), (E2, 'eta2')):
            (a.axhline if a in ax[:2] else a.axvline)(e, color='k', ls=':', lw=1)
        a.grid(alpha=0.25); a.legend(fontsize=7, loc='best')
    ax[0].set_title('(a) q = e_int/tau of the energy state, median and 1-99% (Bb10)', fontsize=10)
    ax[1].set_title('(b) q_ad on the initial adiabat, median and 1-99% (Bb10)', fontsize=10)
    ax[2].set_title('(c) energy-only K/K0 - 1 vs q_ad, t < t_p (median, 5-95%)', fontsize=10)
    ax[3].set_title('(d) energy-only K/K0 - 1 vs q_ad, t > t_p (median, 5-95%)', fontsize=10)
    fig.suptitle('c187 eta1 calibration (energy-only B.1, Bb10): dotted eta1 = 1.5e-4, eta2 = 1e-3; dashed = t_p', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out, dpi=95); print('wrote', out)


if __name__ == '__main__':
    main()
