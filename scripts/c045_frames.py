#!/usr/bin/env python3
"""C-045: comoving (si = 0) vs BH-frame L4 runs, one figure.

  R6 (comoving rp10) vs Q1c (D_e + RQ1, BH frame rp10)
  R7 (comoving rp18) vs Q3 (BH frame rp18, si = 1)

Panels: code dt per cycle and cumulative node-hours (ensemble.out); the offline CFL
attribution by cell role (c045_dt.txt); mass-weighted BH-frame Killing E, L_z drift, bound
fraction and self-bound remnant (c044_series_*.txt); quantile shift of the debris+remnant
energy distribution at t = 300 and dM/dE with the comoving run shifted by the median
offset (c044_mdot cache).

    c045_frames.py [--out DIR]
"""
import argparse
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

C = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc'
P = f'{C}/plots/phase2_m1'
DE = 0.677 / 16.4**2
RUNS = {'R6': (('R6_rp10_comov_L4_fix', 'R6x_rp10_comov_L4_t300'), 7, 'C0', '-'),
        'Q1c': (('D_e_Q1deck_kfix_t80', 'RQ1_bh_L4_t300_from_De'), 16, 'C3', '-'),
        'R7': (('R7_rp18_comov_L4_fix',), 9, 'C2', '--'),
        'Q3': (('Q3_P2_rp18_L4',), 16, 'C1', '--')}
LAB = {'R6': 'R6 comoving rp10', 'Q1c': 'Q1c BH frame rp10', 'R7': 'R7 comoving rp18', 'Q3': 'Q3 BH frame rp18 (si=1)'}
CACHE = {'R6': 'R6x_rp10_comov_L4_t300_300.0', 'Q1c': 'RQ1_bh_L4_t300_from_De_300.0',
         'R7': 'R7_rp18_comov_L4_fix_300.0', 'Q3': 'Q3_P2_rp18_L4_300.0'}


def log(segs, nodes):
    """t, dt, cumulative node-hours from the per-cycle log lines"""
    T, DT, NH, off = [], [], [], 0.0
    for s in segs:
        a = np.array([[float(x) for x in m.groups()] for ln in open(f'{C}/{s}/ensemble.out')
                      for m in [re.match(r'elapsed=(\S+) cycle=\d+ time=(\S+) dt=(\S+)', ln)] if m])
        a = a[1:]                                     # cycle 0's elapsed includes start-up
        if T:
            a = a[a[:, 1] > T[-1][-1]]
        T.append(a[:, 1]); DT.append(a[:, 2]); NH.append(off + (a[:, 0] - a[0, 0]) * nodes / 3600)
        off = NH[-1][-1]
    return np.concatenate(T), np.concatenate(DT), np.concatenate(NH)


def series(lab):
    return np.loadtxt(f'{P}/c044_series_{lab}.txt')   # t, warm5, warm10, remnant, bound, <E>, <L>


def dtroles():
    out = {}
    cur = None
    for ln in open(f'{P}/c045_dt.txt'):
        m = re.match(r'(\S+) t = \s*(\S+): offline min dt (\S+)', ln)
        if m:
            cur = (m.group(1), float(m.group(2))); out[cur] = {'all': float(m.group(3))}
            continue
        m = re.match(r'\s+(\S+)\s+min dt (\S+)', ln)
        if m and cur:
            out[cur][m.group(1)] = float(m.group(2))
    return out


def hill(lab):
    """(t, mass self-bound within the Hill radius [M*]) from c045_r049.txt"""
    rows, on = [], False
    for ln in open(f'{P}/c045_r049.txt'):
        if ln.startswith('=='):
            on = ln.startswith(f'== {lab}:'); continue
        m = re.search(r't=\s*(\S+) .*M\(<r_H, bound\) (\S+)', ln)
        if on and m:
            rows.append((float(m.group(1)), float(m.group(2))))
    return np.array(rows)


def quant(E, m, q):
    o = np.argsort(E); c = np.cumsum(m[o]) / m.sum()
    return np.interp(q, c, E[o])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=P)
    a = ap.parse_args()
    fig, ax = plt.subplots(3, 3, figsize=(17, 13), constrained_layout=True)
    ax = ax.ravel()
    for lab, (segs, nodes, col, ls) in RUNS.items():
        t, dt, nh = log(segs, nodes)
        k = 25
        sm = np.convolve(dt, np.ones(k) / k, mode='same')
        ax[0].plot(t[k:-k], sm[k:-k], ls, color=col, label=f'{LAB[lab]} ({len(t)} cycles)')
        ax[2].plot(t, nh, ls, color=col, label=f'{LAB[lab]} ({nodes} nodes): {nh[-1]:.0f} node-h')
        s = series(lab)
        ax[3].plot(s[:, 0], s[:, 5] / DE, 'o' + ls, color=col, ms=3, label=LAB[lab])
        ax[4].plot(s[:, 0], (s[:, 6] - s[0, 6]) / s[0, 6], 'o' + ls, color=col, ms=3, label=LAB[lab])
        ax[5].plot(s[:, 0], s[:, 4], 'o' + ls, color=col, ms=3, label=LAB[lab])
        ax[6].plot(s[:, 0], s[:, 3], 'o' + ls, color=col, ms=3, label=LAB[lab])
        if lab in ('R7', 'Q3'):
            h = hill(lab)
            ax[6].plot(h[:, 0], h[:, 1], ':', color=col, lw=1.5, label=f'{lab}: self-bound within r_H')
    roles = dtroles()
    for lab, (_, _, col, ls) in RUNS.items():
        p = np.array(sorted((t, v['nearBH'], min(v.get(r, np.inf) for r in ('remnant', 'debris', 'atmos')))
                            for (l, t), v in roles.items() if l == lab))
        ax[1].semilogy(p[:, 0], p[:, 1], 'o' + ls, color=col, ms=5, label=f'{lab}: near BH (r_BH < 6)')
        ax[1].semilogy(p[:, 0], p[:, 2], 'x:', color=col, ms=6, lw=0.8, label=f'{lab}: elsewhere (star, debris, atmosphere)')
    q = np.linspace(0.01, 0.99, 99)
    e = np.linspace(-4, 4, 81)
    for (ca, cb), col in ((('R6', 'Q1c'), 'C0'), (('R7', 'Q3'), 'C2')):
        za, zb = np.load(f'{P}/c044_mdot_cache/{CACHE[ca]}.npz'), np.load(f'{P}/c044_mdot_cache/{CACHE[cb]}.npz')
        qa, qb = quant(za['E'], za['dM'], q), quant(zb['E'], zb['dM'], q)
        sh = np.median(qb - qa)
        fa = za['dM'][za['E'] < 0].sum() / za['dM'].sum(); fb = zb['dM'][zb['E'] < 0].sum() / zb['dM'].sum()
        fs = za['dM'][za['E'] + sh < 0].sum() / za['dM'].sum()
        ax[7].plot(q, (qb - qa) / DE, '-', color=col,
                   label=f'{cb} - {ca}: median {sh / DE:+.3f} dE; bound {fa:.3f} -> {fs:.3f} shifted (vs {fb:.3f})')
        for z, c2, l2, d in ((za, RUNS[ca][2], ca, 0.0), (zb, RUNS[cb][2], cb, 0.0), (za, RUNS[ca][2], f'{ca} shifted', sh)):
            h, _ = np.histogram((z['E'] + d) / DE, bins=e, weights=z['dM'] / z['dM'].sum())
            ax[8].plot(0.5 * (e[1:] + e[:-1]), h / np.diff(e), ':' if d else ('-' if ca == 'R6' else '--'),
                       color=c2, lw=1.0 if d else 1.5, label=l2)
    ax[0].set(xlabel='t [M]', ylabel='code dt (25-cycle mean)', title='timestep (CFL 0.4, L4 dx = 1/16)')
    ax[1].set(xlabel='t [M]', ylabel='offline CFL dt of the limiting cell', ylim=(0.02, 0.5),
              title='offline CFL dt: near-BH cells vs the rest (c045_dt.py)')
    ax[2].set(xlabel='t [M]', ylabel='cumulative node-hours', title='cost')
    ax[3].set(xlabel='t [M]', ylabel=r'$\langle E\rangle/\Delta E$', title='mass-weighted BH-frame Killing E (r_BH > 3)')
    ax[4].set(xlabel='t [M]', ylabel=r'$\Delta L_z / L_z$', title='angular momentum drift')
    ax[5].set(xlabel='t [M]', ylabel='bound fraction (E < 0)', title='bound fraction')
    ax[6].set(xlabel='t [M]', ylabel='self-bound remnant [M*]', title='remnant (remnant.py); dotted: Hill-sphere metric', yscale='log', ylim=(0.1, 1.1))
    ax[7].set(xlabel='mass quantile', ylabel=r'$(E_{BH} - E_{comov})/\Delta E$',
              title='energy-distribution offset at t = 300 (all gas, r_BH > 3)')
    ax[7].axhline(0, color='k', lw=0.5)
    ax[8].set(xlabel=r'$E/\Delta E$', ylabel='dM/d(E/ΔE) [fraction]', yscale='log', ylim=(1e-3, 2),
              title='dM/dE at t = 300 (dotted: comoving shifted)')
    for x in ax:
        x.grid(alpha=0.3); x.legend(fontsize=7)
    fig.suptitle('C-045: comoving (si = 0) vs BH-frame L4 runs')
    fig.savefig(f'{a.out}/c045_frames.png', dpi=70)


if __name__ == '__main__':
    main()
