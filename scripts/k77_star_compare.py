#!/usr/bin/env python3
"""DEVELOPMENT item 77 star test: Q1 restarted at t = 121.5 with the corrected trumpet
Ahat0 (psi^6/psi^12), against Q1 (old fill) and the N5 run-to-run floor (N5 vs Q1).

Slices (every 1 M, tde_t11_surface_diag.measure): rho_max, K at rho_max, dense-gas mean K,
puffed mass, matched by dump index. Restarts: mass-weighted Killing E and L_z of all gas
outside r_BH < 3, M3 warm > 5%, self-bound mass and bound fraction (tde_ovn_plots.rst_series
and tde_dmde.cells); Q1 and N5 interpolated in time to the test run's restart times.

    k77_star_compare.py [--out DIR]
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tde_t11_surface_diag as T                  # noqa: E402
import tde_ovn_plots as O                         # noqa: E402
import tde_dmde as D                              # noqa: E402

RUN = T.RUN
TDE = 'cfc_tde_wd_imbh_parabolic_tracker'
DIRS = {'K77': f'{RUN}/K77_q1_rst5', 'Q1': f'{RUN}/Q1_V5L4_fix_t300', 'N5': f'{RUN}/N5_Q1_repeat_t150'}
DE = 0.677 / 16.4**2


def orbit(d, tmin, tmax):
    rows = []
    for f in sorted(glob.glob(f'{d}/rst/*.rst')):
        c = D.cells(f)
        if not (tmin - 25 <= c['t'] <= tmax + 25):
            continue
        m, xb = c['dM'], c['xbh']
        L = (c['x'] - xb[0]) * c['uy'] - (c['y'] - xb[1]) * c['ux']
        rows.append((c['t'], (c['E'] * m).sum() / m.sum(), (L * m).sum() / m.sum()))
    return np.array(sorted(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{RUN}/plots/overnight')
    a = ap.parse_args()
    for lab, d in DIRS.items():
        T.RUNS[lab] = (d, TDE, 10, 'k', '-')
        O.RUNS[lab] = (d, 'k', '-', lab)
    S = {lab: T.series(lab, f'{RUN}/plots/dual_energy') for lab in DIRS}
    k = S['K77']
    t0, t1 = k['t'][0], k['t'][-1]
    rows = []
    fig, ax = plt.subplots(2, 4, figsize=(20, 8.5), constrained_layout=True)
    for j, key in enumerate(('rhomax', 'Kmax', 'Kw01', 'puff')):
        for lab in DIRS:
            s = S[lab]; sel = (s['t'] >= t0 - 5) & (s['t'] <= t1 + 1)
            ax[0, j].plot(s['t'][sel], s[key][sel], label=lab)
        q = np.interp(k['t'], S['Q1']['t'], S['Q1'][key]); n = np.interp(k['t'], S['N5']['t'], S['N5'][key])
        dk, dn = np.abs(k[key] - q) / np.abs(q), np.abs(n - q) / np.abs(q)
        rows.append(f'{key:6s}: |K77 - Q1|/Q1 max {dk.max():.4f} mean {dk.mean():.4f};  '
                    f'N5 floor max {dn.max():.4f} mean {dn.mean():.4f}  (t = {t0:.0f}-{t1:.0f})')
        ax[0, j].set_title(key)
    R = {lab: O.rst_series(lab) for lab in ('K77',)}
    oc = np.load(f'{RUN}/plots/overnight/ovn_cache.npz', allow_pickle=True)
    R['Q1'] = oc['Q1_rst']; R['N5'] = np.load(f'{RUN}/plots/overnight/r040_5_n5_rst.npy')
    Ob = {lab: orbit(d, t0, t1) for lab, d in DIRS.items()}
    for j, (src, col, name) in enumerate(((R, 1, 'M3 warm>5%'), (R, 4, 'E<0 fraction'),
                                          (Ob, 1, 'mean E / dE'), (Ob, 2, 'mean L_z'))):
        tk = src['K77'][:, 0]
        for lab in DIRS:
            r = src[lab]; sel = (r[:, 0] >= t0 - 25) & (r[:, 0] <= t1 + 25)
            y = r[sel, col] / (DE if name.startswith('mean E') else 1.0)
            ax[1, j].plot(r[sel, 0], y, 'o-', label=lab)
        q = np.interp(tk, src['Q1'][:, 0], src['Q1'][:, col])
        n = np.interp(tk, src['N5'][:, 0], src['N5'][:, col])
        sc = DE if name.startswith('mean E') else 1.0
        rows.append(f'{name:12s}: K77 - Q1 at t = ' + ', '.join(f'{t:.0f}: {(v - w) / sc:+.4f}' for t, v, w in
                    zip(tk, src['K77'][:, col], q)) + '   (N5 - Q1: ' +
                    ', '.join(f'{(w2 - w) / sc:+.4f}' for w2, w in zip(n, q)) + ')')
        ax[1, j].set_title(name)
    for x in ax.flat:
        x.set_xlabel('t [M]'); x.grid(alpha=0.3); x.legend(fontsize=7)
    fig.suptitle('Item 77: corrected trumpet Ahat0 (K77) vs old fill (Q1), restart at t = 121.5; N5 = run-to-run floor')
    fig.savefig(f'{a.out}/k77_star_test.png', dpi=90)
    with open(f'{a.out}/k77_star_test.txt', 'w') as f:
        f.write('\n'.join(rows) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
