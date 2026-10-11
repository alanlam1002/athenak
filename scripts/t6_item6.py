#!/usr/bin/env python3
"""R-040 item 6: the comoving rp10 run (BH box L4 or L3, corrected trumpet fill) against the
BH-frame Q1 (old fill), the N5 run-to-run floor (N5 vs Q1), and K77 (Q1 restarted at t = 121.5
with the corrected fill, to t = 160).

Slices (every 1 M, tde_t11_surface_diag.measure): rho_max, K at rho_max, dense-gas mean K
(rho > 0.1 rho0), puffed mass. Restarts (tde_ovn_plots.rst_series, frame-aware through
tde_dmde.cells): M3 warm > 5%, self-bound remnant, bound fraction; orbit: mass-weighted
BH-frame Killing E and L_z about x_BH; dM/dE at the last common restart. dt per cycle from
ensemble.out.

    t6_item6.py LABEL=DIR [--out DIR]
"""
import argparse
import glob
import os
import re
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
REF = {'Q1': f'{RUN}/Q1_V5L4_fix_t300', 'N5': f'{RUN}/N5_Q1_repeat_t150', 'K77': f'{RUN}/K77_q1_rst5'}
DE = 0.677 / 16.4**2


def orbit(d):
    rows = []
    for f in sorted(glob.glob(f'{d}/rst/*.rst')):
        c = D.cells(f)
        m, xb = c['dM'], c['xbh']
        L = (c['x'] - xb[0]) * c['uy'] - (c['y'] - xb[1]) * c['ux']
        rows.append((c['t'], (c['E'] * m).sum() / m.sum(), (L * m).sum() / m.sum()))
    return np.array(sorted(rows))


def dts(d):
    r = [(float(m.group(1)), float(m.group(2))) for m in
         (re.search(r'time=([-+.\deE]+) dt=([-+.\deE]+)', l) for l in open(f'{d}/ensemble.out')) if m]
    return np.array(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--out', default=f'{RUN}/plots/phase2_m1')
    a = ap.parse_args()
    lab, d = a.run.split('=')
    dirs = dict(REF, **{lab: d})
    for k, v in dirs.items():
        T.RUNS[k] = (v, TDE, 10, 'k', '-')
        O.RUNS[k] = (v, 'k', '-', k)
    S = {k: T.series(k, f'{RUN}/plots/dual_energy') for k in dirs}
    oc = np.load(f'{RUN}/plots/overnight/ovn_cache.npz', allow_pickle=True)
    R = {'Q1': oc['Q1_rst'], 'N5': np.load(f'{RUN}/plots/overnight/r040_5_n5_rst.npy')}
    for k in (lab, 'K77'):
        fn = f'{a.out}/{k}_rst.npy'
        R[k] = np.load(fn) if os.path.exists(fn) and k == 'K77' else O.rst_series(k)
        np.save(fn, R[k])
    Ob = {k: orbit(dirs[k]) for k in (lab, 'Q1', 'N5', 'K77')}
    rows = []
    fig, ax = plt.subplots(3, 4, figsize=(21, 13), constrained_layout=True)
    s0 = S[lab]
    for j, key in enumerate(('rhomax', 'Kmax', 'Kw01', 'puff')):
        for k in dirs:
            ax[0, j].plot(S[k]['t'], S[k][key], label=k)
        q = np.interp(s0['t'], S['Q1']['t'], S['Q1'][key])
        tn = S['N5']['t']; qn = np.interp(tn, S['Q1']['t'], S['Q1'][key])
        for t0, t1 in ((0, 100), (100, 150), (150, 200)):
            sel = (s0['t'] >= t0) & (s0['t'] <= t1); seln = (tn >= t0) & (tn <= t1)
            dk = np.abs(s0[key][sel] - q[sel]) / np.abs(q[sel])
            fl = (np.abs(S['N5'][key][seln] - qn[seln]) / np.abs(qn[seln])).max() if seln.any() else np.nan
            rows.append(f'{key:6s} t={t0:3d}-{t1:3d}: |{lab}-Q1|/Q1 max {dk.max():.4f} mean {dk.mean():.4f}; '
                        f'N5 floor max {fl:.4f}; {lab} at {s0["t"][sel][-1]:.0f} = {s0[key][sel][-1]:.4g} '
                        f'(Q1 {q[sel][-1]:.4g})')
        ax[0, j].set_title(key)
    for j, (col, name) in enumerate(((1, 'M3 warm>5%'), (3, 'bound remnant/M*'), (4, 'E<0 fraction'))):
        for k in R:
            ax[1, j].plot(R[k][:, 0], R[k][:, col], 'o-', ms=3, label=k)
        r = R[lab]
        q = np.interp(r[:, 0], R['Q1'][:, 0], R['Q1'][:, col])
        rows.append(f'{name:16s}: ' + ', '.join(f't={t:.0f} {v:.3f} (Q1 {w:.3f})' for t, v, w in zip(r[:, 0], r[:, col], q)))
        ax[1, j].set_title(name)
    for k, o in Ob.items():
        ax[1, 3].plot(o[:, 0], (o[:, 1] - Ob['Q1'][0, 1]) / DE, 'o-', ms=3, label=f'{k} E')
        ax[2, 0].plot(o[:, 0], o[:, 2] - Ob['Q1'][0, 2], 'o-', ms=3, label=f'{k} L_z')
        rows.append(f'orbit {k:5s}: ' + ', '.join(f't={t:.0f} dE/ΔE {(e - Ob["Q1"][0, 1]) / DE:+.4f} dL {l - Ob["Q1"][0, 2]:+.4f}'
                                                 for t, e, l in o[::2]))
    ax[1, 3].set_title('⟨E⟩ − Q1(t0), units of ΔE'); ax[2, 0].set_title('⟨L_z⟩ − Q1(t0)')
    for k in (lab, 'Q1'):
        t = dts(dirs[k]); ax[2, 1].plot(t[:, 0], t[:, 1], lw=0.7, label=k)
        for t0, t1 in ((0, 100), (100, 150), (150, 200)):
            sel = (t[:, 0] >= t0) & (t[:, 0] <= t1)
            rows.append(f'dt {k:5s} t={t0}-{t1}: mean {t[sel, 1].mean():.4f} min {t[sel, 1].min():.4f}, cycles {sel.sum()}')
    ax[2, 1].set_title('dt'); ax[2, 1].set_ylim(0, None)
    f1 = sorted(glob.glob(f'{d}/rst/*.rst'))[-1]
    t1, h1 = O.dmde(lab, f1)
    fq = min(glob.glob(f'{REF["Q1"]}/rst/*.rst'), key=lambda f: abs(D.cells(f)['t'] - t1))
    tq, hq = O.dmde('Q1', fq)
    xc = 0.5 * (O.EDGES[1:] + O.EDGES[:-1])
    ax[2, 2].semilogy(xc, h1 + 1e-12, label=f'{lab} t={t1:.0f}'); ax[2, 2].semilogy(xc, hq + 1e-12, label=f'Q1 t={tq:.0f}')
    ax[2, 2].set_title('dM/dE (E/ΔE)')
    rows.append(f'dM/dE at t={t1:.0f} vs Q1 t={tq:.0f}: sum|diff|/sum = {np.abs(h1 - hq).sum() / hq.sum():.3f}')
    ax[2, 3].axis('off')
    for x in ax.flat:
        x.set_xlabel('t [M]'); x.grid(alpha=0.3)
        if x.has_data():
            x.legend(fontsize=7)
    fig.suptitle(f'R-040 item 6: {lab} (comoving, corrected fill) vs Q1 (BH frame, old fill), N5 (floor), K77 (BH frame, corrected fill)')
    fig.savefig(f'{a.out}/item6_{lab}.png', dpi=80)
    with open(f'{a.out}/item6_{lab}.txt', 'w') as f:
        f.write('\n'.join(rows) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
