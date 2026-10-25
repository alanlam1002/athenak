#!/usr/bin/env python3
"""R-131 item 2 (c121): the BH frame's seeded spread sigma, and the rp10 2x2 differences judged against it.
BNs = BN (c110_2x2) restarted from its 00004 restart (t = 105.5) with <job> seed_tau_eps = 1e-12, seed_id = 1
(restart_state.cpp:87-88). sigma(t) = |BNs - BN| (one pair: the RMS of the pairwise differences, as c103's
sigma_diff), and sigma_w(t) = its maximum over a +-10 M window (a single pair's difference crosses zero).
Rows (frame-independent, c110_analysis.py's series): rho_max (user hst), the grid mass (mhd hst), M_bound (CRECOV
Mbound_all), the star-BH separation (the z = 0 density maximum minus x_BH), K/K0 at the z = 0 density maximum
(K0 = BN's t = 0 value; one z = 0 cell, so noisy). The 2x2 differences BN-CN, BD-CD, CD-CN, BD-BN and the
interaction (BD-BN)-(CD-CN) at the listed times, in units of sigma_w.
    c121_seeded.py OUTCSV CN=DIR BN=DIR CD=DIR BD=DIR BNs=DIR [--t 150,200,250,300]
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c110_analysis as A                         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--t', default='150,200,250,300')
    ap.add_argument('--nproc', type=int, default=12)
    a = ap.parse_args()
    D = dict(r.split('=', 1) for r in a.runs)
    S = {k: A.series(k, d, a.nproc) for k, d in D.items()}
    # raw K at the density maximum: series() normalises by the run's own t = 0 dump (NaN for BNs); undo it with BN's
    K0 = A.dump(sorted(glob.glob(f'{D["BN"]}/bin/{A.B}.prim_xy.*.bin'))[0])[3]
    for k, d in D.items():
        if not np.isfinite(S[k]['Kc']).any():
            P = np.array([A.dump(f) for f in sorted(glob.glob(f'{d}/bin/{A.B}.prim_xy.*.bin'))])
            P = P[np.isfinite(P[:, 0])]
            S[k]['Kc'] = P[np.argsort(P[:, 0]), 3] / K0      # same order and filter as series()' t_dump

    def q(k, name, t):
        s = S[k]
        if name == 'rhomax':
            return np.interp(t, s['t_u'], s['rhomax']) / A.RHOC
        if name == 'mass':
            return np.interp(t, s['t_m'], s['mass']) / A.MSTAR
        if name == 'Mbound':
            return np.interp(t, s['mb'][:, 0], s['mb'][:, 1]) / A.MSTAR
        if name == 'sep':
            return np.interp(t, s['t_dump'], s['sep'])
        if name == 'K_over_K0':
            return np.interp(t, s['t_dump'], s['Kc'])

    names = ['rhomax', 'mass', 'Mbound', 'sep', 'K_over_K0']
    t0 = max(S['BNs']['t_dump'][0], S['BNs']['t_u'][0]) + 2.0
    tg = np.arange(np.ceil(t0), 300.0 + 1e-9, 1.0)
    out = ['quantity,t,BN,sigma,sigma_w,rel_sigma_w,BN-CN,BD-CD,CD-CN,BD-BN,interaction,'
           'BN-CN_sig,BD-CD_sig,CD-CN_sig,BD-BN_sig,interaction_sig']
    for nm in names:
        sig = np.abs(q('BNs', nm, tg) - q('BN', nm, tg))
        sw = np.array([sig[np.abs(tg - t) <= 10].max() for t in tg])
        for t in [float(x) for x in a.t.split(',')]:
            if t < tg[0]:
                continue
            v = {k: q(k, nm, t) for k in ('CN', 'BN', 'CD', 'BD')}
            s1, s2 = np.interp(t, tg, sig), np.interp(t, tg, sw)
            dd = [v['BN'] - v['CN'], v['BD'] - v['CD'], v['CD'] - v['CN'], v['BD'] - v['BN'],
                  (v['BD'] - v['BN']) - (v['CD'] - v['CN'])]
            out.append(f'{nm},{t:g},{v["BN"]:.6g},{s1:.3g},{s2:.3g},{s2 / abs(v["BN"]):.3g},'
                       + ','.join(f'{x:.4g}' for x in dd) + ',' + ','.join(f'{x / s2:.3g}' for x in dd))
    open(a.out, 'w').write('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
