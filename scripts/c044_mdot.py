#!/usr/bin/env python3
"""C-044: fallback rate Mdot(t) from dM/dE (frozen-in, Keplerian return), R6/R7 against
their BH-frame baselines (RQ1, Q3).

Per restart (scripts/tde_dmde.py cells: BH-frame Killing energy E = -u_t - 1, cells with
r_BH < 3 excluded; the self-bound remnant from scripts/remnant.py removed), each bound
debris element returns to periapsis after one Keplerian period,

    t_ret = t_p + 2 pi (-2E)^(-3/2),        a = 1/(-2E),

with t_p the periapsis time of the release geodesic (tde_parabolic_orbit_ic). Mdot is the
dM-weighted histogram of t_ret (log bins) divided by the bin width: the standard
Mdot = dM/dE |dE/dt|. Newtonian return (argued fine: the most bound debris has
a >~ 40 M); it ignores the remnant's pull on the debris and any hydrodynamics on the way
back. Later restarts test whether dM/dE is frozen in.

Units: G = c = M_BH = 1. The deck's rhoc = 3.24e-12 (M_sun = 1 units) -> 3.24e-4 by
lambda^2 gives M_BH = 1e4 M_sun, so 1 M = 4.925e-2 s and M* = 7.47e-5 M = 0.747 M_sun.

    c044_mdot.py [--nproc 6] [--out DIR]
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tde_dmde as D                              # noqa: E402
import remnant as RM                              # noqa: E402
import rst_tree as RT                             # noqa: E402

C = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc'
MSTAR = 7.47e-5
TSEC = 4.925e-2                                   # s per M for M_BH = 1e4 M_sun
MSUN_STAR = 0.747
DE = 0.677 / 16.4**2
RUNS = {'R6 (comoving rp10, si = 0)': ('R6_rp10_comov_L4_fix', 10.0, (171.1, 182.9, 195.2, 200.0), 'C0'),
        'RQ1 (BH frame rp10, corrected fill)': ('RQ1_bh_L4_t300_from_De', 10.0, (176.2, 195.2, 236.5, 300.0), 'C3'),
        'R7 (comoving rp18, si = 0)': ('R7_rp18_comov_L4_fix', 18.0, (253.7, 300.0), 'C2'),
        'Q3 (BH frame rp18, si = 1)': ('Q3_P2_rp18_L4', 18.0, (256.6, 300.0), 'C1')}
TB = np.logspace(2.5, 6.5, 81)                    # return-time bins [M]


def t_peri(d, rp):
    import tde_parabolic_orbit_ic as P
    from anim_global_local import par_get
    x0 = par_get(f'{C}/{d}', 'star_center_x1') - par_get(f'{C}/{d}', 'puncture_x1')
    s = P.solve_ic(x0, rp)
    o = P.integrate(x0, s['px'], s['py'], 400.0, 0.01)
    return float(o[np.argmin(np.hypot(o[:, 1], o[:, 2])), 0])


def rst_at(d, t):
    fs = sorted(glob.glob(f'{C}/{d}/rst/*.rst'))
    ts = np.array([RT.read_tree(f)['time'] for f in fs])
    i = int(np.argmin(np.abs(ts - t)))
    return fs[i] if abs(ts[i] - t) < 0.5 else None


def one(job):
    d, tt = job
    cache = f'{C}/plots/phase2_m1/c044_mdot_cache/{d}_{tt:.1f}.npz'
    if os.path.exists(cache):
        z = np.load(cache)
        return d, tt, z['t'], z['E'], z['dM'], z['rem']
    c = D.cells(rst_at(d, tt), rcut=3.0)
    _, _, _, sel = RM.bound_mass(c)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, t=c['t'], E=c['E'], dM=c['dM'], rem=sel)
    return d, tt, c['t'], c['E'], c['dM'], sel


def mdot(E, dM, tp):
    b = E < 0
    tr = tp + 2 * np.pi * (-2 * E[b])**-1.5
    h, _ = np.histogram(tr, bins=TB, weights=dM[b])
    return h / np.diff(TB) / MSTAR                  # M*/M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nproc', type=int, default=6)
    ap.add_argument('--out', default=f'{C}/plots/phase2_m1')
    a = ap.parse_args()
    jobs = [(d, t) for d, _, ts, _ in RUNS.values() for t in ts]
    with Pool(a.nproc) as p:
        res = {(d, t): r for d, t, *r in p.map(one, jobs)}
    tc = np.sqrt(TB[1:] * TB[:-1])
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    rows = [f'Mdot from dM/dE (Keplerian return from the release geodesic\'s periapsis; r_BH < 3 '
            f'and the self-bound remnant excluded). M*/M; x {MSUN_STAR / TSEC:.2f} -> M_sun/s for '
            f'M_BH = 1e4 M_sun (1 M = {TSEC} s).']
    for lab, (d, rp, ts, col) in RUNS.items():
        tp = t_peri(d, rp)
        rows.append(f'== {lab}: periapsis t_p = {tp:.1f}')
        for n, tt in enumerate(ts):
            t, E, dM, rem = res[(d, tt)]
            deb = ~rem
            md = mdot(E[deb], dM[deb], tp)
            mda = mdot(E, dM, tp)
            last = n == len(ts) - 1
            ax[0].loglog(tc, np.where(md > 0, md, np.nan), '-', color=col, lw=1.6 if last else 0.7,
                         alpha=1 if last else 0.5, label=f'{lab}, t = {t:.0f}' if last or n == 0 else None)
            if last:
                ax[0].loglog(tc, np.where(mda > 0, mda, np.nan), ':', color=col, lw=1)
                h, e = np.histogram(E / DE, bins=np.linspace(-6, 6, 121), weights=dM / MSTAR)
                ax[1].semilogy(0.5 * (e[1:] + e[:-1]), np.where(h > 0, h / np.diff(e), np.nan), '-',
                               color=col, label=f'{lab}, t = {t:.0f} (all gas)')
                h, _ = np.histogram(E[deb] / DE, bins=e, weights=dM[deb] / MSTAR)
                ax[1].semilogy(0.5 * (e[1:] + e[:-1]), np.where(h > 0, h / np.diff(e), np.nan), '--',
                               color=col, lw=0.8)
            i = int(np.argmax(md))
            Eb = E[deb & (E < 0)]
            mb = dM[deb & (E < 0)].sum() / MSTAR
            late = (tc > 3 * tc[i]) & (tc < 30 * tc[i]) & (md > 0)
            sl = np.polyfit(np.log(tc[late]), np.log(md[late]), 1)[0] if late.sum() > 3 else np.nan
            tr = np.sort(tp + 2 * np.pi * (-2 * Eb)**-1.5)          # returned mass vs time
            cm = np.cumsum(dM[deb & (E < 0)][np.argsort(tp + 2 * np.pi * (-2 * Eb)**-1.5)]) / MSTAR
            t1 = tr[np.searchsorted(cm, 1e-3)] if cm[-1] > 1e-3 else np.nan
            t10 = tc[np.argmax(md >= 0.1 * md[i])]
            rows.append(f'  t = {t:6.1f}: bound debris (no remnant) {mb:.4f} M*, remnant {dM[rem].sum() / MSTAR:.4f} M*; '
                        f'1e-3 M* returned by {t1:.0f} M, Mdot > 10% of peak from {t10:.3g} M; '
                        f'peak Mdot {md[i]:.2e} M*/M = {md[i] * MSUN_STAR / TSEC:.2e} M_sun/s at t = {tc[i]:.3g} M '
                        f'({tc[i] * TSEC / 60:.1f} min); late slope {sl:+.2f}; Mdot at 1e4/1e5 M '
                        f'{np.interp(1e4, tc, md):.2e}/{np.interp(1e5, tc, md):.2e}')
            print(rows[-1], flush=True)
    t = np.logspace(3.5, 6.5, 20)
    ax[0].loglog(t, 1e-4 * (t / 1e4)**(-5 / 3), 'k--', lw=0.8, label=r'$t^{-5/3}$')
    ax[0].set_xlabel('t [M]  (1 M = 0.049 s for M_BH = 1e4 M_sun)'); ax[0].set_ylabel(r'$\dot M$ [M*/M]')
    ax[0].set_title('fallback rate (solid: remnant excluded, thin = earlier restarts; dotted: remnant included)')
    ax[0].set_ylim(1e-9, 1e-2)
    ax[1].set_xlabel(r'$E/\Delta E$'); ax[1].set_ylabel(r'dM/d(E/ΔE) [M*]')
    ax[1].set_title('dM/dE at the last restart (dashed: remnant excluded)')
    for x in ax:
        x.grid(alpha=0.3, which='both'); x.legend(fontsize=7)
    fig.suptitle('C-044: fallback rate from dM/dE, comoving si = 0 reruns vs BH-frame baselines')
    fig.savefig(f'{a.out}/c044_mdot.png', dpi=80)
    with open(f'{a.out}/c044_mdot.txt', 'w') as f:
        f.write('\n'.join(rows) + '\n')


if __name__ == '__main__':
    main()
