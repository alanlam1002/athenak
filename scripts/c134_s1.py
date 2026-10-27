#!/usr/bin/env python3
"""c134 pilots through S1 (R-145 item 5; PHASE1_SPECS S-C): the periapsis passage of rp18 from r0 = 30, 50, 82
and rp10 from r0 = 82, aligned on t - t_p (each pilot's geodesic periapsis time).
  1. From the checkpoint restarts (full 3D state, tde_dmde.cells; comoving restarts give the BH-frame Killing
     energy) at C0 (t_p - 50) and C1 (t_p + 50): the bound fraction (E < 0; Bernoulli E_B < 0), the dM/dE
     quantiles q10/q50/q90 and the mean of E/dE (dE = R_*/r_t^2), the mass-weighted specific L_z about the BH
     (x u_y - y u_x relative to x_BH), the self-bound remnant (remnant.bound_mass) and its K/K0 - 1.
     -> OUT/c134_s1_rst.csv, OUT/c134_s1_dmde.png (1/M dM/d(E/dE) at C1)
  2. Time series against t - t_p (c110_analysis.series): rho_max, the star-BH separation (and the geodesic),
     M_bound (CRECOV), the frame-controller mass fc_mass, the entropy tracer scal-0, the wall time per M.
     -> OUT/c134_s1_series.png
    c134_s1.py OUT LABEL=RUNDIR:TP:RP [...] [--nproc 8]
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
import c110_analysis as A   # noqa: E402
import remnant as RM        # noqa: E402
import tde_dmde as T        # noqa: E402

GAM, K0, MSTAR = 5.0 / 3.0, 1.2346626297e-2, 7.47e-5
DE = 0.677 / 16.4**2
EDGES = np.linspace(-4, 4, 81)
COL = {'P18_r30': 'C0', 'P18_r50': 'C1', 'P18_r82': 'C2', 'P10_r82': 'C3'}


def wq(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o]) / w.sum()
    return np.interp(q, c, x[o])


def rst_stats(arg):
    lab, stage, fn, tp = arg
    c = T.cells(fn, rcut=3.0)
    E, dM = c['E'], c['dM']
    lz = (c['x'] - c['xbh'][0]) * c['uy'] - (c['y'] - c['xbh'][1]) * c['ux']
    Mb, xc, vc, sel = RM.bound_mass(c)
    k = c['P'][sel] / c['rho'][sel]**GAM / K0 - 1.0
    e = E / DE
    H, _ = np.histogram(e, bins=EDGES, weights=dM)
    r = dict(label=lab, stage=stage, t=c['t'], t_minus_tp=c['t'] - tp, M_in=dM.sum() / MSTAR,
             f_bound=dM[E < 0].sum() / dM.sum(), f_bound_B=dM[c['EB'] < 0].sum() / dM.sum(),
             q10=wq(e, dM, 0.1), q50=wq(e, dM, 0.5), q90=wq(e, dM, 0.9), E_mean=np.average(e, weights=dM),
             Lz_mean=np.average(lz, weights=dM), M_rem=Mb / MSTAR,
             r_rem=float(np.linalg.norm(xc[:2] - c['xbh'][:2])),
             K_rem=np.average(k, weights=c['dM'][sel]) if sel.any() else np.nan)
    print(r, flush=True)
    return r, H / dM.sum() / np.diff(EDGES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--nproc', type=int, default=8)
    ap.add_argument('--rst-nproc', type=int, default=2)   # ~2-3 GB of cell arrays per restart
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = {}
    for spec in a.runs:
        lab, rest = spec.split('=', 1)
        d, tp, rp = rest.split(':')
        R[lab] = (d, float(tp), float(rp))

    jobs = [(lab, st, sorted(glob.glob(f'{d}/checkpoints/{st}/*.rst'))[-1], tp)
            for lab, (d, tp, rp) in R.items() for st in ('C0', 'C1')]
    with Pool(a.rst_nproc) as pool:
        res = pool.map(rst_stats, jobs)
    keys = list(res[0][0].keys())
    with open(f'{a.out}/c134_s1_rst.csv', 'w') as f:
        f.write(','.join(keys) + '\n')
        for r, _ in res:
            f.write(','.join(f'{r[k]:.6g}' if isinstance(r[k], float) else str(r[k]) for k in keys) + '\n')
    fig, ax = plt.subplots(figsize=(7.5, 5))
    xc = 0.5 * (EDGES[1:] + EDGES[:-1])
    for r, h in res:
        if r['stage'] == 'C1':
            ax.step(xc, h, where='mid', color=COL.get(r['label'], '0.5'), label=r['label'])
    ax.set_yscale('log'); ax.set_ylim(1e-4, None)
    ax.set_xlabel(r'$E/\Delta E$ ($\Delta E = R_*/r_t^2$, BH-frame Killing energy)')
    ax.set_ylabel(r'$(1/M)\,dM/d(E/\Delta E)$'); ax.legend()
    ax.set_title('c134 C1 (t_p + 50): dM/dE, r_BH ≥ 3')
    fig.tight_layout(); fig.savefig(f'{a.out}/c134_s1_dmde.png', dpi=100); plt.close(fig)

    S = {lab: A.series(lab, d, a.nproc) for lab, (d, tp, rp) in R.items()}
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    ax = axs.ravel()
    for lab, (d, tp, rp) in R.items():
        s, c = S[lab], COL.get(lab, '0.5')
        ax[0].plot(s['t_u'] - tp, s['rhomax'] / A.RHOC, c, label=lab)
        ax[1].plot(s['t_dump'] - tp, s['sep'], c)
        tg, rg = A.geodesic(d, float(np.nanmax(s['t_u'])), rp)
        ax[1].plot(tg - tp, rg, c, ls=':', lw=0.8)
        if len(s['mb']):
            ax[2].plot(s['mb'][:, 0] - tp, s['mb'][:, 1] / MSTAR, c)
        t, fc = A.col(d, 'user', 'fc_mass')
        ax[3].plot(t - tp, fc / fc[0], c)
        ax[4].plot(s['t_m'] - tp, s['scal0'] / s['scal0'][0] - 1.0, c)
        T_, W = s['T'], s['W']
        g = np.arange(T_[0], T_[-1], 5.0)
        if len(g) > 2:
            w = np.interp(g, T_, W)
            ax[5].plot(0.5 * (g[1:] + g[:-1]) - tp, np.diff(w) / 5.0, c)
    tit = [r'$\rho_{max}/\rho_c$', 'star–BH separation [M] (dotted: geodesic)', r'$M_{bound}/M_*$ (CRECOV)',
           'fc_mass / fc_mass(0)', 'scal-0 / scal-0(0) − 1', 'wall s per M (8 nodes)']
    for k in range(6):
        ax[k].set_title(tit[k]); ax[k].axvline(0, color='0.6', lw=0.8); ax[k].axvline(-50, color='0.8', lw=0.6)
        ax[k].axvline(50, color='0.8', lw=0.6)
    ax[0].set_yscale('log'); ax[1].set_yscale('log'); ax[0].legend()
    for k in (3, 4, 5):
        ax[k].set_xlabel(r'$t - t_p$ [M]')
    fig.suptitle('c134 pilots through S1 (base9 9805e76a, L4, ±128 box); grey: t_p, C0, C1')
    fig.tight_layout(); fig.savefig(f'{a.out}/c134_s1_series.png', dpi=100); plt.close(fig)


if __name__ == '__main__':
    main()
