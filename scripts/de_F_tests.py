#!/usr/bin/env python3
"""T-13 F: dual-energy entropy variable D*K vs D*K^(1/Gamma) (research R-029 §3, R-030 F;
src/cfc/DEVELOPMENT.md item 71). CPU 1D tests, ppmx + HLLE + FOFC, 256 cells:

  c10sup   pressure-equilibrium contact, rho 1 | 0.1, P = 1e-4, v = 0.5 (grid Mach 12-40,
           HLLE fan closed: pure upwind), t = 0.4
  c10sub   same contact at v = 0.005 (subsonic: HLL fan open), t = 8
  c1e4     a "stellar surface": rho 1 | 1e-4, P = 1e-6, v = 0.5; the light side is below
           rho_switch (energy branch), so its K = 4.6 (1e6 x the dense side) is what can
           leak into the dense side through the tracer
  sod      Sod (V1), t = 0.4
  wave     grid-Mach-200 entropy wave (V1), periodic, t = 8
  c1e4d2/3 c1e4 along x2 (2D, 4x256) and x3 (3D, 4x4x256): the x2/x3 flux paths and the
           3D scalar positivity limiter (dyn_grmhd_fofc.cpp)

Variants (directory suffix): off (energy only), k (D*K, the V1-V4 build), k1gY
(D*K^(1/G), Y reconstructed), k1g (D*K^(1/G), rho*Y reconstructed),
k1ghll (k1g + HLLE flux of the tracer itself; the default).

    de_F_tests.py [--dir DIR] [--out DIR]
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

GAM = 5.0 / 3.0
VARS = ['off', 'k', 'k1gY', 'k1g', 'k1ghll']
LAB = {'off': 'energy only', 'k': 'D·K (V1–V4)', 'k1gY': 'D·K^(1/Γ), Y recon',
       'k1g': 'D·K^(1/Γ), ρY recon, mass-flux tracer', 'k1ghll': 'D·K^(1/Γ), ρY + HLL tracer flux (default)'}
COL = {'off': 'k', 'k': '#d95f02', 'k1gY': '#7570b3', 'k1g': '#1b9e77', 'k1ghll': '#e7298a'}


def last(d):
    fn = sorted(glob.glob(f'{d}/tab/*.tab'))[-1]
    a = np.loadtxt(fn, ndmin=2)
    return dict(x=a[:, 2], rho=a[:, 3], v=a[:, 4], P=a[:, 7], Y=a[:, 8])


def contact_metrics(s, P0, rhol):
    dP = np.abs(s['P'] / P0 - 1)
    dense = s['rho'] > 0.95 * rhol      # unmixed dense side (mixed cells excluded)
    K = s['P'] / s['rho']**GAM
    Kl = P0 / rhol**GAM
    return dP.max(), np.abs(K[dense] / Kl - 1).max(), int((dP > 1e-6).sum())


def sod_metrics(s, ref):
    """plateau P error against the energy-only run (same scheme): the span where the
    energy-only P is within 1% of its plateau value, shrunk by 3 cells per side."""
    x = s['x']
    Pst = np.median(ref['P'][(x > 0.6) & (x < 0.8)])
    ok = np.where((np.abs(ref['P'] / Pst - 1) < 0.01) & (x > 0.45))[0]
    i0, i1 = ok.min() + 3, ok.max() - 3
    sl = slice(i0, i1 + 1)
    e = s['P'][sl] / Pst - 1
    xc = 0.5 * (x[i0] + x[i1])
    # shock position: rho crosses the midpoint of post-shock and right states
    rr, xr = s['rho'][i1:], x[i1:]
    mid = 0.5 * (np.median(ref['rho'][i1 - 10:i1]) + 0.125)
    ksh = np.where(rr < mid)[0]
    xsh = xr[ksh[0]] if len(ksh) else np.nan
    ps = slice(i1 - 10, i1)
    Kps = np.median(s['P'][ps] / s['rho'][ps]**GAM)
    return e.min(), np.abs(e).max(), np.percentile(np.abs(e), 90), xsh, Kps, (x[i0], x[i1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='/lus/flare/projects/CompactBinaryMerger/tlam/'
                    'athenak_run/cfc/dual_energy_tests_F')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out = a.out or f'{a.dir}/plots'
    os.makedirs(out, exist_ok=True)
    D = a.dir
    rows = []
    fig, ax = plt.subplots(2, 3, figsize=(18, 9.5), constrained_layout=True)
    for test, P0, rhol, axx in (('c10sup', 1e-4, 1.0, ax[0, 0]), ('c10sub', 1e-4, 1.0, ax[0, 1]),
                                ('c1e4', 1e-6, 1.0, ax[0, 2])):
        for v in VARS:
            s = last(f'{D}/{test}_{v}')
            dP, dK, n = contact_metrics(s, P0, rhol)
            rows.append(f'{test:7s} {v:7s} max|dP/P| {dP:9.2e}  dense-side max|dK/K| {dK:9.2e}'
                        f'  cells with |dP/P|>1e-6: {n}')
            axx.semilogy(s['x'], np.maximum(np.abs(s['P'] / P0 - 1), 1e-17), color=COL[v],
                         lw=1.2, label=LAB[v])
        axx.set_title({'c10sup': 'contact ρ 1|0.1, v = 0.5 (supersonic): |P/P₀−1|',
                       'c10sub': 'contact ρ 1|0.1, v = 0.005 (subsonic): |P/P₀−1|',
                       'c1e4': 'surface ρ 1|1e-4 (light side < ρ_sw), v = 0.5: |P/P₀−1|'}[test])
        axx.set_ylim(1e-17, 3)
    ax[0, 0].legend(fontsize=8)
    for test in ('c1e4d2', 'c1e4d3'):   # the c1e4 surface along x2 / x3 (3D: 4x4x256)
        for v in VARS:
            if not os.path.isdir(f'{D}/{test}_{v}'):
                continue
            s = last(f'{D}/{test}_{v}')
            dP, _, n = contact_metrics(s, 1e-6, 1.0)
            rows.append(f'{test:7s} {v:7s} max|dP/P| {dP:9.2e}  cells with |dP/P|>1e-6: {n}')
    ref = last(f'{D}/sod_off')
    for v in VARS:
        s = last(f'{D}/sod_{v}')
        dip, emax, rip, xsh, Kps, span = sod_metrics(s, ref)
        rows.append(f'sod     {v:7s} plateau x {span[0]:.3f}-{span[1]:.3f}: min dP/P* {dip:+.2e}'
                    f'  max|dP/P*| {emax:.2e}  p90 {rip:.2e}  shock x {xsh:.4f}'
                    f'  post-shock K {Kps:.4f}')
        ax[1, 0].plot(s['x'], s['P'], color=COL[v], lw=1.2, label=LAB[v])
        ax[1, 1].plot(s['x'], s['P'], color=COL[v], lw=1.2)
    ax[1, 0].set_title('Sod t = 0.4: P'); ax[1, 1].set_title('Sod: P, zoom on the plateau')
    ax[1, 1].set_xlim(0.45, 0.95); ax[1, 1].set_ylim(0.22, 0.36)
    for v in VARS:
        s = last(f'{D}/wave_{v}')
        dP = np.abs(s['P'] / 1e-6 - 1)
        rows.append(f'wave    {v:7s} max|dP/P| {dP.max():9.2e}')
        ax[1, 2].semilogy(s['x'], np.maximum(dP, 1e-17), color=COL[v], lw=1.2)
    ax[1, 2].set_title('entropy wave, grid Mach 200, t = 8: |P/P₀−1|')
    for x in ax.flat:
        x.set_xlabel('x'); x.grid(alpha=0.3)
    fig.suptitle('T-13 F: D·K vs D·K^(1/Γ) (ppmx + HLLE + FOFC, 256 cells, CPU)')
    fig.savefig(f'{out}/de_F_tests.png', dpi=100)
    with open(f'{out}/de_F_table.txt', 'w') as f:
        f.write('\n'.join(rows) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
