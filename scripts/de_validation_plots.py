#!/usr/bin/env python3
"""T-13 dual-energy validation (src/cfc/DEVELOPMENT.md items 69/70; research R-028 C).

  V2   static WD, L4, dual energy on                        vs G0 (off)
  V3   boosted WD, contracted ID, BH frame, dual energy on  vs V3c (same ID, off),
       the old boosted run (old ID, off) and G2c (comoving gauge, contracted ID, off)
  V4   rp16 L4, item-69 ID, dual energy on, t <= 160        vs ens1 rp16 (old ID, off)
  G    G2c + dual energy (comoving gauge, research R-030 G)  vs G2c (off) and V3

Per dump (z=0 slices, tde_t11_surface_diag.measure): rho_max, K/K_0 at the max, min K/K_0
over rho > 0.1 rho_0, puffed mass fraction; plus the history's dual-energy counters
(de-ndense, de-nentropy, de-dE-total) when present. Writes de_V23_summary.png,
de_V4_summary.png and a table.

    de_validation_plots.py [--out DIR] [--v4-only] [--no-v4]
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gauge_g2g3_plots as G      # noqa: E402  (registers G0/G2/G2c/G3/G3b)
import tde_t11_surface_diag as T  # noqa: E402

RUN, ISO, TDE = T.RUN, T.ISO, 'cfc_tde_wd_imbh_parabolic_tracker'
T.RUNS.update({
    'V2':  (f'{RUN}/de_V2_static_L4', ISO, None, '#1b9e77', '--'),
    'V3':  (f'{RUN}/de_V3_boost_contract_L4', ISO, None, '#d95f02', '--'),
    'V3c': (f'{RUN}/de_V3c_boost_contract_noDE_L4', ISO, None, '#7570b3', '-'),
    'V4':  (f'{RUN}/de_V4_rp16_L4', TDE, 16, '#e7298a', '-'),
    'G2cDE': (f'{RUN}/de_G_G2c_DE_L4', ISO, None, '#a6761d', '-'),   # R-030 G
    'V3F': (f'{RUN}/de_V3F_boost_contract_L4', ISO, None, '#1b9e77', '-'),   # R-031 I
    'V3FJ': (f'{RUN}/de_V3FJ_boost_contract_L4', ISO, None, '#e7298a', '-'),  # F + J
    'GJ': (f'{RUN}/de_GJ_G2c_DE_L4', ISO, None, '#e6ab02', '-'),             # F + J
    'V4F': (f'{RUN}/de_V4F_rp16_L4', TDE, 16, '#1b9e77', '-'),                # R-031 K
})
T.RUNS['iso_boost'] = T.RUNS['iso_boost'][:3] + ('0.55', ':')         # was V2's colour
NAMES = {'G0': 'G0 static, DE off', 'V2': 'V2 static, DE on',
         'iso_boost': 'boosted, old ID, DE off', 'V3c': 'boosted, contracted ID, DE off',
         'V3': 'V3 boosted, contracted ID, DE on', 'G2c': 'G2c comoving, contracted ID, DE off',
         'rp16': 'rp16 ens1 (old ID, DE off)', 'V4': 'V4 rp16 (new ID, DE on)',
         'G2cDE': 'G: G2c + DE (D·K)', 'V3F': 'V3F boosted, contracted ID, DE with F',
         'V3FJ': 'V3FJ boosted, DE with F + J', 'GJ': 'GJ: G2c + DE with F + J',
         'V4F': 'V4F rp16 (new ID, DE with F + J)'}


def style(lab):
    _, _, _, c, ls = T.RUNS[lab]
    return dict(color=c, ls=ls, lw=1.5, label=NAMES.get(lab, lab))


def panels(S, labs, out, name, title, tmax):
    fig, ax = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    for lab in labs:
        if lab not in S:
            continue
        s = S[lab]; k = s['t'] <= tmax
        h = G.hst_cols(lab, ['rho-max', 'de-ndense', 'de-nentropy', 'de-dE-total'])
        if 'rho-max' in h:
            t, v = h['rho-max']; kk = t <= tmax
            ax[0, 0].semilogy(t[kk], v[kk] / v[0], **style(lab))
        ax[0, 1].semilogy(s['t'][k], np.maximum(s['Kmax'][k], 1e-20), **style(lab))
        ax[0, 2].semilogy(s['t'][k], np.maximum(s['Kmin01'][k], 1e-20), **style(lab))
        ax[1, 0].semilogy(s['t'][k], np.maximum(s['puff'][k], 1e-6), **style(lab))
        if 'de-nentropy' in h and 'de-ndense' in h:
            t, ne = h['de-nentropy']; _, nd = h['de-ndense']; kk = t <= tmax
            ax[1, 1].plot(t[kk], ne[kk] / np.maximum(nd[kk], 1), **style(lab))
        if 'de-dE-total' in h:
            t, v = h['de-dE-total']; kk = t <= tmax
            ax[1, 2].plot(t[kk], v[kk], **style(lab))
    ttl = [r'$\rho_{max}/\rho_{max}(0)$ (hst, 3D)', r'$K/K_0$ at the density max',
           r'min $K/K_0$ over $\rho>0.1\rho_0$', r'slice-mass fraction beyond $R_*$',
           'fraction of dense cells on the entropy branch', r'cumulative $\int\sqrt{\gamma}\,\Delta\tau\,dV$ from the resync']
    for x, tl in zip(ax.flat, ttl):
        x.set_title(tl); x.set_xlabel('t [M]'); x.grid(alpha=0.3)
    for x in (ax[0, 1], ax[0, 2]):
        x.axhline(1, color='0.5', lw=0.8)
    ax[0, 0].legend(fontsize=8)
    fig.suptitle(title)
    fig.savefig(f'{out}/{name}', dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{RUN}/plots/dual_energy')
    ap.add_argument('--v4-only', action='store_true')
    ap.add_argument('--no-v4', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    S = {}
    groups = []
    if not a.v4_only:
        groups.append((['G0', 'V2', 'iso_boost', 'V3c', 'V3', 'G2c'], 'de_V23_summary.png',
                       'T-13 V2/V3: isolated WD, L4 (dual energy on vs off; contracted ID)', 100))
        groups.append((['G0', 'V2', 'V3c', 'V3', 'V3F', 'V3FJ', 'G2c', 'G2cDE', 'GJ'],
                       'de_FJ_summary.png', 'F + J: V3FJ (BH-frame boosted) and GJ (comoving) '
                       'vs earlier variants', 100))
        groups.append((['G0', 'V3c', 'V3', 'V3F'], 'de_V3F_summary.png',
                       'R-031 I: V3F (BH-frame boosted star, DE with F = D·K^(1/Γ)) vs V3 (D·K)'
                       ' and V3c (DE off)', 100))
        groups.append((['G0', 'V2', 'G2c', 'G2cDE', 'V3c', 'V3'], 'de_G_summary.png',
                       'R-030 G: G2c (comoving gauge) with dual energy (D·K), vs V3 (BH frame)',
                       100))
    if not a.no_v4:
        groups.append((['rp16', 'V4', 'V4F'], 'de_V4_summary.png',
                       'T-13 V4/V4F: rp16 (beta=1.03), L4 -- the T-11 test', 160))
    rows = []
    for labs, name, title, tmax in groups:
        for lab in labs:
            d = T.RUNS[lab][0]
            if os.path.isdir(f'{d}/bin'):
                S[lab] = T.series(lab, a.out)
        panels(S, labs, a.out, name, title, tmax)
        for lab in labs:
            if lab not in S:
                continue
            s = S[lab]
            for t in (40, 100, 120, 160):
                k = np.argmin(abs(s['t'] - t))
                if abs(s['t'][k] - t) > 1 or t > tmax:
                    continue
                rows.append(f"{lab:10s} {s['t'][k]:6.1f} rho_max/rho0 {s['rhomax'][k]:.4f} "
                            f"Kmax {s['Kmax'][k]:.4g} Kmin01 {s['Kmin01'][k]:.4g} "
                            f"puff {s['puff'][k]:.3e}")
    with open(f'{a.out}/de_validation_table.txt', 'w') as f:
        f.write('\n'.join(rows) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
