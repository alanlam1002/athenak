#!/usr/bin/env python3
"""R-040 item 5: run-to-run scatter of the BH-frame V5-L4 deck. N5 repeats Q1 with the same
deck and binary (d6e0071e) to t = 150; any difference is GPU nondeterminism (RESULTS §9.5).

Slices (every 1 M, scripts/tde_t11_surface_diag.measure): rho_max, K at rho_max, puffed
mass (the C-031 Q1-gate metrics). Restarts (tde_ovn_plots.rst_series): M3 warm > 5%, self-
bound mass, bound fraction. Each pair -- N5 vs Q1 (noise), Q1 vs old V5-L4 (binary + noise,
C-031's gate), Q6 vs Q1 (box) -- is reported as the max and mean |relative difference| over
t <= 150, slices matched by dump index (same output cadence), restarts by interpolating
B linearly to A's restart times.

    n5_noise_floor.py [--out DIR]
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
import tde_t11_surface_diag as T                  # noqa: E402
import tde_ovn_plots as O                         # noqa: E402

RUN = T.RUN
TDE = 'cfc_tde_wd_imbh_parabolic_tracker'
for lab, d, c in (('N5', 'N5_Q1_repeat_t150', '#1b9e77'), ('Q1', 'Q1_V5L4_fix_t300', '#d95f02'),
                  ('V5L4', 'de_V5_rp10_L4', '#e6ab02'), ('Q6', 'Q6_V5L4_box128', '#7570b3')):
    T.RUNS[lab] = (f'{RUN}/{d}', TDE, 10, c, '-')
O.RUNS['N5'] = (f'{RUN}/N5_Q1_repeat_t150', '#1b9e77', '-', 'N5  Q1 repeat (same deck + binary)')
TMAX = 150.0
PAIRS = (('N5', 'Q1', 'run-to-run (same deck, binary)'),
         ('Q1', 'V5L4', 'binary fe9a62e8 + run-to-run (C-031 gate)'),
         ('Q6', 'Q1', 'box +-128 vs +-64'))


def reldiff(a, b):
    return np.abs(a - b) / np.maximum(np.abs(b), 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{RUN}/plots/overnight')
    a = ap.parse_args()
    cache = f'{RUN}/plots/dual_energy'
    S = {lab: T.series(lab, cache) for lab in ('N5', 'Q1', 'V5L4', 'Q6')}
    oc = np.load(f'{a.out}/ovn_cache.npz', allow_pickle=True)   # tde_ovn_plots.py's cache
    R = {lab: oc[f'{lab}_rst'] for lab in ('Q1', 'V5L4', 'Q6')}
    nf = f'{a.out}/r040_5_n5_rst.npy'                              # N5's own restart series
    if os.path.exists(nf):
        R['N5'] = np.load(nf)
    else:
        R['N5'] = O.rst_series('N5'); np.save(nf, R['N5'])
    rows = []
    fig, ax = plt.subplots(2, 3, figsize=(17, 8.5), constrained_layout=True)
    for p, (A, B, what) in enumerate(PAIRS):
        sa, sb = S[A], S[B]
        n = min(len(sa['t']), len(sb['t']))
        k = sa['t'][:n] <= TMAX
        t = sa['t'][:n][k]
        for j, key in enumerate(('rhomax', 'Kmax', 'puff')):
            d = reldiff(sa[key][:n][k], sb[key][:n][k])
            rows.append(f'{A} vs {B} ({what}): {key:6s} max {d.max():.4f} mean {d.mean():.4f} '
                        f'(t <= {t[-1]:.0f}; at t <= 100: max {d[t <= 100].max():.4f})')
            ax[0, j].plot(t, d, label=f'{A} vs {B}: {what}')
        ra, rb = R[A], R[B]
        for j, (col, name) in enumerate(((1, 'M3 warm>5%'), (3, 'bound remnant/M*'), (4, 'E<0 fraction'))):
            # the same restart cadence (400 cycles) lands at different times in each run
            # (different dt sequences): interpolate B linearly to A's restart times
            sel = (ra[:, 0] <= TMAX + 1) & (ra[:, 0] >= rb[0, 0]) & (ra[:, 0] <= rb[-1, 0])
            sel &= np.isfinite(ra[:, col])
            d = np.column_stack([ra[sel, 0], ra[sel, col], np.interp(ra[sel, 0], rb[:, 0], rb[:, col])])
            if len(d):
                ad = np.abs(d[:, 1] - d[:, 2])
                rows.append(f'{A} vs {B}: {name:16s} max |diff| {ad.max():.4f} (absolute) '
                            f'over {len(d)} restarts; at the last: {d[-1, 1]:.4f} vs {d[-1, 2]:.4f} (t={d[-1, 0]:.0f})')
                ax[1, j].plot(d[:, 0], d[:, 1] - d[:, 2], 'o-', label=f'{A} − {B}')
    for j, key in enumerate(('ρ_max', 'K/K0 at ρ_max', 'puffed mass')):
        ax[0, j].set_title(f'|relative difference|: {key}'); ax[0, j].set_yscale('log')
    for j, name in enumerate(('M3 warm > 5% (3D)', 'self-bound mass / M*', 'bound fraction E < 0')):
        ax[1, j].set_title(f'difference: {name}'); ax[1, j].axhline(0, color='k', lw=0.5)
    for x in ax.flat:
        x.set_xlabel('t [M]'); x.grid(alpha=0.3); x.legend(fontsize=7)
    fig.suptitle('R-040 item 5: BH-frame V5-L4 run-to-run scatter (N5 vs Q1) against the binary and box differences')
    fig.savefig(f'{a.out}/r040_5_noise_floor.png', dpi=90)
    with open(f'{a.out}/r040_5_noise_floor.txt', 'w') as f:
        f.write('\n'.join(rows) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
