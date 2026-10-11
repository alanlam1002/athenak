#!/usr/bin/env python3
"""Q2 (R-038): warm-entropy fraction of the isolated boosted star from z = 0 slices, L4 vs L5,
BH frame (V3FJ) vs comoving (GJ). A slice proxy for scripts/m3_entropy.py (3D restarts).

    q2_warm_slice.py
"""
import glob, sys
sys.path.insert(0, '/home/tlam/athenak_tde/scripts')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from tde_ensemble_plots import bc
C = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc'
K0, RHOC, G = 1.2346626297e-2, 3.24e-4, 5/3
R = {'V3FJ L4': ('de_V3FJ_boost_contract_L4', '#e7298a', '--'), 'V3FJ L5 (Q2)': ('Q2_V3FJ_L5', '#e7298a', '-'),
     'GJ L4': ('de_GJ_G2c_DE_L4', '#e6ab02', '--'), 'GJ L5 (Q2)': ('Q2_GJ_L5', '#e6ab02', '-')}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
for lab, (d, c, ls) in R.items():
    T, W, M = [], [], []
    for f in sorted(glob.glob(f'{C}/{d}/bin/*prim_xy*.bin'))[::2]:
        p = bc.read_binary(f); g = np.asarray(p['mb_geometry']); nx = p['nx1_out_mb']
        dn = np.asarray(p['mb_data']['dens'])[:, 0]; P = np.asarray(p['mb_data']['press'])[:, 0]
        a = (((g[:, 1] - g[:, 0]) / nx)**2)[:, None, None] * np.ones_like(dn)
        s = dn > 0.1 * RHOC; w = (dn * a)[s]; x = (P / dn**G / K0 - 1)[s]
        T.append(p['time']); W.append(w[x > 0.05].sum() / w.sum()); M.append(np.average(x, weights=w))
    print(lab, f't={T[-1]:.0f} warm>5% max {max(W):.4f} last {W[-1]:.4f}; mean K/K0-1 last {M[-1]:+.4f}', flush=True)
    ax[0].plot(T, W, ls, color=c, label=lab); ax[1].plot(T, M, ls, color=c, label=lab)
ax[0].set_title('z=0 slice proxy: fraction of dense (ρ>0.1ρ_c) slice mass with K/K0−1 > 5%', fontsize=9)
ax[1].set_title('z=0 slice: ρ-weighted mean K/K0 − 1 over dense gas', fontsize=9)
for a in ax: a.set_xlabel('t [M]'); a.grid(alpha=.3); a.legend(fontsize=8)
fig.suptitle('Q2: isolated boosted star, BH frame (V3FJ) vs comoving (GJ), L4 vs L5 (slice-based, not the 3D M3 metric)', fontsize=10)
fig.savefig(f'{C}/plots/dual_energy/Q2_warm_slice.png', dpi=100)
