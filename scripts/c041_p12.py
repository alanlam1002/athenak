#!/usr/bin/env python3
"""C-040 P1/P2 (user-approved 2026-10-12): BH-frame switch maps, and the dual-energy flag
fix candidates on the item 6 L4 restart (t = 60.7 -> 72).

  P1   D_e (BH frame, corrected fill) restarted at t = 47.0 -> 68 with <mhd> dual_energy_diag:
       the switch census (d_r043_maps.census) next to arm (a)'s comoving census.
  P2   iso (isolated flagged cells demoted), p3 / p6 (flag persistence 3 / 6 C2P calls),
       isop3; (iii) si0 (metric solved every RK stage), cfl02 (half dt): dense mean K/K0
       (rho > 0.1 rho0, z = 0) and K at rho_max against L4, (a), (c).

    c041_p12.py [--out DIR]
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
import d_r043_maps as R                           # noqa: E402

RUN = T.RUN
B = R.B
P2 = {'L4': 'T6b_rp10_comov_bhL4', 'a': 'D_a_dediag', 'c': 'D_c_noshock', 'iso': 'P2_iso',
      'p3': 'P2_p3', 'p6': 'P2_p6', 'isop3': 'P2_isop3', 'si0': 'T3_si0', 'cfl02': 'T3_cfl02'}
P1 = {'P1 BH frame': 'P1_bh_dediag', 'a comoving': 'D_a_dediag'}
COL = {'L4': 'k', 'a': '0.5', 'c': 'C0', 'iso': 'C1', 'p3': 'C2', 'p6': 'C3', 'isop3': 'C4',
       'si0': 'C5', 'cfl02': 'C6'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{RUN}/plots/phase2_m1')
    a = ap.parse_args()
    rows = []
    fig, ax = plt.subplots(1, 3, figsize=(19, 5), constrained_layout=True)
    for lab, d in P2.items():
        if not glob.glob(f'{RUN}/{d}/bin/{B}.prim_xy.*.bin'):
            continue
        T.RUNS[f'C41_{lab}'] = (f'{RUN}/{d}', B, 10, 'k', '-')
        s = T.series(f'C41_{lab}', f'{RUN}/plots/dual_energy')
        sel = (s['t'] >= 60) & (s['t'] <= 72.01)
        ls = '--' if lab in ('L4', 'a', 'c') else 'o-'
        ax[0].plot(s['t'][sel], s['Kw01'][sel], ls, ms=3, color=COL[lab], label=lab)
        ax[1].plot(s['t'][sel], s['Kmax'][sel], ls, ms=3, color=COL[lab], label=lab)
        pick = [62, 64, 66, 68, 70, 72]
        rows.append(f'P2 {lab:6s} Kw01: ' + ', '.join(
            f't={t:g} {np.interp(t, s["t"], s["Kw01"]):.4f}' for t in pick if s['t'][sel].max() >= t - 0.3)
            + f' | Kmax at {s["t"][sel][-1]:.1f}: {s["Kmax"][sel][-1]:.4f}')
    ax[0].set_title('dense mean K/K0 (ρ > 0.1 ρ0, z = 0)'); ax[1].set_title('K/K0 at ρ_max')
    # P1 vs (a): census time series
    for lab, d in P1.items():
        ns = sorted(int(f.split('.')[-2]) for f in glob.glob(f'{RUN}/{d}/bin/{B}.dediag_xy.*.bin'))
        if not ns:
            continue
        C = np.array([R.census(R.load(f'{RUN}/{d}', 'xy', n)) for n in ns])
        np.savetxt(f'{a.out}/c041_P1_census_{d}.txt', C, header=R.HDR, fmt='%.5g')
        en_frac = (C[:, 5] + C[:, 6]) / C[:, 1]
        ax[2].plot(C[:, 0], C[:, 8], 'o-', ms=3, label=f'{lab}: entropy-branch ⟨K_en/K_tr − 1⟩')
        ax[2].plot(C[:, 0], en_frac, 's--', ms=3, label=f'{lab}: energy-branch fraction')
        late = C[:, 0] >= C[-1, 0] - 7.5
        rows.append(f'P1 {lab}: t={C[late, 0].min():.1f}-{C[late, 0].max():.1f} '
                    f'entropy-branch <Ken/Ktr-1> mean {np.nanmean(C[late, 8]):+.4f} '
                    f'(p99 mean {np.nanmean(C[late, 9]):+.4f}); energy-branch fraction mean '
                    f'{en_frac[late].mean():.4f}; resync <dK> {np.nanmean(C[late, 13]):+.4f}; '
                    f'flagged fraction {np.nanmean(C[late, 18]):.4f}')
    ax[2].set_title('switch census, xy plane: P1 (BH frame) vs (a) (comoving)')
    ax[2].set_yscale('symlog', linthresh=1e-3)
    for x in ax:
        x.set_xlabel('t [M]'); x.grid(alpha=0.3); x.legend(fontsize=7)
    fig.suptitle('C-040 P1/P2: BH-frame switch maps; flag-fix candidates on item 6 L4 (t = 60.7 → 72)')
    fig.savefig(f'{a.out}/c041_p12.png', dpi=80)
    with open(f'{a.out}/c041_p12.txt', 'w') as f:
        f.write('\n'.join(rows) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
