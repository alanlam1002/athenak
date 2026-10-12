#!/usr/bin/env python3
"""C-041 (C-040 P2(iii) diagnosis): where does the comoving energy solution's excess come from?

<mhd> dual_energy_diag slots 8/9 (athenak C-041 build) hold the last RK stage's internal-energy
budget W (dE - v^a dS_a) - dD, split into the flux divergence (de_eint_flux) and the source
terms (de_eint_src), each as a fractional rate of W rho eps. Times dt they are per-step
fractional internal-energy changes, comparable with de_ken_ktr (K_energy/K_tracer - 1, the
excess the entropy branch discards each step).

For dense cells (branch > 0) in the xy plane, per dump: mean / p1 / p99 of dt*flux, dt*src,
dt*(flux + src) and K_en/K_tr - 1, split into rim (rho < 0.1 rho_max) and interior, for the
comoving run (B1) and the BH frame (B2) at the same epoch (t = 61-64).

    c041_eint_budget.py [--out DIR]
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tde_t11_surface_diag as T                  # noqa: E402,F401 (sets up bc)
from tde_ensemble_plots import bc                 # noqa: E402

RUN = T.RUN
B = 'cfc_tde_wd_imbh_parabolic_tracker'
ARMS = {'comoving (B1)': 'B1_comov_eint', 'BH frame (B2)': 'B2_bh_eint'}


def dt_at(d, t):
    r = [(float(m.group(1)), float(m.group(2))) for m in
         (re.search(r'time=([-+.\deE]+) dt=([-+.\deE]+)', l) for l in open(f'{d}/ensemble.out')) if m]
    r = np.array(r)
    return float(np.interp(t, r[:, 0], r[:, 1]))


def stats(x):
    return f'{np.mean(x):+.2e} [{np.percentile(x, 1):+.2e}, {np.percentile(x, 99):+.2e}]' if x.size else 'n/a'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{RUN}/plots/phase2_m1')
    a = ap.parse_args()
    rows = []
    for lab, d in ARMS.items():
        d = f'{RUN}/{d}'
        acc = {k: [] for k in ('fl', 'sr', 'tot', 'ken')}
        for f in sorted(glob.glob(f'{d}/bin/{B}.dediag_xy.*.bin')):
            q = bc.read_binary(f)
            if q['time'] < 61.0:
                continue
            dt = dt_at(d, q['time'])
            md = q['mb_data']
            br = np.asarray(md['de_branch'])[:, 0]
            dense = (br > 0.5) & (br < 2.5)          # entropy branch: the energy excess is measured
            fl = dt * np.asarray(md['de_eint_flux'])[:, 0][dense]
            sr = dt * np.asarray(md['de_eint_src'])[:, 0][dense]
            ken = np.asarray(md['de_ken_ktr'])[:, 0][dense]
            for k, v in (('fl', fl), ('sr', sr), ('tot', fl + sr), ('ken', ken)):
                acc[k].append(v)
        if not acc['fl']:
            continue
        rows.append(f'{lab}: entropy-branch dense cells, t >= 61, {len(acc["fl"])} dumps, '
                    f'mean [p1, p99] of per-step fractional internal-energy change:')
        for k, nm in (('fl', 'flux divergence'), ('sr', 'sources'), ('tot', 'flux + sources'),
                      ('ken', 'K_en/K_tr - 1 (measured excess)')):
            rows.append(f'   {nm:32s} {stats(np.concatenate(acc[k]))}')
        x = np.concatenate(acc['tot']); y = np.concatenate(acc['ken'])
        rows.append(f'   corr(flux + sources, K_en/K_tr - 1) = {np.corrcoef(x, y)[0, 1]:.3f}; '
                    f'corr(flux, excess) = {np.corrcoef(np.concatenate(acc["fl"]), y)[0, 1]:.3f}; '
                    f'corr(sources, excess) = {np.corrcoef(np.concatenate(acc["sr"]), y)[0, 1]:.3f}')
    with open(f'{a.out}/c041_eint_budget.txt', 'w') as f:
        f.write('\n'.join(rows) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
