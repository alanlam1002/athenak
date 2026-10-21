#!/usr/bin/env python3
"""C-089 (R-092 item 3 = R-090 Q1 Step 1): <Lz> and rest-mass drift against t = 0.

For RC_R6X11 (cold recovery) and each noise-ensemble member (AX2, N1, N2c), all from the
R6 00006 restart: over every 3D restart of the run, c051_s2.rst_pass (cells at r_BH >= 3,
the project's own Lz = (x - x_BH) u_y - (y - y_BH) u_x per cell), report
  <Lz> = sum Lz dM / sum dM  (c051_s2.physics, the C-087 row),
  J_z  = sum Lz dM           (r_BH >= 3),
  M3   = sum dM              (r_BH >= 3),
each relative to the shared start restart 00006; and the grid's total rest mass from the
.mhd.hst (all cells) relative to its first row.

    c089_lzdrift.py [--out FILE]
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c051_s2 as S   # noqa: E402

RUNS = ['RC_R6X11', 'RC_R6AX2', 'RC_R6N1', 'RC_R6N2c']
T0 = f'{S.C}/R6_rp10_comov_L4_fix/rst/cfc_tde_wd_imbh_parabolic_tracker.00006.rst'


def quants(fn):
    z = S.rst_pass(fn)
    dM = z['dM']
    J = float((z['Lz'] * dM).sum())
    M = float(dM.sum())
    return float(z['t']), J / M, J, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out')
    a = ap.parse_args()
    t0, l0, j0, m0 = quants(T0)
    out = [f'start {os.path.basename(T0)} t = {t0:.3f}: <Lz> {l0:.9f}  J_z {j0:.9e}  M(r>=3) {m0:.9e}',
           f'{"run":10s} {"rst":>6s} {"t":>8s} {"<Lz>/<Lz>0-1":>14s} {"J/J0-1":>12s} {"M3/M30-1":>12s}']
    final = {}
    for r in RUNS:
        for fn in sorted(glob.glob(f'{S.C}/{r}/rst/*.rst')):
            t, l, j, m = quants(fn)
            out.append(f'{r:10s} {fn.split(".")[-2]:>6s} {t:8.3f} {l/l0-1:+14.4e} {j/j0-1:+12.4e} '
                       f'{m/m0-1:+12.4e}')
            final[r] = (t, l / l0 - 1, j / j0 - 1, m / m0 - 1)
        h = np.loadtxt(f'{S.C}/{r}/cfc_tde_wd_imbh_parabolic_tracker.mhd.hst')
        _, i = np.unique(h[:, 0], return_index=True)
        h = h[i]
        out.append(f'{r:10s} hst total rest mass t {h[0,0]:.2f} -> {h[-1,0]:.2f}: '
                   f'{h[0,2]:.12e} -> {h[-1,2]:.12e}  rel {h[-1,2]/h[0,2]-1:+.3e}')
    out.append('final (t = 170): run  d<Lz>  dJ  dM3  (relative to t = 0)')
    for r in RUNS:
        t, dl, dj, dm = final[r]
        out.append(f'  {r:10s} t {t:.2f}  {dl:+.4e}  {dj:+.4e}  {dm:+.4e}')
    txt = '\n'.join(out)
    print(txt)
    if a.out:
        open(a.out, 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
