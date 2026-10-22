#!/usr/bin/env python3
"""C-100 (R-112 item 5): domain check for the 2 x 2 comparison (offline).

Per prim_xy/adm_xy dump (the z = 0 orbital plane) of a run, against t:
  - the extent of gas with rho > 1e-10 rho_c: max |x|, max |y| (grid coordinates; the box is the
    deck's x1min..x1max) and the largest distance from the grid origin;
  - the same for BOUND gas (rho > 1e-10 rho_c and E = -u_t - 1 < 0, the BH-frame Killing energy:
    -u_t = alpha W - (beta^i - xidot^i) u_i, with u_i = g_ij u~^j, W = sqrt(1 + u~^i u_i), the
    stored velocities being u~^i = W v^i; xidot from user.hst frame_vel_x*, 0 in the BH frame;
    enthalpy ignored, as tde_dmde.py's E);
  - the BH's grid position (puncture_x - frame_disp from user.hst; the origin in the BH frame).
ARGUED: the in-plane extent bounds the 3D one (the debris is a thin stream in the orbital plane).

    c100_domain.py LABEL:RUNDIR [...] [--every 5] [--out FILE]
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tde_ensemble_plots import bc                  # noqa: E402
import tde_dmde as D                               # noqa: E402

BASE = 'cfc_tde_wd_imbh_parabolic_tracker'
RHOC, THR = 3.24e-4, 1e-10


def parv(txt, key, dflt=0.0):
    m = re.search(rf'^\s*{key}\s*=\s*([-+.\deE]+)', txt, re.M)
    return float(m.group(1)) if m else dflt


def series(d, every):
    txt = open(f'{d}/parfile.par').read() if os.path.exists(f'{d}/parfile.par') else \
        D.header(sorted(glob.glob(f'{d}/rst/*.rst'))[0])['par']
    moving = re.search(r'^\s*puncture_moving\s*=\s*true', txt, re.M) is not None
    px = [parv(txt, f'puncture_x{a}') for a in (1, 2, 3)]
    half = parv(txt, 'x1max', 64.0)
    u = np.loadtxt(glob.glob(f'{d}/{BASE}.user.hst')[0])
    _, i = np.unique(u[:, 0], return_index=True); u = u[i]
    hdr = open(glob.glob(f'{d}/{BASE}.user.hst')[0]).readlines()[1]
    has_frame = 'frame_disp_x1' in hdr
    rows = []
    for fp in sorted(glob.glob(f'{d}/bin/{BASE}.prim_xy.*.bin'))[::every]:
        fa = fp.replace('prim_xy', 'adm_xy')
        if not os.path.exists(fa):
            continue
        q, a = bc.read_binary(fp), bc.read_binary(fa)
        t = q['time']
        g = np.asarray(q['mb_geometry']); nx = q['nx1_out_mb']
        md = {k: np.asarray(v)[:, 0] for k, v in q['mb_data'].items()}
        ad = {k: np.asarray(v)[:, 0] for k, v in a['mb_data'].items()}
        X = g[:, 0, None, None] + (np.arange(nx)[None, None, :] + .5) * ((g[:, 1] - g[:, 0]) / nx)[:, None, None]
        Y = g[:, 2, None, None] + (np.arange(nx)[None, :, None] + .5) * ((g[:, 3] - g[:, 2]) / nx)[:, None, None]
        X, Y = np.broadcast_to(X, md['dens'].shape), np.broadcast_to(Y, md['dens'].shape)
        s = md['dens'] > THR * RHOC
        if has_frame and moving:
            xd = [np.interp(t, u[:, 0], u[:, 4 + k]) for k in range(3)]
            xb = [px[k] - np.interp(t, u[:, 0], u[:, 7 + k]) for k in range(3)]
        else:
            xd, xb = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        ut = [md['velx'][s], md['vely'][s], md['velz'][s]]
        gd = [[ad['adm_gxx'][s], ad['adm_gxy'][s], ad['adm_gxz'][s]],
              [ad['adm_gxy'][s], ad['adm_gyy'][s], ad['adm_gyz'][s]],
              [ad['adm_gxz'][s], ad['adm_gyz'][s], ad['adm_gzz'][s]]]
        ud = [sum(gd[i][j] * ut[j] for j in range(3)) for i in range(3)]
        W = np.sqrt(1 + sum(ut[i] * ud[i] for i in range(3)))
        bet = [ad['adm_betax'][s], ad['adm_betay'][s], ad['adm_betaz'][s]]
        E = ad['adm_alpha'][s] * W - sum((bet[i] - xd[i]) * ud[i] for i in range(3)) - 1
        x, y = X[s], Y[s]
        b = E < 0
        rows.append((t, np.abs(x).max(), np.abs(y).max(), np.hypot(x, y).max(),
                     np.abs(x[b]).max() if b.any() else np.nan, np.abs(y[b]).max() if b.any() else np.nan,
                     np.hypot(x[b], y[b]).max() if b.any() else np.nan, xb[0], xb[1], s.sum()))
    return half, moving, np.array(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--every', type=int, default=5)
    ap.add_argument('--out')
    a = ap.parse_args()
    out = [f'extent of rho > {THR:g} rho_c gas (z = 0 plane) and of its bound part, grid coordinates; x_BH = BH grid position']
    for spec in a.runs:
        lab, d = spec.split(':', 1)
        half, mov, R = series(d, a.every)
        out.append(f'== {lab} ({os.path.basename(d)}, {"comoving" if mov else "BH frame"}, box half-width {half:g}); '
                   f't {R[0, 0]:.0f}-{R[-1, 0]:.0f}')
        out.append('   t      max|x|  max|y|  max r  | bound: max|x| max|y| max r | x_BH    y_BH')
        for r in R:
            out.append(f'   {r[0]:6.1f} {r[1]:7.2f} {r[2]:7.2f} {r[3]:6.2f} | {r[4]:7.2f} {r[5]:7.2f} {r[6]:6.2f} | '
                       f'{r[7]:+7.2f} {r[8]:+7.2f}')
        mx = np.nanmax(np.maximum(R[:, 1], R[:, 2]))
        mb = np.nanmax(np.maximum(R[:, 4], R[:, 5]))
        mbh = np.max(np.maximum(np.abs(R[:, 7]), np.abs(R[:, 8])))
        out.append(f'   MAX over t: gas max(|x|,|y|) {mx:.2f}; bound {mb:.2f}; |x_BH| {mbh:.2f}; '
                   f'1.25 x max(gas, BH) = {1.25 * max(mx, mbh):.1f} (box half-width {half:g})')
    txt = '\n'.join(out)
    print(txt)
    if a.out:
        open(a.out, 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
