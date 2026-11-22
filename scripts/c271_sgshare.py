#!/usr/bin/env python3
"""c271 (R-264 item 1, user-approved 2026-11-18): the S5 guard's self-gravitating share vs t, offline, on a run's prim_3d dumps.

Replica of CFC::MeasureSelfGravShare (cfc.cpp, wip/bind-refine 95fc5c9b :3647; wip/s5-phaseout 4cf36891):
  over cells with rho > frame_track_rel_floor and r_BH >= 6,
  share = sum(rho dV; rho > rho_tid = 3 M_BH/(4 pi r_BH^3)) / sum(rho dV)   (coordinate dV = dx^3; w0 density),
  x_BH(t) = puncture_x - frame_disp(t) (cfc.hpp:734 BHPosition; frame_disp = user.hst columns 8-10, interpolated in t).
Also (diagnostic, not the guard): the same share at Lambda = rho/rho_tid > 0.5 and > 2, the Lambda > 1 mass itself and its r range.
    c271_sgshare.py RUN_DIR [first_id last_id]
"""
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'vis', 'python'))
import bin_convert as bc   # noqa: E402

B = 'cfc_tde_wd_imbh_parabolic_tracker'


def deck(d, key):
    par = open(f'{d}/parfile.par').read()
    return float(re.search(rf'^\s*{key}\s*=\s*(\S+)', par, re.M).group(1))


def one(fn, d, h):
    p = bc.read_binary(fn)
    t = p['time']
    g = np.asarray(p['mb_geometry'])
    nx = p['nx1_out_mb']
    rho = np.asarray(p['mb_data']['dens'])            # (nmb, nz, ny, nx)
    disp = np.array([np.interp(t, h[:, 0], h[:, 7 + a]) for a in range(3)])
    xbh = np.array([deck(d, f'puncture_x{a}') for a in (1, 2, 3)]) - disp
    mbh, rfl = deck(d, 'puncture_mass'), deck(d, 'frame_track_rel_floor')
    dx = (g[:, 1] - g[:, 0]) / nx
    c = (np.arange(nx) + 0.5)
    X = g[:, 0, None] + c[None, :] * dx[:, None] - xbh[0]
    Y = g[:, 2, None] + c[None, :] * dx[:, None] - xbh[1]
    Z = g[:, 4, None] + c[None, :] * dx[:, None] - xbh[2]
    r = np.sqrt(Z[:, :, None, None]**2 + Y[:, None, :, None]**2 + X[:, None, None, :]**2)
    dm = rho * (dx**3)[:, None, None, None]
    sel = (rho > rfl) & (r >= 6.0)
    lam = np.where(sel, rho * (4 * np.pi * r**3) / (3 * mbh), 0.0)
    den = dm[sel].sum()
    out = dict(t=t, mtot=den)
    for th in (0.5, 1.0, 2.0):
        out[th] = dm[sel & (lam > th)].sum() / den if den > 0 else 0.0
    s1 = sel & (lam > 1.0)
    out['m1'] = dm[s1].sum()
    out['rmin'], out['rmax'] = (r[s1].min(), r[s1].max()) if s1.any() else (np.nan, np.nan)
    out['lammax'] = lam.max()
    return out


if __name__ == '__main__':
    d = sys.argv[1]
    fs = sorted(glob.glob(f'{d}/bin/{B}.prim_3d.*.bin'))
    if len(sys.argv) > 3:
        lo, hi = int(sys.argv[2]), int(sys.argv[3])
        fs = [f for f in fs if lo <= int(f.split('.')[-2]) <= hi]
    h = np.loadtxt(f'{d}/{B}.user.hst')
    print(f'# {d}\n#   t        share(L>1)=guard  share(L>0.5)  share(L>2)  M(L>1)      M(rho>floor,r>=6)  r range of L>1     max L')
    for fn in fs:
        o = one(fn, d, h)
        print(f"{o['t']:9.3f}  {o[1.0]:.4e}        {o[0.5]:.4e}    {o[2.0]:.4e}  {o['m1']:.4e}  {o['mtot']:.4e}         "
              f"{o['rmin']:7.1f}-{o['rmax']:7.1f}  {o['lammax']:.3e}", flush=True)
