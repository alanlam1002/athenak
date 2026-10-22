#!/usr/bin/env python3
"""C-094 (R-097 item 1(a)): the vacuum null test on the existing Milestone 1 outputs.

The exact discrete answer in vacuum is delta_psi = delta_(alpha psi) = 0, i.e. alpha = alpha0,
psi = psi0 (R-096). For each adm_xy dump (z = 0 slice, every MeshBlock cell) of the vacuum runs,
in BH-centred coordinates (x_BH(t) = puncture_x - frame_vel t when puncture_moving), report for
cells with r_BH <= RMAX:
  - the number of cells at the alpha floor (alpha <= 1.0000001 alpha_floor), and their max r;
  - min alpha / alpha0, and where;
  - along the four in-plane radial lines (+x, -x, +y, -y; the cell row nearest to the line),
    alpha, alpha0, alpha/alpha0, delta_psi = psi - psi0, delta_(alpha psi) = alpha psi - alpha0 psi0
    at r <= RMAX.
alpha0, psi0 are the static analytic trumpet (m1_vacuum.analytic, the port of cfc_puncture.hpp).
Stored alpha is post-floor (CFC::ApplyAlphaFloor, AssembleLapseShiftK), so a negative solved lapse
shows as alpha = alpha_floor.

    c094_vacnull.py RUNDIR [...] [--rmax 3] [--which first,mid,last] [--lines]
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m1_vacuum as V                              # noqa: E402
from tde_ensemble_plots import bc                  # noqa: E402


def setup(d):
    txt = open(f'{d}/parfile.par').read()
    mov = re.search(r'^puncture_moving\s*=\s*true', txt, re.M) is not None
    return (V.par(txt, 'cfc', 'puncture_x1'), V.par(txt, 'cfc', 'frame_vel1'), mov,
            V.par(txt, 'cfc', 'alpha_floor'))


def cells(fn, x0, v, mov, rmax):
    p = bc.read_binary(fn)
    t = p['time']
    xb = x0 - (v * t if mov else 0.0)
    g = np.asarray(p['mb_geometry']); nx = p['nx1_out_mb']
    md = {k: np.asarray(q)[:, 0] for k, q in p['mb_data'].items()}
    dx = ((g[:, 1] - g[:, 0]) / nx)[:, None, None]
    X = g[:, 0, None, None] + (np.arange(nx)[None, None, :] + .5) * dx
    Y = g[:, 2, None, None] + (np.arange(nx)[None, :, None] + .5) * dx
    sh = md['adm_alpha'].shape
    X, Y, D = (np.broadcast_to(a, sh) for a in (X, Y, dx))
    rx, ry = X - xb, Y
    r = np.hypot(rx, ry)
    s = r <= rmax
    out = dict(t=t, xb=xb, rx=rx[s], ry=ry[s], r=r[s], dx=D[s], al=md['adm_alpha'][s],
               psi=md['adm_psi4'][s] ** 0.25)
    p4a, ala, _, _ = V.analytic(out['rx'], out['ry'])
    out['al0'] = ala
    out['psi0'] = p4a ** 0.25
    return out


def report(d, fn, rmax, lines, out):
    x0, v, mov, afl = setup(d)
    c = cells(fn, x0, v, mov, rmax)
    fl = c['al'] <= 1.0000001 * afl
    ratio = c['al'] / np.maximum(c['al0'], 1e-300)
    ok = c['al0'] > 0
    i = np.argmin(np.where(ok, ratio, np.inf))
    dpsi = c['psi'] - c['psi0']
    dap = c['al'] * c['psi'] - c['al0'] * c['psi0']
    unfl = ~fl
    out.append(f'{os.path.basename(d):14s} {fn.split(".")[-2]} t {c["t"]:8.3f} x_BH {c["xb"]:+8.3f} '
               f'dx_min {c["dx"].min():.4f}: cells r<={rmax} {len(c["r"])}, floored {fl.sum()}'
               + (f' (r_max {c["r"][fl].max():.3f})' if fl.any() else '')
               + f'; min alpha/alpha0 {ratio[i]:.4e} at r {c["r"][i]:.3f}'
               f'; unfloored max|d_psi| {np.abs(dpsi[unfl]).max():.3e} '
               f'max|d_(alpha psi)| {np.abs(dap[unfl]).max():.3e} '
               f'max|alpha/alpha0-1| (alpha0>0.3) {np.abs(ratio[unfl & (c["al0"] > 0.3)] - 1).max():.3e}')
    if not lines:
        return
    for name, sel in (('+x', (c['rx'] > 0) & (np.abs(c['ry']) < c['dx'])),
                      ('-x', (c['rx'] < 0) & (np.abs(c['ry']) < c['dx'])),
                      ('+y', (c['ry'] > 0) & (np.abs(c['rx']) < c['dx'])),
                      ('-y', (c['ry'] < 0) & (np.abs(c['rx']) < c['dx']))):
        k = np.argsort(c['r'][sel])
        out.append(f'   line {name}: r  alpha  alpha0  alpha/alpha0  d_psi  d_(alpha psi)')
        for j in np.nonzero(sel)[0][k]:
            out.append(f'     {c["r"][j]:6.3f} {c["al"][j]:.6e} {c["al0"][j]:.6e} {ratio[j]:+.6e} '
                       f'{dpsi[j]:+.3e} {dap[j]:+.3e}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--rmax', type=float, default=3.0)
    ap.add_argument('--which', default='first,mid,last')
    ap.add_argument('--lines', action='store_true')
    ap.add_argument('--out')
    a = ap.parse_args()
    out = []
    for d in a.runs:
        fs = sorted(glob.glob(f'{d}/bin/*adm_xy*.bin'))
        pick = {'first': fs[0], 'mid': fs[len(fs) // 2], 'last': fs[-1]}
        sel = fs if a.which == 'all' else [pick[w] for w in a.which.split(',')]
        for fn in sel:
            report(d, fn, a.rmax, a.lines, out)
    txt = '\n'.join(out)
    print(txt)
    if a.out:
        open(a.out, 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
