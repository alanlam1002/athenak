#!/usr/bin/env python3
"""Phase 2 M1 test 1a (research R-034): a vacuum trumpet moving on the grid.

For each adm_xy slice, in BH-centred coordinates (x_BH(t) = puncture_x - xi(t), xi = frame_vel t):
  - max |alpha - alpha_an| and max |psi4 - psi4_an|/psi4_an for 1.5 <= r_BH <= RMAX, outside
    the excision (alpha > 0.3), where *_an is the static analytic trumpet (scripts/
    tde_elliptical_orbit_ic.py, a port of cfc_puncture.hpp);
  - |beta - beta_P - xidot| over the same cells (the shift must be the trumpet's plus xidot);
  - the offset of the lapse minimum from x_BH (does the puncture sit where it should?).
From ensemble.out: the CFC residual-volume integral I_sum and the ADM mass at r = 20/30/40,
the constraint-like diagnostics the code prints (the Hamiltonian/momentum constraints are
not output directly).

    m1_vacuum.py LABEL=RUNDIR [...] [--rmax 10] [--every 10]
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tde_t11_surface_diag as T                      # noqa: E402,F401 (sets up bc)
from tde_ensemble_plots import bc                     # noqa: E402
import tde_elliptical_orbit_ic as TR                  # noqa: E402


def par(text, blk, key, default=0.0):
    m = re.search(rf'<{blk}>(.*?)(?=\n<|\Z)', text, re.S)
    if not m:
        return default
    mm = re.search(rf'^{key}\s*=\s*([-+.\deE]+)', m.group(1), re.M)
    return float(mm.group(1)) if mm else default


def analytic(dx, dy):
    """psi4, alpha, beta (2D in-plane) of the static trumpet at offsets (dx, dy, 0)"""
    r = np.hypot(dx, dy)
    rho = np.array([TR.trumpet_iso_to_areal(c) for c in r.ravel()]).reshape(r.shape)
    rrs = rho
    rrs3 = rrs**3
    psi = np.sqrt(rho / np.maximum(r, 1e-30))
    alpha = np.sqrt(np.maximum(1.0 - 2.0 / rrs + 1.6875 / (rrs3 * rrs), 0.0))
    bmag = 0.75 * np.sqrt(3.0) / (rho * rrs * rrs)
    return psi**4, alpha, bmag * dx, bmag * dy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--rmax', type=float, default=10.0)
    ap.add_argument('--every', type=int, default=10)
    a = ap.parse_args()
    for spec in a.runs:
        lab, d = spec.split('=', 1)
        txt = open(f'{d}/parfile.par').read()
        x0 = par(txt, 'cfc', 'puncture_x1'); v = par(txt, 'cfc', 'frame_vel1')
        mov = re.search(r'^puncture_moving\s*=\s*true', txt, re.M) is not None
        for fn in sorted(glob.glob(f'{d}/bin/*adm_xy*.bin'))[::a.every]:
            p = bc.read_binary(fn)
            t = p['time']
            xb = x0 - (v * t if mov else 0.0)
            g = np.asarray(p['mb_geometry']); nx = p['nx1_out_mb']
            md = {k: np.asarray(q)[:, 0] for k, q in p['mb_data'].items()}
            dx = ((g[:, 1] - g[:, 0]) / nx)[:, None, None]
            X = g[:, 0, None, None] + (np.arange(nx)[None, None, :] + .5) * dx
            Y = g[:, 2, None, None] + (np.arange(nx)[None, :, None] + .5) * dx
            X, Y = np.broadcast_to(X, md['adm_alpha'].shape), np.broadcast_to(Y, md['adm_alpha'].shape)
            rx, ry = X - xb, Y
            r = np.hypot(rx, ry)
            sel = (r >= 1.5) & (r <= a.rmax) & (md['adm_alpha'] > 0.3)
            p4a, ala, bxa, bya = analytic(rx[sel], ry[sel])
            ea = np.abs(md['adm_alpha'][sel] - ala).max()
            ep = (np.abs(md['adm_psi4'][sel] - p4a) / p4a).max()
            eb = np.hypot(md['adm_betax'][sel] - bxa - (v if mov else 0.0),
                          md['adm_betay'][sel] - bya).max()
            i = np.unravel_index(np.argmin(md['adm_alpha']), md['adm_alpha'].shape)
            off = np.hypot(X[i] - xb, Y[i])
            print(f'{lab:8s} t={t:6.1f} x_BH={xb:+7.2f}  max|d alpha| {ea:.3e}  max|d psi4|/psi4 {ep:.3e}'
                  f'  max|d beta| {eb:.3e}  alpha-min offset {off:.3f} (dx {dx.min():.4f})', flush=True)
        # printed diagnostics
        I, M = [], []
        for line in open(f'{d}/ensemble.out'):
            m = re.search(r'I_sum=([-+.\deE]+)', line)
            if m: I.append(float(m.group(1)))
            m = re.search(r'ADM mass: r=3\.0+e\+01 M_res=([-+.\deE]+)', line)
            if m: M.append(float(m.group(1)))
        if I:
            I, M = np.array(I), np.array(M)
            print(f'{lab:8s} CFC residual volume I_sum: first {I[0]:.3e} last {I[-1]:.3e} max|.| {np.abs(I).max():.3e};'
                  f'  M_res(r=30): first {M[0]:.3e} last {M[-1]:.3e} max|.| {np.abs(M).max():.3e}' if len(M) else '')


if __name__ == '__main__':
    main()
