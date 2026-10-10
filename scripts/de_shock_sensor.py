#!/usr/bin/env python3
"""Shock-sensor diagnostics for the dual-energy flag (research R-031 J; DEVELOPMENT item 72).

From z = 0 bin slices (in-plane x, y only: a compression along z is invisible here),
per dump and over dense cells (rho > 0.1 rho_0):
  jump  max_a |dP|/min(P) to a face neighbour            (item 70's flag, > 0.3)
  psi   max_a |P+ - 2P + P-| / (P+ + 2P + P-)             (Jameson second difference)
  comp  -div v dx / c_s                                  (C-020's proposal)
  vcurv max_a |v_a+ - 2 v_a + v_a-| / c_s                 (velocity curvature)
Prints percentiles, plus psi / jump / comp of the cell where psi peaks and its r/R*.

    de_shock_sensor.py [labels ...] [--times t1,t2,...]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tde_t11_surface_diag as T    # noqa: E402
from tde_ensemble_plots import dump_at   # noqa: E402
import bin_convert as bc            # noqa: E402
import de_validation_plots          # noqa: E402,F401  (registers V2/V3/G runs)

GAM = 5.0 / 3.0


def sensors(pf):
    p = bc.read_binary(pf)
    md = p['mb_data']
    d = np.asarray(md['dens'])[:, 0]
    P = np.asarray(md['press'])[:, 0]
    ux = np.asarray(md['velx'])[:, 0]
    uy = np.asarray(md['vely'])[:, 0]
    W = np.sqrt(1 + ux**2 + uy**2)
    vx, vy = ux / W, uy / W
    g = np.asarray(p['mb_geometry'])
    nx = p['nx1_out_mb']
    nan = np.full_like(d, np.nan)
    psi, jump, div = nan.copy(), nan.copy(), nan.copy()
    c = (slice(None), slice(1, -1), slice(1, -1))
    Pc = P[c]
    Pxp, Pxm = P[:, 1:-1, 2:], P[:, 1:-1, :-2]
    Pyp, Pym = P[:, 2:, 1:-1], P[:, :-2, 1:-1]
    psi[c] = np.maximum(np.abs(Pxp - 2 * Pc + Pxm) / (Pxp + 2 * Pc + Pxm),
                        np.abs(Pyp - 2 * Pc + Pym) / (Pyp + 2 * Pc + Pym))
    jump[c] = np.max([np.abs(q - Pc) / np.minimum(q, Pc) for q in (Pxp, Pxm, Pyp, Pym)], axis=0)
    div[c] = (vx[:, 1:-1, 2:] - vx[:, 1:-1, :-2] + vy[:, 2:, 1:-1] - vy[:, :-2, 1:-1]) / 2
    h = 1 + GAM / (GAM - 1) * P / d
    cs = np.sqrt(GAM * P / (d * h))
    comp = -div / cs                       # (-div v) dx / c_s, since div is per dx
    # velocity curvature along each direction, normalised by c_s: zero for homologous
    # (linear) and uniform flow, O(dv/c_s) at a shock
    vcurv = nan.copy()
    vcurv[c] = np.maximum(np.abs(vx[:, 1:-1, 2:] - 2 * vx[c] + vx[:, 1:-1, :-2]),
                          np.abs(vy[:, 2:, 1:-1] - 2 * vy[c] + vy[:, :-2, 1:-1])) / cs[c]
    m, j, i = np.unravel_index(np.argmax(d), d.shape)
    dx = (g[:, 1] - g[:, 0]) / nx
    xc = g[:, 0, None, None] + (np.arange(nx)[None, None, :] + .5) * dx[:, None, None]
    yc = g[:, 2, None, None] + (np.arange(nx)[None, :, None] + .5) * dx[:, None, None]
    xc, yc = np.broadcast_to(xc, d.shape), np.broadcast_to(yc, d.shape)
    r = np.hypot(xc - xc[m, j, i], yc - yc[m, j, i]) / T.RSTAR
    sep = np.hypot(xc[m, j, i], yc[m, j, i])
    return p['time'], d, psi, jump, comp, r, sep, vcurv


def flag_vcurv(psi, vcurv, comp, thr=0.05, pj=0.25, weak=0.5):
    """item 72's default flag, in-plane: div v <= 0 and (psi_v > thr or psi_P > pj), plus one
    hysteresis pass (weak cells next to a flagged one)."""
    cm = np.nan_to_num(comp, nan=-1) >= 0
    psi, vcurv = np.nan_to_num(psi), np.nan_to_num(vcurv)
    strong = cm & ((vcurv > thr) | (psi > pj))
    wk = cm & ((vcurv > weak * thr) | (psi > weak * pj))
    nb = np.zeros_like(strong)
    nb[:, :, 1:] |= strong[:, :, :-1]; nb[:, :, :-1] |= strong[:, :, 1:]
    nb[:, 1:, :] |= strong[:, :-1, :]; nb[:, :-1, :] |= strong[:, 1:, :]
    return strong | (wk & nb)


def report(lab, t):
    pf = dump_at(lab, t)
    if pf is None:
        return None
    tt, d, psi, jump, comp, r, sep, vcurv = sensors(pf)
    dense = (d > 0.1 * T.RHO0) & np.isfinite(psi)
    q = lambda a: np.percentile(a[dense], [50, 90, 99, 100])
    ps, cp, vc = q(psi), q(comp), q(vcurv)
    k = np.argmax(np.where(dense, psi, -1))
    return (f'{lab:6s} t={tt:6.1f} sep {sep:5.1f}  n {dense.sum():4d}  '
            f'psi p50/90/99/max {ps[0]:.3f}/{ps[1]:.3f}/{ps[2]:.3f}/{ps[3]:.3f}  '
            f'comp {cp[0]:+.3f}/{cp[1]:+.3f}/{cp[2]:+.3f}/{cp[3]:+.3f}  '
            f'vcurv {vc[0]:.3f}/{vc[1]:.3f}/{vc[2]:.3f}/{vc[3]:.3f}  '
            f'jump>0.3 {np.mean(jump[dense] > 0.3):.2f}  '
            f'vcurv-flag {flag_vcurv(psi, vcurv, comp)[dense].mean():.3f}  '
            f'psi-max cell r/R* {r.flat[k]:.2f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('labels', nargs='*', default=['V2', 'G2c', 'G2cDE', 'rp16', 'rp12', 'rp10'])
    ap.add_argument('--times', default='0,40,80,100,110,120,130,140,150,160')
    a = ap.parse_args()
    for lab in a.labels:
        for t in [float(x) for x in a.times.split(',')]:
            s = report(lab, t)
            if s:
                print(s, flush=True)


if __name__ == '__main__':
    main()
