#!/usr/bin/env python3
"""C-100 (R-112 item 3(d)): offline zone-radius test for R6 with the time-dependent excision on.

On R6 restarts, with r < r_f = 2.5 excised and skipped in the CFL (C-098: r_f sits at r_max = 2.5
throughout R6), and the zone modelled as in c046_s0.py / c096_dtwhatif.py (non-excised cells with
r_f <= r < r_s and rho < rho_thr = 1e-10 rho_c get v = beta'/alpha capped at |v|_phys <= 0.9 along
beta', cold), for r_s in {5, 6, 7}: dt, the gain over r_s = 5, the limiting cell (r_iso, rho/rho_c,
alpha, |beta'|_phys/alpha, |beta'|_coord, |lambda|_coord, zone/capped), and the mass the zone touches,
binned in the BH-frame Killing E/dE (E = -u_t - 1, -u_t = alpha W - (beta^i - xidot^i) u_i, enthalpy
ignored as tde_dmde.py; dE = <mhd> atmosphere_zone_de, default 0.677/16.4^2, primitive_solver_hyd.hpp
l. 290) with the zone's edges -6 -3 -2 -1 0 1 2 3 6 (l. 281), the E > 0 bins explicitly.
ARGUED approximations as c046_s0.py (ideal-gas C2P of the stored state; zone not replayed).

    c100_rs.py RESTART [...] [--out FILE]
"""
import argparse
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tde_dmde as D                               # noqa: E402
import c046_s0 as S                                # noqa: E402
import c096_dtwhatif as W                          # noqa: E402
from c045_dt import RHOC, CFL                      # noqa: E402

RF, THR, VMAX = 2.5, 1e-10, 0.9
RS = (5.0, 6.0, 7.0)
EDGES = np.array([-6, -3, -2, -1, 0, 1, 2, 3, 6], float)


def energies(fn, rmax=10.0):
    """per-cell (r, E, dM) for r_BH < rmax, in the order c046_s0.load builds its candidate arrays"""
    t = D.header(fn)
    par = t['par']
    nx, ng, nscal = 16, 4, 1
    no = nx + 2 * ng; nmhd = 5 + nscal; ncc = no**3; bsz = (no + 1) * no * no * 3
    mm = np.memmap(fn, dtype=np.float64, mode='r', offset=t['data_off'], shape=(t['nmb'], t['dsize'] // 8))
    x1min, x2min, x3min, x1max, x2max, x3max = t['size'][:6]
    lev = t['ll'][:, 3]
    w = np.array([x1max - x1min, x2max - x2min, x3max - x3min]); lo = np.array([x1min, x2min, x3min])
    s = slice(ng, ng + nx)
    xbh, xd = D.frame_state(par, t['time'])
    out = []
    for b in range(t['nmb']):
        rec = mm[b]
        cons = rec[:nmhd * ncc].reshape(nmhd, no, no, no)[:, s, s, s]
        adm = rec[nmhd * ncc + bsz:].reshape(17, no, no, no)[:, s, s, s]
        ext = w / 2.0**lev[b]; dx = ext[0] / nx
        ll = t['ll'][b]; ic = np.arange(nx) + 0.5
        Z, Y, X = np.meshgrid(lo[2] + ll[2] * ext[2] + ic * dx, lo[1] + ll[1] * ext[1] + ic * dx,
                              lo[0] + ll[0] * ext[0] + ic * dx, indexing='ij')
        gxx, gxy, gxz, gyy, gyz, gzz = adm[:6]
        det = gxx * (gyy * gzz - gyz * gyz) - gxy * (gxy * gzz - gyz * gxz) + gxz * (gxy * gyz - gyy * gxz)
        sg = np.sqrt(det)
        ixx = (gyy * gzz - gyz * gyz) / det; ixy = (gxz * gyz - gxy * gzz) / det
        ixz = (gxy * gyz - gxz * gyy) / det; iyy = (gxx * gzz - gxz * gxz) / det
        iyz = (gxy * gxz - gxx * gyz) / det; izz = (gxx * gyy - gxy * gxy) / det
        Dd = cons[0] / sg
        Sx, Sy, Sz, tau = cons[1] / sg, cons[2] / sg, cons[3] / sg, cons[4] / sg
        ok = (Dd > 0) & np.isfinite(tau)
        Ssq = ixx * Sx * Sx + iyy * Sy * Sy + izz * Sz * Sz + 2 * (ixy * Sx * Sy + ixz * Sx * Sz + iyz * Sy * Sz)
        rho, P, Wl, h = D.c2p(Dd[ok], Ssq[ok], np.maximum(tau[ok], 0.0))
        uj = [Sx[ok] / (Dd[ok] * h), Sy[ok] / (Dd[ok] * h), Sz[ok] / (Dd[ok] * h)]
        bet = [adm[14][ok] - xd[0], adm[15][ok] - xd[1], adm[16][ok] - xd[2]]
        E = adm[13][ok] * Wl - sum(bet[i] * uj[i] for i in range(3)) - 1.0
        r = np.sqrt((X[ok] - xbh[0])**2 + (Y[ok] - xbh[1])**2 + (Z[ok] - xbh[2])**2)
        c = r < rmax
        if c.any():
            out.append(np.stack([r[c], E[c], (cons[0][ok] * dx**3)[c]]))
    a = np.concatenate(out, axis=1)
    m = re.search(r'^\s*atmosphere_zone_de\s*=\s*([-+.\deE]+)', par, re.M)
    de = float(m.group(1)) if m else 0.677 / (16.4 * 16.4)
    return a[0], a[1], a[2], de


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rst', nargs='+')
    ap.add_argument('--out')
    a = ap.parse_args()
    out = [f'r_f = {RF} excised (CFL-skipped); zone r_f <= r < r_s, rho < {THR:g} rho_c, v capped at {VMAX} (phys); '
           f'touched mass per E/dE bin, edges {EDGES.tolist()}']
    for fn in a.rst:
        t, B, Cc = S.load(fn)
        r_e, E, dM, de = energies(fn)
        # c046's candidate order = this order (same block loop, same ok mask, r < RCAND = 10)
        assert len(r_e) == len(Cc['r']) and np.allclose(r_e, Cc['r']), 'cell order mismatch'
        bco = np.sqrt(Cc['b0']**2 + Cc['b1']**2 + Cc['b2']**2)
        out.append(f'== {os.path.basename(os.path.dirname(os.path.dirname(fn)))}/{fn.split(".")[-2]} t {t:.2f} '
                   f'(dE {de:.3e})')
        base = None
        for rs in RS:
            w = W.variant(B, Cc, 0.25, VMAX, rs, THR)
            zone = w['zone'] & (Cc['r'] >= RF)
            dt = np.where(Cc['r'] < RF, np.inf, w['dt'])
            dmin = min(dt.min(), w['dfar'])
            base = base or dmin
            j = int(np.argmin(dt))
            far = w['dfar'] < dt.min()
            hist, _ = np.histogram(E[zone] / de, bins=EDGES, weights=dM[zone])
            outside = dM[zone & ((E / de < EDGES[0]) | (E / de >= EDGES[-1]))].sum()
            lim = ('far (r >= 10)' if far else
                   f'r {Cc["r"][j]:.3f} rho {Cc["rho"][j] / RHOC:.1e} a {Cc["alpha"][j]:.3f} |b|ph/a {w["vb"][j]:.2f} '
                   f'|b|co {bco[j]:.3f} |lam|co {w["lam"][j]:.3f} '
                   f'{"zone-capped" if w["capped"][j] else ("zone" if w["zone"][j] else "non-zone")}')
            out.append(f'   r_s {rs:g}: dt {dmin:.4f} (x{dmin / base:.3f} vs r_s 5) [{lim}]; zone cells {zone.sum()}, '
                       f'touched M {dM[zone].sum():.2e}; E>0 touched M {dM[zone & (E > 0)].sum():.2e}')
            out.append('      M per E/dE bin: ' + '  '.join(f'[{EDGES[k]:g},{EDGES[k + 1]:g}) {hist[k]:.1e}'
                                                        for k in range(len(hist))) + f'  outside {outside:.1e}')
    txt = '\n'.join(out)
    print(txt)
    if a.out:
        open(a.out, 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
