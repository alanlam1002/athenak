#!/usr/bin/env python3
"""R-127 Q3 check (c117): the plane estimate of |(alpha v - beta).n|_coord against the full 3D periapsis
dump (prim_3d_peri, t = 135.9). The 3D dump has no metric or flags, so on BOTH sides the flag is the
vcurv sensor recomputed from the primitives (div v <= 0, psi_v > 0.05 c_s, one 0.5x dilation pass)
and alpha, beta, psi4 come from the nearest adm_xy slice (t = 136), taken z-independent at each
(x, y) (the finest slice cell containing it). 3D: n = -grad P (3D), all axes; plane: the z = 0
cells of the same 3D dump with the in-plane n and in-plane sensor (as c117_shock_speed for CN/BN).
    c117_check3d.py LABEL=RUNDIR [...]
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tde_ensemble_plots import bc                 # noqa: E402

B = 'cfc_tde_wd_imbh_parabolic_tracker'
GAM, RSW, VCSH = 5.0 / 3.0, 3.2e-8, 0.05


def wq(x, w, qs):
    o = np.argsort(x); c = np.cumsum(w[o]); c /= c[-1]
    return [x[o][np.searchsorted(c, q)] for q in qs]


def lookup(A):
    g = np.asarray(A['mb_geometry']); nx, ny = A['nx1_out_mb'], A['nx2_out_mb']
    dx = (g[:, 1] - g[:, 0]) / nx
    o = np.argsort(-dx)                              # coarse first, finest last wins
    f = {k: np.asarray(A['mb_data'][k])[:, 0] for k in ('adm_alpha', 'adm_betax', 'adm_betay', 'adm_psi4')}
    return g, nx, ny, dx, o, f


def sample(L, X, Y):
    g, nx, ny, dx, o, f = L
    out = {k: np.full(X.shape, np.nan) for k in f}
    for b in o:
        m = (X >= g[b, 0]) & (X < g[b, 1]) & (Y >= g[b, 2]) & (Y < g[b, 3])
        if not m.any():
            continue
        i = np.clip(((X[m] - g[b, 0]) / dx[b]).astype(int), 0, nx - 1)
        j = np.clip(((Y[m] - g[b, 2]) / ((g[b, 3] - g[b, 2]) / ny)).astype(int), 0, ny - 1)
        for k in f:
            out[k][m] = f[k][b, j, i]
    return out


def main():
    for spec in sys.argv[1:]:
        lab, d = spec.split('=', 1)
        fp = sorted(glob.glob(f'{d}/bin/{B}.prim_3d_peri.*.bin'))
        fp = [f for f in fp if abs(bc.read_binary(f)['time'] - 135.9) < 0.5][0]
        P = bc.read_binary(fp)
        A = bc.read_binary(sorted(glob.glob(f'{d}/bin/{B}.adm_xy.*.bin'),
                                  key=lambda f: abs(bc.read_binary(f)['time'] - 136.0))[0])
        L = lookup(A)
        g = np.asarray(P['mb_geometry']); n = P['nx1_out_mb']
        q = {k: np.asarray(P['mb_data'][k]) for k in ('dens', 'velx', 'vely', 'velz', 'press')}
        res = {'3d': ([], []), 'plane': ([], [])}
        for b in range(len(g)):
            h = (g[b, 1] - g[b, 0]) / n
            c = lambda lo: lo + (np.arange(n) + 0.5) * h
            Z, Y, X = np.meshgrid(c(g[b, 4]), c(g[b, 2]), c(g[b, 0]), indexing='ij')
            rho, p = q['dens'][b], q['press'][b]
            if (rho * 1.0 < RSW).all():
                continue
            S = sample(L, X, Y)
            p4 = S['adm_psi4']
            u = [q['velx'][b], q['vely'][b], q['velz'][b]]
            W = np.sqrt(1 + p4 * sum(x * x for x in u))
            v = [x / W for x in u]
            hh = 1 + GAM / (GAM - 1) * p / rho
            cs = np.sqrt(GAM * p / (rho * hh))
            D = rho * W * p4**1.5
            for kind, axes in (('3d', (0, 1, 2)), ('plane', (1, 2))):   # array axes: 0 z, 1 y, 2 x
                psiv = np.zeros_like(rho); div = np.zeros_like(rho); gr = [np.zeros_like(rho) for _ in range(3)]
                inn = np.ones(rho.shape, bool)
                for ax in axes:
                    va = v[2 - ax]
                    sl = [slice(None)] * 3
                    zr = slice(1, -1) if kind == '3d' else slice(None)

                    def sh(o):
                        s = [zr, slice(1, -1), slice(1, -1)]; s[ax] = slice(1 + o, rho.shape[ax] - 1 + o); return tuple(s)
                    core = (zr, slice(1, -1), slice(1, -1))
                    psiv[core] = np.maximum(psiv[core], np.abs(va[sh(1)] - 2 * va[core] + va[sh(-1)]))
                    div[core] += va[sh(1)] - va[sh(-1)]
                    gr[2 - ax][core] = (p[sh(1)] - p[sh(-1)]) / (2 * h)
                inn[:, 0] = inn[:, -1] = False; inn[:, :, 0] = inn[:, :, -1] = False
                if kind == '3d':
                    inn[0] = inn[-1] = False
                s1 = (div <= 0) & (psiv > VCSH * cs); wk = (div <= 0) & (psiv > 0.5 * VCSH * cs)
                nb = np.zeros_like(s1)
                for ax in axes:
                    nb |= np.roll(s1, 1, ax) | np.roll(s1, -1, ax)
                fl = (s1 | (wk & nb)) & inn & (D > RSW) & np.isfinite(S['adm_alpha'])
                if kind == 'plane':
                    zc = Z[:, 0, 0]
                    k0 = np.nonzero((zc - 0.5 * h <= 0) & (zc + 0.5 * h > 0))[0]
                    if len(k0) == 0:
                        continue
                    km = np.zeros(rho.shape[0], bool); km[k0] = True
                    fl &= km[:, None, None]
                    gr[2] = np.zeros_like(rho)
                gn = np.sqrt(gr[0]**2 + gr[1]**2 + gr[2]**2)
                fl &= gn > 0
                if not fl.any():
                    continue
                al, bx, by = S['adm_alpha'][fl], S['adm_betax'][fl], S['adm_betay'][fl]
                Vn = np.abs(((al * v[0][fl] - bx) * -gr[0][fl] + (al * v[1][fl] - by) * -gr[1][fl]
                             + al * v[2][fl] * -gr[2][fl]) / gn[fl])
                res[kind][0].append(Vn); res[kind][1].append((D * h**3)[fl])
        for kind in ('3d', 'plane'):
            Vn, w = np.concatenate(res[kind][0]), np.concatenate(res[kind][1])
            print(lab, kind, 'n=%d' % len(Vn), 'Vn p10/p50/p90 = %.4f %.4f %.4f' % tuple(wq(Vn, w, (0.1, 0.5, 0.9))),
                  'mean %.4f' % ((Vn * w).sum() / w.sum()))


if __name__ == '__main__':
    main()
