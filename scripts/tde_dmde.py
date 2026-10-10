#!/usr/bin/env python3
"""T-8 / M2: debris energy distribution dM/dE from AthenaK restart files (full 3D state).

Restart layout (src/outputs/restart.cpp): header (rst_tree.read_tree) + lloc/cost lists +
data_size (size_t), then per MeshBlock (gid order) a record of data_size bytes:
  MHD conserved  [nmhd = 5 + nscalars][nout3][nout2][nout1]  (densitized by sqrt(gamma))
  B faces        x1f [nout3][nout2][nout1+1], x2f [nout3][nout2+1][nout1], x3f [...]
  ADM            [17][nout3][nout2][nout1]: gxx gxy gxz gyy gyz gzz, Kij (6), psi4,
                 alpha, betax, betay, betaz
nout = nx + 2 ng. Interior cells only.

Per cell: undensitise, recover (rho, P, W) for the ideal gas (Gamma = 5/3) by bisection
in P, then (R-002 conventions)
  E   = -u_t - 1 = alpha W - beta^i u_i - 1,  u_i = S_i/(D h)      (primary)
  E_B = -h u_t - 1                                                   (Bernoulli)
  dM  = sqrt(gamma) D dV.
Cells with r_BH < rcut are excluded (excision region); the excised/accreted mass is
reported separately from the user history. E is normalised by dE = R_*/r_t^2 (geometric,
G = c = M_BH = 1).

    tde_dmde.py RST [RST ...] [--rcut 3] [--rstar 0.677] [--rt 16.4] [--out f.npz]
"""
import argparse
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rst_tree as R   # noqa: E402

GAM = 5.0 / 3.0


def header(fn):
    t = R.read_tree(fn)
    with open(fn, 'rb') as f:
        head = f.read(1 << 20)
        k = head.index(b'<par_end>') + len('<par_end>')
        while head[k:k + 1] in (b'\n', b'\r'):
            k += 1
        off = k + 8 + 72 + 2 * 76 + 16 + 4 + t['nmb'] * 20   # through lloc + cost lists
        f.seek(off)
        dsize, = struct.unpack('Q', f.read(8))
        t['data_off'] = off + 8
    t['dsize'] = dsize
    t['par'] = head[:k].decode(errors='ignore')
    return t


def c2p(D, Ssq, tau):
    """ideal-gas inversion: P with (Gamma-1) rho eps(P) = P; vectorised bisection"""
    lo = np.zeros_like(D)
    hi = np.maximum((GAM - 1.0) * (tau + D) * 2.0, 1e-300)
    for _ in range(80):
        P = 0.5 * (lo + hi)
        Q = tau + D + P
        v2 = np.clip(Ssq / (Q * Q), 0.0, 1.0 - 1e-15)
        W = 1.0 / np.sqrt(1.0 - v2)
        rho = D / W
        eps = (tau + D * (1.0 - W) + P * (1.0 - W * W)) / (D * W)
        f = (GAM - 1.0) * rho * eps - P
        lo = np.where(f > 0, P, lo)
        hi = np.where(f > 0, hi, P)
    P = 0.5 * (lo + hi)
    Q = tau + D + P
    v2 = np.clip(Ssq / (Q * Q), 0.0, 1.0 - 1e-15)
    W = 1.0 / np.sqrt(1.0 - v2)
    rho = D / W
    h = 1.0 + GAM / (GAM - 1.0) * P / np.maximum(rho, 1e-300)
    return rho, P, W, h


def energies(fn, rcut=3.0, nscal=1):
    c = cells(fn, rcut, nscal)
    return c['t'], c['E'], c['EB'], c['dM']


def cells(fn, rcut=3.0, nscal=1):
    """per-cell E, E_B, dM, rho, P outside r_BH < rcut (rcut=0: all cells)"""
    t = header(fn)
    nmb, root = t['nmb'], t['root']
    x1min, x2min, x3min, x1max, x2max, x3max = t['size'][:6]
    nx, ng = 16, 4
    no = nx + 2 * ng
    nmhd = 5 + nscal
    ncc = no**3
    bsz = (no + 1) * no * no * 3
    exp = (nmhd * ncc + bsz + 17 * ncc) * 8
    if exp != t['dsize']:
        raise SystemExit(f'data_size {t["dsize"]} != expected {exp} (nscal={nscal}?)')
    mm = np.memmap(fn, dtype=np.float64, mode='r', offset=t['data_off'],
                   shape=(nmb, t['dsize'] // 8))
    ll, lev = t['ll'], t['ll'][:, 3]
    w = np.array([x1max - x1min, x2max - x2min, x3max - x3min])
    lo = np.array([x1min, x2min, x3min])
    Es, EBs, dMs, rhos, Ps = [], [], [], [], []
    s = slice(ng, ng + nx)
    ic = (np.arange(nx) + 0.5)
    for b in range(nmb):
        rec = mm[b]
        cons = rec[:nmhd * ncc].reshape(nmhd, no, no, no)[:, s, s, s]
        adm = rec[nmhd * ncc + bsz:].reshape(17, no, no, no)[:, s, s, s]
        ext = w / 2.0**lev[b]
        dx = ext / nx
        x = lo[0] + ll[b, 0] * ext[0] + ic * dx[0]
        y = lo[1] + ll[b, 1] * ext[1] + ic * dx[1]
        z = lo[2] + ll[b, 2] * ext[2] + ic * dx[2]
        Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
        keep = np.sqrt(X**2 + Y**2 + Z**2) >= rcut
        if rcut <= 0:
            keep = np.ones_like(X, dtype=bool)
        if not keep.any():
            continue
        g = adm[:6]
        gxx, gxy, gxz, gyy, gyz, gzz = g
        det = (gxx * (gyy * gzz - gyz * gyz) - gxy * (gxy * gzz - gyz * gxz)
               + gxz * (gxy * gyz - gyy * gxz))
        sg = np.sqrt(det)
        ixx = (gyy * gzz - gyz * gyz) / det; ixy = (gxz * gyz - gxy * gzz) / det
        ixz = (gxy * gyz - gxz * gyy) / det; iyy = (gxx * gzz - gxz * gxz) / det
        iyz = (gxy * gxz - gxx * gyz) / det; izz = (gxx * gyy - gxy * gxy) / det
        D = cons[0] / sg
        Sx, Sy, Sz = cons[1] / sg, cons[2] / sg, cons[3] / sg
        tau = cons[4] / sg
        Ssq = (ixx * Sx * Sx + iyy * Sy * Sy + izz * Sz * Sz
               + 2 * (ixy * Sx * Sy + ixz * Sx * Sz + iyz * Sy * Sz))
        ok = keep & (D > 0) & np.isfinite(tau)
        D, Ssq, tau = D[ok], Ssq[ok], np.maximum(tau[ok], 0.0)
        rho, P, W, h = c2p(D, Ssq, tau)
        alpha = adm[13][ok]
        bx, by, bz = adm[14][ok], adm[15][ok], adm[16][ok]
        ut_neg = alpha * W - (bx * Sx[ok] + by * Sy[ok] + bz * Sz[ok]) / (D * h)
        Es.append(ut_neg - 1.0)
        EBs.append(h * ut_neg - 1.0)
        dMs.append(cons[0][ok] * dx.prod())
        rhos.append(rho); Ps.append(P)
    cat = np.concatenate
    return dict(t=t['time'], E=cat(Es), EB=cat(EBs), dM=cat(dMs), rho=cat(rhos), P=cat(Ps),
                par=t['par'])


def hist(E, dM, dE, edges):
    hm, _ = np.histogram(E / dE, bins=edges, weights=dM)
    return hm / dM.sum() / np.diff(edges)          # (1/M) dM/d(E/dE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rst', nargs='+')
    ap.add_argument('--rcut', type=float, default=3.0)
    ap.add_argument('--rstar', type=float, default=0.677)
    ap.add_argument('--rt', type=float, default=16.4)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    dE = a.rstar / a.rt**2
    edges = np.linspace(-4, 4, 81)
    res = {}
    for fn in a.rst:
        tt, E, EB, dM = energies(fn, a.rcut)
        H, HB = hist(E, dM, dE, edges), hist(EB, dM, dE, edges)
        fb, fbB = dM[E < 0].sum() / dM.sum(), dM[EB < 0].sum() / dM.sum()
        core = dM[np.abs(E / dE) < 0.25].sum() / dM.sum()
        print(f'{fn.split("/")[-2]}/{fn.split("/")[-1]}: t={tt:.2f} M_in={dM.sum():.6e} '
              f'bound {fb:.4f} (Bernoulli {fbB:.4f}) |E|<0.25dE {core:.4f}')
        res[fn] = dict(t=tt, H=H, HB=HB, fb=fb, fbB=fbB, M=dM.sum())
    if a.out:
        np.savez(a.out, edges=edges, dE=dE, **{f'{k}_{i}': v for i, (fn, r) in
                 enumerate(res.items()) for k, v in r.items()})


if __name__ == '__main__':
    main()
