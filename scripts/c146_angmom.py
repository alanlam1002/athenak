#!/usr/bin/env python3
"""c146 (R-155 item 4, user-approved 2026-10-29): the gas angular momentum about the hole, offline from full-3D restarts.
  L_BH = sum over the mesh of (x - x_BH(t)) x (sqrt(gamma) S) dV, with sqrt(gamma) S_k the conserved momentum
  (cons IM1..IM3, densitised; covariant: L_z = x S_y - y S_x is the Noether charge of the background's rotational
  Killing vector x d_y - y d_x about the hole), dV the coordinate cell volume, x_BH(t) from the restart's parameter
  copy (tde_dmde.frame_state; the BH frame: the origin). S is the Eulerian momentum, so L_BH is the same in either
  gauge (x' - x'_BH = x - x_BH). Per restart:
    M (sum D dV), L (3 components), L excluding r_BH < rcut (the excision zone), L and M inside r_BH < R for
    R in RS, and sum |(x - x_BH) x S| dV (a scale for the cancellation in L).
    c146_angmom.py OUT.csv LABEL=RUNDIR [...] [--rcut 3]
RUNDIR/rst/*.rst (C collation order; derived <N>x<W>.rst restarts are included with their own time)."""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tde_dmde as T   # noqa: E402

RS = (10.0, 30.0, 100.0)


def one(lab, fn, rcut, nscal=1):
    t = T.header(fn)
    nmb = t['nmb']
    x1min, x2min, x3min, x1max, x2max, x3max = t['size'][:6]
    nx, ng = 16, 4
    no = nx + 2 * ng
    nmhd = 5 + nscal
    ncc = no**3
    mm = np.memmap(fn, dtype=np.float64, mode='r', offset=t['data_off'], shape=(nmb, t['dsize'] // 8))
    ll, lev = t['ll'], t['ll'][:, 3]
    w = np.array([x1max - x1min, x2max - x2min, x3max - x3min])
    lo = np.array([x1min, x2min, x3min])
    xbh, _ = T.frame_state(t['par'], t['time'])
    s = slice(ng, ng + nx)
    ic = np.arange(nx) + 0.5
    acc = dict(M=0.0, L=np.zeros(3), Lx=np.zeros(3), Labs=0.0, Mx=0.0)
    for R in RS:
        acc[f'L{R:g}'] = np.zeros(3)
        acc[f'M{R:g}'] = 0.0
    for b in range(nmb):
        c = np.asarray(mm[b, :4 * ncc]).reshape(4, no, no, no)[:, s, s, s]
        ext = w / 2.0**lev[b]
        dx = ext / nx
        dv = dx.prod()
        x = lo[0] + ll[b, 0] * ext[0] + ic * dx[0] - xbh[0]
        y = lo[1] + ll[b, 1] * ext[1] + ic * dx[1] - xbh[1]
        z = lo[2] + ll[b, 2] * ext[2] + ic * dx[2] - xbh[2]
        Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
        r = np.sqrt(X * X + Y * Y + Z * Z)
        D, Sx, Sy, Sz = c
        Lv = np.stack([Y * Sz - Z * Sy, Z * Sx - X * Sz, X * Sy - Y * Sx]) * dv
        acc['M'] += D.sum() * dv
        acc['L'] += Lv.reshape(3, -1).sum(1)
        acc['Labs'] += np.sqrt((Lv**2).sum(0)).sum()
        k = r >= rcut
        acc['Lx'] += Lv[:, k].sum(1)
        acc['Mx'] += D[k].sum() * dv
        for R in RS:
            k = r < R
            acc[f'L{R:g}'] += Lv[:, k].sum(1)
            acc[f'M{R:g}'] += D[k].sum() * dv
    row = dict(label=lab, file=os.path.basename(fn), t=t['time'], nmb=nmb, xbh=float(np.linalg.norm(xbh)),
               M=acc['M'], Mx=acc['Mx'], Lx=acc['L'][0], Ly=acc['L'][1], Lz=acc['L'][2], Labs=acc['Labs'],
               Lz_exc=acc['Lx'][2])
    for R in RS:
        row[f'M_r{R:g}'] = acc[f'M{R:g}']
        row[f'Lz_r{R:g}'] = acc[f'L{R:g}'][2]
    print(row, flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--rcut', type=float, default=3.0)
    a = ap.parse_args()
    rows = []
    for spec in a.runs:
        lab, d = spec.split('=', 1)
        for fn in sorted(glob.glob(f'{d}/rst/*.rst')):
            rows.append(one(lab, fn, a.rcut))
            with open(a.out, 'w') as fo:
                fo.write(','.join(rows[0].keys()) + '\n')
                for r in rows:
                    fo.write(','.join(f'{v:.12g}' if isinstance(v, float) else str(v) for v in r.values()) + '\n')


if __name__ == '__main__':
    main()
