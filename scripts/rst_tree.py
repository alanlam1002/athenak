#!/usr/bin/env python3
"""Offline mesh-tree replay from an AthenaK restart file (layout: src/outputs/restart.cpp).

Header: parameter text up to and including '<par_end>\\n', nmb_total, root_level (int32),
RegionSize (9 doubles), 2 x RegionIndcs (19 int32 each), time, dt (double), ncycle (int32),
then nmb_total x LogicalLocation (lx1, lx2, lx3, level as int32) and nmb_total costs (float).
Ranks are assigned as Mesh::LoadBalance (src/mesh/load_balance.cpp) does: contiguous gid
ranges filled from the last rank backwards with target cost = remaining / remaining ranks.

For each requested rank: its MeshBlocks' levels, centres, distance from the origin (BH),
and how many of their 26 neighbours are at a different level (coarse-fine faces).

    rst_tree.py FILE.rst --nranks 192 --ranks 83,183
"""
import argparse
import struct

import numpy as np


def read_tree(fn):
    with open(fn, 'rb') as f:
        head = f.read(1 << 20)
        k = head.index(b'<par_end>') + len('<par_end>')
        while head[k:k + 1] in (b'\n', b'\r'):
            k += 1
        f.seek(k)
        nmb, root = struct.unpack('2i', f.read(8))
        size = struct.unpack('9d', f.read(72))
        f.read(2 * 19 * 4)
        time, dt = struct.unpack('2d', f.read(16))
        ncycle, = struct.unpack('i', f.read(4))
        ll = np.frombuffer(f.read(nmb * 16), dtype=np.int32).reshape(nmb, 4).copy()
        cost = np.frombuffer(f.read(nmb * 4), dtype=np.float32).copy()
    return dict(nmb=nmb, root=root, size=size, time=time, dt=dt, ncycle=ncycle, ll=ll, cost=cost)


def load_balance(cost, nranks):
    nb = len(cost)
    r = np.zeros(nb, int)
    total = float(cost.sum()); j = nranks - 1; target = total / nranks; my = 0.0
    for i in range(nb - 1, -1, -1):
        my += cost[i]; r[i] = j
        if my >= target and j > 0:
            j -= 1; total -= my; my = 0.0; target = total / (j + 1)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rst')
    ap.add_argument('--nranks', type=int, default=192)
    ap.add_argument('--ranks', default='83,183')
    a = ap.parse_args()
    t = read_tree(a.rst)
    ll, root = t['ll'], t['root']
    x1min, x2min, x3min, x1max, x2max, x3max = t['size'][:6]
    rank = load_balance(t['cost'], a.nranks)
    lev = ll[:, 3]
    print(f"{a.rst.split('/')[-1]}: t={t['time']:.3f} cycle={t['ncycle']} nmb={t['nmb']} "
          f"root_level={root} levels {lev.min() - root}-{lev.max() - root} (physical)")
    print('blocks per level:', {int(l - root): int((lev == l).sum()) for l in np.unique(lev)})
    # block centres
    nroot = 2**root
    w = np.array([x1max - x1min, x2max - x2min, x3max - x3min])
    lo = np.array([x1min, x2min, x3min])
    ext = w[None, :] / (2.0**lev)[:, None]          # block width at its level (root mesh = 2^root blocks)
    ctr = lo[None, :] + (ll[:, :3] + 0.5) * ext
    rbh = np.linalg.norm(ctr, axis=1)
    # neighbour levels: map each fine-unit cell index to its block level (unit = finest level)
    lmax = lev.max()
    key = {}
    for b in range(t['nmb']):
        key[(int(lev[b]), int(ll[b, 0]), int(ll[b, 1]), int(ll[b, 2]))] = b

    def find(level, i, j, k):
        """block containing the level-`level` cell index (i, j, k), searching up and down"""
        for L in range(level, lev.min() - 1, -1):
            s = level - L
            b = key.get((L, i >> s, j >> s, k >> s))
            if b is not None:
                return b
        return None

    def cf_faces(b):
        L = int(lev[b]); n = 0; finer = 0
        for d in [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]:
            i, j, k = (int(ll[b, 0]) + d[0], int(ll[b, 1]) + d[1], int(ll[b, 2]) + d[2])
            if min(i, j, k) < 0 or max(i, j, k) >= nroot * 2**(L - root):
                continue
            nb = find(L, i, j, k)
            if nb is None:
                finer += 1            # the neighbour at this level is refined
            elif lev[nb] != L:
                n += 1                # coarser neighbour
        return n, finer

    for rk in [int(x) for x in a.ranks.split(',')]:
        bs = np.where(rank == rk)[0]
        if len(bs) == 0:
            print(f'rank {rk}: no blocks'); continue
        print(f"rank {rk}: gids {bs[0]}-{bs[-1]} ({len(bs)} blocks), levels "
              f"{sorted(set(int(x - root) for x in lev[bs]))}, r_BH {rbh[bs].min():.2f}-{rbh[bs].max():.2f}")
        for b in bs:
            c, f = cf_faces(b)
            print(f"   gid {b:5d} L{lev[b] - root} centre ({ctr[b, 0]:7.2f},{ctr[b, 1]:7.2f},{ctr[b, 2]:7.2f}) "
                  f"r_BH {rbh[b]:6.2f} coarser-nbrs {c:2d} finer-nbrs {f:2d}")
    cnt = np.bincount(rank, minlength=a.nranks)
    print(f'blocks per rank: min {cnt.min()} max {cnt.max()} (ranks 83/183: {cnt[83]}/{cnt[183] if a.nranks > 183 else "-"})')


if __name__ == '__main__':
    main()
