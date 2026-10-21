#!/usr/bin/env python3
"""C-088 (R-092 item 1): offline check of the MG unpack ranges for overlapping writes.

The UnpackMG kernel (src/multigrid/multigrid_bvals.cpp, RecvAndUnpackMG) runs one GPU team
per (MeshBlock m, neighbour slot n, variable v), all concurrently. Each team copies its
payload into u (same-level: recv_mg_indcs_[n][lev].isame; from finer: .ifine) or into
coarse_buf_ (from coarser: .icoar). Two slots of one MeshBlock whose ranges intersect in
the same target array are concurrent writers of those cells.

This script rebuilds the neighbour table of a real restart offline (nghbr_symmetry_offline.py,
rule 'parity' = proj/tde's current SetNeighbors, src/mesh/meshblock.cpp do_register) and
evaluates ComputePerLevelIndices' recv formulas (multigrid_bvals.cpp, compute_recv) for every
MeshBlock, MG block level and populated slot; it lists every intersecting pair.

    c088_mgoverlap.py RESTART.rst [--nx 16] [--ngh 1]
"""
import argparse
import collections
import itertools
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nghbr_symmetry_offline as NS   # noqa: E402

# slot -> (ox1, ox2, ox3, f1, f2), the loops of ComputePerLevelIndices (3D, multilevel)
SLOTS = {}
for n in (-1, 1):
    for fz in (0, 1):
        for fy in (0, 1):
            SLOTS[NS.NeighborIndex(n, 0, 0, fy, fz)] = (n, 0, 0, fy, fz)
for m in (-1, 1):
    for fz in (0, 1):
        for fx in (0, 1):
            SLOTS[NS.NeighborIndex(0, m, 0, fx, fz)] = (0, m, 0, fx, fz)
for m in (-1, 1):
    for n in (-1, 1):
        for fz in (0, 1):
            SLOTS[NS.NeighborIndex(n, m, 0, fz, 0)] = (n, m, 0, fz, 0)
for l in (-1, 1):
    for fy in (0, 1):
        for fx in (0, 1):
            SLOTS[NS.NeighborIndex(0, 0, l, fx, fy)] = (0, 0, l, fx, fy)
for l in (-1, 1):
    for n in (-1, 1):
        for fy in (0, 1):
            SLOTS[NS.NeighborIndex(n, 0, l, fy, 0)] = (n, 0, l, fy, 0)
for l in (-1, 1):
    for m in (-1, 1):
        for fx in (0, 1):
            SLOTS[NS.NeighborIndex(0, m, l, fx, 0)] = (0, m, l, fx, 0)
for l in (-1, 1):
    for m in (-1, 1):
        for n in (-1, 1):
            SLOTS[NS.NeighborIndex(n, m, l, 0, 0)] = (n, m, l, 0, 0)


def recv_boxes(ox, f1, f2, ncells, ngh):
    """compute_recv: (isame or None, icoar, ifine) as ((is,ie),(js,je),(ks,ke))"""
    ox1, ox2, ox3 = ox
    s, e = ngh, ngh + ncells - 1
    cnx = ncells // 2
    cs, ce = ngh, ngh + cnx - 1

    def same(o):
        return (s, e) if o == 0 else ((e + 1, e + ngh) if o > 0 else (s - ngh, s - 1))
    isame = (same(ox1), same(ox2), same(ox3)) if (f1 == 0 and f2 == 0) else None

    def coar(o, f):
        if o == 0:
            return (cs, ce + ngh) if f == 0 else (cs - ngh, ce)
        return (ce + 1, ce + ngh) if o > 0 else (cs - ngh, cs - 1)
    fj = f1 if ox1 != 0 else f2
    fk = f1 if (ox1 != 0 and ox2 != 0) else f2
    icoar = (coar(ox1, f1), coar(ox2, fj), coar(ox3, fk))

    def fine(o, f):
        if o == 0:
            return (s + cnx, e) if f == 1 else (s, e - cnx)
        return (e + 1, e + ngh) if o > 0 else (s - ngh, s - 1)
    ifine = (fine(ox1, f1), fine(ox2, fj), fine(ox3, fk))
    return isame, icoar, ifine


def inter(a, b):
    r = []
    for (al, ah), (bl, bh) in zip(a, b):
        lo, hi = max(al, bl), min(ah, bh)
        if lo > hi:
            return 0, None
        r.append((lo, hi))
    n = 1
    for lo, hi in r:
        n *= hi - lo + 1
    return n, tuple(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('restart')
    ap.add_argument('--nx', type=int, default=16)
    ap.add_argument('--ngh', type=int, default=1)
    a = ap.parse_args()
    d = NS.read_restart_topology(a.restart)
    bcs = {}
    for key in ('ix1_bc', 'ox1_bc', 'ix2_bc', 'ox2_bc', 'ix3_bc', 'ox3_bc'):
        mm = re.search(r'^\s*%s\s*=\s*(\S+)' % key, d['pardump'], re.M)
        bcs[key] = mm.group(1) if mm else 'outflow'
    mi, mbi = d['mesh_indcs'], d['mb_indcs']
    nmb_rootx = (mi[1] // mbi[1], mi[2] // mbi[2], mi[3] // mbi[3])
    tree = NS.Tree(d['llocs'], d['root_level'], nmb_rootx, bcs)
    ng = NS.set_neighbors(tree, d['llocs'], 'parity')
    print(f'{a.restart}: {len(d["llocs"])} MeshBlocks, cycle {d["ncycle"]}, MG nx {a.nx}, ngh {a.ngh}')
    nlevel = a.nx.bit_length()            # 16 -> 5 levels: 16 8 4 2 1
    stats = collections.Counter()
    cells = collections.Counter()
    examples = {}
    for b, row in enumerate(ng):
        mlev = d['llocs'][b][3]
        for lev in range(nlevel):
            ncells = a.nx >> (nlevel - 1 - lev)
            skip_fc = ncells < 2
            writes = []   # (target, box, slot, gid, kind)
            for n, ent in enumerate(row):
                if ent is None or n not in SLOTS:
                    continue
                gid, nlev, dest = ent
                if gid < 0:
                    continue
                ox1, ox2, ox3, f1, f2 = SLOTS[n]
                isame, icoar, ifine = recv_boxes((ox1, ox2, ox3), f1, f2, ncells, a.ngh)
                if nlev == mlev:
                    if isame is None:
                        stats['same-level in f!=0 slot (isame unset)'] += 1
                        continue
                    writes.append(('u', isame, n, gid, 'same'))
                elif skip_fc:
                    continue
                elif nlev < mlev:
                    writes.append(('cbuf', icoar, n, gid, 'coarser'))
                else:
                    writes.append(('u', ifine, n, gid, 'finer'))
            for w1, w2 in itertools.combinations(writes, 2):
                if w1[0] != w2[0]:
                    continue
                nc, box = inter(w1[1], w2[1])
                if nc == 0:
                    continue
                key = (w1[0], f'L{lev}(n={ncells})', '-'.join(sorted((w1[4], w2[4]))),
                       'same sender' if w1[3] == w2[3] else 'different senders')
                stats[key] += 1
                cells[key] += nc
                examples.setdefault(key, (b, d['llocs'][b], w1[2], SLOTS[w1[2]], w1[3],
                                          w2[2], SLOTS[w2[2]], w2[3], box))
    if not cells:
        print('NO overlapping unpack ranges')
    for key in sorted(cells):
        print(f'  {key}: pairs {stats[key]}, overlapping cells (per variable) {cells[key]}')
        e = examples[key]
        print(f'      e.g. gid {e[0]} lloc {e[1]}: slot {e[2]} {e[3]} from gid {e[4]}  vs  '
              f'slot {e[5]} {e[6]} from gid {e[7]}  overlap {e[8]}')
    for k, v in stats.items():
        if isinstance(k, str):
            print(f'  NOTE {k}: {v}')


if __name__ == '__main__':
    main()
