#!/usr/bin/env python3
"""C-055 (R-069 item 1): ghost vs owner-interior dual-energy state at the dominant C-054 faces,
from the DEPROBE lines (<mhd> dual_energy_probe_*). Every cell copy in the probe boxes prints,
per RK-stage C2P: cycle stage gid i j k ghost x y z shk0 flag dbr D tau DY | previous rho P vx
vy vz | new rho P vx. Copies are grouped by (cycle, stage, cell centre); each ghost copy is
compared with the owner's (ghost = 0) copy:
  cons   (D, tau, DY): must match exactly (the exchanged data);
  prev   (previous-stage primitives): R-068 path two (stored ghost prims);
  shk0   (pre-dilation sensor): path one's origin (clamped neighbours);
  flag   (after dilation), dbr (branch), new prims.
A mismatch is attributed to the earliest differing quantity. Ghost depth = distance of the
copy from its block's interior, in cells (g1 = first ghost layer).

    c055_probe.py OUTLOG [--nx 16 --ng 4]
"""
import argparse
import collections

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('--nx', type=int, default=16)
    ap.add_argument('--ng', type=int, default=4)
    a = ap.parse_args()
    cells = collections.defaultdict(list)
    bad = 0
    text = open(a.log, errors='replace').read()
    for ln in text.split('DEPROBE ')[1:]:
        ln = ln.split('\n')[0]          # device printf from several ranks may interleave
        try:
            L, P, N = ln.split('|')
            q = L.split()
            assert len(q) == 16 and len(P.split()) == 5 and len(N.split()) == 3
            cyc, st, gid, i, j, k, gh = (int(v) for v in q[:7])
            x, y, z = (float(v) for v in q[7:10])
            shk0, flag, dbr = (int(v) for v in q[10:13])
            cons = tuple(float(v) for v in q[13:16])
            prev = tuple(float(v) for v in P.split())
            new = tuple(float(v) for v in N.split())
        except (ValueError, AssertionError):
            bad += 1
            continue
        lo, hi = a.ng, a.ng + a.nx - 1
        depth = max(max(lo - c, c - hi, 0) for c in (i, j, k))
        key = (cyc, st, round(x, 6), round(y, 6), round(z, 6))
        cells[key].append(dict(gid=gid, gh=gh, depth=depth, shk0=shk0, flag=flag, dbr=dbr,
                               cons=cons, prev=prev, new=new))
    n_cmp = 0
    first = collections.Counter()
    by_depth = collections.defaultdict(collections.Counter)
    by_stage = collections.defaultdict(collections.Counter)
    newdiff = collections.Counter()
    ex = collections.defaultdict(list)
    no_owner = 0
    for key, cs in cells.items():
        own = [c for c in cs if c['gh'] == 0]
        if len(own) != 1:
            no_owner += 1
            continue
        o = own[0]
        for c in cs:
            if c['gh'] == 0:
                continue
            n_cmp += 1
            if c['cons'] != o['cons']:
                f = 'cons'
            elif c['prev'] != o['prev']:
                f = 'prev'
            elif c['shk0'] != o['shk0']:
                f = 'shk0'
            elif c['flag'] != o['flag']:
                f = 'flag'
            elif c['dbr'] != o['dbr']:
                f = 'dbr'
            elif c['new'] != o['new']:
                f = 'new-only'
            else:
                f = 'identical'
            first[f] += 1
            by_depth[c['depth']][f] += 1
            by_stage[key[1]][f] += 1
            if c['new'] != o['new']:
                newdiff[f] += 1
            if f not in ('identical',) and len(ex[f]) < 4:
                ex[f].append((key, c['depth'], {k: (o[k], c[k]) for k in ('shk0', 'flag', 'dbr')},
                              np.max(np.abs(np.array(c['prev']) - np.array(o['prev'])) /
                                     np.maximum(np.abs(np.array(o['prev'])), 1e-300)),
                              np.max(np.abs(np.array(c['new']) - np.array(o['new'])) /
                                     np.maximum(np.abs(np.array(o['new'])), 1e-300))))
    print(f'malformed (interleaved) records skipped: {bad}')
    print(f'cell copies compared (ghost vs owner): {n_cmp}; positions without a unique owner: {no_owner}')
    print('earliest differing quantity: ' + ', '.join(f'{k} {v}' for k, v in first.most_common()))
    print('of which the new primitives differ: ' + ', '.join(f'{k} {v}' for k, v in newdiff.most_common()))
    print('by ghost depth (g1 = first ghost layer):')
    for d in sorted(by_depth):
        print(f'  g{d}: ' + ', '.join(f'{k} {v}' for k, v in by_depth[d].most_common()))
    print('by stage:')
    for s in sorted(by_stage):
        print(f'  stage {s}: ' + ', '.join(f'{k} {v}' for k, v in by_stage[s].most_common()))
    print('examples (key, depth, (owner, ghost) shk0/flag/dbr, max rel diff prev, new):')
    for f, es in ex.items():
        for e in es:
            print(f'  [{f}] {e[0]} g{e[1]} {e[2]} prev {e[3]:.2e} new {e[4]:.2e}')


if __name__ == '__main__':
    main()
