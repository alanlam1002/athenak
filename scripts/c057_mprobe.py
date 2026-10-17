#!/usr/bin/env python3
"""C-057 (R-072 item 1): ghost vs owner-interior metric and primitives from the MPROBE lines
(<mhd> dual_energy_probe_*; MetricProbe in primitive_solver_hyd.hpp). Line: MPROBE tag cycle
stage gid i j k ghost x y z psi4 alpha bx by bz gxx gxy gxz gyy gyz gzz Kxx Kxy Kxz Kyy Kyz Kzz
| rho P vx vy vz; tag 0 = before the stage's C2P, 1 = at the start of the stage's fluxes.
Copies are grouped by (tag, cycle, stage, cell centre) and every ghost copy is compared with
the owner's bitwise, per quantity group; also the size of the relative difference.

    c057_mprobe.py OUTLOG [--nx 16 --ng 4]
"""
import argparse
import collections

import numpy as np

GROUPS = (('psi4', slice(0, 1)), ('alpha', slice(1, 2)), ('beta', slice(2, 5)), ('g_dd', slice(5, 11)),
          ('K_dd', slice(11, 17)), ('prims', slice(17, 22)))
TAG = {0: 'C2P', 1: 'flux'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('--nx', type=int, default=16)
    ap.add_argument('--ng', type=int, default=4)
    a = ap.parse_args()
    cells = collections.defaultdict(list)
    bad = 0
    for ln in open(a.log, errors='replace').read().split('MPROBE ')[1:]:
        ln = ln.split('\n')[0]
        try:
            L, P = ln.split('|')
            q, p = L.split(), P.split()
            assert len(q) == 27 and len(p) == 5
            tag, cyc, st, gid, i, j, k, gh = (int(v) for v in q[:8])
            x = tuple(round(float(v), 6) for v in q[8:11])
            v = np.array([float(u) for u in q[11:]] + [float(u) for u in p])
        except (ValueError, AssertionError):
            bad += 1
            continue
        lo, hi = a.ng, a.ng + a.nx - 1
        depth = max(max(lo - c, c - hi, 0) for c in (i, j, k))
        cells[(tag, cyc, st) + x].append((gh, depth, gid, v))
    print(f'malformed (interleaved) records skipped: {bad}')
    for tag in (0, 1):
        n = 0
        mis = collections.Counter()
        rel = collections.defaultdict(list)
        bydepth = collections.defaultdict(collections.Counter)
        nd = collections.Counter()
        for key, cs in cells.items():
            if key[0] != tag:
                continue
            own = [c for c in cs if c[0] == 0]
            if len(own) != 1:
                continue
            ov = own[0][3]
            for gh, d, gid, v in cs:
                if gh == 0:
                    continue
                n += 1
                nd[d] += 1
                for name, sl in GROUPS:
                    if np.any(v[sl] != ov[sl]):
                        mis[name] += 1
                        bydepth[d][name] += 1
                        r = np.abs(v[sl] - ov[sl]) / np.maximum(np.abs(ov[sl]), 1e-300)
                        rel[name].append(float(np.max(r)))
        print(f'== at the {TAG[tag]} call: {n} ghost copies compared')
        for name, _ in GROUPS:
            r = np.array(rel[name]) if rel[name] else np.zeros(1)
            print(f'  {name:6s}: differ in {mis[name]:7d} ({mis[name] / max(n, 1):.4f}); max rel diff median '
                  f'{np.median(r):.2e} p90 {np.quantile(r, 0.9):.2e} max {r.max():.2e}')
        for d in sorted(bydepth):
            print(f'    g{d} ({nd[d]} copies): ' + ', '.join(f'{k} {v}' for k, v in bydepth[d].most_common()))


if __name__ == '__main__':
    main()
