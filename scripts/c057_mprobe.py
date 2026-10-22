#!/usr/bin/env python3
"""C-057 (R-072 item 1): ghost vs owner-interior metric and primitives from the MPROBE lines
(<mhd> dual_energy_probe_*; MetricProbe in primitive_solver_hyd.hpp). Line: MPROBE tag cycle
stage gid i j k ghost x y z psi4 alpha bx by bz gxx gxy gxz gyy gyz gzz Kxx Kxy Kxz Kyy Kyz Kzz
| rho P vx vy vz; tag 0 = before the stage's C2P, 1 = at the start of the stage's fluxes.
Copies are grouped by (tag, cycle, stage, cell centre) and every ghost copy is compared with
the owner's bitwise, per quantity group; also the size of the relative difference.

    c057_mprobe.py OUTLOG|RUNDIR [--nx 16 --ng 4]   (RUNDIR: the per-rank mprobe files)
"""
import argparse
import collections

import numpy as np

GROUPS = (('psi4', slice(0, 1)), ('alpha', slice(1, 2)), ('beta', slice(2, 5)), ('g_dd', slice(5, 11)),
          ('K_dd', slice(11, 17)), ('prims', slice(17, 22)), ('cons', slice(22, 28)),
          ('flag', slice(28, 29)), ('masks', slice(29, 31)))   # masks: excision_floor, excision_flux (R-107 item 2)
TAG = {0: 'C2P', 1: 'flux'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('--nx', type=int, default=16)
    ap.add_argument('--ng', type=int, default=4)
    a = ap.parse_args()
    cells = collections.defaultdict(list)
    bad = 0
    import glob
    import os
    if os.path.isdir(a.log):
        # host-written per-rank files (no interleaving): MPROBE tag cyc stage gid i j k gh
        # x y z psi4 alpha b(3) g(6) K(6) rho P v(3) D S(3) tau DY
        lines = [ln for f in sorted(glob.glob(f'{a.log}/mprobe_rank*.txt')) for ln in open(f)]
        recs = []
        for ln in lines:
            q = ln.split()
            if q[0] != 'MPROBE' or len(q) not in (40, 41, 43):
                bad += 1
                continue
            recs.append(q[1:] + ['nan'] * (43 - len(q) if len(q) < 43 else 0))
    else:
        recs = []
        for ln in open(a.log, errors='replace').read().split('MPROBE ')[1:]:
            ln = ln.split('\n')[0]
            try:
                L, P = ln.split('|')
                q, p = L.split(), P.split()
                assert len(q) == 28 and len(p) == 5
                recs.append(q + p + ['nan'] * 9)
            except (ValueError, AssertionError):
                bad += 1
    for q in recs:
        try:
            tag, cyc, st, gid, i, j, k, gh = (int(v) for v in q[:8])
            x = tuple(round(float(v), 6) for v in q[8:11])
            v = np.array([float(u) for u in q[11:]])
        except ValueError:
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
    # R-107 item 2: per cycle, the first flux call (tag 1, stage 1) -- the ghosts the next cycle's first
    # face fluxes read after the previous cycle's last C2P / F1 / mask rebuild
    per = collections.defaultdict(lambda: [0, collections.Counter()])
    for key, cs in cells.items():
        if key[0] != 1 or key[2] != 1:
            continue
        own = [c for c in cs if c[0] == 0]
        if len(own) != 1:
            continue
        ov = own[0][3]
        for gh, d, gid, v in cs:
            if gh == 0:
                continue
            per[key[1]][0] += 1
            for name, sl in GROUPS:
                if np.any((v[sl] != ov[sl]) & ~(np.isnan(v[sl]) & np.isnan(ov[sl]))):
                    per[key[1]][1][name] += 1
    print('== first flux call of each cycle (tag 1, stage 1): ghost copies, mismatches per group')
    for cyc in sorted(per):
        n, c = per[cyc]
        print(f'  cycle {cyc}: {n} ghost copies; ' + (', '.join(f'{k} {v}' for k, v in c.most_common()) or 'all bitwise equal'))


if __name__ == '__main__':
    main()
