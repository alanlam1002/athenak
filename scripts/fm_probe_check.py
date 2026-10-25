#!/usr/bin/env python3
"""<dyn_grmhd> face_probe check (R-140 item 1, the design rule on the face metric arrays):
every same-level copy of one physical face (the owner's face and the neighbours' ghost-region
copies) must carry bitwise-identical alpha, beta^i, psi. Reads fprobe_rank*.txt lines
  FPROBE cycle stage dir gid i j k level x y z alpha bx by bz psi   (%.17g)
groups by (cycle, stage, dir, level, x, y, z) as printed (exact decimal strings) and counts
groups with >1 copy and those whose value strings differ, separately for faces inside the
physical domain (|x_i| <= L, the face itself may lie on the boundary) and outside it (the
physical-boundary ghost region, not read by any interior update).
    fm_probe_check.py [--L 64] RUNDIR [RUNDIR ...]"""
import glob
import sys
from collections import defaultdict

args = sys.argv[1:]
L = 64.0
if args and args[0] == '--L':
    L = float(args[1]); args = args[2:]
for d in args:
    G = defaultdict(list)
    nrec = 0
    for f in sorted(glob.glob(f'{d}/fprobe_rank*.txt')):
        for ln in open(f):
            t = ln.split()
            if not t or t[0] != 'FPROBE':
                continue
            nrec += 1
            G[(t[1], t[2], t[3], t[8], t[9], t[10], t[11])].append((tuple(t[12:17]), t[4]))
    for name, inside in (('inside', True), ('outside', False)):
        keys = [k for k in G if (max(abs(float(x)) for x in k[4:7]) <= L) == inside]
        multi = [k for k in keys if len(G[k]) > 1]
        bad = [k for k in multi if len({x[0] for x in G[k]}) > 1]
        print(f'{d} [{name} |x|<={L:g}]: faces={len(keys)} shared={len(multi)} '
              f'copies={sum(len(G[k]) for k in multi)} mismatched={len(bad)}')
        for k in bad[:3]:
            print('  MISMATCH', k, G[k])
    print(f'{d}: records={nrec}')
