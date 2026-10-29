#!/usr/bin/env python3
"""c147 (R-155 item 1: "the ghost/owner probe"): <dyn_grmhd> killing_probe_cycles check of the design rule on the
Killing-energy face flux. Each block writes T's face flux on its own faces, its block-boundary faces included, so a
face on a same-level block boundary is written by both blocks. Every such pair must agree bitwise in F_D, F_S and F_T
(F_T = alpha0_f (F_tau + F_D) - beta0_f.F_S - w_f F_D, the weights pointwise in the face coordinates). Reads
kprobe_rank*.txt lines
  KPROBE cycle stage dir gid i j k level x y z F_D F_S1 F_S2 F_S3 F_T   (%.17g)
groups by (cycle, stage, dir, level, x, y, z) as printed and counts groups with >1 copy and those whose values differ.
    c147_kprobe_check.py RUNDIR [RUNDIR ...]"""
import glob
import sys
from collections import defaultdict

for d in sys.argv[1:]:
    G = defaultdict(list)
    nrec = 0
    for f in sorted(glob.glob(f'{d}/kprobe_rank*.txt')):
        for ln in open(f):
            t = ln.split()
            if not t or t[0] != 'KPROBE':
                continue
            nrec += 1
            G[(t[1], t[2], t[3], t[8], t[9], t[10], t[11])].append((tuple(t[12:17]), t[4]))
    multi = [k for k in G if len(G[k]) > 1]
    bad = [k for k in multi if len({x[0] for x in G[k]}) > 1]
    badT = [k for k in multi if len({x[0][4] for x in G[k]}) > 1]
    print(f'{d}: records={nrec} faces={len(G)} shared={len(multi)} '
          f'copies={sum(len(G[k]) for k in multi)} mismatched={len(bad)} (in F_T: {len(badT)})')
    for k in bad[:3]:
        print('  MISMATCH', k, G[k])
