#!/usr/bin/env python3
"""C-100: first divergence between two runs' <job> checksum_cycles files (utils/chksum.hpp) for one cycle.
Lines are keyed (stage, task, occurrence, array); scratch arrays that a restart leaves zero but that are
written before they are read (u1, the face fluxes, lazily allocated MG aggregation buffers) are skipped.
Prints, per rank, the first differing key in file order, and over all ranks the earliest one.

    c100_chkdiff.py RUN_A RUN_B CYCLE [--nranks 108] [--skip u1,flx1,...]
"""
import argparse, collections
ap = argparse.ArgumentParser()
ap.add_argument('a'); ap.add_argument('b'); ap.add_argument('cycle')
ap.add_argument('--nranks', type=int, default=108)
ap.add_argument('--skip', default='u1,flx1,flx2,flx3,aggrecv,aggsend,cbuf,rbufs')
a = ap.parse_args()
skip = set(a.skip.split(','))


def load(d, r):
    D = collections.OrderedDict()
    for l in open(f'{d}/chksum.{r}.txt'):
        x = l.split()
        if x and x[0] == a.cycle:
            D[(x[1], x[2], x[3], x[4])] = x[5:]
    return D


best = None
for r in range(a.nranks):
    A, B = load(a.a, r), load(a.b, r)
    for i, (k, v) in enumerate(A.items()):
        if k[3] in skip or k[1].startswith('MG_') and not k[3].startswith(('blk', 'x')):
            continue
        if k in B and B[k] != v:
            if best is None or i < best[0]:
                best = (i, r, k, v, B[k])
            break
if best is None:
    print('no differing state array in cycle', a.cycle)
else:
    i, r, k, v, w = best
    print(f'earliest differing state: line {i} rank {r} stage {k[0]} task {k[1]} occ {k[2]} array {k[3]}')
    print(f'   {a.a}: {" ".join(v)[:200]}')
    print(f'   {a.b}: {" ".join(w)[:200]}')
    A = load(a.a, r)
    keys = list(A.keys())
    print('   preceding keys:', [' '.join(x) for x in keys[max(0, i - 6):i]])
