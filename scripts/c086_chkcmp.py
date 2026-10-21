#!/usr/bin/env python3
"""C-086 (R-087 item 10): compare two runs' <debug> checksum_cycles files.

Each run dir holds chksum.<rank>.txt (src/utils/chksum.hpp, written by
tasklist/numerical_relativity.cpp ChecksumAfterTask): lines
  cycle stage task occurrence array hexsum            (array rows)
  cycle stage task occurrence host  hex x 8           (dt, time, frame_vel_u[3], bc_center[3])
A row's key is (rank, cycle, stage, task, occurrence, array). Tasks are walked in rank 0's
order of run A; for each, report which arrays differ on any rank. The first such task is
the first task after which the two states are not bitwise equal.

    c086_chkcmp.py RUN_A RUN_B [--arrays u0,w0,...] [--all]
"""
import argparse
import glob
import os
import re


def load(d):
    rows, order = {}, []
    for f in glob.glob(os.path.join(d, 'chksum.*.txt')):
        r = int(re.search(r'chksum\.(\d+)\.txt', f).group(1))
        for ln in open(f):
            v = ln.split()
            if len(v) < 6:
                continue
            key = (int(v[0]), int(v[1]), v[2], int(v[3]))
            rows[(r,) + key + (v[4],)] = tuple(v[5:])
            if r == 0 and (not order or order[-1] != key):
                order.append(key)
    return rows, order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--all', action='store_true', help='list every differing task, not just the first 12')
    args = ap.parse_args()
    A, order = load(args.a)
    B, _ = load(args.b)
    ranks = sorted({k[0] for k in A})
    arrays = sorted({k[-1] for k in A})
    print(f'{args.a} vs {args.b}: {len(ranks)} ranks, {len(order)} task rows on rank 0, arrays {arrays}')
    missing = len(set(A) ^ set(B))
    print(f'keys only in one run: {missing}')
    shown = 0
    first = None
    for key in order:
        diff = {}
        for arr in arrays:
            nr = [r for r in ranks if (r,) + key + (arr,) in A and (r,) + key + (arr,) in B
                  and A[(r,) + key + (arr,)] != B[(r,) + key + (arr,)]]
            if nr:
                diff[arr] = nr
        if diff:
            if first is None:
                first = key
            if shown < 12 or args.all:
                print('  cycle %d stage %d %-22s #%d differs: ' % key +
                      ', '.join(f'{a}({len(r)} ranks, first {r[0]})' for a, r in diff.items()))
                shown += 1
    if first is None:
        print('ALL IDENTICAL over the common task rows')
    else:
        i = order.index(first)
        print('first differing task:', first, '| preceding task (identical):',
              order[i - 1] if i > 0 else None)


if __name__ == '__main__':
    main()
