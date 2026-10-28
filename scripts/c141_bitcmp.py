#!/usr/bin/env python3
"""c141 (R-152 items 3(ii), 5, 6): bitwise comparison of two run directories.
  history: every *.hst present in both, the numeric rows compared exactly (comment/header lines skipped: a run with
           extra GetOrAdd keys writes the same columns);
  dumps:   every bin/*.bin present in both, the time, the MeshBlock geometry and every variable's data compared
           exactly (the file header holds the parameter text, which differs when keys differ, so the files
           themselves are not compared with cmp).
Prints one line per file (identical / the first difference) and a summary; exit status 0 iff all identical.
    c141_bitcmp.py DIR_A DIR_B [--maxcyc N]
--maxcyc: compare only history rows up to that many rows (the shorter run's length is used by default)."""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tde_ensemble_plots import bc                 # noqa: E402


def rows(fn):
    return np.array([[float(x) for x in l.split()] for l in open(fn) if l.strip() and not l.startswith('#')])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a')
    ap.add_argument('b')
    a = ap.parse_args()
    bad = 0
    nh = nd = 0
    for fa in sorted(glob.glob(f'{a.a}/*.hst')):
        fb = f'{a.b}/{os.path.basename(fa)}'
        if not os.path.exists(fb):
            continue
        ra, rb = rows(fa), rows(fb)
        n = min(len(ra), len(rb))
        nh += 1
        if ra.shape[1] != rb.shape[1]:
            print(f'HST {os.path.basename(fa)}: column counts differ {ra.shape[1]} vs {rb.shape[1]}')
            bad += 1
            continue
        d = np.nonzero(np.any(ra[:n] != rb[:n], axis=1))[0]
        if len(d):
            bad += 1
            i = d[0]
            c = np.nonzero(ra[i] != rb[i])[0]
            print(f'HST {os.path.basename(fa)}: {len(d)}/{n} rows differ; first row {i} t={ra[i, 0]} cols {list(c[:8])} '
                  f'max|rel| {np.max(np.abs(ra[:n] - rb[:n]) / np.maximum(np.abs(ra[:n]), 1e-300)):.3e}')
        else:
            print(f'HST {os.path.basename(fa)}: {n} rows identical (lengths {len(ra)}, {len(rb)})')
    for fa in sorted(glob.glob(f'{a.a}/bin/*.bin')):
        fb = f'{a.b}/bin/{os.path.basename(fa)}'
        if not os.path.exists(fb):
            continue
        A, B = bc.read_binary(fa), bc.read_binary(fb)
        nd += 1
        msg = []
        if float(A['time']) != float(B['time']):
            msg.append(f"time {A['time']} vs {B['time']}")
        ga, gb = np.asarray(A['mb_geometry']), np.asarray(B['mb_geometry'])
        if ga.shape != gb.shape or np.any(ga != gb):
            msg.append('geometry differs')
        else:
            for v in A['mb_data']:
                if v not in B['mb_data']:
                    msg.append(f'{v} missing in B')
                    continue
                xa = np.asarray(A['mb_data'][v])
                xb = np.asarray(B['mb_data'][v])
                if xa.shape != xb.shape:
                    msg.append(f'{v} shape')
                    continue
                nne = np.count_nonzero((xa != xb) & ~(np.isnan(xa) & np.isnan(xb)))
                if nne:
                    msg.append(f'{v}: {nne} cells differ, max|d| {np.nanmax(np.abs(xa - xb)):.3e}')
        if msg:
            bad += 1
            print(f'BIN {os.path.basename(fa)}: ' + '; '.join(msg[:4]))
    print(f'SUMMARY {a.a} vs {a.b}: {nh} history files, {nd} dumps compared; {bad} differ')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
