#!/usr/bin/env python3
"""Live MPI requests per rank (reqtrack.<rank>.txt from libreqtrack.so) created at
PackAndSendMG's MPI_Isend (located per binary with addr2line), against the regrid count
from ensemble.out, for one or more runs (DEVELOPMENT.md item 76).

Regrid times: each 'ReinitializeMetricForAMR: post-regrid' line is stamped with the
elapsed= of the last preceding cycle line (reqtrack times count from the first MPI call,
a few seconds after start: close enough to line the two up).

    reqtrack_plot.py LABEL=RUNDIR [...] [--out f.png]
"""
import argparse
import glob
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402


def tracker(d, site_addr):
    runs = []
    for f in glob.glob(f'{d}/reqtrack.*.txt'):
        t, live, site = [], [], []
        for line in open(f):
            m = re.match(r'([\d.]+) live=(\d+)', line)
            if not m:
                continue
            t.append(float(m.group(1))); live.append(int(m.group(2)))
            ms = re.search(site_addr + r':(\d+)', line)    # PackAndSendMG's MPI_Isend
            site.append(int(ms.group(1)) if ms else 0)
        runs.append((np.array(t), np.array(live), np.array(site)))
    return runs


def pack_site(d):
    """the creating address of PackAndSendMG's MPI_Isend in this run's binary (addr2line)"""
    import subprocess
    sites = set()
    for f in glob.glob(f'{d}/reqtrack.*.txt')[:4]:
        for line in open(f):
            sites |= set(re.findall(r'(0x[0-9a-f]+):', line))
    for a in sorted(sites):
        fn = subprocess.run(['addr2line', '-f', '-C', '-e', f'{d}/athena', a],
                            capture_output=True, text=True).stdout.split('\n')[0]
        if 'PackAndSendMG' in fn:
            return a
    return '0xNONE'


def regrids(d):
    el, out = 0.0, []
    for line in open(f'{d}/ensemble.out'):
        m = re.search(r'elapsed=([-+.\deE]+)', line)
        if m:
            el = float(m.group(1))
        elif 'ReinitializeMetricForAMR: post-regrid' in line:
            out.append(el)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--out', default='reqtrack.png')
    a = ap.parse_args()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for spec in a.runs:
        lab, d = spec.split('=', 1)
        tr, rg = tracker(d, pack_site(d)), regrids(d)
        tmax = max(t[-1] for t, _, _ in tr if len(t))
        grid = np.linspace(0, tmax, 200)
        mx = np.max([np.interp(grid, t, s, right=np.nan) for t, _, s in tr if len(t)], axis=0)
        nreg = np.searchsorted(rg, grid)
        l, = ax[0].plot(grid, mx, label=f'{lab} (max over ranks)')
        ax[1].plot(nreg, mx, '.', color=l.get_color(), ms=3, label=lab)
        print(f'{lab}: {len(tr)} ranks, {len(rg)} regrids by {tmax:.0f} s; PackAndSendMG live '
              f'requests, max over ranks: start {mx[np.isfinite(mx)][0]:.0f}, '
              f'end {mx[np.isfinite(mx)][-1]:.0f}')
    ax[0].set_xlabel('wall time [s]'); ax[1].set_xlabel('regrids so far')
    for x in ax:
        x.set_ylabel('live MPI requests from PackAndSendMG'); x.grid(alpha=0.3); x.legend()
    fig.suptitle('Rank-packed MG send requests never completed (libreqtrack), Q1 deck from t = 98, 24/12 ranks')
    fig.savefig(a.out, dpi=90)


if __name__ == '__main__':
    main()
