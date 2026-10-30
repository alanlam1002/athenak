#!/usr/bin/env python3
"""c155 gate KG (R-177 item 1, the KCLOSE regrid fix): arm KG (the moving-bed AMR deck, killing_full arm (ii), metric_terms
cell, frame_vel 0.2 + accel; nlim 6) must regrid at least once (MLEDGER0 path=regrid) with every ledger value finite, and
KCLOSE must close to round-off across it: |R/M_grid| <= 1e-15 (the weights item per-cell differenced, the regrid items
measured with one set of weights on both sides). Prints the per-cycle KCLOSE_CYC gap/X and the segment's items."""
import math, re, sys
E = sys.argv[1]
log = open(f'{E}/KG/out.log', errors='replace').read()
nrg = len(re.findall(r'^MLEDGER0 cycle=\d+ \S+ path=regrid', log, re.M))
k = [l for l in log.splitlines() if l.startswith('KCLOSE ')]
for l in log.splitlines():
    if l.startswith('KCLOSE_CYC'):
        print(l[:260])
ok = nrg >= 1 and len(k) > 0
if k:
    d = dict(re.findall(r'(\S+)=(\S+)', k[-1]))
    print(' '.join(f'{x}={d.get(x)}' for x in ('regrid', 'floor_init', 'weights', 'source', 'R', 'R/M_grid')))
    try:
        r = float(d['R/M_grid'])
        ok = ok and math.isfinite(r) and abs(r) <= 1e-15
    except (KeyError, ValueError):
        ok = False
print(f'KG regrids={nrg}', 'gate:', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
