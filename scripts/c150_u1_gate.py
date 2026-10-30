#!/usr/bin/env python3
"""c150 U1 gate (R-168): ISO_A, 20 cycles, riemann_frame = boosted with riemann_frame_U =
eulerian (U1i_feul) vs the default HLLE (U1i_def). Both history files (mhd.hst, user.hst) must
agree to round-off: max over rows/columns of |a - b| / (the column's max |a|) <= 1e-10
(column-scaled, so a column that is round-off noise about zero, e.g. a net y-momentum, does not
fail the gate) (the frame
solver reproduces HLLE to ~1e-16 per face; 20 cycles of an RK3 ISO_A run amplify that to well
below 1e-10, cf. the 1D tube: 6e-16 after 1 cycle, 4e-15 after 10). Exit 0 = pass.
    u1_gate.py RUNDIR"""
import glob
import sys
import numpy as np
E = sys.argv[1]
worst = 0.0
ok = True
for h in ('mhd.hst', 'user.hst'):
    fa = glob.glob(f'{E}/U1i_def/*.{h}')
    fb = glob.glob(f'{E}/U1i_feul/*.{h}')
    if not fa or not fb:
        print(f'U1 gate: {h} missing'); ok = False; continue
    a, b = np.loadtxt(fa[0], ndmin=2), np.loadtxt(fb[0], ndmin=2)
    n = min(len(a), len(b))
    if n < 2 or len(a) != len(b):
        print(f'U1 gate: {h} rows {len(a)} vs {len(b)}'); ok = False
    d = np.abs(a[:n] - b[:n])/np.maximum(np.abs(a[:n]).max(axis=0), 1e-300)
    worst = max(worst, float(d.max()))
    print(f'U1 gate: {h} rows {n}, max rel diff {d.max():.3e} (column {int(np.argmax(d.max(axis=0)))})')
for r in ('U1i_def', 'U1i_feul'):
    if 'Terminating on cycle limit' not in open(f'{E}/{r}/out.log').read():
        print(f'U1 gate: {r} did not reach its cycle limit'); ok = False
ok = ok and worst <= 1e-10
print(f'U1 gate: {"PASS" if ok else "FAIL"} (worst {worst:.3e})')
sys.exit(0 if ok else 1)
