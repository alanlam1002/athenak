#!/usr/bin/env python3
"""c150 R-171 gate (state: which checks): (1) Valencia, metric_terms = cell, Michel L2, 100 cycles (B2C_val_L2)
vs the existing B2_val_L2 (bin bdfcadbb, same deck and 12 ranks): mhd.hst/user.hst rows bitwise;
(2) killing_t1, cell (B2C_t1_L2) vs B2_t1_L2: AddKillingTerms' kernel changed, so PVC fast-fp codegen may move it
at round-off (C-137: 4e-12): column-scaled max relative difference <= 1e-9; (3) killing_t1 with analytic_bg,
2 cycles (B2P_t1): the ghost/owner probe of the face fluxes, 0 mismatches."""
import glob, subprocess, sys
import numpy as np
E = sys.argv[1]
K = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/c147_killing'
ok = True
for v, tol in (('val', 0.0), ('t1', 1e-9)):
    for h in ('mhd.hst', 'user.hst'):
        a = np.loadtxt(glob.glob(f'{E}/B2C_{v}_L2/*.{h}')[0], ndmin=2)
        b = np.loadtxt(glob.glob(f'{K}/B2_{v}_L2/*.{h}')[0], ndmin=2)
        n = len(a) - 1                       # drop the stop row
        d = np.abs(a[:n] - b[:n])/np.maximum(np.abs(b[:n]).max(axis=0), 1e-300)
        print(f'B2C_{v}_L2 {h}: {n} rows, max column-scaled rel diff {d.max():.3e} (tol {tol:g})')
        ok = ok and n >= 5 and d.max() <= tol
o = subprocess.run(['python3', '/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde/'
                    'wt_tde_scripts/scripts/c147_kprobe_check.py', f'{E}/B2P_t1'],
                   stdout=subprocess.PIPE, universal_newlines=True).stdout
print(o.strip())
ok = ok and ('mismatched=0 ' in o) and ('shared=0 ' not in o)
print('B2 gate:', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
