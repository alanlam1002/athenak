#!/usr/bin/env python3
"""c150 R-171 gate (state: which checks): (1) Valencia, metric_terms = cell, Michel L2, 100 cycles (B2C_val_L2)
vs the existing B2_val_L2 (bin bdfcadbb, same deck and 12 ranks): mhd.hst/user.hst rows bitwise;
(2) killing_t1, cell (B2C_t1_L2) vs KD_t1_L2 (bin 954f600a: the same Michel deck plus killing_source_diag, which
does not perturb t1, C-137 KV2): rows bitwise. (Not vs B2_t1_L2: bdfcadbb differs from 954f600a by 4e-12 at
cycle 3, codegen, which the Michel excision region amplifies to ~1e-4 in KE and ~1e-3 in rho-max by t = 20,
job 8988632; a 1e-9 gate against it failed.); (3) killing_t1 with analytic_bg,
2 cycles (B2P_t1): the ghost/owner probe of the face fluxes, 0 mismatches."""
import glob, subprocess, sys
import numpy as np
E = sys.argv[1]
K = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/c147_killing'
ok = True
for v, ref, tol in (('val', 'B2_val_L2', 0.0), ('t1', 'KD_t1_L2', 0.0)):
    for h in ('mhd.hst', 'user.hst'):
        a = np.loadtxt(glob.glob(f'{E}/B2C_{v}_L2/*.{h}')[0], ndmin=2)
        b = np.loadtxt(glob.glob(f'{K}/{ref}/*.{h}')[0], ndmin=2)
        n = len(a) - 1                       # drop the stop row
        d = np.abs(a[:n] - b[:n])/np.maximum(np.abs(b[:n]).max(axis=0), 1e-300)
        print(f'B2C_{v}_L2 vs {ref} {h}: {n} rows, max column-scaled rel diff {d.max():.3e} (tol {tol:g})')
        ok = ok and n >= 5 and d.max() <= tol
o = subprocess.run(['python3', '/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde/'
                    'wt_tde_scripts/scripts/c147_kprobe_check.py', f'{E}/B2P_t1'],
                   stdout=subprocess.PIPE, universal_newlines=True).stdout
print(o.strip())
ok = ok and ('mismatched=0 ' in o) and ('shared=0 ' not in o)
print('B2 gate:', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
