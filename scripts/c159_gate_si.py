#!/usr/bin/env python3
"""c159 KF source-zero gate (replaces the flat gate "== Valencia", which no T-form meets on this frame-at-rest bed:
the killing_t1 control drifts from Valencia exactly as KF_z does, CPU strict, c159): on the flat static metric
(alpha = 1, beta = 0, K = 0, residual_scale 0, no hole) s_z and s_CFC vanish identically, so killing_fz and
killing_fcfc must equal killing_full arm (i) BITWISE (same binary), after 1 and 20 cycles: every dump and history row."""
import sys
sys.path.insert(0, sys.argv[1])
from gate_common import E, cmp, done
ok = True
for n in ('1', '20'):
    for v in ('gZz', 'gCc'):
        ok = cmp(f'{E}/{v}{n}', f'{E}/gZi{n}', 0.0) and ok
done('SZ', ok)
