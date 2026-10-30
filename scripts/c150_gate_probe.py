#!/usr/bin/env python3
"""c150 U3 gate: every probed run (U3_*) must have 0 ghost/owner mismatches (c147_kprobe_check.py)."""
import subprocess, sys
E = sys.argv[1]
ok = True
for r in ('U3_R2', 'U3_hllcN', 'U3_hllcG'):
    o = subprocess.run(['python3', '/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde/'
                        'wt_tde_scripts/scripts/c147_kprobe_check.py', f'{E}/{r}'], capture_output=True, text=True).stdout
    print(o.strip())
    ok = ok and ('mismatched=0 ' in o) and ('shared=0 ' not in o)
print('U3 gate:', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
