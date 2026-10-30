#!/usr/bin/env python3
"""c154 R-174 gates (1)-(2): Valencia (gV, ISO_val's deck to cycle 100) bitwise vs ISO_val (bin c4d5ac63);
killing_full arm (ii) without a hole (gK, KF_ii's deck to cycle 200, (c) inactive) vs KF_ii: tolerance 1e-9
column-scaled (round-off, as R-174 asks; the measured value is printed: 0 = bitwise)."""
import sys; sys.path.insert(0, sys.argv[1])
from gate_common import E, cmp, done
K = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/c147_killing'
ok = cmp(f'{E}/gV', f'{K}/ISO_val', 0.0)
ok = cmp(f'{E}/gK', f'{K}/KF_ii', 1e-9) and ok
done('VK', ok)
