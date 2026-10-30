#!/usr/bin/env python3
"""c154 R-175 G2: a fixed frame (BH frame, Michel L2, analytic_bg): killing_weights = comoving (gG2, to cycle 100)
vs the existing killing_t1 + analytic_bg B2A_t1_L2 (bin 37cb71b9, same deck and 12 ranks): history rows bitwise."""
import sys; sys.path.insert(0, sys.argv[1])
from gate_common import E, cmp, done
done('G2', cmp(f'{E}/gG2', '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/c150_rframe/B2A_t1_L2', 0.0))
