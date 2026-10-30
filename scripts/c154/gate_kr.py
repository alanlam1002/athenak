#!/usr/bin/env python3
"""c154 R-174 gate (4): restart bitwise across a stage boundary: gKR1 (P3's deck, fresh, to cycle 40, restarts at
cycles 20 and 40) vs gKR2 (restarted from gKR1's cycle-20 restart to cycle 40): every common history row after
the restart and the common dumps bitwise. The first stage after the restart uses kf_prev (dyngr.kfprev)."""
import sys; sys.path.insert(0, sys.argv[1])
from gate_common import E, cmp, done
done('KR', cmp(f'{E}/gKR2', f'{E}/gKR1', 0.0))
