#!/usr/bin/env python3
"""c154 B.1 gates (108 ranks, from B1N_cell's cycle-500 restart): G3, the ghost/owner probe of comoving t1 (bTP)
and killing_full arm (ii) with the moving background (bKP), 2 cycles each, 0 mismatches; G4 and R-174 (4) on
the real geometry (frame controller): bTR2 / bKR2 (restarted at cycle 510 from bTR1 / bKR1's restart) vs bTR1 /
bKR1 (to cycle 520): the common history rows bitwise."""
import sys; sys.path.insert(0, sys.argv[1])
from gate_common import E, cmp, probe, done
ok = probe(f'{E}/bTP')
ok = probe(f'{E}/bKP') and ok
ok = cmp(f'{E}/bTR2', f'{E}/bTR1', 0.0) and ok
ok = cmp(f'{E}/bKR2', f'{E}/bKR1', 0.0) and ok
done('B1', ok)
