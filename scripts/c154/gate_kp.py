#!/usr/bin/env python3
"""c154 R-174 gate (3): killing_full arm (ii) ghost/owner probe of the face fluxes (F_D, F_S, F_T), 2 cycles."""
import sys; sys.path.insert(0, sys.argv[1])
from gate_common import E, probe, done
done('KP', probe(f'{E}/gKP'))
