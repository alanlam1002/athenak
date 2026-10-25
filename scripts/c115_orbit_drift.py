#!/usr/bin/env python3
"""R-127 Q2 (ii) (c115): the mass-weighted BH-frame Killing energy <E> = <-u_t - 1> and <L_z> about
x_BH, over all cells outside r_BH < 3 (C-044's c044_series definition, tde_dmde.cells), per surviving
restart of the four rp10 2x2 arms (base8), plus the grid mass. The DE-on arms' restarts carry the
flag slot after the tracer (nscal = 2 in the u0 layout).
    c115_orbit_drift.py OUT.csv LABEL=RUNDIR:NSCAL [...]
"""
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tde_dmde as D                              # noqa: E402

B = 'cfc_tde_wd_imbh_parabolic_tracker'


def one(arg):
    lab, f, ns = arg
    c = D.cells(f, rcut=3.0, nscal=ns)
    dM, xb = c['dM'], c['xbh']
    L = (c['x'] - xb[0]) * c['uy'] - (c['y'] - xb[1]) * c['ux']
    return (lab, c['t'], dM.sum(), (c['E'] * dM).sum() / dM.sum(), (L * dM).sum() / dM.sum(),
            dM[c['E'] < 0].sum() / dM.sum())


jobs = []
for spec in sys.argv[2:]:
    lab, rest = spec.split('=', 1)
    d, ns = rest.rsplit(':', 1)
    jobs += [(lab, f, int(ns)) for f in sorted(glob.glob(f'{d}/rst/{B}.*.rst'))]
with Pool(min(len(jobs), 10)) as p:
    R = p.map(one, jobs)
rows = ['label,t,grid_mass,E_mean,Lz_mean,bound_frac'] + [f'{a},{t:.3f},{m:.6e},{e:.6e},{l:.6f},{b:.4f}' for a, t, m, e, l, b in R]
open(sys.argv[1], 'w').write('\n'.join(rows) + '\n')
print('\n'.join(rows))
