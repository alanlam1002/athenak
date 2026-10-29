#!/usr/bin/env python3
"""c145 (R-152 3(d), R-153): per 3D dump (prim_3d), per density band of YCLOSE (rho/rho_c >= 0.1, [0.01, 0.1),
[0.001, 0.01), [1e-4, 0.001); rho_c = 3.24e-4), the band's mass sum rho dV, its tracer content sum rho Y dV
(Y = s_00; coordinate volume, W psi^6 ~ 1 + O(1e-2) omitted: a normalisation for the YCLOSE items only) and its
mass-weighted <K/K0 - 1> (K = P/rho^Gamma, K0 = kappa). Float32 dumps.
    c145_bands_dump.py OUT.csv LABEL=RUNDIR [...]"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tde_ensemble_plots import bc   # noqa: E402

RHOC, K0, GAM = 3.24e-4, 1.2346626297e-2, 5.0 / 3.0
EDGES = [np.inf, 0.1, 0.01, 0.001, 1e-4]
rows = []
for spec in sys.argv[2:]:
    lab, d = spec.split('=', 1)
    for f in sorted(glob.glob(f'{d}/bin/*.prim_3d.*.bin')):
        P = bc.read_binary(f)
        g = np.asarray(P['mb_geometry'])
        acc = np.zeros((4, 3))
        for b in range(len(g)):
            rho = np.asarray(P['mb_data']['dens'][b], dtype=np.float64)
            pr = np.asarray(P['mb_data']['press'][b], dtype=np.float64)
            y = np.asarray(P['mb_data']['s_00'][b], dtype=np.float64)
            dv = np.prod(g[b, [1, 3, 5]] - g[b, [0, 2, 4]]) / rho.size
            x = rho / RHOC
            k = pr / np.maximum(rho, 1e-300)**GAM / K0 - 1.0
            for q in range(4):
                m = (x < EDGES[q]) & (x >= EDGES[q + 1])
                if m.any():
                    acc[q] += [rho[m].sum() * dv, (rho[m] * y[m]).sum() * dv, (rho[m] * k[m]).sum() * dv]
        r = dict(label=lab, t=float(P['time']))
        for q in range(4):
            r[f'M_b{q}'] = acc[q, 0]
            r[f'DY_b{q}'] = acc[q, 1]
            r[f'K_b{q}'] = acc[q, 2] / acc[q, 0] if acc[q, 0] > 0 else np.nan
        rows.append(r)
        print({k: (f'{v:.5g}' if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
with open(sys.argv[1], 'w') as fo:
    fo.write(','.join(rows[0]) + '\n')
    for r in rows:
        fo.write(','.join(f'{v:.8g}' if isinstance(v, float) else str(v) for v in r.values()) + '\n')
