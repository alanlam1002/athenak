#!/usr/bin/env python3
"""R-127 Q2 (iii) (c116): where the entropy-tracer total (mhd hst scal-0 = int D Y dV) rises, for the
DE-on rp10 arms. By the C2P (primitive_solver_hyd.hpp), the tracer total changes only by
  R  the energy-branch resync of dense FLAGGED cells and of all non-dense cells (D <= rho_sw),
  C  the cold recovery (CRECOV dDY, summed per cycle; dense unflagged cells keep D Y otherwise),
  F  transport through the outer boundary and the excision (conservative inside);
so R - F = d(scal-0) - C. Per phase (rp10 t_p = 135.9: pre < t_p - 10, peri t_p +- 10, early post
to t_p + 50, late post to 300) this prints d(scal-0), C, and the DESHK flagged-cell counts summed
over cycles and RK stages: dense flagged (class 6), dense flagged on the energy branch, and
dense flagged that the guard put on the entropy branch (dense - energy), as cell x M (each
DESHK line's counts x its mean dt / 3 stages); and R - F = d(scal-0) - C. Last occurrence wins
where segments repeat cycles.
    c116_tracer_budget.py OUT.csv LABEL=RUNDIR [...]
"""
import re
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c110_analysis import col                       # noqa: E402

TP = 135.9
PH = [('pre', 0, TP - 10), ('peri', TP - 10, TP + 10), ('early', TP + 10, TP + 50), ('late', TP + 50, 300.01)]


def logs(d):
    ct, cr, dk = {}, {}, {}
    for l in open(f'{d}/out.log', errors='replace'):
        g = re.match(r'elapsed=\S+ cycle=(\d+) time=(\S+)', l)
        if g:
            ct[int(g.group(1))] = float(g.group(2)); continue
        g = re.match(r'CRECOV cycle=(\d+) path=steps .*dDY=(\S+) ', l)
        if g:
            cr[int(g.group(1))] = float(g.group(2)); continue
        g = re.match(r'DESHK cycle=(\d+) ncyc=\d+ stage=(\d) calls=\d+ all=\S+ dense=(\S+) energy=(\S+)', l)
        if g:
            dn = [int(x) for x in g.group(3).split(',')]
            en = [int(x) for x in g.group(4).split(',')]
            dk[(int(g.group(1)), int(g.group(2)), int(re.search(r'ncyc=(\d+)', l).group(1)))] = (dn[6], en[6])
    return ct, cr, dk


rows = ['label,phase,t0,t1,dS,dS_over_S0,C_coldrec,C_over_dS,dense_flagged_cellM,energy_branch_cellM,guard_entropy_cellM,R_minus_F']
for spec in sys.argv[2:]:
    lab, d = spec.split('=', 1)
    t, s = col(d, 'mhd', 'scal-0')
    ct, cr, dk = logs(d)
    cyc = np.array(sorted(ct)); tc = np.array([ct[c] for c in cyc])
    for ph, a, b in PH:
        s0, s1 = np.interp(a, t, s), np.interp(min(b, t[-1]), t, s)
        C = sum(v for c, v in cr.items() if c in ct and a <= ct[c] < b)
        nd = ne = 0.0
        for (c, st, nc), (x, y) in dk.items():
            if c in ct and (c - nc) in ct and a <= ct[c] < b:
                w = (ct[c] - ct[c - nc]) / nc / 3.0       # cell-calls -> cell x M (3 RK stages)
                nd += x * w; ne += y * w
        rows.append(f'{lab},{ph},{a:.1f},{min(b, t[-1]):.1f},{s1 - s0:.4e},{(s1 - s0) / s[0]:.4f},{C:.4e},'
                    f'{C / (s1 - s0) if s1 != s0 else np.nan:.4f},{nd:.4e},{ne:.4e},{nd - ne:.4e},{s1 - s0 - C:.4e}')
open(sys.argv[1], 'w').write('\n'.join(rows) + '\n')
print('\n'.join(rows))
