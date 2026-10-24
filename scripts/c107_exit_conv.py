"""R-118 Q1 / R-122 item 6: the item-4 exit convergence test (c107). The exit-vs-no-exit difference
Delta_exit(L) = Q_exit - Q_noexit at L4 and L5, both pairs from R7 00013 (L5 = that L4 state refined at
restart, option (A)). Criterion: |Delta_exit(L5)| <= 1/2 |Delta_exit(L4)| (at least first order) in rho_max,
the entropy-tracer total (scal-0) and the bound mass. Also, per run: the MCLOSE line, the dt series across
the ramp (largest cycle-to-cycle jump |dt_n/dt_(n-1) - 1|) and the regrid count.

    c107_exit_conv.py OUT.csv NOEXIT_L4 EXIT_L4 NOEXIT_L5 EXIT_L5
"""
import re
import sys

import numpy as np

CSV, RUNS = sys.argv[1], dict(zip(('n4', 'x4', 'n5', 'x5'), sys.argv[2:6]))
BN = 'cfc_tde_wd_imbh_parabolic_tracker'
T_SW, T_RAMP = 253.82, 20.0
TIMES = (T_SW + 0.5 * T_RAMP, T_SW + T_RAMP, 285.0, 290.0, 295.0, 299.9)


def hst(d, f):
    hdr = [l for l in open(f'{d}/{BN}.{f}.hst') if l.startswith('#')][-1]
    names = [x.split('=')[-1] for x in hdr.replace('#', '').split()]
    a = np.loadtxt(f'{d}/{BN}.{f}.hst')
    _, i = np.unique(a[:, 0], return_index=True)   # the terminal Finalize row repeats a time
    return names, a[np.sort(i)]


def series(d, q):
    if q == 'Mbound':
        t, m, tc = [], [], {}
        for l in open(f'{d}/out.log'):
            g = re.match(r'elapsed=\S+ cycle=(\d+) time=(\S+)', l)
            if g:
                tc[int(g.group(1))] = float(g.group(2))
            g = re.match(r'CRECOV cycle=(\d+) .*Mbound_all=(\S+)', l)
            if g and int(g.group(1)) in tc:
                t.append(tc[int(g.group(1))])
                m.append(float(g.group(2)))
        return np.array(t), np.array(m)
    f = 'user' if q in ('rho-max', 'alpha-min') else 'mhd'
    names, a = hst(d, f)
    return a[:, 0], a[:, names.index(q)]


def runinfo(d):
    mc = [l.split()[1] + ' ' + ('PASS' if 'PASS' in l else 'FAIL' if 'FAIL' in l else '?')
          for l in open(f'{d}/out.log') if l.startswith('MCLOSE ')]
    dts = [float(g.group(1)) for l in open(f'{d}/out.log')
           for g in [re.match(r'elapsed=\S+ cycle=\d+ time=\S+ dt=(\S+)', l)] if g]
    jump = max(abs(dts[i] / dts[i - 1] - 1) for i in range(1, len(dts))) if len(dts) > 1 else np.nan
    nrg = sum(1 for l in open(f'{d}/out.log') if re.match(r'MLEDGER0 cycle=\d+ .*path=regrid', l))
    return mc, jump, nrg


rows = ['quantity,t,Q_noexit_L4,Delta_L4,Q_noexit_L5,Delta_L5,ratio_L5_over_L4,pass_half']
for q in ('rho-max', 'scal-0', 'mass', 'Mbound', 'tot-E', 'alpha-min'):
    s = {k: series(v, q) for k, v in RUNS.items()}
    for t in TIMES:
        v = {k: np.interp(t, *s[k]) for k in s}
        d4, d5 = v['x4'] - v['n4'], v['x5'] - v['n5']
        r = abs(d5) / abs(d4) if d4 != 0 else np.nan
        rows.append(f'{q},{t:.2f},{v["n4"]:.10g},{d4:.4e},{v["n5"]:.10g},{d5:.4e},{r:.3f},'
                    f'{"yes" if r <= 0.5 else "no"}')
rows.append('')
rows.append('run,dir,MCLOSE,max_dt_jump,regrids')
for k, v in RUNS.items():
    mc, jump, nrg = runinfo(v)
    rows.append(f'{k},{v},{";".join(mc) or "none"},{jump:.3e},{nrg}')
open(CSV, 'w').write('\n'.join(rows) + '\n')
print('\n'.join(rows))
