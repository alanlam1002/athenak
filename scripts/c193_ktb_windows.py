#!/usr/bin/env python3
"""c193 (R-225 item 3): the tau budget per window from c157_ktb.py's _step.csv, arms side by side.
Per group (star bins by rho/rho_c; dense = rho >= 0.01 rho_c): D eps at the window start, its change over the window
(sum dDeps), the kcfc_bg correction sum Delta_amt (dtau_K - dtau_V, x dt), and the regrid-step share of dDeps.
    c193_ktb_windows.py T0:T1[,T0:T1...] LABEL=PREFIX ..."""
import csv, sys
from collections import defaultdict
win = [tuple(map(float, w.split(':'))) for w in sys.argv[1].split(',')]
arms = [a.split('=', 1) for a in sys.argv[2:]]
G = ['b0.5_int', 'b0.5_surf', 'b0.1_int', 'b0.1_surf', 'b0.01_int', 'b0.01_surf', 'dense']
for t0, t1 in win:
    print(f'== window t = {t0:g} -> {t1:g}: per group Deps(t0) | dDeps/Deps(t0) [%] | Delta/Deps(t0) [%] | regrid share of dDeps')
    rows = {}
    for lab, pre in arms:
        S = defaultdict(lambda: [None, 0.0, 0.0, 0.0, None, None])
        for r in csv.DictReader(open(pre + '_step.csv')):
            t = float(r['t'])
            if t < t0 - 1e-9 or t >= t1 or r['dDeps'] == '':
                continue
            s = S[r['group']]
            if s[0] is None:
                s[0] = float(r['Deps']); s[4] = t
            s[1] += float(r['dDeps']); s[2] += float(r['Delta_amt'])
            if r['contains_regrid'] == '1':
                s[3] += float(r['dDeps'])
            s[5] = t
        rows[lab] = S
    for g in G:
        line = []
        for lab, _ in arms:
            s = rows[lab].get(g)
            if not s or s[0] is None or s[0] == 0:
                line.append(f'{lab}: n/a'); continue
            line.append(f'{lab}: {s[0]:.3e} {100*s[1]/s[0]:+7.2f} {100*s[2]/s[0]:+7.2f} {s[3]/s[1] if s[1] else 0:+5.2f} [{s[4]:.1f}-{s[5]:.1f}]')
        print(f'{g:11s} ' + ' | '.join(line))
