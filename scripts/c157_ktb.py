#!/usr/bin/env python3
"""c157 (R-184 items 1-4, user-approved 2026-10-30): the KTB lines of <mhd> tau_budget_every (athenak wip/tau-budget,
dyn_grmhd_taubudget.cpp TauBudgetPost), one per stage: per bin the sums over the bin's cells (x dV_coord) of
  dv, D, D eps                                   (sqrt(g) W rho, sqrt(g) W rho eps)
  Delta, |Delta|, w |Delta|, w                    (Delta = dtau_K - dtau_V per unit time; w = alpha0 (t1) or alpha (killing_full))
  A0tr, A0tf, Artr, Artf, B                        (killing_full: A = -1/2 sqrt(g) S^ij V_ij, background/residual x trace/rest;
                                                    B = Delta - A. t1: B is R = Delta - M_T/alpha0)
  Sm, Sdisc, M                                     (comoving t1, tau level: S_motion/alpha0, S_motion,disc/alpha0, their difference)
  nfb                                              (cells whose Valencia face flux was inverted from the flux-corrected fluxes)
Stage rates are combined into a step amount with SSP-RK3's weights (1/6, 1/6, 2/3) x the step's dt (stage 1's dt; the
stage dt's ratios 1 : 1/4 : 2/3 are checked). D eps per step = stage-1 values of consecutive cycles.
Steps are split by the regrid flag (a MeshBlock created or deleted since the previous cycle): the D eps interval n -> n+1
"contains a regrid" when cycle n+1's flag is set; the rates of step n+1 are "post-regrid" (killing_full: no chord there).
    c157_ktb.py OUT_PREFIX LOG [LOG ...]
Writes OUT_PREFIX_stage.csv (cycle, stage, bin rows), OUT_PREFIX_step.csv (per step, per group) and OUT_PREFIX_sum.csv
(window totals per group: the star bins 0-7 each, dense = rho >= 0.01 rho_c (bins 0-5), surf/int dense; or the shells)."""
import csv
import re
import sys

import numpy as np

COMP = ['dv', 'D', 'Deps', 'Delta', 'absDelta', 'wabsDelta', 'w', 'A0tr', 'A0tf', 'Artr', 'Artf', 'B', 'Sm', 'Sdisc', 'M',
        'nfb']
RATES = ['Delta', 'A0tr', 'A0tf', 'Artr', 'Artf', 'B', 'Sm', 'Sdisc', 'M']
WRK3 = {1: 1.0/6.0, 2: 1.0/6.0, 3: 2.0/3.0}
STAR = ['b0.5_int', 'b0.5_surf', 'b0.1_int', 'b0.1_surf', 'b0.01_int', 'b0.01_surf', 'b1e-3_int', 'b1e-3_surf']
SHELL = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 13.0, 16.0, 20.0]


def parse(logs):
    rows = {}
    for f in logs:
        for ln in open(f, errors='replace'):
            if not ln.startswith('KTB '):
                continue
            d = dict(re.findall(r'(\w+)=(\S+)', ln))
            nb = int(d['nb'])
            r = dict(cycle=int(d['cycle']), stage=int(d['stage']), time=float(d['time']), dt=float(d['dt']),
                     regrid=int(d['regrid']), mode=d['mode'], bins=d['bins'],
                     b=np.array([[float(x) for x in d[f'b{q}'].split(',')] for q in range(nb)]))
            rows[(r['cycle'], r['stage'])] = r   # a restart's repeated cycle: the later wins
    return [rows[k] for k in sorted(rows)]


def groups(bins, nb):
    if bins == 'star':
        g = {STAR[q]: [q] for q in range(nb)}
        g.update(dense=[0, 1, 2, 3, 4, 5], dense_int=[0, 2, 4], dense_surf=[1, 3, 5], all=list(range(nb)))
        return g
    g = {f'r{SHELL[q]:g}-{SHELL[q + 1]:g}': [q] for q in range(nb)}
    g.update({'r2-3': [1, 2], 'r3-5': [3, 4, 5], 'r5-10': [6, 7, 8], 'r10-20': [9, 10, 11]})
    return g


def main():
    out, logs = sys.argv[1], sys.argv[2:]
    st = parse(logs)
    if not st:
        sys.exit('no KTB lines')
    nb, bins = st[0]['b'].shape[0], st[0]['bins']
    G = groups(bins, nb)
    with open(out + '_stage.csv', 'w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow(['cycle', 'stage', 'time', 'dt', 'regrid', 'bin'] + COMP)
        for r in st:
            for q in range(nb):
                w.writerow([r['cycle'], r['stage'], r['time'], r['dt'], r['regrid'], q] + list(r['b'][q]))
    by = {}
    for r in st:
        by.setdefault(r['cycle'], {})[r['stage']] = r
    cyc = sorted(c for c in by if set(by[c]) == {1, 2, 3})
    bad = [c for c in cyc if abs(by[c][2]['dt']/by[c][1]['dt'] - 0.25) > 1e-9 or
           abs(by[c][3]['dt']/by[c][1]['dt'] - 2.0/3.0) > 1e-9]
    if bad:
        sys.exit(f'stage dt ratios are not SSP-RK3 at cycles {bad[:5]}')
    ri = [COMP.index(k) for k in RATES]
    steps = []
    for n, c in enumerate(cyc):
        s = by[c]
        dt = s[1]['dt']
        amt = sum(WRK3[q]*s[q]['b'][:, ri] for q in (1, 2, 3))*dt          # (nb, nrates)
        nxt = by.get(c + 1, {}).get(1)
        deps = (nxt['b'][:, 2] - s[1]['b'][:, 2]) if nxt is not None else None
        steps.append(dict(cycle=c, t=s[1]['time'], dt=dt, post=s[1]['regrid'],
                          contains=(nxt['regrid'] if nxt is not None else -1), amt=amt, deps=deps,
                          eps=s[1]['b'][:, 2], D=s[1]['b'][:, 1], dv=s[1]['b'][:, 0], absd=s[1]['b'][:, 4],
                          wabsd=s[1]['b'][:, 5], wv=s[1]['b'][:, 6], nfb=s[1]['b'][:, 15]))
    with open(out + '_step.csv', 'w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow(['cycle', 't', 'dt', 'post_regrid', 'contains_regrid', 'group', 'dDeps', 'Deps', 'D', 'dv']
                   + [k + '_amt' for k in RATES])
        for p in steps:
            for gn, ix in G.items():
                dd = '' if p['deps'] is None else p['deps'][ix].sum()
                w.writerow([p['cycle'], p['t'], p['dt'], p['post'], p['contains'], gn, dd, p['eps'][ix].sum(),
                            p['D'][ix].sum(), p['dv'][ix].sum()] + list(p['amt'][ix].sum(axis=0)))
    with open(out + '_sum.csv', 'w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow(['group', 'steps', 'nsteps', 't0', 't1', 'dDeps', 'Deps_t0', 'D_mean', 'dv_mean', 'absDelta_per_dv',
                    'wabsDelta_per_dv', 'w_mean', 'nfb_mean'] + [k + '_amt' for k in RATES])
        for gn, ix in G.items():
            for sel, f in (('all', lambda p: True), ('contains_regrid', lambda p: p['contains'] == 1),
                           ('no_regrid', lambda p: p['contains'] == 0), ('post_regrid', lambda p: p['post'] == 1)):
                ps = [p for p in steps if f(p)]
                if not ps:
                    continue
                dd = sum(p['deps'][ix].sum() for p in ps if p['deps'] is not None)
                a = sum(p['amt'][ix].sum(axis=0) for p in ps)
                dv = np.mean([p['dv'][ix].sum() for p in ps])
                w.writerow([gn, sel, len(ps), ps[0]['t'], ps[-1]['t'] + ps[-1]['dt'], dd, ps[0]['eps'][ix].sum(),
                            np.mean([p['D'][ix].sum() for p in ps]), dv,
                            np.mean([p['absd'][ix].sum() for p in ps])/dv if dv else np.nan,
                            np.mean([p['wabsd'][ix].sum() for p in ps])/dv if dv else np.nan,
                            np.mean([p['wv'][ix].sum() for p in ps])/dv if dv else np.nan,
                            np.mean([p['nfb'][ix].sum() for p in ps])] + list(a))
    print(f'{out}: {len(st)} stage lines, {len(steps)} steps, mode {st[0]["mode"]}, bins {bins}')


if __name__ == '__main__':
    main()
