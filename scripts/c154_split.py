#!/usr/bin/env python3
"""c154 (R-174 items 1(e), 2, 3): killing_full arm (ii)'s s_T from the KFSPLIT lines (dyn_grmhd_killing.cpp
ReportKillingStage: per stage, sum sqrt(g) s_T dV of the background part (analytic rate, xi-ddot) and the residual
part (the stage-differenced chord), over all cells and rho >= killing_full_split_rho; and the background's chord
effect, the full-metric chord minus the residual chord, which checks the analytic rate).
Per run: the time integrals int s_T dt with SSP-RK3's step weights b = (1/6, 1/6, 2/3) (Shu-Osher form, the
step's dt = stage 1's beta dt); the stage-to-stage jitter of the dense residual integral (each stage value minus
the median of its +-7 neighbours, rms; a noise-floor measure), and the stages without a previous metric
(the run's first, regrids).
    c154_split.py OUT.csv RUNDIR[:LABEL] [...]   (reads RUNDIR/out.log)"""
import csv
import os
import re
import sys

import numpy as np

BW = {1: 1.0 / 6.0, 2: 1.0 / 6.0, 3: 2.0 / 3.0}
KEYS = ['bg_all', 'res_all', 'bg_dense', 'res_dense', 'bgchord_all', 'bgchord_dense']


def parse(fn):
    rows = []
    for l in open(fn):
        if not l.startswith('KFSPLIT '):
            continue
        d = dict(re.findall(r'(\w+)=(\S+)', l))
        rows.append({k: float(v) for k, v in d.items()})
    return rows


def main():
    out = sys.argv[1]
    res = []
    for s in sys.argv[2:]:
        d, lab = (s.split(':') + [None])[:2]
        lab = lab or os.path.basename(d.rstrip('/'))
        R = parse(f'{d}/out.log')
        if not R:
            print(lab, 'no KFSPLIT lines'); continue
        dt = None
        I = {k: 0.0 for k in KEYS}
        for r in R:
            st = int(r['stage'])
            if st == 1:
                dt = r['beta_dt']
            if dt is None:
                continue
            for k in KEYS:
                I[k] += BW[st] * dt * r[k]
        x = np.array([r['res_dense'] for r in R])
        med = np.array([np.median(x[max(0, i - 7):i + 8]) for i in range(len(x))])
        jit = float(np.sqrt(np.mean((x - med)**2)))
        scale = float(np.sqrt(np.mean(med**2)))
        nop = sum(1 for r in R if r['have_prev'] == 0)
        bg = np.array([r['bg_all'] for r in R]); bc = np.array([r['bgchord_all'] for r in R])
        m = np.abs(bg) > 0
        rel = float(np.median(np.abs(bc[m] - bg[m]) / np.abs(bg[m]))) if m.any() else float('nan')
        row = dict(run=lab, nstage=len(R), t_end=R[-1]['tstage'], stages_without_prev=nop,
                   **{f'int_{k}': I[k] for k in KEYS}, jitter_res_dense=jit, rms_res_dense=scale,
                   bgchord_vs_bg_median_rel=rel)
        res.append(row)
        print(row, flush=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
        w.writeheader(); w.writerows(res)


if __name__ == '__main__':
    main()
