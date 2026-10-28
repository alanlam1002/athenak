#!/usr/bin/env python3
"""c144 (R-152 item 6 test (3), user-approved 2026-10-28): the S1 window of <cfc> solve_every_stages = N against N = 1
(c144_s1win W1/W2/W4: c110_2x2 CD's restart t = 115.45 -> 165, t_p = 135.9; deterministic, 24 ranks each).
  rst:    per arm and restart index (the common indices; times printed): c134_s1.rst_stats (f_bound, f_bound_B, the
          dM/dE quantiles q10/q50/q90 in units of dE, E_mean, M_rem, K_rem), and the differences to REF;
  dumps:  per arm and prim_3d index: c137_iso.stats (dense <K/K0 - 1>, warm/cold fractions, rho_max/rho_c), and the
          held-metric error against REF in the orbital plane (adm_xy vs REF's adm_xy; c143_cadence.adm_err with
          prim_xy for the density mask; dump times differ by < 1 dt, printed);
  hst:    rho-max (user.hst) and the dual-energy energy-branch fraction of the dense cells 1 - de-nentropy/de-ndense,
          their max |difference| to REF over REF's rows (each arm's series interpolated to REF's times);
  logs:   MCLOSE / ECLOSE R/M* and PASS, PHASE_TIMERS_FINAL wall, mg, cycles -> wall per M.
    c144_s1win.py OUT REF ARM [ARM ...] [--nproc 2]
-> OUT/c144_rst.csv, OUT/c144_dumps.csv, OUT/c144_summary.csv"""
import argparse
import glob
import os
import re
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c134_s1 as S1          # noqa: E402
import c137_iso as I          # noqa: E402
import c143_cadence as C      # noqa: E402

TP = 135.9
RHOC = 3.24e-4
K0 = 1.2346626297e-2
PFX = 'cfc_tde_wd_imbh_parabolic_tracker'


def hst(d):
    f = f'{d}/{PFX}.user.hst'
    h = [l for l in open(f) if l.startswith('#')][-1]
    names = [re.sub(r'^\[\d+\]=', '', x) for x in h[1:].split()]
    a = np.array([[float(x) for x in l.split()] for l in open(f) if l.strip() and not l.startswith('#')])
    return {n: a[:, i] for i, n in enumerate(names)}


def logs(d):
    s = open(f'{d}/out.log').read()
    out = {}
    for tag in ('MCLOSE', 'ECLOSE'):
        m = re.findall(rf'(?m)^{tag} cycles=.*?R/M\*=(\S+).*? (PASS|FAIL)', s)
        out[f'{tag}_R_over_Mstar'] = ';'.join(x for x, _ in m)
        out[f'{tag}_pass'] = ';'.join(p for _, p in m)
    m = re.findall(r'PHASE_TIMERS_FINAL .*?time=(\S+) cycles=(\d+) wall=(\S+) .*?mg=(\S+)', s)
    t1, cyc, wall, mg = (float(m[-1][0]), int(m[-1][1]), float(m[-1][2]), float(m[-1][3])) if m else (np.nan,) * 4
    out.update(cycles=cyc, wall_s=wall, mg_s=mg, t_end=t1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('ref')
    ap.add_argument('arms', nargs='+')
    ap.add_argument('--nproc', type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dirs = [a.ref] + a.arms
    lab = {d: os.path.basename(d.rstrip('/')) for d in dirs}

    idx = sorted(set.intersection(*[{f.split('.')[-2] for f in glob.glob(f'{d}/rst/*.rst')} for d in dirs]))
    jobs = [(lab[d], i, f'{d}/rst/{PFX}.{i}.rst', TP) for d in dirs for i in idx]
    with Pool(a.nproc) as p:
        res = [r for r, _ in p.map(S1.rst_stats, jobs)]
    ref = {r['stage']: r for r in res if r['label'] == lab[a.ref]}
    keys = ['f_bound', 'f_bound_B', 'q10', 'q50', 'q90', 'E_mean', 'M_rem', 'K_rem']
    for r in res:
        for k in keys:
            r[f'd_{k}'] = r[k] - ref[r['stage']][k]
    rows = []
    for d in dirs:
        for f in sorted(glob.glob(f'{d}/bin/{PFX}.prim_3d.*.bin')):
            t, rho, pr, dv, x = I.load(f)
            r = dict(arm=lab[d], idx=f.split('.')[-2])
            r.update(I.stats(t, rho, pr, dv, x, K0, RHOC))
            rows.append(r)
        if d == a.ref:
            continue
        for f in sorted(glob.glob(f'{d}/bin/{PFX}.adm_xy.*.bin')):
            i = f.split('.')[-2]
            fr = f'{a.ref}/bin/{PFX}.adm_xy.{i}.bin'
            fp = f'{a.ref}/bin/{PFX}.prim_xy.{i}.bin'
            if os.path.exists(fr) and os.path.exists(fp):
                r = dict(arm=lab[d], idx=i, kind='adm_xy')
                r.update(C.adm_err(fr, f, fp, RHOC))
                rows.append(r)
    H = {d: hst(d) for d in dirs}
    hr = H[a.ref]
    summ = []
    for d in dirs:
        h = H[d]
        fe = 1.0 - h['de-nentropy'] / np.maximum(h['de-ndense'], 1)
        fer = 1.0 - hr['de-nentropy'] / np.maximum(hr['de-ndense'], 1)
        tr = hr['time'][hr['time'] <= h['time'][-1]]
        s = dict(arm=lab[d])
        s.update(logs(d))
        s['wall_per_M'] = s['wall_s'] / (s['t_end'] - 115.45)
        s['rhomax_peak_over_rhoc'] = float(h['rho-max'].max() / RHOC)
        s['rhomax_end_over_rhoc'] = float(h['rho-max'][-1] / RHOC)
        s['max_rel_d_rhomax'] = float(np.max(np.abs(np.interp(tr, h['time'], h['rho-max']) /
                                                    hr['rho-max'][:len(tr)] - 1.0)))
        s['fen_mean'] = float(fe.mean())
        s['fen_max_abs_d'] = float(np.max(np.abs(np.interp(tr, h['time'], fe) - fer[:len(tr)])))
        s['de_dE_total_end'] = float(h['de-dE-total'][-1])
        summ.append(s)
        print(s, flush=True)
    for fn, R in (('c144_rst.csv', res), ('c144_dumps.csv', rows), ('c144_summary.csv', summ)):
        ks = []
        for r in R:
            ks += [k for k in r if k not in ks]
        with open(f'{a.out}/{fn}', 'w') as fo:
            fo.write(','.join(ks) + '\n')
            for r in R:
                fo.write(','.join(f'{r[k]:.6g}' if isinstance(r.get(k), float) else str(r.get(k, '')) for k in ks) + '\n')


if __name__ == '__main__':
    main()
