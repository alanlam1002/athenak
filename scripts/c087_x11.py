#!/usr/bin/env python3
"""C-087 (R-087 item 11): judge a run against the three-sample R6 noise ensemble.

Physics rows from c051_s2.py's own functions (rst_pass, physics, fallback; c044_mdot.mdot)
on each run's last restart (t = 170). Pass: |X - mean(AX2, N1, N2c)| <= 2 sigma_diff, with
sigma_diff = RMS of the three pairwise differences (C-081's definition). Also the run's own
logs: MLEDGER closure, CRECOV totals (cold recovery, FOFC tally), C2P errors, dt.

    c087_x11.py RUN_DIR [--out FILE]
"""
import argparse
import glob
import itertools
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c051_s2 as S          # noqa: E402
import c044_mdot as M        # noqa: E402

ENS = ['RC_R6AX2', 'RC_R6N1', 'RC_R6N2c']
ROWS = [('K', 'dense K/K0', 'rel'), ('M3', 'M3', 'abs'), ('rem', 'remnant', 'rel'),
        ('mb', 'bound, all', 'rel'), ('bound', 'bound frac (sec. 7)', 'abs'),
        ('mbd', 'debris bound', 'rel'), ('f1', 't(M_fb 1%)', 'rel'), ('f10', 't(M_fb 10%)', 'rel'),
        ('thalf', 't_half', 'rel'), ('pk', 'peak Mdot', 'rel'), ('qb', 'bound quantiles', 'q'),
        ('qu', 'unbound quantiles', 'q'), ('Lz', '<Lz>', 'rel')]


def rows(run, tp):
    fn = sorted(glob.glob(f'{S.C}/{run}/rst/*.rst'))[-1]
    z = S.rst_pass(fn)
    ph = S.physics(z)
    fb = S.fallback(z, tp)
    deb = ~z['rem'].astype(bool)
    md = M.mdot(z['E'][deb], z['dM'][deb], tp)
    return fn, dict(K=ph['K'], M3=ph['M3'], rem=ph['rem'], bound=ph['bound'], mb=fb[2],
                    mbd=fb[4], f1=fb[3][0], f10=fb[3][1], thalf=fb[3][2], pk=md.max(),
                    qb=np.asarray(ph['qb']), qu=np.asarray(ph['qu']), Lz=ph['Lz'], t=ph['t'])


def diff(x, y, kind):
    if kind == 'rel':
        return x / y - 1
    if kind == 'abs':
        return x - y
    return float(np.max(np.abs(np.asarray(x) - np.asarray(y))))


def logs(run):
    d = f'{S.C}/{run}'
    files = sorted(glob.glob(f'{d}/seg_*.out')) or [f'{d}/ensemble.out']
    led, cre, err, dts = [], [], 0, {}
    for f in files:
        for ln in open(f, errors='ignore'):
            if ln.startswith('MLEDGER cycle=') or ln.startswith('CRECOV '):
                kv = dict(x.split('=', 1) for x in ln.split()[1:] if '=' in x)
                (led if ln.startswith('MLEDGER') else cre).append(kv)
                if ln.startswith('MLEDGER'):
                    dts[int(kv['cycle'])] = float(kv['dt'])
            elif 'error occurred during the primitive solve' in ln:
                err += 1
    return led, cre, err, dts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--out')
    a = ap.parse_args()
    cfg = S.RUNS['R6']
    tp = M.t_peri(cfg['src'], cfg['rp'])
    out = []
    V = {}
    for r in ENS + [a.run]:
        fn, V[r] = rows(r, tp)
        out.append(f'{r}: {os.path.basename(fn)} t = {V[r]["t"]:.2f}')
    out.append(f'{"row":20s} {"X":>12s} {"mean":>12s} {"X-mean":>11s} {"sig_diff":>10s} {"/sig":>7s}  verdict')
    npass = 0
    for key, lab, kind in ROWS:
        pairs = [diff(V[x][key], V[y][key], kind) for x, y in itertools.combinations(ENS, 2)]
        sig = float(np.sqrt(np.mean(np.square(pairs))))
        if kind == 'q':
            mean = np.mean([V[r][key] for r in ENS], axis=0)
            dx = diff(V[a.run][key], mean, 'q')
            xs, ms = f'{np.max(V[a.run][key]):.4f}', '(vector)'
        else:
            mean = float(np.mean([V[r][key] for r in ENS]))
            dx = diff(V[a.run][key], mean, kind)
            xs, ms = f'{float(V[a.run][key]):.6g}', f'{mean:.6g}'
        ok = abs(dx) <= 2 * sig
        npass += ok
        out.append(f'{lab:20s} {xs:>12s} {ms:>12s} {dx:+11.3e} {sig:10.3e} {abs(dx)/sig:7.2f}  '
                   f'{"ok" if ok else "OUTSIDE 2 sig"}')
    out.append(f'physics rows within 2 sig_diff: {npass}/{len(ROWS)}')
    # logs
    for r in [a.run] + ENS:
        led, cre, err, dts = logs(r)
        X = sum(float(k['X']) for k in led)
        Y = sum(float(k['Y']) + float(k['Fout']) for k in led)
        med = np.median(list(dts.values())) if dts else float('nan')
        line = f'{r}: cycles {len(led)} median dt {med:.6g} C2P errors {err} stage X {X:.3e} Y+Fout {Y:.3e}'
        if cre:
            st = [k for k in cre if k['path'] == 'steps']
            g = lambda k: sum(float(c.get(k, 0)) for c in st)
            line += (f' | CRECOV steps n {g("n"):.3e} dtau {g("dtau"):.3e} DSchanged {g("DSchanged"):.0f}'
                     f' dDY {g("dDY"):.3e} m_erased {g("m_erased"):.3e} fofc_rescued {g("fofc_rescued"):.3e}'
                     f' fofc_hydro {g("fofc_hydro"):.3e} fofc_exc {g("fofc_exc"):.3e}'
                     f' faces {g("fofc_faces"):.3e} sum|dF| {g("fofc_dFscal"):.3e} sum|Fup| {g("fofc_Fup"):.3e}'
                     f' | regrid/init n {sum(float(c["n"]) for c in cre if c["path"] != "steps"):.3e}'
                     f' dtau_total {cre[-1]["dtau_total"]}')
        out.append(line)
    txt = '\n'.join(out)
    print(txt)
    if a.out:
        open(a.out, 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
