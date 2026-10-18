#!/usr/bin/env python3
"""C-053 (R-066 item 2): the per-stage rest-mass ledger (<mhd> mass_ledger) and the per-output
I/O timers (<time> io_timers) of the ml_dbg debug runs.

Per arm: the grid's change M_end(last) - M(u1)(first) split into
  Y   = sum_s w_s (M_pre(s) - gam0 M_end(s-1) - gam1 M(u1))  (the RK update's own change),
  Fout = outflow through the mesh's outer faces (Y should equal -Fout if the update conserves),
  X   = the C2P's change, by class x r_BH x removed/added,
  G   = M(u1)(n+1) - M_end(n) (between cycles: regrid, anything outside the stages),
with Y split by regrid / non-regrid cycles (regrid = a CFC 'post-regrid' line before the cycle),
and checked against the hst mass. Then the IO_TIMER statistics per output block.

    c053_ledger.py [--runs DIR] [--arms T,A,io]
"""
import argparse
import os
import re

import numpy as np

CLS = ('exc floor+flux', 'floor only', 'zone/t3', 'C2P-floored', 'other')
RB = ('<1.88', '1.88-2.92', '2.92-6', '6-12', '>=12')
NUM = r'([-+0-9.eE]+)'


def parse(fn):
    led, xb, reg, io, cyc = [], {}, set(), [], []
    pend = False
    for ln in open(fn, errors='replace'):
        if 'post-regrid' in ln:
            pend = True
        elif ln.startswith('MLEDGER '):
            d = dict(re.findall(r'(\w+)=([^ ]+)', ln))
            c = int(d['cycle'])
            row = dict(cycle=c, **{k: float(d[k]) for k in ('time', 'dt', 'Mu1', 'Mend', 'Y', 'Fout', 'Fexc',
                                                              'X', 'fofc6', 'fofc12')})
            row['regrid'] = pend
            pend = False
            led.append(row)
        elif ln.startswith('MLEDGER_X '):
            p = ln.split()
            xb[int(p[1].split('=')[1])] = np.array([float(v) for v in p[2:]])
        elif ln.startswith('IO_TIMER '):
            d = dict(re.findall(r'(\w+)=([^ ]+)', ln))
            io.append((d['block'], d['type'], d.get('id', ''), float(d['load']), float(d['write'])))
        elif ln.startswith('elapsed='):
            m = re.match(r'elapsed=(\S+) cycle=(\d+) time=(\S+) dt=(\S+)', ln)
            if m:
                cyc.append((float(m.group(1)), int(m.group(2))))
    return led, xb, io, cyc


def ledger2(fn):
    """MLEDGER2 (tau, DY): per variable sum of Y, Fout, X over the run"""
    out = {}
    for ln in open(fn, errors='replace'):
        if ln.startswith('MLEDGER2 '):
            d = dict(re.findall(r'(\w+)=([^ ]+)', ln))
            o = out.setdefault(d['var'], [0.0, 0.0, 0.0, 0])
            o[0] += float(d['Y']); o[1] += float(d['Fout']); o[2] += float(d['X']); o[3] += 1
    return out


def ledger(arm, d, out):
    fn = f'{d}/{arm}/out.log'
    if not os.path.exists(fn):
        out.append(f'== {arm}: no out.log'); return
    led, xb, io, cyc = parse(fn)
    if not led:
        out.append(f'== {arm}: no MLEDGER lines'); return
    a = {k: np.array([r[k] for r in led]) for k in led[0]}
    G = a['Mu1'][1:] - a['Mend'][:-1]
    dM = a['Mend'][-1] - a['Mu1'][0]
    reg = a['regrid'].astype(bool)
    out.append(f'== {arm}: cycles {a["cycle"][0]}-{a["cycle"][-1]} ({len(led)}), t {a["time"][0]:.3f}-'
               f'{a["time"][-1] + a["dt"][-1]:.3f}, regrid cycles {reg.sum()}')
    out.append(f'  grid dM = {dM:+.4e}  = sum Y {a["Y"].sum():+.4e} + sum X {a["X"].sum():+.4e} + sum G {G.sum():+.4e}'
               f'  (closure residual {dM - a["Y"].sum() - a["X"].sum() - G.sum():+.2e})')
    out.append(f'  Y + Fout (update non-conservation) {(a["Y"] + a["Fout"]).sum():+.4e}; Fout {a["Fout"].sum():+.2e};'
               f' Fexc (into excision-flux cells) {a["Fexc"].sum():+.4e}')
    for lab, k in (('regrid', reg), ('non-regrid', ~reg)):
        if k.any():
            y = a['Y'][k]
            out.append(f'  Y on {lab:10s} cycles: n {k.sum():4d} sum {y.sum():+.4e} mean {y.mean():+.3e} '
                       f'median {np.median(y):+.3e} min {y.min():+.3e} max {y.max():+.3e}; frac<0 {np.mean(y < 0):.2f}')
    out.append(f'  first cycle (restart, empty masks): Y {a["Y"][0]:+.3e} X {a["X"][0]:+.3e}')
    if len(G):
        out.append(f'  G (between cycles): sum {G.sum():+.3e}, max |G| {np.abs(G).max():.3e}, on regrid {G[reg[1:]].sum():+.3e}')
    out.append(f'  FOFC cells/cycle r<6: mean {a["fofc6"].mean():.1f}, 6-12: {a["fofc12"].mean():.1f}')
    # Y per M, worst cycles
    i = np.argsort(a['Y'])[:5]
    out.append('  most negative Y: ' + '; '.join(f'c{a["cycle"][j]} {a["Y"][j]:+.2e}{" R" if reg[j] else ""}' for j in i))
    if xb:
        X = np.sum([xb[c] for c in xb], axis=0).reshape(5, 5, 2)
        out.append('  X by class (removed / added) per r bin ' + ' | '.join(RB) + ':')
        for c in range(5):
            if np.any(X[c]):
                out.append(f'    {CLS[c]:15s} ' + ' | '.join(f'{X[c, r, 0]:.2e}/{X[c, r, 1]:.2e}' for r in range(5)))
    # hst check
    h = f'{d}/{arm}/cfc_tde_wd_imbh_parabolic_tracker.mhd.hst'
    if os.path.exists(h):
        hs = np.loadtxt(h)
        hs = hs[np.argsort(hs[:, 0])]
        k = (hs[:, 0] >= a['time'][0]) & (hs[:, 0] <= a['time'][-1] + a['dt'][-1] + 1e-9)
        if k.sum() >= 2:
            j0 = int(np.argmin(np.abs(a['time'] - hs[k][0, 0])))
            j1 = int(np.argmin(np.abs(a['time'] + a['dt'] - hs[k][-1, 0])))
            out.append(f'  hst mass change t {hs[k][0, 0]:.3f}-{hs[k][-1, 0]:.3f}: {hs[k][-1, 2] - hs[k][0, 2]:+.4e} vs ledger '
                       f'{a["Mend"][j1] - a["Mend"][max(j0 - 1, 0)]:+.4e}')
    if cyc:
        w = np.array(cyc)
        out.append(f'  wall {w[-1, 0] - w[0, 0]:.0f} s for {int(w[-1, 1] - w[0, 1])} cycles')
    for v, (y, f, x, n) in ledger2(fn).items():
        out.append(f'  {v}: sum Y {y:+.4e}, Fout {f:+.3e}, Y + Fout {y + f:+.4e}, X {x:+.4e} ({n} cycles;'
                   f' Y includes the source terms for tau)')
    iot(arm, io, out)


def iot(arm, io, out):
    if not io:
        return
    out.append(f'  IO_TIMER ({arm}): block type id: n, load mean, write mean / max [s]')
    keys = sorted(set((b, t, i) for b, t, i, _, _ in io))
    for key in keys:
        r = np.array([(l, w) for b, t, i, l, w in io if (b, t, i) == key])
        out.append(f'    {key[0]:8s} {key[1]:4s} {key[2]:10s}: n {len(r):3d}, load {r[:, 0].mean():.3f}, write '
                   f'{r[:, 1].mean():.3f} / {r[:, 1].max():.3f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/ml_dbg')
    ap.add_argument('--arms', default='T,A,io')
    ap.add_argument('--out', default='/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/plots/phase2_m1/c053_ledger.txt')
    a = ap.parse_args()
    out = ['R-066 item 2: rest-mass ledger and I/O timers (c053_ledger.py). Masses in M_BH units.']
    for arm in a.arms.split(','):
        ledger(arm, a.runs, out)
    print('\n'.join(out))
    with open(a.out, 'w') as f:
        f.write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
