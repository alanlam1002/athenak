#!/usr/bin/env python3
"""c201 Step C input: the measured cost of a pilot chain by stage, from its out.log ('elapsed=' lines; segments detected
by elapsed decreasing; a repeated cycle keeps the later segment). Wall per stage = sum of within-segment increments.
    c201_pilot_cost.py OUTLOG t_p  [bounds relative to t_p, default -50 50 300 500 800 1000]"""
import re, sys
import numpy as np
f, tp = sys.argv[1], float(sys.argv[2])
rel = [float(x) for x in sys.argv[3:]] or [-50, 50, 300, 500, 800, 1000]
seg, last, rows = 0, None, {}
pt = {}
for l in open(f):
    m = re.match(r'elapsed=([-0-9.e+]+) cycle=(\d+) time=([-0-9.e+]+) dt=([-0-9.e+]+)', l)
    if m:
        e, c, t, dt = float(m[1]), int(m[2]), float(m[3]), float(m[4])
        if last is not None and e < last: seg += 1
        last = e; rows[c] = (seg, e, t, dt)
    elif l.startswith('PHASE_TIMERS '):
        d = {k: float(v) for k, v in re.findall(r'(\w+)=([-0-9.e+]+)', l)}
        pt[int(d['cycle'])] = d
cs = np.array(sorted(rows)); sg = np.array([rows[c][0] for c in cs]); el = np.array([rows[c][1] for c in cs])
tt = np.array([rows[c][2] for c in cs]); dts = np.array([rows[c][3] for c in cs])
inc = np.where((sg[1:] == sg[:-1]) & (cs[1:] == cs[:-1] + 1), el[1:] - el[:-1], np.nan)
w_med = np.nanmedian(inc)
inc = np.where(np.isnan(inc), w_med, inc)          # a cycle across a segment boundary: the median (argued)
b = [0.0] + [tp + r for r in rel]
print(f'{f}: t_p {tp}; {len(cs)} cycles to t {tt[-1]:.1f}; {sg[-1]+1} segments; total wall {inc.sum()/3600:.2f} h on 8 nodes = {inc.sum()/3600*8:.0f} node-h')
for lo, hi in zip(b[:-1], b[1:]):
    s = (tt[1:] > lo) & (tt[1:] <= hi)
    if not s.any(): continue
    w = inc[s].sum(); nc = s.sum(); c0, c1 = cs[1:][s][0], cs[1:][s][-1]
    nm = [d['nmb'] for c, d in pt.items() if c0 <= c <= c1]
    mg = [(pt[c]['mg'] - pt[c - 100]['mg'])/(pt[c]['stagen'] - pt[c - 100]['stagen']) for c in pt if c0 <= c - 100 and c <= c1 and c - 100 in pt]
    print(f'  t {lo:7.1f}-{hi:7.1f} (t_p{lo-tp:+.0f}..{hi-tp:+.0f}): cycles {nc:6d} wall {w/3600:6.2f} h ({w/3600*8:5.0f} node-h) '
          f'{w/nc:5.2f} s/cycle  dt {np.median(dts[1:][s]):.4f}  node-h/M {w/3600*8/(hi-lo):.4f}  nmb {min(nm) if nm else 0:.0f}-{max(nm) if nm else 0:.0f}'
          f'  mg share {np.median(mg) if mg else float("nan"):.2f}')
