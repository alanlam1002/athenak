#!/usr/bin/env python3
"""c202 (R-236 item 1): the strong-scaling readout. Per arm <B16|F16>_n<nodes> (12 ranks/node): from PHASE_TIMERS (every 25 cycles)
over cycles 50 -> the last row (PHASE_TIMERS_FINAL if present): wall/cycle and its split, node-h per M (nodes x wall / simulated time),
M per wall-h, nmb and blocks/rank n, the load-balance bound ceil(nmb/R)/(nmb/R); the GPU memory (xpu-smi 'Memory Used' samples on the
arm's first node, max over samples and tiles); then fits w = a + b n per resolution and combined.
    c202_scaling_ana.py RUNDIR ARM..."""
import re, sys, math
import numpy as np
d0 = sys.argv[1]; res = []
def pt(l): return {k: float(v) for k, v in re.findall(r'(\w+)=([-0-9.e+]+)', l)}
for arm in sys.argv[2:]:
    nodes = int(arm.split('_n')[1]); R = 12*nodes
    try: log = open(f'{d0}/{arm}/out.log').read().splitlines()
    except OSError: print(f'{arm}: no out.log'); continue
    rows = {int(r['cycle']): r for r in (pt(l) for l in log if l.startswith('PHASE_TIMERS'))}
    if 50 not in rows or max(rows) <= 50: print(f'{arm}: no window', sorted(rows)[-2:] if rows else ''); continue
    a, b = rows[50], rows[max(rows)]; nc = b['cycle'] - a['cycle']; w = (b['wall'] - a['wall'])/nc
    sp = {k: (b[k] - a[k])/nc for k in ('stagen', 'mg', 'hydro_etc', 'regrid', 'io')}
    nmb = np.mean([r['nmb'] for c, r in rows.items() if c >= 50]); n = nmb/R
    dt = (b['time'] - a['time']); nhM = nodes*(b['wall'] - a['wall'])/3600/dt
    mem = []
    try:
        mem = [float(x) for x in re.findall(r'[Mm]emory [Uu]sed[^\d]*([\d.]+)', open(f'{d0}/{arm}/mem.log').read())]
    except OSError: pass
    res.append((arm, nodes, n, w))
    print(f'{arm:7s} {R:3d} ranks: cycles {int(a["cycle"])}-{int(b["cycle"])} wall/cycle {w:6.3f} s (mg {sp["mg"]:.3f} = {sp["mg"]/sp["stagen"]:.2f} of stage loop; '
          f'hydro_etc {sp["hydro_etc"]:.3f} regrid {sp["regrid"]:.3f} io {sp["io"]:.3f}) | nmb {nmb:.0f} n {n:6.1f} lb bound {math.ceil(nmb/R)/(nmb/R):.3f} '
          f'| M per wall-h {dt/(b["wall"]-a["wall"])*3600:6.2f} node-h/M {nhM:.4f} | GPU mem used max {max(mem) if mem else float("nan"):.0f} MiB ({len(mem)} samples)')
for tag in ('B16', 'F16', ''):
    s = [(n, w) for arm, _, n, w in res if arm.startswith(tag)]
    if len(s) >= 2:
        bb, aa = np.polyfit([x[0] for x in s], [x[1] for x in s], 1)
        print(f'fit {tag or "all"}: w = {aa:.3f} + {bb:.5f} n  (n = blocks/rank; {len(s)} arms)')
