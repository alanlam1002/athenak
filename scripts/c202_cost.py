#!/usr/bin/env python3
"""c202 (R-236 item 4 prerequisite): updated node-h estimates for the resolution ladder and the long runs.
ARGUED model on MEASURED inputs (c202 = athenak_run/cfc/c202_scaling, c202_l64; c201 = c201_blocks):
  - wall/cycle in the PRODUCTION configuration vs blocks/rank n: w = 2.910 + 0.02643 n s (c202 scaling, 8 arms B16/F16 on 1-8 nodes,
    cycles 50-200; no production factor needed);
  - the c134 originals' stage table (cycles, nmb, finest-level blocks; c201_stepc.py / c201_pilot_cost.py);
  - extra blocks early: +2030 per one level (F16 3893-3907 vs B16 ~1870, c201) and +5570 for two levels (H16 7421 steady, c202_l64,
    variant A at the equal cut; the t = 0 seed transient peaks at 14428);
  - dt x 0.48 per level (F16 vs B16 at t ~ 10); the 1/64 start-up transient (dt 0.0104 at t = 2, ratio 0.22 to F16) is ignored
    beyond S0 (argued; see C-202);
  - GPU memory per tile: 2.6 GB + 54 MiB per max_nmb_per_rank slot (xpu-smi, c202); usable <= ~1000 slots per tile; the deck
    sets max_nmb_per_rank ~ 1.3 x the peak blocks/rank, so a run needs peak/rank <= ~770.
Node count per run: the smallest of 2/4/8/16 that holds the run's peak.  c202_cost.py"""
import math
A, B = 2.910, 0.02643
DT_L = 0.48
DELTA = {0: 0, 1: 2030, 2: 5570}
PEAK_PER_RANK = 770
P18 = [('S0-S2 0..tp-50', 3898, 1870, 200), ('tp-50..+50', 1397, 1990, 376), ('+50..+300', 1832, 2790, 1216),
       ('+300..+500', 1183, 4050, 1736), ('+500..+800', 1775, 6370, 3008), ('+800..+1000', 1316, 8760, 3808)]
P10 = [('S0-S2 0..tp-50', 3834, 1820, 216), ('tp-50..+50', 1953, 1975, 784), ('+50..+250', 3070, 3860, 2280),
       ('+250..+450', 3373, 7600, 4224), ('+450..+600', 2662, 12090, 7880)]


def blocks(stages, L, var, g):
    out = []
    for i, (lab, N, nmb, fin) in enumerate(stages):
        nb = nmb + DELTA[L]
        if var == 'B' and i >= 2 and L:
            nb += fin*(g - 1)
        out.append((N/DT_L**L, nb))
    return out


def cost(rows, nodes):
    return sum(nc*(A + B*nb/(12*nodes))*nodes/3600 for nc, nb in rows), sum(nc*(A + B*nb/(12*nodes))/3600 for nc, nb in rows)


def pick(rows, seed=0):
    peak = max(max(nb for _, nb in rows), seed)
    for nodes in (2, 4, 8, 16):
        if peak/(12*nodes) <= PEAK_PER_RANK:
            return nodes, peak
    return None, peak


print(f'w = {A} + {B} n s/cycle (production); dt x {DT_L}/level; extra blocks {DELTA}; peak <= {PEAK_PER_RANK}/rank')
for name, st, end in (('P18', P18, '+1000'), ('P10', P10, '+600')):
    for L in (0, 1, 2):
        seed = 14428 if L == 2 else 0
        r50 = blocks(st[:2], L, 'A', 1)
        n50, pk = pick(r50, seed)
        c50, h50 = cost(r50, n50)
        line = f'{name} dx 1/{16*2**L}: S0-S3 to t_p+50 on {n50} nodes: {c50:5.0f} node-h, {h50:5.0f} wall-h (peak {pk:,.0f})'
        for var, gs in (('A', (1,)), ('B', (4, 8) if L == 1 else (16, 64) if L == 2 else (1,))):
            for g in gs:
                r = blocks(st, L, var, g)
                nn, pk = pick(r, seed)
                if nn is None:
                    line += f' | end {var}x{g}: infeasible <=16 nodes (peak {pk:,.0f})'
                    continue
                c, h = cost(r, nn)
                line += f' | to t_p{end} {var}' + (f'x{g}' if var == 'B' else '') + f' on {nn}n: {c:5.0f} node-h {h:5.0f} wall-h (peak {pk:,.0f})'
        print(line)
