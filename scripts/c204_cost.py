#!/usr/bin/env python3
"""c204 (R-239 item 3 -> item 4 gate): the ladder's node-h with the MEASURED dx 1/64 dt (c203_chain/D64s, 2 nodes, t = 0 -> 20).
Same ARGUED model and stage table as c202_cost.py; changes, all MEASURED in c203_chain:
  - dt ratio dx 1/64 : dx 1/16 settles at 0.26 (t = 14-20; P18_r82 vs D64s user.hst medians per 2 M), not 0.48^2 = 0.23;
  - the start-up transient: D64s needs 516 cycles to t = 6 where the settled ratio needs ~150: +370 cycles charged once per 1/64 run;
  - dx 1/32 keeps 0.48 (c202 F16 vs B16, no 1/32 transient measured: argued small, F16 averaged 0.39 over t = 0-10, charged as +60 cycles);
  - the seed: 6812 blocks at t = 0, 7421 after the first regrid (cycle 25), peak 7435 = 310/rank on 2 nodes.
  c204_cost.py"""
from c202_cost import A, B, DELTA, P18, P10, pick
DT = {1: 0.48, 2: 0.26}
EXTRA = {1: 60, 2: 370}
tot = 0
for name, st in (('P18', P18), ('P10', P10)):
    for L in (1, 2):
        rows = [(N/(DT[L]**(1 if L == 2 else L)), nmb + DELTA[L]) for _, N, nmb, _ in st[:2]]
        rows[0] = (rows[0][0] + EXTRA[L], rows[0][1])
        nn, pk = pick(rows, 7435 if L == 2 else 0)
        c = sum(nc*(A + B*nb/(12*nn))*nn/3600 for nc, nb in rows)
        h = c/nn
        tot += c
        print(f'{name} dx 1/{16*2**L}: S0-S3 to t_p+50 on {nn} nodes: {sum(r[0] for r in rows):7.0f} cycles {c:5.0f} node-h {h:5.1f} wall-h (peak {pk:,.0f} = {pk/(12*nn):.0f}/rank)')
print(f'ladder total {tot:.0f} node-h')
