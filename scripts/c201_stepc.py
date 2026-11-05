#!/usr/bin/env python3
"""c201 Step C (R-234 item 4): node-h estimates for the science runs at finest dx 1/32 (L=1) and 1/64 (L=2) on the star, 16^3 blocks,
production configuration (C-198/C-199: rms 1e-8 + one forced V-cycle, D-ent).  ARGUED model built on MEASURED inputs:
  - the c134 originals' cost by stage (c201_pilot_cost.py: cycles N_s, s/cycle, nmb), 8 nodes = 96 ranks;
  - wall/cycle vs blocks/rank from those stages: w_orig(n) = a + b n (least squares);
  - the production factor f_prod = B16 / c134 wall/cycle over the same cycles 100-200 (3.585 / 2.232);
  - one level finer (F16, measured): dt x 0.48, +2030 blocks early (3900 vs 1870);  two levels finer: +~6600 (argued: the 4 M star
    cube at width 0.25 = 4096, its nesting shell ~1000, minus F16's 512);
  - late stages: variant A = only the star/remnant goes finer (blocks + the same Delta; NEEDS a second density cut, not in the code),
    variant B = all gas above amr_density_frac goes finer (the current tracker): the finest-level debris blocks x4 (sheet) to x8 (volume) per level.
    c201_stepc.py"""
import numpy as np
F_PROD, DT_L = 3.585/2.232, 0.48
DELTA = {0: 0, 1: 2030, 2: 6600}
# stage: (label, t-range rel t_p, cycles, s/cycle, nmb mean, finest-level blocks at stage end)  -- measured, c134 originals
P18 = [('S0-S2 0..tp-50', 3898, 2.14, 1870, 200), ('tp-50..+50', 1397, 1.96, 1990, 376), ('+50..+300', 1832, 2.59, 2790, 1216),
       ('+300..+500', 1183, 3.02, 4050, 1736), ('+500..+800', 1775, 3.41, 6370, 3008), ('+800..+1000', 1316, 4.05, 8760, 3808)]
P10 = [('S0-S2 0..tp-50', 3834, 2.10, 1820, 216), ('tp-50..+50', 1953, 1.85, 1975, 784), ('+50..+250', 3070, 2.65, 3860, 2280),
       ('+250..+450', 3373, 3.54, 7600, 4224), ('+450..+600', 2662, 4.65, 12090, 7880)]
n = np.array([s[3]/96 for s in P18 + P10]); w = np.array([s[2] for s in P18 + P10])
b, a = np.polyfit(n, w, 1)
print(f'w_orig(n) = {a:.2f} + {b:.4f} n  (n = blocks/rank; fit to the 11 c134 stages, rms resid {np.std(w - a - b*n):.2f} s); f_prod {F_PROD:.2f}')
def cost(stages, L, var, nodes=8, upto=None):
    out, tot = [], 0.0
    for i, (lab, N, ws, nmb, fin) in enumerate(stages[:upto]):
        early = i < 2
        if var == 'B' and not early and L:
            grow = {1: (4, 8), 2: (16, 64)}[L]
            nb = [nmb + DELTA[L] + fin*(g - 1) for g in grow]
        else:
            nb = [nmb + DELTA[L]]*2
        nc = N/DT_L**L
        nh = [nc*(a + b*x/(12*nodes))*F_PROD*nodes/3600 for x in nb]
        out.append((lab, nc, nb, nh)); tot += np.array(nh)
    return out, tot
for name, st, tp_end in (('P18', P18, '+1000'), ('P10', P10, '+600')):
    for L in (0, 1, 2):
        _, t50 = cost(st, L, 'A', upto=2)
        _, tA = cost(st, L, 'A'); rowsB, tB = cost(st, L, 'B')
        maxB = max(r[2][1] for r in rowsB)
        print(f'{name} dx 1/{16*2**L}: S0-S3 to t_p+50: {t50[0]:6.0f} node-h | to t_p{tp_end}: A {tA[0]:6.0f} node-h, '
              f'B {tB[0]:6.0f}-{tB[1]:6.0f} node-h (max blocks {maxB:,.0f} = {maxB/96:.0f}/rank on 8 nodes)'
              f' | cycles {sum(r[1] for r in rowsB):,.0f}')
