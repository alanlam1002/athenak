"""R-110 / R-117 Q3.2: the flat-space moving-shock companions of the 2 x 2. C-097 E3's flat-metric
colliding flows (shock_tube pgen, schwarzschild = false, x in [19, 21]): two Mach-M shocks in the
collision frame, boosted to a bulk velocity vb (relativistic addition), the collision point ending
near x = 20. M in {1.2, 2}, vb in {0, 0.1, 0.2, 0.3}, nx in {256, 512, 1024}; dual energy off (the
R-110 spec: the base scheme's boost-dependent shock heating) and on (production keys, covariant).
Templates: athenak_run/cfc/c097_e3/flatbulk/col_M1.2_vb+0.0_{off,covariant}.athinput.

    c103_e3b_mk.py OUTDIR
"""
import math
import os
import re
import sys

OUT = sys.argv[1]
TPL = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/c097_e3/flatbulk'
os.makedirs(OUT, exist_ok=True)
GAM = 5.0 / 3.0
RHO, P = 1.0, 0.01
h = 1 + GAM / (GAM - 1) * P / RHO
CS = math.sqrt(GAM * P / (RHO * h))
TLIM = 1.35


def make(name, var, nx, vl, vr, xs):
    s = open(f'{TPL}/col_M1.2_vb+0.0_{var}.athinput').read()
    s = re.sub(r'(<mesh>.*?)nx1\s*=\s*\S+', lambda m: m.group(1) + f'nx1 = {nx}', s, count=1, flags=re.S)
    s = re.sub(r'(<meshblock>.*?)nx1\s*=\s*\S+', lambda m: m.group(1) + f'nx1 = {nx}', s, count=1,
               flags=re.S)
    s = re.sub(r'xshock\s*=\s*\S+', f'xshock = {xs!r}', s, count=1)
    for k, v in (('ul', vl), ('ur', vr)):
        s = re.sub(rf'^{k}\s*=\s*\S+', f'{k} = {v!r}', s, count=1, flags=re.M)
    assert 'schwarzschild = false' in s and f'tlim = {TLIM}' in s
    open(f'{OUT}/{name}.athinput', 'w').write(s)


n = 0
for M in (1.2, 2.0):
    d = 2 * CS / (GAM + 1) * (M - 1 / M)          # each side's speed in the collision frame
    for vb in (0.0, 0.1, 0.2, 0.3):
        vl = (vb + d) / (1 + vb * d)
        vr = (vb - d) / (1 - vb * d)
        xs = 20.0 - vb * TLIM
        for nx in (256, 512, 1024):
            for var in ('off', 'covariant'):
                make(f'col_M{M:g}_vb{vb:.1f}_n{nx}_{var}', var, nx, vl, vr, xs)
                n += 1
print(n, 'inputs; cs =', CS)
