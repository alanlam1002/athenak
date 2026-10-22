"""E3 (R-103): moving shocks in a curved background. shock_tube pgen with <problem> schwarzschild =
true (src/pgen/tests/shock_tube.cpp SetADMVariablesToSchwarzschild: x1 = r, alpha = sqrt(1-2/r),
g_11 = 1/(1-2/r), g_22 = g_33 = r^2, beta = 0, K = 0; static), so sqrt(gamma) = r^2/sqrt(1-2/r) and
d_r ln sqrt(gamma) = 2/r - 1/(r^2 (1-2/r)) ~ 0.097 at r = 20. Colliding flows (two shocks of Mach M in
the collision frame) boosted to a bulk velocity vb (relativistic addition), or a smooth uniform flow."""
import os, re, sys, math
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
tpl = open('flat/col_M1.5_coord.athinput').read()
GAM = 5.0 / 3.0
RHO, P = 1.0, 0.01
h = 1 + GAM / (GAM - 1) * P / RHO
CS = math.sqrt(GAM * P / (RHO * h))
TLIM = 1.35
X0, X1, NX = 19.0, 21.0, 512
def make(name, vl, vr, xs, var):
    s = tpl
    s = re.sub(r'(<mesh>.*?)nx1\s*=\s*\S+', lambda m: m.group(1) + f'nx1 = {NX}', s, count=1, flags=re.S)
    s = re.sub(r'x1min\s*=\s*\S+', f'x1min = {X0}', s, count=1)
    s = re.sub(r'x1max\s*=\s*\S+', f'x1max = {X1}', s, count=1)
    s = re.sub(r'(<meshblock>.*?)nx1\s*=\s*\S+', lambda m: m.group(1) + f'nx1 = {NX}', s, count=1, flags=re.S)
    s = re.sub(r'tlim\s*=\s*\S+', f'tlim = {TLIM}', s, count=1)
    s = re.sub(r'xshock\s*=\s*\S+', f'xshock = {xs}', s, count=1)
    for k, v in (('dl', RHO), ('dr', RHO), ('pl', P), ('pr', P), ('ul', vl), ('ur', vr)):
        s = re.sub(rf'^{k}\s*=\s*\S+', f'{k} = {v!r}', s, count=1, flags=re.M)
    s = s.replace('<problem>\n', '<problem>\nschwarzschild = true\n', 1)
    s = re.sub(r'(<output[0-9]>[^<]*?)dt\s*=\s*\S+', lambda m: m.group(1) + f'dt = {TLIM/10:.6g}', s, flags=re.S)
    s = re.sub(r'dual_energy_div = \S+', f'dual_energy_div = {"coord" if var == "off" else var}', s)
    if var == 'off':
        s = s.replace('dual_energy = true', 'dual_energy = false').replace('dual_energy_diag = true\n', '')
        s = re.sub(r'\n<output7>.*$', '\n', s, flags=re.S)
    open(f'{OUT}/{name}_{var}.athinput', 'w').write(s)
n = 0
for M in (1.2, 1.5, 2.0, 3.0):
    d = 2 * CS / (GAM + 1) * (M - 1 / M)
    for vb in (0.0, 0.3, -0.3):
        vl = (vb + d) / (1 + vb * d); vr = (vb - d) / (1 - vb * d)
        xs = 20.0 - vb * TLIM          # collision point ends near r = 20
        for var in ('off', 'coord', 'covariant'):
            make(f'col_M{M:g}_vb{vb:+.1f}', vl, vr, xs, var); n += 1
for vb in (0.0, 0.3, -0.3):
    for var in ('off', 'coord', 'covariant'):
        make(f'smooth_vb{vb:+.1f}', vb, vb, 20.0, var); n += 1
print(n, 'inputs; cs =', CS)
