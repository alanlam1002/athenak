#!/usr/bin/env python3
"""c198 (user 2026-11-05, item 5): the 1D check of a higher dual_energy_kmax. From C-190's conditioning decks
(c190_dent/cpu/{flat,cold,build}/*_cnd.athinput), with the production D-ent settings (q_delta 0.05):
  k1 / k100 / k1000 / k10000: dual_energy_kmax = 1, 100, 1e3, 1e4 (k_1g: Y = (K/kmax)^(1/gamma), so a larger kmax only
    rescales Y by kmax^(-0.6): the check is that the solution is unchanged where the ceiling never binds, and what
    precision is lost at small Y);
  cold/*_kc10: kmax = 10 P0 (= 10 K0, rho = 1 upstream), a ceiling that binds in the shocked slab (B.1's 81 K0 analogue).
    c198_mk_kmax.py OUTDIR"""
import glob, os, re, sys
SRC = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/c190_dent/cpu'
out = sys.argv[1]
n = 0
for f in sorted(glob.glob(f'{SRC}/flat/*_cnd.athinput') + glob.glob(f'{SRC}/cold/cold_*_cnd.athinput') +
                glob.glob(f'{SRC}/build/*_cnd.athinput')):
    grp = f.split('/')[-2]; base = os.path.basename(f)[:-len('_cnd.athinput')]
    s = open(f).read()
    s, c = re.subn(r'(?m)^dual_energy_q_delta\s*=\s*\S+', 'dual_energy_q_delta = 0.05', s); assert c == 1, f
    os.makedirs(f'{out}/{grp}', exist_ok=True)
    kms = [('k1', '1.0'), ('k100', '100.0'), ('k1000', '1000.0'), ('k10000', '10000.0')]
    if grp == 'cold':
        P0 = float(re.search(r'_P([\d.e+-]+)_M', base).group(1)); kms.append(('kc10', repr(10.0*P0)))
    for tag, km in kms:
        t, c = re.subn(r'(?m)^dual_energy_kmax\s*=\s*\S+', f'dual_energy_kmax = {km}', s); assert c == 1, f
        open(f'{out}/{grp}/{base}_{tag}.athinput', 'w').write(t); n += 1
print(n, 'decks')
