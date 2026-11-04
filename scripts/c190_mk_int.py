#!/usr/bin/env python3
"""c190 (R-225 item 2) D-int decks: every conditioning deck X_cnd -> X_int with dual_energy_variable = eint, dual_energy_flux = mass,
dual_energy_recon_rhoy = false, scalar_flux_hll = false (the partner moves with the mass flux). GPU: Bcr/Ic/Tsc -> Bir/Ii/Tsi;
CPU: cpu/{flat,cold,build}/*_cnd -> *_int, cpu/aside_cold_v1 ghost pair -> cpu/ghost_int/, cpu/cpuP18c.par -> cpu/cpuP18i.par."""
import glob, re, os
def toint(s):
    for k, v in (('dual_energy_flux', 'mass'), ('dual_energy_recon_rhoy', 'false'), ('scalar_flux_hll', 'false')):
        if re.search(rf'^{k}\s*=', s, re.M):
            s = re.sub(rf'^({k}\s*=\s*)[^\n#]*', rf'\g<1>{v} ', s, count=1, flags=re.M)
        else:
            s = s.replace('dual_energy_mode = conditioning', f'dual_energy_mode = conditioning\n{k} = {v}', 1)
    assert s.count('dual_energy_mode = conditioning') >= 1
    s = re.sub(r'^(dual_energy_mode\s*=\s*conditioning[^\n]*)$', r'\1\ndual_energy_variable = eint   # c190 D-int', s, count=1, flags=re.M)
    s = re.sub(r'^dual_energy_variable\s*=\s*k_1g[^\n]*\n', '', s, flags=re.M)
    assert len(re.findall(r'^dual_energy_variable\s*=', s, re.M)) == 1
    return s
n = 0
for a, b in (('Bcr.par', 'Bir.par'), ('Ic.par', 'Ii.par'), ('Tsc.par', 'Tsi.par'), ('cpu/cpuP18c.par', 'cpu/cpuP18i.par')):
    open(b, 'w').write(f'# c190 D-int deck: {a} with dual_energy_variable = eint (mk190i.py)\n' + toint(open(a).read())); n += 1
for f in glob.glob('cpu/flat/*_cnd.athinput') + glob.glob('cpu/cold/*_cnd.athinput') + glob.glob('cpu/build/*_cnd.athinput'):
    open(f.replace('_cnd.athinput', '_int.athinput'), 'w').write(toint(open(f).read())); n += 1
os.makedirs('cpu/ghost_int', exist_ok=True)
for f in glob.glob('cpu/aside_cold_v1/cold/ghost_mb*_cnd.athinput'):
    open('cpu/ghost_int/' + os.path.basename(f).replace('_cnd.', '_int.'), 'w').write(toint(open(f).read())); n += 1
print(n, 'decks')
