"""Generate the flat-space CPU dual-energy tests (C-041/de_J set) with the production keys,
for dual_energy_div = coord / covariant, plus energy-only references."""
import re, os, sys
SRC = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/dual_energy_tests_P2'
OUT = sys.argv[1]
PROD = """dual_energy_flux = hll
scalar_flux_hll = true
dual_energy_recon_rhoy = true
fofc_scalar_consistent = true
cold_recovery = true
fofc_cold_recovery = true
dual_energy_diag = true
"""
TESTS = [f'col_M{m}' for m in ('1.2', '1.5', '2', '3')] + [f'hom_c{c}' for c in ('0.05', '0.1', '0.2', '0.4')] + ['sod']
os.makedirs(OUT, exist_ok=True)
for t in TESTS:
    s = open(f'{SRC}/{t}_p3.athinput').read()
    s = s.replace('dual_energy_shock_persist = 3\n', '')
    assert 'dual_energy_shock_persist' not in s and 'dual_energy_div' not in s
    dt = re.search(r'<output6>.*?dt\s*=\s*(\S+)', s, re.S).group(1)
    diag = f"\n<output7>\nfile_type = tab\nvariable = mhd_de_diag\ndata_format = %24.17e\ndt = {dt}\n"
    for v in ('off', 'coord', 'covariant'):
        u = s
        if v == 'off':
            u = u.replace('dual_energy = true', 'dual_energy = false')
        else:
            u = u.replace('dual_energy = true\n', 'dual_energy = true\n' + PROD + f'dual_energy_div = {v}\n')
            u = u.rstrip('\n') + '\n' + diag
        open(f'{OUT}/{t}_{v}.athinput', 'w').write(u)
print(len(TESTS) * 3, 'inputs')
