#!/usr/bin/env python3
"""c198 (user 2026-11-05, items 2-3): the production pilots P18_r82 / P10_r82 rerun with the full new configuration.
Decks = athenak inputs/dyn_grmhd/tde_{P18,P10}_prod.athinput (kcfc_bg, the converged metric rms 1e-8 + mg_min_vcycles 1,
the regrid max norm; = the c134 pilot decks plus those) + D-ent as the production DE (C-195: dual_energy_mode conditioning,
q_eta1 3e-4, q_eta2 3e-3, q_delta 0.05, rho_switch 3.24e-10 = 1e-6 rho_c) + a 3D de_diag dump with the 3D prims (dt 12);
tlim = the original pilots' end (P18 t_p + 1000, P10 t_p + 600); stages via TSTOP/STAGES (tp198.sh).
    c198_mk_pilots.py SRCTREE OUTDIR KMAX"""
import re, sys
src, out, kmax = sys.argv[1], sys.argv[2], sys.argv[3]
DENT = ('dual_energy_mode = conditioning   # c198: D-ent, the production DE (user 2026-11-05)\n'
        'dual_energy_q_eta1 = 3.0e-4\ndual_energy_q_eta2 = 3.0e-3\ndual_energy_q_delta = 0.05   # C-195\n')
for nm, tl in (('P18', 1491.1), ('P10', 1059.8)):
    s = open(f'{src}/inputs/dyn_grmhd/tde_{nm}_prod.athinput').read()
    s, c = re.subn(r'(?m)^dual_energy = true.*$', lambda m: m.group(0) + '\n' + DENT.rstrip('\n'), s, count=1); assert c == 1
    s, c = re.subn(r'(?m)^dual_energy_rho_switch\s*=.*$', 'dual_energy_rho_switch = 3.24e-10   # c198: 1e-6 rho_c (C-194/C-195)', s); assert c == 1
    s, c = re.subn(r'(?m)^dual_energy_kmax\s*=.*$', f'dual_energy_kmax = {kmax}   # c198', s); assert c == 1
    s, c = re.subn(r'(?m)^tlim\s*=.*$', f'tlim = {tl}   # c198: the c134 pilot end; stages by TSTOP', s); assert c == 1
    assert 'mhd_de_diag' in s and '<output10>' not in s
    s = s.rstrip('\n') + '\n\n<output10>\nfile_type = bin\nvariable = mhd_de_diag\nid = dediag_3d\ndt = 12.0   # c198: with prim_3d\n'
    open(f'{out}/{nm}_r82.par', 'w').write(s)
print('ok')
