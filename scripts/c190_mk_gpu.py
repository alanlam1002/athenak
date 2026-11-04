#!/usr/bin/env python3
"""c190 (R-225 item 1) GPU decks.
  Bdr  = ../c186_kfta_runs/Bbr.par (converged: rms 1e-8 + 1 V-cycle, regrid max, mg_log 1; Valencia + kcfc_bg) + Bd10's
         DE keys (dual_energy, _restart_init, _diag, _flag_exchange, the 3D dediag output): production DE (legacy)
  Bcd  = Bdr + dual_energy_mode = conditioning (eta1 3e-4, eta2 3e-3, delta 0.1: the defaults, explicit)
  Id/Ic = ../c186_kfta_runs/Iv.par (ISO_A, DE off) with production DE (legacy) / conditioning
  Ts0/Tsd/Tsc = Iv/Id/Ic with star_vel_x1 = 0 and frame_vel1 = 0: a static TOV"""
import re
def setk(s, blk, k, v, add=False):
    i = re.search(rf'^<{blk}>', s, re.M).start(); j = s.find('\n<', i + 1); j = len(s) if j < 0 else j
    seg = s[i:j]
    seg2, n = re.subn(rf'^({re.escape(k)}\s*=\s*)[^\n#]*', rf'\g<1>{v} ', seg, count=1, flags=re.M)
    if n == 0:
        assert add, (blk, k)
        seg2 = seg.replace(f'<{blk}>', f'<{blk}>\n{k} = {v}   # c190', 1)
    return s[:i] + seg2 + s[j:]
COND = [('dual_energy_mode', 'conditioning'), ('dual_energy_q_eta1', '3.0e-4'), ('dual_energy_q_eta2', '3.0e-3'),
        ('dual_energy_q_delta', '0.1')]
b = open('../c186_kfta_runs/Bbr.par').read()
for k, v in (('dual_energy', 'true'), ('dual_energy_diag', 'true'), ('dual_energy_flag_exchange', 'true')):
    b = setk(b, 'mhd', k, v)
b = setk(b, 'mhd', 'dual_energy_restart_init', 'true', add=True)
b = b.rstrip('\n') + '\n\n<output10>\nfile_type = bin\nvariable = mhd_de_diag\ndt = 12.0\nid = dediag_3d\n'
open('Bdr.par', 'w').write('# c190 deck Bdr: ../c186_kfta_runs/Bbr.par + Bd10\'s DE keys (production DE, legacy)\n' + b)
c = b
for k, v in COND:
    c = setk(c, 'mhd', k, v, add=True)
open('Bcd.par', 'w').write('# c190 deck Bcd: Bdr + dual_energy_mode = conditioning (D-ent)\n' + c)
r = setk(c, 'output4', 'dcycle', '100')
open('Bcr.par', 'w').write('# c190 deck Bcr: Bcd with restarts every 100 cycles (the restart gate)\n' + r)
iv = open('../c186_kfta_runs/Iv.par').read()
idd = setk(iv, 'mhd', 'dual_energy', 'true')
idd = setk(idd, 'mhd', 'dual_energy_diag', 'true')
ic = idd
for k, v in COND:
    ic = setk(ic, 'mhd', k, v, add=True)
open('Id.par', 'w').write('# c190 deck Id: ../c186_kfta_runs/Iv.par (ISO_A) with dual_energy = true (production keys, legacy)\n' + idd)
open('Ic.par', 'w').write('# c190 deck Ic: Id + dual_energy_mode = conditioning\n' + ic)
for nm, s in (('Ts0', iv), ('Tsd', idd), ('Tsc', ic)):
    s = setk(s, 'problem', 'star_vel_x1', '0.0')
    s = setk(s, 'cfc', 'frame_vel1', '0.0') if re.search(r'^frame_vel1', s[re.search(r'^<cfc>', s, re.M).start():], re.M) else s
    open(f'{nm}.par', 'w').write(f'# c190 deck {nm}: static TOV = ISO_A with star_vel_x1 = 0, frame_vel1 = 0\n' + s)
print('ok')
