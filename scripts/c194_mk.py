#!/usr/bin/env python3
"""c194 decks (R-227): from ../c192_b1de/De.par (C-193's D-ent):
  De2 : (i) dual_energy_rho_switch = 3.24e-10 (1e-6 rho_c); (ii) dual_energy_q_delta = 0.05; (iii) lagrangian_tags = 2
        (tag 0 = the initial 0.01-0.1 rho_c band, tag 1 = rho >= 0.1 rho_c, both on rho at t = 41.78), nscalars = 3 (tags, then
        the K tracer, last), scalar_flux_hll = false (the tags by the upwind mass flux; the tracer keeps its HLL flux through
        dual_energy_flux = hll, hlle_dyn_grmhd.hpp:292 / dyn_grmhd_fluxes.cpp:217)
  G1  : De2 (the tag gate's tags-on arm; run to cycle 700)
  G0  : De2 without tags: nscalars = 1, no lagrangian keys (the tags-off arm)"""
import re
SRC = '../c192_b1de/De.par'
src = open(SRC).read()


def setkey(s, blk, key, val, note):
    pat = re.compile(rf'^({re.escape(key)}\s*=).*$', re.M)
    body = re.search(rf'^<{blk}>\n(.*?)(?=^<)', s, re.M | re.S)
    seg = body.group(1)
    if pat.search(seg):
        seg2 = pat.sub(lambda m: f'{key} = {val}   # c194: {note}', seg, count=1)
    else:
        seg2 = seg + f'{key} = {val}   # c194: {note}\n'
    return s[:body.start(1)] + seg2 + s[body.end(1):]


def delkey(s, blk, key):
    body = re.search(rf'^<{blk}>\n(.*?)(?=^<)', s, re.M | re.S)
    seg = re.sub(rf'^{re.escape(key)}\s*=.*\n', '', body.group(1), flags=re.M)
    return s[:body.start(1)] + seg + s[body.end(1):]


de2 = src
de2 = setkey(de2, 'mhd', 'dual_energy_rho_switch', '3.24e-10', '(i) 1e-6 rho_c (was 3.2e-8 = 1e-4 rho_c)')
de2 = setkey(de2, 'mhd', 'dual_energy_q_delta', '0.05', '(ii) (was 0.1)')
de2 = setkey(de2, 'mhd', 'nscalars', '3', '(iii) tags 0, 1 + the K tracer (last)')
de2 = setkey(de2, 'mhd', 'scalar_flux_hll', 'false', '(iii) tags by the upwind mass flux; the tracer keeps HLL (dual_energy_flux = hll)')
de2 = setkey(de2, 'mhd', 'lagrangian_tags', '2', '(iii) diagnostic tags (wip/dent c194)')
de2 = setkey(de2, 'mhd', 'lagrangian_tag_rho', '3.24e-6:3.24e-5,3.24e-5:1.0e30', '(iii) tag 0 = 0.01-0.1 rho_c, tag 1 = >= 0.1 rho_c at t = 41.78')
hdr = '# c194 deck {}: ../c192_b1de/De.par (C-193 D-ent) + R-227 (i) rho_switch 1e-6 rho_c, (ii) delta 0.05, (iii) 2 Lagrangian tags{}\n'
open('De2.par', 'w').write(hdr.format('De2', '') + de2)
open('G1.par', 'w').write(hdr.format('G1', '; the tag gate, tags on') + de2)
g0 = setkey(de2, 'mhd', 'nscalars', '1', 'G0: tags off (the K tracer only)')
g0 = delkey(delkey(g0, 'mhd', 'lagrangian_tags'), 'mhd', 'lagrangian_tag_rho')
open('G0.par', 'w').write(hdr.format('G0', '; the tag gate, tags OFF (nscalars 1)') + g0)
