#!/usr/bin/env python3
"""c202 (R-236 item 3): the dx 1/64 block test and the variant-A functional gate.
  H16 / H32: from c202's F16_n2.par (the P18 production deck, finest dx 1/32 at 16^3, 24 ranks): num_levels 8, refined_region2 (the star
    at t = 0) level 7, amr_level_debris 6 + variant A amr_level_debris_fine 7 at amr_density_frac_fine = amr_density_frac (1e-2: the
    ladder's setting, the whole star incl. its outer layers at dx 1/64); meshblock 16^3 / 32^3; max_nmb_per_rank 850 / 140
    (c202 measured ~54 MiB per 16^3 slot + ~2.6 GB, so <= ~49 GB per tile).  The puncture stays at level 5 (dx 1/16).
  G4: c202_gateA/G2b.par with amr_density_frac_fine 0.3 (12 ranks, 40 cycles): the fine cut must change the refinement.
    c202_mk_l64.py C202_SCALING_DIR C202_GATEA_DIR OUTDIR"""
import re, sys
sc, ga, out = sys.argv[1:4]
def sub(s, block, key, val, add=False):
    pat = re.compile(rf'(<{block}>[^<]*?^\s*{key}\s*=\s*)(\S+)', re.M | re.S)
    s, c = pat.subn(lambda m: m.group(1) + str(val), s, count=1)
    if c == 0 and add:
        s, c = re.subn(rf'(<{block}>\n)', rf'\g<1>{key} = {val}\n', s, count=1)
    assert c == 1, (block, key); return s
s0 = open(f'{sc}/F16_n2.par').read()
for nm, nb, mx in (('H16', 16, 850), ('H32', 32, 140)):
    s = s0
    for k in ('nx1', 'nx2', 'nx3'): s = sub(s, 'meshblock', k, nb)
    s = sub(s, 'mesh_refinement', 'max_nmb_per_rank', mx)
    s = sub(s, 'mesh_refinement', 'num_levels', 8)
    s = sub(s, 'refined_region2', 'level', 7)
    s = sub(s, 'problem', 'amr_level_debris', 6)
    s = sub(s, 'problem', 'amr_level_debris_fine', 7, add=True)
    s = sub(s, 'problem', 'amr_density_frac_fine', '1.0e-2', add=True)
    s = f'# c202 dx 1/64 block test {nm}: meshblock {nb}^3, 24 ranks, max_nmb_per_rank {mx}, num_levels 8, variant A (debris 6, fine 7 at 1e-2)\n' + s
    open(f'{out}/{nm}.par', 'w').write(s)
g = open(f'{ga}/G2b.par').read()
g = sub(g, 'problem', 'amr_density_frac_fine', '3.0e-1')
open(f'{out}/G4.par', 'w').write('# c202 variant-A functional gate G4: G2b with amr_density_frac_fine 0.3\n' + g)
print('ok')
