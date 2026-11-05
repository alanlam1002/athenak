#!/usr/bin/env python3
"""c201 (R-229 Step A / R-234 item 2): the mesh-block-size study on the production P18 deck (c198_pilots/P18_r82.par =
inputs/dyn_grmhd/tde_P18_prod.athinput + D-ent + kmax 1000 + 3D de_diag), fresh start, nlim 300, 96 ranks:
  B16 / B32 / B64: <meshblock> nx 16 / 32 / 64, the same root grid (128^3, dx 2) and levels (6: finest dx 1/16);
  F16 / F32: one level finer around the star (num_levels 7, refined_region2 (the star) and amr_level_debris 6 -> finest dx 1/32
    on the star; the puncture stays at level 5).
  max_nmb_per_rank sized to the block volume (Kokkos pools scale with it: c198, 2000 x 16^3 blocks exhausted a tile).
    c201_mk_blocks.py SRC_DECK OUTDIR"""
import re, sys
src, out = sys.argv[1], sys.argv[2]
s0 = open(src).read()
def sub(s, block, key, val):
    pat = re.compile(rf'(<{block}>[^<]*?^\s*{key}\s*=\s*)(\S+)', re.M | re.S)
    s, c = pat.subn(lambda m: m.group(1) + str(val), s, count=1); assert c == 1, (block, key)
    return s
ARMS = {'B16': (16, 400, False), 'B32': (32, 80, False), 'B64': (64, 20, False), 'F16': (16, 600, True), 'F32': (32, 120, True)}
for nm, (nb, mx, fine) in ARMS.items():
    s = s0
    for k in ('nx1', 'nx2', 'nx3'):
        s = sub(s, 'meshblock', k, nb)
    s = sub(s, 'mesh_refinement', 'max_nmb_per_rank', mx)
    s = sub(s, 'time', 'nlim', 200)
    if fine:
        s = sub(s, 'mesh_refinement', 'num_levels', 7)
        s = sub(s, 'refined_region2', 'level', 6)
        s = sub(s, 'problem', 'amr_level_debris', 6)
    s = f'# c201 block-size study arm {nm}: meshblock {nb}^3, max_nmb_per_rank {mx}' + (', one level finer on the star (finest dx 1/32)' if fine else '') + '\n' + s
    open(f'{out}/{nm}.par', 'w').write(s)
print('ok')
