#!/usr/bin/env python3
"""c203 (R-239 item 4): the resolution-ladder decks from the 2-node pilot decks (c203_chain/P18_r82.par, P10_r82.par = the
full production configuration). L32_<p>: finest dx 1/32 on the star (num_levels 7, refined_region2 level 6); L64_<p>: dx 1/64
(num_levels 8, refined_region2 level 7). Variant A at the EQUAL cut: amr_level_debris stays 5 (today's finest), amr_level_debris_fine
= the new finest level, amr_density_frac_fine = amr_density_frac (bitwise to raising amr_level_debris, gate G2). The puncture keeps
amr_level_bh 5. max_nmb_per_rank for 24 ranks (C-202: <= ~770 blocks/rank peak). SEED64 (optional) shrinks the 1/64 seed box
refined_region2 to x1,x2 in [-a, a], x3 in [-b, b]. STAGES_<arm>: C0 (t_p - 50), C1 (t_p + 50).
    c203_mk_ladder.py CHAINDIR [--seed64 a,b] [--max32 N] [--max64 N]"""
import argparse, re
ap = argparse.ArgumentParser(); ap.add_argument('d'); ap.add_argument('--seed64'); ap.add_argument('--max32', type=int, default=300)
ap.add_argument('--max64', type=int, default=700); a = ap.parse_args()
TP = {'P18_r82': 491.1, 'P10_r82': 459.8}


def sub(s, pat, rep):
    s2, n = re.subn(pat, rep, s, flags=re.M); assert n == 1, pat; return s2


for p, tp in TP.items():
    base = open(f'{a.d}/{p}.par').read()
    for L, (nl, lev, mx) in {32: (7, 6, a.max32), 64: (8, 7, a.max64)}.items():
        s = base
        s = sub(s, r'^num_levels\s*=\s*6 ', f'num_levels          = {nl} ')
        s = sub(s, r'^max_nmb_per_rank\s*=\s*\d+', f'max_nmb_per_rank    = {mx}')
        i = s.index('<refined_region2>'); j = s.index('<refined_region3>'); blk = s[i:j]
        blk = sub(blk, r'^level = 5 ', f'level = {lev} ')
        if L == 64 and a.seed64:
            x, z = a.seed64.split(',')
            for k, v in (('x1min', f'-{x}'), ('x1max', x), ('x2min', f'-{x}'), ('x2max', x), ('x3min', f'-{z}'), ('x3max', z)):
                blk = sub(blk, rf'^{k} = [-.\d]+', f'{k} = {v}')
        s = s[:i] + blk + s[j:]
        s = sub(s, r'^(amr_density_frac = 1.0e-2.*)$', r'\1\namr_level_debris_fine = %d   # c203 variant A at the equal cut (R-239 item 4)\namr_density_frac_fine = 1.0e-2' % lev)
        seed = f'; seed box x1,x2 +-{a.seed64.split(",")[0]}, x3 +-{a.seed64.split(",")[1]}' if (L == 64 and a.seed64) else ''
        hdr = (f'# c203 ladder L{L}_{p} (R-239 item 4): finest dx 1/{L} on the star (num_levels {nl}, refined_region2 level {lev}{seed}),\n'
               f'# variant A at the equal cut (amr_level_debris 5, amr_level_debris_fine {lev}, frac_fine = frac 1e-2), 2 nodes, max_nmb_per_rank {mx}\n')
        open(f'{a.d}/L{L}_{p}.par', 'w').write(hdr + s)
        open(f'{a.d}/STAGES_L{L}_{p}', 'w').write(f'{tp - 50:.1f} C0\n{tp + 50:.1f} C1\n')
        print(f'L{L}_{p}: STAGES {tp - 50:.1f} C0, {tp + 50:.1f} C1')
