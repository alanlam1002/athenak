#!/usr/bin/env python3
"""c198 (C-196 checks; bin dumps are single precision): unfixed vs fixed multipole Allreduce, run pairs A B (same deck, same restart).
  hst: rows compared bitwise; first differing row; at the last common row max relative difference per column.
  adm_3d (same cycle, matched MeshBlock by geometry): per ADM variable max |dB - dA| and its ratio to max |A| over
    all blocks, over the blocks on the domain's outer faces, and over the finest-level blocks (the star / hole).
  prim_3d at common times: rho-weighted K/K0 - 1 of rho >= 0.01 rho_c and 0.1 rho_c in A and B; max |drho|/rho there.
    c198_mp_cmp.py RUN_A RUN_B"""
import glob, os, sys, types
import numpy as np
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
import bin_convert as bc   # noqa: E402
a, b = sys.argv[1], sys.argv[2]
GAM = 5.0/3.0


def rows(d, k):
    f = glob.glob(f'{d}/*.{k}.hst')
    return [l.split() for l in open(f[0]) if not l.startswith('#')] if f else []


for k in ('mhd', 'user'):
    A, B = rows(a, k), rows(b, k); n = min(len(A), len(B))
    bad = [i for i in range(n) if A[i] != B[i]]
    s = f'hst {k}: rows {len(A)}/{len(B)}, identical {n - len(bad)}/{n}'
    if bad:
        x, y = np.array(A[n-1], float), np.array(B[n-1], float)
        rel = np.abs(y - x)/np.maximum(np.abs(x), 1e-300)
        j = np.argsort(rel)[::-1][:4]
        s += f'; first diff row {bad[0]} (t={A[bad[0]][0]}); last row max rel diff cols ' + ', '.join(f'{c+1}:{rel[c]:.2e}' for c in j)
    print(s)


def blocks(D):
    g = np.asarray(D['mb_geometry'], float)
    return {tuple(np.round(r, 9)): i for i, r in enumerate(g)}, g


def cmp_dump(fa, fb):
    A, B = bc.read_binary(fa), bc.read_binary(fb)
    ka, ga = blocks(A); kb, gb = blocks(B)
    common = [k for k in ka if k in kb]
    lo = ga[:, [0, 2, 4]].min(0); hi = ga[:, [1, 3, 5]].max(0)
    h = ga[:, 1] - ga[:, 0]; hmin = h.min()
    print(f"  t={A['time']:.4f}/{B['time']:.4f} blocks {len(ka)}/{len(kb)} common {len(common)}")
    for v in A['var_names']:
        if v not in B['var_names']:
            continue
        out = {}
        for nm in ('all', 'outer', 'finest'):
            dm = am = 0.0
            for k in common:
                i, j = ka[k], kb[k]; G = ga[i]
                if nm == 'outer' and not (np.any(np.isclose(G[[0, 2, 4]], lo)) or np.any(np.isclose(G[[1, 3, 5]], hi))):
                    continue
                if nm == 'finest' and not np.isclose(G[1] - G[0], hmin):
                    continue
                x = np.asarray(A['mb_data'][v][i], float); y = np.asarray(B['mb_data'][v][j], float)
                dm = max(dm, np.abs(y - x).max()); am = max(am, np.abs(x).max())
            out[nm] = (dm, dm/am if am > 0 else np.nan)
        print(f"    {v:10s} " + '  '.join(f'{nm} {out[nm][0]:.2e} ({out[nm][1]:.1e})' for nm in out))


for kind in ('adm_3d', 'prim_3d'):
    # matched by output number (the arms' times drift apart once dt differs)
    num = lambda f: f.split('.')[-2]
    ma = {num(f): f for f in glob.glob(f'{a}/bin/*.{kind}.*.bin')}
    mb = {num(f): f for f in glob.glob(f'{b}/bin/*.{kind}.*.bin')}
    ts = sorted(set(ma) & set(mb))
    print(f'{kind}: common output numbers {ts}')
    if kind == 'adm_3d':
        for t in ts:
            cmp_dump(ma[t], mb[t])
    else:
        for t in ts[-2:]:
            r = []
            for f in (ma[t], mb[t]):
                P = bc.read_binary(f); g = np.asarray(P['mb_geometry'], float); n = P['nx1_out_mb']
                cat = lambda k: np.concatenate([np.asarray(P['mb_data'][k][q]).ravel() for q in range(len(g))]).astype(float)
                rho, p = cat('dens'), cat('press'); m = rho*np.repeat(((g[:, 1] - g[:, 0])/n)**3, n**3)
                rc, k0 = 3.24e-4, 1.2346626297e-2; K = p/rho**GAM/k0 - 1
                r.append([(m[s]*K[s]).sum()/m[s].sum() for s in (rho >= 0.01*rc, rho >= 0.1*rc)] + [rho.max()/rc])
            print(f'  #{t} t={bc.read_binary(ma[t])["time"]:.4f}/{bc.read_binary(mb[t])["time"]:.4f}: A K01 {r[0][0]:+.6e} K1 {r[0][1]:+.6e} rmax {r[0][2]:.8f} | B K01 {r[1][0]:+.6e} K1 {r[1][1]:+.6e} rmax {r[1][2]:.8f}')
