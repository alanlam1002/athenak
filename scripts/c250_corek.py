#!/usr/bin/env python3
"""c250 (R-249 item 2): mass-weighted K/K0 in the core (r < 0.2 R*, c247_coldcore/c160 definition), 0.2-0.5 R* and 0.5-0.8 R*
(rho >= 0.01 rho_c) about the rho-centroid of rho >= 0.01 rho_c, from prim_3d alone (works with dual_energy off).
    c250_corek.py RUNDIR OUT.txt"""
import sys, types, glob
import numpy as np
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
import bin_convert as bc
GAM, RC, K0, RS = 5/3, 3.24e-4, 1.2346626297e-2, 0.677
run, out = sys.argv[1], sys.argv[2]
rows = []
for f in sorted(glob.glob(f'{run}/bin/*.prim_3d.0*.bin')):
    P = bc.read_binary(f); t = P['time']; g = np.asarray(P['mb_geometry'], float); n = P['nx1_out_mb']
    X, Y, Z, V = [], [], [], []
    for q in range(len(g)):
        c = [np.linspace(g[q, 2*k], g[q, 2*k+1], n+1) for k in range(3)]; c = [0.5*(x[1:]+x[:-1]) for x in c]
        zz, yy, xx = np.meshgrid(c[2], c[1], c[0], indexing='ij')
        X.append(xx.ravel()); Y.append(yy.ravel()); Z.append(zz.ravel()); V.append(np.full(n**3, ((g[q, 1]-g[q, 0])/n)**3))
    X, Y, Z, V = map(np.concatenate, (X, Y, Z, V))
    cat = lambda k: np.concatenate([np.asarray(P['mb_data'][k][q]).ravel() for q in range(len(g))]).astype(float)
    rho, p = cat('dens'), cat('press'); m = rho*V; d0 = rho >= 0.01*RC
    cx, cy, cz = (np.average(w[d0], weights=m[d0]) for w in (X, Y, Z))
    rr = np.sqrt((X-cx)**2+(Y-cy)**2+(Z-cz)**2)/RS; K = p/rho**GAM/K0
    r = [t]
    for s in (rr < 0.2, d0 & (rr >= 0.2) & (rr < 0.5), d0 & (rr >= 0.5) & (rr < 0.8)):
        r.append(np.average(K[s], weights=m[s]))
    rows.append(r); print(' '.join(f'{x:.5f}' for x in r), flush=True)
np.savetxt(out, rows, header='t core s25 s58', fmt='%.6f')
