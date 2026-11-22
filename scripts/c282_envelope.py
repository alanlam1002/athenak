#!/usr/bin/env python3
"""c282 (R-268, user question 2026-11-21): how much gas, hot and cold, surrounds the star in WENOZ W32 vs PPMX L32.
Per prim_3d dump (whole 3D grid): star centre = rho-weighted centroid of rho >= 0.01 rho_c (as c206_outflow.py); d = |x - centre|.
Mass (fraction of M*) and mass-weighted P/rho and K/K0 (K = P/rho^Gamma) of the gas in
  shells   d/R* in [1.0, 1.5), [1.5, 3), [3, 6), [6, 20)
  bands    rho/rho_c >= 1e-7 (c206's 'gas' cut: the R-268 wind), [1e-12, 1e-7), [1e-18, 1e-12), < 1e-18 (at/near the floor 3.1e-21)
plus, per shell, the 'hot' mass with P/rho > 1e-3 (any density). Coordinate volume (no sqrt(gamma)), as c206.
    c282_envelope.py OUT.csv RUNDIR [--tmax T]"""
import argparse, glob, sys, types
import numpy as np
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde/wt_tde_scripts/vis/python')
import bin_convert as bc   # noqa: E402
GAM, RC, K0, RS, MS = 5/3, 3.24e-4, 1.2346626297e-2, 0.677, 7.47e-5
SH = [(1.0, 1.5), (1.5, 3.0), (3.0, 6.0), (6.0, 20.0)]
BD = [('ge1e-7', 1e-7, np.inf), ('1e-12_1e-7', 1e-12, 1e-7), ('1e-18_1e-12', 1e-18, 1e-12), ('lt1e-18', 0.0, 1e-18)]
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('run'); ap.add_argument('--tmax', type=float, default=1e9)
a = ap.parse_args()
rows = []
for f in sorted(glob.glob(f'{a.run}/bin/*.prim_3d.*.bin')):
    P = bc.read_binary(f); t = P['time']
    if t > a.tmax: continue
    g = np.asarray(P['mb_geometry'], float); n = P['nx1_out_mb']
    X, Y, Z, V = [], [], [], []
    for q in range(len(g)):
        c = [np.linspace(g[q, 2*k], g[q, 2*k+1], n+1) for k in range(3)]; c = [0.5*(x[1:]+x[:-1]) for x in c]
        zz, yy, xx = np.meshgrid(c[2], c[1], c[0], indexing='ij')
        X.append(xx.ravel()); Y.append(yy.ravel()); Z.append(zz.ravel()); V.append(np.full(n**3, ((g[q, 1]-g[q, 0])/n)**3))
    X, Y, Z, V = map(np.concatenate, (X, Y, Z, V))
    cat = lambda k: np.concatenate([np.asarray(P['mb_data'][k][q]).ravel() for q in range(len(g))]).astype(float)
    rho, p = cat('dens'), cat('press')
    m = rho*V; d0 = rho >= 0.01*RC
    cx, cy, cz = (np.average(w[d0], weights=m[d0]) for w in (X, Y, Z))
    d = np.sqrt((X-cx)**2+(Y-cy)**2+(Z-cz)**2)/RS
    T = p/rho; K = p/rho**GAM/K0; r = rho/RC
    for s0, s1 in SH:
        ins = (d >= s0) & (d < s1)
        for name, lo, hi in BD + [('hot_Tgt1e-3', None, None)]:
            sel = ins & ((T > 1e-3) if lo is None else ((r >= lo) & (r < hi)))
            M = m[sel].sum()
            rows.append(dict(t=t, shell=f'{s0:g}-{s1:g}', band=name, M=M/MS,
                             T=(np.average(T[sel], weights=m[sel]) if M > 0 else np.nan),
                             K=(np.average(K[sel], weights=m[sel]) if M > 0 else np.nan),
                             vol=V[sel].sum()))
    print(f't={t:.2f} done', flush=True)
with open(a.out, 'w') as fo:
    fo.write('t,shell,band,M_over_Mstar,T_mw,K_over_K0_mw,volume\n')
    for q in rows:
        fo.write(f"{q['t']:.4f},{q['shell']},{q['band']},{q['M']:.6e},{q['T']:.6e},{q['K']:.6e},{q['vol']:.6e}\n")
print('wrote', a.out)
