#!/usr/bin/env python3
"""c206 (user question 2026-11-07): pre-periapsis mass outflow from the stellar surface, original (c134) vs new (c203) pilots.
Per prim_3d dump: the star's centre = rho-weighted centroid of rho >= 0.01 rho_c; with d = |x - centre| and the BH distance:
  M(d > f R*) for f = 1.2, 1.5, 2, 3 (cells with rho > 1e3 x the atmosphere rho_atm = 1e-10 rho_c, so the atmosphere is excluded),
  as fractions of M*; mass-weighted K/K0 and radial velocity (v.d/|d|, coordinate) of the gas at d > 1.5 R*;
  and (r_t/r)^3, the tide-to-self-gravity ratio at the star's distance r from the hole.
    c206_outflow.py OUT.csv RUNDIR [--tmax T] [--every N]"""
import argparse, glob, sys, types
import numpy as np
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde/wt_tde_scripts/scripts')
import bin_convert as bc   # noqa: E402
from c160_coldmax import xbh   # noqa: E402
GAM, RC, K0, RS, MS, RT = 5/3, 3.24e-4, 1.2346626297e-2, 0.677, 7.47e-5, 16.4
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('run'); ap.add_argument('--tmax', type=float, default=1e9)
ap.add_argument('--every', type=int, default=1); a = ap.parse_args()
fs = sorted(glob.glob(f'{a.run}/bin/*.prim_3d.*.bin'))
rows = []
for f in fs[::a.every]:
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
    rho, p = cat('dens'), cat('press'); vx, vy, vz = cat('velx'), cat('vely'), cat('velz')
    m = rho*V; d0 = rho >= 0.01*RC
    cx, cy, cz = (np.average(w[d0], weights=m[d0]) for w in (X, Y, Z))
    vcx, vcy, vcz = (np.average(w[d0], weights=m[d0]) for w in (vx, vy, vz))
    dx, dy, dz = X-cx, Y-cy, Z-cz; d = np.sqrt(dx*dx+dy*dy+dz*dz)
    gas = rho > 1e3*1e-10*RC
    xb = xbh(a.run, t); r = np.sqrt((cx-xb[0])**2+(cy-xb[1])**2+(cz-xb[2])**2)
    row = dict(t=t, r_bh=r, tide=(RT/r)**3)
    for fct in (1.2, 1.5, 2.0, 3.0):
        s = gas & (d > fct*RS); row[f'M_gt{fct}'] = m[s].sum()/MS
    s = gas & (d > 1.5*RS)
    if s.any():
        K = p[s]/rho[s]**GAM/K0; vr = ((vx[s]-vcx)*dx[s]+(vy[s]-vcy)*dy[s]+(vz[s]-vcz)*dz[s])/d[s]
        row.update(K15=np.average(K, weights=m[s]), K15_med=np.median(K), vr15=np.average(vr, weights=m[s]))
    rows.append(row); print({k: (f'{v:.4g}' if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
import csv
with open(a.out, 'w') as fo:
    w = csv.DictWriter(fo, sorted({k for r in rows for k in r}, key=lambda k: list(rows[-1]).index(k) if k in rows[-1] else 99)); w.writeheader(); w.writerows(rows)
