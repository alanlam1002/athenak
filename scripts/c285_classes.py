#!/usr/bin/env python3
"""c285 test 4 (R-271): the debris-safety check of <mhd> atm_ceiling. Per prim_3d dump of a run, each physical class's minimum
density against the ceiling threshold rho_thr = factor x dfloor, and the mass in the band (rho < rho_thr) in and next to each class.
    c285_classes.py OUT.csv RUNDIR DECK [--tmin T] [--factor 100] [--nbr 1.0]
Classes (geometric / kinematic, so membership does not depend on a density cut where avoidable):
  centre   the rho-weighted centroid of rho >= 0.01 rho_max (the remnant or the densest debris)
  core     d < 0.2 R* about the centre                       (only if rho_max >= 1e-3 rho_c, i.e. a remnant exists)
  outer    0.5 R* <= d < 1.0 R*                              (same condition)
  remnant  d < r_H, r_H = r_BH (M_H/(3 M_BH))^(1/3) iterated from M_H = mass within r_H (start 2 R*)
  stream   outside r_H, rho >= 1e-10 rho_c (the debris; a census cut, ~9 decades above rho_thr = 3.1e-19 rho_c at factor 100)
  stream_nbr  cells within --nbr M of a stream cell, not stream, not remnant: the gas around the stream, where a too-wide band
           would bite first (its minimum density is the margin that matters)
BH position x_BH = puncture_x - frame_disp(t) (cfc.hpp; the run's user.hst). Coordinate volume, as c206/c282_envelope."""
import argparse, glob, re, sys, types
import numpy as np
from scipy.spatial import cKDTree
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde/wt_tde_scripts/vis/python')
import bin_convert as bc   # noqa: E402
RC, RS, MS = 3.24e-4, 0.677, 7.47e-5
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('run'); ap.add_argument('deck')
ap.add_argument('--tmin', type=float, default=-1.0); ap.add_argument('--tmax', type=float, default=1e30); ap.add_argument('--factor', type=float, default=100.0)
ap.add_argument('--nbr', type=float, default=1.0); a = ap.parse_args()


def key(name):
    for l in open(a.deck):
        m = re.match(rf'\s*{name}\s*=\s*([-0-9.eE+]+)', l)
        if m: return float(m.group(1))
    raise KeyError(name)


dfloor = key('dfloor'); thr = a.factor*dfloor/RC          # in rho_c units
p0 = np.array([key(f'puncture_x{k}') for k in (1, 2, 3)])
fh = glob.glob(f'{a.run}/*.user.hst')[0]; hdr = open(fh).readlines()[1]; H = np.loadtxt(fh)
cfd = [int(re.search(rf'\[(\d+)\]=frame_disp_x{k}', hdr).group(1)) - 1 for k in (1, 2, 3)]
rows = []
for f in sorted(glob.glob(f'{a.run}/bin/*.prim_3d.*.bin')):
    P = bc.read_binary(f); t = P['time']
    if t < a.tmin or t > a.tmax: continue
    g = np.asarray(P['mb_geometry'], float); n = P['nx1_out_mb']
    X, Y, Z, V = [], [], [], []
    for q in range(len(g)):
        c = [np.linspace(g[q, 2*k], g[q, 2*k+1], n+1) for k in range(3)]; c = [0.5*(x[1:]+x[:-1]) for x in c]
        zz, yy, xx = np.meshgrid(c[2], c[1], c[0], indexing='ij')
        X.append(xx.ravel()); Y.append(yy.ravel()); Z.append(zz.ravel()); V.append(np.full(n**3, ((g[q, 1]-g[q, 0])/n)**3))
    X, Y, Z, V = map(np.concatenate, (X, Y, Z, V))
    rho = np.concatenate([np.asarray(P['mb_data']['dens'][q]).ravel() for q in range(len(g))]).astype(float)/RC
    m = rho*RC*V
    xb = p0 - H[np.argmin(abs(H[:, 0] - t)), cfd]
    rmax = rho.max(); d0 = rho >= 0.01*rmax
    cx, cy, cz = (np.average(w[d0], weights=m[d0]) for w in (X, Y, Z))
    d = np.sqrt((X-cx)**2 + (Y-cy)**2 + (Z-cz)**2); rbh = np.sqrt(((np.array([cx, cy, cz]) - xb)**2).sum())
    rH = 2*RS
    for _ in range(30):
        MH = m[d < rH].sum(); rn = rbh*(MH/3.0)**(1/3)
        if abs(rn - rH) < 1e-4: break
        rH = rn
    cls = {}
    if rmax >= 1e-3:
        cls['core'] = d < 0.2*RS; cls['outer'] = (d >= 0.5*RS) & (d < RS)
    cls['remnant'] = d < rH
    st = (~cls['remnant']) & (rho >= 1e-10); cls['stream'] = st
    pts = np.c_[X, Y, Z]
    if st.any():
        lo, hi = pts[st].min(0) - a.nbr, pts[st].max(0) + a.nbr
        cand = (~st) & (~cls['remnant']) & np.all((pts >= lo) & (pts <= hi), axis=1)
        dist, _ = cKDTree(pts[st]).query(pts[cand], distance_upper_bound=a.nbr)
        nb = np.zeros_like(st); nb[np.nonzero(cand)[0][np.isfinite(dist)]] = True
        cls['stream_nbr'] = nb
    band = rho < thr
    row = dict(t=t, rho_max=rmax, r_bh=rbh, r_H=rH, thr=thr, M_band_all=m[band].sum()/MS)
    for k in range(-21, 1):   # whole-domain mass per density decade [1e(k-1), 1e(k)) rho_c, for the above-band comparison
        row[f'Mdec_{k:+03d}'] = m[(rho >= 10.0**(k-1)) & (rho < 10.0**k)].sum()/MS
    for k, s in cls.items():
        row[f'{k}_n'] = int(s.sum())
        row[f'{k}_min'] = rho[s].min() if s.any() else np.nan
        row[f'{k}_min_over_thr'] = (rho[s].min()/thr) if s.any() else np.nan
        row[f'{k}_Mband'] = m[s & band].sum()/MS
        row[f'{k}_M'] = m[s].sum()/MS
    rows.append(row); print({k: (f'{v:.4g}' if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
keys = sorted({k for r in rows for k in r}, key=lambda k: (k != 't', k))
with open(a.out, 'w') as fo:
    fo.write(','.join(keys) + '\n')
    for r in rows: fo.write(','.join(f'{r.get(k, np.nan):.6e}' if not isinstance(r.get(k), int) else str(r[k]) for k in keys) + '\n')
print('wrote', a.out)
