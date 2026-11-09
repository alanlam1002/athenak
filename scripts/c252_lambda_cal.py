#!/usr/bin/env python3
"""c252 (R-252 item 1a, user-approved 2026-11-09): offline calibration of the relative frame-tracking loss criterion (R-250).
Per 3D dump (prim_3d + prim_3d_peri, time-sorted), mimicking the controller:
  target p = the previous dump's tracked centroid (first dump: rho-centroid of rho >= 0.01 rho_c);
  rho_max,local = the peak within radius R of p; tracked set = cells within R with rho > f_rel rho_max,local (f_rel 0.3);
  'no target' if rho_max,local <= 1e-4 rho_c;
  R = max(3 M, 2 r_enc), r_enc = the radius about the tracked centroid enclosing all of the tracked mass; iterated
  (R = 3 -> set -> centroid, r_enc -> R; NIT passes, p moves to the centroid each pass);
  rho_bar = M_trk/V_trk (cell volumes), rho_tid = 3 M_BH/(4 pi r^3), r = |centroid - x_BH| (M_BH = 1; x_BH from user.hst
  frame_disp, c160_coldmax.xbh); Lambda = rho_bar/rho_tid.
Columns: t, Lambda, M_trk/M*, V_trk, rho_bar/rho_c, rho_max_loc/rho_c, R, r_enc, r, the centroid x,y,z and its offset |c| from the
grid origin, n_trk, ok (0 = no target).
    c252_lambda_cal.py OUT.csv RUNDIR [--frel 0.3] [--nit 3]"""
import argparse, glob, sys, types
import numpy as np
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde/wt_tde_scripts/scripts')
import bin_convert as bc   # noqa: E402
from c160_coldmax import xbh   # noqa: E402

RC, MS, FLOOR = 3.24e-4, 7.47e-5, 1e-4
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('run')
ap.add_argument('--frel', type=float, default=0.3); ap.add_argument('--nit', type=int, default=3); a = ap.parse_args()
fs = glob.glob(f'{a.run}/bin/*.prim_3d.0*.bin') + glob.glob(f'{a.run}/bin/*.prim_3d_peri.0*.bin')
ft = sorted(((bc.read_binary(f)['time'], f) for f in fs))
seen = set(); ft = [x for x in ft if not (round(x[0], 4) in seen or seen.add(round(x[0], 4)))]
p, rows = None, []
for t, f in ft:
    P = bc.read_binary(f); g = np.asarray(P['mb_geometry'], float); n = P['nx1_out_mb']
    dens = P['mb_data']['dens']
    if p is None:   # first dump: rho-centroid of rho >= 0.01 rho_c over the whole mesh
        s = w = 0.0
        for q in range(len(g)):
            c = [0.5*(np.linspace(g[q, 2*k], g[q, 2*k+1], n+1)[1:] + np.linspace(g[q, 2*k], g[q, 2*k+1], n+1)[:-1]) for k in range(3)]
            zz, yy, xx = np.meshgrid(c[2], c[1], c[0], indexing='ij'); r_ = np.asarray(dens[q], float)
            m = r_*((g[q, 1]-g[q, 0])/n)**3; sel = r_ >= 0.01*RC
            s = s + np.array([(m*xx)[sel].sum(), (m*yy)[sel].sum(), (m*zz)[sel].sum()]); w += m[sel].sum()
        p = s/w
    # cells of blocks within 40 M of p (R stays well below that)
    near = [q for q in range(len(g)) if np.all(np.maximum(0, np.maximum(g[q, 0::2] - p, p - g[q, 1::2])) ** 2 @ np.ones(3) < 40.0**2)]
    X, Y, Z, V, D = [], [], [], [], []
    for q in near:
        c = [np.linspace(g[q, 2*k], g[q, 2*k+1], n+1) for k in range(3)]; c = [0.5*(x[1:]+x[:-1]) for x in c]
        zz, yy, xx = np.meshgrid(c[2], c[1], c[0], indexing='ij')
        X.append(xx.ravel()); Y.append(yy.ravel()); Z.append(zz.ravel())
        V.append(np.full(n**3, ((g[q, 1]-g[q, 0])/n)**3)); D.append(np.asarray(dens[q], float).ravel())
    X, Y, Z, V, D = map(np.concatenate, (X, Y, Z, V, D)); M = D*V
    xb = xbh(a.run, t)
    R, ok, row = 3.0, 1, None
    for _ in range(a.nit):
        d2 = (X-p[0])**2 + (Y-p[1])**2 + (Z-p[2])**2; inR = d2 < R*R
        rmx = D[inR].max() if inR.any() else 0.0
        if rmx <= FLOOR*RC:
            ok = 0; break
        trk = inR & (D > a.frel*rmx)
        mt = M[trk].sum(); cc = np.array([(M*X)[trk].sum(), (M*Y)[trk].sum(), (M*Z)[trk].sum()])/mt
        renc = np.sqrt(((X[trk]-cc[0])**2 + (Y[trk]-cc[1])**2 + (Z[trk]-cc[2])**2).max())
        vt = V[trk].sum(); r = np.linalg.norm(cc - xb); rbar = mt/vt; rtid = 3.0/(4*np.pi*r**3)
        row = dict(t=t, Lambda=rbar/rtid, M_trk=mt/MS, V_trk=vt, rhobar=rbar/RC, rhomax_loc=rmx/RC, R=R, r_enc=renc, r=r,
                   cx=cc[0], cy=cc[1], cz=cc[2], off=np.linalg.norm(cc), n_trk=int(trk.sum()), ok=1)
        p = cc; R = max(3.0, 2.0*renc)
    if not ok:
        row = dict(t=t, Lambda=np.nan, M_trk=np.nan, V_trk=np.nan, rhobar=np.nan, rhomax_loc=rmx/RC, R=R, r_enc=np.nan,
                   r=np.linalg.norm(p - xb), cx=p[0], cy=p[1], cz=p[2], off=np.linalg.norm(p), n_trk=0, ok=0)
    rows.append(row); print({k: (f'{v:.4g}' if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
import csv
with open(a.out, 'w') as fo:
    w = csv.DictWriter(fo, list(rows[0])); w.writeheader(); w.writerows(rows)
