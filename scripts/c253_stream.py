#!/usr/bin/env python3
"""c253 (R-253 items 2-3, user-approved 2026-11-09): the debris stream's refinement-level census and transverse
structure from the prim_3d dumps of one run (offline).
Per dump (t - t_p in [--t0, --t1]):
  remnant: centroid x_c of rho >= 0.01 rho_c (if none: the densest cell); r = |x_c - x_BH| (x_BH from the hst
    frame columns, c160_coldmax.xbh); the self-consistent Hill sphere r_H = r (M_H/3M)^(1/3), M_H the mass inside it
    (as c142's M_Hsc; M = M_BH = 1, coordinate volumes);
  stream = 1e-6 <= rho/rho_c < 0.01 (c169/c193's class), outside r_H about x_c and outside r_BH < 3;
  (2) census: the stream mass at each cell size dx (from mb_geometry), as a share of the stream mass;
  (3) at distances R from the hole (|r_BH - R| < 1; cells > --excl (5) M from x_c, so the arms, not the remnant's envelope): the azimuth phi (orbital plane, about x_BH) with the most stream
    mass (2-deg bins); the cells within +-15 deg of it: the mass-weighted Gaussian FWHM (2.3548 sigma) across the
    stream in z (thin direction) and in-plane (R sigma_phi), the mass-weighted dx and FWHM/dx, the peak density, and
    the mass-weighted K/K0. A moment width, not a fitted profile: the same method in every rung.
    c253_stream.py OUT_PREFIX RUNDIR TP [--t0 -60] [--t1 1e9] [--every 1] [--radii 10,20,40,80]"""
import argparse, glob, sys, types, csv
import numpy as np
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde/wt_tde_scripts/scripts')
import bin_convert as bc   # noqa: E402
from c160_coldmax import xbh   # noqa: E402

GAM, RC, K0, MS = 5/3, 3.24e-4, 1.2346626297e-2, 7.47e-5
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('run'); ap.add_argument('tp', type=float)
ap.add_argument('--t0', type=float, default=-60.0); ap.add_argument('--t1', type=float, default=1e9)
ap.add_argument('--every', type=int, default=1); ap.add_argument('--radii', default='10,20,40,80')
ap.add_argument('--excl', type=float, default=5.0)   # transverse: only cells this far from x_c
ap.add_argument('--dt', type=float, default=12.0)   # the prim_3d cadence: dump i is at t ~ i dt (pre-filter, avoids reading)
a = ap.parse_args()
RAD = [float(x) for x in a.radii.split(',')]
fs = sorted(glob.glob(f'{a.run}/bin/*.prim_3d.0*.bin'))
fs = [f for f in fs if a.t0 - a.dt <= int(f.split('.')[-2])*a.dt - a.tp <= a.t1 + a.dt]
cen, tra = [], []
for f in fs[::a.every]:
    P = bc.read_binary(f); t = P['time']
    if not (a.t0 <= t - a.tp <= a.t1):
        continue
    g = np.asarray(P['mb_geometry'], float); n = P['nx1_out_mb']
    X, Y, Z, H = [], [], [], []
    for q in range(len(g)):
        c = [np.linspace(g[q, 2*k], g[q, 2*k+1], n+1) for k in range(3)]; c = [0.5*(x[1:]+x[:-1]) for x in c]
        zz, yy, xx = np.meshgrid(c[2], c[1], c[0], indexing='ij')
        X.append(xx.ravel()); Y.append(yy.ravel()); Z.append(zz.ravel()); H.append(np.full(n**3, (g[q, 1]-g[q, 0])/n))
    X, Y, Z, H = map(np.concatenate, (X, Y, Z, H))
    cat = lambda k: np.concatenate([np.asarray(P['mb_data'][k][q]).ravel() for q in range(len(g))]).astype(float)
    rho, p = cat('dens'), cat('press')
    m = rho*H**3
    xb = xbh(a.run, t)
    rbh = np.sqrt((X-xb[0])**2 + (Y-xb[1])**2 + (Z-xb[2])**2)
    d1 = rho >= 0.01*RC
    if d1.any():
        xc = np.array([np.average(w[d1], weights=m[d1]) for w in (X, Y, Z)]); MH = m[d1].sum()
    else:
        i = np.argmax(np.where(rbh > 3, rho, 0)); xc = np.array([X[i], Y[i], Z[i]]); MH = 0.0
    r = np.linalg.norm(xc - xb); dc = np.sqrt((X-xc[0])**2 + (Y-xc[1])**2 + (Z-xc[2])**2)
    rH = 0.0
    for _ in range(40):
        rH = r*(MH/3.0)**(1/3); MH_new = m[dc < rH].sum()
        if abs(MH_new - MH) <= 1e-10*max(MH, 1e-300): break
        MH = MH_new
    st = (rho >= 1e-6*RC) & (rho < 0.01*RC) & (dc >= rH) & (rbh >= 3.0)
    Ms = m[st].sum()
    row = dict(t=t, t_tp=t - a.tp, rho_max=rho.max()/RC, r_rem=r, r_H=rH, M_H=MH/MS, M_stream=Ms/MS)
    for h in sorted(set(np.round(H, 8))):
        row[f'f_dx{1/h:g}'] = m[st & (np.abs(H - h) < 1e-9)].sum()/Ms if Ms > 0 else np.nan
    # the stream mass above the finest dx's 1/2-level: mass-weighted dx
    row['dx_mw'] = np.average(H[st], weights=m[st]) if Ms > 0 else np.nan
    cen.append(row)
    print({k: (f'{v:.4g}' if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    K = p/np.maximum(rho, 1e-300)**GAM/K0
    phi = np.degrees(np.arctan2(Y - xb[1], X - xb[0]))
    for R in RAD:
        s = st & (np.abs(rbh - R) < 1.0) & (dc > a.excl)   # the arms, not the remnant's envelope
        if m[s].sum() <= 0:
            continue
        hm, e = np.histogram(phi[s], bins=np.arange(-180, 181, 2), weights=m[s])
        ph = 0.5*(e[np.argmax(hm)] + e[np.argmax(hm)+1])
        dphi = (phi - ph + 180) % 360 - 180
        q = s & (np.abs(dphi) < 15)
        w = m[q]; W = w.sum()
        zc = np.average(Z[q], weights=w); sz = np.sqrt(np.average((Z[q]-zc)**2, weights=w))
        pc = np.average(dphi[q], weights=w); sp = np.radians(np.sqrt(np.average((dphi[q]-pc)**2, weights=w)))*R
        dxm = np.average(H[q], weights=w)
        tra.append(dict(t=t, t_tp=t - a.tp, R=R, phi=ph, M=W/MS, fwhm_z=2.3548*sz, fwhm_ip=2.3548*sp, dx_mw=dxm,
                        Nz=2.3548*sz/dxm, Nip=2.3548*sp/dxm, rho_pk=rho[q].max()/RC, K=np.average(K[q], weights=w)))
for name, rows in (('census', cen), ('transverse', tra)):
    if rows:
        keys = []
        for r_ in rows:
            keys += [k for k in r_ if k not in keys]
        with open(f'{a.out}_{name}.csv', 'w') as fo:
            wr = csv.DictWriter(fo, keys); wr.writeheader(); wr.writerows(rows)
