#!/usr/bin/env python3
"""c311 (R-283 items 1, 2a, 2d, 4; user-approved 2026-11-25; offline, read-only): the hot dilute class that sets global dt.
Class H: rho < --rhomax (1e-10 code units) and P/rho > --tmin (0.01), rho > --rhoatm (1e-22 = 100 dfloor, the ceiling's band).
From one prim_3d dump (and the dediag_3d at the same time if present), STREAMED block by block (B0's dumps are 10 GB):
  - mass m = rho dV (coordinate volume; psi^6 W ~ 1.0-1.05 at r > 20 M, argued) and E_int = P/(Gamma-1) dV, in M* (7.47e-5);
    totals of all gas (rho > rhoatm) and of the debris (outside the remnant's Hill sphere) for the fractions;
  - P/rho mass quantiles 10/50/90/99 and the mass per log10(P/rho) bin; the virial ratio (P/rho)/(M_BH/r_BH);
  - where: near-hole (r_BH < 12, R-278 §2's zone-edge region), remnant (inside 3 r_H of the remnant centroid rho >= 0.01 rho_c),
    stream surface (within --dsurf 2 M of dense gas rho >= 1e-6 rho_c), tails bound/unbound (Newtonian 0.5 v^2 - M/r_BH + h-1);
    the distance to the dense gas (mass quantiles), |v|;
  - the DE branch codes (dediag_3d de_branch: 0 energy-only (non-dense), 11-15 per primitive_solver_hyd.hpp:2843-2846).
Prims: dens, velx..z (= W v^i, Eulerian), press. x_BH from c160_coldmax.xbh (the frame displacement).
    c311_hotdilute.py LABEL RUNDIR T [--out CSVPREFIX] [--dump N]"""
import argparse, glob, os, sys, types, json
import numpy as np
from scipy.spatial import cKDTree
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde/wt_tde_scripts/scripts')
from c160_coldmax import xbh, par   # noqa: E402

GAM = 5.0/3.0; MSTAR = 7.47e-5
ap = argparse.ArgumentParser()
ap.add_argument('label'); ap.add_argument('rundir'); ap.add_argument('t', type=float)
ap.add_argument('--rhomax', type=float, default=1e-10); ap.add_argument('--tmin', type=float, default=0.01)
ap.add_argument('--rhoatm', type=float, default=1e-22); ap.add_argument('--dsurf', type=float, default=2.0)
ap.add_argument('--rnear', type=float, default=12.0); ap.add_argument('--out', default=None)
ap.add_argument('--cls', default=None, help='write the class cells (x,y,z,rho,P,v,code) to this .npz')
a = ap.parse_args()


def header(fn):
    fp = open(fn, 'rb'); fp.readline()
    n = int(fp.readline().split(b'=')[-1]); ph = {}
    for _ in range(n - 1):
        k, v = [s.strip() for s in fp.readline().decode().split('=')]; ph[k] = v
    nv = int(fp.readline().split(b'=')[-1]); var = [v.decode() for v in fp.readline().split()[1:]]
    hs = int(fp.readline().split(b'=')[-1]); hdr = fp.read(hs).decode()
    ng = int([l.split('=')[1].split('#')[0] for l in hdr.split('\n') if l.strip().startswith('nghost')][0])
    return fp, float(ph['time']), int(ph['size of location']), int(ph['size of variable']), var, ng


def blocks(fn):
    fp, t, ls, vs, var, ng = header(fn)
    fp.seek(0, 2); size = fp.tell(); fp.close()
    fp, *_ = header(fn)
    lt = np.float64 if ls == 8 else np.float32; vt = np.float64 if vs == 8 else np.float32
    while fp.tell() < size:
        idx = np.frombuffer(fp.read(24), dtype=np.int32).astype(np.int64)
        n1, n2, n3 = idx[1] - idx[0] + 1, idx[3] - idx[2] + 1, idx[5] - idx[4] + 1
        fp.read(16); g = np.frombuffer(fp.read(6*ls), dtype=lt).astype(float)
        d = np.fromfile(fp, dtype=vt, count=n1*n2*n3*len(var)).reshape(len(var), n3, n2, n1)
        yield g, {v: d[i] for i, v in enumerate(var)}
    fp.close()


def nearest(pattern):
    fs = glob.glob(f'{a.rundir}/bin/*.{pattern}.*.bin')
    if not fs: return None, None
    tt = [(abs(header(f)[1] - a.t), f) for f in fs]; dt, f = min(tt)
    return f, header(f)[1]


fp3, t3 = nearest('prim_3d')
if fp3 is None or abs(t3 - a.t) > 1.0:
    alt, ta = nearest('prim_3d_peri')
    if alt is not None and abs(ta - a.t) < abs((t3 if t3 is not None else 1e30) - a.t): fp3, t3 = alt, ta
fd, td = nearest('dediag_3d')
use_de = fd is not None and td == t3
X = xbh(a.rundir, t3); rhoc = par(a.rundir, 'rhoc'); mbh = par(a.rundir, 'puncture_mass', 1.0)
print(f'{a.label}: {fp3} t = {t3:.3f}; x_BH {np.round(X, 3)}; dediag {"yes" if use_de else "no"}', flush=True)

# pass 1: dense cells (for the KD tree), the remnant centroid, totals
dense_pts = []; mc = 0.0; xc = np.zeros(3); Mtot = Etot = 0.0
for g, v in blocks(fp3):
    n = v['dens'].shape[-1]; h = (g[1] - g[0])/n; dV = h**3
    r = v['dens']; P = v['press']; gas = r > a.rhoatm
    Mtot += r[gas].sum()*dV; Etot += P[gas].sum()*dV/(GAM - 1)
    xs = g[0] + (np.arange(n) + 0.5)*h; ys = g[2] + (np.arange(n) + 0.5)*h; zs = g[4] + (np.arange(n) + 0.5)*h
    Z, Y, Xg = np.meshgrid(zs, ys, xs, indexing='ij')
    dn = r >= 1e-6*rhoc
    if dn.any(): dense_pts.append(np.stack([Xg[dn], Y[dn], Z[dn]], 1))
    hi = r >= 0.01*rhoc
    if hi.any():
        w = r[hi]*dV; mc += w.sum(); xc += [(Xg[hi]*w).sum(), (Y[hi]*w).sum(), (Z[hi]*w).sum()]
xc /= mc
dense_pts = np.concatenate(dense_pts); tree = cKDTree(dense_pts)
Dsep = np.linalg.norm(xc - X)
# Hill radius of the remnant (iterate on the mass inside it; c194's rule)
mrem = mc
for _ in range(3):
    rh = Dsep*(mrem/(3.0*mbh))**(1.0/3.0)
    mrem = 0.0
    for g, v in blocks(fp3):
        n = v['dens'].shape[-1]; h = (g[1] - g[0])/n
        cen = np.array([g[0] + g[1], g[2] + g[3], g[4] + g[5]])/2
        if np.linalg.norm(cen - xc) > rh + 0.9*n*h: continue
        xs = g[0] + (np.arange(n) + 0.5)*h; ys = g[2] + (np.arange(n) + 0.5)*h; zs = g[4] + (np.arange(n) + 0.5)*h
        Z, Y, Xg = np.meshgrid(zs, ys, xs, indexing='ij')
        ins = (Xg - xc[0])**2 + (Y - xc[1])**2 + (Z - xc[2])**2 < rh*rh
        mrem += v['dens'][ins & (v['dens'] > a.rhoatm)].sum()*h**3
print(f'  remnant centroid {np.round(xc, 2)}, r_BH {Dsep:.2f}, r_H {rh:.2f}, M(<r_H) {mrem/MSTAR:.4f} M*; '
      f'all gas {Mtot/MSTAR:.4f} M*, E_int {Etot/MSTAR:.4e} M*', flush=True)

# pass 2: the class
C = {k: [] for k in ('x', 'y', 'z', 'rho', 'T', 'm', 'E', 'v', 'rbh', 'bern', 'code', 'lev')}
Mdeb = Edeb = 0.0
de_iter = blocks(fd) if use_de else None
for g, v in blocks(fp3):
    dg = next(de_iter) if use_de else None
    n = v['dens'].shape[-1]; h = (g[1] - g[0])/n; dV = h**3
    xs = g[0] + (np.arange(n) + 0.5)*h; ys = g[2] + (np.arange(n) + 0.5)*h; zs = g[4] + (np.arange(n) + 0.5)*h
    Z, Y, Xg = np.meshgrid(zs, ys, xs, indexing='ij')
    r = v['dens']; P = v['press']; gas = r > a.rhoatm
    outH = (Xg - xc[0])**2 + (Y - xc[1])**2 + (Z - xc[2])**2 >= rh*rh
    Mdeb += r[gas & outH].sum()*dV; Edeb += P[gas & outH].sum()*dV/(GAM - 1)
    T = np.where(r > 0, P/np.where(r > 0, r, 1), 0)
    s = gas & (r < a.rhomax) & (T > a.tmin)
    if not s.any(): continue
    u2 = v['velx'][s]**2 + v['vely'][s]**2 + v['velz'][s]**2; vv = np.sqrt(u2/(1 + u2))
    rb = np.sqrt((Xg[s] - X[0])**2 + (Y[s] - X[1])**2 + (Z[s] - X[2])**2)
    hh = 1 + GAM/(GAM - 1)*T[s]
    for k, val in (('x', Xg[s]), ('y', Y[s]), ('z', Z[s]), ('rho', r[s]), ('T', T[s]), ('m', r[s]*dV), ('E', P[s]*dV/(GAM - 1)),
                   ('v', vv), ('rbh', rb), ('bern', 0.5*vv**2 - mbh/rb + hh - 1), ('lev', np.full(s.sum(), h))):
        C[k].append(np.asarray(val, float))
    C['code'].append(dg[1]['de_branch'][s].round().astype(float) if use_de else np.full(s.sum(), -1.0))
C = {k: np.concatenate(val) if val else np.zeros(0) for k, val in C.items()}
pts = np.stack([C['x'], C['y'], C['z']], 1)
dd = tree.query(pts)[0] if len(pts) else np.zeros(0)
rr = np.sqrt(((pts - xc)**2).sum(1)) if len(pts) else np.zeros(0)


def wq(x, w, qs):
    if len(x) == 0: return [np.nan]*len(qs)
    o = np.argsort(x); c = np.cumsum(w[o]); c /= c[-1]
    return [x[o][min(np.searchsorted(c, q), len(x) - 1)] for q in qs]


m = C['m']; M = m.sum(); E = C['E'].sum()
near = C['rbh'] < a.rnear; rem = (rr < 3*rh) & ~near; surf = (dd < a.dsurf) & ~near & ~rem
tail = ~near & ~rem & ~surf
loc = {'near_hole': near, 'remnant_3rH': rem, 'stream_surface': surf, 'tail_bound': tail & (C['bern'] < 0),
       'tail_unbound': tail & (C['bern'] >= 0)}
R = dict(label=a.label, run=a.rundir, t=t3, x_bh=X.tolist(), xc=xc.tolist(), r_H=rh, M_rem_Ms=mrem/MSTAR,
         M_gas_Ms=Mtot/MSTAR, E_gas_Ms=Etot/MSTAR, M_deb_Ms=Mdeb/MSTAR, E_deb_Ms=Edeb/MSTAR, ncell=int(len(m)),
         M_cls_Ms=M/MSTAR, E_cls_Ms=E/MSTAR, fM_deb=M/Mdeb if Mdeb else np.nan, fE_deb=E/Edeb if Edeb else np.nan,
         fE_gas=E/Etot, T_q=wq(C['T'], m, (0.1, 0.5, 0.9, 0.99)),
         Tvir_q=wq(C['T']*C['rbh']/mbh, m, (0.1, 0.5, 0.9)), v_q=wq(C['v'], m, (0.1, 0.5, 0.9)),
         rho_q=wq(C['rho'], m, (0.1, 0.5, 0.9)), dist_dense_q=wq(dd, m, (0.1, 0.5, 0.9)),
         rbh_q=wq(C['rbh'], m, (0.1, 0.5, 0.9)),
         loc={k: [float(m[s].sum()/MSTAR), float(C['E'][s].sum()/MSTAR), int(s.sum()),
                  *[float(q) for q in wq(C['T'][s], m[s], (0.5, 0.9))]] for k, s in loc.items()},
         codes={str(int(c)): float(m[C['code'] == c].sum()/M) for c in np.unique(C['code'])} if M else {},
         dx={f'{h:.5g}': float(m[C['lev'] == h].sum()/M) for h in np.unique(C['lev'])} if M else {})
edges = np.arange(-2.0, 1.01, 0.25); R['Thist_edges'] = edges.tolist()
R['Thist_Ms'] = (np.histogram(np.log10(C['T']), edges, weights=m)[0]/MSTAR).tolist() if M else []
# the same mass per log10 rho bin (class only)
redges = np.arange(-22, -9.99, 1.0); R['rhohist_edges'] = redges.tolist()
R['rhohist_Ms'] = (np.histogram(np.log10(C['rho']), redges, weights=m)[0]/MSTAR).tolist() if M else []
print(json.dumps(R, indent=None, default=float))
if a.out:
    json.dump(R, open(a.out + '.json', 'w'), indent=1, default=float)
if a.cls:
    np.savez_compressed(a.cls, dist=dd, rrem=rr, **C)
