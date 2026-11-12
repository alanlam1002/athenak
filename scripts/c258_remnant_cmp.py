#!/usr/bin/env python3
"""c258 (R-257 item 1, user-approved 2026-11-12): the remnant at matched t - t_p across the P18 runs (P18T, the original
P18 = c134, P18_r82, L32_P18), one row per full-3D restart. Offline, read-only.
Per restart (tde_dmde.cells, rcut 3, as c142_remnant.py):
  rho_max/rho_c        the densest cell outside r_BH < 3 (RC = 3.24e-4)
  M_rem, x_c, v_c      remnant.bound_mass (the energy-bound set, monopole self-potential; c142)
  r_bh, r_H, M_Hill    r_H = r_bh (M_rem/3)^(1/3), M_Hill the gas mass within r_H of x_c (C-123; c142)
  r_Hsc, M_Hsc         the self-consistent Hill sphere (c142)
  R50, R90             radii about x_c enclosing 50% / 90% of the bound set's mass
  K_Hill               mass-weighted K/K0 inside r_H (K = P/rho^Gamma, K0 = kappa; c142 reports K/K0 - 1)
  K_in50               mass-weighted K/K0 of the innermost 50% (by |x - x_c|) of M_Hsc
  Lambda_trk, fc_mass  the relative tracker's set, offline (c252_lambda_cal.py's definition, rho dV masses, f_rel 0.3,
                       3 passes) seeded at x_c instead of the previous dump's centroid; fc_mass in M*
  Mdot_H50             the net mass leaving the Hill sphere per 50 M: sum over a shell |r - r_H| < h/2 (h = max(0.05 r_H,
                       2 dx_local)) of dM (u . r_hat)/h, u = V - V_c (V^i = alpha v^i - beta'^i, c142), x 50, in M*
  de_*                 DE switch code mass shares inside r_H (c192's codes, from the dediag_3d dump nearest in time;
                       positions shifted by V_c (t_dd - t); weights rho dV from the prim_3d dump at the same time)
    c258_remnant_cmp.py OUT.csv LABEL=RST:TP:RUNDIR [...] [--nproc 2]"""
import argparse
import csv
import glob
import os
import sys
import types
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
import remnant as RM      # noqa: E402
import tde_dmde as T      # noqa: E402
import bin_convert as bc  # noqa: E402

GAM, K0, MSTAR, MBH, RC = 5.0 / 3.0, 1.2346626297e-2, 7.47e-5, 1.0, 3.24e-4
CODES = (0, 11, 12, 13, 14, 15, 16, 17)


def lam_trk(X, Y, Z, rho, dV, p, xbh, frel=0.3, nit=3):
    R, row = 3.0, None
    for _ in range(nit):
        d2 = (X - p[0])**2 + (Y - p[1])**2 + (Z - p[2])**2
        inR = d2 < R * R
        rmx = rho[inR].max() if inR.any() else 0.0
        if rmx <= 1e-4 * RC:
            return np.nan, np.nan
        trk = inR & (rho > frel * rmx)
        m = rho[trk] * dV[trk]
        mt = m.sum()
        cc = np.array([(m * X[trk]).sum(), (m * Y[trk]).sum(), (m * Z[trk]).sum()]) / mt
        renc = np.sqrt(((X[trk] - cc[0])**2 + (Y[trk] - cc[1])**2 + (Z[trk] - cc[2])**2).max())
        r = np.linalg.norm(cc - xbh)
        row = ((mt / dV[trk].sum()) / (3.0 / (4 * np.pi * r**3)), mt / MSTAR)
        p, R = cc, max(3.0, 2.0 * renc)
    return row


def bin_cells(f, keys):
    P = bc.read_binary(f)
    g = np.asarray(P['mb_geometry'], float)
    n = P['nx1_out_mb']
    h = (g[:, 1] - g[:, 0]) / n
    ax = lambda lo: g[:, lo][:, None] + (np.arange(n) + 0.5)[None, :] * h[:, None]
    X, Y, Z = ax(0), ax(2), ax(4)
    nb = len(g)
    x = np.broadcast_to(X[:, None, None, :], (nb, n, n, n)).ravel()
    y = np.broadcast_to(Y[:, None, :, None], (nb, n, n, n)).ravel()
    z = np.broadcast_to(Z[:, :, None, None], (nb, n, n, n)).ravel()
    out = {k: np.concatenate([np.asarray(P['mb_data'][k][b], float).ravel() for b in range(nb)]) for k in keys}
    return P['time'], g, x, y, z, np.repeat(h**3, n**3), out


def btime(f):
    """a bin file's time from its text header (no data read)"""
    with open(f, 'rb') as fh:
        for ln in fh.read(512).split(b'\n'):
            if ln.strip().startswith(b'time='):
                return float(ln.split(b'=')[1])
    return bc.read_binary(f)['time']


def de_shares(rundir, t, xc, Vc, rH):
    fd = glob.glob(f'{rundir}/bin/*.dediag_3d.*.bin')
    fp = glob.glob(f'{rundir}/bin/*.prim_3d.0*.bin')
    if not fd or not fp:
        return {}
    td = {btime(f): f for f in fd}
    t_dd = min(td, key=lambda s: abs(s - t))
    tp = {btime(f): f for f in fp}
    t_pp = min(tp, key=lambda s: abs(s - t_dd))
    _, g, x, y, z, dv, D = bin_cells(td[t_dd], ['de_branch'])
    _, gp, _, _, _, _, Pp = bin_cells(tp[t_pp], ['dens'])
    if gp.shape != g.shape or not np.allclose(gp, g):
        return dict(de_t=t_dd, de_note='prim/dediag meshes differ')
    c = xc + Vc * (t_dd - t)
    inh = (x - c[0])**2 + (y - c[1])**2 + (z - c[2])**2 < rH**2
    m = Pp['dens'] * dv
    br = D['de_branch'].round().astype(int)
    mt = m[inh].sum()
    out = dict(de_t=t_dd, de_dt_prim=t_pp - t_dd)
    for k in CODES:
        out[f'de_{k}'] = m[inh & (br == k)].sum() / mt if mt > 0 else np.nan
    return out


def one(arg):
    lab, fn, tp, rundir = arg
    c = T.cells(fn, rcut=3.0)
    Mb, xc, _, sel = RM.bound_mass(c)
    X = np.stack([c['x'], c['y'], c['z']])
    V = c['alpha'] * np.stack([c['vx'], c['vy'], c['vz']]) - np.stack([c['betax'], c['betay'], c['betaz']])
    dM, rho, dV = c['dM'], c['rho'], c['dV']
    Vc = (V * dM * sel).sum(1) / (dM * sel).sum()
    d = X - xc[:, None]
    r = np.linalg.norm(d, axis=0)
    xbh = np.asarray(c['xbh'], float)
    rbh = float(np.linalg.norm(xc - xbh))
    rH = rbh * (Mb / (3.0 * MBH))**(1.0 / 3.0)
    inH = r < rH
    mh = dM[inH].sum()
    rh2, mh2 = rH, mh
    for _ in range(50):
        r_new = rbh * (mh2 / (3.0 * MBH))**(1.0 / 3.0)
        m_new = dM[r < r_new].sum()
        done = abs(m_new - mh2) <= 1e-12 * max(mh2, 1e-300)
        rh2, mh2 = r_new, m_new
        if done:
            break
    K = c['P'] / rho**GAM / K0
    o = np.argsort(r[sel])
    cm = np.cumsum(dM[sel][o]) / Mb
    R50 = float(r[sel][o][np.searchsorted(cm, 0.5)])
    R90 = float(r[sel][o][np.searchsorted(cm, 0.9)])
    inS = r < rh2
    o2 = np.argsort(r[inS])
    cm2 = np.cumsum(dM[inS][o2])
    k50 = o2[: max(1, np.searchsorted(cm2, 0.5 * mh2))]
    K_in50 = np.average(K[inS][k50], weights=dM[inS][k50])
    lam, fcm = lam_trk(c['x'], c['y'], c['z'], rho, dV, xc, xbh)
    dxl = dV**(1.0 / 3.0)
    hs = np.maximum(0.05 * rH, 2.0 * dxl)
    sh = np.abs(r - rH) < 0.5 * hs
    ur = ((V - Vc[:, None]) * d).sum(0) / np.maximum(r, 1e-12)
    mdot = (dM[sh] * ur[sh] / hs[sh]).sum()
    row = dict(label=lab, t=c['t'], t_minus_tp=c['t'] - tp, rhomax_rhoc=rho.max() / RC,
               M_rem=Mb / MSTAR, M_Hill=mh / MSTAR, M_Hsc=mh2 / MSTAR, r_H=rH, r_Hsc=rh2, r_bh=rbh,
               R50=R50, R90=R90, K_Hill=np.average(K[inH], weights=dM[inH]) if inH.any() else np.nan,
               K_in50=K_in50, Lambda_trk=lam, fc_mass=fcm, Mdot_H50=50.0 * mdot / MSTAR)
    row.update(de_shares(rundir, c['t'], xc, Vc, rH))
    print({k: (f'{v:.4g}' if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--nproc', type=int, default=2)
    a = ap.parse_args()
    args = []
    for s in a.runs:
        lab, rest = s.split('=', 1)
        fn, tp, rd = rest.split(':')
        args.append((lab, fn, float(tp), rd))
    with Pool(a.nproc) as p:
        rows = p.map(one, args)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(a.out, 'w') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
