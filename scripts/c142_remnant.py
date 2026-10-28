#!/usr/bin/env python3
"""c142 (R-152 item 3, user-approved 2026-10-28): the remnant's B series and its spin against break-up, on full-3D
restarts (tde_dmde.cells, rcut 3; remnant.bound_mass for the self-bound set and its centre x_c, as c138_audit).
  B (one row per restart): remnant.py's bound mass M_rem and its K_rem = <K/K0 - 1> (mass-weighted; K0 = kappa);
      M_Hill = the gas mass inside r_H = r (M_rem/3M)^(1/3) (C-123's definition, r = |x_c - x_BH|, M = 1); and
      the self-consistent Hill mass M_Hsc (r_H iterated with the mass inside it, M_H -> r_H(M_H) -> M_H).
  spin (i): in the remnant's frame, u = V - V_c (V^i = alpha v^i - beta'^i, the coordinate velocity as c138;
      V_c the bound set's mass-weighted V), about the axis e = J/|J| of the bound set's angular momentum
      J = sum dM d x u (d = x - x_c); per radial shell [r_k, r_k+1) of |d| (edges in units of r_H):
        Omega_sh = sum dM (d x u).e / sum dM R^2           (R = |d - (d.e)e|, the shell's mean angular velocity)
        Omega_eq = mass-weighted u_phi/R over the shell's equatorial band |d.e|/|d| < sin 30 deg
        Omega_K  = (M(<r)/r^3)^(1/2), M(<r) the gas mass within the shell's outer radius (G = 1), r that radius
      and the ratios Omega/Omega_K (an equatorial envelope at >~ 0.8 Omega_K is shedding; R-151). The remnant's
      orbital angular velocity about the hole, Omega_orb = ((x_c - x_BH) x (V_c - V_BH)).e/|x_c - x_BH|^2 (V_BH =
      -xidot, the hole's grid velocity: x_BH = puncture_x - xi(t)), is reported too, and Omega - Omega_orb, the
      spin in the frame co-rotating with the orbit (a tidally locked remnant has Omega = Omega_orb). Lengths, speeds
      and Omega are |.|_coord (psi^2 - 1 ~ 1e-2 at r_BH ~ 100: the physical values differ by ~1%).
    c142_remnant.py OUT LABEL=RST:TP [...] [--nproc 2]
-> OUT/c142_B.csv (per restart), OUT/c142_spin.csv (per restart and shell)"""
import argparse
import csv
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import remnant as RM   # noqa: E402
import tde_dmde as T   # noqa: E402

GAM = 5.0 / 3.0
K0 = 1.2346626297e-2
MSTAR = 7.47e-5
MBH = 1.0
EDGES = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 3.0])
S30 = np.sin(np.radians(30.0))


def one(arg):
    lab, fn, tp = arg
    c = T.cells(fn, rcut=3.0)
    Mb, xc, _, sel = RM.bound_mass(c)
    X = np.stack([c['x'], c['y'], c['z']])
    v = np.stack([c['vx'], c['vy'], c['vz']])
    beta = np.stack([c['betax'], c['betay'], c['betaz']])
    V = c['alpha'] * v - beta
    dM = c['dM']
    Vc = (V * dM * sel).sum(1) / (dM * sel).sum()
    d = X - xc[:, None]
    u = V - Vc[:, None]
    r = np.linalg.norm(d, axis=0)
    xbh = np.asarray(c['xbh'], dtype=float)
    rbh = float(np.linalg.norm(xc - xbh))
    rH = rbh * (Mb / (3.0 * MBH))**(1.0 / 3.0)
    mh = dM[r < rH].sum()
    rh2, mh2 = rH, mh                       # the self-consistent Hill sphere
    for _ in range(50):
        r_new = rbh * (mh2 / (3.0 * MBH))**(1.0 / 3.0)
        m_new = dM[r < r_new].sum()
        if abs(m_new - mh2) <= 1e-12 * max(mh2, 1e-300):
            rh2, mh2 = r_new, m_new
            break
        rh2, mh2 = r_new, m_new
    k = c['P'] / c['rho']**GAM / K0 - 1.0
    L = np.cross(d.T, u.T).T                # specific angular momentum about x_c
    J = (L * dM * sel).sum(1)
    e = J / np.linalg.norm(J)
    Lz = (L * e[:, None]).sum(0)
    z = (d * e[:, None]).sum(0)
    dbh = xc - xbh
    om_orb = float(np.dot(np.cross(dbh, Vc + np.asarray(c['xdot'], dtype=float)), e)) / rbh**2
    R2 = np.maximum(r**2 - z**2, 1e-30)
    eq = np.abs(z) < S30 * np.maximum(r, 1e-12)
    row = dict(label=lab, t=c['t'], t_minus_tp=c['t'] - tp, M_rem=Mb / MSTAR, K_rem=np.average(k[sel], weights=dM[sel]),
               M_Hill=mh / MSTAR, r_H=rH, M_Hsc=mh2 / MSTAR, r_Hsc=rh2, r_bh=rbh,
               K_Hill=np.average(k[r < rH], weights=dM[r < rH]) if (r < rH).any() else np.nan,
               J_rem=float(np.linalg.norm(J)) / MSTAR, e_tilt_deg=float(np.degrees(np.arccos(abs(e[2])))),
               Vc=float(np.linalg.norm(Vc)), Omega_orb=om_orb)
    print(row, flush=True)
    prof = []
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        s = (r >= a * rH) & (r < b * rH)
        if dM[s].sum() <= 0:
            continue
        rout = b * rH
        ok = np.sqrt(dM[r < rout].sum() / rout**3)
        om = (dM[s] * Lz[s]).sum() / (dM[s] * R2[s]).sum()
        se = s & eq
        oe = np.average(Lz[se] / R2[se], weights=dM[se]) if se.any() else np.nan
        prof.append(dict(label=lab, t_minus_tp=c['t'] - tp, r_lo=a, r_hi=b, r_H=rH, m_shell=dM[s].sum() / MSTAR,
                         m_enc=dM[r < rout].sum() / MSTAR, Omega_sh=om, Omega_eq=oe, Omega_K=ok,
                         sh_over_K=om / ok, eq_over_K=oe / ok, Omega_orb=om_orb,
                         rot_sh_over_K=(om - om_orb) / ok, rot_eq_over_K=(oe - om_orb) / ok,
                         bound_frac=dM[s & sel].sum() / dM[s].sum(),
                         K_mean=np.average(k[s], weights=dM[s])))
    return row, prof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--nproc', type=int, default=2)
    a = ap.parse_args()
    args = []
    for s in a.runs:
        lab, rest = s.split('=', 1)
        fn, tp = rest.rsplit(':', 1)
        args.append((lab, fn, float(tp)))
    with Pool(a.nproc) as p:
        res = p.map(one, args)
    os.makedirs(a.out, exist_ok=True)
    rows = [r for r, _ in res]
    with open(f'{a.out}/c142_B.csv', 'w') as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    prof = [q for _, p in res for q in p]
    with open(f'{a.out}/c142_spin.csv', 'w') as f:
        w = csv.DictWriter(f, list(prof[0].keys()))
        w.writeheader()
        w.writerows(prof)


if __name__ == '__main__':
    main()
