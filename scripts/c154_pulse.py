#!/usr/bin/env python3
"""c154 (R-174 item 2, user-approved 2026-10-30): the pulsation energy budget of the moving pulsating star (ISO_A
with <problem> v_pert), from each pair of 3D dumps (prim_3d + adm_3d at the same time), in the star's frame:
  dm = D sqrt(gamma) dV (D = rho W, W = sqrt(1 + psi^4 |u|^2), u^i = the dump's velx..z = W v^i, sqrt(gamma) =
  psi^6), the Eulerian v^i = u^i/W, the bulk velocity v_b = <v>_m;
  E_kin = sum dm 1/2 psi^4 |v - v_b|^2 (|.|_phys; the motion about the bulk), E_int = sum dm P/((Gamma-1) rho),
  W = 1/2 sum dm ln(alpha) (the star's own lapse: no hole on this bed; weak field, Phi ~ ln alpha),
  E = E_kin + E_int + W, and the pulsation amplitude R_m = sqrt(<psi^4 |x - x_c|^2>_m) about the centre of mass.
Over cells with rho >= rho_min (default 1e-8 rho_c: the star, not the floor). Newtonian-order bookkeeping of a
weak-field star (M/R ~ 1e-4), the "closest available diagnostic" R-174 allows; argued, not exact GR.
    c154_pulse.py OUT.csv ARMDIR[:LABEL] [...] [--rho-min 1e-8]"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tde_ensemble_plots import bc   # noqa: E402

GAM = 5.0 / 3.0
RHOC = 3.24e-4
B = 'cfc_tde_wd_imbh_parabolic_tracker'


def cells(P, keys):
    g = np.asarray(P['mb_geometry'])
    nx = P['nx1_out_mb']
    out = {k: [] for k in keys}
    X, Y, Z, V = [], [], [], []
    for b in range(len(g)):
        x0, x1, y0, y1, z0, z1 = g[b][:6]
        dx, dy, dz = (x1 - x0) / nx, (y1 - y0) / nx, (z1 - z0) / nx
        c = (np.arange(nx) + 0.5)
        zz, yy, xx = np.meshgrid(z0 + c * dz, y0 + c * dy, x0 + c * dx, indexing='ij')
        X.append(xx.ravel()); Y.append(yy.ravel()); Z.append(zz.ravel())
        V.append(np.full(xx.size, dx * dy * dz))
        for k in keys:
            out[k].append(np.asarray(P['mb_data'][k][b], dtype=np.float64).ravel())
    cat = np.concatenate
    return {k: cat(v) for k, v in out.items()}, cat(X), cat(Y), cat(Z), cat(V)


def one(arm, lab, rmin):
    rows = []
    for fp in sorted(glob.glob(f'{arm}/bin/{B}.prim_3d.*.bin')):
        fa = fp.replace('.prim_3d.', '.adm_3d.')
        if not os.path.exists(fa):
            continue
        P, A = bc.read_binary(fp), bc.read_binary(fa)
        if P['time'] != A['time']:
            continue
        q, x, y, z, dv = cells(P, ['dens', 'velx', 'vely', 'velz', 'press'])
        a, _, _, _, _ = cells(A, ['adm_psi4', 'adm_alpha'])
        s = q['dens'] >= rmin * RHOC
        p4, al = a['adm_psi4'][s], a['adm_alpha'][s]
        u = np.stack([q['velx'][s], q['vely'][s], q['velz'][s]])
        W = np.sqrt(1.0 + p4 * (u**2).sum(0))
        dm = q['dens'][s] * W * p4**1.5 * dv[s]
        M = dm.sum()
        v = u / W
        vb = (v * dm).sum(1) / M
        X = np.stack([x[s], y[s], z[s]])
        xc = (X * dm).sum(1) / M
        ek = 0.5 * (dm * p4 * ((v - vb[:, None])**2).sum(0)).sum()
        ei = (dm * q['press'][s] / ((GAM - 1.0) * q['dens'][s])).sum()
        wg = 0.5 * (dm * np.log(al)).sum()
        rm = np.sqrt((dm * p4 * ((X - xc[:, None])**2).sum(0)).sum() / M)
        rows.append(dict(arm=lab, t=P['time'], M=M, E_kin=ek, E_int=ei, W=wg, E=ek + ei + wg, R_m=rm,
                         xc=xc[0], vb=vb[0], rho_max=q['dens'].max() / RHOC))
        print(lab, f"t={P['time']:.2f} Ekin={ek:.4e} Eint={ei:.4e} W={wg:.4e} E={ek + ei + wg:.6e} Rm={rm:.5f}",
              flush=True)
    return rows


def main():
    out = sys.argv[1]
    rmin = 1e-8
    args = sys.argv[2:]
    if '--rho-min' in args:
        i = args.index('--rho-min'); rmin = float(args[i + 1]); del args[i:i + 2]
    R = []
    for s in args:
        arm, lab = (s.split(':') + [None])[:2]
        R += one(arm, lab or os.path.basename(arm.rstrip('/')), rmin)
    import csv
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(R[0].keys()))
        w.writeheader(); w.writerows(R)


if __name__ == '__main__':
    main()
