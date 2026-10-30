#!/usr/bin/env python3
"""c155 (R-176 addendum / R-177 item 4, user-approved 2026-10-30): B1T's late cold, dense spot (C-143: rho_max/rho_c
1.22 -> 2.13 and K/K0 at rho_max 0.63 -> 0.10 over t = 108 -> 120). From the 3D primitive dumps, per dump and arm:
  - the rho_max cell: rho/rho_c, K/K0 (K = P/rho^Gamma, K0 = kappa), its refinement level, its position in grid
    (comoving) coordinates, its distance |.|_coord to the hole (x_BH = puncture_x - frame_disp, the user history's
    frame_disp_x1..3 at the dump time) and to the star's centre (rho-weighted over rho >= 0.01 rho_c);
  - the "cold dense" set C = {rho >= rho_c, K/K0 < 0.5}: its mass (rho dV_coord, as c122_b1) as a fraction of the
    dense mass (rho >= 0.01 rho_c), its cell count, levels, centroid (distances as above) and the mass-weighted
    |u - u_bulk| of C (u^i = the dump's W v^i, |.|_coord) against the dense gas's;
  - the dense gas's K/K0 percentiles (1, 5, 50) and the scalar s_00 (the dual-energy tracer slot; DE off here) at
    rho_max.
    c155_coldspot.py OUT.csv RUNDIR[:LABEL] [...]   (deck <dir>/../<basename>.par, history <dir>/*.user.hst)"""
import glob
import os
import re
import sys
import types

import numpy as np

sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
import bin_convert as bc   # noqa: E402

GAM = 5.0 / 3.0
B = 'cfc_tde_wd_imbh_parabolic_tracker'


def par(d, key):
    s = open(f'{d}/../{os.path.basename(d.rstrip("/"))}.par').read()
    return float(re.search(r'(?m)^%s\s*=\s*([-+.\deE]+)' % key, s).group(1))


def frame_disp(d):
    f = f'{d}/{B}.user.hst'
    names = re.findall(r'\[\d+\]=(\S+)', ''.join(l for l in open(f) if l.startswith('#')))
    h = np.loadtxt(f, ndmin=2)
    t = h[:, 0]
    o = np.argsort(t, kind='stable')
    return t[o], np.stack([h[o, names.index(f'frame_disp_x{a}')] for a in (1, 2, 3)])


def load(f):
    P = bc.read_binary(f)
    g = np.asarray(P['mb_geometry'])
    lev = np.asarray(P['mb_logical'])[:, 3]
    n = P['nx1_out_mb']
    q = P['mb_data']
    c = np.arange(n) + 0.5
    X, Y, Z, V, L = [], [], [], [], []
    for b in range(len(g)):
        x0, x1, y0, y1, z0, z1 = g[b][:6]
        zz, yy, xx = np.meshgrid(z0 + c * (z1 - z0) / n, y0 + c * (y1 - y0) / n, x0 + c * (x1 - x0) / n, indexing='ij')
        X.append(xx.ravel()); Y.append(yy.ravel()); Z.append(zz.ravel())
        V.append(np.full(n**3, ((x1 - x0) / n)**3)); L.append(np.full(n**3, lev[b]))
    cat = np.concatenate
    get = lambda k: cat([np.asarray(q[k][b], dtype=np.float64).ravel() for b in range(len(g))])
    return (float(P['time']), get('dens'), get('press'), np.stack([get('velx'), get('vely'), get('velz')]),
            get('s_00') if 's_00' in q else None, np.stack([cat(X), cat(Y), cat(Z)]), cat(V), cat(L))


def one(d, lab):
    rhoc, k0 = par(d, 'rhoc'), par(d, 'kappa')
    xp = np.array([par(d, f'puncture_x{a}') for a in (1, 2, 3)])
    th, fd = frame_disp(d)
    rows = []
    for f in sorted(glob.glob(f'{d}/bin/{B}.prim_3d.*.bin')):
        t, rho, p, u, s0, x, dv, lev = load(f)
        xbh = xp - np.array([np.interp(t, th, fd[a]) for a in range(3)])
        kk = p / rho**GAM / k0
        dn = rho >= 0.01 * rhoc
        w = rho[dn] * dv[dn]
        xc = (x[:, dn] * w).sum(1) / w.sum()
        ub = (u[:, dn] * w).sum(1) / w.sum()
        i = int(np.argmax(rho))
        C = dn & (rho >= rhoc) & (kk < 0.5)
        wc = rho[C] * dv[C]
        r = dict(arm=lab, t=t, rhomax=rho[i] / rhoc, K_at_rhomax=kk[i], lev_rhomax=int(lev[i]),
                 x_rhomax=x[0, i], y_rhomax=x[1, i], z_rhomax=x[2, i],
                 d_bh_rhomax=float(np.linalg.norm(x[:, i] - xbh)), d_c_rhomax=float(np.linalg.norm(x[:, i] - xc)),
                 d_bh_centre=float(np.linalg.norm(xc - xbh)),
                 s00_rhomax=(s0[i] if s0 is not None else np.nan),
                 cold_mass_frac=wc.sum() / w.sum(), cold_ncell=int(C.sum()),
                 cold_levels='/'.join(str(v) for v in sorted(set(lev[C].astype(int)))) if C.any() else '',
                 K_p1=np.percentile(kk[dn], 1), K_p5=np.percentile(kk[dn], 5), K_p50=np.percentile(kk[dn], 50))
        if C.any():
            cc = (x[:, C] * wc).sum(1) / wc.sum()
            r.update(d_bh_cold=float(np.linalg.norm(cc - xbh)), d_c_cold=float(np.linalg.norm(cc - xc)),
                     urel_cold=float((np.linalg.norm(u[:, C] - ub[:, None], axis=0) * wc).sum() / wc.sum()))
        else:
            r.update(d_bh_cold=np.nan, d_c_cold=np.nan, urel_cold=np.nan)
        r['urel_dense'] = float((np.linalg.norm(u[:, dn] - ub[:, None], axis=0) * w).sum() / w.sum())
        rows.append(r)
        print(lab, ' '.join(f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}' for k, v in r.items() if k != 'arm'),
              flush=True)
    return rows


def main():
    out, R = sys.argv[1], []
    for s in sys.argv[2:]:
        d, lab = (s.split(':') + [None])[:2]
        R += one(d, lab or os.path.basename(d.rstrip('/')))
    import csv
    keys = list(R[0].keys())
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(R)


if __name__ == '__main__':
    main()
