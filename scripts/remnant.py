#!/usr/bin/env python3
"""P2 (research R-034): does a self-bound remnant survive? Iterative energy criterion (in the
spirit of GRR13, R-002 §1.4), Newtonian, monopole self-potential about the clump.

From a restart (3D): start from cells with rho > 1e-3 rho_max; repeat
  CoM x_c, v_c (mass-weighted over the current bound set);
  Phi(r) = -M(<r)/r - sum_{r'>r} dm'/r'   (monopole about x_c, all mass);
  e = |v - v_c|^2/2 + eps + Phi,  eps = P/((Gamma-1) rho);
  bound = e < 0;
until the bound mass changes < 1e-4. Ignores the BH's tide (argued: small at the late
times used). Reports the bound mass fraction of the initial star.

    remnant.py RST [...] [--mstar 7.47e-5]
"""
import argparse, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import tde_dmde as T   # noqa: E402
GAM = 5.0 / 3.0


def bound_mass(c, itmax=50):
    rho, dM = c['rho'], c['dM']
    sel = rho > 1e-3 * rho.max()
    Mb = dM[sel].sum(); X = np.stack([c['x'], c['y'], c['z']]); V = np.stack([c['vx'], c['vy'], c['vz']])
    eps = c['P'] / ((GAM - 1.0) * rho)
    for _ in range(itmax):
        w = dM * sel
        xc = (X * w).sum(1) / w.sum(); vc = (V * w).sum(1) / w.sum()
        r = np.linalg.norm(X - xc[:, None], axis=0)
        o = np.argsort(r); rs = r[o]; ms = dM[o]
        Min = np.cumsum(ms); outer = np.cumsum((ms / np.maximum(rs, 1e-6))[::-1])[::-1] - ms / np.maximum(rs, 1e-6)
        phi = np.empty_like(r); phi[o] = -Min / np.maximum(rs, 1e-6) - outer
        e = 0.5 * ((V - vc[:, None])**2).sum(0) + eps + phi
        new = e < 0
        Mn = dM[new].sum()
        if abs(Mn - Mb) < 1e-4 * max(Mb, 1e-300):
            sel = new; Mb = Mn; break
        sel = new; Mb = Mn
    return Mb, xc, vc, sel


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('rst', nargs='+'); ap.add_argument('--mstar', type=float, default=7.47e-5)
    a = ap.parse_args()
    for fn in a.rst:
        c = T.cells(fn, rcut=3.0)
        Mb, xc, vc, sel = bound_mass(c)
        rb = np.hypot(xc[0], xc[1])
        K0 = 1.2346626297e-2
        k = c['P'][sel] / c['rho'][sel]**GAM / K0 - 1; m = c['dM'][sel]
        print(f'{os.path.basename(fn)}: t={c["t"]:6.1f} bound remnant {Mb / a.mstar:.4f} M* at r_BH={rb:.1f}, '
              f'|v_c|={np.linalg.norm(vc):.3f}; remnant K/K0-1: mass-weighted mean {np.average(k, weights=m):+.4f}, '
              f'warm>5% {m[k > 0.05].sum() / m.sum():.3f}, cold<-5% {m[k < -0.05].sum() / m.sum():.3f}', flush=True)


if __name__ == '__main__':
    main()
