#!/usr/bin/env python3
"""R-147 item 4 (TDE_DEBRIS_SPEC §8): the Marck-tensor checklist against the table that rung 2 itself wrote
(<cfc> tidal_table_file: tau t r angle Exx Exy Exz Eyy Eyz Ezz, every 10th row; cfc_tidal.cpp:146-151).
Reference (Kesden 2012 eq. 9 from Marck 1983, a = 0, E = 1, K = L^2 = 2 M r_p^2/(r_p - 2M); frame 1, 3 in the
orbital plane, 2 normal; r areal):
  C11 = g[1 - 3c cos^2 Psi], C13 = -3 c g cos Psi sin Psi, C33 = g[1 - 3c sin^2 Psi], C22 = g(1 + 3K/r^2),
  g = M/r^3, c = (r^2 + K)/r^2; dPsi/dtau = E L/(r^2 + L^2).
Checks (each prints its max deviation):
  (i)   every component, at every table row, against the formulas with (r, Psi) = (table r, table angle), with
        frame (1, 3, 2) -> grid (x, y, z); plus the eigenvalues against M/r^3, (1 + 3K/r^2)M/r^3,
        -2(1 + 3K/2r^2)M/r^3, and the trace;
  (ii)  the Psi law in proper time: d(angle)/d(tau) from the table (centred differences) against E L/(r^2 + L^2);
  (iii) the time variable: dt/dtau from the table against E/(1 - 2M/r), and the geodesic radial equation
        (dr/dtau)^2 = E^2 - (1 - 2M/r)(1 + L^2/r^2) (so r is the areal Schwarzschild radius, (iv));
  (v)   the sign: the radial eigenvalue is the negative one (stretching), z is the normal axis.
    c132_marck_check.py TABLE --rp 10 [--m 1]
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('table')
    ap.add_argument('--rp', type=float, default=10.0)
    ap.add_argument('--m', type=float, default=1.0)
    a = ap.parse_args()
    M, rp = a.m, a.rp
    T = np.loadtxt(a.table)
    tau, t, r, psi = T[:, 0], T[:, 1], T[:, 2], T[:, 3]
    Exx, Exy, Exz, Eyy, Eyz, Ezz = T[:, 4:10].T
    K = 2.0 * M * rp**2 / (rp - 2.0 * M); L = np.sqrt(K)
    g = M / r**3; c = (r**2 + K) / r**2
    C11 = g * (1 - 3 * c * np.cos(psi)**2); C13 = -3 * c * g * np.cos(psi) * np.sin(psi)
    C33 = g * (1 - 3 * c * np.sin(psi)**2); C22 = g * (1 + 3 * K / r**2)
    sc = np.abs(g * (2 + 3 * K / r**2))          # |stretching eigenvalue| as the scale
    print(f'table {a.table}: {len(T)} rows, tau 0..{tau[-1]:.1f}, r {r.min():.4f}..{r.max():.2f}; K = L^2 = {K:.6f}')
    print('(i) components vs Kesden eq. 9 (rel. to |lambda_str|): '
          f'xx {np.max(np.abs(Exx - C11) / sc):.2e}  xy {np.max(np.abs(Exy - C13) / sc):.2e}  '
          f'yy {np.max(np.abs(Eyy - C33) / sc):.2e}  zz {np.max(np.abs(Ezz - C22) / sc):.2e}  '
          f'xz {np.max(np.abs(Exz)):.1e}  yz {np.max(np.abs(Eyz)):.1e}')
    ev = np.array([np.linalg.eigvalsh(np.array([[Exx[k], Exy[k], Exz[k]], [Exy[k], Eyy[k], Eyz[k]],
                                                 [Exz[k], Eyz[k], Ezz[k]]])) for k in range(len(T))])
    ref = np.sort(np.stack([g, g * (1 + 3 * K / r**2), -2 * g * (1 + 1.5 * K / r**2)], 1), 1)
    print(f'    eigenvalues vs M/r^3, (1+3K/r^2)M/r^3, -2(1+3K/2r^2)M/r^3: {np.max(np.abs(ev - ref) / sc[:, None]):.2e};'
          f' trace {np.max(np.abs(Exx + Eyy + Ezz) / sc):.2e}')
    kp = np.argmin(r)
    print(f'    at periapsis (row {kp}): r = {r[kp]:.5f}, tau = {tau[kp]:.3f}, t = {t[kp]:.3f}; lambda_str = '
          f'{ev[kp, 0]:.6e} (Newtonian -2M/r^3 = {-2 * g[kp]:.6e}, ratio {ev[kp, 0] / (-2 * g[kp]):.4f}); '
          f'E_zz = {Ezz[kp]:.6e} (ratio to M/r^3 {Ezz[kp] / g[kp]:.4f})')
    for rr in (40.0, 20.0, 12.0):
        for side, sel in (('in', np.arange(len(T)) < kp), ('out', np.arange(len(T)) > kp)):
            k = np.where(sel)[0][np.argmin(np.abs(r[sel] - rr))]
            print(f'    sample r = {r[k]:7.3f} ({side:3s}) Psi = {psi[k]:+.4f}: Exx {Exx[k]:+.6e} vs {C11[k]:+.6e}, '
                  f'Exy {Exy[k]:+.6e} vs {C13[k]:+.6e}, Eyy {Eyy[k]:+.6e} vs {C33[k]:+.6e}, Ezz {Ezz[k]:+.6e} vs {C22[k]:+.6e}')
    i = slice(1, -1)
    dpsi = (psi[2:] - psi[:-2]) / (tau[2:] - tau[:-2])
    ref_psi = L / (r[i]**2 + L**2)
    dphi = L / r[i]**2
    print(f'(ii) dPsi/dtau vs E L/(r^2 + L^2): max rel {np.max(np.abs(dpsi / ref_psi - 1)):.2e} '
          f'(against the orbital dphi/dtau = L/r^2 instead: max rel {np.max(np.abs(dpsi / dphi - 1)):.2e}); '
          f'total swing Psi = {psi[-1] - psi[0]:.4f} rad over the table')
    dtdtau = (t[2:] - t[:-2]) / (tau[2:] - tau[:-2])
    print(f'(iii) dt/dtau vs E/(1 - 2M/r): max rel {np.max(np.abs(dtdtau * (1 - 2 * M / r[i]) - 1)):.2e}; '
          f'dt/dtau at periapsis {1 / (1 - 2 * M / r[kp]):.4f}')
    drdtau = (r[2:] - r[:-2]) / (tau[2:] - tau[:-2])
    lhs, rhs = drdtau**2, 1.0 - (1 - 2 * M / r[i]) * (1 + K / r[i]**2)
    w = np.abs(r[i] - r[kp]) > 0.5              # away from the turning point (centred differences)
    print(f'(iv) (dr/dtau)^2 vs E^2 - (1 - 2M/r)(1 + L^2/r^2) (areal Schwarzschild r): max |diff|/max(rhs) '
          f'{np.max(np.abs(lhs - rhs)[w]) / np.max(rhs):.2e}')
    print(f'(v) sign: the negative eigenvalue lies along the radial direction n = (cos Psi, sin Psi, 0): '
          f'max |n.E.n - lambda_str|/|lambda_str| = '
          f'{np.max(np.abs(Exx * np.cos(psi)**2 + 2 * Exy * np.cos(psi) * np.sin(psi) + Eyy * np.sin(psi)**2 - ev[:, 0]) / sc):.2e}; '
          f'z is the normal axis (Exz = Eyz = 0, Ezz = (1 + 3K/r^2) M/r^3 > 0)')


if __name__ == '__main__':
    main()
