#!/usr/bin/env python3
"""c159 (R-188 item 3, R-187 sec. 3): the lapse profile's curvature per AREAL-radius shell for the two slicings, in the coordinates
each code differences (M = 1; analytic, no run data):
  trumpet (CFC puncture background, the B.2 runs): alpha0(R) = sqrt(1 - 2/R + C^2/R^4), C = 3 sqrt(3)/4, on the isotropic grid r_iso = chi(R)
    (c122_b23.py chi, TrumpetArealToIso); d/dr_iso = (dR/dr_iso) d/dR, dR/dr_iso = R alpha0/r_iso;
  Kerr-Schild (a = 0, Step A/B): alpha(r) = (1 + 2/r)^(-1/2), r_KS = R.
For a radial f(r) on a Cartesian grid the Hessian d_i d_j f has eigenvalues f'' (radial) and f'/r (twice): |d^2 alpha| = sqrt(f''^2 +
2 (f'/r)^2) (Frobenius, |.|_coord, per M^2). Shell values: the mean over the shell in areal R weighted by the coordinate volume
(r^2 dr of the grid's own radius), and the radial part f'' alone. Also the gradient |d alpha| = |f'|.
    c159_d2alpha.py OUT.csv [R edges, default the Step A/B areal shells]"""
import sys

import numpy as np
from scipy.optimize import brentq

C = 0.75*np.sqrt(3.0)
EDGES = [1.6, 2.0, 2.2, 2.65, 3.1, 4.1, 6.0, 11.0, 21.0, 39.0]


def chi(R):    # areal -> isotropic (M = 1), TrumpetArealToIso
    t = 2.0*R - 3.0
    s1 = np.sqrt(4*R*R + 4*R + 3); s2 = np.sqrt(8*R*R + 8*R + 6)
    return 0.25*(2*R + 1 + s1)*((4 + 3*np.sqrt(2))*t/(8*R + 6 + 3*s2))**(1/np.sqrt(2))


def slicing(name, R):
    """(r_grid, alpha, f', f'') on the grid radius for the areal radii R"""
    if name == 'KS':
        r = R.copy()
        a = (1 + 2/r)**-0.5
    else:
        r = chi(R)
        a = np.sqrt(np.maximum(1 - 2/R + C*C/R**4, 0.0))
    # derivatives in the grid radius by finite differences on a fine log grid (smooth for R > 1.5)
    lr = np.log(r)
    da = np.gradient(a, lr)/r                       # da/dr
    d2a = np.gradient(da, lr)/r                     # d2a/dr2
    return r, a, da, d2a


def main():
    out = sys.argv[1]
    edges = [float(v) for v in sys.argv[2].split(',')] if len(sys.argv) > 2 else EDGES
    R = np.geomspace(1.5 + 1e-6, 60.0, 400001)
    rows = ['slicing,shell_R,R_lo,R_hi,r_lo,r_hi,alpha_mean,abs_dalpha,abs_d2alpha,abs_d2alpha_radial,abs_dalpha_over_r']
    for name in ('trumpet', 'KS'):
        r, a, da, d2a = slicing(name, R)
        h = np.sqrt(d2a**2 + 2*(da/r)**2)
        w = r*r*np.gradient(r)                      # coordinate volume per sample
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (R >= lo) & (R < hi)
            if not m.any():
                continue
            wm = w[m]/w[m].sum()
            rows.append(f'{name},R{lo:g}-{hi:g},{lo:g},{hi:g},{r[m].min():.4f},{r[m].max():.4f},{(a[m]*wm).sum():.4f},'
                        f'{(np.abs(da[m])*wm).sum():.4e},{(h[m]*wm).sum():.4e},{(np.abs(d2a[m])*wm).sum():.4e},'
                        f'{(np.abs(da[m]/r[m])*wm).sum():.4e}')
    open(out, 'w').write('\n'.join(rows) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
