#!/usr/bin/env python3
"""c254 (R-254 items 2-3, user-approved 2026-11-09): debris + remnant initial data for the test-particle fallback integration,
from one restart (tde_dmde.cells: BH-frame Killing E = -u_t - 1 with xi-dot removed, covariant u_i, dM = sqrt(gamma) D dV).
  remnant: remnant.bound_mass (the energy-criterion self-bound set; c142's definition): M_rem, centroid x_c, and its mean
           (mass-weighted) Killing energy and angular-momentum vector; R_rem = the radius about x_c enclosing 90% of M_rem;
           r_H = |x_c - x_BH| (M_rem/3)^(1/3) (M_BH = 1).
  debris:  cells outside the Hill sphere (|x - x_c| > r_H), r_BH >= 3 (rcut), rho > 1e3 x the atmosphere (1e-10 rho_c, as
           c206_outflow), E/dE < 1 (strongly unbound gas cannot return; dE = R*/r_t^2).
  Per debris cell: position relative to the hole in grid (isotropic-like) coordinates, eps = E + 1, the angular-momentum vector
  L_i = eps_ijk x^j u_k (covariant u_k: coordinate-independent under the radial relabelling below), the radial sign s = sign(x.u),
  dM. The integrator maps r_iso -> r_areal = r_iso (1 + M/2 r_iso)^2 (Schwarzschild isotropic; the CFC puncture background agrees
  with it to O(M/r) at the debris' r >= 30 M, ARGUED) and rebuilds dx/dtau from (eps, L, r, s).
    c254tp_extract.py RST OUT.npz [--rcut 3]"""
import argparse, os, sys
import numpy as np
HERE = '/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde/wt_tde_scripts/scripts'
sys.path.insert(0, HERE)
import tde_dmde as T   # noqa: E402
import remnant as RM   # noqa: E402

RC, MS, RS, RT = 3.24e-4, 7.47e-5, 0.677, 16.4
DE = RS/RT**2
ap = argparse.ArgumentParser(); ap.add_argument('rst'); ap.add_argument('out'); ap.add_argument('--rcut', type=float, default=3.0)
a = ap.parse_args()
c = T.cells(a.rst, rcut=a.rcut)
xb = np.asarray(c['xbh'], float)
Mb, xc, vc, sel = RM.bound_mass(c)
X = np.stack([c['x'], c['y'], c['z']]) - xb[:, None]
U = np.stack([c['ux'], c['uy'], c['uz']])
eps = c['E'] + 1.0
Lv = np.cross(X.T, U.T).T
dM = c['dM']
xcr = xc - xb
r_bh = np.linalg.norm(xcr)
rH = r_bh*(Mb/3.0)**(1.0/3.0)
d = np.linalg.norm(X - xcr[:, None], axis=0)
o = np.argsort(d[sel]); cm = np.cumsum(dM[sel][o]); R90 = float(d[sel][o][np.searchsorted(cm, 0.9*cm[-1])])
w = dM*sel
rem = dict(M=Mb, x=xcr, eps=float((eps*w).sum()/w.sum()), L=(Lv*w).sum(1)/w.sum(), s=float(np.sign(((X*U).sum(0)*w).sum())),
           rH=rH, R90=R90, r=r_bh)
keep = (d > rH) & (c['rho'] > 1e3*1e-10*RC) & (c['E']/DE < 1.0) & ~sel
s = np.sign((X*U).sum(0))
np.savez(a.out, t=c['t'], X=X[:, keep], eps=eps[keep], L=Lv[:, keep], s=s[keep], dM=dM[keep], E=c['E'][keep],
         rem_M=Mb, rem_x=xcr, rem_eps=rem['eps'], rem_L=rem['L'], rem_s=rem['s'], rem_rH=rH, rem_R90=R90,
         M_bound_all=float(dM[(c['E'] < 0) & ~sel & (d > rH)].sum()), M_debris=float(dM[keep].sum()), rst=a.rst)
print(f"{os.path.basename(a.rst)} t={c['t']:.2f}: M_rem={Mb/MS:.4f} M* at r={r_bh:.1f} (eps={rem['eps']:.6f}, E/dE={(rem['eps']-1)/DE:+.3f}), "
      f"r_H={rH:.2f}, R90={R90:.2f}; debris cells {keep.sum()} M={dM[keep].sum()/MS:.4f} M*, bound {dM[keep & (c['E'] < 0)].sum()/MS:.4f} M*")
