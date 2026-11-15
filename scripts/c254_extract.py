#!/usr/bin/env python3
"""c254 (R-254 items 1, 3, user-approved 2026-11-09): per-cell debris state from a restart, for the ballistic fallback rate.
tde_dmde.cells (E = -u_t - 1, the BH-frame Killing energy with xi-dot removed; covariant u_i; cells with r_BH >= rcut = 3).
Debris = cells outside the self-consistent Hill sphere about the remnant (centroid of rho >= 0.01 rho_c, r_H = r (M_H/3)^(1/3)
iterated, as c253_dmde.py / c142's M_Hsc); for a full disruption M_H -> 0 and nothing is removed.
Saved (npz, bound cells E < 0 only, plus totals): E, dM, the angular momentum vector L = x_rel x u_cov (x_rel = x - x_BH, grid =
isotropic coordinates; for g_ij = psi^4 delta_ij the components of x x u_cov are the Killing angular momenta exactly), the
isotropic radius r_iso = |x_rel|, the radial sign s = sign(x_rel . u_cov), and rho.
    c254_extract.py OUT.npz RST"""
import sys
import numpy as np
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde/wt_tde_scripts/scripts')
import tde_dmde as T   # noqa: E402

RC = 3.24e-4
out, fn = sys.argv[1], sys.argv[2]
c = T.cells(fn, rcut=3.0)
E, dM, rho = c['E'], c['dM'], c['rho']
X = np.stack([c['x'], c['y'], c['z']]); xb = np.asarray(c['xbh'], float)
d1 = rho >= 0.01*RC
if d1.any():
    xc = (X[:, d1]*dM[d1]).sum(1)/dM[d1].sum(); MH = dM[d1].sum()
else:
    xc = X[:, np.argmax(rho)]; MH = 0.0
r = float(np.linalg.norm(xc - xb)); dc = np.linalg.norm(X - xc[:, None], axis=0); rH = 0.0
for _ in range(60):
    rH = r*(MH/3.0)**(1/3); MN = dM[dc < rH].sum()
    if abs(MN - MH) <= 1e-10*max(MH, 1e-300):
        break
    MH = MN
deb = dc >= rH
b = deb & (E < 0)
xr = X - xb[:, None]
U = np.stack([c['ux'], c['uy'], c['uz']])
L = np.cross(xr.T, U.T).T
riso = np.linalg.norm(xr, axis=0)
s = np.sign((xr*U).sum(0))
np.savez(out, t=c['t'], E=E[b], dM=dM[b], Lx=L[0, b], Ly=L[1, b], Lz=L[2, b], riso=riso[b], s=s[b], rho=rho[b],
         M_all=dM.sum(), M_deb=dM[deb].sum(), M_deb_bound=dM[b].sum(), M_H=MH, r_H=rH, r_rem=r, xbh=xb, xc=xc)
print(f'{fn}: t={c["t"]:.2f} cells {E.size} debris {deb.sum()} bound {b.sum()}  M_all {dM.sum():.4e} M_deb {dM[deb].sum():.4e} '
      f'bound {dM[b].sum():.4e}  M_H {MH:.4e} r_H {rH:.2f} r_rem {r:.2f}', flush=True)
