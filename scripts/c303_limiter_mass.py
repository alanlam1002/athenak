#!/usr/bin/env python3
"""c303 (R-278 addendum item 2): the mass and state of the gas that sets the global dt near the hole.
Per cell, in every block whose centre lies within --rblk M of the hole, the code's own GR CFL rule (as c298_dt_census.py,
the same speeds(): GetGRFastMagnetosonicSpeeds' form, excised cells skipped as dyn_grmhd_newdt.cpp:150), dt_cell = cfl * min_i
dx_i / lambda_i. The global dt_min is taken over ALL blocks (as c298). Reported:
  - the conserved mass (cons[0] dV = D sqrt(gamma) dV) of the near-hole cells with dt_cell < f dt_min, f = 1.05, 1.25, 1.5, 2, 4;
    its mass-weighted rho, c_s, |v|, |v_coord| and r_BH, and the r_BH range; in M (M_BH = 1) and in M* (7.47e-5)
  - the total mass of the near-hole region by r_BH shell (unexcised cells), for scale
  - dt_min over all blocks with those cells removed (the gain of a cold reset that took exactly them out of the CFL)
  - dt_min with every unexcised cell inside r_BH < R removed, R = 6 (the zone radius), 7, 8 (a larger excision)
Offline, read-only.     c303_limiter_mass.py RST [--rblk 12] [--cfl 0.4] [--out PREFIX]"""
import argparse, re, sys
import numpy as np
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde/wt_tde_scripts/scripts')
import tde_dmde as T

GAM = 5.0/3.0; RC = 3.24e-4; MSTAR = 7.47e-5
ap = argparse.ArgumentParser()
ap.add_argument('rst'); ap.add_argument('--cfl', type=float, default=0.4); ap.add_argument('--rblk', type=float, default=12.0)
ap.add_argument('--nscal', type=int, default=1); ap.add_argument('--out', default=None)
a = ap.parse_args()

t = T.header(a.rst); par = t['par']
def pget(key, default):
    m = re.search(rf'^{key}\s*=\s*([-+.\deE]+)', par, re.M)
    return float(m.group(1)) if m else default
exc_lapse = pget('excise_lapse', 0.25); rf = pget('excision_rf_state', 0.0)
nmb, root = t['nmb'], t['root']
x1min, x2min, x3min, x1max, x2max, x3max = t['size'][:6]
nx, ng = 16, 4; no = nx + 2*ng; nmhd = 5 + a.nscal; ncc = no**3; bsz = (no + 1)*no*no*3
mm = np.memmap(a.rst, dtype=np.float64, mode='r', offset=t['data_off'], shape=(nmb, t['dsize']//8))
ll, lev = t['ll'], t['ll'][:, 3]
w = np.array([x1max - x1min, x2max - x2min, x3max - x3min]); lo = np.array([x1min, x2min, x3min])
xbh, xdot = T.frame_state(par, t['time'])
s = slice(ng, ng + nx); ic = np.arange(nx) + 0.5

def speeds(alpha, beta, vi, v2, cs2, gii):   # identical to c298_dt_census.py
    disc = np.maximum((1 - v2)*(gii*(1 - v2*cs2) - vi*vi*(1 - cs2)), 0.0)
    den = 1.0 - v2*cs2
    lp = alpha/den*(vi*(1 - cs2) + np.sqrt(cs2*disc)) - beta
    lm = alpha/den*(vi*(1 - cs2) - np.sqrt(cs2*disc)) - beta
    return np.maximum(np.abs(lm), lp)

far_min = np.inf        # min dt over blocks NOT near the hole
near = []               # per-cell arrays of the near-hole blocks
for b in range(nmb):
    ext = w/2.0**lev[b]; dx = ext/nx
    cen = lo + ll[b, :3]*ext + ext/2
    isnear = np.linalg.norm(cen - xbh) < a.rblk
    rec = mm[b]
    cons = rec[:nmhd*ncc].reshape(nmhd, no, no, no)[:, s, s, s]
    adm = rec[nmhd*ncc + bsz:nmhd*ncc + bsz + 17*ncc].reshape(17, no, no, no)[:, s, s, s]
    x = lo[0] + ll[b, 0]*ext[0] + ic*dx[0]; y = lo[1] + ll[b, 1]*ext[1] + ic*dx[1]; z = lo[2] + ll[b, 2]*ext[2] + ic*dx[2]
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    gxx, gxy, gxz, gyy, gyz, gzz = adm[:6]
    det = gxx*(gyy*gzz - gyz*gyz) - gxy*(gxy*gzz - gyz*gxz) + gxz*(gxy*gyz - gyy*gxz)
    sg = np.sqrt(det)
    ixx = (gyy*gzz - gyz*gyz)/det; ixy = (gxz*gyz - gxy*gzz)/det; ixz = (gxy*gyz - gxz*gyy)/det
    iyy = (gxx*gzz - gxz*gxz)/det; iyz = (gxy*gxz - gxx*gyz)/det; izz = (gxx*gyy - gxy*gxy)/det
    D = cons[0]/sg; Sx, Sy, Sz = cons[1]/sg, cons[2]/sg, cons[3]/sg; tau = np.maximum(cons[4]/sg, 0.0)
    Ssq = ixx*Sx*Sx + iyy*Sy*Sy + izz*Sz*Sz + 2*(ixy*Sx*Sy + ixz*Sx*Sz + iyz*Sy*Sz)
    alpha = adm[13]; bx, by, bz = adm[14], adm[15], adm[16]
    rbh = np.sqrt((X - xbh[0])**2 + (Y - xbh[1])**2 + (Z - xbh[2])**2)
    ok = (D > 0) & np.isfinite(tau) & ~((alpha < exc_lapse) | (rbh < rf))
    if not ok.any():
        continue
    rho, P, W, h = T.c2p(D[ok], Ssq[ok], tau[ok])
    uj = [Sx[ok]/(D[ok]*h), Sy[ok]/(D[ok]*h), Sz[ok]/(D[ok]*h)]
    gi = [[ixx[ok], ixy[ok], ixz[ok]], [ixy[ok], iyy[ok], iyz[ok]], [ixz[ok], iyz[ok], izz[ok]]]
    v = [sum(gi[i][j]*uj[j] for j in range(3))/W for i in range(3)]
    v2 = 1.0 - 1.0/(W*W); cs2 = GAM*P/(rho*h)
    al = alpha[ok]; be = [bx[ok], by[ok], bz[ok]]
    dtc = a.cfl*np.minimum.reduce([dx[i]/speeds(al, be[i], v[i], v2, cs2, gi[i][i]) for i in range(3)])
    if not isnear:
        far_min = min(far_min, dtc.min()); continue
    vc = np.sqrt(sum((al*v[i] - be[i])**2 for i in range(3)))
    near.append(dict(dt=dtc, m=cons[0][ok]*dx.prod(), rho=rho, cs=np.sqrt(cs2), v=np.sqrt(v2), vc=vc, r=rbh[ok], lev=np.full(ok.sum(), lev[b] - root)))

N = {k: np.concatenate([c[k] for c in near]) for k in near[0]}
dtmin = min(far_min, N['dt'].min())
L = [a.rst, f"t = {t['time']:.4f}, cycle {t['ncycle']}, x_BH {np.round(xbh, 3)}, r_f {rf}, excise_lapse {exc_lapse}, cfl {a.cfl}; "
     f"near-hole blocks: centre within {a.rblk:g} M ({len(near)} blocks with unexcised cells, {N['dt'].size} cells)",
     f"global dt_min = {dtmin:.5e} (near-hole cells {N['dt'].min():.5e}; all other blocks {far_min:.5e}, ratio {far_min/dtmin:.3f})", "",
     "near-hole cells with dt_cell < f dt_min:",
     "   f    cells   mass [M]     mass [M*]   <rho>/rho_c  <c_s>   <|v|>   <|v_coord|>  <r_BH>  r_BH range     levels   dt_min if removed (x)"]
for f in (1.05, 1.25, 1.5, 2.0, 4.0):
    sel = N['dt'] < f*dtmin
    m = N['m'][sel]; M = m.sum()
    wav = lambda q: (q[sel]*m).sum()/M if M > 0 else np.nan
    rest = min(far_min, N['dt'][~sel].min()) if (~sel).any() else far_min
    L.append(f"  {f:4.2f} {sel.sum():7d}  {M:10.3e}  {M/MSTAR:10.3e}  {wav(N['rho'])/RC:10.3e}  {wav(N['cs']):6.3f}  {wav(N['v']):6.3f}  "
             f"{wav(N['vc']):8.3f}   {wav(N['r']):6.2f}  {N['r'][sel].min():5.2f}-{N['r'][sel].max():5.2f}  "
             f"{sorted(set(N['lev'][sel].tolist()))}   {rest/dtmin:.3f}")
L += ["", "near-hole mass by r_BH shell (unexcised cells):", "   shell        mass [M]     mass [M*]    <rho>/rho_c   min dt_cell/dt_min"]
for r0, r1 in ((0, 3), (3, 5), (5, 6), (6, 7), (7, 8), (8, 10), (10, 12)):
    sel = (N['r'] >= r0) & (N['r'] < r1)
    if not sel.any(): continue
    M = N['m'][sel].sum()
    L.append(f"   {r0:4.1f}-{r1:4.1f}  {M:10.3e}  {M/MSTAR:10.3e}  {(N['rho'][sel]*N['m'][sel]).sum()/M/RC:10.3e}   {N['dt'][sel].min()/dtmin:8.3f}")
L += ["", "dt_min with every unexcised cell inside r_BH < R removed (a larger excision; mass removed):"]
for R in (6.0, 6.5, 7.0, 8.0):
    sel = N['r'] < R
    rest = min(far_min, N['dt'][~sel].min()) if (~sel).any() else far_min
    L.append(f"   R = {R:3.1f}: dt_min x{rest/dtmin:.3f}; mass inside {N['m'][sel].sum():.3e} M = {N['m'][sel].sum()/MSTAR:.3e} M*")
L += ["", "dt_min with a cold grid-rest reset on r_s <= r_BH < R_c and rho < rho_c3 (tier 3's state, D kept; the reset cells taken",
      "out of the CFL, their reset speed alpha c_s(T_floor) being negligible; r_s = 6):"]
for Rc in (9.0, 12.0):
    for th in (1e-16, 1e-14, 1e-12, 1e-10):
        sel = (N['r'] >= 6.0) & (N['r'] < Rc) & (N['rho'] < th*RC)
        rest = min(far_min, N['dt'][~sel].min()) if (~sel).any() else far_min
        L.append(f"   R_c = {Rc:4.1f}, rho_c3 = {th:.0e} rho_c: dt_min x{rest/dtmin:.3f}; cells {sel.sum()}, mass {N['m'][sel].sum()/MSTAR:.3e} M*")
txt = '\n'.join(L); print(txt)
if a.out:
    open(a.out + '.txt', 'w').write(txt + '\n')
