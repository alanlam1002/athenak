#!/usr/bin/env python3
"""c298 (R-277, user-approved 2026-11-22): the per-block time-step census of a TDE run from one restart (offline).

Per cell, the code's own CFL limit (src/dyn_grmhd/dyn_grmhd_newdt.cpp:158-258, <time> gr_dt = true): for each direction i,
  lambda_pm = alpha/(1 - v^2 c_s^2) [v^i (1 - c_s^2) pm c_s sqrt((1 - v^2)(g^ii (1 - v^2 c_s^2) - v^i v^i (1 - c_s^2)))] - beta^i
(GetGRFastMagnetosonicSpeeds with b^2 = 0: this run's B = 0), dt_cell = min_i dx_i / max(|lambda_-|, lambda_+), with the
stored metric (gamma_ij, alpha, beta^i: the restart's ADM slots, which carry the comoving frame's xidot in beta) and the
primitives recovered from the stored conserved state (tde_dmde.c2p, ideal gas Gamma = 5/3: c_s^2 = Gamma P/(rho h)).
Excised cells are skipped as with <time> dt_skip_excised (:150): alpha < excise_lapse OR r_BH < r_f (excision.cpp:193;
r_f = <coord> excision_rf_state from the restart's parameter copy). The global dt = cfl * min over cells.
Per block: dt_b = cfl * min over its cells, its level (dx), and a region from its centre:
  hole  : r_BH(centre) < --rhole (default 8 = the zone radius 6 + one shell);
  core / shell / surface : d_star < 0.2 / 0.8 / 1.5 R*, d_star = the distance from the star's centre (the centroid of
    rho >= 1e-2 rho_c) to the block's AABB (blocks are 0.5 M wide at dx 1/32, so a centre test would leave the core empty);
  debris: the block's max rho >= 1e-6 rho_c and d_star >= 1.5 R*;
  atm   : everything else (the atmosphere, the ceiling band and the floor).
Outputs: per region x level the block count and the dt_b quantiles (in units of dt_min); the limiting blocks (the 10 smallest
dt_b) with their region, level, r_BH, and the limiting cell's c_s, |v|, alpha, |beta|, and its coordinate speed;
the LTS gains (c255's definitions): per-block (power of 2 multiples of dt_min, capped at 2^kmax), per-level (Berger-Oliger:
from the finest level up, dt'_L = dt_min 2^k <= min(dt_L, 2 dt'_{L+1})), each also with the hole region excluded (its blocks
moved to their own dt as if the hole were not there: the gain if the near-hole tier did not set dt_min), for each
--rhole_ex radius (default 8,16). Excluding blocks is NOT a coarse hole: coarsening also changes dx in the shells around it.
    c298_dt_census.py RST [--cfl 0.4] [--kmax 3] [--rhole 8] [--rstar 0.677] [--out PREFIX]"""
import argparse, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tde_dmde as T   # noqa: E402

GAM, RC = 5.0/3.0, 3.24e-4
ap = argparse.ArgumentParser()
ap.add_argument('rst'); ap.add_argument('--cfl', type=float, default=0.4); ap.add_argument('--kmax', type=int, default=3)
ap.add_argument('--rhole', type=float, default=8.0); ap.add_argument('--rstar', type=float, default=0.677)
ap.add_argument('--rhole_ex', default='8,16'); ap.add_argument('--out', default=None); ap.add_argument('--nscal', type=int, default=1)
a = ap.parse_args()

t = T.header(a.rst)
par = t['par']
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

def speeds(alpha, beta, vi, v2, cs2, gii):
    disc = np.maximum((1 - v2)*(gii*(1 - v2*cs2) - vi*vi*(1 - cs2)), 0.0)
    den = 1.0 - v2*cs2
    lp = alpha/den*(vi*(1 - cs2) + np.sqrt(cs2*disc)) - beta
    lm = alpha/den*(vi*(1 - cs2) - np.sqrt(cs2*disc)) - beta
    return np.maximum(np.abs(lm), lp)

B = dict(dt=np.full(nmb, np.inf), lev=lev - root, dx=np.zeros(nmb), cen=np.zeros((nmb, 3)), rhomax=np.zeros(nmb),
         lim=[None]*nmb)
cx = np.zeros(3); cm = 0.0
for b in range(nmb):
    rec = mm[b]
    cons = rec[:nmhd*ncc].reshape(nmhd, no, no, no)[:, s, s, s]
    adm = rec[nmhd*ncc + bsz:nmhd*ncc + bsz + 17*ncc].reshape(17, no, no, no)[:, s, s, s]
    ext = w/2.0**lev[b]; dx = ext/nx
    x = lo[0] + ll[b, 0]*ext[0] + ic*dx[0]; y = lo[1] + ll[b, 1]*ext[1] + ic*dx[1]; z = lo[2] + ll[b, 2]*ext[2] + ic*dx[2]
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    B['dx'][b] = dx[0]; B['cen'][b] = [x.mean(), y.mean(), z.mean()]
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
    v2 = 1.0 - 1.0/(W*W)
    cs2 = GAM*P/(rho*h)
    al = alpha[ok]; be = [bx[ok], by[ok], bz[ok]]
    dtc = np.minimum.reduce([dx[i]/speeds(al, be[i], v[i], v2, cs2, gi[i][i]) for i in range(3)])
    k = np.argmin(dtc)
    B['dt'][b] = a.cfl*dtc[k]; B['rhomax'][b] = rho.max()
    vc = [al[k]*v[i][k] - be[i][k] for i in range(3)]   # coordinate velocity of the gas (Eulerian v -> grid)
    B['lim'][b] = dict(cs=np.sqrt(cs2[k]), v=np.sqrt(v2[k]), alpha=al[k], beta=np.sqrt(sum(be[i][k]**2 for i in range(3))),
                       vcoord=np.sqrt(sum(c*c for c in vc)), rho=rho[k], psi4=adm[12][ok][k],
                       x=(X[ok][k], Y[ok][k], Z[ok][k]), rbh=rbh[ok][k])
    hi = rho >= 1e-2*RC
    if hi.any():
        wgt = rho[hi]*dx.prod(); cm += wgt.sum()
        cx += np.array([(X[ok][hi]*wgt).sum(), (Y[ok][hi]*wgt).sum(), (Z[ok][hi]*wgt).sum()])
xs = cx/cm if cm > 0 else np.zeros(3)
hw = np.stack([B['dx']*nx/2]*3, 1)   # block half-widths (cubic blocks)
ds = np.linalg.norm(np.maximum(np.abs(B['cen'] - xs) - hw, 0.0), axis=1)/a.rstar
dh = np.linalg.norm(B['cen'] - xbh, axis=1)
reg = np.where(dh < a.rhole, 'hole', np.where(ds < 0.2, 'core', np.where(ds < 0.8, 'shell', np.where(ds < 1.5, 'surface',
               np.where(B['rhomax'] >= 1e-6*RC, 'debris', 'atm')))))
good = np.isfinite(B['dt'])
dtb = B['dt']; dtmin = dtb[good].min()
L = sorted(set(B['lev'][good]))
lines = [f"{a.rst}", f"t = {t['time']:.4f}, cycle {t['ncycle']}, blocks {nmb} ({good.sum()} with unexcised cells), x_BH {np.round(xbh, 3)}, "
         f"x_star {np.round(xs, 3)}, |x_star - x_BH| {np.linalg.norm(xs - xbh):.3f}, r_f {rf}, xidot {np.round(xdot, 5)}",
         f"global dt (cfl {a.cfl}) = {dtmin:.5e}; the restart's own dt = {t['dt']:.5e} (ratio {t['dt']/dtmin:.4f})", "",
         "region  level  dx       blocks   dt_b/dt_min: min    q10    q50    q90"]
for r in ('hole', 'core', 'shell', 'surface', 'debris', 'atm'):
    for l in L:
        sel = good & (reg == r) & (B['lev'] == l)
        if not sel.any(): continue
        q = dtb[sel]/dtmin
        lines.append(f"{r:8s} {l:3d}  {B['dx'][sel][0]:.5f} {sel.sum():7d}   {q.min():8.2f} {np.quantile(q, .1):6.2f} "
                     f"{np.quantile(q, .5):6.2f} {np.quantile(q, .9):6.2f}")
lines += ["", "limiting blocks (smallest dt_b):",
          " dt_b/dt_min region   lev  r_BH(cell)  rho/rho_c    c_s      |v|     alpha   |beta|  |v_coord|  psi4"]
for b in np.argsort(np.where(good, dtb, np.inf))[:10]:
    m = B['lim'][b]
    lines.append(f"   {dtb[b]/dtmin:7.3f}   {reg[b]:8s} {B['lev'][b]:3d}  {m['rbh']:9.3f}  {m['rho']/RC:10.3e} {m['cs']:8.5f} "
                 f"{m['v']:8.5f} {m['alpha']:7.4f} {m['beta']:7.4f} {m['vcoord']:8.5f} {m['psi4']:7.3f}")

def gains(sel_all, dmin):
    """global vs per-block vs per-level costs over the blocks in sel_all, dt_min = dmin"""
    d = dtb[sel_all]; lv = B['lev'][sel_all]
    cost_glob = len(d)/dmin
    kb = np.clip(np.floor(np.log2(d/dmin) + 1e-12), 0, a.kmax)
    cost_blk = (1.0/(dmin*2.0**kb)).sum()
    step = {}; prev = None
    for l in sorted(set(lv), reverse=True):
        cap = d[lv == l].min() if prev is None else min(d[lv == l].min(), 2*prev)
        k = min(max(int(np.floor(np.log2(cap/dmin) + 1e-12)), 0), a.kmax); step[l] = dmin*2.0**k; prev = step[l]
    cost_lvl = sum((lv == l).sum()/step[l] for l in step)
    return cost_glob/cost_blk, cost_glob/cost_lvl, {l: step[l]/dmin for l in sorted(step)}, np.bincount(kb.astype(int), minlength=a.kmax + 1)

gb, gl, st, kb = gains(good, dtmin)
lines += ["", f"LTS (k_max {a.kmax}): per-block ideal gain {gb:.2f}, per-level {gl:.2f}; blocks per k {list(map(int, kb))}; "
          f"level steps {dict((int(k), float(v)) for k, v in st.items())}",
          f"realistic (L-011 calibration S_lb/S = 1.40/2.40 = 0.583 on the current partition): per-block {gb*0.583:.2f}"]
for rx in [float(v) for v in a.rhole_ex.split(',')]:
    nh = good & (dh >= rx); dmin_nh = dtb[nh].min()
    gb2, gl2, st2, kb2 = gains(nh, dmin_nh)
    lines.append(f"without the blocks within {rx:g} M of the hole ({(good & ~nh).sum()} blocks): dt_min x{dmin_nh/dtmin:.2f}, "
                 f"per-block LTS on top {gb2:.2f}, per-level {gl2:.2f}; blocks per k {list(map(int, kb2))}")
txt = '\n'.join(lines)
print(txt)
if a.out:
    open(a.out + '.txt', 'w').write(txt + '\n')
    np.savez(a.out + '.npz', dt=dtb, lev=B['lev'], dx=B['dx'], cen=B['cen'], reg=reg, rhomax=B['rhomax'], xbh=xbh, xs=xs,
             time=t['time'], dtmin=dtmin)
