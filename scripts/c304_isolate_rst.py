#!/usr/bin/env python3
"""c304 (R-278 addendum item 6, user-approved 2026-11-24): make an isolated-remnant restart from a TDE restart.
Copies SRC to DST (the copy is the only file written) and, in DST, sets every cell (ghosts included, by its own position)
farther than --rcut from the remnant centroid x_c to the floor state at Eulerian rest:
  D = sqrt(gamma) rho_f, S_i = 0, tau = sqrt(gamma) P_f/(Gamma - 1) with rho_f = --rho_floor (default the deck's dfloor),
  P_f = rho_f T_f (T_f = the deck's tfloor), the passive scalars scaled with D (D Y kept at the cell's old Y; Y = 0 if D_old <= 0).
x_c = the mass centroid of rho > --frac rho_max (default 0.3: the frame controller's relative tracker, <cfc> frame_track_rel = 0.3 in L32T.par;
1e-3 (tier 3's dense gas) takes in the debris and shifts x_c by 0.85 M at t_p + 1000). The ADM/CFC fields and everything else are untouched; the
run's deck then turns the puncture off (<cfc> puncture_enabled = false) and the per-step CFC solve recomputes the metric.
The atm_ceiling resets the floor gas to a cold grid-rest atmosphere on its first C2P (rho_f < 100 dfloor).
Prints the mass kept/removed (conserved D sqrt(gamma) dV over interior cells) and the bound/total split inside rcut.
    c304_isolate_rst.py SRC DST [--rcut 5.0] [--rho_floor F] [--nscal 1] [--dry]"""
import argparse, os, re, shutil, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tde_dmde as T

RC = 3.24e-4; MSTAR = 7.47e-5; GAM = 5.0/3.0
ap = argparse.ArgumentParser()
ap.add_argument('src'); ap.add_argument('dst'); ap.add_argument('--rcut', type=float, default=5.0)
ap.add_argument('--rho_floor', type=float, default=None); ap.add_argument('--nscal', type=int, default=1)
ap.add_argument('--dry', action='store_true'); ap.add_argument('--frac', type=float, default=0.3)
a = ap.parse_args()

t = T.header(a.src); par = t['par']
def pget(key, default):
    m = re.search(rf'^{key}\s*=\s*([-+.\deE]+)', par, re.M)
    return float(m.group(1)) if m else default
dfloor = pget('dfloor', 1e-24); tfloor = pget('tfloor', 1e-22)
rho_f = a.rho_floor if a.rho_floor is not None else dfloor
P_f = rho_f*tfloor
nmb, root = t['nmb'], t['root']
x1min, x2min, x3min, x1max, x2max, x3max = t['size'][:6]
nx, ng = 16, 4; no = nx + 2*ng; nmhd = 5 + a.nscal; ncc = no**3; bsz = (no + 1)*no*no*3
ll, lev = t['ll'], t['ll'][:, 3]
w = np.array([x1max - x1min, x2max - x2min, x3max - x3min]); lo = np.array([x1min, x2min, x3min])
ic = np.arange(no) - ng + 0.5            # all cells incl. ghosts
si = slice(ng, ng + nx)

def geom(b):
    ext = w/2.0**lev[b]; dx = ext/nx
    x = lo[0] + ll[b, 0]*ext[0] + ic*dx[0]; y = lo[1] + ll[b, 1]*ext[1] + ic*dx[1]; z = lo[2] + ll[b, 2]*ext[2] + ic*dx[2]
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    return X, Y, Z, dx

def sqrtg(rec):
    adm = rec[nmhd*ncc + bsz:nmhd*ncc + bsz + 6*ncc].reshape(6, no, no, no)
    gxx, gxy, gxz, gyy, gyz, gzz = adm
    return np.sqrt(gxx*(gyy*gzz - gyz*gyz) - gxy*(gxy*gzz - gyz*gxz) + gxz*(gxy*gyz - gyy*gxz))

src = np.memmap(a.src, dtype=np.float64, mode='r', offset=t['data_off'], shape=(nmb, t['dsize']//8))
# pass 1: rho_max; pass 2: the centroid of rho > frac rho_max (the relative tracker's definition;
# D / sqrt(gamma) as the density proxy). An absolute cut fails late: rho_max = 0.0068 rho_c at t_p + 1000.
rmax = 0.0
for b in range(nmb):
    rec = src[b]; D = rec[:ncc].reshape(no, no, no)[si, si, si]
    rmax = max(rmax, (D/sqrtg(rec)[si, si, si]).max())
cx = np.zeros(3); cm = 0.0
for b in range(nmb):
    rec = src[b]; D = rec[:ncc].reshape(no, no, no)[si, si, si]
    sg = sqrtg(rec)[si, si, si]; rho = D/sg
    hi = rho > a.frac*rmax
    if hi.any():
        X, Y, Z, dx = geom(b); wgt = (D*dx.prod())[hi]; cm += wgt.sum()
        cx += [(X[si, si, si][hi]*wgt).sum(), (Y[si, si, si][hi]*wgt).sum(), (Z[si, si, si][hi]*wgt).sum()]
xc = cx/cm
print(f'{a.src}: t = {t["time"]:.4f}, cycle {t["ncycle"]}, {nmb} blocks; x_c = {np.round(xc, 5)} (rho_max {rmax/RC:.4g} rho_c; rho > {a.frac:g} rho_max mass {cm/MSTAR:.4f} M*)')
print(f'rcut = {a.rcut} M; floor state rho_f = {rho_f:.3e}, P_f = {P_f:.3e} (dfloor {dfloor:.1e}, tfloor {tfloor:.1e})')
if not a.dry:
    os.makedirs(os.path.dirname(os.path.abspath(a.dst)), exist_ok=True)
    subprocess.run(['lfs', 'setstripe', '-S', '4m', '-c', '16', os.path.dirname(os.path.abspath(a.dst))], check=False)
    shutil.copyfile(a.src, a.dst)
    dst = np.memmap(a.dst, dtype=np.float64, mode='r+', offset=t['data_off'], shape=(nmb, t['dsize']//8))
Mkeep = Mcut = 0.0; ncut_blk = 0
for b in range(nmb):
    X, Y, Z, dx = geom(b)
    r = np.sqrt((X - xc[0])**2 + (Y - xc[1])**2 + (Z - xc[2])**2)
    out = r > a.rcut
    rec = src[b]; D = rec[:ncc].reshape(no, no, no)
    Dint = D[si, si, si]*dx.prod()
    Mkeep += Dint[~out[si, si, si]].sum(); Mcut += Dint[out[si, si, si]].sum()
    if not out.any() or a.dry:
        continue
    ncut_blk += 1
    sg = sqrtg(rec)
    cons = np.array(rec[:nmhd*ncc]).reshape(nmhd, no, no, no)
    Dnew = sg*rho_f
    for n in range(5, nmhd):   # passive scalars: keep the cell's Y = (D Y)/D
        Yv = np.where(cons[0] > 0, cons[n]/np.where(cons[0] > 0, cons[0], 1.0), 0.0)
        cons[n] = np.where(out, Dnew*Yv, cons[n])
    cons[0] = np.where(out, Dnew, cons[0])
    for n in (1, 2, 3): cons[n] = np.where(out, 0.0, cons[n])
    cons[4] = np.where(out, sg*P_f/(GAM - 1.0), cons[4])
    dst[b, :nmhd*ncc] = cons.reshape(-1)
if not a.dry:
    dst.flush(); del dst
print(f'mass kept (r <= rcut) {Mkeep/MSTAR:.5f} M*, removed {Mcut/MSTAR:.5f} M*; blocks modified {ncut_blk} of {nmb}')
