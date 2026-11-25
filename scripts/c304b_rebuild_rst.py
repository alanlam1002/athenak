#!/usr/bin/env python3
"""c304b (R-278 item 6 rerun, user-approved 2026-11-25): rebuild an isolated-remnant restart whose conserved variables are
consistent with the hole-free metric.
Why: c304 kept L32T's conserved (D, S_i, tau) and only cut the debris. Removing the puncture lowers psi^4 at the remnant from 1.0142
to ~1.0000, and with the stored S_i the cold kinetic part of tau then exceeds the stored tau (tau/D - its cold minimum: +3e-6 ->
-9.6e-5 at the centre), so the energy C2P failed in 99.8% of the remnant (code 13; c306 v1, C-272).
How:
  METRIC  a restart written ONE cycle after loading the c304 restart with regridding off (c306r1): its u_adm (gamma_ij, psi4, alpha,
          beta) and CFC warm starts are the hole-free solution on the same block layout. Its file is the base of DST.
  PRIMS   L32T's restart (SRC): its restart_state dyngr.w0 = (rho, W v^i (Eulerian), P, Y) and its densitized D.
Per cell (ghosts included), with x_c the centroid of rho > 0.3 rho_max (c304's) and r = |x - x_c|:
  r > rcut: the floor at Eulerian rest (rho_f = dfloor, P_f = dfloor tfloor), D = sqrt(g) rho_f, the old Y.
  r <= rcut: keep the densitized D (the rest mass) and W v^i; W = sqrt(1 + g_ij Wv^i Wv^j) in the new metric;
             rho = D / (sqrt(g) W); P = P_old (rho/rho_old)^Gamma (the entropy kept); Y kept.
  then S_i = sqrt(g) rho h W g_ij Wv^j, tau = sqrt(g) (rho h W^2 - P) - D, DY = D Y (h = 1 + Gamma/(Gamma-1) P/rho), and the
  restart_state's dyngr.w0 = (rho, Wv, P, Y), dyngr.temp = P/rho. Everything else (flags, bcc0, CFC warm starts, u_adm) is METRIC's.
    c304b_rebuild_rst.py SRC METRIC DST [--rcut 5.0] [--frac 0.3]"""
import argparse, os, re, shutil, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tde_dmde as T

MSTAR = 7.47e-5; GAM = 5.0/3.0
ap = argparse.ArgumentParser()
ap.add_argument('src'); ap.add_argument('metric'); ap.add_argument('dst')
ap.add_argument('--rcut', type=float, default=5.0); ap.add_argument('--frac', type=float, default=0.3)
a = ap.parse_args()

ts, tm = T.header(a.src), T.header(a.metric)
def pget(par, key, default):
    m = re.search(rf'^{key}\s*=\s*([-+.\deE]+)', par, re.M)
    return float(m.group(1)) if m else default
dfloor = pget(tm['par'], 'dfloor', 1e-24); tfloor = pget(tm['par'], 'tfloor', 1e-22)
rho_f, P_f = dfloor, dfloor*tfloor
assert ts['nmb'] == tm['nmb'] and np.array_equal(ts['ll'], tm['ll']), 'block layouts differ (run METRIC with regridding off)'
assert ts['dsize'] == tm['dsize'], 'record sizes differ'
nmb = ts['nmb']; ll, lev = ts['ll'], ts['ll'][:, 3]
x1min, x2min, x3min, x1max, x2max, x3max = ts['size'][:6]
nx, ng = 16, 4; no = nx + 2*ng; ncc = no**3; nmhd = 6; bsz = (no + 1)*no*no*3; nadm = 17
w = np.array([x1max - x1min, x2max - x2min, x3max - x3min]); lo = np.array([x1min, x2min, x3min])
ic = np.arange(no) - ng + 0.5; si = slice(ng, ng + nx)
def layout(par):
    m = re.search(r'^restart_state_layout\s*=\s*(\S+)', par, re.M); goff = {}; o = 0
    for g in m.group(1).split(','):
        nm, n = g.rsplit(':', 1); goff[nm] = (o, int(n)); o += int(n)
    return goff
gs, gm = layout(ts['par']), layout(tm['par'])
assert gs == gm, 'restart_state layouts differ'
st0 = nmhd*ncc + bsz + nadm*ncc
assert ts['dsize']//8 == st0 + sum(v[1] for v in gs.values())*ncc
ow, nw = gs['dyngr.w0']; otp, _ = gs['dyngr.temp']; assert nw == 6
S = np.memmap(a.src, dtype=np.float64, mode='r', offset=ts['data_off'], shape=(nmb, ts['dsize']//8))
def geom(b):
    ext = w/2.0**lev[b]; dx = ext/nx
    x = lo[0] + ll[b, 0]*ext[0] + ic*dx[0]; y = lo[1] + ll[b, 1]*ext[1] + ic*dx[1]; z = lo[2] + ll[b, 2]*ext[2] + ic*dx[2]
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij'); return X, Y, Z, dx
def w0of(rec): return np.array(rec[st0 + ow*ncc:st0 + (ow + 6)*ncc]).reshape(6, no, no, no)
# the centroid from SRC's primitives (rho > frac rho_max)
rmax = max(w0of(S[b])[0][si, si, si].max() for b in range(nmb))
cx = np.zeros(3); cm = 0.0
for b in range(nmb):
    r0 = w0of(S[b])[0][si, si, si]; hi = r0 > a.frac*rmax
    if hi.any():
        X, Y, Z, dx = geom(b); wt = r0[hi]*dx.prod(); cm += wt.sum()
        cx += [(X[si, si, si][hi]*wt).sum(), (Y[si, si, si][hi]*wt).sum(), (Z[si, si, si][hi]*wt).sum()]
xc = cx/cm
print(f'SRC {a.src}: t {ts["time"]:.4f} cycle {ts["ncycle"]}; METRIC {a.metric}: t {tm["time"]:.4f} cycle {tm["ncycle"]}')
print(f'x_c = {np.round(xc, 5)} (rho > {a.frac} rho_max); rcut {a.rcut}; floor rho {rho_f:.1e} P {P_f:.1e}')
os.makedirs(os.path.dirname(os.path.abspath(a.dst)), exist_ok=True)
subprocess.run(['lfs', 'setstripe', '-S', '4m', '-c', '16', os.path.dirname(os.path.abspath(a.dst))], check=False)
shutil.copyfile(a.metric, a.dst)
O = np.memmap(a.dst, dtype=np.float64, mode='r+', offset=tm['data_off'], shape=(nmb, tm['dsize']//8))
Min = Mout = 0.0; dtau_min = np.inf; drho = []
for b in range(nmb):
    X, Y, Z, dx = geom(b)
    out = np.sqrt((X - xc[0])**2 + (Y - xc[1])**2 + (Z - xc[2])**2) > a.rcut
    rs, rm = S[b], O[b]
    adm = np.array(rm[nmhd*ncc + bsz:nmhd*ncc + bsz + nadm*ncc]).reshape(nadm, no, no, no)
    gxx, gxy, gxz, gyy, gyz, gzz = adm[:6]
    sg = np.sqrt(gxx*(gyy*gzz - gyz*gyz) - gxy*(gxy*gzz - gyz*gxz) + gxz*(gxy*gyz - gyy*gxz))
    w0s = w0of(rs); Ds = np.array(rs[:ncc]).reshape(no, no, no)
    u1, u2, u3 = [np.where(out, 0.0, w0s[q]) for q in (1, 2, 3)]
    l1 = gxx*u1 + gxy*u2 + gxz*u3; l2 = gxy*u1 + gyy*u2 + gyz*u3; l3 = gxz*u1 + gyz*u2 + gzz*u3
    W = np.sqrt(1.0 + u1*l1 + u2*l2 + u3*l3)
    D = np.where(out, sg*rho_f, Ds)
    rho = D/(sg*W)
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.where(out, P_f, w0s[4]*np.where(w0s[0] > 0, rho/w0s[0], 1.0)**GAM)
    Yv = w0s[5]
    h = 1.0 + GAM/(GAM - 1.0)*P/rho
    cons = np.empty((nmhd, no, no, no))
    cons[0] = D
    cons[1], cons[2], cons[3] = sg*rho*h*W*l1, sg*rho*h*W*l2, sg*rho*h*W*l3
    cons[4] = sg*(rho*h*W*W - P) - D
    cons[5] = D*Yv
    O[b, :nmhd*ncc] = cons.reshape(-1)
    w0n = np.stack([rho, u1, u2, u3, P, Yv])
    O[b, st0 + ow*ncc:st0 + (ow + 6)*ncc] = w0n.reshape(-1)
    O[b, st0 + otp*ncc:st0 + (otp + 1)*ncc] = (P/rho).reshape(-1)
    dV = dx.prod(); inn = ~out[si, si, si]
    Min += (D[si, si, si]*dV)[inn].sum(); Mout += (Ds[si, si, si]*dV)[~inn].sum()
    if inn.any():
        # thermal room: tau/D minus the cold minimum (P -> 0) at the same D, S
        S2 = (cons[1]*u1 + cons[2]*u2 + cons[3]*u3)[si, si, si][inn]/np.maximum(W[si, si, si][inn], 1)  # S_i v^i W, diagnostic only
        dtau_min = min(dtau_min, (P/rho)[si, si, si][inn].min())
        drho.append(np.abs(rho/np.where(w0s[0] > 0, w0s[0], rho) - 1)[si, si, si][inn].max())
O.flush(); del O
print(f'mass kept (r <= rcut) {Min/MSTAR:.5f} M* (D unchanged), removed {Mout/MSTAR:.5f} M*; max |rho/rho_old - 1| inside {max(drho):.3e}; '
      f'min P/rho inside {dtau_min:.3e}')
