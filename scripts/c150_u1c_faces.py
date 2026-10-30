#!/usr/bin/env python3
"""c150 U1c (and U1 per face): the probe's stage-1 x1-face fluxes (cycle 0, donor cell, so the
face states are the t = 0 cells) against an independent lab-frame SR solver in Python:
HLLC transcribed from src/hydro/rsolvers/hllc_srhyd.hpp:40-199, HLLE the same HLL fan (Toro
10.43 speeds), both sampled at the interface moving at v_if = xidot (flat space, beta^x =
xidot: a grid point moves at +xidot in the lab, so the coordinate flux is F_K - v_if U_K).
Compared: F_D, F_S1, F_S2, F_S3, F_tau (dyn_grmhd densitized, alpha = sqrt(gamma) = 1).
    u1c_faces.py RUNDIR SOLVER XIDOT"""
import glob, re, sys
import numpy as np
d, solver, xd = sys.argv[1], sys.argv[2], float(sys.argv[3])
g = 5.0/3.0
gp = g/(g - 1.0)
a = np.loadtxt(sorted(glob.glob(f'{d}/tab/*.00000.tab'))[0])
x, rho, u1, u2, u3, p = a[:, 2], a[:, 3], a[:, 4], a[:, 5], a[:, 6], a[:, 7]
n = len(x)


def speeds(r, pp, ux, ut):   # Toro 10.43 via the sound speed (AthenaK IdealSRHydroSoundSpeeds)
    w = r + gp*pp
    cs2 = g*pp/w
    vx = ux/ut
    v2 = 1.0 - 1.0/ut**2
    q = (1 - v2)*(1 - v2*cs2 - vx*vx*(1 - cs2))
    den = 1 - v2*cs2
    return (vx*(1 - cs2) + np.sqrt(cs2*q))/den, (vx*(1 - cs2) - np.sqrt(cs2*q))/den


def flux(i):
    L, R = (i - 1) % n, i
    st = []
    for k in (L, R):
        ut = np.sqrt(1 + u1[k]**2 + u2[k]**2 + u3[k]**2)
        w = rho[k] + gp*p[k]
        c = np.array([rho[k]*ut, w*ut*ut - p[k], w*u1[k]*ut, w*u2[k]*ut, w*u3[k]*ut])
        f = np.array([rho[k]*u1[k], w*ut*u1[k], w*u1[k]*u1[k] + p[k], w*u2[k]*u1[k],
                      w*u3[k]*u1[k]])
        lp, lm = speeds(rho[k], p[k], u1[k], ut)
        st.append((c, f, lp, lm, p[k], u1[k]/ut))
    (cl, fl, lpl, lml, pl, vl), (cr, fr, lpr, lmr, pr, vr) = st
    ll, lr = min(lml, lmr), max(lpl, lpr)
    v = xd
    chll = (lr*cr - ll*cl + fl - fr)/(lr - ll)
    fhll = (lr*fl - ll*fr + ll*lr*(cr - cl))/(lr - ll)
    if solver == 'hlle':
        if ll >= v: out = fl - v*cl
        elif lr <= v: out = fr - v*cr
        else: out = fhll - v*chll
    else:
        b = -(chll[1] + fhll[2])
        if abs(fhll[1] - fhll[0]) > 1e-12:
            ls = -2.0*chll[2]/(b - np.sqrt(b*b - 4.0*fhll[1]*chll[2]))
        else:
            ls = -chll[2]/b
        ps = -fhll[1]*ls + fhll[2]
        cls = cl*(ll - vl); cls[1] += ps*ls - pl*vl; cls[2] += ps - pl; cls /= (ll - ls)
        crs = cr*(lr - vr); crs[1] += ps*ls - pr*vr; crs[2] += ps - pr; crs /= (lr - ls)
        if ll >= v: out = fl - v*cl
        elif lr <= v: out = fr - v*cr
        elif ls >= v: out = fl + ll*(cls - cl) - v*cls
        else: out = fr + lr*(crs - cr) - v*crs
    return np.array([out[0], out[2], out[3], out[4], out[1] - out[0]])


rec = {}
for ln in open(f'{d}/kprobe_rank00000.txt'):
    t = ln.split()
    if t[0] != 'KPROBE' or t[1] != '0' or t[2] != '1' or t[3] != '0':
        continue
    rec[int(t[5]) - 4] = np.array([float(z) for z in t[12:17]])   # i - is (nghost 4)
worst, scale = 0.0, 0.0
for i, fc in sorted(rec.items()):
    if i >= n:   # the ie+1 face is the periodic image of face 0
        i = 0
    fp = flux(i)
    worst = max(worst, np.max(np.abs(fc - fp)))
    scale = max(scale, np.max(np.abs(fp)))
print(f'{d}: {len(rec)} x1 faces, max |F_code - F_python| = {worst:.3e}, max |F| = {scale:.3e}, '
      f'relative {worst/scale:.2e}')
