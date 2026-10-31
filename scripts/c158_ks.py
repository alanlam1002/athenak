#!/usr/bin/env python3
"""c158 (R-186 item 3): errors of Michel accretion in Cartesian Kerr-Schild (a = 0) against the exact solution, from the
prim_3d dumps (Step A: the GR hydro module + gr_bondi, basename ks_michel; Step B: dyn_grmhd on the fixed KS metric).
Exact solution, computed here independently of the code (numpy/scipy, not the pgen): HSW (1984) eqs. 68-74 in r (KS r = the
areal radius for a = 0; the supersonic root inside r_c), u^r = -C1/(T^n r^2) (the same in BL and KS), and the KS 3+1 split
alpha = 1/sqrt(1 + 2M/r), beta^r = (2M/r)/(1 + 2M/r), gamma_rr = 1 + 2M/r: the code primitive u~^r = u^r + u^t beta^r
(W/alpha = u^t from the 4-velocity's normalization), and u~^i = u~^r x^i/r.
Per dump: L1 and Linf relative errors of rho, u~^r = u~.x/r, K = P/rho^Gamma and the slicing-invariant u^R = dR/dtau, in AREAL-radius shells over the cells
with r > r_ex (the excised r_KS <= 1 of Step A, or alpha <= excise_lapse) and r < 60 (inside the box faces' fixed inflow).
    c158_ks.py OUT.csv RUNDIR [RUNDIR ...]     (C158_SHELLS="R0,R1,..." overrides the shell edges)"""
import glob
import os
import re
import sys

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tde_ensemble_plots import bc                 # noqa: E402

M, GAM = 1.0, 5.0/3.0
N = 1.0/(GAM - 1.0)
EDGES = [1.2, 1.6, 2.0, 2.2, 2.65, 3.1, 4.1, 6.0, 11.0, 21.0, 39.0]
if os.environ.get('C158_SHELLS'):
    EDGES = [float(v) for v in os.environ['C158_SHELLS'].split(',')]


def par(d, blk, key, default):
    fs = glob.glob(f'{d}/*.par') or glob.glob(f'{d}/../{os.path.basename(d.rstrip("/"))}.par')
    s = open(fs[0]).read()
    m = re.search(r'<%s>(.*?)(\n<|\Z)' % blk, s, re.S)
    mm = re.search(r'(?m)^\s*%s\s*=\s*(\S+)' % key, m.group(1)) if m else None
    return float(mm.group(1)) if mm else default


def michel_ks(rc, kadi, r):
    """rho, u~^r on KS at r (areal), the sonic point rc, T = K rho^(1/n) with K = kadi"""
    uc2 = M/(2*rc); tc = N/(N + 1)*uc2/(1 - (N + 3)*uc2)
    c1 = tc**N*np.sqrt(uc2)*rc**2; c2 = (1 + (N + 1)*tc)**2*(1 - 1.5*M/rc)
    rho, ut = np.zeros_like(r), np.zeros_like(r)
    for i, R in enumerate(r):
        f = lambda T: (1 + (N + 1)*T)**2*(1 - 2*M/R + (c1/(T**N*R**2))**2) - c2
        Ts = np.logspace(-6, 1, 4000); im = int(np.argmin([f(T) for T in Ts])); tmin = Ts[im]
        if 0 < im < len(Ts) - 1:   # an interior minimum (outside the horizon); inside it f decreases monotonically
            tmin = np.exp(minimize_scalar(lambda lt: f(np.exp(lt)), bracket=(np.log(tmin) - 0.01, np.log(tmin) + 0.01),
                                          tol=1e-14).x)
        if f(tmin) >= 0.0:
            T = tmin
        elif R <= rc:
            T = brentq(f, 1e-6, tmin, xtol=1e-30, rtol=1e-15, maxiter=1000)
        else:
            T = brentq(f, tmin, 10.0, xtol=1e-30, rtol=1e-15, maxiter=1000)
        ur = -c1/(T**N*R**2)
        # u^t from g_tt (u^t)^2 + 2 g_tr u^t u^r + g_rr (u^r)^2 = -1 (KS: g_tt = -(1 - 2M/r), g_tr = 2M/r,
        # g_rr = 1 + 2M/r), the future root (valid inside the horizon too); u~^r = u^r + (W/alpha) beta^r = u^r + u^t beta^r
        a, b, c = -(1 - 2*M/R), 4*M/R*ur, (1 + 2*M/R)*ur*ur + 1
        utt = (-b - np.sqrt(b*b - 4*a*c))/(2*a) if a != 0.0 else -c/b
        q = ur + utt*(2*M/R)/(1 + 2*M/R)
        rho[i] = (T/kadi)**N
        ut[i] = q
    return rho, ut


def u_areal(ut, r):
    """u^R = dR/dtau (slicing-invariant) from the KS code primitive u~^r: u^r = u~^r - W beta^r/alpha"""
    al = 1/np.sqrt(1 + 2*M/r); bR = (2*M/r)/(1 + 2*M/r)
    return ut - np.sqrt(1 + (1 + 2*M/r)*ut*ut)*bR/al


def cells(f):
    P = bc.read_binary(f)
    g = np.asarray(P['mb_geometry']); n = P['nx1_out_mb']
    X, Y, Z, D = [], [], [], {}
    for b in range(len(g)):
        h = (g[b, 1] - g[b, 0])/n
        c = lambda lo: lo + (np.arange(n) + 0.5)*h
        z, y, x = np.meshgrid(c(g[b, 4]), c(g[b, 2]), c(g[b, 0]), indexing='ij')
        X.append(x.ravel()); Y.append(y.ravel()); Z.append(z.ravel())
        for q in ('dens', 'eint' if 'eint' in P['mb_data'] else 'press', 'velx', 'vely', 'velz'):
            D.setdefault(q, []).append(np.asarray(P['mb_data'][q][b]).ravel())
    return float(P['time']), np.concatenate(X), np.concatenate(Y), np.concatenate(Z), \
        {k: np.concatenate(v) for k, v in D.items()}


def main():
    out, rows = sys.argv[1], []
    for d in sys.argv[2:]:
        lab = os.path.basename(d.rstrip('/'))
        fs = sorted(glob.glob(f'{d}/bin/*.prim_3d.*.bin'))
        if not fs:
            continue
        rc = par(d, 'problem', 'r_crit', 8.0)
        kadi = par(d, 'problem', 'k_adi', 1.0)
        exl = par(d, 'coord', 'excise_lapse', -1.0)
        t0, X, Y, Z, D0 = cells(fs[0])
        r = np.sqrt(X*X + Y*Y + Z*Z)
        rg = np.geomspace(max(r.min(), 0.5), r.max(), 1500)
        rho_e, ut_e = michel_ks(rc, kadi, rg)
        rhoE = np.exp(CubicSpline(np.log(rg), np.log(rho_e))(np.log(r)))
        utE = CubicSpline(np.log(rg), ut_e)(np.log(r))
        alc = 1/np.sqrt(1 + 2*M/r)
        rex = 1.0 if exl < 0 else 2*M/(1/exl**2 - 1)      # Step A: r_KS <= 1; Step B: alpha <= excise_lapse
        pk = 'eint' if 'eint' in D0 else 'press'
        Kof = lambda Dd: (Dd[pk]*(GAM - 1.0) if pk == 'eint' else Dd[pk])/Dd['dens']**GAM
        Kexact = np.full_like(rhoE, kadi)                  # T = kadi rho^(1/n): P/rho^Gamma = kadi
        ev = (r > rex) & (r < 60.0)
        for f in fs:
            t, X, Y, Z, Dd = cells(f)
            rho = Dd['dens']
            ut = (Dd['velx']*X + Dd['vely']*Y + Dd['velz']*Z)/np.maximum(r, 1e-30)
            for a, b in zip(EDGES[:-1], EDGES[1:]):
                m = ev & (r >= a) & (r < b)
                if not m.any():
                    continue
                er = np.abs(rho[m] - rhoE[m])/rhoE[m]
                eu = np.abs(ut[m] - utE[m])/np.abs(utE[m])     # u~^r: changes sign in KS (r ~ 13 M); see uR
                uR, uRE = u_areal(ut[m], r[m]), u_areal(utE[m], r[m])
                eR = np.abs(uR - uRE)/np.abs(uRE)
                kk = Kof(Dd)[m]/Kexact[m] - 1.0
                rows.append(f'{lab},{t:.4g},R{a:g}-{b:g},{m.sum()},{er.mean():.3e},{er.max():.3e},'
                            f'{eu.mean():.3e},{eu.max():.3e},{np.abs(ut[m]-utE[m]).max():.3e},'
                            f'{kk.mean():.3e},{np.abs(kk).mean():.3e},{np.abs(kk).max():.3e},{eR.mean():.3e},{eR.max():.3e},'
                            f'{alc[m].mean():.4f}')
    hdr = ('run,t,shell_R,ncell,rho_L1rel,rho_Linf_rel,ur_L1rel,ur_Linf_rel,ur_Linf_abs,K_mean,K_L1,K_Linf,uR_L1rel,uR_Linf_rel,'
           'alpha_mean')
    open(out, 'w').write(hdr + '\n' + '\n'.join(rows) + '\n')
    print(hdr); print('\n'.join(rows))


if __name__ == '__main__':
    main()
