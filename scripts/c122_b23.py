#!/usr/bin/env python3
"""c122 B.2/B.3 (R-140 item 1, PHASE1_SPECS S-B): errors of the stationary test flows on the
fixed trumpet against the exact solution, from the prim_3d dumps.
Exact solution, computed here independently of the code (numpy/scipy, not the pgen):
 - michel: HSW (1984) eqs. 68-74 in the areal R (T = P/rho from (1+(n+1)T)^2 (1 - 2M/R + u^2)
   = C2, u = C1/(T^n R^2), the supersonic root inside r_c), mapped to the trumpet slicing and
   isotropic radius: R(r_iso) by inverting chi(R) (cfc_puncture.hpp TrumpetArealToIso),
   alpha^2 = 1 - 2M/R + C^2/R^4, beta^R = alpha C/R^2, u^R = q - W beta^R/alpha,
   W = sqrt(1 + q^2/alpha^2), u~^r_iso = W (q/W) r/(R alpha);
 - hydrostatic: the initial state (h alpha_K = const, grid rest; it is the exact solution).
Per dump: L1 and Linf relative errors of rho and of u~^r = u~.x/r, and of K = P/rho^Gamma, in
radial shells, over evolved cells (alpha_trumpet > excise_lapse, r_iso < r_hold); and the
mass-flux constancy Mdot(r) = 4 pi r^2 psi^6 rho W v^r over the shells (michel).
    c122_b23.py OUT.csv RUNDIR [RUNDIR ...]"""
import glob
import os
import sys

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tde_ensemble_plots import bc                 # noqa: E402

B = 'cfc_tde_wd_imbh_parabolic_tracker'
M, GAM = 1.0, 5.0/3.0
N = 1.0/(GAM - 1.0)
CC = 0.75*np.sqrt(3.0)*M*M
SHELLS = [(2.0, 3.0), (3.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 38.0)]


def chi(rho):    # areal (M = 1) -> isotropic, TrumpetArealToIso
    t = 2.0*rho - 3.0
    s1 = np.sqrt(4*rho*rho + 4*rho + 3); s2 = np.sqrt(8*rho*rho + 8*rho + 6)
    return 0.25*(2*rho + 1 + s1)*((4 + 3*np.sqrt(2))*t/(8*rho + 6 + 3*s2))**(1/np.sqrt(2))


def areal(r):
    return np.array([brentq(lambda R: chi(R) - x, 1.5 + 1e-14, x + 10.0, xtol=1e-15)
                     for x in np.atleast_1d(r)])


def par(d, blk, key, default):
    import re
    s = open(glob.glob(f'{d}/*.par')[0] if glob.glob(f'{d}/*.par') else f'{d}/parfile.par').read()
    m = re.search(r'<%s>(.*?)(\n<|\Z)' % blk, s, re.S)
    mm = re.search(r'(?m)^\s*%s\s*=\s*(\S+)' % key, m.group(1)) if m else None
    return float(mm.group(1)) if mm else default


def michel_profile(rc, rhoc, r):
    uc2 = M/(2*rc); tc = N/(N + 1)*uc2/(1 - (N + 3)*uc2)
    k = tc/rhoc**(1/N); c1 = tc**N*np.sqrt(uc2)*rc**2; c2 = (1 + (N + 1)*tc)**2*(1 - 1.5*M/rc)
    R = areal(r)
    rho, ur = np.zeros_like(r), np.zeros_like(r)
    for i, Ri in enumerate(R):
        f = lambda T: (1 + (N + 1)*T)**2*(1 - 2*M/Ri + (c1/(T**N*Ri**2))**2) - c2
        Ts = np.logspace(-12, 1, 4000); fs = np.array([f(T) for T in Ts])
        tmin = Ts[np.argmin(fs)]
        from scipy.optimize import minimize_scalar
        tmin = np.exp(minimize_scalar(lambda lt: f(np.exp(lt)), bracket=(np.log(tmin) - 0.01, np.log(tmin) + 0.01),
                                      tol=1e-14).x)
        if f(tmin) >= 0.0:      # at the sonic point the two roots merge
            T = tmin
        else:
            T = brentq(f, 1e-12, tmin, xtol=1e-30, rtol=1e-15, maxiter=1000) if Ri <= rc else brentq(f, tmin, 10.0, xtol=1e-30, rtol=1e-15, maxiter=1000)
        uR = -c1/(T**N*Ri**2)
        al = np.sqrt(1 - 2*M/Ri + CC**2/Ri**4); bR = al*CC/Ri**2
        q = brentq(lambda q: q - np.sqrt(1 + q*q/al**2)*bR/al - uR, uR - 10, uR + 10, xtol=1e-30, rtol=1e-15, maxiter=1000)
        W = np.sqrt(1 + q*q/al**2)
        rho[i] = (T/k)**N
        ur[i] = W*(q/W)*r[i]/(Ri*al)
    return rho, ur, k


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
        fs = sorted(glob.glob(f'{d}/bin/{B}.prim_3d.*.bin'))
        if not fs:
            continue
        test = 'michel' if lab.startswith('B2') else 'hydrostatic'
        rout = par(d, 'problem', 'test_hold_rout', 40.0)
        exl = par(d, 'coord', 'excise_lapse', 0.25)
        t0, X, Y, Z, D0 = cells(fs[0])
        r = np.sqrt(X*X + Y*Y + Z*Z)
        # the exact profile on a radial grid, interpolated in log r
        rg = np.geomspace(max(r.min(), 1.0), r.max(), 1500)   # r_iso >= 1 (R > 2.3 M): outside the excised throat
        Rg = areal(rg)
        alg = np.sqrt(np.maximum(1 - 2*M/Rg + CC**2/Rg**4, 0))
        if test == 'michel':
            rho_e, ur_e, kE = michel_profile(par(d, 'problem', 'test_michel_rc', 8.0),
                                            par(d, 'problem', 'test_michel_rhoc', 1e-8), rg)
            rhoE = np.exp(CubicSpline(np.log(rg), np.log(rho_e))(np.log(r)))
            urE = CubicSpline(np.log(rg), ur_e)(np.log(r))
        alc = np.interp(np.log(r), np.log(rg), alg)
        pk = 'eint' if 'eint' in D0 else 'press'
        Kof = lambda Dd: (Dd[pk]*(GAM - 1.0) if pk == 'eint' else Dd[pk])/Dd['dens']**GAM
        if test == 'hydrostatic':
            kE = Kof(D0)
        ev = (alc > exl) & (r < rout - 1.0) & (r >= 1.0)
        for f in fs:
            t, X, Y, Z, Dd = cells(f)
            rho = Dd['dens']
            ur = (Dd['velx']*X + Dd['vely']*Y + Dd['velz']*Z)/np.maximum(r, 1e-30)
            if test == 'hydrostatic':
                rhoE, urE = D0['dens'], (D0['velx']*X + D0['vely']*Y + D0['velz']*Z)/np.maximum(r, 1e-30)
            for (a, b) in SHELLS:
                m = ev & (r >= a) & (r < b)
                if not m.any():
                    continue
                er = np.abs(rho[m] - rhoE[m])/rhoE[m]
                eu = np.abs(ur[m] - urE[m])/np.maximum(np.abs(urE[m]), 1e-3 if test == 'hydrostatic' else 1e-30)
                kk = Kof(Dd)[m]/(kE[m] if np.ndim(kE) else kE) - 1.0
                rows.append(f'{lab},{t:.4g},{a:g}-{b:g},{m.sum()},{er.mean():.3e},{er.max():.3e},'
                            f'{eu.mean():.3e},{eu.max():.3e},{np.abs(ur[m]-urE[m]).max():.3e},'
                            f'{kk.mean():.3e},{np.abs(kk).mean():.3e},{np.abs(kk).max():.3e}')
    hdr = 'run,t,shell_r_iso,ncell,rho_L1rel,rho_Linf_rel,ur_L1rel,ur_Linf_rel,ur_Linf_abs,K_mean,K_L1,K_Linf'
    open(out, 'w').write(hdr + '\n' + '\n'.join(rows) + '\n')
    print(hdr); print('\n'.join(rows))


if __name__ == '__main__':
    main()
