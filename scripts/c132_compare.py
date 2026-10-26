#!/usr/bin/env python3
"""c132 (R-145 item 3; TDE_DEBRIS_SPEC.md §6): the local tidal star (tde_local, <cfc> tidal_lapse)
against the comoving full run CD rp10 (c110_2x2/CD) at t_p + 20, 50 and 100.
Times: CD is in the full run's coordinate time (periapsis t_p = 135.9, C-110); a Marck run in
the star's proper time tau, mapped by its own table (Schwarzschild t(tau); periapsis
tau_p = 119.5 at t = 135.97 from areal r_0 = 31); a Newtonian run at tau_p + dt (its periapsis
from its table). The nearest 3D dump to each target is used (CD: every 12 M).
Per dump, over gas with rho > 1e-8 rho_c, about the centre of mass:
  the ridge density (max rho over the bins along the stream axis, as c132_local.py), the
  stream length and sigma_major/minor, the fraction of mass with N_perp >= N_w, the core
  mass (rho > 0.01 rho_c), the mass-weighted <K/K0> over rho >= 0.1 rho_c, and rho_max.
Argued caveat: the local run's quadratic tide is exact only near the core (~L/R error).
    c132_compare.py OUT.csv CD=DIR [L_newt=DIR] [L_marck=DIR] [--dt 20,50,100]"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tde_ensemble_plots import bc                 # noqa: E402

B = 'cfc_tde_wd_imbh_parabolic_tracker'
RHOC, KAPPA, GAM = 3.24e-4, 1.2346626297e-2, 5.0/3.0
TP_CD = 135.9


def dumps(d):
    return sorted(glob.glob(f'{d}/bin/{B}.prim_3d.*.bin'))


def load(f):
    P = bc.read_binary(f)
    g = np.asarray(P['mb_geometry']); n = P['nx1_out_mb']
    q = P['mb_data']
    pk = 'press' if 'press' in q else 'eint'
    X, Y, Z, R, PR, DX = [], [], [], [], [], []
    for b in range(len(g)):
        h = (g[b, 1] - g[b, 0])/n
        c = lambda lo: lo + (np.arange(n) + 0.5)*h
        z, y, x = np.meshgrid(c(g[b, 4]), c(g[b, 2]), c(g[b, 0]), indexing='ij')
        X.append(x.ravel()); Y.append(y.ravel()); Z.append(z.ravel())
        R.append(np.asarray(q['dens'][b]).ravel()); PR.append(np.asarray(q[pk][b]).ravel())
        DX.append(np.full(n**3, h))
    cat = np.concatenate
    p = cat(PR)
    if pk == 'eint':
        p = (GAM - 1.0)*p
    return float(P['time']), cat(X), cat(Y), cat(Z), cat(R), p, cat(DX)


def stats(f, nw=4.0, ds=0.25):
    t, X, Y, Z, rho, p, dx = load(f)
    dm = rho*dx**3
    m = rho > 1e-8*RHOC
    M = dm[m].sum()
    c = np.array([(X*dm)[m].sum(), (Y*dm)[m].sum(), (Z*dm)[m].sum()])/M
    r = np.stack([X - c[0], Y - c[1], Z - c[2]], axis=1)[m]
    w = dm[m]
    I = (r*w[:, None]).T @ r/M
    ev, evec = np.linalg.eigh(I)
    s = r @ evec[:, -1]
    perp2 = np.sum(r*r, axis=1) - s*s
    edges = np.arange(np.floor(s.min()/ds)*ds, s.max() + ds, ds)
    ib = np.digitize(s, edges) - 1
    ridge, ok = [], 0.0
    for b in range(len(edges) - 1):
        k = ib == b
        if not k.any():
            continue
        wb = w[k]
        fwhm = 2.0*np.sqrt(2.0*np.log(2.0))*np.sqrt((perp2[k]*wb).sum()/wb.sum())
        if fwhm/((dx[m][k]*wb).sum()/wb.sum()) >= nw:
            ok += wb.sum()
        ridge.append(rho[m][k].max())
    core = dm[rho > 1e-2*RHOC].sum()
    d1 = rho >= 0.1*RHOC
    kk = (dm[d1]*(p[d1]/rho[d1]**GAM/KAPPA)).sum()/dm[d1].sum() if d1.any() else np.nan
    return dict(t=t, M=M, length=s.max() - s.min(), sig_major=np.sqrt(ev[-1]),
                sig_minor=np.sqrt(ev[0]), frac_nperp=ok/M, core_mass=core, K_over_K0=kk,
                rhomax=rho.max()/RHOC, ridge_median=np.median(ridge)/RHOC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('runs', nargs='+')
    ap.add_argument('--dt', default='20,50,100')
    a = ap.parse_args()
    R = dict(x.split('=', 1) for x in a.runs)
    rows = []
    for lab, d in R.items():
        fs = dumps(d)
        ts = np.array([float(bc.read_binary(f)['time']) for f in fs])
        for dtp in [float(x) for x in a.dt.split(',')]:
            if lab == 'CD':
                target = TP_CD + dtp
            else:
                T = np.loadtxt(glob.glob(f'{d}/tidal_*.txt')[0])
                kp = np.argmin(T[:, 2])
                if 'marck' in lab:          # CD's t_p + dt in the full run's time -> tau
                    target = np.interp(T[kp, 1] + dtp, T[:, 1], T[:, 0])
                else:
                    target = T[kp, 0] + dtp
            i = int(np.argmin(np.abs(ts - target)))
            o = stats(fs[i])
            rows.append(f'{lab},{dtp:g},{target:.4g},' + ','.join(f'{o[k]:.6g}' for k in
                        ('t', 'M', 'length', 'sig_major', 'sig_minor', 'frac_nperp', 'core_mass',
                         'K_over_K0', 'rhomax', 'ridge_median')))
    hdr = ('run,dt_after_tp,target_time,t_dump,M_stream,length,sigma_major,sigma_minor,'
           'frac_mass_Nperp_ge4,core_mass,K_over_K0_dense,rhomax_over_rhoc,ridge_median_over_rhoc')
    open(a.out, 'w').write(hdr + '\n' + '\n'.join(rows) + '\n')
    print(hdr); print('\n'.join(rows))


if __name__ == '__main__':
    main()
