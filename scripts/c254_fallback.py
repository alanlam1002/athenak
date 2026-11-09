#!/usr/bin/env python3
"""c254 (R-254 items 1, 3, user-approved 2026-11-09): the frozen-in ballistic fallback rate from the bound debris (c254_extract.py).
G = c = M_BH = 1; E = -u_t - 1 per unit rest mass; times from pericentre t_p.
 (a) Keplerian: t = 2 pi/(2|E|)^(3/2); Mdot as the mass-weighted histogram of t (= (1/3)(2 pi)^(2/3) (dM/dE) t^(-5/3)).
 (b) Schwarzschild geodesics: eps = 1 + E, L = |x_rel x u_cov| (the total specific angular momentum; Lz/L shows the tilt);
     areal R = r_iso (1 + 1/(2 r_iso))^2 (isotropic Schwarzschild; the trumpet slicing differs only at r ~ M);
     turning points from R^3 (eps^2 - 1) + 2 R^2 - L^2 R + 2 L^2 = 0 (roots R3 < R1 < R2);
     dt = eps/(1 - 2/R) R^(3/2) dchi / sqrt((1 - eps^2)(R - R3)), R = (R1+R2)/2 - (R2-R1)/2 cos chi (smooth; Gauss-Legendre 48).
     (b1) T_r from t_p (the full radial period; as (a) but relativistic);
     (b2) the phase-aware first return: (t_now - t_p) + the time from the cell's R (and the sign of its radial velocity) to its
          next pericentre (outgoing: R -> R2 -> R1; incoming: R -> R1). Cells with R outside [R1, R2] (pressure-supported,
          not ballistic) are clipped to the nearer turning point and counted (f_clip).
 (c) captured: bound debris with one real root (no turning point: the orbit plunges; L below the capture limit for its E).
Mdot in code units (M_BH per M) on log bins; peak time and rate, t_first (0.1% of the returning mass), the 10%-of-peak rise time,
the late slope (fit of log Mdot vs log t over [3, 30] t_peak); physical units for M_BH = 1e4 M_sun; Eddington ratio for eta = 0.1.
    c254_fallback.py OUTDIR LABEL=NPZ:t_p [...]"""
import sys, csv
import numpy as np

MSUN_G, C_CGS, YR = 1.98847e33, 2.99792458e10, 3.15576e7
MBH = 1.0e4                                  # M_sun
TU = 4.925490947e-6*MBH                      # GM/c^3 [s]
MDOT_U = MBH/TU*YR                           # code Mdot -> M_sun/yr
LEDD = 1.2572e38*MBH                         # erg/s (electron scattering, X = 0: 4 pi G M m_p c / sigma_T)
ETA = 0.1
MDOT_EDD = LEDD/(ETA*C_CGS**2)/MSUN_G*YR     # M_sun/yr
MS = 7.47e-5                                 # M* in M_BH (0.747 M_sun for M_BH = 1e4 M_sun)
xg, wg = np.polynomial.legendre.leggauss(48)
TB = np.logspace(3, 8, 251)                  # time bins [M]


def cubic_roots(eps, L):
    """real roots of (eps^2-1) R^3 + 2 R^2 - L^2 R + 2 L^2; returns R3 <= R1 <= R2 (nan where only one real root)."""
    a = eps**2 - 1.0
    b, c, d = 2.0/a, -L**2/a, 2.0*L**2/a
    p = c - b*b/3.0
    q = 2*b**3/27.0 - b*c/3.0 + d
    disc = (q/2)**2 + (p/3)**3
    three = disc < 0
    R = np.full((3, eps.size), np.nan)
    m = 2*np.sqrt(np.where(three, -p/3, 1.0))
    th = np.arccos(np.clip(np.where(three, 3*q/(p*m + 1e-300), 0.0), -1, 1))/3
    for k in range(3):
        R[k] = np.where(three, m*np.cos(th - 2*np.pi*k/3) - b/3, np.nan)
    R.sort(0)
    return R[0], R[1], R[2], three


def tint(eps, R1, R2, R3, c0, c1):
    """coordinate time over chi in [c0, c1] (arrays)."""
    tot = np.zeros(eps.size)
    h = 0.5*(c1 - c0); mid = 0.5*(c1 + c0)
    Rm, Ra = 0.5*(R1 + R2), 0.5*(R2 - R1)
    for x, w in zip(xg, wg):
        ch = mid + h*x
        R = Rm - Ra*np.cos(ch)
        f = eps/(1 - 2/R)*R**1.5/np.sqrt((1 - eps**2)*(R - R3))
        tot += w*f
    return tot*h


def curve(t, w):
    m, _ = np.histogram(t, bins=TB, weights=w)
    return m/np.diff(TB)


def stats(md, total):
    tc = np.sqrt(TB[1:]*TB[:-1])
    k = np.argmax(md)
    tp, mp = tc[k], md[k]
    rise = tc[np.argmax(md >= 0.1*mp)]
    cm = np.cumsum(md*np.diff(TB)); tf = tc[np.searchsorted(cm, 1e-3*cm[-1])]
    s = (tc >= 3*tp) & (tc <= 30*tp) & (md > 0)
    slope = np.polyfit(np.log(tc[s]), np.log(md[s]), 1)[0] if s.sum() > 3 else np.nan
    return dict(t_peak=tp, mdot_peak=mp, t_rise10=rise, t_first=tf, slope_3_30=slope)


out = sys.argv[1]
rows, curves = [], {}
for arg in sys.argv[2:]:
    lab, rest = arg.split('=', 1); fn, tpr = rest.rsplit(':', 1); tpr = float(tpr)
    z = np.load(fn)
    E, dM = z['E'], z['dM']
    eps = 1.0 + E
    L = np.sqrt(z['Lx']**2 + z['Ly']**2 + z['Lz']**2)
    tilt = np.abs(z['Lz'])/np.maximum(L, 1e-300)
    ri = z['riso']; R = ri*(1 + 0.5/ri)**2
    tK = 2*np.pi/(2*np.abs(E))**1.5
    R3, R1, R2, three = cubic_roots(eps, L)
    ok = three & (R1 > 2.0) & (R2 > R1)
    cap = ~ok
    Rc = np.clip(R, np.where(ok, R1, 0), np.where(ok, R2, np.inf))
    clip = ok & ((R < R1) | (R > R2))
    T1 = np.full(E.size, np.nan); T2 = np.full(E.size, np.nan)
    o = np.nonzero(ok)[0]
    for i0 in range(0, o.size, 2_000_000):
        j = o[i0:i0 + 2_000_000]
        e, r1, r2, r3 = eps[j], R1[j], R2[j], R3[j]
        half = tint(e, r1, r2, r3, np.zeros(j.size), np.full(j.size, np.pi))
        T1[j] = 2*half
        chi = np.arccos(np.clip(((r1 + r2)/2 - Rc[j])/((r2 - r1)/2), -1, 1))
        out_ = z['s'][j] > 0
        rem = np.where(out_, tint(e, r1, r2, r3, chi, np.full(j.size, np.pi)) + half,
                       tint(e, r1, r2, r3, np.zeros(j.size), chi))
        T2[j] = (float(z['t']) - tpr) + rem
    Mb = dM.sum()
    cK = curve(tK, dM); c1 = curve(T1[ok], dM[ok]); c2 = curve(T2[ok], dM[ok])
    curves[lab] = dict(K=cK, b1=c1, b2=c2)
    r = dict(label=lab, t_dump=float(z['t']), t_since_tp=float(z['t']) - tpr, M_bound_deb=Mb/MS, M_deb=float(z['M_deb'])/MS,
             M_H=float(z['M_H'])/MS, f_capt=dM[cap].sum()/Mb, f_clip=dM[clip].sum()/Mb,
             tilt_med=float(np.median(tilt)), L_med=float(np.median(L)),
             E_mb=float(np.quantile(E, 1e-3)), tK_mb=float(2*np.pi/(2*abs(np.quantile(E, 1e-3)))**1.5))
    for k, cv in (('a', cK), ('b1', c1), ('b2', c2)):
        st = stats(cv, Mb)
        for kk, v in st.items():
            r[f'{k}_{kk}'] = v
        r[f'{k}_mdot_peak_msun_yr'] = st['mdot_peak']*MDOT_U
        r[f'{k}_peak_edd'] = st['mdot_peak']*MDOT_U/MDOT_EDD
        r[f'{k}_t_peak_s'] = st['t_peak']*TU
    rows.append(r)
    print({k: (f'{v:.4g}' if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
tc = np.sqrt(TB[1:]*TB[:-1])
np.savez(f'{out}/c254_curves.npz', t=tc, **{f'{l}_{k}': v for l, d in curves.items() for k, v in d.items()})
with open(f'{out}/c254_curves.csv', 'w') as f:
    keys = [f'{l}_{k}' for l, d in curves.items() for k in d]
    f.write('t_M,' + ','.join(keys) + '\n')
    for i, t in enumerate(tc):
        f.write(f'{t:.6g},' + ','.join(f'{curves[k.rsplit("_", 1)[0]][k.rsplit("_", 1)[1]][i]:.6g}' for k in keys) + '\n')
with open(f'{out}/c254_summary.csv', 'w') as f:
    w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
print(f'units: t_unit {TU:.5g} s; Mdot unit {MDOT_U:.5g} M_sun/yr; L_Edd {LEDD:.4g} erg/s; Mdot_Edd(eta={ETA}) {MDOT_EDD:.4g} M_sun/yr')
