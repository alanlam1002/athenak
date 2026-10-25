#!/usr/bin/env python3
"""R-140 item 8 (D10 offline, c129): temperature and radiation pressure (gas + radiation in LTE), peak T and rho
for burning, and an along-stream field-amplification estimate, from the rp10/rp18 2x2 dumps.

Units (G = c = M_BH = 1, M_BH = 1e4 Msun): length GM/c^2 = 1.4766e9 cm, time GM/c^3 = 4.9255e-2 s, density
M/L^3 = 6.177e9 g/cm^3 (the R-140 value 6.17e9), pressure/energy density 6.177e9 c^2 = 5.552e30 erg/cm^3.
Check: rho_c = 3.24e-4 -> 2.00e6 g/cm^3, M_* = 7.47e-5 -> 0.747 Msun.
Composition: C/O 50/50 by mass, fully ionised: 1/mu = 0.5 (7/12) + 0.5 (9/16), mu = 1.745 (argued choice).
The code's EOS is an ideal gas (Gamma = 5/3, no radiation); the WD's own pressure is (in nature) electron
degeneracy, which the code represents by the initial adiabat P_cold = kappa rho^Gamma (kappa = 1.23466e-2,
parfile). So T is computed two ways:
  T_th  (primary): the THERMAL excess over the initial adiabat, e_th = (P - kappa rho^Gamma)/(Gamma - 1) >= 0,
        shared between an ideal ion+electron gas and LTE radiation: e_th = 1.5 n k T + a T^4, n = rho/(mu m_u);
  T_all (upper bound): all of P read as gas + radiation pressure: P = n k T + a T^4/3.
P_rad/P_gas = (a T^4/3)/(n k T) for each. Mass-weighted over bound/unbound debris (far-field E proxy, as
c119_domain.py), rho > 1e-6 rho_c (2 g/cm^3) for the temperature statistics (the hot floor atmosphere excluded).
Field amplification (argued, flux freezing in a stretched tube): B_par/B0 = (rho/rho0)(l/l0) for a field along
the stretch, B_perp/B0 = (rho/rho0)(l_perp/l_perp0) across it, with l = 2 sqrt(lambda_1) and l_perp =
2 sqrt(lambda_3), the principal axes of the mass-weighted position covariance of rho > 1e-6 rho_c gas, and rho
its mass-weighted mean density. One global tube: no curvature or local shear (the curved stream's l is an
underestimate of its arc length late on).
    c129_physics.py OUTDIR LABEL=RUNDIR:TP [...] [--maps-only]
"""
import glob
import os
import re
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'vis', 'python'))
import bin_convert as bc                          # noqa: E402

B = 'cfc_tde_wd_imbh_parabolic_tracker'
RHOC = 3.24e-4
GAM = 5.0 / 3.0
KAPPA = 1.2346626297e-2
RHO_U = 6.177e9            # g/cm^3
P_U = RHO_U * 2.99792458e10**2
KB = 1.380649e-16; MU_AMU = 1.66053907e-24; ARAD = 7.5657e-15
MU = 1.0 / (0.5 * 7 / 12 + 0.5 * 9 / 16)
DENSE = 1e-6 * RHOC
THR = 1e-10 * RHOC


def solve_T(u, n, mode):
    """mode 'e': u = 1.5 n k T + a T^4; mode 'p': u = n k T + a T^4/3 (cgs); bisection in log T"""
    lo = np.zeros_like(u); hi = np.full_like(u, 13.0)
    cg, cr = (1.5, 1.0) if mode == 'e' else (1.0, 1.0 / 3.0)
    for _ in range(80):
        m = 0.5 * (lo + hi); T = 10.0**m
        f = cg * n * KB * T + cr * ARAD * T**4 > u
        hi = np.where(f, m, hi); lo = np.where(f, lo, m)
    return 10.0**(0.5 * (lo + hi))


def thermo(rho, P):
    rcgs = rho * RHO_U
    n = rcgs / (MU * MU_AMU)
    eth = np.clip(P - KAPPA * rho**GAM, 0.0, None) / (GAM - 1) * P_U
    Tth = solve_T(np.maximum(eth, 1e-300), n, 'e')
    Tall = solve_T(P * P_U, n, 'p')
    rr = lambda T: ARAD * T**4 / 3.0 / (n * KB * T)
    return rcgs, Tth, rr(Tth), Tall, rr(Tall)


def bh_of(d):
    par = open(f'{d}/parfile.par').read()
    px = float(re.search(r'^puncture_x1\s*=\s*(\S+)', par, re.M).group(1))
    if re.search(r'^puncture_moving\s*=\s*true', par, re.M) is None:
        return lambda t: np.zeros(3)
    h = np.loadtxt(f'{d}/{B}.user.hst')
    return lambda t: np.array([px, 0, 0]) - np.array([np.interp(t, h[:, 0], h[:, 7 + k]) for k in range(3)])


def wpct(x, w, q):
    if len(x) == 0:
        return [np.nan] * len(q)
    o = np.argsort(x); c = np.cumsum(w[o]); c /= c[-1]
    return [x[o][min(np.searchsorted(c, qq / 100.0), len(x) - 1)] for qq in q]


def dump(arg):
    lab, d, fp = arg
    P = bc.read_binary(fp)
    t = float(P['time'])
    xb = bh_of(d)(t)
    g = np.asarray(P['mb_geometry']); n = P['nx1_out_mb']
    cols = {k: [] for k in ('x', 'y', 'z', 'rho', 'P', 'E', 'dM')}
    for b in range(len(g)):
        h = (g[b, 1] - g[b, 0]) / n
        c = lambda lo: lo + (np.arange(n) + 0.5) * h
        Z, Y, X = np.meshgrid(c(g[b, 4]), c(g[b, 2]), c(g[b, 0]), indexing='ij')
        rho = np.asarray(P['mb_data']['dens'][b], dtype=np.float64)
        m = rho > THR
        if not m.any():
            continue
        r = np.maximum(np.sqrt((X - xb[0])**2 + (Y - xb[1])**2 + (Z - xb[2])**2), 0.6)
        psi = 1 + 0.5 / r; al = (1 - 0.5 / r) / psi
        u = [np.asarray(P['mb_data'][k][b], dtype=np.float64) for k in ('velx', 'vely', 'velz')]
        W = np.sqrt(1 + psi**4 * sum(x * x for x in u))
        for k, v in (('x', X), ('y', Y), ('z', Z), ('rho', rho), ('E', al * W - 1), ('dM', rho * W * psi**6 * h**3),
                     ('P', np.asarray(P['mb_data']['press'][b], dtype=np.float64))):
            cols[k].append(v[m & (r >= 3)])
    C = {k: np.concatenate(v) for k, v in cols.items()}
    rcgs, Tth, Rth, Tall, Rall = thermo(C['rho'], C['P'])
    o = dict(label=lab, t=t)
    dn = C['rho'] > DENSE
    w = C['dM']
    i = np.argmax(np.where(dn, Tth, 0))
    o.update(Tth_peak=Tth[i], rho_at_Tth_peak=rcgs[i], Tall_peak=Tall[dn].max(), rho_peak=rcgs.max(),
             Tth_at_rho_peak=Tth[np.argmax(rcgs)])
    for T0 in (1e8, 5e8, 1e9):
        o[f'M_Tth_gt{T0:.0e}'] = w[dn & (Tth > T0)].sum() / 7.47e-5
    for nm, m in (('bound', dn & (C['E'] < 0)), ('unbound', dn & (C['E'] >= 0)), ('all', dn)):
        p = wpct(np.log10(Rth[m]), w[m], [50, 95]); pa = wpct(np.log10(Rall[m]), w[m], [50, 95])
        pt = wpct(np.log10(Tth[m]), w[m], [50, 95])
        o.update({f'{nm}_log10Rth_p50': p[0], f'{nm}_log10Rth_p95': p[1], f'{nm}_log10Rall_p50': pa[0],
                  f'{nm}_log10Rall_p95': pa[1], f'{nm}_log10Tth_p50': pt[0], f'{nm}_log10Tth_p95': pt[1]})
    # the stretched tube
    m = dn
    X = np.stack([C['x'][m], C['y'][m], C['z'][m]]); ww = w[m] / w[m].sum()
    xc = (X * ww).sum(1); dX = X - xc[:, None]
    cov = (dX * ww) @ dX.T
    lam = np.sort(np.linalg.eigvalsh(cov))[::-1]
    o.update(l_par=2 * np.sqrt(lam[0]), l_mid=2 * np.sqrt(lam[1]), l_perp=2 * np.sqrt(lam[2]),
             rho_mean=(C['rho'][m] * ww).sum(), M_dense=w[m].sum() / 7.47e-5)
    return o


def slicemap(fp, d, out_png, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    P = bc.read_binary(fp)
    t = float(P['time']); xb = bh_of(d)(t)
    g = np.asarray(P['mb_geometry']); n = P['nx1_out_mb']
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    ims = [None] * 3
    lo, hi = np.array([1e9, 1e9]), np.array([-1e9, -1e9])
    for b in range(len(g)):
        rho = np.asarray(P['mb_data']['dens'][b][0], dtype=np.float64)
        if (rho < DENSE).all():
            continue
        pr = np.asarray(P['mb_data']['press'][b][0], dtype=np.float64)
        rc, Tth, Rth, Tall, Rall = thermo(rho, pr)
        msk = rho < DENSE
        ext = [g[b, 0] - xb[0], g[b, 1] - xb[0], g[b, 2] - xb[1], g[b, 3] - xb[1]]
        lo = np.minimum(lo, [ext[0], ext[2]]); hi = np.maximum(hi, [ext[1], ext[3]])
        for k, (v, nrm, cm) in enumerate(((rc, LogNorm(1e0, 2e6), 'viridis'), (Tth, LogNorm(1e6, 3e9), 'inferno'),
                                          (Rth, LogNorm(1e-3, 1e3), 'coolwarm'))):
            ims[k] = ax[k].imshow(np.ma.array(v, mask=msk), origin='lower', extent=ext, norm=nrm, cmap=cm,
                                  interpolation='nearest')
    for k, s in enumerate(('rho [g/cm^3]', 'T_th [K]', 'P_rad/P_gas (T_th)')):
        if ims[k] is not None:
            fig.colorbar(ims[k], ax=ax[k], label=s)
        c0 = 0.5 * (lo + hi); hw = 0.55 * max(hi - lo) + 1.0   # the window around the dense gas (incl. the hole)
        c0 = 0.5 * (np.minimum(c0 - hw, 0) + np.maximum(c0 + hw, 0)); hw = max(hw, 0.5 * max(np.abs(c0) * 2) + 2)
        ax[k].set_xlim(c0[0] - hw, c0[0] + hw); ax[k].set_ylim(c0[1] - hw, c0[1] + hw); ax[k].set_aspect('equal')
        ax[k].set_xlabel('x - x_BH [M]'); ax[k].plot(0, 0, 'k+')
    ax[0].set_ylabel('y - y_BH [M]')
    fig.suptitle(f'{title}  t = {t:.1f} M  (z = 0; rho > 1e-6 rho_c)')
    fig.tight_layout(); fig.savefig(out_png, dpi=70); plt.close(fig)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    jobs, maps = [], []
    for spec in [a for a in sys.argv[2:] if not a.startswith('--')]:
        lab, rest = spec.split('=', 1)
        d, tp = rest.rsplit(':', 1); tp = float(tp)
        jobs += [(lab, d, f) for f in sorted(glob.glob(f'{d}/bin/{B}.prim_3d*.bin'))]
        sl = sorted(glob.glob(f'{d}/bin/{B}.prim_xy.*.bin'))
        ts = np.array([float(open(f, 'rb').read(200).split(b'time=')[1].split(b'\n')[0]) for f in sl])
        for tt in (tp, tp + 50, tp + 100, 300.0):
            maps.append((sl[int(np.argmin(np.abs(ts - tt)))], d, f'{out}/c129_map_{lab}_t{tt:.0f}.png', lab))
    with Pool(6) as p:
        p.starmap(slicemap, maps)
        if '--maps-only' in sys.argv:
            return
        R = p.map(dump, jobs)
    R.sort(key=lambda o: (o['label'], o['t']))
    keys = list(R[0].keys())
    with open(f'{out}/c129_physics.csv', 'w') as f:
        f.write(','.join(keys) + '\n')
        for o in R:
            f.write(','.join(f'{o[k]:.5g}' if not isinstance(o[k], str) else o[k] for k in keys) + '\n')


if __name__ == '__main__':
    main()
