#!/usr/bin/env python3
"""R-140 item 5 (D7 offline, c126): the size of the CFC residual metric, the stream self-gravity
parameter Lambda = rho/rho_tid, and (rp18) the remnant's self-boundness, from the rp10/rp18 2x2 runs.

(a) Residual metric. The code assembles psi = psi0 + delta_psi and alpha = [(alpha psi)0 + delta(alpha psi)]/psi
    at cell centres (cfc_reconstruct.cpp:221-234, AssembleLapseShiftK), with the trumpet background
    psi0 = sqrt(R/r), alpha0 = sqrt(1 - 2/R + 27/(16 R^4)) (cfc_puncture.hpp:122-136, M_BH = 1), R the areal
    radius of the isotropic r = |x - x_BH| (TrumpetIsoToAreal, cfc_puncture.hpp:73-101). The restarts carry
    the assembled psi4 and alpha (ADM slots 12, 13; tde_dmde.cells), so
        delta_psi/psi0 = psi4^(1/4)/psi0 - 1,   delta(alpha psi)/(alpha psi)0 = alpha psi/(alpha0 psi0) - 1,
    with psi0, alpha0 recomputed here (the same closed forms, R by bisection). Cells at the alpha floor
    (alpha == alpha_floor, cfc_reconstruct.cpp:234) are dropped from the (alpha psi) residual.
    Restarts: mass-weighted percentiles over bound (E < 0) and unbound debris (rho > 1e-10 rho_c, r >= 3),
    the dense gas (rho > 1e-3 rho_c), and volume-weighted radial profiles near the hole (all cells).
    Slices (adm_xy + prim_xy every 2 M, z = 0, float32): the z = 0 rho-weighted mean and max over the
    debris, the value at the density maximum, and shell means at r = 2, 3, 5, 10 against t.
(b) Lambda = rho / rho_tid, rho_tid = M_BH/(2 pi r^3), r = the ISOTROPIC coordinate distance to x_BH (the grid
    radius). Convention: the tidal density at which the stream's self-gravity balances the tide for a
    cylinder is O(1) times this (Lambda = 1 is the nominal threshold, not a sharp one); the areal radius
    R = r (1 + 1/2r)^2 (far field) would raise rho_tid^-1 by (R/r)^3 = 1.18 at r = 18, 1.09 at r = 50: reported
    as Lambda_R > 1 alongside. From the 3D dumps (prim_3d every 12 M + periapsis), mass-weighted over bound and
    unbound debris (far-field proxy E = alpha W - 1 and dm = rho W psi^6 dV as c119_domain.py).
(c) rp18 remnant (and the others): remnant.bound_mass over the rho > 1e-10 rho_c gas outside r = 3 (iterative Newtonian energy criterion about the clump,
    monopole self-potential, no tide) on each 3D dump, plus the same set cut to the Hill sphere
    d < r_H = r_BH (M_rem/3)^(1/3) about the clump's centre (argued: the tide is the missing term).
    c126_residual_lambda.py OUTDIR LABEL=RUNDIR [...] [--parts rst,slice,dump,plot]
"""
import argparse
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
import tde_dmde as TD                             # noqa: E402
import remnant as RM                              # noqa: E402

B = 'cfc_tde_wd_imbh_parabolic_tracker'
RHOC = 3.24e-4
THR = 1e-10 * RHOC
MSTAR = 7.47e-5
GAM = 5.0 / 3.0
AFLOOR = 1.0e-6


def iso_of_areal(R):
    """TrumpetArealToIso (cfc_puncture.hpp:50-59), vectorised"""
    t = 2.0 * R - 3.0
    s1 = np.sqrt(4 * R * R + 4 * R + 3); s2 = np.sqrt(8 * R * R + 8 * R + 6)
    q = np.clip((4 + 3 * np.sqrt(2.0)) * t / (8 * R + 6 + 3 * s2), 0.0, None)
    return np.where(t <= 0, 0.0, 0.25 * (2 * R + 1 + s1) * q**(1 / np.sqrt(2.0)))


def areal_of_iso(r):
    """TrumpetIsoToAreal by bisection on [1.5, r + 1.5] (the code's bracket, cfc_puncture.hpp:87-88)"""
    lo = np.full_like(r, 1.5); hi = np.maximum(1.5, r + 1.5)
    for _ in range(64):
        m = 0.5 * (lo + hi)
        f = iso_of_areal(m) > r
        hi = np.where(f, m, hi); lo = np.where(f, lo, m)
    return 0.5 * (lo + hi)


def background(r):
    R = areal_of_iso(r)
    psi0 = np.sqrt(R / np.maximum(r, 1e-30))
    a2 = 1.0 - 2.0 / R + 1.6875 / R**4
    return R, psi0, np.sqrt(np.clip(a2, 0.0, None))


def wpct(x, w, q):
    if len(x) == 0 or w.sum() <= 0:
        return [np.nan] * len(q)
    o = np.argsort(x); c = np.cumsum(w[o]); c /= c[-1]
    return [x[o][min(np.searchsorted(c, qq / 100.0), len(x) - 1)] for qq in q]


def is_moving(d):
    return re.search(r'^puncture_moving\s*=\s*true', open(f'{d}/parfile.par').read(), re.M) is not None


def bh_of(d):
    """x_BH(t) and xidot(t) from the user history (c119_domain.bh_of, without its matplotlib import)"""
    par = open(f'{d}/parfile.par').read()
    px = float(re.search(r'^puncture_x1\s*=\s*(\S+)', par, re.M).group(1))
    if not is_moving(d):
        return lambda t: (np.zeros(3), np.zeros(3))
    h = np.loadtxt(f'{d}/{B}.user.hst')
    return lambda t: (np.array([px, 0, 0]) - np.array([np.interp(t, h[:, 0], h[:, 7 + k]) for k in range(3)]),
                      np.array([np.interp(t, h[:, 0], h[:, 4 + k]) for k in range(3)]))


# ---------------------------------------------------------------------------------------------- (a) restarts
RBINS = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0])
Q = [5, 50, 95, 99]


def rst(arg):
    lab, fp = arg
    c = TD.cells(fp, rcut=0.0, nscal=1)
    xb = c['xbh']
    r = np.sqrt((c['x'] - xb[0])**2 + (c['y'] - xb[1])**2 + (c['z'] - xb[2])**2)
    R, psi0, a0 = background(r)
    psi = c['psi4']**0.25
    dpsi = psi / psi0 - 1.0
    okap = (c['alpha'] > AFLOOR) & (a0 > 0)
    dap = np.where(okap, c['alpha'] * psi / np.where(a0 > 0, a0 * psi0, 1.0) - 1.0, np.nan)
    rows = []
    sets = {'bound': (c['rho'] > THR) & (r >= 3) & (c['E'] < 0),
            'unbound': (c['rho'] > THR) & (r >= 3) & (c['E'] >= 0),
            'dense': (c['rho'] > 1e-3 * RHOC) & (r >= 3)}
    for nm, m in sets.items():
        w = c['dM'][m]
        for q, x in (('dpsi', dpsi[m]), ('dap', dap[m])):
            k = np.isfinite(x)
            p = wpct(x[k], w[k], Q)
            pa = wpct(np.abs(x[k]), w[k], [50, 99])
            rows.append(dict(label=lab, t=c['t'], set=nm, q=q, mass=w.sum(), p5=p[0], p50=p[1], p95=p[2], p99=p[3],
                             abs50=pa[0], abs99=pa[1], absmax=np.abs(x[k]).max() if k.any() else np.nan))
    prof = []
    for a, b in zip(RBINS[:-1], RBINS[1:]):
        m = (r >= a) & (r < b)
        if not m.any():
            continue
        v = c['dV'][m]
        for q, x in (('dpsi', dpsi[m]), ('dap', dap[m])):
            k = np.isfinite(x)
            p = wpct(x[k], v[k], [5, 50, 95]) if k.any() else [np.nan] * 3
            prof.append(dict(label=lab, t=c['t'], r_lo=a, r_hi=b, q=q, ncell=int(k.sum()), p5=p[0], p50=p[1],
                             p95=p[2], absmax=np.abs(x[k]).max() if k.any() else np.nan,
                             floored=int((~okap[m]).sum()) if q == 'dap' else 0))
    return rows, prof


# ---------------------------------------------------------------------------------------------- (a') slices
def slice_pair(arg):
    lab, d, fa, fp = arg
    A = bc.read_binary(fa); P = bc.read_binary(fp)
    t = float(A['time'])
    xb, _ = bh_of(d)(t)
    g = np.asarray(A['mb_geometry']); n = A['nx1_out_mb']
    gp = np.asarray(P['mb_geometry'])
    assert len(g) == len(gp) and np.allclose(g, gp), (fa, fp)
    acc = dict(w=0.0, wdp=0.0, wda=0.0, mdp=0.0, mda=0.0)
    shells = {rr: [] for rr in (2.0, 3.0, 5.0, 10.0)}
    best = (-1.0, np.nan, np.nan, np.nan)
    for b in range(len(g)):
        h = (g[b, 1] - g[b, 0]) / n
        xc = g[b, 0] + (np.arange(n) + 0.5) * h
        yc = g[b, 2] + (np.arange(n) + 0.5) * h
        Y, X = np.meshgrid(yc, xc, indexing='ij')
        z0 = g[b, 4] + (np.floor((0.0 - g[b, 4]) / h) + 0.5) * h   # centre of the cell holding z = 0
        Z = z0 * np.ones_like(X)
        r = np.sqrt((X - xb[0])**2 + (Y - xb[1])**2 + (Z - xb[2])**2)
        R, psi0, a0 = background(r)
        psi = np.asarray(A['mb_data']['adm_psi4'][b][0], dtype=np.float64)**0.25
        al = np.asarray(A['mb_data']['adm_alpha'][b][0], dtype=np.float64)
        dp = psi / psi0 - 1
        ok = (al > 1.001 * AFLOOR) & (a0 > 0)
        da = np.where(ok, al * psi / np.where(a0 > 0, a0 * psi0, 1) - 1, np.nan)
        rho = np.asarray(P['mb_data']['dens'][b][0], dtype=np.float64)
        m = (rho > THR) & (r >= 3) & ok
        w = rho * h * h
        acc['w'] += w[m].sum(); acc['wdp'] += (w * np.abs(dp))[m].sum(); acc['wda'] += (w * np.abs(da))[m].sum()
        if m.any():
            acc['mdp'] = max(acc['mdp'], np.abs(dp[m]).max()); acc['mda'] = max(acc['mda'], np.abs(da[m]).max())
        im = np.unravel_index(np.argmax(rho), rho.shape)
        if rho[im] > best[0]:
            best = (rho[im], dp[im], da[im], r[im])
        for rr in shells:
            k = (np.abs(r - rr) < 0.5 * h + 0.25) & ok
            if k.any():
                shells[rr].append(np.c_[dp[k], da[k]])
    out = dict(label=lab, t=t, mean_abs_dpsi=acc['wdp'] / max(acc['w'], 1e-300),
               mean_abs_dap=acc['wda'] / max(acc['w'], 1e-300), max_abs_dpsi=acc['mdp'], max_abs_dap=acc['mda'],
               rhomax=best[0], dpsi_at_rhomax=best[1], dap_at_rhomax=best[2], r_rhomax=best[3])
    for rr, v in shells.items():
        v = np.concatenate(v) if v else np.full((1, 2), np.nan)
        out[f'dpsi_r{rr:g}'] = np.nanmean(v[:, 0]); out[f'dap_r{rr:g}'] = np.nanmean(v[:, 1])
    return out


# ---------------------------------------------------------------------------------------------- (b, c) dumps
LQ = [5, 25, 50, 75, 95]


def dump(arg):
    lab, d, fp = arg
    P = bc.read_binary(fp)
    t = float(P['time'])
    xb, _ = bh_of(d)(t)
    g = np.asarray(P['mb_geometry']); n = P['nx1_out_mb']
    cols = {k: [] for k in ('x', 'y', 'z', 'rho', 'P', 'vx', 'vy', 'vz', 'E', 'dM', 'r')}
    for b in range(len(g)):
        h = (g[b, 1] - g[b, 0]) / n
        c = lambda lo: lo + (np.arange(n) + 0.5) * h
        Z, Y, X = np.meshgrid(c(g[b, 4]), c(g[b, 2]), c(g[b, 0]), indexing='ij')
        r = np.sqrt((X - xb[0])**2 + (Y - xb[1])**2 + (Z - xb[2])**2)
        rr = np.maximum(r, 0.6)
        psi = 1 + 0.5 / rr
        al = (1 - 0.5 / rr) / psi
        rho = np.asarray(P['mb_data']['dens'][b], dtype=np.float64)
        keep = (rho > THR) & (r >= 3)           # the floor atmosphere and the excised hole carry no debris
        if not keep.any():
            continue
        u = [np.asarray(P['mb_data'][k][b], dtype=np.float64) for k in ('velx', 'vely', 'velz')]
        W = np.sqrt(1 + psi**4 * sum(x * x for x in u))
        for k, v in (('x', X), ('y', Y), ('z', Z), ('rho', rho), ('vx', u[0] / W), ('vy', u[1] / W),
                     ('vz', u[2] / W), ('E', al * W - 1), ('dM', rho * W * psi**6 * h**3), ('r', r),
                     ('P', np.asarray(P['mb_data']['press'][b], dtype=np.float64))):
            cols[k].append(v[keep])
    C = {k: np.concatenate(v) for k, v in cols.items()}
    r = C['r']; rho = C['rho']
    lam = rho * 2 * np.pi * r**3                    # M_BH = 1
    R, _, _ = background(r)
    lamR = rho * 2 * np.pi * R**3
    out = dict(label=lab, t=t)
    # (c) the remnant (rho > 1e-10 rho_c gas outside r < 3; remnant.bound_mass needs rho, P, dM, x, v)
    c = {q: C[q] for q in ('rho', 'P', 'dM', 'x', 'y', 'z', 'vx', 'vy', 'vz')}
    Mb, xc, vc, sel = RM.bound_mass(c)
    rbh = np.linalg.norm(xc - xb)
    rH = rbh * (Mb / 3.0)**(1.0 / 3.0)
    dd = np.sqrt((c['x'] - xc[0])**2 + (c['y'] - xc[1])**2 + (c['z'] - xc[2])**2)
    rem = sel & (dd < rH)
    imax = np.argmax(c['rho'])
    out.update(rem_Mself=Mb / MSTAR, rem_rbh=rbh, rem_rH=rH, rem_MselfHill=c['dM'][rem].sum() / MSTAR,
               rem_Mhill_all=c['dM'][dd < rH].sum() / MSTAR, rhomax=c['rho'][imax] / RHOC,
               rem_Lambda_c=c['rho'][imax] * 2 * np.pi * rbh**3)
    # (b) Lambda over the debris, and over the stream (the debris less the self-bound remnant within r_H)
    for nm, m in (('bound', C['E'] < 0), ('unbound', C['E'] >= 0),
                  ('sbound', (C['E'] < 0) & ~rem), ('sunbound', (C['E'] >= 0) & ~rem)):
        w = C['dM'][m]
        out[f'{nm}_mass'] = w.sum()
        p = wpct(np.log10(np.maximum(lam[m], 1e-300)), w, LQ)
        for q, v in zip(LQ, p):
            out[f'{nm}_log10L_p{q}'] = v
        out[f'{nm}_fL1'] = w[lam[m] > 1].sum() / max(w.sum(), 1e-300)
        out[f'{nm}_fL10'] = w[lam[m] > 10].sum() / max(w.sum(), 1e-300)
        out[f'{nm}_fLR1'] = w[lamR[m] > 1].sum() / max(w.sum(), 1e-300)
    return out


def write(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, 'w') as f:
        f.write(','.join(keys) + '\n')
        for o in rows:
            f.write(','.join(f'{o[k]:.6g}' if isinstance(o[k], (float, np.floating)) else str(o[k]) for k in keys) + '\n')


def plot(out):
    """figures from the csvs: residual vs t (slices), Lambda and the remnant vs t (dumps)"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    f = f'{out}/c126_residual_slices.csv'
    if os.path.exists(f):
        d = pd.read_csv(f)
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        for lab, s in d.groupby('label'):
            ls = '-' if lab.split('_')[1][0] == 'C' else '--'
            ax[0].semilogy(s.t, s.max_abs_dpsi, ls, label=lab)
            ax[1].semilogy(s.t, s.max_abs_dap, ls, label=lab)
            ax[2].semilogy(s.t, np.abs(s.dpsi_r2), ls, label=lab)
        for a, tt in zip(ax, ('max |dpsi/psi0| over z=0 debris', 'max |d(alpha psi)/(alpha psi)0| over z=0 debris',
                              '|dpsi/psi0| at r_BH = 2 (shell mean, z=0)')):
            a.set_title(tt, fontsize=9); a.set_xlabel('t [M]')
        ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(f'{out}/c126_residual_vs_t.png', dpi=75); plt.close(fig)
    f = f'{out}/c126_lambda_remnant.csv'
    if os.path.exists(f):
        d = pd.read_csv(f)
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        for lab, s in d.groupby('label'):
            ls = '-' if lab.split('_')[1][0] == 'C' else '--'
            ax[0].plot(s.t, s.bound_log10L_p50, ls, label=lab + ' bound')
            ax[0].plot(s.t, s.unbound_log10L_p50, ls, alpha=0.4)
            ax[1].plot(s.t, s.bound_fL1, ls, label=lab)
            ax[2].plot(s.t, s.rem_MselfHill, ls, label=lab)
        ax[0].axhline(0, color='k', lw=0.5)
        ax[0].set_title('median log10 Lambda (bound solid-color, unbound faint)', fontsize=9)
        ax[1].set_title('bound-debris mass fraction with Lambda > 1', fontsize=9)
        ax[2].set_title('self-bound remnant within r_H [M_*]', fontsize=9)
        for a in ax:
            a.set_xlabel('t [M]')
        ax[2].legend(fontsize=7); fig.tight_layout(); fig.savefig(f'{out}/c126_lambda_remnant.png', dpi=75)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--parts', default='rst,slice,dump')
    ap.add_argument('--nproc', type=int, default=12)
    a = ap.parse_args()
    D = dict(s.split('=', 1) for s in a.runs)
    parts = a.parts.split(',')
    os.makedirs(a.out, exist_ok=True)
    if 'rst' in parts:
        jobs = [(lab, f) for lab, d in D.items() for f in sorted(glob.glob(f'{d}/rst/{B}.*.rst'))]
        with Pool(min(4, a.nproc)) as p:
            R = p.map(rst, jobs)
        write(f'{a.out}/c126_residual_rst.csv', sorted([o for x in R for o in x[0]], key=lambda o: (o['label'], o['t'])))
        write(f'{a.out}/c126_residual_profile.csv', sorted([o for x in R for o in x[1]],
                                                           key=lambda o: (o['label'], o['t'], o['q'], o['r_lo'])))
    if 'slice' in parts:
        jobs = []
        for lab, d in D.items():
            for fa in sorted(glob.glob(f'{d}/bin/{B}.adm_xy.*.bin')):
                fp = fa.replace('adm_xy', 'prim_xy')
                if os.path.exists(fp):
                    jobs.append((lab, d, fa, fp))
        with Pool(a.nproc) as p:
            S = p.map(slice_pair, jobs)
        write(f'{a.out}/c126_residual_slices.csv', sorted(S, key=lambda o: (o['label'], o['t'])))
    if 'dump' in parts:
        jobs = [(lab, d, f) for lab, d in D.items() for f in sorted(glob.glob(f'{d}/bin/{B}.prim_3d*.bin'))]
        with Pool(min(8, a.nproc)) as p:
            S = p.map(dump, jobs)
        write(f'{a.out}/c126_lambda_remnant.csv', sorted(S, key=lambda o: (o['label'], o['t'])))
    if 'plot' in parts:
        plot(a.out)


if __name__ == '__main__':
    main()
