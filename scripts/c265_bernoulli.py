#!/usr/bin/env python3
"""c265 (user 2026-11-16): a metric (Bernoulli) bound/unbound criterion for the remnant, against the Newtonian bound set
(remnant.bound_mass) and the Hill radii (r_H, r_Hsc as c258). Offline, read-only, full-3D restarts (tde_dmde.cells).

Frame. The time vector of a frame comoving with the remnant, xi = d_t + w^i d_i (grid coordinates; the stored shift
beta' already carries the tracker's xidot), with
  free: w = V_c                           (non-rotating, the remnant's coordinate velocity; c258's V = alpha v - beta')
  rot : w = V_c + Omega x (x - x_c)       (Omega = (x_c - x_BH) x (V_c - V_BH) / r^2, V_BH = -xidot; a Jacobi analogue)
Per cell, -u_xi = alpha W - (beta' + w).u  ( = E + 1 - (xidot + w).u with tde_dmde's E, u_i ),
  ln B   = ln h + ln(-u_xi)      (Bernoulli, h = 1 + Gamma/(Gamma-1) P/rho)
  ln B1  = ln(-u_xi)             (the h = 1 variant: no thermal term)
For a parcel at rest in that frame, with CFC's gamma_ij = psi4 delta_ij exactly,
  Psi(x) = (1/2) ln(alpha^2 - psi4 |beta' + w|^2)   (the effective potential: self + hole + frame terms)
Uniform field. The frame does not accelerate with the remnant, so Psi has a linear part across it (~ M r_H / r^2, the size
of the self-binding). Its coefficient g comes from a dV-weighted least-squares fit Psi ~ a + g.(x - x_c) over the ball
|x - x_c| < r_Hsc: a monopole about x_c and the (even) tidal quadrupole drop out of g by symmetry. Psi, ln B and ln B1 all
get - g.(x - x_c).
Threshold. The corrected Psi along the line x_c -> x_BH (L1) and the opposite way (L2): cylinder radius --cyl, bins
--ds, dV-weighted bin means; the saddle is the first bin maximum beyond the core; Psi_s = min(Psi_L1, Psi_L2),
d_s = max(d_L1, d_L2). A cell is bound iff ln B_corr < Psi_s and |x - x_c| < d_s (the lobe; outside it the linear
correction is not meaningful).
Iteration: x_c, V_c (mass-weighted over the bound set) -> g, saddles -> the bound set, until M changes < 1e-4 (<= 8).
Hydrostatic check: a static star has h exp(Psi) = const (= exp(Psi_surface)), so over the innermost 50% of the bound mass
of the Newtonian set: sd_hyd = the mass-weighted spread of ln h + Psi_corr, dhyd = its mean - Psi_s (dhyd_out: the same
over the 50-90% shell): < 0 inside the lobe, > 0 overflowing it. fstatic (fstatic_h1): the Newtonian set's mass share
with ln h + Psi_corr < Psi_s (Psi_corr < Psi_s), i.e. bound if at rest. sfail: the saddle search failed on the last pass. well = Psi_s - min Psi_corr (r < R50/2). Mhist: M per iteration. Only cells with |x - x_c,N| < --ball
r_Hsc enter the metric variants (M_ball: their mass).
Also: M_ejB, the mass with -h u_t > 1 in the hole frame (tde_dmde's E_B > 0: unbound from the system).
    c265_bernoulli.py OUT.csv LABEL=RST:TP [...] [--nproc 2] [--cyl 0.3] [--ds 0.05] [--ball 4] [--itmax 12]"""
import argparse
import csv
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import remnant as RM   # noqa: E402
import tde_dmde as T   # noqa: E402

GAM, MSTAR, MBH = 5.0 / 3.0, 7.47e-5, 1.0
OPT = dict(cyl=0.3, ds=0.05, ball=4.0, itmax=12)


def hill(dM, r, rbh, M0):
    """c258's r_H (from M0) and the self-consistent r_Hsc, M_Hsc"""
    rH = rbh * (M0 / (3.0 * MBH))**(1.0 / 3.0)
    rh2, mh2 = rH, dM[r < rH].sum()
    for _ in range(50):
        r_new = rbh * (mh2 / (3.0 * MBH))**(1.0 / 3.0)
        m_new = dM[r < r_new].sum()
        done = abs(m_new - mh2) <= 1e-12 * max(mh2, 1e-300)
        rh2, mh2 = r_new, m_new
        if done:
            break
    return rH, rh2, mh2


def saddle(Pc, d, n, r_core):
    """the first maximum of the dV-weighted bin mean of Pc along +n beyond r_core; (Psi_s, distance) or (nan, nan)"""
    s = d.T @ n
    perp = np.sqrt(np.maximum((d * d).sum(0) - s * s, 0.0))
    q = (s > 0) & (perp < OPT['cyl'])
    if q.sum() < 10:
        return np.nan, np.nan
    e = np.arange(0.0, s[q].max() + OPT['ds'], OPT['ds'])
    k = np.digitize(s[q], e) - 1
    cnt = np.bincount(k, minlength=len(e))
    mean = np.bincount(k, weights=Pc[q], minlength=len(e)) / np.maximum(cnt, 1)
    mid = 0.5 * (e[:-1] + e[1:])[:len(mean)] if len(e) > 1 else e
    mean, mid, cnt = mean[:len(mid)], mid, cnt[:len(mid)]
    ok = cnt > 0
    mean, mid = mean[ok], mid[ok]
    for i in range(1, len(mean) - 1):   # first local max past the core (Psi rises out of the self-potential well)
        if mid[i] > r_core and mean[i] >= mean[i - 1] and mean[i] > mean[i + 1]:
            # require it to stay a max over +-3 bins (bin noise)
            lo, hi = max(0, i - 3), min(len(mean), i + 4)
            if mean[i] >= mean[lo:hi].max():
                return float(mean[i]), float(mid[i])
    return np.nan, np.nan


def classify(c, X, xc, Vc, Vbh, xbh, rHsc, rot, use_h, lnh, R50):
    d = X - xc[:, None]
    rr = np.sqrt((d * d).sum(0))
    w = np.repeat(Vc[:, None], X.shape[1], axis=1)
    Om = np.zeros(3)
    if rot:
        rv = xc - xbh
        Om = np.cross(rv, Vc - Vbh) / (rv @ rv)
        w = w + np.cross(Om, d.T).T
    U = np.stack([c['ux'], c['uy'], c['uz']])
    B = np.stack([c['betax'], c['betay'], c['betaz']])
    mut = c['E'] + 1.0 - ((c['xdot'][:, None] + w) * U).sum(0)
    bw = B + w
    arg = c['alpha']**2 - c['psi4'] * (bw * bw).sum(0)
    Psi = 0.5 * np.log(np.where(arg > 0, arg, np.nan))
    ball = (rr < rHsc) & np.isfinite(Psi)
    A = np.column_stack([np.ones(ball.sum()), d[:, ball].T])
    sw = np.sqrt(c['dV'][ball])
    coef = np.linalg.lstsq(A * sw[:, None], Psi[ball] * sw, rcond=None)[0]
    g = coef[1:]
    lin = g @ d
    Pc = Psi - lin
    nb = (xbh - xc) / np.linalg.norm(xbh - xc)
    fin = np.isfinite(Pc)
    PL1, dL1 = saddle(Pc[fin], d[:, fin], nb, 0.5 * R50)
    PL2, dL2 = saddle(Pc[fin], d[:, fin], -nb, 0.5 * R50)
    Ps = np.nanmin([PL1, PL2]) if np.isfinite([PL1, PL2]).any() else np.nan
    ds = np.nanmax([dL1, dL2]) if np.isfinite([dL1, dL2]).any() else np.nan
    lnB = np.log(np.maximum(mut, 1e-300)) + (lnh if use_h else 0.0) - lin
    bound = (lnB < Ps) & (rr < ds) & (mut > 0)
    return bound, dict(g=float(np.linalg.norm(g)), g_dot_n=float(g @ nb), Om=float(np.linalg.norm(Om)),
                       Psi_L1=PL1, d_L1=dL1, Psi_L2=PL2, d_L2=dL2, Psi_s=Ps, d_s=ds,
                       Psi_c=float(np.nanmin(Pc[rr < 0.5 * R50])) if (rr < 0.5 * R50).any() else np.nan), Pc


def one(arg):
    lab, fn, tp = arg
    c = T.cells(fn, rcut=3.0)
    X = np.stack([c['x'], c['y'], c['z']])
    dM, rho = c['dM'], c['rho']
    V = c['alpha'] * np.stack([c['vx'], c['vy'], c['vz']]) - np.stack([c['betax'], c['betay'], c['betaz']])
    xbh, xdot = np.asarray(c['xbh'], float), np.asarray(c['xdot'], float)
    Vbh = -xdot
    lnh = np.log1p(GAM / (GAM - 1.0) * c['P'] / rho)
    # Newtonian reference (c258)
    MbN, xcN, _, selN = RM.bound_mass(c)
    VcN = (V * dM * selN).sum(1) / (dM * selN).sum()
    rN = np.linalg.norm(X - xcN[:, None], axis=0)
    rbh = float(np.linalg.norm(xcN - xbh))
    rH, rHsc, MHsc = hill(dM, rN, rbh, MbN)
    o = np.argsort(rN[selN]); cm = np.cumsum(dM[selN][o]) / MbN
    R50 = float(rN[selN][o][np.searchsorted(cm, 0.5)])
    row = dict(label=lab, t=c['t'], t_minus_tp=c['t'] - tp, r_bh=rbh, M_N=MbN / MSTAR, r_H=rH, r_Hsc=rHsc,
               M_Hsc=MHsc / MSTAR, R50_N=R50, M_ejB=dM[c['EB'] > 0].sum() / MSTAR)
    # the metric variants only need the cells near the remnant: |x - x_c,N| < --ball r_Hsc (the lobe is ~ r_Hsc)
    sub = rN < OPT['ball'] * rHsc
    row['M_ball'] = dM[sub].sum() / MSTAR
    c = {k: (v[sub] if isinstance(v, np.ndarray) and v.shape == dM.shape else v) for k, v in c.items()}
    X, V, dM, lnh, selN = X[:, sub], V[:, sub], dM[sub], lnh[sub], selN[sub]
    for name, rot, use_h in (('free_h', False, True), ('free_h1', False, False), ('rot_h', True, True),
                             ('rot_h1', True, False)):
        sel, xc, Vc, Mb, it, hist = selN, xcN, VcN, MbN, 0, []
        for it in range(1, OPT['itmax'] + 1):
            bound, info, Pc = classify(c, X, xc, Vc, Vbh, xbh, rHsc, rot, use_h, lnh, R50)
            Mn = dM[bound].sum()
            hist.append(Mn / MSTAR)
            if Mn <= 0:   # nothing bound (first pass: M = 0; later: the saddle search failed, keep the last set)
                if it == 1:
                    sel, Mb = bound, 0.0
                break
            xc = (X * dM * bound).sum(1) / Mn
            Vc = (V * dM * bound).sum(1) / Mn
            conv = abs(Mn - Mb) < 1e-4 * Mb
            sel, Mb = bound, Mn
            if conv:
                break
        # hydrostatic diagnostics over the Newtonian set (not the variant's own set, which is biased to bound cells):
        # ln h + Psi_corr - Psi_s per cell; a static parcel with it < 0 is bound (kinetic term left out)
        rq = np.linalg.norm(X - xcN[:, None], axis=0)
        oN = np.argsort(rq[selN]); cN = np.cumsum(dM[selN][oN]) / dM[selN].sum()
        inner = oN[: max(1, np.searchsorted(cN, 0.5))]
        outer = oN[np.searchsorted(cN, 0.5):np.searchsorted(cN, 0.9)]
        q = (lnh + Pc - info['Psi_s'])[selN]
        mN = dM[selN]
        hyd, mw = q[inner], mN[inner]
        mu = np.average(hyd, weights=mw)
        row.update({f'{name}_dhyd_out': float(np.average(q[outer], weights=mN[outer])) if len(outer) else np.nan,
                    f'{name}_fstatic': float(mN[q < 0].sum() / mN.sum()),
                    f'{name}_fstatic_h1': float(mN[(Pc - info['Psi_s'])[selN] < 0].sum() / mN.sum()),
                    f'{name}_sfail': int(len(hist) > 1 and hist[-1] <= 0)})
        r = np.linalg.norm(X - xc[:, None], axis=0)
        o = np.argsort(r[sel]); cm = np.cumsum(dM[sel][o]) / max(Mb, 1e-300)
        row.update({f'{name}_M': Mb / MSTAR, f'{name}_it': it, f'{name}_Mhist': ' '.join(f'{m:.4g}' for m in hist),
                    f'{name}_dhyd': mu,
                    f'{name}_both': dM[sel & selN].sum() / MSTAR,
                    f'{name}_R90': float(r[sel][o][min(len(o) - 1, np.searchsorted(cm, 0.9))]) if sel.any() else np.nan,
                    f'{name}_sd_hyd': float(np.sqrt(np.average((hyd - mu)**2, weights=mw))),
                    f'{name}_well': info['Psi_s'] - info['Psi_c']})
        row.update({f'{name}_{k}': v for k, v in info.items()})
    print({k: (f'{v:.5g}' if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--nproc', type=int, default=2)
    ap.add_argument('--cyl', type=float, default=0.3)
    ap.add_argument('--ds', type=float, default=0.05)
    ap.add_argument('--ball', type=float, default=4.0)
    ap.add_argument('--itmax', type=int, default=12)
    a = ap.parse_args()
    OPT.update(cyl=a.cyl, ds=a.ds, ball=a.ball, itmax=a.itmax)
    args = []
    for s in a.runs:
        lab, rest = s.split('=', 1)
        fn, tp = rest.rsplit(':', 1)
        args.append((lab, fn, float(tp)))
    with Pool(a.nproc) as p:
        rows = p.map(one, args, chunksize=1)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(a.out, 'w') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
