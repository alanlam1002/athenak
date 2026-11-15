#!/usr/bin/env python3
"""c258 (R-257 item 3(i), user-approved 2026-11-12): the stream self-intersection radius r_I from Schwarzschild geodesics of
the most-bound debris (c254_extract.py's per-cell E, L; G = c = M_BH = 1; E = -u_t - 1, eps = 1 + E; L = |x_rel x u_cov|).
Orbit equation in u = 1/R (areal R): (du/dphi)^2 = f(u) = 2u^3 - u^2 + 2u/L^2 + (eps^2 - 1)/L^2 = 2 (u - u_a)(u_p - u)(u_3 - u),
u_a = 1/R2 (apocentre), u_p = 1/R1 (pericentre), u_3 = 1/2 - u_a - u_p. With u = (u_p+u_a)/2 + (u_p-u_a)/2 cos chi:
dphi = dchi / sqrt(2 (u_3 - u)); Phi = 2 phi(chi = pi) the azimuth per radial period, delta_omega = Phi - 2 pi.
Crossing: the debris leaving its first return (orbit 2, pericentre at azimuth Phi) meets the same-energy debris still falling in
on orbit 1 (azimuth psi in (Phi/2, Phi) from the first pericentre). At azimuth theta: r(Phi - theta) = r(theta - delta_omega)
(r(psi) symmetric about the apocentre), so theta = pi + delta_omega and both points sit at psi = pi from their own pericentre:
r_I = R(psi = pi) exactly (Newtonian limit: p/(1 - e cos(delta_omega/2)), R-257 §2(i)). Reported as areal R and isotropic
r_iso = (R - 1 + sqrt(R (R - 2)))/2 (the grid's radius from the hole).
Rows: the nominal (E = -1/(2 a_mb), a_mb = 200; L = 5, the r_p = 10 parabolic geodesic: L^2 = 2 r_p^2/(r_p - 2)) and, per NPZ,
the mass quantiles of r_I over the most-bound f of the bound debris (f = 0.1%, 1%, 10%) and at the peak-fallback energy band.
    c258_rI.py OUT.csv LABEL=NPZ [...]"""
import csv, sys
import numpy as np


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


  # copied from c254_fallback.py:32-48 (that module runs on import)

NCH = 4001
CH = np.linspace(0.0, np.pi, NCH)


def r_cross(eps, L):
    eps, L = np.atleast_1d(eps).astype(float), np.atleast_1d(L).astype(float)
    R3, R1, R2, three = cubic_roots(eps, L)
    ok = three & (R1 > 2.0) & (R2 > R1)
    out = dict(RI=np.full(eps.size, np.nan), dw=np.full(eps.size, np.nan), R1=R1, R2=R2, ok=ok)
    j = np.nonzero(ok)[0]
    for i0 in range(0, j.size, 20000):
        k = j[i0:i0 + 20000]
        ua, up = 1.0 / R2[k], 1.0 / R1[k]
        u3 = 0.5 - ua - up
        u = 0.5 * (up + ua)[:, None] + 0.5 * (up - ua)[:, None] * np.cos(CH)[None, :]
        g = 1.0 / np.sqrt(2.0 * (u3[:, None] - u))
        phi = np.concatenate([np.zeros((k.size, 1)), np.cumsum(0.5 * (g[:, 1:] + g[:, :-1]) * np.diff(CH), axis=1)], axis=1)
        half = phi[:, -1]
        out['dw'][k] = 2.0 * half - 2.0 * np.pi
        ii = np.clip((phi < np.pi).sum(1), 1, NCH - 1)          # phi(chi) is increasing: first index with phi >= pi
        rr = np.arange(k.size)
        p0, p1 = phi[rr, ii - 1], phi[rr, ii]
        c = CH[ii - 1] + (np.pi - p0) / (p1 - p0) * (CH[ii] - CH[ii - 1])
        uu = 0.5 * (up + ua) + 0.5 * (up - ua) * np.cos(c)
        out['RI'][k] = np.where(half > np.pi, 1.0 / uu, np.nan)   # half > pi: the orbit reaches azimuth pi before apocentre
    return out


def iso(R):
    return 0.5 * (R - 1.0 + np.sqrt(R * (R - 2.0)))


def wq(a, w, qs):
    o = np.argsort(a); c = np.cumsum(w[o]); c /= c[-1]
    return [float(a[o][min(np.searchsorted(c, q), a.size - 1)]) for q in qs]


def main():
    out, rows = sys.argv[1], []
    E0, L0 = -1.0 / 400.0, 5.0
    o = r_cross(1.0 + E0, L0)
    rows.append(dict(label='nominal', sel='E=-1/400,L=5', E_med=E0, L_med=L0, dw_med=o['dw'][0], R1=o['R1'][0], R2=o['R2'][0],
                     RI_q10=o['RI'][0], RI_q50=o['RI'][0], RI_q90=o['RI'][0], rI_iso_q50=iso(o['RI'][0]), m_frac=np.nan))
    print(rows[-1], flush=True)
    for s in sys.argv[2:]:
        lab, fn = s.split('=', 1)
        z = np.load(fn)
        E, dM = z['E'], z['dM']
        L = np.sqrt(z['Lx']**2 + z['Ly']**2 + z['Lz']**2)
        b = E < 0
        E, dM, L = E[b], dM[b], L[b]
        oE = np.argsort(E); cm = np.cumsum(dM[oE]) / dM.sum()
        sels = {f'most-bound {f*100:g}%': oE[:max(1, np.searchsorted(cm, f))] for f in (0.001, 0.01, 0.1)}
        # the peak-fallback band: Keplerian return time t = 2 pi/(2|E|)^1.5 within a factor 1.1 of the dM/dt peak
        tK = 2 * np.pi / (2 * np.abs(E))**1.5
        tb = np.logspace(3, 6, 121)
        h, _ = np.histogram(tK, tb, weights=dM)
        ip = np.argmax(h / np.diff(tb))
        sels['peak band'] = np.nonzero((tK >= tb[ip]) & (tK < tb[ip + 1]))[0]
        for nm, ix in sels.items():
            if ix.size > 200000:
                ix = np.random.default_rng(0).choice(ix, 200000, replace=False)
            o = r_cross(1.0 + E[ix], L[ix])
            ok = o['ok'] & np.isfinite(o['RI'])
            w = dM[ix][ok]
            q = wq(o['RI'][ok], w, (0.1, 0.5, 0.9))
            rows.append(dict(label=lab, sel=nm, E_med=wq(E[ix][ok], w, (0.5,))[0], L_med=wq(L[ix][ok], w, (0.5,))[0],
                             dw_med=wq(o['dw'][ok], w, (0.5,))[0], R1=wq(o['R1'][ok], w, (0.5,))[0],
                             R2=wq(o['R2'][ok], w, (0.5,))[0], RI_q10=q[0], RI_q50=q[1], RI_q90=q[2], rI_iso_q50=iso(q[1]),
                             m_frac=dM[ix].sum() / dM.sum() if nm == 'peak band' else float(nm.split()[1][:-1]) / 100))
            print({k: (f'{v:.5g}' if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    with open(out, 'w') as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    main()
