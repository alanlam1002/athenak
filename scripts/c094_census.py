#!/usr/bin/env python3
"""C-094 (R-097 items 1(a) and 3): the near-hole lapse census from a restart (offline).

From a restart's stored ADM state (alpha, g_dd = psi^4 delta_ij, vK_dd = Ahat_ij psi^-2;
cfc_reconstruct.cpp AssembleConformalMetric / AssembleLapseShiftK) and conserved state, for
cells with r_BH <= RMAX (r_BH from x_BH(t), tde_dmde.frame_state; m_bh = <cfc> puncture_mass of
the restart's parameter copy):
  psi, alpha, Ahat^2 = sum_ij Ahat_ij^2 (flat), Utilde = cons(IEN) + cons(IDN) (cfc.cpp l. 2241),
  Stilde = sqrt(g) (S_i v^i + 3P) (hydro part of RescaleMatterSources; P, v from the ideal-gas
  inversion tde_dmde.c2p -- ARGUED approximation, the code's C2P/floors are not replayed),
  K = 2 pi (Utilde + 2 Stilde) psi^-2 + (7/8) Ahat^2 psi^-8 (mg_cfc_lapse.cpp l. 103-108),
and two backgrounds:
  'true' = the analytic trumpet (areal radius from a converged root-find), and
  'pinned' = the same formulas with the areal radius pinned at the throat (R = 1.5 m_bh) for
             r_iso < 1.5 m_bh -- the hypothesis from the vacuum runs (c094_vacnull.py),
for each: psi0, v0 = alpha0 psi0, Ahat0^2 (with the psi^6 cap, cfc_puncture.hpp), K0, x = dpsi/psi0,
u = alpha psi - v0 (alpha is post-floor, so u is a lower bound where alpha = alpha_floor).
Prints radial-bin summaries and the +x/-x/+y/-y/+z/-z lines (cells within one dx of the line).

    c094_census.py RESTART [--rmax 3] [--lines]
"""
import argparse
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tde_dmde as D                               # noqa: E402
import tde_elliptical_orbit_ic as TR               # noqa: E402

PSI6MAX = 163.0407655498103                        # cfc_puncture.hpp kPsi6Max


def bg(r, m, pinned):
    """psi0, alpha0, Ahat0^2 of TrumpetBackground at iso radius r (array)"""
    R = np.array([TR.trumpet_iso_to_areal(c / m) for c in r]) * m
    if pinned:
        R = np.where(r < 1.5 * m, 1.5 * m, R)
    rrs = R / m
    psi0 = np.sqrt(R / np.sqrt(r * r + 1e-30))
    a2 = 1.0 - 2.0 / rrs + 1.6875 / rrs**4
    al0 = np.sqrt(np.maximum(a2, 0.0))
    p6 = np.minimum(psi0**6, PSI6MAX)
    A0sq = 10.125 * m**4 / R**6 * p6 * p6
    return psi0, al0, A0sq


def load(fn, rmax, nscal=1):
    t = D.header(fn)
    nmb = t['nmb']
    x1min, x2min, x3min, x1max, x2max, x3max = t['size'][:6]
    nx, ng = 16, 4
    no = nx + 2 * ng
    nmhd = 5 + nscal
    ncc = no**3
    bsz = (no + 1) * no * no * 3
    if (nmhd * ncc + bsz + 17 * ncc) * 8 != t['dsize']:
        raise SystemExit('unexpected restart layout')
    mm = np.memmap(fn, dtype=np.float64, mode='r', offset=t['data_off'], shape=(nmb, t['dsize'] // 8))
    lev = t['ll'][:, 3]
    w = np.array([x1max - x1min, x2max - x2min, x3max - x3min])
    lo = np.array([x1min, x2min, x3min])
    xbh, _ = D.frame_state(t['par'], t['time'])
    s = slice(ng, ng + nx)
    ic = np.arange(nx) + 0.5
    acc = {k: [] for k in ('x', 'y', 'z', 'dx', 'D', 'tau', 'Sx', 'Sy', 'Sz', 'g', 'A2', 'al')}
    for b in range(nmb):
        ext = w / 2.0**lev[b]
        dx = ext / nx
        blo = lo + t['ll'][b, :3] * ext
        if np.any(blo - xbh > rmax) or np.any(blo + ext - xbh < -rmax):
            continue
        rec = mm[b]
        cons = rec[:nmhd * ncc].reshape(nmhd, no, no, no)[:, s, s, s]
        adm = rec[nmhd * ncc + bsz:].reshape(17, no, no, no)[:, s, s, s]
        x, y, z = (blo[a] + ic * dx[a] for a in range(3))
        Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
        keep = np.sqrt((X - xbh[0])**2 + (Y - xbh[1])**2 + (Z - xbh[2])**2) <= rmax
        if not keep.any():
            continue
        gxx = adm[0]
        Kd = adm[6:12]                 # vK_dd = a_dd psi^-2 -> a_dd = psi^2 vK_dd = sqrt(gxx) vK
        a = Kd * np.sqrt(gxx)
        A2 = a[0]**2 + a[3]**2 + a[5]**2 + 2 * (a[1]**2 + a[2]**2 + a[4]**2)
        for k, v in (('x', X), ('y', Y), ('z', Z), ('D', cons[0]), ('tau', cons[4]), ('Sx', cons[1]),
                     ('Sy', cons[2]), ('Sz', cons[3]), ('g', gxx), ('A2', A2), ('al', adm[13])):
            acc[k].append(v[keep])
        acc['dx'].append(np.full(keep.sum(), dx[0]))
    c = {k: np.concatenate(v) for k, v in acc.items()}
    c['rx'], c['ry'], c['rz'] = c['x'] - xbh[0], c['y'] - xbh[1], c['z'] - xbh[2]
    c['r'] = np.sqrt(c['rx']**2 + c['ry']**2 + c['rz']**2)
    c['t'], c['xbh'], c['par'] = t['time'], xbh, t['par']
    return c


def parv(par, key, default=0.0):
    m = re.search(rf'^{key}\s*=\s*([-+.\deE]+)', par, re.M)
    return float(m.group(1)) if m else default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rst')
    ap.add_argument('--rmax', type=float, default=3.0)
    ap.add_argument('--lines', action='store_true')
    ap.add_argument('--out')
    a = ap.parse_args()
    c = load(a.rst, a.rmax)
    par = c['par']
    m = parv(par, 'puncture_mass', 1.0)
    afl = parv(par, 'alpha_floor', 0.0)
    exl = parv(par, 'excise_lapse', 0.0)
    zr, zrho = parv(par, 'atmosphere_zone_radius', 0.0), parv(par, 'atmosphere_zone_rho', 0.0)
    psi = c['g']**0.25
    sg = psi**6
    Dd, tau = c['D'] / sg, np.maximum(c['tau'] / sg, 0.0)
    Sx, Sy, Sz = c['Sx'] / sg, c['Sy'] / sg, c['Sz'] / sg
    ig = 1.0 / c['g']
    Ssq = ig * (Sx * Sx + Sy * Sy + Sz * Sz)
    rho, P, W, h = D.c2p(np.maximum(Dd, 1e-300), Ssq, tau)
    Sv = Ssq / (np.maximum(Dd, 1e-300) * h * W)           # S_i v^i, v^i = gamma^ij S_j/(D h W)
    Ut = c['tau'] + c['D']
    St = sg * (Sv + 3.0 * P)
    U2S = Ut + 2.0 * St
    K = 2 * math.pi * U2S / psi**2 + 0.875 * c['A2'] / psi**8
    r = c['r']
    fl = c['al'] <= 1.0000001 * afl
    out = [f'{a.rst}: t {c["t"]:.3f} x_BH {np.round(c["xbh"], 4).tolist()} m_bh {m:.9f} '
           f'alpha_floor {afl} excise_lapse {exl} zone r_s {zr} rho_thr {zrho}',
           f'cells r <= {a.rmax}: {len(r)}; alpha at floor: {fl.sum()} '
           f'(r range {r[fl].min():.4f} .. {r[fl].max():.4f}); unfloored cells with r < 1.5 m: '
           f'{((r < 1.5 * m) & ~fl).sum()}; floored cells with r >= 1.5 m: {((r >= 1.5 * m) & fl).sum()}; '
           f'min unfloored r {r[~fl].min():.4f}']
    for name, pin in (('true', False), ('pinned', True)):
        psi0, al0, A0sq = bg(r, m, pin)
        v0 = al0 * psi0
        K0 = 0.875 * A0sq / psi0**8
        x = psi / psi0 - 1.0
        u = c['al'] * psi - v0
        out.append(f'-- background "{name}"')
        out.append('  r-bin        n  |psi/psi0-1|max  med(u/v0)   max|U+2S|*2pi/psi^2  max|7/8 dA2/psi^8|  '
                   'med K0   med K   med|K-K0|/K0  floored')
        edges = [0, 0.3, 0.6, 0.9, 1.2, 1.4, 1.5 * m, 1.6, 2.0, 2.5, 3.0]
        for lo_, hi_ in zip(edges[:-1], edges[1:]):
            s = (r >= lo_) & (r < hi_)
            if not s.any():
                continue
            mat = 2 * math.pi * np.abs(U2S[s]) / psi[s]**2
            dA = 0.875 * np.abs(c['A2'][s] - A0sq[s]) / psi[s]**8
            rel = np.abs(K[s] - K0[s]) / np.maximum(K0[s], 1e-300)
            uv = np.where(v0[s] > 0, u[s] / np.maximum(v0[s], 1e-300), np.nan)
            out.append(f'  [{lo_:4.2f},{hi_:4.2f}) {s.sum():6d}  {np.abs(x[s]).max():.3e}  '
                       f'{np.nanmedian(uv) if np.isfinite(uv).any() else float("nan"):+.3e}  {mat.max():.3e}  '
                       f'{dA.max():.3e}  {np.median(K0[s]):.3e} {np.median(K[s]):.3e}  {np.median(rel):.3e}  '
                       f'{fl[s].sum()}')
        if a.lines and name == 'true':
            pp0, pal0, pA0 = bg(r, m, True)
            out.append('  lines: dir r  rho  alpha  alpha0  psi  psi0(true)  psi0(pinned)  U+2S  '
                       'A2  A0^2(true)  A0^2(pinned)  K  K0(true)  K0(pinned)  class')
            for dn, ax_, sg_ in (('+x', 'rx', 1), ('-x', 'rx', -1), ('+y', 'ry', 1), ('-y', 'ry', -1),
                                 ('+z', 'rz', 1), ('-z', 'rz', -1)):
                others = [q for q in ('rx', 'ry', 'rz') if q != ax_]
                s = (sg_ * c[ax_] > 0) & (np.abs(c[others[0]]) < c['dx']) & (np.abs(c[others[1]]) < c['dx'])
                for j in np.nonzero(s)[0][np.argsort(r[s])]:
                    cls = ('excised(a<%.2f)' % exl if c['al'][j] < exl else
                           ('zone-gate' if (r[j] < zr and rho[j] < zrho) else 'matter/atm'))
                    out.append(f'   {dn} {r[j]:6.3f} {rho[j]:.2e} {c["al"][j]:.4e} {al0[j]:.4e} '
                               f'{psi[j]:.5f} {psi0[j]:.5f} {pp0[j]:.5f} {U2S[j]:.2e} {c["A2"][j]:.4e} '
                               f'{A0sq[j]:.4e} {pA0[j]:.4e} {K[j]:.4e} {K0[j]:.4e} '
                               f'{0.875 * pA0[j] / pp0[j]**8:.4e} {cls}')
    txt = '\n'.join(out)
    print(txt)
    if a.out:
        open(a.out, 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
