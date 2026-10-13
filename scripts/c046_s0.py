#!/usr/bin/env python3
"""C-046 (R-054 S0): offline what-if for R-053's near-BH "grid-rest" atmosphere zone.

For non-excised cells within r_s of x_BH with rho < rho_thr, substitute the Eulerian velocity
v^i = beta'^i/alpha (gas at rest on the grid; capped at |v| <= v_max along beta') and a cold
P (cs = 0; the floor's P/rho = tfloor ~ 1e-22 gives cs ~ 1e-11), then recompute the CFL dt as
in c045_dt.py (dyn_grmhd_newdt.cpp, gr_dt = true, B = 0, excised cells dropped, the comoving
floor-velocity substitution kept for cells outside the zone). With cs = 0 the zone cells'
speed is |alpha v^i - beta'^i|: zero where uncapped, (1 - v_max alpha/|beta'|_g) |beta'^i| where
the cap acts.

Per restart and per (rho_thr, r_s, v_max): dt (and the gain over the baseline), the limiting
cell (in or out of the zone; role, level, r_BH, |lambda|, alpha-term, shift), the zone's cells,
mass, |momentum| and kinetic energy (W - 1) D before the reset, and the dense gas the gate
protects (mass and max rho/rho_c of rho >= rho_thr cells within r_s; by construction none of
them is in the zone -- asserted).

    c046_s0.py [--rs-scan] [--out DIR]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tde_dmde as D                              # noqa: E402
import remnant as RM                              # noqa: E402
from c045_dt import rst_at, logged_dt, C, RHOC, GAM, CFL   # noqa: E402

MSTAR = 7.47e-5
CASES = (('R6', 'R6_rp10_comov_L4_fix,R6x_rp10_comov_L4_t300', (129.6, 144.8, 200.0, 300.0)),
         ('R7', 'R7_rp18_comov_L4_fix', (137.4, 300.0)))
THR = (1e-12, 1e-10, 1e-8)
RS = (3.0, 6.0)
VMAX = (0.9, 0.95)
RCAND = 10.0


def lam_hydro(alpha, beta, v, gii, v2, cs2):
    """max over directions of the GR sound-wave speeds; returns (|lam|, dir, alpha-term, shift)"""
    out, at = [], []
    for i in range(3):
        root = np.sqrt(np.maximum((1 - v2) * (gii[i] * (1 - v2 * cs2) - v[i]**2 * (1 - cs2)), 0))
        ap = alpha / (1 - v2 * cs2) * (v[i] * (1 - cs2) + np.sqrt(cs2) * root)
        am = alpha / (1 - v2 * cs2) * (v[i] * (1 - cs2) - np.sqrt(cs2) * root)
        lp, lm = ap - beta[i], am - beta[i]
        big = np.abs(lm) > lp
        out.append(np.where(big, np.abs(lm), lp)); at.append(np.where(big, am, ap))
    out = np.array(out); k = np.argmax(out, axis=0); n = np.arange(out.shape[1])
    return out[k, n], k, np.array(at)[k, n], np.array(beta)[k, n]


def load(fn):
    """baseline per-cell dt for all cells, plus full state for cells within RCAND of x_BH"""
    import re
    t = D.header(fn)
    par = t['par']
    x1min, x2min, x3min, x1max, x2max, x3max = t['size'][:6]
    nx, ng, nscal = 16, 4, 1
    no = nx + 2 * ng; nmhd = 5 + nscal; ncc = no**3; bsz = (no + 1) * no * no * 3
    mm = np.memmap(fn, dtype=np.float64, mode='r', offset=t['data_off'], shape=(t['nmb'], t['dsize'] // 8))
    lev = t['ll'][:, 3]
    w = np.array([x1max - x1min, x2max - x2min, x3max - x3min]); lo = np.array([x1min, x2min, x3min])
    s = slice(ng, ng + nx)
    xbh, xdot = D.frame_state(par, t['time'])
    fv = np.asarray(xdot)

    def get(sec, k, dflt):
        m = re.search(rf'<{sec}>(?:(?!<).)*?^\s*{k}\s*=\s*(\S+)', par, re.S | re.M)
        return float(m.group(1)) if m else dflt
    exc_lapse = get('coord', 'excise_lapse', 0.0)
    gate = get('mhd', 'atmosphere_frame_gate', 0.0)
    afv = re.search(r'^\s*atmosphere_frame_velocity\s*=\s*true', par, re.M) is not None
    root = t['root']
    base = {k: [] for k in ('dt', 'r', 'rho', 'lev', 'lam', 'at', 'bt')}
    cand = {k: [] for k in ('r', 'rho', 'alpha', 'b0', 'b1', 'b2', 'g00', 'g01', 'g02', 'g11', 'g12', 'g22',
                            'gi0', 'gi1', 'gi2', 'dx', 'dM', 'P0', 'P1', 'P2', 'KE', 'lev', 'exc', 'dt')}
    for b in range(t['nmb']):
        rec = mm[b]
        cons = rec[:nmhd * ncc].reshape(nmhd, no, no, no)[:, s, s, s]
        adm = rec[nmhd * ncc + bsz:].reshape(17, no, no, no)[:, s, s, s]
        ext = w / 2.0**lev[b]; dx = ext[0] / nx
        ll = t['ll'][b]; ic = np.arange(nx) + 0.5
        Z, Y, X = np.meshgrid(lo[2] + ll[2] * ext[2] + ic * dx, lo[1] + ll[1] * ext[1] + ic * dx,
                              lo[0] + ll[0] * ext[0] + ic * dx, indexing='ij')
        gxx, gxy, gxz, gyy, gyz, gzz = adm[:6]
        det = gxx * (gyy * gzz - gyz * gyz) - gxy * (gxy * gzz - gyz * gxz) + gxz * (gxy * gyz - gyy * gxz)
        sg = np.sqrt(det)
        ixx = (gyy * gzz - gyz * gyz) / det; ixy = (gxz * gyz - gxy * gzz) / det
        ixz = (gxy * gyz - gxz * gyy) / det; iyy = (gxx * gzz - gxz * gxz) / det
        iyz = (gxy * gxz - gxx * gyz) / det; izz = (gxx * gyy - gxy * gxy) / det
        Dd = cons[0] / sg
        Sx, Sy, Sz, tau = cons[1] / sg, cons[2] / sg, cons[3] / sg, cons[4] / sg
        ok = (Dd > 0) & np.isfinite(tau)
        Ssq = ixx * Sx * Sx + iyy * Sy * Sy + izz * Sz * Sz + 2 * (ixy * Sx * Sy + ixz * Sx * Sz + iyz * Sy * Sz)
        rho, P, W, h = D.c2p(Dd[ok], Ssq[ok], np.maximum(tau[ok], 0.0))
        g = [[ixx[ok], ixy[ok], ixz[ok]], [ixy[ok], iyy[ok], iyz[ok]], [ixz[ok], iyz[ok], izz[ok]]]
        uj = [Sx[ok] / (Dd[ok] * h), Sy[ok] / (Dd[ok] * h), Sz[ok] / (Dd[ok] * h)]
        v = [sum(g[i][j] * uj[j] for j in range(3)) / W for i in range(3)]
        alpha = adm[13][ok]; beta = [adm[14][ok], adm[15][ok], adm[16][ok]]
        xr, yr, zr = X[ok] - xbh[0], Y[ok] - xbh[1], Z[ok] - xbh[2]
        r = np.sqrt(xr * xr + yr * yr + zr * zr)
        floor = rho < 1e-23                      # C2P-floored proxy: within 10x dfloor (1e-24)
        if afv and gate > 0:
            sub = floor & (r > gate)
            v = [np.where(sub, fv[i], v[i]) for i in range(3)]
        gd = [[gxx[ok], gxy[ok], gxz[ok]], [gxy[ok], gyy[ok], gyz[ok]], [gxz[ok], gyz[ok], gzz[ok]]]
        v2 = np.minimum(sum(gd[i][j] * v[i] * v[j] for i in range(3) for j in range(3)), 0.99)
        cs2 = np.clip(GAM * P / (rho * h), 0, 1)
        lam, _, at, bt = lam_hydro(alpha, beta, v, [g[0][0], g[1][1], g[2][2]], v2, cs2)
        exc = alpha < exc_lapse
        dtc = np.where(exc, np.inf, CFL * dx / lam)
        base['dt'].append(dtc); base['r'].append(r); base['rho'].append(rho)
        base['lev'].append(np.full(r.size, lev[b] - root)); base['lam'].append(lam)
        base['at'].append(at); base['bt'].append(bt)
        c = r < RCAND
        if c.any():
            for k, val in (('r', r), ('rho', rho), ('alpha', alpha), ('b0', beta[0]), ('b1', beta[1]),
                           ('b2', beta[2]), ('g00', gd[0][0]), ('g01', gd[0][1]), ('g02', gd[0][2]),
                           ('g11', gd[1][1]), ('g12', gd[1][2]), ('g22', gd[2][2]), ('gi0', g[0][0]),
                           ('gi1', g[1][1]), ('gi2', g[2][2]), ('dt', dtc), ('exc', exc)):
                cand[k].append(val[c])
            cand['dx'].append(np.full(c.sum(), dx))
            cand['lev'].append(np.full(c.sum(), lev[b] - root))
            cand['dM'].append((cons[0][ok] * dx**3)[c])
            for i, Sk in enumerate((cons[1], cons[2], cons[3])):
                cand[f'P{i}'].append((Sk[ok] * dx**3)[c])
            cand['KE'].append(((W - 1) * cons[0][ok] * dx**3)[c])
    B = {k: np.concatenate(v) for k, v in base.items()}
    Cc = {k: np.concatenate(v) for k, v in cand.items()}
    return t['time'], B, Cc


def whatif(B, Cc, thr, rs, vmax):
    zone = (Cc['r'] < rs) & (Cc['rho'] < thr * RHOC) & ~Cc['exc']
    prot = (Cc['r'] < rs) & (Cc['rho'] >= thr * RHOC)
    assert not (zone & prot).any()
    z = {k: v[zone] for k, v in Cc.items()}
    beta = [z['b0'], z['b1'], z['b2']]
    gd = [[z['g00'], z['g01'], z['g02']], [z['g01'], z['g11'], z['g12']], [z['g02'], z['g12'], z['g22']]]
    bnorm = np.sqrt(np.maximum(sum(gd[i][j] * beta[i] * beta[j] for i in range(3) for j in range(3)), 0))
    vmag = bnorm / z['alpha']
    f = np.minimum(1.0, vmax / np.maximum(vmag, 1e-300))
    v = [f * beta[i] / z['alpha'] for i in range(3)]
    v2 = np.minimum((f * vmag)**2, 0.99)
    lam, k, at, bt = lam_hydro(z['alpha'], beta, v, [z['gi0'], z['gi1'], z['gi2']], v2, np.zeros_like(v2))
    dtz = CFL * z['dx'] / np.maximum(lam, 1e-300)
    # cells outside the zone keep their baseline dt; the zone is a subset of the candidates
    # (r < RCAND), so the outside minimum = min(non-candidate baseline, non-zone candidates)
    outside_c = Cc['dt'][~zone]
    m_out_c = outside_c.min() if outside_c.size else np.inf
    nc = B['r'] >= RCAND
    i_nc = np.argmin(np.where(nc, B['dt'], np.inf))
    if zone.any() and dtz.min() < min(m_out_c, B['dt'][i_nc]):
        j = np.argmin(dtz)
        lim = dict(where='zone (capped)' if f[j] < 1 else 'zone', r=z['r'][j], lev=z['lev'][j], lam=lam[j],
                   at=at[j], bt=bt[j], rho=z['rho'][j] / RHOC, dt=dtz[j], vb=vmag[j])
    elif m_out_c < B['dt'][i_nc]:
        jj = np.nonzero(~zone)[0][np.argmin(outside_c)]
        lim = dict(where='near BH, outside zone', r=Cc['r'][jj], lev=Cc['lev'][jj], lam=np.nan, at=np.nan,
                   bt=np.nan, rho=Cc['rho'][jj] / RHOC, dt=Cc['dt'][jj], vb=np.nan)
        i = np.nonzero((B['r'] == Cc['r'][jj]) & (B['dt'] == Cc['dt'][jj]))[0]
        if i.size:
            lim.update(lam=B['lam'][i[0]], at=B['at'][i[0]], bt=B['bt'][i[0]])
    else:
        lim = dict(where=f'r_BH >= {RCAND:g}', r=B['r'][i_nc], lev=B['lev'][i_nc], lam=B['lam'][i_nc],
                   at=B['at'][i_nc], bt=B['bt'][i_nc], rho=B['rho'][i_nc] / RHOC, dt=B['dt'][i_nc], vb=np.nan)
    P = np.sqrt(sum(z[f'P{i}'].sum()**2 for i in range(3)))
    capped = (f < 1).sum()
    return dict(n=int(zone.sum()), ncap=int(capped), M=z['dM'].sum() / MSTAR, P=P / MSTAR, KE=z['KE'].sum() / MSTAR,
                zmaxrho=(z['rho'].max() / RHOC if zone.any() else np.nan),
                protM=Cc['dM'][prot].sum() / MSTAR, protmax=(Cc['rho'][prot].max() / RHOC if prot.any() else np.nan),
                lim=lim, dtzmin=(dtz.min() if zone.any() else np.inf))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{C}/plots/phase2_m1')
    ap.add_argument('--only', default='')
    ap.add_argument('--rs-scan', action='store_true',
                    help='extra: r_s in {3, 4, 5, 6, 8, 10} at rho_thr = 1e-10 rho_c, v_max = 0.9')
    a = ap.parse_args()
    rows = ['R-054 S0: grid-rest zone what-if. dt0 = baseline offline dt (code dt in brackets); per '
            '(rho_thr/rho_c, r_s, v_max): dt, gain, limiting cell, zone cells (capped), zone mass / '
            '|momentum| / kinetic energy [M*], max rho in zone, gas the gate protects within r_s.']
    for lab, d, ts in CASES:
        if a.only and lab not in a.only.split(','):
            continue
        for tt in ts:
            fn = rst_at(d, tt)
            t, B, Cc = load(fn)
            dt0 = B['dt'].min()
            rows.append(f'== {lab} t = {t:.1f}: dt0 {dt0:.4f} (code {logged_dt(d, tt):.4f}); '
                        f'candidate cells r_BH < {RCAND:g}: {Cc["r"].size}')
            print(rows[-1], flush=True)
            grid = ([(1e-10, r, 0.9) for r in (3.0, 4.0, 5.0, 6.0, 8.0, 10.0)] if a.rs_scan else
                    [(thr, rs, vm) for thr in THR for rs in RS for vm in VMAX])
            for thr, rs, vm in grid:
                w = whatif(B, Cc, thr, rs, vm)
                L = w['lim']
                rows.append(
                    f'  thr {thr:.0e} r_s {rs:g} vmax {vm:.2f}: dt {L["dt"]:.4f} (x{L["dt"] / dt0:.2f}) '
                    f'lim [{L["where"]}, L{L["lev"]}, r_BH {L["r"]:.2f}, rho {L["rho"]:.1e} rho_c, '
                    f'|lam| {L["lam"]:.3f} = |a-term {L["at"]:+.3f} - shift {L["bt"]:+.3f}|'
                    + (f', |beta|/alpha {L["vb"]:.2f}' if np.isfinite(L['vb']) else '') + '] '
                    f'zone {w["n"]} cells ({w["ncap"]} capped), M {w["M"]:.1e} P {w["P"]:.1e} '
                    f'KE {w["KE"]:.1e}, max rho {w["zmaxrho"]:.1e}; protected M {w["protM"]:.1e} '
                    f'(max rho {w["protmax"]:.1e}); zone-only min dt {w["dtzmin"]:.3f}')
                print(rows[-1], flush=True)
    if not a.only:
        with open(f'{a.out}/c046_s0' + ('_rs' if a.rs_scan else '') + '.txt', 'w') as f:
            f.write('\n'.join(rows) + '\n')


if __name__ == '__main__':
    main()
