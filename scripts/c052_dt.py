#!/usr/bin/env python3
"""C-052 (R-066 item 1): offline dt checks on the S2 runs (no compute).

(a) dt dips vs regrid cycles. A regrid is identified per cycle from the log: CFC prints
    "CFC::ReinitializeMetricForAMR: post-regrid ..." once per regrid, just before the
    "elapsed= cycle=N time= dt=" line whose dt is the first one computed on the new mesh
    (driver.cpp: AdaptiveMeshRefinement, then Mesh::NewTimeStep). Dips = dt < 0.6x the
    window median (C-051 windows). Per window: dips, dips on regrid cycles, regrid cycles,
    wall share of the dip steps, and the cycles / wall saved if the dip steps had the window
    median dt (wall saved = cycles saved x the median non-regrid step wall).
(b) At the final restarts (and R6 T's t = 136.67), re-evaluate the CFL with the grid-rest state
    (v = beta'/alpha capped at |v| <= 0.9 along beta', cold) in the cells that carry the
    excision reset state (v = 0 exactly, rho = dexcise) but lie outside the current lapse mask
    alpha < excise_lapse (= the flux mask the CFL skips). Baseline as c046_s0.load (zone cells
    already carry the zone state in T's restarts).
(c) R6 T: code dt on the non-regrid cycles bracketing the t = 136.67 restart.

    c052_dt.py [--out DIR]
"""
import argparse
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tde_dmde as D                              # noqa: E402
import c046_s0 as S                               # noqa: E402
from c045_dt import C, RHOC, GAM, CFL             # noqa: E402

WIN = {'R6': ((113.7, 136.0), (136.0, 150.0), (150.0, 170.0)), 'R7': ((253.7, 275.0), (275.0, 300.0))}
VMAX = 0.9
RCAND = 10.0


def cycles(run):
    """per printed cycle: elapsed, cycle, time, dt, regrid flag (regrid right before this dt)"""
    rows, rg = [], False
    for ln in open(f'{C}/{run}/ensemble.out'):
        if ln.startswith('CFC::ReinitializeMetricForAMR'):
            rg = True
        m = re.match(r'elapsed=(\S+) cycle=(\d+) time=(\S+) dt=(\S+)', ln)
        if m:
            rows.append((float(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4)), rg))
            rg = False
    a = np.array(rows, dtype=float)
    # the step taken with dt[i] costs elapsed[i+1] - elapsed[i]
    wall = np.append(np.diff(a[:, 0]), np.nan)
    return dict(t=a[:, 2], dt=a[:, 3], rg=a[:, 4] > 0, wall=wall, cyc=a[:, 1].astype(int))


def part_a(rows):
    rows.append('(a) dt dips (< 0.6x window median) vs regrid cycles; measured from ensemble.out')
    for lab in ('R6', 'R7'):
        for arm in ('T', 'A', 'B'):
            L = cycles(f'S2{arm}_{lab}')
            last = L['t'] + L['dt'] > max(w[1] for w in WIN[lab]) - 1e-9   # final step clipped to tlim
            for w0, w1 in WIN[lab]:
                k = (L['t'] >= w0) & (L['t'] < w1) & ~last & np.isfinite(L['wall'])
                med = np.median(L['dt'][k])
                dip = k & (L['dt'] < 0.6 * med)
                rgk = k & L['rg']
                on_rg = dip & (L['rg'] | np.roll(L['rg'], 1))
                saved = (1.0 - L['dt'][dip] / med).sum()             # cycles saved at the median dt
                wmed = np.median(L['wall'][k & ~L['rg'] & ~dip])
                rows.append(f'  {lab} {arm} t {w0:5.1f}-{w1:5.1f}: {k.sum():4d} cycles, median dt {med:.4f}; '
                            f'regrid cycles {rgk.sum():3d} (median dt/med {np.median(L["dt"][rgk]) / med if rgk.any() else np.nan:.2f}); '
                            f'dips {dip.sum():3d}, on regrid (or next) {on_rg.sum():3d}; dip-step wall share '
                            f'{np.nansum(L["wall"][dip]) / np.nansum(L["wall"][k]):.3f} (regrid-cycle wall share '
                            f'{np.nansum(L["wall"][rgk]) / np.nansum(L["wall"][k]):.3f}); sim-time in dips '
                            f'{L["dt"][dip].sum() / L["dt"][k].sum():.3f}; cycles saved at median {saved:.1f} '
                            f'= x{k.sum() / (k.sum() - saved):.3f} in cycles, wall saved {saved * wmed:.0f} s of '
                            f'{np.nansum(L["wall"][k]):.0f} s')
                print(rows[-1], flush=True)


def part_c(rows):
    L = cycles('S2T_R6')
    rows.append('(c) R6 T: code dt around the t = 136.67 restart (offline replica 0.067 in C-051 §4); measured')
    i0 = int(np.argmin(np.abs(L['t'] - 136.675)))
    for i in range(max(i0 - 8, 0), min(i0 + 9, L['t'].size)):
        rows.append(f'  cycle {L["cyc"][i]} t {L["t"][i]:.3f} dt {L["dt"][i]:.4f}' + ('  <- regrid' if L['rg'][i] else ''))
    k = (np.abs(L['t'] - 136.67) < 2.0) & ~L['rg']
    rows.append(f'  non-regrid cycles |t - 136.67| < 2: n {k.sum()}, dt median {np.median(L["dt"][k]):.4f}, '
                f'min {L["dt"][k].min():.4f}, max {L["dt"][k].max():.4f}')
    print('\n'.join(rows[-(L['t'][max(i0 - 8, 0):i0 + 9].size + 2):]), flush=True)


def load(fn):
    """c046_s0.load plus |v|, T-proxy and the excision-state flag for the candidate cells"""
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
    dexc = get('coord', 'dexcise', 0.0)
    gate = get('mhd', 'atmosphere_frame_gate', 0.0)
    afv = re.search(r'^\s*atmosphere_frame_velocity\s*=\s*true', par, re.M) is not None
    root = t['root']
    base = {k: [] for k in ('dt', 'r', 'rho', 'lev', 'lam', 'at', 'bt')}
    keys = ('r', 'rho', 'alpha', 'b0', 'b1', 'b2', 'g00', 'g01', 'g02', 'g11', 'g12', 'g22', 'gi0', 'gi1', 'gi2',
            'dx', 'lev', 'exc', 'dt', 'vabs', 'lam', 'at', 'bt', 'vraw', 'dM')
    cand = {k: [] for k in keys}
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
        gd = [[gxx[ok], gxy[ok], gxz[ok]], [gxy[ok], gyy[ok], gyz[ok]], [gxz[ok], gyz[ok], gzz[ok]]]
        vraw = np.sqrt(np.maximum(sum(gd[i][j] * v[i] * v[j] for i in range(3) for j in range(3)), 0))
        floor = rho < 1e-23
        if afv and gate > 0:
            sub = floor & (r > gate)
            v = [np.where(sub, fv[i], v[i]) for i in range(3)]
        v2 = np.minimum(sum(gd[i][j] * v[i] * v[j] for i in range(3) for j in range(3)), 0.99)
        cs2 = np.clip(GAM * P / (rho * h), 0, 1)
        lam, _, at, bt = S.lam_hydro(alpha, beta, v, [g[0][0], g[1][1], g[2][2]], v2, cs2)
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
                           ('gi1', g[1][1]), ('gi2', g[2][2]), ('dt', dtc), ('exc', exc), ('vabs', np.sqrt(v2)),
                           ('lam', lam), ('at', at), ('bt', bt), ('vraw', vraw)):
                cand[k].append(val[c])
            cand['dx'].append(np.full(c.sum(), dx)); cand['lev'].append(np.full(c.sum(), lev[b] - root))
            cand['dM'].append((cons[0][ok] * dx**3)[c])
    B = {k: np.concatenate(v) for k, v in base.items()}
    Cc = {k: np.concatenate(v) for k, v in cand.items()}
    return t['time'], B, Cc, dict(exc_lapse=exc_lapse, dexc=dexc)


def gridrest_dt(Cc, mod):
    z = {k: v[mod] for k, v in Cc.items()}
    beta = [z['b0'], z['b1'], z['b2']]
    gd = [[z['g00'], z['g01'], z['g02']], [z['g01'], z['g11'], z['g12']], [z['g02'], z['g12'], z['g22']]]
    bn = np.sqrt(np.maximum(sum(gd[i][j] * beta[i] * beta[j] for i in range(3) for j in range(3)), 0))
    vb = bn / z['alpha']
    f = np.minimum(1.0, VMAX / np.maximum(vb, 1e-300))
    vv = [f * beta[i] / z['alpha'] for i in range(3)]
    lam, _, at, bt = S.lam_hydro(z['alpha'], beta, vv, [z['gi0'], z['gi1'], z['gi2']], np.minimum((f * vb)**2, 0.99),
                                 np.zeros_like(vb))
    return CFL * z['dx'] / np.maximum(lam, 1e-300), lam, at, bt, f


def describe(Cc, B, dtc, extra=None):
    """global limiter from candidate dt array dtc (r < RCAND) and the baseline r >= RCAND"""
    nc = np.where(B['r'] >= RCAND, B['dt'], np.inf)
    i = int(np.argmin(nc)); j = int(np.argmin(dtc))
    if dtc[j] < nc[i]:
        lam, at, bt = (extra[j] if extra is not None and j in extra else (Cc['lam'][j], Cc['at'][j], Cc['bt'][j]))
        return (dtc[j], f'r_BH {Cc["r"][j]:.2f} L{Cc["lev"][j]} alpha {Cc["alpha"][j]:.3f} rho/rho_c '
                f'{Cc["rho"][j] / RHOC:.1e} |v| {Cc["vraw"][j]:.3f} |lam| {lam:.3f} = |a-term {at:+.3f} - shift {bt:+.3f}|')
    return (nc[i], f'r_BH {B["r"][i]:.2f} L{B["lev"][i]} rho/rho_c {B["rho"][i] / RHOC:.1e} |lam| {B["lam"][i]:.3f} '
            f'= |a-term {B["at"][i]:+.3f} - shift {B["bt"][i]:+.3f}|')


def part_b(rows):
    rows.append('(b) CFL with grid-rest in the excision-state cells outside the lapse (flux) mask; offline')
    from c045_dt import rst_at
    for lab, run, tt in (('R7 T', 'S2T_R7', 300.0), ('R7 A', 'S2A_R7', 300.0), ('R6 T', 'S2T_R6', 170.0),
                         ('R6 T', 'S2T_R6', 136.67), ('R6 A', 'S2A_R6', 170.0)):
        fn = rst_at(run, tt)
        t, B, Cc, p = load(fn)
        excst = (Cc['vraw'] == 0.0) & ~Cc['exc']        # excision reset state outside the mask
        v0 = (Cc['vraw'] < 1e-12) & ~Cc['exc']
        dt0, s0 = describe(Cc, B, Cc['dt'])
        rows.append(f'== {lab} t = {t:.2f} ({os.path.basename(fn)}): in mask (alpha < {p["exc_lapse"]:g}) '
                    f'{int(Cc["exc"].sum())} cells, r_BH max {Cc["r"][Cc["exc"]].max():.2f}; outside the mask with v = 0 '
                    f'exactly {int(excst.sum())} (|v| < 1e-12: {int(v0.sum())}), r_BH {Cc["r"][excst].min() if excst.any() else np.nan:.2f}-'
                    f'{Cc["r"][excst].max() if excst.any() else np.nan:.2f}, alpha {Cc["alpha"][excst].min() if excst.any() else np.nan:.3f}-'
                    f'{Cc["alpha"][excst].max() if excst.any() else np.nan:.3f}, rho/rho_c max '
                    f'{Cc["rho"][excst].max() / RHOC if excst.any() else np.nan:.1e}, mass {Cc["dM"][excst].sum():.2e}')
        rows.append(f'   baseline dt {dt0:.4f}: {s0}')
        if excst.any():
            dtz, lam, at, bt, f = gridrest_dt(Cc, excst)
            dtc = Cc['dt'].copy(); dtc[excst] = dtz
            idx = np.nonzero(excst)[0]
            extra = {int(idx[q]): (lam[q], at[q], bt[q]) for q in range(idx.size)}
            dt1, s1 = describe(Cc, B, dtc, extra)
            rows.append(f'   grid-rest there: dt {dt1:.4f} (x{dt1 / dt0:.2f}); {int((f < 1).sum())} of them capped; limiter {s1}')
            # what the next limiters are (ladder): the 5 smallest cell dt outside r_BH >= RCAND
            o = np.argsort(dtc)[:5]
            rows.append('   next candidates: ' + '; '.join(f'{dtc[q]:.4f} r {Cc["r"][q]:.2f} a {Cc["alpha"][q]:.2f} |v| {Cc["vraw"][q]:.2f}'
                                                         f'{" exc-state" if excst[q] else ""}' for q in o))
        o = np.argsort(Cc['dt'])[:5]
        rows.append('   baseline 5 smallest (r < 10): ' + '; '.join(f'{Cc["dt"][q]:.4f} r {Cc["r"][q]:.2f} a {Cc["alpha"][q]:.2f} '
                                                                    f'|v| {Cc["vraw"][q]:.2g}{" exc-state" if excst[q] else ""}' for q in o))
        print('\n'.join(rows[-5 if excst.any() else -3:]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{C}/plots/phase2_m1')
    ap.add_argument('--skip-b', action='store_true')
    a = ap.parse_args()
    rows = ['C-052 / R-066 item 1: offline dt checks on the S2 runs (c052_dt.py).']
    part_a(rows)
    part_c(rows)
    if not a.skip_b:
        part_b(rows)
    with open(f'{a.out}/c052_dt.txt', 'w') as f:
        f.write('\n'.join(rows) + '\n')


if __name__ == '__main__':
    main()
