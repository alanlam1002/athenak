#!/usr/bin/env python3
"""C-045 (R-051 dt attribution): which cells set the CFL timestep, comoving vs BH frame.

Per restart, every cell's hydro CFL limit is rebuilt offline as in dyn_grmhd_newdt.cpp
(gr_dt = true; B = 0): for each direction i the GR sound-wave speeds

    lam_+- = alpha/(1 - v^2 cs^2) [v^i (1 - cs^2) +- cs sqrt((1 - v^2)(g^ii (1 - v^2 cs^2)
             - v^i v^i (1 - cs^2)))] - beta^i,      cs^2 = Gamma P/(rho h),

dt_cell = cfl min_i dx/max|lam|. The stored shift beta is the code's (comoving runs: it
carries the frame velocity). Excised cells (alpha < excise_lapse, dt_skip_excised) are
dropped. The <mhd> atmosphere_frame_velocity substitution (floored cells farther than
atmosphere_frame_gate from the BH get v = frame velocity) is applied to cells at the
density floor. Cells are binned by role: near-BH (r_BH < 6), remnant (remnant.py's
self-bound set), debris (rho > 1e-10 rho_c) and atmosphere; per role the minimum dt and
the speed budget (alpha-term vs shift) of the limiting cell are printed.

    c045_dt.py [--out DIR]
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tde_dmde as D                              # noqa: E402
import remnant as RM                              # noqa: E402
import rst_tree as RT                             # noqa: E402

C = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc'
RHOC, GAM, CFL = 3.24e-4, 5.0 / 3.0, 0.4
CASES = (('R6', 'R6_rp10_comov_L4_fix,R6x_rp10_comov_L4_t300', (24.4, 129.6, 144.8, 200.0, 300.0)),
         ('Q1c', 'D_e_Q1deck_kfix_t80,RQ1_bh_L4_t300_from_De', (24.9, 123.8, 142.6, 195.2, 300.0)),
         ('R7', 'R7_rp18_comov_L4_fix', (25.0, 137.4, 193.4, 300.0)),
         ('Q3', 'Q3_P2_rp18_L4', (23.5, 133.9, 207.7, 300.0)))


def rst_at(d, t):
    fs = sorted(f for s in d.split(',') for f in glob.glob(f'{C}/{s}/rst/*.rst'))
    ts = np.array([RT.read_tree(f)['time'] for f in fs])
    i = int(np.argmin(np.abs(ts - t)))
    return fs[i] if abs(ts[i] - t) < 0.5 else None


def logged_dt(d, t):
    """median code dt over the 20 cycles after t (from ensemble.out)"""
    rows = []
    for s in d.split(','):
        for ln in open(f'{C}/{s}/ensemble.out'):
            m = re.match(r'elapsed=\S+ cycle=\d+ time=(\S+) dt=(\S+)', ln)
            if m:
                rows.append((float(m.group(1)), float(m.group(2))))
    a = np.array(rows)
    k = (a[:, 0] > t + 1e-6) & (a[:, 0] < t + 1.0)
    return np.median(a[k, 1]) if k.any() else np.nan


def speeds(fn, nscal=1):
    """per cell (same order/mask as tde_dmde.cells(rcut=0)): dt_cell, alpha-term, shift of the
    limiting direction, alpha, and the floored flag"""
    t = D.header(fn)
    par = t['par']
    nmb = t['nmb']
    x1min, x2min, x3min, x1max, x2max, x3max = t['size'][:6]
    nx, ng = 16, 4
    no = nx + 2 * ng
    nmhd = 5 + nscal
    ncc = no**3
    bsz = (no + 1) * no * no * 3
    mm = np.memmap(fn, dtype=np.float64, mode='r', offset=t['data_off'], shape=(nmb, t['dsize'] // 8))
    lev = t['ll'][:, 3]
    w = np.array([x1max - x1min, x2max - x2min, x3max - x3min])
    s = slice(ng, ng + nx)
    xbh, xdot = D.frame_state(par, t['time'])
    fv = np.asarray(xdot)          # the frame velocity the floor gets beyond the gate
    get = lambda sec, k, dflt: float(re.search(rf'<{sec}>(?:(?!<).)*?^\s*{k}\s*=\s*(\S+)', par, re.S | re.M).group(1)) \
        if re.search(rf'<{sec}>(?:(?!<).)*?^\s*{k}\s*=', par, re.S | re.M) else dflt
    exc_lapse = get('coord', 'excise_lapse', 0.0)
    gate = get('mhd', 'atmosphere_frame_gate', 0.0)
    afv = re.search(r'^\s*atmosphere_frame_velocity\s*=\s*true', par, re.M) is not None
    out = {k: [] for k in ('dt', 'aterm', 'beta', 'alpha', 'floor', 'dir')}
    for b in range(nmb):
        rec = mm[b]
        cons = rec[:nmhd * ncc].reshape(nmhd, no, no, no)[:, s, s, s]
        adm = rec[nmhd * ncc + bsz:].reshape(17, no, no, no)[:, s, s, s]
        dx = (w / 2.0**lev[b] / nx)[0]
        gxx, gxy, gxz, gyy, gyz, gzz = adm[:6]
        det = (gxx * (gyy * gzz - gyz * gyz) - gxy * (gxy * gzz - gyz * gxz) + gxz * (gxy * gyz - gyy * gxz))
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
        alpha = adm[13][ok]
        beta = [adm[14][ok], adm[15][ok], adm[16][ok]]
        floor = rho < 1e-12 * RHOC
        if afv and gate > 0:
            nz, ny, nxx = np.nonzero(ok)
            ll = t['ll'][b]; ext = w / 2.0**lev[b]
            X = x1min + ll[0] * ext[0] + (nxx + .5) * dx - xbh[0]
            Y = x2min + ll[1] * ext[1] + (ny + .5) * dx - xbh[1]
            Z = x3min + ll[2] * ext[2] + (nz + .5) * dx - xbh[2]
            sub = floor & (X * X + Y * Y + Z * Z > gate * gate)
            for i in range(3):
                v[i] = np.where(sub, fv[i], v[i])
        gd = [[gxx[ok], gxy[ok], gxz[ok]], [gxy[ok], gyy[ok], gyz[ok]], [gxz[ok], gyz[ok], gzz[ok]]]
        v2 = sum(gd[i][j] * v[i] * v[j] for i in range(3) for j in range(3))
        v2 = np.minimum(v2, 0.99)
        cs2 = np.clip(GAM * P / (rho * h), 0, 1)
        lam, at, bt = [], [], []
        for i in range(3):
            gii = g[i][i]
            root = np.sqrt(np.maximum((1 - v2) * (gii * (1 - v2 * cs2) - v[i]**2 * (1 - cs2)), 0))
            ap = alpha / (1 - v2 * cs2) * (v[i] * (1 - cs2) + np.sqrt(cs2) * root)
            am = alpha / (1 - v2 * cs2) * (v[i] * (1 - cs2) - np.sqrt(cs2) * root)
            lp, lm = ap - beta[i], am - beta[i]
            big = np.abs(lm) > lp
            lam.append(np.where(big, np.abs(lm), lp)); at.append(np.where(big, am, ap)); bt.append(beta[i])
        lam = np.array(lam); k = np.argmax(lam, axis=0); idx = np.arange(lam.shape[1])
        exc = alpha < exc_lapse
        dtc = np.where(exc, np.inf, CFL * dx / lam[k, idx])
        out['dt'].append(dtc); out['aterm'].append(np.array(at)[k, idx]); out['beta'].append(np.array(bt)[k, idx])
        out['alpha'].append(alpha); out['floor'].append(floor); out['dir'].append(k)
    return {k: np.concatenate(v) for k, v in out.items()}


def one(lab, d, tt):
    fn = rst_at(d, tt)
    c = D.cells(fn, rcut=0.0)
    sp = speeds(fn)
    xb = np.asarray(c['xbh'])
    rbh = np.sqrt((c['x'] - xb[0])**2 + (c['y'] - xb[1])**2 + (c['z'] - xb[2])**2)
    far = rbh >= 3.0
    # remnant.py works on the r_BH >= 3 subset
    sub = {k: (v[far] if isinstance(v, np.ndarray) and v.shape == far.shape else v) for k, v in c.items()}
    rem = np.zeros(far.size, bool); rem[np.nonzero(far)[0][RM.bound_mass(sub)[3]]] = True
    role = np.where(rbh < 6, 'nearBH', np.where(rem, 'remnant', np.where(c['rho'] > 1e-10 * RHOC, 'debris', 'atmos')))
    dt = sp['dt']
    rows = [f'{lab} t = {c["t"]:6.1f}: offline min dt {dt.min():.4f} (code dt after this restart {logged_dt(d, tt):.4f})']
    for r in ('nearBH', 'remnant', 'debris', 'atmos'):
        m = role == r
        if not m.any():
            continue
        i = np.nonzero(m)[0][np.argmin(dt[m])]
        rows.append(f'    {r:8s} min dt {dt[i]:.4f} (L{c["lev"][i]}, r_BH {rbh[i]:5.2f}, dir {"xyz"[sp["dir"][i]]}): '
                    f'alpha {sp["alpha"][i]:.3f}, alpha-term {sp["aterm"][i]:+.3f}, shift {sp["beta"][i]:+.3f}, '
                    f'|v| {np.sqrt(c["vx"][i]**2 + c["vy"][i]**2 + c["vz"][i]**2):.3f}, floor {bool(sp["floor"][i])}')
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{C}/plots/phase2_m1')
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    rows = []
    for lab, d, ts in CASES:
        if a.only and lab not in a.only.split(','):
            continue
        for tt in ts:
            r = one(lab, d, tt)
            print('\n'.join(r), flush=True)
            rows += r
    if not a.only:
        with open(f'{a.out}/c045_dt.txt', 'w') as f:
            f.write('\n'.join(rows) + '\n')


if __name__ == '__main__':
    main()
