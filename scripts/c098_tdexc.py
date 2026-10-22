#!/usr/bin/env python3
"""C-098 (R-107 item 1 = R-106 test (a)): offline what-if of a time-dependent ('flowing') excision radius.

R-106's rule, applied per dump / restart:
    r_f = max(r_lapse, min(r_max, d_min - m dx - v_in dt_upd)),   m = 4 cells, v_in = 0.6,
with d_min = distance from x_BH to the nearest cell with rho > rho_thr (1e-10 rho_c, the zone threshold),
dx = the cell size at that distance's location near the hole (the BH box, L4: 1/16), dt_upd = the cycle dt
(mhd.hst), r_lapse = the lapse excision radius (alpha0 = 0.25 at r_iso 0.592 on the true trumpet).
Also reported: d_min to the nearest 'debris-band' cell, taken as rho > 1e-16 rho_c (the zone's tier-3
threshold, <mhd> atmosphere_zone_tier3_rho default) -- a labelled choice, not a code quantity.

Part A, per 2D dump (prim_xy, the z = 0 orbital plane; x_BH(t) = puncture_x - frame_disp from user.hst,
interpolated): d_min, r_f for r_max in {2.5, 2.92, 3.5}, the guard margin d_min - r_f, and the shrink rate
-dr_f/dt between dumps against one cell per cycle (dx / dt_cycle). ARGUED: the in-plane d_min equals the 3D
one to within the star's/debris' vertical offset (part B checks it on the restarts).
Part B, per 3D restart: the 3D d_min, r_f, and c096_dtwhatif's dt with cells r < r_f excised and skipped in
the CFL (base deck otherwise: excise_lapse 0.25, v_max 0.9, zone r_s 6, rho_thr 1e-10 rho_c): dt, the gain
over base, the limiting cell (r_iso, rho, |lambda|, the term, zone-capped or not, |beta'|_phys/alpha and
|beta'|_coord), and the rest mass in r_lapse < r < r_f (what growing to r_f would absorb). Restarts written
by binaries before 30c52c1d had the pinned background inside r_iso < 1.5 only; for r_f >= 1.5 that region is
excised and the what-if is valid (the exterior was the exact trumpet to 1e-6, C-094); rows with r_f < 1.5
on such restarts are marked '(pinned bg)'.

    c098_tdexc.py LABEL:RUNDIR[,RUNDIR2...]:TP [...] [--every 1] [--no-restarts] [--nproc N] [--out FILE]
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
import tde_dmde as D                               # noqa: E402
import c046_s0 as S                                # noqa: E402
import c096_dtwhatif as W                          # noqa: E402
from c045_dt import RHOC, CFL                      # noqa: E402
from tde_ensemble_plots import bc                  # noqa: E402

BASE = 'cfc_tde_wd_imbh_parabolic_tracker'
RLAPSE, M_CELLS, VIN = 0.5917, 4, 0.6
RMAX = (2.5, 2.92, 3.5)
THR, BAND = 1e-10, 1e-16
FIXED = '30c52c1d'


def par(d):
    return open(f'{d}/parfile.par').read() if os.path.exists(f'{d}/parfile.par') else ''


def hst(d):
    u = np.loadtxt(glob.glob(f'{d}/{BASE}.user.hst')[0])
    m = np.loadtxt(glob.glob(f'{d}/{BASE}.mhd.hst')[0])
    _, i = np.unique(u[:, 0], return_index=True); u = u[i]
    _, i = np.unique(m[:, 0], return_index=True); m = m[i]
    return u, m


def rf(dmin, dx, dt):
    return {rm: max(RLAPSE, min(rm, dmin - M_CELLS * dx - VIN * dt)) for rm in RMAX}


def slice_series(dirs, every):
    rows = []
    for d in dirs:
        txt = par(d)
        if 'puncture_x1' not in txt:                     # run dirs started with -i elsewhere: use a restart's copy
            rs = sorted(glob.glob(f'{d}/rst/*.rst'))
            txt = D.header(rs[0])['par'] if rs else txt
        px = [S_par(txt, f'puncture_x{a}') for a in (1, 2, 3)]
        u, m = hst(d)
        for fn in sorted(glob.glob(f'{d}/bin/{BASE}.prim_xy.*.bin'))[::every]:
            q = bc.read_binary(fn)
            t = q['time']
            disp = [np.interp(t, u[:, 0], u[:, 7 + a]) for a in range(3)]
            xb = np.array([px[a] - disp[a] for a in range(3)])
            dtc = np.interp(t, m[:, 0], m[:, 1])
            g = np.asarray(q['mb_geometry']); nx = q['nx1_out_mb']
            rho = np.asarray(q['mb_data']['dens'])[:, 0]
            dx = ((g[:, 1] - g[:, 0]) / nx)[:, None, None] * np.ones_like(rho)
            X = g[:, 0, None, None] + (np.arange(nx)[None, None, :] + .5) * (g[:, 1] - g[:, 0])[:, None, None] / nx
            Y = g[:, 2, None, None] + (np.arange(nx)[None, :, None] + .5) * (g[:, 3] - g[:, 2])[:, None, None] / nx
            r = np.hypot(np.broadcast_to(X, rho.shape) - xb[0], np.broadcast_to(Y, rho.shape) - xb[1])
            near = r < 4.0
            dxh = dx[near].min() if near.any() else 0.0625
            s1, s2 = rho > THR * RHOC, rho > BAND * RHOC
            d1 = r[s1].min() if s1.any() else np.inf
            d2 = r[s2].min() if s2.any() else np.inf
            rows.append((t, d1, d2, dxh, dtc))
    rows.sort()
    return np.array(rows)


def S_par(txt, key):
    m = re.search(rf'^\s*{key}\s*=\s*([-+.\deE]+)', txt, re.M)
    return float(m.group(1)) if m else 0.0


def rst_job(fn):
    t, B, Cc = S.load(fn)
    h = D.header(fn)
    rho_all, r_all = B['rho'], B['r']
    d1 = r_all[rho_all > THR * RHOC].min()
    d2 = r_all[rho_all > BAND * RHOC].min()
    m = re.search(r'^\s*dual_energy_div', h['par'], re.M)
    w0 = W.variant(B, Cc, 0.25, 0.90, 6.0, THR)
    base = min(w0['dt'].min(), w0['dfar'])
    out = dict(t=t, d1=d1, d2=d2, base=base, rows={})
    bcoord = np.sqrt(Cc['b0']**2 + Cc['b1']**2 + Cc['b2']**2)
    # cycle dt for the look-ahead: the base what-if dt (it reproduces the code's, C-096)
    for rm, r_f in rf(d1, 0.0625, base).items():
        dt = np.where(Cc['r'] < r_f, np.inf, w0['dt'])
        dmin = min(dt.min(), w0['dfar'])
        j = int(np.argmin(dt))
        far = w0['dfar'] < dt.min()
        shell = (Cc['r'] >= RLAPSE) & (Cc['r'] < r_f)
        out['rows'][rm] = dict(
            r_f=r_f, dt=dmin, gain=dmin / base, far=far, r=Cc['r'][j], rho=Cc['rho'][j] / RHOC, alpha=Cc['alpha'][j],
            lam=w0['lam'][j], at=w0['at'][j], bt=w0['bt'][j], zone=bool(w0['zone'][j]), capped=bool(w0['capped'][j]),
            bphys=w0['vb'][j], bcoord=bcoord[j], absorbed=Cc['dM'][shell].sum())
    return fn, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', help='LABEL:DIR[,DIR2]:T_PERIAPSIS')
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--no-restarts', action='store_true')
    ap.add_argument('--nproc', type=int, default=6)
    ap.add_argument('--out')
    a = ap.parse_args()
    out = [f'R-106 rule: r_f = max({RLAPSE}, min(r_max, d_min - {M_CELLS} dx - {VIN} dt_cycle)); rho_thr {THR:g} rho_c, '
           f'debris band rho > {BAND:g} rho_c. Radii r_iso.']
    for spec in a.runs:
        lab, dirs, tp = spec.split(':')
        dirs = dirs.split(',')
        A = slice_series(dirs, a.every)
        t, d1, d2, dxh, dtc = A.T
        out.append(f'== {lab} ({", ".join(os.path.basename(d) for d in dirs)}), periapsis ~{tp}; {len(t)} dumps, '
                   f't {t[0]:.1f}-{t[-1]:.1f}')
        out.append(f'   d_min(rho > {THR:g} rho_c): min {d1.min():.3f} at t {t[np.argmin(d1)]:.1f}; d_min(band): min '
                   f'{d2.min():.3f} at t {t[np.argmin(d2)]:.1f}')
        for rm in RMAX:
            R = np.array([max(RLAPSE, min(rm, x - M_CELLS * h - VIN * k)) for x, h, k in zip(d1, dxh, dtc)])
            marg = d1 - R
            dRdt = np.diff(R) / np.maximum(np.diff(t), 1e-12)
            cell_rate = (dxh / np.maximum(dtc, 1e-12))[1:]       # one cell per cycle, in r per unit t
            shr = -dRdt
            worst = np.argmax(shr / cell_rate) if len(shr) else 0
            at_max = np.mean(R >= rm - 1e-12)
            at_lapse = np.mean(R <= RLAPSE + 1e-12)
            out.append(f'   r_max {rm:4.2f}: r_f at r_max {at_max:.2f} of dumps, at r_lapse {at_lapse:.2f}; min guard '
                       f'margin d_min - r_f {marg.min():.3f} (t {t[np.argmin(marg)]:.1f}); max shrink rate '
                       f'{shr.max() if len(shr) else 0:.3f}/M = {(shr / cell_rate).max() if len(shr) else 0:.3f} '
                       f'cells/cycle (t {t[1:][worst] if len(shr) else 0:.1f})')
        # the time series, compactly
        sel = np.unique(np.r_[np.arange(0, len(t), max(1, len(t) // 25)), np.argmin(d1), len(t) - 1])
        out.append('   t      d_min   d_band  dx      dt_cyc  r_f(2.5) r_f(2.92) r_f(3.5)')
        for i in sel:
            rr = rf(d1[i], dxh[i], dtc[i])
            out.append(f'   {t[i]:6.1f} {d1[i]:7.3f} {d2[i]:7.3f} {dxh[i]:.4f} {dtc[i]:.4f}  {rr[2.5]:.3f}    '
                       f'{rr[2.92]:.3f}     {rr[3.5]:.3f}')
        if a.no_restarts:
            continue
        fns = []
        for d in dirs:
            fns += sorted(glob.glob(f'{d}/rst/*.rst'))
        with Pool(a.nproc) as pool:
            res = pool.map(rst_job, fns)
        out.append('   restarts (3D): t  d_min d_band base_dt | r_max: r_f dt gain limiter [r, rho/rho_c, alpha, '
                   '|beta|_phys/alpha, |beta|_coord, |lam|, zone/capped] absorbed M')
        for fn, o in sorted(res, key=lambda x: x[1]['t']):
            fixed = 'fixed' if any(k in fn for k in ('c094_p3', 'c096_', 'c097_')) else 'pinned bg'
            out.append(f'   {os.path.basename(os.path.dirname(os.path.dirname(fn)))}/{fn.split(".")[-2]} '
                       f't {o["t"]:6.1f} d_min {o["d1"]:.3f} d_band {o["d2"]:.3f} base dt {o["base"]:.4f} [{fixed}]')
            for rm, z in o['rows'].items():
                lim = ('far (r >= 10) ' if z['far'] else '') + (f'r {z["r"]:.3f} rho {z["rho"]:.1e} a {z["alpha"]:.3f} '
                       f'|b|ph/a {z["bphys"]:.2f} |b|co {z["bcoord"]:.3f} |lam| {z["lam"]:.3f} '
                       f'{"zone-capped" if z["capped"] else ("zone" if z["zone"] else "non-zone")}')
                note = ' (pinned bg)' if (fixed == 'pinned bg' and z['r_f'] < 1.5) else ''
                out.append(f'      r_max {rm:4.2f}: r_f {z["r_f"]:.3f} dt {z["dt"]:.4f} x{z["gain"]:.3f} [{lim}] '
                           f'absorbed {z["absorbed"]:.2e}{note}')
    txt = '\n'.join(out)
    print(txt)
    if a.out:
        open(a.out, 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
