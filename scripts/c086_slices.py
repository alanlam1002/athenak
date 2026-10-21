#!/usr/bin/env python3
"""C-086 items 8 and 9 from the z = 0 slices (dt = 1): the dual-energy shock flag that the code itself
stored for cells BELOW dual_energy_rho_switch, and the slice proxy of the bound mass below the switch.

Per dump (prim_xy + adm_xy + dediag_xy, same MeshBlock order; the z = 0 layer is the k = 0 cells of the
blocks with x3min = 0, i.e. z = dx/2):
  W   = sqrt(1 + g_ij u~^i u~^j)  (u~ = the stored velx/vely/velz, as the C++ vel() lambda,
                                   src/eos/primitive_solver_hyd.hpp:1614-1620)
  D   = rho W                     (the switch variable: cons(IDN)/sqrt(gamma), ibid. 1977, 2252-2253)
  below = de_branch == 0          (dbr = !dense ? 0 : ..., ibid. 2258; stored at 2355)
  flag  = de_shock > 0.5          (the final flag the C2P used, stored at ibid. 2356; computed for every
                                   cell, dense or not: ComputeShockFlags 1589-1757 has no density test)
  ratio = de_sensor               (= max(psi_v/(vcurv c_s), psi_P/psi_jump), ibid. 1671-1681; pre-dilation,
                                   no div-v test)
  c_s from the dump's primitives (as ibid. 1666-1667); psi_v estimated as ratio * vcurv * c_s
  (psi_jump = 1e30 in these decks, so the psi_P term is ~0).
Weights: count and the slice "mass" proxy sqrt(gamma) D dx^2 (area density, NOT a 3D mass).
Item 9 slice proxy: BH-frame Killing E = alpha W - (beta'^i - xidot^i) u_i - 1, u_i = g_ij u~^j, as
tde_dmde.cells (scripts/tde_dmde.py:164-167), xidot and xi from the run's user.hst frame_vel/frame_disp
(src/pgen/dyn_grmhd/dyngr_tov.cpp:1193-1201), x_BH = puncture_x - xi (tde_dmde.frame_state), r_BH >= 3
(tde_dmde.cells rcut). The remnant cut (remnant.py) needs 3D and is not applied here.

    c086_slices.py [--out DIR] [--nproc N]
"""
import argparse
import glob
import os
import re
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'vis', 'python'))
import bin_convert as bc                          # noqa: E402

C = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc'
B = 'cfc_tde_wd_imbh_parabolic_tracker'
RHOC, GAM, VCSH = 3.24e-4, 5.0 / 3.0, 0.05
RSW = 3.2e-8
EDGES = RHOC * np.array([1e-9, 1e-8, 1e-7, 1e-6, 1e-5, RSW / RHOC])
BLAB = ['1e-9..1e-8', '1e-8..1e-7', '1e-7..1e-6', '1e-6..1e-5', '1e-5..sw']
RUNS = {'R6': ['R6_rp10_comov_L4_fix', 'R6x_rp10_comov_L4_t300'],
        'R6N1': ['RC_R6N1'],
        'R7': ['R7_rp18_comov_L4_fix', 'R7x_rp18_comov_L4_t500']}
WIN = {'R6': (0, 80, 110, 130, 150, 170, 200, 250, 300.01),
       'R6N1': (113, 130, 150, 170.01),
       'R7': (0, 80, 110, 130, 150, 200, 250, 300, 400, 500.01)}


def hst(d):
    a = np.loadtxt(f'{d}/{B}.user.hst')
    par = open(f'{d}/parfile.par').read()
    px = [float(re.search(rf'^puncture_x{i}\s*=\s*(\S+)', par, re.M).group(1)) for i in (1, 2, 3)]
    return a[:, 0], a[:, 4:7], a[:, 7:10], np.array(px)


def one(job):
    lab, d, num = job
    f = lambda k: f'{C}/{d}/bin/{B}.{k}.{num}.bin'
    p, a, q = bc.read_binary(f('prim_xy')), bc.read_binary(f('adm_xy')), bc.read_binary(f('dediag_xy'))
    g = np.asarray(p['mb_geometry'])
    assert np.array_equal(g, np.asarray(a['mb_geometry'])) and np.array_equal(g, np.asarray(q['mb_geometry']))
    t = p['time']
    P_, A_, Q_ = p['mb_data'], a['mb_data'], q['mb_data']
    v = lambda dd, k: np.asarray(dd[k])[:, 0]
    rho, pr = v(P_, 'dens'), v(P_, 'press')
    u = [v(P_, 'velx'), v(P_, 'vely'), v(P_, 'velz')]
    gn = ['adm_gxx', 'adm_gxy', 'adm_gxz', 'adm_gyy', 'adm_gyz', 'adm_gzz']
    gxx, gxy, gxz, gyy, gyz, gzz = (v(A_, k) for k in gn)
    G = [[gxx, gxy, gxz], [gxy, gyy, gyz], [gxz, gyz, gzz]]
    ul = [sum(G[i][j] * u[j] for j in range(3)) for i in range(3)]
    W = np.sqrt(1 + sum(ul[i] * u[i] for i in range(3)))
    det = gxx * (gyy * gzz - gyz * gyz) - gxy * (gxy * gzz - gyz * gxz) + gxz * (gxy * gyz - gyy * gxz)
    sg = np.sqrt(det)
    D = rho * W
    nx = p['nx1_out_mb']
    dx = (g[:, 1] - g[:, 0]) / nx
    dA = (dx**2)[:, None, None] * np.ones_like(D)
    m = sg * D * dA
    th, xd, xi, px = HST[d]
    xdot = np.array([np.interp(t, th, xd[:, i]) for i in range(3)])
    xbh = px - np.array([np.interp(t, th, xi[:, i]) for i in range(3)])
    xc = g[:, 0, None, None] + (np.arange(nx)[None, None, :] + .5) * dx[:, None, None]
    yc = g[:, 2, None, None] + (np.arange(nx)[None, :, None] + .5) * dx[:, None, None]
    xc, yc = np.broadcast_to(xc, D.shape), np.broadcast_to(yc, D.shape)
    zc = (g[:, 4] + 0.5 * dx)[:, None, None] * np.ones_like(D)
    rbh = np.sqrt((xc - xbh[0])**2 + (yc - xbh[1])**2 + (zc - xbh[2])**2)
    al = v(A_, 'adm_alpha')
    be = [v(A_, 'adm_betax') - xdot[0], v(A_, 'adm_betay') - xdot[1], v(A_, 'adm_betaz') - xdot[2]]
    E = al * W - sum(be[i] * ul[i] for i in range(3)) - 1.0
    br, ratio, shk = np.rint(v(Q_, 'de_branch')), v(Q_, 'de_sensor'), v(Q_, 'de_shock') > 0.5
    h = 1 + GAM / (GAM - 1) * pr / np.maximum(rho, 1e-300)
    cs = np.sqrt(np.maximum(GAM * pr / (np.maximum(rho, 1e-300) * h), 1e-300))
    psiv = ratio * VCSH * cs
    out = r_ = rbh >= 3.0
    below = (br == 0) & r_
    k = np.argmax(np.where(r_, rho, -1))
    sep = rbh.flat[k]
    res = dict(t=t, sep=sep, nbelow_sw_check=int(((D > RSW) & below).sum()), ndense_check=int(((D <= RSW) & (br > 0) & r_).sum()))
    rows = []
    for b in range(len(BLAB)):
        s = below & (D > EDGES[b]) & (D <= EDGES[b + 1])
        fl = s & shk
        rows.append((s.sum(), fl.sum(), m[s].sum(), m[fl].sum(), np.median(cs[s]) if s.any() else np.nan,
                     np.median(cs[fl]) if fl.any() else np.nan, np.median(psiv[fl]) if fl.any() else np.nan,
                     np.median(psiv[s]) if s.any() else np.nan))
    res['bins'] = np.array(rows, float)
    # above the switch, for scale
    s = (br > 0) & r_
    res['dense'] = (s.sum(), (s & shk).sum(), m[s].sum(), m[s & shk].sum())
    # flagged below-switch cells: (log10 psi_v, log10 c_s) samples, all bins
    s = below & (D > EDGES[0]) & shk
    res['fl_samples'] = np.stack([np.log10(np.maximum(psiv[s], 1e-30)), np.log10(cs[s]), np.log10(D[s] / RHOC)])
    # item 9 slice proxy
    bnd = out & (E < 0)
    mb = m[bnd].sum()
    res['i9'] = (mb, m[bnd & (D < RSW)].sum(), m[bnd & (D < 1e-9 * RHOC)].sum(),
                 m[bnd & (D < RSW) & (D >= 1e-9 * RHOC)].sum(), m[out].sum())
    return lab, d, res


HST = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{C}/plots/phase2_m1')
    ap.add_argument('--nproc', type=int, default=24)
    a = ap.parse_args()
    jobs = []
    for lab, ds in RUNS.items():
        for d in ds:
            HST[d] = hst(f'{C}/{d}')
            for fn in sorted(glob.glob(f'{C}/{d}/bin/{B}.dediag_xy.*.bin')):
                jobs.append((lab, d, fn.split('.')[-2]))
    with Pool(a.nproc) as pool:
        res = pool.map(one, jobs)
    L = ['C-086 items 8/9 from the z = 0 slices (scripts/c086_slices.py). Stored in-code flag (dediag de_shock) of cells',
         'BELOW the switch (de_branch == 0, D <= 3.2e-8), r_BH >= 3, binned in D/rho_c. "frac_n" = flagged count fraction,',
         '"frac_m" = flagged fraction of the slice mass proxy sqrt(gamma) D dx^2. cs, psi_v medians (psi_v = de_sensor x 0.05 x c_s).',
         'Consistency: cells with de_branch == 0 but D > rho_sw, and de_branch > 0 but D <= rho_sw, are counted (D from the dump',
         'primitives; the C2P tests the pre-update D, so a few mismatches are expected).', '']
    per = {}
    seen = set()
    for lab, d, r in sorted(res, key=lambda x: (x[0], x[2]['t'])):
        key = (lab, round(r['t'], 3))
        if key in seen:                       # a dump repeated at a segment boundary
            continue
        seen.add(key)
        per.setdefault(lab, []).append(r)
    np.savez_compressed(f'{a.out}/c086_slices.npz', **{f'{lab}_{k}': np.array([r[k] for r in rs]) for lab, rs in per.items()
                                                       for k in ('t', 'sep', 'bins', 'dense', 'i9')})
    for lab, rs in per.items():
        L.append(f'=== {lab} ({", ".join(RUNS[lab])}): {len(rs)} dumps, t {rs[0]["t"]:.1f}..{rs[-1]["t"]:.1f}; mismatches '
                 f'{sum(r["nbelow_sw_check"] for r in rs)} / {sum(r["ndense_check"] for r in rs)}')
        L.append('  window      sep(r_BH of rho_max)  | per D/rho_c bin: N_below  frac_n  frac_m  [med c_s all/flagged, med psi_v flagged]'
                 '  || dense (above sw) frac_n frac_m')
        w = WIN[lab]
        for t0, t1 in zip(w[:-1], w[1:]):
            sel = [r for r in rs if t0 <= r['t'] < t1]
            if not sel:
                continue
            Bn = np.sum([r['bins'][:, :4] for r in sel], axis=0)
            L.append(f'  t {t0:5.0f}-{t1:5.0f} ({len(sel):3d} dumps) sep {np.min([r["sep"] for r in sel]):5.1f}-'
                     f'{np.max([r["sep"] for r in sel]):5.1f}')
            for b in range(len(BLAB)):
                cs_all = np.nanmedian([r['bins'][b, 4] for r in sel])
                cs_fl = np.nanmedian([r['bins'][b, 5] for r in sel])
                ps_fl = np.nanmedian([r['bins'][b, 6] for r in sel])
                L.append(f'      {BLAB[b]:11s} N {int(Bn[b, 0]):8d}  frac_n {Bn[b, 1] / max(Bn[b, 0], 1):.4f}  frac_m '
                         f'{Bn[b, 3] / max(Bn[b, 2], 1e-300):.4f}   c_s {cs_all:.2e}/{cs_fl:.2e}  psi_v_fl {ps_fl:.2e}')
            dn = np.sum([r['dense'] for r in sel], axis=0)
            L.append(f'      dense (D > sw)  N {int(dn[0]):8d}  frac_n {dn[1] / max(dn[0], 1):.4f}  frac_m {dn[3] / max(dn[2], 1e-300):.4f}')
        L.append('')
        # flagged-cell drivers: pooled samples
        S = np.concatenate([r['fl_samples'] for r in rs], axis=1)
        if S.shape[1]:
            L.append(f'  flagged below-switch cells, all dumps pooled ({S.shape[1]}): log10 psi_v p10/50/90 '
                     + '/'.join(f'{x:.2f}' for x in np.percentile(S[0], [10, 50, 90]))
                     + '; log10 c_s p10/50/90 ' + '/'.join(f'{x:.2f}' for x in np.percentile(S[1], [10, 50, 90])))
            for b in range(len(BLAB)):
                lo, hi = np.log10(EDGES[b] / RHOC), np.log10(EDGES[b + 1] / RHOC)
                s = (S[2] > lo) & (S[2] <= hi)
                if s.sum() > 5:
                    L.append(f'      {BLAB[b]:11s} n {s.sum():7d}: log10 psi_v p10/50/90 '
                             + '/'.join(f'{x:.2f}' for x in np.percentile(S[0][s], [10, 50, 90]))
                             + '   log10 c_s ' + '/'.join(f'{x:.2f}' for x in np.percentile(S[1][s], [10, 50, 90])))
        L.append('')
        L.append(f'  item 9 SLICE PROXY ({lab}): bound (E < 0, r_BH >= 3) slice-mass fraction with D < rho_sw and D < 1e-9 rho_c')
        for t0, t1 in zip(w[:-1], w[1:]):
            sel = [r for r in rs if t0 <= r['t'] < t1]
            if not sel:
                continue
            i9 = np.array([r['i9'] for r in sel])
            f1 = i9[:, 1] / np.maximum(i9[:, 0], 1e-300)
            f2 = i9[:, 2] / np.maximum(i9[:, 0], 1e-300)
            fb = i9[:, 0] / np.maximum(i9[:, 4], 1e-300)
            L.append(f'      t {t0:5.0f}-{t1:5.0f}: bound share of slice mass {np.median(fb):.3f}; below sw: median {np.median(f1):.3e} '
                     f'(range {f1.min():.2e}-{f1.max():.2e}); below 1e-9 rho_c: median {np.median(f2):.3e} (max {f2.max():.2e})')
        L.append('')
    print('\n'.join(L))
    with open(f'{a.out}/c086_slices.txt', 'w') as fo:
        fo.write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
