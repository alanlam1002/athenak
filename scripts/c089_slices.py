#!/usr/bin/env python3
"""C-089 (R-092 item 4): the z = 0 slice part of the sub-switch census, at EXACTLY matched times (dump number).

Reader and definitions as scripts/c086_slices.py:47-121 (prim_xy + adm_xy + dediag_xy, same MeshBlock order;
W from the stored u~ and g_ij; D = rho W; below the switch = stored de_branch == 0; flag = stored de_shock > 0.5,
the final flag the C2P used; psi_v = de_sensor x vcurv x c_s; slice "mass" = sqrt(gamma) D dx^2 (an AREA-density
proxy, not a 3D mass); x_BH, xidot from user.hst via c086_slices.hst; r_BH >= 3).
Adds:
  hot      = K/K0 = P/(K0 rho^Gamma) > 100, from the stored dens/press (the code's own primitives);
  T floor  = the stored temperature (prim PTM written at primitive_solver_hyd.hpp:2431, output as "temperature",
             outputs/basetype_output.cpp:483) equal to tfloor to 1e-9 relative; also T < 1e-14 (the 3D proxy);
  fv-reset = |u~ - W xidot| <= 1e-6 |u~| (the old-path floor velocity, primitive_solver_hyd.hpp:2213-2230;
             xidot from user.hst frame_vel interpolated to the dump time, so only approximate).
Run sets (labels): matched L4/L5 pairs (R6 vs L5c; RC_R6B vs RC_L5B), zone pairs (S2T_R6 vs S2B_R6; RC_R7T vs
RC_R7B), floor census (RC_R6X11 vs RC_R6N1).

    c089_slices.py [--nproc N] [--cache FILE]
"""
import argparse
import glob
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c086_slices as S86                         # noqa: E402
bc = S86.bc

C = S86.C
B = S86.B
RHOC, GAM, VCSH, RSW = 3.24e-4, 5.0 / 3.0, 0.05, 3.2e-8
K0 = 1.2346626297e-2
LO = 1e-9 * RHOC
TFLOOR = 1e-22                                    # deck tfloor (verified in each restart's parameter dump)
REDG = np.array([3, 5, 8, 15, 30, np.inf])
SETS = {'R6': ['R6_rp10_comov_L4_fix', 'R6x_rp10_comov_L4_t300'], 'L5': ['L5c_rp10_comov_t200'],
        'R6B': ['RC_R6B'], 'L5B': ['RC_L5B'],
        'S2T': ['S2T_R6'], 'S2B': ['S2B_R6'], 'R7T': ['RC_R7T'], 'R7B': ['RC_R7B'],
        'X11': ['RC_R6X11'], 'N1': ['RC_R6N1'], 'N2c': ['RC_R6N2c']}
STEP = {'R6': 10, 'L5': 10}                      # every 10th dump for the long L4/L5 pair; all dumps otherwise


def one(job):
    lab, d, num = job
    f = lambda k: f'{C}/{d}/bin/{B}.{k}.{num}.bin'
    p, a, q = bc.read_binary(f('prim_xy')), bc.read_binary(f('adm_xy')), bc.read_binary(f('dediag_xy'))
    g = np.asarray(p['mb_geometry'])
    t = p['time']
    v = lambda dd, k: np.asarray(dd[k])[:, 0]
    P_, A_, Q_ = p['mb_data'], a['mb_data'], q['mb_data']
    rho, pr, T = v(P_, 'dens'), v(P_, 'press'), v(P_, 'temperature')
    u = [v(P_, 'velx'), v(P_, 'vely'), v(P_, 'velz')]
    gxx, gxy, gxz, gyy, gyz, gzz = (v(A_, k) for k in ['adm_gxx', 'adm_gxy', 'adm_gxz', 'adm_gyy', 'adm_gyz', 'adm_gzz'])
    G = [[gxx, gxy, gxz], [gxy, gyy, gyz], [gxz, gyz, gzz]]
    ul = [sum(G[i][j] * u[j] for j in range(3)) for i in range(3)]
    W = np.sqrt(1 + sum(ul[i] * u[i] for i in range(3)))
    det = gxx * (gyy * gzz - gyz * gyz) - gxy * (gxy * gzz - gyz * gxz) + gxz * (gxy * gyz - gyy * gxz)
    sg = np.sqrt(det)
    Dd = rho * W
    nx = p['nx1_out_mb']
    dx = (g[:, 1] - g[:, 0]) / nx
    m = sg * Dd * (dx**2)[:, None, None]
    th, xd, xi, px = S86.HST[d]
    xdot = np.array([np.interp(t, th, xd[:, i]) for i in range(3)])
    xbh = px - np.array([np.interp(t, th, xi[:, i]) for i in range(3)])
    xc = g[:, 0, None, None] + (np.arange(nx)[None, None, :] + .5) * dx[:, None, None]
    yc = g[:, 2, None, None] + (np.arange(nx)[None, :, None] + .5) * dx[:, None, None]
    xc, yc = np.broadcast_to(xc, Dd.shape), np.broadcast_to(yc, Dd.shape)
    zc = (g[:, 4] + 0.5 * dx)[:, None, None] * np.ones_like(Dd)
    rbh = np.sqrt((xc - xbh[0])**2 + (yc - xbh[1])**2 + (zc - xbh[2])**2)
    br, ratio, shk = np.rint(v(Q_, 'de_branch')), v(Q_, 'de_sensor'), v(Q_, 'de_shock') > 0.5
    h = 1 + GAM / (GAM - 1) * pr / np.maximum(rho, 1e-300)
    cs = np.sqrt(np.maximum(GAM * pr / (np.maximum(rho, 1e-300) * h), 1e-300))
    psiv = ratio * VCSH * cs
    kk0 = pr / (K0 * np.maximum(rho, 1e-300)**GAM)
    band = (br == 0) & (Dd > LO) & (Dd <= RSW) & (rbh >= 3.0)
    hot = band & (kk0 > 100)
    tfl = band & (np.abs(T / TFLOOR - 1) < 1e-9)
    tlo = band & (T < 1e-14)
    fvr = np.sqrt(sum((u[i] - W * xdot[i])**2 for i in range(3)))
    un = np.sqrt(sum(u[i]**2 for i in range(3)))
    fvreset = tfl & (fvr <= 1e-6 * np.maximum(un, 1e-30))
    dxl = np.broadcast_to(dx[:, None, None], Dd.shape)
    levs = np.unique(np.round(dx, 8))
    res = dict(lab=lab, d=d, num=num, t=t)
    res['M_band'], res['M_flag'], res['M_hot'] = m[band].sum(), m[band & shk].sum(), m[hot].sum()
    res['M_tfl'], res['M_tlo'], res['M_fvreset'] = m[tfl].sum(), m[tlo].sum(), m[fvreset].sum()
    res['N_band'], res['N_tfl'] = int(band.sum()), int(tfl.sum())
    res['M_tfl_hot'] = m[tfl & hot].sum()
    # by r_BH bin: band, hot, flagged
    ib = np.digitize(rbh, REDG) - 1
    res['r_band'] = np.array([m[band & (ib == k)].sum() for k in range(len(REDG) - 1)])
    res['r_hot'] = np.array([m[hot & (ib == k)].sum() for k in range(len(REDG) - 1)])
    res['r_flag'] = np.array([m[band & shk & (ib == k)].sum() for k in range(len(REDG) - 1)])
    # fine r profile of hot mass (0.5 M bins, 3..20) for the zone test
    fe = np.arange(3, 20.01, 0.5)
    res['rf_hot'] = np.histogram(rbh[hot], fe, weights=m[hot])[0]
    res['rf_band'] = np.histogram(rbh[band], fe, weights=m[band])[0]
    # flagged cells: psi_v (mass-weighted samples) and by dx
    s = band & shk
    res['psiv_fl'] = (psiv[s].astype(np.float32), m[s].astype(np.float32))
    res['by_dx'] = {float(L): (m[band & (np.abs(dxl - L) < 1e-8)].sum(), m[s & (np.abs(dxl - L) < 1e-8)].sum(),
                               m[hot & (np.abs(dxl - L) < 1e-8)].sum()) for L in levs}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nproc', type=int, default=16)
    ap.add_argument('--cache', required=True)
    a = ap.parse_args()
    jobs = []
    for lab, ds in SETS.items():
        for d in ds:
            S86.HST[d] = S86.hst(f'{C}/{d}')
            fs = sorted(glob.glob(f'{C}/{d}/bin/{B}.dediag_xy.*.bin'))
            for fn in fs:
                n = fn.split('.')[-2]
                if int(n) % STEP.get(lab, 1) == 0:
                    jobs.append((lab, d, n))
    with Pool(a.nproc) as pool:
        res = pool.map(one, jobs, chunksize=2)
    with open(a.cache, 'wb') as fo:
        pickle.dump(res, fo)
    print(len(res), 'dumps')


if __name__ == '__main__':
    main()
