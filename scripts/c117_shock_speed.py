#!/usr/bin/env python3
"""R-127 Q3 / R-128 (c117): the shock's grid speed |(alpha v - beta) . n|_coord at the flagged dense
cells of the rp10 2x2 arms, from the z = 0 slices every 2 M (prim_xy W v^i, rho, P; adm_xy alpha,
beta^i, psi4; dediag_xy de_shock). n = -grad P/|grad P| in the plane (central differences inside
each MeshBlock; edge cells skipped). Eulerian v^i = u^i/W, W = sqrt(1 + psi4 |u|^2) (CFC: gamma_ij =
psi4 delta_ij); |v|_phys = psi2 |v|; KE/e_int = (W - 1)/eps. Mass weight rho W psi6 dA.
Flags: DE-on arms the code's final flag (dediag de_shock, the last C2P's); DE-off arms (no dediag)
the vcurv sensor recomputed IN THE PLANE (psi_v over x and y only, div v <= 0 in-plane, one
dilation pass at 0.5 x threshold) -- an approximation. Dense = D = rho W psi6 > rho_sw = 3.2e-8.
Phase bins (rp10, t_p = 135.9): pre < t_p - 10, peri t_p +- 10, early to t_p + 50, late to 300.
Per arm x phase: weighted quantiles of |V.n|_coord, |v|_phys, KE/e_int over flagged dense cells, the
flagged mass by density band, and the mass-weighted |V.n| per band.
    c117_shock_speed.py OUT.csv LABEL=RUNDIR [...] [--nproc 16] [--every 1]
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tde_ensemble_plots import bc                 # noqa: E402

B = 'cfc_tde_wd_imbh_parabolic_tracker'
RHOC, GAM, RSW, VCSH = 3.24e-4, 5.0 / 3.0, 3.2e-8, 0.05
TP = 135.9
PH = [('pre', -1, TP - 10), ('peri', TP - 10, TP + 10), ('early', TP + 10, TP + 50), ('late', TP + 50, 300.01)]
BANDS = [(1e-1, 1e9), (1e-2, 1e-1), (1e-3, 1e-2), (0, 1e-3)]   # rho/rho_c


def recomputed_flag(vx, vy, rho, p, inn):
    h = 1 + GAM / (GAM - 1) * p / rho
    cs = np.sqrt(GAM * p / (rho * h))
    psiv = np.zeros_like(rho)
    div = np.zeros_like(rho)
    psiv[:, 1:-1, :] = np.maximum(psiv[:, 1:-1, :], np.abs(vy[:, 2:, :] - 2 * vy[:, 1:-1, :] + vy[:, :-2, :]))
    psiv[:, :, 1:-1] = np.maximum(psiv[:, :, 1:-1], np.abs(vx[:, :, 2:] - 2 * vx[:, :, 1:-1] + vx[:, :, :-2]))
    div[:, 1:-1, :] += vy[:, 2:, :] - vy[:, :-2, :]
    div[:, :, 1:-1] += vx[:, :, 2:] - vx[:, :, :-2]
    sh = (div <= 0) & (psiv > VCSH * cs)
    wk = (div <= 0) & (psiv > 0.5 * VCSH * cs)
    nb = np.zeros_like(sh)
    nb[:, 1:, :] |= sh[:, :-1, :]; nb[:, :-1, :] |= sh[:, 1:, :]
    nb[:, :, 1:] |= sh[:, :, :-1]; nb[:, :, :-1] |= sh[:, :, 1:]
    return (sh | (wk & nb)) & inn


def one(arg):
    lab, d, fp, deon = arg
    try:
        P = bc.read_binary(fp)
        A = bc.read_binary(fp.replace('prim_xy', 'adm_xy'))
        G = bc.read_binary(fp.replace('prim_xy', 'dediag_xy')) if deon else None
    except Exception:
        return None
    t = P['time']
    g = np.asarray(P['mb_geometry'])
    if not np.array_equal(g, np.asarray(A['mb_geometry'])):
        return None
    nx, ny = P['nx1_out_mb'], P['nx2_out_mb']
    q = {k: np.asarray(P['mb_data'][k])[:, 0] for k in ('dens', 'velx', 'vely', 'velz', 'press')}
    al = np.asarray(A['mb_data']['adm_alpha'])[:, 0]
    bx, by = np.asarray(A['mb_data']['adm_betax'])[:, 0], np.asarray(A['mb_data']['adm_betay'])[:, 0]
    p4 = np.asarray(A['mb_data']['adm_psi4'])[:, 0]
    rho, p = q['dens'], q['press']
    u2 = p4 * (q['velx']**2 + q['vely']**2 + q['velz']**2)
    W = np.sqrt(1 + u2)
    vx, vy = q['velx'] / W, q['vely'] / W
    hx = ((g[:, 1] - g[:, 0]) / nx)[:, None, None]
    hy = ((g[:, 3] - g[:, 2]) / ny)[:, None, None]
    inn = np.zeros(rho.shape, bool)
    inn[:, 1:-1, 1:-1] = True
    if deon:
        fl = (np.asarray(G['mb_data']['de_shock'])[:, 0] > 0.5) & inn
    else:
        fl = recomputed_flag(vx, vy, rho, p, inn)
    gx = np.zeros_like(p); gy = np.zeros_like(p)
    gx[:, :, 1:-1] = (p[:, :, 2:] - p[:, :, :-2]) / (2 * hx)
    gy[:, 1:-1, :] = (p[:, 2:, :] - p[:, :-2, :]) / (2 * hy)
    gn = np.hypot(gx, gy)
    D = rho * W * p4**1.5
    sel = fl & (D > RSW) & (gn > 0)
    nxh, nyh = -gx[sel] / gn[sel], -gy[sel] / gn[sel]
    Vn = np.abs((al[sel] * vx[sel] - bx[sel]) * nxh + (al[sel] * vy[sel] - by[sel]) * nyh)
    vphys = np.sqrt(u2[sel]) / W[sel]
    eps = p[sel] / ((GAM - 1) * rho[sel])
    ke = (W[sel] - 1) / eps
    w = (rho * W * p4**1.5 * hx * hy)[sel]
    dense_mass = (rho * W * p4**1.5 * hx * hy)[(D > RSW) & inn].sum()
    return lab, t, Vn, vphys, ke, rho[sel] / RHOC, w, dense_mass


def wq(x, w, qs):
    if len(x) == 0:
        return [np.nan] * len(qs)
    o = np.argsort(x)
    c = np.cumsum(w[o]); c /= c[-1]
    return [x[o][np.searchsorted(c, qq)] for qq in qs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--nproc', type=int, default=16)
    ap.add_argument('--every', type=int, default=1)
    a = ap.parse_args()
    jobs = []
    for spec in a.runs:
        lab, d = spec.split('=', 1)
        deon = bool(glob.glob(f'{d}/bin/{B}.dediag_xy.*.bin'))
        jobs += [(lab, d, f, deon) for f in sorted(glob.glob(f'{d}/bin/{B}.prim_xy.*.bin'))[::a.every]]
    with Pool(a.nproc) as pool:
        R = [r for r in pool.map(one, jobs) if r is not None]
    rows = ['label,phase,ndumps,flag_mass_frac,Vn_p10,Vn_p50,Vn_p90,Vn_mean,vphys_p50,vphys_p90,KEe_p10,KEe_p50,KEe_p90,'
            + ','.join(f'band{i}_massfrac,band{i}_Vn_mean' for i in range(len(BANDS)))]
    for lab in dict.fromkeys(r[0] for r in R):
        for ph, t0, t1 in PH:
            S = [r for r in R if r[0] == lab and t0 <= r[1] < t1]
            if not S:
                continue
            cat = [np.concatenate([s[k] for s in S]) for k in range(2, 7)]
            Vn, vp, ke, rr, w = cat
            dm = sum(s[7] for s in S)
            fr = w.sum() / dm if dm > 0 else np.nan
            line = [lab, ph, str(len(S)), f'{fr:.4f}'] + [f'{x:.4f}' for x in wq(Vn, w, (0.1, 0.5, 0.9))]
            line += [f'{(Vn * w).sum() / w.sum():.4f}' if len(w) else 'nan']
            line += [f'{x:.4f}' for x in wq(vp, w, (0.5, 0.9))] + [f'{x:.4g}' for x in wq(ke, w, (0.1, 0.5, 0.9))]
            for lo, hi in BANDS:
                m = (rr >= lo) & (rr < hi)
                line += [f'{w[m].sum() / w.sum() if len(w) else np.nan:.4f}',
                         f'{(Vn[m] * w[m]).sum() / w[m].sum() if m.any() else np.nan:.4f}']
            rows.append(','.join(line))
    open(a.out, 'w').write('\n'.join(rows) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
