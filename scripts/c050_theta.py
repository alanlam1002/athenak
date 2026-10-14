#!/usr/bin/env python3
"""C-050 (R-064 step 1): offline expansion-scalar gate comparison, coord vs covariant vs theta.

For every dense cell (D > dual_energy_rho_switch) on the z = 0 layer of each L5 / R6 restart
(blocks with x3min = 0, interior k = 0, the layer the *_xy slices store):

  theta = W (D_i v^i + v^i d_i ln alpha) + (W/alpha) (d_t + V^i d_i) ln W - W K,
  D_i v^i = d_i v^i + v^i d_i ln sqrt(gamma),   V^i = alpha v^i - beta'^i,   K = gamma^ij K_ij,

from (1/(alpha sqrt gamma)) d_a(alpha sqrt(gamma) u^a), u^t = W/alpha, u^i = W(v^i - beta^i/alpha),
d_t ln sqrt(gamma) = -alpha K + D_i beta^i. It is exact for any shift (comoving gauge included).
The last-but-one term is d ln W/d tau (tau = proper time, d/dtau = u^a d_a); R-064 writes it as
(1/alpha)(...) (Eulerian time); both forms are evaluated ('theta' and 'theta_E').

Spatial derivatives: central differences on the restart's 3D block (ghosts included), as the
de_shock_flag kernel (v^a = u~^a/W, W from the ideal-gas C2P of the conserved state). The
time derivative d_t ln W at fixed grid coordinates comes from the prim_xy + adm_xy dumps
bracketing the restart time (W = sqrt(1 + g_ij u~^i u~^j)): 'dt1' = the two dumps around t
(spacing ~1 M), 'dt2' = one dump further on each side (~3 M), as a convergence check. Cells
whose slice block does not match the restart block (regrid between dump and restart) are
dropped from theta and counted.

Flag replica: the kernel's vcurv sensor (ratio = psi_v/(vcurv cs) or psi_P/psi_jump) with the
divergence gate from each form; the gate statistics use the sign test div <= 0 per cell.

Validation (first): bulk star (rho > 0.1 rho_max on the layer, coord flag off) before
periapsis: median |theta| against median |d_i v^i| and |v.grad ln sqrt(gamma)|; cells with
sensor ratio > 2 (captured shocks): fraction theta < 0 and median theta.
Gates per window (t < 80, 80-120, 120-160, 160-200) and run:
  (a) cells where sign(coord) != sign(covariant): fraction where covariant agrees with theta;
  (b) cells with sensor ratio > 0.5: fraction where coord disagrees with theta;
  (c) all dense cells: fraction where theta and covariant disagree.
R-064 rule: GO if (a) >= 0.90 in t < 80 and 80-120 for both runs and (b) >= 0.10 in at least
one of those; STOP if (c) > 0.10 in any pre-periapsis window (takes precedence); else NO-GO.

    c050_theta.py [--out DIR] [--nproc N]
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tde_dmde as D                              # noqa: E402
from tde_ensemble_plots import bc                 # noqa: E402
from c049_divflag import deck, C, RUNS, BASE, PHASES   # noqa: E402

PRE = ((0, 80), (80, 120))


def slice_lnW(d, num):
    """{block key: (ln W, ln rho) on the 16x16 layer}, dump time"""
    qp = bc.read_binary(f'{d}/bin/{BASE}.prim_xy.{num:05d}.bin')
    qa = bc.read_binary(f'{d}/bin/{BASE}.adm_xy.{num:05d}.bin')
    g = np.asarray(qp['mb_geometry'])
    u = [np.asarray(qp['mb_data'][k])[:, 0] for k in ('velx', 'vely', 'velz')]
    G = {k: np.asarray(qa['mb_data'][f'adm_g{k}'])[:, 0] for k in ('xx', 'xy', 'xz', 'yy', 'yz', 'zz')}
    gd = [[G['xx'], G['xy'], G['xz']], [G['xy'], G['yy'], G['yz']], [G['xz'], G['yz'], G['zz']]]
    usq = sum(gd[i][j] * u[i] * u[j] for i in range(3) for j in range(3))
    lnW = 0.5 * np.log1p(np.maximum(usq, 0))
    lnr = np.log(np.maximum(np.asarray(qp['mb_data']['dens'])[:, 0], 1e-300))
    out = {}
    for i in range(g.shape[0]):
        out[tuple(round(x, 6) for x in (g[i, 0], g[i, 2], g[i, 1], g[i, 3]))] = (lnW[i], lnr[i])
    return out, qp['time']


def block(rec, nscal, dx, p):
    nx, ng = 16, 4
    no = nx + 2 * ng; nmhd = 5 + nscal; ncc = no**3; bsz = (no + 1) * no * no * 3
    cons = rec[:nmhd * ncc].reshape(nmhd, no, no, no)
    adm = rec[nmhd * ncc + bsz:].reshape(17, no, no, no)
    gxx, gxy, gxz, gyy, gyz, gzz = adm[:6]
    det = gxx * (gyy * gzz - gyz * gyz) - gxy * (gxy * gzz - gyz * gxz) + gxz * (gxy * gyz - gyy * gxz)
    sg = np.sqrt(np.maximum(det, 1e-300))
    I = slice(ng, ng + nx)
    Dd = cons[0] / sg
    dense = Dd[ng, I, I] > p['rho_sw']                       # z = 0 layer only (interior k = 0)
    if not dense.any():
        return None
    ixx = (gyy * gzz - gyz * gyz) / det; ixy = (gxz * gyz - gxy * gzz) / det
    ixz = (gxy * gyz - gxz * gyy) / det; iyy = (gxx * gzz - gxz * gxz) / det
    iyz = (gxy * gxz - gxx * gyz) / det; izz = (gxx * gyy - gxy * gxy) / det
    gi = [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]
    S = [cons[1] / sg, cons[2] / sg, cons[3] / sg]; tau = cons[4] / sg
    ok = (Dd > 0) & np.isfinite(tau)
    Dc = np.where(ok, Dd, 1e-30); tc = np.where(ok, np.maximum(tau, 0), 0)
    Ssq = np.where(ok, sum(gi[i][j] * S[i] * S[j] for i in range(3) for j in range(3)), 0)
    rho, P, W, h = D.c2p(Dc, Ssq, tc)
    v = [np.where(ok, sum(gi[i][j] * S[j] for j in range(3)) / (Dc * h * W), 0) for i in range(3)]
    alpha = adm[13]; beta = adm[14:17]
    Kij = adm[6:12]                                            # Kxx Kxy Kxz Kyy Kyz Kzz
    trK = (ixx * Kij[0] + iyy * Kij[3] + izz * Kij[5] + 2 * (ixy * Kij[1] + ixz * Kij[2] + iyz * Kij[4]))
    lsg, lal, lW = np.log(sg), np.log(np.maximum(alpha, 1e-300)), np.log(W)
    gam = p['gam']
    cs = np.sqrt(np.maximum(gam * P / (np.maximum(rho, 1e-300) * (1 + gam / (gam - 1) * P / np.maximum(rho, 1e-300))), 1e-300))
    k = ng                                                     # the z = 0 layer, j/i interior
    def d(a, q):                                               # central difference on the layer
        if a == 0:
            return (q[k, I, ng + 1:ng + nx + 1] - q[k, I, ng - 1:ng + nx - 1]) / (2 * dx)
        if a == 1:
            return (q[k, ng + 1:ng + nx + 1, I] - q[k, ng - 1:ng + nx - 1, I]) / (2 * dx)
        return (q[k + 1, I, I] - q[k - 1, I, I]) / (2 * dx)
    def c2(a, q):
        if a == 0:
            return q[k, I, ng + 1:ng + nx + 1] - 2 * q[k, I, I] + q[k, I, ng - 1:ng + nx - 1]
        if a == 1:
            return q[k, ng + 1:ng + nx + 1, I] - 2 * q[k, I, I] + q[k, ng - 1:ng + nx - 1, I]
        return q[k + 1, I, I] - 2 * q[k, I, I] + q[k - 1, I, I]
    L = lambda q: q[k, I, I]
    vL = [L(v[a]) for a in range(3)]
    divc = sum(d(a, v[a]) for a in range(3))
    cov = sum(vL[a] * d(a, lsg) for a in range(3))
    lap = sum(vL[a] * d(a, lal) for a in range(3))
    V = [L(alpha) * vL[a] - L(beta[a]) for a in range(3)]
    advW = sum(V[a] * d(a, lW) for a in range(3))
    advR = sum(V[a] * d(a, np.log(np.maximum(rho, 1e-300))) for a in range(3))
    psiv = np.max([np.abs(c2(a, v[a])) for a in range(3)], axis=0)
    psip = np.max([np.abs(c2(a, P)) / np.maximum(c2(a, P) + 4 * L(P), 1e-300) for a in range(3)], axis=0)
    thr = p['vcsh'] * L(cs)
    ratio = np.maximum(psiv / thr, psip / p['psijp'])
    return dict(dense=dense, rho=L(rho), W=L(W), alpha=L(alpha), K=L(trK), divc=divc, cov=cov, lap=lap,
                advW=advW, advR=advR, ratio=ratio)


def one(job):
    lab, fn = job
    t = D.header(fn)
    p = deck(t['par'])
    d = os.path.dirname(os.path.dirname(fn))
    fs = sorted(glob.glob(f'{d}/bin/{BASE}.prim_xy.*.bin'))
    nums = np.array([int(f.split('.')[-2]) for f in fs])
    tt = t['time']
    nb = int(nums[nums <= tt + 1e-9].max()) if (nums <= tt + 1e-9).any() else int(nums.min())
    na = nb + 1 if nb + 1 <= nums.max() else nb - 1         # bracket; at the end, backward
    pair1 = tuple(sorted((nb, na)))
    pair2 = (max(pair1[0] - 1, nums.min()), min(pair1[1] + 1, nums.max()))
    S = {}
    for n in set(pair1) | set(pair2):
        S[n] = slice_lnW(d, n)
    mm = np.memmap(fn, dtype=np.float64, mode='r', offset=t['data_off'], shape=(t['nmb'], t['dsize'] // 8))
    x1min, x2min, x3min, x1max, x2max, x3max = t['size'][:6]
    w = np.array([x1max - x1min, x2max - x2min, x3max - x3min]); lo = np.array([x1min, x2min, x3min])
    lev = t['ll'][:, 3]
    cols = {k: [] for k in ('rho', 'W', 'alpha', 'K', 'divc', 'cov', 'lap', 'advW', 'advR', 'ratio', 'dt1', 'dt2',
                            'rt1', 'rt2')}
    nmiss = 0
    for b in range(t['nmb']):
        ext = w / 2.0**lev[b]
        bmin = lo + t['ll'][b, :3] * ext
        if abs(bmin[2]) > 1e-9:
            continue
        o = block(mm[b], 1, ext[0] / 16, p)
        if o is None:
            continue
        key = (round(bmin[0], 6), round(bmin[1], 6), round(bmin[0] + ext[0], 6), round(bmin[1] + ext[1], 6))
        dn = o['dense']
        dts = []
        for pr in (pair1, pair2):
            if all(key in S[n][0] for n in pr):
                dts.append([(S[pr[1]][0][key][q] - S[pr[0]][0][key][q]) / (S[pr[1]][1] - S[pr[0]][1])
                            for q in (0, 1)])
            else:
                dts.append([np.full((16, 16), np.nan)] * 2)
        if np.isnan(dts[0][0]).all():
            nmiss += dn.sum()
        for kk in cols:
            if kk in ('dt1', 'dt2', 'rt1', 'rt2'):
                cols[kk].append(dts[0 if kk[-1] == '1' else 1][0 if kk[0] == 'd' else 1][dn])
            else:
                cols[kk].append(o[kk][dn])
    c = {k: np.concatenate(v) if v else np.zeros(0) for k, v in cols.items()}
    tsp = (S[pair1[1]][1] - S[pair1[0]][1], S[pair2[1]][1] - S[pair2[0]][1])
    return lab, tt, p, c, nmiss, tsp, (S[pair1[0]][1], S[pair1[1]][1])


def theta_cont(c, which='rt1'):
    """continuity: theta = -d ln rho/d tau = -(W/alpha)(d_t + V.grad) ln rho (exact in the continuum)"""
    return -c['W'] / c['alpha'] * (c[which] + c['advR'])


def thetas(c, which='dt1'):
    sp = c['W'] * (c['divc'] + c['cov'] + c['lap']) - c['W'] * c['K']
    dlnW = c[which] + c['advW']
    return sp + c['W'] / c['alpha'] * dlnW, sp + dlnW / c['alpha']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{C}/plots/phase2_m1')
    ap.add_argument('--nproc', type=int, default=13)
    a = ap.parse_args()
    jobs = [(lab, f) for lab, d in RUNS for f in sorted(glob.glob(f'{C}/{d}/rst/*.rst'))[1:]]
    with Pool(a.nproc) as pool:
        res = pool.map(one, jobs)
    rows = ['R-064 step 1: theta = W(D_i v^i + v.grad ln alpha) + (W/alpha)(d_t + V.grad) ln W - W K vs the coord / '
            'covariant divergence gates (c050_theta.py). z = 0 layer of each restart, dense cells (D > rho_switch).',
            'theta_E = R-064 form, (1/alpha) on the d ln W term. dt1 / dt2 = d_t ln W from the bracketing dumps / one dump '
            'further on each side. cont = theta from continuity, -(W/alpha)(d_t + V.grad) ln rho (independent check).']
    rows.append('')
    rows.append('== Per restart: term sizes (medians over dense cells), validation')
    agg = {}
    for lab, tt, p, c, nmiss, tsp, tb in res:
        good = np.isfinite(c['dt1'])
        th, thE = thetas(c, 'dt1')
        th2, _ = thetas(c, 'dt2')
        tcn = theta_cont(c)
        bulk = good & (c['rho'] > 0.1 * c['rho'].max()) & (c['ratio'] <= 1)   # unflagged-by-sensor bulk star
        shk = good & (c['ratio'] > 2)
        med = lambda x, m: np.median(np.abs(x[m])) if m.any() else np.nan
        rows.append(f'{lab} t {tt:6.1f}: n_dense {good.size} (no slice match {nmiss}); dumps {tb[0]:.2f}/{tb[1]:.2f} '
                    f'(dt1 spacing {tsp[0]:.2f}, dt2 {tsp[1]:.2f})')
        rows.append(f'   dense medians |.|: d_i v^i {med(c["divc"], good):.2e}  v.grad ln sg {med(c["cov"], good):.2e}  '
                    f'v.grad ln a {med(c["lap"], good):.2e}  d_t lnW {med(c["dt1"], good):.2e}  V.grad lnW {med(c["advW"], good):.2e}  '
                    f'WK {med(c["W"] * c["K"], good):.2e}  theta {med(th, good):.2e}')
        if bulk.any():
            rows.append(f'   bulk star (rho > 0.1 rho_max, sensor ratio <= 1; {bulk.sum()} cells): median |theta| {med(th, bulk):.2e} '
                        f'vs |d_i v^i| {med(c["divc"], bulk):.2e}, |v.grad ln sg| {med(c["cov"], bulk):.2e}, |D_i v^i| '
                        f'{med(c["divc"] + c["cov"], bulk):.2e}; |theta(dt2) - theta(dt1)| {med(th2 - th, bulk):.2e}')
        if bulk.sum() > 2:
            rows.append(f'   continuity check, bulk: theta_cont = -(W/alpha) d ln rho: median |theta_cont| {med(tcn, bulk):.2e}, '
                        f'median |theta - theta_cont| {med(th - tcn, bulk):.2e}, corr {np.corrcoef(th[bulk], tcn[bulk])[0, 1]:.3f}, '
                        f'mean theta {np.mean(th[bulk]):+.2e} vs theta_cont {np.mean(tcn[bulk]):+.2e}')
        gt = good & (c['ratio'] > 0.5)
        if gt.any():
            rows.append(f'   gated cells (sensor ratio > 0.5; {gt.sum()}): sign(theta) == sign(theta_cont) in '
                        f'{np.mean((th[gt] <= 0) == (tcn[gt] <= 0)):.3f}; median |theta| {med(th, gt):.2e}, '
                        f'|theta - theta_cont| {med(th - tcn, gt):.2e}')
        if shk.any():
            rows.append(f'   shocks (sensor ratio > 2; {shk.sum()} cells): theta < 0 in {np.mean(th[shk] < 0):.3f}, median theta '
                        f'{np.median(th[shk]):.2e}; coord div < 0 in {np.mean(c["divc"][shk] <= 0):.3f}')
        for form, T in (('W', th), ('E', thE), ('W_dt2', th2), ('cont', tcn)):
            sc = c['divc'] <= 0; sv = c['divc'] + c['cov'] <= 0; st = T <= 0
            diff = good & (sc != sv); gate = good & (c['ratio'] > 0.5)
            A = agg.setdefault((lab, form), {})
            for ph in PHASES:
                if ph[0] <= tt < ph[1]:
                    z = A.setdefault(ph, np.zeros(6))
                    z += [(diff & (sv == st)).sum(), diff.sum(), (gate & (sc != st)).sum(), gate.sum(),
                          (good & (st != sv)).sum(), good.sum()]
    rows.append('')
    rows.append('== Gates per window: (a) cov agrees with theta where coord != cov; (b) coord disagrees with theta where sensor '
                'ratio > 0.5; (c) theta != cov over all dense cells. [counts]')
    res_rule = {}
    for form in ('W', 'E', 'W_dt2', 'cont'):
        rows.append(f' theta form {form}:')
        for lab, _ in RUNS:
            for ph in PHASES:
                z = agg.get((lab, form), {}).get(ph)
                if z is None:
                    continue
                fa, fb, fc = z[0] / max(z[1], 1), z[2] / max(z[3], 1), z[4] / max(z[5], 1)
                res_rule[(form, lab, ph)] = (fa, fb, fc)
                rows.append(f'   {lab} t {ph[0]:g}-{ph[1]:.0f}: (a) {fa:.3f} [{int(z[0])}/{int(z[1])}]  (b) {fb:.3f} '
                            f'[{int(z[2])}/{int(z[3])}]  (c) {fc:.3f} [{int(z[4])}/{int(z[5])}]')
    rows.append('')
    for form in ('W', 'E', 'W_dt2', 'cont'):
        pre = [res_rule[(form, lab, ph)] for lab, _ in RUNS for ph in PRE if (form, lab, ph) in res_rule]
        stop = any(x[2] > 0.10 for x in pre)
        go = all(x[0] >= 0.90 for x in pre) and any(x[1] >= 0.10 for x in pre)
        rows.append(f'R-064 rule (theta form {form}): ' + ('STOP ((c) > 0.10 in a pre-periapsis window)' if stop else
                                                          ('GO' if go else 'NO-GO')))
    print('\n'.join(rows))
    with open(f'{a.out}/c050_theta.txt', 'w') as f:
        f.write('\n'.join(rows) + '\n')


if __name__ == '__main__':
    main()
