#!/usr/bin/env python3
"""R-044 §1: close the C-041 internal-energy budget with the metric-change term (offline).

C-041's budget (dual_energy_diag slots 8/9) gives, per step and as a fraction of W rho eps,
the internal-energy change carried by the flux divergence and by the source terms. In the
comoving frame the remainder is the metric changing under gas at rest on the grid:

    metric term = [ (W rho eps)(U, g(t_b)) - (W rho eps)(U, g(t_a)) ] / (W rho eps)_a
                  * dt / (t_b - t_a)

at FIXED densitized conserved variables U = sqrt(gamma) (D, S_i, tau), built from a prim dump
and the metric at t_a and re-inverted (ideal-gas C2P, Gamma = 5/3) with the metric at t_b.
The metric comes from consecutive adm_xy dumps (1 M apart, conformally flat: gamma_ij from
adm_g**), so this is the step-averaged rate, linear in dt. Expect flux + sources + metric ~
the measured excess K_en/K_tr - 1: ~0 with solve_interval = 0, +8% with 1.

Entropy-branch dense cells (branch 1, 2) of the xy plane, per adm pair, for T3_si0 (si = 0)
and B1_comov_eint (si = 1), both restarts of item 6 L4 at t = 60.7.

    c041_budget_close.py [--out DIR]
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tde_t11_surface_diag as T                  # noqa: E402,F401 (sets up bc)
from tde_ensemble_plots import bc                 # noqa: E402

RUN = T.RUN
B = 'cfc_tde_wd_imbh_parabolic_tracker'
GAM = 5.0 / 3.0
ARMS = {'si = 0 (T3_si0)': 'T3_si0', 'si = 1 (B1)': 'B1_comov_eint'}
G = ('gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz')


def metric(q):
    md = q['mb_data']
    g = np.stack([np.asarray(md[f'adm_{c}'])[:, 0] for c in G], axis=-1)   # (mb, j, i, 6)
    gm = np.empty(g.shape[:-1] + (3, 3))
    for (a, b), n in zip(((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)), range(6)):
        gm[..., a, b] = gm[..., b, a] = g[..., n]
    return gm


def cons(rho, P, u, gm):
    """densitized (D, S_i, tau) from prims (rho, P, u^i = W v^i) and the 3-metric"""
    usq = np.einsum('...i,...ij,...j->...', u, gm, u)
    W = np.sqrt(1.0 + usq)
    h = 1.0 + GAM / (GAM - 1.0) * P / rho
    sg = np.sqrt(np.linalg.det(gm))
    D = rho * W
    S = (rho * h * W)[..., None] * np.einsum('...ij,...j->...i', gm, u)
    tau = rho * h * W**2 - P - D
    return sg * D, sg[..., None] * S, sg * tau


def c2p(Dt, St, taut, gm, P0):
    """ideal-gas C2P at metric gm from densitized conserved; returns W rho eps"""
    sg = np.sqrt(np.linalg.det(gm))
    D, S, tau = Dt / sg, St / sg[..., None], taut / sg
    gi = np.linalg.inv(gm)
    S2 = np.einsum('...i,...ij,...j->...', S, gi, S)
    E = tau + D

    def res(P):
        z = E + P
        v2 = S2 / z**2
        W = 1.0 / np.sqrt(1.0 - v2)
        rho = D / W
        # eps from z = rho h W^2 = W^2 (rho + Gam/(Gam-1) P): solve the ideal-gas relation
        return P - (GAM - 1.0) / GAM * (z / W**2 - rho), W, rho

    P = P0.copy()                                     # the dump's P: the metric change is small
    for _ in range(60):
        r, W, rho = res(P)
        dP = 1e-7 * np.abs(P) + 1e-40
        r2, _, _ = res(P + dP)
        step = r / ((r2 - r) / dP)
        P = np.maximum(P - step, 1e-6 * P)
        if np.nanmax(np.abs(step) / P) < 1e-14:
            break
    _, W, rho = res(P)
    return W * P / (GAM - 1.0), W, rho


def dt_at(d, t):
    r = np.array([(float(m.group(1)), float(m.group(2))) for m in
                  (re.search(r'time=([-+.\deE]+) dt=([-+.\deE]+)', l) for l in open(f'{d}/ensemble.out')) if m])
    return float(np.interp(t, r[:, 0], r[:, 1]))


def nearest(files, t):
    ts = [bc.read_binary(f)['time'] for f in files]
    return files[int(np.argmin(np.abs(np.array(ts) - t)))]


def pct(x):
    return f'{np.mean(x):+.3e} [{np.percentile(x, 1):+.2e}, {np.percentile(x, 99):+.2e}]'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{RUN}/plots/phase2_m1')
    a = ap.parse_args()
    rows = []
    for lab, d in ARMS.items():
        d = f'{RUN}/{d}'
        adm = sorted(glob.glob(f'{d}/bin/{B}.adm_xy.*.bin'))
        pf = sorted(glob.glob(f'{d}/bin/{B}.prim_xy.*.bin'))
        df = sorted(glob.glob(f'{d}/bin/{B}.dediag_xy.*.bin'))
        acc = {k: [] for k in ('fl', 'sr', 'mt', 'sum', 'ken', 'mtW')}
        for fa, fb in zip(adm[:-1], adm[1:]):
            qa, qb = bc.read_binary(fa), bc.read_binary(fb)
            ta, tb = qa['time'], qb['time']
            if not np.array_equal(np.asarray(qa['mb_geometry']), np.asarray(qb['mb_geometry'])):
                continue                                  # a regrid between the two
            tm = 0.5 * (ta + tb)
            qp, qd = bc.read_binary(nearest(pf, tm)), bc.read_binary(nearest(df, tm))
            if not (np.array_equal(np.asarray(qp['mb_geometry']), np.asarray(qa['mb_geometry'])) and
                    np.array_equal(np.asarray(qd['mb_geometry']), np.asarray(qa['mb_geometry']))):
                continue
            md, dd = qp['mb_data'], qd['mb_data']
            br = np.asarray(dd['de_branch'])[:, 0]
            sel = (br > 0.5) & (br < 2.5)
            rho = np.asarray(md['dens'])[:, 0][sel]
            P = np.asarray(md['press'])[:, 0][sel]
            u = np.stack([np.asarray(md[c])[:, 0][sel] for c in ('velx', 'vely', 'velz')], -1)
            ga, gb = metric(qa)[sel], metric(qb)[sel]
            Dt, St, taut = cons(rho, P, u, ga)
            ea, Wa, ra = c2p(Dt, St, taut, ga, P)
            eb, Wb, rb = c2p(Dt, St, taut, gb, P)
            dt = dt_at(d, tm)
            mt = (eb - ea) / ea * dt / (tb - ta)
            # check: the C2P round trip at g_a reproduces the dump's P
            rt = np.abs(ea * (GAM - 1.0) / Wa / P - 1)
            fl = dt * np.asarray(dd['de_eint_flux'])[:, 0][sel]
            sr = dt * np.asarray(dd['de_eint_src'])[:, 0][sel]
            ken = np.asarray(dd['de_ken_ktr'])[:, 0][sel]
            for k, v in (('fl', fl), ('sr', sr), ('mt', mt), ('sum', fl + sr + mt), ('ken', ken)):
                acc[k].append(v)
            rows.append(f'  {lab} adm {ta:.2f}->{tb:.2f}, prim/diag at {qp["time"]:.2f}/{qd["time"]:.2f}, '
                        f'dt {dt:.4f}, {sel.sum()} cells, C2P round-trip max |dP/P| {rt.max():.1e}')
        if not acc['fl']:
            continue
        rows.append(f'{lab}: per-step fractional change of W rho eps, entropy-branch dense cells, '
                    f'mean [p1, p99]:')
        for k, nm in (('fl', 'flux divergence'), ('sr', 'sources'), ('mt', 'metric change (U fixed)'),
                      ('sum', 'flux + sources + metric'), ('ken', 'measured K_en/K_tr - 1')):
            rows.append(f'   {nm:26s} {pct(np.concatenate(acc[k]))}')
        s, k = np.concatenate(acc['sum']), np.concatenate(acc['ken'])
        rows.append(f'   corr(sum, measured excess) = {np.corrcoef(s, k)[0, 1]:.3f}; '
                    f'corr(metric, -sources) = {np.corrcoef(np.concatenate(acc["mt"]), -np.concatenate(acc["sr"]))[0, 1]:.3f}')
    with open(f'{a.out}/c041_budget_close.txt', 'w') as f:
        f.write('\n'.join(rows) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
