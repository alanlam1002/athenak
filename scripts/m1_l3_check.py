#!/usr/bin/env python3
"""Phase 2 M2 groundwork (R-040 2b): is a BH box at L3 acceptable where the star lives?

From the M1 test-1b vacuum runs (xidot = 0.45, t <= 20), BH-centred bins r_BH in [8,12),
[12,20), [20,40):
  - L3 - L4 pointwise, on cells of MeshBlocks with identical geometry in both dumps (same
    level, same cell centres; the coverage fraction is reported). The runs' dump times differ
    by up to ~0.06 M (the BH moves 0.03 M, worth ~2e-4 in psi at 8 M), so the difference is
    taken of each run's error against the analytic trumpet at its own time and x_BH:
    (f_L3 - f_an(t3)) - (f_L4 - f_an(t4));
  - each run against the analytic static trumpet (m1_vacuum.analytic), on all its cells;
for alpha and psi (psi = psi4^(1/4); delta psi = delta psi4 / (4 psi^3)).
The comparator: the star's own self-field, delta psi_* = psi - psi_trumpet and
delta alpha_* at the density maximum of Q1 (BH frame, rp10) at a few times.

    m1_l3_check.py [--out DIR]
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m1_vacuum as MV                                # noqa: E402
from m1_vacuum import bc                              # noqa: E402

C = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc'
B = 'cfc_tde_wd_imbh_parabolic_tracker'
BINS = ((8.0, 12.0), (12.0, 20.0), (20.0, 40.0))
PAIRS = {'no skip': ('M1_1b_L3_skipfalse', 'M1_1b_L4_skipfalse'),
         'skip': ('M1_1b_L3_skiptrue', 'M1_1a_xd045b')}
X0, V = 45.0, 0.45


def load(f):
    p = bc.read_binary(f)
    g = np.asarray(p['mb_geometry']); nx = p['nx1_out_mb']
    md = {k: np.asarray(q)[:, 0] for k, q in p['mb_data'].items()}
    return p['time'], g, nx, md


def cells(g, nx):
    dx = ((g[:, 1] - g[:, 0]) / nx)[:, None, None]
    X = g[:, 0, None, None] + (np.arange(nx)[None, None, :] + .5) * dx
    Y = g[:, 2, None, None] + (np.arange(nx)[None, :, None] + .5) * dx
    return np.broadcast_to(X, (len(g), nx, nx)), np.broadcast_to(Y, (len(g), nx, nx))


def vs_analytic(g, nx, md, xb):
    X, Y = cells(g, nx)
    r = np.hypot(X - xb, Y)
    out = {}
    for a, b in BINS:
        s = (r >= a) & (r < b)
        p4a, ala, _, _ = MV.analytic(X[s] - xb, Y[s])
        psi = md['adm_psi4'][s]**0.25
        out[(a, b)] = (np.abs(md['adm_alpha'][s] - ala).max(), np.abs(psi - p4a**0.25).max())
    return out


def errfield(g, nx, md, xb):
    X, Y = cells(g, nx)
    p4a, ala, _, _ = MV.analytic(X - xb, Y)
    return md['adm_alpha'] - ala, md['adm_psi4']**0.25 - p4a**0.25


def pointwise(A, Bd, xa, xbb):
    """(L3 - an) - (L4 - an) on blocks with identical geometry; bins about L3's x_BH"""
    (gA, nx, mA), (gB, _, mB) = A, Bd
    eaA, epA = errfield(gA, nx, mA, xa); eaB, epB = errfield(gB, nx, mB, xbb)
    key = {tuple(np.round(r, 6)): i for i, r in enumerate(gB)}
    pa, pb = [], []
    for i, r in enumerate(gA):
        j = key.get(tuple(np.round(r, 6)))
        if j is not None:
            pa.append(i); pb.append(j)
    pa, pb = np.array(pa), np.array(pb)
    X, Y = cells(gA, nx)
    r = np.hypot(X - xa, Y)
    out = {}
    for a, b in BINS:
        inbin = (r >= a) & (r < b)
        s = np.zeros_like(inbin); s[pa] = inbin[pa]
        da = np.abs(eaA[pa] - eaB[pb])[s[pa]]
        dp = np.abs(epA[pa] - epB[pb])[s[pa]]
        out[(a, b)] = (da.max() if da.size else np.nan, da.mean() if da.size else np.nan,
                       dp.max() if dp.size else np.nan, dp.mean() if dp.size else np.nan,
                       s.sum() / max(inbin.sum(), 1))
    return out


def star_field(times=(0, 60, 120, 135)):
    rows = []
    d = f'{C}/Q1_V5L4_fix_t300/bin'
    for t in times:
        tA, g, nx, md = load(f'{d}/{B}.adm_xy.{t:05d}.bin')
        _, gp, _, mp = load(f'{d}/{B}.prim_xy.{t:05d}.bin')
        m, j, i = np.unravel_index(np.argmax(mp['dens']), mp['dens'].shape)
        X, Y = cells(g, nx)
        x, y = X[m, j, i], Y[m, j, i]
        p4a, ala, _, _ = MV.analytic(np.array([x]), np.array([y]))
        psi = md['adm_psi4'][m, j, i]**0.25
        rows.append((tA, np.hypot(x, y), psi - p4a[0]**0.25, md['adm_alpha'][m, j, i] - ala[0]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{C}/plots/phase2_m1')
    a = ap.parse_args()
    star = star_field()
    for t, r, dp, da in star:
        print(f'star (Q1) t={t:5.1f} r_BH={r:5.2f}: delta psi_* = {dp:+.3e}, delta alpha_* = {da:+.3e}')
    res = {}
    for lab, (l3, l4) in PAIRS.items():
        f3 = sorted(glob.glob(f'{C}/{l3}/bin/{B}.adm_xy.*.bin'))
        f4 = {int(f.split('.')[-2]): f for f in glob.glob(f'{C}/{l4}/bin/{B}.adm_xy.*.bin')}
        res[lab] = []
        for f in f3:
            n = int(f.split('.')[-2])
            if n not in f4 or n % 2:
                continue
            A = load(f); Bd = load(f4[n])
            if abs(A[0] - Bd[0]) > 0.1:                    # e.g. L4's last dump is t = 19.38
                continue
            xb, x4 = X0 - V * A[0], X0 - V * Bd[0]
            pw = pointwise(A[1:], Bd[1:], xb, x4)
            e3 = vs_analytic(*A[1:], xb); e4 = vs_analytic(*Bd[1:], x4)
            res[lab].append((A[0], pw, e3, e4))
            print(f'{lab:7s} t={A[0]:5.1f} ' + '  '.join(
                f'[{a:.0f},{b:.0f}) L3-L4 |dpsi| max {pw[(a, b)][2]:.1e} mean {pw[(a, b)][3]:.1e} |dalpha| max {pw[(a, b)][0]:.1e}'
                f' (cov {pw[(a, b)][4]:.2f}); vs an |dpsi| L3 {e3[(a, b)][1]:.1e} L4 {e4[(a, b)][1]:.1e}'
                for a, b in BINS), flush=True)
    # figure: per bin, max |L3-L4| and |L3-an|, |L4-an| in psi vs t, star delta psi as band
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    cols = ('#1b9e77', '#d95f02', '#7570b3')
    dps = [abs(s[2]) for s in star]; das = [abs(s[3]) for s in star]
    for k, (q, nm) in enumerate(((2, 'psi'), (0, 'alpha'))):
        a_ = ax[k]
        for c, bn in zip(cols, BINS):
            rows = res['no skip']
            t = [r[0] for r in rows]
            a_.semilogy(t, [r[1][bn][q] for r in rows], 'o-', color=c, label=f'|L3 − L4| max, r_BH ∈ [{bn[0]:.0f},{bn[1]:.0f})')
            j = 1 if nm == 'psi' else 0
            a_.semilogy(t, [r[2][bn][j] for r in rows], '--', color=c, alpha=0.7, label=f'|L3 − trumpet| max')
            a_.semilogy(t, [r[3][bn][j] for r in rows], ':', color=c, alpha=0.7, label=f'|L4 − trumpet| max')
        band = dps if nm == 'psi' else das
        a_.axhspan(min(band), max(band), color='0.85', label=f'star self-field |δ{nm}_*| (Q1, at ρ_max, t=0–135)')
        a_.set_xlabel('t [M]'); a_.set_title(f'{nm}: BH box L3 vs L4 (ξ̇ = 0.45, no skip), ≥ 8 M from the BH')
        a_.grid(alpha=0.3)
    ax[0].legend(fontsize=7, ncol=2)
    fig.savefig(f'{a.out}/m2_L3_vs_L4.png', dpi=100)


if __name__ == '__main__':
    main()
