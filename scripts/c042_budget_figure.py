#!/usr/bin/env python3
"""Methods figure (R-047 item 1): the comoving internal-energy budget and the slope-1/3 law.

(a) per-step budget of W rho eps on entropy-branch dense cells (flux, sources, metric change at
    fixed U, their sum, and the measured K_en/K_tr - 1) for solve_interval = 0 and 1 (C-042);
(b) K_en/K_tr - 1 against the metric term per cell: slope 1/3 at si = 1, 0 at si = 0;
(c) per C2P stage (R-047 item 2; S2_stage_si1, <mhd> dual_energy_diag slots de_ken_s1..s3):
    K_en/K_tr - 1 against the step's metric term, one slope per stage, against the SSP-RK3
    toy model (0, -1/2, +1/3; R-045 §2).

    c042_budget_figure.py [--out DIR]
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
import c041_budget_close as C                     # noqa: E402
from tde_ensemble_plots import bc                 # noqa: E402

RUN = C.RUN
B = C.B


def cells(d, extra=()):
    """per-cell metric term (per step) and diag slots for each adm pair of run d"""
    d = f'{RUN}/{d}'
    adm = sorted(glob.glob(f'{d}/bin/{B}.adm_xy.*.bin'))
    pf = sorted(glob.glob(f'{d}/bin/{B}.prim_xy.*.bin'))
    df = sorted(glob.glob(f'{d}/bin/{B}.dediag_xy.*.bin'))
    out = {k: [] for k in ('mt', 'fl', 'sr', 'ken') + tuple(extra)}
    for fa, fb in zip(adm[:-1], adm[1:]):
        qa, qb = bc.read_binary(fa), bc.read_binary(fb)
        g0 = np.asarray(qa['mb_geometry'])
        if not np.array_equal(g0, np.asarray(qb['mb_geometry'])):
            continue
        tm = 0.5 * (qa['time'] + qb['time'])
        qp, qd = bc.read_binary(C.nearest(pf, tm)), bc.read_binary(C.nearest(df, tm))
        if not (np.array_equal(np.asarray(qp['mb_geometry']), g0) and
                np.array_equal(np.asarray(qd['mb_geometry']), g0)):
            continue
        md, dd = qp['mb_data'], qd['mb_data']
        br = np.asarray(dd['de_branch'])[:, 0]
        sel = (br > 0.5) & (br < 2.5)
        rho = np.asarray(md['dens'])[:, 0][sel]
        P = np.asarray(md['press'])[:, 0][sel]
        u = np.stack([np.asarray(md[c])[:, 0][sel] for c in ('velx', 'vely', 'velz')], -1)
        ga, gb = C.metric(qa)[sel], C.metric(qb)[sel]
        Dt, St, tt = C.cons(rho, P, u, ga)
        ea, _, _ = C.c2p(Dt, St, tt, ga, P)
        eb, _, _ = C.c2p(Dt, St, tt, gb, P)
        dt = C.dt_at(d, tm)
        out['mt'].append((eb - ea) / ea * dt / (qb['time'] - qa['time']))
        out['fl'].append(dt * np.asarray(dd['de_eint_flux'])[:, 0][sel])
        out['sr'].append(dt * np.asarray(dd['de_eint_src'])[:, 0][sel])
        out['ken'].append(np.asarray(dd['de_ken_ktr'])[:, 0][sel])
        for k in extra:
            out[k].append(np.asarray(dd[k])[:, 0][sel])
    return {k: np.concatenate(v) for k, v in out.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{RUN}/plots/phase2_m1')
    a = ap.parse_args()
    rows = []
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)
    runs = {'solve_interval = 1': ('B1_comov_eint', 'C3'), 'solve_interval = 0': ('T3_si0', 'C0')}
    terms = ('flux', 'sources', 'metric', 'sum', 'measured')
    w = 0.38
    for n, (lab, (d, col)) in enumerate(runs.items()):
        c = cells(d)
        vals = [c['fl'].mean(), c['sr'].mean(), c['mt'].mean(), (c['fl'] + c['sr'] + c['mt']).mean(),
                c['ken'].mean()]
        ax[0].bar(np.arange(5) + (n - 0.5) * w, vals, w, color=col, label=lab)
        sl, ic = np.polyfit(c['mt'], c['ken'], 1)
        ax[1].plot(c['mt'], c['ken'], '.', ms=3, color=col, alpha=0.5,
                   label=f'{lab}: slope {sl:.3f}, r = {np.corrcoef(c["mt"], c["ken"])[0, 1]:.3f}')
        rows.append(f'{lab}: ' + ', '.join(f'{t} {v:+.4f}' for t, v in zip(terms, vals))
                    + f'; slope {sl:.4f} intercept {ic:+.5f} ({c["mt"].size} cells)')
    ax[0].set_xticks(range(5)); ax[0].set_xticklabels(['flux', 'sources', 'metric change\n(U fixed)',
                                                       'sum', 'measured\nK_en/K_tr − 1'])
    ax[0].axhline(0, color='k', lw=0.6); ax[0].legend()
    ax[0].set_title('(a) per-step change of W ρ ε, fraction (mean, dense cells)')
    x = np.linspace(0, 1.2, 10)
    ax[1].plot(x, x / 3, 'k--', lw=1, label='slope 1/3')
    ax[1].set_xlabel('metric term per step (U fixed)'); ax[1].set_ylabel('K_en/K_tr − 1')
    ax[1].set_title('(b) energy-solution excess vs the metric change'); ax[1].legend(fontsize=8)
    # (c) per stage
    if glob.glob(f'{RUN}/S2_stage_si1/bin/{B}.dediag_xy.*.bin'):
        c = cells('S2_stage_si1', extra=('de_ken_s1', 'de_ken_s2', 'de_ken_s3'))
        # R-045's toy (metric jump at stage 1) vs the code's order (solve at the LAST stage,
        # stages 1-2 on the old metric, entropy resync at every stage): -1, -1/4, +1/3
        pred = {1: 0.0, 2: -0.5, 3: 1.0 / 3.0}
        code = {1: -1.0, 2: -0.25, 3: 1.0 / 3.0}
        for s, col in zip((1, 2, 3), ('C2', 'C1', 'C3')):
            y = c[f'de_ken_s{s}']
            ok = y != 0.0                     # slot not written at this stage (not dense / bad)
            sl, ic = np.polyfit(c['mt'][ok], y[ok], 1)
            ax[2].plot(c['mt'][ok], y[ok], '.', ms=3, color=col, alpha=0.5,
                       label=f'stage {s}: slope {sl:+.3f} (solve-last {code[s]:+.3f}; R-045 {pred[s]:+.3f})')
            ax[2].plot(x, code[s] * x, '-', color=col, lw=0.8)
            ax[2].plot(x, pred[s] * x, ':', color=col, lw=1)
            rows.append(f'per-stage (S2, si = 1) stage {s}: slope {sl:+.4f} intercept {ic:+.5f} '
                        f'r {np.corrcoef(c["mt"][ok], y[ok])[0, 1]:.3f} ({ok.sum()}/{ok.size} cells written); '
                        f'solve-last model {code[s]:+.4f}, R-045 toy {pred[s]:+.4f}')
        ax[2].legend(fontsize=8)
    ax[2].set_xlabel('metric term per step (U fixed)'); ax[2].set_ylabel('K_en/K_tr − 1 at the stage C2P')
    ax[2].set_title('(c) per C2P stage, si = 1 (solid: solve at last stage; dotted: R-045)')
    for x_ in ax:
        x_.grid(alpha=0.3)
    fig.suptitle('Comoving frame: once-per-step metric solves and the dual-energy excess (item 6 L4, t ≈ 61-68)')
    fig.savefig(f'{a.out}/c042_budget_figure.png', dpi=90)
    with open(f'{a.out}/c042_budget_figure.txt', 'w') as f:
        f.write('\n'.join(rows) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
