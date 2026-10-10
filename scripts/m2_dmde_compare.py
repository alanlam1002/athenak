#!/usr/bin/env python3
"""M2 (research R-034): T-8 dM/dE on V5-L4 vs V5-L5 (and Q1, the fixed-binary L4 rerun).

dM/dE at each restart time (scripts/tde_dmde.py), its dump-to-dump stability (R-002), the
bound fraction, and the L4-L5 difference over the bulk of the distribution: the bins
holding the central 90% of the mass, and where in E/dE the difference sits.

    m2_dmde_compare.py [--q1] [--out DIR]
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
import tde_dmde as T   # noqa: E402
import rst_tree as R    # noqa: E402

C = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc'
RUNS = {
    'L4 (V5, old binary)': (f'{C}/de_V5_rp10_L4', '#e6ab02'),
    'L5 (V5, fixed from t=159)': (f'{C}/de_V5_rp10_L5_mgfix', '#1b9e77'),
}
Q1 = ('L4 (Q1, fixed binary)', (f'{C}/Q1_V5L4_fix_t300', '#d95f02'))
DE = 0.677 / 16.4**2
EDGES = np.linspace(-4, 4, 81)


def runset(d, tmin):
    out = []
    for f in sorted(glob.glob(f'{d}/rst/*.rst')):
        t = R.read_tree(f)['time']
        if t >= tmin - 0.5:
            out.append(f)
    return out


def bulk_diff(h1, h2):
    """max and mass-weighted mean |h1 - h2| / h over the bins holding the central 90%"""
    c = np.cumsum(h1 * np.diff(EDGES)); c /= c[-1]
    sel = (c >= 0.05) & (c <= 0.95)
    ref = np.maximum(h1[sel], 1e-30)
    rel = np.abs(h1[sel] - h2[sel]) / ref
    return rel.max(), np.average(rel, weights=h1[sel]), sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--q1', action='store_true')
    ap.add_argument('--tmin', type=float, default=170.0)
    ap.add_argument('--out', default=f'{C}/plots/dual_energy')
    a = ap.parse_args()
    runs = dict(RUNS)
    if a.q1:
        runs[Q1[0]] = Q1[1]
    res = {}
    for lab, (d, col) in runs.items():
        res[lab] = []
        for f in runset(d, a.tmin):
            t, E, EB, dM = T.energies(f)
            res[lab].append(dict(t=t, H=T.hist(E, dM, DE, EDGES), HB=T.hist(EB, dM, DE, EDGES),
                                 fb=dM[E < 0].sum() / dM.sum(), fbB=dM[EB < 0].sum() / dM.sum(),
                                 M=dM.sum()))
            r = res[lab][-1]
            print(f'{lab:28s} t={t:6.1f} bound {r["fb"]:.4f} (Bernoulli {r["fbB"]:.4f}) M_in {r["M"]:.5e}',
                  flush=True)
    rows = []
    xc = 0.5 * (EDGES[1:] + EDGES[:-1])
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)
    for lab, (d, col) in runs.items():
        rs = res[lab]
        for i, r in enumerate(rs):
            ax[0].semilogy(xc, np.maximum(r['H'], 1e-6), color=col, lw=0.8 + 0.6 * (i == len(rs) - 1),
                           alpha=0.4 + 0.6 * (i == len(rs) - 1), label=f'{lab}, t={r["t"]:.0f}' if i == len(rs) - 1 else None)
        # dump-to-dump stability
        for r0, r1 in zip(rs[:-1], rs[1:]):
            mx, mean, _ = bulk_diff(r1['H'], r0['H'])
            rows.append(f'{lab}: t={r0["t"]:.0f}->{r1["t"]:.0f}: bulk |d(dM/dE)|/(dM/dE) max {mx:.3f} mean {mean:.3f}; '
                        f'bound {r0["fb"]:.4f}->{r1["fb"]:.4f}')
        ax[2].plot([r['t'] for r in rs], [r['fb'] for r in rs], 'o-', color=col, label=lab)
    # L4 vs L5 at the latest common time (t = 200)
    labs = list(runs)
    ref = labs[1]
    for lab in [l for l in labs if l != ref]:
        rA = min(res[lab], key=lambda r: abs(r['t'] - 200)); rB = min(res[ref], key=lambda r: abs(r['t'] - 200))
        mx, mean, sel = bulk_diff(rB['H'], rA['H'])
        rel = (rA['H'] - rB['H']) / np.maximum(rB['H'], 1e-30)
        worst = xc[sel][np.argmax(np.abs(rel[sel]))]
        rows.append(f'{lab} vs {ref} at t={rA["t"]:.0f}/{rB["t"]:.0f}: bulk (central 90% of mass) |diff| max '
                    f'{mx:.3f} mean {mean:.3f}, largest at E/dE={worst:+.2f}; bound {rA["fb"]:.4f} vs {rB["fb"]:.4f} '
                    f'(Bernoulli {rA["fbB"]:.4f} vs {rB["fbB"]:.4f})')
        ax[1].plot(xc[sel], rel[sel], 'o-', color=runs[lab][1], label=f'({lab} - L5)/L5')
    ax[0].set_xlabel('E / ΔE  (ΔE = R*/r_t²)'); ax[0].set_ylabel('(1/M) dM/d(E/ΔE)'); ax[0].legend(fontsize=8)
    ax[0].set_title('V5 (rp10, β=1.64): dM/dE, faint = earlier dumps')
    ax[1].axhspan(-0.1, 0.1, color='0.9'); ax[1].set_xlabel('E / ΔE'); ax[1].legend(fontsize=8)
    ax[1].set_title('relative difference vs L5 at t = 200 (central 90% of mass)')
    ax[2].set_xlabel('t [M]'); ax[2].set_title('bound fraction (E < 0)'); ax[2].legend(fontsize=8)
    for x in ax:
        x.grid(alpha=0.3)
    fig.savefig(f'{a.out}/M2_dmde.png', dpi=100)
    with open(f'{a.out}/M2_dmde.txt', 'w') as f:
        f.write('\n'.join(rows) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
