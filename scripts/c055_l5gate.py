#!/usr/bin/env python3
"""C-055 (R-066 item 4, PROVISIONAL per R-067 Q5): the L5 dual-energy gate pair, B - A at L5
against B - A at L4 (C-051 §7), and whether B(L4) or A(L4) lies closer to L5.

Arms (zone off, booking only, binary d21072a5):
  L5A = L5A_rp10_gate (coord), L5B = L5B_rp10_gate (covariant): L5 restart 00006 t = 108.84 -> 170;
  L4A = S2A_R6 (coord), L4B = S2B_R6 (covariant): R6 L4 restart 00006 t = 113.7 -> 170;
  L5c = L5c_rp10_comov_t200 (the original L5 run, coord by construction, older binary) as a
        same-physics cross-check of L5A.
Quantities as c051_s2.py §7/§9: 3D physics rows at the t = 170 restarts (dense K/K0, M3,
remnant, bound, <E>), the z = 0 slice series (dense K/K0, warm > 5% / > 10%) at matched
integer times, and the dediag_xy switch census (flagged, energy-branch fractions) before /
after periapsis (t < 136 / >= 136).

    c055_l5gate.py [--nproc 16]
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c051_s2 as S                               # noqa: E402
import rst_tree as RT                             # noqa: E402

C, P, B = S.C, S.P, S.B
ARM = {'L5A': 'L5A_rp10_gate', 'L5B': 'L5B_rp10_gate', 'L4A': 'S2A_R6', 'L4B': 'S2B_R6',
       'L5c': 'L5c_rp10_comov_t200'}


def rst_near(d, t):
    fs = sorted(glob.glob(f'{C}/{d}/rst/*.rst'))
    ts = np.array([RT.read_tree(f)['time'] for f in fs])
    i = int(np.argmin(np.abs(ts - t)))
    return fs[i], ts[i]


def series(d, nproc):
    fs = sorted(glob.glob(f'{C}/{d}/bin/{B}.prim_xy.*.bin'))
    with Pool(nproc) as p:
        return {round(t): (k, w5, w10) for t, k, w5, w10 in p.map(S.slice_K, fs)}


def census(d, nproc):
    ns = sorted(int(f.split('.')[-2]) for f in glob.glob(f'{C}/{d}/bin/{B}.dediag_xy.*.bin'))
    with Pool(nproc) as p:
        return np.array(p.starmap(S._census, [(f'{C}/{d}', n) for n in ns]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nproc', type=int, default=16)
    a = ap.parse_args()
    rows = ['C-055 (R-066 item 4): L5 dual-energy gate pair. PROVISIONAL (R-067 Q5): every arm carries the '
            'C-054 core flux leak (same-level face non-conservation, ghost metric/flag), which the fix may change.',
            'A = coord, B = covariant; zone off. L5 windows start t = 108.84, L4 at 113.7. dense = rho > 0.1 rho_c.']
    # 1. 3D at t = 170
    ph = {}
    for x, d in ARM.items():
        fn, t = rst_near(d, 170.0)
        ph[x] = S.physics(S.rst_pass(fn))
        p_ = ph[x]
        rows.append(f'  {x:4s} ({d}, t {t:.2f}): K/K0 {p_["K"]:.5f} M3 {p_["M3"]:.4f} (>10% {p_["M10"]:.4f}) '
                    f'<E> {p_["E"]:+.4f} dE bound {p_["bound"]:.5f} remnant {p_["rem"]:.4f} M*')
    rows.append('== B - A (3D, t = 170)')
    d3 = {}
    for lv in ('L5', 'L4'):
        u, v = ph[lv + 'B'], ph[lv + 'A']
        d3[lv] = dict(K=u['K'] / v['K'] - 1, M3=u['M3'] - v['M3'], rem=u['rem'] / v['rem'] - 1,
                      bound=u['bound'] - v['bound'], E=u['E'] - v['E'])
        rows.append(f'  {lv}: dK/K {d3[lv]["K"]:+.2e} dM3 {d3[lv]["M3"]:+.4f} drem/rem {d3[lv]["rem"]:+.4f} '
                    f'dbound {d3[lv]["bound"]:+.5f} d<E> {d3[lv]["E"]:+.4f} dE')
    rows.append(f'  ratio L5/L4: K {d3["L5"]["K"] / d3["L4"]["K"]:+.2f}, M3 {d3["L5"]["M3"] / d3["L4"]["M3"]:+.2f}, '
                f'rem {d3["L5"]["rem"] / d3["L4"]["rem"]:+.2f}')
    rows.append('== which L4 arm is closer to L5 (3D, t = 170): |L4x - L5x| for the same gate, and against both L5 arms')
    for q in ('K', 'M3', 'rem'):
        f = (lambda u, v: u / v - 1) if q in ('K', 'rem') else (lambda u, v: u - v)
        line = f'  {q:3s}:'
        for l4 in ('L4A', 'L4B'):
            line += f'  {l4}-L5A {f(ph[l4][q], ph["L5A"][q]):+.4f} {l4}-L5B {f(ph[l4][q], ph["L5B"][q]):+.4f} |'
        rows.append(line)
    rows.append(f'  L5A vs L5c (same gate, older binary): dK/K {ph["L5A"]["K"] / ph["L5c"]["K"] - 1:+.2e} '
                f'dM3 {ph["L5A"]["M3"] - ph["L5c"]["M3"]:+.4f} drem {ph["L5A"]["rem"] / ph["L5c"]["rem"] - 1:+.4f}')
    # 2. slice series
    ser = {x: series(d, a.nproc) for x, d in ARM.items()}
    ts = sorted(set(ser['L5A']) & set(ser['L5B']) & set(ser['L4A']) & set(ser['L4B']))
    rows.append(f'== slice series (z = 0 prim_xy), {len(ts)} matched integer times {ts[0]}-{ts[-1]}')
    arr = {x: np.array([ser[x][t] for t in ts]) for x in ('L5A', 'L5B', 'L4A', 'L4B')}
    for lv in ('L5', 'L4'):
        dk = arr[lv + 'B'][:, 0] / arr[lv + 'A'][:, 0] - 1
        dw = arr[lv + 'B'][:, 1] - arr[lv + 'A'][:, 1]
        i = int(np.argmax(np.abs(dk)))
        rows.append(f'  {lv} B-A: max |K_B/K_A - 1| {abs(dk[i]):.2e} (at t {ts[i]}, {dk[i]:+.2e}); final {dk[-1]:+.2e};'
                    f' mean {dk.mean():+.2e}; warm>5% B-A mean {dw.mean():+.4f}, max |.| {np.abs(dw).max():.4f}')
    for x in ('L4A', 'L4B'):
        dA = arr[x][:, 0] / arr['L5A'][:, 0] - 1
        dB = arr[x][:, 0] / arr['L5B'][:, 0] - 1
        rows.append(f'  slice K: {x} vs L5A rms {np.sqrt(np.mean(dA**2)):.2e} (final {dA[-1]:+.2e}); vs L5B rms '
                    f'{np.sqrt(np.mean(dB**2)):.2e} (final {dB[-1]:+.2e})')
    for t in ts[::max(1, len(ts) // 8)] + [ts[-1]]:
        rows.append(f'    t {t:4d}: K/K0 ' + ' '.join(f'{x} {ser[x][t][0]:.5f}' for x in ('L5A', 'L5B', 'L4A', 'L4B'))
                    + ';  warm>5% ' + ' '.join(f'{x} {ser[x][t][1]:.4f}' for x in ('L5A', 'L5B', 'L4A', 'L4B')))
    # 3. census
    rows.append('== switch census (dediag_xy; fractions of dense cells, mean over dumps)')
    cen = {x: census(ARM[x], a.nproc) for x in ('L5A', 'L5B', 'L4A', 'L4B')}
    for lo, hi in ((108.0, 136.0), (113.0, 136.0), (136.0, 170.1)):
        line = f'  t {lo:.0f}-{hi:.0f}:'
        for x in ('L5A', 'L5B', 'L4A', 'L4B'):
            A_ = cen[x]
            k = (A_[:, 0] >= lo) & (A_[:, 0] < hi) & (A_[:, 1] > 0)
            if not k.any():
                line += f' {x} -- |'; continue
            nd = A_[k, 1]
            line += f' {x} flagged {np.nanmean(A_[k, 18]):.4f} energy {np.mean((A_[k, 5] + A_[k, 6]) / nd):.4f} |'
        rows.append(line)
    txt = '\n'.join(rows) + '\n'
    open(f'{P}/c055_l5gate.txt', 'w').write(txt)
    print(txt)
    # figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    col = {'A': '#d95f02', 'B': '#1b9e77'}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for x in ('L5A', 'L5B', 'L4A', 'L4B'):
        t_ = sorted(ser[x]); k_ = [ser[x][t][0] for t in t_]
        ax[0].plot(t_, k_, color=col[x[-1]], ls='-' if x.startswith('L5') else '--', lw=1.6,
                   label=f'{x[:2]} {"coord" if x[-1] == "A" else "covariant"}')
        w_ = [ser[x][t][1] for t in t_]
        ax[1].plot(t_, w_, color=col[x[-1]], ls='-' if x.startswith('L5') else '--', lw=1.6)
    for lv, ls in (('L5', '-'), ('L4', '--')):
        ax[2].plot(ts, 100 * (arr[lv + 'B'][:, 0] / arr[lv + 'A'][:, 0] - 1), color='k', ls=ls, lw=1.6,
                   label=f'{lv} B/A - 1')
    ax[0].set_title('dense K/K0 (z = 0 slice)'); ax[1].set_title('warm (K/K0 > 1.05) fraction, dense')
    ax[2].set_title('gate effect K_B/K_A - 1 [%]'); ax[2].axhline(0, color='0.6', lw=0.8)
    for a_ in ax:
        a_.axvspan(136, 150, color='0.92', zorder=0); a_.set_xlabel('t [M]'); a_.grid(alpha=0.3)
    ax[0].legend(fontsize=9); ax[2].legend(fontsize=9)
    fig.suptitle('L5 vs L4 dual-energy gate pair (A coord, B covariant; zone off) -- PROVISIONAL (C-054 core flux leak)',
                 fontsize=11)
    fig.tight_layout()
    out = f'{P}/c055_l5gate.png'
    fig.savefig(out, dpi=110)
    os.system(f'cp {out} /home/tlam/athenak_tde_project/code/figures/')


if __name__ == '__main__':
    main()
