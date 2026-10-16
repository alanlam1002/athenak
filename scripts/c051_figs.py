#!/usr/bin/env python3
"""C-051 figures: the S2 restart tests (T = zone on + covariant, A = zone off + coord,
B = zone off + covariant). Reuses c051_s2.py's parsers and its restart cache (c051_cache).

  c051_dt.png    code dt(t) per arm, R6 / R7, window medians and the design targets
  c051_mass.png  grid rest-mass change, the booked (zone + excision) part, the unbooked rest
  c051_mdot.png  bound F(E), dM/dE and the frozen-in fallback Mdot at the final restarts
  c051_gate.png  dense K/K0 and warm (>5%) fractions on the z = 0 slices, R6 and R7

    c051_figs.py [--nproc 8] [--out DIR]
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c051_s2 as S                                # noqa: E402

COL = {'T': '#2a78d6', 'A': '#eb6834', 'B': '#1baf7a'}
LAB = {'T': 'T: zone + covariant', 'A': 'A: coord (control)', 'B': 'B: covariant (control)'}
INK, MUTED = '#1f1f1e', '#6b6a64'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.6, 'lines.linewidth': 1.3,
                     'legend.frameon': False})


def final_rst(r, x):
    return sorted(glob.glob(f'{S.C}/S2{x}_{r}/rst/*.rst'))[-1]


def fig_dt(L, out):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, (r, cfg) in zip(ax, S.RUNS.items()):
        for x in S.ARMS:
            d = L[(r, x)]['dt'][:-2]
            a.plot(d[:, 0], d[:, 1], color=COL[x], lw=0.8, label=LAB[x])
        for lo, hi in cfg['win']:
            med = {}
            for x in S.ARMS:
                d = L[(r, x)]['dt'][:-2]
                k = (d[:, 0] >= lo) & (d[:, 0] < hi)
                med[x] = np.median(d[k, 1])
                a.hlines(med[x], lo, hi, color=COL[x], lw=2.2)
            ref = min(med['A'], med['B'])
            a.hlines(cfg['gain'] * ref, lo, hi, color=INK, lw=1, ls='--')
            a.text(0.5 * (lo + hi), med['T'] * 1.04, f'x{med["T"] / med["A"]:.2f}', ha='center', va='bottom',
                   color=INK, fontsize=8)
        if r == 'R6':
            a.axvspan(136, 150, color='#eeede9', zorder=0)
            a.text(143, a.get_ylim()[1] * 0.97, 'periapsis', ha='center', va='top', color=MUTED, fontsize=8)
        a.set_title(f'{r} (r_p = {cfg["rp"]:g} M): code dt', loc='left', color=INK)
        a.set_xlabel('t [M]'); a.set_ylabel('dt [M]')
    ax[0].plot([], [], color=INK, ls='--', lw=1, label='target (x1.4 R6 / x2.0 R7) on the control median')
    fig.legend(*ax[0].get_legend_handles_labels(), loc='lower center', ncol=4, fontsize=7.5)
    fig.text(0.01, 0.005, 'thick bars: window medians; labels: T/A gain. Last two steps (clipped to tlim) dropped.',
             color=MUTED, fontsize=7)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(out, dpi=130); plt.close(fig)


def fig_mass(L, out):
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.2), sharex='col')
    for j, r in enumerate(S.RUNS):
        for x in S.ARMS:
            h = np.loadtxt(glob.glob(f'{S.C}/S2{x}_{r}/*.mhd.hst')[0])
            z = L[(r, x)]['z']
            dm = h[:, 2] - h[0, 2]
            book = np.interp(h[:, 0], z[:, 0], z[:, 9] + z[:, 6])          # excised + zone removed
            ax[0, j].plot(h[:, 0], dm, color=COL[x], label=LAB[x])
            ax[1, j].semilogy(h[:, 0], np.abs(dm + book), color=COL[x], label=LAB[x])
            ax[1, j].semilogy(h[:, 0], np.abs(book), color=COL[x], ls=':', lw=1.0)
        ax[0, j].set_title(f'{r}: grid rest mass M(t) - M(t0)', loc='left', color=INK)
        ax[1, j].set_title(f'{r}: |unbooked| (solid) vs |booked| (dotted)', loc='left', color=INK)
        ax[1, j].set_xlabel('t [M]')
        ax[0, j].set_ylabel('[M_BH units]'); ax[1, j].set_ylabel('|mass| [M_BH units]')
        ax[0, j].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax[1, j].set_ylim(1e-20, 1e-7)
    ax[0, 0].legend(fontsize=7.5)
    fig.text(0.01, 0.005, 'booked = excised + zone-removed rest mass (cum_dM_exc + cum_dM, ATMZONE, stage-weighted; A/B: excised only); '
             'unbooked = dM + booked. M* = 7.47e-5.', color=MUTED, fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(out, dpi=130); plt.close(fig)


def fig_mdot(Z, out):
    import c044_mdot as M
    fig, ax = plt.subplots(2, 3, figsize=(12, 6.4))
    tc = np.sqrt(M.TB[1:] * M.TB[:-1])
    for i, (r, cfg) in enumerate(S.RUNS.items()):
        tp = M.t_peri(cfg['src'], cfg['rp'])
        eb = np.linspace(-6, 6, 97)
        for x in S.ARMS:
            z = Z[(r, x)]; deb = ~z['rem'].astype(bool)
            E, dM = z['E'][deb] / S.DE, z['dM'][deb]
            b = E < 0
            o = np.argsort(E[b]); F = np.cumsum(dM[b][o]); Mb = F[-1]
            ax[i, 0].plot(E[b][o][::50], (F / Mb)[::50], color=COL[x], label=LAB[x])
            h, _ = np.histogram(E, eb, weights=dM)
            ax[i, 1].step(eb[:-1], h / np.diff(eb) / S.MSTAR, where='post', color=COL[x], lw=1.0)
            md = M.mdot(z['E'][deb], dM, tp)
            ax[i, 2].loglog(tc, np.where(md > 0, md, np.nan), color=COL[x], lw=1.1)
        ax[i, 0].set_xlim(-6, 0)
        ax[i, 0].set_title(f'{r}: bound F(E) / M_bound', loc='left', color=INK)
        ax[i, 1].set_title(f'{r}: dM/dE (debris) [M*/dE]', loc='left', color=INK)
        ax[i, 2].set_title(f'{r}: fallback Mdot [M*/M]', loc='left', color=INK)
        ax[i, 0].set_xlabel('E / dE'); ax[i, 1].set_xlabel('E / dE'); ax[i, 2].set_xlabel('t [M]')
        ax[i, 1].set_yscale('log')
    ax[0, 0].legend(fontsize=7.5, loc='upper left')
    fig.text(0.01, 0.005, 'final restarts (R6 t = 170, R7 t = 300); remnant excluded, r_BH >= 3; dE = 0.677/16.4^2.',
             color=MUTED, fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(out, dpi=130); plt.close(fig)


def fig_gate(ser, out):
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.2))
    for j, r in enumerate(S.RUNS):
        for x in S.ARMS:
            ts = sorted(ser[(r, x)])
            v = np.array([ser[(r, x)][t] for t in ts])
            ax[0, j].plot(ts, v[:, 0], color=COL[x], label=LAB[x])
            if r == 'R6':
                ax[1, j].plot(ts, v[:, 1], color=COL[x], label=LAB[x])
        ax[0, j].set_title(f'{r}: dense mean K/K0 (z = 0 slice)', loc='left', color=INK)
        if r == 'R6':
            ax[1, j].set_title(f'{r}: warm fraction (K > 1.05 K0)', loc='left', color=INK)
        else:
            for rr, ls in (('R6', '-'), ('R7', '--')):
                ts = sorted(set(ser[(rr, 'A')]) & set(ser[(rr, 'B')]) & set(ser[(rr, 'T')]))
                k = {x: np.array([ser[(rr, x)][t][0] for t in ts]) for x in S.ARMS}
                tt = np.array(ts) - ts[0]
                ax[1, j].plot(tt, k['B'] / k['A'] - 1, color=COL['B'], ls=ls, label=f'{rr} B/A - 1 (gate)')
                ax[1, j].plot(tt, k['T'] / k['B'] - 1, color=COL['T'], ls=ls, label=f'{rr} T/B - 1 (zone)')
            ax[1, j].set_title('K ratios (R7 warm fraction is 0 throughout)', loc='left', color=INK)
            ax[1, j].set_xlabel('t - t_restart [M]'); ax[1, j].legend(fontsize=7)
            continue
        ax[1, j].set_xlabel('t [M]')
        if r == 'R6':
            for a in ax[:, j]:
                a.axvspan(136, 150, color='#eeede9', zorder=0)
    ax[0, 0].legend(fontsize=7.5)
    fig.text(0.01, 0.005, 'B - A = the divergence-gate effect (coord vs covariant); T - B = the zone effect. '
             'Area- and density-weighted, rho > 0.1 rho_c.', color=MUTED, fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(out, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nproc', type=int, default=8)
    ap.add_argument('--out', default=S.P)
    a = ap.parse_args()
    L = {(r, x): S.parse(f'{S.C}/S2{x}_{r}') for r in S.RUNS for x in S.ARMS}
    fig_dt(L, f'{a.out}/c051_dt.png')
    fig_mass(L, f'{a.out}/c051_mass.png')
    Z = {(r, x): S.rst_pass(final_rst(r, x)) for r in S.RUNS for x in S.ARMS}   # cached
    fig_mdot(Z, f'{a.out}/c051_mdot.png')
    ser = {}
    for r in S.RUNS:
        for x in S.ARMS:
            fs = sorted(glob.glob(f'{S.C}/S2{x}_{r}/bin/{S.B}.prim_xy.*.bin'))
            with Pool(min(a.nproc, len(fs))) as p:
                ser[(r, x)] = {round(t): (k, w5) for t, k, w5, _ in p.map(S.slice_K, fs)}
    fig_gate(ser, f'{a.out}/c051_gate.png')


if __name__ == '__main__':
    main()
