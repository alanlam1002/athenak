#!/usr/bin/env python3
"""C-067 figures: the R-076 recheck (fixed binary: M1, excision_mask_refresh, mass ledger, ROMIO v3)
against the pre-fix S2 runs. Recheck dirs RC_{R6,R7}{A,B,T}, RC_Q1c (BH frame); pre-fix S2{A,B,T}_{R6,R7}
and RQ1_bh_L4_t300_from_De. (RC_L5A is being rerun; L5 is not plotted.)

  c067_dt.png        code dt(t), recheck T/A/B and pre-fix T (dashed); regrid cycles marked
  c067_ledger.png    cumulative rest-mass ledger per recheck arm (MLEDGER): grid dM, C2P, update,
                     between cycles; |.| on log axes; the 1e-10 M* closure tolerance
  c067_gate.png      dense K/K0 and warm (>5%) fraction on z = 0 slices, fixed vs pre-fix; B/A - 1
  c067_fallback.png  bound F(E) (all bound gas incl. remnant, per-arm normalised) and fallback Mdot
  c067_q1c.png       Q1c fixed vs pre-fix: K/K0, warm fraction, rho_max, grid mass
  c067_cost.png      PHASE_TIMERS_FINAL breakdown per arm, recheck vs pre-fix S2

    c067_figs.py [--nproc 8] [--out DIR]
"""
import argparse
import glob
import os
import re
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c051_s2 as S                                # noqa: E402

C = S.C
COL = {'T': '#2a78d6', 'A': '#eb6834', 'B': '#1baf7a'}
LAB = {'T': 'T: zone + covariant', 'A': 'A: coord', 'B': 'B: covariant'}
INK, MUTED = '#1f1f1e', '#6b6a64'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.6, 'lines.linewidth': 1.3,
                     'legend.frameon': False})
ARMS = ('T', 'A', 'B')
RS = ('R6', 'R7')


def rc(r, x):
    return f'{C}/RC_{r}{x}'


def pre(r, x):
    return f'{C}/S2{x}_{r}'


def logf(d):
    return f'{d}/ensemble.out' if os.path.exists(f'{d}/ensemble.out') else f'{d}/out.log'


def regrid_times(d):
    out, pend = [], False
    for ln in open(logf(d), errors='replace'):
        if 'post-regrid' in ln:
            pend = True
            continue
        m = re.match(r'elapsed=\S+ cycle=\d+ time=(\S+) dt=', ln)
        if m and pend:
            out.append(float(m.group(1))); pend = False
    return np.array(out)


def fig_dt(L, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a, r in zip(ax, RS):
        for x in ARMS:
            d = L[('rc', r, x)]['dt'][:-2]
            a.plot(d[:, 0], d[:, 1], color=COL[x], lw=0.8, label=f'{LAB[x]} (fixed)')
        d = L[('pre', r, 'T')]['dt'][:-2]
        a.plot(d[:, 0], d[:, 1], color=COL['T'], lw=0.8, ls='--', alpha=0.7, label='T pre-fix (S2)')
        y0 = a.get_ylim()[0]
        for x, off in (('T', 0.0), ('A', 0.004)):
            rt = regrid_times(rc(r, x))
            a.plot(rt, np.full_like(rt, y0 + off), '|', color=COL[x], ms=6)
        med = {x: np.median(L[('rc', r, x)]['dt'][:-2, 1]) for x in ARMS}
        a.set_title(f'{r}: code dt — T/A median gain x{med["T"] / med["A"]:.2f}', loc='left', color=INK)
        a.set_xlabel('t [M]'); a.set_ylabel('dt [M]')
        if r == 'R6':
            a.axvspan(136, 150, color='#eeede9', zorder=0)
    ax[0].legend(fontsize=7.5, loc='upper right')
    fig.text(0.01, 0.01, 'ticks at the bottom: regrid cycles (T blue, A orange). Fixed binary: M1 + excision_mask_refresh '
             '(no regrid dips); pre-fix T: S2 (C-051).', color=MUTED, fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(out, dpi=120); plt.close(fig)


def ledger_series(o):
    z = o['led']                                # cycle time Mu1 Mend Y Fout X
    t = z[:, 1]
    grid = z[:, 3] - z[0, 2]
    c2p = np.cumsum(z[:, 6])
    upd = np.cumsum(z[:, 4] + z[:, 5])
    g = np.concatenate([[0.0], np.cumsum(z[1:, 2] - z[:-1, 3])])
    return t, grid, c2p, upd, g


def fig_ledger(L, out):
    fig, ax = plt.subplots(2, 3, figsize=(13, 6.4), sharex='row')
    tol = S.CLOSE_TOL
    for i, r in enumerate(RS):
        for j, x in enumerate(ARMS):
            a = ax[i, j]
            t, grid, c2p, upd, g = ledger_series(L[('rc', r, x)])
            for y, lab, c, ls in ((grid, 'grid dM', INK, '-'), (c2p, 'C2P (floors, zone, excision)', '#8a5cd0', '-'),
                                  (g, 'between cycles (regrid / restart)', '#d03a5c', '-'),
                                  (upd, 'update Y + F_out', '#2a9d8f', '-')):
                a.semilogy(t, np.abs(y), color=c, ls=ls, lw=1.1, label=lab)
            a.axhline(tol, color=MUTED, ls='--', lw=1)
            a.text(t[0], tol * 1.4, '1e-10 M* tolerance', color=MUTED, fontsize=7)
            a.set_ylim(1e-21, 1e-8)
            a.set_title(f'{r} {x}: |cumulative| [M_BH units]', loc='left', color=INK)
            if i == 1:
                a.set_xlabel('t [M]')
    ax[0, 0].legend(fontsize=7, loc='upper left')
    fig.text(0.01, 0.01, 'MLEDGER per cycle: update = sum w_s (Y_s + F_out,s) (the flux update\'s non-conservation); C2P = sum w_s X_s; '
             'between cycles = M(u1)(n+1) - M_end(n). grid dM = C2P + update - F_out + between.', color=MUTED, fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(out, dpi=110); plt.close(fig)


def slices(d):
    return sorted(glob.glob(f'{d}/bin/{S.B}.prim_xy.*.bin'))


def series(job):
    d = job
    out = {}
    fs = slices(d)
    if 'RQ1_' in d:          # the pre-fix Q1c run spans t = 80-300; keep the recheck window
        fs = [f for f in fs if 118 <= int(f.split('.')[-2]) <= 172]
    for f in fs:
        try:
            t, k, w5, w10 = S.slice_K(f)
        except Exception:
            continue
        if abs(t - round(t)) < 0.1:            # dumps every 1 M: key on the nearest integer time
            out[float(round(t))] = (k, w5)
    return d, out


def fig_gate(ser, out):
    fig, ax = plt.subplots(3, 2, figsize=(11, 8.6))
    for j, r in enumerate(RS):
        for x in ARMS:
            s = ser[rc(r, x)]; ts = sorted(s)
            ax[0, j].plot(ts, [s[t][0] for t in ts], color=COL[x], label=f'{LAB[x]} fixed')
            ax[1, j].plot(ts, [s[t][1] for t in ts], color=COL[x], label=f'{LAB[x]} fixed')
        for x in ('A', 'B'):
            s = ser[pre(r, x)]; ts = sorted(s)
            ax[0, j].plot(ts, [s[t][0] for t in ts], color=COL[x], ls='--', alpha=0.7, label=f'{x} pre-fix')
            ax[1, j].plot(ts, [s[t][1] for t in ts], color=COL[x], ls='--', alpha=0.7, label=f'{x} pre-fix')
        for tag, f, ls in (('fixed', rc, '-'), ('pre-fix', pre, '--')):
            sa, sb = ser[f(r, 'A')], ser[f(r, 'B')]
            ts = sorted(set(sa) & set(sb))
            ax[2, j].plot(ts, [sb[t][0] / sa[t][0] - 1 for t in ts], color=COL['B'], ls=ls, label=f'B/A - 1 {tag}')
        ax[2, j].axhline(0, color=MUTED, lw=0.8)
        ax[0, j].set_title(f'{r}: dense mean K/K0 (z = 0)', loc='left', color=INK)
        ax[1, j].set_title(f'{r}: warm fraction (K > 1.05 K0)', loc='left', color=INK)
        ax[2, j].set_title(f'{r}: gate effect K_B/K_A - 1', loc='left', color=INK)
        ax[2, j].set_xlabel('t [M]')
        if r == 'R6':
            for a in ax[:, j]:
                a.axvspan(136, 150, color='#eeede9', zorder=0)
    ax[0, 0].legend(fontsize=7, ncol=2); ax[2, 0].legend(fontsize=7)
    fig.text(0.01, 0.005, 'solid: fixed binary (recheck); dashed: pre-fix S2. Area- and density-weighted, rho > 0.1 rho_c.',
             color=MUTED, fontsize=7)
    fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(out, dpi=110); plt.close(fig)


def final(d):
    return sorted(glob.glob(f'{d}/rst/*.rst'))[-1]


def fig_fallback(Z, out):
    import c044_mdot as M
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.4))
    tc = np.sqrt(M.TB[1:] * M.TB[:-1])
    for i, r in enumerate(RS):
        cfg = S.RUNS[r]
        tp = M.t_peri(cfg['src'], cfg['rp'])
        for tag, f, xs, ls in (('fixed', rc, ARMS, '-'), ('pre-fix', pre, ('A',), '--')):
            for x in xs:
                z = Z[f(r, x)]
                E, F, Mall, times, Mb = S.fallback(z, tp)
                ax[i, 0].plot(E[::40] / S.DE, F[::40], color=COL[x], ls=ls, label=f'{x} {tag}')
                deb = ~z['rem'].astype(bool)
                md = M.mdot(z['E'][deb], z['dM'][deb], tp)
                ax[i, 1].loglog(tc, np.where(md > 0, md, np.nan), color=COL[x], ls=ls, lw=1.1, label=f'{x} {tag}')
        ax[i, 0].set_xlim(-1.5, 0)
        ax[i, 0].set_title(f'{r}: F(E) over all bound gas / own M_bound', loc='left', color=INK)
        ax[i, 1].set_title(f'{r}: fallback Mdot [M*/M] (debris)', loc='left', color=INK)
        ax[i, 0].set_xlabel('E / dE'); ax[i, 1].set_xlabel('t [M]')
    ax[0, 0].legend(fontsize=7.5, loc='upper left')
    fig.text(0.01, 0.005, 'final restarts (R6 t = 170, R7 t = 300); r_BH >= 3; Keplerian return from the release periapsis.',
             color=MUTED, fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(out, dpi=110); plt.close(fig)


def hst(d, name='mhd'):
    f = glob.glob(f'{d}/{S.B}.{name}.hst')[0]
    return np.loadtxt(f)


def fig_q1c(ser, out):
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.2))
    for tag, d, ls, c in (('fixed', f'{C}/RC_Q1c', '-', '#2a78d6'), ('pre-fix', f'{C}/RQ1_bh_L4_t300_from_De', '--', '#eb6834')):
        s = ser[d]; ts = [t for t in sorted(s) if 123.0 <= t <= 170.5]
        ax[0, 0].plot(ts, [s[t][0] for t in ts], color=c, ls=ls, label=tag)
        ax[0, 1].plot(ts, [s[t][1] for t in ts], color=c, ls=ls, label=tag)
        u = hst(d, 'user'); k = (u[:, 0] >= 123.0) & (u[:, 0] <= 170.5)
        ax[1, 0].plot(u[k, 0], u[k, 2], color=c, ls=ls, label=tag)
        h = hst(d); k = (h[:, 0] >= 123.0) & (h[:, 0] <= 170.5)
        m0 = np.interp(123.8, h[:, 0], h[:, 2])
        ax[1, 1].plot(h[k, 0], h[k, 2] - m0, color=c, ls=ls, label=tag)
    ax[0, 0].set_title('Q1c (BH frame): dense mean K/K0 (z = 0)', loc='left', color=INK)
    ax[0, 1].set_title('Q1c: warm fraction (K > 1.05 K0)', loc='left', color=INK)
    ax[1, 0].set_title('Q1c: rho_max', loc='left', color=INK)
    ax[1, 1].set_title('Q1c: grid rest mass M(t) - M(123.8) [M_BH units]', loc='left', color=INK)
    for a in ax[1]:
        a.set_xlabel('t [M]')
    ax[0, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def fig_cost(L, out):
    keys = [('rc', r, x) for r in RS for x in ARMS] + [('pre', r, x) for r in RS for x in ARMS]
    lab, parts = [], []
    for k in keys:
        tm = L[k]['timers']
        if tm is None:
            continue
        w = tm['wall']
        mg, he, rg, io = tm['mg'], tm['stagen'] - tm['mg'], tm['regrid'], tm['io']
        oth = max(w - mg - he - rg - io, 0.0)
        lab.append(f'{k[1]} {k[2]}\n{"fixed" if k[0] == "rc" else "pre-fix"}')
        parts.append((mg, he, rg, io, oth, w))
    p = np.array(parts)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.0))
    names = ('multigrid', 'hydro + C2P', 'regrid', 'I/O', 'other')
    cols = ('#2a78d6', '#1baf7a', '#8a5cd0', '#d03a5c', '#bdbcb6')
    x = np.arange(len(lab)); bot = np.zeros(len(lab))
    for n, c, v in zip(names, cols, p[:, :5].T):
        ax[0].bar(x, v / p[:, 5], bottom=bot, color=c, label=n, width=0.7); bot += v / p[:, 5]
    ax[0].set_xticks(x); ax[0].set_xticklabels(lab, fontsize=7)
    ax[0].set_ylabel('fraction of wall time'); ax[0].set_title('wall-time breakdown (rank 0, fenced)', loc='left', color=INK)
    ax[0].legend(fontsize=7, ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.18))
    ax[1].bar(x, p[:, 5] / 3600, color=[COL[l.split()[1][0]] for l in lab], width=0.7)
    ax[1].set_xticks(x); ax[1].set_xticklabels(lab, fontsize=7)
    ax[1].set_ylabel('wall [h]'); ax[1].set_title('wall time per window', loc='left', color=INK)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nproc', type=int, default=8)
    ap.add_argument('--out', default=S.P)
    a = ap.parse_args()
    L = {}
    for r in RS:
        for x in ARMS:
            L[('rc', r, x)] = S.parse(rc(r, x))
            L[('pre', r, x)] = S.parse(pre(r, x))
    fig_dt(L, f'{a.out}/c067_dt.png'); print('dt', flush=True)
    fig_ledger(L, f'{a.out}/c067_ledger.png'); print('ledger', flush=True)
    fig_cost(L, f'{a.out}/c067_cost.png'); print('cost', flush=True)
    dirs = [rc(r, x) for r in RS for x in ARMS] + [pre(r, x) for r in RS for x in ('A', 'B')] + \
           [f'{C}/RC_Q1c', f'{C}/RQ1_bh_L4_t300_from_De']
    with Pool(a.nproc) as p:
        ser = dict(p.map(series, dirs))
    fig_gate(ser, f'{a.out}/c067_gate.png'); print('gate', flush=True)
    fig_q1c(ser, f'{a.out}/c067_q1c.png'); print('q1c', flush=True)
    fdirs = [rc(r, x) for r in RS for x in ARMS] + [pre(r, 'A') for r in RS]
    with Pool(min(a.nproc, len(fdirs))) as p:
        zs = p.map(S.rst_pass, [final(d) for d in fdirs])
    fig_fallback(dict(zip(fdirs, zs)), f'{a.out}/c067_fallback.png'); print('fallback', flush=True)


if __name__ == '__main__':
    main()
