#!/usr/bin/env python3
"""Plots and animations for the R-038 overnight TDE runs (Q1 V5-L4 rerun, Q6 +-128 box,
Q3 r_p = 18, Q4 2e4 Msun at L6) with the V5 references (old-binary L4, L5).

  ovn_timeseries.png    from prim_xy dumps (z = 0): rho_max/rho_c, K/K0 at rho_max, distance of
                        rho_max from the BH; from restarts (3D): M3 warm fractions (>5%) of the
                        dense gas and the GRR13-style bound remnant mass (scripts/remnant.py)
  ovn_dmde.png          dM/dE (scripts/tde_dmde.py) at the latest restart of each run, and
                        Q1 vs Q6 at the common time (box-size systematic)
  ovn_debris_anim.gif   BH-centred log rho (z = 0), the four runs side by side
  ovn_star_anim.gif     star-centred rho (top) and K/K0 (bottom), same four runs
  ovn_debris_snapshots.png  the debris view at a few fixed times

Units follow each run's <problem> rhoc / kappa (G = c = M_BH = 1), so the 2e4 Msun run (Q4)
has R* = 0.677 / s, r_t = 16.4 s^(-2/3), M* = 7.47e-5 / s with s = sqrt(rhoc / 3.24e-4) = 2. Restart-derived quantities are cached in
OUT/ovn_cache.npz (--recompute to redo).

    tde_ovn_plots.py [--out DIR] [--every 2] [--no-anim] [--recompute]
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
from matplotlib.colors import LogNorm               # noqa: E402
from matplotlib import animation                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tde_t11_surface_diag as _T                   # noqa: E402,F401 (sets up bc)
from tde_ensemble_plots import bc, draw             # noqa: E402
import tde_dmde as D                                # noqa: E402
import remnant as REM                               # noqa: E402

C = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc'
BASE = 'cfc_tde_wd_imbh_parabolic_tracker'
GAM = 5.0 / 3.0
MSTAR = 7.47e-5                                     # star mass in M_BH = 1e4 Msun units
# label -> (dir, colour, linestyle, title)
RUNS = {
    'Q1':   (f'{C}/Q1_V5L4_fix_t300', '#d95f02', '-', 'Q1  V5 rp10 L4 (fixed binary)'),
    'Q6':   (f'{C}/Q6_V5L4_box128', '#7570b3', '-', 'Q6  V5 rp10 L4, ±128 box'),
    'Q3':   (f'{C}/Q3_P2_rp18_L4', '#1b9e77', '-', 'Q3  rp18 L4 (partial)'),
    'Q4':   ((f'{C}/Q4_P1_2e4_L6', f'{C}/Q4_P1_2e4_L6_rst14'), '#e7298a', '-', 'Q4  2e4 Msun, β≈1.5, L6'),
    'V5L4': (f'{C}/de_V5_rp10_L4', '#e6ab02', ':', 'V5 L4 (old binary)'),
    'V5L5': (f'{C}/de_V5_rp10_L5_mgfix', '#666666', '--', 'V5 L5'),
}
ANIM = ['Q1', 'Q6', 'Q3', 'Q4']


def dirs(lab):
    """a run's directories in order (a restart after a crash goes to a fresh directory)"""
    d = RUNS[lab][0]
    return [x for x in ((d,) if isinstance(d, str) else d) if os.path.isdir(x)]


def rsts(lab):
    return sorted(f for d in dirs(lab) for f in glob.glob(f'{d}/rst/*.rst'))


def par(d, key):
    txt = open(f'{d}/parfile.par').read()
    return float(re.search(rf'^{key}\s*=\s*([-+.\deE]+)', txt, re.M).group(1))


def consts(lab):
    d = dirs(lab)[0]
    rhoc, k0 = par(d, 'rhoc'), par(d, 'kappa')
    s = np.sqrt(rhoc / 3.24e-4)                    # = M_BH / 1e4 Msun
    return dict(rhoc=rhoc, k0=k0, rstar=0.677 / s, mscale=s, mstar=MSTAR / s)


def dumps(lab):
    out = {}
    for d in dirs(lab):                             # later directories override overlapping dumps
        out.update({int(f.split('.')[-2]): f for f in glob.glob(f'{d}/bin/{BASE}.prim_xy.*.bin')})
    return out


def read(f, k):
    p = bc.read_binary(f)
    g = np.asarray(p['mb_geometry'])
    d = np.asarray(p['mb_data']['dens'])[:, 0]
    P = np.asarray(p['mb_data']['press'])[:, 0]
    m, j, i = np.unravel_index(np.argmax(d), d.shape)
    nx = p['nx1_out_mb']
    x0 = g[m, 0] + (i + .5) * (g[m, 1] - g[m, 0]) / nx
    y0 = g[m, 2] + (j + .5) * (g[m, 3] - g[m, 2]) / nx
    K = np.where(d > 1e-4 * k['rhoc'], P / d**GAM / k['k0'], np.nan)
    return dict(t=p['time'], x0=x0, y0=y0, g=g, d=d, K=K,
                rmax=d.max() / k['rhoc'], kmax=P[m, j, i] / d[m, j, i]**GAM / k['k0'])


def slice_series(lab):
    k = consts(lab)
    out = []
    for n, f in sorted(dumps(lab).items()):
        try:
            s = read(f, k)
        except Exception:
            continue
        out.append((s['t'], s['rmax'], s['kmax'], np.hypot(s['x0'], s['y0'])))
    return np.array(out)


def rst_series(lab):
    """t, M3 warm>5%, warm>10%, bound remnant / M*, bound fraction (E<0) per restart"""
    k = consts(lab)
    rows = []
    for f in rsts(lab):
        try:
            c = D.cells(f, rcut=3.0)
        except Exception as e:
            print(f'  skip {f}: {e}', flush=True); continue
        dense = c['rho'] > 0.1 * k['rhoc']
        m = c['dM'][dense]; x = c['P'][dense] / c['rho'][dense]**GAM / k['k0'] - 1
        w5 = m[x > 0.05].sum() / m.sum() if m.sum() > 0 else np.nan
        w10 = m[x > 0.10].sum() / m.sum() if m.sum() > 0 else np.nan
        Mb = REM.bound_mass(c)[0] / k['mstar']
        fb = c['dM'][c['E'] < 0].sum() / c['dM'].sum()
        rows.append((c['t'], w5, w10, Mb, fb))
        print(f'  {lab} t={c["t"]:6.1f} warm>5% {w5:.3f} bound remnant {Mb:.3f} M* E<0 {fb:.3f}', flush=True)
    return np.array(rows)


def dmde(lab, f):
    k = consts(lab)
    t, E, EB, dM = D.energies(f, rcut=3.0)
    rt = 16.4 * k['mscale']**(-2.0 / 3.0)          # r_t / M_BH = (R*/M_BH) (M_BH/M*)^(1/3)
    return t, D.hist(E, dM, k['rstar'] / rt**2, EDGES)


EDGES = np.linspace(-4, 4, 81)


def cache(out, recompute):
    fn = f'{out}/ovn_cache.npz'
    old = {} if recompute or not os.path.exists(fn) else dict(np.load(fn, allow_pickle=True))
    res = {}
    todo = [lab for lab in RUNS if not (f'{lab}_rst' in old and len(old[f'{lab}_rst'])
                                        == len(rsts(lab)))]
    with Pool(len(todo) or 1) as pool:                # restarts are ~1 min each: one process per run
        new = dict(zip(todo, pool.map(rst_series, todo)))
    for lab in RUNS:
        key = f'{lab}_rst'
        res[key] = new[lab] if lab in new else old[key]
        nd = len(dumps(lab))
        key = f'{lab}_sl'
        if key in old and len(old[key]) == nd:
            res[key] = old[key]
        else:
            print(f'slices: {lab}', flush=True)
            res[key] = slice_series(lab)
        key = f'{lab}_dmde'
        last = rsts(lab)[-1]
        if key in old and old[f'{key}_f'] == last:
            res[key], res[f'{key}_f'] = old[key], old[f'{key}_f']
        else:
            res[key] = np.array(dmde(lab, last), dtype=object); res[f'{key}_f'] = last
    # Q1 at the time of Q6's last restart (box comparison)
    t6 = res['Q6_dmde'][0]
    q1 = min(rsts('Q1'),
             key=lambda f: abs(D.R.read_tree(f)['time'] - t6))
    if 'Q1at6' in old and old['Q1at6_f'] == q1:
        res['Q1at6'], res['Q1at6_f'] = old['Q1at6'], old['Q1at6_f']
    else:
        res['Q1at6'] = np.array(dmde('Q1', q1), dtype=object); res['Q1at6_f'] = q1
    np.savez(fn, **res)
    return res


def periapsis(sl):
    i = np.argmin(sl[:, 3])
    return sl[i, 0]


def fig_series(res, out):
    fig, ax = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    for lab, (d, col, ls, title) in RUNS.items():
        sl, rs = res[f'{lab}_sl'], res[f'{lab}_rst']
        kw = dict(color=col, ls=ls, lw=1.3, label=title)
        if len(sl):
            ax[0, 0].plot(sl[:, 0], sl[:, 1], **kw)
            ax[0, 1].plot(sl[:, 0], sl[:, 2], **kw)
            ax[0, 2].plot(sl[:, 0], sl[:, 3], **kw)
            if lab in ANIM:
                tp = periapsis(sl)
                for a in ax.flat:
                    a.axvline(tp, color=col, lw=0.6, alpha=0.5)
        if len(rs):
            ax[1, 0].plot(rs[:, 0], rs[:, 1], 'o' + ls, color=col, ms=4, label=title)
            ax[1, 0].plot(rs[:, 0], rs[:, 2], 'x' + ls, color=col, ms=4, alpha=0.5)
            ax[1, 1].plot(rs[:, 0], rs[:, 3], 'o' + ls, color=col, ms=4, label=title)
            ax[1, 2].plot(rs[:, 0], rs[:, 4], 'o' + ls, color=col, ms=4, label=title)
    ax[0, 0].set_yscale('log'); ax[0, 0].set_title(r'$\rho_{max}/\rho_c$ (z = 0 slice)')
    ax[0, 1].set_title(r'$K/K_0$ at $\rho_{max}$'); ax[0, 1].set_ylim(0.9, 1.3)
    ax[0, 1].axhspan(0.98, 1.02, color='0.9')
    ax[0, 2].set_title(r'distance of $\rho_{max}$ from the BH [$M_{BH}$]')
    ax[0, 2].set_yscale('log')
    ax[1, 0].set_title('M3: dense-mass fraction > 5% warm (o) and > 10% (x), 3D restarts')
    ax[1, 1].set_title('self-bound remnant mass / M* (Newtonian monopole, iterative)')
    ax[1, 2].set_title('bound fraction of the gas on the grid, E = −u_t − 1 < 0')
    for a in ax.flat:
        a.set_xlabel('t [M]'); a.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=8)
    fig.suptitle('R-038 overnight TDE runs (vertical lines: periapsis of ρ_max, per run)')
    fig.savefig(f'{out}/ovn_timeseries.png', dpi=90); plt.close(fig)


def fig_dmde(res, out):
    xc = 0.5 * (EDGES[1:] + EDGES[:-1])
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for lab, (d, col, ls, title) in RUNS.items():
        t, H = res[f'{lab}_dmde']
        ax[0].semilogy(xc, np.maximum(H, 1e-6), color=col, ls=ls, label=f'{title}, t={t:.0f}')
    t6, H6 = res['Q6_dmde']; t1, H1 = res['Q1at6']
    ax[1].semilogy(xc, np.maximum(H1, 1e-6), color=RUNS['Q1'][1], label=f'Q1 ±64, t={t1:.0f}')
    ax[1].semilogy(xc, np.maximum(H6, 1e-6), color=RUNS['Q6'][1], label=f'Q6 ±128, t={t6:.0f}')
    a2 = ax[1].twinx()
    a2.plot(xc, (H6 - H1) / np.maximum(H1, 1e-30), 'k.', ms=3, label='(Q6 − Q1)/Q1')
    a2.set_ylim(-0.5, 0.5); a2.set_ylabel('(Q6 − Q1)/Q1'); a2.axhline(0, color='k', lw=0.5)
    for a in ax:
        a.set_xlabel('E / ΔE   (ΔE = R*/r_t², per run)'); a.set_ylabel('(1/M) dM/d(E/ΔE)')
        a.set_ylim(1e-4, 3); a.grid(alpha=0.3); a.legend(fontsize=8, loc='upper left')
    ax[0].set_title('dM/dE at the latest restart of each run')
    ax[1].set_title('box-size systematic: Q1 (±64) vs Q6 (±128)')
    fig.savefig(f'{out}/ovn_dmde.png', dpi=90); plt.close(fig)


CR = plt.get_cmap('inferno').copy(); CR.set_bad('k')
CK = plt.get_cmap('coolwarm').copy(); CK.set_bad('0.15')
NK = LogNorm(vmin=0.5, vmax=2.0)


def blocks(s, f):
    return [(g[0], g[1], g[2], g[3], v) for g, v in zip(s['g'], f)]


def nearest(dd, t):
    if not dd:
        return None
    n = min(dd, key=lambda m: abs(m - t))
    return dd[n] if abs(n - t) <= 1 else None


def debris_panel(a, lab, s, half):
    k = consts(lab)
    hb = half
    draw(a, blocks(s, s['d'] / k['rhoc']), ((-hb, hb), (-hb, hb)),
         LogNorm(vmin=1e-9, vmax=1.0), CR, outline=False)
    a.plot(0, 0, 'c+', ms=6)
    a.set_xticks([]); a.set_yticks([])


def star_panel(a, lab, s, field, half=2.0):
    k = consts(lab)
    h = half * k['rstar']
    view = ((s['x0'] - h, s['x0'] + h), (s['y0'] - h, s['y0'] + h))
    if field == 'rho':
        draw(a, blocks(s, s['d'] / k['rhoc']), view, LogNorm(vmin=1e-7, vmax=1.2), CR, outline=True)
    else:
        draw(a, blocks(s, s['K']), view, NK, CK, outline=False)
    a.add_patch(plt.Circle((s['x0'], s['y0']), k['rstar'], fill=False, color='c', lw=0.6, ls='--'))
    a.set_xticks([]); a.set_yticks([])


def fig_snap(out, times=(0, 100, 140, 180, 240, 300), half=40):
    dd = {lab: dumps(lab) for lab in ANIM}
    fig, ax = plt.subplots(len(ANIM), len(times), figsize=(2.6 * len(times), 2.7 * len(ANIM)),
                           constrained_layout=True)
    for r, lab in enumerate(ANIM):
        for c, t in enumerate(times):
            a = ax[r, c]; f = nearest(dd[lab], t)
            if f is None:
                a.set_facecolor('0.85'); a.set_xticks([]); a.set_yticks([])
                a.text(.5, .5, 'no data', ha='center', transform=a.transAxes)
            else:
                debris_panel(a, lab, read(f, consts(lab)), half)
            if r == 0:
                a.set_title(f't = {t}')
            if c == 0:
                a.set_ylabel(RUNS[lab][3] + f'\n±{half} M',
                             fontsize=8)
    fig.colorbar(plt.cm.ScalarMappable(norm=LogNorm(1e-9, 1), cmap=CR), ax=ax, shrink=0.6,
                 label=r'$\rho/\rho_c$ (z = 0), BH-centred (cyan +); lengths in $M_{BH}$')
    fig.savefig(f'{out}/ovn_debris_snapshots.png', dpi=90); plt.close(fig)


def anim(out, every, kind, tmax=300, half=40):
    dd = {lab: dumps(lab) for lab in ANIM}
    times = list(range(0, tmax + 1, every))
    if kind == 'debris':
        fig, ax = plt.subplots(1, len(ANIM), figsize=(4.0 * len(ANIM), 4.4), constrained_layout=True)
        ax = ax[None, :]
        fig.colorbar(plt.cm.ScalarMappable(norm=LogNorm(1e-9, 1), cmap=CR), ax=ax, shrink=0.8,
                     label=r'$\rho/\rho_c$')
    else:
        fig, ax = plt.subplots(2, len(ANIM), figsize=(3.2 * len(ANIM), 6.6), constrained_layout=True)
        fig.colorbar(plt.cm.ScalarMappable(norm=LogNorm(1e-7, 1.2), cmap=CR), ax=ax[0], shrink=0.8,
                     label=r'$\rho/\rho_c$')
        fig.colorbar(plt.cm.ScalarMappable(norm=NK, cmap=CK), ax=ax[1], shrink=0.8, label=r'$K/K_0$')
    last = {lab: None for lab in ANIM}

    def update(n):
        t = times[n]
        for c, lab in enumerate(ANIM):
            f = nearest(dd[lab], t)
            ended = f is None
            if ended:
                f = last[lab]
            last[lab] = f
            for r in range(ax.shape[0]):
                a = ax[r, c]; a.clear()
                if f is None:
                    a.axis('off'); continue
                s = read(f, consts(lab))
                if kind == 'debris':
                    debris_panel(a, lab, s, half)
                else:
                    star_panel(a, lab, s, 'rho' if r == 0 else 'K')
            if f is not None:
                sub = (f'±{half} M' if kind == 'debris'
                       else rf'$\rho_{{max}}/\rho_c$={s["rmax"]:.3f}')
                ax[0, c].set_title(f'{RUNS[lab][3]}\n{sub}' + (f'  (ended t={s["t"]:.0f})' if ended else ''),
                                   fontsize=8)
        fig.suptitle(('BH-centred debris, ' if kind == 'debris' else 'star-centred ±2 R*, ') + f't = {t}')
        return []

    an = animation.FuncAnimation(fig, update, frames=len(times), blit=False)
    an.save(f'{out}/ovn_{kind}_anim.gif', writer=animation.PillowWriter(fps=10), dpi=65)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{C}/plots/overnight')
    ap.add_argument('--every', type=int, default=2)
    ap.add_argument('--no-anim', action='store_true')
    ap.add_argument('--only-anim', default=None, help='debris or star')
    ap.add_argument('--recompute', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.only_anim:
        anim(a.out, a.every, a.only_anim); print(f'{a.only_anim} animation', flush=True); return
    res = cache(a.out, a.recompute)
    fig_series(res, a.out); print('time series', flush=True)
    fig_dmde(res, a.out); print('dM/dE', flush=True)
    fig_snap(a.out); print('debris snapshots', flush=True)
    if not a.no_anim:
        for kind in ('debris', 'star'):
            anim(a.out, a.every, kind); print(f'{kind} animation', flush=True)


if __name__ == '__main__':
    main()
