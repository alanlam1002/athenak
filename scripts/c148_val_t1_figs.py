#!/usr/bin/env python3
"""c148 (R-155 item 3(a)/(b), C-132..C-134): Valencia vs killing_t1 (and ta where it exists) on the
stationary test beds, figures and slice animations.
 michel_radial.png  signed errors vs r_iso at t = 120 (rho/rho_E - 1, K/K_E - 1, u~^r - u~^r_E), L2/L3/L4,
                    binned means over evolved cells of the last prim_3d dump
 michel_conv.png    rho L1 at 2-3 M vs level, and |KCLOSE source| vs level (KCLOSE_RUN lines in out.log)
 michel_time.png    rho and K L1 at 2-3 M vs t (the c122_b23.py csvs b2_L{2,3,4}.csv)
 b3_radial.png      hydrostatic: the same errors vs the initial state at t = 100, L4
 michel_slices.png  z = 0 slices at t = 120, L4: rho and K errors for val, t1 and t1 - val (meshblock outlines)
 michel_L4_val_t1.gif, b3_L4_val_t1.gif   the same slices for every prim_xy dump
The exact Michel solution and the helpers are c122_b23.py's; the hydrostatic reference is the t = 0 dump.
    c148_val_t1_figs.py RUNROOT CSVDIR OUTDIR [--mesh 0.4] [--half 10] [--only figs|anim]"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                   # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from scipy.interpolate import CubicSpline         # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tde_ensemble_plots import bc                 # noqa: E402
import c122_b23 as cb                              # noqa: E402

COL = {'val': '#2a78d6', 't1': '#eb6834', 'ta': '#1baf7a'}      # categorical slots 1-3, fixed order
NAME = {'val': 'Valencia', 't1': 'killing_t1', 'ta': 'killing_ta'}
LS = {2: ':', 3: '--', 4: '-'}
B = cb.B
GAM = cb.GAM


def kclose_source(d):
    s = [l for l in open(f'{d}/out.log') if l.startswith('KCLOSE_RUN')]
    return float(re.search(r'source=(\S+)', s[-1]).group(1)) if s else np.nan


class Michel:
    """the exact profile on a log grid, interpolated in log r"""
    def __init__(self, d):
        rg = np.geomspace(1.0, 70.0, 700)
        rho, ur, self.k = cb.michel_profile(cb.par(d, 'problem', 'test_michel_rc', 8.0),
                                            cb.par(d, 'problem', 'test_michel_rhoc', 1e-8), rg)
        self.lr, self.rho, self.ur = np.log(rg), CubicSpline(np.log(rg), np.log(rho)), CubicSpline(np.log(rg), ur)

    def __call__(self, r):
        lr = np.log(np.maximum(r, 1.0))
        return np.exp(self.rho(lr)), self.ur(lr), self.k


def Kof(D):
    return D['press']/D['dens']**GAM


def errors(D, x, y, z, ref):
    """signed rho, K and u~^r errors; ref = Michel instance or the reference cell dict (hydrostatic)"""
    r = np.sqrt(x*x + y*y + z*z)
    ur = (D['velx']*x + D['vely']*y + D['velz']*z)/np.maximum(r, 1e-30)
    if isinstance(ref, Michel):
        rhoE, urE, kE = ref(r)
    else:
        rhoE, kE = ref['dens'], Kof(ref)
        urE = (ref['velx']*x + ref['vely']*y + ref['velz']*z)/np.maximum(r, 1e-30)
    return r, D['dens']/rhoE - 1.0, Kof(D)/kE - 1.0, ur - urE


def cells3d(f):
    P = bc.read_binary(f)
    g = np.asarray(P['mb_geometry']); n = P['nx1_out_mb']
    X, Y, Z, D = [], [], [], {}
    for b in range(len(g)):
        h = (g[b, 1] - g[b, 0])/n
        c = lambda lo: lo + (np.arange(n) + 0.5)*h
        zz, yy, xx = np.meshgrid(c(g[b, 4]), c(g[b, 2]), c(g[b, 0]), indexing='ij')
        X.append(xx.ravel()); Y.append(yy.ravel()); Z.append(zz.ravel())
        for q in ('dens', 'press', 'velx', 'vely', 'velz'):
            D.setdefault(q, []).append(np.asarray(P['mb_data'][q][b]).ravel())
    return float(P['time']), np.concatenate(X), np.concatenate(Y), np.concatenate(Z), \
        {k: np.concatenate(v) for k, v in D.items()}


def binned(r, e, edges):
    i = np.digitize(r, edges) - 1
    ok = (i >= 0) & (i < len(edges) - 1) & np.isfinite(e)
    s = np.bincount(i[ok], e[ok], len(edges) - 1); n = np.bincount(i[ok], None, len(edges) - 1)
    m = np.where(n > 0, s/np.maximum(n, 1), np.nan)
    return np.sqrt(edges[1:]*edges[:-1]), m


def style(ax):
    ax.grid(True, lw=0.4, alpha=0.35)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def radial_fig(runs, refs, out, title, rmax, tlab, symlog=None):
    edges = np.geomspace(1.0, rmax, 70)
    fig, axs = plt.subplots(3, 1, figsize=(7.5, 9), sharex=True)
    labs = [r'$\rho/\rho_{\rm E}-1$', r'$K/K_{\rm E}-1$', r'$\tilde u^r-\tilde u^r_{\rm E}$  (coord)']
    for (var, L, d) in runs:
        f = sorted(glob.glob(f'{d}/bin/{B}.prim_3d.*.bin'))[-1]
        t, x, y, z, D = cells3d(f)
        ref = refs[d]
        if not isinstance(ref, Michel):
            ref = ref[1]
        r, er, ek, eu = errors(D, x, y, z, ref)
        for ax, e in zip(axs, (er, ek, eu)):
            rc, m = binned(r, e, edges)
            ax.plot(rc, m, color=COL[var], ls=LS[L], lw=2 if L == 4 else 1.4, label=f'{NAME[var]} L{L}')
    for ax, lab in zip(axs, labs):
        ax.set_xscale('log'); ax.set_ylabel(lab); ax.axhline(0, color='#52514e', lw=0.8)
        if symlog:
            ax.set_yscale('symlog', linthresh=symlog)
        ax.axvspan(2, 3, color='#52514e', alpha=0.07, lw=0)
        ax.axvline(4.0, color='#52514e', lw=0.7, ls='-.'); ax.axvline(8.0, color='#52514e', lw=0.7, ls='-.')
        style(ax)
    axs[0].text(4.05, axs[0].get_ylim()[1], ' L4 edge', va='top', fontsize=8, color='#52514e')
    axs[0].text(8.1, axs[0].get_ylim()[1], ' L3 edge', va='top', fontsize=8, color='#52514e')
    axs[-1].set_xlabel(r'$r_{\rm iso}$ [M]')
    axs[0].legend(ncol=2, fontsize=8, frameon=False)
    axs[0].set_title(f'{title}: shell-mean signed error at {tlab} (shaded: 2-3 M)', fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print('wrote', out)


def conv_fig(root, csvdir, out):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for var in ('val', 't1', 'ta'):
        Ls, e, s = [], [], []
        for L in (2, 3, 4):
            fn = f'{csvdir}/b2_L{L}.csv'
            if not os.path.exists(fn):
                continue
            rows = [l.split(',') for l in open(fn)][1:]
            rr = [w for w in rows if w[0] == f'B2_{var}_L{L}' and w[2] == '2-3' and float(w[1]) >= 119]
            if not rr:
                continue
            Ls.append(L); e.append(float(rr[-1][4])); s.append(abs(kclose_source(f'{root}/B2_{var}_L{L}')))
        if Ls:
            axs[0].plot(Ls, e, 'o-', color=COL[var], lw=2, ms=8, label=NAME[var])
            axs[1].plot(Ls, s, 'o-', color=COL[var], lw=2, ms=8,
                        label=NAME[var] + (' (diagnostic t1 ledger)' if var == 'val' else ''))
    for ax, yl, tt in ((axs[0], r'$\rho$ L1 rel, 2-3 M, t = 120', 'Michel: near-hole density error'),
                       (axs[1], '|KCLOSE source| (cumulative, t = 120)', 'Killing-energy residual source')):
        ax.set_yscale('log'); ax.set_xticks([2, 3, 4]); ax.set_xticklabels(['L2', 'L3', 'L4'])
        ax.set_ylabel(yl); ax.set_title(tt, fontsize=10); style(ax); ax.legend(fontsize=8, frameon=False)
    for ax, y0 in ((axs[0], 9e-3), (axs[1], 4e-5)):
        for p, lab in ((1, 'p = 1'), (2, 'p = 2')):
            ax.plot([2, 4], [y0, y0/2**(2*p)], color='#52514e', lw=0.8, ls=':')
            ax.text(4.02, y0/2**(2*p), lab, fontsize=8, color='#52514e', va='center')
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print('wrote', out)


def time_fig(csvdir, out):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for L in (2, 3, 4):
        fn = f'{csvdir}/b2_L{L}.csv'
        if not os.path.exists(fn):
            continue
        rows = [l.split(',') for l in open(fn)][1:]
        for var in ('val', 't1'):
            rr = [w for w in rows if w[0] == f'B2_{var}_L{L}' and w[2] == '2-3' and float(w[1]) > 0]   # t = 0: exact
            if not rr:
                continue
            t = [float(w[1]) for w in rr]
            axs[0].plot(t, [float(w[4]) for w in rr], color=COL[var], ls=LS[L], lw=2 if L == 4 else 1.4,
                        marker='o', ms=4, label=f'{NAME[var]} L{L}')
            axs[1].plot(t, [float(w[10]) for w in rr], color=COL[var], ls=LS[L], lw=2 if L == 4 else 1.4,
                        marker='o', ms=4)
    for ax, yl in ((axs[0], r'$\rho$ L1 rel, 2-3 M'), (axs[1], r'$K$ L1 rel, 2-3 M')):
        ax.set_yscale('log'); ax.set_xlabel('t [M]'); ax.set_ylabel(yl); style(ax)
    fig.legend(*axs[0].get_legend_handles_labels(), ncol=6, fontsize=8, frameon=False, loc='lower center')
    fig.suptitle('Michel: near-hole errors vs time', fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(out, dpi=130); plt.close(fig)
    print('wrote', out)


# ---- z = 0 slices ----------------------------------------------------------------------------------------
def slice_blocks(f, half):
    P = bc.read_binary(f)
    g = np.asarray(P['mb_geometry']); n1, n2 = P['nx1_out_mb'], P['nx2_out_mb']
    out = []
    for b in range(len(g)):
        if g[b, 0] > half or g[b, 1] < -half or g[b, 2] > half or g[b, 3] < -half:
            continue
        h = (g[b, 1] - g[b, 0])/P['nx1_mb']
        x = g[b, 0] + (np.arange(n1) + 0.5)*(g[b, 1] - g[b, 0])/n1
        y = g[b, 2] + (np.arange(n2) + 0.5)*(g[b, 3] - g[b, 2])/n2
        Y, X = np.meshgrid(y, x, indexing='ij')
        D = {q: np.asarray(P['mb_data'][q][b])[0] for q in ('dens', 'press', 'velx', 'vely', 'velz')}
        out.append((g[b], X, Y, 0.5*h, D))
    return float(P['time']), out


def slice_err(blocks, ref, refblocks=None):
    res = []
    for k, (g, X, Y, z, D) in enumerate(blocks):
        Z = np.full_like(X, z)
        rr = refblocks[k][4] if refblocks is not None else ref
        r, er, ek, _ = errors(D, X, Y, Z, rr)
        m = r < 1.0                                         # excised throat (c122_b23: r_iso >= 1 only)
        res.append((g, X, Y, np.where(m, np.nan, er), np.where(m, np.nan, ek)))
    return res


def draw(ax, sl, which, vmax, mesh, half, cmap='RdBu_r'):
    im = None
    for (g, X, Y, er, ek) in sl:
        v = er if which == 'rho' else ek
        h = (g[1] - g[0])/X.shape[1]
        im = ax.imshow(v, origin='lower', extent=(g[0], g[1], g[2], g[3]), cmap=cmap, vmin=-vmax, vmax=vmax,
                       interpolation='nearest')
    if mesh:
        segs = []
        for (g, *_r) in sl:
            x0, x1, y0, y1 = g[0], g[1], g[2], g[3]
            segs += [[(x0, y0), (x1, y0)], [(x1, y0), (x1, y1)], [(x1, y1), (x0, y1)], [(x0, y1), (x0, y0)]]
        ax.add_collection(LineCollection(segs, colors='k', linewidths=mesh, alpha=0.35))
    ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_aspect('equal')
    return im


def diff_err(a, b):
    return [(ga, X, Y, eb - ea, kb - ka) for (ga, X, Y, ea, ka), (_g, _X, _Y, eb, kb) in zip(a, b)]


def slice_panel(fig, axs, sa, sb, t, A, title):
    vr, vk = A.vmax, 2*A.vmax
    ims = []
    for j, (sl, lab) in enumerate(((sa, NAME['val']), (sb, NAME['t1']), (diff_err(sa, sb), 't1 - val'))):
        for i, (which, v, nm) in enumerate((('rho', vr, r'$\rho/\rho_{\rm E}-1$'), ('K', vk, r'$K/K_{\rm E}-1$'))):
            ax = axs[i, j]; ax.clear()
            im = draw(ax, sl, which, v, A.mesh, A.half)
            ax.set_title(f'{lab}: {nm}', fontsize=9)
            if i == 1:
                ax.set_xlabel('x [M]')
            if j == 0:
                ax.set_ylabel('y [M]')
            ims.append(im)
    fig.suptitle(f'{title}   t = {t:.1f} M   (colour: rho ±{vr:g}, K ±{vk:g}; meshblock outlines; white disc r < 1 not shown)',
                 fontsize=10)
    return ims


def slices_and_anim(root, A, bed, L, tag, outdir):
    da, db = f'{root}/{bed}_val_L{L}', f'{root}/{bed}_t1_L{L}'
    fa = sorted(glob.glob(f'{da}/bin/{B}.prim_xy.*.bin')); fb = sorted(glob.glob(f'{db}/bin/{B}.prim_xy.*.bin'))
    n = min(len(fa), len(fb))
    ref = Michel(da) if bed == 'B2' else None
    ra = rb = None
    if bed == 'B3':
        ra, rb = slice_blocks(fa[0], A.half)[1], slice_blocks(fb[0], A.half)[1]

    def frame(k):
        ta, ba = slice_blocks(fa[k], A.half); tb, bb = slice_blocks(fb[k], A.half)
        assert abs(ta - tb) < 0.2, (ta, tb)        # each run dumps at its own first cycle past the output time
        return ta, slice_err(ba, ref, ra), slice_err(bb, ref, rb)

    title = {'B2': f'Michel inflow L{L}', 'B3': f'Hydrostatic atmosphere L{L} (error vs t = 0)'}[bed]
    fig, axs = plt.subplots(2, 3, figsize=(12.5, 8.4), layout='constrained')
    t, sa, sb = frame(n - 1)
    ims = slice_panel(fig, axs, sa, sb, t, A, title)
    fig.colorbar(ims[0], ax=axs[0, :], shrink=0.85, label=r'$\rho$ signed rel. error')
    fig.colorbar(ims[1], ax=axs[1, :], shrink=0.85, label='K signed rel. error')
    p = f'{outdir}/{tag}_slices.png'; fig.savefig(p, dpi=110); print('wrote', p)
    if A.only == 'figs':
        plt.close(fig); return

    def upd(k):
        t, sa, sb = frame(k)
        slice_panel(fig, axs, sa, sb, t, A, title)
        return []
    anim = FuncAnimation(fig, upd, frames=range(0, n, A.every), blit=False)
    p = f'{outdir}/{tag}_L{L}_val_t1.gif'
    anim.save(p, writer=PillowWriter(fps=6), dpi=A.dpi); plt.close(fig)
    print('wrote', p, f'({(n + A.every - 1)//A.every} frames, {os.path.getsize(p)/1e6:.2f} MB)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root'); ap.add_argument('csvdir'); ap.add_argument('outdir')
    ap.add_argument('--mesh', type=float, default=0.4)
    ap.add_argument('--half', type=float, default=10.0)
    ap.add_argument('--vmax', type=float, default=5e-3)
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--dpi', type=int, default=70)
    ap.add_argument('--only', choices=('figs', 'anim', 'all'), default='all')
    A = ap.parse_args()
    os.makedirs(A.outdir, exist_ok=True)
    R = A.root
    if A.only != 'anim':
        runs = [(v, L, f'{R}/B2_{v}_L{L}') for L in (2, 3, 4) for v in ('val', 't1')]
        mic = Michel(runs[0][2])
        radial_fig(runs, {d: mic for (_v, _L, d) in runs}, f'{A.outdir}/michel_radial.png', 'Michel inflow', 38.0,
                   't = 120', symlog=1e-4)
        b3 = [(v, 4, f'{R}/B3_{v}_L4') for v in ('val', 't1', 'ta')]
        refs = {d: (None, cells3d(sorted(glob.glob(f'{d}/bin/{B}.prim_3d.*.bin'))[0])[4]) for (_v, _L, d) in b3}
        radial_fig(b3, refs, f'{A.outdir}/b3_radial.png', 'Hydrostatic atmosphere L4 (vs t = 0)', 38.0, 't = 100')
        conv_fig(R, A.csvdir, f'{A.outdir}/michel_conv.png')
        time_fig(A.csvdir, f'{A.outdir}/michel_time.png')
    slices_and_anim(R, A, 'B2', 4, 'michel', A.outdir)
    slices_and_anim(R, A, 'B3', 4, 'b3', A.outdir)


if __name__ == '__main__':
    main()
