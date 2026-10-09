#!/usr/bin/env python3
"""Summary plots for the parabolic-TDE periapsis scan (prod ensemble 8914650 and
continuations): time series of rho_max, the star's own lapse dip, orbit and orbital
invariants for every member, a star-frame snapshot grid, and density animations.

Per-dump quantities are measured on the z=0 bin slices and cached in
OUT/cache/<label>.npz (rebuilt when a run has new dumps):
  rho_max (slice), CoM (rho > 1e-2 rho_max), R_areal of the CoM,
  dalpha_min = min over r_iso > 3 of (alpha - alpha_BH(r)), where alpha_BH is the
      analytic maximal-slicing trumpet lapse of the bare puncture
      (tde_elliptical_orbit_ic.metric_at). It is the star's own lapse dip, a
      gauge-level proxy for its potential depth / compactness.
  E, L (orbit_invariants.measure, core rho > 0.5 rho_max), MeshBlocks in the slice.
History-file rho_max (3D, every 0.1 M) is used for the rho_max figure.

Needs numpy + matplotlib (module load frameworks/2026.1.0).

    tde_ensemble_plots.py [--out DIR] [--anim rp10,rp16] [--anim-every 2] [--no-anim]
"""
import argparse
import glob
import math
import os
import sys
import types

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.colors import LogNorm               # noqa: E402
from matplotlib.collections import LineCollection   # noqa: E402
from matplotlib import animation                    # noqa: E402

sys.modules.setdefault('h5py', types.ModuleType('h5py'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'vis', 'python'))
sys.path.insert(0, HERE)
import bin_convert as bc                 # noqa: E402
import tde_elliptical_orbit_ic as ic     # noqa: E402
import orbit_invariants as oi            # noqa: E402
from tde_density_snapshots import analytic_orbit, par   # noqa: E402

RUN = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc'
TDE = 'cfc_tde_wd_imbh_parabolic_tracker'
ISO = 'cfc_wd_isolated'
RT = 16.4
# label: (run dir, basename, design r_p or None, colour, linestyle)
RUNS = {
    'rp08':      (f'{RUN}/tde_parabolic_rp08_ens1', TDE, 8, '#08306b', '-'),
    'rp09':      (f'{RUN}/tde_parabolic_rp09_ens1', TDE, 9, '#2171b5', '-'),
    'rp10':      (f'{RUN}/tde_parabolic_rp10_ens1', TDE, 10, '#000000', '-'),
    'rp11':      (f'{RUN}/tde_parabolic_rp11_ens1', TDE, 11, '#41ab5d', '-'),
    'rp12':      (f'{RUN}/tde_parabolic_rp12_ens1', TDE, 12, '#fd8d3c', '-'),
    'rp14':      (f'{RUN}/tde_parabolic_rp14_ens1', TDE, 14, '#e31a1c', '-'),
    'rp16':      (f'{RUN}/tde_parabolic_rp16_ens1', TDE, 16, '#800026', '-'),
    'rp10_si10': (f'{RUN}/tde_parabolic_rp10_si10_ens1', TDE, 10, '#777777', '--'),
    'rp10_L5':   (f'{RUN}/tde_parabolic_rp10_L5_ens1', TDE, 10, '#6a3d9a', '--'),
    'iso_boost': (f'{RUN}/wd_isolated_boost_L4_si1_ens1', ISO, None, '#1b9e77', ':'),
    'iso_static': (f'{RUN}/wd_isolated_L4_si1/output-0000', ISO, None, '#a6761d', ':'),
}
SCAN = ['rp08', 'rp09', 'rp10', 'rp11', 'rp12', 'rp14', 'rp16']

# analytic bare-puncture lapse, spherically symmetric: tabulate once
_RT = np.concatenate([np.linspace(0.05, 5, 2000), np.linspace(5.01, 120, 3000)])
_AT = np.array([ic.metric_at(r, 0.0, 0.0)[1] for r in _RT])


def alpha_bh(r):
    return np.interp(r, _RT, _AT)


def cell_xy(g, m, nx, ny):
    dx, dy = (g[m, 1] - g[m, 0]) / nx, (g[m, 3] - g[m, 2]) / ny
    return np.meshgrid(g[m, 0] + (np.arange(nx) + .5) * dx,
                       g[m, 2] + (np.arange(ny) + .5) * dy)


def measure_dump(pf, af, has_bh):
    p = bc.read_binary(pf)
    a = bc.read_binary(af)
    nx, ny = p['nx1_out_mb'], p['nx2_out_mb']
    g = np.asarray(p['mb_geometry'])
    dens = np.asarray(p['mb_data']['dens'])
    rmax = float(dens.max())
    cw = cx = cy = 0.0
    for m in range(len(g)):
        d = dens[m, 0]
        c = d > 1e-2 * rmax
        if c.any():
            X, Y = cell_xy(g, m, nx, ny)
            dx = (g[m, 1] - g[m, 0]) / nx
            w = d[c] * dx * dx
            cw += w.sum(); cx += (w * X[c]).sum(); cy += (w * Y[c]).sum()
    cx, cy = cx / cw, cy / cw
    ga = np.asarray(a['mb_geometry'])
    al = np.asarray(a['mb_data']['adm_alpha'])
    dmin, dxy = np.inf, (np.nan, np.nan)
    for m in range(len(ga)):
        X, Y = cell_xy(ga, m, a['nx1_out_mb'], a['nx2_out_mb'])
        r = np.hypot(X, Y)
        da = al[m, 0] - (alpha_bh(r) if has_bh else 1.0)
        da = np.where(r > 3.0, da, np.inf) if has_bh else da
        k = np.unravel_index(np.argmin(da), da.shape)
        if da[k] < dmin:
            dmin, dxy = float(da[k]), (float(X[k]), float(Y[k]))
    if has_bh:
        t, E, L, _, _ = oi.measure(pf, af, 0.5, 1e-2)
    else:
        t, E, L = p['time'], np.nan, np.nan
    r = math.hypot(cx, cy)
    R = ic.trumpet_iso_to_areal(r) if has_bh else np.nan
    return [p['time'], rmax, cx, cy, R, dmin, dxy[0], dxy[1], E, L, len(g)]


COLS = ['t', 'rhomax', 'cx', 'cy', 'R', 'dalpha', 'dax', 'day', 'E', 'L', 'nmb']


def series(label, out):
    d, base, rp, _, _ = RUNS[label]
    pfs = sorted(glob.glob(f'{d}/bin/{base}.prim_xy.*.bin'))
    cf = f'{out}/cache/{label}.npz'
    if os.path.exists(cf):
        z = np.load(cf)
        if len(z['t']) == len(pfs):
            return {k: z[k] for k in COLS}
    rows = []
    for pf in pfs:
        af = pf.replace('.prim_xy.', '.adm_xy.')
        if os.path.exists(af):
            rows.append(measure_dump(pf, af, rp is not None))
    a = np.array(rows, dtype=float)
    res = {k: a[:, i] for i, k in enumerate(COLS)}
    np.savez(cf, **res)
    return res


def hst_rhomax(label):
    d, base, _, _, _ = RUNS[label]
    h = np.loadtxt(f'{d}/{base}.user.hst')
    return h[:, 0], h[:, 2] / h[0, 2]


def style(label):
    return dict(color=RUNS[label][3], ls=RUNS[label][4], lw=1.4, label=label)


def fig_rhomax(S, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    for lab in RUNS:
        t, r = hst_rhomax(lab)
        a1.semilogy(t, r, **style(lab))
    a1.set_xlabel('t [M]'); a1.set_ylabel(r'$\rho_{\max}/\rho_{\max}(0)$  (3D, history)')
    a1.axvspan(130, 140, color='0.9', zorder=0)
    a1.text(131, 60, 'periapsis', fontsize=8)
    a1.set_title('central density, all members'); a1.legend(fontsize=7, ncol=2)
    a1.set_xlim(0, 300); a1.grid(alpha=.3, which='both')
    for lab in SCAN + ['rp10_si10', 'rp10_L5']:
        s = S[lab]
        k = np.argmin(s['R']) if len(s['R']) else 0
        inb = np.arange(len(s['t'])) <= k            # infall branch only
        a2.semilogy(s['R'][inb], s['rhomax'][inb] / s['rhomax'][0], **style(lab))
    a2.axvline(RT, color='deepskyblue', ls=':', label='r_t')
    a2.invert_xaxis(); a2.set_xlabel('CoM areal radius R [M] (infall branch, up to periapsis)')
    a2.set_ylabel(r'$\rho_{\max}/\rho_{\max}(0)$  (z=0 slice)')
    a2.set_title('compression vs distance to the BH'); a2.grid(alpha=.3, which='both')
    a2.legend(fontsize=7, ncol=2)
    fig.savefig(f'{out}/rhomax_evolution.png', dpi=120); plt.close(fig)


def fig_dalpha(S, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    for lab in RUNS:
        s = S[lab]
        a1.plot(s['t'], -s['dalpha'], **style(lab))
        a2.loglog(s['rhomax'] / s['rhomax'][0], -s['dalpha'] / -s['dalpha'][0], '.',
                  ms=3, color=RUNS[lab][3], label=lab)
    a1.set_yscale('log')
    a1.set_xlabel('t [M]')
    a1.set_ylabel(r'$-\min_{r>3}(\alpha-\alpha_{\rm BH})$')
    a1.set_title(r"star's own lapse dip (CFC lapse minus bare-puncture trumpet lapse)")
    a1.grid(alpha=.3, which='both'); a1.legend(fontsize=7, ncol=2); a1.set_xlim(0, 300)
    x = np.logspace(0, 1.7, 20)
    a2.loglog(x, x ** (1 / 3), 'k:', lw=1, label=r'$\propto\rho^{1/3}$ (homologous, n=1.5)')
    a2.set_xlabel(r'$\rho_{\max}/\rho_{\max}(0)$'); a2.set_ylabel('lapse dip / initial')
    a2.set_title('does the metric see the compression?'); a2.grid(alpha=.3, which='both')
    a2.legend(fontsize=7, ncol=2, markerscale=3)
    fig.savefig(f'{out}/lapse_dip_evolution.png', dpi=120); plt.close(fig)


def fig_orbit(S, out):
    fig, ax = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    a0, a1, a2, a3 = ax.flat
    th = np.linspace(0, 2 * np.pi, 400)
    for lab in SCAN + ['rp10_si10', 'rp10_L5']:
        s = S[lab]
        a0.plot(s['cx'], s['cy'], **style(lab))
        a1.plot(s['t'], s['R'], **style(lab))
        a2.plot(s['t'], 100 * (s['L'] / s['L'][0] - 1), **style(lab))
        a3.plot(s['t'], s['E'] - s['E'][0], **style(lab))
    for lab in SCAN:
        o = analytic_orbit(f'{RUNS[lab][0]}/parfile.par', 300.0)
        a0.plot(o[:, 1], o[:, 2], color=RUNS[lab][3], lw=0.6, alpha=.5)
        Ra = [ic.trumpet_iso_to_areal(math.hypot(x, y)) for x, y in o[::20, 1:3]]
        a1.plot(o[::20, 0], Ra, color=RUNS[lab][3], lw=0.6, alpha=.5)
    r_t_iso = ic.trumpet_areal_to_iso(RT)
    a0.plot(r_t_iso * np.cos(th), r_t_iso * np.sin(th), color='deepskyblue', ls=':')
    a0.plot(0, 0, 'k+'); a0.set_aspect('equal'); a0.set_xlim(-40, 40); a0.set_ylim(-40, 40)
    a0.set_title('star CoM (thick) vs analytic geodesic (thin); dotted: r_t')
    a0.set_xlabel('x [M]'); a0.set_ylabel('y [M]'); a0.legend(fontsize=7, ncol=2)
    a1.axhline(RT, color='deepskyblue', ls=':'); a1.set_xlabel('t [M]')
    a1.set_ylabel('CoM areal radius R [M]'); a1.set_title('radius: simulation vs geodesic')
    a1.set_ylim(0, 40); a1.grid(alpha=.3)
    a2.set_xlabel('t [M]'); a2.set_ylabel(r'$\Delta L/L_0$ [%]')
    a2.set_title('Killing angular momentum of the core (rho > 0.5 rho_max)'); a2.grid(alpha=.3)
    a3.set_xlabel('t [M]'); a3.set_ylabel(r'$E-E_0$'); a3.grid(alpha=.3)
    a3.set_title('Killing energy of the core')
    fig.savefig(f'{out}/orbit_invariants.png', dpi=110); plt.close(fig)
    with open(f'{out}/periapsis_table.txt', 'w') as f:
        f.write(f"{'run':10s} {'design':>6s} {'R_min':>7s} {'t_min':>6s} {'t_end':>6s} "
                f"{'peak rho/rho0 (hst)':>20s}\n")
        for lab in SCAN + ['rp10_si10', 'rp10_L5']:
            s = S[lab]; k = int(np.argmin(s['R']))
            th_, rh = hst_rhomax(lab)
            j = int(np.argmax(rh))
            reached = 'yes' if k < len(s['t']) - 3 else 'no '
            f.write(f"{lab:10s} {RUNS[lab][2]:6.1f} {s['R'][k]:7.3f} {s['t'][k]:6.1f} "
                    f"{s['t'][-1]:6.1f}   {rh[j]:7.2f} at t={th_[j]:6.1f}  "
                    f"periapsis passed: {reached}\n")


def read_blocks(pf):
    p = bc.read_binary(pf)
    g = np.asarray(p['mb_geometry'])
    d = np.asarray(p['mb_data']['dens'])
    return p['time'], [(g[m, 0], g[m, 1], g[m, 2], g[m, 3], d[m, 0]) for m in range(len(g))]


def draw(ax, blocks, view, norm, cmap, outline=True):
    (x0, x1), (y0, y1) = view
    segs = []
    for (xa, xb, ya, yb, d) in sorted(blocks, key=lambda b: -(b[1] - b[0])):
        if xb < x0 or xa > x1 or yb < y0 or ya > y1:
            continue
        ax.imshow(np.ma.masked_invalid(d), origin='lower', extent=(xa, xb, ya, yb), norm=norm,
                  cmap=cmap, interpolation='nearest')
        segs += [[(xa, ya), (xb, ya)], [(xb, ya), (xb, yb)], [(xb, yb), (xa, yb)],
                 [(xa, yb), (xa, ya)]]
    if outline:
        ax.add_collection(LineCollection(segs, colors='w', linewidths=0.2, alpha=0.5))
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect('equal')


def dump_at(label, t):
    d, base, _, _, _ = RUNS[label]
    pfs = sorted(glob.glob(f'{d}/bin/{base}.prim_xy.*.bin'))
    n = min(range(len(pfs)), key=lambda i: abs(int(pfs[i].split('.')[-2]) - t))
    return pfs[n] if abs(int(pfs[n].split('.')[-2]) - t) <= 1 else None


def fig_grid(S, out, labels=('rp10', 'rp12', 'rp16'), times=(0, 60, 100, 120, 130, 140, 160, 300),
             half=2.5):
    rhoc = 3.24e-4
    norm = LogNorm(vmin=rhoc * 1e-6, vmax=rhoc * 50)
    cmap = plt.get_cmap('inferno').copy(); cmap.set_bad('k')
    fig, ax = plt.subplots(len(labels), len(times), figsize=(2.3 * len(times), 2.5 * len(labels)),
                           constrained_layout=True)
    for i, lab in enumerate(labels):
        s = S[lab]
        for j, t in enumerate(times):
            a = ax[i, j]; a.set_xticks([]); a.set_yticks([])
            pf = dump_at(lab, t)
            if pf is None:
                a.set_facecolor('0.85'); a.text(.5, .5, 'not reached', ha='center',
                                                transform=a.transAxes, fontsize=8)
                continue
            tt, blocks = read_blocks(pf)
            k = int(np.argmin(abs(s['t'] - tt)))
            cx, cy = s['cx'][k], s['cy'][k]
            draw(a, blocks, ((cx - half, cx + half), (cy - half, cy + half)), norm, cmap)
            a.set_title(f't={tt:.0f}  R={s["R"][k]:.1f}\n'
                        rf'$\rho_{{max}}/\rho_0$={s["rhomax"][k] / s["rhomax"][0]:.2f}', fontsize=7)
            if j == 0:
                a.set_ylabel(f'{lab} (beta={RT / RUNS[lab][2]:.2f})', fontsize=9)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, shrink=0.7, label=r'$\rho$ (z=0), star frame $\pm$%.1f M' % half)
    fig.savefig(f'{out}/star_frame_grid.png', dpi=110); plt.close(fig)


def fig_knot(out, cases=(('rp16', (100, 110, 120, 130)), ('rp10', (110, 125, 131, 135)))):
    """x- and y-profiles of rho through the slice maximum, one marker per cell: how many
    cells the peak spans (the resolution question behind the compression anomaly)."""
    fig, ax = plt.subplots(2, len(cases), figsize=(7 * len(cases), 8), constrained_layout=True)
    for c, (lab, times) in enumerate(cases):
        for t in times:
            pf = dump_at(lab, t)
            if pf is None:
                continue
            p = bc.read_binary(pf)
            nx, ny = p['nx1_out_mb'], p['nx2_out_mb']
            g = np.asarray(p['mb_geometry']); d = np.asarray(p['mb_data']['dens'])
            m, _, j, i = np.unravel_index(np.argmax(d), d.shape)
            X, Y = cell_xy(g, m, nx, ny)
            x0, y0, dx = X[j, i], Y[j, i], (g[m, 1] - g[m, 0]) / nx
            px, py = [], []
            for k in range(len(g)):          # gather cells on the row/column through the max
                Xk, Yk = cell_xy(g, k, nx, ny)
                if g[k, 2] <= y0 < g[k, 3]:
                    jj = int((y0 - g[k, 2]) / (g[k, 3] - g[k, 2]) * ny)
                    px += list(zip(Xk[jj], d[k, 0, jj]))
                if g[k, 0] <= x0 < g[k, 1]:
                    ii = int((x0 - g[k, 0]) / (g[k, 1] - g[k, 0]) * nx)
                    py += list(zip(Yk[:, ii], d[k, 0, :, ii]))
            for row, prof, c0 in ((0, px, x0), (1, py, y0)):
                q = np.array(sorted(prof))
                w = abs(q[:, 0] - c0) < 1.5
                ax[row, c].semilogy(q[w, 0] - c0, q[w, 1], '.-', ms=4, lw=0.8,
                                    label=f't={p["time"]:.0f} (dx={dx:.4f})')
        for row, n in ((0, 'x'), (1, 'y')):
            a = ax[row, c]
            a.axhline(3.24e-4, color='k', ls=':', lw=0.8)
            a.set_xlabel(f'{n} - {n}(rho_max) [M]   (one marker per cell)')
            a.set_ylabel(r'$\rho$'); a.set_title(f'{lab}: {n}-profile through the density peak')
            a.grid(alpha=.3, which='both'); a.legend(fontsize=8); a.set_ylim(1e-7, 2e-2)
    fig.savefig(f'{out}/peak_profiles.png', dpi=110); plt.close(fig)


def animate(label, S, out, every=2, half=2.5, G=40.0):
    d, base, rp, _, _ = RUNS[label]
    pfs = sorted(glob.glob(f'{d}/bin/{base}.prim_xy.*.bin'))[::every]
    s = S[label]
    rhoc = 3.24e-4
    norm = LogNorm(vmin=rhoc * 1e-10, vmax=rhoc * 50)
    cmap = plt.get_cmap('inferno').copy(); cmap.set_bad('k')
    orb = analytic_orbit(f'{d}/parfile.par', 300.0)
    th = np.linspace(0, 2 * np.pi, 300)
    r_t_iso = ic.trumpet_areal_to_iso(RT)
    ht, hr = hst_rhomax(label)
    fig = plt.figure(figsize=(15, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1], wspace=0.25)
    ag, al, ar = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    def frame(i):
        for a in (ag, al, ar):
            a.cla()
        tt, blocks = read_blocks(pfs[i])
        k = int(np.argmin(abs(s['t'] - tt)))
        cx, cy = s['cx'][k], s['cy'][k]
        draw(ag, blocks, ((-G, G), (-G, G)), norm, cmap)
        ag.plot(orb[:, 1], orb[:, 2], 'c--', lw=0.7)
        ag.plot(r_t_iso * np.cos(th), r_t_iso * np.sin(th), color='deepskyblue', ls=':', lw=0.8)
        ag.add_patch(plt.Rectangle((cx - half, cy - half), 2 * half, 2 * half, fill=False,
                                   ec='lime', lw=0.7))
        ag.set_title(f'{label}: global   t = {tt:6.1f} M', fontsize=9)
        draw(al, blocks, ((cx - half, cx + half), (cy - half, cy + half)), norm, cmap)
        al.set_title(f'star frame  R_areal = {s["R"][k]:.2f}', fontsize=9)
        ar.semilogy(ht, hr, color='k', lw=1)
        ar.axvline(tt, color='r', lw=1)
        ar.set_xlim(0, 300); ar.set_xlabel('t [M]')
        ar.set_ylabel(r'$\rho_{\max}/\rho_{\max}(0)$'); ar.grid(alpha=.3, which='both')
        ar.set_title(f'beta = {RT / rp:.2f}', fontsize=9)
        return []

    ani = animation.FuncAnimation(fig, frame, frames=len(pfs), blit=False)
    ani.save(f'{out}/anim_{label}.gif', writer=animation.PillowWriter(fps=8), dpi=70)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=f'{RUN}/plots/tde_parabolic_ens1')
    ap.add_argument('--anim', default='rp10,rp16')
    ap.add_argument('--anim-every', type=int, default=2)
    ap.add_argument('--no-anim', action='store_true')
    ap.add_argument('--only-anim', action='store_true')
    a = ap.parse_args()
    os.makedirs(f'{a.out}/cache', exist_ok=True)
    S = {}
    for lab in RUNS:
        S[lab] = series(lab, a.out)
        print(f'{lab}: {len(S[lab]["t"])} dumps, t_end={S[lab]["t"][-1]:.1f}', flush=True)
    if not a.only_anim:
        fig_rhomax(S, a.out); fig_dalpha(S, a.out); fig_orbit(S, a.out); fig_grid(S, a.out)
        fig_knot(a.out)
        print('static figures done', flush=True)
    if not a.no_anim:
        for lab in a.anim.split(','):
            animate(lab, S, a.out, every=a.anim_every)
            print(f'anim_{lab}.gif done', flush=True)


if __name__ == '__main__':
    main()
