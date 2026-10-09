#!/usr/bin/env python3
"""T-11 entropy test and stellar-surface diagnostics on the z=0 bin slices
(research requests R-011, R-015). Per dump:

  K at the density maximum, K = P / rho^(5/3), against K_0 = kappa of the IC;
  slice rho-weighted K (and its min) over rho > 3 rho_0, and over rho > 0.1 rho_0;
  knot mass: rho(r) azimuthally averaged in the slice about the density max,
      M(<r) = sum 4 pi r^2 rho dr out to where the profile falls below 3 rho_0
      (and to 1 R_*) -- assumes spherical symmetry about the max, so it is an
      estimate; validated on the t = 0 star;
  surface: slice-mass fraction at distance > R_* from the core CoM (rho >
      0.5 rho_max), its rho-weighted speed relative to the core velocity and
      its rho-weighted radial velocity, and the dipole moments <cos> of that
      mass along the core velocity and along the BH direction.
Slice masses are rho dA on z = 0 (coordinate rho; no W psi^6), i.e. proxies.

    tde_t11_surface_diag.py [--out DIR] [--plots] [labels ...]
"""
import argparse
import glob
import os
import sys
import types

import numpy as np

sys.modules.setdefault('h5py', types.ModuleType('h5py'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'vis', 'python'))
sys.path.insert(0, HERE)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.colors import LogNorm               # noqa: E402
import bin_convert as bc                 # noqa: E402
from tde_ensemble_plots import (RUNS, RUN, ISO, cell_xy, hst_rhomax, style, dump_at,  # noqa: E402
                                read_blocks, draw)

RUNS['iso_static_L5'] = (f'{RUN}/wd_isolated_L5_si1/output-0000', ISO, None,
                         '#e6ab02', ':')
K0 = 1.2346626297e-2
RHO0 = 3.174214651708447e-4      # IC central density (hst rho-max at t = 0)
RSTAR = 0.677                    # isotropic radius of the IC star, ~10.8 L4 cells
GAM = 5.0 / 3.0


def flatten(pf):
    p = bc.read_binary(pf)
    g = np.asarray(p['mb_geometry'])
    nx, ny = p['nx1_out_mb'], p['nx2_out_mb']
    cols = {k: [] for k in ('x', 'y', 'dA', 'rho', 'P', 'vx', 'vy')}
    md = p['mb_data']
    for m in range(len(g)):
        X, Y = cell_xy(g, m, nx, ny)
        dx = (g[m, 1] - g[m, 0]) / nx
        cols['x'].append(X.ravel()); cols['y'].append(Y.ravel())
        cols['dA'].append(np.full(X.size, dx * dx))
        cols['rho'].append(np.asarray(md['dens'])[m, 0].ravel())
        cols['P'].append(np.asarray(md['press'])[m, 0].ravel())
        cols['vx'].append(np.asarray(md['velx'])[m, 0].ravel())
        cols['vy'].append(np.asarray(md['vely'])[m, 0].ravel())
    return p['time'], {k: np.concatenate(v) for k, v in cols.items()}


def knot_mass(c, x0, y0, dr):
    r = np.hypot(c['x'] - x0, c['y'] - y0)
    sel = r < 2.0 * RSTAR
    edges = np.arange(0.0, 2.0 * RSTAR + dr, dr)
    w, _ = np.histogram(r[sel], edges, weights=c['rho'][sel] * c['dA'][sel])
    a, _ = np.histogram(r[sel], edges, weights=c['dA'][sel])
    prof = np.where(a > 0, w / np.maximum(a, 1e-30), 0.0)
    rc = 0.5 * (edges[1:] + edges[:-1])
    mcum = np.cumsum(4 * np.pi * rc**2 * prof * dr)
    below = np.nonzero(prof < 3 * RHO0)[0]
    i3 = below[0] if len(below) else len(prof)
    m3 = mcum[i3 - 1] if i3 > 0 else 0.0
    m1 = mcum[np.searchsorted(rc, RSTAR)]
    return m3, rc[i3 - 1] if i3 > 0 else 0.0, m1


def measure(pf, has_bh):
    t, c = flatten(pf)
    rho, P = c['rho'], c['P']
    K = P / rho**GAM
    i = np.argmax(rho)
    rmax, kmax = rho[i], K[i]
    out = [t, rmax / RHO0, kmax / K0]
    for thr in (3.0, 0.1):
        s = rho > thr * RHO0
        if s.any():
            w = rho[s] * c['dA'][s]
            out += [(w * K[s]).sum() / w.sum() / K0, K[s].min() / K0]
        else:
            out += [np.nan, np.nan]
    dr = np.sqrt(c['dA'][i])
    out += list(knot_mass(c, c['x'][i], c['y'][i], dr))
    # core CoM and velocity
    s = rho > 0.5 * rmax
    w = rho[s] * c['dA'][s]
    cx, cy = (w * c['x'][s]).sum() / w.sum(), (w * c['y'][s]).sum() / w.sum()
    vx, vy = (w * c['vx'][s]).sum() / w.sum(), (w * c['vy'][s]).sum() / w.sum()
    dx, dy = c['x'] - cx, c['y'] - cy
    r = np.hypot(dx, dy)
    wall = rho * c['dA']
    star = rho > 1e-12           # well above the 1e-24 floor
    tot = wall[star].sum()
    pf_ = star & (r > RSTAR)
    wp = wall[pf_]
    frac = wp.sum() / tot
    if wp.sum() > 0:
        ux, uy = c['vx'][pf_] - vx, c['vy'][pf_] - vy
        spd = (wp * np.hypot(ux, uy)).sum() / wp.sum()
        vr = (wp * (ux * dx[pf_] + uy * dy[pf_]) / r[pf_]).sum() / wp.sum()
        nx_, ny_ = dx[pf_] / r[pf_], dy[pf_] / r[pf_]
        vn = np.hypot(vx, vy)
        cv = (wp * (nx_ * vx + ny_ * vy) / vn).sum() / wp.sum() if vn > 1e-6 else np.nan
        if has_bh:
            bn = np.hypot(cx, cy)
            bx, by = -cx / bn, -cy / bn
            cb = (wp * (nx_ * bx + ny_ * by)).sum() / wp.sum()
            q = (wp * (2 * (nx_ * bx + ny_ * by)**2 - 1)).sum() / wp.sum()
        else:
            cb = q = np.nan
    else:
        spd = vr = cv = cb = q = np.nan
    out += [frac, spd, vr, cv, cb, q, np.hypot(vx, vy)]
    return out


COLS = ['t', 'rhomax', 'Kmax', 'Kw3', 'Kmin3', 'Kw01', 'Kmin01', 'Mknot', 'rknot',
        'M1R', 'puff', 'puff_v', 'puff_vr', 'puff_cos_v', 'puff_cos_bh',
        'puff_q_bh', 'vcore']


def series(label, out, every=1, tmax=None):
    d, base, rp, _, _ = RUNS[label]
    pfs = sorted(glob.glob(f'{d}/bin/{base}.prim_xy.*.bin'))[::every]
    cf = f'{out}/{label}.npz'
    if os.path.exists(cf):
        z = np.load(cf)
        if len(z['t']) == len(pfs):
            return {k: z[k] for k in COLS}
    rows = [measure(pf, rp is not None) for pf in pfs]
    a = np.array(rows, dtype=float)
    res = {k: a[:, i] for i, k in enumerate(COLS)}
    np.savez(cf, **res)
    return res


def fig_entropy(S, out, labs=('rp16', 'rp12', 'rp10', 'rp10_L5', 'iso_boost')):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for lab in labs:
        if lab not in S:
            continue
        s = S[lab]
        ax[0].semilogy(s['t'], s['Kmax'], **style(lab))
        ax[1].semilogy(s['t'], np.maximum(s['Kmin01'], 1e-20), **style(lab))
        ax[2].semilogy(s['t'], s['rhomax'], **style(lab))
    ax[0].set_ylabel(r'$K/K_0$ at the density maximum')
    ax[1].set_ylabel(r'min $K/K_0$ over $\rho > 0.1\rho_0$')
    ax[2].set_ylabel(r'$\rho_{max}/\rho_0$ (slice)')
    for a in ax:
        a.set_xlabel('t [M]'); a.axhline(1, color='0.6', lw=0.8); a.grid(alpha=0.3)
    ax[0].set_ylim(1e-3, 3); ax[1].set_ylim(1e-20, 3)
    ax[0].legend(fontsize=8)
    fig.suptitle('T-11 entropy test: K = P / rho^(5/3), z=0 slice. '
                 'K at the floor is ~1e-18 K_0 (P = rho T_floor)')
    fig.savefig(f'{out}/t11_entropy.png', dpi=110); plt.close(fig)


def fig_kmaps(out, lab='rp16', times=(40, 60, 80, 100, 110, 120), half=1.5):
    fig, ax = plt.subplots(2, len(times), figsize=(2.8 * len(times), 5.6), constrained_layout=True)
    nr = LogNorm(vmin=RHO0 * 1e-4, vmax=RHO0 * 30)
    nk = LogNorm(vmin=1e-2, vmax=3)
    for j, t in enumerate(times):
        pf = dump_at(lab, t)
        p = bc.read_binary(pf)
        g = np.asarray(p['mb_geometry'])
        d = np.asarray(p['mb_data']['dens']); P = np.asarray(p['mb_data']['press'])
        K = np.where(d > 1e-4 * RHO0, P / d**GAM / K0, np.nan)
        m, _, jj, ii = np.unravel_index(np.argmax(d), d.shape)
        X, Y = cell_xy(g, m, p['nx1_out_mb'], p['nx2_out_mb'])
        view = ((X[jj, ii] - half, X[jj, ii] + half), (Y[jj, ii] - half, Y[jj, ii] + half))
        draw(ax[0, j], [(g[k, 0], g[k, 1], g[k, 2], g[k, 3], d[k, 0]) for k in range(len(g))],
             view, nr, 'inferno', outline=False)
        draw(ax[1, j], [(g[k, 0], g[k, 1], g[k, 2], g[k, 3], K[k, 0]) for k in range(len(g))],
             view, nk, 'viridis', outline=False)
        ax[0, j].set_title(f'{lab} t={p["time"]:.0f}', fontsize=9)
        for a in ax[:, j]:
            a.set_xticks([]); a.set_yticks([])
    fig.colorbar(plt.cm.ScalarMappable(norm=nr, cmap='inferno'), ax=ax[0], label=r'$\rho$')
    fig.colorbar(plt.cm.ScalarMappable(norm=nk, cmap='viridis'), ax=ax[1],
                 label=r'$K/K_0$ ($\rho>10^{-4}\rho_0$)')
    fig.savefig(f'{out}/t11_kmaps_{lab}.png', dpi=110); plt.close(fig)


def fig_l4l5(out):
    fig, a = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for lab in ('rp10', 'rp10_L5', 'iso_boost', 'iso_static', 'iso_static_L5'):
        t, r = hst_rhomax(lab)
        k = t <= 76
        a.plot(t[k], r[k], **style(lab))
    a.set_xlabel('t [M]'); a.set_ylabel(r'$\rho_{max}(t)/\rho_{max}(0)$ (hst, 3D)')
    a.grid(alpha=0.3); a.legend()
    a.set_title('r_p=10: L4 vs L5, with static (L4, L5) and boosted (L4) isolated stars; all si=1')
    fig.savefig(f'{out}/rp10_L4_vs_L5.png', dpi=110); plt.close(fig)


def fig_puff(S, out, labs=('iso_static', 'iso_static_L5', 'iso_boost', 'rp10', 'rp10_L5', 'rp16')):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for lab in labs:
        if lab not in S:
            continue
        s = S[lab]; k = s['t'] <= 100
        ax[0].semilogy(s['t'][k], s['puff'][k], **style(lab))
        ax[1].plot(s['t'][k], s['puff_v'][k], **style(lab))
        ax[2].plot(s['t'][k], s['puff_cos_v'][k], **style(lab))
        if RUNS[lab][2] is not None:
            ax[2].plot(s['t'][k], s['puff_q_bh'][k], color=RUNS[lab][3], ls='-.', lw=1)
    ax[0].set_ylabel(r'slice-mass fraction beyond $R_*$ of the core CoM')
    ax[1].set_ylabel('rho-weighted |v - v_core| of that gas')
    ax[2].set_ylabel(r'<cos> along v_core (solid/dash); <cos 2$\phi$> to BH (dash-dot)')
    for a in ax:
        a.set_xlabel('t [M]'); a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.suptitle(f'Stellar-surface puffing (z=0 slice, R_* = {RSTAR} M)')
    fig.savefig(f'{out}/surface_puff.png', dpi=110); plt.close(fig)


def fig_envelope(S, out, cases=(('iso_static', 40), ('iso_boost', 60), ('rp10', 60),
                                 ('rp16', 60)), half=1.5):
    norm = LogNorm(vmin=RHO0 * 1e-6, vmax=RHO0 * 2)
    cmap = plt.get_cmap('inferno').copy(); cmap.set_bad('k')
    fig, ax = plt.subplots(1, len(cases), figsize=(4.2 * len(cases), 4.4), constrained_layout=True)
    for a, (lab, t) in zip(ax, cases):
        pf = dump_at(lab, t)
        tt, blocks = read_blocks(pf)
        s = S[lab]; k = int(np.argmin(abs(s['t'] - tt)))
        c = flatten(pf)[1]
        sel = c['rho'] > 0.5 * c['rho'].max()
        w = c['rho'][sel]
        cx, cy = (w * c['x'][sel]).sum() / w.sum(), (w * c['y'][sel]).sum() / w.sum()
        vx, vy = (w * c['vx'][sel]).sum() / w.sum(), (w * c['vy'][sel]).sum() / w.sum()
        draw(a, blocks, ((cx - half, cx + half), (cy - half, cy + half)), norm, cmap)
        a.add_patch(plt.Circle((cx, cy), RSTAR, fill=False, color='c', lw=0.8, ls='--'))
        vn = np.hypot(vx, vy)
        if vn > 1e-3:
            a.annotate('', (cx + 1.2 * vx / vn, cy + 1.2 * vy / vn), (cx, cy),
                       arrowprops=dict(color='w', arrowstyle='->'))
        if RUNS[lab][2] is not None:
            bn = np.hypot(cx, cy)
            a.annotate('', (cx - 1.2 * cx / bn, cy - 1.2 * cy / bn), (cx, cy),
                       arrowprops=dict(color='lime', arrowstyle='->'))
        a.set_title(f'{lab} t={tt:.0f}  puff={s["puff"][k]:.1e}', fontsize=9)
        a.set_xticks([]); a.set_yticks([])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.8,
                 label=r'$\rho$ (z=0); white arrow = core velocity, green = BH; '
                       r'dashed = $R_*$')
    fig.savefig(f'{out}/envelope_t60.png', dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{RUN}/plots/tde_parabolic_ens1/t11_surface')
    ap.add_argument('labels', nargs='*',
                    default=['rp10', 'rp12', 'rp16', 'rp10_L5', 'iso_static',
                             'iso_static_L5', 'iso_boost'])
    ap.add_argument('--plots', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    S = {}
    for lab in a.labels:
        S[lab] = s = series(lab, a.out)
        print(f'# {lab}', flush=True)
        print('  ' + ' '.join(f'{k:>10s}' for k in COLS))
        for j in range(len(s['t'])):
            print('  ' + ' '.join(f'{s[k][j]:10.4g}' for k in COLS))
    if a.plots:
        fig_entropy(S, a.out); fig_kmaps(a.out); fig_kmaps(a.out, 'rp12', (60, 80, 90, 100, 110, 125))
        fig_l4l5(a.out); fig_puff(S, a.out); fig_envelope(S, a.out)


if __name__ == '__main__':
    main()
