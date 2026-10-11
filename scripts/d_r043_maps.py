#!/usr/bin/env python3
"""R-043 item 1 (diagnostic D): item 6 L4 restarted at t = 60.7 in three arms.

  (a) current binary + <mhd> dual_energy_diag switch maps (mhd_de_diag slices, 0.5 M)
  (b) old-fill binary; (c) shock flag off; L4/L3 = the item-6 runs themselves.

Series (prim_xy, tde_t11_surface_diag.series): dense mean K/K0 (Kw01) and core K per arm.
Maps (arm a): for dense cells (branch > 0) in the xy and yz slices, the branch census
and, split by branch, the K/K0 - 1 of the cell, the energy solution's K_en/K_tr - 1,
the K resync and the entropy-branch dtau/D; plus the tau source terms. Figure: per-time
maps of K/K0, branch, sensor/threshold, K_en/K_tr - 1, at the first, middle and last dump.

    d_r043_maps.py [--out DIR]
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.colors import ListedColormap, TwoSlopeNorm   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tde_t11_surface_diag as T                  # noqa: E402
from tde_ensemble_plots import bc                 # noqa: E402

RUN = T.RUN
B = 'cfc_tde_wd_imbh_parabolic_tracker'
ARMS = {'a': 'D_a_dediag', 'b': 'D_b_oldfill', 'c': 'D_c_noshock'}
RHOC, K0, GAM = 3.24e-4, 0.012346626297, 5.0 / 3.0
BRANCH = ['not dense', 'entropy', 'entropy (guard)', 'entropy (floor/fail)', 'energy+resync',
          'energy, no resync']


_TS = {}


def load(d, plane, n):
    # prim_xy keeps the restart's output schedule; the new blocks start at the restart time,
    # so xy pairs with the nearest prim dump (offset <= 0.25 M, recorded); yz matches exactly
    q = bc.read_binary(f'{d}/bin/{B}.dediag_{plane}.{n:05d}.bin')
    fp = sorted(glob.glob(f'{d}/bin/{B}.prim_{plane}.*.bin'))
    ts = [bc.read_binary(f)['time'] for f in fp] if (d, plane) not in _TS else _TS[(d, plane)]
    _TS[(d, plane)] = ts
    w = bc.read_binary(fp[int(np.argmin(np.abs(np.array(ts) - q['time'])))])
    g = np.asarray(q['mb_geometry'])
    rho = np.asarray(w['mb_data']['dens'])[:, 0]
    P = np.asarray(w['mb_data']['press'])[:, 0]
    if rho.shape != np.asarray(q['mb_data']['de_branch'])[:, 0].shape:   # a regrid in between
        rho = np.full(np.asarray(q['mb_data']['de_branch'])[:, 0].shape, np.nan); P = rho
        g = np.asarray(q['mb_geometry'])
    out = dict(t=q['time'], dt_prim=w['time'] - q['time'], g=g, nx=w['nx1_out_mb'], rho=rho, K=P / rho**GAM / K0)
    for k in ('de_branch', 'de_sensor', 'de_shock', 'de_ken_ktr', 'de_dk_resync', 'de_dtau',
              'de_src_kS', 'de_src_dalpha'):
        out[k] = np.asarray(q['mb_data'][k])[:, 0]
    return out


def census(s):
    br = np.rint(s['de_branch']).astype(int)
    dense = br > 0                       # D > dual_energy_rho_switch, as the switch sees it
    row = [s['t'], dense.sum()]
    for b in range(1, 6):
        row.append(((br == b) & dense).sum())
    for b in (1, 4):
        sel = (br == b) & dense
        row += [np.mean(s['K'][sel] - 1) if sel.any() else np.nan,
                np.mean(s['de_ken_ktr'][sel]) if sel.any() else np.nan,
                np.percentile(s['de_ken_ktr'][sel], 99) if sel.any() else np.nan]
    e = (br == 4) & dense
    row += [np.mean(s['de_dk_resync'][e]) if e.any() else np.nan,
            np.percentile(s['de_dk_resync'][e], 99) if e.any() else np.nan]
    en = (br == 1) & dense
    row += [np.mean(s['de_dtau'][en]) if en.any() else np.nan,
            np.mean(np.abs(s['de_src_kS'][dense])), np.mean(np.abs(s['de_src_dalpha'][dense])),
            (s['de_shock'][dense] > 0.5).mean()]
    return row


HDR = ('t, n_dense, n_ent, n_guard, n_floor, n_energy, n_energy_noresync, '
       'ent:<K-1>, ent:<Ken/Ktr-1>, ent:p99(Ken/Ktr-1), en:<K-1>, en:<Ken/Ktr-1>, en:p99(Ken/Ktr-1), '
       'en:<dK resync>, en:p99(dK resync), ent:<dtau/D>, <|aKS|>, <|S.dalpha|>, flagged fraction')


def draw(ax, s, f, view, **kw):
    g, nx = s['g'], s['nx']
    for m in range(len(g)):
        if g[m, 1] < view[0][0] or g[m, 0] > view[0][1] or g[m, 3] < view[1][0] or g[m, 2] > view[1][1]:
            continue
        im = ax.imshow(f[m], origin='lower', extent=(g[m, 0], g[m, 1], g[m, 2], g[m, 3]), **kw)
    ax.set_xlim(*view[0]); ax.set_ylim(*view[1])
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{RUN}/plots/phase2_m1')
    a = ap.parse_args()
    rows = []
    # series per arm vs the item-6 runs
    S = {}
    for lab, d in list(ARMS.items()) + [('L4', 'T6b_rp10_comov_bhL4'), ('L3', 'T6b_rp10_comov_bhL3')]:
        if not glob.glob(f'{RUN}/{d}/bin/{B}.prim_xy.*.bin'):
            continue
        T.RUNS[f'D_{lab}'] = (f'{RUN}/{d}', B, 10, 'k', '-')
        S[lab] = T.series(f'D_{lab}', f'{RUN}/plots/dual_energy')
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for lab, s in S.items():
        sel = (s['t'] >= 58) & (s['t'] <= 80)
        ax[0].plot(s['t'][sel], s['Kw01'][sel], 'o-' if lab in ARMS else '--', ms=3, label=lab)
        ax[1].plot(s['t'][sel], s['Kmax'][sel], 'o-' if lab in ARMS else '--', ms=3, label=lab)
        rows.append(f'{lab}: ' + ', '.join(f't={t:.1f} Kw01 {k:.4f}' for t, k in zip(s['t'][sel], s['Kw01'][sel])))
    ax[0].set_title('dense mean K/K0 (ρ > 0.1 ρ0, z = 0)'); ax[1].set_title('K/K0 at ρ_max')
    for x in ax:
        x.set_xlabel('t [M]'); x.grid(alpha=0.3); x.legend()
    fig.suptitle('R-043 diagnostic D: (a) current, (b) old fill, (c) shock flag off; L4/L3 item 6')
    fig.savefig(f'{a.out}/r043_D_series.png', dpi=80)
    # maps (arm a)
    d = f'{RUN}/{ARMS["a"]}'
    for plane in ('xy', 'yz'):
        ns = sorted(int(f.split('.')[-2]) for f in glob.glob(f'{d}/bin/{B}.dediag_{plane}.*.bin'))
        if not ns:
            continue
        C = np.array([census(load(d, plane, n)) for n in ns])
        np.savetxt(f'{a.out}/r043_D_census_{plane}.txt', C, header=HDR, fmt='%.5g')
        rows.append(f'census {plane} ({HDR}):')
        rows += ['  ' + ' '.join(f'{v:.4g}' for v in r) for r in C]
        pick = [ns[0], ns[len(ns) // 2], ns[-1]]
        fig, ax = plt.subplots(len(pick), 5, figsize=(22, 4.3 * len(pick)), constrained_layout=True)
        cm = ListedColormap(['#222222', '#4575b4', '#91bfdb', '#fee090', '#d73027', '#7f0000'])
        for r, n in enumerate(pick):
            s = load(d, plane, n)
            m, j, i = np.unravel_index(np.argmax(np.where(np.isfinite(s['rho']), s['rho'], -1) if np.isfinite(s['rho']).any()
                                                 else s['de_branch'] * 0 + (s['de_branch'] > 0.5)), s['rho'].shape)
            g = s['g']; nx = s['nx']
            xc = g[m, 0] + (i + .5) * (g[m, 1] - g[m, 0]) / nx; yc = g[m, 2] + (j + .5) * (g[m, 3] - g[m, 2]) / nx
            view = ((xc - 1.6, xc + 1.6), (yc - 1.6, yc + 1.6))
            dense = s['de_branch'] > 0.5
            K = np.where(dense, s['K'], np.nan)
            im = draw(ax[r, 0], s, K, view, vmin=0.9, vmax=2.0, cmap='magma'); plt.colorbar(im, ax=ax[r, 0])
            ax[r, 0].set_title(f'{plane} t={s["t"]:.2f} K/K0 (ρ, K at {s["dt_prim"]:+.2f})')
            br = np.where(dense, s['de_branch'], np.nan)
            im = draw(ax[r, 1], s, br, view, vmin=-0.5, vmax=5.5, cmap=cm, interpolation='nearest')
            cb = plt.colorbar(im, ax=ax[r, 1], ticks=range(6)); cb.ax.set_yticklabels(BRANCH, fontsize=6)
            ax[r, 1].set_title('branch')
            im = draw(ax[r, 2], s, np.where(dense, np.log10(s['de_sensor'] + 1e-6), np.nan), view, vmin=-2, vmax=1,
                      cmap='viridis'); plt.colorbar(im, ax=ax[r, 2]); ax[r, 2].set_title('log10 sensor/threshold')
            im = draw(ax[r, 3], s, np.where(dense, s['de_ken_ktr'], np.nan), view, norm=TwoSlopeNorm(0, -0.05, 0.3),
                      cmap='RdBu_r'); plt.colorbar(im, ax=ax[r, 3]); ax[r, 3].set_title('K_energy/K_tracer − 1')
            im = draw(ax[r, 4], s, np.where(dense, s['de_src_kS'], np.nan), view, cmap='RdBu_r',
                      norm=TwoSlopeNorm(0)); plt.colorbar(im, ax=ax[r, 4]); ax[r, 4].set_title('α K_ij S^ij')
        fig.savefig(f'{a.out}/r043_D_maps_{plane}.png', dpi=60)
    with open(f'{a.out}/r043_D.txt', 'w') as f:
        f.write('\n'.join(rows) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
