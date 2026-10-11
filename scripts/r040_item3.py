#!/usr/bin/env python3
"""R-040 item 3 (analysis, research NEXT_TASKS_R040.md section 3), from restarts (3D) and z = 0 slices.

(a) rp10 remnant history: GRR13-style self-bound mass (scripts/remnant.py) at every restart of
    Q1, Q6, V5-L4, V5-L5, plus a 1 M-cadence 2D proxy from prim_xy slices: rho_max, and the slice
    "column" mass sum(rho dA) within 2 R* of the density max (a z = 0 slice proxy, not a mass).
(b) Q1 vs Q6 (box check): dense-mass fraction per refinement level (above the root), blocks per
    level within 3 and 10 R* of the density max, the orbit of the densest gas (E = -u_t - 1 and
    L_z = x u_y - y u_x, mass-weighted over rho > 0.1 rho_c), and the bound mass (E < 0) per
    radius-from-the-star bin.
(c) mass-weighted mean <K/K0 - 1> over the dense gas vs t, and its time integral to periapsis,
    for Q4 vs V5-L5 (V5-L4, Q1 for reference).

Per-restart results are cached in OUT/r040_item3_cache/ (one npz per restart).

    r040_item3.py [--out DIR] [--nproc 24]
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                   # noqa: E402
from matplotlib.colors import LogNorm              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tde_t11_surface_diag as _T                  # noqa: E402,F401 (sets up bc)
from tde_ensemble_plots import bc, draw            # noqa: E402
import tde_dmde as D                               # noqa: E402
import remnant as REM                              # noqa: E402
import rst_tree as RT                              # noqa: E402

C = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc'
BASE = 'cfc_tde_wd_imbh_parabolic_tracker'
GAM = 5.0 / 3.0
MSTAR = 7.47e-5
# label -> (dirs, colour, linestyle, M_BH / 1e4 Msun, periapsis time (rho_max track))
RUNS = {
    'Q1':   ((f'{C}/Q1_V5L4_fix_t300',), '#d95f02', '-', 1.0),
    'Q6':   ((f'{C}/Q6_V5L4_box128',), '#7570b3', '-', 1.0),
    'V5L4': ((f'{C}/de_V5_rp10_L4',), '#e6ab02', ':', 1.0),
    'V5L5': ((f'{C}/de_V5_rp10_L5', f'{C}/de_V5_rp10_L5_mgfix'), '#666666', '--', 1.0),
    'Q4':   ((f'{C}/Q4_P1_2e4_L6', f'{C}/Q4_P1_2e4_L6_rst14'), '#e7298a', '-', 2.0),
}
RHOC0, K00 = 3.24e-4, 1.2346626297e-2


def consts(lab):
    s = RUNS[lab][3]
    return dict(rhoc=RHOC0 * s**2, k0=K00 * s**(-4.0 / 3.0), rstar=0.677 / s, mstar=MSTAR / s)


def rsts(lab):
    out = {}
    for d in RUNS[lab][0]:                         # later directories override equal times
        for f in glob.glob(f'{d}/rst/*.rst'):
            out[round(RT.read_tree(f)['time'], 1)] = f
    return [out[t] for t in sorted(out)]


def slices(lab):
    out = {}
    for d in RUNS[lab][0]:
        out.update({int(f.split('.')[-2]): f for f in glob.glob(f'{d}/bin/{BASE}.prim_xy.*.bin')})
    return [out[n] for n in sorted(out)]


def one_rst(args):
    lab, f, cdir = args
    key = f'{cdir}/{lab}_{os.path.basename(os.path.dirname(os.path.dirname(f)))}_{os.path.basename(f)}.npz'
    if os.path.exists(key):
        return key
    k = consts(lab)
    c = D.cells(f, rcut=3.0)
    tr = RT.read_tree(f)
    dense = c['rho'] > 0.1 * k['rhoc']
    m = c['dM']
    Mb, xc, vc, sel = REM.bound_mass(c)
    i = np.argmax(c['rho'])
    xs = np.array([c['x'][i], c['y'][i], c['z'][i]])
    rs = np.sqrt((c['x'] - xs[0])**2 + (c['y'] - xs[1])**2 + (c['z'] - xs[2])**2)
    kx = c['P'] / c['rho']**GAM / k['k0'] - 1.0
    Lz = c['x'] * c['uy'] - c['y'] * c['ux']
    # levels: dense-mass fraction per level; blocks per level near the star
    levm = np.array([m[dense & (c['lev'] == L)].sum() for L in range(8)])
    ll = tr['ll']; root = tr['root']
    w = np.array(tr['size'][3:6]) - np.array(tr['size'][:3])
    ext = w[None, :] / 2.0**ll[:, 3:4]
    cen = np.array(tr['size'][:3])[None, :] + (ll[:, :3] + 0.5) * ext
    db = np.linalg.norm(cen - xs[None, :], axis=1)
    nb3 = np.array([((ll[:, 3] - root == L) & (db < 3 * k['rstar'] + ext[:, 0])).sum() for L in range(8)])
    nb10 = np.array([((ll[:, 3] - root == L) & (db < 10 * k['rstar'] + ext[:, 0])).sum() for L in range(8)])
    nbl = np.array([(ll[:, 3] - root == L).sum() for L in range(8)])
    # bound mass per radius-from-the-star bin
    redges = np.array([0, 0.5, 1, 1.5, 2, 3, 5, 10, 20, 1e9]) * k['rstar']
    mb_r = np.histogram(rs, redges, weights=m * (c['E'] < 0))[0]
    mt_r = np.histogram(rs, redges, weights=m)[0]
    np.savez(key, t=c['t'], Mb=Mb / k['mstar'], xc=xc, vc=vc,
             rho_rem=c['rho'][sel].max() / k['rhoc'] if sel.any() else 0.0,
             rho_max=c['rho'].max() / k['rhoc'], xs=xs,
             fb=m[c['E'] < 0].sum() / m.sum(), Mtot=m.sum() / k['mstar'],
             Md=m[dense].sum() / k['mstar'],
             Ed=np.average(c['E'][dense], weights=m[dense]) if dense.any() else np.nan,
             Ld=np.average(Lz[dense], weights=m[dense]) if dense.any() else np.nan,
             Erem=np.average(c['E'][sel], weights=m[sel]) if sel.any() else np.nan,
             Lrem=np.average(Lz[sel], weights=m[sel]) if sel.any() else np.nan,
             kmean=np.average(kx[dense], weights=m[dense]) if dense.any() else np.nan,
             w5=m[dense & (kx > 0.05)].sum() / max(m[dense].sum(), 1e-300),
             kmean_rem=np.average(kx[sel], weights=m[sel]) if sel.any() else np.nan,
             levm=levm, nb3=nb3, nb10=nb10, nbl=nbl, root=root, nmb=tr['nmb'],
             redges=redges / k['rstar'], mb_r=mb_r / k['mstar'], mt_r=mt_r / k['mstar'])
    print(f'{lab} t={c["t"]:6.1f} Mb={Mb / k["mstar"]:.3f} kmean={np.load(key)["kmean"]:+.4f}', flush=True)
    return key


def one_slice(args):
    lab, f = args
    k = consts(lab)
    p = bc.read_binary(f)
    g = np.asarray(p['mb_geometry']); nx = p['nx1_out_mb']
    d = np.asarray(p['mb_data']['dens'])[:, 0]
    mI, j, i = np.unravel_index(np.argmax(d), d.shape)
    x0 = g[mI, 0] + (i + .5) * (g[mI, 1] - g[mI, 0]) / nx
    y0 = g[mI, 2] + (j + .5) * (g[mI, 3] - g[mI, 2]) / nx
    dx = (g[:, 1] - g[:, 0]) / nx
    X = g[:, 0, None, None] + (np.arange(nx)[None, None, :] + .5) * dx[:, None, None]
    Y = g[:, 2, None, None] + (np.arange(nx)[None, :, None] + .5) * dx[:, None, None]
    r = np.hypot(X - x0, Y - y0)
    # each slice location is covered by exactly one leaf block, so sums are unique
    dA = (dx**2)[:, None, None] * np.ones_like(d)
    col = (d * dA * (r < 2 * k['rstar'])).sum() / k['rhoc']
    col_all = (d * dA).sum() / k['rhoc']
    return p['time'], d.max() / k['rhoc'], col, col_all, x0, y0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{C}/plots/overnight')
    ap.add_argument('--nproc', type=int, default=24)
    a = ap.parse_args()
    cdir = f'{a.out}/r040_item3_cache'
    os.makedirs(cdir, exist_ok=True)
    jobs = [(lab, f, cdir) for lab in RUNS for f in rsts(lab)]
    with Pool(a.nproc) as pool:
        keys = pool.map(one_rst, jobs, chunksize=1)
        R = {}
        for (lab, f, _), kf in zip(jobs, keys):
            R.setdefault(lab, []).append(dict(np.load(kf)))
        S = {}
        for lab in RUNS:
            S[lab] = np.array(pool.map(one_slice, [(lab, f) for f in slices(lab)], chunksize=4))
    np.savez(f'{a.out}/r040_item3_slices.npz', **S)
    report(R, S, a.out)


def tperi(S):
    return S[np.argmin(np.hypot(S[:, 4], S[:, 5])), 0]


def arr(rs, k):
    return np.array([r[k] for r in rs])


def report(R, S, out):
    # (a)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    for lab in ('Q1', 'Q6', 'V5L4', 'V5L5'):
        _, col, ls, _ = RUNS[lab]
        rs = R[lab]; s = S[lab]
        ax[0].plot(arr(rs, 't'), arr(rs, 'Mb'), 'o' + ls, color=col, ms=4, label=lab)
        ax[1].semilogy(s[:, 0], s[:, 1], ls, color=col, lw=1, label=lab)
        ax[1].semilogy(arr(rs, 't'), arr(rs, 'rho_rem'), 'o', color=col, ms=4)
        ax[2].semilogy(s[:, 0], s[:, 2] / s[0, 2], ls, color=col, lw=1, label=lab)
        for x in ax:
            x.axvline(tperi(s), color=col, lw=0.5, alpha=0.5)
    ax[0].set_title('3D self-bound mass / M* (remnant.py, every restart)')
    ax[1].set_title(r'$\rho_{max}/\rho_c$: z = 0 slice (lines, 1 M), 3D bound clump max (o)')
    ax[2].set_title('2D proxy: slice column within 2 R* of ρ_max, / t = 0')
    for x in ax:
        x.set_xlabel('t [M]'); x.grid(alpha=0.3); x.legend(fontsize=8)
    fig.suptitle('R-040 3(a): rp10 remnant history (vertical lines: periapsis)')
    fig.savefig(f'{out}/r040_3a_remnant_history.png', dpi=90); plt.close(fig)
    # (b)
    fig, ax = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    for lab in ('Q1', 'Q6', 'V5L4'):
        _, col, ls, _ = RUNS[lab]
        rs = R[lab]; t = arr(rs, 't')
        levm = arr(rs, 'levm'); Md = levm.sum(1)
        ax[0, 0].plot(t, levm[:, 4] / np.maximum(Md, 1e-300), 'o' + ls, color=col, ms=4, label=f'{lab} L4')
        ax[0, 0].plot(t, levm[:, 3] / np.maximum(Md, 1e-300), 'x' + ls, color=col, ms=4, alpha=0.6)
        ax[0, 1].plot(t, arr(rs, 'nb3')[:, 4], 'o' + ls, color=col, ms=4, label=f'{lab} L4, <3R*')
        ax[0, 1].plot(t, arr(rs, 'nb10')[:, 3], 'x' + ls, color=col, ms=4, alpha=0.6, label=f'{lab} L3, <10R*')
        ax[0, 2].plot(t, arr(rs, 'fb'), 'o' + ls, color=col, ms=4, label=lab)
        ax[1, 0].plot(t, arr(rs, 'Ed'), 'o' + ls, color=col, ms=4, label=lab)
        ax[1, 1].plot(t, arr(rs, 'Ld'), 'o' + ls, color=col, ms=4, label=lab)
        s = S[lab]
        ax[1, 2].plot(s[:, 0], np.hypot(s[:, 4], s[:, 5]), ls, color=col, label=f'{lab}: peri t={tperi(s):.1f}')
    ax[0, 0].set_title('dense-mass fraction on L4 (o) and L3 (x)')
    ax[0, 1].set_title('blocks near ρ_max: L4 within 3R* (o), L3 within 10R* (x)')
    ax[0, 2].set_title('bound fraction (E<0) of all gas')
    ax[1, 0].set_title('mass-weighted E = −u_t − 1 of dense gas')
    ax[1, 1].set_title('mass-weighted L_z = x u_y − y u_x of dense gas')
    ax[1, 2].set_title('|x| of ρ_max (slices)'); ax[1, 2].set_yscale('log')
    for x in ax.flat:
        x.set_xlabel('t [M]'); x.grid(alpha=0.3); x.legend(fontsize=8)
    fig.suptitle('R-040 3(b): Q1 (±64) vs Q6 (±128) vs old V5-L4')
    fig.savefig(f'{out}/r040_3b_box.png', dpi=90); plt.close(fig)
    # (b) bound mass per radius bin at t ~ 25, 50, 75, 100
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2), constrained_layout=True)
    for c_, tt in enumerate((25, 48, 75, 100)):
        for lab in ('Q1', 'Q6', 'V5L4'):
            r = min(R[lab], key=lambda r: abs(r['t'] - tt))
            e = r['redges']; xc = np.arange(len(e) - 1)
            ax[c_].step(xc, r['mb_r'], where='mid', color=RUNS[lab][1], label=f'{lab} t={r["t"]:.0f} (E<0)')
            ax[c_].step(xc, r['mt_r'], where='mid', color=RUNS[lab][1], ls=':', alpha=0.6)
        ax[c_].set_xticks(xc); ax[c_].set_xticklabels([f'{a:g}-{b:g}' for a, b in zip(e[:-1], e[1:])],
                                                       rotation=45, fontsize=7)
        ax[c_].set_yscale('log'); ax[c_].set_xlabel('distance from ρ_max [R*]'); ax[c_].legend(fontsize=7)
        ax[c_].set_title(f't ≈ {tt}: bound mass (solid), all mass (dotted) / M*')
    fig.savefig(f'{out}/r040_3b_bound_by_radius.png', dpi=90); plt.close(fig)
    # (b) layouts
    fig, ax = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for r_, lab in enumerate(('Q1', 'Q6')):
        fs = {int(f.split('.')[-2]): f for f in slices(lab)}
        for c_, tt in enumerate((0, 50, 100, 135)):
            p = bc.read_binary(fs[tt])
            g = np.asarray(p['mb_geometry']); d = np.asarray(p['mb_data']['dens'])[:, 0]
            mI, j, i = np.unravel_index(np.argmax(d), d.shape); nx = p['nx1_out_mb']
            x0 = g[mI, 0] + (i + .5) * (g[mI, 1] - g[mI, 0]) / nx
            y0 = g[mI, 2] + (j + .5) * (g[mI, 3] - g[mI, 2]) / nx
            h = 6.0
            draw(ax[r_, c_], [(q[0], q[1], q[2], q[3], v / RHOC0) for q, v in zip(g, d)],
                 ((x0 - h, x0 + h), (y0 - h, y0 + h)), LogNorm(1e-9, 1), plt.get_cmap('inferno'),
                 outline=True)
            ax[r_, c_].set_title(f'{lab} t={tt}', fontsize=9)
    fig.suptitle('MeshBlock layout (white) around ρ_max, ±6 M, z = 0')
    fig.savefig(f'{out}/r040_3b_layout.png', dpi=80); plt.close(fig)
    # (c)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    rows = []
    for lab in ('Q4', 'V5L5', 'V5L4', 'Q1'):
        _, col, ls, _ = RUNS[lab]
        rs = R[lab]; t = arr(rs, 't'); km = arr(rs, 'kmean'); tp = tperi(S[lab])
        ax[0].plot(t, km, 'o' + ls, color=col, ms=4, label=f'{lab} (peri {tp:.0f})')
        ax[1].plot(t / tp, km, 'o' + ls, color=col, ms=4, label=lab)
        ax[2].plot(t / tp, arr(rs, 'w5'), 'o' + ls, color=col, ms=4, label=lab)
        s = t <= tp + 1e-6
        ok = s & np.isfinite(km)
        I = np.trapz(km[ok], t[ok]) if ok.sum() > 1 else np.nan
        Iu = np.trapz(km[ok], t[ok] / tp) if ok.sum() > 1 else np.nan
        kp = np.interp(tp, t[np.isfinite(km)], km[np.isfinite(km)])
        rows.append(f'{lab}: t_peri {tp:.1f}; <K/K0-1> at periapsis (interp) {kp:+.4f}; '
                    f'int_0^tp <K/K0-1> dt = {I:.3f} M; int d(t/tp) = {Iu:.4f}; '
                    f'restarts used t = {", ".join(f"{x:.0f}" for x in t)}')
    ax[0].set_title('mass-weighted <K/K0 − 1>, ρ > 0.1 ρ_c'); ax[0].set_xlabel('t [M]')
    ax[1].set_title('same vs t / t_peri'); ax[1].set_xlabel('t / t_peri')
    ax[2].set_title('M3 warm > 5% fraction vs t / t_peri'); ax[2].set_xlabel('t / t_peri')
    for x in ax:
        x.grid(alpha=0.3); x.legend(fontsize=8)
    fig.suptitle('R-040 3(c): mean warm excess, Q4 (2e4, L6) vs V5 (1e4)')
    fig.savefig(f'{out}/r040_3c_mean_warm.png', dpi=90); plt.close(fig)
    with open(f'{out}/r040_item3.txt', 'w') as fh:
        for lab in RUNS:
            for r in R[lab]:
                fh.write(f'{lab:5s} t={r["t"]:6.1f} Mb={r["Mb"]:.4f} rho_max={r["rho_max"]:.4f} '
                         f'rho_rem={r["rho_rem"]:.4f} fb={r["fb"]:.4f} Ed={r["Ed"]:+.3e} Ld={r["Ld"]:.4f} '
                         f'Erem={r["Erem"]:+.3e} Lrem={r["Lrem"]:.4f} kmean={r["kmean"]:+.4f} '
                         f'kmean_rem={r["kmean_rem"]:+.4f} w5={r["w5"]:.3f} Md={r["Md"]:.4f} '
                         f'root={int(r["root"])} nmb={int(r["nmb"])} levfrac={np.round(r["levm"] / max(r["levm"].sum(), 1e-300), 3).tolist()} '
                         f'nbl={r["nbl"].tolist()} nb3={r["nb3"].tolist()} nb10={r["nb10"].tolist()} '
                         f'mb_r={np.round(r["mb_r"], 4).tolist()} mt_r={np.round(r["mt_r"], 4).tolist()}\n')
        fh.write('\n'.join(rows) + '\n')
        for lab in RUNS:
            s = S[lab]
            fh.write(f'{lab} slices: n={len(s)} t_peri={tperi(s):.1f}\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
