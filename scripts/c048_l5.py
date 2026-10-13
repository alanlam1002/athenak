#!/usr/bin/env python3
"""C-048 (R-057 item 2): the L5 comoving rp10 run (L5c_rp10_comov_t200, debris/star at L5,
BH box L4, 10 nodes, t = 0 -> 200) against R6 (the same deck at L4, 7 nodes).

Items (each restart is paired with R6's closest restart; offsets are printed):
  r049     stream width l/dx and the Hill-sphere metric (c045_r049.one, cached)
  entropy  dense mean K/K0 and K at rho_max from prim_xy (tde_t11_surface_diag.series, as C-044),
           M3 warm > 5% / > 10% of the dense mass from restarts (c044_series.per_rst's definition),
           and the dual-energy switch census from the dediag_xy slices (d_r043_maps.census),
           averaged over t < 80, 80-120, 120-160, 160-200 (R6: c044_census_R6.txt)
  dmde     dM/dE and Mdot (c044_mdot.mdot; remnant excluded, r_BH >= 3): quantile shift L5 - R6,
           core width, wings |E| > 2 dE, bound fraction, Mdot peak / rise, and L5's late restarts
           against each other
  orbit    mass-weighted BH-frame Killing <E>, <L_z> (r_BH >= 3) vs t; min star-BH distance
           (c044_series.bh_track + prim_xy rho_max)
  cost     dt per cycle, node-h per M by window (ensemble.out elapsed), MeshBlocks per level
  dt       the CFL attribution (c045_dt.one)
  s0       the single grid-rest zone what-if (c046_s0.whatif, r_s = 5, rho_thr 1e-10, v_max 0.9)

    c048_l5.py [--items r049,entropy,dmde,orbit,cost,dt,s0] [--nproc 8] [--out DIR] [--fig]
"""
import argparse
import glob
import os
import re
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tde_dmde as D                              # noqa: E402
import remnant as RM                              # noqa: E402
import rst_tree as RT                             # noqa: E402

C = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc'
P = f'{C}/plots/phase2_m1'
B = 'cfc_tde_wd_imbh_parabolic_tracker'
RHOC, K0, MSTAR, GAM = 3.24e-4, 1.2346626297e-2, 7.47e-5, 5.0 / 3.0
DE = 0.677 / 16.4**2
L5 = 'L5c_rp10_comov_t200'
R6 = 'R6_rp10_comov_L4_fix'
R6S = (R6, 'R6x_rp10_comov_L4_t300')
NODES = {'L5': 10, 'R6': 7}
CACHE = f'{P}/c048_cache'


def rsts(segs):
    byt = {}
    for s in segs:
        for f in sorted(glob.glob(f'{C}/{s}/rst/*.rst')):
            byt[round(RT.read_tree(f)['time'], 1)] = f          # later segments win
    return byt


def pair(t5, r6):
    t6 = min(r6, key=lambda x: abs(x - t5))
    return t6, r6[t6]


def rst_pass(fn):
    """one cells() pass per restart: E, dM, L_z, remnant mask, dense K (cached)"""
    tag = fn.split('/')[-3] + '_' + fn.split('.')[-2]
    cf = f'{CACHE}/{tag}.npz'
    if os.path.exists(cf):
        z = np.load(cf)
        return {k: z[k] for k in z.files}
    c = D.cells(fn, rcut=3.0)
    Mb, xc, vc, rem = RM.bound_mass(c)
    xb = c['xbh']
    Lz = (c['x'] - xb[0]) * c['uy'] - (c['y'] - xb[1]) * c['ux']
    dense = c['rho'] > 0.1 * RHOC
    k = c['P'][dense] / c['rho'][dense]**GAM / K0 - 1
    out = dict(t=c['t'], E=c['E'], dM=c['dM'], Lz=Lz, rem=rem, Mb=Mb / MSTAR, kd=k, md=c['dM'][dense])
    os.makedirs(CACHE, exist_ok=True)
    np.savez(cf, **out)
    return out


def quant(E, m, q):
    o = np.argsort(E); c = np.cumsum(m[o]) / m.sum()
    return np.interp(q, c, E[o])


def item_r049(r5, r6, nproc, rows):
    import c045_r049 as R
    ts = [124.8, 139.9, 153.9, 167.1, 179.2, 191.6, 200.0]
    jobs = [r5[t] for t in ts] + [pair(t, r6)[1] for t in ts]
    with Pool(min(nproc, 6)) as p:
        res = dict(zip(jobs, p.map(R.one, jobs)))
    rows.append('== r049: stream width l/dx (mass-weighted over debris rho > 1e-6 rho_c; in units of the '
                "cell's own dx), bound-wing (E < -0.5 dE) mass with l/dx < 4 / < 8 and by level; Hill metric")
    for t in ts:
        t6, f6 = pair(t, r6)
        for lab, f in (('L5', r5[t]), ('R6', f6)):
            r = res[f]
            wl = ' '.join(f'L{L} {v:.2f}' for L, v in r['wl'].items() if v > 0.005)
            rows.append(f'  {lab} t={r["t"]:6.1f}: l/dx med {r["med"]:5.1f} p10 {r["p10"]:4.1f}; wing l/dx<4 {r["f4"]:.2f} '
                        f'<8 {r["f8"]:.2f} [{wl}] | remnant {r["Mb"]:.4f} M* R50 {r["R50"]:.2f} R90 {r["R90"]:.2f} '
                        f'd {r["d"]:.1f} r_H {r["rH"]:.2f} r_H/R90 {r["rH"] / r["R90"]:.2f} M(<r_H, bound) {r["Mh"]:.4f}')
    return res


def item_entropy(r5, r6, nproc, rows, S):
    import tde_t11_surface_diag as T
    import d_r043_maps as R
    rows.append('== entropy: dense mean K/K0 (prim_xy, rho > 0.1 rho0, z = 0; Kw01) and K/K0 at rho_max; '
                'M3 = mass fraction of rho > 0.1 rho_c (3D, r_BH >= 3) with K/K0 - 1 > 5% (> 10%)')
    ks = {}
    for lab, segs in (('L5', (L5,)), ('R6', R6S)):
        best = {}
        for seg in segs:
            T.RUNS[f'C48_{seg}'] = (f'{C}/{seg}', B, 10, 'k', '-')
            s = T.series(f'C48_{seg}', f'{C}/plots/dual_energy')
            for i, t in enumerate(s['t']):
                best[round(float(t), 3)] = (s['Kw01'][i], s['Kmax'][i])
        t = np.array(sorted(best)); v = np.array([best[x] for x in t])
        ks[lab] = (t, v[:, 0], v[:, 1])
    S['ks'] = ks
    for t in sorted(S['L5']):
        if t <= 0:
            continue
        a = S['L5'][t]; t6, _ = pair(t, r6); b = S['R6'][t6]
        line = f'  t {t:6.1f} | R6 {t6:6.1f}:'
        for lab, z, tt in (('L5', a, t), ('R6', b, t6)):
            kt, kw, kmx = ks[lab]
            i = int(np.argmin(np.abs(kt - tt)))
            m3 = z['md'][z['kd'] > 0.05].sum() / z['md'].sum(); m10 = z['md'][z['kd'] > 0.10].sum() / z['md'].sum()
            line += f'  {lab} K/K0 {kw[i]:.3f} (rho_max {kmx[i]:.3f}) M3 {m3:.3f} (>10% {m10:.3f})'
        rows.append(line)
    # switch census
    d = f'{C}/{L5}'
    ns = sorted(int(f.split('.')[-2]) for f in glob.glob(f'{d}/bin/{B}.dediag_xy.*.bin'))
    cf = f'{CACHE}/census_L5.txt'
    if os.path.exists(cf) and len(np.loadtxt(cf)) == len(ns):
        C5 = np.loadtxt(cf)
    else:
        with Pool(min(nproc, 8)) as p:
            C5 = np.array(p.starmap(_census, [(d, n) for n in ns]))
        os.makedirs(CACHE, exist_ok=True)
        np.savetxt(cf, C5, header=R.HDR, fmt='%.5g')
    C6 = np.loadtxt(f'{P}/c044_census_R6.txt')
    rows.append('  switch census (xy slice, dense = branch > 0): window | entropy-branch <K_en/K_tr - 1> (mean of '
                'per-dump p99) | flagged | energy-branch | guard  [fractions of dense cells, mean over dumps]')
    for lo, hi in ((0, 80), (80, 120), (120, 160), (160, 200.01)):
        line = f'    t {lo:3.0f}-{hi:3.0f}:'
        for lab, A in (('L5', C5), ('R6', C6)):
            k = (A[:, 0] >= lo) & (A[:, 0] < hi) & (A[:, 1] > 0)
            if not k.any():
                continue
            nd = A[k, 1]
            line += (f'  {lab} {np.nanmean(A[k, 8]):+.1e} ({np.nanmean(A[k, 9]):+.3f}) | {np.nanmean(A[k, 18]):.3f} | '
                     f'{np.mean((A[k, 5] + A[k, 6]) / nd):.3f} | {np.mean(A[k, 3] / nd):.3f}  (n_dense {np.mean(nd):.0f})')
        rows.append(line)
    S['census'] = (C5, C6)


def _census(d, n):
    import d_r043_maps as R
    return R.census(R.load(d, 'xy', n))


def item_dmde(S, rows):
    import c044_mdot as M
    tp = M.t_peri(L5, 10.0)
    rows.append(f'== dM/dE and Mdot (r_BH >= 3, remnant excluded unless noted; Keplerian return from t_p = {tp:.1f})')
    q = np.linspace(0.01, 0.99, 99)
    for t in (153.9, 167.1, 179.2, 191.6, 200.0):
        a = S['L5'][t]; t6 = min(S['R6'], key=lambda x: abs(x - t)); b = S['R6'][t6]
        line = f'  t {t:6.1f} | R6 {t6:6.1f}:'
        for lab, z in (('L5', a), ('R6', b)):
            deb = ~z['rem'].astype(bool); E, dM = z['E'][deb] / DE, z['dM'][deb]
            core = np.abs(E) < 2
            sd = np.sqrt(np.average(E[core]**2, weights=dM[core]) - np.average(E[core], weights=dM[core])**2)
            md = M.mdot(z['E'][deb], dM, tp); tc = np.sqrt(M.TB[1:] * M.TB[:-1]); i = int(np.argmax(md))
            t10 = tc[np.argmax(md >= 0.1 * md[i])]
            line += (f'  {lab}: bound {z["dM"][z["E"] < 0].sum() / z["dM"].sum():.4f} remnant {z["Mb"]:.4f} '
                     f'core std(|E|<2dE) {sd:.3f} IQR {np.subtract(*quant(E, dM, [0.75, 0.25])):.3f} '
                     f'wings E<-2dE {dM[E < -2].sum() / dM.sum():.2e} E>2dE {dM[E > 2].sum() / dM.sum():.2e} '
                     f'Mdot peak {md[i]:.2e} M*/M at {tc[i]:.0f} M, >10% peak from {t10:.0f} M;')
        qa, qb = quant(a['E'], a['dM'], q), quant(b['E'], b['dM'], q)
        line += (f'  quantile shift L5 - R6 (all gas): median {np.median(qa - qb) / DE:+.3f} dE, '
                 f'q10/50/90 {(qa - qb)[9] / DE:+.3f}/{(qa - qb)[49] / DE:+.3f}/{(qa - qb)[89] / DE:+.3f} dE')
        rows.append(line)
    # dump-to-dump stability of L5 (frozen-in)
    ref = S['L5'][200.0]
    for t in (179.2, 191.6):
        z = S['L5'][t]
        qa, qb = quant(z['E'], z['dM'], q), quant(ref['E'], ref['dM'], q)
        rows.append(f'  L5 frozen-in: t {t} vs 200: quantile shift median {np.median(qa - qb) / DE:+.4f} dE, max |.| '
                    f'{np.max(np.abs(qa - qb)) / DE:.4f} dE; bound {z["dM"][z["E"] < 0].sum() / z["dM"].sum():.4f} vs '
                    f'{ref["dM"][ref["E"] < 0].sum() / ref["dM"].sum():.4f}')
    S['tp'] = tp


def item_orbit(S, rows):
    import c044_series as CS
    rows.append('== orbit: mass-weighted BH-frame Killing <E> and <L_z> about x_BH (r_BH >= 3)')
    for lab in ('L5', 'R6'):
        ts = sorted(S[lab])
        E = [np.average(S[lab][t]['E'], weights=S[lab][t]['dM']) for t in ts]
        L = [np.average(S[lab][t]['Lz'], weights=S[lab][t]['dM']) for t in ts]
        rows.append(f'  {lab}: ' + '  '.join(f't {t:.0f} <E> {e:+.4e} <L> {l:.5f}' for t, e, l in zip(ts, E, L)))
    for lab, segs in (('L5', (L5,)), ('R6', (R6,))):
        tr = [CS.bh_track(f'{C}/{s}') for s in segs]
        th = np.concatenate([t for t, _ in tr]); xh = np.concatenate([x for _, x in tr])
        o = np.argsort(th, kind='stable'); th, xh = th[o], xh[o]
        fs = [f for s in segs for f in sorted(glob.glob(f'{C}/{s}/bin/{B}.prim_xy.*.bin'))]
        dd = np.array([CS.per_dump(f) for f in fs if 110 <= int(f.split('.')[-2]) <= 170])   # dump n ~ t = n M
        r = np.hypot(dd[:, 1] - np.interp(dd[:, 0], th, xh[:, 0]), dd[:, 2] - np.interp(dd[:, 0], th, xh[:, 1]))
        k = int(np.argmin(r))
        rows.append(f'  {lab}: min star(rho_max)-BH coordinate distance {r[k]:.2f} M at t = {dd[k, 0]:.1f}')


def log(seg):
    a = np.array([[float(x) for x in m.groups()] for ln in open(f'{C}/{seg}/ensemble.out')
                  for m in [re.match(r'elapsed=(\S+) cycle=\d+ time=(\S+) dt=(\S+)', ln)] if m])
    return a[1:]


def item_cost(r5, r6, rows):
    rows.append('== cost: node-h per M (ensemble.out elapsed; cycle 0 dropped), cycles per M, mean dt; '
                'MG vs hydro split NOT available (no timers in ensemble.out)')
    A = {'L5': log(L5), 'R6': log(R6)}
    for lo, hi in ((0, 80), (80, 120), (120, 160), (160, 200)):
        line = f'  t {lo:3d}-{hi:3d}:'
        for lab, a in A.items():
            k = (a[:, 1] >= lo) & (a[:, 1] <= hi)
            el, tt = a[k, 0], a[k, 1]
            nh = (el[-1] - el[0]) * NODES[lab] / 3600 / (tt[-1] - tt[0])
            line += f'  {lab} {nh:.3f} node-h/M, {k.sum() / (tt[-1] - tt[0]):.1f} cycles/M, dt {np.mean(a[k, 2]):.4f};'
        rows.append(line)
    for lab, a in A.items():
        rows.append(f'  {lab} total t 0-200: {(a[-1, 0] - a[0, 0]) * NODES[lab] / 3600:.1f} node-h, {len(a)} cycles '
                    f'({NODES[lab]} nodes)')
    rows.append('  MeshBlocks per level (above root) at matched restarts:')
    for t in (0.0, 74.8, 124.8, 139.9, 153.9, 200.0):
        t6, f6 = pair(t, r6)
        line = f'    t {t:6.1f}:'
        for lab, f in (('L5', r5[t]), ('R6', f6)):
            h = D.header(f); lv = h['ll'][:, 3] - h['root']
            line += f'  {lab} ({lab == "R6" and t6 or t}) ' + ' '.join(f'L{L}:{(lv == L).sum()}' for L in range(lv.max() + 1)) \
                + f' tot {len(lv)};'
        rows.append(line)


def item_dt(rows):
    import c045_dt as T
    rows.append('== dt attribution (c045_dt.one)')
    for t in (139.9, 200.0):
        rows += T.one('L5', L5, t)


def item_s0(r5, rows):
    import c046_s0 as S0
    rows.append('== S0 single zone (r_s 5, rho_thr 1e-10 rho_c, v_max 0.9) on L5 restarts')
    for t in (139.9, 200.0):
        tt, B_, Cc = S0.load(r5[t])
        dt0 = B_['dt'].min()
        w = S0.whatif(B_, Cc, 1e-10, 5.0, 0.9); L = w['lim']
        rows.append(f'  L5 t {tt:.1f}: dt0 {dt0:.4f} (code {S0.logged_dt(L5, t):.4f}) -> {L["dt"]:.4f} (x{L["dt"] / dt0:.2f}); '
                    f'lim [{L["where"]}, L{L["lev"]}, r_BH {L["r"]:.2f}, rho {L["rho"]:.1e}, |lam| {L["lam"]:.3f} = '
                    f'|a-term {L["at"]:+.3f} - shift {L["bt"]:+.3f}|]; zone {w["n"]} cells, M {w["M"]:.1e} M*, '
                    f'max rho {w["zmaxrho"]:.1e}')


def figure(S, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import c044_mdot as M
    fig, ax = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    ax = ax.ravel()
    col = {'L5': 'C3', 'R6': 'C0'}
    for lab in ('L5', 'R6'):
        kt, kw, kmx = S['ks'][lab]
        k = kt <= 200.5
        ax[0].plot(kt[k], kw[k], color=col[lab], label=f'{lab} dense mean K/K0 (xy)')
        ts = sorted(S[lab])
        m3 = [S[lab][t]['md'][S[lab][t]['kd'] > 0.05].sum() / S[lab][t]['md'].sum() for t in ts]
        ax[1].plot(ts, m3, 'o-', color=col[lab], label=f'{lab} M3 (warm > 5%)')
        ax[1].plot(ts, [S[lab][t]['Mb'] for t in ts], 's--', color=col[lab], ms=4, label=f'{lab} self-bound remnant / M*')
        ax[2].plot(ts, [np.average(S[lab][t]['E'], weights=S[lab][t]['dM']) / DE for t in ts], 'o-', color=col[lab], label=lab)
    C5, C6 = S['census']
    for lab, A in (('L5', C5), ('R6', C6)):
        k = (A[:, 1] > 0) & (A[:, 0] <= 200.5)
        ax[3].plot(A[k, 0], np.convolve(A[k, 18], np.ones(5) / 5, 'same'), color=col[lab], label=f'{lab} flagged')
        ax[3].plot(A[k, 0], np.convolve((A[k, 5] + A[k, 6]) / A[k, 1], np.ones(5) / 5, 'same'), ':', color=col[lab],
                   label=f'{lab} energy branch')
    e = np.linspace(-6, 6, 121); tc = np.sqrt(M.TB[1:] * M.TB[:-1])
    for lab, t in (('L5', 200.0), ('R6', 200.0)):
        z = S[lab][t]; deb = ~z['rem'].astype(bool)
        h, _ = np.histogram(z['E'][deb] / DE, bins=e, weights=z['dM'][deb] / MSTAR)
        ax[4].semilogy(0.5 * (e[1:] + e[:-1]), np.where(h > 0, h / np.diff(e), np.nan), color=col[lab], label=f'{lab} t = 200')
        md = M.mdot(z['E'][deb], z['dM'][deb], S['tp'])
        ax[5].loglog(tc, np.where(md > 0, md, np.nan), color=col[lab], label=f'{lab} t = 200')
    for t, ls in ((179.2, ':'), (191.6, '--')):
        z = S['L5'][t]; deb = ~z['rem'].astype(bool)
        md = M.mdot(z['E'][deb], z['dM'][deb], S['tp'])
        ax[5].loglog(tc, np.where(md > 0, md, np.nan), ls, color='C3', lw=0.8, label=f'L5 t = {t}')
    ax[0].set(xlabel='t [M]', ylabel='K/K0', title='dense mean K/K0 (xy slice)')
    ax[1].set(xlabel='t [M]', title='M3 and self-bound remnant (3D)')
    ax[2].set(xlabel='t [M]', ylabel=r'$\langle E\rangle/\Delta E$', title='orbit: mass-weighted Killing E')
    ax[3].set(xlabel='t [M]', ylabel='fraction of dense cells', title='dual-energy switch (5-dump mean)')
    ax[4].set(xlabel=r'$E/\Delta E$', ylabel='dM/d(E/ΔE) [M*]', ylim=(1e-6, 2), title='dM/dE, debris (remnant excluded)')
    ax[5].set(xlabel='t [M]', ylabel=r'$\dot M$ [M*/M]', ylim=(1e-9, 1e-2), title='frozen-in Mdot')
    for x in ax:
        x.grid(alpha=0.3); x.legend(fontsize=7)
    fig.suptitle('C-048: L5 comoving rp10 (debris/star L5, 10 nodes) vs R6 (L4, 7 nodes)')
    fig.savefig(f'{out}/c048_l5.png', dpi=60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--items', default='r049,entropy,dmde,orbit,cost,dt,s0')
    ap.add_argument('--nproc', type=int, default=8)
    ap.add_argument('--out', default=P)
    ap.add_argument('--fig', action='store_true')
    a = ap.parse_args()
    it = a.items.split(',')
    r5, r6 = rsts((L5,)), rsts(R6S)
    r6 = {t: f for t, f in r6.items() if t <= 200.0}
    rows = [f'C-048: {L5} vs R6 ({", ".join(R6S)}); R6 paired with its closest restart.']
    S = {}
    if {'entropy', 'dmde', 'orbit'} & set(it) or a.fig:
        j5 = [f for t, f in r5.items() if t > 0]
        j6 = [f for t, f in r6.items() if t > 0]
        with Pool(min(a.nproc, 8)) as p:
            res = dict(zip(j5 + j6, p.map(rst_pass, j5 + j6)))
        S['L5'] = {t: res[f] for t, f in r5.items() if t > 0}
        S['R6'] = {t: res[f] for t, f in r6.items() if t > 0}
    if 'r049' in it:
        item_r049(r5, r6, a.nproc, rows)
    if 'entropy' in it or a.fig:
        item_entropy(r5, r6, a.nproc, rows, S)
    if 'dmde' in it or a.fig:
        item_dmde(S, rows)
    if 'orbit' in it:
        item_orbit(S, rows)
    if 'cost' in it:
        item_cost(r5, r6, rows)
    if 'dt' in it:
        item_dt(rows)
    if 's0' in it:
        item_s0(r5, rows)
    print('\n'.join(rows), flush=True)
    with open(f'{a.out}/c048_l5_{"_".join(it)}.txt', 'w') as f:
        f.write('\n'.join(rows) + '\n')
    if a.fig:
        figure(S, a.out)


if __name__ == '__main__':
    main()
