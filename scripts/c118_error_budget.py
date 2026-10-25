#!/usr/bin/env python3
"""R-131 item 1 (c118): the dual-energy error budget for comoving production runs. The star-like flat table
f_RH(M, V) (c113, lo family, production DE 'on', n512) is convolved with the measured 3D distribution of the
shock's grid speed |V.n|_coord at the flagged dense cells of a 2x2 arm (CD; BD for contrast), per phase and
density band. The implied error in the shock-heated entropy per shock passage, as a fraction of the pre-shock K:
    e = (f_RH(M, V) - 1) (K_RH(M)/K_1 - 1)
(the flat table's f_RH is the captured fraction of the exact jump K_RH - K_1).

Per 3D dump (prim_3d every 12 M plus prim_3d_peri), as c117_check3d.py:
 - flag = the vcurv sensor recomputed in 3D from the primitives (div v <= 0, psi_v > 0.05 c_s, one 0.5x dilation
   pass; primitive_solver_hyd.hpp:1822-1864); the dump has no de_diag, so this is an approximation of the code's
   flag. Dense = D = rho W psi6 > rho_sw = 3.2e-8.
 - alpha, beta^x, beta^y, psi4 from the nearest adm_xy slice, taken z-independent (the 3D dumps carry no metric).
   Sensitivity: |V.n| recomputed with the metric sampled at (x + |z|, y) and (x, y + |z|); the mass-weighted
   mean |change| is reported (dVn_metric).
 - n = -grad P / |grad P| (3D central differences); V.n = (alpha v^i - beta^i) n_i, beta^z = 0 assumed.
 - local Mach number: the converging velocity jump du = v_a(-2) - v_a(+2) of the velocity component along the axis
   a of the largest |d2 v| at the cell (within the MeshBlock), inverted with the Newtonian RH relation
   du = 2 c_s (M - 1/M)/(Gamma + 1) (the lo family is non-relativistic, c_s ~ 1e-2), c_s = the larger of the two
   end cells' (M_lo) or the smaller (M_hi, an upper estimate); clipped to [1, 3]; e uses K_RH(M) from the same
   Newtonian relation. M < 1.2 uses the M = 1.2 row of f. (A pressure-ratio estimate over the same window saturates:
   the hydrostatic gradient across 5 cells dominates it.)
 - f(M, V): linear in M between 1.2, 2, 3 and in V between the table's V = 0, 0.1, 0.3, 0.5 (clamped at 0.5).
   V = 0 rows: ex0.0 / co0.1 / co0.3 / co0.5 interpolated at the cell's |v|_E (Eulerian speed); V = 0.1: ex0.1;
   V = 0.3, 0.5: du0.3, du0.5. The table has no point between V = 0 and 0.1, where nearly all of CD's shocks sit,
   so two brackets are also reported: 'lo' (f = f(V=0) for V < 0.1) and 'hi' (f = f(V=0.1) for 0 < V < 0.1).
 - --supp CSV (D3, c124): the measured supplement rows at V = 0.003, 0.01, 0.03 (lo, on, n512; c113_metrics.py
   format) become extra V nodes. 'sup': V nodes 0 (f0(vE), as above), 0.003, 0.01, 0.03 from the dual rows
   (v_E = 0, beta = -V: pure grid speed), then 0.1 (ex0.1), 0.3, 0.5 as before; 'supex': the same with the
   0.003-0.03 nodes from the ex rows (v_E = V, beta = 0); 'supdu': 'sup' with the 0.1 node from du0.1 instead of
   ex0.1. Columns e_sup_mean, e_sup_absmean, e_sup_p90abs, e_supex_mean, e_supdu_mean, e_sup_Mhi_mean.
    c118_error_budget.py OUT.csv LABEL=RUNDIR [...] [--tp 135.9] [--nproc 8] [--supp c124_results.csv]
"""
import argparse
import csv
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tde_ensemble_plots import bc                 # noqa: E402

B = 'cfc_tde_wd_imbh_parabolic_tracker'
RHOC, GAM, RSW, VCSH = 3.24e-4, 5.0 / 3.0, 3.2e-8, 0.05
BANDS = [(1e-1, 1e9), (1e-2, 1e-1), (1e-3, 1e-2), (0, 1e-3)]   # rho/rho_c
C113 = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/c113_gve/c113_results.csv'
MS, VS, VE = (1.2, 2.0, 3.0), (0.0, 0.1, 0.3, 0.5), (0.0, 0.1, 0.3, 0.5)


def table():
    f = {}
    for r in csv.DictReader(open(C113)):
        if r['kind'] == 'shock' and r['fam'] == 'lo' and r['var'] == 'on' and r['res'] == 'n512':
            f[(float(r['M']), r['cfg'], float(r['vE']), float(r['V']))] = float(r['f_RH'])
    f0 = np.array([[f[(m, 'ex', 0.0, 0.0)], f[(m, 'co', 0.1, 0.0)], f[(m, 'co', 0.3, 0.0)], f[(m, 'co', 0.5, 0.0)]]
                   for m in MS])                                  # [M][vE] at V = 0
    fv = np.array([[f[(m, 'ex', 0.1, 0.1)], f[(m, 'du', 0.0, 0.3)], f[(m, 'du', 0.0, 0.5)]] for m in MS])  # V>0
    return f0, fv


F0, FV = table()
VSUP = (0.003, 0.01, 0.03)
SUP = {}                                                          # mode -> [M][V node] for V = 0.003 .. 0.5


def load_supp(path):
    f = {}
    for r in csv.DictReader(open(path)):
        if r['kind'] == 'shock' and r['fam'] == 'lo' and r['var'] == 'on' and r['res'] == 'n512':
            f[(float(r['M']), r['cfg'], round(float(r['V']), 6))] = float(r['f_RH'])
    for md, cfg, n01 in (('sup', 'du', None), ('supex', 'ex', None), ('supdu', 'du', 'du')):
        SUP[md] = np.array([[f[(m, cfg, v)] for v in VSUP]
                            + [f[(m, 'du', 0.1)] if n01 else FV[i, 0], FV[i, 1], FV[i, 2]]
                            for i, m in enumerate(MS)])


def f_of(M, V, vE, mode):
    Mc = np.clip(M, MS[0], MS[-1])
    w = lambda arr: np.array([np.interp(Mc, MS, arr[:, k]) for k in range(arr.shape[1])])   # [k][cell]
    a0 = w(F0); av = w(FV)
    f0 = np.array([np.interp(vE[i], VE, a0[:, i]) for i in range(len(M))]) if len(M) else np.zeros(0)
    fV = np.vstack([f0, av])                                      # V = 0, 0.1, 0.3, 0.5
    Vc = np.clip(V, 0, VS[-1])
    if mode in SUP:
        fS = np.vstack([f0, w(SUP[mode])])                        # V = 0, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5
        vs = (0.0,) + VSUP + VS[1:]
        return np.array([np.interp(Vc[i], vs, fS[:, i]) for i in range(len(M))])
    if mode == 'lin':
        return np.array([np.interp(Vc[i], VS, fV[:, i]) for i in range(len(M))])
    lin = np.array([np.interp(Vc[i], VS, fV[:, i]) for i in range(len(M))])
    small = Vc < VS[1]
    return np.where(small, fV[0] if mode == 'lo' else fV[1], lin)


def krh(M):
    p21 = (2 * GAM * M**2 - (GAM - 1)) / (GAM + 1)
    r21 = (GAM + 1) * M**2 / ((GAM - 1) * M**2 + 2)
    return p21 * r21**(-GAM) - 1


def mach(du, cs):
    x = 0.5 * (GAM + 1) * np.maximum(du, 0) / cs                  # M - 1/M
    return np.clip(0.5 * (x + np.sqrt(x * x + 4)), 1, 3)


def adm_lookup(A):
    g = np.asarray(A['mb_geometry']); nx, ny = A['nx1_out_mb'], A['nx2_out_mb']
    dx = (g[:, 1] - g[:, 0]) / nx
    o = np.argsort(-dx)                                           # coarse first, finest last wins
    f = {k: np.asarray(A['mb_data'][k])[:, 0] for k in ('adm_alpha', 'adm_betax', 'adm_betay', 'adm_psi4')}
    return g, nx, ny, dx, o, f


def sample(L, X, Y):
    g, nx, ny, dx, o, f = L
    out = {k: np.full(X.shape, np.nan) for k in f}
    x0, x1, y0, y1 = X.min(), X.max(), Y.min(), Y.max()
    for b in o:
        if g[b, 0] > x1 or g[b, 1] <= x0 or g[b, 2] > y1 or g[b, 3] <= y0:
            continue
        m = (X >= g[b, 0]) & (X < g[b, 1]) & (Y >= g[b, 2]) & (Y < g[b, 3])
        if not m.any():
            continue
        i = np.clip(((X[m] - g[b, 0]) / dx[b]).astype(int), 0, nx - 1)
        j = np.clip(((Y[m] - g[b, 2]) / ((g[b, 3] - g[b, 2]) / ny)).astype(int), 0, ny - 1)
        for k in f:
            out[k][m] = f[k][b, j, i]
    return out


def ends(f, ax, k=2):
    """f at -k and +k cells along ax (edge-clamped within the MeshBlock)."""
    pad = [(0, 0)] * 3; pad[ax] = (k, k)
    q = np.pad(f, pad, mode='edge')
    n = f.shape[ax]
    sl = lambda o: tuple(slice(o, o + n) if a == ax else slice(None) for a in range(3))
    return q[sl(0)], q[sl(2 * k)]


def one(arg, want_layer=False):
    lab, d, fp = arg
    P = bc.read_binary(fp)
    t = P['time']
    adms = sorted(glob.glob(f'{d}/bin/{B}.adm_xy.*.bin'))
    ta = np.array([float(bc.read_binary(f)['time']) if False else 2.0 * int(f.split('.')[-2]) for f in adms])
    A = bc.read_binary(adms[int(np.argmin(np.abs(ta - t)))])
    L = adm_lookup(A)
    g = np.asarray(P['mb_geometry']); n = P['nx1_out_mb']
    q = {k: np.asarray(P['mb_data'][k]) for k in ('dens', 'velx', 'vely', 'velz', 'press')}
    acc = {k: [] for k in ('Vn', 'w', 'rr', 'M', 'Mhi', 'vE', 'dVn', 'z')}
    dense_mass = np.zeros(len(BANDS))
    zl = []
    for b in range(len(g)):
        rho, p = q['dens'][b], q['press'][b]
        u = [q['velx'][b], q['vely'][b], q['velz'][b]]
        if (rho * np.sqrt(1 + sum(x * x for x in u)) < 0.5 * RSW).all():
            continue
        h = (g[b, 1] - g[b, 0]) / n
        c = lambda lo: lo + (np.arange(n) + 0.5) * h
        Z, Y, X = np.meshgrid(c(g[b, 4]), c(g[b, 2]), c(g[b, 0]), indexing='ij')
        hh = 1 + GAM / (GAM - 1) * p / rho
        cs = np.sqrt(GAM * p / (rho * hh))
        S = sample(L, X, Y)
        p4 = S['adm_psi4']
        W = np.sqrt(1 + p4 * sum(x * x for x in u))
        v = [x / W for x in u]
        D = rho * W * p4**1.5
        dm = D * h**3
        inn = np.zeros(rho.shape, bool); inn[1:-1, 1:-1, 1:-1] = True
        dn = (D > RSW) & inn & np.isfinite(D)
        rr = rho / RHOC
        for i, (lo, hi) in enumerate(BANDS):
            dense_mass[i] += dm[dn & (rr >= lo) & (rr < hi)].sum()
        psiv = np.zeros_like(rho); div = np.zeros_like(rho); gr = [np.zeros_like(rho) for _ in range(3)]
        axm = np.zeros(rho.shape, int)
        core = (slice(1, -1),) * 3
        for ax in (0, 1, 2):                                      # array axes: 0 z, 1 y, 2 x
            va = v[2 - ax]

            def sh(o):
                s = [slice(1, -1)] * 3; s[ax] = slice(1 + o, rho.shape[ax] - 1 + o); return tuple(s)
            d2 = np.zeros_like(rho); d2[core] = np.abs(va[sh(1)] - 2 * va[core] + va[sh(-1)])
            axm = np.where(d2 > psiv, ax, axm)
            psiv = np.maximum(psiv, d2)
            div[core] += va[sh(1)] - va[sh(-1)]
            gr[2 - ax][core] = (p[sh(1)] - p[sh(-1)]) / (2 * h)
        s1 = (div <= 0) & (psiv > VCSH * cs); wk = (div <= 0) & (psiv > 0.5 * VCSH * cs)
        nb = np.zeros_like(s1)
        for ax in (0, 1, 2):
            nb |= np.roll(s1, 1, ax) | np.roll(s1, -1, ax)
        gn = np.sqrt(gr[0]**2 + gr[1]**2 + gr[2]**2)
        fl = (s1 | (wk & nb)) & dn & np.isfinite(S['adm_alpha']) & (gn > 0)
        zc = Z[:, 0, 0]
        k0 = np.nonzero((zc - 0.5 * h <= 0) & (zc + 0.5 * h > 0))[0]
        if len(k0):   # the layer next to z = 0 (z = 0 is a block face, k = 0, outside the stencil), for the flag check
            k1 = k0[0] + 1 if k0[0] == 0 else k0[0]
            zl.append((X[k1][dn[k1]], Y[k1][dn[k1]], fl[k1][dn[k1]], dm[k1][dn[k1]] / h))
        if not fl.any():
            continue
        du, cmx, cmn = (np.zeros_like(rho) for _ in range(3))
        for ax in (0, 1, 2):
            vm, vp = ends(v[2 - ax], ax); cm, cp = ends(cs, ax)
            on = axm == ax
            du[on] = (vm - vp)[on]; cmx[on] = np.maximum(cm, cp)[on]; cmn[on] = np.minimum(cm, cp)[on]

        def vn(al, bx, by):
            return np.abs(((al * v[0][fl] - bx) * -gr[0][fl] + (al * v[1][fl] - by) * -gr[1][fl]
                           + al * v[2][fl] * -gr[2][fl]) / gn[fl])
        V0 = vn(S['adm_alpha'][fl], S['adm_betax'][fl], S['adm_betay'][fl])
        dV = np.zeros_like(V0)
        for Xs, Ys in ((X + np.abs(Z), Y), (X, Y + np.abs(Z))):
            S2 = sample(L, Xs, Ys)
            ok = np.isfinite(S2['adm_alpha'][fl])
            V2 = vn(np.where(ok, S2['adm_alpha'][fl], S['adm_alpha'][fl]),
                    np.where(ok, S2['adm_betax'][fl], S['adm_betax'][fl]),
                    np.where(ok, S2['adm_betay'][fl], S['adm_betay'][fl]))
            dV = np.maximum(dV, np.abs(V2 - V0))
        acc['Vn'].append(V0); acc['w'].append(dm[fl]); acc['rr'].append(rr[fl]); acc['M'].append(mach(du[fl], cmx[fl])); acc['Mhi'].append(mach(du[fl], cmn[fl]))
        acc['vE'].append(np.sqrt(p4[fl] * (v[0][fl]**2 + v[1][fl]**2 + v[2][fl]**2)))
        acc['dVn'].append(dV); acc['z'].append(np.abs(Z[fl]))
    out = {k: (np.concatenate(x) if x else np.zeros(0)) for k, x in acc.items()}
    if want_layer:
        return lab, t, out, dense_mass, zl
    return lab, t, out, dense_mass


def flag_check(lab, d, fp):
    """The recomputed 3D flag at the dense cells of the layer at z = 1.5 h (z = 0 is a block face, outside the
    3D stencil) against the code's flag in the adjacent z = 0.5 h slice cell (dediag_xy de_shock, the last C2P's)
    at the same time: flagged dense-mass fraction of each, and the mass-weighted overlap (Jaccard)."""
    r = one((lab, d, fp), want_layer=True)
    t, zl = r[1], r[4]
    gs = sorted(glob.glob(f'{d}/bin/{B}.dediag_xy.*.bin'))
    G = bc.read_binary(gs[int(round(t / 2.0))])
    if abs(G['time'] - t) > 1e-6:
        return lab, t, np.nan, np.nan, np.nan
    g = np.asarray(G['mb_geometry']); nx, ny = G['nx1_out_mb'], G['nx2_out_mb']
    fc = np.asarray(G['mb_data']['de_shock'])[:, 0] > 0.5
    X = np.concatenate([z[0] for z in zl]); Y = np.concatenate([z[1] for z in zl])
    F = np.concatenate([z[2] for z in zl]); W = np.concatenate([z[3] for z in zl])
    C = np.zeros(len(X), bool); hit = np.zeros(len(X), bool)
    for b in range(len(g)):
        m = (X > g[b, 0]) & (X < g[b, 1]) & (Y > g[b, 2]) & (Y < g[b, 3])
        if not m.any():
            continue
        hx, hy = (g[b, 1] - g[b, 0]) / nx, (g[b, 3] - g[b, 2]) / ny
        i = ((X[m] - g[b, 0]) / hx).astype(int); j = ((Y[m] - g[b, 2]) / hy).astype(int)
        C[m] = fc[b, j, i]; hit[m] = True
    W = W * hit
    return (lab, t, (W * C).sum() / W.sum(), (W * F).sum() / W.sum(),
            (W * (C & F)).sum() / max((W * (C | F)).sum(), 1e-300))


def wq(x, w, qs):
    if len(x) == 0:
        return [np.nan] * len(qs)
    o = np.argsort(x); c = np.cumsum(w[o]); c /= c[-1]
    return [x[o][min(np.searchsorted(c, qq), len(x) - 1)] for qq in qs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--tp', type=float, default=135.9)
    ap.add_argument('--nproc', type=int, default=8)
    ap.add_argument('--supp', default=None)
    a = ap.parse_args()
    if a.supp:
        load_supp(a.supp)
    PH = [('pre', -1, a.tp - 10), ('peri', a.tp - 10, a.tp + 10), ('early', a.tp + 10, a.tp + 50),
          ('late', a.tp + 50, 1e9)]
    jobs = []
    for spec in a.runs:
        lab, d = spec.split('=', 1)
        fs = sorted(glob.glob(f'{d}/bin/{B}.prim_3d.*.bin')) + sorted(glob.glob(f'{d}/bin/{B}.prim_3d_peri.*.bin'))
        jobs += [(lab, d, f) for f in fs]
    with Pool(a.nproc) as pool:
        R = pool.map(one, jobs)
        FC = pool.starmap(flag_check, [j for j in jobs if 'prim_3d.' in j[2] and glob.glob(f'{j[1]}/bin/{B}.dediag_xy.*')])
    fcl = ['label,t,code_flag_frac_z0,recomputed3d_flag_frac_z0,overlap_jaccard']
    fcl += [f'{l},{t:.2f},{a:.4f},{b:.4f},{c:.4f}' for l, t, a, b, c in sorted(FC, key=lambda x: (x[0], x[1]))]
    open(a.out.replace('.csv', '_flagcheck.csv'), 'w').write('\n'.join(fcl) + '\n')
    print('\n'.join(fcl))
    seen = set(); RR = []
    for r in sorted(R, key=lambda r: (r[0], r[1])):              # periapsis dump at t = 0 / t_end duplicates
        k = (r[0], round(r[1], 3))
        if k not in seen:
            seen.add(k); RR.append(r)
    np.savez_compressed(a.out.replace('.csv', '.npz'),
                        **{f'{r[0]}_{r[1]:.2f}_{k}': v for r in RR for k, v in r[2].items()})
    hdr = ('label,phase,ndumps,band,flag_mass_frac,Vn_p50,Vn_p90,dVn_metric,M_p10,M_p50,M_p90,'
           'e_lin_mean,e_lin_absmean,e_lin_p90abs,e_lo_mean,e_hi_mean,e_Mhi_mean,e_Mhi_hi_mean,e_M2fixed_mean,e_M3fixed_mean'
           + (',e_sup_mean,e_sup_absmean,e_sup_p90abs,e_supex_mean,e_supdu_mean,e_sup_Mhi_mean' if SUP else ''))
    rows = [hdr]
    for lab in dict.fromkeys(r[0] for r in RR):
        for ph, t0, t1 in PH:
            S = [r for r in RR if r[0] == lab and t0 <= r[1] < t1]
            if not S:
                continue
            cat = {k: np.concatenate([s[2][k] for s in S]) for k in S[0][2]}
            dmass = sum(s[3] for s in S)
            for bi, (lo, hi) in enumerate(list(BANDS) + [(0, 1e9)]):
                m = (cat['rr'] >= lo) & (cat['rr'] < hi)
                w = cat['w'][m]; Vn = cat['Vn'][m]; M = cat['M'][m]; vE = cat['vE'][m]
                bname = 'all' if bi == len(BANDS) else f'{lo:g}-{hi:g}'
                dm = dmass.sum() if bi == len(BANDS) else dmass[bi]
                if w.sum() == 0:
                    rows.append(f'{lab},{ph},{len(S)},{bname},0' + ',nan' * (21 if SUP else 15))
                    continue
                kj = krh(np.maximum(M, 1.0))
                e = {md: (f_of(M, Vn, vE, md) - 1) * kj for md in ('lin', 'lo', 'hi')}
                Mh = cat['Mhi'][m]; kh = krh(np.maximum(Mh, 1.0))
                eh = (f_of(Mh, Vn, vE, 'lin') - 1) * kh
                ehh = (f_of(Mh, Vn, vE, 'hi') - 1) * kh
                e2 = (f_of(np.full_like(M, 2.0), Vn, vE, 'lin') - 1) * krh(2.0)
                e3 = (f_of(np.full_like(M, 3.0), Vn, vE, 'lin') - 1) * krh(3.0)
                mw = lambda x: (x * w).sum() / w.sum()
                line = [lab, ph, str(len(S)), bname, f'{w.sum() / dm:.4f}']
                line += [f'{x:.4f}' for x in wq(Vn, w, (0.5, 0.9))] + [f'{mw(cat["dVn"][m]):.4f}']
                line += [f'{x:.3f}' for x in wq(M, w, (0.1, 0.5, 0.9))]
                line += [f'{100 * mw(e["lin"]):.3f}', f'{100 * mw(np.abs(e["lin"])):.3f}',
                         f'{100 * wq(np.abs(e["lin"]), w, (0.9,))[0]:.3f}',
                         f'{100 * mw(e["lo"]):.3f}', f'{100 * mw(e["hi"]):.3f}',
                         f'{100 * mw(eh):.3f}', f'{100 * mw(ehh):.3f}',
                         f'{100 * mw(e2):.3f}', f'{100 * mw(e3):.3f}']
                if SUP:
                    es = {md: (f_of(M, Vn, vE, md) - 1) * kj for md in SUP}
                    esh = (f_of(Mh, Vn, vE, 'sup') - 1) * kh
                    line += [f'{100 * mw(es["sup"]):.3f}', f'{100 * mw(np.abs(es["sup"])):.3f}',
                             f'{100 * wq(np.abs(es["sup"]), w, (0.9,))[0]:.3f}', f'{100 * mw(es["supex"]):.3f}',
                             f'{100 * mw(es["supdu"]):.3f}', f'{100 * mw(esh):.3f}']
                rows.append(','.join(line))
    open(a.out, 'w').write('\n'.join(rows) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
