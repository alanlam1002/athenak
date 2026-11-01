#!/usr/bin/env python3
"""c160 (R-196, user-approved 2026-11-01): the DE-ON production pericentre census, offline on existing dumps.
Per 3D dump (prim_3d, prim_3d_peri) in [t_p - 50, t_p + 100] (and the next NEXT dumps after it):
  1. floored cells in the dense bands (rho >= 0.01, >= 0.1 rho_c; rho_c = <problem> rhoc, the initial central density): the
     temperature floor (T <= 1.0001 tfloor; the prim dump's temperature) -- count, mass, fraction of the band's mass, the hole-side
     share of the floored mass, the densest floored cell (rho/rho_c, r/R* from the rho-centroid of rho >= 0.01 rho_c, |x - x_BH|,
     level). The density floor (dfloor) cannot act in these bands (rho >= 0.01 rho_c >> dfloor); CRECOV is not dumped per cell
     (out.log totals are reported separately by --crecov);
  2. the core (r < 0.2 R* about the centroid): mass-weighted K/K0 and rho/rho_c; the density maximum: rho/rho_c, K/K0, r/R*,
     whether one of its same-block neighbours is floored;
  3. the DE branch (dediag_xy de_branch, the z = 0 plane at the same time): the mass share (rho dA) of each code in the dense
     bands and the core disc, codes per primitive_solver_hyd.hpp (proj/tde :2527-2596): 0 below rho_sw, 1 entropy,
     2 entropy by the second-law guard (flagged cell, K_en <= K_tr), 3 entropy after a floored / failed C2P, 4 energy (flagged),
     5 dense, C2P bad and the entropy inversion failed (the floored state kept);
  4. the dense-band mean K/K0 - 1 with and without floored cells; warm (> +5%) / cold (< -5%) shares of the unfloored dense mass.
    c160_census.py OUT.csv RUNDIR TP [--next 2] [--rstar 0.677]"""
import glob, re, sys, types
from multiprocessing import Pool
import numpy as np
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
import bin_convert as bc   # noqa: E402

GAM = 5.0/3.0
B = 'cfc_tde_wd_imbh_parabolic_tracker'
ARGS = {}


def par(d, k, dflt=None):
    m = re.search(rf'^\s*{k}\s*=\s*([-+.\deE]+)', open(f'{d}/parfile.par').read(), re.M)
    return float(m.group(1)) if m else dflt


def xbh(d, t):
    p0 = np.array([par(d, f'puncture_x{a}', 0.0) for a in (1, 2, 3)])
    f = glob.glob(f'{d}/*.user.hst')[0]
    hdr = [l for l in open(f) if l.startswith('#')][-1]
    if 'frame_disp_x1' not in hdr:
        return p0
    h = np.loadtxt(f, ndmin=2)
    c = [int(re.search(rf'\[(\d+)\]=frame_disp_x{a}', hdr).group(1)) - 1 for a in (1, 2, 3)]
    i = np.argmin(np.abs(h[:, 0] - t))
    return p0 - h[i, c]


def cells(P):
    g = np.asarray(P['mb_geometry'], dtype=float); n = P['nx1_out_mb']; nz = P['nx3_out_mb']
    lev = np.asarray(P['mb_logical'])[:, 3]
    X, L, DV = [], [], []
    for b in range(len(g)):
        dx = (g[b, 1] - g[b, 0])/n; c = (np.arange(n) + 0.5)*dx
        cz = g[b, 4] + (np.arange(nz) + 0.5)*(g[b, 5] - g[b, 4])/nz
        z, y, x = np.meshgrid(cz, g[b, 2] + c, g[b, 0] + c, indexing='ij')
        X.append(np.stack([x.ravel(), y.ravel(), z.ravel()], 1)); L.append(np.full(x.size, lev[b]))
        DV.append(np.full(x.size, dx*dx*((g[b, 5] - g[b, 4])/nz)))
    return g, np.concatenate(X), np.concatenate(L), np.concatenate(DV)


def one(f):
    d, tp, rs = ARGS['d'], ARGS['tp'], ARGS['rs']
    rhoc, k0, tfl = par(d, 'rhoc'), par(d, 'kappa'), par(d, 'tfloor')
    P = bc.read_binary(f); t = float(P['time']); q = P['mb_data']
    g, X, L, DV = cells(P)
    cat = lambda k: np.concatenate([np.asarray(q[k][b]).ravel() for b in range(len(g))]).astype(float)
    rho, p, T = cat('dens'), cat('press'), cat('temperature')
    K = p/rho**GAM/k0; fl = T <= 1.0001*tfl; w = rho*DV
    m01 = rho >= 0.01*rhoc; m1 = rho >= 0.1*rhoc
    cen = (X[m01]*w[m01, None]).sum(0)/w[m01].sum()
    xb = xbh(d, t); nh = (xb - cen)/np.linalg.norm(xb - cen)
    r = np.linalg.norm(X - cen, axis=1)/rs
    o = {'t': t, 't_minus_tp': t - tp, 'cen_bh': float(np.linalg.norm(xb - cen))}
    for lab, m in (('d01', m01), ('d1', m1)):
        mf = m & fl
        o[f'{lab}_nfl'] = int(mf.sum()); o[f'{lab}_mfl'] = float(w[mf].sum())
        o[f'{lab}_flfrac'] = float(w[mf].sum()/w[m].sum())
        o[f'{lab}_mean'] = float((w[m]*(K[m] - 1)).sum()/w[m].sum())
        mu = m & ~fl
        o[f'{lab}_mean_unfl'] = float((w[mu]*(K[mu] - 1)).sum()/w[mu].sum())
        o[f'{lab}_warm_unfl'] = float(w[mu][K[mu] > 1.05].sum()/w[mu].sum())
        o[f'{lab}_cold_unfl'] = float(w[mu][K[mu] < 0.95].sum()/w[mu].sum())
    mf = m01 & fl
    if mf.any():
        i = np.flatnonzero(mf)[np.argmax(rho[mf])]
        o.update(fl_max_rho=rho[i]/rhoc, fl_max_r=r[i], fl_max_dbh=float(np.linalg.norm(X[i] - xb)), fl_max_lev=int(L[i]),
                 fl_holeside=float(w[mf][((X[mf] - cen)@nh) > 0].sum()/w[mf].sum()), fl_levels=' '.join(map(str, sorted(set(L[mf].tolist())))))
    else:
        o.update(fl_max_rho=0.0, fl_max_r=np.nan, fl_max_dbh=np.nan, fl_max_lev=-1, fl_holeside=np.nan, fl_levels='')
    mc = r < 0.2
    o['core_K'] = float((w[mc]*K[mc]).sum()/w[mc].sum()) if mc.any() else np.nan
    o['core_rho'] = float((w[mc]*rho[mc]).sum()/w[mc].sum()/rhoc) if mc.any() else np.nan
    o['core_nfl'] = int((mc & fl).sum())
    i = int(np.argmax(rho))
    o.update(rhomax=rho[i]/rhoc, K_at_rhomax=K[i], rhomax_r=r[i], rhomax_lev=int(L[i]))
    # a floored neighbour of the density maximum (within 1.5 cell widths, any block)
    dx = DV[i]**(1/3); near = np.linalg.norm(X - X[i], axis=1) < 1.5*dx
    o['rhomax_next_to_floor'] = int((near & fl).any())
    # DE branch on the z = 0 plane (dediag_xy) at the same time
    fx = sorted(glob.glob(f'{d}/bin/{B}.dediag_xy.*.bin')); fp = sorted(glob.glob(f'{d}/bin/{B}.prim_xy.*.bin'))
    tx = ARGS['txy']; j = int(np.argmin(np.abs(tx - t)))
    o['xy_dt'] = float(tx[j] - t)
    S = bc.read_binary(fx[j]); Pq = bc.read_binary(fp[j])
    gs = np.asarray(S['mb_geometry'], dtype=float); ns = S['nx1_out_mb']
    br = np.concatenate([np.asarray(S['mb_data']['de_branch'][b]).ravel() for b in range(len(gs))])
    rq = np.concatenate([np.asarray(Pq['mb_data']['dens'][b]).ravel() for b in range(len(gs))]).astype(float)
    Tq = np.concatenate([np.asarray(Pq['mb_data']['temperature'][b]).ravel() for b in range(len(gs))]).astype(float)
    xs, ys, dA = [], [], []
    for b in range(len(gs)):
        h = (gs[b, 1] - gs[b, 0])/ns; c = (np.arange(ns) + 0.5)*h
        yy, xx = np.meshgrid(gs[b, 2] + c, gs[b, 0] + c, indexing='ij')
        xs.append(xx.ravel()); ys.append(yy.ravel()); dA.append(np.full(xx.size, h*h))
    xs, ys, dA = np.concatenate(xs), np.concatenate(ys), np.concatenate(dA)
    wq = rq*dA; rq2 = np.hypot(xs - cen[0], ys - cen[1])/rs
    for lab, m in (('d01', rq >= 0.01*rhoc), ('d1', rq >= 0.1*rhoc), ('core', rq2 < 0.2)):
        W = wq[m].sum()
        for c in range(6):
            o[f'xy_{lab}_br{c}'] = float(wq[m & (br == c)].sum()/W) if W > 0 else np.nan
        o[f'xy_{lab}_Tfl'] = float(wq[m & (Tq <= 1.0001*tfl)].sum()/W) if W > 0 else np.nan
    return o


def main():
    out, d, tp = sys.argv[1], sys.argv[2], float(sys.argv[3])
    nxt = int(sys.argv[sys.argv.index('--next') + 1]) if '--next' in sys.argv else 2
    rs = float(sys.argv[sys.argv.index('--rstar') + 1]) if '--rstar' in sys.argv else 0.677
    fs = sorted(set(glob.glob(f'{d}/bin/{B}.prim_3d.*.bin') + glob.glob(f'{d}/bin/{B}.prim_3d_peri.*.bin')))
    ts = {}
    for f in fs:
        ts.setdefault(round(bc.read_binary(f)['time'], 6), f)   # one file per time
    tt = np.array(sorted(ts))
    sel = [t for t in tt if tp - 50 - 1e-6 <= t <= tp + 100 + 1e-6] + [t for t in tt if t > tp + 100 + 1e-6][:nxt]
    txy = np.array([bc.read_binary(f)['time'] for f in sorted(glob.glob(f'{d}/bin/{B}.dediag_xy.*.bin'))])
    ARGS.update(d=d, tp=tp, rs=rs, txy=txy)
    with Pool(min(8, len(sel))) as pool:
        rows = pool.map(one, [ts[t] for t in sel])
    keys = list(rows[0].keys())
    with open(out, 'w') as fo:
        fo.write(','.join(keys) + '\n')
        for r in rows:
            fo.write(','.join(str(r.get(k, '')) for k in keys) + '\n')
    for r in rows:
        print(f"t-tp={r['t_minus_tp']:+7.1f} d01 floored {r['d01_nfl']:5d} ({100*r['d01_flfrac']:.3f}%) d1 {r['d1_nfl']:4d} "
              f"({100*r['d1_flfrac']:.3f}%) maxfl {r['fl_max_rho']:.3f} | core K {r['core_K']:.3f} rho {r['core_rho']:.3f} "
              f"| rhomax {r['rhomax']:.3f} K {r['K_at_rhomax']:.3f} r/R* {r['rhomax_r']:.2f} nf {r['rhomax_next_to_floor']} "
              f"| d01 {100*r['d01_mean']:+.2f}% unfl {100*r['d01_mean_unfl']:+.2f}% | xy d01 br " +
              ' '.join(f"{100*r[f'xy_d01_br{c}']:.1f}" for c in range(6)) + f" core br1 {100*r['xy_core_br1']:.1f}", flush=True)


if __name__ == '__main__':
    main()
