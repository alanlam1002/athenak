#!/usr/bin/env python3
"""c130 tde_debris rung 1 (R-140 item 9): the stretching-filament test, from the 3D primitive dumps.
Per dump (cycle, t):
  - width resolution: over the cells the width test sees (rho > rho_F with all 6 face neighbours
    > rho_F, tde_debris_refine.hpp:136-149; block-interior neighbours only here), the local width
    l = rho/|grad rho| (centred differences) in cells, l/dx; min and the fraction with l/dx < N_H,
    split by whether the cell's block sits at the cap amr_level_stream;
  - the filament's transverse width H from a fit ln rho = a - r_perp^2/2H^2 (rho > rho_F,
    |x - x_c| < L/4 of the ballistic centre) against the ballistic H0 (1 - q t/t_s), and cells per H at the axis;
  - blocks per level; per block the criterion's want_ref/want_deref recomputed from the dump
    (TDEDebrisCellWant, tde_debris_refine.hpp:60-84; core/ladder/width/Jeans; no BH box);
  - derefinement behind: sibling groups (8 children of one parent) all above their want_deref,
    i.e. groups the criterion would derefine; tracked across dumps by logical location, the
    streak (cycles) for which such a group survives.
    c130_filament.py RUNDIR OUTCSV [OUT.png]  (deck values below must match the run's r1.par)
"""
import glob
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'vis', 'python'))
import bin_convert as bc   # noqa: E402

RHO_REF = 3.24e-4
P = dict(core_frac=1e-2, floor_frac=1e-5, ladder_factor=8.0, nh=2.0, nj=0.0, lcore=4, lfloor=0,
         lstream=4, s=0.5, gam=5.0/3.0, min_res=10, h0=1.0, ts=40.0, q=0.75, x0=-6.0, vx=0.25,
         L0=8.0)


def cell_want(rho, prs, ell, gate, sc):
    """per-cell (ladder, width, jeans) wants, TDEDebrisCellWant with scale sc"""
    rho_a, rho_f = sc*P['core_frac']*RHO_REF, sc*P['floor_frac']*RHO_REF
    ok = rho > rho_f
    lad = np.where(ok & (rho <= rho_a),
                   np.maximum(P['lcore'] - np.ceil(np.log(rho_a/np.maximum(rho, 1e-300)) /
                                                   np.log(P['ladder_factor'])), P['lfloor']), -1)
    with np.errstate(divide='ignore', invalid='ignore'):
        lw = np.ceil(np.log2(P['nh']/sc*1.0/ell))     # N/sc: the fixed hysteresis
    wid = np.where(ok & gate & np.isfinite(lw), np.minimum(np.maximum(lw, 0), P['lstream']), -1)
    h = 1.0 + P['gam']/(P['gam'] - 1.0)*prs/np.maximum(rho, 1e-300)
    lam = np.sqrt(P['gam']*prs/(np.maximum(rho, 1e-300)*h))*np.sqrt(np.pi/np.maximum(rho, 1e-300))
    with np.errstate(divide='ignore', invalid='ignore'):
        lj = np.ceil(np.log2(P['nj']/sc*1.0/lam))
    jea = np.where(ok & (prs > 0) & np.isfinite(lj), np.minimum(np.maximum(lj, 0), P['lcore']), -1)
    return lad, wid, jea


def analyse(f):
    d = bc.read_binary(f)
    t, cyc = float(d['time']), int(d['cycle'])
    g, lg = np.asarray(d['mb_geometry']), np.asarray(d['mb_logical'])
    rho_f = P['floor_frac']*RHO_REF
    rows, blocks = [], []
    lmin, nbad_cap, nbad_free, ngate = np.inf, 0, 0, 0
    tau = t/P['ts']
    xc = P['x0'] + P['vx']*t                       # ballistic centre
    Lh = 0.5*P['L0']*(1 + tau)
    fit_r2, fit_lr, axis_dx = [], [], []
    for b in range(len(g)):
        rho = np.asarray(d['mb_data']['dens'][b]); prs = np.asarray(d['mb_data']['eint'][b]) \
            if 'eint' in d['mb_data'] else np.asarray(d['mb_data']['press'][b])
        nz, ny, nx = rho.shape
        dx = (g[b, 1] - g[b, 0])/nx
        lev = int(round(np.log2(1.0/dx)))
        z = g[b, 4] + (np.arange(nz) + 0.5)*dx
        y = g[b, 2] + (np.arange(ny) + 0.5)*dx
        x = g[b, 0] + (np.arange(nx) + 0.5)*dx
        Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
        # gate and l on interior cells (6 neighbours inside the block)
        c = rho[1:-1, 1:-1, 1:-1]
        nb = [rho[2:, 1:-1, 1:-1], rho[:-2, 1:-1, 1:-1], rho[1:-1, 2:, 1:-1], rho[1:-1, :-2, 1:-1],
              rho[1:-1, 1:-1, 2:], rho[1:-1, 1:-1, :-2]]
        gate = (c > rho_f) & np.all([n > rho_f for n in nb], axis=0)
        gr = np.sqrt(((nb[0] - nb[1])/(2*dx))**2 + ((nb[2] - nb[3])/(2*dx))**2 +
                     ((nb[4] - nb[5])/(2*dx))**2)
        with np.errstate(divide='ignore', invalid='ignore'):
            ell = np.where(gate & (gr > 0), c/gr, np.inf)
        if gate.any():
            ld = ell[gate]/dx
            ngate += gate.sum()
            lmin = min(lmin, ld.min())
            bad = (ld < P['nh']).sum()
            if lev >= P['lstream']:
                nbad_cap += bad
            else:
                nbad_free += bad
        cp = prs[1:-1, 1:-1, 1:-1]
        wr = cell_want(c, cp, ell, gate, 1.0)
        wd = cell_want(c, cp, ell, gate, P['s'])
        core_r = P['lcore'] if rho.max() > P['core_frac']*RHO_REF else -1
        core_d = P['lcore'] if rho.max() > P['s']*P['core_frac']*RHO_REF else -1
        want_r = max([core_r] + [int(w.max()) for w in wr])
        want_d = max([core_d] + [int(w.max()) for w in wd])
        blocks.append((tuple(lg[b][:3]), lev, want_r, want_d, g[b, 0], g[b, 1]))
        # transverse width in the central part of the filament
        sel = (np.abs(X - xc) < 0.5*Lh) & (rho > rho_f)
        fit_r2.append((Y[sel]**2 + Z[sel]**2).ravel()); fit_lr.append(np.log(rho[sel]).ravel())
        if g[b, 0] <= xc <= g[b, 1] and g[b, 2] <= 0 <= g[b, 3] and g[b, 4] <= 0 <= g[b, 5]:
            axis_dx.append(dx)
    r2, lr = np.concatenate(fit_r2), np.concatenate(fit_lr)
    H = np.sqrt(-0.5/np.polyfit(r2, lr, 1)[0]) if r2.size > 10 else np.nan   # ln rho = a - r^2/2H^2
    Hb = P['h0']*(1 - P['q']*tau)
    nlev = defaultdict(int)
    for bl in blocks:
        nlev[bl[1]] += 1
    under = sum(1 for bl in blocks if bl[1] < bl[2])
    # sibling groups the criterion would derefine (all 8 children above want_deref)
    grp = defaultdict(list)
    for (l1, l2, l3), lev, wr_, wd_, *_ in blocks:
        if lev > 0:
            grp[(l1//2, l2//2, l3//2, lev - 1)].append(lev > wd_)
    dgrp = {k for k, v in grp.items() if len(v) == 8 and all(v)}
    return dict(t=t, cycle=cyc, nblocks=len(blocks), lev=dict(nlev), ngate=int(ngate),
                lmin_cells=lmin, nbad_free=int(nbad_free), nbad_cap=int(nbad_cap),
                H=H, H_ballistic=Hb, cells_per_H=(H/min(axis_dx)) if axis_dx else np.nan,
                under=under, dgroups=dgrp)


def main():
    run, out = sys.argv[1], sys.argv[2]
    fs = sorted(glob.glob(f'{run}/bin/*.prim3d.*.bin'))
    R = [analyse(f) for f in fs]
    first = {}
    streak = defaultdict(int)
    lines = ['t,cycle,nblocks,lev0,lev1,lev2,lev3,lev4,ngate,lmin_cells,nbad_free,nbad_cap,H,H_ballistic,'
             'cells_per_H,blocks_under_want_ref,deref_groups_pending,max_pending_age_cycles']
    for r in R:
        for k in list(first):
            if k not in r['dgroups']:
                del first[k]
        for k in r['dgroups']:
            first.setdefault(k, r['cycle'])
        age = max([r['cycle'] - c for c in first.values()], default=0)
        lv = [r['lev'].get(l, 0) for l in range(5)]
        lines.append(f"{r['t']:.4f},{r['cycle']},{r['nblocks']}," + ','.join(map(str, lv)) +
                     f",{r['ngate']},{r['lmin_cells']:.4g},{r['nbad_free']},{r['nbad_cap']},"
                     f"{r['H']:.4g},{r['H_ballistic']:.4g},{r['cells_per_H']:.4g},{r['under']},"
                     f"{len(r['dgroups'])},{age}")
    open(out, 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    if len(sys.argv) > 3:                      # optional summary figure
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        t = np.array([r['t'] for r in R])
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        ax[0].plot(t, [r['lmin_cells'] for r in R], 'o-', ms=3, label='min l/dx (gated cells)')
        ax[0].axhline(P['nh'], color='k', ls='--', lw=1, label='N_H')
        ax[0].set_xlabel('t [M]'); ax[0].set_ylabel('cells across l = rho/|grad rho|'); ax[0].legend()
        for l in range(5):
            ax[1].plot(t, [r['lev'].get(l, 0) for r in R], '-', label=f'level {l}')
        ax[1].set_xlabel('t [M]'); ax[1].set_ylabel('MeshBlocks'); ax[1].legend(fontsize=7)
        ax[2].plot(t, [r['H'] for r in R], 'o', ms=3, label='H (fit)')
        ax[2].plot(t, [r['H_ballistic'] for r in R], '-', label='H0 (1 - q t/t_s)')
        ax[2].plot(t, [r['cells_per_H']/10 for r in R], 's', ms=3, label='cells per H / 10 (axis)')
        ax[2].set_xlabel('t [M]'); ax[2].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(sys.argv[3], dpi=90)


if __name__ == '__main__':
    main()
