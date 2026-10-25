#!/usr/bin/env python3
"""R-131 item 4 (c119): data for the production-domain discussion, per 2x2 arm, against t.
 (ii) the extent of rho > 1e-10 rho_c gas and of the bound debris, and the BH-remnant / BH-debris-centroid
      separation, in each arm's own grid frame;
 (i)  the outer-boundary mass flux split into bound and unbound (estimated from the 3D dumps; MCLOSE's 'out'
      item, mass_closure.cpp, is the exact total but carries no split).
Two sources:
 - 3D dumps (prim_3d every 12 M, full box): no metric, so a far-field proxy. Around the puncture,
   alpha(r) = (1 - 1/2r)/(1 + 1/2r), psi = 1 + 1/2r (isotropic Schwarzschild, M = 1, r = |x - x_BH|), and the
   BH-frame Killing energy E ~ alpha W - 1 (the true shift beta' - xidot is ~0 away from the hole; tde_dmde.cells).
   Mass dm = rho W psi^6 dV. Validated against the exact E at the t = 300 restart (same time).
 - restarts (the kept milestones, ~5 per arm): the exact E = -u_t - 1 from tde_dmde.cells.
x_BH: BH frame the origin; comoving arms (puncture_x1, 0, 0) - frame_disp(t) from the user history (checked
against tde_dmde.frame_state on the CD t = 300 restart). Cells with r_BH < 3 are excluded.
Extents: r_max (largest r_BH), r99 (99th mass percentile of r_BH), Linf (largest |x_i| on the grid: the box is
+-64) of rho > 1e-10 rho_c gas, and the same for the bound (E < 0) part; the mass of each within 8 cells
(|x|_inf > 56) of the boundary. Separations: |x_rhomax - x_BH| and |centroid - x_BH| (all and bound
rho > 1e-10 rho_c gas). Boundary flux: sum over the six outer faces of the outermost cells of
rho W psi^6 (v^n - xidot^n) dA (BH frame xidot = 0), outflow only, split by E sign, per dump.
    c119_domain.py OUTDIR LABEL=RUNDIR [...]
"""
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tde_ensemble_plots import bc                 # noqa: E402
import tde_dmde as TD                             # noqa: E402

B = 'cfc_tde_wd_imbh_parabolic_tracker'
RHOC = 3.24e-4
THR = 1e-10 * RHOC
L = 64.0


def bh_of(d):
    par = open(f'{d}/parfile.par').read()
    import re
    mv = re.search(r'^puncture_moving\s*=\s*true', par, re.M) is not None
    px = float(re.search(r'^puncture_x1\s*=\s*(\S+)', par, re.M).group(1))
    h = np.loadtxt(f'{d}/{B}.user.hst')
    if not mv:
        return lambda t: (np.zeros(3), np.zeros(3))
    return lambda t: (np.array([px, 0, 0]) - np.array([np.interp(t, h[:, 0], h[:, 7 + k]) for k in range(3)]),
                      np.array([np.interp(t, h[:, 0], h[:, 4 + k]) for k in range(3)]))


def stats(t, X, Y, Z, rho, E, dm, xb, src):
    r = np.sqrt((X - xb[0])**2 + (Y - xb[1])**2 + (Z - xb[2])**2)
    k = r >= 3
    X, Y, Z, rho, E, dm, r = X[k], Y[k], Z[k], rho[k], E[k], dm[k], r[k]
    linf = np.maximum(np.maximum(np.abs(X), np.abs(Y)), np.abs(Z))
    out = dict(t=t, src=src)
    im = np.argmax(rho)
    out['sep_rhomax'] = r[im]
    for nm, m in (('gas', rho > THR), ('bound', (rho > THR) & (E < 0))):
        if not m.any():
            for q in ('mass', 'r_max', 'r99', 'Linf', 'edge_mass', 'sep_centroid'):
                out[f'{nm}_{q}'] = np.nan
            continue
        w = dm[m]
        o = np.argsort(r[m]); c = np.cumsum(w[o]) / w.sum()
        cx = np.array([(X[m] * w).sum(), (Y[m] * w).sum(), (Z[m] * w).sum()]) / w.sum()
        out.update({f'{nm}_mass': w.sum(), f'{nm}_r_max': r[m].max(), f'{nm}_r99': r[m][o][np.searchsorted(c, 0.99)],
                    f'{nm}_Linf': linf[m].max(), f'{nm}_edge_mass': w[linf[m] > L - 8].sum(),
                    f'{nm}_sep_centroid': np.linalg.norm(cx - xb)})
    out['unbound_mass'] = dm[(rho > THR) & (E >= 0)].sum()
    return out


def dump(arg):
    lab, d, fp = arg
    P = bc.read_binary(fp)
    t = float(P['time'])
    xb, xd = bh_of(d)(t)
    g = np.asarray(P['mb_geometry']); n = P['nx1_out_mb']
    q = {k: np.asarray(P['mb_data'][k]) for k in ('dens', 'velx', 'vely', 'velz')}
    Xs, Ys, Zs, R, E, M = [], [], [], [], [], []
    flux = {'bound': 0.0, 'unbound': 0.0}
    for b in range(len(g)):
        h = (g[b, 1] - g[b, 0]) / n
        c = lambda lo: lo + (np.arange(n) + 0.5) * h
        Z, Y, X = np.meshgrid(c(g[b, 4]), c(g[b, 2]), c(g[b, 0]), indexing='ij')
        r = np.maximum(np.sqrt((X - xb[0])**2 + (Y - xb[1])**2 + (Z - xb[2])**2), 0.6)
        psi = 1 + 0.5 / r
        al = (1 - 0.5 / r) / psi
        rho = q['dens'][b]
        u = [q['velx'][b], q['vely'][b], q['velz'][b]]
        W = np.sqrt(1 + psi**4 * sum(x * x for x in u))
        e = al * W - 1
        dm = rho * W * psi**6 * h**3
        Xs.append(X.ravel()); Ys.append(Y.ravel()); Zs.append(Z.ravel()); R.append(rho.ravel()); E.append(e.ravel())
        M.append(dm.ravel())
        for ax, coord in ((2, X), (1, Y), (0, Z)):                # array axis of x, y, z
            for side, idx in ((-1, 0), (1, -1)):
                lo, hi = g[b, 2 * (2 - ax)], g[b, 2 * (2 - ax) + 1]
                if (side < 0 and abs(lo + L) > 1e-9) or (side > 0 and abs(hi - L) > 1e-9):
                    continue
                sl = [slice(None)] * 3; sl[ax] = idx; sl = tuple(sl)
                vn = side * (u[2 - ax][sl] / W[sl] - xd[2 - ax])
                fl = np.where(vn > 0, rho[sl] * W[sl] * psi[sl]**6 * vn * h * h, 0.0)
                bd = e[sl] < 0
                flux['bound'] += fl[bd].sum(); flux['unbound'] += fl[~bd].sum()
    cat = np.concatenate
    o = stats(t, cat(Xs), cat(Ys), cat(Zs), cat(R), cat(E), cat(M), xb, 'dump')
    o['flux_bound'] = flux['bound']; o['flux_unbound'] = flux['unbound']
    return lab, o


def rst(arg):
    lab, d, fp, ns = arg
    c = TD.cells(fp, rcut=0.0, nscal=ns)
    return lab, stats(c['t'], c['x'], c['y'], c['z'], c['rho'], c['E'], c['dM'], c['xbh'], 'rst')


def main():
    out = sys.argv[1]
    jobs, rjobs = [], []
    for spec in sys.argv[2:]:
        lab, d = spec.split('=', 1)
        jobs += [(lab, d, f) for f in sorted(glob.glob(f'{d}/bin/{B}.prim_3d.*.bin'))]
        ns = 1                                    # DE-on restarts: u0 carries the tracer only (C-109)
        rjobs += [(lab, d, f, ns) for f in sorted(glob.glob(f'{d}/rst/{B}.*.rst'))]
    with Pool(12) as p:
        R = p.map(dump, jobs)
    with Pool(4) as p:
        RR = p.map(rst, rjobs)
    keys = ['t', 'src', 'sep_rhomax', 'gas_mass', 'gas_r_max', 'gas_r99', 'gas_Linf', 'gas_edge_mass',
            'gas_sep_centroid', 'bound_mass', 'bound_r_max', 'bound_r99', 'bound_Linf', 'bound_edge_mass',
            'bound_sep_centroid', 'unbound_mass', 'flux_bound', 'flux_unbound']
    rows = ['label,' + ','.join(keys)]
    for lab, o in sorted(R + RR, key=lambda x: (x[0], x[1]['t'], x[1]['src'])):
        rows.append(lab + ',' + ','.join(str(o.get(k, '')) if k == 'src' else
                                         (f'{o[k]:.6g}' if k in o else '') for k in keys))
    open(f'{out}/c119_domain.csv', 'w').write('\n'.join(rows) + '\n')
    # time-integrated boundary outflow (trapezoid over the 12-M dumps)
    lines = ['label,int_flux_bound,int_flux_unbound']
    for lab in dict.fromkeys(r[0] for r in R):
        S = sorted([o for l, o in R if l == lab], key=lambda o: o['t'])
        t = np.array([o['t'] for o in S])
        lines.append(f'{lab},{np.trapezoid([o["flux_bound"] for o in S], t):.4e},'
                     f'{np.trapezoid([o["flux_unbound"] for o in S], t):.4e}')
    open(f'{out}/c119_flux.csv', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(rows)); print('\n'.join(lines))


if __name__ == '__main__':
    main()
