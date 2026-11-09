#!/usr/bin/env python3
"""c253 (R-253 item 1, user-approved 2026-11-09): deliverable A's dM/dE across the resolution ladder, from restarts.
Per restart: tde_dmde.cells (E = -u_t - 1, the BH-frame Killing energy with xi-dot removed, cells with r_BH >= 3), E in
units of dE = R*/r_t^2 (R* 0.677, r_t 16.4). Two sets:
  all    = every cell outside r_BH < 3 (tde_dmde.py's distribution);
  debris = all minus the self-consistent Hill sphere about the remnant (centroid of rho >= 0.01 rho_c; r_H = r (M_H/3)^(1/3),
           iterated; as c142's M_Hsc), so the remnant's spike does not dominate a partial's histogram.
Reported: M, the bound fraction (E < 0), E/dE at the mass quantiles 0.1, 0.5, 0.9, and (1/M) dM/d(E/dE) on 161 edges in
[-4, 4] (npz).
    c253_dmde.py OUT_PREFIX LABEL=RST [LABEL=RST ...]"""
import sys, csv
import numpy as np
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde/wt_tde_scripts/scripts')
import tde_dmde as T   # noqa: E402

RC, MS = 3.24e-4, 7.47e-5
dE = 0.677/16.4**2
edges = np.linspace(-4, 4, 161)
out = sys.argv[1]
rows, H = [], {}


def wq(a, w, qs):
    o = np.argsort(a); c = np.cumsum(w[o]); c /= c[-1]
    return [float(a[o][min(np.searchsorted(c, q), len(a) - 1)]) for q in qs]


for arg in sys.argv[2:]:
    lab, fn = arg.split('=', 1)
    c = T.cells(fn, rcut=3.0)
    E, dM, rho = c['E']/dE, c['dM'], c['rho']
    X = np.stack([c['x'], c['y'], c['z']]); xb = np.asarray(c['xbh'], float)
    d1 = rho >= 0.01*RC
    if d1.any():
        xc = (X[:, d1]*dM[d1]).sum(1)/dM[d1].sum(); MH = dM[d1].sum()
    else:
        xc = X[:, np.argmax(rho)]; MH = 0.0
    r = float(np.linalg.norm(xc - xb)); dc = np.linalg.norm(X - xc[:, None], axis=0); rH = 0.0
    for _ in range(40):
        rH = r*(MH/3.0)**(1/3); MN = dM[dc < rH].sum()
        if abs(MN - MH) <= 1e-10*max(MH, 1e-300): break
        MH = MN
    for sel, s in (('all', np.ones_like(dM, bool)), ('debris', dc >= rH)):
        w = dM[s]; e = E[s]
        h, _ = np.histogram(e, bins=edges, weights=w); H[f'{lab}_{sel}'] = h/w.sum()/np.diff(edges)
        q = wq(e, w, (0.1, 0.5, 0.9))
        rows.append(dict(label=lab, set=sel, t=c['t'], M=w.sum()/MS, f_bound=w[e < 0].sum()/w.sum(),
                         E10=q[0], E50=q[1], E90=q[2], M_H=MH/MS, r_H=rH, r_rem=r))
        print(rows[-1], flush=True)
    del c, E, dM, rho, X, dc
np.savez(f'{out}.npz', edges=edges, **H)
with open(f'{out}.csv', 'w') as fo:
    wr = csv.DictWriter(fo, list(rows[0])); wr.writeheader(); wr.writerows(rows)
