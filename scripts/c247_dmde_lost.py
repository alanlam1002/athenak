#!/usr/bin/env python3
"""c247 (R-247 item 4): dM/dE of the mass lost from the remnant, from restarts (tde_dmde.cells: BH-frame Killing
energy E = -u_t - 1, comoving gauge handled there). Conventions as c206_outflow.py: centre = mass centroid of
rho >= 0.01 rho_c; lost = gas (rho > 1e-7 rho_c) at d > 1.5 R* from the centre; remnant = d < 1.5 R*.
Per restart: M_lost/M*, its E in units of dE = R*/r_t^2 (absolute, and relative to the remnant's mass-weighted
E_rem), the fraction bound to the hole (E < 0) and to the remnant (E - E_rem < 0, crude: no self-gravity term),
quantiles, and a histogram (1/M_lost) dM/d((E - E_rem)/dE) on [-4, 4].
    c247_dmde_lost.py OUT.npz RST [RST ...]"""
import sys
import numpy as np
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde/wt_tde_scripts/scripts')
import tde_dmde as T   # noqa: E402
RC, RS, MS, RT = 3.24e-4, 0.677, 7.47e-5, 16.4
dE = RS/RT**2
edges = np.linspace(-4, 4, 81)
out, res = sys.argv[1], {}
for i, fn in enumerate(sys.argv[2:]):
    c = T.cells(fn, rcut=3.0)
    rho, dM = c['rho'], c['dM']
    d0 = rho >= 0.01*RC
    cen = [np.average(c[k][d0], weights=dM[d0]) for k in ('x', 'y', 'z')]
    d = np.sqrt((c['x'] - cen[0])**2 + (c['y'] - cen[1])**2 + (c['z'] - cen[2])**2)
    gas = rho > 1e-7*RC
    L, R = gas & (d > 1.5*RS), d < 1.5*RS
    E = c['E']
    Erem = np.average(E[R], weights=dM[R])
    e = (E[L] - Erem)/dE; m = dM[L]
    q = lambda v, p: float(np.interp(p, np.cumsum(m[np.argsort(v)])/m.sum(), np.sort(v))) if m.size else np.nan
    h, _ = np.histogram(e, bins=edges, weights=m)
    r = dict(t=c['t'], M_lost=m.sum()/MS, M_rem=dM[R].sum()/MS, Erem_dE=Erem/dE, E_lost_mean_dE=np.average(E[L], weights=m)/dE,
             f_bound_bh=m[E[L] < 0].sum()/m.sum(), f_below_rem=m[e < 0].sum()/m.sum(),
             e10=q(e, 0.1), e50=q(e, 0.5), e90=q(e, 0.9), emin=e.min(), emax=e.max(),
             H=h/m.sum()/np.diff(edges))
    print(' '.join(f'{k}={v:.4g}' for k, v in r.items() if k != 'H'), flush=True)
    res.update({f'{k}_{i}': v for k, v in r.items()})
np.savez(out, edges=edges, dE=dE, **res)
