#!/usr/bin/env python3
"""c254: validate the integrator on a ballistic (--soft none) run: for a random subset of returned bound particles, the
analytic Schwarzschild return time to R_ret from the cell's conserved (eps, L): t = t0 + int dt/dr dr along the path
(out to apocentre if moving outward, then in to R_ret), dt/dr = (eps/f)/|u^r|, (u^r)^2 = eps^2 - f(1 + L^2/r^2), r from the
same isotropic -> areal mapping as the integrator. Prints the relative error statistics.
    c254tp_check.py IC.npz RUN.npz [--n 300]"""
import argparse
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

ap = argparse.ArgumentParser(); ap.add_argument('ic'); ap.add_argument('run'); ap.add_argument('--n', type=int, default=300)
a = ap.parse_args()
Z, R = np.load(a.ic), np.load(a.run)
assert str(R['soft']) == 'none'
idx, tev, st = R['idx'], R['t_ev'], R['status']
Rret, t0 = float(R['Rret']), float(R['t0'])
rng = np.random.default_rng(7)
cand = np.nonzero((st == 1) & (Z['E'][idx] < 0))[0]
pick = rng.choice(cand, size=min(a.n, len(cand)), replace=False)
err = []
for k in pick:
    i = idx[k]
    X = Z['X'][:, i]; ri = np.linalg.norm(X); r0 = ri*(1 + 0.5/ri)**2
    if r0 < Rret: continue
    eps = Z['eps'][i]; L2 = (Z['L'][:, i]**2).sum(); s = Z['s'][i]
    V = lambda r: eps**2 - (1 - 2/r)*(1 + L2/r**2)
    g = lambda r: (eps/(1 - 2/r))/np.sqrt(max(V(r), 1e-300))
    ra = brentq(V, max(r0, Rret), 1e7) if V(max(r0, Rret)) > 0 else max(r0, Rret)
    # time from r0 out to apocentre (if outbound) and in to R_ret; the sqrt singularity at ra is integrable (quad handles it)
    t = 0.0
    if s > 0:
        t += quad(g, r0, ra, limit=400)[0] + quad(g, Rret, ra, limit=400)[0]
    else:
        t += quad(g, Rret, r0, limit=400)[0]
    err.append((t0 + t - tev[k])/(tev[k] - t0))
err = np.abs(np.array(err))
print(f'{len(err)} returned bound particles: |relative error| of (t_return - t0): median {np.median(err):.2e}, 95% {np.quantile(err, 0.95):.2e}, max {err.max():.2e}')
