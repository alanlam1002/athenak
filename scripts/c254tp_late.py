#!/usr/bin/env python3
"""c254: the late return-rate slope from the gap-window runs (--ewin): for t beyond the full run's cutoff, every returning particle
has E inside the window, so the window run's rate is the full rate there. Poisson maximum-likelihood power law t^s on [ta, tb]
(and on two sub-ranges), the rate at 1e5 M (M*/M), the particle counts; bootstrap 1-sigma on s.
    c254tp_late.py LABEL=RUN.npz:ta:tb [...]"""
import sys
import numpy as np
from scipy.optimize import minimize_scalar
MS = 7.47e-5
def fit(x, a, b):
    lx = np.log(x)
    def nll(s):
        Z = np.log(b/a) if abs(s + 1) < 1e-9 else (b**(s + 1) - a**(s + 1))/(s + 1)
        return -(s*lx.sum() - len(x)*np.log(Z))
    return minimize_scalar(nll, bounds=(-10, 2), method='bounded').x
rng = np.random.default_rng(3)
print(f"{'run':16s} {'range (M)':>18s} {'n':>6s} {'slope':>7s} {'+-':>5s} {'1st half':>8s} {'2nd half':>8s} {'Mdot(1e5) M*/M':>14s}")
for arg in sys.argv[1:]:
    lab, rest = arg.split('=', 1); f, ta, tb = rest.split(':'); ta, tb = float(ta), float(tb)
    R = np.load(f); tp, W = float(R['tp']), float(R['W'])
    tr = R['t_ev'][R['status'] == 1] - tp
    x = tr[(tr >= ta) & (tr <= tb)]
    s = fit(x, ta, tb)
    bs = np.std([fit(rng.choice(x, len(x)), ta, tb) for _ in range(100)])
    tm = np.sqrt(ta*tb)
    s1 = fit(x[x <= tm], ta, tm); s2 = fit(x[x > tm], tm, tb)
    near = np.sum(np.abs(np.log(tr/1e5)) < 0.1)*W/(1e5*(np.exp(0.1) - np.exp(-0.1)))/MS
    print(f"{lab:16s} {ta:8.0f}-{tb:8.0f} {len(x):6d} {s:7.2f} {bs:5.2f} {s1:8.2f} {s2:8.2f} {near:14.3e}")
