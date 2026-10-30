#!/usr/bin/env python3
"""c150 U2: L1 errors of the 1D Riemann runs against the exact solution (srrp_exact.py).
Grid x' (frame_vel1 = xd) maps to the lab x = x' + xd t; dyn_grmhd's velx = W v (Eulerian =
lab observers for alpha = 1 and a constant shift)."""
import glob, re, sys
import numpy as np
import c150_srrp_exact as S
g = 5.0/3.0
cases = {'rest': (0.0, 0.0), 'lab': (0.4, 0.0), 'com': (0.4, 0.4), 'comR1': (0.4, 0.4),
         'comR2': (0.4, 0.4)}
rows = []
for c, (v, xd) in cases.items():
    smp, _ = S.solve((10.0, 13.33, v), (1.0, 0.1, v), g)
    for N in (200, 400, 800, 1600):
        f = sorted(glob.glob(f'U2_{c}_N{N}/tab/*.tab'))[-1]
        t = float(re.search(r'time=(\S+)', open(f).readline()).group(1))
        a = np.loadtxt(f)
        x, rho, ux, p = a[:, 2], a[:, 3], a[:, 4], a[:, 7]
        vv = ux/np.sqrt(1 + ux*ux)
        ex = np.array([smp((xi + xd*t)/t) for xi in x])
        dx = 1.0/N
        e = [np.sum(np.abs(rho - ex[:, 0]))*dx, np.sum(np.abs(vv - ex[:, 2]))*dx,
             np.sum(np.abs(p - ex[:, 1]))*dx]
        rows.append((c, N, t, *e))
        print(f'{c:6s} N={N:5d} t={t:.6f}  L1 rho {e[0]:.4e}  v {e[1]:.4e}  p {e[2]:.4e}')
np.savetxt('u2_errors.csv', np.array([[list(cases).index(r[0]), *r[1:]] for r in rows]),
           delimiter=',', header='case(0 rest,1 lab,2 com,3 comR1,4 comR2),N,t,L1rho,L1v,L1p')
