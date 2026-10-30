#!/usr/bin/env python3
"""c150 U2T: the transverse-boost tubes (uniform v2 = 0.4, the Riemann problem along x1; grid
velocity 0 or (0, 0.4)) against a high-resolution reference: U2T_ref_N25600 (default HLLE, lab
grid) block-averaged onto each N. The comoving grid moves along x2 only, so x1 is the same
coordinate in every arm. L1 of rho, v^x, v^y (Eulerian, from W v) and p; and the reference's
own frame sensitivity (U2T_refR1_N25600, grid_rest R1 on the comoving grid)."""
import glob, re
import numpy as np


def load(d):
    f = sorted(glob.glob(f'{d}/tab/*.tab'))[-1]
    a = np.loadtxt(f)
    t = float(re.search(r'time=(\S+)', open(f).readline()).group(1))
    W = np.sqrt(1 + a[:, 4]**2 + a[:, 5]**2 + a[:, 6]**2)
    return t, np.stack([a[:, 3], a[:, 4]/W, a[:, 5]/W, a[:, 7]], axis=1)


tr, ref = load('U2T_ref_N25600')
tq, refq = load('U2T_refR1_N25600')
nr = len(ref)
print(f'reference t = {tr:.6f}; grid_rest R1 reference vs it, L1 (rho, vx, vy, p): ' +
      ' '.join(f'{x:.3e}' for x in np.abs(refq - ref).mean(axis=0)))
out = []
for c in ('lab', 'com', 'comR1', 'comR2'):
    for N in (200, 400, 800, 1600):
        t, q = load(f'U2T_{c}_N{N}')
        r = ref.reshape(N, nr//N, 4).mean(axis=1)
        e = np.abs(q - r).mean(axis=0)
        out.append((c, N, *e))
        print(f'{c:6s} N={N:5d} t={t:.6f}  L1 rho {e[0]:.4e}  vx {e[1]:.4e}  vy {e[2]:.4e}  p {e[3]:.4e}')
np.savetxt('u2t_errors.csv', np.array([[['lab', 'com', 'comR1', 'comR2'].index(o[0]), *o[1:]]
                                         for o in out]), delimiter=',',
           header='case(0 lab,1 com,2 comR1,3 comR2),N,L1rho,L1vx,L1vy,L1p')
