#!/usr/bin/env python3
"""c259 (R-257 item 2): choosing variant A's fine cut (rho > f rho_max -> dx 1/32) so it covers the remnant, not the stream.
Per restart (tde_dmde.cells, rcut 3; remnant.bound_mass; the Hill radius r_H = r_bh (M_rem/3)^(1/3) as c142) and per f:
  bound_frac  = fraction of M_rem (the energy-bound set) in cells with rho > f rho_max
  hill_frac   = fraction of M_Hill in such cells
  V_out       = volume of cells with rho > f rho_max outside r_H (the stream that the fine level would also take), and the
                same in 1/32 MeshBlock equivalents (16 cells of 1/32 = 0.5 M per side: 0.125 M^3)
  r_max_in    = the largest |x - x_c| of a cell above the cut inside r_H
    c259_fcut.py LABEL=RST [...]"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import remnant as RM, tde_dmde as T   # noqa: E402
from multiprocessing import Pool
FS = (0.003, 0.01, 0.03, 0.1, 0.3)


def one(arg):
    lab, fn = arg
    c = T.cells(fn, rcut=3.0)
    Mb, xc, _, sel = RM.bound_mass(c)
    X = np.stack([c['x'], c['y'], c['z']]); r = np.linalg.norm(X - xc[:, None], axis=0)
    rbh = float(np.linalg.norm(xc - np.asarray(c['xbh'], float))); rH = rbh * (Mb / 3.0)**(1 / 3)
    inH = r < rH; rho, dM, dV = c['rho'], c['dM'], c['dV']; rm = rho.max()
    out = [f"{lab} t={c['t']:.1f} rho_max/rho_c={rm / 3.24e-4:.4f} M_rem={Mb / 7.47e-5:.3f} r_H={rH:.2f}"]
    for f in FS:
        a = rho > f * rm
        out.append(f"  f={f:<6g} bound_frac={dM[a & sel].sum() / dM[sel].sum():.3f} hill_frac={dM[a & inH].sum() / dM[inH].sum():.3f} "
                   f"V_out={dV[a & ~inH].sum():8.2f} M^3 ({dV[a & ~inH].sum() / 0.125:7.0f} blocks@1/32) "
                   f"V_in={dV[a & inH].sum():7.2f} ({dV[a & inH].sum() / 0.125:6.0f}) r_max_in={r[a & inH].max() if (a & inH).any() else 0:.2f}")
    s = '\n'.join(out); print(s, flush=True); return s


if __name__ == '__main__':
    with Pool(3) as p:
        res = p.map(one, [x.split('=', 1) for x in sys.argv[1:]])
    open('fcut.txt', 'w').write('\n'.join(res) + '\n')
