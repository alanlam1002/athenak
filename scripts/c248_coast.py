#!/usr/bin/env python3
"""c248 (R-248 Q3): where the gas sits on the grid while the comoving frame coasts (ξ̇ frozen once nothing is tracked,
cfc.cpp:3181-3184). Per prim_3d / prim_3d_peri dump with t >= --t0: the densest cell (rho/rho_c, grid position, its block's
dx), the rho-centroid of rho >= 0.01 rho_max and its rms extent, the largest |x_i| of gas with rho >= 1e-3 / 1e-2 rho_max,
the mass fractions (of the gas in the box at t0) outside the grid-fixed pin box |x_i| < --pin (dyngr_tov.cpp:1523) and
in blocks coarser than the finest level present at t0, and the mass by block dx. Grid coordinates throughout.
    c248_coast.py OUT.csv RUNDIR [--t0 533.5] [--pin 64]"""
import glob
import sys
import types
import numpy as np
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde/wt_tde_scripts/scripts')
import bin_convert as bc   # noqa: E402
from c160_coldmax import xbh, par   # noqa: E402

B = 'cfc_tde_wd_imbh_parabolic_tracker'


def main():
    out, d = sys.argv[1], sys.argv[2]
    t0 = float(sys.argv[sys.argv.index('--t0') + 1]) if '--t0' in sys.argv else 533.5
    pin = float(sys.argv[sys.argv.index('--pin') + 1]) if '--pin' in sys.argv else 64.0
    rhoc = par(d, 'rhoc')
    fs = {}
    for f in glob.glob(f'{d}/bin/{B}.prim_3d.*.bin') + glob.glob(f'{d}/bin/{B}.prim_3d_peri.*.bin'):
        fs[round(bc.read_binary(f)['time'], 6)] = f
    rows, M0, hmin0 = [], None, None
    for t in sorted(fs):
        if t < t0 - 30:
            continue
        P = bc.read_binary(fs[t])
        g = np.asarray(P['mb_geometry'], float); n = P['nx1_out_mb']
        h = (g[:, 1] - g[:, 0])/n
        rho = np.concatenate([np.asarray(P['mb_data']['dens'][b]).ravel() for b in range(len(g))]).astype(float)
        ax = lambda lo: (g[:, lo][:, None] + (np.arange(n) + 0.5)[None, :]*h[:, None])
        X, Y, Z = ax(0), ax(2), ax(4)
        x = np.broadcast_to(X[:, None, None, :], (len(g), n, n, n)).ravel()
        y = np.broadcast_to(Y[:, None, :, None], (len(g), n, n, n)).ravel()
        z = np.broadcast_to(Z[:, :, None, None], (len(g), n, n, n)).ravel()
        hc = np.repeat(h, n**3)
        m = rho*hc**3
        i = int(np.argmax(rho)); rmax = rho[i]
        s2 = rho >= 1e-2*rmax; s3 = rho >= 1e-3*rmax
        xc = np.array([(m[s2]*c[s2]).sum()/m[s2].sum() for c in (x, y, z)])
        ext = np.sqrt((m[s2]*((x[s2] - xc[0])**2 + (y[s2] - xc[1])**2 + (z[s2] - xc[2])**2)).sum()/m[s2].sum())
        amax = lambda s: float(np.max(np.maximum(np.abs(x[s]), np.maximum(np.abs(y[s]), np.abs(z[s]))))) if s.any() else np.nan
        dense = rho >= 1e-6*rhoc
        if M0 is None and t >= t0 - 1e-6:
            M0, hmin0 = m[dense].sum(), h.min()
        xb = xbh(d, t)
        outpin = dense & ((np.abs(x) >= pin) | (np.abs(y) >= pin) | (np.abs(z) >= pin))
        r = dict(t=t, rhomax=rmax/rhoc, x_max=x[i], y_max=y[i], z_max=z[i], dx_at_max=hc[i],
                 r_max_bh=float(np.linalg.norm(np.array([x[i], y[i], z[i]]) - xb)),
                 xc=xc[0], yc=xc[1], zc=xc[2], ext=ext, amax_1e2=amax(s2), amax_1e3=amax(s3),
                 M_dense=m[dense].sum(), f_out_pin=m[outpin].sum()/m[dense].sum(),
                 dxmin=h.min(), nmb=len(g))
        for dxv in sorted(set(np.round(h, 6))):
            r[f'f_dx{dxv:g}'] = m[dense & (np.round(hc, 6) == dxv)].sum()/m[dense].sum()
        rows.append(r)
        print(f"t={t:7.1f} rho_max {rmax/rhoc:.3e} at ({x[i]:7.2f},{y[i]:7.2f},{z[i]:6.2f}) dx {hc[i]:.4f} | xc ({xc[0]:7.2f},{xc[1]:7.2f}) "
              f"ext {ext:6.2f} | max|x_i| 1e-2: {r['amax_1e2']:6.1f} 1e-3: {r['amax_1e3']:6.1f} | out pin {r['f_out_pin']:.2e} "
              f"| nmb {len(g)} dxmin {h.min():.4f}", flush=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k.startswith('f_dx'), list(rows[0].keys()).index(k) if k in rows[0] else 99, k))
    with open(out, 'w') as fo:
        fo.write(','.join(keys) + '\n')
        for r in rows:
            fo.write(','.join(str(r.get(k, 0.0)) for k in keys) + '\n')


if __name__ == '__main__':
    main()
