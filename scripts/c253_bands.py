#!/usr/bin/env python3
"""c253 (R-253 item 2, follow-up): the stream's mass and its share at each dx per density band relative to the global
rho_max (the tracker's own variable: refine cut 1e-2 rho_max, derefine cut 1e-3 rho_max; dyngr_tov.cpp:1420-1424), for
single prim_3d dumps. Stream = 1e-6 <= rho/rho_c < 0.01 outside the self-consistent Hill sphere (as c253_stream.py).
    c253_bands.py RUNDIR DUMP_INDEX [...]"""
import sys, types
import numpy as np
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde/wt_tde_scripts/scripts')
import bin_convert as bc   # noqa: E402
from c160_coldmax import xbh   # noqa: E402
RC, MS = 3.24e-4, 7.47e-5
run = sys.argv[1]
for idx in sys.argv[2:]:
    P = bc.read_binary(f'{run}/bin/cfc_tde_wd_imbh_parabolic_tracker.prim_3d.{int(idx):05d}.bin'); t = P['time']
    g = np.asarray(P['mb_geometry'], float); n = P['nx1_out_mb']
    X, Y, Z, H = [], [], [], []
    for q in range(len(g)):
        c = [np.linspace(g[q, 2*k], g[q, 2*k+1], n+1) for k in range(3)]; c = [0.5*(x[1:]+x[:-1]) for x in c]
        zz, yy, xx = np.meshgrid(c[2], c[1], c[0], indexing='ij')
        X.append(xx.ravel()); Y.append(yy.ravel()); Z.append(zz.ravel()); H.append(np.full(n**3, (g[q, 1]-g[q, 0])/n))
    X, Y, Z, H = map(np.concatenate, (X, Y, Z, H))
    rho = np.concatenate([np.asarray(P['mb_data']['dens'][q]).ravel() for q in range(len(g))]).astype(float)
    m = rho*H**3; rmax = rho.max(); xb = xbh(run, t)
    rbh = np.sqrt((X-xb[0])**2 + (Y-xb[1])**2 + (Z-xb[2])**2)
    d1 = rho >= 0.01*RC; xc = np.array([np.average(w[d1], weights=m[d1]) for w in (X, Y, Z)]) if d1.any() else \
        np.array([X[np.argmax(rho)], Y[np.argmax(rho)], Z[np.argmax(rho)]])
    MH = m[d1].sum(); r = np.linalg.norm(xc - xb); dc = np.sqrt((X-xc[0])**2 + (Y-xc[1])**2 + (Z-xc[2])**2)
    for _ in range(40):
        rH = r*(MH/3.0)**(1/3); MN = m[dc < rH].sum()
        if abs(MN - MH) <= 1e-10*max(MH, 1e-300): break
        MH = MN
    st = (rho >= 1e-6*RC) & (rho < 0.01*RC) & (dc >= rH) & (rbh >= 3.0)
    hmin = H[st].min(); Ms = m[st].sum()
    print(f'{run.split("/")[-1]} t={t:.1f} rho_max={rmax/RC:.3g} rho_c  stream M={Ms/MS:.3g} M*  finest dx={hmin:g}')
    for lo, hi in ((1e-2, 1e9), (1e-3, 1e-2), (1e-4, 1e-3), (1e-5, 1e-4), (0, 1e-5)):
        b = st & (rho >= lo*rmax) & (rho < hi*rmax)
        if m[b].sum() == 0:
            print(f'   rho/rho_max [{lo:g},{hi:g}): none'); continue
        vol = (H[b]**3).sum()
        print(f'   rho/rho_max [{lo:g},{hi:g}): mass share {m[b].sum()/Ms:.3f}  volume {vol:.3g} M^3  at finest dx: mass {m[b & (H == hmin)].sum()/m[b].sum():.3f}, '
              f'volume {(H[b & (H == hmin)]**3).sum()/vol:.3f}  <dx>_vol {np.average(H[b], weights=H[b]**3):.3g}')
