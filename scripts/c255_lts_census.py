#!/usr/bin/env python3
"""c255 (R-255 item 5(a)): the local-time-stepping gain, measured on a post-disruption P10 prim_3d dump.
Per MeshBlock (16^3, from mb_geometry): level from dx; the block's CFL limit dt_b = C * dx / max_cells(lambda), with the
coordinate signal speed (BH frame, ARGUED approximation of dyn_grmhd_newdt.cpp's GR fast speeds without B):
  lambda = alpha (|v| + c_s)/(1 + |v| c_s) / psi^2 + |beta_bg|,
  v = |u~|/W (u~ = the dump's vel*, W = sqrt(1 + psi^4 |u~|^2)), c_s^2 = Gamma P/(rho h), h = 1 + Gamma/(Gamma-1) P/rho,
  isotropic-Schwarzschild alpha = (1 - M/2r)/(1 + M/2r), psi = 1 + M/2r about x_BH (<cfc> puncture position minus xi from the
  user.hst frame_disp), beta_bg = 0 (BH frame; the trumpet shift matters only at r < ~5, inside or next to the excision),
  cells with r < r_exc (excised) skipped. --frame-shift V subtracts nothing; --add-shift adds the coasting |xidot| (production
  P10_r82) to lambda, to reproduce that run's limiter.
Outputs per level: blocks, cell share, min dt_b (and its location), and the costs:
  global stepping: N_total * T/dt_min;
  per-level LTS (power-of-2 subcycling, dt_L = min over the level, then dt'_L = dt_min * 2^floor(log2(min(dt_L', 2 dt'_{L+1}...))),
    i.e. each level steps at the largest power-of-2 multiple of dt_min that is <= its own limit and <= 2x the next finer level's
    step (Berger-Oliger nesting));
  per-block LTS (ideal, power-of-2 multiples of dt_min, no nesting constraint), and with the 2:1 neighbour constraint ignored.
    c255_lts_census.py PRIM_3D.bin RUNDIR [--cfl 0.4] [--rexc 2.5] [--add-shift S]"""
import sys, types, glob, re, argparse
import numpy as np
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde/wt_tde_scripts/scripts')
import bin_convert as bc   # noqa: E402
from c160_coldmax import xbh   # noqa: E402

GAM = 5.0/3.0
ap = argparse.ArgumentParser(); ap.add_argument('dump'); ap.add_argument('run')
ap.add_argument('--cfl', type=float, default=0.4); ap.add_argument('--rexc', type=float, default=2.5)
ap.add_argument('--add-shift', type=float, default=0.0)
a = ap.parse_args()
P = bc.read_binary(a.dump); t = P['time']; g = np.asarray(P['mb_geometry'], float); n = P['nx1_out_mb']
xb = xbh(a.run, t)
nb = len(g); dxb = (g[:, 1] - g[:, 0])/n
dtb = np.full(nb, np.inf); rmin_b = np.zeros(nb); lam_b = np.zeros(nb)
c = (np.arange(n) + 0.5)/n
for q in range(nb):
    x = g[q, 0] + c*(g[q, 1] - g[q, 0]); y = g[q, 2] + c*(g[q, 3] - g[q, 2]); z = g[q, 4] + c*(g[q, 5] - g[q, 4])
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    r = np.sqrt((X - xb[0])**2 + (Y - xb[1])**2 + (Z - xb[2])**2)
    rho = np.asarray(P['mb_data']['dens'][q], float); p = np.asarray(P['mb_data']['press'][q], float)
    ux, uy, uz = (np.asarray(P['mb_data'][k][q], float) for k in ('velx', 'vely', 'velz'))
    psi = 1.0 + 0.5/np.maximum(r, 1e-3); al = (1.0 - 0.5/np.maximum(r, 1e-3))/psi
    u2 = psi**4*(ux*ux + uy*uy + uz*uz); W = np.sqrt(1.0 + u2); v = np.sqrt(u2)/W
    h = 1.0 + GAM/(GAM - 1.0)*p/np.maximum(rho, 1e-300); cs = np.sqrt(np.clip(GAM*p/(np.maximum(rho, 1e-300)*h), 0, 1))
    lam = al*(v + cs)/(1.0 + v*cs)/psi**2 + a.add_shift
    ok = r >= a.rexc
    if ok.any():
        lam_b[q] = lam[ok].max(); dtb[q] = a.cfl*dxb[q]/lam_b[q]; rmin_b[q] = r.min()
levs = np.round(np.log2(dxb.max()/dxb)).astype(int)
dtmin = dtb[np.isfinite(dtb)].min()
L = sorted(set(levs))
print(f'dump {a.dump.split("/")[-1]} t = {t:.2f}, blocks {nb}, x_BH = {np.round(xb, 2)}, dt_min = {dtmin:.4e} (CFL {a.cfl}, '
      f'add_shift {a.add_shift})')
print('level  dx       blocks  share   min dt_L   r_BH of the limiting block  max lambda')
dtL = {}
for l in L:
    s = levs == l; d = dtb[s]; i = np.argmin(d)
    dtL[l] = d.min()
    print(f'{l:5d}  {dxb[s][0]:.4f}  {s.sum():6d}  {s.mean():.4f}  {d.min():.4e}  {rmin_b[s][i]:8.2f}               {lam_b[s].max():.3f}')
# per-level Berger-Oliger: from the finest level up, dt'_L = dt_min * 2^k <= min(dt_L, 2 dt'_{L+1})
step = {}
prev = None
for l in sorted(L, reverse=True):
    cap = dtL[l] if prev is None else min(dtL[l], 2.0*prev)
    k = int(np.floor(np.log2(cap/dtmin) + 1e-12)); step[l] = dtmin*2.0**max(k, 0); prev = step[l]
cost_glob = nb/dtmin
cost_lvl = sum((levs == l).sum()/step[l] for l in L)
kb = np.floor(np.log2(dtb/dtmin) + 1e-12); kb[~np.isfinite(kb)] = 0
cost_blk = (1.0/(dtmin*2.0**np.maximum(kb, 0))).sum()
print('per-level steps (units of dt_min): ' + ', '.join(f'L{l}: {step[l]/dtmin:.0f}' for l in L))
print(f'gain per-level LTS = {cost_glob/cost_lvl:.2f}; ideal per-block (power-of-2, no nesting) = {cost_glob/cost_blk:.2f}')
fin = levs == max(L)
f = fin.mean(); r = np.median(dtb[~fin & np.isfinite(dtb)])/dtmin if (~fin).any() else 1.0
print(f'two-class view: f (finest-level block share) = {f:.3f}, r (median coarse dt_b / dt_min) = {r:.1f}, '
      f'1/[f + (1-f)/r] = {1.0/(f + (1.0 - f)/r):.2f}')
