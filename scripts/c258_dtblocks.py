#!/usr/bin/env python3
"""c258 (R-257 item 3(ii)): the Killing energy of L-011's dt-limiting blocks, and the finest-level census by binding class.
Inputs: a full-3D restart (tde_dmde.cells, rcut 3: E = -u_t - 1 in the BH frame, positions, levels) and LTS_BINS dumps
(code/lts/results/lj/lts_bins.*.txt.gz: gid level lx1 lx2 lx3 dt_b k_raw k hole in_H rank; header dt_fine).
(1) Limiting blocks: per dump, blocks with q = dt_b/dt_fine < QMAX and r_BH >= 16 (L-011's 'far' set); each block's box
    (absolute logical level, the restart's mesh bounds), widened by PAD M for the dump-restart time offset; the cells of
    the restart inside: mass-weighted E/dE quantiles (10/50/90%) and the unbound mass fraction (E > EPS dE).
(2) Census: cells at the finest level, by class: unbound (E > EPS dE) at r > R_UNB; unbound at r <= R_UNB; bound at
    r < R_ZONE; bound beyond. Volume (finest-level block-equivalents = volume/(16 dx)^3) and mass shares.
    c258_dtblocks.py OUT_PREFIX RST DUMP.gz [...] [--rzone 120] [--eps 0.05] [--dE 2.517e-3]"""
import argparse, gzip, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tde_dmde as T   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('rst'); ap.add_argument('dumps', nargs='+')
ap.add_argument('--rzone', type=float, default=120.0); ap.add_argument('--runb', type=float, default=30.0)
ap.add_argument('--eps', type=float, default=0.05); ap.add_argument('--dE', type=float, default=0.677 / 16.4**2)
ap.add_argument('--qmax', type=float, default=1.5); ap.add_argument('--pad', type=float, default=1.0)
a = ap.parse_args()
h = T.header(a.rst)
lo = np.array(h['size'][:3]); hi = np.array(h['size'][3:6]); root = h['root']
c = T.cells(a.rst, rcut=3.0)
X = np.stack([c['x'], c['y'], c['z']]); E = c['E'] / a.dE; dM = c['dM']; xbh = np.asarray(c['xbh'], float)
r = np.linalg.norm(X - xbh[:, None], axis=0)
print(f"restart t={c['t']:.3f} xbh={xbh} cells={E.size} root={root} box={lo}..{hi}", flush=True)
lines = []
for f in a.dumps:
    with gzip.open(f, 'rt') as fh:
        hd = fh.readline().split(); dtf = float(hd[hd.index('dt_fine') + 1]); t = float(hd[hd.index('time') + 1])
        B = np.loadtxt(fh, comments='#')
    lev, lx = B[:, 1].astype(int), B[:, 2:5]
    w = (hi - lo)[None, :] / 2.0**lev[:, None]
    b0 = lo[None, :] + lx * w; b1 = b0 + w
    bc_ = 0.5 * (b0 + b1); rb = np.linalg.norm(bc_ - xbh[None, :], axis=1)
    q = B[:, 5] / dtf
    sel = np.nonzero((q < a.qmax) & (rb >= 16.0))[0]
    for s in sel[np.argsort(q[sel])]:
        ins = np.all((X >= (b0[s] - a.pad)[:, None]) & (X < (b1[s] + a.pad)[:, None]), axis=0)
        if not ins.any():
            lines.append(f'{t:.2f} gid={int(B[s, 0])} L{lev[s]} q={q[s]:.3f} r={rb[s]:.1f}: no restart cells'); continue
        o = np.argsort(E[ins]); cm = np.cumsum(dM[ins][o]); cm /= cm[-1]
        qs = [E[ins][o][min(np.searchsorted(cm, p), o.size - 1)] for p in (0.1, 0.5, 0.9)]
        fu = dM[ins][E[ins] > a.eps].sum() / dM[ins].sum()
        lines.append(f'{t:.2f} gid={int(B[s, 0])} L{lev[s]} q={q[s]:.3f} r_BH={rb[s]:.1f} centre=({bc_[s][0]:.1f},{bc_[s][1]:.1f},'
                     f'{bc_[s][2]:.1f}) E/dE q10/50/90 = {qs[0]:+.3f}/{qs[1]:+.3f}/{qs[2]:+.3f} unbound(E>{a.eps}dE) mass frac {fu:.3f}')
with open(a.out + '_limiting.txt', 'w') as fo:
    fo.write('\n'.join(lines) + '\n')
print('\n'.join(lines[:40]), flush=True)
fin = c['lev'] == c['lev'].max()
dV = c['dV']; dxf = dV[fin][0]**(1 / 3); bv = (16 * dxf)**3
cls = {'unbound r>R_unb': fin & (E > a.eps) & (r > a.runb), 'unbound r<=R_unb': fin & (E > a.eps) & (r <= a.runb),
       'bound r<R_zone': fin & (E <= a.eps) & (r < a.rzone), 'bound r>=R_zone': fin & (E <= a.eps) & (r >= a.rzone)}
with open(a.out + '_census.txt', 'w') as fo:
    fo.write(f'finest level {int(c["lev"].max())} (dx {dxf:.5g}); R_unb {a.runb}, R_zone {a.rzone}, eps {a.eps}, dE {a.dE:.5g}\n')
    for k, s in cls.items():
        fo.write(f'{k:18s} blocks-equiv {dV[s].sum() / bv:9.0f}  volume share {dV[s].sum() / dV[fin].sum():.3f}  '
                 f'mass share {dM[s].sum() / dM[fin].sum():.3f}\n')
print(open(a.out + '_census.txt').read(), flush=True)
