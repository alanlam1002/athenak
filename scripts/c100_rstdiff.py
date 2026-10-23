#!/usr/bin/env python3
"""C-100: variable-by-variable bitwise diff of two restart files of the same mesh (proj/tde layout:
per MeshBlock u0 (nmhd + nscal), b0 x1f/x2f/x3f, u_adm (17), then the <job> restart_state block
(outputs/restart_state.cpp: w0, flag slot, bcc0, T, [de_streak], delta_psi, delta_alpha_psi, MG psi, alpha,
pietax(4), pietabeta(4))). Per variable: cells that differ (all / interior), max |a-b|/|b|.

    c100_rstdiff.py A.rst B.rst [--nx 16 --ng 4 --nmhd 6]
"""
import argparse, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tde_dmde as D                               # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('a'); ap.add_argument('b')
ap.add_argument('--nx', type=int, default=16); ap.add_argument('--ng', type=int, default=4)
ap.add_argument('--nmhd', type=int, default=6)
ap.add_argument('--where', type=int, default=0, help='print the N worst cells of each differing CC variable')
a = ap.parse_args()
ta, tb = D.header(a.a), D.header(a.b)
assert ta['nmb'] == tb['nmb'] and ta['dsize'] == tb['dsize'], 'different layouts'
nst = int(re.search(r'^\s*restart_state_nvar\s*=\s*(\d+)', ta['par'], re.M).group(1)) \
    if re.search(r'restart_state_nvar', ta['par']) else 0
no = a.nx + 2 * a.ng; ncc = no**3
segs = [('u0', a.nmhd, (no, no, no)), ('b0.x1f', 1, (no, no, no + 1)), ('b0.x2f', 1, (no, no + 1, no)),
        ('b0.x3f', 1, (no + 1, no, no)), ('adm', 17, (no, no, no)), ('state', nst, (no, no, no))]
names = {'state': ['w0_%d' % i for i in range(a.nmhd)] + ['flag'] + ['bcc%d' % i for i in range(3)] + ['T']
         + ['dpsi', 'dapsi', 'mg_psi', 'mg_alpha'] + ['mg_x%d' % i for i in range(4)] + ['mg_b%d' % i for i in range(4)]}
print(f'time A {ta["time"]!r} B {tb["time"]!r}; nmb {ta["nmb"]}; state nvar {nst}')
ma = np.memmap(a.a, np.float64, 'r', ta['data_off'], (ta['nmb'], ta['dsize'] // 8))
mb = np.memmap(a.b, np.float64, 'r', tb['data_off'], (tb['nmb'], tb['dsize'] // 8))
off = 0
s = slice(a.ng, a.ng + a.nx)
x1min, x2min, x3min, x1max, x2max, x3max = ta['size'][:6]
W = np.array([x1max - x1min, x2max - x2min, x3max - x3min]); LO = np.array([x1min, x2min, x3min])
xbh, _ = D.frame_state(tb['par'], tb['time'])
D0 = np.asarray(mb[:, :ncc]).reshape(-1, no, no, no)


def where(dd, A, B, n):
    rel = np.where(dd, np.abs(A - B) / np.maximum(np.abs(B), 1e-300), -1.0)
    for f in np.argsort(rel.ravel())[::-1][:n]:
        m, k, j, i = np.unravel_index(f, rel.shape)
        ll = ta['ll'][m]; ext = W / 2.0**ll[3]; dx = ext[0] / a.nx
        x = LO + np.array([ll[0], ll[1], ll[2]]) * ext + (np.array([i, j, k]) - a.ng + 0.5) * dx
        r = np.sqrt(((x - np.asarray(xbh))**2).sum())
        gh = not (a.ng <= i < a.ng + a.nx and a.ng <= j < a.ng + a.nx and a.ng <= k < a.ng + a.nx)
        print(f'      mb {m} lev {ll[3]} ijk ({i},{j},{k}){" ghost" if gh else ""} r_BH {r:.3f} D {D0[m, k, j, i]:.3e} '
              f'A {A[m, k, j, i]:.10e} B {B[m, k, j, i]:.10e} rel {rel[m, k, j, i]:.2e}')
for nm, nv, shp in segs:
    n1 = int(np.prod(shp))
    for v in range(nv):
        A = np.asarray(ma[:, off:off + n1]).reshape((-1,) + shp)
        B = np.asarray(mb[:, off:off + n1]).reshape((-1,) + shp)
        off += n1
        d = (A != B) & ~(np.isnan(A) & np.isnan(B))
        if d.any():
            di = d[:, s, s, s].sum() if shp == (no, no, no) else -1
            rel = np.abs(A - B)[d] / np.maximum(np.abs(B[d]), 1e-300)
            lab = names.get(nm, [])[v] if nm in names and v < len(names[nm]) else f'{nm}[{v}]'
            print(f'  {lab:10s} differ {d.sum():9d} (interior {di}) of {d.size}; max rel {rel.max():.2e}; '
                  f'blocks {np.unique(np.where(d)[0]).size}')
            if a.where and shp == (no, no, no):
                where(d, A, B, a.where)
assert off * 8 == ta['dsize'], (off * 8, ta['dsize'])
print('done')
