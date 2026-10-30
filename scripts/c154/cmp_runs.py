#!/usr/bin/env python3
"""c154: compare two runs: every *.hst file row by row (common rows; the column-scaled max relative difference), and
every common bin dump (bytes after the header, as float arrays: the max abs difference per variable).
    cmp_runs.py A B [tol] [--from-row N]   -> prints, exit 0 iff every difference <= tol (default 0: bitwise)"""
import glob, os, sys, types
import numpy as np
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
import bin_convert as bc   # noqa: E402
a, b = sys.argv[1], sys.argv[2]
tol = float(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else 0.0
r0 = int(sys.argv[sys.argv.index('--from-row') + 1]) if '--from-row' in sys.argv else 0
ok = True
for fa in sorted(glob.glob(f'{a}/*.hst')):
    fb = f'{b}/{os.path.basename(fa)}'
    if not os.path.exists(fb): print('missing', fb); ok = False; continue
    A, B = np.loadtxt(fa, ndmin=2), np.loadtxt(fb, ndmin=2)
    # rows matched by time (a restart's rows start later)
    ta, tb = A[:, 0], B[:, 0]
    common = np.intersect1d(ta, tb)
    ia = np.searchsorted(ta, common); ib = np.searchsorted(tb, common)
    ia, ib = ia[r0:], ib[r0:]
    if len(ia) == 0: print(os.path.basename(fa), 'no common rows'); ok = False; continue
    X, Y = A[ia], B[ib]
    d = np.abs(X - Y) / np.maximum(np.abs(Y).max(axis=0), 1e-300)
    d[~np.isfinite(d)] = np.inf
    print(f'{os.path.basename(fa)}: {len(ia)} common rows, max col-scaled diff {d.max():.3e}')
    ok = ok and d.max() <= tol
for fa in sorted(glob.glob(f'{a}/bin/*.bin')):
    fb = f'{b}/bin/{os.path.basename(fa)}'
    if not os.path.exists(fb): continue
    P, Q = bc.read_binary(fa), bc.read_binary(fb)
    worst = 0.0
    for k in P['mb_data']:
        x = np.concatenate([np.asarray(v).ravel() for v in P['mb_data'][k]])
        y = np.concatenate([np.asarray(v).ravel() for v in Q['mb_data'][k]])
        worst = max(worst, float(np.max(np.abs(x - y))) if len(x) == len(y) else np.inf)
    print(f'{os.path.basename(fa)}: t={P["time"]:.6g}/{Q["time"]:.6g} max abs diff {worst:.3e}')
    ok = ok and worst <= tol and P['time'] == Q['time']
print('RESULT', 'PASS' if ok else 'FAIL', f'(tol {tol:g})')
sys.exit(0 if ok else 1)
