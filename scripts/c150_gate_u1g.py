#!/usr/bin/env python3
"""c150 U1 (retest) gate: ISO_A with <time> gr_dt = false (dt from the light speed in both runs, so the
times agree exactly), riemann_frame = boosted + U = eulerian (U1gN_feul) vs default HLLE (U1gN_def), N = 1, 20
cycles. The prim_3d dumps are float32, so round-off agreement is tested at that resolution, over the dense
cells (rho >= 0.01 rho_c): rho, P and W v^x within 2 float32 ulps of each cell's value; the transverse W v^y,
W v^z (noise about zero for a star moving along x) within 1e-9 of the dense max |W v^x|. The atmosphere's
difference is reported, not gated (the default HLLE's GR wave speeds carry ~1e-8 cancellation noise there,
primitive_solver_hyd.hpp:3069-3077, so it diverges from cycle 1). First version (1e-10 of the band max) was
below float32 resolution and failed on 1-ulp differences (job 8988424)."""
import glob, sys, types
import numpy as np
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
import bin_convert as bc   # noqa: E402
E = sys.argv[1]
RHOC = 3.24e-4
ok = True
for n in ('1', '20'):
    fa = sorted(glob.glob(f'{E}/U1g{n}_def/bin/*.prim_3d.*.bin'))[-1]
    fb = sorted(glob.glob(f'{E}/U1g{n}_feul/bin/*.prim_3d.*.bin'))[-1]
    a, b = bc.read_binary(fa), bc.read_binary(fb)
    if a['time'] != b['time']:
        print(f'U1g{n}: times differ {a["time"]} {b["time"]}'); ok = False; continue
    get = lambda P, k: np.concatenate([np.asarray(x).ravel() for x in P['mb_data'][k]])
    dense = get(a, 'dens') >= 0.01*RHOC
    vs = float(np.abs(get(a, 'velx')[dense]).max())
    out = []
    for k in ('dens', 'press', 'velx'):
        x, y = get(a, k), get(b, k)
        nu = (np.abs(x.astype(float) - y)/np.spacing(np.abs(x)))[dense].max()
        out.append(f'{k} {nu:.0f} ulp ({np.sum((x != y) & dense)} cells)')
        ok = ok and nu <= 2
    for k in ('vely', 'velz'):
        x, y = get(a, k).astype(float), get(b, k).astype(float)
        r = np.abs(x - y)[dense].max()/vs
        out.append(f'{k} {r:.1e} of |Wv_x|')
        ok = ok and r <= 1e-9
    na = np.sum((get(a, 'dens') != get(b, 'dens')) & ~dense)
    print(f'U1g{n} t={a["time"]:.6f} dense: ' + ', '.join(out) + f'; atmosphere rho differs in {na} cells')
print('U1g gate:', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
