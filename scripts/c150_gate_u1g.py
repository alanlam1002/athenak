#!/usr/bin/env python3
"""c150 U1 (retest) gate: ISO_A with <time> gr_dt = false (dt from the light speed in both runs, so the
times agree exactly), riemann_frame = boosted + U = eulerian (U1gN_feul) vs default HLLE (U1gN_def), N = 1, 20
cycles. The final prim_3d dump's dense cells (rho >= 0.01 rho_c) must agree to 1e-10 relative in rho, P and
W v^i (scaled by the dense band's max); the atmosphere's difference is reported, not gated (floor/FOFC
decisions there act on HLLE speeds with ~1e-8 cancellation noise in the default solver)."""
import glob, sys
import numpy as np
import types
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
    rho = get(a, 'dens')
    dense = rho >= 0.01*RHOC
    out = []
    for k in ('dens', 'velx', 'vely', 'velz', 'press'):
        x, y = get(a, k), get(b, k)
        sd = np.abs(x[dense]).max()
        dd = np.abs(x - y)[dense].max()/sd
        da = np.abs(x - y)[~dense].max()/max(np.abs(x[~dense]).max(), 1e-300)
        out.append(f'{k} dense {dd:.2e} atm {da:.2e}')
        ok = ok and dd <= 1e-10
    print(f'U1g{n} t={a["time"]:.6f}: ' + ', '.join(out))
print('U1g gate:', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
