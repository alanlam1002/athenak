#!/usr/bin/env python3
"""c154 R-175 G1: no hole (ISO_A flat bed, comoving frame_vel1 = 0.4), killing_t1 with killing_weights = comoving
(weights (1, 0)) vs Valencia, both gr_dt = false (dt from the light speed, so times agree exactly), after 1 and 20
cycles. Round-off at the float32 dumps' resolution over the dense cells (rho >= 0.01 rho_c): rho, P, W v^x
within 2 ulps; W v^y, W v^z within 1e-9 of the dense max |W v^x| (as c150 gate_u1g); also the history rows
(column-scaled, 1e-9). The atmosphere is reported, not gated."""
import glob, sys, types
import numpy as np
sys.path.insert(0, sys.argv[1])
from gate_common import E, cmp, done
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
import bin_convert as bc   # noqa: E402
RHOC = 3.24e-4
ok = True
for n in ('1', '20'):
    fa = sorted(glob.glob(f'{E}/gG1v{n}/bin/*.prim_3d.*.bin'))[-1]
    fb = sorted(glob.glob(f'{E}/gG1t{n}/bin/*.prim_3d.*.bin'))[-1]
    a, b = bc.read_binary(fa), bc.read_binary(fb)
    if a['time'] != b['time']:
        print(f'G1 {n}: times differ {a["time"]} {b["time"]}'); ok = False; continue
    get = lambda P, k: np.concatenate([np.asarray(x).ravel() for x in P['mb_data'][k]]).astype(np.float64)
    dense = get(a, 'dens') >= 0.01*RHOC
    for k in ('dens', 'eint' if 'eint' in a['mb_data'] else 'press', 'velx'):
        x, y = get(a, k)[dense], get(b, k)[dense]
        ulp = np.spacing(np.abs(x).astype(np.float32)).astype(np.float64)
        r = float(np.max(np.abs(x - y)/ulp))
        print(f'G1 cycle {n} {k}: dense max |diff| = {r:.2f} float32 ulp'); ok = ok and r <= 2.0
    vm = np.max(np.abs(get(a, 'velx')[dense]))
    for k in ('vely', 'velz'):
        r = float(np.max(np.abs(get(a, k)[dense] - get(b, k)[dense])))/vm
        print(f'G1 cycle {n} {k}: dense max |diff| / max|Wv_x| = {r:.2e}'); ok = ok and r <= 1e-9
    at = ~dense
    print(f'G1 cycle {n} atmosphere (info): max |d rho|/rho = '
          f'{np.max(np.abs(get(a, "dens")[at] - get(b, "dens")[at])/np.maximum(get(a, "dens")[at], 1e-300)):.2e}')
    cmp(f'{E}/gG1t{n}', f'{E}/gG1v{n}', 1e-9)
done('G1', ok)
