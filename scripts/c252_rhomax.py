#!/usr/bin/env python3
"""c252 (R-250 item 3): the global density maximum rho_max/rho_c per prim_3d / prim_3d_peri dump (rho_c = 3.24e-4).
    c252_rhomax.py OUT.csv RUNDIR"""
import sys, types, glob, csv
import numpy as np
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
import bin_convert as bc   # noqa: E402
RC = 3.24e-4
rows = []
for f in sorted(glob.glob(f'{sys.argv[2]}/bin/*.prim_3d*.bin')):
    P = bc.read_binary(f)
    m = max(float(np.max(np.asarray(P['mb_data']['dens'][q]))) for q in range(len(P['mb_geometry'])))
    rows.append((P['time'], m/RC)); print(f'{P["time"]:.2f} {m/RC:.4f}', flush=True)
rows.sort()
with open(sys.argv[1], 'w') as fo:
    w = csv.writer(fo); w.writerow(['t', 'rhomax']); w.writerows(rows)
