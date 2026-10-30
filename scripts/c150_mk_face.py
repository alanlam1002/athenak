#!/usr/bin/env python3
"""c150 U1c/U1 per-face check decks: 3D 64x4x4 flat box, donor cell, FOFC off, 1 cycle, the
ghost/owner probe on (it prints every face flux at each stage), t = 0 tab of the x1 line.
mk_face.py NAME TYPE XIDOT SOLVER  (TYPE sound_wave | riemann; SOLVER hlle | hllc; U = n)"""
import subprocess, sys
name, typ, xd, solver = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
subprocess.run(['python3', 'mk_tube.py', name, '64', '0.3', xd, 'riemann_frame=boosted',
                'riemann_frame_U=eulerian', f'riemann_frame_solver={solver}'], check=True)
s = open(f'{name}.athinput').read()
rep = {'nx2 = 1\n': 'nx2 = 4\n', 'nx3 = 1\n': 'nx3 = 4\n', 'fofc = true': 'fofc = false',
       'reconstruct = ppmx': 'reconstruct = dc', 'type = riemann': f'type = {typ}',
       'nlim = -1': 'nlim = 1', 'ix1_bc = outflow': 'ix1_bc = periodic',
       'ox1_bc = outflow': 'ox1_bc = periodic'}
for a, b in rep.items():
    assert s.count(a) >= 1, a
    s = s.replace(a, b)
s = s.replace('xshock = 0.0\n', 'xshock = 0.0\namp = 0.3\nrho0 = 2.0\np0 = 1.0\nv1 = 0.3\nv2 = 0.2\n')
s = s.replace('variable = mhd_w_bcc\n', 'variable = mhd_w_bcc\nslice_x2 = 0.1\nslice_x3 = 0.1\n')
s += ('\n<dyn_grmhd>\nkilling_probe_cycles = 1\nface_probe_xmin = -1.0\nface_probe_xmax = 1.0\n'
      'face_probe_ymin = 0.0\nface_probe_ymax = 0.25\nface_probe_zmin = 0.0\nface_probe_zmax = 0.25\n')
open(f'{name}.athinput', 'w').write(s)
