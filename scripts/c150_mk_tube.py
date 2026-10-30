#!/usr/bin/env python3
"""c150 (R-168 U1/U2): 1D Riemann problems on dyngr_frame_shift (type riemann), flat space.
mk_tube.py NAME N V XIDOT [key=value ...]: V = the tube's Eulerian velocity (both states),
XIDOT = <problem> frame_vel1 (the grid's velocity); extra keys go to <mhd>."""
import sys
name, n, v, xd = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
mhd_extra = '\n'.join(a.replace('=', ' = ', 1) for a in sys.argv[5:])
tlim = 0.4
deck = f"""<comment>
problem = c150 {name}: 1D relativistic Riemann problem (rho, p) = (10, 13.33) | (1, 0.1), Gamma 5/3, tube velocity {v}, grid velocity {xd}

<job>
basename = tube

<mesh>
nghost = 4
nx1 = {n}
x1min = -0.5
x1max = 0.5
ix1_bc = outflow
ox1_bc = outflow
nx2 = 1
x2min = 0.0
x2max = 1.0
ix2_bc = periodic
ox2_bc = periodic
nx3 = 1
x3min = 0.0
x3max = 1.0
ix3_bc = periodic
ox3_bc = periodic

<meshblock>
nx1 = {n}
nx2 = 1
nx3 = 1

<time>
evolution = dynamic
integrator = rk3
cfl_number = 0.4
gr_dt = true
nlim = -1
tlim = {tlim}
ndiag = 1

<coord>
general_rel = true
minkowski = true
excise = false
m = 0.0
a = 0.0

<mhd>
fofc = true
nscalars = 1
scalar_flux_hll = true
eos = ideal
dyn_eos = ideal
dyn_error = reset_floor
reconstruct = ppmx
rsolver = hlle
gamma = 1.6666666666666667
dfloor = 1.0e-13
tfloor = 1.0e-14
dthreshold = 1.0
dyn_scratch = 1
{mhd_extra}

<adm>

<problem>
pgen_name = dyngr_frame_shift
type = riemann
rhol = 10.0
pl = 13.33
rhor = 1.0
pr = 0.1
vl = {v}
vr = {v}
xshock = 0.0
frame_vel1 = {xd}

<output1>
file_type = hst
data_format = %24.17e
dt = 0.01

<output2>
file_type = tab
variable = mhd_w_bcc
data_format = %24.17e
dt = {tlim}
"""
open(f'{name}.athinput', 'w').write(deck)
