#!/usr/bin/env python
"""Standalone boosted-LTE-relaxation check for the grey photon M1 single zone
test, with a nonzero (mildly relativistic) fluid velocity v_x.

Not wired into the tst/ pytest harness yet -- run by hand after executing
AthenaK with inputs/tests/rad_m1_photon_vx_singlezone.athinput, e.g.:

    ./src/athena -i ../inputs/tests/rad_m1_photon_vx_singlezone.athinput
    python ../inputs/tests/check_rad_m1_photon_vx_singlezone.py

Run from the build directory (so the default "tab/" output directory is
found next to it), or pass --tab-dir explicitly.

Same physical setup as check_rad_m1_photon_singlezone.py (static-fluid LTE
relaxation), except the fluid now moves at v_x. Still homogeneous/periodic,
so there is no spatial transport involved -- this isolates the lab-frame
<-> comoving-frame boost inside the source solver (u^a, W, proj_ud in
radiation_m1_sources.hpp/radiation_m1_helpers.hpp), which the static tests
never exercise.

Only the FINAL-time equilibrium is checked here (not the full trajectory the
way the static tests are), since that requires only H^a=0 (comoving flux
vanishes) at equilibrium, independent of the transient path. With
closure_fun=eddington pinning chi=1/3 exactly (which zeroes the anisotropic
F_a*F_b/|F|^2 term in the closure -- see radiation_m1_helpers.hpp's
apply_closure/Pthin_dd/Pthick_dd, and radiation_m1_closure.hpp's Eddington
case), the radiation stress-energy tensor at equilibrium is the pure
Eddington form T_ab = (4/3)*J*u_a*u_b + (J/3)*g_ab. Contracting with n^a
(Eulerian normal) and projecting for a flat, alpha=1, beta^i=0 metric with
u^a = W*(1, v_x, 0, 0) gives (independently re-derived from
assemble_rT/calc_J_from_rT/calc_H_from_rT in radiation_m1_helpers.hpp):

    E_eq   = J_eq * (4*W^2 - 1) / 3
    F_x_eq = J_eq * (4/3) * W^2 * v_x,   J_eq = a_rad * T_gas^4

(sanity check: v_x=0 => W=1 => E_eq=J_eq, F_x_eq=0, matching the static test).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from m1_tab_utils import load_series  # noqa: E402

# Must match inputs/tests/rad_m1_photon_vx_singlezone.athinput
RHO = 1.0
TEMP = 1.0
VX = 0.3
ARAD = 1.0
BASENAME = "photon_vx_singlezone"

W = 1.0 / np.sqrt(1.0 - VX**2)
J_EQ = ARAD * TEMP**4
E_EQ = J_EQ * (4.0 * W**2 - 1.0) / 3.0
FX_EQ = J_EQ * (4.0 / 3.0) * W**2 * VX


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tab-dir", default="tab", help="directory with .tab output")
    parser.add_argument("--tol", type=float, default=1e-3,
                        help="relative tolerance on late-time E, Fx vs equilibrium")
    args = parser.parse_args()

    t, E = load_series(args.tab_dir, BASENAME, "rad_m1_E", "E:0")
    _, Fx = load_series(args.tab_dir, BASENAME, "rad_m1_F", "Fx:0")

    print("W = {:.6f}".format(W))
    print("J_eq = a_rad * T^4 = {:.6e}".format(J_EQ))
    print("E_eq = J_eq*(4W^2-1)/3 = {:.6e}".format(E_EQ))
    print("Fx_eq = J_eq*(4/3)*W^2*vx = {:.6e}".format(FX_EQ))

    print("\n   t          E(t)          Fx(t)")
    for ti, Ei, Fxi in zip(t, E, Fx):
        print("{:10.4f}  {:12.6e}  {:12.6e}".format(ti, Ei, Fxi))

    E_final_rel_err = abs(E[-1] - E_EQ) / E_EQ
    Fx_final_rel_err = abs(Fx[-1] - FX_EQ) / abs(FX_EQ)
    print("\nfinal E = {:.6e}, target E_eq = {:.6e}, relative error = {:.3e}".format(
        E[-1], E_EQ, E_final_rel_err))
    print("final Fx = {:.6e}, target Fx_eq = {:.6e}, relative error = {:.3e}".format(
        Fx[-1], FX_EQ, Fx_final_rel_err))

    if E_final_rel_err < args.tol and Fx_final_rel_err < args.tol:
        print("PASS: late-time (E, Fx) converged to the boosted-LTE equilibrium "
              "within tolerance {:.1e}".format(args.tol))
    else:
        print("FAIL: late-time (E, Fx) did NOT converge to the boosted-LTE "
              "equilibrium within tolerance {:.1e}".format(args.tol))
        sys.exit(1)


if __name__ == "__main__":
    main()
