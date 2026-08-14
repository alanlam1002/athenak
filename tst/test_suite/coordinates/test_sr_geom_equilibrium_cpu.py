"""
Unit test for Task G1: SR geometric source terms (the rho*h-and-u^i generalization of
the Newtonian rho-and-v^i centrifugal/pressure terms) preserve the same rotating
equilibrium as the Newtonian case at small v0 -- the v/c->0 cross-check against the
already-validated Newtonian result (test_geom_equilibrium_cpu.py), read via the same
RMS-L1 column of the generic OutputErrors()-produced -errs.dat file.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import test_suite.testutils as testutils  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "vis", "python"))
import athena_read  # noqa: E402

_RMS_L1_TOL = 1.0e-2


def test_sr_geom_equilibrium_axisym():
    try:
        assert testutils.run("inputs/ut_sr_geom_equilibrium_axisym.athinput")
        data = athena_read.error_dat("ut_sr_geom_equilibrium_axisym-errs.dat")
        rms_l1 = data[0][4]
        assert rms_l1 < _RMS_L1_TOL, (
            f"SR rotating equilibrium did not stay static: RMS-L1={rms_l1:g} "
            f"(tolerance {_RMS_L1_TOL:g})"
        )
    finally:
        testutils.cleanup()
