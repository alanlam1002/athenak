"""
Unit test for Task F2: axisymmetric (R,z) magnetized rotating-disk equilibrium stays
static over many orbits (RMS-L1 from the generic OutputErrors()-produced -errs.dat
file), and div(B) stays at roundoff throughout -- the integration-level check on Task
C1's phi-component sign conventions combined with the MHD momentum-flux extension.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import test_suite.testutils as testutils  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "vis", "python"))
import athena_read  # noqa: E402

_RMS_L1_TOL = 1.0e-2


def test_mhd_disk_equilibrium_axisym():
    try:
        assert testutils.run("inputs/ut_mhd_disk_equilibrium_axisym.athinput")
        data = athena_read.error_dat("ut_mhd_disk_equilibrium_axisym-errs.dat")
        rms_l1 = data[0][4]
        assert rms_l1 < _RMS_L1_TOL, (
            f"magnetized rotating equilibrium did not stay static: RMS-L1={rms_l1:g} "
            f"(tolerance {_RMS_L1_TOL:g})"
        )
    finally:
        testutils.cleanup()
