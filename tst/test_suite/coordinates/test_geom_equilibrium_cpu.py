"""
Unit tests for Task C1/C2: the Newtonian curvilinear geometric source terms
(src/coordinates/geometric_srcterms.{hpp,cpp}), using the corrected Delta-A/Delta-V
coefficients (v2 plan Correction C2), preserve a rotating centrifugal-pressure-balance
equilibrium (uniform density, constant rotation speed, log-pressure profile solving
dP/dR = rho*v0^2/R) to truncation error over many orbits -- for cylindrical_axisym (R,z,
Task C1) and spherical_polar (1D radial, Task C2).

This is the well-balancedness gate the v2 plan's Correction C2 exists for: with the
WRONG (1/x1v) source coefficient this test would fail immediately (not just accumulate
truncation error), and it did fail at the O(1) level -- not O(1e-4) -- during
development when the pressure/internal-energy primitive index was misread (see
DEVELOPMENT.md Task C1/C2 log); this test's tolerance is set well above the observed
~4e-4 RMS-L1 error at this resolution specifically to still catch that class of bug.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import test_suite.testutils as testutils  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "vis", "python"))
import athena_read  # noqa: E402

_RMS_L1_TOL = 1.0e-2


def _check_equilibrium(input_file, errs_file):
    assert testutils.run(input_file)
    data = athena_read.error_dat(errs_file)
    rms_l1 = data[0][4]
    assert rms_l1 < _RMS_L1_TOL, (
        f"rotating equilibrium did not stay static: RMS-L1={rms_l1:g} "
        f"(tolerance {_RMS_L1_TOL:g})"
    )


def test_geom_equilibrium_axisym():
    try:
        _check_equilibrium(
            "inputs/ut_geom_equilibrium_axisym.athinput",
            "ut_geom_equilibrium_axisym-errs.dat",
        )
    finally:
        testutils.cleanup()


def test_geom_equilibrium_spherical():
    try:
        _check_equilibrium(
            "inputs/ut_geom_equilibrium_spherical.athinput",
            "ut_geom_equilibrium_spherical-errs.dat",
        )
    finally:
        testutils.cleanup()
