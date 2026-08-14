"""
Unit tests for Task D3: the REAL correctness gate for the area/edge-length-weighted CT
curl (Task D1) -- physical induction tests, as opposed to Task D2's topological
div(B)=0 check (which cannot detect a wrong edge length, handedness, or EMF sign; see
v2 plan Correction C6).

- test_ct_field_loop_axisym: a weak magnetic loop advected by a uniform z-velocity in
  cylindrical_axisym (R,z) returns to its initial shape/amplitude (to truncation error,
  read from the RMS-L1 column of the generic OutputErrors()-produced -errs.dat file)
  after exactly one z-period.
- test_ct_monopole_stationarity_spherical: in the required 1D-radial spherical layout,
  B_r*r^2 stays exactly constant under a radial wind (CT's B1 update is unconditionally
  skipped when multi_d=false, so this checks that nothing ELSE in the code perturbs it).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import test_suite.testutils as testutils  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "vis", "python"))
import athena_read  # noqa: E402

_FIELD_LOOP_RMS_L1_TOL = 5.0e-3


def test_ct_field_loop_axisym():
    try:
        assert testutils.run("inputs/ut_ct_field_loop_axisym.athinput")
        data = athena_read.error_dat("ut_ct_field_loop_axisym-errs.dat")
        rms_l1 = data[0][4]
        assert rms_l1 < _FIELD_LOOP_RMS_L1_TOL, (
            f"field loop did not return to its initial shape after one period: "
            f"RMS-L1={rms_l1:g} (tolerance {_FIELD_LOOP_RMS_L1_TOL:g})"
        )
    finally:
        testutils.cleanup()


def test_ct_monopole_stationarity_spherical():
    try:
        assert testutils.run("inputs/ut_ct_monopole_stationarity_spherical.athinput")
    finally:
        testutils.cleanup()
