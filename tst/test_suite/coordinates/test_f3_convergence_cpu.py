"""
Unit test for Task F3: formal convergence-order verification. A smooth entropy-mode
density pulse (see smooth_pulse_convergence_test.cpp's docstring for why it is an EXACT,
shape-preserving passive-advection solution) is advected by uniform z-velocity in
cylindrical_axisym (R,z), on a domain periodic in z, at three resolutions. After exactly
one z-period the exact solution equals the t=0 IC, so the RMS-L1 column of the generic
OutputErrors()-produced -errs.dat file is the true solution error at each resolution;
the observed convergence order (log2 of the ratio of successive errors) should be close
to PLM's design order (2), not merely "small at any one resolution" (which Task B6's
recon_exact_gradient_test and this test's own single-resolution check in
test_geom_equilibrium_cpu.py-style tests already establish separately).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import test_suite.testutils as testutils  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "vis", "python"))
import athena_read  # noqa: E402

_INPUT = "inputs/ut_smooth_pulse_convergence_axisym.athinput"
_ERRS_FILE = "ut_smooth_pulse_convergence_axisym-errs.dat"
_RESOLUTIONS = (24, 48, 96)
# generous band around the PLM design order (2): coarse resolutions + a single-period
# entropy-mode test are not a pristine asymptotic-convergence setup, so this checks
# "converging at roughly the right rate," not a tight formal order measurement.
_MIN_ORDER = 1.3
_MAX_ORDER = 2.8


def test_f3_smooth_pulse_convergence_order():
    try:
        if os.path.exists(_ERRS_FILE):
            os.remove(_ERRS_FILE)
        for n in _RESOLUTIONS:
            flags = [
                f"mesh/nx1={n}", f"mesh/nx2={n}",
                f"meshblock/nx1={n}", f"meshblock/nx2={n}",
            ]
            assert testutils.run(_INPUT, flags)
        data = athena_read.error_dat(_ERRS_FILE)
        rms_l1 = [row[4] for row in data]
        assert len(rms_l1) == len(_RESOLUTIONS)
        orders = []
        for k in range(len(_RESOLUTIONS)-1):
            ratio = rms_l1[k]/rms_l1[k+1]
            order = math.log(ratio, 2)
            orders.append(order)
        for n, order in zip(_RESOLUTIONS[1:], orders):
            assert _MIN_ORDER < order < _MAX_ORDER, (
                f"observed convergence order {order:g} at resolution {n} outside "
                f"[{_MIN_ORDER},{_MAX_ORDER}] (errors={rms_l1})"
            )
    finally:
        testutils.cleanup()
