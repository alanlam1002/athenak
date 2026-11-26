"""M2b: Cartoon-aware Sommerfeld boundary, gauge pulse, CPU. See cartoon_pulse_common."""

import pytest
import test_suite.testutils as testutils
from test_suite.z4c import cartoon_pulse_common as c


@pytest.mark.parametrize("ng", [2, 4])
def test_cartoon_m2b_pulse_linear_cpu(ng):
    try:
        c.check_linear(ng)
    finally:
        c.cleanup()
        testutils.cleanup()


def test_cartoon_m2b_pulse_nonlinear_cpu():
    try:
        c.check_nonlinear()
    finally:
        c.cleanup()
        testutils.cleanup()


def test_cartoon_m2b_pulse_2d_vs_3d_cpu():
    try:
        c.check_2d_vs_3d()
    finally:
        c.cleanup()
        testutils.cleanup()
