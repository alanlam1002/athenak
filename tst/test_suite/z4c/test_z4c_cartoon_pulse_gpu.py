"""M2b: Cartoon-aware Sommerfeld boundary, gauge pulse, PVC. See cartoon_pulse_common."""

import pytest
import test_suite.testutils as testutils
from test_suite.z4c import cartoon_pulse_common as c


@pytest.mark.parametrize("ng", [2, 4])
def test_cartoon_m2b_pulse_linear_gpu(ng):
    try:
        c.check_linear(ng)
    finally:
        c.cleanup()
        testutils.cleanup()


def test_cartoon_m2b_pulse_nonlinear_gpu():
    try:
        c.check_nonlinear()
    finally:
        c.cleanup()
        testutils.cleanup()


def test_cartoon_m2b_pulse_2d_vs_3d_gpu():
    try:
        c.check_2d_vs_3d()
    finally:
        c.cleanup()
        testutils.cleanup()


def test_cartoon_history_volume_gpu():
    try:
        c.check_history_volume()
    finally:
        c.cleanup()
        testutils.cleanup()


@pytest.mark.parametrize("dim", [2, 3])
def test_constraint_gate_gpu(dim):
    try:
        c.check_constraint_gate(dim)
    finally:
        c.cleanup()
        testutils.cleanup()
