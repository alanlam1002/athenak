"""M2d: vacuum tests T1, T2, T3 (FastFlow cadence), T5 at pytest size. See
cartoon_m2d_common."""

import pytest
import test_suite.testutils as testutils
from test_suite.z4c import cartoon_m2d_common as c


@pytest.mark.parametrize("ng", [2, 4])
def test_cartoon_m2d_t1_pulse_orders_gpu(ng):
    try:
        c.check_t1(ng)
    finally:
        c.cleanup()
        testutils.cleanup()


@pytest.mark.parametrize("ng", [2, 4])
def test_cartoon_m2d_t2_trumpet_gpu(ng):
    try:
        c.check_t2(ng)
    finally:
        c.cleanup()
        testutils.cleanup()


@pytest.mark.parametrize("ng", [2, 4])
def test_cartoon_m2d_t5_2d_vs_3d_gpu(ng):
    try:
        c.check_t5(ng)
    finally:
        c.cleanup()
        testutils.cleanup()


def test_cartoon_m2d_t3_find_dt_gpu():
    try:
        c.check_t3_find_dt()
    finally:
        c.cleanup()
        testutils.cleanup()
