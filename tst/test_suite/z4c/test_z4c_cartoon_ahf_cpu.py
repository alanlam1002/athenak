"""M2f: Cartoon-aware FastFlow on spinning-puncture data, CPU. See cartoon_ahf_common."""

import pytest
import test_suite.testutils as testutils
from test_suite.z4c import cartoon_ahf_common as c


@pytest.mark.parametrize("ng", [2, 4])
def test_cartoon_m2f_ahf_les_cpu(ng):
    try:
        c.check_les(ng)
    finally:
        c.cleanup()
        testutils.cleanup()


def test_cartoon_m2f_ahf_qi_cpu():
    try:
        c.check_qi()
    finally:
        c.cleanup()
        testutils.cleanup()


def test_cartoon_m2f_ahf_2d_vs_3d_cpu():
    try:
        c.check_2d_vs_3d()
    finally:
        c.cleanup()
        testutils.cleanup()
