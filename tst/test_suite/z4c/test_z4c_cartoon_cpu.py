"""
Z4c Cartoon fill unit test (cpu build): exact checks, convergence orders and a
deliberately wrong parity entry, on a 2D coord=cylindrical_axisym mesh.
See cartoon_common.py and src/pgen/unit_tests/z4c_cartoon_test.cpp.
"""

import pytest
import test_suite.testutils as testutils
from test_suite.z4c import cartoon_common as cc


@pytest.mark.parametrize("ng", [2, 3, 4])
def test_cartoon_production_stencil_cpu(ng):
    try:
        cc.check_production(ng)
    finally:
        testutils.cleanup()


@pytest.mark.parametrize("ng", [2, 3, 4])
def test_cartoon_narrow_stencil_cpu(ng):
    try:
        cc.check_narrow_stencil(ng)
    finally:
        testutils.cleanup()


@pytest.mark.parametrize("ng", [2, 3, 4])
def test_cartoon_wrong_parity_fails_cpu(ng):
    try:
        cc.check_wrong_parity_fails(ng)
    finally:
        testutils.cleanup()
