"""
Unit tests for Task E2: at the coordinate origin (x1min=0), Area1(is)=0 exactly, the
(pre-existing, unmodified) reflect BC mirrors ghost-zone data with a sign flip on
exactly the radial velocity component and identity on the tangential ones (using the
same mirror-index convention as Task B6/B7's ghost-geometry mirroring), and no NaN/Inf
appears anywhere after evolving. Total-mass-conservation-to-roundoff at the origin is
already exercised by Task B4's cylindrical_axisym/spherical mass-conservation tests.
"""

import test_suite.testutils as testutils


def test_origin_conservation_axisym():
    try:
        assert testutils.run("inputs/ut_origin_conservation_axisym.athinput")
    finally:
        testutils.cleanup()


def test_origin_conservation_spherical():
    try:
        assert testutils.run("inputs/ut_origin_conservation_spherical.athinput")
    finally:
        testutils.cleanup()
