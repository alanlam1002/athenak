"""
Unit tests for Task B7: PPM4/PPMX's generalized (Mignone 2014 eq. B.4/B.9/B.14) 4-point
interpolation weights exactly reconstruct a field that is an exact cubic polynomial in x1,
given the correct metric-weighted cell average as input, evaluated at the true face
position, to roundoff -- for Cartesian (regression: must stay exact, as the flat CW/CS
formula this replaced always was) and for cylindrical/spherical (the actual curvilinear
case this task adds support for).
"""

import test_suite.testutils as testutils


def test_recon_exact_cubic_cartesian():
    try:
        assert testutils.run("inputs/ut_recon_exact_cubic_cartesian.athinput")
    finally:
        testutils.cleanup()


def test_recon_exact_cubic_cylindrical():
    try:
        assert testutils.run("inputs/ut_recon_exact_cubic_cylindrical.athinput")
    finally:
        testutils.cleanup()


def test_recon_exact_cubic_spherical():
    try:
        assert testutils.run("inputs/ut_recon_exact_cubic_spherical.athinput")
    finally:
        testutils.cleanup()
