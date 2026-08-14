"""
Unit tests for Task B6: PLM's generalized (Mignone 2014) non-uniform limiter exactly
reconstructs a field that is an exact linear function of the stored volumetric centroid,
evaluated at the true face position, to roundoff -- for Cartesian (regression: must stay
exact, as the simplified uniform van Leer formula this replaced always was) and for
cylindrical/spherical (the actual non-uniform-spacing case this task adds support for).
"""

import test_suite.testutils as testutils


def test_recon_exact_gradient_cartesian():
    try:
        assert testutils.run("inputs/ut_recon_exact_gradient_cartesian.athinput")
    finally:
        testutils.cleanup()


def test_recon_exact_gradient_cylindrical():
    try:
        assert testutils.run("inputs/ut_recon_exact_gradient_cylindrical.athinput")
    finally:
        testutils.cleanup()


def test_recon_exact_gradient_spherical():
    try:
        assert testutils.run("inputs/ut_recon_exact_gradient_spherical.athinput")
    finally:
        testutils.cleanup()
