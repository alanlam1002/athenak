"""
Unit tests for Task B4: MeshGeometry/GeomData construction for the three curvilinear
coordinate systems (cylindrical, cylindrical_axisym, spherical_polar). Each geometry
factory's output is compared, inside the pgen (geometry_curvilinear_test.cpp), against
independently re-derived analytic formulas -- not copy-pasted from the production
factory code -- on a small hand-computable grid. The spherical case also includes a
full-sphere Area1=4*pi*r^2 sanity check as an independent end-to-end cross-check.
"""

import test_suite.testutils as testutils


def test_geom_cylindrical_construction():
    try:
        assert testutils.run("inputs/ut_geometry_cylindrical.athinput")
    finally:
        testutils.cleanup()


def test_geom_cylindrical_axisym_construction():
    try:
        assert testutils.run("inputs/ut_geometry_cylindrical_axisym.athinput")
    finally:
        testutils.cleanup()


def test_geom_spherical_construction():
    try:
        assert testutils.run("inputs/ut_geometry_spherical.athinput")
    finally:
        testutils.cleanup()


def test_geom_spherical_2d_construction():
    # Exercises the non-degenerate trig-based theta-centroid formula and the angular
    # CenterWidth2/3 CFL-width fix (Task B5), both trivial/degenerate in the required
    # 1D-radial layout tested above.
    try:
        assert testutils.run("inputs/ut_geometry_spherical_2d.athinput")
    finally:
        testutils.cleanup()
