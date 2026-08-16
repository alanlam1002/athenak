"""
Unit test for the COARSE GeomData that curvilinear SMR/AMR prolongation reads, and for
the two telescoping identities the rest of Phase 1 is built on:

    V_coarse == sum(V_fine)        A_coarse == sum(A_fine)

Those identities are why restriction and flux correction need no coarse geometry at all:
a coarse cell is exactly the union of its children, so the normalisation can be formed
from the fine arrays a block already owns. The pgen also asserts that the coarse
volumetric centroid is the volume-weighted mean of its children -- the property that
makes the centroid-based prolongation exactly conservative -- plus an independent
analytic check, so a coarse build that is internally consistent but systematically wrong
(e.g. one that used the FINE cell width) cannot pass.
"""

import test_suite.testutils as testutils


def _run(name):
    try:
        assert testutils.run(f"inputs/ut_coarse_geometry_{name}.athinput")
    finally:
        testutils.cleanup()


def test_coarse_geometry_cartesian():
    _run("cartesian")


def test_coarse_geometry_cylindrical():
    _run("cylindrical")


def test_coarse_geometry_spherical():
    _run("spherical")
