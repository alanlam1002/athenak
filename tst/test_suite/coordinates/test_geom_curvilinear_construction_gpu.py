"""
GPU variants of the curvilinear MeshGeometry/GeomData construction tests.

These exist because EVERY other test in tst/test_suite/coordinates/ is named `_cpu`,
which means `run_test_suite.py --gpu` (which selects with `-k _gpu`) would build with
CUDA and then exercise ZERO curvilinear cells -- a device build could pass the whole
suite while the curvilinear code path had never run on a device at all.

This module is the most valuable one to run first on a GPU: geometry_curvilinear_test.cpp
is the pgen that reads GeomData most directly, so a device-side failure here localizes
the problem to the geometry arrays rather than to the physics kernels that consume them.

Resolution is raised over the `_cpu` siblings, per the convention used by the upstream
`_gpu` tests (e.g. tst/test_suite/nr/test_nr_lwave3d_amr_gpu.py), so the launch actually
fills a GPU rather than measuring kernel-launch overhead.
"""

import test_suite.testutils as testutils

# nx1 must stay a multiple of the meshblock size; these inputs use a single MeshBlock,
# so both are overridden together.
_RES = ["mesh/nx1=128", "meshblock/nx1=128"]


def test_geom_cylindrical_construction_gpu():
    try:
        assert testutils.run("inputs/ut_geometry_cylindrical.athinput", flags=_RES)
    finally:
        testutils.cleanup()


def test_geom_cylindrical_axisym_construction_gpu():
    try:
        assert testutils.run("inputs/ut_geometry_cylindrical_axisym.athinput", flags=_RES)
    finally:
        testutils.cleanup()


def test_geom_spherical_construction_gpu():
    try:
        assert testutils.run("inputs/ut_geometry_spherical.athinput", flags=_RES)
    finally:
        testutils.cleanup()


def test_geom_spherical_2d_construction_gpu():
    # 2D (r,theta): exercises the trig-based theta-centroid formula and the angular
    # CenterWidth2/3 CFL widths, both degenerate in the 1D-radial layout above.
    try:
        assert testutils.run(
            "inputs/ut_geometry_spherical_2d.athinput",
            flags=["mesh/nx1=128", "meshblock/nx1=128",
                   "mesh/nx2=128", "meshblock/nx2=128"],
        )
    finally:
        testutils.cleanup()
