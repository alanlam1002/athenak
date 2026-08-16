"""
GPU variants of the reconstruction exactness tests (Tasks B6/B7).

Covers the third group of GeomData arrays that the two other `_gpu` modules do not: the
reconstruction coefficient tables. That includes plm_c1/plm_c2/plm_c3, which are rank-3
`(m, component, cell)` arrays -- a different shape from every other array in GeomData, so
they are worth exercising on a device specifically. Both pgens assert exactness
(a linear profile must reconstruct exactly at faces for PLM; a cubic must for PPM4/PPMX),
which makes them sensitive to any coefficient that fails to reach the device correctly.

Note these two pgens do their checking on the HOST, from explicit
`create_mirror_view_and_copy` mirrors of the coefficient arrays. That is deliberate and
is exactly what makes them device-safe: calling GeomData's KOKKOS_INLINE_FUNCTION
accessors directly from host code compiles fine but dereferences a device pointer at
runtime. See geometry_cartesian_test.cpp for the pattern to copy.
"""

import test_suite.testutils as testutils

_RES = ["mesh/nx1=128", "meshblock/nx1=128"]


def test_recon_exact_gradient_cartesian_gpu():
    try:
        assert testutils.run("inputs/ut_recon_exact_gradient_cartesian.athinput",
                             flags=_RES)
    finally:
        testutils.cleanup()


def test_recon_exact_gradient_cylindrical_gpu():
    try:
        assert testutils.run("inputs/ut_recon_exact_gradient_cylindrical.athinput",
                             flags=_RES)
    finally:
        testutils.cleanup()


def test_recon_exact_gradient_spherical_gpu():
    try:
        assert testutils.run("inputs/ut_recon_exact_gradient_spherical.athinput",
                             flags=_RES)
    finally:
        testutils.cleanup()


def test_recon_exact_cubic_cartesian_gpu():
    try:
        assert testutils.run("inputs/ut_recon_exact_cubic_cartesian.athinput", flags=_RES)
    finally:
        testutils.cleanup()


def test_recon_exact_cubic_cylindrical_gpu():
    try:
        assert testutils.run("inputs/ut_recon_exact_cubic_cylindrical.athinput",
                             flags=_RES)
    finally:
        testutils.cleanup()


def test_recon_exact_cubic_spherical_gpu():
    try:
        assert testutils.run("inputs/ut_recon_exact_cubic_spherical.athinput", flags=_RES)
    finally:
        testutils.cleanup()
