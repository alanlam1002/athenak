"""
Unit test for Task A2: MeshGeometry/GeomData Cartesian geometry factory.

Verifies that geometry_cartesian.cpp's factored Area1/2/3, Vol, Len1/2/3, and x1v/x2v/x3v
accessors reproduce, to machine precision, the same arithmetic the rest of the code
already computes directly from mb_size.dx1/dx2/dx3 and CellCenterX(). The actual
comparison happens inside the pgen (src/pgen/unit_tests/geometry_cartesian_test.cpp),
which exit(EXIT_FAILURE)s on any mismatch -- this wrapper just runs it and checks the
process return code, following the established convention (see test_gauss_legendre_cpu.py).
"""

import test_suite.testutils as testutils


def test_geometry_cartesian_construction():
    input_file = "inputs/ut_geometry_cartesian.athinput"
    try:
        assert testutils.run(input_file)
    finally:
        testutils.cleanup()
