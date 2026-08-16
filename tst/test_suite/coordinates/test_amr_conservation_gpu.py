"""
GPU variants of the curvilinear AMR conservation gate.

The refinement machinery is the largest body of never-device-compiled code this project
has added: restriction, prolongation and flux correction all now read GeomData from
inside kernels, and the regrid path rebuilds geometry. Running the conservation gate on a
device is the cheapest way to find out whether any of that misbehaves there, because the
answer is a single scalar that must be right to 1e-11.

Never executed -- there is no GPU on this cluster. See DEVELOPMENT.md.
"""

import test_suite.testutils as testutils

_RES = ["mesh/nx1=256", "meshblock/nx1=32"]


def _run(name):
    try:
        assert testutils.run(f"inputs/ut_amr_conservation_{name}.athinput", flags=_RES)
    finally:
        testutils.cleanup()


def test_amr_conservation_cartesian_gpu():
    _run("cartesian")


def test_amr_conservation_cylindrical_gpu():
    _run("cylindrical")


def test_amr_conservation_spherical_gpu():
    _run("spherical")
