"""
GPU variant of the curvilinear SMR/AMR Phase 2 acceptance gate (face-centered / MHD): div(B) must stay
at roundoff, and mass must be conserved, across many refine AND derefine cycles.

The check lives in the pgen (src/pgen/unit_tests/amr_divb_test.cpp), not here, for the
same reason as the Phase 1 gate: the .hst file carries only ~6 significant digits, far
too coarse for a roundoff-level assertion.

VERIFIED TO HAVE TEETH. Forcing GeomData::cells_uniform and cubic_cells true -- which
reverts all four face-centered weightings (RestrictFC, ProlongFCShared*,
ProlongFCInternal, and the Len-weighted EMF correction) to upstream's unweighted
constants -- makes rel_divb jump by 13-14 orders of magnitude:

    cylindrical  2.04e-16 -> 4.22e-02
    axisym       4.04e-16 -> 1.41e-03
    spherical    2.96e-16 -> 1.35e-02
    cartesian    2.13e-16 -> 2.13e-16   (unchanged: it takes the fast path either way,
                                         which is exactly what makes it the control)

Additionally, the Toth-Roe piece has an independent single-piece check: upstream's
internal-face kernel is only divergence-preserving for CUBIC cells, so running the
Cartesian AMR problem at aspect ratio 2 gave 6.9e-03 before the flux-form recast and
1.8e-16 after. See DEVELOPMENT.md, "Phase 2 finding".

NOTE ON BOUNDARY CONDITIONS. The spherical case uses REFLECTING theta walls, not
periodic. theta is not a periodic coordinate, and a periodic BC there loses mass by
construction (measured 2.9e-04, and 4.3e-04 with refinement disabled -- i.e. nothing to
do with AMR). See the comment in the input file.

NOT YET EXECUTED: this cluster has no GPU (see DEVELOPMENT.md, "Performance and GPU
status"), so like every other _gpu module here this has never run. It exists so the
device-compiled surface does not grow without a corresponding test. Runs at higher
resolution, where a device-side indexing error in the face-centered kernels is more
likely to show up than at the CPU sizes.
"""

import test_suite.testutils as testutils

_RES = ["mesh/nx1=128", "mesh/nx2=128", "meshblock/nx1=32", "meshblock/nx2=32"]


def _run(name):
    try:
        assert testutils.run(f"inputs/ut_amr_divb_{name}.athinput", _RES)
    finally:
        testutils.cleanup()


def test_amr_divb_gpu_cartesian():
    """Control: cubic cells take the fast path, so this must be bitwise unchanged."""
    _run("cartesian")


def test_amr_divb_gpu_cylindrical():
    """(R,phi): Area1 ~ R varies across every level boundary."""
    _run("cylindrical")


def test_amr_divb_gpu_axisym():
    """(R,z): Area3 ~ R dR weights the z-faces, Area1 ~ R the radial ones."""
    _run("axisym")


def test_amr_divb_gpu_spherical():
    """(r,theta): the harshest case, Area1 ~ r^2 and Area2 ~ r sin(theta)."""
    _run("spherical")
