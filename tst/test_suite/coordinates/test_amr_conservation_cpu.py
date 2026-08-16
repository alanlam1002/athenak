"""
Acceptance gate for curvilinear SMR/AMR Phase 1 (cell-centered).

Total mass sum(rho*Vol) must be conserved to roundoff across many refine AND derefine
cycles on a curvilinear grid. The check itself lives in the pgen
(src/pgen/unit_tests/amr_conservation_test.cpp) rather than here, because the history
file only carries ~6 significant digits -- far too coarse to be a gate. The pgen computes
the mass in double precision and fatals on drift, so these wrappers only have to assert
that the run succeeds.

VERIFIED TO HAVE TEETH. Both halves of the change were reverted in turn and this gate
caught each one, while the cartesian control kept passing (its children have equal
volumes, so the weighted forms reduce exactly to the old constants):

    RestrictCC -> unweighted   : cylindrical/axisym 5.3e-06, spherical 1.0e-05 drift
    ProlongCC  -> uniform offs : cylindrical 1.0e-05, spherical 2.7e-05 drift

against a 1e-11 tolerance, i.e. detected by ~6 orders of magnitude. An earlier
.hst-based version of this test passed in BOTH broken states and was discarded; see the
pgen's file comment for why.
"""

import test_suite.testutils as testutils


def _run(name):
    try:
        assert testutils.run(f"inputs/ut_amr_conservation_{name}.athinput")
    finally:
        testutils.cleanup()


def test_amr_conservation_cartesian():
    # Control: must pass before and after Phase 1, and in both broken states above.
    _run("cartesian")


def test_amr_conservation_cylindrical():
    _run("cylindrical")


def test_amr_conservation_axisym():
    _run("axisym")


def test_amr_conservation_spherical():
    # Largest volume contrast between sibling cells (V ~ r^3), so the most sensitive.
    _run("spherical")
