"""
Unit test for Task F1: spherical radial Sod shock tube on a closed domain (reflect at
r=0 and the outer boundary). Total mass and total energy must be conserved to roundoff
(the reflect BCs make this an exact discrete property, same as Task B4's conservation
tests, just now exercising a genuine discontinuous/nonlinear Riemann problem rather than
a smooth profile). Formal shock-front convergence-order verification is covered by the
dedicated Task F3 (smooth-pulse convergence test) rather than duplicated here.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import test_suite.testutils as testutils  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "vis", "python"))
import athena_read  # noqa: E402

_REL_TOL = 1.0e-11


def test_f1_spherical_sod_conservation():
    input_file = "inputs/ut_f1_spherical_sod.athinput"
    hst_file = "ut_f1_spherical_sod.hydro.hst"
    try:
        assert testutils.run(input_file)
        data = athena_read.hst(hst_file)
        for label in ("mass", "tot-E"):
            vals = data[label]
            assert len(vals) >= 3, f"expected multiple hst samples, got {len(vals)}"
            v0 = vals[0]
            max_rel_dev = max(abs(v - v0) / abs(v0) for v in vals)
            assert max_rel_dev < _REL_TOL, (
                f"{label} not conserved to roundoff on closed spherical domain: "
                f"max relative deviation {max_rel_dev:g} (tolerance {_REL_TOL:g})"
            )
    finally:
        testutils.cleanup()
