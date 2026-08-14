"""
Unit test for Task A3: mass conservation under the new generic area/volume-weighted
flux-divergence kernel (src/hydro/hydro_update.cpp).

Runs a fully periodic, single-level (no AMR) smooth linear-wave perturbation and checks
that total mass (the "mass" column of the standard <output> file_type=hst history file)
stays constant to roundoff over the run. On a periodic/closed domain, Sum_cells Vol*divF
telescopes exactly to the (zero) boundary-flux residual regardless of the physical flux
values themselves -- see tst/inputs/ut_hydro_mass_conservation.athinput for the full
derivation. Any deviation beyond roundoff here indicates a real bug in the area/volume
weighting introduced in Task A3, not accumulated floating-point truncation error.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import test_suite.testutils as testutils  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "vis", "python"))
import athena_read  # noqa: E402

# Roundoff-level tolerance: mass conservation is an exact discrete property of the
# finite-volume method on a periodic domain, not a truncation-error-limited quantity, so
# any deviation above ~1e-11 relative indicates a real conservation bug, not float noise.
_REL_TOL = 1.0e-11


def test_hydro_mass_conservation():
    input_file = "inputs/ut_hydro_mass_conservation.athinput"
    hst_file = "ut_hydro_mass_conservation.hydro.hst"
    try:
        assert testutils.run(input_file)
        data = athena_read.hst(hst_file)
        mass = data["mass"]
        assert len(mass) >= 3, f"expected multiple hst samples, got {len(mass)}"
        m0 = mass[0]
        max_rel_dev = max(abs(m - m0) / abs(m0) for m in mass)
        assert max_rel_dev < _REL_TOL, (
            f"total mass not conserved to roundoff: max relative deviation "
            f"{max_rel_dev:g} (tolerance {_REL_TOL:g}), mass[0]={m0:g}, "
            f"mass values={list(mass)}"
        )
    finally:
        testutils.cleanup()
