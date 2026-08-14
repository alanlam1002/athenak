"""
Unit test for Task A4: mass conservation under the new generic area/volume-weighted
flux-divergence kernel (src/mhd/mhd_update.cpp). MHD analogue of
test_hydro_mass_conservation_cpu.py -- see that file and
tst/inputs/ut_hydro_mass_conservation.athinput for the full derivation.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import test_suite.testutils as testutils  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "vis", "python"))
import athena_read  # noqa: E402

_REL_TOL = 1.0e-11


def test_mhd_mass_conservation():
    input_file = "inputs/ut_mhd_mass_conservation.athinput"
    hst_file = "ut_mhd_mass_conservation.mhd.hst"
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
