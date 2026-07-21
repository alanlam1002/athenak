"""
Grey photon M1 single-zone boosted (finite v_x) LTE-relaxation test.

Same physical setup as test_rad_m1_photon_singlezone_cpu.py, except the
fluid moves at a mildly relativistic v_x -- isolates the lab-frame <->
comoving-frame boost inside the implicit source solver. Checks the late-time
(E, F_x) against the boosted-LTE equilibrium. See
inputs/tests/check_rad_m1_photon_vx_singlezone.py for the full derivation.
"""

# Modules
import test_suite.testutils as testutils
import check_rad_m1_photon_vx_singlezone as check

input_file = "inputs/rad_m1_photon_vx_singlezone.athinput"


def test_run():
    """Run a single test."""
    try:
        results = testutils.run(input_file)
        assert results, "Photon M1 boosted (vx) single-zone test run failed."
        assert check.main([]), "Photon M1 boosted (vx) single-zone check failed."
    finally:
        testutils.cleanup()
