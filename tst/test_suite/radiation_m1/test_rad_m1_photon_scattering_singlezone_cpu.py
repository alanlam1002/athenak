"""
Grey photon M1 single-zone scattering-only null test.

kappa_p=kappa_a=0, compton=false, so there is no channel exchanging energy
between matter and radiation -- only elastic scattering. E(t) should stay
pinned at its initial (floor) value for the whole run. Guards against kscat
leaking into the emission/absorption term. See
inputs/tests/check_rad_m1_photon_scattering_singlezone.py for the full
rationale.
"""

# Modules
import test_suite.testutils as testutils
import check_rad_m1_photon_scattering_singlezone as check

input_file = "inputs/rad_m1_photon_scattering_singlezone.athinput"


def test_run():
    """Run a single test."""
    try:
        results = testutils.run(input_file)
        assert results, "Photon M1 scattering-only single-zone test run failed."
        assert check.main([]), "Photon M1 scattering-only single-zone check failed."
    finally:
        testutils.cleanup()
