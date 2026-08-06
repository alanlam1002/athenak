"""
Grey photon M1 single-zone LTE-relaxation test (backreact=false).

Static, homogeneous, periodic single zone with a fixed-temperature IdealGas
fluid; checks E(t) relaxes toward a_rad*T^4 following the analytic
absorption-only relaxation law. See inputs/tests/check_rad_m1_photon_singlezone.py
for the full derivation.
"""

# Modules
import test_suite.testutils as testutils
import check_rad_m1_photon_singlezone as check

input_file = "inputs/rad_m1_photon_singlezone.athinput"


def test_run():
    """Run a single test."""
    try:
        results = testutils.run(input_file)
        assert results, "Photon M1 single-zone LTE test run failed."
        assert check.main([]), "Photon M1 single-zone LTE check failed."
    finally:
        testutils.cleanup()
