"""
Grey photon M1 1D free-streaming beam test, single block.

Scattering-only (kappa_a=kappa_p=0), a causality-saturated beam (E=1, F_x=E)
is continuously injected at x=0 and should free-stream at c=1. Checks the
front position x_front~c*t, the illuminated-plateau E, and F_x/E all stay
close to their injected values. See
inputs/tests/check_rad_m1_photon_beam_1d.py for the full derivation.
"""

# Modules
import test_suite.testutils as testutils
import check_rad_m1_photon_beam_1d as check

input_file = "inputs/rad_m1_photon_beam_1d.athinput"


def test_run():
    """Run a single test."""
    try:
        results = testutils.run(input_file)
        assert results, "Photon M1 1D beam test run failed."
        assert check.main([]), "Photon M1 1D beam check failed."
    finally:
        testutils.cleanup()
