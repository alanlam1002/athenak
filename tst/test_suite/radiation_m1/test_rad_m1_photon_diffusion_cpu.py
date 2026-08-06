"""
Grey photon M1 optically-thick (kappa_s=200) diffusion test, single block.

Scattering-only (kappa_p=kappa_a=0), so a Gaussian E(x) pulse should spread
by pure radiative diffusion, sigma^2(t) = sigma0^2 + 2*D*t, D=1/(3*kappa_s*rho).
Also checks near-exact total-energy conservation. See
inputs/tests/check_rad_m1_photon_diffusion.py for the full derivation.
"""

# Modules
import test_suite.testutils as testutils
import check_rad_m1_photon_diffusion as check

input_file = "inputs/rad_m1_photon_diffusion.athinput"


def test_run():
    """Run a single test."""
    try:
        results = testutils.run(input_file)
        assert results, "Photon M1 diffusion test run failed."
        assert check.main([]), "Photon M1 diffusion check failed."
    finally:
        testutils.cleanup()
