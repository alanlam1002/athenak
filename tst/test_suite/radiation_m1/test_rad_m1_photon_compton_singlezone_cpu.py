"""
Grey photon M1 single-zone Compton-relaxation test.

kappa_p=kappa_a=0, so the only channel driving E toward equilibrium is the
Compton term folded into eta_1/abs_1 by CalcOpacityPhotons_IdealGas_. Checks
E(t) against the same relaxation-law form as the pure-absorption LTE test,
but with abs_1 derived from the Compton rate. See
inputs/tests/check_rad_m1_photon_compton_singlezone.py for the full
derivation.
"""

# Modules
import test_suite.testutils as testutils
import check_rad_m1_photon_compton_singlezone as check

input_file = "inputs/rad_m1_photon_compton_singlezone.athinput"


def test_run():
    """Run a single test."""
    try:
        results = testutils.run(input_file)
        assert results, "Photon M1 Compton single-zone test run failed."
        assert check.main([]), "Photon M1 Compton single-zone check failed."
    finally:
        testutils.cleanup()
