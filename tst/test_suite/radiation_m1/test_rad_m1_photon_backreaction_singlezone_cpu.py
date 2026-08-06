"""
Grey photon M1 single-zone radiation-pressure backreaction test (backreact=true).

Same LTE single-zone setup as test_rad_m1_photon_singlezone_cpu.py, except the
gas temperature now responds to the radiation-matter energy exchange, so E
and T_gas jointly relax to the equilibrium set by total energy conservation.
See inputs/tests/check_rad_m1_photon_backreaction_singlezone.py for the full
derivation (energy conservation + independently-solved joint equilibrium).
"""

# Modules
import test_suite.testutils as testutils
import check_rad_m1_photon_backreaction_singlezone as check

input_file = "inputs/rad_m1_photon_backreaction_singlezone.athinput"


def test_run():
    """Run a single test."""
    try:
        results = testutils.run(input_file)
        assert results, "Photon M1 backreaction single-zone test run failed."
        assert check.main([]), "Photon M1 backreaction single-zone check failed."
    finally:
        testutils.cleanup()
