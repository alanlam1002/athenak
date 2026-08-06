"""
Grey photon M1 optically-thick (kappa_s=200) diffusion test, MPI-decomposed.

Same physics as test_rad_m1_photon_diffusion_cpu.py, but the domain is split
into 4 meshblocks run under 4 MPI ranks (meshblock/nx1=32 over mesh/nx1=128),
to confirm the diffusion result is unaffected by domain decomposition.
"""

# Modules
import test_suite.testutils as testutils
import check_rad_m1_photon_diffusion as check

input_file = "inputs/rad_m1_photon_diffusion_mpi.athinput"


def test_run():
    """Run a single test with MPI."""
    try:
        results = testutils.mpi_run(input_file, threads=4)
        assert results, "Photon M1 MPI diffusion test run failed."
        assert check.main([]), "Photon M1 MPI diffusion check failed."
    finally:
        testutils.cleanup()
