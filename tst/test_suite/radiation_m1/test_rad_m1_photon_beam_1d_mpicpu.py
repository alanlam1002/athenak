"""
Grey photon M1 1D free-streaming beam test, MPI-decomposed.

Same physics as test_rad_m1_photon_beam_1d_cpu.py, but the domain is split
into 4 meshblocks run under 4 MPI ranks (meshblock/nx1=100 over
mesh/nx1=400), to confirm the beam propagation result is unaffected by
domain decomposition.
"""

# Modules
import test_suite.testutils as testutils
import check_rad_m1_photon_beam_1d as check

input_file = "inputs/rad_m1_photon_beam_1d_mpi.athinput"


def test_run():
    """Run a single test with MPI."""
    try:
        results = testutils.mpi_run(input_file, threads=4)
        assert results, "Photon M1 MPI 1D beam test run failed."
        assert check.main([]), "Photon M1 MPI 1D beam check failed."
    finally:
        testutils.cleanup()
