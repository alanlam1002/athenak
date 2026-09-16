"""
Regression/tracking test for cross-rank MPI aborts out of
`MeshBlock::SetNeighbors` (src/mesh/meshblock.cpp): at 8 MPI ranks on real
CFC + BH-puncture + TOV physics, an asymmetric neighbor registration -- one
where block b names (T, dest=d) but T's own row at slot d does not name b
back -- becomes a send with no matching recv, and kills the run with
"Fatal error in internal_Wait: Message truncated" before cycle 0.

History, since the docstring has been rewritten more than once as the
diagnosis changed (SETNEIGHBORS_HANDOFF.md has the full account):
  - a cross-rank `nvars`-cache desync was blamed, then refuted (section 3);
  - an "orphaned registration when one side defers" mechanism was blamed
    next, and a rule change landed for it (sections 2b / 5);
  - that rule change turned out to be the actual source of the asymmetries,
    and the original octant-parity guard it replaced was correct all along
    (section 8). The guard is restored as of commit b0d987ee.

The direct check for what this test is really about is now
`ATHENAK_CHECK_NGHBR_SYMMETRY=1`, which makes MeshBlock::CheckNeighborSymmetry
report asymmetric registrations by (block, slot) at mesh-build time rather than
leaving them to surface as an opaque MPI abort later. Setting it here would
make a failure of this test far easier to attribute.

The `xfail` marker stays for a reason unrelated to any of the above: this
input separately hits a CFC elliptic-solver convergence failure at cycle-0
metric initialization, producing widespread NANS_IN_CONS, plus a later SIGBUS
after AthenaK's own clean nlim-termination print. That symptom is also present
in the unmodified-code control, i.e. it is not caused by any SetNeighbors
change -- see SETNEIGHBORS_HANDOFF.md sections 5 and 8.8. Remove `xfail` only
once THAT is separately fixed or shown unrelated; until then this test will not
exit 0 regardless of the neighbor table being correct.

Requires a build configured with -DPROBLEM=dyn_grmhd/dyngr_tov (the CFC +
BH-puncture + TOV problem generator) -- NOT the built_in_pgens default the
rest of this test suite builds against. Skips cleanly (not xfail) if that
pgen isn't compiled into the current ./athena binary, since a normal
`run_test_suite.py --mpicpu` invocation builds built_in_pgens by default.
"""

# Modules
import subprocess
import pytest
import test_suite.testutils as testutils


def _cfc_tov_pgen_available() -> bool:
    """Check whether the currently-built ./athena has dyn_grmhd/dyngr_tov
    compiled in, via `-h`'s ShowConfig() dump (src/utils/show_config.cpp)."""
    try:
        result = subprocess.run(
            ["mpirun", "-np", "1", "./athena", "-h"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception:
        return False
    return "dyn_grmhd/dyngr_tov" in result.stdout


@pytest.mark.xfail(
    reason=(
        "src/cfc/SETNEIGHBORS_HANDOFF.md section 5: the original "
        "'internal_Wait: Message truncated' abort (item 39c) is fixed and "
        "confirmed (job 8830513). xfail remains for a separate, apparently "
        "pre-existing elliptic-solver-convergence/SIGBUS issue with this "
        "same input -- see the handoff note."
    ),
    strict=False,
)
def test_setneighbors_cfc_amr():
    if not _cfc_tov_pgen_available():
        pytest.skip(
            "requires a build configured with -DPROBLEM=dyn_grmhd/dyngr_tov; "
            "the built_in_pgens build this test suite uses by default does "
            "not include the TOV/puncture problem generator this input needs"
        )
    input_file = "inputs/cfc_setneighbors_amr_smoke.athinput"
    testutils.mpi_run(input_file, threads=8)
