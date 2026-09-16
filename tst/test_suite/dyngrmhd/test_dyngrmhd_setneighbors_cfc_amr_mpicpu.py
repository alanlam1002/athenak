"""
Regression/tracking test for item 39c (src/cfc/SETNEIGHBORS_HANDOFF.md,
section 2b): an orphaned/asymmetric neighbor registration in
`MeshBlock::SetNeighbors` (src/mesh/meshblock.cpp) -- when one block defers
registering a coarser diagonal neighbor (trusting the other side's own
unconditional branch to "cover" it), that block's own row is left with
nothing, but its own row is what its own rank's send/recv bookkeeping
(`MeshBoundaryValues::BuildRankPackedVarMetadata`, src/bvals/bvals.cpp) is
built from -- producing an orphaned registration that aborts real
multi-rank runs with "Fatal error in internal_Wait: Message truncated"
before cycle 0, at 8 MPI ranks, on real CFC + BH-puncture + TOV physics.
(An earlier theory blaming a cross-rank `nvars`-cache desync in the same
function was investigated and refuted -- see SETNEIGHBORS_HANDOFF.md
section 3, step 3, for why.)

A minimal, targeted patch for this (SETNEIGHBORS_HANDOFF.md section 5) is
now applied and CONFIRMED to fix this specific abort (job 8830513, 2026-09-16
-- zero occurrences of internal_Wait/internal_Waitall/Message truncated/
Abort(, first time this test's own scenario has gotten past cycle 0 at all).
The `xfail` marker stays for now anyway: the same job surfaced a SEPARATE,
apparently pre-existing problem with this input (a CFC elliptic-solver
convergence failure at cycle-0 metric init, producing widespread
NANS_IN_CONS, plus a later SIGBUS after AthenaK's own clean nlim-termination
print) that also matches the unmodified-code control test's own symptom --
see SETNEIGHBORS_HANDOFF.md section 5 for the full trace. Remove `xfail`
only once THAT is separately fixed or shown unrelated to this test's actual
purpose; until then this test still won't exit 0.

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
