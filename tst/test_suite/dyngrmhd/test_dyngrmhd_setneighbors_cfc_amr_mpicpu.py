"""
Regression test for cross-rank MPI aborts out of `MeshBlock::SetNeighbors`
(src/mesh/meshblock.cpp), on real CFC + BH-puncture + TOV physics at 8 ranks.

WHAT IS BEING CHECKED, AND WHY NOT THE EXIT CODE

The nghbr table must satisfy, for every populated slot,

    nghbr(b,n) = {gid=T, dest=d}   =>   nghbr(T,d) = {gid=b, dest=n}

because MeshBoundaryValues::BuildRankPackedVarMetadata (src/bvals/bvals.cpp) and
the separate reimplementation in src/multigrid/multigrid_bvals.cpp both build
their MPI messages from each rank's own rows, with no cross-rank negotiation. A
violation is a send with no matching recv, and it kills a multi-rank run with
"Fatal error in internal_Wait(all)" before cycle 0.

This test asserts exactly that: the audit is clean, and no MPI truncation/abort
signature appears. It deliberately does NOT assert on the process exit code or
on NANS_IN_CONS, because this input separately hits an unrelated, unresolved
problem (see "known separate failure" below) that would make an exit-code
assertion fail for reasons that have nothing to do with the neighbor table.
Asserting only what this test can actually attribute is the point; the previous
version of this file asserted nothing at all beyond the exit code and was marked
xfail on top of that, so it could neither pass meaningfully nor fail usefully.

HISTORY -- worth reading before trusting any earlier version of this docstring,
which has been rewritten three times as the diagnosis changed. The full account
is src/cfc/SETNEIGHBORS_HANDOFF.md section 8:
  - a cross-rank `nvars`-cache desync was blamed, then refuted (section 3);
  - an "orphaned registration when one side defers" mechanism was blamed next,
    and a rule change landed for it (sections 2b / 5) -- this test was added by
    that same commit, fee50981, to validate it;
  - that rule change turned out to be the source of the asymmetries, and the
    octant-parity guard it replaced (which is also upstream AthenaK's own rule)
    was correct all along. Restored in commit b0d987ee.
On this input's own topology, measured directly: 624 MeshBlocks, 12472
registrations and 0 asymmetric under the restored rule, versus 12664
registrations and 192 asymmetric under the rule that replaced it.

KNOWN SEPARATE FAILURE, NOT THIS TEST'S SUBJECT

This input also produces a CFC elliptic-solver convergence failure at cycle-0
metric initialization, widespread NANS_IN_CONS, and a SIGBUS after AthenaK's own
clean nlim-termination print. That is present with the unmodified upstream rule
too, so it is not a SetNeighbors problem -- see SETNEIGHBORS_HANDOFF.md sections
5 and 8.8. It is unresolved and is why the exit code is not asserted here.

Requires a build configured with -DPROBLEM=dyn_grmhd/dyngr_tov (the CFC +
BH-puncture + TOV problem generator) -- NOT the built_in_pgens default the rest
of this test suite builds against. Skips cleanly if that pgen is not compiled
into the current ./athena binary, since a normal `run_test_suite.py --mpicpu`
invocation builds built_in_pgens by default.
"""

# Modules
import os
import subprocess

import pytest

import test_suite.testutils as testutils

INPUT_FILE = "inputs/cfc_setneighbors_amr_smoke.athinput"
NRANKS = 8

# MPI signatures specific to an unmatched/truncated boundary message. Kept
# deliberately narrow: a bare "Abort(" would also match the unrelated SIGBUS
# described above, and this test must only fail for things it can attribute.
ABORT_SIGNATURES = (
    "internal_Wait",
    "Message truncated",
)


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


def test_setneighbors_cfc_amr():
    if not _cfc_tov_pgen_available():
        pytest.skip(
            "requires a build configured with -DPROBLEM=dyn_grmhd/dyngr_tov; "
            "the built_in_pgens build this test suite uses by default does "
            "not include the TOV/puncture problem generator this input needs"
        )

    # The audit is opt-in so ordinary runs pay nothing for it.
    prev = os.environ.get(testutils.NGHBR_AUDIT_ENV)
    os.environ[testutils.NGHBR_AUDIT_ENV] = "1"

    # The log file is shared and appended to across the whole pytest session, so
    # only inspect the portion this run appends.
    log_offset = 0
    if os.path.exists(testutils.LOG_FILE_PATH):
        log_offset = os.path.getsize(testutils.LOG_FILE_PATH)

    try:
        # run_command rather than mpi_run: mpi_run raises on a nonzero exit, and a
        # nonzero exit here is expected for the unrelated reason in the docstring.
        # The return value is intentionally ignored; the assertions below are the
        # test.
        testutils.run_command(
            ["mpirun", "-np", str(NRANKS), "./athena", "-i", INPUT_FILE]
        )
    finally:
        if prev is None:
            del os.environ[testutils.NGHBR_AUDIT_ENV]
        else:
            os.environ[testutils.NGHBR_AUDIT_ENV] = prev

    with open(testutils.LOG_FILE_PATH, "r") as log_file:
        log_file.seek(log_offset)
        log_text = log_file.read()

    # 1. every SetNeighbors call in the run (initial build plus each AMR remesh)
    #    produced a symmetric table
    n_audits, nmb = testutils.assert_nghbr_symmetry_clean(
        log_text, context=f" in {INPUT_FILE} at {NRANKS} ranks"
    )
    assert nmb > 100, (
        f"audit saw only {nmb} MeshBlocks; this input should build a few hundred, "
        "so the topology is not the one this test is meant to cover"
    )

    # 2. and no rank died of a truncated/unmatched boundary message
    found = [sig for sig in ABORT_SIGNATURES if sig in log_text]
    assert not found, (
        f"MPI abort signature(s) {found} in the run log -- a boundary message had "
        "no matching recv. This is the failure mode this test exists to catch; see "
        f"src/cfc/SETNEIGHBORS_HANDOFF.md section 8.\n(audits seen: {n_audits})\n"
        + log_text[-4000:]
    )
