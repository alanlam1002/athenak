"""
Regression test for the MeshBlock::SetNeighbors neighbor-table symmetry
invariant (src/cfc/SETNEIGHBORS_HANDOFF.md section 8).

Every consumer of the `nghbr` table -- MeshBoundaryValues::BuildRankPackedVarMetadata
in src/bvals/bvals.cpp, and the separate reimplementation in
src/multigrid/multigrid_bvals.cpp -- builds its MPI messages purely from each
rank's own rows, with no cross-rank negotiation. That is correct only if

    nghbr(b,n) = {gid=T, dest=d}   =>   nghbr(T,d) = {gid=b, dest=n}

holds for every populated slot. When it does not, one rank posts a send with no
matching recv, the aggregated per-peer message sizes disagree, and the run dies
in `MPI internal_Waitall` / `Message truncated` inside whichever subsystem
exchanges first -- with nothing in the message pointing at SetNeighbors.

MeshBlock::CheckNeighborSymmetry audits the invariant directly when
ATHENAK_CHECK_NGHBR_SYMMETRY is set; this test runs it on a topology known to
expose the defect and asserts the audit comes back clean.

WHY THIS TOPOLOGY. It is three levels deep with the finest region placed
ASYMMETRICALLY inside the coarser one (see the input file's own header). That
asymmetry is the whole point: the other SetNeighbors reproducers in this suite
are octant-symmetric, so every coarser diagonal candidate in them is
parity-matched and they report zero asymmetries under every registration rule
tried -- they cannot distinguish a correct rule from a broken one. This one can:
2164 registrations / 0 asymmetric under the correct rule, versus 2288 / 124
under the rule that blocked the TDE production restart.

Serial, 127 MeshBlocks, nlim=0 -- a few seconds.
"""

# Modules
import os

import test_suite.testutils as testutils


def test_nghbr_symmetry():
    input_file = "inputs/amr_hydro_nghbr_asymmetry.athinput"

    # The audit only runs when explicitly enabled, so that ordinary runs pay
    # nothing for it.
    prev = os.environ.get(testutils.NGHBR_AUDIT_ENV)
    os.environ[testutils.NGHBR_AUDIT_ENV] = "1"

    # The log file is shared and appended to across the whole pytest session,
    # so only inspect the portion this run appends.
    log_offset = 0
    if os.path.exists(testutils.LOG_FILE_PATH):
        log_offset = os.path.getsize(testutils.LOG_FILE_PATH)

    try:
        testutils.run(input_file)
    finally:
        if prev is None:
            del os.environ[testutils.NGHBR_AUDIT_ENV]
        else:
            os.environ[testutils.NGHBR_AUDIT_ENV] = prev

    with open(testutils.LOG_FILE_PATH, "r") as log_file:
        log_file.seek(log_offset)
        log_text = log_file.read()

    n_audits, nmb = testutils.assert_nghbr_symmetry_clean(
        log_text, context=f" in {input_file}"
    )

    # Guard against the topology silently changing out from under the test: if
    # the tree is not the one described in the input file, a clean audit proves
    # nothing about the case this test exists to cover.
    assert nmb == 127, (
        f"expected the 127-MeshBlock 3-level tree this test is built around, "
        f"got {nmb} -- check inputs/amr_hydro_nghbr_asymmetry.athinput"
    )
    assert n_audits >= 1
