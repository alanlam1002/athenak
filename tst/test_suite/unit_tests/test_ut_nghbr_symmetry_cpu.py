"""
Regression test for the MeshBlock::SetNeighbors neighbor-table symmetry
invariant (src/cfc/SETNEIGHBORS_HANDOFF.md).

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
ATHENAK_CHECK_NGHBR_SYMMETRY is set; this test runs it on a topology that is
known to expose the defect and asserts the audit comes back clean.

The topology (see the input file's own header for the full rationale) is three
levels deep with the finest region placed ASYMMETRICALLY inside the coarser
one. That asymmetry is essential: the older two-level, octant-symmetric
reproducer (lwave_hydro_diag_collision.athinput) reports zero asymmetries under
every rule tried so far and therefore cannot catch this.

Serial, ~127 MeshBlocks, nlim=0 -- a few seconds.
"""

# Modules
import os
import re
import test_suite.testutils as testutils


def test_nghbr_symmetry():
    input_file = "inputs/amr_hydro_nghbr_asymmetry.athinput"

    # The audit only runs when explicitly enabled, so that ordinary runs pay
    # nothing for it. Mode 1 reports and continues; the assertions below read
    # the report out of the shared session log.
    prev = os.environ.get("ATHENAK_CHECK_NGHBR_SYMMETRY")
    os.environ["ATHENAK_CHECK_NGHBR_SYMMETRY"] = "1"

    # The log file is shared and appended to across the whole pytest session,
    # so only inspect the portion this run appends.
    log_offset = 0
    if os.path.exists(testutils.LOG_FILE_PATH):
        log_offset = os.path.getsize(testutils.LOG_FILE_PATH)

    try:
        testutils.run(input_file)
    finally:
        if prev is None:
            del os.environ["ATHENAK_CHECK_NGHBR_SYMMETRY"]
        else:
            os.environ["ATHENAK_CHECK_NGHBR_SYMMETRY"] = prev

    with open(testutils.LOG_FILE_PATH, "r") as log_file:
        log_file.seek(log_offset)
        log_text = log_file.read()

    match = re.search(
        r"nghbr symmetry audit: (\d+) MeshBlocks, (\d+) registrations, "
        r"(\d+) asymmetric",
        log_text,
    )
    # A missing audit line means the run never reached the mesh build, or the
    # binary predates CheckNeighborSymmetry -- either way the test has not
    # actually checked anything, so fail rather than pass vacuously.
    assert match is not None, (
        "no 'nghbr symmetry audit' line in the run log -- "
        "MeshBlock::CheckNeighborSymmetry did not run"
    )

    nmb, nreg, nbad = (int(g) for g in match.groups())
    # Guard against the topology silently changing out from under the test: if
    # the tree is not the one described in the input file, a clean audit proves
    # nothing about the case this test exists to cover.
    assert nmb == 127, f"expected the 127-MeshBlock 3-level tree, got {nmb}"
    assert nreg > 2000, f"implausibly few registrations ({nreg}) for this tree"
    assert nbad == 0, (
        f"{nbad} asymmetric neighbor registrations on a topology that must be "
        "symmetric. Each one is a send with no matching recv; at multi-rank "
        "scale they abort the run in MPI_Waitall during the first boundary "
        "exchange. See src/cfc/SETNEIGHBORS_HANDOFF.md.\n" + log_text[-4000:]
    )
