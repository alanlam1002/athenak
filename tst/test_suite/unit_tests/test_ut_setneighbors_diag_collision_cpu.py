"""
Regression test for the MeshBlock::SetNeighbors edge/corner diagonal-
registration bug (src/cfc/DEVELOPMENT.md items 38/39a-39g/42/63).

Runs a minimal, CFC-independent, plain-hydro AMR reproducer (one root
MeshBlock of a 4x4x4 root grid statically refined one level, periodic BCs)
that item 39g's original investigation used to demonstrate item 39d's
diagonal-neighbor slot-collision mechanism, and item 63 reused to confirm the
recip==nullptr fix (all 4 sites of src/mesh/meshblock.cpp) avoids it. This
topology is fully symmetric (every coarser diagonal candidate is genuinely
redundant with an existing same-level or finer-branch registration), so it
does not exercise the recip==nullptr ("genuinely needed") path itself -- see
item 63 for the CFC+puncture+TOV smoke test that does. A clean exit and no
NANS_IN_CONS in the run log is exactly what this fix should keep producing on
this geometry; a nonzero exit or any NANS_IN_CONS occurrence here would mean
this specific collision-avoidance mechanism has regressed.
"""

# Modules
import os
import test_suite.testutils as testutils


def test_diag_collision():
    input_file = "inputs/lwave_hydro_diag_collision.athinput"

    # testutils.run() only checks the process exit code; also check the run
    # log itself for the specific failure signature this test exists to catch
    # (a silent slot collision could in principle exit 0 while still logging
    # primitive-recovery failures -- exit code alone would miss that). The
    # log file is shared/appended-to across the whole pytest session, so only
    # inspect the portion this specific run appends, not the whole file.
    log_offset = 0
    if os.path.exists(testutils.LOG_FILE_PATH):
        log_offset = os.path.getsize(testutils.LOG_FILE_PATH)

    testutils.run(input_file)

    with open(testutils.LOG_FILE_PATH, "r") as log_file:
        log_file.seek(log_offset)
        log_text = log_file.read()
    assert "NANS_IN_CONS" not in log_text, (
        "NANS_IN_CONS appeared in the run log -- the SetNeighbors "
        "recip==nullptr fix (DEVELOPMENT.md item 63) may have regressed "
        "item 39d's diagonal-neighbor slot-collision avoidance."
    )
