"""
Regression test for the MeshBlock::SetNeighbors edge/corner diagonal-
registration bug (src/cfc/DEVELOPMENT.md items 38/39a-39g/42/63).

Runs a minimal, CFC-independent, plain-hydro AMR reproducer (one root
MeshBlock of a 4x4x4 root grid statically refined one level, periodic BCs)
that item 39g's original investigation used to demonstrate item 39d's
diagonal-neighbor slot-collision mechanism, and item 63 reused to confirm the
registration rule avoids it. This topology is two-level and fully
octant-symmetric (every coarser diagonal candidate is parity-matched, hence
genuinely redundant with an existing same-level or finer-branch registration),
which means it reports zero asymmetries under every rule tried so far and
CANNOT distinguish between them. Keep it as a cheap clean-registration check,
but for a topology that actually discriminates see
test_ut_nghbr_symmetry_cpu.py (three levels, finest region placed
asymmetrically) and SETNEIGHBORS_HANDOFF.md section 8. A clean exit and no
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
    # Enable the neighbor-table audit for this run. Without it this test checks
    # only "exited 0 and logged no NaNs", which on this deliberately symmetric
    # topology is true under every registration rule tried so far -- i.e. it
    # asserted almost nothing. The audit at least makes the table itself a
    # checked property. (For a topology that DISCRIMINATES between rules, see
    # test_ut_nghbr_symmetry_cpu.py.)
    prev = os.environ.get(testutils.NGHBR_AUDIT_ENV)
    os.environ[testutils.NGHBR_AUDIT_ENV] = "1"

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
    testutils.assert_nghbr_symmetry_clean(log_text, context=f" in {input_file}")

    assert "NANS_IN_CONS" not in log_text, (
        "NANS_IN_CONS appeared in the run log -- the SetNeighbors "
        "diagonal-registration rule may have regressed item 39d's "
        "slot-collision avoidance. See SETNEIGHBORS_HANDOFF.md section 8."
    )
