"""
Unit tests for Task A1: CoordinateGeneral input parsing and Mesh::ValidateCoordGeneral().

All cases build the Mesh (nlim=0, tlim=0) and exit without evolving, using the shared
base input tst/inputs/ut_coord_validation.athinput with block/param=value command-line
overrides. "Good" cases must exit 0; "bad" cases are expected to hit one of the FATAL
ERROR checks in Mesh::ValidateCoordGeneral() (or the pre-existing coord-string-parsing
check) and exit nonzero, which testutils.run() surfaces as a RuntimeError.
"""

import subprocess

import pytest
import test_suite.testutils as testutils

INPUT_FILE = "inputs/ut_coord_validation.athinput"


def _run_ok(flags):
    assert testutils.run(INPUT_FILE, flags)


def _run_fails(flags):
    with pytest.raises(RuntimeError):
        testutils.run(INPUT_FILE, flags)


def test_default_cartesian_succeeds():
    try:
        _run_ok([])
    finally:
        testutils.cleanup()


def test_spherical_polar_positive_x1min_succeeds():
    # Flipped back to _run_ok() now that Task B3 (spherical geometry factory) is done.
    try:
        _run_ok(["mesh/coord=spherical_polar"])
    finally:
        testutils.cleanup()


def test_cylindrical_axisym_nx3_one_succeeds():
    # Flipped back to _run_ok() now that Task B2 (cylindrical_axisym factory) is done.
    try:
        _run_ok(["mesh/coord=cylindrical_axisym"])
    finally:
        testutils.cleanup()


def test_unrecognized_coord_string_fails():
    try:
        _run_fails(["mesh/coord=not_a_real_coord"])
    finally:
        testutils.cleanup()


def test_cylindrical_negative_x1min_fails():
    try:
        _run_fails(["mesh/coord=cylindrical", "mesh/x1min=-1.0"])
    finally:
        testutils.cleanup()


def test_cylindrical_axisym_negative_x1min_fails():
    try:
        _run_fails(["mesh/coord=cylindrical_axisym", "mesh/x1min=-1.0"])
    finally:
        testutils.cleanup()


def test_spherical_polar_negative_x1min_fails():
    try:
        _run_fails(["mesh/coord=spherical_polar", "mesh/x1min=-1.0"])
    finally:
        testutils.cleanup()


def test_spherical_polar_bad_theta_range_fails():
    # nx2 must be >=4 to trigger multi_d (which is what the theta-range check requires);
    # bump meshblock/nx2 to match so it stays a single MeshBlock in x2.
    flags = [
        "mesh/coord=spherical_polar",
        "mesh/nx2=4",
        "mesh/x2min=-1.0",
        "mesh/x2max=1.0",
        "meshblock/nx2=4",
    ]
    try:
        _run_fails(flags)
    finally:
        testutils.cleanup()


def test_cylindrical_axisym_nx3_not_one_fails():
    flags = ["mesh/coord=cylindrical_axisym", "mesh/nx3=4", "meshblock/nx3=4"]
    try:
        _run_fails(flags)
    finally:
        testutils.cleanup()


def test_cylindrical_positive_x1min_succeeds():
    # Smoke test only (build+run to nlim=0 without crashing) -- full analytic
    # verification of the cylindrical geometry factory's actual values is Task B4.
    # Added alongside Task B1 to catch gross construction errors before B2/B3 build on
    # top of the same pattern.
    try:
        _run_ok(["mesh/coord=cylindrical"])
    finally:
        testutils.cleanup()


def test_multilevel_with_curvilinear_hydro_succeeds():
    # Cell-centered restriction/prolongation/flux-correction are now volume- and
    # area-weighted, so hydro + curvilinear + refinement is supported. This used to be
    # rejected; the coverage is kept (rather than deleted) and inverted, so that a
    # regression which re-broke conservation-safe refinement would be noticed.
    flags = ["mesh/coord=spherical_polar", "mesh_refinement/refinement=static"]
    try:
        _run_ok(flags)
    finally:
        testutils.cleanup()


def test_multilevel_with_curvilinear_mhd_fails():
    # The face-centered half is NOT done: RestrictFC/ProlongFC* still average without
    # Area weighting and flux_correct_fc.cpp averages EMFs without Len weighting, so MHD
    # + curvilinear + refinement would break both conservation and div(B)=0. Must stay
    # a hard error until that lands.
    flags = ["mesh/coord=spherical_polar", "mesh_refinement/refinement=static"]
    try:
        with pytest.raises(RuntimeError):
            testutils.run("inputs/ut_coord_validation_mhd.athinput", flags)
    finally:
        testutils.cleanup()


def test_origin_outflow_fails():
    # Task E1: x1min=0 is the coordinate singularity, not an ordinary boundary --
    # outflow there is silently unphysical and must be rejected.
    flags = ["mesh/coord=spherical_polar", "mesh/x1min=0.0"]  # ix1_bc stays outflow
    try:
        _run_fails(flags)
    finally:
        testutils.cleanup()


def test_origin_reflect_succeeds():
    flags = ["mesh/coord=cylindrical_axisym", "mesh/x1min=0.0", "mesh/ix1_bc=reflect"]
    try:
        _run_ok(flags)
    finally:
        testutils.cleanup()


# D-009: vacuum z4c (no <hydro>/<mhd>, general_rel=false) on a curvilinear mesh must be
# fatal. The old guard only tested is_dynamical_relativistic, which also needs a fluid,
# so this combination slipped through. The fail cases check the guard's own message, so
# the test cannot pass by dying for an unrelated reason. The deck has the same mesh as
# ut_coord_validation.athinput (x1min > 0), so ValidateCoordGeneral() passes for every
# coord and only the z4c guard can fire.
Z4C_INPUT = "inputs/ut_coord_validation_z4c.athinput"
GUARD_MSG = "GR and dynamical-GR (z4c/adm) require coord=cartesian"


def _run_z4c(flags):
    cmd = ["./athena", "-i", Z4C_INPUT] + flags
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True)
    return proc.returncode, proc.stdout


def test_vacuum_z4c_cartesian_succeeds():
    try:
        rc, out = _run_z4c([])
        assert rc == 0, out[-2000:]
    finally:
        testutils.cleanup()


@pytest.mark.parametrize("coord", ["cylindrical", "spherical_polar"])
def test_vacuum_z4c_curvilinear_fails(coord):
    try:
        rc, out = _run_z4c([f"mesh/coord={coord}"])
        assert rc != 0
        assert GUARD_MSG in out, out[-2000:]
    finally:
        testutils.cleanup()


# D-014: 2D Cartoon is the one curvilinear coord the guard allows for z4c/adm: vacuum
# (no <hydro>/<mhd>, no GR) on a 2D cylindrical_axisym mesh. The 2D deck runs the
# Cartoon unit-test pgen to nlim=0. Everything else stays fatal with the guard's message,
# including the 1D deck above with cylindrical_axisym (Cartoon needs a 2D mesh there).
Z4C_2D_INPUT = "inputs/ut_coord_validation_z4c2d.athinput"
Z4C_2D_HYDRO_INPUT = "inputs/ut_coord_validation_z4c2d_hydro.athinput"


def _run_deck(deck, flags):
    cmd = ["./athena", "-i", deck] + flags
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True)
    return proc.returncode, proc.stdout


def test_vacuum_z4c_cylindrical_axisym_2d_succeeds():
    try:
        rc, out = _run_deck(Z4C_2D_INPUT, [])
        assert rc == 0, out[-2000:]
        assert "Z4c Cartoon axisym" in out, out[-2000:]
    finally:
        testutils.cleanup()


def test_vacuum_z4c_spherical_polar_2d_fails():
    try:
        rc, out = _run_deck(Z4C_2D_INPUT, ["mesh/coord=spherical_polar"])
        assert rc != 0
        assert GUARD_MSG in out, out[-2000:]
    finally:
        testutils.cleanup()


def test_vacuum_z4c_cylindrical_axisym_1d_fails():
    try:
        rc, out = _run_z4c(["mesh/coord=cylindrical_axisym"])
        assert rc != 0
        assert GUARD_MSG in out, out[-2000:]
    finally:
        testutils.cleanup()


def test_z4c_hydro_cylindrical_axisym_2d_fails():
    try:
        rc, out = _run_deck(Z4C_2D_HYDRO_INPUT, [])
        assert rc != 0
        assert GUARD_MSG in out, out[-2000:]
    finally:
        testutils.cleanup()
