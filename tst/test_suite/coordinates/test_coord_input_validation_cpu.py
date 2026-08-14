"""
Unit tests for Task A1: CoordinateGeneral input parsing and Mesh::ValidateCoordGeneral().

All cases build the Mesh (nlim=0, tlim=0) and exit without evolving, using the shared
base input tst/inputs/ut_coord_validation.athinput with block/param=value command-line
overrides. "Good" cases must exit 0; "bad" cases are expected to hit one of the FATAL
ERROR checks in Mesh::ValidateCoordGeneral() (or the pre-existing coord-string-parsing
check) and exit nonzero, which testutils.run() surfaces as a RuntimeError.
"""

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


def test_multilevel_with_curvilinear_fails():
    flags = ["mesh/coord=spherical_polar", "mesh_refinement/refinement=static"]
    try:
        _run_fails(flags)
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
