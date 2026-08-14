"""
Unit tests for Task D2: div(B)=0 is preserved to roundoff by the area/edge-length-
weighted CT curl (src/mhd/mhd_ct.cpp, Task D1), for cartesian (control), cylindrical,
cylindrical_axisym, and spherical_polar. B is initialized as the discrete curl of an
edge-centered vector potential using the SAME Area/Len tables CT itself reads, so this
is a topological identity (see the pgen's docstring for why it is necessary but not
sufficient -- Task D3's physical induction tests are the real correctness gate).
"""

import test_suite.testutils as testutils


def test_ct_divb_cartesian():
    try:
        assert testutils.run("inputs/ut_ct_divb_cartesian.athinput")
    finally:
        testutils.cleanup()


def test_ct_divb_cylindrical():
    try:
        assert testutils.run("inputs/ut_ct_divb_cylindrical.athinput")
    finally:
        testutils.cleanup()


def test_ct_divb_axisym():
    try:
        assert testutils.run("inputs/ut_ct_divb_axisym.athinput")
    finally:
        testutils.cleanup()


def test_ct_divb_spherical():
    try:
        assert testutils.run("inputs/ut_ct_divb_spherical.athinput")
    finally:
        testutils.cleanup()
