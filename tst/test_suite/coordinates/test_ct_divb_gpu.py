"""
GPU variant of the CT div(B)=0 preservation test (Task D1/D2), for all four coordinate
systems.

This is the second-most valuable curvilinear test to run on a device, after geometry
construction: it drives the area/edge-length-weighted CT curl in src/mhd/mhd_ct.cpp,
which is the kernel that reads the most distinct GeomData arrays (Area1/2/3 plus
Len1/2/3 -- 15 array reads per face), and div(B)=0 is a roundoff-level identity, so any
device-side geometry corruption shows up immediately rather than as a slow drift.

Watch for one specific device failure mode here: at x1min=0 the geometry factory sets
a1i(is) to exactly 0 (see src/coordinates/geometry_spherical.cpp), and mhd_ct.cpp
divides by Area1. Correctness relies on the numerator vanishing there too. If it ever
does not, the result is a silent NaN rather than a trap, so a div(B) check that comes
back as NaN (not merely large) points at that division, not at the CT stencil.
"""

import test_suite.testutils as testutils

_RES = ["mesh/nx1=128", "meshblock/nx1=128", "mesh/nx2=128", "meshblock/nx2=128"]


def test_ct_divb_cartesian_gpu():
    try:
        assert testutils.run("inputs/ut_ct_divb_cartesian.athinput", flags=_RES)
    finally:
        testutils.cleanup()


def test_ct_divb_cylindrical_gpu():
    try:
        assert testutils.run("inputs/ut_ct_divb_cylindrical.athinput", flags=_RES)
    finally:
        testutils.cleanup()


def test_ct_divb_axisym_gpu():
    try:
        assert testutils.run("inputs/ut_ct_divb_axisym.athinput", flags=_RES)
    finally:
        testutils.cleanup()


def test_ct_divb_spherical_gpu():
    try:
        assert testutils.run("inputs/ut_ct_divb_spherical.athinput", flags=_RES)
    finally:
        testutils.cleanup()
