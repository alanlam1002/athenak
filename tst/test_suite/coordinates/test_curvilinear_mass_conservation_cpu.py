"""
Unit tests for Task B4 (conservation half): mass conservation under the Task A3
area/volume-weighted flux-divergence kernel on genuinely curvilinear grids (cylindrical,
cylindrical_axisym, spherical), extending test_hydro_mass_conservation_cpu.py's
Cartesian-only check. Unlike the Cartesian case (periodic domain), these use reflecting
boundaries (including the true origin for cylindrical_axisym and spherical, which is
the required layout in each case) -- a reflecting-wall Riemann flux has exactly zero
mass flux through that face, so the same telescoping-to-zero argument applies to a
closed reflecting/periodic domain as to a periodic one. Mass conservation is exact
regardless of reconstruction accuracy (Task B6 curvilinear-aware PLM is not yet done) --
conservation is a property of the divergence-operator's telescoping-sum structure, not
of how accurately the fluxes it sums approximate the true physical flux.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import test_suite.testutils as testutils  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "vis", "python"))
import athena_read  # noqa: E402

_REL_TOL = 1.0e-11


def _check_mass_conservation(input_file, hst_file):
    try:
        assert testutils.run(input_file)
        data = athena_read.hst(hst_file)
        mass = data["mass"]
        assert len(mass) >= 3, f"expected multiple hst samples, got {len(mass)}"
        m0 = mass[0]
        max_rel_dev = max(abs(m - m0) / abs(m0) for m in mass)
        assert max_rel_dev < _REL_TOL, (
            f"total mass not conserved to roundoff: max relative deviation "
            f"{max_rel_dev:g} (tolerance {_REL_TOL:g}), mass[0]={m0:g}, "
            f"mass values={list(mass)}"
        )
    finally:
        testutils.cleanup()


def test_cylindrical_mass_conservation():
    _check_mass_conservation(
        "inputs/ut_cylindrical_mass_conservation.athinput",
        "ut_cylindrical_mass_conservation.hydro.hst",
    )


def test_cylindrical_axisym_mass_conservation():
    _check_mass_conservation(
        "inputs/ut_cylindrical_axisym_mass_conservation.athinput",
        "ut_cylindrical_axisym_mass_conservation.hydro.hst",
    )


def test_spherical_mass_conservation():
    _check_mass_conservation(
        "inputs/ut_spherical_mass_conservation.athinput",
        "ut_spherical_mass_conservation.hydro.hst",
    )
