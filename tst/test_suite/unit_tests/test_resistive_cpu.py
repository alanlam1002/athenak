"""
Unit tests for resistive GRMHD building blocks (07_scoping_v2 step 0: K1, K6,
diagnostics) and the input guards (risk R1, configuration checks).
"""

# Modules
import subprocess
import pytest
import test_suite.testutils as testutils


def test_resistive_unit_3d():
    testutils.run("inputs/ut_resistive.athinput")


def test_resistive_unit_2d():
    testutils.run("inputs/ut_resistive.athinput",
                  ["mesh/nx3=1", "meshblock/nx3=1", "problem/ntrial=512"])


_lwave = ["time/nlim=0", "time/tlim=0", "mesh/nx2=1", "mesh/nx3=1",
          "meshblock/nx2=1", "meshblock/nx3=1", "mesh_refinement/refinement=none",
          "job/basename=ResistGuard"]


def _run_capture(mhd_keys, flags, tmp_path):
    """Run lwave_dyngrmhd with extra <mhd>/<time> keys (new keys cannot be added
    from the command line, so they are inserted into a temporary input file)."""
    with open("inputs/lwave_dyngrmhd.athinput") as f:
        text = f.read()
    text = text.replace("<mhd>\n", "<mhd>\n" + "".join(k + "\n" for k in mhd_keys), 1)
    inp = tmp_path / "resist_guard.athinput"
    inp.write_text(text)
    proc = subprocess.run(["./athena", "-i", str(inp)] + _lwave + flags,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True)
    return proc.returncode, proc.stdout


@pytest.mark.parametrize("keys,flags,msg", [
    (["eta_ohm = 0.01"], [], "eta_ohm/eta_ad"),
    (["eta_ad = 0.01"], [], "eta_ohm/eta_ad"),
    (["resistive = true", "resistive_sigma = 10"], [], "requires <time> integrator"),
    (["resistive = true"], ["time/integrator=imex2"], "resistive_sigma is required"),
    (["resistive = true", "resistive_sigma = 10"], ["time/integrator=imex2"],
     "not implemented yet"),
    (["resistive = true", "resistive_sigma = 10", "resistive_charge_flux = bogus"],
     ["time/integrator=imex2"], "resistive_charge_flux"),
    (["resistive = true", "resistive_sigma_model = density", "resistive_sigma_in = 1e6"],
     ["time/integrator=imex2"], "resistive_sigma_out is required"),
])
def test_resistive_guards(keys, flags, msg, tmp_path):
    rc, out = _run_capture(keys, flags, tmp_path)
    assert rc != 0, out
    assert msg in out, out


def test_resistive_allow_incomplete_initialises(tmp_path):
    keys = ["resistive = true", "resistive_sigma = 10", "resistive_allow_incomplete = true"]
    rc, out = _run_capture(keys, ["time/integrator=imex2"], tmp_path)
    assert rc == 0, out
    assert "INCOMPLETE" in out, out


def test_resistive_off_is_default(tmp_path):
    rc, out = _run_capture([], [], tmp_path)
    assert rc == 0, out
    assert "INCOMPLETE" not in out, out
