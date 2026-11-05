"""
Flat-space mesh tests of resistive GRMHD on the dyngr path (07_scoping_v2 step 1a):
1D/2D convergence and limits, ideal-limit regression, constraints C1, C2, C4, C5,
ideal-gas and tabulated EOS.  Uses pgen resistive_tests (inputs/resistive_flat.athinput).
"""

import glob
import math
import os
import shutil
import subprocess
import pytest

_INPUT = "inputs/resistive_flat.athinput"
# optional launcher prefix, e.g. "mpiexec -n 1 gpu_tile_compact.sh" for GPU builds
_LAUNCH = os.environ.get("ATHENA_LAUNCHER", "").split()


def _run(name, extra_keys=(), flags=(), restart=None, cwd=None):
    """Run athena; new <problem>/<output> keys are appended to a temporary input file
    (new keys cannot be added from the command line). Returns the parsed .resist.dat."""
    with open(_INPUT) as f:
        text = f.read()
    text += "".join(k + "\n" for k in extra_keys)
    inp = "resist_" + name + ".athinput"
    with open(inp, "w") as f:
        f.write(text)
    if restart is None:
        cmd = _LAUNCH + ["./athena", "-i", inp, "job/basename=" + name] + list(flags)
    else:
        cmd = _LAUNCH + [os.path.abspath("athena"), "-r", restart]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True, cwd=cwd)
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert "resistive c2p failed" not in proc.stdout, proc.stdout[-3000:]
    dat = os.path.join(cwd or ".", name + ".resist.dat")
    out = {}
    with open(dat) as f:
        for line in f:
            k, v = line.split()
            out[k] = float(v)
    return out


def _state(base):
    d = {}
    for fn in glob.glob(base + ".state.*.txt"):
        with open(fn) as f:
            for line in f:
                p = line.split()
                d[(int(p[0]), int(p[1]), int(p[2]))] = p[3:]
    return d


def _l1(a, b, col):
    keys = sorted(a)
    return sum(abs(float.fromhex(a[k][col]) - float.fromhex(b[k][col]))
               for k in keys)/len(keys)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    for pat in ["resist_*.athinput", "*.resist.dat", "*.state.*.txt", "rt_*.rst",
                "rst", "rt_rr"]:
        for fn in glob.glob(pat):
            if os.path.isdir(fn):
                shutil.rmtree(fn)
            else:
                os.remove(fn)


def _rate(e1, e2):
    return math.log(e1/e2)/math.log(2.0)


def test_emwave_1d_convergence():
    r = [_run("rt_em%d" % n, flags=["mesh/nx1=%d" % n, "meshblock/nx1=%d" % n])
         for n in (64, 128)]
    assert _rate(r[0]["err_e3"], r[1]["err_e3"]) > 1.9
    assert _rate(r[0]["err_b2"], r[1]["err_b2"]) > 1.9
    assert r[1]["err_e3"] < 3.0e-6
    for x in r:
        assert x["max_divb"] == 0.0 and abs(x["charge"]) < 1e-14


def test_emwave_2d_convergence():
    r = []
    for n in (64, 128):
        r.append(_run("rt_em2d%d" % n, ["amp2 = 5.0e-4"],
                      ["mesh/nx1=%d" % n, "meshblock/nx1=%d" % n,
                       "mesh/nx2=%d" % (n//2), "meshblock/nx2=%d" % (n//2),
                       "mesh/x2max=0.5"]))
    for c in ("err_e1", "err_e2", "err_e3", "err_b1", "err_b2", "err_b3"):
        assert _rate(r[0][c], r[1][c]) > 1.6, c
    for x in r:
        assert x["max_divb"] < 1e-13 and abs(x["charge"]) < 1e-16


@pytest.mark.parametrize("sigma,ecut,rmin", [(1, 2.0e-7, 1.8), (10, 2.0e-8, 1.8),
                                              (1000, 5.0e-9, None)])
def test_telegraph(sigma, ecut, rmin):
    r = [_run("rt_tg%d" % n, flags=["mesh/nx1=%d" % n, "meshblock/nx1=%d" % n,
                                    "problem/test=telegraph",
                                    "mhd/resistive_sigma=%g" % sigma])
         for n in (64, 128)]
    assert r[1]["err_e2"] < ecut and r[1]["err_b3"] < 10*ecut
    if rmin is not None:
        assert _rate(r[0]["err_e2"], r[1]["err_e2"]) > rmin
        assert _rate(r[0]["err_b3"], r[1]["err_b3"]) > rmin
    assert r[1]["newton_fail"] == 0 and r[1]["newton_fallback"] == 0


def test_current_sheet():
    r = _run("rt_cs", flags=["mesh/nx1=200", "meshblock/nx1=200", "mesh/x1min=-1.5",
                             "mesh/x1max=1.5", "mesh/ix1_bc=outflow",
                             "mesh/ox1_bc=outflow", "problem/test=cursheet",
                             "problem/pgas=50", "mhd/resistive_sigma=100",
                             "time/tlim=9"])
    assert r["err_b2"] < 2.0e-3 and r["err_e3"] < 4.0e-4
    assert r["max_divb"] == 0.0 and r["newton_fail"] == 0


_RIEMANN = ["x0 = 0.5", "rho_l = 1", "p_l = 1", "rho_r = 0.125", "p_r = 0.1",
            "bx_l = 0.5", "bx_r = 0.5", "by_l = 1", "by_r = -1"]
_RFLAGS = ["problem/test=riemann", "mhd/gamma=2", "time/tlim=0.4", "mesh/nx1=400",
           "meshblock/nx1=400", "problem/dump_state=true", "mesh/ix1_bc=outflow",
           "mesh/ox1_bc=outflow"]


def test_riemann_ideal_limit():
    _run("rt_rpi", _RIEMANN, _RFLAGS + ["mhd/resistive=false"])
    ref = _state("rt_rpi")
    d = []
    for s in (10, 100, 1000, 1000000):
        r = _run("rt_rp%d" % s, _RIEMANN, _RFLAGS + ["mhd/resistive_sigma=%g" % s])
        assert r["newton_fail"] == 0 and r["newton_fallback"] == 0
        d.append(_l1(_state("rt_rp%d" % s), ref, 6))     # B^y face (b0.x2f)
    assert all(d[i+1] < 0.5*d[i] for i in range(len(d) - 1)), d
    assert d[-1] < 1.0e-3, d


_OT = ["problem/test=orszag_tang", "mesh/nx1=64", "mesh/nx2=64", "time/tlim=0.25",
       "problem/dump_state=true"]


def test_orszag_tang_constraints_and_decomposition():
    """C1 div B, C2 total charge (periodic), C4 MeshBlock invariance, ideal limit."""
    a = _run("rt_ot1", flags=_OT + ["meshblock/nx1=64", "meshblock/nx2=64",
                                    "mhd/resistive_sigma=1e6"])
    b = _run("rt_ot4", flags=_OT + ["meshblock/nx1=32", "meshblock/nx2=16",
                                    "mhd/resistive_sigma=1e6"])
    i = _run("rt_oti", flags=_OT + ["meshblock/nx1=64", "meshblock/nx2=64",
                                    "mhd/resistive=false"])
    for x in (a, b):
        assert x["max_divb"] < 1e-12
        assert abs(x["charge"] - x["charge0"]) < 1e-14
        assert x["newton_fail"] == 0 and x["newton_fallback"] == 0
    sa, sb = _state("rt_ot1"), _state("rt_ot4")
    assert sa == sb, "C4: results differ between MeshBlock decompositions"
    assert _l1(sa, _state("rt_oti"), 0) < 5.0e-3


def test_restart_bitwise():
    """C5: E~ is in the restart file; restarted run is bitwise identical."""
    keys = ["<output1>", "file_type = rst", "dt = 0.125"]
    flags = _OT + ["meshblock/nx1=32", "meshblock/nx2=32", "mhd/resistive_sigma=100"]
    _run("rt_c5", keys, flags)
    os.makedirs("rt_rr", exist_ok=True)
    _run("rt_c5", restart=os.path.abspath("rst/rt_c5.00001.rst"), cwd="rt_rr")
    assert _state("rt_c5") == _state("rt_rr/rt_c5")


_TAB = ["table = inputs/tables/SFHo_reduced.athtab", "nscalars = 1", "use_NQT = false"]


def _tab_input(name, extra_flags):
    with open(_INPUT) as f:
        text = f.read()
    text = text.replace("dyn_eos     = ideal", "dyn_eos     = compose")
    text = text.replace("<mhd>\n", "<mhd>\n" + "".join(k + "\n" for k in _TAB), 1)
    text += "".join(k + "\n" for k in
                    ["x0 = 0.5", "rho_l = 1.0e-3", "p_l = 1.0e-5", "rho_r = 1.25e-4",
                     "p_r = 1.0e-6", "bx_l = 0.5e-3", "bx_r = 0.5e-3", "by_l = 1.0e-3",
                     "by_r = -1.0e-3"])
    inp = "resist_" + name + ".athinput"
    with open(inp, "w") as f:
        f.write(text)
    flags = ["problem/test=riemann", "time/tlim=0.4", "mesh/nx1=256",
             "meshblock/nx1=128", "problem/dump_state=true", "mesh/ix1_bc=outflow",
             "mesh/ox1_bc=outflow", "mhd/dfloor=1.28e-13", "mhd/tfloor=0.01",
             "job/basename=" + name] + extra_flags
    proc = subprocess.run(_LAUNCH + ["./athena", "-i", inp] + flags, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, universal_newlines=True)
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert "resistive c2p failed" not in proc.stdout, proc.stdout[-3000:]
    out = {}
    with open(name + ".resist.dat") as f:
        for line in f:
            k, v = line.split()
            out[k] = float(v)
    return out


def test_tabulated_eos_riemann():
    """Resistive c2p and implicit step with the CompOSE table policy (SFHo)."""
    _tab_input("rt_tabi", ["mhd/resistive=false"])
    r = _tab_input("rt_tab", ["mhd/resistive_sigma=1e6"])
    assert r["newton_fail"] == 0 and r["newton_fallback"] == 0 and r["newton_cells"] > 0
    d = _l1(_state("rt_tab"), _state("rt_tabi"), 7)   # 5 cons + Y_e, then b0 x1, x2
    # |B^y| = 1e-3; the residual is confined to the shocks, where resistive LF (light-speed
    # dissipation) and the ideal solver (cold-plasma wave speeds) differ.
    assert d < 1.0e-5, d
