"""
Flat-space mesh tests of resistive GRMHD on the dyngr path (07_scoping_v2 step 1a):
1D/2D convergence and limits, ideal-limit regression, constraints C1-C5 (C3 with SMR),
CP Alfven, rotor, ideal-gas and tabulated EOS; step 1b sigma(rho) closure, sigma_max
ceiling and sigma_M histogram on the KH layer (inputs/resistive_kh.athinput, T-S2 smoke).
Uses pgen resistive_tests (inputs/resistive_flat.athinput). The tearing and Sweet-
Parker S-scans are heavy (GPU) and run only with RESIST_HEAVY=1;
RESIST_HEAVY_LAUNCHERS="cmd1;cmd2;..." runs the scan members concurrently with one
launcher each.
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


def _cmd(name, extra_keys=(), flags=(), restart=None, launch=None, base=_INPUT):
    with open(base) as f:
        text = f.read()
    text += "".join(k + "\n" for k in extra_keys)
    inp = "resist_" + name + ".athinput"
    with open(inp, "w") as f:
        f.write(text)
    launch = _LAUNCH if launch is None else launch
    if restart is None:
        return launch + ["./athena", "-i", inp, "job/basename=" + name] + list(flags)
    return launch + [os.path.abspath("athena"), "-r", restart]


def _check(proc_out, rc):
    assert rc == 0, proc_out[-3000:]
    assert "resistive c2p failed" not in proc_out, proc_out[-3000:]


def _dat(name, cwd=None):
    dat = os.path.join(cwd or ".", name + ".resist.dat")
    out = {}
    with open(dat) as f:
        for line in f:
            k, v = line.split()
            out[k] = float(v)
    return out


def _run(name, extra_keys=(), flags=(), restart=None, cwd=None):
    """Run athena; new <problem>/<output> keys are appended to a temporary input file
    (new keys cannot be added from the command line). Returns the parsed .resist.dat."""
    proc = subprocess.run(_cmd(name, extra_keys, flags, restart), stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, universal_newlines=True, cwd=cwd)
    _check(proc.stdout, proc.returncode)
    return _dat(name, cwd)


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
    if os.environ.get("RESIST_KEEP", "0") == "1":
        return
    for pat in ["resist_*.athinput", "*.resist.dat", "*.state.*.txt", "rt_*.rst",
                "rst", "rt_rr", "rt_*.hst", "rt_*.log", "rt_*.rhist.txt"]:
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


# Palenzuela et al. (2009, MNRAS 394, 1727) sec. 5.1.3 Brio-Wu variant: B^x = 0
_RIEMANN_P09 = ["x0 = 0.5", "rho_l = 1", "p_l = 1", "rho_r = 0.125", "p_r = 0.1",
                "by_l = 0.5", "by_r = -0.5"]


@pytest.mark.parametrize("keys", [_RIEMANN, _RIEMANN_P09], ids=["bx05", "palenzuela09"])
def test_riemann_ideal_limit(keys):
    _run("rt_rpi", keys, _RFLAGS + ["mhd/resistive=false"])
    ref = _state("rt_rpi")
    d = []
    for s in (10, 100, 1000, 1000000):
        r = _run("rt_rp%d" % s, keys, _RFLAGS + ["mhd/resistive_sigma=%g" % s,
                                                 "mhd/resistive_rsolver=lf"])
        assert r["newton_fail"] == 0 and r["newton_fallback"] == 0
        d.append(_l1(_state("rt_rp%d" % s), ref, 6))     # B^y face (b0.x2f)
    assert all(d[i+1] < 0.5*d[i] for i in range(len(d) - 1)), d
    assert d[-1] < 1.0e-3, d


def _l1_fine(a, fine, col, rf):
    """L1 distance of a 1D state from a 1D reference rf times finer (cell averages)."""
    s = 0.0
    for (k, j, i), v in a.items():
        ref = sum(float.fromhex(fine[(k, j, rf*i + q)][col]) for q in range(rf))/rf
        s += abs(float.fromhex(v[col]) - ref)
    return s/len(a)


def test_riemann_mhllc():
    """Tetrad MHLLC (decision 4): converges to the ideal solution as sigma -> inf and
    is more accurate than the LF fallback against a 8x finer ideal reference."""
    _run("rt_rpf", _RIEMANN, _RFLAGS + ["mhd/resistive=false", "mesh/nx1=3200",
                                        "meshblock/nx1=3200"])
    fine = _state("rt_rpf")
    err = {}
    for rs in ("mhllc", "lf"):
        for s in (100, 1000000):
            name = "rt_rp%s%d" % (rs, s)
            r = _run(name, _RIEMANN, _RFLAGS + ["mhd/resistive_sigma=%g" % s,
                                                "mhd/resistive_rsolver=" + rs])
            assert r["newton_fail"] == 0 and r["newton_fallback"] == 0
            st = _state(name)
            err[(rs, s)] = (_l1_fine(st, fine, 0, 8), _l1_fine(st, fine, 4, 8))
    for c in range(2):                                     # D and tau
        assert err[("mhllc", 1000000)][c] < err[("mhllc", 100)][c], err
        assert err[("mhllc", 1000000)][c] < 0.9*err[("lf", 1000000)][c], err


# Komissarov (2007, MNRAS 382, 995; arXiv:0708.0323) sec. 6.1.1-2 stationary shocks,
# Gamma = 4/3: (rho, p, u = gamma v, B) left/right, tlim
_KOM = {"fast": ((1.0, 10.0, (0.6082, 0, 0), (5.0, 3.511, 0)),
                 (1.485, 17.03, (0.4096, -0.2147, 0), (5.0, 2.287, 0)), 3.0),
        "slow": ((1.0, 10.0, (4.925, 0, 0), (5.0, 15.08, 0)),
                 (7.930, 274.1, (0.6209, 0.1009, 0), (5.0, 28.92, 0)), 4.0)}


@pytest.mark.parametrize("shock", ["fast", "slow"])
@pytest.mark.parametrize("sigma", [100, 1000000])
def test_komissarov_stationary_shock(shock, sigma):
    """Stationary shock stays at x = 0 with the ideal jump states (100 cells on
    [-1, 1]; sigma = 100 is the paper's eta = 0.01)."""
    left, right, tlim = _KOM[shock]
    keys, dens = ["x0 = 0"], []
    for side, (rho, p, u, b) in (("l", left), ("r", right)):
        g = math.sqrt(1.0 + sum(x*x for x in u))
        keys += ["rho_%s = %.17g" % (side, rho), "p_%s = %.17g" % (side, p)]
        keys += ["v%s_%s = %.17g" % (a, side, x/g) for a, x in zip("xyz", u)]
        keys += ["b%s_%s = %.17g" % (a, side, x) for a, x in zip("xyz", b)]
        dens.append(rho*g)
    name = "rt_kom%s%d" % (shock, sigma)
    r = _run(name, keys, ["problem/test=riemann", "time/tlim=%g" % tlim,
                          "mesh/nx1=100", "meshblock/nx1=100", "mesh/x1min=-1",
                          "mesh/x1max=1", "mesh/ix1_bc=outflow", "mesh/ox1_bc=outflow",
                          "problem/dump_state=true", "time/cfl_number=0.3",
                          "mhd/resistive_sigma=%g" % sigma])
    assert r["newton_fail"] == 0 and r["newton_fallback"] == 0
    st = _state(name)
    d = [float.fromhex(st[k][0]) for k in sorted(st)]
    x = [-1.0 + 0.02*(k[2] + 0.5) for k in sorted(st)]
    mid = 0.5*(dens[0] + dens[1])
    xs = [x[i] for i in range(len(d) - 1) if (d[i] - mid)*(d[i+1] - mid) <= 0]
    assert len(xs) == 1 and abs(xs[0]) < 0.05, xs
    assert abs(d[5]/dens[0] - 1) < 0.01 and abs(d[-5]/dens[1] - 1) < 0.01, (d[5], d[-5])


_OT = ["problem/test=orszag_tang", "mesh/nx1=64", "mesh/nx2=64", "time/tlim=0.25",
       "problem/dump_state=true"]


def test_orszag_tang_constraints_and_decomposition():
    """C1 div B, C2 total charge (periodic), C4 MeshBlock invariance, ideal limit."""
    a = _run("rt_ot1", flags=_OT + ["meshblock/nx1=64", "meshblock/nx2=64",
                                    "mhd/resistive_sigma=1e6"])
    b = _run("rt_ot4", flags=_OT + ["meshblock/nx1=32", "meshblock/nx2=16",
                                    "mhd/resistive_sigma=1e6"])
    _run("rt_oti", flags=_OT + ["meshblock/nx1=64", "meshblock/nx2=64",
                                "mhd/resistive=false"])
    for x in (a, b):
        assert x["max_divb"] < 1e-12
        assert abs(x["charge"] - x["charge0"]) < 1e-14
        assert x["newton_fail"] == 0 and x["newton_fallback"] == 0
    sa, sb = _state("rt_ot1"), _state("rt_ot4")
    assert sa == sb, "C4: results differ between MeshBlock decompositions"
    assert _l1(sa, _state("rt_oti"), 0) < 5.0e-3


def _smr(x1, x2):
    return ["<mesh_refinement>", "refinement = static", "<refined_region1>", "level = 1",
            "x1min = %g" % x1[0], "x1max = %g" % x1[1], "x2min = %g" % x2[0],
            "x2max = %g" % x2[1]]


def test_smr_emwave():
    """Step 1a SMR (07 D3): a 2D EM wave on a two-level mesh is more accurate than the
    coarse uniform mesh and less than the fine one; div B and charge stay at roundoff."""
    base = ["mesh/nx2=32", "mesh/x2max=0.5", "meshblock/nx1=16", "meshblock/nx2=16",
            "time/tlim=0.2"]
    u1 = _run("rt_smru1", ["amp2 = 5.0e-4"], base + ["mesh/nx1=64"])
    s = _run("rt_smrs", ["amp2 = 5.0e-4"] + _smr((0.3, 0.7), (0.1, 0.4)),
             base + ["mesh/nx1=64"])
    u2 = _run("rt_smru2", ["amp2 = 5.0e-4"], base[2:] + ["mesh/nx1=128", "mesh/nx2=64",
                                                         "mesh/x2max=0.5"])
    for c in ("err_e1", "err_e2", "err_e3", "err_b1", "err_b2", "err_b3"):
        assert u2[c] < s[c] < 0.7*u1[c], c
    assert s["max_divb"] < 1e-13 and abs(s["charge"]) < 1e-18
    assert s["newton_fail"] == 0 and s["newton_fallback"] == 0


def test_smr_charge_conservation():
    """C3 on the mesh: with in-plane E (B_z != 0) charge is generated locally; the
    shared-face E~ sync keeps the periodic total at roundoff (without the sync the total
    drifts to ~3e-4 on this setup).  div B stays at roundoff."""
    r = _run("rt_smrot", ["bz_ot = 0.3"] + _smr((0.3, 0.6), (0.2, 0.7)),
             _OT[:-1] + ["meshblock/nx1=16", "meshblock/nx2=16",
                         "mhd/resistive_sigma=100"])
    assert r["max_abs_q"] > 0.5
    assert abs(r["charge"]) < 1e-14 and abs(r["charge_boundary"]) < 1e-14
    assert r["max_divb"] < 1e-12
    assert r["newton_fail"] == 0 and r["newton_fallback"] == 0


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
    proc = subprocess.run(_LAUNCH + ["./athena", "-i", inp] + flags,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True)
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
    # |B^y| = 1e-3; the residual is confined to the shocks, where resistive LF
    # (light-speed dissipation) and the ideal solver (cold-plasma wave speeds) differ.
    assert d < 1.0e-5, d


@pytest.mark.parametrize("sigma", [1, 10, 100, 1000000])
def test_cp_alfven(sigma):
    """1D circularly polarised Alfven wave along B_x: exact damped eigenmode of the
    linearised resistive equations (vacuum-like at sigma = 1, strongly damped at 10,
    ideal Alfven at 1e6); second-order convergence and polarisation symmetry."""
    r = [_run("rt_cp%d" % n, flags=["problem/test=cpalfven", "problem/amp1=1e-4",
                                    "mhd/resistive_sigma=%g" % sigma,
                                    "mesh/nx1=%d" % n, "meshblock/nx1=%d" % n])
         for n in (64, 128)]
    for c in ("err_e3", "err_b2", "err_b3"):
        assert _rate(r[0][c], r[1][c]) > 1.7, c
    assert r[1]["err_b2"] < 1.2e-7 and r[1]["err_e3"] < 1.2e-7
    assert abs(r[1]["err_b2"] - r[1]["err_b3"]) < 1e-6*r[1]["err_b2"]
    assert abs(r[1]["err_e3"] - r[1]["err_e2"]) < 1e-6*r[1]["err_e3"]
    assert r[1]["newton_fail"] == 0 and r[1]["newton_fallback"] == 0


_ROT = ["problem/test=rotor", "mhd/gamma=1.6666666666666667", "mesh/nx1=128",
        "mesh/nx2=128", "meshblock/nx1=64", "meshblock/nx2=64", "mesh/x1min=-0.5",
        "mesh/x1max=0.5", "mesh/x2min=-0.5", "mesh/x2max=0.5", "time/tlim=0.4",
        "problem/dump_state=true"]


def test_rotor():
    """Relativistic rotor (Lorentz factor ~10): sigma = 1e6 approaches the ideal run
    (the remaining gap, ~5% in D, is the MHLLC/LLF solver difference and is 3.5x smaller
    than the 128 vs 256 self-convergence error); sigma = 100 departs further; div B,
    charge and the implicit step stay healthy."""
    _run("rt_roti", flags=_ROT + ["mhd/resistive=false"])
    ref = _state("rt_roti")
    norm = sum(abs(float.fromhex(v[0])) for v in ref.values())/len(ref)
    d = {}
    for s in (100, 1000000):
        r = _run("rt_rot%d" % s, flags=_ROT + ["mhd/resistive_sigma=%g" % s])
        assert r["max_divb"] < 1e-11 and abs(r["charge"]) < 1e-14
        assert r["newton_fail"] == 0 and r["newton_fallback"] == 0
        d[s] = _l1(_state("rt_rot%d" % s), ref, 0)/norm
    assert d[1000000] < 0.07 and d[100] > 1.5*d[1000000], d


_HEAVY = os.environ.get("RESIST_HEAVY", "0") == "1"
_HL = [x.split() for x in os.environ.get("RESIST_HEAVY_LAUNCHERS", "").split(";")
       if x.strip()]


def _hist(name):
    rows = [list(map(float, ln.split())) for ln in open(name + ".user.hst")
            if not ln.startswith("#")]
    return [x[0] for x in rows], [x[2] for x in rows], [x[3] for x in rows]


def _scan(names, keys, flags):
    """Run the scan members concurrently (one launcher each) if launchers are given.
    Output goes to <name>.log (a pipe per run would block once its buffer is full)."""
    procs = []
    for n, (name, fl) in enumerate(zip(names, flags)):
        launch = _HL[n % len(_HL)] if _HL else None
        with open(name + ".log", "w") as log:
            procs.append(subprocess.Popen(_cmd(name, keys, fl + ["time/ndiag=100"],
                                               launch=launch),
                                          stdout=log, stderr=subprocess.STDOUT))
        if not _HL:
            procs[-1].wait()
    for name, p in zip(names, procs):
        p.wait()
        with open(name + ".log") as log:
            _check(log.read(), p.returncode)
    return [_dat(n) for n in names]


def _slope(x, y):
    lx = [math.log(v) for v in x]
    ly = [math.log(v) for v in y]
    mx, my = sum(lx)/len(lx), sum(ly)/len(ly)
    return (sum((a - mx)*(b - my) for a, b in zip(lx, ly)) /
            sum((a - mx)**2 for a in lx))


def _fit_growth(t, f, lo, hi):
    """Exponential growth rate of f between f0*lo and min(f0*hi, max(f)/2): least
    squares slope of ln f vs t."""
    top = min(hi*f[0], 0.5*max(f))
    sel = [(a, math.log(b)) for a, b in zip(t, f) if lo*f[0] < b < top]
    assert len(sel) > 5, (t, f)
    ma = sum(a for a, _ in sel)/len(sel)
    mb = sum(b for _, b in sel)/len(sel)
    return (sum((a - ma)*(b - mb) for a, b in sel) /
            sum((a - ma)**2 for a, _ in sel))


_HARRIS_KEYS = ["a_sheet = 0.1", "eps = 1.0e-6", "user_hist = true", "<output1>",
                "file_type = hst", "dt = 0.5", "user_hist_only = true"]
# rho = p = 0.2, gamma = 4/3: h = 1, v_A = b0/sqrt(h + b0^2) = 1/sqrt(2); S = a v_A sigma
_VA = 1.0/math.sqrt(2.0)


def _tearing_linear(s, ka):
    """Linear tearing growth rate gamma tau_A of a Harris sheet from the general
    (FKR/Coppi) inner-layer dispersion relation, Delta' delta = -(pi/8) Q^(5/4)
    Gamma((Q^(3/2) - 1)/4)/Gamma((Q^(3/2) + 5)/4), delta = a (ka S)^(-1/3),
    gamma tau_A = Q (ka)^(2/3) S^(-1/3); Q -> 0 recovers FKR 0.55 (Delta'a)^(4/5)
    (ka)^(2/5) S^(-3/5)."""
    rhs = 2.0*(1.0/ka - ka)*(ka*s)**(-1.0/3.0)

    def f(q):
        x = q**1.5
        return -math.pi/8.0*q**1.25*math.gamma(0.25*(x - 1.0))/math.gamma(0.25*(x + 5.0))

    lo, hi = 1.0e-8, 1.0 - 1.0e-12
    for _ in range(200):
        mid = 0.5*(lo + hi)
        lo, hi = (mid, hi) if f(mid) < rhs else (lo, mid)
    return 0.5*(lo + hi)*ka**(2.0/3.0)*s**(-1.0/3.0)


@pytest.mark.skipif(not _HEAVY, reason="heavy (GPU): set RESIST_HEAVY=1")
def test_tearing_scaling():
    """Resistive tearing of a force-free double Harris sheet (ka = 0.5, a/dy = 32): the
    linear growth rate gamma tau_A tracks the inner-layer dispersion relation over
    S = 250-2000. resistive_freeze_background (eta-compensation) keeps the equilibrium
    from diffusing on the tearing time scale, so the eigenmode rate is fitted after the
    initial transient (psi/psi0 = 10-1000). At these S the inner layer is not thin
    (delta/a ~ 0.1-0.15), so gamma is 0.7-0.95 of the asymptotic rate and approaches it
    monotonically with S; a/dy = 64 agrees within 6% (07 v2.1 4.1)."""
    svals = (250, 500, 1000, 2000)
    names = ["rt_tear%d" % s for s in svals]
    lx = 2.0*math.pi/5.0
    base = ["problem/test=harris", "problem/rho=0.2", "problem/pgas=0.2",
            "mesh/nx1=32", "mesh/nx2=512", "meshblock/nx1=32", "meshblock/nx2=512",
            "mesh/x1min=%.17g" % (-0.5*lx), "mesh/x1max=%.17g" % (0.5*lx),
            "mesh/x2min=-0.8", "mesh/x2max=0.8"]
    tl = {250: 45, 500: 60, 1000: 85, 2000: 120}
    fl = [base + ["mhd/resistive_sigma=%.17g" % (s/(0.1*_VA)), "time/tlim=%g" % tl[s]]
          for s in svals]
    keys = _HARRIS_KEYS[:2] + ["<mhd>", "resistive_freeze_background = true",
                               "<problem>"] + _HARRIS_KEYS[2:]
    res = _scan(names, keys, fl)
    tau = 0.1/_VA
    g, th = [], []
    for s, name, r in zip(svals, names, res):
        assert r["max_divb"] < 1e-10 and r["newton_fail"] == 0
        t, _, psi = _hist(name)
        g.append(_fit_growth(t, psi, 10.0, 1000.0)*tau)
        th.append(_tearing_linear(s, 0.5))
    k = _slope(svals, g)
    ratio = [a/b for a, b in zip(g, th)]
    print("tearing gamma*tau_A", g, "linear theory", th, "ratio", ratio, "slope", k)
    assert all(0.65 < r < 1.1 for r in ratio), ratio
    assert all(b > a - 0.02 for a, b in zip(ratio, ratio[1:])), ratio
    assert -0.7 < k < -0.4, (g, k)


@pytest.mark.skipif(not _HEAVY, reason="heavy (GPU): set RESIST_HEAVY=1")
def test_sweet_parker_scaling():
    """X-point driven reconnection of a double Harris sheet (a = 0.06, 1024^2, eta-
    compensated background): after t ~ 2 the reconnection rate d psi/dt / (b0 v_A) is
    steady to ~10% (Sweet-Parker plateau) and its mean over t = 2-9 scales as S_L^-1/2,
    S_L = L v_A sigma with L = 0.5. delta_SP = L S_L^-1/2 is 4-8 cells here, so the
    slope (-0.42 at 1024^2, -0.25 at 512^2) is resolution-limited: [VERIFY] 2048^2."""
    svals = (1000, 2000, 4000)
    names = ["rt_sp%d" % s for s in svals]
    keys = ["a_sheet = 0.06", "eps = 0.1", "lam = 0.25", "pert = xpoint",
            "<mhd>", "resistive_freeze_background = true", "<problem>",
            "user_hist = true", "<output1>", "file_type = hst", "dt = 0.05",
            "user_hist_only = true"]
    base = ["problem/test=harris", "problem/rho=0.2", "problem/pgas=0.2",
            "mesh/nx1=1024", "mesh/nx2=1024", "meshblock/nx1=1024",
            "meshblock/nx2=1024", "mesh/x1min=-1", "mesh/x1max=1", "mesh/x2min=-1",
            "mesh/x2max=1", "time/tlim=9"]
    fl = [base + ["mhd/resistive_sigma=%.17g" % (s/(0.5*_VA))] for s in svals]
    res = _scan(names, keys, fl)

    def mean_rate(t, psi, lo, hi):
        sel = [(a, b) for a, b in zip(t, psi) if lo <= a <= hi]
        ma = sum(a for a, _ in sel)/len(sel)
        mb = sum(b for _, b in sel)/len(sel)
        return (sum((a - ma)*(b - mb) for a, b in sel) /
                sum((a - ma)**2 for a, _ in sel))/_VA

    rate = []
    for s, name, r in zip(svals, names, res):
        assert r["max_divb"] < 1e-10 and r["newton_fail"] == 0
        t, _, psi = _hist(name)
        r1, r2 = mean_rate(t, psi, 2.0, 5.5), mean_rate(t, psi, 5.5, 9.0)
        assert abs(r2/r1 - 1.0) < 0.15, (s, r1, r2)
        rate.append(mean_rate(t, psi, 2.0, 9.0))
        assert 0.3 < rate[-1]*math.sqrt(s) < 1.0, (s, rate[-1])
    k = _slope(svals, rate)
    print("sweet-parker rate", rate, "slope", k)
    assert -0.65 < k < -0.35, (rate, k)


_KH_FLAGS = ["mesh/nx1=32", "mesh/nx2=64", "meshblock/nx1=32", "meshblock/nx2=32",
             "time/tlim=1.0", "time/ndiag=100"]
_KH_DENSITY = ["resistive_sigma_model = density", "resistive_sigma_in = 1.0e5",
               "resistive_sigma_out = 10.0", "resistive_rho_transition = 0.1",
               "resistive_sigma_width = 0.5", "resistive_hist_dt = 0.5"]


def _run_kh(name, mhd_keys, flags=()):
    """KH layer (T-S2 setup) with extra <mhd> keys: they are spliced into the <mhd>
    block of inputs/resistive_kh.athinput (new keys cannot be set on the command line)."""
    with open("inputs/resistive_kh.athinput") as f:
        text = f.read()
    text = text.replace("<mhd>\n", "<mhd>\n" + "".join(k + "\n" for k in mhd_keys))
    inp = "resist_" + name + "_kh.athinput"
    with open(inp, "w") as f:
        f.write(text)
    proc = subprocess.run(_cmd(name, flags=list(_KH_FLAGS) + list(flags), base=inp),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True)
    _check(proc.stdout, proc.returncode)
    rows = [list(map(float, ln.split())) for ln in open(name + ".rhist.txt")
            if not ln.startswith("#")]
    hst = [list(map(float, ln.split())) for ln in open(name + ".user.hst")
           if not ln.startswith("#")]
    os.remove(name + ".user.hst")
    return _dat(name), rows, hst


def test_kh_sigma_rho_histogram():
    """Step 1b (07 v2.2 3.2, T-S2 smoke): density-dependent sigma(rho) on the KH layer
    with both implicit residual forms; every cell converges (no retry, continuation or
    fallback), the sigma_M histogram accounts for all interior cells, constraints hold,
    and forms g (c2p) and f (momentum) give the same KH amplitude."""
    dvy = []
    for form in ("c2p", "momentum"):
        r, rows, hst = _run_kh("rt_kh" + form, _KH_DENSITY +
                               ["resistive_implicit_residual = " + form])
        assert r["newton_fail"] == 0 and r["newton_fallback"] == 0
        assert r["max_divb"] < 1e-12 and abs(r["charge"]) < 1e-14
        cells = sum(x[4] for x in rows)
        assert cells == sum(x[5] for x in rows) and cells > 0
        assert sum(x[6] + x[7] + x[12] for x in rows) == 0     # retry, fallback, cont.
        assert len({x[0] for x in rows}) == 2                   # t = 0.5, 1.0
        assert abs(hst[-1][6] - r["max_divb"]) <= 1e-5*r["max_divb"]  # divb (07 v2.4)
        assert max(x[6] for x in hst) < 1e-12
        dvy.append(hst[-1][4])
    assert abs(dvy[1]/dvy[0] - 1.0) < 1e-6, dvy


def test_sigma_max_ceiling():
    """sigma_max (07 v2.2 3.2) caps b^2/rho at 50 on the KH layer (initially ~200 in the
    light fluid): ceiling events are counted, no cell is left above 10^2, and the run
    keeps div B and charge conservation."""
    r, rows, _ = _run_kh("rt_khcap", _KH_DENSITY + ["sigma_max = 50.0"],
                         ["time/tlim=0.5"])
    assert r["newton_fail"] == 0 and r["newton_fallback"] == 0
    assert r["max_divb"] < 1e-12 and abs(r["charge"]) < 1e-14
    assert sum(x[9] for x in rows) > 0
    assert sum(x[4] for x in rows if x[3] >= 2) == 0
