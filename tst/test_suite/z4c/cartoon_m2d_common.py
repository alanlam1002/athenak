"""
Shared driver for the M2d vacuum tests at pytest size (test_z4c_cartoon_m2d_cpu/_gpu.py).
The long runs (D-026: T2-T4 at L = 32M to t = 100M, T5 at dx = M/16, T6) are batch jobs;
these checks are their cheap counterparts. Analysis: vis/python/cartoon_m2d.py.

  * T1 (gauge pulse, z4c_cartoon_pulse): at t = 2 and 4 the Hamiltonian constraint in
    the causal window (outside the outer boundary's domain of influence, D-026)
    converges at about the finite-difference order between N = 32 and 64 per 8M.
  * T2 (trumpet, static gauge, z4c_cartoon_trumpet, dx = M/8 and M/16, t = 5M): the
    initial-data constraints converge (TRUMPET_CON); ||H|| over 1M < r in the causal
    window converges at about the finite-difference order; the deviation from the exact
    stationary solution for r > 1M (whole domain, so including the region the outer
    boundary has reached) stays small.
  * T3 (Brill-Lindquist puncture, moving-puncture gauge, z4c_cartoon_punc, dx = M/8,
    t = 2M): FastFlow with find_dt_0 = 0.5 searches on the first cycle whose start time
    is at or after each multiple of 0.5M (t = 0, 0.5, 1, 1.5); the horizon mass stays
    near M and the spin vanishes.
  * T5 (spinning puncture chi = 0.95, z4c_cartoon_spin vs z4c_spin_3d, dx = M/8, t = 1
    and 2M): on y = 0, in both domains of dependence, the 2D Cartoon and 3D Cartesian
    states agree to truncation level, and every odd-y component (g_xy, g_yz, A_xy, A_yz,
    beta^y, Gamma^y) has the same sign in both wherever it is above 1e-3 of its maximum.
Thresholds sit beyond the values measured in aurora/0018 (results/N_m2d_probe).
"""

import glob
import math
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "vis", "python"))
import cartoon_m2d as cm  # noqa: E402

_RUN = [0]


def cleanup():
    for d in glob.glob("m2d_run_*"):
        shutil.rmtree(d, ignore_errors=True)


def run(deck, extra=()):
    _RUN[0] += 1
    d = f"m2d_run_{_RUN[0]}"
    os.makedirs(d, exist_ok=True)
    proc = subprocess.run([os.path.abspath("athena"), "-i",
                           os.path.abspath(os.path.join("inputs", deck))] + list(extra),
                          cwd=d, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True)
    assert proc.returncode == 0, proc.stdout[-3000:]
    return d, proc.stdout


def planes(d, kind):
    """{time: plane} of the bin files of one kind ('z4c' or 'con')."""
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "bin", "*.bin"))):
        if f".{kind}." in os.path.basename(f):
            p = cm.read_plane(f)
            out[round(p["time"], 6)] = p
    return out


def _read_user_hst(d):
    f = [g for g in glob.glob(os.path.join(d, "*.hst"))
         if ".z4c." not in os.path.basename(g)]
    assert len(f) == 1, f
    lines = open(f[0]).read().splitlines()
    hdr = [ln for ln in lines if ln.startswith("#") and "[1]" in ln][-1]
    lab = {v: int(k) - 1 for k, v in re.findall(r"\[(\d+)\]=(\S+)", hdr)}
    rows = [list(map(float, ln.split())) for ln in lines
            if ln.strip() and not ln.startswith("#")]
    return lab, rows


# --- T1 -------------------------------------------------------------------------------
T1_ORDER_MIN = {2: 1.5, 4: 4.5}     # measured >= 2.09 / 6.02 (N = 32 -> 64)


def check_t1(ng):
    """||H|| in the causal window at t = 2, 4 for N = 32, 64 (per 8M): the ratio gives
    an order of at least T1_ORDER_MIN[ng] (nominal 2 (ng - 1))."""
    cs = []
    for n in (32, 64):
        d, _ = run("z4c_cartoon_pulse.athinput",
                   [f"mesh/nghost={ng}", f"mesh/nx1={n}", f"mesh/nx2={2 * n}",
                    "time/tlim=4.0"])
        cs.append(planes(d, "con"))
    for t in (2.0, 4.0):
        assert all(t in c for c in cs), (t, [sorted(c) for c in cs])
        h = [cm.l2(c[t], "con_H", cm.causal_mask(c[t])) for c in cs]
        assert all(math.isfinite(x) and x > 0 for x in h), h
        order = math.log2(h[0] / h[1])
        assert order >= T1_ORDER_MIN[ng], (ng, t, h, order)


# --- T2 -------------------------------------------------------------------------------
T2_ORDER_MIN = {2: 1.5, 4: 4.5}     # ||H|| window order at t = 5 (measured 2.00 / 7.63)
T2_DEV_MAX = 2e-3                   # max deviation norm at dx = M/8 (measured <= 1.8e-4)
T2_DEVS = ("da-norm2", "dchi-norm2", "dKhat-nrm2", "dAt-norm2")


def _trumpet(ng, n):
    d, out = run("z4c_cartoon_trumpet.athinput",
                 [f"mesh/nghost={ng}", f"mesh/nx1={n}", f"mesh/nx2={2 * n}",
                  "time/tlim=5.0"])
    con = [ln.split() for ln in out.splitlines() if ln.startswith("TRUMPET_CON")]
    assert con, out[-3000:]
    lab, rows = _read_user_hst(d)
    assert rows[-1][0] >= 5.0 - 1e-9, rows[-1][0]
    dev = {k: [math.sqrt(r[lab[k]] / r[lab["vol-err"]]) for r in rows] for k in T2_DEVS}
    return dict(maxH=float(con[0][2]), maxM=float(con[0][4]), dev=dev,
                con=planes(d, "con"))


def check_t2(ng):
    """Static-gauge trumpet to t = 5M at dx = M/8 and M/16."""
    a, b = _trumpet(ng, 128), _trumpet(ng, 256)
    f = 2.0 ** (T2_ORDER_MIN[ng])
    assert b["maxH"] * f < a["maxH"] and b["maxM"] * f < a["maxM"], (a, b)
    for k in T2_DEVS:
        assert all(math.isfinite(x) for x in a["dev"][k] + b["dev"][k]), k
        assert max(a["dev"][k]) <= T2_DEV_MAX, (k, max(a["dev"][k]))
    for t in (0.0, 5.0):
        h = [cm.l2(c[t], "con_H", cm.causal_mask(c[t], rmin=1.0))
             for c in (a["con"], b["con"])]
        assert all(math.isfinite(x) and x > 0 for x in h), (t, h)
        assert math.log2(h[0] / h[1]) >= T2_ORDER_MIN[ng], (ng, t, h)


# --- T5 -------------------------------------------------------------------------------
# max |2D - 3D| / max |3D| over r >= 1M in both windows, dx = M/8 (measured
# alpha / chi / Khat: 1.1e-3 / 7.6e-4 / 1.5e-2 at ng = 2, 4e-5 / 1e-4 / 1.7e-3 at ng = 4)
T5_REL_MAX = {2: 5e-2, 4: 1e-2}
T5_SCALARS = ("z4c_alpha", "z4c_chi", "z4c_Khat")


def check_t5(ng=2):
    """2D Cartoon vs 3D Cartesian on y = 0 at dx = M/8, t = 1 and 2M."""
    d2, _ = run("z4c_cartoon_spin.athinput", [f"mesh/nghost={ng}"])
    d3, _ = run("z4c_spin_3d.athinput", [f"mesh/nghost={ng}"])
    p2, p3 = planes(d2, "z4c"), planes(d3, "z4c")
    for t in (1.0, 2.0):
        assert t in p2 and t in p3, (t, sorted(p2), sorted(p3))
        r = cm.compare(p2[t], p3[t], rmin=1.0)
        assert r["n_order"] > 100, r["n_order"]
        for v in T5_SCALARS:
            assert r["vars"][v]["rel_max"] <= T5_REL_MAX[ng], (t, v, r["vars"][v])
        checked = {k: s for k, s in r["sign"].items() if s["n_checked"] > 0}
        assert {"z4c_Axy", "z4c_Ayz", "z4c_Gamy", "z4c_betay"} <= set(checked), r["sign"]
        assert r["sign_ok"], (t, r["sign"])


# --- T3 / FastFlow cadence -------------------------------------------------------------
T3_MASS_TOL = 2e-2          # |M_AH - 1| at dx = M/8, t <= 1.5M (measured 3.6e-3 - 6.2e-3)


def check_t3_find_dt():
    d, _ = run("z4c_cartoon_punc.athinput")
    f = glob.glob(os.path.join(d, "*.horizon_summary_0.txt"))
    assert len(f) == 1, f
    rows = [list(map(float, ln.split())) for ln in open(f[0])
            if ln.strip() and not ln.startswith("#")]
    times = [r[1] for r in rows]
    # the finder runs at the cycle's start time: one search per 0.5M window up to tlim
    assert len(rows) == 4, times
    for k, t in enumerate(times):
        assert 0.5 * k - 1e-9 <= t < 0.5 * k + 1.0 / 32.0, (k, times)
    for r in rows:
        assert abs(r[2] - 1.0) <= T3_MASS_TOL, (r[1], r[2])
        assert max(abs(r[3]), abs(r[4]), abs(r[5])) <= 1e-12, r[3:6]
