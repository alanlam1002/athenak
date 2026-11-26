"""
Shared driver for the M2b outer-boundary tests (test_z4c_cartoon_pulse_cpu/_gpu.py).

z4c_gauge_pulse puts an axisymmetric lapse pulse,
alpha = 1 + A exp(-(R^2 + (z-z0)^2)/w^2), on flat space (w = 1, z0 = 1, puncture
gauge). Under 2D Cartoon the outer R face and both z faces are outflow, so Z4c applies
its Sommerfeld RHS there (z4c_Sbc.cpp, M2b); the gauge waves must leave the domain
without growth. The history columns are integrals with the axisymmetric volume element
(2 pi R dR dz), so before the waves reach the boundary they equal the 3D integrals of
the same field on a 3D Cartesian mesh.
Checks (chat/0009 M2b):
  * the pulse leaves: ||alpha - 1|| falls to a small fraction of its initial value;
  * no growth: the late-time maxima stay below the mid-time ones, and the final value
    below the late maximum, for alpha - 1 and Khat;
  * the residual left in the domain shrinks with resolution;
  * a nonlinear pulse (A = 0.1) also leaves;
  * 2D Cartoon vs 3D Cartesian before the boundary: the same integrals within the
    truncation error.
Thresholds sit under the values measured in aurora/0015 (results/J_m2b).
"""

import glob
import math
import os
import re
import shutil
import subprocess

DECK2 = "inputs/z4c_cartoon_pulse.athinput"
DECK3 = "inputs/z4c_pulse_3d.athinput"
_RUN = [0]


def cleanup():
    for d in glob.glob("pulse_run_*"):
        shutil.rmtree(d, ignore_errors=True)


def _read_hst(d):
    cols = {}
    # the pgen's own columns only (<basename>.user.hst; the Z4c file has other labels)
    for f in sorted(glob.glob(os.path.join(d, "*.hst"))):
        if ".z4c." in os.path.basename(f):
            continue
        lines = open(f).read().splitlines()
        hdr = [ln for ln in lines if ln.startswith("#") and "[1]" in ln]
        if not hdr:
            continue
        labels = re.findall(r"\[(\d+)\]=(\S+)", hdr[-1])
        data = [list(map(float, ln.split())) for ln in lines
                if ln.strip() and not ln.startswith("#")]
        for idx, lab in labels:
            cols[lab] = [r[int(idx) - 1] for r in data]
    return cols


def run(dim, ng, n, amp=1e-3, extra=()):
    _RUN[0] += 1
    d = f"pulse_run_{_RUN[0]}"
    os.makedirs(d, exist_ok=True)
    flags = [f"mesh/nghost={ng}", f"problem/pulse_amp={amp}"]
    if dim == 2:
        deck = DECK2
        flags += [f"mesh/nx1={n}", f"mesh/nx2={2 * n}"]
    else:
        deck = DECK3
        flags += [f"mesh/nx1={2 * n}", f"mesh/nx2={2 * n}", f"mesh/nx3={2 * n}"]
    proc = subprocess.run([os.path.abspath("athena"), "-i", os.path.abspath(deck)]
                          + flags + list(extra), cwd=d, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    assert proc.returncode == 0, proc.stdout[-3000:]
    c = _read_hst(d)
    assert "a-1-norm2" in c and "Khat-norm2" in c, proc.stdout[-3000:]
    for k in ("a-1-norm2", "Khat-norm2", "Thet-norm2", "Gam-norm2"):
        assert all(math.isfinite(x) for x in c[k]), k
    return c


def _norm(c, k):
    return [math.sqrt(x / v) for x, v in zip(c[k], c["vol-axi"])]


def summary(c):
    t = c["time"]
    na, nk = _norm(c, "a-1-norm2"), _norm(c, "Khat-norm2")
    mid = [i for i, x in enumerate(t) if 8.0 <= x < 16.0]
    late = [i for i, x in enumerate(t) if x >= 16.0]
    return dict(t_end=t[-1], na0=na[0], na_end=na[-1], nk_max=max(nk),
                na_mid=max(na[i] for i in mid), na_late=max(na[i] for i in late),
                nk_mid=max(nk[i] for i in mid), nk_late=max(nk[i] for i in late),
                nk_end=nk[-1])


def check_leaves(s, tag, f_end):
    assert s["t_end"] >= 24.0 - 1e-9, (tag, s)
    assert s["na_end"] <= f_end * s["na0"], (tag, s)
    assert s["na_late"] <= 0.5 * s["na_mid"], (tag, "alpha grows late", s)
    assert s["na_end"] <= s["na_late"], (tag, s)
    assert s["nk_late"] <= 0.2 * s["nk_max"], (tag, "Khat grows late", s)
    assert s["nk_end"] <= s["nk_late"], (tag, s)


def check_linear(ng):
    """Linear pulse (A = 1e-3) at N = 32 and 64: leaves, no growth, residual shrinks.
    Measured: na_end/na0 = 0.009 / 0.0036, na_late/na_mid <= 0.32, nk_late/nk_max <= 0.09,
    late residual ratio N 64 / 32 = 0.50."""
    s32, s64 = summary(run(2, ng, 32)), summary(run(2, ng, 64))
    check_leaves(s32, f"ng={ng} N=32", 0.02)
    check_leaves(s64, f"ng={ng} N=64", 0.01)
    assert s64["na_late"] <= 0.7 * s32["na_late"], (s32, s64)
    assert s64["nk_late"] <= 0.7 * s32["nk_late"], (s32, s64)


def check_nonlinear(ng=2):
    """A = 0.1 at N = 64 (measured na_end/na0 = 0.023)."""
    check_leaves(summary(run(2, ng, 64, amp=0.1)), "A=0.1", 0.05)


def check_2d_vs_3d(n=32):
    """Before the waves reach the outer boundary (t <= 4) the axisymmetric-weighted 2D
    integrals equal the 3D ones up to truncation (measured 5.3e-3 and 1.2e-2 at
    N = 32)."""
    c2 = run(2, 2, n, extra=["time/tlim=4.0"])
    c3 = run(3, 2, n, extra=["time/tlim=4.0"])
    n_t = min(len(c2["time"]), len(c3["time"]))
    assert n_t >= 16, n_t
    for k, tol in (("a-1-norm2", 1e-2), ("Khat-norm2", 2.5e-2)):
        a = [math.sqrt(x) for x in c2[k][:n_t]]
        b = [math.sqrt(x) for x in c3[k][:n_t]]
        scale = max(b)
        err = max(abs(x - y) for x, y in zip(a, b)) / scale
        assert err <= tol, (k, err)
