"""
Shared driver for the M2f apparent-horizon tests (test_z4c_cartoon_ahf_cpu/_gpu.py).

z4c_spinning_puncture builds a maximal Kerr slice through the puncture (M = 1, spin
along physical +z) and runs the <fastflow> finder once on the initial data
(problem/ahf_at_init, nlim = 0), so every error below is initial-data interpolation
error, not evolution error. Under 2D Cartoon FastFlow maps each surface point to the
plane, interpolates there, takes the out-of-plane derivative from axisymmetry and
rotates (z4c/fastflow_cartoon.hpp); it reports physical Sx, Sy, Sz (D-017).
Checks (chat/0010, chat/0012, chat/0013):
  * Liu-Etienne-Shapiro data (T4 coordinate, D-023), chi = 0, 0.5, 0.95, -0.5: the
    horizon mass and chi = Sz/M^2 converge to the input at about the interpolation order;
  * J-sign gate: Sz has the sign of chi (zero for chi = 0), Sx = Sy = 0 to round-off, and
    chi -> -chi flips Sz and nothing else;
  * the momentum and Hamiltonian constraints of the data converge (SPIN_PUNC_CON);
  * Brandt-Seidel quasi-isotropic data (qi) at chi = 0.5;
  * 2D Cartoon vs 3D Cartesian, same data and finder, same dx: agree within the
    interpolation error, with the same spin sign.
Thresholds sit under the values measured in aurora/0012 (results/G_m2f*).
"""

import glob
import math
import os
import shutil
import subprocess

DECK2 = "inputs/z4c_cartoon_ahf.athinput"
DECK3 = "inputs/z4c_spin_ahf_3d.athinput"
COLS = ("mass", "Sx", "Sy", "Sz", "S", "area", "hrms", "hmean", "meanradius", "minradius")
_RUN = [0]


def cleanup():
    for d in glob.glob("ahf_run_*"):
        shutil.rmtree(d, ignore_errors=True)


def rh(coord, chi):
    s = math.sqrt(1.0 - chi * chi)
    return 0.25 * (1.0 + s) if coord == "les" else 0.5 * s


def run(dim, ng, n, chi, coord="les", extra=()):
    """One finder run; returns a dict of the last summary row plus the constraints."""
    _RUN[0] += 1
    d = f"ahf_run_{_RUN[0]}"
    os.makedirs(d, exist_ok=True)
    flags = [f"mesh/nghost={ng}", f"problem/punc_chi={chi}", f"problem/spin_coord={coord}",
             f"fastflow/initial_radius_0={1.1 * rh(coord, chi)}"]
    if dim == 2:
        deck, base = DECK2, "cahf"
        flags += [f"mesh/nx1={n}", f"mesh/nx2={2 * n}"]
    else:
        deck, base = DECK3, "sahf3"
        flags += [f"mesh/nx1={2 * n}", f"mesh/nx2={2 * n}", f"mesh/nx3={2 * n}"]
    proc = subprocess.run([os.path.abspath("athena"), "-i", os.path.abspath(deck)]
                          + flags + list(extra), cwd=d, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    res = {"rc": proc.returncode, "out": proc.stdout[-3000:]}
    for ln in proc.stdout.splitlines():
        if ln.startswith("SPIN_PUNC_CON"):
            t = ln.split()
            res["maxH"], res["maxM"] = float(t[2]), float(t[4])
    f = os.path.join(d, f"{base}.horizon_summary_0.txt")
    if os.path.exists(f):
        rows = [ln.split() for ln in open(f) if ln.strip() and not ln.startswith("#")]
        res["header"] = [ln for ln in open(f) if ln.startswith("#")]
        if rows:
            res.update({k: float(v) for k, v in zip(COLS, rows[-1][2:])})
    assert res["rc"] == 0, res["out"]
    assert "mass" in res and math.isfinite(res["mass"]), res["out"]
    return res


def chi_ah(r):
    return r["Sz"] / r["mass"] ** 2


def sign_gate(r, chi, tag):
    assert abs(r["Sx"]) <= 1e-12 and abs(r["Sy"]) <= 1e-12, (tag, r["Sx"], r["Sy"])
    if chi > 0:
        assert r["Sz"] > 0, (tag, "S_z must be > 0 for chi > 0", r["Sz"])
    elif chi < 0:
        assert r["Sz"] < 0, (tag, "S_z must be < 0 for chi < 0", r["Sz"])
    else:
        assert abs(r["Sz"]) <= 1e-14, (tag, r["Sz"])


TOL_M = {2: 2e-5, 4: 2e-8}       # |M - 1| at N = 64 (measured 4.7e-6, 3.0e-9)
TOL_CHI = {2: 1e-6, 4: 1e-9}     # |chi_AH - chi| at N = 64 (measured 3.8e-7, 2.2e-11)


def check_les(ng):
    """les data in 2D Cartoon at N = 32, 64: convergence, sign gate, mirror, constraints."""
    chis = (0.0, 0.5, 0.95, -0.5)
    r = {(n, c): run(2, ng, n, c) for n in (32, 64) for c in chis}
    assert any("2D Cartoon" in h for h in r[(64, 0.5)]["header"]), r[(64, 0.5)]["header"]
    for c in chis:
        lo, hi = r[(32, c)], r[(64, c)]
        sign_gate(lo, c, f"ng={ng} N=32 chi={c}")
        sign_gate(hi, c, f"ng={ng} N=64 chi={c}")
        e32, e64 = abs(lo["mass"] - 1.0), abs(hi["mass"] - 1.0)
        assert e64 <= TOL_M[ng], (ng, c, e64)
        assert math.log2(e32 / e64) >= 2 * ng - 2, (ng, c, e32, e64)
        assert abs(chi_ah(hi) - c) <= TOL_CHI[ng], (ng, c, chi_ah(hi))
        # constraints of the data: H (and M_i when chi != 0) at about the FD order
        # (ng = 4, chi = 0.95 is pre-asymptotic at N = 32: measured 5.66 / 5.42)
        assert math.log2(lo["maxH"] / hi["maxH"]) >= 2 * (ng - 1) - 0.7, (ng, c, lo, hi)
        if c != 0.0:
            assert math.log2(lo["maxM"] / hi["maxM"]) >= 2 * (ng - 1) - 0.7, (ng, c)
        else:
            assert hi["maxM"] == 0.0, hi["maxM"]
    # chi -> -chi: Sz flips, the rest is unchanged
    for n in (32, 64):
        p, m = r[(n, 0.5)], r[(n, -0.5)]
        assert abs(p["mass"] - m["mass"]) <= 1e-13, (n, p["mass"], m["mass"])
        assert abs(p["Sz"] + m["Sz"]) <= 1e-13, (n, p["Sz"], m["Sz"])


def check_qi(ng=4):
    """Brandt-Seidel quasi-isotropic data, chi = +-0.5, N = 64."""
    for c in (0.5, -0.5):
        r = run(2, ng, 64, c, coord="qi")
        sign_gate(r, c, f"qi chi={c}")
        assert abs(r["mass"] - 1.0) <= 1e-7, r["mass"]   # measured 1.4e-8
        assert abs(chi_ah(r) - c) <= 1e-6, chi_ah(r)     # measured 9.3e-8


def check_2d_vs_3d(ng=2, n=32):
    """2D Cartoon vs 3D Cartesian at the same dx: same horizon within the interpolation
    error, same spin sign (chat/0012 item 2 at the finder level)."""
    for c in (0.5, -0.5):
        r2, r3 = run(2, ng, n, c), run(3, ng, n, c)
        sign_gate(r2, c, f"2D chi={c}")
        sign_gate(r3, c, f"3D chi={c}")
        e = max(abs(r2["mass"] - 1.0), abs(r3["mass"] - 1.0))
        assert e <= 3e-5, (r2["mass"], r3["mass"])                  # measured 1.4e-5
        assert abs(r2["mass"] - r3["mass"]) <= 2.0 * e, (r2["mass"], r3["mass"])
        assert abs(chi_ah(r2) - chi_ah(r3)) <= 3e-5, (chi_ah(r2), chi_ah(r3))
