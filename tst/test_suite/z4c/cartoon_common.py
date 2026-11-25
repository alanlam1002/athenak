"""
Shared driver for the Z4c Cartoon fill unit tests (test_z4c_cartoon_cpu/_gpu.py).

The pgen (src/pgen/unit_tests/z4c_cartoon_test.cpp) does the work on a 2D
coord=cylindrical_axisym mesh and exits nonzero if an exact check fails: the axis
boundary on u0, the pi-copy vs direct interpolation at x < 0, and the position helper vs
the analytic midpoints. Here we scan resolution and check convergence orders from its
CARTOON_RESULT lines:
  * npt = 2*ng+1 (production): ghost values at about the interpolation order npt, and
    d_y, d_yy, d_xy, d_yz at y = 0 at the FD order 2(ng-1);
  * npt = 2*ng-1: the second derivatives d_yy, d_xy lose about one order
    (O(dx^(2ng-3)), chat/0004), which shows the order is set by the stencil width;
  * a deliberately wrong parity entry (beta^y as even) must make the run fail.
M2a (chat/0009), same runs with the production stencil:
  * ADMToZ4c on rotating analytic data: the pointwise part is an exact gate in the pgen;
    Gamma-tilde^i = -d_j g-tilde^ij converges at the FD order 2(ng-1);
  * H (Schwarzschild, Kerr-Schild with spin, D-017 mapping) and M_i, Z (Kerr) from the
    Cartoon-aware ADM constraints converge to 0 at the FD order; Z (Schwarzschild,
    conformally flat) stays at round-off;
  * negative control: K's odd-y components with the wrong sign must spoil M_i.
M2f (chat/0010, chat/0013): the FastFlow Cartoon point evaluation at rho = 0.03 ... 10 dx1
against the analytic field: g, K at order 2 ng, d_k g at 2 ng - 1, the near-axis
(rho < dx1) derivative error no larger than twice the error away from the axis.
Measured orders (N = 32 -> 64; CPU and PVC agree to round-off) are in the 1C report;
the thresholds below sit 0.3-0.6 under them.
"""

import math
import subprocess

DECK = "inputs/ut_z4c_cartoon.athinput"
NS = (16, 32, 64)
DERIVS = ("err_dy", "err_dyy", "err_dxy", "err_dyz")


def run(ng, npt, n, extra=()):
    b = n // 2
    flags = [f"mesh/nghost={ng}", f"mesh/nx1={n}", f"mesh/nx2={n}",
             f"meshblock/nx1={b}", f"meshblock/nx2={b}", f"problem/npt={npt}"]
    proc = subprocess.run(["./athena", "-i", DECK] + flags + list(extra),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    res = {}
    for ln in proc.stdout.splitlines():
        if ln.startswith("CARTOON_RESULT"):
            t = ln.split()
            try:
                res[t[1]] = float(t[2])
            except ValueError:
                pass
    return proc.returncode, proc.stdout, res


def orders(ng, npt):
    """Run the three resolutions; return {key: order on the finest pair}."""
    rs = []
    for n in NS:
        rc, out, res = run(ng, npt, n)
        assert rc == 0, out[-3000:]
        assert "CARTOON_TEST PASS" in out, out[-3000:]
        rs.append(res)
    return {k: math.log2(rs[1][k] / rs[2][k])
            for k in ("err_ghost_int",) + DERIVS}


def check_production(ng):
    npt = 2 * ng + 1
    o = orders(ng, npt)
    assert o["err_ghost_int"] >= npt - 0.6, o
    for k in DERIVS:
        assert o[k] >= 2 * (ng - 1) - 0.3, (k, o)


def check_narrow_stencil(ng):
    npt = 2 * ng - 1
    o = orders(ng, npt)
    for k in ("err_dyy", "err_dxy"):
        assert o[k] >= 2 * ng - 3 - 0.5, (k, o)
        assert o[k] <= 2 * (ng - 1) - 0.5, (k, o)


def check_wrong_parity_fails(ng):
    rc, out, _ = run(ng, 2 * ng + 1, 32, ["problem/wrong_parity=z4c_betaz"])
    assert rc != 0, out[-3000:]
    assert "CARTOON_TEST FAIL" in out, out[-3000:]


M2A_KEYS = ("err_gam", "err_H_schw", "err_H_kerr", "err_M_kerr", "err_Z_kerr")


def check_m2a(ng):
    """ADMToZ4c and ADM constraints under Cartoon (production stencil)."""
    npt = 2 * ng + 1
    rs = []
    for n in NS:
        rc, out, res = run(ng, npt, n)
        assert rc == 0, out[-3000:]
        assert "CARTOON_TEST PASS" in out, out[-3000:]
        rs.append(res)
    o = {k: math.log2(rs[1][k] / rs[2][k]) for k in M2A_KEYS}
    for k in M2A_KEYS:
        assert o[k] >= 2 * (ng - 1) - 0.3, (k, o)
    for r in rs:
        assert r["err_adm2z4c"] <= 1e-12, r
        assert r["err_Z_schw"] <= 1e-12, r


def check_m2a_sign_control(ng):
    """An inconsistent sign on K's odd-y components is an O(1) error in the data: the
    momentum constraint stays far above the correct-data value and does not converge."""
    npt = 2 * ng + 1
    rc0, out0, good = run(ng, npt, 32)
    rc1, out1, bad = run(ng, npt, 32, ["problem/kerr_flip_k_odd=true"])
    rc2, out2, bad2 = run(ng, npt, 64, ["problem/kerr_flip_k_odd=true"])
    assert rc0 == 0 and rc1 == 0 and rc2 == 0, (out0[-2000:], out1[-2000:], out2[-2000:])
    assert bad["err_M_kerr"] > 1.0e4 * good["err_M_kerr"], (
        good["err_M_kerr"], bad["err_M_kerr"])
    assert math.log2(bad["err_M_kerr"] / bad2["err_M_kerr"]) < 0.5, (bad, bad2)


M2F_KEYS = ("err_ffg", "err_ffK", "err_ffdg_far", "err_ffdg_axis")


def check_m2f_point(ng):
    """FastFlow Cartoon point evaluation (z4c/fastflow_cartoon.hpp, chat/0013): g, K at
    the interpolation order 2 ng, d_k g at 2 ng - 1, and near the axis (rho < dx1,
    where the out-of-plane derivative has a 1/rho) no worse than away from it."""
    npt = 2 * ng + 1
    rs = []
    for n in NS:
        rc, out, res = run(ng, npt, n)
        assert rc == 0 and "CARTOON_TEST PASS" in out, out[-3000:]
        rs.append(res)
    o = {k: math.log2(rs[1][k] / rs[2][k]) for k in M2F_KEYS}
    for k in ("err_ffg", "err_ffK"):
        assert o[k] >= 2 * ng - 0.5, (k, o)
    for k in ("err_ffdg_far", "err_ffdg_axis"):
        assert o[k] >= 2 * ng - 1 - 0.5, (k, o)
    # measured axis/far at N = 64: 0.91 (ng 2), 0.92 (3), 0.93 (4)
    assert rs[2]["err_ffdg_axis"] <= 2.0 * rs[2]["err_ffdg_far"], rs[2]
