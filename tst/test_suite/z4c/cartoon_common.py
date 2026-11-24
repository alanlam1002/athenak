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
