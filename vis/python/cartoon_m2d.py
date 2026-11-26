#! /usr/bin/env python3
"""
Analysis helpers for the 2D Cartoon vacuum tests (M2d, D-026).

Reads AthenaK bin output of a 2D Cartoon run (coord = cylindrical_axisym, mesh x1 = R,
x2 = z) or of a 3D Cartesian run sliced at y = 0 (`slice_x2 = 0` with a cell centred on
y = 0), and returns the data on the physical (x, z) half-plane y = 0, x > 0 in the
PHYSICAL frame. Under Cartoon (D-017) physical (x, y, z) = mesh (x1, -x3, x2), so a
component changes sign once per physical y index, e.g. g_xy = -g_{13}(mesh),
beta^y = -beta^3(mesh), g_xz = g_{12}(mesh), g_zz = g_{22}(mesh).

Functions:
  read_plane(file)            -> plane dict (time, X, Z, dist, f[name] = 2D array [z, x])
  causal_mask(plane, t, ...)  -> points whose distance to the nearest outer boundary
                                 exceeds sqrt(2) t + ncell dx (D-026: 1+log gauge speed)
  l2(plane, name, mask)       -> axisymmetric L2 norm (weight 2 pi R) over the mask
  restrict(a, factor)         -> fine-grid scalar on the coarse cell centres (8-point
                                 midpoint Lagrange, axis by even parity)
  self_convergence(...)       -> order from three resolutions in a causal window
  compare(p2, p3, ...)        -> 2D Cartoon vs 3D Cartesian on y = 0 (T5)

Command line (T5): cartoon_m2d.py compare 2D.bin 3D.bin [--rmin 1] [--ncell 4]
"""

import argparse
import json
import math
import sys

import numpy as np

import bin_convert

SQRT2 = math.sqrt(2.0)

# physical axis -> (mesh index, sign) under D-017; mesh labels x, y, z = indices 1, 2, 3
_MAP2D = {"x": ("x", 1.0), "y": ("z", -1.0), "z": ("y", 1.0)}
_VECTORS = ("z4c_Gam", "z4c_beta", "con_M", "adm_beta")
_TENSORS = ("z4c_g", "z4c_A", "adm_g", "adm_K")
_SCALARS = ("z4c_chi", "z4c_Khat", "z4c_Theta", "z4c_alpha", "con_C", "con_H",
            "con_M", "con_Z", "adm_psi4", "adm_alpha")
# components that change sign under y -> -y (odd number of physical y indices)
ODD_Y = ("z4c_gxy", "z4c_gyz", "z4c_Axy", "z4c_Ayz", "z4c_Gamy", "z4c_betay")


def _sorted_pair(a, b):
    return "".join(sorted(a + b))


def _physical_names(var_names):
    """Physical component names available from the file's variables."""
    out = []
    for v in var_names:
        if v in _SCALARS:
            out.append(v)
            continue
        for base in _VECTORS:
            if v.startswith(base) and len(v) == len(base) + 1:
                out.append(v)
        for base in _TENSORS:
            if v.startswith(base) and len(v) == len(base) + 2:
                out.append(v)
    return out


def _mesh_to_phys_2d(name, mesh):
    """Physical component `name` from the mesh-frame dict `mesh` (2D Cartoon, D-017)."""
    if name in _SCALARS:
        return mesh[name]
    for base in _VECTORS:
        if name.startswith(base) and len(name) == len(base) + 1:
            m, s = _MAP2D[name[-1]]
            return s * mesh[base + m]
    for base in _TENSORS:
        if name.startswith(base) and len(name) == len(base) + 2:
            (ma, sa), (mb, sb) = _MAP2D[name[-2]], _MAP2D[name[-1]]
            return sa * sb * mesh[base + _sorted_pair(ma, mb)]
    raise KeyError(name)


def read_plane(fname):
    """Read one bin file. Returns dict(time, cycle, dim, X, Z, dx, dist, f) where X and Z
    are the cell-centre coordinates of the physical half-plane y = 0 (x > 0 for 3D: the
    3D slice is cut to x > 0), dist[k, i] is the distance of (X_i, 0, Z_k) to the
    nearest outer boundary of the run's domain, and f maps physical component names to
    arrays [k, i] (scalars and the physical-frame components listed in ODD_Y etc.)."""
    fd = bin_convert.read_binary(fname)
    hdr = fd["header"]
    coord = "cartesian"
    block = None
    for ln in hdr:
        if ln.startswith("<"):
            block = ln
            continue
        k, v = [s.strip() for s in ln.split("=", 1)]
        if block == "<mesh>" and k == "coord":
            coord = v
    n1, n2, n3 = fd["Nx1"], fd["Nx2"], fd["Nx3"]
    x1min, x1max = fd["x1min"], fd["x1max"]
    x2min, x2max = fd["x2min"], fd["x2max"]
    x3min, x3max = fd["x3min"], fd["x3max"]
    dx1 = (x1max - x1min) / n1
    if coord == "cylindrical_axisym":
        dim, nz, dz, zmin = 2, n2, (x2max - x2min) / n2, x2min
    else:
        assert n3 > 1, "3D input expected"
        dim, nz, dz, zmin = 3, n3, (x3max - x3min) / n3, x3min
        dy = (x2max - x2min) / n2
        # the slice must be the cell centred on y = 0
        jy = int((0.0 - x2min) / dy)
        assert abs(x2min + (jy + 0.5) * dy) <= 1e-12 * dy, \
            "3D run needs a cell centred on y = 0 (odd nx2, symmetric x2 range)"
    assert abs(dx1 - dz) <= 1e-12 * dx1, "uniform grid expected"
    mesh = {v: np.full((nz, n1), np.nan) for v in fd["var_names"]}
    for b in range(fd["n_mbs"]):
        g = fd["mb_geometry"][b]
        i0 = int(round((g[0] - x1min) / dx1))
        if dim == 2:
            k0 = int(round((g[2] - x2min) / dz))
        else:
            k0 = int(round((g[4] - x3min) / dz))
        for v in fd["var_names"]:
            a = fd["mb_data"][v][b]           # [n3_out, n2_out, n1_out]
            if dim == 2:
                blk = a[0, :, :]
            else:
                assert a.shape[1] == 1, "3D input must be sliced at y = 0"
                blk = a[:, 0, :]
            mesh[v][k0:k0 + blk.shape[0], i0:i0 + blk.shape[1]] = blk
    X = x1min + (np.arange(n1) + 0.5) * dx1
    Z = zmin + (np.arange(nz) + 0.5) * dz
    for v, a in mesh.items():
        assert np.all(np.isfinite(a)), f"{fname}: {v} has unfilled or non-finite cells"
    if dim == 2:
        f = {n: _mesh_to_phys_2d(n, mesh) for n in _physical_names(fd["var_names"])}
        zz, xx = np.meshgrid(Z, X, indexing="ij")
        dist = np.minimum(x1max - xx, np.minimum(x2max - zz, zz - x2min))
    else:
        keep = X > 0.0
        X = X[keep]
        f = {n: mesh[n][:, keep] for n in _physical_names(fd["var_names"])}
        zz, xx = np.meshgrid(Z, X, indexing="ij")
        dist = np.minimum.reduce([x1max - xx, xx - x1min, x3max - zz, zz - x3min,
                                  np.full_like(xx, min(x2max, -x2min))])
    return dict(time=fd["time"], cycle=fd["cycle"], dim=dim, X=X, Z=Z, dx=dx1,
                dist=dist, f=f, file=fname)


def radius(p):
    zz, xx = np.meshgrid(p["Z"], p["X"], indexing="ij")
    return np.sqrt(xx * xx + zz * zz)


def causal_mask(p, t=None, ncell=4, rmin=0.0, rmax=np.inf, speed=SQRT2):
    """Points outside the outer boundary's domain of influence at time t (D-026), with
    rmin <= r <= rmax."""
    t = p["time"] if t is None else t
    r = radius(p)
    return (p["dist"] > speed * t + ncell * p["dx"]) & (r >= rmin) & (r <= rmax)


def l2(p, name, mask, value=None):
    """Axisymmetric L2 norm sqrt(int f^2 dV / int dV) over the mask, dV = 2 pi R dR dz."""
    a = p["f"][name] if value is None else value
    w = np.broadcast_to(p["X"][None, :], a.shape)
    w = np.where(mask, w, 0.0)
    a = np.where(mask, a, 0.0)
    return float(np.sqrt(np.sum(w * a * a) / np.sum(w)))


_W8 = np.array([-5.0, 49.0, -245.0, 1225.0, 1225.0, -245.0, 49.0, -5.0]) / 2048.0


def _restrict_axis(a, factor, axis, parity):
    """Value at the midpoint of fine cells (factor i + factor/2 - 1, factor i +
    factor/2), i.e. at the coarse cell centres, with the 8-point Lagrange stencil.
    parity is None (no symmetry: NaN where the stencil leaves the grid) or +-1 for a
    reflection about the lower face (the Cartoon axis)."""
    a = np.moveaxis(a, axis, -1)
    n = a.shape[-1]
    pad = 3
    if parity is None:
        lo = np.full(a.shape[:-1] + (pad,), np.nan)
    else:
        lo = parity * a[..., :pad][..., ::-1]
    hi = np.full(a.shape[:-1] + (pad + 1,), np.nan)
    ap = np.concatenate([lo, a, hi], axis=-1)
    nc = n // factor
    left = factor * np.arange(nc) + factor // 2 - 1 + pad
    out = np.zeros(a.shape[:-1] + (nc,))
    for s, w in enumerate(_W8):
        out = out + w * ap[..., left - 3 + s]
    return np.moveaxis(out, -1, axis)


def restrict(a, factor, axis_parity=1.0):
    """Fine-grid scalar [k, i] onto the coarse cell centres (factor = 2^m), 8th order."""
    assert factor >= 2 and factor & (factor - 1) == 0
    a = _restrict_axis(a, factor, 1, axis_parity)   # R (x1): the axis is a mirror
    return _restrict_axis(a, factor, 0, None)        # z


def self_convergence(p1, p2, p4, name, mask):
    """Order from runs at dx, dx/2, dx/4 (planes p1, p2, p4 at the same time):
    log2(||u1 - u2|| / ||u2 - u4||) over the mask (on p1's grid). Scalars only."""
    assert name in _SCALARS
    u1 = p1["f"][name]
    u2 = restrict(p2["f"][name], 2)
    u4 = restrict(p4["f"][name], 4)
    m = mask & np.isfinite(u2) & np.isfinite(u4)
    e12, e24 = l2(p1, name, m, u1 - u2), l2(p1, name, m, u2 - u4)
    return dict(e12=e12, e24=e24, order=math.log2(e12 / e24) if e24 > 0 else np.inf)


def compare(p2, p3, rmin=1.0, ncell=4, sign_floor=1e-3, abs_floor=1e-10, nbins=8):
    """T5: 2D Cartoon (p2) vs 3D Cartesian (p3) on the y = 0 half-plane at one time.
    Points: outside both domains of influence (causal_mask on each). The difference
    norms and the odd-y sign gate exclude r < rmin (the puncture, where 2D and 3D differ
    at O(1) in the cells next to it); the r profile and sign_all (reported only) do not.
    Returns dict(time, vars -> {max3, max_abs, rel_max, l2_abs, profile}, sign / sign_all
    -> {var: n_bad, n_checked, max3, bad_rmax}, sign_ok)."""
    assert p2["dim"] == 2 and p3["dim"] == 3
    assert abs(p2["time"] - p3["time"]) <= 1e-9 * max(1.0, p2["time"]), \
        (p2["time"], p3["time"])
    assert p2["X"].shape == p3["X"].shape and np.allclose(p2["X"], p3["X"]), "x grids"
    assert p2["Z"].shape == p3["Z"].shape and np.allclose(p2["Z"], p3["Z"]), "z grids"
    window = causal_mask(p2, ncell=ncell) & causal_mask(p3, ncell=ncell)
    order_m = window & (radius(p2) >= rmin)
    assert np.any(order_m), "empty comparison window"
    r = radius(p2)
    edges = np.linspace(0.0, float(r[window].max()) + 1e-12, nbins + 1)
    res = dict(time=p2["time"], dx=p2["dx"], n_window=int(window.sum()),
               n_order=int(order_m.sum()), rmax_window=float(r[window].max()),
               vars={}, sign={}, sign_all={})
    for name in sorted(set(p2["f"]) & set(p3["f"])):
        a, b = p2["f"][name], p3["f"][name]
        d = a - b
        scale = float(np.abs(b[order_m]).max())
        prof = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = window & (r >= lo) & (r < hi)
            prof.append(float(np.abs(d[sel]).max()) if np.any(sel) else None)
        res["vars"][name] = dict(
            max3=scale, max_abs=float(np.abs(d[order_m]).max()),
            rel_max=float(np.abs(d[order_m]).max()) / scale if scale > 0 else None,
            l2_abs=l2(p2, name, order_m, d), profile=prof)
        if name in ODD_Y:
            mx = float(np.abs(b[window]).max())
            for key, region in (("sign", order_m), ("sign_all", window)):
                if mx <= abs_floor:
                    res[key][name] = dict(n_bad=0, n_checked=0, max3=mx, bad_rmax=None)
                    continue
                sel = region & (np.abs(b) > sign_floor * mx)
                bad = sel & (np.sign(a) != np.sign(b))
                res[key][name] = dict(
                    n_bad=int(bad.sum()), n_checked=int(sel.sum()), max3=mx,
                    bad_rmax=float(r[bad].max()) if bad.any() else None)
    res["profile_edges"] = edges.tolist()
    res["sign_ok"] = all(v["n_bad"] == 0 for v in res["sign"].values())
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare", help="T5: 2D Cartoon vs 3D Cartesian on y = 0")
    c.add_argument("bin2d")
    c.add_argument("bin3d")
    c.add_argument("--rmin", type=float, default=1.0)
    c.add_argument("--ncell", type=int, default=4)
    s = sub.add_parser("selfconv", help="order from three 2D runs at dx, dx/2, dx/4")
    s.add_argument("bins", nargs=3)
    s.add_argument("--var", default="z4c_alpha")
    s.add_argument("--rmin", type=float, default=0.0)
    s.add_argument("--rmax", type=float, default=np.inf)
    s.add_argument("--ncell", type=int, default=4)
    n = sub.add_parser("norms", help="causal-window L2 norms of 2D runs")
    n.add_argument("bins", nargs="+")
    n.add_argument("--vars", nargs="+", default=["con_H", "con_M"])
    n.add_argument("--rmin", type=float, default=0.0)
    n.add_argument("--rmax", type=float, default=np.inf)
    n.add_argument("--ncell", type=int, default=4)
    a = ap.parse_args(argv)
    if a.cmd == "compare":
        out = compare(read_plane(a.bin2d), read_plane(a.bin3d), rmin=a.rmin,
                      ncell=a.ncell)
    elif a.cmd == "selfconv":
        p = [read_plane(f) for f in a.bins]
        m = causal_mask(p[0], ncell=a.ncell, rmin=a.rmin, rmax=a.rmax)
        out = dict(time=p[0]["time"], dx=[q["dx"] for q in p], var=a.var,
                   **self_convergence(*p, a.var, m))
    else:
        out = []
        for f in a.bins:
            p = read_plane(f)
            m = causal_mask(p, ncell=a.ncell, rmin=a.rmin, rmax=a.rmax)
            out.append(dict(file=f, time=p["time"], dx=p["dx"],
                            **{v: l2(p, v, m) for v in a.vars}))
    json.dump(out, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
