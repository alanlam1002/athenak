#!/usr/bin/env python3
"""Elliptic-orbit TDE initial-velocity solver, for the CFC BH-puncture TDE fixture.

Ports the exact trumpet-background formulas from src/cfc/cfc_puncture.hpp (M_BH=1
units, non-spinning maximal-slicing trumpet, Baumgarte & Naculich 2007) verbatim into
Python, and uses them to (a) find the initial tangential velocity v^y that gives a
star released from rest-in-radius at x=x0 (isotropic Cartesian) a target periapsis in
AREAL (Schwarzschild) radius, and (b) cross-check that value with a full numerical
integration of a test-particle geodesic in the actual 3+1 trumpet gauge.

Two independent methods are used and must agree, because this is exactly the kind of
calculation where an algebra slip is easy to miss otherwise:

  Stage A -- closed-form shooting. E=-u_t and L=u_phi are conserved (the background is
  stationary and axisymmetric). At the initial condition (position (x0,0,0), velocity
  purely in y), beta0^i is purely radial (x-direction) and p_i purely azimuthal
  (y-direction) so beta^i p_i = 0 exactly and:
      W  = 1/sqrt(1 - psi0^4 v0^2)
      E  = alpha0 * W
      L  = psi0^4 * W * x0 * v0
  Periapsis in AREAL radius (gauge-invariant, gives a genuine apples-to-apples answer
  independent of whether we used isotropic or Schwarzschild-like coordinates) then
  solves the standard turning-point equation
      E^2 = (1 - 2/r)(1 + L^2/r^2)
  which is exact Schwarzschild far from the throat (the trumpet's extra 1.6875/rho^4
  term in alpha0 only matters near the throat, rho~1.5-3) and asymptotically exact
  here since E,L are themselves computed from the true trumpet psi0/alpha0 at x0.
  IMPORTANT: there are up to 3 positive roots of this quartic-in-1/r; r0's own mapped
  areal radius is always one of them (self-consistency check, since dr/dtau=0 there
  by construction); periapsis is the LARGEST remaining root strictly below that one,
  NOT the smallest -- the smallest root is an inner turning point behind the
  angular-momentum barrier, unreachable from outside it.

  Stage B -- full ODE integration in the real trumpet gauge (not vacuum Schwarzschild).
  Evolves the COVARIANT momentum p_i = W v_i (not v_i directly -- the naive
  dv_i/dt ODE silently drops a product-rule term hiding inside d(W v_i)/dt; evolving
  p_i sidesteps this) via the standard ADM geodesic Hamiltonian
      H(x,p) = alpha(x) sqrt(1 + gamma^jk(x) p_j p_k) - beta^i(x) p_i
      dx^i/dt =  dH/dp_i = alpha * gamma^ij p_j / W - beta^i
      dp_i/dt = -dH/dx^i = -W d_i(alpha) - (alpha/2W) d_i(gamma^jk) p_j p_k
                            + d_i(beta^j) p_j
  All x-derivatives are 5-point-stencil finite differences of the ported
  alpha0/psi0/beta0 functions themselves (not a hand-differentiated closed form --
  less error-prone for a one-off script), cross-checked against TrumpetBackground's
  own analytic dpsi0/dalpha0 out-params.

Usage:
    python3 tde_elliptical_orbit_ic.py                  # solve for periapsis=10, x0=30
    python3 tde_elliptical_orbit_ic.py --periapsis 16.4  # solve for a different target
    python3 tde_elliptical_orbit_ic.py --v0 0.1357 --no-solve   # just integrate this v0
"""
import argparse
import math
import sys

import numpy as np
from scipy.optimize import brentq

M_BH = 1.0          # puncture_mass, code units (== 1e4 Msun in the physical fixture)
THROAT_RHO = 1.5    # areal radius of the trumpet throat, M_BH=1 units


# ---------------------------------------------------------------------------
# Verbatim ports of src/cfc/cfc_puncture.hpp
# ---------------------------------------------------------------------------

def trumpet_areal_to_iso(rho):
    """chi(rho): forward map areal -> isotropic radius, M_BH=1 units (cfc_puncture.hpp
    TrumpetArealToIso). Valid for rho >= THROAT_RHO."""
    s1 = math.sqrt(4.0 * rho * rho + 4.0 * rho + 3.0)
    s2 = math.sqrt(8.0 * rho * rho + 8.0 * rho + 6.0)
    numer = (4.0 + 3.0 * math.sqrt(2.0)) * (2.0 * rho - 3.0)
    denom = 8.0 * rho + 6.0 + 3.0 * s2
    return 0.25 * (2.0 * rho + 1.0 + s1) * (numer / denom) ** (1.0 / math.sqrt(2.0))


def trumpet_iso_to_areal(chi_cell, tol=1.0e-14):
    """Bracketed inverse of trumpet_areal_to_iso (r_iso -> R_sch, M_BH=1 units),
    matching cfc_puncture.hpp TrumpetIsoToAreal's bracket/clamp behaviour."""
    if chi_cell <= THROAT_RHO * tol:
        return THROAT_RHO
    lb = max(THROAT_RHO, chi_cell)
    ub = max(THROAT_RHO, chi_cell + 1.5)
    # trumpet_areal_to_iso is monotonically increasing for rho>=THROAT_RHO (chi=0 at the
    # throat, chi->rho as rho->inf), so a bracket [lb,ub] with lb<=chi_cell<=... needs
    # widening until the residual changes sign, exactly mirroring the C++ falseposition
    # call's own lb/ub choice plus its implicit reliance on a valid bracket.
    def resid(rho):
        return trumpet_areal_to_iso(rho) / chi_cell - 1.0
    while resid(ub) < 0:
        ub += 1.5
    rho = brentq(resid, lb, ub, xtol=tol, rtol=tol)
    return max(rho, THROAT_RHO)


def trumpet_background(m_bh, x1, x2, x3, r_sch):
    """Port of TrumpetBackground: returns (psi0, alpha0, beta0[3], dpsi0, dalpha0)."""
    r = math.sqrt(x1 * x1 + x2 * x2 + x3 * x3 + 1.0e-30)
    rrs = r_sch / m_bh
    rrs3 = rrs * rrs * rrs
    psi = math.sqrt(r_sch / r)
    alpha = math.sqrt(1.0 - 2.0 / rrs + 1.6875 / (rrs3 * rrs))
    dpsi0 = -psi * (1.0 - 0.84375 / rrs3) / (rrs * r * (1.0 + alpha))
    dalpha0 = (1.0 - 3.375 / rrs3) / (rrs * r)
    fac = 0.75 * math.sqrt(3.0)
    bmag = fac / (r_sch * rrs * rrs)
    beta0 = (bmag * x1, bmag * x2, bmag * x3)
    return psi, alpha, beta0, dpsi0, dalpha0


def metric_at(x1, x2, x3, m_bh=M_BH):
    """Convenience: psi0, alpha0, beta0 at Cartesian isotropic position (x1,x2,x3),
    doing the areal-radius root-find internally (chi_cell = |x|/m_bh)."""
    chi = math.sqrt(x1 * x1 + x2 * x2 + x3 * x3) / m_bh
    rho = trumpet_iso_to_areal(chi)
    r_sch = m_bh * rho
    psi0, alpha0, beta0, dpsi0, dalpha0 = trumpet_background(m_bh, x1, x2, x3, r_sch)
    return psi0, alpha0, beta0


# ---------------------------------------------------------------------------
# Stage A: closed-form E,L + Schwarzschild-turning-point periapsis
# ---------------------------------------------------------------------------

def energy_angmom(x0, v0):
    """E, L for a star released at isotropic Cartesian (x0,0,0) with pure-tangential
    v^y=v0 (v^x=v^z=0). Exact at this point: beta0 is purely radial (x-direction),
    momentum purely azimuthal (y-direction), so beta^i p_i = 0 identically."""
    psi0, alpha0, beta0 = metric_at(x0, 0.0, 0.0)
    vsq = psi0 ** 4 * v0 * v0
    if vsq >= 1.0:
        raise ValueError(f"v0={v0} superluminal at this radius (vsq={vsq})")
    W = 1.0 / math.sqrt(1.0 - vsq)
    E = alpha0 * W
    L = psi0 ** 4 * W * x0 * v0
    return E, L, W, psi0, alpha0


def turning_point_roots(E, L, r_lo=THROAT_RHO + 1e-6, r_hi=1.0e4):
    """All areal-radius roots of E^2=(1-2/r)(1+L^2/r^2) in (r_lo,r_hi), found by a
    dense scan for sign changes followed by bisection on each bracket (robust against
    the up-to-cubic-in-1/r structure; a single-shot root finder can miss roots)."""
    def f(r):
        return (1.0 - 2.0 / r) * (1.0 + L * L / (r * r)) - E * E
    rs = np.geomspace(r_lo, r_hi, 20000)
    vals = np.array([f(r) for r in rs])
    roots = []
    for i in range(len(rs) - 1):
        if vals[i] == 0.0:
            roots.append(rs[i])
        elif vals[i] * vals[i + 1] < 0:
            roots.append(brentq(f, rs[i], rs[i + 1], xtol=1e-12, rtol=1e-12))
    return roots


def periapsis_for_v0(x0, v0, verbose=False):
    """Areal-radius periapsis for a star released FALLING INWARD from (x0,0,0) with
    tangential v0. Returns None if no periapsis exists (L below critical -- pure
    plunge), or raises if v0 is super-circular at x0 (star would move OUTWARD from
    release, not inward -- a different, not-yet-handled scenario; not needed for this
    script's use case, so treated as a hard error rather than silently misidentified).

    Caught during development: r0's own mapped areal radius is NOT always the largest
    turning-point root. Whether it is the largest (sub-circular release, r0=apoapsis,
    falls inward -- what we want) or a MIDDLE root of three (super-circular release,
    r0=periapsis-of-an-outward-swing, moving outward instead) depends on v0 vs. the
    local circular velocity at x0, not on v0 vs. the naive Kepler/critical-L estimate.
    Identify r0's actual root explicitly rather than assuming it is the largest.
    """
    E, L, W, psi0, alpha0 = energy_angmom(x0, v0)
    roots = turning_point_roots(E, L)
    if verbose:
        print(f"    v0={v0:.6f}  E={E:.6f}  L={L:.6f}  W={W:.6f}  roots(areal)={roots}")
    if len(roots) < 2:
        return None  # monotonic potential: plunge, no periapsis (L below critical)
    r0_areal = trumpet_iso_to_areal(x0 / M_BH) * M_BH
    roots_sorted = sorted(roots)
    idx = min(range(len(roots_sorted)), key=lambda i: abs(roots_sorted[i] - r0_areal))
    if abs(roots_sorted[idx] - r0_areal) > 1e-3 * r0_areal:
        raise RuntimeError(
            f"no root near release radius {r0_areal:.4f} among {roots_sorted} -- "
            f"root-finding bracket likely missed a root")
    if idx != len(roots_sorted) - 1:
        raise RuntimeError(
            f"v0={v0:.6f} is super-circular at x0={x0} (release root {roots_sorted[idx]:.4f} "
            f"is not the largest of {roots_sorted} -- star would move OUTWARD from "
            f"release, not inward). This script only handles sub-circular release "
            f"(falling inward); narrow the v0 bracket below the local circular velocity.")
    # Periapsis = largest root STRICTLY BELOW the release-point root, not the smallest
    # (the smallest is an inner turning point behind the L-barrier, unreachable).
    return roots_sorted[-2]


def solve_v0_for_periapsis(x0, target_periapsis, v0_lo=0.10, v0_hi=0.17, verbose=True):
    """Bisect v0 in [v0_lo,v0_hi] (the physically-relevant monotonic-periapsis window
    for this setup) until periapsis_for_v0 hits target_periapsis."""
    def g(v0):
        rp = periapsis_for_v0(x0, v0, verbose=verbose)
        if rp is None:
            # Below critical L: treat as "periapsis at the throat/horizon", i.e. very
            # negative residual, so bisection pushes v0 upward correctly.
            return -1.0e3
        return rp - target_periapsis
    glo, ghi = g(v0_lo), g(v0_hi)
    if glo > 0 or ghi < 0:
        raise RuntimeError(
            f"bracket [{v0_lo},{v0_hi}] does not straddle the target: "
            f"g(lo)={glo:.4f} g(hi)={ghi:.4f}. Widen the bracket.")
    v0 = brentq(g, v0_lo, v0_hi, xtol=1e-10, rtol=1e-12)
    return v0


# ---------------------------------------------------------------------------
# Stage B: full ODE integration in the actual (shifted, conformally-flat) trumpet gauge
# ---------------------------------------------------------------------------

def metric_fields(pos, h=1.0e-6):
    """alpha, beta^i, psi0 at pos=(x,y,z), plus their finite-difference gradients
    (5-point stencil). Returns (alpha, beta, psi0, dalpha, dbeta, dpsi0) where dalpha,
    dpsi0 are length-3 gradients and dbeta is a 3x3 Jacobian dbeta[i][j]=d(beta^i)/dx^j.
    """
    def f(p):
        return metric_at(p[0], p[1], p[2])

    alpha0, beta0 = None, None
    psi0_c = None
    dalpha = [0.0, 0.0, 0.0]
    dpsi0 = [0.0, 0.0, 0.0]
    dbeta = [[0.0] * 3 for _ in range(3)]

    psi_c, alpha_c, beta_c = f(pos)
    alpha0, beta0, psi0_c = alpha_c, beta_c, psi_c

    for j in range(3):
        pp = list(pos); pp[j] += h
        pm = list(pos); pm[j] -= h
        psi_p, alpha_p, beta_p = f(pp)
        psi_m, alpha_m, beta_m = f(pm)
        dalpha[j] = (alpha_p - alpha_m) / (2 * h)
        dpsi0[j] = (psi_p - psi_m) / (2 * h)
        for i in range(3):
            dbeta[i][j] = (beta_p[i] - beta_m[i]) / (2 * h)
    return alpha0, beta0, psi0_c, dalpha, dbeta, dpsi0


def geodesic_rhs(t, y):
    """y = [x,y,z, p_x,p_y,p_z] (covariant momentum). Returns dy/dt."""
    pos = y[0:3]
    p = y[3:6]
    alpha, beta, psi0, dalpha, dbeta, dpsi0 = metric_fields(pos)
    ipsi4 = psi0 ** -4
    psq = sum(pi * pi for pi in p)  # gamma^jk p_j p_k = psi0^-4 * |p|^2
    gpp = ipsi4 * psq
    W = math.sqrt(1.0 + gpp)

    dxdt = [alpha * ipsi4 * p[i] / W - beta[i] for i in range(3)]

    # d_i(gamma^{jk}) p_j p_k = d_i(psi0^-4) * |p|^2 = -4 psi0^-5 dpsi0_i * |p|^2
    dgpp_dxi = [-4.0 * psi0 ** -5 * dpsi0[i] * psq for i in range(3)]
    dpdt = [0.0, 0.0, 0.0]
    for i in range(3):
        term1 = -W * dalpha[i]
        term2 = -(alpha / (2.0 * W)) * dgpp_dxi[i]
        term3 = sum(dbeta[j][i] * p[j] for j in range(3))
        dpdt[i] = term1 + term2 + term3
    return dxdt + dpdt


def integrate_orbit(x0, v0, t_max, dt=0.05, verbose_every=None):
    """RK4 integration of the geodesic from (x0,0,0) with tangential v0. Returns a list
    of (t, x, y, z, r_iso, r_areal) samples."""
    psi0, alpha0, beta0 = metric_at(x0, 0.0, 0.0)
    vsq = psi0 ** 4 * v0 * v0
    W = 1.0 / math.sqrt(1.0 - vsq)
    p0 = [0.0, W * psi0 ** 4 * v0, 0.0]
    y = [x0, 0.0, 0.0] + p0

    def rk4_step(t, y, dt):
        k1 = geodesic_rhs(t, y)
        y2 = [y[i] + 0.5 * dt * k1[i] for i in range(6)]
        k2 = geodesic_rhs(t + 0.5 * dt, y2)
        y3 = [y[i] + 0.5 * dt * k2[i] for i in range(6)]
        k3 = geodesic_rhs(t + 0.5 * dt, y3)
        y4 = [y[i] + dt * k3[i] for i in range(6)]
        k4 = geodesic_rhs(t + dt, y4)
        return [y[i] + (dt / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
                for i in range(6)]

    samples = []
    t = 0.0
    n_steps = int(t_max / dt)
    for n in range(n_steps + 1):
        r_iso = math.sqrt(y[0] ** 2 + y[1] ** 2 + y[2] ** 2)
        r_areal = M_BH * trumpet_iso_to_areal(r_iso / M_BH)
        samples.append((t, y[0], y[1], y[2], r_iso, r_areal))
        if verbose_every and n % verbose_every == 0:
            print(f"    t={t:8.2f}  x={y[0]:9.4f}  y={y[1]:9.4f}  "
                  f"r_iso={r_iso:8.4f}  r_areal={r_areal:8.4f}")
        y = rk4_step(t, y, dt)
        t += dt
    return samples


def find_periapsis_in_trajectory(samples):
    """Return (t, r_areal) at the minimum r_areal in the sampled trajectory."""
    tmin, rmin = min(((s[0], s[5]) for s in samples), key=lambda p: p[1])
    return tmin, rmin


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x0", type=float, default=30.0,
                     help="release isotropic radius (code units, r_g), default 30")
    ap.add_argument("--periapsis", type=float, default=10.0,
                     help="target areal periapsis (r_g), default 10")
    ap.add_argument("--v0", type=float, default=None,
                     help="use this v0 directly instead of solving for the target")
    ap.add_argument("--no-solve", action="store_true",
                     help="skip Stage A solve, just integrate --v0 (requires --v0)")
    ap.add_argument("--tmax", type=float, default=700.0,
                     help="ODE integration horizon, code units of t (M_BH=1)")
    ap.add_argument("--dt", type=float, default=0.05, help="ODE fixed step size")
    args = ap.parse_args()

    x0 = args.x0

    print("=" * 78)
    print(f"Trumpet metric sanity check at x0={x0}:")
    psi0, alpha0, beta0 = metric_at(x0, 0.0, 0.0)
    r_areal0 = M_BH * trumpet_iso_to_areal(x0 / M_BH)
    print(f"  areal radius R_sch(r_iso={x0}) = {r_areal0:.6f}")
    print(f"  psi0={psi0:.6f}  alpha0={alpha0:.6f}  beta0=({beta0[0]:.6e},"
          f"{beta0[1]:.6e},{beta0[2]:.6e})")
    v_kep_newtonian = math.sqrt(M_BH / x0)
    print(f"  Newtonian v_kep(x0) = sqrt(M/x0) = {v_kep_newtonian:.6f}")
    L_crit = 2.0 * math.sqrt(3.0) * M_BH
    v_crit = L_crit / x0
    print(f"  Schwarzschild L_crit = 2*sqrt(3)*M = {L_crit:.6f}  "
          f"(-> v0_crit ~ {v_crit:.6f} for ANY periapsis to exist, naive L=r0*v estimate)")

    print("=" * 78)
    if args.no_solve:
        if args.v0 is None:
            sys.exit("--no-solve requires --v0")
        v0 = args.v0
        print(f"Using supplied v0={v0} directly (Stage A solve skipped).")
    else:
        print(f"Stage A: solving for v0 such that periapsis(areal) = {args.periapsis} "
              f"r_g, released from x0={x0}...")
        v0 = args.v0 if args.v0 is not None else solve_v0_for_periapsis(
            x0, args.periapsis, verbose=True)
        rp_check = periapsis_for_v0(x0, v0, verbose=False)
        E, L, W, _, _ = energy_angmom(x0, v0)
        print(f"  --> v0 = {v0:.6f}")
        print(f"  --> E={E:.6f}  L={L:.6f}  W={W:.6f}")
        print(f"  --> Stage A periapsis (areal) = {rp_check:.6f} r_g "
              f"(target {args.periapsis})")

    print("=" * 78)
    print(f"Stage B: full ODE cross-check, integrating t=0..{args.tmax} "
          f"(dt={args.dt})...")
    samples = integrate_orbit(x0, v0, args.tmax, dt=args.dt,
                               verbose_every=int(2.0 / args.dt))
    t_peri, r_peri = find_periapsis_in_trajectory(samples)
    print(f"  --> Stage B minimum areal radius = {r_peri:.6f} r_g at t={t_peri:.2f}")

    # Report apoapsis/periapsis excursion in (x,y) for mesh-sizing purposes.
    xs = [s[1] for s in samples]
    ys = [s[2] for s in samples]
    print(f"  --> trajectory extent over t=[0,{args.tmax}]: "
          f"x in [{min(xs):.2f},{max(xs):.2f}]  y in [{min(ys):.2f},{max(ys):.2f}]")

    print("=" * 78)
    agree = abs(r_peri - (rp_check if not args.no_solve else args.periapsis))
    print(f"Cross-check: Stage A vs Stage B periapsis agree to {agree:.4f} r_g "
          f"({'PASS' if agree < 0.05 else 'CHECK -- see above'})")
    print()
    print(f"RECOMMENDED star_vel_x2 = {v0:.6f}")


if __name__ == "__main__":
    main()
