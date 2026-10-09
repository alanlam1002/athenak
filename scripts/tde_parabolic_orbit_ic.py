#!/usr/bin/env python3
"""Parabolic-orbit TDE initial velocity for the CFC BH-puncture TDE fixture.

Companion to tde_elliptical_orbit_ic.py, whose verbatim ports of cfc_puncture.hpp
(trumpet psi0/alpha0/beta0 and the iso<->areal map) and whose Stage-B geodesic
integrator are imported unchanged.

WHAT IS DIFFERENT FROM THE ELLIPTIC CASE. The elliptic setup releases the star at
apocentre, so its velocity is purely tangential and beta^i p_i = 0 at release. A
parabolic orbit has no apocentre: released at any finite x0 the star is already
falling in, so the velocity has a RADIAL component and the shift term enters the
energy. Everything below uses the full Killing energy

    E = -u_t = alpha W - beta^i p_i ,   W = sqrt(1 + psi^-4 |p|^2) ,   p_i = u_i

(the same H(x,p) the elliptic script's Stage B integrates), and L = x p_y - y p_x.

Stage A (closed form). Parabolic means marginally bound: E = 1 exactly. For E = 1 the
Schwarzschild turning-point condition E^2 = (1-2/r)(1+L^2/r^2) at areal periapsis r_p
gives L^2 = 2 r_p^2 / (r_p - 2) exactly (r_p = 10 -> L = 5; the other root is the
unreachable inner turning point r = 2.5). At release (x0,0,0): p_y = L/x0, and p_x < 0
(infalling) is the root of E(p_x) = 1. Then v^i = psi0^-4 p_i / W -- the contravariant
Eulerian velocity, which is what <problem> star_vel_x1/x2 mean in dyngr_tov.cpp (it
builds u^i = W v^i). The elliptic run's t=0 dump confirmed that this convention, after
CFC::InitializeMetric, reproduces the designed E and L to 1e-4.

Stage B (independent check). Integrate the geodesic in the real trumpet gauge from the
Stage-A momentum and confirm the minimum areal radius.

KNOWN LIMITATION, measured (NANCASCADE_HANDOFF / parabolic plan): in the elliptic
production run the star's L decayed ~5% over 250 M BEFORE disruption, so the achieved
periapsis (~6.5-7 r_g) fell well short of the designed 10. This script computes the
test-particle IC correctly; it cannot compensate for a drag the simulation adds.
Verify the achieved periapsis in a short calibration run before trusting it.
"""
import argparse
import math

import numpy as np
from scipy.optimize import brentq

import tde_elliptical_orbit_ic as ic


def parabolic_L(rp):
    """Exact E=1 angular momentum for areal periapsis rp (Schwarzschild, M=1)."""
    if rp <= 4.0:
        raise ValueError("rp<=4: an E=1 orbit with this periapsis is a plunge/whirl")
    return math.sqrt(2.0 * rp * rp / (rp - 2.0))


def energy(x0, px, py):
    psi0, alpha0, beta0 = ic.metric_at(x0, 0.0, 0.0)
    W = math.sqrt(1.0 + psi0 ** -4 * (px * px + py * py))
    return alpha0 * W - (beta0[0] * px + beta0[1] * py), W, psi0, alpha0, beta0


def solve_ic(x0, rp, E_target=1.0):
    L = parabolic_L(rp)
    py = L / x0
    f = lambda px: energy(x0, px, py)[0] - E_target
    # infalling root: p_x < 0. Bracket from 0 down until the sign flips.
    lo = -1e-6
    while f(lo) < 0:
        lo *= 2.0
        if lo < -10:
            raise RuntimeError("no infalling root: x0 too close for this E,L")
    px = brentq(f, lo, 0.0, xtol=1e-15, rtol=1e-15)
    E, W, psi0, alpha0, beta0 = energy(x0, px, py)
    vx, vy = psi0 ** -4 * px / W, psi0 ** -4 * py / W
    return dict(L=L, E=E, px=px, py=py, W=W, psi0=psi0, alpha0=alpha0,
                beta0=beta0, vx=vx, vy=vy)


def integrate(x0, px, py, t_max, dt):
    y = [x0, 0.0, 0.0, px, py, 0.0]
    out = []
    t = 0.0
    for n in range(int(t_max / dt) + 1):
        r_iso = math.sqrt(y[0] ** 2 + y[1] ** 2 + y[2] ** 2)
        out.append((t, y[0], y[1], r_iso, ic.trumpet_iso_to_areal(r_iso)))
        k1 = ic.geodesic_rhs(t, y)
        k2 = ic.geodesic_rhs(t, [y[i] + 0.5 * dt * k1[i] for i in range(6)])
        k3 = ic.geodesic_rhs(t, [y[i] + 0.5 * dt * k2[i] for i in range(6)])
        k4 = ic.geodesic_rhs(t, [y[i] + dt * k3[i] for i in range(6)])
        y = [y[i] + dt / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(6)]
        t += dt
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x0", type=float, default=30.0, help="release isotropic radius")
    ap.add_argument("--periapsis", type=float, default=10.0, help="target areal r_p")
    ap.add_argument("--E", type=float, default=1.0, help="Killing energy (1=parabolic)")
    ap.add_argument("--tmax", type=float, default=400.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--box", type=float, default=64.0, help="domain half-width")
    a = ap.parse_args()

    s = solve_ic(a.x0, a.periapsis, a.E)
    print("=" * 78)
    print(f"Stage A: E={s['E']:.12f}  L={s['L']:.6f}  at x0={a.x0} "
          f"(R_areal={ic.trumpet_iso_to_areal(a.x0):.4f})")
    print(f"  psi0={s['psi0']:.6f} alpha0={s['alpha0']:.6f} beta0^x={s['beta0'][0]:.6e}")
    print(f"  p_x={s['px']:.8f}  p_y={s['py']:.8f}  W={s['W']:.8f}")
    print(f"  shift term beta^i p_i = {s['beta0'][0]*s['px']:.3e}  "
          f"(zero in the elliptic case; NOT negligible here)")
    tr = ic.turning_point_roots(s['E'], s['L'])
    print(f"  turning points (areal): {[round(r, 6) for r in tr]}")

    o = integrate(a.x0, s['px'], s['py'], a.tmax, a.dt)
    k = int(np.argmin(o[:, 4]))
    tp, Rp = o[k, 0], o[k, 4]
    print("=" * 78)
    print(f"Stage B: min R_areal = {Rp:.5f} at t = {tp:.2f}  "
          f"(Stage A {a.periapsis}; diff {abs(Rp-a.periapsis):.2e} "
          f"{'PASS' if abs(Rp-a.periapsis) < 0.05 else 'CHECK'})")
    print(f"  periapsis position (x,y) = ({o[k,1]:.3f},{o[k,2]:.3f}), r_iso={o[k,3]:.4f}")
    for dtp in (50, 100, 150, 200, 250):
        j = int(np.argmin(abs(o[:, 0] - (tp + dtp))))
        if o[j, 0] < tp + dtp - 1:
            break
        cheb = max(abs(o[j, 1]), abs(o[j, 2]))
        print(f"  t_peri+{dtp:3d} (t={o[j,0]:6.1f}): r_iso={o[j,3]:6.2f}  "
              f"(x,y)=({o[j,1]:7.2f},{o[j,2]:7.2f})  chebyshev={cheb:5.1f}/{a.box}")
    out = np.where(np.maximum(abs(o[:, 1]), abs(o[:, 2])) > a.box)[0]
    if len(out):
        print(f"  star CENTRE leaves the |x|,|y|<{a.box} box at t={o[out[0],0]:.1f} "
              f"(t_peri+{o[out[0],0]-tp:.1f})")
    print("=" * 78)
    print(f"RECOMMENDED  star_vel_x1 = {s['vx']:.8f}")
    print(f"             star_vel_x2 = {s['vy']:.8f}")
    print(f"             star_vel_x3 = 0.0   star_center = ({a.x0}, 0, 0)")


if __name__ == "__main__":
    main()
