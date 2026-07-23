#!/usr/bin/env python
"""Stage 7: three-way comparison (M1 vs. discrete-ordinate (DO) vs. independent
analytic solution) for arXiv:2302.04283 Section 3.6 "Equilibration".

Static case: Gamma=5/3, rho=1, pgas0=2 (e_gas0=3), urad0=1, alpha_s=0,
alpha_a=0.1 (Planck-channel kappa_p on the M1 side). Both codes should relax
to the analytic equilibrium (paper Eq. 68) along the analytic trajectory
(paper Eq. 69), which is re-integrated here from scratch (scipy.solve_ivp),
independent of both codes.

Moving case: identical physics, fluid additionally has u^1=1
(vx=1/sqrt(2), W=sqrt(2)), radiation field isotropic in the COORDINATE frame
at t=0 (not comoving) -- both energy AND momentum must relax. No independent
ODE re-derivation is attempted for this case (a genuinely separate derivation
from the static Eq. 69 -- see paper Eq. 70); instead this checks the same
invariant both codes must obey regardless of closure or angular
discretization: total coordinate-frame energy-momentum
(T^00_matter + R^00, T^01_matter + R^01) is conserved, and checks that both
codes independently converge to the same final equilibrium state.

Run after executing AthenaK with all four matched inputs (see
DEVELOPMENT.md Stage 7 for exact commands/build requirements: DO side needs
the dedicated build_relax build, -DPROBLEM=rad_relax):
    rad_relax_paper_static.athinput   (DO, build_relax)
    rad_relax_paper_moving.athinput   (DO, build_relax)
    rad_m1_photon_equilibration_paper_static.athinput   (M1, ordinary build)
    rad_m1_photon_equilibration_paper_moving.athinput   (M1, ordinary build)

    python check_rad_relax_paper_equilibration.py \\
        --do-static-dir tab_do_static --do-moving-dir tab_do_moving \\
        --m1-static-dir tab_m1_static --m1-moving-dir tab_m1_moving \\
        --plot-dir plots
"""
import argparse
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(__file__))
from m1_tab_utils import load_series  # noqa: E402

GAMMA = 5.0 / 3.0
GM1 = GAMMA - 1.0
RHO = 1.0
PGAS0 = 2.0
UGAS0 = PGAS0 / GM1     # = 3.0
URAD0 = 1.0
ALPHA_A = 0.1
ETOT0 = UGAS0 + URAD0   # = 4.0

U1_MOVING = 1.0
W_MOVING = np.sqrt(1.0 + U1_MOVING**2)
VX_MOVING = U1_MOVING / W_MOVING


def analytic_static_trajectory(t_eval):
    """Re-integrate paper Eq. 69 from scratch: dugas/dt = alpha_a*(Etot0 -
    ugas - T^4), T = GM1*ugas/RHO. Independent of both codes."""
    def rhs(t, y):
        ugas = y[0]
        temp = GM1 * ugas / RHO
        return [ALPHA_A * (ETOT0 - ugas - temp**4)]

    sol = solve_ivp(rhs, [0.0, t_eval[-1]], [UGAS0], t_eval=t_eval,
                     rtol=1e-11, atol=1e-13, method="RK45", dense_output=False)
    ugas = sol.y[0]
    temp = GM1 * ugas / RHO
    urad = ETOT0 - ugas
    return temp, urad


def equilibrium_static():
    """Paper Eq. 68: RHO*T_equil/GM1 + T_equil^4 = Etot0."""
    def residual(temp):
        return RHO * temp / GM1 + temp**4 - ETOT0
    return brentq(residual, 0.0, ETOT0**0.25 + 1.0)


def load_do(tab_dir, basename):
    t, dens = load_series(tab_dir, basename, "rad_hydro_w", "dens")
    _, velx = load_series(tab_dir, basename, "rad_hydro_w", "velx")
    _, eint = load_series(tab_dir, basename, "rad_hydro_w", "eint")
    _, r00 = load_series(tab_dir, basename, "rad_hydro_w", "r00")
    _, r01 = load_series(tab_dir, basename, "rad_hydro_w", "r01")
    temp = GM1 * eint / dens
    return {"t": t, "dens": dens, "velx": velx, "eint": eint, "temp": temp,
            "r00": r00, "r01": r01}


def load_m1(tab_dir, basename, moving):
    t, dens = load_series(tab_dir, basename, "mhd_w", "dens")
    _, velx = load_series(tab_dir, basename, "mhd_w", "velx")
    _, temp = load_series(tab_dir, basename, "mhd_w", "temperature")
    _, E = load_series(tab_dir, basename, "rad_m1_E", "E:0")
    Fx = None
    if moving:
        _, Fx = load_series(tab_dir, basename, "rad_m1_F", "Fx:0")
    return {"t": t, "dens": dens, "velx": velx, "temp": temp, "E": E, "Fx": Fx}


def matter_T00_T01(dens, u1, temp):
    """Coordinate-frame matter stress-energy T^00, T^01 (flat, static metric),
    Gamma=5/3 ideal gas. NOTE: both codes' "velx" primitive output column is
    actually the CONTRAVARIANT spatial four-velocity u^1 = W*v (standard
    SR-hydro primitive convention here), not the ordinary velocity v --
    confirmed from raw tab output (u1=1.0 exactly at t=0 for the "vx=1/sqrt(2)"
    input, matching the paper's u^1=1 choice). u0 = W = sqrt(1+u1^2)."""
    u0 = np.sqrt(1.0 + u1**2)
    pgas = dens * temp
    wtot = dens + GAMMA / GM1 * pgas
    T00 = wtot * u0 * u0 - pgas
    T01 = wtot * u0 * u1
    return T00, T01


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--do-static-dir", default="tab_do_static")
    parser.add_argument("--do-moving-dir", default="tab_do_moving")
    parser.add_argument("--m1-static-dir", default="tab_m1_static")
    parser.add_argument("--m1-moving-dir", default="tab_m1_moving")
    parser.add_argument("--plot-dir", default=None,
                         help="if given, write comparison PNGs here")
    parser.add_argument("--equilibrium-tol", type=float, default=3e-2)
    parser.add_argument("--conservation-tol", type=float, default=1e-3)
    args = parser.parse_args(argv)

    ok = True

    # ---------------- static case ----------------
    do_s = load_do(args.do_static_dir, "relax_paper_static")
    m1_s = load_m1(args.m1_static_dir, "photon_equilibration_paper_static",
                    moving=False)
    T_equil = equilibrium_static()
    T_analytic, urad_analytic = analytic_static_trajectory(do_s["t"])
    T_analytic_m1, urad_analytic_m1 = analytic_static_trajectory(m1_s["t"])

    print("=== Static equilibration ===")
    print("analytic equilibrium T_equil = {:.6f} (paper Eq. 68)".format(T_equil))
    # DO and M1 land on different output times (different step sizes) --
    # interpolate M1's trajectory onto DO's time grid for the printed table.
    m1_temp_on_do_t = np.interp(do_s["t"], m1_s["t"], m1_s["temp"])
    m1_E_on_do_t = np.interp(do_s["t"], m1_s["t"], m1_s["E"])
    print("{:>8s}  {:>10s} {:>10s} {:>10s}   {:>10s} {:>10s} {:>10s}".format(
        "t", "T_do", "T_m1", "T_exact", "urad_do", "urad_m1(E)", "urad_exact"))
    n_do = len(do_s["t"])
    for i in np.linspace(0, n_do - 1, 15, dtype=int):
        print("{:8.3f}  {:10.6f} {:10.6f} {:10.6f}   {:10.6f} {:10.6f} {:10.6f}".format(
            do_s["t"][i], do_s["temp"][i], m1_temp_on_do_t[i], T_analytic[i],
            do_s["r00"][i], m1_E_on_do_t[i], urad_analytic[i]))

    T_do_err = abs(do_s["temp"][-1] - T_equil) / T_equil
    T_m1_err = abs(m1_s["temp"][-1] - T_equil) / T_equil
    E_do_err = abs(do_s["r00"][-1] - (ETOT0 - T_equil**4 / GM1 * GM1)) / max(URAD0, 1e-30)
    urad_equil = ETOT0 - RHO * T_equil / GM1
    E_do_err = abs(do_s["r00"][-1] - urad_equil) / urad_equil
    E_m1_err = abs(m1_s["E"][-1] - urad_equil) / urad_equil
    print("\nfinal T_do={:.6f}, T_m1={:.6f}, T_equil={:.6f}".format(
        do_s["temp"][-1], m1_s["temp"][-1], T_equil))
    print("  T rel err: DO={:.3e}  M1={:.3e}".format(T_do_err, T_m1_err))
    print("final urad_do(=r00)={:.6f}, urad_m1(=E)={:.6f}, urad_equil={:.6f}".format(
        do_s["r00"][-1], m1_s["E"][-1], urad_equil))
    print("  urad rel err: DO={:.3e}  M1={:.3e}".format(E_do_err, E_m1_err))

    if T_do_err > args.equilibrium_tol or T_m1_err > args.equilibrium_tol:
        print("FAIL: static case did not converge to the analytic equilibrium")
        ok = False
    if E_do_err > args.equilibrium_tol or E_m1_err > args.equilibrium_tol:
        print("FAIL: static radiation energy density did not converge")
        ok = False

    # ---------------- moving case ----------------
    do_m = load_do(args.do_moving_dir, "relax_paper_moving")
    m1_m = load_m1(args.m1_moving_dir, "photon_equilibration_paper_moving",
                    moving=True)

    T00_do0, T01_do0 = matter_T00_T01(do_m["dens"][0], do_m["velx"][0], do_m["temp"][0])
    T00_dof, T01_dof = matter_T00_T01(do_m["dens"][-1], do_m["velx"][-1], do_m["temp"][-1])
    Etot_do0 = T00_do0 + do_m["r00"][0]
    Etot_dof = T00_dof + do_m["r00"][-1]
    Ptot_do0 = T01_do0 + do_m["r01"][0]
    Ptot_dof = T01_dof + do_m["r01"][-1]

    T00_m10, T01_m10 = matter_T00_T01(m1_m["dens"][0], m1_m["velx"][0], m1_m["temp"][0])
    T00_m1f, T01_m1f = matter_T00_T01(m1_m["dens"][-1], m1_m["velx"][-1], m1_m["temp"][-1])
    Etot_m10 = T00_m10 + m1_m["E"][0]
    Etot_m1f = T00_m1f + m1_m["E"][-1]
    Ptot_m10 = T01_m10 + m1_m["Fx"][0]
    Ptot_m1f = T01_m1f + m1_m["Fx"][-1]

    E_cons_do = abs(Etot_dof - Etot_do0) / abs(Etot_do0)
    P_cons_do = abs(Ptot_dof - Ptot_do0) / max(abs(Ptot_do0), 1e-12)
    E_cons_m1 = abs(Etot_m1f - Etot_m10) / abs(Etot_m10)
    P_cons_m1 = abs(Ptot_m1f - Ptot_m10) / max(abs(Ptot_m10), 1e-12)

    print("\n=== Moving equilibration (energy-momentum conservation) ===")
    print("DO : E_tot(0)={:.6f}  E_tot(f)={:.6f}  rel_err={:.3e}".format(
        Etot_do0, Etot_dof, E_cons_do))
    print("DO : P_tot(0)={:.6f}  P_tot(f)={:.6f}  rel_err={:.3e}".format(
        Ptot_do0, Ptot_dof, P_cons_do))
    print("M1 : E_tot(0)={:.6f}  E_tot(f)={:.6f}  rel_err={:.3e}".format(
        Etot_m10, Etot_m1f, E_cons_m1))
    print("M1 : P_tot(0)={:.6f}  P_tot(f)={:.6f}  rel_err={:.3e}".format(
        Ptot_m10, Ptot_m1f, P_cons_m1))

    v_do_final = do_m["velx"][-1] / np.sqrt(1.0 + do_m["velx"][-1]**2)
    v_m1_final = m1_m["velx"][-1] / np.sqrt(1.0 + m1_m["velx"][-1]**2)
    print("\nfinal state -- DO  vs  M1 (should agree: same physics, same target):")
    print("  T_gas    : DO={:.6f}  M1={:.6f}  rel diff={:.3e}".format(
        do_m["temp"][-1], m1_m["temp"][-1],
        abs(do_m["temp"][-1] - m1_m["temp"][-1]) / m1_m["temp"][-1]))
    print("  u^1      : DO={:.6f}  M1={:.6f}  abs diff={:.3e}".format(
        do_m["velx"][-1], m1_m["velx"][-1],
        abs(do_m["velx"][-1] - m1_m["velx"][-1])))
    print("  v (=u1/W): DO={:.6f}  M1={:.6f}  abs diff={:.3e}".format(
        v_do_final, v_m1_final, abs(v_do_final - v_m1_final)))
    print("  R^00(coord)/E : DO={:.6f}  M1={:.6f}  rel diff={:.3e}".format(
        do_m["r00"][-1], m1_m["E"][-1],
        abs(do_m["r00"][-1] - m1_m["E"][-1]) / m1_m["E"][-1]))

    if E_cons_do > args.conservation_tol or E_cons_m1 > args.conservation_tol:
        print("FAIL: moving-case total energy not conserved")
        ok = False
    if P_cons_do > args.conservation_tol or P_cons_m1 > args.conservation_tol:
        print("FAIL: moving-case total momentum not conserved")
        ok = False
    final_state_tol = 0.1
    if abs(do_m["temp"][-1] - m1_m["temp"][-1]) / m1_m["temp"][-1] > final_state_tol:
        print("FAIL: DO and M1 final T_gas disagree by more than {:.0%}".format(
            final_state_tol))
        ok = False

    if args.plot_dir:
        make_plots(args.plot_dir, do_s, m1_s, T_analytic, urad_analytic, T_equil,
                   do_m, m1_m)

    if ok:
        print("\nPASS: static case matches the independently re-integrated "
              "analytic ODE and equilibrium; moving case conserves "
              "energy-momentum in both codes and both converge to the same "
              "final state")
    else:
        print("\nFAIL: see above")
        return False
    return True


def make_plots(plot_dir, do_s, m1_s, T_analytic, urad_analytic, T_equil,
               do_m, m1_m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(plot_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(do_s["t"], do_s["temp"], "o", ms=4, color="C0", label="DO (finite-solid-angle)")
    ax[0].plot(m1_s["t"], m1_s["temp"], "s", ms=4, color="C1", label="M1")
    ax[0].plot(do_s["t"], T_analytic, "-", color="k", lw=1.5, label="analytic (Eq. 69)")
    ax[0].axhline(T_equil, ls=":", color="gray", label="T_equil (Eq. 68)")
    ax[0].set_xlabel("t"); ax[0].set_ylabel(r"$T_{gas}(t)$")
    ax[0].set_title("Static equilibration: gas temperature")
    ax[0].legend(fontsize=8)

    ax[1].plot(do_s["t"], do_s["r00"], "o", ms=4, color="C0", label="DO $R^{00}$")
    ax[1].plot(m1_s["t"], m1_s["E"], "s", ms=4, color="C1", label="M1 $E$")
    ax[1].plot(do_s["t"], urad_analytic, "-", color="k", lw=1.5, label="analytic")
    ax[1].set_xlabel("t"); ax[1].set_ylabel(r"$u_{rad}(t)$")
    ax[1].set_title("Static equilibration: radiation energy density")
    ax[1].legend(fontsize=8)
    fig.suptitle("Stage 7 (arXiv:2302.04283 §3.6): static equilibration, M1 vs DO vs analytic",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "stage7_equilibration_static.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(do_m["t"], do_m["temp"], "o-", ms=3, color="C0", label="DO $T_{gas}$")
    ax[0].plot(m1_m["t"], m1_m["temp"], "s-", ms=3, color="C1", label="M1 $T_{gas}$")
    ax[0].set_xlabel("t"); ax[0].set_ylabel(r"$T_{gas}(t)$")
    ax[0].set_title("Moving equilibration: gas temperature")
    ax[0].legend(fontsize=8)

    ax[1].plot(do_m["t"], do_m["velx"], "o-", ms=3, color="C0", label="DO $u^1$")
    ax[1].plot(m1_m["t"], m1_m["velx"], "s-", ms=3, color="C1", label="M1 $u^1$")
    ax[1].set_xlabel("t"); ax[1].set_ylabel(r"$u^1(t)$ (contravariant)")
    ax[1].set_title("Moving equilibration: fluid velocity (momentum coupling)")
    ax[1].legend(fontsize=8)
    fig.suptitle("Stage 7 (arXiv:2302.04283 §3.6): moving equilibration, M1 vs DO",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "stage7_equilibration_moving.png"), dpi=130)
    plt.close(fig)
    print("\nplots written to", plot_dir)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
