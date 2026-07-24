#!/usr/bin/env python
"""Radiation-hydrodynamic linear wave check (arXiv:2302.04283 Section 3.9, Stage 13).

Compares the DO module's rad_linear_wave.cpp (already-existing, already-validated,
CI-wired capability -- inputs/tests/rad_linwave_uniform.athinput) against the new
M1-side counterpart (RadiationM1LinearWave, inputs/tests/
rad_m1_photon_linear_wave.athinput) on the paper's own gas-dominated "H1" case
(p_rad/p_gas=1/10), at two resolutions (32, 64 cells), reproducing the paper's own
convergence-with-resolution check (Eq. 78a/78b).

Both athinputs' <problem> blocks carry the *same* background state and complex
eigenvalue/eigenvector, read verbatim from the paper's own Appendix A Table 1/2
"H1" values (also reproduced here, so this script has no runtime dependency on
either athinput file). Unlike DO's own internal error-computation hook
(RadiationLinearWaveErrors, called via pgen_final_func), this script independently
reconstructs the *full* time-dependent analytic solution -- including the damping
envelope exp(omega_imag*t) and the phase cos/sin(omega_real*t - k*x) -- evaluated at
whatever time each code's saved output actually reports, and compares directly
against it. Primitives (rho, pgas, u_x) are compared directly; the radiation
moments are compared in the *lab* (Eulerian-observer) frame -- both codes' native
storage convention -- rather than the paper's own fluid-frame (Ebar, Fbar_x), via
the same fluid-frame-to-lab-frame Lorentz boost used to set up the M1 initial
condition (rad_m1_linear_wave.cpp) -- a valid, equivalent choice of comparison
variables (any invertible linear/nonlinear recombination of the correct primitives
converges at the same rate).

Usage (run after executing AthenaK for all four DO/M1 x 32/64 combinations):
    python check_rad_m1_photon_linear_wave.py \\
        --do-32-dir /path/to/do_32/tab --do-64-dir /path/to/do_64/tab \\
        --m1-32-dir /path/to/m1_32/tab --m1-64-dir /path/to/m1_64/tab \\
        --plot-dir /path/to/plots
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                 "inputs", "tests"))
from m1_tab_utils import read_profile  # noqa: E402

# Background state + complex eigenvalue/eigenvector -- paper's Appendix A Table 1/2
# "H1" (gas-dominated, p_rad/p_gas=1/10) case, matching both athinputs verbatim.
RHO0 = 1.0
PGAS0 = 2.497687326549491e-01
ERAD0 = 7.493061979648474e-02
GAMMA = 5.0 / 3.0
DELTA = 1.0e-4
OMEGA_REAL = 3.1488157526582414e+00
OMEGA_IMAG = -2.6190006385782953e-02
DRHO_REAL, DRHO_IMAG = 8.3877889167048014e-01, 0.0
DPGAS_REAL, DPGAS_IMAG = 3.2084488925731219e-01, -9.9134535607493107e-03
DUX_REAL, DUX_IMAG = 4.2035369927276667e-01, -3.4962560317943620e-03
DERAD_REAL, DERAD_IMAG = 1.2904189937790903e-01, 1.5203926879094193e-03
DFXRAD_REAL, DFXRAD_IMAG = 1.3260665610966586e-03, -6.7017329068802516e-03
K_PAR = 2.0 * np.pi  # domain length 1, one wavelength


def analytic(x, t):
    """Full time-dependent analytic solution at (x, t): (rho, pgas, ux, E_lab, Fx_lab)."""
    en = np.exp(OMEGA_IMAG * t)
    cn = np.cos(OMEGA_REAL * t - K_PAR * x)
    sn = np.sin(OMEGA_REAL * t - K_PAR * x)

    rho = RHO0 + DELTA * en * (DRHO_REAL * cn + DRHO_IMAG * sn)
    pgas = PGAS0 + DELTA * en * (DPGAS_REAL * cn + DPGAS_IMAG * sn)
    ux = DELTA * en * (DUX_REAL * cn + DUX_IMAG * sn)
    erad = ERAD0 + DELTA * en * (DERAD_REAL * cn + DERAD_IMAG * sn)
    fxrad = DELTA * en * (DFXRAD_REAL * cn + DFXRAD_IMAG * sn)

    # fluid-frame -> lab-frame boost (Eddington closure, pure x1 boost) -- same
    # formula as rad_m1_linear_wave.cpp
    u0 = np.sqrt(1.0 + ux**2)
    lam00, lam01 = u0, ux
    lam11 = 1.0 + ux**2 / (1.0 + u0)
    rf00, rf01, rf11 = erad, fxrad, erad / 3.0
    e_lab = lam00**2 * rf00 + 2.0 * lam00 * lam01 * rf01 + lam01**2 * rf11
    fx_lab = (lam00 * lam01 * rf00 + (lam00 * lam11 + lam01**2) * rf01 +
              lam01 * lam11 * rf11)
    return rho, pgas, ux, e_lab, fx_lab


def load_do(tab_dir):
    w_files = sorted(glob.glob(os.path.join(tab_dir, "*.hydro_w.*.tab")))
    r_files = sorted(glob.glob(os.path.join(tab_dir, "*.rad_coord.*.tab")))
    x1v, w = read_profile(w_files[-1])
    _, r = read_profile(r_files[-1])
    rho = w["dens"]
    pgas = w["eint"] * (GAMMA - 1.0)  # "eint" is volumetric internal energy density
                                       # (basetype_output.cpp emplaces w0's raw IEN
                                       # slot, itself set as pgas/(gamma-1) with no
                                       # division by density -- NOT specific/per-mass
                                       # internal energy, despite the column name)
    ux = w["velx"]
    e_lab = r["r00"]
    fx_lab = r["r01"]
    with open(w_files[-1]) as f:
        t = float(re.search(r"time=(\S+)", f.readline()).group(1))
    return t, x1v, rho, pgas, ux, e_lab, fx_lab


def load_m1(tab_dir):
    w_files = sorted(glob.glob(os.path.join(tab_dir, "*.mhd_w.*.tab")))
    e_files = sorted(glob.glob(os.path.join(tab_dir, "*.rad_m1_E.*.tab")))
    f_files = sorted(glob.glob(os.path.join(tab_dir, "*.rad_m1_F.*.tab")))
    x1v, w = read_profile(w_files[-1])
    _, ed = read_profile(e_files[-1])
    _, fd = read_profile(f_files[-1])
    rho = w["dens"]
    pgas = w["press"]
    ux = w["velx"]
    e_lab = ed["E:0"]
    fx_lab = fd["Fx:0"]
    with open(w_files[-1]) as f:
        t = float(re.search(r"time=(\S+)", f.readline()).group(1))
    return t, x1v, rho, pgas, ux, e_lab, fx_lab


def l1_error(x1v, field, field_exact):
    return np.trapz(np.abs(field - field_exact), x1v) / DELTA


def total_error(t, x1v, rho, pgas, ux, e_lab, fx_lab):
    rho_a, pgas_a, ux_a, e_a, fx_a = analytic(x1v, t)
    eps = [l1_error(x1v, rho, rho_a), l1_error(x1v, pgas, pgas_a),
           l1_error(x1v, ux, ux_a), l1_error(x1v, e_lab, e_a),
           l1_error(x1v, fx_lab, fx_a)]
    return float(np.sqrt(np.mean(np.square(eps)))), (rho_a, pgas_a, ux_a, e_a, fx_a)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--do-32-dir", required=True)
    parser.add_argument("--do-64-dir", required=True)
    parser.add_argument("--m1-32-dir", required=True)
    parser.add_argument("--m1-64-dir", required=True)
    parser.add_argument("--plot-dir", default=None)
    args = parser.parse_args(argv)

    do32 = load_do(args.do_32_dir)
    do64 = load_do(args.do_64_dir)
    m132 = load_m1(args.m1_32_dir)
    m164 = load_m1(args.m1_64_dir)

    eps_do32, an32 = total_error(*do32)
    eps_do64, an64 = total_error(*do64)
    eps_m132, anm32 = total_error(*m132)
    eps_m164, anm64 = total_error(*m164)

    print("DO t={:.4f} (32), t={:.4f} (64); M1 t={:.4f} (32), t={:.4f} (64)".format(
        do32[0], do64[0], m132[0], m164[0]))
    print("\n{:>6s} {:>14s} {:>14s}".format("res", "DO eps", "M1 eps"))
    print("{:>6d} {:14.6e} {:14.6e}".format(32, eps_do32, eps_m132))
    print("{:>6d} {:14.6e} {:14.6e}".format(64, eps_do64, eps_m164))
    print("\nconvergence ratio eps(64)/eps(32): DO={:.4f}  M1={:.4f}".format(
        eps_do64 / eps_do32, eps_m164 / eps_m132))
    print("(expect ~0.5 for 2nd-order convergence, ~0.25 for 3rd/4th order schemes;")
    print(" the DO CI test's own threshold is < 0.23 at this same 32/64 pair)")

    if args.plot_dir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(args.plot_dir, exist_ok=True)

        t64, x64, rho64, pgas64, ux64, e64, fx64 = do64
        _, an_rho, an_pgas, an_ux, an_e = anm64[0], anm64[1], anm64[2], anm64[3], None
        rho_a, pgas_a, ux_a, e_a, fx_a = analytic(x64, t64)

        tm64, xm64, rhom64, pgasm64, uxm64, em64, fxm64 = m164

        fig, axes = plt.subplots(1, 5, figsize=(22, 4))
        labels = ["rho", "pgas", "u_x", "E (lab)", "F_x (lab)"]
        do_fields = [rho64, pgas64, ux64, e64, fx64]
        m1_fields = [rhom64, pgasm64, uxm64, em64, fxm64]
        an_fields = [rho_a, pgas_a, ux_a, e_a, fx_a]
        for ax, label, do_f, m1_f, an_f in zip(axes, labels, do_fields, m1_fields,
                                                an_fields):
            ax.plot(x64, an_f, "k-", lw=2, label="exact")
            ax.plot(x64, do_f, "C0o", ms=3, label="DO")
            ax.plot(xm64, m1_f, "C1x", ms=4, label="M1")
            ax.set_xlabel("x")
            ax.set_title(label)
            ax.legend(fontsize=8)
        fig.suptitle("Linear wave, H1 case, nx1=64, t={:.2f}".format(t64))
        fig.tight_layout()
        fig.savefig(os.path.join(args.plot_dir, "stage13_linear_wave_profiles.png"),
                    dpi=150)

        fig2, ax2 = plt.subplots(figsize=(6, 5))
        res = [32, 64]
        ax2.loglog(res, [eps_do32, eps_do64], "C0o-", label="DO")
        ax2.loglog(res, [eps_m132, eps_m164], "C1x-", label="M1")
        ax2.loglog(res, [eps_do32, eps_do32 * 0.5], "k--", lw=1, label="2nd order")
        ax2.set_xlabel("resolution (nx1)")
        ax2.set_ylabel(r"$\epsilon$")
        ax2.set_title("Linear wave convergence (H1 case)")
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig(os.path.join(args.plot_dir, "stage13_linear_wave_convergence.png"),
                     dpi=150)
        print("\nplots written to {}".format(args.plot_dir))

    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
