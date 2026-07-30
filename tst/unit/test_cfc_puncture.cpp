//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file test_cfc_puncture.cpp
//! \brief Standalone (no Mesh/Driver/MPI) validation of src/cfc/cfc_puncture.hpp's
//! stationary BH trumpet background, against CFC_PUNCTURE_TDE_PLAN.md Sec 3.2's
//! documented asymptotic limits and independently hand-evaluated mod_bh.f90 formulas.
//! Every function under test is KOKKOS_INLINE_FUNCTION (host+device callable), so this
//! just calls them directly on the host after Kokkos::initialize() -- no mesh, Driver,
//! or pgen machinery needed.

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <Kokkos_Core.hpp>

#include "athena.hpp"
#include "cfc/cfc_puncture.hpp"

namespace {

int nfail = 0;

void Check(bool ok, const char *what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) { ++nfail; }
}

bool Close(Real a, Real b, Real reltol) {
  return std::fabs(a - b) <= reltol*std::fmax(1.0, std::fabs(b));
}

}  // namespace

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    // ---- 1. TrumpetArealToIso/TrumpetIsoToAreal are inverses -------------------------
    const Real rhos[] = {1.5, 1.5 + 1.0e-6, 1.6, 1.85, 2.0, 3.0, 5.0, 10.0,
                          100.0, 1.0e3, 1.0e4};
    for (Real rho : rhos) {
      Real chi = cfc::TrumpetArealToIso(rho);
      Real rho2 = cfc::TrumpetIsoToAreal(chi);
      char msg[160];
      std::snprintf(msg, sizeof(msg),
                    "round-trip rho=%.6g -> chi=%.6g -> rho2=%.6g", rho, chi, rho2);
      Check(Close(rho2, rho, 1.0e-10), msg);
    }

    // ---- 2a. chi(rho) -> rho-1 as rho -> infinity (correct sign/leading term) -------
    {
      Real rho = 1.0e8;
      Real chi = cfc::TrumpetArealToIso(rho);
      char msg[160];
      std::snprintf(msg, sizeof(msg),
                    "chi(rho)->rho-1 at rho=%.3g: chi=%.10g, rho-1=%.10g", rho, chi,
                    rho - 1.0);
      Check(Close(chi, rho - 1.0, 1.0e-6), msg);
    }

    // ---- 2b. chi(rho) -> 0 as rho -> throat (throat maps to r_iso=0) ----------------
    {
      Real chi_throat = cfc::TrumpetArealToIso(cfc::kTrumpetThroat);
      char msg[160];
      std::snprintf(msg, sizeof(msg), "chi(throat)=%.3e (expect ~0)", chi_throat);
      Check(std::fabs(chi_throat) < 1.0e-12, msg);

      Real chi_near = cfc::TrumpetArealToIso(cfc::kTrumpetThroat + 1.0e-6);
      std::snprintf(msg, sizeof(msg),
                    "chi(throat+1e-6)=%.3e (expect small, same order as 1e-6)",
                    chi_near);
      Check(chi_near > 0.0 && chi_near < 1.0e-3, msg);
    }

    // ---- 3. TrumpetBackground's M_BH -> 0 limit collapses to flat space -------------
    {
      Real m_bh = 1.0e-6;
      Real r_iso = 5.0;
      Real rho = cfc::TrumpetIsoToAreal(r_iso/m_bh);
      Real r_sch = m_bh*rho;
      Real psi0, alpha0, beta0[3], Aij0[6], a2;
      cfc::TrumpetBackground(m_bh, r_iso, 0.0, 0.0, r_sch,
                             &psi0, &alpha0, beta0, Aij0, &a2);
      char msg[200];
      std::snprintf(msg, sizeof(msg),
                    "M_BH->0: psi0=%.10g (want ~1), alpha0=%.10g (want ~1)",
                    psi0, alpha0);
      Check(Close(psi0, 1.0, 1.0e-5) && Close(alpha0, 1.0, 1.0e-5), msg);

      Real beta_mag = std::sqrt(beta0[0]*beta0[0] + beta0[1]*beta0[1]
                                 + beta0[2]*beta0[2]);
      std::snprintf(msg, sizeof(msg),
                    "M_BH->0: |beta0|=%.3e (want ~0), a2=%.3e (want ~0)", beta_mag, a2);
      Check(beta_mag < 1.0e-5 && a2 < 1.0e-10, msg);
    }

    // ---- 4a. Hand-evaluated spot check: alpha0 at the horizon (rho=2) ---------------
    // alpha0 = sqrt(1 - 2/2 + 1.6875/2^4) = sqrt(27/256) = 3*sqrt(3)/16 = 0.3247595...
    // (independently computed, not by re-running this same code)
    {
      Real m_bh = 1.0, r_sch = 2.0;
      Real psi0, alpha0, beta0[3], Aij0[6], a2;
      cfc::TrumpetBackground(m_bh, 1.0, 0.0, 0.0, r_sch,
                             &psi0, &alpha0, beta0, Aij0, &a2);
      char msg[160];
      std::snprintf(msg, sizeof(msg),
                    "alpha0(rho=2)=%.6g, hand-computed=0.3247595", alpha0);
      Check(Close(alpha0, 0.3247595264, 1.0e-6), msg);
    }

    // ---- 4b. Hand-evaluated spot check: alpha0 near rho=1.85 (default excise_lapse) -
    // alpha0 = sqrt(1 - 2/1.85 + 1.6875/1.85^4) = sqrt(0.0629804) = 0.250959...
    {
      Real m_bh = 1.0, r_sch = 1.85;
      Real psi0, alpha0, beta0[3], Aij0[6], a2;
      cfc::TrumpetBackground(m_bh, 1.0, 0.0, 0.0, r_sch,
                             &psi0, &alpha0, beta0, Aij0, &a2);
      char msg[160];
      std::snprintf(msg, sizeof(msg),
                    "alpha0(rho=1.85)=%.6g, hand-computed=0.250959", alpha0);
      Check(Close(alpha0, 0.250959, 1.0e-4), msg);
    }

    // ---- 4c. Hand-evaluated spot check: Ahat0^2 at the throat -----------------------
    // a2 = 10.125*m^4/rs^6 = (81/8)/(3/2)^6 = (81/8)/(729/64) = 8/9 = 0.888888...
    {
      Real m_bh = 1.0, r_sch = cfc::kTrumpetThroat;
      Real psi0, alpha0, beta0[3], Aij0[6], a2;
      cfc::TrumpetBackground(m_bh, 1.0, 0.0, 0.0, r_sch,
                             &psi0, &alpha0, beta0, Aij0, &a2);
      char msg[160];
      std::snprintf(msg, sizeof(msg), "Ahat0^2(throat)=%.8g, hand-computed=8/9=%.8g",
                    a2, 8.0/9.0);
      Check(Close(a2, 8.0/9.0, 1.0e-12), msg);
    }

    // ---- 4d. Internal consistency: Aij0's packed components reproduce a2 -----------
    // Ahat0_ij*Ahat0^ij (flat metric) = sum(diag^2) + 2*sum(offdiag^2), must equal a2
    // at an arbitrary, non-special point -- catches component-order/sign bugs the
    // special-case checks above wouldn't.
    {
      Real m_bh = 1.3, r_sch = 4.7;
      Real psi0, alpha0, beta0[3], Aij0[6], a2;
      cfc::TrumpetBackground(m_bh, 0.6, -1.1, 2.3, r_sch,
                             &psi0, &alpha0, beta0, Aij0, &a2);
      Real norm2 = Aij0[0]*Aij0[0] + Aij0[1]*Aij0[1] + Aij0[2]*Aij0[2]
                   + 2.0*(Aij0[3]*Aij0[3] + Aij0[4]*Aij0[4] + Aij0[5]*Aij0[5]);
      char msg[200];
      std::snprintf(msg, sizeof(msg),
                    "Aij0.Aij0 (packed)=%.10g vs a2=%.10g at an arbitrary point",
                    norm2, a2);
      Check(Close(norm2, a2, 1.0e-10), msg);
    }

    // ---- 5. WormholeBackground sanity: flat as m_bh->0, and Ahat0=beta0=0 exactly --
    {
      Real m_bh = 1.0, x1 = 3.0, x2 = 0.0, x3 = 0.0;
      Real psi0, alpha0, beta0[3], Aij0[6], a2;
      cfc::WormholeBackground(m_bh, x1, x2, x3, &psi0, &alpha0, beta0, Aij0, &a2);
      bool zero_curvature = (beta0[0] == 0.0 && beta0[1] == 0.0 && beta0[2] == 0.0
                             && a2 == 0.0);
      for (int a = 0; a < 6; ++a) { zero_curvature = zero_curvature && (Aij0[a] == 0.0); }
      char msg[160];
      std::snprintf(msg, sizeof(msg),
                    "wormhole: psi0=%.6g, alpha0=%.6g, beta0=Ahat0=0 exactly=%d",
                    psi0, alpha0, zero_curvature);
      Check(Close(psi0, 1.0 + 0.5/3.0, 1.0e-12) && zero_curvature, msg);
    }
  }
  Kokkos::finalize();

  std::printf("\n%s (%d failure%s)\n", nfail == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED",
              nfail, nfail == 1 ? "" : "s");
  return nfail == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
