//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file test_cfc_puncture_device.cpp
//! \brief Device (DevExeSpace) counterpart of test_cfc_puncture.cpp: evaluates the REAL
//! src/cfc/cfc_puncture.hpp functions inside a Kokkos kernel and compares them, value by
//! value, with the same calls on the host (C-094 P1). test_cfc_puncture.cpp only ever ran
//! them on the host, where they are correct.
//!
//! Part A records, on the device and on the host, every intermediate of TrumpetArealToIso
//! at the throat rho = 1.5 (where numer = 0 and the result is pow(0, 1/sqrt 2)), the bracket
//! values TrumpetIsoToAreal hands NumTools::Root::FalsePosition for r_iso < 1.5 (lb = 1.5),
//! FalsePosition's return value and root, and a few pow(0, y) variants. Part B evaluates
//! TrumpetIsoToAreal and TrumpetBackground's psi0/alpha0 on an r_iso grid. Inputs reach the
//! kernel through a View so nothing is constant-folded. Exit code = number of failed checks.
//!
//! Not a CMake target (see run_test_cfc_puncture_device.sh): it reuses the athena build's
//! own compile flags and Kokkos libraries.

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <Kokkos_Core.hpp>

#include "athena.hpp"
#include "cfc/cfc_puncture.hpp"

namespace {

constexpr int kNA = 16;   // Part A slots

// Part A: everything TrumpetIsoToAreal(chi) does for chi < 1.5, step by step.
KOKKOS_INLINE_FUNCTION
void PartA(Real rho, Real chi, Real zero, Real *o) {
  Real s1 = std::sqrt(4.0*rho*rho + 4.0*rho + 3.0);
  Real s2 = std::sqrt(8.0*rho*rho + 8.0*rho + 6.0);
  Real numer = (4.0 + 3.0*std::sqrt(2.0)) * (2.0*rho - 3.0);
  Real denom = 8.0*rho + 6.0 + 3.0*s2;
  Real q = numer/denom;
  Real p = std::pow(q, 1.0/std::sqrt(2.0));
  o[0] = numer;
  o[1] = q;
  o[2] = p;
  o[3] = 0.25*(2.0*rho + 1.0 + s1);
  o[4] = cfc::TrumpetArealToIso(rho);                  // the real function
  cfc::impl::TrumpetArealResidual res;
  Real lb = std::fmax(cfc::kTrumpetThroat, chi);
  Real ub = std::fmax(cfc::kTrumpetThroat, chi + 1.5);
  o[5] = res(lb, chi);                                 // f(lb)
  o[6] = res(ub, chi);                                 // f(ub)
  NumTools::Root root;
  root.iterations = 100;
  Real x = -1.0;
  bool ok = root.FalsePosition(res, lb, ub, x, 1.0e-15, chi);
  o[7] = ok ? 1.0 : 0.0;
  o[8] = x;
  o[9] = cfc::TrumpetIsoToAreal(chi);                  // the real function
  o[10] = std::pow(zero, 1.0/std::sqrt(2.0));          // pow(0, y), runtime zero
  o[11] = std::pow(zero, 0.5);
  o[12] = std::pow(zero, 2.0);
  o[13] = std::exp((1.0/std::sqrt(2.0))*std::log(zero));
  o[14] = std::log(zero);
  o[15] = std::sqrt(zero);
}

int nfail = 0;

void Check(bool ok, const char *what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) { ++nfail; }
}

bool Same(Real a, Real b, Real reltol) {
  if (std::isnan(a) || std::isnan(b)) { return std::isnan(a) && std::isnan(b); }
  return std::fabs(a - b) <= reltol*std::fmax(1.0e-300, std::fabs(b));
}

}  // namespace

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    std::printf("execution space: %s\n", DevExeSpace::name());
    // ---- Part A ----------------------------------------------------------------------
    const Real chis[] = {0.044, 0.7, 1.4, 1.4999, 2.0};
    const int nchi = sizeof(chis)/sizeof(chis[0]);
    Kokkos::View<Real*> in("in", nchi + 2);
    auto in_h = Kokkos::create_mirror_view(in);
    in_h(0) = 1.5; in_h(1) = 0.0;
    for (int c = 0; c < nchi; ++c) { in_h(2 + c) = chis[c]; }
    Kokkos::deep_copy(in, in_h);
    Kokkos::View<Real**> outa("outa", nchi, kNA);
    Kokkos::parallel_for("test_cfc_puncture_device_A", Kokkos::RangePolicy<DevExeSpace>(0, nchi),
    KOKKOS_LAMBDA(const int c) {
      Real o[kNA];
      PartA(in(0), in(2 + c), in(1), o);
      for (int s = 0; s < kNA; ++s) { outa(c, s) = o[s]; }
    });
    auto outa_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), outa);
    const char *names[kNA] = {"numer", "numer/denom", "pow(q,1/sqrt2)", "prefactor",
                              "TrumpetArealToIso(1.5)", "f(lb)", "f(ub)",
                              "FalsePosition ok", "FalsePosition x",
                              "TrumpetIsoToAreal(chi)", "pow(0,1/sqrt2)", "pow(0,0.5)",
                              "pow(0,2)", "exp(y log 0)", "log(0)", "sqrt(0)"};
    for (int c = 0; c < nchi; ++c) {
      Real o[kNA];
      volatile Real rho = in_h(0), chi = in_h(2 + c), zero = in_h(1);
      PartA(rho, chi, zero, o);
      std::printf("-- chi = %.6g  (lb = %.6g)\n", chis[c], std::fmax(1.5, chis[c]));
      for (int s = 0; s < kNA; ++s) {
        if (c > 0 && (s < 5 || s >= 10)) { continue; }   // chi-independent, print once
        std::printf("   %-24s device % .17e   host % .17e  %s\n", names[s], outa_h(c, s), o[s],
                    Same(outa_h(c, s), o[s], 1.0e-12) ? "" : "<-- DIFFERS");
      }
    }

    // ---- Part B ----------------------------------------------------------------------
    const int nr = 300;
    Kokkos::View<Real**> outb("outb", nr, 3);
    Kokkos::parallel_for("test_cfc_puncture_device_B", Kokkos::RangePolicy<DevExeSpace>(0, nr),
    KOKKOS_LAMBDA(const int n) {
      Real r = 0.01*(n + 1);                    // r_iso = 0.01 .. 3.00
      Real rs = cfc::TrumpetIsoToAreal(r);
      Real psi0, alpha0, beta0[3], aij0[6], a2;
      cfc::TrumpetBackground(1.0, r, 0.0, 0.0, rs, &psi0, &alpha0, beta0, aij0, &a2);
      outb(n, 0) = rs; outb(n, 1) = psi0; outb(n, 2) = alpha0;
    });
    auto outb_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), outb);
    int nbad = 0, nthroat = 0;
    Real rmin_bad = 1e30, rmax_bad = -1.0;
    for (int n = 0; n < nr; ++n) {
      volatile Real rv = 0.01*(n + 1);
      Real r = rv;
      Real rs = cfc::TrumpetIsoToAreal(r);
      Real psi0, alpha0, beta0[3], aij0[6], a2;
      cfc::TrumpetBackground(1.0, r, 0.0, 0.0, rs, &psi0, &alpha0, beta0, aij0, &a2);
      bool ok = Same(outb_h(n, 0), rs, 1.0e-12) && Same(outb_h(n, 1), psi0, 1.0e-12) &&
                Same(outb_h(n, 2), alpha0, 1.0e-9);
      if (outb_h(n, 0) == cfc::kTrumpetThroat) { ++nthroat; }
      if (!ok) {
        ++nbad; rmin_bad = std::fmin(rmin_bad, r); rmax_bad = std::fmax(rmax_bad, r);
      }
      if (n % 25 == 24 || n == 148 || n == 149) {
        std::printf("   r_iso %.2f  R device %.15f host %.15f | alpha0 device %.6e host %.6e\n",
                    r, outb_h(n, 0), rs, outb_h(n, 2), alpha0);
      }
    }
    char msg[200];
    std::snprintf(msg, sizeof(msg), "Part B: device == host for r_iso in [0.01, 3.00] "
                  "(%d of %d differ, r_iso %.2f..%.2f; %d device values exactly at the throat)",
                  nbad, nr, nbad ? rmin_bad : 0.0, nbad ? rmax_bad : 0.0, nthroat);
    Check(nbad == 0, msg);
    std::snprintf(msg, sizeof(msg), "Part A: device TrumpetArealToIso(1.5) == 0 (got %.17g)",
                  outa_h(0, 4));
    Check(outa_h(0, 4) == 0.0, msg);
  }
  Kokkos::finalize();
  return nfail;
}
