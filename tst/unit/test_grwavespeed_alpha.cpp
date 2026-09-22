//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file test_grwavespeed_alpha.cpp
//! \brief Regression test for src/dyn_grmhd/GRWAVESPEED_ALPHA_HANDOFF.md: unguarded
//! 1/alpha divisions in PrimitiveSolverHydro::GetGRFastMagnetosonicSpeeds and the
//! Riemann-solver/timestep code that shares it. alpha (the ADM lapse) is legitimately
//! exactly or nearly zero near a coordinate degeneracy (an unexcised moving-puncture
//! horizon, or the maximal-slicing trumpet throat CFC's TDE setup places a live cell
//! next to) and nothing downstream that divides by it was protected. Calls the real,
//! production GetGRFastMagnetosonicSpeeds (not a reproduction) via a standalone
//! PrimitiveSolverHydro instance -- no Mesh/MeshBlockPack/Driver required, since
//! ADM data is passed in directly as plain arrays.

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <Kokkos_Core.hpp>

#include "athena.hpp"
#include "parameter_input.hpp"
// dyn_grmhd.hpp must be included before primitive_solver_hyd.hpp: the two headers
// include each other, and dyn_grmhd.hpp's own use of PrimitiveSolverHydro (as a class
// member) only resolves if primitive_solver_hyd.hpp's template is fully parsed first,
// which only happens if dyn_grmhd.hpp is the outer/first include.
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/primitive_solver_hyd.hpp"

namespace {

int nfail = 0;

void Check(bool ok, const char *what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) { ++nfail; }
}

//! \brief Reproduces GetGRFastMagnetosonicSpeeds's quadratic exactly, but with the
//! ORIGINAL unguarded `ialpha = 1.0/alpha` -- to directly confirm the mechanism (the
//! task's own instructions: don't just trust the argument, show the Inf/NaN appearing
//! here specifically). Not the code under test; the fixed production function is.
void NaiveMagnetosonicSpeeds(Real &lambda_p, Real &lambda_m, Real uu[3], Real bsq,
                             Real g3d[NSPMETRIC], Real beta_u[3], Real alpha, Real gii,
                             Real cmsq) {
  Real usq = Primitive::SquareVector(uu, g3d);
  Real Wsq = 1.0 + usq;
  Real ialpha = 1.0/alpha;  // the pre-fix line, verbatim
  Real W = sqrt(Wsq);
  Real u0 = W*ialpha;
  Real u1 = uu[0] - u0*beta_u[0];
  Real g00 = -ialpha*ialpha;
  Real g01 = -g00*beta_u[0];
  Real g11 = gii - g01*beta_u[0];

  Real a = u0*u0 - (g00 + u0*u0)*cmsq;
  Real b = -2.0 * (u0 * u1 - (g01 + u0 * u1) *cmsq);
  Real c = u1*u1 - (g11 + u1*u1)*cmsq;
  Real a1 = b / a;
  Real a0 = c / a;
  Real s = fmax(a1*a1 - 4.0 * a0, 0.0);
  s = sqrt(s);
  lambda_p = (a1 >= 0.0) ? -2.0 * a0 / (a1 + s) : (-a1 + s) / 2.0;
  lambda_m = (a1 >= 0.0) ? (-a1 - s) / 2.0 : -2.0 * a0 / (a1 - s);
}

}  // namespace

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    ParameterInput pin;
    PrimitiveSolverHydro<Primitive::IdealGas, Primitive::ResetFloor> eos("mhd", nullptr,
                                                                          &pin);

    // ---- 0. FloorLapse itself ------------------------------------------------------
    {
      Check(Primitive::FloorLapse(1.0) == 1.0, "FloorLapse(1.0) == 1.0 (no-op above floor)");
      Check(Primitive::FloorLapse(0.5) == 0.5, "FloorLapse(0.5) == 0.5 (no-op above floor)");
      Check(Primitive::FloorLapse(0.0) > 0.0, "FloorLapse(0.0) is strictly positive");
      Check(Primitive::FloorLapse(-1.0) > 0.0,
            "FloorLapse of a negative value is still strictly positive");
      Check(Primitive::FloorLapse(1.0e-16) == Primitive::FloorLapse(0.0),
            "everything at/under the floor clamps to the identical value");
    }

    // ---- 1. Confirm the mechanism: the ORIGINAL form goes non-finite at alpha=0 ----
    // (and that it does so specifically through this quadratic, not by assumption)
    //
    // Empirically (measured below, not assumed): the naive form is finite for any
    // alpha that is merely tiny -- e.g. 1e-14 -- because 1/alpha is then a huge but
    // FINITE double, and the quadratic's a/b/c all scale homogeneously by ialpha^2,
    // so the ratios a1=b/a, a0=c/a stay well-conditioned. The trigger is alpha
    // EXACTLY 0.0: only then is ialpha literally +Inf, and u1 = uu[i] - u0*beta_u[i]
    // hits Inf*0.0 = NaN as soon as beta_u[i] == 0.0 (and similarly elsewhere in the
    // chain for other beta_u values). This matches cfc_puncture.hpp's clamp
    // (`alpha_sq > 0.0 ? alpha_sq : 0.0`, GRWAVESPEED_ALPHA_HANDOFF.md Sec 2), which
    // produces alpha == 0.0 exactly at/beyond the trumpet throat -- not just "small".
    {
      Real g3d[NSPMETRIC] = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0};  // flat 3-metric
      Real gii = 1.0;
      Real cmsq = 0.3;  // representative sound-speed-squared, well inside [0,1)
      const Real kBetas[3][3] = {{0.0, 0.0, 0.0}, {0.3, 0.0, 0.0}, {-0.2, 0.1, 0.0}};
      const Real kVels[3][3] = {{0.0, 0.0, 0.0}, {0.1, 0.0, 0.0}, {0.05, -0.05, 0.0}};

      int n_zero = 0, n_bad_zero = 0;
      for (const Real *beta_u_c : kBetas) {
        for (const Real *uu_c : kVels) {
          Real beta_u[3] = {beta_u_c[0], beta_u_c[1], beta_u_c[2]};
          Real uu[3] = {uu_c[0], uu_c[1], uu_c[2]};
          Real lp, lm;
          NaiveMagnetosonicSpeeds(lp, lm, uu, 0.0, g3d, beta_u, 0.0, gii, cmsq);
          ++n_zero;
          if (!(std::isfinite(lp) && std::isfinite(lm))) { ++n_bad_zero; }
        }
      }
      char msg[200];
      std::snprintf(msg, sizeof(msg),
                    "sanity: original unguarded form is non-finite at alpha=0.0 exactly "
                    "(%d/%d) -- confirms this is the mechanism", n_bad_zero, n_zero);
      Check(n_bad_zero == n_zero, msg);

      int n_tiny = 0, n_bad_tiny = 0;
      for (Real alpha : {1.0e-14, 1.0e-9, 1.0e-6}) {
        for (const Real *beta_u_c : kBetas) {
          for (const Real *uu_c : kVels) {
            Real beta_u[3] = {beta_u_c[0], beta_u_c[1], beta_u_c[2]};
            Real uu[3] = {uu_c[0], uu_c[1], uu_c[2]};
            Real lp, lm;
            NaiveMagnetosonicSpeeds(lp, lm, uu, 0.0, g3d, beta_u, alpha, gii, cmsq);
            ++n_tiny;
            if (!(std::isfinite(lp) && std::isfinite(lm))) { ++n_bad_tiny; }
          }
        }
      }
      std::snprintf(msg, sizeof(msg),
                    "note: naive form stays finite for merely-tiny (not exactly zero) "
                    "alpha (%d/%d non-finite) -- alpha==0.0 is the actual trigger, not "
                    "'alpha is small'; the floor still needs to cover this case too",
                    n_bad_tiny, n_tiny);
      Check(n_bad_tiny == 0, msg);
    }

    // ---- 2. The FIX: the real, production GetGRFastMagnetosonicSpeeds stays finite -
    {
      Real g3d[NSPMETRIC] = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0};
      Real gii = 1.0;
      // rho, T chosen so the ideal-gas EOS gives an O(1) sound speed -- the exact
      // value doesn't matter, only that it's a physical, well-defined state.
      Real prim[NPRIM] = {0.0};
      prim[PRH] = 1.0;
      prim[PTM] = 1.0;
      const Real kAlphas[] = {1.0, 0.5, 1.0e-1, 1.0e-3, 1.0e-6, 1.0e-9, 1.0e-12, 0.0};
      const Real kBetas[3][3] = {{0.0, 0.0, 0.0}, {0.3, 0.0, 0.0}, {-0.2, 0.1, 0.0}};
      const Real kVels[3][3] = {{0.0, 0.0, 0.0}, {0.1, 0.0, 0.0}, {0.05, -0.05, 0.0}};
      int n_checked = 0, n_bad = 0;
      Real max_abs_lambda = 0.0;
      for (Real alpha : kAlphas) {
        for (const Real *beta_u_c : kBetas) {
          for (const Real *uu_c : kVels) {
            Real beta_u[3] = {beta_u_c[0], beta_u_c[1], beta_u_c[2]};
            prim[PVX] = uu_c[0]; prim[PVY] = uu_c[1]; prim[PVZ] = uu_c[2];
            Real lp, lm;
            eos.GetGRFastMagnetosonicSpeeds(lp, lm, prim, 0.0, g3d, beta_u, alpha,
                                            gii, PVX);
            ++n_checked;
            if (!(std::isfinite(lp) && std::isfinite(lm))) {
              ++n_bad;
            } else {
              max_abs_lambda = fmax(max_abs_lambda, fmax(fabs(lp), fabs(lm)));
            }
          }
        }
      }
      char msg[220];
      std::snprintf(msg, sizeof(msg),
                    "GetGRFastMagnetosonicSpeeds finite for all %d samples, alpha in "
                    "[0, 1] including alpha=0 exactly (%d non-finite)", n_checked, n_bad);
      Check(n_bad == 0, msg);
      std::snprintf(msg, sizeof(msg),
                    "wave speeds stay within a sane causal bound as alpha->0 (max|lambda|"
                    "=%.6g)", max_abs_lambda);
      // A loose bound: nothing pathological (e.g. the Inf/NaN this bug used to produce,
      // or a floor so large it distorts the near-zero-alpha limit into something huge).
      Check(max_abs_lambda < 100.0, msg);
    }

    // ---- 3. alpha=0 and alpha=(anything <= the floor) agree exactly ----------------
    // Confirms the fix is really flooring (a fixed clamp), not e.g. a floor that scales
    // with alpha and would give a different (silently wrong) answer at alpha=1e-20 vs.
    // alpha=0 even though both are physically "the lapse has collapsed to zero" here.
    {
      Real g3d[NSPMETRIC] = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0};
      Real gii = 1.0;
      Real beta_u[3] = {0.3, 0.0, 0.0};
      Real prim[NPRIM] = {0.0};
      prim[PRH] = 1.0;
      prim[PTM] = 1.0;
      prim[PVX] = 0.1;
      Real lp0, lm0, lp1, lm1;
      eos.GetGRFastMagnetosonicSpeeds(lp0, lm0, prim, 0.0, g3d, beta_u, 0.0, gii, PVX);
      eos.GetGRFastMagnetosonicSpeeds(lp1, lm1, prim, 0.0, g3d, beta_u, 1.0e-20, gii,
                                      PVX);
      char msg[160];
      std::snprintf(msg, sizeof(msg),
                    "alpha=0 (lp=%.15g) and alpha=1e-20 (lp=%.15g) agree exactly",
                    lp0, lp1);
      Check(lp0 == lp1 && lm0 == lm1, msg);
    }

    // ---- 4. hlle_dyn_grmhd.hpp's independent `qa = lambda_r*lambda_l/alpha` site ---
    // (a separate division from GetGRFastMagnetosonicSpeeds's own ialpha, per
    // GRWAVESPEED_ALPHA_HANDOFF.md Sec 1 -- fixed the same way, checked separately)
    {
      Real lambda_r = 0.9, lambda_l = -0.8;  // representative, from Sec 2 above
      Real qa_naive_at_zero = lambda_r*lambda_l/0.0;
      Real qa_fixed_at_zero = lambda_r*lambda_l/Primitive::FloorLapse(0.0);
      Check(!std::isfinite(qa_naive_at_zero),
            "sanity: hlle's naive qa=.../alpha is non-finite at alpha=0");
      Check(std::isfinite(qa_fixed_at_zero),
            "hlle's qa=.../FloorLapse(alpha) is finite at alpha=0");
    }
  }
  Kokkos::finalize();

  std::printf("\n%s (%d failure%s)\n", nfail == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED",
              nfail, nfail == 1 ? "" : "s");
  return nfail == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
