#ifndef PGEN_DYN_GRMHD_RNS_GRASS_MAGNETIC_WEB_HPP_
#define PGEN_DYN_GRMHD_RNS_GRASS_MAGNETIC_WEB_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file grass_magnetic_web.hpp
//  \brief Pure math for the "magnetic web" + dipole initial magnetic field, per
//  magnetic_web_id_athenak.md (repo root); physics from Skoutnev & Beloborodov 2025
//  (arXiv:2504.07223). Array-touching orchestration (host loops, curl kernels, MPI
//  reductions) lives in dyngr_grass.cpp's BuildMagneticField(); this header holds only
//  the field-defining formulas.
//
//  Density confinement envelope: a C^inf bump vanishing (with all derivatives) at both
//  edges of a mass-shell window rho_lo < rho < rho_hi, so B vanishes identically
//  outside the shell:
//
//    u(rho) = 2*ln(rho/rho_lo)/ln(rho_hi/rho_lo) - 1
//    h(rho) = exp(1 - 1/(1-u^2))   if |u| < 1,   else 0
//
//  Web vector potential (Chandrasekhar poloidal-toroidal decomposition; envelope
//  applied to A itself, so no derivatives of rho are needed):
//
//    A_web_i(x) = h(rho) * [ lambda_T*Psi_T*x_i + lambda_P*eps_ijk (d_j Psi_P) x_k ]
//
//  WebA() below returns the bracket alone (not multiplied by h) -- dyngr_grass.cpp
//  caches h(rho) once per grid point and reuses it across the poloidal/toroidal/
//  dipole passes.
//
//  Psi_P, Psi_T are random-phase plane-wave superpositions over N modes (generated
//  once on the host with a fixed seed, identical on every rank -- see
//  GenerateWebModeTable()):
//
//    Psi_P(x) = sum_p a_p * sin(k_p . x + phi_p)
//    Psi_T(x) = mu_star * sum_p k_p * a_p * sin(k_p . x + phi_p)     (k_p = |k_p|)
//    d_j Psi_P = sum_p a_p * k_{p,j} * cos(k_p . x + phi_p)
//
//  Psi_T reuses Psi_P's amplitudes/phases, scaled by mu_star*k_p -- this makes
//  helicity one-signed at every scale and ties E_tor/E_pol to mu_star, only
//  approximately (the exact ratio is enforced numerically by dyngr_grass.cpp's
//  quadratic solve).
//
//  Dipole: DipoleA1/DipoleA2 are the classic current-loop potential, ported verbatim
//  from lorene_bns.cpp's A1/A2 (center=0, one star at the origin). dipole_confine=1
//  multiplies it by the same h(rho) envelope as the web.

#include <cmath>
#include <cstdint>
#include <random>

#include "athena.hpp"
#include "parameter_input.hpp"

namespace grass {

//----------------------------------------------------------------------------------------
//! \fn Real WebEnvelope
//! \brief h(rho): the C^inf mass-shell bump function. rho, rho_lo, rho_hi in the
//  SAME units (this pgen: AthenaK code-unit mass density). Guards rho<=0 (never
//  calls log of a non-positive value) and rho_hi<=rho_lo (misconfigured window).

inline Real WebEnvelope(Real rho, Real rho_lo, Real rho_hi) {
  if (rho <= 0.0 || rho_hi <= rho_lo) { return 0.0; }
  Real u = 2.0*std::log(rho/rho_lo)/std::log(rho_hi/rho_lo) - 1.0;
  if (std::abs(u) >= 1.0) { return 0.0; }
  return std::exp(1.0 - 1.0/(1.0 - u*u));
}

//----------------------------------------------------------------------------------------
//! \struct WebModeTable
//! \brief Fixed-size storage for the N-mode random-phase plane-wave table (host-only;
//  the vector-potential construction never leaves the host, see grass_reader.hpp).

struct WebModeTable {
  static constexpr int kMaxModes = 512;
  int nmodes = 0;
  Real kx[kMaxModes], ky[kMaxModes], kz[kMaxModes];  // mode wavevector, Cartesian
  Real k[kMaxModes];                                  // |k_p|
  Real amp[kMaxModes];                                // a_p = zeta_p * k_p^(-alpha)
  Real phase[kMaxModes];                              // phi_p
};

//----------------------------------------------------------------------------------------
//! \fn void GenerateWebModeTable
//! \brief Fills a WebModeTable from <problem> web_nmodes/web_kmin/web_kmax/
//  web_alpha/web_seed. Depends on nothing rank- or mesh-local, so every rank builds
//  the identical table with no broadcast needed. Draw order per mode is the
//  reproducibility contract -- do not reorder: k_p (log-uniform) -> direction
//  (cos(theta)~U(-1,1), phi~U(0,2pi)) -> phase (~U(0,2pi)) -> zeta_p~N(0,1) ->
//  amplitude a_p = zeta_p*k_p^(-alpha).

inline void GenerateWebModeTable(ParameterInput *pin, WebModeTable *table) {
  int nmodes = pin->GetOrAddInteger("problem", "web_nmodes", 128);
  nmodes = std::min(nmodes, WebModeTable::kMaxModes);
  Real kmin = pin->GetReal("problem", "web_kmin");
  Real kmax = pin->GetReal("problem", "web_kmax");
  Real alpha = pin->GetOrAddReal("problem", "web_alpha", 1.0);
  int64_t seed = pin->GetOrAddInteger("problem", "web_seed", 20260808);

  table->nmodes = nmodes;
  std::mt19937_64 rng(static_cast<uint64_t>(seed));
  std::uniform_real_distribution<Real> uniform01(0.0, 1.0);
  std::uniform_real_distribution<Real> uniform_2pi(0.0, 2.0*M_PI);
  std::normal_distribution<Real> normal01(0.0, 1.0);

  Real lnkmin = std::log(kmin), lnkmax = std::log(kmax);
  for (int p = 0; p < nmodes; ++p) {
    Real u_k = uniform01(rng);
    Real k_p = std::exp(lnkmin + u_k*(lnkmax - lnkmin));
    Real costheta = 2.0*uniform01(rng) - 1.0;
    Real phi_k = uniform_2pi(rng);
    Real phase = uniform_2pi(rng);
    Real zeta = normal01(rng);
    Real amp = zeta * std::pow(k_p, -alpha);

    Real sintheta = std::sqrt(std::max(0.0, 1.0 - costheta*costheta));
    table->kx[p] = k_p*sintheta*std::cos(phi_k);
    table->ky[p] = k_p*sintheta*std::sin(phi_k);
    table->kz[p] = k_p*costheta;
    table->k[p] = k_p;
    table->amp[p] = amp;
    table->phase[p] = phase;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real PsiP / Real PsiT / void DPsiP
//! \brief The plane-wave-superposition potentials and Psi_P's gradient, per the
//  formulas in the file header. Pure sums over table.nmodes -- cheap, called
//  many times per point (once per WebA() call).

inline Real PsiP(Real x, Real y, Real z, const WebModeTable &table) {
  Real sum = 0.0;
  for (int p = 0; p < table.nmodes; ++p) {
    Real kdotx = table.kx[p]*x + table.ky[p]*y + table.kz[p]*z;
    sum += table.amp[p] * std::sin(kdotx + table.phase[p]);
  }
  return sum;
}

inline Real PsiT(Real x, Real y, Real z, const WebModeTable &table, Real mu_star) {
  Real sum = 0.0;
  for (int p = 0; p < table.nmodes; ++p) {
    Real kdotx = table.kx[p]*x + table.ky[p]*y + table.kz[p]*z;
    sum += table.k[p] * table.amp[p] * std::sin(kdotx + table.phase[p]);
  }
  return mu_star * sum;
}

inline void DPsiP(Real x, Real y, Real z, const WebModeTable &table,
                   Real *dpx, Real *dpy, Real *dpz) {
  Real sx = 0.0, sy = 0.0, sz = 0.0;
  for (int p = 0; p < table.nmodes; ++p) {
    Real kdotx = table.kx[p]*x + table.ky[p]*y + table.kz[p]*z;
    Real c = table.amp[p] * std::cos(kdotx + table.phase[p]);
    sx += c*table.kx[p];
    sy += c*table.ky[p];
    sz += c*table.kz[p];
  }
  *dpx = sx; *dpy = sy; *dpz = sz;
}

//----------------------------------------------------------------------------------------
//! \fn void WebA
//! \brief The web vector potential's BRACKET (no h(rho) factor -- see file
//  header for why the caller applies h itself):
//    [ lambda_T*Psi_T*x_i + lambda_P*eps_ijk (d_j Psi_P) x_k ]

inline void WebA(Real x, Real y, Real z, const WebModeTable &table, Real mu_star,
                  Real lam_T, Real lam_P, Real *ax, Real *ay, Real *az) {
  Real psiT = PsiT(x, y, z, table, mu_star);
  Real dpx, dpy, dpz;
  DPsiP(x, y, z, table, &dpx, &dpy, &dpz);
  *ax = lam_T*psiT*x + lam_P*(dpy*z - dpz*y);
  *ay = lam_T*psiT*y + lam_P*(dpz*x - dpx*z);
  *az = lam_T*psiT*z + lam_P*(dpx*y - dpy*x);
}

//----------------------------------------------------------------------------------------
//! \fn Real DipoleA1 / Real DipoleA2
//! \brief Current-loop dipole vector-potential components, ported verbatim from
//  lorene_bns.cpp's A1/A2 (center=0, one star at the origin; caller passes x,y,z
//  already centered). I_0 is the loop current, matching lorene_bns.cpp's own
//  I_0 = 4*r_0*b_max/(23*pi). a3 is identically zero (a current loop in the x-y
//  plane has no z-component of A), so unlike the web the dipole needs no a3 AMR
//  correction.

inline Real DipoleA1(Real x, Real y, Real z, Real I_0, Real r_0) {
  Real w2 = SQR(x) + SQR(y);
  Real r2 = w2 + SQR(z);
  return -y * M_PI * SQR(r_0)*I_0 / std::pow(SQR(r_0) + r2, 1.5) *
         (1.0 + 15.0/8.0*SQR(r_0)*(SQR(r_0)+w2)/SQR(SQR(r_0)+r2));
}

inline Real DipoleA2(Real x, Real y, Real z, Real I_0, Real r_0) {
  Real w2 = SQR(x) + SQR(y);
  Real r2 = w2 + SQR(z);
  return x * M_PI * SQR(r_0)*I_0 / std::pow(SQR(r_0) + r2, 1.5) *
         (1.0 + 15.0/8.0*SQR(r_0)*(SQR(r_0)+w2)/SQR(SQR(r_0)+r2));
}

}  // namespace grass

#endif  // PGEN_DYN_GRMHD_RNS_GRASS_MAGNETIC_WEB_HPP_
