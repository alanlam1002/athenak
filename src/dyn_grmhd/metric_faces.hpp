#ifndef DYN_GRMHD_METRIC_FACES_HPP_
#define DYN_GRMHD_METRIC_FACES_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file metric_faces.hpp
//! \brief <dyn_grmhd> metric_terms (R-140 item 1, research PHASE1_SPECS.md S-A): the hydro's
//! view of the CFC + trumpet-puncture metric.
//!   cell         -- the present scheme (default, untouched): fluxes take the 2-point
//!                   average of the assembled cell metric (adm.hpp Face1Metric), sources
//!                   Dx<NGHOST> of it (dyn_grmhd.cpp AddCoordTermsEOS).
//!   analytic_bg  -- option (d): fluxes take the face metric below (exact trumpet
//!                   background at the face + interpolated CFC residual); sources take the
//!                   analytic background derivative + Dx<NGHOST> of the residual.
//!   wellbalanced -- option (e): fluxes as (d); sources difference the SAME face values,
//!                   d_d g|_i = (g_f(i+1/2) - g_f(i-1/2))/dx_d.
//!   kerr_schild  -- R-188 item 3 (Step B): a FIXED analytic Kerr-Schild metric (<adm> without
//!                   CFC/Z4c: ADM::SetADMVariablesToKerrSchild at the cells); the faces take the
//!                   same closed form (ComputeADMDecomposition) at the face, full gamma_ij; the
//!                   sources difference the face values as wellbalanced.
//! The face metric is built once per stage (DynGRMHD::ComputeFaceMetric) into per-direction
//! arrays fm_x{1,2,3}(m, n, k, j, i) with n = 0 alpha, 1-3 beta^i, 4 psi, 5-10 gamma_ij
//! (xx, xy, xz, yy, yz, zz; the trumpet modes: psi^4 delta_ij), read by the Riemann solvers,
//! FOFC and the sources alike.
//!
//! The background at a face needs the areal radius R(r_iso), a root-find per point
//! (cfc_puncture.hpp TrumpetIsoToAreal). TrumpetTable replaces it by a quintic Hermite
//! table of rho = R/M against w = sqrt(r_iso/M) (uniform in w; rho - 3/2 ~ r^sqrt2 at the
//! throat), with the w-derivatives in closed form from dR/dr_iso = R alpha0/r_iso:
//! rho_w = 2 rho alpha0/w, alpha0_w = 2 (1 - 3.375/rho^3)/(rho w), rho_ww = 2 [(rho_w alpha0
//! + rho alpha0_w)/w - rho alpha0/w^2], alpha0 = (2 rho - 3) s/(4 rho^2), s = sqrt(4 rho^2 +
//! 4 rho + 3). Evaluating it takes only sqrt and rational arithmetic: no log/exp, whose
//! vectorised (SVML) and scalar versions differ in the last bit on the CPU and broke the
//! ghost/owner identity of the face metric in the first CPU probe (c122). psi0, alpha0,
//! beta0 and their radial derivatives then follow from TrumpetBackground with that R (the
//! closed forms the cell fill uses).

#include <cmath>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "cfc/cfc_puncture.hpp"

namespace dyngr {

enum class MetricTerms {cell, analytic_bg, wellbalanced, kerr_schild};

// face-array components
constexpr int kFmAlpha = 0, kFmBeta = 1, kFmPsi = 4, kFmG = 5, kFmN = 11;

//----------------------------------------------------------------------------------------
//! \struct TrumpetTable
//! \brief rho(w), w = sqrt(r_iso/M), on w = k h, k = 0..n-1; d(k, 0..2) = rho, rho_w,
//! rho_ww. Beyond the last node rho = x + 1 + 1/(4x) (x = r_iso/M; the large-r expansion).
//! Built on the host (BuildTrumpetTable in dyn_grmhd_metric.cpp, long double Newton on the
//! closed-form chi(rho)), copied to the device once.
struct TrumpetTable {
  DvceArray2D<Real> d;
  Real h = 1.0, ih = 1.0;
  int n = 0;
};

//! host build (dyn_grmhd_metric.cpp): nodes w = k h up to w_max
void BuildTrumpetTable(TrumpetTable &t, Real w_max, Real h);

//! rho = R/M at x = r_iso/M
template<class TableView>
KOKKOS_INLINE_FUNCTION
Real TrumpetTableRho(const TableView &d, const Real h, const Real ih, const int n,
                     const Real x) {
  const Real s = sqrt(x)*ih;
  if (s >= static_cast<Real>(n - 1)) {
    return x + 1.0 + 0.25/x;
  }
  const int k = static_cast<int>(s);
  const Real t = s - static_cast<Real>(k);
  const Real t2 = t*t, t3 = t2*t, t4 = t3*t, t5 = t4*t;
  const Real h00 = 1.0 - 10.0*t3 + 15.0*t4 - 6.0*t5;
  const Real h10 = t - 6.0*t3 + 8.0*t4 - 3.0*t5;
  const Real h20 = 0.5*(t2 - 3.0*t3 + 3.0*t4 - t5);
  const Real h01 = 10.0*t3 - 15.0*t4 + 6.0*t5;
  const Real h11 = -4.0*t3 + 7.0*t4 - 3.0*t5;
  const Real h21 = 0.5*(t3 - 2.0*t4 + t5);
  return h00*d(k, 0) + h10*h*d(k, 1) + h20*h*h*d(k, 2)
       + h01*d(k + 1, 0) + h11*h*d(k + 1, 1) + h21*h*h*d(k + 1, 2);
}

//! the trumpet background at x (relative to the hole) through the table:
//! psi0, (alpha psi)0, beta0^i, and the isotropic radial derivatives dpsi0/dr, dalpha0/dr,
//! with beta0^i = b(r) x^i, db/dr = -3 b alpha0/r.
KOKKOS_INLINE_FUNCTION
void TrumpetTableBackground(const DvceArray2D<Real> &d, const Real h, const Real ih,
                            const int n, const Real m_bh,
                            const Real x1, const Real x2, const Real x3,
                            Real &psi0, Real &ap0, Real beta0[3],
                            Real &dpsi0, Real &dalpha0, Real &b, Real &db) {
  const Real r = sqrt(x1*x1 + x2*x2 + x3*x3 + 1.0e-30);
  const Real rho = TrumpetTableRho(d, h, ih, n, r/m_bh);
  Real alpha0, Aij0[6], a2;
  cfc::TrumpetBackground(m_bh, x1, x2, x3, m_bh*rho, &psi0, &alpha0, beta0, Aij0, &a2,
                         &dpsi0, &dalpha0);
  ap0 = alpha0*psi0;
  b = 0.75*sqrt(3.0)/(m_bh*rho*rho*rho);
  db = -3.0*b*alpha0/r;
}

//----------------------------------------------------------------------------------------
//! \fn void TrumpetFaceDiff
//! \brief <dyn_grmhd> killing_momentum_bg = face_diff (R-178): the background's centred face
//! difference along one direction, (g0(xp) - g0(xm))*idx, of alpha0, beta0^b and psi0^4, with
//! xm, xp the cell's two faces (relative to the hole). The values are killing_t1's face
//! weights' (dyn_grmhd_killing.cpp KillingBackground: the same table and closed-form calls)
KOKKOS_INLINE_FUNCTION
void TrumpetFaceDiff(const DvceArray2D<Real> &d, const Real h, const Real ih, const int n,
                     const Real m_bh, const Real xm[3], const Real xp[3], const Real idx,
                     Real &dal0, Real db0[3], Real &dp40) {
  Real ps[2], al[2], be[2][3];
  for (int s = 0; s < 2; ++s) {
    const Real *x = s ? xp : xm;
    const Real r = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + 1.0e-30);
    const Real rho = TrumpetTableRho(d, h, ih, n, r/m_bh);
    Real A0[6], a2, dpsi0, dalpha0;
    cfc::TrumpetBackground(m_bh, x[0], x[1], x[2], m_bh*rho, &ps[s], &al[s], be[s], A0, &a2,
                           &dpsi0, &dalpha0);
  }
  dal0 = (al[1] - al[0])*idx;
  for (int b = 0; b < 3; ++b) db0[b] = (be[1][b] - be[0][b])*idx;
  dp40 = ((ps[1]*ps[1])*(ps[1]*ps[1]) - (ps[0]*ps[0])*(ps[0]*ps[0]))*idx;
}

//----------------------------------------------------------------------------------------
//! \fn void FaceMetricFromArrays
//! \brief the Riemann solvers' face metric from the per-stage face arrays (direction
//! ivx - IVX); same outputs as adm::Face1Metric etc.
template<int ivx>
KOKKOS_INLINE_FUNCTION
void FaceMetricFromArrays(const DvceArray5D<Real> &fm, const int m, const int k,
                          const int j, const int i, Real g3d[NSPMETRIC], Real beta_u[3],
                          Real &alpha) {
  alpha = fm(m, kFmAlpha, k, j, i);
  for (int a = 0; a < 3; ++a) beta_u[a] = fm(m, kFmBeta + a, k, j, i);
  // R-188 item 3: the full gamma_ij (the trumpet modes store psi^4 delta_ij, the same
  // arithmetic as before, at the fill)
  g3d[S11] = fm(m, kFmG + 0, k, j, i); g3d[S12] = fm(m, kFmG + 1, k, j, i);
  g3d[S13] = fm(m, kFmG + 2, k, j, i); g3d[S22] = fm(m, kFmG + 3, k, j, i);
  g3d[S23] = fm(m, kFmG + 4, k, j, i); g3d[S33] = fm(m, kFmG + 5, k, j, i);
}

}  // namespace dyngr

#endif  // DYN_GRMHD_METRIC_FACES_HPP_
