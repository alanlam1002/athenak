#ifndef DYN_GRMHD_RESISTIVE_RESISTIVE_MESH_UTIL_HPP_
#define DYN_GRMHD_RESISTIVE_RESISTIVE_MESH_UTIL_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file resistive_mesh_util.hpp
//! \brief Small device helpers shared by the resistive mesh kernels: ADM loads and
//!        averages (07 D1), inverse-metric components and the PLM slope limiter.

#include <cmath>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "eos/primitive-solver/ps_types.hpp"

namespace resistive {

struct PointMetric {
  Real g[NSPMETRIC];
  Real alpha;
  Real beta[3];
};

KOKKOS_INLINE_FUNCTION
void LoadMetric(const adm::ADM::ADM_vars &adm, const int m, const int k, const int j,
                const int i, PointMetric &pm) {
  pm.g[S11] = adm.g_dd(m, 0, 0, k, j, i);
  pm.g[S12] = adm.g_dd(m, 0, 1, k, j, i);
  pm.g[S13] = adm.g_dd(m, 0, 2, k, j, i);
  pm.g[S22] = adm.g_dd(m, 1, 1, k, j, i);
  pm.g[S23] = adm.g_dd(m, 1, 2, k, j, i);
  pm.g[S33] = adm.g_dd(m, 2, 2, k, j, i);
  pm.alpha = adm.alpha(m, k, j, i);
  for (int a = 0; a < 3; ++a) pm.beta[a] = adm.beta_u(m, a, k, j, i);
}

// pm = (sum of n loaded metrics)/n, accumulated by the caller with AddMetric.
KOKKOS_INLINE_FUNCTION
void ZeroMetric(PointMetric &pm) {
  for (int n = 0; n < NSPMETRIC; ++n) pm.g[n] = 0.0;
  pm.alpha = 0.0;
  for (int a = 0; a < 3; ++a) pm.beta[a] = 0.0;
}

KOKKOS_INLINE_FUNCTION
void AddMetric(const adm::ADM::ADM_vars &adm, const int m, const int k, const int j,
               const int i, const Real w, PointMetric &pm) {
  PointMetric q;
  LoadMetric(adm, m, k, j, i, q);
  for (int n = 0; n < NSPMETRIC; ++n) pm.g[n] += w*q.g[n];
  pm.alpha += w*q.alpha;
  for (int a = 0; a < 3; ++a) pm.beta[a] += w*q.beta[a];
}

KOKKOS_INLINE_FUNCTION
Real MetricDet(const Real g[NSPMETRIC]) {
  return g[S11]*g[S22]*g[S33] + 2.0*g[S12]*g[S13]*g[S23]
       - (g[S11]*g[S23]*g[S23] + g[S12]*g[S12]*g[S33] + g[S13]*g[S13]*g[S22]);
}

KOKKOS_INLINE_FUNCTION
void MetricInverse(const Real g[NSPMETRIC], Real gu[NSPMETRIC]) {
  const Real idet = 1.0/MetricDet(g);
  gu[S11] = (g[S22]*g[S33] - g[S23]*g[S23])*idet;
  gu[S12] = (g[S13]*g[S23] - g[S12]*g[S33])*idet;
  gu[S13] = (g[S12]*g[S23] - g[S13]*g[S22])*idet;
  gu[S22] = (g[S11]*g[S33] - g[S13]*g[S13])*idet;
  gu[S23] = (g[S12]*g[S13] - g[S11]*g[S23])*idet;
  gu[S33] = (g[S11]*g[S22] - g[S12]*g[S12])*idet;
}

KOKKOS_INLINE_FUNCTION
Real InverseDiag(const Real gu[NSPMETRIC], const int n) {
  return (n == 0) ? gu[S11] : ((n == 1) ? gu[S22] : gu[S33]);
}

// van Leer (harmonic) limited slope; zero at extrema and for zero differences.
KOKKOS_INLINE_FUNCTION
Real VanLeerSlope(const Real dqm, const Real dqp) {
  const Real p = dqm*dqp;
  return (p > 0.0) ? 2.0*p/(dqm + dqp) : 0.0;
}

}  // namespace resistive

#endif  // DYN_GRMHD_RESISTIVE_RESISTIVE_MESH_UTIL_HPP_
