#ifndef DYN_GRMHD_RESISTIVE_RESISTIVE_DIAGNOSTICS_HPP_
#define DYN_GRMHD_RESISTIVE_RESISTIVE_DIAGNOSTICS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file resistive_diagnostics.hpp
//! \brief Diagnostics for resistive GRMHD (07_scoping_v2 §4 step 0, D4).
//!
//!  * TotalCharge     : Q = sum_cells dV q~_c,  q~_c = sum_d (E~_d(+) - E~_d(-))/dx_d
//!                      (q~ == div E~ is diagnosed, decision 5; Gauss's law then holds by
//!                      construction and the "Gauss residual" is the discrete charge
//!                      balance Q^{n+1} - Q^n + dt * (boundary charge flux), test C2).
//!  * BoundaryFlux    : sum over the outer faces of each MeshBlock of +-E~ dA; equals Q
//!                      to roundoff (discrete divergence theorem).
//!  * MaxDivB         : max_cells |div B~| (densitised face B), test C1.
//!  * ECellFaceMismatch: max and volume-mean |E_c - avg(E_f)| per component (D4).
//!  * NewtonStats     : device counters for the cell Newton (iterations, fallbacks,
//!                      failures).
//! All reductions are over the local MeshBlockPack and, with MPI, over all ranks.

#include "athena.hpp"

class MeshBlockPack;

namespace resistive {

struct ChargeDiag {
  Real total;      // sum dV q~
  Real boundary;   // sum of outward face E~ dA over MeshBlock boundaries
  Real max_abs_q;  // max |q~_c|
};

struct MismatchDiag {
  Real max_abs;
  Real mean_abs;
};

ChargeDiag TotalCharge(MeshBlockPack *pmbp, const DvceFaceFld4D<Real> &ef);
Real MaxDivB(MeshBlockPack *pmbp, const DvceFaceFld4D<Real> &bf);
MismatchDiag ECellFaceMismatch(MeshBlockPack *pmbp, const DvceArray5D<Real> &ec,
                               const DvceFaceFld4D<Real> &ef);

KOKKOS_INLINE_FUNCTION
Real ChargeBalanceResidual(const Real q_new, const Real q_old,
                           const Real dt_boundary_outflux) {
  return q_new - q_old + dt_boundary_outflux;
}

// Newton statistics: counters live on device and are updated with atomics.
enum NewtonCounter {kNewtonCells = 0, kNewtonIterSum = 1, kNewtonIterMax = 2,
                    kNewtonFallback = 3, kNewtonFail = 4, kNewtonNCounters = 5};

struct NewtonStats {
  DvceArray1D<int64_t> d;
  NewtonStats() : d("resist_newton_stats", kNewtonNCounters) {}
  void Reset() const { Kokkos::deep_copy(d, 0); }
  KOKKOS_INLINE_FUNCTION
  void Record(const int iters, const bool fallback, const bool fail) const {
    Kokkos::atomic_add(&d(kNewtonCells), static_cast<int64_t>(1));
    Kokkos::atomic_add(&d(kNewtonIterSum), static_cast<int64_t>(iters));
    Kokkos::atomic_max(&d(kNewtonIterMax), static_cast<int64_t>(iters));
    if (fallback) Kokkos::atomic_add(&d(kNewtonFallback), static_cast<int64_t>(1));
    if (fail) Kokkos::atomic_add(&d(kNewtonFail), static_cast<int64_t>(1));
  }
  // Host copy, reduced over MPI ranks.
  void Read(int64_t out[kNewtonNCounters]) const;
};

}  // namespace resistive

#endif  // DYN_GRMHD_RESISTIVE_RESISTIVE_DIAGNOSTICS_HPP_
