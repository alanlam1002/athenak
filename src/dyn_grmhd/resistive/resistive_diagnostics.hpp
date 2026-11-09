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
//!  * SigmaMHist      : K5-run per-sigma_M-decade histogram of the implicit solve.
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
  // Restart: rank 0 holds the MPI-reduced totals, other ranks start from zero.
  void Load(const int64_t in[kNewtonNCounters]) const;
};

// K5-run histogram (07 v2.2 §4.1, decision 10; v2.5 decision 16): counters per decade
// of the magnetisation, interior cells only, accumulated over an output interval, in two
// binnings with the same edges: kBinSigM by sigma_M = B^2/rho (Eulerian B^2), kBinMagn by
// b^2/(rho h) (comoving b). Bins: <1e-2, [1e-2,1e-1), ..., [1e3,1e4), >=1e4.
enum HistCounter {kHCells = 0, kHConverged = 1, kHRetry = 2, kHFallback = 3,
                  kHPrestep = 4, kHCeiling = 5, kHFloor = 6, kHEgtB = 7, kHContinued = 8,
                  kHNCounters = 9};
enum HistBinning {kBinSigM = 0, kBinMagn = 1, kHistBinnings = 2};
constexpr int kHistBins = 8;

struct SigmaMHist {
  DvceArray3D<int64_t> d;
  SigmaMHist() : d("resist_sigm_hist", kHistBinnings, kHistBins, kHNCounters) {}
  void Reset() const { Kokkos::deep_copy(d, 0); }
  KOKKOS_INLINE_FUNCTION
  static int Bin(const Real b2, const Real rho) {
    if (!(rho > 0.0) || !(b2 > 0.0)) return (b2 > 0.0) ? kHistBins - 1 : 0;
    const int b = static_cast<int>(floor(log10(b2/rho))) + 3;
    return (b < 0) ? 0 : ((b > kHistBins - 1) ? kHistBins - 1 : b);
  }
  // Counts an event in bin bs of the sigma_M binning and bin bm of the b^2/(rho h) one.
  KOKKOS_INLINE_FUNCTION
  void Add(const int bs, const int bm, const int c) const {
    Kokkos::atomic_add(&d(kBinSigM, bs, c), static_cast<int64_t>(1));
    Kokkos::atomic_add(&d(kBinMagn, bm, c), static_cast<int64_t>(1));
  }
  // Host copy summed over MPI ranks.
  void Read(int64_t out[kHistBinnings][kHistBins][kHNCounters]) const;
  // Restart: rank 0 holds the MPI-reduced counts, other ranks start from zero.
  void Load(const int64_t in[kHistBinnings][kHistBins][kHNCounters]) const;
};

}  // namespace resistive

#endif  // DYN_GRMHD_RESISTIVE_RESISTIVE_DIAGNOSTICS_HPP_
