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
                  // 07 v2.8 §3.2.3(e) fallback breakdown (fallback = any non-Newton exit)
                  kHFA = 9, kHFAReject = 10, kHBlend = 11, kHIL = 12, kHILFloor = 13,
                  kHKeep = 14, kHNCounters = 15};
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

// 07 v2.8 §3.2.3(d,e) prototype log histograms, per b^2/(rho h) decade (kBinMagn edges):
// eps_D4 = |avg(E_f) - E_cell|/|B| of solved cells (unflagged / flagged by FA or
// blend/IL) and the FA energy deficit |dtau|/(tau + D). Bins: [0] < 1e-12, [1..56]
// quarter decades from 1e-12 to 1e2, [57] >= 1e2 or NaN; plus the exact maximum per
// (set, magn bin). Written to <basename>.rlog.txt at the rhist times; not carried
// across restarts. kLogDoff (resistive_d4 = offset|offset_emc, 07 v2.10 §3.2.5(g)):
// |dE_c|/|B| of all interior cells at every recorded c2p.
enum LogSet {kLogEpsUnflagged = 0, kLogEpsFlagged = 1, kLogFADeficit = 2, kLogDoff = 3,
             kNLogSets = 4};
constexpr int kLogBins = 58;

struct ProtoLog {
  DvceArray3D<int64_t> d;
  DvceArray2D<Real> mx;
  ProtoLog() : d("resist_proto_log", kNLogSets, kHistBins, kLogBins),
               mx("resist_proto_max", kNLogSets, kHistBins) {}
  void Reset() const { Kokkos::deep_copy(d, 0); Kokkos::deep_copy(mx, 0.0); }
  KOKKOS_INLINE_FUNCTION
  static int LogBin(const Real x) {
    if (!(x >= 1.0e-12)) return (x == x) ? 0 : kLogBins - 1;
    const Real f = 4.0*(log10(x) + 12.0);
    if (!(f < static_cast<Real>(kLogBins - 2))) return kLogBins - 1;
    return 1 + static_cast<int>(f);
  }
  KOKKOS_INLINE_FUNCTION
  void Add(const int set, const int bm, const Real x) const {
    Kokkos::atomic_add(&d(set, bm, LogBin(x)), static_cast<int64_t>(1));
    if (x == x) Kokkos::atomic_max(&mx(set, bm), x);
  }
  // Host copy, summed (counts) and maximised (mx) over MPI ranks.
  void Read(int64_t out[kNLogSets][kHistBins][kLogBins],
            Real mxo[kNLogSets][kHistBins]) const;
};

// 07 v2.8 §3.2.3(e) failure reasons of non-converged interior cells by radius bin (dx
// wide from the pgen's centre, else one bin), written per (cycle, stage) for the first
// resistive_fail_reason_cycles cycles to <basename>.rfail.txt.
constexpr int kRFailReasons = 8;   // FailReason (resistive_implicit.hpp)
constexpr int kRFailRBins = 1024;  // the last bin collects r >= (kRFailRBins - 1) dx

struct RFailCount {
  DvceArray2D<int64_t> d;
  RFailCount() : d("resist_rfail", kRFailReasons, kRFailRBins) {}
  void Reset() const { Kokkos::deep_copy(d, 0); }
  KOKKOS_INLINE_FUNCTION
  void Add(const int reason, const int rbin) const {
    Kokkos::atomic_add(&d(reason, rbin), static_cast<int64_t>(1));
  }
  // Host copy (size kRFailReasons*kRFailRBins, reason-major) summed over MPI ranks.
  void Read(int64_t *out) const;
};

}  // namespace resistive

#endif  // DYN_GRMHD_RESISTIVE_RESISTIVE_DIAGNOSTICS_HPP_
