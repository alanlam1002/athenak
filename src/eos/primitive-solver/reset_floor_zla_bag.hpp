#ifndef EOS_PRIMITIVE_SOLVER_RESET_FLOOR_ZLA_BAG_HPP_
#define EOS_PRIMITIVE_SOLVER_RESET_FLOOR_ZLA_BAG_HPP_
//========================================================================================
// PrimitiveSolver equation-of-state framework
// Copyright(C) 2023 Jacob M. Fields <jmf6719@psu.edu>
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file reset_floor.hpp
//  \brief Describes an error floor that simply resets nonphysical values.
//
//  If the density or pressure fall below the atmosphere, they get floored.
//  We impose similar limits for D and tau. If the density is floored,
//  the velocity is zeroed out and the pressure is also reset to the floor.
//  If the pressure is floored, all other quantities are ignored.
//  If the primitive solve fails, all points are set to floor.

#include <math.h>

#include "ps_types.hpp"
#include "error_policy_interface.hpp"
#include "ps_error.hpp"

namespace Primitive {

class ResetFloorZlaBag : public ErrorPolicyInterface {
 protected:
  /// Constructor
  ResetFloorZlaBag() {
    fail_conserved_floor = false;
    fail_primitive_floor = false;
    adjust_conserved = true;
    q_snap = 0.0;   // overwritten from <mhd>/yn_snap in SetPolicyParams()
  }

  /// Floor for primitive variables
  KOKKOS_INLINE_FUNCTION bool PrimitiveFloor(Real& n, Real v[3], Real& T, Real *Y,
                                             int n_species) const {
    if (n < n_atm*n_threshold) {
      n = n_atm;
      v[0] = 0.0;
      v[1] = 0.0;
      v[2] = 0.0;
      T = T_atm;
      for (int i = 0; i < n_species; i++) {
        Y[i] = Y_atm[i];
      }
      return true;
    } else if (T < T_atm) {
      T = T_atm;
      return true;
    }
    return false;
  }

  /// Floor for conserved variables
  KOKKOS_INLINE_FUNCTION bool ConservedFloor(Real& D, Real Sd[3], Real& tau, Real *Y,
                                Real D_floor, Real tau_floor, Real tau_abs_floor,
                                int n_species) const {
    if (D < D_floor*n_threshold) {
      D = D_floor;
      Sd[0] = 0.0;
      Sd[1] = 0.0;
      Sd[2] = 0.0;
      tau = tau_abs_floor;
      for (int i = 0; i < n_species; i++) {
        Y[i] = Y_atm[i];
      }
      return true;
    } else if (tau < tau_floor) {
      tau = tau_floor;
      return true;
    }
    return false;
  }

  /// Response to excess magnetization
  KOKKOS_INLINE_FUNCTION Error MagnetizationResponse(Real& bsq, Real b_u[3]) const {
    if (bsq > max_bsq) {
      Real factor = sqrt(max_bsq/bsq);
      bsq = max_bsq;

      b_u[0] /= factor;
      b_u[1] /= factor;
      b_u[2] /= factor;

      return Error::CONS_ADJUSTED;
    }
    return Error::SUCCESS;
  }

  /// Policy for resetting density
  KOKKOS_INLINE_FUNCTION void DensityLimits(Real& n, Real n_min, Real n_max) const {
    n = fmax(n_min, fmin(n_max, n));
  }

  /// Policy for resetting temperature
  KOKKOS_INLINE_FUNCTION void TemperatureLimits(Real& T, Real T_min, Real T_max) const {
    T = fmax(T_min, fmin(T_max, T));
  }

  /// Policy for resetting species fractions
  KOKKOS_INLINE_FUNCTION bool SpeciesLimits(Real* Y, const Real* Y_min, const Real* Y_max,
                                            int n_species) const {
    bool adjusted = false;
    /// Volume fraction (f)
    if (Y[0] < Y_min[0]) {
      adjusted = true;
      Y[0] = Y_min[0];
    } else if (Y[0] > Y_max[0]) {
      adjusted = true;
      Y[0] = Y_max[0];
    }
    /// Nucleons Fraction (f nB,N / nB)
    Real Y_min_ = Y_min[1] * Y[0];
    Real Y_max_ = Y_max[1] * Y[0];
    if (Y[0] == Y_max[0] && Y[1] != Y_max_) {
      adjusted = true;
      Y[1] = Y_max_;
    } else if (Y[1] < Y_min_) {
      adjusted = true;
      Y[1] = Y_min_;
    } else if (Y[1] > Y_max_) {
      adjusted = true;
      Y[1] = Y_max_;
    }
    /// Nucleons Leptons Fraction (f nB,N y_lN / nB)
    Y_min_ = Y_min[2] * Y[1];
    Y_max_ = Y_max[2] * Y[1];
    if (Y[2] < Y_min_) {
      adjusted = true;
      Y[2] = Y_min_;
    } else if (Y[2] > Y_max_) {
      adjusted = true;
      Y[2] = Y_max_;
    }
    /// Quarks Leptons Fraction ((1-f) nB,Q y_lQ / nB)
    // The band is [Y_min[3], Y_max[3]] * (1 - Y[1]). As (1 - Y[1]) reaches the
    // round-off floor the band collapses onto the noise in Y[3], and pinning Y[3] to an
    // edge of it is meaningless: y_lQ = Y[3]/(1 - Y[1]) is then a ratio of two
    // noise-level numbers. Measured saturating at exactly -1 on 1.09e9 cell-visits with
    // (1 - Y_N) down to 1e-9, which drained 72% of int(D*Y_3). Leaving Y[3] alone keeps
    // the primitive exactly equal to the conserved value it came from; ConvertPrimitive
    // independently guards the division that consumes it (see EOSZlaBag::yn_snap).
    // FLOOR the reference rather than skipping the test. Skipping it removed the only
    // bound on Y[3]: run v7 then let |y_lQ| reach 1.06e+03, which inflates the
    // atmosphere pressure by eleven decades (P goes 1.8e-23 -> 1.1e-12 as Y[3] goes
    // 1e-9 -> 1e6; unit test I) and the run died with 411,155 NANS_IN_CONS 29 M after
    // restart. Flooring keeps both properties at once:
    //   * the band is ~6 decades wider than the noise band at the default q_snap, so
    //     the clamp almost never fires there and no longer drains int(D*Y_3);
    //   * Y[3] is still bounded by a fixed, RESOLVED quantity, so it cannot run away.
    // The degenerate Y[1] >= 1 case is covered too: the reference floors at q_snap
    // instead of collapsing to 0 (or inverting), so the band never becomes empty.
    // With q_snap = 0 this is identical to the original for every Y[1] <= 1.
    const Real omYN = fmax(1.0 - Y[1], q_snap);
    Y_min_ = Y_min[3] * omYN;
    Y_max_ = Y_max[3] * omYN;
    if (Y[3] < Y_min_) {
      adjusted = true;
      Y[3] = Y_min_;
    } else if (Y[3] > Y_max_) {
      adjusted = true;
      Y[3] = Y_max_;
    }
    return adjusted;
  }

  /// Policy for resetting pressure
  KOKKOS_INLINE_FUNCTION void PressureLimits(Real& P, Real P_min, Real P_max) const {
    P = fmax(P_min, fmin(P_max, P));
  }

  /// Policy for resetting energy density
  KOKKOS_INLINE_FUNCTION void EnergyLimits(Real& e, Real e_min, Real e_max) const {
    e = fmax(e_min, fmin(e_max, e));
  }

  /// Policy for resetting internal energy density
  KOKKOS_INLINE_FUNCTION void InternalEnergyLimits(Real& e, Real e_min, Real e_max) const {
    e = fmax(e_min, fmin(e_max, e));
  }

  /// Policy for dealing with failed points
  KOKKOS_INLINE_FUNCTION bool FailureResponse(Real prim[NPRIM]) const {
    prim[PRH] = n_atm;
    prim[PVX] = 0.0;
    prim[PVY] = 0.0;
    prim[PVZ] = 0.0;
    prim[PTM] = T_atm;
    for (int i = 0; i < MAX_SPECIES; i++) {
      prim[PYF + i] = Y_atm[i];
    }
    return true;
  }

 public:
  /// Set the failure mode for conserved flooring
  KOKKOS_INLINE_FUNCTION void SetConservedFloorFailure(bool failure) {
    fail_conserved_floor = failure;
  }

  /// Set the failure mode for primitive flooring
  KOKKOS_INLINE_FUNCTION void SetPrimitiveFloorFailure(bool failure) {
    fail_primitive_floor = failure;
  }

  /// Set whether or not it's okay to adjust the conserved variables.
  KOKKOS_INLINE_FUNCTION void SetAdjustConserved(bool adjust) {
    adjust_conserved = adjust;
  }

  /// Set the quark-phase resolvability tolerance (<mhd>/yn_snap). Below it the species
  /// band on Y[3] is skipped rather than applied to round-off; see SpeciesLimits().
  KOKKOS_INLINE_FUNCTION void SetQuarkSnapTol(Real tol) {
    q_snap = tol;
  }

  /// The same tolerance, for the flux limiter. dyn_grmhd_fofc.cpp must floor its own
  /// Y[3] reference identically or the band it certifies differs from the band
  /// SpeciesLimits() then applies.
  KOKKOS_INLINE_FUNCTION Real GetQuarkSnapTol() const {
    return q_snap;
  }
};

} // namespace Primitive

#endif  // EOS_PRIMITIVE_SOLVER_RESET_FLOOR_ZLA_BAG_HPP_
