#ifndef EOS_PRIMITIVE_SOLVER_PS_ERROR_HPP_
#define EOS_PRIMITIVE_SOLVER_PS_ERROR_HPP_
//========================================================================================
// PrimitiveSolver equation-of-state framework
// Copyright(C) 2023 Jacob M. Fields <jmf6719@psu.edu>
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ps_error.hpp
//  \brief defines an enumerator struct for error types.

#include "ps_types.hpp"   // Real, for the y3 diagnostic fields in SolverResult

namespace Primitive {
enum struct Error {
  SUCCESS,
  RHO_TOO_BIG,
  RHO_TOO_SMALL,
  NANS_IN_CONS,
  MAG_TOO_BIG,
  BRACKETING_FAILED,
  NO_SOLUTION,
  CONS_FLOOR,
  PRIM_FLOOR,
  CONS_ADJUSTED,
};

struct SolverResult {
  Error error;
  int  iterations;
  bool cons_floor;
  bool prim_floor;
  bool cons_adjusted;
  // Diagnostic for the fourth ZLA-bag species, Y[3] = (1 - Y_N) y_lQ. Its admissible
  // band is [min_Y[3], max_Y[3]] * (1 - Y[1]), which COLLAPSES TO {0} wherever the
  // nucleon fraction reaches one -- so any Y[3] that has diffused into f == 1 material
  // is pinned to zero and, because ConToPrim writes the clamped composition back into
  // the conserved variables, never recovers. These fields record that event so the
  // caller can report it; they are only meaningful when y3_clamped is true.
  bool species_adjusted;   // ApplySpeciesLimits changed some Y; NOT propagated to cons
  // Which stage actually moved Y[3]: 1=ConservedFloor, 2=SpeciesLimits,
  // 4=PrimitiveFloor. Set per stage by before/after comparison, because both
  // floors have a tau-only / T-only branch that returns true without touching Y.
  int  y3_chan;
  Real y3_entry;           // Y[3] as derived from the incoming conserved state
  Real y3_final;           // Y[3] after every adjustment stage
  bool y3_clamped;
  Real y3_before;    // Y[3] as it arrived from the conserved variables
  Real y3_after;     // Y[3] after the clamp (i.e. the bound it was pinned to)
  Real y3_lo;        // lower edge of the admissible band
  Real y3_hi;        // upper edge of the admissible band
  Real y3_omYN;      // (1 - Y[1]) as ENFORCED: post-cascade and floored at q_snap
  //! Scalar-3 band reference BEFORE the SpeciesLimits cascade ran, i.e. the quantity
  //! dyn_grmhd_fofc.cpp built its certificate from. y3_omYN is the one actually
  //! enforced. Their difference is the band disagreement (Phase 2 defect 3).
  Real y3_omYN_pre;
};

} // namespace Primitive

#endif  // EOS_PRIMITIVE_SOLVER_PS_ERROR_HPP_
