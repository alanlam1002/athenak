#ifndef RADIATION_RADIATION_BUDGET_HPP_
#define RADIATION_RADIATION_BUDGET_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_budget.hpp
//! \brief Optional energy-budget accounting for radiation (M)HD (bhstar Stage 0).
//!
//! Enabled with <radiation>/energy_budget = true (default false; nothing below runs
//! otherwise). Energies are "energy at infinity minus rest mass", summed over the active
//! cells of this rank, in code units:
//!     E_gas = -sum u0(IEN) dV        (AthenaK GR evolves IEN = T^t_t + D)
//!     E_rad = -sum_n w_n i0_n dV     (the coupling adds sum_n w_n i0_n to IEN)
//! Cartesian Kerr-Schild has sqrt(-g) = 1, so dV = dx1 dx2 dx3.
//!
//! Each counter c[X] holds the energy that process X has put into the current state.
//! Because the RK update forms u = gam0*u + gam1*u_base + ..., the counters are combined
//! with the same weights once per stage (Combine), so that at all times
//!     (E_gas + E_rad)(t) - (E_gas + E_rad)(0) = sum over CLOSURE terms of c[X](t)
//! holds to round-off. BREAKDOWN terms split a closure term and are not summed.

#include "athena.hpp"

class MeshBlockPack;
class Driver;

namespace radiation {

enum BudgetTerm {
  // closure terms
  kBudInj = 0,     // radiation source terms (rad_shell, rad_beam, user source)
  kBudRadUpd,      // radiation RK update: boundary flux + angular flux + floor + excision
  kBudGasUpd,      // (M)HD RK update: boundary flux (+ coordinate/zoom flux terms)
  kBudGasSrc,      // (M)HD source-term task (coordinate sources, user sources)
  kBudCoupl,       // RadFluidCoupling total (c2p floors/excision + intensity excision)
  kBudC2P,         // (M)HD ConToPrim task after the BCs (floors, excision resets)
  kBudMask,        // cyclic-zoom mask (ApplyMask)
  kBudRegrid,      // mesh refinement / zoom transitions between steps
  // breakdown terms
  kBudRadFlux,     // part of kBudRadUpd: radiation energy flux through domain faces
  kBudRadFloor,    // part of kBudRadUpd: intensity floor fmax(..,0)
  kBudRadExc,      // part of kBudRadUpd: excision zeroing
  kBudGasFlux,     // part of kBudGasUpd: gas energy flux through domain faces
  kBudCouplC2P,    // part of kBudCoupl: ConsToPrim floors/excision inside the coupling
  kBudCouplExc,    // part of kBudCoupl: excision zeroing of intensities in the coupling
  kNBudget
};
constexpr int kNBudgetClosure = 8;

class RadBudget {
 public:
  explicit RadBudget(MeshBlockPack *pp);

  Real c[kNBudget];        // counters (energy put into the current state by each term)
  Real c_base[kNBudget];   // counters at the start of the step (RK base state)
  Real eg_base, er_base;   // E_gas, E_rad at the start of the step
  Real nbadcell;           // cumulative number of skipped coupling cells (not RK-combined)
  DvceArray1D<Real> acc;   // device accumulators for rare in-kernel events (atomics)

  Real GasEnergy();        // E_gas of this rank (0 if no fluid)
  Real RadEnergy();        // E_rad of this rank
  void StageBase(int stage);                   // at CopyCons
  void Combine(Real gam0, Real gam1);          // once per stage, before any update
  Real RadFaceFlux(Real beta_dt);              // energy in through domain faces (rad)
  Real GasFaceFlux(Real beta_dt);              // energy in through domain faces (gas)
  void ZeroAcc();
  void ReadAcc(Real *a, int n);

 private:
  MeshBlockPack *pmy_pack;
  Real GasFaceFluxOf(const DvceFaceFld5D<Real> &flx, Real beta_dt);
};

} // namespace radiation
#endif // RADIATION_RADIATION_BUDGET_HPP_
