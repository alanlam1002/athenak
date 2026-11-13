#ifndef CFC_CFC_Z4C_HANDOFF_HPP_
#define CFC_CFC_Z4C_HANDOFF_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cfc_z4c_handoff.hpp
//! \brief one-way, in-run switch of the spacetime solver from CFC to Z4c
//! (DEVELOPMENT.md item 64).
//!
//! A run with both <cfc> (handoff = true) and <z4c> blocks starts as a plain CFC run:
//! MeshBlockPack::AddPhysics() builds CFC + ADM (CFC layout) and *defers* Z4c, so
//! pz4c == nullptr and nothing about the CFC evolution changes. Z4cHandoff is checked
//! at the end of every cycle (and once at the end of Driver::Initialize); when a
//! trigger fires it converts the current CFC hypersurface (conformally flat metric,
//! maximal slicing K = 0, CFC lapse/shift) into Z4c initial data, deletes CFC, rebuilds
//! the NumericalRelativity task graph and the outputs, and the run continues as a
//! native Z4c + dyn_grmhd run. <cfc>/handoff_done is written back to the ParameterInput
//! so restarts taken after the switch are rebuilt directly as Z4c runs.

#include <string>

#include "athena.hpp"

class MeshBlockPack;
class ParameterInput;
class Driver;
class Outputs;

namespace cfc {

enum class HandoffState {none, pending, done};

// none: no <cfc>/handoff = true. pending: handoff requested, not yet executed (CFC
// active, Z4c deferred). done: <cfc>/handoff_done = true (Z4c active, CFC never built).
HandoffState GetHandoffState(ParameterInput *pin);

class Z4cHandoff {
 public:
  Z4cHandoff(MeshBlockPack *ppack, ParameterInput *pin);
  ~Z4cHandoff() = default;

  // Tests the triggers and, if one fires, performs the switch. Returns true once the
  // switch has been made; the caller then deletes this object. Collective over MPI.
  bool CheckAndExecute(Driver *pdriver, ParameterInput *pin, Outputs *pout);

  // Diagnostics (collective over MPI)
  Real GlobalAlphaMin();
  // Conformal-flatness residual: RMS over the grid of
  //   |(L_beta gamma)^TF_ij - 2 alpha K_ij|_gamma
  // divided by max(RMS |2 alpha K_ij|_gamma, handoff_cf_floor). For a conformally flat
  // 3-metric, d_t gamma_ij's trace-free part is (L_beta gamma)^TF_ij - 2 alpha K_ij, so
  // this measures the d_t (conformal metric) that CFC drops.
  Real CFResidual();

 private:
  MeshBlockPack *pmy_pack;
  Real t_switch_;           // <cfc>/handoff_time   (< 0: disabled)
  int cycle_switch_;        // <cfc>/handoff_cycle  (< 0: disabled)
  Real alpha_min_thr_;      // <cfc>/handoff_alpha_min   (<= 0: disabled)
  Real cf_res_thr_;         // <cfc>/handoff_cf_residual (<= 0: disabled)
  Real cf_floor_;           // <cfc>/handoff_cf_floor
  int check_dcycle_;        // <cfc>/handoff_check_dcycle, cadence of the auto criteria
  bool resolve_;            // <cfc>/handoff_resolve: re-solve CFC before converting
  bool force_output_;       // <cfc>/handoff_output: write all outputs at the switch
  bool verbose_;            // <cfc>/handoff_verbose: print criteria at every check

  bool ShouldSwitch(Driver *pdriver, std::string &reason);
  void Execute(Driver *pdriver, ParameterInput *pin, Outputs *pout,
               const std::string &reason);
  void WriteAllOutputs(ParameterInput *pin, Outputs *pout);
};

} // namespace cfc

#endif // CFC_CFC_Z4C_HANDOFF_HPP_
