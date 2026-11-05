#ifndef DYN_GRMHD_RESISTIVE_RESISTIVE_DYNGR_HPP_
#define DYN_GRMHD_RESISTIVE_RESISTIVE_DYNGR_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file resistive_dyngr.hpp
//! \brief State and EOS-independent tasks of resistive GRMHD on the dyngr path
//!        (07_scoping_v2 §2.2, §2.3, §3.3, D2-D5).
//!
//! Owned data:
//!  * e0, e1  : face-centred densitised E~^i = sqrt(gamma) E^i on x_i faces
//!              (e1 holds E~^n for the RK register combination, like b1).
//!  * hfld    : edge fluxes h* of E~ (Ampere), updated by the CT kernel with e -> h.
//!  * qflx    : explicit convective charge current through each face (07 §2.3).
//!  * emc     : cell-centred undensitised Eulerian (E^x,E^y,E^z,B^x,B^y,B^z), set by c2p.
//!  * cwork   : cell work array of the implicit step: u^i, R_e^i, B^i, sdt.
//! The Ohmic stiff sources of E~ are stored in Driver::impl_src_e.
//!
//! Step 1a scope: static, spatially constant metric (no coordinate source terms), uniform
//! grid (no SMR/AMR), imex2/imex3. Unsupported configurations stop with a fatal error.

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"
#include "dyn_grmhd/resistive/resistive_params.hpp"
#include "dyn_grmhd/resistive/conductivity.hpp"
#include "dyn_grmhd/resistive/resistive_diagnostics.hpp"

class MeshBlockPack;
class Driver;

namespace resistive {

// cell layout of emc
enum {kEX = 0, kEY = 1, kEZ = 2, kBX = 3, kBY = 4, kBZ = 5, kNEMC = 6};
// cell layout of cwork
enum {kWU = 0, kWR = 3, kWB = 6, kWS = 9, kNWORK = 10};

class ResistiveDynGR {
 public:
  ResistiveDynGR(MeshBlockPack *ppack, ParameterInput *pin);
  ~ResistiveDynGR();

  ResistiveParams params;
  Conductivity cond;
  NewtonStats newton;

  DvceFaceFld4D<Real> e0, e1, coarse_e0;
  DvceFaceFld4D<Real> qflx;
  DvceEdgeFld4D<Real> hfld;
  DvceArray5D<Real> emc;
  DvceArray5D<Real> cwork;

  // Separate boundary objects, one per E~ exchange round within a stage (each owns its
  // own MPI communicators and buffers): explicit round (+ h* edge sync), post-implicit
  // round, and the two stage-1 pre-implicit rounds of the ImEx scheme.
  MeshBoundaryValuesFC *pbval_e;
  MeshBoundaryValuesFC *pbval_ei;
  MeshBoundaryValuesFC *pbval_ep;
  MeshBoundaryValuesFC *pbval_ep2;

  // task functions (EOS independent)
  TaskStatus InitRecv(Driver *d, int stage);
  TaskStatus CopyE(Driver *d, int stage);
  TaskStatus EdgeSolve(Driver *d, int stage);
  TaskStatus SendH(Driver *d, int stage);
  TaskStatus RecvH(Driver *d, int stage);
  TaskStatus CTE(Driver *d, int stage);
  TaskStatus SendE(Driver *d, int stage);
  TaskStatus RecvE(Driver *d, int stage);
  TaskStatus SendEI(Driver *d, int stage);
  TaskStatus RecvEI(Driver *d, int stage);
  TaskStatus SendEP(Driver *d, int stage);
  TaskStatus RecvEP(Driver *d, int stage);
  TaskStatus SendEP2(Driver *d, int stage);
  TaskStatus RecvEP2(Driver *d, int stage);
  TaskStatus ApplyBCsE(Driver *d, int stage);
  TaskStatus ApplyBCsEP(Driver *d, int stage);
  TaskStatus ClearSend(Driver *d, int stage);
  TaskStatus ClearRecv(Driver *d, int stage);

  // Blocking exchange of E~ ghost faces + physical BCs (initialisation and restart).
  void ExchangeEBlocking();
  // Adds sum_s a_twid[istage-2][s] dt R_s to E~ (all faces), as in IonNeutral.
  void AddImexSources(Driver *d, int istage);
  // Checks that the ADM metric is spatially uniform and static (Step 1a/2 scope).
  void CheckConstantMetric() const;

 private:
  MeshBlockPack *pmy_pack;
};

}  // namespace resistive

#endif  // DYN_GRMHD_RESISTIVE_RESISTIVE_DYNGR_HPP_
