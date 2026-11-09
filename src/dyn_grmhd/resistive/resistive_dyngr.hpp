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
//! SMR (static refinement, 07 D3): E~ is restricted (face-area average) before and
//! prolongated (shared-face minmod + Toth-Roe internal, the B~ operators) after every
//! E~ ghost exchange; h* uses the FC edge flux correction; and after every update of
//! E~ the coarse face shared with finer blocks is overwritten by the area average of
//! the fine faces (fine->coarse message, CC flux-correction machinery on an alias of
//! E~), so the face flux of E~ is single valued and charge is conserved to roundoff.
//!
//! Step 1a scope: static, spatially constant metric (no coordinate source terms), static
//! refinement only (no AMR), imex2/imex3. Unsupported configurations stop with a fatal
//! error.

#include <cstddef>
#include <string>

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
  SigmaMHist hist;        // K5-run histogram, written to <basename>.rhist.txt

  DvceFaceFld4D<Real> e0, e1, coarse_e0;
  DvceFaceFld4D<Real> qflx;
  DvceEdgeFld4D<Real> hfld;
  DvceArray5D<Real> emc;
  DvceArray5D<Real> cwork;
  // resistive_freeze_background only: -(initial discrete dE~/dt), added in CTE
  DvceFaceFld4D<Real> esrc;
  bool esrc_set = false;

  // Separate boundary objects, one per E~ exchange round within a stage (each owns its
  // own MPI communicators and buffers): explicit round (+ h* edge sync), post-implicit
  // round, and the two stage-1 pre-implicit rounds of the ImEx scheme.
  MeshBoundaryValuesFC *pbval_e;
  MeshBoundaryValuesFC *pbval_ei;
  MeshBoundaryValuesFC *pbval_ep;
  MeshBoundaryValuesFC *pbval_ep2;
  // SMR only: fine->coarse shared-face sync of E~, one object per E~ update round.
  enum SyncRound {kSyncEP = 0, kSyncEP2 = 1, kSyncE = 2, kSyncEI = 3, kNSync = 4};
  MeshBoundaryValuesCC *psync[kNSync];
  DvceFaceFld5D<Real> e0_alias;  // unmanaged (nmb,1,...) view of e0 for psync

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
  TaskStatus SyncSendEP(Driver *d, int stage);
  TaskStatus SyncRecvEP(Driver *d, int stage);
  TaskStatus SyncSendEP2(Driver *d, int stage);
  TaskStatus SyncRecvEP2(Driver *d, int stage);
  TaskStatus SyncSendE(Driver *d, int stage);
  TaskStatus SyncRecvE(Driver *d, int stage);
  TaskStatus SyncSendEI(Driver *d, int stage);
  TaskStatus SyncRecvEI(Driver *d, int stage);
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
  // resistive_freeze_background (test only), called from CTE
  void FreezeBackground(Real gam0, Real gam1, Real beta_dt, int mode);
  // esrc sidecar <basename>.esrc.r<rank> (written at capture, read on restart)
  void FreezeFile(bool write);
  // Writes and resets the K5-run histogram when t_end reaches the next output time.
  void MaybeWriteHist(Real t_end, int cycle);
  // Restart-file state (07 v2.5 §4.1): Newton counters, pending histogram counts and the
  // next histogram time, MPI-reduced, so counters accumulate across restart links.
  // Pack is collective; the restart file stores the blob after the root-process state.
  static constexpr std::size_t RestartStateSize() {
    return 2*sizeof(Real) + sizeof(int64_t)*(kNewtonNCounters +
                                             kHistBinnings*kHistBins*kHNCounters);
  }
  void PackRestartState(char *buf) const;
  void UnpackRestartState(const char *buf);

 private:
  MeshBlockPack *pmy_pack;
  bool multilevel;
  Real hist_dt = -1.0, hist_next = 0.0;
  bool hist_header = false;
  std::string hist_fname;
  std::string esrc_fname;
  TaskStatus SyncSend(int r);
  TaskStatus SyncRecv(int r);
  void RestrictE();
  void ProlongE();
};

}  // namespace resistive

#endif  // DYN_GRMHD_RESISTIVE_RESISTIVE_DYNGR_HPP_
