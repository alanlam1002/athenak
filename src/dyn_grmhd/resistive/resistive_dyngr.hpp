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
//!  * doff    : resistive_d4 = offset|offset_emc only: persistent cell offset dE_c.
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
#include <cstdint>
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
// cell layout of cwork; kWF/kWE (07 v2.8 §3.2.3(d)): flag bits of the last implicit solve
// (consumed and cleared by ResistiveC2P) and the cell E it returned (R if it failed)
enum {kWU = 0, kWR = 3, kWB = 6, kWS = 9, kWF = 10, kWE = 11, kNWORK = 14};
// kFlagFailed (resistive_d4 = offset|offset_emc|resplit only): the solve hard-failed
enum {kFlagSolved = 1, kFlagFallback = 2, kFlagFailed = 4};
// test-only probe (07 v2.9 §3.2.4(e), resistive_probe.hpp): per-cell implicit-solve
// record (reason, CellSolveResult flag bits, iterations, W of the ideal c2p on the
// pre-solve totals and its error code, W and E of the returned state) and the EdgeSolve
// split of the E~ edge EMF (unsloped four-cell average, sloped average, full value)
enum {kPReason = 0, kPFlags = 1, kPIter = 2, kPWIL = 3, kPILErr = 4, kPWSol = 5,
      kPE = 6, kNPST = 9};
enum {kPEdgeAvg = 0, kPEdgeSlope = 1, kPEdgeFull = 2, kNPEDGE = 9};

class ResistiveDynGR {
 public:
  ResistiveDynGR(MeshBlockPack *ppack, ParameterInput *pin);
  ~ResistiveDynGR();

  ResistiveParams params;
  Conductivity cond;
  NewtonStats newton;
  SigmaMHist hist;        // K5-run histogram, written to <basename>.rhist.txt
  ProtoLog plog;          // 07 v2.8 prototype eps_D4/FA-deficit, <basename>.rlog.txt
  RFailCount rfail;       // 07 v2.8 failure reasons, <basename>.rfail.txt
  // rfail radius origin, set by the pgen (e.g. the rotor centre); else a single bin
  bool rfail_center_set = false;
  Real rfail_center[3] = {0.0, 0.0, 0.0};

  DvceFaceFld4D<Real> e0, e1, coarse_e0;
  DvceFaceFld4D<Real> qflx;
  DvceEdgeFld4D<Real> hfld;
  DvceArray5D<Real> emc;
  DvceArray5D<Real> cwork;
  // resistive_d4 = offset|offset_emc only (07 v2.10 §3.2.5): persistent cell offset
  // dE_c = E_solve - FaceAvg(E~)/sqrt(gamma) (undensitised), set at the c2p after each
  // cell solve, restarted; (nmb, 3, n3, n2, n1)
  DvceArray5D<Real> doff;
  // resistive_d4 = resplit only: cumulative volume-integrated EM energy change of the
  // re-split (interior cells, this rank; reduced when written to the rlog)
  Real resplit_de = 0.0;
  // resistive_freeze_background only: -(initial discrete dE~/dt), added in CTE
  DvceFaceFld4D<Real> esrc;
  bool esrc_set = false;
  // test-only probe: first probe_cycles cycles (0 = off); arrays allocated only if on
  int probe_cycles = 0;
  std::string probe_base;
  DvceArray5D<Real> pst;    // (nmb, kNPST, n3, n2, n1)
  DvceArray5D<Real> pedge;  // (nmb, 3*kk + part, n3+1, n2+1, n1+1)
  bool Probing() const;

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
  // Appends the failure-reason counts of one implicit stage to <basename>.rfail.txt and
  // resets them (collective).
  void WriteRFail(int cycle, int stage);
  // Writes and resets the prototype log histograms (called with the rhist output).
  void MaybeWriteRLog(Real t_end, int cycle);
  // Restart-file state (07 v2.5 §4.1): Newton counters, pending histogram counts and the
  // next histogram time, MPI-reduced, so counters accumulate across restart links.
  // Pack is collective; the restart file stores the blob after the root-process state.
  // The blob starts with a format header (07 v2.6 R1): 8-byte magic, uint32 version,
  // uint32 payload size. Bump kRestartVersion whenever the payload layout changes.
  static constexpr char kRestartMagic[9] = "ATKRESIS";
  // 2: rhist columns of 07 v2.8; 3: + uint32 dE_c flag and pad (07 v2.10 §3.2.5(e)),
  // the dE_c cell data follow e0 in the MeshBlock data. Version 2 is still read (no
  // dE_c), except under resistive_d4 = offset|offset_emc.
  static constexpr uint32_t kRestartVersion = 3;
  static constexpr std::size_t RestartHeaderSize() { return 8 + 2*sizeof(uint32_t); }
  static constexpr std::size_t RestartStateSize(uint32_t ver = kRestartVersion) {
    return 2*sizeof(Real) + sizeof(int64_t)*(kNewtonNCounters +
                                             kHistBinnings*kHistBins*kHNCounters) +
           ((ver >= 3) ? 2*sizeof(uint32_t) : 0);
  }
  // header + payload, as written to the restart file
  static constexpr std::size_t RestartBlockSize() {
    return RestartHeaderSize() + RestartStateSize();
  }
  // Aborts with a clear message unless the header is a format this binary reads;
  // returns the payload size.
  static std::size_t CheckRestartHeader(const char *hdr);
  void PackRestartState(char *buf) const;      // writes header + payload
  // reads the payload of the given size (header checked first); aborts if the file's
  // dE_c presence does not match resistive_d4
  void UnpackRestartState(const char *buf, std::size_t size);

 private:
  MeshBlockPack *pmy_pack;
  bool multilevel;
  Real hist_dt = -1.0, hist_next = 0.0;
  bool hist_header = false;
  std::string hist_fname;
  std::string rlog_fname, rfail_fname;
  bool rlog_header = false, rfail_header = false;
  std::string esrc_fname;
  TaskStatus SyncSend(int r);
  TaskStatus SyncRecv(int r);
  void RestrictE();
  void ProlongE();
};

}  // namespace resistive

#endif  // DYN_GRMHD_RESISTIVE_RESISTIVE_DYNGR_HPP_
