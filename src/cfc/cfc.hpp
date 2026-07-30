#ifndef CFC_CFC_HPP_
#define CFC_CFC_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cfc.hpp
//! \brief CFC: conformally-flat-condition (XCFC) metric solver (Cheong et al. 2021,
//! arXiv:2012.07322, sec. 2.6).
//!
//! Unlike gravity::Gravity, CFC queues its steps into the shared NumericalRelativity
//! task graph (QueueCFCTasks()) instead of calling Driver::Execute() directly, since
//! its solve must interleave with dyn_grmhd's hydro update and share a single con2prim
//! call with it. Matter sources are built from MeshBlockPack::pmhd->u0/w0 directly, not
//! MeshBlockPack::ptmunu (only populated when z4c is active). Output metric is written
//! into MeshBlockPack::padm->u_adm.
//!
//! Fields differentiated by cfc_reconstruct.cpp need a full mesh-NGHOST-deep ghost
//! exchange after each multigrid solve, since a solver's own ghost width (ngh_) is
//! generally shallower -- each gets one MeshBoundaryValuesCC Rest->Send->Recv->Prolong
//! round (mirrors z4c::Z4c::pbval_u/coarse_u0).

// Athenak headers
#include "../athena.hpp"
#include "../athena_tensor.hpp"
#include "../mesh/meshblock_pack.hpp"
#include "../parameter_input.hpp"
#include "../tasklist/task_list.hpp"
#include "../bvals/bvals.hpp"
#include "mg_cfc_vector_poisson.hpp"
#include "mg_cfc_conformal_factor.hpp"
#include "mg_cfc_lapse.hpp"

class MeshBlockPack;
class ParameterInput;
class Driver;
class MeshBoundaryValuesCC;

namespace cfc {

class CFC {
 public:
  CFC(MeshBlockPack *pmbp, ParameterInput *pin);
  ~CFC();

  MeshBlockPack *pmy_pack;

  // Intermediate fields on the finest mesh grid. Vector/tensor quantities are
  // AthenaTensor views (as in z4c/adm) shallow-sliced from an underlying "u_*"
  // storage array; scalars are plain DvceArray5D<Real>. All sized at mesh-NGHOST
  // depth (nmb, ncomp, nx3+2*ng, nx2+2*ng, nx1+2*ng).

  DvceArray5D<Real> u_beta;
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> beta_u;   // beta^i, shift vector

  DvceArray5D<Real> u_adual;
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> a_dd;     // Adual^ij (eq. 76)

  DvceArray5D<Real> a_sq;   // Ahat^2 = f_ik f_jl Adual^kl Adual^ij

  // Store psi-1 / alpha*psi-1, not the physical field -- this is exactly what the
  // multigrid solve iterates on internally, and avoids losing precision far from
  // the star where the physical value is ~1+tiny. Consumers add 1.0 back at the
  // point of use (AssembleConformalMetric/AssembleLapseShiftK,
  // AssembleVectorSource's for_shift branch, BuildShiftSource,
  // MGCFCLapseDriver::LoadReactionCoefficient).
  DvceArray5D<Real> delta_psi;         // psi - 1
  DvceArray5D<Real> delta_alpha_psi;   // alpha*psi - 1

  // U-tilde = psi^6*U: built directly from the evolved conserved state
  // (pmy_pack->pmhd->u0), not MeshBlockPack::ptmunu -- Tmunu is only populated when
  // z4c is active, but CFC's primary use case has no z4c free evolution.
  DvceArray5D<Real> u_tilde;

  // Persistent scratch (avoids a fresh device allocation every stage).
  DvceArray5D<Real> u_plus_2s;     // SolveLapse: u_tilde + 2*s_tilde
  DvceArray5D<Real> u_alpha_psi6;  // BuildShiftSource: alpha*psi^-6

  // Undensitized U = Utilde/sqrt(detg), computed alongside u_tilde using the same
  // g_dd that built the conserved state -- feeds the alternate psi^5 Newton
  // formulation used only by InitializeMetric() (see mg_cfc_conformal_factor.cpp's
  // ConformalFactorRHS). Cheap, so computed unconditionally.
  DvceArray5D<Real> u_raw;

  DvceArray5D<Real> u_stilde;
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> s_tilde_d;  // S-tilde_i = psi^6 S_i

  DvceArray5D<Real> s_tilde;       // S-tilde = psi^6 S (trace of S_ij)

  // Shibata (1999) sec. 3: each vector equation (X^i, beta^i) decomposes into a
  // vector potential P_i (eq. 3.10: Delta P_i = S_i) and scalar eta (eq. 3.11:
  // Delta eta = -S_i x^i). Both are packed into one nvar_=4 array (P_i at channels
  // 0-2, eta at channel 3) and solved together by one MGCFCVectorPoissonDriver::
  // Solve() call -- eta has no dedicated view, read/written as channel 3 of
  // u_p_x/u_p_beta directly. beta^i is reconstructed into beta_u by
  // cfc::ReconstructVectorFromPotentials; X^i is never materialized at all --
  // Adual^ij is computed directly from (P_i, eta) via
  // cfc::ComputeADualFromPotentials (X^i's own definition substituted into and
  // expanded through Adual's formula), which needs no ghost exchange of its own.
  //
  // Unlike u_p_src below, these are solve OUTPUTS that cfc_reconstruct.cpp then
  // differentiates -- sized at mesh-NGHOST depth (not this solver's own shallower
  // ngh_), with their own post-solve MeshBoundaryValuesCC round (pbval_pietax
  // below) before being differentiated. MGCFCVectorPoissonDriver::RetrieveSolution
  // accounts for the size mismatch internally.
  DvceArray5D<Real> u_p_x;                        // ch. 0-2 = P_i, ch. 3 = eta
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> p_x; // P_i view (ch. 0-2 of u_p_x)

  DvceArray5D<Real> u_p_beta;                     // ch. 0-2 = P_i, ch. 3 = eta
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> p_beta;  // P_i view (ch. 0-2 of u_p_beta)

  // P_i/eta's packed right-hand side (S_i at channels 0-2, -S_i.x^i at channel 3),
  // built by AssembleVectorSource() and consumed by LoadPoissonSource(). Read
  // pointwise only (never differentiated or ghost-exchanged), so this stays at
  // this solver's own (shallower) ngh_ depth, unlike u_p_x/u_p_beta above. A raw
  // DvceArray5D<Real>, not an AthenaTensor, since an AthenaTensor's backing
  // storage is a Kokkos::subview result that doesn't bind to
  // MGCFCVectorPoissonDriver::LoadPoissonSource()'s parameter type. Shared
  // sequentially by the X^i and beta^i solves (never needed simultaneously).
  DvceArray5D<Real> u_p_src;
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> p_src; // S_i view (ch. 0-2 of u_p_src)
  int mg_nghost_;  // this solver's own ghost width, cached for index translation

  // Multigrid solvers, one per elliptic equation. pmgd_pietax/pmgd_pietabeta each
  // solve the packed (P_i, eta) system for their Shibata pair in one call.
  MGCFCVectorPoissonDriver *pmgd_pietax;      // solves X^i's packed (P_i, eta)
  MGCFCVectorPoissonDriver *pmgd_pietabeta;   // solves beta^i's packed (P_i, eta)
  MGCFCConformalFactorDriver *pmgd_psi;
  MGCFCLapseDriver *pmgd_alpha;

  // One-shot: seed the very first SolveConformalFactor/SolveLapse call's V-cycle
  // from the pgen's/restart's own ADM data (padm->adm.psi4/alpha) instead of a cold
  // zero start. Left false thereafter (natural multigrid warm start from the
  // previous stage/cycle).
  bool psi_seeded_ = false;
  bool alpha_psi_seeded_ = false;

  // InitializeMetric()'s X^i/psi fixed-point-iteration controls.
  int cfc_init_iter_max_;
  Real cfc_init_tol_;
  bool cfc_init_verbose_;
  // Under-relaxation for InitializeMetric()'s outer Picard loop (1.0 = none).
  // Needed when that loop itself oscillates/diverges for compact stars -- a
  // separate failure mode from cfc_init_use_psi5_'s own divergence below.
  Real cfc_init_omega_;

  // <cfc> init_use_psi5_source (default true): InitializeMetric()'s
  // SolveConformalFactor() uses the self-consistent U_raw*psi^5 Newton
  // formulation instead of Utilde*psi^-1 (see mg_cfc_conformal_factor.cpp's
  // ConformalFactorRHS). Converges much faster for most stars but diverges to
  // NaN for very compact ones, which must set this false explicitly. Never
  // affects the per-stage CFC_SolvePsi task (always Utilde*psi^-1). Mutually
  // incompatible with cfc_init_freeze_conserved_ below -- see cfc.cpp's
  // constructor for the mutual-exclusivity enforcement.
  bool cfc_init_use_psi5_;

  // <cfc> init_freeze_conserved (default false): InitializeMetric() calls
  // RunXPsiSolvePass() once instead of Picard-iterating it -- Utilde/S-tilde_i
  // built once from the pgen's initial guess and held fixed ("one CFC step",
  // no outer loop). Its resulting metric's implied primitives may not exactly
  // match the pgen's originals, the same tradeoff every per-stage CFC_SolvePsi
  // call already lives with.
  bool cfc_init_freeze_conserved_;

  // Post-multigrid ghost exchange: one MeshBoundaryValuesCC + coarse shadow array
  // per field cfc_reconstruct.cpp differentiates (mirrors z4c::Z4c::pbval_u/
  // coarse_u0; is_z4c=true throughout, reusing z4c's higher-order Lagrange
  // restrict/prolong path -- see cfc.cpp's constructor comment). coarse_* arrays
  // are only sized under SMR/AMR (RestrictCC/ProlongateCC are no-ops otherwise).
  // u_p_x/u_p_beta each get one round covering all 4 packed channels at once.
  MeshBoundaryValuesCC *pbval_pietax;
  MeshBoundaryValuesCC *pbval_psi, *pbval_alpha_psi;
  MeshBoundaryValuesCC *pbval_pietabeta;
  DvceArray5D<Real> coarse_u_pietax;
  DvceArray5D<Real> coarse_psi, coarse_alpha_psi;
  DvceArray5D<Real> coarse_u_pietabeta;

  // AssembleConformalMetric/AssembleLapseShiftK only write padm->u_adm's interior
  // (unlike z4c's Z4cToADM, which fills the full ghost-inclusive extent) -- without
  // this round, u_adm's ghosts stay at their t=0 pgen value forever, silently wrong
  // at inter-MeshBlock boundaries in any multi-MeshBlock run. One combined round
  // for the whole u_adm array (g_dd, vK_dd, psi4, alpha, beta_u -- CFC never runs
  // with z4c active, so no channel aliases into a z4c-owned array), run once per
  // stage right after CFC_AssembleFinal. Unlike the coarse_* buffers above, this
  // reuses padm->coarse_u_adm directly (same nmb/nccells sizing, adm.cpp:57-67)
  // rather than keeping a second, redundant CFC-owned copy -- padm->coarse_u_adm
  // is already the buffer the generic AMR regrid pipeline (mesh_refinement.cpp/
  // load_balance.cpp) expects, and RestADMTask fully refreshes it via RestrictCC
  // at the start of every stage, so there's no staleness risk from the two use
  // sites (per-stage ghost exchange vs. regrid transfer) sharing one array.
  MeshBoundaryValuesCC *pbval_adm;

  // Queues this module's tasks into the shared NumericalRelativity task graph
  // (mirrors dyngr::DynGRMHD::QueueDynGRMHDTasks()/z4c::Z4c::QueueZ4cTasks()).
  // Called once, from NumericalRelativity::AssembleNumericalRelativityTasks(), not
  // from Driver::Execute() directly (unlike gravity::Gravity), since CFC's steps
  // must interleave with dyn_grmhd's hydro/con2prim tasks -- see cfc.cpp for the
  // full per-stage dependency chain. dyn_grmhd's MHD_C2P/MHD_Newdt take
  // CFC_SolvePsi/CFC_AssembleFinal as *optional* dependencies, so a single con2prim
  // per stage serves both dyn_grmhd's own needs and CFC's.
  void QueueCFCTasks();

  // One-time metric initialization: the problem generator sets padm->adm directly
  // (e.g. a TOV profile), never through CFC's own solve, so this Picard-iterates
  // X^i/psi (holding primitives fixed at the pgen's values) until psi converges.
  // Called once from Driver::Initialize() (skipped on restart -- already
  // self-consistent); non-convergence is a warning, not fatal. <cfc>
  // init_freeze_conserved selects a one-shot variant instead (see
  // cfc_init_freeze_conserved_ above). Re-runs
  // Driver::InitBoundaryValuesAndPrimitives() at the end so ghost cells stay
  // consistent with the newly-solved metric.
  void InitializeMetric(Driver *pdriver);

  // Regrid counterpart to InitializeMetric(), called once per regrid event from
  // Driver::InitBoundaryValuesAndPrimitives (is_amr_regrid-gated), mirroring
  // z4c::Z4c::ConvertZ4cToADM's placement. Does NOT call InitializeMetric() --
  // that would rebuild conserved variables from primitives (PrimToConInit),
  // discarding the already-evolved, mass-conserving pmhd->u0. Instead runs one
  // plain CFC step holding conserved variables fixed, with its own
  // interior-then-ghost con2prim split standing in for the driver-level
  // ConToPrim this replaces.
  //
  // Resets psi_seeded_/alpha_psi_seeded_ so the two Newton solves re-seed from
  // adm.psi4/alpha -- safe even across a cross-rank regrid move since this
  // re-solve unconditionally overwrites every u_adm field every call.
  // padm->u_adm itself transfers correctly across regrid moves regardless
  // (adm::ADM owns it directly, wired into the generic AMR pipeline the same
  // way pz4c->u0 is); mesh_refinement.cpp also skips the generic
  // padm->SetADMVariables regrid callback for CFC runs (it would otherwise
  // re-derive the metric from the pgen's static t=0 ID, discarding CFC's own
  // dynamical evolution).
  void ReinitializeMetricForAMR(Driver *pdriver);

  // Task-graph entry points (TaskStatus(Driver*, int), required by
  // NumericalRelativity::QueueTask). Each wraps the like-named private Step method
  // below; kept separate so the "Step N" structure stays visible.
  TaskStatus SolveVecXTask(Driver *pdriver, int stage);       // step 1 (CFC_BuildSrcX)
  TaskStatus ComputeADualTask(Driver *pdriver, int stage);    // step 2 (CFC_ComputeADual)
  TaskStatus SolvePsiTask(Driver *pdriver, int stage);        // step 3
  TaskStatus RescaleSrcTask(Driver *pdriver, int stage);      // step 4
  TaskStatus SolveLapseTask(Driver *pdriver, int stage);      // step 5
  TaskStatus SolveShiftTask(Driver *pdriver, int stage);      // step 6 (CFC_BuildSrcBeta)
  TaskStatus ReconstructBetaTask(Driver *pdriver, int stage); // CFC_ReconstructBeta
  TaskStatus AssembleFinalTask(Driver *pdriver, int stage);   // final assembly

  // Ghost-exchange task-graph entry points, one Rest/Send/Recv/Prolong quartet per
  // field above (mirrors z4c::Z4c::RestrictU/SendU/RecvU/Prolongate).
  TaskStatus RestPiEtaXTask(Driver *pdriver, int stage);
  TaskStatus SendPiEtaXTask(Driver *pdriver, int stage);
  TaskStatus RecvPiEtaXTask(Driver *pdriver, int stage);
  TaskStatus ProlongPiEtaXTask(Driver *pdriver, int stage);
  TaskStatus BCSPiEtaXTask(Driver *pdriver, int stage);
  TaskStatus RestPsiTask(Driver *pdriver, int stage);
  TaskStatus SendPsiTask(Driver *pdriver, int stage);
  TaskStatus RecvPsiTask(Driver *pdriver, int stage);
  TaskStatus ProlongPsiTask(Driver *pdriver, int stage);
  TaskStatus BCSPsiTask(Driver *pdriver, int stage);
  TaskStatus RestAlphaPsiTask(Driver *pdriver, int stage);
  TaskStatus SendAlphaPsiTask(Driver *pdriver, int stage);
  TaskStatus RecvAlphaPsiTask(Driver *pdriver, int stage);
  TaskStatus ProlongAlphaPsiTask(Driver *pdriver, int stage);
  TaskStatus BCSAlphaPsiTask(Driver *pdriver, int stage);
  TaskStatus RestPiEtaBetaTask(Driver *pdriver, int stage);
  TaskStatus SendPiEtaBetaTask(Driver *pdriver, int stage);
  TaskStatus RecvPiEtaBetaTask(Driver *pdriver, int stage);
  TaskStatus ProlongPiEtaBetaTask(Driver *pdriver, int stage);
  TaskStatus BCSPiEtaBetaTask(Driver *pdriver, int stage);
  // Ghost-exchanges padm->u_adm after AssembleFinalTask -- see pbval_adm's comment.
  TaskStatus RestADMTask(Driver *pdriver, int stage);
  TaskStatus SendADMTask(Driver *pdriver, int stage);
  TaskStatus RecvADMTask(Driver *pdriver, int stage);
  TaskStatus ProlongADMTask(Driver *pdriver, int stage);
  TaskStatus BCSADMTask(Driver *pdriver, int stage);

  // Post all fields' non-blocking MPI receives up front (Task_Start) and wait on
  // outstanding sends/receives at the very end (Task_End) -- mirrors
  // z4c::Z4c::InitRecv/ClearSend/ClearRecv. Without this, RecvAndUnpackCC's
  // completion check would test an MPI_Request that was never posted (stuck at
  // MPI_REQUEST_NULL, which MPI_Test/MPI_Wait always report complete on), silently
  // unpacking garbage instead of the neighbor's real data.
  TaskStatus InitRecvTask(Driver *pdriver, int stage);
  TaskStatus ClearSendTask(Driver *pdriver, int stage);
  TaskStatus ClearRecvTask(Driver *pdriver, int stage);

 private:
  // Builds Shibata eq. 3.10-3.11's packed source (P_i's RHS S_i at channels 0-2,
  // eta's RHS -S_i.x^i at channel 3) into u_p_src/p_src -- for_shift selects X^i's
  // source (eq. 72, from pmhd->u0) or beta^i's (eq. 75, from alpha/psi/Adual^ij/
  // S-tilde_i).
  void AssembleVectorSource(bool for_shift);

  // Steps 1-6 of the per-stage/per-Picard-iteration solve pipeline; see the
  // like-named TaskStatus wrapper functions in cfc.cpp for the full per-step
  // rationale (equation references, dependency reasons, use_psi5_source, etc).
  void SolveVectorPotential(Driver *pdriver, int stage);        // step 1: eq. 72 -> u_p_x
  void ComputeADual();                                    // step 2: eq. 76, from u_p_x
  void SolveConformalFactor(Driver *pdriver, int stage,         // step 3: eq. 73, psi
                            bool use_psi5_source = false);
  void RescaleMatterSources(Driver *pdriver, int stage);      // step 4: rebuild S-tilde
  void SolveLapse(Driver *pdriver, int stage);           // step 5: eq. 74, alpha*psi
  void SolveShift(Driver *pdriver, int stage);            // step 6: eq. 75 -> u_p_beta

  void ReconstructShift();  // CFC_ReconstructBeta: beta_u from u_p_beta

  // Final assembly: vK_dd, alpha, beta_u -> padm->u_adm (psi4/g_dd already written
  // by SolveConformalFactor).
  void AssembleADM();

  // One pass of "solve X^i, ghost-exchange, compute Adual^ij/Ahat^2, solve psi" --
  // extracted from InitializeMetric()'s Picard-loop body so it can be called either
  // repeatedly (the iterative default) or exactly once (<cfc>
  // init_freeze_conserved=true). Callers are responsible for refreshing conserved
  // variables from primitives (PrimToConInit) before calling this when
  // appropriate -- see InitializeMetric()'s and ReinitializeMetricForAMR's own call
  // sites for which do and don't.
  void RunXPsiSolvePass(Driver *pdriver);

  // Tail of InitializeMetric()/ReinitializeMetricForAMR: RescaleMatterSources/
  // SolveLapse/SolveShift/AssembleADM. Callers must reconcile primitives vs.
  // conserved variables themselves before calling this, in whichever direction
  // matches what was held fixed during the solve just completed (PrimToConInit if
  // primitives were fixed, ConToPrim if conserved variables were fixed --
  // mirroring the per-stage task graph's own MHD_C2P, which runs right after
  // CFC_SolvePsi for the same reason).
  void RunLapseShiftAssemblePass(Driver *pdriver);

  // InitializeMetric() runs its X^i/psi loop outside the normal per-stage task
  // graph, so it can't use the all-field InitRecvTask/ClearSendTask/ClearRecvTask
  // above -- those would post MPI_Irecv's for fields not sent this iteration, and
  // the matching ClearRecv would hang waiting on them. These four scope
  // Init/Clear to exactly the fields each of InitializeMetric's two phases
  // actually exercises.
  void InitRecvXFields();
  void ClearXFields();
  void InitRecvTailFields();
  void ClearTailFields();
};

}  // namespace cfc

#endif  // CFC_CFC_HPP_
