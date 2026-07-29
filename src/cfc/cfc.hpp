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

  DvceArray5D<Real> u_x;
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> x_u;      // X^i, vector potential (eq. 72)

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
  // u_p_x/u_p_beta directly. Reconstructed into x_u/beta_u by
  // cfc::ReconstructVectorFromPotentials.
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
  // Under-relaxation for InitializeMetric()'s psi update (1.0 = no relaxation).
  // Needed for compact/relativistic stars where the plain Picard iteration
  // diverges.
  Real cfc_init_omega_;

  // <cfc> init_use_psi5_source (default true): InitializeMetric()'s
  // SolveConformalFactor() calls use the self-consistent U_raw*psi^5 Newton
  // formulation instead of Utilde*psi^-1 (see mg_cfc_conformal_factor.cpp's
  // ConformalFactorRHS). Converges much faster for most stars but diverges to NaN
  // for very compact ones (e.g. the migration test's unstable star), which must
  // set this false explicitly. Never affects the per-stage CFC_SolvePsi task
  // (always Utilde*psi^-1).
  //
  // Mutually incompatible with cfc_init_freeze_conserved_ below (constructor
  // forces this false when that's true): psi5 holds U_raw (primitives) fixed
  // while psi iterates; freeze_conserved holds Utilde fixed instead --
  // contradictory assumptions.
  bool cfc_init_use_psi5_;

  // <cfc> init_freeze_conserved (default false): InitializeMetric() calls
  // RunXPsiSolvePass() exactly once instead of Picard-iterating it -- Utilde/
  // S-tilde_i are built once from the pgen's initial metric guess and held fixed,
  // rather than rebuilt from fixed primitives every outer iteration ("the same as
  // one CFC step", no outer loop). The resulting metric's implied primitives may
  // not exactly match the pgen's original ones -- the same tradeoff every
  // per-stage CFC_SolvePsi call already lives with. See cfc_init_use_psi5_'s
  // comment above for the mutual-exclusivity rule.
  bool cfc_init_freeze_conserved_;

  // Post-multigrid ghost exchange: one MeshBoundaryValuesCC + coarse shadow array
  // per field cfc_reconstruct.cpp differentiates (mirrors z4c::Z4c::pbval_u/
  // coarse_u0; is_z4c=false throughout). coarse_* arrays are only sized under
  // SMR/AMR (RestrictCC/ProlongateCC are no-ops otherwise). u_p_x/u_p_beta each get
  // one round covering all 4 packed channels at once.
  MeshBoundaryValuesCC *pbval_pietax, *pbval_x;
  MeshBoundaryValuesCC *pbval_psi, *pbval_alpha_psi;
  MeshBoundaryValuesCC *pbval_pietabeta;
  DvceArray5D<Real> coarse_u_pietax, coarse_u_x;
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
  // (e.g. a TOV profile) but never through CFC's own solve, so this converges it --
  // iterating X^i/psi via PrimToCon(Init) <-> CFC's vector-Poisson/conformal-factor
  // solve until psi stops changing, holding primitives fixed at the pgen's values.
  // Called once from Driver::Initialize(), gated on !res_flag (a restart's
  // checkpointed metric is already self-consistent with its checkpointed conserved
  // variables). Non-convergence is a warning, not fatal. Takes a real Driver* (not
  // nullptr): the multigrid solves' own iteration caps use it to truncate the run
  // on failure.
  //
  // <cfc> init_freeze_conserved (default false) selects a one-shot mode instead --
  // see cfc_init_freeze_conserved_'s comment above.
  //
  // Finishes by re-running Driver::InitBoundaryValuesAndPrimitives() (the same
  // call already run once before this solve): PrimToConInit/ConToPrim above only
  // reconcile one of {pmhd->u0, pmhd->w0} (and, for PrimToConInit, only the
  // interior), so the other quantity's ghost cells -- or the metric ghosts
  // ConToPrim read, stale until RunLapseShiftAssemblePass's own padm->u_adm
  // exchange completes -- would otherwise stay inconsistent with the just-solved
  // metric until the next real hydro ghost exchange.
  void InitializeMetric(Driver *pdriver);

  // Regrid counterpart to InitializeMetric(), called once per regrid event from
  // inside Driver::InitBoundaryValuesAndPrimitives itself (gated on that
  // function's is_amr_regrid flag), mirroring where z4c::Z4c::ConvertZ4cToADM
  // sits relative to ConToPrim -- see that function's own comment. Does NOT
  // call InitializeMetric() -- that would rebuild conserved variables from
  // primitives (PrimToConInit), discarding the already-evolved, correctly
  // remapped, mass-conserving pmhd->u0. Instead runs one plain CFC step
  // holding conserved variables fixed, with its own interior-then-ghost
  // con2prim split standing in for the driver-level ConToPrim call this
  // replaces: an interior-only pass right after solving X^i/psi (all
  // RescaleMatterSources needs), then a ghost-shell-only pass right after
  // solving lapse/shift (once padm->u_adm's own ghost exchange, at that
  // pass's own tail, has completed) -- so hydro's own ghost cells never read
  // a stale/zero metric, without redundantly re-exchanging pmhd->u0's own
  // ghosts a second time (already done earlier in the same enclosing driver
  // call).
  //
  // Resets psi_seeded_/alpha_psi_seeded_ so the two nonlinear Newton solves
  // re-seed from adm.psi4/alpha (safe even across a cross-rank regrid move, since
  // CFC's own re-solve unconditionally overwrites every u_adm field every call --
  // whatever it holds beforehand is only ever a best-effort initial guess, never
  // doubled down on). padm->u_adm itself IS correctly transferred across a
  // cross-rank regrid move: adm::ADM owns it directly (coordinates/adm.cpp) and
  // it's wired into the generic AMR pipeline (src/mesh/mesh_refinement.cpp +
  // load_balance.cpp) as its own independent physics-module block, the same way
  // pz4c->u0 is.
  //
  // mesh_refinement.cpp also skips the generic padm->SetADMVariables(...) regrid
  // callback for CFC runs (pcfc == nullptr guard) -- that callback re-derives the
  // metric from the pgen's static t=0 ID, which would otherwise discard CFC's
  // dynamical metric evolution on every regrid.
  void ReinitializeMetricForAMR(Driver *pdriver);

  // Task-graph entry points (TaskStatus(Driver*, int), required by
  // NumericalRelativity::QueueTask). Each wraps the like-named private Step method
  // below; kept separate so the "Step N" structure stays visible.
  TaskStatus SolveVecXTask(Driver *pdriver, int stage);       // step 1 (CFC_BuildSrcX)
  TaskStatus ReconstructXTask(Driver *pdriver, int stage);    // CFC_ReconstructX
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
  TaskStatus RestXTask(Driver *pdriver, int stage);
  TaskStatus SendXTask(Driver *pdriver, int stage);
  TaskStatus RecvXTask(Driver *pdriver, int stage);
  TaskStatus ProlongXTask(Driver *pdriver, int stage);
  TaskStatus BCSXTask(Driver *pdriver, int stage);
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

  // Step 1: build eq. 72's source from pmhd->u0, solve pmgd_pietax (packed P_i,
  // eta) into u_p_x. x_u itself is reconstructed later, after u_p_x's own ghost
  // exchange (see ReconstructVectorPotential).
  void SolveVectorPotential(Driver *pdriver, int stage);

  void ReconstructVectorPotential();  // CFC_ReconstructX: x_u from u_p_x

  void ComputeADual();  // step 2: Adual^ij (eq. 76) and Ahat^2, from x_u

  // Step 3: solve eq. 73 for psi (nonlinear), then write psi4/g_dd into
  // padm->u_adm -- the con2prim shared with dyn_grmhd (MHD_C2P) depends on this
  // step, needing a valid g_dd. use_psi5_source selects the alternate Newton
  // formulation (see cfc_init_use_psi5_'s comment); only InitializeMetric() ever
  // passes true.
  void SolveConformalFactor(Driver *pdriver, int stage, bool use_psi5_source = false);

  // Step 4: rebuild S-tilde (trace of S_ij) from the primitives MHD_C2P just
  // recovered -- no con2prim here (RescaleSrcTask depends on MHD_C2P). Utilde and
  // S-tilde_i don't need rebuilding: already built directly from the
  // psi^6-densitized conserved state in steps 1/3.
  void RescaleMatterSources(Driver *pdriver, int stage);

  void SolveLapse(Driver *pdriver, int stage);  // step 5: alpha*psi (nonlinear, eq. 74)

  // Step 6: build eq. 75's source (needs psi/alpha_psi's own ghost exchange first),
  // solve pmgd_pietabeta into u_p_beta. beta_u reconstructed later (see
  // ReconstructShift).
  void SolveShift(Driver *pdriver, int stage);

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
