#ifndef CFC_CFC_HPP_
#define CFC_CFC_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cfc.hpp
//! \brief defines the CFC class: the conformally-flat-condition (XCFC) metric solver.
//!
//! Mirrors gravity::Gravity in that it's a thin physics-facing orchestrator that owns
//! a set of Multigrid/MultigridDriver subclass pairs (see mg_cfc_vector_poisson.hpp,
//! mg_cfc_conformal_factor.hpp, mg_cfc_lapse.hpp) driving
//! them through the 6-step XCFC algorithm (Cheong et al. 2021 [arXiv:2012.07322]
//! sec. 2.6). Unlike gravity::Gravity, CFC does NOT call Driver::Execute() directly:
//! its steps must interleave with dyn_grmhd's own per-stage task graph (hydro
//! flux/update, then CFC's vector potential + conformal factor, then a single
//! conserved-to-primitive recovery shared with dyn_grmhd, then the remaining CFC
//! steps), which is only possible by queuing individual tasks into the shared
//! NumericalRelativity task graph (see QueueCFCTasks() and tasklist/
//! numerical_relativity.hpp) exactly as dyngr::DynGRMHD and z4c::Z4c already do.
//! Reads matter data directly from MeshBlockPack::pmhd->u0/w0 (not
//! MeshBlockPack::ptmunu -- see cfc.cpp) and writes the resulting metric into
//! MeshBlockPack::padm->u_adm (consumed by dyn_grmhd's Riemann solver/ConToPrim).
//!
//! Every multigrid solve here only converges its own (generally shallow) ghost width
//! ngh_ -- shallower than the mesh's own NGHOST that cfc_reconstruct.cpp's finite
//! differences need. Fields that get differentiated after a solve (u_p_x/x_u,
//! u_p_beta, delta_psi/delta_alpha_psi) therefore each get one MeshBoundaryValuesCC
//! Rest->Send->Recv->Prolong round (mirroring z4c::Z4c::pbval_u/coarse_u0 exactly,
//! is_z4c=false throughout) between RetrieveSolution() and whatever differentiates
//! them next -- see QueueCFCTasks()'s dependency list below for exactly where each
//! round sits in the per-stage pipeline. u_p_x/u_p_beta each pack P_i (channels 0-2)
//! and its paired scalar eta (channel 3) into one array, one solve, and one ghost
//! round (see mg_cfc_vector_poisson.hpp).

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

  // intermediate fields, all defined on the finest mesh grid. Vector/tensor physical
  // quantities are represented as AthenaTensor views (as in the z4c/adm modules,
  // e.g. adm::ADM::ADM_vars), each shallow-sliced (InitWithShallowSlice) from an
  // underlying flat "u_*" storage array; genuine scalars remain plain
  // DvceArray5D<Real> (as gravity::Gravity::phi does). All are sized at mesh-NGHOST
  // depth (nmb, ncomp, nx3+2*ng, nx2+2*ng, nx1+2*ng), matching gravity::Gravity::phi.

  DvceArray5D<Real> u_x;                              // storage backing x_u (3 comp.)
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> x_u;      // X^i, vector potential (eq. 72)

  DvceArray5D<Real> u_beta;                            // storage backing beta_u (3 comp.)
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> beta_u;   // beta^i, shift vector

  DvceArray5D<Real> u_adual;                           // storage backing a_dd (6 comp.)
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> a_dd;     // Adual^ij (Gmunu eq. 76)

  DvceArray5D<Real> a_sq;        // Ahat^2 = f_ik f_jl Adual^kl Adual^ij, scalar

  // Store the deviation from the flat-space asymptotic value (1), not the physical
  // field itself: delta_psi = psi - 1, delta_alpha_psi = alpha*psi - 1. This is
  // exactly the unknown the multigrid solve already iterates on internally
  // (RetrieveSolution hands back the raw deviation verbatim, see
  // mg_cfc_conformal_factor.hpp/mg_cfc_lapse.hpp), so storing it here too -- rather
  // than adding 1 back after every solve, as an earlier version of this code did --
  // both drops that pointwise pass and keeps more significant digits in the
  // (frequently tiny, far from the star) deviation than a Real holding ~1+1e-12
  // ever could. Every consumer that needs the physical field reconstructs it inline
  // (+1.0) at the point of use: AssembleConformalMetric/AssembleLapseShiftK
  // (cfc_reconstruct.cpp), AssembleVectorSource's for_shift branch and
  // BuildShiftSource (cfc.cpp), and MGCFCLapseDriver::LoadReactionCoefficient
  // (mg_cfc_lapse.cpp).
  DvceArray5D<Real> delta_psi;         // psi - 1, scalar
  DvceArray5D<Real> delta_alpha_psi;   // alpha*psi - 1, scalar

  // matter source terms rescaled by the current psi^6 (Gmunu sec. 2.6, U-tilde etc.).
  // U and S_i are built directly from the evolved conserved state
  // (pmy_pack->pmhd->u0: D, S_i, tau -- see AssembleVectorSource/SolveConformalFactor
  // in cfc.cpp), NOT from MeshBlockPack::ptmunu: Tmunu is only populated when a z4c
  // block is active (dyn_grmhd.cpp's QueueDynGRMHDTasks), but CFC's primary use case
  // has no z4c free evolution, so ptmunu may not exist at all.
  DvceArray5D<Real> u_tilde;       // Ũ = psi^6 U, scalar

  // 2026-07-25: raw (undensitized) U = Utilde/sqrt(detg), computed alongside
  // u_tilde in AssembleVectorSource (an exact un-densitization, using the same
  // g_dd that built the conserved state u_tilde derives from -- not a re-
  // derivation from primitives). Used only by SolveConformalFactor's psi^5
  // formulation (see mg_cfc_conformal_factor.cpp's ConformalFactorRHS doc
  // comment), which InitializeMetric() uses by default (<cfc>
  // init_use_psi5_source, default true since item 28) -- unused (but still
  // computed; cheap) only for inputs that explicitly opt out, and never read
  // by the per-stage CFC_SolvePsi task.
  DvceArray5D<Real> u_raw;

  DvceArray5D<Real> u_stilde;                            // storage backing s_tilde_d
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> s_tilde_d;  // S-tilde_i = psi^6 S_i

  DvceArray5D<Real> s_tilde;       // S-tilde = psi^6 S (trace of S_ij), scalar

  // Shibata (1999) sec. 3 decomposition: each vector equation (X^i, beta^i) reduces
  // to one vector potential P_i (eq. 3.10: Delta P_i = S_i) plus one scalar
  // potential eta (eq. 3.11: Delta eta = -S_i x^i). P_i and eta are mutually
  // independent equations sharing the same known vector source S_i, so both are
  // packed into one nvar_=4 array (P_i at channels 0-2, eta at channel 3) and solved
  // together in a single MGCFCVectorPoissonDriver::Solve() call (see
  // mg_cfc_vector_poisson.hpp) -- there is no separate eta storage or eta-specific
  // driver. p_x is a 3-component AthenaTensor shallow-slice over channels 0-2 of
  // u_p_x; eta itself has no dedicated view, it's read/written as channel 3 of
  // u_p_x directly (see AssembleVectorSource/ReconstructVectorFromPotentials in
  // cfc.cpp). Reconstructed into x_u/beta_u by
  // cfc::ReconstructVectorFromPotentials.
  //
  // Unlike u_p_src below, these are the OUTPUTS of a multigrid solve that
  // cfc_reconstruct.cpp then finite-differences -- sized at mesh-NGHOST depth (like
  // x_u/psi above), NOT this solver's own (shallower) ngh_ depth, and given their
  // own post-RetrieveSolution MeshBoundaryValuesCC round (pbval_pietax below)
  // before ReconstructVectorFromPotentials touches them (plan addendum #4, Finding
  // F). MGCFCVectorPoissonDriver::RetrieveSolution already accounts for this size
  // mismatch internally.
  DvceArray5D<Real> u_p_x;                        // storage backing p_x AND eta (ch.
                                                   // 0-2 = P_i, ch. 3 = eta), 4 comp.
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> p_x; // P_i view (ch. 0-2 of u_p_x)

  DvceArray5D<Real> u_p_beta;                     // storage backing p_beta AND eta
                                                   // (ch. 0-2 = P_i, ch. 3 = eta)
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> p_beta;  // P_i view (ch. 0-2 of u_p_beta)

  // P_i/eta's own packed right-hand side (eq. 72/75's S_i at channels 0-2, and
  // -S_i.x^i at channel 3), built by AssembleVectorSource() and consumed by
  // LoadPoissonSource() on the multigrid driver below. Genuinely ngh_-deep (this
  // solver's own multigrid ghost width, via GetGhostCells()), NOT mesh-NGHOST-deep
  // like u_p_x above: this is a pure LoadSource() input, read pointwise on the
  // interior+ngh_ ring only, never finite-differenced and never ghost-exchanged
  // (contrast with u_p_x's opposite sizing, immediately above -- easy to conflate,
  // hence the explicit contrast here). p_src needs its own storage separate from
  // x_u/p_x/etc. (rather than a local temporary inside AssembleVectorSource) because
  // MGCFCVectorPoissonDriver::LoadPoissonSource() takes a raw DvceArray5D<Real>&, not
  // an AthenaTensor -- an AthenaTensor's backing storage is a Kokkos::subview result,
  // a different type that does not bind to Multigrid::LoadSource()'s (non-templated)
  // parameter. eta's source (channel 3) is written/read as a raw index into u_p_src
  // directly, no dedicated view. Shared sequentially by the X^i and beta^i solves
  // (never needed simultaneously).
  DvceArray5D<Real> u_p_src;                        // storage backing p_src AND eta's
                                                     // source (ch. 0-2 = S_i, ch. 3 =
                                                     // -S_i.x^i), 4 comp.
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> p_src; // S_i view (ch. 0-2 of u_p_src)
  // Ghost width u_p_src was allocated with (the "cfc"/"mg_nghost" input
  // parameter, cached here since AssembleVectorSource/BuildShiftSource need it on
  // every call to translate mesh-NGHOST-indexed loop indices into this shallower
  // array's own index space -- see AssembleVectorSource's definition.
  int mg_nghost_;

  // multigrid solvers, one per distinct elliptic equation (one MGCFCVectorPoisson-
  // Driver per Shibata pair, packing P_i+eta into a single nvar_=4 solve -- see
  // mg_cfc_vector_poisson.hpp)
  MGCFCVectorPoissonDriver *pmgd_pietax;      // solves X^i's packed (P_i, eta)
  MGCFCVectorPoissonDriver *pmgd_pietabeta;   // solves beta^i's packed (P_i, eta)
  MGCFCConformalFactorDriver *pmgd_psi;
  MGCFCLapseDriver *pmgd_alpha;

  // Item 10 (DEVELOPMENT.md): one-shot flags so the very first CFC_SolvePsi/
  // CFC_SolveLapse call ever made seeds its V-cycle's initial guess from the
  // problem generator's own ADM data (padm->adm.psi4/alpha) instead of a cold
  // Kokkos-zero start -- see SolveConformalFactor/SolveLapse in cfc.cpp. False by
  // construction (both a fresh run and a restart reconstruct CFC from scratch, and
  // in both cases padm->adm holds a real, non-default metric guess -- the pgen's
  // analytic profile on a fresh run, the checkpointed converged metric on a
  // restart -- by the time the first real Solve() call happens; see cfc.cpp's
  // doc comment on SolveConformalFactor for why seeding can't happen here in the
  // constructor itself). Every subsequent call is left alone (its own natural
  // multigrid warm start, whatever u_[finest] already holds from the previous
  // stage/cycle, untouched by this).
  bool psi_seeded_ = false;
  bool alpha_psi_seeded_ = false;

  // Item 11 (DEVELOPMENT.md): InitializeMetric()'s X^i/psi fixed-point-iteration
  // controls -- see InitializeMetric's public doc comment above.
  int cfc_init_iter_max_;
  Real cfc_init_tol_;
  bool cfc_init_verbose_;
  // Under-relaxation factor for InitializeMetric()'s psi update (1.0 = no
  // relaxation, byte-identical to the original unrelaxed Picard iteration). Added
  // after the iteration was found to diverge for a more compact/relativistic star
  // than any previously tested here (2M/R~0.34) -- see InitializeMetric's doc
  // comment for the mechanism. Default 1.0 keeps every existing test unaffected.
  Real cfc_init_omega_;

  // 2026-07-25: <cfc> init_use_psi5_source (default true, see item 27/28,
  // DEVELOPMENT.md). When true, InitializeMetric()'s SolveConformalFactor()
  // calls use the self-consistent U_raw*psi^5 Newton formulation instead of
  // the Utilde*psi^-1 one -- see mg_cfc_conformal_factor.cpp's
  // ConformalFactorRHS doc comment. Converges much faster and just as
  // accurately for every star tested so far EXCEPT the migration test's
  // compact/unstable star, which diverges to NaN under it -- inputs known to
  // be that compact MUST explicitly set this false. Does NOT affect the
  // per-stage CFC_SolvePsi task, which always uses the default (false/
  // Utilde*psi^-1) formulation regardless of this setting.
  bool cfc_init_use_psi5_;

  // 2026-07-25: <cfc> init_freeze_conserved (default false, see item 29,
  // DEVELOPMENT.md). When true, InitializeMetric() calls RunXPsiSolvePass()
  // exactly ONCE instead of iterating it in a Picard loop: Utilde/S-tilde_i
  // (the weighted/densitized conserved matter source) are built once from the
  // pgen's own initial metric guess and held fixed, rather than being rebuilt
  // from fixed primitives against an evolving metric every outer iteration --
  // "the same as doing one CFC step," with no outer convergence loop at all.
  // Motivated by item 24's own finding that a single V-cycle seeded from the
  // exact analytic solution is highly accurate for the migration test's
  // compact star, unlike the ~80-iteration outer loop, which drifts to a
  // wrong fixed point. The resulting metric's implied primitives may not
  // exactly match the pgen's original ones -- an accepted tradeoff, the same
  // one every per-stage CFC_SolvePsi call already lives with. Orthogonal to
  // cfc_init_use_psi5_ above (RunXPsiSolvePass respects whatever that's set
  // to either way).
  bool cfc_init_freeze_conserved_;

  // Post-multigrid ghost exchange, one MeshBoundaryValuesCC + coarse shadow array per
  // field that cfc_reconstruct.cpp later finite-differences (mirrors
  // z4c::Z4c::pbval_u/coarse_u0 exactly; is_z4c=false throughout since CFC is not
  // z4c). coarse_* arrays are only sized when pmy_pack->pmesh->multilevel (see
  // cfc.cpp's constructor, matching z4c.cpp's identical guard) -- RestrictCC/
  // ProlongateCC are internally no-ops otherwise, same as z4c's own Rest/Prolong
  // tasks. u_p_x (P_i packed with eta) gets one round covering all 4 channels at
  // once (PackAndSendCC et al. take one (fine,coarse) array pair, and both channel
  // ranges now live in the same array) -- likewise u_p_beta and
  // delta_psi/delta_alpha_psi (the pbval_psi/coarse_psi names below are kept as-is,
  // referring to "the psi field's exchange machinery" regardless of the delta-vs-
  // physical storage convention the underlying array uses).
  MeshBoundaryValuesCC *pbval_pietax, *pbval_x;
  MeshBoundaryValuesCC *pbval_psi, *pbval_alpha_psi;
  MeshBoundaryValuesCC *pbval_pietabeta;
  DvceArray5D<Real> coarse_u_pietax, coarse_u_x;
  DvceArray5D<Real> coarse_psi, coarse_alpha_psi;
  DvceArray5D<Real> coarse_u_pietabeta;

  // Round 19 fix: AssembleConformalMetric/AssembleLapseShiftK (cfc_reconstruct.cpp)
  // only ever write pmy_pack->padm->u_adm's INTERIOR (is..ie) -- unlike z4c's
  // Z4cToADM, which fills u_adm over the full ghost-inclusive extent by converting
  // from z4c's own already-ghost-exchanged state (z4c_adm.cpp's explicit "sets the
  // ADM variables everywhere in the MeshBlock" comment). CFC has no such implicit
  // mechanism, so without this round u_adm's ghost cells were never updated past
  // whatever the pgen set at t=0 -- invisible in every single-MeshBlock test this
  // investigation ran (rounds 8-18), but silently wrong at every inter-MeshBlock
  // boundary in a real multi-MeshBlock run (MHD_C2P/MHD_Flux both read u_adm over
  // the full array extent, ghosts included). One combined round for the whole
  // u_adm array (all adm::ADM::nadm channels: g_dd, vK_dd, psi4, alpha, beta_u --
  // CFC never runs with z4c active, so none of these channels alias into a
  // z4c-owned array the way adm::ADM::ADM(...) does when pz4c != nullptr), run
  // once per stage right after CFC_AssembleFinal.
  MeshBoundaryValuesCC *pbval_adm;
  DvceArray5D<Real> coarse_u_adm;

  // Queues this module's tasks into the shared NumericalRelativity task graph
  // (pmy_pack->pnr), mirroring dyngr::DynGRMHD::QueueDynGRMHDTasks()/
  // z4c::Z4c::QueueZ4cTasks(). Called once, from NumericalRelativity::
  // AssembleNumericalRelativityTasks() (tasklist/numerical_relativity.cpp), NOT
  // called directly from Driver::Execute() the way gravity::Gravity is: CFC's steps
  // must interleave with dyn_grmhd's own hydro/con2prim tasks (see cfc.cpp and
  // tasklist/numerical_relativity.hpp's CFC_* TaskName values), which is only
  // possible through the task graph. Full per-stage chain (see plan addendum #3 for
  // the packed-(P_i,eta) merge that collapsed the PX+EtaX and PBeta+EtaBeta ghost-
  // exchange groups down to one PiEtaX/PiEtaBeta group each, and addendum #4 for the
  // complete original dependency derivation):
  //   CFC_BuildSrcX      depends on {MHD_AddSrc} (post flux+source-update u0);
  //                      solves the packed (P_i, eta) for X^i, retrieves into u_p_x.
  //   CFC_Rest/Send/Recv/ProlongPiEtaX   ghost-exchange u_p_x (all 4 channels).
  //   CFC_ReconstructX   depends on {CFC_ProlongPiEtaX}; builds x_u.
  //   CFC_Rest/Send/Recv/ProlongX             ghost-exchange x_u.
  //   CFC_ComputeADual   depends on {..ProlongX}; builds a_dd/a_sq.
  //   CFC_SolvePsi       depends on {CFC_ComputeADual}; writes psi4/g_dd.
  //   CFC_RescaleSrc     depends on {MHD_C2P} (the SAME con2prim dyn_grmhd already
  //                      runs, queued to depend on CFC_SolvePsi -- see below).
  //   CFC_SolveLapse     depends on {CFC_RescaleSrc}.
  //   CFC_Rest/Send/Recv/ProlongPsi, ...AlphaPsi   ghost-exchange psi, alpha_psi.
  //   CFC_BuildSrcBeta   depends on {..ProlongPsi, ..ProlongAlphaPsi}; solves the
  //                      packed (P_i, eta) for beta^i, retrieves into u_p_beta.
  //   CFC_Rest/Send/Recv/ProlongPiEtaBeta   ghost-exchange u_p_beta (all 4 channels).
  //   CFC_ReconstructBeta   depends on {CFC_ProlongPiEtaBeta}; builds beta_u.
  //   CFC_AssembleFinal  depends on {CFC_ReconstructBeta}; writes vK_dd/alpha/beta_u.
  //   CFC_Rest/Send/Recv/ProlongADM  depends on {CFC_AssembleFinal}; ghost-exchanges
  //                      the whole padm->u_adm array (round 19 fix -- see pbval_adm's
  //                      doc comment above).
  // dyn_grmhd.cpp's MHD_C2P/MHD_Newdt tasks in turn take CFC_SolvePsi/
  // CFC_AssembleFinal as *optional* dependencies, so a single con2prim per stage
  // serves both dyn_grmhd's own needs and CFC's (no second con2prim call here).
  void QueueCFCTasks();

  // Item 11 (DEVELOPMENT.md): one-time initialization. The problem generator sets
  // padm->adm directly (e.g. a 1D TOV profile mapped onto the 3D grid) -- never
  // passed through CFC's own constraint solve, so it's generally NOT the actual
  // self-consistent CFC solution for that matter distribution. This converges it:
  // holds the primitives (pmhd->w0) exactly as the pgen set them, and iterates
  // X^i/psi (the only mutually-coupled pair -- alpha/beta don't feed back into
  // either) via PrimToCons (metric-dependent) <-> CFC's vector-Poisson/conformal-
  // factor solve, until psi stops changing. Called once from Driver::Initialize(),
  // gated on !res_flag (a restart's checkpointed metric is already self-consistent
  // with its checkpointed conserved variables -- re-deriving it from primitives
  // would discard that, not just redundantly recompute it). Non-convergence within
  // init_iter_max iterations is a warning, not fatal (see InitializeMetric's body).
  // Takes a real Driver* (not nullptr): the per-field multigrid solves' own
  // internal iteration caps use it to gracefully truncate the run on failure
  // (pdriver->nlim = ...), which would segfault on a null pointer.
  //
  // 2026-07-25 (item 29, DEVELOPMENT.md): <cfc> init_freeze_conserved (default
  // false) selects a second mode -- instead of the Picard loop described above
  // (fixed primitives, conserved state and X^i/psi re-solved every iteration
  // against an evolving metric), call RunXPsiSolvePass() exactly ONCE, freezing
  // Utilde/S-tilde_i at whatever the pgen's own initial metric guess implies.
  // See cfc_init_freeze_conserved_'s doc comment (below) for the full rationale.
  void InitializeMetric(Driver *pdriver);

  // Task-graph entry points (TaskStatus(Driver*, int) is the signature
  // NumericalRelativity::QueueTask requires). Each is a thin wrapper around the
  // like-named private Step method below; kept separate so the "Step N" structure
  // from the original design doc stays visible.
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
  // field (see the MeshBoundaryValuesCC members above). Each is a thin one-liner
  // mirroring z4c::Z4c::RestrictU/SendU/RecvU/Prolongate's exact shape.
  TaskStatus RestPiEtaXTask(Driver *pdriver, int stage);
  TaskStatus SendPiEtaXTask(Driver *pdriver, int stage);
  TaskStatus RecvPiEtaXTask(Driver *pdriver, int stage);
  TaskStatus ProlongPiEtaXTask(Driver *pdriver, int stage);
  TaskStatus RestXTask(Driver *pdriver, int stage);
  TaskStatus SendXTask(Driver *pdriver, int stage);
  TaskStatus RecvXTask(Driver *pdriver, int stage);
  TaskStatus ProlongXTask(Driver *pdriver, int stage);
  TaskStatus RestPsiTask(Driver *pdriver, int stage);
  TaskStatus SendPsiTask(Driver *pdriver, int stage);
  TaskStatus RecvPsiTask(Driver *pdriver, int stage);
  TaskStatus ProlongPsiTask(Driver *pdriver, int stage);
  TaskStatus RestAlphaPsiTask(Driver *pdriver, int stage);
  TaskStatus SendAlphaPsiTask(Driver *pdriver, int stage);
  TaskStatus RecvAlphaPsiTask(Driver *pdriver, int stage);
  TaskStatus ProlongAlphaPsiTask(Driver *pdriver, int stage);
  TaskStatus RestPiEtaBetaTask(Driver *pdriver, int stage);
  TaskStatus SendPiEtaBetaTask(Driver *pdriver, int stage);
  TaskStatus RecvPiEtaBetaTask(Driver *pdriver, int stage);
  TaskStatus ProlongPiEtaBetaTask(Driver *pdriver, int stage);
  // Round 19 fix: ghost-exchange padm->u_adm after AssembleFinalTask -- see
  // pbval_adm's doc comment above.
  TaskStatus RestADMTask(Driver *pdriver, int stage);
  TaskStatus SendADMTask(Driver *pdriver, int stage);
  TaskStatus RecvADMTask(Driver *pdriver, int stage);
  TaskStatus ProlongADMTask(Driver *pdriver, int stage);

  // Round 19 fix: post the non-blocking MPI receives for all 6 ghost-exchange
  // rounds above once, up front (Task_Start), and wait on the outstanding
  // sends/receives once, at the very end (Task_End) -- mirrors z4c::Z4c::InitRecv/
  // ClearSend/ClearRecv (z4c_tasks.cpp) exactly, just batched over CFC's 6 fields
  // instead of z4c's single u0. Without these, PackAndSendCC still issues a real
  // MPI_Isend (it self-builds its own buffer/offset metadata on first use), but
  // RecvAndUnpackCC's completion check tests an MPI_Request that was never posted
  // (stuck at MPI_REQUEST_NULL, which MPI_Test/MPI_Wait always report as complete
  // on), so the "receive" silently unpacks whatever zero-initialized garbage sits
  // in the never-actually-filled aggregate recv buffer instead of the neighbor's
  // real data. Each of the 6 MeshBoundaryValuesCC instances owns its own
  // MPI_Comm_dup'd communicator (bvals.cpp), so batching all 6 into one
  // Task_Start/two Task_End tasks is safe -- no cross-field message collisions,
  // and the before_stagen/stagen/after_stagen task-list phases already run as
  // fully separate, sequential passes (driver.cpp), so Task_End is guaranteed to
  // run only after every Task_Run task (including CFC_ProlongADM) has completed.
  TaskStatus InitRecvTask(Driver *pdriver, int stage);
  TaskStatus ClearSendTask(Driver *pdriver, int stage);
  TaskStatus ClearRecvTask(Driver *pdriver, int stage);

 private:
  // shared helper: build the Shibata (1999) eq. 3.10-3.11 sources into the packed
  // member array u_p_src/p_src (P_i's vector right-hand side S_i at channels 0-2,
  // eta's scalar right-hand side -S_i x^i at channel 3, built from that same S_i) --
  // for either the X^i solve (for_shift=false, built directly from the
  // post-source-update conserved momentum pmy_pack->pmhd->u0 per eq. 72) or the
  // beta^i solve (for_shift=true, built from alpha, psi, Adual^ij, S-tilde_i per
  // eq. 75). P_i and eta are independent equations sharing one source array and one
  // solve (see SolveVectorPotential/SolveShift below).
  void AssembleVectorSource(bool for_shift);

  // Step 1: build the eq. 72 source directly from pmy_pack->pmhd->u0 (the conserved
  // state right after this stage's hydro flux+source update -- see AssembleVectorSource),
  // solve pmgd_pietax for the packed (P_i, eta) (u_p_x). Does NOT reconstruct x_u
  // itself (needs u_p_x's own ghost exchange first -- see ReconstructVectorPotential,
  // CFC_ReconstructX).
  void SolveVectorPotential(Driver *pdriver, int stage);

  // CFC_ReconstructX: cfc::ReconstructVectorFromPotentials(pmy_pack, p_x, u_p_x, x_u,
  // 3) once u_p_x's post-solve ghost exchange (pbval_pietax) has completed.
  void ReconstructVectorPotential();

  // Step 2: Adual^ij from X^i (eq. 76), then Ahat^2 (cfc_reconstruct.hpp). Runs after
  // x_u's own ghost exchange (pbval_x).
  void ComputeADual();

  // Step 3: solve eq. 73 for psi (nonlinear), then immediately write psi4/g_dd into
  // pmy_pack->padm->u_adm via cfc::AssembleConformalMetric -- the single con2prim
  // shared with dyn_grmhd (MHD_C2P) needs a valid g_dd to invert conserved to
  // primitive variables, and is queued to depend on this step (see QueueCFCTasks).
  // use_psi5_source (default false, the per-stage CFC_SolvePsi task's behavior):
  // selects the alternate self-consistent Newton formulation (2026-07-25), see
  // cfc_init_use_psi5_'s doc comment above -- only InitializeMetric() ever passes
  // true.
  void SolveConformalFactor(Driver *pdriver, int stage, bool use_psi5_source = false);

  // Step 4: build S-tilde (trace of S_ij, needed by the lapse equation) from the
  // primitives MHD_C2P just recovered (pmy_pack->pmhd->w0) and EOS enthalpy -- no
  // con2prim call here: RescaleSrcTask depends on MHD_C2P, so w0 is already fresh
  // against the psi/g_dd SolveConformalFactor() wrote. Ũ and S-tilde_i do NOT need
  // rebuilding here either: they were already built directly from the
  // psi^6-densitized evolved conserved state (pmy_pack->pmhd->u0: D, S_i, tau) in
  // steps 1 and 3, since sqrt(gamma) = psi^6 is already baked into those conserved
  // variables -- only the trace source requires primitives.
  void RescaleMatterSources(Driver *pdriver, int stage);

  // Step 5: solve eq. 74 for alpha*psi (nonlinear); extract alpha = (alpha*psi)/psi.
  void SolveLapse(Driver *pdriver, int stage);

  // Step 6: build the eq. 75 source (needs psi/alpha_psi's own ghost exchange
  // first, see QueueCFCTasks), solve pmgd_pietabeta for the packed (P_i, eta)
  // (u_p_beta). Does NOT reconstruct beta_u itself (needs u_p_beta's own ghost
  // exchange first -- see ReconstructShift, CFC_ReconstructBeta).
  void SolveShift(Driver *pdriver, int stage);

  // CFC_ReconstructBeta: cfc::ReconstructVectorFromPotentials(pmy_pack, p_beta,
  // u_p_beta, beta_u, 3) once u_p_beta's post-solve ghost exchange has completed.
  void ReconstructShift();

  // Final assembly: vK_dd, alpha, beta_u -> pmy_pack->padm->u_adm (via
  // cfc::AssembleLapseShiftK). psi4/g_dd were already written by
  // SolveConformalFactor(), right after step 3.
  void AssembleADM();

  // 2026-07-25 (item 29, DEVELOPMENT.md): one pass of "refresh cons from the
  // fixed primitives + current metric, solve X^i, ghost-exchange, compute
  // Adual^ij/Ahat^2, solve psi" -- extracted from InitializeMetric()'s own loop
  // body (a pure refactor, no behavior change for the existing iterative path)
  // so it can be called either repeatedly (Picard iteration, the default) or
  // exactly once (<cfc> init_freeze_conserved=true, see
  // cfc_init_freeze_conserved_'s doc comment above). Always uses stage=0,
  // matching every other InitializeMetric()-internal call.
  void RunXPsiSolvePass(Driver *pdriver);

  // Item 11 (DEVELOPMENT.md): InitializeMetric() runs its X^i/psi fixed-point loop
  // entirely outside the normal per-stage task graph (it can't reuse it -- a single
  // pass through "stagen" would also flux-update/RK-evolve the hydro state, which
  // must NOT happen here), so it can't reuse CFC_InitRecv/ClearSend/ClearRecv
  // either: those post/wait on all 6 MeshBoundaryValuesCC instances at once, but
  // InitializeMetric's loop only ever sends/receives 2 of them (u_p_x, x_u)
  // per iteration -- calling the all-6 versions would post MPI_Irecv's for the
  // other 4 that never get a matching send that iteration, and ClearRecv's
  // MPI_Wait on those would hang. These two pairs scope InitRecv/ClearSend/
  // ClearRecv to exactly the fields actually exercised in each of InitializeMetric's
  // two phases (the iterated X^i/psi loop, and the one-shot lapse/shift/final tail).
  void InitRecvXFields();
  void ClearXFields();
  void InitRecvTailFields();
  void ClearTailFields();
};

}  // namespace cfc

#endif  // CFC_CFC_HPP_
