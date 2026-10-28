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
#include <vector>

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

  // Analytic stationary BH "trumpet" background (CFC_PUNCTURE_TDE_PLAN.md Sec 3.2/3.6,
  // cfc_puncture.hpp). Time-independent (puncture fixed at x=0) -- populated once by
  // FillPunctureBackground(), at construction and again after every AMR regrid (mirrors
  // cfc_puncture arrays, same "analytic, cheap to just recompute" treatment
  // ReinitializeMetricForAMR already gives every other CFC-owned field); never touched
  // in the per-stage solve. Gated by <cfc> puncture_enabled_ (default false), but
  // u_psi0/u_alpha0_psi0/u_a0dual/a0_sq are always allocated (same AMR headroom as
  // every other CFC array) and filled with the exact flat-space values (psi0=1,
  // alpha0*psi0=1, Ahat0_ij=0, Ahat0^2=0) when disabled, since cfc_reconstruct.cpp/
  // cfc.cpp/mg_cfc_lapse.cpp/mg_cfc_conformal_factor.cpp now read them unconditionally
  // (Sec 5 Phase A items 3, 5, 6) -- every existing non-TDE run is a bit-for-bit no-op.
  // beta0_u/grad_ap6_0, like psi0/alpha0_psi0/a0_dd/a0_sq above, are always allocated
  // and read unconditionally every stage (AssembleLapseShiftK, BuildShiftSourceImpl --
  // Sec 5 Phase A item 7), filled with the flat-space defaults (beta0=0, grad_ap6_0=0)
  // when puncture_enabled_ is false.
  DvceArray5D<Real> u_psi0;          // psi0
  DvceArray5D<Real> u_alpha0_psi0;   // alpha0*psi0 (the product, matching how
                                     // delta_alpha_psi stores alpha*psi -- see Sec 3.3)
  DvceArray5D<Real> u_beta0;
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> beta0_u;   // beta0^i

  DvceArray5D<Real> u_a0dual;
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> a0_dd;     // Ahat0_ij, packed like a_dd

  DvceArray5D<Real> a0_sq;   // Ahat0_ij*Ahat0^ij

  // Shift-source ingredient grad_ap6_0^i = D_i(alpha0*psi0^-6) (Sec 3.9), the *bare*
  // closed-form gradient from TrumpetBackground's dpsi0/dalpha0 outputs (NOT contracted
  // with Ahat0^ij -- that contraction cancels out of the final shift residual, Sec 3.9).
  // Consumed by BuildShiftSourceImpl's "2*Ahats^ij*D_j(alpha0*psi0^-6)" term (Sec 5
  // Phase A item 7).
  DvceArray5D<Real> u_grad_ap6_0;
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> grad_ap6_0;

  // <cfc> puncture_enabled (default false): opt-in switch for the trumpet background
  // above. <cfc> puncture_mass (default 1.0): the BH mass M_BH the background is built
  // for -- see cfc.cpp's constructor for both.
  bool puncture_enabled_;
  Real puncture_mass_;
  // x_BH of the last FillPunctureBackground (u_psi0/u_alpha0_psi0/beta0_u are centred on
  // it): <dyn_grmhd> metric_terms evaluates the face background about the same point
  Real bg_xbh_[3] = {0.0, 0.0, 0.0};
  // <cfc> alpha_floor (default 0.0): lower clamp on adm.alpha (AssembleLapseShiftK)
  // so the fluid update never sees a negative lapse. See DEVELOPMENT.md item 59.
  Real alpha_floor_;

  // <cfc> residual_scale (R-140 item 6, D7): the hydro sees psi0 + f*delta_psi etc., the
  // solves are unchanged. f = residual_scale (constant), or a C^1 smoothstep from
  // residual_scale_f0 (default 1) at residual_scale_t0 to residual_scale_f1 at
  // residual_scale_t1. No key = off (f = 1, bitwise-neutral: the scaled kernels never
  // run while f == 1). f == 0 skips the four multigrid solves: background only.
  bool rs_on_ = false;
  bool rs_ramp_ = false;
  Real rs_const_ = 1.0, rs_f0_ = 1.0, rs_f1_ = 1.0, rs_t0_ = 0.0, rs_t1_ = 0.0;
  int rs_log_ = 0;           // <cfc> residual_scale_log_interval (cycles; 0 = no log)
  Real rs_f_ = 1.0;          // f of the state the current stage's CFC tasks see
  bool rs_bg_only_ = false;  // rs_f_ == 0: skip the MG solves
  Real ResidualScale(Real t) const;
  void SetResidualScale(Real t, int stage, bool last_stage);
  void AssemblePsiMetric(bool with_ghosts);

  // Running totals of what excision has removed from the grid over the whole run.
  // accreted_mass_ is ADM-weighted (int psi^5 E dV) and is what Step 3 hands to
  // puncture_mass_; accreted_mom_ is a DIAGNOSTIC only -- the puncture is fixed at
  // the origin and non-spinning by design, so the swallowed momentum is discarded,
  // and this records how much recoil that neglects.
  Real accreted_mass_ = 0.0;
  Real accreted_mom_[3] = {0.0, 0.0, 0.0};
  // Per-RK-stage excision removals, [stage][{dM_adm, dPx, dPy, dPz, dM_raw}].  Drained
  // once per stage and combined on the last stage with the RK propagation weights
  // w_s = prod_{j>s} gam0_j -- summing them raw over-counts by ~2x, because the SSP-RK
  // u1 register re-injects matter a previous stage already excised.  See
  // AccreteExcisedMass and item 60.  Sized for Driver::gam0[4], the max stage count.
  Real stage_tally_[4][5] = {};
  // <cfc> accrete_to_puncture (default true when puncture_enabled): actually grow
  // M_BH by the swallowed energy.  Set false to keep the tally as a pure diagnostic,
  // which is how the psi^5-vs-psi^6 weighting A/B is run.
  bool accrete_to_puncture_;
  Real puncture_mass_init_ = 0.0;  // for reporting M_BH growth

  // ADM-mass surface integral (<cfc> adm_mass_r0, adm_mass_r1, ... ; adm_mass_dr,
  // adm_mass_nlev).  The INDEPENDENT check that mass handed to the puncture actually
  // shows up in the spacetime: everything else (the excision tally, M_BH itself) is
  // internal bookkeeping that could be self-consistently wrong.
  std::vector<Real> adm_mass_radii_;
  Real adm_mass_dr_ = 1.0;
  int  adm_mass_nlev_ = 3;
  // <cfc> adm_mass_center = origin (default) | com.  With "com" the extraction spheres
  // are centered on r_com_mass_, the SAME centroid SolveConformalFactor() already hands
  // to MultigridDriver::SetRobinCenter() -- so the diagnostic and the solver agree on
  // where the residual's 1/r falloff is anchored.  Requires puncture_enabled_, since
  // r_com_mass_ is only ever filled on that path.  See ComputeADMMass() for what this
  // does and does not change.
  bool adm_mass_center_com_ = false;
  void ComputeADMMass();
  // Volume-integral counterparts of the ADM surface integral, split into the two
  // sources the Hamiltonian constraint actually has.  Exact identity being tested:
  //     M_res = int(psi^5 E) dV + (1/16pi) int[psi^-7 Ahat^2 - psi0^-7 Ahat0^2] dV
  // The first term is what AccreteExcisedMass hands to the puncture; the second is
  // the extrinsic-curvature energy, which changes when excision removes the matter
  // momentum that sources Ahat and which is currently NOT transferred.  Comparing
  // the sum against M_res both quantifies that omission and validates the surface
  // integral against an independent volume integral.
  void ComputeResidualMassVolume(Real *i_e, Real *i_a);
  void AccreteExcisedMass(Driver *pdriver, int stage);

  // Mass-weighted centroid of pmy_pack->pmhd->u0(IDN), recomputed every
  // SolveConformalFactor() call and reused by SolveLapse() the same stage (Sec 3.10
  // item 1/Sec 5 Phase A item 8a) -- psi and alpha*psi share this one centroid since
  // both have the same mass-like leading source. Feeds MultigridDriver::
  // SetRobinCenter() so the mg_robin outer BC falls off around the matter, not the
  // fixed coordinate origin the BH puncture sits at. Zero (its constructor default)
  // whenever puncture_enabled_ is false -- ComputeMassCentroid()/SetRobinCenter() are
  // simply never called then, a bit-for-bit no-op.
  Real r_com_mass_[3];

  // Physical-S-squared-weighted centroid of X^i's own source (Sec 3.10 item 2/Sec 5
  // Phase A item 8b), recomputed every SolveVectorPotential() call from cons(IM1+a)
  // undensitized by psi^12 (raw S-tilde_a = psi^6*S_a; squaring it directly would
  // spuriously bias toward the puncture -- must divide out psi^12 before squaring).
  // Feeds MGCFCVectorPoissonDriver::SetMultipoleOrigin() for pmgd_pietax AND the
  // AssembleVectorSource(!for_shift) eta-source xk offset AND
  // ComputeADualFromPotentials's reconstruction cross-term offset -- all three MUST
  // use this same value (Sec 3.10 item 3: shifting only one silently reconstructs
  // the wrong vector field). Deliberately NOT forced equal to r_com_beta_/r_com_mass_
  // -- logged each call as a cheap sanity check only. Zero (constructor default)
  // whenever puncture_enabled_ is false -- ComputeVectorCentroidX()/
  // SetMultipoleOrigin() are simply never called then, a bit-for-bit no-op.
  Real r_com_X_[3];

  // Same idea as r_com_X_, for beta^i's own source -- but weighted by p_src_ as
  // assembled (16*pi*alpha*psi^-6*S-tilde_a + BuildShiftSource's Ahat-gradient term),
  // squared directly with NO psi^12 correction: the matter term's alpha*psi^-6
  // already exactly cancels S-tilde's psi^6 (Sec 3.10 item 2). Computed inside
  // AssembleVectorSource(/*for_shift=*/true), immediately after BuildShiftSource()
  // finishes (not before -- needs the fully-assembled source). Feeds
  // MGCFCVectorPoissonDriver::SetMultipoleOrigin() for pmgd_pietabeta AND that same
  // branch's eta-source xk offset AND ReconstructVectorFromPotentials's cross-term
  // offset. Independent from r_com_X_ by design (item 2: NOT a shared "momentum"
  // proxy) -- do not conflate. Zero whenever puncture_enabled_ is false, same
  // reasoning as r_com_X_.
  Real r_com_beta_[3];

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
  DvceArray5D<Real> u_alpha_psi6;  // BuildShiftSource: Delta(alpha*psi^-6),
                                   // cancellation-free (Sec 3.7/3.9)

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
  // restart_state: the seed flags as the restarted run left them (AccreteExcisedMass
  // clears them at the end of a step that accreted, so the next step's first solves
  // re-seed from adm); read in the constructor, applied by LoadRestartState
  bool rst_psi_seeded_ = true, rst_alpha_psi_seeded_ = true;

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

  // <cfc> solve_interval (default 0 = every substage every cycle). >=1 solves
  // only on the last substage of every Nth cycle, reusing the previous solve
  // otherwise (padm->adm/psi/alpha_psi/beta are read-only downstream). Ported
  // from ~/athenak_cfc; see DEVELOPMENT.md item 6.
  int cfc_solve_interval_;

  // Comoving gauge (DEVELOPMENT.md item 68): x' = x - xi(t) with the slicing
  // unchanged, i.e. a spatially uniform xidot^i(t) added to the shift that the fluid
  // sees (adm.beta_u), never to the solved beta_u. <cfc> frame_vel1/2/3 and
  // frame_accel1/2/3 (default 0): xidot(t) = frame_vel + frame_accel*t, so
  // xi(t) = frame_vel*t + frame_accel*t^2/2 (xi(0) = 0). The value currently inside
  // adm.beta_u is padm->frame_vel_u.
  Real frame_vel0_[3], frame_accel_[3];
  bool puncture_moving_ = false;    // <cfc> puncture_moving (Phase 2 M1)
  Real xbh0_[3] = {0.0, 0.0, 0.0};  // <cfc> puncture_x1/2/3: x_BH at t = 0
  bool frame_enabled_;
  bool frame_assembled_;   // padm->frame_vel_u reflects adm.beta_u (false until the
                           // first assembly; on a restart the rst's adm.beta_u is
                           // taken to hold xidot(time), see ApplyFrameShiftUpdate)

  // Phase 2 M2 (research R-039/R-040 item 4; COMOVING_GAUGE_DESIGN.md §2.3):
  // <cfc> frame_controller = true replaces the analytic xidot(t) with a per-step
  // controller. xidot is held fixed over each step and xi is integrated,
  // xi(t) = fc_xi_ + fc_xd_*(t - fc_t_), so the moving puncture and every other
  // FrameDisp/FrameVel user see a continuous xi. Once per step (last RK stage) the
  // tracked gas -- rest mass with rho > frame_track_rho within frame_track_radius of the
  // target point x'_tgt on the grid -- gives its grid velocity V' = alpha u^i/W - beta^i
  // and centroid x'_c, and
  //   xidot_new = V_BH + a*dt + (x'_c - x'_tgt)/frame_tau_x,  V_BH = V' + xidot_old,
  // where a is V_BH's finite-difference rate low-passed over frame_tau_a (the
  // feed-forward). Tracking within a radius of the target keeps the controller on the
  // clump pinned there (the remnant after periapsis), not on the debris centroid.
  // xidot(0) = V_BH of the tracked gas after InitializeMetric (= alpha v - beta). State
  // is written to the restart's parameter copy (SaveFrameState).
  bool frame_ctrl_ = false;
  Real fc_t_ = 0.0;                       // time at which fc_xi_ holds
  Real fc_xi_[3] = {0.0, 0.0, 0.0};
  Real fc_xd_[3] = {0.0, 0.0, 0.0};       // xidot over the current step
  Real fc_tgt_[3] = {0.0, 0.0, 0.0};      // x'_tgt
  bool fc_tgt_given_ = false;             // <cfc> frame_target_x* set
  Real fc_vprev_[3] = {0.0, 0.0, 0.0};    // previous V_BH (for the feed-forward)
  Real fc_acc_[3] = {0.0, 0.0, 0.0};      // low-passed dV_BH/dt
  bool fc_init_ = false;                  // controller state valid (set or restored)
  Real fc_rho_thr_, fc_rtrk_, fc_tau_x_, fc_tau_a_;
  Real fc_atm_rho_ = 0.0, fc_atm_gate_ = 0.0;  // initial comoving atmosphere
  Real fc_meas_[5] = {0.0, 0.0, 0.0, 0.0, 0.0};  // last V'(3), tracked mass, |x'_c-tgt|
  // tracked-gas reduction: out = {m, m x', m y', m z', m V'x, m V'y, m V'z}
  void MeasureTracked(Real out[7]);
  // Research R-050 B2 (DEBRIS_REFINEMENT_DESIGN §4): what the controller follows.
  // <cfc> frame_target_mode = remnant (default: the tracked clump, as above) |
  // bound_centroid (mass-weighted centroid and grid velocity of gas with BH-frame
  // E = -u_t - 1 < 0, rho > frame_bound_rho, |x' - x_BH| >= frame_bound_rmin (default 6 M,
  // R-054: both gates must cover the near-BH atmosphere zone's rho_thr and r_s); the
  // commanded xidot is low-passed over frame_tau_a) | ramp (xidot -> 0 linearly over
  // frame_ramp_time from its value at the switch). Automatic switch: in remnant mode,
  // when the tracked mass falls below frame_switch_mass_frac * m_ref (m_ref = the t = 0
  // tracked mass, or <cfc> frame_switch_mref), switch to frame_switch_mode. Default
  // frame_switch_mass_frac = 0 (off), so existing decks are unchanged. Mode, switch time,
  // switch xidot and m_ref are stored in restarts.
  int fc_mode_ = 0;                       // 0 remnant, 1 bound_centroid, 2 ramp
  int fc_switch_mode_ = 1;
  Real fc_switch_frac_ = 0.0;
  Real fc_m0_ = 0.0;                      // m_ref (0: unknown -> no automatic switch)
  Real fc_tsw_ = 0.0;                     // switch time
  Real fc_xdsw_[3] = {0.0, 0.0, 0.0};     // xidot at the switch (ramp start)
  Real fc_ramp_T_ = 50.0;
  Real fc_bound_rho_ = 0.0, fc_bound_rmin_ = 6.0;
  // bound-gas reduction, same layout as MeasureTracked
  void MeasureBound(Real out[7]);
  // R-112 item 4 (the comoving-gauge exit; all default off, so existing decks are
  // unchanged). <cfc> frame_ramp_shape = linear (default) | smoothstep (C^1: xidot =
  // xidot_sw (1 - 3s^2 + 2s^3), s = (t - t_sw)/frame_ramp_time; no jump in xi'' at the
  // ends). Two more switches to ramp mode, checked in remnant and bound_centroid modes
  // (the first to fire wins, with the tracked-mass switch above):
  //   debris approach, <cfc> frame_switch_approach_radius R_trig > 0: the bound gas (the
  //     MeasureBound gates) has min over cells of (r - max(v_in, 0) T_ramp) <= R_trig,
  //     i.e. d_min - v_in T_ramp at each cell (v_in the inward speed relative to the BH),
  //     so the ramp completes before returning gas nears the hole;
  //   accretion onset, <cfc> frame_switch_accretion_rate > 0 (per unit time, in units of
  //     m_ref): the net inflow of gas with rho > frame_bound_rho through the shell
  //     frame_switch_accretion_radius (R_acc, default 6 M) <= r < R_acc +
  //     frame_switch_accretion_shell (default 2 M), sum(D dV v_in)/shell, exceeds it.
  int fc_ramp_shape_ = 0;                 // 0 linear, 1 smoothstep
  Real fc_app_r_ = 0.0, fc_acc_rate_ = 0.0, fc_acc_r_ = 6.0, fc_acc_w_ = 2.0;
  Real fc_exit_[2] = {0.0, 0.0};          // last MeasureExit: approach measure, inflow
  // out = {min (r - max(v_in, 0) T_ramp) over the bound gas, the shell inflow rate}
  void MeasureExit(Real out[2]);
  void FrameSwitch(Real t_new, int mode, const char *why);
  void FrameControlStep(Real t_new);

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
  // R1 embedding at load (restart_state::EmbedFill): set padm->adm in the blocks with
  // newmb(m) = 1 (ghosts included) to the zero-residual metric, i.e. what AssembleADM gives
  // there with delta_psi = delta_alpha_psi = 0, b = 0 and Ahat = Ahat0: psi = psi0,
  // alpha = (alpha psi)0/psi0 (alpha_floor), beta = beta0 + xidot, K_ij = psi0^-2 Ahat0_ij.
  // Provisional: ReinitializeMetricForAMR re-solves it.
  void FillEmbedADM(const DvceArray1D<int> &newmb, Real time);

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
  TaskStatus AssemblePsiGhostsTask(Driver *pdriver, int stage);   // R-072 M1
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
  // <cfc> src_nan_check: when >0, scan the elliptic solves' SOURCES and OUTPUTS for
  // non-finite values after every solve and report the first offender's location.
  // Purpose (NANCASCADE_HANDOFF.md §13.5/§14): the cascade begins with the VECTOR
  // Poisson solve returning an all-NaN beta^i while psi/alpha/K and every primitive
  // are still clean, and while the multigrid reports convergence. This distinguishes
  // "the source handed to the solver was already bad" from "the solver produced NaN
  // from a finite source" -- which no existing diagnostic separates. 0 = off (default,
  // zero cost: the checks are behind this branch).
  int src_nan_check_;

  // Mesh ghost depth, cached for CheckFinite's interior restriction.
  int indcs_ng_;

  // <cfc> psi_floor: lower clamp on the conformal factor psi, applied to the psi
  // solve's OUTPUT (see ApplyPsiFloor). <=0 (default) disables it, leaving every
  // existing fixture bitwise unchanged. See cfc.cpp for why the clamp belongs at the
  // solve rather than at the ADM write.
  Real psi_floor_;

  // Clamp psi >= psi_floor_ / alpha >= alpha_floor_ on the SOLVE OUTPUT, before that
  // field's ghost exchange, so no non-positive value reaches a downstream solver.
  // Both operate on the RESIDUAL arrays (psi = delta_psi + psi0), so the clamp is
  // applied to the sum and rearranged onto the residual.
  void ApplyPsiFloor();
  void ApplyAlphaFloor();

  // Scan `a`'s first `nchan` channels over interior+ghost cells for NaN/Inf. Returns
  // the global count; rank 0 prints a one-line report naming the first offending
  // cell (globally lowest flat index, MPI_MINLOC) with its coordinates and value.
  // Collective: every rank must call it.
  // `ngh` is the array's own ghost depth: cells shallower than that are skipped, so
  // never-written ghost zones (u_p_src's, for one -- it is pure solver input, never
  // ghost-exchanged) cannot register as false positives.
  int CheckFinite(const char *label, const DvceArray5D<Real> &a, int nchan,
                  int stage, int ngh) const;

  // cfc_solve_interval_ gate: every CFC task calls this first and early-returns
  // TaskStatus::complete when false (a skipped task satisfies the task graph the
  // same as a completed one). stage<1 (direct calls from InitializeMetric()/
  // ReinitializeMetricForAMR) always returns true -- see cfc.cpp, this avoids a
  // real deadlock.
  bool DoSolveThisStage(Driver *pdriver, int stage) const;

  // Builds Shibata eq. 3.10-3.11's packed source (P_i's RHS S_i at channels 0-2,
  // eta's RHS -S_i.x^i at channel 3) into u_p_src/p_src -- for_shift selects X^i's
  // source (eq. 72, from pmhd->u0) or beta^i's (eq. 75, from alpha/psi/Adual^ij/
  // S-tilde_i). When puncture_enabled_, the for_shift=true branch also calls
  // ComputeVectorCentroidBeta() right after BuildShiftSource() finishes assembling
  // p_src_ (Sec 3.10 item 2/Sec 5 Phase A item 8b) -- this is the one place a
  // puncture-gated side effect lives inside an otherwise puncture-agnostic function.
  void AssembleVectorSource(bool for_shift);

  // Fills r_com_mass_ from a mass-weighted reduction over pmhd->u0(IDN) (Sec 3.10
  // item 1) -- mirrors MultigridDriver::CalculateCenterOfMass()'s reduction shape,
  // but reads u0 directly instead of src_ (psi/alpha_psi's matter lives in coeff_,
  // Finding B, so their src_ is always zero -- CalculateCenterOfMass() would divide
  // by zero if reused as-is). Called from SolveConformalFactor() only; SolveLapse()
  // reuses that same call's result (u0 is unchanged in between -- RescaleMatterSources
  // only rewrites s_tilde_).
  void ComputeMassCentroid();

  // Fills r_com_X_ from a physical-S-squared-weighted reduction over
  // pmy_pack->pmhd->u0(IM1..IM3), undensitized by psi^12 (Sec 3.10 item 2). Mirrors
  // ComputeMassCentroid()'s reduction shape but with a different weight, and reads
  // delta_psi_/u_psi0_ for the psi^12 correction -- NOT this stage's freshly-solved
  // psi (X^i solves before psi does, steps 1 vs 3), just whatever psi this solver
  // currently holds (last stage's converged value, or the analytic/zero seed before
  // the very first-ever solve -- always well-defined, no bootstrap NaN risk). Called
  // from SolveVectorPotential() only, before AssembleVectorSource(/*for_shift=*/
  // false) and before pmgd_pietax->Solve().
  void ComputeVectorCentroidX();

  // Fills r_com_beta_ from a p_src_-squared-weighted reduction (Sec 3.10 item 2), no
  // psi^12 correction needed (see r_com_beta_'s own doc comment). Called from
  // AssembleVectorSource(/*for_shift=*/true) only, immediately after
  // BuildShiftSource() returns and before that branch's eta-source loop.
  void ComputeVectorCentroidBeta();

  // Shared reflecting-mesh-boundary override, factored out of
  // ComputeMassCentroid()'s original inline logic -- identical correction applies to
  // any centroid computed from a reduction over the (possibly octant-symmetric)
  // stored domain: force each axis to its own reflecting plane's coordinate rather
  // than trusting the raw (spuriously off-center) integral. Called by
  // ComputeMassCentroid(), ComputeVectorCentroidX(), and ComputeVectorCentroidBeta().
  void ApplyReflectingBoundaryCorrection(Real center[3]);

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
  void AssembleADM(Real time);

  // Time of the state that this stage's CFC tasks see (the state just updated by
  // the MHD tasks): pmesh->time + tau_stage*dt for the SSP-RK integrators; stage < 1
  // (InitializeMetric/ReinitializeMetricForAMR) is pmesh->time.
  Real StageTime(Driver *pdriver, int stage) const;
  // Non-solve stages (solve_interval > 1): shift adm.beta_u (all cells, ghosts
  // included) by xidot(time) - padm->frame_vel_u, so the gauge part is current every
  // stage even while the solved part is reused.
  void ApplyFrameShiftUpdate(Real time);

 public:
  bool FrameShiftEnabled() const { return frame_enabled_; }
  void FrameVel(Real t, Real xd[3]) const {
    if (frame_ctrl_) { for (int a = 0; a < 3; ++a) xd[a] = fc_xd_[a]; return; }
    for (int a = 0; a < 3; ++a) xd[a] = frame_vel0_[a] + frame_accel_[a]*t;
  }
  void FrameDisp(Real t, Real x[3]) const {
    if (frame_ctrl_) {
      for (int a = 0; a < 3; ++a) x[a] = fc_xi_[a] + fc_xd_[a]*(t - fc_t_);
      return;
    }
    for (int a = 0; a < 3; ++a) x[a] = frame_vel0_[a]*t + 0.5*frame_accel_[a]*t*t;
  }
  bool FrameController() const { return frame_ctrl_; }
  // last controller measurement: V'(3) of the tracked gas, its mass, |x'_c - x'_tgt|
  const Real *FrameMeasurement() const { return fc_meas_; }
  // write xi, xidot and the controller state into pin (restart files carry pin)
  void SaveFrameState(ParameterInput *pin) const;
  // restart_state (outputs/restart_state.hpp): delta_psi, delta_alpha_psi and the four
  // MG solvers' finest-level solutions (warm starts); loading marks the one-shot seeding
  // (SolveConformalFactor/SolveLapse) done
  int RestartStateVars() const;
  std::string RestartStateLayout() const;   // named groups (restart_state.hpp)
  void SaveRestartState(DvceArray5D<Real> &dst, int ns);
  void LoadRestartState(const DvceArray5D<Real> &src, int ns);
  void SaveRestartScalars(ParameterInput *pin) const;   // the seed flags
  // after InitializeMetric (fresh start): xidot(0) from the tracked gas, target, and
  // the comoving initial atmosphere
  void InitFrameControl(Driver *pdriver);
  // Phase 2 M1 (design §2.1): puncture position on the grid at time t,
  // x_BH(t) = x_BH(0) - xi(t) when <cfc> puncture_moving, else x_BH(0) (origin default).
  bool PunctureMoving() const { return puncture_moving_; }
  void BHPosition(Real t, Real x[3]) const {
    Real d[3] = {0.0, 0.0, 0.0};
    if (puncture_moving_) { FrameDisp(t, d); }
    for (int a = 0; a < 3; ++a) x[a] = xbh0_[a] - d[a];
  }
  // refill the analytic background at x_BH(t) and point the ADM outer-BC falloff there
  void RefillMovingPuncture(Real t);

 private:

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
