#ifndef CFC_MG_CFC_CONFORMAL_FACTOR_HPP_
#define CFC_MG_CFC_CONFORMAL_FACTOR_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mg_cfc_conformal_factor.hpp
//! \brief defines MGCFCConformalFactor[Driver], solving Gmunu (2021) eq. 73 for the
//! conformal factor psi:
//!   Delta psi = -2 pi Ũ psi^-1 - (1/8) Ahat^2 psi^-7,
//! where Ahat^2 = f_ik f_jl Adual^kl Adual^ij is precomputed once per outer iteration
//! from Adual^ij (does not depend on psi) and Ũ = psi^6 U is the (once-per-outer-
//! -iteration-rescaled) matter energy density.
//!
//! Unlike gravity's Poisson equation or the CFC vector-potential equations, this
//! operator is nonlinear in the unknown (self-coupled through psi^-7): the generic
//! constant-diagonal Smooth<StencilOp> template in Multigrid assumes a linear operator
//! and cannot be reused here. SmoothPack/CalculateDefectPack/CalculateFASRHSPack are
//! therefore overridden with hand-written Newton-Gauss-Seidel point relaxation
//! (Gmunu sec. 2.7.2, eq. 94): u_new = u_old - L(u_old, f)/(dL/du)|_{u=u_old}.
//!
//! As in the reference implementations, the solve is done for the deviation
//! delta_psi = psi - 1 (so that the isolated-system boundary condition maps onto the
//! existing 1/r multipole falloff BC), not psi directly.
//!
//! Reuses the (otherwise-unused-by-gravity) Multigrid::coeff_/ncoeff_ scaffolding to
//! carry both Ũ and the precomputed Ahat^2 field alongside the solution (ncoeff_=2):
//! neither can live in the generic src_/LoadSource() path, because src_ is exactly
//! what the V-cycle's FAS machinery restricts *and* corrects at coarser levels, which
//! would corrupt these fields' physical values where RHS(u) needs them pristine (see
//! mg_cfc_conformal_factor.cpp's file-level comment for the full derivation).

// C++ headers
#include <string>

// Athenak headers
#include "../athena.hpp"
#include "../multigrid/multigrid.hpp"

class MeshBlockPack;
class ParameterInput;
class Multigrid;
class MultigridDriver;

//! \class MGCFCConformalFactor
//! \brief Multigrid object for delta_psi = psi - 1

class MGCFCConformalFactor : public Multigrid {
 public:
  MGCFCConformalFactor(MultigridDriver *pmd, MeshBlockPack *pmbp, int nghost,
                       bool on_host = false);
  ~MGCFCConformalFactor();

  void SmoothPack(int color) final;
  void CalculateDefectPack() final;
  void CalculateFASRHSPack() final;

  // Public accessor so MGCFCConformalFactorDriver::TransferCoeffToRoot() (see .cpp)
  // can reach into an arbitrary level's coeff_ storage directly. Needed because
  // coeff_/ncoeff_ are protected members declared on the *base* Multigrid class:
  // a friend of this (derived) class is not automatically a friend of Multigrid,
  // so MultigridDriver-derived callers can't reach coeff_ via friendship the way
  // MGCFCConformalFactor itself can. Kept as a plain public one-liner (same shape
  // as Multigrid's own GetCurrentData_h()-style accessors) rather than adding a
  // new friend declaration or touching src/multigrid/ itself.
  DualArray5D<Real> &CoeffAtLevel(int l) { return coeff_[l]; }
};


//! \class MGCFCConformalFactorDriver
//! \brief Multigrid driver for delta_psi = psi - 1 (Gmunu eq. 73), isolated (1/r
//! falloff, mg_multipole) boundary conditions.

class MGCFCConformalFactorDriver : public MultigridDriver {
  public:
    MGCFCConformalFactorDriver(MeshBlockPack *pmbp, ParameterInput *pin);
    ~MGCFCConformalFactorDriver();

    void Solve(Driver *pdriver, int stage, Real dt = 0.0) final;

    // load Ũ (matter energy density source, raw/unscaled -- the 2*pi factor is
    // applied inside the Newton kernel itself, mg_cfc_conformal_factor.cpp). Stored
    // in coeff_ (channel 0), NOT via Multigrid::LoadSource()/src_ -- see this file's
    // top docstring and the .cpp's file-level comment for why. ngh is the depth
    // u_tilde itself is padded to (mirrors Multigrid::LoadSource/LoadCoefficients'
    // own ngh parameter, multigrid.cpp:289/327) -- callers pass the mesh's own
    // NGHOST here, NOT this driver's (generally shallower) ngh_, since u_tilde is
    // also differentiated/ghost-exchanged elsewhere and must be sized accordingly
    // (plan addendum #4, Finding H).
    void LoadMatterSource(const DvceArray5D<Real> &u_tilde, int ngh);

    // load Ahat^2 = f_ik f_jl Adual^kl Adual^ij (from cfc::ComputeADualFromX), stored
    // in coeff_ channel 1 (ncoeff_ = 2 total). Multigrid::LoadCoefficients() can't be
    // reused for either load (it copies all ncoeff_ channels in one shot, no
    // per-channel offset) -- both loaders do their own single-channel par_for via
    // the CoeffAtLevel() accessor, but mirror LoadCoefficients' offset-aware ngh
    // handling (Finding H, same reasoning as LoadMatterSource above).
    void LoadNonlinearCoefficient(const DvceArray5D<Real> &a_sq, int ngh);

    // retrieve the converged delta_psi = psi - 1 solution after Solve() completes.
    void RetrieveSolution(DvceArray5D<Real> &dst);

    // seed the finest level's own solution array (the V-cycle's initial guess) from
    // an externally-supplied delta_psi field, instead of leaving it at whatever it
    // already held (a cold Kokkos-zero-initialized guess, on the very first call
    // this driver's Solve() ever makes). Thin wrapper around Multigrid::
    // LoadFinestData (mirrors gravity::MGGravityDriver::Solve()'s own
    // mglevels_->LoadFinestData call) -- ngh is guess's own padding depth, same
    // convention as LoadMatterSource/LoadNonlinearCoefficient above.
    void SeedInitialGuess(const DvceArray5D<Real> &guess, int ngh);

    void SmoothOctet(MGOctet &oct, int rlev, int color) final;
    void CalculateDefectOctet(MGOctet &oct, int rlev) final;
    void CalculateFASRHSOctet(MGOctet &oct, int rlev) final;

    friend class MGCFCConformalFactor;

  private:
    // Newton-relaxation controls (Gmunu sec. 2.7.2, eq. 94). mg_omega_psi_ damps
    // the per-point Newton step (1.0 = undamped); psi_floor_ prevents psi = u+1
    // from being driven non-positive (psi^-7 is ill-defined there) on a bad guess.
    Real mg_omega_psi_, psi_floor_;

    // Relative-change convergence check (DEVELOPMENT.md item 20) -- tried, found
    // no measurable improvement over the base class's defect-norm SolveMG(), and
    // reverted (see Solve()'s doc comment in the .cpp for the still-present,
    // commented-out implementation). u_prev_ is unused while that's reverted.
    // DvceArray5D<Real> u_prev_;

    // MultigridDriver::TransferFromBlocksToRoot (multigrid_driver.cpp) aggregates
    // every rank's coarsest per-block cell into the distributed root grid (mgroot_)
    // via MPI_Allgatherv, but only for src_/u_ -- never coeff_. That transfer runs
    // for any multi-meshblock mesh (not just AMR), so mgroot_ needs its own coeff_
    // (Ũ, Ahat^2) populated the same way before the V-cycle can reach the root level.
    // Deliberately NOT a change to src/multigrid/: duplicates the relevant slice of
    // TransferFromBlocksToRoot's logic locally, restricted to the non-refined
    // (octet-free) case, since AMR+CFC is guarded against in Solve() (see .cpp).
    void TransferCoeffToRoot();

    // Temporary diagnostic (2026-07-20, plan addendum): reports this rank's
    // worst-converged (max |defect|) cell, split by whether it belongs to a
    // root-level or a refined MeshBlock, to help root-cause the AMR
    // refinement-boundary residual-floor issue (DEVELOPMENT.md item 12). Gated
    // on mg_debug_defect_by_level_ (default false, <cfc> mg_debug_defect_by_level
    // input) -- meant to be deleted once the root-cause question is answered, not
    // kept as a permanent feature (see DEVELOPMENT.md's new item for this).
    bool mg_debug_defect_by_level_;
    void DebugReportDefectByLevel();

    // Shared worst-cell-finder used by both DebugReportDefectByLevel and
    // DebugAnalyticResidualTest below -- recomputes def_ at the finest level,
    // prints this rank's worst |defect| cell split by root/refined (label
    // prepended to both printed lines), and hands back the worst REFINED cell's
    // (m,k,j,i,gid) so callers that need it (e.g. the stencil dump) don't have to
    // redo the search. ref_gid is left at -1 if this rank owns no refined blocks.
    void DebugReportWorstDefect(const std::string &label, int &ref_m, int &ref_k,
                                int &ref_j, int &ref_i, int &ref_gid);

    // Temporary diagnostic (2026-07-21): companion to DebugReportWorstDefect --
    // reports the worst |u - analytic| (actual solution-value error, not the
    // equation's residual), split root/refined, same classification pattern.
    // Used by DebugAnalyticResidualTest to show how far a single V-cycle moves
    // the solution from the exact analytic answer.
    void DebugReportWorstSolutionError(const std::string &label);

    // Temporary diagnostic (2026-07-21): seeds delta_psi at every cell -- including
    // every ghost cell, at the refinement boundary too -- from the exact analytic
    // isotropic TOV solution (same tov::TOVStar/PolytropeEOS machinery dyngr_tov.cpp
    // itself uses), measures the discrete residual with no smoother involved at all,
    // then replaces just the ghost cells with one real ghost-communication round
    // (the same FillCoarseBoundary/.../ProlongateFCBoundary sequence the V-cycle's
    // finest level uses) and measures again. Distinguishes "the ghost-fill machinery
    // itself introduces a large error at the coarse-fine interface" from "the Newton
    // relaxation is the problem" -- see DEVELOPMENT.md's entry for this addendum.
    // 2026-07-21 (user request): also runs exactly one real V-cycle afterward
    // (needs pdriver, forwarded from Solve()) and reports both the defect and the
    // actual solution-value error against analytic truth once more, to show how
    // much a single relaxation sweep moves the solution.
    // Gated on mg_debug_analytic_residual_test_ (default false); PolytropeEOS-only;
    // to be deleted alongside the rest of this investigation's diagnostics.
    bool mg_debug_analytic_residual_test_;
    ParameterInput *pin_;
    void DebugAnalyticResidualTest(Driver *pdriver);

    // Temporary diagnostic (2026-07-21): confirmation instrumentation for the
    // "stale coarse_buf_ slot at the high-child transverse edge" hypothesis --
    // see the .cpp doc comment for the full index trace. Called from
    // DebugAnalyticResidualTest, not separately gated.
    void DebugDumpCoarseBuf();

    // Temporary diagnostic (2026-07-21): reports the worst |defect| restricted to
    // just the layer of fine cells immediately adjacent to a coarse-fine +x1
    // interface (same block-selection logic as DebugDumpCoarseBuf), at a FIXED
    // physical location independent of wherever the domain-wide worst cell
    // happens to be. Unlike DebugReportWorstDefect, this isn't contaminated by
    // the star's own density-profile features moving the global worst cell to a
    // different location at different resolutions -- see the .cpp doc comment.
    // Called twice from DebugAnalyticResidualTest (before and after the real
    // ghost-comm round), not separately gated.
    void DebugDumpInterfaceDefect(const std::string &label);

    // Temporary diagnostic (2026-07-21): confirmation instrumentation for the
    // hypothesis that MultigridDriver::RestrictCoeffOctets() never restricts an
    // octet-level-0's Coeff() down into mgroot_'s own corresponding root-level cell
    // (unlike the generic, per-V-cycle RestrictOctets(), which has an explicit
    // "octets to root grid" branch for u_/src_). Called from Solve(), gated on
    // mg_debug_analytic_residual_test_ (same flag as the rest of this investigation's
    // diagnostics -- see the .cpp doc comment for the full trace).
    void DebugDumpRootCoeffUnderOctet();
};

#endif  // CFC_MG_CFC_CONFORMAL_FACTOR_HPP_
