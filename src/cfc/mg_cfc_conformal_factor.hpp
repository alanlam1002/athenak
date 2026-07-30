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
//! where Ahat^2 = f_ik f_jl Adual^kl Adual^ij and Ũ = psi^6 U is the matter energy
//! density. Solved for the deviation delta_psi = psi - 1. Nonlinear in the unknown
//! (self-coupled through psi^-7), so SmoothPack/CalculateDefectPack/
//! CalculateFASRHSPack are hand-written Newton-Gauss-Seidel point relaxation (Gmunu
//! sec. 2.7.2, eq. 94) rather than Multigrid's generic constant-diagonal
//! Smooth<StencilOp> template. Ũ/Ahat^2 are carried in Multigrid::coeff_/ncoeff_
//! (ncoeff_=2) rather than src_, since src_ is what the V-cycle's FAS machinery
//! restricts/corrects at coarser levels. See mg_cfc_conformal_factor.cpp's file
//! header for the full sign-convention derivation.

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

  // Public accessor so MGCFCConformalFactorDriver::TransferCoeffToRoot() (a
  // MultigridDriver-derived caller, not a friend of the base Multigrid class) can
  // reach into an arbitrary level's coeff_ storage directly.
  DualArray5D<Real> &CoeffAtLevel(int l) { return coeff_[l]; }

 private:
  // Two algebraically-identical matter-source formulations, see ConformalFactorRHS's
  // doc comment in the .cpp. Compiled as separate template instantiations;
  // SmoothPack/CalculateDefectPack/CalculateFASRHSPack read
  // MGCFCConformalFactorDriver::use_psi5_source_ once and dispatch accordingly.
  template <bool UsePsi5> void SmoothPackImpl(int color);
  template <bool UsePsi5> void CalculateDefectPackImpl();
  template <bool UsePsi5> void CalculateFASRHSPackImpl();
};


//! \class MGCFCConformalFactorDriver
//! \brief Multigrid driver for delta_psi = psi - 1 (Gmunu eq. 73), isolated (1/r
//! falloff, mg_multipole) boundary conditions.

class MGCFCConformalFactorDriver : public MultigridDriver {
  public:
    MGCFCConformalFactorDriver(MeshBlockPack *pmbp, ParameterInput *pin);
    ~MGCFCConformalFactorDriver();

    void Solve(Driver *pdriver, int stage, Real dt = 0.0) final;

    // Load Ũ (matter energy density source, raw/unscaled -- the 2*pi factor is
    // applied inside the Newton kernel). Stored in coeff_ (channel 0), not via
    // Multigrid::LoadSource()/src_ -- see this file's docstring for why. ngh is
    // the depth u_tilde is padded to (the mesh's own NGHOST, not this driver's
    // shallower ngh_, since u_tilde is also differentiated/ghost-exchanged
    // elsewhere).
    void LoadMatterSource(const DvceArray5D<Real> &u_tilde, int ngh);

    // Load Ahat^2 = f_ik f_jl Adual^kl Adual^ij (from cfc::ComputeADualFromPotentials),
    // stored in coeff_ channel 1 (ncoeff_ = 2 total). Can't reuse Multigrid::
    // LoadCoefficients() (copies all ncoeff_ channels at once, no per-channel
    // offset) -- does its own single-channel par_for via CoeffAtLevel(), mirroring
    // LoadCoefficients' offset-aware ngh handling.
    void LoadNonlinearCoefficient(const DvceArray5D<Real> &a_sq, int ngh);

    // retrieve the converged delta_psi = psi - 1 solution after Solve() completes.
    void RetrieveSolution(DvceArray5D<Real> &dst);

    // Selects the matter-source formulation (see ConformalFactorRHS's doc comment
    // in the .cpp). Must be called before LoadMatterSource()/Solve() each time,
    // since coeff_ channel 0's physical meaning depends on which was requested.
    void SetUsePsi5Source(bool flag) { use_psi5_source_ = flag; }

    // Seed the finest level's own solution array (the V-cycle's initial guess)
    // from an externally-supplied delta_psi field, instead of a cold Kokkos-zero
    // start. Thin wrapper around Multigrid::LoadFinestData; ngh is the guess's own
    // padding depth, same convention as LoadMatterSource/LoadNonlinearCoefficient
    // above.
    void SeedInitialGuess(const DvceArray5D<Real> &guess, int ngh);

    void SmoothOctet(MGOctet &oct, int rlev, int color) final;
    void CalculateDefectOctet(MGOctet &oct, int rlev) final;
    void CalculateFASRHSOctet(MGOctet &oct, int rlev) final;

    // Template-dispatched Octet-level counterparts of SmoothPackImpl/
    // CalculateDefectPackImpl/CalculateFASRHSPackImpl above -- same UsePsi5
    // compile-time split.
    template <bool UsePsi5> void SmoothOctetImpl(MGOctet &oct, int rlev, int color);
    template <bool UsePsi5> void CalculateDefectOctetImpl(MGOctet &oct, int rlev);
    template <bool UsePsi5> void CalculateFASRHSOctetImpl(MGOctet &oct, int rlev);

    // Damps the FAS coarse-grid correction (u - uold) applied during prolongation,
    // separate from mg_omega_psi_'s per-point Newton damping below. Kept as a
    // generic, zero-risk-when-unused knob.
    Real CorrectionOmega() const override { return mg_correction_omega_; }

    friend class MGCFCConformalFactor;

  private:
    // Newton-relaxation controls (Gmunu sec. 2.7.2, eq. 94). mg_omega_psi_ damps
    // the per-point Newton step (1.0 = undamped); psi_floor_ prevents psi = u+1
    // from being driven non-positive (psi^-7 is ill-defined there) on a bad guess.
    Real mg_omega_psi_, psi_floor_;

    // Selects the matter-source formulation, see SetUsePsi5Source() above.
    // Constructed false; only CFC::InitializeMetric() sets this true (by default,
    // via <cfc> init_use_psi5_source), except for inputs too compact/unstable for
    // it (diverges to NaN there).
    bool use_psi5_source_ = false;

    // Damping factor for the coarse-grid correction (see CorrectionOmega() above).
    // Default 1.0 (undamped) via <cfc> mg_correction_omega.
    Real mg_correction_omega_;

    // mgroot_ never receives coeff_ via the generic TransferFromBlocksToRoot
    // (src_/u_ only) -- see the .cpp's doc comment for this function.
    void TransferCoeffToRoot();

    // Debug-only: gated on mg_debug_defect_by_level_ (<cfc> mg_debug_defect_by_level,
    // default false). Reports this rank's worst-converged (max |defect|) cell,
    // split by whether it belongs to a root-level or refined MeshBlock.
    bool mg_debug_defect_by_level_;
    void DebugReportDefectByLevel();

    // Shared worst-cell finder for DebugReportDefectByLevel/
    // DebugAnalyticResidualTest: recomputes def_ at the finest level, prints this
    // rank's worst |defect| cell split root/refined, and hands back the worst
    // REFINED cell's (m,k,j,i,gid) (left at -1 if this rank owns no refined
    // blocks).
    void DebugReportWorstDefect(const std::string &label, int &ref_m, int &ref_k,
                                int &ref_j, int &ref_i, int &ref_gid);

    // Debug-only companion to DebugReportWorstDefect: reports the worst
    // |u - analytic| (solution-value error, not residual), split root/refined.
    void DebugReportWorstSolutionError(const std::string &label);

    // Debug-only, gated on mg_debug_analytic_residual_test_ (<cfc>
    // mg_debug_analytic_residual_test, default false; PolytropeEOS-only). Seeds
    // delta_psi from the exact analytic isotropic TOV solution everywhere
    // (including ghosts), measures the discrete residual with no smoother, then
    // re-measures after one real ghost-comm round and one real V-cycle -- isolates
    // whether ghost-fill or Newton relaxation introduces error at a coarse-fine
    // interface.
    bool mg_debug_analytic_residual_test_;
    ParameterInput *pin_;
    void DebugAnalyticResidualTest(Driver *pdriver);

    // Debug-only: dumps the coarse_buf_ slot at a coarse-fine +x1 interface.
    // Called from DebugAnalyticResidualTest.
    void DebugDumpCoarseBuf();

    // Debug-only: reports the worst |defect| restricted to fine cells adjacent to
    // a coarse-fine +x1 interface, at a fixed physical location (unlike
    // DebugReportWorstDefect's domain-wide search). Called from
    // DebugAnalyticResidualTest.
    void DebugDumpInterfaceDefect(const std::string &label);

    // Debug-only: checks whether MultigridDriver::RestrictCoeffOctets() restricts
    // an octet-level-0's Coeff() down into mgroot_'s corresponding root-level
    // cell. Called from Solve(), gated on mg_debug_analytic_residual_test_.
    void DebugDumpRootCoeffUnderOctet();
};

#endif  // CFC_MG_CFC_CONFORMAL_FACTOR_HPP_
