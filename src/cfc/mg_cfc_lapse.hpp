#ifndef CFC_MG_CFC_LAPSE_HPP_
#define CFC_MG_CFC_LAPSE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mg_cfc_lapse.hpp
//! \brief defines MGCFCLapse[Driver], solving Gmunu (2021) eq. 74 for the lapse times
//! conformal factor, alpha*psi:
//!   Delta (alpha psi) = (alpha psi) [ 2 pi (Ũ + 2 S̃) psi^-2
//!                                     + (7/8) Ahat^2 psi^-8 ],
//! where psi (already solved, see mg_cfc_conformal_factor.hpp) and
//! Ahat^2 = f_ik f_jl Adual^kl Adual^ij (already computed from Adual^ij) are known,
//! fixed fields for this solve, and Ũ, S̃ are the psi^6-rescaled matter source terms.
//!
//! Unlike eq. 73 (psi), this operator is NOT nonlinear: psi/Ahat^2 are already
//! converged by the time this solve runs, so K(x) := 2 pi (Ũ + 2 S̃) psi^-2 +
//! (7/8) Ahat^2 psi^-8 depends only on known fields, not on the unknown alpha*psi.
//! The equation is affine in delta_(alpha psi) = alpha*psi - 1: Delta(u+1) -
//! K(x)*(u+1) = 0. Still can't reuse the generic Smooth<StencilOp> template (its
//! diagonal is constant; here 6 + dx^2*K(x) varies per point), so SmoothPack/
//! CalculateDefectPack/CalculateFASRHSPack are hand-written -- but since F(u) is
//! affine, the per-point "Newton" step is an *exact* one-step Gauss-Seidel solve,
//! needing no damping or positivity floor the way eq. 73's does.
//!
//! CFC_PUNCTURE_TDE_PLAN.md Sec 3.8/Sec 5 Phase A item 4's lapse-side follow-up
//! (closed here): K(x)/S are no longer fused into two precomputed scalars at the
//! finest level only -- matter-derived ingredients (Ũ+2S̃, delta_psi, DeltaAhat^2)
//! and analytic-background ingredients (psi0, Ahat0^2, alpha0*psi0) are kept as
//! six separate coeff_ channels (ncoeff_=6, see the .cpp's channel-layout comment),
//! mirroring MGCFCConformalFactor's own psi0/Ahat0^2 split exactly. Matter channels
//! restrict normally; background channels get overwritten per level by
//! FillPunctureCoefficients() (same mechanism as the psi driver's, see that file's
//! doc comment) instead of being smoothed by plain 8-cell averaging. The actual
//! K(x)/S combine happens fresh, per point, per level, inside SmoothPack/
//! CalculateDefectPack/CalculateFASRHSPack via the shared LapseReactionRHS()
//! helper (mg_cfc_lapse.cpp) -- exactly the role ConformalFactorRHS() plays for
//! the psi driver's three analogous kernels.

// Athenak headers
#include "../athena.hpp"
#include "../multigrid/multigrid.hpp"

class MeshBlockPack;
class ParameterInput;
class Multigrid;
class MultigridDriver;

//! \class MGCFCLapse
//! \brief Multigrid object for delta_(alpha psi) = alpha*psi - 1

class MGCFCLapse : public Multigrid {
 public:
  MGCFCLapse(MultigridDriver *pmd, MeshBlockPack *pmbp, int nghost,
            bool on_host = false);
  ~MGCFCLapse();

  void SmoothPack(int color) final;
  void CalculateDefectPack() final;
  void CalculateFASRHSPack() final;

  // See MGCFCConformalFactor::CoeffAtLevel's docstring: a public one-liner so
  // MGCFCLapseDriver::TransferCoeffToRoot() can reach coeff_ without needing
  // friendship of the base Multigrid class.
  DualArray5D<Real> &CoeffAtLevel(int l) { return coeff_[l]; }

  // CFC_PUNCTURE_TDE_PLAN.md Sec 3.8/Sec 5 Phase A item 4 (lapse-side follow-up):
  // overwrite coeff_ channels 3 (psi0)/4 (Ahat0^2)/5 (alpha0*psi0) at EVERY
  // internal level of this object with a fresh analytic evaluation of the trumpet
  // background at that level's own cell centers, instead of the plain-averaged
  // value RestrictCoefficients() would otherwise leave there -- near-identical to
  // MGCFCConformalFactor::FillPunctureCoefficients (same doc-comment rationale
  // applies: psi0 diverges at the puncture, so restriction's 8-cell average smooths
  // the peak and degrades the coarse-grid operator exactly where it matters most),
  // except this version also keeps alpha0 (which the psi-side version discards) to
  // build channel 5. Octet-refined patches are NOT touched here (still restricted
  // the old way), same Phase C deferral as the psi-side version.
  void FillPunctureCoefficients(Real m_bh);
};


//! \class MGCFCLapseDriver
//! \brief Multigrid driver for delta_(alpha psi) (Gmunu eq. 74), isolated (1/r
//! falloff, mg_multipole) boundary conditions.

class MGCFCLapseDriver : public MultigridDriver {
  public:
    MGCFCLapseDriver(MeshBlockPack *pmbp, ParameterInput *pin);
    ~MGCFCLapseDriver();

    void Solve(Driver *pdriver, int stage, Real dt = 0.0) final;

    // Load the matter-derived ingredients of K(x)/S -- Ũ+2S̃ (channel 0), delta_psi
    // = psi-psi0 (channel 1, cfc::CFC::delta_psi), and DeltaAhat^2 = Ahat^2_total -
    // Ahat0^2 (channel 2, subtracted here at the finest grid so what lands in
    // coeff_ is smooth/safe to restrict -- same reasoning as
    // MGCFCConformalFactorDriver::LoadNonlinearCoefficient's identical comment).
    // Finest level only -- these restrict normally to coarser levels via the
    // inherited RestrictCoefficients(), same treatment matter always gets. Split
    // out from the old fused LoadReactionCoefficient (Sec 3.8/Sec 5 Phase A item 4
    // lapse-side follow-up) so the actual K(x)/S combine can move into
    // LapseReactionRHS(), called fresh per point per level by SmoothPack/
    // CalculateDefectPack/CalculateFASRHSPack -- see this file's header comment.
    void LoadMatterCoefficients(const DvceArray5D<Real> &u_plus_2s_tilde,
                                 const DvceArray5D<Real> &delta_psi,
                                 const DvceArray5D<Real> &a_sq,
                                 const DvceArray5D<Real> &a0_sq, int ngh);

    // Load the analytic trumpet background's psi0 (channel 3), Ahat0^2 (channel 4),
    // and alpha0*psi0 (channel 5, u_alpha0_psi0 already *is* this product --
    // cfc::CFC maintains it directly, no extra multiply needed) -- LapseReactionRHS
    // needs all three to build the Sec 3.7 regularized K(x)/S. Zero-cost/inert when
    // <cfc> puncture_enabled is false (psi0=1, Ahat0^2=0, alpha0*psi0=1 everywhere,
    // cfc.cpp's ctor). Finest level only, like MGCFCConformalFactorDriver::
    // LoadPunctureCoefficients -- mostly redundant once Solve()'s
    // FillPunctureCoefficients() runs, but matters when puncture_enabled_ is false
    // (no FillPunctureCoefficients call then, so this is the only writer).
    void LoadPunctureCoefficients(const DvceArray5D<Real> &u_psi0,
                                   const DvceArray5D<Real> &a0_sq,
                                   const DvceArray5D<Real> &u_alpha0_psi0, int ngh);

    // retrieve the converged delta_(alpha psi) solution after Solve() completes.
    void RetrieveSolution(DvceArray5D<Real> &dst);

    // seed the finest level's own solution array (the V-cycle's initial guess) from
    // an externally-supplied delta_(alpha*psi) field -- see MGCFCConformalFactor-
    // Driver::SeedInitialGuess's doc comment for the full rationale (same pattern).
    void SeedInitialGuess(const DvceArray5D<Real> &guess, int ngh);

    void SmoothOctet(MGOctet &oct, int rlev, int color) final;
    void CalculateDefectOctet(MGOctet &oct, int rlev) final;
    void CalculateFASRHSOctet(MGOctet &oct, int rlev) final;

    friend class MGCFCLapse;

  private:
    // <cfc> puncture_enabled/puncture_mass -- read directly from pin at
    // construction, same convention as MGCFCConformalFactorDriver's identical
    // members, so Solve() can call FillPunctureCoefficients() itself without any
    // new call-site plumbing from cfc.cpp.
    bool puncture_enabled_;
    Real puncture_mass_;

 public:
  //! \brief Update the puncture mass mid-run (CFC accretes swallowed matter into the
  //! hole -- see CFC::AccreteExcisedMass).  Only the stored value needs changing:
  //! Solve() already re-runs FillPunctureCoefficients(puncture_mass_) over every
  //! multigrid level on every call, so the per-level analytic coefficients pick the
  //! new mass up automatically.  Mirrors the existing SetRobinCenter/
  //! SetMultipoleOrigin setters that cfc::CFC already calls each stage.
  void SetPunctureMass(Real m) { puncture_mass_ = m; }

 protected:

    // mgroot_ never receives coeff_ data via the generic TransferFromBlocksToRoot
    // (src_/u_ only) -- duplicates the relevant slice of that logic locally rather
    // than touching src/multigrid/. See MGCFCConformalFactorDriver::
    // TransferCoeffToRoot for the full rationale.
    void TransferCoeffToRoot();
};

#endif  // CFC_MG_CFC_LAPSE_HPP_
