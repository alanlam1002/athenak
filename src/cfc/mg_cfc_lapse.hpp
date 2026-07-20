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
//! Unlike eq. 73 (psi), this operator is NOT actually nonlinear: by the time this
//! solve runs, psi and Ahat^2 are already converged, fixed fields (from the earlier
//! X^i/psi steps), so the bracketed factor
//!   K(x) := 2 pi (Ũ + 2 S̃) psi^-2 + (7/8) Ahat^2 psi^-8
//! depends only on those known fields, not on the unknown alpha*psi itself. The
//! equation is therefore an affine (screened/Helmholtz-type) equation in
//! delta_(alpha psi) = alpha*psi - 1: Delta(u+1) - K(x)*(u+1) = 0. It still can't
//! reuse the generic Smooth<StencilOp> template (that assumes a *constant* diagonal
//! via omega_over_diag; here the diagonal 6 + dx^2*K(x) varies per point), so
//! SmoothPack/CalculateDefectPack/CalculateFASRHSPack are still hand-written -- but
//! since F(u) is affine in u, the per-point "Newton" step (u_new = u_old -
//! F(u_old)/F'(u_old)) is an *exact* one-step Gauss-Seidel solve, not an approximate
//! linearization, and needs no damping or positivity floor the way eq. 73's does.
//! The solve is done for the deviation delta_(alpha psi) = alpha*psi - 1.

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
};


//! \class MGCFCLapseDriver
//! \brief Multigrid driver for delta_(alpha psi) (Gmunu eq. 74), isolated (1/r
//! falloff, mg_multipole) boundary conditions.

class MGCFCLapseDriver : public MultigridDriver {
  public:
    MGCFCLapseDriver(MeshBlockPack *pmbp, ParameterInput *pin);
    ~MGCFCLapseDriver();

    void Solve(Driver *pdriver, int stage, Real dt = 0.0) final;

    // Compute K(x) = 2*pi*(Utilde+2*Stilde)*psi^-2 + (7/8)*Ahat^2*psi^-8 once, at the
    // finest level, from the three known fixed fields it depends on, and store only
    // that single value in coeff_ (channel 0, ncoeff_ = 1). Round 16 fix: an earlier
    // version stored Utilde+2*Stilde/psi/Ahat^2 as three separate coeff_ channels and
    // recomputed K(x) fresh at every V-cycle level (including coarse ones) from
    // independently-restricted copies of each -- inconsistent with FAS, since
    // restrict(f(a,b)) != f(restrict(a), restrict(b)) for this nonlinear combination;
    // psi and Ahat^2 are genuinely fixed *coefficients* here (unlike psi's own solver,
    // where psi is the local unknown u+1, correctly carried through the standard FAS
    // u_ restriction instead). Precomputing K(x) once and restricting it directly
    // (via the ordinary RestrictCoefficients()) is the FAS-consistent treatment; see
    // this file's header comment (mg_cfc_lapse.cpp) for the full derivation. NOT
    // stored in src_ via Multigrid::LoadSource(): src_ is exactly what the generic
    // V-cycle machinery restricts *and* adds FAS tau-corrections into, which would
    // corrupt K(x) (Finding B, plan addendum #3). u_plus_2s_tilde/delta_psi/a_sq are
    // all assumed padded to the same depth ngh -- the mesh's own NGHOST, not this
    // driver's shallower ngh_ (plan addendum #4, Finding H). delta_psi is psi - 1
    // (see cfc::CFC::delta_psi's doc comment, cfc.hpp) -- the physical psi that K(x)
    // actually needs is reconstructed internally (+1.0).
    void LoadReactionCoefficient(const DvceArray5D<Real> &u_plus_2s_tilde,
                                 const DvceArray5D<Real> &delta_psi,
                                 const DvceArray5D<Real> &a_sq, int ngh);

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
    // Finding C (plan addendum #3): mgroot_ never receives coeff_ data via the
    // generic TransferFromBlocksToRoot (src_/u_ only) -- duplicates the relevant
    // slice of that logic locally rather than touching src/multigrid/. See
    // MGCFCConformalFactorDriver::TransferCoeffToRoot for the full rationale.
    void TransferCoeffToRoot();

    // Relative-change convergence check (DEVELOPMENT.md item 20) -- tried, found
    // no measurable improvement over the base class's defect-norm SolveMG(), and
    // reverted (see Solve()'s doc comment in the .cpp for the still-present,
    // commented-out implementation). u_prev_ is unused while that's reverted.
    // DvceArray5D<Real> u_prev_;
};

#endif  // CFC_MG_CFC_LAPSE_HPP_
