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
    // top docstring and the .cpp's file-level comment for why.
    void LoadMatterSource(const DvceArray5D<Real> &u_tilde);

    // load Ahat^2 = f_ik f_jl Adual^kl Adual^ij (from cfc::ComputeADualFromX), stored
    // in coeff_ channel 1 (ncoeff_ = 2 total). Multigrid::LoadCoefficients() can't be
    // reused for either load (it copies all ncoeff_ channels in one shot, no
    // per-channel offset) -- both loaders do their own single-channel par_for via
    // the CoeffAtLevel() accessor.
    void LoadNonlinearCoefficient(const DvceArray5D<Real> &a_sq);

    // retrieve the converged delta_psi = psi - 1 solution after Solve() completes.
    void RetrieveSolution(DvceArray5D<Real> &dst);

    void SmoothOctet(MGOctet &oct, int rlev, int color) final;
    void CalculateDefectOctet(MGOctet &oct, int rlev) final;
    void CalculateFASRHSOctet(MGOctet &oct, int rlev) final;

    friend class MGCFCConformalFactor;

  private:
    // Newton-relaxation controls (Gmunu sec. 2.7.2, eq. 94). mg_omega_psi_ damps
    // the per-point Newton step (1.0 = undamped); psi_floor_ prevents psi = u+1
    // from being driven non-positive (psi^-7 is ill-defined there) on a bad guess.
    Real mg_omega_psi_, psi_floor_;

    // MultigridDriver::TransferFromBlocksToRoot (multigrid_driver.cpp) aggregates
    // every rank's coarsest per-block cell into the distributed root grid (mgroot_)
    // via MPI_Allgatherv, but only for src_/u_ -- never coeff_. That transfer runs
    // for any multi-meshblock mesh (not just AMR), so mgroot_ needs its own coeff_
    // (Ũ, Ahat^2) populated the same way before the V-cycle can reach the root level.
    // Deliberately NOT a change to src/multigrid/: duplicates the relevant slice of
    // TransferFromBlocksToRoot's logic locally, restricted to the non-refined
    // (octet-free) case, since AMR+CFC is guarded against in Solve() (see .cpp).
    void TransferCoeffToRoot();
};

#endif  // CFC_MG_CFC_CONFORMAL_FACTOR_HPP_
