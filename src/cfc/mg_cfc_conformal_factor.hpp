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
//! carry the precomputed Ahat^2 field alongside the solution.

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
};


//! \class MGCFCConformalFactorDriver
//! \brief Multigrid driver for delta_psi = psi - 1 (Gmunu eq. 73), isolated (1/r
//! falloff, mg_multipole) boundary conditions.

class MGCFCConformalFactorDriver : public MultigridDriver {
  public:
    MGCFCConformalFactorDriver(MeshBlockPack *pmbp, ParameterInput *pin);
    ~MGCFCConformalFactorDriver();

    void Solve(Driver *pdriver, int stage, Real dt = 0.0) final;

    // load -2 pi Ũ (matter energy density source, see cfc::CFC::RescaleMatterSources)
    void LoadMatterSource(const DvceArray5D<Real> &u_tilde);

    // load Ahat^2 = f_ik f_jl Adual^kl Adual^ij (from cfc::ComputeADualFromX), stored
    // via the base class's under-used coeff_/ncoeff_ scaffolding (ncoeff_ = 1).
    void LoadNonlinearCoefficient(const DvceArray5D<Real> &a_sq);

    // retrieve the converged delta_psi = psi - 1 solution after Solve() completes.
    void RetrieveSolution(DvceArray5D<Real> &dst);

    void SmoothOctet(MGOctet &oct, int rlev, int color) final;
    void CalculateDefectOctet(MGOctet &oct, int rlev) final;
    void CalculateFASRHSOctet(MGOctet &oct, int rlev) final;

    friend class MGCFCConformalFactor;
};

#endif  // CFC_MG_CFC_CONFORMAL_FACTOR_HPP_
