#ifndef CFC_MG_CFC_SCALAR_POISSON_HPP_
#define CFC_MG_CFC_SCALAR_POISSON_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mg_cfc_scalar_poisson.hpp
//! \brief defines MGCFCScalarPoisson[Driver], solving Shibata (1999) eq. 3.11:
//!   Delta eta = -S_i x^i,
//! the scalar half of the Shibata decomposition (the other half, P_i, is solved
//! first by mg_cfc_vector_poisson.hpp). eta's source -S_i x^i is built directly from
//! the same known vector source S_i used for P_i (Gmunu eq. 72's S-tilde_i, or eq.
//! 75's combination), so cfc::CFC solves P_i to completion first, then assembles and
//! solves eta's independent scalar equation (see cfc::CFC::SolveVectorPotential /
//! SolveShift in cfc.cpp).
//!
//! Linear, constant-diagonal flat 7-point Laplacian (nvar_ = 1) -- reuses the generic
//! templated Smooth/CalculateDefect/CalculateFASRHS helpers in Multigrid, same as
//! mg_cfc_vector_poisson.hpp (no Newton-Gauss-Seidel override needed).

// Athenak headers
#include "../athena.hpp"
#include "../multigrid/multigrid.hpp"

class MeshBlockPack;
class ParameterInput;
class Multigrid;
class MultigridDriver;

//! \class MGCFCScalarPoisson
//! \brief Multigrid object for the scalar potential eta

class MGCFCScalarPoisson : public Multigrid {
 public:
  MGCFCScalarPoisson(MultigridDriver *pmd, MeshBlockPack *pmbp, int nghost,
                     bool on_host = false);
  ~MGCFCScalarPoisson();

  void SmoothPack(int color) final;
  void CalculateDefectPack() final;
  void CalculateFASRHSPack() final;
};


//! \class MGCFCScalarPoissonDriver
//! \brief Multigrid driver shared by the X^i and beta^i eta solves. As with
//! MGCFCVectorPoissonDriver, exposes a lower-level LoadPoissonSource/RetrieveSolution
//! API (rather than gravity::MGGravityDriver's self-contained Solve()) so cfc::CFC
//! can reuse one class for two physically distinct equations with different sources.

class MGCFCScalarPoissonDriver : public MultigridDriver {
  public:
    MGCFCScalarPoissonDriver(MeshBlockPack *pmbp, ParameterInput *pin);
    ~MGCFCScalarPoissonDriver();

    // run the V-cycle/FMG solve on eta's right-hand side; assumes LoadPoissonSource()
    // was already called for this cycle.
    void Solve(Driver *pdriver, int stage, Real dt = 0.0) final;

    // load eta's scalar right-hand side -S_i x^i (Shibata eq. 3.11), computed by
    // cfc::CFC::AssembleVectorSource() from the already-solved P_i and the original
    // vector source S_i, onto the finest grid.
    void LoadPoissonSource(const DvceArray5D<Real> &eta_src);

    // retrieve the converged eta solution after Solve() completes.
    void RetrieveSolution(DvceArray5D<Real> &eta_dst);

    // octet-level (AMR) physics, mirroring gravity::MGGravityDriver
    void SmoothOctet(MGOctet &oct, int rlev, int color) final;
    void CalculateDefectOctet(MGOctet &oct, int rlev) final;
    void CalculateFASRHSOctet(MGOctet &oct, int rlev) final;

    friend class MGCFCScalarPoisson;
  private:
    Real omega_;  // smoothing relaxation parameter (mirrors gravity::omega_)
};

#endif  // CFC_MG_CFC_SCALAR_POISSON_HPP_
