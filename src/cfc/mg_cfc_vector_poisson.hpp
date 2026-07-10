#ifndef CFC_MG_CFC_VECTOR_POISSON_HPP_
#define CFC_MG_CFC_VECTOR_POISSON_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mg_cfc_vector_poisson.hpp
//! \brief defines MGCFCVectorPoisson[Driver], the multigrid solver shared by both CFC
//! vector-elliptic equations (the vector potential X^i, Gmunu eq. 72, and the shift
//! beta^i, Gmunu eq. 75). Both equations have the form
//!   Delta V^i + (1/3) D^i(D_j V^j) = Source^i,
//! which per Shibata (1999) sec. 3 decomposes into 4 *independent* flat scalar Poisson
//! equations for (P_x, P_y, P_z, eta): Delta P_i = S_i, Delta eta = -S_i x^i. This class
//! solves exactly those 4 decoupled scalar equations (nvar_ = 4) with a plain flat
//! 7-point Laplacian stencil, reusing the generic templated Smooth/CalculateDefect/
//! CalculateFASRHS helpers in Multigrid (unlike the conformal-factor/lapse solvers,
//! this operator is linear, so no Newton-Gauss-Seidel override is needed).
//! One instance is used to solve for X^i's potentials, a second (separate) instance is
//! used to solve for beta^i's potentials; cfc::CFC owns both instances and reloads the
//! source/retrieves the result for each equation in turn (see cfc.hpp).

#include <vector>

// Athenak headers
#include "../athena.hpp"
#include "../multigrid/multigrid.hpp"

class MeshBlockPack;
class ParameterInput;
class Multigrid;
class MultigridDriver;

//----------------------------------------------------------------------------------------
//! \struct CFCVectorPoissonStencil
//! \brief flat 7-point Laplacian stencil, decoupled across the 4 potential channels
//! (P_x, P_y, P_z, eta); identical in form to gravity::GravityStencil.

struct CFCVectorPoissonStencil {
  Real omega_over_diag;

  template <typename ViewType>
  KOKKOS_INLINE_FUNCTION
  Real Apply(const ViewType &u, const ViewType &coeff,
             int m, int v, int k, int j, int i) const {
    return 6.0*u(m,v,k,j,i) - u(m,v,k+1,j,i) - u(m,v,k,j+1,i)
           - u(m,v,k,j,i+1) - u(m,v,k-1,j,i) - u(m,v,k,j-1,i)
           - u(m,v,k,j,i-1);
  }
};

//! \class MGCFCVectorPoisson
//! \brief Multigrid object for one set of 4 decomposed vector-Poisson scalars

class MGCFCVectorPoisson : public Multigrid {
 public:
  MGCFCVectorPoisson(MultigridDriver *pmd, MeshBlockPack *pmbp, int nghost,
                     bool on_host = false);
  ~MGCFCVectorPoisson();

  void SmoothPack(int color) final;
  void CalculateDefectPack() final;
  void CalculateFASRHSPack() final;
};


//! \class MGCFCVectorPoissonDriver
//! \brief Multigrid driver shared by the X^i and beta^i vector-Poisson solves.
//! Unlike gravity::MGGravityDriver::Solve() (which loads its own source from the
//! Gravity object and retrieves its own result), this driver exposes a lower-level API
//! (LoadPoissonSource/RetrieveSolution) so cfc::CFC can reuse one class for two
//! physically distinct equations with different sources.

class MGCFCVectorPoissonDriver : public MultigridDriver {
  public:
    MGCFCVectorPoissonDriver(MeshBlockPack *pmbp, ParameterInput *pin);
    ~MGCFCVectorPoissonDriver();

    // load the (4-component) right-hand side S_i, -S_i x^i and run the V-cycle/FMG
    // solve; assumes LoadPoissonSource() was already called for this cycle.
    void Solve(Driver *pdriver, int stage, Real dt = 0.0) final;

    // load the 4-component source (P_x, P_y, P_z, eta right-hand sides) computed by
    // cfc::CFC::AssembleVectorSource() onto the finest grid.
    void LoadPoissonSource(const DvceArray5D<Real> &src);

    // retrieve the converged (P_x, P_y, P_z, eta) solution after Solve() completes.
    void RetrieveSolution(DvceArray5D<Real> &dst);

    // octet-level (AMR) physics, mirroring gravity::MGGravityDriver
    void SmoothOctet(MGOctet &oct, int rlev, int color) final;
    void CalculateDefectOctet(MGOctet &oct, int rlev) final;
    void CalculateFASRHSOctet(MGOctet &oct, int rlev) final;
    void ProlongateOctetBoundariesFluxCons(MGOctet &oct,
         std::vector<Real> &cbuf, const std::vector<bool> &ncoarse) final;

    friend class MGCFCVectorPoisson;
  private:
    Real omega_;  // smoothing relaxation parameter (mirrors gravity::omega_)
};

#endif  // CFC_MG_CFC_VECTOR_POISSON_HPP_
