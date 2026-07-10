#ifndef CFC_MG_CFC_VECTOR_POISSON_HPP_
#define CFC_MG_CFC_VECTOR_POISSON_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mg_cfc_vector_poisson.hpp
//! \brief defines MGCFCVectorPoisson[Driver], the multigrid solver shared by both CFC
//! vector-potential equations (Shibata 1999 eq. 3.10): Delta P_i = S_i, one of the two
//! pieces the Shibata decomposition splits each CFC vector equation (X^i, Gmunu eq.
//! 72; beta^i, Gmunu eq. 75) into. The 3 components P_x, P_y, P_z are fully
//! independent of each other (and of eta) -- nvar_ = 3, plain flat 7-point Laplacian,
//! reusing the generic templated Smooth/CalculateDefect/CalculateFASRHS helpers in
//! Multigrid (unlike the conformal-factor/lapse solvers, this operator is linear, so
//! no Newton-Gauss-Seidel override is needed).
//!
//! cfc::CFC solves P_i here *first* (LoadPoissonSource/Solve/RetrieveSolution), then
//! builds and solves eta's scalar equation (Shibata eq. 3.11, see
//! mg_cfc_scalar_poisson.hpp) using the same known source S_i. One instance of this
//! class is used for X^i's P_i, a second (separate) instance is used for beta^i's
//! P_i; cfc::CFC owns both (see cfc.hpp).

#include <vector>

// Athenak headers
#include "../athena.hpp"
#include "../athena_tensor.hpp"
#include "../multigrid/multigrid.hpp"

class MeshBlockPack;
class ParameterInput;
class Multigrid;
class MultigridDriver;

//----------------------------------------------------------------------------------------
//! \struct CFCVectorPoissonStencil
//! \brief flat 7-point Laplacian stencil, decoupled across the 3 vector components;
//! identical in form to gravity::GravityStencil.

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
//! \brief Multigrid object for the 3-component vector potential P_i

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
//! \brief Multigrid driver shared by the X^i and beta^i vector-potential (P_i)
//! solves. Unlike gravity::MGGravityDriver::Solve() (which loads its own source from
//! the Gravity object and retrieves its own result), this driver exposes a
//! lower-level API (LoadPoissonSource/RetrieveSolution) so cfc::CFC can reuse one
//! class for two physically distinct equations with different sources.

class MGCFCVectorPoissonDriver : public MultigridDriver {
  public:
    MGCFCVectorPoissonDriver(MeshBlockPack *pmbp, ParameterInput *pin);
    ~MGCFCVectorPoissonDriver();

    // run the V-cycle/FMG solve on P_i's right-hand side; assumes LoadPoissonSource()
    // was already called for this cycle.
    void Solve(Driver *pdriver, int stage, Real dt = 0.0) final;

    // load P_i's vector right-hand side S_i (Shibata eq. 3.10), computed by
    // cfc::CFC::AssembleVectorSource(), onto the finest grid.
    void LoadPoissonSource(const AthenaTensor<Real, TensorSymm::NONE, 3, 1> &p_src);

    // retrieve the converged P_i solution after Solve() completes. cfc::CFC then uses
    // both P_i and the original source S_i to build and solve eta's scalar equation
    // (see mg_cfc_scalar_poisson.hpp) before reconstructing the physical vector.
    void RetrieveSolution(AthenaTensor<Real, TensorSymm::NONE, 3, 1> &p_dst);

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
