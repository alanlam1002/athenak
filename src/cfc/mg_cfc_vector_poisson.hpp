#ifndef CFC_MG_CFC_VECTOR_POISSON_HPP_
#define CFC_MG_CFC_VECTOR_POISSON_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mg_cfc_vector_poisson.hpp
//! \brief defines MGCFCVectorPoisson[Driver], the multigrid solver for BOTH pieces of
//! one Shibata decomposition (Shibata 1999 eqs. 3.10-3.11): Delta P_i = S_i (the
//! 3-component vector potential) AND Delta eta = -S_i x^i (its paired scalar), packed
//! into one nvar_ = 4 solve (channels 0-2 = P_i, channel 3 = eta). All 4 channels are
//! fully independent flat Poisson equations (P_i of each other and of eta; eta simply
//! shares the same known source S_i) -- plain flat 7-point Laplacian per channel,
//! reusing the generic templated Smooth/CalculateDefect/CalculateFASRHS helpers in
//! Multigrid (unlike the conformal-factor/lapse solvers, this operator is linear, so
//! no Newton-Gauss-Seidel override is needed). Packing P_i and eta into one driver
//! (rather than two, as in an earlier version of this file) halves the number of
//! V-cycle solves and multipole-moment MPI_Allreduce calls per Shibata pair, since
//! both channels share the exact same boundary-condition configuration already.
//!
//! cfc::CFC calls LoadPoissonSource/Solve/RetrieveSolution once per Shibata pair, with
//! P_i's source packed at channels 0-2 and eta's source (Shibata eq. 3.11) packed at
//! channel 3 of the same array (see cfc::CFC::AssembleVectorSource). One instance of
//! this class is used for X^i (P_i+eta), a second (separate) instance is used for
//! beta^i (P_i+eta); cfc::CFC owns both (see cfc.hpp).

// Athenak headers
#include "../athena.hpp"
#include "../multigrid/multigrid.hpp"

class MeshBlockPack;
class ParameterInput;
class Multigrid;
class MultigridDriver;

//----------------------------------------------------------------------------------------
//! \struct CFCVectorPoissonStencil
//! \brief flat 7-point Laplacian stencil, decoupled across all 4 channels (P_i's 3
//! components plus eta); identical in form to gravity::GravityStencil.

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
//! \brief Multigrid object for the packed (P_i, eta) 4-channel potential

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
//! \brief Multigrid driver used once each for X^i and beta^i's packed (P_i, eta)
//! solve. Unlike gravity::MGGravityDriver::Solve() (which loads its own source from
//! the Gravity object and retrieves its own result), this driver exposes a
//! lower-level API (LoadPoissonSource/RetrieveSolution) so cfc::CFC can reuse one
//! class for two physically distinct equations with different sources.

class MGCFCVectorPoissonDriver : public MultigridDriver {
  public:
    MGCFCVectorPoissonDriver(MeshBlockPack *pmbp, ParameterInput *pin);
    ~MGCFCVectorPoissonDriver();

    // run the V-cycle/FMG solve on the packed (P_i, eta) right-hand side; assumes
    // LoadPoissonSource() was already called for this cycle.
    void Solve(Driver *pdriver, int stage, Real dt = 0.0) final;

    // load the packed 4-channel right-hand side (P_i's S_i at channels 0-2, Shibata
    // eq. 3.10; eta's -S_i x^i at channel 3, Shibata eq. 3.11), computed by
    // cfc::CFC::AssembleVectorSource(), onto the finest grid. Takes the raw backing
    // storage (e.g. cfc::CFC::u_p_src), not the AthenaTensor view over it:
    // Multigrid::LoadSource() requires a genuine DvceArray5D<Real>&, which an
    // AthenaTensor's Kokkos::subview-backed storage does not type-check against.
    void LoadPoissonSource(const DvceArray5D<Real> &p_src);

    // retrieve the converged (P_i, eta) solution after Solve() completes into the raw
    // backing storage (e.g. cfc::CFC::u_p_x) -- same reasoning as LoadPoissonSource.
    // cfc::CFC then reconstructs the physical vector from both channel ranges (see
    // cfc_reconstruct.hpp::ReconstructVectorFromPotentials).
    void RetrieveSolution(DvceArray5D<Real> &p_dst);

    // octet-level (AMR) physics, mirroring gravity::MGGravityDriver
    void SmoothOctet(MGOctet &oct, int rlev, int color) final;
    void CalculateDefectOctet(MGOctet &oct, int rlev) final;
    void CalculateFASRHSOctet(MGOctet &oct, int rlev) final;
    // ProlongateOctetBoundariesFluxCons is intentionally NOT overridden: the
    // MultigridDriver base default (plain nvar_-generic trilinear prolongation) is
    // sufficient here. Gravity overrides it for exact flux conservation, which
    // matters for a conserved potential; P_i/eta are auxiliary elliptic potentials
    // with no conservation law of their own, so the simpler base behavior applies.

    friend class MGCFCVectorPoisson;
  private:
    Real omega_;  // smoothing relaxation parameter (mirrors gravity::omega_)
};

#endif  // CFC_MG_CFC_VECTOR_POISSON_HPP_
