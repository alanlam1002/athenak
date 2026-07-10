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
//! As with the conformal factor, this operator is nonlinear in the unknown (self-
//! coupled multiplicatively on the right-hand side): SmoothPack/CalculateDefectPack/
//! CalculateFASRHSPack are overridden with hand-written Newton-Gauss-Seidel point
//! relaxation rather than reusing the generic linear Smooth<StencilOp> template.
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
};


//! \class MGCFCLapseDriver
//! \brief Multigrid driver for delta_(alpha psi) (Gmunu eq. 74), isolated (1/r
//! falloff, mg_multipole) boundary conditions.

class MGCFCLapseDriver : public MultigridDriver {
  public:
    MGCFCLapseDriver(MeshBlockPack *pmbp, ParameterInput *pin);
    ~MGCFCLapseDriver();

    void Solve(Driver *pdriver, int stage, Real dt = 0.0) final;

    // load 2 pi (Ũ + 2 S̃), the linear-in-source part of the right-hand side.
    void LoadMatterSource(const DvceArray5D<Real> &u_plus_2s_tilde);

    // load the known fixed fields (psi, Ahat^2) this equation's nonlinear
    // coefficients depend on; stored via the base class's coeff_/ncoeff_
    // scaffolding (ncoeff_ = 2: channel 0 = psi, channel 1 = Ahat^2).
    void LoadKnownFields(const DvceArray5D<Real> &psi, const DvceArray5D<Real> &a_sq);

    // retrieve the converged delta_(alpha psi) solution after Solve() completes.
    void RetrieveSolution(DvceArray5D<Real> &dst);

    void SmoothOctet(MGOctet &oct, int rlev, int color) final;
    void CalculateDefectOctet(MGOctet &oct, int rlev) final;
    void CalculateFASRHSOctet(MGOctet &oct, int rlev) final;

    friend class MGCFCLapse;
};

#endif  // CFC_MG_CFC_LAPSE_HPP_
