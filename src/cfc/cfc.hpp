#ifndef CFC_CFC_HPP_
#define CFC_CFC_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cfc.hpp
//! \brief defines the CFC class: the conformally-flat-condition (XCFC) metric solver.
//!
//! Mirrors gravity::Gravity: a thin physics-facing orchestrator that owns a set of
//! Multigrid/MultigridDriver subclass pairs (see mg_cfc_vector_poisson.hpp,
//! mg_cfc_conformal_factor.hpp, mg_cfc_lapse.hpp) and drives them through the 6-step
//! XCFC algorithm (Cheong et al. 2021 [arXiv:2012.07322] sec. 2.6) each time Solve()
//! is called. Reads the matter stress-energy tensor from MeshBlockPack::ptmunu
//! (populated by dyngr::DynGRMHD::SetTmunu) and writes the resulting metric into
//! MeshBlockPack::padm->u_adm (consumed by dyn_grmhd's Riemann solver/ConToPrim).
//!
//! Solve() is called directly from Driver::Execute(), once per RK stage, exactly like
//! gravity::Gravity::pmgd->Solve() -- see driver.cpp.

// Athenak headers
#include "../athena.hpp"
#include "../mesh/meshblock_pack.hpp"
#include "../parameter_input.hpp"
#include "mg_cfc_vector_poisson.hpp"
#include "mg_cfc_conformal_factor.hpp"
#include "mg_cfc_lapse.hpp"

class MeshBlockPack;
class ParameterInput;
class Driver;

namespace cfc {

class CFC {
 public:
  CFC(MeshBlockPack *pmbp, ParameterInput *pin);
  ~CFC();

  MeshBlockPack *pmy_pack;

  // intermediate fields (device arrays), all defined on the finest mesh grid
  DvceArray5D<Real> x_u;         // X^i, vector potential for eq. 72 (3 components)
  DvceArray5D<Real> a_dd;        // Adual^ij (Gmunu eq. 76), symmetric (6 components)
  DvceArray5D<Real> a_sq;        // Ahat^2 = f_ik f_jl Adual^kl Adual^ij (1 component)
  DvceArray5D<Real> psi;         // psi (conformal factor), scalar
  DvceArray5D<Real> alpha_psi;   // alpha*psi (lapse times conformal factor), scalar
  DvceArray5D<Real> beta_u;      // beta^i (shift vector), 3 components

  // matter source terms rescaled by the current psi^6 (Gmunu sec. 2.6, U-tilde etc.)
  DvceArray5D<Real> u_tilde;       // Ũ = psi^6 U
  DvceArray5D<Real> s_tilde_d;     // S-tilde_i = psi^6 S_i
  DvceArray5D<Real> s_tilde;       // S-tilde = psi^6 S (trace of S_ij)

  // multigrid solvers, one per distinct elliptic equation (shared class for the two
  // vector-Poisson solves -- see mg_cfc_vector_poisson.hpp)
  MGCFCVectorPoissonDriver *pmgd_vecx;    // solves for X^i's 4 scalar potentials
  MGCFCVectorPoissonDriver *pmgd_vecbeta; // solves for beta^i's 4 scalar potentials
  MGCFCConformalFactorDriver *pmgd_psi;
  MGCFCLapseDriver *pmgd_alpha;

  // Main entry point: run the full 6-step XCFC solve for the current stage and write
  // the result into pmy_pack->padm->u_adm. Called directly from Driver::Execute(),
  // once per RK stage (mirrors gravity::Gravity's pmgd->Solve() call site).
  void Solve(Driver *pdriver, int stage);

 private:
  // shared helper: build the 4-component Poisson source (P_x, P_y, P_z, eta
  // right-hand sides) for either the X^i solve (for_shift=false, source built from
  // S-tilde_i per eq. 72) or the beta^i solve (for_shift=true, source built from
  // alpha, psi, Adual^ij, S-tilde_i per eq. 75).
  void AssembleVectorSource(DvceArray5D<Real> &src, bool for_shift);

  // Step 1: build S-tilde_i as the eq. 72 source and solve for X^i's 4 potentials.
  void SolveVectorPotential(Driver *pdriver, int stage);

  // Step 2: Adual^ij from X^i (eq. 76), then Ahat^2 (cfc_reconstruct.hpp).
  void ComputeADual();

  // Step 3: solve eq. 73 for psi (nonlinear).
  void SolveConformalFactor(Driver *pdriver, int stage);

  // Step 4: rescale Ũ, S-tilde, S-tilde_i using the newly solved psi^6.
  void RescaleMatterSources();

  // Step 5: solve eq. 74 for alpha*psi (nonlinear); extract alpha = (alpha*psi)/psi.
  void SolveLapse(Driver *pdriver, int stage);

  // Step 6: build the eq. 75 source and solve for beta^i's 4 scalar potentials, then
  // reconstruct beta^i (cfc_reconstruct.hpp).
  void SolveShift(Driver *pdriver, int stage);

  // Final assembly: psi4, g_dd, vK_dd, alpha, beta_u -> pmy_pack->padm->u_adm.
  void AssembleADM();
};

}  // namespace cfc

#endif  // CFC_CFC_HPP_
