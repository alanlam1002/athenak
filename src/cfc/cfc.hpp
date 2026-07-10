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
//! Mirrors gravity::Gravity in that it's a thin physics-facing orchestrator that owns
//! a set of Multigrid/MultigridDriver subclass pairs (see mg_cfc_vector_poisson.hpp,
//! mg_cfc_scalar_poisson.hpp, mg_cfc_conformal_factor.hpp, mg_cfc_lapse.hpp) driving
//! them through the 6-step XCFC algorithm (Cheong et al. 2021 [arXiv:2012.07322]
//! sec. 2.6). Unlike gravity::Gravity, CFC does NOT call Driver::Execute() directly:
//! its steps must interleave with dyn_grmhd's own per-stage task graph (hydro
//! flux/update, then CFC's vector potential + conformal factor, then a single
//! conserved-to-primitive recovery shared with dyn_grmhd, then the remaining CFC
//! steps), which is only possible by queuing individual tasks into the shared
//! NumericalRelativity task graph (see QueueCFCTasks() and tasklist/
//! numerical_relativity.hpp) exactly as dyngr::DynGRMHD and z4c::Z4c already do.
//! Reads matter data directly from MeshBlockPack::pmhd->u0/w0 (not
//! MeshBlockPack::ptmunu -- see cfc.cpp) and writes the resulting metric into
//! MeshBlockPack::padm->u_adm (consumed by dyn_grmhd's Riemann solver/ConToPrim).

// Athenak headers
#include "../athena.hpp"
#include "../athena_tensor.hpp"
#include "../mesh/meshblock_pack.hpp"
#include "../parameter_input.hpp"
#include "../tasklist/task_list.hpp"
#include "mg_cfc_vector_poisson.hpp"
#include "mg_cfc_scalar_poisson.hpp"
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

  // intermediate fields, all defined on the finest mesh grid. Vector/tensor physical
  // quantities are represented as AthenaTensor views (as in the z4c/adm modules,
  // e.g. adm::ADM::ADM_vars), each shallow-sliced (InitWithShallowSlice) from an
  // underlying flat "u_*" storage array; genuine scalars remain plain
  // DvceArray5D<Real> (as gravity::Gravity::phi does).

  DvceArray5D<Real> u_x;                              // storage backing x_u (3 comp.)
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> x_u;      // X^i, vector potential (eq. 72)

  DvceArray5D<Real> u_beta;                            // storage backing beta_u (3 comp.)
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> beta_u;   // beta^i, shift vector

  DvceArray5D<Real> u_adual;                           // storage backing a_dd (6 comp.)
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> a_dd;     // Adual^ij (Gmunu eq. 76)

  DvceArray5D<Real> a_sq;        // Ahat^2 = f_ik f_jl Adual^kl Adual^ij, scalar
  DvceArray5D<Real> psi;         // psi (conformal factor), scalar
  DvceArray5D<Real> alpha_psi;   // alpha*psi (lapse times conformal factor), scalar

  // matter source terms rescaled by the current psi^6 (Gmunu sec. 2.6, U-tilde etc.).
  // U and S_i are built directly from the evolved conserved state
  // (pmy_pack->pmhd->u0: D, S_i, tau -- see AssembleVectorSource/SolveConformalFactor
  // in cfc.cpp), NOT from MeshBlockPack::ptmunu: Tmunu is only populated when a z4c
  // block is active (dyn_grmhd.cpp's QueueDynGRMHDTasks), but CFC's primary use case
  // has no z4c free evolution, so ptmunu may not exist at all.
  DvceArray5D<Real> u_tilde;       // Ũ = psi^6 U, scalar

  DvceArray5D<Real> u_stilde;                            // storage backing s_tilde_d
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> s_tilde_d;  // S-tilde_i = psi^6 S_i

  DvceArray5D<Real> s_tilde;       // S-tilde = psi^6 S (trace of S_ij), scalar

  // Shibata (1999) sec. 3 decomposition: each vector equation (X^i, beta^i) reduces
  // to one vector potential P_i (eq. 3.10: Delta P_i = S_i) plus one scalar
  // potential eta (eq. 3.11: Delta eta = -S_i x^i) -- P_i is a genuine vector
  // (AthenaTensor), eta is a genuine scalar (plain DvceArray5D<Real>, like
  // psi/alpha_psi above). P_x, P_y, P_z, and eta are all mutually independent
  // equations, but P_i is solved first (MGCFCVectorPoissonDriver) and eta is solved
  // afterward (MGCFCScalarPoissonDriver), since eta's source is assembled from the
  // same known vector source S_i used for P_i. Both are reconstructed into
  // x_u/beta_u by cfc::ReconstructVectorFromPotentials.
  DvceArray5D<Real> u_p_x;                            // storage backing p_x (3 comp.)
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> p_x;     // P_i for the X^i decomposition
  DvceArray5D<Real> eta_x;                            // eta for the X^i decomposition

  DvceArray5D<Real> u_p_beta;                         // storage backing p_beta (3 comp.)
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> p_beta;  // P_i for the beta^i decomposition
  DvceArray5D<Real> eta_beta;                         // eta for the beta^i decomposition

  // multigrid solvers, one per distinct elliptic equation (shared classes for the
  // two vector solves and the two scalar solves -- see mg_cfc_vector_poisson.hpp /
  // mg_cfc_scalar_poisson.hpp)
  MGCFCVectorPoissonDriver *pmgd_px;      // solves for X^i's P_i (first)
  MGCFCScalarPoissonDriver *pmgd_etax;    // solves for X^i's eta (second)
  MGCFCVectorPoissonDriver *pmgd_pbeta;   // solves for beta^i's P_i (first)
  MGCFCScalarPoissonDriver *pmgd_etabeta; // solves for beta^i's eta (second)
  MGCFCConformalFactorDriver *pmgd_psi;
  MGCFCLapseDriver *pmgd_alpha;

  // Queues this module's tasks into the shared NumericalRelativity task graph
  // (pmy_pack->pnr), mirroring dyngr::DynGRMHD::QueueDynGRMHDTasks()/
  // z4c::Z4c::QueueZ4cTasks(). Called once, from NumericalRelativity::
  // AssembleNumericalRelativityTasks() (tasklist/numerical_relativity.cpp), NOT
  // called directly from Driver::Execute() the way gravity::Gravity is: CFC's steps
  // must interleave with dyn_grmhd's own hydro/con2prim tasks (see cfc.cpp and
  // tasklist/numerical_relativity.hpp's CFC_* TaskName values), which is only
  // possible through the task graph. Wires:
  //   CFC_SolveVecX      depends on {MHD_AddSrc}   (post flux+source-update u0)
  //   CFC_SolvePsi       depends on {CFC_SolveVecX}
  //   CFC_RescaleSrc     depends on {MHD_C2P}       (the SAME con2prim dyn_grmhd
  //                                                   already runs -- see below)
  //   CFC_SolveLapse     depends on {CFC_RescaleSrc}
  //   CFC_SolveShift     depends on {CFC_SolveLapse}
  //   CFC_AssembleFinal  depends on {CFC_SolveShift}
  // dyn_grmhd.cpp's MHD_C2P/MHD_Newdt tasks in turn take CFC_SolvePsi/
  // CFC_AssembleFinal as *optional* dependencies, so a single con2prim per stage
  // serves both dyn_grmhd's own needs and CFC's (no second con2prim call here).
  void QueueCFCTasks();

  // Task-graph entry points (TaskStatus(Driver*, int) is the signature
  // NumericalRelativity::QueueTask requires). Each is a thin wrapper around the
  // like-named private Step method below; kept separate so the "Step N" structure
  // from the original design doc stays visible.
  TaskStatus SolveVecXTask(Driver *pdriver, int stage);       // steps 1-2
  TaskStatus SolvePsiTask(Driver *pdriver, int stage);        // step 3
  TaskStatus RescaleSrcTask(Driver *pdriver, int stage);      // step 4
  TaskStatus SolveLapseTask(Driver *pdriver, int stage);      // step 5
  TaskStatus SolveShiftTask(Driver *pdriver, int stage);      // step 6
  TaskStatus AssembleFinalTask(Driver *pdriver, int stage);   // final assembly

 private:
  // shared helper: build the Shibata (1999) eq. 3.10-3.11 sources -- p_src (P_i's
  // vector right-hand side S_i) and eta_src (eta's scalar right-hand side -S_i x^i,
  // built from that same S_i) -- for either the X^i solve (for_shift=false, built
  // directly from the post-source-update conserved momentum pmy_pack->pmhd->u0 per
  // eq. 72) or the beta^i solve (for_shift=true, built from alpha, psi, Adual^ij,
  // S-tilde_i per eq. 75). P_i and eta are independent equations;
  // SolveVectorPotential/SolveShift solve P_i to completion first and eta second
  // (see below), rather than solving them simultaneously.
  void AssembleVectorSource(AthenaTensor<Real, TensorSymm::NONE, 3, 1> &p_src,
                            DvceArray5D<Real> &eta_src, bool for_shift);

  // Step 1: build the eq. 72 source directly from pmy_pack->pmhd->u0 (the conserved
  // state right after this stage's hydro flux+source update -- see AssembleVectorSource),
  // solve pmgd_px for P_i (p_x) first, then solve pmgd_etax for eta (eta_x).
  void SolveVectorPotential(Driver *pdriver, int stage);

  // Step 2: Adual^ij from X^i (eq. 76), then Ahat^2 (cfc_reconstruct.hpp).
  void ComputeADual();

  // Step 3: solve eq. 73 for psi (nonlinear), then immediately write psi4/g_dd into
  // pmy_pack->padm->u_adm via cfc::AssembleConformalMetric -- the single con2prim
  // shared with dyn_grmhd (MHD_C2P) needs a valid g_dd to invert conserved to
  // primitive variables, and is queued to depend on this step (see QueueCFCTasks).
  void SolveConformalFactor(Driver *pdriver, int stage);

  // Step 4: build S-tilde (trace of S_ij, needed by the lapse equation) from the
  // primitives MHD_C2P just recovered (pmy_pack->pmhd->w0) and EOS enthalpy -- no
  // con2prim call here: RescaleSrcTask depends on MHD_C2P, so w0 is already fresh
  // against the psi/g_dd SolveConformalFactor() wrote. Ũ and S-tilde_i do NOT need
  // rebuilding here either: they were already built directly from the
  // psi^6-densitized evolved conserved state (pmy_pack->pmhd->u0: D, S_i, tau) in
  // steps 1 and 3, since sqrt(gamma) = psi^6 is already baked into those conserved
  // variables -- only the trace source requires primitives.
  void RescaleMatterSources(Driver *pdriver, int stage);

  // Step 5: solve eq. 74 for alpha*psi (nonlinear); extract alpha = (alpha*psi)/psi.
  void SolveLapse(Driver *pdriver, int stage);

  // Step 6: build the eq. 75 source, solve pmgd_pbeta for P_i (p_beta) first, then
  // solve pmgd_etabeta for eta (eta_beta), then reconstruct beta^i
  // (cfc_reconstruct.hpp).
  void SolveShift(Driver *pdriver, int stage);

  // Final assembly: vK_dd, alpha, beta_u -> pmy_pack->padm->u_adm (via
  // cfc::AssembleLapseShiftK). psi4/g_dd were already written by
  // SolveConformalFactor(), right after step 3.
  void AssembleADM();
};

}  // namespace cfc

#endif  // CFC_CFC_HPP_
