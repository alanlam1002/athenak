#ifndef CFC_CFC_RECONSTRUCT_HPP_
#define CFC_CFC_RECONSTRUCT_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cfc_reconstruct.hpp
//! \brief free-function Kokkos kernels used by cfc::CFC that are not multigrid solves:
//! algebraic reconstruction of Adual^ij from a vector potential, reconstruction of a
//! vector field from the Shibata (1999) 4-scalar decomposition, and the two-phase
//! assembly of the XCFC solution into the ADM variables consumed by dyn_grmhd (an
//! early psi4/g_dd write needed before primitive recovery, then a final write of
//! alpha/beta_u/vK_dd -- see AssembleConformalMetric/AssembleLapseShiftK below).
//!
//! Vector/tensor arguments use AthenaTensor views (as in the z4c/adm modules); genuine
//! scalars (eta, psi, alpha_psi) remain plain DvceArray5D<Real>. AssembleConformalMetric/
//! AssembleLapseShiftK take delta_psi = psi - 1 / delta_alpha_psi = alpha*psi - 1 (see
//! cfc::CFC::delta_psi/delta_alpha_psi's doc comment in cfc.hpp for why), not the
//! physical fields directly, and reconstruct the physical value internally (+1.0).

#include "athena.hpp"
#include "athena_tensor.hpp"

// forward declarations
class MeshBlockPack;

namespace cfc {

//----------------------------------------------------------------------------------------
//! \fn void ComputeADualFromX(MeshBlockPack *pmbp,
//!                             const AthenaTensor<Real, TensorSymm::NONE, 3, 1> &x_u,
//!                             AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &a_dd)
//! \brief Gmunu (2021) eq. 76: Adual^ij ~= D^i X^j + D^j X^i - (2/3) D_k X^k f^ij,
//! evaluated with flat-space finite differences (f_ij = delta_ij in Cartesian).
//! x_u: X^i. a_dd: Adual^ij (symmetric).
void ComputeADualFromX(MeshBlockPack *pmbp,
                        const AthenaTensor<Real, TensorSymm::NONE, 3, 1> &x_u,
                        AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &a_dd);

//----------------------------------------------------------------------------------------
//! \fn void ReconstructVectorFromPotentials(MeshBlockPack *pmbp,
//!            const AthenaTensor<Real, TensorSymm::NONE, 3, 1> &p_i,
//!            const DvceArray5D<Real> &eta,
//!            AthenaTensor<Real, TensorSymm::NONE, 3, 1> &v_u, int eta_chan = 0)
//! \brief Shibata (1999) eq. 3.9: V^j = (7/8) P_j - (1/8)(eta,_j + P_k,_j x^k),
//! reconstructing a vector field from the decomposed vector potential P_i and scalar
//! potential eta (both solved together, packed into one nvar_=4 array -- P_i at
//! channels 0-2, eta at channel 3 -- see mg_cfc_vector_poisson.hpp). Used for both
//! X^i (internal, feeds ComputeADualFromX) and beta^i (written to the ADM shift).
//! p_i: P_i (vector). eta: the array eta_chan is sliced from -- typically the same
//! packed array p_i's own backing storage is a view of, with eta_chan=3. v_u:
//! reconstructed vector, e.g. X^i/beta^i.
void ReconstructVectorFromPotentials(MeshBlockPack *pmbp,
                                      const AthenaTensor<Real, TensorSymm::NONE, 3, 1>
                                          &p_i,
                                      const DvceArray5D<Real> &eta,
                                      AthenaTensor<Real, TensorSymm::NONE, 3, 1> &v_u,
                                      int eta_chan = 0);

//----------------------------------------------------------------------------------------
//! \fn void AssembleConformalMetric(MeshBlockPack *pmbp,
//!                                   const DvceArray5D<Real> &delta_psi)
//! \brief Writes psi4 = psi^4 and g_dd = psi^4 * delta_ij into pmbp->padm->u_adm.
//! Called right after the conformal factor psi is solved (XCFC step 3), *before*
//! MHD_C2P -- dyn_grmhd's own per-stage con2prim task, queued to depend on
//! CFC_SolvePsi (see cfc::CFC::QueueCFCTasks()) -- runs: PrimitiveSolverHydro::
//! ConsToPrim (src/eos/primitive_solver_hyd.hpp) reads padm->adm.g_dd directly to
//! invert conserved to primitive variables, so g_dd must already reflect the
//! newly-solved psi by the time that con2prim call happens. delta_psi = psi - 1
//! (see cfc.hpp); psi = delta_psi + 1.0 is reconstructed internally.
void AssembleConformalMetric(MeshBlockPack *pmbp, const DvceArray5D<Real> &delta_psi);

//----------------------------------------------------------------------------------------
//! \fn void AssembleLapseShiftK(MeshBlockPack *pmbp, const DvceArray5D<Real> &delta_psi,
//!            const DvceArray5D<Real> &delta_alpha_psi,
//!            const AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &a_dd,
//!            const AthenaTensor<Real, TensorSymm::NONE, 3, 1> &beta_u)
//! \brief Final step of the XCFC solve: writes vK_dd = psi^-2 * Adual_ij (maximal
//! slicing, K=0), alpha = (alpha*psi)/psi, and beta_u into pmbp->padm->u_adm,
//! matching the layout of adm::ADM::ADM_vars. psi4/g_dd are already set by
//! AssembleConformalMetric() (called earlier, right after step 3) and are not
//! rewritten here. delta_psi/delta_alpha_psi = psi - 1 / alpha*psi - 1 (see
//! cfc.hpp); the physical values are reconstructed internally (+1.0).
void AssembleLapseShiftK(MeshBlockPack *pmbp, const DvceArray5D<Real> &delta_psi,
                          const DvceArray5D<Real> &delta_alpha_psi,
                          const AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &a_dd,
                          const AthenaTensor<Real, TensorSymm::NONE, 3, 1> &beta_u);

}  // namespace cfc

#endif  // CFC_CFC_RECONSTRUCT_HPP_
