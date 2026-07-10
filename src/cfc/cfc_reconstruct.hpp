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
//! vector field from the Shibata (1999) 4-scalar decomposition, and final assembly of
//! the XCFC solution into the ADM variables consumed by dyn_grmhd.
//!
//! Vector/tensor arguments use AthenaTensor views (as in the z4c/adm modules); genuine
//! scalars (eta, psi, alpha_psi) remain plain DvceArray5D<Real>.

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
//!            AthenaTensor<Real, TensorSymm::NONE, 3, 1> &v_u)
//! \brief Shibata (1999) eq. 3.9: V^j = (7/8) P_j - (1/8)(eta,_j + P_k,_j x^k),
//! reconstructing a vector field from the decomposed vector potential P_i (solved
//! first, see mg_cfc_vector_poisson.hpp) and scalar potential eta (solved second,
//! see mg_cfc_scalar_poisson.hpp). Used for both X^i (internal, feeds
//! ComputeADualFromX) and beta^i (written to the ADM shift).
//! p_i: P_i (vector). eta: eta (scalar). v_u: reconstructed vector, e.g. X^i/beta^i.
void ReconstructVectorFromPotentials(MeshBlockPack *pmbp,
                                      const AthenaTensor<Real, TensorSymm::NONE, 3, 1>
                                          &p_i,
                                      const DvceArray5D<Real> &eta,
                                      AthenaTensor<Real, TensorSymm::NONE, 3, 1> &v_u);

//----------------------------------------------------------------------------------------
//! \fn void AssembleADMFromCFC(MeshBlockPack *pmbp, const DvceArray5D<Real> &psi,
//!            const DvceArray5D<Real> &alpha_psi,
//!            const AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &a_dd,
//!            const AthenaTensor<Real, TensorSymm::NONE, 3, 1> &beta_u)
//! \brief Final step of the XCFC solve: writes psi4 = psi^4, g_dd = psi^4 * delta_ij,
//! vK_dd = psi^-2 * Adual_ij (maximal slicing, K=0), alpha = (alpha*psi)/psi, and
//! beta_u into pmbp->padm->u_adm, matching the layout of adm::ADM::ADM_vars.
void AssembleADMFromCFC(MeshBlockPack *pmbp, const DvceArray5D<Real> &psi,
                         const DvceArray5D<Real> &alpha_psi,
                         const AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &a_dd,
                         const AthenaTensor<Real, TensorSymm::NONE, 3, 1> &beta_u);

}  // namespace cfc

#endif  // CFC_CFC_RECONSTRUCT_HPP_
