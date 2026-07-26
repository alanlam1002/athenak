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
//! vector field from the Shibata (1999) 4-scalar decomposition, and assembly of the
//! XCFC solution into pmbp->padm->u_adm.
//!
//! Vector/tensor arguments use AthenaTensor views (as in z4c/adm); scalars remain plain
//! DvceArray5D<Real>. AssembleConformalMetric/AssembleLapseShiftK take
//! delta_psi = psi - 1 / delta_alpha_psi = alpha*psi - 1 (see cfc::CFC's doc comment in
//! cfc.hpp), reconstructing the physical value internally (+1.0).

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
void ComputeADualFromX(MeshBlockPack *pmbp,
                        const AthenaTensor<Real, TensorSymm::NONE, 3, 1> &x_u,
                        AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &a_dd);

//----------------------------------------------------------------------------------------
//! \fn void ReconstructVectorFromPotentials(MeshBlockPack *pmbp,
//!            const AthenaTensor<Real, TensorSymm::NONE, 3, 1> &p_i,
//!            const DvceArray5D<Real> &eta,
//!            AthenaTensor<Real, TensorSymm::NONE, 3, 1> &v_u, int eta_chan = 0)
//! \brief Shibata (1999) eq. 3.9: V^j = (7/8) P_j - (1/8)(eta,_j + P_k,_j x^k) --
//! reconstructs a vector field from the packed vector potential P_i (channels 0-2) and
//! scalar potential eta (channel eta_chan), solved together by one
//! MGCFCVectorPoissonDriver (see mg_cfc_vector_poisson.hpp). Used for both X^i
//! (feeds ComputeADualFromX) and beta^i (written to the ADM shift).
void ReconstructVectorFromPotentials(MeshBlockPack *pmbp,
                                      const AthenaTensor<Real, TensorSymm::NONE, 3, 1>
                                          &p_i,
                                      const DvceArray5D<Real> &eta,
                                      AthenaTensor<Real, TensorSymm::NONE, 3, 1> &v_u,
                                      int eta_chan = 0);

//----------------------------------------------------------------------------------------
//! \fn void AssembleConformalMetric(MeshBlockPack *pmbp,
//!                                   const DvceArray5D<Real> &delta_psi)
//! \brief writes psi4 = psi^4 and g_dd = psi^4*delta_ij into pmbp->padm->u_adm. Must
//! run before MHD_C2P (dyn_grmhd's per-stage con2prim, queued to depend on
//! CFC_SolvePsi): PrimitiveSolverHydro::ConsToPrim reads padm->adm.g_dd directly to
//! invert conserved to primitive variables.
void AssembleConformalMetric(MeshBlockPack *pmbp, const DvceArray5D<Real> &delta_psi);

//----------------------------------------------------------------------------------------
//! \fn void AssembleLapseShiftK(MeshBlockPack *pmbp, const DvceArray5D<Real> &delta_psi,
//!            const DvceArray5D<Real> &delta_alpha_psi,
//!            const AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &a_dd,
//!            const AthenaTensor<Real, TensorSymm::NONE, 3, 1> &beta_u)
//! \brief final XCFC step: writes vK_dd = psi^-2*Adual_ij (maximal slicing, K=0),
//! alpha = (alpha*psi)/psi, and beta_u into pmbp->padm->u_adm. psi4/g_dd are already
//! set by AssembleConformalMetric() and not rewritten here.
void AssembleLapseShiftK(MeshBlockPack *pmbp, const DvceArray5D<Real> &delta_psi,
                          const DvceArray5D<Real> &delta_alpha_psi,
                          const AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &a_dd,
                          const AthenaTensor<Real, TensorSymm::NONE, 3, 1> &beta_u);

}  // namespace cfc

#endif  // CFC_CFC_RECONSTRUCT_HPP_
