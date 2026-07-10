//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cfc_reconstruct.cpp
//! \brief implementation of free-function Kokkos kernels declared in cfc_reconstruct.hpp

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "mesh/meshblock_pack.hpp"
#include "coordinates/adm.hpp"
#include "cfc_reconstruct.hpp"

namespace cfc {

//----------------------------------------------------------------------------------------
//! \fn void ComputeADualFromX(...)

void ComputeADualFromX(MeshBlockPack *pmbp,
                        const AthenaTensor<Real, TensorSymm::NONE, 3, 1> &x_u,
                        AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &a_dd) {
  // TODO(cfc): finite-difference D_i X_j on the finest grid and combine per eq. 76.
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ReconstructVectorFromPotentials(...)

void ReconstructVectorFromPotentials(MeshBlockPack *pmbp,
                                      const AthenaTensor<Real, TensorSymm::NONE, 3, 1>
                                          &p_i,
                                      const DvceArray5D<Real> &eta,
                                      AthenaTensor<Real, TensorSymm::NONE, 3, 1> &v_u) {
  // TODO(cfc): evaluate V^j = (7/8) P_j - (1/8)(eta,_j + P_k,_j x^k) per Shibata eq.
  // 3.9, finite-differencing eta and P_k on the finest grid.
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void AssembleConformalMetric(...)

void AssembleConformalMetric(MeshBlockPack *pmbp, const DvceArray5D<Real> &psi) {
  // TODO(cfc): write psi4 = psi^4, g_dd = psi^4 * delta_ij into pmbp->padm->u_adm.
  // Must run before RescaleMatterSources()'s con2prim call (padm->adm.g_dd is read
  // directly by PrimitiveSolverHydro::ConsToPrim).
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void AssembleLapseShiftK(...)

void AssembleLapseShiftK(MeshBlockPack *pmbp, const DvceArray5D<Real> &psi,
                          const DvceArray5D<Real> &alpha_psi,
                          const AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &a_dd,
                          const AthenaTensor<Real, TensorSymm::NONE, 3, 1> &beta_u) {
  // TODO(cfc): write vK_dd = psi^-2 * Adual_ij, alpha = (alpha*psi)/psi, beta_u into
  // pmbp->padm->u_adm. psi4/g_dd already written by AssembleConformalMetric().
  return;
}

}  // namespace cfc
