//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cfc_reconstruct.cpp
//! \brief implementation of free-function Kokkos kernels declared in cfc_reconstruct.hpp

#include "athena.hpp"
#include "mesh/meshblock_pack.hpp"
#include "coordinates/adm.hpp"
#include "cfc_reconstruct.hpp"

namespace cfc {

//----------------------------------------------------------------------------------------
//! \fn void ComputeADualFromX(...)

void ComputeADualFromX(MeshBlockPack *pmbp, const DvceArray5D<Real> &x_u,
                        DvceArray5D<Real> &a_dd) {
  // TODO(cfc): finite-difference D_i X_j on the finest grid and combine per eq. 76.
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ReconstructVectorFromPotentials(...)

void ReconstructVectorFromPotentials(MeshBlockPack *pmbp,
                                      const DvceArray5D<Real> &p_eta,
                                      DvceArray5D<Real> &v_u) {
  // TODO(cfc): evaluate V^j = (7/8) P_j - (1/8)(eta,_j + P_k,_j x^k) per Shibata eq. 3.9.
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void AssembleADMFromCFC(...)

void AssembleADMFromCFC(MeshBlockPack *pmbp, const DvceArray5D<Real> &psi,
                         const DvceArray5D<Real> &alpha_psi,
                         const DvceArray5D<Real> &a_dd,
                         const DvceArray5D<Real> &beta_u) {
  // TODO(cfc): write psi4, g_dd, vK_dd, alpha, beta_u into pmbp->padm->u_adm.
  return;
}

}  // namespace cfc
