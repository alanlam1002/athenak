//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cfc_reconstruct.cpp
//! \brief implementation of free-function Kokkos kernels declared in cfc_reconstruct.hpp

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "utils/finite_diff.hpp"
#include "cfc_reconstruct.hpp"

namespace cfc {

namespace {

//----------------------------------------------------------------------------------------
//! \fn void ComputeADualFromXImpl<NGHOST>(...)
//! \brief Gmunu eq. 76: Adual^ij = D^i X^j + D^j X^i - (2/3) D_k X^k f^ij. Since
//! addition is commutative, symmetrizing the two raw derivative terms gives the same
//! result regardless of which index of dX[][] is "component" vs. "direction".

template <int NGHOST>
void ComputeADualFromXImpl(MeshBlockPack *pmbp,
                            const AthenaTensor<Real, TensorSymm::NONE, 3, 1> &x_u,
                            AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &a_dd) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;

  par_for("cfc_adual", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real idx[] = {1.0/size.d_view(m).dx1, 1.0/size.d_view(m).dx2,
                  1.0/size.d_view(m).dx3};
    Real dX[3][3];
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        dX[a][b] = Dx<NGHOST>(b, idx, x_u, m, a, k, j, i);
      }
    }
    Real trace = dX[0][0] + dX[1][1] + dX[2][2];
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        a_dd(m,a,b,k,j,i) = dX[a][b] + dX[b][a] - (a == b ? (2./3.)*trace : 0.0);
      }
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn void ReconstructVectorFromPotentialsImpl<NGHOST>(...)
//! \brief Shibata (1999) eq. 3.9: V^j = (7/8) P_j - (1/8)(eta,_j + P_k,_j x^k).
//! eta is a plain DvceArray5D<Real> (single component), so it's locally shallow-sliced
//! into a rank-0 AthenaTensor to reuse the generic Dx<NGHOST> scalar overload.

template <int NGHOST>
void ReconstructVectorFromPotentialsImpl(MeshBlockPack *pmbp,
    const AthenaTensor<Real, TensorSymm::NONE, 3, 1> &p_i,
    const DvceArray5D<Real> &eta,
    AthenaTensor<Real, TensorSymm::NONE, 3, 1> &v_u) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;

  AthenaTensor<Real, TensorSymm::NONE, 3, 0> eta_view;
  eta_view.InitWithShallowSlice(eta, 0);

  par_for("cfc_vec_reconstruct", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real idx[] = {1.0/size.d_view(m).dx1, 1.0/size.d_view(m).dx2,
                  1.0/size.d_view(m).dx3};
    Real &x1min = size.d_view(m).x1min; Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
    Real &x2min = size.d_view(m).x2min; Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
    Real &x3min = size.d_view(m).x3min; Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
    Real xk[3] = {x1v, x2v, x3v};

    for (int jdir = 0; jdir < 3; ++jdir) {
      Real deta_ddir = Dx<NGHOST>(jdir, idx, eta_view, m, k, j, i);
      Real sum = 0.0;
      for (int kdir = 0; kdir < 3; ++kdir) {
        sum += Dx<NGHOST>(jdir, idx, p_i, m, kdir, k, j, i) * xk[kdir];
      }
      v_u(m,jdir,k,j,i) = 0.875*p_i(m,jdir,k,j,i) - 0.125*(deta_ddir + sum);
    }
  });
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ComputeADualFromX(...)

void ComputeADualFromX(MeshBlockPack *pmbp,
                        const AthenaTensor<Real, TensorSymm::NONE, 3, 1> &x_u,
                        AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &a_dd) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  switch (indcs.ng) {
    case 2: ComputeADualFromXImpl<2>(pmbp, x_u, a_dd); break;
    case 3: ComputeADualFromXImpl<3>(pmbp, x_u, a_dd); break;
    case 4: ComputeADualFromXImpl<4>(pmbp, x_u, a_dd); break;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ReconstructVectorFromPotentials(...)

void ReconstructVectorFromPotentials(MeshBlockPack *pmbp,
                                      const AthenaTensor<Real, TensorSymm::NONE, 3, 1>
                                          &p_i,
                                      const DvceArray5D<Real> &eta,
                                      AthenaTensor<Real, TensorSymm::NONE, 3, 1> &v_u) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  switch (indcs.ng) {
    case 2: ReconstructVectorFromPotentialsImpl<2>(pmbp, p_i, eta, v_u); break;
    case 3: ReconstructVectorFromPotentialsImpl<3>(pmbp, p_i, eta, v_u); break;
    case 4: ReconstructVectorFromPotentialsImpl<4>(pmbp, p_i, eta, v_u); break;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void AssembleConformalMetric(...)

void AssembleConformalMetric(MeshBlockPack *pmbp, const DvceArray5D<Real> &psi) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  auto &adm = pmbp->padm->adm;

  par_for("cfc_assemble_conformal_metric", DevExeSpace(),
          0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real psi_val = psi(m,0,k,j,i);
    Real psi4 = psi_val*psi_val*psi_val*psi_val;
    adm.psi4(m,k,j,i) = psi4;
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        adm.g_dd(m,a,b,k,j,i) = (a == b ? psi4 : 0.0);
      }
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn void AssembleLapseShiftK(...)

void AssembleLapseShiftK(MeshBlockPack *pmbp, const DvceArray5D<Real> &psi,
                          const DvceArray5D<Real> &alpha_psi,
                          const AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &a_dd,
                          const AthenaTensor<Real, TensorSymm::NONE, 3, 1> &beta_u) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  auto &adm = pmbp->padm->adm;

  par_for("cfc_assemble_lapse_shift_k", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real psi_val = psi(m,0,k,j,i);
    Real ipsi2 = 1.0/(psi_val*psi_val);
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        adm.vK_dd(m,a,b,k,j,i) = a_dd(m,a,b,k,j,i)*ipsi2;
      }
    }
    adm.alpha(m,k,j,i) = alpha_psi(m,0,k,j,i)/psi_val;
    for (int a = 0; a < 3; ++a) {
      adm.beta_u(m,a,k,j,i) = beta_u(m,a,k,j,i);
    }
  });
}

}  // namespace cfc
