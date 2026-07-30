//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cfc_puncture.cpp
//! \brief implementation of FillPunctureBackground (see cfc_puncture.hpp)

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "coordinates/cell_locations.hpp"
#include "cfc_puncture.hpp"

namespace cfc {

void FillPunctureBackground(MeshBlockPack *pmbp, Real m_bh,
                            DvceArray5D<Real> &u_psi0, DvceArray5D<Real> &u_alpha0_psi0,
                            AthenaTensor<Real, TensorSymm::NONE, 3, 1> &beta0_u,
                            AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &a0_dd,
                            DvceArray5D<Real> &a0_sq,
                            AthenaTensor<Real, TensorSymm::NONE, 3, 1> &s0_beta_u) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int isg = is-indcs.ng; int ieg = ie+indcs.ng;
  int jsg = js-indcs.ng; int jeg = je+indcs.ng;
  int ksg = ks-indcs.ng; int keg = ke+indcs.ng;
  int nmb = pmbp->nmb_thispack;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;

  par_for("cfc_fill_puncture_background", DevExeSpace(), 0, nmb-1, ksg, keg, jsg, jeg,
  isg, ieg,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real x2v = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real x3v = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    Real r = sqrt(x1v*x1v + x2v*x2v + x3v*x3v);

    Real rho = TrumpetIsoToAreal(r/m_bh);
    Real r_sch = m_bh*rho;

    Real psi0, alpha0, beta0[3], Aij0[6], a2, dpsi0, dalpha0;
    TrumpetBackground(m_bh, x1v, x2v, x3v, r_sch, &psi0, &alpha0, beta0, Aij0, &a2,
                      &dpsi0, &dalpha0);

    u_psi0(m,0,k,j,i) = psi0;
    u_alpha0_psi0(m,0,k,j,i) = alpha0*psi0;
    a0_sq(m,0,k,j,i) = a2;
    for (int a = 0; a < 3; ++a) { beta0_u(m,a,k,j,i) = beta0[a]; }
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        // Aij0 packed (xx,yy,zz,xy,xz,yz); map (a,b) -> that same packed index.
        int idx = (a == b) ? a : (a + b + 2);
        a0_dd(m,a,b,k,j,i) = Aij0[idx];
      }
    }

    // s0_beta^i = 2*Ahat0^ij*D_j(alpha0*psi0^-6), D_j(alpha0*psi0^-6) =
    // (alpha0*psi0^-6)*[dalpha0/alpha0 - 6*dpsi0/psi0]*(x_j/r) -- Sec 3.9.
    Real psi0_inv6 = 1.0/(psi0*psi0*psi0*psi0*psi0*psi0);
    Real ap6 = alpha0*psi0_inv6;
    Real dfac = dalpha0/alpha0 - 6.0*dpsi0/psi0;
    Real grad[3] = {ap6*dfac*x1v/r, ap6*dfac*x2v/r, ap6*dfac*x3v/r};
    for (int a = 0; a < 3; ++a) {
      Real contraction = 0.0;
      for (int b = 0; b < 3; ++b) {
        contraction += a0_dd(m,a,b,k,j,i)*grad[b];
      }
      s0_beta_u(m,a,k,j,i) = 2.0*contraction;
    }
  });
}

}  // namespace cfc
