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
                            AthenaTensor<Real, TensorSymm::NONE, 3, 1> &grad_ap6_0) {
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

    // Bare gradient D_j(alpha0*psi0^-6) = (alpha0*psi0^-6)*[dalpha0/alpha0 -
    // 6*dpsi0/psi0]*(x_j/r) -- Sec 3.9's closed-form shift-source ingredient. Stored
    // uncontracted (not dotted with Ahat0^ij here): cfc.cpp::BuildShiftSourceImpl dots
    // it against the matter-only Ahats^ij = Ahat^ij-Ahat0^ij at the point of use, since
    // the Ahat0-contracted combination cancels out of the final residual (Sec 3.9).
    Real psi0_inv6 = 1.0/(psi0*psi0*psi0*psi0*psi0*psi0);
    // alpha0 CANCELS ANALYTICALLY, and forming the quotient is fatal:
    //   ap6*dfac = (alpha0*psi0^-6) * (dalpha0/alpha0 - 6*dpsi0/psi0)
    //            =  psi0^-6 * (dalpha0 - 6*alpha0*dpsi0/psi0)
    // The original wrote the first form, which evaluates dalpha0/alpha0. At the
    // maximal-slicing trumpet THROAT (rrs = r_sch/m_bh = 3/2) BOTH vanish --
    // alpha0^2 = 1 - 2/rrs + 1.6875/rrs^4 has a DOUBLE ROOT there, and
    // dalpha0 ~ (1 - 3.375/rrs^3) vanishes at the same point -- so the quotient is
    // 0/0 = NaN. Because it is a double root, alpha0^2 also loses all significance to
    // catastrophic cancellation nearby: sampling rrs within 1e-7 of 3/2, 2.8% of
    // positions give exactly 0 and a further 2.4% give a NEGATIVE alpha0^2.
    // This was the origin of the NANS_IN_CONS cascade -- grad_ap6_0 is consumed ONLY
    // by BuildShiftSource, so the corruption appeared as beta^i coming back 100% NaN
    // with psi/alpha/K_dd and every primitive still clean. See
    // src/cfc/NANCASCADE_HANDOFF.md section 15.
    // The second form never divides by alpha0; psi0 = sqrt(r_sch/r) > 0 always.
    Real amp = psi0_inv6*(dalpha0 - 6.0*alpha0*dpsi0/psi0);
    grad_ap6_0(m,0,k,j,i) = amp*x1v/r;
    grad_ap6_0(m,1,k,j,i) = amp*x2v/r;
    grad_ap6_0(m,2,k,j,i) = amp*x3v/r;
  });
}

}  // namespace cfc
