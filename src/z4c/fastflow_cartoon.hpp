#ifndef Z4C_FASTFLOW_CARTOON_HPP_
#define Z4C_FASTFLOW_CARTOON_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file fastflow_cartoon.hpp
//! \brief 2D Cartoon point evaluation for FastFlow (M2f, chat/0010, D-022).
//!
//! The mesh is the plane mesh x3 = 0 (one cell in x3); physical (x, y, z) =
//! mesh (x1, -x3, x2) (D-017). A point at physical (x, y, z) maps to the plane at
//! (rho, X2) = (sqrt(x^2 + y^2), z). There the 2D Lagrange interpolant of u_adm gives
//! g_ij, K_ij, d_1 g_ij and d_2 g_ij exactly (the derivatives are those of the
//! interpolant, as InterpolateLagrangeDeriv in 3D). The out-of-plane derivative follows
//! from axisymmetry: with xi = X1 d_3 - X3 d_1, L_xi g = 0 at X3 = 0 gives
//!   d_3 g_ij = B_ij(g)/rho,  B_ij(T) = -(d_i0 T_2j - d_i2 T_0j + d_j0 T_i2 - d_j2 T_i0)
//! (mesh indices 0 = x1, 1 = x2, 2 = x3; the sign of xi drops out). B(g) vanishes on the
//! axis only up to interpolation error, so for rho < dx1 the numerator is taken as
//! B(g)(rho) - B(g)(0) from the SAME interpolant: that quotient is the mean x1-derivative
//! of B over [0, rho], so the axis error cancels instead of being amplified by 1/rho
//! (chat/0013). For rho below round-off it is B(d_1 g)(rho), its limit. Then g, K and
//! d g are rotated to the point's azimuth (the Cartoon-fill rotation, c = X1/rho,
//! s = X3/rho) and mapped to physical components.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "utils/inline_interpolator.hpp"

namespace z4c {

// Value (dir = -1) or exact x1 / x2 derivative (dir = 0 / 1) of the 2D interpolant;
// w1 overrides the x1 basis when non-null (used for the axis value on the same stencil).
template <int NGHOST>
KOKKOS_INLINE_FUNCTION
Real CartoonInterp2D(const DvceArray5D<Real> &val, int var, const RegionIndcs &indcs,
                     const IndAndWghts<NGHOST> &w, int dir, const Real *w1o = nullptr) {
  Real ival = 0.0;
  for (int i = 0; i < 2*NGHOST; ++i) {
    const Real w1 = (w1o != nullptr) ? w1o[i] : ((dir == 0) ? w.dwght1[i] : w.wght1[i]);
    for (int j = 0; j < 2*NGHOST; ++j) {
      const Real w2 = (dir == 1) ? w.dwght2[j] : w.wght2[j];
      ival += w1*w2*val(w.ii0, var, indcs.ks, w.ii2 - (NGHOST - j - indcs.js) + 1,
                        w.ii1 - (NGHOST - i - indcs.is) + 1);
    }
  }
  return ival;
}

KOKKOS_INLINE_FUNCTION
void CartoonAxisymBracket(const Real T[3][3], Real B[3][3]) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      Real b = 0.0;
      if (i == 0) b += T[2][j];
      if (i == 2) b -= T[0][j];
      if (j == 0) b += T[i][2];
      if (j == 2) b -= T[i][0];
      B[i][j] = -b;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn bool CartoonFastFlowPoint
//! \brief Physical-frame g (6), K (6) and d_k g (18, k-major, FastFlow order xx xy xz yy
//! yz zz) at physical (x, y, z) from u_adm on the Cartoon plane. Returns false if the
//! mapped point is not on this rank. gind / Kind: u_adm indices in that order.
template <int NGHOST>
KOKKOS_INLINE_FUNCTION
bool CartoonFastFlowPoint(const DvceArray5D<Real> &u_adm, const int gind[6],
                          const int Kind[6], const RegionIndcs &indcs,
                          const DualArray1D<RegionSize> &size, const int nmb,
                          const Real x3plane, const Real x, const Real y, const Real z,
                          Real gp[6], Real Kp[6], Real dgp[18]) {
  const Real X1 = x, X2 = z, X3 = -y;
  const Real rho = Kokkos::sqrt(X1*X1 + X3*X3);
  Real pos[3] = {rho, X2, x3plane};
  auto iw = IndicesAndWeights<NGHOST>(indcs, size, pos, nmb);
  if (!iw.point_exist) return false;
  const Real dx1 = size.d_view(iw.ii0).dx1;

  const int sidx[3][3] = {
    {0, 1, 2}, {1, 3, 4}, {2, 4, 5}
  };
  Real G[3][3], K[3][3], D[3][3][3];
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      const int n = sidx[a][b];
      G[a][b] = CartoonInterp2D<NGHOST>(u_adm, gind[n], indcs, iw, -1);
      K[a][b] = CartoonInterp2D<NGHOST>(u_adm, Kind[n], indcs, iw, -1);
      D[0][a][b] = CartoonInterp2D<NGHOST>(u_adm, gind[n], indcs, iw, 0);
      D[1][a][b] = CartoonInterp2D<NGHOST>(u_adm, gind[n], indcs, iw, 1);
    }
  }

  // d_3 g on the plane from axisymmetry
  Real B[3][3];
  if (rho >= dx1) {
    CartoonAxisymBracket(G, B);
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) D[2][a][b] = B[a][b]/rho;
  } else if (rho > 1.0e-10*dx1) {
    // near the axis: (B(rho) - B(0))/rho on the same stencil. x1 basis at X1 = 0:
    Real w10[2*NGHOST], xn[2*NGHOST];
    const auto mb = size.d_view(iw.ii0);
    for (int i = 0; i < 2*NGHOST; ++i) {
      xn[i] = CellCenterX(iw.ii1 - NGHOST + i + 1, indcs.nx1, mb.x1min, mb.x1max);
    }
    for (int i = 0; i < 2*NGHOST; ++i) {
      w10[i] = 1.0;
      for (int j = 0; j < 2*NGHOST; ++j) {
        if (j != i) w10[i] *= (0.0 - xn[j])/(xn[i] - xn[j]);
      }
    }
    Real G0[3][3], B0[3][3];
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        G0[a][b] = CartoonInterp2D<NGHOST>(u_adm, gind[sidx[a][b]], indcs, iw, -1, w10);
    CartoonAxisymBracket(G, B);
    CartoonAxisymBracket(G0, B0);
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) D[2][a][b] = (B[a][b] - B0[a][b])/rho;
  } else {
    CartoonAxisymBracket(D[0], B);
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) D[2][a][b] = B[a][b];
  }

  // M = L R: R rotates the plane data to the azimuth about mesh x2 (the Cartoon-fill
  // rotation R = [[c, -s], [s, c]] in (x1, x3)); L maps mesh to physical components.
  const Real c = (rho > 0.0) ? X1/rho : 1.0;
  const Real s = (rho > 0.0) ? X3/rho : 0.0;
  const Real R[3][3] = {
    {c, 0.0, -s}, {0.0, 1.0, 0.0}, {s, 0.0, c}
  };
  const Real L[3][3] = {
    {1.0, 0.0, 0.0}, {0.0, 0.0, -1.0}, {0.0, 1.0, 0.0}
  };
  Real M[3][3];
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) {
      M[a][b] = 0.0;
      for (int e = 0; e < 3; ++e) M[a][b] += L[a][e]*R[e][b];
    }
  const int pa[6] = {0, 0, 0, 1, 1, 2}, pb[6] = {0, 1, 2, 1, 2, 2};
  for (int n = 0; n < 6; ++n) {
    const int i = pa[n], j = pb[n];
    Real gv = 0.0, kv = 0.0;
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) {
        const Real mm = M[i][a]*M[j][b];
        gv += mm*G[a][b];
        kv += mm*K[a][b];
      }
    gp[n] = gv;
    Kp[n] = kv;
    for (int k = 0; k < 3; ++k) {
      Real dv = 0.0;
      for (int e = 0; e < 3; ++e)
        for (int a = 0; a < 3; ++a)
          for (int b = 0; b < 3; ++b) dv += M[k][e]*M[i][a]*M[j][b]*D[e][a][b];
      dgp[6*k + n] = dv;
    }
  }
  return true;
}

}  // namespace z4c

#endif  // Z4C_FASTFLOW_CARTOON_HPP_
