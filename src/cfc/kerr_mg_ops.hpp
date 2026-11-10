#ifndef CFC_KERR_MG_OPS_HPP_
#define CFC_KERR_MG_OPS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file kerr_mg_ops.hpp
//! \brief W3a (K-003, Option A with Rule R; kerr_methods.tex sec. 5.1, 5.3): the Kerr-only
//! building blocks of the variable-coefficient multigrid drivers. Nothing in src/multigrid/
//! changes; every function here acts on a Multigrid level's own arrays.
//!
//!  - Face coefficients: gt^ij at the LOWER face of each cell, per direction d,
//!    coeff(m, kFace + 6 d + s) (s = xx xy xz yy yz zz), evaluated pointwise at the face
//!    centre from the W1 evaluators (or a flat/synthetic metric for tests), never averaged.
//!    The upper face of the last interior cell is the lower face of the first ghost, so
//!    every cell including the ghosts is filled.
//!  - Scalar operator (eq. G3, flux form, det gt = 1): h^2 Dt^2 u = sum_d [hF^d(+) - hF^d(-)],
//!      hF^x_{i-1/2} = gxx (u_i - u_{i-1})
//!                   + gxy [(u_{i,j+1} - u_{i,j-1}) + (u_{i-1,j+1} - u_{i-1,j-1})]/4
//!                   + gxz [(u_{i,k+1} - u_{i,k-1}) + (u_{i-1,k+1} - u_{i-1,k-1})]/4,
//!    a 19-point stencil that reads face and EDGE ghosts and never a corner ghost.
//!  - Edge flags: per block, bit q (0..11) of an int is set when edge q is UNSAFE. Edge q
//!    runs along axis e = q/4; its transverse axes are a = (e+1)%3, b = (e+2)%3 at the sides
//!    sa = (q&1) ? + : -, sb = (q&2) ? + : -. A block edge is same-level (safe) iff its
//!    neighbour slot 0 holds a block (gid >= 0) at the block's own level and slot 1 is empty
//!    or also at that level (a same-level neighbour fills slot 0 only, meshblock.cpp:150ff).
//!  - Rule R (eq. 20): every unsafe edge ghost, first ghost layer, interior range along the
//!    edge, is overwritten by
//!      g = u(-1,0) + u(0,-1) - u(0,0) + [u(1,1) - u(1,0) - u(0,1) + u(0,0)]
//!    in the (a, b) plane from the block's corner cell (0,0); the bracket is dropped (the
//!    bilinear fill B) where the block has a single cell along a or b. It reads face ghosts
//!    and interior cells only.

#include <cstdint>
#include <vector>

#include <Kokkos_BitManipulation.hpp>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "kerr_background.hpp"

namespace cfc {
namespace kmg {

inline constexpr int kFace = 0;    // + 6 d + s
inline constexpr int kNFace = 18;
inline constexpr int kChC = 18;      // the linear test equation's c (cell centre)
inline constexpr int kChS = 19;      // its right-hand side s (cell centre)
inline constexpr int kNCoeffScalar = 20;

// W3b (eq:cpsi, eq:cchi): the channel layout of the psi and chi correction equations.
// Background channels (B) are evaluated pointwise on every level, root and octet, never
// restricted; matter channels (M) are formed on the finest level and restricted
// (kerr_methods.tex sec. 4.2, FAS consistency).
inline constexpr int kChR8   = 18;   // B  Rt/8
inline constexpr int kChM    = 19;   // M  psi: E*; chi: E* + 2S*
inline constexpr int kChDA2  = 20;   // M  Ahat^2 - Ahat0^2
inline constexpr int kChPsi0 = 21;   // B  psi0
inline constexpr int kChA02  = 22;   // B  Ahat0^2 = Ahat0^ij Ahat0_ij
inline constexpr int kChDPsi = 23;   // M  delta psi (chi only; psi is solved first)
inline constexpr int kChChi0 = 24;   // B  chi0 = alpha0 psi0
inline constexpr int kNCoeffKerr = 25;

//! The equation solved by a driver: 0 the linear test equation (c, s); 1 the psi
//! correction (eq:cpsi); 2 the lapse correction (eq:cchi).
inline constexpr int kKindLinear = 0;
inline constexpr int kKindPsi = 1;
inline constexpr int kKindChi = 2;

KOKKOS_INLINE_FUNCTION bool IsBgChannel(int q) {
  return q < kNFace || q == kChR8 || q == kChPsi0 || q == kChA02 || q == kChChi0;
}

enum class EdgeFill : int { rule_r = 0, bilinear = 1, off = 2 };

KOKKOS_INLINE_FUNCTION void EdgeAxes(int q, int &e, int &a, int &b, int &sa, int &sb) {
  e = q >> 2;
  a = (e + 1) % 3;
  b = (e + 2) % 3;
  sa = (q & 1) ? 1 : -1;
  sb = (q & 2) ? 1 : -1;
}

//! exponent all ones = Inf or NaN; a bit test so that fast-math cannot fold it away
KOKKOS_INLINE_FUNCTION bool NonFinite(Real x) {
#if SINGLE_PRECISION_ENABLED
  const std::uint32_t b = Kokkos::bit_cast<std::uint32_t>(x);
  return (b & 0x7f800000U) == 0x7f800000U;
#else
  const std::uint64_t b = Kokkos::bit_cast<std::uint64_t>(x);
  return (b & 0x7ff0000000000000ULL) == 0x7ff0000000000000ULL;
#endif
}

//! \struct CoefSource
//! \brief Where the conformal metric comes from: 0 flat, 1 synthetic det = 1 metric
//! (gt = M det(M)^(-1/3), M = 1 + eps S(x), off-diagonals O(eps); E2), 2 the Kerr table
//! (W1, about the hole at xbh, mass m_bh; exact = the spectral evaluation instead of F2).
struct CoefSource {
  int mode = 0;
  KerrTable tab;
  Real m_bh = 1.0;
  Real xbh[3] = {0.0, 0.0, 0.0};
  bool exact = false;
  Real eps = 0.2, kw = 1.0;
};

//! \fn SyntheticM(cs, x, y, z, M, dM)
//! \brief M_ab = delta_ab + eps S_ab(x) and d_k M_ab (packed xx xy xz yy yz zz).
KOKKOS_INLINE_FUNCTION
void SyntheticM(const CoefSource &cs, Real x, Real y, Real z, Real M[6], Real dM[3][6]) {
  const Real e = cs.eps, k = cs.kw;
  // S_xx, S_yy, S_zz, S_xy, S_xz, S_yz, each a smooth bounded function of order 1
  const Real sxx = 0.5*Kokkos::sin(k*x + 0.3), cxx = 0.5*k*Kokkos::cos(k*x + 0.3);
  const Real syy = 0.5*Kokkos::cos(k*y + 0.7), cyy = -0.5*k*Kokkos::sin(k*y + 0.7);
  const Real szz = 0.5*Kokkos::sin(k*z + 1.1), czz = 0.5*k*Kokkos::cos(k*z + 1.1);
  const Real axy = k*(x - y) + 0.2, sxy = Kokkos::cos(axy), dxy = -k*Kokkos::sin(axy);
  const Real axz = k*(x + z) + 0.5, sxz = Kokkos::sin(axz), dxz = k*Kokkos::cos(axz);
  const Real ayz = k*(y + 0.5*z) - 0.4, syz = Kokkos::cos(ayz), dyz = -k*Kokkos::sin(ayz);
  M[0] = 1.0 + e*sxx; M[1] = e*sxy; M[2] = e*sxz;
  M[3] = 1.0 + e*syy; M[4] = e*syz; M[5] = 1.0 + e*szz;
  for (int d = 0; d < 3; ++d) {
    for (int q = 0; q < 6; ++q) { dM[d][q] = 0.0; }
  }
  dM[0][0] = e*cxx; dM[1][3] = e*cyy; dM[2][5] = e*czz;
  dM[0][1] = e*dxy; dM[1][1] = -e*dxy;
  dM[0][2] = e*dxz; dM[2][2] = e*dxz;
  dM[1][4] = e*dyz; dM[2][4] = 0.5*e*dyz;
}

//! \fn Inv3Sym(g, gi) -> det
KOKKOS_INLINE_FUNCTION Real Inv3Sym(const Real g[6], Real gi[6]) {
  const Real c00 = g[3]*g[5] - g[4]*g[4];
  const Real c01 = g[2]*g[4] - g[1]*g[5];
  const Real c02 = g[1]*g[4] - g[2]*g[3];
  const Real det = g[0]*c00 + g[1]*c01 + g[2]*c02;
  const Real id = 1.0/det;
  gi[0] = c00*id;
  gi[1] = c01*id;
  gi[2] = c02*id;
  gi[3] = (g[0]*g[5] - g[2]*g[2])*id;
  gi[4] = (g[1]*g[2] - g[0]*g[4])*id;
  gi[5] = (g[0]*g[3] - g[1]*g[1])*id;
  return det;
}

//! \fn PointMetric(cs, x, y, z, gt, gtu, dgt)
//! \brief gt_ij, gt^ij and d_k gt_ij at the point (x, y, z) (mesh coordinates), all
//! analytic: flat; synthetic (d_k gt = e (d_k M - M tr(M^-1 d_k M)/3), e = det M^(-1/3));
//! Kerr (KerrNodeCartesian + KerrConformal, spectral derivatives).
KOKKOS_INLINE_FUNCTION
void PointMetric(const CoefSource &cs, Real x, Real y, Real z, Real gt[6], Real gtu[6],
                 Real dgt[3][6]) {
  if (cs.mode == 2) {
    Real v[kKnNV];
    KerrNodeCartesian(cs.tab, cs.m_bh, x - cs.xbh[0], y - cs.xbh[1], z - cs.xbh[2], v,
                      cs.exact);
    KerrBgPoint b;
    KerrConformal(v, b);
    for (int q = 0; q < 6; ++q) {
      gt[q] = b.gt[q];
      gtu[q] = b.gtu[q];
      for (int d = 0; d < 3; ++d) { dgt[d][q] = b.dgt[d][q]; }
    }
    return;
  }
  if (cs.mode == 1) {
    Real M[6], dM[3][6], Mi[6];
    SyntheticM(cs, x, y, z, M, dM);
    const Real det = Inv3Sym(M, Mi);
    const Real e = Kokkos::cbrt(1.0/det);
    for (int q = 0; q < 6; ++q) {
      gt[q] = e*M[q];
      gtu[q] = Mi[q]/e;
    }
    for (int d = 0; d < 3; ++d) {
      // tr(M^-1 dM) with packed symmetric storage: diagonal once, off-diagonal twice
      const Real tr = Mi[0]*dM[d][0] + Mi[3]*dM[d][3] + Mi[5]*dM[d][5]
                    + 2.0*(Mi[1]*dM[d][1] + Mi[2]*dM[d][2] + Mi[4]*dM[d][4]);
      for (int q = 0; q < 6; ++q) { dgt[d][q] = e*(dM[d][q] - M[q]*tr/3.0); }
    }
    return;
  }
  for (int q = 0; q < 6; ++q) {
    gt[q] = (q == 0 || q == 3 || q == 5) ? 1.0 : 0.0;
    gtu[q] = gt[q];
    for (int d = 0; d < 3; ++d) { dgt[d][q] = 0.0; }
  }
}

//! \fn PointDgtu(gtu, dgt, dgtu)
//! \brief d_k gt^ij = -gt^ia gt^jb d_k gt_ab (analytic, from the analytic d_k gt_ab).
KOKKOS_INLINE_FUNCTION
void PointDgtu(const Real gtu[6], const Real dgt[3][6], Real dgtu[3][6]) {
  using kerr_impl::Sym;
  using kerr_impl::IdxI;
  using kerr_impl::IdxJ;
  for (int d = 0; d < 3; ++d) {
    for (int q = 0; q < 6; ++q) {
      const int i = IdxI(q), j = IdxJ(q);
      Real s = 0.0;
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          s += gtu[Sym(i, a)]*gtu[Sym(j, b)]*dgt[d][Sym(a, b)];
        }
      }
      dgtu[d][q] = -s;
    }
  }
}

//! \struct Manuf
//! \brief A manufactured solution u* for the linear test equation
//!   A u = -Dt^2 u + c u = s,  Dt^2 u = d_i(gt^ij d_j u)  (det gt = 1),
//! with s evaluated analytically (the flux form, d_i gt^ij from the analytic d gt):
//!   mode 0: amp sin(pi X) sin(pi Y) sin(pi Z), X = (x - xmin)/L (zero on the box faces);
//!   mode 1: amp/sqrt(|x - x0|^2 + w^2) (a 1/r tail, for the Robin outer boundary);
//!   mode 2: amp exp(-|x - x0|^2/w^2).
struct Manuf {
  int mode = 0;
  Real amp = 1.0, w = 1.0, c = 0.0;
  Real x0[3] = {0.0, 0.0, 0.0};
  Real xmin[3] = {0.0, 0.0, 0.0}, len[3] = {1.0, 1.0, 1.0};
};

//! u*, d_i u*, d_ij u* (packed)
KOKKOS_INLINE_FUNCTION
void ManufEval(const Manuf &mf, Real x, Real y, Real z, Real &u, Real du[3], Real ddu[6]) {
  using kerr_impl::Sym;
  const Real X[3] = {x, y, z};
  if (mf.mode == 0) {
    const Real pi = 3.14159265358979323846;
    Real s[3], c[3], k[3];
    for (int d = 0; d < 3; ++d) {
      k[d] = pi/mf.len[d];
      s[d] = Kokkos::sin(k[d]*(X[d] - mf.xmin[d]));
      c[d] = Kokkos::cos(k[d]*(X[d] - mf.xmin[d]));
    }
    u = mf.amp*s[0]*s[1]*s[2];
    du[0] = mf.amp*k[0]*c[0]*s[1]*s[2];
    du[1] = mf.amp*k[1]*s[0]*c[1]*s[2];
    du[2] = mf.amp*k[2]*s[0]*s[1]*c[2];
    ddu[Sym(0, 0)] = -k[0]*k[0]*u;
    ddu[Sym(1, 1)] = -k[1]*k[1]*u;
    ddu[Sym(2, 2)] = -k[2]*k[2]*u;
    ddu[Sym(0, 1)] = mf.amp*k[0]*k[1]*c[0]*c[1]*s[2];
    ddu[Sym(0, 2)] = mf.amp*k[0]*k[2]*c[0]*s[1]*c[2];
    ddu[Sym(1, 2)] = mf.amp*k[1]*k[2]*s[0]*c[1]*c[2];
    return;
  }
  const Real r[3] = {x - mf.x0[0], y - mf.x0[1], z - mf.x0[2]};
  const Real r2 = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
  if (mf.mode == 1) {
    const Real q = r2 + mf.w*mf.w;
    const Real iq = 1.0/q, sq = Kokkos::sqrt(iq);
    u = mf.amp*sq;
    for (int i = 0; i < 3; ++i) { du[i] = -mf.amp*r[i]*sq*iq; }
    for (int i = 0; i < 3; ++i) {
      for (int j = i; j < 3; ++j) {
        ddu[Sym(i, j)] = mf.amp*sq*iq*(3.0*r[i]*r[j]*iq - ((i == j) ? 1.0 : 0.0));
      }
    }
    return;
  }
  const Real iw2 = 1.0/(mf.w*mf.w);
  u = mf.amp*Kokkos::exp(-r2*iw2);
  for (int i = 0; i < 3; ++i) { du[i] = -2.0*r[i]*iw2*u; }
  for (int i = 0; i < 3; ++i) {
    for (int j = i; j < 3; ++j) {
      ddu[Sym(i, j)] = (4.0*r[i]*r[j]*iw2*iw2 - ((i == j) ? 2.0*iw2 : 0.0))*u;
    }
  }
}

//! s = -[gt^ij d_ij u* + (d_i gt^ij) d_j u*] + c u*
KOKKOS_INLINE_FUNCTION
Real ManufSource(const CoefSource &cs, const Manuf &mf, Real x, Real y, Real z) {
  using kerr_impl::Sym;
  Real gt[6], gtu[6], dgt[3][6], dgtu[3][6];
  PointMetric(cs, x, y, z, gt, gtu, dgt);
  PointDgtu(gtu, dgt, dgtu);
  Real u, du[3], ddu[6];
  ManufEval(mf, x, y, z, u, du, ddu);
  Real lap = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      lap += gtu[Sym(i, j)]*ddu[Sym(i, j)] + dgtu[i][Sym(i, j)]*du[j];
    }
  }
  return -lap + mf.c*u;
}

//----------------------------------------------------------------------------------------
// W3b: the psi and chi correction equations, written as -Dt^2 u + c u = s(u).

//! \struct BgPoint
//! \brief The background scalars a correction equation needs at a cell centre.
struct BgPoint {
  Real r8, psi0, a02, chi0;
};

//! \fn PointBg(cs, x, y, z, bg)
//! \brief Kerr: Rt/8, psi0, Ahat0^ij Ahat0_ij and alpha0 psi0 from W1 (KerrConformal);
//! flat and synthetic: the flat-space values (0, 1, 0, 1).
KOKKOS_INLINE_FUNCTION
void PointBg(const CoefSource &cs, Real x, Real y, Real z, BgPoint &bg) {
  if (cs.mode == 2) {
    Real v[kKnNV];
    KerrNodeCartesian(cs.tab, cs.m_bh, x - cs.xbh[0], y - cs.xbh[1], z - cs.xbh[2], v,
                      cs.exact);
    KerrBgPoint b;
    KerrConformal(v, b);
    bg.r8 = 0.125*b.rs;
    bg.psi0 = b.psi0;
    bg.a02 = b.ahu[0]*b.ahd[0] + b.ahu[3]*b.ahd[3] + b.ahu[5]*b.ahd[5]
           + 2.0*(b.ahu[1]*b.ahd[1] + b.ahu[2]*b.ahd[2] + b.ahu[4]*b.ahd[4]);
    bg.chi0 = b.alpha0*b.psi0;
    return;
  }
  bg.r8 = 0.0;
  bg.psi0 = 1.0;
  bg.a02 = 0.0;
  bg.chi0 = 1.0;
}

//! \fn KerrPsiRHS(u, e_star, da2, psi0, a02, rhs, drhs)
//! \brief A Kerr-only copy of ConformalFactorRHS<false> (mg_cfc_conformal_factor.cpp:
//! 122-166, the cancellation-free factoring psi^-7 - psi0^-7 = -psi0^-7 x P7(x)/(1+x)^7,
//! x = u/psi0): rhs = 2 pi E* psi^-1 + Ahat^2 delta / 8 psi^-7 - Ahat0^2/8 reg(x), so that
//! eq:cpsi reads -Dt^2 u + (Rt/8) u = rhs(u); drhs = d rhs/du.
KOKKOS_INLINE_FUNCTION
void KerrPsiRHS(Real u, Real e_star, Real da2, Real psi0, Real a02, Real &rhs,
                Real &drhs) {
  const Real psi = u + psi0;
  const Real psi_inv = 1.0/psi;
  const Real psi_inv2 = psi_inv*psi_inv;
  const Real psi_inv7 = psi_inv2*psi_inv2*psi_inv2*psi_inv;
  const Real psi_inv8 = psi_inv7*psi_inv;
  const Real psi0_inv = 1.0/psi0;
  const Real x = u*psi0_inv;
  const Real opx = psi0*psi_inv;
  const Real opx2 = opx*opx;
  const Real opx7 = opx2*opx2*opx2*opx;
  const Real p02 = psi0_inv*psi0_inv;
  const Real p07 = p02*p02*p02*psi0_inv;
  const Real p7 = 7.0 + x*(21.0 + x*(35.0 + x*(35.0 + x*(21.0 + x*(7.0 + x)))));
  const Real reg = p07*x*p7*opx7;
  rhs = 2.0*M_PI*e_star*psi_inv + 0.125*da2*psi_inv7 - 0.125*a02*reg;
  drhs = -2.0*M_PI*e_star*psi_inv2 - 0.875*(da2 + a02)*psi_inv8;
}

//! \fn KerrChiRHS(u2s, dpsi, da2, psi0, a02, chi0, kx, sl)
//! \brief A Kerr-only copy of LapseReactionRHS (mg_cfc_lapse.cpp:121-142):
//! kx = 2 pi (E* + 2S*) psi^-2 + 7/8 Ahat^2 psi^-8 and
//! sl = chi0 [2 pi (E* + 2S*) psi^-2 + 7/8 dA2 psi^-8 - 7/8 Ahat0^2 reg8(x)]
//!    = chi0 (Q - Q0) (Rt cancels), so that eq:cchi reads -Dt^2 u + (Rt/8 + kx) u = -sl.
KOKKOS_INLINE_FUNCTION
void KerrChiRHS(Real u2s, Real dpsi, Real da2, Real psi0, Real a02, Real chi0, Real &kx,
                Real &sl) {
  const Real psi = dpsi + psi0;
  const Real psi_inv2 = 1.0/(psi*psi);
  const Real psi_inv8 = psi_inv2*psi_inv2*psi_inv2*psi_inv2;
  kx = 2.0*M_PI*u2s*psi_inv2 + 0.875*(da2 + a02)*psi_inv8;
  const Real psi0_inv = 1.0/psi0;
  const Real x = dpsi*psi0_inv;
  const Real opx = psi0*(1.0/psi);
  const Real opx2 = opx*opx;
  const Real opx8 = opx2*opx2*opx2*opx2;
  const Real p02 = psi0_inv*psi0_inv;
  const Real p08 = p02*p02*p02*p02;
  const Real p8 = 8.0 + x*(28.0 + x*(56.0 + x*(70.0 + x*(56.0 + x*(28.0 +
                  x*(8.0 + x))))));
  const Real reg8 = p08*x*p8*opx8;
  sl = chi0*(2.0*M_PI*u2s*psi_inv2 + 0.875*da2*psi_inv8 - 0.875*a02*reg8);
}

//! \fn PointTerm(kind, A, k, j, i, u, c, s, dsdu)
//! \brief The pointwise part of -Dt^2 u + c u = s(u) at cell (k, j, i): the linear test
//! equation (c, s from channels 18, 19), eq:cpsi (c = Rt/8, s = rhs(u)) or eq:cchi
//! (c = Rt/8 + kx, s = -sl). The smoother's Newton diagonal is c - dsdu - SDiag/h^2.
template <class Acc>
KOKKOS_INLINE_FUNCTION
void PointTerm(int kind, const Acc &A, int k, int j, int i, Real u, Real &c, Real &s,
               Real &dsdu) {
  if (kind == kKindPsi) {
    c = A.C(kChR8, k, j, i);
    KerrPsiRHS(u, A.C(kChM, k, j, i), A.C(kChDA2, k, j, i), A.C(kChPsi0, k, j, i),
               A.C(kChA02, k, j, i), s, dsdu);
  } else if (kind == kKindChi) {
    Real kx, sl;
    KerrChiRHS(A.C(kChM, k, j, i), A.C(kChDPsi, k, j, i), A.C(kChDA2, k, j, i),
               A.C(kChPsi0, k, j, i), A.C(kChA02, k, j, i), A.C(kChChi0, k, j, i), kx,
               sl);
    c = A.C(kChR8, k, j, i) + kx;
    s = -sl;
    dsdu = 0.0;
  } else {
    c = A.C(kChC, k, j, i);
    s = A.C(kChS, k, j, i);
    dsdu = 0.0;
  }
}

//! \struct MatterManuf
//! \brief Manufactured matter for the psi/chi tests: dA2 = da2_amp exp(-r^2/da2_w^2) and
//! (chi) delta psi = dpsi_amp exp(-r^2/dpsi_w^2), r = |x - mf.x0|; E* (psi) or E* + 2S*
//! (chi) is then solved for so that u* (Manuf) satisfies the continuum equation exactly.
struct MatterManuf {
  Real da2_amp = 0.0, da2_w = 1.0;
  Real dpsi_amp = 0.0, dpsi_w = 1.0;
};

//! \fn ManufMatter(kind, cs, mf, mm, x, y, z, m, da2, dpsi)
KOKKOS_INLINE_FUNCTION
void ManufMatter(int kind, const CoefSource &cs, const Manuf &mf, const MatterManuf &mm,
                 Real x, Real y, Real z, Real &m, Real &da2, Real &dpsi) {
  Manuf m0 = mf;
  m0.c = 0.0;
  const Real L = ManufSource(cs, m0, x, y, z);   // -Dt^2 u*
  Real us, du[3], ddu[6];
  ManufEval(mf, x, y, z, us, du, ddu);
  BgPoint bg;
  PointBg(cs, x, y, z, bg);
  const Real r2 = (x - mf.x0[0])*(x - mf.x0[0]) + (y - mf.x0[1])*(y - mf.x0[1])
                + (z - mf.x0[2])*(z - mf.x0[2]);
  da2 = mm.da2_amp*Kokkos::exp(-r2/(mm.da2_w*mm.da2_w));
  if (kind == kKindPsi) {
    dpsi = 0.0;
    // L + r8 u* = 2 pi E*/psi + [dA2 psi^-7 - Ahat0^2 reg]/8: rhs with E* = 0, then E*
    Real rhs0, d0;
    KerrPsiRHS(us, 0.0, da2, bg.psi0, bg.a02, rhs0, d0);
    m = (L + bg.r8*us - rhs0)*(us + bg.psi0)/(2.0*M_PI);
  } else {
    dpsi = mm.dpsi_amp*Kokkos::exp(-r2/(mm.dpsi_w*mm.dpsi_w));
    // L + (r8 + kx) u* = sl, both affine in u2s: kx = kx0 + 2 pi u2s/psi^2, likewise sl
    Real kx0, sl0;
    KerrChiRHS(0.0, dpsi, da2, bg.psi0, bg.a02, bg.chi0, kx0, sl0);
    const Real psi = dpsi + bg.psi0;
    const Real g = 2.0*M_PI/(psi*psi);
    m = -(L + (bg.r8 + kx0)*us + sl0)/(g*(us + bg.chi0));
  }
}

//----------------------------------------------------------------------------------------
// The scalar flux operator, written once over an accessor (device views or a host octet).
// A.U(k, j, i) is the unknown, A.C(c, k, j, i) a coefficient channel.

template <class Acc>
KOKKOS_INLINE_FUNCTION Real FluxLoX(const Acc &A, int k, int j, int i) {
  return A.C(kFace + 0, k, j, i)*(A.U(k, j, i) - A.U(k, j, i-1))
       + 0.25*A.C(kFace + 1, k, j, i)*((A.U(k, j+1, i) - A.U(k, j-1, i))
                                      + (A.U(k, j+1, i-1) - A.U(k, j-1, i-1)))
       + 0.25*A.C(kFace + 2, k, j, i)*((A.U(k+1, j, i) - A.U(k-1, j, i))
                                      + (A.U(k+1, j, i-1) - A.U(k-1, j, i-1)));
}
template <class Acc>
KOKKOS_INLINE_FUNCTION Real FluxLoY(const Acc &A, int k, int j, int i) {
  return A.C(kFace + 9, k, j, i)*(A.U(k, j, i) - A.U(k, j-1, i))
       + 0.25*A.C(kFace + 7, k, j, i)*((A.U(k, j, i+1) - A.U(k, j, i-1))
                                      + (A.U(k, j-1, i+1) - A.U(k, j-1, i-1)))
       + 0.25*A.C(kFace + 10, k, j, i)*((A.U(k+1, j, i) - A.U(k-1, j, i))
                                       + (A.U(k+1, j-1, i) - A.U(k-1, j-1, i)));
}
template <class Acc>
KOKKOS_INLINE_FUNCTION Real FluxLoZ(const Acc &A, int k, int j, int i) {
  return A.C(kFace + 17, k, j, i)*(A.U(k, j, i) - A.U(k-1, j, i))
       + 0.25*A.C(kFace + 14, k, j, i)*((A.U(k, j, i+1) - A.U(k, j, i-1))
                                       + (A.U(k-1, j, i+1) - A.U(k-1, j, i-1)))
       + 0.25*A.C(kFace + 16, k, j, i)*((A.U(k, j+1, i) - A.U(k, j-1, i))
                                       + (A.U(k-1, j+1, i) - A.U(k-1, j-1, i)));
}

//! h^2 Dt^2 u at cell (k, j, i)
template <class Acc>
KOKKOS_INLINE_FUNCTION Real ScalarS(const Acc &A, int k, int j, int i) {
  return (FluxLoX(A, k, j, i+1) - FluxLoX(A, k, j, i))
       + (FluxLoY(A, k, j+1, i) - FluxLoY(A, k, j, i))
       + (FluxLoZ(A, k+1, j, i) - FluxLoZ(A, k, j, i));
}

//! d(h^2 Dt^2 u)/d u_c (< 0)
template <class Acc>
KOKKOS_INLINE_FUNCTION Real ScalarSDiag(const Acc &A, int k, int j, int i) {
  return -(A.C(kFace + 0, k, j, i) + A.C(kFace + 0, k, j, i+1)
         + A.C(kFace + 9, k, j, i) + A.C(kFace + 9, k, j+1, i)
         + A.C(kFace + 17, k, j, i) + A.C(kFace + 17, k+1, j, i));
}

template <class V>
struct ViewAcc {
  V u, c;
  int m, v;
  KOKKOS_INLINE_FUNCTION Real U(int k, int j, int i) const { return u(m, v, k, j, i); }
  KOKKOS_INLINE_FUNCTION Real C(int q, int k, int j, int i) const {
    return c(m, q, k, j, i);
  }
};

//! \fn PointResidual(kind, A, k, j, i, ih2, diag)
//! \brief s(u) + S/h^2 - c u at (k, j, i) (the defect without the FAS source), and the
//! Newton-Gauss-Seidel diagonal diag = c - ds/du - SDiag/h^2 (> 0 for SDiag < 0 and h
//! small enough even where Rt/8 < 0).
template <class Acc>
KOKKOS_INLINE_FUNCTION
Real PointResidual(int kind, const Acc &A, int k, int j, int i, Real ih2, Real &diag) {
  const Real u = A.U(k, j, i);
  Real c, s, ds;
  PointTerm(kind, A, k, j, i, u, c, s, ds);
  diag = c - ds - ScalarSDiag(A, k, j, i)*ih2;
  return s + ScalarS(A, k, j, i)*ih2 - c*u;
}

//! \fn FloorPsi(kind, u, psi0, floor)
//! \brief the flat driver's psi floor (mg_cfc_conformal_factor.cpp:251), psi equation only
KOKKOS_INLINE_FUNCTION
Real FloorPsi(int kind, Real u, Real psi0, Real floor) {
  return (kind == kKindPsi && u + psi0 < floor) ? floor - psi0 : u;
}

//----------------------------------------------------------------------------------------
// Host-side helpers (kerr_mg_ops.cpp)

//! Per-block unsafe-edge bitmasks from the pack's neighbour table (nmb entries); phys:
//! the unsafe edges with a physical (non-block, non-periodic) face beside them.
void ComputeBlockEdgeMasks(MeshBlockPack *pmbp, DualArray1D<int> &mask,
                           DualArray1D<int> &phys);
//! The root grid's mask (1 entry): edge q is unsafe iff a face beside it is mg_robin or
//! mg_multipole (MGRootBoundary fills those over the interior transverse range only);
//! periodic, zerofixed and zerograd fill the whole range (multigrid_driver.cpp:2473ff).
void ComputeRootEdgeMask(const BoundaryFlag bcs[6], DualArray1D<int> &mask,
                         DualArray1D<int> &phys);
//! Counts of unsafe edges: [0] total, [1] on the domain boundary (a face beside it is a
//! physical boundary), [2] interior (the rest: a coarser/finer edge neighbour).
void CountEdges(const DualArray1D<int> &mask, const DualArray1D<int> &phys, int nmb,
                long counts[3]);

//! Rule R / B / off (+ NaN poison first) on every unsafe edge, first ghost layer, of the
//! level array u (nmmb x nvar x (nz+2ngh) x (ny+2ngh) x (nx+2ngh)); device.
void ApplyEdgeFill(const DvceArray5D<Real> &u, const DvceArray1D<int> &mask, int nmmb,
                   int nvar, int ngh, int nx, int ny, int nz, EdgeFill mode, bool poison);
//! NaN at the 8 corner ghosts (first layer) of every block: the 19-point stencil must
//! never read them.
void PoisonCorners(const DvceArray5D<Real> &u, int nmmb, int nvar, int ngh, int nx, int ny,
                   int nz);

//! Channels 0-17: gt^ij at the lower faces of every cell (ghosts included); 18: c; 19: s
//! at the cell centres (from mf). blk(m) is the extent covered by nx x ny x nz interior
//! cells (a block, the whole mesh for the root, or an octet).
void FillLevelCoeffs(const DvceArray5D<Real> &coeff, const DualArray1D<RegionSize> &blk,
                     int nmmb, int ngh, int nx, int ny, int nz, const CoefSource &cs,
                     const Manuf &mf);
//! u* at every cell (ghosts included) of the level array u, variable v
void FillManufU(const DvceArray5D<Real> &u, const DualArray1D<RegionSize> &blk, int nmmb,
                int ngh, int nx, int ny, int nz, const Manuf &mf);
//! W3b: the background channels only (faces 0-17, Rt/8, psi0, Ahat0^2, chi0), every cell
//! including the ghosts; the matter channels are left untouched.
void FillLevelBg(const DvceArray5D<Real> &coeff, const DualArray1D<RegionSize> &blk,
                 int nmmb, int ngh, int nx, int ny, int nz, const CoefSource &cs);
//! W3b test: the manufactured matter channels (ManufMatter), every cell.
void FillLevelMatterManuf(const DvceArray5D<Real> &coeff,
                          const DualArray1D<RegionSize> &blk, int nmmb, int ngh, int nx,
                          int ny, int nz, int kind, const CoefSource &cs, const Manuf &mf,
                          const MatterManuf &mm);
//! Non-finite entries of a(m, v, interior) (a bit test, immune to fast-math)
long CountNonFinite(const DvceArray5D<Real> &a, int nmmb, int nvar, int ngh, int nx,
                    int ny, int nz);

}  // namespace kmg
}  // namespace cfc

#endif  // CFC_KERR_MG_OPS_HPP_
