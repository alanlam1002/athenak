#ifndef CFC_KERR_MG_VOPS_HPP_
#define CFC_KERR_MG_VOPS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file kerr_mg_vops.hpp
//! \brief W3c (K-003): the building blocks of the Kerr vector multigrid
//! (mg_kerr_vector.hpp) for the conformal vector Laplacian on gt (det gt = 1),
//!     -DL V = s,  DL V^i = d_j (Lt V)^ij + Gt^i_jk (Lt V)^jk,
//!     (Lt V)^ij = gt^ik d_k V^j + gt^jk d_k V^i - V^k d_k gt^ij - 2/3 gt^ij d_k V^k
//! (kerr_methods.tex eq. G1/G2; Dt_j T^ij = d_j T^ij + Gt^i_jk T^jk + Gt^j_jk T^ik and
//! Gt^j_jk = d_k ln sqrt(det gt) = 0).
//!
//! Discretisation (h the cell width; T = h Lt V, a face tensor):
//!  - at the LOWER d-face of cell c (cm = c - e_d): the normal difference V(c) - V(cm), the
//!    transverse ones [(V(c+e_k) - V(c-e_k)) + (V(cm+e_k) - V(cm-e_k))]/4, the face value
//!    Vf = (V(c) + V(cm))/2, and gt^ij, d_k gt^ij evaluated analytically AT the face;
//!  - S^a = sum_d [T_{d}(c+e_d)^ad - T_d(c)^ad] + h Gt^a_jk(c) Tbar^jk, Tbar the mean of
//!    the cell's 6 face tensors; S = h^2 DL V + O(h^4);
//!  - the same 19-point stencil (faces and edges, never a corner) as the scalar operator,
//!    so Rule R on the unsafe edges (kerr_mg_ops.hpp) applies unchanged.
//! Defect def = src + S/h^2 (src = s on the finest level, the FAS source below it).
//! Point-block smoothing: V_c += omega h^2 Dinv (src + S/h^2), Dinv = (-dS/dV_c)^-1 per
//! cell (3x3, analytic coefficients, computed by linearity per level).
//!
//! Coefficient channels (all analytic at every level, root and octet, never restricted):
//!  - kVFace + 24 d + q:          gt^ij (q packed xx xy xz yy yz zz) at the lower d-face;
//!  - kVFace + 24 d + 6 + 6 k + q: d_k gt^ij at the lower d-face;
//!  - kVGam + 6 i + q:            Gt^i_q at the cell centre;
//!  - kVDinv + 3 a + b:           (-dS/dV_c)^-1, row-major (interior cells only).
//!
//! Outer boundary "kelvin" (the vector analogue of the framework's multipole BC,
//! multigrid_tasks.cpp:195-202): ghost = 2 V_K(face centre) - interior mirror on the
//! interior transverse range of each physical face, V_K the flat far field of
//! -DL_f V = q (DL_f V = Lap V + grad div V / 3), from the Kelvin tensor
//!     G_ij(x) = (7 delta_ij/r + x_i x_j/r^3)/(32 pi)
//! expanded to the quadrupole about x0:
//!     V_i = G_ij Q_j - d_a G_ij D_ja + 1/2 d_ab G_ij Q_jab,
//!     Q_j = int q_j, D_ja = int q_j y_a, Q_jab = int q_j y_a y_b, y = x - x0.
//! q is either src ("rhs") or src + (S - S_f)/h^2 ("self"; S_f the same discrete
//! operator with gt = delta), so that -DL_f V = q holds exactly in the continuum.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "kerr_background.hpp"
#include "kerr_mg_ops.hpp"

namespace cfc {
namespace kmg {

inline constexpr int kVFace = 0;
inline constexpr int kVFaceStride = 24;
inline constexpr int kVGam = 72;
inline constexpr int kVDinv = 90;
inline constexpr int kNCoeffVec = 99;
inline constexpr int kNMom = 39;   // Q_j (3), D_ja (9), Q_jab (27)

//! the Kelvin multipole moments (see the file comment): v[j], v[3 + 3 j + a],
//! v[12 + 9 j + 3 a + b]
struct VMoments {
  Real v[kNMom];
};

//! \fn SyntheticM2(cs, x, y, z, M, dM, ddM)
//! \brief SyntheticM and the second derivatives d_kl M_ab, ddM[Sym(k, l)][ab].
KOKKOS_INLINE_FUNCTION
void SyntheticM2(const CoefSource &cs, Real x, Real y, Real z, Real M[6], Real dM[3][6],
                 Real ddM[6][6]) {
  using kerr_impl::Sym;
  SyntheticM(cs, x, y, z, M, dM);
  const Real e = cs.eps, k = cs.kw, ek2 = e*k*k;
  for (int p = 0; p < 6; ++p) {
    for (int q = 0; q < 6; ++q) { ddM[p][q] = 0.0; }
  }
  ddM[Sym(0, 0)][0] = -0.5*ek2*Kokkos::sin(k*x + 0.3);
  ddM[Sym(1, 1)][3] = -0.5*ek2*Kokkos::cos(k*y + 0.7);
  ddM[Sym(2, 2)][5] = -0.5*ek2*Kokkos::sin(k*z + 1.1);
  const Real cxy = Kokkos::cos(k*(x - y) + 0.2);
  ddM[Sym(0, 0)][1] = -ek2*cxy;
  ddM[Sym(0, 1)][1] = ek2*cxy;
  ddM[Sym(1, 1)][1] = -ek2*cxy;
  const Real sxz = Kokkos::sin(k*(x + z) + 0.5);
  ddM[Sym(0, 0)][2] = -ek2*sxz;
  ddM[Sym(0, 2)][2] = -ek2*sxz;
  ddM[Sym(2, 2)][2] = -ek2*sxz;
  const Real cyz = Kokkos::cos(k*(y + 0.5*z) - 0.4);
  ddM[Sym(1, 1)][4] = -ek2*cyz;
  ddM[Sym(1, 2)][4] = -0.5*ek2*cyz;
  ddM[Sym(2, 2)][4] = -0.25*ek2*cyz;
}

//! \fn ConformalFromPhys(g, dg, ddg, gt, gtu, dgt, ddgt)
//! \brief gt = det(g)^(-1/3) g and its first and second derivatives from those of g
//! (KerrConformal's formulas, kerr_background.hpp:563-583); ddg, ddgt [Sym(k, l)][ij].
KOKKOS_INLINE_FUNCTION
void ConformalFromPhys(const Real g[6], const Real dg[3][6], const Real ddg[6][6],
                       Real gt[6], Real gtu[6], Real dgt[3][6], Real ddgt[6][6]) {
  using kerr_impl::Sym;
  Real gi[6];
  const Real D = Inv3Sym(g, gi);
  Real Lk[3], Lkl[3][3];
  for (int k = 0; k < 3; ++k) {
    Real a = 0.0;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) { a += gi[Sym(i, j)]*dg[k][Sym(i, j)]; }
    }
    Lk[k] = a;
  }
  for (int k = 0; k < 3; ++k) {
    for (int l = k; l < 3; ++l) {
      Real a = 0.0;
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          a += gi[Sym(i, j)]*ddg[Sym(k, l)][Sym(i, j)];
          for (int c = 0; c < 3; ++c) {
            for (int d = 0; d < 3; ++d) {
              a -= gi[Sym(i, c)]*gi[Sym(j, d)]*dg[l][Sym(c, d)]*dg[k][Sym(i, j)];
            }
          }
        }
      }
      Lkl[k][l] = a;
      Lkl[l][k] = a;
    }
  }
  const Real e = Kokkos::cbrt(1.0/D);
  for (int q = 0; q < 6; ++q) {
    gt[q] = e*g[q];
    gtu[q] = gi[q]/e;
  }
  for (int k = 0; k < 3; ++k) {
    for (int q = 0; q < 6; ++q) { dgt[k][q] = e*(dg[k][q] - Lk[k]*g[q]/3.0); }
  }
  for (int k = 0; k < 3; ++k) {
    for (int l = k; l < 3; ++l) {
      const int p = Sym(k, l);
      for (int q = 0; q < 6; ++q) {
        ddgt[p][q] = e*(ddg[p][q] - Lk[l]*dg[k][q]/3.0 - Lk[k]*dg[l][q]/3.0
                        - Lkl[k][l]*g[q]/3.0 + Lk[k]*Lk[l]*g[q]/9.0);
      }
    }
  }
}

//! \fn PointMetric2(cs, x, y, z, gt, gtu, dgt, ddgt)
//! \brief PointMetric plus d_kl gt_ij (all analytic: the table's spectral ddg for Kerr,
//! SyntheticM2 for the synthetic metric, zero for flat).
KOKKOS_INLINE_FUNCTION
void PointMetric2(const CoefSource &cs, Real x, Real y, Real z, Real gt[6], Real gtu[6],
                  Real dgt[3][6], Real ddgt[6][6]) {
  if (cs.mode == 2) {
    Real v[kKnNV];
    KerrNodeCartesian(cs.tab, cs.m_bh, x - cs.xbh[0], y - cs.xbh[1], z - cs.xbh[2], v,
                      cs.exact);
    Real g[6], dg[3][6], ddg[6][6];
    for (int q = 0; q < 6; ++q) {
      g[q] = v[kKnF + q];
      for (int k = 0; k < 3; ++k) { dg[k][q] = v[kKnDG + 6*k + q]; }
      for (int p = 0; p < 6; ++p) { ddg[p][q] = v[kKnDDG + 6*p + q]; }
    }
    ConformalFromPhys(g, dg, ddg, gt, gtu, dgt, ddgt);
    return;
  }
  if (cs.mode == 1) {
    Real M[6], dM[3][6], ddM[6][6];
    SyntheticM2(cs, x, y, z, M, dM, ddM);
    ConformalFromPhys(M, dM, ddM, gt, gtu, dgt, ddgt);
    return;
  }
  for (int q = 0; q < 6; ++q) {
    gt[q] = (q == 0 || q == 3 || q == 5) ? 1.0 : 0.0;
    gtu[q] = gt[q];
    for (int d = 0; d < 3; ++d) { dgt[d][q] = 0.0; }
    for (int p = 0; p < 6; ++p) { ddgt[p][q] = 0.0; }
  }
}

//! \fn ChristoffelT(gtu, dgt, gam)
//! \brief Gt^i_jk = gt^il (d_j gt_lk + d_k gt_jl - d_l gt_jk)/2, gam[i][Sym(j, k)]
KOKKOS_INLINE_FUNCTION
void ChristoffelT(const Real gtu[6], const Real dgt[3][6], Real gam[3][6]) {
  using kerr_impl::Sym;
  using kerr_impl::IdxI;
  using kerr_impl::IdxJ;
  for (int i = 0; i < 3; ++i) {
    for (int q = 0; q < 6; ++q) {
      const int j = IdxI(q), k = IdxJ(q);
      Real a = 0.0;
      for (int l = 0; l < 3; ++l) {
        a += gtu[Sym(i, l)]*(dgt[j][Sym(l, k)] + dgt[k][Sym(j, l)] - dgt[l][Sym(j, k)]);
      }
      gam[i][q] = 0.5*a;
    }
  }
}

//! \struct VManuf
//! \brief a manufactured vector V*^a, one scalar Manuf per component
struct VManuf {
  Manuf c[3];
};

//! \fn ManufVecSource(cs, vm, x, y, z, s)
//! \brief s = -DL V* at (x, y, z), analytic (d_mn gt^ij from PointMetric2 by the product
//! rule on d_m gt^ij = -gt^ia gt^jb d_m gt_ab).
KOKKOS_INLINE_FUNCTION
void ManufVecSource(const CoefSource &cs, const VManuf &vm, Real x, Real y, Real z,
                    Real s[3]) {
  using kerr_impl::Sym;
  using kerr_impl::IdxI;
  using kerr_impl::IdxJ;
  Real gt[6], gtu[6], dgt[3][6], ddgt[6][6], dgtu[3][6], ddgtu[6][6], gam[3][6];
  PointMetric2(ManufCoef(cs), x, y, z, gt, gtu, dgt, ddgt);
  PointDgtu(gtu, dgt, dgtu);
  ChristoffelT(gtu, dgt, gam);
  for (int m = 0; m < 3; ++m) {
    for (int n = m; n < 3; ++n) {
      const int p = Sym(m, n);
      for (int q = 0; q < 6; ++q) {
        const int i = IdxI(q), j = IdxJ(q);
        Real a = 0.0;
        for (int aa = 0; aa < 3; ++aa) {
          for (int bb = 0; bb < 3; ++bb) {
            a += dgtu[n][Sym(i, aa)]*gtu[Sym(j, bb)]*dgt[m][Sym(aa, bb)]
               + gtu[Sym(i, aa)]*dgtu[n][Sym(j, bb)]*dgt[m][Sym(aa, bb)]
               + gtu[Sym(i, aa)]*gtu[Sym(j, bb)]*ddgt[p][Sym(aa, bb)];
          }
        }
        ddgtu[p][q] = -a;
      }
    }
  }
  Real V[3], dV[3][3], ddV[3][6];   // dV[comp][dir]
  for (int c = 0; c < 3; ++c) { ManufEval(vm.c[c], x, y, z, V[c], dV[c], ddV[c]); }
  const Real div = dV[0][0] + dV[1][1] + dV[2][2];
  Real L[6];
  for (int q = 0; q < 6; ++q) {
    const int i = IdxI(q), j = IdxJ(q);
    Real a = 0.0;
    for (int k = 0; k < 3; ++k) {
      a += gtu[Sym(i, k)]*dV[j][k] + gtu[Sym(j, k)]*dV[i][k] - V[k]*dgtu[k][q];
    }
    L[q] = a - 2.0/3.0*gtu[q]*div;
  }
  for (int i = 0; i < 3; ++i) {
    Real a = 0.0;
    for (int j = 0; j < 3; ++j) {
      Real ddivj = 0.0;
      for (int k = 0; k < 3; ++k) { ddivj += ddV[k][Sym(j, k)]; }
      for (int k = 0; k < 3; ++k) {
        a += dgtu[j][Sym(i, k)]*dV[j][k] + gtu[Sym(i, k)]*ddV[j][Sym(j, k)]
           + dgtu[j][Sym(j, k)]*dV[i][k] + gtu[Sym(j, k)]*ddV[i][Sym(j, k)]
           - dV[k][j]*dgtu[k][Sym(i, j)] - V[k]*ddgtu[Sym(j, k)][Sym(i, j)];
        a += gam[i][Sym(j, k)]*L[Sym(j, k)];
      }
      a -= 2.0/3.0*(dgtu[j][Sym(i, j)]*div + gtu[Sym(i, j)]*ddivj);
    }
    s[i] = -a;
  }
}

//----------------------------------------------------------------------------------------
// The vector operator over an accessor: A.U(v, k, j, i), A.C(q, k, j, i).

template <class V>
struct VViewAcc {
  V u, c;
  int m;
  KOKKOS_INLINE_FUNCTION Real U(int v, int k, int j, int i) const {
    return u(m, v, k, j, i);
  }
  KOKKOS_INLINE_FUNCTION Real C(int q, int k, int j, int i) const {
    return c(m, q, k, j, i);
  }
};

//! the same unknowns on the flat metric (gt = delta, d gt = 0, Gt = 0)
template <class Acc>
struct FlatVAcc {
  Acc a;
  KOKKOS_INLINE_FUNCTION Real U(int v, int k, int j, int i) const {
    return a.U(v, k, j, i);
  }
  KOKKOS_INLINE_FUNCTION Real C(int q, int k, int j, int i) const {
    if (q < kVGam) {
      const int r = q % kVFaceStride;
      return (r == 0 || r == 3 || r == 5) ? 1.0 : 0.0;
    }
    return 0.0;
  }
};

//! unknowns = e_b at cell (kc, jc, ic), zero elsewhere (for dS/dV_c by linearity)
template <class Acc>
struct DeltaVAcc {
  Acc a;
  int b, kc, jc, ic;
  KOKKOS_INLINE_FUNCTION Real U(int v, int k, int j, int i) const {
    return (v == b && k == kc && j == jc && i == ic) ? 1.0 : 0.0;
  }
  KOKKOS_INLINE_FUNCTION Real C(int q, int k, int j, int i) const {
    return a.C(q, k, j, i);
  }
};

//! \fn FaceT(A, d, k, j, i, h, T)
//! \brief T = h (Lt V)^ab at the lower d-face of cell (k, j, i) (see the file comment)
template <class Acc>
KOKKOS_INLINE_FUNCTION
void FaceT(const Acc &A, int d, int k, int j, int i, Real h, Real T[6]) {
  using kerr_impl::Sym;
  using kerr_impl::IdxI;
  using kerr_impl::IdxJ;
  const int E[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};   // e_d as (di, dj, dk)
  const int mi = i - E[d][0], mj = j - E[d][1], mk = k - E[d][2];
  Real dV[3][3], Vf[3];   // dV[dir][comp]
  for (int c = 0; c < 3; ++c) {
    const Real u0 = A.U(c, k, j, i), um = A.U(c, mk, mj, mi);
    Vf[c] = 0.5*(u0 + um);
    for (int t = 0; t < 3; ++t) {
      if (t == d) {
        dV[t][c] = u0 - um;
      } else {
        const int di = E[t][0], dj = E[t][1], dk = E[t][2];
        dV[t][c] = 0.25*((A.U(c, k + dk, j + dj, i + di) - A.U(c, k - dk, j - dj, i - di))
                       + (A.U(c, mk + dk, mj + dj, mi + di)
                          - A.U(c, mk - dk, mj - dj, mi - di)));
      }
    }
  }
  const int base = kVFace + kVFaceStride*d;
  Real g[6];
  for (int q = 0; q < 6; ++q) { g[q] = A.C(base + q, k, j, i); }
  const Real div = dV[0][0] + dV[1][1] + dV[2][2];
  for (int q = 0; q < 6; ++q) {
    const int a = IdxI(q), b = IdxJ(q);
    Real s = 0.0;
    for (int t = 0; t < 3; ++t) {
      s += g[Sym(a, t)]*dV[t][b] + g[Sym(b, t)]*dV[t][a]
         - h*Vf[t]*A.C(base + 6 + 6*t + q, k, j, i);
    }
    T[q] = s - 2.0/3.0*g[q]*div;
  }
}

//! \fn VecS(A, k, j, i, h, S)
//! \brief S = h^2 DL V at cell (k, j, i) (see the file comment)
template <class Acc>
KOKKOS_INLINE_FUNCTION
void VecS(const Acc &A, int k, int j, int i, Real h, Real S[3]) {
  using kerr_impl::Sym;
  Real Tl[3][6], Tu[3][6];
  FaceT(A, 0, k, j, i, h, Tl[0]);
  FaceT(A, 0, k, j, i + 1, h, Tu[0]);
  FaceT(A, 1, k, j, i, h, Tl[1]);
  FaceT(A, 1, k, j + 1, i, h, Tu[1]);
  FaceT(A, 2, k, j, i, h, Tl[2]);
  FaceT(A, 2, k + 1, j, i, h, Tu[2]);
  Real Tb[6];
  for (int q = 0; q < 6; ++q) {
    Tb[q] = (Tl[0][q] + Tu[0][q] + Tl[1][q] + Tu[1][q] + Tl[2][q] + Tu[2][q])/6.0;
  }
  for (int a = 0; a < 3; ++a) {
    Real s = 0.0;
    for (int d = 0; d < 3; ++d) { s += Tu[d][Sym(a, d)] - Tl[d][Sym(a, d)]; }
    Real gs = 0.0;
    for (int q = 0; q < 6; ++q) {
      const Real w = (q == 0 || q == 3 || q == 5) ? 1.0 : 2.0;
      gs += w*A.C(kVGam + 6*a + q, k, j, i)*Tb[q];
    }
    S[a] = s + h*gs;
  }
}

//! \fn Inv3(M, Mi) -> det (general 3x3, row-major)
KOKKOS_INLINE_FUNCTION Real Inv3(const Real M[9], Real Mi[9]) {
  const Real c00 = M[4]*M[8] - M[5]*M[7];
  const Real c01 = M[5]*M[6] - M[3]*M[8];
  const Real c02 = M[3]*M[7] - M[4]*M[6];
  const Real det = M[0]*c00 + M[1]*c01 + M[2]*c02;
  const Real id = 1.0/det;
  Mi[0] = c00*id;
  Mi[1] = (M[2]*M[7] - M[1]*M[8])*id;
  Mi[2] = (M[1]*M[5] - M[2]*M[4])*id;
  Mi[3] = c01*id;
  Mi[4] = (M[0]*M[8] - M[2]*M[6])*id;
  Mi[5] = (M[2]*M[3] - M[0]*M[5])*id;
  Mi[6] = c02*id;
  Mi[7] = (M[1]*M[6] - M[0]*M[7])*id;
  Mi[8] = (M[0]*M[4] - M[1]*M[3])*id;
  return det;
}

//! \fn VecDinv(A, k, j, i, h, Di) -> det(-dS/dV_c)
template <class Acc>
KOKKOS_INLINE_FUNCTION Real VecDinv(const Acc &A, int k, int j, int i, Real h, Real Di[9]) {
  Real M[9];
  for (int b = 0; b < 3; ++b) {
    DeltaVAcc<Acc> D{A, b, k, j, i};
    Real S[3];
    VecS(D, k, j, i, h, S);
    for (int a = 0; a < 3; ++a) { M[3*a + b] = -S[a]; }
  }
  return Inv3(M, Di);
}

//! \fn KelvinV(mom, x, y, z, V)
//! \brief the quadrupole Kelvin far field at y = (x, y, z) - x0 (see the file comment)
KOKKOS_INLINE_FUNCTION
void KelvinV(const VMoments &mom, Real x, Real y, Real z, Real V[3]) {
  const Real X[3] = {x, y, z};
  const Real r2 = x*x + y*y + z*z;
  const Real ir2 = 1.0/r2, ir = Kokkos::sqrt(ir2);
  const Real ir3 = ir*ir2, ir5 = ir3*ir2, ir7 = ir5*ir2;
  const Real c = 1.0/(32.0*M_PI);
  for (int i = 0; i < 3; ++i) {
    Real s = 0.0;
    for (int j = 0; j < 3; ++j) {
      const Real dij = (i == j) ? 1.0 : 0.0;
      s += c*(7.0*dij*ir + X[i]*X[j]*ir3)*mom.v[j];
      for (int a = 0; a < 3; ++a) {
        const Real dai = (a == i) ? 1.0 : 0.0, daj = (a == j) ? 1.0 : 0.0;
        s -= c*(-7.0*dij*X[a]*ir3 + (dai*X[j] + daj*X[i])*ir3
                - 3.0*X[i]*X[j]*X[a]*ir5)*mom.v[3 + 3*j + a];
        for (int b = 0; b < 3; ++b) {
          const Real dbi = (b == i) ? 1.0 : 0.0, dbj = (b == j) ? 1.0 : 0.0;
          const Real dab = (a == b) ? 1.0 : 0.0;
          const Real ddg = 7.0*dij*(3.0*X[a]*X[b]*ir5 - dab*ir3)
                         + (dai*dbj + daj*dbi)*ir3
                         - 3.0*(dai*X[j]*X[b] + daj*X[i]*X[b] + dbi*X[j]*X[a]
                                + dbj*X[i]*X[a] + dab*X[i]*X[j])*ir5
                         + 15.0*X[i]*X[j]*X[a]*X[b]*ir7;
          s += 0.5*c*ddg*mom.v[12 + 9*j + 3*a + b];
        }
      }
    }
    V[i] = s;
  }
}

//----------------------------------------------------------------------------------------
// Host-side helpers (kerr_mg_vops.cpp)

//! faces, d gt^ij and Gt on every cell (ghosts included), then Dinv on the interior
//! cells; returns the number of interior cells with det(-dS/dV_c) <= 0 or non-finite
long FillLevelVecCoeffs(const DvceArray5D<Real> &coeff, const DualArray1D<RegionSize> &blk,
                        int nmmb, int ngh, int nx, int ny, int nz, const CoefSource &cs);
//! V* at every cell (ghosts included), variables 0-2
void FillManufV(const DvceArray5D<Real> &u, const DualArray1D<RegionSize> &blk, int nmmb,
                int ngh, int nx, int ny, int nz, const VManuf &vm);
//! s = -DL V* at every cell, variables 0-2
void FillManufVecSrc(const DvceArray5D<Real> &src, const DualArray1D<RegionSize> &blk,
                     int nmmb, int ngh, int nx, int ny, int nz, const CoefSource &cs,
                     const VManuf &vm);
//! the kelvin outer boundary on the faces flagged in faces(m) (bit f = 2 dir + side):
//! ghost = 2 V_K(face centre) - mirror, interior transverse range, all ghost layers
void KelvinFaceFill(const DvceArray5D<Real> &u, const DualArray1D<RegionSize> &blk,
                    const DvceArray1D<int> &faces, int nmmb, int ngh, int nx, int ny,
                    int nz, const Real x0[3], const VMoments &mom);
//! this rank's moments (unreduced) of q over the interior cells of a finest level:
//! self = false: q = src; self = true: q = src + (S - S_f)/h^2
void LocalMoments(const DvceArray5D<Real> &u, const DvceArray5D<Real> &src,
                  const DvceArray5D<Real> &coeff, const DualArray1D<RegionSize> &blk,
                  int nmmb, int ngh, int nx, int ny, int nz, bool self, const Real x0[3],
                  Real mom[kNMom]);
//! a check of PointMetric2 on npts pseudo-random points of the box [lo, lo + len]:
//! out[0] max |dgt - dgt(PointMetric)|, out[1] max |ddgt - FD(dgt(PointMetric))| (central,
//! step fd), out[2] max |ddgt| (scale), out[3] max |gt - gt(PointMetric)|
void MetricSelfCheck(const CoefSource &cs, const Real lo[3], const Real len[3], int npts,
                     Real fd, Real out[4]);

}  // namespace kmg
}  // namespace cfc

#endif  // CFC_KERR_MG_VOPS_HPP_
