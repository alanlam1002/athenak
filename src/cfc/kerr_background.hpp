#ifndef CFC_KERR_BACKGROUND_HPP_
#define CFC_KERR_BACKGROUND_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file kerr_background.hpp
//! \brief W1 (K-001): the Kerr maximal-trumpet background for CFC on a spinning IMBH, from
//! the KSTAB v1 (A3) tables and their KC-24 F2 half-plane product. Spin along +z, the table
//! mass M = 1 (scaled to m_bh here), the A3 azimuth.
//!
//! Inputs (read on the host, copied to the device):
//!  - F2: <prefix>.txt (header: table_a, r_min, r_max, nr, nth, nv) and <prefix>.bin,
//!    little-endian float64 [nr][nth][100]; node (i, j) at r_f = exp(ln r_min + i h_u),
//!    th = j pi/(nth - 1), the point r_f (sin th, 0, cos th) (phi = 0). Per node, Cartesian:
//!      0..15 g_ij (xx xy xz yy yz zz), K_ij, alpha, beta^i;  16..33 d_k g_ij (k outer);
//!      34..69 d_kl g_ij (kl = xx xy xz yy yz zz outer);  70..87 d_k K_ij;  88..90 d_k alpha;
//!      91..99 d_k beta^i (k outer).  All derivatives spectral (the table differentiated).
//!  - the KSTAB table itself (text, Chebyshev x Fourier coefficients): the fallback exponents
//!    p (asymp line) and the exact spectral evaluation used beyond F2's r_max.
//!
//! Evaluation at a point x (relative to the hole; KerrNodeCartesian):
//!  - r_min <= r_f <= r_max: tensor-product quintic Lagrange in (ln r_f, th) of all 100 node
//!    values, stencil as kslice/kstab/kc24_a2.cpp do_f2 (one-sided, clamped, at the edges).
//!    The interpolation never reaches below r_min: the stencil is clamped to row 0 = r_min
//!    (row 0 replaced by the exact nodes at r_min at load: KerrSetRow0Exact).
//!  - r_f < r_min: the table's own C0 cylinder fallback f(r_min, th) (r_f/r_min)^p per family
//!    (kstab.hpp eval_polar_d). Row 0 is interpolated in th only (no radial interpolation);
//!    its derivatives are replaced by those of the homogeneous extension (KerrFallbackComp:
//!    coordinate-free, exact for a degree-p homogeneous function) and then scaled by
//!    s^p, s^(p-1), s^(p-2), s = r_f/r_min. Every r_min is < 1.5e-4 M, deep inside the
//!    excision (D4) and the flat blend (D3).
//!  - r_f > r_max (= 1e3 M, F2's edge): the exact spectral evaluation (the log segments and the
//!    inv segment to infinity), KerrNodeSpectral, on the device. C0 at r_max to the quintic
//!    error (V0(f) measures the jump).
//!  - then R_z(phi) on every index (tensor R T R^T, vector R v, derivative indices too).
//! No finite differences of any background field (R2).
//!
//! KerrConformal forms psi0, gt, gt^-1, d gt, Gt^k_ij, Rt_ij, Rt, alpha0, d alpha0, beta0,
//! d beta0, d ln psi0, K0_ij, Ahat0^ij = psi0^10 K0^ij, Ahat0_ij = psi0^2 K0_ij from the
//! node values (the formulas of kc24_a2.cpp conf_spectral). FillKerrBackground stores them
//! per cell (kKbgN components, kKbg* offsets) about the hole position xbh.

#include <cmath>
#include <string>

#include "athena.hpp"

// forward declarations
class MeshBlockPack;

namespace cfc {

// ---- F2 node layout
inline constexpr int kKnNV  = 100;
inline constexpr int kKnF   = 0;    // g(6) K(6) alpha beta(3)
inline constexpr int kKnDG  = 16;   // d_k g_ij: kKnDG + 6k + ij
inline constexpr int kKnDDG = 34;   // d_kl g_ij: kKnDDG + 6 kl + ij
inline constexpr int kKnDK  = 70;   // d_k K_ij: kKnDK + 6k + ij
inline constexpr int kKnDA  = 88;   // d_k alpha: kKnDA + k
inline constexpr int kKnDB  = 91;   // d_k beta^i: kKnDB + 3k + i
inline constexpr int kKMaxN = 64;   // max Chebyshev order per segment (checked at load)
inline constexpr int kKMaxK = 64;   // max Fourier order + 1 (checked at load)

// ---- per-cell output (FillKerrBackground); symmetric pairs packed xx xy xz yy yz zz
inline constexpr int kKbgPsi0    = 0;
inline constexpr int kKbgAlpha0  = 1;
inline constexpr int kKbgBeta0   = 2;    // + i
inline constexpr int kKbgGt      = 5;    // + ij
inline constexpr int kKbgGtu     = 11;   // + ij
inline constexpr int kKbgDGt     = 17;   // + 6k + ij   d_k gt_ij
inline constexpr int kKbgGam     = 35;   // + 6k + ij   Gt^k_ij
inline constexpr int kKbgRt      = 53;   // + ij        Rt_ij
inline constexpr int kKbgRs      = 59;   //             Rt
inline constexpr int kKbgDAlpha0 = 60;   // + k
inline constexpr int kKbgDBeta0  = 63;   // + 3k + i    d_k beta0^i
inline constexpr int kKbgDLnPsi0 = 72;   // + k
inline constexpr int kKbgK0      = 75;   // + ij        K0_ij
inline constexpr int kKbgAhu     = 81;   // + ij        Ahat0^ij
inline constexpr int kKbgAhd     = 87;   // + ij        Ahat0_ij
inline constexpr int kKbgN       = 93;

//! \struct KerrTable
//! \brief Device-side table data (Views: cheap to copy into a kernel).
struct KerrTable {
  // F2
  DvceArray1D<Real> f2;      // [nr*nth*kKnNV]
  int nr = 0, nth = 0;
  Real u0 = 0, hu = 0, ht = 0, rmin = 0, rmax = 0;
  Real a = 0, azimuth_c = 0;
  Real p[4] = {-2, 0, 0, 1};  // fallback exponents: g, K, alpha, beta
  // KSTAB spectral (all segments)
  int nseg = 0, kmax = 0;
  Real r1 = 0;
  DvceArray1D<Real> seg_lo, seg_hi;   // seg_hi unused for inv
  DvceArray1D<int> seg_n, seg_inv, seg_off;  // seg_off: offset of [16][N][kmax+1]
  DvceArray1D<Real> coef;
};

//! \fn void LoadKerrTable(const std::string &f2_prefix, const std::string &kstab_path,
//!                        KerrTable &t)
//! \brief Read F2 (<prefix>.txt, <prefix>.bin) and the KSTAB table on the host and copy
//! both to the device. Checks: the F2 header (nv = 100, table_a and r_min equal to the
//! KSTAB table's), the .bin size. Throws std::runtime_error on any mismatch.
void LoadKerrTable(const std::string &f2_prefix, const std::string &kstab_path,
                   KerrTable &t);

//! \fn void KerrSetRow0Exact(KerrTable &t)
//! \brief Replace F2's row 0 by the spectral nodes at r_f = r_min exactly. Some of F2's
//! row-0 nodes are evaluated 1 ulp below r_min, where the generator's KSTAB reader returns
//! its cylinder fallback (radial derivatives p F/r instead of the table's): at a = 0 and 0.6
//! because the row sits at exp(ln r_min), and at 0.97, 0.98, 0.995 and 0.997 because the
//! reader's recomputed r_f = hypot(r sin th, r cos th) (kstab.hpp:275) rounds below r_min
//! at some th (KS-108, RK-017). The replacement evaluates r_f = r_min as given. Called
//! by the CFC path at load unless <cfc> kerr_row0_exact = false (RK-004 sec. 8.1); the unit
//! test does the same unless KERR_ROW0_EXACT=0.
void KerrSetRow0Exact(KerrTable &t);

namespace kerr_impl {

KOKKOS_INLINE_FUNCTION int Sym(int i, int j) {
  return (i == j) ? (i == 0 ? 0 : (i == 1 ? 3 : 5))
                  : ((i + j == 1) ? 1 : ((i + j == 2) ? 2 : 4));
}
KOKKOS_INLINE_FUNCTION int IdxI(int q) { return (q < 3) ? 0 : ((q < 5) ? 1 : 2); }
KOKKOS_INLINE_FUNCTION int IdxJ(int q) {
  return (q == 0) ? 0 : ((q == 1 || q == 3) ? 1 : 2);
}
//! node value -> family (0 g, 1 K, 2 alpha, 3 beta)
KOKKOS_INLINE_FUNCTION int NodeFamily(int q) {
  if (q < 16) { return (q < 6) ? 0 : ((q < 12) ? 1 : ((q == 12) ? 2 : 3)); }
  if (q < 70) { return 0; }
  if (q < 88) { return 1; }
  if (q < 91) { return 2; }
  return 3;
}
//! the KSTAB field is in the sin k th family
KOKKOS_INLINE_FUNCTION bool OddField(int f) {
  return f == 2 || f == 4 || f == 8 || f == 10 || f == 13 || f == 14;
}

//! M . f for a 16-vector of fields (g, K tensors; alpha scalar; beta vector):
//! generator: ad = M T - T M (M antisymmetric) / M v / 0; else M T M^T / M v / unchanged.
KOKKOS_INLINE_FUNCTION
void ActFields(const Real M[3][3], const Real *in, Real *o, bool generator) {
  for (int t = 0; t < 2; ++t) {
    Real T0[3][3];
    for (int q = 0; q < 6; ++q) {
      T0[IdxI(q)][IdxJ(q)] = in[6*t + q];
      T0[IdxJ(q)][IdxI(q)] = in[6*t + q];
    }
    for (int q = 0; q < 6; ++q) {
      const int i = IdxI(q), j = IdxJ(q);
      Real v = 0.0;
      for (int m = 0; m < 3; ++m) {
        if (generator) {
          v += M[i][m]*T0[m][j] + T0[i][m]*M[j][m];
        } else {
          for (int n = 0; n < 3; ++n) { v += M[i][m]*M[j][n]*T0[m][n]; }
        }
      }
      o[6*t + q] = v;
    }
  }
  o[12] = generator ? 0.0 : in[12];
  for (int i = 0; i < 3; ++i) {
    o[13 + i] = M[i][0]*in[13] + M[i][1]*in[14] + M[i][2]*in[15];
  }
}

//! gather the 16-field first derivative d_k f from a node (entries absent from the node:
//! none; every field has a first derivative)
KOKKOS_INLINE_FUNCTION void NodeD1(const Real *v, int k, Real *d) {
  for (int q = 0; q < 6; ++q) { d[q] = v[kKnDG + 6*k + q]; }
  for (int q = 0; q < 6; ++q) { d[6 + q] = v[kKnDK + 6*k + q]; }
  d[12] = v[kKnDA + k];
  for (int i = 0; i < 3; ++i) { d[13 + i] = v[kKnDB + 3*k + i]; }
}
KOKKOS_INLINE_FUNCTION void NodeSetD1(Real *v, int k, const Real *d) {
  for (int q = 0; q < 6; ++q) { v[kKnDG + 6*k + q] = d[q]; }
  for (int q = 0; q < 6; ++q) { v[kKnDK + 6*k + q] = d[6 + q]; }
  v[kKnDA + k] = d[12];
  for (int i = 0; i < 3; ++i) { v[kKnDB + 3*k + i] = d[13 + i]; }
}

//! the homogeneous (degree p) extension of one component off the sphere r = r0 through
//! direction n: given F, grad F (and the Hessian H, packed, if H != nullptr) at r0 n, replace
//! the radial derivative by p F/r0 (Euler) and return F s^p, grad s^(p-1), H s^(p-2).
//! grad_fb = grad + n (p F/r0 - n.grad);
//! H_fb = P H P + (delta/r0) P + (p-1)/r0 (n t + t n) + p(p-1) F/r0^2 n n,  t = P grad.
KOKKOS_INLINE_FUNCTION
void FallbackComp(Real p, Real r0, Real s, const Real n[3], Real *F, Real *g0, Real *g1,
                  Real *g2, Real *H) {
  Real g[3] = {*g0, *g1, *g2};
  const Real Fr = n[0]*g[0] + n[1]*g[1] + n[2]*g[2];
  const Real delta = p*(*F)/r0 - Fr;
  const Real sp = std::pow(s, p), sp1 = std::pow(s, p - 1.0), sp2 = std::pow(s, p - 2.0);
  if (H != nullptr) {
    Real t[3], Hm[3][3], Hn[3], nHn = 0.0;
    for (int i = 0; i < 3; ++i) { t[i] = g[i] - n[i]*Fr; }
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) { Hm[i][j] = H[Sym(i, j)]; }
    }
    for (int i = 0; i < 3; ++i) {
      Hn[i] = Hm[i][0]*n[0] + Hm[i][1]*n[1] + Hm[i][2]*n[2];
      nHn += n[i]*Hn[i];
    }
    for (int q = 0; q < 6; ++q) {
      const int i = IdxI(q), j = IdxJ(q);
      const Real Pij = ((i == j) ? 1.0 : 0.0) - n[i]*n[j];
      const Real PHP = Hm[i][j] - n[i]*Hn[j] - Hn[i]*n[j] + n[i]*n[j]*nHn;
      H[q] = sp2*(PHP + delta/r0*Pij + (p - 1.0)/r0*(n[i]*t[j] + t[i]*n[j])
                  + p*(p - 1.0)*(*F)/(r0*r0)*n[i]*n[j]);
    }
  }
  *g0 = sp1*(g[0] + n[0]*delta);
  *g1 = sp1*(g[1] + n[1]*delta);
  *g2 = sp1*(g[2] + n[2]*delta);
  *F = sp*(*F);
}

//! apply FallbackComp to every component of a node at r0 n (half-plane, n = (sin, 0, cos))
KOKKOS_INLINE_FUNCTION
void FallbackNode(const KerrTable &t, Real r0, Real s, const Real n[3], Real *v) {
  for (int q = 0; q < 6; ++q) {
    Real H[6];
    for (int m = 0; m < 6; ++m) { H[m] = v[kKnDDG + 6*m + q]; }
    FallbackComp(t.p[0], r0, s, n, &v[q], &v[kKnDG + q], &v[kKnDG + 6 + q],
                 &v[kKnDG + 12 + q], H);
    for (int m = 0; m < 6; ++m) { v[kKnDDG + 6*m + q] = H[m]; }
  }
  for (int q = 0; q < 6; ++q) {
    FallbackComp(t.p[1], r0, s, n, &v[6 + q], &v[kKnDK + q], &v[kKnDK + 6 + q],
                 &v[kKnDK + 12 + q], nullptr);
  }
  FallbackComp(t.p[2], r0, s, n, &v[12], &v[kKnDA], &v[kKnDA + 1], &v[kKnDA + 2],
               nullptr);
  for (int i = 0; i < 3; ++i) {
    FallbackComp(t.p[3], r0, s, n, &v[13 + i], &v[kKnDB + i], &v[kKnDB + 3 + i],
                 &v[kKnDB + 6 + i], nullptr);
  }
}

}  // namespace kerr_impl

//----------------------------------------------------------------------------------------
//! \fn void KerrNodeInterp(const KerrTable &t, Real u, Real th, Real *v)
//! \brief Quintic Lagrange interpolation of the 100 node values at (u = ln r_f, th), the
//! stencil of kc24_a2.cpp do_f2 (6 x 6 nodes; start floor(x) - 2, clamped to the grid).
KOKKOS_INLINE_FUNCTION
void KerrNodeInterp(const KerrTable &t, Real u, Real th, Real *v) {
  constexpr int np = 6;
  const Real xu = (u - t.u0)/t.hu, xt = th/t.ht;
  int iu = static_cast<int>(std::floor(xu)) - (np/2 - 1);
  int it = static_cast<int>(std::floor(xt)) - (np/2 - 1);
  iu = (iu < 0) ? 0 : ((iu > t.nr - np) ? t.nr - np : iu);
  it = (it < 0) ? 0 : ((it > t.nth - np) ? t.nth - np : it);
  Real wu[np], wt[np];
  for (int a = 0; a < np; ++a) {
    wu[a] = 1.0; wt[a] = 1.0;
    for (int b = 0; b < np; ++b) {
      if (b == a) continue;
      wu[a] *= (xu - (iu + b))/static_cast<Real>(a - b);
      wt[a] *= (xt - (it + b))/static_cast<Real>(a - b);
    }
  }
  // one scalar accumulator per value (keeps the 100-value node out of the inner loops)
  const size_t base = (static_cast<size_t>(iu)*t.nth + it)*kKnNV;
  const size_t rowstride = static_cast<size_t>(t.nth)*kKnNV;
  for (int q = 0; q < kKnNV; ++q) {
    Real acc = 0.0;
    for (int a = 0; a < np; ++a) {
      const size_t ra = base + a*rowstride + q;
      Real accb = 0.0;
      for (int b = 0; b < np; ++b) { accb += wt[b]*t.f2(ra + b*kKnNV); }
      acc += wu[a]*accb;
    }
    v[q] = acc;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void KerrNodeSpectral(const KerrTable &t, Real rf, Real th, Real *v)
//! \brief The exact node values at (r_f >= r_min, th) on the half-plane phi = 0: the KSTAB
//! expansion differentiated (Chebyshev derivative recurrences; d/dth of cos k th, sin k th),
//! then the polar chain rule and the axisymmetry generator for the y-derivatives (on the
//! axis, rho <= 1e-12 r_f: R_z(pi/2), R_z(pi/4)). Same algorithm as the KSTAB v1 reader's
//! eval_polar_d/eval_d (KC-24), at phi = 0.
KOKKOS_INLINE_FUNCTION
void KerrNodeSpectral(const KerrTable &t, Real rf, Real th, Real *v) {
  using namespace kerr_impl;  // NOLINT(build/namespaces)
  int s = t.nseg - 1;
  for (int i = 0; i < t.nseg; ++i) {
    if (!t.seg_inv(i) && rf <= t.seg_hi(i)) { s = i; break; }
  }
  const int N = t.seg_n(s), K1 = t.kmax + 1;
  Real x, dx, ddx;
  if (t.seg_inv(s)) {
    x = 2.0*t.r1/rf - 1.0;
    dx = -2.0*t.r1/(rf*rf);
    ddx = 4.0*t.r1/(rf*rf*rf);
  } else {
    const Real L = std::log(t.seg_hi(s)) - std::log(t.seg_lo(s));
    x = 2.0*(std::log(rf) - std::log(t.seg_lo(s)))/L - 1.0;
    dx = 2.0/(rf*L);
    ddx = -2.0/(rf*rf*L);
  }
  Real T[kKMaxN], dT[kKMaxN], d2T[kKMaxN], ck[kKMaxK], sk[kKMaxK];
  T[0] = 1.0; dT[0] = 0.0; d2T[0] = 0.0;
  if (N > 1) { T[1] = x; dT[1] = 1.0; d2T[1] = 0.0; }
  for (int n = 2; n < N; ++n) {
    T[n] = 2.0*x*T[n-1] - T[n-2];
    dT[n] = 2.0*T[n-1] + 2.0*x*dT[n-1] - dT[n-2];
    d2T[n] = 4.0*dT[n-1] + 2.0*x*d2T[n-1] - d2T[n-2];
  }
  const Real c1 = std::cos(th), s1 = std::sin(th);
  ck[0] = 1.0; sk[0] = 0.0;
  for (int k = 1; k <= t.kmax; ++k) {
    ck[k] = ck[k-1]*c1 - sk[k-1]*s1;
    sk[k] = sk[k-1]*c1 + ck[k-1]*s1;
  }
  Real f[16], fr[16], ft[16], frr[16], frt[16], ftt[16];
  for (int q = 0; q < 16; ++q) {
    const bool odd = OddField(q);
    const int off = t.seg_off(s) + q*N*K1;
    Real val = 0, vx = 0, vxx = 0, vt = 0, vxt = 0, vtt = 0;
    for (int n = 0; n < N; ++n) {
      Real w = 0, wt = 0, wtt = 0;
      for (int k = 0; k <= t.kmax; ++k) {
        const Real c = t.coef(off + n*K1 + k);
        const Real e = odd ? sk[k] : ck[k], de = odd ? k*ck[k] : -k*sk[k];
        w += c*e;
        wt += c*de;
        wtt -= c*k*k*e;
      }
      val += T[n]*w;
      vx += dT[n]*w;
      vxx += d2T[n]*w;
      vt += T[n]*wt;
      vxt += dT[n]*wt;
      vtt += T[n]*wtt;
    }
    f[q] = val;
    fr[q] = vx*dx;
    frr[q] = vxx*dx*dx + vx*ddx;
    ft[q] = vt;
    frt[q] = vxt*dx;
    ftt[q] = vtt;
  }
  // Cartesian derivatives at (rho, 0, z): P (x y z), Q (xx xy xz yy yz zz)
  const Real sn = std::sin(th), cs = std::cos(th), rho = rf*sn;
  Real P[3][16], Q[6][16];
  for (int q = 0; q < 16; ++q) {
    P[0][q] = sn*fr[q] + cs/rf*ft[q];
    P[2][q] = cs*fr[q] - sn/rf*ft[q];
    Q[0][q] = sn*sn*frr[q] + 2*sn*cs/rf*frt[q] + cs*cs/(rf*rf)*ftt[q] + cs*cs/rf*fr[q]
              - 2*sn*cs/(rf*rf)*ft[q];
    Q[5][q] = cs*cs*frr[q] - 2*sn*cs/rf*frt[q] + sn*sn/(rf*rf)*ftt[q] + sn*sn/rf*fr[q]
              + 2*sn*cs/(rf*rf)*ft[q];
    Q[2][q] = sn*cs*frr[q] + (cs*cs - sn*sn)/rf*frt[q] - sn*cs/(rf*rf)*ftt[q]
              - sn*cs/rf*fr[q] - (cs*cs - sn*sn)/(rf*rf)*ft[q];
  }
  if (rho > 1e-12*rf) {
    const Real Jz[3][3] = {{0, -1, 0}, {1, 0, 0}, {0, 0, 0}};
    Real adf[16], adP0[16], adP2[16], adad[16];
    ActFields(Jz, f, adf, true);
    ActFields(Jz, adf, adad, true);
    ActFields(Jz, P[0], adP0, true);
    ActFields(Jz, P[2], adP2, true);
    for (int q = 0; q < 16; ++q) {
      P[1][q] = adf[q]/rho;
      Q[3][q] = P[0][q]/rho + adad[q]/(rho*rho);
      Q[1][q] = adP0[q]/rho - adf[q]/(rho*rho);
      Q[4][q] = adP2[q]/rho;
    }
  } else {
    const Real h = std::sqrt(0.5);
    const Real R90[3][3] = {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}};
    const Real R45[3][3] = {{h, -h, 0}, {h, h, 0}, {0, 0, 1}};
    Real r45[16];
    ActFields(R90, P[0], P[1], false);
    ActFields(R90, Q[0], Q[3], false);
    ActFields(R90, Q[2], Q[4], false);
    ActFields(R45, Q[0], r45, false);
    for (int q = 0; q < 16; ++q) { Q[1][q] = r45[q] - 0.5*(Q[0][q] + Q[3][q]); }
  }
  for (int q = 0; q < 16; ++q) { v[q] = f[q]; }
  for (int k = 0; k < 3; ++k) { NodeSetD1(v, k, P[k]); }
  for (int m = 0; m < 6; ++m) {
    for (int q = 0; q < 6; ++q) { v[kKnDDG + 6*m + q] = Q[m][q]; }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void KerrNodePolar(const KerrTable &t, Real rf, Real th, Real *v, bool exact)
//! \brief The node values on the half-plane phi = 0 at (r_f, th): interpolated (exact =
//! false; spectral beyond r_max) or spectral everywhere (exact = true, the device reference
//! for V0). Below r_min: the cylinder fallback from the r_min data (see the file header).
KOKKOS_INLINE_FUNCTION
void KerrNodePolar(const KerrTable &t, Real rf, Real th, Real *v, bool exact) {
  if (rf < t.rmin) {
    if (exact) {
      KerrNodeSpectral(t, t.rmin, th, v);
    } else {
      KerrNodeInterp(t, t.u0, th, v);   // xu = 0: row 0 exactly, interpolated in th only
    }
    const Real n[3] = {std::sin(th), 0.0, std::cos(th)};
    kerr_impl::FallbackNode(t, t.rmin, rf/t.rmin, n, v);
  } else if (exact || rf > t.rmax) {
    KerrNodeSpectral(t, rf, th, v);
  } else {
    KerrNodeInterp(t, std::log(rf), th, v);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void KerrRotateNode(Real c, Real s, const Real *v0, Real *v)
//! \brief R_z(phi) (c = cos phi, s = sin phi) on every index of a node: the field indices
//! (tensors R T R^T, vectors R v) and the derivative indices (vector / symmetric tensor).
KOKKOS_INLINE_FUNCTION
void KerrRotateNode(Real c, Real s, const Real *v0, Real *v) {
  using namespace kerr_impl;  // NOLINT(build/namespaces)
  const Real R[3][3] = {{c, -s, 0}, {s, c, 0}, {0, 0, 1}};
  ActFields(R, v0, v, false);
  Real Pr[3][16];
  for (int k = 0; k < 3; ++k) {
    Real d[16];
    NodeD1(v0, k, d);
    ActFields(R, d, Pr[k], false);
  }
  for (int k = 0; k < 3; ++k) {
    Real d[16];
    for (int q = 0; q < 16; ++q) { d[q] = R[k][0]*Pr[0][q] + R[k][1]*Pr[1][q] + R[k][2]*Pr[2][q]; }
    NodeSetD1(v, k, d);
  }
  // second derivatives of g: the field indices, then the derivative pair
  Real Qr[6][6];
  for (int m = 0; m < 6; ++m) {
    Real T0[3][3];
    for (int q = 0; q < 6; ++q) {
      T0[IdxI(q)][IdxJ(q)] = v0[kKnDDG + 6*m + q];
      T0[IdxJ(q)][IdxI(q)] = v0[kKnDDG + 6*m + q];
    }
    for (int q = 0; q < 6; ++q) {
      const int i = IdxI(q), j = IdxJ(q);
      Real a = 0.0;
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) { a += R[i][k]*R[j][l]*T0[k][l]; }
      }
      Qr[m][q] = a;
    }
  }
  for (int m = 0; m < 6; ++m) {
    const int k = IdxI(m), l = IdxJ(m);
    for (int q = 0; q < 6; ++q) {
      Real a = 0.0;
      for (int aa = 0; aa < 3; ++aa) {
        for (int b = 0; b < 3; ++b) { a += R[k][aa]*R[l][b]*Qr[Sym(aa, b)][q]; }
      }
      v[kKnDDG + 6*m + q] = a;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void KerrNodeCartesian(const KerrTable &t, Real m_bh, Real x, Real y, Real z,
//!                            Real *v, bool exact)
//! \brief The 100 node values at the Cartesian offset (x, y, z) from the hole, for a hole of
//! mass m_bh (the table has M = 1: evaluated at x/m_bh; K, first derivatives scaled by
//! 1/m_bh, d K and second derivatives by 1/m_bh^2).
KOKKOS_INLINE_FUNCTION
void KerrNodeCartesian(const KerrTable &t, Real m_bh, Real x, Real y, Real z, Real *v,
                       bool exact) {
  const Real im = 1.0/m_bh;
  const Real xs = x*im, ys = y*im, zs = z*im;
  const Real rho = std::sqrt(xs*xs + ys*ys);
  Real rf = std::sqrt(rho*rho + zs*zs);
  // r_f = 0 (a cell centre exactly on the puncture): any direction, at 1e-6 r_min [A]
  if (!(rf > 1.0e-6*t.rmin)) { rf = 1.0e-6*t.rmin; }
  const Real th = std::atan2(rho, zs);
  Real v0[kKnNV];
  KerrNodePolar(t, rf, th, v0, exact);
  Real c = 1.0, s = 0.0;
  if (rho > 0.0) { c = xs/rho; s = ys/rho; }
  KerrRotateNode(c, s, v0, v);
  if (m_bh != 1.0) {
    for (int q = 6; q < 12; ++q) { v[q] *= im; }
    for (int q = kKnDG; q < kKnDDG; ++q) { v[q] *= im; }
    for (int q = kKnDDG; q < kKnDA; ++q) { v[q] *= im*im; }   // d d g and d K
    for (int q = kKnDA; q < kKnNV; ++q) { v[q] *= im; }
  }
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

//! \fn ConformalFromPhys(g, dg, ddg, gt, gtu, dgt, ddgt)
//! \brief gt = det(g)^(-1/3) g and its first and second derivatives from those of g
//! (KerrConformal's formulas, below); ddg, ddgt [Sym(k, l)][ij].
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

//! \fn BlendNode(x, y, z, r1, r2, v)
//! \brief The D3 blend (SPIN_DESIGN_SPEC D3; K-003, RK-012): blend the conformal metric of
//! the node v (offset (x, y, z) from the hole, mesh units) toward flat, gt_b = M det(M)^(-1/3),
//! M = delta + w (gt - delta), with w = 0 for r <= r1, the quintic smootherstep
//! 6s^5 - 15s^4 + 10s^3 (s = (r - r1)/(r2 - r1)) between, 1 for r >= r2 (C2, analytic
//! d w and d d w). det g, so psi0, is kept: g_b = det(g)^(1/3) gt_b, with d and d d by the
//! product rule; K, alpha and beta are untouched. Only g, d g, d d g of v are rewritten.
KOKKOS_INLINE_FUNCTION
void BlendNode(Real x, Real y, Real z, Real r1, Real r2, Real *v) {
  using kerr_impl::Sym;
  const Real r = Kokkos::sqrt(x*x + y*y + z*z);
  if (!(r < r2)) return;
  Real w = 0.0, w1 = 0.0, w2 = 0.0;
  if (r > r1) {
    const Real L = r2 - r1, s = (r - r1)/L;
    w = s*s*s*(10.0 + s*(-15.0 + 6.0*s));
    w1 = 30.0*s*s*(1.0 - s)*(1.0 - s)/L;
    w2 = 60.0*s*(1.0 - s)*(1.0 - 2.0*s)/(L*L);
  }
  const Real X[3] = {x, y, z};
  Real dw[3] = {0.0, 0.0, 0.0}, ddw[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  if (r > r1 && r > 0.0) {
    const Real ir = 1.0/r;
    for (int k = 0; k < 3; ++k) {
      dw[k] = w1*X[k]*ir;
      for (int l = k; l < 3; ++l) {
        const Real nn = X[k]*X[l]*ir*ir;
        ddw[Sym(k, l)] = w2*nn + w1*ir*(((k == l) ? 1.0 : 0.0) - nn);
      }
    }
  }
  Real g[6], dg[3][6], ddg[6][6];
  for (int q = 0; q < 6; ++q) {
    g[q] = v[kKnF + q];
    for (int k = 0; k < 3; ++k) { dg[k][q] = v[kKnDG + 6*k + q]; }
    for (int p = 0; p < 6; ++p) { ddg[p][q] = v[kKnDDG + 6*p + q]; }
  }
  // L_k = d_k ln det g, L_kl = d_kl ln det g
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
  Real gt[6], gtu[6], dgt[3][6], ddgt[6][6];
  ConformalFromPhys(g, dg, ddg, gt, gtu, dgt, ddgt);
  Real M[6], dM[3][6], ddM[6][6];
  for (int q = 0; q < 6; ++q) {
    const Real h = gt[q] - ((q == 0 || q == 3 || q == 5) ? 1.0 : 0.0);
    M[q] = ((q == 0 || q == 3 || q == 5) ? 1.0 : 0.0) + w*h;
    for (int k = 0; k < 3; ++k) { dM[k][q] = dw[k]*h + w*dgt[k][q]; }
    for (int k = 0; k < 3; ++k) {
      for (int l = k; l < 3; ++l) {
        const int p = Sym(k, l);
        ddM[p][q] = ddw[p]*h + dw[k]*dgt[l][q] + dw[l]*dgt[k][q] + w*ddgt[p][q];
      }
    }
  }
  Real gb[6], gbu[6], dgb[3][6], ddgb[6][6];
  ConformalFromPhys(M, dM, ddM, gb, gbu, dgb, ddgb);
  const Real P = Kokkos::cbrt(D);
  Real dP[3];
  for (int k = 0; k < 3; ++k) { dP[k] = P*Lk[k]/3.0; }
  for (int q = 0; q < 6; ++q) {
    v[kKnF + q] = P*gb[q];
    for (int k = 0; k < 3; ++k) { v[kKnDG + 6*k + q] = dP[k]*gb[q] + P*dgb[k][q]; }
    for (int k = 0; k < 3; ++k) {
      for (int l = k; l < 3; ++l) {
        const int p = Sym(k, l);
        const Real ddP = P*(Lkl[k][l]/3.0 + Lk[k]*Lk[l]/9.0);
        v[kKnDDG + 6*p + q] = ddP*gb[q] + dP[k]*dgb[l][q] + dP[l]*dgb[k][q]
                              + P*ddgb[p][q];
      }
    }
  }
}

//! \struct KerrBgPoint
//! \brief The conformal background at a point (symmetric pairs packed xx xy xz yy yz zz).
struct KerrBgPoint {
  Real psi0, alpha0, beta0[3];
  Real gt[6], gtu[6];
  Real dgt[3][6];     // d_k gt_ij
  Real gam[3][6];     // Gt^k_ij
  Real rt[6], rs;     // Rt_ij, Rt
  Real dalpha0[3];
  Real dbeta0[3][3];  // d_k beta0^i  [k][i]
  Real dlnpsi0[3];
  Real k0[6], ahu[6], ahd[6];
};

//----------------------------------------------------------------------------------------
//! \fn void KerrConformal(const Real *v, KerrBgPoint &b)
//! \brief psi0 = det(g)^(1/12), gt = psi0^-4 g, and the rest from the spectral derivatives
//! (no finite differences): with L_k = g^ab d_k g_ab, L_kl = g^ab d_kl g_ab
//! - g^ac g^bd d_l g_cd d_k g_ab and e = det^(-1/3):
//!   d_k gt_ij = e (d_k g_ij - L_k g_ij/3),
//!   d_kl gt_ij = e (d_kl g_ij - L_l d_k g_ij/3 - L_k d_l g_ij/3 - L_kl g_ij/3
//!                   + L_k L_l g_ij/9),
//!   Gt^k_ij = gt^kl (d_i gt_lj + d_j gt_il - d_l gt_ij)/2, d_m Gt^k_ij likewise,
//!   Rt_ij = d_k Gt^k_ij - d_j Gt^k_ik + Gt^k_kl Gt^l_ij - Gt^k_jl Gt^l_ik, Rt = gt^ij Rt_ij,
//!   d_k ln psi0 = L_k/12, Ahat0^ij = psi0^10 g^ia g^jb K_ab, Ahat0_ij = psi0^2 K_ij.
KOKKOS_INLINE_FUNCTION
void KerrConformal(const Real *v, KerrBgPoint &b) {
  using namespace kerr_impl;  // NOLINT(build/namespaces)
  Real g[3][3], gi[3][3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) { g[i][j] = v[Sym(i, j)]; }
  }
  const Real D = g[0][0]*(g[1][1]*g[2][2] - g[1][2]*g[2][1])
               - g[0][1]*(g[1][0]*g[2][2] - g[1][2]*g[2][0])
               + g[0][2]*(g[1][0]*g[2][1] - g[1][1]*g[2][0]);
  gi[0][0] = (g[1][1]*g[2][2] - g[1][2]*g[2][1])/D;
  gi[0][1] = (g[0][2]*g[2][1] - g[0][1]*g[2][2])/D;
  gi[0][2] = (g[0][1]*g[1][2] - g[0][2]*g[1][1])/D;
  gi[1][0] = gi[0][1];
  gi[1][1] = (g[0][0]*g[2][2] - g[0][2]*g[2][0])/D;
  gi[1][2] = (g[0][2]*g[1][0] - g[0][0]*g[1][2])/D;
  gi[2][0] = gi[0][2];
  gi[2][1] = gi[1][2];
  gi[2][2] = (g[0][0]*g[1][1] - g[0][1]*g[1][0])/D;

  Real Lk[3], Lkl[3][3];
  for (int k = 0; k < 3; ++k) {
    Real a = 0.0;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) { a += gi[i][j]*v[kKnDG + 6*k + Sym(i, j)]; }
    }
    Lk[k] = a;
  }
  for (int k = 0; k < 3; ++k) {
    for (int l = k; l < 3; ++l) {
      Real a = 0.0;
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          a += gi[i][j]*v[kKnDDG + 6*Sym(k, l) + Sym(i, j)];
          for (int c = 0; c < 3; ++c) {
            for (int d = 0; d < 3; ++d) {
              a -= gi[i][c]*gi[j][d]*v[kKnDG + 6*l + Sym(c, d)]*v[kKnDG + 6*k + Sym(i, j)];
            }
          }
        }
      }
      Lkl[k][l] = a;
      Lkl[l][k] = a;
    }
  }
  const Real e = std::pow(D, -1.0/3.0);
  b.psi0 = std::pow(D, 1.0/12.0);
  const Real ie = 1.0/e;
  for (int q = 0; q < 6; ++q) {
    const int i = IdxI(q), j = IdxJ(q);
    b.gt[q] = e*g[i][j];
    b.gtu[q] = ie*gi[i][j];
    b.k0[q] = v[6 + q];
  }
  for (int k = 0; k < 3; ++k) { b.dlnpsi0[k] = Lk[k]/12.0; }
  // d gt and d d gt
  Real dgt[3][6], ddgt[6][6];
  for (int k = 0; k < 3; ++k) {
    for (int q = 0; q < 6; ++q) {
      const int i = IdxI(q), j = IdxJ(q);
      dgt[k][q] = e*(v[kKnDG + 6*k + q] - Lk[k]*g[i][j]/3.0);
      b.dgt[k][q] = dgt[k][q];
    }
  }
  for (int m = 0; m < 6; ++m) {
    const int k = IdxI(m), l = IdxJ(m);
    for (int q = 0; q < 6; ++q) {
      const int i = IdxI(q), j = IdxJ(q);
      ddgt[m][q] = e*(v[kKnDDG + 6*m + q] - Lk[l]*v[kKnDG + 6*k + q]/3.0
                      - Lk[k]*v[kKnDG + 6*l + q]/3.0 - Lkl[k][l]*g[i][j]/3.0
                      + Lk[k]*Lk[l]*g[i][j]/9.0);
    }
  }
  // Gamma_{l ij}, d_m Gamma_{l ij}; Gt^k_ij, d_m Gt^k_ij
  Real Gl[3][6], G[3][6], dG[3][3][6];
  for (int l = 0; l < 3; ++l) {
    for (int q = 0; q < 6; ++q) {
      const int i = IdxI(q), j = IdxJ(q);
      Gl[l][q] = 0.5*(dgt[i][Sym(l, j)] + dgt[j][Sym(i, l)] - dgt[l][q]);
    }
  }
  Real gtu[3][3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) { gtu[i][j] = b.gtu[Sym(i, j)]; }
  }
  for (int k = 0; k < 3; ++k) {
    for (int q = 0; q < 6; ++q) {
      G[k][q] = gtu[k][0]*Gl[0][q] + gtu[k][1]*Gl[1][q] + gtu[k][2]*Gl[2][q];
      b.gam[k][q] = G[k][q];
    }
  }
  for (int m = 0; m < 3; ++m) {
    // d_m gt^kl = -gt^ka d_m gt_ab gt^bl
    Real dgu[3][3];
    for (int k = 0; k < 3; ++k) {
      for (int l = 0; l < 3; ++l) {
        Real a = 0.0;
        for (int aa = 0; aa < 3; ++aa) {
          for (int bb = 0; bb < 3; ++bb) { a -= gtu[k][aa]*dgt[m][Sym(aa, bb)]*gtu[bb][l]; }
        }
        dgu[k][l] = a;
      }
    }
    for (int q = 0; q < 6; ++q) {
      const int i = IdxI(q), j = IdxJ(q);
      Real dGl[3];
      for (int l = 0; l < 3; ++l) {
        dGl[l] = 0.5*(ddgt[Sym(m, i)][Sym(l, j)] + ddgt[Sym(m, j)][Sym(i, l)]
                      - ddgt[Sym(m, l)][q]);
      }
      for (int k = 0; k < 3; ++k) {
        Real a = 0.0;
        for (int l = 0; l < 3; ++l) { a += dgu[k][l]*Gl[l][q] + gtu[k][l]*dGl[l]; }
        dG[m][k][q] = a;
      }
    }
  }
  b.rs = 0.0;
  for (int q = 0; q < 6; ++q) {
    const int i = IdxI(q), j = IdxJ(q);
    Real a = 0.0;
    for (int k = 0; k < 3; ++k) {
      a += dG[k][k][q] - dG[j][k][Sym(i, k)];
      for (int l = 0; l < 3; ++l) {
        a += G[k][Sym(k, l)]*G[l][q] - G[k][Sym(j, l)]*G[l][Sym(i, k)];
      }
    }
    b.rt[q] = a;
  }
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) { b.rs += gtu[i][j]*b.rt[Sym(i, j)]; }
  }
  // lapse, shift and their derivatives
  b.alpha0 = v[12];
  for (int i = 0; i < 3; ++i) { b.beta0[i] = v[13 + i]; }
  for (int k = 0; k < 3; ++k) {
    b.dalpha0[k] = v[kKnDA + k];
    for (int i = 0; i < 3; ++i) { b.dbeta0[k][i] = v[kKnDB + 3*k + i]; }
  }
  // Ahat0^ij = psi0^10 K^ij, Ahat0_ij = psi0^2 K_ij
  const Real p2 = b.psi0*b.psi0, p10 = p2*p2*p2*p2*p2;
  for (int q = 0; q < 6; ++q) {
    const int i = IdxI(q), j = IdxJ(q);
    Real a = 0.0;
    for (int aa = 0; aa < 3; ++aa) {
      for (int bb = 0; bb < 3; ++bb) { a += gi[i][aa]*gi[j][bb]*v[6 + Sym(aa, bb)]; }
    }
    b.ahu[q] = p10*a;
    b.ahd[q] = p2*v[6 + q];
  }
}

//----------------------------------------------------------------------------------------
//! \fn void FillKerrBackground(MeshBlockPack *pmbp, const KerrTable &t, Real m_bh,
//!                             const Real xbh[3], Real blend_r1, Real blend_r2,
//!                             DvceArray5D<Real> &bg)
//! \brief Every cell (ghosts included) of the pack: the KerrConformal quantities about the
//! hole at xbh, into bg(m, kKbg*, k, j, i) (bg needs kKbgN components). With blend_r2 > 0
//! the node's gt is first blended toward flat between blend_r1 and blend_r2 (mesh units
//! about xbh; BlendNode, D3), the same gt the Kerr MG coefficients use (kmg::CoefSource).
void FillKerrBackground(MeshBlockPack *pmbp, const KerrTable &t, Real m_bh,
                        const Real xbh[3], Real blend_r1, Real blend_r2,
                        DvceArray5D<Real> &bg);

}  // namespace cfc

#endif  // CFC_KERR_BACKGROUND_HPP_
