//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_cartoon_test.cpp
//! \brief Unit test of the Z4c Cartoon fill (M1, chat/0003 1.5 + chat/0004 1C).
//!
//! An analytic, exactly axisymmetric 3D field is written on the y = 0 plane (u0, every
//! cell with x > 0, x1/x2 ghosts included). Then
//!   1. the D-002 axis boundary fills x < 0 on u0          -> err_axis   (exact)
//!   2. CartoonFill (K1-K4) fills u_cart                    -> err_ghost  (interp order)
//!                                                            err_corner (x < 0, y != 0)
//!   3. the D-008 pi-copy is cross-checked against direct
//!      interpolation at rho for the x < 0 cells            -> err_xcheck (exact)
//!   4. d_y, d_yy, d_xy, d_yz at y = 0 by the Z4c FD
//!      operators on u_cart vs analytic derivatives         -> err_dy ... (FD order)
//! The field has a rotating part (beta^y, Gam^y, A_xy, A_yz, g_xy, g_yz all nonzero on
//! y = 0), so every entry of the parity table and of the rotation is exercised. g-tilde
//! and A-tilde satisfy det = 1 and tr = 0 exactly, so K4 (AlgC on the ghost planes) moves
//! the ghosts by no more than the interpolation error. Analytic derivatives use
//! hyper-dual numbers (exact first and mixed second derivatives).
//!
//! The exact checks (1, 3) are gates: the pgen exits nonzero if they fail, so a run with
//! a deliberately wrong parity entry (<problem> wrong_parity = <z4c variable name>) must
//! fail. Z4cCartesianPos is also checked against the analytic midpoints (R6).
//! Convergence orders are judged by the pytest driver from the CARTOON_RESULT
//! lines.
//! <problem> ntime > 0 also times K1-K4 against CalcRHS.
//!
//! Frame (D-017): physical (x, y, z) = mesh (x1, -x3, x2); symmetry axis = mesh x2.
//! Every field is built and checked in mesh components; the analytic state is
//! evaluated on the plane at mesh (x1, x2) = (R, z). AnalyticState takes the MESH
//! coordinates (x1, x3, x2) as its (x, y, z) arguments and returns mesh components: a
//! field in mesh coordinates, so its handedness does not matter to the fill.
//!
//! M2a (chat/0009) adds, after the fill checks:
//!   A. ADMToZ4c on rotating axisymmetric ADM data built from AnalyticState
//!      (g = psi^4 g-tilde, K = psi^4 (A-tilde + g-tilde Khat/3)):
//!        err_adm2z4c  chi, g-tilde, Khat, A-tilde on the whole plane    (exact, gate)
//!        err_gam      Gamma-tilde^i = -d_j g-tilde^ij in the interior   (FD order)
//!   B. Schwarzschild, isotropic (psi = 1 + M/2r, K = 0), and
//!   C. Kerr in Kerr-Schild coordinates, spin bh_spin along physical +z = mesh +x2,
//!      mapped to mesh components with D-017 (odd number of y indices -> minus sign);
//!      both centred on the axis at mesh x2 = bh_zc, outside the domain. ADMToZ4c, then
//!      ADMConstraints (Cartoon caches) on the exact ADM data:
//!        err_H_*, err_M_* (max |M_i|), err_Z_* (sqrt max Z) in the interior (FD order)
//!
//! M2f (chat/0010, chat/0013) adds the FastFlow Cartoon point evaluation
//! (z4c/fastflow_cartoon.hpp) on the same rotating field (g = g-tilde, K = A-tilde in
//! u_adm): at physical points with rho = f dx1 (f = 0.03 ... 10) and several azimuths,
//! the physical-frame g, K and d_k g against the analytic values and hyper-dual
//! derivatives:
//!        err_ffg, err_ffK           (interpolation order 2 ng)
//!        err_ffdg_far  (rho >= dx1), err_ffdg_axis (rho < dx1)   (order 2 ng - 1)

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "utils/finite_diff.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_pos.hpp"
#include "z4c/fastflow_cartoon.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {

using z4c::Z4c;
using z4c::Z4cCartesianPos;
constexpr Z4c::CartoonMode kAx = Z4c::CartoonMode::axisym;

//----------------------------------------------------------------------------------------
// hyper-dual numbers: v + a e1 + b e2 + ab e1 e2, e1^2 = e2^2 = 0
struct HD {
  Real v, a, b, ab;
  KOKKOS_INLINE_FUNCTION HD() : v(0), a(0), b(0), ab(0) {}
  KOKKOS_INLINE_FUNCTION HD(Real v_) : v(v_), a(0), b(0), ab(0) {}  // NOLINT
  KOKKOS_INLINE_FUNCTION HD(Real v_, Real a_, Real b_, Real ab_)
      : v(v_), a(a_), b(b_), ab(ab_) {}
};
KOKKOS_INLINE_FUNCTION HD operator+(const HD &x, const HD &y) {
  return HD(x.v+y.v, x.a+y.a, x.b+y.b, x.ab+y.ab);
}
KOKKOS_INLINE_FUNCTION HD operator-(const HD &x, const HD &y) {
  return HD(x.v-y.v, x.a-y.a, x.b-y.b, x.ab-y.ab);
}
KOKKOS_INLINE_FUNCTION HD operator-(const HD &x) { return HD(-x.v, -x.a, -x.b, -x.ab); }
KOKKOS_INLINE_FUNCTION HD operator*(const HD &x, const HD &y) {
  return HD(x.v*y.v, x.v*y.a + x.a*y.v, x.v*y.b + x.b*y.v,
            x.v*y.ab + x.a*y.b + x.b*y.a + x.ab*y.v);
}
// f(x) for a scalar function with f, f', f'' at x.v
KOKKOS_INLINE_FUNCTION HD Chain(const HD &x, Real f, Real f1, Real f2) {
  return HD(f, f1*x.a, f1*x.b, f1*x.ab + f2*x.a*x.b);
}
KOKKOS_INLINE_FUNCTION HD operator/(const HD &x, const HD &y) {
  const Real iv = 1.0/y.v;
  return x*Chain(y, iv, -iv*iv, 2.0*iv*iv*iv);
}
KOKKOS_INLINE_FUNCTION Real Exp(Real x) { return Kokkos::exp(x); }
KOKKOS_INLINE_FUNCTION HD Exp(const HD &x) {
  const Real e = Kokkos::exp(x.v);
  return Chain(x, e, e, e);
}
KOKKOS_INLINE_FUNCTION Real Cbrt(Real x) { return Kokkos::cbrt(x); }
KOKKOS_INLINE_FUNCTION HD Cbrt(const HD &x) {
  const Real c = Kokkos::cbrt(x.v);
  return Chain(x, c, c/(3.0*x.v), -2.0*c/(9.0*x.v*x.v));
}
KOKKOS_INLINE_FUNCTION Real Val(Real x) { return x; }
KOKKOS_INLINE_FUNCTION Real Val(const HD &x) { return x.v; }

// physical index (x=0, y=1, z=2) -> mesh component index (x=0, z=1, y=2)
KOKKOS_INLINE_FUNCTION int MeshComp(int p) { return p == 0 ? 0 : (p == 1 ? 2 : 1); }
// mesh component pair (a <= b) -> offset within a symmetric-tensor block
KOKKOS_INLINE_FUNCTION int SymOff(int a, int b) {
  if (a > b) { int t = a; a = b; b = t; }
  return a == 0 ? b : (a == 1 ? 2 + b : 5);
}

//----------------------------------------------------------------------------------------
// The analytic state at physical (x, y, z), in Z4c variable order (mesh components).
// Built only from rotation-equivariant fields R = (x, y, 0), phi = (-y, x, 0), zhat with
// coefficients that are functions of (rho^2, z): exactly axisymmetric and smooth.
template <typename T>
KOKKOS_INLINE_FUNCTION
void AnalyticState(const T &x, const T &y, const T &z, T *u) {
  const T rho2 = x*x + y*y;
  const T e1 = Exp(-(rho2 + z*z)*T(1.0/0.36));                  // w = 0.6
  const T e2 = Exp(-(rho2 + (z - T(0.3))*(z - T(0.3)))*T(1.0/0.49));  // w = 0.7
  const T R[3] = {x, y, T(0.0)};
  const T P[3] = {-y, x, T(0.0)};
  const T Z[3] = {T(0.0), T(0.0), T(1.0)};

  u[Z4c::I_Z4C_CHI]   = T(1.0) - T(0.3)*e1;
  u[Z4c::I_Z4C_KHAT]  = T(0.2)*e2*(T(1.0) + T(0.5)*z);
  u[Z4c::I_Z4C_THETA] = T(0.05)*e1*(rho2 - z);
  u[Z4c::I_Z4C_ALPHA] = T(1.0) - T(0.25)*e2;

  // vectors: b1 R + b2 phi + b3 zhat
  {
    const T b1 = T(0.3)*e1, b2 = T(0.4)*e2*(T(1.0) + z), b3 = T(0.2)*e1*(z - rho2);
    for (int p = 0; p < 3; ++p)
      u[Z4c::I_Z4C_BETAX + MeshComp(p)] = b1*R[p] + b2*P[p] + b3*Z[p];
    const T g1 = T(0.1)*e2, g2 = T(-0.2)*e1, g3 = T(0.15)*e2*z;
    for (int p = 0; p < 3; ++p)
      u[Z4c::I_Z4C_GAMX + MeshComp(p)] = g1*R[p] + g2*P[p] + g3*Z[p];
  }

  // symmetric tensors: c0 delta + c1 RR + c2 PP + c3 ZZ + c4 (RP+PR) + c5 (RZ+ZR)
  //                    + c6 (PZ+ZP)
  T M[3][3], N[3][3];
  {
    const T a0 = T(1.0), a1 = T(0.2)*e1, a2 = T(0.1)*e2, a3 = T(0.15)*e1,
            a4 = T(0.12)*e2, a5 = T(0.1)*e1*z, a6 = T(0.08)*e2;
    const T c0 = T(0.05)*e2, c1 = T(0.1)*e1, c2 = T(-0.07)*e2, c3 = T(0.05)*e1,
            c4 = T(0.09)*e1, c5 = T(0.06)*e2, c6 = T(0.11)*e1;
    for (int p = 0; p < 3; ++p)
    for (int q = 0; q < 3; ++q) {
      const T d = T(p == q ? 1.0 : 0.0);
      const T RR = R[p]*R[q], PP = P[p]*P[q], ZZ = Z[p]*Z[q];
      const T RP = R[p]*P[q] + P[p]*R[q], RZ = R[p]*Z[q] + Z[p]*R[q];
      const T PZ = P[p]*Z[q] + Z[p]*P[q];
      M[p][q] = a0*d + a1*RR + a2*PP + a3*ZZ + a4*RP + a5*RZ + a6*PZ;
      N[p][q] = c0*d + c1*RR + c2*PP + c3*ZZ + c4*RP + c5*RZ + c6*PZ;
    }
  }
  // g-tilde = M / det(M)^(1/3)  (det = 1)
  const T detM = M[0][0]*(M[1][1]*M[2][2] - M[1][2]*M[2][1])
               - M[0][1]*(M[1][0]*M[2][2] - M[1][2]*M[2][0])
               + M[0][2]*(M[1][0]*M[2][1] - M[1][1]*M[2][0]);
  const T s = T(1.0)/Cbrt(detM);
  T g[3][3], gi[3][3];
  for (int p = 0; p < 3; ++p)
  for (int q = 0; q < 3; ++q) g[p][q] = s*M[p][q];
  // inverse of g (det g = 1 analytically; divide anyway for consistency)
  const T detg = g[0][0]*(g[1][1]*g[2][2] - g[1][2]*g[2][1])
               - g[0][1]*(g[1][0]*g[2][2] - g[1][2]*g[2][0])
               + g[0][2]*(g[1][0]*g[2][1] - g[1][1]*g[2][0]);
  gi[0][0] = (g[1][1]*g[2][2] - g[1][2]*g[2][1])/detg;
  gi[0][1] = (g[0][2]*g[2][1] - g[0][1]*g[2][2])/detg;
  gi[0][2] = (g[0][1]*g[1][2] - g[0][2]*g[1][1])/detg;
  gi[1][1] = (g[0][0]*g[2][2] - g[0][2]*g[2][0])/detg;
  gi[1][2] = (g[0][2]*g[1][0] - g[0][0]*g[1][2])/detg;
  gi[2][2] = (g[0][0]*g[1][1] - g[0][1]*g[1][0])/detg;
  gi[1][0] = gi[0][1]; gi[2][0] = gi[0][2]; gi[2][1] = gi[1][2];
  // A-tilde = N - (1/3) g tr(g^-1 N)  (trace free)
  T tr = T(0.0);
  for (int p = 0; p < 3; ++p)
  for (int q = 0; q < 3; ++q) tr = tr + gi[p][q]*N[p][q];
  for (int p = 0; p < 3; ++p)
  for (int q = p; q < 3; ++q) {
    const int o = SymOff(MeshComp(p), MeshComp(q));
    u[Z4c::I_Z4C_GXX + o] = g[p][q];
    u[Z4c::I_Z4C_AXX + o] = N[p][q] - T(1.0/3.0)*g[p][q]*tr;
  }
}

// one Z4c variable of a 5D array as a 4D accessor for the finite_diff templates
struct VarN {
  DvceArray5D<Real> u;
  int n;
  KOKKOS_INLINE_FUNCTION Real operator()(int m, int k, int j, int i) const {
    return u(m,n,k,j,i);
  }
};

struct Errs {
  Real v[Z4c::nz4c];
};

// max over (selected cells) of |u_cart - analytic| per variable
// sel: 0 ghost planes x > 0 (all j, i), 1 ghost planes x > 0 interior (is..ie, js..je),
//      2 ghost planes x < 0, 3 plane 0 (all cells)
Errs GhostErrors(MeshBlockPack *pmbp, DvceArray5D<Real> uc, int sel) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int ng = indcs.ng, is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int n1 = uc.extent_int(4), n2 = uc.extent_int(3);
  const int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;
  const RegionIndcs indcs_ = pmbp->pmesh->mb_indcs;
  Errs e;
  for (int n = 0; n < Z4c::nz4c; ++n) {
    Real emax = 0.0;
    Kokkos::parallel_reduce("cartoon_test_ghost_err",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(DevExeSpace(), {0, 0, 0, 0},
                                             {nmb, 2*ng+1, n2, n1}),
    KOKKOS_LAMBDA(const int m, const int kc, const int j, const int i, Real &mx) {
      const Real dx = size.d_view(m).dx1;
      Real x, z, yp;
      Z4cCartesianPos(kAx, size.d_view(m), indcs_, 0, j, i, x, z,
                      yp);
      const int l = kc - ng;
      bool use = false;
      if (sel == 0) use = (l != 0 && x > 0.0);
      if (sel == 1) use = (l != 0 && x > 0.0 && i >= is && i <= ie && j >= js && j <= je);
      if (sel == 2) use = (l != 0 && x < 0.0);
      if (sel == 3) use = (l == 0);
      if (!use) return;
      Real a[Z4c::nz4c];
      AnalyticState<Real>(x, l*dx, z, a);
      mx = Kokkos::fmax(mx, Kokkos::fabs(uc(m,n,kc,j,i) - a[n]));
    }, Kokkos::Max<Real>(emax));
    e.v[n] = emax;
  }
  return e;
}

// max over the interior of the y = 0 plane of |FD derivative - analytic| per variable
// which: 0 d_y, 1 d_yy, 2 d_xy, 3 d_yz
template <int NG>
Errs DerivErrors(MeshBlockPack *pmbp, DvceArray5D<Real> uc, int which) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;
  const RegionIndcs indcs_ = pmbp->pmesh->mb_indcs;
  Errs e;
  for (int n = 0; n < Z4c::nz4c; ++n) {
    Real emax = 0.0;
    Kokkos::parallel_reduce("cartoon_test_deriv_err",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>(DevExeSpace(), {0, js, is},
                                             {nmb, je+1, ie+1}),
    KOKKOS_LAMBDA(const int m, const int j, const int i, Real &mx) {
      const Real dx = size.d_view(m).dx1, dz = size.d_view(m).dx2;
      Real x, z, yp;
      Z4cCartesianPos(kAx, size.d_view(m), indcs_, 0, j, i, x, z,
                      yp);
      const Real idx[3] = {1.0/dx, 1.0/dz, 1.0/dx};
      VarN q{uc, n};
      const int k = NG;   // y = 0 plane of u_cart
      Real fd;
      // seeds: e1 on the first direction, e2 on y
      HD X(x), Y(0.0), Z(z);
      if (which == 0) {
        fd = Dx<NG>(2, idx, q, m, k, j, i);
        Y = HD(0.0, 1, 1, 0);
      } else if (which == 1) {
        fd = Dxx<NG>(2, idx, q, m, k, j, i);
        Y = HD(0.0, 1, 1, 0);
      } else if (which == 2) {
        fd = Dxy<NG>(0, 2, idx, q, m, k, j, i);
        X = HD(x, 1, 0, 0); Y = HD(0.0, 0, 1, 0);
      } else {
        fd = Dxy<NG>(1, 2, idx, q, m, k, j, i);
        Z = HD(z, 1, 0, 0); Y = HD(0.0, 0, 1, 0);
      }
      HD a[Z4c::nz4c];
      AnalyticState<HD>(X, Y, Z, a);
      const Real ex = (which == 0) ? a[n].b : a[n].ab;
      mx = Kokkos::fmax(mx, Kokkos::fabs(fd - ex));
    }, Kokkos::Max<Real>(emax));
    e.v[n] = emax;
  }
  return e;
}

Real MaxOf(const Errs &e, int &nmax) {
  Real mx = -1.0; nmax = 0;
  for (int n = 0; n < Z4c::nz4c; ++n) if (e.v[n] > mx) { mx = e.v[n]; nmax = n; }
  return mx;
}

void Report(const char *key, const Errs &e) {
  int nmax;
  Real mx = MaxOf(e, nmax);
  if (global_variable::my_rank == 0) {
    std::printf("CARTOON_RESULT %s %.17e %s\n", key, mx, Z4c::Z4c_names[nmax]);
  }
}

template <int NG>
void RunDerivs(MeshBlockPack *pmbp, DvceArray5D<Real> uc) {
  Report("err_dy",  DerivErrors<NG>(pmbp, uc, 0));
  Report("err_dyy", DerivErrors<NG>(pmbp, uc, 1));
  Report("err_dxy", DerivErrors<NG>(pmbp, uc, 2));
  Report("err_dyz", DerivErrors<NG>(pmbp, uc, 3));
}

//----------------------------------------------------------------------------------------
// M2a helpers

// Inverse conformal metric g-tilde^ab of AnalyticState (same frame and the same metric
// construction), in mesh components.
template <typename T>
KOKKOS_INLINE_FUNCTION
void AnalyticConfInv(const T &x, const T &y, const T &z, T gm[3][3]) {
  const T rho2 = x*x + y*y;
  const T e1 = Exp(-(rho2 + z*z)*T(1.0/0.36));
  const T e2 = Exp(-(rho2 + (z - T(0.3))*(z - T(0.3)))*T(1.0/0.49));
  const T R[3] = {x, y, T(0.0)};
  const T P[3] = {-y, x, T(0.0)};
  const T Z[3] = {T(0.0), T(0.0), T(1.0)};
  const T a0 = T(1.0), a1 = T(0.2)*e1, a2 = T(0.1)*e2, a3 = T(0.15)*e1,
          a4 = T(0.12)*e2, a5 = T(0.1)*e1*z, a6 = T(0.08)*e2;
  T M[3][3];
  for (int p = 0; p < 3; ++p)
  for (int q = 0; q < 3; ++q) {
    const T d = T(p == q ? 1.0 : 0.0);
    const T RR = R[p]*R[q], PP = P[p]*P[q], ZZ = Z[p]*Z[q];
    const T RP = R[p]*P[q] + P[p]*R[q], RZ = R[p]*Z[q] + Z[p]*R[q];
    const T PZ = P[p]*Z[q] + Z[p]*P[q];
    M[p][q] = a0*d + a1*RR + a2*PP + a3*ZZ + a4*RP + a5*RZ + a6*PZ;
  }
  const T detM = M[0][0]*(M[1][1]*M[2][2] - M[1][2]*M[2][1])
               - M[0][1]*(M[1][0]*M[2][2] - M[1][2]*M[2][0])
               + M[0][2]*(M[1][0]*M[2][1] - M[1][1]*M[2][0]);
  // g-tilde = M det(M)^(-1/3)  =>  g-tilde^-1 = det(M)^(1/3) M^-1
  const T s = Cbrt(detM)/detM;
  T gi[3][3];
  gi[0][0] = s*(M[1][1]*M[2][2] - M[1][2]*M[2][1]);
  gi[0][1] = s*(M[0][2]*M[2][1] - M[0][1]*M[2][2]);
  gi[0][2] = s*(M[0][1]*M[1][2] - M[0][2]*M[1][1]);
  gi[1][1] = s*(M[0][0]*M[2][2] - M[0][2]*M[2][0]);
  gi[1][2] = s*(M[0][2]*M[1][0] - M[0][0]*M[1][2]);
  gi[2][2] = s*(M[0][0]*M[1][1] - M[0][1]*M[1][0]);
  gi[1][0] = gi[0][1]; gi[2][0] = gi[0][2]; gi[2][1] = gi[1][2];
  for (int p = 0; p < 3; ++p)
  for (int q = 0; q < 3; ++q) gm[MeshComp(p)][MeshComp(q)] = gi[p][q];
}

// Exact vacuum ADM data at mesh (x1, x2, x3) in mesh components (D-017): which = 0
// Schwarzschild isotropic, 1 Kerr-Schild (M = 1, spin a along physical +z = mesh +x2).
// Black hole on the axis at mesh x2 = zc. Out: g[6], K[6] (mesh order 00 01 02 11 12 22),
// psi4. flip_k (negative control): K's odd-y components keep the physical sign, an
// inconsistent D-017 mapping that is no longer a solution.
KOKKOS_INLINE_FUNCTION
void ExactVacuum(const int which, const Real a, const Real zc, const Real x1,
                 const Real x2, const Real x3, const bool flip_k, Real *g, Real *K,
                 Real &psi4) {
  const Real x = x1, y = -x3, z = x2 - zc;   // physical, centred
  if (which == 0) {
    const Real r = Kokkos::sqrt(x*x + y*y + z*z);
    const Real psi = 1.0 + 0.5/r;
    psi4 = psi*psi*psi*psi;
    for (int n = 0; n < 6; ++n) { g[n] = 0.0; K[n] = 0.0; }
    g[0] = psi4; g[3] = psi4; g[5] = psi4;
    return;
  }
  Real alp, bx, by, bz, gxx, gxy, gxz, gyy, gyz, gzz, kxx, kxy, kxz, kyy, kyz, kzz;
  ComputeADMDecomposition(x, y, z, false, a, &alp, &bx, &by, &bz, &psi4,
                          &gxx, &gxy, &gxz, &gyy, &gyz, &gzz,
                          &kxx, &kxy, &kxz, &kyy, &kyz, &kzz);
  // mesh (x1, x2, x3) = physical (x, z, -y): a component gets one minus sign per y index
  g[0] = gxx; g[1] = gxz; g[2] = -gxy; g[3] = gzz; g[4] = -gyz; g[5] = gyy;
  K[0] = kxx; K[1] = kxz; K[2] = -kxy; K[3] = kzz; K[4] = -kyz; K[5] = kyy;
  if (flip_k) { K[2] = -K[2]; K[4] = -K[4]; }
}

// u0 = 0 except alpha = 1 (alpha, beta alias u0 under <z4c>), so Theta = 0 and the
// Gamma ghosts are finite
void ResetZ4c(MeshBlockPack *pmbp) {
  auto u0 = pmbp->pz4c->u0;
  Kokkos::deep_copy(u0, 0.0);
  auto alpha = Kokkos::subview(u0, Kokkos::ALL, static_cast<int>(Z4c::I_Z4C_ALPHA),
                               Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(alpha, 1.0);
}

template <int NG>
void RunADMToZ4c(MeshBlockPack *pmbp, ParameterInput *pin) {
  pmbp->pz4c->ADMToZ4c<NG>(pmbp, pin);
}
template <int NG>
void RunADMConstraints(MeshBlockPack *pmbp) {
  pmbp->pz4c->ADMConstraints<NG>(pmbp);
}
void CallADMToZ4c(int ng, MeshBlockPack *pmbp, ParameterInput *pin) {
  switch (ng) {
    case 2: RunADMToZ4c<2>(pmbp, pin); break;
    case 3: RunADMToZ4c<3>(pmbp, pin); break;
    case 4: RunADMToZ4c<4>(pmbp, pin); break;
  }
}
void CallADMConstraints(int ng, MeshBlockPack *pmbp) {
  switch (ng) {
    case 2: RunADMConstraints<2>(pmbp); break;
    case 3: RunADMConstraints<3>(pmbp); break;
    case 4: RunADMConstraints<4>(pmbp); break;
  }
}

// A: rotating analytic data through ADMToZ4c. Returns err_adm2z4c (exact); reports
// err_adm2z4c and err_gam.
Real TestADMToZ4c(MeshBlockPack *pmbp, ParameterInput *pin) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int ng = indcs.ng, is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks;
  const int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;
  const RegionIndcs indcs_ = indcs;
  auto u_adm = pmbp->padm->u_adm;
  auto u0 = pmbp->pz4c->u0;
  const int n1 = u0.extent_int(4), n2 = u0.extent_int(3);
  const Real p = pmbp->pz4c->opt.chi_psi_power;
  using adm::ADM;

  ResetZ4c(pmbp);
  par_for("cartoon_test_m2a_admA", DevExeSpace(), 0, nmb-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int j, const int i) {
    Real x, z, yp;
    Z4cCartesianPos(kAx, size.d_view(m), indcs_, 0, j, i, x, z, yp);
    Real a[Z4c::nz4c];
    AnalyticState<Real>(x, 0.0, z, a);
    const Real psi4 = Kokkos::pow(a[Z4c::I_Z4C_CHI], 4.0/p);
    for (int o = 0; o < 6; ++o) {
      u_adm(m,ADM::I_ADM_GXX+o,ks,j,i) = psi4*a[Z4c::I_Z4C_GXX+o];
      u_adm(m,ADM::I_ADM_KXX+o,ks,j,i) = psi4*(a[Z4c::I_Z4C_AXX+o] +
                                         a[Z4c::I_Z4C_GXX+o]*a[Z4c::I_Z4C_KHAT]/3.0);
    }
    u_adm(m,ADM::I_ADM_PSI4,ks,j,i) = psi4;
  });
  CallADMToZ4c(ng, pmbp, pin);

  // pointwise part: chi, g-tilde, Khat, A-tilde everywhere on the plane
  Real ept = 0.0;
  Kokkos::parallel_reduce("cartoon_test_m2a_pt",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>(DevExeSpace(), {0, 0, 0}, {nmb, n2, n1}),
  KOKKOS_LAMBDA(const int m, const int j, const int i, Real &mx) {
    Real x, z, yp;
    Z4cCartesianPos(kAx, size.d_view(m), indcs_, 0, j, i, x, z, yp);
    Real a[Z4c::nz4c];
    AnalyticState<Real>(x, 0.0, z, a);
    Real e = Kokkos::fabs(u0(m,Z4c::I_Z4C_CHI,ks,j,i) - a[Z4c::I_Z4C_CHI]);
    e = Kokkos::fmax(e, Kokkos::fabs(u0(m,Z4c::I_Z4C_KHAT,ks,j,i) - a[Z4c::I_Z4C_KHAT]));
    for (int o = 0; o < 6; ++o) {
      e = Kokkos::fmax(e, Kokkos::fabs(u0(m,Z4c::I_Z4C_GXX+o,ks,j,i) -
                                       a[Z4c::I_Z4C_GXX+o]));
      e = Kokkos::fmax(e, Kokkos::fabs(u0(m,Z4c::I_Z4C_AXX+o,ks,j,i) -
                                       a[Z4c::I_Z4C_AXX+o]));
    }
    mx = Kokkos::fmax(mx, e);
  }, Kokkos::Max<Real>(ept));

  // Gamma-tilde^a = -sum_b d_b g-tilde^ab, b over mesh directions x1 (x), x2 (z), x3 (y)
  Real egam = 0.0;
  Kokkos::parallel_reduce("cartoon_test_m2a_gam",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>(DevExeSpace(), {0, js, is},
                                           {nmb, je+1, ie+1}),
  KOKKOS_LAMBDA(const int m, const int j, const int i, Real &mx) {
    Real x, z, yp;
    Z4cCartesianPos(kAx, size.d_view(m), indcs_, 0, j, i, x, z, yp);
    Real gam[3] = {0.0, 0.0, 0.0};
    for (int b = 0; b < 3; ++b) {
      HD X(x), Y(0.0), Zc(z);
      if (b == 0) X = HD(x, 1, 0, 0);
      if (b == 1) Zc = HD(z, 1, 0, 0);
      if (b == 2) Y = HD(0.0, 1, 0, 0);
      HD gm[3][3];
      AnalyticConfInv<HD>(X, Y, Zc, gm);
      for (int c = 0; c < 3; ++c) gam[c] -= gm[c][b].a;
    }
    Real e = 0.0;
    for (int c = 0; c < 3; ++c) {
      e = Kokkos::fmax(e, Kokkos::fabs(u0(m,Z4c::I_Z4C_GAMX+c,ks,j,i) - gam[c]));
    }
    mx = Kokkos::fmax(mx, e);
  }, Kokkos::Max<Real>(egam));

  if (global_variable::my_rank == 0) {
    std::printf("CARTOON_RESULT err_adm2z4c %.17e all\n", ept);
    std::printf("CARTOON_RESULT err_gam %.17e all\n", egam);
  }
  return ept;
}

// B/C: exact vacuum data -> ADMToZ4c -> ADMConstraints; report H, M_i, Z
void TestConstraints(MeshBlockPack *pmbp, ParameterInput *pin, const int which,
                     const char *tag) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int ng = indcs.ng, is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks;
  const int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;
  const RegionIndcs indcs_ = indcs;
  auto u_adm = pmbp->padm->u_adm;
  const int n1 = u_adm.extent_int(4), n2 = u_adm.extent_int(3);
  const Real spin = pin->GetOrAddReal("problem", "bh_spin", 0.6);
  const Real zc = pin->GetOrAddReal("problem", "bh_zc", -4.0);
  const bool flip_k = pin->GetOrAddBoolean("problem", "kerr_flip_k_odd", false);
  using adm::ADM;

  ResetZ4c(pmbp);
  par_for("cartoon_test_m2a_vac", DevExeSpace(), 0, nmb-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int j, const int i) {
    Real x1, x2, x3;
    Z4cCartesianPos(kAx, size.d_view(m), indcs_, 0, j, i, x1, x2, x3);
    Real g[6], K[6], psi4;
    ExactVacuum(which, spin, zc, x1, x2, x3, flip_k, g, K, psi4);
    for (int o = 0; o < 6; ++o) {
      u_adm(m,ADM::I_ADM_GXX+o,ks,j,i) = g[o];
      u_adm(m,ADM::I_ADM_KXX+o,ks,j,i) = K[o];
    }
    u_adm(m,ADM::I_ADM_PSI4,ks,j,i) = psi4;
  });
  CallADMToZ4c(ng, pmbp, pin);
  CallADMConstraints(ng, pmbp);

  auto con = pmbp->pz4c->con;
  Real e[3] = {0.0, 0.0, 0.0};   // |H|, |M_i|, Z
  for (int q = 0; q < 3; ++q) {
    Kokkos::parallel_reduce("cartoon_test_m2a_con",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>(DevExeSpace(), {0, js, is},
                                             {nmb, je+1, ie+1}),
    KOKKOS_LAMBDA(const int m, const int j, const int i, Real &mx) {
      Real v;
      if (q == 0) {
        v = Kokkos::fabs(con.H(m,ks,j,i));
      } else if (q == 1) {
        v = 0.0;
        for (int a = 0; a < 3; ++a) {
          v = Kokkos::fmax(v, Kokkos::fabs(con.M_d(m,a,ks,j,i)));
        }
      } else {
        v = con.Z(m,ks,j,i);
      }
      mx = Kokkos::fmax(mx, v);
    }, Kokkos::Max<Real>(e[q]));
  }
  const Real eh = e[0], em = e[1], ez = e[2];
  if (global_variable::my_rank == 0) {
    std::printf("CARTOON_RESULT err_H_%s %.17e all\n", tag, eh);
    std::printf("CARTOON_RESULT err_M_%s %.17e all\n", tag, em);
    std::printf("CARTOON_RESULT err_Z_%s %.17e all\n", tag, std::sqrt(ez));
  }
}

// M2f: FastFlow Cartoon point evaluation against the rotating analytic field
template <int NG>
void TestFastFlowPoint(MeshBlockPack *pmbp) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int ks = indcs.ks;
  const int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;
  const RegionIndcs indcs_ = indcs;
  auto u_adm = pmbp->padm->u_adm;
  const int n1 = u_adm.extent_int(4), n2 = u_adm.extent_int(3);
  using adm::ADM;
  // u_adm on the whole plane (x1 ghosts included): g = g-tilde, K = A-tilde (mesh comps)
  par_for("cartoon_test_m2f_fill", DevExeSpace(), 0, nmb-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int j, const int i) {
    Real x, z, yp;
    Z4cCartesianPos(kAx, size.d_view(m), indcs_, 0, j, i, x, z, yp);
    Real a[Z4c::nz4c];
    AnalyticState<Real>(x, 0.0, z, a);
    for (int o = 0; o < 6; ++o) {
      u_adm(m,ADM::I_ADM_GXX+o,ks,j,i) = a[Z4c::I_Z4C_GXX+o];
      u_adm(m,ADM::I_ADM_KXX+o,ks,j,i) = a[Z4c::I_Z4C_AXX+o];
    }
    u_adm(m,ADM::I_ADM_PSI4,ks,j,i) = 1.0;
  });
  const int gind[6] = {ADM::I_ADM_GXX, ADM::I_ADM_GXY, ADM::I_ADM_GXZ,
                       ADM::I_ADM_GYY, ADM::I_ADM_GYZ, ADM::I_ADM_GZZ};
  const int Kind[6] = {ADM::I_ADM_KXX, ADM::I_ADM_KXY, ADM::I_ADM_KXZ,
                       ADM::I_ADM_KYY, ADM::I_ADM_KYZ, ADM::I_ADM_KZZ};
  const Real dx1 = size.h_view(0).dx1;
  const Real x3plane = 0.5*(pmbp->pmesh->mesh_size.x3min + pmbp->pmesh->mesh_size.x3max);
  const Real zlo = pmbp->pmesh->mesh_size.x2min, zhi = pmbp->pmesh->mesh_size.x2max;
  // points: rho = f dx1, 8 azimuths, 3 heights
  constexpr int NF = 9, NPHI = 8, NZ = 3;
  const Real fr[NF] = {0.03, 0.1, 0.3, 0.6, 0.99, 1.0, 1.5, 3.0, 10.0};
  Real e[4] = {0.0, 0.0, 0.0, 0.0};   // g, K, dg far, dg axis
  for (int q = 0; q < 4; ++q) {
    Kokkos::parallel_reduce("cartoon_test_m2f_pt",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, NF*NPHI*NZ),
    KOKKOS_LAMBDA(const int n, Real &mx) {
      const int f = n % NF, ip = (n/NF) % NPHI, iz = n/(NF*NPHI);
      const Real rho = fr[f]*dx1;
      const Real ph = 0.37 + 2.0*M_PI*ip/NPHI;
      const Real X1 = rho*Kokkos::cos(ph), X3 = rho*Kokkos::sin(ph);
      const Real X2 = zlo + (0.27 + 0.23*iz)*(zhi - zlo);
      Real gp[6], Kp[6], dgp[18];
      if (!z4c::CartoonFastFlowPoint<NG>(u_adm, gind, Kind, indcs_, size, nmb, x3plane,
                                         X1, -X3, X2, gp, Kp, dgp)) return;
      // analytic: AnalyticState takes (x1, x3, x2) and returns mesh components
      const int mc[3] = {0, 2, 1};     // physical index -> mesh index
      const Real sg[3] = {1.0, -1.0, 1.0};   // physical y = -mesh x3
      const int pa[6] = {0, 0, 0, 1, 1, 2}, pb[6] = {0, 1, 2, 1, 2, 2};
      Real v = 0.0;
      if (q < 2) {
        Real a[Z4c::nz4c];
        AnalyticState<Real>(X1, X3, X2, a);
        const int base = (q == 0) ? Z4c::I_Z4C_GXX : Z4c::I_Z4C_AXX;
        for (int c = 0; c < 6; ++c) {
          const int i = pa[c], j = pb[c];
          const Real an = sg[i]*sg[j]*a[base + SymOff(mc[i], mc[j])];
          v = Kokkos::fmax(v, Kokkos::fabs(((q == 0) ? gp[c] : Kp[c]) - an));
        }
      } else {
        if ((q == 2) != (fr[f] >= 1.0)) return;
        for (int k = 0; k < 3; ++k) {     // physical derivative direction
          // d/d(phys k) = sg[k] d/d(mesh mc[k]); AnalyticState args (x1, x3, x2)
          HD A1(X1), A3(X3), A2(X2);
          if (mc[k] == 0) A1 = HD(X1, 1, 0, 0);
          if (mc[k] == 1) A2 = HD(X2, 1, 0, 0);
          if (mc[k] == 2) A3 = HD(X3, 1, 0, 0);
          HD a[Z4c::nz4c];
          AnalyticState<HD>(A1, A3, A2, a);
          for (int c = 0; c < 6; ++c) {
            const int i = pa[c], j = pb[c];
            const Real an = sg[k]*sg[i]*sg[j]*a[Z4c::I_Z4C_GXX + SymOff(mc[i], mc[j])].a;
            v = Kokkos::fmax(v, Kokkos::fabs(dgp[6*k + c] - an));
          }
        }
      }
      mx = Kokkos::fmax(mx, v);
    }, Kokkos::Max<Real>(e[q]));
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, e, 4, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  if (global_variable::my_rank == 0) {
    std::printf("CARTOON_RESULT err_ffg %.17e all\n", e[0]);
    std::printf("CARTOON_RESULT err_ffK %.17e all\n", e[1]);
    std::printf("CARTOON_RESULT err_ffdg_far %.17e all\n", e[2]);
    std::printf("CARTOON_RESULT err_ffdg_axis %.17e all\n", e[3]);
  }
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::Z4cCartoonTest()

void ProblemGenerator::Z4cCartoonTest(ParameterInput *pin, const bool restart) {
  if (restart) return;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  z4c::Z4c *pz4c = pmbp->pz4c;
  if (pz4c == nullptr || !pz4c->IsCartoon()) {
    std::cout << "### FATAL ERROR z4c_cartoon_test requires <z4c> on a 2D "
              << "coord = cylindrical_axisym mesh (2D Cartoon)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto &indcs = pmy_mesh_->mb_indcs;
  const int ng = indcs.ng, is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nmb = pmbp->nmb_thispack;
  const int npt = pin->GetOrAddInteger("problem", "npt", 2*ng + 1);
  const Real tol_exact = pin->GetOrAddReal("problem", "tol_exact", 1.0e-12);
  const std::string wrong = pin->GetOrAddString("problem", "wrong_parity", "none");
  const int ntime = pin->GetOrAddInteger("problem", "ntime", 0);

  Z4c::CartoonParity par = Z4c::CartoonParityTable();
  if (wrong != "none") {
    int nw = -1;
    for (int n = 0; n < Z4c::nz4c; ++n) {
      if (wrong == Z4c::Z4c_names[n]) nw = n;
    }
    if (nw < 0) {
      std::cout << "### FATAL ERROR unknown wrong_parity '" << wrong << "'" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    par.p[nw] = -par.p[nw];
    if (global_variable::my_rank == 0) {
      std::printf("CARTOON_TEST deliberately WRONG parity for %s: %+d\n",
                  Z4c::Z4c_names[nw], par.p[nw]);
    }
  }

  // 1. analytic field on the plane, x > 0 only; x < 0 gets a sentinel
  auto u0 = pz4c->u0;
  auto &size = pmbp->pmb->mb_size;
  const RegionIndcs indcs_ = pmbp->pmesh->mb_indcs;
  const int n1 = u0.extent_int(4), n2 = u0.extent_int(3);
  par_for("cartoon_test_init", DevExeSpace(), 0, nmb-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int j, const int i) {
    Real x, z, yp;
    Z4cCartesianPos(kAx, size.d_view(m), indcs_, 0, j, i, x, z,
                    yp);
    Real a[Z4c::nz4c];
    AnalyticState<Real>(x, 0.0, z, a);
    for (int n = 0; n < Z4c::nz4c; ++n) u0(m,n,ks,j,i) = (x > 0.0) ? a[n] : 1.0e30;
  });

  // 2. axis boundary on u0, then compare the x < 0 cells with the analytic field
  pz4c->CartoonAxisBC(pmbp, u0, par);
  Errs eaxis;
  for (int n = 0; n < Z4c::nz4c; ++n) {
    Real emax = 0.0;
    Kokkos::parallel_reduce("cartoon_test_axis_err",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>(DevExeSpace(), {0, 0, 0}, {nmb, n2, n1}),
    KOKKOS_LAMBDA(const int m, const int j, const int i, Real &mx) {
      Real x, z, yp;
      Z4cCartesianPos(kAx, size.d_view(m), indcs_, 0, j, i, x, z,
                      yp);
      if (x > 0.0) return;
      Real a[Z4c::nz4c];
      AnalyticState<Real>(x, 0.0, z, a);
      mx = Kokkos::fmax(mx, Kokkos::fabs(u0(m,n,ks,j,i) - a[n]));
    }, Kokkos::Max<Real>(emax));
    eaxis.v[n] = emax;
  }

  // 3. fill with the pi-copy for x < 0 (production path), keep a copy, then the direct
  //    interpolation at rho for x < 0, and compare the two on the x < 0 ghost cells
  pz4c->CartoonFill(pmbp, npt, par, true);
  DvceArray5D<Real> udirect("cartoon_udirect", pz4c->u_cart.extent(0),
                            pz4c->u_cart.extent(1), pz4c->u_cart.extent(2),
                            pz4c->u_cart.extent(3), pz4c->u_cart.extent(4));
  Kokkos::deep_copy(udirect, pz4c->u_cart);
  pz4c->CartoonFill(pmbp, npt, par, false);
  auto uc = pz4c->u_cart;
  Real exc = 0.0;
  Kokkos::parallel_reduce("cartoon_test_xcheck",
    Kokkos::MDRangePolicy<Kokkos::Rank<5>>(DevExeSpace(), {0, 0, 0, 0, 0},
                                           {nmb, Z4c::nz4c, 2*ng+1, n2, n1}),
  KOKKOS_LAMBDA(const int m, const int n, const int kc, const int j, const int i,
                Real &mx) {
    Real x, z, yp;
    Z4cCartesianPos(kAx, size.d_view(m), indcs_, 0, j, i, x, z,
                    yp);
    if (x > 0.0 || kc == ng) return;
    mx = Kokkos::fmax(mx, Kokkos::fabs(uc(m,n,kc,j,i) - udirect(m,n,kc,j,i)));
  }, Kokkos::Max<Real>(exc));

  // R6: Z4cCartesianPos returns mesh order (R, z, 0) at the arithmetic midpoints (never
  // the centroids); D-017
  Real epos = 0.0;
  Kokkos::parallel_reduce("cartoon_test_pos",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>(DevExeSpace(), {0, 0, 0}, {nmb, n2, n1}),
  KOKKOS_LAMBDA(const int m, const int j, const int i, Real &mx) {
    Real x, z, yp;
    Z4cCartesianPos(kAx, size.d_view(m), indcs_, 0, j, i, x, z, yp);
    const Real xm = size.d_view(m).x1min + (i - is + 0.5)*size.d_view(m).dx1;
    const Real zm = size.d_view(m).x2min + (j - js + 0.5)*size.d_view(m).dx2;
    mx = Kokkos::fmax(mx, Kokkos::fabs(x - xm) + Kokkos::fabs(yp) + Kokkos::fabs(z - zm));
  }, Kokkos::Max<Real>(epos));

  // 4. report
  std::vector<std::array<int, 4>> clamps;
  pz4c->CartoonClampStats(pmbp, npt, clamps);
  Real chk = 0.0;
  Kokkos::parallel_reduce("cartoon_test_checksum",
    Kokkos::MDRangePolicy<Kokkos::Rank<5>>(DevExeSpace(), {0, 0, 0, 0, 0},
                                           {nmb, Z4c::nz4c, 2*ng+1, n2, n1}),
  KOKKOS_LAMBDA(const int m, const int n, const int kc, const int j, const int i,
                Real &sum) {
    sum += Kokkos::fabs(uc(m,n,kc,j,i));
  }, chk);

  if (global_variable::my_rank == 0) {
    std::printf("CARTOON_RESULT ng %d\nCARTOON_RESULT npt %d\nCARTOON_RESULT nx1 %d\n",
                ng, npt, pmy_mesh_->mesh_indcs.nx1);
    std::printf("CARTOON_RESULT dx %.17e\n", size.h_view(0).dx1);
    std::printf("CARTOON_RESULT checksum %.17e\n", chk);
    std::printf("CARTOON_RESULT err_xcheck %.17e all\n", exc);
    std::printf("CARTOON_RESULT err_pos %.17e all\n", epos);
    for (auto &c : clamps) {
      std::printf("CARTOON_CLAMP l=%d low=%d high=%d total=%d\n", c[0], c[1], c[2], c[3]);
    }
  }
  Report("err_axis", eaxis);
  Report("err_plane0", GhostErrors(pmbp, uc, 3));
  Report("err_ghost", GhostErrors(pmbp, uc, 0));
  Report("err_ghost_int", GhostErrors(pmbp, uc, 1));
  Report("err_corner", GhostErrors(pmbp, uc, 2));
  switch (ng) {
    case 2: RunDerivs<2>(pmbp, uc); break;
    case 3: RunDerivs<3>(pmbp, uc); break;
    case 4: RunDerivs<4>(pmbp, uc); break;
  }

  // 5. optional timing on the production path: K1-K4 vs CalcRHS (which includes a fill)
  if (ntime > 0) {
    Kokkos::Timer timer;
    Kokkos::fence();
    timer.reset();
    for (int r = 0; r < ntime; ++r) pz4c->CartoonFill(pmbp);
    Kokkos::fence();
    const Real tfill = timer.seconds()/ntime;
    timer.reset();
    for (int r = 0; r < ntime; ++r) {
      switch (ng) {
        case 2: pz4c->CalcRHS<2>(nullptr, 0); break;
        case 3: pz4c->CalcRHS<3>(nullptr, 0); break;
        case 4: pz4c->CalcRHS<4>(nullptr, 0); break;
      }
    }
    Kokkos::fence();
    const Real trhs = timer.seconds()/ntime;
    if (global_variable::my_rank == 0) {
      std::printf("CARTOON_RESULT t_fill_ms %.6e\nCARTOON_RESULT t_calcrhs_ms %.6e\n",
                  1e3*tfill, 1e3*trhs);
      std::printf("CARTOON_RESULT fill_over_rhs_without_fill %.6e\n",
                  tfill/(trhs - tfill));
    }
  }

  // 6. M2a: ADMToZ4c and the ADM constraints under Cartoon (production fill)
  Real eadm = 0.0;
  if (pin->GetOrAddBoolean("problem", "m2a", true)) {
    eadm = TestADMToZ4c(pmbp, pin);
    TestConstraints(pmbp, pin, 0, "schw");
    TestConstraints(pmbp, pin, 1, "kerr");
  }

  // 7. M2f: FastFlow Cartoon point evaluation (near-axis included)
  if (pin->GetOrAddBoolean("problem", "m2f", true)) {
    switch (ng) {
      case 2: TestFastFlowPoint<2>(pmbp); break;
      case 3: TestFastFlowPoint<3>(pmbp); break;
      case 4: TestFastFlowPoint<4>(pmbp); break;
    }
  }

  // gates: the exact checks
  int nmax;
  const Real ea = MaxOf(eaxis, nmax);
  bool ok = (ea <= tol_exact) && (exc <= tol_exact) && (epos <= tol_exact) &&
            (eadm <= tol_exact);
  if (global_variable::my_rank == 0) {
    std::printf("CARTOON_TEST %s err_axis=%.3e (%s) err_xcheck=%.3e err_pos=%.3e "
                "err_adm2z4c=%.3e tol=%.1e\n", ok ? "PASS" : "FAIL", ea,
                Z4c::Z4c_names[nmax], exc, epos, eadm, tol_exact);
  }
  if (!ok) std::exit(EXIT_FAILURE);
}
