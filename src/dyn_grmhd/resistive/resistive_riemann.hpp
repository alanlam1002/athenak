#ifndef DYN_GRMHD_RESISTIVE_RESISTIVE_RIEMANN_HPP_
#define DYN_GRMHD_RESISTIVE_RESISTIVE_RIEMANN_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file resistive_riemann.hpp
//! \brief Tetrad MHLLC face flux for resistive GRMHD (07_scoping_v2 §1 decision 4,
//!        §3.4; 05 §2.1; Mignone et al. 2019 §4).
//!
//! At each face an orthonormal triad {e_0, e_1, e_2} of gamma_ij is built with e_0 the
//! unit normal (e_0^i = gamma^{di}/sqrt(gamma^{dd})) and e_1, e_2 by Gram-Schmidt from
//! the next two coordinate axes, so e_1^d = e_2^d = 0 and the triad is right-handed.
//! With the normal observer as e_0hat the states are SR states (no boost). The SR
//! MHLLC (outer exact Maxwell solver, Eq. 70; inner HLLC of Mignone & Bodo 2005 on the
//! hydro-only part, Eqs. 72-78) is solved in the triad and sampled at
//! xi = beta^d/a, a = alpha sqrt(gamma^{dd}); the coordinate flux is
//!   F^d = sqrt(gamma) [ a R(H_tet) - beta^d R(U_tet) ],
//! where R maps triad covector components back to coordinate ones. In flat space the
//! triad is exactly the identity, so this reduces bitwise to the SR solver (K2).
//! Conserved rows: D, S_i, tau = E - D, D Y (dyngr convention, undensitised inside).

#include <cmath>

#include "athena.hpp"
#include "eos/primitive-solver/ps_types.hpp"
#include "dyn_grmhd/resistive/resistive_em.hpp"

namespace resistive {

KOKKOS_INLINE_FUNCTION
int SymIdx(const int i, const int j) {
  // PrimitiveSolver ordering S11,S12,S13,S22,S23,S33
  const int a = (i < j) ? i : j, b = (i < j) ? j : i;
  return (a == 0) ? b : ((a == 1) ? 2 + b : 5);
}

struct FaceTetrad {
  Real e[3][3];  // e[A][i] = e_A^i (contravariant coordinate components)
  Real w[3][3];  // w[A][i] = omega^A_i = gamma_ij e_A^j (dual basis)
  Real a;        // alpha sqrt(gamma^{dd})
};

// g, gu: gamma_ij and gamma^ij (PrimitiveSolver ordering); d: face normal direction.
KOKKOS_INLINE_FUNCTION
void BuildFaceTetrad(const Real g[NSPMETRIC], const Real gu[NSPMETRIC],
                     const Real alpha, const int d, FaceTetrad &t) {
  const int c[3] = {d, (d + 1) % 3, (d + 2) % 3};
  const Real sgdd = sqrt(gu[SymIdx(d, d)]);
  for (int i = 0; i < 3; ++i) t.e[0][i] = gu[SymIdx(d, i)]/sgdd;
  for (int A = 1; A < 3; ++A) {
    Real v[3] = {0.0, 0.0, 0.0};
    v[c[A]] = 1.0;
    for (int B = 0; B < A; ++B) {
      const Real p = Dot(g, v, t.e[B]);
      for (int i = 0; i < 3; ++i) v[i] -= p*t.e[B][i];
    }
    const Real n = sqrt(Dot(g, v, v));
    for (int i = 0; i < 3; ++i) t.e[A][i] = v[i]/n;
  }
  for (int A = 0; A < 3; ++A) Lower(g, t.e[A], t.w[A]);
  t.a = alpha*sgdd;
}

// contravariant vector: coordinate -> triad (V^A = omega^A_i V^i) and back
KOKKOS_INLINE_FUNCTION
void VecToTetrad(const FaceTetrad &t, const Real v[3], Real vt[3]) {
  for (int A = 0; A < 3; ++A) vt[A] = t.w[A][0]*v[0] + t.w[A][1]*v[1] + t.w[A][2]*v[2];
}
KOKKOS_INLINE_FUNCTION
void VecFromTetrad(const FaceTetrad &t, const Real vt[3], Real v[3]) {
  for (int i = 0; i < 3; ++i) v[i] = t.e[0][i]*vt[0] + t.e[1][i]*vt[1] + t.e[2][i]*vt[2];
}
// covector: coordinate -> triad (S_A = e_A^i S_i) and back (S_i = omega^A_i S_A)
KOKKOS_INLINE_FUNCTION
void CovToTetrad(const FaceTetrad &t, const Real s[3], Real st[3]) {
  for (int A = 0; A < 3; ++A) st[A] = t.e[A][0]*s[0] + t.e[A][1]*s[1] + t.e[A][2]*s[2];
}
KOKKOS_INLINE_FUNCTION
void CovFromTetrad(const FaceTetrad &t, const Real st[3], Real s[3]) {
  for (int i = 0; i < 3; ++i) s[i] = t.w[0][i]*st[0] + t.w[1][i]*st[1] + t.w[2][i]*st[2];
}

// One side of the SR (triad-frame) Riemann problem along direction 0.
struct MHLLCSide {
  Real D, Q[3], Eh, p;   // hydro-only conserved: D, Q_A = rho h W^2 v_A, Eh = rho h W^2-p
  Real v[3], cs2;        // 3-velocity (triad) and squared sound speed
  Real Y[MAX_SPECIES];
  Real E[3], B[3];       // Eulerian fields (triad)
};

KOKKOS_INLINE_FUNCTION
void SRAcousticSpeeds(const MHLLCSide &s, Real &lm, Real &lp) {
  const Real v2 = s.v[0]*s.v[0] + s.v[1]*s.v[1] + s.v[2]*s.v[2];
  const Real vn = s.v[0];
  const Real den = 1.0 - v2*s.cs2;
  const Real disc = sqrt(s.cs2*fmax((1.0 - v2)*(den - vn*vn*(1.0 - s.cs2)), 0.0));
  lm = (vn*(1.0 - s.cs2) - disc)/den;
  lp = (vn*(1.0 - s.cs2) + disc)/den;
}

KOKKOS_INLINE_FUNCTION
void HydroFlux(const MHLLCSide &s, const int nscal, Real W[5 + MAX_SPECIES],
               Real H[5 + MAX_SPECIES]) {
  const Real vn = s.v[0];
  W[0] = s.D; W[1] = s.Q[0]; W[2] = s.Q[1]; W[3] = s.Q[2]; W[4] = s.Eh;
  H[0] = s.D*vn;
  H[1] = s.Q[0]*vn + s.p;
  H[2] = s.Q[1]*vn;
  H[3] = s.Q[2]*vn;
  H[4] = s.Q[0];
  for (int n = 0; n < nscal; ++n) {
    W[5 + n] = s.D*s.Y[n];
    H[5 + n] = W[5 + n]*vn;
  }
}

KOKKOS_INLINE_FUNCTION
void EMStateFlux(const Real E[3], const Real B[3], Real U[4], Real F[4]) {
  // triad, orthonormal, right-handed: S = E x B, e = (E^2+B^2)/2, flux along 0
  const Real S[3] = {E[1]*B[2] - E[2]*B[1], E[2]*B[0] - E[0]*B[2],
                     E[0]*B[1] - E[1]*B[0]};
  const Real e = 0.5*(E[0]*E[0] + E[1]*E[1] + E[2]*E[2]
                      + B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);
  U[0] = S[0]; U[1] = S[1]; U[2] = S[2]; U[3] = e;
  F[0] = -E[0]*E[0] - B[0]*B[0] + e;
  F[1] = -E[0]*E[1] - B[0]*B[1];
  F[2] = -E[0]*E[2] - B[0]*B[2];
  F[3] = S[0];
}

// Outer exact Maxwell solver (Mignone et al. 2019 Eq. 70); normal components are
// taken from the left state (continuous at a staggered face).
KOKKOS_INLINE_FUNCTION
void MaxwellStar(const MHLLCSide &l, const MHLLCSide &r, Real E[3], Real B[3]) {
  E[0] = l.E[0];
  B[0] = l.B[0];
  B[1] = 0.5*(l.B[1] + r.B[1]) + 0.5*(r.E[2] - l.E[2]);
  B[2] = 0.5*(l.B[2] + r.B[2]) - 0.5*(r.E[1] - l.E[1]);
  E[1] = 0.5*(l.E[1] + r.E[1]) - 0.5*(r.B[2] - l.B[2]);
  E[2] = 0.5*(l.E[2] + r.E[2]) + 0.5*(r.B[1] - l.B[1]);
}

// Output in the triad: total (hydro + EM) state U and flux H along direction 0 at the
// sampled ray, rows [D, S_0, S_1, S_2, tau, D Y...]. hllc_ok = false if the contact
// estimate was rejected and HLL was used for the hydro part.
KOKKOS_INLINE_FUNCTION
void MHLLCTetrad(const MHLLCSide &l, const MHLLCSide &r, const Real xi, const int nscal,
                 Real U[5 + MAX_SPECIES], Real H[5 + MAX_SPECIES], bool &hllc_ok) {
  constexpr int NW = 5 + MAX_SPECIES;
  const int nw = 5 + nscal;
  Real Wl[NW], Hl[NW], Wr[NW], Hr[NW];
  HydroFlux(l, nscal, Wl, Hl);
  HydroFlux(r, nscal, Wr, Hr);
  Real lml, lpl, lmr, lpr;
  SRAcousticSpeeds(l, lml, lpl);
  SRAcousticSpeeds(r, lmr, lpr);
  const Real lL = fmin(lml, lmr), lR = fmax(lpl, lpr);
  hllc_ok = true;

  Real W[NW], Hs[NW];
  if (xi <= lL) {
    for (int n = 0; n < nw; ++n) { W[n] = Wl[n]; Hs[n] = Hl[n]; }
  } else if (xi >= lR) {
    for (int n = 0; n < nw; ++n) { W[n] = Wr[n]; Hs[n] = Hr[n]; }
  } else {
    const Real idl = 1.0/(lR - lL);
    Real Whll[NW], Hhll[NW];
    for (int n = 0; n < nw; ++n) {
      Whll[n] = (lR*Wr[n] - lL*Wl[n] + Hl[n] - Hr[n])*idl;
      Hhll[n] = (lR*Hl[n] - lL*Hr[n] + lL*lR*(Wr[n] - Wl[n]))*idl;
    }
    // Eq. 76, negative branch in the cancellation-free form 2c/(-b + sqrt(b^2-4ac))
    const Real qa = Hhll[4], qb = -(Whll[4] + Hhll[1]), qc = Whll[1];
    const Real dsc = qb*qb - 4.0*qa*qc;
    const Real lstar = 2.0*qc/(-qb + sqrt(fmax(dsc, 0.0)));
    const Real pstar = Hhll[1] - lstar*Hhll[4];
    hllc_ok = (dsc >= 0.0) && std::isfinite(lstar) && (lstar > lL) && (lstar < lR)
              && (pstar > 0.0);
    if (!hllc_ok) {
      for (int n = 0; n < nw; ++n) { W[n] = Whll[n]; Hs[n] = Hhll[n]; }
    } else {
      const bool left = (xi <= lstar);
      const MHLLCSide &s = left ? l : r;
      const Real lk = left ? lL : lR;
      const Real *Wk = left ? Wl : Wr;
      const Real *Hk = left ? Hl : Hr;
      const Real vn = s.v[0];
      const Real f = 1.0/(lk - lstar);
      W[0] = s.D*(lk - vn)*f;
      W[1] = (s.Q[0]*(lk - vn) + pstar - s.p)*f;
      W[2] = s.Q[1]*(lk - vn)*f;
      W[3] = s.Q[2]*(lk - vn)*f;
      W[4] = (s.Eh*(lk - vn) + pstar*lstar - s.p*vn)*f;
      for (int n = 0; n < nscal; ++n) W[5 + n] = W[0]*s.Y[n];
      for (int n = 0; n < nw; ++n) Hs[n] = Hk[n] + lk*(W[n] - Wk[n]);
    }
  }

  // EM part: outer fan with speeds -1, 0 (normal), +1
  Real Es[3], Bs[3];
  if (xi <= -1.0) {
    for (int a = 0; a < 3; ++a) { Es[a] = l.E[a]; Bs[a] = l.B[a]; }
  } else if (xi >= 1.0) {
    for (int a = 0; a < 3; ++a) { Es[a] = r.E[a]; Bs[a] = r.B[a]; }
  } else {
    MaxwellStar(l, r, Es, Bs);
  }
  Real Uem[4], Fem[4];
  EMStateFlux(Es, Bs, Uem, Fem);

  // total rows [D, S_0, S_1, S_2, tau, DY]
  U[0] = W[0];                  H[0] = Hs[0];
  for (int a = 0; a < 3; ++a) {
    U[1 + a] = W[1 + a] + Uem[a];
    H[1 + a] = Hs[1 + a] + Fem[a];
  }
  U[4] = W[4] - W[0] + Uem[3];  H[4] = Hs[4] - Hs[0] + Fem[3];
  for (int n = 0; n < nscal; ++n) { U[5 + n] = W[5 + n]; H[5 + n] = Hs[5 + n]; }
}

// Fill a triad side from undensitised coordinate quantities: fluid-only conserved
// (D, S_i, tau) from PrimToCon with B = 0, p, Eulerian v^i, cs^2, Y, and E^i, B^i.
KOKKOS_INLINE_FUNCTION
void FillMHLLCSide(const FaceTetrad &t, const Real D, const Real S[3], const Real tau,
                   const Real p, const Real v[3], const Real cs2, const Real *Y,
                   const int nscal, const Real E[3], const Real B[3], MHLLCSide &s) {
  s.D = D;
  CovToTetrad(t, S, s.Q);
  s.Eh = tau + D;
  s.p = p;
  VecToTetrad(t, v, s.v);
  s.cs2 = cs2;
  for (int n = 0; n < nscal; ++n) s.Y[n] = Y[n];
  VecToTetrad(t, E, s.E);
  VecToTetrad(t, B, s.B);
}

// Coordinate densitised flux from the triad solution.
KOKKOS_INLINE_FUNCTION
void MHLLCCoordFlux(const FaceTetrad &t, const Real sg, const Real betad,
                    const int nscal, const Real U[5 + MAX_SPECIES],
                    const Real H[5 + MAX_SPECIES], Real flx[NCONS]) {
  Real Ht[3] = {H[1], H[2], H[3]}, Ut[3] = {U[1], U[2], U[3]}, Hc[3], Uc[3];
  CovFromTetrad(t, Ht, Hc);
  CovFromTetrad(t, Ut, Uc);
  flx[CDN] = sg*(t.a*H[0] - betad*U[0]);
  flx[CSX] = sg*(t.a*Hc[0] - betad*Uc[0]);
  flx[CSY] = sg*(t.a*Hc[1] - betad*Uc[1]);
  flx[CSZ] = sg*(t.a*Hc[2] - betad*Uc[2]);
  flx[CTA] = sg*(t.a*H[4] - betad*U[4]);
  for (int n = 0; n < nscal; ++n) flx[CYD + n] = sg*(t.a*H[5 + n] - betad*U[5 + n]);
}

// Tetrad MHLLC flux at a face from floored PrimitiveSolver primitives (rest-mass number
// density, normal-frame u^i, p, T, Y) and Eulerian E, B (undensitised, coordinate);
// densitised result in the uflx row order. Returns false if HLLC fell back to HLL.
template <class PSolver>
KOKKOS_INLINE_FUNCTION
bool MHLLCFace(const PSolver &ps, const Real g_in[NSPMETRIC], const Real gu[NSPMETRIC],
               const Real sg, const Real alpha, const Real beta[3], const int d,
               const Real priml[NPRIM], const Real primr[NPRIM], const int nscal,
               const Real eml[6], const Real emr[6], Real flx[NCONS]) {
  Real g[NSPMETRIC];
  for (int n = 0; n < NSPMETRIC; ++n) g[n] = g_in[n];
  FaceTetrad t;
  BuildFaceTetrad(g, gu, alpha, d, t);
  MHLLCSide side[2];
  for (int s = 0; s < 2; ++s) {
    Real prim[NPRIM];
    for (int n = 0; n < NPRIM; ++n) prim[n] = (s == 0) ? priml[n] : primr[n];
    const Real *em = (s == 0) ? eml : emr;
    Real bz[NMAG] = {0.0, 0.0, 0.0}, U[NCONS];
    for (int n = 0; n < NCONS; ++n) U[n] = 0.0;
    ps.PrimToCon(prim, U, bz, g);
    const Real u[3] = {prim[PVX], prim[PVY], prim[PVZ]};
    const Real iW = 1.0/sqrt(1.0 + Dot(g, u, u));
    const Real v[3] = {u[0]*iW, u[1]*iW, u[2]*iW};
    const Real cs = ps.GetEOS().GetSoundSpeed(prim[PRH], prim[PTM], &prim[PYF]);
    const Real S[3] = {U[CSX], U[CSY], U[CSZ]};
    const Real E[3] = {em[0], em[1], em[2]}, B[3] = {em[3], em[4], em[5]};
    FillMHLLCSide(t, U[CDN], S, U[CTA], prim[PPR], v, cs*cs, &prim[PYF], nscal, E, B,
                  side[s]);
  }
  Real Ut[5 + MAX_SPECIES], Ht[5 + MAX_SPECIES];
  bool ok;
  MHLLCTetrad(side[0], side[1], beta[d]/t.a, nscal, Ut, Ht, ok);
  MHLLCCoordFlux(t, sg, beta[d], nscal, Ut, Ht, flx);
  return ok;
}

}  // namespace resistive

#endif  // DYN_GRMHD_RESISTIVE_RESISTIVE_RIEMANN_HPP_
