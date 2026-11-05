#ifndef DYN_GRMHD_RESISTIVE_RESISTIVE_IMPLICIT_HPP_
#define DYN_GRMHD_RESISTIVE_RESISTIVE_IMPLICIT_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file resistive_implicit.hpp
//! \brief Implicit Ohmic step of resistive GRMHD (07_scoping_v2 §3.2).
//!
//! With sdt = sigma * alpha * a_ss * dt the stiff Ampere update divided by sqrt(gamma)
//! reads  E = R - sdt [W E + u x B - (E.u) v],  whose closed form at fixed u is
//! (Mignone et al. 2019, Eqs. 52-53, multiplied through by sigma)
//!     E.u = W (R.u) / (W + sdt)
//!     E   = [R - sdt (u x B - (E.u) v)] / (1 + sdt W).
//! (u x B)^i = eps^ijk u_j B_k. All vectors are undensitised Eulerian components.
//!
//! The cell step (resistive c2p) solves for u: at trial u, E(u) is the closed form, the
//! EM part is removed from the fixed totals (D, S_i, tau) and the hydro c2p
//! (PrimitiveSolver, B = 0) returns u_new(u). g(u) = u - u_new(u) = 0 is solved by
//! Newton-Broyden (finite-difference Jacobian once, then Broyden updates). The lagged-v
//! result (closed form at the guess plus one hydro c2p) is the fallback.

#include <cmath>

#include "athena.hpp"
#include "eos/primitive-solver/ps_types.hpp"
#include "eos/primitive-solver/ps_error.hpp"
#include "dyn_grmhd/resistive/resistive_em.hpp"

namespace resistive {

KOKKOS_INLINE_FUNCTION
Real LorentzFromU(const Real g[NSPMETRIC], const Real u[3]) {
  return sqrt(1.0 + Dot(g, u, u));
}

// Closed-form implicit E at fixed four-velocity u^i (Eqs. 52-53). sdt >= 0.
// sdt = 0 returns E = R exactly.
KOKKOS_INLINE_FUNCTION
void OhmicE(const Real g[NSPMETRIC], const Real sg, const Real sdt, const Real u[3],
            const Real B[3], const Real R[3], Real E[3]) {
  const Real W = LorentzFromU(g, u);
  const Real eu = W*Dot(g, R, u)/(W + sdt);
  Real ud[3], Bd[3], uxb[3];
  Lower(g, u, ud);
  Lower(g, B, Bd);
  CrossUp(sg, ud, Bd, uxb);
  const Real iden = 1.0/(1.0 + sdt*W);
  const Real iW = 1.0/W;
  for (int a = 0; a < 3; ++a) {
    E[a] = (R[a] - sdt*(uxb[a] - eu*u[a]*iW))*iden;
  }
}

// Face step: closed form for the left and right reconstructed u, averaged (Eq. 54).
// Returns the full averaged vector; the caller keeps the face-normal component.
KOKKOS_INLINE_FUNCTION
void OhmicFaceE(const Real g[NSPMETRIC], const Real sg, const Real sdt,
                const Real ul[3], const Real ur[3], const Real B[3], const Real R[3],
                Real E[3]) {
  Real el[3], er[3];
  OhmicE(g, sg, sdt, ul, B, R, el);
  OhmicE(g, sg, sdt, ur, B, R, er);
  for (int a = 0; a < 3; ++a) E[a] = 0.5*(el[a] + er[a]);
}

struct CellSolveResult {
  int iterations = 0;        // Newton iterations taken (0 = guess already converged)
  int c2p_calls = 0;         // hydro c2p evaluations
  bool converged = false;    // Newton converged to tol
  bool fallback = false;     // lagged-v result returned instead
  bool failed = false;       // no usable state (hydro c2p failed at the guess)
  bool floor = false;        // hydro c2p applied a floor in the accepted evaluation
};

namespace detail {

KOKKOS_INLINE_FUNCTION
bool Solve3(const Real J[3][3], const Real b[3], Real x[3]) {
  const Real c00 = J[1][1]*J[2][2] - J[1][2]*J[2][1];
  const Real c01 = J[1][2]*J[2][0] - J[1][0]*J[2][2];
  const Real c02 = J[1][0]*J[2][1] - J[1][1]*J[2][0];
  const Real det = J[0][0]*c00 + J[0][1]*c01 + J[0][2]*c02;
  Real scale = 0.0;
  for (int a = 0; a < 3; ++a)
    for (int c = 0; c < 3; ++c) scale = fmax(scale, fabs(J[a][c]));
  if (!(fabs(det) > 1.0e-300) || !(fabs(det) > 1.0e-14*scale*scale*scale)) return false;
  const Real id = 1.0/det;
  Real inv[3][3];
  inv[0][0] = c00*id;
  inv[0][1] = (J[0][2]*J[2][1] - J[0][1]*J[2][2])*id;
  inv[0][2] = (J[0][1]*J[1][2] - J[0][2]*J[1][1])*id;
  inv[1][0] = c01*id;
  inv[1][1] = (J[0][0]*J[2][2] - J[0][2]*J[2][0])*id;
  inv[1][2] = (J[0][2]*J[1][0] - J[0][0]*J[1][2])*id;
  inv[2][0] = c02*id;
  inv[2][1] = (J[0][1]*J[2][0] - J[0][0]*J[2][1])*id;
  inv[2][2] = (J[0][0]*J[1][1] - J[0][1]*J[1][0])*id;
  for (int a = 0; a < 3; ++a) x[a] = inv[a][0]*b[0] + inv[a][1]*b[1] + inv[a][2]*b[2];
  return true;
}

KOKKOS_INLINE_FUNCTION
Real MaxAbs3(const Real a[3]) {
  return fmax(fabs(a[0]), fmax(fabs(a[1]), fabs(a[2])));
}

// One residual evaluation g(u) = u - u_new(u). Returns false if the hydro c2p fails.
template <class PSolver>
KOKKOS_INLINE_FUNCTION
bool Residual(const PSolver &ps, Real g3d[NSPMETRIC], Real g3u[NSPMETRIC], const Real sg,
              const Real sdt, const Real cons_tot[NCONS], const Real B[3],
              const Real R[3], const Real u[3], Real prim[NPRIM], Real E[3],
              Real gres[3], bool &floor) {
  OhmicE(g3d, sg, sdt, u, B, R, E);
  Real s_em[3], e_em;
  EMParts(g3d, sg, E, B, s_em, e_em);
  Real cons[NCONS];
  for (int n = 0; n < NCONS; ++n) cons[n] = cons_tot[n];
  for (int a = 0; a < 3; ++a) cons[CSX+a] -= s_em[a];
  cons[CTA] -= e_em;
  Real b0[NMAG];
  for (int n = 0; n < NMAG; ++n) b0[n] = 0.0;
  Primitive::SolverResult res = ps.ConToPrim(prim, cons, b0, g3d, g3u);
  if (res.error != Primitive::Error::SUCCESS) return false;
  floor = res.cons_floor || res.prim_floor || res.cons_adjusted;
  // Tabulated EOSs clamp T to the table range inside the c2p without reporting a floor;
  // such states do not reproduce the fluid energy and are treated as floored.
  if (!floor) {
    Real chk[NCONS];
    ps.PrimToCon(prim, chk, b0, g3d);
    const Real tau = cons_tot[CTA] - e_em;
    if (fabs(chk[CTA] - tau) > 1.0e-9*(fabs(tau) + cons_tot[CDN])) floor = true;
  }
  for (int a = 0; a < 3; ++a) {
    gres[a] = u[a] - prim[PVX+a];
    if (!(fabs(gres[a]) < 1.0e300)) return false;  // also rejects NaN
  }
  return true;
}

}  // namespace detail

//! Resistive c2p for one cell.
//! cons_tot : undensitised totals (D, S_i, tau [, Y]) in the PrimitiveSolver layout
//! B, R     : undensitised Eulerian B and the explicit-stage E (cell average of face E~
//!            divided by sqrt(gamma))
//! u0       : initial guess for u^i (normally the previous-stage value)
//! On return prim/E hold the accepted state (converged or lagged-v fallback).
template <class PSolver>
KOKKOS_INLINE_FUNCTION
CellSolveResult ResistiveCellSolve(const PSolver &ps, const Real g3d_in[NSPMETRIC],
                                   const Real g3u_in[NSPMETRIC], const Real sdt,
                                   const Real cons_tot[NCONS], const Real B[3],
                                   const Real R[3], const Real u0[3], const Real tol,
                                   const int maxit, const bool allow_fallback,
                                   Real prim[NPRIM], Real E[3]) {
  CellSolveResult out;
  Real g3d[NSPMETRIC], g3u[NSPMETRIC];
  for (int n = 0; n < NSPMETRIC; ++n) { g3d[n] = g3d_in[n]; g3u[n] = g3u_in[n]; }
  const Real sg = SqrtDet(g3d);

  Real u[3] = {u0[0], u0[1], u0[2]};
  Real gk[3];
  bool fl = false;
  ++out.c2p_calls;
  if (!detail::Residual(ps, g3d, g3u, sg, sdt, cons_tot, B, R, u, prim, E, gk, fl)) {
    out.failed = true;
    return out;
  }
  // keep the lagged-v state for the fallback
  Real prim_lag[NPRIM], e_lag[3];
  for (int n = 0; n < NPRIM; ++n) prim_lag[n] = prim[n];
  for (int a = 0; a < 3; ++a) e_lag[a] = E[a];
  const bool fl_lag = fl;
  out.floor = fl;
  if (!fl && detail::MaxAbs3(gk) <= tol*(1.0 + detail::MaxAbs3(u))) {
    out.converged = true;
    return out;
  }

  // finite-difference Jacobian dg/du at the guess
  Real J[3][3];
  Real ptmp[NPRIM], etmp[3], gp[3];
  bool ok = true;
  for (int c = 0; c < 3 && ok; ++c) {
    Real up[3] = {u[0], u[1], u[2]};
    const Real h = 1.0e-7*(1.0 + fabs(u[c]));
    up[c] += h;
    bool fdum;
    ++out.c2p_calls;
    ok = detail::Residual(ps, g3d, g3u, sg, sdt, cons_tot, B, R, up, ptmp, etmp, gp,
                          fdum) && !fdum;
    if (ok) {
      for (int a = 0; a < 3; ++a) J[a][c] = (gp[a] - gk[a])/h;
    }
  }

  for (int it = 1; ok && it <= maxit; ++it) {
    Real mg[3] = {-gk[0], -gk[1], -gk[2]};
    Real du[3];
    if (!detail::Solve3(J, mg, du)) break;
    // backtrack if the hydro c2p fails or the residual grows a lot
    Real un[3], gn[3];
    bool step_ok = false;
    for (int ls = 0; ls < 6; ++ls) {
      for (int a = 0; a < 3; ++a) un[a] = u[a] + du[a];
      ++out.c2p_calls;
      // floored hydro states are rejected: the floor makes g(u) discontinuous and
      // creates spurious roots when the EM energy dominates (high magnetisation)
      if (detail::Residual(ps, g3d, g3u, sg, sdt, cons_tot, B, R, un, prim, E, gn, fl)
          && !fl && detail::MaxAbs3(gn) < 4.0*detail::MaxAbs3(gk) + tol) {
        step_ok = true;
        break;
      }
      for (int a = 0; a < 3; ++a) du[a] *= 0.5;
    }
    if (!step_ok) break;
    out.iterations = it;
    if (detail::MaxAbs3(gn) <= tol*(1.0 + detail::MaxAbs3(un))) {
      out.converged = true;
      out.floor = false;
      return out;
    }
    // Broyden ("good") rank-one update: J += ((dg - J du) du^T)/(du.du)
    const Real dd = du[0]*du[0] + du[1]*du[1] + du[2]*du[2];
    if (dd > 0.0) {
      for (int a = 0; a < 3; ++a) {
        const Real r = (gn[a] - gk[a])
                     - (J[a][0]*du[0] + J[a][1]*du[1] + J[a][2]*du[2]);
        for (int c = 0; c < 3; ++c) J[a][c] += r*du[c]/dd;
      }
    }
    for (int a = 0; a < 3; ++a) { u[a] = un[a]; gk[a] = gn[a]; }
  }

  // not converged
  if (allow_fallback) {
    for (int n = 0; n < NPRIM; ++n) prim[n] = prim_lag[n];
    for (int a = 0; a < 3; ++a) E[a] = e_lag[a];
    out.fallback = true;
    out.floor = fl_lag;
  } else {
    out.failed = true;
  }
  return out;
}

}  // namespace resistive

#endif  // DYN_GRMHD_RESISTIVE_RESISTIVE_IMPLICIT_HPP_
