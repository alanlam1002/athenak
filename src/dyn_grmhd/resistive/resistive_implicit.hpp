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
//!
//! Decision 10 (07 v2.2 §3.2): (A) Newton starts from the previous-stage u (continuity
//! root); (B) residual form f(u) = S^f - (e+p) W u (Mignone 2019 Eq. 56; energy fixed by
//! a 1D EOS inversion at n = D/(m_b W)) as an alternative to g; (C) if the start state
//! is inadmissible (hydro c2p fails or floors, or the energy inversion has no root) the
//! start is moved to u(lam) = lam u_prev + (1-lam) u_drift, lam bisected to the largest
//! admissible value, u_drift the drift four-velocity of (R x B)/B^2 capped at W <= wmax.

#include <cmath>

// The cell solve calls the full hydro c2p from many sites (Newton, FD Jacobian, retry,
// continuation, fallback); inlining all of them makes the SYCL AOT device compile (ocloc)
// run for hours. Calls are cheap next to a c2p, so keep these out of line.
#if defined(__GNUC__) || defined(__clang__)
#define RESISTIVE_NOINLINE __attribute__((noinline))
#else
#define RESISTIVE_NOINLINE
#endif

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

// Face step: closed form for the left and right cell u, each with its own sigma dt,
// averaged (Eq. 54). Returns the full averaged vector; the caller keeps the face-normal
// component. With equal sdtl = sdtr this is the single-sigma form.
KOKKOS_INLINE_FUNCTION
void OhmicFaceE(const Real g[NSPMETRIC], const Real sg, const Real sdtl, const Real sdtr,
                const Real ul[3], const Real ur[3], const Real B[3], const Real R[3],
                Real E[3]) {
  Real el[3], er[3];
  OhmicE(g, sg, sdtl, ul, B, R, el);
  OhmicE(g, sg, sdtr, ur, B, R, er);
  for (int a = 0; a < 3; ++a) E[a] = 0.5*(el[a] + er[a]);
}

enum ImplicitResidualForm {kResidualC2P = 0, kResidualMomentum = 1};

struct CellSolveOptions {
  int form = kResidualC2P;  // residual form g (c2p) or f (momentum)
  bool retry = true;        // option C energy-safe retry
  Real wmax = 10.0;         // Lorentz factor cap of the drift start state
  bool continuation = true; // option A: sigma dt continuation before the fallback
};

struct CellSolveResult {
  int iterations = 0;        // Newton iterations taken (0 = guess already converged)
  int c2p_calls = 0;         // hydro c2p (form g) or EOS inversions (form f)
  bool converged = false;    // Newton converged to tol
  bool fallback = false;     // lagged-v result returned instead
  bool failed = false;       // no usable state (hydro c2p failed at the guess)
  bool floor = false;        // hydro c2p applied a floor in the accepted evaluation
  bool retry = false;        // option C moved the Newton start
  bool prestep_inadmissible = false;  // the E = R state has tau^f + D <= |S^f|
  bool continued = false;    // converged through the sigma dt continuation
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
KOKKOS_INLINE_FUNCTION RESISTIVE_NOINLINE
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

// Momentum residual (form f, Mignone 2019 Eq. 56) scaled by 1/max(|S|_1, 1e-3 (tau + D)):
//   f_i = S^f_i - (e + p) W u_i,  n = D/(m_b W),
// with the energy density e from (e + p(n,e)) W^2 - p(n,e) = tau^f + D (monotone in e,
// root in [e_min(n), (tau^f + D)/W^2]). Returns false (inadmissible) if there is no root.
template <class PSolver>
KOKKOS_INLINE_FUNCTION
bool MomentumResidual(const PSolver &ps, Real g3d[NSPMETRIC], const Real sg,
                      const Real sdt, const Real cons_tot[NCONS], const Real B[3],
                      const Real R[3], const Real u[3], Real prim[NPRIM], Real E[3],
                      Real fres[3]) {
  const auto &eos = ps.GetEOS();
  OhmicE(g3d, sg, sdt, u, B, R, E);
  Real s_em[3], e_em;
  EMParts(g3d, sg, E, B, s_em, e_em);
  const Real D = cons_tot[CDN];
  const Real tf = cons_tot[CTA] - e_em + D;
  const Real W = LorentzFromU(g3d, u);
  const Real W2 = W*W;
  const Real mb = eos.GetBaryonMass();
  const Real n = D/(mb*W);
  Real Y[MAX_SPECIES] = {0.0};
  const int nsp = eos.GetNSpecies();
  for (int s = 0; s < nsp; ++s) Y[s] = cons_tot[CYD + s]/D;
  if (!(n > 0.0) || !(tf > 0.0)) return false;
  auto pres = [&](const Real e, Real &T) {
    T = eos.GetTemperatureFromE(n, e, Y);
    return eos.GetPressure(n, T, Y);
  };
  Real T;
  Real elo = 0.0;
  eos.ApplyEnergyLimits(elo, n, Y);
  Real flo = (elo + pres(elo, T))*W2 - pres(elo, T) - tf;
  if (flo > 0.0) return false;  // no temperature in range reproduces the energy
  Real ehi = tf/W2;
  if (!(ehi > elo)) return false;
  Real phi = pres(ehi, T);
  Real fhi = (ehi + phi)*W2 - phi - tf;
  // Illinois regula falsi on e
  Real e = ehi, fe = fhi;
  int side = 0;
  for (int it = 0; it < 100; ++it) {
    if (!(fhi - flo != 0.0)) break;
    e = (elo*fhi - ehi*flo)/(fhi - flo);
    const Real p = pres(e, T);
    fe = (e + p)*W2 - p - tf;
    if (fabs(fe) <= 1.0e-15*tf || (ehi - elo) <= 1.0e-15*ehi) break;
    if (fe*fhi > 0.0) {
      ehi = e; fhi = fe;
      if (side == -1) flo *= 0.5;
      side = -1;
    } else {
      elo = e; flo = fe;
      if (side == 1) fhi *= 0.5;
      side = 1;
    }
  }
  if (!(fabs(fe) <= 1.0e-12*tf)) return false;
  const Real p = pres(e, T);
  // reject states clamped by the EOS (e.g. tabulated T range)
  if (fabs(eos.GetEnergy(n, T, Y) - e) > 1.0e-9*e) return false;
  Real ud[3];
  Lower(g3d, u, ud);
  const Real hW = (e + p)*W;
  // scale by the momentum totals (floored at 1e-3 (tau + D) for S -> 0), so the
  // tolerance bounds the relative momentum mismatch
  const Real stot = fabs(cons_tot[CSX]) + fabs(cons_tot[CSY]) + fabs(cons_tot[CSZ]);
  const Real isc = 1.0/fmax(stot, 1.0e-3*(cons_tot[CTA] + D));
  for (int a = 0; a < 3; ++a) {
    fres[a] = (cons_tot[CSX+a] - s_em[a] - hW*ud[a])*isc;
    if (!(fabs(fres[a]) < 1.0e300)) return false;
  }
  for (int k = 0; k < NPRIM; ++k) prim[k] = 0.0;
  prim[PRH] = n;
  for (int a = 0; a < 3; ++a) prim[PVX+a] = u[a];
  prim[PPR] = p;
  prim[PTM] = T;
  for (int s = 0; s < nsp; ++s) prim[PYF + s] = Y[s];
  return true;
}

// Residual of the selected form; ok = admissible (no failure, no floor).
template <class PSolver>
KOKKOS_INLINE_FUNCTION RESISTIVE_NOINLINE
bool EvalForm(const int form, const PSolver &ps, Real g3d[NSPMETRIC], Real g3u[NSPMETRIC],
              const Real sg, const Real sdt, const Real cons_tot[NCONS], const Real B[3],
              const Real R[3], const Real u[3], Real prim[NPRIM], Real E[3], Real r[3],
              bool &floor) {
  if (form == kResidualMomentum) {
    floor = false;
    return MomentumResidual(ps, g3d, sg, sdt, cons_tot, B, R, u, prim, E, r);
  }
  return Residual(ps, g3d, g3u, sg, sdt, cons_tot, B, R, u, prim, E, r, floor);
}

// Newton-Broyden from u (in/out) with an FD Jacobian at the start. gk holds the residual
// at u on entry (ok0: that evaluation was admissible). Returns true if converged; then
// prim/E hold the converged state.
template <class PSolver>
KOKKOS_INLINE_FUNCTION RESISTIVE_NOINLINE
bool Newton(const int form, const PSolver &ps, Real g3d[NSPMETRIC], Real g3u[NSPMETRIC],
            const Real sg, const Real sdt, const Real cons_tot[NCONS], const Real B[3],
            const Real R[3], Real u[3], Real gk[3], const bool ok0, const Real tol,
            const int maxit, Real prim[NPRIM], Real E[3], int &iters, int &calls) {
  Real J[3][3];
  Real ptmp[NPRIM], etmp[3], gp[3];
  bool fl = false;
  bool ok = ok0;
  for (int c = 0; c < 3 && ok; ++c) {
    Real up[3] = {u[0], u[1], u[2]};
    const Real h = 1.0e-7*(1.0 + fabs(u[c]));
    up[c] += h;
    bool fdum;
    ++calls;
    ok = EvalForm(form, ps, g3d, g3u, sg, sdt, cons_tot, B, R, up, ptmp, etmp, gp, fdum)
         && !fdum;
    if (ok) {
      for (int a = 0; a < 3; ++a) J[a][c] = (gp[a] - gk[a])/h;
    }
  }
  for (int it = 1; ok && it <= maxit; ++it) {
    Real mg[3] = {-gk[0], -gk[1], -gk[2]};
    Real du[3];
    if (!Solve3(J, mg, du)) break;
    // backtrack if the residual evaluation fails or the residual grows a lot
    Real un[3], gn[3];
    bool step_ok = false;
    for (int ls = 0; ls < 6; ++ls) {
      for (int a = 0; a < 3; ++a) un[a] = u[a] + du[a];
      ++calls;
      // floored hydro states are rejected: the floor makes g(u) discontinuous and
      // creates spurious roots when the EM energy dominates (high magnetisation)
      if (EvalForm(form, ps, g3d, g3u, sg, sdt, cons_tot, B, R, un, prim, E, gn, fl)
          && !fl && MaxAbs3(gn) < 4.0*MaxAbs3(gk) + tol) {
        step_ok = true;
        break;
      }
      for (int a = 0; a < 3; ++a) du[a] *= 0.5;
    }
    if (!step_ok) break;
    iters = it;
    if (MaxAbs3(gn) <= tol*(1.0 + MaxAbs3(un))) {
      for (int a = 0; a < 3; ++a) u[a] = un[a];
      return true;
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
  return false;
}

}  // namespace detail

//! Resistive c2p for one cell.
//! cons_tot : undensitised totals (D, S_i, tau [, Y]) in the PrimitiveSolver layout
//! B, R     : undensitised Eulerian B and the explicit-stage E (cell average of face E~
//!            divided by sqrt(gamma))
//! u0       : initial guess for u^i (normally the previous-stage value)
//! On return prim/E hold the accepted state (converged or lagged-v fallback).
template <class PSolver>
KOKKOS_INLINE_FUNCTION RESISTIVE_NOINLINE
CellSolveResult ResistiveCellSolve(const PSolver &ps, const Real g3d_in[NSPMETRIC],
                                   const Real g3u_in[NSPMETRIC], const Real sdt,
                                   const Real cons_tot[NCONS], const Real B[3],
                                   const Real R[3], const Real u0[3], const Real tol,
                                   const int maxit, const bool allow_fallback,
                                   Real prim[NPRIM], Real E[3],
                                   const CellSolveOptions opt = CellSolveOptions()) {
  CellSolveResult out;
  Real g3d[NSPMETRIC], g3u[NSPMETRIC];
  for (int n = 0; n < NSPMETRIC; ++n) { g3d[n] = g3d_in[n]; g3u[n] = g3u_in[n]; }
  const Real sg = SqrtDet(g3d);
  const int form = opt.form;

  // pre-step (E = R) admissibility, 07 §3.2.1: tau^f + D > |S^f| is necessary
  {
    Real s_em[3], e_em;
    EMParts(g3d, sg, R, B, s_em, e_em);
    Real sf[3];
    for (int a = 0; a < 3; ++a) sf[a] = cons_tot[CSX+a] - s_em[a];
    const Real sfn = sqrt(fmax(Dot(g3u, sf, sf), 0.0));
    out.prestep_inadmissible = !(cons_tot[CTA] - e_em + cons_tot[CDN] > sfn);
  }

  Real u[3] = {u0[0], u0[1], u0[2]};
  Real gk[3];
  bool fl = false;
  ++out.c2p_calls;
  bool ok0 = detail::EvalForm(form, ps, g3d, g3u, sg, sdt, cons_tot, B, R, u, prim, E,
                              gk, fl);
  // lagged-v state (form g at u0) for the fallback; for form g it is this evaluation
  bool have_lag = (form == kResidualC2P) && ok0;
  Real prim_lag[NPRIM], e_lag[3];
  bool fl_lag = fl;
  if (have_lag) {
    for (int n = 0; n < NPRIM; ++n) prim_lag[n] = prim[n];
    for (int a = 0; a < 3; ++a) e_lag[a] = E[a];
  }
  if (ok0 && !fl && detail::MaxAbs3(gk) <= tol*(1.0 + detail::MaxAbs3(u))) {
    out.converged = true;
    return out;
  }

  // option C: move an inadmissible start towards the drift state
  if (!(ok0 && !fl) && opt.retry) {
    Real ud[3];
    {
      Real Rd[3], Bd[3], rxb[3];
      Lower(g3d, R, Rd);
      Lower(g3d, B, Bd);
      CrossUp(sg, Rd, Bd, rxb);
      const Real b2 = Dot(g3d, B, B);
      for (int a = 0; a < 3; ++a) ud[a] = (b2 > 0.0) ? rxb[a]/b2 : 0.0;
      const Real vmax = sqrt(1.0 - 1.0/(opt.wmax*opt.wmax));
      const Real v = sqrt(fmax(Dot(g3d, ud, ud), 0.0));
      const Real sc = (v > vmax) ? vmax/v : 1.0;
      const Real vv = v*sc;
      const Real W = 1.0/sqrt(1.0 - vv*vv);
      for (int a = 0; a < 3; ++a) ud[a] *= sc*W;
    }
    Real ptmp[NPRIM], etmp[3], gtmp[3];
    bool ftmp = false;
    ++out.c2p_calls;
    bool okd = detail::EvalForm(form, ps, g3d, g3u, sg, sdt, cons_tot, B, R, ud, ptmp,
                                etmp, gtmp, ftmp) && !ftmp;
    if (okd) {
      Real lo = 0.0, hi = 1.0;
      for (int b = 0; b < 10; ++b) {
        const Real lm = 0.5*(lo + hi);
        Real ul[3];
        for (int a = 0; a < 3; ++a) ul[a] = lm*u0[a] + (1.0 - lm)*ud[a];
        ++out.c2p_calls;
        if (detail::EvalForm(form, ps, g3d, g3u, sg, sdt, cons_tot, B, R, ul, ptmp, etmp,
                             gtmp, ftmp) && !ftmp) {
          lo = lm;
        } else {
          hi = lm;
        }
      }
      for (int a = 0; a < 3; ++a) u[a] = lo*u0[a] + (1.0 - lo)*ud[a];
      ++out.c2p_calls;
      ok0 = detail::EvalForm(form, ps, g3d, g3u, sg, sdt, cons_tot, B, R, u, prim, E, gk,
                             fl);
      out.retry = true;
      if (ok0 && !fl && detail::MaxAbs3(gk) <= tol*(1.0 + detail::MaxAbs3(u))) {
        out.converged = true;
        return out;
      }
    }
  }

  if (detail::Newton(form, ps, g3d, g3u, sg, sdt, cons_tot, B, R, u, gk, ok0, tol, maxit,
                     prim, E, out.iterations, out.c2p_calls)) {
    out.converged = true;
    out.floor = false;
    return out;
  }

  // option A: continue the root in sigma dt -> s sigma dt from s ~ 0, where it is the
  // (admissible) E = R state, to s = 1 (07 v2.2 §3.2.1); adaptive in log s
  if (opt.continuation && !out.prestep_inadmissible) {
    Real uc[3] = {u0[0], u0[1], u0[2]};
    Real ls = -8.0, dls = 0.25;
    bool started = false;
    for (int step = 0; step < 200 && ls < 0.0; ++step) {
      const Real lt = started ? fmin(ls + dls, 0.0) : ls;
      const Real sdts = (lt < 0.0) ? sdt*pow(10.0, lt) : sdt;
      Real un[3] = {uc[0], uc[1], uc[2]};
      Real gn[3];
      ++out.c2p_calls;
      const bool okn = detail::EvalForm(form, ps, g3d, g3u, sg, sdts, cons_tot, B, R, un,
                                        prim, E, gn, fl) && !fl;
      bool conv = okn && detail::MaxAbs3(gn) <= tol*(1.0 + detail::MaxAbs3(un));
      int itdum = 0;
      if (okn && !conv) {
        conv = detail::Newton(form, ps, g3d, g3u, sg, sdts, cons_tot, B, R, un, gn, okn,
                              tol, maxit, prim, E, itdum, out.c2p_calls);
      }
      Real du[3] = {un[0] - uc[0], un[1] - uc[1], un[2] - uc[2]};
      if (conv && (!started ||
                   sqrt(fmax(Dot(g3d, du, du), 0.0))
                   <= 0.05*(1.0 + sqrt(fmax(Dot(g3d, uc, uc), 0.0))))) {
        for (int a = 0; a < 3; ++a) uc[a] = un[a];
        if (started) {
          ls = lt;
          dls = fmin(1.5*dls, 1.0);
        }
        started = true;
        if (!(ls < 0.0)) {
          out.converged = true;
          out.continued = true;
          out.floor = false;
          return out;
        }
      } else {
        if (!started) break;
        dls *= 0.5;
        if (dls < 1.0e-4) break;
      }
    }
  }

  // not converged: lagged-v fallback (form g at the guess u0)
  if (allow_fallback && !have_lag) {
    Real gdum[3];
    ++out.c2p_calls;
    have_lag = detail::Residual(ps, g3d, g3u, sg, sdt, cons_tot, B, R, u0, prim_lag,
                                e_lag, gdum, fl_lag);
  }
  if (allow_fallback && have_lag) {
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
