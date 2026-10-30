#ifndef DYN_GRMHD_RSOLVERS_HLLE_FRAME_DYN_GRMHD_HPP_
#define DYN_GRMHD_RSOLVERS_HLLE_FRAME_DYN_GRMHD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hlle_frame_dyn_grmhd.hpp
//! \brief <mhd> riemann_frame = boosted (R-168, user-approved 2026-10-30; "level R",
//! research/notes/reference_congruence §8.2): the HLLE Riemann problem solved in the frame of
//! a unit timelike U at each face, with the face metric of HLLE_DYNGR.
//!
//! At a face x^j = const (covector m = dx^j): m_perp = m + (U.m) U, e_(1) = m_perp/|m_perp|,
//! e_(2), e_(3) by Gram-Schmidt of the other two coordinate directions, all in the full face
//! metric. T^mu_nu m_mu = |m_perp| [T^(1)_nu + s T^(0)_nu] with s = U^j/|m_perp|, so in U's
//! frame the face moves at -s along e_(1). The special-relativistic HLLE along e_(1) (Q =
//! T^(0)_nu, F = T^(1)_nu with nu a coordinate index; the mass Q = rho u^(0), F = rho u^(1))
//! is sampled along xi = -s, and sqrt(-g) |m_perp| x the sampled values are the coordinate
//! fluxes sqrt(-g) T^j_nu and sqrt(-g) rho u^j. The Valencia rows: F_S_k = sqrt(-g) T^j_k,
//! F_E = -(sqrt(-g) T^j_nu) n^nu with the face n, F_tau = F_E - F_D.
//! U: eulerian, U = n (this reproduces HLLE_DYNGR to round-off: the frame and coordinate
//! speeds are affinely related and the HLL state is invariant under that map); grid_rest,
//! U = (1, 0, 0, 0)/sqrt(-g_tt), which needs alpha^2 > beta_i beta^i (a face where it fails
//! is counted in err(0), and CalcFluxes stops the run; it then uses n, so the flux stays
//! finite until the stop).
//! Hydrodynamics only: the magnetic field must vanish (checked once in CalcFluxes), and the
//! EMFs are zero.

#include <math.h>

#include "coordinates/adm.hpp"
#include "eos/primitive_solver_hyd.hpp"
#include "eos/primitive-solver/geom_math.hpp"
#include "dyn_grmhd/metric_faces.hpp"

namespace dyngr {

//! the face frame: the 4-metric, U, the tetrad rows e_(a)^mu, the dual rows e^(a)_mu
//! (= eta^{aa} g_{mu nu} e_(a)^nu), |m_perp| and s; dir = 0, 1, 2 the face normal; ugrid:
//! grid_rest (else eulerian); returns false if grid_rest is not timelike (U = n then)
struct FrameFace {
  Real g[4][4];     // g_{mu nu}
  Real E[4][4];     // E[a][mu] = e_(a)^mu, a = 0 is U
  Real D[4][4];     // D[a][mu] = e^(a)_mu
  Real mp, s;       // |m_perp|, U^j/|m_perp|
  Real sqrtmg;      // sqrt(-g) = alpha sqrt(gamma)
};

KOKKOS_INLINE_FUNCTION
bool BuildFrameFace(const Real g3d[NSPMETRIC], const Real beta_u[3], const Real alpha,
                    const int dir, const bool ugrid, FrameFace &f) {
  const Real detg = Primitive::GetDeterminant(g3d);
  Real gu[NSPMETRIC];
  adm::SpatialInv(1.0/detg, g3d[S11], g3d[S12], g3d[S13], g3d[S22], g3d[S23], g3d[S33],
                  &gu[S11], &gu[S12], &gu[S13], &gu[S22], &gu[S23], &gu[S33]);
  const int sidx[3][3] = {{S11, S12, S13}, {S12, S22, S23}, {S13, S23, S33}};
  const Real ia = 1.0/Primitive::FloorLapse(alpha);
  Real bd[3];
  for (int a = 0; a < 3; ++a) {
    bd[a] = 0.0;
    for (int b = 0; b < 3; ++b) bd[a] += g3d[sidx[a][b]]*beta_u[b];
  }
  const Real bsq = bd[0]*beta_u[0] + bd[1]*beta_u[1] + bd[2]*beta_u[2];
  f.g[0][0] = -alpha*alpha + bsq;
  for (int a = 0; a < 3; ++a) {
    f.g[0][a+1] = f.g[a+1][0] = bd[a];
    for (int b = 0; b < 3; ++b) f.g[a+1][b+1] = g3d[sidx[a][b]];
  }
  f.sqrtmg = alpha*sqrt(detg);
  // U
  Real U[4];
  bool ok = true;
  if (ugrid && -f.g[0][0] > 0.0) {
    U[0] = 1.0/sqrt(-f.g[0][0]);
    U[1] = U[2] = U[3] = 0.0;
  } else {
    ok = !ugrid;
    U[0] = ia;
    for (int a = 0; a < 3; ++a) U[a+1] = -beta_u[a]*ia;
  }
  // m_perp^mu = g^{mu j} + U^j U^mu, g^{tj} = beta^j/alpha^2, g^{ij} = gamma^ij - beta^i
  // beta^j/alpha^2; |m_perp|^2 = g^{jj} + (U^j)^2
  const Real uj = U[dir+1];
  Real mperp[4];
  mperp[0] = beta_u[dir]*ia*ia + uj*U[0];
  for (int a = 0; a < 3; ++a) {
    mperp[a+1] = gu[sidx[a][dir]] - beta_u[a]*beta_u[dir]*ia*ia + uj*U[a+1];
  }
  f.mp = sqrt(gu[sidx[dir][dir]] - beta_u[dir]*beta_u[dir]*ia*ia + uj*uj);
  f.s = uj/f.mp;
  for (int mu = 0; mu < 4; ++mu) {
    f.E[0][mu] = U[mu];
    f.E[1][mu] = mperp[mu]/f.mp;
  }
  // e_(2), e_(3): Gram-Schmidt of the other two coordinate directions (cyclic order)
  auto dot = [&](const Real *x, const Real *y) {
    Real r = 0.0;
    for (int mu = 0; mu < 4; ++mu) {
      for (int nu = 0; nu < 4; ++nu) r += f.g[mu][nu]*x[mu]*y[nu];
    }
    return r;
  };
  for (int q = 2; q < 4; ++q) {
    Real v[4] = {0.0, 0.0, 0.0, 0.0};
    v[(dir + q - 1) % 3 + 1] = 1.0;
    for (int b = 0; b < q; ++b) {
      const Real c = dot(v, f.E[b])/dot(f.E[b], f.E[b]);
      for (int mu = 0; mu < 4; ++mu) v[mu] -= c*f.E[b][mu];
    }
    const Real n = 1.0/sqrt(dot(v, v));
    for (int mu = 0; mu < 4; ++mu) f.E[q][mu] = v[mu]*n;
  }
  for (int a = 0; a < 4; ++a) {
    const Real sg = (a == 0) ? -1.0 : 1.0;
    for (int mu = 0; mu < 4; ++mu) {
      Real r = 0.0;
      for (int nu = 0; nu < 4; ++nu) r += f.g[mu][nu]*f.E[a][nu];
      f.D[a][mu] = sg*r;
    }
  }
  return ok;
}

//! u^mu of a primitive state (prim[PVX..PVZ] = W v^i, Eulerian)
KOKKOS_INLINE_FUNCTION
void FourVelocity(const Real prim[NPRIM], const Real g3d[NSPMETRIC], const Real beta_u[3],
                  const Real alpha, Real u[4]) {
  Real uu[3] = {prim[PVX], prim[PVY], prim[PVZ]};
  const Real W = sqrt(1.0 + Primitive::SquareVector(uu, g3d));
  const Real ia = 1.0/Primitive::FloorLapse(alpha);
  u[0] = W*ia;
  for (int a = 0; a < 3; ++a) u[a+1] = uu[a] - W*beta_u[a]*ia;
}

//! the inverse: prim[PVX..PVZ] = W v^i from u^mu (W = alpha u^t)
KOKKOS_INLINE_FUNCTION
void FromFourVelocity(const Real u[4], const Real beta_u[3], const Real alpha,
                      Real prim[NPRIM]) {
  const Real W = alpha*u[0];
  const Real ia = 1.0/Primitive::FloorLapse(alpha);
  for (int a = 0; a < 3; ++a) prim[PVX+a] = u[a+1] + W*beta_u[a]*ia;
}

//! one state in the frame: Q_nu = T^(0)_nu, F_nu = T^(1)_nu, the mass Q, F, and the SR fast
//! speeds along e_(1) (the EOS's own GetGRFastMagnetosonicSpeeds with the flat metric, b = 0)
template<class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION
void FrameState(const PrimitiveSolverHydro<EOSPolicy, ErrorPolicy>& eos,
                const Real prim[NPRIM], const Real g3d[NSPMETRIC], const Real beta_u[3],
                const Real alpha, const FrameFace &f, Real Q[4], Real F[4], Real &qd,
                Real &fd, Real &lp, Real &lm) {
  Real u[4], ud[4], uf[4];
  FourVelocity(prim, g3d, beta_u, alpha, u);
  for (int mu = 0; mu < 4; ++mu) {
    ud[mu] = 0.0;
    for (int nu = 0; nu < 4; ++nu) ud[mu] += f.g[mu][nu]*u[nu];
  }
  for (int a = 0; a < 4; ++a) {
    uf[a] = 0.0;
    for (int mu = 0; mu < 4; ++mu) uf[a] += f.D[a][mu]*u[mu];
  }
  const Real mb = eos.ps.GetEOS().GetBaryonMass();
  const Real rho = mb*prim[PRH];
  Real yy[MAX_SPECIES];
  for (int n = 0; n < MAX_SPECIES; ++n) yy[n] = prim[PYF+n];
  const Real rhoh = rho*eos.ps.GetEOS().GetEnthalpy(prim[PRH], prim[PTM], yy);
  const Real p = prim[PPR];
  for (int nu = 0; nu < 4; ++nu) {
    Q[nu] = rhoh*uf[0]*ud[nu] + p*f.D[0][nu];
    F[nu] = rhoh*uf[1]*ud[nu] + p*f.D[1][nu];
  }
  qd = rho*uf[0];
  fd = rho*uf[1];
  Real pf[NPRIM];
  for (int n = 0; n < NPRIM; ++n) pf[n] = prim[n];
  pf[PVX] = uf[1]; pf[PVY] = uf[2]; pf[PVZ] = uf[3];
  Real gflat[NSPMETRIC] = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0};
  Real b0[3] = {0.0, 0.0, 0.0};
  eos.GetGRFastMagnetosonicSpeeds(lp, lm, pf, 0.0, gflat, b0, 1.0, 1.0, PVX);
}

//----------------------------------------------------------------------------------------
//! \fn SingleStateFrameHLLE
//! \brief the frame HLLE flux of one face from the (already limited/floored) L/R primitives:
//! flux[CDN..CTA] as HLLE_DYNGR's (densitized coordinate fluxes), fsc[n] (if not null) the
//! HLL flux of D*Y with the mass's speeds and branch; returns false if U was not timelike
template<int ivx, class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION
bool SingleStateFrameHLLE(const PrimitiveSolverHydro<EOSPolicy, ErrorPolicy>& eos,
    const Real prim_l[NPRIM], const Real prim_r[NPRIM], const int nscal,
    const Real g3d[NSPMETRIC], const Real beta_u[3], const Real alpha, const bool ugrid,
    Real flux[NCONS], Real *fsc = nullptr, const bool *sel = nullptr) {
  FrameFace f;
  const bool ok = BuildFrameFace(g3d, beta_u, alpha, ivx - IVX, ugrid, f);
  Real ql[4], fl[4], qr[4], fr[4], qdl, fdl, qdr, fdr, lpl, lml, lpr, lmr;
  FrameState(eos, prim_l, g3d, beta_u, alpha, f, ql, fl, qdl, fdl, lpl, lml);
  FrameState(eos, prim_r, g3d, beta_u, alpha, f, qr, fr, qdr, fdr, lpr, lmr);
  const Real lm = fmin(lml, lmr);
  const Real lp = fmax(lpl, lpr);
  const Real s = f.s, xi = -s;
  const int br = (lm >= xi) ? 0 : ((lp <= xi) ? 1 : 2);   // left, right, HLL
  const Real qb = 1.0/(lp - lm);
  auto sample = [&](const Real a_l, const Real b_l, const Real a_r, const Real b_r) {
    // a = Q, b = F of one component; the flux through the face moving at xi: F + s Q
    if (br == 0) return b_l + s*a_l;
    if (br == 1) return b_r + s*a_r;
    const Real fh = (lp*b_l - lm*b_r + lp*lm*(a_r - a_l))*qb;
    const Real qh = (lp*a_r - lm*a_l - (b_r - b_l))*qb;
    return fh + s*qh;
  };
  const Real fac = f.sqrtmg*f.mp;
  Real X[4];
  for (int nu = 0; nu < 4; ++nu) X[nu] = fac*sample(ql[nu], fl[nu], qr[nu], fr[nu]);
  flux[CDN] = fac*sample(qdl, fdl, qdr, fdr);
  flux[CSX] = X[1];
  flux[CSY] = X[2];
  flux[CSZ] = X[3];
  const Real ia = 1.0/Primitive::FloorLapse(alpha);
  const Real fe = -(X[0] - (beta_u[0]*X[1] + beta_u[1]*X[2] + beta_u[2]*X[3]))*ia;
  flux[CTA] = fe - flux[CDN];
  if (fsc != nullptr) {
    for (int n = 0; n < nscal; ++n) {
      if (sel != nullptr && !sel[n]) continue;
      const Real yl = prim_l[PYF+n], yr = prim_r[PYF+n];
      fsc[n] = fac*sample(qdl*yl, fdl*yl, qdr*yr, fdr*yr);
    }
  }
  return ok;
}

//----------------------------------------------------------------------------------------
//! \fn HLLE_FRAME_DYNGR
//! \brief the frame HLLE over a row of faces, with HLLE_DYNGR's face metric and state
//! preparation (density/species limits, primitive floor). r2: wl/wr hold, in IVX..IVZ, the
//! velocity's frame components u^(a) relative to the cell-centre tetrad of this direction
//! (CalcFluxes, riemann_frame_recon = R2); they are read as components in the face tetrad.
//! err(0) counts faces where grid_rest U is not timelike.
template<int ivx, class EOSPolicy, class ErrorPolicy, class... FM>
KOKKOS_INLINE_FUNCTION
void HLLE_FRAME_DYNGR(TeamMember_t const &member,
     const PrimitiveSolverHydro<EOSPolicy, ErrorPolicy>& eos,
     const int m, const int k, const int j, const int il, const int iu,
     const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr,
     const int& nhyd, const int& nscal,
     const adm::ADM::ADM_vars& adm,
     DvceArray5D<Real> flx, DvceArray4D<Real> ey, DvceArray4D<Real> ez,
     const bool ugrid, const bool r2, const DvceArray1D<int> &err,
     const FM&... fmet) {
  par_for_inner(member, il, iu, [&](const int i) {
    Real g3d[NSPMETRIC];
    Real beta_u[3];
    Real alpha;
    if constexpr (sizeof...(FM) > 0) {
      FaceMetricFromArrays<ivx>(fmet..., m, k, j, i, g3d, beta_u, alpha);
    } else if constexpr (ivx == IVX) {
      adm::Face1Metric(m, k, j, i, adm.g_dd, adm.beta_u, adm.alpha, g3d, beta_u, alpha);
    } else if (ivx == IVY) {
      adm::Face2Metric(m, k, j, i, adm.g_dd, adm.beta_u, adm.alpha, g3d, beta_u, alpha);
    } else if (ivx == IVZ) {
      adm::Face3Metric(m, k, j, i, adm.g_dd, adm.beta_u, adm.alpha, g3d, beta_u, alpha);
    }

    Real prim_l[NPRIM], prim_r[NPRIM];
    Real mb = eos.ps.GetEOS().GetBaryonMass();
    prim_l[PRH] = wl(IDN, i)/mb;
    prim_l[PVX] = wl(IVX, i);
    prim_l[PVY] = wl(IVY, i);
    prim_l[PVZ] = wl(IVZ, i);
    prim_r[PRH] = wr(IDN, i)/mb;
    prim_r[PVX] = wr(IVX, i);
    prim_r[PVY] = wr(IVY, i);
    prim_r[PVZ] = wr(IVZ, i);
    if (r2) {   // u^(a) in the face tetrad -> u^mu -> W v^i
      FrameFace f;
      BuildFrameFace(g3d, beta_u, alpha, ivx - IVX, ugrid, f);
      Real *pp[2] = {prim_l, prim_r};
      for (int q = 0; q < 2; ++q) {
        Real *p = pp[q];
        const Real u0 = sqrt(1.0 + p[PVX]*p[PVX] + p[PVY]*p[PVY] + p[PVZ]*p[PVZ]);
        Real u[4];
        for (int mu = 0; mu < 4; ++mu) {
          u[mu] = u0*f.E[0][mu] + p[PVX]*f.E[1][mu] + p[PVY]*f.E[2][mu]
                  + p[PVZ]*f.E[3][mu];
        }
        FromFourVelocity(u, beta_u, alpha, p);
      }
    }
    for (int n = 0; n < nscal; n++) {
      prim_l[PYF + n] = wl(nhyd + n, i);
      prim_r[PYF + n] = wr(nhyd + n, i);
    }
    eos.ps.GetEOS().ApplyDensityLimits(prim_l[PRH]);
    eos.ps.GetEOS().ApplySpeciesLimits(&prim_l[PYF]);
    prim_l[PPR] = wl(IPR, i);
    prim_l[PTM] = eos.ps.GetEOS().GetTemperatureFromP(
                  prim_l[PRH], prim_l[PPR], &prim_l[PYF]);
    eos.ps.GetEOS().ApplyDensityLimits(prim_r[PRH]);
    eos.ps.GetEOS().ApplySpeciesLimits(&prim_r[PYF]);
    prim_r[PPR] = wr(IPR, i);
    prim_r[PTM] = eos.ps.GetEOS().GetTemperatureFromP(
                  prim_r[PRH], prim_r[PPR], &prim_r[PYF]);
    eos.ps.GetEOS().ApplyPrimitiveFloor(prim_l[PRH], &prim_l[PVX], prim_l[PPR],
                                    prim_l[PTM], &prim_l[PYF]);
    eos.ps.GetEOS().ApplyPrimitiveFloor(prim_r[PRH], &prim_r[PVX], prim_r[PPR],
                                    prim_r[PTM], &prim_r[PYF]);

    Real flux[NCONS], fsc[MAX_SPECIES];
    bool sel[MAX_SPECIES];
    for (int s = 0; s < nscal; ++s) {   // as HLLE_DYNGR: the scalars it writes itself
      sel[s] = eos.sf_hll || (eos.de_on && eos.de_flux_hll && s == eos.de_idx);
    }
    const bool ok = SingleStateFrameHLLE<ivx>(eos, prim_l, prim_r, nscal, g3d, beta_u,
                                              alpha, ugrid, flux, fsc, sel);
    if (!ok) Kokkos::atomic_add(&err(0), 1);
    flx(m, IDN, k, j, i) = flux[CDN];
    flx(m, IEN, k, j, i) = flux[CTA];
    flx(m, IVX, k, j, i) = flux[CSX];
    flx(m, IVY, k, j, i) = flux[CSY];
    flx(m, IVZ, k, j, i) = flux[CSZ];
    for (int s = 0; s < nscal; ++s) {
      if (sel[s]) flx(m, nhyd + s, k, j, i) = fsc[s];
    }
    ey(m, k, j, i) = 0.0;
    ez(m, k, j, i) = 0.0;
  });
}

} // namespace dyngr

#endif  // DYN_GRMHD_RSOLVERS_HLLE_FRAME_DYN_GRMHD_HPP_
