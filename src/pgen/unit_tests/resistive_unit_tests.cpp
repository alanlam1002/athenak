//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file resistive_unit_tests.cpp
//! \brief Unit tests for resistive GRMHD building blocks (07_scoping_v2 §4.1).
//!
//!  K1  prim<->cons with E != 0 for both IEN conventions on random non-diagonal metrics:
//!      (a) ideal limit E = -v x B reproduces SingleP2C_IdealGRMHD (fixed-metric
//!          T^t_t + D convention; C++ port of scripts/ien_mapping_check.py);
//!      (b) fluid part agrees with PrimitiveSolver::PrimToCon (dyngr sqrt(gamma) tau);
//!      (c) round trip prim -> cons(E,B) -> subtract EM -> PrimitiveSolver c2p -> prim,
//!          for both dyngr and fixed-metric IEN.
//!  K2  tetrad MHLLC: (a) face triad orthonormal/right-handed, coordinate <-> triad
//!      round trip; (b) SR limit (alpha=1, beta=0, gamma=delta) reproduces the flat
//!      MHLLC bitwise in each direction; (c) exact stationary contact.
//!  K3c ideal-limit (E = -v x B) MHLLC flux equals the dyngr ideal physical flux.
//!  K3e ideal-limit edge EMF e_k = alpha E_k + beta^a B~^b - beta^b B~^a equals e_cc.
//!  K4  implicit Ohmic step: closed form solves the implicit equation; sigma -> 0 and
//!      sigma -> inf limits; Newton from the exact u stops at the lagged-v state and
//!      from a nearby guess converges locally to it.
//!  K5  Newton statistics over sigma_M = B^2/rho in 1e-2..1e4, W <= 10 (ideal gas and
//!      the CompOSE table <problem> k5_table if given), both residual forms, fixed-point
//!      and pre-step sampling, and pre-step with a sigma(rho) profile: converged roots
//!      reproduce the input totals; no hard failures; pre-step roots equal the sigma dt
//!      continuation root with no fallback for sigma_M <= 10.
//!  K6  Conductivity functor values, limits, monotonicity and smoothness.
//!  K7  edge 2D solver: Mignone 2019 Eqs. 46-47 and 1D HLL reduction with GR speeds.
//!  C3  SMR prolongation of E~: fine q~ equals the coarse q~ (Toth-Roe), 2D and 3D.
//!  D   Diagnostics on synthetic mesh fields: Q == boundary flux, q~ for linear E,
//!      div B of a CT-built field, E_c mismatch, Newton counters.
//! Each kernel test is run on the default device and on the host execution space.
//! The run exits with EXIT_FAILURE if any test fails.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/prolongation.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "eos/ideal_c2p_mhd.hpp"
#include "eos/primitive-solver/primitive_solver.hpp"
#include "eos/primitive-solver/idealgas.hpp"
#include "eos/primitive-solver/eos_compose.hpp"
#include "eos/primitive-solver/logs.hpp"
#include "eos/primitive-solver/reset_floor.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/primitive_solver_hyd.hpp"
#include "dyn_grmhd/resistive/resistive_params.hpp"
#include "dyn_grmhd/resistive/resistive_em.hpp"
#include "dyn_grmhd/resistive/conductivity.hpp"
#include "dyn_grmhd/resistive/resistive_diagnostics.hpp"
#include "dyn_grmhd/resistive/resistive_implicit.hpp"
#include "dyn_grmhd/resistive/resistive_edge.hpp"
#include "dyn_grmhd/resistive/resistive_riemann.hpp"
#include "dyn_grmhd/rsolvers/llf_dyn_grmhd.hpp"
#include "pgen/pgen.hpp"

namespace {

using PS = Primitive::PrimitiveSolver<Primitive::IdealGas, Primitive::ResetFloor>;
using PST = Primitive::PrimitiveSolver<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                                       Primitive::ResetFloor>;
using PSH = PrimitiveSolverHydro<Primitive::IdealGas, Primitive::ResetFloor>;

bool g_failed = false;

void Report(const std::string &name, const Real err, const Real tol) {
  const bool ok = std::isfinite(err) && err <= tol;
  if (!ok) g_failed = true;
  if (global_variable::my_rank == 0) {
    std::cout << "  " << std::left << std::setw(46) << name << " err = "
              << std::scientific << std::setprecision(3) << err << "  tol = " << tol
              << (ok ? "  PASS" : "  FAIL") << std::endl;
  }
}

// splitmix64: stateless, device-callable deterministic RNG
KOKKOS_INLINE_FUNCTION
uint64_t SplitMix(uint64_t &s) {
  uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
KOKKOS_INLINE_FUNCTION
Real Uniform(uint64_t &s, Real a, Real b) {
  return a + (b - a)*static_cast<Real>(SplitMix(s) >> 11)*(1.0/9007199254740992.0);
}

struct RandomState {
  Real g[NSPMETRIC];
  Real alpha, beta[3];
  Real rho, p, wv[3], Bcode[3], Ernd[3];
};

// Random SPD non-diagonal gamma_ij, lapse, shift, and fluid/EM state for trial t.
KOKKOS_INLINE_FUNCTION
void MakeRandomState(const int t, RandomState &r) {
  uint64_t s = 0x5EED0000ULL + static_cast<uint64_t>(t)*7919ULL;
  Real M[3][3];
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) M[a][b] = (a == b ? 1.0 : 0.0) + 0.3*Uniform(s, -1., 1.);
  Real G[3][3];
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) {
      G[a][b] = (a == b) ? 0.5 : 0.0;
      for (int c = 0; c < 3; ++c) G[a][b] += 0.5*M[a][c]*M[b][c];
    }
  r.g[S11] = G[0][0]; r.g[S12] = G[0][1]; r.g[S13] = G[0][2];
  r.g[S22] = G[1][1]; r.g[S23] = G[1][2]; r.g[S33] = G[2][2];
  r.alpha = Uniform(s, 0.3, 1.3);
  for (int a = 0; a < 3; ++a) r.beta[a] = Uniform(s, -0.6, 0.6);
  r.rho = Uniform(s, 0.1, 1.1);
  r.p = Uniform(s, 0.01, 1.0);
  for (int a = 0; a < 3; ++a) r.wv[a] = Uniform(s, -1.5, 1.5);
  for (int a = 0; a < 3; ++a) r.Bcode[a] = Uniform(s, -1.0, 1.0);
  for (int a = 0; a < 3; ++a) r.Ernd[a] = Uniform(s, -1.0, 1.0);
}

KOKKOS_INLINE_FUNCTION
void Inverse3(const Real g[NSPMETRIC], Real gi[NSPMETRIC]) {
  const Real det = g[S11]*g[S22]*g[S33] + 2.0*g[S12]*g[S13]*g[S23]
                 - (g[S11]*g[S23]*g[S23] + g[S12]*g[S12]*g[S33] + g[S13]*g[S13]*g[S22]);
  const Real id = 1.0/det;
  gi[S11] = (g[S22]*g[S33] - g[S23]*g[S23])*id;
  gi[S12] = (g[S13]*g[S23] - g[S12]*g[S33])*id;
  gi[S13] = (g[S12]*g[S23] - g[S13]*g[S22])*id;
  gi[S22] = (g[S11]*g[S33] - g[S13]*g[S13])*id;
  gi[S23] = (g[S12]*g[S13] - g[S11]*g[S23])*id;
  gi[S33] = (g[S11]*g[S22] - g[S12]*g[S12])*id;
}

KOKKOS_INLINE_FUNCTION
Real RelErr(const Real a, const Real b, const Real scale) {
  return fabs(a - b)/fmax(scale, 1.0e-300);
}

constexpr Real kGam = 5.0/3.0;

// K1(a): ideal limit vs SingleP2C_IdealGRMHD (fixed-metric convention)
KOKKOS_INLINE_FUNCTION
Real K1IdealLimitTrial(const int t) {
  using namespace resistive;  // NOLINT
  RandomState r;
  MakeRandomState(t, r);
  const Real sg = SqrtDet(r.g);
  Real gi[NSPMETRIC];
  Inverse3(r.g, gi);
  Real bd[3];
  Lower(r.g, r.beta, bd);
  Real gl[4][4], gu[4][4];
  gl[0][0] = -r.alpha*r.alpha + bd[0]*r.beta[0] + bd[1]*r.beta[1] + bd[2]*r.beta[2];
  const int ix[3][3] = {{S11, S12, S13},
                        {S12, S22, S23},
                        {S13, S23, S33}
                       };
  const Real ia2 = 1.0/(r.alpha*r.alpha);
  gu[0][0] = -ia2;
  for (int a = 0; a < 3; ++a) {
    gl[0][a+1] = gl[a+1][0] = bd[a];
    gu[0][a+1] = gu[a+1][0] = r.beta[a]*ia2;
    for (int b = 0; b < 3; ++b) {
      gl[a+1][b+1] = r.g[ix[a][b]];
      gu[a+1][b+1] = gi[ix[a][b]] - r.beta[a]*r.beta[b]*ia2;
    }
  }
  MHDPrim1D w;
  w.d = r.rho; w.e = r.p/(kGam - 1.0);
  w.vx = r.wv[0]; w.vy = r.wv[1]; w.vz = r.wv[2];
  w.bx = r.Bcode[0]; w.by = r.Bcode[1]; w.bz = r.Bcode[2];
  HydCons1D u;
  SingleP2C_IdealGRMHD(gl, gu, w, kGam, u);
  const Real sqmg = r.alpha*sg;

  // 3+1 construction
  const Real W = sqrt(1.0 + Dot(r.g, r.wv, r.wv));
  Real v[3] = {r.wv[0]/W, r.wv[1]/W, r.wv[2]/W};
  Real B[3] = {r.alpha*r.Bcode[0], r.alpha*r.Bcode[1], r.alpha*r.Bcode[2]};
  Real E[3];
  IdealE(r.g, sg, v, B, E);
  const Real h = 1.0 + kGam/(kGam - 1.0)*r.p/r.rho;
  Real idn, im[3], ien;
  PrimToConsDyn(r.g, r.rho, r.p, h, r.wv, E, B, idn, im, ien);
  const Real ien_fix = FixedIENFromDynIEN(ien, idn, im, r.alpha, r.beta);

  const Real scale = fabs(sqmg*u.e) + fabs(sqmg*u.mx) + fabs(sqmg*u.my) +
                     fabs(sqmg*u.mz) + fabs(sqmg*u.d);
  Real err = RelErr(idn, sqmg*u.d, scale);
  err = fmax(err, RelErr(im[0], sqmg*u.mx, scale));
  err = fmax(err, RelErr(im[1], sqmg*u.my, scale));
  err = fmax(err, RelErr(im[2], sqmg*u.mz, scale));
  err = fmax(err, RelErr(ien_fix, sqmg*u.e, scale));
  // inverse map consistency
  err = fmax(err, RelErr(DynIENFromFixedIEN(ien_fix, idn, im, r.alpha, r.beta), ien,
                         scale));
  return err;
}

KOKKOS_INLINE_FUNCTION
void FillPrim(const PS &ps, const RandomState &r, Real prim[NPRIM]) {
  for (int n = 0; n < NPRIM; ++n) prim[n] = 0.0;
  const Real mb = ps.GetEOS().GetBaryonMass();
  prim[PRH] = r.rho/mb;
  prim[PVX] = r.wv[0]; prim[PVY] = r.wv[1]; prim[PVZ] = r.wv[2];
  prim[PPR] = r.p;
  prim[PTM] = r.p/prim[PRH];
}

// K1(b): fluid part of PrimToConsDyn (E = B = 0) equals PrimitiveSolver::PrimToCon.
KOKKOS_INLINE_FUNCTION
Real K1FluidVsPSTrial(const PS &ps, const int t) {
  using namespace resistive;  // NOLINT
  RandomState r;
  MakeRandomState(t, r);
  Real prim[NPRIM], cons[NCONS], b0[NMAG] = {0.0};
  FillPrim(ps, r, prim);
  ps.PrimToCon(prim, cons, b0, r.g);
  const Real sg = SqrtDet(r.g);
  const Real h = 1.0 + kGam/(kGam - 1.0)*r.p/r.rho;
  Real zero[3] = {0.0, 0.0, 0.0};
  Real idn, im[3], ien;
  PrimToConsDyn(r.g, r.rho, r.p, h, r.wv, zero, zero, idn, im, ien);
  const Real scale = fabs(cons[CDN]) + fabs(cons[CTA]) + fabs(cons[CSX]) +
                     fabs(cons[CSY]) + fabs(cons[CSZ]);
  Real err = RelErr(idn/sg, cons[CDN], scale);
  for (int a = 0; a < 3; ++a) err = fmax(err, RelErr(im[a]/sg, cons[CSX+a], scale));
  err = fmax(err, RelErr(ien/sg, cons[CTA], scale));
  return err;
}

// K1(c): round trip with arbitrary E != 0 through the c2p, both conventions.
// fixed = false: dyngr sqrt(gamma) tau;  fixed = true: T^t_t + D.
KOKKOS_INLINE_FUNCTION
Real K1RoundTripTrial(const PS &ps, const int t, const bool fixed) {
  using namespace resistive;  // NOLINT
  RandomState r;
  MakeRandomState(t, r);
  const Real sg = SqrtDet(r.g);
  Real gi[NSPMETRIC];
  Inverse3(r.g, gi);
  Real B[3] = {r.alpha*r.Bcode[0], r.alpha*r.Bcode[1], r.alpha*r.Bcode[2]};
  const Real h = 1.0 + kGam/(kGam - 1.0)*r.p/r.rho;
  Real idn, im[3], ien;
  PrimToConsDyn(r.g, r.rho, r.p, h, r.wv, r.Ernd, B, idn, im, ien);

  Real dim[3], dien, ien_fluid;
  if (fixed) {
    const Real ien_fix = FixedIENFromDynIEN(ien, idn, im, r.alpha, r.beta);
    EMIncrementFixed(r.g, sg, r.alpha, r.beta, r.Ernd, B, dim, dien);
    Real imf[3] = {im[0] - dim[0], im[1] - dim[1], im[2] - dim[2]};
    ien_fluid = DynIENFromFixedIEN(ien_fix - dien, idn, imf, r.alpha, r.beta);
  } else {
    EMIncrementDyn(r.g, sg, r.Ernd, B, dim, dien);
    ien_fluid = ien - dien;
  }
  Real cons[NCONS] = {0.0};
  cons[CDN] = idn/sg;
  for (int a = 0; a < 3; ++a) cons[CSX+a] = (im[a] - dim[a])/sg;
  cons[CTA] = ien_fluid/sg;
  Real prim[NPRIM] = {0.0}, b0[NMAG] = {0.0};
  Real g3d[NSPMETRIC];
  for (int n = 0; n < NSPMETRIC; ++n) g3d[n] = r.g[n];
  Primitive::SolverResult res = ps.ConToPrim(prim, cons, b0, g3d, gi);
  if (res.error != Primitive::Error::SUCCESS || res.cons_floor || res.prim_floor ||
      res.cons_adjusted) {
    return 1.0;
  }
  const Real mb = ps.GetEOS().GetBaryonMass();
  Real err = RelErr(prim[PRH]*mb, r.rho, r.rho);
  err = fmax(err, RelErr(prim[PPR], r.p, r.p));
  const Real wvs = fabs(r.wv[0]) + fabs(r.wv[1]) + fabs(r.wv[2]) + 1.0;
  for (int a = 0; a < 3; ++a) err = fmax(err, RelErr(prim[PVX+a], r.wv[a], wvs));
  return err;
}

template <class ExecSpace, class F>
Real MaxOverTrials(const char *label, const int ntrial, const F &f) {
  Real err = 0.0;
  Kokkos::parallel_reduce(label, Kokkos::RangePolicy<ExecSpace>(0, ntrial),
  KOKKOS_LAMBDA(const int t, Real &e) {
    e = fmax(e, f(t));
  }, Kokkos::Max<Real>(err));
  return err;
}

PS MakeSolver() {
  PS ps;
  ps.GetEOSMutable().SetGamma(kGam);
  ps.GetEOSMutable().SetDensityFloor(1.0e-12);
  ps.GetEOSMutable().SetTemperatureFloor(1.0e-12);
  ps.GetEOSMutable().SetMaxVelocity(sqrt(1.0 - 1.0/(100.0*100.0)));
  ps.tol = 1.0e-15;
  ps.GetRootSolverMutable().iterations = 60;
  return ps;
}

// K1(d): comoving b^2 (K5-run b^2/(rho h) binning). Ideal limit E = -v x B gives
// B^2/W^2 + (v.B)^2; with a random E the invariant b^2 - e^2 = B^2 - E^2 holds, where
// e^2 = W^2 [(E + v x B)^2 - (v.E)^2] is the comoving electric field.
KOKKOS_INLINE_FUNCTION
Real K1ComovingB2Trial(const int t) {
  using namespace resistive;  // NOLINT
  RandomState r;
  MakeRandomState(t, r);
  const Real sg = SqrtDet(r.g);
  const Real W = sqrt(1.0 + Dot(r.g, r.wv, r.wv));
  const Real v[3] = {r.wv[0]/W, r.wv[1]/W, r.wv[2]/W};
  const Real *B = r.Bcode;
  Real Eid[3];
  IdealE(r.g, sg, v, B, Eid);
  const Real vb = Dot(r.g, v, B);
  const Real b2id = Dot(r.g, B, B)/(W*W) + vb*vb;
  Real err = RelErr(ComovingB2(r.g, sg, r.wv, Eid, B), b2id, b2id);
  const Real *E = r.Ernd;
  Real vd[3], Bd[3], vxb[3];
  Lower(r.g, v, vd);
  Lower(r.g, B, Bd);
  CrossUp(sg, vd, Bd, vxb);
  const Real ep[3] = {E[0] + vxb[0], E[1] + vxb[1], E[2] + vxb[2]};
  const Real ve = Dot(r.g, v, E);
  const Real e2 = W*W*(Dot(r.g, ep, ep) - ve*ve);
  const Real inv = Dot(r.g, B, B) - Dot(r.g, E, E);
  const Real scale = Dot(r.g, B, B) + Dot(r.g, E, E);
  err = fmax(err, RelErr(ComovingB2(r.g, sg, r.wv, E, B) - e2, inv, scale*W*W));
  return err;
}

template <class ExecSpace>
void RunK1(const std::string &where, const int ntrial) {
  PS ps = MakeSolver();
  Real e;
  e = MaxOverTrials<ExecSpace>("K1a", ntrial,
      KOKKOS_LAMBDA(const int t) { return K1IdealLimitTrial(t); });
  Report("K1a ideal limit vs SingleP2C [" + where + "]", e, 1.0e-12);
  e = MaxOverTrials<ExecSpace>("K1d", ntrial,
      KOKKOS_LAMBDA(const int t) { return K1ComovingB2Trial(t); });
  Report("K1d comoving b^2: ideal limit, b^2-e^2 invariant [" + where + "]", e, 1.0e-12);
  e = MaxOverTrials<ExecSpace>("K1b", ntrial,
      KOKKOS_LAMBDA(const int t) { return K1FluidVsPSTrial(ps, t); });
  Report("K1b fluid part vs PrimToCon [" + where + "]", e, 1.0e-12);
  e = MaxOverTrials<ExecSpace>("K1c_dyn", ntrial,
      KOKKOS_LAMBDA(const int t) { return K1RoundTripTrial(ps, t, false); });
  Report("K1c round trip, dyngr IEN [" + where + "]", e, 1.0e-9);
  e = MaxOverTrials<ExecSpace>("K1c_fix", ntrial,
      KOKKOS_LAMBDA(const int t) { return K1RoundTripTrial(ps, t, true); });
  Report("K1c round trip, fixed IEN [" + where + "]", e, 1.0e-9);
}

//----------------------------------------------------------------------------------------
// K6: Conductivity
template <class ExecSpace>
void RunK6(const std::string &where) {
  using resistive::Conductivity;
  using resistive::ResistiveParams;
  using resistive::SigmaModel;
  ResistiveParams pu;
  pu.sigma_model = SigmaModel::uniform;
  pu.sigma = 37.5;
  const Conductivity cu(pu);
  Real e = MaxOverTrials<ExecSpace>("K6u", 64, KOKKOS_LAMBDA(const int t) {
    const Real rho = pow(10.0, -12.0 + 0.25*t);
    auto s = cu(rho);
    return fmax(fabs(s.par - 37.5), fabs(s.perp - 37.5));
  });
  Report("K6 uniform value [" + where + "]", e, 0.0);

  ResistiveParams pd;
  pd.sigma_model = SigmaModel::density;
  pd.sigma_in = 1.0e6; pd.sigma_out = 1.0e-2;
  pd.rho_transition = 1.0e-6; pd.sigma_width = 0.5;
  const Conductivity cd(pd);
  const Real lin = log(pd.sigma_in), lout = log(pd.sigma_out);
  // limits, midpoint, rho <= 0
  e = MaxOverTrials<ExecSpace>("K6d_lim", 1, KOKKOS_LAMBDA(const int) {
    Real err = fabs(log(cd(1.0e6).par) - lin)/fabs(lin - lout);
    err = fmax(err, fabs(log(cd(1.0e-18).par) - lout)/fabs(lin - lout));
    err = fmax(err, fabs(log(cd(1.0e-6).par) - 0.5*(lin + lout))/fabs(lin - lout));
    err = fmax(err, fabs(cd(0.0).par - 1.0e-2)/1.0e-2);
    err = fmax(err, fabs(cd(-1.0).par - 1.0e-2)/1.0e-2);
    err = fmax(err, fabs(cd(3.0e-5).par - cd(3.0e-5).perp));
    return err;
  });
  Report("K6 density limits/midpoint [" + where + "]", e, 1.0e-10);
  // monotone, bounded slope |d ln sigma / d log10 rho| <= |lin-lout|/(2 width),
  // and continuity of the slope (second difference of a C^inf function -> O(dx^2))
  const int nfine = 4000;
  const Real x0 = -14.0, dx = 16.0/nfine;
  const Real slope_max = fabs(lin - lout)/(2.0*pd.sigma_width);
  e = MaxOverTrials<ExecSpace>("K6d_smooth", nfine - 2, KOKKOS_LAMBDA(const int t) {
    const Real l0 = log(cd(pow(10.0, x0 + t*dx)).par);
    const Real l1 = log(cd(pow(10.0, x0 + (t+1)*dx)).par);
    const Real l2 = log(cd(pow(10.0, x0 + (t+2)*dx)).par);
    Real err = 0.0;
    if (l1 < l0) err = 1.0;  // non-monotone
    err = fmax(err, fmax(0.0, (l1 - l0)/dx - slope_max*(1.0 + 1.0e-8))/slope_max);
    // |f''| <= slope_max * (2/width) * 0.77 for tanh profile; allow 2x margin
    const Real d2 = fabs(l2 - 2.0*l1 + l0)/(dx*dx);
    err = fmax(err, fmax(0.0, d2 - 2.0*slope_max*2.0/pd.sigma_width)/slope_max);
    return err;
  });
  Report("K6 density monotone/smooth [" + where + "]", e, 0.0);
}

//----------------------------------------------------------------------------------------
// Diagnostics on mesh fields (device kernels)
void RunDiagnostics(MeshBlockPack *pmbp) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int ng = indcs.ng;
  const int nc1 = indcs.nx1 + 2*ng;
  const int nc2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*ng : 1;
  const int nc3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*ng : 1;
  const int nmb = pmbp->nmb_thispack;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  auto &size = pmbp->pmb->mb_size;
  const bool use2 = pmbp->pmesh->multi_d, use3 = pmbp->pmesh->three_d;
  auto &ms = pmbp->pmesh->mesh_size;

  DvceFaceFld4D<Real> ef("resist_ut_e", nmb, nc3, nc2, nc1);
  DvceFaceFld4D<Real> bf("resist_ut_b", nmb, nc3, nc2, nc1);
  DvceArray5D<Real> ec("resist_ut_ec", nmb, 3, nc3, nc2, nc1);
  const Real a1 = 0.7, a2 = -0.3, a3 = 1.9;  // linear E -> q~ = a1 + a2 + a3
  const Real qexact = a1 + (use2 ? a2 : 0.0) + (use3 ? a3 : 0.0);

  // E~ linear plus a smooth divergence-free-ish perturbation; B from CT of A on edges
  par_for("resist_ut_fill", DevExeSpace(), 0, nmb-1, ks, ks+nx3-1, js, js+nx2-1,
          is, is+nx1-1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const auto &sz = size.d_view(m);
    const Real x1l = LeftEdgeX(i-is, nx1, sz.x1min, sz.x1max);
    const Real x1r = LeftEdgeX(i-is+1, nx1, sz.x1min, sz.x1max);
    const Real x2l = LeftEdgeX(j-js, nx2, sz.x2min, sz.x2max);
    const Real x2r = LeftEdgeX(j-js+1, nx2, sz.x2min, sz.x2max);
    const Real x3l = LeftEdgeX(k-ks, nx3, sz.x3min, sz.x3max);
    const Real x3r = LeftEdgeX(k-ks+1, nx3, sz.x3min, sz.x3max);
    const Real x1c = 0.5*(x1l + x1r), x2c = 0.5*(x2l + x2r), x3c = 0.5*(x3l + x3r);
    auto e1 = [&](Real x1, Real x2, Real x3) { return a1*x1 + sin(x2)*cos(x3); };
    auto e2 = [&](Real x1, Real x2, Real x3) { return a2*x2 + cos(x1 + x3); };
    auto e3 = [&](Real x1, Real x2, Real x3) { return a3*x3 + sin(x1)*sin(x2); };
    ef.x1f(m,k,j,i) = e1(x1l, x2c, x3c);
    ef.x1f(m,k,j,i+1) = e1(x1r, x2c, x3c);
    ef.x2f(m,k,j,i) = e2(x1c, x2l, x3c);
    ef.x2f(m,k,j+1,i) = e2(x1c, x2r, x3c);
    ef.x3f(m,k,j,i) = e3(x1c, x2c, x3l);
    ef.x3f(m,k+1,j,i) = e3(x1c, x2c, x3r);
    ec(m,0,k,j,i) = 0.5*(ef.x1f(m,k,j,i) + ef.x1f(m,k,j,i+1));
    ec(m,1,k,j,i) = 0.5*(ef.x2f(m,k,j,i) + ef.x2f(m,k,j+1,i));
    ec(m,2,k,j,i) = 0.5*(ef.x3f(m,k,j,i) + ef.x3f(m,k+1,j,i));
    // vector potential on edges: A1 at (x1c,x2f,x3f), A2 at (x1f,x2c,x3f),
    // A3 at (x1f,x2f,x3c); B = curl A (discrete), so div B = 0 to roundoff.
    auto A1 = [&](Real x2, Real x3) { return sin(2.0*x2 + 0.3)*cos(x3) + x2*x3; };
    auto A2 = [&](Real x1, Real x3) { return cos(1.3*x1)*sin(x3 + 0.1) + x1*x3*x3; };
    auto A3 = [&](Real x1, Real x2) { return sin(x1 - x2) + 0.5*x1*x1*x2; };
    const Real dx1 = sz.dx1, dx2 = sz.dx2, dx3 = sz.dx3;
    auto B1 = [&](Real x1f) {
      (void)x1f;
      Real b = 0.0;
      if (use2) b += (A3(x1f, x2r) - A3(x1f, x2l))/dx2;
      if (use3) b -= (A2(x1f, x3r) - A2(x1f, x3l))/dx3;
      return b;
    };
    auto B2 = [&](Real x2f) {
      Real b = -(A3(x1r, x2f) - A3(x1l, x2f))/dx1;
      if (use3) b += (A1(x2f, x3r) - A1(x2f, x3l))/dx3;
      return b;
    };
    auto B3 = [&](Real x3f) {
      Real b = (A2(x1r, x3f) - A2(x1l, x3f))/dx1;
      if (use2) b -= (A1(x2r, x3f) - A1(x2l, x3f))/dx2;
      return b;
    };
    bf.x1f(m,k,j,i) = B1(x1l);
    bf.x1f(m,k,j,i+1) = B1(x1r);
    bf.x2f(m,k,j,i) = B2(x2l);
    bf.x2f(m,k,j+1,i) = B2(x2r);
    bf.x3f(m,k,j,i) = B3(x3l);
    bf.x3f(m,k+1,j,i) = B3(x3r);
    (void)x1c; (void)x2c; (void)x3c;
  });

  const Real vol = (ms.x1max - ms.x1min)*(ms.x2max - ms.x2min)*(ms.x3max - ms.x3min);
  resistive::ChargeDiag cq = resistive::TotalCharge(pmbp, ef);
  // The sin/cos parts of E~ are divergence-free per component direction only in the
  // continuum; the discrete divergence of e1(x2,x3) along x1 is exactly zero, etc.
  Report("D  total charge == qexact*V", fabs(cq.total - qexact*vol)/(fabs(qexact)*vol),
         1.0e-12);
  Report("D  total charge == boundary flux", fabs(cq.total - cq.boundary)/
         (fabs(cq.boundary) + 1.0), 1.0e-12);
  Report("D  max |q~| == |qexact|", fabs(cq.max_abs_q - fabs(qexact))/fabs(qexact),
         1.0e-10);

  const Real dbmax = resistive::MaxDivB(pmbp, bf);
  Report("D  max |div B| (CT-built B)", dbmax, 1.0e-10);

  resistive::MismatchDiag md = resistive::ECellFaceMismatch(pmbp, ec, ef);
  Report("D  |E_c - avg E_f| == 0", md.max_abs, 0.0);
  const Real delta = 3.25e-3;
  auto ec_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), ec);
  if (nmb > 0) ec_h(0,1,ks,js,is) += delta;
  Kokkos::deep_copy(ec, ec_h);
  md = resistive::ECellFaceMismatch(pmbp, ec, ef);
  // every rank perturbs one cell of its first MeshBlock, so the global max is delta
  Report("D  |E_c - avg E_f| detects perturbation", fabs(md.max_abs - delta)/delta,
         1.0e-10);

  // Newton counters
  resistive::NewtonStats ns;
  ns.Reset();
  const int ncell = 1000;
  Kokkos::parallel_for("resist_ut_newton", Kokkos::RangePolicy<>(DevExeSpace(), 0, ncell),
  KOKKOS_LAMBDA(const int n) {
    ns.Record(n % 7, (n % 11) == 0, (n % 101) == 0);
  });
  int64_t c[resistive::kNewtonNCounters];
  ns.Read(c);
  int64_t isum = 0, nfb = 0, nfl = 0;
  for (int n = 0; n < ncell; ++n) {
    isum += n % 7; nfb += ((n % 11) == 0); nfl += ((n % 101) == 0);
  }
  const int64_t nr = global_variable::nranks;
  const bool ok = (c[resistive::kNewtonCells] == nr*ncell) &&
                  (c[resistive::kNewtonIterSum] == nr*isum) &&
                  (c[resistive::kNewtonIterMax] == 6) &&
                  (c[resistive::kNewtonFallback] == nr*nfb) &&
                  (c[resistive::kNewtonFail] == nr*nfl);
  Report("D  Newton counters", ok ? 0.0 : 1.0, 0.0);
}

//----------------------------------------------------------------------------------------
// K4: implicit Ohmic step (closed form) and K5: resistive c2p (Newton-Broyden)

// K4a: the closed form solves E = R - sdt [W E + u x B - (E.u) v] (sdt in 1e-3..1e8)
KOKKOS_INLINE_FUNCTION
Real K4ResidualTrial(const int t) {
  using namespace resistive;  // NOLINT
  RandomState r;
  MakeRandomState(t, r);
  uint64_t s = 0xA11CE000ULL + static_cast<uint64_t>(t)*104729ULL;
  const Real sdt = pow(10.0, Uniform(s, -3.0, 8.0));
  const Real sg = SqrtDet(r.g);
  Real E[3];
  OhmicE(r.g, sg, sdt, r.wv, r.Bcode, r.Ernd, E);
  const Real W = LorentzFromU(r.g, r.wv);
  Real ud[3], Bd[3], uxb[3];
  Lower(r.g, r.wv, ud);
  Lower(r.g, r.Bcode, Bd);
  CrossUp(sg, ud, Bd, uxb);
  const Real eu = Dot(r.g, E, r.wv);
  Real res = 0.0, scale = 0.0;
  for (int a = 0; a < 3; ++a) {
    const Real v = r.wv[a]/W;
    res = fmax(res, fabs(E[a] - (r.Ernd[a] - sdt*(W*E[a] + uxb[a] - eu*v))));
    scale = fmax(scale, fabs(r.Ernd[a]) +
                 sdt*(W*fabs(E[a]) + fabs(uxb[a]) + fabs(eu*v)));
  }
  return res/scale;
}

// K4b: sigma -> 0 leaves E unchanged (exactly)
KOKKOS_INLINE_FUNCTION
Real K4VacuumTrial(const int t) {
  using namespace resistive;  // NOLINT
  RandomState r;
  MakeRandomState(t, r);
  Real E[3];
  OhmicE(r.g, SqrtDet(r.g), 0.0, r.wv, r.Bcode, r.Ernd, E);
  return fmax(fabs(E[0] - r.Ernd[0]), fmax(fabs(E[1] - r.Ernd[1]),
                                           fabs(E[2] - r.Ernd[2])));
}

// K4c: sigma -> inf gives E -> -v x B with error proportional to 1/(sigma dt):
// returns |log10(err(sdt=1e4)/err(sdt=1e6)) - 2|
KOKKOS_INLINE_FUNCTION
Real K4IdealLimitTrial(const int t) {
  using namespace resistive;  // NOLINT
  RandomState r;
  MakeRandomState(t, r);
  const Real sg = SqrtDet(r.g);
  const Real W = LorentzFromU(r.g, r.wv);
  Real v[3] = {r.wv[0]/W, r.wv[1]/W, r.wv[2]/W};
  Real Eid[3];
  IdealE(r.g, sg, v, r.Bcode, Eid);
  Real err[2];
  const Real sdts[2] = {1.0e4, 1.0e6};
  for (int n = 0; n < 2; ++n) {
    Real E[3];
    OhmicE(r.g, sg, sdts[n], r.wv, r.Bcode, r.Ernd, E);
    Real d[3] = {E[0] - Eid[0], E[1] - Eid[1], E[2] - Eid[2]};
    err[n] = sqrt(Dot(r.g, d, d));
  }
  return fabs(log10(err[0]/err[1]) - 2.0);
}

struct CellTrialOut {
  Real err;    // max relative error vs the reference state (converged trials)
  Real cmis;   // max relative mismatch of the solved state's totals vs the input totals
  Real sigm;   // magnetisation B^2/rho of the trial
  Real rho, temp, ye;  // input fluid state (temp: solved T for tabulated EOS)
  Real cont;   // pre-step mode: |u - u_continuation|/(1 + |u_continuation|)
  Real sdt, w;  // sigma dt and Lorentz factor of the trial
  int pre;     // the E = R state was flagged inadmissible by the solver
  int iters;
  int calls;
  int status;  // 0 converged, 1 fallback, 2 failed
  int retry;   // option C moved the Newton start
  int folded;  // pre-step mode: the dt-continuation could not be completed
};

// K5 sampling configuration (07 v2.2 §4.1 K5): residual form, sampling mode, sigma(rho)
struct K5Config {
  resistive::CellSolveOptions opt;
  int mode = 0;          // 0 fixed point (final state given, R backed out), 1 pre-step
  bool sigrho = false;   // sigma dt = sigma(rho) dt, rho drawn across the transition
  resistive::Conductivity cond;
  Real lrhot = 0.0;      // log10 rho_t (ideal gas: rho drawn in lrhot +- 3)
};

// Thermodynamic state of a K5 trial: (rho, p, h, Y_e) and the enthalpy of a solved state.
// Ideal gas: rho, p from the random state. Tabulated: rho, T, Y_e drawn inside the table.
KOKKOS_INLINE_FUNCTION
void TrialThermo(const PS &, const RandomState &r, uint64_t &, const Real, Real &rho,
                 Real &p, Real &h, Real &ye) {
  rho = r.rho;
  p = r.p;
  h = 1.0 + kGam/(kGam - 1.0)*p/rho;
  ye = 0.0;
}

KOKKOS_INLINE_FUNCTION
Real SolvedEnthalpy(const PS &ps, const Real prim[NPRIM]) {
  const Real rho = prim[PRH]*ps.GetEOS().GetBaryonMass();
  return 1.0 + kGam/(kGam - 1.0)*prim[PPR]/rho;
}

KOKKOS_INLINE_FUNCTION
void TrialThermo(const PST &ps, const RandomState &, uint64_t &s, const Real tc,
                 Real &rho, Real &p, Real &h, Real &ye) {
  // log-uniform in n and T, uniform in Y_e, strictly inside the table (n in EOS units;
  // tc converts the table temperature to code units)
  const auto &eos = ps.GetEOS();
  const Real n = pow(10.0, Uniform(s, log10(eos.GetMinimumDensity()) + 1.0,
                                   log10(eos.GetMaximumDensity()) - 1.0));
  const Real T = pow(10.0, Uniform(s, log10(eos.GetMinimumTemperature()*tc) + 0.3,
                                   log10(eos.GetMaximumTemperature()*tc) - 0.3));
  rho = n*eos.GetBaryonMass();
  Real Y[MAX_SPECIES] = {0.0};
  ye = Uniform(s, eos.GetMinimumSpeciesFraction(0) + 0.02,
               eos.GetMaximumSpeciesFraction(0) - 0.02);
  Y[0] = ye;
  p = eos.GetPressure(n, T, Y);
  h = eos.GetEnthalpy(n, T, Y);
}

KOKKOS_INLINE_FUNCTION
Real SolvedEnthalpy(const PST &ps, const Real prim[NPRIM]) {
  Real Y[MAX_SPECIES] = {0.0};
  Y[0] = prim[PYF];
  return ps.GetEOS().GetEnthalpy(prim[PRH], prim[PTM], Y);
}

// sigma(rho) mode: the ideal-gas rho is redrawn log-uniform over lrhot +- 3 decades at
// fixed p/rho; the tabulated draw already spans the table (lrhot = its mid point).
KOKKOS_INLINE_FUNCTION
void SigRhoThermo(const PS &, uint64_t &s, const Real lrhot, Real &rho, Real &p, Real &) {
  const Real rn = pow(10.0, Uniform(s, lrhot - 3.0, lrhot + 3.0));
  p *= rn/rho;
  rho = rn;
}
KOKKOS_INLINE_FUNCTION
void SigRhoThermo(const PST &, uint64_t &, const Real, Real &, Real &, Real &) {}

// Continue the implicit root in sigma dt -> s sigma dt from s = 0 (where the root is the
// pre-step state u0 itself, E = R) to s = 1, adaptive in log s (notes repo
// scripts/implicit_roots_check.py continue_in_dt). Returns false at a fold.
template <class PSolver>
KOKKOS_INLINE_FUNCTION
bool ContinueInDt(const PSolver &ps, const Real g[NSPMETRIC], const Real gi[NSPMETRIC],
                  const Real sdt, const Real cons[NCONS], const Real B[3],
                  const Real R[3], const Real u0[3], resistive::CellSolveOptions opt,
                  Real u[3]) {
  using resistive::Dot;
  opt.retry = false;
  opt.continuation = false;
  for (int a = 0; a < 3; ++a) u[a] = u0[a];
  Real ls = -8.0, dls = 8.0/200.0;
  for (int guard = 0; guard < 20000 && ls < 0.0; ++guard) {
    const Real lt = fmin(ls + dls, 0.0);
    Real prim[NPRIM], E[3];
    const resistive::CellSolveResult res = resistive::ResistiveCellSolve(
        ps, g, gi, sdt*pow(10.0, lt), cons, B, R, u, 1.0e-11, 30, false, prim, E, opt);
    Real du[3] = {prim[PVX] - u[0], prim[PVY] - u[1], prim[PVZ] - u[2]};
    if (!res.converged || sqrt(Dot(g, du, du)) > 0.05*(1.0 + sqrt(Dot(g, u, u)))) {
      dls *= 0.5;
      if (dls < 1.0e-7) return false;
      continue;
    }
    for (int a = 0; a < 3; ++a) u[a] = prim[PVX+a];
    ls = lt;
    dls = fmin(1.5*dls, 0.1);
  }
  return !(ls < 0.0);
}

// Build totals from an exact state that is a fixed point of the implicit step
// (E* = closed form at the true u), then solve from a perturbed guess.
// lsig: log10 range of the magnetisation B^2/rho; lsdt: log10 range of sigma dt.
template <class PSolver>
KOKKOS_INLINE_FUNCTION
CellTrialOut CellSolveTrial(const PSolver &ps, const int t, const Real lsig_lo,
                            const Real lsig_hi, const Real lsdt_lo, const Real lsdt_hi,
                            const Real wmax, const Real pert, const Real tconv = 1.0,
                            const K5Config cfg = K5Config()) {
  using namespace resistive;  // NOLINT
  RandomState r;
  MakeRandomState(t, r);
  uint64_t s = 0xBEEF0000ULL + static_cast<uint64_t>(t)*15485863ULL;
  Real sdt = pow(10.0, Uniform(s, lsdt_lo, lsdt_hi));
  const Real sigm = pow(10.0, Uniform(s, lsig_lo, lsig_hi));
  Real rho, prs, h, ye;
  TrialThermo(ps, r, s, tconv, rho, prs, h, ye);
  if (cfg.sigrho) {
    SigRhoThermo(ps, s, cfg.lrhot, rho, prs, h);
    sdt = cfg.cond(rho).par*pow(10.0, Uniform(s, -4.0, -1.0));
  }
  const Real sg = SqrtDet(r.g);
  Real gi[NSPMETRIC];
  Inverse3(r.g, gi);
  // four-velocity with W in [1, wmax]
  const Real umag = Uniform(s, 0.0, sqrt(wmax*wmax - 1.0));
  const Real u0n = sqrt(Dot(r.g, r.wv, r.wv));
  Real u[3];
  for (int a = 0; a < 3; ++a) u[a] = r.wv[a]*umag/fmax(u0n, 1.0e-12);
  // field with B^2/rho = sigm
  Real B[3];
  const Real bn = sqrt(Dot(r.g, r.Bcode, r.Bcode));
  for (int a = 0; a < 3; ++a) B[a] = r.Bcode[a]*sqrt(sigm*rho)/bn;
  Real R[3];
  Real Es[3];
  if (cfg.mode == 0) {
    // fixed point: the final state (u, E*) is given and R is backed out
    for (int a = 0; a < 3; ++a) R[a] = r.Ernd[a]*sqrt(sigm*rho);
    OhmicE(r.g, sg, sdt, u, B, R, Es);
  } else {
    // pre-step: admissible state with E = R = -v x B + non-ideal part up to ~|B|
    const Real W = sqrt(1.0 + Dot(r.g, u, u));
    Real v[3] = {u[0]/W, u[1]/W, u[2]/W};
    IdealE(r.g, sg, v, B, Es);
    const Real nu = Uniform(s, 0.0, 1.0)*sqrt(sigm*rho);
    for (int a = 0; a < 3; ++a) {
      Es[a] += nu*r.Ernd[a];
      R[a] = Es[a];
    }
  }
  Real idn, im[3], ien;
  PrimToConsDyn(r.g, rho, prs, h, u, Es, B, idn, im, ien);
  Real cons[NCONS];
  for (int n = 0; n < NCONS; ++n) cons[n] = 0.0;
  cons[CDN] = idn/sg;
  for (int a = 0; a < 3; ++a) cons[CSX+a] = im[a]/sg;
  cons[CTA] = ien/sg;
  cons[CYD] = cons[CDN]*ye;
  Real ug[3];
  for (int a = 0; a < 3; ++a) {
    ug[a] = u[a]*(1.0 + pert*Uniform(s, -1.0, 1.0)) + pert*Uniform(s, -1.0, 1.0);
  }
  if (cfg.mode == 1) {
    for (int a = 0; a < 3; ++a) ug[a] = u[a];  // option A: previous-stage u
  }
  Real prim[NPRIM], E[3];
  CellSolveResult res = ResistiveCellSolve(ps, r.g, gi, sdt, cons, B, R, ug, 1.0e-11, 30,
                                           true, prim, E, cfg.opt);
  CellTrialOut out;
  out.retry = res.retry ? 1 : 0;
  out.sdt = sdt;
  out.w = sqrt(1.0 + Dot(r.g, u, u));
  out.pre = res.prestep_inadmissible ? 1 : 0;
  out.folded = 0;
  out.cont = 0.0;
  out.iters = res.iterations;
  out.calls = res.c2p_calls;
  out.status = res.converged ? 0 : (res.fallback ? 1 : 2);
  out.err = 0.0;
  out.cmis = 0.0;
  out.sigm = sigm;
  out.rho = rho;
  out.temp = prim[PTM];
  out.ye = ye;
  if (res.converged) {
    const Real us = 1.0 + sqrt(Dot(r.g, u, u));
    const Real es = sqrt(Dot(r.g, Es, Es)) + sqrt(Dot(r.g, B, B));
    for (int a = 0; a < 3; ++a) {
      out.err = fmax(out.err, fabs(prim[PVX+a] - u[a])/us);
      out.err = fmax(out.err, fabs(E[a] - Es[a])/es);
    }
    const Real mb = ps.GetEOS().GetBaryonMass();
    out.err = fmax(out.err, RelErr(prim[PRH]*mb, rho, rho));
    // A root is valid if (prim, E(u)) reproduces the input totals. At high sigma_M the
    // implicit system can have more than one such root (see STATUS / proposals).
    {
      const Real rho2 = prim[PRH]*mb, p2 = prim[PPR];
      const Real h2 = SolvedEnthalpy(ps, prim);
      Real u2[3] = {prim[PVX], prim[PVY], prim[PVZ]};
      Real E2[3];
      OhmicE(r.g, sg, sdt, u2, B, R, E2);
      Real idn2, im2[3], ien2;
      PrimToConsDyn(r.g, rho2, p2, h2, u2, E2, B, idn2, im2, ien2);
      Real cm = fabs(idn2 - idn)/fabs(idn);
      const Real ms = fabs(im[0]) + fabs(im[1]) + fabs(im[2]);
      for (int a = 0; a < 3; ++a) cm = fmax(cm, fabs(im2[a] - im[a])/ms);
      cm = fmax(cm, fabs(ien2 - ien)/fabs(ien));
      out.cmis = cm;
    }
  }
  // pre-step: reference continuation root (for fallbacks it tells whether a root on the
  // continuation branch exists at all, e.g. inside the EOS table)
  if (cfg.mode == 1 && !res.failed) {
    Real uc[3];
    if (ContinueInDt(ps, r.g, gi, sdt, cons, B, R, u, cfg.opt, uc)) {
      Real du[3] = {prim[PVX] - uc[0], prim[PVY] - uc[1], prim[PVZ] - uc[2]};
      out.cont = sqrt(Dot(r.g, du, du))/(1.0 + sqrt(Dot(r.g, uc, uc)));
    } else {
      out.folded = 1;
    }
  }
  return out;
}

template <class ExecSpace>
void RunK4(const std::string &where, const int ntrial) {
  PS ps = MakeSolver();
  Real e = MaxOverTrials<ExecSpace>("K4a", ntrial,
      KOKKOS_LAMBDA(const int t) { return K4ResidualTrial(t); });
  Report("K4a closed form solves implicit eq. [" + where + "]", e, 1.0e-13);
  e = MaxOverTrials<ExecSpace>("K4b", ntrial,
      KOKKOS_LAMBDA(const int t) { return K4VacuumTrial(t); });
  Report("K4b sigma->0: E unchanged [" + where + "]", e, 0.0);
  e = MaxOverTrials<ExecSpace>("K4c", ntrial,
      KOKKOS_LAMBDA(const int t) { return K4IdealLimitTrial(t); });
  Report("K4c sigma->inf: |E+vxB| ~ 1/(sigma dt) [" + where + "]", e, 2.0e-2);
  // Newton from the exact u stops at the lagged-v closed form; from a nearby guess it
  // converges locally to the same root.
  e = MaxOverTrials<ExecSpace>("K4d", ntrial, KOKKOS_LAMBDA(const int t) {
    CellTrialOut o = CellSolveTrial(ps, t, -2.0, 2.0, -2.0, 4.0, 5.0, 0.0);
    return (o.status == 0 && o.iters == 0) ? o.err : 1.0;
  });
  Report("K4d guess = exact: lagged-v, 0 iterations [" + where + "]", e, 1.0e-10);
  e = MaxOverTrials<ExecSpace>("K4e", ntrial, KOKKOS_LAMBDA(const int t) {
    CellTrialOut o = CellSolveTrial(ps, t, -2.0, 2.0, -2.0, 4.0, 5.0, 1.0e-6);
    return (o.status == 0) ? o.err : 1.0;
  });
  Report("K4e Newton local convergence to fixed point [" + where + "]", e, 1.0e-9);
}

// K5 (07 v2.2 §4.1): Newton statistics over sigma_M = B^2/rho in 1e-2..1e4, W <= 10,
// for both residual forms (g: hydro c2p, f: Mignone momentum) and three samplings:
//  - fixed point: final state given, R backed out; guesses perturbed by 5%, 1%, 0.1%;
//  - pre-step: admissible state with E = R, guess = that state (option A), and the
//    selected root compared with the sigma dt continuation from s = 0;
//  - pre-step with a sigma(rho) profile (Step 1b): rho across the transition,
//    sigma dt = sigma(rho) dt.
// Ideal gas, and the CompOSE table <problem> k5_table if given.
// Gates: converged roots reproduce the totals (1e-9); no hard failures; pre-step:
// selected root = continuation root, and no fallback for sigma_M <= 10.
// Reported: fallback, C-retry, non-continuation-root and fold counts per sigma_M decade.
template <class PSolver>
void RunK5(const PSolver &ps, const std::string &eosname, const int ntrial,
           const Real tconv, const Real lrhot) {
  using resistive::kResidualC2P;
  using resistive::kResidualMomentum;
  Kokkos::View<CellTrialOut*, DevExeSpace> out("k5out", ntrial);
  resistive::ResistiveParams pd;
  pd.sigma_model = resistive::SigmaModel::density;
  pd.sigma_in = 1.0e6; pd.sigma_out = 1.0e-1;
  pd.rho_transition = pow(10.0, lrhot); pd.sigma_width = 0.5;
  const resistive::Conductivity cond(pd);
  constexpr int nd = 6;  // sigma_M decades [1e-2,1e-1), ..., [1e3,1e4]
  for (const int form : {kResidualC2P, kResidualMomentum}) {
    const std::string fname = (form == kResidualC2P) ? "g" : "f";
    // (mode, sigrho, pert)
    struct Run { int mode; bool sigrho; Real pert; };
    const Run runs[5] = {{0, false, 0.05}, {0, false, 0.01}, {0, false, 0.001},
                         {1, false, 0.0}, {1, true, 0.0}};
    for (const Run &rn : runs) {
      K5Config cfg;
      cfg.opt.form = form;
      cfg.mode = rn.mode;
      cfg.sigrho = rn.sigrho;
      cfg.cond = cond;
      cfg.lrhot = lrhot;
      const Real pert = rn.pert;
      Kokkos::parallel_for("K5", Kokkos::RangePolicy<DevExeSpace>(0, ntrial),
      KOKKOS_LAMBDA(const int t) {
        out(t) = CellSolveTrial(ps, t, -2.0, 4.0, -3.0, 6.0, 10.0, pert, tconv, cfg);
      });
      auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
      int nconv = 0, nfb = 0, nfb0 = 0, nfail = 0, nother = 0, itmax = 0;
      double itsum = 0.0, callsum = 0.0;
      Real cmismax = 0.0;
      int tworst = 0;
      int nsig[nd] = {0}, nfbd[nd] = {0}, nrtd[nd] = {0}, nncd[nd] = {0}, nfod[nd] = {0};
      int nmsd[nd] = {0};  // pre-step fallbacks although a continuation root exists
      for (int t = 0; t < ntrial; ++t) {
        const int d = std::min(nd - 1, std::max(0,
                               static_cast<int>(floor(log10(h(t).sigm))) + 2));
        ++nsig[d];
        nrtd[d] += h(t).retry;
        if (h(t).status == 0) {
          ++nconv;
          itsum += h(t).iters;
          callsum += h(t).calls;
          itmax = std::max(itmax, h(t).iters);
          if (h(t).cmis > cmismax) { cmismax = h(t).cmis; tworst = t; }
          if (rn.mode == 0 && h(t).err > 1.0e-6) ++nother;
          if (rn.mode == 1 && h(t).folded) ++nfod[d];
          if (rn.mode == 1 && !h(t).folded && h(t).cont > 1.0e-6) { ++nother; ++nncd[d]; }
        } else if (h(t).status == 1) {
          ++nfb;
          ++nfbd[d];
          if (rn.mode == 1 && !h(t).folded) ++nmsd[d];
          if (h(t).iters == 0) ++nfb0;
        } else {
          ++nfail;
        }
      }
      std::string label = "K5 " + eosname + " " + fname + " ";
      label += (rn.mode == 0) ? "fixed-point pert " + std::to_string(pert)
                              : (rn.sigrho ? "pre-step sigma(rho)" : "pre-step");
      if (global_variable::my_rank == 0) {
        std::cout << "  " << label << ": converged " << nconv << "/" << ntrial
                  << ", fallback " << nfb << " (" << nfb0 << " before any Newton step)"
                  << ", failed " << nfail << ", mean iters " << std::fixed
                  << std::setprecision(2) << itsum/std::max(nconv, 1) << ", max iters "
                  << itmax << ", mean c2p calls " << callsum/std::max(nconv, 1)
                  << ((rn.mode == 0) ? ", other root " : ", non-continuation root ")
                  << nother << std::endl
                  << "     per sigma_M decade [1e-2..1e4] "
                  << "trials/fallback/retry/non-cont/folded/missed:";
        for (int d = 0; d < nd; ++d) {
          std::cout << " " << nsig[d] << "/" << nfbd[d] << "/" << nrtd[d] << "/"
                    << nncd[d] << "/" << nfod[d] << "/" << nmsd[d];
        }
        int nshow = 0;
        for (int t = 0; t < ntrial && rn.mode == 1 && nshow < 4; ++t) {
          if (h(t).status == 1 && h(t).sigm <= 10.0) {
            if (nshow++ == 0) {
              std::cout << std::endl << "     fallbacks at sigma_M <= 10:";
            }
            std::cout << std::endl << std::scientific << std::setprecision(3)
                      << "       rho " << h(t).rho << " T " << h(t).temp << " Ye "
                      << h(t).ye << " sigma_M " << h(t).sigm << " sigma dt " << h(t).sdt
                      << " W " << h(t).w << " E=R inadmissible " << h(t).pre;
          }
        }
        std::cout << std::endl << "     worst totals mismatch: rho " << std::scientific
                  << h(tworst).rho << " T " << h(tworst).temp << " Ye " << h(tworst).ye
                  << " sigma_M " << h(tworst).sigm << " iters " << h(tworst).iters
                  << std::endl;
      }
      Report(label + " roots reproduce totals", cmismax, 1.0e-9);
      Report(label + " no hard failures", static_cast<Real>(nfail), 0.0);
      if (rn.mode == 1) {
        Report(label + " root = dt-continuation root", static_cast<Real>(nother), 0.0);
        // 07 v2.2: fallback fraction 0 for sigma_M <= 10. A fallback is a solver miss if
        // the reference continuation reaches s = 1; otherwise the continuation branch has
        // no admissible root (e.g. Ohmic heating leaves the EOS table, risk R5) and the
        // lagged-v fallback is the intended outcome. Both counts are reported; the gate
        // is on misses (see impl/STATUS.md and proposal k5-prestep-table-exit).
        if (global_variable::my_rank == 0) {
          std::cout << "     fallbacks sigma_M <= 10: "
                    << nfbd[0] + nfbd[1] + nfbd[2] << " (no continuation root: "
                    << nfbd[0] + nfbd[1] + nfbd[2] - nmsd[0] - nmsd[1] - nmsd[2] << ")"
                    << std::endl;
        }
        Report(label + " no missed root sigma_M <= 10",
               static_cast<Real>(nmsd[0] + nmsd[1] + nmsd[2]), 0.0);
      }
    }
  }
}

//----------------------------------------------------------------------------------------
// C3 (07 D3): the B~ face prolongation (ProlongFCShared*Face + Toth-Roe internal)
// applied to a random coarse E~ (q~ != 0) gives every fine cell exactly the divergence
// of its parent coarse cell (C++ analogue of scripts/toth_roe_div_test.py); the fine
// shared faces also restrict (area average) to the coarse face.
void RunC3(const int ntrial) {
  for (int dim = 2; dim <= 3; ++dim) {
    const bool three_d = (dim == 3);
    const int c3 = three_d ? 3 : 1, f3 = three_d ? 2 : 1;
    DvceFaceFld4D<Real> cb("c3_cb", ntrial, c3, 3, 3), fb("c3_fb", ntrial, f3, 2, 2);
    const int kc = three_d ? 1 : 0;
    Real err_div = 0.0, err_res = 0.0;
    Kokkos::parallel_reduce("C3", Kokkos::RangePolicy<>(DevExeSpace(), 0, ntrial),
    KOKKOS_LAMBDA(const int t, Real &ed, Real &er) {
      uint64_t s = 0xC3C30000ULL + static_cast<uint64_t>(t)*7919ULL +
                   static_cast<uint64_t>(dim);
      for (int k = 0; k < c3; ++k) {
        for (int j = 0; j < 3; ++j) {
          for (int i = 0; i < 4; ++i) cb.x1f(t, k, j, i) = Uniform(s, -1.0, 1.0);
        }
      }
      for (int k = 0; k < c3; ++k) {
        for (int j = 0; j < 4; ++j) {
          for (int i = 0; i < 3; ++i) cb.x2f(t, k, j, i) = Uniform(s, -1.0, 1.0);
        }
      }
      for (int k = 0; k < c3 + 1; ++k) {
        for (int j = 0; j < 3; ++j) {
          for (int i = 0; i < 3; ++i) cb.x3f(t, k, j, i) = Uniform(s, -1.0, 1.0);
        }
      }
      // shared faces of the central coarse cell (kc,1,1)
      for (int o = 0; o < 2; ++o) {
        ProlongFCSharedX1Face(t, kc, 1, 1 + o, 0, 0, 2*o, true, three_d, cb.x1f, fb.x1f);
        ProlongFCSharedX2Face(t, kc, 1 + o, 1, 0, 2*o, 0, three_d, cb.x2f, fb.x2f);
        if (three_d) {
          ProlongFCSharedX3Face(t, kc + o, 1, 1, 2*o, 0, 0, true, cb.x3f, fb.x3f);
        }
      }
      ProlongFCInternal(t, 0, 0, 0, three_d, fb);
      const Real cdiv = (cb.x1f(t, kc, 1, 2) - cb.x1f(t, kc, 1, 1)) +
                        (cb.x2f(t, kc, 2, 1) - cb.x2f(t, kc, 1, 1)) +
                        (three_d ? (cb.x3f(t, kc + 1, 1, 1) - cb.x3f(t, kc, 1, 1)) : 0.0);
      for (int k = 0; k < f3; ++k) {
        for (int j = 0; j < 2; ++j) {
          for (int i = 0; i < 2; ++i) {
            Real fdiv = (fb.x1f(t, k, j, i+1) - fb.x1f(t, k, j, i)) +
                        (fb.x2f(t, k, j+1, i) - fb.x2f(t, k, j, i));
            if (three_d) fdiv += fb.x3f(t, k+1, j, i) - fb.x3f(t, k, j, i);
            ed = fmax(ed, fabs(2.0*fdiv - cdiv));   // fine dx = coarse dx/2
          }
        }
      }
      // restriction of the fine shared faces
      const Real w = three_d ? 0.25 : 0.5;
      for (int o = 0; o < 2; ++o) {
        Real a1 = 0.0, a2 = 0.0, a3 = 0.0;
        for (int p = 0; p < f3; ++p) {
          for (int q = 0; q < 2; ++q) {
            a1 += w*fb.x1f(t, p, q, 2*o);
            a2 += w*fb.x2f(t, p, 2*o, q);
            if (three_d) a3 += w*fb.x3f(t, 2*o, p, q);
          }
        }
        er = fmax(er, fabs(a1 - cb.x1f(t, kc, 1, 1 + o)));
        er = fmax(er, fabs(a2 - cb.x2f(t, kc, 1 + o, 1)));
        if (three_d) er = fmax(er, fabs(a3 - cb.x3f(t, kc + o, 1, 1)));
      }
    }, Kokkos::Max<Real>(err_div), Kokkos::Max<Real>(err_res));
    const std::string d = three_d ? "3D" : "2D";
    Report("C3 fine q~ = coarse q~ (Toth-Roe, " + d + ")", err_div, 1.0e-13);
    Report("C3 fine shared faces restrict to coarse (" + d + ")", err_res, 1.0e-14);
  }
}

//----------------------------------------------------------------------------------------
// K7: edge 2D solver; K3 (edge part): ideal-limit e_k equals AthenaK e_cc

// K7a: unit light speeds reproduce Mignone et al. 2019 Eqs. 46-47
KOKKOS_INLINE_FUNCTION
Real K7MignoneTrial(const int t) {
  using resistive::EdgeUpwind;
  uint64_t s = 0x7ED6E000ULL + static_cast<uint64_t>(t)*2654435761ULL;
  Real ez[2][2], bz[2][2], hz[2][2];
  Real sez = 0.0, sbz = 0.0;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) {
      ez[i][j] = Uniform(s, -1., 1.); bz[i][j] = Uniform(s, -1., 1.);
      hz[i][j] = -bz[i][j];
      sez += ez[i][j]; sbz += bz[i][j];
    }
  const Real bxl = Uniform(s, -1., 1.), bxr = Uniform(s, -1., 1.);  // across y
  const Real byl = Uniform(s, -1., 1.), byr = Uniform(s, -1., 1.);  // across x
  const Real exl = Uniform(s, -1., 1.), exr = Uniform(s, -1., 1.);
  const Real eyl = Uniform(s, -1., 1.), eyr = Uniform(s, -1., 1.);
  const Real e = EdgeUpwind(ez, bxr - bxl, byr - byl, 1.0, 1.0, 1.0, 1.0);
  const Real eq46 = 0.25*sez - 0.5*(bxr - bxl) + 0.5*(byr - byl);
  const Real h = EdgeUpwind(hz, exr - exl, eyr - eyl, 1.0, 1.0, 1.0, 1.0);
  const Real eq47 = 0.25*sbz + 0.5*(exr - exl) - 0.5*(eyr - eyl);
  return fmax(fabs(e - eq46), fabs(-h - eq47));
}

// K7b: with states uniform along one transverse direction the edge value reduces to the
// 1D HLL flux in the other, with GR light-cone speeds (random metric, lapse, shift).
KOKKOS_INLINE_FUNCTION
Real K7ReductionTrial(const int t) {
  using namespace resistive;  // NOLINT
  RandomState r;
  MakeRandomState(t, r);
  Real gi[NSPMETRIC];
  Inverse3(r.g, gi);
  Real aap, aam, abp, abm;
  LightSpeeds(r.alpha, r.beta[0], gi[S11], aap, aam);
  LightSpeeds(r.alpha, r.beta[1], gi[S22], abp, abm);
  uint64_t s = 0x7ED60000ULL + static_cast<uint64_t>(t)*40503ULL;
  const Real fl = Uniform(s, -1., 1.), fr = Uniform(s, -1., 1.);
  const Real du = Uniform(s, -1., 1.);
  // uniform along b: 1D problem along a, flux of U^b along a is -f
  Real fa[2][2] = {{fl, fl}, {fr, fr}};
  const Real ea = EdgeUpwind(fa, 0.0, du, aap, aam, abp, abm);
  const Real hlla = (aap*fl + aam*fr + aap*aam*du)/(aap + aam);
  // uniform along a: 1D problem along b, flux of U^a along b is +f
  Real fb[2][2] = {{fl, fr}, {fl, fr}};
  const Real eb = EdgeUpwind(fb, du, 0.0, aap, aam, abp, abm);
  const Real hllb = (abp*fl + abm*fr - abp*abm*du)/(abp + abm);
  return fmax(fabs(ea - hlla), fabs(eb - hllb));
}

// K3e: with E = -v x B, e_k = alpha E_k + beta^a B~^b - beta^b B~^a equals
// B~^a (alpha v^b - beta^b) - B~^b (alpha v^a - beta^a)  (mhd_corner_e.cpp e_cc)
KOKKOS_INLINE_FUNCTION
Real K3EdgeIdealTrial(const int t) {
  using namespace resistive;  // NOLINT
  RandomState r;
  MakeRandomState(t, r);
  const Real sg = SqrtDet(r.g);
  const Real W = LorentzFromU(r.g, r.wv);
  Real v[3] = {r.wv[0]/W, r.wv[1]/W, r.wv[2]/W};
  Real E[3], Ed[3], bt[3];
  IdealE(r.g, sg, v, r.Bcode, E);
  Lower(r.g, E, Ed);
  for (int a = 0; a < 3; ++a) bt[a] = sg*r.Bcode[a];
  Real err = 0.0, scale = 0.0;
  for (int k = 0; k < 3; ++k) {
    const int a = (k + 1) % 3, b = (k + 2) % 3;
    const Real e = EdgeEMF(r.alpha, r.beta[a], r.beta[b], Ed[k], bt[a], bt[b]);
    const Real ecc = bt[a]*(r.alpha*v[b] - r.beta[b]) - bt[b]*(r.alpha*v[a] - r.beta[a]);
    err = fmax(err, fabs(e - ecc));
    scale = fmax(scale, fabs(bt[a]) + fabs(bt[b]));
  }
  return err/scale;
}

// K2a: face triad is orthonormal, right-handed, e_A^d = 0 (A = 1, 2), and the
// coordinate <-> triad maps of vectors and covectors round-trip (random metrics).
KOKKOS_INLINE_FUNCTION
Real K2TetradTrial(const int t) {
  using namespace resistive;  // NOLINT
  RandomState r;
  MakeRandomState(t, r);
  Real gu[NSPMETRIC];
  Inverse3(r.g, gu);
  const int d = t % 3;
  FaceTetrad T;
  BuildFaceTetrad(r.g, gu, r.alpha, d, T);
  Real err = 0.0;
  for (int A = 0; A < 3; ++A) {
    for (int B = 0; B < 3; ++B) {
      const Real o = T.w[A][0]*T.e[B][0] + T.w[A][1]*T.e[B][1] + T.w[A][2]*T.e[B][2];
      err = fmax(err, fabs(o - ((A == B) ? 1.0 : 0.0)));
    }
    if (A > 0) err = fmax(err, fabs(T.e[A][d]));
  }
  const int c0 = d, c1 = (d + 1) % 3, c2 = (d + 2) % 3;
  const Real det = T.e[0][c0]*(T.e[1][c1]*T.e[2][c2] - T.e[1][c2]*T.e[2][c1])
                 - T.e[0][c1]*(T.e[1][c0]*T.e[2][c2] - T.e[1][c2]*T.e[2][c0])
                 + T.e[0][c2]*(T.e[1][c0]*T.e[2][c1] - T.e[1][c1]*T.e[2][c0]);
  if (!(det > 0.0)) err = 1.0;
  err = fmax(err, fabs(T.a - r.alpha*sqrt(gu[SymIdx(d, d)])));
  Real vt[3], v2[3], st[3], s2[3];
  VecToTetrad(T, r.Ernd, vt);
  VecFromTetrad(T, vt, v2);
  CovToTetrad(T, r.Bcode, st);
  CovFromTetrad(T, st, s2);
  for (int a = 0; a < 3; ++a) {
    err = fmax(err, fabs(v2[a] - r.Ernd[a]));
    err = fmax(err, fabs(s2[a] - r.Bcode[a]));
  }
  const Real n2 = vt[0]*vt[0] + vt[1]*vt[1] + vt[2]*vt[2];
  err = fmax(err, fabs(n2 - Dot(r.g, r.Ernd, r.Ernd))/n2);
  return err;
}

KOKKOS_INLINE_FUNCTION
void RandomSRSide(uint64_t &s, const Real en, const Real bn, resistive::MHLLCSide &x) {
  const Real rho = Uniform(s, 0.1, 2.0), p = Uniform(s, 0.01, 2.0);
  Real v[3];
  for (int a = 0; a < 3; ++a) v[a] = Uniform(s, -0.55, 0.55);
  const Real W = 1.0/sqrt(1.0 - (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]));
  const Real h = 1.0 + kGam/(kGam - 1.0)*p/rho;
  x.D = rho*W;
  for (int a = 0; a < 3; ++a) {
    x.v[a] = v[a];
    x.Q[a] = rho*h*W*W*v[a];
  }
  const Real tau = rho*h*W*W - p - x.D;
  x.Eh = tau + x.D;
  x.p = p;
  x.cs2 = kGam*p/(rho*h);
  x.E[0] = en;
  x.B[0] = bn;
  for (int a = 1; a < 3; ++a) {
    x.E[a] = Uniform(s, -1., 1.);
    x.B[a] = Uniform(s, -1., 1.);
  }
}

// K2b: SR limit. With alpha = 1, beta = 0, gamma = delta, the coordinate MHLLC flux in
// each direction d reproduces the flat (triad) MHLLC bitwise, components permuted.
KOKKOS_INLINE_FUNCTION
Real K2SRLimitTrial(const int t) {
  using namespace resistive;  // NOLINT
  uint64_t s = 0x2B2B0000ULL + static_cast<uint64_t>(t)*104729ULL;
  const Real en = Uniform(s, -1., 1.), bn = Uniform(s, -1., 1.);
  MHLLCSide L, R;
  RandomSRSide(s, en, bn, L);
  RandomSRSide(s, en, bn, R);
  Real U0[5 + MAX_SPECIES], H0[5 + MAX_SPECIES];
  bool ok;
  MHLLCTetrad(L, R, 0.0, 0, U0, H0, ok);
  const Real g[NSPMETRIC] = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0};
  const Real beta[3] = {0.0, 0.0, 0.0};
  Real err = 0.0;
  for (int d = 0; d < 3; ++d) {
    const int c[3] = {d, (d + 1) % 3, (d + 2) % 3};
    FaceTetrad T;
    BuildFaceTetrad(g, g, 1.0, d, T);
    MHLLCSide side[2];
    for (int k = 0; k < 2; ++k) {
      const MHLLCSide &x = (k == 0) ? L : R;
      Real S[3], v[3], E[3], B[3];
      for (int A = 0; A < 3; ++A) {
        S[c[A]] = x.Q[A];
        v[c[A]] = x.v[A];
        E[c[A]] = x.E[A];
        B[c[A]] = x.B[A];
      }
      FillMHLLCSide(T, x.D, S, x.Eh - x.D, x.p, v, x.cs2, x.Y, 0, E, B, side[k]);
    }
    Real U[5 + MAX_SPECIES], H[5 + MAX_SPECIES], flx[NCONS];
    MHLLCTetrad(side[0], side[1], beta[d]/T.a, 0, U, H, ok);
    MHLLCCoordFlux(T, 1.0, beta[d], 0, U, H, flx);
    err = fmax(err, fabs(flx[CDN] - H0[0]));
    for (int A = 0; A < 3; ++A) err = fmax(err, fabs(flx[CSX + c[A]] - H0[1 + A]));
    err = fmax(err, fabs(flx[CTA] - H0[4]));
  }
  return err;
}

// K2c: HLLC resolves an isolated stationary contact exactly (no EM jump): mass and
// energy fluxes vanish and the momentum flux is p (LF/HLL would diffuse it).
KOKKOS_INLINE_FUNCTION
Real K2ContactTrial(const int t) {
  using namespace resistive;  // NOLINT
  uint64_t s = 0xC0C00000ULL + static_cast<uint64_t>(t)*7907ULL;
  const Real en = Uniform(s, -1., 1.), bn = Uniform(s, -1., 1.);
  MHLLCSide L, R;
  RandomSRSide(s, en, bn, L);
  RandomSRSide(s, en, bn, R);
  R.p = L.p;
  for (int a = 0; a < 3; ++a) {
    L.v[a] = R.v[a] = 0.0;
    L.Q[a] = R.Q[a] = 0.0;
    R.E[a] = L.E[a];
    R.B[a] = L.B[a];
  }
  const Real rl = Uniform(s, 0.1, 1.0), rr = Uniform(s, 1.0, 10.0);
  const Real hl = 1.0 + kGam/(kGam - 1.0)*L.p/rl, hr = 1.0 + kGam/(kGam - 1.0)*R.p/rr;
  L.D = rl;
  R.D = rr;
  L.Eh = rl*hl - L.p;
  R.Eh = rr*hr - R.p;
  L.cs2 = kGam*L.p/(rl*hl);
  R.cs2 = kGam*R.p/(rr*hr);
  Real U[5 + MAX_SPECIES], H[5 + MAX_SPECIES], Uem[4], Fem[4];
  bool ok;
  MHLLCTetrad(L, R, 0.0, 0, U, H, ok);
  EMStateFlux(L.E, L.B, Uem, Fem);
  Real err = ok ? 0.0 : 1.0;
  err = fmax(err, fabs(H[0]));
  err = fmax(err, fabs(H[1] - Fem[0] - L.p)/L.p);
  err = fmax(err, fabs(H[2] - Fem[1]));
  err = fmax(err, fabs(H[3] - Fem[2]));
  err = fmax(err, fabs(H[4] - Fem[3]));
  return err;
}

// K3 (cell): ideal limit E = -v x B, single state (L = R): the resistive MHLLC face flux
// of the total T^{mu nu} rows equals the physical flux of the ideal dyngr solver
// (SingleStateLLF_DYNGR with identical states has no dissipative part).
KOKKOS_INLINE_FUNCTION
Real K3CellTrial(const PSH &eosh, const int t) {
  using namespace resistive;  // NOLINT
  RandomState r;
  MakeRandomState(t, r);
  const int d = t % 3;
  Real gu[NSPMETRIC];
  Inverse3(r.g, gu);
  const Real sg = SqrtDet(r.g);
  Real prim[NPRIM];
  FillPrim(eosh.ps, r, prim);
  const Real W = LorentzFromU(r.g, r.wv);
  const Real v[3] = {r.wv[0]/W, r.wv[1]/W, r.wv[2]/W};
  Real E[3];
  IdealE(r.g, sg, v, r.Bcode, E);
  const Real em[6] = {E[0], E[1], E[2], r.Bcode[0], r.Bcode[1], r.Bcode[2]};
  Real flx[NCONS];
  MHLLCFace(eosh.ps, r.g, gu, sg, r.alpha, r.beta, d, prim, prim, 0, em, em, flx);

  Real pl[NPRIM], pr[NPRIM], g[NSPMETRIC];
  Real beta[3] = {r.beta[0], r.beta[1], r.beta[2]};
  for (int n = 0; n < NPRIM; ++n) pl[n] = pr[n] = prim[n];
  for (int n = 0; n < NSPMETRIC; ++n) g[n] = r.g[n];
  Real bl[NPRIM], br[NPRIM];
  for (int a = 0; a < 3; ++a) bl[a] = br[a] = sg*r.Bcode[a];
  Real fid[NCONS], bfid[NMAG];
  if (d == 0) {
    dyngr::SingleStateLLF_DYNGR<IVX>(eosh, pl, pr, bl, br, 0, 0, g, beta, r.alpha,
                                     fid, bfid);
  } else if (d == 1) {
    dyngr::SingleStateLLF_DYNGR<IVY>(eosh, pl, pr, bl, br, 0, 0, g, beta, r.alpha,
                                     fid, bfid);
  } else {
    dyngr::SingleStateLLF_DYNGR<IVZ>(eosh, pl, pr, bl, br, 0, 0, g, beta, r.alpha,
                                     fid, bfid);
  }
  const int rows[5] = {CDN, CSX, CSY, CSZ, CTA};
  Real scale = 0.0, err = 0.0;
  for (int n = 0; n < 5; ++n) scale = fmax(scale, fabs(fid[rows[n]]));
  for (int n = 0; n < 5; ++n) err = fmax(err, fabs(flx[rows[n]] - fid[rows[n]]));
  return err/scale;
}

template <class ExecSpace>
void RunK2K3(const std::string &where, const int ntrial, const PSH &eosh) {
  Real e = MaxOverTrials<ExecSpace>("K2a", ntrial,
      KOKKOS_LAMBDA(const int t) { return K2TetradTrial(t); });
  Report("K2a triad orthonormal, round trip [" + where + "]", e, 1.0e-13);
  e = MaxOverTrials<ExecSpace>("K2b", ntrial,
      KOKKOS_LAMBDA(const int t) { return K2SRLimitTrial(t); });
  Report("K2b SR limit = flat MHLLC (bitwise) [" + where + "]", e, 0.0);
  e = MaxOverTrials<ExecSpace>("K2c", ntrial,
      KOKKOS_LAMBDA(const int t) { return K2ContactTrial(t); });
  Report("K2c HLLC exact stationary contact [" + where + "]", e, 1.0e-13);
  e = MaxOverTrials<ExecSpace>("K3c", ntrial,
      KOKKOS_LAMBDA(const int t) { return K3CellTrial(eosh, t); });
  Report("K3c ideal-limit flux = dyngr physical [" + where + "]", e, 1.0e-12);
}

template <class ExecSpace>
void RunK7(const std::string &where, const int ntrial) {
  Real e = MaxOverTrials<ExecSpace>("K7a", ntrial,
      KOKKOS_LAMBDA(const int t) { return K7MignoneTrial(t); });
  Report("K7a edge solver = Mignone Eqs. 46-47 [" + where + "]", e, 1.0e-15);
  e = MaxOverTrials<ExecSpace>("K7b", ntrial,
      KOKKOS_LAMBDA(const int t) { return K7ReductionTrial(t); });
  Report("K7b 1D reduction = HLL, GR speeds [" + where + "]", e, 1.0e-14);
  e = MaxOverTrials<ExecSpace>("K3e", ntrial,
      KOKKOS_LAMBDA(const int t) { return K3EdgeIdealTrial(t); });
  Report("K3e ideal-limit e_k = e_cc [" + where + "]", e, 1.0e-14);
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::ResistiveUnitTests()

void ProblemGenerator::ResistiveUnitTests(ParameterInput *pin, const bool restart) {
  if (restart) return;
  const int ntrial = pin->GetOrAddInteger("problem", "ntrial", 4096);
  if (global_variable::my_rank == 0) {
    std::cout << "Resistive GRMHD unit tests (K1-K7, diagnostics), ntrial = "
              << ntrial << std::endl;
  }
  RunK1<DevExeSpace>("device", ntrial);
  RunK1<Kokkos::DefaultHostExecutionSpace>("host", ntrial);
  RunK6<DevExeSpace>("device");
  RunK6<Kokkos::DefaultHostExecutionSpace>("host");
  RunDiagnostics(pmy_mesh_->pmb_pack);
  RunK4<DevExeSpace>("device", ntrial);
  RunK4<Kokkos::DefaultHostExecutionSpace>("host", ntrial);
  const int ntrial_k5 = pin->GetOrAddInteger("problem", "ntrial_k5", 16384);
  RunK5(MakeSolver(), "ideal", ntrial_k5, 1.0, -1.0);
  // K5 with the CompOSE table (e.g. inputs/tables/SFHo_reduced.athtab), if given
  if (pin->DoesParameterExist("problem", "k5_table")) {
    PST pst;
    pst.GetEOSMutable().SetCodeUnitSystem(Primitive::MakeGeometricSolar());
    pst.GetEOSMutable().ReadTableFromFile(pin->GetString("problem", "k5_table"));
    pst.GetEOSMutable().SetDensityFloor(1.0e-16);
    pst.GetEOSMutable().SetTemperatureFloor(0.01);
    pst.GetEOSMutable().SetMaxVelocity(sqrt(1.0 - 1.0/(100.0*100.0)));
    pst.tol = 1.0e-15;
    pst.GetRootSolverMutable().iterations = 60;
    const Real tconv =
        Primitive::MakeNuclear().TemperatureConversion(Primitive::MakeGeometricSolar());
    const auto &eos = pst.GetEOS();
    const Real lrhot = 0.5*(log10(eos.GetMinimumDensity())
                            + log10(eos.GetMaximumDensity()))
                       + log10(eos.GetBaryonMass());
    RunK5(pst, "tabulated", ntrial_k5, tconv, lrhot);
  }
  RunK7<DevExeSpace>("device", ntrial);
  RunK7<Kokkos::DefaultHostExecutionSpace>("host", ntrial);
  {
    PSH eosh("resistive_ut_eos", pmy_mesh_->pmb_pack, pin);
    eosh.ps.GetEOSMutable().SetGamma(kGam);
    RunK2K3<DevExeSpace>("device", ntrial, eosh);
    RunK2K3<Kokkos::DefaultHostExecutionSpace>("host", ntrial, eosh);
    RunC3(ntrial);
  }

  if (g_failed) {
    std::cout << "### FATAL ERROR: resistive unit tests FAILED" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (global_variable::my_rank == 0) {
    std::cout << "Resistive GRMHD unit tests: ALL PASSED" << std::endl;
  }
}
