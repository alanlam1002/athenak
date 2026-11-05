//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file resistive_unit_tests.cpp
//! \brief Unit tests for resistive GRMHD building blocks (07_scoping_v2 §4.1, step 0).
//!
//!  K1  prim<->cons with E != 0 for both IEN conventions on random non-diagonal metrics:
//!      (a) ideal limit E = -v x B reproduces SingleP2C_IdealGRMHD (fixed-metric
//!          T^t_t + D convention; C++ port of scripts/ien_mapping_check.py);
//!      (b) fluid part agrees with PrimitiveSolver::PrimToCon (dyngr sqrt(gamma) tau);
//!      (c) round trip prim -> cons(E,B) -> subtract EM -> PrimitiveSolver c2p -> prim,
//!          for both dyngr and fixed-metric IEN.
//!  K6  Conductivity functor values, limits, monotonicity and smoothness.
//!  D   Diagnostics on synthetic mesh fields: Q == boundary flux, q~ for linear E,
//!      div B of a CT-built field, E_c mismatch, Newton counters.
//! Each kernel test is run on the default device and on the host execution space.
//! The run exits with EXIT_FAILURE if any test fails.

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
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "eos/ideal_c2p_mhd.hpp"
#include "eos/primitive-solver/primitive_solver.hpp"
#include "eos/primitive-solver/idealgas.hpp"
#include "eos/primitive-solver/reset_floor.hpp"
#include "dyn_grmhd/resistive/resistive_params.hpp"
#include "dyn_grmhd/resistive/resistive_em.hpp"
#include "dyn_grmhd/resistive/conductivity.hpp"
#include "dyn_grmhd/resistive/resistive_diagnostics.hpp"
#include "pgen/pgen.hpp"

namespace {

using PS = Primitive::PrimitiveSolver<Primitive::IdealGas, Primitive::ResetFloor>;

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

template <class ExecSpace>
void RunK1(const std::string &where, const int ntrial) {
  PS ps = MakeSolver();
  Real e;
  e = MaxOverTrials<ExecSpace>("K1a", ntrial,
      KOKKOS_LAMBDA(const int t) { return K1IdealLimitTrial(t); });
  Report("K1a ideal limit vs SingleP2C [" + where + "]", e, 1.0e-12);
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

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::ResistiveUnitTests()

void ProblemGenerator::ResistiveUnitTests(ParameterInput *pin, const bool restart) {
  if (restart) return;
  const int ntrial = pin->GetOrAddInteger("problem", "ntrial", 4096);
  if (global_variable::my_rank == 0) {
    std::cout << "Resistive GRMHD unit tests (K1, K6, diagnostics), ntrial = " << ntrial
              << std::endl;
  }
  RunK1<DevExeSpace>("device", ntrial);
  RunK1<Kokkos::DefaultHostExecutionSpace>("host", ntrial);
  RunK6<DevExeSpace>("device");
  RunK6<Kokkos::DefaultHostExecutionSpace>("host");
  RunDiagnostics(pmy_mesh_->pmb_pack);

  if (g_failed) {
    std::cout << "### FATAL ERROR: resistive unit tests FAILED" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (global_variable::my_rank == 0) {
    std::cout << "Resistive GRMHD unit tests: ALL PASSED" << std::endl;
  }
}
