//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file kerr_torus.cpp
//! \brief K-007 W (RK-023 W1-W5); see kerr_torus.hpp. Kerr path only (<cfc>
//! kerr_background = true); nothing here runs unless <problem> test_flow = fm_torus or
//! <cfc> kerr_v4_dt > 0.
//!
//! The torus (W1/W2): Fishbone-Moncrief, ideal gas, hydro only, about the fixed hole at
//! <cfc> puncture_x1..3 (M = puncture_mass = 1, the tables' unit). Per cell:
//!  - (r_f, th_f) about the hole -> (r_BL, th_BL): bicubic Lagrange in F1 (kc24 a2 blmap,
//!    400 x 129 in (ln r_f, th_f)), only for 1.5 r_ah_max < r_f < 1000 (else atmosphere);
//!  - ell, ln h, rho, p: gr_torus.cpp (CalculateLFromRPeak :853-860, LogHAux :876-904,
//!    the in-torus test :351-355, rho and p :363-370);
//!  - u^t and Omega = u^phi/u^t: gr_torus.cpp's FM branch of CalculateVelocityInTorus
//!    (FM 3.3; :1046ff), i.e. FM's own equilibrium velocity (K-007: the RK-023 W2 item 4
//!    form -(g_tp + l g_tt)/(g_pp + l g_tp) is the constant-(-u_phi/u_t) rotation law and,
//!    with FM's l = u^t u_phi, is not FM's; measured: 10% super-Keplerian at r_max);
//!  - ut^i = u^t (Omega xi^i + beta0^i), xi = (-(y - y_bh), x - x_bh, 0), beta0 from W1
//!    (kerr_dev_note "FM initial data in slice coordinates").
//!
//! The V4 diagnostics (W5): per shell in r_f, kerr_dev_note.tex "The Killing-energy
//! diagnostic" (K1) and "The J_z diagnostic", with K = Keff (dyn_grmhd_kerr.cpp's src_ks)
//! and the residual derivatives Dx<NGHOST> of KerrBgResidual's fields, as
//! AddCoordTermsKerr (dyn_grmhd_kerr.cpp:163-168). Into <basename>.kerr_v4.txt (rank 0).

#include <math.h>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "dyn_grmhd/kerr_sources.hpp"
#include "utils/finite_diff.hpp"
#include "cfc/cfc.hpp"
#include "cfc/kerr_background.hpp"
#include "cfc/kerr_torus.hpp"

namespace cfc {
namespace {

constexpr int kNR = 400, kNT = 129;   // F1's grid (its header)

//! the torus and F1's (r_BL, th_BL) on the device; captured by value in kernels
struct FMTorus {
  Real a = 0.0, r_in = 6.0, r_max = 12.0, gam = 5.0/3.0, rho_max = 0.0;
  Real l_peak = 0.0, log_h_edge = 0.0, log_h_peak = 0.0;
  Real xbh[3] = {0.0, 0.0, 0.0};
  Real lnr0 = 0.0, dlnr = 0.0, dth = 0.0, rf_lo = 0.0, rf_hi = 1000.0;
  DvceArray1D<Real> rbl, tbl;   // [i*kNT + j]
};

struct TorusState {
  bool on = false;     // test_flow = fm_torus (F1 loaded)
  FMTorus t;
  HostArray1D<Real> rbl_h, tbl_h;
  // V4
  bool v4 = false, v4_test = false;
  Real v4_dt = 0.0, v4_next = 0.0, v4_rmin = 0.0, dfloor = 0.0, gam = 5.0/3.0;
  int v4_dump = 0, v4_dump_left = 0, v4_calls = 0;
  std::string fname;
};
TorusState *pts = nullptr;

KOKKOS_INLINE_FUNCTION Real SQ(const Real x) { return x*x; }

//! gr_torus.cpp:853-860 (prograde)
Real LFromRPeak(const Real a, const Real r) {
  const Real num = (SQ(r*r) + SQ(a*r) - 2.0*SQ(a)*r) - a*(r*r - a*a)*sqrt(r);
  const Real den = SQ(r) - 3.0*r + 2.0*a*sqrt(r);
  return 1.0/r*sqrt(1.0/r)*num/den;
}

//! gr_torus.cpp:876-904 (FM branch)
KOKKOS_INLINE_FUNCTION Real LogHAux(const Real a, const Real l, const Real r,
                                    const Real sin_th) {
  const Real s2 = SQ(sin_th), c2 = 1.0 - s2;
  const Real delta = SQ(r) - 2.0*r + SQ(a);
  const Real sigma = SQ(r) + SQ(a)*c2;
  const Real aa = SQ(SQ(r) + SQ(a)) - delta*SQ(a)*s2;
  const Real exp_2nu = sigma*delta/aa;
  const Real exp_2psi = aa/sigma*s2;
  const Real exp_neg2chi = exp_2nu/exp_2psi;
  const Real omega = 2.0*a*r/aa;
  const Real var_a = sqrt(1.0 + 4.0*SQ(l)*exp_neg2chi);
  return 0.5*log((1.0 + var_a)/(sigma*delta/aa)) - 0.5*var_a - l*omega;
}

//! BL metric (gr_torus.cpp CalculateVelocityInTorus) and FM's u^t, u^phi (FM 3.3, prograde)
KOKKOS_INLINE_FUNCTION void FMVelocity(const Real a, const Real l, const Real r,
                                       const Real sin_th, Real *pu0, Real *pom,
                                       Real *g_00, Real *g_03, Real *g_33) {
  const Real s2 = SQ(sin_th), c2 = 1.0 - s2;
  const Real delta = SQ(r) - 2.0*r + SQ(a);
  const Real sigma = SQ(r) + SQ(a)*c2;
  const Real aa = SQ(SQ(r) + SQ(a)) - delta*SQ(a)*s2;
  *g_00 = -(1.0 - 2.0*r/sigma);
  *g_03 = -2.0*a*r/sigma*s2;
  *g_33 = (sigma + (1.0 + 2.0*r/sigma)*SQ(a)*s2)*s2;
  const Real exp_2nu = sigma*delta/aa;
  const Real exp_2psi = aa/sigma*s2;
  const Real exp_neg2chi = exp_2nu/exp_2psi;
  const Real u_phi_proj = sqrt(0.5*(-1.0 + sqrt(1.0 + 4.0*SQ(l)*exp_neg2chi)));
  const Real u3 = 2.0*a*r*sqrt((1.0 + SQ(u_phi_proj))/(aa*sigma*delta))
                + sqrt(sigma/aa)/sin_th*u_phi_proj;
  const Real u0 = -1.0/(*g_00)*((*g_03)*u3
                  + sqrt((SQ(*g_03) - (*g_00)*(*g_33))*SQ(u3) - (*g_00)));
  *pu0 = u0;
  *pom = u3/u0;
}

KOKKOS_INLINE_FUNCTION void Lag4(const Real u, const int n, int *i0, Real w[4]) {
  int s = static_cast<int>(floor(u)) - 1;
  s = (s < 0) ? 0 : ((s > n - 4) ? n - 4 : s);
  *i0 = s;
  for (int p = 0; p < 4; ++p) {
    Real c = 1.0;
    for (int q = 0; q < 4; ++q) {
      if (q != p) c *= (u - (s + q))/static_cast<Real>(p - q);
    }
    w[p] = c;
  }
}

//! (r_BL, th_BL)(r_f, th_f): bicubic Lagrange in F1 (any 1D view pair)
template <class V>
KOKKOS_INLINE_FUNCTION void BLMap(const FMTorus &t, const V &rbl, const V &tbl,
                                  const Real rf, const Real thf, Real *r, Real *th) {
  int i0, j0;
  Real wr[4], wt[4];
  Lag4((log(rf) - t.lnr0)/t.dlnr, kNR, &i0, wr);
  Lag4(thf/t.dth, kNT, &j0, wt);
  Real sr = 0.0, st = 0.0;
  for (int p = 0; p < 4; ++p) {
    for (int q = 0; q < 4; ++q) {
      const int n = (i0 + p)*kNT + j0 + q;
      sr += wr[p]*wt[q]*rbl(n);
      st += wr[p]*wt[q]*tbl(n);
    }
  }
  *r = sr;
  *th = st;
}

//! FM torus state at a point (x, y, z) relative to the hole: in-torus flag, rho, p, u^t,
//! Omega, and the BL g_tt, g_tphi, g_phiphi there
struct FMPoint {
  bool in;
  Real rho, p, ut, om, g00, g03, g33, rbl, thbl;
};

KOKKOS_INLINE_FUNCTION FMPoint FMAt(const FMTorus &t, const Real dx, const Real dy,
                                    const Real dz) {
  FMPoint q;
  q.in = false; q.rho = 0.0; q.p = 0.0; q.ut = 1.0; q.om = 0.0;
  q.g00 = q.g03 = q.g33 = 0.0; q.rbl = 0.0; q.thbl = 0.0;
  const Real rf = sqrt(dx*dx + dy*dy + dz*dz);
  if (!(rf > t.rf_lo && rf < t.rf_hi)) return q;
  Real cth = dz/rf;
  cth = (cth > 1.0) ? 1.0 : ((cth < -1.0) ? -1.0 : cth);
  const Real thf = acos(cth);
  BLMap(t, t.rbl, t.tbl, rf, thf, &q.rbl, &q.thbl);
  const Real sth = sin(q.thbl);
  if (q.rbl < t.r_in || !(sth > 1.0e-12)) return q;
  const Real lnh = LogHAux(t.a, t.l_peak, q.rbl, sth) - t.log_h_edge;
  if (!(lnh >= 0.0)) return q;   // gr_torus.cpp:351-355
  const Real h = exp(lnh);
  q.rho = t.rho_max*pow((h - 1.0)/(exp(t.log_h_peak) - 1.0), 1.0/(t.gam - 1.0));
  q.p = q.rho*(t.gam - 1.0)/t.gam*(h - 1.0);   // = K rho^gam, K from the peak
  FMVelocity(t.a, t.l_peak, q.rbl, sth, &q.ut, &q.om, &q.g00, &q.g03, &q.g33);
  q.in = true;
  return q;
}

void LoadF1(const std::string &path, const Real a_expect) {
  std::ifstream f(path);
  if (!f.is_open()) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<problem> fm_blmap: cannot open '" << path << "'" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto &t = pts->t;
  Kokkos::realloc(pts->rbl_h, kNR*kNT);
  Kokkos::realloc(pts->tbl_h, kNR*kNT);
  std::string line;
  int n = 0;
  Real a_file = -1.0, rmin = -1.0;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    if (line[0] == '#') {
      auto p = line.find("(a = J = ");
      if (p != std::string::npos) a_file = std::stod(line.substr(p + 9));
      auto q = line.find("r_min = ");
      if (q != std::string::npos) rmin = std::stod(line.substr(q + 8));
      continue;
    }
    std::istringstream ss(line);
    int i, j;
    std::string s[4];
    ss >> i >> j >> s[0] >> s[1] >> s[2] >> s[3];
    if (ss.fail() || i < 0 || i >= kNR || j < 0 || j >= kNT || i*kNT + j != n) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "F1 '" << path << "': bad row " << n << std::endl;
      std::exit(EXIT_FAILURE);
    }
    pts->rbl_h(n) = std::stod(s[2]);
    pts->tbl_h(n) = std::stod(s[3]);
    ++n;
  }
  if (n != kNR*kNT || !(fabs(a_file - a_expect) < 1.0e-12) || !(rmin > 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "F1 '" << path << "': " << n << " rows (need " << kNR*kNT << "), a = "
              << a_file << " (<problem> fm_spin = " << a_expect << "), r_min = " << rmin
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  t.lnr0 = log(rmin);
  t.dlnr = (log(1000.0) - t.lnr0)/(kNR - 1);
  t.dth = M_PI/(kNT - 1);
  Kokkos::realloc(t.rbl, kNR*kNT);
  Kokkos::realloc(t.tbl, kNR*kNT);
  Kokkos::deep_copy(t.rbl, pts->rbl_h);
  Kokkos::deep_copy(t.tbl, pts->tbl_h);
}

//! [M] F1's interpolation error proxy: bicubic against biquintic Lagrange at cell-centre
//! midpoints of the table for r_f in [lo, hi] (max relative, r_BL and th_BL)
void F1InterpCheck(const Real lo, const Real hi, Real *er, Real *et) {
  const auto &t = pts->t;
  auto &rb = pts->rbl_h;
  auto &tb = pts->tbl_h;
  auto lag6 = [](Real u, int n, int *i0, Real w[6]) {
    int s = static_cast<int>(floor(u)) - 2;
    s = (s < 0) ? 0 : ((s > n - 6) ? n - 6 : s);
    *i0 = s;
    for (int p = 0; p < 6; ++p) {
      Real c = 1.0;
      for (int q = 0; q < 6; ++q) if (q != p) c *= (u - (s + q))/static_cast<Real>(p - q);
      w[p] = c;
    }
  };
  *er = 0.0; *et = 0.0;
  for (int i = 2; i < kNR - 3; ++i) {
    const Real u = i + 0.5, rf = exp(t.lnr0 + u*t.dlnr);
    if (rf < lo || rf > hi) continue;
    for (int j = 3; j < kNT - 4; ++j) {
      const Real v = j + 0.5;
      Real r3, t3;
      BLMap(t, rb, tb, rf, v*t.dth, &r3, &t3);
      int i0, j0;
      Real wr[6], wt[6];
      lag6(u, kNR, &i0, wr);
      lag6(v, kNT, &j0, wt);
      Real r5 = 0.0, t5 = 0.0;
      for (int p = 0; p < 6; ++p) {
        for (int q = 0; q < 6; ++q) {
          r5 += wr[p]*wt[q]*rb((i0 + p)*kNT + j0 + q);
          t5 += wr[p]*wt[q]*tb((i0 + p)*kNT + j0 + q);
        }
      }
      *er = std::max(*er, fabs(r3 - r5)/r5);
      *et = std::max(*et, fabs(t3 - t5)/M_PI);
    }
  }
}

//----------------------------------------------------------------------------------------
//! V4: the per-cell record (kerr_torus.hpp V4In/V4Out) -> outputs. Host and device.
//! v[kV4Vol] .. are written; everything per unit sqrt(gamma) except kV4Vol.
KOKKOS_INLINE_FUNCTION void V4Cell(Real v[kV4N], const Real gam) {
  using kerrsrc::Sym;
  const Real *g3d = &v[kV4Gdd];
  const Real alpha = v[kV4Alp], alpha0 = v[kV4Alp0], psi0 = v[kV4Psi0];
  const Real detg = adm::SpatialDet(g3d[0], g3d[1], g3d[2], g3d[3], g3d[4], g3d[5]);
  const Real vol = sqrt(detg);
  Real g3u[6];
  adm::SpatialInv(1.0/detg, g3d[0], g3d[1], g3d[2], g3d[3], g3d[4], g3d[5],
                  &g3u[0], &g3u[1], &g3u[2], &g3u[3], &g3u[4], &g3u[5]);
  const Real p4 = cbrt(detg), p40 = SQ(SQ(psi0));
  const Real *gt = &v[kV4Gt], *dgt = &v[kV4Dgt], *da0 = &v[kV4Da0], *db0 = &v[kV4Db0];
  const Real *dl0 = &v[kV4Dl0], *k0 = &v[kV4K0], *br = &v[kV4Br], *dda = &v[kV4Dda];
  const Real *dell = &v[kV4Dell], *b0 = &v[kV4B0], *kx = &v[kV4Kx];
  Real dbr[3][3];
  for (int a = 0; a < 3; ++a) for (int c = 0; c < 3; ++c) dbr[a][c] = v[kV4Dbr + 3*a + c];
  const Real dal = v[kV4Dal];

  // the fluid (ideal gas, hydro only; AddCoordTermsEOS's E, S_i, S^ij with B = 0)
  const Real rho = v[kV4Rho], pre = v[kV4Pre];
  const Real u[3] = {v[kV4Ux], v[kV4Uy], v[kV4Uz]};
  const Real H = rho + gam/(gam - 1.0)*pre;
  Real usq = 0.0;
  for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) usq += g3d[Sym(a, b)]*u[a]*u[b];
  const Real W = sqrt(1.0 + usq);
  const Real E = H*W*W - pre;
  Real S_d[3], S_u[3], S_uu[3][3];
  for (int a = 0; a < 3; ++a) {
    S_d[a] = 0.0;
    for (int b = 0; b < 3; ++b) S_d[a] += H*W*u[b]*g3d[Sym(a, b)];
  }
  for (int a = 0; a < 3; ++a) {
    S_u[a] = 0.0;
    for (int b = 0; b < 3; ++b) {
      S_u[a] += g3u[Sym(a, b)]*S_d[b];
      S_uu[a][b] = H*u[a]*u[b] + pre*g3u[Sym(a, b)];
    }
  }

  // sigma_eff = alpha S^ij Keff_ij (dyn_grmhd_kerr.cpp:220), sigma_X = alpha S^ij K_X_ij
  const Real sig = kerrsrc::EnergyK(p4, p4/p40, alpha0, gt, dgt, br, dbr, k0, S_uu);
  Real sk0 = 0.0, skx = 0.0, trS = 0.0;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      sk0 += S_uu[a][b]*k0[Sym(a, b)];
      skx += S_uu[a][b]*kx[Sym(a, b)];
      trS += g3d[Sym(a, b)]*S_uu[a][b];
    }
  }
  // (L_beta0 h)_ij, h = (psi^4 - psi0^4) gt
  Real bdh = 0.0;
  for (int k = 0; k < 3; ++k) bdh += b0[k]*(4.0*p4*(dl0[k] + dell[k]) - 4.0*p40*dl0[k]);
  const Real dp4 = p4 - p40;
  Real slh = 0.0;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      Real lgt = 0.0;
      for (int k = 0; k < 3; ++k) {
        lgt += b0[k]*dgt[6*k + Sym(a, b)] + gt[Sym(k, b)]*db0[3*a + k]
             + gt[Sym(a, k)]*db0[3*b + k];
      }
      slh += S_uu[a][b]*(bdh*gt[Sym(a, b)] + dp4*lgt);
    }
  }
  // K1: S_K, S_alpha, S_E, S_beta
  const Real s_k = alpha0*sig - alpha*alpha0*sk0 - 0.5*alpha*slh;
  Real s_a = 0.0, s_e = 0.0, s_b = 0.0;
  for (int j = 0; j < 3; ++j) {
    s_a += -alpha0*S_u[j]*dda[j] + dal*S_u[j]*da0[j];
    s_e += E*(b0[j]*dda[j] - br[j]*da0[j]);
    Real comm = 0.0;   // [beta0, db]^j
    for (int k = 0; k < 3; ++k) comm += b0[k]*dbr[k][j] - br[k]*db0[3*k + j];
    s_b -= S_d[j]*comm;
  }
  // J_z: xi = (xi^x, xi^y, 0)
  const Real xi[3] = {v[kV4Xi], v[kV4Xi + 1], 0.0};
  Real xdda = 0.0, xdell = 0.0, j2 = 0.0, jz = 0.0, tt = alpha0*E;
  for (int a = 0; a < 3; ++a) {
    xdda += xi[a]*dda[a];
    xdell += xi[a]*dell[a];
    jz += S_d[a]*xi[a];
    tt -= b0[a]*S_d[a];
  }
  const Real rot[3] = {br[1], -br[0], 0.0};   // -(b . d) xi
  for (int k = 0; k < 3; ++k) {
    Real xdb = 0.0;
    for (int a = 0; a < 3; ++a) xdb += xi[a]*dbr[a][k];
    j2 += S_d[k]*(xdb + rot[k]);
  }
  v[kV4Vol] = vol;
  v[kV4E] = E;
  v[kV4SK] = s_k;
  v[kV4SA] = s_a;
  v[kV4SE] = s_e;
  v[kV4SB] = s_b;
  v[kV4Sig] = sig;
  v[kV4SigX] = alpha*skx;
  v[kV4T] = tt;
  v[kV4Jz] = jz;
  v[kV4J1] = -E*xdda;
  v[kV4J2] = j2;
  v[kV4J3] = 2.0*alpha*trS*xdell;
}

//! zero-test inputs (RK-023 W5 unit check): delta alpha = b_res = ln(psi/psi0) = 0 with
//! their derivatives, alpha = alpha0, gamma = psi0^4 gt, K_X = K0 (Keff = K0)
KOKKOS_INLINE_FUNCTION void V4ZeroResiduals(Real v[kV4N]) {
  const Real p40 = SQ(SQ(v[kV4Psi0]));
  for (int q = 0; q < 6; ++q) {
    v[kV4Gdd + q] = p40*v[kV4Gt + q];
    v[kV4Kx + q] = v[kV4K0 + q];
  }
  v[kV4Alp] = v[kV4Alp0];
  for (int q = 0; q < 3; ++q) { v[kV4Br + q] = 0.0; v[kV4Dda + q] = 0.0; v[kV4Dell + q] = 0.0; }
  for (int q = 0; q < 9; ++q) v[kV4Dbr + q] = 0.0;
  v[kV4Ell] = 0.0;
  v[kV4Dal] = 0.0;
}

// V4 accumulators: kNB bins x kNQ per-bin sums, then globals
constexpr int kNShell = 16, kNB = kNShell + 2;
enum V4Q { qMD = 0, qT, qJz, qSig, qSigX, qSK, qSA, qSE, qSB, qADX, qJ1, qJ2, qJ3,
           qAbsSum, qSumAbs, qVol, kNQ };
enum V4G { gL1n = 0, gL1d, gDr, gD, gAtm, gNFloor, gNFloorT, gNTorus, gZtK, gZtX, gZtJ,
           gZtScale, gZtScaleX, kNG };
constexpr int kNAcc = kNB*kNQ + kNG;

template <int NG>
void V4Accumulate(MeshBlockPack *pmbp, const TorusState &st, DvceArray1D<Real> acc,
                  Real *rhomax) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;
  auto &w0 = pmbp->pmhd->w0;
  auto &u0 = pmbp->pmhd->u0;
  auto &adm = pmbp->padm->adm;
  pmbp->pdyngr->KerrBgResidual();
  auto bg = pmbp->pcfc->KerrBgForSources();
  auto res = pmbp->pdyngr->kill_res;
  auto ell = pmbp->pdyngr->kbg_ell;
  const FMTorus t = st.t;
  const bool ton = st.on, ztest = st.v4_test;
  const Real gam = st.gam, dfl = st.dfloor*(1.0 + 1.0e-12);
  const Real xb0 = pmbp->pcfc->bg_xbh_[0], xb1 = pmbp->pcfc->bg_xbh_[1];
  const Real xb2 = pmbp->pcfc->bg_xbh_[2];
  const Real lrmin = log(st.v4_rmin), dlr = (log(64.0) - lrmin)/kNShell;
  Real rmx = 0.0;
  Kokkos::parallel_reduce("kerr_v4", Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>(
      {0, ks, js, is}, {nmb, ke + 1, je + 1, ie + 1}),
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, Real &lmax) {
    const Real x = CellCenterX(i - is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    const Real y = CellCenterX(j - js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    const Real z = CellCenterX(k - ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    const Real dV = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    const Real dx = x - xb0, dy = y - xb1, dz = z - xb2;
    const Real rf = sqrt(dx*dx + dy*dy + dz*dz);
    Real v[kV4N];
    v[kV4Rho] = w0(m, IDN, k, j, i);
    v[kV4Pre] = w0(m, IPR, k, j, i);
    v[kV4Ux] = w0(m, IVX, k, j, i);
    v[kV4Uy] = w0(m, IVY, k, j, i);
    v[kV4Uz] = w0(m, IVZ, k, j, i);
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        v[kV4Gdd + kerrsrc::Sym(a, b)] = adm.g_dd(m, a, b, k, j, i);
        v[kV4Kx + kerrsrc::Sym(a, b)] = adm.vK_dd(m, a, b, k, j, i);
      }
    }
    v[kV4Alp] = adm.alpha(m, k, j, i);
    v[kV4Psi0] = bg(m, kKbgPsi0, k, j, i);
    v[kV4Alp0] = bg(m, kKbgAlpha0, k, j, i);
    for (int q = 0; q < 6; ++q) {
      v[kV4Gt + q] = bg(m, kKbgGt + q, k, j, i);
      v[kV4K0 + q] = bg(m, kKbgK0 + q, k, j, i);
    }
    for (int q = 0; q < 18; ++q) v[kV4Dgt + q] = bg(m, kKbgDGt + q, k, j, i);
    for (int q = 0; q < 9; ++q) v[kV4Db0 + q] = bg(m, kKbgDBeta0 + q, k, j, i);
    const Real idx[] = {1./size.d_view(m).dx1, 1./size.d_view(m).dx2,
                        1./size.d_view(m).dx3};
    for (int a = 0; a < 3; ++a) {
      v[kV4B0 + a] = bg(m, kKbgBeta0 + a, k, j, i);
      v[kV4Da0 + a] = bg(m, kKbgDAlpha0 + a, k, j, i);
      v[kV4Dl0 + a] = bg(m, kKbgDLnPsi0 + a, k, j, i);
      v[kV4Br + a] = res(m, 1 + a, k, j, i);
      v[kV4Dda + a] = Dx<NG>(a, idx, res, m, 0, k, j, i);
      v[kV4Dell + a] = Dx<NG>(a, idx, ell, m, k, j, i);
      for (int c = 0; c < 3; ++c) v[kV4Dbr + 3*a + c] = Dx<NG>(a, idx, res, m, 1 + c, k, j, i);
    }
    v[kV4Xi] = -dy;
    v[kV4Xi + 1] = dx;
    v[kV4Gam] = gam;
    v[kV4Ell] = ell(m, k, j, i);
    v[kV4Dal] = res(m, 0, k, j, i);
    V4Cell(v, gam);
    const Real vol = v[kV4Vol];
    int bin;
    if (!(rf >= exp(lrmin))) {
      bin = 0;
    } else if (rf >= 64.0) {
      bin = kNB - 1;
    } else {
      bin = 1 + static_cast<int>((log(rf) - lrmin)/dlr);
      bin = (bin < 1) ? 1 : ((bin > kNShell) ? kNShell : bin);
    }
    Real *b = &acc(bin*kNQ);
    const Real D = u0(m, IDN, k, j, i);   // densitized rest mass (sqrt(gamma) rho W)
    const Real sum4 = v[kV4SK] + v[kV4SA] + v[kV4SE] + v[kV4SB];
    const Real vd = vol*dV;
    Kokkos::atomic_add(&b[qMD], D*dV);
    Kokkos::atomic_add(&b[qT], vd*v[kV4T]);
    Kokkos::atomic_add(&b[qJz], vd*v[kV4Jz]);
    Kokkos::atomic_add(&b[qSig], vd*v[kV4Sig]);
    Kokkos::atomic_add(&b[qSigX], vd*v[kV4SigX]);
    Kokkos::atomic_add(&b[qSK], vd*v[kV4SK]);
    Kokkos::atomic_add(&b[qSA], vd*v[kV4SA]);
    Kokkos::atomic_add(&b[qSE], vd*v[kV4SE]);
    Kokkos::atomic_add(&b[qSB], vd*v[kV4SB]);
    Kokkos::atomic_add(&b[qADX], vd*v[kV4Alp0]*(v[kV4SigX] - v[kV4Sig]));
    Kokkos::atomic_add(&b[qJ1], vd*v[kV4J1]);
    Kokkos::atomic_add(&b[qJ2], vd*v[kV4J2]);
    Kokkos::atomic_add(&b[qJ3], vd*v[kV4J3]);
    Kokkos::atomic_add(&b[qAbsSum], vd*fabs(sum4));
    Kokkos::atomic_add(&b[qSumAbs], vd*(fabs(v[kV4SK]) + fabs(v[kV4SA]) + fabs(v[kV4SE])
                                         + fabs(v[kV4SB])));
    Kokkos::atomic_add(&b[qVol], dV);
    Real *g = &acc(kNB*kNQ);
    const Real rho = v[kV4Rho];
    Real rho0 = 0.0;
    if (ton) {
      const FMPoint q = FMAt(t, dx, dy, dz);
      rho0 = q.in ? q.rho : 0.0;
    }
    if (rho0 > 0.0) {
      Kokkos::atomic_add(&g[gL1n], fabs(rho - rho0)*dV);
      Kokkos::atomic_add(&g[gL1d], rho0*dV);
      Kokkos::atomic_add(&g[gNTorus], 1.0);
      if (rho <= dfl) Kokkos::atomic_add(&g[gNFloorT], 1.0);
    } else {
      Kokkos::atomic_add(&g[gAtm], D*dV);
    }
    Kokkos::atomic_add(&g[gDr], D*dV*rf);
    Kokkos::atomic_add(&g[gD], D*dV);
    if (rho <= dfl) Kokkos::atomic_add(&g[gNFloor], 1.0);
    if (ztest) {   // the unit check: every K1 / J_z source of the zeroed record
      V4ZeroResiduals(v);
      V4Cell(v, gam);
      const Real sc = fabs(v[kV4Alp0]*v[kV4Sig]);
      Kokkos::atomic_add(&g[gZtK], vd*(fabs(v[kV4SK]) + fabs(v[kV4SA]) + fabs(v[kV4SE])
                                         + fabs(v[kV4SB])));
      Kokkos::atomic_add(&g[gZtX], vd*fabs(v[kV4Alp0]*(v[kV4SigX] - v[kV4Sig])));
      Kokkos::atomic_add(&g[gZtJ], vd*(fabs(v[kV4J1]) + fabs(v[kV4J2]) + fabs(v[kV4J3])));
      Kokkos::atomic_add(&g[gZtScale], vd*sc);
      Kokkos::atomic_add(&g[gZtScaleX], vd*fabs(v[kV4Alp0]*v[kV4SigX]));
    }
    lmax = (rho > lmax) ? rho : lmax;
  }, Kokkos::Max<Real>(rmx));
  *rhomax = rmx;
}

//! W5 host check: dump up to n records (inputs + device outputs) of torus cells,
//! (i + j + k) % 7 == 0, rho > 0.1 rho_max, into <basename>.kerr_v4_dump.<rank>.txt
template <int NG>
void V4Dump(MeshBlockPack *pmbp, const TorusState &st, const int nmax) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;
  auto &w0 = pmbp->pmhd->w0;
  auto &adm = pmbp->padm->adm;
  auto bg = pmbp->pcfc->KerrBgForSources();
  auto res = pmbp->pdyngr->kill_res;
  auto ell = pmbp->pdyngr->kbg_ell;
  const Real gam = st.gam, rthr = 0.1*st.t.rho_max;
  const Real xb0 = pmbp->pcfc->bg_xbh_[0], xb1 = pmbp->pcfc->bg_xbh_[1];
  const Real xb2 = pmbp->pcfc->bg_xbh_[2];
  DvceArray2D<Real> rec("kerr_v4_dump", nmax, kV4N + 3);
  DvceArray1D<int> cnt("kerr_v4_cnt", 1);
  par_for("kerr_v4_dump", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    if ((i + j + k) % 7 != 0 || !(w0(m, IDN, k, j, i) > rthr)) return;
    const int s = Kokkos::atomic_fetch_add(&cnt(0), 1);
    if (s >= nmax) return;
    Real v[kV4N];
    v[kV4Rho] = w0(m, IDN, k, j, i);
    v[kV4Pre] = w0(m, IPR, k, j, i);
    v[kV4Ux] = w0(m, IVX, k, j, i);
    v[kV4Uy] = w0(m, IVY, k, j, i);
    v[kV4Uz] = w0(m, IVZ, k, j, i);
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        v[kV4Gdd + kerrsrc::Sym(a, b)] = adm.g_dd(m, a, b, k, j, i);
        v[kV4Kx + kerrsrc::Sym(a, b)] = adm.vK_dd(m, a, b, k, j, i);
      }
    }
    v[kV4Alp] = adm.alpha(m, k, j, i);
    v[kV4Psi0] = bg(m, kKbgPsi0, k, j, i);
    v[kV4Alp0] = bg(m, kKbgAlpha0, k, j, i);
    for (int q = 0; q < 6; ++q) {
      v[kV4Gt + q] = bg(m, kKbgGt + q, k, j, i);
      v[kV4K0 + q] = bg(m, kKbgK0 + q, k, j, i);
    }
    for (int q = 0; q < 18; ++q) v[kV4Dgt + q] = bg(m, kKbgDGt + q, k, j, i);
    for (int q = 0; q < 9; ++q) v[kV4Db0 + q] = bg(m, kKbgDBeta0 + q, k, j, i);
    const Real idx[] = {1./size.d_view(m).dx1, 1./size.d_view(m).dx2,
                        1./size.d_view(m).dx3};
    for (int a = 0; a < 3; ++a) {
      v[kV4B0 + a] = bg(m, kKbgBeta0 + a, k, j, i);
      v[kV4Da0 + a] = bg(m, kKbgDAlpha0 + a, k, j, i);
      v[kV4Dl0 + a] = bg(m, kKbgDLnPsi0 + a, k, j, i);
      v[kV4Br + a] = res(m, 1 + a, k, j, i);
      v[kV4Dda + a] = Dx<NG>(a, idx, res, m, 0, k, j, i);
      v[kV4Dell + a] = Dx<NG>(a, idx, ell, m, k, j, i);
      for (int c = 0; c < 3; ++c) v[kV4Dbr + 3*a + c] = Dx<NG>(a, idx, res, m, 1 + c, k, j, i);
    }
    const Real x = CellCenterX(i - is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    const Real y = CellCenterX(j - js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    const Real z = CellCenterX(k - ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    v[kV4Xi] = -(y - xb1);
    v[kV4Xi + 1] = x - xb0;
    v[kV4Gam] = gam;
    v[kV4Ell] = ell(m, k, j, i);
    v[kV4Dal] = res(m, 0, k, j, i);
    V4Cell(v, gam);
    for (int q = 0; q < kV4N; ++q) rec(s, q) = v[q];
    rec(s, kV4N) = x - xb0;
    rec(s, kV4N + 1) = y - xb1;
    rec(s, kV4N + 2) = z - xb2;
  });
  auto rec_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), rec);
  auto cnt_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), cnt);
  const int n = std::min(cnt_h(0), nmax);
  if (n <= 0) return;
  std::string fn = st.fname;
  fn.replace(fn.rfind(".kerr_v4.txt"), 12, ".kerr_v4_dump." + std::to_string(global_variable::my_rank)
             + ".txt");
  FILE *f = std::fopen(fn.c_str(), "a");
  if (f == nullptr) return;
  std::fprintf(f, "# t = %.17e ncycle = %d; %d records of %d inputs, %d outputs "
               "(kerr_torus.hpp V4In/V4Out), then x-xbh y-ybh z-zbh\n",
               pmbp->pmesh->time, pmbp->pmesh->ncycle, n, static_cast<int>(kV4NIn),
               static_cast<int>(kV4N - kV4NIn));
  for (int s = 0; s < n; ++s) {
    for (int q = 0; q < kV4N + 3; ++q) std::fprintf(f, "%.17e ", rec_h(s, q));
    std::fprintf(f, "\n");
  }
  std::fclose(f);
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void KerrFMTorusInit
//! \brief <problem> test_flow = fm_torus (dyngr_tov.cpp). Loads the torus parameters and
//! F1; unless restart, overwrites w0 inside the torus (over star_enabled = false's floor
//! atmosphere) and prints the self-checks. Restart: parameters and F1 only (rho0 for V4).

void KerrFMTorusInit(ParameterInput *pin, MeshBlockPack *pmbp, bool restart) {
  if (pmbp->pcfc == nullptr || !pmbp->pcfc->KerrOn()) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<problem> test_flow = fm_torus needs <cfc> kerr_background = true"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pin->GetOrAddBoolean("problem", "star_enabled", true)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<problem> test_flow = fm_torus needs <problem> star_enabled = false"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pts == nullptr) pts = new TorusState;
  auto &t = pts->t;
  t.a = pin->GetReal("problem", "fm_spin");
  t.r_in = pin->GetOrAddReal("problem", "fm_r_in", 6.0);
  t.r_max = pin->GetOrAddReal("problem", "fm_r_max", 12.0);
  t.rho_max = pin->GetReal("problem", "fm_rho_max");
  t.gam = pin->GetReal("mhd", "gamma");
  const Real mbh = pin->GetOrAddReal("cfc", "puncture_mass", 1.0);
  if (fabs(mbh - 1.0) > 1.0e-14 || pin->GetOrAddBoolean("cfc", "puncture_moving", false)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "fm_torus: needs <cfc> puncture_mass = 1 (F1's unit) and a fixed hole"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  for (int a = 0; a < 3; ++a) {
    t.xbh[a] = pin->GetOrAddReal("cfc", "puncture_x" + std::to_string(a + 1), 0.0);
    if (fabs(t.xbh[a] - pmbp->pcfc->bg_xbh_[a]) > 1.0e-12) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "fm_torus: puncture_x" << a + 1 << " != the background's hole"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  t.l_peak = LFromRPeak(t.a, t.r_max);
  t.log_h_edge = LogHAux(t.a, t.l_peak, t.r_in, 1.0);
  t.log_h_peak = LogHAux(t.a, t.l_peak, t.r_max, 1.0) - t.log_h_edge;
  t.rf_lo = 1.5*pin->GetReal("cfc", "kerr_r_ah_max");
  t.rf_hi = 1000.0;
  LoadF1(pin->GetString("problem", "fm_blmap"), t.a);
  pts->on = true;
  if (restart) return;

  // [M] F1 interpolation proxy over the torus's radii
  Real er, et;
  F1InterpCheck(4.0, 60.0, &er, &et);

  auto &indcs = pmbp->pmesh->mb_indcs;
  const int ng = indcs.ng;
  const int n1 = indcs.nx1 + 2*ng, n2 = indcs.nx2 + 2*ng, n3 = indcs.nx3 + 2*ng;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;
  auto &w0 = pmbp->pmhd->w0;
  auto bg = pmbp->pcfc->KerrBgForSources();
  const FMTorus tt = t;
  const Real dfloor = pin->GetReal("mhd", "dfloor");
  // the self-checks (max over torus cells), RK-023 W2:
  //  0: |W(v) / (alpha0 u^t) - 1|, v = (Omega xi + beta0)/alpha0, gamma0 = psi0^4 gt
  //  1: |u_t(code) / u_t(BL) - 1|; 2: |u_phi(code) / u_phi(BL) - 1|; 3: max rho/rho_max;
  //  4: max |r_BL - r_f - 1| / r_f (F1's large-r map, a sanity bound only)
  Real ck[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  Kokkos::parallel_reduce("pgen_fm_torus", Kokkos::MDRangePolicy<DevExeSpace,
      Kokkos::Rank<4>>({0, 0, 0, 0}, {nmb, n3, n2, n1}),
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, Real &c0, Real &c1,
                Real &c2, Real &c3, Real &c4) {
    const Real x = CellCenterX(i - is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    const Real y = CellCenterX(j - js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    const Real z = CellCenterX(k - ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    const Real dx = x - tt.xbh[0], dy = y - tt.xbh[1], dz = z - tt.xbh[2];
    const FMPoint q = FMAt(tt, dx, dy, dz);
    if (!q.in || !(q.rho > dfloor)) return;
    const Real xi[3] = {-dy, dx, 0.0};
    const Real psi0 = bg(m, kKbgPsi0, k, j, i), al0 = bg(m, kKbgAlpha0, k, j, i);
    const Real p40 = SQ(SQ(psi0));
    Real b0[3], g0[6];
    for (int a = 0; a < 3; ++a) b0[a] = bg(m, kKbgBeta0 + a, k, j, i);
    for (int a = 0; a < 6; ++a) g0[a] = p40*bg(m, kKbgGt + a, k, j, i);
    Real vv = 0.0, bb = 0.0, bx = 0.0, xx = 0.0;
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        const Real g = g0[kerrsrc::Sym(a, b)];
        const Real va = (q.om*xi[a] + b0[a])/al0, vb = (q.om*xi[b] + b0[b])/al0;
        vv += g*va*vb;
        bb += g*b0[a]*b0[b];
        bx += g*b0[a]*xi[b];
        xx += g*xi[a]*xi[b];
      }
    }
    const Real wa = al0*q.ut, wb = 1.0/sqrt(1.0 - vv);
    const Real ut_code = q.ut*(-al0*al0 + bb + q.om*bx), ut_bl = q.ut*(q.g00 + q.om*q.g03);
    const Real up_code = q.ut*(bx + q.om*xx), up_bl = q.ut*(q.g03 + q.om*q.g33);
    const Real rf = sqrt(dx*dx + dy*dy + dz*dz);
    c0 = fmax(c0, fabs(wb/wa - 1.0));
    c1 = fmax(c1, fabs(ut_code/ut_bl - 1.0));
    c2 = fmax(c2, fabs(up_code/up_bl - 1.0));
    c3 = fmax(c3, q.rho/tt.rho_max);
    c4 = fmax(c4, fabs(q.rbl - rf - 1.0)/rf);
    w0(m, IDN, k, j, i) = q.rho;
    w0(m, IPR, k, j, i) = q.p;
    w0(m, IVX, k, j, i) = q.ut*(q.om*xi[0] + b0[0]);
    w0(m, IVY, k, j, i) = q.ut*(q.om*xi[1] + b0[1]);
    w0(m, IVZ, k, j, i) = q.ut*b0[2];
  }, Kokkos::Max<Real>(ck[0]), Kokkos::Max<Real>(ck[1]), Kokkos::Max<Real>(ck[2]),
     Kokkos::Max<Real>(ck[3]), Kokkos::Max<Real>(ck[4]));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, ck, 5, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  if (global_variable::my_rank == 0) {
    std::printf("KERR FM torus: a=%.6f r_in=%.4f r_max=%.4f gamma=%.10f rho_max=%.6e "
                "l=%.10f ln_h_edge=%.10e ln_h_peak=%.10e (p/rho)_peak=%.10e "
                "hole=(%.6f,%.6f,%.6f) r_f window (%.6f, %.1f)\n",
                t.a, t.r_in, t.r_max, t.gam, t.rho_max, t.l_peak, t.log_h_edge,
                t.log_h_peak, (t.gam - 1.0)/t.gam*(exp(t.log_h_peak) - 1.0),
                t.xbh[0], t.xbh[1], t.xbh[2], t.rf_lo, t.rf_hi);
    std::printf("KERR FM torus F1 interp (bicubic vs biquintic, r_f in [4, 60]): "
                "max rel r_BL %.3e, max th_BL/pi %.3e\n", er, et);
    std::printf("KERR FM torus self-checks (max over torus cells incl. ghosts): "
                "|W(v)/(alpha0 u^t)-1| %.3e  |u_t code/BL-1| %.3e  |u_phi code/BL-1| %.3e  "
                "max rho/rho_max %.12f  max|r_BL-r_f-1|/r_f %.3e\n",
                ck[0], ck[1], ck[2], ck[3], ck[4]);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void KerrV4Setup
//! \brief <cfc> kerr_v4_dt (default 0 = off), kerr_v4_test (the zero-residual unit check,
//! default false), kerr_v4_dump (records per rank per dump, default 0), kerr_v4_ndump
//! (dumps, default 1), kerr_v4_rmin (inner shell edge, default kerr_r_ah_max)

void KerrV4Setup(ParameterInput *pin, MeshBlockPack *pmbp) {
  const Real dt = pin->GetOrAddReal("cfc", "kerr_v4_dt", 0.0);
  if (!(dt > 0.0)) return;
  if (pmbp->pcfc == nullptr || !pmbp->pcfc->KerrOn() || pmbp->pdyngr == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<cfc> kerr_v4_dt > 0 needs <cfc> kerr_background = true" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pts == nullptr) pts = new TorusState;
  pts->v4 = true;
  pts->v4_dt = dt;
  pts->v4_next = pmbp->pmesh->time;
  pts->v4_test = pin->GetOrAddBoolean("cfc", "kerr_v4_test", false);
  pts->v4_dump = pin->GetOrAddInteger("cfc", "kerr_v4_dump", 0);
  pts->v4_dump_left = (pts->v4_dump > 0) ? pin->GetOrAddInteger("cfc", "kerr_v4_ndump", 1)
                                         : 0;
  pts->v4_rmin = pin->GetOrAddReal("cfc", "kerr_v4_rmin",
                                   pin->GetReal("cfc", "kerr_r_ah_max"));
  pts->dfloor = pin->GetReal("mhd", "dfloor");
  pts->gam = pin->GetReal("mhd", "gamma");
  pts->fname = pin->GetString("job", "basename") + ".kerr_v4.txt";
  if (global_variable::my_rank == 0) {
    FILE *f = std::fopen(pts->fname.c_str(), "a");
    if (f != nullptr) {
      std::fprintf(f, "# kerr_v4 (K-007 W5; kerr_torus.cpp) start t = %.17e: dt %.6g, "
                   "zero-test %d, %d shells log-spaced in r_f from %.6f to 64 about "
                   "(%.6f, %.6f, %.6f) + bin 0 (r_f < r_min) + bin %d (r_f >= 64)\n"
                   "# S rows: t bin r_lo r_hi M_D T J_z sigma_eff sigma_X S_K S_alpha S_E "
                   "S_beta alpha0_DX J1 J2 J3 sum|S_K+S_a+S_E+S_b| sum(|S_K|+|S_a|+|S_E|"
                   "+|S_b|) V_coord  (volume integrals sqrt(gamma)(.) dV; M_D = sum D dV)\n"
                   "# G rows: t ncycle rhomax rhomax/fm_rho_max L1 <r_f>_D M_D M_atm "
                   "n_floor n_floor_torus n_torus accreted_mass accreted_mom_x y z "
                   "[zero-test: sum|S|_0/sum|a0 sig|, sum|a0 DX|_0/sum|a0 sigX|, "
                   "sum|J|_0/sum|a0 sig|]\n",
                   pmbp->pmesh->time, dt, pts->v4_test ? 1 : 0, kNShell, pts->v4_rmin,
                   pmbp->pcfc->bg_xbh_[0], pmbp->pcfc->bg_xbh_[1], pmbp->pcfc->bg_xbh_[2],
                   kNB - 1);
      std::fclose(f);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void KerrV4Diag
//! \brief from TOVHistory (every history output, all ranks); fires when time >= the next
//! kerr_v4_dt mark. Inert unless KerrV4Setup enabled it.

void KerrV4Diag(Mesh *pm) {
  if (pts == nullptr || !pts->v4) return;
  if (pm->time < pts->v4_next - 1.0e-12*pts->v4_dt) return;
  while (pts->v4_next <= pm->time + 1.0e-12*pts->v4_dt) pts->v4_next += pts->v4_dt;
  MeshBlockPack *pmbp = pm->pmb_pack;
  DvceArray1D<Real> acc("kerr_v4_acc", kNAcc);
  Real rhomax = 0.0;
  const int ng = pm->mb_indcs.ng;
  switch (ng) {
    case 2: V4Accumulate<2>(pmbp, *pts, acc, &rhomax); break;
    case 3: V4Accumulate<3>(pmbp, *pts, acc, &rhomax); break;
    case 4: V4Accumulate<4>(pmbp, *pts, acc, &rhomax); break;
    default:
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "kerr_v4: nghost " << ng << std::endl;
      std::exit(EXIT_FAILURE);
  }
  if (pts->v4_dump_left > 0) {
    switch (ng) {
      case 2: V4Dump<2>(pmbp, *pts, pts->v4_dump); break;
      case 3: V4Dump<3>(pmbp, *pts, pts->v4_dump); break;
      case 4: V4Dump<4>(pmbp, *pts, pts->v4_dump); break;
    }
    --pts->v4_dump_left;
  }
  auto acc_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), acc);
  std::vector<Real> s(kNAcc);
  for (int q = 0; q < kNAcc; ++q) s[q] = acc_h(q);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, s.data(), kNAcc, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &rhomax, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  ++pts->v4_calls;
  if (global_variable::my_rank != 0) return;
  FILE *f = std::fopen(pts->fname.c_str(), "a");
  if (f == nullptr) return;
  const Real lrmin = log(pts->v4_rmin), dlr = (log(64.0) - lrmin)/kNShell;
  for (int b = 0; b < kNB; ++b) {
    const Real lo = (b == 0) ? 0.0 : exp(lrmin + (b - 1)*dlr);
    const Real hi = (b == kNB - 1) ? 1.0e300 : exp(lrmin + b*dlr);
    std::fprintf(f, "S %.10e %2d %.6e %.6e", pm->time, b, lo, (b == kNB - 1) ? -1.0 : hi);
    for (int q = 0; q < kNQ; ++q) std::fprintf(f, " %.10e", s[b*kNQ + q]);
    std::fprintf(f, "\n");
  }
  const Real *g = &s[kNB*kNQ];
  const Real rm = pts->on ? pts->t.rho_max : 0.0;
  std::fprintf(f, "G %.10e %d %.10e %.10e %.10e %.10e %.10e %.10e %.0f %.0f %.0f %.10e "
               "%.10e %.10e %.10e", pm->time, pm->ncycle, rhomax,
               (rm > 0.0) ? rhomax/rm : 0.0, (g[gL1d] > 0.0) ? g[gL1n]/g[gL1d] : 0.0,
               (g[gD] > 0.0) ? g[gDr]/g[gD] : 0.0, g[gD], g[gAtm], g[gNFloor], g[gNFloorT],
               g[gNTorus], pmbp->pcfc->accreted_mass_, pmbp->pcfc->accreted_mom_[0],
               pmbp->pcfc->accreted_mom_[1], pmbp->pcfc->accreted_mom_[2]);
  if (pts->v4_test) {
    std::fprintf(f, " %.6e %.6e %.6e",
                 (g[gZtScale] > 0.0) ? g[gZtK]/g[gZtScale] : 0.0,
                 (g[gZtScaleX] > 0.0) ? g[gZtX]/g[gZtScaleX] : 0.0,
                 (g[gZtScale] > 0.0) ? g[gZtJ]/g[gZtScale] : 0.0);
  }
  std::fprintf(f, "\n");
  std::fclose(f);
}

void KerrTorusFinalize() {
  if (pts != nullptr) {
    delete pts;
    pts = nullptr;
  }
}

}  // namespace cfc
