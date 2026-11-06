//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file resistive_tests.cpp
//! \brief Flat-space mesh tests of resistive GRMHD on the dyngr path (07_scoping_v2 step
//!        1a, §4.1 "1D (flat)", "2D (flat)", C1, C2, C4, C5).
//!
//! <problem> test =
//!   emwave      vacuum EM plane wave (sigma = 0), 1D or 2D oblique; L1 error vs exact
//!   telegraph   standing EM mode in a static conductor: E_y = e(t) sin kx,
//!               B_z = b(t) cos kx, e'' + sigma e' + k^2 e = 0 (underdamped: telegraph
//!               damping; overdamped: magnetic diffusion)
//!   cursheet    self-similar current sheet B_y = B0 erf(x/sqrt(4 eta t)), eta=1/sigma
//!               (Komissarov 2007)
//!   riemann     shock tube with ideal E = -v x B on both sides (sigma scan)
//!   orszag_tang relativistic Orszag-Tang vortex with ideal E (ideal-limit regression)
//!               optional uniform B_z = bz_ot gives in-plane E and charge (SMR tests)
//!   cpalfven    circularly polarised Alfven wave along B_x = b0 (1D): exact damped
//!               eigenmode of the linearised resistive equations (dispersion
//!               (w^2-k^2)(w + i sigma b0^2/h) + i sigma w^2 = 0,
//!               h = rho + gamma p/(gamma-1))
//!   rotor       relativistic rotor (Del Zanna et al. 2003): rho_in, omega, r0, r1 taper
//!   harris      force-free periodic double Harris sheet B_x = b0 [tanh((y-y1)/a) -
//!               tanh((y-y2)/a) - 1], B_z = sqrt(b0^2 - B_x^2), y1,2 = -/+ L_y/4,
//!               E = J/sigma (mesh centred on x = y = 0);
//!               pert = mode (tearing: A_z += eps b0 a cos(kx) per sheet) or xpoint
//!               (Sweet-Parker: localised A_z bump of size lam at x = 0 on each sheet).
//!               With <problem> user_hist = true the history output has B_y^2 integral
//!               ("by2") and the reconnected flux of sheet 1 ("psi" = 0.5 int |B_y| dx)
//!   kh          double shear layer (Mignone et al. 2019 sec. 5.6, T-S2):
//!               phi = (|y|-1/2)/a_kh,
//!               v_x = v_sh tanh phi, rho = rho_h (1+tanh phi)/2 + rho_l (1-tanh phi)/2,
//!               v_y = sign(y) a0_kh v_sh sin(2 pi x) exp(-(phi a_kh/alpha_kh)^2),
//!               B = (sqrt(2 p mu_p), 0, sqrt(2 p mu_t)), ideal E.  user_hist adds
//!               "dvy" = (max v^y - min v^y)/2, "bp2" = int (B_x^2 + B_y^2) dV and
//!               "divb" = max |div B|
//! Passive scalars (e.g. Y_e for a tabulated EOS) are set uniformly to <problem> ye.
//!
//! The same pgen runs with <mhd> resistive = false (ideal dyngr) for ideal-limit
//! comparisons; then E is not used.  At the end of the run the pgen writes
//! <basename>.resist.dat (errors and diagnostics, rank 0) and, with
//! <problem> dump_state = true, <basename>.state.<rank>.txt (exact hex dump of u0, b0
//! and E~ per global cell) for the decomposition (C4) and restart (C5) tests.

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <complex>
#include <cstdio>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "dyn_grmhd/resistive/resistive_dyngr.hpp"
#include "dyn_grmhd/resistive/resistive_diagnostics.hpp"
#include "pgen/pgen.hpp"
#include "outputs/outputs.hpp"

namespace {

enum RTestType {kEMWave = 0, kTelegraph = 1, kCurSheet = 2, kRiemann = 3, kOT = 4,
                kCPAlfven = 5, kRotor = 6, kHarris = 7, kKH = 8};

struct RTest {
  int type;
  bool potential;              // in-plane B (and E for emwave) from z-potentials
  Real rho, pgas;
  // emwave
  Real amp1, amp2, kx, ky, kmag;
  // telegraph: E_y = et sin(kx x), B_z = bt cos(kx x)
  Real et, bt;
  // current sheet
  Real b0, eta, tsh;
  // riemann
  Real x0, wl[5], wr[5], bl[3], br[3];
  // orszag-tang
  Real vot, bot, bzot;
  // cp alfven: complex amplitudes (re, im) of v_y, B_y, E_z; omega = (omr, omi)
  Real cv[2], cb[2], ce[2], omr, omi;
  // rotor
  Real rhoin, omg, r0, r1;
  // harris sheet
  Real ash, ly, eps, lam;
  bool xpoint;
  // KH double shear layer (Mignone et al. 2019 §5.6): v_sh, a, rho_h, rho_l, A0, alpha,
  // B = (bxkh, 0, bzkh)
  Real vsh, akh, rhoh, rhol, a0kh, alkh, bxkh, bzkh;

  KOKKOS_INLINE_FUNCTION
  void Fluid(const Real x, const Real y, Real w[5]) const {
    // w = (rho, p, v^x, v^y, v^z) (3-velocity)
    w[0] = rho; w[1] = pgas; w[2] = 0.0; w[3] = 0.0; w[4] = 0.0;
    if (type == kRiemann) {
      for (int n = 0; n < 5; ++n) w[n] = (x < x0) ? wl[n] : wr[n];
    } else if (type == kOT) {
      const Real tpi = 2.0*M_PI;
      w[2] = -vot*sin(tpi*y);
      w[3] = vot*sin(tpi*x);
    } else if (type == kCPAlfven) {
      // y polarisation Re(c e^{ikx}); z polarisation Re(-i c e^{ikx})
      const Real cs = cos(kx*x), sn = sin(kx*x);
      w[3] = cv[0]*cs - cv[1]*sn;
      w[4] = cv[1]*cs + cv[0]*sn;
    } else if (type == kRotor) {
      const Real r = sqrt(x*x + y*y);
      const Real f = (r < r0) ? 1.0 : ((r < r1) ? (r1 - r)/(r1 - r0) : 0.0);
      w[0] = rho + (rhoin - rho)*f;
      w[2] = -f*omg*y;
      w[3] = f*omg*x;
    } else if (type == kKH) {
      const Real phi = (fabs(y) - 0.5)/akh;
      const Real th = tanh(phi);
      w[0] = 0.5*(1.0 + th)*rhoh + 0.5*(1.0 - th)*rhol;
      w[2] = vsh*th;
      w[3] = ((y < 0.0) ? -1.0 : 1.0)*a0kh*vsh*sin(2.0*M_PI*x)
             *exp(-phi*phi*akh*akh/(alkh*alkh));
    }
  }

  KOKKOS_INLINE_FUNCTION
  Real HarrisBx(const Real y) const {
    return b0*(tanh((y + 0.25*ly)/ash) - tanh((y - 0.25*ly)/ash) - 1.0);
  }

  KOKKOS_INLINE_FUNCTION
  void Fields(const Real x, const Real y, const Real t, Real E[3], Real B[3]) const {
    for (int a = 0; a < 3; ++a) { E[a] = 0.0; B[a] = 0.0; }
    if (type == kEMWave) {
      const Real s = sin(kx*x + ky*y - kmag*t);
      const Real nx = kx/kmag, ny = ky/kmag;
      E[0] = -amp2*s*ny; E[1] = amp2*s*nx; E[2] = amp1*s;
      B[0] = amp1*s*ny;  B[1] = -amp1*s*nx; B[2] = amp2*s;
    } else if (type == kTelegraph) {
      E[1] = et*sin(kx*x);
      B[2] = bt*cos(kx*x);
    } else if (type == kCurSheet) {
      const Real d = sqrt(4.0*eta*tsh);
      B[1] = b0*erf(x/d);
      E[2] = eta*b0*exp(-x*x/(d*d))*2.0/(sqrt(M_PI)*d);
    } else if (type == kCPAlfven) {
      const Real g = exp(omi*t);
      const Real cs = cos(kx*x - omr*t), sn = sin(kx*x - omr*t);
      B[0] = b0;
      B[1] = g*(cb[0]*cs - cb[1]*sn);
      E[2] = g*(ce[0]*cs - ce[1]*sn);
      B[2] = g*(cb[1]*cs + cb[0]*sn);
      E[1] = -g*(ce[1]*cs + ce[0]*sn);
    } else if (type == kHarris) {
      // background only (the A_z perturbation enters through the potential)
      const Real s1 = 1.0/cosh((y + 0.25*ly)/ash), s2 = 1.0/cosh((y - 0.25*ly)/ash);
      const Real bx = HarrisBx(y);
      const Real dbx = b0*(s1*s1 - s2*s2)/ash;
      const Real bz = sqrt(fmax(b0*b0 - bx*bx, 0.0));
      B[0] = bx;
      B[2] = bz;
      E[0] = (bz > 0.0) ? -eta*bx*dbx/bz : 0.0;  // eta J_x = eta dB_z/dy
      E[2] = -eta*dbx;                            // eta J_z = -eta dB_x/dy
    } else {
      Real w[5];
      Fluid(x, y, w);
      if (type == kRiemann) {
        for (int a = 0; a < 3; ++a) B[a] = (x < x0) ? bl[a] : br[a];
      } else if (type == kRotor) {
        B[0] = b0;
      } else if (type == kKH) {
        B[0] = bxkh;
        B[2] = bzkh;
      } else {
        const Real tpi = 2.0*M_PI;
        B[0] = -bot*sin(tpi*y);
        B[1] = bot*sin(2.0*tpi*x);
        B[2] = bzot;
      }
      const Real *v = &w[2];
      E[0] = -(v[1]*B[2] - v[2]*B[1]);
      E[1] = -(v[2]*B[0] - v[0]*B[2]);
      E[2] = -(v[0]*B[1] - v[1]*B[0]);
    }
  }

  // z-potentials: B_inplane = (d_y Az, -d_x Az), E_inplane = (d_y Psi, -d_x Psi)
  KOKKOS_INLINE_FUNCTION
  Real Az(const Real x, const Real y, const Real t) const {
    if (type == kEMWave) return -(amp1/kmag)*cos(kx*x + ky*y - kmag*t);
    if (type == kHarris) {
      const Real y1 = -0.25*ly, y2 = 0.25*ly;
      Real a = b0*(ash*log(cosh((y - y1)/ash)) - ash*log(cosh((y - y2)/ash)) - y);
      if (xpoint) {
        a += eps*b0*lam*(exp(-(x*x + (y - y1)*(y - y1))/(lam*lam)) -
                         exp(-(x*x + (y - y2)*(y - y2))/(lam*lam)));
      } else {
        a += eps*b0*ash*cos(kx*x)*(1.0/cosh((y - y1)/ash) - 1.0/cosh((y - y2)/ash));
      }
      return a;
    }
    const Real tpi = 2.0*M_PI;
    return bot*(cos(2.0*tpi*x)/(2.0*tpi) + cos(tpi*y)/tpi);
  }
  KOKKOS_INLINE_FUNCTION
  Real Psi(const Real x, const Real y, const Real t) const {
    if (type == kEMWave) return (amp2/kmag)*cos(kx*x + ky*y - kmag*t);
    return 0.0;
  }
};

RTest rt;
Real charge0 = 0.0;
bool have_charge0 = false;

// Telegraph mode amplitudes at time t: e'' + sigma e' + k^2 e = 0, e(0)=A, b(0)=0,
// b = (e' + sigma e)/k.
void TelegraphAmplitudes(Real a, Real sigma, Real k, Real t, Real &e, Real &b) {
  using C = std::complex<double>;
  C disc = std::sqrt(C(0.25*sigma*sigma - k*k, 0.0));
  C l1 = -0.5*sigma + disc, l2 = -0.5*sigma - disc;
  C ee, de;
  if (std::abs(l1 - l2) < 1.0e-12*std::abs(l1)) {
    // critically damped: e = A (1 + (l + sigma) t) e^{l t}... use the limit form
    C l = l1;
    C c2 = (-sigma - l)*a;
    ee = (C(a) + c2*t)*std::exp(l*t);
    de = (l*(C(a) + c2*t) + c2)*std::exp(l*t);
  } else {
    C c1 = a*(-sigma - l2)/(l1 - l2);
    C c2 = C(a) - c1;
    ee = c1*std::exp(l1*t) + c2*std::exp(l2*t);
    de = c1*l1*std::exp(l1*t) + c2*l2*std::exp(l2*t);
  }
  e = ee.real();
  b = ((de + sigma*ee)/k).real();
}

// Fill face B and E~ (flat: densitised == undensitised) at time t over all faces of
// the interior plus ghosts of the pack.
void SetFaceFields(MeshBlockPack *pmbp, const RTest &r, const Real t,
                   DvceFaceFld4D<Real> bf, DvceFaceFld4D<Real> *ef) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const bool multi_d = pmbp->pmesh->multi_d;
  auto &size = pmbp->pmb->mb_size;
  const bool pot = r.potential && multi_d;
  const bool sete = (ef != nullptr);
  DvceFaceFld4D<Real> e = sete ? *ef : bf;

  par_for("rt_face", DevExeSpace(), 0, nmb1, ks, ke+1, js, je+1, is, ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x1min = size.d_view(m).x1min, x1max = size.d_view(m).x1max;
    const Real x2min = size.d_view(m).x2min, x2max = size.d_view(m).x2max;
    const Real dx1 = size.d_view(m).dx1, dx2 = size.d_view(m).dx2;
    const Real x1f = LeftEdgeX(i-is, nx1, x1min, x1max);
    const Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
    const Real x2f = LeftEdgeX(j-js, nx2, x2min, x2max);
    const Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
    const Real x2fp = LeftEdgeX(j+1-js, nx2, x2min, x2max);
    const Real x1fp = LeftEdgeX(i+1-is, nx1, x1min, x1max);
    Real E[3], B[3];
    // x1 faces
    if (j <= je && k <= ke) {
      r.Fields(x1f, x2v, t, E, B);
      Real bx = B[0], ex = E[0];
      if (pot) {
        bx = (r.Az(x1f, x2fp, t) - r.Az(x1f, x2f, t))/dx2;
        if (r.type == kEMWave) ex = (r.Psi(x1f, x2fp, t) - r.Psi(x1f, x2f, t))/dx2;
      }
      bf.x1f(m, k, j, i) = bx;
      if (sete) e.x1f(m, k, j, i) = ex;
    }
    // x2 faces
    if (i <= ie && k <= ke) {
      r.Fields(x1v, multi_d ? x2f : x2v, t, E, B);
      Real by = B[1], ey = E[1];
      if (pot) {
        by = -(r.Az(x1fp, x2f, t) - r.Az(x1f, x2f, t))/dx1;
        if (r.type == kEMWave) ey = -(r.Psi(x1fp, x2f, t) - r.Psi(x1f, x2f, t))/dx1;
      }
      bf.x2f(m, k, j, i) = by;
      if (sete) e.x2f(m, k, j, i) = ey;
    }
    // x3 faces
    if (i <= ie && j <= je) {
      r.Fields(x1v, x2v, t, E, B);
      bf.x3f(m, k, j, i) = B[2];
      if (sete) e.x3f(m, k, j, i) = E[2];
    }
  });
  (void)nx3;
}

void ResistiveTestsFinal(ParameterInput *pin, Mesh *pm);
void ResistiveTestsHistory(HistoryData *pdata, Mesh *pm);

// Root of the CP Alfven dispersion cubic with the largest real part (Durand-Kerner,
// then Newton polish).
std::complex<double> CPAlfvenOmega(Real k, Real sigma, Real b2) {
  using C = std::complex<double>;
  const C c2(0.0, sigma*(1.0 + b2)), c1(-k*k, 0.0), c0(0.0, -sigma*b2*k*k);
  auto f = [&](C w) { return ((w + c2)*w + c1)*w + c0; };
  auto df = [&](C w) { return (3.0*w + 2.0*c2)*w + c1; };
  const double rad = 1.0 + std::max(std::abs(c2), std::max(std::abs(c1), std::abs(c0)));
  C z[3] = {rad*C(0.4, 0.9), rad*C(0.4, 0.9)*C(0.4, 0.9), rad*std::pow(C(0.4, 0.9), 3)};
  for (int it = 0; it < 5000; ++it) {
    for (int a = 0; a < 3; ++a) {
      C den(1.0, 0.0);
      for (int b = 0; b < 3; ++b) {
        if (b != a) den *= (z[a] - z[b]);
      }
      z[a] -= f(z[a])/den;
    }
  }
  int ib = 0;
  for (int a = 1; a < 3; ++a) {
    if (z[a].real() > z[ib].real()) ib = a;
  }
  C w = z[ib];
  for (int it = 0; it < 20; ++it) {
    const C d = df(w);
    if (std::abs(d) == 0.0) break;
    w -= f(w)/d;
  }
  return w;
}

}  // namespace

//----------------------------------------------------------------------------------------
void ProblemGenerator::ResistiveTests(ParameterInput *pin, const bool restart) {
  pgen_final_func = ResistiveTestsFinal;
  user_hist_func = ResistiveTestsHistory;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr || pmbp->pdyngr == nullptr || pmbp->padm == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "resistive_tests requires <mhd> on the dyngr path (<adm>)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::string test = pin->GetString("problem", "test");
  if (test == "emwave") {
    rt.type = kEMWave;
  } else if (test == "telegraph") {
    rt.type = kTelegraph;
  } else if (test == "cursheet") {
    rt.type = kCurSheet;
  } else if (test == "riemann") {
    rt.type = kRiemann;
  } else if (test == "orszag_tang") {
    rt.type = kOT;
  } else if (test == "cpalfven") {
    rt.type = kCPAlfven;
  } else if (test == "rotor") {
    rt.type = kRotor;
  } else if (test == "harris") {
    rt.type = kHarris;
  } else if (test == "kh") {
    rt.type = kKH;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<problem> test = '" << test << "' unknown" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto &ms = pmy_mesh_->mesh_size;
  const Real lx = ms.x1max - ms.x1min, ly = ms.x2max - ms.x2min;
  rt.potential = (rt.type == kEMWave || rt.type == kOT || rt.type == kHarris);
  rt.rho = pin->GetOrAddReal("problem", "rho", 1.0);
  rt.pgas = pin->GetOrAddReal("problem", "pgas", 1.0);
  rt.amp1 = pin->GetOrAddReal("problem", "amp1", 1.0e-3);
  rt.amp2 = pin->GetOrAddReal("problem", "amp2", 0.0);
  rt.kx = 2.0*M_PI/lx;
  rt.ky = pmy_mesh_->multi_d ? 2.0*M_PI/ly : 0.0;
  rt.kmag = sqrt(rt.kx*rt.kx + rt.ky*rt.ky);
  rt.b0 = pin->GetOrAddReal("problem", "b0", 1.0);
  rt.tsh = pin->GetOrAddReal("problem", "t0", 1.0);
  const Real sigma = pin->GetOrAddReal("mhd", "resistive_sigma", 0.0);
  rt.eta = (sigma > 0.0) ? 1.0/sigma : 0.0;
  rt.x0 = pin->GetOrAddReal("problem", "x0", 0.0);
  const char *nm[5] = {"rho", "p", "vx", "vy", "vz"};
  for (int n = 0; n < 5; ++n) {
    const Real dflt = (n < 2) ? 1.0 : 0.0;
    rt.wl[n] = pin->GetOrAddReal("problem", std::string(nm[n]) + "_l", dflt);
    rt.wr[n] = pin->GetOrAddReal("problem", std::string(nm[n]) + "_r", dflt);
  }
  const char *nb[3] = {"bx", "by", "bz"};
  for (int n = 0; n < 3; ++n) {
    rt.bl[n] = pin->GetOrAddReal("problem", std::string(nb[n]) + "_l", 0.0);
    rt.br[n] = pin->GetOrAddReal("problem", std::string(nb[n]) + "_r", 0.0);
  }
  rt.vot = pin->GetOrAddReal("problem", "v_ot", 0.5);
  rt.bot = pin->GetOrAddReal("problem", "b_ot", 1.0/sqrt(4.0*M_PI));
  rt.bzot = pin->GetOrAddReal("problem", "bz_ot", 0.0);
  if (rt.type == kTelegraph) {
    TelegraphAmplitudes(rt.amp1, sigma, rt.kx, 0.0, rt.et, rt.bt);
  }
  rt.rhoin = pin->GetOrAddReal("problem", "rho_in", 10.0);
  rt.omg = pin->GetOrAddReal("problem", "omega", 9.95);
  rt.r0 = pin->GetOrAddReal("problem", "r0", 0.1);
  rt.r1 = pin->GetOrAddReal("problem", "r1", 0.115);
  rt.ash = pin->GetOrAddReal("problem", "a_sheet", 0.1);
  rt.ly = ly;
  rt.eps = pin->GetOrAddReal("problem", "eps", 1.0e-3);
  rt.lam = pin->GetOrAddReal("problem", "lam", 0.1);
  rt.xpoint = (pin->GetOrAddString("problem", "pert", "mode") == "xpoint");
  rt.vsh = pin->GetOrAddReal("problem", "v_sh", 0.5);
  rt.akh = pin->GetOrAddReal("problem", "a_kh", 1.0e-2);
  rt.rhoh = pin->GetOrAddReal("problem", "rho_h", 1.0);
  rt.rhol = pin->GetOrAddReal("problem", "rho_l", 1.0e-2);
  rt.a0kh = pin->GetOrAddReal("problem", "a0_kh", 0.1);
  rt.alkh = pin->GetOrAddReal("problem", "alpha_kh", 0.1);
  rt.bxkh = sqrt(2.0*rt.pgas*pin->GetOrAddReal("problem", "mu_p", 1.0e-2));
  rt.bzkh = sqrt(2.0*rt.pgas*pin->GetOrAddReal("problem", "mu_t", 1.0));
  if (rt.type == kCPAlfven) {
    using C = std::complex<double>;
    const Real gm = pin->GetReal("mhd", "gamma");
    const Real h = rt.rho + gm/(gm - 1.0)*rt.pgas;
    const Real b2 = rt.b0*rt.b0/h;
    const C w = CPAlfvenOmega(rt.kx, sigma, b2);
    rt.omr = w.real();
    rt.omi = w.imag();
    const C ce = rt.amp1*std::abs(w)/rt.kx;                  // |B_y| = amp1
    const C cb = -(rt.kx/w)*ce;
    const C cj = sigma*ce/(1.0 + C(0.0, sigma*b2)/w);
    const C cv = C(0.0, 1.0)*cj*rt.b0/(w*h);
    rt.ce[0] = ce.real(); rt.ce[1] = ce.imag();
    rt.cb[0] = cb.real(); rt.cb[1] = cb.imag();
    rt.cv[0] = cv.real(); rt.cv[1] = cv.imag();
    if (global_variable::my_rank == 0) {
      std::printf("cpalfven: omega = %.12e %+.12e i\n", rt.omr, rt.omi);
    }
  }
  if (restart) return;

  resistive::ResistiveDynGR *pr = pmbp->pdyngr->presist;
  pmbp->padm->SetADMVariables(pmbp);
  SetFaceFields(pmbp, rt, 0.0, pmbp->pmhd->b0, (pr != nullptr) ? &pr->e0 : nullptr);

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const bool multi_d = pmy_mesh_->multi_d, three_d = pmy_mesh_->three_d;
  auto &size = pmbp->pmb->mb_size;
  auto w0 = pmbp->pmhd->w0;
  auto bcc = pmbp->pmhd->bcc0;
  const int nmhd = pmbp->pmhd->nmhd, nscal = pmbp->pmhd->nscalars;
  const Real ye = pin->GetOrAddReal("problem", "ye", 0.1);
  auto bf = pmbp->pmhd->b0;
  const RTest r = rt;
  par_for("rt_cells", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    const Real x2v = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real w[5];
    r.Fluid(x1v, x2v, w);
    const Real lor = 1.0/sqrt(1.0 - (w[2]*w[2] + w[3]*w[3] + w[4]*w[4]));
    w0(m, IDN, k, j, i) = w[0];
    w0(m, IPR, k, j, i) = w[1];
    w0(m, IVX, k, j, i) = lor*w[2];
    w0(m, IVY, k, j, i) = lor*w[3];
    w0(m, IVZ, k, j, i) = lor*w[4];
    for (int n = 0; n < nscal; ++n) w0(m, nmhd + n, k, j, i) = ye;
    bcc(m, IBX, k, j, i) = 0.5*(bf.x1f(m, k, j, i) + bf.x1f(m, k, j, i+1));
    bcc(m, IBY, k, j, i) = multi_d ? 0.5*(bf.x2f(m, k, j, i) + bf.x2f(m, k, j+1, i))
                                   : bf.x2f(m, k, j, i);
    bcc(m, IBZ, k, j, i) = three_d ? 0.5*(bf.x3f(m, k, j, i) + bf.x3f(m, k+1, j, i))
                                   : bf.x3f(m, k, j, i);
  });
  pmbp->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);
  if (pr != nullptr) {
    resistive::ChargeDiag cd = resistive::TotalCharge(pmbp, pr->e0);
    charge0 = cd.total;
    have_charge0 = true;
  }
}

namespace {

//----------------------------------------------------------------------------------------
// L1 errors of the face fields vs the exact solution, normalised by the cell count.
void FaceErrors(MeshBlockPack *pmbp, const RTest &r, const Real t, Real err_e[3],
                Real err_b[3]) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*indcs.ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*indcs.ng : 1;
  const int nmb = pmbp->nmb_thispack;
  DvceFaceFld4D<Real> bx("rt_bx", nmb, n3, n2, n1), ex("rt_ex", nmb, n3, n2, n1);
  SetFaceFields(pmbp, r, t, bx, &ex);
  resistive::ResistiveDynGR *pr = pmbp->pdyngr->presist;
  auto bf = pmbp->pmhd->b0;
  DvceFaceFld4D<Real> ef = (pr != nullptr) ? pr->e0 : ex;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1;
  Real s[6] = {0, 0, 0, 0, 0, 0};
  const int nkji = indcs.nx3*indcs.nx2*indcs.nx1, nji = indcs.nx2*indcs.nx1;
  for (int c = 0; c < 6; ++c) {
    Real sum = 0.0;
    Kokkos::parallel_reduce("rt_err", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*nkji),
    KOKKOS_LAMBDA(const int idx, Real &acc) {
      const int m = idx/nkji;
      int rr = idx - m*nkji;
      const int k = rr/nji + ks;
      rr -= (k-ks)*nji;
      const int j = rr/nx1 + js;
      const int i = rr - (j-js)*nx1 + is;
      Real d = 0.0;
      if (c == 0) d = ef.x1f(m, k, j, i) - ex.x1f(m, k, j, i);
      if (c == 1) d = ef.x2f(m, k, j, i) - ex.x2f(m, k, j, i);
      if (c == 2) d = ef.x3f(m, k, j, i) - ex.x3f(m, k, j, i);
      if (c == 3) d = bf.x1f(m, k, j, i) - bx.x1f(m, k, j, i);
      if (c == 4) d = bf.x2f(m, k, j, i) - bx.x2f(m, k, j, i);
      if (c == 5) d = bf.x3f(m, k, j, i) - bx.x3f(m, k, j, i);
      acc += fabs(d);
    }, Kokkos::Sum<Real>(sum));
    s[c] = sum;
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, s, 6, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  const Real ntot = static_cast<Real>(ncells)*pmbp->pmesh->nmb_total;
  for (int a = 0; a < 3; ++a) { err_e[a] = s[a]/ntot; err_b[a] = s[3+a]/ntot; }
}

// Exact per-cell dump (hex floats) keyed by global cell index.
void DumpState(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nmhd = pmbp->pmhd->nmhd + pmbp->pmhd->nscalars;
  auto u0 = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmbp->pmhd->u0);
  auto b1 = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmbp->pmhd->b0.x1f);
  auto b2 = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmbp->pmhd->b0.x2f);
  auto b3 = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmbp->pmhd->b0.x3f);
  resistive::ResistiveDynGR *pr = pmbp->pdyngr->presist;
  const bool hase = (pr != nullptr);
  auto e1 = Kokkos::create_mirror_view_and_copy(HostMemSpace(),
                                                hase ? pr->e0.x1f : pmbp->pmhd->b0.x1f);
  auto e2 = Kokkos::create_mirror_view_and_copy(HostMemSpace(),
                                                hase ? pr->e0.x2f : pmbp->pmhd->b0.x2f);
  auto e3 = Kokkos::create_mirror_view_and_copy(HostMemSpace(),
                                                hase ? pr->e0.x3f : pmbp->pmhd->b0.x3f);
  std::string fname = pin->GetString("job", "basename") + ".state." +
                      std::to_string(global_variable::my_rank) + ".txt";
  FILE *fp = std::fopen(fname.c_str(), "w");
  for (int m = 0; m < pmbp->nmb_thispack; ++m) {
    auto &lloc = pm->lloc_eachmb[pmbp->gids + m];
    for (int k = 0; k < indcs.nx3; ++k) {
      for (int j = 0; j < indcs.nx2; ++j) {
        for (int i = 0; i < indcs.nx1; ++i) {
          const int64_t gi = lloc.lx1*indcs.nx1 + i;
          const int64_t gj = lloc.lx2*indcs.nx2 + j;
          const int64_t gk = lloc.lx3*indcs.nx3 + k;
          std::fprintf(fp, "%" PRId64 " %" PRId64 " %" PRId64, gk, gj, gi);
          const int kk = k + ks, jj = j + js, ii = i + is;
          for (int n = 0; n < nmhd; ++n) std::fprintf(fp, " %a", u0(m, n, kk, jj, ii));
          std::fprintf(fp, " %a %a %a", b1(m, kk, jj, ii), b2(m, kk, jj, ii),
                       b3(m, kk, jj, ii));
          if (hase) {
            std::fprintf(fp, " %a %a %a", e1(m, kk, jj, ii), e2(m, kk, jj, ii),
                         e3(m, kk, jj, ii));
          }
          std::fprintf(fp, "\n");
        }
      }
    }
  }
  std::fclose(fp);
}

void ResistiveTestsFinal(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  resistive::ResistiveDynGR *pr = pmbp->pdyngr->presist;
  RTest r = rt;
  const Real t = pm->time;
  if (r.type == kTelegraph) {
    const Real sigma = pin->GetOrAddReal("mhd", "resistive_sigma", 0.0);
    TelegraphAmplitudes(r.amp1, sigma, r.kx, t, r.et, r.bt);
  }
  if (r.type == kCurSheet) r.tsh += t;
  Real err_e[3] = {0, 0, 0}, err_b[3] = {0, 0, 0};
  const bool analytic = (r.type == kEMWave || r.type == kTelegraph ||
                         r.type == kCurSheet || r.type == kCPAlfven);
  if (analytic) FaceErrors(pmbp, r, t, err_e, err_b);
  resistive::ChargeDiag cd = {0.0, 0.0, 0.0};
  int64_t ns[resistive::kNewtonNCounters] = {0, 0, 0, 0, 0};
  if (pr != nullptr) {
    cd = resistive::TotalCharge(pmbp, pr->e0);
    pr->newton.Read(ns);
  }
  const Real divb = resistive::MaxDivB(pmbp, pmbp->pmhd->b0);
  if (pin->GetOrAddBoolean("problem", "dump_state", false)) DumpState(pin, pm);
  if (global_variable::my_rank != 0) return;
  std::string fname = pin->GetString("job", "basename") + ".resist.dat";
  FILE *fp = std::fopen(fname.c_str(), "w");
  std::fprintf(fp, "test %d\nnx1 %d\nnx2 %d\ntime %.17e\ncycle %d\n", r.type,
               pm->mesh_indcs.nx1, pm->mesh_indcs.nx2, t, pm->ncycle);
  std::fprintf(fp, "err_e1 %.17e\nerr_e2 %.17e\nerr_e3 %.17e\n", err_e[0], err_e[1],
               err_e[2]);
  std::fprintf(fp, "err_b1 %.17e\nerr_b2 %.17e\nerr_b3 %.17e\n", err_b[0], err_b[1],
               err_b[2]);
  std::fprintf(fp, "charge0 %.17e\ncharge %.17e\ncharge_boundary %.17e\n"
               "max_abs_q %.17e\n", have_charge0 ? charge0 : 0.0, cd.total, cd.boundary,
               cd.max_abs_q);
  std::fprintf(fp, "max_divb %.17e\n", divb);
  const char *nn[5] = {"newton_cells", "newton_iter_sum", "newton_iter_max",
                       "newton_fallback", "newton_fail"};
  for (int n = 0; n < 5; ++n) {
    std::fprintf(fp, "%s %" PRId64 "\n", nn[n], static_cast<int64_t>(ns[n]));
  }
  std::fclose(fp);
}

// User history: by2 = int B_y^2 dV (tearing growth), psi = 0.5 int |B_y| dx along the
// x2 faces at y = -L_y/4 (reconnected flux of sheet 1, Sweet-Parker rate),
// dvy = (max v^y - min v^y)/2 (KH growth, Mignone 2019 sec. 5.6),
// bp2 = int (B_x^2 + B_y^2) dV (poloidal field amplification),
// divb = max |div B| (same as max_divb in .resist.dat; T-S2 criterion 6 over time),
// bxk = max over the rows y < 0 (sheet 1) of the k-mode amplitude
// (2/L_x) |int B_x exp(-i k x) dx|, k = 2 pi/L_x (tearing gate, 07 v2.4 §4.1; uniform
// mesh only).
void ResistiveTestsHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 6;
  pdata->label[0] = "by2";
  pdata->label[1] = "psi";
  pdata->label[2] = "dvy";
  pdata->label[3] = "bp2";
  pdata->label[4] = "divb";
  pdata->label[5] = "bxk";
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2;
  const int nkji = indcs.nx3*nx2*nx1, nji = nx2*nx1;
  const int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;
  auto b2f = pmbp->pmhd->b0.x2f;
  const Real ysh = -0.25*rt.ly;
  Real by2 = 0.0, psi = 0.0;
  Kokkos::parallel_reduce("rt_hist", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*nkji),
  KOKKOS_LAMBDA(const int idx, Real &s1, Real &s2) {
    const int m = idx/nkji;
    int rr = idx - m*nkji;
    const int k = rr/nji + ks;
    rr -= (k-ks)*nji;
    const int j = rr/nx1 + js;
    const int i = rr - (j-js)*nx1 + is;
    const Real dx1 = size.d_view(m).dx1, dx2 = size.d_view(m).dx2;
    const Real dv = dx1*dx2*size.d_view(m).dx3;
    const Real by = b2f(m, k, j, i);
    s1 += by*by*dv;
    const Real yf = LeftEdgeX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    if (fabs(yf - ysh) < 0.25*dx2) s2 += 0.5*fabs(by)*dx1;
  }, Kokkos::Sum<Real>(by2), Kokkos::Sum<Real>(psi));
  auto w0 = pmbp->pmhd->w0;
  auto bcc = pmbp->pmhd->bcc0;
  Real vmax = -1.0, vmin = 1.0, bp2 = 0.0;
  Kokkos::parallel_reduce("rt_hist2", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*nkji),
  KOKKOS_LAMBDA(const int idx, Real &mx, Real &mn, Real &sb) {
    const int m = idx/nkji;
    int rr = idx - m*nkji;
    const int k = rr/nji + ks;
    rr -= (k-ks)*nji;
    const int j = rr/nx1 + js;
    const int i = rr - (j-js)*nx1 + is;
    const Real ux = w0(m, IVX, k, j, i), uy = w0(m, IVY, k, j, i);
    const Real uz = w0(m, IVZ, k, j, i);
    const Real vy = uy/sqrt(1.0 + ux*ux + uy*uy + uz*uz);  // flat background
    mx = fmax(mx, vy);
    mn = fmin(mn, vy);
    const Real bx = bcc(m, IBX, k, j, i), by = bcc(m, IBY, k, j, i);
    sb += (bx*bx + by*by)*size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
  }, Kokkos::Max<Real>(vmax), Kokkos::Min<Real>(vmin), Kokkos::Sum<Real>(bp2));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &vmax, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &vmin, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
#endif
  pdata->hdata[0] = by2;
  pdata->hdata[1] = psi;
  // the history writer sums hdata over ranks: dvy (already global) only on rank 0
  pdata->hdata[2] = (global_variable::my_rank == 0) ? 0.5*(vmax - vmin) : 0.0;
  pdata->hdata[3] = bp2;
  const Real divb = resistive::MaxDivB(pmbp, pmbp->pmhd->b0);  // global max
  pdata->hdata[4] = (global_variable::my_rank == 0) ? divb : 0.0;

  const int nrow = pm->mesh_indcs.nx2;
  const Real y0 = pm->mesh_size.x2min;
  const Real dyr = (pm->mesh_size.x2max - y0)/nrow;
  const Real lx = pm->mesh_size.x1max - pm->mesh_size.x1min, kx = rt.kx;
  DvceArray2D<Real> fk("rt_bxk", 2, nrow);
  par_for("rt_bxk", DevExeSpace(), 0, nmb-1, ks, ks+indcs.nx3-1, js, js+nx2-1, is,
          is+nx1-1, KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    const Real y = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    const int g = fmin(fmax(floor((y - y0)/dyr), 0.0), nrow - 1.0);
    const Real w = bcc(m, IBX, k, j, i)*size.d_view(m).dx1;
    Kokkos::atomic_add(&fk(0, g), w*cos(kx*x));
    Kokkos::atomic_add(&fk(1, g), -w*sin(kx*x));
  });
  auto hk = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fk);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, hk.data(), 2*nrow, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
#endif
  Real bxk = 0.0;
  for (int g = 0; g < nrow && y0 + (g + 0.5)*dyr < 0.0; ++g) {
    bxk = fmax(bxk, 2.0/lx*sqrt(hk(0, g)*hk(0, g) + hk(1, g)*hk(1, g)));
  }
  pdata->hdata[5] = (global_variable::my_rank == 0) ? bxk : 0.0;
}

}  // namespace
