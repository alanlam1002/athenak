//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cfc_tidal.cpp
//! \brief <cfc> tidal_lapse: the host-side E_ij(tau) table (see cfc_tidal.hpp).

#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "cfc_tidal.hpp"

namespace cfc {

namespace {

// E_ij for a tensor with in-plane eigen-direction at angle a: radial eigenvalue er, in-plane
// transverse et, out of plane ez (so E = er n n + et t t + ez z z)
void FillE(Real a, Real er, Real et, Real ez, Real e[6]) {
  const Real c = std::cos(a), s = std::sin(a);
  e[0] = er*c*c + et*s*s;     // xx
  e[1] = (er - et)*c*s;       // xy
  e[2] = 0.0;                 // xz
  e[3] = er*s*s + et*c*c;     // yy
  e[4] = 0.0;                 // yz
  e[5] = ez;                  // zz
}

}  // namespace

void TidalLapse::Setup(ParameterInput *pin) {
  on = pin->GetOrAddBoolean("cfc", "tidal_lapse", false);
  if (!on) return;
  const std::string mode = pin->GetOrAddString("cfc", "tidal_mode", "newtonian");
  if (mode != "newtonian" && mode != "marck") {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<cfc> tidal_mode = '" << mode << "' (newtonian or marck)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  marck = (mode == "marck");
  m = pin->GetOrAddReal("cfc", "tidal_mbh", 1.0);
  rp = pin->GetOrAddReal("cfc", "tidal_rp", 10.0);
  r0 = pin->GetOrAddReal("cfc", "tidal_r0", 30.0);
  fmin = pin->GetOrAddReal("cfc", "tidal_fmin", 0.1);
  dtau = pin->GetOrAddReal("cfc", "tidal_dtau", 0.01);
  const Real tmax = pin->GetOrAddReal("cfc", "tidal_tmax", 1000.0);
  cx = pin->GetOrAddReal("cfc", "tidal_center_x1", 0.0);
  cy = pin->GetOrAddReal("cfc", "tidal_center_x2", 0.0);
  cz = pin->GetOrAddReal("cfc", "tidal_center_x3", 0.0);
  if (!(r0 > rp) || !(rp > (marck ? 4.0*m : 0.0))) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<cfc> tidal_lapse needs tidal_r0 > tidal_rp (> 4 tidal_mbh for marck)"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const int n = static_cast<int>(std::ceil(tmax/dtau)) + 1;
  tau.resize(n); tc.resize(n); rr.resize(n); ang.resize(n);
  for (int q = 0; q < 6; ++q) e[q].resize(n);
  if (!marck) {
    // Barker: D + D^3/3 = (t - t_p)/sqrt(2 r_p^3/M), solved in closed form
    const Real s = std::sqrt(2.0*rp*rp*rp/m);
    const Real d0 = -std::sqrt(r0/rp - 1.0);
    const Real tp = -s*(d0 + d0*d0*d0/3.0);
    const Real nu0 = 2.0*std::atan(d0);
    for (int k = 0; k < n; ++k) {
      const Real t = k*dtau, x = (t - tp)/s;
      const Real d = 2.0*std::sinh(std::asinh(1.5*x)/3.0);
      const Real r = rp*(1.0 + d*d), a = 2.0*std::atan(d) - nu0;
      const Real g = m/(r*r*r);
      tau[k] = t; tc[k] = t; rr[k] = r; ang[k] = a;
      Real ek[6];
      FillE(a, -2.0*g, g, g, ek);
      for (int q = 0; q < 6; ++q) e[q][k] = ek[q];
    }
  } else {
    // Schwarzschild, E = 1: Binet u'' = M/L^2 - u + 3 M u^2 in phi (RK4), with
    // dtau/dphi = 1/(L u^2), dt/dphi = 1/((1 - 2Mu) L u^2), dPsi/dphi = 1/(1 + L^2 u^2)
    const Real l2 = 2.0*m*rp*rp/(rp - 2.0*m), l = std::sqrt(l2);
    long double y[5];   // u, du/dphi, tau, t, Psi
    y[0] = 1.0L/r0;
    const long double up2 = (1.0L - (1.0L - 2.0L*m*y[0])*(1.0L + l2*y[0]*y[0]))/l2;
    y[1] = std::sqrt(std::max(up2, 0.0L));   // inbound: u grows with phi
    y[2] = 0.0L; y[3] = 0.0L; y[4] = 0.0L;
    auto rhs = [&](const long double *a, long double *f) {
      const long double u = a[0];
      f[0] = a[1];
      f[1] = m/l2 - u + 3.0L*m*u*u;
      f[2] = 1.0L/(l*u*u);
      f[3] = 1.0L/((1.0L - 2.0L*m*u)*l*u*u);
      f[4] = 1.0L/(1.0L + l2*u*u);
    };
    // step in phi so that dtau per step <= dtau/8 at the start (r largest)
    const long double hphi = 0.125L*dtau*l*y[0]*y[0];
    std::vector<long double> ph_t, ph_tc, ph_r, ph_a;
    ph_t.push_back(0.0L); ph_tc.push_back(0.0L); ph_r.push_back(r0); ph_a.push_back(0.0L);
    while (ph_t.back() < tmax + dtau && y[0] > 0.0L) {
      long double k1[5], k2[5], k3[5], k4[5], tmp[5];
      rhs(y, k1);
      for (int q = 0; q < 5; ++q) tmp[q] = y[q] + 0.5L*hphi*k1[q];
      rhs(tmp, k2);
      for (int q = 0; q < 5; ++q) tmp[q] = y[q] + 0.5L*hphi*k2[q];
      rhs(tmp, k3);
      for (int q = 0; q < 5; ++q) tmp[q] = y[q] + hphi*k3[q];
      rhs(tmp, k4);
      for (int q = 0; q < 5; ++q) y[q] += hphi*(k1[q] + 2.0L*k2[q] + 2.0L*k3[q] + k4[q])/6.0L;
      ph_t.push_back(y[2]); ph_tc.push_back(y[3]); ph_r.push_back(1.0L/y[0]);
      ph_a.push_back(y[4]);
    }
    size_t j = 0;
    for (int k = 0; k < n; ++k) {
      const long double t = k*static_cast<long double>(dtau);
      while (j + 2 < ph_t.size() && ph_t[j + 1] < t) ++j;
      const long double w = (ph_t[j + 1] > ph_t[j]) ? (t - ph_t[j])/(ph_t[j + 1] - ph_t[j]) : 0;
      const long double r = ph_r[j] + w*(ph_r[j + 1] - ph_r[j]);
      const long double a = ph_a[j] + w*(ph_a[j + 1] - ph_a[j]);
      tau[k] = static_cast<Real>(t);
      tc[k] = static_cast<Real>(ph_tc[j] + w*(ph_tc[j + 1] - ph_tc[j]));
      rr[k] = static_cast<Real>(r); ang[k] = static_cast<Real>(a);
      const long double g = m/(r*r*r), c = 1.0L + l2/(r*r);
      Real ek[6];
      FillE(static_cast<Real>(a), static_cast<Real>(g*(1.0L - 3.0L*c)),
            static_cast<Real>(g), static_cast<Real>(g*(1.0L + 3.0L*l2/(r*r))), ek);
      for (int q = 0; q < 6; ++q) e[q][k] = ek[q];
    }
  }
  // the periapsis (largest |E|) for the log
  int kp = 0;
  for (int k = 1; k < n; ++k) if (rr[k] < rr[kp]) kp = k;
  if (global_variable::my_rank == 0) {
    std::cout << "<cfc> tidal_lapse: " << mode << " M=" << m << " r_p=" << rp << " r_0=" << r0
              << " table n=" << n << " dtau=" << dtau << "; periapsis tau=" << tau[kp]
              << " t=" << tc[kp] << " r=" << rr[kp] << " E_rr=" << e[0][kp]*std::cos(ang[kp])*
              std::cos(ang[kp]) + 2.0*e[1][kp]*std::cos(ang[kp])*std::sin(ang[kp]) +
              e[3][kp]*std::sin(ang[kp])*std::sin(ang[kp]) << " E_zz=" << e[5][kp] << std::endl;
    const std::string fn = pin->GetOrAddString("cfc", "tidal_table_file", "");
    if (!fn.empty()) {
      FILE *fp = std::fopen(fn.c_str(), "w");
      if (fp != nullptr) {
        std::fprintf(fp, "# tau t r angle Exx Exy Exz Eyy Eyz Ezz\n");
        for (int k = 0; k < n; k += 10) {
          std::fprintf(fp, "%.10g %.10g %.10g %.10g", tau[k], tc[k], rr[k], ang[k]);
          for (int q = 0; q < 6; ++q) std::fprintf(fp, " %.10e", e[q][k]);
          std::fprintf(fp, "\n");
        }
        std::fclose(fp);
      }
    }
  }
}

void TidalLapse::Eval(Real t, Real eij[6]) const {
  const int n = static_cast<int>(tau.size());
  Real s = t/dtau;
  if (!(s > 0.0)) s = 0.0;
  if (s >= n - 1) s = n - 1 - 1.0e-12;
  const int k = static_cast<int>(s);
  const Real w = s - k;
  for (int q = 0; q < 6; ++q) eij[q] = e[q][k] + w*(e[q][k + 1] - e[q][k]);
}

}  // namespace cfc
