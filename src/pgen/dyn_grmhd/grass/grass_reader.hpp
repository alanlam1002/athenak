#ifndef PGEN_DYN_GRMHD_RNS_GRASS_READER_HPP_
#define PGEN_DYN_GRMHD_RNS_GRASS_READER_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file grass_reader.hpp
//  \brief Host-side reader/interpolator for GRASS (RNS-family rotating-NS equilibrium
//  code) restart binaries (Res/res.rst). GRASS uses the RNS/Cook-Shapiro-Teukolsky
//  (CST) stationary-axisymmetric metric in spherical-polar coordinates:
//
//    ds^2 = -e^{gama+rho} dt^2 + e^{2alpha}(dr^2 + r^2 dtheta^2)
//           + e^{gama-rho} r^2 sin^2(theta) (dphi - ww dt)^2
//
//  Directly ADM-physical (no conformal decomposition, unlike the sibling
//  rns_st_reader.hpp for scalar-tensor data). ADM split (K_ij from
//  dt(gamma_ij) = -2*alpha*K_ij + D_i(beta_j) + D_j(beta_i)):
//
//    N = e^{(gama+rho)/2},  beta^phi = -ww,  beta^r = beta^theta = 0
//    gamma_rr = e^{2alpha},  gamma_thth = gamma_rr*r^2
//    gamma_phph = e^{gama-rho}*r^2*sin^2(theta)
//    K_rphi = -(gamma_phph/2N)*dr(ww),  K_thphi = -(gamma_phph/2N)*dth(ww)  (rest zero)
//
//  Fluid velocity is ZAMO-frame azimuthal (matches GRASS's own `velocity_sq` field),
//  giving AthenaK's primitive convention u-tilde^i = gamma^ij u_j directly:
//
//    u-tilde^phi = (omg - ww) / (N*sqrt(1 - v^2)),   v = (omg-ww)*r*sin(theta)*e^{-rho}
//
//  Restart binary layout (access='stream', no Fortran record padding, all 8-byte
//  doubles; see GRASS's spin_integration_mod.f90::write_restart_file and
//  regrid_mod.f90::read_binary_restart):
//
//    char[10]   magic = "GRASSRST01"
//    int32[6]   header_ints  = [format_version=1, storage_size=64, field_count=10,
//                                SDIV, MDIV, s_pwr]
//    double[5]  header_meta  = [r_e, e_center, r_ratio, Omega_e, Omega_c]  (GRASS units)
//    double[SDIV]        s_gp
//    double[MDIV]        mu       (= cos(theta), mu[0]=0 equator -> mu[MDIV-1]=1 pole)
//    double[10][SDIV][MDIV]  restart_data, field fastest-varying (Fortran column-major),
//                            field order: alpha,gama,rho,ww,pressure,energy,enthalpy,
//                            velocity_sq,omg,sphi
//
//  Physical radius: r_internal(s) = r_e*(s/(1-s))^s_pwr (GRASS's compactification);
//  ds/dr = s*(1-s)/(s_pwr*r) chain-rules the Lagrange-basis d/ds(ww) into dr(ww).
//
//  Grid is a half-domain in theta (mu=cos(theta) in [0,1], equator->pole). Query at
//  mu'=|cos(theta)|=|z|/r (always in-bounds), flip the sign of the theta-derivative-
//  dependent K_ij components for z<0 (theta'=arccos(|cos(theta)|), dtheta'/dtheta=-1).
//  The Lagrange stencil itself still needs ghost-padded storage near mu=0 -- see the
//  equatorial mirror in Load() below.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "athena.hpp"
#include "grass_eos_table.hpp"
#include "grass_units.hpp"
#include "utils/tov/tov_tabulated.hpp"

namespace grass {

//----------------------------------------------------------------------------------------
//! \class GrassData
//! \brief loads one GRASS restart binary and interpolates it at arbitrary Cartesian
//  points, returning AthenaK-code-unit ADM + hydro-primitive initial data.

class GrassData {
 public:
  struct Point {
    Real alpha;        // ADM lapse N (not GRASS's own `alpha` metric potential)
    Real beta_u[3];    // shift, Cartesian
    Real g_dd[6];      // xx,xy,xz,yy,yz,zz
    Real K_dd[6];      // xx,xy,xz,yy,yz,zz
    Real rho0;         // rest-mass density, AthenaK code units
    Real pres;         // pressure, AthenaK code units
    Real vu[3];        // u-tilde^i = gamma^ij u_j, Cartesian (AthenaK primitive conv.)
    Real Yq;           // Y[e], seeded from the DD2_hot_slice 1D table along the same
                        // (nb,T,Yl) trajectory that produced GRASS's e(n0),p(n0) -- a
                        // composition seed only, not load-bearing for correctness.
  };

  GrassData(const std::string &fname, const GrassUnits &units) : units_(units) {
    Load(fname);
    BuildLengthScales();
  }

  void Interpolate(Real x, Real y, Real z, const GrassEosTable &eos_table,
                    const tov::TabulatedEOS &slice_eos, Point *out) const;

  // Rho-only variant of Interpolate(), for the magnetic-web vector-potential builder
  // (grass_magnetic_web.hpp / dyngr_grass.cpp's BuildMagneticField), which needs
  // h(rho0) at far more points than the ADM/hydro fill loop. Skips the metric/
  // Jacobian/velocity construction, evaluating only {energy, enthalpy}. Shares
  // stencil-location logic with Interpolate() via LocateStencil() below. Returns rho0
  // in AthenaK code units (0.0 outside the star), same convention as Point::rho0.
  Real InterpolateRho(Real x, Real y, Real z, const GrassEosTable &eos_table) const;

 private:
  static constexpr int n_order = 3;         // Lagrange stencil half-width, matches
                                             // the sibling rns_st_reader.hpp's choice
  static constexpr int kNumFields = 10;
  enum FieldIdx {
    F_ALPHA = 0, F_GAMA, F_RHO, F_WW, F_PRESSURE,
    F_ENERGY, F_ENTHALPY, F_VELOCITY_SQ, F_OMG, F_SPHI
  };

  GrassUnits units_;

  int sdiv_ = 0, mdiv_ = 0, s_pwr_ = 1;
  Real r_e_internal_ = 0.0;       // GRASS-internal units (header_meta[0])
  Real r_e_code_ = 0.0;           // AthenaK code units

  // Flat storage, ghost-padded: s in [-n_order,sdiv_-1], mu in [-n_order,mdiv_+n_order-1]
  // Ghost is also needed at the mu=0/equator end, despite the query itself (|z|/r)
  // never going negative: the Lagrange stencil spans [im-n_order, im+n_order] around
  // the nearest grid index `im`, and GRASS's mu grid clusters points near the equator,
  // so `im < n_order` is routine there. Padded by even reflection (physical fields are
  // even functions of cos(theta)), same as the pole (mu=1) ghost below.
  std::vector<Real> s_gp_;        // size sdiv_ + n_order
  std::vector<Real> mu_;          // size mdiv_ + 2*n_order
  std::vector<Real> field_;       // size kNumFields * (sdiv_+n_order) * (mdiv_+2*n_order)

  int SOff() const { return n_order; }               // index of s_gp[0] in s_gp_
  int MOff() const { return n_order; }               // index of mu[0] in mu_
  int SSpan() const { return sdiv_ + n_order; }
  int MSpan() const { return mdiv_ + 2*n_order; }

  Real &S(int i) { return s_gp_[i + SOff()]; }
  Real S(int i) const { return s_gp_[i + SOff()]; }
  Real &Mu(int j) { return mu_[j + MOff()]; }
  Real Mu(int j) const { return mu_[j + MOff()]; }

  int FOff(int field, int i, int j) const {
    return field*SSpan()*MSpan() + (i + SOff())*MSpan() + (j + MOff());
  }
  Real &Field(int field, int i, int j) { return field_[FOff(field, i, j)]; }
  Real Field(int field, int i, int j) const { return field_[FOff(field, i, j)]; }

  void Load(const std::string &fname);
  void BuildLengthScales() { r_e_code_ = units_.LengthToCode(r_e_internal_); }

  // Locates the Lagrange-stencil base indices (is,im) for a query point, i.e.
  // the exact index-clamping logic shared by Interpolate() and
  // InterpolateRho() (see the latter's comment for why this is factored out
  // rather than duplicated).
  void LocateStencil(Real s_query, Real mu_query, int *is_out, int *im_out) const {
    int is = 0;
    for (int ii = 1; ii < sdiv_; ++ii) { if (S(ii) > s_query) { is = ii; break; } is = ii; }
    is = Kokkos::min(sdiv_ - 1 - n_order, Kokkos::max(n_order, is));
    int im = 0;
    for (int jj = 1; jj < mdiv_; ++jj) { if (Mu(jj) > mu_query) { im = jj; break; } im = jj; }
    im = Kokkos::min(mdiv_ - 1 - n_order, Kokkos::max(0, im));
    *is_out = is;
    *im_out = im;
  }

  // r(s) = r_e * (s/(1-s))^s_pwr, GRASS-internal units.
  Real RadiusOfS(Real s) const {
    Real x = s / Kokkos::max(1.0 - s, 1.0e-300);
    return r_e_internal_ * Kokkos::pow(x, static_cast<Real>(s_pwr_));
  }
  // s(r), inverse of the above.
  Real SOfRadius(Real r) const {
    Real x = Kokkos::pow(Kokkos::max(r, 0.0) / Kokkos::max(r_e_internal_, 1.0e-300),
                      1.0 / static_cast<Real>(s_pwr_));
    return x / (1.0 + x);
  }
  // ds/dr = s(1-s)/(s_pwr*r), GRASS-internal units.
  Real DsDr(Real s, Real r) const {
    return s*(1.0 - s) / (static_cast<Real>(s_pwr_) * Kokkos::max(r, 1.0e-300));
  }
};

//----------------------------------------------------------------------------------------
//! \fn void GrassData::Load
//! \brief Reads the restart binary per the exact layout documented in the file header.

inline void GrassData::Load(const std::string &fname) {
  std::ifstream in(fname, std::ios::binary);
  if (!in.is_open()) {
    std::stringstream msg;
    msg << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Could not open GRASS restart file '" << fname << "'" << std::endl;
    throw std::runtime_error(msg.str());
  }

  char magic[10];
  in.read(magic, 10);
  if (!in || std::string(magic, 10) != "GRASSRST01") {
    std::stringstream msg;
    msg << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "'" << fname << "' is not a valid GRASS restart file (bad magic)"
        << std::endl;
    throw std::runtime_error(msg.str());
  }

  int32_t header_ints[6];
  in.read(reinterpret_cast<char*>(header_ints), sizeof(header_ints));
  if (!in || header_ints[0] != 1 || header_ints[1] != 64 || header_ints[2] != kNumFields
      || header_ints[3] < 2 || header_ints[4] < 2) {
    std::stringstream msg;
    msg << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Unexpected GRASS restart header in '" << fname << "' (format_version="
        << header_ints[0] << " storage_size=" << header_ints[1] << " field_count="
        << header_ints[2] << " SDIV=" << header_ints[3] << " MDIV=" << header_ints[4]
        << ")" << std::endl;
    throw std::runtime_error(msg.str());
  }
  sdiv_ = header_ints[3];
  mdiv_ = header_ints[4];
  s_pwr_ = header_ints[5];

  double header_meta[5];
  in.read(reinterpret_cast<char*>(header_meta), sizeof(header_meta));
  r_e_internal_ = header_meta[0];
  // header_meta[1]=e_center, [2]=r_ratio, [3]=Omega_e, [4]=Omega_c -- not needed by
  // the interpolator itself (all recoverable from the field arrays directly at query
  // time), kept undocumented-but-available should future diagnostics want them.

  std::vector<double> s_raw(sdiv_), mu_raw(mdiv_);
  in.read(reinterpret_cast<char*>(s_raw.data()), sdiv_*sizeof(double));
  in.read(reinterpret_cast<char*>(mu_raw.data()), mdiv_*sizeof(double));

  std::vector<double> raw(static_cast<size_t>(kNumFields)*sdiv_*mdiv_);
  in.read(reinterpret_cast<char*>(raw.data()), raw.size()*sizeof(double));
  if (!in) {
    std::stringstream msg;
    msg << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "IO error while reading field data from '" << fname << "'" << std::endl;
    throw std::runtime_error(msg.str());
  }

  // sphi sanity check: GRASS's restart carries a generic scalar-field slot used by a
  // different (scalar-tensor/boson-star) solver mode; this pgen targets plain GR only.
  Real sphi_max = 0.0;
  auto RawIdx = [&](int field, int i, int j) -> size_t {
    // Fortran column-major, field fastest-varying: restart_data(field+1, i+1, j+1).
    return static_cast<size_t>(field) + static_cast<size_t>(kNumFields)*
           (static_cast<size_t>(i) + static_cast<size_t>(sdiv_)*static_cast<size_t>(j));
  };
  for (int i = 0; i < sdiv_; ++i) {
    for (int j = 0; j < mdiv_; ++j) {
      sphi_max = Kokkos::max(sphi_max, Kokkos::abs(raw[RawIdx(F_SPHI, i, j)]));
    }
  }
  if (sphi_max > 1.0e-10) {
    std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "GRASS restart '" << fname << "' has a nonzero scalar field "
              << "(max|sphi|=" << sphi_max << ") -- this pgen ignores it (plain-GR "
              << "reader only); the resulting initial data will be missing that "
              << "physics if the restart was meant for a scalarized run." << std::endl;
  }

  // Allocate ghost-padded storage and copy the real (non-ghost) data in.
  s_gp_.assign(SSpan(), 0.0);
  mu_.assign(mdiv_ + 2*n_order, 0.0);
  field_.assign(static_cast<size_t>(kNumFields)*SSpan()*MSpan(), 0.0);
  for (int i = 0; i < sdiv_; ++i) { S(i) = s_raw[i]; }
  for (int j = 0; j < mdiv_; ++j) { Mu(j) = mu_raw[j]; }
  for (int f = 0; f < kNumFields; ++f) {
    for (int i = 0; i < sdiv_; ++i) {
      for (int j = 0; j < mdiv_; ++j) {
        Field(f, i, j) = raw[RawIdx(f, i, j)];
      }
    }
  }

  // Radial ghost padding: mirror through the center (s<0 <-> s>0, same angular index),
  // valid asymptotically close to r=0 where it's used.
  for (int i = 1; i <= n_order; ++i) {
    S(-i) = -S(i);
    for (int j = 0; j < mdiv_; ++j) {
      for (int f = 0; f < kNumFields; ++f) { Field(f, -i, j) = Field(f, i, j); }
    }
  }
  // Equatorial ghost padding: mirror through mu=0, even reflection (see the
  // class-level storage comment above for why this is needed).
  for (int j = 1; j <= n_order; ++j) {
    Mu(-j) = 2.0*Mu(0) - Mu(j);
    for (int i = -n_order; i < sdiv_; ++i) {
      for (int f = 0; f < kNumFields; ++f) {
        Field(f, i, -j) = Field(f, i, j);
      }
    }
  }
  // Polar ghost padding: mirror through mu=1, even reflection (fields are regular
  // and don't flip sign at the pole in this ansatz).
  for (int j = 1; j <= n_order; ++j) {
    Mu(mdiv_ - 1 + j) = 2.0*Mu(mdiv_ - 1) - Mu(mdiv_ - 1 - j);
    for (int i = -n_order; i < sdiv_; ++i) {
      for (int f = 0; f < kNumFields; ++f) {
        Field(f, i, mdiv_ - 1 + j) = Field(f, i, mdiv_ - 1 - j);
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void GrassData::Interpolate
//! \brief Cartesian point -> AthenaK-code-unit ADM + hydro-primitive Point.

inline void GrassData::Interpolate(Real x, Real y, Real z, const GrassEosTable &eos_table,
                                   const tov::TabulatedEOS &slice_eos, Point *out) const {
  Real r_code = Kokkos::sqrt(x*x + y*y + z*z);
  Real varpi = Kokkos::sqrt(x*x + y*y);
  Real varpi_safe = Kokkos::max(varpi, 1.0e-30*Kokkos::max(r_code, 1.0));

  Real costh, sinth;
  if (r_code < 1.0e-30) {
    costh = 1.0; sinth = 0.0;
  } else {
    costh = z / r_code;
    sinth = varpi / r_code;
  }
  Real cosph = x / varpi_safe, sinph = y / varpi_safe;
  Real zsign = (z >= 0.0) ? 1.0 : -1.0;
  Real mu_query = Kokkos::abs(costh);   // theta' = arccos(|cos theta|), always in [0,1]

  Real r_internal = r_code / Kokkos::max(units_.LengthToCode(1.0), 1.0e-300);
  Real s_query = SOfRadius(r_internal);

  int is, im;
  LocateStencil(s_query, mu_query, &is, &im);

  // Point values of the 8 fields we need directly, plus d/ds and d/dmu of ww (F_WW).
  // F_ENTHALPY is GRASS's own inside-star/vacuum indicator, needed to gate energy/
  // pressure before they're used (the stencil can overshoot near the surface).
  Real val[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  Real dww_ds = 0.0, dww_dmu = 0.0;
  static constexpr int kUse[8] = {F_ALPHA, F_GAMA, F_RHO, F_WW, F_PRESSURE,
                                   F_ENERGY, F_OMG, F_ENTHALPY};

  for (int di = -n_order; di <= n_order; ++di) {
    for (int dj = -n_order; dj <= n_order; ++dj) {
      Real fr = 1.0, fp = 1.0, fdr = 0.0, fdp = 0.0;
      for (int dk = -n_order; dk <= n_order; ++dk) {
        if (dk == di) { continue; }
        fr *= (s_query - S(is + dk)) / (S(is + di) - S(is + dk));
      }
      for (int dk = -n_order; dk <= n_order; ++dk) {
        if (dk == dj) { continue; }
        fp *= (mu_query - Mu(im + dk)) / (Mu(im + dj) - Mu(im + dk));
      }
      for (int dk = -n_order; dk <= n_order; ++dk) {
        if (dk == di) { continue; }
        Real f0 = 1.0 / (S(is + di) - S(is + dk));
        for (int dl = -n_order; dl <= n_order; ++dl) {
          if (dl == dk || dl == di) { continue; }
          f0 *= (s_query - S(is + dl)) / (S(is + di) - S(is + dl));
        }
        fdr += f0;
      }
      for (int dk = -n_order; dk <= n_order; ++dk) {
        if (dk == dj) { continue; }
        Real f0 = 1.0 / (Mu(im + dj) - Mu(im + dk));
        for (int dl = -n_order; dl <= n_order; ++dl) {
          if (dl == dk || dl == dj) { continue; }
          f0 *= (mu_query - Mu(im + dl)) / (Mu(im + dj) - Mu(im + dl));
        }
        fdp += f0;
      }
      for (int c = 0; c < 8; ++c) {
        val[c] += fr*fp*Field(kUse[c], is + di, im + dj);
      }
      Real ww_here = Field(F_WW, is + di, im + dj);
      dww_ds += fdr*fp*ww_here;
      dww_dmu += fr*fdp*ww_here;
    }
  }
  Real alpha_pot = val[0], gama_pot = val[1], rho_pot = val[2], ww = val[3];
  Real pressure_internal = val[4], energy_internal = val[5], omg = val[6];
  Real enthalpy_pot = val[7];
  // GRASS's restart `enthalpy` field is log(h) (h = dimensionless specific enthalpy,
  // h=1 at the vacuum surface), so the inside-star criterion is log(h)>~0, not h>1.
  bool inside_star = (enthalpy_pot > 1.0e-6);
  if (!inside_star) {
    pressure_internal = 0.0;
    energy_internal = 0.0;
  }

  // ---- Convert to AthenaK code units up front -- everything below is code units ---
  Real len_conv = units_.LengthToCode(1.0);   // code units per GRASS-internal length
  Real rate_conv = units_.RateToCode(1.0);    // code units per GRASS-internal rate
  Real ww_code = ww * rate_conv;
  Real omg_code = omg * rate_conv;
  Real ds_dr_internal = DsDr(s_query, r_internal);      // per unit r_internal
  Real ds_dr_code = ds_dr_internal / len_conv;          // per unit r_code (chain rule)
  Real dww_dr_code = (dww_ds * ds_dr_code) * rate_conv; // d(ww_code)/d(r_code)
  Real dww_dtheta_code = zsign * (-sinth * dww_dmu) * rate_conv;  // theta dimensionless

  // ---- ADM quantities, spherical-polar, AthenaK code units ------------------------
  // alpha_pot/gama_pot/rho_pot are dimensionless RNS potentials, unit-system-invariant.
  Real N = Kokkos::exp(0.5*(gama_pot + rho_pot));
  Real gam_rr = Kokkos::exp(2.0*alpha_pot);
  Real gam_thth = gam_rr * r_code*r_code;
  Real gam_phph = Kokkos::exp(gama_pot - rho_pot) * r_code*r_code * sinth*sinth;
  Real K_rphi = -(gam_phph/(2.0*N)) * dww_dr_code;
  Real K_thphi = -(gam_phph/(2.0*N)) * dww_dtheta_code;

  // ---- Spherical -> Cartesian Jacobian (on-axis-safe) -----------------------------
  Real r_safe = Kokkos::max(r_code, 1.0e-30);
  Real dr_dx = x/r_safe, dr_dy = y/r_safe, dr_dz = z/r_safe;
  Real dth_dx = costh*cosph/r_safe, dth_dy = costh*sinph/r_safe, dth_dz = -sinth/r_safe;
  Real dph_dx = -sinph/varpi_safe, dph_dy = cosph/varpi_safe, dph_dz = 0.0;

  Real dr[3]  = {dr_dx, dr_dy, dr_dz};
  Real dth[3] = {dth_dx, dth_dy, dth_dz};
  Real dph[3] = {dph_dx, dph_dy, dph_dz};

  Real g_dd_sph[3][3], K_dd_sph[3][3];
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      g_dd_sph[a][b] = gam_rr*dr[a]*dr[b] + gam_thth*dth[a]*dth[b]
                       + gam_phph*dph[a]*dph[b];
      K_dd_sph[a][b] = K_rphi*(dr[a]*dph[b] + dph[a]*dr[b])
                       + K_thphi*(dth[a]*dph[b] + dph[a]*dth[b]);
    }
  }
  Real beta_phi_up = -ww_code;   // beta^phi, code-rate units
  Real beta_up[3] = { y*beta_phi_up, -x*beta_phi_up, 0.0 };  // d(x,y,z)/dphi = (-y,x,0)

  // ---- Fluid velocity (ZAMO-frame azimuthal, AthenaK contravariant convention) ---
  // v (dimensionless, units of c) is unit-system-invariant, so built directly from
  // the GRASS-internal quantities rather than the code-unit ones above.
  Real vu[3] = {0.0, 0.0, 0.0};
  Real rho0_cgs = 0.0;
  if (inside_star) {
    Real v2 = Kokkos::pow((omg - ww)*r_internal*sinth*Kokkos::exp(-rho_pot), 2);
    v2 = Kokkos::min(Kokkos::max(v2, 0.0), 1.0 - 1.0e-12);
    Real utilde_phi = (omg_code - ww_code) / (N * Kokkos::sqrt(1.0 - v2));
    vu[0] = y*utilde_phi; vu[1] = -x*utilde_phi; vu[2] = 0.0;
    // Floor before the EOS-table log() lookup: near the surface the Lagrange stencil
    // can ring across the density discontinuity and undershoot energy to <=0 even
    // though `inside_star` (a separate enthalpy threshold) is true.
    energy_internal = Kokkos::max(energy_internal, 1.0e-300);
    rho0_cgs = eos_table.N0FromE(energy_internal) * GrassUnits::kGrassMB;  // g/cm^3
  }

  out->alpha = N;   // lapse is dimensionless, no unit conversion
  for (int a = 0; a < 3; ++a) {
    out->beta_u[a] = beta_up[a];
    out->vu[a] = vu[a];
  }
  out->g_dd[0] = g_dd_sph[0][0];  // xx
  out->g_dd[1] = g_dd_sph[0][1];  // xy
  out->g_dd[2] = g_dd_sph[0][2];  // xz
  out->g_dd[3] = g_dd_sph[1][1];  // yy
  out->g_dd[4] = g_dd_sph[1][2];  // yz
  out->g_dd[5] = g_dd_sph[2][2];  // zz
  out->K_dd[0] = K_dd_sph[0][0];
  out->K_dd[1] = K_dd_sph[0][1];
  out->K_dd[2] = K_dd_sph[0][2];
  out->K_dd[3] = K_dd_sph[1][1];
  out->K_dd[4] = K_dd_sph[1][2];
  out->K_dd[5] = K_dd_sph[2][2];

  out->rho0 = units_.RestMassDensityCgsToCode(rho0_cgs);
  out->pres = units_.PressureToCode(pressure_internal);

  // Check finiteness before GetYeFromRho() below: its own lrho<lrho_min guard doesn't
  // catch NaN (compares false against everything), so a NaN rho0 would otherwise reach
  // an undefined int-cast further inside TabulatedEOS's bisection.
  if (!Kokkos::isfinite(out->alpha) || !Kokkos::isfinite(out->rho0) ||
      !Kokkos::isfinite(out->pres)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "NaN/Inf in GRASS interpolated data at (x,y,z)=(" << x << "," << y << ","
        << z << ")" << std::endl;
    throw std::runtime_error(msg.str());
  }

  // GetYeFromRho degrades gracefully to its own atmosphere default below the slice
  // table's floor, so no separate inside_star gate is needed here.
  out->Yq = slice_eos.GetYeFromRho<tov::LocationTag::Host>(out->rho0);
}

//----------------------------------------------------------------------------------------
//! \fn Real GrassData::InterpolateRho
//! \brief Cartesian point -> rest-mass density only, AthenaK code units. See the
//  declaration's comment for why this exists separately from Interpolate().

inline Real GrassData::InterpolateRho(Real x, Real y, Real z,
                                       const GrassEosTable &eos_table) const {
  Real r_code = Kokkos::sqrt(x*x + y*y + z*z);
  Real costh = (r_code < 1.0e-30) ? 1.0 : z / r_code;
  Real mu_query = Kokkos::abs(costh);

  Real r_internal = r_code / Kokkos::max(units_.LengthToCode(1.0), 1.0e-300);
  Real s_query = SOfRadius(r_internal);

  int is, im;
  LocateStencil(s_query, mu_query, &is, &im);

  // Same {energy, enthalpy} fields as Interpolate(); no derivatives needed here, so
  // fr/fp are hoisted out of the inner loop instead of recomputed per (di,dj).
  Real val[2] = {0.0, 0.0};   // {energy, enthalpy}
  static constexpr int kUse[2] = {F_ENERGY, F_ENTHALPY};
  for (int di = -n_order; di <= n_order; ++di) {
    Real fr = 1.0;
    for (int dk = -n_order; dk <= n_order; ++dk) {
      if (dk == di) { continue; }
      fr *= (s_query - S(is + dk)) / (S(is + di) - S(is + dk));
    }
    for (int dj = -n_order; dj <= n_order; ++dj) {
      Real fp = 1.0;
      for (int dk = -n_order; dk <= n_order; ++dk) {
        if (dk == dj) { continue; }
        fp *= (mu_query - Mu(im + dk)) / (Mu(im + dj) - Mu(im + dk));
      }
      for (int c = 0; c < 2; ++c) {
        val[c] += fr*fp*Field(kUse[c], is + di, im + dj);
      }
    }
  }
  Real energy_internal = val[0], enthalpy_pot = val[1];
  bool inside_star = (enthalpy_pot > 1.0e-6);   // same criterion as Interpolate()
  if (!inside_star) { return 0.0; }

  energy_internal = Kokkos::max(energy_internal, 1.0e-300);  // same floor as Interpolate
  Real rho0_cgs = eos_table.N0FromE(energy_internal) * GrassUnits::kGrassMB;
  return units_.RestMassDensityCgsToCode(rho0_cgs);
}

}  // namespace grass

#endif  // PGEN_DYN_GRMHD_RNS_GRASS_READER_HPP_
