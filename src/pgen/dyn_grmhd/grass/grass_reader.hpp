#ifndef PGEN_DYN_GRMHD_RNS_GRASS_READER_HPP_
#define PGEN_DYN_GRMHD_RNS_GRASS_READER_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file grass_reader.hpp
//  \brief Host-side reader/interpolator for GRASS (RNS-family rotating-NS equilibrium
//  code) restart binaries (Res/res.rst). GRASS uses the classic RNS/Cook-Shapiro-
//  Teukolsky (CST) stationary-axisymmetric metric in spherical-polar coordinates:
//
//    ds^2 = -e^{gama+rho} dt^2 + e^{2alpha}(dr^2 + r^2 dtheta^2)
//           + e^{gama-rho} r^2 sin^2(theta) (dphi - ww dt)^2
//
//  This is directly ADM-physical (no conformal decomposition needed, unlike the
//  sibling rns_st_reader.hpp for scalar-tensor SACRA data). Reading off the ADM split
//  and deriving the extrinsic curvature for this stationary metric (verified this
//  session against AthenaK's own Z4c RHS sign convention, dt(gamma_ij) =
//  -2*alpha*K_ij + D_i(beta_j) + D_j(beta_i)):
//
//    N       = e^{(gama+rho)/2}                    (the frame-dragging term in g_tt
//                                                    cancels exactly against beta_i*beta^i)
//    beta^phi = -ww,  beta^r = beta^theta = 0
//    gamma_rr = e^{2alpha}, gamma_thth = e^{2alpha} r^2, gamma_phph = e^{gama-rho} r^2 sin^2(theta)
//    K_rphi   = -(gamma_phph/(2N)) * dr(ww)
//    K_thphi  = -(gamma_phph/(2N)) * dth(ww)        (all other K_ij = 0)
//
//  Only ww's derivatives are needed anywhere in this construction -- alpha, gama, rho
//  are needed only as point values. Fluid velocity (ZAMO-frame azimuthal, matching
//  GRASS's own `velocity_sq` field exactly -- see the derivation in the investigation
//  notes/plan) gives AthenaK's primitive convention u-tilde^i = gamma^ij u_j directly:
//
//    u-tilde^phi = (omg - ww) / (N*sqrt(1 - v^2)),   v = (omg-ww)*r*sin(theta)*e^{-rho}
//
//  Restart binary layout (access='stream', no Fortran record padding, all 8-byte
//  doubles -- confirmed against /u/tlam/GRASS/src/theory/spin_integration_mod.f90's
//  write_restart_file and /u/tlam/GRASS/src/core/regrid_mod.f90's read_binary_restart):
//
//    char[10]   magic = "GRASSRST01"
//    int32[6]   header_ints  = [format_version=1, storage_size=64, field_count=10,
//                                SDIV, MDIV, s_pwr]
//    double[5]  header_meta  = [r_e, e_center, r_ratio, Omega_e, Omega_c]  (GRASS-internal units)
//    double[SDIV]        s_gp
//    double[MDIV]        mu           (= cos(theta), mu[0]=0 equator -> mu[MDIV-1]=1 pole)
//    double[10][SDIV][MDIV]  restart_data, field index fastest-varying (Fortran
//                            column-major), field order:
//                            alpha,gama,rho,ww,pressure,energy,enthalpy,velocity_sq,omg,sphi
//
//  Physical radius reconstruction: r_internal(s) = r_e * (s/(1-s))^s_pwr (GRASS's own
//  compactification, confirmed in analysis_mod.f90/sphere_mod.f90/spin_integration_mod.f90);
//  ds/dr = s*(1-s)/(s_pwr*r), used to chain-rule the Lagrange-basis d/ds(ww) into dr(ww).
//
//  Grid is a half-domain in theta (mu=cos(theta) in [0,1], equator->pole -- confirmed
//  via grid_mod.f90 for all three of GRASS's angular-collocation options). Reused
//  directly from the sibling rns_st_reader.hpp's own existing idiom for its own
//  half-domain angular array: query at mu'=|cos(theta)|=|z|/r (always in-bounds --
//  the QUERY itself never goes negative), and flip the sign of the theta-derivative-
//  dependent extrinsic-curvature components for z<0 (chain rule:
//  theta'=arccos(|cos(theta)|), dtheta'/dtheta=-1 for z<0). Note this does NOT mean
//  the Lagrange stencil itself stays in-bounds near the equator -- it still needs
//  ghost-padded storage at negative angular indices whenever the nearest real grid
//  index is within n_order of mu=0 (routine, since GRASS's mu grid clusters points
//  near the equator); see the equatorial mirror in Load() below.

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
    Real alpha;        // ADM lapse N (AthenaK naming clash with GRASS's own `alpha`
                        // metric potential -- this is the LAPSE, not GRASS's alpha)
    Real beta_u[3];    // shift, Cartesian
    Real g_dd[6];      // xx,xy,xz,yy,yz,zz
    Real K_dd[6];      // xx,xy,xz,yy,yz,zz
    Real rho0;         // rest-mass density, AthenaK code units
    Real pres;         // pressure, AthenaK code units
    Real vu[3];        // u-tilde^i = gamma^ij u_j, Cartesian (AthenaK primitive convention)
    Real Yq;           // electron/charge fraction, seeded from the DD2_hot_slice 1D table
                        // (Yl(nb) along the SAME trajectory that produced GRASS's own
                        // e(n0),p(n0) -- see grass/grass_units.hpp file header and
                        // build_dd2_hot_slice_1d_athtab.py). Not load-bearing for correct
                        // initial data (PrimToCons recovers T internally from the runtime
                        // 3D EOS regardless) -- this only seeds the passive scalar Y[e]
                        // consistently with the trajectory the star was actually built on.
  };

  GrassData(const std::string &fname, const GrassUnits &units) : units_(units) {
    Load(fname);
    BuildLengthScales();
  }

  void Interpolate(Real x, Real y, Real z, const GrassEosTable &eos_table,
                    const tov::TabulatedEOS &slice_eos, Point *out) const;

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

  // Flat storage, ghost-padded: s in [-n_order, sdiv_-1], mu in [-n_order, mdiv_-1+n_order].
  // Ghost IS needed at the mu=0/equator end, despite the query itself (|z|/r) never
  // going negative: the Lagrange stencil is centered on the nearest grid index `im`
  // and always spans [im-n_order, im+n_order] regardless of where the query sits, so
  // any query with `im < n_order` (routine near the equator, since GRASS's mu grid
  // clusters points there -- confirmed this session: mu[1..3] ~ 0.002-0.016 for a
  // representative restart) needs negative-index storage. Originally missing here,
  // causing silent reads into adjacent/out-of-bounds memory (NaN/Inf propagating into
  // rho0, then into tov::TabulatedEOS's bisection -- a Kokkos::View OOB abort) for any
  // query point close enough to the equatorial plane. Fixed the same way as the
  // pre-existing pole (mu=1) ghost below: even reflection, since physical fields are
  // even functions of cos(theta) and this is exactly the mu=0 counterpart of that
  // same symmetry.
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

  // r(s) = r_e * (s/(1-s))^s_pwr, GRASS-internal units.
  Real RadiusOfS(Real s) const {
    Real x = s / std::max(1.0 - s, 1.0e-300);
    return r_e_internal_ * std::pow(x, static_cast<Real>(s_pwr_));
  }
  // s(r), inverse of the above.
  Real SOfRadius(Real r) const {
    Real x = std::pow(std::max(r, 0.0) / std::max(r_e_internal_, 1.0e-300),
                      1.0 / static_cast<Real>(s_pwr_));
    return x / (1.0 + x);
  }
  // ds/dr = s(1-s)/(s_pwr*r), GRASS-internal units.
  Real DsDr(Real s, Real r) const {
    return s*(1.0 - s) / (static_cast<Real>(s_pwr_) * std::max(r, 1.0e-300));
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
      sphi_max = std::max(sphi_max, std::abs(raw[RawIdx(F_SPHI, i, j)]));
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

  // Radial ghost padding: mirror through the center (s<0 <-> s>0, same angular index
  // -- an approximation valid asymptotically close to r=0, exactly where it is used;
  // same idiom as the sibling rns_st_reader.hpp's own S(1-i)=-S(1+i) treatment).
  for (int i = 1; i <= n_order; ++i) {
    S(-i) = -S(i);
    for (int j = 0; j < mdiv_; ++j) {
      for (int f = 0; f < kNumFields; ++f) { Field(f, -i, j) = Field(f, i, j); }
    }
  }
  // Equatorial ghost padding: mirror through the equator (mu=0 boundary), even
  // reflection -- same reasoning/idiom as the pole block just below, just at the
  // other end (see the class-level storage comment for why this is needed at all).
  for (int j = 1; j <= n_order; ++j) {
    Mu(-j) = 2.0*Mu(0) - Mu(j);
    for (int i = -n_order; i < sdiv_; ++i) {
      for (int f = 0; f < kNumFields; ++f) {
        Field(f, i, -j) = Field(f, i, j);
      }
    }
  }
  // Polar ghost padding: mirror through the pole (mu=1 boundary), even reflection --
  // physical fields in this axisymmetric, equatorially-and-polar-regular ansatz do not
  // flip sign there (mirrors rns_st_reader.hpp's own pole treatment).
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
  Real r_code = std::sqrt(x*x + y*y + z*z);
  Real varpi = std::sqrt(x*x + y*y);
  Real varpi_safe = std::max(varpi, 1.0e-30*std::max(r_code, 1.0));

  Real costh, sinth;
  if (r_code < 1.0e-30) {
    costh = 1.0; sinth = 0.0;
  } else {
    costh = z / r_code;
    sinth = varpi / r_code;
  }
  Real cosph = x / varpi_safe, sinph = y / varpi_safe;
  Real zsign = (z >= 0.0) ? 1.0 : -1.0;
  Real mu_query = std::abs(costh);   // theta' = arccos(|cos theta|), always in [0,1]

  Real r_internal = r_code / std::max(units_.LengthToCode(1.0), 1.0e-300);
  Real s_query = SOfRadius(r_internal);

  // Locate and clamp Lagrange-stencil base indices so [base-n_order, base+n_order]
  // stays within the (ghost-padded) stored arrays. Linear scan is adequate at GRASS's
  // grid sizes (SDIV~O(1e3)); replace with a binary search if profiling ever demands it.
  int is = 0;
  for (int ii = 1; ii < sdiv_; ++ii) { if (S(ii) > s_query) { is = ii; break; } is = ii; }
  is = std::min(sdiv_ - 1 - n_order, std::max(n_order, is));
  int im = 0;
  for (int jj = 1; jj < mdiv_; ++jj) { if (Mu(jj) > mu_query) { im = jj; break; } im = jj; }
  im = std::min(mdiv_ - 1 - n_order, std::max(0, im));

  // Point values of the 8 fields we need directly, plus d/ds and d/dmu of ww (F_WW).
  // F_ENTHALPY is included solely as GRASS's own inside-star/vacuum indicator (h>1
  // inside, h<=1 in vacuum -- exactly the check GRASS's own exporter_mod.f90 uses
  // before calling n0_at_e; skipping it risks calling log() on a near-zero or
  // slightly-negative interpolated `energy` just outside the stellar surface, where
  // the Lagrange stencil can overshoot across the density discontinuity).
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
  // Inside-star indicator. GRASS's restart `enthalpy` field is log(h) (h = GRASS's
  // dimensionless specific enthalpy, h=1 at the vacuum surface) -- confirmed directly
  // against a real restart binary this session: enthalpy(center)=0.287 matches
  // GRASS's own printed "h_c" exactly, and enthalpy(exterior)~1e-9 matches the
  // printed "log(h_min)" threshold from GRASS's EOS-table load. So the correct
  // criterion is log(h)>~0 (any small positive constant well below the O(0.1-1)
  // interior values and well above the ~1e-9 exterior floor works safely), NOT h>1.
  bool inside_star = (enthalpy_pot > 1.0e-6);
  if (!inside_star) {
    pressure_internal = 0.0;
    energy_internal = 0.0;
  }

  // ---- Convert to AthenaK code units up front, BEFORE building any derived tensor
  // (deliberately avoids the error-prone alternative of building the metric/curvature
  // in GRASS-internal units and rescaling the assembled tensors afterward -- every
  // quantity below is in AthenaK code units from this point on).
  Real len_conv = units_.LengthToCode(1.0);    // code-length-units per GRASS-internal-length
  Real rate_conv = units_.RateToCode(1.0);     // code-rate-units per GRASS-internal-rate
  Real ww_code = ww * rate_conv;
  Real omg_code = omg * rate_conv;
  Real ds_dr_internal = DsDr(s_query, r_internal);      // per unit r_internal
  Real ds_dr_code = ds_dr_internal / len_conv;          // per unit r_code (chain rule)
  Real dww_dr_code = (dww_ds * ds_dr_code) * rate_conv; // d(ww_code)/d(r_code)
  Real dww_dtheta_code = zsign * (-sinth * dww_dmu) * rate_conv;  // theta is dimensionless

  // ---- ADM quantities, spherical-polar, AthenaK code units ------------------------
  // r_code (already in code units, the Cartesian-derived radius) is used directly for
  // every r-dependent metric potential below -- alpha_pot/gama_pot/rho_pot themselves
  // are dimensionless RNS potentials, unaffected by the unit system.
  Real N = std::exp(0.5*(gama_pot + rho_pot));
  Real gam_rr = std::exp(2.0*alpha_pot);
  Real gam_thth = gam_rr * r_code*r_code;
  Real gam_phph = std::exp(gama_pot - rho_pot) * r_code*r_code * sinth*sinth;
  Real K_rphi = -(gam_phph/(2.0*N)) * dww_dr_code;
  Real K_thphi = -(gam_phph/(2.0*N)) * dww_dtheta_code;

  // ---- Spherical -> Cartesian Jacobian (on-axis-safe) -----------------------------
  Real r_safe = std::max(r_code, 1.0e-30);
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
  // v is dimensionless (a physical velocity in units of c), so the GRASS-internal
  // formula for v^2 needs r_internal (not r_code) even though omg/ww here are already
  // code-unit rates -- rebuild v^2 from the GRASS-internal quantities directly
  // (equivalent either way since v^2 is unit-system-invariant by construction; using
  // the internal quantities avoids re-deriving an r_code-consistent rate/length
  // cancellation).
  Real vu[3] = {0.0, 0.0, 0.0};
  Real rho0_cgs = 0.0;
  if (inside_star) {
    Real v2 = std::pow((omg - ww)*r_internal*sinth*std::exp(-rho_pot), 2);
    v2 = std::min(std::max(v2, 0.0), 1.0 - 1.0e-12);
    Real utilde_phi = (omg_code - ww_code) / (N * std::sqrt(1.0 - v2));
    vu[0] = y*utilde_phi; vu[1] = -x*utilde_phi; vu[2] = 0.0;
    // Floor energy_internal to a tiny positive value before the EOS-table log()
    // lookup. `inside_star` is based on the (separately-interpolated) enthalpy
    // field crossing its own threshold -- it does NOT guarantee energy_internal
    // itself is positive right at the stellar surface, where the Lagrange
    // stencil can ring (Gibbs phenomenon) across the sharp density
    // discontinuity and briefly undershoot to zero/negative just inside the
    // enthalpy-based boundary. N0FromE()'s std::log() of that is NaN, which
    // previously reached slice_eos.GetYeFromRho() below BEFORE the isfinite
    // guard could catch it, crashing inside TabulatedEOS's own bisection
    // (Kokkos::View OOB with an undefined int-cast-of-NaN index) instead of
    // this file's own controlled fatal error.
    energy_internal = std::max(energy_internal, 1.0e-300);
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

  // Check finiteness BEFORE calling into slice_eos.GetYeFromRho() below --
  // GetYeFromRho does its own log(rho) internally and only guards against
  // *small* rho (its own lrho<lrho_min check), not NaN (NaN compares false
  // against everything, so a NaN rho0 silently skips that guard and reaches
  // an undefined int-cast further in, previously surfacing as a Kokkos::View
  // out-of-bounds abort instead of this file's own controlled error).
  if (!std::isfinite(out->alpha) || !std::isfinite(out->rho0) ||
      !std::isfinite(out->pres)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "NaN/Inf in GRASS interpolated data at (x,y,z)=(" << x << "," << y << ","
        << z << ")" << std::endl;
    throw std::runtime_error(msg.str());
  }

  // Seed Yq from the 1D slice table at this point's OWN rho0 -- GetYeFromRho already
  // degrades gracefully to its own atmosphere default (ye_atmosphere, <mhd>
  // s0_atmosphere) whenever rho0 is zero/below the slice table's floor, so no separate
  // inside_star gate is needed here (out->rho0 is already 0 outside the star) -- and by
  // this point out->rho0 is guaranteed finite (see the isfinite check just above).
  out->Yq = slice_eos.GetYeFromRho<tov::LocationTag::Host>(out->rho0);
}

}  // namespace grass

#endif  // PGEN_DYN_GRMHD_RNS_GRASS_READER_HPP_
