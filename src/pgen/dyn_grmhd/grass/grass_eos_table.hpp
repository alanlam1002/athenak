#ifndef PGEN_DYN_GRMHD_RNS_GRASS_EOS_TABLE_HPP_
#define PGEN_DYN_GRMHD_RNS_GRASS_EOS_TABLE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file grass_eos_table.hpp
//  \brief Host-side reader for GRASS's own 4-column EOS table format
//  (num_tab header line, then rows "e[g/cm^3] p[dyn/cm^2] h n0[cm^-3]"), used ONLY to
//  recover rest-mass density from GRASS's restart-file `energy` field -- GRASS's
//  restart already carries a self-consistent (pressure, energy) pair from its own
//  equilibrium solve, so no other EOS call is needed for the initial data.
//
//  This is a direct, line-by-line port of GRASS's own /u/tlam/GRASS/src/core/eos_mod.f90
//  (loadEos, pchip_slopes, endpoint_slope, find_cell, hermite_eval, interp_eos/n0_at_e)
//  -- monotone cubic Hermite (Fritsch-Carlson) interpolation in log-log space, matching
//  GRASS's own scheme exactly rather than bridging into AthenaK's CompOSE-native,
//  rho-indexed TabulatedEOS (different column schema, different interpolation, would
//  reintroduce small inconsistencies at exactly the phase-transition/near-surface
//  regions PCHIP was chosen to handle cleanly in GRASS's own solve).

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "athena.hpp"
#include "grass_units.hpp"

namespace grass {

//----------------------------------------------------------------------------------------
//! \class GrassEosTable
//  \brief Loads one GRASS-format EOS table and provides log(e)->log(n0) monotone cubic
//  Hermite interpolation, i.e. a C++ port of eos_mod.f90's `n0_at_e`.

class GrassEosTable {
 public:
  explicit GrassEosTable(const std::string &fname, const GrassUnits &units) {
    Load(fname, units);
  }

  //! \brief GRASS-internal-units energy density in -> baryon number density out
  //  [cm^-3], mirroring eos_mod.f90's n0_at_e(ee) exactly (same log-space table, same
  //  cell finder, same Hermite evaluator). `e_internal` must be in the SAME
  //  GRASS-internal units as the restart file's raw `energy` field (no pre-conversion
  //  -- this loader applies the same C^2*KSCALE factor GRASS's own loadEos applies
  //  when building log_e_ from the raw cgs table file, so the two match).
  Real N0FromE(Real e_internal) const {
    Real log_in = std::log(e_internal);
    int i = FindCell(log_e_, log_in);
    Real log_out = HermiteEval(log_e_, log_n0_, slope_n0_of_e_, i, log_in);
    return std::exp(log_out);
  }

 private:
  std::vector<Real> log_e_, log_n0_;   // table nodes, log-space
  std::vector<Real> slope_n0_of_e_;    // d(log_n0)/d(log_e) at each node

  void Load(const std::string &fname, const GrassUnits &units) {
    std::ifstream in(fname);
    if (!in.is_open()) {
      std::stringstream msg;
      msg << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
          << "Could not open GRASS EOS table '" << fname << "'" << std::endl;
      throw std::runtime_error(msg.str());
    }
    int num_tab = 0;
    in >> num_tab;
    if (!in || num_tab < 2) {
      std::stringstream msg;
      msg << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
          << "Could not read a valid row count from GRASS EOS table '" << fname << "'"
          << std::endl;
      throw std::runtime_error(msg.str());
    }
    std::vector<Real> e_raw(num_tab), n0_raw(num_tab);
    for (int i = 0; i < num_tab; ++i) {
      Real e, p, h, n0;
      in >> e >> p >> h >> n0;
      if (!in) {
        std::stringstream msg;
        msg << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
            << "Malformed row " << i << " in GRASS EOS table '" << fname << "'"
            << std::endl;
        throw std::runtime_error(msg.str());
      }
      e_raw[i] = e;
      n0_raw[i] = n0;
    }

    // Mirror eos_mod.f90::loadEos exactly: log_e = log(e_cgs * C^2 * KSCALE) so that
    // this table's log_e_ lands in the SAME GRASS-internal units as the restart
    // file's raw `energy` field -- no separate conversion needed at lookup time.
    log_e_.resize(num_tab);
    log_n0_.resize(num_tab);
    for (int i = 0; i < num_tab; ++i) {
      Real e_internal = e_raw[i] * GrassUnits::kGrassC * GrassUnits::kGrassC
                         * GrassUnits::kKscale;
      log_e_[i] = std::log(e_internal);
      log_n0_[i] = std::log(n0_raw[i]);
    }

    slope_n0_of_e_.assign(num_tab, 0.0);
    PchipSlopes(log_e_, log_n0_, slope_n0_of_e_);
  }

  //! \brief Fritsch-Carlson monotone-cubic-Hermite slopes, port of eos_mod.f90's
  //  pchip_slopes (interior: weighted-harmonic-mean of adjacent secants, collapsing to
  //  zero when secants disagree in sign or either is zero; endpoints: 3-point
  //  one-sided formula with the same monotonicity clamp via EndpointSlope).
  static void PchipSlopes(const std::vector<Real> &x, const std::vector<Real> &y,
                          std::vector<Real> &m) {
    int n = static_cast<int>(x.size());
    if (n <= 1) { std::fill(m.begin(), m.end(), 0.0); return; }
    if (n == 2) {
      m[0] = (y[1] - y[0]) / (x[1] - x[0]);
      m[1] = m[0];
      return;
    }
    for (int i = 1; i < n - 1; ++i) {  // 0-based: Fortran i=2..n-1 -> i=1..n-2
      Real h_a = x[i] - x[i-1];
      Real h_b = x[i+1] - x[i];
      Real d_a = (y[i] - y[i-1]) / h_a;
      Real d_b = (y[i+1] - y[i]) / h_b;
      if (d_a * d_b <= 0.0) {
        m[i] = 0.0;
      } else {
        Real w1 = 2.0*h_b + h_a;
        Real w2 = h_b + 2.0*h_a;
        m[i] = (w1 + w2) / (w1/d_a + w2/d_b);
      }
    }
    m[0] = EndpointSlope(x[1]-x[0], x[2]-x[1],
                         (y[1]-y[0])/(x[1]-x[0]), (y[2]-y[1])/(x[2]-x[1]));
    m[n-1] = EndpointSlope(x[n-1]-x[n-2], x[n-2]-x[n-3],
                          (y[n-1]-y[n-2])/(x[n-1]-x[n-2]),
                          (y[n-2]-y[n-3])/(x[n-2]-x[n-3]));
  }

  static Real EndpointSlope(Real h_a, Real h_b, Real d_a, Real d_b) {
    Real m_e = ((2.0*h_a + h_b)*d_a - h_a*d_b) / (h_a + h_b);
    if (m_e * d_a <= 0.0) {
      m_e = 0.0;
    } else if (d_a * d_b < 0.0 && std::abs(m_e) > 3.0*std::abs(d_a)) {
      m_e = 3.0 * d_a;
    }
    return m_e;
  }

  //! \brief Binary-search cell finder, port of eos_mod.f90's find_cell (0-based here;
  //  Fortran's `i` such that x(i)<=xb<x(i+1) becomes C++ `i` such that
  //  x[i]<=xb<x[i+1], 0-based, valid for HermiteEval's x[i]/x[i+1] pair access).
  static int FindCell(const std::vector<Real> &x, Real xb) {
    int n = static_cast<int>(x.size());
    if (n <= 1) { return 0; }
    if (xb <= x[0]) { return 0; }
    if (xb >= x[n-1]) { return n - 2; }
    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
      int mid = (lo + hi) / 2;
      if (x[mid] <= xb) { lo = mid; } else { hi = mid; }
    }
    return lo;
  }

  //! \brief Cubic Hermite basis evaluation, port of eos_mod.f90's hermite_eval.
  static Real HermiteEval(const std::vector<Real> &x, const std::vector<Real> &y,
                          const std::vector<Real> &m, int i, Real xb) {
    Real h = x[i+1] - x[i];
    Real t = (xb - x[i]) / h;
    Real t2 = t*t, t3 = t2*t;
    Real h00 =  2.0*t3 - 3.0*t2 + 1.0;
    Real h10 =      t3 - 2.0*t2 + t;
    Real h01 = -2.0*t3 + 3.0*t2;
    Real h11 =      t3 -     t2;
    return h00*y[i] + h10*h*m[i] + h01*y[i+1] + h11*h*m[i+1];
  }
};

}  // namespace grass

#endif  // PGEN_DYN_GRMHD_RNS_GRASS_EOS_TABLE_HPP_
