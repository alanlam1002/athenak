#ifndef PGEN_DYN_GRMHD_RNS_ST_READER_HPP_
#define PGEN_DYN_GRMHD_RNS_ST_READER_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rns_st_reader.hpp
//  \brief Host-side reader/interpolator for scalarized-neutron-star (massive
//  scalar-tensor) equilibrium initial data produced by the RNS-ST solver, ported from
//  `~/SACRA_2D/SACRA_MPI/read_grass_st.f90` (read_grass_st_initial_data +
//  read_grass_st_interp). Used by dyngr_rns_st.cpp.
//
//  The solver's ASCII output is an axisymmetric grid in a compactified radial
//  coordinate s = 1/(1+(r_ref/r)^(1/nfac)) and p = cos(theta), storing SACRA's own
//  BSSN-conformal variables (lapse, shift, conformal exponent phi_bssn, conformal
//  trace-free extrinsic curvature Atilde_ij, conformal metric perturbation h_ij) plus
//  matter (rest-mass density, covariant velocity u_i) and the scalar field value sphi.
//  Interpolate() reproduces read_grass_st_interp's 7-point (n_order=3) Lagrange scheme
//  exactly; converting the result to AthenaK's physical-ADM/contravariant-velocity
//  convention is the caller's job (see dyngr_rns_st.cpp), not this reader's.

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "athena.hpp"

namespace rns_st {

//----------------------------------------------------------------------------------------
//! \class RnsStData
//! \brief loads one RNS-ST ID file and interpolates it at arbitrary Cartesian points.

class RnsStData {
 public:
  // Interpolated quantities at a point, in SACRA's own conformal-BSSN convention
  // (NOT AthenaK's physical ADM convention -- see dyngr_rns_st.cpp for that conversion).
  struct Point {
    Real alpha;
    Real beta_u[3];     // shift, Cartesian
    Real phi_bssn;      // BSSN conformal exponent: psi^4 = exp(4*phi_bssn)
    Real Ktrace;        // trace K -- always 0 in this data (maximal slicing)
    // Conformal, trace-free extrinsic curvature Atilde_ij (Cartesian)
    Real Atilde_xx, Atilde_yy, Atilde_zz, Atilde_xy, Atilde_xz, Atilde_yz;
    // Conformal metric perturbation h_ij = gamma_tilde_ij - delta_ij (Cartesian)
    Real h_xx, h_yy, h_zz, h_xy, h_xz, h_yz;
    Real rho;           // rest-mass density
    Real u_d[3];        // COVARIANT fluid velocity, Cartesian (lower index)
    Real sphi;          // scalar field value
  };

  explicit RnsStData(const std::string &fname) { Load(fname); }

  void Interpolate(Real x, Real y, Real z, Point *out) const;

 private:
  static constexpr int n_order = 3;
  static constexpr int n_rns = 10;
  // Index (1-based, matches the Fortran source) of the rest-mass density column.
  static constexpr int rho_col = 8;

  // Unit conversion constants from the RNS-ST solver's own units to this code's
  // working (G=c=Msun=1) units -- ported verbatim from read_grass_st.f90, not
  // re-derived (already validated in the SACRA pipeline this data came from).
  static constexpr Real r_uni = 6.77140812e-01;
  static constexpr Real t_uni = 2.03001708e+05;
  static constexpr Real rho_uni = 1.61930347e-18;

  int nr_ = 0, np_ = 0, nfac_ = 1;
  Real nfaci_ = 1.0;
  Real R_ref_ = 0.0;
  Real ref_ = 0.0;     // R_ref * r_uni
  Real re_ = 0.0, rp_ = 0.0;   // equatorial / polar radius
  int ix_e_ = 0, ix_p_ = 0;

  // Flat storage. s_/r_ span fortran index range [1-n_order, nr_] (size nr_+n_order).
  // p_ spans [1-n_order, np_+n_order] (size np_+2*n_order). dat_/ome_ are laid out on
  // the outer product of those two ranges (dat_ additionally over n_rns components).
  std::vector<Real> s_, r_, p_, dat_, ome_;

  int nrspan() const { return nr_ + n_order; }
  int npspan() const { return np_ + 2*n_order; }

  // Map a Fortran-convention index to a flat-storage offset.
  int SIdx(int i) const { return i + (n_order - 1); }
  int PIdx(int j) const { return j + (n_order - 1); }

  Real &S(int i) { return s_[SIdx(i)]; }
  Real S(int i) const { return s_[SIdx(i)]; }
  Real &Rg(int i) { return r_[SIdx(i)]; }
  Real Rg(int i) const { return r_[SIdx(i)]; }
  Real &P(int j) { return p_[PIdx(j)]; }
  Real P(int j) const { return p_[PIdx(j)]; }

  int DatIdx(int comp, int i, int j) const {
    return (comp - 1)*nrspan()*npspan() + SIdx(i)*npspan() + PIdx(j);
  }
  Real &Dat(int comp, int i, int j) { return dat_[DatIdx(comp, i, j)]; }
  Real Dat(int comp, int i, int j) const { return dat_[DatIdx(comp, i, j)]; }

  Real &Ome(int i, int j) { return ome_[SIdx(i)*npspan() + PIdx(j)]; }
  Real Ome(int i, int j) const { return ome_[SIdx(i)*npspan() + PIdx(j)]; }

  void Load(const std::string &fname);
};

//----------------------------------------------------------------------------------------
//! \fn void RnsStData::Load
//  \brief Port of read_grass_st_initial_data (read_grass_st.f90:133-244).

inline void RnsStData::Load(const std::string &fname) {
  std::ifstream in(fname);
  if (!in.is_open()) {
    std::stringstream msg;
    msg << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Could not open RNS-ST initial-data file '" << fname << "'" << std::endl;
    throw std::runtime_error(msg.str());
  }

  // The header line carries only nr, np, nfac, R_ref, but real ID files pad it with
  // extra trailing values (legacy metadata, e.g. a repeated central density/mass) that
  // Fortran's record-based list-directed read silently discards. Do the same
  // explicitly: parse the first line in isolation so the data-section reads below
  // (which are NOT record-bounded) start cleanly at line 2.
  std::string header_line;
  if (!std::getline(in, header_line)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Could not read header line of '" << fname << "'" << std::endl;
    throw std::runtime_error(msg.str());
  }
  std::istringstream header_stream(header_line);
  header_stream >> nr_ >> np_ >> nfac_ >> R_ref_;
  if (!header_stream) {
    std::stringstream msg;
    msg << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Could not parse header line of '" << fname << "'" << std::endl;
    throw std::runtime_error(msg.str());
  }
  nfaci_ = 1.0/static_cast<Real>(nfac_);

  s_.assign(nrspan(), 0.0);
  r_.assign(nrspan(), 0.0);
  p_.assign(npspan(), 0.0);
  dat_.assign(static_cast<size_t>(n_rns)*nrspan()*npspan(), 0.0);
  ome_.assign(static_cast<size_t>(nrspan())*npspan(), 0.0);

  for (int i = 1; i <= nr_; ++i) {
    for (int j = 1; j <= np_; ++j) {
      Real tmp[n_rns - 1];
      Real sphi;
      in >> S(i) >> P(j);
      for (int c = 0; c < n_rns - 1; ++c) { in >> tmp[c]; }
      in >> Ome(i, j) >> sphi;
      if (!in) {
        std::stringstream msg;
        msg << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
            << "IO error reading RNS-ST data at (i,j)=(" << i << "," << j << ") in '"
            << fname << "'" << std::endl;
        throw std::runtime_error(msg.str());
      }
      for (int c = 0; c < n_rns - 1; ++c) { Dat(c + 1, i, j) = tmp[c]; }
      Dat(n_rns, i, j) = sphi;
      // Jordan -> Einstein frame metric-potential shift (beta0=1 hardcoded, matching
      // the rest of this module's convention).
      Dat(1, i, j) -= 0.25*sphi*sphi;
      Dat(2, i, j) -= 0.5*sphi*sphi;
    }
  }

  // Unit conversion (RNS-solver units -> this code's G=c=Msun=1 units).
  for (int i = 1; i <= nr_; ++i) {
    for (int j = 1; j <= np_; ++j) {
      Dat(4, i, j) /= t_uni;
      Dat(rho_col, i, j) *= rho_uni;
      Ome(i, j) /= t_uni;
    }
  }

  // Fill ghost cells by parity/reflection at s=0 and both theta poles.
  for (int i = 1; i <= n_order; ++i) {
    S(1 - i) = -S(1 + i);
    for (int j = 1; j <= np_; ++j) {
      for (int c = 1; c <= n_rns; ++c) { Dat(c, 1 - i, j) = Dat(c, 1 + i, j); }
    }
  }
  for (int j = 1; j <= n_order; ++j) {
    for (int i = 1 - n_order; i <= nr_; ++i) {
      for (int c = 1; c <= n_rns; ++c) {
        Dat(c, i, 1 - j) = Dat(c, i, 1 + j);
        Dat(c, i, np_ + j) = Dat(c, i, np_ - j);
      }
    }
    P(1 - j) = -P(1 + j);
    P(np_ + j) = P(np_)*2.0 - P(np_ - j);
  }
  for (int i = 1; i <= n_order; ++i) {
    for (int j = 1; j <= np_; ++j) { Ome(1 - i, j) = Ome(1 + i, j); }
  }
  for (int j = 1; j <= n_order; ++j) {
    for (int i = 1 - n_order; i <= nr_; ++i) {
      Ome(i, 1 - j) = Ome(i, 1 + j);
      Ome(i, np_ + j) = Ome(i, np_ - j);
    }
  }

  // Locate the stellar surface along the equator (j=1) and pole (j=np_).
  for (int i = 1; i <= nr_; ++i) {
    if (Dat(rho_col, i, 1) > 0.0) { ix_e_ = i; }
    if (Dat(rho_col, i, np_) > 0.0) { ix_p_ = i; }
  }

  ref_ = R_ref_*r_uni;
  for (int i = 1 - n_order; i <= nr_; ++i) {
    Rg(i) = ref_*std::pow(S(i)/(1.0 - S(i)), nfac_);
  }
  re_ = Rg(ix_e_);
  rp_ = Rg(ix_p_);
}

//----------------------------------------------------------------------------------------
//! \fn void RnsStData::Interpolate
//  \brief Port of read_grass_st_interp (read_grass_st.f90:246-477). Gamma-tilde^i
//  (var(19:21) in the Fortran source) is intentionally not computed here: AthenaK's
//  ADMToZ4c derives the equivalent conformal-connection quantity itself via finite
//  differences, so the caller never needs it.

inline void RnsStData::Interpolate(Real x, Real y, Real z, Point *out) const {
  Real r0 = std::sqrt(x*x + y*y + z*z);
  Real s0 = 1.0/(1.0 + std::pow(ref_/r0, nfaci_));
  Real ct0, st0;
  if (r0 < 1.0e-99) {
    ct0 = 1.0; st0 = 0.0;
  } else {
    ct0 = std::abs(z/r0);
    st0 = std::sqrt(x*x + y*y)/r0;
  }
  Real cp0, sp0;
  Real rho_cyl = std::sqrt(x*x + y*y);
  if (rho_cyl < 1.0e-99) {
    cp0 = 1.0; sp0 = 0.0;
  } else {
    cp0 = x/rho_cyl; sp0 = y/rho_cyl;
  }
  Real c2p = cp0*cp0 - sp0*sp0;
  Real s2p = 2.0*sp0*cp0;

  int ii_found = nr_ + 1;
  for (int ii = 2; ii <= nr_; ++ii) {
    if (Rg(ii) > r0) { ii_found = ii; break; }
  }
  int ir = std::min(nr_ - n_order, std::max(1, ii_found - 1));

  int ip = static_cast<int>(ct0*static_cast<Real>(np_));
  ip = std::min(np_, std::max(1, ip));

  Real tmp[14] = {0.0};
  Real sphi_acc = 0.0;
  for (int di = -n_order; di <= n_order; ++di) {
    for (int dj = -n_order; dj <= n_order; ++dj) {
      Real fr = 1.0, fp = 1.0, fdr = 0.0, fdp = 0.0;
      for (int dk = -n_order; dk <= n_order; ++dk) {
        if (dk == di) { continue; }
        fr *= (s0 - S(ir + dk))/(S(ir + di) - S(ir + dk));
      }
      for (int dk = -n_order; dk <= n_order; ++dk) {
        if (dk == dj) { continue; }
        fp *= (ct0 - P(ip + dk))/(P(ip + dj) - P(ip + dk));
      }
      for (int dk = -n_order; dk <= n_order; ++dk) {
        if (dk == di) { continue; }
        Real f0 = 1.0/(S(ir + di) - S(ir + dk));
        for (int dl = -n_order; dl <= n_order; ++dl) {
          if (dl == dk || dl == di) { continue; }
          f0 *= (s0 - S(ir + dl))/(S(ir + di) - S(ir + dl));
        }
        fdr += f0;
      }
      for (int dk = -n_order; dk <= n_order; ++dk) {
        if (dk == dj) { continue; }
        Real f0 = 1.0/(P(ip + dj) - P(ip + dk));
        for (int dl = -n_order; dl <= n_order; ++dl) {
          if (dl == dk || dl == dj) { continue; }
          f0 *= (ct0 - P(ip + dl))/(P(ip + dj) - P(ip + dl));
        }
        fdp += f0;
      }
      for (int c = 0; c < 4; ++c) {
        Real d = Dat(c + 1, ir + di, ip + dj);
        tmp[c] += fr*fp*d;
        tmp[c + 4] += fdr*fp*d;
        tmp[c + 8] += fr*fdp*d;
      }
      sphi_acc += fr*fp*Dat(n_rns, ir + di, ip + dj);
    }
  }
  Real rfac = std::pow(r0/ref_, nfaci_);
  for (int c = 4; c < 8; ++c) {
    tmp[c] *= rfac/((1.0 + rfac)*(1.0 + rfac))/r0/static_cast<Real>(nfac_);
  }

  // Interpolate density (and rotation profile) with a stencil that shrinks toward the
  // stellar surface, matching the Fortran source's handling of the density discontinuity.
  int lo = n_order;
  while (lo > 0) {
    bool all_positive = true;
    for (int di = -lo + 1; di <= lo && all_positive; ++di) {
      for (int dj = -lo + 1; dj <= lo; ++dj) {
        if (Dat(rho_col, ir + di, ip + dj) <= 0.0) { all_positive = false; break; }
      }
    }
    if (all_positive) { break; }
    lo = (lo > 1) ? 1 : lo - 1;
  }
  if (lo == 0) {
    tmp[12] = 0.0; tmp[13] = 0.0;
  } else {
    for (int di = -lo + 1; di <= lo; ++di) {
      for (int dj = -lo + 1; dj <= lo; ++dj) {
        Real fp = 1.0;
        for (int dk = -lo + 1; dk <= lo; ++dk) {
          if (dk == di) { continue; }
          fp *= (s0 - S(ir + dk))/(S(ir + di) - S(ir + dk));
        }
        for (int dk = -lo + 1; dk <= lo; ++dk) {
          if (dk == dj) { continue; }
          fp *= (ct0 - P(ip + dk))/(P(ip + dj) - P(ip + dk));
        }
        tmp[12] += fp*Dat(rho_col, ir + di, ip + dj);
        tmp[13] += fp*Ome(ir + di, ip + dj);
      }
    }
  }

  Real sigma = tmp[0];
  Real eta = tmp[1] - tmp[2];
  Real omega = tmp[3];
  Real omegadr = tmp[7];
  Real omegadt = tmp[11];
  Real rho = tmp[12];

  Real alp = std::exp(0.5*(tmp[1] + tmp[2]));
  Real phi = sigma/3.0 + eta/12.0;
  Real bx = r0*st0*omega*sp0;
  Real by = -r0*st0*omega*cp0;

  Real fp_ext = -0.5*st0*std::exp(eta - 4.0*phi)/alp;
  Real aij1 = -fp_ext*s2p*(omegadr*r0*st0 - omegadt*st0*ct0);
  Real aij2 = -aij1;
  Real aij3 = 0.0;
  Real aij4 = fp_ext*c2p*(omegadr*r0*st0 - omegadt*st0*ct0);
  Real aij5 = fp_ext*sp0*(omegadr*r0*(-ct0) - omegadt*st0*st0);
  Real aij6 = fp_ext*cp0*(omegadr*r0*ct0 - omegadt*st0*(-st0));

  Real fdr_m = std::exp(2.0*sigma - 4.0*phi) - 1.0;
  Real fdp_m = (std::exp(eta) - std::exp(2.0*sigma))*std::exp(-4.0*phi);
  Real hij1 = fdp_m*sp0*sp0 + fdr_m;
  Real hij2 = fdp_m*cp0*cp0 + fdr_m;
  Real hij3 = fdr_m;
  Real hij4 = -fdp_m*sp0*cp0;
  Real hij5 = 0.0;
  Real hij6 = 0.0;

  Real ui1, ui2, ui3 = 0.0;
  if (rho > 0.0) {
    Real vphi = r0*st0*(tmp[13] - omega);
    Real fp_u = std::exp(eta)*vphi/std::sqrt(alp*alp - std::exp(eta)*vphi*vphi);
    ui1 = -fp_u*sp0;
    ui2 = fp_u*cp0;
  } else {
    ui1 = 0.0; ui2 = 0.0;
  }

  // Equatorial-plane parity: components odd under z -> -z (only aij5/aij6 are ever
  // nonzero here -- hij5/hij6/ui3 are identically zero in this axisymmetric ansatz, so
  // the Fortran source's parity flip on those is a no-op and is skipped).
  Real zsign = (z >= 0.0) ? 1.0 : -1.0;
  aij5 *= zsign;
  aij6 *= zsign;

  out->alpha = alp;
  out->beta_u[0] = bx; out->beta_u[1] = by; out->beta_u[2] = 0.0;
  out->phi_bssn = phi;
  out->Ktrace = 0.0;
  out->Atilde_xx = aij1; out->Atilde_yy = aij2; out->Atilde_zz = aij3;
  out->Atilde_xy = aij4; out->Atilde_xz = aij5; out->Atilde_yz = aij6;
  out->h_xx = hij1; out->h_yy = hij2; out->h_zz = hij3;
  out->h_xy = hij4; out->h_xz = hij5; out->h_yz = hij6;
  out->rho = rho;
  out->u_d[0] = ui1; out->u_d[1] = ui2; out->u_d[2] = ui3;
  out->sphi = sphi_acc;

  if (!std::isfinite(out->alpha) || !std::isfinite(out->rho) ||
      !std::isfinite(out->sphi)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "NaN/Inf in RNS-ST interpolated data at (x,y,z)=(" << x << "," << y << ","
        << z << ")" << std::endl;
    throw std::runtime_error(msg.str());
  }
}

} // namespace rns_st

#endif // PGEN_DYN_GRMHD_RNS_ST_READER_HPP_
