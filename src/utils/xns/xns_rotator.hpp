#ifndef UTILS_XNS_XNS_ROTATOR_HPP_
#define UTILS_XNS_XNS_ROTATOR_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file xns_rotator.hpp
//  \brief Reads a 2D axisymmetric equilibrium rotating-NS model produced by the
//  external XNS code (Bucciantini & Del Zanna 2011; Pili et al.) from its
//  Grid.dat/Hydroeq.dat/Surf.dat output files, and interpolates it at an arbitrary
//  (r,theta).
//
//  XNS's Grid.dat: header "NTH NR NRREG RMIN", then NTH theta cell centers (always
//  uniform over [0,pi] by XNS's own convention), then NR r cell centers (uniform
//  unless XNS was run with STRETCH=.TRUE., in which case non-uniform beyond RREG).
//  XNS's Hydroeq.dat: header "NTH NR OMG", then NTH*NR rows (theta-major, r-minor
//  loop order) of 9 columns: rho, press, psi, v^phi, alpha, beta^phi, chi, Qr, Qt.
//  The last three (scalar-field quantities) are identically zero for a GR (non-STT)
//  model and are not read here.
//  XNS's Surf.dat: NTH values (no header), the stellar surface radius R_surf(theta)
//  at each of Grid.dat's own theta cell centers (XNSMAIN.f90: `WRITE(13,*)
//  R(WSURF(IX)+1)`). The star is NOT spherically symmetric (rotational flattening),
//  so this -- not a spherical radius cutoff -- is what determines whether a given
//  (r,theta) point is inside the star or in XNS's own (non-vacuum, nonzero-density)
//  vacuum/atmosphere solution; see SurfaceRadius() below.

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"

namespace xns {

enum class Loc {Host, Device};

KOKKOS_INLINE_FUNCTION
static Real Lerp(Real x, Real x1, Real x2, Real y1, Real y2) {
  Real t = (x - x1)/(x2 - x1);
  return y1*(1. - t) + y2*t;
}

class XNSRotator {
 public:
  explicit XNSRotator(ParameterInput *pin) {
    std::string id_dir = pin->GetString("problem", "id_dir");
    std::string grid_file = id_dir + "/Grid.dat";
    std::string hydro_file = id_dir + "/Hydroeq.dat";
    std::string surf_file = id_dir + "/Surf.dat";

    std::ifstream gfile(grid_file);
    if (!gfile.is_open()) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Could not open XNS grid file '" << grid_file << "'"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    int nrreg;
    Real rmin;
    gfile >> nth_ >> nr_ >> nrreg >> rmin;

    Kokkos::realloc(thetagrid_, nth_);
    for (int it = 0; it < nth_; ++it) {
      gfile >> thetagrid_.h_view(it);
    }
    Kokkos::realloc(rgrid_, nr_);
    for (int ir = 0; ir < nr_; ++ir) {
      gfile >> rgrid_.h_view(ir);
    }
    gfile.close();
    dtheta_ = thetagrid_.h_view(1) - thetagrid_.h_view(0);
    rmax_ = rgrid_.h_view(nr_-1);

    std::ifstream hfile(hydro_file);
    if (!hfile.is_open()) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Could not open XNS hydro file '" << hydro_file << "'"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    int nth_check, nr_check;
    Real omg;
    hfile >> nth_check >> nr_check >> omg;
    if (nth_check != nth_ || nr_check != nr_) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "XNS grid file '" << grid_file << "' (NTH,NR = "
                << nth_ << "," << nr_ << ") and hydro file '" << hydro_file
                << "' (NTH,NR = " << nth_check << "," << nr_check
                << ") disagree" << std::endl;
      std::exit(EXIT_FAILURE);
    }

    int npts = nth_*nr_;
    Kokkos::realloc(rho_, npts);
    Kokkos::realloc(press_, npts);
    Kokkos::realloc(psi_, npts);
    Kokkos::realloc(vphi_, npts);
    Kokkos::realloc(alpha_, npts);
    Kokkos::realloc(betaphi_, npts);
    Real chi, qr, qt;
    for (int idx = 0; idx < npts; ++idx) {
      hfile >> rho_.h_view(idx) >> press_.h_view(idx) >> psi_.h_view(idx)
            >> vphi_.h_view(idx) >> alpha_.h_view(idx) >> betaphi_.h_view(idx)
            >> chi >> qr >> qt;
    }
    hfile.close();

    std::ifstream sfile(surf_file);
    if (!sfile.is_open()) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Could not open XNS surface file '" << surf_file
                << "'" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    Kokkos::realloc(rsurf_, nth_);
    for (int it = 0; it < nth_; ++it) {
      sfile >> rsurf_.h_view(it);
    }
    sfile.close();

    thetagrid_.template modify<HostMemSpace>();
    rgrid_.template modify<HostMemSpace>();
    rsurf_.template modify<HostMemSpace>();
    rho_.template modify<HostMemSpace>();
    press_.template modify<HostMemSpace>();
    psi_.template modify<HostMemSpace>();
    vphi_.template modify<HostMemSpace>();
    alpha_.template modify<HostMemSpace>();
    betaphi_.template modify<HostMemSpace>();

    thetagrid_.template sync<DevExeSpace>();
    rgrid_.template sync<DevExeSpace>();
    rsurf_.template sync<DevExeSpace>();
    rho_.template sync<DevExeSpace>();
    press_.template sync<DevExeSpace>();
    psi_.template sync<DevExeSpace>();
    vphi_.template sync<DevExeSpace>();
    alpha_.template sync<DevExeSpace>();
    betaphi_.template sync<DevExeSpace>();
  }

  Real rmax() const { return rmax_; }

  // Surface radius R_surf(theta) -- the star is oblate (rotational flattening),
  // so this, not a spherical radius comparison, is what callers should use to
  // decide whether a point is inside the star or in XNS's own vacuum/atmosphere
  // solution (which has nonzero, non-monotonic density that can exceed a typical
  // evolution-code dfloor -- see xns_rotstar.cpp).
  template<Loc loc>
  KOKKOS_INLINE_FUNCTION
  Real SurfaceRadius(Real theta) const {
    int lb_th, ub_th;
    Real tth;
    ThetaBracket<loc>(theta, lb_th, ub_th, tth);
    auto& rs_view = GetView<loc>(rsurf_);
    return Lerp(tth, 0., 1., rs_view(lb_th), rs_view(ub_th));
  }

  // Bilinear-interpolate all 6 fields at a given (r, theta). Caller is responsible
  // for only calling this with r <= rmax() -- points outside the table's own domain
  // are the pgen's (not this class's) responsibility to fill with atmosphere/flat
  // values.
  template<Loc loc>
  KOKKOS_INLINE_FUNCTION
  void Interpolate(Real r, Real theta, Real &rho, Real &press, Real &psi,
                    Real &alpha, Real &betaphi, Real &vphi) const {
    auto& r_view = GetView<loc>(rgrid_);

    int lb_th, ub_th;
    Real tth;
    ThetaBracket<loc>(theta, lb_th, ub_th, tth);

    // Binary search for the bracketing r-grid indices (rgrid_ need not be uniform:
    // XNS supports a stretched, non-uniform grid beyond RREG for other models).
    int nr = nr_;
    int lb_r = 0;
    int ub_r = nr - 1;
    while (ub_r - lb_r > 1) {
      int mid = (lb_r + ub_r)/2;
      if (r_view(mid) > r) {
        ub_r = mid;
      } else {
        lb_r = mid;
      }
    }

    Real tr = (r - r_view(lb_r))/(r_view(ub_r) - r_view(lb_r));

    BilinearField<loc>(rho_, nr, lb_th, ub_th, lb_r, ub_r, tth, tr, rho);
    BilinearField<loc>(press_, nr, lb_th, ub_th, lb_r, ub_r, tth, tr, press);
    BilinearField<loc>(psi_, nr, lb_th, ub_th, lb_r, ub_r, tth, tr, psi);
    BilinearField<loc>(alpha_, nr, lb_th, ub_th, lb_r, ub_r, tth, tr, alpha);
    BilinearField<loc>(betaphi_, nr, lb_th, ub_th, lb_r, ub_r, tth, tr, betaphi);
    BilinearField<loc>(vphi_, nr, lb_th, ub_th, lb_r, ub_r, tth, tr, vphi);
  }

 private:
  int nth_, nr_;
  Real dtheta_;
  Real rmax_;
  DualArray1D<Real> thetagrid_;
  DualArray1D<Real> rgrid_;
  DualArray1D<Real> rsurf_;  // R_surf(theta), NTH values, see SurfaceRadius()
  // Flattened theta-major/r-minor (idx = it*nr_ + ir), matching Hydroeq.dat's own
  // row order exactly, so no transpose is needed at load time.
  DualArray1D<Real> rho_, press_, psi_, alpha_, betaphi_, vphi_;

  template<Loc loc>
  KOKKOS_INLINE_FUNCTION
  auto& GetView(const DualArray1D<Real>& arr) const {
    if constexpr (loc == Loc::Host) {
      return arr.h_view;
    } else {
      return arr.d_view;
    }
  }

  // Shared by Interpolate() and SurfaceRadius(): bracketing theta-grid indices
  // and interpolation weight for a given theta. theta is always uniform over
  // [0,pi] by XNS's own convention (unlike the r-grid) -- O(1) index, not a
  // binary search.
  template<Loc loc>
  KOKKOS_INLINE_FUNCTION
  void ThetaBracket(Real theta, int &lb_th, int &ub_th, Real &tth) const {
    auto& th_view = GetView<loc>(thetagrid_);
    int nth = nth_;
    lb_th = static_cast<int>(theta/dtheta_ - 0.5);
    lb_th = (lb_th < 0) ? 0 : ((lb_th > nth-2) ? nth-2 : lb_th);
    ub_th = lb_th + 1;
    tth = (theta - th_view(lb_th))/(th_view(ub_th) - th_view(lb_th));
  }

  template<Loc loc>
  KOKKOS_INLINE_FUNCTION
  void BilinearField(const DualArray1D<Real>& field, int nr, int lb_th, int ub_th,
                      int lb_r, int ub_r, Real tth, Real tr, Real &out) const {
    auto& f = GetView<loc>(field);
    Real f_lbth = Lerp(tr, 0., 1., f(lb_th*nr+lb_r), f(lb_th*nr+ub_r));
    Real f_ubth = Lerp(tr, 0., 1., f(ub_th*nr+lb_r), f(ub_th*nr+ub_r));
    out = Lerp(tth, 0., 1., f_lbth, f_ubth);
  }
};

}  // namespace xns

#endif  // UTILS_XNS_XNS_ROTATOR_HPP_
