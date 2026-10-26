#ifndef CFC_CFC_TIDAL_HPP_
#define CFC_CFC_TIDAL_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cfc_tidal.hpp
//! \brief <cfc> tidal_lapse (tde_debris rung 2, research TDE_DEBRIS_SPEC.md §2; R-145 item 3;
//! default off): the star in its own freely falling, non-rotating frame. To quadratic order
//! in the distance from the centre (Fermi normal coordinates along the centre-of-mass
//! geodesic) g_00 = -(1 + E_ij x^i x^j), so the tide enters through the lapse only:
//!     alpha = alpha_self (1 + 1/2 E_ij x^i x^j),
//! applied after the CFC solve of the star's own gravity (puncture off). The existing
//! -E d_i alpha and -S^i d_i alpha source terms then give the tidal force and its work.
//! The factor is clamped below at tidal_fmin (the expansion fails at |x| ~ R, i.e. in the
//! far atmosphere near periapsis; only the clamp keeps alpha > 0 there).
//!
//! E_ij(tau), tau = the star's proper time = the simulation time, is tabulated on the host
//! at start-up (uniform in tau, linear interpolation) for a parabolic orbit about a mass M
//! with areal periapsis r_p, released at areal radius r_0 on the inbound branch; the orbit
//! lies in the x-y plane, the BH initially along -x from the star, moving counter-clockwise
//! (the c110 rp10 decks' geometry):
//!   newtonian: Barker's equation, R = r_p (1 + D^2), D = tan(nu/2),
//!              t - t_p = sqrt(2 r_p^3/M) (D + D^3/3); E_ij = (M/R^3)(delta_ij - 3 n_i n_j)
//!              with n at angle nu - nu_0 from +x;
//!   marck:     the Schwarzschild geodesic (E = 1, L^2 = 2 M r_p^2/(r_p - 2M)), integrated
//!              in phi through Binet's equation; in the parallel-propagated tetrad (Marck
//!              1983; e.g. Ishii, Shibata & Mino 2005) the tidal tensor is
//!                E_11 = (M/r^3)[1 - 3(1 + L^2/r^2) cos^2 Psi]
//!                E_22 = (M/r^3)[1 - 3(1 + L^2/r^2) sin^2 Psi]
//!                E_12 = -(M/r^3) 3(1 + L^2/r^2) sin Psi cos Psi
//!                E_33 = (M/r^3)(1 + 3 L^2/r^2),
//!              dPsi/dtau = E L/(r^2 + L^2) (axis 3 = the orbit normal = z). Quoted from the
//!              literature from memory (unverified); checks: traceless, and it tends to the
//!              Newtonian tensor with Psi -> phi as L^2/r^2 -> 0.
//! The table also carries the Schwarzschild coordinate time t(tau), r(tau) and Psi(tau)
//! for comparisons with the full runs (written to tidal_table_file if set).

#include <string>
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"

namespace cfc {

struct TidalLapse {
  bool on = false;
  bool marck = false;
  Real m = 1.0, rp = 10.0, r0 = 30.0, fmin = 0.1, dtau = 0.01;
  Real cx = 0.0, cy = 0.0, cz = 0.0;     // the frame origin (the star's centre) on the grid
  std::vector<Real> tau, tc, rr, ang;    // proper time, coordinate time, radius, angle
  std::vector<Real> e[6];                // xx, xy, xz, yy, yz, zz
  void Setup(ParameterInput *pin);
  void Eval(Real t, Real eij[6]) const;  // linear in tau; clamped to the table's ends
};

}  // namespace cfc

#endif  // CFC_CFC_TIDAL_HPP_
