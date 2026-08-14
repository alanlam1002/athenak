#ifndef RECONSTRUCT_PPM_HPP_
#define RECONSTRUCT_PPM_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ppm.hpp
//! \brief piecewise parabolic reconstruction with both Collela-Woodward (CW) limiters
//! (implemented in the PPM4 inline function) and Collela-Sekora (CS) extremum preserving
//! limiters (implemented in the PPMX inline function) for a Cartesian-like coordinates
//! with uniform spacing.
//!
//! This version does not include the extensions to the CS limiters described by
//! McCorquodale et al. and as implemented in Athena++ by K. Felker.  This is to keep the
//! code simple, because Kyle found these extensions did not improve the solution very
//! much in practice, and because they can break monotonicity.
//!
//! REFERENCES:
//! (CW) P. Colella & P. Woodward, "The Piecewise Parabolic Method (PPM) for Gas-Dynamical
//! Simulations", JCP, 54, 174 (1984)
//!
//! (CS) P. Colella & M. Sekora, "A limiter for PPM that preserves accuracy at smooth
//! extrema", JCP, 227, 7069 (2008)
//!
//! (MC) P. McCorquodale & P. Colella, "A high-order finite-volume method for conservation
//! laws on locally refined grids", CAMCoS, 6, 1 (2011)
//!
//! (PH) L. Peterson & G.W. Hammett, "Positivity preservation and advection algorithms
//! with application to edge plasma turbulence", SIAM J. Sci. Com, 35, B576 (2013)

#include <math.h>
#include <algorithm>    // max()

#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \fn PPM4()
//! \brief Original PPM (Colella & Woodward) parabolic reconstruction.  Returns
//! interpolated values at L/R edges of cell i, that is ql(i+1) and qr(i). Works for
//! reconstruction in any dimension by passing in the appropriate q_im2,...,q _ip2.

KOKKOS_INLINE_FUNCTION
void PPM4(const Real &q_im2, const Real &q_im1, const Real &q_i, const Real &q_ip1,
          const Real &q_ip2, Real &ql_ip1, Real &qr_i) {
  //---- Interpolate L/R values (CS eqn 16, PH 3.26 and 3.27) ----
  // qlv = q at left  side of cell-center = q[i-1/2] = a_{j,-} in CS
  // qrv = q at right side of cell-center = q[i+1/2] = a_{j,+} in CS
  Real qlv = (7.*(q_i + q_im1) - (q_im2 + q_ip1))/12.0;
  Real qrv = (7.*(q_i + q_ip1) - (q_im1 + q_ip2))/12.0;

  //---- limit qrv and qlv to neighboring cell-centered values (CS eqn 13) ----
  qlv = fmax(qlv, fmin(q_i, q_im1));
  qlv = fmin(qlv, fmax(q_i, q_im1));
  qrv = fmax(qrv, fmin(q_i, q_ip1));
  qrv = fmin(qrv, fmax(q_i, q_ip1));

  //--- monotonize interpolated L/R states (CS eqns 14, 15) ---
  Real qc = qrv - q_i;
  Real qd = qlv - q_i;
  if ((qc*qd) >= 0.0) {
    qlv = q_i;
    qrv = q_i;
  } else {
    if (fabs(qc) >= 2.0*fabs(qd)) {
      qrv = q_i - 2.0*qd;
    }
    if (fabs(qd) >= 2.0*fabs(qc)) {
      qlv = q_i - 2.0*qc;
    }
  }

  //---- set L/R states ----
  ql_ip1 = qrv;
  qr_i   = qlv;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn PPM4() -- non-uniform/curvilinear overload (Task B7)
//! \brief Generalizes the flat 4-point interpolation weights (-1/12,7/12,7/12,-1/12) and
//! the CW monotonicity-clamp factor "2.0" to the position-dependent coefficients produced
//! by the geometry factories (Mignone 2014 eq. B.9/B.14 for the interpolation weights at
//! faces i and i+1; eq. 48 h_plus/h_minus ratios for the clamp). AthenaK has no
//! mesh-stretching, so the ONLY thing that varies with position is the metric/Jacobian
//! weighting captured by these coefficients -- everything else about the algorithm
//! (limit qlv/qrv to neighboring cell-centered values, then the CW eqn 1.10 monotonicity
//! test) is geometry-independent and unchanged from the flat overload above. Reduces
//! exactly to the flat overload when c1..c4=(-1/12,7/12,7/12,-1/12) and hp=hm=2 (verified:
//! the Cartesian geometry factory fills exactly these constants).
KOKKOS_INLINE_FUNCTION
void PPM4(const Real &q_im2, const Real &q_im1, const Real &q_i, const Real &q_ip1,
          const Real &q_ip2,
          const Real &c1_i, const Real &c2_i, const Real &c3_i, const Real &c4_i,
          const Real &c1_ip1, const Real &c2_ip1, const Real &c3_ip1, const Real &c4_ip1,
          const Real &hp_i, const Real &hm_i, Real &ql_ip1, Real &qr_i) {
  //---- Interpolate L/R values using the position-dependent 4-point weights ----
  // qlv = q at face i (left side of cell i);  qrv = q at face i+1 (right side of cell i)
  Real qlv = c1_i*q_im2 + c2_i*q_im1 + c3_i*q_i + c4_i*q_ip1;
  Real qrv = c1_ip1*q_im1 + c2_ip1*q_i + c3_ip1*q_ip1 + c4_ip1*q_ip2;

  //---- limit qrv and qlv to neighboring cell-centered values (CS eqn 13) ----
  qlv = fmax(qlv, fmin(q_i, q_im1));
  qlv = fmin(qlv, fmax(q_i, q_im1));
  qrv = fmax(qrv, fmin(q_i, q_ip1));
  qrv = fmin(qrv, fmax(q_i, q_ip1));

  //--- monotonize interpolated L/R states, generalized CW eqn 1.10 (Mignone eq. 45) ---
  Real qc = qrv - q_i;
  Real qd = qlv - q_i;
  if ((qc*qd) >= 0.0) {
    qlv = q_i;
    qrv = q_i;
  } else {
    if (fabs(qc) >= hp_i*fabs(qd)) {
      qrv = q_i - hp_i*qd;
    }
    if (fabs(qd) >= hm_i*fabs(qc)) {
      qlv = q_i - hm_i*qc;
    }
  }

  //---- set L/R states ----
  ql_ip1 = qrv;
  qr_i   = qlv;
  return;
}


//----------------------------------------------------------------------------------------
//! \fn PPMX()
//! \brief PPM parabolic reconstruction with Colella & Sekora limiters.  Returns
//! interpolated values at L/R edges of cell i, that is ql(i+1) and qr(i). Works for
//! reconstruction in any dimension by passing in the appropriate q_im2,...,q _ip2.

KOKKOS_INLINE_FUNCTION
void PPMX(const Real &q_im2, const Real &q_im1, const Real &q_i, const Real &q_ip1,
          const Real &q_ip2, Real &ql_ip1, Real &qr_i) {
  //---- Compute L/R values (CS eqns 12-15, PH 3.26 and 3.27) ----
  // qlv = q at left  side of cell-center = q[i-1/2] = a_{j,-} in CS
  // qrv = q at right side of cell-center = q[i+1/2] = a_{j,+} in CS
  Real qlv = (7.*(q_i + q_im1) - (q_im2 + q_ip1))/12.0;
  Real qrv = (7.*(q_i + q_ip1) - (q_im1 + q_ip2))/12.0;

  //---- Apply CS monotonicity limiters to qrv and qlv ----
  // approximate second derivatives at i-1/2 (PH 3.35)
  // KGF: add the off-center quantities first to preserve FP symmetry
  Real d2qc = 3.0*((q_im1 + q_i) - 2.0*qlv);
  Real d2ql = (q_im2 + q_i  ) - 2.0*q_im1;
  Real d2qr = (q_im1 + q_ip1) - 2.0*q_i;

  // limit second derivative (PH 3.36)
  Real d2qlim = 0.0;
  Real lim_slope = fmin(fabs(d2ql),fabs(d2qr));
  if (d2qc > 0.0 && d2ql > 0.0 && d2qr > 0.0) {
    d2qlim = SIGN(d2qc)*fmin(1.25*lim_slope,fabs(d2qc));
  }
  if (d2qc < 0.0 && d2ql < 0.0 && d2qr < 0.0) {
    d2qlim = SIGN(d2qc)*fmin(1.25*lim_slope,fabs(d2qc));
  }
  // compute limited value for qlv (PH 3.33 and 3.34)
  if (((q_im1 - qlv)*(q_i - qlv)) > 0.0) {
    qlv = 0.5*(q_i + q_im1) - d2qlim/6.0;
  }

  // approximate second derivatives at i+1/2 (PH 3.35)
  // KGF: add the off-center quantities first to preserve FP symmetry
  d2qc = 3.0*((q_i + q_ip1) - 2.0*qrv);
  d2ql = d2qr;
  d2qr = (q_i + q_ip2) - 2.0*q_ip1;

  // limit second derivative (PH 3.36)
  d2qlim = 0.0;
  lim_slope = fmin(fabs(d2ql),fabs(d2qr));
  if (d2qc > 0.0 && d2ql > 0.0 && d2qr > 0.0) {
    d2qlim = SIGN(d2qc)*fmin(1.25*lim_slope,fabs(d2qc));
  }
  if (d2qc < 0.0 && d2ql < 0.0 && d2qr < 0.0) {
    d2qlim = SIGN(d2qc)*fmin(1.25*lim_slope,fabs(d2qc));
  }
  // compute limited value for qrv (PH 3.33 and 3.34)
  if (((q_i - qrv)*(q_ip1 - qrv)) > 0.0) {
    qrv = 0.5*(q_i + q_ip1) - d2qlim/6.0;
  }

  //---- identify extrema, use smooth extremum limiter ----
  // CS 20 (missing "OR"), and PH 3.31
  Real qa = (qrv - q_i)*(q_i - qlv);
  Real qb = (q_im1 - q_i)*(q_i - q_ip1);
  if (qa <= 0.0 || qb <= 0.0) {
    // approximate second derivatives (PH 3.37)
    // KGF: add the off-center quantities first to preserve FP symmetry
    Real d2q  = 6.0*(qlv + qrv - 2.0*q_i);
    Real d2qc = (q_im1 + q_ip1) - 2.0*q_i;
    Real d2ql = (q_im2 + q_i  ) - 2.0*q_im1;
    Real d2qr = (q_i   + q_ip2) - 2.0*q_ip1;

    // limit second derivatives (PH 3.38)
    d2qlim = 0.0;
    lim_slope = fmin(fabs(d2ql),fabs(d2qr));
    lim_slope = fmin(fabs(d2qc),lim_slope);
    if (d2qc > 0.0 && d2ql > 0.0 && d2qr > 0.0 && d2q > 0.0) {
      d2qlim = SIGN(d2q)*fmin(1.25*lim_slope,fabs(d2q));
    }
    if (d2qc < 0.0 && d2ql < 0.0 && d2qr < 0.0 && d2q < 0.0) {
      d2qlim = SIGN(d2q)*fmin(1.25*lim_slope,fabs(d2q));
    }

    // limit L/R states at extrema (PH 3.39)
    Real rho = 0.0;
    if ( fabs(d2q) > (1.0e-12)*fmax( fabs(q_im1), fmax(fabs(q_i),fabs(q_ip1))) ) {
      // Limiter is not sensitive to round-off error.  Use limited slope
      rho = d2qlim/d2q;
    }
    qlv = q_i + (qlv - q_i)*rho;
    qrv = q_i + (qrv - q_i)*rho;
  } else {
    // Monotonize again, away from extrema (CW eqn 1.10, PH 3.32)
    Real qc = qrv - q_i;
    Real qd = qlv - q_i;
    if (fabs(qc) >= 2.0*fabs(qd)) {
      qrv = q_i - 2.0*qd;
    }
    if (fabs(qd) >= 2.0*fabs(qc)) {
      qlv = q_i - 2.0*qc;
    }
  }

  //---- set L/R states ----
  ql_ip1 = qrv;
  qr_i   = qlv;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn PPMX() -- non-uniform/curvilinear overload (Task B7)
//! \brief Generalizes only the two geometry-dependent pieces of PPMX: the initial 4-point
//! interpolation (same c1..c4 coefficients as the PPM4 overload above) and the final
//! "away from extrema" CW-eqn-1.10 monotonicity clamp (same hp_i/hm_i ratios). The
//! Colella-Sekora extremum-preserving logic in between (second-derivative estimates,
//! PH eqns 3.35-3.39) operates purely on cell-centered VALUES with an implicit
//! uniform-INDEX-spacing assumption that remains exactly valid here: AthenaK has no
//! mesh-stretching, so grid spacing in index space is always uniform even in curvilinear
//! coordinates -- only the metric/Jacobian weighting captured by c1..c4/hp/hm varies with
//! position. That section is therefore copied unchanged from the flat overload.
KOKKOS_INLINE_FUNCTION
void PPMX(const Real &q_im2, const Real &q_im1, const Real &q_i, const Real &q_ip1,
          const Real &q_ip2,
          const Real &c1_i, const Real &c2_i, const Real &c3_i, const Real &c4_i,
          const Real &c1_ip1, const Real &c2_ip1, const Real &c3_ip1, const Real &c4_ip1,
          const Real &hp_i, const Real &hm_i, Real &ql_ip1, Real &qr_i) {
  //---- Compute L/R values using the position-dependent 4-point weights ----
  Real qlv = c1_i*q_im2 + c2_i*q_im1 + c3_i*q_i + c4_i*q_ip1;
  Real qrv = c1_ip1*q_im1 + c2_ip1*q_i + c3_ip1*q_ip1 + c4_ip1*q_ip2;

  //---- Apply CS monotonicity limiters to qrv and qlv ----
  // approximate second derivatives at i-1/2 (PH 3.35)
  // KGF: add the off-center quantities first to preserve FP symmetry
  Real d2qc = 3.0*((q_im1 + q_i) - 2.0*qlv);
  Real d2ql = (q_im2 + q_i  ) - 2.0*q_im1;
  Real d2qr = (q_im1 + q_ip1) - 2.0*q_i;

  // limit second derivative (PH 3.36)
  Real d2qlim = 0.0;
  Real lim_slope = fmin(fabs(d2ql),fabs(d2qr));
  if (d2qc > 0.0 && d2ql > 0.0 && d2qr > 0.0) {
    d2qlim = SIGN(d2qc)*fmin(1.25*lim_slope,fabs(d2qc));
  }
  if (d2qc < 0.0 && d2ql < 0.0 && d2qr < 0.0) {
    d2qlim = SIGN(d2qc)*fmin(1.25*lim_slope,fabs(d2qc));
  }
  // compute limited value for qlv (PH 3.33 and 3.34)
  if (((q_im1 - qlv)*(q_i - qlv)) > 0.0) {
    qlv = 0.5*(q_i + q_im1) - d2qlim/6.0;
  }

  // approximate second derivatives at i+1/2 (PH 3.35)
  // KGF: add the off-center quantities first to preserve FP symmetry
  d2qc = 3.0*((q_i + q_ip1) - 2.0*qrv);
  d2ql = d2qr;
  d2qr = (q_i + q_ip2) - 2.0*q_ip1;

  // limit second derivative (PH 3.36)
  d2qlim = 0.0;
  lim_slope = fmin(fabs(d2ql),fabs(d2qr));
  if (d2qc > 0.0 && d2ql > 0.0 && d2qr > 0.0) {
    d2qlim = SIGN(d2qc)*fmin(1.25*lim_slope,fabs(d2qc));
  }
  if (d2qc < 0.0 && d2ql < 0.0 && d2qr < 0.0) {
    d2qlim = SIGN(d2qc)*fmin(1.25*lim_slope,fabs(d2qc));
  }
  // compute limited value for qrv (PH 3.33 and 3.34)
  if (((q_i - qrv)*(q_ip1 - qrv)) > 0.0) {
    qrv = 0.5*(q_i + q_ip1) - d2qlim/6.0;
  }

  //---- identify extrema, use smooth extremum limiter ----
  // CS 20 (missing "OR"), and PH 3.31
  Real qa = (qrv - q_i)*(q_i - qlv);
  Real qb = (q_im1 - q_i)*(q_i - q_ip1);
  if (qa <= 0.0 || qb <= 0.0) {
    // approximate second derivatives (PH 3.37)
    // KGF: add the off-center quantities first to preserve FP symmetry
    Real d2q  = 6.0*(qlv + qrv - 2.0*q_i);
    Real d2qc = (q_im1 + q_ip1) - 2.0*q_i;
    Real d2ql = (q_im2 + q_i  ) - 2.0*q_im1;
    Real d2qr = (q_i   + q_ip2) - 2.0*q_ip1;

    // limit second derivatives (PH 3.38)
    d2qlim = 0.0;
    lim_slope = fmin(fabs(d2ql),fabs(d2qr));
    lim_slope = fmin(fabs(d2qc),lim_slope);
    if (d2qc > 0.0 && d2ql > 0.0 && d2qr > 0.0 && d2q > 0.0) {
      d2qlim = SIGN(d2q)*fmin(1.25*lim_slope,fabs(d2q));
    }
    if (d2qc < 0.0 && d2ql < 0.0 && d2qr < 0.0 && d2q < 0.0) {
      d2qlim = SIGN(d2q)*fmin(1.25*lim_slope,fabs(d2q));
    }

    // limit L/R states at extrema (PH 3.39)
    Real rho = 0.0;
    if ( fabs(d2q) > (1.0e-12)*fmax( fabs(q_im1), fmax(fabs(q_i),fabs(q_ip1))) ) {
      // Limiter is not sensitive to round-off error.  Use limited slope
      rho = d2qlim/d2q;
    }
    qlv = q_i + (qlv - q_i)*rho;
    qrv = q_i + (qrv - q_i)*rho;
  } else {
    // Monotonize again, away from extrema, generalized CW eqn 1.10 (Mignone eq. 45)
    Real qc = qrv - q_i;
    Real qd = qlv - q_i;
    if (fabs(qc) >= hp_i*fabs(qd)) {
      qrv = q_i - hp_i*qd;
    }
    if (fabs(qd) >= hm_i*fabs(qc)) {
      qlv = q_i - hm_i*qc;
    }
  }

  //---- set L/R states ----
  ql_ip1 = qrv;
  qr_i   = qlv;
  return;
}
#endif // RECONSTRUCT_PPM_HPP_
