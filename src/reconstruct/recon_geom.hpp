#ifndef RECONSTRUCT_RECON_GEOM_HPP_
#define RECONSTRUCT_RECON_GEOM_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file recon_geom.hpp
//! \brief Curvilinear/non-uniform-spacing variants of the reconstruction kernels, kept
//! in a SEPARATE file from plm.hpp/ppm.hpp on purpose.
//!
//! plm.hpp and ppm.hpp are upstream-owned files that this project deliberately leaves at
//! zero diff, so that pulling in changes from upstream/main never conflicts there. All
//! geometry-aware reconstruction added by the curvilinear-coordinates work lives here
//! instead, and the only upstream file that has to know about it is recon.hpp, which
//! dispatches to `*Geom` when the reconstruction direction has position-dependent
//! geometry. The names are distinct (PLMGeom/PPM4Geom/PPMXGeom) rather than overloads of
//! PLM/PPM4/PPMX so that resolution never depends on which headers happen to be in scope.
//!
//! Each function here reduces EXACTLY to its upstream counterpart when handed the
//! coefficients that geometry_cartesian.cpp produces -- see the individual doc comments.
//!
//! REFERENCES:
//! (Mignone 2014) A. Mignone, "High-order conservative reconstruction schemes for finite
//! volume methods in cylindrical and spherical coordinates", JCP, 270, 784 (2014)

#include <math.h>
#include "athena.hpp"
#include "coordinates/mesh_geometry.hpp"  // PlmCoeff, PlmCoefIdx

//----------------------------------------------------------------------------------------
//! \fn PLMGeom()
//! \brief Reconstructs linear slope in cell i to compute ql(i+1) and qr(i). Works for
//! reconstruction in any dimension by passing in the appropriate q_im1, q_i, and q_ip1,
//! together with the corresponding CENTROID positions x_im1, x_i, x_ip1 (NOT necessarily
//! evenly spaced -- e.g. cylindrical/spherical radial centroids) and the FACE positions
//! xf_i, xf_ip1 bounding cell i (needed because the centroid-to-face offset is generally
//! NOT half the cell width for a non-uniform/curvilinear grid).
//!
//! Formula ported (math only) from old Athena++'s src/reconstruct/plm.cpp: the
//! generalized van Leer limiter with the Mignone (2014) eq. 33/37 correction for
//! non-uniform/curvilinear centroid spacing. Reduces exactly to upstream plm.hpp's
//! simplified uniform-Cartesian van Leer formula when x_ip1-x_i == x_i-x_im1 == dx1f and
//! xf_i/xf_ip1 sit exactly halfway between centroids.
//!
//! Exactness property (verified algebraically, see DEVELOPMENT.md): if q is an exactly
//! linear function of the stored centroid x (q_im1/i/ip1 = a + b*x_im1/i/ip1 for
//! any a,b), this reconstructs the SAME linear function evaluated at the true face
//! positions xf_i/xf_ip1, regardless of how non-uniform the spacing is.

KOKKOS_INLINE_FUNCTION
void PLMGeom(const Real &q_im1, const Real &q_i, const Real &q_ip1,
             const PlmCoeff &c, Real &ql_ip1, Real &qr_i) {
  // compute L/R differences
  Real dwl = (q_i - q_im1);
  Real dwr = (q_ip1 - q_i);

  // Mignone (2014) eq. 33/37 generalized van Leer limiter for non-uniform spacing.
  // Every position-dependent ratio was precomputed by MakePlmCoeff(), so exactly ONE
  // division survives here -- the same count as upstream's uniform-spacing PLM.
  Real dqF = dwr*c.pF;
  Real dqB = dwl*c.pB;
  Real dq2 = dqF*dqB;

  Real dqm = dq2*(c.cf*dqB + c.cb*dqF)/(SQR(dqB) + SQR(dqF) + dq2*c.cc);
  if (dq2 <= 0.0) dqm = 0.0;

  // compute ql_(i+1/2) and qr_(i-1/2) using limited slope and the true face offsets
  ql_ip1 = q_i + c.pP*dqm;
  qr_i   = q_i - c.pM*dqm;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn PLMGeom() -- GeomData array overload
//! \brief loads the seven precomputed factors for cell `idx` out of one of GeomData's
//! plm_c1/c2/c3 arrays and applies the limiter above. This is the form the reconstruction
//! kernel uses; the PlmCoeff form above is the one the unit test drives directly.

KOKKOS_INLINE_FUNCTION
void PLMGeom(const Real &q_im1, const Real &q_i, const Real &q_ip1,
             const DvceArray3D<Real> &pc, const bool uniform,
             const int m, const int idx, Real &ql_ip1, Real &qr_i) {
  if (uniform) {
    // Uniformly spaced direction: the general limiter below provably collapses to this
    // (with factors 1,1,2,2,2,0.5,0.5 the numerator becomes 2*dq2*(dwl+dwr) and the
    // denominator (dwl+dwr)^2, and the face offsets become +/-1/2). Taking the collapsed
    // form directly is both much cheaper -- 1 multiply and no coefficient loads, versus
    // ~10 multiplies and 7 loads -- and BITWISE identical to upstream plm.hpp's PLM(),
    // so a uniform-grid run reproduces a pre-curvilinear build exactly.
    //
    // `uniform` is constant across the whole launch (it is a property of the direction,
    // not of the cell), so this costs one perfectly-predicted branch on CPU and is
    // warp-coherent on GPU. It is NOT a coordinate-system test: cylindrical/spherical
    // reach it too, in their flat phi/z directions.
    Real dql = (q_i - q_im1);
    Real dqr = (q_ip1 - q_i);
    Real dq2 = dql*dqr;
    Real dqm = dq2/(dql + dqr);
    if (dq2 <= 0.0) dqm = 0.0;
    ql_ip1 = q_i + dqm;
    qr_i   = q_i - dqm;
    return;
  }
  PlmCoeff c;
  c.pF = pc(m,IPLM_PF,idx);
  c.pB = pc(m,IPLM_PB,idx);
  c.cf = pc(m,IPLM_CF,idx);
  c.cb = pc(m,IPLM_CB,idx);
  c.cc = pc(m,IPLM_CC,idx);
  c.pP = pc(m,IPLM_PP,idx);
  c.pM = pc(m,IPLM_PM,idx);
  PLMGeom(q_im1, q_i, q_ip1, c, ql_ip1, qr_i);
}

//----------------------------------------------------------------------------------------
//! \fn PPM4Geom()
//! \brief Generalizes upstream PPM4's flat 4-point interpolation weights
//! (-1/12,7/12,7/12,-1/12) and its CW monotonicity-clamp factor "2.0" to the
//! position-dependent coefficients produced by the geometry factories (Mignone 2014 eq.
//! B.9/B.14 for the interpolation weights at faces i and i+1; eq. 48 h_plus/h_minus
//! ratios for the clamp). AthenaK has no mesh-stretching, so the ONLY thing that varies
//! with position is the metric/Jacobian weighting captured by these coefficients --
//! everything else about the algorithm (limit qlv/qrv to neighboring cell-centered
//! values, then the CW eqn 1.10 monotonicity test) is geometry-independent and unchanged
//! from upstream PPM4. Reduces exactly to upstream PPM4 when
//! c1..c4=(-1/12,7/12,7/12,-1/12) and hp=hm=2 (verified: the Cartesian geometry factory
//! fills exactly these constants).

KOKKOS_INLINE_FUNCTION
void PPM4Geom(const Real &q_im2, const Real &q_im1, const Real &q_i, const Real &q_ip1,
              const Real &q_ip2,
              const Real &c1_i, const Real &c2_i, const Real &c3_i, const Real &c4_i,
              const Real &c1_ip1, const Real &c2_ip1, const Real &c3_ip1,
              const Real &c4_ip1,
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
//! \fn PPMXGeom()
//! \brief Generalizes only the two geometry-dependent pieces of upstream PPMX: the
//! initial 4-point interpolation (same c1..c4 coefficients as PPM4Geom above) and the
//! final "away from extrema" CW-eqn-1.10 monotonicity clamp (same hp_i/hm_i ratios). The
//! Colella-Sekora extremum-preserving logic in between (second-derivative estimates,
//! PH eqns 3.35-3.39) operates purely on cell-centered VALUES with an implicit
//! uniform-INDEX-spacing assumption that remains exactly valid here: AthenaK has no
//! mesh-stretching, so grid spacing in index space is always uniform even in curvilinear
//! coordinates -- only the metric/Jacobian weighting captured by c1..c4/hp/hm varies with
//! position. That section is therefore copied unchanged from upstream PPMX.

KOKKOS_INLINE_FUNCTION
void PPMXGeom(const Real &q_im2, const Real &q_im1, const Real &q_i, const Real &q_ip1,
              const Real &q_ip2,
              const Real &c1_i, const Real &c2_i, const Real &c3_i, const Real &c4_i,
              const Real &c1_ip1, const Real &c2_ip1, const Real &c3_ip1,
              const Real &c4_ip1,
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
#endif  // RECONSTRUCT_RECON_GEOM_HPP_
