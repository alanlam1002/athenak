#ifndef MESH_PROLONGATION_HPP_
#define MESH_PROLONGATION_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file prolongation.hpp
//! \brief prolongation operators for cell-centered and face-centered variables,
//! implemented as inline functions so they can be used both in Bval and AMR functions.

#include "z4c/z4c.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProlongCC()
//! \brief 2nd-order (piecewise-linear) CONSERVATIVE prolongation for cell-centered
//! variables, valid on curvilinear as well as Cartesian grids.
//!
//! Writes the reconstruction in terms of TRUE VOLUMETRIC CENTROIDS:
//!
//!     a_f = a_c + sum_d s_d * (x_d,fine - x_d,coarse)
//!
//! which is EXACTLY volume-conservative, i.e. sum_f (V_f * a_f) == V_c * a_c. The reason
//! is that the volumetric centroid satisfies V_c*x_c == sum_f(V_f*x_f) by construction
//! (the moment integral is additive over sub-intervals), so sum_f V_f*(x_f - x_c) == 0
//! and every slope term cancels in the volume-weighted sum. Because GeomData's geometry
//! is separable (V = vi*vj*vk), the 3D sum factorizes into three independent 1D
//! identities, so this holds in 1D/2D/3D alike. Both identities are asserted directly by
//! pgen/unit_tests/coarse_geometry_test.cpp rather than merely assumed here.
//!
//! No correction pass is therefore needed -- conservation is structural, not enforced
//! afterwards.
//!
//! Slopes are min-mod limited on the COARSE centroid spacing (physical units, per unit
//! length), then evaluated at the fine centroid offsets. For a uniform grid
//! x_f - x_c = +/-dx_c/4 and the coarse spacings are equal, so this reduces EXACTLY to
//! the previous `0.125*(SIGN(dl)+SIGN(dr))*fmin(|dl|,|dr|)` form -- Cartesian results are
//! unchanged.
//!
//! Takes the six centroid arrays rather than two whole GeomData structs so the enclosing
//! kernel closure grows by 6 View handles instead of ~92 (see the capture-size note in
//! mesh_geometry.hpp).

KOKKOS_INLINE_FUNCTION
void ProlongCC(const int m, const int v, const int k, const int j, const int i,
               const int fk, const int fj, const int fi,
               const bool multi_d, const bool three_d,
               const DvceArray2D<Real> &cx1v, const DvceArray2D<Real> &cx2v,
               const DvceArray2D<Real> &cx3v,
               const DvceArray2D<Real> &x1v, const DvceArray2D<Real> &x2v,
               const DvceArray2D<Real> &x3v,
               const bool uniform,
               const DvceArray5D<Real> &ca, const DvceArray5D<Real> &a) {
  if (uniform) {
    // Uniform grid: upstream's original expression, BITWISE identical to a
    // pre-curvilinear build. The centroid form below is mathematically equal here but
    // differs in floating-point association, and that alone was enough to break the
    // GR-MHD AMR boundary test (~1e6 C2P failures) and the radiation AMR test.
    Real dl = ca(m,v,k,j,i  ) - ca(m,v,k,j,i-1);
    Real dr = ca(m,v,k,j,i+1) - ca(m,v,k,j,i  );
    Real dvar1 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
    Real dvar2 = 0.0;
    if (multi_d) {
      dl = ca(m,v,k,j  ,i) - ca(m,v,k,j-1,i);
      dr = ca(m,v,k,j+1,i) - ca(m,v,k,j  ,i);
      dvar2 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
    }
    Real dvar3 = 0.0;
    if (three_d) {
      dl = ca(m,v,k  ,j,i) - ca(m,v,k-1,j,i);
      dr = ca(m,v,k+1,j,i) - ca(m,v,k  ,j,i);
      dvar3 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
    }
    a(m,v,fk,fj,fi  ) = ca(m,v,k,j,i) - dvar1 - dvar2 - dvar3;
    a(m,v,fk,fj,fi+1) = ca(m,v,k,j,i) + dvar1 - dvar2 - dvar3;
    if (multi_d) {
      a(m,v,fk,fj+1,fi  ) = ca(m,v,k,j,i) - dvar1 + dvar2 - dvar3;
      a(m,v,fk,fj+1,fi+1) = ca(m,v,k,j,i) + dvar1 + dvar2 - dvar3;
    }
    if (three_d) {
      a(m,v,fk+1,fj  ,fi  ) = ca(m,v,k,j,i) - dvar1 - dvar2 + dvar3;
      a(m,v,fk+1,fj  ,fi+1) = ca(m,v,k,j,i) + dvar1 - dvar2 + dvar3;
      a(m,v,fk+1,fj+1,fi  ) = ca(m,v,k,j,i) - dvar1 + dvar2 + dvar3;
      a(m,v,fk+1,fj+1,fi+1) = ca(m,v,k,j,i) + dvar1 + dvar2 + dvar3;
    }
    return;
  }
  // x1 slope: min-mod limited gradient on the coarse centroid spacing
  Real dl = ca(m,v,k,j,i  ) - ca(m,v,k,j,i-1);
  Real dr = ca(m,v,k,j,i+1) - ca(m,v,k,j,i  );
  Real gl = dl/(cx1v(m,i) - cx1v(m,i-1));
  Real gr = dr/(cx1v(m,i+1) - cx1v(m,i));
  Real s1 = 0.5*(SIGN(gl) + SIGN(gr))*fmin(fabs(gl), fabs(gr));
  Real d1m = x1v(m,fi  ) - cx1v(m,i);
  Real d1p = x1v(m,fi+1) - cx1v(m,i);

  // x2 slope
  Real s2 = 0.0, d2m = 0.0, d2p = 0.0;
  if (multi_d) {
    dl = ca(m,v,k,j  ,i) - ca(m,v,k,j-1,i);
    dr = ca(m,v,k,j+1,i) - ca(m,v,k,j  ,i);
    gl = dl/(cx2v(m,j) - cx2v(m,j-1));
    gr = dr/(cx2v(m,j+1) - cx2v(m,j));
    s2 = 0.5*(SIGN(gl) + SIGN(gr))*fmin(fabs(gl), fabs(gr));
    d2m = x2v(m,fj  ) - cx2v(m,j);
    d2p = x2v(m,fj+1) - cx2v(m,j);
  }

  // x3 slope
  Real s3 = 0.0, d3m = 0.0, d3p = 0.0;
  if (three_d) {
    dl = ca(m,v,k  ,j,i) - ca(m,v,k-1,j,i);
    dr = ca(m,v,k+1,j,i) - ca(m,v,k  ,j,i);
    gl = dl/(cx3v(m,k) - cx3v(m,k-1));
    gr = dr/(cx3v(m,k+1) - cx3v(m,k));
    s3 = 0.5*(SIGN(gl) + SIGN(gr))*fmin(fabs(gl), fabs(gr));
    d3m = x3v(m,fk  ) - cx3v(m,k);
    d3p = x3v(m,fk+1) - cx3v(m,k);
  }

  // interpolate to the finer grid at each child's own centroid
  Real ac = ca(m,v,k,j,i);
  a(m,v,fk,fj,fi  ) = ac + s1*d1m + s2*d2m + s3*d3m;
  a(m,v,fk,fj,fi+1) = ac + s1*d1p + s2*d2m + s3*d3m;
  if (multi_d) {
    a(m,v,fk,fj+1,fi  ) = ac + s1*d1m + s2*d2p + s3*d3m;
    a(m,v,fk,fj+1,fi+1) = ac + s1*d1p + s2*d2p + s3*d3m;
  }
  if (three_d) {
    a(m,v,fk+1,fj  ,fi  ) = ac + s1*d1m + s2*d2m + s3*d3p;
    a(m,v,fk+1,fj  ,fi+1) = ac + s1*d1p + s2*d2m + s3*d3p;
    a(m,v,fk+1,fj+1,fi  ) = ac + s1*d1m + s2*d2p + s3*d3p;
    a(m,v,fk+1,fj+1,fi+1) = ac + s1*d1p + s2*d2p + s3*d3p;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn FCSharedIncrements()
//! \brief the per-direction increment a shared-face prolongation adds to the coarse
//! value, for the two children along one transverse direction.
//!
//! Factored out because every one of the three shared-face kernels exists in TWO copies
//! -- `src/mesh/prolongation.hpp` for the regrid path and the `*Owned` variants in
//! `src/bvals/prolongation.cpp` for the per-stage boundary path, differing only in
//! whether the result is stored directly or routed through the write-ownership check.
//! Six independent copies of this arithmetic is exactly how the two paths come to
//! silently disagree, so all six now call this.
//!
//! `q{m1,0,p1}` are the three coarse values along the transverse direction, `c` the
//! coarse index and `f` the corresponding fine index, `ct`/`t` the coarse/fine
//! area-weighted transverse centroids. Returns the increments for the low and high child.
//!
//! In the uniform branch these are exactly -/+ upstream's `dvar`, so the caller's
//! `q + dm + dm3` is bitwise identical to upstream's `q - dvar2 - dvar3` (negation is
//! exact in IEEE arithmetic, and the operation order is unchanged).

KOKKOS_INLINE_FUNCTION
void FCSharedIncrements(const Real qm1, const Real q0, const Real qp1,
                        const DvceArray2D<Real> &ct, const DvceArray2D<Real> &t,
                        const int m, const int c, const int f,
                        const bool uniform, Real &dm, Real &dp) {
  if (uniform) {
    Real dl = q0 - qm1;
    Real dr = qp1 - q0;
    Real dvar = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
    dm = -dvar;
    dp = dvar;
    return;
  }
  Real gl = (q0 - qm1)/(ct(m,c  ) - ct(m,c-1));
  Real gr = (qp1 - q0)/(ct(m,c+1) - ct(m,c  ));
  Real s = 0.5*(SIGN(gl) + SIGN(gr))*fmin(fabs(gl), fabs(gr));
  dm = s*(t(m,f  ) - ct(m,c));
  dp = s*(t(m,f+1) - ct(m,c));
}

//----------------------------------------------------------------------------------------
//! \fn ProlongFCSharedX1Face()
//! \brief 2nd-order (piecewise-linear) prolongation operator for face-centered variables
//! on shared X1-faces between fine and coarse cells.
//!
//! CURVILINEAR GENERALIZATION (SMR/AMR Phase 2). Upstream's `+/-0.125*(SIGN+SIGN)*fmin`
//! is `+/-(1/4)*minmod` of UNDIVIDED differences, which bakes in two uniform-grid
//! assumptions: that the limiter may compare raw differences (valid only when the two
//! coarse spacings straddling the point are equal), and that a fine face centre sits at
//! exactly a quarter of the coarse spacing from the coarse one.
//!
//! The general form writes the fine value as
//!     B_f = B_c + s2*(y_f - y_c) + s3*(z_f - z_c)
//! with s the slope limited on true centroid spacings and y/z the AREA-WEIGHTED
//! transverse centroids (GeomData::fcN_d). This is EXACTLY flux-conservative --
//! sum(A_f*B_f) == A_c*B_c -- because sum(A_f*y_f) == A_c*y_c holds by construction of
//! that centroid, so the slope terms cancel against the area weights. Note this is why
//! the area-weighted centroid is required and the VOLUMETRIC one (x1v/x2v/x3v) will not
//! do: the two differ for x2/x3-faces in cylindrical and spherical.
//!
//! `uniform` (GeomData::cells_uniform) takes upstream's original expression, which is
//! then bitwise identical rather than merely equivalent -- see GeomData::cells_uniform
//! for why that distinction has already cost real debugging time.

KOKKOS_INLINE_FUNCTION
void ProlongFCSharedX1Face(const int m, const int k, const int j, const int i,
                   const int fk, const int fj, const int fi,
                   const bool multi_d, const bool three_d,
                   const DvceArray2D<Real> &ct1, const DvceArray2D<Real> &ct2,
                   const DvceArray2D<Real> &t1, const DvceArray2D<Real> &t2,
                   const bool uniform,
                   const DvceArray4D<Real> &cbx1f, const DvceArray4D<Real> &bx1f) {
  // Prolongate b.x1f (v=0) by interpolating in x2/x3
  Real d2m = 0.0, d2p = 0.0;
  if (multi_d) {
    FCSharedIncrements(cbx1f(m,k,j-1,i), cbx1f(m,k,j,i), cbx1f(m,k,j+1,i),
                       ct1, t1, m, j, fj, uniform, d2m, d2p);
  }
  Real d3m = 0.0, d3p = 0.0;
  if (three_d) {
    FCSharedIncrements(cbx1f(m,k-1,j,i), cbx1f(m,k,j,i), cbx1f(m,k+1,j,i),
                       ct2, t2, m, k, fk, uniform, d3m, d3p);
  }

  bx1f(m,fk,fj,fi) = cbx1f(m,k,j,i) + d2m + d3m;
  if (multi_d) {
    bx1f(m,fk,fj+1,fi) = cbx1f(m,k,j,i) + d2p + d3m;
  }
  if (three_d) {
    bx1f(m,fk+1,fj  ,fi) = cbx1f(m,k,j,i) + d2m + d3p;
    bx1f(m,fk+1,fj+1,fi) = cbx1f(m,k,j,i) + d2p + d3p;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ProlongFCSharedX2Face()
//! \brief 2nd-order (piecewise-linear) prolongation operator for face-centered variables
//! on shared X2-faces between fine and coarse cells

//! Transverse directions are x1 and x3, so ct1/t1 are the coarse/fine fc2_1 and ct2/t2
//! the fc2_3 arrays. See ProlongFCSharedX1Face for the derivation.

KOKKOS_INLINE_FUNCTION
void ProlongFCSharedX2Face(const int m, const int k, const int j, const int i,
                   const int fk, const int fj, const int fi,
                   const bool three_d,
                   const DvceArray2D<Real> &ct1, const DvceArray2D<Real> &ct2,
                   const DvceArray2D<Real> &t1, const DvceArray2D<Real> &t2,
                   const bool uniform,
                   const DvceArray4D<Real> &cbx2f, const DvceArray4D<Real> &bx2f) {
  // Prolongate b.x2f (v=1) by interpolating in x1/x3
  Real d1m, d1p;
  FCSharedIncrements(cbx2f(m,k,j,i-1), cbx2f(m,k,j,i), cbx2f(m,k,j,i+1),
                     ct1, t1, m, i, fi, uniform, d1m, d1p);
  Real d3m = 0.0, d3p = 0.0;
  if (three_d) {
    FCSharedIncrements(cbx2f(m,k-1,j,i), cbx2f(m,k,j,i), cbx2f(m,k+1,j,i),
                       ct2, t2, m, k, fk, uniform, d3m, d3p);
  }

  bx2f(m,fk  ,fj,fi  ) = cbx2f(m,k,j,i) + d1m + d3m;
  bx2f(m,fk  ,fj,fi+1) = cbx2f(m,k,j,i) + d1p + d3m;
  if (three_d) {
    bx2f(m,fk+1,fj,fi  ) = cbx2f(m,k,j,i) + d1m + d3p;
    bx2f(m,fk+1,fj,fi+1) = cbx2f(m,k,j,i) + d1p + d3p;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ProlongFCSharedX3Face()
//! \brief 2nd-order (piecewise-linear) prolongation operator for face-centered variables
//! on shared X3-faces between fine and coarse cells

//! Transverse directions are x1 and x2, so ct1/t1 are the coarse/fine fc3_1 and ct2/t2
//! the fc3_2 arrays. See ProlongFCSharedX1Face for the derivation.

KOKKOS_INLINE_FUNCTION
void ProlongFCSharedX3Face(const int m, const int k, const int j, const int i,
                   const int fk, const int fj, const int fi,
                   const bool multi_d,
                   const DvceArray2D<Real> &ct1, const DvceArray2D<Real> &ct2,
                   const DvceArray2D<Real> &t1, const DvceArray2D<Real> &t2,
                   const bool uniform,
                   const DvceArray4D<Real> &cbx3f, const DvceArray4D<Real> &bx3f) {
  // Prolongate b.x3f (v=2) by interpolating in x1/x2
  Real d1m, d1p;
  FCSharedIncrements(cbx3f(m,k,j,i-1), cbx3f(m,k,j,i), cbx3f(m,k,j,i+1),
                     ct1, t1, m, i, fi, uniform, d1m, d1p);
  Real d2m = 0.0, d2p = 0.0;
  if (multi_d) {
    FCSharedIncrements(cbx3f(m,k,j-1,i), cbx3f(m,k,j,i), cbx3f(m,k,j+1,i),
                       ct2, t2, m, j, fj, uniform, d2m, d2p);
  }

  bx3f(m,fk,fj  ,fi  ) = cbx3f(m,k,j,i) + d1m + d2m;
  bx3f(m,fk,fj  ,fi+1) = cbx3f(m,k,j,i) + d1p + d2m;
  if (multi_d) {
    bx3f(m,fk,fj+1,fi  ) = cbx3f(m,k,j,i) + d1m + d2p;
    bx3f(m,fk,fj+1,fi+1) = cbx3f(m,k,j,i) + d1p + d2p;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ProlongInternalFC()
//! \brief 2nd-order prolongation operator for face-centered variables on internal edges
//! of new fine cells within one coarse cell using divergence-preserving interpolation
//! scheme of Toth & Roe, JCP 180, 736 (2002).

//! CURVILINEAR GENERALIZATION (SMR/AMR Phase 2): the scheme is run on FACE FLUXES
//! Phi = Area*B_n rather than on the pointwise B_n, which is a change of variable, not a
//! re-derivation. Toth & Roe's constraint is "each of the 8 children is divergence-free",
//! and the discrete divergence is sum(+/- Area*B_n) = 0, i.e. sum(+/- Phi) = 0 --
//! formally identical to the Cartesian unit-cube case the coefficients were derived for.
//! So: read the 12 outer faces as Phi, run the SAME moment arithmetic (1/8 and 1/16)
//! verbatim, then divide each internal-face result by that face's own Area. div(B)=0 is
//! then exact for any cell shape, at second-order-consistent accuracy.
//!
//! This also fixes a latent UPSTREAM bug rather than merely generalizing. Upstream's form
//! mixes x2f and x3f differences with unit weight, so it enforces dB1 + dB2 + dB3 = 0
//! instead of A1*dB1 + A2*dB2 + A3*dB3 = 0 -- equivalent only when A1 == A2 == A3, i.e.
//! for CUBIC cells. Measured on Cartesian with unmodified upstream code, a square grid
//! holds div(B) at 4e-16 while the same grid at aspect ratio 2 reaches 7e-03 after one
//! regrid (DEVELOPMENT.md, "Phase 2 finding").
//!
//! Hence `cubic` is GeomData::cubic_cells and NOT cells_uniform, unlike every other fast
//! path in this file: a uniform but non-cubic Cartesian grid must take the corrected
//! path. Every existing Cartesian AMR test uses cubic cells and so is bitwise unchanged.
//! Note the two forms are not bitwise equal even for cubic cells, because multiplying by
//! Area and dividing it back out rounds -- the distinction that cost ~1e6 C2P failures in
//! Phase 1.

//! \fn ProlongFCInternalValues()
//! \brief computes the 12 (3D) or 4 (2D) internal-face values without storing them, so
//! that the regrid path and the boundary path -- which differ ONLY in whether the result
//! is written directly or routed through the write-ownership check -- share one copy of
//! this arithmetic instead of two. Ordering, per component, is
//!   v1: (fk,fj,fi+1) (fk,fj+1,fi+1) (fk+1,fj,fi+1) (fk+1,fj+1,fi+1)
//!   v2: (fk,fj+1,fi) (fk,fj+1,fi+1) (fk+1,fj+1,fi) (fk+1,fj+1,fi+1)
//!   v3: (fk+1,fj,fi) (fk+1,fj,fi+1) (fk+1,fj+1,fi) (fk+1,fj+1,fi+1)
//! with only the first two entries of v1/v2 used in 2D.

KOKKOS_INLINE_FUNCTION
void ProlongFCInternalValues(const int m, const int fk, const int fj, const int fi,
                             const bool three_d, const GeomData &geom, const bool cubic,
                             const DvceFaceFld4D<Real> &b,
                             Real (&v1)[4], Real (&v2)[4], Real (&v3)[4]) {
  // Face fluxes Phi = Area*B_n, and the inverse map. `cubic` collapses both to the
  // identity, recovering upstream's expressions exactly.
  auto P1 = [&](const int kk, const int jj, const int ii) {
    return cubic ? b.x1f(m,kk,jj,ii) : geom.Area1(m,kk,jj,ii)*b.x1f(m,kk,jj,ii);
  };
  auto P2 = [&](const int kk, const int jj, const int ii) {
    return cubic ? b.x2f(m,kk,jj,ii) : geom.Area2(m,kk,jj,ii)*b.x2f(m,kk,jj,ii);
  };
  auto P3 = [&](const int kk, const int jj, const int ii) {
    return cubic ? b.x3f(m,kk,jj,ii) : geom.Area3(m,kk,jj,ii)*b.x3f(m,kk,jj,ii);
  };
  // An internal face is strictly inside a coarse cell, so it never lands on r=0 or on
  // the polar axis and its area is positive; the guard is defensive only.
  auto ToB = [&](const Real phi, const Real area) {
    return (cubic || !(area > 0.0)) ? phi : phi/area;
  };

  // Prolongate internal fields in 3D
  if (three_d) {
    Real Uxx  = 0.0, Vyy  = 0.0, Wzz  = 0.0;
    Real Uxyz = 0.0, Vxyz = 0.0, Wxyz = 0.0;
    for (int jj=0; jj<2; jj++) {
      int jsgn = 2*jj - 1;
      int fjj  = fj + jj, fjp = fj + 2*jj;
      for (int ii=0; ii<2; ii++) {
        int isgn = 2*ii - 1;
        int fii = fi + ii, fip = fi + 2*ii;
        Uxx += isgn*(jsgn*(P2(fk  ,fjp,fii) + P2(fk+1,fjp,fii)) +
                          (P3(fk+2,fjj,fii) - P3(fk  ,fjj,fii)));

        Vyy += jsgn*(     (P3(fk+2,fjj,fii) - P3(fk  ,fjj,fii)) +
                     isgn*(P1(fk  ,fjj,fip) + P1(fk+1,fjj,fip)));

        Wzz +=       isgn*(P1(fk+1,fjj,fip) - P1(fk  ,fjj,fip)) +
                     jsgn*(P2(fk+1,fjp,fii) - P2(fk  ,fjp,fii));

        Uxyz += isgn*jsgn*(P1(fk+1,fjj,fip) - P1(fk  ,fjj,fip));
        Vxyz += isgn*jsgn*(P2(fk+1,fjp,fii) - P2(fk  ,fjp,fii));
        Wxyz += isgn*jsgn*(P3(fk+2,fjj,fii) - P3(fk  ,fjj,fii));
      }
    }
    Uxx *= 0.125;  Vyy *= 0.125;  Wzz *= 0.125;
    Uxyz *= 0.0625; Vxyz *= 0.0625; Wxyz *= 0.0625;

    v1[0] = ToB(0.5*(P1(fk  ,fj  ,fi  ) + P1(fk  ,fj  ,fi+2))
              + Uxx - Vxyz - Wxyz, geom.Area1(m,fk  ,fj  ,fi+1));
    v1[1] = ToB(0.5*(P1(fk  ,fj+1,fi  ) + P1(fk  ,fj+1,fi+2))
              + Uxx - Vxyz + Wxyz, geom.Area1(m,fk  ,fj+1,fi+1));
    v1[2] = ToB(0.5*(P1(fk+1,fj  ,fi  ) + P1(fk+1,fj  ,fi+2))
              + Uxx + Vxyz - Wxyz, geom.Area1(m,fk+1,fj  ,fi+1));
    v1[3] = ToB(0.5*(P1(fk+1,fj+1,fi  ) + P1(fk+1,fj+1,fi+2))
              + Uxx + Vxyz + Wxyz, geom.Area1(m,fk+1,fj+1,fi+1));
    v2[0] = ToB(0.5*(P2(fk  ,fj  ,fi  ) + P2(fk  ,fj+2,fi  ))
              + Vyy - Uxyz - Wxyz, geom.Area2(m,fk  ,fj+1,fi  ));
    v2[1] = ToB(0.5*(P2(fk  ,fj  ,fi+1) + P2(fk  ,fj+2,fi+1))
              + Vyy - Uxyz + Wxyz, geom.Area2(m,fk  ,fj+1,fi+1));
    v2[2] = ToB(0.5*(P2(fk+1,fj  ,fi  ) + P2(fk+1,fj+2,fi  ))
              + Vyy + Uxyz - Wxyz, geom.Area2(m,fk+1,fj+1,fi  ));
    v2[3] = ToB(0.5*(P2(fk+1,fj  ,fi+1) + P2(fk+1,fj+2,fi+1))
              + Vyy + Uxyz + Wxyz, geom.Area2(m,fk+1,fj+1,fi+1));
    v3[0] = ToB(0.5*(P3(fk+2,fj  ,fi  ) + P3(fk  ,fj  ,fi  ))
              + Wzz - Uxyz - Vxyz, geom.Area3(m,fk+1,fj  ,fi  ));
    v3[1] = ToB(0.5*(P3(fk+2,fj  ,fi+1) + P3(fk  ,fj  ,fi+1))
              + Wzz - Uxyz + Vxyz, geom.Area3(m,fk+1,fj  ,fi+1));
    v3[2] = ToB(0.5*(P3(fk+2,fj+1,fi  ) + P3(fk  ,fj+1,fi  ))
              + Wzz + Uxyz - Vxyz, geom.Area3(m,fk+1,fj+1,fi  ));
    v3[3] = ToB(0.5*(P3(fk+2,fj+1,fi+1) + P3(fk  ,fj+1,fi+1))
              + Wzz + Uxyz + Vxyz, geom.Area3(m,fk+1,fj+1,fi+1));

  // Prolongate internal fields in 2D
  } else {
    Real tmp1 = 0.25*(P2(fk,fj+2,fi+1) - P2(fk,fj,  fi+1)
                    - P2(fk,fj+2,fi  ) + P2(fk,fj,  fi  ));
    Real tmp2 = 0.25*(P1(fk,fj,  fi  ) - P1(fk,fj,  fi+2)
                    - P1(fk,fj+1,fi  ) + P1(fk,fj+1,fi+2));
    v1[0] = ToB(0.5*(P1(fk,fj,  fi  ) + P1(fk,fj,  fi+2)) + tmp1,
                geom.Area1(m,fk,fj  ,fi+1));
    v1[1] = ToB(0.5*(P1(fk,fj+1,fi  ) + P1(fk,fj+1,fi+2)) + tmp1,
                geom.Area1(m,fk,fj+1,fi+1));
    v2[0] = ToB(0.5*(P2(fk,fj,  fi  ) + P2(fk,fj+2,fi  )) + tmp2,
                geom.Area2(m,fk,fj+1,fi  ));
    v2[1] = ToB(0.5*(P2(fk,fj,  fi+1) + P2(fk,fj+2,fi+1)) + tmp2,
                geom.Area2(m,fk,fj+1,fi+1));
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ProlongFCInternal()
//! \brief regrid-path wrapper: compute the internal-face values and store them directly.

KOKKOS_INLINE_FUNCTION
void ProlongFCInternal(const int m, const int fk, const int fj, const int fi,
                       const bool three_d, const GeomData &geom, const bool cubic,
                       const DvceFaceFld4D<Real> &b) {
  Real v1[4], v2[4], v3[4];
  ProlongFCInternalValues(m, fk, fj, fi, three_d, geom, cubic, b, v1, v2, v3);
  b.x1f(m,fk,fj  ,fi+1) = v1[0];
  b.x1f(m,fk,fj+1,fi+1) = v1[1];
  b.x2f(m,fk,fj+1,fi  ) = v2[0];
  b.x2f(m,fk,fj+1,fi+1) = v2[1];
  if (three_d) {
    b.x1f(m,fk+1,fj  ,fi+1) = v1[2];
    b.x1f(m,fk+1,fj+1,fi+1) = v1[3];
    b.x2f(m,fk+1,fj+1,fi  ) = v2[2];
    b.x2f(m,fk+1,fj+1,fi+1) = v2[3];
    b.x3f(m,fk+1,fj  ,fi  ) = v3[0];
    b.x3f(m,fk+1,fj  ,fi+1) = v3[1];
    b.x3f(m,fk+1,fj+1,fi  ) = v3[2];
    b.x3f(m,fk+1,fj+1,fi+1) = v3[3];
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ProlongFCInternal1D()
//! \brief the 1D internal x1-face, which the callers previously open-coded as
//! `0.5*(x1f(fi) + x1f(fi+2))` at three separate sites.
//!
//! Same Phi recast: in 1D the constraint is d(A1*B1)/dx1 = 0, so A1*B1 is constant across
//! the coarse cell and the internal face must carry that same flux -- which the average
//! of the two bracketing FLUXES gives exactly, while the average of the FIELDS does not.

KOKKOS_INLINE_FUNCTION
void ProlongFCInternal1D(const int m, const int fk, const int fj, const int fi,
                         const GeomData &geom, const bool cubic,
                         const DvceFaceFld4D<Real> &b) {
  if (cubic) {
    b.x1f(m,fk,fj,fi+1) = 0.5*(b.x1f(m,fk,fj,fi) + b.x1f(m,fk,fj,fi+2));
    return;
  }
  Real phi = 0.5*(geom.Area1(m,fk,fj,fi  )*b.x1f(m,fk,fj,fi  )
                + geom.Area1(m,fk,fj,fi+2)*b.x1f(m,fk,fj,fi+2));
  Real area = geom.Area1(m,fk,fj,fi+1);
  b.x1f(m,fk,fj,fi+1) = (area > 0.0) ? phi/area
                      : 0.5*(b.x1f(m,fk,fj,fi) + b.x1f(m,fk,fj,fi+2));
  return;
}

template <int NGHOST>
KOKKOS_INLINE_FUNCTION
Real ProlongInterpolation(const int m, const int v, int k, int j, int i,
                            const int nx1, const int nx2, const int nx3,
                            const bool offsetk, const bool offsetj, const bool offseti,
                        const DvceArray5D<Real> &ca, const DualArray3D<Real> &weights) {
  // interpolated value at new grid point
  Real ivals = 0;

  for (int kk=0; kk<NGHOST+1; kk++) {
    for (int jj=0; jj<NGHOST+1; jj++) {
      for (int ii=0; ii<NGHOST+1; ii++) {
        int wghti = (offseti) ? NGHOST-ii : ii;
        int wghtj = (offsetj) ? NGHOST-jj : jj;
        int wghtk = (offsetk) ? NGHOST-kk : kk;
        ivals += weights.d_view(wghtk,wghtj,wghti)*ca(m,v,
                    k-NGHOST/2+kk,j-NGHOST/2+jj,i-NGHOST/2+ii);
      }
    }
  }

  return ivals;
}

//----------------------------------------------------------------------------------------
//! \fn HighOrderProlongCC()
//! \brief high-order prolongation operator for cell-centered variables

template <int NGHOST>
KOKKOS_INLINE_FUNCTION
void HighOrderProlongCC(const int m, const int v, const int k, const int j, const int i,
               const int fk, const int fj, const int fi, const int nx1, const int nx2,
               const int nx3, const DvceArray5D<Real> &ca, const DvceArray5D<Real> &a,
               const DualArray3D<Real> &weights) {
  // stencil size for interpolator
  a(m,v,fk  ,fj  ,fi  ) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1, nx2, nx3,
                                                        false,false,false, ca, weights);
  a(m,v,fk  ,fj  ,fi+1) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1, nx2, nx3,
                                                        false,false, true, ca, weights);
  a(m,v,fk  ,fj+1,fi  ) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1, nx2, nx3,
                                                        false, true,false, ca, weights);
  a(m,v,fk  ,fj+1,fi+1) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1, nx2, nx3,
                                                        false, true, true, ca, weights);
  a(m,v,fk+1,fj  ,fi  ) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1, nx2, nx3,
                                                         true,false,false, ca, weights);
  a(m,v,fk+1,fj  ,fi+1) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1, nx2, nx3,
                                                         true,false, true, ca, weights);
  a(m,v,fk+1,fj+1,fi  ) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1, nx2, nx3,
                                                         true, true,false, ca, weights);
  a(m,v,fk+1,fj+1,fi+1) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1, nx2, nx3,
                                                         true, true, true, ca, weights);
  return;
}

#endif // MESH_PROLONGATION_HPP_

