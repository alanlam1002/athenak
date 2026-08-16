#ifndef COORDINATES_MESH_GEOMETRY_HPP_
#define COORDINATES_MESH_GEOMETRY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mesh_geometry.hpp
//! \brief Defines GeomData (POD struct of precomputed, factored per-direction metric
//! arrays, captured by value inside Kokkos kernels) and MeshGeometry (the class that
//! owns/builds GeomData for a MeshBlockPack). This is a grid-geometry construct --
//! face areas, cell volumes, edge lengths, reconstruction centroids, and geometric
//! source-term coefficients -- and is entirely independent of the relativistic metric
//! machinery in coordinates.hpp (Coordinates/CoordData), which remains untouched.
//!
//! Design: for every coordinate system supported (cartesian, cylindrical,
//! cylindrical_axisym, spherical_polar), every geometric quantity needed by the hot
//! loops (flux divergence, CT curl, reconstruction, geometric source terms) factors
//! into a product of one function of i, one of j, and one of k. GeomData stores these
//! small per-direction factor arrays (tiny compared to the full 4D hydro/MHD state,
//! and cheap to keep resident in cache) instead of full (m,k,j,i) arrays, and exposes
//! a *uniform* accessor interface (Area1/2/3, Vol, Len1/2/3) used identically by every
//! hot-loop kernel regardless of coordinate system -- Cartesian is not special-cased at
//! the kernel level, only at the array-construction level (see geometry_cartesian.cpp).
//!
//! Adding a new orthogonal coordinate system later requires writing exactly one new
//! geometry factory function (see geometry_cartesian.cpp for the template) that fills
//! these arrays; no hot-loop kernel needs to change.

#include "athena.hpp"

// Forward declarations
class MeshBlockPack;
class ParameterInput;

//----------------------------------------------------------------------------------------
//! \struct GeomData
//! \brief POD struct of factored per-direction geometry arrays, passed BY VALUE into
//! Kokkos kernels (mirrors the existing CoordData-by-value pattern used for GR metric
//! parameters, see coordinates.hpp -- DvceArray2D is itself a lightweight, reference-
//! counted Kokkos::View handle, so copying GeomData into a kernel lambda is cheap).
//! All arrays are indexed (m, index), where index is the SAME absolute cell/face index
//! used by the corresponding hydro/MHD state array (e.g. Area1(m,k,j,i) uses the same
//! (m,k,j,i) as flx1(m,n,k,j,i); Vol(m,k,j,i) uses the same (m,k,j,i) as u0(m,n,k,j,i)).
//!
//! Sizing convention (matches DvceFaceFld4D/DvceEdgeFld4D in athena.hpp exactly):
//!   - a "cell" (width-type) factor for direction d has size ncells_d
//!   - a "face" (face-valued) factor for direction d has size ncells_d+1
//! e.g. a1i is a face factor (size ncells1+1, matches x1f/flx1's i-extent), while
//! a1j, a1k are cell factors (size ncells2, ncells3, matching flx1's j,k-extent).
//! This factorization was verified against every Face*Area/Edge*Length formula in
//! old Athena++'s coordinates.cpp/cylindrical.cpp/spherical_polar.cpp before being
//! adopted here (see DEVELOPMENT.md Task A2 log).

//----------------------------------------------------------------------------------------
//! \enum PlmCoefIdx
//! \brief component indices into GeomData::plm_c1/c2/c3 (see the doc comment there).
//! pF/pB scale the forward/backward differences; cf/cb/cc are the limiter's
//! centroid-offset weights (cc == cf+cb-2, precomputed); pP/pM are the centroid-to-face
//! offsets as a fraction of the cell width. pM is stored rather than derived as 1-pP so
//! that the two offsets stay exactly what the position formula produced.

enum PlmCoefIdx {IPLM_PF=0, IPLM_PB, IPLM_CF, IPLM_CB, IPLM_CC, IPLM_PP, IPLM_PM,
                 NPLM_COEF};

//----------------------------------------------------------------------------------------
//! \struct PlmCoeff
//! \brief the seven non-uniform PLM limiter factors for one cell, as plain Reals.

struct PlmCoeff {
  Real pF, pB, cf, cb, cc, pP, pM;
};

//----------------------------------------------------------------------------------------
//! \fn MakePlmCoeff()
//! \brief computes the PLM factors for the cell bounded by faces xf_i/xf_ip1 with
//! centroids x_im1/x_i/x_ip1. This is the ONLY place the position-to-factor formula
//! lives: mesh_geometry.cpp calls it once per cell at construction to fill plm_c1/c2/c3,
//! and the unit test calls it directly, so the tested math and the production math cannot
//! drift apart. Pure scalar arithmetic (no Views), hence safe on host and device.

KOKKOS_INLINE_FUNCTION
PlmCoeff MakePlmCoeff(const Real x_im1, const Real x_i, const Real x_ip1,
                      const Real xf_i, const Real xf_ip1) {
  Real dx1f     = xf_ip1 - xf_i;   // plain width of cell i
  Real dx1v_fwd = x_ip1 - x_i;     // forward centroid spacing
  Real dx1v_bwd = x_i - x_im1;     // backward centroid spacing
  PlmCoeff c;
  c.pF = dx1f/dx1v_fwd;
  c.pB = dx1f/dx1v_bwd;
  c.cf = dx1v_fwd/(xf_ip1 - x_i);
  c.cb = dx1v_bwd/(x_i - xf_i);
  c.cc = c.cf + c.cb - 2.0;
  c.pP = (xf_ip1 - x_i)/dx1f;
  c.pM = (x_i - xf_i)/dx1f;
  return c;
}

//----------------------------------------------------------------------------------------
//! \fn UniformPlmCoeff()
//! \brief the exact uniform-Cartesian factors. Used for the outermost ghost cells, where
//! the i-1/i+1 centroid stencil MakePlmCoeff needs runs off the end of the array. Those
//! cells are never reconstructed (the loop covers [is-1,ie+1], and nghost>=2), so this is
//! only about not leaving NaNs in memory.

KOKKOS_INLINE_FUNCTION
PlmCoeff UniformPlmCoeff() {
  return PlmCoeff{1.0, 1.0, 2.0, 2.0, 2.0, 0.5, 0.5};
}

struct GeomData {
  // Area1(m,k,j,i) = a1i(m,i) * a1j(m,j) * a1k(m,k)
  //   a1i: face factor, size ncells1+1 (own-direction, e.g. R_face, r_face^2)
  //   a1j, a1k: cell factors, size ncells2, ncells3 (transverse widths)
  DvceArray2D<Real> a1i, a1j, a1k;
  // Area2(m,k,j,i) = a2i(m,i) * a2j(m,j) * a2k(m,k)
  //   a2j: face factor, size ncells2+1 (own-direction)
  //   a2i, a2k: cell factors, size ncells1, ncells3 (transverse widths)
  DvceArray2D<Real> a2i, a2j, a2k;
  // Area3(m,k,j,i) = a3i(m,i) * a3j(m,j) * a3k(m,k)
  //   a3k: face factor, size ncells3+1 (own-direction)
  //   a3i, a3j: cell factors, size ncells1, ncells2 (transverse widths)
  DvceArray2D<Real> a3i, a3j, a3k;
  // Vol(m,k,j,i) = vi(m,i) * vj(m,j) * vk(m,k)   (all cell factors)
  DvceArray2D<Real> vi, vj, vk;
  // Len1(m,k,j,i) = l1i(m,i) * l1j(m,j) * l1k(m,k)
  //   l1i: cell factor, size ncells1 (own-direction, always plain width -- edges never
  //        need a metric-weighted own-direction integral in an orthogonal coord system)
  //   l1j, l1k: face factors, size ncells2+1, ncells3+1 (transverse)
  DvceArray2D<Real> l1i, l1j, l1k;
  // Len2(m,k,j,i) = l2i(m,i) * l2j(m,j) * l2k(m,k)
  //   l2j: cell factor, size ncells2 (own-direction)
  //   l2i, l2k: face factors, size ncells1+1, ncells3+1 (transverse)
  DvceArray2D<Real> l2i, l2j, l2k;
  // Len3(m,k,j,i) = l3i(m,i) * l3j(m,j) * l3k(m,k)
  //   l3k: cell factor, size ncells3 (own-direction)
  //   l3i, l3j: face factors, size ncells1+1, ncells2+1 (transverse)
  DvceArray2D<Real> l3i, l3j, l3k;
  // volumetric-centroid cell-center positions (NOT the arithmetic midpoint in
  // cylindrical/spherical, see Mignone 2014 eq. 17), used by reconstruction (Task B6/B7).
  // Cell factors, size ncells1, ncells2, ncells3.
  DvceArray2D<Real> x1v, x2v, x3v;
  // face positions (Task B6), same LeftEdgeX() formula in every coordinate system since
  // face position in INDEX space is coordinate-independent (only the volumetric centroid
  // x1v above differs by system). Needed alongside x1v/x2v/x3v for the non-uniform PLM
  // limiter (Mignone 2014 eq. 33/37), which requires both the centroid-to-centroid
  // spacing AND the centroid-to-face offset -- see plm.hpp. Face factors, size
  // ncells1+1, ncells2+1, ncells3+1.
  DvceArray2D<Real> xf1, xf2, xf3;
  // geometric source-term coefficients (Task C1/C2), Delta-A/Delta-V ratios; zero for
  // cartesian. Cell factor, size ncells1 (indexed by i, the radial-like direction for
  // all three curvilinear systems). NOTE: spherical_polar's theta-momentum source term
  // needs a THIRD, j-indexed coefficient in addition to src1/src2 -- that array is added
  // when Task C2 (spherical geometric source terms) is implemented; not needed/present
  // for Task A2 (cartesian only, both src1 and src2 are simply zero everywhere).
  DvceArray2D<Real> src1, src2;
  // spherical_polar's theta-momentum geometric source coefficients (Task C2), j-indexed,
  // cell factor size ncells2. Not filled (left default-constructed/empty) for cartesian/
  // cylindrical/cylindrical_axisym, which never dispatch to the kernel that reads these.
  // src1_j = (sin th_f,+ - sin th_f,-)/|cos th_j - cos th_j+1| (old Athena++'s
  // coord_src1_j_, numerically identical to coord_src3_j_ there, so only one array is
  // kept here); src2_j = src1_j/(sin th_f,- + sin th_f,+) (old Athena++'s coord_src2_j_).
  DvceArray2D<Real> src1_j, src2_j;
  // CFL-purpose cell widths (Task B5, C4 fix), evaluated at the volumetric CENTROID --
  // distinct from Len2/Len3 above, which are face-valued edge lengths for CT. Matches
  // old Athena++'s CenterWidth2/3 exactly (e.g. spherical_polar.cpp:319-335):
  // CenterWidth2 = x1v(i)*dtheta(j) for spherical/cylindrical (angular direction weighted
  // by the radial centroid); CenterWidth3 = x1v(i)*sin(x2v(j))*dphi(k) for spherical only
  // (cylindrical/axisym/cartesian have a flat, unweighted x3 or x2 direction there -- see
  // each geometry factory for the system-specific values). Factored the same way as
  // Area/Vol/Len: CenterWidth2(m,k,j,i)=cw2i(m,i)*cw2j(m,j),
  // CenterWidth3(m,k,j,i)=cw3i(m,i)*cw3j(m,j)*cw3k(m,k). Cell factors (sizes ncells1,
  // ncells2, ncells3 respectively).
  DvceArray2D<Real> cw2i, cw2j;
  DvceArray2D<Real> cw3i, cw3j, cw3k;
  // PPM4/PPMX interpolation weights and overshoot ratios (Task B7), x1-direction ONLY --
  // x2/x3 keep AthenaK's existing hardcoded uniform-Cartesian PPM4/PPMX formula
  // unchanged (see ppm.hpp's original PPM4/PPMX overloads), since none of the required
  // layouts or this project's general cylindrical/spherical support need a curvilinear
  // x2/x3 (cylindrical's phi/z and axisym's z are flat; spherical's theta IS curvilinear
  // in principle -- old Athena++ generalizes it too -- but is explicitly out of scope
  // here since it needs a genuinely different, non-power-law derivation; the required
  // 1D-radial spherical layout never resolves theta anyway). ppm_c1i..c4i are the
  // Mignone (2014) eq. B.9 (cylindrical/axisym, m_coord=1) / B.14 (spherical, m_coord=2)
  // 4-point Lagrange interpolation weights for the face AT index i (face-indexed, size
  // ncells1+1), reducing to the uniform-Cartesian (-1/12,7/12,7/12,-1/12) (Mignone eq.
  // B.4) for coord=cartesian. ppm_hpi/hmi are the Mignone eq. 48 overshoot-limiter
  // ratios (cell-indexed, size ncells1), reducing to the constant 2.0 used by the
  // original Colella-Woodward/Colella-Sekora limiters for coord=cartesian.
  DvceArray2D<Real> ppm_c1i, ppm_c2i, ppm_c3i, ppm_c4i;
  DvceArray2D<Real> ppm_hpi, ppm_hmi;
  // Precomputed non-uniform PLM limiter factors, one array per direction, indexed
  // (m, component, cell) with the component taken from the PlmCoefIdx enum below.
  //
  // WHY these exist: the generalized Mignone (2014) eq. 33/37 limiter needs SEVEN
  // position-dependent ratios, and computing them from raw centroid/face positions inside
  // the kernel costs 7 divisions per cell PER VARIABLE PER DIRECTION, versus 1 for
  // upstream's uniform-spacing PLM. Measured, that made every run -- Cartesian
  // included -- 23.5% slower (see DEVELOPMENT.md "Performance and GPU status").
  // Every one of those divisors depends only on POSITION, so they are hoisted and
  // computed once at
  // construction, exactly as Task B7 already does for the PPM coefficients. The kernel is
  // then back to 1 division, for curvilinear and Cartesian alike -- no Cartesian
  // fast path and no per-cell coordinate branch is needed or wanted.
  //
  // Stored as one rank-3 array per direction rather than seven rank-2 arrays so that the
  // kernel closure grows by 3 Kokkos::View handles instead of 21 (GeomData is captured by
  // value into every reconstruction launch; see the note on capture size in
  // DEVELOPMENT.md). LayoutRight keeps the fastest index -- the cell index -- contiguous,
  // so each component read is still unit-stride across a warp.
  //
  // For coord=cartesian every entry is exactly (1,1,2,2,2,0.5,0.5), which makes the
  // limiter reduce ALGEBRAICALLY EXACTLY to upstream's dqm = dq2/(dql+dqr) with +/-1/2
  // face offsets -- verified by hand, and these constants are exactly representable, so
  // Cartesian arithmetic is if anything closer to upstream's than the previous
  // position-based form was.
  DvceArray3D<Real> plm_c1, plm_c2, plm_c3;
  // True when a direction's spacing is uniform, i.e. every plm_c* entry is exactly the
  // uniform-Cartesian tuple. Set by BuildPlmFactors(), which also SNAPS the stored
  // factors to the exact constants in that case (they are otherwise computed from two
  // different subtractions and can land an ulp off 1.0 on a perfectly uniform grid, so
  // an equality test alone would spuriously report "non-uniform").
  //
  // This lets the limiter take upstream's original, much cheaper expression where it is
  // exactly equivalent. Note this is deliberately a question about SPACING, not about
  // CoordinateGeneral: the kernel never branches on coordinate system (the confinement
  // principle still holds), and the flags are per-direction, so cylindrical/spherical
  // also take the fast path in their genuinely flat phi/z directions -- a
  // "cartesian-only" fast path would have helped Cartesian alone.
  //
  // The branch is grid-uniform, hence perfectly predicted on CPU and warp-coherent on
  // GPU -- the same argument recon.hpp already relies on for its runtime method dispatch.
  bool plm_uniform1, plm_uniform2, plm_uniform3;
  // Same idea for the x1 PPM coefficients: true when they are exactly the flat
  // (-1/12,7/12,7/12,-1/12) weights with hp=hm=2, letting recon.hpp call upstream's
  // PPM4/PPMX directly. (x2/x3 always use upstream's PPM unconditionally -- the
  // curvilinear PPM generalization is x1-only -- so they need no flag.)
  bool ppm_uniform1;

  KOKKOS_INLINE_FUNCTION
  Real Area1(int m, int k, int j, int i) const { return a1i(m,i)*a1j(m,j)*a1k(m,k); }
  KOKKOS_INLINE_FUNCTION
  Real Area2(int m, int k, int j, int i) const { return a2i(m,i)*a2j(m,j)*a2k(m,k); }
  KOKKOS_INLINE_FUNCTION
  Real Area3(int m, int k, int j, int i) const { return a3i(m,i)*a3j(m,j)*a3k(m,k); }
  KOKKOS_INLINE_FUNCTION
  Real Vol(int m, int k, int j, int i) const { return vi(m,i)*vj(m,j)*vk(m,k); }
  KOKKOS_INLINE_FUNCTION
  Real Len1(int m, int k, int j, int i) const { return l1i(m,i)*l1j(m,j)*l1k(m,k); }
  KOKKOS_INLINE_FUNCTION
  Real Len2(int m, int k, int j, int i) const { return l2i(m,i)*l2j(m,j)*l2k(m,k); }
  KOKKOS_INLINE_FUNCTION
  Real Len3(int m, int k, int j, int i) const { return l3i(m,i)*l3j(m,j)*l3k(m,k); }
  KOKKOS_INLINE_FUNCTION
  Real CenterWidth2(int m, int k, int j, int i) const { return cw2i(m,i)*cw2j(m,j); }
  KOKKOS_INLINE_FUNCTION
  Real CenterWidth3(int m, int k, int j, int i) const {
    return cw3i(m,i)*cw3j(m,j)*cw3k(m,k);
  }
};

//----------------------------------------------------------------------------------------
//! \class MeshGeometry
//! \brief owns and builds GeomData for a MeshBlockPack. Deliberately separate from the
//! GR/SR Coordinates class (coordinates.hpp), which is untouched by this project. Built
//! by MeshBlockPack::AddGeometry(), called immediately after AddMeshBlocks() (both at
//! initial construction and at every SMR regrid -- though SMR+curvilinear is currently
//! guarded against, see Mesh::ValidateCoordGeneral()), since it needs mb_size to be
//! already populated.

class MeshGeometry {
 public:
  MeshGeometry(ParameterInput *pin, MeshBlockPack *ppack);
  ~MeshGeometry() = default;

  GeomData geom_data;

 private:
  MeshBlockPack* pmy_pack;
};

#endif // COORDINATES_MESH_GEOMETRY_HPP_
