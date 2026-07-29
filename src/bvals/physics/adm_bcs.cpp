//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file adm_bcs.cpp
//! \brief Physical (non-block, non-periodic) boundary conditions for
//! MeshBlockPack::padm->u_adm under CFC (no z4c free evolution -- z4c's own BCs,
//! z4c_bcs.cpp, cover u_adm implicitly via Z4cToADM when z4c is active). Without this,
//! u_adm's ghost cells at a genuine physical domain edge were never touched after the
//! problem generator's own t=0 initialization (CFC_RestADM/SendADM/RecvADM/ProlongADM,
//! cfc.cpp, only exchange inter-MeshBlock/periodic/AMR-refined neighbors -- there is no
//! neighbor at a physical boundary for RecvAndUnpackCC to pull from) -- silently frozen
//! at their t=0 value even as the interior metric evolves. Since dyn_grmhd's geometric
//! source terms differentiate alpha/g_dd/beta_u right up to the domain edge, a stale
//! ghost value there contaminates those derivatives with a spurious kink between the
//! (evolving) interior and the (frozen) ghost region.
//!
//! Every channel decays toward its known flat-space value (alpha, psi4, diag(g_dd) -> 1;
//! off-diag(g_dd), vK_dd, beta_u -> 0) with a channel-dependent leading order: n=1 for
//! alpha/psi4/g_dd (mass monopole, ~M/r) and n=2 for vK_dd/beta_u (~1/r^2, the next
//! order for extrinsic curvature/shift around a non-boosted, asymptotically-flat
//! source). At each ghost depth, the deviation from flat space is extrapolated from the
//! single nearest interior (domain-boundary) cell via this power law -- not chained
//! ghost-to-ghost -- using the true 3D coordinate radius from the origin (mirroring the
//! pseudo-radial convention z4c's own Sommerfeld BC, z4c_Sbc.cpp, already uses; unlike
//! that Sommerfeld BC, which injects a radiative-outgoing RHS at the boundary-adjacent
//! *interior* point for evolved z4c fields, this is a ghost-cell fill -- ADM quantities
//! under CFC have no evolution equation/RHS of their own, being pure algebraic outputs
//! of the elliptic solve every stage).
//!
//! Reflection parity per channel/axis mirrors z4c_bcs.cpp's explicit enumeration
//! exactly: a rank-2 tensor component (g_dd, vK_dd) flips sign iff exactly one of its
//! two indices is aligned with the reflected axis (e.g. g_xy is odd under an x1
//! reflection, g_xx/g_yy/g_zz are even); a vector component (beta_u) flips iff its own
//! index is; scalars (alpha, psi4) never flip.
//!
//! ADMBCsImpl takes the active-zone index bounds (is/ie/js/je/ks/ke), the interior
//! cell counts (nx1/nx2/nx3), and the total per-axis extents (n1/n2/n3) as explicit
//! parameters so the same implementation is shared between the fine array (ADMBCs)
//! and the coarse array (ADMBCsCoarse, called before ProlongateCC so the prolongation
//! stencil reads valid coarse ghost data at a physical boundary, mirroring
//! Z4cBCsCoarse). Since the 1/r^order falloff needs the true physical cell-center
//! position via CellCenterX, nx1/nx2/nx3 must be the coarse interior counts for a
//! coarse-array call, or the computed radius would be wrong by the refinement factor.

#include <cmath>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"

namespace {

//! \brief Per-channel description used by both the falloff and reflection-parity
//! logic below. kind: 0=scalar (alpha, psi4), 1=vector (beta_u), 2=rank-2 tensor
//! (g_dd, vK_dd). row/col: tensor index/indices (row only, for vectors; unused for
//! scalars). flat: the channel's flat-space (Minkowski) value. order: leading
//! asymptotic falloff power n in (f - flat) ~ 1/r^n.
struct ADMChannelInfo {
  int kind;
  int row, col;
  Real flat;
  int order;
};

KOKKOS_INLINE_FUNCTION
ADMChannelInfo GetADMChannelInfo(int v) {
  using adm::ADM;
  // (row,col) for the 6 SYM2 tensor channels, in I_ADM_GXX..GZZ/I_ADM_KXX..KZZ order.
  constexpr int kRow[6] = {0, 0, 0, 1, 1, 2};
  constexpr int kCol[6] = {0, 1, 2, 1, 2, 2};
  if (v <= ADM::I_ADM_GZZ) {                          // g_dd: 0..5
    return {2, kRow[v], kCol[v], (kRow[v] == kCol[v]) ? 1.0 : 0.0, 1};
  } else if (v <= ADM::I_ADM_KZZ) {                   // vK_dd: 6..11
    int vv = v - ADM::I_ADM_KXX;
    return {2, kRow[vv], kCol[vv], 0.0, 2};
  } else if (v == ADM::I_ADM_PSI4) {
    return {0, -1, -1, 1.0, 1};
  } else if (v == ADM::I_ADM_ALPHA) {
    return {0, -1, -1, 1.0, 1};
  } else {                                            // beta_u: BETAX..BETAZ
    return {1, v - ADM::I_ADM_BETAX, -1, 0.0, 2};
  }
}

KOKKOS_INLINE_FUNCTION
bool ChannelFlipsAtAxis(const ADMChannelInfo &c, int axis) {
  if (c.kind == 0) return false;                      // scalar: never flips
  if (c.kind == 1) return (c.row == axis);             // vector: flips iff aligned
  return (c.row == axis) != (c.col == axis);           // tensor: flips iff exactly one
}

//----------------------------------------------------------------------------------------
//! \fn void ADMBCsImpl(MeshBlockPack *ppack, DvceArray5D<Real> u0, int is, int ie,
//!                      int js, int je, int ks, int ke, int nx1, int nx2, int nx3,
//!                      int n1, int n2, int n3)
//! \brief shared implementation behind ADMBCs (fine) and ADMBCsCoarse. is/ie/js/je/
//! ks/ke/nx1/nx2/nx3/n1/n2/n3 select fine vs. coarse array index bounds/extents.

void ADMBCsImpl(MeshBlockPack *ppack, DvceArray5D<Real> u0, int is, int ie, int js,
                int je, int ks, int ke, int nx1, int nx2, int nx3, int n1, int n2,
                int n3) {
  auto &pm = ppack->pmesh;
  auto &indcs = ppack->pmesh->mb_indcs;
  auto &size = ppack->pmb->mb_size;
  int &ng = indcs.ng;
  auto &mb_bcs = ppack->pmb->mb_bcs;

  int nvar = u0.extent_int(1);
  int nmb = ppack->nmb_thispack;

  if (pm->mesh_bcs[BoundaryFace::inner_x1] != BoundaryFlag::periodic) {
    par_for("adm_bc_x1", DevExeSpace(), 0, (nmb-1), 0, (nvar-1), 0, (n3-1), 0, (n2-1),
    KOKKOS_LAMBDA(int m, int v, int k, int j) {
      ADMChannelInfo c = GetADMChannelInfo(v);
      Real &x1min = size.d_view(m).x1min; Real &x1max = size.d_view(m).x1max;
      Real &x2min = size.d_view(m).x2min; Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
      Real &x3min = size.d_view(m).x3min; Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

      switch (mb_bcs.d_view(m,BoundaryFace::inner_x1)) {
        case BoundaryFlag::reflect: {
          bool flip = ChannelFlipsAtAxis(c, 0);
          for (int i=0; i<ng; ++i) {
            Real val = u0(m,v,k,j,is+i);
            u0(m,v,k,j,is-i-1) = flip ? -val : val;
          }
          break;
        }
        case BoundaryFlag::outflow: case BoundaryFlag::diode:
        case BoundaryFlag::vacuum: case BoundaryFlag::inflow:
        case BoundaryFlag::user: {
          Real x1_i = CellCenterX(is-is, nx1, x1min, x1max);
          Real r_i = Kokkos::sqrt(SQR(x1_i) + SQR(x2v) + SQR(x3v));
          Real f_i = u0(m,v,k,j,is) - c.flat;
          for (int i=0; i<ng; ++i) {
            Real x1_g = CellCenterX(is-i-1-is, nx1, x1min, x1max);
            Real r_g = Kokkos::sqrt(SQR(x1_g) + SQR(x2v) + SQR(x3v));
            Real ratio = Kokkos::pow(r_i/(r_g + 1.0e-30), c.order);
            u0(m,v,k,j,is-i-1) = c.flat + f_i*ratio;
          }
          break;
        }
        default: break;
      }

      switch (mb_bcs.d_view(m,BoundaryFace::outer_x1)) {
        case BoundaryFlag::reflect: {
          bool flip = ChannelFlipsAtAxis(c, 0);
          for (int i=0; i<ng; ++i) {
            Real val = u0(m,v,k,j,ie-i);
            u0(m,v,k,j,ie+i+1) = flip ? -val : val;
          }
          break;
        }
        case BoundaryFlag::outflow: case BoundaryFlag::diode:
        case BoundaryFlag::vacuum: case BoundaryFlag::inflow:
        case BoundaryFlag::user: {
          Real x1_i = CellCenterX(ie-is, nx1, x1min, x1max);
          Real r_i = Kokkos::sqrt(SQR(x1_i) + SQR(x2v) + SQR(x3v));
          Real f_i = u0(m,v,k,j,ie) - c.flat;
          for (int i=0; i<ng; ++i) {
            Real x1_g = CellCenterX(ie+i+1-is, nx1, x1min, x1max);
            Real r_g = Kokkos::sqrt(SQR(x1_g) + SQR(x2v) + SQR(x3v));
            Real ratio = Kokkos::pow(r_i/(r_g + 1.0e-30), c.order);
            u0(m,v,k,j,ie+i+1) = c.flat + f_i*ratio;
          }
          break;
        }
        default: break;
      }
    });
  }
  if (pm->one_d) return;

  if (pm->mesh_bcs[BoundaryFace::inner_x2] != BoundaryFlag::periodic) {
    par_for("adm_bc_x2", DevExeSpace(), 0, (nmb-1), 0, (nvar-1), 0, (n3-1), 0, (n1-1),
    KOKKOS_LAMBDA(int m, int v, int k, int i) {
      ADMChannelInfo c = GetADMChannelInfo(v);
      Real &x1min = size.d_view(m).x1min; Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
      Real &x2min = size.d_view(m).x2min; Real &x2max = size.d_view(m).x2max;
      Real &x3min = size.d_view(m).x3min; Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

      switch (mb_bcs.d_view(m,BoundaryFace::inner_x2)) {
        case BoundaryFlag::reflect: {
          bool flip = ChannelFlipsAtAxis(c, 1);
          for (int j=0; j<ng; ++j) {
            Real val = u0(m,v,k,js+j,i);
            u0(m,v,k,js-j-1,i) = flip ? -val : val;
          }
          break;
        }
        case BoundaryFlag::outflow: case BoundaryFlag::diode:
        case BoundaryFlag::vacuum: case BoundaryFlag::inflow:
        case BoundaryFlag::user: {
          Real x2_i = CellCenterX(js-js, nx2, x2min, x2max);
          Real r_i = Kokkos::sqrt(SQR(x1v) + SQR(x2_i) + SQR(x3v));
          Real f_i = u0(m,v,k,js,i) - c.flat;
          for (int j=0; j<ng; ++j) {
            Real x2_g = CellCenterX(js-j-1-js, nx2, x2min, x2max);
            Real r_g = Kokkos::sqrt(SQR(x1v) + SQR(x2_g) + SQR(x3v));
            Real ratio = Kokkos::pow(r_i/(r_g + 1.0e-30), c.order);
            u0(m,v,k,js-j-1,i) = c.flat + f_i*ratio;
          }
          break;
        }
        default: break;
      }

      switch (mb_bcs.d_view(m,BoundaryFace::outer_x2)) {
        case BoundaryFlag::reflect: {
          bool flip = ChannelFlipsAtAxis(c, 1);
          for (int j=0; j<ng; ++j) {
            Real val = u0(m,v,k,je-j,i);
            u0(m,v,k,je+j+1,i) = flip ? -val : val;
          }
          break;
        }
        case BoundaryFlag::outflow: case BoundaryFlag::diode:
        case BoundaryFlag::vacuum: case BoundaryFlag::inflow:
        case BoundaryFlag::user: {
          Real x2_i = CellCenterX(je-js, nx2, x2min, x2max);
          Real r_i = Kokkos::sqrt(SQR(x1v) + SQR(x2_i) + SQR(x3v));
          Real f_i = u0(m,v,k,je,i) - c.flat;
          for (int j=0; j<ng; ++j) {
            Real x2_g = CellCenterX(je+j+1-js, nx2, x2min, x2max);
            Real r_g = Kokkos::sqrt(SQR(x1v) + SQR(x2_g) + SQR(x3v));
            Real ratio = Kokkos::pow(r_i/(r_g + 1.0e-30), c.order);
            u0(m,v,k,je+j+1,i) = c.flat + f_i*ratio;
          }
          break;
        }
        default: break;
      }
    });
  }
  if (pm->two_d) return;

  if (pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::periodic) return;
  par_for("adm_bc_x3", DevExeSpace(), 0, (nmb-1), 0, (nvar-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int v, int j, int i) {
    ADMChannelInfo c = GetADMChannelInfo(v);
    Real &x1min = size.d_view(m).x1min; Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
    Real &x2min = size.d_view(m).x2min; Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
    Real &x3min = size.d_view(m).x3min; Real &x3max = size.d_view(m).x3max;

    switch (mb_bcs.d_view(m,BoundaryFace::inner_x3)) {
      case BoundaryFlag::reflect: {
        bool flip = ChannelFlipsAtAxis(c, 2);
        for (int k=0; k<ng; ++k) {
          Real val = u0(m,v,ks+k,j,i);
          u0(m,v,ks-k-1,j,i) = flip ? -val : val;
        }
        break;
      }
      case BoundaryFlag::outflow: case BoundaryFlag::diode:
      case BoundaryFlag::vacuum: case BoundaryFlag::inflow:
      case BoundaryFlag::user: {
        Real x3_i = CellCenterX(ks-ks, nx3, x3min, x3max);
        Real r_i = Kokkos::sqrt(SQR(x1v) + SQR(x2v) + SQR(x3_i));
        Real f_i = u0(m,v,ks,j,i) - c.flat;
        for (int k=0; k<ng; ++k) {
          Real x3_g = CellCenterX(ks-k-1-ks, nx3, x3min, x3max);
          Real r_g = Kokkos::sqrt(SQR(x1v) + SQR(x2v) + SQR(x3_g));
          Real ratio = Kokkos::pow(r_i/(r_g + 1.0e-30), c.order);
          u0(m,v,ks-k-1,j,i) = c.flat + f_i*ratio;
        }
        break;
      }
      default: break;
    }

    switch (mb_bcs.d_view(m,BoundaryFace::outer_x3)) {
      case BoundaryFlag::reflect: {
        bool flip = ChannelFlipsAtAxis(c, 2);
        for (int k=0; k<ng; ++k) {
          Real val = u0(m,v,ke-k,j,i);
          u0(m,v,ke+k+1,j,i) = flip ? -val : val;
        }
        break;
      }
      case BoundaryFlag::outflow: case BoundaryFlag::diode:
      case BoundaryFlag::vacuum: case BoundaryFlag::inflow:
      case BoundaryFlag::user: {
        Real x3_i = CellCenterX(ke-ks, nx3, x3min, x3max);
        Real r_i = Kokkos::sqrt(SQR(x1v) + SQR(x2v) + SQR(x3_i));
        Real f_i = u0(m,v,ke,j,i) - c.flat;
        for (int k=0; k<ng; ++k) {
          Real x3_g = CellCenterX(ke+k+1-ks, nx3, x3min, x3max);
          Real r_g = Kokkos::sqrt(SQR(x1v) + SQR(x2v) + SQR(x3_g));
          Real ratio = Kokkos::pow(r_i/(r_g + 1.0e-30), c.order);
          u0(m,v,ke+k+1,j,i) = c.flat + f_i*ratio;
        }
        break;
      }
      default: break;
    }
  });

  return;
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValues::ADMBCs()
//! \brief Apply physical BCs to pmbp->padm->u_adm (fine array) -- see file doc comment.

void MeshBoundaryValues::ADMBCs(MeshBlockPack *ppack, DvceArray5D<Real> u0) {
  auto &indcs = ppack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  ADMBCsImpl(ppack, u0, indcs.is, indcs.ie, indcs.js, indcs.je, indcs.ks, indcs.ke,
             indcs.nx1, indcs.nx2, indcs.nx3, n1, n2, n3);
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValues::ADMBCsCoarse()
//! \brief Coarse-array counterpart of ADMBCs -- must run before ProlongateCC so the
//! prolongation stencil reads valid coarse ghost data at a physical boundary
//! (mirrors Z4cBCsCoarse). Operates on padm->coarse_u_adm.

void MeshBoundaryValues::ADMBCsCoarse(MeshBlockPack *ppack,
                                      DvceArray5D<Real> coarse_u0) {
  auto &indcs = ppack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int cn1 = indcs.cnx1 + 2*ng;
  int cn2 = (indcs.cnx2 > 1) ? (indcs.cnx2 + 2*ng) : 1;
  int cn3 = (indcs.cnx3 > 1) ? (indcs.cnx3 + 2*ng) : 1;
  ADMBCsImpl(ppack, coarse_u0, indcs.cis, indcs.cie, indcs.cjs, indcs.cje, indcs.cks,
             indcs.cke, indcs.cnx1, indcs.cnx2, indcs.cnx3, cn1, cn2, cn3);
}
