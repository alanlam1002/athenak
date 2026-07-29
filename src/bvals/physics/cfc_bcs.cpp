//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cfc_bcs.cpp
//! \brief Physical (non-block, non-periodic) boundary conditions for the CFC module's
//! own mesh-NGHOST-deep fields (P_i/X^i/beta^i and eta_x/psi/alpha_psi/eta_beta).
//!
//! CFC::QueueCFCTasks' MeshBoundaryValuesCC rounds (cfc.cpp) only cover inter-
//! MeshBlock/periodic communication via RecvAndUnpackCC -- exactly like Hydro/Z4c/
//! radiation, physical boundaries need a *separate* per-module BC pass (mirroring
//! Z4c::ApplyPhysicalBCs -> MeshBoundaryValues::Z4cBCs). Without it, ghost cells at
//! a physical boundary (e.g. the reflecting symmetry plane a single-octant TOV test
//! uses) are simply never written, so cfc_reconstruct.cpp's Dx<NGHOST> reads
//! whatever was left in freshly-allocated (zero-initialized) memory one cell in from
//! that boundary -- not actually NaN by itself, but wrong, and the CalculateFASRHS/
//! Smooth cycle can still produce a genuine NaN, e.g. dividing by a mock-zero psi.
//!
//! Deliberately simpler than HydroBCs: no inflow table (CFC has no analog of
//! u_in -- these are elliptic-equation potentials, not fluid state) and no
//! diode-specific velocity clamping (no flux/flow concept for these fields either).
//! Only reflect gets special (parity-flip) handling; outflow/diode/vacuum/inflow/
//! user all reduce to a zero-gradient copy by default (order=0), or a 1/r^order
//! extrapolation from the boundary-adjacent interior cell (order>0, flat=0) for
//! fields that are themselves an isolated-system Poisson solve's own output
//! (P_i/eta/psi/alpha_psi) -- mirrors adm_bcs.cpp's ADMBCs falloff technique (see
//! its file doc comment for the physical rationale), simplified since none of these
//! fields need a per-channel (flat, order) table the way ADM's alpha/psi4/g_dd/
//! vK_dd/beta_u channels do. vacuum is folded into the same case as outflow/diode/
//! inflow/user (not hard-zeroed) to match ADMBCs' own precedent (adm_bcs.cpp groups
//! vacuum with user/outflow/diode/inflow unconditionally) and to avoid the same
//! artificial-kink problem the order>0 falloff exists to fix in the first place.
//!
//! CFCBCs is the single public entry point (with a CFCBCsCoarse counterpart for the
//! coarse array), parameterized by nvar: 1 for a lone scalar (psi, alpha_psi), 3 for
//! a lone vector (X^i), or 4 for a vector packed with a trailing scalar (P_i at
//! channels 0-2 + eta at channel 3 of u_p_x/u_p_beta) -- one call covers both parts
//! of the packed case, since the reflect-parity flip below only ever matches local
//! channel index n against a fixed axis index (0/1/2), so channel 3 (nvar=4) can
//! never match and always falls through to the scalar (never-flip) treatment
//! without needing its own separate call or a separate "how many channels are
//! vector" argument. Internally shares one implementation, CFCBCsImpl, which also
//! takes the active-zone index bounds (is/ie/js/je/ks/ke), the interior cell counts
//! (nx1/nx2/nx3), and the total per-axis extents (n1/n2/n3) as explicit parameters
//! (mirroring z4c_bcs.cpp's BCHelper<order>) so the same implementation can be
//! shared between the fine array (CFCBCs) and the coarse array (CFCBCsCoarse) --
//! unlike z4c's BCHelper, this implementation is not coordinate-free (the order>0
//! falloff branch needs the true physical cell-center position via CellCenterX), so
//! nx1/nx2/nx3 must be the coarse interior counts for a coarse-array call, or the
//! computed radius would be wrong by the refinement factor.

#include <cstdlib>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"

namespace {

//----------------------------------------------------------------------------------------
//! \fn Real ReflectedValue(Real interior_val, bool flip)
//! \brief shared zero-gradient/vacuum/reflect switch, used identically by all 6 faces.
//! `flip` is true only for a vector channel (local index < 3) aligned with the face
//! being processed (always false for a lone scalar, nvar==1, and for a packed
//! array's trailing scalar channel, local index 3).

KOKKOS_INLINE_FUNCTION
Real ReflectedValue(Real interior_val, bool flip) {
  return flip ? -interior_val : interior_val;
}

//----------------------------------------------------------------------------------------
//! \fn void CFCBCsImpl(MeshBlockPack *ppack, DvceArray5D<Real> u0, int order,
//!                      int chan0, int nvar, int is, int ie, int js, int je, int ks,
//!                      int ke, int nx1, int nx2, int nx3, int n1, int n2, int n3)
//! \brief shared implementation behind CFCBCs/CFCBCsCoarse. Channels live at u0's
//! chan0..chan0+nvar-1; the local per-channel index n (0..nvar-1) drives reflect's
//! axis-aligned parity flip: flip requires nvar>=3 (a vector is present, whether
//! alone at nvar=3 or packed with a trailing scalar at nvar=4) AND n equal to the
//! face's axis index (0/1/2) -- for nvar=1 the flip is never triggered, and for
//! nvar=4's trailing scalar channel (n==3) it can never equal an axis index either,
//! so both scalar cases degrade to "never flip" without any extra guard. is/ie/js/
//! je/ks/ke/nx1/nx2/nx3/n1/n2/n3 select fine vs. coarse array index bounds/extents.

void CFCBCsImpl(MeshBlockPack *ppack, DvceArray5D<Real> u0, int order, int chan0,
                int nvar, int is, int ie, int js, int je, int ks, int ke,
                int nx1, int nx2, int nx3, int n1, int n2, int n3) {
  auto &pm = ppack->pmesh;
  auto &indcs = ppack->pmesh->mb_indcs;
  auto &size = ppack->pmb->mb_size;
  int &ng = indcs.ng;
  auto &mb_bcs = ppack->pmb->mb_bcs;

  int nmb = ppack->nmb_thispack;

  if (pm->mesh_bcs[BoundaryFace::inner_x1] != BoundaryFlag::periodic) {
    par_for("cfc_bc_x1", DevExeSpace(), 0, (nmb-1), 0, (nvar-1), 0, (n3-1), 0, (n2-1),
    KOKKOS_LAMBDA(int m, int n, int k, int j) {
      Real &x1min = size.d_view(m).x1min; Real &x1max = size.d_view(m).x1max;
      Real &x2min = size.d_view(m).x2min; Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
      Real &x3min = size.d_view(m).x3min; Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

      switch (mb_bcs.d_view(m,BoundaryFace::inner_x1)) {
        case BoundaryFlag::reflect: {
          bool flip = (nvar >= 3) && (n == 0);
          for (int i=0; i<ng; ++i) {
            u0(m,chan0+n,k,j,is-i-1) = ReflectedValue(u0(m,chan0+n,k,j,is+i), flip);
          }
          break;
        }
        case BoundaryFlag::outflow: case BoundaryFlag::diode: case BoundaryFlag::vacuum:
        case BoundaryFlag::inflow: case BoundaryFlag::user:
          if (order == 0) {
            for (int i=0; i<ng; ++i) { u0(m,chan0+n,k,j,is-i-1) = u0(m,chan0+n,k,j,is); }
          } else {
            Real x1_i = CellCenterX(is-is, nx1, x1min, x1max);
            Real r_i = Kokkos::sqrt(SQR(x1_i) + SQR(x2v) + SQR(x3v));
            Real f_i = u0(m,chan0+n,k,j,is);
            for (int i=0; i<ng; ++i) {
              Real x1_g = CellCenterX(is-i-1-is, nx1, x1min, x1max);
              Real r_g = Kokkos::sqrt(SQR(x1_g) + SQR(x2v) + SQR(x3v));
              u0(m,chan0+n,k,j,is-i-1) = f_i * Kokkos::pow(r_i/(r_g+1.0e-30), order);
            }
          }
          break;
        default: break;
      }
      switch (mb_bcs.d_view(m,BoundaryFace::outer_x1)) {
        case BoundaryFlag::reflect: {
          bool flip = (nvar >= 3) && (n == 0);
          for (int i=0; i<ng; ++i) {
            u0(m,chan0+n,k,j,ie+i+1) = ReflectedValue(u0(m,chan0+n,k,j,ie-i), flip);
          }
          break;
        }
        case BoundaryFlag::outflow: case BoundaryFlag::diode: case BoundaryFlag::vacuum:
        case BoundaryFlag::inflow: case BoundaryFlag::user:
          if (order == 0) {
            for (int i=0; i<ng; ++i) { u0(m,chan0+n,k,j,ie+i+1) = u0(m,chan0+n,k,j,ie); }
          } else {
            Real x1_i = CellCenterX(ie-is, nx1, x1min, x1max);
            Real r_i = Kokkos::sqrt(SQR(x1_i) + SQR(x2v) + SQR(x3v));
            Real f_i = u0(m,chan0+n,k,j,ie);
            for (int i=0; i<ng; ++i) {
              Real x1_g = CellCenterX(ie+i+1-is, nx1, x1min, x1max);
              Real r_g = Kokkos::sqrt(SQR(x1_g) + SQR(x2v) + SQR(x3v));
              u0(m,chan0+n,k,j,ie+i+1) = f_i * Kokkos::pow(r_i/(r_g+1.0e-30), order);
            }
          }
          break;
        default: break;
      }
    });
  }
  if (pm->one_d) return;

  if (pm->mesh_bcs[BoundaryFace::inner_x2] != BoundaryFlag::periodic) {
    par_for("cfc_bc_x2", DevExeSpace(), 0, (nmb-1), 0, (nvar-1), 0, (n3-1), 0, (n1-1),
    KOKKOS_LAMBDA(int m, int n, int k, int i) {
      Real &x1min = size.d_view(m).x1min; Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
      Real &x2min = size.d_view(m).x2min; Real &x2max = size.d_view(m).x2max;
      Real &x3min = size.d_view(m).x3min; Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

      switch (mb_bcs.d_view(m,BoundaryFace::inner_x2)) {
        case BoundaryFlag::reflect: {
          bool flip = (nvar >= 3) && (n == 1);
          for (int j=0; j<ng; ++j) {
            u0(m,chan0+n,k,js-j-1,i) = ReflectedValue(u0(m,chan0+n,k,js+j,i), flip);
          }
          break;
        }
        case BoundaryFlag::outflow: case BoundaryFlag::diode: case BoundaryFlag::vacuum:
        case BoundaryFlag::inflow: case BoundaryFlag::user:
          if (order == 0) {
            for (int j=0; j<ng; ++j) { u0(m,chan0+n,k,js-j-1,i) = u0(m,chan0+n,k,js,i); }
          } else {
            Real x2_i = CellCenterX(js-js, nx2, x2min, x2max);
            Real r_i = Kokkos::sqrt(SQR(x1v) + SQR(x2_i) + SQR(x3v));
            Real f_i = u0(m,chan0+n,k,js,i);
            for (int j=0; j<ng; ++j) {
              Real x2_g = CellCenterX(js-j-1-js, nx2, x2min, x2max);
              Real r_g = Kokkos::sqrt(SQR(x1v) + SQR(x2_g) + SQR(x3v));
              u0(m,chan0+n,k,js-j-1,i) = f_i * Kokkos::pow(r_i/(r_g+1.0e-30), order);
            }
          }
          break;
        default: break;
      }
      switch (mb_bcs.d_view(m,BoundaryFace::outer_x2)) {
        case BoundaryFlag::reflect: {
          bool flip = (nvar >= 3) && (n == 1);
          for (int j=0; j<ng; ++j) {
            u0(m,chan0+n,k,je+j+1,i) = ReflectedValue(u0(m,chan0+n,k,je-j,i), flip);
          }
          break;
        }
        case BoundaryFlag::outflow: case BoundaryFlag::diode: case BoundaryFlag::vacuum:
        case BoundaryFlag::inflow: case BoundaryFlag::user:
          if (order == 0) {
            for (int j=0; j<ng; ++j) { u0(m,chan0+n,k,je+j+1,i) = u0(m,chan0+n,k,je,i); }
          } else {
            Real x2_i = CellCenterX(je-js, nx2, x2min, x2max);
            Real r_i = Kokkos::sqrt(SQR(x1v) + SQR(x2_i) + SQR(x3v));
            Real f_i = u0(m,chan0+n,k,je,i);
            for (int j=0; j<ng; ++j) {
              Real x2_g = CellCenterX(je+j+1-js, nx2, x2min, x2max);
              Real r_g = Kokkos::sqrt(SQR(x1v) + SQR(x2_g) + SQR(x3v));
              u0(m,chan0+n,k,je+j+1,i) = f_i * Kokkos::pow(r_i/(r_g+1.0e-30), order);
            }
          }
          break;
        default: break;
      }
    });
  }
  if (pm->two_d) return;

  if (pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::periodic) return;
  par_for("cfc_bc_x3", DevExeSpace(), 0, (nmb-1), 0, (nvar-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int n, int j, int i) {
    Real &x1min = size.d_view(m).x1min; Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
    Real &x2min = size.d_view(m).x2min; Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
    Real &x3min = size.d_view(m).x3min; Real &x3max = size.d_view(m).x3max;

    switch (mb_bcs.d_view(m,BoundaryFace::inner_x3)) {
      case BoundaryFlag::reflect: {
        bool flip = (nvar >= 3) && (n == 2);
        for (int k=0; k<ng; ++k) {
          u0(m,chan0+n,ks-k-1,j,i) = ReflectedValue(u0(m,chan0+n,ks+k,j,i), flip);
        }
        break;
      }
      case BoundaryFlag::outflow: case BoundaryFlag::diode: case BoundaryFlag::vacuum:
      case BoundaryFlag::inflow: case BoundaryFlag::user:
        if (order == 0) {
          for (int k=0; k<ng; ++k) { u0(m,chan0+n,ks-k-1,j,i) = u0(m,chan0+n,ks,j,i); }
        } else {
          Real x3_i = CellCenterX(ks-ks, nx3, x3min, x3max);
          Real r_i = Kokkos::sqrt(SQR(x1v) + SQR(x2v) + SQR(x3_i));
          Real f_i = u0(m,chan0+n,ks,j,i);
          for (int k=0; k<ng; ++k) {
            Real x3_g = CellCenterX(ks-k-1-ks, nx3, x3min, x3max);
            Real r_g = Kokkos::sqrt(SQR(x1v) + SQR(x2v) + SQR(x3_g));
            u0(m,chan0+n,ks-k-1,j,i) = f_i * Kokkos::pow(r_i/(r_g+1.0e-30), order);
          }
        }
        break;
      default: break;
    }
    switch (mb_bcs.d_view(m,BoundaryFace::outer_x3)) {
      case BoundaryFlag::reflect: {
        bool flip = (nvar >= 3) && (n == 2);
        for (int k=0; k<ng; ++k) {
          u0(m,chan0+n,ke+k+1,j,i) = ReflectedValue(u0(m,chan0+n,ke-k,j,i), flip);
        }
        break;
      }
      case BoundaryFlag::outflow: case BoundaryFlag::diode: case BoundaryFlag::vacuum:
      case BoundaryFlag::inflow: case BoundaryFlag::user:
        if (order == 0) {
          for (int k=0; k<ng; ++k) { u0(m,chan0+n,ke+k+1,j,i) = u0(m,chan0+n,ke,j,i); }
        } else {
          Real x3_i = CellCenterX(ke-ks, nx3, x3min, x3max);
          Real r_i = Kokkos::sqrt(SQR(x1v) + SQR(x2v) + SQR(x3_i));
          Real f_i = u0(m,chan0+n,ke,j,i);
          for (int k=0; k<ng; ++k) {
            Real x3_g = CellCenterX(ke+k+1-ks, nx3, x3min, x3max);
            Real r_g = Kokkos::sqrt(SQR(x1v) + SQR(x2v) + SQR(x3_g));
            u0(m,chan0+n,ke+k+1,j,i) = f_i * Kokkos::pow(r_i/(r_g+1.0e-30), order);
          }
        }
        break;
      default: break;
    }
  });
  return;
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValues::CFCBCs()
//! \brief Apply physical boundary conditions to nvar contiguous channels of a CFC
//! field starting at chan0: nvar=1 for a lone scalar (psi, alpha_psi), nvar=3 for a
//! lone vector (X^i), or nvar=4 for a vector packed with a trailing scalar (P_i at
//! channels 0-2 + eta at channel 3 of u_p_x/u_p_beta) -- one call covers both parts
//! of the packed case (see file doc comment for why). Thin wrapper around the
//! shared CFCBCsImpl, fine-array index/extent set.

void MeshBoundaryValues::CFCBCs(MeshBlockPack *ppack, DvceArray5D<Real> u0,
                                int nvar, int order, int chan0) {
  auto &indcs = ppack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  CFCBCsImpl(ppack, u0, order, chan0, nvar, indcs.is, indcs.ie, indcs.js, indcs.je,
             indcs.ks, indcs.ke, indcs.nx1, indcs.nx2, indcs.nx3, n1, n2, n3);
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValues::CFCBCsCoarse()
//! \brief Coarse-array counterpart of CFCBCs -- must be called before ProlongateCC
//! so the prolongation stencil reads valid coarse ghost data at a physical boundary
//! (mirrors Z4cBCsCoarse). Same shared CFCBCsImpl, called with the coarse
//! index/extent set instead of the fine one.

void MeshBoundaryValues::CFCBCsCoarse(MeshBlockPack *ppack,
                                      DvceArray5D<Real> coarse_u0, int nvar,
                                      int order, int chan0) {
  auto &indcs = ppack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int cn1 = indcs.cnx1 + 2*ng;
  int cn2 = (indcs.cnx2 > 1) ? (indcs.cnx2 + 2*ng) : 1;
  int cn3 = (indcs.cnx3 > 1) ? (indcs.cnx3 + 2*ng) : 1;
  CFCBCsImpl(ppack, coarse_u0, order, chan0, nvar, indcs.cis, indcs.cie, indcs.cjs,
             indcs.cje, indcs.cks, indcs.cke, indcs.cnx1, indcs.cnx2, indcs.cnx3,
             cn1, cn2, cn3);
}
