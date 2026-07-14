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
//! fields that are themselves an isolated-system Poisson solve's own output (P_i/
//! eta) -- mirrors adm_bcs.cpp's ADMBCs falloff technique (see its file doc comment
//! for the physical rationale), simplified since none of these fields need a
//! per-channel (flat, order) table the way ADM's alpha/psi4/g_dd/vK_dd/beta_u
//! channels do. vacuum is folded into the same case as outflow/diode/inflow/user
//! (not hard-zeroed) to match ADMBCs' own precedent (adm_bcs.cpp groups vacuum with
//! user/outflow/diode/inflow unconditionally) and to avoid the same artificial-kink
//! problem the order>0 falloff exists to fix in the first place.
//!
//! CFCScalarBCs (single channel, always even parity) and CFCVectorBCs (3 channels,
//! odd parity only for the component aligned with the face's own axis) share one
//! implementation, CFCBCsImpl, parameterized by nvar (1 or 3): the two public
//! functions are thin wrappers so every existing call site keeps working verbatim.

#include <cstdlib>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"

namespace {

//----------------------------------------------------------------------------------------
//! \fn Real ReflectedValue(Real interior_val, bool flip)
//! \brief shared zero-gradient/vacuum/reflect switch, used identically by all 6 faces.
//! `flip` is true only for the single vector channel aligned with the face being
//! processed (always false for scalars, nvar==1).

KOKKOS_INLINE_FUNCTION
Real ReflectedValue(Real interior_val, bool flip) {
  return flip ? -interior_val : interior_val;
}

//----------------------------------------------------------------------------------------
//! \fn void CFCBCsImpl(MeshBlockPack *ppack, DvceArray5D<Real> u0, int order,
//!                      int chan0, int nvar)
//! \brief shared implementation behind CFCScalarBCs (nvar=1) and CFCVectorBCs
//! (nvar=3). Channels live at u0's chan0..chan0+nvar-1; the local per-channel index
//! n (0..nvar-1) drives reflect's axis-aligned parity flip (never triggered when
//! nvar==1, since (nvar==3) is false).

void CFCBCsImpl(MeshBlockPack *ppack, DvceArray5D<Real> u0, int order, int chan0,
                int nvar) {
  auto &pm = ppack->pmesh;
  auto &indcs = ppack->pmesh->mb_indcs;
  auto &size = ppack->pmb->mb_size;
  int &ng = indcs.ng;
  auto &mb_bcs = ppack->pmb->mb_bcs;

  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int nmb = ppack->nmb_thispack;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;

  if (pm->mesh_bcs[BoundaryFace::inner_x1] != BoundaryFlag::periodic) {
    par_for("cfc_bc_x1", DevExeSpace(), 0, (nmb-1), 0, (nvar-1), 0, (n3-1), 0, (n2-1),
    KOKKOS_LAMBDA(int m, int n, int k, int j) {
      Real &x1min = size.d_view(m).x1min; Real &x1max = size.d_view(m).x1max;
      Real &x2min = size.d_view(m).x2min; Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
      Real &x3min = size.d_view(m).x3min; Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

      switch (mb_bcs.d_view(m,BoundaryFace::inner_x1)) {
        case BoundaryFlag::reflect: {
          bool flip = (nvar == 3) && (n == 0);
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
            Real x1_i = CellCenterX(is-is, indcs.nx1, x1min, x1max);
            Real r_i = Kokkos::sqrt(SQR(x1_i) + SQR(x2v) + SQR(x3v));
            Real f_i = u0(m,chan0+n,k,j,is);
            for (int i=0; i<ng; ++i) {
              Real x1_g = CellCenterX(is-i-1-is, indcs.nx1, x1min, x1max);
              Real r_g = Kokkos::sqrt(SQR(x1_g) + SQR(x2v) + SQR(x3v));
              u0(m,chan0+n,k,j,is-i-1) = f_i * Kokkos::pow(r_i/(r_g+1.0e-30), order);
            }
          }
          break;
        default: break;
      }
      switch (mb_bcs.d_view(m,BoundaryFace::outer_x1)) {
        case BoundaryFlag::reflect: {
          bool flip = (nvar == 3) && (n == 0);
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
            Real x1_i = CellCenterX(ie-is, indcs.nx1, x1min, x1max);
            Real r_i = Kokkos::sqrt(SQR(x1_i) + SQR(x2v) + SQR(x3v));
            Real f_i = u0(m,chan0+n,k,j,ie);
            for (int i=0; i<ng; ++i) {
              Real x1_g = CellCenterX(ie+i+1-is, indcs.nx1, x1min, x1max);
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
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
      Real &x2min = size.d_view(m).x2min; Real &x2max = size.d_view(m).x2max;
      Real &x3min = size.d_view(m).x3min; Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

      switch (mb_bcs.d_view(m,BoundaryFace::inner_x2)) {
        case BoundaryFlag::reflect: {
          bool flip = (nvar == 3) && (n == 1);
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
            Real x2_i = CellCenterX(js-js, indcs.nx2, x2min, x2max);
            Real r_i = Kokkos::sqrt(SQR(x1v) + SQR(x2_i) + SQR(x3v));
            Real f_i = u0(m,chan0+n,k,js,i);
            for (int j=0; j<ng; ++j) {
              Real x2_g = CellCenterX(js-j-1-js, indcs.nx2, x2min, x2max);
              Real r_g = Kokkos::sqrt(SQR(x1v) + SQR(x2_g) + SQR(x3v));
              u0(m,chan0+n,k,js-j-1,i) = f_i * Kokkos::pow(r_i/(r_g+1.0e-30), order);
            }
          }
          break;
        default: break;
      }
      switch (mb_bcs.d_view(m,BoundaryFace::outer_x2)) {
        case BoundaryFlag::reflect: {
          bool flip = (nvar == 3) && (n == 1);
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
            Real x2_i = CellCenterX(je-js, indcs.nx2, x2min, x2max);
            Real r_i = Kokkos::sqrt(SQR(x1v) + SQR(x2_i) + SQR(x3v));
            Real f_i = u0(m,chan0+n,k,je,i);
            for (int j=0; j<ng; ++j) {
              Real x2_g = CellCenterX(je+j+1-js, indcs.nx2, x2min, x2max);
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
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
    Real &x2min = size.d_view(m).x2min; Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
    Real &x3min = size.d_view(m).x3min; Real &x3max = size.d_view(m).x3max;

    switch (mb_bcs.d_view(m,BoundaryFace::inner_x3)) {
      case BoundaryFlag::reflect: {
        bool flip = (nvar == 3) && (n == 2);
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
          Real x3_i = CellCenterX(ks-ks, indcs.nx3, x3min, x3max);
          Real r_i = Kokkos::sqrt(SQR(x1v) + SQR(x2v) + SQR(x3_i));
          Real f_i = u0(m,chan0+n,ks,j,i);
          for (int k=0; k<ng; ++k) {
            Real x3_g = CellCenterX(ks-k-1-ks, indcs.nx3, x3min, x3max);
            Real r_g = Kokkos::sqrt(SQR(x1v) + SQR(x2v) + SQR(x3_g));
            u0(m,chan0+n,ks-k-1,j,i) = f_i * Kokkos::pow(r_i/(r_g+1.0e-30), order);
          }
        }
        break;
      default: break;
    }
    switch (mb_bcs.d_view(m,BoundaryFace::outer_x3)) {
      case BoundaryFlag::reflect: {
        bool flip = (nvar == 3) && (n == 2);
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
          Real x3_i = CellCenterX(ke-ks, indcs.nx3, x3min, x3max);
          Real r_i = Kokkos::sqrt(SQR(x1v) + SQR(x2v) + SQR(x3_i));
          Real f_i = u0(m,chan0+n,ke,j,i);
          for (int k=0; k<ng; ++k) {
            Real x3_g = CellCenterX(ke+k+1-ks, indcs.nx3, x3min, x3max);
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
//! \fn void MeshBoundaryValues::CFCScalarBCs()
//! \brief Apply physical boundary conditions to a single scalar channel of a CFC
//! field (psi, alpha_psi, or channel 3 -- eta -- of the packed u_p_x/u_p_beta arrays
//! -- all scalars, even parity under reflection). chan0 selects which channel of u0
//! this scalar lives at (0 for psi/alpha_psi, 3 for eta packed alongside P_i). Thin
//! wrapper around the shared CFCBCsImpl (nvar=1).

void MeshBoundaryValues::CFCScalarBCs(MeshBlockPack *ppack, DvceArray5D<Real> u0,
                                      int order, int chan0) {
  CFCBCsImpl(ppack, u0, order, chan0, 1);
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValues::CFCVectorBCs()
//! \brief Apply physical boundary conditions to a 3-channel vector field starting at
//! channel chan0 of u0 (P_i at chan0=0 of the packed u_p_x/u_p_beta arrays, or X^i --
//! the only unpacked case, chan0=0). Odd parity (sign flip) only for the local
//! component aligned with the face's own axis (n==0 on x1 faces, n==1 on x2, n==2 on
//! x3, relative to the vector's own 3 components, not the absolute channel index);
//! the other two components and every non-reflect case match CFCScalarBCs' treatment
//! exactly, per channel. Thin wrapper around the shared CFCBCsImpl (nvar=3).

void MeshBoundaryValues::CFCVectorBCs(MeshBlockPack *ppack, DvceArray5D<Real> u0,
                                      int order, int chan0) {
  CFCBCsImpl(ppack, u0, order, chan0, 3);
}
