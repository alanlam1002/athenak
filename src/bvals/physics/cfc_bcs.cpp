//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cfc_bcs.cpp
//! \brief Physical (non-block, non-periodic) boundary conditions for the CFC module's
//! own mesh-NGHOST-deep fields (P_i/X^i/beta^i and eta_x/psi/alpha_psi/eta_beta).
//!
//! CFC::QueueCFCTasks' 7 MeshBoundaryValuesCC rounds (cfc.cpp) only cover inter-
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
//! outflow/diode/inflow/user all reduce to the same zero-gradient copy; only
//! reflect and vacuum need special handling, exactly as for any other field with no
//! preferred inflow value.

#include <cstdlib>

#include "athena.hpp"
#include "mesh/mesh.hpp"

namespace {

//----------------------------------------------------------------------------------------
//! \fn void ApplyFace(...)
//! \brief shared zero-gradient/vacuum/reflect switch, used identically by all 6 faces
//! of both CFCScalarBCs and CFCVectorBCs below. `flip` is true only for the single
//! vector channel aligned with the face being processed (always false for scalars).

KOKKOS_INLINE_FUNCTION
Real ReflectedValue(Real interior_val, bool flip) {
  return flip ? -interior_val : interior_val;
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValues::CFCScalarBCs()
//! \brief Apply physical boundary conditions to a single-channel CFC field
//! (eta_x, psi, alpha_psi, or eta_beta -- all scalars, even parity under reflection).

void MeshBoundaryValues::CFCScalarBCs(MeshBlockPack *ppack, DvceArray5D<Real> u0) {
  auto &pm = ppack->pmesh;
  auto &indcs = ppack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  auto &mb_bcs = ppack->pmb->mb_bcs;

  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int nmb = ppack->nmb_thispack;

  if (pm->mesh_bcs[BoundaryFace::inner_x1] != BoundaryFlag::periodic) {
    int &is = indcs.is; int &ie = indcs.ie;
    par_for("cfc_scalar_bc_x1", DevExeSpace(), 0, (nmb-1), 0, (n3-1), 0, (n2-1),
    KOKKOS_LAMBDA(int m, int k, int j) {
      switch (mb_bcs.d_view(m,BoundaryFace::inner_x1)) {
        case BoundaryFlag::reflect:
          for (int i=0; i<ng; ++i) { u0(m,0,k,j,is-i-1) = u0(m,0,k,j,is+i); }
          break;
        case BoundaryFlag::vacuum:
          for (int i=0; i<ng; ++i) { u0(m,0,k,j,is-i-1) = 0.0; }
          break;
        case BoundaryFlag::outflow: case BoundaryFlag::diode:
        case BoundaryFlag::inflow: case BoundaryFlag::user:
          for (int i=0; i<ng; ++i) { u0(m,0,k,j,is-i-1) = u0(m,0,k,j,is); }
          break;
        default: break;
      }
      switch (mb_bcs.d_view(m,BoundaryFace::outer_x1)) {
        case BoundaryFlag::reflect:
          for (int i=0; i<ng; ++i) { u0(m,0,k,j,ie+i+1) = u0(m,0,k,j,ie-i); }
          break;
        case BoundaryFlag::vacuum:
          for (int i=0; i<ng; ++i) { u0(m,0,k,j,ie+i+1) = 0.0; }
          break;
        case BoundaryFlag::outflow: case BoundaryFlag::diode:
        case BoundaryFlag::inflow: case BoundaryFlag::user:
          for (int i=0; i<ng; ++i) { u0(m,0,k,j,ie+i+1) = u0(m,0,k,j,ie); }
          break;
        default: break;
      }
    });
  }
  if (pm->one_d) return;

  if (pm->mesh_bcs[BoundaryFace::inner_x2] != BoundaryFlag::periodic) {
    int &js = indcs.js; int &je = indcs.je;
    par_for("cfc_scalar_bc_x2", DevExeSpace(), 0, (nmb-1), 0, (n3-1), 0, (n1-1),
    KOKKOS_LAMBDA(int m, int k, int i) {
      switch (mb_bcs.d_view(m,BoundaryFace::inner_x2)) {
        case BoundaryFlag::reflect:
          for (int j=0; j<ng; ++j) { u0(m,0,k,js-j-1,i) = u0(m,0,k,js+j,i); }
          break;
        case BoundaryFlag::vacuum:
          for (int j=0; j<ng; ++j) { u0(m,0,k,js-j-1,i) = 0.0; }
          break;
        case BoundaryFlag::outflow: case BoundaryFlag::diode:
        case BoundaryFlag::inflow: case BoundaryFlag::user:
          for (int j=0; j<ng; ++j) { u0(m,0,k,js-j-1,i) = u0(m,0,k,js,i); }
          break;
        default: break;
      }
      switch (mb_bcs.d_view(m,BoundaryFace::outer_x2)) {
        case BoundaryFlag::reflect:
          for (int j=0; j<ng; ++j) { u0(m,0,k,je+j+1,i) = u0(m,0,k,je-j,i); }
          break;
        case BoundaryFlag::vacuum:
          for (int j=0; j<ng; ++j) { u0(m,0,k,je+j+1,i) = 0.0; }
          break;
        case BoundaryFlag::outflow: case BoundaryFlag::diode:
        case BoundaryFlag::inflow: case BoundaryFlag::user:
          for (int j=0; j<ng; ++j) { u0(m,0,k,je+j+1,i) = u0(m,0,k,je,i); }
          break;
        default: break;
      }
    });
  }
  if (pm->two_d) return;

  if (pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::periodic) return;
  int &ks = indcs.ks; int &ke = indcs.ke;
  par_for("cfc_scalar_bc_x3", DevExeSpace(), 0, (nmb-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int j, int i) {
    switch (mb_bcs.d_view(m,BoundaryFace::inner_x3)) {
      case BoundaryFlag::reflect:
        for (int k=0; k<ng; ++k) { u0(m,0,ks-k-1,j,i) = u0(m,0,ks+k,j,i); }
        break;
      case BoundaryFlag::vacuum:
        for (int k=0; k<ng; ++k) { u0(m,0,ks-k-1,j,i) = 0.0; }
        break;
      case BoundaryFlag::outflow: case BoundaryFlag::diode:
      case BoundaryFlag::inflow: case BoundaryFlag::user:
        for (int k=0; k<ng; ++k) { u0(m,0,ks-k-1,j,i) = u0(m,0,ks,j,i); }
        break;
      default: break;
    }
    switch (mb_bcs.d_view(m,BoundaryFace::outer_x3)) {
      case BoundaryFlag::reflect:
        for (int k=0; k<ng; ++k) { u0(m,0,ke+k+1,j,i) = u0(m,0,ke-k,j,i); }
        break;
      case BoundaryFlag::vacuum:
        for (int k=0; k<ng; ++k) { u0(m,0,ke+k+1,j,i) = 0.0; }
        break;
      case BoundaryFlag::outflow: case BoundaryFlag::diode:
      case BoundaryFlag::inflow: case BoundaryFlag::user:
        for (int k=0; k<ng; ++k) { u0(m,0,ke+k+1,j,i) = u0(m,0,ke,j,i); }
        break;
      default: break;
    }
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValues::CFCVectorBCs()
//! \brief Apply physical boundary conditions to a 3-channel CFC field (P_i, X^i, or
//! beta^i). Odd parity (sign flip) only for the channel aligned with the face's own
//! axis (n==0 on x1 faces, n==1 on x2, n==2 on x3); the other two channels and every
//! non-reflect case match CFCScalarBCs' treatment exactly, per channel.

void MeshBoundaryValues::CFCVectorBCs(MeshBlockPack *ppack, DvceArray5D<Real> u0) {
  auto &pm = ppack->pmesh;
  auto &indcs = ppack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  auto &mb_bcs = ppack->pmb->mb_bcs;

  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int nmb = ppack->nmb_thispack;
  constexpr int nvar = 3;

  if (pm->mesh_bcs[BoundaryFace::inner_x1] != BoundaryFlag::periodic) {
    int &is = indcs.is; int &ie = indcs.ie;
    par_for("cfc_vector_bc_x1", DevExeSpace(), 0, (nmb-1), 0, (nvar-1), 0, (n3-1),
            0, (n2-1),
    KOKKOS_LAMBDA(int m, int n, int k, int j) {
      switch (mb_bcs.d_view(m,BoundaryFace::inner_x1)) {
        case BoundaryFlag::reflect:
          for (int i=0; i<ng; ++i) {
            u0(m,n,k,j,is-i-1) = ReflectedValue(u0(m,n,k,j,is+i), n==0);
          }
          break;
        case BoundaryFlag::vacuum:
          for (int i=0; i<ng; ++i) { u0(m,n,k,j,is-i-1) = 0.0; }
          break;
        case BoundaryFlag::outflow: case BoundaryFlag::diode:
        case BoundaryFlag::inflow: case BoundaryFlag::user:
          for (int i=0; i<ng; ++i) { u0(m,n,k,j,is-i-1) = u0(m,n,k,j,is); }
          break;
        default: break;
      }
      switch (mb_bcs.d_view(m,BoundaryFace::outer_x1)) {
        case BoundaryFlag::reflect:
          for (int i=0; i<ng; ++i) {
            u0(m,n,k,j,ie+i+1) = ReflectedValue(u0(m,n,k,j,ie-i), n==0);
          }
          break;
        case BoundaryFlag::vacuum:
          for (int i=0; i<ng; ++i) { u0(m,n,k,j,ie+i+1) = 0.0; }
          break;
        case BoundaryFlag::outflow: case BoundaryFlag::diode:
        case BoundaryFlag::inflow: case BoundaryFlag::user:
          for (int i=0; i<ng; ++i) { u0(m,n,k,j,ie+i+1) = u0(m,n,k,j,ie); }
          break;
        default: break;
      }
    });
  }
  if (pm->one_d) return;

  if (pm->mesh_bcs[BoundaryFace::inner_x2] != BoundaryFlag::periodic) {
    int &js = indcs.js; int &je = indcs.je;
    par_for("cfc_vector_bc_x2", DevExeSpace(), 0, (nmb-1), 0, (nvar-1), 0, (n3-1),
            0, (n1-1),
    KOKKOS_LAMBDA(int m, int n, int k, int i) {
      switch (mb_bcs.d_view(m,BoundaryFace::inner_x2)) {
        case BoundaryFlag::reflect:
          for (int j=0; j<ng; ++j) {
            u0(m,n,k,js-j-1,i) = ReflectedValue(u0(m,n,k,js+j,i), n==1);
          }
          break;
        case BoundaryFlag::vacuum:
          for (int j=0; j<ng; ++j) { u0(m,n,k,js-j-1,i) = 0.0; }
          break;
        case BoundaryFlag::outflow: case BoundaryFlag::diode:
        case BoundaryFlag::inflow: case BoundaryFlag::user:
          for (int j=0; j<ng; ++j) { u0(m,n,k,js-j-1,i) = u0(m,n,k,js,i); }
          break;
        default: break;
      }
      switch (mb_bcs.d_view(m,BoundaryFace::outer_x2)) {
        case BoundaryFlag::reflect:
          for (int j=0; j<ng; ++j) {
            u0(m,n,k,je+j+1,i) = ReflectedValue(u0(m,n,k,je-j,i), n==1);
          }
          break;
        case BoundaryFlag::vacuum:
          for (int j=0; j<ng; ++j) { u0(m,n,k,je+j+1,i) = 0.0; }
          break;
        case BoundaryFlag::outflow: case BoundaryFlag::diode:
        case BoundaryFlag::inflow: case BoundaryFlag::user:
          for (int j=0; j<ng; ++j) { u0(m,n,k,je+j+1,i) = u0(m,n,k,je,i); }
          break;
        default: break;
      }
    });
  }
  if (pm->two_d) return;

  if (pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::periodic) return;
  int &ks = indcs.ks; int &ke = indcs.ke;
  par_for("cfc_vector_bc_x3", DevExeSpace(), 0, (nmb-1), 0, (nvar-1), 0, (n2-1),
          0, (n1-1),
  KOKKOS_LAMBDA(int m, int n, int j, int i) {
    switch (mb_bcs.d_view(m,BoundaryFace::inner_x3)) {
      case BoundaryFlag::reflect:
        for (int k=0; k<ng; ++k) {
          u0(m,n,ks-k-1,j,i) = ReflectedValue(u0(m,n,ks+k,j,i), n==2);
        }
        break;
      case BoundaryFlag::vacuum:
        for (int k=0; k<ng; ++k) { u0(m,n,ks-k-1,j,i) = 0.0; }
        break;
      case BoundaryFlag::outflow: case BoundaryFlag::diode:
      case BoundaryFlag::inflow: case BoundaryFlag::user:
        for (int k=0; k<ng; ++k) { u0(m,n,ks-k-1,j,i) = u0(m,n,ks,j,i); }
        break;
      default: break;
    }
    switch (mb_bcs.d_view(m,BoundaryFace::outer_x3)) {
      case BoundaryFlag::reflect:
        for (int k=0; k<ng; ++k) {
          u0(m,n,ke+k+1,j,i) = ReflectedValue(u0(m,n,ke-k,j,i), n==2);
        }
        break;
      case BoundaryFlag::vacuum:
        for (int k=0; k<ng; ++k) { u0(m,n,ke+k+1,j,i) = 0.0; }
        break;
      case BoundaryFlag::outflow: case BoundaryFlag::diode:
      case BoundaryFlag::inflow: case BoundaryFlag::user:
        for (int k=0; k<ng; ++k) { u0(m,n,ke+k+1,j,i) = u0(m,n,ke,j,i); }
        break;
      default: break;
    }
  });
  return;
}
