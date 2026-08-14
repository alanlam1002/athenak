//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_ct.cpp
//  \brief

// Athena++ headers
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/mesh_geometry.hpp"
#include "srcterms/srcterms.hpp"
#include "driver/driver.hpp"
#include "mhd.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn  void MHD::CT
//  \brief Constrained Transport implementation of dB/dt = -Curl(E), where E=-(v X B),
//  written in area/edge-length-weighted Stokes form (Task D1):
//    d(B1*Area1)/dt = -[e3*Len3]_{j+1} + [e3*Len3]_j + [e2*Len2]_{k+1} - [e2*Len2]_k
//  (and cyclic for B2/B3), i.e. the discrete curl of E integrated around the four edges
//  bounding each face, divided by that face's area. This is the exact generalization of
//  the flat/uniform formula below (Area1=dx2*dx3, Len3=dx3, Len2=dx2, etc. reduce it
//  back to the old dx2/dx3-division form exactly -- verified by hand for all three
//  faces, see DEVELOPMENT.md Task D1 log) that only reads the SAME Area1/2/3, Len1/2/3
//  accessors already used by the flux divergence (Task A3/A4) and CFL (Task B5)
//  kernels -- no new GeomData fields needed. To be clear, the edge-centered variable
//  'efld' stores E = -(v X B). Temporal update uses multi-step SSP integrators.

TaskStatus MHD::CT(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  // capture class variables for the kernels
  Real &gam0 = pdriver->gam0[stage-1];
  Real &gam1 = pdriver->gam1[stage-1];
  Real beta_dt = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  auto e1 = efld.x1e;
  auto e2 = efld.x2e;
  auto e3 = efld.x3e;
  auto &geom = pmy_pack->pgeom->geom_data;

  //---- update B1 (only for 2D/3D problems)
  if (multi_d) {
    auto bx1f = b0.x1f;
    auto bx1f_old = b1.x1f;
    par_for("CT-b1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real area1 = geom.Area1(m,k,j,i);
      bx1f(m,k,j,i) = gam0*bx1f(m,k,j,i) + gam1*bx1f_old(m,k,j,i);
      bx1f(m,k,j,i) -= beta_dt*(geom.Len3(m,k,j+1,i)*e3(m,k,j+1,i)
                                - geom.Len3(m,k,j,i)*e3(m,k,j,i))/area1;
      if (three_d) {
        bx1f(m,k,j,i) += beta_dt*(geom.Len2(m,k+1,j,i)*e2(m,k+1,j,i)
                                  - geom.Len2(m,k,j,i)*e2(m,k,j,i))/area1;
      }
    });
  }

  //---- update B2 (curl terms in 1D and 3D problems)
  auto bx2f = b0.x2f;
  auto bx2f_old = b1.x2f;
  par_for("CT-b2", DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real area2 = geom.Area2(m,k,j,i);
    bx2f(m,k,j,i) = gam0*bx2f(m,k,j,i) + gam1*bx2f_old(m,k,j,i);
    bx2f(m,k,j,i) += beta_dt*(geom.Len3(m,k,j,i+1)*e3(m,k,j,i+1)
                              - geom.Len3(m,k,j,i)*e3(m,k,j,i))/area2;
    if (three_d) {
      bx2f(m,k,j,i) -= beta_dt*(geom.Len1(m,k+1,j,i)*e1(m,k+1,j,i)
                                - geom.Len1(m,k,j,i)*e1(m,k,j,i))/area2;
    }
  });

  //---- update B3 (curl terms in 1D and 2D/3D problems)
  auto bx3f = b0.x3f;
  auto bx3f_old = b1.x3f;
  par_for("CT-b3", DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real area3 = geom.Area3(m,k,j,i);
    bx3f(m,k,j,i) = gam0*bx3f(m,k,j,i) + gam1*bx3f_old(m,k,j,i);
    bx3f(m,k,j,i) -= beta_dt*(geom.Len2(m,k,j,i+1)*e2(m,k,j,i+1)
                              - geom.Len2(m,k,j,i)*e2(m,k,j,i))/area3;
    if (multi_d) {
      bx3f(m,k,j,i) += beta_dt*(geom.Len1(m,k,j+1,i)*e1(m,k,j+1,i)
                                - geom.Len1(m,k,j,i)*e1(m,k,j,i))/area3;
    }
  });

  return TaskStatus::complete;
}
} // namespace mhd
