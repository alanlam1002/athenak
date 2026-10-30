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
#include "srcterms/srcterms.hpp"
#include "driver/driver.hpp"
#include "mhd.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn  void MHD::CT
//  \brief Constrained Transport implementation of dB/dt = -Curl(E), where E=-(v X B)
//  To be clear, the edge-centered variable 'efld' stores E = -(v X B).
//  Temporal update uses multi-step SSP integrators, e.g. RK2, RK3

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
  auto &mbsize = pmy_pack->pmb->mb_size;

  //---- update B1 (only for 2D/3D problems)
  if (multi_d) {
    auto bx1f = b0.x1f;
    auto bx1f_old = b1.x1f;
    par_for("CT-b1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      bx1f(m,k,j,i) = gam0*bx1f(m,k,j,i) + gam1*bx1f_old(m,k,j,i);
      bx1f(m,k,j,i) -= beta_dt*(e3(m,k,j+1,i) - e3(m,k,j,i))/mbsize.d_view(m).dx2;
      if (three_d) {
        bx1f(m,k,j,i) += beta_dt*(e2(m,k+1,j,i) - e2(m,k,j,i))/mbsize.d_view(m).dx3;
      }
    });
  }

  //---- update B2 (curl terms in 1D and 3D problems)
  auto bx2f = b0.x2f;
  auto bx2f_old = b1.x2f;
  par_for("CT-b2", DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    bx2f(m,k,j,i) = gam0*bx2f(m,k,j,i) + gam1*bx2f_old(m,k,j,i);
    bx2f(m,k,j,i) += beta_dt*(e3(m,k,j,i+1) - e3(m,k,j,i))/mbsize.d_view(m).dx1;
    if (three_d) {
      bx2f(m,k,j,i) -= beta_dt*(e1(m,k+1,j,i) - e1(m,k,j,i))/mbsize.d_view(m).dx3;
    }
  });

  //---- update B3 (curl terms in 1D and 2D/3D problems)
  auto bx3f = b0.x3f;
  auto bx3f_old = b1.x3f;
  par_for("CT-b3", DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    bx3f(m,k,j,i) = gam0*bx3f(m,k,j,i) + gam1*bx3f_old(m,k,j,i);
    bx3f(m,k,j,i) -= beta_dt*(e2(m,k,j,i+1) - e2(m,k,j,i))/mbsize.d_view(m).dx1;
    if (multi_d) {
      bx3f(m,k,j,i) += beta_dt*(e1(m,k,j+1,i) - e1(m,k,j,i))/mbsize.d_view(m).dx2;
    }
  });

  //---- edge vector potential (helicity diagnostic, <mhd> evolve_avec): the same
  // stage-weighted update with the same (already flux-corrected) edge EMFs, so that
  // curl(avec) tracks b0 exactly; separate kernels keep the B update above unchanged
  if (evolve_avec) {
    auto a1 = avec.x1e, a1_old = avec1.x1e;
    auto a2 = avec.x2e, a2_old = avec1.x2e;
    auto a3 = avec.x3e, a3_old = avec1.x3e;
    par_for("CT-a1", DevExeSpace(), 0, nmb1, ks, ke+1, js, je+1, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      a1(m,k,j,i) = gam0*a1(m,k,j,i) + gam1*a1_old(m,k,j,i) - beta_dt*e1(m,k,j,i);
    });
    par_for("CT-a2", DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      a2(m,k,j,i) = gam0*a2(m,k,j,i) + gam1*a2_old(m,k,j,i) - beta_dt*e2(m,k,j,i);
    });
    par_for("CT-a3", DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      a3(m,k,j,i) = gam0*a3(m,k,j,i) + gam1*a3_old(m,k,j,i) - beta_dt*e3(m,k,j,i);
    });
  }

  return TaskStatus::complete;
}
} // namespace mhd
