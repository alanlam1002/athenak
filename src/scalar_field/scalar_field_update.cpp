//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file scalar_field_update.cpp
//! \brief Performs update of scalar-field variables (u0) for each stage of explicit
//! SSP RK integrators (e.g. RK1, RK2, RK3, RK4). Identical in structure to
//! z4c/z4c_update.cpp -- a generic weighted-average update, with no per-variable logic.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "scalar_field/scalar_field.hpp"

namespace scalarfield {
//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ScalarField::ExpRKUpdate
//! \brief Explicit RK update
TaskStatus ScalarField::ExpRKUpdate(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;

  Real &gam0 = pdriver->gam0[stage-1];
  Real &gam1 = pdriver->gam1[stage-1];
  Real beta_dt = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);
  auto &u0 = pmy_pack->pscalarfield->u0;
  auto &u1 = pmy_pack->pscalarfield->u1;
  auto &u_rhs = pmy_pack->pscalarfield->u_rhs;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nvar = nscalarfield;

  par_for("sf RK update", DevExeSpace(),
      0,nmb1,0,nvar-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(const int m, const int n, const int k, const int j, const int i) {
    u0(m,n,k,j,i) = gam0*u0(m,n,k,j,i) + gam1*u1(m,n,k,j,i) + beta_dt*u_rhs(m,n,k,j,i);
  });
  return TaskStatus::complete;
}
} // namespace scalarfield
