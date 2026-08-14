//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_update.cpp
//! \brief Performs explicit update of Hydro conserved variables (u0) for each stage of
//! the SSP RK integrators (e.g. RK1, RK2, RK3) implemented in AthenaK, using weighted
//! average and partial time step update of flux divergence. Source terms are added in
//! the HydroSrcTerms() function.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "eos/eos.hpp"
#include "hydro.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
//! \fn  void Hydro::Update
//  \brief Explicit RK update including flux divergence terms

TaskStatus Hydro::RKUpdate(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  Real &gam0 = pdriver->gam0[stage-1];
  Real &gam1 = pdriver->gam1[stage-1];
  Real beta_dt = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nvar = nhydro + nscalars;
  auto u0_ = u0;
  auto u1_ = u1;
  auto flx1 = uflx.x1f;
  auto flx2 = uflx.x2f;
  auto flx3 = uflx.x3f;
  auto &geom = pmy_pack->pgeom->geom_data;

  // hierarchical parallel loop that updates conserved variables to intermediate step
  // using weights and fractional time step appropriate to stages of time-integrator.
  // Vector inner loop used for good performance on cpus
  //
  // Generic area-weighted/volume-normalized divergence, dU/dt = -(1/V)*div(A*F): reduces
  // exactly to the previous uniform-Cartesian (flx[i+1]-flx[i])/dx formula since, for
  // coord=cartesian, Area1=dx2*dx3, Area2=dx1*dx3, Area3=dx1*dx2, Vol=dx1*dx2*dx3 (see
  // geometry_cartesian.cpp), so e.g. Area1(i+1)*flx1(i+1)-Area1(i)*flx1(i) all divided by
  // Vol telescopes back to (flx1(i+1)-flx1(i))/dx1. For curvilinear coordinates, Area1/2/3
  // and Vol vary with position (see geometry_cylindrical.cpp etc., Task B1-B3) -- this is
  // the only change needed in this kernel to support them; no coordinate-system branching
  // is introduced here (see mesh_geometry.hpp for the confinement principle).
  int scr_level = 0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1);

  par_for_outer("h_update",DevExeSpace(),scr_size,scr_level,0,nmb1,0,nvar-1,ks,ke,js,je,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int n, const int k, const int j) {
    ScrArray1D<Real> divf(member.team_scratch(scr_level), ncells1);

    // compute d(A1*F1)
    par_for_inner(member, is, ie, [&](const int i) {
      divf(i) = geom.Area1(m,k,j,i+1)*flx1(m,n,k,j,i+1)
              - geom.Area1(m,k,j,i)  *flx1(m,n,k,j,i);
    });
    member.team_barrier();

    // Add d(A2*F2)
    // Fluxes must be summed in pairs to symmetrize round-off error in each dir
    if (multi_d) {
      par_for_inner(member, is, ie, [&](const int i) {
        divf(i) += geom.Area2(m,k,j+1,i)*flx2(m,n,k,j+1,i)
                 - geom.Area2(m,k,j,i)  *flx2(m,n,k,j,i);
      });
      member.team_barrier();
    }

    // Add d(A3*F3)
    // Fluxes must be summed in pairs to symmetrize round-off error in each dir
    if (three_d) {
      par_for_inner(member, is, ie, [&](const int i) {
        divf(i) += geom.Area3(m,k+1,j,i)*flx3(m,n,k+1,j,i)
                 - geom.Area3(m,k,j,i)  *flx3(m,n,k,j,i);
      });
      member.team_barrier();
    }

    par_for_inner(member, is, ie, [&](const int i) {
      u0_(m,n,k,j,i) = gam0*u0_(m,n,k,j,i) + gam1*u1_(m,n,k,j,i)
                     - beta_dt*divf(i)/geom.Vol(m,k,j,i);
    });
  });
  return TaskStatus::complete;
}
} // namespace hydro
