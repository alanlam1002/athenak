//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_update.cpp
//! \brief Performs explicit update of MHD conserved variables (u0) for each stage of the
//! SSP RK integrators (e.g. RK1, RK2, RK3) implemented in AthenaK, using weighted average
//! and partial time update of flux divergence. Source terms are added in the
//! MHDSrcTerms() function.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "eos/eos.hpp"
#include "mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn  void MHD::Update
//  \brief Explicit RK update including flux divergence terms

TaskStatus MHD::RKUpdate(Driver *pdriver, int stage) {
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
  int nv1 = nmhd + nscalars - 1;
  auto u0_ = u0;
  auto u1_ = u1;
  auto flx1 = uflx.x1f;
  auto flx2 = uflx.x2f;
  auto flx3 = uflx.x3f;
  auto &geom = pmy_pack->pgeom->geom_data;

  // hierarchical parallel loop that updates conserved variables to intermediate step
  // using weights and fractional time step appropriate to stages of time-integrator used
  // Vector inner loop used for good performance on cpus
  //
  // Generic area-weighted/volume-normalized divergence -- see the identical rewrite (and
  // its rationale) in src/hydro/hydro_update.cpp for the full derivation. Reduces exactly
  // to the previous uniform-Cartesian formula for coord=cartesian.
  int scr_level = 0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1);

  par_for_outer("mhd_update",DevExeSpace(),scr_size,scr_level,0,nmb1,0,nv1,ks,ke,js,je,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int n, const int k, const int j) {
    ScrArray1D<Real> divf(member.team_scratch(scr_level), ncells1);

    // Hoist the transverse (j,k-indexed) geometry factors out of the inner i loop --
    // they are invariant along i, and are also identical for every variable n. The
    // compiler cannot hoist them itself: Kokkos Views carry no __restrict__, so it must
    // assume the scratch store to divf(i) below could alias them. Written out as
    // factored a*i/a*j/a*k reads rather than Area1/Area2/Area3 calls for that reason
    // only -- the value computed is the same product, just associated so the
    // loop-invariant part is evaluated once per (m,n,k,j) instead of once per cell.

    // compute d(A1*F1)
    const Real a1jk = geom.a1j(m,j)*geom.a1k(m,k);
    par_for_inner(member, is, ie, [&](const int i) {
      divf(i) = a1jk*(geom.a1i(m,i+1)*flx1(m,n,k,j,i+1)
                    - geom.a1i(m,i)  *flx1(m,n,k,j,i));
    });
    member.team_barrier();

    // Add d(A2*F2)
    // Fluxes must be summed in pairs to symmetrize round-off error in each dir
    if (multi_d) {
      const Real a2jp = geom.a2j(m,j+1)*geom.a2k(m,k);
      const Real a2jm = geom.a2j(m,j)  *geom.a2k(m,k);
      par_for_inner(member, is, ie, [&](const int i) {
        divf(i) += geom.a2i(m,i)*(a2jp*flx2(m,n,k,j+1,i)
                                - a2jm*flx2(m,n,k,j,i));
      });
      member.team_barrier();
    }

    // Add d(A3*F3)
    // Fluxes must be summed in pairs to symmetrize round-off error in each dir
    if (three_d) {
      const Real a3kp = geom.a3j(m,j)*geom.a3k(m,k+1);
      const Real a3km = geom.a3j(m,j)*geom.a3k(m,k);
      par_for_inner(member, is, ie, [&](const int i) {
        divf(i) += geom.a3i(m,i)*(a3kp*flx3(m,n,k+1,j,i)
                                - a3km*flx3(m,n,k,j,i));
      });
      member.team_barrier();
    }

    const Real vjk = geom.vj(m,j)*geom.vk(m,k);
    par_for_inner(member, is, ie, [&](const int i) {
      u0_(m,n,k,j,i) = gam0*u0_(m,n,k,j,i) + gam1*u1_(m,n,k,j,i)
                     - beta_dt*divf(i)/(vjk*geom.vi(m,i));
    });
  });
  return TaskStatus::complete;
}
} // namespace mhd
