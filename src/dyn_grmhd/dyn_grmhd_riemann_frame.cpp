//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyn_grmhd_riemann_frame.cpp
//! \brief <mhd> riemann_frame = boosted (R-168, user-approved 2026-10-30; "level R",
//! research/notes/reference_congruence §8.2), the parts outside the Riemann solver
//! (rsolvers/hlle_frame_dyn_grmhd.hpp):
//!   - riemann_frame_recon = R2 (R-168 addendum): before the sweeps, w0's velocity is
//!     replaced by its components u^(i) = e^(i)_mu u^mu in the reference tetrad at the cell
//!     centre (U, then Gram-Schmidt of d_x, d_y, d_z in the cell metric, ghosts included; the
//!     same for every face direction); the solver reads the reconstructed u^(i) in the face's
//!     reference tetrad and rotates into its Riemann tetrad. w0's velocity is saved first and
//!     restored bitwise after the last sweep, so FOFC (the R1 route), the C2P and everything
//!     else see w0;
//!   - the checks: the magnetic field must vanish (the frame solver is hydrodynamic), and a
//!     face where grid_rest U is not timelike (alpha^2 <= beta_i beta^i) stops the run;
//!   - the CFL (<time> gr_dt): the coordinate speed k^j/k^t of k = U + lambda e_(1), lambda the
//!     frame fast speeds, with U and the tetrad of the cell metric.

#include <math.h>

#include <algorithm>
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "dyn_grmhd.hpp"
#include "driver/driver.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "diffusion/conduction.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/resistivity.hpp"
#include "srcterms/srcterms.hpp"
#include "dyn_grmhd/rsolvers/hlle_frame_dyn_grmhd.hpp"

#include "eos/primitive_solver_hyd.hpp"
#include "eos/primitive-solver/idealgas.hpp"
#include "eos/primitive-solver/eos_compose.hpp"
#include "eos/primitive-solver/eos_hybrid.hpp"
#include "eos/primitive-solver/piecewise_polytrope.hpp"
#include "eos/primitive-solver/reset_floor.hpp"

namespace dyngr {

namespace {
//! the cell metric (adm at the cell centre)
KOKKOS_INLINE_FUNCTION
void CellMetric(const adm::ADM::ADM_vars &adm, const int m, const int k, const int j,
                const int i, Real g3d[NSPMETRIC], Real beta_u[3], Real &alpha) {
  g3d[S11] = adm.g_dd(m, 0, 0, k, j, i); g3d[S12] = adm.g_dd(m, 0, 1, k, j, i);
  g3d[S13] = adm.g_dd(m, 0, 2, k, j, i); g3d[S22] = adm.g_dd(m, 1, 1, k, j, i);
  g3d[S23] = adm.g_dd(m, 1, 2, k, j, i); g3d[S33] = adm.g_dd(m, 2, 2, k, j, i);
  for (int a = 0; a < 3; ++a) beta_u[a] = adm.beta_u(m, a, k, j, i);
  alpha = adm.alpha(m, k, j, i);
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void DynGRMHD::RFrameCellVelocities
//! \brief R2: w0(IVX..IVZ) <- u^(1..3) in the cell-centre reference tetrad, every cell
//! (ghosts included), the velocity saved first; called once per stage (dir unused)
void DynGRMHD::RFrameCellVelocities(int dir) {
  auto &w0 = pmy_pack->pmhd->w0;
  auto &sv = rf_vsave;
  auto &adm = pmy_pack->padm->adm;
  auto &err = rf_err;
  const bool ugrid = rf_ugrid;
  const int nmb = pmy_pack->nmb_thispack;
  par_for("rframe_r2_fwd", DevExeSpace(), 0, nmb-1, 0, w0.extent_int(2)-1,
          0, w0.extent_int(3)-1, 0, w0.extent_int(4)-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    for (int a = 0; a < 3; ++a) sv(m, a, k, j, i) = w0(m, IVX + a, k, j, i);
    Real g3d[NSPMETRIC], beta_u[3], alpha;
    CellMetric(adm, m, k, j, i, g3d, beta_u, alpha);
    FrameFace f;
    if (!BuildFrameFace(g3d, beta_u, alpha, 0, ugrid, f)) Kokkos::atomic_add(&err(0), 1);
    Real prim[NPRIM];
    for (int a = 0; a < 3; ++a) prim[PVX + a] = sv(m, a, k, j, i);
    Real u[4];
    FourVelocity(prim, g3d, beta_u, alpha, u);
    for (int a = 1; a < 4; ++a) {   // e^(i)_mu = g_{mu nu} R[i][nu] (spatial)
      Real r = 0.0;
      for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) r += f.g[mu][nu]*f.R[a][nu]*u[mu];
      }
      w0(m, IVX + a - 1, k, j, i) = r;
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn void DynGRMHD::RFrameRestoreVelocities
//! \brief R2: w0(IVX..IVZ) <- the saved velocity (bitwise)
void DynGRMHD::RFrameRestoreVelocities() {
  auto &w0 = pmy_pack->pmhd->w0;
  auto &sv = rf_vsave;
  const int nmb = pmy_pack->nmb_thispack;
  par_for("rframe_r2_bwd", DevExeSpace(), 0, nmb-1, 0, w0.extent_int(2)-1,
          0, w0.extent_int(3)-1, 0, w0.extent_int(4)-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    for (int a = 0; a < 3; ++a) w0(m, IVX + a, k, j, i) = sv(m, a, k, j, i);
  });
}

//----------------------------------------------------------------------------------------
//! \fn void DynGRMHD::RFrameStageCheck
//! \brief before the sweeps: the field check (first call; the frame solver's EMFs are zero, so
//! a zero field stays zero) and err reset; after FOFC: stop if any face had a non-timelike U
void DynGRMHD::RFrameStageCheck(bool after) {
  if (!after) {
    if (!rf_bchecked) {
      auto &bcc = pmy_pack->pmhd->bcc0;
      const int nmb = pmy_pack->nmb_thispack;
      const int n4 = bcc.extent_int(4), n3 = bcc.extent_int(3), n2 = bcc.extent_int(2);
      const int ntot = nmb*3*n2*n3*n4;
      Real bmax = 0.0;
      Kokkos::parallel_reduce("rframe_bcheck", Kokkos::RangePolicy<>(DevExeSpace(), 0, ntot),
      KOKKOS_LAMBDA(const int idx, Real &mx) {
        const int i = idx % n4, j = (idx/n4) % n3, k = (idx/(n4*n3)) % n2;
        const int c = (idx/(n4*n3*n2)) % 3, m = idx/(n4*n3*n2*3);
        mx = fmax(mx, fabs(bcc(m, c, k, j, i)));
      }, Kokkos::Max<Real>(bmax));
#if MPI_PARALLEL_ENABLED
      MPI_Allreduce(MPI_IN_PLACE, &bmax, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
      if (bmax > 0.0) {
        if (global_variable::my_rank == 0) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "<mhd> riemann_frame = boosted is hydrodynamic: the "
                    << "magnetic field must vanish (max |B| = " << bmax << ")" << std::endl;
        }
        std::exit(EXIT_FAILURE);
      }
      rf_bchecked = true;
    }
    Kokkos::deep_copy(rf_err, 0);
    return;
  }
  auto h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), rf_err);
  int bad = h(0);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (bad > 0) {
    if (global_variable::my_rank == 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "<mhd> riemann_frame_U = grid_rest: U = d_t/sqrt(-g_tt) is not timelike "
                << "(alpha^2 <= beta_i beta^i) at " << bad << " faces/cells, cycle "
                << pmy_pack->pmesh->ncycle << std::endl;
    }
    std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus DynGRMHDPS::NewTimeStepFrame
//! \brief NewTimeStep's dynamic gr_dt branch with the frame speeds: per cell and direction,
//! max over lambda_+- of |k^j/k^t|, k = U + lambda e_(1) (cell metric); the cell state as
//! NewTimeStep prepares it (floor, dt_floor_keep_velocity, atmosphere frame velocity). The
//! dt_limiter_report lines are not produced in this mode.
template<class EOSPolicy, class ErrorPolicy>
TaskStatus DynGRMHDPS<EOSPolicy, ErrorPolicy>::NewTimeStepFrame(Driver *pdrive, int stage) {
  mhd::MHD *pmhd = pmy_pack->pmhd;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  Real dt1 = std::numeric_limits<float>::max();
  Real dt2 = std::numeric_limits<float>::max();
  Real dt3 = std::numeric_limits<float>::max();
  auto &w0_ = pmhd->w0;
  auto &adm = pmy_pack->padm->adm;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &dyn_eos_ = eos;
  const bool skip_exc_ = dt_skip_excised && pmy_pack->pcoord->coord_data.bh_excise;
  const bool keepv_ = dt_floor_keep_velocity;
  auto &exc_flux_ = pmy_pack->pcoord->excision_flux;
  const bool atm_fv_ = eos.atm_frame_vel;
  const Real fv0 = pmy_pack->padm->frame_vel_u[0], fv1 = pmy_pack->padm->frame_vel_u[1];
  const Real fv2 = pmy_pack->padm->frame_vel_u[2];
  const Real fg2_ = eos.atm_frame_gate*eos.atm_frame_gate;
  const Real fgx_ = pmy_pack->padm->bc_center[0], fgy_ = pmy_pack->padm->bc_center[1];
  const Real fgz_ = pmy_pack->padm->bc_center[2];
  const bool ugrid = rf_ugrid;
  const bool d2 = pmy_pack->pmesh->multi_d, d3 = pmy_pack->pmesh->three_d;
  int nhyd = pmhd->nmhd;
  int nscal = pmhd->nscalars;
  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;

  Kokkos::parallel_reduce("DynGRMHDNewdtFrame", Kokkos::RangePolicy<>(DevExeSpace(),
  0, nmkji), KOKKOS_LAMBDA(const int &idx, Real &min_dt1, Real &min_dt2,
  Real &min_dt3) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    if (skip_exc_ && exc_flux_(m,k,j,i)) return;

    Real g3d[NSPMETRIC], beta_u[3], alpha;
    CellMetric(adm, m, k, j, i, g3d, beta_u, alpha);
    Real mb = dyn_eos_.ps.GetEOS().GetBaryonMass();
    Real prim[NPRIM];
    prim[PRH] = w0_(m,IDN,k,j,i)/mb;
    prim[PVX] = w0_(m,IVX,k,j,i);
    prim[PVY] = w0_(m,IVY,k,j,i);
    prim[PVZ] = w0_(m,IVZ,k,j,i);
    for (int n = 0; n < nscal; ++n) {
      prim[PYF+n] = w0_(m, nhyd+n, k,j,i);
    }
    dyn_eos_.ps.GetEOS().ApplyDensityLimits(prim[PRH]);
    dyn_eos_.ps.GetEOS().ApplySpeciesLimits(&prim[PYF]);
    prim[PPR] = w0_(m,IPR,k,j,i);
    prim[PTM] = dyn_eos_.ps.GetEOS().GetTemperatureFromP(prim[PRH], prim[PPR],
                                                          &prim[PYF]);
    bool floored_ = dyn_eos_.ps.GetEOS().ApplyPrimitiveFloor(prim[PRH], &prim[PVX],
                                              prim[PPR], prim[PTM], &prim[PYF]);
    if (floored_ && keepv_) {   // as NewTimeStep
      prim[PVX] = w0_(m,IVX,k,j,i);
      prim[PVY] = w0_(m,IVY,k,j,i);
      prim[PVZ] = w0_(m,IVZ,k,j,i);
    }
    bool fv_here = floored_ && atm_fv_;
    if (fv_here && fg2_ > 0.0) {
      Real xg = CellCenterX(i-is, nx1, mbsize.d_view(m).x1min,
                            mbsize.d_view(m).x1max) - fgx_;
      Real yg = CellCenterX(j-js, nx2, mbsize.d_view(m).x2min,
                            mbsize.d_view(m).x2max) - fgy_;
      Real zg = CellCenterX(k-ks, nx3, mbsize.d_view(m).x3min,
                            mbsize.d_view(m).x3max) - fgz_;
      fv_here = (xg*xg + yg*yg + zg*zg > fg2_);
    }
    if (fv_here) {
      Real vsq = g3d[S11]*fv0*fv0 + g3d[S22]*fv1*fv1 + g3d[S33]*fv2*fv2
               + 2.0*(g3d[S12]*fv0*fv1 + g3d[S13]*fv0*fv2 + g3d[S23]*fv1*fv2);
      Real wl = 1.0/sqrt(1.0 - fmin(vsq, 0.99));
      prim[PVX] = wl*fv0; prim[PVY] = wl*fv1; prim[PVZ] = wl*fv2;
    }

    Real max_dv[3] = {1.0, 1.0, 1.0};
    const int nd = d3 ? 3 : (d2 ? 2 : 1);
    for (int d = 0; d < nd; ++d) {
      FrameFace f;
      BuildFrameFace(g3d, beta_u, alpha, d, ugrid, f);
      Real Q[4], F[4], qd, fd, lp, lm;
      FrameState(dyn_eos_, prim, g3d, beta_u, alpha, f, Q, F, qd, fd, lp, lm);
      const Real vp = (f.E[0][d+1] + lp*f.E[1][d+1])/(f.E[0][0] + lp*f.E[1][0]);
      const Real vm = (f.E[0][d+1] + lm*f.E[1][d+1])/(f.E[0][0] + lm*f.E[1][0]);
      max_dv[d] = fmax(fabs(vp), fabs(vm));
    }
    min_dt1 = fmin((mbsize.d_view(m).dx1/max_dv[0]), min_dt1);
    min_dt2 = fmin((mbsize.d_view(m).dx2/max_dv[1]), min_dt2);
    min_dt3 = fmin((mbsize.d_view(m).dx3/max_dv[2]), min_dt3);
  }, Kokkos::Min<Real>(dt1), Kokkos::Min<Real>(dt2), Kokkos::Min<Real>(dt3));

  // as NewTimeStep
  pmhd->dtnew = dt1;
  if (pmy_pack->pmesh->multi_d) { pmhd->dtnew = std::min(pmhd->dtnew, dt2); }
  if (pmy_pack->pmesh->three_d) { pmhd->dtnew = std::min(pmhd->dtnew, dt3); }
  if (pmhd->pcond != nullptr) {
    pmhd->pcond->NewTimeStep(pmhd->w0, pmhd->peos->eos_data);
  }
  if (pmhd->pvisc != nullptr) {
    pmhd->pvisc->NewTimeStep(pmhd->w0, pmhd->peos->eos_data);
  }
  if (pmhd->presist != nullptr) {
    pmhd->presist->NewTimeStep(pmhd->w0, pmhd->peos->eos_data);
  }
  if (pmhd->psrc != nullptr) {
    pmhd->psrc->NewTimeStep(pmhd->w0, pmhd->peos->eos_data);
  }
  return TaskStatus::complete;
}

#define INSTANTIATE_NEW_TIME_STEP_FRAME(EOSPolicy, ErrorPolicy) \
template \
TaskStatus DynGRMHDPS<EOSPolicy, ErrorPolicy>::NewTimeStepFrame(Driver *pdrive, int stage);

INSTANTIATE_NEW_TIME_STEP_FRAME(Primitive::IdealGas, Primitive::ResetFloor)
INSTANTIATE_NEW_TIME_STEP_FRAME(Primitive::PiecewisePolytrope, Primitive::ResetFloor)
INSTANTIATE_NEW_TIME_STEP_FRAME(Primitive::EOSCompOSE<Primitive::NormalLogs>,
                                Primitive::ResetFloor)
INSTANTIATE_NEW_TIME_STEP_FRAME(Primitive::EOSCompOSE<Primitive::NQTLogs>,
                                Primitive::ResetFloor)
INSTANTIATE_NEW_TIME_STEP_FRAME(Primitive::EOSHybrid<Primitive::NormalLogs>,
                                Primitive::ResetFloor)
INSTANTIATE_NEW_TIME_STEP_FRAME(Primitive::EOSHybrid<Primitive::NQTLogs>,
                                Primitive::ResetFloor)

}  // namespace dyngr
