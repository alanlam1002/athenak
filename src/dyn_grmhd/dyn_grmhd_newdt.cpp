//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyn_grmhd_newdt.cpp
//! \brief timestep calculation for dynamical-GR MHD (dyngr::DynGRMHDPS), replacing
//! mhd::MHD::NewTimeStep (mhd_newdt.cpp) for this module's own MHD_Newdt task.
//!
//! mhd::MHD::NewTimeStep hardcodes max_dv=1 (the speed of light) for every
//! is_dynamical_relativistic_ run, regardless of the actual fluid state -- always
//! correct (light is the fastest signal speed in GR) but far more conservative than
//! necessary once a real fast magnetosonic speed can be computed, exactly the gap
//! PR #698 (github.com/IAS-Astrophysics/athenak/pull/698) closes for the *static*-
//! background GR path (is_general_relativistic_, e.g. a fixed Kerr-Schild metric via
//! coordinates/cartesian_ks.hpp's ComputeMetricAndInverse) behind a new <time>/gr_dt
//! flag, using mhd's own ideal-gas-only EquationOfState (eos.IdealGRMHDFastSpeeds).
//! That PR does not touch is_dynamical_relativistic_ at all -- CFC/z4c runs (this
//! module) still fall through to max_dv=1 unconditionally.
//!
//! This file extends the same <time>/gr_dt idea to the dynamical-GR path, but can't
//! reuse mhd_newdt.cpp's approach verbatim for two reasons: (1) the metric here is
//! the actual evolved/solved pointwise ADM data (padm->adm.g_dd/beta_u/alpha) rather
//! than a fixed analytic background evaluated via ComputeMetricAndInverse -- there is
//! no closed-form metric to evaluate at an arbitrary point; and (2) dyn_grmhd uses the
//! primitive-solver EOS infrastructure (PrimitiveSolverHydro<EOSPolicy, ErrorPolicy>,
//! supporting piecewise-polytrope/tabulated/hybrid EOS, not just ideal gas), so the
//! wavespeed calculation must go through PrimitiveSolverHydro::
//! GetGRFastMagnetosonicSpeeds (already used identically by this module's own Riemann
//! solvers, src/dyn_grmhd/rsolvers/{llf,hlle}_dyn_grmhd.hpp) rather than
//! EquationOfState::IdealGRMHDFastSpeeds.
//!
//! gr_dt defaults to false (same key/default as PR #698's Hydro/MHD flag), preserving
//! the old max_dv=1 behavior until a run opts in.

#include <math.h>

#include <limits>
#include <algorithm>

#include "athena.hpp"
#include "mesh/mesh.hpp"
// dyn_grmhd.hpp and eos/primitive_solver_hyd.hpp include each other (the latter needs
// DynGRMHDPS, the former needs PrimitiveSolverHydro) -- dyn_grmhd.hpp must be included
// first in any translation unit, exactly as dyn_grmhd_fluxes.cpp does it, or
// PrimitiveSolverHydro is still an incomplete type when dyn_grmhd.hpp's own body (via
// its include of primitive_solver_hyd.hpp) tries to use it.
#include "dyn_grmhd.hpp"
#include "driver/driver.hpp"
#include "coordinates/adm.hpp"
#include "eos/eos.hpp"
#include "eos/primitive-solver/geom_math.hpp"
#include "mhd/mhd.hpp"
#include "diffusion/conduction.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/resistivity.hpp"
#include "srcterms/srcterms.hpp"

namespace dyngr {

//----------------------------------------------------------------------------------------
//! \fn TaskStatus DynGRMHDPS<EOSPolicy, ErrorPolicy>::NewTimeStep(Driver*, int)
//! \brief calculate the minimum timestep within a MeshBlockPack for dynamical-GR MHD

template<class EOSPolicy, class ErrorPolicy>
TaskStatus DynGRMHDPS<EOSPolicy, ErrorPolicy>::NewTimeStep(Driver *pdrive, int stage) {
  if (stage != (pdrive->nexp_stages)) {
    return TaskStatus::complete;  // only execute last stage
  }

  mhd::MHD *pmhd = pmy_pack->pmhd;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;

  Real dt1 = std::numeric_limits<float>::max();
  Real dt2 = std::numeric_limits<float>::max();
  Real dt3 = std::numeric_limits<float>::max();

  // capture class/struct members for the kernel (never capture `this` directly)
  auto &w0_ = pmhd->w0;
  auto &bcc0_ = pmhd->bcc0;
  auto &adm = pmy_pack->padm->adm;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &dyn_eos_ = eos;
  bool gr_dt_ = gr_dt;
  int nhyd = pmhd->nmhd;
  int nscal = pmhd->nscalars;
  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;

  if (pdrive->time_evolution == TimeEvolution::kinematic) {
    // find smallest (dx/v) in each direction for advection problems (unreachable in
    // practice -- dyn_grmhd always runs with dynamic evolution -- kept for parity
    // with mhd::MHD::NewTimeStep's structure).
    Kokkos::parallel_reduce("DynGRMHDNewdt1", Kokkos::RangePolicy<>(DevExeSpace(),
    0, nmkji), KOKKOS_LAMBDA(const int &idx, Real &min_dt1, Real &min_dt2,
    Real &min_dt3) {
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      min_dt1 = fmin((mbsize.d_view(m).dx1/fabs(w0_(m,IVX,k,j,i))), min_dt1);
      min_dt2 = fmin((mbsize.d_view(m).dx2/fabs(w0_(m,IVY,k,j,i))), min_dt2);
      min_dt3 = fmin((mbsize.d_view(m).dx3/fabs(w0_(m,IVZ,k,j,i))), min_dt3);
    }, Kokkos::Min<Real>(dt1), Kokkos::Min<Real>(dt2), Kokkos::Min<Real>(dt3));
  } else {
    Kokkos::parallel_reduce("DynGRMHDNewdt2", Kokkos::RangePolicy<>(DevExeSpace(),
    0, nmkji), KOKKOS_LAMBDA(const int &idx, Real &min_dt1, Real &min_dt2,
    Real &min_dt3) {
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      // Conservative fallback: the speed of light, always correct (see file doc
      // comment). Overwritten below when gr_dt_ requests the real wavespeed.
      Real max_dv1 = 1.0, max_dv2 = 1.0, max_dv3 = 1.0;

      if (gr_dt_) {
        // Pointwise dynamical 3-metric (unlike a fixed analytic background, this is
        // the actual solved/evolved metric at this cell).
        Real g3d[NSPMETRIC];
        g3d[S11] = adm.g_dd(m,0,0,k,j,i); g3d[S12] = adm.g_dd(m,0,1,k,j,i);
        g3d[S13] = adm.g_dd(m,0,2,k,j,i); g3d[S22] = adm.g_dd(m,1,1,k,j,i);
        g3d[S23] = adm.g_dd(m,1,2,k,j,i); g3d[S33] = adm.g_dd(m,2,2,k,j,i);
        Real beta_u[3] = {adm.beta_u(m,0,k,j,i), adm.beta_u(m,1,k,j,i),
                           adm.beta_u(m,2,k,j,i)};
        Real alpha = adm.alpha(m,k,j,i);
        Real sdetg = sqrt(Primitive::GetDeterminant(g3d));
        Real isdetg = 1.0/sdetg;

        // Pack primitives via this module's own primitive-solver EOS (not
        // EquationOfState::IdealGasPressure -- dyn_grmhd's pressure/temperature are
        // already primitive fields, and the EOS itself may not be an ideal gas).
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
        dyn_eos_.ps.GetEOS().ApplyPrimitiveFloor(prim[PRH], &prim[PVX], prim[PPR],
                                                  prim[PTM], &prim[PYF]);

        // Undensitized cell-centered field (bcc0 stores sqrt(detg)*B^i, matching
        // dyn_grmhd_fluxes.cpp/the Riemann solvers' own convention).
        Real Bu[3] = {bcc0_(m,IBX,k,j,i)*isdetg, bcc0_(m,IBY,k,j,i)*isdetg,
                      bcc0_(m,IBZ,k,j,i)*isdetg};

        // Comoving-frame b^2, mirroring SingleStateFlux's identical calculation
        // (src/dyn_grmhd/rsolvers/flux_dyn_grmhd.hpp).
        Real uu[3] = {prim[PVX], prim[PVY], prim[PVZ]};
        Real ud[3];
        Primitive::LowerVector(ud, uu, g3d);
        Real iWsq = 1.0/(1.0 + Primitive::Contract(uu, ud));
        Real ialpha = 1.0/alpha;
        Real bu0 = Primitive::Contract(Bu, ud)*ialpha;
        Real bsq = (Primitive::SquareVector(Bu, g3d) + SQR(alpha*bu0))*iWsq;

        // Fast magnetosonic speed in each direction (gii is that direction's
        // diagonal inverse-3-metric component, matching {llf,hlle}_dyn_grmhd.hpp's
        // own gii calculation via the determinant/cofactor shortcut).
        Real lp, lm;
        Real gii1 = (g3d[S22]*g3d[S33] - g3d[S23]*g3d[S23])*(isdetg*isdetg);
        dyn_eos_.GetGRFastMagnetosonicSpeeds(lp, lm, prim, bsq, g3d, beta_u, alpha,
                                             gii1, PVX);
        max_dv1 = fmax(fabs(lm), lp);

        Real gii2 = (g3d[S11]*g3d[S33] - g3d[S13]*g3d[S13])*(isdetg*isdetg);
        dyn_eos_.GetGRFastMagnetosonicSpeeds(lp, lm, prim, bsq, g3d, beta_u, alpha,
                                             gii2, PVY);
        max_dv2 = fmax(fabs(lm), lp);

        Real gii3 = (g3d[S11]*g3d[S22] - g3d[S12]*g3d[S12])*(isdetg*isdetg);
        dyn_eos_.GetGRFastMagnetosonicSpeeds(lp, lm, prim, bsq, g3d, beta_u, alpha,
                                             gii3, PVZ);
        max_dv3 = fmax(fabs(lm), lp);
      }

      min_dt1 = fmin((mbsize.d_view(m).dx1/max_dv1), min_dt1);
      min_dt2 = fmin((mbsize.d_view(m).dx2/max_dv2), min_dt2);
      min_dt3 = fmin((mbsize.d_view(m).dx3/max_dv3), min_dt3);
    }, Kokkos::Min<Real>(dt1), Kokkos::Min<Real>(dt2), Kokkos::Min<Real>(dt3));
  }

  // compute minimum of dt1/dt2/dt3 for 1D/2D/3D problems
  pmhd->dtnew = dt1;
  if (pmy_pack->pmesh->multi_d) { pmhd->dtnew = std::min(pmhd->dtnew, dt2); }
  if (pmy_pack->pmesh->three_d) { pmhd->dtnew = std::min(pmhd->dtnew, dt3); }

  // compute timestep for diffusion/source terms, exactly as mhd::MHD::NewTimeStep does
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

// Explicit instantiation for each (EOSPolicy, ErrorPolicy) combination, mirroring
// dyn_grmhd_fluxes.cpp's INSTANTIATE_CALC_FLUXES macro exactly (NewTimeStep isn't
// templated on DynGRMHD_RSolver, so only one instantiation per policy pair is needed).
#define INSTANTIATE_NEW_TIME_STEP(EOSPolicy, ErrorPolicy) \
template \
TaskStatus DynGRMHDPS<EOSPolicy, ErrorPolicy>::NewTimeStep(Driver *pdrive, int stage);

INSTANTIATE_NEW_TIME_STEP(Primitive::IdealGas, Primitive::ResetFloor)
INSTANTIATE_NEW_TIME_STEP(Primitive::PiecewisePolytrope, Primitive::ResetFloor)
INSTANTIATE_NEW_TIME_STEP(Primitive::EOSCompOSE<Primitive::NormalLogs>,
                          Primitive::ResetFloor)
INSTANTIATE_NEW_TIME_STEP(Primitive::EOSCompOSE<Primitive::NQTLogs>,
                          Primitive::ResetFloor)
INSTANTIATE_NEW_TIME_STEP(Primitive::EOSHybrid<Primitive::NormalLogs>,
                          Primitive::ResetFloor)
INSTANTIATE_NEW_TIME_STEP(Primitive::EOSHybrid<Primitive::NQTLogs>,
                          Primitive::ResetFloor)

}  // namespace dyngr
