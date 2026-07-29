//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyngr_rns_st.cpp
//  \brief Problem generator for a scalarized neutron star (massive scalar-tensor
//  gravity) built from RNS-ST equilibrium initial data, ported from
//  ~/SACRA_2D/SACRA_MPI/read_grass_st.f90 (see rns_st_reader.hpp). Requires both
//  <adm>+<z4c> (full dynamical Z4c -- scalar-tensor gravity is inherently dynamical,
//  unlike the plain-GR TOV pgen which also supports a static background) and
//  <scalarfield>.
//  Compile with '-D PROBLEM=dyn_grmhd/dyngr_rns_st' to enroll as user-specific pgen.

#include <math.h>

#include <limits>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "scalar_field/scalar_field.hpp"
#include "utils/tov/tov_utils.hpp"
#include "utils/tov/tov_polytrope.hpp"
#include "utils/tov/tov_tabulated.hpp"
#include "utils/tov/tov_piecewise_poly.hpp"
#include "rns_st_reader.hpp"

// Prototype for user-defined history function
void RnsStHistory(HistoryData *pdata, Mesh *pm);

//----------------------------------------------------------------------------------------
//! \fn void SetupRnsSt
//  \brief Loads the RNS-ST ID file and fills ADM/Z4c/hydro/scalar-field initial data.
//  Host-only: the ID file's interpolation (rns_st::RnsStData) is a plain-CPU port of
//  the Fortran reader, so this mirrors the host-fill-then-deep_copy pattern used by the
//  other external-ID pgens (elliptica/lorene/sgrid/kadath), not dyngr_tov.cpp's
//  device-side par_for (which works directly from a closed-form/ODE solution instead).

template<class TOVEOS>
void SetupRnsSt(ParameterInput *pin, Mesh *pmy_mesh_) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  std::string id_file = pin->GetString("problem", "id_file");
  rns_st::RnsStData data(id_file);

  TOVEOS eos{pin};

  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;

  auto &u_adm = pmbp->padm->u_adm;
  auto &w0 = pmbp->pmhd->w0;
  auto &u_sf = pmbp->pscalarfield->u0;
  int &nvars = pmbp->pmhd->nmhd;
  int &nscal = pmbp->pmhd->nscalars;

  // Because the Fortran-ported interpolator only runs on the CPU, build the data on a
  // host mirror first, then move it to the GPU -- same pattern as the elliptica/
  // lorene/sgrid/kadath external-ID pgens.
  HostArray5D<Real>::HostMirror host_u_adm = Kokkos::create_mirror_view(u_adm);
  HostArray5D<Real>::HostMirror host_w0 = Kokkos::create_mirror_view(w0);
  HostArray5D<Real>::HostMirror host_u_z4c = Kokkos::create_mirror_view(pmbp->pz4c->u0);
  HostArray5D<Real>::HostMirror host_u_sf = Kokkos::create_mirror_view(u_sf);

  // Gauge variables (alpha, beta_u) live in Z4c's own evolved state, not ADM's --
  // ADMToZ4c only derives the geometric/conformal part (chi, gamma_tilde_ij, Atilde_ij,
  // Gamma_tilde^i) from g_dd/vK_dd, it never touches the gauge.
  adm::ADM::ADMhost_vars host_adm;
  host_adm.alpha.InitWithShallowSlice(host_u_z4c, z4c::Z4c::I_Z4C_ALPHA);
  host_adm.beta_u.InitWithShallowSlice(host_u_z4c,
      z4c::Z4c::I_Z4C_BETAX, z4c::Z4c::I_Z4C_BETAZ);
  host_adm.g_dd.InitWithShallowSlice(host_u_adm,
      adm::ADM::I_ADM_GXX, adm::ADM::I_ADM_GZZ);
  host_adm.vK_dd.InitWithShallowSlice(host_u_adm,
      adm::ADM::I_ADM_KXX, adm::ADM::I_ADM_KZZ);

  for (int m = 0; m < nmb; ++m) {
    Real &x1min = size.h_view(m).x1min;
    Real &x1max = size.h_view(m).x1max;
    Real &x2min = size.h_view(m).x2min;
    Real &x2max = size.h_view(m).x2max;
    Real &x3min = size.h_view(m).x3min;
    Real &x3max = size.h_view(m).x3max;
    for (int k = 0; k < n3; ++k) {
      Real x3v = CellCenterX(k - indcs.ks, indcs.nx3, x3min, x3max);
      for (int j = 0; j < n2; ++j) {
        Real x2v = CellCenterX(j - indcs.js, indcs.nx2, x2min, x2max);
        for (int i = 0; i < n1; ++i) {
          Real x1v = CellCenterX(i - indcs.is, indcs.nx1, x1min, x1max);

          rns_st::RnsStData::Point pt;
          data.Interpolate(x1v, x2v, x3v, &pt);

          // Conformal -> physical metric/extrinsic-curvature. K=0 identically in this
          // data (maximal slicing), so no trace term is added to vK_dd.
          Real psi4 = std::exp(4.0*pt.phi_bssn);
          Real gxx = psi4*(1.0 + pt.h_xx);
          Real gyy = psi4*(1.0 + pt.h_yy);
          Real gzz = psi4*(1.0 + pt.h_zz);
          Real gxy = psi4*pt.h_xy;
          Real gxz = psi4*pt.h_xz;
          Real gyz = psi4*pt.h_yz;

          host_adm.alpha(m, k, j, i) = pt.alpha;
          host_adm.beta_u(m, 0, k, j, i) = pt.beta_u[0];
          host_adm.beta_u(m, 1, k, j, i) = pt.beta_u[1];
          host_adm.beta_u(m, 2, k, j, i) = pt.beta_u[2];

          host_adm.g_dd(m, 0, 0, k, j, i) = gxx;
          host_adm.g_dd(m, 0, 1, k, j, i) = gxy;
          host_adm.g_dd(m, 0, 2, k, j, i) = gxz;
          host_adm.g_dd(m, 1, 1, k, j, i) = gyy;
          host_adm.g_dd(m, 1, 2, k, j, i) = gyz;
          host_adm.g_dd(m, 2, 2, k, j, i) = gzz;

          host_adm.vK_dd(m, 0, 0, k, j, i) = psi4*pt.Atilde_xx;
          host_adm.vK_dd(m, 0, 1, k, j, i) = psi4*pt.Atilde_xy;
          host_adm.vK_dd(m, 0, 2, k, j, i) = psi4*pt.Atilde_xz;
          host_adm.vK_dd(m, 1, 1, k, j, i) = psi4*pt.Atilde_yy;
          host_adm.vK_dd(m, 1, 2, k, j, i) = psi4*pt.Atilde_yz;
          host_adm.vK_dd(m, 2, 2, k, j, i) = psi4*pt.Atilde_zz;

          Real rho = pt.rho;
          Real pres = 0.0;
          Real vu[3] = {0.0, 0.0, 0.0};
          if (rho > 0.0) {
            pres = eos.template GetPFromRho<tov::LocationTag::Host>(rho);
            // Raise the interpolator's covariant velocity u_i to AthenaK's
            // contravariant primitive convention u^i = g^ij u_j.
            Real det = adm::SpatialDet(gxx, gxy, gxz, gyy, gyz, gzz);
            Real uxx, uxy, uxz, uyy, uyz, uzz;
            adm::SpatialInv(1.0/det, gxx, gxy, gxz, gyy, gyz, gzz,
                             &uxx, &uxy, &uxz, &uyy, &uyz, &uzz);
            vu[0] = uxx*pt.u_d[0] + uxy*pt.u_d[1] + uxz*pt.u_d[2];
            vu[1] = uxy*pt.u_d[0] + uyy*pt.u_d[1] + uyz*pt.u_d[2];
            vu[2] = uxz*pt.u_d[0] + uyz*pt.u_d[1] + uzz*pt.u_d[2];
          } else {
            rho = 0.0;
          }

          host_w0(m, IDN, k, j, i) = rho;
          host_w0(m, IPR, k, j, i) = pres;
          host_w0(m, IVX, k, j, i) = vu[0];
          host_w0(m, IVY, k, j, i) = vu[1];
          host_w0(m, IVZ, k, j, i) = vu[2];
          if (nscal >= 1) {
            host_w0(m, nvars, k, j, i) = 0.0;
          }

          host_u_sf(m, scalarfield::ScalarField::I_SF_SPHI, k, j, i) = pt.sphi;
          // Pi (scalar momentum) is set to zero: exact for this non-rotating (J=0)
          // star (Pi = -alpha^-1 beta^i di(sphi), and beta^i=0 identically here);
          // only an approximation for a genuinely rotating RNS-ST model.
          host_u_sf(m, scalarfield::ScalarField::I_SF_PI, k, j, i) = 0.0;
        }
      }
    }
  }

  Kokkos::deep_copy(u_adm, host_u_adm);
  Kokkos::deep_copy(w0, host_w0);
  Kokkos::deep_copy(pmbp->pz4c->u0, host_u_z4c);
  Kokkos::deep_copy(u_sf, host_u_sf);

  // This star is unmagnetized -- the ID carries no B-field data (see plan). AthenaK
  // does not zero-initialize device memory by default, so b0/bcc0 must be set
  // explicitly or they're left as garbage, which otherwise poisons the primitive
  // solver with NaN from the very first step.
  Kokkos::deep_copy(pmbp->pmhd->b0.x1f, 0.0);
  Kokkos::deep_copy(pmbp->pmhd->b0.x2f, 0.0);
  Kokkos::deep_copy(pmbp->pmhd->b0.x3f, 0.0);
  Kokkos::deep_copy(pmbp->pmhd->bcc0, 0.0);
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//  \brief Sets initial conditions for a scalarized-NS RNS-ST star.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "RNS-ST scalarized-star problem requires a <z4c> block (full "
              << "dynamical Z4c -- scalar-tensor gravity has no static-background mode)"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->pscalarfield == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "RNS-ST scalarized-star problem requires a <scalarfield> block"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  user_hist_func = &RnsStHistory;

  // No restart-path special-casing: unlike dyngr_tov.cpp's analytic TOVStar (which
  // must be regenerated on restart for adaptive/dynamic backgrounds), this pgen only
  // ever sets initial data once -- evolution afterward proceeds from checkpointed
  // Z4c/hydro/scalar-field state like any other dynamical run.
  if (restart) { return; }

  if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_ideal) {
    SetupRnsSt<tov::PolytropeEOS>(pin, pmy_mesh_);
  } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_compose) {
    SetupRnsSt<tov::TabulatedEOS>(pin, pmy_mesh_);
  } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_hybrid) {
    SetupRnsSt<tov::TabulatedEOS>(pin, pmy_mesh_);
  } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_piecewise_poly) {
    SetupRnsSt<tov::PiecewisePolytropeEOS>(pin, pmy_mesh_);
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "Unknown EOS requested for RNS-ST scalarized-star problem"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;

  pmbp->pdyngr->PrimToConInit(0, (n1-1), 0, (n2-1), 0, (n3-1));

  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
            pmbp->pz4c->ADMConstraints<2>(pmbp);
            break;
    case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
            pmbp->pz4c->ADMConstraints<3>(pmbp);
            break;
    case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
            pmbp->pz4c->ADMConstraints<4>(pmbp);
            break;
  }

  return;
}

//----------------------------------------------------------------------------------------
// History function: rho-max, alpha-min (same diagnostics as dyngr_tov.cpp's
// TOVHistory), plus sphi-max/sphi-min (true signed extrema, a proxy for the central
// scalar value, same max/min-at-center reasoning TOVHistory already relies on for a
// single NS).

void RnsStHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 4;
  pdata->label[0] = "rho-max";
  pdata->label[1] = "alpha-min";
  pdata->label[2] = "sphi-max";
  pdata->label[3] = "sphi-min";

  auto &w0_ = pm->pmb_pack->pmhd->w0;
  auto &adm = pm->pmb_pack->padm->adm;
  auto &sf_ = pm->pmb_pack->pscalarfield->u0;
  int i_sphi = scalarfield::ScalarField::I_SF_SPHI;

  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is; int nx1 = indcs.nx1;
  int js = indcs.js; int nx2 = indcs.nx2;
  int ks = indcs.ks; int nx3 = indcs.nx3;
  const int nmkji = (pm->pmb_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  Real rho_max = std::numeric_limits<Real>::max();
  Real alpha_min = -rho_max;
  Real sphi_max = -rho_max;
  Real sphi_min = rho_max;
  Kokkos::parallel_reduce("RnsStHistSums", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &mb_max, Real &mb_alp_min, Real &mb_sphi_max,
                Real &mb_sphi_min) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    mb_max = fmax(mb_max, w0_(m, IDN, k, j, i));
    mb_alp_min = fmin(mb_alp_min, adm.alpha(m, k, j, i));
    mb_sphi_max = fmax(mb_sphi_max, sf_(m, i_sphi, k, j, i));
    mb_sphi_min = fmin(mb_sphi_min, sf_(m, i_sphi, k, j, i));
  }, Kokkos::Max<Real>(rho_max), Kokkos::Min<Real>(alpha_min), Kokkos::Max<Real>(sphi_max),
     Kokkos::Min<Real>(sphi_min));

#if MPI_PARALLEL_ENABLED
  if (global_variable::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &rho_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &alpha_min, 1, MPI_ATHENA_REAL, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &sphi_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &sphi_min, 1, MPI_ATHENA_REAL, MPI_MIN, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&rho_max, &rho_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&alpha_min, &alpha_min, 1, MPI_ATHENA_REAL, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sphi_max, &sphi_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sphi_min, &sphi_min, 1, MPI_ATHENA_REAL, MPI_MIN, 0, MPI_COMM_WORLD);
    rho_max = 0.;
    alpha_min = 0.;
    sphi_max = 0.;
    sphi_min = 0.;
  }
#endif

  pdata->hdata[0] = rho_max;
  pdata->hdata[1] = alpha_min;
  pdata->hdata[2] = sphi_max;
  pdata->hdata[3] = sphi_min;
}
