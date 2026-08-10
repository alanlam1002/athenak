//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyngr_grass.cpp
//  \brief Problem generator for a (possibly differentially rotating) neutron star built
//  from GRASS (an RNS-family rotating-NS equilibrium code) restart-binary initial data.
//  Requires <z4c> (full dynamical Z4c). Unlike the sibling scalar-tensor dyngr_rns_st.cpp,
//  no <scalarfield> block is needed or used -- GRASS's restart carries a generic scalar
//  slot for a different (scalarized) solver mode, ignored here (see grass/grass_reader.hpp).
//  GRASS's own (pressure, energy) pair from its equilibrium solve is reused directly, so
//  (unlike dyngr_rns_st.cpp / dyngr_tov.cpp) no AthenaK EOS call constructs P(rho) for the
//  initial data -- only a small dedicated GRASS-EOS-table lookup (grass/grass_eos_table.hpp)
//  recovers rest-mass density from the restart's `energy` field.
//  If <mhd> nscalars > 0, the passive scalar Y[e] (composition) is additionally seeded
//  from a separate 1D <problem> table (tov::TabulatedEOS, the same reader dyngr_tov.cpp/
//  kadath_bns.cpp use) -- built to carry Yq(nb) along the SAME (nb,T,Yl) trajectory that
//  produced GRASS's own e(n0),p(n0), so the star starts on the trajectory it was built
//  on when evolved with a genuine 3D tabulated <mhd> dyn_eos=compose EOS. This is purely
//  a composition seed -- temperature is never read from this table; PrimToConInit
//  recovers T internally from (rho0,P,Yq) via whatever 3D EOS is active.
//  Compile with '-D PROBLEM=dyn_grmhd/dyngr_grass' to enroll as user-specific pgen.

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
#include "grass/grass_units.hpp"
#include "grass/grass_eos_table.hpp"
#include "grass/grass_reader.hpp"
#include "utils/tov/tov_tabulated.hpp"

// Prototype for user-defined history function
void GrassHistory(HistoryData *pdata, Mesh *pm);

//----------------------------------------------------------------------------------------
//! \fn void SetupGrass
//  \brief Loads a GRASS restart binary and fills ADM/Z4c/hydro initial data. Host-only:
//  the reader's interpolation (grass::GrassData) is plain-CPU code, so this mirrors the
//  host-fill-then-deep_copy pattern used by the other external-ID pgens (elliptica/
//  lorene/sgrid/kadath/rns_st), not dyngr_tov.cpp's device-side par_for (which works
//  directly from a closed-form/ODE solution instead). Non-templated (unlike the
//  TOV-family pgens' Setup<EOS>) -- see file header, no AthenaK EOS call is needed here.

void SetupGrass(ParameterInput *pin, Mesh *pmy_mesh_) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  std::string id_file = pin->GetString("problem", "id_file");
  std::string eos_table_file = pin->GetString("problem", "grass_eos_table");

  grass::GrassUnits units;
  grass::GrassEosTable eos_table(eos_table_file, units);
  grass::GrassData data(id_file, units);
  // 1D DD2_hot_slice table: Y[e]=Yl(nb) composition seed only (see file header) --
  // <problem> table = ... , same input key tov::TabulatedEOS's other callers
  // (dyngr_tov.cpp, kadath_bns.cpp) already use.
  tov::TabulatedEOS slice_eos(pin);
  const bool read_ye = pin->GetOrAddInteger("mhd", "nscalars", 0) > 0;

  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;

  auto &u_adm = pmbp->padm->u_adm;
  auto &w0 = pmbp->pmhd->w0;

  // Host-side fill, then move to device -- see file header.
  HostArray5D<Real>::HostMirror host_u_adm = Kokkos::create_mirror_view(u_adm);
  HostArray5D<Real>::HostMirror host_w0 = Kokkos::create_mirror_view(w0);
  HostArray5D<Real>::HostMirror host_u_z4c = Kokkos::create_mirror_view(pmbp->pz4c->u0);

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

          grass::GrassData::Point pt;
          data.Interpolate(x1v, x2v, x3v, eos_table, slice_eos, &pt);

          host_adm.alpha(m, k, j, i) = pt.alpha;
          host_adm.beta_u(m, 0, k, j, i) = pt.beta_u[0];
          host_adm.beta_u(m, 1, k, j, i) = pt.beta_u[1];
          host_adm.beta_u(m, 2, k, j, i) = pt.beta_u[2];

          host_adm.g_dd(m, 0, 0, k, j, i) = pt.g_dd[0];
          host_adm.g_dd(m, 0, 1, k, j, i) = pt.g_dd[1];
          host_adm.g_dd(m, 0, 2, k, j, i) = pt.g_dd[2];
          host_adm.g_dd(m, 1, 1, k, j, i) = pt.g_dd[3];
          host_adm.g_dd(m, 1, 2, k, j, i) = pt.g_dd[4];
          host_adm.g_dd(m, 2, 2, k, j, i) = pt.g_dd[5];

          host_adm.vK_dd(m, 0, 0, k, j, i) = pt.K_dd[0];
          host_adm.vK_dd(m, 0, 1, k, j, i) = pt.K_dd[1];
          host_adm.vK_dd(m, 0, 2, k, j, i) = pt.K_dd[2];
          host_adm.vK_dd(m, 1, 1, k, j, i) = pt.K_dd[3];
          host_adm.vK_dd(m, 1, 2, k, j, i) = pt.K_dd[4];
          host_adm.vK_dd(m, 2, 2, k, j, i) = pt.K_dd[5];

          Real rho = (pt.rho0 > 0.0) ? pt.rho0 : 0.0;
          Real pres = (pt.rho0 > 0.0) ? pt.pres : 0.0;
          Real vu[3] = {0.0, 0.0, 0.0};
          if (pt.rho0 > 0.0) {
            vu[0] = pt.vu[0]; vu[1] = pt.vu[1]; vu[2] = pt.vu[2];
          }

          host_w0(m, IDN, k, j, i) = rho;
          host_w0(m, IPR, k, j, i) = pres;
          host_w0(m, IVX, k, j, i) = vu[0];
          host_w0(m, IVY, k, j, i) = vu[1];
          host_w0(m, IVZ, k, j, i) = vu[2];
          if (read_ye) {
            host_w0(m, IYF, k, j, i) = pt.Yq;
          }
        }
      }
    }
  }

  Kokkos::deep_copy(u_adm, host_u_adm);
  Kokkos::deep_copy(w0, host_w0);
  Kokkos::deep_copy(pmbp->pz4c->u0, host_u_z4c);

  // GRASS's ID carries no B-field data. AthenaK does not zero-initialize device
  // memory by default, so b0/bcc0 must be set explicitly or they're left as garbage,
  // which otherwise poisons the primitive solver with NaN from the very first step.
  Kokkos::deep_copy(pmbp->pmhd->b0.x1f, 0.0);
  Kokkos::deep_copy(pmbp->pmhd->b0.x2f, 0.0);
  Kokkos::deep_copy(pmbp->pmhd->b0.x3f, 0.0);
  Kokkos::deep_copy(pmbp->pmhd->bcc0, 0.0);
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//  \brief Sets initial conditions for a GRASS-built (possibly differentially rotating)
//  neutron star.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "GRASS-initial-data problem requires a <z4c> block (full dynamical "
              << "Z4c)" << std::endl;
    exit(EXIT_FAILURE);
  }

  user_hist_func = &GrassHistory;

  // No restart-path special-casing: this pgen only ever sets initial data once --
  // evolution afterward proceeds from checkpointed Z4c/hydro state like any other
  // dynamical run (same reasoning as dyngr_rns_st.cpp).
  if (restart) { return; }

  // Note: GRASS's own (pressure, energy) pair is reused directly -- no eos_policy
  // dispatch/template parameter is needed here, unlike the TOV-family pgens.
  SetupGrass(pin, pmy_mesh_);

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
// History function: rho-max, alpha-min (same diagnostics as dyngr_tov.cpp's TOVHistory
// / dyngr_rns_st.cpp's RnsStHistory, minus the scalar-field entries -- no scalar field
// here).

void GrassHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 2;
  pdata->label[0] = "rho-max";
  pdata->label[1] = "alpha-min";

  auto &w0_ = pm->pmb_pack->pmhd->w0;
  auto &adm = pm->pmb_pack->padm->adm;

  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is; int nx1 = indcs.nx1;
  int js = indcs.js; int nx2 = indcs.nx2;
  int ks = indcs.ks; int nx3 = indcs.nx3;
  const int nmkji = (pm->pmb_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  Real rho_max = std::numeric_limits<Real>::max();
  Real alpha_min = -rho_max;
  Kokkos::parallel_reduce("GrassHistSums", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &mb_max, Real &mb_alp_min) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    mb_max = fmax(mb_max, w0_(m, IDN, k, j, i));
    mb_alp_min = fmin(mb_alp_min, adm.alpha(m, k, j, i));
  }, Kokkos::Max<Real>(rho_max), Kokkos::Min<Real>(alpha_min));

#if MPI_PARALLEL_ENABLED
  if (global_variable::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &rho_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &alpha_min, 1, MPI_ATHENA_REAL, MPI_MIN, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&rho_max, &rho_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&alpha_min, &alpha_min, 1, MPI_ATHENA_REAL, MPI_MIN, 0, MPI_COMM_WORLD);
    rho_max = 0.;
    alpha_min = 0.;
  }
#endif

  pdata->hdata[0] = rho_max;
  pdata->hdata[1] = alpha_min;
}
