//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_m1_photon_singlezone.cpp
//  \brief Radiation M1 single zone LTE equilibration test for grey photon M1 + ideal-gas
//  hydro. Modeled on rad_m1_singlezone.cpp (the bns-nurates/EOSCompOSE analogue), but for
//  Primitive::IdealGas + opacity_type=photons: a uniform, static gas at fixed
//  density/temperature should drive the M1-evolved radiation energy density E toward the
//  LTE blackbody value a_rad*T^4.

// C++ headers

// Athena++ headers
#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "driver/driver.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/eos.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "radiation_m1/radiation_m1.hpp"
#include "radiation_m1/radiation_m1_helpers.hpp"

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::RadiationM1PhotonSingleZoneTest()
//  \brief Sets initial conditions for the grey photon M1 single zone LTE test
void ProblemGenerator::RadiationM1PhotonSingleZoneTest(ParameterInput *pin,
                                                        const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  auto *ptest_ideal =
      dynamic_cast<dyngr::DynGRMHDPS<Primitive::IdealGas, Primitive::ResetFloor> *>(
          pmbp->pdyngr);
  if (ptest_ideal == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "The photon single zone equilibration test problem generator "
                 "requires <mhd> dyn_eos = ideal"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->pradm1 == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "The photon single zone equilibration test problem generator "
                 "requires radiation-m1, but no "
              << "<radiation_m1> block in input file" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->pradm1->params.opacity_type != radiationm1::Photons) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "The photon single zone equilibration test problem generator "
                 "requires opacity_type = photons"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2 * ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * ng) : 1;
  int nmb1 = (pmbp->nmb_thispack - 1);
  auto &w0_ = pmbp->pmhd->w0;
  auto &uradm1_ = pmbp->pradm1->u0;
  auto &nspecies_ = pmbp->pradm1->nspecies;
  auto &m1_params_ = pmbp->pradm1->params;
  auto &m1_nvars_ = pmbp->pradm1->nvars;

  // get problem parameters
  Real rho = pin->GetReal("problem", "rho");
  Real temp = pin->GetReal("problem", "temp");
  Real vx = pin->GetReal("problem", "vx");
  Real vy = pin->GetReal("problem", "vy");
  Real vz = pin->GetReal("problem", "vz");

  Primitive::EOS<Primitive::IdealGas, Primitive::ResetFloor> &eos =
      static_cast<dyngr::DynGRMHDPS<Primitive::IdealGas, Primitive::ResetFloor> *>(
          pmbp->pdyngr)
          ->eos.ps.GetEOSMutable();
  Real mb = eos.GetBaryonMass();

  Real rho_code = rho;
  Real nb = rho / mb;
  Real w_lorentz = 1. / Kokkos::sqrt(1. - vx * vx - vy * vy - vz * vz);

  // initialize ADM variables (flat Minkowski)
  adm::ADM::ADM_vars &adm = pmbp->padm->adm;
  par_for(
      "pgen_metric_initialize", DevExeSpace(), 0, nmb1, 0, (n3 - 1), 0, (n2 - 1), 0,
      (n1 - 1), KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        for (int a = 0; a < 3; ++a)
          for (int b = a; b < 3; ++b) {
            adm.g_dd(m, a, b, k, j, i) = (a == b ? 1. : 0.);
          }

        adm.psi4(m, k, j, i) = 1.;
        adm.alpha(m, k, j, i) = 1.;
      });

  // set primitive variables and radiation ICs
  par_for(
      "pgen_photon_singlezone", DevExeSpace(), 0, nmb1, 0, (n3 - 1), 0, (n2 - 1), 0,
      (n1 - 1), KOKKOS_LAMBDA(int m, int k, int j, int i) {
        Real dummy_y{};

        w0_(m, IDN, k, j, i) = rho_code;
        w0_(m, IVX, k, j, i) = vx * w_lorentz;
        w0_(m, IVY, k, j, i) = vy * w_lorentz;
        w0_(m, IVZ, k, j, i) = vz * w_lorentz;
        w0_(m, IPR, k, j, i) = eos.GetPressure(nb, temp, &dummy_y);

        for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
          uradm1_(m, radiationm1::CombinedIdx(nuidx, M1_E_IDX, m1_nvars_), k, j, i) =
              m1_params_.rad_E_floor;
          if (nspecies_ > 1) {
            uradm1_(m, radiationm1::CombinedIdx(nuidx, M1_N_IDX, m1_nvars_), k, j, i) =
                m1_params_.rad_N_floor;
          }
        }
      });

  // Convert primitives to conserved vars
  pmbp->pdyngr->PrimToConInit(0, (n1 - 1), 0, (n2 - 1), 0, (n3 - 1));
}
