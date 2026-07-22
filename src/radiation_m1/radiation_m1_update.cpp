//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_update.cpp
//! \brief beam time update for grey M1

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/eos.hpp"
#include "globals.hpp"
#include "radiation_m1.hpp"
#include "radiation_m1_calc_closure.hpp"
#include "radiation_m1_compton_implicit.hpp"
#include "radiation_m1_helpers.hpp"
#include "radiation_m1_nurates.hpp"
#include "radiation_m1_sources.hpp"
#include "radiation_m1_closure.hpp"
#include "z4c/z4c.hpp"

namespace radiationm1 {

TaskStatus RadiationM1::TimeUpdate(Driver *pdrive, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;

  // Here we are using dynamic_cast to infer which derived type pdyngr is
  auto *ptest_nqt =
      dynamic_cast<dyngr::DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                                     Primitive::ResetFloor> *>(
          pmy_pack->pdyngr);
  if (ptest_nqt != nullptr) {
    switch (indcs.ng) {
      case 2:
        return TimeUpdate_<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                           Primitive::ResetFloor, 2>(pdrive, stage);
        break;
      case 3:
        return TimeUpdate_<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                           Primitive::ResetFloor, 3>(pdrive, stage);
        break;
      case 4:
        return TimeUpdate_<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                           Primitive::ResetFloor, 4>(pdrive, stage);
        break;
    }
  }

  auto *ptest_nlog = dynamic_cast<dyngr::DynGRMHDPS<
      Primitive::EOSCompOSE<Primitive::NormalLogs>, Primitive::ResetFloor> *>(
      pmy_pack->pdyngr);
  if (ptest_nlog != nullptr) {
    switch (indcs.ng) {
      case 2:
        return TimeUpdate_<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                           Primitive::ResetFloor, 2>(pdrive, stage);
        break;
      case 3:
        return TimeUpdate_<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                           Primitive::ResetFloor, 3>(pdrive, stage);
        break;
      case 4:
        return TimeUpdate_<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                           Primitive::ResetFloor, 4>(pdrive, stage);
        break;
    }
  }

  if (!ismhd && !ishydro) {
    switch (indcs.ng) {
      case 2:
        return TimeUpdate_<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                           Primitive::ResetFloor, 2>(pdrive, stage);
        break;
      case 3:
        return TimeUpdate_<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                           Primitive::ResetFloor, 3>(pdrive, stage);
        break;
      case 4:
        return TimeUpdate_<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                           Primitive::ResetFloor, 4>(pdrive, stage);
        break;
    }
  }

  // Fallthrough for ideal gas or any non-EOSCompOSE EOS.
  // The EOS cast inside TimeUpdate_ is guarded by (nspecies > 1), so for
  // photons (nspecies == 1) the cast is never reached and any EOSPolicy works.
  if (ismhd || ishydro) {
    switch (indcs.ng) {
      case 2:
        return TimeUpdate_<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                           Primitive::ResetFloor, 2>(pdrive, stage);
        break;
      case 3:
        return TimeUpdate_<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                           Primitive::ResetFloor, 3>(pdrive, stage);
        break;
      case 4:
        return TimeUpdate_<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                           Primitive::ResetFloor, 4>(pdrive, stage);
        break;
    }
  }

  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl;
  std::cout << "Unsupported EOS type!\n";
  abort();
}

//! \file radiation_m1_update.cpp
//! \brief perform update for M1 Steps
//!  1. F^m   = F^k + dt/2 [ A[F^k] + S[F^m]   ]
//!  2. F^k+1 = F^k + dt   [ A[F^m] + S[F^k+1] ]
//!  At each step we solve an implicit problem in the form
//!     F = F^* + cdt S[F]
//!  Where F^* = F^k + cdt A
template <class EOSPolicy, class ErrorPolicy, int NGHOST>
TaskStatus RadiationM1::TimeUpdate_(Driver *d, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  auto &u0_ = u0;
  auto &chi_ = chi;
  auto &u1_ = u1;
  auto flx1 = uflx.x1f;
  auto flx2 = uflx.x2f;
  auto flx3 = uflx.x3f;

  auto &eta_0_ = eta_0;
  auto &abs_0_ = abs_0;
  auto &eta_1_ = eta_1;
  auto &abs_1_ = abs_1;
  auto &scat_1_ = scat_1;

  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &nvars_ = nvars;
  auto &nspecies_ = nspecies;

  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  auto &params_ = pmy_pack->pradm1->params;

  DvceArray5D<Real> w0_ = w0;
  DvceArray5D<Real> umhd0_;
  if (ismhd) {
    w0_ = pmy_pack->pmhd->w0;
    umhd0_ = pmy_pack->pmhd->u0;
  }

  auto &photon_op_params_ = pmy_pack->pradm1->photon_op_params;
  Real gm1_{};
  if (ismhd && params_.opacity_type == Photons &&
      photon_op_params_.is_matter_implicit) {
    gm1_ = pmy_pack->pmhd->peos->eos_data.gamma - 1.0;
  }

  auto &BrentFunc_ = pmy_pack->pradm1->BrentFunc;
  auto &HybridsjFunc_ = pmy_pack->pradm1->HybridsjFunc;

  // mb is only used for lepton tracking (nspecies > 1).
  // For photons (nspecies == 1) the EOS cast is skipped entirely.
  Real mb{};
  if ((ismhd || ishydro) && nspecies > 1) {
    Primitive::EOS<EOSPolicy, ErrorPolicy> &eos =
        static_cast<dyngr::DynGRMHDPS<EOSPolicy, ErrorPolicy> *>(
            pmy_pack->pdyngr)
            ->eos.ps.GetEOSMutable();
    mb = eos.GetBaryonMass();
  }

  Real beta[2] = {0.5, 1.};
  Real beta_dt = (beta[stage - 1]) * (pmy_pack->pmesh->dt);

  adm::ADM::ADM_vars &adm = pmy_pack->padm->adm;

  bool ismhd_ = ismhd;
  bool ishydro_ = ishydro;

  par_for(
      "radiation_m1_update", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        // [A] Compute gr quantities: metric, shift, extrinsic curvature, etc.
        Real garr_dd[16];
        Real garr_uu[16];
        AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_dd{};
        AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_uu{};
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> gamma_uu{};
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> K_dd{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> gamma_ud{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> beta_u{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> beta_d{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> n_d{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> n_u{};
        adm::SpacetimeMetric(
            adm.alpha(m, k, j, i), adm.beta_u(m, 0, k, j, i),
            adm.beta_u(m, 1, k, j, i), adm.beta_u(m, 2, k, j, i),
            adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
            adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
            adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i), garr_dd);
        adm::SpacetimeUpperMetric(
            adm.alpha(m, k, j, i), adm.beta_u(m, 0, k, j, i),
            adm.beta_u(m, 1, k, j, i), adm.beta_u(m, 2, k, j, i),
            adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
            adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
            adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i), garr_uu);
        for (int a = 0; a < 4; ++a) {
          for (int b = 0; b < 4; ++b) {
            g_dd(a, b) = garr_dd[a + b * 4];
            g_uu(a, b) = garr_uu[a + b * 4];
          }
        }
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            gamma_uu(a, b) =
                g_uu(a + 1, b + 1) +
                adm.beta_u(m, a, k, j, i) * adm.beta_u(m, b, k, j, i) /
                    (adm.alpha(m, k, j, i) * adm.alpha(m, k, j, i));
            K_dd(a, b) = adm.vK_dd(m, a, b, k, j, i);
          }
        }
        pack_n_d(adm.alpha(m, k, j, i), n_d);
        tensor_contract(g_uu, n_d, n_u);
        for (int a = 0; a < 4; ++a) {
          for (int b = 0; b < 4; ++b) {
            gamma_ud(a, b) = (a == b) + n_u(a) * n_d(b);
          }
        }

        pack_beta_u(adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
                    adm.beta_u(m, 2, k, j, i), beta_u);
        tensor_contract(g_dd, beta_u, beta_d);

        Real gam = adm::SpatialDet(
            adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
            adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
            adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i));
        Real volform = Kokkos::sqrt(gam);

        // [B] Compute derivatives of gr quantities (note: 3-arrays, no \p_t!)
        // [B.1] Derivatives of lapse (\p_i alpha)
        Real ideltax[3] = {1 / mbsize.d_view(m).dx1, 1 / mbsize.d_view(m).dx2,
                           1 / mbsize.d_view(m).dx3};
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dalpha_d{};
        dalpha_d(0) = Dx<NGHOST>(0, ideltax, adm.alpha, m, k, j, i);
        dalpha_d(1) =
            (multi_d) ? Dx<NGHOST>(1, ideltax, adm.alpha, m, k, j, i) : 0.;
        dalpha_d(2) =
            (three_d) ? Dx<NGHOST>(2, ideltax, adm.alpha, m, k, j, i) : 0.;

        // [B.2] Derivatives of shift (\p_i beta_u(j))
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> dbeta_du{};
        for (int a = 0; a < 3; ++a) {
          dbeta_du(0, a) = Dx<NGHOST>(0, ideltax, adm.beta_u, m, a, k, j, i);
          dbeta_du(1, a) =
              (multi_d) ? Dx<NGHOST>(1, ideltax, adm.beta_u, m, a, k, j, i)
                        : 0.;
          dbeta_du(2, a) =
              (three_d) ? Dx<NGHOST>(2, ideltax, adm.beta_u, m, a, k, j, i)
                        : 0.;
        }

        // [B.3] Derivatives of spatial metric (\p_k gamma_ij)
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> dg_ddd{};
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            dg_ddd(0, a, b) =
                Dx<NGHOST>(0, ideltax, adm.g_dd, m, a, b, k, j, i);
            dg_ddd(1, a, b) =
                (multi_d) ? Dx<NGHOST>(1, ideltax, adm.g_dd, m, a, b, k, j, i)
                          : 0.;
            dg_ddd(2, a, b) =
                (three_d) ? Dx<NGHOST>(2, ideltax, adm.g_dd, m, a, b, k, j, i)
                          : 0.;
          }
        }

        // [C] Compute fluid quantities
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> u_u{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> u_d{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> v_u{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> v_d{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> proj_ud{};

        Real w_lorentz{};
        w_lorentz = get_w_lorentz(w0_(m, IVX, k, j, i), w0_(m, IVY, k, j, i),
                                  w0_(m, IVZ, k, j, i), g_dd);
        pack_u_u(w_lorentz / adm.alpha(m, k, j, i),
                 w0_(m, IVX, k, j, i) -
                     w_lorentz * beta_u(1) / adm.alpha(m, k, j, i),
                 w0_(m, IVY, k, j, i) -
                     w_lorentz * beta_u(2) / adm.alpha(m, k, j, i),
                 w0_(m, IVZ, k, j, i) -
                     w_lorentz * beta_u(3) / adm.alpha(m, k, j, i),
                 u_u);
        pack_v_u(u_u(0), u_u(1), u_u(2), u_u(3), adm.alpha(m, k, j, i),
                 adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
                 adm.beta_u(m, 2, k, j, i), v_u);
        tensor_contract(g_dd, u_u, u_d);
        tensor_contract(g_dd, v_u, v_d);
        calc_proj(u_d, u_u, proj_ud);

        // [E] Compute contribution from flux and geometric sources
        Real rEFN[M1_TOTAL_NUM_SPECIES][5];
        Real DDxp[M1_TOTAL_NUM_SPECIES];
        for (int nuidx = 0; nuidx < nspecies_; nuidx++) {
          // [E.1: Contribution from fluxes]
          for (int var = 0; var < nvars_; ++var) {
            rEFN[nuidx][var] =
                -(flx1(m, CombinedIdx(nuidx, var, nvars_), k, j, i + 1) -
                  flx1(m, CombinedIdx(nuidx, var, nvars_), k, j, i)) *
                ideltax[0];
            if (multi_d) {
              rEFN[nuidx][var] +=
                  -(flx2(m, CombinedIdx(nuidx, var, nvars_), k, j + 1, i) -
                    flx2(m, CombinedIdx(nuidx, var, nvars_), k, j, i)) *
                  ideltax[1];
            }
            if (three_d) {
              rEFN[nuidx][var] +=
                  -(flx3(m, CombinedIdx(nuidx, var, nvars_), k + 1, j, i) -
                    flx3(m, CombinedIdx(nuidx, var, nvars_), k, j, i)) *
                  ideltax[2];
            }
          }
          // [E.2 Contribution from geometric sources]
          // Load lab radiation quantities
          if (params_.gr_sources) {
            AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
            const Real E =
                u0_(m, CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, i);
            pack_F_d(beta_u(1), beta_u(2), beta_u(3),
                     u0_(m, CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, i),
                     u0_(m, CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, i),
                     u0_(m, CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, i),
                     F_d);
            AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> F3_d{};
            F3_d(0) = F_d(1);
            F3_d(1) = F_d(2);
            F3_d(2) = F_d(3);

            AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> P_dd{};
            apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d,
                          chi_(m, nuidx, k, j, i), P_dd, params_);
            AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> P3_dd{};
            for (int a = 0; a < 3; ++a) {
              for (int b = 0; b < 3; ++b) {
                P3_dd(a, b) = P_dd(a + 1, b + 1);
              }
            }
            AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> P3_uu{};
            tensor_contract2(gamma_uu, P3_dd, P3_uu);

            // geometric sources
            rEFN[nuidx][M1_E_IDX] +=
                adm.alpha(m, k, j, i) * tensor_dot(P3_uu, K_dd) -
                tensor_dot(gamma_uu, F3_d, dalpha_d);

            for (int a = 0; a < 3; ++a) {
              rEFN[nuidx][a + 1] -= E * dalpha_d(a);
              for (int b = 0; b < 3; ++b) {
                rEFN[nuidx][a + 1] += F3_d(b) * dbeta_du(a, b);
              }
              for (int b = 0; b < 3; ++b) {
                for (int c = 0; c < 3; ++c) {
                  rEFN[nuidx][a + 1] += adm.alpha(m, k, j, i) / 2. *
                                        P3_uu(b, c) * dg_ddd(a, b, c);
                }
              }
            }
          }
        }

        // [F] Compute contribution from matter sources
        Real DrEFN[M1_TOTAL_NUM_SPECIES][5]{};
        Real theta{};
        if (params_.matter_sources) {
          for (int nuidx = 0; nuidx < nspecies_; nuidx++) {
            if (params_.src_update == Explicit) {
              // radiation fields
              AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
              AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> S_d{};
              AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> tS_d{};
              pack_F_d(beta_u(1), beta_u(2), beta_u(3),
                       u0_(m, CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, i),
                       u0_(m, CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, i),
                       u0_(m, CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, i),
                       F_d);
              const Real E =
                  u0_(m, CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, i);
              AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> P_dd{};
              apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E,
                            F_d, chi_(m, nuidx, k, j, i), P_dd, params_);

              // compute radiation quantities in fluid frame
              AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> T_dd{};
              assemble_rT(n_d, E, F_d, P_dd, T_dd);
              AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> H_d{};
              AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> H_u{};
              Real J = calc_J_from_rT(T_dd, u_u);
              calc_H_from_rT(T_dd, u_u, proj_ud, H_d);
              apply_floor(g_uu, J, H_d, params_);

              tensor_contract(g_uu, H_d, H_u);
              const Real Gamma =
                  compute_Gamma(w_lorentz, v_u, J, E, F_d, params_);

              // Compute radiation sources
              calc_rad_sources(eta_1_(m, nuidx, k, j, i) * volform,
                               abs_1_(m, nuidx, k, j, i),
                               scat_1_(m, nuidx, k, j, i), u_d, J, H_d, S_d);
              DrEFN[nuidx][M1_E_IDX] =
                  beta_dt * calc_rE_source(adm.alpha(m, k, j, i), n_u, S_d);

              calc_rF_source(adm.alpha(m, k, j, i), gamma_ud, S_d, tS_d);
              DrEFN[nuidx][M1_FX_IDX] = beta_dt * tS_d(1);
              DrEFN[nuidx][M1_FY_IDX] = beta_dt * tS_d(2);
              DrEFN[nuidx][M1_FZ_IDX] = beta_dt * tS_d(3);

              if (nspecies_ > 1) {
                const Real N =
                    u0_(m, CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, i);
                DrEFN[nuidx][M1_N_IDX] =
                    beta_dt * adm.alpha(m, k, j, i) *
                    (volform * eta_0_(m, nuidx, k, j, i) -
                     abs_0_(m, nuidx, k, j, i) * N / Gamma);
              }
            }

            if (params_.src_update == Implicit) {
              // Boost to the fluid frame, compute fluid matter interaction and
              // boost back. Use these values for implicit solve

              // advect radiation
              Real Estar =
                  u1_(m, CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, i) +
                  beta_dt * rEFN[nuidx][M1_E_IDX];
              AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> Fstar_d{};
              pack_F_d(beta_u(1), beta_u(2), beta_u(3),
                       u1_(m, CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, i) +
                           beta_dt * rEFN[nuidx][M1_FX_IDX],
                       u1_(m, CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, i) +
                           beta_dt * rEFN[nuidx][M1_FY_IDX],
                       u1_(m, CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, i) +
                           beta_dt * rEFN[nuidx][M1_FZ_IDX],
                       Fstar_d);
              apply_floor(g_uu, Estar, Fstar_d, params_);
              Real Nstar{};
              if (nspecies_ > 1) {
                Nstar = Kokkos::max<Real>(
                    u1_(m, CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, i) +
                        beta_dt * rEFN[nuidx][M1_N_IDX],
                    params_.rad_N_floor);
              }

              // Compute quantities in the fluid frame
              AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> P_dd{};
              calc_closure(BrentFunc_, g_dd, g_uu, n_d, w_lorentz, u_u, v_d,
                           proj_ud, Estar, Fstar_d, chi_(m, nuidx, k, j, i),
                           P_dd, params_, params_.closure_type);
              AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> rT_dd{};
              assemble_rT(n_d, Estar, Fstar_d, P_dd, rT_dd);
              Real Jstar = calc_J_from_rT(rT_dd, u_u);
              AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> Hstar_d{};
              calc_H_from_rT(rT_dd, u_u, proj_ud, Hstar_d);

              // Estimate interaction with matter
              const Real dtau = beta_dt * (adm.alpha(m, k, j, i) / w_lorentz);

              // [matter-implicit] Replace the frozen-opacity eta_1_/abs_1_
              // for this cell with a closed-form Planck+Compton quartic
              // pre-solve, self-consistent in T_gas/J at this cell's
              // pre-coupling ("star") state -- see
              // radiation_m1_compton_implicit.hpp and DEVELOPMENT.md's
              // Stage 6 section. Assumes Primitive::IdealGas (safe:
              // EOSCompOSE already hard-errors at opacity-parse time,
              // before this can run). Falls back to the existing
              // frozen-opacity arrays on failure or when the feature is
              // off -- zero behavior change for every currently-validated
              // test.
              //
              // The quartic's own (T_new, J_new) pair is energy-conserving
              // by construction (Etot_star = J_new + (rho/gm1)*T_new, with
              // T_new>=0 enforced), so J_new can never imply more energy
              // transfer than the gas actually holds. J_new is captured
              // here and used *after* source_update() below to correct
              // Enew directly -- not just to refine the opacity target fed
              // into that (gas-energy-blind) solve, which is only
              // guaranteed to reproduce J_new in the flat/static/no-flux
              // limit (see DEVELOPMENT.md Stage 6).
              Real eta_local = eta_1_(m, nuidx, k, j, i);
              Real abs_local = abs_1_(m, nuidx, k, j, i);
              Real J_new{};
              bool matter_implicit_ok = false;
              if (params_.opacity_type == Photons &&
                  photon_op_params_.is_matter_implicit) {
                const Real rho_gas = w0_(m, IDN, k, j, i);
                const Real T_star = w0_(m, IEN, k, j, i) / rho_gas;
                // Gate kappa_s by is_compton, matching the frozen-opacity
                // kernel (radiation_m1_calc_opacities_photons.cpp:94,201):
                // kappa_s alone doesn't disable the Compton channel --
                // compton=false must zero it out even if kappa_s is left
                // nonzero in the athinput.
                const Real kappa_s_gated =
                    photon_op_params_.is_compton ? photon_op_params_.kappa_s : 0.0;
                Real T_new{};
                if (SolveComptonQuartic(
                        rho_gas, T_star, Jstar, gm1_,
                        kappa_s_gated, photon_op_params_.kappa_p,
                        photon_op_params_.arad,
                        photon_op_params_.inv_t_electron, dtau, volform,
                        T_new, J_new)) {
                  matter_implicit_ok = true;
                  // Rate stays frozen at T_star (matching the quartic's own
                  // linearization); only the emission target moves to T_new.
                  const Real sigma_c_star = kappa_s_gated * rho_gas * 4.0 *
                                            T_star *
                                            photon_op_params_.inv_t_electron;
                  abs_local = sigma_c_star + rho_gas * photon_op_params_.kappa_p;
                  eta_local = abs_local * photon_op_params_.arad *
                              SQR(SQR(T_new));
                }
              }

              Real Jnew = (Jstar + dtau * eta_local * volform) /
                          (1. + dtau * abs_local);

              // Only three components of H^a are independent H^0 is found by
              // requiring H^a u_a = 0
              const Real khat = (abs_local + scat_1_(m, nuidx, k, j, i));
              AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> Hnew_d{};
              for (int a = 1; a < 4; ++a) {
                Hnew_d(a) = Hstar_d(a) / (1. + dtau * khat);
              }
              Hnew_d(0) = 0.0;
              for (int a = 1; a < 4; ++a) {
                Hnew_d(0) -= Hnew_d(a) * (u_u(a) / u_u(0));
              }

              // Update Tmunu
              const Real H2 = tensor_dot(g_uu, Hnew_d, Hnew_d);
              const Real xi = Kokkos::sqrt(H2)*(Jnew > params_.rad_E_floor ? 1./Jnew : 0.);
              chi_(m, nuidx, k, j, i) = closure_fun(xi, params_.closure_type);

              const Real dthick = 3. * (1. - chi_(m, nuidx, k, j, i)) / 2.;
              const Real dthin = 1. - dthick;

              for (int a = 0; a < 4; ++a) {
                for (int b = a; b < 4; ++b) {
                  rT_dd(a, b) =
                      Jnew * u_d(a) * u_d(b) + Hnew_d(a) * u_d(b) +
                      Hnew_d(b) * u_d(a) +
                      dthin * Jnew *
                          (Hnew_d(a) * Hnew_d(b) * (H2 > 0 ? 1 / H2 : 0)) +
                      dthick * Jnew * (g_dd(a, b) + u_d(a) * u_d(b)) / 3.;
                }
              }

              // Boost back to the lab frame
              AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> Fnew_d{};
              Real Enew = calc_J_from_rT(rT_dd, n_u);
              calc_H_from_rT(rT_dd, n_u, gamma_ud, Fnew_d);
              apply_floor(g_uu, Enew, Fnew_d, params_);

              auto src_signal = source_update(
                  BrentFunc_, HybridsjFunc_, beta_dt, adm.alpha(m, k, j, i),
                  g_dd, g_uu, n_d, n_u, gamma_ud, u_d, u_u, v_d, v_u, proj_ud,
                  w_lorentz, Estar, Fstar_d, Estar, Fstar_d,
                  volform * eta_local,
                  abs_local, scat_1_(m, nuidx, k, j, i),
                  chi_(m, nuidx, k, j, i), Enew, Fnew_d, params_,
                  params_.closure_type);
              apply_floor(g_uu, Enew, Fnew_d, params_);

              // Update closure
              apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, Enew,
                            Fnew_d, chi_(m, nuidx, k, j, i), P_dd, params_);

              // Compute new radiation energy density in the fluid frame
              AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> T_dd{};
              assemble_rT(n_d, Enew, Fnew_d, P_dd, T_dd);
              Jnew = calc_J_from_rT(T_dd, u_u);

              // [matter-implicit] Make Enew authoritative from the quartic's
              // energy-conserving J_new (see comment above the quartic
              // call), instead of whatever the gas-energy-blind Newton/
              // explicit solve above produced. calc_J_from_rT(assemble_rT(
              // n_d,E,F_d,P_dd),u_u) is exactly linear in E alone (holding
              // F_d, P_dd, n_d, u_u fixed), with dJ/dE = w_lorentz^2 (from
              // n_d.u_u = -w_lorentz) -- so nudging E to move J from Jnew to
              // J_new is this one closed-form correction, not a full
              // reassembly. Fnew_d (momentum/flux) is left exactly as
              // solved -- the quartic has no flux term to correct against.
              // Re-floor afterward: the last apply_floor before this point
              // is at the source_update() call above, so the nudged Enew
              // has not yet been floor/causality-checked.
              if (matter_implicit_ok) {
                Enew += (J_new - Jnew) / SQR(w_lorentz);
                apply_floor(g_uu, Enew, Fnew_d, params_);
                Jnew = J_new;
              }

              // Compute changes in radiation energy and momentum
              DrEFN[nuidx][M1_E_IDX] = Enew - Estar;
              DrEFN[nuidx][M1_FX_IDX] = Fnew_d(1) - Fstar_d(1);
              DrEFN[nuidx][M1_FY_IDX] = Fnew_d(2) - Fstar_d(2);
              DrEFN[nuidx][M1_FZ_IDX] = Fnew_d(3) - Fstar_d(3);

              if (nspecies_ > 1) {
                // Compute updated Gamma
                const Real Gamma =
                    compute_Gamma(w_lorentz, v_u, Jnew, Enew, Fnew_d, params_);

                if (params_.source_therm_limit < 0 ||
                    beta_dt * abs_0_(m, nuidx, k, j, i) <
                        params_.source_therm_limit) {
                  //
                  // N^k+1 = N^* + dt ( eta - abs N^k+1 )
                  DrEFN[nuidx][M1_N_IDX] =
                      (Nstar + beta_dt * adm.alpha(m, k, j, i) * volform *
                                   eta_0_(m, nuidx, k, j, i)) /
                          (1 + beta_dt * adm.alpha(m, k, j, i) *
                                   abs_0_(m, nuidx, k, j, i) / Gamma) -
                      Nstar;
                } else {
                  // The neutrino number density is updated assuming the
                  // neutrino average energies are those of the equilibrium
                  const Real E =
                      u0_(m, CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, i);
                  const Real N =
                      u0_(m, CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, i);
                  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
                  pack_F_d(
                      beta_u(1), beta_u(2), beta_u(3),
                      u0_(m, CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, i),
                      u0_(m, CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, i),
                      u0_(m, CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, i),
                      F_d);
                  const Real nueave =
                      w_lorentz * (E - tensor_dot(F_d, v_u)) / N;
                  DrEFN[nuidx][M1_N_IDX] =
                      (nueave > 0 ? Gamma * Jnew / nueave - Nstar : 0.0);
                }
              }
            }
            // fluid lepton sources
            if (nspecies_ > 1) {
              DDxp[nuidx] = -mb * (DrEFN[nuidx][M1_N_IDX] * (nuidx == 0) -
                                   DrEFN[nuidx][M1_N_IDX] * (nuidx == 1));
            }
          }

          // [G] Limit sources
          theta = 1.0;
          if (params_.theta_limiter && params_.source_limiter >= 0) {
            const Real tau = (ismhd_ || ishydro_) ? umhd0_(m, IEN, k, j, i) : 0.;
            const Real dens = (ismhd_ || ishydro_) ? w0_(m, IDN, k, j, i) : 0.;
            // IYF=5 only exists when nscalars>=1 (e.g. EOSCompOSE with Ye).
            // For photons (nspecies==1) Y_e is never used, so skip the read.
            const Real Y_e = ((ismhd_ || ishydro_) && nspecies_ > 1)
                                 ? w0_(m, IYF, k, j, i) : 0.;

            theta = 1.0;
            Real DTau_sum = 0.0;
            for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
              Real Estar =
                  u1_(m, CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, i) +
                  beta_dt * rEFN[nuidx][M1_E_IDX];
              if (DrEFN[nuidx][M1_E_IDX] < 0) {
                theta = Kokkos::min<Real>(-params_.source_limiter *
                                              Kokkos::max<Real>(Estar, 0.0) /
                                              DrEFN[nuidx][M1_E_IDX],
                                          theta);
              }
              DTau_sum -= DrEFN[nuidx][M1_E_IDX];
            }
            if (DTau_sum < 0) {
              theta = Kokkos::min(-params_.source_limiter *
                                      Kokkos::max<Real>(tau, 0.0) / DTau_sum,
                                  theta);
            }

            if (nspecies_ > 1) {
              Real DDxp_sum = 0.0;
              for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
                Real Nstar =
                    u1_(m, CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, i) +
                    beta_dt * rEFN[nuidx][M1_N_IDX];
                if (DrEFN[nuidx][M1_N_IDX] < 0) {
                  theta = Kokkos::min(-params_.source_limiter *
                                          Kokkos::max<Real>(Nstar, 0.0) /
                                          DrEFN[nuidx][M1_N_IDX],
                                      theta);
                }
                DDxp_sum += DDxp[nuidx];
              }
              const Real DYe = DDxp_sum / dens;
              if (DYe > 0) {
                theta = Kokkos::min<Real>(
                    params_.source_limiter *
                        Kokkos::max<Real>(params_.source_Ye_max - Y_e, 0.0) /
                        DYe,
                    theta);
              } else if (DYe < 0) {
                theta = Kokkos::min<Real>(
                    params_.source_limiter *
                        Kokkos::min(params_.source_Ye_min - Y_e, 0.0) / DYe,
                    theta);
              }
            }
            theta = Kokkos::max<Real>(0.0, theta);
          }  //  if (params_.theta_limiter && params_.source_limiter >= 0)
        } else {
          theta = 0.0;
        }  // if (params_.matter_sources)

        // [H] Update fields
        for (int nuidx = 0; nuidx < nspecies_; nuidx++) {
          Real Ef = u1_(m, CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, i) +
                    beta_dt * rEFN[nuidx][M1_E_IDX] +
                    theta * DrEFN[nuidx][M1_E_IDX];
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> Ff_d{};
          Real Fxf = u1_(m, CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, i) +
                     beta_dt * rEFN[nuidx][M1_FX_IDX] +
                     theta * DrEFN[nuidx][M1_FX_IDX];
          Real Fyf = u1_(m, CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, i) +
                     beta_dt * rEFN[nuidx][M1_FY_IDX] +
                     theta * DrEFN[nuidx][M1_FY_IDX];
          Real Fzf = u1_(m, CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, i) +
                     beta_dt * rEFN[nuidx][M1_FZ_IDX] +
                     theta * DrEFN[nuidx][M1_FZ_IDX];
          pack_F_d(adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
                   adm.beta_u(m, 2, k, j, i), Fxf, Fyf, Fzf, Ff_d);
          apply_floor(g_uu, Ef, Ff_d, params_);
          u0_(m, CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, i) = Ef;
          u0_(m, CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, i) = Ff_d(1);
          u0_(m, CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, i) = Ff_d(2);
          u0_(m, CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, i) = Ff_d(3);

          if (nspecies_ > 1) {
            Real Nf = u1_(m, CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, i) +
                      beta_dt * rEFN[nuidx][M1_N_IDX] +
                      theta * DrEFN[nuidx][M1_N_IDX];
            Nf = Kokkos::max<Real>(Nf, params_.rad_N_floor);
            u0_(m, CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, i) = Nf;
          }

          if (params_.backreact && stage == 2 && (ismhd_)) {
            umhd0_(m, IEN, k, j, i) -= theta * DrEFN[nuidx][M1_E_IDX];
            umhd0_(m, IM1, k, j, i) -= theta * DrEFN[nuidx][M1_FX_IDX];
            umhd0_(m, IM2, k, j, i) -= theta * DrEFN[nuidx][M1_FY_IDX];
            umhd0_(m, IM3, k, j, i) -= theta * DrEFN[nuidx][M1_FZ_IDX];
            if (nspecies_ > 1) {
              umhd0_(m, IYF, k, j, i) += theta * DDxp[nuidx];
            }
          }
        }
      });
  is_chi_updated = false;
  return TaskStatus::complete;
}
}  // namespace radiationm1
