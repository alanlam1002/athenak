//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cfc.cpp
//! \brief implementation of the CFC class

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <string>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock_pack.hpp"
#include "parameter_input.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "utils/finite_diff.hpp"
#include "bvals/bvals.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "driver/driver.hpp"
#include "tasklist/numerical_relativity.hpp"
#include "cfc.hpp"
#include "geodesic-grid/spherical_grid.hpp"
#include "cfc_puncture.hpp"
#include "cfc_reconstruct.hpp"

namespace cfc {

namespace {

//----------------------------------------------------------------------------------------
//! \fn void BuildShiftSourceImpl<NGHOST>(...)
//! \brief Gmunu eq. 75's derivative term, added onto an already-built p_src
//! (16*pi*alpha*psi^-6*S-tilde_i -- see CFC::AssembleVectorSource). Sec 3.9's three
//! background-deviation cross-terms 2*Ahat0^ij*D_j[Delta(alpha*psi^-6)] +
//! 2*Ahats^ij*D_j(alpha0*psi0^-6) + 2*Ahats^ij*D_j[Delta(alpha*psi^-6)] collapse to two
//! (Ahat0^ij*X + Ahats^ij*X = Ahat^ij*X for the shared-gradient terms 1+3):
//!   2*Ahat^ij*D_j[Delta(alpha*psi^-6)]  +  2*Ahats^ij*D_j(alpha0*psi0^-6)
//! using the TOTAL Ahat^ij (a_dd) for the first term and the matter-only
//! Ahats^ij = a_dd-a0_dd for the second. Delta(alpha*psi^-6) = alpha*psi^-6-alpha0*psi0^-6
//! is built cancellation-free (Sec 3.7/3.9): alpha*psi^-6 = (alpha*psi)*psi^-7, and only
//! the psi^-7 factor needs the puncture regularization (alpha*psi = alpha0*psi0+v is a
//! plain bounded sum). Re-derived directly (not copied from the plan doc's RHS-sign
//! convention, which is tied to a different equation):
//!   Delta(alpha*psi^-6) = alpha0*psi0*(psi^-7-psi0^-7) + v*psi^-7
//!                       = v*psi^-7 - alpha0*psi0*reg7,   reg7 = psi0^-7-psi^-7
//! reusing the same x=delta_psi/psi0, P7(x)/(1+x)^7 Horner-polynomial factoring already
//! validated in mg_cfc_conformal_factor.cpp's ConformalFactorRHS (a fresh inline copy
//! here, not a shared helper -- see this stage's plan for why).

template <int NGHOST>
void BuildShiftSourceImpl(MeshBlockPack *pmbp, const DvceArray5D<Real> &delta_psi,
                          const DvceArray5D<Real> &delta_alpha_psi,
                          const DvceArray5D<Real> &u_psi0,
                          const DvceArray5D<Real> &u_alpha0_psi0,
                          const AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &a_dd,
                          const AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &a0_dd,
                          const AthenaTensor<Real, TensorSymm::NONE, 3, 1> &grad_ap6_0,
                          AthenaTensor<Real, TensorSymm::NONE, 3, 1> &p_src,
                          int mg_nghost, DvceArray5D<Real> &dap6) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  int ncells1 = indcs.nx1 + 2*indcs.ng;
  int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*indcs.ng) : 1;
  int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*indcs.ng) : 1;

  // dap6 is a persistent CFC member, sized over the full array extent (not just the
  // interior) so Dx<NGHOST> below has valid neighbor data at every interior point --
  // both fields are already ghost-exchanged by the time this runs (see
  // CFC::QueueCFCTasks).
  par_for("cfc_build_dapsi6", DevExeSpace(), 0, nmb-1, 0, ncells3-1, 0, ncells2-1,
          0, ncells1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real psi0_val = u_psi0(m,0,k,j,i);
    Real dpsi_val = delta_psi(m,0,k,j,i);
    Real psi_val = dpsi_val + psi0_val;
    Real psi_inv = 1.0/psi_val;
    Real psi_inv2 = psi_inv*psi_inv;
    Real psi_inv7 = psi_inv2*psi_inv2*psi_inv2*psi_inv;

    Real psi0_inv = 1.0/psi0_val;
    Real x = dpsi_val*psi0_inv;
    Real onepx_inv = psi0_val*psi_inv;   // = 1/(1+x) = psi0/psi
    Real onepx_inv2 = onepx_inv*onepx_inv;
    Real onepx_inv7 = onepx_inv2*onepx_inv2*onepx_inv2*onepx_inv;
    Real psi0_inv2 = psi0_inv*psi0_inv;
    Real psi0_inv7 = psi0_inv2*psi0_inv2*psi0_inv2*psi0_inv;
    Real p7 = 7.0 + x*(21.0 + x*(35.0 + x*(35.0 + x*(21.0 + x*(7.0 + x)))));
    Real reg7 = psi0_inv7 * x * p7 * onepx_inv7;   // = psi0^-7 - psi^-7

    Real alpha0psi0_val = u_alpha0_psi0(m,0,k,j,i);
    Real dalpha_psi_val = delta_alpha_psi(m,0,k,j,i);
    dap6(m,0,k,j,i) = dalpha_psi_val*psi_inv7 - alpha0psi0_val*reg7;
  });

  AthenaTensor<Real, TensorSymm::NONE, 3, 0> dap6_view;
  dap6_view.InitWithShallowSlice(dap6, 0);

  par_for("cfc_build_shift_src", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const int mk = k - ks + mg_nghost, mj = j - js + mg_nghost, mi = i - is + mg_nghost;
    Real idx[] = {1.0/size.d_view(m).dx1, 1.0/size.d_view(m).dx2,
                  1.0/size.d_view(m).dx3};
    Real ddap6[3];
    for (int b = 0; b < 3; ++b) {
      ddap6[b] = Dx<NGHOST>(b, idx, dap6_view, m, k, j, i);
    }
    for (int a = 0; a < 3; ++a) {
      Real div_total = 0.0, div_bg = 0.0;
      for (int b = 0; b < 3; ++b) {
        Real aab = a_dd(m,a,b,k,j,i);
        div_total += aab*ddap6[b];
        div_bg += (aab - a0_dd(m,a,b,k,j,i))*grad_ap6_0(m,b,k,j,i);
      }
      p_src(m,a,mk,mj,mi) += 2.0*(div_total + div_bg);
    }
  });
}

void BuildShiftSource(MeshBlockPack *pmbp, const DvceArray5D<Real> &delta_psi,
                      const DvceArray5D<Real> &delta_alpha_psi,
                      const DvceArray5D<Real> &u_psi0,
                      const DvceArray5D<Real> &u_alpha0_psi0,
                      const AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &a_dd,
                      const AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &a0_dd,
                      const AthenaTensor<Real, TensorSymm::NONE, 3, 1> &grad_ap6_0,
                      AthenaTensor<Real, TensorSymm::NONE, 3, 1> &p_src,
                      int mg_nghost, DvceArray5D<Real> &dap6) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  switch (indcs.ng) {
    case 2: BuildShiftSourceImpl<2>(pmbp, delta_psi, delta_alpha_psi, u_psi0,
                                     u_alpha0_psi0, a_dd, a0_dd, grad_ap6_0, p_src,
                                     mg_nghost, dap6); break;
    case 3: BuildShiftSourceImpl<3>(pmbp, delta_psi, delta_alpha_psi, u_psi0,
                                     u_alpha0_psi0, a_dd, a0_dd, grad_ap6_0, p_src,
                                     mg_nghost, dap6); break;
    case 4: BuildShiftSourceImpl<4>(pmbp, delta_psi, delta_alpha_psi, u_psi0,
                                     u_alpha0_psi0, a_dd, a0_dd, grad_ap6_0, p_src,
                                     mg_nghost, dap6); break;
  }
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn CFC::CFC(MeshBlockPack *pmbp, ParameterInput *pin)
//! \brief allocates intermediate fields and the 4 multigrid solvers. pmbp->padm/
//! pmbp->ptmunu are already guaranteed non-null by the caller (mesh/meshblock_pack.cpp).

CFC::CFC(MeshBlockPack *pmbp, ParameterInput *pin) :
    pmy_pack(pmbp),
    u_beta("cfc_u_beta", 1, 1, 1, 1, 1),
    u_adual("cfc_u_adual", 1, 1, 1, 1, 1),
    a_sq("cfc_a_sq", 1, 1, 1, 1, 1),
    delta_psi("cfc_delta_psi", 1, 1, 1, 1, 1),
    delta_alpha_psi("cfc_delta_alpha_psi", 1, 1, 1, 1, 1),
    u_tilde("cfc_u_tilde", 1, 1, 1, 1, 1),
    u_raw("cfc_u_raw", 1, 1, 1, 1, 1),
    u_stilde("cfc_u_stilde", 1, 1, 1, 1, 1),
    s_tilde("cfc_s_tilde", 1, 1, 1, 1, 1),
    u_p_x("cfc_u_p_x", 1, 1, 1, 1, 1),
    u_p_beta("cfc_u_p_beta", 1, 1, 1, 1, 1),
    u_p_src("cfc_u_p_src", 1, 1, 1, 1, 1),
    u_psi0("cfc_u_psi0", 1, 1, 1, 1, 1),
    u_alpha0_psi0("cfc_u_alpha0_psi0", 1, 1, 1, 1, 1),
    u_beta0("cfc_u_beta0", 1, 1, 1, 1, 1),
    u_a0dual("cfc_u_a0dual", 1, 1, 1, 1, 1),
    a0_sq("cfc_a0_sq", 1, 1, 1, 1, 1),
    u_grad_ap6_0("cfc_u_grad_ap6_0", 1, 1, 1, 1, 1),
    pmgd_pietax(nullptr),
    pmgd_pietabeta(nullptr),
    pmgd_psi(nullptr),
    pmgd_alpha(nullptr),
    pbval_pietax(nullptr),
    pbval_psi(nullptr), pbval_alpha_psi(nullptr),
    pbval_pietabeta(nullptr),
    coarse_u_pietax("cfc_coarse_u_pietax", 1, 1, 1, 1, 1),
    coarse_psi("cfc_coarse_psi", 1, 1, 1, 1, 1),
    coarse_alpha_psi("cfc_coarse_alpha_psi", 1, 1, 1, 1, 1),
    coarse_u_pietabeta("cfc_coarse_u_pietabeta", 1, 1, 1, 1, 1),
    pbval_adm(nullptr) {
  // Sized with AMR headroom (matches hydro.cpp/z4c.cpp/adm.cpp's
  // std::max(nmb_thispack, nmb_maxperrank) convention) so a dynamic-AMR regrid
  // never needs to reallocate any of CFC's own arrays.
  int nmb = std::max(pmy_pack->nmb_thispack, pmy_pack->pmesh->nmb_maxperrank);
  auto &indcs = pmy_pack->pmesh->mb_indcs;

  // Every field either finite-differenced by cfc_reconstruct.cpp or ghost-exchanged
  // (or both) is sized at mesh-NGHOST depth, matching gravity::Gravity::phi.
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*(indcs.ng)) : 1;
  Kokkos::realloc(u_beta,   nmb, 3, ncells3, ncells2, ncells1);
  Kokkos::realloc(u_adual,  nmb, 6, ncells3, ncells2, ncells1);
  Kokkos::realloc(a_sq,     nmb, 1, ncells3, ncells2, ncells1);
  Kokkos::realloc(delta_psi,       nmb, 1, ncells3, ncells2, ncells1);
  Kokkos::realloc(delta_alpha_psi, nmb, 1, ncells3, ncells2, ncells1);
  Kokkos::realloc(u_tilde,  nmb, 1, ncells3, ncells2, ncells1);
  Kokkos::realloc(u_plus_2s,    nmb, 1, ncells3, ncells2, ncells1);
  Kokkos::realloc(u_alpha_psi6, nmb, 1, ncells3, ncells2, ncells1);
  Kokkos::realloc(u_raw,    nmb, 1, ncells3, ncells2, ncells1);
  Kokkos::realloc(u_stilde, nmb, 3, ncells3, ncells2, ncells1);
  Kokkos::realloc(s_tilde,  nmb, 1, ncells3, ncells2, ncells1);
  Kokkos::realloc(u_p_x,    nmb, 4, ncells3, ncells2, ncells1);
  Kokkos::realloc(u_p_beta, nmb, 4, ncells3, ncells2, ncells1);

  // delta_psi = delta_alpha_psi = 0 (flat space) is the correct value before the
  // first solve of a run -- both are overwritten by the one-shot seeding below on
  // the first real SolveConformalFactor/SolveLapse call, but zero-init keeps the
  // array physically consistent (delta=0 <-> psi=1) until then.
  Kokkos::deep_copy(delta_psi, 0.0);
  Kokkos::deep_copy(delta_alpha_psi, 0.0);

  beta_u.InitWithShallowSlice(u_beta, 0, 2);
  a_dd.InitWithShallowSlice(u_adual, 0, 5);
  s_tilde_d.InitWithShallowSlice(u_stilde, 0, 2);
  p_x.InitWithShallowSlice(u_p_x, 0, 2);
  p_beta.InitWithShallowSlice(u_p_beta, 0, 2);

  // u_p_src: pure LoadPoissonSource input, read pointwise once, never
  // differentiated or ghost-exchanged -- sized at this solver's own (generally
  // shallower) multigrid ghost width, not mesh-NGHOST depth. Read directly here
  // (no driver exists yet in the constructor) using the same "cfc"/"mg_nghost"
  // input every mg_cfc_* driver constructor reads independently with the same
  // default, so this is guaranteed consistent with whatever ngh_ they end up with.
  int mg_nghost = pin->GetOrAddInteger("cfc", "mg_nghost", 1);
  mg_nghost_ = mg_nghost;
  int mncells1 = indcs.nx1 + 2*mg_nghost;
  int mncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*mg_nghost) : 1;
  int mncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*mg_nghost) : 1;
  Kokkos::realloc(u_p_src, nmb, 4, mncells3, mncells2, mncells1);
  p_src.InitWithShallowSlice(u_p_src, 0, 2);

  // 4 multigrid solvers: one per elliptic equation. pmgd_pietax/pmgd_pietabeta each
  // solve the packed (P_i, eta) nvar_=4 system for their Shibata pair in one call.
  pmgd_pietax    = new MGCFCVectorPoissonDriver(pmbp, pin);
  pmgd_pietabeta = new MGCFCVectorPoissonDriver(pmbp, pin);
  pmgd_psi       = new MGCFCConformalFactorDriver(pmbp, pin);
  pmgd_alpha     = new MGCFCLapseDriver(pmbp, pin);
  // Labels only tag MGTimers output; the two vector-Poisson drivers are the same
  // class, so they cannot be told apart from inside the constructor.
  pmgd_pietax->mg_label_    = "X^i     ";
  pmgd_pietabeta->mg_label_ = "beta^i  ";
  pmgd_psi->mg_label_       = "psi     ";
  pmgd_alpha->mg_label_     = "alpha_psi";

  // Post-multigrid ghost exchange: one MeshBoundaryValuesCC + coarse shadow array
  // per field cfc_reconstruct.cpp later differentiates, mirroring
  // z4c::Z4c::pbval_u/coarse_u0. is_z4c=true throughout -- these fields need the
  // same smooth-across-AMR-levels treatment Z4c's own metric variables get, so they
  // reuse z4c's higher-order (Lagrange) restrict/prolong path (see cfc.hpp).
  pbval_pietax = new MeshBoundaryValuesCC(pmbp, pin, true);
  pbval_pietax->InitializeBuffers(4);
  pbval_psi = new MeshBoundaryValuesCC(pmbp, pin, true);
  pbval_psi->InitializeBuffers(1);
  pbval_alpha_psi = new MeshBoundaryValuesCC(pmbp, pin, true);
  pbval_alpha_psi->InitializeBuffers(1);
  pbval_pietabeta = new MeshBoundaryValuesCC(pmbp, pin, true);
  pbval_pietabeta->InitializeBuffers(4);

  // Ghost-exchange for padm->u_adm itself -- see pbval_adm's comment in cfc.hpp.
  // CFC never runs with z4c active, so u_adm always carries the full nadm channels
  // (adm::ADM::ADM's constructor only shrinks it and aliases alpha/beta_u into
  // pz4c->u0 when pz4c != nullptr -- never true here).
  pbval_adm = new MeshBoundaryValuesCC(pmbp, pin, true);
  pbval_adm->InitializeBuffers(adm::ADM::nadm);

  // coarse_* shadow arrays: only needed with SMR/AMR (RestrictCC/ProlongateCC are
  // internal no-ops otherwise), matching z4c.cpp's identical multilevel guard.
  if (pmy_pack->pmesh->multilevel) {
    int nccells1 = indcs.cnx1 + 2*(indcs.ng);
    int nccells2 = (indcs.cnx2 > 1) ? (indcs.cnx2 + 2*(indcs.ng)) : 1;
    int nccells3 = (indcs.cnx3 > 1) ? (indcs.cnx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(coarse_u_pietax,    nmb, 4, nccells3, nccells2, nccells1);
    Kokkos::realloc(coarse_psi,         nmb, 1, nccells3, nccells2, nccells1);
    Kokkos::realloc(coarse_alpha_psi,   nmb, 1, nccells3, nccells2, nccells1);
    Kokkos::realloc(coarse_u_pietabeta, nmb, 4, nccells3, nccells2, nccells1);
    // u_adm's own coarse shadow is padm->coarse_u_adm (adm.cpp:57-67) -- no
    // separate CFC-owned copy; see cfc.hpp's pbval_adm comment.
  }

  // <cfc> puncture_enabled (default false): opt-in stationary BH trumpet background
  // (CFC_PUNCTURE_TDE_PLAN.md Sec 3.2/3.6, cfc_puncture.hpp). u_psi0/u_alpha0_psi0/
  // u_a0dual/a0_sq are always allocated (unconditionally, at the same AMR headroom as
  // every other CFC array above) since cfc_reconstruct.cpp/cfc.cpp/mg_cfc_lapse.cpp/
  // mg_cfc_conformal_factor.cpp now read them unconditionally (Sec 5 Phase A items 3,
  // 5, 6) -- filled with the exact flat-space values (psi0=1, alpha0*psi0=1,
  // Ahat0_ij=0, Ahat0^2=0, beta0^i=0, grad_ap6_0^i=0) when disabled, so every existing
  // non-TDE run is a bit-for-bit no-op.
  puncture_enabled_ = pin->GetOrAddBoolean("cfc", "puncture_enabled", false);
  // Phase 2 Milestone 1 (code/COMOVING_GAUGE_DESIGN.md §2.1; DEVELOPMENT.md item 75):
  // a puncture that moves on the grid as x_BH(t) = x_BH(0) - xi(t) in the comoving gauge.
  puncture_moving_ = pin->GetOrAddBoolean("cfc", "puncture_moving", false);
  for (int a = 0; a < 3; ++a) {
    xbh0_[a] = pin->GetOrAddReal("cfc", "puncture_x" + std::to_string(a + 1), 0.0);
  }
  puncture_mass_ = pin->GetOrAddReal("cfc", "puncture_mass", 1.0);
  // Phase 2 M2 xidot controller (see cfc.hpp). Parsed here, before the puncture fill
  // below, because a restart's x_BH comes from the restored xi.
  frame_ctrl_ = pin->GetOrAddBoolean("cfc", "frame_controller", false);
  // restart_state: the MG seed flags (SaveRestartScalars); absent before it
  if (pin->DoesParameterExist("cfc", "restart_psi_seeded")) {
    rst_psi_seeded_ = pin->GetBoolean("cfc", "restart_psi_seeded");
    rst_alpha_psi_seeded_ = pin->GetBoolean("cfc", "restart_alpha_psi_seeded");
  }
  if (frame_ctrl_) {
    fc_rho_thr_ = pin->GetReal("cfc", "frame_track_rho");        // required (code units)
    fc_rtrk_ = pin->GetReal("cfc", "frame_track_radius");        // required
    fc_tau_x_ = pin->GetOrAddReal("cfc", "frame_tau_x", 20.0);
    fc_tau_a_ = pin->GetOrAddReal("cfc", "frame_tau_a", 2.0);
    fc_atm_rho_ = pin->GetOrAddReal("cfc", "frame_atm_rho", 0.0);
    fc_atm_gate_ = pin->GetOrAddReal("cfc", "frame_atm_gate", 0.0);
    // target point x'_tgt: given, or (default) the t = 0 centroid of the gas tracked
    // around these coordinates (the origin unless set)
    fc_tgt_given_ = pin->DoesParameterExist("cfc", "frame_target_x1") ||
                    pin->DoesParameterExist("cfc", "frame_target_x2") ||
                    pin->DoesParameterExist("cfc", "frame_target_x3");
    for (int a = 0; a < 3; ++a) {
      fc_tgt_[a] = pin->GetOrAddReal("cfc", "frame_target_x" + std::to_string(a + 1), 0.0);
    }
    // R-050 B2: target mode and the automatic switch (cfc.hpp)
    auto mode_id = [](const std::string &m) -> int {
      if (m == "remnant") return 0;
      if (m == "bound_centroid") return 1;
      if (m == "ramp") return 2;
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<cfc> frame_target_mode/frame_switch_mode: unknown mode '"
                << m << "' (remnant, bound_centroid, ramp)" << std::endl;
      std::exit(EXIT_FAILURE);
    };
    fc_mode_ = mode_id(pin->GetOrAddString("cfc", "frame_target_mode", "remnant"));
    fc_switch_mode_ = mode_id(pin->GetOrAddString("cfc", "frame_switch_mode",
                                                  "bound_centroid"));
    fc_switch_frac_ = pin->GetOrAddReal("cfc", "frame_switch_mass_frac", 0.0);
    fc_ramp_T_ = pin->GetOrAddReal("cfc", "frame_ramp_time", 50.0);
    {   // R-112 item 4 (cfc.hpp): the ramp shape and the exit triggers, default off
      const std::string sh = pin->GetOrAddString("cfc", "frame_ramp_shape", "linear");
      if (sh != "linear" && sh != "smoothstep") {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<cfc> frame_ramp_shape: unknown '" << sh
                  << "' (linear, smoothstep)" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      fc_ramp_shape_ = (sh == "smoothstep") ? 1 : 0;
      fc_app_r_ = pin->GetOrAddReal("cfc", "frame_switch_approach_radius", 0.0);
      fc_acc_rate_ = pin->GetOrAddReal("cfc", "frame_switch_accretion_rate", 0.0);
      fc_acc_r_ = pin->GetOrAddReal("cfc", "frame_switch_accretion_radius", 6.0);
      fc_acc_w_ = pin->GetOrAddReal("cfc", "frame_switch_accretion_shell", 2.0);
    }
    fc_bound_rho_ = pin->GetOrAddReal("cfc", "frame_bound_rho", 1.0e-4*fc_rho_thr_);
    // R-054 (R-053 §4): bound_centroid must ignore the near-BH atmosphere zone, so both
    // gates must cover it: frame_bound_rho >= the zone's rho_thr (the default 1e-4 x
    // frame_track_rho = 1e-5 rho_c already exceeds every rho_thr in R-054's sweep, <= 1e-8
    // rho_c) and frame_bound_rmin >= its r_s (6 M: C-046 S0 shows r_s = 3 leaves the limit at
    // r_BH ~ 3). A future zone must enforce both (max of the two) when it is switched on.
    fc_bound_rmin_ = pin->GetOrAddReal("cfc", "frame_bound_rmin", 6.0);
    if (pin->GetOrAddBoolean("mhd", "atmosphere_zone", false)) {   // R-059 S1: enforce it
      Real zr = pin->GetReal("mhd", "atmosphere_zone_radius");
      Real zrho = pin->GetReal("mhd", "atmosphere_zone_rho");
      if (fc_bound_rmin_ < zr || fc_bound_rho_ < zrho) {
        if (global_variable::my_rank == 0) {
          std::cout << "<cfc> atmosphere_zone: frame_bound_rmin " << fc_bound_rmin_
                    << " -> " << std::max(fc_bound_rmin_, zr) << ", frame_bound_rho "
                    << fc_bound_rho_ << " -> " << std::max(fc_bound_rho_, zrho) << std::endl;
        }
        fc_bound_rmin_ = std::max(fc_bound_rmin_, zr);
        fc_bound_rho_ = std::max(fc_bound_rho_, zrho);
      }
    }
    if (pin->DoesParameterExist("cfc", "frame_switch_mref")) {
      fc_m0_ = pin->GetReal("cfc", "frame_switch_mref");
    }
    // restored state (written by SaveFrameState into the restart's parameter copy)
    fc_init_ = pin->DoesParameterExist("cfc", "frame_ctrl_t");
    if (fc_init_) {
      fc_t_ = pin->GetReal("cfc", "frame_ctrl_t");
      for (int a = 0; a < 3; ++a) {
        std::string n = std::to_string(a + 1);
        fc_xi_[a] = pin->GetReal("cfc", "frame_ctrl_xi" + n);
        fc_xd_[a] = pin->GetReal("cfc", "frame_ctrl_xidot" + n);
        fc_tgt_[a] = pin->GetReal("cfc", "frame_ctrl_tgt" + n);
        fc_vprev_[a] = pin->GetReal("cfc", "frame_ctrl_vprev" + n);
        fc_acc_[a] = pin->GetReal("cfc", "frame_ctrl_acc" + n);
      }
      // B2 state (absent in restarts written before R-050: remnant mode, m_ref unknown
      // unless frame_switch_mref is given)
      if (pin->DoesParameterExist("cfc", "frame_ctrl_mode")) {
        fc_mode_ = pin->GetInteger("cfc", "frame_ctrl_mode");
        fc_tsw_ = pin->GetReal("cfc", "frame_ctrl_tsw");
        for (int a = 0; a < 3; ++a) {
          fc_xdsw_[a] = pin->GetReal("cfc", "frame_ctrl_xdsw" + std::to_string(a + 1));
        }
        if (!pin->DoesParameterExist("cfc", "frame_switch_mref")) {
          fc_m0_ = pin->GetReal("cfc", "frame_ctrl_m0");
        }
      }
    }
    if (fc_switch_frac_ > 0.0 && fc_init_ && !(fc_m0_ > 0.0) &&
        global_variable::my_rank == 0) {
      std::cout << "### WARNING <cfc> frame_switch_mass_frac > 0 but the restart carries no "
                << "reference mass (frame_ctrl_m0) and frame_switch_mref is unset: the "
                << "automatic switch is disabled" << std::endl;
    }
  }
  // Lower clamp on the assembled lapse (see AssembleLapseShiftK). Default 0.0
  // leaves every existing run bit-for-bit unchanged wherever alpha > 0.
  alpha_floor_ = pin->GetOrAddReal("cfc", "alpha_floor", 0.0);
  // <cfc> solve_every_stages (see cfc.hpp). Read without GetOrAdd, as residual_scale.
  if (pin->DoesParameterExist("cfc", "solve_every_stages")) {
    solve_every_ = pin->GetInteger("cfc", "solve_every_stages");
    // a restart written in this mode carries the held potentials and a pending forced
    // solve; a fresh start or any other restart solves at its first stage
    if (solve_every_ > 1) {
      force_solve_ = !pin->DoesParameterExist("cfc", "restart_cfc_held") ||
                     !pin->GetBoolean("cfc", "restart_cfc_held") ||
                     pin->GetBoolean("cfc", "restart_cfc_force_solve");
    }
  }
  // <cfc> residual_scale (see cfc.hpp). Read without GetOrAdd, so an input without the
  // keys leaves the parameter block (and the restart files) unchanged.
  if (pin->DoesParameterExist("cfc", "residual_scale")) {
    rs_on_ = true;
    rs_const_ = pin->GetReal("cfc", "residual_scale");
  }
  if (pin->DoesParameterExist("cfc", "residual_scale_t1")) {
    if (rs_on_) {
      std::cout << "### FATAL ERROR in CFC: <cfc> residual_scale and residual_scale_t1 "
                << "are exclusive (a constant f or a ramp)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    rs_on_ = rs_ramp_ = true;
    rs_t1_ = pin->GetReal("cfc", "residual_scale_t1");
    rs_t0_ = pin->DoesParameterExist("cfc", "residual_scale_t0") ?
             pin->GetReal("cfc", "residual_scale_t0") : 0.0;
    rs_f0_ = pin->DoesParameterExist("cfc", "residual_scale_f0") ?
             pin->GetReal("cfc", "residual_scale_f0") : 1.0;
    rs_f1_ = pin->DoesParameterExist("cfc", "residual_scale_f1") ?
             pin->GetReal("cfc", "residual_scale_f1") : 0.0;
    if (!(rs_t1_ > rs_t0_)) {
      std::cout << "### FATAL ERROR in CFC: <cfc> residual_scale_t1 must exceed "
                << "residual_scale_t0" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  if (rs_on_ && solve_every_ > 1) {
    std::cout << "### FATAL ERROR in CFC: <cfc> residual_scale and solve_every_stages > 1 "
              << "are not combined" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (rs_on_) {
    rs_log_ = pin->DoesParameterExist("cfc", "residual_scale_log_interval") ?
              pin->GetInteger("cfc", "residual_scale_log_interval") : 10;
    if (global_variable::my_rank == 0) {
      if (rs_ramp_) {
        std::cout << "CFC residual_scale: C1 ramp f = " << rs_f0_ << " (t <= " << rs_t0_
                  << ") -> " << rs_f1_ << " (t >= " << rs_t1_ << ")" << std::endl;
      } else {
        std::cout << "CFC residual_scale: constant f = " << rs_const_ << std::endl;
      }
    }
  }
  // Lower clamp on psi at the psi solve's output; see ApplyPsiFloor. Default 0.0
  // (disabled) so every existing fixture is bitwise unchanged.
  psi_floor_ = pin->GetOrAddReal("cfc", "psi_floor", 0.0);
  accrete_to_puncture_ = pin->GetOrAddBoolean("cfc", "accrete_to_puncture", true);
  puncture_mass_init_ = puncture_mass_;

  // ADM-mass extraction radii, discovered by existence (<cfc> adm_mass_r0, _r1, ...).
  for (int n = 0; n < 8; ++n) {
    std::string key = "adm_mass_r" + std::to_string(n);
    if (pin->DoesParameterExist("cfc", key)) {
      adm_mass_radii_.push_back(pin->GetReal("cfc", key));
    }
  }
  adm_mass_dr_   = pin->GetOrAddReal("cfc", "adm_mass_dr", 1.0);
  adm_mass_nlev_ = pin->GetOrAddInteger("cfc", "adm_mass_nlev", 3);
  {
    std::string ctr = pin->GetOrAddString("cfc", "adm_mass_center", "origin");
    if (ctr == "origin") {
      adm_mass_center_com_ = false;
    } else if (ctr == "com") {
      adm_mass_center_com_ = true;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<cfc> adm_mass_center = '" << ctr
                << "' not recognized; must be 'origin' or 'com'" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (adm_mass_center_com_ && !puncture_enabled_) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<cfc> adm_mass_center = com requires the puncture "
                << "background: r_com_mass_ is only computed when puncture is enabled"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  r_com_mass_[0] = r_com_mass_[1] = r_com_mass_[2] = 0.0;
  r_com_X_[0] = r_com_X_[1] = r_com_X_[2] = 0.0;
  r_com_beta_[0] = r_com_beta_[1] = r_com_beta_[2] = 0.0;
  Kokkos::realloc(u_psi0,        nmb, 1, ncells3, ncells2, ncells1);
  Kokkos::realloc(u_alpha0_psi0, nmb, 1, ncells3, ncells2, ncells1);
  Kokkos::realloc(u_a0dual,      nmb, 6, ncells3, ncells2, ncells1);
  a0_dd.InitWithShallowSlice(u_a0dual, 0, 5);
  Kokkos::realloc(a0_sq,         nmb, 1, ncells3, ncells2, ncells1);
  Kokkos::realloc(u_beta0,       nmb, 3, ncells3, ncells2, ncells1);
  Kokkos::realloc(u_grad_ap6_0,  nmb, 3, ncells3, ncells2, ncells1);
  beta0_u.InitWithShallowSlice(u_beta0, 0, 2);
  grad_ap6_0.InitWithShallowSlice(u_grad_ap6_0, 0, 2);
  if (puncture_enabled_) {
    // x_BH at the current (start or restart) time; the frame parameters are read
    // further below, so evaluate xi(t) from the input directly here.
    Real xbh[3];
    {
      Real t0 = pmbp->pmesh->time;
      for (int a = 0; a < 3; ++a) {
        std::string n = std::to_string(a + 1);
        Real v0 = pin->GetOrAddReal("cfc", "frame_vel" + n, 0.0);
        Real a0 = pin->GetOrAddReal("cfc", "frame_accel" + n, 0.0);
        Real xi = frame_ctrl_ ? (fc_init_ ? fc_xi_[a] + fc_xd_[a]*(t0 - fc_t_) : 0.0)
                              : v0*t0 + 0.5*a0*t0*t0;
        xbh[a] = xbh0_[a] - (puncture_moving_ ? xi : 0.0);
        pmbp->padm->bc_center[a] = puncture_moving_ ? xbh[a] : 0.0;
      }
    }
    FillPunctureBackground(pmbp, puncture_mass_, xbh, u_psi0, u_alpha0_psi0, beta0_u,
                           a0_dd, a0_sq, grad_ap6_0);
    for (int a = 0; a < 3; ++a) { bg_xbh_[a] = xbh[a]; }
    pmgd_psi->SetPunctureCenter(xbh); pmgd_alpha->SetPunctureCenter(xbh);
  } else {
    Kokkos::deep_copy(u_psi0, 1.0);
    Kokkos::deep_copy(u_alpha0_psi0, 1.0);
    Kokkos::deep_copy(u_a0dual, 0.0);
    Kokkos::deep_copy(a0_sq, 0.0);
    Kokkos::deep_copy(u_beta0, 0.0);
    Kokkos::deep_copy(u_grad_ap6_0, 0.0);
  }

  // X^i/psi fixed-point-iteration controls, used by InitializeMetric() only.
  cfc_init_iter_max_ = pin->GetOrAddInteger("cfc", "init_iter_max", 50);
  cfc_init_tol_ = pin->GetOrAddReal("cfc", "init_tol", 1.0e-10);
  cfc_init_verbose_ = pin->GetOrAddBoolean("cfc", "init_verbose", false);
  cfc_init_omega_ = pin->GetOrAddReal("cfc", "init_omega", 1.0);
  // Default true: the psi^5 formulation converges dramatically faster (2-3 outer
  // iterations vs. 25-80+) and matches the default formulation's accuracy for
  // every star tested except very compact/unstable ones, which diverge to NaN
  // under it -- inputs known to be that compact must explicitly set this false;
  // there is no auto-detection or graceful fallback.
  cfc_init_use_psi5_ = pin->GetOrAddBoolean("cfc", "init_use_psi5_source", true);
  cfc_init_freeze_conserved_ = pin->GetOrAddBoolean("cfc", "init_freeze_conserved",
                                                     false);
  // The two modes above encode mutually incompatible assumptions about what's held
  // fixed. init_freeze_conserved holds Utilde = psi^6*U (the densitized conserved
  // source) fixed across RunXPsiSolvePass's single Newton solve; init_use_psi5_
  // source instead holds U_raw (undensitized, i.e. implicitly the primitives)
  // fixed and lets Utilde vary self-consistently with the Newton iterate -- the
  // opposite assumption. Force init_use_psi5_source off whenever
  // init_freeze_conserved is on (covers both an explicit true and its own
  // default-true silently combining with it).
  if (cfc_init_freeze_conserved_ && cfc_init_use_psi5_) {
    if (global_variable::my_rank == 0) {
      std::cout << "### WARNING in CFC::CFC" << std::endl
                << "<cfc> init_freeze_conserved=true is incompatible with"
                << " init_use_psi5_source=true (the latter defaults to true):"
                << " freezing Utilde and letting"
                << " the Newton solve treat U_raw=Utilde/psi^6 as the fixed"
                << " quantity are contradictory assumptions. Forcing"
                << " init_use_psi5_source=false for this run." << std::endl;
    }
    cfc_init_use_psi5_ = false;
  }

  // Delta-n-cycle solve cadence; see cfc.hpp. <=0 (default) is a no-op.
  cfc_solve_interval_ = pin->GetOrAddInteger("cfc", "solve_interval", 0);
  if (solve_every_ > 0 && cfc_solve_interval_ > 0) {
    std::cout << "### FATAL ERROR in CFC: <cfc> solve_every_stages and solve_interval > 0 "
              << "are exclusive" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Comoving gauge (uniform shift xidot(t)); see cfc.hpp and DEVELOPMENT.md item 68.
  // All zero (default) leaves every run bit-for-bit unchanged.
  frame_enabled_ = false;
  frame_assembled_ = false;
  for (int a = 0; a < 3; ++a) {
    std::string n = std::to_string(a + 1);
    frame_vel0_[a] = pin->GetOrAddReal("cfc", "frame_vel" + n, 0.0);
    frame_accel_[a] = pin->GetOrAddReal("cfc", "frame_accel" + n, 0.0);
    if (frame_vel0_[a] != 0.0 || frame_accel_[a] != 0.0) frame_enabled_ = true;
  }
  if (frame_ctrl_) {
    if (frame_enabled_) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<cfc> frame_controller replaces frame_vel/frame_accel; "
                << "set only one" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    frame_enabled_ = true;
  }
  if (frame_enabled_ && puncture_enabled_ && !puncture_moving_) {
    // Phase 1 has no moving BH: the trumpet background, excision masks, BH refinement
    // and ADM falloff all stay at the grid origin, so xidot != 0 would describe a
    // different spacetime, not a gauge change. Needs Phase 2 (COMOVING_GAUGE_DESIGN.md).
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<cfc> frame_vel/frame_accel cannot be combined with "
              << "puncture_enabled = true unless <cfc> puncture_moving = true (Phase 2 "
              << "M1, DEVELOPMENT.md items 68 and 75)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (frame_enabled_) {
    // A uniform translation normal to a reflecting boundary moves the mirror: not a
    // pure gauge of the reflected problem. Refuse rather than silently mis-reflect.
    auto &mbcs = pmy_pack->pmesh->mesh_bcs;
    const BoundaryFace inner[3] = {BoundaryFace::inner_x1, BoundaryFace::inner_x2,
                                   BoundaryFace::inner_x3};
    const BoundaryFace outer[3] = {BoundaryFace::outer_x1, BoundaryFace::outer_x2,
                                   BoundaryFace::outer_x3};
    for (int a = 0; a < 3; ++a) {
      bool moving = frame_ctrl_ || frame_vel0_[a] != 0.0 || frame_accel_[a] != 0.0;
      if (moving && (mbcs[inner[a]] == BoundaryFlag::reflect ||
                     mbcs[outer[a]] == BoundaryFlag::reflect)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<cfc> frame_vel/frame_accel component " << a + 1
                  << " is nonzero but a boundary normal to x" << a + 1
                  << " is reflecting" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
    if (global_variable::my_rank == 0 && frame_ctrl_) {
      std::cout << "CFC comoving gauge: xidot controller, track rho > " << fc_rho_thr_
                << " within " << fc_rtrk_ << " of the target, tau_x = " << fc_tau_x_
                << ", tau_a = " << fc_tau_a_
                << (fc_init_ ? " (state restored)" : "") << std::endl;
    } else if (global_variable::my_rank == 0) {
      std::cout << "CFC comoving gauge: xidot(t) = (" << frame_vel0_[0] << ", "
                << frame_vel0_[1] << ", " << frame_vel0_[2] << ") + ("
                << frame_accel_[0] << ", " << frame_accel_[1] << ", "
                << frame_accel_[2] << ")*t" << std::endl;
    }
  }

  // The comoving frame velocity the floor atmosphere (atmosphere_frame_velocity), the CFL
  // and ADMBCs read (padm->frame_vel_u) at the start/restart time. AssembleADM sets it every
  // stage, but a restart skips InitializeMetric, so its initial C2P and first dt saw xidot = 0
  // and gave floored gas v = 0 against the shift (C-100; restarts not bit-continuous).
  if (frame_enabled_) {
    Real xd[3];
    FrameVel(pmbp->pmesh->time, xd);
    for (int a = 0; a < 3; ++a) { pmbp->padm->frame_vel_u[a] = xd[a]; }
  }

  // Elliptic source/output NaN scan; see cfc.hpp. Default 0 = off.
  src_nan_check_ = pin->GetOrAddInteger("cfc", "src_nan_check", 0);
  indcs_ng_ = pmy_pack->pmesh->mb_indcs.ng;
}

//----------------------------------------------------------------------------------------
//! \fn bool CFC::DoSolveThisStage(Driver *pdriver, int stage) const
//! \brief see cfc_solve_interval_'s comment in cfc.hpp.

//----------------------------------------------------------------------------------------
//! \fn int CFC::CheckFinite(const char*, const DvceArray5D<Real>&, int, int)
//! \brief see cfc.hpp. Scans for NaN/Inf and reports the first offender globally.

int CFC::CheckFinite(const char *label, const DvceArray5D<Real> &a, int nchan,
                     int stage, int ngh) const {
  if (src_nan_check_ <= 0) { return 0; }

  const int nmb = a.extent_int(0);
  const int n3  = a.extent_int(2);
  const int n2  = a.extent_int(3);
  const int n1  = a.extent_int(4);
  const int nkji = n3*n2*n1, nji = n2*n1;
  const int total = nmb*nchan*nkji;
  auto arr = a;

  int nbad = 0;
  // Flat index of the lowest-indexed offender on this rank (INT_MAX if none), so the
  // MPI_MINLOC below picks a deterministic single cell to report.
  int firstbad = INT_MAX;
  Kokkos::parallel_reduce("cfc_check_finite",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, total),
    KOKKOS_LAMBDA(const int idx, int &lcnt, int &lmin) {
      const int m = idx/(nchan*nkji);
      const int r = idx - m*(nchan*nkji);
      const int n = r/nkji;
      const int t = r - n*nkji;
      const int k = t/nji;
      const int j = (t - k*nji)/n1;
      const int i = t - k*nji - j*n1;
      const Real v = arr(m,n,k,j,i);
      // NaN: v!=v. Inf: exceeds any representable finite magnitude of interest.
      // Interior only: ghost zones of some of these arrays are never written.
      const bool interior = (i >= ngh) && (i < n1-ngh) && (j >= ngh) && (j < n2-ngh) &&
                            (k >= ngh) && (k < n3-ngh);
      const Real vv = interior ? v : static_cast<Real>(0.0);
      const bool bad = !(vv == vv) || (vv > 1.0e300) || (vv < -1.0e300);
      if (bad) { lcnt += 1; if (idx < lmin) { lmin = idx; } }
    }, Kokkos::Sum<int>(nbad), Kokkos::Min<int>(firstbad));
  Kokkos::fence();

  int gbad = nbad;
  struct { int idx; int rank; } lin, gout;
  lin.idx = firstbad; lin.rank = global_variable::my_rank;
  gout = lin;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(&nbad, &gbad, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&lin, &gout, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);
#endif
  if (gbad == 0) { return 0; }

  // Decode and print the winning cell's coordinates/value from its owning rank, then
  // let rank 0 emit the summary line (two short messages beat 192 duplicates).
  if (global_variable::my_rank == gout.rank && gout.idx != INT_MAX) {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    auto &size = pmy_pack->pmb->mb_size;
    const int idx = gout.idx;
    const int m = idx/(nchan*nkji);
    const int r = idx - m*(nchan*nkji);
    const int n = r/nkji;
    const int t = r - n*nkji;
    const int k = t/nji;
    const int j = (t - k*nji)/n1;
    const int i = t - k*nji - j*n1;
    auto hv = Kokkos::create_mirror_view(Kokkos::subview(a, m, n,
                Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()));
    Kokkos::deep_copy(hv, Kokkos::subview(a, m, n,
                Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()));
    size.template sync<HostMemSpace>();
    // These arrays are allocated at differing ghost depths, so (k,j,i) are array
    // indices; convert with the mesh's own ng only when the extents match a
    // mesh-depth array, otherwise report raw indices.
    const bool mesh_depth = (n1 == indcs.nx1 + 2*indcs.ng);
    Real x1v = 0.0, x2v = 0.0, x3v = 0.0;
    if (mesh_depth) {
      x1v = CellCenterX(i-indcs.is, indcs.nx1, size.h_view(m).x1min, size.h_view(m).x1max);
      x2v = CellCenterX(j-indcs.js, indcs.nx2, size.h_view(m).x2min, size.h_view(m).x2max);
      x3v = CellCenterX(k-indcs.ks, indcs.nx3, size.h_view(m).x3min, size.h_view(m).x3max);
    }
    std::cout << "### CFC NONFINITE [" << label << "] stage=" << stage
              << " ncycle=" << pmy_pack->pmesh->ncycle
              << " count=" << gbad
              << " first: rank=" << gout.rank << " mb=" << m << " chan=" << n
              << " (k,j,i)=(" << k << "," << j << "," << i << ")";
    if (mesh_depth) {
      std::cout << " x=(" << x1v << "," << x2v << "," << x3v << ")";
    } else {
      std::cout << " [solver-depth array, raw indices]";
    }
    std::cout << " val=" << hv(k,j,i) << std::endl;
  }
  return gbad;
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::ApplyPsiFloor()
//! \brief Clamp psi >= psi_floor_ on the psi solve's OUTPUT.
//!
//! WHY AT THE SOLVE AND NOT AT THE ADM WRITE. The pre-existing alpha_floor lived in
//! AssembleLapseShiftK (cfc_reconstruct.cpp), which runs in AssembleFinalTask -- the
//! LAST task of the step. That clamps only `adm.alpha`, the copy handed to
//! con2prim/hydro. It does NOT protect:
//!   - the shift source, which consumes the RAW alpha*psi and psi via
//!     ap6 = (delta_alpha_psi + u_alpha0_psi0)/psi^7 (AssembleVectorSource, step 6);
//!   - the next cycle's psi/lapse solves, which re-consume the raw residuals.
//! So a non-positive psi or alpha could enter an elliptic source while every ADM-side
//! diagnostic still read a healthy floored value. Clamping at the solve output closes
//! that gap and makes the floored value the one everything downstream sees.
//!
//! GHOST CELLS. This runs BEFORE the field's ghost exchange
//! (CFC_RestPsi/.../CFC_BCSPsi all depend on CFC_SolveLapse, which is downstream of
//! this), so the exchange propagates already-clamped data into ghosts. The loop
//! nevertheless covers the FULL array extent, not just the interior: that costs
//! nothing and also clears any stale sub-floor value sitting in a ghost from a
//! previous step. (Caveat for a future reader: ProlongPsi interpolates fine-level
//! ghosts from coarse data, and a non-monotone prolongation stencil could in
//! principle undershoot the floor again. If that is ever observed, the fix is a
//! second clamp after CFC_BCSPsi -- not a change here.)
//!
//! RESIDUAL ARITHMETIC. delta_psi is a residual: psi = delta_psi + psi0. Clamping psi
//! therefore rearranges onto the residual as delta_psi >= psi_floor - psi0.

void CFC::ApplyPsiFloor() {
  if (psi_floor_ <= 0.0) { return; }
  auto &dpsi = delta_psi;
  auto &psi0 = u_psi0;
  Real floor_val = psi_floor_;
  const int nmb = dpsi.extent_int(0);
  const int n3 = dpsi.extent_int(2);
  const int n2 = dpsi.extent_int(3);
  const int n1 = dpsi.extent_int(4);
  par_for("cfc_apply_psi_floor", DevExeSpace(), 0, nmb-1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // NOT fmax: IEEE fmax(NaN, x) returns x, which would SILENTLY ABSORB a NaN solve
    // output into the floor value -- masking the very failure this file exists to
    // diagnose and yielding a quietly wrong spacetime instead of a loud one. The
    // explicit comparison is false for NaN, so NaN passes through untouched and is
    // still caught by src_nan_check / NANS_IN_CONS. This clamp is for non-positive
    // FINITE values, which is what it was asked to prevent.
    const Real v = dpsi(m,0,k,j,i);
    const Real lo = floor_val - psi0(m,0,k,j,i);
    dpsi(m,0,k,j,i) = (v < lo) ? lo : v;
  });
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::ApplyAlphaFloor()
//! \brief Clamp alpha >= alpha_floor_ on the lapse solve's OUTPUT.
//!
//! Same rationale and the same ghost-cell ordering as ApplyPsiFloor (this runs before
//! CFC_RestAlphaPsi/.../CFC_BCSAlphaPsi, and before CFC_BuildSrcBeta which depends on
//! CFC_BCSAlphaPsi). The clamp is on alpha, NOT on alpha*psi, to match the existing
//! AssembleLapseShiftK semantics exactly:
//!   alpha = alpha_psi/psi >= alpha_floor   <=>   alpha_psi >= alpha_floor*psi
//! and with alpha_psi = delta_alpha_psi + alpha0*psi0 this rearranges onto the residual
//! as delta_alpha_psi >= alpha_floor*psi - alpha0*psi0. psi here is the ALREADY-FLOORED
//! psi, since ApplyPsiFloor ran in the psi solve one step earlier in the task graph.
//!
//! The late clamp in AssembleLapseShiftK is deliberately LEFT IN PLACE: it is
//! idempotent once this one has run, and it still covers the alpha_floor_ == 0 case
//! (where this routine is a no-op) exactly as before.

void CFC::ApplyAlphaFloor() {
  if (alpha_floor_ <= 0.0) { return; }
  auto &dap = delta_alpha_psi;
  auto &a0p0 = u_alpha0_psi0;
  auto &dpsi = delta_psi;
  auto &psi0 = u_psi0;
  Real floor_val = alpha_floor_;
  const int nmb = dap.extent_int(0);
  const int n3 = dap.extent_int(2);
  const int n2 = dap.extent_int(3);
  const int n1 = dap.extent_int(4);
  par_for("cfc_apply_alpha_floor", DevExeSpace(), 0, nmb-1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // See ApplyPsiFloor on why this is an explicit comparison and not fmax: NaN must
    // pass through rather than be absorbed into the floor.
    const Real psi_val = dpsi(m,0,k,j,i) + psi0(m,0,k,j,i);
    const Real v = dap(m,0,k,j,i);
    const Real lo = floor_val*psi_val - a0p0(m,0,k,j,i);
    dap(m,0,k,j,i) = (v < lo) ? lo : v;
  });
}

bool CFC::DoSolveThisStage(Driver *pdriver, int stage) const {
  // stage<1: a direct call from InitializeMetric()/ReinitializeMetricForAMR, not
  // a normal 1..nexp_stages substage -- always run. Skipping it would no-op a
  // Send while a matching Recv still waits, deadlocking every rank (reproduced
  // on ~/athenak_cfc; DEVELOPMENT.md item 61).
  if (stage < 1) {
    return true;
  }
  if (solve_every_ > 0) {
    return solve_now_;
  }
  if (cfc_solve_interval_ <= 0) {
    return true;
  }
  return (pmy_pack->pmesh->ncycle % cfc_solve_interval_ == 0) &&
         (stage == pdriver->nexp_stages);
}

//----------------------------------------------------------------------------------------
//! \fn CFC::~CFC()

CFC::~CFC() {
  delete pmgd_pietax;
  delete pmgd_pietabeta;
  delete pmgd_psi;
  delete pmgd_alpha;
  delete pbval_pietax;
  delete pbval_psi;
  delete pbval_alpha_psi;
  delete pbval_pietabeta;
  delete pbval_adm;
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::QueueCFCTasks()
//! \brief queues CFC's tasks into the shared NumericalRelativity task graph.

void CFC::QueueCFCTasks() {
  using namespace numrel;  // NOLINT(build/namespaces)
  NumericalRelativity *pnr = pmy_pack->pnr;

  // Post all 5 fields' non-blocking MPI receives up front, before any of this
  // stage's Send/Recv rounds run -- mirrors z4c::Z4c_Recv (Task_Start). See
  // InitRecvTask's comment in cfc.hpp for why this is required for correctness.
  pnr->QueueTask(&CFC::InitRecvTask, this, CFC_InitRecv, "CFC_InitRecv", Task_Start);

  pnr->QueueTask(&CFC::SolveVecXTask, this, CFC_BuildSrcX, "CFC_BuildSrcX",
                 Task_Run, {MHD_AddSrc});

  pnr->QueueTask(&CFC::RestPiEtaXTask, this, CFC_RestPiEtaX, "CFC_RestPiEtaX",
                 Task_Run, {CFC_BuildSrcX});
  pnr->QueueTask(&CFC::SendPiEtaXTask, this, CFC_SendPiEtaX, "CFC_SendPiEtaX",
                 Task_Run, {CFC_RestPiEtaX});
  pnr->QueueTask(&CFC::RecvPiEtaXTask, this, CFC_RecvPiEtaX, "CFC_RecvPiEtaX",
                 Task_Run, {CFC_SendPiEtaX});
  pnr->QueueTask(&CFC::ProlongPiEtaXTask, this, CFC_ProlongPiEtaX, "CFC_ProlongPiEtaX",
                 Task_Run, {CFC_RecvPiEtaX});
  pnr->QueueTask(&CFC::BCSPiEtaXTask, this, CFC_BCSPiEtaX, "CFC_BCSPiEtaX",
                 Task_Run, {CFC_ProlongPiEtaX});

  // Adual^ij is computed directly from u_p_x's already ghost-exchanged (P_i, eta)
  // -- X^i is never materialized, so no reconstruct/ghost-exchange quintet is
  // needed for it (see cfc::ComputeADualFromPotentials).
  pnr->QueueTask(&CFC::ComputeADualTask, this, CFC_ComputeADual, "CFC_ComputeADual",
                 Task_Run, {CFC_BCSPiEtaX});

  pnr->QueueTask(&CFC::SolvePsiTask, this, CFC_SolvePsi, "CFC_SolvePsi",
                 Task_Run, {CFC_ComputeADual});
  pnr->QueueTask(&CFC::RescaleSrcTask, this, CFC_RescaleSrc, "CFC_RescaleSrc",
                 Task_Run, {MHD_C2P});
  pnr->QueueTask(&CFC::SolveLapseTask, this, CFC_SolveLapse, "CFC_SolveLapse",
                 Task_Run, {CFC_RescaleSrc});

  // R-072 M1: psi's ghost exchange right after its solve, then the metric reassembled on
  // the whole block from the exchanged psi, before MHD_C2P (dyn_grmhd.cpp waits on
  // CFC_AssemblePsiG). The ghost cells then hold the owner's metric bitwise when the C2P
  // inverts them (C-055: they were one solve stale).
  pnr->QueueTask(&CFC::RestPsiTask, this, CFC_RestPsi, "CFC_RestPsi",
                 Task_Run, {CFC_SolvePsi});
  pnr->QueueTask(&CFC::SendPsiTask, this, CFC_SendPsi, "CFC_SendPsi",
                 Task_Run, {CFC_RestPsi});
  pnr->QueueTask(&CFC::RecvPsiTask, this, CFC_RecvPsi, "CFC_RecvPsi",
                 Task_Run, {CFC_SendPsi});
  pnr->QueueTask(&CFC::ProlongPsiTask, this, CFC_ProlongPsi, "CFC_ProlongPsi",
                 Task_Run, {CFC_RecvPsi});
  pnr->QueueTask(&CFC::BCSPsiTask, this, CFC_BCSPsi, "CFC_BCSPsi",
                 Task_Run, {CFC_ProlongPsi});
  pnr->QueueTask(&CFC::AssemblePsiGhostsTask, this, CFC_AssemblePsiG, "CFC_AssemblePsiG",
                 Task_Run, {CFC_BCSPsi});

  pnr->QueueTask(&CFC::RestAlphaPsiTask, this, CFC_RestAlphaPsi, "CFC_RestAlphaPsi",
                 Task_Run, {CFC_SolveLapse});
  pnr->QueueTask(&CFC::SendAlphaPsiTask, this, CFC_SendAlphaPsi, "CFC_SendAlphaPsi",
                 Task_Run, {CFC_RestAlphaPsi});
  pnr->QueueTask(&CFC::RecvAlphaPsiTask, this, CFC_RecvAlphaPsi, "CFC_RecvAlphaPsi",
                 Task_Run, {CFC_SendAlphaPsi});
  pnr->QueueTask(&CFC::ProlongAlphaPsiTask, this, CFC_ProlongAlphaPsi,
                 "CFC_ProlongAlphaPsi", Task_Run, {CFC_RecvAlphaPsi});
  pnr->QueueTask(&CFC::BCSAlphaPsiTask, this, CFC_BCSAlphaPsi, "CFC_BCSAlphaPsi",
                 Task_Run, {CFC_ProlongAlphaPsi});

  pnr->QueueTask(&CFC::SolveShiftTask, this, CFC_BuildSrcBeta, "CFC_BuildSrcBeta",
                 Task_Run, {CFC_BCSPsi, CFC_BCSAlphaPsi});

  pnr->QueueTask(&CFC::RestPiEtaBetaTask, this, CFC_RestPiEtaBeta, "CFC_RestPiEtaBeta",
                 Task_Run, {CFC_BuildSrcBeta});
  pnr->QueueTask(&CFC::SendPiEtaBetaTask, this, CFC_SendPiEtaBeta, "CFC_SendPiEtaBeta",
                 Task_Run, {CFC_RestPiEtaBeta});
  pnr->QueueTask(&CFC::RecvPiEtaBetaTask, this, CFC_RecvPiEtaBeta, "CFC_RecvPiEtaBeta",
                 Task_Run, {CFC_SendPiEtaBeta});
  pnr->QueueTask(&CFC::ProlongPiEtaBetaTask, this, CFC_ProlongPiEtaBeta,
                 "CFC_ProlongPiEtaBeta", Task_Run, {CFC_RecvPiEtaBeta});
  pnr->QueueTask(&CFC::BCSPiEtaBetaTask, this, CFC_BCSPiEtaBeta, "CFC_BCSPiEtaBeta",
                 Task_Run, {CFC_ProlongPiEtaBeta});

  pnr->QueueTask(&CFC::ReconstructBetaTask, this, CFC_ReconstructBeta,
                 "CFC_ReconstructBeta", Task_Run, {CFC_BCSPiEtaBeta});

  pnr->QueueTask(&CFC::AssembleFinalTask, this, CFC_AssembleFinal, "CFC_AssembleFinal",
                 Task_Run, {CFC_ReconstructBeta});

  // Ghost-exchange padm->u_adm (see pbval_adm's comment in cfc.hpp) -- nothing else
  // this stage depends on these completing, but the shared task-list machinery
  // still awaits every queued task, so u_adm's ghosts are guaranteed valid by the
  // time the next stage (or MHD_Newdt, this same stage) runs.
  pnr->QueueTask(&CFC::RestADMTask, this, CFC_RestADM, "CFC_RestADM",
                 Task_Run, {CFC_AssembleFinal});
  pnr->QueueTask(&CFC::SendADMTask, this, CFC_SendADM, "CFC_SendADM",
                 Task_Run, {CFC_RestADM});
  pnr->QueueTask(&CFC::RecvADMTask, this, CFC_RecvADM, "CFC_RecvADM",
                 Task_Run, {CFC_SendADM});
  pnr->QueueTask(&CFC::ProlongADMTask, this, CFC_ProlongADM, "CFC_ProlongADM",
                 Task_Run, {CFC_RecvADM});
  pnr->QueueTask(&CFC::BCSADMTask, this, CFC_BCSADM, "CFC_BCSADM",
                 Task_Run, {CFC_ProlongADM});

  // Wait for every outstanding send/receive posted this stage before "stagen" is
  // considered done -- mirrors z4c::Z4c_ClearS/Z4c_ClearR (Task_End). Runs after
  // every Task_Run task via the before_stagen/stagen/after_stagen phase separation
  // in driver.cpp, not an explicit dependency edge.
  pnr->QueueTask(&CFC::ClearSendTask, this, CFC_ClearSend, "CFC_ClearSend", Task_End);
  pnr->QueueTask(&CFC::ClearRecvTask, this, CFC_ClearRecv, "CFC_ClearRecv",
                 Task_End, {CFC_ClearSend});
  return;
}

//----------------------------------------------------------------------------------------
// InitRecv/ClearSend/ClearRecv scoped to exactly the fields InitializeMetric()'s two
// phases exercise -- see the comment on these four declarations in cfc.hpp for why the
// all-6-field CFC_InitRecv/ClearSend/ClearRecv tasks can't be reused here.

void CFC::InitRecvXFields() {
  pbval_pietax->InitRecv(4);
}
void CFC::ClearXFields() {
  pbval_pietax->ClearSend();
  pbval_pietax->ClearRecv();
}
void CFC::InitRecvTailFields() {
  pbval_psi->InitRecv(1);
  pbval_alpha_psi->InitRecv(1);
  pbval_pietabeta->InitRecv(4);
  pbval_adm->InitRecv(adm::ADM::nadm);
}
void CFC::ClearTailFields() {
  pbval_psi->ClearSend();
  pbval_alpha_psi->ClearSend();
  pbval_pietabeta->ClearSend();
  pbval_adm->ClearSend();
  pbval_psi->ClearRecv();
  pbval_alpha_psi->ClearRecv();
  pbval_pietabeta->ClearRecv();
  pbval_adm->ClearRecv();
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::RunXPsiSolvePass(Driver *pdriver)
//! \brief one pass of "solve X^i, ghost-exchange, compute Adual^ij/Ahat^2, solve
//! psi" -- extracted from InitializeMetric()'s Picard-loop body so it can be called
//! either repeatedly (the iterative default) or exactly once (<cfc>
//! init_freeze_conserved=true). Does not itself refresh conserved variables from
//! primitives -- callers do that (or don't) depending on which mode is active; see
//! the call sites in InitializeMetric()/ReinitializeMetricForAMR.

void CFC::RunXPsiSolvePass(Driver *pdriver) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  InitRecvXFields();

  // Solve X^i's packed (P_i, eta) source and ghost-exchange it.
  SolveVectorPotential(pdriver, 0);
  RestPiEtaXTask(pdriver, 0);  SendPiEtaXTask(pdriver, 0);
  while (RecvPiEtaXTask(pdriver, 0) != TaskStatus::complete) {}
  ProlongPiEtaXTask(pdriver, 0);
  BCSPiEtaXTask(pdriver, 0);

  ClearXFields();

  // Adual^ij/Ahat^2 computed directly from the just-exchanged (P_i, eta), then
  // solve psi -- this updates padm->adm.g_dd/psi4 (via AssembleConformalMetric)
  // for whatever comes next.
  ComputeADual();
  SolveConformalFactor(pdriver, 0, cfc_init_use_psi5_);
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::InitializeMetric(Driver *pdriver)
//! \brief runs the X^i/psi fixed-point iteration (PrimToCons <-> vector-Poisson/
//! conformal-factor solve) by hand, entirely outside the normal per-stage task graph
//! (a pass through "stagen" would also flux-update/RK-evolve the hydro state, which
//! must not happen during this one-time initialization). Once X^i/psi converge
//! (tracked via psi alone), solves lapse/shift once and does the final padm->u_adm
//! ghost exchange, mirroring the tail of QueueCFCTasks().
//!
//! <cfc> init_freeze_conserved selects a second mode -- RunXPsiSolvePass() called
//! exactly once instead of iterated, see cfc_init_freeze_conserved_'s comment in
//! cfc.hpp.

void CFC::InitializeMetric(Driver *pdriver) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  if (rs_on_) { SetResidualScale(pmy_pack->pmesh->time, 0, false); }
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  if (cfc_init_freeze_conserved_) {
    // One-shot mode: Utilde/S-tilde_i are built exactly once (inside
    // RunXPsiSolvePass, from the conserved state the pgen's own initial metric
    // guess already produced) and held fixed; no outer Picard loop, no
    // convergence check. Falls through into the same tail section the iterative
    // path also uses, unchanged.
    int nmb = pmy_pack->nmb_thispack;
    // Allocation size must match delta_psi's own (AMR-headroom-sized) extent, not
    // nmb_thispack, or the pointwise ops below (which read/write both arrays at
    // the same index) hit a Kokkos extent-mismatch abort once
    // nmb_maxperrank > nmb_thispack. The par_for/MDRangePolicy loop bounds below
    // still correctly use nmb (nmb_thispack) -- only resident blocks need touching.
    DvceArray5D<Real> psi_before("cfc_init_psi_before", delta_psi.extent_int(0), 1,
                                  delta_psi.extent_int(2), delta_psi.extent_int(3),
                                  delta_psi.extent_int(4));
    auto &adm_ = pmy_pack->padm->adm;
    auto &psi_before_seed = psi_before;
    auto &u_psi0_ = u_psi0;
    par_for("cfc_init_psi_before_seed", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      psi_before_seed(m,0,k,j,i) = Kokkos::pow(adm_.psi4(m,k,j,i), 0.25)
                                    - u_psi0_(m,0,k,j,i);
    });

    RunXPsiSolvePass(pdriver);

    if (cfc_init_verbose_) {
      Real dpsi = 0.0;
      auto &psi_ = delta_psi;
      auto &psi_before_ = psi_before;
      Kokkos::parallel_reduce("cfc_init_dpsi_oneshot",
        Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>({0, ks, js, is},
                                                              {nmb, ke+1, je+1, ie+1}),
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i,
                      Real &local_max) {
          local_max = Kokkos::fmax(local_max,
                            Kokkos::fabs(psi_(m,0,k,j,i) - psi_before_(m,0,k,j,i)));
        }, Kokkos::Max<Real>(dpsi));
#if MPI_PARALLEL_ENABLED
      Real global_dpsi = 0.0;
      MPI_Allreduce(&dpsi, &global_dpsi, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
      dpsi = global_dpsi;
#endif
      if (global_variable::my_rank == 0) {
        std::cout << "CFC::InitializeMetric: one-shot frozen-conserved pass complete"
                  << " (<cfc> init_freeze_conserved=true, no outer iteration)."
                  << " max|delta psi - initial guess| = " << dpsi << std::endl;
      }
    }
  } else {
    int nmb = pmy_pack->nmb_thispack;
    // See the identical comment on psi_before above: allocation size must match
    // delta_psi's own AMR-headroom-sized extent, not nmb_thispack.
    DvceArray5D<Real> psi_old("cfc_init_psi_old", delta_psi.extent_int(0), 1,
                               delta_psi.extent_int(2), delta_psi.extent_int(3),
                               delta_psi.extent_int(4));

    bool converged = false;
    for (int iter = 0; iter < cfc_init_iter_max_; ++iter) {
      Kokkos::deep_copy(psi_old, delta_psi);

      pmy_pack->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);
      RunXPsiSolvePass(pdriver);

      // Under-relax the just-solved psi against the previous iteration's value
      // (cfc_init_omega_ < 1) when the plain Picard step is unstable, then re-run
      // AssembleConformalMetric so g_dd/psi4 reflect the relaxed value the next
      // iteration's PrimToConInit sees. Both interior-only, matching
      // AssembleConformalMetric's own no-ghost-dependency implementation.
      // omega=1.0 (the default) skips this, byte-identical to the unrelaxed
      // iteration.
      if (cfc_init_omega_ != 1.0) {
        Real omega = cfc_init_omega_;
        auto &psi_relax = delta_psi;
        auto &psi_old_relax = psi_old;
        par_for("cfc_init_relax_psi", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
          psi_relax(m,0,k,j,i) = (1.0 - omega)*psi_old_relax(m,0,k,j,i) +
                                  omega*psi_relax(m,0,k,j,i);
        });
        AssemblePsiMetric(false);
      }

      Real dpsi = 0.0;
      auto &psi_ = delta_psi;
      auto &psi_old_ = psi_old;
      Kokkos::parallel_reduce("cfc_init_dpsi",
        Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>({0, ks, js, is},
                                                              {nmb, ke+1, je+1, ie+1}),
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i,
                      Real &local_max) {
          // Kokkos::fmax (like IEEE754 fmax) silently ignores a NaN operand and
          // returns the other one, so a fully-NaN psi_ would otherwise leave
          // local_max untouched at the reducer's identity value instead of
          // signaling non-convergence -- map any non-finite diff to a huge but
          // finite value so it always fails the dpsi < cfc_init_tol_ check below.
          Real diff = Kokkos::fabs(psi_(m,0,k,j,i) - psi_old_(m,0,k,j,i));
          if (!Kokkos::isfinite(diff)) {
            diff = std::numeric_limits<Real>::max();
          }
          local_max = Kokkos::fmax(local_max, diff);
        }, Kokkos::Max<Real>(dpsi));
#if MPI_PARALLEL_ENABLED
      Real global_dpsi = 0.0;
      MPI_Allreduce(&dpsi, &global_dpsi, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
      dpsi = global_dpsi;
#endif
      if (cfc_init_verbose_ && global_variable::my_rank == 0) {
        std::cout << "CFC::InitializeMetric iteration " << iter
                  << ": max|delta psi| = " << dpsi << std::endl;
      }
      if (dpsi < cfc_init_tol_) {
        converged = true;
        break;
      }
    }

    if (!converged && global_variable::my_rank == 0) {
      std::cout << "### WARNING in CFC::InitializeMetric" << std::endl
                << "X^i/psi fixed-point iteration did not converge after "
                << cfc_init_iter_max_ << " iterations (<cfc>/init_iter_max)."
                << " Proceeding with the current (non-converged) metric -- increase"
                << " init_iter_max or loosen <cfc>/init_tol if this is unexpected."
                << std::endl;
    }
  }

  // Lapse/shift/final-assembly tail, shared with CFC::ReinitializeMetricForAMR.
  // Reconcile primitives vs. conserved variables in whichever direction matches
  // what was just held fixed: freeze_conserved held Utilde fixed, so recover
  // primitives via ConToPrim; the iterative loop held primitives fixed throughout,
  // so rebuild conserved variables from them one last time via PrimToConInit.
  if (cfc_init_freeze_conserved_) {
    pmy_pack->pdyngr->ConToPrim(pdriver, 0);
  } else {
    pmy_pack->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);
  }

  RunLapseShiftAssemblePass(pdriver);

  // Re-exchange hydro ghost cells and rebuild primitives everywhere (interior +
  // ghost) against the now-final metric: PrimToConInit/ConToPrim above only
  // touch whichever of {u0, w0} was reconciled, and (for u0) only the interior,
  // so the other quantity's ghost cells -- or, for ConToPrim's ghost-inclusive
  // recompute, the metric ghosts it read -- are stale until padm->u_adm's own
  // ghost exchange (inside RunLapseShiftAssemblePass) completes. Reuses the same
  // function already run once before this solve (Driver::Initialize()); safe to
  // call again since it doesn't touch padm->SetADMVariables.
  pdriver->InitBoundaryValuesAndPrimitives(pmy_pack->pmesh);
  // Phase 2 M2: xidot(0) from the star (fresh starts only; restarts restore the state)
  if (frame_ctrl_ && !fc_init_) { InitFrameControl(pdriver); }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::RunLapseShiftAssemblePass(Driver *pdriver)
//! \brief the tail of InitializeMetric()/ReinitializeMetricForAMR: Rescale
//! MatterSources/SolveLapse/SolveShift/AssembleADM. Callers must reconcile
//! primitives vs. conserved variables themselves before calling this -- see this
//! method's comment in cfc.hpp.

void CFC::RunLapseShiftAssemblePass(Driver *pdriver) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  // Lapse and shift don't feed back into X^i/psi's own equations, so they're
  // solved once here rather than iterated -- mirrors QueueCFCTasks()'s tail.
  RescaleMatterSources(pdriver, 0);
  SolveLapse(pdriver, 0);

  InitRecvTailFields();

  RestPsiTask(pdriver, 0);       SendPsiTask(pdriver, 0);
  RestAlphaPsiTask(pdriver, 0);  SendAlphaPsiTask(pdriver, 0);
  while (RecvPsiTask(pdriver, 0) != TaskStatus::complete) {}
  while (RecvAlphaPsiTask(pdriver, 0) != TaskStatus::complete) {}
  ProlongPsiTask(pdriver, 0);
  BCSPsiTask(pdriver, 0);
  ProlongAlphaPsiTask(pdriver, 0);
  BCSAlphaPsiTask(pdriver, 0);

  SolveShift(pdriver, 0);
  RestPiEtaBetaTask(pdriver, 0);  SendPiEtaBetaTask(pdriver, 0);
  while (RecvPiEtaBetaTask(pdriver, 0) != TaskStatus::complete) {}
  ProlongPiEtaBetaTask(pdriver, 0);
  BCSPiEtaBetaTask(pdriver, 0);

  ReconstructShift();
  AssembleADM(StageTime(pdriver, 0));

  RestADMTask(pdriver, 0);  SendADMTask(pdriver, 0);
  while (RecvADMTask(pdriver, 0) != TaskStatus::complete) {}
  ProlongADMTask(pdriver, 0);
  BCSADMTask(pdriver, 0);

  ClearTailFields();
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::ReinitializeMetricForAMR(Driver *pdriver)
//! \brief see this method's comment in cfc.hpp for the full rationale. Forces
//! SolveConformalFactor's/SolveLapse's one-time psi/alpha_psi seeding to fire again,
//! re-seeding from padm->adm.psi4/adm.alpha rather than reusing CFC's own (post-
//! regrid, block-layout-mismatched) delta_psi/delta_alpha_psi as a Newton starting
//! point. Does NOT call InitializeMetric() itself.

void CFC::ReinitializeMetricForAMR(Driver *pdriver) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  if (rs_on_) { SetResidualScale(pmy_pack->pmesh->time, 0, false); }
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nmb = pmy_pack->nmb_thispack;

  // Re-derive the (analytic, time-independent) trumpet background at the post-regrid
  // mesh's own cell positions -- cheap, so just recompute rather than remap; mirrors
  // this function's own treatment of every other CFC-owned field (re-seed/re-solve,
  // never a cross-layout data transfer). No-op when puncture_enabled_ is false.
  if (puncture_enabled_) {
    Real xbh[3];
    BHPosition(pmy_pack->pmesh->time, xbh);
    FillPunctureBackground(pmy_pack, puncture_mass_, xbh, u_psi0, u_alpha0_psi0, beta0_u,
                           a0_dd, a0_sq, grad_ap6_0);
    for (int a = 0; a < 3; ++a) { bg_xbh_[a] = xbh[a]; }
    pmgd_psi->SetPunctureCenter(xbh); pmgd_alpha->SetPunctureCenter(xbh);
  }

  psi_seeded_ = false;
  alpha_psi_seeded_ = false;
  // u_adm's ghosts: the regrid transfers MeshBlock interiors only, so a new (derefined or
  // refined) or received block's ghosts held whatever memory did, and the re-solve below
  // reads adm before its own ADM exchange (C-102 add. 2). Exchange them first.
  pbval_adm->InitRecv(adm::ADM::nadm);
  RestADMTask(pdriver, 0);  SendADMTask(pdriver, 0);
  while (RecvADMTask(pdriver, 0) != TaskStatus::complete) {}
  ProlongADMTask(pdriver, 0);
  BCSADMTask(pdriver, 0);
  pbval_adm->ClearSend();
  pbval_adm->ClearRecv();
  // CFC's own fields are not remapped by the regrid either, and the X^i solve below reads
  // delta_psi (its source) before SolveConformalFactor re-seeds it: seed delta_psi and
  // delta_alpha_psi from adm here, ghosts included (the seed blocks' formulas), and zero
  // the work arrays, so the re-solve reads no other block's data (C-102 add. 2)
  {
    const int n1 = indcs.nx1 + 2*indcs.ng;
    const int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*indcs.ng) : 1;
    const int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*indcs.ng) : 1;
    auto &adm_s = pmy_pack->padm->adm;
    auto &psi_s = delta_psi;
    auto &apsi_s = delta_alpha_psi;
    auto &psi0_s = u_psi0;
    auto &apsi0_s = u_alpha0_psi0;
    par_for("cfc_regrid_seed", DevExeSpace(), 0, nmb-1, 0, n3-1, 0, n2-1, 0, n1-1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      const Real p = Kokkos::pow(adm_s.psi4(m,k,j,i), 0.25);
      psi_s(m,0,k,j,i) = p - psi0_s(m,0,k,j,i);
      apsi_s(m,0,k,j,i) = adm_s.alpha(m,k,j,i)*p - apsi0_s(m,0,k,j,i);
    });
    Kokkos::deep_copy(DevExeSpace(), u_p_x, 0.0);
    Kokkos::deep_copy(DevExeSpace(), u_p_beta, 0.0);
    Kokkos::deep_copy(DevExeSpace(), u_p_src, 0.0);
    Kokkos::deep_copy(DevExeSpace(), u_adual, 0.0);
    Kokkos::deep_copy(DevExeSpace(), u_beta, 0.0);
    Kokkos::deep_copy(DevExeSpace(), u_tilde, 0.0);
  }
  // The vector-potential solvers (X^i, beta) have no seed: their warm start is the MG
  // finest level's previous solution, which a regrid does not remap (kept as it was when
  // a rank's MeshBlock count is unchanged, zeroed otherwise), so it held other blocks'
  // data and depended on memory history (C-102: restarts not bitwise across a regrid).
  // Cold-start them from zero.
  pmgd_pietax->ZeroFinestSolution();
  pmgd_pietabeta->ZeroFinestSolution();
  numrel::ChecksumAfterTask(pmy_pack, "regrid_enter", -1);   // debug (checksum_cycles)
  numrel::ChecksumPerBlock(pmy_pack, "regrid_enter");

  // Snapshot the true initial guess (psi4^0.25-1, the same formula
  // SolveConformalFactor's own one-time psi_seeded_ block uses) directly from
  // adm.psi4, not delta_psi (which is still at whatever stale value it held before
  // this call and is not what the Newton solve will actually be seeded from).
  DvceArray5D<Real> psi_before("cfc_regrid_psi_before", delta_psi.extent_int(0), 1,
                                delta_psi.extent_int(2), delta_psi.extent_int(3),
                                delta_psi.extent_int(4));
  auto &adm_ = pmy_pack->padm->adm;
  auto &psi_before_seed = psi_before;
  auto &u_psi0_ = u_psi0;
  par_for("cfc_regrid_psi_before_seed", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    psi_before_seed(m,0,k,j,i) = Kokkos::pow(adm_.psi4(m,k,j,i), 0.25)
                                  - u_psi0_(m,0,k,j,i);
  });

  RunXPsiSolvePass(pdriver);
  numrel::ChecksumAfterTask(pmy_pack, "regrid_xpsi", -1);
  // Interior-only: RescaleMatterSources (inside RunLapseShiftAssemblePass
  // below) only reads w0 at these same interior indices, so ghost cells don't
  // need primitives yet -- and the metric they'd be read against isn't final
  // until that pass's own padm->u_adm ghost exchange completes anyway (see
  // the ghost-only pass below).
  pmy_pack->pdyngr->ConToPrimBC(is, ie, js, je, ks, ke);
  numrel::ChecksumAfterTask(pmy_pack, "regrid_c2p", -1);
  RunLapseShiftAssemblePass(pdriver);
  numrel::ChecksumAfterTask(pmy_pack, "regrid_lapse", -1);

  // dual_energy_flag_exchange (R-076 item 1): the owners decide the flags on the new
  // interiors (from the interior C2P just done) and the MHD ghost exchange carries them
  // into the ghosts before the ghost-only C2P below, which then reads them instead of
  // recomputing the flag inside each slab
  if (pmy_pack->pmhd->nflag > 0) {
    pmy_pack->pdyngr->UpdateShockFlags();
    pdriver->ExchangeMHDGhosts(pmy_pack->pmesh);
  }

  // Ghost-shell-only con2prim, now that padm->u_adm is fully final and fully
  // ghost-exchanged (RunLapseShiftAssemblePass's own tail, just completed):
  // recovers w0's ghost cells against the correct metric instead of a
  // zero/stale one. Hydro's own u0/b0 ghosts don't need re-exchanging here --
  // that already happened earlier in the same enclosing
  // Driver::InitBoundaryValuesAndPrimitives call this runs from (mirroring
  // z4c::Z4c::ConvertZ4cToADM's placement). 6-slab decomposition of the full
  // ghost shell (mirrors DynGRMHD::ApplyPhysicalBCs's boundary-strip pattern,
  // but full ghost width): x1-faces get full y/z extent (covering corners
  // too); y-/z-faces only need the interior x-range to avoid redundant work.
  {
    auto *pm = pmy_pack->pmesh;
    auto *pdyngr = pmy_pack->pdyngr;
    int &ng = indcs.ng;
    int n1 = indcs.nx1 + 2*ng;
    int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
    int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
    if (pmy_pack->pmhd->nflag > 0) {
      // owner flags: one C2P over the whole block (interior + ghosts) with the
      // exchanged flags, so interior and ghost primitives come from the same flags
      pdyngr->ConToPrimBC(0, n1-1, 0, n2-1, 0, n3-1);
    } else {
    pdyngr->ConToPrimBC(0, is-1, 0, n2-1, 0, n3-1);
    pdyngr->ConToPrimBC(ie+1, n1-1, 0, n2-1, 0, n3-1);
    if (pm->multi_d) {
      pdyngr->ConToPrimBC(is, ie, 0, js-1, 0, n3-1);
      pdyngr->ConToPrimBC(is, ie, je+1, n2-1, 0, n3-1);
    }
    if (pm->three_d) {
      pdyngr->ConToPrimBC(is, ie, js, je, 0, ks-1);
      pdyngr->ConToPrimBC(is, ie, js, je, ke+1, n3-1);
    }
    }
  }

  if (cfc_init_verbose_) {
    Real dpsi = 0.0;
    auto &psi_ = delta_psi;
    auto &psi_before_ = psi_before;
    Kokkos::parallel_reduce("cfc_regrid_dpsi",
      Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>({0, ks, js, is},
                                                            {nmb, ke+1, je+1, ie+1}),
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i,
                    Real &local_max) {
        local_max = Kokkos::fmax(local_max,
                          Kokkos::fabs(psi_(m,0,k,j,i) - psi_before_(m,0,k,j,i)));
      }, Kokkos::Max<Real>(dpsi));
#if MPI_PARALLEL_ENABLED
    Real global_dpsi = 0.0;
    MPI_Allreduce(&dpsi, &global_dpsi, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
    dpsi = global_dpsi;
#endif
    if (global_variable::my_rank == 0) {
      std::cout << "CFC::ReinitializeMetricForAMR: post-regrid single-pass"
                << " re-solve complete (conserved variables held fixed,"
                << " primitives recovered via ConToPrim)."
                << " max|delta psi - initial guess| = " << dpsi << std::endl;
    }
  }
}

//----------------------------------------------------------------------------------------
// Ghost-exchange task-graph entry points: one Rest/Send/Recv/Prolong quartet per field,
// mirroring z4c::Z4c::RestrictU/SendU/RecvU/Prolongate (RestrictCC/ProlongateCC are
// internal no-ops without SMR/AMR; is_z4c=true throughout, reusing z4c's higher-order
// Lagrange restrict/prolong path -- see the constructor's comment above).

TaskStatus CFC::RestPiEtaXTask(Driver *pdriver, int stage) {
  if (!DoSolveThisStage(pdriver, stage)) { return TaskStatus::complete; }
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(u_p_x, coarse_u_pietax, true);
  }
  return TaskStatus::complete;
}
TaskStatus CFC::SendPiEtaXTask(Driver *pdriver, int stage) {
  if (!DoSolveThisStage(pdriver, stage)) { return TaskStatus::complete; }
  return pbval_pietax->PackAndSendCC(u_p_x, coarse_u_pietax);
}
TaskStatus CFC::RecvPiEtaXTask(Driver *pdriver, int stage) {
  if (!DoSolveThisStage(pdriver, stage)) { return TaskStatus::complete; }
  return pbval_pietax->RecvAndUnpackCC(u_p_x, coarse_u_pietax);
}
TaskStatus CFC::ProlongPiEtaXTask(Driver *pdriver, int stage) {
  if (!DoSolveThisStage(pdriver, stage)) { return TaskStatus::complete; }
  if (pmy_pack->pmesh->multilevel) {
    // Step 1: apply physical BCs to the coarse array so the prolongation stencil
    // reads valid data in coarse ghost zones at a physical boundary.
    if (!(pmy_pack->pmesh->strictly_periodic)) {
      MeshBoundaryValues::CFCBCsCoarse(pmy_pack, coarse_u_pietax, 4, 1);
    }
    // FillCoarseInBndryCC is intentionally not called here (mirrors
    // z4c::Z4c::Prolongate). Its is_z4c branch restricts this block's own
    // just-received same-level ghost strip, but that strip already sits at the
    // edge of the fine array's own ghost padding, so the 4th-order Lagrange
    // stencil needs one cell beyond it (index -1) -- out of bounds. The same
    // same-level coarse data is instead delivered via the isame_z4c payload in
    // Send/RecvPiEtaX, computed from the neighbor's own interior data (which
    // has proper margin), so restricting it again here would be unsafe.
    pbval_pietax->ProlongateCC(u_p_x, coarse_u_pietax, true);
  }
  return TaskStatus::complete;
}
//! \fn TaskStatus CFC::BCSPiEtaXTask
//! \brief Apply physical BCs to u_p_x's fine array. Runs after ProlongPiEtaXTask
//! (task-graph dependency {CFC_ProlongPiEtaX}) so corner ghost zones between a
//! coarse neighbor and a physical boundary read valid, already-prolongated data
//! (mirrors z4c::Z4c::ApplyPhysicalBCs).
TaskStatus CFC::BCSPiEtaXTask(Driver *pdriver, int stage) {
  if (!DoSolveThisStage(pdriver, stage)) { return TaskStatus::complete; }
  if (!(pmy_pack->pmesh->strictly_periodic)) {
    MeshBoundaryValues::CFCBCs(pmy_pack, u_p_x, 4, 1);
  }
  return TaskStatus::complete;
}

TaskStatus CFC::RestPsiTask(Driver *pdriver, int stage) {
  if (!DoSolveThisStage(pdriver, stage)) { return TaskStatus::complete; }
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(delta_psi, coarse_psi, true);
  }
  return TaskStatus::complete;
}
TaskStatus CFC::SendPsiTask(Driver *pdriver, int stage) {
  if (!DoSolveThisStage(pdriver, stage)) { return TaskStatus::complete; }
  return pbval_psi->PackAndSendCC(delta_psi, coarse_psi);
}
TaskStatus CFC::RecvPsiTask(Driver *pdriver, int stage) {
  if (!DoSolveThisStage(pdriver, stage)) { return TaskStatus::complete; }
  return pbval_psi->RecvAndUnpackCC(delta_psi, coarse_psi);
}
TaskStatus CFC::ProlongPsiTask(Driver *pdriver, int stage) {
  if (!DoSolveThisStage(pdriver, stage)) { return TaskStatus::complete; }
  if (pmy_pack->pmesh->multilevel) {
    if (!(pmy_pack->pmesh->strictly_periodic)) {
      MeshBoundaryValues::CFCBCsCoarse(pmy_pack, coarse_psi, 1, 1);
    }
    // FillCoarseInBndryCC intentionally skipped here -- see ProlongPiEtaXTask's
    // comment above (mirrors z4c::Z4c::Prolongate).
    pbval_psi->ProlongateCC(delta_psi, coarse_psi, true);
  }
  return TaskStatus::complete;
}
TaskStatus CFC::BCSPsiTask(Driver *pdriver, int stage) {
  if (!DoSolveThisStage(pdriver, stage)) { return TaskStatus::complete; }
  if (!(pmy_pack->pmesh->strictly_periodic)) {
    MeshBoundaryValues::CFCBCs(pmy_pack, delta_psi, 1, 1);
  }
  return TaskStatus::complete;
}

TaskStatus CFC::AssemblePsiGhostsTask(Driver *pdriver, int stage) {
  if (RefreshThisStage(stage)) {   // the held delta_psi (ghosts as last exchanged)
    AssemblePsiMetric(true);
    return TaskStatus::complete;
  }
  if (!DoSolveThisStage(pdriver, stage)) { return TaskStatus::complete; }
  AssemblePsiMetric(true);
  return TaskStatus::complete;
}

TaskStatus CFC::RestAlphaPsiTask(Driver *pdriver, int stage) {
  if (!DoSolveThisStage(pdriver, stage)) { return TaskStatus::complete; }
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(delta_alpha_psi, coarse_alpha_psi, true);
  }
  return TaskStatus::complete;
}
TaskStatus CFC::SendAlphaPsiTask(Driver *pdriver, int stage) {
  if (!DoSolveThisStage(pdriver, stage)) { return TaskStatus::complete; }
  return pbval_alpha_psi->PackAndSendCC(delta_alpha_psi, coarse_alpha_psi);
}
TaskStatus CFC::RecvAlphaPsiTask(Driver *pdriver, int stage) {
  if (!DoSolveThisStage(pdriver, stage)) { return TaskStatus::complete; }
  return pbval_alpha_psi->RecvAndUnpackCC(delta_alpha_psi, coarse_alpha_psi);
}
TaskStatus CFC::ProlongAlphaPsiTask(Driver *pdriver, int stage) {
  if (!DoSolveThisStage(pdriver, stage)) { return TaskStatus::complete; }
  if (pmy_pack->pmesh->multilevel) {
    if (!(pmy_pack->pmesh->strictly_periodic)) {
      MeshBoundaryValues::CFCBCsCoarse(pmy_pack, coarse_alpha_psi, 1, 1);
    }
    // FillCoarseInBndryCC intentionally skipped here -- see ProlongPiEtaXTask's
    // comment above (mirrors z4c::Z4c::Prolongate).
    pbval_alpha_psi->ProlongateCC(delta_alpha_psi, coarse_alpha_psi, true);
  }
  return TaskStatus::complete;
}
TaskStatus CFC::BCSAlphaPsiTask(Driver *pdriver, int stage) {
  if (!DoSolveThisStage(pdriver, stage)) { return TaskStatus::complete; }
  if (!(pmy_pack->pmesh->strictly_periodic)) {
    MeshBoundaryValues::CFCBCs(pmy_pack, delta_alpha_psi, 1, 1);
  }
  return TaskStatus::complete;
}

TaskStatus CFC::RestPiEtaBetaTask(Driver *pdriver, int stage) {
  if (!DoSolveThisStage(pdriver, stage)) { return TaskStatus::complete; }
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(u_p_beta, coarse_u_pietabeta, true);
  }
  return TaskStatus::complete;
}
TaskStatus CFC::SendPiEtaBetaTask(Driver *pdriver, int stage) {
  if (!DoSolveThisStage(pdriver, stage)) { return TaskStatus::complete; }
  return pbval_pietabeta->PackAndSendCC(u_p_beta, coarse_u_pietabeta);
}
TaskStatus CFC::RecvPiEtaBetaTask(Driver *pdriver, int stage) {
  if (!DoSolveThisStage(pdriver, stage)) { return TaskStatus::complete; }
  return pbval_pietabeta->RecvAndUnpackCC(u_p_beta, coarse_u_pietabeta);
}
TaskStatus CFC::ProlongPiEtaBetaTask(Driver *pdriver, int stage) {
  if (!DoSolveThisStage(pdriver, stage)) { return TaskStatus::complete; }
  if (pmy_pack->pmesh->multilevel) {
    if (!(pmy_pack->pmesh->strictly_periodic)) {
      MeshBoundaryValues::CFCBCsCoarse(pmy_pack, coarse_u_pietabeta, 4, 1);
    }
    // FillCoarseInBndryCC intentionally skipped here -- see ProlongPiEtaXTask's
    // comment above (mirrors z4c::Z4c::Prolongate).
    pbval_pietabeta->ProlongateCC(u_p_beta, coarse_u_pietabeta, true);
  }
  return TaskStatus::complete;
}
TaskStatus CFC::BCSPiEtaBetaTask(Driver *pdriver, int stage) {
  if (!DoSolveThisStage(pdriver, stage)) { return TaskStatus::complete; }
  if (!(pmy_pack->pmesh->strictly_periodic)) {
    MeshBoundaryValues::CFCBCs(pmy_pack, u_p_beta, 4, 1);
  }
  return TaskStatus::complete;
}

// u_adm's ghost exchange (queued in QueueCFCTasks() above) follows the same
// Restrict->Send->Recv->Prolong->ApplyPhysicalBCs order as every other CFC field
// and z4c::Z4c; ADMBCs/ADMBCsCoarse (adm_bcs.cpp) are applied below by
// ProlongADMTask/BCSADMTask.
TaskStatus CFC::RestADMTask(Driver *pdriver, int stage) {
  if (!DoSolveThisStage(pdriver, stage) && !RefreshThisStage(stage)) {
    return TaskStatus::complete;
  }
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(pmy_pack->padm->u_adm, pmy_pack->padm->coarse_u_adm,
                                      true);
  }
  return TaskStatus::complete;
}
TaskStatus CFC::SendADMTask(Driver *pdriver, int stage) {
  if (!DoSolveThisStage(pdriver, stage) && !RefreshThisStage(stage)) {
    return TaskStatus::complete;
  }
  return pbval_adm->PackAndSendCC(pmy_pack->padm->u_adm, pmy_pack->padm->coarse_u_adm);
}
TaskStatus CFC::RecvADMTask(Driver *pdriver, int stage) {
  if (!DoSolveThisStage(pdriver, stage) && !RefreshThisStage(stage)) {
    return TaskStatus::complete;
  }
  return pbval_adm->RecvAndUnpackCC(pmy_pack->padm->u_adm, pmy_pack->padm->coarse_u_adm);
}
TaskStatus CFC::ProlongADMTask(Driver *pdriver, int stage) {
  if (!DoSolveThisStage(pdriver, stage) && !RefreshThisStage(stage)) {
    return TaskStatus::complete;
  }
  if (pmy_pack->pmesh->multilevel) {
    // Step 1: apply physical BCs to the coarse array so the prolongation stencil
    // reads valid data in coarse ghost zones at a physical boundary.
    if (!(pmy_pack->pmesh->strictly_periodic)) {
      MeshBoundaryValues::ADMBCsCoarse(pmy_pack, pmy_pack->padm->coarse_u_adm);
    }
    // FillCoarseInBndryCC intentionally skipped here -- see ProlongPiEtaXTask's
    // comment above (mirrors z4c::Z4c::Prolongate).
    pbval_adm->ProlongateCC(pmy_pack->padm->u_adm, pmy_pack->padm->coarse_u_adm, true);
  }
  return TaskStatus::complete;
}
//! \fn TaskStatus CFC::BCSADMTask
//! \brief Apply physical BCs to padm->u_adm's fine array (1/r^n falloff,
//! adm_bcs.cpp): ghost cells at a genuine physical domain edge would otherwise stay
//! frozen at their t=0 pgen value forever, since RecvAndUnpackCC is a no-op there
//! (no neighbor block). Runs after ProlongADMTask so corner ghost zones between a
//! coarse neighbor and a physical boundary read valid, already-prolongated data.
TaskStatus CFC::BCSADMTask(Driver *pdriver, int stage) {
  if (!DoSolveThisStage(pdriver, stage) && !RefreshThisStage(stage)) {
    return TaskStatus::complete;
  }
  if (!(pmy_pack->pmesh->strictly_periodic)) {
    MeshBoundaryValues::ADMBCs(pmy_pack, pmy_pack->padm->u_adm);
  }
  return TaskStatus::complete;
}

// InitRecv/ClearSend/ClearRecv (bvals_tasks.cpp) all unconditionally return
// TaskStatus::complete, so no incomplete-status propagation is needed -- matches
// z4c::Z4c::InitRecv/ClearSend/ClearRecv's own one-line-wrapper shape, just looped
// over CFC's 5 MeshBoundaryValuesCC instances instead of one.
TaskStatus CFC::InitRecvTask(Driver *pdriver, int stage) {
  // solve_every_stages: this stage's decision, latched here (the first CFC task of the
  // stage) so that every task of the stage sees the same one
  if (solve_every_ > 0 && stage >= 1) {
    const int64_t k = static_cast<int64_t>(pmy_pack->pmesh->ncycle)*pdriver->nexp_stages
                      + (stage - 1);
    solve_now_ = force_solve_ || (k % solve_every_ == 0);
    if (solve_now_) { force_solve_ = false; }
    if (!solve_now_) {   // a refresh stage exchanges u_adm only
      pbval_adm->InitRecv(adm::ADM::nadm);
      return TaskStatus::complete;
    }
  }
  if (!DoSolveThisStage(pdriver, stage)) { return TaskStatus::complete; }
  pbval_pietax->InitRecv(4);
  pbval_psi->InitRecv(1);
  pbval_alpha_psi->InitRecv(1);
  pbval_pietabeta->InitRecv(4);
  pbval_adm->InitRecv(adm::ADM::nadm);
  return TaskStatus::complete;
}
TaskStatus CFC::ClearSendTask(Driver *pdriver, int stage) {
  if (RefreshThisStage(stage)) { pbval_adm->ClearSend(); return TaskStatus::complete; }
  if (!DoSolveThisStage(pdriver, stage)) { return TaskStatus::complete; }
  pbval_pietax->ClearSend();
  pbval_psi->ClearSend();
  pbval_alpha_psi->ClearSend();
  pbval_pietabeta->ClearSend();
  pbval_adm->ClearSend();
  return TaskStatus::complete;
}
TaskStatus CFC::ClearRecvTask(Driver *pdriver, int stage) {
  if (RefreshThisStage(stage)) { pbval_adm->ClearRecv(); return TaskStatus::complete; }
  if (!DoSolveThisStage(pdriver, stage)) { return TaskStatus::complete; }
  pbval_pietax->ClearRecv();
  pbval_psi->ClearRecv();
  pbval_alpha_psi->ClearRecv();
  pbval_pietabeta->ClearRecv();
  pbval_adm->ClearRecv();
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
// step 1: X^i's packed (P_i, eta) right-hand side, solved. Runs after MHD_AddSrc,
// once this stage's hydro flux+source update has produced the conserved state
// AssembleVectorSource reads from.

TaskStatus CFC::SolveVecXTask(Driver *pdriver, int stage) {
  if (RefreshThisStage(stage)) {   // the background at this stage's x_BH(t)
    if (puncture_moving_) { RefillMovingPuncture(StageTime(pdriver, stage)); }
    return TaskStatus::complete;
  }
  if (!DoSolveThisStage(pdriver, stage)) { return TaskStatus::complete; }
  // Phase 2 M1: the analytic background follows the puncture; refilled once per solve
  // (the solved delta fields are re-solved right after, so nothing needs re-anchoring)
  if (puncture_moving_) { RefillMovingPuncture(StageTime(pdriver, stage)); }
  // residual_scale: this stage's f, fixed for all of its CFC tasks (the first of them)
  if (rs_on_) {
    SetResidualScale(StageTime(pdriver, stage), stage, stage == pdriver->nexp_stages);
    if (rs_bg_only_) { return TaskStatus::complete; }
  }
  // Matter FIRST, before any elliptic work this cycle. This is the ordering test the
  // first instrumented run (job 8835173) could not make: if the conserved state is
  // already non-finite when the cycle's solves begin, every elliptic NaN downstream is
  // a consequence; if it is clean here and beta comes back NaN, the solver originated
  // it. u0 is at mesh-NGHOST depth.
  CheckFinite("matter.in[pmhd->u0]", pmy_pack->pmhd->u0, 5, stage, indcs_ng_);
  SolveVectorPotential(pdriver, stage);
  // src_nan_check: u_p_src is this solve's SOURCE (rebuilt inside SolveVectorPotential
  // by AssembleVectorSource(false)) and u_p_x its OUTPUT. Checking both separates "the
  // source was already bad" from "the solver made NaN from a finite source" -- the X
  // solve is the control here, since it uses the SAME MGCFCVectorPoisson machinery as
  // the shift solve but a far simpler source (8*pi*S_i, no psi^-7, no derivatives).
  CheckFinite("X.src[u_p_src]", u_p_src, 4, stage, mg_nghost_);
  CheckFinite("X.out[u_p_x]",   u_p_x,   4, stage, mg_nghost_);
  return TaskStatus::complete;
}

// step 2: Adual^ij/Ahat^2, computed directly from u_p_x's (P_i, eta) once its own
// ghost exchange (CFC_BCSPiEtaX) completes -- X^i is never materialized.

TaskStatus CFC::ComputeADualTask(Driver *pdriver, int stage) {
  if (RefreshThisStage(stage)) {   // the held X^i potentials against the new background
    ComputeADual();
    return TaskStatus::complete;
  }
  if (!DoSolveThisStage(pdriver, stage)) { return TaskStatus::complete; }
  if (rs_bg_only_) { return TaskStatus::complete; }   // scaled by f = 0
  ComputeADual();
  // Adual^ij feeds BOTH scalar solves (psi's Ahat^2 term and the lapse's) AND the
  // shift source. If this is finite here but K_dd ends up NaN, the corruption is
  // downstream of Adual, not in it.
  CheckFinite("adual[u_adual]", u_adual, 6, stage, indcs_ng_);
  return TaskStatus::complete;
}

// step 3: psi (nonlinear), then the early psi4/g_dd write MHD_C2P depends on.

TaskStatus CFC::SolvePsiTask(Driver *pdriver, int stage) {
  if (!DoSolveThisStage(pdriver, stage)) { return TaskStatus::complete; }
  SolveConformalFactor(pdriver, stage);
  CheckFinite("psi.out[delta_psi]", delta_psi, 1, stage, indcs_ng_);
  return TaskStatus::complete;
}

// step 4: rebuild S-tilde from the primitives MHD_C2P (this task's dependency) just
// recovered -- no con2prim call here, see RescaleMatterSources.

TaskStatus CFC::RescaleSrcTask(Driver *pdriver, int stage) {
  if (!DoSolveThisStage(pdriver, stage)) { return TaskStatus::complete; }
  if (rs_bg_only_) { return TaskStatus::complete; }   // the lapse solve is skipped
  RescaleMatterSources(pdriver, stage);
  return TaskStatus::complete;
}

// step 5: alpha*psi (nonlinear).

TaskStatus CFC::SolveLapseTask(Driver *pdriver, int stage) {
  if (!DoSolveThisStage(pdriver, stage)) { return TaskStatus::complete; }
  SolveLapse(pdriver, stage);
  CheckFinite("lapse.out[delta_alpha_psi]", delta_alpha_psi, 1, stage, indcs_ng_);
  return TaskStatus::complete;
}

// step 6: beta^i's packed (P_i, eta) right-hand side, solved (not yet reconstructed --
// needs u_p_beta's own ghost exchange first, see CFC_ReconstructBeta). Runs after
// psi/alpha_psi's own ghost exchange (both needed by the eq. 75 source term).

TaskStatus CFC::SolveShiftTask(Driver *pdriver, int stage) {
  if (!DoSolveThisStage(pdriver, stage)) { return TaskStatus::complete; }
  SolveShift(pdriver, stage);
  // THE measurement this instrumentation exists for. Ordered as the data flows, so
  // the first label to report names the earliest corrupted object:
  //   dap6   = Delta(alpha*psi^-6), the regularized quantity BuildShiftSource
  //            differences -- the one place psi^-7 and a psi0*(1/psi) product appear
  //   u_p_src= the assembled shift source (16*pi*alpha*psi^-6*S~_i + 2*Ahat*grad(dap6))
  //   u_p_beta = the solve OUTPUT
  // If dap6/u_p_src are finite and u_p_beta is NaN, the multigrid produced NaN from a
  // finite source -- which is what §13.5 infers but has never measured.
  // ROUND 3 (NANCASCADE §14.8): at the first failing cycle only shift.src reported,
  // with matter/adual/delta_psi/delta_alpha_psi/dap6-interior ALL clean. That leaves
  // exactly these inputs of BuildShiftSource unchecked. ngh=0 scans ghosts too --
  // `Dx` differences dap6 across its GHOST cells, so an interior-only scan can miss
  // the very thing that poisons the interior source.
  //   grad_ap6_0 is used ONLY here, in the shift source, and nowhere else in CFC --
  //   which would explain why the shift source is the sole object to go bad.
  //   psi0/alpha0psi0/a0dual are the analytic BACKGROUNDS, which AccreteExcisedMass
  //   REFILLS whenever the puncture mass changes (item 57). The big accretion drain
  //   landed at cycle 10001, nine cycles before the failure.
  CheckFinite("shift.dap6.FULL[u_alpha_psi6]", u_alpha_psi6, 1, stage, 0);
  CheckFinite("shift.grad_ap6_0[u_grad_ap6_0]", u_grad_ap6_0, 3, stage, 0);
  CheckFinite("bg.a0dual[u_a0dual]", u_a0dual, 6, stage, 0);
  CheckFinite("bg.psi0[u_psi0]", u_psi0, 1, stage, 0);
  CheckFinite("bg.alpha0psi0[u_alpha0_psi0]", u_alpha0_psi0, 1, stage, 0);
  CheckFinite("shift.dap6[u_alpha_psi6]", u_alpha_psi6, 1, stage, indcs_ng_);
  CheckFinite("shift.src[u_p_src]",       u_p_src,      4, stage, mg_nghost_);
  CheckFinite("shift.out[u_p_beta]",      u_p_beta,     4, stage, indcs_ng_);
  return TaskStatus::complete;
}

// CFC_ReconstructBeta: build beta_u from u_p_beta (packed P_i, eta) once its ghost
// exchange has completed.

TaskStatus CFC::ReconstructBetaTask(Driver *pdriver, int stage) {
  if (RefreshThisStage(stage)) {   // the held shift potentials
    ReconstructShift();
    return TaskStatus::complete;
  }
  if (!DoSolveThisStage(pdriver, stage)) { return TaskStatus::complete; }
  if (rs_bg_only_) { return TaskStatus::complete; }   // scaled by f = 0
  ReconstructShift();
  return TaskStatus::complete;
}

// final step: vK_dd/alpha/beta_u -> padm->u_adm.

TaskStatus CFC::AssembleFinalTask(Driver *pdriver, int stage) {
  if (RefreshThisStage(stage)) {
    const Real m0 = puncture_mass_;
    AssembleADM(StageTime(pdriver, stage));
    AccreteExcisedMass(pdriver, stage);   // drains this stage's tally (disjoint slots)
    if (puncture_mass_ != m0) { force_solve_ = true; }
    if (frame_ctrl_ && stage == pdriver->nexp_stages) {
      FrameControlStep(StageTime(pdriver, stage));
    }
    return TaskStatus::complete;
  }
  if (!DoSolveThisStage(pdriver, stage)) {
    // The solved part of adm.beta_u is reused; keep the gauge part current.
    ApplyFrameShiftUpdate(StageTime(pdriver, stage));
    if (frame_ctrl_ && stage == pdriver->nexp_stages) {
      FrameControlStep(StageTime(pdriver, stage));
    }
    return TaskStatus::complete;
  }
  AssembleADM(StageTime(pdriver, stage));
  const Real m0 = puncture_mass_;
  AccreteExcisedMass(pdriver, stage);
  // solve_every_stages > 1: the background was re-anchored, so the next stage solves
  if (solve_every_ > 1 && puncture_mass_ != m0) { force_solve_ = true; }
  if (stage == pdriver->nexp_stages) { ComputeADMMass(); }
  // Phase 2 M2: next step's xidot (before CFC_RestADM packs adm.beta_u's ghosts)
  if (frame_ctrl_ && stage == pdriver->nexp_stages) {
    FrameControlStep(StageTime(pdriver, stage));
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::ComputeADMMass()
//! \brief Surface-integral ADM mass on coordinate spheres, reported at each requested
//! radius.  This is the only INDEPENDENT check that mass transferred to the puncture
//! by AccreteExcisedMass actually appears in the spacetime -- the excision tally and
//! M_BH itself are internal bookkeeping that could be self-consistently wrong.
//!
//! For a conformally-flat slice, psi -> 1 + M/(2r), and
//!     M_ADM = -(1/2pi) * closed_integral( grad(psi) . dS ) = -2 r^2 d<psi>/dr.
//!
//! CRUCIALLY this is applied to the RESIDUAL, not the full psi, and the background's
//! contribution is taken analytically:
//!     M_total = M_BH  +  ( -2 r^2 d<delta_psi>/dr ).
//! Integrating the full psi would throw away the entire point of the puncture split.
//! The background carries almost all the mass, so its truncation error swamps the
//! quantity of interest: at r=16 the full-psi integral is only ~1% accurate, i.e. ~1e-2
//! absolute.  In the TDE case this is fatal -- a 0.7 Msun star against a 1e4 Msun hole
//! is 7e-5 in units of M_BH, so the error on the hole alone would be two orders of
//! magnitude larger than the entire object being measured.  Splitting makes the
//! numerical error scale with the RESIDUAL's size instead, which is the same reason
//! the puncture background is split out of the evolution in the first place.
//!
//! Reported per radius: M_res (the residual surface integral -- the matter's own ADM
//! contribution), M_BH (analytic), M_total = sum, and bg_num.  The last is the same
//! surface integral applied to psi0 alone: it should reproduce M_BH, and is carried
//! only to verify that the background's ADM mass really is the analytic value -- it
//! is NOT used in M_total.  Radius-independence of M_res is the real test; a single
//! number at a single radius proves nothing.
//!
//! The spheres are rebuilt on every call rather than cached: SphericalGrid stores
//! interpolation indices/weights tied to the CURRENT MeshBlock layout, which any AMR
//! regrid invalidates.  Rebuilding is ~nangles*nmb index lookups (92 points at the
//! default nlev=3), i.e. negligible against a V-cycle.
//!
//! psi is reassembled as delta_psi + psi0, since CFC stores the residual.
//!
//! EXTRACTION CENTER (<cfc> adm_mass_center = origin | com).  With "com" the spheres are
//! centered on r_com_mass_, the same centroid SolveConformalFactor() hands to
//! SetRobinCenter(), so the diagnostic and the solver's outer BC share an origin.  Be
//! precise about what that does and does not buy, because the naive expectation is wrong:
//!
//!   - It is NOT a first-order correction to the monopole.  By the mean-value property,
//!     the spherical average of a harmonic field over a shell equals 1/r times the
//!     enclosed mass no matter WHERE inside the shell that mass sits, and no matter where
//!     the shell is centered.  For the exact solution, -2 r^2 d<psi>/dr returns the same
//!     enclosed mass either way.  Recentering does not fix a bias in an otherwise correct
//!     extraction.
//!   - What it changes is (a) WHAT IS ENCLOSED and (b) how much high-l power the finite
//!     quadrature has to cancel.  The nlev geodesic quadrature is exact only up to some
//!     multipole degree; a sphere centered where the residual is most nearly monopolar
//!     leaks least.  Centered on the origin with the star far off-center, the residual's
//!     dominant structure across the sphere is l>=1, and the extraction is measuring
//!     quadrature leakage rather than mass.
//!   - Corollary: with "com", bg_num stops being a check on M_BH.  psi0's puncture sits
//!     at the ORIGIN, so once the sphere is centered elsewhere and no longer encloses it,
//!     the same mean-value property drives bg_num toward zero.  That is correct
//!     behaviour, not a regression -- bg_num only validates M_BH for origin-centered
//!     spheres.

//----------------------------------------------------------------------------------------
//! \fn void CFC::ComputeResidualMassVolume(Real *i_e, Real *i_a)
//! \brief The two volume integrals that the residual surface integral must equal.
//!
//! With K=0 the Hamiltonian constraint is  Delta psi = -2pi psi^5 E - (1/8) psi^-7 Ahat^2
//! (see ConformalFactorRHS, mg_cfc_conformal_factor.cpp:160, whose 2pi/0.125 coefficients
//! fix this normalisation), so integrating against the puncture-split field gives
//!
//!     M_res = int(psi^5 E) dV  +  (1/16pi) int[psi^-7 Ahat^2 - psi0^-7 Ahat0^2] dV
//!             \______i_e______/    \______________i_a______________________________/
//!
//! i_e is exactly the quantity AccreteExcisedMass transfers to the puncture on excision.
//! i_a is NOT transferred: Ahat is *solved* from the momentum constraint sourced by S_i,
//! so deleting matter drops S_i and the next elliptic solve returns a smaller Ahat, with
//! nothing absorbing the difference.  That omission is the energy face of the momentum
//! the pinned, non-spinning puncture already discards by design -- one approximation with
//! two faces, not two independent ones.  Reporting i_e and i_a separately turns it from
//! an acknowledged approximation into a measured one, and (i_e + i_a) vs M_res is an
//! independent volume-vs-surface check on the extraction itself.
//!
//! NUMERICS: near the puncture psi0 -> inf and Ahat0^2 diverges, so forming
//! psi^-7 Ahat^2 - psi0^-7 Ahat0^2 by direct subtraction is catastrophic cancellation of
//! two large nearly-equal numbers.  It is regrouped as
//!     psi^-7 (Ahat^2 - Ahat0^2)  +  Ahat0^2 (psi^-7 - psi0^-7)
//! with the second difference evaluated as psi0^-7 * [(1+r)^-7 - 1], r = delta_psi/psi0,
//! via expm1/log1p so it stays accurate when r << 1 -- which is exactly the regime near
//! the puncture, where the physical answer is that the two cancel.

void CFC::ComputeResidualMassVolume(Real *i_e, Real *i_a) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nmb = pmy_pack->nmb_thispack;
  auto &size = pmy_pack->pmb->mb_size;

  auto &utilde_ = u_tilde;      // psi^6 E  (cons(IEN)+cons(IDN))
  auto &dpsi_   = delta_psi;    // psi - psi0
  auto &psi0_   = u_psi0;       // psi0
  auto &asq_    = a_sq;         // Ahat^2   (total)
  auto &a0sq_   = a0_sq;        // Ahat0^2  (background)

  DvceArray2D<Real> partial("cfc_resmass_partial", nmb, 2);
  par_for("cfc_residual_mass_volume", DevExeSpace(), 0, nmb-1,
  KOKKOS_LAMBDA(const int m) {
    Real dx1 = (size.d_view(m).x1max - size.d_view(m).x1min)/static_cast<Real>(indcs.nx1);
    Real dx2 = (size.d_view(m).x2max - size.d_view(m).x2min)/static_cast<Real>(indcs.nx2);
    Real dx3 = (size.d_view(m).x3max - size.d_view(m).x3min)/static_cast<Real>(indcs.nx3);
    Real vol = dx1*dx2*dx3;
    Real se = 0.0, sa = 0.0;
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          Real p0 = psi0_(m,0,k,j,i);
          Real dp = dpsi_(m,0,k,j,i);
          Real psi = p0 + dp;
          // psi^5 E = (psi^6 E)/psi.  Same weighting AccreteExcisedMass uses.
          if (psi > 0.0) { se += (utilde_(m,0,k,j,i)/psi)*vol; }
          // Extrinsic-curvature term, regrouped to avoid cancelling large numbers.
          if (psi > 0.0 && p0 > 0.0) {
            Real ipsi7  = 1.0/(psi*psi*psi*psi*psi*psi*psi);
            Real ipsi07 = 1.0/(p0*p0*p0*p0*p0*p0*p0);
            Real r = dp/p0;
            Real dinv7 = (r > -1.0) ? ipsi07*Kokkos::expm1(-7.0*Kokkos::log1p(r))
                                    : (ipsi7 - ipsi07);
            sa += (ipsi7*(asq_(m,0,k,j,i) - a0sq_(m,0,k,j,i))
                   + a0sq_(m,0,k,j,i)*dinv7)*vol;
          }
        }
      }
    }
    partial(m,0) = se;
    partial(m,1) = sa;
  });

  auto partial_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), partial);
  Real totals[2] = {};
  for (int m = 0; m < nmb; ++m) {
    totals[0] += partial_h(m,0);
    totals[1] += partial_h(m,1);
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, totals, 2, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  *i_e = totals[0];
  *i_a = totals[1]/(16.0*M_PI);
  return;
}

//----------------------------------------------------------------------------------------

void CFC::ComputeADMMass() {
  if (adm_mass_radii_.empty()) { return; }

  // Volume counterparts, for comparison against the surface integral below.
  Real i_e = 0.0, i_a = 0.0;
  ComputeResidualMassVolume(&i_e, &i_a);
  if (global_variable::my_rank == 0) {
    std::streamsize prec0 = std::cout.precision();
    std::cout << std::scientific << std::setprecision(8)
              << "### CFC residual volume: I_E=" << i_e
              << " I_A=" << i_a
              << " I_sum=" << (i_e + i_a) << std::endl;
    std::cout.unsetf(std::ios_base::floatfield);
    std::cout.precision(prec0);
  }

  // Extraction center.  nullptr => origin-centered, the historical behaviour.
  const Real *ctr = adm_mass_center_com_ ? r_com_mass_ : nullptr;

  // InterpolateToSphere leaves interp_vals ZERO for every angle whose interpolation
  // stencil does not live on this rank, so the host sums below are PARTIAL sums and
  // must be reduced -- same pattern as Z4c's wave extraction (z4c_wave_extr.cpp:115).
  // All radii are accumulated into one buffer so this costs a single collective per
  // call rather than four per radius.  solid_angles is deliberately NOT reduced: it is
  // the full geodesic-grid weight array, identical and complete on every rank, so wtot
  // is already 4*pi locally.
  const size_t nrad = adm_mass_radii_.size();
  std::vector<Real> part(4*nrad, 0.0);   // [4n+0,+1] = <delta_psi> at r-dr, r+dr
  Real wtot = 0.0;                       // [4n+2,+3] = <psi0>      at r-dr, r+dr

  for (size_t n = 0; n < nrad; ++n) {
    const Real rad = adm_mass_radii_[n];
    for (int side = 0; side < 2; ++side) {
      Real r = rad + (side == 0 ? -adm_mass_dr_ : +adm_mass_dr_);
      SphericalGrid sph(pmy_pack, adm_mass_nlev_, r, -1, ctr);

      sph.InterpolateToSphere(1, delta_psi);
      Real s_d = 0.0;
      Real w_here = 0.0;
      for (int ip = 0; ip < sph.nangles; ++ip) {
        Real w = sph.solid_angles.h_view(ip);
        s_d += w*sph.interp_vals.h_view(ip,0);
        w_here += w;
      }
      // Every sphere shares the same nlev, hence the same complete 4*pi weight set;
      // assign rather than accumulate so the value is independent of loop trip count.
      wtot = w_here;
      sph.InterpolateToSphere(1, u_psi0);
      Real s_0 = 0.0;
      for (int ip = 0; ip < sph.nangles; ++ip) {
        s_0 += sph.solid_angles.h_view(ip)*sph.interp_vals.h_view(ip,0);
      }
      part[4*n + side]     = s_d;
      part[4*n + 2 + side] = s_0;
    }
  }

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, part.data(), static_cast<int>(part.size()),
                MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  for (size_t n = 0; n < nrad; ++n) {
    const Real rad = adm_mass_radii_[n];
    // Solid-angle averages of the RESIDUAL and of the BACKGROUND, kept separate.
    Real dpsi_at[2] = {part[4*n + 0]/wtot, part[4*n + 1]/wtot};
    Real psi0_at[2] = {part[4*n + 2]/wtot, part[4*n + 3]/wtot};

    // M = -(r^2/2pi) * closed_integral(grad psi . dS) = -2 r^2 d<psi>/dr.
    Real m_res    = -2.0*rad*rad*(dpsi_at[1] - dpsi_at[0])/(2.0*adm_mass_dr_);
    Real m_bg_num = -2.0*rad*rad*(psi0_at[1] - psi0_at[0])/(2.0*adm_mass_dr_);
    Real m_total  = puncture_mass_ + m_res;

    if (global_variable::my_rank == 0) {
      std::streamsize prec0 = std::cout.precision();
      std::cout << std::scientific << std::setprecision(8)
                << "### CFC ADM mass: r=" << rad
                << " M_res=" << m_res
                << " M_BH=" << puncture_mass_
                << " M_total=" << m_total
                << " M_accreted=" << accreted_mass_
                << " [bg_num=" << m_bg_num;
      if (adm_mass_center_com_) {
        std::cout << " ctr=(" << r_com_mass_[0] << "," << r_com_mass_[1] << ","
                  << r_com_mass_[2] << ")";
      }
      std::cout << "]" << std::endl;
      std::cout.unsetf(std::ios_base::floatfield);
      std::cout.precision(prec0);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::AccreteExcisedMass(Driver *pdriver, int stage)
//! \brief Drain the excision tally once per cycle and add it to the running totals.
//!
//! CFC's metric comes from elliptic constraints sourced by the INSTANTANEOUS matter
//! distribution, so it has no memory: excising matter also removes its gravity in the
//! same step, and the black hole would lose the mass it just swallowed.  (Z4c is not
//! affected -- its hyperbolic evolution keeps that information in the metric.)  The
//! energy banked here is therefore destined for puncture_mass_.
//!
//! RK-STAGE WEIGHTING (item 60).  The excision reset fires on EVERY RK stage, and
//! naively summing those removals over-counts by ~2x.  The reason is the SSP-RK3
//! register: each stage forms u0 = gam0*u0 + gam1*u1 + beta*dt*L, and u1 still holds
//! the START-of-cycle state, so every stage RE-INJECTS a fraction of the matter that
//! the previous stage already excised, and the next reset removes it again.
//!
//! What is physically removed over the cycle is the influx F, once.  A removal applied
//! at the end of stage s reaches the final state with weight
//!     w_s = prod_{j>s} gam0_j,     w_last = 1,
//! i.e. (1/6, 2/3, 1) for rk3 (gam0 = 0, 1/4, 2/3; driver.cpp:120-129).  Draining each
//! stage separately and combining with those weights returns exactly F: with per-stage
//! removals (F, F/4, 2F/3) the weighted sum is F/6 + F/6 + 2F/3 = F, where the naive
//! sum gives 1.92F.  That predicted 1.92x is what the t=145->230 TDE run measured as
//! 1.95x (M_accreted 1.9176e-4 against a peak residual mass of 9.83e-5).
//!
//! Draining every stage costs one extra MPI_Allreduce per stage on a 5-element buffer,
//! against the elliptic solves in the same stage -- negligible.

void CFC::AccreteExcisedMass(Driver *pdriver, int stage) {
  if (!pmy_pack->pcoord->coord_data.bh_excise) { return; }
  const int ns = pdriver->nexp_stages;
  if (stage < 1 || stage > ns) { return; }

  // Drain THIS stage's removals into its own slot.  DrainExcisedTally zeroes the
  // device tally, so the slots are disjoint by construction.
  Real tot[5];
  pmy_pack->pcoord->DrainExcisedTally(tot);
  for (int c = 0; c < 5; ++c) { stage_tally_[stage-1][c] = tot[c]; }

  if (stage != ns) { return; }

  // Combine the stages with their propagation weights to the final state.
  Real w[4];
  w[ns-1] = 1.0;
  for (int s = ns-2; s >= 0; --s) { w[s] = pdriver->gam0[s+1]*w[s+1]; }
  for (int c = 0; c < 5; ++c) {
    tot[c] = 0.0;
    for (int s = 0; s < ns; ++s) { tot[c] += w[s]*stage_tally_[s][c]; }
  }

  accreted_mass_  += tot[0];
  accreted_mom_[0] += tot[1];
  accreted_mom_[1] += tot[2];
  accreted_mom_[2] += tot[3];

  // Hand the swallowed energy to the puncture.  Gated on puncture_enabled_ (there is
  // no hole to grow otherwise) and on accrete_to_puncture_, so the tally can be run as
  // a pure diagnostic for comparison.
  if (puncture_enabled_ && accrete_to_puncture_ && tot[0] != 0.0) {
    puncture_mass_ += tot[0];

    // All three copies of M_BH must move together.  The two multigrid drivers read
    // their own copies from <cfc> at construction and feed them to
    // FillPunctureCoefficients on every Solve(); if only one were updated, psi's
    // coarse levels would use the new mass while alpha's used the old.  Updated here,
    // at the END of the last stage, so the change is in place before any of the next
    // cycle's solves rather than part-way through this one.
    if (pmgd_psi != nullptr) { pmgd_psi->SetPunctureMass(puncture_mass_); }
    if (pmgd_alpha != nullptr) { pmgd_alpha->SetPunctureMass(puncture_mass_); }

    // Re-derive the analytic background at the new mass.  Cheap: one ghost-inclusive
    // par_for with a root-find per cell, against the ~4.5 finest-grid-equivalent
    // sweeps the per-level coefficient fills already do every stage.
    Real xbh[3];
    BHPosition(pmy_pack->pmesh->time, xbh);
    FillPunctureBackground(pmy_pack, puncture_mass_, xbh, u_psi0, u_alpha0_psi0, beta0_u,
                           a0_dd, a0_sq, grad_ap6_0);
    for (int a = 0; a < 3; ++a) { bg_xbh_[a] = xbh[a]; }
    pmgd_psi->SetPunctureCenter(xbh); pmgd_alpha->SetPunctureCenter(xbh);

    // Re-anchor the residuals.  delta_psi stores psi - psi0, so raising psi0 without
    // this would make the PHYSICAL psi jump by delta(psi0) everywhere at once.
    // Clearing the seed flags makes the next solve re-derive the residual from the
    // unchanged ADM fields minus the NEW background (cfc.cpp's seed blocks), which
    // keeps the physical metric continuous and lets the mass change enter through the
    // elliptic solve instead of as a step.  It also avoids the multigrid warm start
    // being a guess for the old background, which would otherwise spike V-cycles.
    psi_seeded_ = false;
    alpha_psi_seeded_ = false;
  }

  if (cfc_init_verbose_ && global_variable::my_rank == 0 && tot[0] != 0.0) {
    // NB high precision: M_BH changes are many orders below its O(1) value, so at
    // default stream precision the growth is invisible.
    std::streamsize prec0 = std::cout.precision();
    std::cout << std::scientific << std::setprecision(10)
              << "### CFC accretion: dM_adm=" << tot[0]
              << " dM_raw=" << tot[4]
              << " <1/psi>=" << (tot[4] != 0.0 ? tot[0]/tot[4] : 0.0)
              << " M_accreted=" << accreted_mass_
              << " M_BH-M0=" << (puncture_mass_ - puncture_mass_init_)
              << " dPx=" << tot[1] << std::endl;
    std::cout.unsetf(std::ios_base::floatfield);
    std::cout.precision(prec0);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::ComputeVectorCentroidX()
//! \brief physical-S-squared-weighted centroid of X^i's own source (Sec 3.10 item 2/
//! Sec 5 Phase A item 8b), for the mg_multipole outer BC's recentering. Mirrors
//! ComputeMassCentroid()'s reduction shape, but weights by the PHYSICAL S_aS^a --
//! cons(IM1+a) is the raw densitized S-tilde_a = psi^6*S_a, so squaring it directly
//! would introduce a spurious psi^12 factor that biases the centroid toward the
//! puncture (where psi blows up); dividing by psi^12 first undoes that. Uses whatever
//! psi this solver currently holds (last stage's converged delta_psi_/u_psi0_, not
//! this stage's not-yet-solved value -- X^i solves before psi does) -- a one-stage
//! lag that only affects where the BC is centered, not any physical RHS.

void CFC::ComputeVectorCentroidX() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nmb = pmy_pack->nmb_thispack;
  auto &size = pmy_pack->pmb->mb_size;
  auto &cons = pmy_pack->pmhd->u0;
  auto &delta_psi_ = delta_psi;
  auto &u_psi0_ = u_psi0;

  DvceArray2D<Real> partial("cfc_com_x_partial", nmb, 4);
  par_for("cfc_vec_centroid_x", DevExeSpace(), 0, nmb-1,
  KOKKOS_LAMBDA(const int m) {
    Real dx1 = (size.d_view(m).x1max - size.d_view(m).x1min) / static_cast<Real>(indcs.nx1);
    Real dx2 = (size.d_view(m).x2max - size.d_view(m).x2min) / static_cast<Real>(indcs.nx2);
    Real dx3 = (size.d_view(m).x3max - size.d_view(m).x3min) / static_cast<Real>(indcs.nx3);
    Real vol = dx1*dx2*dx3;
    Real m0 = 0.0, m1 = 0.0, m2 = 0.0, m3 = 0.0;
    for (int k = ks; k <= ke; ++k) {
      Real z = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
      for (int j = js; j <= je; ++j) {
        Real y = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
        for (int i = is; i <= ie; ++i) {
          Real x = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
          Real psi_val = delta_psi_(m,0,k,j,i) + u_psi0_(m,0,k,j,i);
          Real psi2 = psi_val*psi_val;
          Real psi6 = psi2*psi2*psi2;
          Real psi12 = psi6*psi6;
          Real s2 = 0.0;
          for (int a = 0; a < 3; ++a) {
            Real sa = cons(m,IM1+a,k,j,i);
            s2 += sa*sa;
          }
          Real w = (s2/psi12)*vol;
          m0 += w;
          m1 += w*x;
          m2 += w*y;
          m3 += w*z;
        }
      }
    }
    partial(m,0) = m0;
    partial(m,1) = m1;
    partial(m,2) = m2;
    partial(m,3) = m3;
  });

  auto partial_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), partial);
  Real totals[4] = {};
  for (int m = 0; m < nmb; ++m) {
    for (int c = 0; c < 4; ++c) {
      totals[c] += partial_h(m,c);
    }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, totals, 4, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  Real im = 1.0/(totals[0] + 1.0e-30);
  r_com_X_[0] = im*totals[1];
  r_com_X_[1] = im*totals[2];
  r_com_X_[2] = im*totals[3];

  ApplyReflectingBoundaryCorrection(r_com_X_);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::ComputeVectorCentroidBeta()
//! \brief p_src_-squared-weighted centroid of beta^i's own (fully-assembled) source
//! (Sec 3.10 item 2/Sec 5 Phase A item 8b). No psi^12 correction needed here, unlike
//! ComputeVectorCentroidX() -- the matter term's alpha*psi^-6 prefactor already
//! exactly cancels S-tilde's psi^6 (confirmed: ap6*s_tilde_d = (alpha*psi^-6)*
//! (psi^6*S_a) = alpha*S_a), so p_src_ as assembled is already the physical source,
//! safe to square directly. Must be called AFTER BuildShiftSource() has added its
//! Ahat-gradient term onto p_src_ -- called from AssembleVectorSource's for_shift
//! branch, not from SolveShift(), specifically so it sees the complete source.
//! p_src_/u_p_src_ live at this solver's own (shallower) mg_nghost_ depth, unlike
//! ComputeMassCentroid()/ComputeVectorCentroidX() which read mesh-NGHOST-depth
//! arrays directly -- same (k,j,i)->(mk,mj,mi) translation AssembleVectorSource uses.

void CFC::ComputeVectorCentroidBeta() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nmb = pmy_pack->nmb_thispack;
  auto &size = pmy_pack->pmb->mb_size;
  auto p_src_ = p_src;
  int mg_nghost = mg_nghost_;

  DvceArray2D<Real> partial("cfc_com_beta_partial", nmb, 4);
  par_for("cfc_vec_centroid_beta", DevExeSpace(), 0, nmb-1,
  KOKKOS_LAMBDA(const int m) {
    Real dx1 = (size.d_view(m).x1max - size.d_view(m).x1min) / static_cast<Real>(indcs.nx1);
    Real dx2 = (size.d_view(m).x2max - size.d_view(m).x2min) / static_cast<Real>(indcs.nx2);
    Real dx3 = (size.d_view(m).x3max - size.d_view(m).x3min) / static_cast<Real>(indcs.nx3);
    Real vol = dx1*dx2*dx3;
    Real m0 = 0.0, m1 = 0.0, m2 = 0.0, m3 = 0.0;
    for (int k = ks; k <= ke; ++k) {
      Real z = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
      for (int j = js; j <= je; ++j) {
        Real y = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
        for (int i = is; i <= ie; ++i) {
          Real x = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
          const int mk = k - ks + mg_nghost, mj = j - js + mg_nghost, mi = i - is + mg_nghost;
          Real p2 = 0.0;
          for (int a = 0; a < 3; ++a) {
            Real pa = p_src_(m,a,mk,mj,mi);
            p2 += pa*pa;
          }
          Real w = p2*vol;
          m0 += w;
          m1 += w*x;
          m2 += w*y;
          m3 += w*z;
        }
      }
    }
    partial(m,0) = m0;
    partial(m,1) = m1;
    partial(m,2) = m2;
    partial(m,3) = m3;
  });

  auto partial_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), partial);
  Real totals[4] = {};
  for (int m = 0; m < nmb; ++m) {
    for (int c = 0; c < 4; ++c) {
      totals[c] += partial_h(m,c);
    }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, totals, 4, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  Real im = 1.0/(totals[0] + 1.0e-30);
  r_com_beta_[0] = im*totals[1];
  r_com_beta_[1] = im*totals[2];
  r_com_beta_[2] = im*totals[3];

  ApplyReflectingBoundaryCorrection(r_com_beta_);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::AssembleVectorSource(bool for_shift)
//! \brief Gmunu eq. 72 (for_shift=false) / eq. 75 (for_shift=true) right-hand sides,
//! plus Shibata eq. 3.11's eta source (packed at channel 3 of u_p_src), built from
//! the same S_i either way.

void CFC::AssembleVectorSource(bool for_shift) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nmb = pmy_pack->nmb_thispack;

  auto &adm = pmy_pack->padm->adm;
  auto s_tilde_d_ = s_tilde_d;
  auto p_src_ = p_src;
  auto &u_p_src_ = u_p_src;
  // p_src_/u_p_src_ are allocated at this solver's own (shallower) mg_nghost_
  // depth, not mesh-NGHOST depth like everything else this function touches --
  // every write below needs the loop's mesh-indexed (k,j,i) translated into their
  // own index space first. eta's source is channel 3 of u_p_src_.
  int mg_nghost = mg_nghost_;

  if (!for_shift) {
    // Step 1 (Gmunu eq. 72): U/S_i built directly from the evolved conserved state
    // (pmy_pack->pmhd->u0), mirroring dyn_grmhd.cpp's DynGRMHD::SetTmunu -- no
    // primitives needed. cons is already densitized by sqrt(detg), and
    // psi^6 == sqrt(detg) exactly for the conformally-flat ansatz (g_dd =
    // psi^4*delta_ij, always) -- so U-tilde = psi^6*U collapses to cons itself
    // (and likewise S-tilde_i), with no detg/psi^6 computation needed here.
    auto &cons = pmy_pack->pmhd->u0;
    auto &u_tilde_ = u_tilde;
    auto &u_raw_ = u_raw;
    Real rcx = r_com_X_[0], rcy = r_com_X_[1], rcz = r_com_X_[2];
    par_for("cfc_assemble_vecX_src", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      const int mk = k - ks + mg_nghost, mj = j - js + mg_nghost, mi = i - is + mg_nghost;
      u_tilde_(m,0,k,j,i) = cons(m,IEN,k,j,i) + cons(m,IDN,k,j,i);

      // U_raw = Utilde/sqrt(detg): the undensitized ADM energy density, using the
      // same g_dd that built cons -- feeds the alternate psi^5 Newton formulation
      // (InitializeMetric()-only, see mg_cfc_conformal_factor.cpp). Cheap, so
      // computed unconditionally.
      Real gxx = adm.g_dd(m,0,0,k,j,i), gxy = adm.g_dd(m,0,1,k,j,i);
      Real gxz = adm.g_dd(m,0,2,k,j,i), gyy = adm.g_dd(m,1,1,k,j,i);
      Real gyz = adm.g_dd(m,1,2,k,j,i), gzz = adm.g_dd(m,2,2,k,j,i);
      Real detg = adm::SpatialDet(gxx, gxy, gxz, gyy, gyz, gzz);
      u_raw_(m,0,k,j,i) = u_tilde_(m,0,k,j,i) / Kokkos::sqrt(detg);

      Real &x1min = size.d_view(m).x1min; Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
      Real &x2min = size.d_view(m).x2min; Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
      Real &x3min = size.d_view(m).x3min; Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
      // Recentered on r_com_X_ (Sec 3.10 items 2-3/Sec 5 Phase A item 8b-8c) --
      // (0,0,0) whenever puncture_enabled_ is false, an exact no-op.
      Real xk[3] = {x1v - rcx, x2v - rcy, x3v - rcz};

      Real eta_val = 0.0;
      for (int a = 0; a < 3; ++a) {
        Real s_a = cons(m,IM1+a,k,j,i);
        s_tilde_d_(m,a,k,j,i) = s_a;
        Real p_a = 8.0*M_PI*s_a;  // Gmunu eq. 72 rhs, f^ij = delta^ij (Cartesian)
        p_src_(m,a,mk,mj,mi) = p_a;
        eta_val -= p_a*xk[a];  // Shibata eq. 3.11: eta_src = -S_i x^i
      }
      u_p_src_(m,3,mk,mj,mi) = eta_val;
    });
  } else {
    // Step 6 (Gmunu eq. 75): pointwise part (16*pi*alpha*psi^-6*S-tilde_i) first,
    // using this stage's newly-solved psi/alpha_psi and the s_tilde_d step 1 already
    // built.
    auto &delta_psi_ = delta_psi;
    auto &delta_alpha_psi_ = delta_alpha_psi;
    auto &u_psi0_ = u_psi0;
    auto &u_alpha0_psi0_ = u_alpha0_psi0;
    par_for("cfc_assemble_shift_src", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      const int mk = k - ks + mg_nghost, mj = j - js + mg_nghost, mi = i - is + mg_nghost;
      Real psi_val = delta_psi_(m,0,k,j,i) + u_psi0_(m,0,k,j,i);
      Real psi2 = psi_val*psi_val;
      Real psi7 = psi2*psi2*psi2*psi_val;
      // alpha*psi^-6 = (alpha_psi/psi)*psi^-6 = alpha_psi*psi^-7.
      Real ap6 = (delta_alpha_psi_(m,0,k,j,i) + u_alpha0_psi0_(m,0,k,j,i))/psi7;
      for (int a = 0; a < 3; ++a) {
        p_src_(m,a,mk,mj,mi) = 16.0*M_PI*ap6*s_tilde_d_(m,a,k,j,i);
      }
    });
    // Derivative part (Sec 3.9's regularized cross-terms), added onto p_src in place.
    BuildShiftSource(pmy_pack, delta_psi, delta_alpha_psi, u_psi0, u_alpha0_psi0, a_dd,
                     a0_dd, grad_ap6_0, p_src, mg_nghost, u_alpha_psi6);

    // r_com_beta_ (Sec 3.10 item 2/Sec 5 Phase A item 8b) needs the now-complete
    // p_src_ -- must run after BuildShiftSource(), before this branch's eta-source
    // loop and before SolveShift()'s SetMultipoleOrigin() call.
    if (puncture_enabled_) {
      ComputeVectorCentroidBeta();
    }
    Real rcx = r_com_beta_[0], rcy = r_com_beta_[1], rcz = r_com_beta_[2];

    // eta_src = -S_i x^i, same formula as step 1, using the now-complete p_src.
    par_for("cfc_assemble_shift_eta_src", DevExeSpace(), 0, nmb-1, ks, ke, js, je,
            is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      const int mk = k - ks + mg_nghost, mj = j - js + mg_nghost, mi = i - is + mg_nghost;
      Real &x1min = size.d_view(m).x1min; Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
      Real &x2min = size.d_view(m).x2min; Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
      Real &x3min = size.d_view(m).x3min; Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
      // Recentered on r_com_beta_ (Sec 3.10 items 2-3/Sec 5 Phase A item 8b-8c) --
      // (0,0,0) whenever puncture_enabled_ is false, an exact no-op.
      Real xk[3] = {x1v - rcx, x2v - rcy, x3v - rcz};
      Real eta_val = 0.0;
      for (int a = 0; a < 3; ++a) {
        eta_val -= p_src_(m,a,mk,mj,mi)*xk[a];
      }
      u_p_src_(m,3,mk,mj,mi) = eta_val;
    });
  }
  return;
}

//----------------------------------------------------------------------------------------

void CFC::SolveVectorPotential(Driver *pdriver, int stage) {
  if (rs_bg_only_) { return; }   // residual_scale f = 0: no solve
  // r_com_X_ (Sec 3.10 items 2-3/Sec 5 Phase A item 8b-8c) only needs u0/psi, both
  // already available -- must run before AssembleVectorSource(false), whose
  // eta-source loop reads r_com_X_.
  if (puncture_enabled_) {
    ComputeVectorCentroidX();
    pmgd_pietax->SetMultipoleOrigin(r_com_X_[0], r_com_X_[1], r_com_X_[2]);
  }
  AssembleVectorSource(/*for_shift=*/false);

  pmgd_pietax->LoadPoissonSource(u_p_src);
  {
    MGScopedTimer pt(MultigridDriver::phase_on, MultigridDriver::phase_solve_sec);
    pmgd_pietax->Solve(pdriver, stage);
  }
  pmgd_pietax->RetrieveSolution(u_p_x);
  return;
}

void CFC::ComputeADual() {
  cfc::ComputeADualFromPotentials(pmy_pack, p_x, u_p_x, a0_dd, a_dd, 3, r_com_X_);

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nmb = pmy_pack->nmb_thispack;
  auto a_dd_ = a_dd;
  auto &a_sq_ = a_sq;
  par_for("cfc_ahat_sq", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Ahat^2 = f_ik f_jl Adual^kl Adual^ij = sum_{a,b} (Adual^ab)^2 (flat metric).
    Real sq = 0.0;
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        Real aab = a_dd_(m,a,b,k,j,i);
        sq += aab*aab;
      }
    }
    a_sq_(m,0,k,j,i) = sq;
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::ComputeMassCentroid()
//! \brief mass-weighted centroid of pmhd->u0(IDN) (Sec 3.10 item 1), for the mg_robin
//! outer BC's recentering (Sec 5 Phase A item 8a). Mirrors MultigridDriver::
//! CalculateCenterOfMass()'s reduction shape (per-MeshBlock partial sums -> host mirror
//! -> MPI_Allreduce), but reads u0(IDN) directly rather than src_ -- psi/alpha_psi's
//! matter lives in coeff_ (Finding B), so their src_ is always zero and
//! CalculateCenterOfMass() would divide by zero if reused as-is here. u0(IDN) is
//! already the fully densitized conserved rest-mass density (sqrt(g)*rho*W) -- no
//! extra sqrt(g)/psi^6 factor is applied, which would double-count it.

void CFC::ComputeMassCentroid() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nmb = pmy_pack->nmb_thispack;
  auto &size = pmy_pack->pmb->mb_size;
  auto &cons = pmy_pack->pmhd->u0;

  DvceArray2D<Real> partial("cfc_com_partial", nmb, 4);
  par_for("cfc_mass_centroid", DevExeSpace(), 0, nmb-1,
  KOKKOS_LAMBDA(const int m) {
    Real dx1 = (size.d_view(m).x1max - size.d_view(m).x1min) / static_cast<Real>(indcs.nx1);
    Real dx2 = (size.d_view(m).x2max - size.d_view(m).x2min) / static_cast<Real>(indcs.nx2);
    Real dx3 = (size.d_view(m).x3max - size.d_view(m).x3min) / static_cast<Real>(indcs.nx3);
    Real vol = dx1*dx2*dx3;
    Real m0 = 0.0, m1 = 0.0, m2 = 0.0, m3 = 0.0;
    for (int k = ks; k <= ke; ++k) {
      Real z = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
      for (int j = js; j <= je; ++j) {
        Real y = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
        for (int i = is; i <= ie; ++i) {
          Real x = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
          Real s = cons(m,IDN,k,j,i)*vol;
          m0 += s;
          m1 += s*x;
          m2 += s*y;
          m3 += s*z;
        }
      }
    }
    partial(m,0) = m0;
    partial(m,1) = m1;
    partial(m,2) = m2;
    partial(m,3) = m3;
  });

  auto partial_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), partial);
  Real totals[4] = {};
  for (int m = 0; m < nmb; ++m) {
    for (int c = 0; c < 4; ++c) {
      totals[c] += partial_h(m,c);
    }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, totals, 4, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  Real im = 1.0/(totals[0] + 1.0e-30);
  r_com_mass_[0] = im*totals[1];
  r_com_mass_[1] = im*totals[2];
  r_com_mass_[2] = im*totals[3];

  ApplyReflectingBoundaryCorrection(r_com_mass_);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::ApplyReflectingBoundaryCorrection(Real center[3])
//! \brief shared override for any mass/momentum-like centroid reduction (Sec 3.10
//! items 1-2/Sec 5 Phase A items 8a-8b), factored out of ComputeMassCentroid()'s
//! original inline logic -- identical correction applies to ComputeVectorCentroidX()/
//! ComputeVectorCentroidBeta() too, since all three integrate over the same (possibly
//! octant-symmetric) stored domain. A reflecting mesh boundary means the true
//! (unreflected) matter distribution is symmetric about that face's own coordinate --
//! the true centroid component along that axis is exactly the reflecting plane's
//! location, not whatever the raw reduction gives (which only ever sees the stored
//! octant/quadrant of a symmetric star, and is spuriously off-center as a result).
//! Every existing CFC test uses reflecting BCs this way (octant-symmetric TOV stars);
//! without this correction, enabling recentering on one of those fixtures would
//! spuriously shift the BC away from the star's/puncture's shared true center at the
//! domain corner, degrading convergence for no physical reason (confirmed
//! empirically: the near-vacuum puncture smoke test needed more V-cycle iterations
//! without this fix).

void CFC::ApplyReflectingBoundaryCorrection(Real center[3]) {
  auto &mesh_bcs = pmy_pack->pmesh->mesh_bcs;
  auto &msize = pmy_pack->pmesh->mesh_size;
  if (mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::reflect) {
    center[0] = msize.x1min;
  } else if (mesh_bcs[BoundaryFace::outer_x1] == BoundaryFlag::reflect) {
    center[0] = msize.x1max;
  }
  if (mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::reflect) {
    center[1] = msize.x2min;
  } else if (mesh_bcs[BoundaryFace::outer_x2] == BoundaryFlag::reflect) {
    center[1] = msize.x2max;
  }
  if (mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::reflect) {
    center[2] = msize.x3min;
  } else if (mesh_bcs[BoundaryFace::outer_x3] == BoundaryFlag::reflect) {
    center[2] = msize.x3max;
  }
}

void CFC::SolveConformalFactor(Driver *pdriver, int stage, bool use_psi5_source) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  // residual_scale f = 0: no solve, the background metric (psi0 + 0*delta_psi)
  if (rs_bg_only_) {
    AssemblePsiMetric(false);
    return;
  }

  // On the very first call ever, seed the V-cycle's initial guess from the problem
  // generator's own ADM data (padm->adm.psi4) instead of a cold Kokkos-zero start.
  if (!psi_seeded_) {
    psi_seeded_ = true;
    int &is = indcs.is; int &ie = indcs.ie;
    int &js = indcs.js; int &je = indcs.je;
    int &ks = indcs.ks; int &ke = indcs.ke;
    int nmb = pmy_pack->nmb_thispack;
    auto &adm = pmy_pack->padm->adm;
    auto &psi_ = delta_psi;
    auto &u_psi0_ = u_psi0;
    par_for("cfc_seed_psi", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      psi_(m,0,k,j,i) = Kokkos::pow(adm.psi4(m,k,j,i), 0.25) - u_psi0_(m,0,k,j,i);
    });
    pmgd_psi->SeedInitialGuess(delta_psi, indcs.ng);
  }

  pmgd_psi->SetUsePsi5Source(use_psi5_source);
  pmgd_psi->LoadMatterSource(use_psi5_source ? u_raw : u_tilde, indcs.ng);
  pmgd_psi->LoadNonlinearCoefficient(a_sq, a0_sq, indcs.ng);
  pmgd_psi->LoadPunctureCoefficients(u_psi0, a0_sq, indcs.ng);
  if (puncture_enabled_) {
    ComputeMassCentroid();
    pmgd_psi->SetRobinCenter(r_com_mass_[0], r_com_mass_[1], r_com_mass_[2]);
  }
  {
    MGScopedTimer pt(MultigridDriver::phase_on, MultigridDriver::phase_solve_sec);
    pmgd_psi->Solve(pdriver, stage);
  }
  pmgd_psi->RetrieveSolution(delta_psi);

  // Clamp psi BEFORE AssembleConformalMetric, so psi4/g_dd -- which the shared
  // con2prim at step 4 consumes -- are built from the floored psi too, and before the
  // psi ghost exchange (CFC_RestPsi onward). See ApplyPsiFloor.
  ApplyPsiFloor();

  AssemblePsiMetric(false);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::RescaleMatterSources(Driver *pdriver, int stage)
//! \brief step 4: rebuild S-tilde (trace of S_ij) from the fresh primitives MHD_C2P
//! just recovered. Mirrors dyn_grmhd.cpp's DynGRMHD::SetTmunu's S_dd formula
//! exactly, including the magnetic-field terms (NOT the pure-fluid closed form
//! rho*h*W^2*v^2+3*P alone, which would silently drop magnetic contributions for
//! any magnetized run), then contracts to the trace via adm::Trace. No con2prim
//! call here: this task depends on MHD_C2P, so pmy_pack->pmhd->w0 is already fresh
//! against the g_dd SolveConformalFactor() wrote.

void CFC::RescaleMatterSources(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nmb = pmy_pack->nmb_thispack;

  auto &adm = pmy_pack->padm->adm;
  auto &prim = pmy_pack->pmhd->w0;
  auto &cons = pmy_pack->pmhd->u0;
  auto &bcc = pmy_pack->pmhd->bcc0;
  auto &s_tilde_ = s_tilde;

  par_for("cfc_rescale_matter_src", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real gxx = adm.g_dd(m,0,0,k,j,i), gxy = adm.g_dd(m,0,1,k,j,i);
    Real gxz = adm.g_dd(m,0,2,k,j,i), gyy = adm.g_dd(m,1,1,k,j,i);
    Real gyz = adm.g_dd(m,1,2,k,j,i), gzz = adm.g_dd(m,2,2,k,j,i);
    Real detg = adm::SpatialDet(gxx, gxy, gxz, gyy, gyz, gzz);
    Real sqrtdetg = Kokkos::sqrt(detg);
    Real ivol = 1.0/sqrtdetg;
    Real detginv = 1.0/detg;

    Real v_d[3] = {0.0};
    Real iW = 0.0;
    Real B_d[3] = {0.0};
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        v_d[a] += prim(m, IVX+b, k, j, i)*adm.g_dd(m, a, b, k, j, i);
        iW += prim(m, IVX+a, k, j, i)*prim(m, IVX+b, k, j, i)*adm.g_dd(m, a, b, k, j, i);
        B_d[a] += bcc(m, b, k, j, i)*adm.g_dd(m, a, b, k, j, i)*ivol;
      }
    }
    iW = 1.0/Kokkos::sqrt(1. + iW);
    Real Bv = 0.0, Bsq = 0.0;
    for (int a = 0; a < 3; ++a) {
      Bv += bcc(m, a, k, j, i)*v_d[a]*ivol;
      Bsq += bcc(m, a, k, j, i)*B_d[a]*ivol;
    }
    Real bsq = (Bsq + Bv*Bv)*(iW*iW);

    Real S_dd[3][3];
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        S_dd[a][b] = cons(m, IM1+a, k, j, i)*ivol*v_d[b]*iW
                     - (B_d[a] + Bv*v_d[a])*SQR(iW)*B_d[b]
                     + (prim(m, IPR, k, j, i) + 0.5*bsq)*adm.g_dd(m, a, b, k, j, i);
        S_dd[b][a] = S_dd[a][b];
      }
    }

    Real trace_S = adm::Trace(detginv, gxx, gxy, gxz, gyy, gyz, gzz,
                               S_dd[0][0], S_dd[0][1], S_dd[0][2],
                               S_dd[1][1], S_dd[1][2], S_dd[2][2]);

    // psi^6 == sqrt(detg) for the conformally-flat ansatz -- reuse sqrtdetg
    // already computed above instead of re-deriving it from psi.
    s_tilde_(m,0,k,j,i) = sqrtdetg*trace_S;
  });
  return;
}

void CFC::SolveLapse(Driver *pdriver, int stage) {
  if (rs_bg_only_) { return; }   // residual_scale f = 0: no solve
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int nmb = pmy_pack->nmb_thispack;
  int ncells1 = u_tilde.extent_int(4);
  int ncells2 = u_tilde.extent_int(3);
  int ncells3 = u_tilde.extent_int(2);

  auto &u_tilde_ = u_tilde;
  auto &s_tilde_ = s_tilde;
  auto &u_plus_2s_ = u_plus_2s;
  par_for("cfc_u_plus_2s", DevExeSpace(), 0, nmb-1, 0, ncells3-1, 0, ncells2-1,
          0, ncells1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    u_plus_2s_(m,0,k,j,i) = u_tilde_(m,0,k,j,i) + 2.0*s_tilde_(m,0,k,j,i);
  });

  // Same one-shot warm-start idea as SolveConformalFactor above, for alpha*psi.
  // Uses padm->adm.alpha (still the pgen's/restart's raw value here --
  // AssembleLapseShiftK runs later this stage) times this stage's own
  // just-converged psi.
  if (!alpha_psi_seeded_) {
    alpha_psi_seeded_ = true;
    int &is = indcs.is; int &ie = indcs.ie;
    int &js = indcs.js; int &je = indcs.je;
    int &ks = indcs.ks; int &ke = indcs.ke;
    auto &adm = pmy_pack->padm->adm;
    auto &psi_c = delta_psi;
    auto &alpha_psi_c = delta_alpha_psi;
    auto &u_psi0_ = u_psi0;
    auto &u_alpha0_psi0_ = u_alpha0_psi0;
    par_for("cfc_seed_alpha_psi", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      alpha_psi_c(m,0,k,j,i) = adm.alpha(m,k,j,i)*(psi_c(m,0,k,j,i) + u_psi0_(m,0,k,j,i))
                               - u_alpha0_psi0_(m,0,k,j,i);
    });
    pmgd_alpha->SeedInitialGuess(delta_alpha_psi, indcs.ng);
  }

  pmgd_alpha->LoadMatterCoefficients(u_plus_2s, delta_psi, a_sq, a0_sq, indcs.ng);
  pmgd_alpha->LoadPunctureCoefficients(u_psi0, a0_sq, u_alpha0_psi0, indcs.ng);
  if (puncture_enabled_) {
    // Reuses this stage's r_com_mass_ (set by SolveConformalFactor(), which always
    // runs first -- u0 is unchanged in between, see ComputeMassCentroid()'s own doc
    // comment) -- no need to recompute.
    pmgd_alpha->SetRobinCenter(r_com_mass_[0], r_com_mass_[1], r_com_mass_[2]);
  }
  {
    MGScopedTimer pt(MultigridDriver::phase_on, MultigridDriver::phase_solve_sec);
    pmgd_alpha->Solve(pdriver, stage);
  }
  pmgd_alpha->RetrieveSolution(delta_alpha_psi);

  // Clamp alpha before the alpha*psi ghost exchange (CFC_RestAlphaPsi onward) and
  // before CFC_BuildSrcBeta, so no non-positive alpha reaches the shift source.
  ApplyAlphaFloor();
  return;
}

void CFC::SolveShift(Driver *pdriver, int stage) {
  if (rs_bg_only_) { return; }   // residual_scale f = 0: no solve
  // AssembleVectorSource(true) fills r_com_beta_ internally (right after
  // BuildShiftSource() completes, Sec 3.10 item 2/Sec 5 Phase A item 8b) -- by the
  // time it returns, r_com_beta_ is ready for SetMultipoleOrigin() below.
  AssembleVectorSource(/*for_shift=*/true);

  if (puncture_enabled_) {
    pmgd_pietabeta->SetMultipoleOrigin(r_com_beta_[0], r_com_beta_[1], r_com_beta_[2]);
  }
  pmgd_pietabeta->LoadPoissonSource(u_p_src);
  {
    MGScopedTimer pt(MultigridDriver::phase_on, MultigridDriver::phase_solve_sec);
    pmgd_pietabeta->Solve(pdriver, stage);
  }
  pmgd_pietabeta->RetrieveSolution(u_p_beta);
  return;
}

void CFC::ReconstructShift() {
  cfc::ReconstructVectorFromPotentials(pmy_pack, p_beta, u_p_beta, beta_u, 3, r_com_beta_);
  return;
}

void CFC::RefillMovingPuncture(Real t) {
  if (!puncture_enabled_) { return; }
  Real xbh[3];
  BHPosition(t, xbh);
  for (int a = 0; a < 3; ++a) { pmy_pack->padm->bc_center[a] = xbh[a]; }
  FillPunctureBackground(pmy_pack, puncture_mass_, xbh, u_psi0, u_alpha0_psi0, beta0_u,
                         a0_dd, a0_sq, grad_ap6_0);
  for (int a = 0; a < 3; ++a) { bg_xbh_[a] = xbh[a]; }
  pmgd_psi->SetPunctureCenter(xbh); pmgd_alpha->SetPunctureCenter(xbh);
}

//----------------------------------------------------------------------------------------
//! \fn Real CFC::ResidualScale(Real t) const
//! \brief <cfc> residual_scale's f(t): the constant, or the C^1 smoothstep
//! f0 + (f1 - f0) x^2 (3 - 2x), x = (t - t0)/(t1 - t0) clamped to [0, 1] (exactly f0 at
//! x = 0 and f1 at x = 1). A function of t only: no restart state.

Real CFC::ResidualScale(Real t) const {
  if (!rs_on_) { return 1.0; }
  if (!rs_ramp_) { return rs_const_; }
  if (t <= rs_t0_) { return rs_f0_; }
  if (t >= rs_t1_) { return rs_f1_; }
  Real x = (t - rs_t0_)/(rs_t1_ - rs_t0_);
  return rs_f0_ + (rs_f1_ - rs_f0_)*(x*x*(3.0 - 2.0*x));
}

void CFC::SetResidualScale(Real t, int stage, bool last_stage) {
  const bool was = rs_bg_only_;
  rs_f_ = ResidualScale(t);
  rs_bg_only_ = (rs_f_ == 0.0);
  if (global_variable::my_rank != 0) { return; }
  const int cyc = pmy_pack->pmesh->ncycle;
  if (rs_bg_only_ != was) {
    std::cout << "CFC residual_scale: cycle=" << cyc << " stage=" << stage << " t=" << t
              << (rs_bg_only_ ? " f=0: background only, multigrid solves skipped"
                              : " f>0: multigrid solves resumed") << std::endl;
  }
  if (rs_log_ > 0 && (stage < 1 || (last_stage && cyc % rs_log_ == 0))) {
    std::cout << "CFC residual_scale: cycle=" << cyc << " stage=" << stage
              << std::setprecision(17) << " t=" << t << " f=" << rs_f_
              << std::setprecision(6) << std::endl;
  }
}

void CFC::AssemblePsiMetric(bool with_ghosts) {
  if (rs_on_ && rs_f_ != 1.0) {
    cfc::AssembleConformalMetricScaled(pmy_pack, delta_psi, u_psi0, with_ghosts, rs_f_);
  } else {
    cfc::AssembleConformalMetric(pmy_pack, delta_psi, u_psi0, with_ghosts);
  }
}

void CFC::AssembleADM(Real time) {
  Real xd[3] = {0.0, 0.0, 0.0};
  if (frame_enabled_) { FrameVel(time, xd); }
  if (rs_on_ && rs_f_ != 1.0) {
    cfc::AssembleLapseShiftKScaled(pmy_pack, delta_psi, delta_alpha_psi, u_psi0,
                                   u_alpha0_psi0, a_dd, a0_dd, beta_u, beta0_u,
                                   alpha_floor_, xd, rs_f_);
  } else {
    cfc::AssembleLapseShiftK(pmy_pack, delta_psi, delta_alpha_psi, u_psi0, u_alpha0_psi0,
                             a_dd, beta_u, beta0_u, alpha_floor_, xd);
  }
  // ADMBCs reads this as beta_u's asymptotic value (adm_bcs.cpp).
  for (int a = 0; a < 3; ++a) { pmy_pack->padm->frame_vel_u[a] = xd[a]; }
  frame_assembled_ = true;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::FillEmbedADM(const DvceArray1D<int> &newmb, Real time)
//! \brief see cfc.hpp

void CFC::FillEmbedADM(const DvceArray1D<int> &newmb, Real time) {
  Real xd[3] = {0.0, 0.0, 0.0};
  if (frame_enabled_) { FrameVel(time, xd); }
  const Real xd0 = xd[0], xd1 = xd[1], xd2 = xd[2];
  const Real afl = alpha_floor_;
  auto &adm = pmy_pack->padm->adm;
  auto &psi0 = u_psi0;
  auto &ap0 = u_alpha0_psi0;
  auto &b0 = beta0_u;
  auto &a0 = a0_dd;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*indcs.ng) : 1;
  const int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*indcs.ng) : 1;
  par_for("cfc_embed_adm", DevExeSpace(), 0, pmy_pack->nmb_thispack-1, 0, n3-1, 0, n2-1,
          0, n1-1, KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    if (newmb(m) == 0) return;
    const Real p = psi0(m,0,k,j,i);
    const Real p2 = p*p, p4 = p2*p2;
    adm.psi4(m,k,j,i) = p4;
    adm.g_dd(m,0,0,k,j,i) = p4; adm.g_dd(m,1,1,k,j,i) = p4; adm.g_dd(m,2,2,k,j,i) = p4;
    adm.g_dd(m,0,1,k,j,i) = 0.0; adm.g_dd(m,0,2,k,j,i) = 0.0; adm.g_dd(m,1,2,k,j,i) = 0.0;
    adm.alpha(m,k,j,i) = fmax(ap0(m,0,k,j,i)/p, afl);
    adm.beta_u(m,0,k,j,i) = b0(m,0,k,j,i) + xd0;
    adm.beta_u(m,1,k,j,i) = b0(m,1,k,j,i) + xd1;
    adm.beta_u(m,2,k,j,i) = b0(m,2,k,j,i) + xd2;
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) { adm.vK_dd(m,a,b,k,j,i) = a0(m,a,b,k,j,i)/p2; }
    }
  });
  for (int a = 0; a < 3; ++a) { pmy_pack->padm->frame_vel_u[a] = xd[a]; }
}

//----------------------------------------------------------------------------------------
//! \fn Real CFC::StageTime(Driver *pdriver, int stage) const
//! \brief see cfc.hpp. For the SSP-RK integrators u0 <- gam0*u0 + gam1*u1 + beta*dt*L
//! (u1 = start-of-step state, mhd_update.cpp), so the stage-s state sits at
//! tau_s = gam0[s-1]*tau_{s-1} + beta[s-1] (rk3: 1, 1/2, 1). Exact for rk1/rk2/rk3;
//! rk4/ImEx's extra u1 updates are not modelled (only xidot's O(dt) timing is affected).

Real CFC::StageTime(Driver *pdriver, int stage) const {
  Real t = pmy_pack->pmesh->time;
  if (stage < 1) { return t; }
  Real tau = 0.0;
  for (int s = 0; s < stage; ++s) { tau = pdriver->gam0[s]*tau + pdriver->beta[s]; }
  return t + tau*pmy_pack->pmesh->dt;
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::ApplyFrameShiftUpdate(Real time)
//! \brief see cfc.hpp. Adds the same constant to every cell (ghosts included), so no
//! ghost exchange is needed and the update commutes with prolongation/restriction.

void CFC::ApplyFrameShiftUpdate(Real time) {
  if (!frame_enabled_) { return; }
  auto &gx = pmy_pack->padm->frame_vel_u;
  if (!frame_assembled_) {
    // Restart with solve_interval > 1 before the first solve: the restart file's
    // adm.beta_u holds the shift assembled/updated at the end of the last cycle,
    // i.e. xidot(pmesh->time).
    FrameVel(pmy_pack->pmesh->time, gx);
    frame_assembled_ = true;
  }
  Real xd[3];
  FrameVel(time, xd);
  const Real d0 = xd[0] - gx[0], d1 = xd[1] - gx[1], d2 = xd[2] - gx[2];
  if (d0 != 0.0 || d1 != 0.0 || d2 != 0.0) {
    auto &adm = pmy_pack->padm->adm;
    int nmb = pmy_pack->nmb_thispack;
    auto &u_adm = pmy_pack->padm->u_adm;
    int n1 = u_adm.extent_int(4), n2 = u_adm.extent_int(3), n3 = u_adm.extent_int(2);
    par_for("cfc_frame_shift_update", DevExeSpace(), 0, nmb-1, 0, n3-1, 0, n2-1, 0, n1-1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      adm.beta_u(m,0,k,j,i) += d0;
      adm.beta_u(m,1,k,j,i) += d1;
      adm.beta_u(m,2,k,j,i) += d2;
    });
  }
  for (int a = 0; a < 3; ++a) { gx[a] = xd[a]; }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::MeasureTracked(Real out[7])
//! \brief Phase 2 M2 controller input (see cfc.hpp): over interior cells with
//! rho > frame_track_rho and |x' - x'_tgt| < frame_track_radius, the sums
//! {m, m x', m y', m z', m V'^x, m V'^y, m V'^z}, m = sqrt(gamma) D dV (u0's IDN),
//! V'^i = alpha u^i/W - beta^i the grid velocity (beta = adm.beta_u, which carries the
//! current xidot). rho and u^i come from w0, which at the last RK stage may lag the
//! conserved state by one stage (C2P is not ordered against CFC_AssembleFinal).
//! Summed over ranks, then broadcast from rank 0 so every rank holds identical values.

void CFC::MeasureTracked(Real out[7]) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmb = pmy_pack->nmb_thispack;
  const int nkji = nx3*nx2*nx1, nji = nx2*nx1;
  auto &size = pmy_pack->pmb->mb_size;
  auto &u0 = pmy_pack->pmhd->u0;
  auto &w0 = pmy_pack->pmhd->w0;
  auto &adm = pmy_pack->padm->adm;
  const Real rho_thr = fc_rho_thr_, r2trk = fc_rtrk_*fc_rtrk_;
  const Real tx = fc_tgt_[0], ty = fc_tgt_[1], tz = fc_tgt_[2];
  Real s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0, s5 = 0.0, s6 = 0.0;
  Kokkos::parallel_reduce("cfc_frame_track", Kokkos::RangePolicy<>(DevExeSpace(), 0,
                          nmb*nkji),
  KOKKOS_LAMBDA(const int idx, Real &a0, Real &a1, Real &a2, Real &a3, Real &a4,
                Real &a5, Real &a6) {
    const int m = idx/nkji;
    const int k = (idx - m*nkji)/nji + ks;
    const int j = (idx - m*nkji - (k - ks)*nji)/nx1 + js;
    const int i = idx - m*nkji - (k - ks)*nji - (j - js)*nx1 + is;
    const Real rho = w0(m, IDN, k, j, i);
    if (!(rho > rho_thr)) return;
    const Real x = CellCenterX(i - is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    const Real y = CellCenterX(j - js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    const Real z = CellCenterX(k - ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    if ((x - tx)*(x - tx) + (y - ty)*(y - ty) + (z - tz)*(z - tz) >= r2trk) return;
    const Real dm = u0(m, IDN, k, j, i)*size.d_view(m).dx1*size.d_view(m).dx2*
                    size.d_view(m).dx3;
    const Real ux = w0(m, IVX, k, j, i), uy = w0(m, IVY, k, j, i), uz = w0(m, IVZ, k, j, i);
    const Real usq = adm.g_dd(m,0,0,k,j,i)*ux*ux + adm.g_dd(m,1,1,k,j,i)*uy*uy
                   + adm.g_dd(m,2,2,k,j,i)*uz*uz
                   + 2.0*(adm.g_dd(m,0,1,k,j,i)*ux*uy + adm.g_dd(m,0,2,k,j,i)*ux*uz
                          + adm.g_dd(m,1,2,k,j,i)*uy*uz);
    const Real aw = adm.alpha(m,k,j,i)/sqrt(1.0 + usq);
    a0 += dm; a1 += dm*x; a2 += dm*y; a3 += dm*z;
    a4 += dm*(aw*ux - adm.beta_u(m,0,k,j,i));
    a5 += dm*(aw*uy - adm.beta_u(m,1,k,j,i));
    a6 += dm*(aw*uz - adm.beta_u(m,2,k,j,i));
  }, Kokkos::Sum<Real>(s0), Kokkos::Sum<Real>(s1), Kokkos::Sum<Real>(s2),
     Kokkos::Sum<Real>(s3), Kokkos::Sum<Real>(s4), Kokkos::Sum<Real>(s5),
     Kokkos::Sum<Real>(s6));
  out[0] = s0; out[1] = s1; out[2] = s2; out[3] = s3; out[4] = s4; out[5] = s5;
  out[6] = s6;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, out, 7, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Bcast(out, 7, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::FrameControlStep(Real t_new)
//! \brief once per step, at the last RK stage after AssembleADM (state at t_new =
//! t + dt): advance xi with the step's xidot, measure the tracked gas, set the next
//! step's xidot (cfc.hpp) and shift adm.beta_u (all cells, ghosts included) to it.

void CFC::FrameControlStep(Real t_new) {
  const Real dt = t_new - fc_t_;
  for (int a = 0; a < 3; ++a) { fc_xi_[a] += fc_xd_[a]*dt; }
  fc_t_ = t_new;
  Real s[7];
  if (fc_mode_ == 0) {
    MeasureTracked(s);
    // R-050 B2 automatic switch (off unless frame_switch_mass_frac > 0)
    if (fc_switch_frac_ > 0.0 && fc_m0_ > 0.0 && s[0] < fc_switch_frac_*fc_m0_) {
      FrameSwitch(t_new, fc_switch_mode_, "tracked mass below frame_switch_mass_frac");
    }
  }
  // R-112 item 4: the exit triggers (off unless set; no reduction at all then)
  if (fc_mode_ != 2 && (fc_app_r_ > 0.0 || fc_acc_rate_ > 0.0)) {
    MeasureExit(fc_exit_);
    if (fc_app_r_ > 0.0 && fc_exit_[0] <= fc_app_r_) {
      FrameSwitch(t_new, 2, "debris approach: d_min - v_in T_ramp <= "
                            "frame_switch_approach_radius");
    } else if (fc_acc_rate_ > 0.0 && fc_m0_ > 0.0 && fc_exit_[1] > fc_acc_rate_*fc_m0_) {
      FrameSwitch(t_new, 2, "accretion onset: inflow through "
                            "frame_switch_accretion_radius");
    }
  }
  if (fc_mode_ == 2) {                 // ramp: xidot -> 0 from the switch value
    const Real s = fmin(1.0, fmax(0.0, (t_new - fc_tsw_)/fc_ramp_T_));
    const Real f = (fc_ramp_shape_ == 1) ? 1.0 - s*s*(3.0 - 2.0*s) : 1.0 - s;
    for (int a = 0; a < 3; ++a) { fc_xd_[a] = f*fc_xdsw_[a]; fc_meas_[a] = 0.0; }
    fc_meas_[3] = 0.0; fc_meas_[4] = 0.0;
    ApplyFrameShiftUpdate(t_new);
    return;
  }
  if (fc_mode_ == 1) {                 // bound_centroid: same law on the bound gas,
    MeasureBound(s);                   // commanded xidot low-passed over frame_tau_a
    if (!(s[0] > 0.0)) { fc_meas_[3] = 0.0; return; }
    Real off2 = 0.0;
    const Real w = (dt > 0.0) ? fmin(1.0, dt/fc_tau_a_) : 0.0;
    for (int a = 0; a < 3; ++a) {
      const Real xc = s[1 + a]/s[0], vg = s[4 + a]/s[0];
      const Real vbh = vg + fc_xd_[a];
      if (dt > 0.0) {
        const Real arate = (vbh - fc_vprev_[a])/dt;
        fc_acc_[a] += (arate - fc_acc_[a])*w;
      }
      fc_vprev_[a] = vbh;
      const Real cmd = vbh + fc_acc_[a]*dt + (xc - fc_tgt_[a])/fc_tau_x_;
      fc_xd_[a] += (cmd - fc_xd_[a])*w;
      fc_meas_[a] = vg;
      off2 += (xc - fc_tgt_[a])*(xc - fc_tgt_[a]);
    }
    fc_meas_[3] = s[0];
    fc_meas_[4] = sqrt(off2);
    ApplyFrameShiftUpdate(t_new);
    return;
  }
  if (!(s[0] > 0.0)) {                 // nothing tracked: keep xidot
    fc_meas_[3] = 0.0;
    return;
  }
  Real xd_new[3];
  Real off2 = 0.0;
  for (int a = 0; a < 3; ++a) {
    const Real xc = s[1 + a]/s[0], vg = s[4 + a]/s[0];
    const Real vbh = vg + fc_xd_[a];
    if (dt > 0.0) {
      const Real arate = (vbh - fc_vprev_[a])/dt;
      fc_acc_[a] += (arate - fc_acc_[a])*fmin(1.0, dt/fc_tau_a_);
    }
    fc_vprev_[a] = vbh;
    xd_new[a] = vbh + fc_acc_[a]*dt + (xc - fc_tgt_[a])/fc_tau_x_;
    fc_meas_[a] = vg;
    off2 += (xc - fc_tgt_[a])*(xc - fc_tgt_[a]);
  }
  fc_meas_[3] = s[0];
  fc_meas_[4] = sqrt(off2);
  for (int a = 0; a < 3; ++a) { fc_xd_[a] = xd_new[a]; }
  ApplyFrameShiftUpdate(t_new);        // adm.beta_u += xidot_new - padm->frame_vel_u
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::InitFrameControl(Driver *pdriver)
//! \brief fresh start only (end of InitializeMetric, adm.beta_u carrying xidot = 0):
//! x'_tgt = <cfc> frame_target_x1/2/3 if given, else the centroid of the gas tracked
//! around the origin;
//! xidot(0) = V_BH of the tracked gas = alpha v - beta; then optionally the comoving
//! initial atmosphere (<cfc> frame_atm_rho > 0: cells with rho <= frame_atm_rho beyond
//! frame_atm_gate of x_BH get u^i = W xidot), so the floor does not start at rest.

void CFC::InitFrameControl(Driver *pdriver) {
  for (int a = 0; a < 3; ++a) { fc_xi_[a] = 0.0; fc_xd_[a] = 0.0; fc_acc_[a] = 0.0; }
  fc_t_ = pmy_pack->pmesh->time;
  Real s[7];
  MeasureTracked(s);
  if (!(s[0] > 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<cfc> frame_controller: no gas above frame_track_rho within "
              << "frame_track_radius of the target at t = 0" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  for (int a = 0; a < 3; ++a) {
    if (!fc_tgt_given_) { fc_tgt_[a] = s[1 + a]/s[0]; }
    fc_xd_[a] = s[4 + a]/s[0];         // padm->frame_vel_u = 0 here
    fc_vprev_[a] = fc_xd_[a];
    fc_meas_[a] = fc_xd_[a];
  }
  fc_meas_[3] = s[0];
  fc_init_ = true;
  if (!(fc_m0_ > 0.0)) { fc_m0_ = s[0]; }   // B2 reference mass (unless frame_switch_mref)
  if (fc_mode_ != 0) {                      // non-default target mode from the start
    const int m0 = fc_mode_;
    fc_mode_ = 0;
    FrameSwitch(fc_t_, m0, "frame_target_mode");
  }
  if (global_variable::my_rank == 0) {
    std::cout << std::setprecision(10) << "CFC frame controller: x'_tgt = (" << fc_tgt_[0]
              << ", " << fc_tgt_[1] << ", " << fc_tgt_[2] << "), xidot(0) = (" << fc_xd_[0]
              << ", " << fc_xd_[1] << ", " << fc_xd_[2] << "), tracked mass " << s[0]
              << std::setprecision(6) << std::endl;
  }
  ApplyFrameShiftUpdate(fc_t_);
  if (fc_atm_rho_ > 0.0) {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    const int is = indcs.is, js = indcs.js, ks = indcs.ks;
    const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
    auto &size = pmy_pack->pmb->mb_size;
    auto &w0 = pmy_pack->pmhd->w0;
    auto &adm = pmy_pack->padm->adm;
    const Real thr = fc_atm_rho_, g2 = fc_atm_gate_*fc_atm_gate_;
    const Real bx = pmy_pack->padm->bc_center[0], by = pmy_pack->padm->bc_center[1];
    const Real bz = pmy_pack->padm->bc_center[2];
    const Real v0 = fc_xd_[0], v1 = fc_xd_[1], v2 = fc_xd_[2];
    par_for("cfc_comoving_atm", DevExeSpace(), 0, pmy_pack->nmb_thispack - 1,
            ks, ks + nx3 - 1, js, js + nx2 - 1, is, is + nx1 - 1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      if (w0(m, IDN, k, j, i) > thr) return;
      const Real x = CellCenterX(i - is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      const Real y = CellCenterX(j - js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      const Real z = CellCenterX(k - ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
      if ((x - bx)*(x - bx) + (y - by)*(y - by) + (z - bz)*(z - bz) <= g2) return;
      // Eulerian v = xidot (the C2P's comoving floor state): u^i = W v^i
      const Real vsq = adm.g_dd(m,0,0,k,j,i)*v0*v0 + adm.g_dd(m,1,1,k,j,i)*v1*v1
                     + adm.g_dd(m,2,2,k,j,i)*v2*v2
                     + 2.0*(adm.g_dd(m,0,1,k,j,i)*v0*v1 + adm.g_dd(m,0,2,k,j,i)*v0*v2
                            + adm.g_dd(m,1,2,k,j,i)*v1*v2);
      const Real wl = 1.0/sqrt(1.0 - fmin(vsq, 0.99));
      w0(m, IVX, k, j, i) = wl*v0; w0(m, IVY, k, j, i) = wl*v1; w0(m, IVZ, k, j, i) = wl*v2;
    });
    pmy_pack->pdyngr->PrimToConInit(is, is + nx1 - 1, js, js + nx2 - 1, ks, ks + nx3 - 1);
    pdriver->InitBoundaryValuesAndPrimitives(pmy_pack->pmesh);
  }
}

//----------------------------------------------------------------------------------------
//! \fn CFC::RestartStateVars / SaveRestartState / LoadRestartState
//! \brief see cfc.hpp. Order: delta_psi, delta_alpha_psi, then the MG finest u of
//! pmgd_psi, pmgd_alpha, pmgd_pietax, pmgd_pietabeta (padded to the mesh's ng).

int CFC::RestartStateVars() const {
  return 2 + pmgd_psi->NumVars() + pmgd_alpha->NumVars() + pmgd_pietax->NumVars()
           + pmgd_pietabeta->NumVars() + (solve_every_ > 1 ? 8 : 0);
}

std::string CFC::RestartStateLayout() const {
  return "cfc.dpsi:1,cfc.dap:1,cfc.mg_psi:" + std::to_string(pmgd_psi->NumVars())
         + ",cfc.mg_alpha:" + std::to_string(pmgd_alpha->NumVars())
         + ",cfc.mg_pietax:" + std::to_string(pmgd_pietax->NumVars())
         + ",cfc.mg_pietabeta:" + std::to_string(pmgd_pietabeta->NumVars())
         + (solve_every_ > 1 ? ",cfc.upx:4,cfc.upbeta:4" : "");
}

void CFC::SaveRestartState(DvceArray5D<Real> &dst, int ns) {
  const int nmb = pmy_pack->nmb_thispack, ng = pmy_pack->pmesh->mb_indcs.ng;
  auto mb = std::make_pair(0, nmb);
  auto all = Kokkos::ALL;
  Kokkos::deep_copy(Kokkos::subview(dst, all, std::make_pair(ns, ns + 1), all, all, all),
                    Kokkos::subview(delta_psi, mb, all, all, all, all));
  Kokkos::deep_copy(Kokkos::subview(dst, all, std::make_pair(ns + 1, ns + 2), all, all, all),
                    Kokkos::subview(delta_alpha_psi, mb, all, all, all, all));
  int n = ns + 2;
  for (MultigridDriver *d : {static_cast<MultigridDriver*>(pmgd_psi),
                             static_cast<MultigridDriver*>(pmgd_alpha),
                             static_cast<MultigridDriver*>(pmgd_pietax),
                             static_cast<MultigridDriver*>(pmgd_pietabeta)}) {
    d->SaveFinestState(dst, n, ng);
    n += d->NumVars();
  }
  if (solve_every_ > 1) {   // the held potentials (ghosts included), see cfc.hpp
    Kokkos::deep_copy(Kokkos::subview(dst, all, std::make_pair(n, n + 4), all, all, all),
                      Kokkos::subview(u_p_x, mb, all, all, all, all));
    Kokkos::deep_copy(Kokkos::subview(dst, all, std::make_pair(n + 4, n + 8), all, all, all),
                      Kokkos::subview(u_p_beta, mb, all, all, all, all));
  }
}

void CFC::LoadRestartState(const DvceArray5D<Real> &src, int ns) {
  const int nmb = pmy_pack->nmb_thispack, ng = pmy_pack->pmesh->mb_indcs.ng;
  auto mb = std::make_pair(0, nmb);
  auto all = Kokkos::ALL;
  Kokkos::deep_copy(Kokkos::subview(delta_psi, mb, all, all, all, all),
                    Kokkos::subview(src, all, std::make_pair(ns, ns + 1), all, all, all));
  Kokkos::deep_copy(Kokkos::subview(delta_alpha_psi, mb, all, all, all, all),
                    Kokkos::subview(src, all, std::make_pair(ns + 1, ns + 2), all, all, all));
  int n = ns + 2;
  for (MultigridDriver *d : {static_cast<MultigridDriver*>(pmgd_psi),
                             static_cast<MultigridDriver*>(pmgd_alpha),
                             static_cast<MultigridDriver*>(pmgd_pietax),
                             static_cast<MultigridDriver*>(pmgd_pietabeta)}) {
    d->LoadFinestState(src, n, ng);
    n += d->NumVars();
  }
  if (solve_every_ > 1) {
    Kokkos::deep_copy(Kokkos::subview(u_p_x, mb, all, all, all, all),
                      Kokkos::subview(src, all, std::make_pair(n, n + 4), all, all, all));
    Kokkos::deep_copy(Kokkos::subview(u_p_beta, mb, all, all, all, all),
                      Kokkos::subview(src, all, std::make_pair(n + 4, n + 8), all, all, all));
  }
  // as the restarted run left them: false after a step that accreted, so the next solves
  // re-seed from adm exactly as the continuous run does (C-100, c100_rst7)
  psi_seeded_ = rst_psi_seeded_;
  alpha_psi_seeded_ = rst_alpha_psi_seeded_;
}

void CFC::SaveRestartScalars(ParameterInput *pin) const {
  pin->SetBoolean("cfc", "restart_psi_seeded", psi_seeded_);
  pin->SetBoolean("cfc", "restart_alpha_psi_seeded", alpha_psi_seeded_);
  // solve_every_stages > 1: the file carries the held potentials (cfc.upx/upbeta) and the
  // pending forced solve; a file written otherwise must not claim them (a later N > 1
  // restart from it then solves first)
  if (solve_every_ > 1) {
    pin->SetBoolean("cfc", "restart_cfc_held", true);
    pin->SetBoolean("cfc", "restart_cfc_force_solve", force_solve_);
  } else if (pin->DoesParameterExist("cfc", "restart_cfc_held")) {
    pin->SetBoolean("cfc", "restart_cfc_held", false);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::SaveFrameState(ParameterInput *pin) const
//! \brief controller state -> pin (the restart file's parameter copy), full precision
//! (ParameterInput::SetReal writes max_digits10), so a restart restores it bitwise.

void CFC::SaveFrameState(ParameterInput *pin) const {
  if (!frame_ctrl_ || !fc_init_) { return; }
  pin->SetReal("cfc", "frame_ctrl_t", fc_t_);
  for (int a = 0; a < 3; ++a) {
    std::string n = std::to_string(a + 1);
    pin->SetReal("cfc", "frame_ctrl_xi" + n, fc_xi_[a]);
    pin->SetReal("cfc", "frame_ctrl_xidot" + n, fc_xd_[a]);
    pin->SetReal("cfc", "frame_ctrl_tgt" + n, fc_tgt_[a]);
    pin->SetReal("cfc", "frame_ctrl_vprev" + n, fc_vprev_[a]);
    pin->SetReal("cfc", "frame_ctrl_acc" + n, fc_acc_[a]);
    pin->SetReal("cfc", "frame_ctrl_xdsw" + n, fc_xdsw_[a]);
  }
  pin->SetInteger("cfc", "frame_ctrl_mode", fc_mode_);
  pin->SetReal("cfc", "frame_ctrl_tsw", fc_tsw_);
  pin->SetReal("cfc", "frame_ctrl_m0", fc_m0_);
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::FrameSwitch(Real t_new, int mode, const char *why)
//! \brief R-050 B2: leave remnant mode for bound_centroid (target = the bound gas's
//! current centroid, so the offset term starts at zero) or ramp (from the current
//! xidot). Called on every rank with identical (reduced) inputs.

void CFC::FrameSwitch(Real t_new, int mode, const char *why) {
  if (mode == fc_mode_) return;
  fc_mode_ = mode;
  fc_tsw_ = t_new;
  for (int a = 0; a < 3; ++a) { fc_xdsw_[a] = fc_xd_[a]; }
  Real b[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  if (mode == 1) {
    MeasureBound(b);
    if (b[0] > 0.0) {
      for (int a = 0; a < 3; ++a) {
        fc_tgt_[a] = b[1 + a]/b[0];
        fc_vprev_[a] = b[4 + a]/b[0] + fc_xd_[a];
      }
    }
  }
  if (global_variable::my_rank == 0) {
    static const char *nm[3] = {"remnant", "bound_centroid", "ramp"};
    std::cout << std::setprecision(10) << "CFC frame controller: switch to " << nm[mode]
              << " at t = " << t_new << " (" << why << "); xidot = (" << fc_xd_[0] << ", "
              << fc_xd_[1] << ", " << fc_xd_[2] << "), m_ref = " << fc_m0_
              << ", bound mass = " << b[0] << std::setprecision(6) << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::MeasureBound(Real out[7])
//! \brief R-050 B2 bound_centroid input: as MeasureTracked, over interior cells with
//! rho > frame_bound_rho, |x' - x_BH| >= frame_bound_rmin and BH-frame Killing energy
//! E = -u_t - 1 < 0, -u_t = alpha W - (beta' - xidot)^i u_i, u_i = gamma_ij u^j
//! (beta' = adm.beta_u carries this step's xidot; tde_dmde.py's convention).

void CFC::MeasureBound(Real out[7]) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmb = pmy_pack->nmb_thispack;
  const int nkji = nx3*nx2*nx1, nji = nx2*nx1;
  auto &size = pmy_pack->pmb->mb_size;
  auto &u0 = pmy_pack->pmhd->u0;
  auto &w0 = pmy_pack->pmhd->w0;
  auto &adm = pmy_pack->padm->adm;
  Real xb[3];
  BHPosition(fc_t_, xb);
  const Real bx = xb[0], by = xb[1], bz = xb[2];
  const Real rho_thr = fc_bound_rho_, r2min = fc_bound_rmin_*fc_bound_rmin_;
  const Real x0 = fc_xd_[0], x1 = fc_xd_[1], x2 = fc_xd_[2];
  Real s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0, s5 = 0.0, s6 = 0.0;
  Kokkos::parallel_reduce("cfc_frame_bound", Kokkos::RangePolicy<>(DevExeSpace(), 0,
                          nmb*nkji),
  KOKKOS_LAMBDA(const int idx, Real &a0, Real &a1, Real &a2, Real &a3, Real &a4,
                Real &a5, Real &a6) {
    const int m = idx/nkji;
    const int k = (idx - m*nkji)/nji + ks;
    const int j = (idx - m*nkji - (k - ks)*nji)/nx1 + js;
    const int i = idx - m*nkji - (k - ks)*nji - (j - js)*nx1 + is;
    const Real rho = w0(m, IDN, k, j, i);
    if (!(rho > rho_thr)) return;
    const Real x = CellCenterX(i - is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    const Real y = CellCenterX(j - js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    const Real z = CellCenterX(k - ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    if ((x - bx)*(x - bx) + (y - by)*(y - by) + (z - bz)*(z - bz) < r2min) return;
    const Real ux = w0(m, IVX, k, j, i), uy = w0(m, IVY, k, j, i), uz = w0(m, IVZ, k, j, i);
    const Real gxx = adm.g_dd(m,0,0,k,j,i), gxy = adm.g_dd(m,0,1,k,j,i);
    const Real gxz = adm.g_dd(m,0,2,k,j,i), gyy = adm.g_dd(m,1,1,k,j,i);
    const Real gyz = adm.g_dd(m,1,2,k,j,i), gzz = adm.g_dd(m,2,2,k,j,i);
    const Real lx = gxx*ux + gxy*uy + gxz*uz;
    const Real ly = gxy*ux + gyy*uy + gyz*uz;
    const Real lz = gxz*ux + gyz*uy + gzz*uz;
    const Real W = sqrt(1.0 + ux*lx + uy*ly + uz*lz);
    const Real al = adm.alpha(m,k,j,i);
    const Real bpx = adm.beta_u(m,0,k,j,i), bpy = adm.beta_u(m,1,k,j,i);
    const Real bpz = adm.beta_u(m,2,k,j,i);
    const Real mut = al*W - ((bpx - x0)*lx + (bpy - x1)*ly + (bpz - x2)*lz);
    if (!(mut - 1.0 < 0.0)) return;
    const Real dm = u0(m, IDN, k, j, i)*size.d_view(m).dx1*size.d_view(m).dx2*
                    size.d_view(m).dx3;
    const Real aw = al/W;
    a0 += dm; a1 += dm*x; a2 += dm*y; a3 += dm*z;
    a4 += dm*(aw*ux - bpx);
    a5 += dm*(aw*uy - bpy);
    a6 += dm*(aw*uz - bpz);
  }, Kokkos::Sum<Real>(s0), Kokkos::Sum<Real>(s1), Kokkos::Sum<Real>(s2),
     Kokkos::Sum<Real>(s3), Kokkos::Sum<Real>(s4), Kokkos::Sum<Real>(s5),
     Kokkos::Sum<Real>(s6));
  out[0] = s0; out[1] = s1; out[2] = s2; out[3] = s3; out[4] = s4; out[5] = s5;
  out[6] = s6;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, out, 7, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Bcast(out, 7, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::MeasureExit(Real out[2])
//! \brief R-112 item 4's trigger inputs (cfc.hpp): out[0] = min over the bound gas
//! (the MeasureBound gates) of r - max(v_in, 0) frame_ramp_time; out[1] = the net inflow
//! of gas with rho > frame_bound_rho through the shell R_acc <= r < R_acc + w,
//! sum(D dV v_in)/w. v_in = -(V - V_BH).(x - x_BH)/r, the grid velocity
//! V = alpha u/W - beta' relative to the hole's, V_BH = -xidot (x_BH = x_BH0 - xi).

void CFC::MeasureExit(Real out[2]) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmb = pmy_pack->nmb_thispack;
  const int nkji = nx3*nx2*nx1, nji = nx2*nx1;
  auto &size = pmy_pack->pmb->mb_size;
  auto &u0 = pmy_pack->pmhd->u0;
  auto &w0 = pmy_pack->pmhd->w0;
  auto &adm = pmy_pack->padm->adm;
  Real xb[3];
  BHPosition(fc_t_, xb);
  const Real bx = xb[0], by = xb[1], bz = xb[2];
  const Real rho_thr = fc_bound_rho_, r2min = fc_bound_rmin_*fc_bound_rmin_;
  const Real x0 = fc_xd_[0], x1 = fc_xd_[1], x2 = fc_xd_[2];
  const Real tr = fc_ramp_T_, ra = fc_acc_r_, rw = fc_acc_w_;
  Real dmin = std::numeric_limits<Real>::max(), flux = 0.0;
  Kokkos::parallel_reduce("cfc_frame_exit", Kokkos::RangePolicy<>(DevExeSpace(), 0,
                          nmb*nkji),
  KOKKOS_LAMBDA(const int idx, Real &dm_min, Real &fl) {
    const int m = idx/nkji;
    const int k = (idx - m*nkji)/nji + ks;
    const int j = (idx - m*nkji - (k - ks)*nji)/nx1 + js;
    const int i = idx - m*nkji - (k - ks)*nji - (j - js)*nx1 + is;
    const Real rho = w0(m, IDN, k, j, i);
    if (!(rho > rho_thr)) return;
    auto &sz = size.d_view(m);
    const Real x = CellCenterX(i - is, nx1, sz.x1min, sz.x1max) - bx;
    const Real y = CellCenterX(j - js, nx2, sz.x2min, sz.x2max) - by;
    const Real z = CellCenterX(k - ks, nx3, sz.x3min, sz.x3max) - bz;
    const Real r2 = x*x + y*y + z*z;
    const Real r = sqrt(r2);
    const Real ux = w0(m, IVX, k, j, i), uy = w0(m, IVY, k, j, i);
    const Real uz = w0(m, IVZ, k, j, i);
    const Real gxx = adm.g_dd(m,0,0,k,j,i), gxy = adm.g_dd(m,0,1,k,j,i);
    const Real gxz = adm.g_dd(m,0,2,k,j,i), gyy = adm.g_dd(m,1,1,k,j,i);
    const Real gyz = adm.g_dd(m,1,2,k,j,i), gzz = adm.g_dd(m,2,2,k,j,i);
    const Real lx = gxx*ux + gxy*uy + gxz*uz;
    const Real ly = gxy*ux + gyy*uy + gyz*uz;
    const Real lz = gxz*ux + gyz*uy + gzz*uz;
    const Real W = sqrt(1.0 + ux*lx + uy*ly + uz*lz);
    const Real al = adm.alpha(m,k,j,i);
    const Real bpx = adm.beta_u(m,0,k,j,i), bpy = adm.beta_u(m,1,k,j,i);
    const Real bpz = adm.beta_u(m,2,k,j,i);
    const Real aw = al/W;
    const Real vx = aw*ux - bpx + x0, vy = aw*uy - bpy + x1, vz = aw*uz - bpz + x2;
    const Real vin = (r > 0.0) ? -(vx*x + vy*y + vz*z)/r : 0.0;
    if (r >= ra && r < ra + rw) {
      fl += u0(m, IDN, k, j, i)*sz.dx1*sz.dx2*sz.dx3*vin/rw;
    }
    if (r2 < r2min) return;
    const Real mut = al*W - ((bpx - x0)*lx + (bpy - x1)*ly + (bpz - x2)*lz);
    if (!(mut - 1.0 < 0.0)) return;
    dm_min = fmin(dm_min, r - fmax(vin, 0.0)*tr);
  }, Kokkos::Min<Real>(dmin), Kokkos::Sum<Real>(flux));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &dmin, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &flux, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Bcast(&dmin, 1, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  MPI_Bcast(&flux, 1, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
#endif
  out[0] = dmin;
  out[1] = flux;
}

}  // namespace cfc
