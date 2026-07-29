//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cfc.cpp
//! \brief implementation of the CFC class

#include <algorithm>
#include <cmath>
#include <iostream>

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
#include "cfc_reconstruct.hpp"

namespace cfc {

namespace {

//----------------------------------------------------------------------------------------
//! \fn void BuildShiftSourceImpl<NGHOST>(...)
//! \brief Gmunu eq. 75's derivative term, 2*Adual^ij*D_j(alpha*psi^-6), added onto an
//! already-built p_src (16*pi*alpha*psi^-6*S-tilde_i -- see CFC::AssembleVectorSource).

template <int NGHOST>
void BuildShiftSourceImpl(MeshBlockPack *pmbp, const DvceArray5D<Real> &delta_psi,
                          const DvceArray5D<Real> &delta_alpha_psi,
                          const AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &a_dd,
                          AthenaTensor<Real, TensorSymm::NONE, 3, 1> &p_src,
                          int mg_nghost, DvceArray5D<Real> &ap6) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  int ncells1 = indcs.nx1 + 2*indcs.ng;
  int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*indcs.ng) : 1;
  int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*indcs.ng) : 1;

  // alpha*psi^-6 = alpha_psi * psi^-7 (alpha = alpha_psi/psi). ap6 is a persistent
  // CFC member, sized over the full array extent (not just the interior) so
  // Dx<NGHOST> below has valid neighbor data at every interior point -- both fields
  // are already ghost-exchanged by the time this runs (see CFC::QueueCFCTasks).
  par_for("cfc_build_alpha_psi6", DevExeSpace(), 0, nmb-1, 0, ncells3-1, 0, ncells2-1,
          0, ncells1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real psi_val = delta_psi(m,0,k,j,i) + 1.0;
    Real psi7 = psi_val*psi_val*psi_val*psi_val*psi_val*psi_val*psi_val;
    ap6(m,0,k,j,i) = (delta_alpha_psi(m,0,k,j,i) + 1.0)/psi7;
  });

  AthenaTensor<Real, TensorSymm::NONE, 3, 0> ap6_view;
  ap6_view.InitWithShallowSlice(ap6, 0);

  par_for("cfc_build_shift_src", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const int mk = k - ks + mg_nghost, mj = j - js + mg_nghost, mi = i - is + mg_nghost;
    Real idx[] = {1.0/size.d_view(m).dx1, 1.0/size.d_view(m).dx2,
                  1.0/size.d_view(m).dx3};
    Real dap6[3];
    for (int b = 0; b < 3; ++b) {
      dap6[b] = Dx<NGHOST>(b, idx, ap6_view, m, k, j, i);
    }
    for (int a = 0; a < 3; ++a) {
      Real div = 0.0;
      for (int b = 0; b < 3; ++b) {
        div += a_dd(m,a,b,k,j,i)*dap6[b];
      }
      p_src(m,a,mk,mj,mi) += 2.0*div;
    }
  });
}

void BuildShiftSource(MeshBlockPack *pmbp, const DvceArray5D<Real> &delta_psi,
                      const DvceArray5D<Real> &delta_alpha_psi,
                      const AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &a_dd,
                      AthenaTensor<Real, TensorSymm::NONE, 3, 1> &p_src,
                      int mg_nghost, DvceArray5D<Real> &ap6) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  switch (indcs.ng) {
    case 2: BuildShiftSourceImpl<2>(pmbp, delta_psi, delta_alpha_psi, a_dd, p_src,
                                     mg_nghost, ap6); break;
    case 3: BuildShiftSourceImpl<3>(pmbp, delta_psi, delta_alpha_psi, a_dd, p_src,
                                     mg_nghost, ap6); break;
    case 4: BuildShiftSourceImpl<4>(pmbp, delta_psi, delta_alpha_psi, a_dd, p_src,
                                     mg_nghost, ap6); break;
  }
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn CFC::CFC(MeshBlockPack *pmbp, ParameterInput *pin)
//! \brief allocates intermediate fields and the 4 multigrid solvers. pmbp->padm/
//! pmbp->ptmunu are already guaranteed non-null by the caller (mesh/meshblock_pack.cpp).

CFC::CFC(MeshBlockPack *pmbp, ParameterInput *pin) :
    pmy_pack(pmbp),
    u_x("cfc_u_x", 1, 1, 1, 1, 1),
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
    pmgd_pietax(nullptr),
    pmgd_pietabeta(nullptr),
    pmgd_psi(nullptr),
    pmgd_alpha(nullptr),
    pbval_pietax(nullptr), pbval_x(nullptr),
    pbval_psi(nullptr), pbval_alpha_psi(nullptr),
    pbval_pietabeta(nullptr),
    coarse_u_pietax("cfc_coarse_u_pietax", 1, 1, 1, 1, 1),
    coarse_u_x("cfc_coarse_u_x", 1, 1, 1, 1, 1),
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
  Kokkos::realloc(u_x,      nmb, 3, ncells3, ncells2, ncells1);
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

  x_u.InitWithShallowSlice(u_x, 0, 2);
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

  // Post-multigrid ghost exchange: one MeshBoundaryValuesCC + coarse shadow array
  // per field cfc_reconstruct.cpp later differentiates, mirroring
  // z4c::Z4c::pbval_u/coarse_u0 (is_z4c=false throughout -- see cfc.hpp).
  pbval_pietax = new MeshBoundaryValuesCC(pmbp, pin, false);
  pbval_pietax->InitializeBuffers(4);
  pbval_x = new MeshBoundaryValuesCC(pmbp, pin, false);
  pbval_x->InitializeBuffers(3);
  pbval_psi = new MeshBoundaryValuesCC(pmbp, pin, false);
  pbval_psi->InitializeBuffers(1);
  pbval_alpha_psi = new MeshBoundaryValuesCC(pmbp, pin, false);
  pbval_alpha_psi->InitializeBuffers(1);
  pbval_pietabeta = new MeshBoundaryValuesCC(pmbp, pin, false);
  pbval_pietabeta->InitializeBuffers(4);

  // Ghost-exchange for padm->u_adm itself -- see pbval_adm's comment in cfc.hpp.
  // CFC never runs with z4c active, so u_adm always carries the full nadm channels
  // (adm::ADM::ADM's constructor only shrinks it and aliases alpha/beta_u into
  // pz4c->u0 when pz4c != nullptr -- never true here).
  pbval_adm = new MeshBoundaryValuesCC(pmbp, pin, false);
  pbval_adm->InitializeBuffers(adm::ADM::nadm);

  // coarse_* shadow arrays: only needed with SMR/AMR (RestrictCC/ProlongateCC are
  // internal no-ops otherwise), matching z4c.cpp's identical multilevel guard.
  if (pmy_pack->pmesh->multilevel) {
    int nccells1 = indcs.cnx1 + 2*(indcs.ng);
    int nccells2 = (indcs.cnx2 > 1) ? (indcs.cnx2 + 2*(indcs.ng)) : 1;
    int nccells3 = (indcs.cnx3 > 1) ? (indcs.cnx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(coarse_u_pietax,    nmb, 4, nccells3, nccells2, nccells1);
    Kokkos::realloc(coarse_u_x,         nmb, 3, nccells3, nccells2, nccells1);
    Kokkos::realloc(coarse_psi,         nmb, 1, nccells3, nccells2, nccells1);
    Kokkos::realloc(coarse_alpha_psi,   nmb, 1, nccells3, nccells2, nccells1);
    Kokkos::realloc(coarse_u_pietabeta, nmb, 4, nccells3, nccells2, nccells1);
    // u_adm's own coarse shadow is padm->coarse_u_adm (adm.cpp:57-67) -- no
    // separate CFC-owned copy; see cfc.hpp's pbval_adm comment.
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
                << " init_use_psi5_source=true (the latter defaults to true --"
                << " see DEVELOPMENT.md items 27-29): freezing Utilde and letting"
                << " the Newton solve treat U_raw=Utilde/psi^6 as the fixed"
                << " quantity are contradictory assumptions. Forcing"
                << " init_use_psi5_source=false for this run." << std::endl;
    }
    cfc_init_use_psi5_ = false;
  }
}

//----------------------------------------------------------------------------------------
//! \fn CFC::~CFC()

CFC::~CFC() {
  delete pmgd_pietax;
  delete pmgd_pietabeta;
  delete pmgd_psi;
  delete pmgd_alpha;
  delete pbval_pietax;
  delete pbval_x;
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

  // Post all 6 fields' non-blocking MPI receives up front, before any of this
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

  pnr->QueueTask(&CFC::ReconstructXTask, this, CFC_ReconstructX, "CFC_ReconstructX",
                 Task_Run, {CFC_BCSPiEtaX});

  pnr->QueueTask(&CFC::RestXTask, this, CFC_RestX, "CFC_RestX",
                 Task_Run, {CFC_ReconstructX});
  pnr->QueueTask(&CFC::SendXTask, this, CFC_SendX, "CFC_SendX",
                 Task_Run, {CFC_RestX});
  pnr->QueueTask(&CFC::RecvXTask, this, CFC_RecvX, "CFC_RecvX",
                 Task_Run, {CFC_SendX});
  pnr->QueueTask(&CFC::ProlongXTask, this, CFC_ProlongX, "CFC_ProlongX",
                 Task_Run, {CFC_RecvX});
  pnr->QueueTask(&CFC::BCSXTask, this, CFC_BCSX, "CFC_BCSX",
                 Task_Run, {CFC_ProlongX});

  pnr->QueueTask(&CFC::ComputeADualTask, this, CFC_ComputeADual, "CFC_ComputeADual",
                 Task_Run, {CFC_BCSX});

  pnr->QueueTask(&CFC::SolvePsiTask, this, CFC_SolvePsi, "CFC_SolvePsi",
                 Task_Run, {CFC_ComputeADual});
  pnr->QueueTask(&CFC::RescaleSrcTask, this, CFC_RescaleSrc, "CFC_RescaleSrc",
                 Task_Run, {MHD_C2P});
  pnr->QueueTask(&CFC::SolveLapseTask, this, CFC_SolveLapse, "CFC_SolveLapse",
                 Task_Run, {CFC_RescaleSrc});

  pnr->QueueTask(&CFC::RestPsiTask, this, CFC_RestPsi, "CFC_RestPsi",
                 Task_Run, {CFC_SolveLapse});
  pnr->QueueTask(&CFC::SendPsiTask, this, CFC_SendPsi, "CFC_SendPsi",
                 Task_Run, {CFC_RestPsi});
  pnr->QueueTask(&CFC::RecvPsiTask, this, CFC_RecvPsi, "CFC_RecvPsi",
                 Task_Run, {CFC_SendPsi});
  pnr->QueueTask(&CFC::ProlongPsiTask, this, CFC_ProlongPsi, "CFC_ProlongPsi",
                 Task_Run, {CFC_RecvPsi});
  pnr->QueueTask(&CFC::BCSPsiTask, this, CFC_BCSPsi, "CFC_BCSPsi",
                 Task_Run, {CFC_ProlongPsi});

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
  pbval_x->InitRecv(3);
}
void CFC::ClearXFields() {
  pbval_pietax->ClearSend();
  pbval_x->ClearSend();
  pbval_pietax->ClearRecv();
  pbval_x->ClearRecv();
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

  // Solve X^i: build S_i-tilde/U-tilde from the current cons, solve the packed
  // (P_i, eta), ghost-exchange it, reconstruct x_u, ghost-exchange it too.
  SolveVectorPotential(pdriver, 0);
  RestPiEtaXTask(pdriver, 0);  SendPiEtaXTask(pdriver, 0);
  while (RecvPiEtaXTask(pdriver, 0) != TaskStatus::complete) {}
  ProlongPiEtaXTask(pdriver, 0);
  BCSPiEtaXTask(pdriver, 0);

  ReconstructVectorPotential();
  RestXTask(pdriver, 0);  SendXTask(pdriver, 0);
  while (RecvXTask(pdriver, 0) != TaskStatus::complete) {}
  ProlongXTask(pdriver, 0);
  BCSXTask(pdriver, 0);

  ClearXFields();

  // Adual^ij/Ahat^2 from the just-exchanged x_u, then solve psi -- this updates
  // padm->adm.g_dd/psi4 (via AssembleConformalMetric) for whatever comes next.
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
    par_for("cfc_init_psi_before_seed", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      psi_before_seed(m,0,k,j,i) = Kokkos::pow(adm_.psi4(m,k,j,i), 0.25) - 1.0;
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
        cfc::AssembleConformalMetric(pmy_pack, delta_psi);
      }

      Real dpsi = 0.0;
      auto &psi_ = delta_psi;
      auto &psi_old_ = psi_old;
      Kokkos::parallel_reduce("cfc_init_dpsi",
        Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>({0, ks, js, is},
                                                              {nmb, ke+1, je+1, ie+1}),
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i,
                      Real &local_max) {
          local_max = Kokkos::fmax(local_max,
                                    Kokkos::fabs(psi_(m,0,k,j,i) - psi_old_(m,0,k,j,i)));
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
  // function already run once before this solve (Driver::Initialize()), safe to
  // call again: confirmed to not touch padm->SetADMVariables.
  pdriver->InitBoundaryValuesAndPrimitives(pmy_pack->pmesh);
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
  AssembleADM();

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
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nmb = pmy_pack->nmb_thispack;

  psi_seeded_ = false;
  alpha_psi_seeded_ = false;

  // Snapshot the true initial guess (psi4^0.25-1, the same formula
  // SolveConformalFactor's own one-time psi_seeded_ block uses) directly from
  // adm.psi4, not delta_psi (which is still at whatever stale value it held before
  // this call and is not what the Newton solve will actually be seeded from).
  DvceArray5D<Real> psi_before("cfc_regrid_psi_before", delta_psi.extent_int(0), 1,
                                delta_psi.extent_int(2), delta_psi.extent_int(3),
                                delta_psi.extent_int(4));
  auto &adm_ = pmy_pack->padm->adm;
  auto &psi_before_seed = psi_before;
  par_for("cfc_regrid_psi_before_seed", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    psi_before_seed(m,0,k,j,i) = Kokkos::pow(adm_.psi4(m,k,j,i), 0.25) - 1.0;
  });

  RunXPsiSolvePass(pdriver);
  // Interior-only: RescaleMatterSources (inside RunLapseShiftAssemblePass
  // below) only reads w0 at these same interior indices, so ghost cells don't
  // need primitives yet -- and the metric they'd be read against isn't final
  // until that pass's own padm->u_adm ghost exchange completes anyway (see
  // the ghost-only pass below).
  pmy_pack->pdyngr->ConToPrimBC(is, ie, js, je, ks, ke);
  RunLapseShiftAssemblePass(pdriver);

  // Ghost-shell-only con2prim, now that padm->u_adm is fully final and fully
  // ghost-exchanged (RunLapseShiftAssemblePass's own tail, just completed):
  // recovers w0's ghost cells against the *correct* metric, closing the
  // window where they'd otherwise read a zero/stale one (DEVELOPMENT.md item
  // 37/38). Hydro's own u0/b0 ghosts don't need re-exchanging here -- that
  // already happened once, earlier in the same enclosing
  // Driver::InitBoundaryValuesAndPrimitives call this function is now called
  // from (mirroring z4c::Z4c::ConvertZ4cToADM's placement) -- so unlike the
  // old design, no second full InitBoundaryValuesAndPrimitives call is
  // needed. 6-slab decomposition of the full ghost shell (mirrors
  // DynGRMHD::ApplyPhysicalBCs's own boundary-strip pattern, but full ghost
  // width instead of a thin band): x1-faces get full y/z extent (covering
  // corners too); y-/z-faces only need the interior x-range (already covered
  // by the x1-faces) to avoid redundant work at edges/corners.
  {
    auto *pm = pmy_pack->pmesh;
    auto *pdyngr = pmy_pack->pdyngr;
    int &ng = indcs.ng;
    int n1 = indcs.nx1 + 2*ng;
    int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
    int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
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
// internal no-ops without SMR/AMR; is_z4c=false throughout since CFC is not z4c).

TaskStatus CFC::RestPiEtaXTask(Driver *pdriver, int stage) {
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(u_p_x, coarse_u_pietax, false);
  }
  return TaskStatus::complete;
}
TaskStatus CFC::SendPiEtaXTask(Driver *pdriver, int stage) {
  return pbval_pietax->PackAndSendCC(u_p_x, coarse_u_pietax);
}
TaskStatus CFC::RecvPiEtaXTask(Driver *pdriver, int stage) {
  return pbval_pietax->RecvAndUnpackCC(u_p_x, coarse_u_pietax);
}
TaskStatus CFC::ProlongPiEtaXTask(Driver *pdriver, int stage) {
  if (pmy_pack->pmesh->multilevel) {
    // Step 1: apply physical BCs to the coarse array so the prolongation stencil
    // reads valid data in coarse ghost zones at a physical boundary.
    if (!(pmy_pack->pmesh->strictly_periodic)) {
      MeshBoundaryValues::CFCBCsCoarse(pmy_pack, coarse_u_pietax, 4, 1);
    }
    pbval_pietax->FillCoarseInBndryCC(u_p_x, coarse_u_pietax);
    pbval_pietax->ProlongateCC(u_p_x, coarse_u_pietax, false);
  }
  return TaskStatus::complete;
}
//! \fn TaskStatus CFC::BCSPiEtaXTask
//! \brief Apply physical BCs to u_p_x's fine array. Runs after ProlongPiEtaXTask
//! (task-graph dependency {CFC_ProlongPiEtaX}) so corner ghost zones between a
//! coarse neighbor and a physical boundary read valid, already-prolongated data
//! (mirrors z4c::Z4c::ApplyPhysicalBCs).
TaskStatus CFC::BCSPiEtaXTask(Driver *pdriver, int stage) {
  if (!(pmy_pack->pmesh->strictly_periodic)) {
    MeshBoundaryValues::CFCBCs(pmy_pack, u_p_x, 4, 1);
  }
  return TaskStatus::complete;
}

TaskStatus CFC::RestXTask(Driver *pdriver, int stage) {
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(u_x, coarse_u_x, false);
  }
  return TaskStatus::complete;
}
TaskStatus CFC::SendXTask(Driver *pdriver, int stage) {
  return pbval_x->PackAndSendCC(u_x, coarse_u_x);
}
TaskStatus CFC::RecvXTask(Driver *pdriver, int stage) {
  return pbval_x->RecvAndUnpackCC(u_x, coarse_u_x);
}
TaskStatus CFC::ProlongXTask(Driver *pdriver, int stage) {
  if (pmy_pack->pmesh->multilevel) {
    if (!(pmy_pack->pmesh->strictly_periodic)) {
      MeshBoundaryValues::CFCBCsCoarse(pmy_pack, coarse_u_x, 3, 1);
    }
    pbval_x->FillCoarseInBndryCC(u_x, coarse_u_x);
    pbval_x->ProlongateCC(u_x, coarse_u_x, false);
  }
  return TaskStatus::complete;
}
TaskStatus CFC::BCSXTask(Driver *pdriver, int stage) {
  if (!(pmy_pack->pmesh->strictly_periodic)) {
    MeshBoundaryValues::CFCBCs(pmy_pack, u_x, 3, 1);
  }
  return TaskStatus::complete;
}

TaskStatus CFC::RestPsiTask(Driver *pdriver, int stage) {
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(delta_psi, coarse_psi, false);
  }
  return TaskStatus::complete;
}
TaskStatus CFC::SendPsiTask(Driver *pdriver, int stage) {
  return pbval_psi->PackAndSendCC(delta_psi, coarse_psi);
}
TaskStatus CFC::RecvPsiTask(Driver *pdriver, int stage) {
  return pbval_psi->RecvAndUnpackCC(delta_psi, coarse_psi);
}
TaskStatus CFC::ProlongPsiTask(Driver *pdriver, int stage) {
  if (pmy_pack->pmesh->multilevel) {
    if (!(pmy_pack->pmesh->strictly_periodic)) {
      MeshBoundaryValues::CFCBCsCoarse(pmy_pack, coarse_psi, 1, 1);
    }
    pbval_psi->FillCoarseInBndryCC(delta_psi, coarse_psi);
    pbval_psi->ProlongateCC(delta_psi, coarse_psi, false);
  }
  return TaskStatus::complete;
}
TaskStatus CFC::BCSPsiTask(Driver *pdriver, int stage) {
  if (!(pmy_pack->pmesh->strictly_periodic)) {
    MeshBoundaryValues::CFCBCs(pmy_pack, delta_psi, 1, 1);
  }
  return TaskStatus::complete;
}

TaskStatus CFC::RestAlphaPsiTask(Driver *pdriver, int stage) {
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(delta_alpha_psi, coarse_alpha_psi, false);
  }
  return TaskStatus::complete;
}
TaskStatus CFC::SendAlphaPsiTask(Driver *pdriver, int stage) {
  return pbval_alpha_psi->PackAndSendCC(delta_alpha_psi, coarse_alpha_psi);
}
TaskStatus CFC::RecvAlphaPsiTask(Driver *pdriver, int stage) {
  return pbval_alpha_psi->RecvAndUnpackCC(delta_alpha_psi, coarse_alpha_psi);
}
TaskStatus CFC::ProlongAlphaPsiTask(Driver *pdriver, int stage) {
  if (pmy_pack->pmesh->multilevel) {
    if (!(pmy_pack->pmesh->strictly_periodic)) {
      MeshBoundaryValues::CFCBCsCoarse(pmy_pack, coarse_alpha_psi, 1, 1);
    }
    pbval_alpha_psi->FillCoarseInBndryCC(delta_alpha_psi, coarse_alpha_psi);
    pbval_alpha_psi->ProlongateCC(delta_alpha_psi, coarse_alpha_psi, false);
  }
  return TaskStatus::complete;
}
TaskStatus CFC::BCSAlphaPsiTask(Driver *pdriver, int stage) {
  if (!(pmy_pack->pmesh->strictly_periodic)) {
    MeshBoundaryValues::CFCBCs(pmy_pack, delta_alpha_psi, 1, 1);
  }
  return TaskStatus::complete;
}

TaskStatus CFC::RestPiEtaBetaTask(Driver *pdriver, int stage) {
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(u_p_beta, coarse_u_pietabeta, false);
  }
  return TaskStatus::complete;
}
TaskStatus CFC::SendPiEtaBetaTask(Driver *pdriver, int stage) {
  return pbval_pietabeta->PackAndSendCC(u_p_beta, coarse_u_pietabeta);
}
TaskStatus CFC::RecvPiEtaBetaTask(Driver *pdriver, int stage) {
  return pbval_pietabeta->RecvAndUnpackCC(u_p_beta, coarse_u_pietabeta);
}
TaskStatus CFC::ProlongPiEtaBetaTask(Driver *pdriver, int stage) {
  if (pmy_pack->pmesh->multilevel) {
    if (!(pmy_pack->pmesh->strictly_periodic)) {
      MeshBoundaryValues::CFCBCsCoarse(pmy_pack, coarse_u_pietabeta, 4, 1);
    }
    pbval_pietabeta->FillCoarseInBndryCC(u_p_beta, coarse_u_pietabeta);
    pbval_pietabeta->ProlongateCC(u_p_beta, coarse_u_pietabeta, false);
  }
  return TaskStatus::complete;
}
TaskStatus CFC::BCSPiEtaBetaTask(Driver *pdriver, int stage) {
  if (!(pmy_pack->pmesh->strictly_periodic)) {
    MeshBoundaryValues::CFCBCs(pmy_pack, u_p_beta, 4, 1);
  }
  return TaskStatus::complete;
}

// Ghost-exchange padm->u_adm itself, once per stage, right after AssembleFinalTask
// writes its interior (see pbval_adm's comment in cfc.hpp). Physical-boundary BCs
// (ADMBCs/ADMBCsCoarse, adm_bcs.cpp) are applied by ProlongADMTask/BCSADMTask below,
// in the same Restrict->Send->Recv->Prolong->ApplyPhysicalBCs order every other CFC
// field and z4c::Z4c use.
TaskStatus CFC::RestADMTask(Driver *pdriver, int stage) {
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(pmy_pack->padm->u_adm, pmy_pack->padm->coarse_u_adm,
                                      false);
  }
  return TaskStatus::complete;
}
TaskStatus CFC::SendADMTask(Driver *pdriver, int stage) {
  return pbval_adm->PackAndSendCC(pmy_pack->padm->u_adm, pmy_pack->padm->coarse_u_adm);
}
TaskStatus CFC::RecvADMTask(Driver *pdriver, int stage) {
  return pbval_adm->RecvAndUnpackCC(pmy_pack->padm->u_adm, pmy_pack->padm->coarse_u_adm);
}
TaskStatus CFC::ProlongADMTask(Driver *pdriver, int stage) {
  if (pmy_pack->pmesh->multilevel) {
    // Step 1: apply physical BCs to the coarse array so the prolongation stencil
    // reads valid data in coarse ghost zones at a physical boundary.
    if (!(pmy_pack->pmesh->strictly_periodic)) {
      MeshBoundaryValues::ADMBCsCoarse(pmy_pack, pmy_pack->padm->coarse_u_adm);
    }
    pbval_adm->FillCoarseInBndryCC(pmy_pack->padm->u_adm, pmy_pack->padm->coarse_u_adm);
    pbval_adm->ProlongateCC(pmy_pack->padm->u_adm, pmy_pack->padm->coarse_u_adm, false);
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
  if (!(pmy_pack->pmesh->strictly_periodic)) {
    MeshBoundaryValues::ADMBCs(pmy_pack, pmy_pack->padm->u_adm);
  }
  return TaskStatus::complete;
}

// InitRecv/ClearSend/ClearRecv (bvals_tasks.cpp) all unconditionally return
// TaskStatus::complete, so no incomplete-status propagation is needed -- matches
// z4c::Z4c::InitRecv/ClearSend/ClearRecv's own one-line-wrapper shape, just looped
// over CFC's 6 MeshBoundaryValuesCC instances instead of one.
TaskStatus CFC::InitRecvTask(Driver *pdriver, int stage) {
  pbval_pietax->InitRecv(4);
  pbval_x->InitRecv(3);
  pbval_psi->InitRecv(1);
  pbval_alpha_psi->InitRecv(1);
  pbval_pietabeta->InitRecv(4);
  pbval_adm->InitRecv(adm::ADM::nadm);
  return TaskStatus::complete;
}
TaskStatus CFC::ClearSendTask(Driver *pdriver, int stage) {
  pbval_pietax->ClearSend();
  pbval_x->ClearSend();
  pbval_psi->ClearSend();
  pbval_alpha_psi->ClearSend();
  pbval_pietabeta->ClearSend();
  pbval_adm->ClearSend();
  return TaskStatus::complete;
}
TaskStatus CFC::ClearRecvTask(Driver *pdriver, int stage) {
  pbval_pietax->ClearRecv();
  pbval_x->ClearRecv();
  pbval_psi->ClearRecv();
  pbval_alpha_psi->ClearRecv();
  pbval_pietabeta->ClearRecv();
  pbval_adm->ClearRecv();
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
// step 1: X^i's packed (P_i, eta) right-hand side, solved (not yet reconstructed into
// x_u -- needs u_p_x's own ghost exchange first, see CFC_ReconstructX). Runs after
// MHD_AddSrc, once this stage's hydro flux+source update has produced the conserved
// state AssembleVectorSource reads from.

TaskStatus CFC::SolveVecXTask(Driver *pdriver, int stage) {
  SolveVectorPotential(pdriver, stage);
  return TaskStatus::complete;
}

// CFC_ReconstructX: build x_u from u_p_x (packed P_i, eta) once its ghost exchange
// has completed.

TaskStatus CFC::ReconstructXTask(Driver *pdriver, int stage) {
  ReconstructVectorPotential();
  return TaskStatus::complete;
}

// step 2: Adual^ij/Ahat^2 from x_u, once x_u's own ghost exchange completes.

TaskStatus CFC::ComputeADualTask(Driver *pdriver, int stage) {
  ComputeADual();
  return TaskStatus::complete;
}

// step 3: psi (nonlinear), then the early psi4/g_dd write MHD_C2P depends on.

TaskStatus CFC::SolvePsiTask(Driver *pdriver, int stage) {
  SolveConformalFactor(pdriver, stage);
  return TaskStatus::complete;
}

// step 4: rebuild S-tilde from the primitives MHD_C2P (this task's dependency) just
// recovered -- no con2prim call here, see RescaleMatterSources.

TaskStatus CFC::RescaleSrcTask(Driver *pdriver, int stage) {
  RescaleMatterSources(pdriver, stage);
  return TaskStatus::complete;
}

// step 5: alpha*psi (nonlinear).

TaskStatus CFC::SolveLapseTask(Driver *pdriver, int stage) {
  SolveLapse(pdriver, stage);
  return TaskStatus::complete;
}

// step 6: beta^i's packed (P_i, eta) right-hand side, solved (not yet reconstructed --
// needs u_p_beta's own ghost exchange first, see CFC_ReconstructBeta). Runs after
// psi/alpha_psi's own ghost exchange (both needed by the eq. 75 source term).

TaskStatus CFC::SolveShiftTask(Driver *pdriver, int stage) {
  SolveShift(pdriver, stage);
  return TaskStatus::complete;
}

// CFC_ReconstructBeta: build beta_u from u_p_beta (packed P_i, eta) once its ghost
// exchange has completed.

TaskStatus CFC::ReconstructBetaTask(Driver *pdriver, int stage) {
  ReconstructShift();
  return TaskStatus::complete;
}

// final step: vK_dd/alpha/beta_u -> padm->u_adm.

TaskStatus CFC::AssembleFinalTask(Driver *pdriver, int stage) {
  AssembleADM();
  return TaskStatus::complete;
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
      Real xk[3] = {x1v, x2v, x3v};

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
    par_for("cfc_assemble_shift_src", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      const int mk = k - ks + mg_nghost, mj = j - js + mg_nghost, mi = i - is + mg_nghost;
      Real psi_val = delta_psi_(m,0,k,j,i) + 1.0;
      Real psi2 = psi_val*psi_val;
      Real psi7 = psi2*psi2*psi2*psi_val;
      // alpha*psi^-6 = (alpha_psi/psi)*psi^-6 = alpha_psi*psi^-7.
      Real ap6 = (delta_alpha_psi_(m,0,k,j,i) + 1.0)/psi7;
      for (int a = 0; a < 3; ++a) {
        p_src_(m,a,mk,mj,mi) = 16.0*M_PI*ap6*s_tilde_d_(m,a,k,j,i);
      }
    });
    // Derivative part (2*Adual^ij*D_j(alpha*psi^-6)), added onto p_src in place.
    BuildShiftSource(pmy_pack, delta_psi, delta_alpha_psi, a_dd, p_src, mg_nghost,
                      u_alpha_psi6);

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
      Real xk[3] = {x1v, x2v, x3v};
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
  AssembleVectorSource(/*for_shift=*/false);

  pmgd_pietax->LoadPoissonSource(u_p_src);
  pmgd_pietax->Solve(pdriver, stage);
  pmgd_pietax->RetrieveSolution(u_p_x);
  return;
}

void CFC::ReconstructVectorPotential() {
  cfc::ReconstructVectorFromPotentials(pmy_pack, p_x, u_p_x, x_u, 3);
  return;
}

void CFC::ComputeADual() {
  cfc::ComputeADualFromX(pmy_pack, x_u, a_dd);

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

void CFC::SolveConformalFactor(Driver *pdriver, int stage, bool use_psi5_source) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;

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
    par_for("cfc_seed_psi", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      psi_(m,0,k,j,i) = Kokkos::pow(adm.psi4(m,k,j,i), 0.25) - 1.0;
    });
    pmgd_psi->SeedInitialGuess(delta_psi, indcs.ng);
  }

  pmgd_psi->SetUsePsi5Source(use_psi5_source);
  pmgd_psi->LoadMatterSource(use_psi5_source ? u_raw : u_tilde, indcs.ng);
  pmgd_psi->LoadNonlinearCoefficient(a_sq, indcs.ng);
  pmgd_psi->Solve(pdriver, stage);
  pmgd_psi->RetrieveSolution(delta_psi);

  cfc::AssembleConformalMetric(pmy_pack, delta_psi);
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
    par_for("cfc_seed_alpha_psi", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      alpha_psi_c(m,0,k,j,i) = adm.alpha(m,k,j,i)*(psi_c(m,0,k,j,i) + 1.0) - 1.0;
    });
    pmgd_alpha->SeedInitialGuess(delta_alpha_psi, indcs.ng);
  }

  pmgd_alpha->LoadReactionCoefficient(u_plus_2s, delta_psi, a_sq, indcs.ng);
  pmgd_alpha->Solve(pdriver, stage);
  pmgd_alpha->RetrieveSolution(delta_alpha_psi);
  return;
}

void CFC::SolveShift(Driver *pdriver, int stage) {
  AssembleVectorSource(/*for_shift=*/true);

  pmgd_pietabeta->LoadPoissonSource(u_p_src);
  pmgd_pietabeta->Solve(pdriver, stage);
  pmgd_pietabeta->RetrieveSolution(u_p_beta);
  return;
}

void CFC::ReconstructShift() {
  cfc::ReconstructVectorFromPotentials(pmy_pack, p_beta, u_p_beta, beta_u, 3);
  return;
}

void CFC::AssembleADM() {
  cfc::AssembleLapseShiftK(pmy_pack, delta_psi, delta_alpha_psi, a_dd, beta_u);
  return;
}

}  // namespace cfc
