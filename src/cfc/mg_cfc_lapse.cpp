//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mg_cfc_lapse.cpp
//! \brief implementation of MGCFCLapse[Driver]
//!
//! Sign/scaling convention (see mg_cfc_conformal_factor.cpp's file-level comment for
//! the AthenaK stencil-sign derivation this reuses: real_Laplacian(u) = -lap(u)/dx^2,
//! lap(u) = 6*u_c - sum(6 nbrs)). Eq. 74 (Gmunu 2021) is
//!   Delta(alpha*psi) = (alpha*psi) * [2*pi*(Utilde+2*Stilde)*psi^-2
//!                                     + (7/8)*Ahat^2*psi^-8] =: (alpha*psi)*K(x),
//! with alpha*psi = u+1 (u = delta_(alpha*psi)). Substituting:
//!   -lap(u)/dx^2 = (u+1)*K(x)
//!   lap(u) + dx^2*K(x)*(u+1) = 0  =:  F(u) = 0
//! K(x) depends only on already-fixed fields (psi, Ahat^2 from earlier steps;
//! Utilde+2*Stilde is this equation's own known matter source) -- not on u at all --
//! so F(u) is affine in u and F'(u) = 6 + dx^2*K(x) is u-independent: the "Newton"
//! step below is an exact one-step Gauss-Seidel solve, matching mg_cfc_lapse.hpp's
//! docstring. K(x) itself lives in coeff_ (ncoeff_=1), not src_, for the identical
//! FAS-consistency reason documented in mg_cfc_conformal_factor.cpp (src_ is the FAS
//! tau-correction accumulator; K(x) must stay pristine at every level).
//!
//! Round 16 fix: K(x) is precomputed ONCE, at the finest level, from psi/Ahat^2/
//! Utilde+2*Stilde (LoadReactionCoefficient below), and only the resulting scalar
//! K(x) is carried through coeff_/RestrictCoefficients() down to coarser levels --
//! NOT its three raw ingredients restricted separately and recombined at each level.
//! The earlier (pre-round-16) version stored psi and Ahat^2 as two more coeff_
//! channels and called LapseReactionCoeff() (still below, now a load-time helper
//! rather than a per-kernel-call one) fresh at every level, including coarse ones.
//! That is inconsistent with FAS: unlike psi's own solver (mg_cfc_conformal_
//! factor.cpp), where "psi" is the local unknown u+1 and is correctly carried
//! through the standard FAS u_ restriction, here psi/Ahat^2 are genuinely fixed
//! *coefficients* -- and restrict(f(a,b)) != f(restrict(a), restrict(b)) for the
//! nonlinear psi^-2/psi^-8 combination K(x) uses. Restricting psi and Ahat^2
//! independently then recombining them at each coarse level therefore gave every
//! level below the finest a systematically wrong K(x), i.e. a coarse-grid operator
//! inconsistent with the fine-grid equation -- exactly the kind of defect that
//! produces the round-16-observed symptom (SolveIterative failing to reach its
//! convergence threshold, with the stalled defect *growing* rather than shrinking
//! at higher resolution, since deeper V-cycles exercise more, and more divergent,
//! coarse levels). Restricting the already-combined K(x) directly is the standard,
//! FAS-consistent treatment for a nonlinearly-derived reaction coefficient.
//!
//! Round 18 fix (found by the user reading this file directly): the round-16 fix
//! above did not, by itself, resolve SolveIterative's non-convergence -- a second,
//! independent bug remained. SmoothPack/CalculateDefectPack never read src_ at all,
//! even though CalculateFASRHSPack (below) correctly accumulates the FAS
//! tau-correction into it -- the exact same bug class as MGCFCConformalFactor's
//! Bug 1 (round 15): the coarse-grid correction was computed and stored every
//! V-cycle descent, then silently discarded, leaving every level below the finest
//! relaxing its own homogeneous equation decoupled from the fine grid's actual
//! defect. Fixed by adding src(m,0,k,j,i) into both formulas, mirroring
//! MGCFCConformalFactor::SmoothPack/CalculateDefectPack's identical fix exactly.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "multigrid/multigrid.hpp"
#include "mg_cfc_lapse.hpp"

namespace {

// K(x) = 2*pi*(Utilde+2*Stilde)*psi^-2 + (7/8)*Ahat^2*psi^-8, from the three fixed
// coeff_ channels. u-independent (psi here is the *already-solved* conformal factor,
// not this equation's own unknown), shared by all three Pack methods below.
KOKKOS_INLINE_FUNCTION
Real LapseReactionCoeff(Real u_plus_2s_tilde, Real psi_known, Real ahat_sq) {
  Real psi_inv2 = 1.0 / (psi_known * psi_known);
  Real psi_inv8 = psi_inv2*psi_inv2*psi_inv2*psi_inv2;
  return 2.0*M_PI*u_plus_2s_tilde*psi_inv2 + 0.875*ahat_sq*psi_inv8;
}

template <typename ViewType>
KOKKOS_INLINE_FUNCTION
Real LapseLap(const ViewType &u, int m, int k, int j, int i) {
  return 6.0*u(m,0,k,j,i) - u(m,0,k+1,j,i) - u(m,0,k,j+1,i) - u(m,0,k,j,i+1)
         - u(m,0,k-1,j,i) - u(m,0,k,j-1,i) - u(m,0,k,j,i-1);
}

// Item 12: octet-indexed counterpart of LapseLap above, mirroring
// MGCFCConformalFactorDriver's OctConformalFactorLap.
inline Real OctLapseLap(const MGOctet &oct, int k, int j, int i) {
  return 6.0*oct.U(0,k,j,i) - oct.U(0,k+1,j,i) - oct.U(0,k,j+1,i) - oct.U(0,k,j,i+1)
         - oct.U(0,k-1,j,i) - oct.U(0,k,j-1,i) - oct.U(0,k,j,i-1);
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn MGCFCLapse::MGCFCLapse(...)

MGCFCLapse::MGCFCLapse(MultigridDriver *pmd, MeshBlockPack *pmbp, int nghost,
                       bool on_host)
    : Multigrid(pmd, pmbp, nghost, on_host) {
  // Finding B (plan addendum #3): see MGCFCConformalFactor's ctor -- coeff_/ncoeff_
  // are never allocated by the base Multigrid ctor, so we do it ourselves.
  ncoeff_ = 1;  // channel 0 = K(x), precomputed at the finest level (round 16 fix)
  for (int l = 0; l < nlevel_; l++) {
    int ll = nlevel_-1-l;
    int ncx = (indcs_.nx1>>ll)+2*ngh_;
    int ncy = (indcs_.nx2>>ll)+2*ngh_;
    int ncz = (indcs_.nx3>>ll)+2*ngh_;
    Kokkos::realloc(coeff_[l], nmmb_, ncoeff_, ncz, ncy, ncx);
  }
}

MGCFCLapse::~MGCFCLapse() {
}

void MGCFCLapse::SmoothPack(int color) {
  int ll = nlevel_-1-current_level_;
  int is = ngh_, ie = is+(indcs_.nx1>>ll)-1;
  int js = ngh_, je = js+(indcs_.nx2>>ll)-1;
  int ks = ngh_, ke = ks+(indcs_.nx3>>ll)-1;
  int lev = current_level_;
  int rlev = -ll;
  int c0 = color ^ pmy_driver_->GetCoffset();
  auto brdx = block_rdx_.d_view;
  auto u = u_[lev].d_view;
  auto coeff = coeff_[lev].d_view;
  // FAS tau-correction accumulator (round 18 fix -- see this file's top-of-file
  // comment): zero at the finest level, nonzero at every coarser level once
  // CalculateFASRHSPack has run there. Must be added here, mirroring
  // MGCFCConformalFactor::SmoothPack's identical fix (round 15, Bug 1) -- or the
  // coarse-grid correction this array exists to carry is silently dropped and every
  // level below the finest just relaxes its own homogeneous equation, decoupled
  // from the fine grid's actual defect.
  auto src = src_[lev].d_view;
  par_for("MGCFCLapse::SmoothPack", DevExeSpace(), 0, nmmb_-1, ks, ke, js, je,
  KOKKOS_LAMBDA(const int m, const int k, const int j) {
    Real dx = (rlev <= 0) ? brdx(m) * static_cast<Real>(1<<(-rlev))
                          : brdx(m) / static_cast<Real>(1<<rlev);
    Real dx2 = dx * dx;
    const int c = (c0 + k + j) & 1;
    for (int i = is + c; i <= ie; i += 2) {
      Real kx = coeff(m,0,k,j,i);  // K(x), precomputed at load time (round 16 fix)
      Real lap = LapseLap(u, m, k, j, i);
      Real u_old = u(m,0,k,j,i);
      Real fval = lap + dx2*kx*(u_old + 1.0) - dx2*src(m,0,k,j,i);
      Real fprime = 6.0 + dx2*kx;
      u(m,0,k,j,i) = u_old - fval/fprime;
    }
  });
}

void MGCFCLapse::CalculateDefectPack() {
  int ll = nlevel_-1-current_level_;
  int is = ngh_, ie = is+(indcs_.nx1>>ll)-1;
  int js = ngh_, je = js+(indcs_.nx2>>ll)-1;
  int ks = ngh_, ke = ks+(indcs_.nx3>>ll)-1;
  int lev = current_level_;
  int rlev = -ll;
  auto brdx = block_rdx_.d_view;
  auto u = u_[lev].d_view;
  auto def = def_[lev].d_view;
  auto coeff = coeff_[lev].d_view;
  // Same FAS tau-correction as SmoothPack above -- must be included here too, or
  // RestrictPack's downstream Restrict(src_[coarser], def_[this level], ...) would
  // restrict a defect that ignores whatever correction this level itself already
  // received, corrupting the correction chain for every level further down
  // (round 18 fix, mirroring MGCFCConformalFactor::CalculateDefectPack, round 15).
  auto src = src_[lev].d_view;
  par_for("MGCFCLapse::CalculateDefectPack", DevExeSpace(), 0, nmmb_-1,
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real dx = (rlev <= 0) ? brdx(m) * static_cast<Real>(1<<(-rlev))
                          : brdx(m) / static_cast<Real>(1<<rlev);
    Real idx2 = 1.0 / (dx*dx);
    Real kx = coeff(m,0,k,j,i);  // K(x), precomputed at load time (round 16 fix)
    Real lap = LapseLap(u, m, k, j, i);
    // def = (RHS(u) + src) - lap(u)*idx2, RHS(u) = -kx*(u+1)
    // (F(u) = lap(u) - dx^2*(RHS(u) + src)).
    def(m,0,k,j,i) = (-kx*(u(m,0,k,j,i) + 1.0) + src(m,0,k,j,i)) - lap*idx2;
  });
}

void MGCFCLapse::CalculateFASRHSPack() {
  int ll = nlevel_-1-current_level_;
  int is = ngh_, ie = is+(indcs_.nx1>>ll)-1;
  int js = ngh_, je = js+(indcs_.nx2>>ll)-1;
  int ks = ngh_, ke = ks+(indcs_.nx3>>ll)-1;
  int lev = current_level_;
  int rlev = -ll;
  auto brdx = block_rdx_.d_view;
  auto u = u_[lev].d_view;
  auto src = src_[lev].d_view;
  auto coeff = coeff_[lev].d_view;
  par_for("MGCFCLapse::CalculateFASRHSPack", DevExeSpace(), 0, nmmb_-1,
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real dx = (rlev <= 0) ? brdx(m) * static_cast<Real>(1<<(-rlev))
                          : brdx(m) / static_cast<Real>(1<<rlev);
    Real idx2 = 1.0 / (dx*dx);
    Real kx = coeff(m,0,k,j,i);  // K(x), precomputed at load time (round 16 fix)
    Real lap = LapseLap(u, m, k, j, i);
    // src += lap(u)*idx2 - RHS(u) = lap(u)*idx2 + kx*(u+1).
    src(m,0,k,j,i) += lap*idx2 + kx*(u(m,0,k,j,i) + 1.0);
  });
}


//----------------------------------------------------------------------------------------
//! \fn MGCFCLapseDriver::MGCFCLapseDriver(...)
//! \brief nvar_ = 1, ncoeff_ = 1 (K(x), precomputed at the finest level -- round 16
//! fix, see this file's header comment); mg_robin boundary conditions by default
//! (Gmunu eq. 78, isolated/asymptotically-flat falloff).

MGCFCLapseDriver::MGCFCLapseDriver(MeshBlockPack *pmbp, ParameterInput *pin)
    : MultigridDriver(pmbp, 1) {
  ncoeff_ = 1;
  eps_ = pin->GetOrAddReal("cfc", "mg_threshold", 1.0e-10);
  fshowdef_ = pin->GetOrAddInteger("cfc", "mg_verbose", 0);
  mg_verbose_ = fshowdef_;
  full_multigrid_ = false;
  // Item 12: see MGCFCConformalFactorDriver's identical comment -- AMR-refined
  // meshes need more smoothing per level than the base-class default of 1.
  npresmooth_ = pin->GetOrAddInteger("cfc", "mg_npresmooth", npresmooth_);
  npostsmooth_ = pin->GetOrAddInteger("cfc", "mg_npostsmooth", npostsmooth_);

  // Outer (non-periodic, non-reflecting) faces default to BoundaryFlag::mg_robin
  // (local 1/r^n extrapolation, no multipole-moment integral) -- see
  // mg_cfc_conformal_factor.cpp's constructor comment for the full rationale,
  // including why mg_multipole was tried first and found buggy here
  // (CalculateCenterOfMass()/CalculateMultipoleCoefficients() integrate src_, which
  // this solver never populates -- Utilde+2*Stilde/psi/Ahat^2 all live in coeff_ --
  // so the multipole path divides by zero and poisons the outer ghost cells with
  // NaN; DEVELOPMENT.md item 9, rounds 1-6) and why mg_zerofixed (the interim
  // workaround) was only the leading-order truncation of the true falloff.
  // mporder_/autompo_/AllocateMultipoleCoefficients() below are left in place,
  // inert. <cfc> mg_outer_bc ("robin" [default] or "zerofixed") allows falling
  // back to the old Dirichlet-zero behavior for direct A/B comparison.
  //
  // Faces where the *mesh* itself is reflecting still need BoundaryFlag::mg_zerograd,
  // not plain BoundaryFlag::reflect (mg_mesh_bcs_ is multigrid-internal state;
  // MGRootBoundary's device path only recognizes
  // periodic/mg_zerofixed/mg_zerograd/mg_robin).
  robin_order_ = pin->GetOrAddInteger("cfc", "mg_robin_order", 1);
  std::string outer_bc_str = pin->GetOrAddString("cfc", "mg_outer_bc", "robin");
  BoundaryFlag outer_bc;
  if (outer_bc_str == "robin") {
    outer_bc = BoundaryFlag::mg_robin;
  } else if (outer_bc_str == "zerofixed") {
    outer_bc = BoundaryFlag::mg_zerofixed;
  } else {
    std::cout << "### FATAL ERROR in MGCFCLapseDriver" << std::endl
              << "cfc/mg_outer_bc must be 'robin' or 'zerofixed'." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  for (int f = 0; f < 6; ++f) {
    if (pmbp->pmesh->mesh_bcs[f] == BoundaryFlag::reflect) {
      mg_mesh_bcs_[f] = BoundaryFlag::mg_zerograd;
    } else if (pmbp->pmesh->mesh_bcs[f] != BoundaryFlag::periodic) {
      mg_mesh_bcs_[f] = outer_bc;
    }
  }
  mporder_ = pin->GetOrAddInteger("cfc", "mporder", 4);
  autompo_ = pin->GetOrAddBoolean("cfc", "auto_mporigin", true);
  nodipole_ = pin->GetOrAddBoolean("cfc", "nodipole", false);
  if (mporder_ != 2 && mporder_ != 4) {
    std::cout << "### FATAL ERROR in MGCFCLapseDriver" << std::endl
              << "mporder must be 2 (quadrupole) or 4 (hexadecapole)." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!autompo_) {
    mpo_[0] = pin->GetOrAddReal("cfc", "mporigin_x1", 0.0);
    mpo_[1] = pin->GetOrAddReal("cfc", "mporigin_x2", 0.0);
    mpo_[2] = pin->GetOrAddReal("cfc", "mporigin_x3", 0.0);
  }
  AllocateMultipoleCoefficients();
  fsubtract_average_ = false;

  int nghost = pin->GetOrAddInteger("cfc", "mg_nghost", 1);
  bool root_on_host = pin->GetOrAddBoolean("cfc", "root_on_host", false);
  mgroot_ = new MGCFCLapse(this, nullptr, nghost, root_on_host);
  mglevels_ = new MGCFCLapse(this, pmbp, nghost);
  mglevels_->pbval = new MultigridBoundaryValues(pmbp, pin, false, mglevels_);
  mglevels_->pbval->InitializeBuffers(nvar_);
  mglevels_->pbval->RemapIndicesForMG();
  mglevels_->pbval->ComputePerLevelIndices();
}

MGCFCLapseDriver::~MGCFCLapseDriver() {
  delete mgroot_;
  delete mglevels_;
}

//----------------------------------------------------------------------------------------
//! \fn void MGCFCLapseDriver::Solve(Driver *pdriver, int stage, Real dt)
//! \brief run the V-cycle solve for delta_(alpha*psi). Assumes
//! LoadReactionCoefficient() was already called for this stage.

void MGCFCLapseDriver::Solve(Driver *pdriver, int stage, Real dt) {
  PrepareForAMR();
  mglevels_->RestrictCoefficients();
  TransferCoeffToRoot();
  // Item 12: one-time coefficient restriction through the octet hierarchy (a
  // no-op when nreflevel_==0) -- must run after TransferCoeffToRoot has
  // populated each octet level's own Coeff() from its real MeshBlock children,
  // and before SetupMultigrid()/SolveMG() begins reading Coeff() at every level.
  RestrictCoeffOctets();

  SetupMultigrid(dt, false);

  // See MGCFCConformalFactorDriver::Solve's identical comment: no mg_multipole face
  // is ever set, so skip the (otherwise dead, and internally NaN-producing via
  // CalculateCenterOfMass's division by an always-zero src_) multipole setup call.

  SolveMG(pdriver);
  Kokkos::fence();

  // No self-retrieve here, matching MGCFCVectorPoissonDriver::Solve()'s convention
  // (item 2): the caller (cfc::CFC::SolveLapse) calls RetrieveSolution() separately
  // once this returns.
  return;
}

// Round 16 fix: combines the old LoadMatterSource/LoadKnownFields pair into one
// call that evaluates K(x) = LapseReactionCoeff(...) once per point, at the finest
// level, and writes only that single value into coeff_ -- see this file's header
// comment for why restricting K(x) directly (rather than restricting psi/Ahat^2/
// Utilde+2*Stilde separately and recombining them nonlinearly at every coarser
// level) is required for a FAS-consistent coarse-grid operator. u_plus_2s_tilde/
// delta_psi/a_sq are assumed padded to the same depth ngh (mesh-NGHOST-deep CFC
// fields, per cfc.cpp), not this driver's own (generally shallower) ngh_ --
// Finding H. delta_psi stores psi - 1 (see cfc::CFC::delta_psi's doc comment,
// cfc.hpp); the physical psi LapseReactionCoeff needs is reconstructed (+1.0).
void MGCFCLapseDriver::LoadReactionCoefficient(
    const DvceArray5D<Real> &u_plus_2s_tilde, const DvceArray5D<Real> &delta_psi,
    const DvceArray5D<Real> &a_sq, int ngh) {
  auto &cm = mglevels_->CoeffAtLevel(mglevels_->GetNumberOfLevels()-1);
  int lngh = mglevels_->GetGhostCells();
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  int is = 0, ie = indcs.nx1 + 2*lngh - 1;
  int js = 0, je = indcs.nx2 + 2*lngh - 1;
  int ks = 0, ke = indcs.nx3 + 2*lngh - 1;
  const int off = ngh - lngh;  // Finding H
  auto cm_d = cm.d_view;
  int nmmb = pmy_pack_->nmb_thispack;
  par_for("MGCFCLapseDriver::LoadReactionCoefficient", DevExeSpace(),
          0, nmmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int mk, const int mj, const int mi) {
    Real u2s = u_plus_2s_tilde(m, 0, mk+off, mj+off, mi+off);
    Real psi_known = delta_psi(m, 0, mk+off, mj+off, mi+off) + 1.0;
    Real ahat_sq = a_sq(m, 0, mk+off, mj+off, mi+off);
    cm_d(m, 0, mk, mj, mi) = LapseReactionCoeff(u2s, psi_known, ahat_sq);
  });
}

void MGCFCLapseDriver::RetrieveSolution(DvceArray5D<Real> &dst) {
  // dst (alpha_psi) is mesh-NGHOST-deep, not this solver's own (generally
  // shallower) ngh_ -- same bug/fix as MGCFCConformalFactorDriver::RetrieveSolution
  // (src/cfc/mg_cfc_conformal_factor.cpp): passing GetGhostCells() here collapsed
  // RetrieveResult's dst_off to 0 instead of (mesh NGHOST - ngh_), silently
  // corrupting the outermost interior cells near every domain face. Matches
  // gravity's and MGCFCVectorPoissonDriver's own (correct) RetrieveResult calls.
  mglevels_->RetrieveResult(dst, 0, pmy_pack_->pmesh->mb_indcs.ng);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MGCFCLapseDriver::SeedInitialGuess(...)

void MGCFCLapseDriver::SeedInitialGuess(const DvceArray5D<Real> &guess, int ngh) {
  mglevels_->LoadFinestData(guess, 0, ngh);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MGCFCLapseDriver::TransferCoeffToRoot()
//! \brief Finding C: see MGCFCConformalFactorDriver::TransferCoeffToRoot for the full
//! rationale -- duplicated here (not shared) since each driver owns a distinct
//! mgroot_/mglevels_ pair of a different concrete Multigrid subclass. Item 12 added
//! the octet-parented branch (see the conformal-factor driver's identical addition).

void MGCFCLapseDriver::TransferCoeffToRoot() {
  const int nc = ncoeff_;
  auto *mgc_lvl = static_cast<MGCFCLapse*>(mglevels_);
  auto *mgc_root = static_cast<MGCFCLapse*>(mgroot_);
  auto &coeff_lvl = mgc_lvl->CoeffAtLevel(0);
  const int ngh_mb = mgc_lvl->GetGhostCells();
  int nmmb = pmy_pack_->nmb_thispack - 1;
  int padding = nslist_[global_variable::my_rank];

  DualArray2D<Real> coeffbuf;
  Kokkos::realloc(coeffbuf, nc, nbtotal_);
  auto coeffbuf_d = coeffbuf.d_view;
  auto coeff_lvl_d = coeff_lvl.d_view;
  par_for("MGCFCLapseDriver::SaveCoeffToRoot", DevExeSpace(), 0, nmmb,
  KOKKOS_LAMBDA(const int m) {
    for (int v = 0; v < nc; ++v) {
      coeffbuf_d(v, m+padding) = coeff_lvl_d(m, v, ngh_mb, ngh_mb, ngh_mb);
    }
  });
  coeffbuf.template modify<DevExeSpace>();
  coeffbuf.template sync<HostExeSpace>();
#if MPI_PARALLEL_ENABLED
  for (int v = 0; v < nc; ++v) {
    MPI_Allgatherv(MPI_IN_PLACE, nblist_[global_variable::my_rank], MPI_ATHENA_REAL,
        &coeffbuf.h_view(v,0), nblist_, nslist_, MPI_ATHENA_REAL, MPI_COMM_WORLD);
  }
#endif

  const auto loc = pmy_mesh_->lloc_eachmb;
  int rootlevel = locrootlevel_;
  int ngh = mgc_root->GetGhostCells();
  auto &coeff_root = mgc_root->CoeffAtLevel(mgc_root->GetNumberOfLevels()-1);
  auto root_coeff_h = coeff_root.h_view;
  for (int n = 0; n < nbtotal_; ++n) {
    int i = static_cast<int>(loc[n].lx1);
    int j = static_cast<int>(loc[n].lx2);
    int k = static_cast<int>(loc[n].lx3);
    if (loc[n].level == rootlevel) {
      for (int v = 0; v < nc; ++v) {
        root_coeff_h(0, v, k+ngh, j+ngh, i+ngh) = coeffbuf.h_view(v, n);
      }
    } else {
      // Item 12: block refined past the root level -- write into its parent
      // octet's Coeff() instead (mirrors TransferFromBlocksToRoot's identical
      // else-branch for Src()/U()).
      LogicalLocation oloc;
      oloc.lx1 = (loc[n].lx1 >> 1);
      oloc.lx2 = (loc[n].lx2 >> 1);
      oloc.lx3 = (loc[n].lx3 >> 1);
      oloc.level = loc[n].level - 1;
      int olev = oloc.level - rootlevel;
      int oid = octetmap_[olev][oloc];
      int oi = (i & 1) + ngh;
      int oj = (j & 1) + ngh;
      int ok = (k & 1) + ngh;
      MGOctet &oct = octets_[olev][oid];
      for (int v = 0; v < nc; ++v) {
        oct.Coeff(v, ok, oj, oi) = coeffbuf.h_view(v, n);
      }
    }
  }
  if (!mgc_root->OnHost()) {
    Kokkos::deep_copy(coeff_root.d_view, coeff_root.h_view);
  }
  // Item 12: see MGCFCConformalFactorDriver::TransferCoeffToRoot's identical
  // comment -- mgroot_ can itself span multiple V-cycle levels, and only its
  // finest one was ever populated above; every coarser root level's coeff_
  // needs the same restriction mglevels_->RestrictCoefficients() already gives
  // the per-block hierarchy.
  mgc_root->RestrictCoefficients();
  return;
}

// Item 12: octet-scale exact one-step Gauss-Seidel (Finding A -- this equation is
// affine in u once psi/Ahat^2 are fixed, unlike the conformal factor's genuine
// Newton iteration), exactly the same math as MGCFCLapse::SmoothPack (per-level)
// above. K(x) is read directly from Coeff(0,...) (already fully evaluated by
// LoadReactionCoefficient at load time, round 16 fix) -- LapseReactionCoeff itself
// is never called here, same as the per-level Pack methods.
void MGCFCLapseDriver::SmoothOctet(MGOctet &oct, int rlev, int color) {
  int ngh = mgroot_->GetGhostCells();
  Real root_dx = mgroot_->GetRootDx();
  Real dx = root_dx / static_cast<Real>(1 << rlev);
  Real dx2 = dx * dx;
  int c = color ^ coffset_;
  for (int k = ngh; k <= ngh+1; ++k) {
    for (int j = ngh; j <= ngh+1; ++j) {
      for (int i = ngh + ((c^k^j)&1); i <= ngh+1; i += 2) {
        Real kx = oct.Coeff(0,k,j,i);
        Real lap = OctLapseLap(oct, k, j, i);
        Real u_old = oct.U(0,k,j,i);
        Real fval = lap + dx2*kx*(u_old + 1.0) - dx2*oct.Src(0,k,j,i);
        Real fprime = 6.0 + dx2*kx;
        oct.U(0,k,j,i) = u_old - fval/fprime;
      }
    }
  }
}

void MGCFCLapseDriver::CalculateDefectOctet(MGOctet &oct, int rlev) {
  int ngh = mgroot_->GetGhostCells();
  Real root_dx = mgroot_->GetRootDx();
  Real dx = root_dx / static_cast<Real>(1 << rlev);
  Real idx2 = 1.0 / (dx*dx);
  for (int k = ngh; k <= ngh+1; ++k) {
    for (int j = ngh; j <= ngh+1; ++j) {
      for (int i = ngh; i <= ngh+1; ++i) {
        Real kx = oct.Coeff(0,k,j,i);
        Real lap = OctLapseLap(oct, k, j, i);
        oct.Def(0,k,j,i) = (-kx*(oct.U(0,k,j,i) + 1.0) + oct.Src(0,k,j,i)) - lap*idx2;
      }
    }
  }
}

void MGCFCLapseDriver::CalculateFASRHSOctet(MGOctet &oct, int rlev) {
  int ngh = mgroot_->GetGhostCells();
  Real root_dx = mgroot_->GetRootDx();
  Real dx = root_dx / static_cast<Real>(1 << rlev);
  Real idx2 = 1.0 / (dx*dx);
  for (int k = ngh; k <= ngh+1; ++k) {
    for (int j = ngh; j <= ngh+1; ++j) {
      for (int i = ngh; i <= ngh+1; ++i) {
        Real kx = oct.Coeff(0,k,j,i);
        Real lap = OctLapseLap(oct, k, j, i);
        oct.Src(0,k,j,i) += lap*idx2 + kx*(oct.U(0,k,j,i) + 1.0);
      }
    }
  }
}
