//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mg_cfc_conformal_factor.cpp
//! \brief implementation of MGCFCConformalFactor[Driver]
//!
//! Sign/scaling convention, derived once here rather than re-derived at every call
//! site: Multigrid::Smooth's stencil.Apply computes lap(u) = 6*u_c - sum(6 nbrs),
//! and the generic linear solve converges when lap(u)/dx^2 = src, i.e.
//! real_Laplacian(u) = -src (confirmed against gravity: LoadSource(u0, IDN, ng,
//! -four_pi_G_) gives src=-4*pi*G*rho, converging to Delta(phi) = -src = 4*pi*G*rho,
//! the standard Poisson equation). Eq. 73 (Gmunu 2021) is
//!   Delta psi = -2*pi*Utilde*psi^-1 - (1/8)*Ahat^2*psi^-7,
//! with psi = u+1 (u = delta_psi), so Delta(u) = Delta(psi). Substituting into the
//! AthenaK convention above (real_Laplacian(u) = -lap(u)/dx^2):
//!   -lap(u)/dx^2 = -2*pi*Utilde*psi^-1 - (1/8)*Ahat^2*psi^-7
//!   lap(u) = dx^2 * [2*pi*Utilde*psi^-1 + (1/8)*Ahat^2*psi^-7] =: dx^2*RHS(u)
//! i.e. F(u) := lap(u) - dx^2*RHS(u) = 0 is the discrete equation solved per point.
//!
//! Utilde and Ahat^2 are both fixed, externally-supplied fields (never depend on this
//! equation's own unknown u) -- but they are loaded into coeff_ (channel 0 = Utilde,
//! channel 1 = Ahat^2, ncoeff_=2), NOT into src_ via LoadSource. This is a deliberate,
//! non-obvious choice: RHS(u) reads Utilde every time it's evaluated (including at
//! every coarser V-cycle level), but src_ is exactly the array that
//! MultigridDriver's generic (non-virtual) V-cycle machinery restricts *and* adds FAS
//! tau-corrections into (via RestrictSourcePack + this file's CalculateFASRHSPack) --
//! if Utilde lived in src_, those FAS corrections would silently corrupt the physical
//! Utilde field that RHS(u) needs at coarser levels. Since this equation has no
//! separate additive "given" term at all (F(u)=0 is fully homogeneous in u), src_'s
//! entire role here is the FAS correction accumulator gravity's pattern already
//! establishes (starts at zero -- Kokkos::realloc zero-initializes -- and is only
//! ever touched by the generic machinery and by this file's CalculateFASRHSPack).
//!
//! Multigrid::LoadCoefficients() copies coeff channels 0..ncoeff_-1 in one shot (no
//! per-channel "ns" offset the way LoadSource has), so it can't be called twice (once
//! per physical field) without the second call clobbering the first. LoadMatterSource/
//! LoadNonlinearCoefficient below therefore each do their own tiny single-channel
//! par_for via the CoeffAtLevel() accessor instead of calling LoadCoefficients.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "multigrid/multigrid.hpp"
#include "mg_cfc_conformal_factor.hpp"

namespace {

// Per-point RHS(u) = 2*pi*Utilde*psi^-1 + (1/8)*Ahat^2*psi^-7 and its derivative
// w.r.t. u (psi = u+1), shared by SmoothPack/CalculateDefectPack/CalculateFASRHSPack
// so the three can never drift out of sync with each other.
KOKKOS_INLINE_FUNCTION
void ConformalFactorRHS(Real u, Real u_tilde, Real ahat_sq, Real *rhs, Real *drhs_du) {
  Real psi = u + 1.0;
  Real psi_inv = 1.0 / psi;
  Real psi_inv2 = psi_inv * psi_inv;
  Real psi_inv7 = psi_inv2 * psi_inv2 * psi_inv2 * psi_inv;
  Real psi_inv8 = psi_inv7 * psi_inv;
  *rhs = 2.0 * M_PI * u_tilde * psi_inv + 0.125 * ahat_sq * psi_inv7;
  *drhs_du = -2.0 * M_PI * u_tilde * psi_inv2 - 0.875 * ahat_sq * psi_inv8;
}

template <typename ViewType>
KOKKOS_INLINE_FUNCTION
Real ConformalFactorLap(const ViewType &u, int m, int k, int j, int i) {
  return 6.0*u(m,0,k,j,i) - u(m,0,k+1,j,i) - u(m,0,k,j+1,i) - u(m,0,k,j,i+1)
         - u(m,0,k-1,j,i) - u(m,0,k,j-1,i) - u(m,0,k,j,i-1);
}

// Item 12: octet-indexed counterpart of ConformalFactorLap above (oct.U(0,k,j,i)
// instead of u(m,0,k,j,i)) -- same 7-point stencil, mechanically identical in
// shape to gravity::OctLaplacian / MGCFCVectorPoissonDriver's own OctLaplacian.
inline Real OctConformalFactorLap(const MGOctet &oct, int k, int j, int i) {
  return 6.0*oct.U(0,k,j,i) - oct.U(0,k+1,j,i) - oct.U(0,k,j+1,i) - oct.U(0,k,j,i+1)
         - oct.U(0,k-1,j,i) - oct.U(0,k,j-1,i) - oct.U(0,k,j,i-1);
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn MGCFCConformalFactor::MGCFCConformalFactor(...)

MGCFCConformalFactor::MGCFCConformalFactor(MultigridDriver *pmd, MeshBlockPack *pmbp,
                                           int nghost, bool on_host)
    : Multigrid(pmd, pmbp, nghost, on_host) {
  // Finding B (plan addendum #3): the base Multigrid ctor allocates u_/src_/def_/
  // uold_ per level but never coeff_/matrix_, and never sets ncoeff_ from the
  // driver -- both are the responsibility of the first real user of coeff_, i.e. us.
  ncoeff_ = 2;  // channel 0 = Utilde, channel 1 = Ahat^2
  for (int l = 0; l < nlevel_; l++) {
    int ll = nlevel_-1-l;
    int ncx = (indcs_.nx1>>ll)+2*ngh_;
    int ncy = (indcs_.nx2>>ll)+2*ngh_;
    int ncz = (indcs_.nx3>>ll)+2*ngh_;
    Kokkos::realloc(coeff_[l], nmmb_, ncoeff_, ncz, ncy, ncx);
  }
}

MGCFCConformalFactor::~MGCFCConformalFactor() {
}

void MGCFCConformalFactor::SmoothPack(int color) {
  int ll = nlevel_-1-current_level_;
  int is = ngh_, ie = is+(indcs_.nx1>>ll)-1;
  int js = ngh_, je = js+(indcs_.nx2>>ll)-1;
  int ks = ngh_, ke = ks+(indcs_.nx3>>ll)-1;
  Real omega = static_cast<MGCFCConformalFactorDriver*>(pmy_driver_)->mg_omega_psi_;
  Real psi_floor = static_cast<MGCFCConformalFactorDriver*>(pmy_driver_)->psi_floor_;
  int lev = current_level_;
  int rlev = -ll;
  int c0 = color ^ pmy_driver_->GetCoffset();
  auto brdx = block_rdx_.d_view;
  auto u = u_[lev].d_view;
  auto coeff = coeff_[lev].d_view;
  // FAS tau-correction accumulator (see this file's top-of-file comment): zero at
  // the finest level, nonzero at every coarser level once CalculateFASRHSPack has
  // run there. Must be added to the nonlinear RHS here -- mirroring the generic
  // linear Multigrid::Smooth's `u -= (lap - src*dx2)*odiag` (multigrid.hpp:606) --
  // or the coarse-grid correction this array exists to carry is silently dropped
  // and every level below the finest just relaxes its own homogeneous equation,
  // decoupled from the fine grid's actual defect.
  auto src = src_[lev].d_view;
  par_for("MGCFCConformalFactor::SmoothPack", DevExeSpace(), 0, nmmb_-1, ks, ke, js, je,
  KOKKOS_LAMBDA(const int m, const int k, const int j) {
    Real dx = (rlev <= 0) ? brdx(m) * static_cast<Real>(1<<(-rlev))
                          : brdx(m) / static_cast<Real>(1<<rlev);
    Real dx2 = dx * dx;
    const int c = (c0 + k + j) & 1;
    for (int i = is + c; i <= ie; i += 2) {
      Real u_old = u(m,0,k,j,i);
      Real rhs, drhs_du;
      ConformalFactorRHS(u_old, coeff(m,0,k,j,i), coeff(m,1,k,j,i), &rhs, &drhs_du);
      Real lap = ConformalFactorLap(u, m, k, j, i);
      Real fprime = 6.0 - dx2*drhs_du;
      Real u_new = u_old - omega*(lap - (rhs + src(m,0,k,j,i))*dx2)/fprime;
      if (u_new + 1.0 < psi_floor) u_new = psi_floor - 1.0;
      u(m,0,k,j,i) = u_new;
    }
  });
}

void MGCFCConformalFactor::CalculateDefectPack() {
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
  // received, corrupting the correction chain for every level further down.
  auto src = src_[lev].d_view;
  par_for("MGCFCConformalFactor::CalculateDefectPack", DevExeSpace(), 0, nmmb_-1,
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real dx = (rlev <= 0) ? brdx(m) * static_cast<Real>(1<<(-rlev))
                          : brdx(m) / static_cast<Real>(1<<rlev);
    Real idx2 = 1.0 / (dx*dx);
    Real rhs, drhs_du;
    ConformalFactorRHS(u(m,0,k,j,i), coeff(m,0,k,j,i), coeff(m,1,k,j,i), &rhs, &drhs_du);
    Real lap = ConformalFactorLap(u, m, k, j, i);
    def(m,0,k,j,i) = (rhs + src(m,0,k,j,i)) - lap*idx2;
  });
}

void MGCFCConformalFactor::CalculateFASRHSPack() {
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
  par_for("MGCFCConformalFactor::CalculateFASRHSPack", DevExeSpace(), 0, nmmb_-1,
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real dx = (rlev <= 0) ? brdx(m) * static_cast<Real>(1<<(-rlev))
                          : brdx(m) / static_cast<Real>(1<<rlev);
    Real idx2 = 1.0 / (dx*dx);
    Real rhs, drhs_du;
    ConformalFactorRHS(u(m,0,k,j,i), coeff(m,0,k,j,i), coeff(m,1,k,j,i), &rhs, &drhs_du);
    Real lap = ConformalFactorLap(u, m, k, j, i);
    src(m,0,k,j,i) += lap*idx2 - rhs;
  });
}


//----------------------------------------------------------------------------------------
//! \fn MGCFCConformalFactorDriver::MGCFCConformalFactorDriver(...)
//! \brief nvar_ = 1, ncoeff_ = 2 (carries Utilde, Ahat^2); mg_multipole boundary
//! conditions (Gmunu eq. 77, isolated/asymptotically-flat falloff).

MGCFCConformalFactorDriver::MGCFCConformalFactorDriver(MeshBlockPack *pmbp,
                                                       ParameterInput *pin)
    : MultigridDriver(pmbp, 1) {
  ncoeff_ = 2;
  eps_ = pin->GetOrAddReal("cfc", "mg_threshold", 1.0e-10);
  fshowdef_ = pin->GetOrAddInteger("cfc", "mg_verbose", 0);
  mg_verbose_ = fshowdef_;
  full_multigrid_ = false;
  // Item 12: AMR-refined meshes need more smoothing per level to fully converge
  // than a uniform-resolution mesh does with the base-class default of 1 (matches
  // binary_gravity.athinput's own top-of-file advice: "more smoothing (npresmooth/
  // npostsmooth = 2 or 3), or refinement = none" -- a known, pre-existing
  // characteristic of this multigrid implementation at refinement boundaries, not
  // specific to CFC). Left at the base default (1) unless overridden.
  npresmooth_ = pin->GetOrAddInteger("cfc", "mg_npresmooth", npresmooth_);
  npostsmooth_ = pin->GetOrAddInteger("cfc", "mg_npostsmooth", npostsmooth_);
  mg_omega_psi_ = pin->GetOrAddReal("cfc", "mg_omega_psi", 1.0);
  psi_floor_ = pin->GetOrAddReal("cfc", "psi_floor", 0.05);

  // Outer (non-periodic, non-reflecting) faces use MultigridDriver's own base-
  // constructor default, BoundaryFlag::mg_zerofixed (Dirichlet delta_psi=0, i.e.
  // psi=1 exactly at the boundary) -- the leading-order truncation of Gmunu eq. 77's
  // true isolated 1/r falloff. Multipole boundaries (mg_multipole) were tried first
  // and found buggy: CalculateCenterOfMass()/CalculateMultipoleCoefficients()
  // (multigrid_driver.cpp) integrate mglevels_->src_ as "the density," which is
  // exactly what gravity puts there -- but this solver's own Utilde/Ahat^2 live in
  // coeff_, not src_ (see this file's top-of-file comment), by design (so V-cycle
  // FAS tau-corrections into src_ can't corrupt them). src_ is therefore always zero
  // when Solve() calls these, so CalculateCenterOfMass's im = 1.0/totals[0] divides
  // by zero, poisoning mpo_/mpcoeff_ with NaN/inf that MGRootBoundary's mg_multipole
  // branch then evaluates directly into the domain's outer ghost cells on the very
  // first V-cycle -- root cause of the boundary NaN documented in DEVELOPMENT.md item
  // 9 (rounds 1-5). Sidestepped by not using mg_multipole here at all: with the
  // domain boundary placed well outside the star (as in the current test case), the
  // zerofixed approximation's O(M/r_boundary) error is an acceptable tradeoff versus
  // fixing/duplicating the multipole moment integration to read coeff_ instead of
  // src_. mporder_/autompo_/AllocateMultipoleCoefficients() below are left in place,
  // inert, so multipole support can be revisited later without redoing this input
  // parsing -- see DEVELOPMENT.md item 9 round 6.
  //
  // Faces where the *mesh* itself is reflecting (e.g. an octant-reduced domain's
  // inner symmetry planes) still need BoundaryFlag::mg_zerograd, not plain
  // BoundaryFlag::reflect (mg_mesh_bcs_ is multigrid-internal state; MGRootBoundary's
  // device path only recognizes periodic/mg_zerofixed/mg_zerograd, and a face left at
  // ordinary BoundaryFlag::reflect falls through every branch silently).
  for (int f = 0; f < 6; ++f) {
    if (pmbp->pmesh->mesh_bcs[f] == BoundaryFlag::reflect) {
      mg_mesh_bcs_[f] = BoundaryFlag::mg_zerograd;
    }
  }
  mporder_ = pin->GetOrAddInteger("cfc", "mporder", 4);
  autompo_ = pin->GetOrAddBoolean("cfc", "auto_mporigin", true);
  nodipole_ = pin->GetOrAddBoolean("cfc", "nodipole", false);
  if (mporder_ != 2 && mporder_ != 4) {
    std::cout << "### FATAL ERROR in MGCFCConformalFactorDriver" << std::endl
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
  mgroot_ = new MGCFCConformalFactor(this, nullptr, nghost, root_on_host);
  mglevels_ = new MGCFCConformalFactor(this, pmbp, nghost);
  mglevels_->pbval = new MultigridBoundaryValues(pmbp, pin, false, mglevels_);
  mglevels_->pbval->InitializeBuffers(nvar_);
  mglevels_->pbval->RemapIndicesForMG();
  mglevels_->pbval->ComputePerLevelIndices();
}

MGCFCConformalFactorDriver::~MGCFCConformalFactorDriver() {
  delete mgroot_;
  delete mglevels_;
}

//----------------------------------------------------------------------------------------
//! \fn void MGCFCConformalFactorDriver::Solve(Driver *pdriver, int stage, Real dt)
//! \brief run the FAS V-cycle solve for delta_psi. Assumes LoadMatterSource()/
//! LoadNonlinearCoefficient() were already called for this stage.

void MGCFCConformalFactorDriver::Solve(Driver *pdriver, int stage, Real dt) {
  PrepareForAMR();
  mglevels_->RestrictCoefficients();
  TransferCoeffToRoot();
  // Item 12: one-time coefficient restriction through the octet hierarchy (a
  // no-op when nreflevel_==0) -- must run after TransferCoeffToRoot has
  // populated each octet level's own Coeff() from its real MeshBlock children,
  // and before SetupMultigrid()/SolveMG() begins reading Coeff() at every level.
  RestrictCoeffOctets();

  SetupMultigrid(dt, false);

  // No mg_multipole face is ever set (see the constructor's boundary-flag comment),
  // so CalculateCenterOfMass()/CalculateMultipoleCoefficients() would be pure dead
  // work here -- and worse, CalculateCenterOfMass's 1.0/totals[0] divides by an
  // always-zero src_ for this solver regardless of whether the result is used,
  // producing an inf/NaN internally every call. Skipped entirely rather than left in
  // as unused-but-still-computed.

  SolveMG(pdriver);
  Kokkos::fence();

  // No self-retrieve here, matching MGCFCVectorPoissonDriver::Solve()'s convention
  // (item 2): the caller (cfc::CFC::SolveConformalFactor) calls RetrieveSolution()
  // separately once this returns.
  return;
}

// Both loaders assume u_tilde/a_sq are sized with ngh_-deep (multigrid-width, not
// mesh-NGHOST-deep) ghost padding -- matching item 2's choice for p_src/eta_src.
// Neither field is ever finite-differenced (only read pointwise inside the Newton
// kernels above and in the lapse equation's K(x)), so there's no reason to pay for
// mesh-NGHOST-deep ghost exchange on them; cfc.cpp (item 4) should size u_tilde/
// a_sq accordingly. This is why a single Multigrid::LoadCoefficients(coeff, ngh)
// call can't be reused here anyway (Finding B) -- writing our own zero-offset
// per-channel copy costs nothing extra beyond what LoadCoefficients would do.

void MGCFCConformalFactorDriver::LoadMatterSource(const DvceArray5D<Real> &u_tilde,
                                                   int ngh) {
  auto &cm = mglevels_->CoeffAtLevel(mglevels_->GetNumberOfLevels()-1);
  int lngh = mglevels_->GetGhostCells();
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  int is = 0, ie = indcs.nx1 + 2*lngh - 1;
  int js = 0, je = indcs.nx2 + 2*lngh - 1;
  int ks = 0, ke = indcs.nx3 + 2*lngh - 1;
  // u_tilde is padded to depth ngh, not this driver's own (generally shallower)
  // lngh -- mirror Multigrid::LoadCoefficients' offset-aware indexing (Finding H)
  // rather than assuming the two depths match.
  const int off = ngh - lngh;
  auto cm_d = cm.d_view;
  int nmmb = pmy_pack_->nmb_thispack;
  par_for("MGCFCConformalFactorDriver::LoadMatterSource", DevExeSpace(),
          0, nmmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int mk, const int mj, const int mi) {
    cm_d(m, 0, mk, mj, mi) = u_tilde(m, 0, mk+off, mj+off, mi+off);
  });
}

void MGCFCConformalFactorDriver::LoadNonlinearCoefficient(
    const DvceArray5D<Real> &a_sq, int ngh) {
  auto &cm = mglevels_->CoeffAtLevel(mglevels_->GetNumberOfLevels()-1);
  int lngh = mglevels_->GetGhostCells();
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  int is = 0, ie = indcs.nx1 + 2*lngh - 1;
  int js = 0, je = indcs.nx2 + 2*lngh - 1;
  int ks = 0, ke = indcs.nx3 + 2*lngh - 1;
  const int off = ngh - lngh;  // see LoadMatterSource above (Finding H)
  auto cm_d = cm.d_view;
  int nmmb = pmy_pack_->nmb_thispack;
  par_for("MGCFCConformalFactorDriver::LoadNonlinearCoefficient", DevExeSpace(),
          0, nmmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int mk, const int mj, const int mi) {
    cm_d(m, 1, mk, mj, mi) = a_sq(m, 0, mk+off, mj+off, mi+off);
  });
}

void MGCFCConformalFactorDriver::RetrieveSolution(DvceArray5D<Real> &dst) {
  // dst (psi) is mesh-NGHOST-deep, not this solver's own (generally shallower) ngh_
  // -- passing GetGhostCells() here made RetrieveResult's dst_off = ngh - ngh_
  // collapse to 0 instead of (mesh NGHOST - ngh_), silently copying the solved
  // interior 3 cells too low (for NGHOST=4, ngh_=1) and leaving the outermost
  // interior cells near every domain face never written at all (stuck at their
  // pre-solve 1.0 default, becoming 2.0 after the +1 pass below). Matches gravity's
  // and MGCFCVectorPoissonDriver's own (correct) RetrieveResult calls, which both
  // pass the mesh's true NGHOST.
  mglevels_->RetrieveResult(dst, 0, pmy_pack_->pmesh->mb_indcs.ng);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MGCFCConformalFactorDriver::SeedInitialGuess(...)

void MGCFCConformalFactorDriver::SeedInitialGuess(const DvceArray5D<Real> &guess,
                                                   int ngh) {
  mglevels_->LoadFinestData(guess, 0, ngh);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MGCFCConformalFactorDriver::TransferCoeffToRoot()
//! \brief Finding C: MultigridDriver::TransferFromBlocksToRoot (multigrid_driver.cpp)
//! aggregates every rank's coarsest per-block cell into the distributed root grid
//! (mgroot_) via MPI_Allgatherv, but only for src_/u_ -- never coeff_. That transfer
//! runs for any multi-meshblock mesh (not AMR-specific), so mgroot_ needs its own
//! Utilde/Ahat^2 populated the same way before the V-cycle can reach the root level.
//! Deliberately duplicates the relevant logic here rather than touching
//! src/multigrid/ -- see plan addendum #3, Finding C. Item 12 extended this with the
//! octet-parented branch TransferFromBlocksToRoot itself already has (blocks refined
//! past the root level write into their parent octet's Coeff() instead of the root
//! grid) -- the nreflevel_==0 guard that used to make this branch unreachable is
//! gone now that Solve() supports AMR.

void MGCFCConformalFactorDriver::TransferCoeffToRoot() {
  const int nc = ncoeff_;
  auto *mgc_lvl = static_cast<MGCFCConformalFactor*>(mglevels_);
  auto *mgc_root = static_cast<MGCFCConformalFactor*>(mgroot_);
  auto &coeff_lvl = mgc_lvl->CoeffAtLevel(0);
  const int ngh_mb = mgc_lvl->GetGhostCells();
  int nmmb = pmy_pack_->nmb_thispack - 1;
  int padding = nslist_[global_variable::my_rank];

  DualArray2D<Real> coeffbuf;
  Kokkos::realloc(coeffbuf, nc, nbtotal_);
  auto coeffbuf_d = coeffbuf.d_view;
  auto coeff_lvl_d = coeff_lvl.d_view;
  par_for("MGCFCConformalFactorDriver::SaveCoeffToRoot", DevExeSpace(), 0, nmmb,
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
  // Item 12 (found while debugging the AMR smoke test, not AMR-specific): the
  // above only ever populates mgroot_'s OWN finest internal level (nrootlevel_-1
  // -- the root grid can itself span multiple V-cycle levels, e.g. 4x4x4 -> 2x2x2
  // -> 1x1x1, whenever there are enough root-level blocks/octets). Every coarser
  // root level's coeff_ was left at its post-construction default (0) -- the
  // Newton kernel there would then solve against a wrong (all-zero K(x)-ingredient)
  // equation, corrupting the FAS coarse-grid correction fed back up and stalling
  // convergence. mglevels_->RestrictCoefficients() (called by Solve() just before
  // this function) only restricts the *per-block* hierarchy; mgroot_ needs the
  // identical treatment applied to itself.
  mgc_root->RestrictCoefficients();
  return;
}

// Item 12: octet-scale Newton-Gauss-Seidel smoothing, exactly the same math as
// MGCFCConformalFactor::SmoothPack (per-level) above -- ConformalFactorRHS is
// reused verbatim (a pure scalar function, no view dependency); only the
// Laplacian stencil and the u/src/coeff access pattern change (oct.U/Src/Coeff
// instead of u_[lev]/src_[lev]/coeff_[lev] views). mg_omega_psi_/psi_floor_ are
// this driver's own members, no static_cast needed (unlike SmoothPack, which is
// a Multigrid, not MultigridDriver, method).
void MGCFCConformalFactorDriver::SmoothOctet(MGOctet &oct, int rlev, int color) {
  int ngh = mgroot_->GetGhostCells();
  Real root_dx = mgroot_->GetRootDx();
  Real dx = root_dx / static_cast<Real>(1 << rlev);
  Real dx2 = dx * dx;
  int c = color ^ coffset_;
  for (int k = ngh; k <= ngh+1; ++k) {
    for (int j = ngh; j <= ngh+1; ++j) {
      for (int i = ngh + ((c^k^j)&1); i <= ngh+1; i += 2) {
        Real u_old = oct.U(0,k,j,i);
        Real rhs, drhs_du;
        ConformalFactorRHS(u_old, oct.Coeff(0,k,j,i), oct.Coeff(1,k,j,i),
                            &rhs, &drhs_du);
        Real lap = OctConformalFactorLap(oct, k, j, i);
        Real fprime = 6.0 - dx2*drhs_du;
        Real u_new = u_old - mg_omega_psi_*(lap - (rhs + oct.Src(0,k,j,i))*dx2)/fprime;
        if (u_new + 1.0 < psi_floor_) u_new = psi_floor_ - 1.0;
        oct.U(0,k,j,i) = u_new;
      }
    }
  }
}

void MGCFCConformalFactorDriver::CalculateDefectOctet(MGOctet &oct, int rlev) {
  int ngh = mgroot_->GetGhostCells();
  Real root_dx = mgroot_->GetRootDx();
  Real dx = root_dx / static_cast<Real>(1 << rlev);
  Real idx2 = 1.0 / (dx*dx);
  for (int k = ngh; k <= ngh+1; ++k) {
    for (int j = ngh; j <= ngh+1; ++j) {
      for (int i = ngh; i <= ngh+1; ++i) {
        Real rhs, drhs_du;
        ConformalFactorRHS(oct.U(0,k,j,i), oct.Coeff(0,k,j,i), oct.Coeff(1,k,j,i),
                            &rhs, &drhs_du);
        Real lap = OctConformalFactorLap(oct, k, j, i);
        oct.Def(0,k,j,i) = (rhs + oct.Src(0,k,j,i)) - lap*idx2;
      }
    }
  }
}

void MGCFCConformalFactorDriver::CalculateFASRHSOctet(MGOctet &oct, int rlev) {
  int ngh = mgroot_->GetGhostCells();
  Real root_dx = mgroot_->GetRootDx();
  Real dx = root_dx / static_cast<Real>(1 << rlev);
  Real idx2 = 1.0 / (dx*dx);
  for (int k = ngh; k <= ngh+1; ++k) {
    for (int j = ngh; j <= ngh+1; ++j) {
      for (int i = ngh; i <= ngh+1; ++i) {
        Real rhs, drhs_du;
        ConformalFactorRHS(oct.U(0,k,j,i), oct.Coeff(0,k,j,i), oct.Coeff(1,k,j,i),
                            &rhs, &drhs_du);
        Real lap = OctConformalFactorLap(oct, k, j, i);
        oct.Src(0,k,j,i) += lap*idx2 - rhs;
      }
    }
  }
}
