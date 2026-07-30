//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mg_cfc_conformal_factor.cpp
//! \brief implementation of MGCFCConformalFactor[Driver]
//!
//! Sign convention: Multigrid::Smooth's stencil.Apply computes
//! lap(u) = 6*u_c - sum(6 nbrs); the generic linear solve converges when
//! lap(u)/dx^2 = src, i.e. real_Laplacian(u) = -src (matches gravity's own
//! LoadSource(u0, IDN, ng, -four_pi_G_) -> src=-4*pi*G*rho for Delta(phi)=4*pi*G*rho).
//! Eq. 73 (Gmunu 2021) is
//!   Delta psi = -2*pi*Utilde*psi^-1 - (1/8)*Ahat^2*psi^-7,
//! with psi = u+1 (u = delta_psi). Substituting:
//!   lap(u) = dx^2 * [2*pi*Utilde*psi^-1 + (1/8)*Ahat^2*psi^-7] =: dx^2*RHS(u)
//! i.e. F(u) := lap(u) - dx^2*RHS(u) = 0 is the discrete equation solved per point.
//!
//! Utilde and Ahat^2 are fixed external fields (never depend on u), loaded into
//! coeff_ (channel 0 = Utilde, channel 1 = Ahat^2, ncoeff_=2) rather than src_ via
//! LoadSource: src_ is what the generic V-cycle machinery restricts and adds FAS
//! tau-corrections into, which would corrupt Utilde/Ahat^2. Since F(u)=0 is
//! homogeneous in u, src_'s only role here is the FAS correction accumulator
//! (starts at zero, touched only by the generic machinery and CalculateFASRHSPack).
//!
//! Multigrid::LoadCoefficients() copies all coeff channels in one shot (no
//! per-channel offset), so it can't be called twice without clobbering --
//! LoadMatterSource/LoadNonlinearCoefficient below each do their own single-channel
//! par_for via the CoeffAtLevel() accessor instead.
//!
//! coeff_ channel 0 can instead hold U_raw = Utilde/sqrt(detg) (raw, undensitized
//! energy density), writing the U-term as U_raw*psi^5 instead of Utilde*psi^-1 --
//! algebraically identical (Utilde == psi^6*U_raw), but numerically different
//! whenever the psi that built Utilde (a stale outer-loop iterate) differs from the
//! live Newton iterate. Used only by cfc::CFC::InitializeMetric() (<cfc>
//! init_use_psi5_source): faster convergence for most stars, but diverges to NaN
//! for very compact/unstable ones. See ConformalFactorRHS below for the full
//! derivation; selected via a compile-time template parameter (UsePsi5), not a
//! runtime branch.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "mesh/nghbr_index.hpp"
#include "parameter_input.hpp"
#include "multigrid/multigrid.hpp"
#include "utils/tov/tov.hpp"
#include "utils/tov/tov_polytrope.hpp"
#include "mg_cfc_conformal_factor.hpp"

namespace {

// Per-point RHS(u) and its derivative w.r.t. u (psi = u+1), shared by
// SmoothPack/CalculateDefectPack/CalculateFASRHSPack (and their Octet
// counterparts) so all six can never drift out of sync with each other.
//
// Two algebraically-identical formulations of the U-matter term, selected at
// compile time via the UsePsi5 template parameter (if constexpr, C++17, so the
// per-point hot loop never re-checks a flag):
//   UsePsi5 = false (default, every per-stage CFC_SolvePsi call): u_matter is
//     Utilde = psi^6*U, held fixed for the whole nonlinear solve. RHS's U-term
//     is Utilde*psi^-1 -- effectively psi_prev^6*U*psi_current^-1, not the true
//     equation's U*psi_current^5, whenever the psi that built Utilde
//     (psi_prev) differs from the live Newton iterate.
//   UsePsi5 = true (CFC::InitializeMetric() only, <cfc> init_use_psi5_source):
//     u_matter is U_raw = Utilde/sqrt(detg) (raw, undensitized energy density --
//     see cfc.cpp's AssembleVectorSource). RHS's U-term is U_raw*psi^5, using
//     the same live Newton iterate for the entire power, removing the
//     psi_prev/psi_current staleness above entirely.
template <bool UsePsi5>
KOKKOS_INLINE_FUNCTION
void ConformalFactorRHS(Real u, Real u_matter, Real ahat_sq, Real *rhs, Real *drhs_du) {
  Real psi = u + 1.0;
  Real psi_inv = 1.0 / psi;
  Real psi_inv2 = psi_inv * psi_inv;
  Real psi_inv7 = psi_inv2 * psi_inv2 * psi_inv2 * psi_inv;
  Real psi_inv8 = psi_inv7 * psi_inv;
  Real u_term, du_term;
  if constexpr (UsePsi5) {
    Real psi4 = psi * psi * psi * psi;
    u_term = u_matter * psi4 * psi;      // U_raw * psi^5
    du_term = 5.0 * u_matter * psi4;
  } else {
    u_term = u_matter * psi_inv;         // Utilde * psi^-1
    du_term = -u_matter * psi_inv2;
  }
  *rhs = 2.0 * M_PI * u_term + 0.125 * ahat_sq * psi_inv7;
  *drhs_du = 2.0 * M_PI * du_term - 0.875 * ahat_sq * psi_inv8;
}

template <typename ViewType>
KOKKOS_INLINE_FUNCTION
Real ConformalFactorLap(const ViewType &u, int m, int k, int j, int i) {
  return 6.0*u(m,0,k,j,i) - u(m,0,k+1,j,i) - u(m,0,k,j+1,i) - u(m,0,k,j,i+1)
         - u(m,0,k-1,j,i) - u(m,0,k,j-1,i) - u(m,0,k,j,i-1);
}

// Octet-indexed counterpart of ConformalFactorLap above (oct.U(0,k,j,i) instead of
// u(m,0,k,j,i)) -- same 7-point stencil, mechanically identical in shape to
// gravity::OctLaplacian / MGCFCVectorPoissonDriver's own OctLaplacian.
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
  // The base Multigrid ctor allocates u_/src_/def_/uold_ per level but never
  // coeff_/matrix_, and never sets ncoeff_ from the driver -- both are the
  // responsibility of the first real user of coeff_, i.e. us.
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
  bool use_psi5 = static_cast<MGCFCConformalFactorDriver*>(pmy_driver_)->use_psi5_source_;
  if (use_psi5) { SmoothPackImpl<true>(color); } else { SmoothPackImpl<false>(color); }
}

template <bool UsePsi5>
void MGCFCConformalFactor::SmoothPackImpl(int color) {
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
      ConformalFactorRHS<UsePsi5>(u_old, coeff(m,0,k,j,i), coeff(m,1,k,j,i), &rhs, &drhs_du);
      Real lap = ConformalFactorLap(u, m, k, j, i);
      Real fprime = 6.0 - dx2*drhs_du;
      Real u_new = u_old - omega*(lap - (rhs + src(m,0,k,j,i))*dx2)/fprime;
      if (u_new + 1.0 < psi_floor) u_new = psi_floor - 1.0;
      u(m,0,k,j,i) = u_new;
    }
  });
}

void MGCFCConformalFactor::CalculateDefectPack() {
  bool use_psi5 = static_cast<MGCFCConformalFactorDriver*>(pmy_driver_)->use_psi5_source_;
  if (use_psi5) { CalculateDefectPackImpl<true>(); } else { CalculateDefectPackImpl<false>(); }
}

template <bool UsePsi5>
void MGCFCConformalFactor::CalculateDefectPackImpl() {
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
    ConformalFactorRHS<UsePsi5>(u(m,0,k,j,i), coeff(m,0,k,j,i), coeff(m,1,k,j,i),
                                 &rhs, &drhs_du);
    Real lap = ConformalFactorLap(u, m, k, j, i);
    def(m,0,k,j,i) = (rhs + src(m,0,k,j,i)) - lap*idx2;
  });
}

void MGCFCConformalFactor::CalculateFASRHSPack() {
  bool use_psi5 = static_cast<MGCFCConformalFactorDriver*>(pmy_driver_)->use_psi5_source_;
  if (use_psi5) { CalculateFASRHSPackImpl<true>(); } else { CalculateFASRHSPackImpl<false>(); }
}

template <bool UsePsi5>
void MGCFCConformalFactor::CalculateFASRHSPackImpl() {
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
    ConformalFactorRHS<UsePsi5>(u(m,0,k,j,i), coeff(m,0,k,j,i), coeff(m,1,k,j,i),
                                 &rhs, &drhs_du);
    Real lap = ConformalFactorLap(u, m, k, j, i);
    src(m,0,k,j,i) += lap*idx2 - rhs;
  });
}


//----------------------------------------------------------------------------------------
//! \fn MGCFCConformalFactorDriver::MGCFCConformalFactorDriver(...)
//! \brief nvar_ = 1, ncoeff_ = 2 (carries Utilde, Ahat^2); mg_robin boundary
//! conditions by default (Gmunu eq. 77, isolated/asymptotically-flat falloff).

MGCFCConformalFactorDriver::MGCFCConformalFactorDriver(MeshBlockPack *pmbp,
                                                       ParameterInput *pin)
    : MultigridDriver(pmbp, 1) {
  ncoeff_ = 2;
  eps_ = pin->GetOrAddReal("cfc", "mg_threshold", 1.0e-10);
  fshowdef_ = pin->GetOrAddInteger("cfc", "mg_verbose", 0);
  mg_verbose_ = fshowdef_;
  full_multigrid_ = false;
  // AMR-refined meshes need more smoothing per level to fully converge than a
  // uniform-resolution mesh does with the base-class default of 1 -- a known,
  // pre-existing characteristic of this multigrid implementation at refinement
  // boundaries, not specific to CFC.
  npresmooth_ = pin->GetOrAddInteger("cfc", "mg_npresmooth", npresmooth_);
  npostsmooth_ = pin->GetOrAddInteger("cfc", "mg_npostsmooth", npostsmooth_);
  mg_omega_psi_ = pin->GetOrAddReal("cfc", "mg_omega_psi", 1.0);
  psi_floor_ = pin->GetOrAddReal("cfc", "psi_floor", 0.05);
  // Damps the FAS coarse-grid correction itself (see CorrectionOmega() in
  // mg_cfc_conformal_factor.hpp) -- distinct from mg_omega_psi_ above, which damps
  // each point's own Newton step, not the correction prolongated in from a coarser
  // level. Default 1.0 = undamped.
  mg_correction_omega_ = pin->GetOrAddReal("cfc", "mg_correction_omega", 1.0);
  // Debug-only, see DebugReportDefectByLevel's comment in the header. Default
  // false, zero cost when left off.
  mg_debug_defect_by_level_ = pin->GetOrAddBoolean("cfc", "mg_debug_defect_by_level",
                                                    false);
  // Debug-only, see DebugAnalyticResidualTest's comment in the header. Default
  // false, zero cost when left off. pin_ is stashed only so this diagnostic can
  // (re)construct the analytic TOV solution on demand.
  mg_debug_analytic_residual_test_ = pin->GetOrAddBoolean(
      "cfc", "mg_debug_analytic_residual_test", false);
  pin_ = pin;

  // Outer (non-periodic, non-reflecting) faces default to BoundaryFlag::mg_robin:
  // ghost = interior_anchor * (r_anchor/r_ghost)^mg_robin_order, a local
  // extrapolation of Gmunu eq. 77's isolated 1/r^n falloff with no matter integral
  // or MPI reduction. mg_multipole is not used: CalculateCenterOfMass()/
  // CalculateMultipoleCoefficients() integrate src_ as "the density," but Utilde/
  // Ahat^2 live in coeff_ here (src_ is always zero), so its 1.0/totals[0] would
  // divide by zero. mporder_/autompo_/AllocateMultipoleCoefficients() below are
  // left in place, inert. <cfc> mg_outer_bc ("robin" [default] or "zerofixed")
  // allows falling back to the old Dirichlet-zero behavior without a rebuild.
  //
  // Faces where the *mesh* itself is reflecting still need BoundaryFlag::mg_zerograd,
  // not plain BoundaryFlag::reflect: MGRootBoundary's device path has no case for
  // plain reflect and leaves that ghost cell untouched.
  robin_order_ = pin->GetOrAddInteger("cfc", "mg_robin_order", 1);
  std::string outer_bc_str = pin->GetOrAddString("cfc", "mg_outer_bc", "robin");
  BoundaryFlag outer_bc;
  if (outer_bc_str == "robin") {
    outer_bc = BoundaryFlag::mg_robin;
  } else if (outer_bc_str == "zerofixed") {
    outer_bc = BoundaryFlag::mg_zerofixed;
  } else {
    std::cout << "### FATAL ERROR in MGCFCConformalFactorDriver" << std::endl
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
  // One-time coefficient restriction through the octet hierarchy (a no-op when
  // nreflevel_==0) -- must run after TransferCoeffToRoot has populated each octet
  // level's own Coeff() from its real MeshBlock children, and before
  // SetupMultigrid()/the V-cycle loop below begins reading Coeff() at every level.
  RestrictCoeffOctets();
  // Must run after RestrictCoeffOctets(), not inside TransferCoeffToRoot() --
  // otherwise mgroot_'s coarser internal levels get restricted from a finest level
  // that's still missing the octet-0-to-root contribution RestrictCoeffOctets()
  // just added, under any refined patch.
  mgroot_->RestrictCoefficients();

  if (mg_debug_analytic_residual_test_) {
    DebugDumpRootCoeffUnderOctet();
  }

  SetupMultigrid(dt, false);

  // No mg_multipole face is ever set (see ctor's boundary-flag comment), so
  // CalculateCenterOfMass()/CalculateMultipoleCoefficients() are skipped entirely
  // (would otherwise divide by an always-zero src_).
  if (mg_debug_analytic_residual_test_) {
    DebugAnalyticResidualTest(pdriver);
  }

  SolveMG(pdriver);
  Kokkos::fence();

  if (mg_debug_defect_by_level_) {
    DebugReportDefectByLevel();
  }

  // No self-retrieve here, matching MGCFCVectorPoissonDriver::Solve()'s
  // convention: the caller (cfc::CFC::SolveConformalFactor) calls
  // RetrieveSolution() separately once this returns.
  return;
}

// u_tilde/a_sq are sized at mesh-NGHOST depth (they're read pointwise here at
// arbitrary offset ngh, not assumed to match this driver's own shallower ngh_).

void MGCFCConformalFactorDriver::LoadMatterSource(const DvceArray5D<Real> &u_tilde,
                                                   int ngh) {
  // par_for's loop bound below uses pmy_pack_->nmb_thispack directly (the current
  // block count) rather than mglevels_'s own cached nmmb_ -- correct only if
  // coeff_'s "m" dimension is already sized to match. Under dynamic AMR, cfc.cpp
  // calls LoadMatterSource() before Solve() (whose own PrepareForAMR() normally
  // keeps coeff_'s size in sync via ReallocateForAMR()) -- without this call, the
  // first LoadMatterSource() after a regrid would write out of bounds into
  // coeff_'s stale (pre-regrid) size. ReallocateForAMR() is an idempotent
  // early-return no-op when nmmb_ already matches nmb_thispack, so calling it here
  // is safe regardless of whether Solve() already ran this cycle.
  mglevels_->ReallocateForAMR();
  auto &cm = mglevels_->CoeffAtLevel(mglevels_->GetNumberOfLevels()-1);
  int lngh = mglevels_->GetGhostCells();
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  int is = 0, ie = indcs.nx1 + 2*lngh - 1;
  int js = 0, je = indcs.nx2 + 2*lngh - 1;
  int ks = 0, ke = indcs.nx3 + 2*lngh - 1;
  // u_tilde is padded to depth ngh, not this driver's own (generally shallower)
  // lngh -- mirror Multigrid::LoadCoefficients' offset-aware indexing rather than
  // assuming the two depths match.
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
  // See LoadMatterSource's identical comment above.
  mglevels_->ReallocateForAMR();
  auto &cm = mglevels_->CoeffAtLevel(mglevels_->GetNumberOfLevels()-1);
  int lngh = mglevels_->GetGhostCells();
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  int is = 0, ie = indcs.nx1 + 2*lngh - 1;
  int js = 0, je = indcs.nx2 + 2*lngh - 1;
  int ks = 0, ke = indcs.nx3 + 2*lngh - 1;
  const int off = ngh - lngh;  // see LoadMatterSource above
  auto cm_d = cm.d_view;
  int nmmb = pmy_pack_->nmb_thispack;
  par_for("MGCFCConformalFactorDriver::LoadNonlinearCoefficient", DevExeSpace(),
          0, nmmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int mk, const int mj, const int mi) {
    cm_d(m, 1, mk, mj, mi) = a_sq(m, 0, mk+off, mj+off, mi+off);
  });
}

void MGCFCConformalFactorDriver::RetrieveSolution(DvceArray5D<Real> &dst) {
  // dst (psi) is mesh-NGHOST-deep, not this solver's own (generally shallower)
  // ngh_ -- passing GetGhostCells() here would make RetrieveResult's
  // dst_off = ngh - ngh_ collapse to 0 instead of (mesh NGHOST - ngh_), silently
  // mis-offsetting the copy. Matches gravity's and MGCFCVectorPoissonDriver's own
  // RetrieveResult calls, which both pass the mesh's true NGHOST.
  mglevels_->RetrieveResult(dst, 0, pmy_pack_->pmesh->mb_indcs.ng);
  return;
}

void MGCFCConformalFactorDriver::SeedInitialGuess(const DvceArray5D<Real> &guess,
                                                   int ngh) {
  mglevels_->LoadFinestData(guess, 0, ngh);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MGCFCConformalFactorDriver::TransferCoeffToRoot()
//! \brief MultigridDriver::TransferFromBlocksToRoot (multigrid_driver.cpp) aggregates
//! every rank's coarsest per-block cell into the distributed root grid (mgroot_) via
//! MPI_Allgatherv, but only for src_/u_ -- never coeff_. That transfer runs for any
//! multi-meshblock mesh (not AMR-specific), so mgroot_ needs its own Utilde/Ahat^2
//! populated the same way before the V-cycle can reach the root level. Duplicates the
//! relevant logic here rather than touching src/multigrid/, including the
//! octet-parented branch TransferFromBlocksToRoot itself has (blocks refined past the
//! root level write into their parent octet's Coeff() instead of the root grid).

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
      // Block refined past the root level -- write into its parent octet's
      // Coeff() instead (mirrors TransferFromBlocksToRoot's identical else-branch
      // for Src()/U()).
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
  // The above only populates mgroot_'s own finest internal level (the root grid can
  // itself span multiple V-cycle levels). Every coarser root level's coeff_ still
  // needs restricting from this one -- mgc_root->RestrictCoefficients() (called in
  // Solve(), right after RestrictCoeffOctets()) does that.
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MGCFCConformalFactorDriver::DebugReportDefectByLevel()
//! \brief Debug-only: reports this rank's worst |defect| cell at the finest
//! per-block level, split by whether it belongs to a root-level (unrefined) or a
//! refined MeshBlock. No MPI_Allreduce (one-shot diagnostic, not a convergence
//! criterion) -- a category is skipped if this rank owns no MeshBlocks of that
//! kind.

void MGCFCConformalFactorDriver::DebugReportDefectByLevel() {
  int ref_m, ref_k, ref_j, ref_i, ref_gid;
  DebugReportWorstDefect("", ref_m, ref_k, ref_j, ref_i, ref_gid);

  if (ref_gid >= 0) {
    // Dump the stencil inputs at the worst refined-block cell -- its own
    // U/Coeff and its 6 face-neighbor U values.
    auto u_d = mglevels_->GetCurrentData();
    auto u_h = Kokkos::create_mirror_view(u_d);
    Kokkos::deep_copy(u_h, u_d);
    auto &cm = mglevels_->CoeffAtLevel(mglevels_->GetNumberOfLevels()-1);
    auto coeff_h = Kokkos::create_mirror_view(cm.d_view);
    Kokkos::deep_copy(coeff_h, cm.d_view);

    std::cout << "CFC debug [rank " << global_variable::my_rank << "]: stencil at "
              << "worst REFINED cell (gid=" << ref_gid << ", m=" << ref_m
              << ", k=" << ref_k << ", j=" << ref_j << ", i=" << ref_i << "):"
              << " U=" << u_h(ref_m, 0, ref_k, ref_j, ref_i)
              << " Coeff0(Utilde)=" << coeff_h(ref_m, 0, ref_k, ref_j, ref_i)
              << " Coeff1(Ahat2)=" << coeff_h(ref_m, 1, ref_k, ref_j, ref_i)
              << " U(i-1)=" << u_h(ref_m, 0, ref_k, ref_j, ref_i-1)
              << " U(i+1)=" << u_h(ref_m, 0, ref_k, ref_j, ref_i+1)
              << " U(j-1)=" << u_h(ref_m, 0, ref_k, ref_j-1, ref_i)
              << " U(j+1)=" << u_h(ref_m, 0, ref_k, ref_j+1, ref_i)
              << " U(k-1)=" << u_h(ref_m, 0, ref_k-1, ref_j, ref_i)
              << " U(k+1)=" << u_h(ref_m, 0, ref_k+1, ref_j, ref_i)
              << std::endl;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MGCFCConformalFactorDriver::DebugReportWorstDefect(...)
//! \brief Debug-only: shared worst-cell finder used by DebugReportDefectByLevel and
//! DebugAnalyticResidualTest.

void MGCFCConformalFactorDriver::DebugReportWorstDefect(const std::string &label,
                                                         int &ref_m, int &ref_k,
                                                         int &ref_j, int &ref_i,
                                                         int &ref_gid) {
  mglevels_->CalculateDefectPack();
  auto def_d = mglevels_->GetCurrentDefect();
  auto def_h = Kokkos::create_mirror_view(def_d);
  Kokkos::deep_copy(def_h, def_d);

  int ngh = mglevels_->GetGhostCells();
  int ncx = def_h.extent_int(4), ncy = def_h.extent_int(3), ncz = def_h.extent_int(2);
  int is = ngh, ie = ncx - ngh - 1;
  int js = ngh, je = ncy - ngh - 1;
  int ks = ngh, ke = ncz - ngh - 1;
  int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
  int nmb = pmy_pack_->nmb_thispack;

  auto &size = pmy_pack_->pmb->mb_size;
  auto &gid_h = pmy_pack_->pmb->mb_gid.h_view;
  const auto loc = pmy_mesh_->lloc_eachmb;

  Real root_max = 0.0, ref_max = 0.0;
  int root_gid = -1;
  Real root_x1 = 0.0, root_x2 = 0.0, root_x3 = 0.0;
  Real ref_x1 = 0.0, ref_x2 = 0.0, ref_x3 = 0.0;
  ref_gid = -1;
  ref_m = ref_i = ref_j = ref_k = -1;

  for (int m = 0; m < nmb; ++m) {
    int gid = gid_h(m);
    bool refined = (loc[gid].level > locrootlevel_);
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          Real val = Kokkos::fabs(def_h(m, 0, k, j, i));
          Real &cur_max = refined ? ref_max : root_max;
          if (val > cur_max) {
            cur_max = val;
            Real x1v = CellCenterX(i-is, nx1, size.h_view(m).x1min, size.h_view(m).x1max);
            Real x2v = CellCenterX(j-js, nx2, size.h_view(m).x2min, size.h_view(m).x2max);
            Real x3v = CellCenterX(k-ks, nx3, size.h_view(m).x3min, size.h_view(m).x3max);
            if (refined) {
              ref_gid = gid; ref_x1 = x1v; ref_x2 = x2v; ref_x3 = x3v;
              ref_m = m; ref_i = i; ref_j = j; ref_k = k;
            } else {
              root_gid = gid; root_x1 = x1v; root_x2 = x2v; root_x3 = x3v;
            }
          }
        }
      }
    }
  }

  if (root_gid >= 0) {
    std::cout << "CFC debug [rank " << global_variable::my_rank << "]: " << label
              << "worst |defect| on ROOT blocks    = " << root_max << " at gid="
              << root_gid << " (x1,x2,x3)=(" << root_x1 << "," << root_x2 << ","
              << root_x3 << ")" << std::endl;
  }
  if (ref_gid >= 0) {
    std::cout << "CFC debug [rank " << global_variable::my_rank << "]: " << label
              << "worst |defect| on REFINED blocks = " << ref_max << " at gid="
              << ref_gid << " (x1,x2,x3)=(" << ref_x1 << "," << ref_x2 << ","
              << ref_x3 << ")" << std::endl;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MGCFCConformalFactorDriver::DebugAnalyticResidualTest()
//! \brief Debug-only, see mg_cfc_conformal_factor.hpp's doc comment. Seeds
//! delta_psi = psi_analytic - 1 at every cell of the finest per-block level,
//! including every ghost cell (evaluated at that cell's own physical position),
//! measures the residual with no smoother involved, then overwrites just the
//! ghost cells with one real ghost-communication round and measures again.

void MGCFCConformalFactorDriver::DebugAnalyticResidualTest(Driver *pdriver) {
  tov::PolytropeEOS eos(pin_);
  tov::TOVStar tov_star = tov::TOVStar::ConstructTOV(pin_, eos, false);

  auto u_d = mglevels_->GetCurrentData();
  auto u_h = Kokkos::create_mirror_view(u_d);
  Kokkos::deep_copy(u_h, u_d);

  int ngh = mglevels_->GetGhostCells();
  int ncx = u_h.extent_int(4), ncy = u_h.extent_int(3), ncz = u_h.extent_int(2);
  int is = ngh, ie = ncx - ngh - 1;
  int js = ngh, je = ncy - ngh - 1;
  int ks = ngh, ke = ncz - ngh - 1;
  int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
  int nmb = pmy_pack_->nmb_thispack;
  auto &size = pmy_pack_->pmb->mb_size;

  // Seed EVERY cell (0..ncx/y/z-1, i.e. interior AND every ghost depth), evaluating
  // the analytic solution at that cell's own physical position -- CellCenterX is
  // linear in the cell index, so passing i-is for i outside [is,ie] correctly
  // extrapolates into the ghost region rather than needing special-casing.
  for (int m = 0; m < nmb; ++m) {
    for (int k = 0; k < ncz; ++k) {
      Real x3v = CellCenterX(k-ks, nx3, size.h_view(m).x3min, size.h_view(m).x3max);
      for (int j = 0; j < ncy; ++j) {
        Real x2v = CellCenterX(j-js, nx2, size.h_view(m).x2min, size.h_view(m).x2max);
        for (int i = 0; i < ncx; ++i) {
          Real x1v = CellCenterX(i-is, nx1, size.h_view(m).x1min, size.h_view(m).x1max);
          Real r = std::sqrt(x1v*x1v + x2v*x2v + x3v*x3v);
          Real rho, p, mass, alp;
          tov_star.GetPrimitivesAtIsoPoint(eos, r, rho, p, mass, alp);
          Real psi = 1.0;
          if (r > 0.0) {
            Real r_schw = tov_star.FindSchwarzschildR(r, mass);
            psi = std::sqrt(r_schw / r);
          }
          u_h(m, 0, k, j, i) = psi - 1.0;
        }
      }
    }
  }
  Kokkos::deep_copy(u_d, u_h);

  int m0, k0, j0, i0, g0;
  DebugReportWorstDefect("analytic test, ANALYTIC ghosts (no comm, no smoother): ",
                         m0, k0, j0, i0, g0);
  DebugDumpInterfaceDefect("BEFORE ghost-comm: ");

  // One real ghost-communication round, bypassing the smoother entirely -- mirrors
  // SetMGTaskListToFiner's flag==2 "final boundary exchange" block
  // (multigrid_tasks.cpp), called directly since this is a one-shot diagnostic, not
  // part of a real V-cycle. ClearSend/ClearRecv leave MPI state clean for the real
  // SolveMG() call that follows this in Solve().
  pmg = mglevels_;
  FillCoarseBoundary(nullptr, 0);
  StartReceive(nullptr, 0);
  SendBoundary(nullptr, 0);
  while (RecvBoundary(nullptr, 0) == TaskStatus::incomplete) {}
  PhysicalBoundary(nullptr, 0);
  ProlongateFCBoundary(nullptr, 0);

  DebugDumpCoarseBuf();

  ClearSend(nullptr, 0);
  ClearRecv(nullptr, 0);

  DebugReportWorstDefect("analytic test, AFTER one MG ghost-comm round: ",
                         m0, k0, j0, i0, g0);
  DebugDumpInterfaceDefect("AFTER ghost-comm: ");
  DebugReportWorstSolutionError("analytic test, AFTER ghost-comm (no smoother yet): ");

  // Run exactly one V-cycle from the analytically-seeded state above, then compare
  // both the defect and the solution value against the analytic truth -- shows how
  // much a single relaxation sweep moves the solution.
  SolveVCycle(pdriver, npresmooth_, npostsmooth_);
  Kokkos::fence();
  DebugReportWorstDefect("analytic test, AFTER one V-cycle (smoother applied): ",
                         m0, k0, j0, i0, g0);
  DebugReportWorstSolutionError("analytic test, AFTER one V-cycle: ");
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MGCFCConformalFactorDriver::DebugReportWorstSolutionError(...)
//! \brief Debug-only companion to DebugReportWorstDefect: reports the worst
//! |delta_psi - analytic| (the solution value error, not the equation's residual),
//! split root/refined, same classification pattern.

void MGCFCConformalFactorDriver::DebugReportWorstSolutionError(
    const std::string &label) {
  tov::PolytropeEOS eos(pin_);
  tov::TOVStar tov_star = tov::TOVStar::ConstructTOV(pin_, eos, false);

  auto u_d = mglevels_->GetCurrentData();
  auto u_h = Kokkos::create_mirror_view(u_d);
  Kokkos::deep_copy(u_h, u_d);

  int ngh = mglevels_->GetGhostCells();
  int ncx = u_h.extent_int(4), ncy = u_h.extent_int(3), ncz = u_h.extent_int(2);
  int is = ngh, ie = ncx - ngh - 1;
  int js = ngh, je = ncy - ngh - 1;
  int ks = ngh, ke = ncz - ngh - 1;
  int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
  int nmb = pmy_pack_->nmb_thispack;

  auto &size = pmy_pack_->pmb->mb_size;
  auto &gid_h = pmy_pack_->pmb->mb_gid.h_view;
  const auto loc = pmy_mesh_->lloc_eachmb;

  Real root_max = 0.0, ref_max = 0.0;
  int root_gid = -1, ref_gid = -1;
  Real root_x1 = 0.0, root_x2 = 0.0, root_x3 = 0.0;
  Real ref_x1 = 0.0, ref_x2 = 0.0, ref_x3 = 0.0;

  for (int m = 0; m < nmb; ++m) {
    int gid = gid_h(m);
    bool refined = (loc[gid].level > locrootlevel_);
    for (int k = ks; k <= ke; ++k) {
      Real x3v = CellCenterX(k-ks, nx3, size.h_view(m).x3min, size.h_view(m).x3max);
      for (int j = js; j <= je; ++j) {
        Real x2v = CellCenterX(j-js, nx2, size.h_view(m).x2min, size.h_view(m).x2max);
        for (int i = is; i <= ie; ++i) {
          Real x1v = CellCenterX(i-is, nx1, size.h_view(m).x1min, size.h_view(m).x1max);
          Real r = std::sqrt(x1v*x1v + x2v*x2v + x3v*x3v);
          Real rho, p, mass, alp;
          tov_star.GetPrimitivesAtIsoPoint(eos, r, rho, p, mass, alp);
          Real psi = 1.0;
          if (r > 0.0) {
            Real r_schw = tov_star.FindSchwarzschildR(r, mass);
            psi = std::sqrt(r_schw / r);
          }
          Real val = Kokkos::fabs(u_h(m, 0, k, j, i) - (psi - 1.0));
          Real &cur_max = refined ? ref_max : root_max;
          if (val > cur_max) {
            cur_max = val;
            if (refined) {
              ref_gid = gid; ref_x1 = x1v; ref_x2 = x2v; ref_x3 = x3v;
            } else {
              root_gid = gid; root_x1 = x1v; root_x2 = x2v; root_x3 = x3v;
            }
          }
        }
      }
    }
  }

  if (root_gid >= 0) {
    std::cout << "CFC debug [rank " << global_variable::my_rank << "]: " << label
              << "worst |u-analytic| on ROOT blocks    = " << root_max << " at gid="
              << root_gid << " (x1,x2,x3)=(" << root_x1 << "," << root_x2 << ","
              << root_x3 << ")" << std::endl;
  }
  if (ref_gid >= 0) {
    std::cout << "CFC debug [rank " << global_variable::my_rank << "]: " << label
              << "worst |u-analytic| on REFINED blocks = " << ref_max << " at gid="
              << ref_gid << " (x1,x2,x3)=(" << ref_x1 << "," << ref_x2 << ","
              << ref_x3 << ")" << std::endl;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MGCFCConformalFactorDriver::DebugDumpCoarseBuf()
//! \brief Debug-only: investigates whether ProlongateFCMG's coarse-face transverse
//! gradient (multigrid_bvals.cpp) reads a stale, self-restricted coarse_buf_ slot
//! instead of the true coarse-neighbor value at a block's far transverse edge.
//! Dumps cbuf's full transverse row at the coarse-facing index for every
//! high-child block with a real coarser +x1 neighbor, alongside what
//! FillCoarseMG's own self-restriction formula would give at each slot.

void MGCFCConformalFactorDriver::DebugDumpCoarseBuf() {
  auto *pbval = mglevels_->pbval;
  auto cbuf_d = pbval->coarse_buf_;
  auto cbuf_h = Kokkos::create_mirror_view(cbuf_d);
  Kokkos::deep_copy(cbuf_h, cbuf_d);

  auto u_d = mglevels_->GetCurrentData();
  auto u_h = Kokkos::create_mirror_view(u_d);
  Kokkos::deep_copy(u_h, u_d);

  int ngh_l = mglevels_->GetGhostCells();
  int shift = mglevels_->GetLevelShift();
  int ncells_l = mglevels_->GetSize() >> shift;
  int half = ncells_l / 2;

  int nmb = pmy_pack_->nmb_thispack;
  auto &gid_h = pmy_pack_->pmb->mb_gid.h_view;
  const auto loc = pmy_mesh_->lloc_eachmb;
  auto &nghbr_h = pmy_pack_->pmb->nghbr;
  auto &mblev_h = pmy_pack_->pmb->mb_lev;
  int nnghbr = pmy_pack_->pmb->nnghbr;

  for (int m = 0; m < nmb; ++m) {
    int gid = gid_h(m);
    if (loc[gid].level <= locrootlevel_) continue;  // only refined blocks
    int child_x = static_cast<int>(loc[gid].lx1) & 1;
    int child_y = static_cast<int>(loc[gid].lx2) & 1;
    int child_z = static_cast<int>(loc[gid].lx3) & 1;

    // A fine block's single coarser neighbor is registered at slot
    // NeighborIndex(n,0,0,myfx2,myfx3) -- using this block's own child parity as
    // the subface index, not (0,0) (meshblock.cpp::SetNeighbors, "neighbor at
    // coarser level" branches).
    int n = NeighborIndex(1, 0, 0, child_y, child_z);
    if (n < 0 || n >= nnghbr) continue;
    if (nghbr_h.h_view(m, n).gid < 0) continue;
    int nlev = nghbr_h.h_view(m, n).lev;
    if (nlev >= mblev_h.h_view(m)) continue;  // only if +x1 neighbor is actually coarser

    int si = ngh_l + half;
    int sk = ngh_l;
    std::cout << "CFC debug [rank " << global_variable::my_rank << "]: coarse_buf_ "
              << "dump gid=" << gid << " (child_y=" << child_y << ",child_z=" << child_z
              << ") +x1 coarser neighbor, si=" << si << " sk=" << sk << ":";
    for (int sj = ngh_l - 1; sj <= ngh_l + half; ++sj) {
      std::cout << " cbuf[sj=" << sj << "]=" << cbuf_h(m, 0, sk, sj, si);
    }
    std::cout << std::endl;

    // Self-restriction comparison: FillCoarseMG's own face-average formula (this
    // block's own u, not the neighbor's), evaluated at the same (sk, sj, si) slots
    // it would have written before RecvAndUnpackMG ran. Only defined for sj in
    // [ngh_l, ngh_l+half-1] (FillCoarseMG's own valid range).
    std::cout << "CFC debug [rank " << global_variable::my_rank << "]: self-restrict "
              << "gid=" << gid << " (this block's own +x1 face average) fi="
              << (ngh_l + ncells_l - 1) << ":";
    int fi = ngh_l + ncells_l - 1;
    for (int sj = ngh_l; sj <= ngh_l + half; ++sj) {
      int fj = ngh_l + 2*(sj - ngh_l);
      int fk = ngh_l;  // sk = ngh_l fixed above -> c1 = 0 -> fk = ngh_l
      if (fj+1 >= u_h.extent_int(3) || fk+1 >= u_h.extent_int(2)) {
        std::cout << " self[sj=" << sj << "]=<out-of-range>";
        continue;
      }
      Real self_val = 0.25*(u_h(m,0,fk,  fj,  fi) + u_h(m,0,fk,  fj+1,fi) +
                             u_h(m,0,fk+1,fj,  fi) + u_h(m,0,fk+1,fj+1,fi));
      std::cout << " self[sj=" << sj << "]=" << self_val;
    }
    std::cout << std::endl;

    // Compare the actual post-prolongation +x1 ghost value (u_h was mirrored after
    // ProlongateFCBoundary ran) against the true analytic psi-1 at each fine
    // cell's physical position.
    int fig = ngh_l + ncells_l;
    tov::PolytropeEOS eos2(pin_);
    tov::TOVStar tov_star2 = tov::TOVStar::ConstructTOV(pin_, eos2, false);
    auto &size = pmy_pack_->pmb->mb_size;
    int ncx = u_h.extent_int(4), ncy = u_h.extent_int(3), ncz = u_h.extent_int(2);
    int is = ngh_l, ie = ncx - ngh_l - 1, js = ngh_l, je = ncy - ngh_l - 1;
    int ks = ngh_l, ke = ncz - ngh_l - 1;
    int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
    std::cout << "CFC debug [rank " << global_variable::my_rank << "]: ghost-vs-"
              << "analytic gid=" << gid << " fig=" << fig << ":";
    for (int fk = ngh_l; fk <= ngh_l + 1; ++fk) {
      Real x3v = CellCenterX(fk-ks, nx3, size.h_view(m).x3min, size.h_view(m).x3max);
      for (int fj = ngh_l; fj <= ngh_l + ncells_l - 1; ++fj) {
        Real x1v = CellCenterX(fig-is, nx1, size.h_view(m).x1min, size.h_view(m).x1max);
        Real x2v = CellCenterX(fj-js, nx2, size.h_view(m).x2min, size.h_view(m).x2max);
        Real r = std::sqrt(x1v*x1v + x2v*x2v + x3v*x3v);
        Real rho, p, mass, alp;
        tov_star2.GetPrimitivesAtIsoPoint(eos2, r, rho, p, mass, alp);
        Real r_schw = tov_star2.FindSchwarzschildR(r, mass);
        Real analytic = std::sqrt(r_schw / r) - 1.0;
        Real actual = u_h(m, 0, fk, fj, fig);
        std::cout << " [fk=" << fk << ",fj=" << fj << "] actual=" << actual
                  << " analytic=" << analytic << " diff=" << (actual-analytic);
      }
    }
    std::cout << std::endl;

    // A corner child of the refined patch may also border a coarser neighbor in
    // +x2 -- check that ghost fill too.
    int n2 = NeighborIndex(0, 1, 0, child_x, child_z);
    if (n2 >= 0 && n2 < nnghbr && nghbr_h.h_view(m, n2).gid >= 0 &&
        nghbr_h.h_view(m, n2).lev < mblev_h.h_view(m)) {
      int fjg = ngh_l + ncells_l;
      std::cout << "CFC debug [rank " << global_variable::my_rank << "]: ghost-vs-"
                << "analytic gid=" << gid << " (+x2 neighbor) fjg=" << fjg << ":";
      for (int fk = ngh_l; fk <= ngh_l + 1; ++fk) {
        Real x3v = CellCenterX(fk-ks, nx3, size.h_view(m).x3min, size.h_view(m).x3max);
        for (int fi = ngh_l; fi <= ngh_l + ncells_l - 1; ++fi) {
          Real x1v = CellCenterX(fi-is, nx1, size.h_view(m).x1min, size.h_view(m).x1max);
          Real x2v = CellCenterX(fjg-js, nx2, size.h_view(m).x2min, size.h_view(m).x2max);
          Real r = std::sqrt(x1v*x1v + x2v*x2v + x3v*x3v);
          Real rho, p, mass, alp;
          tov_star2.GetPrimitivesAtIsoPoint(eos2, r, rho, p, mass, alp);
          Real r_schw = tov_star2.FindSchwarzschildR(r, mass);
          Real analytic = std::sqrt(r_schw / r) - 1.0;
          Real actual = u_h(m, 0, fk, fjg, fi);
          std::cout << " [fk=" << fk << ",fi=" << fi << "] actual=" << actual
                    << " analytic=" << analytic << " diff=" << (actual-analytic);
        }
      }
      std::cout << std::endl;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MGCFCConformalFactorDriver::DebugDumpInterfaceDefect(const std::string&)
//! \brief Debug-only: DebugReportWorstDefect's "worst cell in the whole refined
//! patch" is contaminated by the star's own density-profile features moving the
//! worst-cell location between runs, so this instead reports the worst |defect|
//! restricted to the single layer of fine cells immediately adjacent to a
//! coarse-fine +x1 interface (same block-selection logic as DebugDumpCoarseBuf), at
//! a fixed relative location every time. Called twice from
//! DebugAnalyticResidualTest (before/after the real ghost-comm round) so the same
//! physical cells' defect can be compared directly.

void MGCFCConformalFactorDriver::DebugDumpInterfaceDefect(const std::string &label) {
  mglevels_->CalculateDefectPack();
  auto def_d = mglevels_->GetCurrentDefect();
  auto def_h = Kokkos::create_mirror_view(def_d);
  Kokkos::deep_copy(def_h, def_d);

  int ngh_l = mglevels_->GetGhostCells();
  int shift = mglevels_->GetLevelShift();
  int ncells_l = mglevels_->GetSize() >> shift;

  int nmb = pmy_pack_->nmb_thispack;
  auto &gid_h = pmy_pack_->pmb->mb_gid.h_view;
  const auto loc = pmy_mesh_->lloc_eachmb;
  auto &nghbr_h = pmy_pack_->pmb->nghbr;
  auto &mblev_h = pmy_pack_->pmb->mb_lev;
  int nnghbr = pmy_pack_->pmb->nnghbr;

  Real ifmax = 0.0;
  int if_gid = -1, if_fj = -1, if_fk = -1;
  Real sumsq = 0.0;
  int count = 0;
  int fi = ngh_l + ncells_l - 1;  // last real interior cell, adjacent to the +x1 ghost
  for (int m = 0; m < nmb; ++m) {
    int gid = gid_h(m);
    if (loc[gid].level <= locrootlevel_) continue;
    int child_y = static_cast<int>(loc[gid].lx2) & 1;
    int child_z = static_cast<int>(loc[gid].lx3) & 1;
    int n = NeighborIndex(1, 0, 0, child_y, child_z);
    if (n < 0 || n >= nnghbr) continue;
    if (nghbr_h.h_view(m, n).gid < 0) continue;
    if (nghbr_h.h_view(m, n).lev >= mblev_h.h_view(m)) continue;

    for (int fk = ngh_l; fk <= ngh_l + ncells_l - 1; ++fk) {
      for (int fj = ngh_l; fj <= ngh_l + ncells_l - 1; ++fj) {
        Real val = Kokkos::fabs(def_h(m, 0, fk, fj, fi));
        // RMS over the interface layer is a resolution-fair metric a plain max
        // isn't: max-of-N-samples grows with N from extreme-value statistics
        // alone, and doubling resolution quadruples the transverse cell count.
        sumsq += val*val;
        ++count;
        if (val > ifmax) {
          ifmax = val; if_gid = gid; if_fj = fj; if_fk = fk;
        }
      }
    }
  }
  if (if_gid >= 0) {
    Real rms = std::sqrt(sumsq / count);
    std::cout << "CFC debug [rank " << global_variable::my_rank << "]: " << label
              << "worst |defect| on the +x1-interface cell layer = " << ifmax
              << " at gid=" << if_gid << " (fj=" << if_fj << ",fk=" << if_fk << ")"
              << " RMS over layer (n=" << count << ") = " << rms
              << std::endl;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MGCFCConformalFactorDriver::DebugDumpRootCoeffUnderOctet()
//! \brief Debug-only: checks whether MultigridDriver::RestrictCoeffOctets() actually
//! pushes an octet-level-0's Coeff() down into mgroot_'s own corresponding
//! root-level cell (unlike the generic RestrictOctets(), which has an explicit
//! "octets to root grid" branch for Src()/U(), RestrictCoeffOctets() only mirrors
//! the "fine octet to coarser octet" branch). Dumps, for every level-0 octet,
//! mgroot_'s actual Coeff() at the corresponding root cell alongside
//! RestrictOneCoeff's volume-averaged expectation from that octet's own Coeff().

void MGCFCConformalFactorDriver::DebugDumpRootCoeffUnderOctet() {
  if (nreflevel_ <= 0) return;

  auto *mgc_root = static_cast<MGCFCConformalFactor*>(mgroot_);
  int ngh = mgc_root->GetGhostCells();
  auto &coeff_root = mgc_root->CoeffAtLevel(mgc_root->GetNumberOfLevels()-1);
  auto coeff_root_h = Kokkos::create_mirror_view(coeff_root.d_view);
  Kokkos::deep_copy(coeff_root_h, coeff_root.d_view);

  for (int o = 0; o < noctets_[0]; ++o) {
    MGOctet &oct = octets_[0][o];
    const LogicalLocation &oloc = oct.loc;
    int ri = static_cast<int>(oloc.lx1);
    int rj = static_cast<int>(oloc.lx2);
    int rk = static_cast<int>(oloc.lx3);
    std::cout << "CFC debug [rank " << global_variable::my_rank << "]: root coeff_ "
              << "under octet (lx1,lx2,lx3)=(" << ri << "," << rj << "," << rk << "):";
    for (int c = 0; c < ncoeff_; ++c) {
      Real actual = coeff_root_h(0, c, rk+ngh, rj+ngh, ri+ngh);
      Real expected = RestrictOneCoeff(oct, c, ngh, ngh, ngh);
      std::cout << " c" << c << ": actual=" << actual << " expected(from octet avg)="
                << expected << " diff=" << (actual - expected);
    }
    std::cout << std::endl;
  }
  return;
}

// Octet-scale Newton-Gauss-Seidel smoothing, exactly the same math as
// MGCFCConformalFactor::SmoothPack (per-level) above -- ConformalFactorRHS is
// reused verbatim; only the Laplacian stencil and the u/src/coeff access pattern
// change (oct.U/Src/Coeff instead of u_[lev]/src_[lev]/coeff_[lev] views).
void MGCFCConformalFactorDriver::SmoothOctet(MGOctet &oct, int rlev, int color) {
  if (use_psi5_source_) { SmoothOctetImpl<true>(oct, rlev, color); }
  else { SmoothOctetImpl<false>(oct, rlev, color); }
}

template <bool UsePsi5>
void MGCFCConformalFactorDriver::SmoothOctetImpl(MGOctet &oct, int rlev, int color) {
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
        ConformalFactorRHS<UsePsi5>(u_old, oct.Coeff(0,k,j,i), oct.Coeff(1,k,j,i),
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
  if (use_psi5_source_) { CalculateDefectOctetImpl<true>(oct, rlev); }
  else { CalculateDefectOctetImpl<false>(oct, rlev); }
}

template <bool UsePsi5>
void MGCFCConformalFactorDriver::CalculateDefectOctetImpl(MGOctet &oct, int rlev) {
  int ngh = mgroot_->GetGhostCells();
  Real root_dx = mgroot_->GetRootDx();
  Real dx = root_dx / static_cast<Real>(1 << rlev);
  Real idx2 = 1.0 / (dx*dx);
  for (int k = ngh; k <= ngh+1; ++k) {
    for (int j = ngh; j <= ngh+1; ++j) {
      for (int i = ngh; i <= ngh+1; ++i) {
        Real rhs, drhs_du;
        ConformalFactorRHS<UsePsi5>(oct.U(0,k,j,i), oct.Coeff(0,k,j,i), oct.Coeff(1,k,j,i),
                                     &rhs, &drhs_du);
        Real lap = OctConformalFactorLap(oct, k, j, i);
        oct.Def(0,k,j,i) = (rhs + oct.Src(0,k,j,i)) - lap*idx2;
      }
    }
  }
}

void MGCFCConformalFactorDriver::CalculateFASRHSOctet(MGOctet &oct, int rlev) {
  if (use_psi5_source_) { CalculateFASRHSOctetImpl<true>(oct, rlev); }
  else { CalculateFASRHSOctetImpl<false>(oct, rlev); }
}

template <bool UsePsi5>
void MGCFCConformalFactorDriver::CalculateFASRHSOctetImpl(MGOctet &oct, int rlev) {
  int ngh = mgroot_->GetGhostCells();
  Real root_dx = mgroot_->GetRootDx();
  Real dx = root_dx / static_cast<Real>(1 << rlev);
  Real idx2 = 1.0 / (dx*dx);
  for (int k = ngh; k <= ngh+1; ++k) {
    for (int j = ngh; j <= ngh+1; ++j) {
      for (int i = ngh; i <= ngh+1; ++i) {
        Real rhs, drhs_du;
        ConformalFactorRHS<UsePsi5>(oct.U(0,k,j,i), oct.Coeff(0,k,j,i), oct.Coeff(1,k,j,i),
                                     &rhs, &drhs_du);
        Real lap = OctConformalFactorLap(oct, k, j, i);
        oct.Src(0,k,j,i) += lap*idx2 - rhs;
      }
    }
  }
}
