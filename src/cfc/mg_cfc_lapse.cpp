//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mg_cfc_lapse.cpp
//! \brief implementation of MGCFCLapse[Driver]
//!
//! Sign convention: see mg_cfc_conformal_factor.cpp's file header (same stencil,
//! real_Laplacian(u) = -lap(u)/dx^2). Eq. 74 (Gmunu 2021) is
//!   Delta(alpha*psi) = (alpha*psi) * [2*pi*(Utilde+2*Stilde)*psi^-2
//!                                     + (7/8)*Ahat^2*psi^-8] =: (alpha*psi)*K(x),
//! with alpha*psi = u+v0 (u = delta_(alpha*psi), v0 = alpha0*psi0, the analytic
//! trumpet background's alpha0*psi0 product -- 1 everywhere unless <cfc>
//! puncture_enabled). K(x) depends only on already-fixed fields (psi, Ahat^2 from
//! earlier steps; Utilde+2*Stilde is this equation's own known source), not on u --
//! affine in u, no Newton iteration needed (mg_cfc_lapse.hpp).
//!
//! Puncture regularization (CFC_PUNCTURE_TDE_PLAN.md Sec 3.7/Sec 5 Phase A item 5):
//! unlike eq. 73 (psi), this equation is HOMOGENEOUS in the physical field
//! (Delta v = K(x)*v, no separate additive source -- matter is folded entirely into
//! K(x)). The background v0 independently satisfies the vacuum equation
//! Delta v0 = K0*v0, K0 = (7/8)*Ahat0^2*psi0^-8 (no matter term, exact for the
//! analytic trumpet solution). Writing v = v0+u and subtracting:
//!   Delta u = Delta v - Delta v0 = K(x)*v - K0*v0 = K(x)*u + v0*(K(x)-K0)
//! so the DEVIATION equation gains a genuine additive source term
//! S := v0*(K(x)-K0) that the unregularized (flat-background) code has no channel
//! for at all. Computing S via a direct K(x)-K0 subtraction would reintroduce
//! exactly the cancellation this exercise avoids (K(x) and K0 agree to relative
//! order x=delta_psi/psi0 near the puncture) -- instead, expand and apply the same
//! psi^-8-psi0^-8 factoring already validated in mg_cfc_conformal_factor.cpp:
//!   K(x)-K0 = 2*pi*(Utilde+2*Stilde)*psi^-2 + (7/8)*DeltaAhat^2*psi^-8
//!             - (7/8)*Ahat0^2*reg8(x)
//!   reg8(x) = psi0^-8 * x*P8(x)/(1+x)^8,  x = delta_psi/psi0  (psi's OWN deviation
//!             ratio -- psi is already solved/fixed by the time this solve runs, so
//!             x is a known input here, not this equation's unknown)
//!   P8(x)   = 8+28x+56x^2+70x^3+56x^4+28x^5+8x^6+x^7  (same Horner pattern as
//!             mg_cfc_conformal_factor.cpp's P7, one degree higher)
//!   DeltaAhat^2 = Ahat^2_total - Ahat0^2 (safe direct subtraction, Ahat0^2 is
//!             bounded/O(1), doesn't diverge at the puncture)
//!   ==> S = v0*[2*pi*(Utilde+2*Stilde)*psi^-2 + (7/8)*DeltaAhat^2*psi^-8
//!             - (7/8)*Ahat0^2*reg8(x)]
//! With <cfc> puncture_enabled=false (psi0=1, v0=1, Ahat0^2=0 identically), S
//! reduces to exactly K(x), and Delta u = K(x)*u + S becomes Delta u = K(x)*(u+1),
//! i.e. today's formula -- confirms the derivation.
//!
//! Per-point update: F(u) = lap(u) + dx^2*[K(x)*u + S] = 0. No Newton-Jacobian
//! analog to worry about (this equation stays affine): S is precomputed once at
//! load time, same as K(x) already is, so F'(u) = 6 + dx^2*K(x) is UNCHANGED from
//! the pre-regularization formula -- the "Newton" step is still an exact one-step
//! Gauss-Seidel solve.
//!
//! K(x) and S are each precomputed ONCE, at the finest level, from
//! psi/Ahat^2/psi0/Ahat0^2/Utilde+2*Stilde (LoadReactionCoefficient below); only
//! the two combined scalars are then carried through coeff_/RestrictCoefficients()
//! to coarser levels (ncoeff_=2) -- restricting the raw ingredients separately and
//! recombining at each level would be FAS-inconsistent (restrict(f(a,b)) !=
//! f(restrict(a), restrict(b)) for K(x)/S's nonlinear psi^-2/psi^-8 combinations).
//!
//! NOTE: bit-for-bit reduction to the pre-regularization code is NOT expected in
//! the puncture_enabled=false case, even though S reduces to exactly K(x) there --
//! `kx*u+kx` and `kx*(u+1.0)` are different floating-point operation sequences
//! (multiply-then-add vs add-then-multiply) and are not guaranteed bit-identical in
//! IEEE754 in general. Validate via numerical closeness instead (see
//! CFC_PUNCTURE_TDE_PLAN.md Sec 5 Phase A item 5's stage-5b validation notes).
//!
//! SmoothPack/CalculateDefectPack must also add src(m,0,k,j,i) (the FAS tau-
//! correction CalculateFASRHSPack accumulates) -- omitting it silently discards
//! the coarse-grid correction from every level below the finest.

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

// Octet-indexed counterpart of LapseLap above, mirroring
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
  // See MGCFCConformalFactor's ctor -- coeff_/ncoeff_ are never allocated by the
  // base Multigrid ctor, so we do it ourselves.
  ncoeff_ = 2;  // channel 0 = K(x), channel 1 = S (Sec 3.7 puncture
                // regularization source term), both precomputed at the
                // finest level
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
  // FAS tau-correction accumulator, see this file's header comment above -- must
  // be added here or the coarse-grid correction is silently dropped.
  auto src = src_[lev].d_view;
  par_for("MGCFCLapse::SmoothPack", DevExeSpace(), 0, nmmb_-1, ks, ke, js, je,
  KOKKOS_LAMBDA(const int m, const int k, const int j) {
    Real dx = (rlev <= 0) ? brdx(m) * static_cast<Real>(1<<(-rlev))
                          : brdx(m) / static_cast<Real>(1<<rlev);
    Real dx2 = dx * dx;
    const int c = (c0 + k + j) & 1;
    for (int i = is + c; i <= ie; i += 2) {
      Real kx = coeff(m,0,k,j,i);  // K(x), precomputed at load time
      Real s = coeff(m,1,k,j,i);   // S, Sec 3.7 regularization source term
      Real lap = LapseLap(u, m, k, j, i);
      Real u_old = u(m,0,k,j,i);
      Real fval = lap + dx2*(kx*u_old + s) - dx2*src(m,0,k,j,i);
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
  // received, corrupting the correction chain for every level further down.
  auto src = src_[lev].d_view;
  par_for("MGCFCLapse::CalculateDefectPack", DevExeSpace(), 0, nmmb_-1,
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real dx = (rlev <= 0) ? brdx(m) * static_cast<Real>(1<<(-rlev))
                          : brdx(m) / static_cast<Real>(1<<rlev);
    Real idx2 = 1.0 / (dx*dx);
    Real kx = coeff(m,0,k,j,i);  // K(x), precomputed at load time
    Real s = coeff(m,1,k,j,i);   // S, Sec 3.7 regularization source term
    Real lap = LapseLap(u, m, k, j, i);
    // def = (RHS(u) + src) - lap(u)*idx2, RHS(u) = -(kx*u+s)
    // (F(u) = lap(u) - dx^2*(RHS(u) + src)).
    def(m,0,k,j,i) = (-(kx*u(m,0,k,j,i) + s) + src(m,0,k,j,i)) - lap*idx2;
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
    Real kx = coeff(m,0,k,j,i);  // K(x), precomputed at load time
    Real s = coeff(m,1,k,j,i);   // S, Sec 3.7 regularization source term
    Real lap = LapseLap(u, m, k, j, i);
    // src += lap(u)*idx2 - RHS(u) = lap(u)*idx2 + (kx*u+s).
    src(m,0,k,j,i) += lap*idx2 + (kx*u(m,0,k,j,i) + s);
  });
}


//----------------------------------------------------------------------------------------
//! \fn MGCFCLapseDriver::MGCFCLapseDriver(...)
//! \brief nvar_ = 1, ncoeff_ = 2 (K(x), S -- both precomputed at the finest level,
//! see this file's header comment); mg_robin boundary conditions by default (Gmunu
//! eq. 78, isolated/asymptotically-flat falloff).

MGCFCLapseDriver::MGCFCLapseDriver(MeshBlockPack *pmbp, ParameterInput *pin)
    : MultigridDriver(pmbp, 1) {
  ncoeff_ = 2;
  eps_ = pin->GetOrAddReal("cfc", "mg_threshold", 1.0e-10);
  fshowdef_ = pin->GetOrAddInteger("cfc", "mg_verbose", 0);
  mg_verbose_ = fshowdef_;
  full_multigrid_ = false;
  // See MGCFCConformalFactorDriver's identical comment -- AMR-refined meshes need
  // more smoothing per level than the base-class default of 1.
  npresmooth_ = pin->GetOrAddInteger("cfc", "mg_npresmooth", npresmooth_);
  npostsmooth_ = pin->GetOrAddInteger("cfc", "mg_npostsmooth", npostsmooth_);

  // Outer (non-periodic, non-reflecting) faces default to BoundaryFlag::mg_robin --
  // see mg_cfc_conformal_factor.cpp's constructor comment for the full rationale
  // (including why mg_multipole would divide by zero here). <cfc> mg_outer_bc
  // ("robin" [default] or "zerofixed") allows falling back to the old
  // Dirichlet-zero behavior.
  //
  // Faces where the *mesh* itself is reflecting still need BoundaryFlag::mg_zerograd,
  // not plain BoundaryFlag::reflect (same reasoning as mg_cfc_conformal_factor.cpp).
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
  // Octet-hierarchy coefficient restriction and ordering, same reasoning as
  // MGCFCConformalFactorDriver::Solve.
  RestrictCoeffOctets();
  mgroot_->RestrictCoefficients();

  SetupMultigrid(dt, false);

  // See MGCFCConformalFactorDriver::Solve's identical comment -- no mg_multipole
  // face is ever set, so the multipole setup call is skipped.

  SolveMG(pdriver);
  Kokkos::fence();

  // No self-retrieve here, matching MGCFCVectorPoissonDriver::Solve()'s
  // convention: the caller (cfc::CFC::SolveLapse) calls RetrieveSolution()
  // separately once this returns.
  return;
}

// Evaluates K(x) = LapseReactionCoeff(...) and the Sec 3.7 regularization source
// term S = v0*(K(x)-K0) once per point, at the finest level, and writes both values
// into coeff_ (see this file's header comment for the full derivation).
// u_plus_2s_tilde/delta_psi/a_sq/u_psi0/a0_sq/u_alpha0_psi0 are padded to depth ngh
// (mesh-NGHOST, per cfc.cpp). delta_psi stores psi - psi0 (cfc::CFC::delta_psi,
// cfc.hpp); the physical psi LapseReactionCoeff needs is reconstructed (+u_psi0, 1.0
// everywhere unless <cfc> puncture_enabled).
void MGCFCLapseDriver::LoadReactionCoefficient(
    const DvceArray5D<Real> &u_plus_2s_tilde, const DvceArray5D<Real> &delta_psi,
    const DvceArray5D<Real> &u_psi0, const DvceArray5D<Real> &a_sq,
    const DvceArray5D<Real> &a0_sq, const DvceArray5D<Real> &u_alpha0_psi0, int ngh) {
  // See MGCFCConformalFactorDriver::LoadMatterSource's identical comment.
  mglevels_->ReallocateForAMR();
  auto &cm = mglevels_->CoeffAtLevel(mglevels_->GetNumberOfLevels()-1);
  int lngh = mglevels_->GetGhostCells();
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  int is = 0, ie = indcs.nx1 + 2*lngh - 1;
  int js = 0, je = indcs.nx2 + 2*lngh - 1;
  int ks = 0, ke = indcs.nx3 + 2*lngh - 1;
  const int off = ngh - lngh;
  auto cm_d = cm.d_view;
  int nmmb = pmy_pack_->nmb_thispack;
  par_for("MGCFCLapseDriver::LoadReactionCoefficient", DevExeSpace(),
          0, nmmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int mk, const int mj, const int mi) {
    Real u2s = u_plus_2s_tilde(m, 0, mk+off, mj+off, mi+off);
    Real dpsi = delta_psi(m, 0, mk+off, mj+off, mi+off);
    Real psi0 = u_psi0(m, 0, mk+off, mj+off, mi+off);
    Real psi_known = dpsi + psi0;
    Real ahat_sq = a_sq(m, 0, mk+off, mj+off, mi+off);
    Real a0sq = a0_sq(m, 0, mk+off, mj+off, mi+off);
    Real v0 = u_alpha0_psi0(m, 0, mk+off, mj+off, mi+off);
    cm_d(m, 0, mk, mj, mi) = LapseReactionCoeff(u2s, psi_known, ahat_sq);

    // Sec 3.7 regularization source term S = v0*(K(x)-K0), built via the safe
    // reg8(x) factoring -- see this file's header comment. delta_a_sq is a safe
    // direct subtraction (Ahat0^2 bounded/O(1), unlike psi0 doesn't diverge at the
    // puncture); reg8(x) is the cancellation-free replacement for
    // (psi^-8 - psi0^-8).
    Real delta_a_sq = ahat_sq - a0sq;
    Real psi_inv2 = 1.0 / (psi_known * psi_known);
    Real psi_inv8 = psi_inv2*psi_inv2*psi_inv2*psi_inv2;
    Real psi0_inv = 1.0 / psi0;
    Real x = dpsi * psi0_inv;
    Real onepx_inv = psi0 * (1.0 / psi_known);   // = 1/(1+x) = psi0/psi
    Real onepx_inv2 = onepx_inv * onepx_inv;
    Real onepx_inv8 = onepx_inv2*onepx_inv2*onepx_inv2*onepx_inv2;
    Real psi0_inv2 = psi0_inv * psi0_inv;
    Real psi0_inv8 = psi0_inv2*psi0_inv2*psi0_inv2*psi0_inv2;
    Real p8 = 8.0 + x*(28.0 + x*(56.0 + x*(70.0 + x*(56.0 + x*(28.0 +
                  x*(8.0 + x))))));
    Real reg8 = psi0_inv8 * x * p8 * onepx_inv8;
    cm_d(m, 1, mk, mj, mi) = v0 * (2.0*M_PI*u2s*psi_inv2
                                    + 0.875*delta_a_sq*psi_inv8
                                    - 0.875*a0sq*reg8);
  });
}

void MGCFCLapseDriver::RetrieveSolution(DvceArray5D<Real> &dst) {
  // dst (alpha_psi) is mesh-NGHOST-deep -- same reasoning as
  // MGCFCConformalFactorDriver::RetrieveSolution.
  mglevels_->RetrieveResult(dst, 0, pmy_pack_->pmesh->mb_indcs.ng);
  return;
}

void MGCFCLapseDriver::SeedInitialGuess(const DvceArray5D<Real> &guess, int ngh) {
  mglevels_->LoadFinestData(guess, 0, ngh);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MGCFCLapseDriver::TransferCoeffToRoot()
//! \brief see MGCFCConformalFactorDriver::TransferCoeffToRoot for the full
//! rationale -- duplicated here (not shared) since each driver owns a distinct
//! mgroot_/mglevels_ pair of a different concrete Multigrid subclass.

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
  // See MGCFCConformalFactorDriver::TransferCoeffToRoot's identical comment.
  return;
}

// Octet-scale exact one-step Gauss-Seidel (this equation is affine in u once
// psi/Ahat^2 are fixed, unlike the conformal factor's genuine Newton iteration),
// exactly the same math as MGCFCLapse::SmoothPack (per-level) above. K(x) is read
// directly from Coeff(0,...) (already fully evaluated by LoadReactionCoefficient
// at load time) -- LapseReactionCoeff itself is never called here, same as the
// per-level Pack methods.
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
        Real s = oct.Coeff(1,k,j,i);
        Real lap = OctLapseLap(oct, k, j, i);
        Real u_old = oct.U(0,k,j,i);
        Real fval = lap + dx2*(kx*u_old + s) - dx2*oct.Src(0,k,j,i);
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
        Real s = oct.Coeff(1,k,j,i);
        Real lap = OctLapseLap(oct, k, j, i);
        oct.Def(0,k,j,i) = (-(kx*oct.U(0,k,j,i) + s) + oct.Src(0,k,j,i)) - lap*idx2;
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
        Real s = oct.Coeff(1,k,j,i);
        Real lap = OctLapseLap(oct, k, j, i);
        oct.Src(0,k,j,i) += lap*idx2 + (kx*oct.U(0,k,j,i) + s);
      }
    }
  }
}
