//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mg_cfc_vector_poisson.cpp
//! \brief implementation of MGCFCVectorPoisson[Driver]

#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "multigrid/multigrid.hpp"
#include "mg_cfc_vector_poisson.hpp"

//----------------------------------------------------------------------------------------
//! \fn MGCFCVectorPoisson::MGCFCVectorPoisson(...)

MGCFCVectorPoisson::MGCFCVectorPoisson(MultigridDriver *pmd, MeshBlockPack *pmbp,
                                       int nghost, bool on_host)
    : Multigrid(pmd, pmbp, nghost, on_host) {
}

MGCFCVectorPoisson::~MGCFCVectorPoisson() {
}

namespace {

//----------------------------------------------------------------------------------------
// Multigrid::Smooth/CalculateDefect/CalculateFASRHS (multigrid.hpp) hardcode variable
// index 0, so a nvar_=3 solver like MGCFCVectorPoisson must call them once per channel,
// each time on a rank-preserving Kokkos::subview restricted to that one channel -- the
// same idiom AthenaTensor<...,1>::InitWithShallowSlice (athena_tensor.hpp) uses to
// slice N contiguous variable channels while keeping the view 5D.
//
// coeff/matrix are NOT threaded through here: CFCVectorPoissonStencil::Apply() never
// reads either (flat decoupled Laplacian, no coefficient term), and neither
// Multigrid::coeff_ (ncoeff_ stays 0 -- this solver never sets it, unlike
// MGCFCConformalFactor/MGCFCLapse's genuinely-coupled equations) nor Multigrid::
// matrix_ (nmatrix_/matrix_ are never allocated by *any* solver in this codebase,
// gravity included) actually hold usable per-channel data here. Smooth/
// CalculateDefect/CalculateFASRHS still require a same-typed 4th/5th ViewType
// argument, so u/src's own already-valid per-channel subview is passed through in
// that slot instead of constructing a subview of an unallocated array (which is
// what crashed under Kokkos bounds checking before this comment was written).

template <typename StencilOp>
void SmoothChannels(Multigrid *mg, DualArray5D<Real> &u_lv, DualArray5D<Real> &src_lv,
                    const StencilOp &stencil, int rlev,
                    int il, int iu, int jl, int ju, int kl, int ku,
                    int color, bool on_host) {
  for (int v = 0; v < 3; ++v) {
    auto vr = std::make_pair(v, v + 1);
    if (on_host) {
      auto u = Kokkos::subview(u_lv.h_view, Kokkos::ALL, vr,
                                Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      auto src = Kokkos::subview(src_lv.h_view, Kokkos::ALL, vr,
                                  Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      mg->Smooth(u, src, u, src, stencil, rlev, il, iu, jl, ju, kl, ku,
                 color, false);
    } else {
      auto u = Kokkos::subview(u_lv.d_view, Kokkos::ALL, vr,
                                Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      auto src = Kokkos::subview(src_lv.d_view, Kokkos::ALL, vr,
                                  Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      mg->Smooth(u, src, u, src, stencil, rlev, il, iu, jl, ju, kl, ku,
                 color, false);
    }
  }
}

template <typename StencilOp>
void CalculateDefectChannels(Multigrid *mg, DualArray5D<Real> &def_lv,
                             DualArray5D<Real> &u_lv, DualArray5D<Real> &src_lv,
                             const StencilOp &stencil, int rlev,
                             int il, int iu, int jl, int ju, int kl, int ku,
                             bool on_host) {
  for (int v = 0; v < 3; ++v) {
    auto vr = std::make_pair(v, v + 1);
    if (on_host) {
      auto def = Kokkos::subview(def_lv.h_view, Kokkos::ALL, vr,
                                  Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      auto u = Kokkos::subview(u_lv.h_view, Kokkos::ALL, vr,
                                Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      auto src = Kokkos::subview(src_lv.h_view, Kokkos::ALL, vr,
                                  Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      mg->CalculateDefect(def, u, src, u, src, stencil, rlev,
                          il, iu, jl, ju, kl, ku, false);
    } else {
      auto def = Kokkos::subview(def_lv.d_view, Kokkos::ALL, vr,
                                  Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      auto u = Kokkos::subview(u_lv.d_view, Kokkos::ALL, vr,
                                Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      auto src = Kokkos::subview(src_lv.d_view, Kokkos::ALL, vr,
                                  Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      mg->CalculateDefect(def, u, src, u, src, stencil, rlev,
                          il, iu, jl, ju, kl, ku, false);
    }
  }
}

template <typename StencilOp>
void CalculateFASRHSChannels(Multigrid *mg, DualArray5D<Real> &src_lv,
                             DualArray5D<Real> &u_lv, const StencilOp &stencil,
                             int rlev, int il, int iu, int jl, int ju, int kl, int ku,
                             bool on_host) {
  for (int v = 0; v < 3; ++v) {
    auto vr = std::make_pair(v, v + 1);
    if (on_host) {
      auto src = Kokkos::subview(src_lv.h_view, Kokkos::ALL, vr,
                                  Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      auto u = Kokkos::subview(u_lv.h_view, Kokkos::ALL, vr,
                                Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      mg->CalculateFASRHS(src, u, u, src, stencil, rlev,
                          il, iu, jl, ju, kl, ku, false);
    } else {
      auto src = Kokkos::subview(src_lv.d_view, Kokkos::ALL, vr,
                                  Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      auto u = Kokkos::subview(u_lv.d_view, Kokkos::ALL, vr,
                                Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      mg->CalculateFASRHS(src, u, u, src, stencil, rlev,
                          il, iu, jl, ju, kl, ku, false);
    }
  }
}

}  // namespace

void MGCFCVectorPoisson::SmoothPack(int color) {
  int ll = nlevel_-1-current_level_;
  int is = ngh_, ie = is+(indcs_.nx1>>ll)-1;
  int js = ngh_, je = js+(indcs_.nx2>>ll)-1;
  int ks = ngh_, ke = ks+(indcs_.nx3>>ll)-1;
  CFCVectorPoissonStencil stencil{
      static_cast<MGCFCVectorPoissonDriver*>(pmy_driver_)->omega_/6.0};
  SmoothChannels(this, u_[current_level_], src_[current_level_],
                stencil, -ll, is, ie, js, je, ks, ke, color, on_host_);
}

void MGCFCVectorPoisson::CalculateDefectPack() {
  int ll = nlevel_-1-current_level_;
  int is = ngh_, ie = is+(indcs_.nx1>>ll)-1;
  int js = ngh_, je = js+(indcs_.nx2>>ll)-1;
  int ks = ngh_, ke = ks+(indcs_.nx3>>ll)-1;
  CFCVectorPoissonStencil stencil{0.0};
  CalculateDefectChannels(this, def_[current_level_], u_[current_level_],
                          src_[current_level_], stencil, -ll,
                          is, ie, js, je, ks, ke, on_host_);
}

void MGCFCVectorPoisson::CalculateFASRHSPack() {
  int ll = nlevel_-1-current_level_;
  int is = ngh_, ie = is+(indcs_.nx1>>ll)-1;
  int js = ngh_, je = js+(indcs_.nx2>>ll)-1;
  int ks = ngh_, ke = ks+(indcs_.nx3>>ll)-1;
  CFCVectorPoissonStencil stencil{0.0};
  CalculateFASRHSChannels(this, src_[current_level_], u_[current_level_],
                          stencil, -ll, is, ie, js, je, ks, ke, on_host_);
}


//----------------------------------------------------------------------------------------
//! \fn MGCFCVectorPoissonDriver::MGCFCVectorPoissonDriver(...)
//! \brief constructs the root + meshblock-level Multigrid hierarchies with nvar_ = 3.
//! P_i's true outer boundary (X^i -> 0 / beta^i -> 0 as r -> infinity, Gmunu eq.
//! 79/80) is an *asymptotic* falloff (~1/r, like a Newtonian vector potential
//! sourced by a compact S_i-tilde distribution), not an exact zero at any finite
//! radius -- BoundaryFlag::mg_multipole (default here, via <cfc> mg_poisson_outer_bc)
//! captures that falloff (with real angular structure up to mg_poisson_mporder);
//! the previous default, mg_zerofixed (still available via mg_poisson_outer_bc=
//! zerofixed), was only ever the leading-order (monopole-zero) truncation of it --
//! see mg_cfc_conformal_factor.cpp's constructor comment and DEVELOPMENT.md item 9
//! for why this same tradeoff mattered enough to fix (via Robin, there) for psi/
//! alpha_psi. Multipole is safe here in a way it never was for psi/alpha_psi:
//! P_i's source is a real, unmodified src_ (LoadPoissonSource's fac=1.0, loaded via
//! the generic Multigrid::LoadSource -- no coeff_ involved, no custom Newton
//! relaxation), so ScaleMultipoleCoefficients()'s normalization (the generic
//! Green's-function constant for any -Delta u = src equation) applies with zero
//! re-derivation. autompo_ is deliberately left false (fixed origin (0,0,0), the
//! base constructor's own default) rather than exposed as an input: every current
//! CFC test problem's star sits at the coordinate origin, and skipping
//! CalculateCenterOfMass() avoids a second channel-0-only function this pass would
//! otherwise also need to generalize (see multigrid_driver.cpp's per-channel
//! CalculateMultipoleCoefficients()/ScaleMultipoleCoefficients()/SyncMultipoleToDevice()
//! and both ghost-fill sites -- gravity's own nvar_=1 usage is unaffected, confirmed
//! by construction: every new per-channel loop reduces to its original single
//! iteration there).
//!
//! Faces where the *mesh* itself is reflecting (e.g. an octant-reduced domain like a
//! single-star test) still need BoundaryFlag::mg_zerograd, not plain
//! BoundaryFlag::reflect: mg_mesh_bcs_ is multigrid-internal state read only by
//! MultigridDriver's own boundary code (PhysicalBoundary, MGRootBoundary), and
//! MGRootBoundary's device path (multigrid_driver.cpp) has no case at all for plain
//! BoundaryFlag::reflect -- passing it falls through every branch there silently,
//! leaving that ghost cell untouched (confirmed: this was tried first and left the
//! root grid's ghost cells at stale zero, seeding a NaN cascade through the Newton
//! solve within the first V-cycle). mg_zerograd's ghost = interior formula is
//! exactly the even mirror a symmetry plane needs.

MGCFCVectorPoissonDriver::MGCFCVectorPoissonDriver(MeshBlockPack *pmbp,
                                                   ParameterInput *pin)
    : MultigridDriver(pmbp, 3) {
  autompo_ = false;
  std::string outer_bc_str = pin->GetOrAddString("cfc", "mg_poisson_outer_bc",
                                                 "multipole");
  BoundaryFlag outer_bc;
  if (outer_bc_str == "multipole") {
    outer_bc = BoundaryFlag::mg_multipole;
    mporder_ = pin->GetOrAddInteger("cfc", "mg_poisson_mporder", 4);
    if (mporder_ != 2 && mporder_ != 4) {
      std::cout << "### FATAL ERROR in MGCFCVectorPoissonDriver" << std::endl
                << "mg_poisson_mporder must be 2 (quadrupole) or 4 (hexadecapole)."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    AllocateMultipoleCoefficients();
  } else if (outer_bc_str == "zerofixed") {
    outer_bc = BoundaryFlag::mg_zerofixed;
  } else if (outer_bc_str == "robin") {
    outer_bc = BoundaryFlag::mg_robin;
    robin_order_ = pin->GetOrAddInteger("cfc", "mg_robin_order", 1);
  } else {
    std::cout << "### FATAL ERROR in MGCFCVectorPoissonDriver" << std::endl
              << "cfc/mg_poisson_outer_bc must be 'multipole', 'zerofixed', "
              << "or 'robin'." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  for (int f = 0; f < 6; ++f) {
    if (pmbp->pmesh->mesh_bcs[f] == BoundaryFlag::reflect) {
      mg_mesh_bcs_[f] = BoundaryFlag::mg_zerograd;
    } else if (pmbp->pmesh->mesh_bcs[f] != BoundaryFlag::periodic) {
      mg_mesh_bcs_[f] = outer_bc;
    }
  }
  omega_ = pin->GetOrAddReal("cfc", "mg_omega", 1.15);
  eps_ = pin->GetOrAddReal("cfc", "mg_threshold", 1.0e-10);
  fshowdef_ = pin->GetOrAddInteger("cfc", "mg_verbose", 0);
  mg_verbose_ = fshowdef_;
  full_multigrid_ = false;

  int nghost = pin->GetOrAddInteger("cfc", "mg_nghost", 1);
  bool root_on_host = pin->GetOrAddBoolean("cfc", "root_on_host", false);
  mgroot_ = new MGCFCVectorPoisson(this, nullptr, nghost, root_on_host);
  mglevels_ = new MGCFCVectorPoisson(this, pmbp, nghost);
  mglevels_->pbval = new MultigridBoundaryValues(pmbp, pin, false, mglevels_);
  mglevels_->pbval->InitializeBuffers(nvar_);
  mglevels_->pbval->RemapIndicesForMG();
  mglevels_->pbval->ComputePerLevelIndices();
}

MGCFCVectorPoissonDriver::~MGCFCVectorPoissonDriver() {
  delete mgroot_;
  delete mglevels_;
}

//----------------------------------------------------------------------------------------
//! \fn void MGCFCVectorPoissonDriver::Solve(Driver *pdriver, int stage, Real dt)
//! \brief run the V-cycle solve on P_i's right-hand side. Assumes LoadPoissonSource()
//! was already called for this cycle. Unlike gravity::MGGravityDriver::Solve(), this
//! does not load its own source/initial-guess or retrieve its own result -- cfc::CFC
//! drives those explicitly via LoadPoissonSource()/RetrieveSolution() since this one
//! driver is reused for two physically distinct equations (X^i's and beta^i's P_i).

void MGCFCVectorPoissonDriver::Solve(Driver *pdriver, int stage, Real dt) {
  PrepareForAMR();
  SetupMultigrid(dt, false);
  // autompo_ is always false here (see constructor comment), so no
  // CalculateCenterOfMass() call -- mirrors gravity's own if(autompo_) gate.
  if (mporder_ > 0) {
    CalculateMultipoleCoefficients();
    SyncMultipoleToDevice();
  }
  SolveMG(pdriver);
  return;
}

void MGCFCVectorPoissonDriver::LoadPoissonSource(const DvceArray5D<Real> &p_src) {
  mglevels_->LoadSource(p_src, 0, mglevels_->GetGhostCells(), 1.0);
  return;
}

void MGCFCVectorPoissonDriver::RetrieveSolution(DvceArray5D<Real> &p_dst) {
  // p_dst (cfc::CFC::u_p_x/u_p_beta) is sized at mesh-NGHOST depth, not this solver's
  // own (generally shallower) ngh_ -- Multigrid::RetrieveResult's ngh argument is the
  // depth p_dst itself is padded to, not the multigrid's, so it must be the mesh's
  // ng here (see plan addendum #4, Finding F). RetrieveResult only ever fills the
  // inner min(ngh_, ng) ring regardless; the outer ring is left for cfc::CFC's own
  // post-retrieve MeshBoundaryValuesCC exchange to fill from neighboring blocks.
  mglevels_->RetrieveResult(p_dst, 0, pmy_pack_->pmesh->mb_indcs.ng);
  return;
}

//----------------------------------------------------------------------------------------
// Host-side octet physics for MGCFCVectorPoissonDriver. Same 7-point Laplacian as
// gravity::MGGravityDriver's octet functions, generalized to loop over the 3
// independent P_i channels instead of gravity's hardcoded single channel (v=0).

namespace {
inline Real OctLaplacian(const MGOctet &o, int v, int k, int j, int i) {
  return (6.0*o.U(v,k,j,i) - o.U(v,k+1,j,i) - o.U(v,k,j+1,i)
          - o.U(v,k,j,i+1) - o.U(v,k-1,j,i) - o.U(v,k,j-1,i)
          - o.U(v,k,j,i-1));
}
}  // namespace

void MGCFCVectorPoissonDriver::SmoothOctet(MGOctet &oct, int rlev, int color) {
  int ngh = mgroot_->GetGhostCells();
  Real root_dx = mgroot_->GetRootDx();
  Real dx = root_dx / static_cast<Real>(1 << rlev);
  Real dx2 = dx * dx;
  Real isix = omega_ / 6.0;
  int c = color ^ coffset_;
  for (int v = 0; v < 3; ++v) {
    for (int k = ngh; k <= ngh+1; ++k) {
      for (int j = ngh; j <= ngh+1; ++j) {
        for (int i = ngh + ((c^k^j)&1); i <= ngh+1; i += 2) {
          Real lap = OctLaplacian(oct, v, k, j, i);
          oct.U(v,k,j,i) -= (lap - oct.Src(v,k,j,i)*dx2)*isix;
        }
      }
    }
  }
}

void MGCFCVectorPoissonDriver::CalculateDefectOctet(MGOctet &oct, int rlev) {
  int ngh = mgroot_->GetGhostCells();
  Real root_dx = mgroot_->GetRootDx();
  Real dx = root_dx / static_cast<Real>(1 << rlev);
  Real idx2 = 1.0 / (dx * dx);
  for (int v = 0; v < 3; ++v) {
    for (int k = ngh; k <= ngh+1; ++k) {
      for (int j = ngh; j <= ngh+1; ++j) {
        for (int i = ngh; i <= ngh+1; ++i) {
          oct.Def(v,k,j,i) = oct.Src(v,k,j,i) - OctLaplacian(oct, v, k, j, i) * idx2;
        }
      }
    }
  }
}

void MGCFCVectorPoissonDriver::CalculateFASRHSOctet(MGOctet &oct, int rlev) {
  int ngh = mgroot_->GetGhostCells();
  Real root_dx = mgroot_->GetRootDx();
  Real dx = root_dx / static_cast<Real>(1 << rlev);
  Real idx2 = 1.0 / (dx * dx);
  for (int v = 0; v < 3; ++v) {
    for (int k = ngh; k <= ngh+1; ++k) {
      for (int j = ngh; j <= ngh+1; ++j) {
        for (int i = ngh; i <= ngh+1; ++i) {
          oct.Src(v,k,j,i) += OctLaplacian(oct, v, k, j, i) * idx2;
        }
      }
    }
  }
}
