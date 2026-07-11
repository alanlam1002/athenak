//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mg_cfc_vector_poisson.cpp
//! \brief implementation of MGCFCVectorPoisson[Driver]

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

template <typename StencilOp>
void SmoothChannels(Multigrid *mg, DualArray5D<Real> &u_lv, DualArray5D<Real> &src_lv,
                    DualArray5D<Real> &coeff_lv, DualArray5D<Real> &matrix_lv,
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
      auto coeff = Kokkos::subview(coeff_lv.h_view, Kokkos::ALL, vr,
                                    Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      auto matrix = Kokkos::subview(matrix_lv.h_view, Kokkos::ALL, vr,
                                     Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      mg->Smooth(u, src, coeff, matrix, stencil, rlev, il, iu, jl, ju, kl, ku,
                 color, false);
    } else {
      auto u = Kokkos::subview(u_lv.d_view, Kokkos::ALL, vr,
                                Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      auto src = Kokkos::subview(src_lv.d_view, Kokkos::ALL, vr,
                                  Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      auto coeff = Kokkos::subview(coeff_lv.d_view, Kokkos::ALL, vr,
                                    Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      auto matrix = Kokkos::subview(matrix_lv.d_view, Kokkos::ALL, vr,
                                     Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      mg->Smooth(u, src, coeff, matrix, stencil, rlev, il, iu, jl, ju, kl, ku,
                 color, false);
    }
  }
}

template <typename StencilOp>
void CalculateDefectChannels(Multigrid *mg, DualArray5D<Real> &def_lv,
                             DualArray5D<Real> &u_lv, DualArray5D<Real> &src_lv,
                             DualArray5D<Real> &coeff_lv, DualArray5D<Real> &matrix_lv,
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
      auto coeff = Kokkos::subview(coeff_lv.h_view, Kokkos::ALL, vr,
                                    Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      auto matrix = Kokkos::subview(matrix_lv.h_view, Kokkos::ALL, vr,
                                     Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      mg->CalculateDefect(def, u, src, coeff, matrix, stencil, rlev,
                          il, iu, jl, ju, kl, ku, false);
    } else {
      auto def = Kokkos::subview(def_lv.d_view, Kokkos::ALL, vr,
                                  Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      auto u = Kokkos::subview(u_lv.d_view, Kokkos::ALL, vr,
                                Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      auto src = Kokkos::subview(src_lv.d_view, Kokkos::ALL, vr,
                                  Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      auto coeff = Kokkos::subview(coeff_lv.d_view, Kokkos::ALL, vr,
                                    Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      auto matrix = Kokkos::subview(matrix_lv.d_view, Kokkos::ALL, vr,
                                     Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      mg->CalculateDefect(def, u, src, coeff, matrix, stencil, rlev,
                          il, iu, jl, ju, kl, ku, false);
    }
  }
}

template <typename StencilOp>
void CalculateFASRHSChannels(Multigrid *mg, DualArray5D<Real> &src_lv,
                             DualArray5D<Real> &u_lv, DualArray5D<Real> &coeff_lv,
                             DualArray5D<Real> &matrix_lv, const StencilOp &stencil,
                             int rlev, int il, int iu, int jl, int ju, int kl, int ku,
                             bool on_host) {
  for (int v = 0; v < 3; ++v) {
    auto vr = std::make_pair(v, v + 1);
    if (on_host) {
      auto src = Kokkos::subview(src_lv.h_view, Kokkos::ALL, vr,
                                  Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      auto u = Kokkos::subview(u_lv.h_view, Kokkos::ALL, vr,
                                Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      auto coeff = Kokkos::subview(coeff_lv.h_view, Kokkos::ALL, vr,
                                    Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      auto matrix = Kokkos::subview(matrix_lv.h_view, Kokkos::ALL, vr,
                                     Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      mg->CalculateFASRHS(src, u, coeff, matrix, stencil, rlev,
                          il, iu, jl, ju, kl, ku, false);
    } else {
      auto src = Kokkos::subview(src_lv.d_view, Kokkos::ALL, vr,
                                  Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      auto u = Kokkos::subview(u_lv.d_view, Kokkos::ALL, vr,
                                Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      auto coeff = Kokkos::subview(coeff_lv.d_view, Kokkos::ALL, vr,
                                    Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      auto matrix = Kokkos::subview(matrix_lv.d_view, Kokkos::ALL, vr,
                                     Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      mg->CalculateFASRHS(src, u, coeff, matrix, stencil, rlev,
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
                coeff_[current_level_], matrix_[current_level_],
                stencil, -ll, is, ie, js, je, ks, ke, color, on_host_);
}

void MGCFCVectorPoisson::CalculateDefectPack() {
  int ll = nlevel_-1-current_level_;
  int is = ngh_, ie = is+(indcs_.nx1>>ll)-1;
  int js = ngh_, je = js+(indcs_.nx2>>ll)-1;
  int ks = ngh_, ke = ks+(indcs_.nx3>>ll)-1;
  CFCVectorPoissonStencil stencil{0.0};
  CalculateDefectChannels(this, def_[current_level_], u_[current_level_],
                          src_[current_level_], coeff_[current_level_],
                          matrix_[current_level_], stencil, -ll,
                          is, ie, js, je, ks, ke, on_host_);
}

void MGCFCVectorPoisson::CalculateFASRHSPack() {
  int ll = nlevel_-1-current_level_;
  int is = ngh_, ie = is+(indcs_.nx1>>ll)-1;
  int js = ngh_, je = js+(indcs_.nx2>>ll)-1;
  int ks = ngh_, ke = ks+(indcs_.nx3>>ll)-1;
  CFCVectorPoissonStencil stencil{0.0};
  CalculateFASRHSChannels(this, src_[current_level_], u_[current_level_],
                          coeff_[current_level_], matrix_[current_level_],
                          stencil, -ll, is, ie, js, je, ks, ke, on_host_);
}


//----------------------------------------------------------------------------------------
//! \fn MGCFCVectorPoissonDriver::MGCFCVectorPoissonDriver(...)
//! \brief constructs the root + meshblock-level Multigrid hierarchies with nvar_ = 3.
//! Boundary conditions are NOT set explicitly here: the MultigridDriver base
//! constructor already defaults every non-periodic mg_mesh_bcs_[f] to
//! BoundaryFlag::mg_zerofixed (multigrid_driver.cpp), which is exactly what P_i needs
//! (X^i|_rmax = 0, Gmunu eq. 80 / beta^i|_rmax = 0, eq. 79) -- unlike gravity, CFC has
//! no configurable mg_bc input option, so there is nothing to override.

MGCFCVectorPoissonDriver::MGCFCVectorPoissonDriver(MeshBlockPack *pmbp,
                                                   ParameterInput *pin)
    : MultigridDriver(pmbp, 3) {
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
  SolveMG(pdriver);
  return;
}

void MGCFCVectorPoissonDriver::LoadPoissonSource(const DvceArray5D<Real> &p_src) {
  mglevels_->LoadSource(p_src, 0, mglevels_->GetGhostCells(), 1.0);
  return;
}

void MGCFCVectorPoissonDriver::RetrieveSolution(DvceArray5D<Real> &p_dst) {
  mglevels_->RetrieveResult(p_dst, 0, mglevels_->GetGhostCells());
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
