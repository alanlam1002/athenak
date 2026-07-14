//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mg_cfc_scalar_poisson.cpp
//! \brief implementation of MGCFCScalarPoisson[Driver]

#include <cstdlib>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "multigrid/multigrid.hpp"
#include "mg_cfc_vector_poisson.hpp"  // reuses CFCVectorPoissonStencil (nvar-agnostic)
#include "mg_cfc_scalar_poisson.hpp"

//----------------------------------------------------------------------------------------
//! \fn MGCFCScalarPoisson::MGCFCScalarPoisson(...)

MGCFCScalarPoisson::MGCFCScalarPoisson(MultigridDriver *pmd, MeshBlockPack *pmbp,
                                       int nghost, bool on_host)
    : Multigrid(pmd, pmbp, nghost, on_host) {
}

MGCFCScalarPoisson::~MGCFCScalarPoisson() {
}

//----------------------------------------------------------------------------------------
// nvar_ = 1 here, so (unlike MGCFCVectorPoisson) the generic Smooth/CalculateDefect/
// CalculateFASRHS templates' hardcoded variable index 0 already addresses the only
// channel eta has -- no per-channel subview loop needed, this is a literal port of
// gravity::MGGravity's three Pack methods.

void MGCFCScalarPoisson::SmoothPack(int color) {
  int ll = nlevel_-1-current_level_;
  int is = ngh_, ie = is+(indcs_.nx1>>ll)-1;
  int js = ngh_, je = js+(indcs_.nx2>>ll)-1;
  int ks = ngh_, ke = ks+(indcs_.nx3>>ll)-1;
  CFCVectorPoissonStencil stencil{
      static_cast<MGCFCScalarPoissonDriver*>(pmy_driver_)->omega_/6.0};
  if (on_host_) {
    Smooth(u_[current_level_].h_view, src_[current_level_].h_view,
           coeff_[current_level_].h_view, matrix_[current_level_].h_view,
           stencil, -ll, is, ie, js, je, ks, ke, color, false);
  } else {
    Smooth(u_[current_level_].d_view, src_[current_level_].d_view,
           coeff_[current_level_].d_view, matrix_[current_level_].d_view,
           stencil, -ll, is, ie, js, je, ks, ke, color, false);
  }
}

void MGCFCScalarPoisson::CalculateDefectPack() {
  int ll = nlevel_-1-current_level_;
  int is = ngh_, ie = is+(indcs_.nx1>>ll)-1;
  int js = ngh_, je = js+(indcs_.nx2>>ll)-1;
  int ks = ngh_, ke = ks+(indcs_.nx3>>ll)-1;
  CFCVectorPoissonStencil stencil{0.0};
  if (on_host_) {
    CalculateDefect(def_[current_level_].h_view, u_[current_level_].h_view,
                    src_[current_level_].h_view, coeff_[current_level_].h_view,
                    matrix_[current_level_].h_view,
                    stencil, -ll, is, ie, js, je, ks, ke, false);
  } else {
    CalculateDefect(def_[current_level_].d_view, u_[current_level_].d_view,
                    src_[current_level_].d_view, coeff_[current_level_].d_view,
                    matrix_[current_level_].d_view,
                    stencil, -ll, is, ie, js, je, ks, ke, false);
  }
}

void MGCFCScalarPoisson::CalculateFASRHSPack() {
  int ll = nlevel_-1-current_level_;
  int is = ngh_, ie = is+(indcs_.nx1>>ll)-1;
  int js = ngh_, je = js+(indcs_.nx2>>ll)-1;
  int ks = ngh_, ke = ks+(indcs_.nx3>>ll)-1;
  CFCVectorPoissonStencil stencil{0.0};
  if (on_host_) {
    CalculateFASRHS(src_[current_level_].h_view, u_[current_level_].h_view,
                    coeff_[current_level_].h_view, matrix_[current_level_].h_view,
                    stencil, -ll, is, ie, js, je, ks, ke, false);
  } else {
    CalculateFASRHS(src_[current_level_].d_view, u_[current_level_].d_view,
                    coeff_[current_level_].d_view, matrix_[current_level_].d_view,
                    stencil, -ll, is, ie, js, je, ks, ke, false);
  }
}


//----------------------------------------------------------------------------------------
//! \fn MGCFCScalarPoissonDriver::MGCFCScalarPoissonDriver(...)
//! \brief constructs the root + meshblock-level Multigrid hierarchies with nvar_ = 1.
//! Defaults to BoundaryFlag::mg_multipole (via <cfc> mg_poisson_outer_bc) -- see
//! mg_cfc_vector_poisson.cpp's constructor comment for the full rationale (eta's
//! true outer boundary is an asymptotic ~1/r falloff, not exact zero at any finite
//! radius; multipole is safe here since eta's source is a real, unmodified src_,
//! unlike psi/alpha_psi's coeff_-based nonlinear equations). autompo_ is left false
//! (fixed origin (0,0,0)) for the same reason as the vector driver. Restores
//! BoundaryFlag::mg_zerograd (NOT plain BoundaryFlag::reflect -- see
//! mg_cfc_vector_poisson.cpp's constructor comment for why) on any reflecting mesh
//! face.

MGCFCScalarPoissonDriver::MGCFCScalarPoissonDriver(MeshBlockPack *pmbp,
                                                   ParameterInput *pin)
    : MultigridDriver(pmbp, 1) {
  autompo_ = false;
  std::string outer_bc_str = pin->GetOrAddString("cfc", "mg_poisson_outer_bc",
                                                 "multipole");
  BoundaryFlag outer_bc;
  if (outer_bc_str == "multipole") {
    outer_bc = BoundaryFlag::mg_multipole;
    mporder_ = pin->GetOrAddInteger("cfc", "mg_poisson_mporder", 4);
    if (mporder_ != 2 && mporder_ != 4) {
      std::cout << "### FATAL ERROR in MGCFCScalarPoissonDriver" << std::endl
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
    std::cout << "### FATAL ERROR in MGCFCScalarPoissonDriver" << std::endl
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
  mgroot_ = new MGCFCScalarPoisson(this, nullptr, nghost, root_on_host);
  mglevels_ = new MGCFCScalarPoisson(this, pmbp, nghost);
  mglevels_->pbval = new MultigridBoundaryValues(pmbp, pin, false, mglevels_);
  mglevels_->pbval->InitializeBuffers(nvar_);
  mglevels_->pbval->RemapIndicesForMG();
  mglevels_->pbval->ComputePerLevelIndices();
}

MGCFCScalarPoissonDriver::~MGCFCScalarPoissonDriver() {
  delete mgroot_;
  delete mglevels_;
}

//----------------------------------------------------------------------------------------
//! \fn void MGCFCScalarPoissonDriver::Solve(Driver *pdriver, int stage, Real dt)
//! \brief run the V-cycle solve on eta's right-hand side. Assumes LoadPoissonSource()
//! was already called for this cycle -- see MGCFCVectorPoissonDriver::Solve()'s
//! comment for why this driver doesn't load its own source or retrieve its own result.

void MGCFCScalarPoissonDriver::Solve(Driver *pdriver, int stage, Real dt) {
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

void MGCFCScalarPoissonDriver::LoadPoissonSource(const DvceArray5D<Real> &eta_src) {
  mglevels_->LoadSource(eta_src, 0, mglevels_->GetGhostCells(), 1.0);
  return;
}

void MGCFCScalarPoissonDriver::RetrieveSolution(DvceArray5D<Real> &eta_dst) {
  // eta_dst (cfc::CFC::eta_x/eta_beta) is sized at mesh-NGHOST depth -- see
  // MGCFCVectorPoissonDriver::RetrieveSolution's identical comment (plan addendum
  // #4, Finding F) for why this must be the mesh's ng, not this solver's own ngh_.
  mglevels_->RetrieveResult(eta_dst, 0, pmy_pack_->pmesh->mb_indcs.ng);
  return;
}

//----------------------------------------------------------------------------------------
// Host-side octet physics for MGCFCScalarPoissonDriver. nvar_ = 1, so this is a literal
// port of gravity::MGGravityDriver's octet functions (no per-channel loop needed).

namespace {
inline Real OctLaplacian(const MGOctet &o, int v, int k, int j, int i) {
  return (6.0*o.U(v,k,j,i) - o.U(v,k+1,j,i) - o.U(v,k,j+1,i)
          - o.U(v,k,j,i+1) - o.U(v,k-1,j,i) - o.U(v,k,j-1,i)
          - o.U(v,k,j,i-1));
}
}  // namespace

void MGCFCScalarPoissonDriver::SmoothOctet(MGOctet &oct, int rlev, int color) {
  int ngh = mgroot_->GetGhostCells();
  Real root_dx = mgroot_->GetRootDx();
  Real dx = root_dx / static_cast<Real>(1 << rlev);
  Real dx2 = dx * dx;
  Real isix = omega_ / 6.0;
  int c = color ^ coffset_;
  for (int k = ngh; k <= ngh+1; ++k) {
    for (int j = ngh; j <= ngh+1; ++j) {
      for (int i = ngh + ((c^k^j)&1); i <= ngh+1; i += 2) {
        Real lap = OctLaplacian(oct, 0, k, j, i);
        oct.U(0,k,j,i) -= (lap - oct.Src(0,k,j,i)*dx2)*isix;
      }
    }
  }
}

void MGCFCScalarPoissonDriver::CalculateDefectOctet(MGOctet &oct, int rlev) {
  int ngh = mgroot_->GetGhostCells();
  Real root_dx = mgroot_->GetRootDx();
  Real dx = root_dx / static_cast<Real>(1 << rlev);
  Real idx2 = 1.0 / (dx * dx);
  for (int k = ngh; k <= ngh+1; ++k) {
    for (int j = ngh; j <= ngh+1; ++j) {
      for (int i = ngh; i <= ngh+1; ++i) {
        oct.Def(0,k,j,i) = oct.Src(0,k,j,i) - OctLaplacian(oct, 0, k, j, i) * idx2;
      }
    }
  }
}

void MGCFCScalarPoissonDriver::CalculateFASRHSOctet(MGOctet &oct, int rlev) {
  int ngh = mgroot_->GetGhostCells();
  Real root_dx = mgroot_->GetRootDx();
  Real dx = root_dx / static_cast<Real>(1 << rlev);
  Real idx2 = 1.0 / (dx * dx);
  for (int k = ngh; k <= ngh+1; ++k) {
    for (int j = ngh; j <= ngh+1; ++j) {
      for (int i = ngh; i <= ngh+1; ++i) {
        oct.Src(0,k,j,i) += OctLaplacian(oct, 0, k, j, i) * idx2;
      }
    }
  }
}
