//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file bhstar_bondi.cpp
//! \brief bhstar Stage 0 (numerics-005): marginally bound radial free fall onto a
//! Schwarzschild black hole in Cartesian Kerr-Schild coordinates (a = 0), with radiation.
//!
//! Initial state:  u_t = -1, u^r = -sqrt(2M/r), rho = Mdot/(4 pi r^2 sqrt(2M/r)),
//!                 T = t_res everywhere (p = rho k T/(mu m_u)), B = 0, radiation in LTE
//!                 and isotropic in the fluid frame at T.
//! Boundaries:     user BC on all six faces. Fluid: the analytic state in the ghost zones.
//!                 Radiation: per angle, copy i0 from the last active cell if the ray
//!                 leaves the domain through that face, zero it if it enters (vacuum).
//! History:        <problem>/user_hist = true writes the energy budget
//!                 (needs <radiation>/energy_budget = true, see radiation_budget.hpp).
//! Inputs:         <problem>/mdot_cgs (g/s, default 1.50e25), <problem>/t_res (K,
//!                 default 7000). Requires <units> (bhmass_msun, density_cgs, mu).
//!
//! Units with G = M = c = 1: length GM/c^2, time GM/c^3, density <units>/density_cgs,
//! so Mdot_code = Mdot_cgs / (mass_cgs/time_cgs) and T_code = p/rho = T_K/temperature_cgs.

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "outputs/outputs.hpp"
#include "radiation/radiation.hpp"
#include "radiation/radiation_budget.hpp"
#include "srcterms/srcterms.hpp"
#include "units/units.hpp"
#include "pgen/pgen.hpp"

namespace {

struct bhstar_bondi_pgen {
  int setup;        // 0: free fall (Stage 0); 1: uniform medium (unit tests U1, U2)
  Real rho0, pgas0, vx0;   // uniform medium: density, pressure, 3-velocity along x
  Real blob_e, blob_r;     // uniform medium: initial isotropic radiation blob (U2)
  Real rho_coef;    // Mdot/(4 pi) in code units
  Real p_over_rho;  // T_res in code units (p/rho)
  Real arad;        // radiation constant in code units (0 if no radiation)
  Real gm1;         // gamma - 1
  bool flat;
  Real spin;
};

bhstar_bondi_pgen bb;

//----------------------------------------------------------------------------------------
//! \fn FreeFallState
//! \brief Primitive state of marginally bound free fall at Cartesian KS point (x,y,z).
//! Returns rho, p and AthenaK's primitive velocity uu^i = u^i - (g^{0i}/g^{00}) u^0
//! (the spatial four-velocity seen by the normal observer).
//! With x = sqrt(2/r): u^r = -x, and in ingoing KS (a = 0)
//!   u^t = u^t_Schw + (2/r)/(1-2/r) u^r = (1 - x^3)/(1 - x^2) = (1 + x + x^2)/(1 + x),
//! which is regular at the horizon. In Cartesian KS with a = 0, u^i = u^r x^i/r.

KOKKOS_INLINE_FUNCTION
void FreeFallState(Real x1, Real x2, Real x3, const bhstar_bondi_pgen &p,
                   Real &rho, Real &pgas, Real &uu1, Real &uu2, Real &uu3) {
  if (p.setup == 1) {  // uniform medium moving along x (flat space tests)
    rho = p.rho0; pgas = p.pgas0;
    uu1 = p.vx0/sqrt(1.0 - SQR(p.vx0)); uu2 = 0.0; uu3 = 0.0;
    return;
  }
  Real r = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  r = fmax(r, 0.5);  // inside the excision radius; keeps the state finite at r -> 0
  Real xs = sqrt(2.0/r);
  rho = p.rho_coef/(SQR(r)*xs);
  pgas = rho*p.p_over_rho;
  Real ut = (1.0 + xs + SQR(xs))/(1.0 + xs);
  Real ur = -xs;
  Real u1 = ur*x1/r, u2 = ur*x2/r, u3 = ur*x3/r;
  Real glower[4][4], gupper[4][4];
  ComputeMetricAndInverse(x1, x2, x3, p.flat, p.spin, glower, gupper);
  uu1 = u1 - gupper[0][1]/gupper[0][0]*ut;
  uu2 = u2 - gupper[0][2]/gupper[0][0]*ut;
  uu3 = u3 - gupper[0][3]/gupper[0][0]*ut;
}

} // namespace

void BhstarBondiBC(Mesh *pm);
void BhstarBudgetHistory(HistoryData *pdata, Mesh *pm);
Real RadMomentumX(MeshBlockPack *pmbp);

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::BhstarBondi()

void ProblemGenerator::BhstarBondi(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!pmbp->pcoord->is_general_relativistic || pmbp->pdyngr != nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "bhstar_bondi needs GR on a fixed metric (<coord> general_rel = true)"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->pcoord->coord_data.bh_spin != 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "bhstar_bondi is implemented for a = 0 only" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::string setup = pin->GetOrAddString("problem", "setup", "free_fall");
  bb.setup = (setup.compare("uniform") == 0) ? 1 : 0;
  if (bb.setup == 0 && pmbp->punit == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "bhstar_bondi needs a <units> block" << std::endl;
    exit(EXIT_FAILURE);
  }

  // enroll BCs and history
  user_bcs_func = BhstarBondiBC;
  if (user_hist) user_hist_func = BhstarBudgetHistory;

  // problem parameters (also needed by the BCs after a restart)
  Real mdot_cgs = 0.0, t_res = 0.0;
  if (bb.setup == 0) {
    mdot_cgs = pin->GetOrAddReal("problem", "mdot_cgs", 1.50e25);
    t_res = pin->GetOrAddReal("problem", "t_res", 7000.0);
    bb.rho_coef = mdot_cgs/(pmbp->punit->mass_cgs()/pmbp->punit->time_cgs())/(4.0*M_PI);
    bb.p_over_rho = t_res/pmbp->punit->temperature_cgs();
  } else {
    // test setup (code units): uniform rho0, pgas0, 3-velocity vx0 along x; optional
    // isotropic radiation blob of energy density blob_e*exp(-r^2/(2 blob_r^2))
    bb.rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
    bb.pgas0 = pin->GetOrAddReal("problem", "pgas0", 1.0e-3);
    bb.vx0 = pin->GetOrAddReal("problem", "vx0", 0.0);
    bb.blob_e = pin->GetOrAddReal("problem", "blob_e", 0.0);
    bb.blob_r = pin->GetOrAddReal("problem", "blob_r", 1.0);
    bb.rho_coef = 0.0;
    bb.p_over_rho = bb.pgas0/bb.rho0;
  }
  bb.arad = (pmbp->prad != nullptr) ? pmbp->prad->arad : 0.0;
  bb.gm1 = ((pmbp->pmhd != nullptr) ? pmbp->pmhd->peos->eos_data.gamma
             : pmbp->phydro->peos->eos_data.gamma) - 1.0;
  bb.flat = pmbp->pcoord->coord_data.is_minkowski;
  bb.spin = pmbp->pcoord->coord_data.bh_spin;
  if (global_variable::my_rank == 0 && bb.setup == 0) {
    std::cout << "bhstar_bondi: Mdot = " << mdot_cgs << " g/s = "
              << 4.0*M_PI*bb.rho_coef << " (code), rho(r=10) = "
              << bb.rho_coef/(100.0*sqrt(0.2)) << " (code), T_res = " << t_res
              << " K = " << bb.p_over_rho << " (code)" << std::endl;
  }
  if (restart) return;

  // select fluid arrays
  DvceArray5D<Real> w0_, u0_;
  if (pmbp->pmhd != nullptr) {
    w0_ = pmbp->pmhd->w0;
    u0_ = pmbp->pmhd->u0;
  } else if (pmbp->phydro != nullptr) {
    w0_ = pmbp->phydro->w0;
    u0_ = pmbp->phydro->u0;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "bhstar_bondi needs <hydro> or <mhd>" << std::endl;
    exit(EXIT_FAILURE);
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &js = indcs.js; int &ks = indcs.ks;
  int n1 = indcs.nx1 + 2*indcs.ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*indcs.ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*indcs.ng) : 1;
  int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;
  auto p = bb;

  // fluid primitives everywhere, including ghost zones
  par_for("pgen_bhstar_bondi", DevExeSpace(), 0,(nmb-1),0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x1v = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real x2v = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real x3v = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    Real rho, pgas, uu1, uu2, uu3;
    FreeFallState(x1v, x2v, x3v, p, rho, pgas, uu1, uu2, uu3);
    w0_(m,IDN,k,j,i) = rho;
    w0_(m,IEN,k,j,i) = pgas/p.gm1;
    w0_(m,IVX,k,j,i) = uu1;
    w0_(m,IVY,k,j,i) = uu2;
    w0_(m,IVZ,k,j,i) = uu3;
  });

  // B = 0 and conserved variables
  if (pmbp->pmhd != nullptr) {
    auto &b0 = pmbp->pmhd->b0;
    Kokkos::deep_copy(b0.x1f, 0.0);
    Kokkos::deep_copy(b0.x2f, 0.0);
    Kokkos::deep_copy(b0.x3f, 0.0);
    Kokkos::deep_copy(pmbp->pmhd->bcc0, 0.0);
    pmbp->pmhd->peos->PrimToCons(w0_, pmbp->pmhd->bcc0, u0_, 0, (n1-1), 0, (n2-1),
                                 0, (n3-1));
  } else {
    pmbp->phydro->peos->PrimToCons(w0_, u0_, 0, (n1-1), 0, (n2-1), 0, (n3-1));
  }

  // radiation: LTE at T = p/rho, isotropic in the fluid frame. In the fluid frame
  // I_cm = arad T^4 c/(4 pi) for every direction; the stored variable is
  // i0 = n^0 n_0 I_cm/(4 pi n0_cm^4) in the code's normalisation (as in gr_torus).
  if (pmbp->prad != nullptr) {
    auto &i0_ = pmbp->prad->i0;
    auto &nh_c_ = pmbp->prad->nh_c;
    auto &tet_c_ = pmbp->prad->tet_c;
    auto &tetcov_c_ = pmbp->prad->tetcov_c;
    auto &norm_to_tet_ = pmbp->prad->norm_to_tet;
    int nang = pmbp->prad->prgeo->nangles;
    par_for("pgen_bhstar_bondi_rad", DevExeSpace(), 0,(nmb-1),0,(n3-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real x1v = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      Real x2v = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      Real x3v = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1v, x2v, x3v, p.flat, p.spin, glower, gupper);
      Real uu1 = w0_(m,IVX,k,j,i), uu2 = w0_(m,IVY,k,j,i), uu3 = w0_(m,IVZ,k,j,i);
      Real q = glower[1][1]*uu1*uu1 + 2.0*glower[1][2]*uu1*uu2 + 2.0*glower[1][3]*uu1*uu3
             + glower[2][2]*uu2*uu2 + 2.0*glower[2][3]*uu2*uu3 + glower[3][3]*uu3*uu3;
      Real uu0 = sqrt(1.0 + q);
      Real u_tet[4];
      for (int a=0; a<4; ++a) {
        u_tet[a] = norm_to_tet_(m,a,0,k,j,i)*uu0 + norm_to_tet_(m,a,1,k,j,i)*uu1
                 + norm_to_tet_(m,a,2,k,j,i)*uu2 + norm_to_tet_(m,a,3,k,j,i)*uu3;
      }
      Real temp = p.p_over_rho;
      Real urad = p.arad*SQR(SQR(temp));
      if (p.setup == 1) {
        urad = p.blob_e*exp(-0.5*(SQR(x1v) + SQR(x2v) + SQR(x3v))/SQR(p.blob_r));
      }
      Real n0 = tet_c_(m,0,0,k,j,i);
      for (int n=0; n<nang; ++n) {
        Real n0_f = u_tet[0]*nh_c_.d_view(n,0) - u_tet[1]*nh_c_.d_view(n,1)
                  - u_tet[2]*nh_c_.d_view(n,2) - u_tet[3]*nh_c_.d_view(n,3);
        Real n_0 = 0.0;
        for (int d=0; d<4; ++d) { n_0 += tetcov_c_(m,d,0,k,j,i)*nh_c_.d_view(n,d); }
        i0_(m,n,k,j,i) = n0*n_0*(urad/(4.0*M_PI))/SQR(SQR(n0_f));
      }
    });
    // fix the shell-source normalisation now, before any output is written (the
    // value is stored in the parameter file and reused on restart)
    if (pmbp->prad->psrc != nullptr && pmbp->prad->psrc->rad_shell
        && !pmbp->prad->psrc->shell_norm_set) {
      pmbp->prad->psrc->ShellNormalise();
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn BhstarBondiBC
//! \brief User BC on all faces (see file header). Called once per stage from
//! Radiation::ApplyPhysicalBCs (after the zoom mask) or (M)HD::ApplyPhysicalBCs.

void BhstarBondiBC(Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;  int &ie = indcs.ie;
  int &js = indcs.js;  int &je = indcs.je;
  int &ks = indcs.ks;  int &ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  auto &mb_bcs = pmbp->pmb->mb_bcs;
  auto &size = pmbp->pmb->mb_size;
  auto p = bb;

  DvceArray5D<Real> w0_, u0_;
  if (pmbp->pmhd != nullptr) {
    w0_ = pmbp->pmhd->w0;
    u0_ = pmbp->pmhd->u0;
  } else {
    w0_ = pmbp->phydro->w0;
    u0_ = pmbp->phydro->u0;
  }
  const bool mhd = (pmbp->pmhd != nullptr);

  // ---- fluid: analytic state in all ghost zones of user faces (B = 0) ----
  // a ghost cell is set if it lies outside the active range in a direction whose face
  // is a user boundary (edges/corners included)
  par_for("bhstar_bc_fluid", DevExeSpace(), 0,(nmb-1),0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    bool set = false;
    if (i < is && mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) set = true;
    if (i > ie && mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) set = true;
    if (n2 > 1) {
      if (j < js && mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) set=true;
      if (j > je && mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) set=true;
    }
    if (n3 > 1) {
      if (k < ks && mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) set=true;
      if (k > ke && mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) set=true;
    }
    if (!set) return;
    Real x1v = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real x2v = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real x3v = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    Real rho, pgas, uu1, uu2, uu3;
    FreeFallState(x1v, x2v, x3v, p, rho, pgas, uu1, uu2, uu3);
    w0_(m,IDN,k,j,i) = rho;
    w0_(m,IEN,k,j,i) = pgas/p.gm1;
    w0_(m,IVX,k,j,i) = uu1;
    w0_(m,IVY,k,j,i) = uu2;
    w0_(m,IVZ,k,j,i) = uu3;
  });
  if (mhd) {
    // B = 0 in the ghost zones of user faces (face fields of ghost cells)
    auto &b0 = pmbp->pmhd->b0;
    par_for("bhstar_bc_b", DevExeSpace(), 0,(nmb-1),0,(n3-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      bool ghost_user =
        (i < is && mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) ||
        (i > ie && mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) ||
        (n2 > 1 && j < js && mb_bcs.d_view(m,BoundaryFace::inner_x2)==BoundaryFlag::user) ||
        (n2 > 1 && j > je && mb_bcs.d_view(m,BoundaryFace::outer_x2)==BoundaryFlag::user) ||
        (n3 > 1 && k < ks && mb_bcs.d_view(m,BoundaryFace::inner_x3)==BoundaryFlag::user) ||
        (n3 > 1 && k > ke && mb_bcs.d_view(m,BoundaryFace::outer_x3)==BoundaryFlag::user);
      if (!ghost_user) return;
      b0.x1f(m,k,j,i) = 0.0; b0.x2f(m,k,j,i) = 0.0; b0.x3f(m,k,j,i) = 0.0;
      if (i == n1-1) b0.x1f(m,k,j,i+1) = 0.0;
      if (j == n2-1) b0.x2f(m,k,j+1,i) = 0.0;
      if (k == n3-1) b0.x3f(m,k+1,j,i) = 0.0;
    });
    auto &bcc0_ = pmbp->pmhd->bcc0;
    // PrimToCons on the ghost slabs (bcc is zero there: B = 0 everywhere in Stage 0)
    pmbp->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,is-1,0,(n2-1),0,(n3-1));
    pmbp->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,ie+1,n1-1,0,(n2-1),0,(n3-1));
    if (n2 > 1) {
      pmbp->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,(n1-1),0,js-1,0,(n3-1));
      pmbp->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,(n1-1),je+1,n2-1,0,(n3-1));
    }
    if (n3 > 1) {
      pmbp->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,(n1-1),0,(n2-1),0,ks-1);
      pmbp->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,(n1-1),0,(n2-1),ke+1,n3-1);
    }
  } else {
    pmbp->phydro->peos->PrimToCons(w0_,u0_,0,is-1,0,(n2-1),0,(n3-1));
    pmbp->phydro->peos->PrimToCons(w0_,u0_,ie+1,n1-1,0,(n2-1),0,(n3-1));
    if (n2 > 1) {
      pmbp->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),0,js-1,0,(n3-1));
      pmbp->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),je+1,n2-1,0,(n3-1));
    }
    if (n3 > 1) {
      pmbp->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),0,(n2-1),0,ks-1);
      pmbp->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),0,(n2-1),ke+1,n3-1);
    }
  }
  // Note: PrimToCons on these slabs also rewrites ghost cells that belong to block
  // (internal) faces, from the w0 they already hold. Those w0 come from the last
  // ConToPrim of the same u0, so this only round-trips them; they are overwritten by the
  // next boundary exchange anyway.

  // ---- radiation: outgoing rays copied, incoming rays zero (vacuum) ----
  if (pmbp->prad == nullptr) return;
  auto &i0_ = pmbp->prad->i0;
  auto &nh_c_ = pmbp->prad->nh_c;
  auto &tet_c_ = pmbp->prad->tet_c;
  int nang1 = pmbp->prad->prgeo->nangles - 1;
  // coordinate direction n^d = sum_a e_(a)^d nh^a (tet_c(m,a,d) = e_(a)^d)
  par_for("bhstar_bc_rad_x1", DevExeSpace(),0,(nmb-1),0,nang1,0,(n3-1),0,(n2-1),
  KOKKOS_LAMBDA(int m, int n, int k, int j) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
      for (int g=0; g<ng; ++g) {
        int i = is-g-1;
        Real nd = 0.0;
        for (int a=0; a<4; ++a) nd += tet_c_(m,a,1,k,j,i)*nh_c_.d_view(n,a);
        i0_(m,n,k,j,i) = (nd < 0.0) ? i0_(m,n,k,j,is) : 0.0;
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
      for (int g=0; g<ng; ++g) {
        int i = ie+g+1;
        Real nd = 0.0;
        for (int a=0; a<4; ++a) nd += tet_c_(m,a,1,k,j,i)*nh_c_.d_view(n,a);
        i0_(m,n,k,j,i) = (nd > 0.0) ? i0_(m,n,k,j,ie) : 0.0;
      }
    }
  });
  if (n2 > 1) {
    par_for("bhstar_bc_rad_x2", DevExeSpace(),0,(nmb-1),0,nang1,0,(n3-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int n, int k, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
        for (int g=0; g<ng; ++g) {
          int j = js-g-1;
          Real nd = 0.0;
          for (int a=0; a<4; ++a) nd += tet_c_(m,a,2,k,j,i)*nh_c_.d_view(n,a);
          i0_(m,n,k,j,i) = (nd < 0.0) ? i0_(m,n,k,js,i) : 0.0;
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
        for (int g=0; g<ng; ++g) {
          int j = je+g+1;
          Real nd = 0.0;
          for (int a=0; a<4; ++a) nd += tet_c_(m,a,2,k,j,i)*nh_c_.d_view(n,a);
          i0_(m,n,k,j,i) = (nd > 0.0) ? i0_(m,n,k,je,i) : 0.0;
        }
      }
    });
  }
  if (n3 > 1) {
    par_for("bhstar_bc_rad_x3", DevExeSpace(),0,(nmb-1),0,nang1,0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int n, int j, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
        for (int g=0; g<ng; ++g) {
          int k = ks-g-1;
          Real nd = 0.0;
          for (int a=0; a<4; ++a) nd += tet_c_(m,a,3,k,j,i)*nh_c_.d_view(n,a);
          i0_(m,n,k,j,i) = (nd < 0.0) ? i0_(m,n,ks,j,i) : 0.0;
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
        for (int g=0; g<ng; ++g) {
          int k = ke+g+1;
          Real nd = 0.0;
          for (int a=0; a<4; ++a) nd += tet_c_(m,a,3,k,j,i)*nh_c_.d_view(n,a);
          i0_(m,n,k,j,i) = (nd > 0.0) ? i0_(m,n,ke,j,i) : 0.0;
        }
      }
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn BhstarBudgetHistory
//! \brief Energy budget history (rank sums; the history output adds them over ranks).
//! Closure: (E_gas + E_rad)(t) - (E_gas + E_rad)(0) = inj + rad_upd + gas_upd + gas_src
//!          + coupl + c2p + mask + regrid  (all cumulative since the start of this run).

void BhstarBudgetHistory(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  radiation::RadBudget *pb = (pmbp->prad != nullptr) ? pmbp->prad->pbudget : nullptr;
  const char *names[radiation::kNBudget] = {
    "inj", "rad_upd", "gas_upd", "gas_src", "coupl", "c2p", "mask", "regrid",
    "rad_flux", "rad_floor", "rad_exc", "gas_flux", "coupl_c2p", "coupl_exc"};
  int nh = 0;
  pdata->label[nh] = "E_gas";
  pdata->hdata[nh++] = (pb != nullptr) ? pb->GasEnergy() : 0.0;
  pdata->label[nh] = "E_rad";
  pdata->hdata[nh++] = (pb != nullptr) ? pb->RadEnergy() : 0.0;
  for (int x=0; x<radiation::kNBudget; ++x) {
    pdata->label[nh] = names[x];
    pdata->hdata[nh++] = (pb != nullptr) ? pb->c[x] : 0.0;
  }
  pdata->label[nh] = "n_badcell";
  pdata->hdata[nh++] = (pb != nullptr) ? pb->nbadcell : 0.0;
  pdata->label[nh] = "M1_rad";   // sum_n w_n i0_n n_1/n_0 dV (x-momentum, as in coupling)
  pdata->hdata[nh++] = (pmbp->prad != nullptr) ? RadMomentumX(pmbp) : 0.0;
  pdata->nhist = nh;
}

//----------------------------------------------------------------------------------------
//! \fn RadMomentumX
//! \brief sum over active cells and angles of w_n i0_n n_1/n_0 dV: the radiation
//! x-momentum in the form the coupling adds to the fluid momentum IM1 (unit test U1)

Real RadMomentumX(MeshBlockPack *pmbp) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int nang = pmbp->prad->prgeo->nangles;
  const int nkji = nx3*nx2*nx1, nji = nx2*nx1, nnkji = nang*nkji;
  const int ntot = (pmbp->nmb_thispack)*nnkji;
  auto &size = pmbp->pmb->mb_size;
  auto &i0_ = pmbp->prad->i0;
  auto &w_ = pmbp->prad->prgeo->solid_angles;
  auto &tc = pmbp->prad->tetcov_c;
  auto &nh_c_ = pmbp->prad->nh_c;
  Real sum = 0.0;
  Kokkos::parallel_reduce("bhstar_m1rad", Kokkos::RangePolicy<>(DevExeSpace(), 0, ntot),
  KOKKOS_LAMBDA(const int &idx, Real &s) {
    int m = idx/nnkji;
    int n = (idx - m*nnkji)/nkji;
    int k = (idx - m*nnkji - n*nkji)/nji;
    int j = (idx - m*nnkji - n*nkji - k*nji)/nx1;
    int i = (idx - m*nnkji - n*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    Real n_0 = 0.0, n_1 = 0.0;
    for (int a=0; a<4; ++a) {
      n_0 += tc(m,a,0,k,j,i)*nh_c_.d_view(n,a);
      n_1 += tc(m,a,1,k,j,i)*nh_c_.d_view(n,a);
    }
    Real dv = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    s += w_.d_view(n)*i0_(m,n,k,j,i)*n_1/n_0*dv;
  }, Kokkos::Sum<Real>(sum));
  return sum;
}
