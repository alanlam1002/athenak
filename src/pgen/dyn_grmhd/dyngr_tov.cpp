//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyngr_tov.cpp
//  \brief Problem generator for TOV star. Only works when ADM is enabled.

#include <math.h>     // abs(), cos(), exp(), log(), NAN, pow(), sin(), sqrt()

#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "utils/tov/tov.hpp"
#include "utils/tov/tov_polytrope.hpp"
#include "utils/tov/tov_tabulated.hpp"
#include "utils/tov/tov_piecewise_poly.hpp"

#include <Kokkos_Random.hpp>

// Prototypes for vector potential
template<class TOVEOS>
KOKKOS_INLINE_FUNCTION
static Real A1(const tov::TOVStar& tov_, const TOVEOS& eos, bool isotropic, Real pcut,
               Real magindex, Real x1, Real x2, Real x3);
template<class TOVEOS>
KOKKOS_INLINE_FUNCTION
static Real A2(const tov::TOVStar& tov_, const TOVEOS& eos, bool isotropic, Real pcut,
               Real magindex, Real x1, Real x2, Real x3);

// Prototypes for user-defined BCs and history
void TOVHistory(HistoryData *pdata, Mesh *pm);

// Prototype for the optional tracked-refinement criterion (<problem>
// amr_condition = tde_track). See TDERefineTracker's own comment below.
void TDERefineTracker(MeshBlockPack *pmbp);

namespace {
// Parameters for TDERefineTracker (<problem> amr_condition = tde_track).
struct TDERefineParams {
  Real r0;          // star's initial x-position (<problem> star_center_x1),
                    // used only to seed x1_star before the first reduction
  Real rad_bh;      // refinement radius around the (fixed) puncture at the origin
  Real rad_star;    // refinement radius around the star
  Real rad_bh_fine; // inner, higher-level radius around the puncture (0 disables)
  int  lev_bh;      // target level within rad_bh
  int  lev_star;    // target level within rad_star
  int  lev_bh_fine; // target level within rad_bh_fine (must exceed lev_bh)
};
TDERefineParams tde_ref;
} // namespace

namespace {
struct TOVParams {
  tov::TOVStar my_tov;
  bool isotropic;
  bool minkowski;
  // Star-center offset (<problem> star_center_x1/2/3, default 0.0 -- every existing
  // fixture is unaffected). Lets the star sit away from the coordinate origin, which
  // the (separate, always-at-x=0) CFC BH puncture background does not move with.
  Real x0, y0, z0;

  TOVParams(tov::TOVStar& tov_star, bool isotropic_, bool minkowski_,
            Real x0_, Real y0_, Real z0_) :
      my_tov(std::move(tov_star)) {
    isotropic = isotropic_;
    minkowski = minkowski_;
    x0 = x0_; y0 = y0_; z0 = z0_;
  }
};

TOVParams *ptov_params;
} // namespace

void SetADMVariablesToTOV(MeshBlockPack *pmbp);
void FinalizeTOV(ParameterInput *pin, Mesh *pm);

template<class TOVEOS>
void SolveTOV(ParameterInput *pin, Mesh* pmy_mesh_) {
  bool isotropic = pin->GetOrAddBoolean("problem", "isotropic", false);
  bool minkowski = pin->GetOrAddBoolean("problem", "minkowski", false);
  Real x0 = pin->GetOrAddReal("problem", "star_center_x1", 0.0);
  Real y0 = pin->GetOrAddReal("problem", "star_center_x2", 0.0);
  Real z0 = pin->GetOrAddReal("problem", "star_center_x3", 0.0);

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  // If the metric is adaptive or dynamical ADM is enabled, we need to regenerate the
  // TOV solution, since it is not stored in the restart file.
  if (pmbp->padm->is_dynamic || pmy_mesh_->adaptive) {
    TOVEOS eos{pin};
    auto my_tov = tov::TOVStar::ConstructTOV(pin, eos, false);
    ptov_params = new TOVParams(my_tov, isotropic, minkowski, x0, y0, z0);
  }
}

template<class TOVEOS>
void SetupTOV(ParameterInput *pin, Mesh* pmy_mesh_) {
  Real v_pert = pin->GetOrAddReal("problem", "v_pert", 0.0);
  Real p_pert = pin->GetOrAddReal("problem", "p_pert", 0.0);
  bool isotropic = pin->GetOrAddBoolean("problem", "isotropic", false);
  Real x0 = pin->GetOrAddReal("problem", "star_center_x1", 0.0);
  Real y0 = pin->GetOrAddReal("problem", "star_center_x2", 0.0);
  Real z0 = pin->GetOrAddReal("problem", "star_center_x3", 0.0);

  // Uniform bulk (center-of-mass) boost of the whole star, for elliptic-orbit
  // setups -- see scripts/tde_elliptical_orbit_ic.py. Distinct from v_pert above
  // (a zero-net-momentum radial pulsation). Default 0.0: has_boost=false skips
  // the Lorentz-factor multiply, keeping every non-boosted fixture bit-for-bit
  // unchanged.
  Real star_vel_x1 = pin->GetOrAddReal("problem", "star_vel_x1", 0.0);
  Real star_vel_x2 = pin->GetOrAddReal("problem", "star_vel_x2", 0.0);
  Real star_vel_x3 = pin->GetOrAddReal("problem", "star_vel_x3", 0.0);
  bool has_boost = (star_vel_x1 != 0.0) || (star_vel_x2 != 0.0) || (star_vel_x3 != 0.0);
  if (has_boost && !isotropic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "star_vel_x1/2/3 (bulk boost) requires <problem>/isotropic=true "
              << "(CFC's own gauge); the non-isotropic branch's metric assembly is not "
              << "shifted to star_center and was never validated for this feature."
              << std::endl;
    exit(EXIT_FAILURE);
  }

  bool minkowski = pin->GetOrAddBoolean("problem", "minkowski", false);

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  // Use the TOV solver with the specified EOS.
  TOVEOS eos{pin};
  auto my_tov = tov::TOVStar::ConstructTOV(pin, eos);


  constexpr bool use_ye = tov::UsesYe<TOVEOS>;
  Real ye_atmo = pin->GetOrAddReal("mhd", "s0_atmosphere", 0.5);

  //auto& u0_ = pmbp->pmhd->u0;
  auto& w0_ = pmbp->pmhd->w0;
  int& nvars_ = pmbp->pmhd->nmhd;
  int& nscal_ = pmbp->pmhd->nscalars;

  // Capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  int &ie = indcs.ie;
  int &je = indcs.je;
  int &ke = indcs.ke;
  int nmb1 = pmbp->nmb_thispack - 1;

  auto &size = pmbp->pmb->mb_size;
  auto &adm = pmbp->padm->adm;
  auto &tov_ = my_tov;
  auto &eos_ = eos;
  Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
  par_for("pgen_tov1", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    // Calculate the rest-mass density, pressure, and mass for a specific isotropic
    // radial coordinate, relative to the star's own center (x0,y0,z0) -- 0 by default,
    // so this is a no-op for every existing fixture. Only the isotropic branch below
    // (mandatory for CFC) also shifts the metric assembly to match; the non-isotropic
    // (Schwarzschild-gauge) branch's own x1v*x1v-type metric terms are NOT shifted,
    // since CFC never uses that branch and star_center is only meant to be combined
    // with isotropic=true.
    Real r = sqrt(SQR(x1v-x0) + SQR(x2v-y0) + SQR(x3v-z0));
    Real s = sqrt(SQR(x1v-x0) + SQR(x2v-y0));
    Real rho, p, mass, alp, r_schw;
    Real vr = 0.;
    Real p_pert = 0.;
    Real ye = ye_atmo;
    Real vbx = 0., vby = 0., vbz = 0.;
    auto &use_ye_ = use_ye;
    if (!isotropic) {
      tov_.GetPrimitivesAtPoint(eos_, r, rho, p, mass, alp);
      if (r <= tov_.R_edge) {
        Real x = r/tov_.R_edge;
        vr = 0.5*v_pert*(3.0*x - x*x*x);
        auto rand_gen = rand_pool64.get_state();
        p_pert = 2.0*p_pert*(rand_gen.frand() - 0.5);
        rand_pool64.free_state(rand_gen);
        if constexpr (use_ye) {
          ye = eos_.template GetYeFromRho<tov::LocationTag::Device>(rho);
        }
      }
    } else {
      tov_.GetPrimitivesAtIsoPoint(eos_, r, rho, p, mass, alp);
      r_schw = tov_.FindSchwarzschildR(r, mass);
      if (r_schw <= tov_.R_edge) {
        Real x = r_schw/tov_.R_edge;
        vr = 0.5*v_pert*(3.0*x - x*x*x);
        vbx = star_vel_x1; vby = star_vel_x2; vbz = star_vel_x3;
        auto rand_gen = rand_pool64.get_state();
        p_pert = 2.0*p_pert*(rand_gen.frand() - 0.5);
        rand_pool64.free_state(rand_gen);
        if constexpr (use_ye) {
          ye = eos_.template GetYeFromRho<tov::LocationTag::Device>(rho);
        }
      }
    }

    // Set hydrodynamic quantities
    //w0_(m,IDN,k,j,i) = fmax(rho, tov_.dfloor);
    //w0_(m,IPR,k,j,i) = fmax(p*(1. + p_pert), tov_.pfloor);
    w0_(m,IDN,k,j,i) = rho;
    w0_(m,IPR,k,j,i) = p*(1. + p_pert);

    // Bulk boost added to the v_pert radial pulsation before one combined
    // Lorentz-factor conversion. vbx/vby/vbz are nonzero only inside the star
    // (same gate v_pert uses), matching its atmosphere-stays-at-rest behavior.
    Real vx_tot = vr*(x1v-x0)/r + vbx;
    Real vy_tot = vr*(x2v-y0)/r + vby;
    Real vz_tot = vr*(x3v-z0)/r + vbz;
    if (has_boost) {
      // gamma_ij = psi4*delta_ij (CFC gauge); psi4 of the star's own isolated TOV
      // field -- same template as xns_rotstar.cpp's Wv assembly.
      Real fmet_local = 1.;
      if (r > 0) { fmet_local = r_schw/r; }
      Real psi4_local = fmet_local*fmet_local;
      Real vsq = psi4_local*(vx_tot*vx_tot + vy_tot*vy_tot + vz_tot*vz_tot);
      vsq = fmin(vsq, 0.9999);
      Real lorentz_w = 1.0/sqrt(1.0 - vsq);
      w0_(m,IVX,k,j,i) = lorentz_w*vx_tot;
      w0_(m,IVY,k,j,i) = lorentz_w*vy_tot;
      w0_(m,IVZ,k,j,i) = lorentz_w*vz_tot;
    } else {
      // Bit-identical to the pre-boost expression: vx_tot == vr*(unit)+0.0.
      w0_(m,IVX,k,j,i) = vx_tot;
      w0_(m,IVY,k,j,i) = vy_tot;
      w0_(m,IVZ,k,j,i) = vz_tot;
    }
    auto &nvars = nvars_;
    auto &nscal = nscal_;
    if (use_ye && nscal >= 1) {
      w0_(m,nvars,k,j,i) = ye;
    }

    // Set ADM variables
    adm.alpha(m,k,j,i) = alp;
    if (minkowski) {
      adm.g_dd(m,0,0,k,j,i) = adm.g_dd(m,1,1,k,j,i) = adm.g_dd(m,2,2,k,j,i) = 1.0;
      adm.g_dd(m,0,1,k,j,i) = adm.g_dd(m,0,2,k,j,i) = adm.g_dd(m,1,2,k,j,i) = 0.0;
      adm.alpha(m,k,j,i) = 1.0;
    } else if (!isotropic) {
      // Auxiliary metric quantities
      Real fmet = 0.0;
      if (r > 0) {
        fmet = (1./(1. - 2*mass/r) - 1.)/(r*r);
      }

      adm.g_dd(m,0,0,k,j,i) = x1v*x1v*fmet + 1.0;
      adm.g_dd(m,0,1,k,j,i) = x1v*x2v*fmet;
      adm.g_dd(m,0,2,k,j,i) = x1v*x3v*fmet;
      adm.g_dd(m,1,1,k,j,i) = x2v*x2v*fmet + 1.0;
      adm.g_dd(m,1,2,k,j,i) = x2v*x3v*fmet;
      adm.g_dd(m,2,2,k,j,i) = x3v*x3v*fmet + 1.0;
      Real det = adm::SpatialDet(
              adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
              adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
              adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i));
      adm.psi4(m,k,j,i) = pow(det, 1./3.);
    } else {
      Real fmet = 1.;
      if (r > 0) {
        fmet = r_schw/r;
      }
      Real psi4 = fmet*fmet;

      adm.g_dd(m,0,0,k,j,i) = adm.g_dd(m,1,1,k,j,i) = adm.g_dd(m,2,2,k,j,i) = psi4;
      adm.g_dd(m,0,1,k,j,i) = adm.g_dd(m,0,2,k,j,i) = adm.g_dd(m,1,2,k,j,i) = 0.0;
      adm.psi4(m,k,j,i) = psi4;
    }
    adm.beta_u(m,0,k,j,i) = adm.beta_u(m,1,k,j,i) = adm.beta_u(m,2,k,j,i) = 0.0;
    adm.vK_dd(m,0,0,k,j,i) = adm.vK_dd(m,0,1,k,j,i) = adm.vK_dd(m,0,2,k,j,i) = 0.0;
    adm.vK_dd(m,1,1,k,j,i) = adm.vK_dd(m,1,2,k,j,i) = adm.vK_dd(m,2,2,k,j,i) = 0.0;
  });

  // parse some parameters
  Real b_norm = pin->GetOrAddReal("problem", "b_norm", 0.0);
  Real pcut = pin->GetOrAddReal("problem", "pcut", 1e-6);
  Real magindex = pin->GetOrAddReal("problem", "magindex", 2);

  // If use_pcut_rel = true, we take pcut to be a percentage of pmax rather than
  // an absolute cutoff
  if (pin->GetOrAddBoolean("problem", "use_pcut_rel", false)) {
    Real pmax = eos_.template GetPFromRho<tov::LocationTag::Host>(tov_.rhoc);
    pcut = pcut * pmax;
  }

  // compute vector potential over all faces
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*(indcs.ng)) : 1;
  int nmb = pmbp->nmb_thispack;
  DvceArray4D<Real> a1, a2, a3;
  Kokkos::realloc(a1, nmb, ncells3, ncells2, ncells1);
  Kokkos::realloc(a2, nmb, ncells3, ncells2, ncells1);
  Kokkos::realloc(a3, nmb, ncells3, ncells2, ncells1);

  auto &nghbr = pmbp->pmb->nghbr;
  auto &mblev = pmbp->pmb->mb_lev;

  par_for("pgen_potential", DevExeSpace(), 0,nmb-1,ks,ke+1,js,je+1,is,ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
    Real x1f = LeftEdgeX(i-is,nx1,x1min,x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
    Real x2f = LeftEdgeX(j-js,nx2,x2min,x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
    Real x3f = LeftEdgeX(k-ks,nx3,x3min,x3max);

    Real x1fp1 = LeftEdgeX(i+1-is, nx1, x1min, x1max);
    Real x2fp1 = LeftEdgeX(j+1-js, nx2, x2min, x2max);
    Real x3fp1 = LeftEdgeX(k+1-ks, nx3, x3min, x3max);
    Real dx1 = size.d_view(m).dx1;
    Real dx2 = size.d_view(m).dx2;
    Real dx3 = size.d_view(m).dx3;

    a1(m,k,j,i) = A1(tov_, eos_, isotropic, pcut, magindex, x1v, x2f, x3f);
    a2(m,k,j,i) = A2(tov_, eos_, isotropic, pcut, magindex, x1f, x2v, x3f);
    a3(m,k,j,i) = 0.0;

    // When neighboring MeshBock is at finer level, compute vector potential as sum of
    // values at fine grid resolution.  This guarantees flux on shared fine/coarse
    // faces is identical.

    // Correct A1 at x2-faces, x3-faces, and x2x3-edges
    if ((nghbr.d_view(m,8 ).lev > mblev.d_view(m) && j==js) ||
        (nghbr.d_view(m,9 ).lev > mblev.d_view(m) && j==js) ||
        (nghbr.d_view(m,10).lev > mblev.d_view(m) && j==js) ||
        (nghbr.d_view(m,11).lev > mblev.d_view(m) && j==js) ||
        (nghbr.d_view(m,12).lev > mblev.d_view(m) && j==je+1) ||
        (nghbr.d_view(m,13).lev > mblev.d_view(m) && j==je+1) ||
        (nghbr.d_view(m,14).lev > mblev.d_view(m) && j==je+1) ||
        (nghbr.d_view(m,15).lev > mblev.d_view(m) && j==je+1) ||
        (nghbr.d_view(m,24).lev > mblev.d_view(m) && k==ks) ||
        (nghbr.d_view(m,25).lev > mblev.d_view(m) && k==ks) ||
        (nghbr.d_view(m,26).lev > mblev.d_view(m) && k==ks) ||
        (nghbr.d_view(m,27).lev > mblev.d_view(m) && k==ks) ||
        (nghbr.d_view(m,28).lev > mblev.d_view(m) && k==ke+1) ||
        (nghbr.d_view(m,29).lev > mblev.d_view(m) && k==ke+1) ||
        (nghbr.d_view(m,30).lev > mblev.d_view(m) && k==ke+1) ||
        (nghbr.d_view(m,31).lev > mblev.d_view(m) && k==ke+1) ||
        (nghbr.d_view(m,40).lev > mblev.d_view(m) && j==js && k==ks) ||
        (nghbr.d_view(m,41).lev > mblev.d_view(m) && j==js && k==ks) ||
        (nghbr.d_view(m,42).lev > mblev.d_view(m) && j==je+1 && k==ks) ||
        (nghbr.d_view(m,43).lev > mblev.d_view(m) && j==je+1 && k==ks) ||
        (nghbr.d_view(m,44).lev > mblev.d_view(m) && j==js && k==ke+1) ||
        (nghbr.d_view(m,45).lev > mblev.d_view(m) && j==js && k==ke+1) ||
        (nghbr.d_view(m,46).lev > mblev.d_view(m) && j==je+1 && k==ke+1) ||
        (nghbr.d_view(m,47).lev > mblev.d_view(m) && j==je+1 && k==ke+1)) {
      Real xl = x1v + 0.25*dx1;
      Real xr = x1v - 0.25*dx1;
      a1(m,k,j,i) = 0.5*(A1(tov_, eos_, isotropic, pcut, magindex, xl,x2f,x3f) +
                         A1(tov_, eos_, isotropic, pcut, magindex, xr,x2f,x3f));
    }

    // Correct A2 at x1-faces, x3-faces, and x1x3-edges
    if ((nghbr.d_view(m,0 ).lev > mblev.d_view(m) && i==is) ||
        (nghbr.d_view(m,1 ).lev > mblev.d_view(m) && i==is) ||
        (nghbr.d_view(m,2 ).lev > mblev.d_view(m) && i==is) ||
        (nghbr.d_view(m,3 ).lev > mblev.d_view(m) && i==is) ||
        (nghbr.d_view(m,4 ).lev > mblev.d_view(m) && i==ie+1) ||
        (nghbr.d_view(m,5 ).lev > mblev.d_view(m) && i==ie+1) ||
        (nghbr.d_view(m,6 ).lev > mblev.d_view(m) && i==ie+1) ||
        (nghbr.d_view(m,7 ).lev > mblev.d_view(m) && i==ie+1) ||
        (nghbr.d_view(m,24).lev > mblev.d_view(m) && k==ks) ||
        (nghbr.d_view(m,25).lev > mblev.d_view(m) && k==ks) ||
        (nghbr.d_view(m,26).lev > mblev.d_view(m) && k==ks) ||
        (nghbr.d_view(m,27).lev > mblev.d_view(m) && k==ks) ||
        (nghbr.d_view(m,28).lev > mblev.d_view(m) && k==ke+1) ||
        (nghbr.d_view(m,29).lev > mblev.d_view(m) && k==ke+1) ||
        (nghbr.d_view(m,30).lev > mblev.d_view(m) && k==ke+1) ||
        (nghbr.d_view(m,31).lev > mblev.d_view(m) && k==ke+1) ||
        (nghbr.d_view(m,32).lev > mblev.d_view(m) && i==is && k==ks) ||
        (nghbr.d_view(m,33).lev > mblev.d_view(m) && i==is && k==ks) ||
        (nghbr.d_view(m,34).lev > mblev.d_view(m) && i==ie+1 && k==ks) ||
        (nghbr.d_view(m,35).lev > mblev.d_view(m) && i==ie+1 && k==ks) ||
        (nghbr.d_view(m,36).lev > mblev.d_view(m) && i==is && k==ke+1) ||
        (nghbr.d_view(m,37).lev > mblev.d_view(m) && i==is && k==ke+1) ||
        (nghbr.d_view(m,38).lev > mblev.d_view(m) && i==ie+1 && k==ke+1) ||
        (nghbr.d_view(m,39).lev > mblev.d_view(m) && i==ie+1 && k==ke+1)) {
      Real xl = x2v + 0.25*dx2;
      Real xr = x2v - 0.25*dx2;
      a2(m,k,j,i) = 0.5*(A2(tov_, eos_, isotropic, pcut, magindex, x1f,xl,x3f) +
                         A2(tov_, eos_, isotropic, pcut, magindex, x1f,xr,x3f));
    }
  });

  auto &b0 = pmbp->pmhd->b0;
  par_for("pgen_Bfc", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // Compute face-centered fields from curl(A).
    Real dx1 = size.d_view(m).dx1;
    Real dx2 = size.d_view(m).dx2;
    Real dx3 = size.d_view(m).dx3;

    b0.x1f(m,k,j,i) = b_norm*((a3(m,k,j+1,i) - a3(m,k,j,i))/dx2 -
                       (a2(m,k+1,j,i) - a2(m,k,j,i))/dx3);
    b0.x2f(m,k,j,i) = b_norm*((a1(m,k+1,j,i) - a1(m,k,j,i))/dx3 -
                       (a3(m,k,j,i+1) - a3(m,k,j,i))/dx1);
    b0.x3f(m,k,j,i) = b_norm*((a2(m,k,j,i+1) - a2(m,k,j,i))/dx1 -
                       (a1(m,k,j+1,i) - a1(m,k,j,i))/dx2);

    // Include extra face-component at edge of block in each direction
    if (i==ie) {
      b0.x1f(m,k,j,i+1) = b_norm*((a3(m,k,j+1,i+1) - a3(m,k,j,i+1))/dx2 -
                           (a2(m,k+1,j,i+1) - a2(m,k,j,i+1))/dx3);
    }
    if (j==je) {
      b0.x2f(m,k,j+1,i) = b_norm*((a1(m,k+1,j+1,i) - a1(m,k,j+1,i))/dx3 -
                           (a3(m,k,j+1,i+1) - a3(m,k,j+1,i))/dx1);
    }
    if (k==ke) {
      b0.x3f(m,k+1,j,i) = b_norm*((a2(m,k+1,j,i+1) - a2(m,k+1,j,i))/dx1 -
                           (a1(m,k+1,j+1,i) - a1(m,k+1,j,i))/dx2);
    }
  });

  // Compute cell-centered fields
  auto &bcc_ = pmbp->pmhd->bcc0;
  par_for("pgen_Bcc", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // cell-centered fields are simple linear average of face-centered fields
    Real& w_bx = bcc_(m,IBX,k,j,i);
    Real& w_by = bcc_(m,IBY,k,j,i);
    Real& w_bz = bcc_(m,IBZ,k,j,i);
    w_bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
    w_by = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
    w_bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
  });

  // Copy the TOV to another object for storage if needed.
  if (pmbp->padm->is_dynamic || pmy_mesh_->adaptive == true) {
    ptov_params = new TOVParams(my_tov, isotropic, minkowski, x0, y0, z0);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//  \brief Sets initial conditions for TOV star in DynGRMHD
//  Compile with '-D PROBLEM=dyngr_tov' to enroll as user-specific problem generator

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!pmbp->pcoord->is_dynamical_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "TOV star problem can only be run when <adm> block is present"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  user_hist_func = &TOVHistory;
  pgen_final_func = &FinalizeTOV;
  pmbp->padm->SetADMVariables = &SetADMVariablesToTOV;

  // Optional tracked refinement for the TDE setup (<problem> amr_condition).
  // Default "none" leaves user_ref_func null, so every existing fixture -- all of
  // which use static refinement -- is completely unaffected.
  if (pin->GetOrAddString("problem", "amr_condition", "none") == "tde_track") {
    tde_ref.r0       = pin->GetOrAddReal("problem", "star_center_x1", 0.0);
    tde_ref.rad_bh      = pin->GetOrAddReal("problem", "amr_radius_bh", 2.0);
    tde_ref.rad_star    = pin->GetOrAddReal("problem", "amr_radius_star", 2.0);
    tde_ref.rad_bh_fine = pin->GetOrAddReal("problem", "amr_radius_bh_fine", 0.0);
    tde_ref.lev_bh      = pin->GetOrAddInteger("problem", "amr_level_bh", 5);
    tde_ref.lev_star    = pin->GetOrAddInteger("problem", "amr_level_star", 5);
    tde_ref.lev_bh_fine = pin->GetOrAddInteger("problem", "amr_level_bh_fine", 6);
    user_ref_func = &TDERefineTracker;
  }

  // initialize primitive variables for restart
  if (restart) {
    if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_ideal) {
      SolveTOV<tov::PolytropeEOS>(pin, pmy_mesh_);
    } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_compose) {
      SolveTOV<tov::TabulatedEOS>(pin, pmy_mesh_);
    } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_hybrid) {
      SolveTOV<tov::TabulatedEOS>(pin, pmy_mesh_);
    } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_piecewise_poly) {
      SolveTOV<tov::PiecewisePolytropeEOS>(pin, pmy_mesh_);
    }
    return;
  }

  // Select the right TOV template based on the EOS we need.
  if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_ideal) {
    SetupTOV<tov::PolytropeEOS>(pin, pmy_mesh_);
  } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_compose) {
    SetupTOV<tov::TabulatedEOS>(pin, pmy_mesh_);
  } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_hybrid) {
    SetupTOV<tov::TabulatedEOS>(pin, pmy_mesh_);
  } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_piecewise_poly) {
    SetupTOV<tov::PiecewisePolytropeEOS>(pin, pmy_mesh_);
  } else {
    std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Unknown EOS requested for TOV star problem" << std::endl
              << "Defaulting to fixed polytropic EOS" << std::endl;
    SetupTOV<tov::PolytropeEOS>(pin, pmy_mesh_);
  }

  // Mesh block info for loop limits
  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;

  // Convert primitives to conserved
  pmbp->pdyngr->PrimToConInit(0, (n1-1), 0, (n2-1), 0, (n3-1));

  if (pmbp->pz4c != nullptr) {
    switch (indcs.ng) {
      case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
              pmbp->pz4c->ADMConstraints<2>(pmbp);
              break;
      case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
              pmbp->pz4c->ADMConstraints<3>(pmbp);
              break;
      case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
              pmbp->pz4c->ADMConstraints<4>(pmbp);
              break;
    }
  }

  return;
}

// NOT shifted by star_center_x1/2/3: B = b_norm*curl(A) (dyngr_tov.cpp:360-364) is a
// pure multiplicative scale with no normalization, so b_norm=0 (every existing/planned
// unmagnetized fixture, including the off-center TOV test) makes the resulting field
// identically zero regardless of what A1/A2 evaluate here. Revisit if a magnetized
// off-center star is ever needed.
template<class TOVEOS>
KOKKOS_INLINE_FUNCTION
static Real A1(const tov::TOVStar& tov_, const TOVEOS& eos, bool isotropic, Real pcut,
               Real magindex, Real x1, Real x2, Real x3) {
  Real r = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  Real p, rho;
  if (!isotropic) {
    tov_.GetPandRho(eos, r, rho, p);
  } else {
    tov_.GetPandRhoIso(eos, r, rho, p);
  }
  return -x2*fmax(p - pcut, 0.0)*pow(1.0 - rho/tov_.rhoc,magindex);
}

template<class TOVEOS>
KOKKOS_INLINE_FUNCTION
static Real A2(const tov::TOVStar& tov_, const TOVEOS& eos, bool isotropic, Real pcut,
               Real magindex, Real x1, Real x2, Real x3) {
  Real r = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  Real p, rho;
  if (!isotropic) {
    tov_.GetPandRho(eos, r, rho, p);
  } else {
    tov_.GetPandRhoIso(eos, r, rho, p);
  }
  return x1*fmax(p - pcut, 0.0)*pow(1.0 - rho/tov_.rhoc,magindex);
}

// Metric update function
void SetADMVariablesToTOV(MeshBlockPack *pmbp) {
  auto &adm = pmbp->padm->adm;
  auto &size = pmbp->pmb->mb_size;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;

  auto& tov_ = ptov_params->my_tov;
  bool isotropic = ptov_params->isotropic;
  bool minkowski = ptov_params->minkowski;
  Real x0 = ptov_params->x0, y0 = ptov_params->y0, z0 = ptov_params->z0;
  par_for("update_adm_vars", DevExeSpace(), 0,nmb-1,0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    // Relative to the star's own center (x0,y0,z0) -- see SetupTOV's own comment.
    Real r = sqrt(SQR(x1v-x0) + SQR(x2v-y0) + SQR(x3v-z0));
    Real s = sqrt(SQR(x1v-x0) + SQR(x2v-y0));

    Real mass, alp, r_schw;
    if (isotropic) {
      tov_.GetMandAlphaIso(r, mass, alp);
      r_schw = tov_.FindSchwarzschildR(r, mass);
    } else {
      tov_.GetMandAlpha(r, mass, alp);
    }

    // Set ADM variables
    adm.alpha(m,k,j,i) = alp;
    if (minkowski) {
      adm.g_dd(m,0,0,k,j,i) = adm.g_dd(m,1,1,k,j,i) = adm.g_dd(m,2,2,k,j,i) = 1.0;
      adm.g_dd(m,0,1,k,j,i) = adm.g_dd(m,0,2,k,j,i) = adm.g_dd(m,1,2,k,j,i) = 0.0;
      adm.alpha(m,k,j,i) = 1.0;
    } else if (!isotropic) {
      // Auxiliary metric quantities
      Real fmet = 0.0;
      if (r > 0) {
        fmet = (1./(1. - 2*mass/r) - 1.)/(r*r);
      }

      adm.g_dd(m,0,0,k,j,i) = x1v*x1v*fmet + 1.0;
      adm.g_dd(m,0,1,k,j,i) = x1v*x2v*fmet;
      adm.g_dd(m,0,2,k,j,i) = x1v*x3v*fmet;
      adm.g_dd(m,1,1,k,j,i) = x2v*x2v*fmet + 1.0;
      adm.g_dd(m,1,2,k,j,i) = x2v*x3v*fmet;
      adm.g_dd(m,2,2,k,j,i) = x3v*x3v*fmet + 1.0;
      Real det = adm::SpatialDet(
              adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
              adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
              adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i));
      adm.psi4(m,k,j,i) = pow(det, 1./3.);
    } else {
      Real fmet = 1.;
      if (r > 0) {
        fmet = r_schw/r;
      }
      Real psi4 = fmet*fmet;

      adm.g_dd(m,0,0,k,j,i) = adm.g_dd(m,1,1,k,j,i) = adm.g_dd(m,2,2,k,j,i) = psi4;
      adm.g_dd(m,0,1,k,j,i) = adm.g_dd(m,0,2,k,j,i) = adm.g_dd(m,1,2,k,j,i) = 0.0;
      adm.psi4(m,k,j,i) = psi4;
    }
    adm.beta_u(m,0,k,j,i) = adm.beta_u(m,1,k,j,i) = adm.beta_u(m,2,k,j,i) = 0.0;
    adm.vK_dd(m,0,0,k,j,i) = adm.vK_dd(m,0,1,k,j,i) = adm.vK_dd(m,0,2,k,j,i) = 0.0;
    adm.vK_dd(m,1,1,k,j,i) = adm.vK_dd(m,1,2,k,j,i) = adm.vK_dd(m,2,2,k,j,i) = 0.0;
  });
}

//----------------------------------------------------------------------------------------
//! \fn void TDERefineTracker(MeshBlockPack *pmbp)
//! \brief Refinement criterion for the TDE setup: keep a fine box around the BH
//! puncture (fixed at the origin) and a second one around the infalling star,
//! instead of the single long static "tube" spanning the whole trajectory that
//! wastes ~90% of its finest-level cost on space that is empty at any instant.
//!
//! Follows dynbbh.cpp's RefineTracker (an analytic-trajectory user_ref_func that
//! needs no z4c object) rather than z4c's CompactObjectTracker, which is
//! unreachable here: <cfc> and <z4c> are mutually exclusive (meshblock_pack.cpp),
//! so pz4c is always nullptr in a CFC run and ptracker never exists.
//!
//! The star position comes from the analytic Schwarzschild radial free-fall from
//! rest, r(t) = (r0/2)(1+cos eta) with t = sqrt(r0^3/2M)(eta+sin eta)/2, inverted
//! by bisection. Checked against the production run: accurate to <0.19 code units
//! through t~60, but it drifts badly later (-7 at t=180) because the collapsing
//! lapse slows coordinate infall relative to the Schwarzschild areal result. So
//! this is adequate for a short regrid-behaviour test, NOT for production -- a
//! real run needs a genuine tracker (e.g. a density maximum).
//!
//! Note this writes refine_flag through h_view, not d_view. dynbbh.cpp's version
//! writes d_view from a host loop, which happens to work only because the two
//! views alias on a CPU-only build; z4c_amr.cpp's RefineTracker uses h_view, and
//! that is the form that is also correct for a GPU build.

void TDERefineTracker(MeshBlockPack *pmbp) {
  Mesh *pmesh       = pmbp->pmesh;
  auto &refine_flag = pmesh->pmr->refine_flag;
  auto &size        = pmbp->pmb->mb_size;
  int nmb           = pmbp->nmb_thispack;
  int mbs           = pmesh->gids_eachrank[global_variable::my_rank];

  // Locate the star as the global maximum of rest-mass density. (The earlier
  // analytic Schwarzschild free-fall estimate was accurate to <0.19 through
  // t~60 but drifts to ~-2.4 by t~145 -- larger than a sensible box half-width,
  // so the box would slide off the star. Widening the box instead is not an
  // option: radius 4 would cost ~4096 finest MeshBlocks, as much as the tube
  // this design exists to replace.)
  Real x1_star = tde_ref.r0, x2_star = 0.0, x3_star = 0.0;
  {
    auto &w0_    = pmbp->pmhd->w0;
    auto &indcs  = pmesh->mb_indcs;
    int is = indcs.is, js = indcs.js, ks = indcs.ks;
    int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
    const int nkji = nx3*nx2*nx1, nji = nx2*nx1;
    const int nmkji = nmb*nkji;

    using MaxLocR = Kokkos::MaxLoc<Real, int>;
    typename MaxLocR::value_type mxl;
    Kokkos::parallel_reduce("TDEStarLoc",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int &idx, typename MaxLocR::value_type &lmx) {
        int m = idx/nkji;
        int k = (idx - m*nkji)/nji;
        int j = (idx - m*nkji - k*nji)/nx1;
        int i = (idx - m*nkji - k*nji - j*nx1) + is;
        k += ks; j += js;
        Real d = w0_(m,IDN,k,j,i);
        if (d > lmx.val) { lmx.val = d; lmx.loc = idx; }
      }, MaxLocR(mxl));

    // Convert this rank's winning flat index back to coordinates.
    int m = mxl.loc/nkji;
    int k = (mxl.loc - m*nkji)/nji;
    int j = (mxl.loc - m*nkji - k*nji)/nx1;
    int i = (mxl.loc - m*nkji - k*nji - j*nx1);
    Real xl = CellCenterX(i, nx1, size.h_view(m).x1min, size.h_view(m).x1max);
    Real yl = CellCenterX(j, nx2, size.h_view(m).x2min, size.h_view(m).x2max);
    Real zl = CellCenterX(k, nx3, size.h_view(m).x3min, size.h_view(m).x3max);

#if MPI_PARALLEL_ENABLED
    // Pick the globally-densest cell, then broadcast its position from the
    // owning rank. MPI_MAXLOC on {value, rank} identifies the owner.
    struct { double val; int rank; } lin, gout;
    lin.val = static_cast<double>(mxl.val);
    lin.rank = global_variable::my_rank;
    MPI_Allreduce(&lin, &gout, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
    Real pos[3] = {xl, yl, zl};
    MPI_Bcast(pos, 3, MPI_ATHENA_REAL, gout.rank, MPI_COMM_WORLD);
    xl = pos[0]; yl = pos[1]; zl = pos[2];
#endif
    x1_star = xl; x2_star = yl; x3_star = zl;
  }

  const Real r2_bh      = SQR(tde_ref.rad_bh);
  const Real r2_star    = SQR(tde_ref.rad_star);
  const Real r2_bh_fine = SQR(tde_ref.rad_bh_fine);

  for (int m = 0; m < nmb; ++m) {
    Real &x1min = size.h_view(m).x1min;
    Real &x1max = size.h_view(m).x1max;
    Real &x2min = size.h_view(m).x2min;
    Real &x2max = size.h_view(m).x2max;
    Real &x3min = size.h_view(m).x3min;
    Real &x3max = size.h_view(m).x3max;

    // Distance from each target to the closest point of this MeshBlock's AABB
    // (0 when the target is inside it), matching z4c_amr.cpp's RefineTracker.
    auto dist2_to_block = [&](Real px, Real py, Real pz) -> Real {
      Real cx = std::fmax(x1min, std::fmin(px, x1max));
      Real cy = std::fmax(x2min, std::fmin(py, x2max));
      Real cz = std::fmax(x3min, std::fmin(pz, x3max));
      return SQR(px - cx) + SQR(py - cy) + SQR(pz - cz);
    };

    // Per-target requested levels, compared against this block's CURRENT level
    // (the z4c_amr.cpp:84 pattern). A single "refine or derefine" flag is not
    // enough once different targets want different depths: the puncture needs a
    // deeper level than the star, to widen the gap between the excision surface
    // and the horizon, while refining the STAR that deep would halve dt (the
    // star sits where alpha~1, so it is what sets the global timestep).
    int level = pmesh->lloc_eachmb[m + mbs].level - pmesh->root_level;
    Real d2_bh = dist2_to_block(0.0, 0.0, 0.0);
    Real d2_st = dist2_to_block(x1_star, x2_star, x3_star);

    int want = -1;
    if (d2_bh < r2_bh)                          { want = std::max(want, tde_ref.lev_bh); }
    if (r2_bh_fine > 0.0 && d2_bh < r2_bh_fine) { want = std::max(want, tde_ref.lev_bh_fine); }
    if (d2_st < r2_star)                        { want = std::max(want, tde_ref.lev_star); }

    int flag;
    if (want < 0)             { flag = -1; }        // outside every target
    else if (level < want)    { flag =  1; }
    else if (level == want)   { flag =  0; }
    else                      { flag = -1; }
    refine_flag.h_view(m + mbs) = flag;
  }

  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
}

// Cleanup at the end of the run
void FinalizeTOV(ParameterInput *pin, Mesh *pm) {
  // This function is only needed to delete the TOV solver data, which is stored inside
  // the dynamically allocated TOVParams object.
  if (ptov_params != nullptr) {
    delete ptov_params;
  }
}

// History function
void TOVHistory(HistoryData *pdata, Mesh *pm) {
  // Select the number of outputs and create labels for them.
  pdata->nhist = 2;
  pdata->label[0] = "rho-max";
  pdata->label[1] = "alpha-min";

  // capture class variables for kernel
  auto &w0_ = pm->pmb_pack->pmhd->w0;
  auto &adm = pm->pmb_pack->padm->adm;

  // loop over all MeshBlocks in this pack
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is; int nx1 = indcs.nx1;
  int js = indcs.js; int nx2 = indcs.nx2;
  int ks = indcs.ks; int nx3 = indcs.nx3;
  const int nmkji = (pm->pmb_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  // Kokkos::Max/Min overwrite these with their own reduction identities regardless,
  // but they were previously swapped (each seeded with the OTHER's identity) --
  // written correctly here.
  Real rho_max = std::numeric_limits<Real>::lowest();
  Real alpha_min = std::numeric_limits<Real>::max();
  Kokkos::parallel_reduce("TOVHistSums",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &mb_max, Real &mb_alp_min) {
    // compute n,k,j,i indices of thread
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    mb_max = fmax(mb_max, w0_(m,IDN,k,j,i));
    mb_alp_min = fmin(mb_alp_min, adm.alpha(m, k, j, i));
  }, Kokkos::Max<Real>(rho_max), Kokkos::Min<Real>(alpha_min));

  // Currently AthenaK only supports MPI_SUM operations between ranks, but we need MPI_MAX
  // and MPI_MIN operations instead. This is a cheap hack to make it work as intended.
#if MPI_PARALLEL_ENABLED
  if (global_variable::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &rho_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &alpha_min, 1, MPI_ATHENA_REAL, MPI_MIN, 0, MPI_COMM_WORLD);
  } else {
    // Non-root: sendbuf/recvbuf must not alias (only MPI_IN_PLACE, root-side,
    // may reuse the same buffer) -- send into a separate throwaway.
    Real rho_max_out, alpha_min_out;
    MPI_Reduce(&rho_max, &rho_max_out, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&alpha_min, &alpha_min_out, 1, MPI_ATHENA_REAL, MPI_MIN, 0,
               MPI_COMM_WORLD);
    rho_max = 0.;
    alpha_min = 0.;
  }
#endif

  // store data in hdata array
  pdata->hdata[0] = rho_max;
  pdata->hdata[1] = alpha_min;
}

