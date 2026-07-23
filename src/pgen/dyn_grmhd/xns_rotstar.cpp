//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file xns_rotstar.cpp
//  \brief Problem generator for a rotating neutron star, using initial data produced
//  by the external XNS code (Bucciantini & Del Zanna 2011; Pili et al.). Only works
//  with the CFC (Conformally Flat Condition) metric solver: the pgen only sets an
//  initial guess for the ADM metric and hydro primitives, from XNS's tabulated
//  axisymmetric equilibrium; CFC::InitializeMetric() (invoked automatically by
//  Driver::Initialize()) then Picard-iterates this to a self-consistent solution.
//
//  Compile with '-D PROBLEM=dyn_grmhd/xns_rotstar' to enroll as user-specific
//  problem generator.

#include <float.h>
#include <math.h>     // atan2(), fmax(), sqrt()

#include <iostream>
#include <limits>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "cfc/cfc.hpp"
#include "utils/xns/xns_rotator.hpp"

namespace {
struct XNSStarParams {
  xns::XNSRotator xns_star;
  Real dfloor;
  Real pfloor;

  XNSStarParams(xns::XNSRotator&& xns_star_, Real dfloor_, Real pfloor_) :
      xns_star(std::move(xns_star_)), dfloor(dfloor_), pfloor(pfloor_) {}
};

XNSStarParams *pxns_params;
} // namespace

void SetADMVariablesToXNS(MeshBlockPack *pmbp);
void XNSRotStarHistory(HistoryData *pdata, Mesh *pm);

// Shared point evaluator: given a Cartesian position, returns hydro primitives and
// ADM metric variables (as an initial guess) for the XNS-tabulated rotating star, or
// flat-space/atmosphere values outside the table's own domain (r > xns.rmax()). Used
// identically by both the initial setup below and SetADMVariablesToXNS (the AMR
// regrid/restart callback) -- factored into one function (unlike dyngr_tov.cpp's
// duplicated closed-form TOV formulas) because a 2D table lookup plus the phi-vector
// Cartesian transform is real duplication risk, not a trivial one-liner to repeat.
KOKKOS_INLINE_FUNCTION
static void XNSInterpToADMAndPrim(const xns::XNSRotator &xns_star, Real dfloor,
                                   Real pfloor, Real x1, Real x2, Real x3,
                                   Real &rho, Real &p, Real &vx, Real &vy, Real &vz,
                                   Real &alpha, Real &bx, Real &by, Real &bz,
                                   Real &psi4) {
  Real r = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  Real s = sqrt(SQR(x1) + SQR(x2));
  Real theta = atan2(s, x3);

  if (r > xns_star.rmax()) {
    rho = dfloor;
    p = pfloor;
    vx = vy = vz = 0.0;
    alpha = 1.0;
    bx = by = bz = 0.0;
    psi4 = 1.0;
    return;
  }

  Real psi, vphi, betaphi;
  xns_star.Interpolate<xns::Loc::Device>(r, theta, rho, p, psi, alpha,
                                          betaphi, vphi);
  psi4 = SQR(psi)*SQR(psi);

  // The star is oblate (rotational flattening), so whether this point is inside
  // the star or in XNS's own vacuum/atmosphere solution -- which has nonzero,
  // non-monotonic density that can exceed dfloor, so a bare density threshold is
  // not a reliable atmosphere test -- is decided by the actual surface
  // R_surf(theta) (Surf.dat), not a spherical radius comparison.
  Real r_surf = xns_star.SurfaceRadius<xns::Loc::Device>(theta);
  bool atmosphere = (r > r_surf);

  if (atmosphere) {
    rho = dfloor;
    p = pfloor;
    vx = vy = vz = 0.0;
    bx = by = bz = 0.0;
  } else {
    rho = fmax(rho, dfloor);
    p = fmax(p, pfloor);
    // beta^phi and v^phi are contravariant coordinate-basis phi-components of
    // vector fields in flat/conformally-flat 3-space (CFC's spatial metric is
    // conformally flat), so the Cartesian transform is the exact flat-space
    // identity for a pure-azimuthal coordinate vector: V^i d_i = V^phi d_phi =
    // V^phi*(-y d_x + x d_y).
    vx = -vphi*x2;
    vy = vphi*x1;
    vz = 0.0;
    bx = -betaphi*x2;
    by = betaphi*x1;
    bz = 0.0;

    // vx,vy,vz above are the Eulerian-observer 3-velocity v^i (coordinate basis).
    // AthenaK's dyn_grmhd primitives (w0(IVX/IVY/IVZ)) store W*v^i, not v^i itself
    // (confirmed via primitive_solver.hpp's "Athena passes in Wv, not v" and every
    // other ID-import pgen in this tree, e.g. lorene_bns.cpp/kadath_bns.cpp/
    // sgrid_bns.cpp/elliptica.cpp, which all multiply by the Lorentz factor before
    // writing to IVX/IVY/IVZ) -- this pgen was missing that factor entirely.
    // v^2 = gamma_ij v^i v^j with gamma_ij = psi4*delta_ij (conformally flat,
    // Cartesian): v^2 = psi4*(vx^2+vy^2+vz^2). Clamp defensively for numerical
    // safety near the surface/mass-shedding limit (XNS's own equilibrium is
    // subluminal by construction, so this should only ever bite on
    // interpolation overshoot right at the stellar edge).
    Real vsq = psi4*(SQR(vx) + SQR(vy) + SQR(vz));
    vsq = fmin(vsq, 0.9999);
    Real lorentz_w = 1.0/sqrt(1.0 - vsq);
    vx *= lorentz_w;
    vy *= lorentz_w;
    vz *= lorentz_w;
  }
}

void SetupXNSRotStar(ParameterInput *pin, Mesh *pmy_mesh_) {
  Real dfloor = pin->GetOrAddReal("mhd", "dfloor", (FLT_MIN));
  Real pfloor = pin->GetOrAddReal("mhd", "pfloor", (FLT_MIN));

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  pxns_params = new XNSStarParams(xns::XNSRotator(pin), dfloor, pfloor);

  auto &w0_ = pmbp->pmhd->w0;
  auto &adm = pmbp->padm->adm;

  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  int nmb1 = pmbp->nmb_thispack - 1;

  auto &size = pmbp->pmb->mb_size;
  auto &xns_star = pxns_params->xns_star;
  auto &dfloor_ = pxns_params->dfloor;
  auto &pfloor_ = pxns_params->pfloor;

  par_for("pgen_xns_rotstar", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
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

    Real rho, p, vx, vy, vz, alpha, bx, by, bz, psi4;
    XNSInterpToADMAndPrim(xns_star, dfloor_, pfloor_, x1v, x2v, x3v,
                          rho, p, vx, vy, vz, alpha, bx, by, bz, psi4);

    w0_(m,IDN,k,j,i) = rho;
    w0_(m,IPR,k,j,i) = p;
    w0_(m,IVX,k,j,i) = vx;
    w0_(m,IVY,k,j,i) = vy;
    w0_(m,IVZ,k,j,i) = vz;

    // Conformally flat: g_ij = psi^4 * delta_ij exactly in Cartesian coordinates,
    // regardless of the star's shape/rotation, since the ID is constructed in the
    // CFC/XCFC gauge (same isotropic-branch assembly dyngr_tov.cpp uses for TOV).
    adm.alpha(m,k,j,i) = alpha;
    adm.g_dd(m,0,0,k,j,i) = adm.g_dd(m,1,1,k,j,i) = adm.g_dd(m,2,2,k,j,i) = psi4;
    adm.g_dd(m,0,1,k,j,i) = adm.g_dd(m,0,2,k,j,i) = adm.g_dd(m,1,2,k,j,i) = 0.0;
    adm.psi4(m,k,j,i) = psi4;
    adm.beta_u(m,0,k,j,i) = bx;
    adm.beta_u(m,1,k,j,i) = by;
    adm.beta_u(m,2,k,j,i) = bz;
    // Initial guess only -- always overwritten by CFC::InitializeMetric()'s own
    // solve (AssembleLapseShiftK), exactly as dyngr_tov.cpp sets this to 0 for TOV.
    adm.vK_dd(m,0,0,k,j,i) = adm.vK_dd(m,0,1,k,j,i) = adm.vK_dd(m,0,2,k,j,i) = 0.0;
    adm.vK_dd(m,1,1,k,j,i) = adm.vK_dd(m,1,2,k,j,i) = adm.vK_dd(m,2,2,k,j,i) = 0.0;
  });
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//  \brief Sets initial conditions for a rotating NS (XNS initial data) in CFC

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pcfc == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "XNS rotating-star problem requires the <cfc> block "
              << "(CFC metric solver)" << std::endl;
    exit(EXIT_FAILURE);
  }

  pmbp->padm->SetADMVariables = &SetADMVariablesToXNS;
  user_hist_func = &XNSRotStarHistory;

  // Reload the table for restarts too (cheap: a few MB of text), matching the
  // TOV pgen's own restart-reconstructs-the-solver precedent.
  if (restart) {
    Real dfloor = pin->GetOrAddReal("mhd", "dfloor", (FLT_MIN));
    Real pfloor = pin->GetOrAddReal("mhd", "pfloor", (FLT_MIN));
    pxns_params = new XNSStarParams(xns::XNSRotator(pin), dfloor, pfloor);
    return;
  }

  SetupXNSRotStar(pin, pmy_mesh_);

  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;

  pmbp->pdyngr->PrimToConInit(0, (n1-1), 0, (n2-1), 0, (n3-1));
}

// AMR regrid / restart callback: MeshRefinement::AdaptiveMeshRefinement() calls
// padm->SetADMVariables(...) unconditionally on every regrid event for any CFC-only
// (non-Z4c) run, regardless of padm->is_dynamic -- without this override every
// regrid would silently reseed all blocks with the default Kerr-Schild metric
// instead of this star.
void SetADMVariablesToXNS(MeshBlockPack *pmbp) {
  auto &adm = pmbp->padm->adm;
  auto &size = pmbp->pmb->mb_size;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int nmb = pmbp->nmb_thispack;

  auto &xns_star = pxns_params->xns_star;
  auto &dfloor_ = pxns_params->dfloor;
  auto &pfloor_ = pxns_params->pfloor;

  par_for("update_adm_vars_xns", DevExeSpace(), 0,nmb-1,0,(n3-1),0,(n2-1),0,(n1-1),
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

    Real rho, p, vx, vy, vz, alpha, bx, by, bz, psi4;
    XNSInterpToADMAndPrim(xns_star, dfloor_, pfloor_, x1v, x2v, x3v,
                          rho, p, vx, vy, vz, alpha, bx, by, bz, psi4);

    adm.alpha(m,k,j,i) = alpha;
    adm.g_dd(m,0,0,k,j,i) = adm.g_dd(m,1,1,k,j,i) = adm.g_dd(m,2,2,k,j,i) = psi4;
    adm.g_dd(m,0,1,k,j,i) = adm.g_dd(m,0,2,k,j,i) = adm.g_dd(m,1,2,k,j,i) = 0.0;
    adm.psi4(m,k,j,i) = psi4;
    adm.beta_u(m,0,k,j,i) = bx;
    adm.beta_u(m,1,k,j,i) = by;
    adm.beta_u(m,2,k,j,i) = bz;
    adm.vK_dd(m,0,0,k,j,i) = adm.vK_dd(m,0,1,k,j,i) = adm.vK_dd(m,0,2,k,j,i) = 0.0;
    adm.vK_dd(m,1,1,k,j,i) = adm.vK_dd(m,1,2,k,j,i) = adm.vK_dd(m,2,2,k,j,i) = 0.0;
  });
}

//----------------------------------------------------------------------------------------
//! \fn void XNSRotStarHistory()
//  \brief History function: tracks rho-max/alpha-min (mirrors TOVHistory in
//  dyngr_tov.cpp) plus the total angular momentum about the z-axis.

void XNSRotStarHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 3;
  pdata->label[0] = "rho-max";
  pdata->label[1] = "alpha-min";
  pdata->label[2] = "ang-mom";

  auto &w0_ = pm->pmb_pack->pmhd->w0;
  auto &u0_ = pm->pmb_pack->pmhd->u0;
  auto &adm = pm->pmb_pack->padm->adm;
  auto &size = pm->pmb_pack->pmb->mb_size;

  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is; int nx1 = indcs.nx1;
  int js = indcs.js; int nx2 = indcs.nx2;
  int ks = indcs.ks; int nx3 = indcs.nx3;
  const int nmkji = (pm->pmb_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;

  Real rho_max = std::numeric_limits<Real>::max();
  Real alpha_min = -rho_max;
  Real ang_mom = 0.0;
  Kokkos::parallel_reduce("XNSRotStarHistSums",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &mb_max, Real &mb_alp_min, Real &mb_jz) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    mb_max = fmax(mb_max, w0_(m,IDN,k,j,i));
    mb_alp_min = fmin(mb_alp_min, adm.alpha(m,k,j,i));

    // J_z = integral sqrt(gamma)*(x*S_y - y*S_x) d^3x, S_i the undensitized ADM
    // momentum density. u0_(IM1+a) is already sqrt(gamma)*S_a (see
    // dyn_grmhd.cpp's SetTmunu: tmunu.S_d(a) = cons(IM1+a)*ivol, ivol =
    // 1/sqrt(gamma)), so sqrt(gamma)*S_a = u0_(IM1+a) exactly -- no extra
    // metric factor needed, mirroring history.cpp's own LoadMHDHistoryData
    // mass-sum idiom (vol=dx1*dx2*dx3, hvars[IDN]=vol*u0_(IDN), no sqrt(detg)).
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
    Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    mb_jz += vol*(x1v*u0_(m,IM2,k,j,i) - x2v*u0_(m,IM1,k,j,i));
  }, Kokkos::Max<Real>(rho_max), Kokkos::Min<Real>(alpha_min),
     Kokkos::Sum<Real>(ang_mom));

  // Currently AthenaK only supports MPI_SUM operations between ranks, but we need
  // MPI_MAX and MPI_MIN for rho_max/alpha_min -- same cheap hack TOVHistory uses
  // (dyngr_tov.cpp): manually MAX/MIN-reduce those two onto rank 0, then zero every
  // other rank's copy so the framework's own generic post-reduction MPI_SUM (over
  // the whole hdata array, history.cpp) is a no-op for them.
#if MPI_PARALLEL_ENABLED
  if (global_variable::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &rho_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &alpha_min, 1, MPI_ATHENA_REAL, MPI_MIN, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&rho_max, &rho_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&alpha_min, &alpha_min, 1, MPI_ATHENA_REAL, MPI_MIN, 0, MPI_COMM_WORLD);
    rho_max = 0.;
    alpha_min = 0.;
  }
#endif
  // ang_mom is a genuine volume-integral sum -- left as this rank's own local
  // partial sum; the framework's generic MPI_SUM reduction over hdata combines
  // it correctly across ranks with no manual reduction needed here.

  pdata->hdata[0] = rho_max;
  pdata->hdata[1] = alpha_min;
  pdata->hdata[2] = ang_mom;
}
