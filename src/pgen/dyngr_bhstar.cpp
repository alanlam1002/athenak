//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyngr_bhstar.cpp
//  \brief Problem generator for a BH-star quasi-star Bondi accretion problem
//  with grey M1 LTE radiation. Only works when ADM is enabled.

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
#include "radiation_m1/radiation_m1.hpp"
#include "radiation_m1/radiation_m1_macro.hpp"
#include "radiation_m1/radiation_m1_helpers.hpp"
#include "radiation_m1/radiation_m1_tensors.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"

// Prototypes for vector potential
KOKKOS_INLINE_FUNCTION
static Real A1(Real eosk, Real gamma, Real bondi_rs, Real pcut, Real rhoc,
               Real magindex, Real x1, Real x2, Real x3);
KOKKOS_INLINE_FUNCTION
static Real A2(Real eosk, Real gamma, Real bondi_rs, Real pcut, Real rhoc,
               Real magindex, Real x1, Real x2, Real x3);

// Prototypes for user-defined BCs and history
void BHStarHistory(HistoryData *pdata, Mesh *pm);

void SetADMVariables(MeshBlockPack *pmbp);
void FinalizeBHStar(ParameterInput *pin, Mesh *pm);

void SetupProblem(ParameterInput *pin, Mesh* pmy_mesh_, bool enable_radiation) {
  Real eosk = pin->GetOrAddReal("problem", "eosk", 1.1126500560536184e-9);
  Real gamma = pin->GetOrAddReal("problem", "gamma", 5.0/3.0);
  Real bondi_rs = pin->GetOrAddReal("problem", "bondi_rs", 4.49e8);

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  // Density at the inner cutoff r=0.5 (where rsch=2), matching the density
  // field below -- used to normalize the vector-potential/magnetization envelope.
  Real rhoc = 0.0625 / pow( 2.0 * bondi_rs, 1.5 );

  auto& w0_ = pmbp->pmhd->w0;

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
  par_for("pgen_bhstar_hydro", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
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
    // radial coordinate.
    Real r = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));

    // Set ADM variables
    Real psi2 = pow(1.0 + 0.5 / r, 2);
    Real psi4 = psi2 * psi2;
    adm.alpha(m,k,j,i) = 1.0 / psi2;
    adm.g_dd(m,0,0,k,j,i) = adm.g_dd(m,1,1,k,j,i) = adm.g_dd(m,2,2,k,j,i) = psi4;
    adm.g_dd(m,0,1,k,j,i) = adm.g_dd(m,0,2,k,j,i) = adm.g_dd(m,1,2,k,j,i) = 0.0;
    adm.psi4(m,k,j,i) = psi4;
    adm.beta_u(m,0,k,j,i) = adm.beta_u(m,1,k,j,i) = adm.beta_u(m,2,k,j,i) = 0.0;
    adm.vK_dd(m,0,0,k,j,i) = adm.vK_dd(m,0,1,k,j,i) = adm.vK_dd(m,0,2,k,j,i) = 0.0;
    adm.vK_dd(m,1,1,k,j,i) = adm.vK_dd(m,1,2,k,j,i) = adm.vK_dd(m,2,2,k,j,i) = 0.0;

    Real rho, p;
    Real vr = 0.;
    if ( r > 0.5 ) {
      Real rsch = r * pow( 1.0 + 0.5 / r, 2 );
      vr = - 0.5 * sqrt( 2.0 / rsch ) * ( 1.0 + 0.5 / sqrt( fmax( 1.e-8, rsch - 1.0 ) ) );
      rho = 0.0625 / pow( rsch * bondi_rs, 1.5 );
      p = eosk * pow(rho, gamma);
    } else {
      rho = 0;
      p = 0;
    }

    Real vu[3] = { vr * x1v / r,
                   vr * x2v / r,
                   vr * x3v / r };

    // Set hydrodynamic quantities
    w0_(m,IDN,k,j,i) = rho;
    w0_(m,IPR,k,j,i) = p;
    w0_(m,IVX,k,j,i) = vu[0];
    w0_(m,IVY,k,j,i) = vu[1];
    w0_(m,IVZ,k,j,i) = vu[2];
  });

  if (enable_radiation) {
    auto &uradm1_ = pmbp->pradm1->u0;
    auto &nspecies_ = pmbp->pradm1->nspecies;
    auto &m1_nvars_ = pmbp->pradm1->nvars;
    auto &m1_params_ = pmbp->pradm1->params;

    // M1 radiation initial condition parameters (all in code units). arad
    // comes from the same photon_op_params the opacity module actually uses
    // (<photons>/arad or units-derived), not an independent <problem> key
    // (UserProblem already enforces opacity_type == photons above), so the
    // IC and the evolved opacities can never silently diverge.
    Real arad    = pmbp->pradm1->photon_op_params.arad;
    Real T_ph    = pin->GetOrAddReal("problem", "T_photosphere", 0.0);

    // Read kappa_s from <photons> block (same source as the M1 opacity module)
    // so the optical depth profile is consistent with the opacity that will run.
    Real kappa_s = pin->GetOrAddReal("photons", "kappa_s", 0.0);

    // Default luminosity = Eddington luminosity: L_Edd = 4pi G M / kappa_es
    // In code units (G=c=M_BH=1): L_Edd = 4pi / kappa_s
    Real lum_edd = (kappa_s > 0.0) ? 4.0 * M_PI / kappa_s : 0.0;
    Real lum     = pin->GetOrAddReal("problem", "luminosity", lum_edd);

    par_for("pgen_bhstar_rad", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
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

      Real r = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
      Real psi2 = pow(1.0 + 0.5 / r, 2);
      Real psi4 = psi2 * psi2;

      Real vr = 0.;
      if (r > 0.5) {
        Real rsch = r * pow(1.0 + 0.5 / r, 2);
        vr = - 0.5 * sqrt( 2.0 / rsch ) * ( 1.0 + 0.5 / sqrt( fmax( 1.e-8, rsch - 1.0 ) ) );
      }
      Real vsq = psi4 * vr * vr + 1.0;
      Real lfac = sqrt(vsq);

      // Inverse 4-metric (diagonal: beta=0, spatial metric = psi4 * delta_ij)
      AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_uu{};
      Real alp_m1 = 1.0 / psi2;
      g_uu(0,0) = -1.0 / (alp_m1 * alp_m1);
      g_uu(1,1) = g_uu(2,2) = g_uu(3,3) = 1.0 / psi4;

      // LTE energy density from photospheric temperature
      Real E_lte = arad * pow(T_ph, 4.0);

      // rsch at the field point (clamped to the inner-cutoff value inside r<=0.5,
      // matching the hydro kernel's excised-region convention).
      Real rsch_loc = (r > 0.5) ? r * pow(1.0 + 0.5/r, 2.0) : 2.0;

      // Optical depth via free-fall Bondi approximation
      Real tau = kappa_s * 0.125 / M_PI / bondi_rs / sqrt( bondi_rs * rsch_loc );

      // Thick-to-thin flux interpolation factor
      Real f_tau = (1.0 - exp(-tau)) / (1.0 + tau);

      // Comoving radial flux + lab-frame advection correction (lfac: Lorentz boost)
      Real Fr_hat = f_tau * lum / (4.0 * M_PI * rsch_loc * rsch_loc);
      Real Fr_lab = Fr_hat + vr * E_lte / lfac;

      Real E_rad = E_lte;
      // pack_F_d: F_d(0)=beta^i F_i=0, F_d(1..3)=spatial components
      // Guard against r=0 by using safe direction cosines
      AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
      Real inv_r = (r > 0.0) ? 1.0 / r : 0.0;
      pack_F_d(0.0, 0.0, 0.0,
               Fr_lab * x1v * inv_r,
               Fr_lab * x2v * inv_r,
               Fr_lab * x3v * inv_r, F_d);

      radiationm1::apply_floor(g_uu, E_rad, F_d, m1_params_);

      for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
        uradm1_(m, radiationm1::CombinedIdx(nuidx, M1_E_IDX,  m1_nvars_), k, j, i) = E_rad;
        uradm1_(m, radiationm1::CombinedIdx(nuidx, M1_FX_IDX, m1_nvars_), k, j, i) = F_d(1);
        uradm1_(m, radiationm1::CombinedIdx(nuidx, M1_FY_IDX, m1_nvars_), k, j, i) = F_d(2);
        uradm1_(m, radiationm1::CombinedIdx(nuidx, M1_FZ_IDX, m1_nvars_), k, j, i) = F_d(3);
        if (nspecies_ > 1) {
          uradm1_(m, radiationm1::CombinedIdx(nuidx, M1_N_IDX, m1_nvars_), k, j, i) =
              m1_params_.rad_N_floor;
        }
      }
    });
  }

  // parse some parameters
  Real b_norm = pin->GetOrAddReal("problem", "b_norm", 1.e-10);
  Real pcut = pin->GetOrAddReal("problem", "pcut", 1e-20);
  Real magindex = pin->GetOrAddReal("problem", "magindex", 2);

  // If use_pcut_rel = true, we take pcut to be a percentage of pmax rather than
  // an absolute cutoff
  if (pin->GetOrAddBoolean("problem", "use_pcut_rel", true)) {
    Real pmax = eosk * pow(rhoc, gamma);
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

    a1(m,k,j,i) = A1(eosk, gamma, bondi_rs, pcut, rhoc, magindex, x1v, x2f, x3f);
    a2(m,k,j,i) = A2(eosk, gamma, bondi_rs, pcut, rhoc, magindex, x1f, x2v, x3f);
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
      a1(m,k,j,i) = 0.5*(A1(eosk, gamma, bondi_rs, pcut, rhoc, magindex, xl,x2f,x3f) +
                         A1(eosk, gamma, bondi_rs, pcut, rhoc, magindex, xr,x2f,x3f));
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
      a2(m,k,j,i) = 0.5*(A2(eosk, gamma, bondi_rs, pcut, rhoc, magindex, x1f,xl,x3f) +
                         A2(eosk, gamma, bondi_rs, pcut, rhoc, magindex, x1f,xr,x3f));
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
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//  \brief Sets initial conditions for a BH-star Bondi accretion problem in DynGRMHD,
//  with optional grey M1 LTE radiation.
//  Compile with '-D PROBLEM=dyngr_bhstar' to enroll as user-specific problem generator

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!pmbp->pcoord->is_dynamical_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "BH star problem can only be run when <adm> block is present"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->pmhd == nullptr || pmbp->pdyngr == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "BH star problem can only be run with <mhd> and dynamical GRMHD "
                 "(<mhd>/dyn_eos) enabled" << std::endl;
    exit(EXIT_FAILURE);
  }

  // enable_radiation=true (default) requires <radiation_m1>; enable_radiation=false
  // runs the Bondi hydro/metric IC alone, skipping the M1 radiation IC entirely.
  bool enable_radiation = pin->GetOrAddBoolean("problem", "enable_radiation", true);
  if (enable_radiation && pmbp->pradm1 == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "BH star problem has enable_radiation=true (the default) but no "
              << "<radiation_m1> block in input file -- either add one or set "
                 "problem/enable_radiation=false for a hydro-only run" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (enable_radiation &&
      pmbp->pradm1->params.opacity_type != radiationm1::Photons) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "BH star problem's radiation initial data assumes "
                 "radiation_m1/opacity_type = photons (it builds the IC from "
                 "<photons>/kappa_s); got a different opacity_type instead"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  user_hist_func = &BHStarHistory;
  pgen_final_func = &FinalizeBHStar;
  pmbp->padm->SetADMVariables = &SetADMVariables;

  // initialize primitive variables for restart
  if (restart) {
    return;
  }

  SetupProblem(pin, pmy_mesh_, enable_radiation);

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

KOKKOS_INLINE_FUNCTION
static Real A1(Real eosk, Real gamma, Real bondi_rs, Real pcut, Real rhoc,
               Real magindex, Real x1, Real x2, Real x3) {
  Real r = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  Real rsch = r * pow( 1.0 + 0.5 / r, 2 );
  Real rho = 0.0;
  Real p = 0.0;
  if ( r > 0.5 ) {
    rho = 0.0625 / pow( rsch * bondi_rs, 1.5 );
    p = eosk * pow(rho, gamma);
  }
  return -x2*fmax(p - pcut, 0.0)*pow(1.0 - rho/rhoc,magindex);
}

KOKKOS_INLINE_FUNCTION
static Real A2(Real eosk, Real gamma, Real bondi_rs, Real pcut, Real rhoc,
               Real magindex, Real x1, Real x2, Real x3) {
  Real r = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  Real rsch = r * pow( 1.0 + 0.5 / r, 2 );
  Real rho = 0.0;
  Real p = 0.0;
  if ( r > 0.5 ) {
    rho = 0.0625 / pow( rsch * bondi_rs, 1.5 );
    p = eosk * pow(rho, gamma);
  }
  return x1*fmax(p - pcut, 0.0)*pow(1.0 - rho/rhoc,magindex);
}

// Metric update function
void SetADMVariables(MeshBlockPack *pmbp) {
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

    Real r = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
    Real s = sqrt(SQR(x1v) + SQR(x2v));

    // Set ADM variables
    Real psi2 = pow(1.0 + 0.5 / r, 2);
    Real psi4 = psi2 * psi2;
    adm.alpha(m,k,j,i) = 1.0 / psi2;
    adm.g_dd(m,0,0,k,j,i) = adm.g_dd(m,1,1,k,j,i) = adm.g_dd(m,2,2,k,j,i) = psi4;
    adm.g_dd(m,0,1,k,j,i) = adm.g_dd(m,0,2,k,j,i) = adm.g_dd(m,1,2,k,j,i) = 0.0;
    adm.psi4(m,k,j,i) = psi4;
    adm.beta_u(m,0,k,j,i) = adm.beta_u(m,1,k,j,i) = adm.beta_u(m,2,k,j,i) = 0.0;
    adm.vK_dd(m,0,0,k,j,i) = adm.vK_dd(m,0,1,k,j,i) = adm.vK_dd(m,0,2,k,j,i) = 0.0;
    adm.vK_dd(m,1,1,k,j,i) = adm.vK_dd(m,1,2,k,j,i) = adm.vK_dd(m,2,2,k,j,i) = 0.0;
  });
}

// Cleanup at the end of the run
void FinalizeBHStar(ParameterInput *pin, Mesh *pm) {
  return;
}

// History function
void BHStarHistory(HistoryData *pdata, Mesh *pm) {
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
  Real rho_max = std::numeric_limits<Real>::max();
  Real alpha_min = -rho_max;
  Kokkos::parallel_reduce("BHStarHistSums",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &mb_max, Real &mb_alp_min) {
    // coompute n,k,j,i indices of thread
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
    MPI_Reduce(&rho_max, &rho_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&alpha_min, &alpha_min, 1, MPI_ATHENA_REAL, MPI_MIN, 0, MPI_COMM_WORLD);
    rho_max = 0.;
    alpha_min = 0.;
  }
#endif

  // store data in hdata array
  pdata->hdata[0] = rho_max;
  pdata->hdata[1] = alpha_min;
}
