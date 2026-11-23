//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file srcterms.cpp
//  Implements various (physics) source terms to be added to the Hydro or MHD eqns.
//  Source terms objects are stored in the respective fluid class, so that
//  Hydro/MHD can have different source terms

#include "srcterms.hpp"

#include <iostream>
#include <string> // string
#include <cstdio>  // snprintf
#include <cstdlib> // strtod

#include "athena.hpp"
#include "globals.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "gravity/gravity.hpp"
#include "hydro/hydro.hpp"
#include "ismcooling.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "radiation/radiation.hpp"
#include "radiation/radiation_tetrad.hpp"
#include "units/units.hpp"

//----------------------------------------------------------------------------------------
// constructor, parses input file and initializes data structures and parameters
// Only source terms specified in input file are initialized.
// Block name passed to constructor will be one of "hydro_srcterms", "mhd_srcterms",
// or "rad_srcterms"

SourceTerms::SourceTerms(std::string block, MeshBlockPack *pp, ParameterInput *pin) :
    pmy_pack(pp), pmy_pin(pin) {
  // Read flags for each source term implemented (default false)
  const_accel = pin->GetOrAddBoolean(block, "const_accel", false);
  ism_cooling = pin->GetOrAddBoolean(block, "ism_cooling", false);
  rel_cooling = pin->GetOrAddBoolean(block, "rel_cooling", false);
  rad_beam = pin->GetOrAddBoolean(block, "rad_beam", false);
  rad_shell = pin->GetOrAddBoolean(block, "rad_shell", false);
  self_gravity = pin->GetOrAddBoolean(block, "self_gravity", false);

  // (1) read data for (constant) gravitational acceleration
  if (const_accel) {
    const_accel_val = pin->GetReal(block, "const_accel_val");
    const_accel_dir = pin->GetInteger(block, "const_accel_dir");
    if (const_accel_dir < 1 || const_accel_dir > 3) {
      std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
                << "const_accle_dir must be 1,2, or 3" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // (2) Optically thin ISM cooling
  if (ism_cooling) {
    hrate = pin->GetReal(block, "hrate");
  }

  // (3) optically thin relativistic cooling
  if (rel_cooling) {
    crate_rel = pin->GetReal(block, "crate_rel");
    cpower_rel = pin->GetOrAddReal(block, "cpower_rel", 1.);
  }

  // (4) radiation beam source (radiation)
  if (rad_beam) {
    dii_dt = pin->GetReal(block, "dii_dt");
    pos1 = pin->GetReal(block, "pos_1");
    pos2 = pin->GetReal(block, "pos_2");
    pos3 = pin->GetReal(block, "pos_3");
    dir1 = pin->GetReal(block, "dir_1");
    dir2 = pin->GetReal(block, "dir_2");
    dir3 = pin->GetReal(block, "dir_3");
    width = pin->GetReal(block, "width");
    spread = pin->GetReal(block, "spread");
  }

  // (5) radiation shell source (radiation; bhstar Stage 0)
  if (rad_shell) {
    shell_r = pin->GetReal(block, "shell_r");
    shell_sigma = pin->GetReal(block, "shell_sigma");
    // total rate: either shell_lum (code units) or shell_lum_edd * L_Edd. In code units
    // (G = M = c = 1) L_Edd = 4 pi G M c / kappa_s = 4 pi / (kappa_s rho_unit l_unit),
    // the same product as sigma_s = rho kappa_s rho_unit l_unit in the opacity function.
    if (pin->DoesParameterExist(block, "shell_lum")) {
      shell_q = pin->GetReal(block, "shell_lum");
    } else {
      Real lum_edd = pin->GetReal(block, "shell_lum_edd");
      Real kap = pin->GetReal("radiation", "kappa_s");
      if (pmy_pack->punit == nullptr || kap <= 0.0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "rad_shell with shell_lum_edd needs <units> and "
                  << "kappa_s > 0; otherwise give shell_lum in code units" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      Real l_edd = 4.0*M_PI/(kap*pmy_pack->punit->density_cgs()
                             *pmy_pack->punit->length_cgs());
      shell_q = lum_edd*l_edd;
    }
    // normalisation from an earlier start of this run (restart): stored as a hex float
    // so that it is bit-exact (ParameterInput::SetReal keeps only 6 digits)
    if (pin->DoesParameterExist(block, "shell_norm_hex")) {
      shell_norm = std::strtod(pin->GetString(block, "shell_norm_hex").c_str(), nullptr);
      shell_norm_set = true;
    }
  }
}

//----------------------------------------------------------------------------------------
// destructor

SourceTerms::~SourceTerms() {
}

//----------------------------------------------------------------------------------------
//! \fn SourceTerms::ApplySrcTerms
//! \brief Applies selected source terms to input arrays. Two different versions are
//! implemented for fluid and radiation fields, distinguished by their argument lists

void SourceTerms::ApplySrcTerms(const DvceArray5D<Real> &w0, const EOS_Data &eos_data,
                                const Real bdt, DvceArray5D<Real> &u0) {
  // NOTE source terms must be computed using primitive (w0) and NOT conserved (u0) vars
  if (const_accel) ConstantAccel(w0, eos_data,  bdt, u0);
  if (ism_cooling) ISMCooling(w0, eos_data, bdt, u0);
  if (rel_cooling) RelCooling(w0, eos_data, bdt, u0);
  if (self_gravity) SelfGravity(w0, eos_data, bdt, u0);
  return;
}

void SourceTerms::ApplySrcTerms(DvceArray5D<Real> &i0, const Real bdt) {
  if (rad_beam) BeamSource(i0, bdt);
  if (rad_shell) ShellSource(i0, bdt);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn SourceTerms::ConstantAccel
//! \brief Add constant acceleration
//! NOTE source terms must be computed using primitive (w0) and NOT conserved (u0) vars

void SourceTerms::ConstantAccel(const DvceArray5D<Real> &w0, const EOS_Data &eos_data,
                                const Real bdt, DvceArray5D<Real> &u0) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  Real &g = const_accel_val;
  int &dir = const_accel_dir;

  par_for("const_acc", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real src = bdt*g*w0(m,IDN,k,j,i);
    u0(m,dir,k,j,i) += src;
    if (eos_data.is_ideal) { u0(m,IEN,k,j,i) += src*w0(m,dir,k,j,i); }
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SourceTerms::ISMCooling()
//! \brief Add explict ISM cooling and heating source terms in the energy equations.
//! NOTE source terms must be computed using primitive (w0) and NOT conserved (u0) vars

void SourceTerms::ISMCooling(const DvceArray5D<Real> &w0, const EOS_Data &eos_data,
                             const Real bdt, DvceArray5D<Real> &u0) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  Real gamma = eos_data.gamma;
  Real gm1 = gamma - 1.0;
  Real heating_rate = hrate;
  Real temp_unit = pmy_pack->punit->temperature_cgs();
  Real n_unit = pmy_pack->punit->density_cgs()/pmy_pack->punit->mu()
                /pmy_pack->punit->atomic_mass_unit_cgs;
  Real cooling_unit = pmy_pack->punit->pressure_cgs()/pmy_pack->punit->time_cgs()
                      /n_unit/n_unit;
  Real heating_unit = pmy_pack->punit->pressure_cgs()/pmy_pack->punit->time_cgs()/n_unit;

  par_for("cooling", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // temperature in cgs unit
    Real temp = temp_unit*w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
    Real lambda_cooling = ISMCoolFn(temp)/cooling_unit;
    Real gamma_heating = heating_rate/heating_unit;

    u0(m,IEN,k,j,i) -= bdt * w0(m,IDN,k,j,i) *
                        (w0(m,IDN,k,j,i) * lambda_cooling - gamma_heating);
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SourceTerms::RelCooling()
//! \brief Add explict relativistic cooling in the energy and momentum equations.
//! NOTE source terms must be computed using primitive (w0) and NOT conserved (u0) vars

void SourceTerms::RelCooling(const DvceArray5D<Real> &w0, const EOS_Data &eos_data,
                             const Real bdt, DvceArray5D<Real> &u0) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  Real gamma = eos_data.gamma;
  Real gm1 = gamma - 1.0;
  Real cooling_rate = crate_rel;
  Real cooling_power = cpower_rel;

  par_for("cooling", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // temperature in cgs unit
    Real temp = w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;

    auto &ux = w0(m,IVX,k,j,i);
    auto &uy = w0(m,IVY,k,j,i);
    auto &uz = w0(m,IVZ,k,j,i);
    Real ut = sqrt(1.0 + ux*ux + uy*uy + uz*uz);

    u0(m,IEN,k,j,i) -= bdt*w0(m,IDN,k,j,i)*ut*pow((temp*cooling_rate), cooling_power);
    u0(m,IM1,k,j,i) -= bdt*w0(m,IDN,k,j,i)*ux*pow((temp*cooling_rate), cooling_power);
    u0(m,IM2,k,j,i) -= bdt*w0(m,IDN,k,j,i)*uy*pow((temp*cooling_rate), cooling_power);
    u0(m,IM3,k,j,i) -= bdt*w0(m,IDN,k,j,i)*uz*pow((temp*cooling_rate), cooling_power);
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn SourceTerms::SelfGravity
//! \brief Adds source terms for self-gravitational acceleration to conserved variables
//! \note
//! This implements the source term formula in Mullen, Hanawa and Gammie 2020, but only
//! for the momentum part. The energy source term is not conservative in this version.
//! Also note that this implementation is not exactly conservative when the potential
//! contains a residual error (Multigrid has small but non-zero residual).

void SourceTerms::SelfGravity(const DvceArray5D<Real> &w0, const EOS_Data &eos_data,
                              const Real bdt, DvceArray5D<Real> &u0) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmy_pack->nmb_thispack;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  auto &mbsize = pmy_pack->pmb->mb_size;
  // Get gravitational potential - check if gravity is enabled
  if (pmy_pack->pgrav == nullptr) {
    std::cout << "### ERROR in SourceTerms::SelfGravity" << std::endl
              << "self_gravity source term enabled but pgrav is null" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &phi = pmy_pack->pgrav->phi;

  // Get Godunov density fluxes from Riemann solver
  // (following Mullen, Hanawa & Gammie 2020 for the energy source term)
  DvceArray5D<Real> flx1, flx2, flx3;
  if (pmy_pack->pmhd != nullptr) {
    flx1 = pmy_pack->pmhd->uflx.x1f;
    flx2 = pmy_pack->pmhd->uflx.x2f;
    flx3 = pmy_pack->pmhd->uflx.x3f;
  } else {
    flx1 = pmy_pack->phydro->uflx.x1f;
    flx2 = pmy_pack->phydro->uflx.x2f;
    flx3 = pmy_pack->phydro->uflx.x3f;
  }

  // x1-direction momentum and energy source terms
  par_for("selfgrav_x1",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real dx1 = mbsize.d_view(m).dx1;
    Real hdtodx1 = 0.5*bdt/dx1;
    Real dpl = -(phi(m,0,k,j,i  ) - phi(m,0,k,j,i-1));
    Real dpr = -(phi(m,0,k,j,i+1) - phi(m,0,k,j,i  ));

    // Add momentum source term
    u0(m,IM1,k,j,i) += hdtodx1 * w0(m,IDN,k,j,i) * (dpl + dpr);

    // Add energy source term using Godunov fluxes (ideal EOS only)
    if (eos_data.is_ideal) {
      u0(m,IEN,k,j,i) += hdtodx1 * (flx1(m,IDN,k,j,i  ) * dpl
                                    + flx1(m,IDN,k,j,i+1) * dpr);
    }
  });

  if (multi_d) {
    // x2-direction momentum and energy source terms
    par_for("selfgrav_x2",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real dx2 = mbsize.d_view(m).dx2;
      Real hdtodx2 = 0.5*bdt/dx2;
      Real dpl = -(phi(m,0,k,j,  i) - phi(m,0,k,j-1,i));
      Real dpr = -(phi(m,0,k,j+1,i) - phi(m,0,k,j,  i));

      // Add momentum source term
      u0(m,IM2,k,j,i) += hdtodx2 * w0(m,IDN,k,j,i) * (dpl + dpr);

      // Add energy source term using Godunov fluxes (ideal EOS only)
      if (eos_data.is_ideal) {
        u0(m,IEN,k,j,i) += hdtodx2 * (flx2(m,IDN,k,j,  i) * dpl
                                      + flx2(m,IDN,k,j+1,i) * dpr);
      }
    });
  }

  if (three_d) {
    // x3-direction momentum and energy source terms
    par_for("selfgrav_x3",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real dx3 = mbsize.d_view(m).dx3;
      Real hdtodx3 = 0.5*bdt/dx3;
      Real dpl = -(phi(m,0,k,  j,i) - phi(m,0,k-1,j,i));
      Real dpr = -(phi(m,0,k+1,j,i) - phi(m,0,k,  j,i));

      // Add momentum source term
      u0(m,IM3,k,j,i) += hdtodx3 * w0(m,IDN,k,j,i) * (dpl + dpr);

      // Add energy source term using Godunov fluxes (ideal EOS only)
      if (eos_data.is_ideal) {
        u0(m,IEN,k,j,i) += hdtodx3 * (flx3(m,IDN,k,  j,i) * dpl
                                      + flx3(m,IDN,k+1,j,i) * dpr);
      }
    });
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn SourceTerms::BeamSource()
//! \brief Add beam of radiation at position (pos1,pos2,pos3) moving in direction
//! (dir1,dir2,dir3) with physical width and angular spread (width,spread)

void SourceTerms::BeamSource(DvceArray5D<Real> &i0, const Real bdt) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = (pmy_pack->nmb_thispack-1);
  int nang1 = (pmy_pack->prad->prgeo->nangles-1);

  auto &size = pmy_pack->pmb->mb_size;
  auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmy_pack->pcoord->coord_data.bh_spin;

  Real &p1 = pos1, &p2 = pos2, &p3 = pos3;
  Real &d1 = dir1, &d2 = dir2, &d3 = dir3;
  Real &dii_dt_ = dii_dt;
  Real &width_ = width;
  Real &spread_ = spread;

  auto &tc = pmy_pack->prad->tetcov_c;
  auto &nh_c_ = pmy_pack->prad->nh_c;
  auto &tet_c_ = pmy_pack->prad->tet_c;
  par_for("rad_beam",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
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

    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1v,x2v,x3v,flat,spin,glower,gupper);
    Real dgx[4][4], dgy[4][4], dgz[4][4];
    ComputeMetricDerivatives(x1v,x2v,x3v,flat,spin,dgx,dgy,dgz);
    Real e[4][4], e_cov[4][4], omega[4][4][4];
    ComputeTetrad(x1v,x2v,x3v,flat,spin,glower,gupper,dgx,dgy,dgz,e,e_cov,omega);

    // Calculate proper distance to beam origin and minimum angle between directions
    Real dx1 = x1v - p1;
    Real dx2 = x2v - p2;
    Real dx3 = x3v - p3;
    Real dx_sq = glower[1][1]*dx1*dx1 +2.0*glower[1][2]*dx1*dx2 + 2.0*glower[1][3]*dx1*dx3
               + glower[2][2]*dx2*dx2 +2.0*glower[2][3]*dx2*dx3
               + glower[3][3]*dx3*dx3;
    Real mu_min = cos(spread_/2.0*M_PI/180.0);

    // Calculate contravariant time component of direction
    Real temp_a = glower[0][0];
    Real temp_b = 2.0*(glower[0][1]*d1 + glower[0][2]*d2 + glower[0][3]*d3);
    Real temp_c = glower[1][1]*d1*d1 + 2.0*glower[1][2]*d1*d2 + 2.0*glower[1][3]*d1*d3
                + glower[2][2]*d2*d2 + 2.0*glower[2][3]*d2*d3
                + glower[3][3]*d3*d3;
    Real d0 = ((-temp_b - sqrt(SQR(temp_b) - 4.0*temp_a*temp_c))/(2.0*temp_a));

    // lower indices
    Real dc0 = glower[0][0]*d0 + glower[0][1]*d1 + glower[0][2]*d2 + glower[0][3]*d3;
    Real dc1 = glower[0][1]*d0 + glower[1][1]*d1 + glower[1][2]*d2 + glower[1][3]*d3;
    Real dc2 = glower[0][2]*d0 + glower[1][2]*d1 + glower[2][2]*d2 + glower[2][3]*d3;
    Real dc3 = glower[0][3]*d0 + glower[1][3]*d1 + glower[2][3]*d2 + glower[3][3]*d3;

    // Calculate covariant direction in tetrad frame
    Real dtc0 = (tet_c_(m,0,0,k,j,i)*dc0 + tet_c_(m,0,1,k,j,i)*dc1 +
                 tet_c_(m,0,2,k,j,i)*dc2 + tet_c_(m,0,3,k,j,i)*dc3);
    Real dtc1 = (tet_c_(m,1,0,k,j,i)*dc0 + tet_c_(m,1,1,k,j,i)*dc1 +
                 tet_c_(m,1,2,k,j,i)*dc2 + tet_c_(m,1,3,k,j,i)*dc3)/(-dtc0);
    Real dtc2 = (tet_c_(m,2,0,k,j,i)*dc0 + tet_c_(m,2,1,k,j,i)*dc1 +
                 tet_c_(m,2,2,k,j,i)*dc2 + tet_c_(m,2,3,k,j,i)*dc3)/(-dtc0);
    Real dtc3 = (tet_c_(m,3,0,k,j,i)*dc0 + tet_c_(m,3,1,k,j,i)*dc1 +
                 tet_c_(m,3,2,k,j,i)*dc2 + tet_c_(m,3,3,k,j,i)*dc3)/(-dtc0);

    // Go through angles
    for (int n=0; n<=nang1; ++n) {
      Real mu = (nh_c_.d_view(n,1) * dtc1
               + nh_c_.d_view(n,2) * dtc2
               + nh_c_.d_view(n,3) * dtc3);
      if ((dx_sq < SQR(width_/2.0)) && (mu > mu_min)) {
        Real n0 = tet_c_(m,0,0,k,j,i);
        Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,0,k,j,i)*nh_c_.d_view(n,1)
                 + tc(m,2,0,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
        i0(m,n,k,j,i) += n0*n_0*dii_dt_*bdt;
      }
    }
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn SourceTerms::ShellSource()
//! \brief Comoving-isotropic radiation source in a Gaussian shell in KS radius
//! (bhstar Stage 0, numerics-005 item 2).
//!
//! The code stores i0 = n^0 n_0 I_tet/(4 pi) per angle, and the comoving (fluid-frame)
//! intensity is I_cm = 4 pi (i0/(n^0 n_0)) (n0_cm)^4, with n0_cm = -u.n the photon energy
//! seen by the fluid for unit tetrad-frame energy (RadFluidCoupling uses the same
//! relations). An isotropic comoving source adds the same dI_cm to every angle:
//!     d i0_n = n^0 n_0 dI_cm / (4 pi n0_cm^4),     dI_cm = shell_norm * G(r) * dt,
//! G(r) = exp(-(r - shell_r)^2 / (2 shell_sigma^2)). Angles along the flow (n0_cm < 1)
//! get more lab-frame intensity: this is the Doppler beaming of an isotropic source.
//! shell_norm is fixed once (ShellNormalise) so that the code's own energy measure of
//! the injected energy, -sum_n w_n d i0_n dV, equals shell_q * dt for the initial state.

void SourceTerms::ShellSource(DvceArray5D<Real> &i0, const Real bdt) {
  if (!shell_norm_set) ShellNormalise();
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = (pmy_pack->nmb_thispack-1);
  int nang1 = (pmy_pack->prad->prgeo->nangles-1);
  auto &size = pmy_pack->pmb->mb_size;
  auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmy_pack->pcoord->coord_data.bh_spin;
  auto &tc = pmy_pack->prad->tetcov_c;
  auto &tt = pmy_pack->prad->tet_c;
  auto &nh_c_ = pmy_pack->prad->nh_c;
  auto &norm_to_tet_ = pmy_pack->prad->norm_to_tet;
  const bool fluid = (pmy_pack->pmhd != nullptr) || (pmy_pack->phydro != nullptr);
  DvceArray5D<Real> w0_;
  if (pmy_pack->pmhd != nullptr) {
    w0_ = pmy_pack->pmhd->w0;
  } else if (pmy_pack->phydro != nullptr) {
    w0_ = pmy_pack->phydro->w0;
  }
  const Real r0 = shell_r, sig = shell_sigma;
  const Real amp = shell_norm*bdt;

  par_for("rad_shell",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x1v = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real x2v = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real x3v = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    // KS radius (Cartesian radius in flat space)
    Real rr2 = SQR(x1v) + SQR(x2v) + SQR(x3v);
    Real a2 = flat ? 0.0 : SQR(spin);
    Real r = sqrt(0.5*(rr2 - a2 + sqrt(SQR(rr2 - a2) + 4.0*a2*SQR(x3v))));
    Real xg = (r - r0)/sig;
    if (fabs(xg) > 12.0) return;   // exp(-72) ~ 5e-32: nothing to add
    Real g = exp(-0.5*SQR(xg));

    // fluid four-velocity in the tetrad frame (as in RadFluidCoupling)
    Real u_tet[4] = {1.0, 0.0, 0.0, 0.0};
    if (fluid) {
      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1v,x2v,x3v,flat,spin,glower,gupper);
      Real wvx = w0_(m,IVX,k,j,i), wvy = w0_(m,IVY,k,j,i), wvz = w0_(m,IVZ,k,j,i);
      Real q = glower[1][1]*wvx*wvx + 2.0*glower[1][2]*wvx*wvy + 2.0*glower[1][3]*wvx*wvz
             + glower[2][2]*wvy*wvy + 2.0*glower[2][3]*wvy*wvz + glower[3][3]*wvz*wvz;
      Real gam = sqrt(1.0 + q);
      for (int a=0; a<4; ++a) {
        u_tet[a] = norm_to_tet_(m,a,0,k,j,i)*gam + norm_to_tet_(m,a,1,k,j,i)*wvx
                 + norm_to_tet_(m,a,2,k,j,i)*wvy + norm_to_tet_(m,a,3,k,j,i)*wvz;
      }
    }
    Real n0 = tt(m,0,0,k,j,i);
    for (int n=0; n<=nang1; ++n) {
      Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,0,k,j,i)*nh_c_.d_view(n,1)
               + tc(m,2,0,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
      Real n0_cm = u_tet[0]*nh_c_.d_view(n,0) - u_tet[1]*nh_c_.d_view(n,1)
                 - u_tet[2]*nh_c_.d_view(n,2) - u_tet[3]*nh_c_.d_view(n,3);
      i0(m,n,k,j,i) += n0*n_0*amp*g/(4.0*M_PI*SQR(SQR(n0_cm)));
    }
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn SourceTerms::ShellNormalise()
//! \brief Fix shell_norm so that the injected energy at infinity per coordinate time,
//! -sum_cells dV sum_n w_n d i0_n / dt, equals shell_q for the current fluid state
//! (called once: at start-up, from the problem generator or the first source call).
//! The result is stored in the parameter file as a hex float, so a restart reuses it.

void SourceTerms::ShellNormalise() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1, nji = nx2*nx1;
  const int nmkji = (pmy_pack->nmb_thispack)*nkji;
  int nang1 = (pmy_pack->prad->prgeo->nangles-1);
  auto &size = pmy_pack->pmb->mb_size;
  auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmy_pack->pcoord->coord_data.bh_spin;
  auto &tc = pmy_pack->prad->tetcov_c;
  auto &tt = pmy_pack->prad->tet_c;
  auto &nh_c_ = pmy_pack->prad->nh_c;
  auto &w_ = pmy_pack->prad->prgeo->solid_angles;
  auto &norm_to_tet_ = pmy_pack->prad->norm_to_tet;
  const bool fluid = (pmy_pack->pmhd != nullptr) || (pmy_pack->phydro != nullptr);
  DvceArray5D<Real> w0_;
  if (pmy_pack->pmhd != nullptr) {
    w0_ = pmy_pack->pmhd->w0;
  } else if (pmy_pack->phydro != nullptr) {
    w0_ = pmy_pack->phydro->w0;
  }
  const Real r0 = shell_r, sig = shell_sigma;
  Real wsum = 0.0;
  Kokkos::parallel_reduce("rad_shell_norm", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &sum) {
    int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    Real x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real x2v = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real x3v = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    Real rr2 = SQR(x1v) + SQR(x2v) + SQR(x3v);
    Real a2 = flat ? 0.0 : SQR(spin);
    Real r = sqrt(0.5*(rr2 - a2 + sqrt(SQR(rr2 - a2) + 4.0*a2*SQR(x3v))));
    Real xg = (r - r0)/sig;
    if (fabs(xg) > 12.0) return;
    Real g = exp(-0.5*SQR(xg));
    Real u_tet[4] = {1.0, 0.0, 0.0, 0.0};
    if (fluid) {
      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1v,x2v,x3v,flat,spin,glower,gupper);
      Real wvx = w0_(m,IVX,k,j,i), wvy = w0_(m,IVY,k,j,i), wvz = w0_(m,IVZ,k,j,i);
      Real q = glower[1][1]*wvx*wvx + 2.0*glower[1][2]*wvx*wvy + 2.0*glower[1][3]*wvx*wvz
             + glower[2][2]*wvy*wvy + 2.0*glower[2][3]*wvy*wvz + glower[3][3]*wvz*wvz;
      Real gam = sqrt(1.0 + q);
      for (int a=0; a<4; ++a) {
        u_tet[a] = norm_to_tet_(m,a,0,k,j,i)*gam + norm_to_tet_(m,a,1,k,j,i)*wvx
                 + norm_to_tet_(m,a,2,k,j,i)*wvy + norm_to_tet_(m,a,3,k,j,i)*wvz;
      }
    }
    Real n0 = tt(m,0,0,k,j,i);
    Real dv = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    for (int n=0; n<=nang1; ++n) {
      Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,0,k,j,i)*nh_c_.d_view(n,1)
               + tc(m,2,0,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
      Real n0_cm = u_tet[0]*nh_c_.d_view(n,0) - u_tet[1]*nh_c_.d_view(n,1)
                 - u_tet[2]*nh_c_.d_view(n,2) - u_tet[3]*nh_c_.d_view(n,3);
      sum += w_.d_view(n)*n0*n_0*g*dv/(4.0*M_PI*SQR(SQR(n0_cm)));
    }
  }, Kokkos::Sum<Real>(wsum));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &wsum, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (wsum == 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "rad_shell: the shell does not overlap the mesh" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // injected energy per unit amplitude is -wsum (E = -sum w i0 dV, and n^0 n_0 < 0)
  shell_norm = -shell_q/wsum;
  shell_norm_set = true;
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%a", shell_norm);
  pmy_pin->SetString("rad_srcterms", "shell_norm_hex", std::string(buf));
  if (global_variable::my_rank == 0) {
    std::cout << "rad_shell: total rate Q = " << shell_q << " (code units), amplitude "
              << shell_norm << " (stored as " << buf << ")" << std::endl;
  }
}
