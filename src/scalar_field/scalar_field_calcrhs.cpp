//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file scalar_field_calcrhs.cpp
//! \brief RHS for the scalar-field sector.
//!
//! The scalar's own (massive, Phase 4) Klein-Gordon-like RHS, reading the Z4c/ADM
//! geometry and (Phase 3) the fluid's matter trace, but not writing back to the Z4c
//! equations itself (that back-reaction lives in z4c/z4c_calcrhs.cpp, Phase 2-4). The
//! continuum equations being discretized (see src/scalar_field/PLAN.md) are:
//!
//!   dt(sphi) = -alpha*Pi                                         (+ shift advection)
//!   dt(Pi)   = alpha*(K+2*Theta)*Pi - alpha*D2(sphi)
//!              - g^ij di(alpha) dj(sphi) - alpha*sphi*(|D(sphi)|^2 - Pi^2)
//!              + 2*pi*alpha*omega_c*T*sphi + m^2*alpha*sphi*A(sphi)
//!
//! where D2/|D.|^2 are built from the *physical* 3-metric, and T is the fluid's matter
//! trace (Einstein frame -- Tmunu has already been rescaled by 1/A(sphi) by
//! ScalarField::RescaleTmunu by the time this runs, see scalar_field_tasks.cpp), zero in
//! vacuum. The mass term (Phase 4) is fully explicit -- see PLAN.md's "Mass-term
//! treatment" section for why AthenaK's global (non-subcycled) timestep makes this safe,
//! unlike SACRA's implicit Newton-solve treatment of the same term.
//!
//! Derivative convention: mirrors z4c/z4c_calcrhs.cpp exactly -- all derivatives use the
//! *conformal* metric (pz4c->z4c.g_dd) and its Christoffel symbols, with physical
//! quantities recovered via the oopsi4 = 1/psi^4 conformal factor. The scalar field's
//! covariant Hessian D_iD_j(sphi) is built the same way z4c_calcrhs.cpp builds the
//! lapse's covariant Hessian Ddalpha_dd -- i.e. the conformal-covariant Hessian plus the
//! two chi-derivative correction terms that convert it to the physical-metric Hessian.
//! (Note: z4c_calcrhs.cpp's local variable name "dphi_d" refers to the derivative of the
//! BSSN conformal exponent, NOT our scalar field -- renamed dphibssn_d here to avoid
//! confusion with our own field, which is named sphi throughout this module.)

#include <math.h>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "coordinates/adm.hpp"
#include "z4c/z4c.hpp"
#include "z4c/tmunu.hpp"
#include "scalar_field/scalar_field.hpp"

namespace scalarfield {
//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ScalarField::CalcRHS
//! \brief compute rhs of the scalar-field equations
template <int NGHOST>
TaskStatus ScalarField::CalcRHS(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  int nmb = pmy_pack->nmb_thispack;

  auto &z4c = pmy_pack->pz4c->z4c;
  auto &zopt = pmy_pack->pz4c->opt;
  auto &sf = pmy_pack->pscalarfield->sf;
  auto &rhs = pmy_pack->pscalarfield->rhs;
  Real omega_c = pmy_pack->pscalarfield->opt.omega_c;
  Real mass2 = pmy_pack->pscalarfield->opt.mass2;

  bool is_vacuum = (pmy_pack->ptmunu == nullptr) ? true : false;
  Tmunu::Tmunu_vars tmunu;
  if (!is_vacuum) tmunu = pmy_pack->ptmunu->tmunu;

  par_for("sf rhs loop", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // inverse of the conformal 3-metric
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> g_uu;
    // Christoffel symbols of the conformal metric
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_ddd;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_udd;

    // 1st derivatives
    AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dalpha_d;
    AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dchi_d;
    AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dphibssn_d;  // BSSN conf. exponent
    AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dsphi_d;

    // metric 1st derivatives (needed for the conformal Christoffels)
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> dg_ddd;

    // 2nd derivatives of sphi and its physical-metric covariant Hessian
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> ddsphi_dd;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> Ddsphi_dd;

    Real idx[] = {1/size.d_view(m).dx1, 1/size.d_view(m).dx2, 1/size.d_view(m).dx3};

    Gamma_udd.ZeroClear();

    // -----------------------------------------------------------------------------------
    // 1st derivatives
    for (int a = 0; a < 3; ++a) {
      dalpha_d(a) = Dx<NGHOST>(a, idx, z4c.alpha, m,k,j,i);
      dchi_d(a)   = Dx<NGHOST>(a, idx, z4c.chi,   m,k,j,i);
      dsphi_d(a)  = Dx<NGHOST>(a, idx, sf.sphi,   m,k,j,i);
    }
    for (int a = 0; a < 3; ++a)
    for (int b = a; b < 3; ++b)
    for (int c = 0; c < 3; ++c) {
      dg_ddd(c,a,b) = Dx<NGHOST>(c, idx, z4c.g_dd, m,a,b,k,j,i);
    }

    // -----------------------------------------------------------------------------------
    // 2nd derivatives of sphi
    for (int a = 0; a < 3; ++a) {
      ddsphi_dd(a,a) = Dxx<NGHOST>(a, idx, sf.sphi, m,k,j,i);
      for (int b = a + 1; b < 3; ++b) {
        ddsphi_dd(a,b) = Dxy<NGHOST>(a, b, idx, sf.sphi, m,k,j,i);
      }
    }

    // -----------------------------------------------------------------------------------
    // Advective (Lie) derivatives along the shift
    Real Lsphi = 0.0;
    Real Lpi = 0.0;
    for (int a = 0; a < 3; ++a) {
      Lsphi += Lx<NGHOST>(a, idx, z4c.beta_u, sf.sphi, m,a,k,j,i);
      Lpi   += Lx<NGHOST>(a, idx, z4c.beta_u, sf.vpi,  m,a,k,j,i);
    }

    // -----------------------------------------------------------------------------------
    // Inverse conformal metric
    Real detg = adm::SpatialDet(z4c.g_dd(m,0,0,k,j,i), z4c.g_dd(m,0,1,k,j,i),
                                 z4c.g_dd(m,0,2,k,j,i), z4c.g_dd(m,1,1,k,j,i),
                                 z4c.g_dd(m,1,2,k,j,i), z4c.g_dd(m,2,2,k,j,i));
    adm::SpatialInv(1.0/detg,
               z4c.g_dd(m,0,0,k,j,i), z4c.g_dd(m,0,1,k,j,i), z4c.g_dd(m,0,2,k,j,i),
               z4c.g_dd(m,1,1,k,j,i), z4c.g_dd(m,1,2,k,j,i), z4c.g_dd(m,2,2,k,j,i),
               &g_uu(0,0), &g_uu(0,1), &g_uu(0,2),
               &g_uu(1,1), &g_uu(1,2), &g_uu(2,2));

    // -----------------------------------------------------------------------------------
    // Christoffel symbols of the conformal metric
    for (int c = 0; c < 3; ++c)
    for (int a = 0; a < 3; ++a)
    for (int b = a; b < 3; ++b) {
      Gamma_ddd(c,a,b) = 0.5*(dg_ddd(a,b,c) + dg_ddd(b,a,c) - dg_ddd(c,a,b));
    }
    for (int c = 0; c < 3; ++c)
    for (int a = 0; a < 3; ++a)
    for (int b = a; b < 3; ++b)
    for (int d = 0; d < 3; ++d) {
      Gamma_udd(c,a,b) += g_uu(c,d)*Gamma_ddd(d,a,b);
    }

    // -----------------------------------------------------------------------------------
    // Derivative of the BSSN conformal exponent (this is NOT our scalar field)
    Real chi_guarded = (z4c.chi(m,k,j,i) > zopt.chi_div_floor)
                          ? z4c.chi(m,k,j,i) : zopt.chi_div_floor;
    Real oopsi4 = pow(chi_guarded, -4./zopt.chi_psi_power);
    for (int a = 0; a < 3; ++a) {
      dphibssn_d(a) = dchi_d(a)/(chi_guarded * zopt.chi_psi_power);
    }

    // -----------------------------------------------------------------------------------
    // Physical-metric covariant Hessian of sphi (same construction as Z4c's Ddalpha_dd)
    for (int a = 0; a < 3; ++a)
    for (int b = a; b < 3; ++b) {
      Ddsphi_dd(a,b) = ddsphi_dd(a,b)
                     - 2.*(dphibssn_d(a)*dsphi_d(b) + dphibssn_d(b)*dsphi_d(a));
      for (int c = 0; c < 3; ++c) {
        Ddsphi_dd(a,b) -= Gamma_udd(c,a,b)*dsphi_d(c);
        for (int d = 0; d < 3; ++d) {
          Ddsphi_dd(a,b) += 2.*z4c.g_dd(m,a,b,k,j,i) * g_uu(c,d)
                          * dphibssn_d(c) * dsphi_d(d);
        }
      }
    }

    // Physical D^2(sphi), |D(sphi)|^2, and g^ij di(alpha) dj(sphi)
    Real Ddsphi = 0.0;
    Real gradsphi2 = 0.0;
    Real dalpha_dsphi = 0.0;
    for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) {
      Ddsphi       += oopsi4 * g_uu(a,b) * Ddsphi_dd(a,b);
      gradsphi2    += oopsi4 * g_uu(a,b) * dsphi_d(a) * dsphi_d(b);
      dalpha_dsphi += oopsi4 * g_uu(a,b) * dalpha_d(a) * dsphi_d(b);
    }

    // -----------------------------------------------------------------------------------
    // Matter trace T = -E + gamma^ij*S_ij (bssn_st.f90's ttrace); zero in vacuum. Tmunu
    // has already been rescaled to the Einstein frame by ScalarField::RescaleTmunu.
    Real T_matter = 0.0;
    if (!is_vacuum) {
      Real S_trace = 0.0;
      for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) {
        S_trace += oopsi4 * g_uu(a,b) * tmunu.S_dd(m,a,b,k,j,i);
      }
      T_matter = -tmunu.E(m,k,j,i) + S_trace;
    }

    // -----------------------------------------------------------------------------------
    // Assemble RHS
    Real K = z4c.vKhat(m,k,j,i) + 2.*z4c.vTheta(m,k,j,i);
    Real alpha = z4c.alpha(m,k,j,i);
    Real sphi_ = sf.sphi(m,k,j,i);
    Real pi_ = sf.vpi(m,k,j,i);
    // A(sphi) = exp(0.5*beta0*sphi^2), beta0=1 hardcoded (matches RescaleTmunu and every
    // other Phase 2/3 coefficient's implicit beta0=1, see PLAN.md).
    Real A_sphi = exp(0.5*SQR(sphi_));

    rhs.sphi(m,k,j,i) = -alpha*pi_ + Lsphi;
    rhs.vpi(m,k,j,i) = alpha*K*pi_ - alpha*Ddsphi - dalpha_dsphi
                     - alpha*sphi_*(gradsphi2 - SQR(pi_))
                     + 2.0*M_PI*alpha*omega_c*T_matter*sphi_
                     + mass2*alpha*sphi_*A_sphi + Lpi;
  });

  // ===================================================================================
  // Add dissipation for stability
  Real &diss = pmy_pack->pscalarfield->diss;
  auto &u0 = pmy_pack->pscalarfield->u0;
  auto &u_rhs = pmy_pack->pscalarfield->u_rhs;
  par_for("SF K-O Dissipation", DevExeSpace(),0,nmb-1,0,nscalarfield-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(const int m, const int n, const int k, const int j, const int i) {
    Real idx[] = {1/size.d_view(m).dx1, 1/size.d_view(m).dx2, 1/size.d_view(m).dx3};
    for (int a = 0; a < 3; ++a) {
      u_rhs(m,n,k,j,i) += Diss<NGHOST>(a, idx, u0, m, n, k, j, i)*diss;
    }
  });

  return TaskStatus::complete;
}

template TaskStatus ScalarField::CalcRHS<2>(Driver *pdriver, int stage);
template TaskStatus ScalarField::CalcRHS<3>(Driver *pdriver, int stage);
template TaskStatus ScalarField::CalcRHS<4>(Driver *pdriver, int stage);

} // namespace scalarfield
