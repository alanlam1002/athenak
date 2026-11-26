//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_trumpet.cpp
//! \brief Stationary maximal-slicing trumpet black hole (2502.03223 Sec. III A 1), ported
//! from SACRA (sacra-aei/mod_bh.f90 tbh_get_var, tbh_iso_to_areal; background.f90
//! background_tbh) for M2d T2 (chat/0011).
//!
//! In isotropic radius r with areal radius R(r) (R >= 3M/2, the throat at r = 0):
//!   psi = sqrt(R/r), gamma_ij = psi^4 delta_ij, K = 0,
//!   alpha = sqrt(1 - 2M/R + C^2/R^4), C^2 = (27/16) M^4,
//!   beta^i = (3 sqrt(3)/4) M^2 x^i / R^3  (SACRA: fac x^i / (R (R/M)^2)),
//!   A-tilde_ij = (3 sqrt(3)/4) M^2 / R^3 (delta_ij - 3 x_i x_j / r^2),
//!   K_ij = psi^4 A-tilde_ij.
//! The data are spherically symmetric, so every component is the same formula in any
//! orthonormal frame; they are evaluated directly in mesh coordinates. Under 2D Cartoon
//! the centre must be on the symmetry axis (mesh x2). With the static gauge of S2
//! (d_t alpha = -alpha (1 - alpha) Khat, frozen shift) the solution is stationary.
//!
//! <problem> keys: punc_ADM_mass (M), punc_center_x1..x3 (mesh coordinates),
//! report_constraints / report_rmin / report_rmax (max |H|, |M_i| at t = 0), user_hist
//! (history: deviation from the exact solution for r > hist_rmin, axisymmetric volume
//! element under Cartoon):
//!   da-norm2, dchi-norm2, dKhat-norm2, dAt-norm2, vol-err

#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/adm.hpp"
#include "z4c/z4c_pos.hpp"
#include "outputs/outputs.hpp"
#include "pgen/pgen.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {

// problem constants for the history function
Real trumpet_m = 1.0, trumpet_c[3] = {0.0, 0.0, 0.0}, trumpet_hist_rmin = 1.0;

// Areal radius R (units of M) of isotropic radius rc (units of M): the root of
// r(R) = rc, with r(R) = (2R + 1 + sqrt(4R^2 + 4R + 3))/4 * q^(1/sqrt 2) and
// q = (4 + 3 sqrt 2)(2R - 3) / (8R + 6 + 3 sqrt(8R^2 + 8R + 6))
// (SACRA tbh_areal_to_iso). r(R) increases from 0 at R = 3/2; bisection to round-off.
// 2R - 3 is floored at 0 before the product: with fast fp on PVC the product can be
// distributed and come out -O(1e-15) at R = 3/2, and pow would give NaN (TDE C-095).
KOKKOS_INLINE_FUNCTION
Real TrumpetIsoOf(const Real z) {
  const Real d = Kokkos::fmax(2.0*z - 3.0, 0.0);
  return 0.25*(2.0*z + 1.0 + Kokkos::sqrt(4.0*z*z + 4.0*z + 3.0))
       * Kokkos::pow((4.0 + 3.0*Kokkos::sqrt(2.0))*d
                     /(8.0*z + 6.0 + 3.0*Kokkos::sqrt(8.0*z*z + 8.0*z + 6.0)),
                     1.0/Kokkos::sqrt(2.0));
}

KOKKOS_INLINE_FUNCTION
Real TrumpetIsoToAreal(const Real rc) {
  if (rc <= 0.0) return 1.5;
  Real lo = 1.5, hi = Kokkos::fmax(1.5, rc + 1.5);
  while (TrumpetIsoOf(hi) < rc) hi = 2.0*hi;
  for (int it = 0; it < 200; ++it) {
    const Real mid = 0.5*(lo + hi);
    if (mid <= lo || mid >= hi) break;
    if (TrumpetIsoOf(mid) < rc) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return Kokkos::fmax(0.5*(lo + hi), 1.5);
}

// Exact data at mesh offset X from the centre: psi, alpha, beta^i, A-tilde_ij (mesh
// order 00 01 02 11 12 22). SACRA tbh_get_var with m = M.
KOKKOS_INLINE_FUNCTION
void TrumpetExact(const Real m, const Real X[3], Real &psi, Real &alpha, Real beta[3],
                  Real At[6]) {
  const Real r = Kokkos::sqrt(X[0]*X[0] + X[1]*X[1] + X[2]*X[2]);
  const Real rs = m*TrumpetIsoToAreal(r/m);
  const Real rrs = rs/m;
  const Real fac = 0.75*Kokkos::sqrt(3.0);
  psi = Kokkos::sqrt(rs/r);
  alpha = Kokkos::sqrt(Kokkos::fmax(1.0 - 2.0/rrs + 1.6875/(rrs*rrs*rrs*rrs), 0.0));
  for (int a = 0; a < 3; ++a) beta[a] = fac*X[a]/(rs*rrs*rrs);
  const Real f = fac/(rs*rs*rs)*m*m;
  const int pa[6] = {0, 0, 0, 1, 1, 2}, pb[6] = {0, 1, 2, 1, 2, 2};
  for (int n = 0; n < 6; ++n) {
    At[n] = f*(((pa[n] == pb[n]) ? 1.0 : 0.0) - 3.0*X[pa[n]]*X[pb[n]]/(r*r));
  }
}

void TrumpetHistory(HistoryData *pdata, Mesh *pm);

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::Z4cTrumpet()

void ProblemGenerator::Z4cTrumpet(ParameterInput *pin, const bool restart) {
  user_hist_func = TrumpetHistory;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;
  if (pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "z4c_trumpet needs a <z4c> block" << std::endl;
    exit(EXIT_FAILURE);
  }
  const Real m0 = pin->GetOrAddReal("problem", "punc_ADM_mass", 1.0);
  const Real cx1 = pin->GetOrAddReal("problem", "punc_center_x1", 0.0);
  const Real cx2 = pin->GetOrAddReal("problem", "punc_center_x2", 0.0);
  const Real cx3 = pin->GetOrAddReal("problem", "punc_center_x3", 0.0);
  trumpet_m = m0; trumpet_c[0] = cx1; trumpet_c[1] = cx2; trumpet_c[2] = cx3;
  trumpet_hist_rmin = pin->GetOrAddReal("problem", "hist_rmin", 1.0);
  const bool cartoon = pmbp->pz4c->IsCartoon();
  if (cartoon && (cx1 != 0.0 || cx3 != 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << std::endl << "z4c_trumpet: under "
              << "2D Cartoon the black hole must lie on the symmetry axis (mesh x2): "
              << "punc_center_x1 = punc_center_x3 = 0 (D-017)" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (restart) return;
  const z4c::Z4c::CartoonMode cmode = pmbp->pz4c->opt.cartoon;

  auto &size = pmbp->pmb->mb_size;
  const int ng2 = (indcs.nx2 > 1) ? indcs.ng : 0;
  const int ng3 = (indcs.nx3 > 1) ? indcs.ng : 0;
  const int isg = indcs.is - indcs.ng, ieg = indcs.ie + indcs.ng;
  const int jsg = indcs.js - ng2, jeg = indcs.je + ng2;
  const int ksg = indcs.ks - ng3, keg = indcs.ke + ng3;
  const int nmb = pmbp->nmb_thispack;
  adm::ADM::ADM_vars &adm = pmbp->padm->adm;
  const RegionIndcs indcs_ = indcs;
  par_for("pgen trumpet", DevExeSpace(), 0, nmb-1, ksg, keg, jsg, jeg, isg, ieg,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real X[3];
    z4c::Z4cCartesianPos(cmode, size.d_view(m), indcs_, k, j, i, X[0], X[1], X[2]);
    X[0] -= cx1; X[1] -= cx2; X[2] -= cx3;
    Real psi, alpha, beta[3], At[6];
    TrumpetExact(m0, X, psi, alpha, beta, At);
    const Real psi4 = psi*psi*psi*psi;
    int n = 0;
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        adm.g_dd(m,a,b,k,j,i) = (a == b) ? psi4 : 0.0;
        adm.vK_dd(m,a,b,k,j,i) = psi4*At[n];
        ++n;
      }
      adm.beta_u(m,a,k,j,i) = beta[a];
    }
    adm.alpha(m,k,j,i) = alpha;
  });

  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin); break;
    case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin); break;
    case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin); break;
  }
  pmbp->pz4c->Z4cToADM(pmbp);
  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMConstraints<2>(pmbp); break;
    case 3: pmbp->pz4c->ADMConstraints<3>(pmbp); break;
    case 4: pmbp->pz4c->ADMConstraints<4>(pmbp); break;
  }

  // Optional: max |H|, |M_i| over interior cells with report_rmin < r < report_rmax
  if (pin->GetOrAddBoolean("problem", "report_constraints", false)) {
    const Real rmin = pin->GetOrAddReal("problem", "report_rmin", 1.0);
    const Real rmax = pin->GetOrAddReal("problem", "report_rmax", 4.0);
    auto &con = pmbp->pz4c->u_con;
    const int iH = z4c::Z4c::I_CON_H, iMx = z4c::Z4c::I_CON_MX;
    Real eh = 0.0, em = 0.0;
    Kokkos::parallel_reduce("trumpet_con",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(DevExeSpace(),
                                             {0, indcs.ks, indcs.js, indcs.is},
                                             {nmb, indcs.ke+1, indcs.je+1, indcs.ie+1}),
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, Real &h, Real &mm) {
      Real X[3];
      z4c::Z4cCartesianPos(cmode, size.d_view(m), indcs_, k, j, i, X[0], X[1], X[2]);
      X[0] -= cx1; X[1] -= cx2; X[2] -= cx3;
      const Real r = Kokkos::sqrt(X[0]*X[0] + X[1]*X[1] + X[2]*X[2]);
      if (r <= rmin || r >= rmax) return;
      h = Kokkos::fmax(h, Kokkos::fabs(con(m,iH,k,j,i)));
      for (int a = 0; a < 3; ++a) mm = Kokkos::fmax(mm, Kokkos::fabs(con(m,iMx+a,k,j,i)));
    }, Kokkos::Max<Real>(eh), Kokkos::Max<Real>(em));
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &eh, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &em, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
    if (global_variable::my_rank == 0) {
      std::printf("TRUMPET_CON maxH %.17e maxM %.17e\n", eh, em);
    }
  }
  if (global_variable::my_rank == 0) {
    std::cout << "z4c_trumpet: M = " << m0 << (cartoon ? " (2D Cartoon)" : " (3D)")
              << std::endl;
  }
}

namespace {

//----------------------------------------------------------------------------------------
// sum over interior cells with r > hist_rmin of (q - q_exact)^2 dV for alpha, chi,
// Khat and the A-tilde components, and of dV (2 pi R dR dz under Cartoon).
void TrumpetHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 5;
  pdata->label[0] = "da-norm2";
  pdata->label[1] = "dchi-norm2";
  pdata->label[2] = "dKhat-nrm2";
  pdata->label[3] = "dAt-norm2";
  pdata->label[4] = "vol-err";
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  auto &z4c = pmbp->pz4c->z4c;
  const z4c::Z4c::CartoonMode cmode = pmbp->pz4c->opt.cartoon;
  const bool cartoon = pmbp->pz4c->IsCartoon();
  const Real chi_pow = pmbp->pz4c->opt.chi_psi_power;
  const RegionIndcs indcs_ = indcs;
  const Real m0 = trumpet_m, rmin = trumpet_hist_rmin;
  const Real c0 = trumpet_c[0], c1 = trumpet_c[1], c2 = trumpet_c[2];
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1, nji = nx2*nx1;
  const int nmkji = pmbp->nmb_thispack*nkji;
  array_sum::GlobalSum sum;
  Kokkos::parallel_reduce("TrumpetHist", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &s) {
    const int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    const int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    array_sum::GlobalSum h;
    for (int n = 0; n < NHISTORY_VARIABLES; ++n) h.the_array[n] = 0.0;
    const auto &sz = size.d_view(m);
    Real X[3];
    z4c::Z4cCartesianPos(cmode, sz, indcs_, k, j, i, X[0], X[1], X[2]);
    const Real R = X[0];
    X[0] -= c0; X[1] -= c1; X[2] -= c2;
    const Real r = Kokkos::sqrt(X[0]*X[0] + X[1]*X[1] + X[2]*X[2]);
    if (r > rmin) {
      const Real dv = cartoon ? 2.0*M_PI*R*sz.dx1*sz.dx2 : sz.dx1*sz.dx2*sz.dx3;
      Real psi, alpha, beta[3], At[6];
      TrumpetExact(m0, X, psi, alpha, beta, At);
      // chi = psi^(chi_psi_power) (AthenaK: chi = (det g)^(chi_psi_power/12))
      const Real chi = Kokkos::pow(psi, chi_pow);
      // conformal A-tilde of the Z4c state = chi-scaled: A-tilde(z4c) = psi^-4 K_ij = At
      Real da2 = 0.0;
      int n = 0;
      for (int a = 0; a < 3; ++a) {
        for (int b = a; b < 3; ++b) {
          da2 += SQR(z4c.vA_dd(m,a,b,k,j,i) - At[n]);
          ++n;
        }
      }
      h.the_array[0] = dv*SQR(z4c.alpha(m,k,j,i) - alpha);
      h.the_array[1] = dv*SQR(z4c.chi(m,k,j,i) - chi);
      h.the_array[2] = dv*SQR(z4c.vKhat(m,k,j,i));
      h.the_array[3] = dv*da2;
      h.the_array[4] = dv;
    }
    s += h;
  }, Kokkos::Sum<array_sum::GlobalSum>(sum));
  for (int n = 0; n < pdata->nhist; ++n) pdata->hdata[n] = sum.the_array[n];
}

}  // namespace
