//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_kerr_trumpet.cpp
//! \brief Kerr's stationary maximal trumpet slice (spinning_trumpet, kslice KC-9).
//!
//! Two sources of data, chosen by <problem>/kstab_file:
//!  - empty (default): the analytic Schwarzschild maximal trumpet, arXiv:2502.03223
//!    Eqs. 65-66 (M = 1): gamma_ij = (R/r)^2 delta_ij, K_ij = psi^4 (C/R^3)(delta_ij -
//!    3 n_i n_j), alpha = sqrt(1 - 2/R + C^2/R^4), beta^i = C x^i/R^3,
//!    C = 3 sqrt(3)/4, with
//!    the areal R from the isotropic r by inverting Eq. 66;
//!  - a KSTAB v1 table (kslice KC-8; read with kstab.hpp, vendored from
//!    spinning_trumpet): the 16 Cartesian fields at (x, y, z), its fallback below r_min
//!    as KSTAB v1 specifies.
//! The gauge is the slice's own stationary one: z4c.alpha = alpha, z4c.beta_u = beta^i,
//! B^i = 0 (no GaugePreCollapsedLapse). The data are evaluated on the host and copied.
//! <problem>: kstab_file (string), mass (must be 1: the tables and Eq. 66 are for M = 1).

#include <cmath>
#include <cstdio>
#include <iostream>   // endl
#include <string>     // c_str(), string

#include "athena.hpp"
#include "parameter_input.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "pgen/z4c/kstab.hpp"

namespace {
// Eq. 66 (M = 1): isotropic r of the areal R on the maximal trumpet
double RIsotropic(double R) {
  return ((2.0*R + 1.0 + std::sqrt(4.0*R*R + 4.0*R + 3.0))/4.0) *
         std::pow((4.0 + 3.0*std::sqrt(2.0))*(2.0*R - 3.0) /
                  (8.0*R + 6.0 + 3.0*std::sqrt(8.0*R*R + 8.0*R + 6.0)),
                  1.0/std::sqrt(2.0));
}
// R(r) by bisection in u = ln(R - 3/2), polished by Newton (r(R) is monotonic)
double RAreal(double r) {
  double lo = -745.0, hi = std::log(2.0*r + 10.0);
  for (int it = 0; it < 200 && hi - lo > 1e-15*(1.0 + std::fabs(hi)); ++it) {
    double mid = 0.5*(lo + hi);
    if (RIsotropic(1.5 + std::exp(mid)) < r) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  double u = 0.5*(lo + hi);
  for (int it = 0; it < 3; ++it) {   // d r/du by central differences in u
    double R = 1.5 + std::exp(u), h = 1e-6;
    double f = RIsotropic(R) - r;
    double d = (RIsotropic(1.5 + std::exp(u + h)) - RIsotropic(1.5 + std::exp(u - h)))
               / (2.0*h);
    if (d > 0.0) u -= f/d;
  }
  return 1.5 + std::exp(u);
}
// the analytic fields at (x, y, z), in KSTAB order:
//   gxx gxy gxz gyy gyz gzz Kxx .. Kzz alp beta^i
void AnalyticTrumpet(double x, double y, double z, double f[16]) {
  const double r = std::sqrt(x*x + y*y + z*z);
  const double R = RAreal(r);
  const double C = 3.0*std::sqrt(3.0)/4.0, psi4 = (R/r)*(R/r);
  const double n[3] = {x/r, y/r, z/r};
  const int ii[6] = {0, 0, 0, 1, 1, 2}, jj[6] = {0, 1, 2, 1, 2, 2};
  for (int q = 0; q < 6; ++q) {
    const double d = (ii[q] == jj[q]) ? 1.0 : 0.0;
    f[q] = psi4*d;
    f[6 + q] = psi4*(C/(R*R*R))*(d - 3.0*n[ii[q]]*n[jj[q]]);
  }
  // sqrt(1 - 2/R + 27/(16 R^4)) without the cancellation at the cylinder R -> 3/2
  f[12] = (R - 1.5)*std::sqrt(R*R + R + 0.75)/(R*R);
  f[13] = C*x/(R*R*R);
  f[14] = C*y/(R*R*R);
  f[15] = C*z/(R*R*R);
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Kerr trumpet initial data (analytic a = 0, or a KSTAB v1 table)
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;
  if (pmbp->pz4c == nullptr || pmbp->padm == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "z4c_kerr_trumpet needs a <z4c> block" << std::endl;
    exit(EXIT_FAILURE);
  }
  const std::string tabfile = pin->GetOrAddString("problem", "kstab_file", "");
  const Real mass = pin->GetOrAddReal("problem", "mass", 1.0);
  if (mass != 1.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "z4c_kerr_trumpet: the data are for M = 1 (problem/mass = 1)"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  kstab::Table *tab = nullptr;
  if (!tabfile.empty()) {
    tab = new kstab::Table(tabfile);
    if (global_variable::my_rank == 0) {
      std::cout << "z4c_kerr_trumpet: KSTAB table " << tabfile << ", a = " << tab->a
                << ", r_min = " << tab->rmin << std::endl;
    }
  } else if (global_variable::my_rank == 0) {
    std::cout << "z4c_kerr_trumpet: analytic Schwarzschild trumpet (Eqs. 65-66)"
              << std::endl;
  }

  // host mirrors of the ADM and Z4c arrays
  auto &u_adm = pmbp->padm->u_adm;
  auto &u0 = pmbp->pz4c->u0;
  auto h_adm = Kokkos::create_mirror_view(u_adm);
  auto h_u0 = Kokkos::create_mirror_view(u0);
  Kokkos::deep_copy(h_adm, u_adm);
  Kokkos::deep_copy(h_u0, u0);
  auto &size = pmbp->pmb->mb_size;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks, ng = indcs.ng;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmb = pmbp->nmb_thispack;
  const int ia = adm::ADM::I_ADM_GXX, ik = adm::ADM::I_ADM_KXX;
  double rmin_seen = 1e300;
  for (int m = 0; m < nmb; ++m) {
    const Real x1min = size.h_view(m).x1min, x1max = size.h_view(m).x1max;
    const Real x2min = size.h_view(m).x2min, x2max = size.h_view(m).x2max;
    const Real x3min = size.h_view(m).x3min, x3max = size.h_view(m).x3max;
    for (int k = ks - ng; k <= indcs.ke + ng; ++k)
    for (int j = js - ng; j <= indcs.je + ng; ++j)
    for (int i = is - ng; i <= indcs.ie + ng; ++i) {
      const double x = CellCenterX(i - is, nx1, x1min, x1max);
      const double y = CellCenterX(j - js, nx2, x2min, x2max);
      const double z = CellCenterX(k - ks, nx3, x3min, x3max);
      rmin_seen = std::fmin(rmin_seen, std::sqrt(x*x + y*y + z*z));
      double f[16];
      if (tab != nullptr) tab->eval(x, y, z, f);
      else AnalyticTrumpet(x, y, z, f);
      for (int q = 0; q < 6; ++q) {
        h_adm(m, ia + q, k, j, i) = f[q];
        h_adm(m, ik + q, k, j, i) = f[6 + q];
      }
      // psi^4 = det(gamma)^(1/3) (ADMToZ4c uses det g; psi4 is kept consistent)
      const double detg = adm::SpatialDet(f[0], f[1], f[2], f[3], f[4], f[5]);
      h_adm(m, adm::ADM::I_ADM_PSI4, k, j, i) = std::cbrt(detg);
      // with z4c, adm.alpha and adm.beta_u alias z4c.alpha and z4c.beta_u (adm.cpp)
      h_u0(m, z4c::Z4c::I_Z4C_ALPHA, k, j, i) = f[12];
      for (int a = 0; a < 3; ++a) {
        h_u0(m, z4c::Z4c::I_Z4C_BETAX + a, k, j, i) = f[13 + a];
        h_u0(m, z4c::Z4c::I_Z4C_BX + a, k, j, i) = 0.0;
      }
    }
  }
  Kokkos::deep_copy(u_adm, h_adm);
  Kokkos::deep_copy(u0, h_u0);
  if (tab != nullptr) {
    std::printf("z4c_kerr_trumpet: rank %d, smallest cell-centre radius %.4e, KSTAB "
                "fallback (r_f < r_min) used at %ld points\n", global_variable::my_rank,
                rmin_seen, tab->fallback_count());
    delete tab;
  }

  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
            break;
    case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
            break;
    case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
            break;
  }
  pmbp->pz4c->Z4cToADM(pmbp);
  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMConstraints<2>(pmbp);
            break;
    case 3: pmbp->pz4c->ADMConstraints<3>(pmbp);
            break;
    case 4: pmbp->pz4c->ADMConstraints<4>(pmbp);
            break;
  }
  if (global_variable::my_rank == 0) {
    std::cout << "z4c_kerr_trumpet initialized." << std::endl;
  }
  return;
}
