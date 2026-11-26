//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_gauge_pulse.cpp
//! \brief Flat space with an axisymmetric pure-gauge lapse pulse (M2b, M2d T1;
//! chat/0009, chat/0011). g_ij = delta_ij, K_ij = 0, beta^i = 0 and
//!   alpha = 1 + A exp(-((R - R0)^2 + (z - z0)^2)/w^2),  R = sqrt(x^2 + y^2),
//! in the physical frame (axis = physical z). Under 2D Cartoon physical (x, y, z) =
//! mesh (x1, -x3, x2) (D-017), so the axis is mesh x2; on a 3D Cartesian mesh physical =
//! mesh and the axis is mesh x3. The data go through ADMToZ4c. The gauge evolution then
//! turns the lapse pulse into outgoing gauge waves, so the test watches them leave
//! through the Sommerfeld outer boundary.
//!
//! <problem> keys: pulse_amp (A), pulse_width (w), pulse_r0 (R0, ring radius; 0 = a blob
//! on the axis), pulse_z0 (z0, along the axis). user_hist = true adds the history columns
//!   a-1-norm2, Khat-norm2, Thet-norm2, Gam-norm2, vol-axi
//! (history labels are cut to 10 characters)
//! = sum of (.)^2 dV and sum of dV, with the AXISYMMETRIC volume element: under Cartoon
//! dV = 2 pi R dR dz (each plane cell stands for a ring), on a 3D mesh dV = dx dy dz.

#include <cmath>
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

namespace {

void GaugePulseHistory(HistoryData *pdata, Mesh *pm);

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::Z4cGaugePulse()

void ProblemGenerator::Z4cGaugePulse(ParameterInput *pin, const bool restart) {
  user_hist_func = GaugePulseHistory;
  if (restart) return;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;
  if (pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "z4c_gauge_pulse needs a <z4c> block" << std::endl;
    exit(EXIT_FAILURE);
  }
  const Real amp = pin->GetOrAddReal("problem", "pulse_amp", 1.0e-3);
  const Real w = pin->GetOrAddReal("problem", "pulse_width", 1.0);
  const Real r0 = pin->GetOrAddReal("problem", "pulse_r0", 0.0);
  const Real z0 = pin->GetOrAddReal("problem", "pulse_z0", 0.0);
  const z4c::Z4c::CartoonMode cmode = pmbp->pz4c->opt.cartoon;
  const bool cartoon = pmbp->pz4c->IsCartoon();

  auto &size = pmbp->pmb->mb_size;
  const int ng2 = (indcs.nx2 > 1) ? indcs.ng : 0;
  const int ng3 = (indcs.nx3 > 1) ? indcs.ng : 0;
  const int isg = indcs.is - indcs.ng, ieg = indcs.ie + indcs.ng;
  const int jsg = indcs.js - ng2, jeg = indcs.je + ng2;
  const int ksg = indcs.ks - ng3, keg = indcs.ke + ng3;
  const int nmb = pmbp->nmb_thispack;
  adm::ADM::ADM_vars &adm = pmbp->padm->adm;
  const RegionIndcs indcs_ = indcs;
  par_for("pgen gauge pulse", DevExeSpace(), 0, nmb-1, ksg, keg, jsg, jeg, isg, ieg,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real X[3];
    z4c::Z4cCartesianPos(cmode, size.d_view(m), indcs_, k, j, i, X[0], X[1], X[2]);
    // physical (x, y, z): Cartoon (X1, -X3, X2); 3D physical = mesh
    const Real xp = X[0];
    const Real yp = cartoon ? -X[2] : X[1];
    const Real zp = cartoon ? X[1] : X[2];
    const Real R = Kokkos::sqrt(xp*xp + yp*yp);
    const Real d2 = (R - r0)*(R - r0) + (zp - z0)*(zp - z0);
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        adm.g_dd(m,a,b,k,j,i) = (a == b) ? 1.0 : 0.0;
        adm.vK_dd(m,a,b,k,j,i) = 0.0;
      }
      adm.beta_u(m,a,k,j,i) = 0.0;
    }
    adm.alpha(m,k,j,i) = 1.0 + amp*Kokkos::exp(-d2/(w*w));
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
  if (global_variable::my_rank == 0) {
    std::cout << "z4c_gauge_pulse: A = " << amp << ", w = " << w << ", R0 = " << r0
              << ", z0 = " << z0 << (cartoon ? " (2D Cartoon, axis = mesh x2)"
                                             : " (3D, axis = mesh x3)") << std::endl;
  }
}

namespace {

//----------------------------------------------------------------------------------------
// sum (alpha - 1)^2 dV, Khat^2 dV, Theta^2 dV, |Gamma|^2 dV, dV over the interior, with
// the axisymmetric volume element (2 pi R dR dz under Cartoon). Summed over ranks by the
// history output.
void GaugePulseHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 5;
  pdata->label[0] = "a-1-norm2";
  pdata->label[1] = "Khat-norm2";
  pdata->label[2] = "Thet-norm2";
  pdata->label[3] = "Gam-norm2";
  pdata->label[4] = "vol-axi";
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  auto &z4c = pmbp->pz4c->z4c;
  const z4c::Z4c::CartoonMode cmode = pmbp->pz4c->opt.cartoon;
  const bool cartoon = pmbp->pz4c->IsCartoon();
  const RegionIndcs indcs_ = indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1, nji = nx2*nx1;
  const int nmkji = pmbp->nmb_thispack*nkji;
  array_sum::GlobalSum sum;
  Kokkos::parallel_reduce("GaugePulseHist",
                          Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &s) {
    const int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    const int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    const auto &sz = size.d_view(m);
    Real dv;
    if (cartoon) {
      Real X[3];
      z4c::Z4cCartesianPos(cmode, sz, indcs_, k, j, i, X[0], X[1], X[2]);
      dv = 2.0*M_PI*X[0]*sz.dx1*sz.dx2;
    } else {
      dv = sz.dx1*sz.dx2*sz.dx3;
    }
    array_sum::GlobalSum h;
    for (int n = 0; n < NHISTORY_VARIABLES; ++n) h.the_array[n] = 0.0;
    const Real da = z4c.alpha(m,k,j,i) - 1.0;
    h.the_array[0] = dv*da*da;
    h.the_array[1] = dv*SQR(z4c.vKhat(m,k,j,i));
    h.the_array[2] = dv*SQR(z4c.vTheta(m,k,j,i));
    Real g2 = 0.0;
    for (int a = 0; a < 3; ++a) g2 += SQR(z4c.vGam_u(m,a,k,j,i));
    h.the_array[3] = dv*g2;
    h.the_array[4] = dv;
    s += h;
  }, Kokkos::Sum<array_sum::GlobalSum>(sum));
  for (int n = 0; n < pdata->nhist; ++n) pdata->hdata[n] = sum.the_array[n];
}

}  // namespace
