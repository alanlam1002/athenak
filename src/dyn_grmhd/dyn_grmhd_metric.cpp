//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyn_grmhd_metric.cpp
//! \brief <dyn_grmhd> metric_terms = analytic_bg | wellbalanced (see metric_faces.hpp):
//! the trumpet table, the per-stage face metric, its ghost/owner probe, and the coordinate
//! source terms that read it (AddCoordTermsFM; the cell scheme's AddCoordTermsEOS is
//! untouched).

#include <math.h>

#include <cstdio>
#include <iostream>
#include <iomanip>
#include <random>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "cfc/cfc.hpp"
#include "cfc/cfc_puncture.hpp"
#include "mhd/mhd.hpp"
#include "utils/finite_diff.hpp"
#include "dyn_grmhd.hpp"
#include "metric_faces.hpp"

#include "eos/primitive_solver_hyd.hpp"
#include "eos/primitive-solver/idealgas.hpp"
#include "eos/primitive-solver/eos_compose.hpp"
#include "eos/primitive-solver/eos_hybrid.hpp"
#include "eos/primitive-solver/piecewise_polytrope.hpp"
#include "eos/primitive-solver/reset_floor.hpp"

namespace dyngr {

namespace {

// chi(q) = r_iso/M at rho = 3/2 + q (cfc_puncture.hpp TrumpetArealToIso, written in q so
// the throat keeps full precision), and d ln chi/d ln q = 2 rho/s
long double LnChi(long double q) {
  const long double rho = 1.5L + q;
  const long double s1 = sqrtl(4.0L*rho*rho + 4.0L*rho + 3.0L);
  const long double s2 = sqrtl(2.0L)*s1;
  const long double numer = (4.0L + 3.0L*sqrtl(2.0L))*2.0L*q;
  const long double denom = 8.0L*rho + 6.0L + 3.0L*s2;
  return logl(0.25L*(2.0L*rho + 1.0L + s1)) + logl(numer/denom)/sqrtl(2.0L);
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void BuildTrumpetTable
//! \brief nodes w_k = k h, x = w^2 = r_iso/M: q = rho - 3/2 by Newton in v = ln q on
//! ln chi(e^v) = ln x (monotone, d ln chi/dv = 2 rho/s), in long double; then rho, rho_w and
//! rho_ww in closed form (metric_faces.hpp). The w = 0 node is the throat (rho = 3/2 with
//! vanishing derivatives, rho - 3/2 ~ w^(2 sqrt2)).
void BuildTrumpetTable(TrumpetTable &t, Real w_max, Real h) {
  t.n = static_cast<int>(std::ceil(w_max/h)) + 1;
  t.h = h; t.ih = 1.0/h;
  Kokkos::realloc(t.d, t.n, 3);
  auto hd = Kokkos::create_mirror_view(t.d);
  hd(0, 0) = 1.5; hd(0, 1) = 0.0; hd(0, 2) = 0.0;
  for (int k = 1; k < t.n; ++k) {
    const long double w = k*static_cast<long double>(h), x = w*w, u = logl(x);
    long double v = (x < 1.0L) ? u*sqrtl(2.0L) : logl(std::max(x - 1.0L, 1.0e-30L));
    for (int it = 0; it < 200; ++it) {
      const long double q = expl(v), rho = 1.5L + q;
      const long double s = sqrtl(4.0L*rho*rho + 4.0L*rho + 3.0L);
      const long double dv = (LnChi(q) - u)*s/(2.0L*rho);
      v -= dv;
      if (fabsl(dv) < 1.0e-19L*(1.0L + fabsl(v))) break;
    }
    const long double q = expl(v), rho = 1.5L + q;
    const long double s = sqrtl(4.0L*rho*rho + 4.0L*rho + 3.0L);
    const long double al = 2.0L*q*s/(4.0L*rho*rho);
    const long double rw = 2.0L*rho*al/w;
    const long double aw = 2.0L*(1.0L - 3.375L/(rho*rho*rho))/(rho*w);
    const long double rww = 2.0L*((rw*al + rho*aw)/w - rho*al/(w*w));
    hd(k, 0) = static_cast<Real>(rho);
    hd(k, 1) = static_cast<Real>(rw);
    hd(k, 2) = static_cast<Real>(rww);
  }
  Kokkos::deep_copy(t.d, hd);
}

//----------------------------------------------------------------------------------------
//! \fn void DynGRMHD::TrumpetTableTest
//! \brief <dyn_grmhd> metric_table_test: psi0, (alpha psi)0, |beta0| and the radial
//! derivatives through the table against the direct fill (TrumpetIsoToAreal +
//! TrumpetBackground, as cfc_puncture.cpp), at npts log-uniform radii in [1e-3, 1e4] M, on
//! the device and on the host. Prints MTABLE_TEST lines (max relative differences, overall
//! and where alpha0 > 0.01).
void DynGRMHD::TrumpetTableTest(int npts) {
  const Real m_bh = (pmy_pack->pcfc != nullptr) ? pmy_pack->pcfc->puncture_mass_ : 1.0;
  DvceArray2D<Real> pts("mtab_pts", npts, 3), out("mtab_out", npts, 14);
  auto hp = Kokkos::create_mirror_view(pts);
  std::mt19937_64 gen(12345);
  std::uniform_real_distribution<double> ul(std::log(1.0e-3), std::log(1.0e4));
  std::normal_distribution<double> nd(0.0, 1.0);
  for (int p = 0; p < npts; ++p) {
    double a = nd(gen), b = nd(gen), c = nd(gen);
    const double nr = std::sqrt(a*a + b*b + c*c), r = m_bh*std::exp(ul(gen));
    hp(p, 0) = r*a/nr; hp(p, 1) = r*b/nr; hp(p, 2) = r*c/nr;
  }
  Kokkos::deep_copy(pts, hp);
  auto &td = ttab.d;
  const Real th = ttab.h, tih = ttab.ih;
  const int n = ttab.n;
  auto eval = [=] KOKKOS_FUNCTION (const Real x1, const Real x2, const Real x3, Real o[14]) {
    Real psi0, ap0, b0[3], dpsi0, dal0, b, db;
    TrumpetTableBackground(td, th, tih, n, m_bh, x1, x2, x3, psi0, ap0, b0, dpsi0,
                           dal0, b, db);
    const Real r = sqrt(x1*x1 + x2*x2 + x3*x3);
    const Real rs = m_bh*cfc::TrumpetIsoToAreal(r/m_bh);
    Real p1, a1, bb[3], A[6], a2, dp1, da1;
    cfc::TrumpetBackground(m_bh, x1, x2, x3, rs, &p1, &a1, bb, A, &a2, &dp1, &da1);
    o[0] = psi0; o[1] = ap0; o[2] = b*r; o[3] = dpsi0; o[4] = dal0;
    o[5] = p1; o[6] = a1*p1; o[7] = sqrt(bb[0]*bb[0] + bb[1]*bb[1] + bb[2]*bb[2]);
    o[8] = dp1; o[9] = da1; o[10] = a1;
    o[11] = sqrt(b0[0]*b0[0] + b0[1]*b0[1] + b0[2]*b0[2]); o[12] = r; o[13] = db;
  };
  Kokkos::parallel_for("mtab_test", Kokkos::RangePolicy<>(DevExeSpace(), 0, npts),
  KOKKOS_LAMBDA(const int p) {
    Real o[14];
    eval(pts(p, 0), pts(p, 1), pts(p, 2), o);
    for (int q = 0; q < 14; ++q) out(p, q) = o[q];
  });
  auto ho = Kokkos::create_mirror_view_and_copy(HostMemSpace(), out);
  // the host path: the same functions on the host copy of the table
  auto htd = Kokkos::create_mirror_view_and_copy(HostMemSpace(), ttab.d);
  const char *nm[5] = {"psi0", "alpha0psi0", "beta0", "dpsi0", "dalpha0"};
  for (int side = 0; side < 2; ++side) {
    Real emax[5] = {0.0}, eok[5] = {0.0};
    for (int p = 0; p < npts; ++p) {
      Real o[14];
      if (side == 0) {
        for (int q = 0; q < 14; ++q) o[q] = ho(p, q);
      } else {
        Real psi0, ap0, b0[3], dpsi0, dal0, b, db;
        const Real x1 = hp(p, 0), x2 = hp(p, 1), x3 = hp(p, 2);
        const Real r = std::sqrt(x1*x1 + x2*x2 + x3*x3 + 1.0e-30);
        const Real rho = TrumpetTableRho(htd, th, tih, n, r/m_bh);
        Real al0, A[6], a2;
        cfc::TrumpetBackground(m_bh, x1, x2, x3, m_bh*rho, &psi0, &al0, b0, A, &a2, &dpsi0,
                               &dal0);
        b = 0.75*std::sqrt(3.0)/(m_bh*rho*rho*rho);
        const Real rr = std::sqrt(x1*x1 + x2*x2 + x3*x3);
        const Real rs = m_bh*cfc::TrumpetIsoToAreal(rr/m_bh);
        Real p1, a1, bb[3], AA[6], a22, dp1, da1;
        cfc::TrumpetBackground(m_bh, x1, x2, x3, rs, &p1, &a1, bb, AA, &a22, &dp1, &da1);
        o[0] = psi0; o[1] = al0*psi0; o[2] = b*rr; o[3] = dpsi0; o[4] = dal0;
        o[5] = p1; o[6] = a1*p1; o[7] = std::sqrt(bb[0]*bb[0] + bb[1]*bb[1] + bb[2]*bb[2]);
        o[8] = dp1; o[9] = da1; o[10] = a1;
      }
      for (int q = 0; q < 5; ++q) {
        const Real e = std::fabs(o[q] - o[5 + q])/std::fmax(std::fabs(o[5 + q]), 1.0e-300);
        emax[q] = std::fmax(emax[q], e);
        if (o[10] > 0.01) eok[q] = std::fmax(eok[q], e);
      }
    }
    if (global_variable::my_rank == 0) {
      std::cout << std::setprecision(4) << "MTABLE_TEST " << (side == 0 ? "device" : "host")
                << " npts=" << npts << " n=" << n << " h=" << th;
      for (int q = 0; q < 5; ++q) {
        std::cout << " " << nm[q] << "=" << emax[q] << "/" << eok[q];
      }
      std::cout << "   (max rel diff: all / alpha0 > 0.01)" << std::endl;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void DynGRMHD::ComputeFaceMetric
//! \brief (1) the residual at every cell (ghosts included) from the assembled, exchanged
//! adm metric and the CFC background arrays: delta psi = psi4^(1/4) - psi0, delta(alpha psi)
//! = alpha psi - (alpha psi)0, beta_res = beta - beta0 - xidot (padm->frame_vel_u);
//! (2) at every face whose interpolation stencil lies in the block, the total metric
//! psi_f = psi0(x_f) + I[delta psi], (alpha psi)_f = (alpha psi)0(x_f) + I[delta(alpha psi)],
//! alpha_f = (alpha psi)_f/psi_f with the CFC alpha floor (cfc_reconstruct.cpp
//! AssembleLapseShiftK), beta_f = beta0(x_f) + I[beta_res] + xidot. x_f is relative to the
//! centre of the last background fill (CFC::bg_xbh_). Every input is owner-consistent at
//! same-level faces (exchanged adm, the pointwise background, the global xidot).
void DynGRMHD::ComputeFaceMetric(int stage) {
  if (metric_terms == MetricTerms::kerr_schild) {
    ComputeFaceMetricKS();
    if (fm_probe_cycles > 0) FaceMetricProbe(stage);
    return;
  }
  auto *pcfc = pmy_pack->pcfc;
  if (pcfc == nullptr || !pcfc->puncture_enabled_) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<dyn_grmhd> metric_terms needs CFC with the puncture background" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int n1 = fm_res.extent_int(4), n2 = fm_res.extent_int(3), n3 = fm_res.extent_int(2);
  const int nmb = pmy_pack->nmb_thispack;
  auto &size = pmy_pack->pmb->mb_size;
  auto &adm = pmy_pack->padm->adm;
  auto &psi0c = pcfc->u_psi0;
  auto &ap0c = pcfc->u_alpha0_psi0;
  auto &b0c = pcfc->beta0_u;
  const Real fv0 = pmy_pack->padm->frame_vel_u[0], fv1 = pmy_pack->padm->frame_vel_u[1];
  const Real fv2 = pmy_pack->padm->frame_vel_u[2];
  const Real cx = pcfc->bg_xbh_[0], cy = pcfc->bg_xbh_[1], cz = pcfc->bg_xbh_[2];
  const Real m_bh = pcfc->puncture_mass_;
  const Real afl = pcfc->alpha_floor_;
  auto &res = fm_res;
  par_for("fm_residual", DevExeSpace(), 0, nmb-1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real p = sqrt(sqrt(adm.psi4(m,k,j,i)));
    res(m,0,k,j,i) = p - psi0c(m,0,k,j,i);
    res(m,1,k,j,i) = adm.alpha(m,k,j,i)*p - ap0c(m,0,k,j,i);
    res(m,2,k,j,i) = adm.beta_u(m,0,k,j,i) - b0c(m,0,k,j,i) - fv0;
    res(m,3,k,j,i) = adm.beta_u(m,1,k,j,i) - b0c(m,1,k,j,i) - fv1;
    res(m,4,k,j,i) = adm.beta_u(m,2,k,j,i) - b0c(m,2,k,j,i) - fv2;
  });
  auto &td = ttab.d;
  const Real th = ttab.h, tih = ttab.ih;
  const int nt = ttab.n;
  const bool o4 = (face_residual_interp == 4);
  const int lo = o4 ? 2 : 1;      // first face whose stencil (i-2..i+1 or i-1..i) fits
  for (int dir = 0; dir < 3; ++dir) {
    if ((dir == 1 && nx2 == 1) || (dir == 2 && nx3 == 1)) continue;
    auto &fm = (dir == 0) ? fm_x1 : ((dir == 1) ? fm_x2 : fm_x3);
    const int il = (dir == 0) ? lo : 0, iu = (dir == 0) ? n1 - lo : n1 - 1;
    const int jl = (dir == 1) ? lo : 0, ju = (dir == 1) ? n2 - lo : n2 - 1;
    const int kl = (dir == 2) ? lo : 0, ku = (dir == 2) ? n3 - lo : n3 - 1;
    const int si = (dir == 0), sj = (dir == 1), sk = (dir == 2);
    par_for("fm_faces", DevExeSpace(), 0, nmb-1, kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      const auto &sz = size.d_view(m);
      const Real x = (si ? LeftEdgeX(i-is, nx1, sz.x1min, sz.x1max)
                         : CellCenterX(i-is, nx1, sz.x1min, sz.x1max)) - cx;
      const Real y = (sj ? LeftEdgeX(j-js, nx2, sz.x2min, sz.x2max)
                         : CellCenterX(j-js, nx2, sz.x2min, sz.x2max)) - cy;
      const Real z = (sk ? LeftEdgeX(k-ks, nx3, sz.x3min, sz.x3max)
                         : CellCenterX(k-ks, nx3, sz.x3min, sz.x3max)) - cz;
      Real r5[5];
      for (int c = 0; c < 5; ++c) {
        const Real a = res(m, c, k - sk, j - sj, i - si), b = res(m, c, k, j, i);
        if (o4) {
          const Real a2 = res(m, c, k - 2*sk, j - 2*sj, i - 2*si);
          const Real b2 = res(m, c, k + sk, j + sj, i + si);
          r5[c] = (9.0*(a + b) - (a2 + b2))*0.0625;
        } else {
          r5[c] = 0.5*(a + b);
        }
      }
      Real psi0, ap0, b0[3], dpsi0, dal0, bb, db;
      TrumpetTableBackground(td, th, tih, nt, m_bh, x, y, z, psi0, ap0, b0, dpsi0,
                             dal0, bb, db);
      const Real psif = psi0 + r5[0];
      const Real alf = (ap0 + r5[1])/psif;
      fm(m, kFmAlpha, k, j, i) = (alf < afl) ? afl : alf;
      fm(m, kFmBeta + 0, k, j, i) = b0[0] + r5[2] + fv0;
      fm(m, kFmBeta + 1, k, j, i) = b0[1] + r5[3] + fv1;
      fm(m, kFmBeta + 2, k, j, i) = b0[2] + r5[4] + fv2;
      fm(m, kFmPsi, k, j, i) = psif;
      const Real p4 = (psif*psif)*(psif*psif);   // gamma_ij = psi^4 delta_ij (metric_faces.hpp)
      fm(m, kFmG + 0, k, j, i) = p4; fm(m, kFmG + 1, k, j, i) = 0.0;
      fm(m, kFmG + 2, k, j, i) = 0.0; fm(m, kFmG + 3, k, j, i) = p4;
      fm(m, kFmG + 4, k, j, i) = 0.0; fm(m, kFmG + 5, k, j, i) = p4;
    });
  }
  if (fm_probe_cycles > 0) FaceMetricProbe(stage);
}

//----------------------------------------------------------------------------------------
//! \fn void DynGRMHD::ComputeFaceMetricKS
//! \brief metric_terms = kerr_schild (R-188 item 3): every face of the block (ghost faces
//! included) from the closed form the cell fill uses (ComputeADMDecomposition, cartesian_ks.hpp;
//! M = 1, <coord> a), pointwise in the face coordinates: a ghost's face equals its owner's
void DynGRMHD::ComputeFaceMetricKS() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmb = pmy_pack->nmb_thispack;
  auto &size = pmy_pack->pmb->mb_size;
  const Real a = pmy_pack->pcoord->coord_data.bh_spin;
  const bool mink = pmy_pack->pcoord->coord_data.is_minkowski;
  for (int dir = 0; dir < 3; ++dir) {
    if ((dir == 1 && nx2 == 1) || (dir == 2 && nx3 == 1)) continue;
    auto &fm = (dir == 0) ? fm_x1 : ((dir == 1) ? fm_x2 : fm_x3);
    const int si = (dir == 0), sj = (dir == 1), sk = (dir == 2);
    par_for("fm_faces_ks", DevExeSpace(), 0, nmb-1, 0, fm.extent_int(2)-1, 0, fm.extent_int(3)-1,
            0, fm.extent_int(4)-1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      const auto &sz = size.d_view(m);
      const Real x = si ? LeftEdgeX(i-is, nx1, sz.x1min, sz.x1max)
                        : CellCenterX(i-is, nx1, sz.x1min, sz.x1max);
      const Real y = sj ? LeftEdgeX(j-js, nx2, sz.x2min, sz.x2max)
                        : CellCenterX(j-js, nx2, sz.x2min, sz.x2max);
      const Real z = sk ? LeftEdgeX(k-ks, nx3, sz.x3min, sz.x3max)
                        : CellCenterX(k-ks, nx3, sz.x3min, sz.x3max);
      Real al, b[3], p4, g[6], K[6];
      ComputeADMDecomposition(x, y, z, mink, a, &al, &b[0], &b[1], &b[2], &p4,
                              &g[0], &g[1], &g[2], &g[3], &g[4], &g[5],
                              &K[0], &K[1], &K[2], &K[3], &K[4], &K[5]);
      fm(m, kFmAlpha, k, j, i) = al;
      for (int c = 0; c < 3; ++c) fm(m, kFmBeta + c, k, j, i) = b[c];
      fm(m, kFmPsi, k, j, i) = sqrt(sqrt(p4));
      for (int c = 0; c < 6; ++c) fm(m, kFmG + c, k, j, i) = g[c];
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void DynGRMHD::FaceMetricProbe
//! \brief <dyn_grmhd> face_probe_cycles / face_probe_box (xmin xmax ymin ymax zmin zmax):
//! every computed face inside the box, in every MeshBlock that computes it (its own faces
//! and its ghost-region faces), one record per stage: FPROBE cycle stage dir gid i j k level
//! x y z alpha beta^x beta^y beta^z psi (%.17g), to fprobe_rank<NNNNN>.txt. Same-level
//! copies of one physical face must agree bitwise (scripts/fm_probe_check.py).
void DynGRMHD::FaceMetricProbe(int stage) {
  const int cyc = pmy_pack->pmesh->ncycle;
  if (fm_probe_c0 < 0) fm_probe_c0 = cyc;
  if (cyc >= fm_probe_c0 + fm_probe_cycles) return;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmb = pmy_pack->nmb_thispack, gid0 = pmy_pack->gids;
  const int n1 = fm_res.extent_int(4), n2 = fm_res.extent_int(3), n3 = fm_res.extent_int(2);
  auto &size = pmy_pack->pmb->mb_size;
  auto &lev = pmy_pack->pmb->mb_lev;
  Kokkos::Array<Real, 6> pb;
  for (int q = 0; q < 6; ++q) pb[q] = fm_probe_box[q];
  const int nrec = 15, cap = 2000000;
  DvceArray2D<Real> buf("fprobe_buf", cap, nrec);
  DvceArray1D<int> cnt("fprobe_cnt", 1);
  const int lo = (face_residual_interp == 4) ? 2 : 1;
  for (int dir = 0; dir < 3; ++dir) {
    if ((dir == 1 && nx2 == 1) || (dir == 2 && nx3 == 1)) continue;
    auto &fm = (dir == 0) ? fm_x1 : ((dir == 1) ? fm_x2 : fm_x3);
    const int il = (dir == 0) ? lo : 0, iu = (dir == 0) ? n1 - lo : n1 - 1;
    const int jl = (dir == 1) ? lo : 0, ju = (dir == 1) ? n2 - lo : n2 - 1;
    const int kl = (dir == 2) ? lo : 0, ku = (dir == 2) ? n3 - lo : n3 - 1;
    const int si = (dir == 0), sj = (dir == 1), sk = (dir == 2);
    par_for("fm_probe", DevExeSpace(), 0, nmb-1, kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      const auto &sz = size.d_view(m);
      const Real x = si ? LeftEdgeX(i-is, nx1, sz.x1min, sz.x1max)
                        : CellCenterX(i-is, nx1, sz.x1min, sz.x1max);
      const Real y = sj ? LeftEdgeX(j-js, nx2, sz.x2min, sz.x2max)
                        : CellCenterX(j-js, nx2, sz.x2min, sz.x2max);
      const Real z = sk ? LeftEdgeX(k-ks, nx3, sz.x3min, sz.x3max)
                        : CellCenterX(k-ks, nx3, sz.x3min, sz.x3max);
      if (!(x > pb[0] && x < pb[1] && y > pb[2] && y < pb[3] && z > pb[4] && z < pb[5])) {
        return;
      }
      const int r = Kokkos::atomic_fetch_add(&cnt(0), 1);
      if (r >= cap) return;
      Real v[15] = {static_cast<Real>(dir), static_cast<Real>(gid0 + m),
                    static_cast<Real>(i), static_cast<Real>(j), static_cast<Real>(k),
                    static_cast<Real>(lev.d_view(m)), x, y, z,
                    fm(m, 0, k, j, i), fm(m, 1, k, j, i), fm(m, 2, k, j, i),
                    fm(m, 3, k, j, i), fm(m, 4, k, j, i), 0.0};
      for (int q = 0; q < 15; ++q) buf(r, q) = v[q];
    });
  }
  // the residual itself at every cell copy in the box (dir = -1; i j k the cell), so a face
  // mismatch can be traced to its inputs
  {
    auto &res = fm_res;
    par_for("fm_probe_res", DevExeSpace(), 0, nmb-1, 0, n3-1, 0, n2-1, 0, n1-1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      const auto &sz = size.d_view(m);
      const Real x = CellCenterX(i-is, nx1, sz.x1min, sz.x1max);
      const Real y = CellCenterX(j-js, nx2, sz.x2min, sz.x2max);
      const Real z = CellCenterX(k-ks, nx3, sz.x3min, sz.x3max);
      if (!(x > pb[0] && x < pb[1] && y > pb[2] && y < pb[3] && z > pb[4] && z < pb[5])) {
        return;
      }
      const int r = Kokkos::atomic_fetch_add(&cnt(0), 1);
      if (r >= cap) return;
      Real v[15] = {-1.0, static_cast<Real>(gid0 + m), static_cast<Real>(i),
                    static_cast<Real>(j), static_cast<Real>(k),
                    static_cast<Real>(lev.d_view(m)), x, y, z,
                    res(m, 0, k, j, i), res(m, 1, k, j, i), res(m, 2, k, j, i),
                    res(m, 3, k, j, i), res(m, 4, k, j, i), 0.0};
      for (int q = 0; q < 15; ++q) buf(r, q) = v[q];
    });
  }
  auto hc = Kokkos::create_mirror_view_and_copy(HostMemSpace(), cnt);
  const int nr = (hc(0) < cap) ? hc(0) : cap;
  auto hb = Kokkos::create_mirror_view_and_copy(HostMemSpace(), buf);
  char fn[64];
  snprintf(fn, sizeof(fn), "fprobe_rank%05d.txt", global_variable::my_rank);
  FILE *fp = fopen(fn, "a");
  if (fp == nullptr) return;
  for (int r = 0; r < nr; ++r) {
    fprintf(fp, "FPROBE %d %d", cyc, stage);
    for (int q = 0; q < 6; ++q) fprintf(fp, " %d", static_cast<int>(hb(r, q)));
    for (int q = 6; q < 14; ++q) fprintf(fp, " %.17g", hb(r, q));
    fprintf(fp, "\n");
  }
  if (hc(0) > cap) fprintf(fp, "FPROBE_OVERFLOW %d %d %d\n", cyc, stage, hc(0));
  fclose(fp);
}

//----------------------------------------------------------------------------------------
//! \fn void DynGRMHDPS::AddCoordTermsFM
//! \brief AddCoordTermsEOS (dyn_grmhd.cpp) with the metric derivatives replaced:
//!   wellbalanced: d_d alpha, d_d beta^a, d_d gamma_bc = differences of the face arrays
//!     the fluxes used, (f(i+1/2) - f(i-1/2))/dx_d, gamma_f = psi_f^4 delta;
//!   analytic_bg: d psi = dpsi0/dr x^/r + Dx(delta psi), d(alpha psi) likewise,
//!     d alpha = (d(alpha psi) - alpha d psi)/psi, d_a beta^b = b delta_ab + db x_a x^b/r
//!     + Dx_a(beta_res^b), d gamma = 4 psi^3 d psi delta.
//! Everything undifferentiated (alpha, gamma, vol, K_ij, S, E) stays cell-centred, as in the
//! cell scheme; the dual-energy diagnostics and the smooth-excision damping are the same.
template<class EOSPolicy, class ErrorPolicy> template<int NGHOST>
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddCoordTermsFM(const DvceArray5D<Real> &prim,
    const DvceArray5D<Real> &bcc,
    const Real dt, DvceArray5D<Real> &rhs) {
  if (fixed_evolution) {
    return;
  }
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size  = pmy_pack->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;

  int nmb = pmy_pack->nmb_thispack;

  auto &adm = pmy_pack->padm->adm;
  auto &eos_ = eos.ps.GetEOS();

  bool smoothing = pmy_pack->pcoord->coord_data.smooth_excision;
  auto &floor = pmy_pack->pcoord->excision_floor;
  Real &dexcise = pmy_pack->pcoord->coord_data.dexcise;
  Real &texcise = pmy_pack->pcoord->coord_data.texcise;
  Real &tdamp = pmy_pack->pcoord->coord_data.tdamp;

  int &nhyd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;

  const Real mb = eos.ps.GetEOS().GetBaryonMass();
  const int imap[3][3] = {
    {S11, S12, S13},
    {S12, S22, S23},
    {S13, S23, S33}
  };

  int ndim;
  if (pmy_pack->pmesh->one_d) {
    ndim = 1;
  } else if (pmy_pack->pmesh->two_d) {
    ndim = 2;
  } else {
    ndim = 3;
  }

  const bool wb = (metric_terms == MetricTerms::wellbalanced);
  const bool mfd = kill_mom_fd;
  const bool kks = (metric_terms == MetricTerms::kerr_schild);   // R-188 item 3: full gamma_ij
  auto &fm1 = fm_x1;
  auto &fm2 = fm_x2;
  auto &fm3 = fm_x3;
  auto &res = fm_res;
  auto &td = ttab.d;
  const Real th = ttab.h, tih = ttab.ih;
  const int nt = ttab.n;
  auto *pcfc = pmy_pack->pcfc;
  const Real cx = kks ? 0.0 : pcfc->bg_xbh_[0], cy = kks ? 0.0 : pcfc->bg_xbh_[1];
  const Real cz = kks ? 0.0 : pcfc->bg_xbh_[2], m_bh = kks ? 1.0 : pcfc->puncture_mass_;

  const bool ddon_ = eos.de_diag_on;
  auto &dd_ = de_diag;
  auto &fx1_ = pmy_pack->pmhd->uflx.x1f;
  auto &fx2_ = pmy_pack->pmhd->uflx.x2f;
  auto &fx3_ = pmy_pack->pmhd->uflx.x3f;
  const bool md_ = pmy_pack->pmesh->multi_d, d3_ = pmy_pack->pmesh->three_d;
  par_for("coord_src_fm", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real g3d[NSPMETRIC] = {adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
                           adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
                           adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i)};
    const Real& alpha = adm.alpha(m, k, j, i);
    Real detg = adm::SpatialDet(g3d[S11], g3d[S12], g3d[S13],
                                g3d[S22], g3d[S23], g3d[S33]);
    Real vol = sqrt(detg);
    Real g3u[NSPMETRIC] = {0.};
    adm::SpatialInv(1.0/detg, g3d[S11], g3d[S12], g3d[S13], g3d[S22], g3d[S23], g3d[S33],
                    &g3u[S11], &g3u[S12], &g3u[S13], &g3u[S22], &g3u[S23], &g3u[S33]);

    Real idx[] = {1./size.d_view(m).dx1, 1./size.d_view(m).dx2, 1./size.d_view(m).dx3};
    Real dalpha_d[3] = {0.};
    Real dbeta_du[3][3] = {};
    Real dg_ddd[3][3][3] = {};
    // the momentum rows' (killing_momentum_bg = face_diff: their own, below; else these)
    Real dam[3] = {0.}, dbm[3][3] = {}, dgm[3][3][3] = {};
    if (kks) {   // the face values differenced, every gamma_ij (as wellbalanced)
      const int gmap[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
      for (int a = 0; a < ndim; a++) {
        const auto &fm = (a == 0) ? fm1 : ((a == 1) ? fm2 : fm3);
        const int ip = i + (a == 0), jp = j + (a == 1), kp = k + (a == 2);
        dalpha_d[a] = (fm(m, kFmAlpha, kp, jp, ip) - fm(m, kFmAlpha, k, j, i))*idx[a];
        for (int b = 0; b < 3; b++) {
          dbeta_du[a][b] = (fm(m, kFmBeta + b, kp, jp, ip) - fm(m, kFmBeta + b, k, j, i))*
                           idx[a];
          for (int c = 0; c < 3; c++) {
            const int q = kFmG + gmap[b][c];
            dg_ddd[a][b][c] = (fm(m, q, kp, jp, ip) - fm(m, q, k, j, i))*idx[a];
          }
        }
      }
    } else if (wb) {
      for (int a = 0; a < ndim; a++) {
        const auto &fm = (a == 0) ? fm1 : ((a == 1) ? fm2 : fm3);
        const int ip = i + (a == 0), jp = j + (a == 1), kp = k + (a == 2);
        dalpha_d[a] = (fm(m, kFmAlpha, kp, jp, ip) - fm(m, kFmAlpha, k, j, i))*idx[a];
        for (int b = 0; b < 3; b++) {
          dbeta_du[a][b] = (fm(m, kFmBeta + b, kp, jp, ip) - fm(m, kFmBeta + b, k, j, i))*
                           idx[a];
        }
        const Real pp = fm(m, kFmPsi, kp, jp, ip), pm = fm(m, kFmPsi, k, j, i);
        const Real dg = ((pp*pp)*(pp*pp) - (pm*pm)*(pm*pm))*idx[a];
        for (int b = 0; b < 3; b++) dg_ddd[a][b][b] = dg;
      }
    } else {
      const auto &sz = size.d_view(m);
      const Real x = CellCenterX(i-is, nx1, sz.x1min, sz.x1max) - cx;
      const Real y = CellCenterX(j-js, nx2, sz.x2min, sz.x2max) - cy;
      const Real z = CellCenterX(k-ks, nx3, sz.x3min, sz.x3max) - cz;
      Real psi0, ap0, b0[3], dpsi0, dal0, bb, db;
      TrumpetTableBackground(td, th, tih, nt, m_bh, x, y, z, psi0, ap0, b0, dpsi0,
                             dal0, bb, db);
      const Real r = sqrt(x*x + y*y + z*z + 1.0e-30);
      const Real xh[3] = {x/r, y/r, z/r}, xv[3] = {x, y, z};
      const Real al0 = ap0/psi0;
      const Real p = sqrt(sqrt(adm.psi4(m,k,j,i)));
      const Real p3 = p*p*p;
      for (int a = 0; a < ndim; a++) {
        const Real dpsi = dpsi0*xh[a] + Dx<NGHOST>(a, idx, res, m, 0, k, j, i);
        const Real dap = (dal0*psi0 + al0*dpsi0)*xh[a] + Dx<NGHOST>(a, idx, res, m, 1, k, j, i);
        dalpha_d[a] = (dap - alpha*dpsi)/p;
        for (int b = 0; b < 3; b++) {
          dbeta_du[a][b] = ((a == b) ? bb : 0.0) + db*xh[a]*xv[b] +
                           Dx<NGHOST>(a, idx, res, m, 2 + b, k, j, i);
          dg_ddd[a][b][b] = 4.0*p3*dpsi;
        }
      }
      if (mfd) {
        // R-178: the background part of the momentum rows' d alpha, d beta, d gamma -> the
        // centred differences of killing_t1's face-weight values (TrumpetFaceDiff); the
        // residual part as above (d alpha: the full minus alpha0' x^/r; d gamma: minus
        // 4 psi0^3 psi0' x^/r); the energy row keeps dalpha_d
        const int ic[3] = {i - is, j - js, k - ks}, nc[3] = {nx1, nx2, nx3};
        const Real lo[3] = {sz.x1min, sz.x2min, sz.x3min};
        const Real hi[3] = {sz.x1max, sz.x2max, sz.x3max};
        const Real c0[3] = {cx, cy, cz};
        const Real p03 = psi0*psi0*psi0;
        for (int a = 0; a < ndim; a++) {
          Real xm[3] = {x, y, z}, xp[3] = {x, y, z};
          xm[a] = LeftEdgeX(ic[a], nc[a], lo[a], hi[a]) - c0[a];
          xp[a] = LeftEdgeX(ic[a] + 1, nc[a], lo[a], hi[a]) - c0[a];
          Real fa, fb[3], fg;
          TrumpetFaceDiff(td, th, tih, nt, m_bh, xm, xp, idx[a], fa, fb, fg);
          dam[a] = dalpha_d[a] - dal0*xh[a] + fa;
          for (int b = 0; b < 3; b++) {
            dbm[a][b] = fb[b] + Dx<NGHOST>(a, idx, res, m, 2 + b, k, j, i);
            dgm[a][b][b] = dg_ddd[a][b][b] - 4.0*p03*dpsi0*xh[a] + fg;
          }
        }
      }
    }
    if (!mfd) {
      for (int a = 0; a < 3; a++) {
        dam[a] = dalpha_d[a];
        for (int b = 0; b < 3; b++) {
          dbm[a][b] = dbeta_du[a][b];
          for (int c = 0; c < 3; c++) dgm[a][b][c] = dg_ddd[a][b][c];
        }
      }
    }

    // Fluid quantities (as AddCoordTermsEOS)
    Real prim_pt[NPRIM] = {0.0};
    prim_pt[PRH] = prim(m, IDN, k, j, i)/mb;
    prim_pt[PVX] = prim(m, IVX, k, j, i);
    prim_pt[PVY] = prim(m, IVY, k, j, i);
    prim_pt[PVZ] = prim(m, IVZ, k, j, i);
    for (int s = 0; s < nscal; s++) {
      prim_pt[PYF + s] = prim(m, nhyd + s, k, j, i);
    }
    prim_pt[PPR] = prim(m, IPR, k, j, i);
    prim_pt[PTM] = eos_.GetTemperatureFromP(prim_pt[PRH], prim_pt[PPR], &prim_pt[PYF]);

    Real H =
      prim(m, IDN, k, j, i)*eos_.GetEnthalpy(prim_pt[PRH], prim_pt[PTM], &prim_pt[PYF]);
    Real usq = Primitive::SquareVector(&prim_pt[PVX], g3d);
    Real const Wsq = 1.0 + usq;
    Real const W = sqrt(Wsq);
    Real B_u[NMAG] = {bcc(m, IBX, k, j, i)/vol,
                      bcc(m, IBY, k, j, i)/vol,
                      bcc(m, IBZ, k, j, i)/vol};
    Real Bv = 0.0;
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3; b++) {
        Bv += adm.g_dd(m,a,b,k,j,i)*prim_pt[PVX + a]*B_u[b];
      }
    }
    Real Bsq = Primitive::SquareVector(B_u, g3d);
    Bv = Bv/W;
    Real bsq = Bv*Bv + Bsq/Wsq;

    Real E = (H*Wsq + Bsq) - prim_pt[PPR] - 0.5*bsq;

    Real S_d[3] = {0.0};
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3; b++) {
        S_d[a] += ((H*Wsq + Bsq)*prim_pt[PVX + b]/W - Bv*B_u[b])*g3d[imap[a][b]];
      }
    }

    Real S_uu[3][3];
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b <= a; b++) {
        S_uu[a][b] = (H + Bsq/Wsq)*prim_pt[PVX + a]*prim_pt[PVX + b]
                      - B_u[a]*B_u[b]/Wsq
                      - Bv*(B_u[a]*prim_pt[PVX + b] + B_u[b]*prim_pt[PVX + a])/W
                      + (prim_pt[PPR] + 0.5*bsq)*g3u[imap[a][b]];
        S_uu[b][a] = S_uu[a][b];
      }
    }

    // Assemble energy RHS
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3; b++) {
        rhs(m, IEN, k, j, i) += dt*vol*(alpha*adm.vK_dd(m, a, b, k, j, i)*S_uu[a][b] -
            g3u[imap[a][b]] * S_d[a]*dalpha_d[b]);
      }
    }
    if (ddon_) {
      Real src_ks = 0.0, src_da = 0.0;
      for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
          src_ks += alpha*adm.vK_dd(m, a, b, k, j, i)*S_uu[a][b];
          src_da -= g3u[imap[a][b]] * S_d[a]*dalpha_d[b];
        }
      }
      dd_(m, 6, k, j, i) = src_ks;
      dd_(m, 7, k, j, i) = src_da;
      Real srcS[3];
      for (int a = 0; a < 3; a++) {
        srcS[a] = -E*dam[a];
        for (int b = 0; b < 3; b++) {
          srcS[a] += S_d[b]*dbm[a][b];
          for (int c = 0; c < 3; c++) {
            srcS[a] += 0.5*alpha*S_uu[b][c]*dgm[a][b][c];
          }
        }
      }
      auto divf = [&](int n) {
        Real d = (fx1_(m, n, k, j, i+1) - fx1_(m, n, k, j, i))/size.d_view(m).dx1;
        if (md_) d += (fx2_(m, n, k, j+1, i) - fx2_(m, n, k, j, i))/size.d_view(m).dx2;
        if (d3_) d += (fx3_(m, n, k+1, j, i) - fx3_(m, n, k, j, i))/size.d_view(m).dx3;
        return d/vol;
      };
      Real dD = -divf(IDN), dE = -divf(IEN) + dD;
      Real vdS_f = 0.0, vdS_s = 0.0;
      for (int a = 0; a < 3; a++) {
        Real va = prim_pt[PVX + a]/W;
        vdS_f -= va*divf(IM1 + a);
        vdS_s += va*srcS[a];
      }
      Real rho_ = prim(m, IDN, k, j, i);
      Real e_int = W*fmax(H - rho_ - prim_pt[PPR], 1.0e-300);
      dd_(m, 8, k, j, i) = (W*(dE - vdS_f) - dD)/e_int;
      dd_(m, 9, k, j, i) = W*(src_ks + src_da - vdS_s)/e_int;
    }

    // Assemble momentum RHS
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3; b++) {
        for (int c = 0; c < 3; c++) {
          rhs(m,IM1+a, k, j, i) += 0.5*dt*alpha*vol*S_uu[b][c]*dgm[a][b][c];
        }
        rhs(m, IM1+a, k, j, i) += dt*vol*S_d[b]*dbm[a][b];
      }
      rhs(m, IM1+a, k, j, i) -= dt*vol*E*dam[a];
    }

    if (smoothing) {
      Real D = prim(m, IDN, k, j, i) * W;
      Real tau = E - D;
      Real tau_ex = eos_.GetEnergy(dexcise/mb, texcise, &prim_pt[PYF])
                                    + 0.5*Bsq - dexcise;

      rhs(m, IDN, k, j, i) -= (dt*vol*floor(m,k,j,i)*(D-dexcise))/tdamp;
      rhs(m, IM1, k, j, i) -= (dt*floor(m,k,j,i)*vol*S_d[0])/tdamp;
      rhs(m, IM2, k, j, i) -= (dt*floor(m,k,j,i)*vol*S_d[1])/tdamp;
      rhs(m, IM3, k, j, i) -= (dt*floor(m,k,j,i)*vol*S_d[2])/tdamp;
      rhs(m, IEN, k, j, i) -= dt*vol*floor(m,k,j,i)*(tau - tau_ex)/tdamp;
      for (int s = 0; s < nscal; s++) {
        rhs(m, IYF+s, k, j, i) -= (dt*vol*floor(m,k,j,i)*(D-dexcise)*prim_pt[PYF+s])
                                      /tdamp;
      }
    }
  });
}

#define INSTANTIATE_COORD_TERMS_FM(EOSPolicy, ErrorPolicy) \
template \
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddCoordTermsFM<2>( \
      const DvceArray5D<Real> &prim, \
      const DvceArray5D<Real> &bcc, const Real dt, DvceArray5D<Real> &rhs); \
template \
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddCoordTermsFM<3>( \
      const DvceArray5D<Real> &prim, \
      const DvceArray5D<Real> &bcc, const Real dt, DvceArray5D<Real> &rhs); \
template \
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddCoordTermsFM<4>( \
      const DvceArray5D<Real> &prim, \
      const DvceArray5D<Real> &bcc, const Real dt, DvceArray5D<Real> &rhs);

INSTANTIATE_COORD_TERMS_FM(Primitive::IdealGas, Primitive::ResetFloor);
INSTANTIATE_COORD_TERMS_FM(Primitive::PiecewisePolytrope, Primitive::ResetFloor);
INSTANTIATE_COORD_TERMS_FM(Primitive::EOSCompOSE<Primitive::NormalLogs>,
                           Primitive::ResetFloor);
INSTANTIATE_COORD_TERMS_FM(Primitive::EOSCompOSE<Primitive::NQTLogs>,
                           Primitive::ResetFloor);
INSTANTIATE_COORD_TERMS_FM(Primitive::EOSHybrid<Primitive::NormalLogs>,
                           Primitive::ResetFloor);
INSTANTIATE_COORD_TERMS_FM(Primitive::EOSHybrid<Primitive::NQTLogs>,
                           Primitive::ResetFloor);

}  // namespace dyngr
