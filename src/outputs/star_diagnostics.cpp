//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file star_diagnostics.cpp
//! \brief Star-centred diagnostics for dynamical-GR (M)HD stars (file_type = star).
//!
//! For each star k (1 or 2), every output:
//!  - region: rho_* > cut*max(rho_*) on star k's side of the plane bisecting the two seeds
//!    (if two stars), within rmax of the seed. rho_* = sqrt(gamma) rho W = conserved dens.
//!    Seeds: the z4c compact-object trackers, or <block>/star<k>_x,_y,_z without z4c.
//!  - COM x_k = int rho_* x dV / int rho_* dV (coordinate dV); also for cut*10 and cut/10.
//!  - mass-weighted u_i (covariant, u_i = gamma_ij W v^j) and coordinate velocity
//!    V^i = alpha v^i - beta^i.
//!  - spin J = int rho_* h [dx x (u - ubar)]  (dx = x - x_k; conventional sign
//!    J_z = dx (u_y - ubar_y) - dy (u_x - ubar_x); Lam+ 2023 Eq. 59, Kuan+ 2024 Eq. 4
//!    write the opposite orientation). h = [(tau + D)/sqrt(gamma) + p]/(rho W^2) from the
//!    densitized conserved variables (no EOS call; B = 0 assumed).
//!  - multipoles I_lm = int rho_* r^l Y*_lm dV about x_k, l <= 4, m >= 0, complex
//!    orthonormal Y_lm with the Condon-Shortley phase.
//!  - every shell_every-th output, per radial shell j about x_k (nshell shells to rshell):
//!    int_shell rho_* Q Y*_lm dV for Q = 1, V^r, f (scalar 0), and the horizontal
//!    projections V.(r grad Y*_lm) (polar) and V.(rhat x r grad Y*_lm) (axial), plus the
//!    shell mass and int rho_* Omega dV with Omega = (dx V^y - dy V^x)/(dx^2 + dy^2);
//!    then int_shell rho_* p Y*_lm dV (l <= 2) and the plain monopoles int_shell rho_* Q dV
//!    for Q = eps, psi = gamma^(1/12), alpha, scalars 1-3 as advected (zla flat packing:
//!    Y_N/f, y_lN, y_lQ), and the shell coordinate volume of the cells used (rho_* above the
//!    cut). eps = h - 1 - p/rho (no EOS call).
//!  - global: int rho_* h (x u_y - y u_x) dV about the origin.
//! Bitant meshes (reflecting x3 = 0): the upper half is integrated and doubled for parity-even
//! quantities; z_k, J_x, J_y and the l+m odd harmonics are set to 0.
//! Writes <basename>.star<k>.txt (one row per output) and <basename>.star<k>_shells.bin
//! (per record: time, then nshell x NSHF doubles; layout in the .txt header).

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "mhd/mhd.hpp"
#include "z4c/z4c.hpp"
#include "z4c/compact_object_tracker.hpp"
#include "outputs.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace star_diag {

constexpr int NLM = 15;            // (l,m), l = 0..4, m = 0..l; index l(l+1)/2 + m
constexpr int NLP = 6;             // l <= 2 for the pressure projection
constexpr int NPL = 7;             // plain monopoles: eps, psi, alpha, scalars 1-3, volume
// 5 complex projections + shell mass + int rho_* Omega + p (l <= 2) + plain monopoles
constexpr int NSHF = 5*2*NLM + 2 + 2*NLP + NPL;

//! conj(Y_lm) at unit vector (nx, ny, nz) for all l <= 4, m >= 0 (re, im interleaved)
KOKKOS_INLINE_FUNCTION
void YlmConj(Real nx, Real ny, Real nz, Real *yr, Real *yi) {
  const Real pi = 3.14159265358979323846;
  const Real mu = nz, mu2 = mu*mu;
  // m-th derivatives of P_l at mu
  const Real P[NLM] = {1.0,
                       mu, 1.0,
                       0.5*(3*mu2 - 1), 3*mu, 3.0,
                       0.5*(5*mu2*mu - 3*mu), 0.5*(15*mu2 - 3), 15*mu, 15.0,
                       0.125*(35*mu2*mu2 - 30*mu2 + 3), 0.5*(35*mu2*mu - 15*mu),
                       0.5*(105*mu2 - 15), 105*mu, 105.0};
  // N_lm = sqrt((2l+1)/(4 pi) (l-m)!/(l+m)!)
  const Real fac[9] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320};
  // (nx - i ny)^m (conjugate), times (-1)^m
  Real cr = 1.0, ci = 0.0;
  Real pr[5], pim[5];
  for (int m = 0; m <= 4; ++m) {
    pr[m] = cr; pim[m] = ci;
    Real t = cr*nx + ci*ny; ci = ci*nx - cr*ny; cr = t;
  }
  for (int l = 0; l <= 4; ++l) {
    for (int m = 0; m <= l; ++m) {
      int n = l*(l + 1)/2 + m;
      Real N = Kokkos::sqrt((2*l + 1)/(4*pi)*fac[l - m]/fac[l + m]);
      Real s = ((m & 1) ? -1.0 : 1.0)*N*P[n];
      yr[n] = s*pr[m]; yi[n] = s*pim[m];
    }
  }
}

//! d/de conj(Y_lm)(normalize(n + e d)) at e = 0, by a centred difference (r grad Y . d)
KOKKOS_INLINE_FUNCTION
void YlmConjDeriv(Real nx, Real ny, Real nz, Real dx, Real dy, Real dz,
                  Real *gr, Real *gi) {
  const Real e = 1.0e-4;
  Real ar[NLM], ai[NLM], br[NLM], bi[NLM];
  Real px = nx + e*dx, py = ny + e*dy, pz = nz + e*dz;
  Real ip = 1.0/Kokkos::sqrt(px*px + py*py + pz*pz);
  YlmConj(px*ip, py*ip, pz*ip, ar, ai);
  px = nx - e*dx; py = ny - e*dy; pz = nz - e*dz;
  ip = 1.0/Kokkos::sqrt(px*px + py*py + pz*pz);
  YlmConj(px*ip, py*ip, pz*ip, br, bi);
  for (int n = 0; n < NLM; ++n) { gr[n] = (ar[n] - br[n])/(2*e); gi[n] = (ai[n] - bi[n])/(2*e); }
}

}  // namespace star_diag

//----------------------------------------------------------------------------------------
//! \fn StarDiagnostics::FindCentres
//! \brief COM of each star (the first pass of the diagnostics). Also used by re-centred
//! sph outputs. Fills com[k][3]. Returns false if a star has no cells in its region.

bool StarDiagnostics::FindCentres(Mesh *pm, Real com[2][3]) {
  Real seed[2][3];
  GetSeeds(pm, seed);
  Real sums[2][20];
  Pass1(pm, seed, sums);
  bool ok = true;
  for (int k = 0; k < nstar; ++k) {
    if (!(sums[k][0] > 0.0)) { ok = false; continue; }
    for (int a = 0; a < 3; ++a) com[k][a] = sums[k][1 + a]/sums[k][0];
    if (bitant) com[k][2] = 0.0;
  }
  return ok;
}

StarDiagnostics::StarDiagnostics(ParameterInput *pin, Mesh *pm, std::string block) {
  nstar = pin->GetOrAddInteger(block, "nstar", 2);
  rho_cut = pin->GetOrAddReal(block, "rho_cut", 1.0e-3);
  rmax = pin->GetOrAddReal(block, "rmax", 12.0);
  nshell = pin->GetOrAddInteger(block, "nshell", 32);
  rshell = pin->GetOrAddReal(block, "rshell", 10.0);
  for (int k = 0; k < 2; ++k) {
    std::string s = "star" + std::to_string(k);
    seed_par[k][0] = pin->GetOrAddReal(block, s + "_x", 0.0);
    seed_par[k][1] = pin->GetOrAddReal(block, s + "_y", 0.0);
    seed_par[k][2] = pin->GetOrAddReal(block, s + "_z", 0.0);
  }
  if (nstar < 1 || nstar > 2) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<" << block << ">/nstar must be 1 or 2" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  bitant = (pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::reflect &&
            std::abs(pm->mesh_size.x3min) < 1.0e-12);
}

void StarDiagnostics::GetSeeds(Mesh *pm, Real seed[2][3]) {
  auto pz4c = pm->pmb_pack->pz4c;
  for (int k = 0; k < nstar; ++k) {
    bool trk = (pz4c != nullptr && static_cast<int>(pz4c->ptracker.size()) > k);
    for (int a = 0; a < 3; ++a) {
      seed[k][a] = trk ? pz4c->ptracker[k]->GetPos(a) : seed_par[k][a];
    }
    if (bitant) seed[k][2] = 0.0;
  }
}

//! region mask parameters for star k: plane normal (nrm), plane point (mid), side sign
void StarDiagnostics::Plane(const Real seed[2][3], int k, Real nrm[3], Real mid[3]) {
  if (nstar == 1) { for (int a = 0; a < 3; ++a) { nrm[a] = 0.0; mid[a] = 0.0; } return; }
  int o = 1 - k; Real d2 = 0.0;
  for (int a = 0; a < 3; ++a) { nrm[a] = seed[k][a] - seed[o][a]; d2 += nrm[a]*nrm[a]; }
  Real id = (d2 > 0.0) ? 1.0/std::sqrt(d2) : 0.0;
  for (int a = 0; a < 3; ++a) { nrm[a] *= id; mid[a] = 0.5*(seed[k][a] + seed[o][a]); }
}

Real StarDiagnostics::MaxDens(Mesh *pm) {
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int nmb = pm->pmb_pack->nmb_thispack;
  auto &u0 = pm->pmb_pack->pmhd->u0;
  Real dmax = 0.0;
  Kokkos::parallel_reduce("star_dmax",
    Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>({0, 0, 0, 0}, {nmb, nx3, nx2, nx1}),
    KOKKOS_LAMBDA(int m, int k, int j, int i, Real &mx) {
      mx = fmax(mx, u0(m, IDN, k + ks, j + js, i + is));
    }, Kokkos::Max<Real>(dmax));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &dmax, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  return dmax;
}

//! pass 1: [0] M, [1..3] M x, [4..6] M u_i, [7..9] M V^i, [10..12] M,Mx,My (cut*10),
//! [13..15] M,Mx,My (cut/10); bitant doubling applied
void StarDiagnostics::Pass1(Mesh *pm, const Real seed[2][3], Real sums[2][20]) {
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int nmb = pm->pmb_pack->nmb_thispack;
  auto &size = pm->pmb_pack->pmb->mb_size;
  auto &u0 = pm->pmb_pack->pmhd->u0;
  auto &w0 = pm->pmb_pack->pmhd->w0;
  auto &adm = pm->pmb_pack->padm->adm;
  Real dcut = rho_cut*MaxDens(pm);
  const Real rmx2 = rmax*rmax;
  for (int s = 0; s < nstar; ++s) {
    Real nrm[3], mid[3]; Plane(seed, s, nrm, mid);
    const Real sx = seed[s][0], sy = seed[s][1], sz = seed[s][2];
    const Real n0 = nrm[0], n1 = nrm[1], n2 = nrm[2], m0 = mid[0], m1 = mid[1], m2 = mid[2];
    array_sum::GlobalSum res;
    Kokkos::parallel_reduce("star_pass1",
      Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>({0, 0, 0, 0}, {nmb, nx3, nx2, nx1}),
      KOKKOS_LAMBDA(int m, int k, int j, int i, array_sum::GlobalSum &acc) {
        Real x = CellCenterX(i, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
        Real y = CellCenterX(j, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
        Real z = CellCenterX(k, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
        Real ddx = x - sx, ddy = y - sy, ddz = z - sz;
        if (ddx*ddx + ddy*ddy + ddz*ddz > rmx2) return;
        if ((x - m0)*n0 + (y - m1)*n1 + (z - m2)*n2 < 0.0) return;
        Real D = u0(m, IDN, k + ks, j + js, i + is);
        if (D <= 0.1*dcut) return;
        Real dV = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
        Real Dv = D*dV;
        // cut/10 variant (always passes here)
        acc.the_array[13] += Dv; acc.the_array[14] += Dv*x; acc.the_array[15] += Dv*y;
        if (D <= dcut) return;
        Real g[3][3] = {{adm.g_dd(m, 0, 0, k + ks, j + js, i + is),
                         adm.g_dd(m, 0, 1, k + ks, j + js, i + is),
                         adm.g_dd(m, 0, 2, k + ks, j + js, i + is)},
                        {0, adm.g_dd(m, 1, 1, k + ks, j + js, i + is),
                         adm.g_dd(m, 1, 2, k + ks, j + js, i + is)},
                        {0, 0, adm.g_dd(m, 2, 2, k + ks, j + js, i + is)}};
        g[1][0] = g[0][1]; g[2][0] = g[0][2]; g[2][1] = g[1][2];
        Real ut[3] = {w0(m, IVX, k + ks, j + js, i + is), w0(m, IVY, k + ks, j + js, i + is),
                      w0(m, IVZ, k + ks, j + js, i + is)};
        Real ud[3];
        for (int a = 0; a < 3; ++a) ud[a] = g[a][0]*ut[0] + g[a][1]*ut[1] + g[a][2]*ut[2];
        Real W = Kokkos::sqrt(1.0 + ud[0]*ut[0] + ud[1]*ut[1] + ud[2]*ut[2]);
        Real alp = adm.alpha(m, k + ks, j + js, i + is);
        acc.the_array[0] += Dv;
        acc.the_array[1] += Dv*x; acc.the_array[2] += Dv*y; acc.the_array[3] += Dv*z;
        for (int a = 0; a < 3; ++a) {
          acc.the_array[4 + a] += Dv*ud[a];
          acc.the_array[7 + a] += Dv*(alp*ut[a]/W - adm.beta_u(m, a, k + ks, j + js, i + is));
        }
        if (D > 10.0*dcut) {
          acc.the_array[10] += Dv; acc.the_array[11] += Dv*x; acc.the_array[12] += Dv*y;
        }
      }, Kokkos::Sum<array_sum::GlobalSum>(res));
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &res.the_array[0], 16, MPI_ATHENA_REAL, MPI_SUM,
                  MPI_COMM_WORLD);
#endif
    for (int n = 0; n < 20; ++n) sums[s][n] = (bitant ? 2.0 : 1.0)*res.the_array[n];
    if (bitant) { sums[s][3] = 0.0; sums[s][6] = 0.0; sums[s][9] = 0.0; }
  }
}

//! pass 2 about com: J (3), I_lm (15 complex), I_22 for cut*10 and cut/10, and (if
//! shells) the shell projections into shell[nshell*NSHF]. ubar = mass-weighted u_i.
void StarDiagnostics::Pass2(Mesh *pm, const Real seed[2][3], int s, const Real com[3],
                            const Real ubar[3], Real out[40], bool do_shells,
                            std::vector<Real> &shell) {
  using namespace star_diag;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int nmb = pm->pmb_pack->nmb_thispack;
  auto &size = pm->pmb_pack->pmb->mb_size;
  auto &u0 = pm->pmb_pack->pmhd->u0;
  auto &w0 = pm->pmb_pack->pmhd->w0;
  auto &adm = pm->pmb_pack->padm->adm;
  const int nmhd = pm->pmb_pack->pmhd->nmhd;
  const int nscal = pm->pmb_pack->pmhd->nscalars;
  Real dcut = rho_cut*MaxDens(pm);
  const Real rmx2 = rmax*rmax;
  Real nrm[3], mid[3]; Plane(seed, s, nrm, mid);
  const Real sx = seed[s][0], sy = seed[s][1], sz = seed[s][2];
  const Real n0 = nrm[0], n1 = nrm[1], n2 = nrm[2], m0 = mid[0], m1 = mid[1], m2 = mid[2];
  const Real cx = com[0], cy = com[1], cz = com[2];
  const Real ub0 = ubar[0], ub1 = ubar[1], ub2 = ubar[2];
  const int nsh = nshell; const Real rsh = rshell;
  DvceArray1D<Real> shv("star_shells", do_shells ? nsh*NSHF : 1);
  Kokkos::deep_copy(shv, 0.0);
  const bool dsh = do_shells;
  array_sum::GlobalSum ra, rb;
  for (int pass = 0; pass < 2; ++pass) {
    array_sum::GlobalSum res;
    Kokkos::parallel_reduce("star_pass2",
      Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>({0, 0, 0, 0}, {nmb, nx3, nx2, nx1}),
      KOKKOS_LAMBDA(int m, int k, int j, int i, array_sum::GlobalSum &acc) {
        Real x = CellCenterX(i, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
        Real y = CellCenterX(j, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
        Real z = CellCenterX(k, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
        Real ddx = x - sx, ddy = y - sy, ddz = z - sz;
        if (ddx*ddx + ddy*ddy + ddz*ddz > rmx2) return;
        if ((x - m0)*n0 + (y - m1)*n1 + (z - m2)*n2 < 0.0) return;
        Real D = u0(m, IDN, k + ks, j + js, i + is);
        if (D <= 0.1*dcut) return;
        Real dV = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
        Real Dv = D*dV;
        Real rx = x - cx, ry = y - cy, rz = z - cz;
        Real r2 = rx*rx + ry*ry + rz*rz, r = Kokkos::sqrt(r2);
        Real ir = (r > 0.0) ? 1.0/r : 0.0;
        Real nx = rx*ir, ny = ry*ir, nz = (r > 0.0) ? rz*ir : 1.0;
        Real yr[NLM], yi[NLM];
        YlmConj(nx, ny, nz, yr, yi);
        if (pass == 0) {
          // I_22 for the cut variants (n = 5)
          Real r2l = r2;
          acc.the_array[15] += Dv*r2l*yr[5]; acc.the_array[16] += Dv*r2l*yi[5];
          if (D > 10.0*dcut) {
            acc.the_array[17] += Dv*r2l*yr[5]; acc.the_array[18] += Dv*r2l*yi[5];
          }
        }
        if (D <= dcut) return;
        Real rl[5] = {1.0, r, r2, r2*r, r2*r2};
        if (pass == 0) {
          Real g[3][3] = {{adm.g_dd(m, 0, 0, k + ks, j + js, i + is),
                           adm.g_dd(m, 0, 1, k + ks, j + js, i + is),
                           adm.g_dd(m, 0, 2, k + ks, j + js, i + is)},
                          {0, adm.g_dd(m, 1, 1, k + ks, j + js, i + is),
                           adm.g_dd(m, 1, 2, k + ks, j + js, i + is)},
                          {0, 0, adm.g_dd(m, 2, 2, k + ks, j + js, i + is)}};
          g[1][0] = g[0][1]; g[2][0] = g[0][2]; g[2][1] = g[1][2];
          Real detg = g[0][0]*(g[1][1]*g[2][2] - g[1][2]*g[2][1])
                    - g[0][1]*(g[1][0]*g[2][2] - g[1][2]*g[2][0])
                    + g[0][2]*(g[1][0]*g[2][1] - g[1][1]*g[2][0]);
          Real ut[3] = {w0(m, IVX, k + ks, j + js, i + is), w0(m, IVY, k + ks, j + js, i + is),
                        w0(m, IVZ, k + ks, j + js, i + is)};
          Real ud[3];
          for (int a = 0; a < 3; ++a) ud[a] = g[a][0]*ut[0] + g[a][1]*ut[1] + g[a][2]*ut[2];
          Real W2 = 1.0 + ud[0]*ut[0] + ud[1]*ut[1] + ud[2]*ut[2];
          Real rho = w0(m, IDN, k + ks, j + js, i + is);
          Real h = ((u0(m, IEN, k + ks, j + js, i + is) + D)/Kokkos::sqrt(detg)
                    + w0(m, IPR, k + ks, j + js, i + is))/(rho*W2);
          Real vx = ud[0] - ub0, vy = ud[1] - ub1, vz = ud[2] - ub2;
          acc.the_array[0] += Dv*h*(ry*vz - rz*vy);
          acc.the_array[1] += Dv*h*(rz*vx - rx*vz);
          acc.the_array[2] += Dv*h*(rx*vy - ry*vx);
          for (int n = 0; n < 6; ++n) {   // l <= 2
            int l = (n == 0) ? 0 : (n < 3 ? 1 : 2);
            acc.the_array[3 + 2*n] += Dv*rl[l]*yr[n];
            acc.the_array[4 + 2*n] += Dv*rl[l]*yi[n];
          }
          if (dsh && r < rsh) {
            int jsh = static_cast<int>(r/rsh*nsh);
            if (jsh >= nsh) jsh = nsh - 1;
            Real alp = adm.alpha(m, k + ks, j + js, i + is), iW = 1.0/Kokkos::sqrt(W2);
            Real V[3];
            for (int a = 0; a < 3; ++a)
              V[a] = alp*ut[a]*iW - adm.beta_u(m, a, k + ks, j + js, i + is);
            Real Vr = V[0]*nx + V[1]*ny + V[2]*nz;
            Real Vp[3] = {V[0] - Vr*nx, V[1] - Vr*ny, V[2] - Vr*nz};   // polar direction
            Real Va[3] = {V[1]*nz - V[2]*ny, V[2]*nx - V[0]*nz, V[0]*ny - V[1]*nx};  // V x n
            Real f = (nscal > 0) ? w0(m, nmhd, k + ks, j + js, i + is) : 1.0;
            Real gpr[NLM], gpi[NLM], gar[NLM], gai[NLM];
            YlmConjDeriv(nx, ny, nz, Vp[0], Vp[1], Vp[2], gpr, gpi);
            YlmConjDeriv(nx, ny, nz, Va[0], Va[1], Va[2], gar, gai);
            Real *b = &shv(jsh*NSHF);
            for (int n = 0; n < NLM; ++n) {
              Kokkos::atomic_add(&b[2*n], Dv*yr[n]);
              Kokkos::atomic_add(&b[2*n + 1], Dv*yi[n]);
              Kokkos::atomic_add(&b[2*NLM + 2*n], Dv*Vr*yr[n]);
              Kokkos::atomic_add(&b[2*NLM + 2*n + 1], Dv*Vr*yi[n]);
              Kokkos::atomic_add(&b[4*NLM + 2*n], Dv*f*yr[n]);
              Kokkos::atomic_add(&b[4*NLM + 2*n + 1], Dv*f*yi[n]);
              Kokkos::atomic_add(&b[6*NLM + 2*n], Dv*gpr[n]);
              Kokkos::atomic_add(&b[6*NLM + 2*n + 1], Dv*gpi[n]);
              Kokkos::atomic_add(&b[8*NLM + 2*n], Dv*gar[n]);
              Kokkos::atomic_add(&b[8*NLM + 2*n + 1], Dv*gai[n]);
            }
            Real rc2 = rx*rx + ry*ry;
            Kokkos::atomic_add(&b[10*NLM], Dv);
            if (rc2 > 0.0) Kokkos::atomic_add(&b[10*NLM + 1], Dv*(rx*V[1] - ry*V[0])/rc2);
            Real p = w0(m, IPR, k + ks, j + js, i + is);
            Real *bp = b + 10*NLM + 2, *bq = bp + 2*NLP;
            for (int n = 0; n < NLP; ++n) {
              Kokkos::atomic_add(&bp[2*n], Dv*p*yr[n]);
              Kokkos::atomic_add(&bp[2*n + 1], Dv*p*yi[n]);
            }
            Kokkos::atomic_add(&bq[0], Dv*(h - 1.0 - p/rho));
            Kokkos::atomic_add(&bq[1], Dv*Kokkos::pow(detg, 1.0/12.0));
            Kokkos::atomic_add(&bq[2], Dv*alp);
            for (int q = 1; q < 4; ++q) {
              Real sq = (nscal > q) ? w0(m, nmhd + q, k + ks, j + js, i + is) : 0.0;
              Kokkos::atomic_add(&bq[2 + q], Dv*sq);
            }
            Kokkos::atomic_add(&bq[6], dV);
          }
        } else {
          for (int n = 6; n < NLM; ++n) {  // l = 3, 4
            int l = (n < 10) ? 3 : 4;
            acc.the_array[2*(n - 6)] += Dv*rl[l]*yr[n];
            acc.the_array[2*(n - 6) + 1] += Dv*rl[l]*yi[n];
          }
        }
      }, Kokkos::Sum<array_sum::GlobalSum>(res));
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &res.the_array[0], 20, MPI_ATHENA_REAL, MPI_SUM,
                  MPI_COMM_WORLD);
#endif
    if (pass == 0) ra = res; else rb = res;
  }
  // assemble: out[0..2] J, out[3..32] I_lm (15 complex), out[33..36] I_22 cut*10, cut/10
  auto lpm = [](int n) { int l = 0; while ((l + 1)*(l + 2)/2 <= n) ++l; return l + (n - l*(l + 1)/2); };
  Real bf = bitant ? 2.0 : 1.0;
  out[0] = bitant ? 0.0 : ra.the_array[0];
  out[1] = bitant ? 0.0 : ra.the_array[1];
  out[2] = bf*ra.the_array[2];
  for (int n = 0; n < NLM; ++n) {
    Real re = (n < 6) ? ra.the_array[3 + 2*n] : rb.the_array[2*(n - 6)];
    Real im = (n < 6) ? ra.the_array[4 + 2*n] : rb.the_array[2*(n - 6) + 1];
    Real w = bitant ? ((lpm(n) & 1) ? 0.0 : 2.0) : 1.0;
    out[3 + 2*n] = w*re; out[4 + 2*n] = w*im;
  }
  out[33] = bf*ra.the_array[17]; out[34] = bf*ra.the_array[18];
  out[35] = bf*ra.the_array[15]; out[36] = bf*ra.the_array[16];
  if (do_shells) {
    auto h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), shv);
    shell.assign(nsh*NSHF, 0.0);
    for (int q = 0; q < nsh*NSHF; ++q) shell[q] = h(q);
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, shell.data(), nsh*NSHF, MPI_ATHENA_REAL, MPI_SUM,
                  MPI_COMM_WORLD);
#endif
    for (int jj = 0; jj < nsh; ++jj) {
      Real *b = &shell[jj*NSHF];
      for (int n = 0; n < NLM; ++n) {
        int par = lpm(n) & 1;   // 0: Y even in z
        Real we = bitant ? (par ? 0.0 : 2.0) : 1.0, wo = bitant ? (par ? 2.0 : 0.0) : 1.0;
        for (int f = 0; f < 4; ++f) { b[2*f*NLM + 2*n] *= we; b[2*f*NLM + 2*n + 1] *= we; }
        b[8*NLM + 2*n] *= wo; b[8*NLM + 2*n + 1] *= wo;   // axial: opposite parity
      }
      b[10*NLM] *= bf; b[10*NLM + 1] *= bf;
      for (int n = 0; n < NLP; ++n) {
        Real we = bitant ? ((lpm(n) & 1) ? 0.0 : 2.0) : 1.0;
        b[10*NLM + 2 + 2*n] *= we; b[10*NLM + 3 + 2*n] *= we;
      }
      for (int q = 0; q < NPL; ++q) b[10*NLM + 2 + 2*NLP + q] *= bf;
    }
  }
}

//! global J_z about the origin: int rho_* h (x u_y - y u_x) dV over the whole domain
Real StarDiagnostics::GlobalJz(Mesh *pm) {
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int nmb = pm->pmb_pack->nmb_thispack;
  auto &size = pm->pmb_pack->pmb->mb_size;
  auto &u0 = pm->pmb_pack->pmhd->u0;
  auto &w0 = pm->pmb_pack->pmhd->w0;
  auto &adm = pm->pmb_pack->padm->adm;
  Real jz = 0.0;
  Kokkos::parallel_reduce("star_globalJ",
    Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>({0, 0, 0, 0}, {nmb, nx3, nx2, nx1}),
    KOKKOS_LAMBDA(int m, int k, int j, int i, Real &acc) {
      Real x = CellCenterX(i, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      Real y = CellCenterX(j, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      Real D = u0(m, IDN, k + ks, j + js, i + is);
      Real g00 = adm.g_dd(m, 0, 0, k + ks, j + js, i + is), g01 = adm.g_dd(m, 0, 1, k + ks, j + js, i + is),
           g02 = adm.g_dd(m, 0, 2, k + ks, j + js, i + is), g11 = adm.g_dd(m, 1, 1, k + ks, j + js, i + is),
           g12 = adm.g_dd(m, 1, 2, k + ks, j + js, i + is), g22 = adm.g_dd(m, 2, 2, k + ks, j + js, i + is);
      Real ut0 = w0(m, IVX, k + ks, j + js, i + is), ut1 = w0(m, IVY, k + ks, j + js, i + is),
           ut2 = w0(m, IVZ, k + ks, j + js, i + is);
      Real ux = g00*ut0 + g01*ut1 + g02*ut2, uy = g01*ut0 + g11*ut1 + g12*ut2,
           uz = g02*ut0 + g12*ut1 + g22*ut2;
      Real W2 = 1.0 + ux*ut0 + uy*ut1 + uz*ut2;
      Real detg = g00*(g11*g22 - g12*g12) - g01*(g01*g22 - g12*g02) + g02*(g01*g12 - g11*g02);
      Real rho = w0(m, IDN, k + ks, j + js, i + is);
      if (!(rho > 0.0)) return;
      Real h = ((u0(m, IEN, k + ks, j + js, i + is) + D)/Kokkos::sqrt(detg)
                + w0(m, IPR, k + ks, j + js, i + is))/(rho*W2);
      acc += D*size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3*h*(x*uy - y*ux);
    }, Kokkos::Sum<Real>(jz));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &jz, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  return (bitant ? 2.0 : 1.0)*jz;
}

//----------------------------------------------------------------------------------------
// StarDiagOutput: the <output> wrapper

StarDiagOutput::StarDiagOutput(ParameterInput *pin, Mesh *pm, OutputParameters op)
    : BaseTypeOutput(pin, pm, op), diag(pin, pm, op.block_name) {
  shell_every = pin->GetOrAddInteger(op.block_name, "shell_every", 5);
  if (pm->pmb_pack->pmhd == nullptr || pm->pmb_pack->padm == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "star output needs <mhd> and the ADM variables" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void StarDiagOutput::LoadOutputData(Mesh *pm) {
  using star_diag::NSHF;
  Real seed[2][3];
  diag.GetSeeds(pm, seed);
  Real s1[2][20];
  diag.Pass1(pm, seed, s1);
  do_shells_now = (shell_every > 0) && (ncall % shell_every == 0);
  ncall++;
  for (int k = 0; k < diag.nstar; ++k) {
    Real M = s1[k][0];
    Real com[3] = {0.0, 0.0, 0.0}, ub[3] = {0.0, 0.0, 0.0}, vb[3] = {0.0, 0.0, 0.0};
    if (M > 0.0) {
      for (int a = 0; a < 3; ++a) {
        com[a] = s1[k][1 + a]/M; ub[a] = s1[k][4 + a]/M; vb[a] = s1[k][7 + a]/M;
      }
    }
    Real out[40] = {0.0};
    if (M > 0.0) diag.Pass2(pm, seed, k, com, ub, out, do_shells_now, shells[k]);
    Real *r = row[k];
    r[0] = M;
    for (int a = 0; a < 3; ++a) { r[1 + a] = com[a]; r[4 + a] = ub[a]; r[7 + a] = vb[a]; }
    for (int n = 0; n < 37; ++n) r[10 + n] = out[n];
    r[47] = s1[k][10]; r[48] = (s1[k][10] > 0.0) ? s1[k][11]/s1[k][10] : 0.0;
    r[49] = (s1[k][10] > 0.0) ? s1[k][12]/s1[k][10] : 0.0;
    r[50] = s1[k][13]; r[51] = (s1[k][13] > 0.0) ? s1[k][14]/s1[k][13] : 0.0;
    r[52] = (s1[k][13] > 0.0) ? s1[k][15]/s1[k][13] : 0.0;
    for (int a = 0; a < 3; ++a) r[53 + a] = seed[k][a];
    if (!do_shells_now || !(M > 0.0)) shells[k].clear();
  }
  global_jz = diag.GlobalJz(pm);
}

void StarDiagOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  using star_diag::NSHF;
  static const char *lm[15] = {"00", "10", "11", "20", "21", "22", "30", "31", "32", "33",
                               "40", "41", "42", "43", "44"};
  for (int k = 0; k < diag.nstar && global_variable::my_rank == 0; ++k) {
    std::string fn = out_params.file_basename + ".star" + std::to_string(k) + ".txt";
    FILE *fp = std::fopen(fn.c_str(), "a");
    if (fp == nullptr) continue;
    if (std::ftell(fp) == 0) {
      std::fprintf(fp, "# star diagnostics (star_diagnostics.cpp); rho_* = sqrt(gamma) rho W; "
                       "cut %g x max rho_*, rmax %g; bitant %d\n", diag.rho_cut, diag.rmax,
                       static_cast<int>(diag.bitant));
      std::fprintf(fp, "# shells file: %s.star%d_shells.bin, records = time + nshell(%d) x %d "
                       "doubles; per shell: [rho_* Y*, rho_* V^r Y*, rho_* f Y*, polar V.rgradY*, "
                       "axial V.(rhat x rgradY*)] x 15 complex (lm order 00,10,11,20,...,44; "
                       "re,im), shell mass, int rho_* Omega, rho_* p Y* x 6 complex (l <= 2), "
                       "int rho_* {eps, psi=gamma^(1/12), alpha, s1, s2, s3} "
                       "(zla flat s1-3: Y_N/f, y_lN, y_lQ), coord volume of cells used; "
                       "shells uniform in r to %g\n",
                   out_params.file_basename.c_str(), k, diag.nshell, NSHF, diag.rshell);
      std::string h = "# [1]time [2]M [3]x [4]y [5]z [6]ubar_x [7]ubar_y [8]ubar_z [9]Vbar_x "
                      "[10]Vbar_y [11]Vbar_z [12]J_x [13]J_y [14]J_z";
      int c = 15;
      for (int n = 0; n < 15; ++n) {
        h += " [" + std::to_string(c++) + "]ReI" + lm[n];
        h += " [" + std::to_string(c++) + "]ImI" + lm[n];
      }
      for (auto s : {"ReI22_cutx10", "ImI22_cutx10", "ReI22_cutd10", "ImI22_cutd10",
                     "M_cutx10", "x_cutx10", "y_cutx10", "M_cutd10", "x_cutd10", "y_cutd10",
                     "seed_x", "seed_y", "seed_z", "Jz_global_origin", "shells"}) {
        h += " [" + std::to_string(c++) + "]" + s;
      }
      std::fprintf(fp, "%s\n", h.c_str());
    }
    std::fprintf(fp, "%.17e", pm->time);
    for (int n = 0; n < 56; ++n) std::fprintf(fp, " %.17e", row[k][n]);
    std::fprintf(fp, " %.17e %d\n", global_jz, static_cast<int>(!shells[k].empty()));
    std::fclose(fp);
    if (!shells[k].empty()) {
      std::string fb = out_params.file_basename + ".star" + std::to_string(k) + "_shells.bin";
      FILE *fq = std::fopen(fb.c_str(), "ab");
      if (fq != nullptr) {
        double t = pm->time;
        std::fwrite(&t, sizeof(double), 1, fq);
        std::fwrite(shells[k].data(), sizeof(double), shells[k].size(), fq);
        std::fclose(fq);
      }
    }
  }
  // advance the output counter/time as the other outputs do
  out_params.file_number++;
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetInteger(out_params.block_name, "file_number", out_params.file_number);
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);
}
