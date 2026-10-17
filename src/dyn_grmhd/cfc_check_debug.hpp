#ifndef DYN_GRMHD_CFC_CHECK_DEBUG_HPP_
#define DYN_GRMHD_CFC_CHECK_DEBUG_HPP_
//========================================================================================
//! \file cfc_check_debug.hpp
//! \brief DEBUG instrumentation for the ghost-metric conservation check (TDE project
//! R-072 item 9; not for merging). <cfc_check> ledger = true: a per-RK-stage interior
//! rest-mass ledger, one CHKLEDGER line per cycle (rank 0):
//!   Y = sum_s w_s (M_pre(s) - gam0_s M_end(s-1) - gam1_s M(u1))   (the update's change;
//!       = -Fout to round-off if the flux update conserves),  Fout = outflow through the
//!       mesh's outer faces,  X = the C2P's change (floors),  M_end, M(u1).
//! <cfc_check> probe_box = "x0 x1 y0 y1 z0 z1", probe_cycles = N: for the first N cycles,
//! every cell copy (owner interior or ghost) in the box is written per stage, before the
//! C2P (tag 0) and at the fluxes (tag 1), to cfccheck_rank<NNNNN>.txt: tag cycle stage gid
//! i j k ghost x y z psi4 alpha beta(3) g_dd(6) K_dd(6) w0 rho P v(3) u0 D S(3) tau.
//========================================================================================
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "driver/driver.hpp"

namespace cfccheck {

struct State {
  bool ledger = false;
  int pcyc = 0, pcyc0 = -1;
  Real pb[6] = {0.0, -1.0, 0.0, -1.0, 0.0, -1.0};
  Real st[4][5] = {};   // per stage: M_pre, M_end, M(u1), Fout(beta dt), unused
  DvceArray2D<Real> buf;
  DvceArray1D<int> cnt;
};
inline State &S() { static State *s = new State; return *s; }   // never freed: Views must not outlive Kokkos::finalize

inline void Init(ParameterInput *pin) {
  auto &s = S();
  s.ledger = pin->GetOrAddBoolean("cfc_check", "ledger", false);
  s.pcyc = pin->GetOrAddInteger("cfc_check", "probe_cycles", 0);
  std::istringstream is(pin->GetOrAddString("cfc_check", "probe_box", ""));
  Real v[6];
  if (is >> v[0] >> v[1] >> v[2] >> v[3] >> v[4] >> v[5]) {
    for (int q = 0; q < 6; ++q) s.pb[q] = v[q];
  }
}

inline Real Mass(MeshBlockPack *pp, const DvceArray5D<Real> &a, int n) {
  auto &indcs = pp->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmb = pp->nmb_thispack, nkji = nx3*nx2*nx1, nji = nx2*nx1;
  auto &size = pp->pmb->mb_size;
  Real s = 0.0;
  Kokkos::parallel_reduce("chk_mass", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*nkji),
  KOKKOS_LAMBDA(const int idx, Real &a0) {
    const int m = idx/nkji;
    const int k = (idx - m*nkji)/nji + ks;
    const int j = (idx - m*nkji - (k - ks)*nji)/nx1 + js;
    const int i = idx - m*nkji - (k - ks)*nji - (j - js)*nx1 + is;
    a0 += a(m, n, k, j, i)*size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
  }, Kokkos::Sum<Real>(s));
  return s;
}

inline Real Fout(MeshBlockPack *pp) {
  auto &indcs = pp->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmb = pp->nmb_thispack, nkji = nx3*nx2*nx1, nji = nx2*nx1;
  auto &size = pp->pmb->mb_size;
  auto &flx = pp->pmhd->uflx;
  auto &ms = pp->pmesh->mesh_size;
  const Real mx1 = ms.x1min, Mx1 = ms.x1max, mx2 = ms.x2min, Mx2 = ms.x2max;
  const Real mx3 = ms.x3min, Mx3 = ms.x3max;
  Real fo = 0.0;
  Kokkos::parallel_reduce("chk_fout", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*nkji),
  KOKKOS_LAMBDA(const int idx, Real &a1) {
    const int m = idx/nkji;
    const int k = (idx - m*nkji)/nji + ks;
    const int j = (idx - m*nkji - (k - ks)*nji)/nx1 + js;
    const int i = idx - m*nkji - (k - ks)*nji - (j - js)*nx1 + is;
    auto &sz = size.d_view(m);
    const Real a1f = sz.dx2*sz.dx3, a2f = sz.dx1*sz.dx3, a3f = sz.dx1*sz.dx2;
    const Real tol = 1.0e-6*sz.dx1;
    if (i == is && sz.x1min <= mx1 + tol) a1 -= flx.x1f(m, IDN, k, j, i)*a1f;
    if (i == ie && sz.x1max >= Mx1 - tol) a1 += flx.x1f(m, IDN, k, j, i+1)*a1f;
    if (j == js && sz.x2min <= mx2 + tol) a1 -= flx.x2f(m, IDN, k, j, i)*a2f;
    if (j == je && sz.x2max >= Mx2 - tol) a1 += flx.x2f(m, IDN, k, j+1, i)*a2f;
    if (k == ks && sz.x3min <= mx3 + tol) a1 -= flx.x3f(m, IDN, k, j, i)*a3f;
    if (k == ke && sz.x3max >= Mx3 - tol) a1 += flx.x3f(m, IDN, k+1, j, i)*a3f;
  }, Kokkos::Sum<Real>(fo));
  return fo;
}

//! before the stage's C2P
inline void Pre(MeshBlockPack *pp, int stage) {
  auto &s = S();
  if (!s.ledger || stage < 1 || stage > 4) return;
  s.st[stage-1][0] = Mass(pp, pp->pmhd->u0, IDN);
  s.st[stage-1][2] = Mass(pp, pp->pmhd->u1, IDN);
}

//! after the stage's C2P; on the last stage, reduce and print
inline void Post(MeshBlockPack *pp, Driver *pd, int stage) {
  auto &s = S();
  const int ns = pd->nexp_stages;
  if (!s.ledger || stage < 1 || stage > ns || ns > 4) return;
  s.st[stage-1][1] = Mass(pp, pp->pmhd->u0, IDN);
  s.st[stage-1][3] = pd->beta[stage-1]*pp->pmesh->dt*Fout(pp);
  if (stage != ns) return;
  Real a[4][5];
  for (int q = 0; q < 4; ++q) for (int n = 0; n < 5; ++n) a[q][n] = (q < ns) ? s.st[q][n] : 0.0;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &a[0][0], 20, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  Real w[4];
  w[ns-1] = 1.0;
  for (int q = ns-2; q >= 0; --q) w[q] = pd->gam0[q+1]*w[q+1];
  const Real mu1 = a[0][2];
  Real Y = 0.0, X = 0.0, F = 0.0, mp = mu1;
  for (int q = 0; q < ns; ++q) {
    Y += w[q]*(a[q][0] - pd->gam0[q]*mp - pd->gam1[q]*mu1);
    X += w[q]*(a[q][1] - a[q][0]);
    F += w[q]*a[q][3];
    mp = a[q][1];
  }
  if (global_variable::my_rank == 0) {
    std::cout << std::setprecision(10) << "CHKLEDGER cycle=" << pp->pmesh->ncycle
              << " time=" << pp->pmesh->time << " dt=" << pp->pmesh->dt << " Mu1=" << mu1
              << " Mend=" << a[ns-1][1] << " Y=" << Y << " Fout=" << F << " X=" << X
              << std::endl;
  }
}

//! the ghost/owner probe (host-written, one file per rank)
inline void Probe(MeshBlockPack *pp, int tag, int stage) {
  auto &s = S();
  if (s.pcyc <= 0) return;
  const int cyc = pp->pmesh->ncycle;
  if (s.pcyc0 < 0) s.pcyc0 = cyc;
  if (cyc >= s.pcyc0 + s.pcyc || stage < 1 || stage > 4) return;
  auto &indcs = pp->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3, ng = indcs.ng;
  const int nmb = pp->nmb_thispack, gid0 = pp->gids;
  auto &size = pp->pmb->mb_size;
  auto &adm = pp->padm->adm;
  auto &w0 = pp->pmhd->w0;
  auto &u0 = pp->pmhd->u0;
  const int cap = 400000;
  if (static_cast<int>(s.buf.extent(0)) < cap) {
    Kokkos::realloc(s.buf, cap, 35);
    Kokkos::realloc(s.cnt, 1);
  }
  Kokkos::deep_copy(s.cnt, 0);
  auto &buf = s.buf;
  auto &cnt = s.cnt;
  const Real b0 = s.pb[0], b1 = s.pb[1], b2 = s.pb[2], b3 = s.pb[3], b4 = s.pb[4];
  const Real b5 = s.pb[5];
  par_for("chk_probe", DevExeSpace(), 0, nmb-1, 0, nx3+2*ng-1, 0, nx2+2*ng-1, 0,
          nx1+2*ng-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real x = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    const Real y = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    const Real z = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    if (!(x > b0 && x < b1 && y > b2 && y < b3 && z > b4 && z < b5)) return;
    const int r = Kokkos::atomic_fetch_add(&cnt(0), 1);
    if (r >= cap) return;
    const int gh = (i < is || i > ie || j < js || j > je || k < ks || k > ke) ? 1 : 0;
    Real v[35] = {static_cast<Real>(gid0 + m), static_cast<Real>(i), static_cast<Real>(j),
                  static_cast<Real>(k), static_cast<Real>(gh), x, y, z,
                  adm.psi4(m,k,j,i), adm.alpha(m,k,j,i), adm.beta_u(m,0,k,j,i),
                  adm.beta_u(m,1,k,j,i), adm.beta_u(m,2,k,j,i),
                  adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i), adm.g_dd(m,0,2,k,j,i),
                  adm.g_dd(m,1,1,k,j,i), adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i),
                  adm.vK_dd(m,0,0,k,j,i), adm.vK_dd(m,0,1,k,j,i), adm.vK_dd(m,0,2,k,j,i),
                  adm.vK_dd(m,1,1,k,j,i), adm.vK_dd(m,1,2,k,j,i), adm.vK_dd(m,2,2,k,j,i),
                  w0(m,IDN,k,j,i), w0(m,IPR,k,j,i), w0(m,IVX,k,j,i), w0(m,IVY,k,j,i),
                  w0(m,IVZ,k,j,i), u0(m,IDN,k,j,i), u0(m,IM1,k,j,i), u0(m,IM2,k,j,i),
                  u0(m,IM3,k,j,i), u0(m,IEN,k,j,i)};
    for (int q = 0; q < 35; ++q) buf(r, q) = v[q];
  });
  auto hc = Kokkos::create_mirror_view_and_copy(HostMemSpace(), cnt);
  const int nr = (hc(0) < cap) ? hc(0) : cap;
  auto hb = Kokkos::create_mirror_view_and_copy(HostMemSpace(), buf);
  char fn[64];
  snprintf(fn, sizeof(fn), "cfccheck_rank%05d.txt", global_variable::my_rank);
  FILE *fp = fopen(fn, "a");
  if (fp == nullptr) return;
  for (int r = 0; r < nr; ++r) {
    fprintf(fp, "MPROBE %d %d %d", tag, cyc, stage);
    for (int q = 0; q < 5; ++q) fprintf(fp, " %d", static_cast<int>(hb(r, q)));
    for (int q = 5; q < 35; ++q) fprintf(fp, " %.17g", hb(r, q));
    fprintf(fp, " nan\n");   // (no DY slot on cfc; keeps c057_mprobe.py's 40 fields)
  }
  fclose(fp);
}

}  // namespace cfccheck
#endif  // DYN_GRMHD_CFC_CHECK_DEBUG_HPP_
