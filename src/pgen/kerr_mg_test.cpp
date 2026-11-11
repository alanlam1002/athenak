//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file kerr_mg_test.cpp
//! \brief W3a/V2a test harness (K-003) for the Kerr variable-coefficient scalar multigrid
//! (src/cfc/mg_kerr_scalar.hpp): no evolution (<time> evolution = static). Everything
//! happens in the problem generator; output lines start with "KMG" for parsing.
//!
//! <problem> (test options)
//!   edges_only  : true -> only the unsafe-edge census of the mesh (no multigrid)
//!   coef        : flat | synthetic | kerr   (synthetic: eps, kw; kerr: kerr_f2,
//!                 kerr_kstab, kerr_row0_exact, m_bh, xbh1..3, exact, src_exact: the
//!                 manufactured source/matter from the spectral background, the operator
//!                 from F2, V2a-e)
//!   manuf       : 0 sine (zero on the box faces) | 1 amp/sqrt(r^2+w^2) | 2 gaussian;
//!                 amp, w, c, x01..3
//!   kind        : linear (c, s) | psi (eq:cpsi) | chi (eq:cchi) -- the driver's option;
//!                 psi/chi: the matter is manufactured (kmg::ManufMatter) from
//!                 da2_amp, da2_w (Ahat^2 - Ahat0^2) and dpsi_amp, dpsi_w (delta psi, chi)
//!   op_tests    : the operator checks on the finest block level (exact edges, R, B, off,
//!                 E1 poison with R and with off, corner poison)
//!   runs        : comma-separated smoother/fill pairs, e.g. "gs8/rule_r,rb/rule_r";
//!                 each: u = 0, ncycles V-cycles, defect per cycle, error against u*
//!   e1_cycles   : V-cycles of E1 inside the solver (poison every hook), R then off
//!   kerr_blend  : D3 test option (BlendNode, kerr_background.hpp): blend gt to flat between
//!                 kerr_blend_f1 (0.25) and kerr_blend_f2 (0.5) times kerr_r_ah (table
//!                 units, max_theta r_AH) about xbh; default false
//!   rad_shells  : comma-separated shell edges (mesh units, about xbh): the error per
//!                 radial shell after each run (KMG ERRR / VERRR)
//!   init_noise  : scalar runs start from u = init_noise * xi (xi in [-1, 1), a hash of
//!                 the cell), the V2a-U zero right-hand-side test (amp = 0)
//!   chk_near    : > 0: also the metric self-check in the cube |x - xbh| < chk_near
//!                 (KMG VCHKN; chk_npts points)
//!   amp2, w2, x2_1..3 : manuf 2 only: a second Gaussian amp2 exp(-|x - x2|^2/w2^2) (K-004 A5)
//!   ssrc        : kind psi/chi: manuf (default) | blob (no u*: the matter channel is the
//!                 manuf field itself, dA2 = delta psi = 0; K-004 A5 blend sensitivity)
//!   dump_u      : non-empty: after each run, the finest block level's u to the binary file
//!                 <dump_u>_<run>.r<rank> (int32 nmb nx ny nz nvar; per block 6 doubles
//!                 x1min x1max x2min x2max x3min x3max then u[v][k][j][i] interior, double)
//! The multigrid options are also read from <problem> (MGKerrScalarDriver), among them
//! hook_scope = operator (default; edge fill / kelvin faces only inside the kernels) |
//! persist (hooked ghosts stay in u; with uold_edge_sync = none | free | all).
//!
//! W3c (driver = vector, MGKerrVectorDriver: -DL V = s on gt, mg_kerr_vector.hpp):
//!   V*'s component c is the scalar manuf with amp*vamp<c+1>, centre x0 + vofs e_c and,
//!   for the sine mode with vnh_mix, harmonics (1,1,1), (2,1,1), (1,1,2);
//!   vsrc        : manuf (s = -DL V*, error against V*) | blob (src = V*: a source with a
//!                 monopole and no exact solution, for the box-doubling probes)
//!   runs        : smoother/fill[/self|rhs] (the last: mp_source for that run); smoother
//!                 gs8 | rb | pbj (damped point-block Jacobi, damping pbj_omega)
//!   probe_r     : > 0: V at 7 points (+-r on the axes, r(1,1,1)/sqrt3) from the mp origin
//!   chk_npts    : the PointMetric2 self-check points (kerr/synthetic)
//!   the driver's options: mg_outer_bc_vec (kelvin | zerofixed), mp_source, mp_outer,
//!   mp_x01..3, smoother, edge_fill, mg_* as for the scalar driver.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "driver/driver.hpp"
#include "pgen/pgen.hpp"
#include "cfc/kerr_background.hpp"
#include "cfc/kerr_mg_ops.hpp"
#include "cfc/mg_kerr_scalar.hpp"
#include "cfc/kerr_mg_vops.hpp"
#include "cfc/mg_kerr_vector.hpp"

namespace {

long SumL(long x) {
#if MPI_PARALLEL_ENABLED
  long y = 0;
  MPI_Allreduce(&x, &y, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  return y;
#else
  return x;
#endif
}
Real SumR(Real x) {
#if MPI_PARALLEL_ENABLED
  Real y = 0;
  MPI_Allreduce(&x, &y, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  return y;
#else
  return x;
#endif
}
Real MaxR(Real x) {
#if MPI_PARALLEL_ENABLED
  Real y = 0;
  MPI_Allreduce(&x, &y, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  return y;
#else
  return x;
#endif
}

bool Root() { return global_variable::my_rank == 0; }

template <class LV>
void DumpFinest(LV *lv, int nvar, const std::string &stem, const std::string &tok) {
  if (stem.empty()) return;
  std::string run = tok;
  std::replace(run.begin(), run.end(), '/', '-');
  const std::string path = stem + "_" + run + ".r" + std::to_string(global_variable::my_rank);
  const int lev = lv->GetNumberOfLevels() - 1;
  const int ngh = lv->GetGhostCells();
  const int hdr[5] = {lv->NumMB(), lv->LevelN(lev, 0), lv->LevelN(lev, 1),
                      lv->LevelN(lev, 2), nvar};
  auto h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), lv->UAt(lev).d_view);
  std::FILE *f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) {
    std::printf("KMG DUMP failed to open %s\n", path.c_str());
    return;
  }
  std::fwrite(hdr, sizeof(int), 5, f);
  std::vector<double> buf(static_cast<size_t>(hdr[1])*hdr[2]*hdr[3]*nvar);
  for (int m = 0; m < hdr[0]; ++m) {
    const RegionSize &b = lv->Extents().h_view(m);
    const double e[6] = {b.x1min, b.x1max, b.x2min, b.x2max, b.x3min, b.x3max};
    std::fwrite(e, sizeof(double), 6, f);
    size_t q = 0;
    for (int v = 0; v < nvar; ++v)
      for (int k = 0; k < hdr[3]; ++k)
        for (int j = 0; j < hdr[2]; ++j)
          for (int i = 0; i < hdr[1]; ++i) buf[q++] = h(m, v, k + ngh, j + ngh, i + ngh);
    std::fwrite(buf.data(), sizeof(double), buf.size(), f);
  }
  std::fclose(f);
  if (Root()) std::printf("KMG DUMP %s nmb %d\n", path.c_str(), hdr[0]);
}

void PrintOp(const char *tag, const cfc::KerrOpStats &s0) {
  cfc::KerrOpStats s;
  for (int c = 0; c < 3; ++c) {
    s.maxd[c] = MaxR(s0.maxd[c]);
    s.l2[c] = SumR(s0.l2[c]);
    s.vol[c] = SumR(s0.vol[c]);
    s.ncell[c] = SumL(s0.ncell[c]);
  }
  s.nonfinite = SumL(s0.nonfinite);
  s.max_s7 = MaxR(s0.max_s7);
  if (!Root()) return;
  const char *nm[3] = {"bulk", "bedge", "iedge"};
  std::printf("KMG OP %-14s nonfinite %ld max|S-S7| %.3e", tag, s.nonfinite, s.max_s7);
  for (int c = 0; c < 3; ++c) {
    const Real l2 = (s.vol[c] > 0.0) ? std::sqrt(s.l2[c]/s.vol[c]) : 0.0;
    std::printf(" | %s n %ld max %.6e l2 %.6e", nm[c], s.ncell[c], s.maxd[c], l2);
  }
  std::printf("\n");
  std::fflush(stdout);
}

void EdgeCensus(MeshBlockPack *pmbp) {
  DualArray1D<int> mask, phys;
  cfc::kmg::ComputeBlockEdgeMasks(pmbp, mask, phys);
  const int nmb = pmbp->nmb_thispack;
  const int nlev = 32;
  std::vector<long> nb(nlev, 0), nbu(nlev, 0), ne(nlev, 0), nbd(nlev, 0), nin(nlev, 0);
  for (int m = 0; m < nmb; ++m) {
    const int l = pmbp->pmb->mb_lev.h_view(m) - pmbp->pmesh->root_level;
    ++nb[l];
    const int msk = mask.h_view(m), phy = phys.h_view(m);
    if (msk) ++nbu[l];
    for (int q = 0; q < 12; ++q) {
      if (!((msk >> q) & 1)) continue;
      ++ne[l];
      if ((phy >> q) & 1) { ++nbd[l]; } else { ++nin[l]; }
    }
  }
  long tot[5] = {0, 0, 0, 0, 0};
  for (int l = 0; l < nlev; ++l) {
    const long v[5] = {SumL(nb[l]), SumL(nbu[l]), SumL(ne[l]), SumL(nbd[l]), SumL(nin[l])};
    for (int c = 0; c < 5; ++c) tot[c] += v[c];
    if (v[0] > 0 && Root()) {
      std::printf("KMG EDGES level %d blocks %ld blocks_unsafe %ld edges_unsafe %ld "
                  "boundary %ld interior %ld\n", l, v[0], v[1], v[2], v[3], v[4]);
    }
  }
  if (Root()) {
    std::printf("KMG EDGES total blocks %ld blocks_unsafe %ld edges_unsafe %ld boundary %ld "
                "interior %ld (of %ld block edges)\n", tot[0], tot[1], tot[2], tot[3],
                tot[4], 12*tot[0]);
    std::fflush(stdout);
  }
}

void PrintDefs(const char *tag, const std::string &tok, const std::vector<Real> &d) {
  std::printf("KMG %s %s", tag, tok.c_str());
  for (size_t n = 0; n < d.size(); ++n) std::printf(" %.6e", d[n]);
  std::printf("\nKMG %sRHO %s", tag, tok.c_str());
  for (size_t n = 1; n < d.size(); ++n) {
    std::printf(" %.4f", (d[n-1] > 0.0) ? d[n]/d[n-1] : 0.0);
  }
  std::printf("\n");
}

//! max |V_K| over the 6 face centres and 8 corners of the mesh, and max |V*| there
void KelvinOnBox(const cfc::MGKerrVectorDriver &drv, const RegionSize &ms, Real &vk,
                 Real &vs) {
  vk = vs = 0.0;
  const Real lo[3] = {ms.x1min, ms.x2min, ms.x3min}, hi[3] = {ms.x1max, ms.x2max, ms.x3max};
  std::vector<std::array<Real, 3>> P;
  for (int d = 0; d < 3; ++d) {
    for (int sd = 0; sd < 2; ++sd) {
      std::array<Real, 3> x;
      for (int e = 0; e < 3; ++e) x[e] = 0.5*(lo[e] + hi[e]);
      x[d] = sd ? hi[d] : lo[d];
      P.push_back(x);
    }
  }
  for (int c = 0; c < 8; ++c) {
    P.push_back({(c & 1) ? hi[0] : lo[0], (c & 2) ? hi[1] : lo[1], (c & 4) ? hi[2] : lo[2]});
  }
  for (const auto &x : P) {
    Real V[3];
    cfc::kmg::KelvinV(drv.mom_, x[0] - drv.mpo_[0], x[1] - drv.mpo_[1], x[2] - drv.mpo_[2],
                      V);
    for (int c = 0; c < 3; ++c) {
      Real us, du[3], ddu[6];
      cfc::kmg::ManufEval(drv.vm_.c[c], x[0], x[1], x[2], us, du, ddu);
      vk = std::max(vk, std::abs(V[c]));
      vs = std::max(vs, std::abs(us));
    }
  }
}

std::vector<Real> ParseList(const std::string &str) {
  std::vector<Real> v;
  std::stringstream ss(str);
  std::string x;
  while (std::getline(ss, x, ',')) {
    if (!x.empty()) v.push_back(std::stod(x));
  }
  return v;
}

//! the error per radial shell about xbh (drv.ErrorRadial), one line per run
template <class Drv>
void PrintRadial(const char *tag, const std::string &tok, Drv &drv,
                 const cfc::kmg::CoefSource &cs, const std::vector<Real> &e) {
  if (e.empty()) return;
  std::vector<Real> em, s2, vol;
  std::vector<long> n;
  drv.ErrorRadial(cs.xbh, e, em, s2, vol, n);
  if (Root()) std::printf("KMG %s %s", tag, tok.c_str());
  for (size_t q = 0; q < e.size(); ++q) {
    const Real m = MaxR(em[q]), a = SumR(s2[q]), w = SumR(vol[q]);
    const long c = SumL(n[q]);
    if (Root()) {
      std::printf(" | r>=%.4g n %ld max %.6e l2 %.6e", e[q], c, m,
                  (w > 0.0) ? std::sqrt(a/w) : 0.0);
    }
  }
  if (Root()) {
    std::printf("\n");
    std::fflush(stdout);
  }
}

//! the metric self-check (PointMetric2 against PointMetric and its FD) in a cube
void MetricCheck(const char *tag, const cfc::kmg::CoefSource &cs, const Real lo[3],
                 const Real len[3], int np) {
  for (Real fd : {1.0e-3, 1.0e-4}) {
    Real o[4];
    cfc::kmg::MetricSelfCheck(cs, lo, len, np, fd, o);
    if (Root()) {
      std::printf("KMG %s npts %d fd %.1e max|dgt2-dgt| %.3e max|ddgt-FD| %.3e "
                  "max|ddgt| %.3e max|gt2-gt| %.3e\n", tag, np, fd, o[0], o[1], o[2], o[3]);
      std::fflush(stdout);
    }
  }
}

void RunVector(ParameterInput *pin, MeshBlockPack *pmbp, Mesh *pm,
               const cfc::kmg::CoefSource &cs, const cfc::kmg::Manuf &mf,
               const std::string &coef) {
  const std::string B = "problem";
  cfc::kmg::VManuf vm;
  const Real vamp[3] = {pin->GetOrAddReal(B, "vamp1", 1.0), pin->GetOrAddReal(B, "vamp2", 0.7),
                        pin->GetOrAddReal(B, "vamp3", -0.5)};
  const Real vofs = pin->GetOrAddReal(B, "vofs", 0.0);
  const bool nhmix = pin->GetOrAddBoolean(B, "vnh_mix", true);
  for (int c = 0; c < 3; ++c) {
    vm.c[c] = mf;
    vm.c[c].amp = mf.amp*vamp[c];
    vm.c[c].amp2 = mf.amp2*vamp[c];
    vm.c[c].x0[c] += vofs;
  }
  if (nhmix) {
    vm.c[1].nh[0] = 2.0;
    vm.c[2].nh[2] = 2.0;
  }
  const std::string vsrc = pin->GetOrAddString(B, "vsrc", "manuf");

  Kokkos::Timer t0;
  cfc::MGKerrVectorDriver drv(pmbp, pin, B, cs, vm);
  drv.src_blob_ = (vsrc == "blob");
  drv.Prepare();
  const double tprep = t0.seconds();
  const RegionSize &ms = pm->mesh_size;
  {
    long bd[3] = {SumL(drv.bad_dinv_[0]), SumL(drv.bad_dinv_[1]), SumL(drv.bad_dinv_[2])};
    const long nmb = SumL(pmbp->nmb_thispack);
    if (Root()) {
      std::printf("KMG VSETUP coef %s manuf %d vsrc %s nmb %ld root_mask 0x%03x outer %s "
                  "mp_source %s mp_origin %.4f %.4f %.4f bad_dinv %ld %ld %ld "
                  "boundary_octets %ld prepare_s %.3f\n", coef.c_str(), mf.mode,
                  vsrc.c_str(), nmb, drv.RootMask(), drv.kelvin_ ? "kelvin" : "zerofixed",
                  drv.mp_self_ ? "self" : "rhs", drv.mpo_[0], drv.mpo_[1], drv.mpo_[2],
                  bd[0], bd[1], bd[2], drv.boundary_octets_, tprep);
    }
    EdgeCensus(pmbp);
  }
  if (cs.mode != 0) {
    const int np = pin->GetOrAddInteger(B, "chk_npts", 20000);
    const Real lo[3] = {ms.x1min, ms.x2min, ms.x3min};
    const Real len[3] = {ms.x1max - ms.x1min, ms.x2max - ms.x2min, ms.x3max - ms.x3min};
    MetricCheck("VCHK", cs, lo, len, np);
    const Real cn = pin->GetOrAddReal(B, "chk_near", 0.0);
    if (cn > 0.0) {
      const Real lon[3] = {cs.xbh[0] - cn, cs.xbh[1] - cn, cs.xbh[2] - cn};
      const Real lenn[3] = {2.0*cn, 2.0*cn, 2.0*cn};
      MetricCheck("VCHKN", cs, lon, lenn, np);
    }
  }
  const std::vector<Real> shells = ParseList(pin->GetOrAddString(B, "rad_shells", ""));

  Kokkos::Timer timer;
  Driver dv(pin, pm, 1.0e30, &timer);

  // ---- operator checks (manufactured source only)
  if (vsrc == "manuf" && pin->GetOrAddBoolean(B, "op_tests", true)) {
    const auto fill0 = drv.edge_fill_;
    using EF = cfc::kmg::EdgeFill;
    PrintOp("v_exact", drv.OperatorTest(true, false));
    drv.edge_fill_ = EF::rule_r;   PrintOp("v_rule_r", drv.OperatorTest(false, false));
    drv.edge_fill_ = EF::bilinear; PrintOp("v_bilinear", drv.OperatorTest(false, false));
    drv.edge_fill_ = EF::off;      PrintOp("v_off_stale", drv.OperatorTest(false, false));
    drv.edge_poison_ = true;
    drv.edge_fill_ = EF::rule_r;   PrintOp("v_E1_rule_r", drv.OperatorTest(false, false));
    drv.edge_fill_ = EF::off;      PrintOp("v_E1_off", drv.OperatorTest(false, false));
    drv.edge_poison_ = false;
    drv.edge_fill_ = EF::rule_r;   PrintOp("v_corner_r", drv.OperatorTest(false, true));
    drv.edge_fill_ = fill0;
    // the kelvin field of q from u = V*: self (q = -DL_f V*, its far field is V* ~ 0
    // outside a compact V*) against rhs (q = s)
    const bool self0 = drv.mp_self_;
    for (int sm = 0; sm < 2; ++sm) {
      drv.ResetSolution();
      drv.OperatorTest(true, false);   // finest u = V* (ghosts exact)
      drv.mp_self_ = (sm == 0);
      drv.ComputeMoments();
      Real vk, vs;
      KelvinOnBox(drv, ms, vk, vs);
      Real qn = 0.0, dn = 0.0, qq = 0.0;
      for (int q = 0; q < 3; ++q) qn = std::max(qn, std::abs(drv.mom_.v[q]));
      for (int q = 3; q < 12; ++q) dn = std::max(dn, std::abs(drv.mom_.v[q]));
      for (int q = 12; q < 39; ++q) qq = std::max(qq, std::abs(drv.mom_.v[q]));
      if (Root()) {
        std::printf("KMG MOM ustar %s max|Q| %.3e max|D| %.3e max|Qab| %.3e "
                    "box max|V_K| %.3e max|V*| %.3e\n", sm == 0 ? "self" : "rhs", qn, dn,
                    qq, vk, vs);
      }
    }
    drv.mp_self_ = self0;
  }

  // ---- V-cycle runs
  const int ncyc = pin->GetOrAddInteger(B, "ncycles", 10);
  std::vector<Real> probes;
  const Real pr = pin->GetOrAddReal(B, "probe_r", 0.0);
  if (pr > 0.0) {
    const Real q3 = pr/std::sqrt(3.0);
    const Real D[7][3] = {{pr, 0, 0}, {0, pr, 0}, {0, 0, pr}, {-pr, 0, 0}, {0, -pr, 0},
                          {0, 0, -pr}, {q3, q3, q3}};
    for (int p = 0; p < 7; ++p) {
      for (int d = 0; d < 3; ++d) probes.push_back(drv.mpo_[d] + D[p][d]);
    }
  }
  std::string runs = pin->GetOrAddString(B, "runs", "gs8/rule_r");
  std::stringstream ss(runs);
  std::string tok;
  const bool self0 = drv.mp_self_;
  while (std::getline(ss, tok, ',')) {
    std::vector<std::string> f;
    {
      std::stringstream st(tok);
      std::string x;
      while (std::getline(st, x, '/')) f.push_back(x);
    }
    if (f.size() < 2) continue;
    drv.smoother_ = (f[0] == "rb") ? 1 : ((f[0] == "pbj") ? 2 : 0);
    drv.edge_fill_ = (f[1] == "bilinear") ? cfc::kmg::EdgeFill::bilinear
                   : ((f[1] == "off") ? cfc::kmg::EdgeFill::off
                                      : cfc::kmg::EdgeFill::rule_r);
    drv.mp_self_ = (f.size() > 2) ? (f[2] != "rhs") : self0;
    drv.edge_poison_ = false;
    drv.count_nonfinite_ = false;
    drv.ResetSolution();
    Kokkos::fence();
    Kokkos::Timer tv;
    std::vector<Real> d = drv.RunCycles(&dv, ncyc);
    const double tsec = tv.seconds();
    Real cmax[3], crms[3];
    long cn[3];
    drv.ErrorVsManufByClass(cmax, crms, cn);
    for (int c = 0; c < 3; ++c) {
      cmax[c] = MaxR(cmax[c]);
      cn[c] = SumL(cn[c]);
    }
    PrintRadial("VERRR", tok, drv, cs, shells);
    DumpFinest(drv.Levels(), 3, pin->GetOrAddString(B, "dump_u", ""), tok);
    std::vector<Real> pv, pc;
    std::vector<int> po;
    if (!probes.empty()) {
      drv.Probe(probes, pv, pc, po);
      for (auto &x : pv) x = SumR(x);
      for (auto &x : pc) x = SumR(x);
      for (auto &x : po) x = static_cast<int>(SumL(x));
    }
    if (Root()) {
      std::printf("KMG VRUN %s ncyc %d s_per_vcycle %.4e err_max %.6e\n", tok.c_str(), ncyc,
                  tsec/std::max(ncyc, 1), std::max(cmax[0], std::max(cmax[1], cmax[2])));
      std::printf("KMG VERRC %s bulk %.6e %.6e %ld bedge %.6e %.6e %ld iedge %.6e %.6e "
                  "%ld\n", tok.c_str(), cmax[0], crms[0], cn[0], cmax[1], crms[1], cn[1],
                  cmax[2], crms[2], cn[2]);
      PrintDefs("VDEF", tok, d);
      Real qn = 0.0;
      for (int q = 0; q < 3; ++q) qn = std::max(qn, std::abs(drv.mom_.v[q]));
      std::printf("KMG VMOMQ %s max|Q| %.6e Q %.6e %.6e %.6e\n", tok.c_str(), qn,
                  drv.mom_.v[0], drv.mom_.v[1], drv.mom_.v[2]);
      for (size_t p = 0; p < po.size(); ++p) {
        std::printf("KMG VPROBE %s p %zu own %d x %.6f %.6f %.6f V %.10e %.10e %.10e\n",
                    tok.c_str(), p, po[p], (po[p] ? pc[3*p]/po[p] : 0.0),
                    (po[p] ? pc[3*p + 1]/po[p] : 0.0), (po[p] ? pc[3*p + 2]/po[p] : 0.0),
                    pv[3*p], pv[3*p + 1], pv[3*p + 2]);
      }
      std::fflush(stdout);
    }
  }
  drv.mp_self_ = self0;

  // ---- E1 inside the solver
  const int ne1 = pin->GetOrAddInteger(B, "e1_cycles", 3);
  if (ne1 > 0) {
    const cfc::kmg::EdgeFill fills[2] = {cfc::kmg::EdgeFill::rule_r,
                                         cfc::kmg::EdgeFill::off};
    const char *names[2] = {"rule_r", "off"};
    for (int f = 0; f < 2; ++f) {
      drv.smoother_ = 0;
      drv.edge_fill_ = fills[f];
      drv.edge_poison_ = true;
      drv.count_nonfinite_ = true;
      drv.ResetSolution();
      std::vector<Real> d = drv.RunCycles(&dv, ne1);
      long c[3];
      drv.NonFiniteCounts(c);
      const long cb = SumL(c[0]), cr = SumL(c[1]), co = SumL(c[2]);
      if (Root()) {
        std::printf("KMG VE1 %s nonfinite_defects blocks %ld root %ld octets %ld | def",
                    names[f], cb, cr, co);
        for (size_t n = 0; n < d.size(); ++n) std::printf(" %.6e", d[n]);
        std::printf("\n");
        std::fflush(stdout);
      }
    }
  }
  if (Root()) {
    std::printf("KMG DONE\n");
    std::fflush(stdout);
  }
}

}  // namespace

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  // AthenaK accepts only whitelisted block names (parameter_input.cpp:124), so the test
  // and driver options share <problem>.
  const std::string B = "problem";

  // a valid (uniform) fluid so that the static driver's initialisation has a state
  if (pmbp->phydro != nullptr) {
    auto u0 = pmbp->phydro->u0;
    const Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;
    const bool ideal = pmbp->phydro->peos->eos_data.is_ideal;
    par_for("kmg_fluid", DevExeSpace(), 0, pmbp->nmb_thispack-1, 0,
            static_cast<int>(u0.extent(2))-1, 0, static_cast<int>(u0.extent(3))-1, 0,
            static_cast<int>(u0.extent(4))-1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      u0(m, IDN, k, j, i) = 1.0;
      u0(m, IM1, k, j, i) = 0.0;
      u0(m, IM2, k, j, i) = 0.0;
      u0(m, IM3, k, j, i) = 0.0;
      if (ideal) u0(m, IEN, k, j, i) = 1.0/gm1;
    });
  }

  if (pin->GetOrAddBoolean(B, "edges_only", false)) {
    EdgeCensus(pmbp);
    return;
  }

  // ---- metric
  cfc::kmg::CoefSource cs;
  const std::string coef = pin->GetOrAddString(B, "coef", "flat");
  if (coef == "flat") {
    cs.mode = 0;
  } else if (coef == "synthetic") {
    cs.mode = 1;
    cs.eps = pin->GetOrAddReal(B, "eps", 0.2);
    cs.kw = pin->GetOrAddReal(B, "kw", 1.0);
  } else if (coef == "kerr") {
    cs.mode = 2;
    cfc::LoadKerrTable(pin->GetString(B, "kerr_f2"), pin->GetString(B, "kerr_kstab"),
                       cs.tab);
    if (pin->GetOrAddBoolean(B, "kerr_row0_exact", true)) cfc::KerrSetRow0Exact(cs.tab);
    cs.m_bh = pin->GetOrAddReal(B, "m_bh", 1.0);
    cs.xbh[0] = pin->GetOrAddReal(B, "xbh1", 0.0);
    cs.xbh[1] = pin->GetOrAddReal(B, "xbh2", 0.0);
    cs.xbh[2] = pin->GetOrAddReal(B, "xbh3", 0.0);
    cs.exact = pin->GetOrAddBoolean(B, "exact", false);
    cs.src_exact = pin->GetOrAddBoolean(B, "src_exact", false);
    if (pin->GetOrAddBoolean(B, "kerr_blend", false)) {
      const Real rah = pin->GetReal(B, "kerr_r_ah");
      cs.blend_r1 = pin->GetOrAddReal(B, "kerr_blend_f1", 0.25)*rah*cs.m_bh;
      cs.blend_r2 = pin->GetOrAddReal(B, "kerr_blend_f2", 0.5)*rah*cs.m_bh;
      if (!(cs.blend_r2 > cs.blend_r1) || cs.blend_r1 < 0.0) {
        std::cout << "### FATAL ERROR: kerr_blend needs 0 <= f1 < f2" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (Root()) {
        std::printf("KMG BLEND r1 %.6e r2 %.6e (mesh units about xbh)\n", cs.blend_r1,
                    cs.blend_r2);
      }
    }
  } else {
    std::cout << "### FATAL ERROR: <problem>/coef must be flat, synthetic or kerr"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // ---- manufactured solution
  cfc::kmg::Manuf mf;
  mf.mode = pin->GetOrAddInteger(B, "manuf", 0);
  mf.amp = pin->GetOrAddReal(B, "amp", 1.0);
  mf.w = pin->GetOrAddReal(B, "w", 1.0);
  mf.c = pin->GetOrAddReal(B, "c", 0.0);
  mf.x0[0] = pin->GetOrAddReal(B, "x01", 0.0);
  mf.x0[1] = pin->GetOrAddReal(B, "x02", 0.0);
  mf.x0[2] = pin->GetOrAddReal(B, "x03", 0.0);
  mf.amp2 = pin->GetOrAddReal(B, "amp2", 0.0);
  mf.w2 = pin->GetOrAddReal(B, "w2", 1.0);
  mf.x2[0] = pin->GetOrAddReal(B, "x2_1", 0.0);
  mf.x2[1] = pin->GetOrAddReal(B, "x2_2", 0.0);
  mf.x2[2] = pin->GetOrAddReal(B, "x2_3", 0.0);
  const RegionSize &ms = pmy_mesh_->mesh_size;
  mf.xmin[0] = ms.x1min; mf.len[0] = ms.x1max - ms.x1min;
  mf.xmin[1] = ms.x2min; mf.len[1] = ms.x2max - ms.x2min;
  mf.xmin[2] = ms.x3min; mf.len[2] = ms.x3max - ms.x3min;

  if (pin->GetOrAddString(B, "driver", "scalar") == "vector") {
    RunVector(pin, pmbp, pmy_mesh_, cs, mf, coef);
    return;
  }

  Kokkos::Timer t0;
  cfc::MGKerrScalarDriver drv(pmbp, pin, B, cs, mf);
  drv.mm_.da2_amp = pin->GetOrAddReal(B, "da2_amp", 0.0);
  drv.mm_.da2_w = pin->GetOrAddReal(B, "da2_w", 1.0);
  drv.mm_.dpsi_amp = pin->GetOrAddReal(B, "dpsi_amp", 0.0);
  drv.mm_.dpsi_w = pin->GetOrAddReal(B, "dpsi_w", 1.0);
  drv.mm_.blob = (pin->GetOrAddString(B, "ssrc", "manuf") == "blob");
  drv.init_noise_ = pin->GetOrAddReal(B, "init_noise", 0.0);
  const std::vector<Real> shells = ParseList(pin->GetOrAddString(B, "rad_shells", ""));
  drv.Prepare();
  const double tprep = t0.seconds();
  {
    long c[3];
    drv.EdgeCounts(c);
    const long nmb = SumL(pmbp->nmb_thispack);
    if (Root()) {
      std::printf("KMG SETUP coef %s manuf %d kind %d nmb %ld root_mask 0x%03x "
                  "prepare_s %.3f\n", coef.c_str(), mf.mode, drv.kind_, nmb, drv.RootMask(),
                  tprep);
    }
    EdgeCensus(pmbp);
  }
  {
    const Real cn = pin->GetOrAddReal(B, "chk_near", 0.0);
    if (cn > 0.0 && cs.mode != 0) {
      const int np = pin->GetOrAddInteger(B, "chk_npts", 20000);
      const Real lon[3] = {cs.xbh[0] - cn, cs.xbh[1] - cn, cs.xbh[2] - cn};
      const Real lenn[3] = {2.0*cn, 2.0*cn, 2.0*cn};
      MetricCheck("VCHKN", cs, lon, lenn, np);
    }
  }

  Kokkos::Timer timer;
  Driver dv(pin, pmy_mesh_, 1.0e30, &timer);

  // ---- operator checks on the finest block level
  if (pin->GetOrAddBoolean(B, "op_tests", true)) {
    const auto fill0 = drv.edge_fill_;
    using EF = cfc::kmg::EdgeFill;
    PrintOp("exact", drv.OperatorTest(true, false));
    drv.edge_fill_ = EF::rule_r;   PrintOp("rule_r", drv.OperatorTest(false, false));
    drv.edge_fill_ = EF::bilinear; PrintOp("bilinear", drv.OperatorTest(false, false));
    drv.edge_fill_ = EF::off;      PrintOp("off_stale", drv.OperatorTest(false, false));
    drv.edge_poison_ = true;
    drv.edge_fill_ = EF::rule_r;   PrintOp("E1_rule_r", drv.OperatorTest(false, false));
    drv.edge_fill_ = EF::off;      PrintOp("E1_off", drv.OperatorTest(false, false));
    drv.edge_poison_ = false;
    drv.edge_fill_ = EF::rule_r;   PrintOp("corner_rule_r", drv.OperatorTest(false, true));
    drv.edge_fill_ = fill0;
  }

  // ---- V-cycle runs
  const int ncyc = pin->GetOrAddInteger(B, "ncycles", 10);
  std::string runs = pin->GetOrAddString(B, "runs", "gs8/rule_r");
  std::stringstream ss(runs);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    const auto p = tok.find('/');
    if (p == std::string::npos) continue;
    const std::string sm = tok.substr(0, p), ef = tok.substr(p + 1);
    drv.smoother_ = (sm == "rb") ? 1 : 0;
    drv.edge_fill_ = (ef == "bilinear") ? cfc::kmg::EdgeFill::bilinear
                   : ((ef == "off") ? cfc::kmg::EdgeFill::off
                                    : cfc::kmg::EdgeFill::rule_r);
    drv.edge_poison_ = false;
    drv.count_nonfinite_ = false;
    drv.ResetSolution();
    Kokkos::fence();
    Kokkos::Timer tv;
    std::vector<Real> d = drv.RunCycles(&dv, ncyc);
    const double tsec = tv.seconds();
    Real emax, erms;
    drv.ErrorVsManuf(emax, erms);
    emax = MaxR(emax);
    Real cmax[3], crms[3];
    long cn[3];
    drv.ErrorVsManufByClass(cmax, crms, cn);
    for (int c = 0; c < 3; ++c) {
      cmax[c] = MaxR(cmax[c]);
      cn[c] = SumL(cn[c]);
    }
    PrintRadial("ERRR", tok, drv, cs, shells);
    DumpFinest(drv.Levels(), 1, pin->GetOrAddString(B, "dump_u", ""), tok);
    // ErrorVsManuf's rms is rank-local; with one rank it is global
    if (Root()) {
      std::printf("KMG RUN %s ncyc %d s_per_vcycle %.4e err_max %.6e err_rms_rank0 %.6e\n",
                  tok.c_str(), ncyc, tsec/std::max(ncyc, 1), emax, erms);
      std::printf("KMG ERRC %s bulk %.6e %.6e %ld bedge %.6e %.6e %ld iedge %.6e %.6e %ld\n",
                  tok.c_str(), cmax[0], crms[0], cn[0], cmax[1], crms[1], cn[1], cmax[2],
                  crms[2], cn[2]);
      std::printf("KMG DEF %s", tok.c_str());
      for (size_t n = 0; n < d.size(); ++n) std::printf(" %.6e", d[n]);
      std::printf("\nKMG RHO %s", tok.c_str());
      for (size_t n = 1; n < d.size(); ++n) {
        std::printf(" %.4f", (d[n-1] > 0.0) ? d[n]/d[n-1] : 0.0);
      }
      std::printf("\n");
      std::fflush(stdout);
    }
  }

  // ---- E1 inside the solver: every hook poisons the unsafe edge ghosts with NaN first
  const int ne1 = pin->GetOrAddInteger(B, "e1_cycles", 3);
  if (ne1 > 0) {
    const cfc::kmg::EdgeFill fills[2] = {cfc::kmg::EdgeFill::rule_r,
                                         cfc::kmg::EdgeFill::off};
    const char *names[2] = {"rule_r", "off"};
    for (int f = 0; f < 2; ++f) {
      drv.smoother_ = 0;
      drv.edge_fill_ = fills[f];
      drv.edge_poison_ = true;
      drv.count_nonfinite_ = true;
      drv.ResetSolution();
      std::vector<Real> d = drv.RunCycles(&dv, ne1);
      long c[3];
      drv.NonFiniteCounts(c);
      const long cb = SumL(c[0]), cr = SumL(c[1]), co = SumL(c[2]);
      if (Root()) {
        std::printf("KMG E1 %s nonfinite_defects blocks %ld root %ld octets %ld | def",
                    names[f], cb, cr, co);
        for (size_t n = 0; n < d.size(); ++n) std::printf(" %.6e", d[n]);
        std::printf("\n");
        std::fflush(stdout);
      }
    }
    drv.edge_poison_ = false;
    drv.count_nonfinite_ = false;
  }
  if (Root()) {
    std::printf("KMG DONE\n");
    std::fflush(stdout);
  }
}
