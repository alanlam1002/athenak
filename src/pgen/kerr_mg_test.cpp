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
//! <kerr_mg_test>
//!   edges_only  : true -> only the unsafe-edge census of the mesh (no multigrid)
//!   coef        : flat | synthetic | kerr   (synthetic: eps, kw; kerr: kerr_f2,
//!                 kerr_kstab, kerr_row0_exact, m_bh, xbh1..3, exact)
//!   manuf       : 0 sine (zero on the box faces) | 1 amp/sqrt(r^2+w^2) | 2 gaussian;
//!                 amp, w, c, x01..3
//!   op_tests    : the operator checks on the finest block level (exact edges, R, B, off,
//!                 E1 poison with R and with off, corner poison)
//!   runs        : comma-separated smoother/fill pairs, e.g. "gs8/rule_r,rb/rule_r";
//!                 each: u = 0, ncycles V-cycles, defect per cycle, error against u*
//!   e1_cycles   : V-cycles of E1 inside the solver (poison every hook), R then off
//! The multigrid options are read from <kerr_mg> (MGKerrScalarDriver).

#include <algorithm>
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

}  // namespace

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  const std::string B = "kerr_mg_test";

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
  } else {
    std::cout << "### FATAL ERROR: <kerr_mg_test>/coef must be flat, synthetic or kerr"
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
  const RegionSize &ms = pmy_mesh_->mesh_size;
  mf.xmin[0] = ms.x1min; mf.len[0] = ms.x1max - ms.x1min;
  mf.xmin[1] = ms.x2min; mf.len[1] = ms.x2max - ms.x2min;
  mf.xmin[2] = ms.x3min; mf.len[2] = ms.x3max - ms.x3min;

  Kokkos::Timer t0;
  cfc::MGKerrScalarDriver drv(pmbp, pin, "kerr_mg", cs, mf);
  drv.Prepare();
  const double tprep = t0.seconds();
  {
    long c[3];
    drv.EdgeCounts(c);
    const long nmb = SumL(pmbp->nmb_thispack);
    if (Root()) {
      std::printf("KMG SETUP coef %s manuf %d nmb %ld root_mask 0x%03x prepare_s %.3f\n",
                  coef.c_str(), mf.mode, nmb, drv.RootMask(), tprep);
    }
    EdgeCensus(pmbp);
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
    // ErrorVsManuf's rms is rank-local; with one rank it is global
    if (Root()) {
      std::printf("KMG RUN %s ncyc %d s_per_vcycle %.4e err_max %.6e err_rms_rank0 %.6e\n",
                  tok.c_str(), ncyc, tsec/std::max(ncyc, 1), emax, erms);
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
