//========================================================================================
// AthenaXXX astrophysical plasma code
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mg_prof.cpp
//! \brief CFC/multigrid stage profiler, see mg_prof.hpp. Off unless <cfc> mg_log >= 3.

#include "mg_prof.hpp"

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace mgprof {

bool on = false;

namespace {

using Clock = std::chrono::steady_clock;

struct Frame {
  int solver, stage;
  Clock::time_point t0;
  double child;
};

std::vector<Frame> stack_;
double excl_[kNSolver][kNStage] = {};
double incl_[kNSolver][kNStage] = {};
double calls_[kNSolver][kNStage] = {};
double extra_[kNSolver][kNStage] = {};
bool solved_ = false;          // any timer ran since the last CycleWall
double cyc_wall_nosolve_ = 0.0;
int cyc_n_nosolve_ = 0;

const char *kSolverName[kNSolver] = {"X", "psi", "alpha_psi", "beta", "cfc"};
const char *kStageName[kNStage] = {
  "solve_call", "iterative", "vcycle", "tasklist",
  "prepare_amr", "coeff", "setup",
  "smooth", "fas_rhs", "restrict", "prolong", "bnd_send", "bnd_recv", "bnd_other",
  "bnd_start", "bnd_fill", "bnd_prolfc", "bnd_phys", "bnd_clear",
  "norm_dev", "norm_mpi",
  "toroot_dev", "toroot_mpi", "toroot_host", "fromroot_oct", "fromroot_set", "root_sync",
  "octet", "root", "log",
  "c_source", "c_load", "c_retrieve", "c_metric", "c_adual", "c_rescale", "c_recon",
  "c_assemble", "c_exchange", "c_initclear"};

}  // namespace

const char *StageName(int s) { return kStageName[s]; }

int SolverId(const std::string &name) {
  if (name == "X") return kX;
  if (name == "psi") return kPsi;
  if (name == "alpha_psi") return kAlphaPsi;
  if (name == "beta") return kBeta;
  return kCFC;
}

void Start(int solver, int stage) {
  Kokkos::fence();
  solved_ = true;
  stack_.push_back({solver, stage, Clock::now(), 0.0});
}

void Stop() {
  Kokkos::fence();
  Frame f = stack_.back();
  stack_.pop_back();
  double dt = std::chrono::duration<double>(Clock::now() - f.t0).count();
  excl_[f.solver][f.stage] += dt - f.child;
  incl_[f.solver][f.stage] += dt;
  calls_[f.solver][f.stage] += 1.0;
  if (!stack_.empty()) stack_.back().child += dt;
}

void Count(int solver, int stage, long n) {
  extra_[solver][stage] += static_cast<double>(n);
}

void CycleWall(int ncycle, double sec, bool) {
  if (global_variable::my_rank == 0) {
    std::printf("MGPROF_CYC %d wall %.6f solve %d\n", ncycle, sec, solved_ ? 1 : 0);
    std::fflush(stdout);
  }
  if (!solved_) {
    cyc_wall_nosolve_ += sec;
    ++cyc_n_nosolve_;
  }
  solved_ = false;
}

void Report(int ncycle) {
  const int n = static_cast<int>(kNSolver)*static_cast<int>(kNStage);
  std::vector<double> mx(n), sm(n), inc(n);
#if MPI_PARALLEL_ENABLED
  MPI_Reduce(&excl_[0][0], mx.data(), n, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&excl_[0][0], sm.data(), n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&incl_[0][0], inc.data(), n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#else
  for (int i = 0; i < n; ++i) { mx[i] = sm[i] = (&excl_[0][0])[i];
                                inc[i] = (&incl_[0][0])[i]; }
#endif
  if (global_variable::my_rank == 0) {
    const double nr = static_cast<double>(global_variable::nranks);
    std::printf("MGPROF_SET cyc %d nranks %d mean_nosolve_cycle %.6f (n=%d)\n", ncycle,
                global_variable::nranks,
                cyc_n_nosolve_ > 0 ? cyc_wall_nosolve_/cyc_n_nosolve_ : 0.0,
                cyc_n_nosolve_);
    std::printf("MGPROF  solver     stage            excl_mean   excl_max    incl_mean"
                "    calls      extra\n");
    for (int s = 0; s < kNSolver; ++s) {
      double tot = 0.0;
      for (int t = 0; t < kNStage; ++t) {
        int i = s*kNStage + t;
        tot += sm[i]/nr;
        if (calls_[s][t] == 0.0 && extra_[s][t] == 0.0 && mx[i] == 0.0) continue;
        std::printf("MGPROF  %-10s %-15s %10.5f %10.5f %10.5f %9.0f %10.0f\n",
                    kSolverName[s], kStageName[t], sm[i]/nr, mx[i], inc[i]/nr,
                    calls_[s][t], extra_[s][t]);
      }
      std::printf("MGPROF  %-10s %-15s %10.5f\n", kSolverName[s], "TOTAL_excl", tot);
    }
    std::fflush(stdout);
  }
  for (int s = 0; s < kNSolver; ++s) {
    for (int t = 0; t < kNStage; ++t) {
      excl_[s][t] = incl_[s][t] = calls_[s][t] = extra_[s][t] = 0.0;
    }
  }
  cyc_wall_nosolve_ = 0.0;
  cyc_n_nosolve_ = 0;
}

}  // namespace mgprof
