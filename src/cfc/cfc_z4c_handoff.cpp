//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cfc_z4c_handoff.cpp
//! \brief one-way, in-run switch of the spacetime solver from CFC to Z4c. See
//! cfc_z4c_handoff.hpp and DEVELOPMENT.md item 64.

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "outputs/outputs.hpp"
#include "coordinates/adm.hpp"
#include "z4c/z4c.hpp"
#include "z4c/cce/cce.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "cfc/cfc.hpp"
#include "utils/finite_diff.hpp"
#include "cfc/cfc_z4c_handoff.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace cfc {

//----------------------------------------------------------------------------------------
//! \fn HandoffState GetHandoffState(ParameterInput *pin)

HandoffState GetHandoffState(ParameterInput *pin) {
  if (!pin->DoesBlockExist("cfc")) {
    return HandoffState::none;
  }
  bool done = pin->DoesParameterExist("cfc", "handoff_done") &&
              pin->GetBoolean("cfc", "handoff_done");
  bool requested = pin->DoesParameterExist("cfc", "handoff") &&
                   pin->GetBoolean("cfc", "handoff");
  if (!done && !requested) {
    return HandoffState::none;
  }
  if (!pin->DoesBlockExist("z4c")) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<cfc>/handoff = true needs a <z4c> block (the Z4c evolution the run "
              << "switches to)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return done ? HandoffState::done : HandoffState::pending;
}

//----------------------------------------------------------------------------------------
// Device kernels. Kept as free functions: nvcc rejects KOKKOS_LAMBDA inside private
// member functions.

//! RMS conformal-flatness residual (see Z4cHandoff::CFResidual). Returns the ratio.
template <int NGHOST>
Real ComputeCFResidual(MeshBlockPack *pmbp, Real cf_floor) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  int ndim = pmbp->pmesh->one_d ? 1 : (pmbp->pmesh->two_d ? 2 : 3);
  auto &adm = pmbp->padm->adm;

  array_sum::GlobalSum sum;
  Kokkos::parallel_reduce("cfc_handoff_cf_residual",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real g[3][3], gu[3][3];
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        g[a][b] = adm.g_dd(m, a, b, k, j, i);
      }
    }
    Real detg = adm::SpatialDet(g[0][0], g[0][1], g[0][2], g[1][1], g[1][2], g[2][2]);
    Real uxx, uxy, uxz, uyy, uyz, uzz;
    adm::SpatialInv(1.0/detg, g[0][0], g[0][1], g[0][2], g[1][1], g[1][2], g[2][2],
                    &uxx, &uxy, &uxz, &uyy, &uyz, &uzz);
    gu[0][0] = uxx; gu[0][1] = uxy; gu[0][2] = uxz;
    gu[1][0] = uxy; gu[1][1] = uyy; gu[1][2] = uyz;
    gu[2][0] = uxz; gu[2][1] = uyz; gu[2][2] = uzz;

    Real idx_[] = {1./size.d_view(m).dx1, 1./size.d_view(m).dx2, 1./size.d_view(m).dx3};
    Real dbeta[3][3] = {};   // dbeta[b][a] = d_b beta^a
    Real dg[3][3][3] = {};   // dg[c][a][b] = d_c g_ab
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < ndim; ++b) {
        dbeta[b][a] = Dx<NGHOST>(b, idx_, adm.beta_u, m, a, k, j, i);
      }
    }
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        for (int c = 0; c < ndim; ++c) {
          dg[c][a][b] = Dx<NGHOST>(c, idx_, adm.g_dd, m, a, b, k, j, i);
        }
      }
    }
    Real beta[3] = {adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
                    adm.beta_u(m, 2, k, j, i)};
    Real alpha = adm.alpha(m, k, j, i);

    // (L_beta gamma)_ab = beta^c d_c g_ab + g_ac d_b beta^c + g_bc d_a beta^c
    Real Lg[3][3];
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        Real s = 0.0;
        for (int c = 0; c < 3; ++c) {
          s += beta[c]*dg[c][a][b] + g[a][c]*dbeta[b][c] + g[b][c]*dbeta[a][c];
        }
        Lg[a][b] = s;
      }
    }
    Real tr = 0.0;
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        tr += gu[a][b]*Lg[a][b];
      }
    }
    Real R[3][3], S[3][3];
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        S[a][b] = 2.0*alpha*adm.vK_dd(m, a, b, k, j, i);
        R[a][b] = Lg[a][b] - tr/3.0*g[a][b] - S[a][b];
      }
    }
    Real nR = 0.0, nS = 0.0;
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        for (int c = 0; c < 3; ++c) {
          for (int d = 0; d < 3; ++d) {
            nR += gu[a][c]*gu[b][d]*R[a][b]*R[c][d];
            nS += gu[a][c]*gu[b][d]*S[a][b]*S[c][d];
          }
        }
      }
    }
    Real dv = sqrt(fabs(detg))*size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    array_sum::GlobalSum loc;
    loc.the_array[0] = dv*nR;
    loc.the_array[1] = dv*nS;
    loc.the_array[2] = dv;
    mb_sum += loc;
  }, Kokkos::Sum<array_sum::GlobalSum>(sum));

  Real s[3] = {sum.the_array[0], sum.the_array[1], sum.the_array[2]};
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, s, 3, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (s[2] <= 0.0) return 0.0;
  Real num = std::sqrt(s[0]/s[2]);
  Real den = std::sqrt(s[1]/s[2]);
  return num/std::max(den, cf_floor);
}

template Real ComputeCFResidual<2>(MeshBlockPack *pmbp, Real cf_floor);
template Real ComputeCFResidual<3>(MeshBlockPack *pmbp, Real cf_floor);
template Real ComputeCFResidual<4>(MeshBlockPack *pmbp, Real cf_floor);

//! Interior max |u_new - u_old| over [n0, n1) channels of two ADM arrays.
Real MaxAbsDiff(MeshBlockPack *pmbp, const DvceArray5D<Real> &u_old,
                const DvceArray5D<Real> &u_new, int n0, int n1) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  Real dmax = 0.0;
  Kokkos::parallel_reduce("cfc_handoff_roundtrip",
  Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>({0, ks, js, is},
                                                      {nmb, ke+1, je+1, ie+1}),
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, Real &lmax) {
    for (int n = n0; n < n1; ++n) {
      lmax = fmax(lmax, fabs(u_new(m, n, k, j, i) - u_old(m, n, k, j, i)));
    }
  }, Kokkos::Max<Real>(dmax));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &dmax, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  return dmax;
}

//! Volume-weighted L2 norms of the Hamiltonian and momentum constraints (as in the
//! Z4c history output) and max|H|: out = {||H||_2, ||M||_2, max|H|}.
void ConstraintNorms(MeshBlockPack *pmbp, Real out[3]) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  auto &adm = pmbp->padm->adm;
  auto &u_con = pmbp->pz4c->u_con;
  const int iH = z4c::Z4c::I_CON_H, iM = z4c::Z4c::I_CON_M;

  array_sum::GlobalSum sum;
  Kokkos::parallel_reduce("cfc_handoff_constraints",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    Real detg = adm::SpatialDet(adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
                                adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
                                adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i));
    Real dv = sqrt(fabs(detg))*size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    array_sum::GlobalSum loc;
    loc.the_array[0] = dv*SQR(u_con(m, iH, k, j, i));
    loc.the_array[1] = dv*u_con(m, iM, k, j, i);   // M comes already squared
    loc.the_array[2] = dv;
    mb_sum += loc;
  }, Kokkos::Sum<array_sum::GlobalSum>(sum));

  int is_ = indcs.is, ie = indcs.ie, js_ = indcs.js, je = indcs.je;
  int ks_ = indcs.ks, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  Real hmax = 0.0;
  Kokkos::parallel_reduce("cfc_handoff_hmax",
  Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>({0, ks_, js_, is_},
                                                      {nmb, ke+1, je+1, ie+1}),
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, Real &lmax) {
    lmax = fmax(lmax, fabs(u_con(m, iH, k, j, i)));
  }, Kokkos::Max<Real>(hmax));

  Real s[3] = {sum.the_array[0], sum.the_array[1], sum.the_array[2]};
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, s, 3, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &hmax, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  out[0] = (s[2] > 0.0) ? std::sqrt(s[0]/s[2]) : 0.0;
  out[1] = (s[2] > 0.0) ? std::sqrt(std::fabs(s[1])/s[2]) : 0.0;
  out[2] = hmax;
}

//----------------------------------------------------------------------------------------
//! \fn Z4cHandoff::Z4cHandoff()

Z4cHandoff::Z4cHandoff(MeshBlockPack *ppack, ParameterInput *pin) : pmy_pack(ppack) {
  t_switch_      = pin->GetOrAddReal("cfc", "handoff_time", -1.0);
  cycle_switch_  = pin->GetOrAddInteger("cfc", "handoff_cycle", -1);
  alpha_min_thr_ = pin->GetOrAddReal("cfc", "handoff_alpha_min", -1.0);
  cf_res_thr_    = pin->GetOrAddReal("cfc", "handoff_cf_residual", -1.0);
  cf_floor_      = pin->GetOrAddReal("cfc", "handoff_cf_floor", 1.0e-10);
  check_dcycle_  = pin->GetOrAddInteger("cfc", "handoff_check_dcycle", 1);
  resolve_       = pin->GetOrAddBoolean("cfc", "handoff_resolve", true);
  force_output_  = pin->GetOrAddBoolean("cfc", "handoff_output", false);
  verbose_       = pin->GetOrAddBoolean("cfc", "handoff_verbose", false);

  auto fatal = [](const std::string &msg) {
    std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
              << "<cfc>/handoff: " << msg << std::endl;
    std::exit(EXIT_FAILURE);
  };
  if (t_switch_ < 0.0 && cycle_switch_ < 0 && alpha_min_thr_ <= 0.0 &&
      cf_res_thr_ <= 0.0) {
    fatal("no trigger enabled; set at least one of handoff_time, handoff_cycle, "
          "handoff_alpha_min, handoff_cf_residual");
  }
  if (check_dcycle_ < 1) {
    fatal("handoff_check_dcycle must be >= 1");
  }
  int ng = ppack->pmesh->mb_indcs.ng;
  if (ng < 2 || ng > 4) {
    fatal("Z4c needs <mesh>/nghost = 2, 3 or 4");
  }
  if (ppack->pcfc == nullptr || ppack->padm == nullptr || ppack->pdyngr == nullptr ||
      ppack->pmhd == nullptr || ppack->ptmunu == nullptr) {
    fatal("needs an active CFC + <mhd> (dyn_grmhd) run");
  }
  // These modules add their own tasks to the shared before_/after_stagen task lists,
  // which the switch rebuilds from the NumericalRelativity queues only.
  if (pin->DoesBlockExist("turb_driving") || pin->DoesBlockExist("particles") ||
      pin->DoesBlockExist("radiation")) {
    fatal("not supported together with <turb_driving>, <particles> or <radiation>");
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real Z4cHandoff::GlobalAlphaMin()

Real Z4cHandoff::GlobalAlphaMin() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmy_pack->nmb_thispack;
  auto &adm = pmy_pack->padm->adm;
  Real amin = std::numeric_limits<Real>::max();
  Kokkos::parallel_reduce("cfc_handoff_alpha_min",
  Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>({0, ks, js, is},
                                                      {nmb, ke+1, je+1, ie+1}),
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, Real &lmin) {
    lmin = fmin(lmin, adm.alpha(m, k, j, i));
  }, Kokkos::Min<Real>(amin));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &amin, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
#endif
  return amin;
}

//----------------------------------------------------------------------------------------
//! \fn Real Z4cHandoff::CFResidual()

Real Z4cHandoff::CFResidual() {
  switch (pmy_pack->pmesh->mb_indcs.ng) {
    case 2: return ComputeCFResidual<2>(pmy_pack, cf_floor_);
    case 3: return ComputeCFResidual<3>(pmy_pack, cf_floor_);
    default: return ComputeCFResidual<4>(pmy_pack, cf_floor_);
  }
}

//----------------------------------------------------------------------------------------
//! \fn bool Z4cHandoff::ShouldSwitch()
//! \brief true if any enabled trigger fires. Every decision is made from MPI-reduced
//! (or rank-identical) quantities, so all ranks agree.

bool Z4cHandoff::ShouldSwitch(Driver *pdriver, std::string &reason) {
  Mesh *pm = pmy_pack->pmesh;
  std::ostringstream msg;
  msg << std::setprecision(6);
  if (cycle_switch_ >= 0 && pm->ncycle >= cycle_switch_) {
    msg << "handoff_cycle = " << cycle_switch_ << " reached (ncycle = " << pm->ncycle
        << ")";
    reason = msg.str();
    return true;
  }
  if (t_switch_ >= 0.0 &&
      static_cast<float>(pm->time) >= static_cast<float>(t_switch_)) {
    msg << "handoff_time = " << t_switch_ << " reached (time = " << pm->time << ")";
    reason = msg.str();
    return true;
  }
  bool use_alpha = (alpha_min_thr_ > 0.0), use_cf = (cf_res_thr_ > 0.0);
  if (!(use_alpha || use_cf) || (pm->ncycle % check_dcycle_ != 0)) {
    return false;
  }
  Real amin = use_alpha ? GlobalAlphaMin() : 0.0;
  Real eps = use_cf ? CFResidual() : 0.0;
  if (verbose_ && global_variable::my_rank == 0) {
    std::cout << "CFC->Z4c handoff check: cycle=" << pm->ncycle << " time=" << pm->time;
    if (use_alpha) std::cout << " alpha_min=" << amin;
    if (use_cf) std::cout << " cf_residual=" << eps;
    std::cout << std::endl;
  }
  if (use_alpha && amin < alpha_min_thr_) {
    msg << "alpha_min = " << amin << " < handoff_alpha_min = " << alpha_min_thr_;
    reason = msg.str();
    return true;
  }
  if (use_cf && eps > cf_res_thr_) {
    msg << "cf_residual = " << eps << " > handoff_cf_residual = " << cf_res_thr_;
    reason = msg.str();
    return true;
  }
  return false;
}

//----------------------------------------------------------------------------------------
//! \fn bool Z4cHandoff::CheckAndExecute()

bool Z4cHandoff::CheckAndExecute(Driver *pdriver, ParameterInput *pin, Outputs *pout) {
  std::string reason;
  if (!ShouldSwitch(pdriver, reason)) {
    return false;
  }
  Execute(pdriver, pin, pout, reason);
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn void Z4cHandoff::WriteAllOutputs()
//! \brief writes every output once, outside the normal dt/dcycle schedule, leaving that
//! schedule (last_time) untouched. Blocks that have never written (last_time < 0) keep
//! the time of this write as the start of their schedule.

void Z4cHandoff::WriteAllOutputs(ParameterInput *pin, Outputs *pout) {
  Mesh *pm = pmy_pack->pmesh;
  for (auto &out : pout->pout_list) {
    Real saved = out->out_params.last_time;
    out->LoadOutputData(pm);
    out->WriteOutputFile(pm, pin);
    if (saved >= 0.0) {
      out->out_params.last_time = saved;
      pin->SetReal(out->out_params.block_name, "last_time", saved);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void Z4cHandoff::Execute()
//! \brief converts the current CFC slice to Z4c and swaps the spacetime solver.

void Z4cHandoff::Execute(Driver *pdriver, ParameterInput *pin, Outputs *pout,
                         const std::string &reason) {
  Mesh *pm = pmy_pack->pmesh;
  MeshBlockPack *pmbp = pmy_pack;
  const int ng = pm->mb_indcs.ng;
  const int nmb = pmbp->nmb_thispack;
  const bool root = (global_variable::my_rank == 0);
  if (root) {
    std::cout << std::endl << "=== CFC -> Z4c handoff at cycle " << pm->ncycle
              << ", time " << std::setprecision(10) << pm->time << ": " << reason
              << std::endl;
  }

  // (1) bring the metric in line with the current conserved variables: the last CFC
  // solve used the matter of the last stage's start (or of an older cycle with
  // cfc_solve_interval > 1). Same path as an AMR regrid: ReinitializeMetricForAMR.
  if (resolve_) {
    int saved_nlim = pdriver->nlim;
    pdriver->InitBoundaryValuesAndPrimitives(pm, true);
    if (pdriver->nlim != saved_nlim) {
      if (root) {
        std::cout << "### WARNING: CFC re-solve before the Z4c handoff did not reach "
                  << "the multigrid threshold; continuing with that metric" << std::endl;
      }
      pdriver->nlim = saved_nlim;
    }
  }

  // (2) optional snapshot of the final CFC slice, i.e. the Z4c initial data (the
  // t = 0 one is Driver::Initialize's own). A restart from it re-runs the handoff.
  if (force_output_ && pm->ncycle > 0) {
    WriteAllOutputs(pin, pout);
  }

  // (3) build Z4c (and CCE dumps) exactly as MeshBlockPack::AddPhysics would have
  z4c::Z4c *pz4c = new z4c::Z4c(pmbp, pin);
  pmbp->pz4c = pz4c;
  int ncce = pin->GetOrAddInteger("cce", "num_radii", 0);
  pmbp->pz4c_cce.reserve(ncce);
  for (int n = 0; n < ncce; ++n) {
    pmbp->pz4c_cce.push_back(new z4c::CCE(pm, pin, n));
  }

  // (4) CFC lapse/shift become Z4c's initial gauge (ghosts included: u_adm is fully
  // exchanged at the end of every CFC solve)
  adm::ADM *padm_old = pmbp->padm;
  auto mbr = std::make_pair(0, nmb);
  Kokkos::deep_copy(
    Kokkos::subview(pz4c->u0, mbr, static_cast<int>(z4c::Z4c::I_Z4C_ALPHA),
                    Kokkos::ALL, Kokkos::ALL, Kokkos::ALL),
    Kokkos::subview(padm_old->u_adm, mbr, static_cast<int>(adm::ADM::I_ADM_ALPHA),
                    Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
  Kokkos::deep_copy(
    Kokkos::subview(pz4c->u0, mbr,
                    std::make_pair(static_cast<int>(z4c::Z4c::I_Z4C_BETAX),
                                   static_cast<int>(z4c::Z4c::I_Z4C_BETAZ) + 1),
                    Kokkos::ALL, Kokkos::ALL, Kokkos::ALL),
    Kokkos::subview(padm_old->u_adm, mbr,
                    std::make_pair(static_cast<int>(adm::ADM::I_ADM_BETAX),
                                   static_cast<int>(adm::ADM::I_ADM_BETAZ) + 1),
                    Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));

  // (5) ADM in the Z4c layout (alpha/beta alias z4c->u0); carry g_dd, K_dd, psi4 over
  adm::ADM *padm_new = new adm::ADM(pmbp, pin);
  auto nadm_r = std::make_pair(0, static_cast<int>(adm::ADM::I_ADM_PSI4) + 1);
  Kokkos::deep_copy(
    Kokkos::subview(padm_new->u_adm, mbr, nadm_r, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL),
    Kokkos::subview(padm_old->u_adm, mbr, nadm_r, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
  padm_new->SetADMVariables = padm_old->SetADMVariables;
  padm_new->is_dynamic = padm_old->is_dynamic;
  pmbp->padm = padm_new;

  // (6) ADM -> Z4c: conformal decomposition, Khat = K (0 under maximal slicing),
  // Theta = 0, Gamma^i from the conformal metric (interior; ghosts exchanged in (9))
  switch (ng) {
    case 2: pz4c->ADMToZ4c<2>(pmbp, pin); break;
    case 3: pz4c->ADMToZ4c<3>(pmbp, pin); break;
    case 4: pz4c->ADMToZ4c<4>(pmbp, pin); break;
  }

  // (7) retire CFC
  delete pmbp->pcfc;
  pmbp->pcfc = nullptr;
  pmbp->pdyngr->DisableTauSourceKCFC();

  // (8) native Z4c + dyn_grmhd task graph (adds MHD_SetTmunu)
  pmbp->RebuildNumericalRelativityTasks(pin);

  // (9) Z4c ghost exchange/BCs, MHD ghosts, Z4c -> ADM and con2prim
  pdriver->InitBoundaryValuesAndPrimitives(pm);

  // (10) T_munu for the first Z4c RHS and the constraint diagnostics
  pmbp->pdyngr->SetTmunu(pdriver, 0);
  switch (ng) {
    case 2: pz4c->ADMConstraints<2>(pmbp); break;
    case 3: pz4c->ADMConstraints<3>(pmbp); break;
    case 4: pz4c->ADMConstraints<4>(pmbp); break;
  }
  {
    Real dg = MaxAbsDiff(pmbp, padm_old->u_adm, padm_new->u_adm,
                         adm::ADM::I_ADM_GXX, adm::ADM::I_ADM_GZZ + 1);
    Real dk = MaxAbsDiff(pmbp, padm_old->u_adm, padm_new->u_adm,
                         adm::ADM::I_ADM_KXX, adm::ADM::I_ADM_KZZ + 1);
    Real dp = MaxAbsDiff(pmbp, padm_old->u_adm, padm_new->u_adm,
                         adm::ADM::I_ADM_PSI4, adm::ADM::I_ADM_PSI4 + 1);
    Real con[3];
    ConstraintNorms(pmbp, con);
    if (root) {
      std::cout << std::scientific << std::setprecision(6)
                << "  CFC->Z4c->ADM round trip (interior max abs diff): g_dd " << dg
                << ", K_dd " << dk << ", psi4 " << dp << std::endl
                << "  constraints after handoff: ||H||_2 " << con[0] << ", ||M||_2 "
                << con[1] << ", max|H| " << con[2] << std::endl;
      std::cout.unsetf(std::ios_base::floatfield);
    }
  }

  // (11) outputs: z4c/con/weyl blocks skipped so far, adm outputs re-pointed
  pout->Rebuild(pin, pm);

  // (12) timestep inputs from the new physics (Mesh::NewTimeStep follows in the Driver)
  (void) pz4c->NewTimeStep(pdriver, pdriver->nexp_stages);
  (void) pmbp->pdyngr->NewTimeStep(pdriver, pdriver->nexp_stages);

  // (13) restarts written from now on rebuild a native Z4c run
  pin->SetBoolean("cfc", "handoff_done", true);
  pin->SetReal("cfc", "handoff_done_time", pm->time);
  pin->SetInteger("cfc", "handoff_done_cycle", pm->ncycle);
  delete padm_old;

  if (force_output_) {
    WriteAllOutputs(pin, pout);
  }
  if (root) {
    std::cout << "=== CFC -> Z4c handoff complete; continuing as a Z4c run" << std::endl
              << std::endl;
  }
}

} // namespace cfc
