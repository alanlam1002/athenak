//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file restart_state.cpp
//! \brief see restart_state.hpp. Layout of the block: dyn_grmhd's variables first, then
//! cfc's.

#include <cstdint>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "cfc/cfc.hpp"
#include "mhd/mhd.hpp"
#include "outputs/restart_state.hpp"


namespace restart_state {
namespace {
DvceArray5D<Real> staged;
int staged_nvar = 0;
}  // namespace

int NumVars(MeshBlockPack *pmbp) {
  int n = 0;
  if (pmbp->pdyngr != nullptr) n += pmbp->pdyngr->RestartStateVars();
  if (pmbp->pcfc != nullptr) n += pmbp->pcfc->RestartStateVars();
  return n;
}

void Pack(MeshBlockPack *pmbp, DvceArray5D<Real> &dst) {
  int n = 0;
  if (pmbp->pdyngr != nullptr) {
    pmbp->pdyngr->SaveRestartState(dst, n);
    n += pmbp->pdyngr->RestartStateVars();
  }
  if (pmbp->pcfc != nullptr) pmbp->pcfc->SaveRestartState(dst, n);
  Kokkos::fence();
}

void Stage(const DvceArray5D<Real> &src, int nvar) {
  staged = src;
  staged_nvar = nvar;
}

bool ApplyStaged(MeshBlockPack *pmbp) {
  if (staged_nvar == 0) return false;
  if (staged_nvar != NumVars(pmbp)) {
    // a different physics configuration than the one that wrote the file: the state
    // cannot be mapped, so fall back to the init C2P (not bitwise continuous)
    if (global_variable::my_rank == 0) {
      std::cout << "### WARNING restart_state: the restart carries " << staged_nvar
                << " state variables, this configuration needs " << NumVars(pmbp)
                << "; ignored (init C2P instead, not bitwise continuous)" << std::endl;
    }
  } else {
    int n = 0;
    if (pmbp->pdyngr != nullptr) {
      pmbp->pdyngr->LoadRestartState(staged, n);
      n += pmbp->pdyngr->RestartStateVars();
    }
    if (pmbp->pcfc != nullptr) pmbp->pcfc->LoadRestartState(staged, n);
    Kokkos::fence();
  }
  const bool applied = (staged_nvar == NumVars(pmbp));
  staged = DvceArray5D<Real>();
  staged_nvar = 0;
  return applied;
}

namespace {
KOKKOS_INLINE_FUNCTION uint64_t Mix(uint64_t z) {   // splitmix64 finalizer
  z += 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30))*0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27))*0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
}  // namespace

bool Seed(Mesh *pm, ParameterInput *pin, bool res_flag) {
  const Real eps = pin->GetOrAddReal("job", "seed_tau_eps", 0.0);
  const int sid = pin->GetOrAddInteger("job", "seed_id", 0);
  const bool done = pin->GetOrAddBoolean("job", "seed_applied", false);
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (!res_flag || eps == 0.0 || done || pmbp->pmhd == nullptr) return false;
  const bool de_on = pin->GetOrAddBoolean("mhd", "dual_energy", false);
  const Real rsw = pin->DoesParameterExist("mhd", "dual_energy_rho_switch") ?
                   pin->GetReal("mhd", "dual_energy_rho_switch") : 0.0;
  const int nmhd = pmbp->pmhd->nmhd;
  const int idy = de_on ? nmhd + pmbp->pmhd->nscalars - 1 : -1;   // the K tracer's slot

  // global cell offsets of each MeshBlock at its own level (host, then device)
  const int nmb = pmbp->nmb_thispack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  DualArray2D<int64_t> off("seed_off", nmb, 4);
  for (int m = 0; m < nmb; ++m) {
    const LogicalLocation &ll = pm->lloc_eachmb[pmbp->gids + m];
    off.h_view(m, 0) = static_cast<int64_t>(ll.lx1)*nx1;
    off.h_view(m, 1) = static_cast<int64_t>(ll.lx2)*nx2;
    off.h_view(m, 2) = static_cast<int64_t>(ll.lx3)*nx3;
    off.h_view(m, 3) = ll.level;
  }
  off.template modify<HostMemSpace>();
  off.template sync<DevExeSpace>();
  auto doff = off.d_view;
  auto u0 = pmbp->pmhd->u0;
  const uint64_t s0 = Mix(static_cast<uint64_t>(sid));
  par_for("restart_seed", DevExeSpace(), 0, nmb-1, ks, ks+nx3-1, js, js+nx2-1,
          is, is+nx1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    if (!(u0(m, IDN, k, j, i) > rsw)) return;
    uint64_t h = Mix(s0 ^ static_cast<uint64_t>(doff(m, 3)));
    h = Mix(h ^ static_cast<uint64_t>(doff(m, 0) + (i - is)));
    h = Mix(h ^ static_cast<uint64_t>(doff(m, 1) + (j - js)));
    h = Mix(h ^ static_cast<uint64_t>(doff(m, 2) + (k - ks)));
    const Real f = 1.0 + ((h >> 63) ? eps : -eps);
    u0(m, IEN, k, j, i) *= f;
    if (idy >= 0) u0(m, idy, k, j, i) *= f;
  });
  Kokkos::fence();
  pin->SetBoolean("job", "seed_applied", true);
  if (global_variable::my_rank == 0) {
    std::cout << "restart_state: seeded tau" << (de_on ? " and D Y" : "")
              << " (eps " << eps << ", seed_id " << sid << ") in cells with D > " << rsw
              << std::endl;
  }
  return true;
}
}  // namespace restart_state
