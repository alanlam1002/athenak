//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file restart_state.cpp
//! \brief see restart_state.hpp. Layout of the block: dyn_grmhd's variables first, then
//! cfc's.

#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "cfc/cfc.hpp"
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
}  // namespace restart_state
