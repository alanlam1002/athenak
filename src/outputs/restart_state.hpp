#ifndef OUTPUTS_RESTART_STATE_HPP_
#define OUTPUTS_RESTART_STATE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file restart_state.hpp
//! \brief <job> restart_state (default true; C-100, user-approved): the evolution state a restart needs to
//! continue bitwise but that u0/b0/u_adm do not carry, written as one more per-MeshBlock
//! cell-centred block after the ADM data (ghosts included, the mesh's ng):
//!   dyn_grmhd: w0 (the last C2P's primitives: the next stage's reconstruction and the
//!     dual-energy shock sensor read them), u0's non-evolved flag slot(s)
//!     (dual_energy_flag_exchange), bcc0, temperature, de_streak (dual_energy_shock_persist
//!     > 1 only);
//!   cfc: delta_psi, delta_alpha_psi, and each multigrid solver's finest-level solution
//!     (the next solve's warm start; its initial defect reads the MG ghosts too).
//! A restart that carries the block skips the init C2P (driver.cpp), which otherwise ran
//! once more on the already-inverted u0 from w0 = 0 (C-100: first dt 3e-4 off). The count
//! is written into the restart's parameter copy as <job> restart_state_nvar, so restarts
//! written before this (nvar absent = 0) load as before. Also needed for a bitwise restart
//! (C-100): the tier-3 state and the CFC seed flags (parameter copy), and defined MG
//! coarse-boundary buffers (MultigridBoundaryValues::FillCoarseMG zeroes coarse_buf_). The
//! MG levels below the finest need not be saved (c100_rst13b: bitwise without them).

#include "athena.hpp"

class MeshBlockPack;

namespace restart_state {
// number of variables this pack's state needs
int NumVars(MeshBlockPack *pmbp);
// pack into dst(m, n, k, j, i), n < NumVars (dst sized by the caller)
void Pack(MeshBlockPack *pmbp, DvceArray5D<Real> &dst);
// the reader (pgen.cpp) stages the block here; Driver::Initialize applies it
void Stage(const DvceArray5D<Real> &src, int nvar);
// true if a staged block was applied (the caller then skips the init C2P)
bool ApplyStaged(MeshBlockPack *pmbp);
}  // namespace restart_state

#endif  // OUTPUTS_RESTART_STATE_HPP_
