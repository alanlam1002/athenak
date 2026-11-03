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
//!
//! Forward compatibility (R-140 item 4, D6): every restart written by this binary carries
//! <job> restart_format_version (kFormatVersion; absent = 0, i.e. base8 and earlier) and
//! <job> restart_state_layout, the block's named groups in file order ("dyngr.w0:7,
//! dyngr.bcc0:3,...,cfc.mg_psi:1,..."). A reader maps the groups it needs by name:
//!   - a version-0 file (no layout) loads as before: the block must match this
//!     configuration's count exactly (it then has the same groups in the same order);
//!   - a group present in the file with the needed count is copied;
//!   - a group absent from the file is zero-filled if it is not a core group (kCore below):
//!     **the rule for new state** is that zero (or, for parameter-copy scalars, the
//!     GetOrAdd default) is the feature-off state, so a new binary continues an old
//!     restart bitwise with the new features off. A missing core group, or a count
//!     mismatch, falls back to the init C2P (not bitwise; a warning says so);
//!   - groups in the file that this configuration does not use are ignored (reported).
//! Bump kFormatVersion when a change cannot be expressed as a new group or a new defaulted
//! scalar (e.g. the fixed header's binary layout); the standing test is
//! scripts/rst_compat_test.sh (a base8 checkpoint continued by the base8 and the new
//! binary, features off: hst and dumps must match bitwise).

#include "athena.hpp"

class Mesh;
class MeshBlockPack;
class ParameterInput;

namespace restart_state {
constexpr int kFormatVersion = 1;
// the group layout this pack's state needs ("name:n,...", Pack's order)
std::string Layout(MeshBlockPack *pmbp);
// number of variables this pack's state needs
int NumVars(MeshBlockPack *pmbp);
// pack into dst(m, n, k, j, i), n < NumVars (dst sized by the caller)
void Pack(MeshBlockPack *pmbp, DvceArray5D<Real> &dst);
// the reader (pgen.cpp) stages the block here, with the file's format version and layout
// ("" for version 0); Driver::Initialize applies it
void Stage(const DvceArray5D<Real> &src, int nvar, int version, const std::string &layout);
// true if a staged block was applied (the caller then skips the init C2P)
bool ApplyStaged(MeshBlockPack *pmbp);
// <job> seed_tau_eps (default 0 = off), seed_id (R-115 Q3, user-approved): once, at a
// restart, perturb the dense cells (u0's densitized D > <mhd> dual_energy_rho_switch;
// the selection only needs to be decomposition-independent) by the factor (1 + eps xi)
// in tau and, with <mhd> dual_energy, in the entropy tracer D Y (dense unshocked gas
// takes P from K and resyncs tau, primitive_solver_hyd.hpp:146-147, so a tau-only seed
// would be erased there). xi = +-1 from a hash of the seed id and the cell's global
// (level, i, j, k) index at its block's level: the same perturbation on any
// decomposition. D and S_i are untouched (mass and momentum exact). All cells, ghosts
// included: a same-level ghost gets its owner's factor bitwise; a coarse/fine ghost
// (prolonged or restricted, no owner cell) gets its own O(eps) factor. No ghost exchange
// (it would re-derive the restored ghosts, which is not bitwise). Sets <job>
// seed_applied = true (carried by later restarts, so a chain seeds once). Returns true
// if applied.
bool Seed(Mesh *pm, ParameterInput *pin, bool res_flag);
// <job> restart_embed_fill (R1, R-150 item 6; research REGRID_SPEC §2): a restart written by
// scripts/r1_embed.py, the old box (<job> restart_embed_box = x1min,x1max,...,x3max)
// embedded bitwise into a box twice as wide, its new outer blocks zero. Once (sets <job>
// restart_embed_done = ncycle, kept by later restarts; a deck cannot unset it): the
// blocks outside the old box get the zero-residual CFC metric (CFC::FillEmbedADM) and the
// floor atmosphere at rest (rho = dfloor, P = rho tfloor, u^i = 0, as the pgen's vacuum
// branch) converted with that metric, every other block's u0 and w0 unchanged bitwise;
// the excision masks are rebuilt. MCLOSE marks the mass before (the old domain) and after
// (EmbedMark 0, 1). Must follow ApplyStaged; the caller then runs the regrid path's
// re-solve and C2P (Driver::Initialize). Returns true if applied.
bool EmbedFill(Mesh *pm, ParameterInput *pin);
}  // namespace restart_state

#endif  // OUTPUTS_RESTART_STATE_HPP_
