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
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "coordinates/coordinates.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/mass_closure.hpp"
#include "cfc/cfc.hpp"
#include "mhd/mhd.hpp"
#include "outputs/restart_state.hpp"


namespace restart_state {
namespace {
DvceArray5D<Real> staged;
int staged_nvar = 0, staged_version = 0;
std::string staged_layout;

// the groups present in format version 1: without them the state cannot be continued
const char *const kCore[] = {"dyngr.w0", "dyngr.bcc0", "dyngr.temp", "cfc.dpsi", "cfc.dap",
                             "cfc.mg_psi", "cfc.mg_alpha", "cfc.mg_pietax",
                             "cfc.mg_pietabeta"};

bool IsCore(const std::string &g) {
  for (const char *c : kCore) if (g == c) return true;
  return false;
}

// "name:n,name:n" -> {(name, n)}
std::vector<std::pair<std::string, int>> Parse(const std::string &s) {
  std::vector<std::pair<std::string, int>> v;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    const auto c = tok.rfind(':');
    if (tok.empty() || c == std::string::npos) continue;
    v.emplace_back(tok.substr(0, c), std::stoi(tok.substr(c + 1)));
  }
  return v;
}

// the modules' loads from a block in this configuration's layout
void Load(MeshBlockPack *pmbp, const DvceArray5D<Real> &src) {
  int n = 0;
  if (pmbp->pdyngr != nullptr) {
    pmbp->pdyngr->LoadRestartState(src, n);
    n += pmbp->pdyngr->RestartStateVars();
  }
  if (pmbp->pcfc != nullptr) pmbp->pcfc->LoadRestartState(src, n);
  Kokkos::fence();
}
}  // namespace

std::string Layout(MeshBlockPack *pmbp) {
  std::string s;
  if (pmbp->pdyngr != nullptr) s += pmbp->pdyngr->RestartStateLayout();
  if (pmbp->pcfc != nullptr) {
    const std::string c = pmbp->pcfc->RestartStateLayout();
    s += (s.empty() || c.empty()) ? c : "," + c;
  }
  return s;
}

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

void Stage(const DvceArray5D<Real> &src, int nvar, int version, const std::string &layout) {
  staged = src;
  staged_nvar = nvar;
  staged_version = version;
  staged_layout = layout;
}

bool ApplyStaged(MeshBlockPack *pmbp) {
  if (staged_nvar == 0) return false;
  const int nvar = NumVars(pmbp);
  const std::string cur = Layout(pmbp);
  bool applied = false;
  std::string why, zeroed, ignored;
  if (staged_layout.empty() || staged_layout == cur) {
    // version 0 (no layout: the same groups in the same order iff the count matches), or
    // the same layout: the block loads as it is
    applied = (staged_nvar == nvar);
    if (applied) {
      Load(pmbp, staged);
    } else {
      why = "the restart carries " + std::to_string(staged_nvar) + " state variables, this"
            " configuration needs " + std::to_string(nvar);
    }
  } else {
    // map the groups by name into a block in this configuration's layout
    const auto fv = Parse(staged_layout), cv = Parse(cur);
    std::vector<int> foff(fv.size(), 0);
    int nf = 0;
    for (size_t q = 0; q < fv.size(); ++q) { foff[q] = nf; nf += fv[q].second; }
    applied = (nf == staged_nvar);
    if (!applied) why = "the layout's count (" + std::to_string(nf) + ") is not nvar";
    DvceArray5D<Real> map("rst-state-map", staged.extent_int(0), nvar, staged.extent_int(2),
                          staged.extent_int(3), staged.extent_int(4));   // zero
    auto all = Kokkos::ALL;
    int nc = 0;
    for (size_t g = 0; g < cv.size() && applied; ++g) {
      size_t q = 0;
      while (q < fv.size() && fv[q].first != cv[g].first) ++q;
      if (q == fv.size()) {
        if (IsCore(cv[g].first)) {
          applied = false;
          why = "the core group " + cv[g].first + " is missing";
        } else {
          zeroed += (zeroed.empty() ? "" : ",") + cv[g].first;
        }
      } else if (fv[q].second != cv[g].second) {
        applied = false;
        why = "the group " + cv[g].first + " has " + std::to_string(fv[q].second) +
              " variables, this configuration needs " + std::to_string(cv[g].second);
      } else {
        Kokkos::deep_copy(
            Kokkos::subview(map, all, std::make_pair(nc, nc + cv[g].second), all, all, all),
            Kokkos::subview(staged, all, std::make_pair(foff[q], foff[q] + fv[q].second),
                            all, all, all));
      }
      nc += cv[g].second;
    }
    for (const auto &f : fv) {
      bool used = false;
      for (const auto &c : cv) used = used || (c.first == f.first);
      if (!used) ignored += (ignored.empty() ? "" : ",") + f.first;
    }
    if (applied) Load(pmbp, map);
  }
  if (global_variable::my_rank == 0) {
    if (!applied) {
      // a different physics configuration than the one that wrote the file: the state
      // cannot be mapped, so fall back to the init C2P (not bitwise continuous)
      std::cout << "### WARNING restart_state: " << why << "; ignored (init C2P instead, "
                << "not bitwise continuous)" << std::endl;
    } else if (!zeroed.empty() || !ignored.empty() || staged_version != kFormatVersion) {
      std::cout << "restart_state: format version " << staged_version << " read by "
                << kFormatVersion << "; zero-filled (feature off): "
                << (zeroed.empty() ? "none" : zeroed) << "; ignored: "
                << (ignored.empty() ? "none" : ignored) << std::endl;
    }
  }
  staged = DvceArray5D<Real>();
  staged_nvar = 0;
  staged_version = 0;
  staged_layout.clear();
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
  const int nc1 = indcs.nx1 + 2*indcs.ng;
  const int nc2 = (nx2 > 1) ? indcs.nx2 + 2*indcs.ng : 1;
  const int nc3 = (nx3 > 1) ? indcs.nx3 + 2*indcs.ng : 1;
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
  // every cell, ghosts included: a same-level ghost has its owner's global index and
  // value, so it gets the owner's factor bitwise (a ghost exchange here instead would
  // re-derive the restored ghosts, which c102_b5 showed is not bitwise: 1e-10 with
  // eps = 1e-300)
  par_for("restart_seed", DevExeSpace(), 0, nmb-1, 0, nc3-1, 0, nc2-1, 0, nc1-1,
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

bool EmbedFill(Mesh *pm, ParameterInput *pin) {
  if (!pin->GetOrAddBoolean("job", "restart_embed_fill", false) ||
      pin->DoesParameterExist("job", "restart_embed_done")) {
    return false;
  }
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->pcfc == nullptr || pmbp->pdyngr == nullptr || pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<job> restart_embed_fill needs dyn_grmhd with CFC" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::vector<Real> box;
  {
    std::stringstream ss(pin->GetString("job", "restart_embed_box"));
    std::string tok;
    while (std::getline(ss, tok, ',')) box.push_back(std::stod(tok));
  }
  if (box.size() != 6) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<job> restart_embed_box needs x1min,x1max,x2min,x2max,x3min,x3max"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // the new blocks: outside the old box (r1_embed.py never makes a block straddle it)
  const int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;
  DualArray1D<int> nm("embed_new", nmb);
  int nnew = 0;
  for (int m = 0; m < nmb; ++m) {
    const Real cx = 0.5*(size.h_view(m).x1min + size.h_view(m).x1max);
    const Real cy = 0.5*(size.h_view(m).x2min + size.h_view(m).x2max);
    const Real cz = 0.5*(size.h_view(m).x3min + size.h_view(m).x3max);
    nm.h_view(m) = (cx < box[0] || cx > box[1] || cy < box[2] || cy > box[3] ||
                    cz < box[4] || cz > box[5]) ? 1 : 0;
    nnew += nm.h_view(m);
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &nnew, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
  // an embedded restart has new blocks outside the old box; none means the deck asks for
  // an embedding the restart does not have (C-125: the un-embedded restart was loaded)
  if (nnew == 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<job> restart_embed_fill is set but the restart has no MeshBlock outside "
              << "restart_embed_box (not the r1_embed.py output?)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  nm.template modify<HostMemSpace>();
  nm.template sync<DevExeSpace>();
  auto newmb = nm.d_view;
  mclose::MassClosure *mc = pmbp->pdyngr->pmclose;
  if (mc != nullptr) mc->EmbedMark(0);

  // the metric there: the zero-residual CFC metric (re-solved by the caller)
  pmbp->pcfc->FillEmbedADM(newmb, pm->time);

  // the floor atmosphere at rest, every cell of the new blocks (ghosts included)
  auto &indcs = pm->mb_indcs;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*indcs.ng) : 1;
  const int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*indcs.ng) : 1;
  const Real rho_v = pin->GetReal("mhd", "dfloor");
  const Real p_v = rho_v*pin->GetReal("mhd", "tfloor");
  auto &w0 = pmbp->pmhd->w0;
  auto &u0 = pmbp->pmhd->u0;
  const int nw = w0.extent_int(1), nu = u0.extent_int(1);
  par_for("restart_embed_w0", DevExeSpace(), 0, nmb-1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    if (newmb(m) == 0) return;
    w0(m, IDN, k, j, i) = rho_v;
    w0(m, IPR, k, j, i) = p_v;
    w0(m, IVX, k, j, i) = 0.0; w0(m, IVY, k, j, i) = 0.0; w0(m, IVZ, k, j, i) = 0.0;
    for (int n = IPR + 1; n < nw; ++n) w0(m, n, k, j, i) = 0.0;   // scalars (DE: from P)
  });
  // conserved: PrimToConInit converts every block (and writes the dual-energy tracer
  // into w0), so the old blocks' u0 and w0 are saved and restored bitwise
  DvceArray5D<Real> usave("embed_u0", u0.extent(0), u0.extent(1), u0.extent(2),
                          u0.extent(3), u0.extent(4));
  DvceArray5D<Real> wsave("embed_w0", w0.extent(0), w0.extent(1), w0.extent(2),
                          w0.extent(3), w0.extent(4));
  Kokkos::deep_copy(usave, u0);
  Kokkos::deep_copy(wsave, w0);
  pmbp->pdyngr->PrimToConInit(0, n1-1, 0, n2-1, 0, n3-1);
  par_for("restart_embed_keep", DevExeSpace(), 0, nmb-1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    if (newmb(m) != 0) return;
    for (int n = 0; n < nu; ++n) u0(m, n, k, j, i) = usave(m, n, k, j, i);
    for (int n = 0; n < nw; ++n) w0(m, n, k, j, i) = wsave(m, n, k, j, i);
  });
  Kokkos::fence();
  if (mc != nullptr) mc->EmbedMark(1);

  // the excision masks on the new mesh (the regrid path the caller runs does not rebuild
  // them; the restored path does it in InitBoundaryValuesAndPrimitives)
  if (pmbp->pcoord->coord_data.bh_excise) {
    pmbp->pdyngr->SetExcisionGeometry();
    pmbp->pcoord->UpdateExcisionMasks();
  }
  pin->SetInteger("job", "restart_embed_done", pm->ncycle);
  if (global_variable::my_rank == 0) {
    std::cout << "restart_state: R1 embed: " << nnew << " new MeshBlocks outside the old box "
              << box[0] << "," << box[1] << "," << box[2] << "," << box[3] << "," << box[4]
              << "," << box[5] << " filled (zero-residual metric, floor atmosphere at rest);"
              << " the metric is re-solved at load (regrid path)" << std::endl;
  }
  return true;
}
}  // namespace restart_state
