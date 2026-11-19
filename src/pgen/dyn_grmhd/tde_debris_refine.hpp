#ifndef PGEN_DYN_GRMHD_TDE_DEBRIS_REFINE_HPP_
#define PGEN_DYN_GRMHD_TDE_DEBRIS_REFINE_HPP_
//========================================================================================
// TDE debris refinement criterion (<problem> amr_condition = tde_debris), research R-050
// (athenak_tde_project research/notes/NEXT_TASKS_R050.md B1, DEBRIS_REFINEMENT_DESIGN.md
// §3). Included only by dyngr_tov.cpp. TDERefineTracker (tde_track) is untouched.
//
// Every density threshold is ABSOLUTE, a fraction of rho_ref (<problem> amr_rho_ref,
// default the TOV central density <problem> rhoc) -- no global rho_max is used, so a
// surviving remnant no longer pins the cut. Per MeshBlock the target level is the max of
//   1. core:   block max rho > rho_A = amr_core_frac*rho_ref        -> amr_level_core
//   2. ladder: per cell rho_F < rho <= rho_A (rho_F = amr_rho_floor_frac*rho_ref),
//              L = L_core - ceil(ln(rho_A/rho)/ln f_L), f_L = amr_ladder_factor,
//              floored at amr_level_floor (one level per factor f_L in density)
//   3. width:  cells with rho > rho_F whose 6 face neighbours are all > rho_F (the gate:
//              l -> 0 at a free surface), l = rho/|grad rho| (centred differences),
//              smallest L with dx_L <= l/N_H, N_H = amr_width_cells, capped at
//              amr_level_stream (N_H = 0 disables)
//   4. Jeans:  rho > rho_F, lambda_J = c_s sqrt(pi/rho), c_s^2 = Gamma P/(rho h); smallest
//              L with dx_L <= lambda_J/N_J, N_J = amr_jeans_cells, capped at L_core
//              (N_J = 0 disables)
//   5. BH box: exactly tde_track's (amr_radius_bh/amr_level_bh, optional fine box)
// Hysteresis: want_ref uses the thresholds above, want_deref relaxed ones (densities times
// amr_deref_factor < 1, N_H and N_J divided by it, so want_deref >= want_ref). Flag +1 if level <
// want_ref; -1 if level > want_deref AND the block has sat at its level for >=
// amr_min_residence cycles; else 0. Residence (C-233): the first check cycle at which each
// (lx1, lx2, lx3, level) existed, kept for EVERY block of the mesh on every rank (each rank
// replays the same global block list, so the map is identical on all ranks and a block that
// migrates ranks keeps its count; before C-233 it was per rank and migration restarted it),
// and saved in the restart (<problem> amr_residence_state, TDEDebrisSaveResidence): a
// restart continues the counts, so its flags are the continuous run's. A restart without the
// key (older files), or whose block list differs from the one saved (an R1 embed), starts
// every count at its first check, as before.
// Cells at/below rho_F never request refinement (this replaces tde_track's rho_max
// corruption guard); non-finite cells are skipped, with a one-time warning.
// amr_dryrun = true: the mesh follows tde_track, and tde_debris only computes and logs
// what it would request (sizing on real runs). Diagnostics every amr_diag_ncycle cycles.
//========================================================================================

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct TDEDebrisParams {
  Real rho_ref;               // amr_rho_ref (default <problem> rhoc)
  Real core_frac, floor_frac; // rho_A/rho_ref, rho_F/rho_ref
  Real ladder_factor;         // f_L
  bool ladder;
  Real width_cells, jeans_cells;  // N_H, N_J (0 disables)
  int lev_core, lev_floor, lev_stream;
  Real deref_factor;
  int min_residence;
  bool dryrun;
  int diag_ncycle;
  Real gamma;                 // ideal-gas Gamma for the Jeans length
};
TDEDebrisParams tde_deb;

// residence (C-233): key (lx1, lx2, lx3, level) -> the first check cycle it existed
std::map<std::array<int, 4>, int> tde_deb_first_seen;
// a restart's saved residence, applied at the first check (the mesh is complete then)
std::string tde_deb_rst_state;

// order-dependent hash of the mesh's block list (gid order), to match a saved state to its mesh
uint64_t TDEDebrisTreeHash(const Mesh *pm) {
  uint64_t h = 1469598103934665603ULL;   // FNV-1a over the 4 ints of each block
  for (int g = 0; g < pm->nmb_total; ++g) {
    const LogicalLocation &ll = pm->lloc_eachmb[g];
    const int v[4] = {ll.lx1, ll.lx2, ll.lx3, ll.level};
    for (int q = 0; q < 4; ++q) {
      uint64_t x = static_cast<uint32_t>(v[q]);
      for (int b = 0; b < 4; ++b) { h ^= (x >> (8*b)) & 0xffULL; h *= 1099511628211ULL; }
    }
  }
  return h;
}

// apply a staged restart state to tde_deb_first_seen (first check after the restart)
void TDEDebrisApplyResidence(const Mesh *pm) {
  std::stringstream ss(tde_deb_rst_state);
  tde_deb_rst_state.clear();
  std::string tok;
  long ncyc = -1, nmb = -1, minres = -1;
  unsigned long long hash = 0;
  std::getline(ss, tok, ';');
  if (std::sscanf(tok.c_str(), "v1 %ld %ld %ld %llx", &ncyc, &nmb, &minres, &hash) != 4) {
    if (global_variable::my_rank == 0)
      std::cout << "### WARNING tde_debris: amr_residence_state unreadable; every residence "
                << "starts now (not bitwise continuous)" << std::endl;
    return;
  }
  if (nmb != pm->nmb_total || hash != TDEDebrisTreeHash(pm)) {
    if (global_variable::my_rank == 0)
      std::cout << "tde_debris: amr_residence_state was saved for another block list (" << nmb
                << " blocks vs " << pm->nmb_total << "; an R1 embed?): every residence "
                << "starts now" << std::endl;
    return;
  }
  // listed: young blocks (c0) and blocks created by the last regrid (c0 = -1: not yet seen);
  // unlisted: resident >= minres cycles at the save
  std::map<std::array<int, 4>, int> listed;
  while (std::getline(ss, tok, ';')) {
    int k[5];
    if (std::sscanf(tok.c_str(), "%d,%d,%d,%d,%d", &k[0], &k[1], &k[2], &k[3], &k[4]) == 5)
      listed[{k[0], k[1], k[2], k[3]}] = k[4];
  }
  tde_deb_first_seen.clear();
  for (int g = 0; g < pm->nmb_total; ++g) {
    const LogicalLocation &ll = pm->lloc_eachmb[g];
    const std::array<int, 4> key = {ll.lx1, ll.lx2, ll.lx3, ll.level};
    auto it = listed.find(key);
    if (it == listed.end()) {
      tde_deb_first_seen[key] = static_cast<int>(ncyc - minres);
    } else if (it->second >= 0) {
      tde_deb_first_seen[key] = it->second;
    }
  }
  if (global_variable::my_rank == 0)
    std::cout << "tde_debris: residence restored for " << pm->nmb_total << " blocks ("
              << listed.size() << " listed) from cycle " << ncyc << std::endl;
}

// criterion ids for the diagnostics (want_ref attribution, ties in this order)
enum TDEDebCrit {kDebCore = 0, kDebLadder = 1, kDebWidth = 2, kDebJeans = 3, kDebBH = 4,
                 kDebNone = 5, kDebNCrit = 6};

// per-cell target levels (-1: none) for one set of thresholds (scale s = 1 refine,
// s = amr_deref_factor de-refine). gate = all 6 face neighbours above rho_F.
KOKKOS_INLINE_FUNCTION
void TDEDebrisCellWant(const Real rho, const Real prs, const Real grad, const bool gate,
                       const Real rho_a, const Real rho_f, const Real lnf, const bool ladder,
                       const int lcore, const int lfloor, const int lstream,
                       const Real nh, const Real nj, const Real dx_root, const Real gam,
                       int &l_lad, int &l_wid, int &l_jea) {
  l_lad = -1; l_wid = -1; l_jea = -1;
  if (!(rho > rho_f)) return;
  if (ladder && rho <= rho_a) {
    int l = lcore - static_cast<int>(ceil(log(rho_a/rho)/lnf));
    l_lad = (l > lfloor) ? l : lfloor;
  }
  if (nh > 0.0 && gate && grad > 0.0) {
    const Real ell = rho/grad;
    int l = static_cast<int>(ceil(log2(nh*dx_root/ell)));
    l = (l < 0) ? 0 : l;
    l_wid = (l < lstream) ? l : lstream;
  }
  if (nj > 0.0 && prs > 0.0) {
    const Real h = 1.0 + gam/(gam - 1.0)*prs/rho;
    const Real lam = sqrt(gam*prs/(rho*h))*sqrt(M_PI/rho);
    int l = static_cast<int>(ceil(log2(nj*dx_root/lam)));
    l = (l < 0) ? 0 : l;
    l_jea = (l < lcore) ? l : lcore;
  }
}

// per-block reductions: [0] max rho, then {ladder, width, jeans} want for refine (1..3)
// and de-refine (4..6), [7] number of non-finite cells
constexpr int kDebNRed = 8;

void TDEDebrisBlockWants(MeshBlockPack *pmbp, const TDEDebrisParams &p,
                         HostArray2D<Real> &out) {
  Mesh *pmesh = pmbp->pmesh;
  auto &w0_ = pmbp->pmhd->w0;
  auto &indcs = pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1, nji = nx2*nx1;
  const int nmb = pmbp->nmb_thispack;
  const Real dx_root = pmesh->mesh_size.dx1;
  const Real rho_a = p.core_frac*p.rho_ref, rho_f = p.floor_frac*p.rho_ref;
  const Real s = p.deref_factor;
  const Real lnf = log(p.ladder_factor);
  const bool ladder = p.ladder;
  const int lcore = p.lev_core, lfloor = p.lev_floor, lstream = p.lev_stream;
  const Real nh = p.width_cells, nj = p.jeans_cells, gam = p.gamma;

  DvceArray2D<Real> red("tde_debris_red", nmb, kDebNRed);
  par_for_outer("TDEDebrisBlockWants", DevExeSpace(), 0, 0, 0, nmb-1,
  KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
    const Real dx = size.d_view(m).dx1, dy = size.d_view(m).dx2, dz = size.d_view(m).dx3;
    for (int q = 0; q < kDebNRed; ++q) {
      Real team_max = (q == 0) ? std::numeric_limits<Real>::lowest() : -1.0;
      Real team_sum = 0.0;
      if (q == kDebNRed - 1) {
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
        [=](const int idx, Real &lsum) {
          const int k = idx/nji + ks, j = (idx%nji)/nx1 + js, i = idx%nx1 + is;
          if (!Kokkos::isfinite(w0_(m,IDN,k,j,i))) lsum += 1.0;
        }, Kokkos::Sum<Real>(team_sum));
        Kokkos::single(Kokkos::PerTeam(tmember), [&]() { red(m, q) = team_sum; });
        continue;
      }
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
      [=](const int idx, Real &lmax) {
        const int k = idx/nji + ks, j = (idx%nji)/nx1 + js, i = idx%nx1 + is;
        const Real rho = w0_(m,IDN,k,j,i);
        if (!Kokkos::isfinite(rho)) return;
        if (q == 0) { lmax = fmax(lmax, rho); return; }
        const Real sc = (q <= 3) ? 1.0 : s;
        const Real rf = sc*rho_f;
        if (!(rho > rf)) return;
        const Real prs = w0_(m,IPR,k,j,i);
        const Real rxp = w0_(m,IDN,k,j,i+1), rxm = w0_(m,IDN,k,j,i-1);
        const Real ryp = w0_(m,IDN,k,j+1,i), rym = w0_(m,IDN,k,j-1,i);
        const Real rzp = w0_(m,IDN,k+1,j,i), rzm = w0_(m,IDN,k-1,j,i);
        const bool gate = (rxp > rf) && (rxm > rf) && (ryp > rf) && (rym > rf) &&
                          (rzp > rf) && (rzm > rf);
        Real grad = 0.0;
        if (gate) {
          const Real gx = (rxp - rxm)/(2.0*dx), gy = (ryp - rym)/(2.0*dy);
          const Real gz = (rzp - rzm)/(2.0*dz);
          grad = sqrt(gx*gx + gy*gy + gz*gz);
          if (!Kokkos::isfinite(grad)) grad = 0.0;
        }
        int la, lw, lj;
        // de-refinement: the density thresholds are scaled DOWN (sc < 1, more cells qualify:
        // want_deref >= want_ref), but the resolution tests need MORE cells, N/sc, so that
        // the coarser level still holds >= N cells. sc*N (before R-140 item 9) gave
        // want_deref = want_ref - 1 at sc = 1/2: blocks at want_ref derefined after
        // amr_min_residence and were refined back (c130). sc = 1: N/1 = N exactly.
        TDEDebrisCellWant(rho, prs, grad, gate, sc*rho_a, rf, lnf, ladder, lcore, lfloor,
                          lstream, nh/sc, nj/sc, dx_root, gam, la, lw, lj);
        const int c = (q - 1)%3;
        const int l = (c == 0) ? la : ((c == 1) ? lw : lj);
        lmax = fmax(lmax, static_cast<Real>(l));
      }, Kokkos::Max<Real>(team_max));
      Kokkos::single(Kokkos::PerTeam(tmember), [&]() { red(m, q) = team_max; });
    }
  });
  auto red_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), red);
  out = HostArray2D<Real>("tde_debris_red_h", nmb, kDebNRed);
  for (int m = 0; m < nmb; ++m)
    for (int q = 0; q < kDebNRed; ++q) out(m, q) = red_h(m, q);
}

//! \fn void TDERefineDebris(MeshBlockPack *pmbp)
//! \brief the tde_debris criterion (header comment). In dry-run mode the flags are only
//! logged; the caller has already set the mesh's flags with TDERefineTracker.
void TDERefineDebrisImpl(MeshBlockPack *pmbp, bool apply) {
  Mesh *pmesh = pmbp->pmesh;
  auto &refine_flag = pmesh->pmr->refine_flag;
  auto &size = pmbp->pmb->mb_size;
  const int nmb = pmbp->nmb_thispack;
  const int mbs = pmesh->gids_eachrank[global_variable::my_rank];
  const TDEDebrisParams &p = tde_deb;

  HostArray2D<Real> red;
  TDEDebrisBlockWants(pmbp, p, red);

  Real xbh[3] = {0.0, 0.0, 0.0};
  if (pmbp->pcfc != nullptr) { pmbp->pcfc->BHPosition(pmesh->time, xbh); }
  const Real r2_bh = SQR(tde_ref.rad_bh), r2_bh_fine = SQR(tde_ref.rad_bh_fine);
  const Real rho_a = p.core_frac*p.rho_ref;

  // residence (C-233): the first check cycle at which each block existed, for every block of
  // the mesh (identical on all ranks); a staged restart state is applied first
  if (!tde_deb_rst_state.empty()) TDEDebrisApplyResidence(pmesh);
  static bool warned_nan = false;
  std::map<std::array<int, 4>, int> seen_now;
  for (int g = 0; g < pmesh->nmb_total; ++g) {
    const LogicalLocation &lg = pmesh->lloc_eachmb[g];
    const std::array<int, 4> kg = {lg.lx1, lg.lx2, lg.lx3, lg.level};
    auto ig = tde_deb_first_seen.find(kg);
    seen_now[kg] = (ig == tde_deb_first_seen.end()) ? pmesh->ncycle : ig->second;
  }

  // wants are clipped to the mesh's max level only when applied; a dry run reports
  // what deeper levels would need (sizing on runs with fewer levels)
  const int maxl = pmesh->max_level - pmesh->root_level;
  const int topl = std::max({maxl, p.lev_core, p.lev_stream, tde_ref.lev_bh,
                             (tde_ref.rad_bh_fine > 0.0) ? tde_ref.lev_bh_fine : 0});
  const int cap = apply ? maxl : topl;
  const int nl = topl + 1;
  // diagnostics: [crit][level] blocks whose want_ref was set by crit at that want level,
  // blocks per current level, flags, projected blocks per level (no 2:1 buffers)
  std::vector<Real> d_crit(kDebNCrit*(nl + 1), 0.0), d_lev(nl, 0.0), d_proj(nl, 0.0);
  Real d_flag[3] = {0.0, 0.0, 0.0};   // -1, 0, +1
  Real n_nan = 0.0;

  for (int m = 0; m < nmb; ++m) {
    const LogicalLocation &ll = pmesh->lloc_eachmb[m + mbs];
    const int level = ll.level - pmesh->root_level;
    const std::array<int, 4> key = {ll.lx1, ll.lx2, ll.lx3, ll.level};
    const int residence = pmesh->ncycle - seen_now[key];

    // BH box distance (tde_track's AABB form)
    auto dist2 = [&](const Real px, const Real py, const Real pz) {
      const Real cx = std::fmax(size.h_view(m).x1min, std::fmin(px, size.h_view(m).x1max));
      const Real cy = std::fmax(size.h_view(m).x2min, std::fmin(py, size.h_view(m).x2max));
      const Real cz = std::fmax(size.h_view(m).x3min, std::fmin(pz, size.h_view(m).x3max));
      return SQR(px - cx) + SQR(py - cy) + SQR(pz - cz);
    };
    const Real d2 = dist2(xbh[0], xbh[1], xbh[2]);
    int want_bh = -1;
    if (d2 < r2_bh) want_bh = tde_ref.lev_bh;
    if (r2_bh_fine > 0.0 && d2 < r2_bh_fine) want_bh = std::max(want_bh, tde_ref.lev_bh_fine);

    n_nan += red(m, kDebNRed - 1);
    const Real bmax = red(m, 0);
    const int core_r = (bmax > rho_a) ? p.lev_core : -1;
    const int core_d = (bmax > p.deref_factor*rho_a) ? p.lev_core : -1;
    const int cand_r[5] = {core_r, static_cast<int>(red(m, 1)), static_cast<int>(red(m, 2)),
                           static_cast<int>(red(m, 3)), want_bh};
    const int cand_d[5] = {core_d, static_cast<int>(red(m, 4)), static_cast<int>(red(m, 5)),
                           static_cast<int>(red(m, 6)), want_bh};
    int want_r = -1, want_d = -1, crit = kDebNone;
    for (int c = 0; c < 5; ++c) {
      if (cand_r[c] > want_r) { want_r = cand_r[c]; crit = c; }
      want_d = std::max(want_d, cand_d[c]);
    }
    want_r = std::min(want_r, cap);
    want_d = std::min(want_d, cap);

    int flag;
    if (level < want_r) {
      flag = 1;
    } else if (level > want_d && residence >= p.min_residence) {
      flag = -1;
    } else {
      flag = 0;
    }
    if (apply) refine_flag.h_view(m + mbs) = flag;

    d_flag[flag + 1] += 1.0;
    d_lev[level] += 1.0;
    d_crit[crit*(nl + 1) + ((want_r >= 0) ? want_r : nl)] += 1.0;
    if (want_r > level) {
      d_proj[want_r] += std::pow(8.0, want_r - level);
    } else {
      const int keep = std::max(want_d, 0);
      const int lt = (level > keep) ? keep : level;
      d_proj[lt] += std::pow(8.0, -(level - lt));
    }
  }
  tde_deb_first_seen.swap(seen_now);

  if (apply) {
    refine_flag.template modify<HostMemSpace>();
    refine_flag.template sync<DevExeSpace>();
  }

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &n_nan, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (n_nan > 0.0 && !warned_nan && global_variable::my_rank == 0) {
    std::cout << "### WARNING tde_debris: " << n_nan << " non-finite density cells skipped "
              << "by the refinement criterion (warned once)" << std::endl;
    warned_nan = true;
  }

  if (p.diag_ncycle > 0 && (pmesh->ncycle % p.diag_ncycle == 0)) {
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, d_crit.data(), d_crit.size(), MPI_ATHENA_REAL, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, d_lev.data(), nl, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, d_proj.data(), nl, MPI_ATHENA_REAL, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, d_flag, 3, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
    if (global_variable::my_rank == 0) {
      static const char *cn[kDebNCrit] = {"core", "ladder", "width", "jeans", "bh", "none"};
      std::ostringstream os;
      os << "tde_debris" << (apply ? "" : " [dryrun]") << " cycle=" << pmesh->ncycle
         << " time=" << std::setprecision(8) << pmesh->time << std::setprecision(6)
         << " blocks/level:";
      for (int l = 0; l < nl; ++l) os << " " << static_cast<long>(d_lev[l]);
      os << " | flags -1/0/+1: " << static_cast<long>(d_flag[0]) << "/"
         << static_cast<long>(d_flag[1]) << "/" << static_cast<long>(d_flag[2])
         << " | projected/level:";
      for (int l = 0; l < nl; ++l) os << " " << static_cast<long>(d_proj[l] + 0.5);
      os << " | want_ref set by (crit: per want level 0.." << topl << ", none):";
      for (int c = 0; c < kDebNCrit; ++c) {
        os << " " << cn[c] << "[";
        for (int l = 0; l <= nl; ++l) {
          os << static_cast<long>(d_crit[c*(nl + 1) + l]) << (l < nl ? "," : "]");
        }
      }
      std::cout << os.str() << std::endl;
    }
  }
}

}  // namespace

//! \fn void TDEDebrisSaveResidence(Mesh *pm, ParameterInput *pin)
//! \brief restart writer hook (C-233): <problem> amr_residence_state = "v1 ncycle nmb_total
//! min_residence hash;" + one "lx1,lx2,lx3,level,c0;" per block that matters: resident fewer
//! than min_residence cycles (c0), or created by this cycle's regrid, after the check (c0 = -1).
//! Every other block is resident >= min_residence. Identical on all ranks.
void TDEDebrisSaveResidence(Mesh *pm, ParameterInput *pin) {
  std::ostringstream os;
  os << "v1 " << pm->ncycle << " " << pm->nmb_total << " " << tde_deb.min_residence << " "
     << std::hex << TDEDebrisTreeHash(pm) << std::dec << ";";
  long nlist = 0;
  for (int g = 0; g < pm->nmb_total; ++g) {
    const LogicalLocation &ll = pm->lloc_eachmb[g];
    const std::array<int, 4> key = {ll.lx1, ll.lx2, ll.lx3, ll.level};
    auto it = tde_deb_first_seen.find(key);
    int c0;
    if (it == tde_deb_first_seen.end()) {
      c0 = -1;
    } else if (pm->ncycle - it->second < tde_deb.min_residence) {
      c0 = it->second;
    } else {
      continue;
    }
    os << ll.lx1 << "," << ll.lx2 << "," << ll.lx3 << "," << ll.level << "," << c0 << ";";
    ++nlist;
  }
  if (os.str().size() > 512*1024 && global_variable::my_rank == 0) {
    std::cout << "### WARNING tde_debris: amr_residence_state is " << os.str().size()
              << " bytes (" << nlist << " blocks); the parameter copy is read up to 1 MB"
              << std::endl;
  }
  pin->SetString("problem", "amr_residence_state", os.str());
}

//! \fn void TDEDebrisStageResidence(ParameterInput *pin, bool restart)
//! \brief at a restart, stage the saved residence for the first check (C-233)
void TDEDebrisStageResidence(ParameterInput *pin, bool restart) {
  tde_deb_rst_state.clear();
  if (restart && pin->DoesParameterExist("problem", "amr_residence_state")) {
    tde_deb_rst_state = pin->GetString("problem", "amr_residence_state");
    if (tde_deb_rst_state.empty()) tde_deb_rst_state = " ";   // unreadable -> warning
  }
}

void TDERefineDebris(MeshBlockPack *pmbp) {
  if (tde_deb.dryrun) {
    TDERefineTracker(pmbp);              // the mesh follows tde_track
    TDERefineDebrisImpl(pmbp, false);    // log only
  } else {
    TDERefineDebrisImpl(pmbp, true);
  }
}

#endif  // PGEN_DYN_GRMHD_TDE_DEBRIS_REFINE_HPP_
