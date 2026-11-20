//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file multigrid_bvals.cpp
//! \brief implementation of MultigridBoundaryValues: boundary communication for the
//!        multigrid solver (fill coarse, prolongate, pack/send, recv/unpack, init recv)

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../athena.hpp"
#include "../globals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../coordinates/cell_locations.hpp"
#include "../mesh/mesh.hpp"
#include "../mesh/nghbr_index.hpp"
#include "../parameter_input.hpp"
#include "multigrid.hpp"

//----------------------------------------------------------------------------------------
//! \fn MultigridBoundaryValues::MultigridBoundaryValues()
//! \brief Constructor for multigrid boundary values object
//----------------------------------------------------------------------------------------

MultigridBoundaryValues::MultigridBoundaryValues(
    MeshBlockPack *pmbp, ParameterInput *pin,
    bool coarse, Multigrid *pmg)
  : MeshBoundaryValuesCC(pmbp, pin, coarse), pmy_mg(pmg) {
  // Multigrid boundary communications always use the rank-packed scheme; there is
  // no runtime switch to select a different communication path.
  // <cfc> mg_packlist_check: read only if present, so it never enters the parameter
  // dump (src/cfc/DEVELOPMENT.md item 64).
  if (pin->DoesParameterExist("cfc", "mg_packlist_check")) {
    mg_packlist_check_ = (pin->GetInteger("cfc", "mg_packlist_check") != 0) ? 1 : 0;
    if (global_variable::my_rank == 0 && mg_packlist_check_ > 0) {
      std::cout << "MGPACKLIST self-check on (<cfc> mg_packlist_check)" << std::endl;
    }
  }
  // src/cfc/DEVELOPMENT.md item 67: the coarser-face prolongation (ProlongateFCMG) always
  // clears the level's part of coarse_buf_ before each fill and reads the coarse face
  // neighbour's transverse range only where filled (proj/tde aa824220's rule), for every
  // multigrid solver. The former switch <cfc> mg_fc_fix (0 = old rule) was removed on the
  // user's decision (cfc_opt, 2026-11-20); a deck that still sets it to 2 is accepted, any
  // other value stops the run so the old rule is never assumed silently.
  // <cfc> mg_fc_prefill_nan = 1 (debug, read only if present) clears with NaN instead of 0.
  if (pin->DoesParameterExist("cfc", "mg_fc_fix") &&
      pin->GetInteger("cfc", "mg_fc_fix") != 2) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<cfc> mg_fc_fix was removed: the multigrid coarser-face rule of former "
              << "mode 2 is always used. Delete the key (or set it to 2)." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pin->DoesParameterExist("cfc", "mg_fc_prefill_nan")) {
    mg_fc_prefill_nan_ = (pin->GetInteger("cfc", "mg_fc_prefill_nan") != 0) ? 1 : 0;
  }
  if (global_variable::my_rank == 0 && mg_fc_prefill_nan_ > 0) {
    std::cout << "MGFCFIX coarse_buf_ prefill NaN (<cfc> mg_fc_prefill_nan)" << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn MultigridBoundaryValues::~MultigridBoundaryValues()
//! \brief Destructor for multigrid boundary values object. Provides the out-of-line
//! anchor (key function) so the vtable is emitted now that the base MeshBoundaryValues
//! destructor is virtual.

MultigridBoundaryValues::~MultigridBoundaryValues() {
}

#if MPI_PARALLEL_ENABLED
void MultigridBoundaryValues::BuildRankPackedMGMetadata(const int nvars, const int lev,
                                                        const bool skip_fc) {
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;

  mg_send_var_entries_.clear();
  mg_recv_var_entries_.clear();
  mg_send_var_msgs_.clear();
  mg_recv_var_msgs_.clear();
  mg_send_var_reqs_.clear();
  mg_recv_var_reqs_.clear();
  mg_send_var_hdr_reqs_.clear();
  mg_recv_var_hdr_reqs_.clear();

  std::map<int, std::vector<RankPackedVarEntry>> send_by_rank;
  std::map<int, std::vector<RankPackedVarEntry>> recv_by_rank;

  for (int m = 0; m < nmb; ++m) {
    for (int n = 0; n < nnghbr; ++n) {
      if (nghbr.h_view(m, n).gid < 0) continue;
      int nlev = nghbr.h_view(m, n).lev;
      int mlev = mblev.h_view(m);
      bool is_fc = (nlev != mlev);
      if (is_fc && skip_fc) continue;
      int drank = nghbr.h_view(m, n).rank;
      if (drank == my_rank) continue;
      int lid = nghbr.h_view(m, n).gid - pmy_pack->pmesh->gids_eachrank[drank];
      int dn = nghbr.h_view(m, n).dest;

      int send_size = 0;
      int recv_size = 0;
      if (nlev < mlev) {
        send_size = nvars * send_mg_indcs_[n][lev].icoar_ndat;
        recv_size = nvars * recv_mg_indcs_[n][lev].icoar_ndat;
      } else if (nlev == mlev) {
        send_size = nvars * send_mg_indcs_[n][lev].isame_ndat;
        recv_size = nvars * recv_mg_indcs_[n][lev].isame_ndat;
      } else {
        send_size = nvars * send_mg_indcs_[n][lev].ifine_ndat;
        recv_size = nvars * recv_mg_indcs_[n][lev].ifine_ndat;
      }

      if (send_size > 0) {
        send_by_rank[drank].push_back({m, n, lid, dn, send_size, 0});
      }
      if (recv_size > 0) {
        recv_by_rank[drank].push_back({m, n, 0, 0, recv_size, 0});
      }
    }
  }

  // Deterministic, rank-agnostic payload ordering so the two ranks of a pair
  // agree on the byte layout WITHOUT exchanging headers. For a given face
  // exchange, the sender's (lid,dn) == the receiver's (m,n) (same receiving
  // block + neighbour slot), so sorting sends by (lid,dn) and recvs by (m,n)
  // produces identical orderings on both sides. This lets the receiver build
  // its unpack offsets purely locally (see mg_recv_agg_offset_ below).
  for (auto &kv : send_by_rank) {
    std::sort(kv.second.begin(), kv.second.end(),
              [](const RankPackedVarEntry &a, const RankPackedVarEntry &b) {
                return (a.lid != b.lid) ? (a.lid < b.lid) : (a.dn < b.dn);
              });
  }
  for (auto &kv : recv_by_rank) {
    std::sort(kv.second.begin(), kv.second.end(),
              [](const RankPackedVarEntry &a, const RankPackedVarEntry &b) {
                return (a.m != b.m) ? (a.m < b.m) : (a.n < b.n);
              });
  }

  int send_data_off = 0;
  int send_entry_off = 0;
  int send_hdr_off = 0;
  for (auto &kv : send_by_rank) {
    RankPackedVarMessage msg;
    msg.rank = kv.first;
    msg.nentries = static_cast<int>(kv.second.size());
    msg.entry_offset = send_entry_off;
    msg.hdr_offset = send_hdr_off;
    msg.offset = send_data_off;
    msg.data_size = 0;
    for (auto &e : kv.second) {
      e.offset = send_data_off;
      msg.data_size += e.data_size;
      send_data_off += e.data_size;
      mg_send_var_entries_.push_back(e);
      ++send_entry_off;
    }
    send_hdr_off += 3 * msg.nentries;
    mg_send_var_msgs_.push_back(msg);
  }

  int recv_data_off = 0;
  int recv_entry_off = 0;
  int recv_hdr_off = 0;
  for (auto &kv : recv_by_rank) {
    RankPackedVarMessage msg;
    msg.rank = kv.first;
    msg.nentries = static_cast<int>(kv.second.size());
    msg.entry_offset = recv_entry_off;
    msg.hdr_offset = recv_hdr_off;
    msg.offset = recv_data_off;
    msg.data_size = 0;
    for (auto &e : kv.second) {
      e.offset = recv_data_off;
      msg.data_size += e.data_size;
      recv_data_off += e.data_size;
      mg_recv_var_entries_.push_back(e);
      ++recv_entry_off;
    }
    recv_hdr_off += 3 * msg.nentries;
    mg_recv_var_msgs_.push_back(msg);
  }

  mg_rank_sendbuf_vars_ = DvceArray1D<Real>("rank_sendbuf_mg", send_data_off);
  mg_rank_recvbuf_vars_ = DvceArray1D<Real>("rank_recvbuf_mg", recv_data_off);
  mg_rank_sendhdr_vars_ = DvceArray1D<int>("rank_sendhdr_mg", send_hdr_off);
  mg_rank_recvhdr_vars_ = DvceArray1D<int>("rank_recvhdr_mg", recv_hdr_off);

  // Mirror the entry tables to device for the fused pack/unpack kernels.
  {
    const std::size_t nsend = mg_send_var_entries_.size();
    const std::size_t nrecv = mg_recv_var_entries_.size();
    mg_send_var_entries_d_ =
        DvceArray1D<RankPackedVarEntry>("mg_send_var_entries_d", nsend);
    mg_recv_var_entries_d_ =
        DvceArray1D<RankPackedVarEntry>("mg_recv_var_entries_d", nrecv);
    auto h_send = Kokkos::create_mirror_view(mg_send_var_entries_d_);
    auto h_recv = Kokkos::create_mirror_view(mg_recv_var_entries_d_);
    for (std::size_t i = 0; i < nsend; ++i) h_send(i) = mg_send_var_entries_[i];
    for (std::size_t i = 0; i < nrecv; ++i) h_recv(i) = mg_recv_var_entries_[i];
    Kokkos::deep_copy(mg_send_var_entries_d_, h_send);
    Kokkos::deep_copy(mg_recv_var_entries_d_, h_recv);
  }

  // Per-(m,n) base offsets into the aggregate buffers. -1 == on-rank / no
  // neighbour / skipped. The pack/unpack kernels use these to read/write the
  // aggregate buffer directly, so no companion gather/scatter kernel is needed.
  // Recv offsets come straight from this rank's own sorted recv entries (no
  // header needed): the sort guarantees they match the sender's layout.
  {
    const int map_len = nmb*nnghbr;
    mg_send_agg_offset_ = DvceArray1D<int>("mg_send_agg_offset", std::max(1, map_len));
    mg_recv_agg_offset_ = DvceArray1D<int>("mg_recv_agg_offset", std::max(1, map_len));
    auto send_off_h = Kokkos::create_mirror_view(mg_send_agg_offset_);
    auto recv_off_h = Kokkos::create_mirror_view(mg_recv_agg_offset_);
    for (int i = 0; i < std::max(1, map_len); ++i) {
      send_off_h(i) = -1;
      recv_off_h(i) = -1;
    }
    for (const auto &e : mg_send_var_entries_) {
      send_off_h(e.m*nnghbr + e.n) = e.offset;
    }
    for (const auto &e : mg_recv_var_entries_) {
      recv_off_h(e.m*nnghbr + e.n) = e.offset;
    }
    Kokkos::deep_copy(mg_send_agg_offset_, send_off_h);
    Kokkos::deep_copy(mg_recv_agg_offset_, recv_off_h);
  }

  mg_send_var_reqs_.assign(mg_send_var_msgs_.size(), MPI_REQUEST_NULL);
  mg_recv_var_reqs_.assign(mg_recv_var_msgs_.size(), MPI_REQUEST_NULL);
  mg_send_var_hdr_reqs_.assign(mg_send_var_msgs_.size(), MPI_REQUEST_NULL);
  mg_recv_var_hdr_reqs_.assign(mg_recv_var_msgs_.size(), MPI_REQUEST_NULL);
  ++mg_buf_gen_;   // invalidates the pack lists (see GetPackList)
}
#endif


//----------------------------------------------------------------------------------------
//! \fn void MultigridBoundaryValues::RemapIndicesForMG()
//! \brief Remap isame/icoar/ifine indices from hydro coordinates (ng ghost cells) to MG
//! coordinates (ngh_ ghost cells). Must be called AFTER InitializeBuffers.

void MultigridBoundaryValues::RemapIndicesForMG() {
  int ng  = pmy_pack->pmesh->mb_indcs.ng;
  int ngh = pmy_mg->GetGhostCells();
  int nx1 = pmy_pack->pmesh->mb_indcs.nx1;
  int nx2 = pmy_pack->pmesh->mb_indcs.nx2;
  int nx3 = pmy_pack->pmesh->mb_indcs.nx3;
  int nnghbr = pmy_pack->pmb->nnghbr;

  if (ng != ngh) {
    int is_h = ng, ie_h = ng + nx1 - 1;
    int js_h = ng, je_h = ng + nx2 - 1;
    int ks_h = ng, ke_h = ng + nx3 - 1;
    int is_m = ngh, ie_m = ngh + nx1 - 1;
    int js_m = ngh, je_m = ngh + nx2 - 1;
    int ks_m = ngh, ke_m = ngh + nx3 - 1;
    int ng1_m = ngh - 1;

    auto remap_send = [](int &lo, int &hi,
                         int s_h, int e_h,
                         int s_m, int e_m, int ng1) {
      if (lo == s_h && hi == e_h) {
        lo = s_m; hi = e_m;
      } else if (lo > s_h) {
        lo = e_m - ng1; hi = e_m;
      } else {
        lo = s_m; hi = s_m + ng1;
      }
    };
    auto remap_recv = [](int &lo, int &hi,
                         int s_h, int e_h,
                         int s_m, int e_m, int ng_m) {
      if (lo >= s_h && hi <= e_h) {
        lo = s_m; hi = e_m;
      } else if (lo > e_h) {
        lo = e_m + 1; hi = e_m + ng_m;
      } else {
        lo = s_m - ng_m; hi = s_m - 1;
      }
    };

    for (int n = 0; n < nnghbr; ++n) {
      auto &si = sendbuf[n].isame[0];
      remap_send(si.bis, si.bie, is_h, ie_h, is_m, ie_m, ng1_m);
      remap_send(si.bjs, si.bje, js_h, je_h, js_m, je_m, ng1_m);
      remap_send(si.bks, si.bke, ks_h, ke_h, ks_m, ke_m, ng1_m);
      sendbuf[n].isame_ndat = (si.bie-si.bis+1)*(si.bje-si.bjs+1)*(si.bke-si.bks+1);

      auto &ri = recvbuf[n].isame[0];
      remap_recv(ri.bis, ri.bie, is_h, ie_h, is_m, ie_m, ngh);
      remap_recv(ri.bjs, ri.bje, js_h, je_h, js_m, je_m, ngh);
      remap_recv(ri.bks, ri.bke, ks_h, ke_h, ks_m, ke_m, ngh);
      recvbuf[n].isame_ndat = (ri.bie-ri.bis+1)*(ri.bje-ri.bjs+1)*(ri.bke-ri.bks+1);
    }
  }

  // Recompute icoar/ifine indices from scratch using MG mesh parameters.
  // These are needed for inter-level (fine-coarse) boundary communication.
  int ng1_m = ngh - 1;
  int cnx1 = nx1 / 2, cnx2 = nx2 / 2, cnx3 = nx3 / 2;
  int is_m = ngh, ie_m = ngh + nx1 - 1;
  int js_m = ngh, je_m = ngh + nx2 - 1;
  int ks_m = ngh, ke_m = ngh + nx3 - 1;
  int cis_m = ngh, cie_m = ngh + cnx1 - 1;
  int cjs_m = ngh, cje_m = ngh + cnx2 - 1;
  int cks_m = ngh, cke_m = ngh + cnx3 - 1;

  // Recover (ox1,ox2,ox3,f1,f2) from the buffer index n and recompute icoar/ifine.
  // We iterate the same way InitializeBuffers does.
  int nfx = 1, nfy = 1, nfz = 1;
  if (pmy_pack->pmesh->multilevel) {
    nfx = 2;
    if (pmy_pack->pmesh->multi_d) nfy = 2;
    if (pmy_pack->pmesh->three_d) nfz = 2;
  }

  auto compute_send_icoar = [&](MeshBoundaryBuffer &buf,
                                int ox1, int ox2, int ox3) {
    auto &ic = buf.icoar[0];
    ic.bis = (ox1 > 0) ? (cie_m - ng1_m) : cis_m;
    ic.bie = (ox1 < 0) ? (cis_m + ng1_m) : cie_m;
    ic.bjs = (ox2 > 0) ? (cje_m - ng1_m) : cjs_m;
    ic.bje = (ox2 < 0) ? (cjs_m + ng1_m) : cje_m;
    ic.bks = (ox3 > 0) ? (cke_m - ng1_m) : cks_m;
    ic.bke = (ox3 < 0) ? (cks_m + ng1_m) : cke_m;
    buf.icoar_ndat = (ic.bie-ic.bis+1)*(ic.bje-ic.bjs+1)*(ic.bke-ic.bks+1);
  };

  auto compute_send_ifine = [&](MeshBoundaryBuffer &buf,
                                int ox1, int ox2, int ox3, int f1, int f2) {
    auto &ifn = buf.ifine[0];
    ifn.bis = (ox1 > 0) ? (ie_m - ng1_m) : is_m;
    ifn.bie = (ox1 < 0) ? (is_m + ng1_m) : ie_m;
    ifn.bjs = (ox2 > 0) ? (je_m - ng1_m) : js_m;
    ifn.bje = (ox2 < 0) ? (js_m + ng1_m) : je_m;
    ifn.bks = (ox3 > 0) ? (ke_m - ng1_m) : ks_m;
    ifn.bke = (ox3 < 0) ? (ks_m + ng1_m) : ke_m;
    if (ox1 == 0) {
      if (f1 == 1) {
        ifn.bis += cnx1 - ngh;
      } else {
        ifn.bie -= cnx1 - ngh;
      }
    }
    if (ox2 == 0 && nx2 > 1) {
      if (ox1 != 0) {
        if (f1 == 1) {
          ifn.bjs += cnx2 - ngh;
        } else {
          ifn.bje -= cnx2 - ngh;
        }
      } else {
        if (f2 == 1) {
          ifn.bjs += cnx2 - ngh;
        } else {
          ifn.bje -= cnx2 - ngh;
        }
      }
    }
    if (ox3 == 0 && nx3 > 1) {
      if (ox1 != 0 && ox2 != 0) {
        if (f1 == 1) {
          ifn.bks += cnx3 - ngh;
        } else {
          ifn.bke -= cnx3 - ngh;
        }
      } else {
        if (f2 == 1) {
          ifn.bks += cnx3 - ngh;
        } else {
          ifn.bke -= cnx3 - ngh;
        }
      }
    }
    buf.ifine_ndat =
        (ifn.bie-ifn.bis+1)
        * (ifn.bje-ifn.bjs+1)
        * (ifn.bke-ifn.bks+1);
  };

  auto compute_recv_icoar = [&](MeshBoundaryBuffer &buf,
                                int ox1, int ox2, int ox3,
                                int f1, int f2) {
    auto &ic = buf.icoar[0];
    if (ox1 == 0) {
      ic.bis = cis_m; ic.bie = cie_m;
      if (f1 == 0) {
        ic.bie += ngh;
      } else {
        ic.bis -= ngh;
      }
    } else if (ox1 > 0) {
      ic.bis = cie_m + 1; ic.bie = cie_m + ngh;
    } else {
      ic.bis = cis_m - ngh; ic.bie = cis_m - 1;
    }
    if (ox2 == 0) {
      ic.bjs = cjs_m; ic.bje = cje_m;
      if (nx2 > 1) {
        if (ox1 != 0) {
          if (f1 == 0) {
            ic.bje += ngh;
          } else {
            ic.bjs -= ngh;
          }
        } else {
          if (f2 == 0) {
            ic.bje += ngh;
          } else {
            ic.bjs -= ngh;
          }
        }
      }
    } else if (ox2 > 0) {
      ic.bjs = cje_m + 1; ic.bje = cje_m + ngh;
    } else {
      ic.bjs = cjs_m - ngh; ic.bje = cjs_m - 1;
    }
    if (ox3 == 0) {
      ic.bks = cks_m; ic.bke = cke_m;
      if (nx3 > 1) {
        if (ox1 != 0 && ox2 != 0) {
          if (f1 == 0) {
            ic.bke += ngh;
          } else {
            ic.bks -= ngh;
          }
        } else {
          if (f2 == 0) {
            ic.bke += ngh;
          } else {
            ic.bks -= ngh;
          }
        }
      }
    } else if (ox3 > 0) {
      ic.bks = cke_m + 1; ic.bke = cke_m + ngh;
    } else {
      ic.bks = cks_m - ngh; ic.bke = cks_m - 1;
    }
    buf.icoar_ndat =
        (ic.bie-ic.bis+1)
        * (ic.bje-ic.bjs+1)
        * (ic.bke-ic.bks+1);
  };

  auto compute_recv_ifine = [&](MeshBoundaryBuffer &buf,
                                int ox1, int ox2, int ox3,
                                int f1, int f2) {
    auto &ifn = buf.ifine[0];
    if (ox1 == 0) {
      ifn.bis = is_m; ifn.bie = ie_m;
      if (f1 == 1) {
        ifn.bis += cnx1;
      } else {
        ifn.bie -= cnx1;
      }
    } else if (ox1 > 0) {
      ifn.bis = ie_m + 1; ifn.bie = ie_m + ngh;
    } else {
      ifn.bis = is_m - ngh; ifn.bie = is_m - 1;
    }
    if (ox2 == 0) {
      ifn.bjs = js_m; ifn.bje = je_m;
      if (nx2 > 1) {
        if (ox1 != 0) {
          if (f1 == 1) {
            ifn.bjs += cnx2;
          } else {
            ifn.bje -= cnx2;
          }
        } else {
          if (f2 == 1) {
            ifn.bjs += cnx2;
          } else {
            ifn.bje -= cnx2;
          }
        }
      }
    } else if (ox2 > 0) {
      ifn.bjs = je_m + 1; ifn.bje = je_m + ngh;
    } else {
      ifn.bjs = js_m - ngh; ifn.bje = js_m - 1;
    }
    if (ox3 == 0) {
      ifn.bks = ks_m; ifn.bke = ke_m;
      if (nx3 > 1) {
        if (ox1 != 0 && ox2 != 0) {
          if (f1 == 1) {
            ifn.bks += cnx3;
          } else {
            ifn.bke -= cnx3;
          }
        } else {
          if (f2 == 1) {
            ifn.bks += cnx3;
          } else {
            ifn.bke -= cnx3;
          }
        }
      }
    } else if (ox3 > 0) {
      ifn.bks = ke_m + 1; ifn.bke = ke_m + ngh;
    } else {
      ifn.bks = ks_m - ngh; ifn.bke = ks_m - 1;
    }
    buf.ifine_ndat =
        (ifn.bie-ifn.bis+1)
        * (ifn.bje-ifn.bjs+1)
        * (ifn.bke-ifn.bks+1);
  };

  // Iterate over all buffer directions (mirrors InitializeBuffers order)
  // x1 faces
  for (int n=-1; n<=1; n+=2) {
    for (int fz=0; fz<nfz; fz++) {
      for (int fy=0; fy<nfy; fy++) {
        int idx = NeighborIndex(n,0,0,fy,fz);
        compute_send_icoar(sendbuf[idx], n, 0, 0);
        compute_send_ifine(sendbuf[idx], n, 0, 0, fy, fz);
        compute_recv_icoar(recvbuf[idx], n, 0, 0, fy, fz);
        compute_recv_ifine(recvbuf[idx], n, 0, 0, fy, fz);
      }
    }
  }
  if (pmy_pack->pmesh->multi_d) {
    // x2 faces
    for (int m=-1; m<=1; m+=2) {
      for (int fz=0; fz<nfz; fz++) {
        for (int fx=0; fx<nfx; fx++) {
          int idx = NeighborIndex(0,m,0,fx,fz);
          compute_send_icoar(sendbuf[idx], 0, m, 0);
          compute_send_ifine(sendbuf[idx], 0, m, 0, fx, fz);
          compute_recv_icoar(recvbuf[idx], 0, m, 0, fx, fz);
          compute_recv_ifine(recvbuf[idx], 0, m, 0, fx, fz);
        }
      }
    }
    // x1x2 edges
    for (int m=-1; m<=1; m+=2) {
      for (int n=-1; n<=1; n+=2) {
        for (int fz=0; fz<nfz; fz++) {
          int idx = NeighborIndex(n,m,0,fz,0);
          compute_send_icoar(sendbuf[idx], n, m, 0);
          compute_send_ifine(sendbuf[idx], n, m, 0, fz, 0);
          compute_recv_icoar(recvbuf[idx], n, m, 0, fz, 0);
          compute_recv_ifine(recvbuf[idx], n, m, 0, fz, 0);
        }
      }
    }
  }
  if (pmy_pack->pmesh->three_d) {
    // x3 faces
    for (int l=-1; l<=1; l+=2) {
      for (int fy=0; fy<nfy; fy++) {
        for (int fx=0; fx<nfx; fx++) {
          int idx = NeighborIndex(0,0,l,fx,fy);
          compute_send_icoar(sendbuf[idx], 0, 0, l);
          compute_send_ifine(sendbuf[idx], 0, 0, l, fx, fy);
          compute_recv_icoar(recvbuf[idx], 0, 0, l, fx, fy);
          compute_recv_ifine(recvbuf[idx], 0, 0, l, fx, fy);
        }
      }
    }
    // x3x1 edges
    for (int l=-1; l<=1; l+=2) {
      for (int n=-1; n<=1; n+=2) {
        for (int fy=0; fy<nfy; fy++) {
          int idx = NeighborIndex(n,0,l,fy,0);
          compute_send_icoar(sendbuf[idx], n, 0, l);
          compute_send_ifine(sendbuf[idx], n, 0, l, fy, 0);
          compute_recv_icoar(recvbuf[idx], n, 0, l, fy, 0);
          compute_recv_ifine(recvbuf[idx], n, 0, l, fy, 0);
        }
      }
    }
    // x2x3 edges
    for (int l=-1; l<=1; l+=2) {
      for (int m=-1; m<=1; m+=2) {
        for (int fx=0; fx<nfx; fx++) {
          int idx = NeighborIndex(0,m,l,fx,0);
          compute_send_icoar(sendbuf[idx], 0, m, l);
          compute_send_ifine(sendbuf[idx], 0, m, l, fx, 0);
          compute_recv_icoar(recvbuf[idx], 0, m, l, fx, 0);
          compute_recv_ifine(recvbuf[idx], 0, m, l, fx, 0);
        }
      }
    }
    // corners
    for (int l=-1; l<=1; l+=2) {
      for (int m=-1; m<=1; m+=2) {
        for (int n=-1; n<=1; n+=2) {
          int idx = NeighborIndex(n,m,l,0,0);
          compute_send_icoar(sendbuf[idx], n, m, l);
          compute_send_ifine(sendbuf[idx], n, m, l, 0, 0);
          compute_recv_icoar(recvbuf[idx], n, m, l, 0, 0);
          compute_recv_ifine(recvbuf[idx], n, m, l, 0, 0);
        }
      }
    }
  }

  int nvar = pmy_mg->nvar_;
  int nmb = std::max(pmy_pack->nmb_thispack, pmy_pack->pmesh->nmb_maxperrank);
  if (pmy_pack->pmesh->multilevel) {
    for (int n = 0; n < nnghbr; ++n) {
      int smax = std::max(sendbuf[n].isame_ndat,
                   std::max(sendbuf[n].icoar_ndat, sendbuf[n].ifine_ndat));
      if (nvar * smax > sendbuf[n].vars.extent_int(1)) {
        Kokkos::realloc(sendbuf[n].vars, nmb, nvar * smax);
      }
      int rmax = std::max(recvbuf[n].isame_ndat,
                   std::max(recvbuf[n].icoar_ndat, recvbuf[n].ifine_ndat));
      if (nvar * rmax > recvbuf[n].vars.extent_int(1)) {
        Kokkos::realloc(recvbuf[n].vars, nmb, nvar * rmax);
      }
    }

    int cbnx3 = cnx3 + 2*ngh;
    int cbnx2 = cnx2 + 2*ngh;
    int cbnx1 = cnx1 + 2*ngh;
    Kokkos::realloc(coarse_buf_, nmb, nvar, cbnx3, cbnx2, cbnx1);
  }
  ++mg_buf_gen_;   // boxes/buffers may have changed (GetPackList)
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridBoundaryValues::ComputePerLevelIndices()
//! \brief Pre-compute isame/icoar/ifine send and recv indices for every MG level and
//! every neighbor direction.  This replaces the fragile runtime shift logic in
//! PackAndSendMG / RecvAndUnpackMG with exact, pre-computed values.  Uses the same index
//! formulas as buffs_cc.cpp but parameterized by MG ngh and the per-level cell count.
//! Must be called AFTER InitializeBuffers (so the 56 buffer slots exist).

void MultigridBoundaryValues::ComputePerLevelIndices() {
  int ngh    = pmy_mg->GetGhostCells();
  int ng1    = ngh - 1;
  int nx_max = pmy_mg->GetSize();            // finest-level cell count per direction
  int nlevel = pmy_mg->GetNumberOfLevels();

  bool md = pmy_pack->pmesh->multi_d;
  bool td = pmy_pack->pmesh->three_d;
  bool ml = pmy_pack->pmesh->multilevel;

  int nfx = ml ? 2 : 1;
  int nfy = (ml && md) ? 2 : 1;
  int nfz = (ml && td) ? 2 : 1;

  // Helper lambdas that mirror buffs_cc.cpp formulas but with MG parameters.
  // ncells = active cells in this direction at the current MG level.
  auto compute_send = [&](MGPerLevelIndcs &out,
                          int ox1, int ox2, int ox3, int f1, int f2,
                          int ncells) {
    int is_m  = ngh, ie_m  = ngh + ncells - 1;
    int js_m  = ngh, je_m  = ngh + ncells - 1;
    int ks_m  = ngh, ke_m  = ngh + ncells - 1;
    int cnx   = ncells / 2;
    int cis_m = ngh, cie_m = ngh + cnx - 1;
    int cjs_m = ngh, cje_m = ngh + cnx - 1;
    int cks_m = ngh, cke_m = ngh + cnx - 1;

    // -- isame (same-level send) --
    if (f1 == 0 && f2 == 0) {
      auto &s = out.isame;
      s.bis = (ox1 > 0) ? (ie_m - ng1) : is_m;
      s.bie = (ox1 < 0) ? (is_m + ng1) : ie_m;
      s.bjs = (ox2 > 0) ? (je_m - ng1) : js_m;
      s.bje = (ox2 < 0) ? (js_m + ng1) : je_m;
      s.bks = (ox3 > 0) ? (ke_m - ng1) : ks_m;
      s.bke = (ox3 < 0) ? (ks_m + ng1) : ke_m;
      out.isame_ndat = (s.bie-s.bis+1)*(s.bje-s.bjs+1)*(s.bke-s.bks+1);
    }

    // -- icoar (send to coarser) --
    // Face neighbors (nface==1): indices point to coarse_buf_ ghost cells
    // where face-aligned 2x2 averages are stored by FillCoarseMG.
    // Edge/corner neighbors (nface>1): indices point to coarse_buf_ interior
    // where volume averages are stored by FillCoarseMG.
    {
      auto &c = out.icoar;
      int nface = (ox1!=0?1:0) + (ox2!=0?1:0) + (ox3!=0?1:0);
      if (nface == 1) {
        c.bis = (ox1 > 0) ? (cie_m + 1)   : (ox1 < 0) ? (cis_m - ngh) : cis_m;
        c.bie = (ox1 > 0) ? (cie_m + ngh)  : (ox1 < 0) ? (cis_m - 1)  : cie_m;
        c.bjs = (ox2 > 0) ? (cje_m + 1)   : (ox2 < 0) ? (cjs_m - ngh) : cjs_m;
        c.bje = (ox2 > 0) ? (cje_m + ngh)  : (ox2 < 0) ? (cjs_m - 1)  : cje_m;
        c.bks = (ox3 > 0) ? (cke_m + 1)   : (ox3 < 0) ? (cks_m - ngh) : cks_m;
        c.bke = (ox3 > 0) ? (cke_m + ngh)  : (ox3 < 0) ? (cks_m - 1)  : cke_m;
      } else {
        c.bis = (ox1 > 0) ? (cie_m - ng1) : cis_m;
        c.bie = (ox1 < 0) ? (cis_m + ng1) : cie_m;
        c.bjs = (ox2 > 0) ? (cje_m - ng1) : cjs_m;
        c.bje = (ox2 < 0) ? (cjs_m + ng1) : cje_m;
        c.bks = (ox3 > 0) ? (cke_m - ng1) : cks_m;
        c.bke = (ox3 < 0) ? (cks_m + ng1) : cke_m;
      }
      out.icoar_ndat = (c.bie-c.bis+1)*(c.bje-c.bjs+1)*(c.bke-c.bks+1);
    }

    // -- ifine (send to finer) --
    {
      auto &f = out.ifine;
      f.bis = (ox1 > 0) ? (ie_m - ng1) : is_m;
      f.bie = (ox1 < 0) ? (is_m + ng1) : ie_m;
      f.bjs = (ox2 > 0) ? (je_m - ng1) : js_m;
      f.bje = (ox2 < 0) ? (js_m + ng1) : je_m;
      f.bks = (ox3 > 0) ? (ke_m - ng1) : ks_m;
      f.bke = (ox3 < 0) ? (ks_m + ng1) : ke_m;
      if (ox1 == 0) {
        if (f1 == 1) {
          f.bis += cnx - ngh;
        } else {
          f.bie -= cnx - ngh;
        }
      }
      if (ox2 == 0 && md) {
        if (ox1 != 0) {
          if (f1 == 1) {
            f.bjs += cnx - ngh;
          } else {
            f.bje -= cnx - ngh;
          }
        } else {
          if (f2 == 1) {
            f.bjs += cnx - ngh;
          } else {
            f.bje -= cnx - ngh;
          }
        }
      }
      if (ox3 == 0 && td) {
        if (ox1 != 0 && ox2 != 0) {
          if (f1 == 1) {
            f.bks += cnx - ngh;
          } else {
            f.bke -= cnx - ngh;
          }
        } else {
          if (f2 == 1) {
            f.bks += cnx - ngh;
          } else {
            f.bke -= cnx - ngh;
          }
        }
      }
      out.ifine_ndat =
          (f.bie-f.bis+1)
          * (f.bje-f.bjs+1)
          * (f.bke-f.bks+1);
    }
  };

  auto compute_recv = [&](MGPerLevelIndcs &out,
                          int ox1, int ox2, int ox3, int f1, int f2,
                          int ncells) {
    int is_m  = ngh, ie_m  = ngh + ncells - 1;
    int js_m  = ngh, je_m  = ngh + ncells - 1;
    int ks_m  = ngh, ke_m  = ngh + ncells - 1;
    int cnx   = ncells / 2;
    int cis_m = ngh, cie_m = ngh + cnx - 1;
    int cjs_m = ngh, cje_m = ngh + cnx - 1;
    int cks_m = ngh, cke_m = ngh + cnx - 1;

    // -- isame (same-level recv) --
    if (f1 == 0 && f2 == 0) {
      auto &s = out.isame;
      if (ox1 == 0) {
        s.bis = is_m; s.bie = ie_m;
      } else if (ox1 > 0) {
        s.bis = ie_m + 1; s.bie = ie_m + ngh;
      } else {
        s.bis = is_m - ngh; s.bie = is_m - 1;
      }
      if (ox2 == 0) {
        s.bjs = js_m; s.bje = je_m;
      } else if (ox2 > 0) {
        s.bjs = je_m + 1; s.bje = je_m + ngh;
      } else {
        s.bjs = js_m - ngh; s.bje = js_m - 1;
      }
      if (ox3 == 0) {
        s.bks = ks_m; s.bke = ke_m;
      } else if (ox3 > 0) {
        s.bks = ke_m + 1; s.bke = ke_m + ngh;
      } else {
        s.bks = ks_m - ngh; s.bke = ks_m - 1;
      }
      out.isame_ndat =
          (s.bie-s.bis+1)
          * (s.bje-s.bjs+1)
          * (s.bke-s.bks+1);
    }

    // -- icoar (recv from coarser, matches send-to-finer) --
    {
      auto &c = out.icoar;
      if (ox1 == 0) {
        c.bis = cis_m; c.bie = cie_m;
        if (f1 == 0) {
          c.bie += ngh;
        } else {
          c.bis -= ngh;
        }
      } else if (ox1 > 0) {
        c.bis = cie_m + 1; c.bie = cie_m + ngh;
      } else {
        c.bis = cis_m - ngh; c.bie = cis_m - 1;
      }

      if (ox2 == 0) {
        c.bjs = cjs_m; c.bje = cje_m;
        if (md) {
          if (ox1 != 0) {
            if (f1 == 0) {
              c.bje += ngh;
            } else {
              c.bjs -= ngh;
            }
          } else {
            if (f2 == 0) {
              c.bje += ngh;
            } else {
              c.bjs -= ngh;
            }
          }
        }
      } else if (ox2 > 0) {
        c.bjs = cje_m + 1; c.bje = cje_m + ngh;
      } else {
        c.bjs = cjs_m - ngh; c.bje = cjs_m - 1;
      }

      if (ox3 == 0) {
        c.bks = cks_m; c.bke = cke_m;
        if (td) {
          if (ox1 != 0 && ox2 != 0) {
            if (f1 == 0) {
              c.bke += ngh;
            } else {
              c.bks -= ngh;
            }
          } else {
            if (f2 == 0) {
              c.bke += ngh;
            } else {
              c.bks -= ngh;
            }
          }
        }
      } else if (ox3 > 0) {
        c.bks = cke_m + 1; c.bke = cke_m + ngh;
      } else {
        c.bks = cks_m - ngh; c.bke = cks_m - 1;
      }
      out.icoar_ndat =
          (c.bie-c.bis+1)
          * (c.bje-c.bjs+1)
          * (c.bke-c.bks+1);
    }

    // -- ifine (recv from finer, matches send-to-coarser) --
    {
      auto &fn = out.ifine;
      if (ox1 == 0) {
        fn.bis = is_m; fn.bie = ie_m;
        if (f1 == 1) {
          fn.bis += cnx;
        } else {
          fn.bie -= cnx;
        }
      } else if (ox1 > 0) {
        fn.bis = ie_m + 1; fn.bie = ie_m + ngh;
      } else {
        fn.bis = is_m - ngh; fn.bie = is_m - 1;
      }

      if (ox2 == 0) {
        fn.bjs = js_m; fn.bje = je_m;
        if (md) {
          if (ox1 != 0) {
            if (f1 == 1) {
              fn.bjs += cnx;
            } else {
              fn.bje -= cnx;
            }
          } else {
            if (f2 == 1) {
              fn.bjs += cnx;
            } else {
              fn.bje -= cnx;
            }
          }
        }
      } else if (ox2 > 0) {
        fn.bjs = je_m + 1; fn.bje = je_m + ngh;
      } else {
        fn.bjs = js_m - ngh; fn.bje = js_m - 1;
      }

      if (ox3 == 0) {
        fn.bks = ks_m; fn.bke = ke_m;
        if (td) {
          if (ox1 != 0 && ox2 != 0) {
            if (f1 == 1) {
              fn.bks += cnx;
            } else {
              fn.bke -= cnx;
            }
          } else {
            if (f2 == 1) {
              fn.bks += cnx;
            } else {
              fn.bke -= cnx;
            }
          }
        }
      } else if (ox3 > 0) {
        fn.bks = ke_m + 1; fn.bke = ke_m + ngh;
      } else {
        fn.bks = ks_m - ngh; fn.bke = ks_m - 1;
      }
      out.ifine_ndat =
          (fn.bie-fn.bis+1)
          * (fn.bje-fn.bjs+1)
          * (fn.bke-fn.bks+1);
    }
  };

  // Fill indices for each MG level and each neighbor direction.
  for (int lev = 0; lev < nlevel && lev < kMaxMGLevels; ++lev) {
    int shift = nlevel - 1 - lev;
    int ncells = nx_max >> shift;
    if (ncells < 1) ncells = 1;

    // x1 faces
    for (int n = -1; n <= 1; n += 2) {
      for (int fz = 0; fz < nfz; fz++) {
        for (int fy = 0; fy < nfy; fy++) {
          int idx = NeighborIndex(n, 0, 0, fy, fz);
          compute_send(send_mg_indcs_[idx][lev], n, 0, 0, fy, fz, ncells);
          compute_recv(recv_mg_indcs_[idx][lev], n, 0, 0, fy, fz, ncells);
        }
      }
    }
    if (md) {
      // x2 faces
      for (int m = -1; m <= 1; m += 2) {
        for (int fz = 0; fz < nfz; fz++) {
          for (int fx = 0; fx < nfx; fx++) {
            int idx = NeighborIndex(0, m, 0, fx, fz);
            compute_send(send_mg_indcs_[idx][lev], 0, m, 0, fx, fz, ncells);
            compute_recv(recv_mg_indcs_[idx][lev], 0, m, 0, fx, fz, ncells);
          }
        }
      }
      // x1x2 edges
      for (int m = -1; m <= 1; m += 2) {
        for (int n = -1; n <= 1; n += 2) {
          for (int fz = 0; fz < nfz; fz++) {
            int idx = NeighborIndex(n, m, 0, fz, 0);
            compute_send(send_mg_indcs_[idx][lev], n, m, 0, fz, 0, ncells);
            compute_recv(recv_mg_indcs_[idx][lev], n, m, 0, fz, 0, ncells);
          }
        }
      }
    }
    if (td) {
      // x3 faces
      for (int l = -1; l <= 1; l += 2) {
        for (int fy = 0; fy < nfy; fy++) {
          for (int fx = 0; fx < nfx; fx++) {
            int idx = NeighborIndex(0, 0, l, fx, fy);
            compute_send(send_mg_indcs_[idx][lev], 0, 0, l, fx, fy, ncells);
            compute_recv(recv_mg_indcs_[idx][lev], 0, 0, l, fx, fy, ncells);
          }
        }
      }
      // x3x1 edges
      for (int l = -1; l <= 1; l += 2) {
        for (int n = -1; n <= 1; n += 2) {
          for (int fy = 0; fy < nfy; fy++) {
            int idx = NeighborIndex(n, 0, l, fy, 0);
            compute_send(send_mg_indcs_[idx][lev], n, 0, l, fy, 0, ncells);
            compute_recv(recv_mg_indcs_[idx][lev], n, 0, l, fy, 0, ncells);
          }
        }
      }
      // x2x3 edges
      for (int l = -1; l <= 1; l += 2) {
        for (int m = -1; m <= 1; m += 2) {
          for (int fx = 0; fx < nfx; fx++) {
            int idx = NeighborIndex(0, m, l, fx, 0);
            compute_send(send_mg_indcs_[idx][lev], 0, m, l, fx, 0, ncells);
            compute_recv(recv_mg_indcs_[idx][lev], 0, m, l, fx, 0, ncells);
          }
        }
      }
      // corners
      for (int l = -1; l <= 1; l += 2) {
        for (int m = -1; m <= 1; m += 2) {
          for (int n = -1; n <= 1; n += 2) {
            int idx = NeighborIndex(n, m, l, 0, 0);
            compute_send(send_mg_indcs_[idx][lev], n, m, l, 0, 0, ncells);
            compute_recv(recv_mg_indcs_[idx][lev], n, m, l, 0, 0, ncells);
          }
        }
      }
    }
  }

  int nvar = pmy_mg->nvar_;
  int nmb = std::max(pmy_pack->nmb_thispack, pmy_pack->pmesh->nmb_maxperrank);
  int nnghbr = pmy_pack->pmb->nnghbr;
  int finest = nlevel - 1;

  if (pmy_pack->pmesh->multilevel) {
    for (int n = 0; n < nnghbr; ++n) {
      int smax = std::max(send_mg_indcs_[n][finest].isame_ndat,
                   std::max(send_mg_indcs_[n][finest].icoar_ndat,
                            send_mg_indcs_[n][finest].ifine_ndat));
      if (nvar * smax > sendbuf[n].vars.extent_int(1)) {
        Kokkos::realloc(sendbuf[n].vars, nmb, nvar * smax);
      }
      int rmax = std::max(recv_mg_indcs_[n][finest].isame_ndat,
                   std::max(recv_mg_indcs_[n][finest].icoar_ndat,
                            recv_mg_indcs_[n][finest].ifine_ndat));
      if (nvar * rmax > recvbuf[n].vars.extent_int(1)) {
        Kokkos::realloc(recvbuf[n].vars, nmb, nvar * rmax);
      }
    }

    int cnx_f = nx_max / 2;
    int cbn = cnx_f + 2*ngh;
    Kokkos::realloc(coarse_buf_, nmb, nvar, cbn, cbn, cbn);
  }
  ++mg_buf_gen_;   // boxes/buffers may have changed (GetPackList)
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridBoundaryValues::FillCoarseMG()
//! \brief Restrict MG data interior into coarse_buf_ interior so that the prolongation
//! kernel has gradient context from the block's own data.

void MultigridBoundaryValues::FillCoarseMG(const DvceArray5D<Real> &u) {
  if (pmy_mg == nullptr) return;
  int nvar = u.extent_int(1);
  int shift = pmy_mg->GetLevelShift();
  int ngh = pmy_mg->GetGhostCells();
  int nx = pmy_mg->GetSize();
  int ncells = nx >> shift;
  if (ncells < 2) return;
  int cnc = ncells / 2;
  int nmb = pmy_pack->nmb_thispack;
  auto cbuf = coarse_buf_;
  {
    // Item 67: clear the part of coarse_buf_ this level addresses, [0, cnc + 2 ngh) per
    // direction (this fill, the icoar pack/unpack boxes of ComputePerLevelIndices and the
    // ProlongateFCMG reads all lie inside it), so no read can see an earlier level's or
    // solve's data. Same as proj/tde 1eb11a70's whole-buffer fill except in extent.
    const Real fill = mg_fc_prefill_nan_ ? std::numeric_limits<Real>::quiet_NaN() : 0.0;
    const int cb = cnc + 2*ngh;
    Kokkos::deep_copy(DevExeSpace(),
                      Kokkos::subview(coarse_buf_, Kokkos::ALL, Kokkos::ALL,
                                      std::make_pair(0, cb), std::make_pair(0, cb),
                                      std::make_pair(0, cb)), fill);
  }

  // Volume-average restriction: fill coarse_buf_ interior
  Kokkos::parallel_for("FillCoarseMG",
    Kokkos::MDRangePolicy<Kokkos::Rank<4>, DevExeSpace>(
      {0, 0, 0, 0}, {nmb * nvar, cnc, cnc, cnc}),
    KOKKOS_LAMBDA(const int mv, const int ck, const int cj, const int ci) {
      int m = mv / nvar;
      int v = mv - m * nvar;
      int fi = ngh + 2*ci;
      int fj = ngh + 2*cj;
      int fk = ngh + 2*ck;
      cbuf(m, v, ngh + ck, ngh + cj, ngh + ci) = 0.125 * (
        u(m,v,fk,  fj,  fi) + u(m,v,fk,  fj,  fi+1) +
        u(m,v,fk,  fj+1,fi) + u(m,v,fk,  fj+1,fi+1) +
        u(m,v,fk+1,fj,  fi) + u(m,v,fk+1,fj,  fi+1) +
        u(m,v,fk+1,fj+1,fi) + u(m,v,fk+1,fj+1,fi+1));
    });

  if (!(pmy_pack->pmesh->multilevel)) return;

  // Face-aligned restriction: store 2x2 face averages in coarse_buf_ ghost cells.
  // These are consumed by PackAndSendMG for fine-to-coarse face sends.
  // face_id: 0=x-left, 1=x-right, 2=y-left, 3=y-right, 4=z-left, 5=z-right
  Kokkos::parallel_for("FillCoarseMG_faces",
    Kokkos::MDRangePolicy<Kokkos::Rank<4>, DevExeSpace>(
      {0, 0, 0, 0}, {nmb * nvar, 6, cnc, cnc}),
    KOKKOS_LAMBDA(const int mv, const int face, const int c1, const int c0) {
      int m = mv / nvar;
      int v = mv - m * nvar;
      if (face < 2) {
        int fj = ngh + 2*c0;
        int fk = ngh + 2*c1;
        int fi = (face == 0) ? ngh : ngh + ncells - 1;
        Real fav = 0.25 * (
          u(m,v,fk,  fj,  fi) + u(m,v,fk,  fj+1,fi) +
          u(m,v,fk+1,fj,  fi) + u(m,v,fk+1,fj+1,fi));
        // The send-to-coarser (icoar) face extent spans ngh coarse ghost cells, so
        // every one of them must carry the face average; otherwise the message
        // carries stale data for ngh > 1.
        for (int d = 0; d < ngh; ++d) {
          int cid = (face == 0) ? ngh - 1 - d : ngh + cnc + d;
          cbuf(m, v, ngh + c1, ngh + c0, cid) = fav;
        }
      } else if (face < 4) {
        int fi = ngh + 2*c0;
        int fk = ngh + 2*c1;
        int fj = (face == 2) ? ngh : ngh + ncells - 1;
        Real fav = 0.25 * (
          u(m,v,fk,  fj,fi  ) + u(m,v,fk,  fj,fi+1) +
          u(m,v,fk+1,fj,fi  ) + u(m,v,fk+1,fj,fi+1));
        for (int d = 0; d < ngh; ++d) {
          int cjd = (face == 2) ? ngh - 1 - d : ngh + cnc + d;
          cbuf(m, v, ngh + c1, cjd, ngh + c0) = fav;
        }
      } else {
        int fi = ngh + 2*c0;
        int fj = ngh + 2*c1;
        int fk = (face == 4) ? ngh : ngh + ncells - 1;
        Real fav = 0.25 * (
          u(m,v,fk,fj,  fi  ) + u(m,v,fk,fj,  fi+1) +
          u(m,v,fk,fj+1,fi  ) + u(m,v,fk,fj+1,fi+1));
        for (int d = 0; d < ngh; ++d) {
          int ckd = (face == 4) ? ngh - 1 - d : ngh + cnc + d;
          cbuf(m, v, ckd, ngh + c1, ngh + c0) = fav;
        }
      }
    });
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridBoundaryValues::BuildFCPairs()
//! \brief V-002 C1: list the (m, neighbour) pairs ProlongateFCMG writes, in the same
//! (ox3, ox2, ox1, f2, f1) order as the original per-block loop. Built from device copies of
//! nghbr/mb_lev (exactly what the kernel used to read), so it cannot see stale host data.

void MultigridBoundaryValues::BuildFCPairs() {
  int nmb = pmy_pack->nmb_thispack;
  int seq = pmy_pack->pmesh->GetAMRLoadBalanceUpdateSeq();
  if (fc_pairs_seq_ == seq && fc_pairs_nmb_ == nmb && fc_pairs_.extent_int(0) > 0) return;
  auto nghbr_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(),
                                                     pmy_pack->pmb->nghbr.d_view);
  auto mblev_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(),
                                                     pmy_pack->pmb->mb_lev.d_view);
  std::vector<int> rows;
  for (int m = 0; m < nmb; ++m) {
    int m_lev = mblev_h(m);
    for (int ox3 = -1; ox3 <= 1; ++ox3) {
      for (int ox2 = -1; ox2 <= 1; ++ox2) {
        for (int ox1 = -1; ox1 <= 1; ++ox1) {
          if (ox1 == 0 && ox2 == 0 && ox3 == 0) continue;
          int nface = (ox1!=0?1:0) + (ox2!=0?1:0) + (ox3!=0?1:0);
          int f2_max = (nface == 1) ? 1 : 0;
          int f1_max = (nface <= 2) ? 1 : 0;
          for (int f2 = 0; f2 <= f2_max; ++f2) {
            for (int f1 = 0; f1 <= f1_max; ++f1) {
              int n = NeighborIndex(ox1, ox2, ox3, f1, f2);
              if (n < 0 || n >= 56) continue;
              if (nghbr_h(m, n).gid < 0) continue;
              int nlev = nghbr_h(m, n).lev;
              int kind;
              if (nlev > m_lev && nface == 1) {
                kind = 0;
              } else if (nlev >= m_lev) {
                continue;
              } else {
                kind = (nface == 1) ? 1 : 2;
              }
              int r[7] = {m, ox1, ox2, ox3, f1, f2, kind};
              rows.insert(rows.end(), r, r + 7);
            }
          }
        }
      }
    }
  }
  fc_npairs_ = static_cast<int>(rows.size()/7);
  Kokkos::realloc(fc_pairs_, std::max(fc_npairs_, 1), 7);
  auto fc_h = Kokkos::create_mirror_view(fc_pairs_);
  for (int p = 0; p < fc_npairs_; ++p)
    for (int c = 0; c < 7; ++c) fc_h(p, c) = rows[7*p + c];
  Kokkos::deep_copy(fc_pairs_, fc_h);
  fc_pairs_seq_ = seq;
  fc_pairs_nmb_ = nmb;
  // Item 67: for every coarser-face pair (kind 1), whether
  // the edge neighbour beyond each end of the two transverse directions is coarser
  // (its icoar receive then fills the coarse_buf_ cell just past the face neighbour's
  // range; proj/tde aa824220's coarser_at, evaluated once per mesh instead of per
  // thread). Bits: 1/2 = first transverse direction low/high end, 4/8 = second; the
  // transverse pairs are (x2, x3) for an x1 face, (x1, x3) for x2, (x1, x2) for x3.
  // With <cfc> mg_fc_prefill_nan or mg_packlist_check on, also prints per rank how many
  // ends each part of the rule touches (MGFCENDS; relative to the former rule).
  {
    auto coarser_at = [&](const int m, const int a, const int b, const int c) {
      for (int f = 0; f <= 1; ++f) {
        const int nn = NeighborIndex(a, b, c, f, 0);
        if (nn >= 0 && nn < 56 && nghbr_h(m, nn).gid >= 0 &&
            nghbr_h(m, nn).lev < mblev_h(m)) {
          return true;
        }
      }
      return false;
    };
    auto cx_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmy_mg->fc_childx_);
    auto cy_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmy_mg->fc_childy_);
    auto cz_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmy_mg->fc_childz_);
    Kokkos::realloc(fc_ends_, std::max(fc_npairs_, 1));
    auto ends_h = Kokkos::create_mirror_view(fc_ends_);
    int nk1 = 0, n_stale = 0, n_ext = 0, n_slope = 0;
    for (int p = 0; p < fc_npairs_; ++p) {
      ends_h(p) = 0;
      if (fc_h(p, 6) != 1) continue;
      const int m = fc_h(p, 0);
      const int ox1 = fc_h(p, 1), ox2 = fc_h(p, 2), ox3 = fc_h(p, 3);
      int bits[4], child[2];
      if (ox1 != 0) {
        bits[0] = coarser_at(m, ox1, -1, 0);
        bits[1] = coarser_at(m, ox1, 1, 0);
        bits[2] = coarser_at(m, ox1, 0, -1);
        bits[3] = coarser_at(m, ox1, 0, 1);
        child[0] = cy_h(m);
        child[1] = cz_h(m);
      } else if (ox2 != 0) {
        bits[0] = coarser_at(m, -1, ox2, 0);
        bits[1] = coarser_at(m, 1, ox2, 0);
        bits[2] = coarser_at(m, 0, ox2, -1);
        bits[3] = coarser_at(m, 0, ox2, 1);
        child[0] = cx_h(m);
        child[1] = cz_h(m);
      } else {
        bits[0] = coarser_at(m, -1, 0, ox3);
        bits[1] = coarser_at(m, 1, 0, ox3);
        bits[2] = coarser_at(m, 0, -1, ox3);
        bits[3] = coarser_at(m, 0, 1, ox3);
        child[0] = cx_h(m);
        child[1] = cy_h(m);
      }
      ends_h(p) = bits[0] | (bits[1] << 1) | (bits[2] << 2) | (bits[3] << 3);
      ++nk1;
      for (int d = 0; d < 2; ++d) {
        const bool lo_c = bits[2*d], hi_c = bits[2*d + 1];
        if (child[d] && !hi_c) ++n_stale;    // high end: stale read removed, one-sided
        if (!child[d] && lo_c) ++n_ext;      // low end: one more (filled) cell read
        if (!child[d] && !lo_c) ++n_slope;   // low end: one-sided slope no longer halved
      }
    }
    Kokkos::deep_copy(fc_ends_, ends_h);
    if (mg_fc_prefill_nan_ > 0 || mg_packlist_check_ > 0) {
      std::cout << "MGFCENDS rank " << global_variable::my_rank << " pairs " << fc_npairs_
                << " kind1 " << nk1 << " ends " << 4*nk1 << " stale_removed " << n_stale
                << " extended " << n_ext << " slope_only " << n_slope << std::endl;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MultigridBoundaryValues::ProlongateFCMG()
//! \brief Prolongate from coarse_buf_ to fine ghost cells using the same flux-conserving
//! formulas as FillFineCoarseMGGhosts.  For face neighbors at coarser level, uses
//! gradient-based prolongation.  For edge/corner neighbors at coarser level, uses simple
//! injection.  For finer neighbors, restriction was already done inline in unpack.
//!
//! V-002 C1: one team per (pair, variable) from the cached BuildFCPairs() list, team
//! threads over the pair's ghost cells (formerly one work-item per MeshBlock with all loops
//! serial). Every pair writes a disjoint ghost region and reads only coarse_buf_, interior
//! cells and its own region; the one in-region dependence (finer-face neighbour, ngh > 1:
//! the update reads the next cell along the face normal) is kept serial, in the original
//! order, inside one thread. Arithmetic expressions are unchanged.

TaskStatus MultigridBoundaryValues::ProlongateFCMG(DvceArray5D<Real> &u) {
  if (pmy_mg == nullptr) return TaskStatus::complete;

  int nvar = u.extent_int(1);
  int shift = pmy_mg->GetLevelShift();
  int ngh = pmy_mg->GetGhostCells();
  int nx = pmy_mg->GetSize();
  int ncells = nx >> shift;
  if (ncells < 2) return TaskStatus::complete;

  BuildFCPairs();
  if (fc_npairs_ == 0) return TaskStatus::complete;
  LaunchProlongateFCMG(u, nvar, ngh, ncells);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridBoundaryValues::LaunchProlongateFCMG()
//! \brief The ProlongateFCMG kernel, with proj/tde aa824220's read range (item 67).

void MultigridBoundaryValues::LaunchProlongateFCMG(DvceArray5D<Real> &u, const int nvar,
                                                   const int ngh, const int ncells) {
  auto pairs = fc_pairs_;
  auto fce = fc_ends_;
  auto fc_cx = pmy_mg->fc_childx_;
  auto fc_cy = pmy_mg->fc_childy_;
  auto fc_cz = pmy_mg->fc_childz_;
  auto cbuf = coarse_buf_;

  const int nvar_l = nvar;
  const int ngh_l = ngh;
  const int ncells_l = ncells;
  const int half = ncells / 2;
  constexpr Real ot = 1.0/3.0;

  using TeamPol = Kokkos::TeamPolicy<DevExeSpace>;
  using Member = TeamPol::member_type;
  auto kern = KOKKOS_LAMBDA(const Member &tm) {
      const int pv = tm.league_rank();
      const int p = pv / nvar_l;
      const int v = pv - p*nvar_l;
      const int m = pairs(p, 0);
      const int ox1 = pairs(p, 1), ox2 = pairs(p, 2), ox3 = pairs(p, 3);
      const int f1 = pairs(p, 4), f2 = pairs(p, 5), kind = pairs(p, 6);

      if (kind == 0) {
        // From finer face neighbor: apply FC correction. Ghost cells already contain the
        // restricted face avg from unpack. Threads over the two transverse directions,
        // the face-normal direction serial in the original (ascending) order.
        int oi = (ox1 < 0) ? 1 : (ox1 > 0) ? -1 : 0;
        int oj = (ox2 < 0) ? 1 : (ox2 > 0) ? -1 : 0;
        int ok = (ox3 < 0) ? 1 : (ox3 > 0) ? -1 : 0;
        int sub_x = 0, sub_y = 0, sub_z = 0;
        if (ox1 != 0) {
          sub_y = f1; sub_z = f2;
        } else if (ox2 != 0) {
          sub_x = f1; sub_z = f2;
        } else {
          sub_x = f1; sub_y = f2;
        }
        int gis, gie, gjs, gje, gks, gke;
        if (ox1 < 0) {
          gis = 0; gie = ngh_l - 1;
        } else if (ox1 > 0) {
          gis = ngh_l + ncells_l; gie = ngh_l + ncells_l + ngh_l - 1;
        } else {
          gis = ngh_l + sub_x*half; gie = ngh_l + sub_x*half + half - 1;
        }
        if (ox2 < 0) {
          gjs = 0; gje = ngh_l - 1;
        } else if (ox2 > 0) {
          gjs = ngh_l + ncells_l; gje = ngh_l + ncells_l + ngh_l - 1;
        } else {
          gjs = ngh_l + sub_y*half; gje = ngh_l + sub_y*half + half - 1;
        }
        if (ox3 < 0) {
          gks = 0; gke = ngh_l - 1;
        } else if (ox3 > 0) {
          gks = ngh_l + ncells_l; gke = ngh_l + ncells_l + ngh_l - 1;
        } else {
          gks = ngh_l + sub_z*half; gke = ngh_l + sub_z*half + half - 1;
        }
        Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, half*half), [&](const int t) {
          const int ta = t % half, tb = t / half;
          if (ox1 != 0) {
            const int gj = gjs + ta, gk = gks + tb;
            for (int gi = gis; gi <= gie; ++gi) {
              Real avg = u(m,v,gk,gj,gi);
              u(m,v,gk,gj,gi) = ot*(4.0*avg - u(m,v,gk+ok,gj+oj,gi+oi));
            }
          } else if (ox2 != 0) {
            const int gi = gis + ta, gk = gks + tb;
            for (int gj = gjs; gj <= gje; ++gj) {
              Real avg = u(m,v,gk,gj,gi);
              u(m,v,gk,gj,gi) = ot*(4.0*avg - u(m,v,gk+ok,gj+oj,gi+oi));
            }
          } else {
            const int gi = gis + ta, gj = gjs + tb;
            for (int gk = gks; gk <= gke; ++gk) {
              Real avg = u(m,v,gk,gj,gi);
              u(m,v,gk,gj,gi) = ot*(4.0*avg - u(m,v,gk+ok,gj+oj,gi+oi));
            }
          }
        });
      } else if (kind == 1) {
        // Face neighbor from coarser: flux-conserving prolongation from coarse_buf_ into
        // fine ghost cells of u; one thread per coarse face cell (writes 4 fine ghosts).
        // Transverse read range (item 67, proj/tde aa824220): the cells the coarse face
        // neighbour sent for this child (low child: [ngh, ngh+half], high child:
        // [ngh-1, ngh+half-1]); each end extends one cell further only where the edge
        // neighbour beyond is coarser (fc_ends_: its icoar receive fills that cell). A
        // slope clamped to one side takes its full value (0.25, applied as 2 * 0.125 * d,
        // exact).
        const int child_x = fc_cx(m), child_y = fc_cy(m), child_z = fc_cz(m);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, half*half), [&](const int t) {
          const int ta = t % half, tb = t / half;
          if (ox1 != 0) {
            int fig = (ox1 < 0) ? ngh_l - 1 : ngh_l + ncells_l;
            int fi  = (ox1 < 0) ? ngh_l : ngh_l + ncells_l - 1;
            int si  = (ox1 < 0) ? ngh_l - 1 : ngh_l + half;
            int sj0 = ngh_l, sk0 = ngh_l;
            const int ends = fce(p);
            const int y_lo = (child_y || (ends & 1)) ? ngh_l - 1 : ngh_l;
            const int y_hi = (!child_y || (ends & 2)) ? ngh_l + half : ngh_l + half - 1;
            const int z_lo = (child_z || (ends & 4)) ? ngh_l - 1 : ngh_l;
            const int z_hi = (!child_z || (ends & 8)) ? ngh_l + half : ngh_l + half - 1;
            int sj = sj0 + ta, sk = sk0 + tb;
            int fj = ngh_l + 2*(sj - sj0);
            int fk = ngh_l + 2*(sk - sk0);
            Real cc = cbuf(m,v,sk,sj,si);
            int sjm = (sj > y_lo) ? sj-1 : sj;
            int sjp = (sj < y_hi) ? sj+1 : sj;
            int skm = (sk > z_lo) ? sk-1 : sk;
            int skp = (sk < z_hi) ? sk+1 : sk;
            Real gy = 0.125*(cbuf(m,v,sk,sjp,si)-cbuf(m,v,sk,sjm,si));
            Real gz = 0.125*(cbuf(m,v,skp,sj,si)-cbuf(m,v,skm,sj,si));
            if (sjp - sjm != 2) gy *= 2.0;   // one-sided: full slope
            if (skp - skm != 2) gz *= 2.0;
            u(m,v,fk  ,fj  ,fig)=ot*(2.0*(cc-gy-gz)+u(m,v,fk  ,fj  ,fi));
            u(m,v,fk  ,fj+1,fig)=ot*(2.0*(cc+gy-gz)+u(m,v,fk  ,fj+1,fi));
            u(m,v,fk+1,fj  ,fig)=ot*(2.0*(cc-gy+gz)+u(m,v,fk+1,fj  ,fi));
            u(m,v,fk+1,fj+1,fig)=ot*(2.0*(cc+gy+gz)+u(m,v,fk+1,fj+1,fi));
          } else if (ox2 != 0) {
            int fjg = (ox2 < 0) ? ngh_l - 1 : ngh_l + ncells_l;
            int fj  = (ox2 < 0) ? ngh_l : ngh_l + ncells_l - 1;
            int sj  = (ox2 < 0) ? ngh_l - 1 : ngh_l + half;
            int si0 = ngh_l, sk0 = ngh_l;
            const int ends = fce(p);
            const int x_lo = (child_x || (ends & 1)) ? ngh_l - 1 : ngh_l;
            const int x_hi = (!child_x || (ends & 2)) ? ngh_l + half : ngh_l + half - 1;
            const int z_lo = (child_z || (ends & 4)) ? ngh_l - 1 : ngh_l;
            const int z_hi = (!child_z || (ends & 8)) ? ngh_l + half : ngh_l + half - 1;
            int si = si0 + ta, sk = sk0 + tb;
            int fi = ngh_l + 2*(si - si0);
            int fk = ngh_l + 2*(sk - sk0);
            Real cc = cbuf(m,v,sk,sj,si);
            int sim = (si > x_lo) ? si-1 : si;
            int sip = (si < x_hi) ? si+1 : si;
            int skm = (sk > z_lo) ? sk-1 : sk;
            int skp = (sk < z_hi) ? sk+1 : sk;
            Real gx = 0.125*(cbuf(m,v,sk,sj,sip)-cbuf(m,v,sk,sj,sim));
            Real gz = 0.125*(cbuf(m,v,skp,sj,si)-cbuf(m,v,skm,sj,si));
            if (sip - sim != 2) gx *= 2.0;   // one-sided: full slope
            if (skp - skm != 2) gz *= 2.0;
            u(m,v,fk  ,fjg,fi  )=ot*(2.0*(cc-gx-gz)+u(m,v,fk  ,fj,fi  ));
            u(m,v,fk  ,fjg,fi+1)=ot*(2.0*(cc+gx-gz)+u(m,v,fk  ,fj,fi+1));
            u(m,v,fk+1,fjg,fi  )=ot*(2.0*(cc-gx+gz)+u(m,v,fk+1,fj,fi  ));
            u(m,v,fk+1,fjg,fi+1)=ot*(2.0*(cc+gx+gz)+u(m,v,fk+1,fj,fi+1));
          } else {
            int fkg = (ox3 < 0) ? ngh_l - 1 : ngh_l + ncells_l;
            int fk  = (ox3 < 0) ? ngh_l : ngh_l + ncells_l - 1;
            int sk  = (ox3 < 0) ? ngh_l - 1 : ngh_l + half;
            int si0 = ngh_l, sj0 = ngh_l;
            const int ends = fce(p);
            const int x_lo = (child_x || (ends & 1)) ? ngh_l - 1 : ngh_l;
            const int x_hi = (!child_x || (ends & 2)) ? ngh_l + half : ngh_l + half - 1;
            const int y_lo = (child_y || (ends & 4)) ? ngh_l - 1 : ngh_l;
            const int y_hi = (!child_y || (ends & 8)) ? ngh_l + half : ngh_l + half - 1;
            int si = si0 + ta, sj = sj0 + tb;
            int fi = ngh_l + 2*(si - si0);
            int fj = ngh_l + 2*(sj - sj0);
            Real cc = cbuf(m,v,sk,sj,si);
            int sim = (si > x_lo) ? si-1 : si;
            int sip = (si < x_hi) ? si+1 : si;
            int sjm = (sj > y_lo) ? sj-1 : sj;
            int sjp = (sj < y_hi) ? sj+1 : sj;
            Real gx = 0.125*(cbuf(m,v,sk,sj,sip)-cbuf(m,v,sk,sj,sim));
            Real gy = 0.125*(cbuf(m,v,sk,sjp,si)-cbuf(m,v,sk,sjm,si));
            if (sip - sim != 2) gx *= 2.0;   // one-sided: full slope
            if (sjp - sjm != 2) gy *= 2.0;
            u(m,v,fkg,fj  ,fi  )=ot*(2.0*(cc-gx-gy)+u(m,v,fk,fj  ,fi  ));
            u(m,v,fkg,fj  ,fi+1)=ot*(2.0*(cc+gx-gy)+u(m,v,fk,fj  ,fi+1));
            u(m,v,fkg,fj+1,fi  )=ot*(2.0*(cc-gx+gy)+u(m,v,fk,fj+1,fi  ));
            u(m,v,fkg,fj+1,fi+1)=ot*(2.0*(cc+gx+gy)+u(m,v,fk,fj+1,fi+1));
          }
        });
      } else {
        // Edge/corner neighbor from coarser: injection, one thread per ghost cell.
        int gis, gie, gjs, gje, gks, gke;
        if (ox1 < 0) {
          gis = 0; gie = ngh_l - 1;
        } else if (ox1 > 0) {
          gis = ngh_l + ncells_l; gie = ngh_l + ncells_l + ngh_l - 1;
        } else {
          gis = ngh_l; gie = ngh_l + ncells_l - 1;
        }
        if (ox2 < 0) {
          gjs = 0; gje = ngh_l - 1;
        } else if (ox2 > 0) {
          gjs = ngh_l + ncells_l; gje = ngh_l + ncells_l + ngh_l - 1;
        } else {
          gjs = ngh_l; gje = ngh_l + ncells_l - 1;
        }
        if (ox3 < 0) {
          gks = 0; gke = ngh_l - 1;
        } else if (ox3 > 0) {
          gks = ngh_l + ncells_l; gke = ngh_l + ncells_l + ngh_l - 1;
        } else {
          gks = ngh_l; gke = ngh_l + ncells_l - 1;
        }
        const int ni = gie - gis + 1, nj = gje - gjs + 1, nk = gke - gks + 1;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, ni*nj*nk), [&](const int t) {
          const int gi = gis + t % ni;
          const int gj = gjs + (t / ni) % nj;
          const int gk = gks + t / (ni*nj);
          int ci, cj, ck;
          if (ox1 < 0)      ci = ngh_l - 1;
          else if (ox1 > 0) ci = ngh_l + half;
          else              ci = ngh_l + (gi - ngh_l)/2;
          if (ox2 < 0)      cj = ngh_l - 1;
          else if (ox2 > 0) cj = ngh_l + half;
          else              cj = ngh_l + (gj - ngh_l)/2;
          if (ox3 < 0)      ck = ngh_l - 1;
          else if (ox3 > 0) ck = ngh_l + half;
          else              ck = ngh_l + (gk - ngh_l)/2;
          u(m, v, gk, gj, gi) = cbuf(m, v, ck, cj, ci);
        });
      }
  };
  if (mg_packlist_check_ > 0 && !mg_fc_size_printed_ && global_variable::my_rank == 0) {
    std::cout << "MGFCFIX functor bytes: ProlongateFCMG " << sizeof(kern)
              << " (Kokkos-SYCL launches indirectly from 1800)" << std::endl;
  }
  mg_fc_size_printed_ = true;
  Kokkos::parallel_for("ProlongateFCMG", TeamPol(fc_npairs_*nvar_l, Kokkos::AUTO), kern);
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MultigridBoundaryValues::FillFineCoarseMGGhosts()
//! \brief Fill ghost cells at fine-coarse boundaries.
//! Faces use flux-conserving prolongation/restriction matching Athena++ formulas.
//! Edges and corners use simple injection/restriction. Same-rank only.

TaskStatus MultigridBoundaryValues::FillFineCoarseMGGhosts(DvceArray5D<Real> &u) {
  if (pmy_mg == nullptr) return TaskStatus::complete;

  int nvar = u.extent_int(1);
  int shift = pmy_mg->GetLevelShift();
  int ngh = pmy_mg->GetGhostCells();
  int nx = pmy_mg->GetSize();
  int ncells = nx >> shift;

  if (ncells < 1) return TaskStatus::complete;

  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  int my_rank = global_variable::my_rank;
  auto nghbr_d = pmy_pack->pmb->nghbr.d_view;
  auto mblev_d = pmy_pack->pmb->mb_lev.d_view;
  auto mbgid_d = pmy_pack->pmb->mb_gid.d_view;

#ifndef NDEBUG
  {
    static bool warned_cross_rank_fc = false;
    if (!warned_cross_rank_fc) {
      auto &nghbr_h = pmy_pack->pmb->nghbr;
      auto &mblev_h = pmy_pack->pmb->mb_lev;
      for (int m = 0; m < nmb && !warned_cross_rank_fc; ++m) {
        for (int n = 0; n < nnghbr && !warned_cross_rank_fc; ++n) {
          if (nghbr_h.h_view(m,n).gid >= 0
              && nghbr_h.h_view(m,n).lev != mblev_h.h_view(m)
              && nghbr_h.h_view(m,n).rank != my_rank) {
            std::cout << "### MG WARNING: cross-rank fine-coarse neighbor detected "
                      << "(m=" << m << " n=" << n << " rank=" << nghbr_h.h_view(m,n).rank
                      << "). FillFineCoarseMGGhosts skips cross-rank neighbors."
                      << std::endl;
            warned_cross_rank_fc = true;
          }
        }
      }
    }
  }
#endif
  auto fc_cx = pmy_mg->fc_childx_;
  auto fc_cy = pmy_mg->fc_childy_;
  auto fc_cz = pmy_mg->fc_childz_;

  int nmb_l = nmb;
  int nnghbr_l = nnghbr;
  int my_rank_l = my_rank;
  int nvar_l = nvar;
  int ngh_l = ngh;
  int ncells_l = ncells;
  constexpr Real ot = 1.0/3.0;

  Kokkos::parallel_for("FillFCMGGhosts",
    Kokkos::RangePolicy<DevExeSpace>(0, nmb),
    KOKKOS_LAMBDA(const int m) {
      int m_lev = mblev_d(m);
      int child_x = fc_cx(m);
      int child_y = fc_cy(m);
      int child_z = fc_cz(m);
      int half = ncells_l / 2;

      for (int ox3 = -1; ox3 <= 1; ++ox3) {
        for (int ox2 = -1; ox2 <= 1; ++ox2) {
          for (int ox1 = -1; ox1 <= 1; ++ox1) {
            if (ox1 == 0 && ox2 == 0 && ox3 == 0) continue;
            int nface = (ox1!=0?1:0) + (ox2!=0?1:0) + (ox3!=0?1:0);
            int f2_max = (nface == 1) ? 1 : 0;
            int f1_max = (nface <= 2) ? 1 : 0;

            for (int f2 = 0; f2 <= f2_max; ++f2) {
              for (int f1 = 0; f1 <= f1_max; ++f1) {
                int n = NeighborIndex(ox1, ox2, ox3, f1, f2);
                if (n < 0 || n >= nnghbr_l) continue;
                if (nghbr_d(m, n).gid < 0) continue;

                int nlev = nghbr_d(m, n).lev;
                if (nlev == m_lev) continue;
                if (nghbr_d(m, n).rank != my_rank_l) continue;

                int dm = nghbr_d(m, n).gid - mbgid_d(0);
                if (dm < 0 || dm >= nmb_l) continue;

                if (nlev < m_lev && nface == 1) {
                  // Coarser neighbor, face: flux-conserving prolongation
                  if (ox1 != 0) {
                    int fig = (ox1 < 0) ? ngh_l - 1 : ngh_l + ncells_l;
                    int fi  = (ox1 < 0) ? ngh_l : ngh_l + ncells_l - 1;
                    int si  = (ox1 < 0) ? ngh_l + ncells_l - 1 : ngh_l;
                    int sj0 = ngh_l + child_y * half;
                    int sk0 = ngh_l + child_z * half;
                    for (int v = 0; v < nvar_l; ++v) {
                      for (int sk = sk0; sk < sk0 + half; ++sk) {
                        for (int sj = sj0; sj < sj0 + half; ++sj) {
                          int fj = ngh_l + 2*(sj - sj0);
                          int fk = ngh_l + 2*(sk - sk0);
                          Real cc = u(dm,v,sk,sj,si);
                          int sjm = (sj > ngh_l) ? sj-1 : sj;
                          int sjp = (sj < ngh_l+ncells_l-1) ? sj+1 : sj;
                          int skm = (sk > ngh_l) ? sk-1 : sk;
                          int skp = (sk < ngh_l+ncells_l-1) ? sk+1 : sk;
                          Real gy = 0.125*(u(dm,v,sk,sjp,si)-u(dm,v,sk,sjm,si));
                          Real gz = 0.125*(u(dm,v,skp,sj,si)-u(dm,v,skm,sj,si));
                          u(m,v,fk  ,fj  ,fig)=ot*(2.0*(cc-gy-gz)+u(m,v,fk  ,fj  ,fi));
                          u(m,v,fk  ,fj+1,fig)=ot*(2.0*(cc+gy-gz)+u(m,v,fk  ,fj+1,fi));
                          u(m,v,fk+1,fj  ,fig)=ot*(2.0*(cc-gy+gz)+u(m,v,fk+1,fj  ,fi));
                          u(m,v,fk+1,fj+1,fig)=ot*(2.0*(cc+gy+gz)+u(m,v,fk+1,fj+1,fi));
                        }
                      }
                    }
                  } else if (ox2 != 0) {
                    int fjg = (ox2 < 0) ? ngh_l - 1 : ngh_l + ncells_l;
                    int fj  = (ox2 < 0) ? ngh_l : ngh_l + ncells_l - 1;
                    int sj  = (ox2 < 0) ? ngh_l + ncells_l - 1 : ngh_l;
                    int si0 = ngh_l + child_x * half;
                    int sk0 = ngh_l + child_z * half;
                    for (int v = 0; v < nvar_l; ++v) {
                      for (int sk = sk0; sk < sk0 + half; ++sk) {
                        for (int si = si0; si < si0 + half; ++si) {
                          int fi = ngh_l + 2*(si - si0);
                          int fk = ngh_l + 2*(sk - sk0);
                          Real cc = u(dm,v,sk,sj,si);
                          int sim = (si > ngh_l) ? si-1 : si;
                          int sip = (si < ngh_l+ncells_l-1) ? si+1 : si;
                          int skm = (sk > ngh_l) ? sk-1 : sk;
                          int skp = (sk < ngh_l+ncells_l-1) ? sk+1 : sk;
                          Real gx = 0.125*(u(dm,v,sk,sj,sip)-u(dm,v,sk,sj,sim));
                          Real gz = 0.125*(u(dm,v,skp,sj,si)-u(dm,v,skm,sj,si));
                          u(m,v,fk  ,fjg,fi  )=ot*(2.0*(cc-gx-gz)+u(m,v,fk  ,fj,fi  ));
                          u(m,v,fk  ,fjg,fi+1)=ot*(2.0*(cc+gx-gz)+u(m,v,fk  ,fj,fi+1));
                          u(m,v,fk+1,fjg,fi  )=ot*(2.0*(cc-gx+gz)+u(m,v,fk+1,fj,fi  ));
                          u(m,v,fk+1,fjg,fi+1)=ot*(2.0*(cc+gx+gz)+u(m,v,fk+1,fj,fi+1));
                        }
                      }
                    }
                  } else {
                    int fkg = (ox3 < 0) ? ngh_l - 1 : ngh_l + ncells_l;
                    int fk  = (ox3 < 0) ? ngh_l : ngh_l + ncells_l - 1;
                    int sk  = (ox3 < 0) ? ngh_l + ncells_l - 1 : ngh_l;
                    int si0 = ngh_l + child_x * half;
                    int sj0 = ngh_l + child_y * half;
                    for (int v = 0; v < nvar_l; ++v) {
                      for (int sj = sj0; sj < sj0 + half; ++sj) {
                        for (int si = si0; si < si0 + half; ++si) {
                          int fi = ngh_l + 2*(si - si0);
                          int fj = ngh_l + 2*(sj - sj0);
                          Real cc = u(dm,v,sk,sj,si);
                          int sim = (si > ngh_l) ? si-1 : si;
                          int sip = (si < ngh_l+ncells_l-1) ? si+1 : si;
                          int sjm = (sj > ngh_l) ? sj-1 : sj;
                          int sjp = (sj < ngh_l+ncells_l-1) ? sj+1 : sj;
                          Real gx = 0.125*(u(dm,v,sk,sj,sip)-u(dm,v,sk,sj,sim));
                          Real gy = 0.125*(u(dm,v,sk,sjp,si)-u(dm,v,sk,sjm,si));
                          u(m,v,fkg,fj  ,fi  )=ot*(2.0*(cc-gx-gy)+u(m,v,fk,fj  ,fi  ));
                          u(m,v,fkg,fj  ,fi+1)=ot*(2.0*(cc+gx-gy)+u(m,v,fk,fj  ,fi+1));
                          u(m,v,fkg,fj+1,fi  )=ot*(2.0*(cc-gx+gy)+u(m,v,fk,fj+1,fi  ));
                          u(m,v,fkg,fj+1,fi+1)=ot*(2.0*(cc+gx+gy)+u(m,v,fk,fj+1,fi+1));
                        }
                      }
                    }
                  }

                } else if (nlev > m_lev && nface == 1) {
                  // Finer neighbor, face: flux-conserving restriction
                  int sub_x = 0, sub_y = 0, sub_z = 0;
                  if (ox1 != 0) {
                    sub_y = f1; sub_z = f2;
                  }
                  if (ox2 != 0) {
                    sub_x = f1; sub_z = f2;
                  }
                  if (ox3 != 0) {
                    sub_x = f1; sub_y = f2;
                  }

                  int gis, gie, gjs, gje, gks, gke;
                  if (ox1 < 0) {
                    gis = 0; gie = ngh_l - 1;
                  } else if (ox1 > 0) {
                    gis = ngh_l + ncells_l;
                    gie = ngh_l + ncells_l + ngh_l - 1;
                  } else {
                    gis = ngh_l + sub_x*half;
                    gie = ngh_l + sub_x*half + half - 1;
                  }
                  if (ox2 < 0) {
                    gjs = 0; gje = ngh_l - 1;
                  } else if (ox2 > 0) {
                    gjs = ngh_l + ncells_l;
                    gje = ngh_l + ncells_l + ngh_l - 1;
                  } else {
                    gjs = ngh_l + sub_y*half;
                    gje = ngh_l + sub_y*half + half - 1;
                  }
                  if (ox3 < 0) {
                    gks = 0; gke = ngh_l - 1;
                  } else if (ox3 > 0) {
                    gks = ngh_l + ncells_l;
                    gke = ngh_l + ncells_l + ngh_l - 1;
                  } else {
                    gks = ngh_l + sub_z*half;
                    gke = ngh_l + sub_z*half + half - 1;
                  }

                  int oi = (ox1 < 0) ? 1 : (ox1 > 0) ? -1 : 0;
                  int oj = (ox2 < 0) ? 1 : (ox2 > 0) ? -1 : 0;
                  int ok = (ox3 < 0) ? 1 : (ox3 > 0) ? -1 : 0;

                  if (ox1 != 0) {
                    int fi = (ox1 > 0) ? ngh_l : ngh_l + ncells_l - 1;
                    for (int v = 0; v < nvar_l; ++v) {
                      for (int gk = gks; gk <= gke; ++gk) {
                        for (int gj = gjs; gj <= gje; ++gj) {
                          int fj0 = ngh_l + 2*(gj - (ngh_l + sub_y*half));
                          int fk0 = ngh_l + 2*(gk - (ngh_l + sub_z*half));
                          Real favg = 0.25*(u(dm,v,fk0,fj0,fi)+u(dm,v,fk0,fj0+1,fi)
                                           +u(dm,v,fk0+1,fj0,fi)+u(dm,v,fk0+1,fj0+1,fi));
                          u(m,v,gk,gj,gis) = ot*(4.0*favg - u(m,v,gk+ok,gj+oj,gis+oi));
                        }
                      }
                    }
                  } else if (ox2 != 0) {
                    int fj = (ox2 > 0) ? ngh_l : ngh_l + ncells_l - 1;
                    for (int v = 0; v < nvar_l; ++v) {
                      for (int gk = gks; gk <= gke; ++gk) {
                        for (int gi = gis; gi <= gie; ++gi) {
                          int fi0 = ngh_l + 2*(gi - (ngh_l + sub_x*half));
                          int fk0 = ngh_l + 2*(gk - (ngh_l + sub_z*half));
                          Real favg = 0.25*(u(dm,v,fk0,fj,fi0)+u(dm,v,fk0,fj,fi0+1)
                                           +u(dm,v,fk0+1,fj,fi0)+u(dm,v,fk0+1,fj,fi0+1));
                          u(m,v,gk,gjs,gi) = ot*(4.0*favg - u(m,v,gk+ok,gjs+oj,gi+oi));
                        }
                      }
                    }
                  } else {
                    int fk = (ox3 > 0) ? ngh_l : ngh_l + ncells_l - 1;
                    for (int v = 0; v < nvar_l; ++v) {
                      for (int gj = gjs; gj <= gje; ++gj) {
                        for (int gi = gis; gi <= gie; ++gi) {
                          int fi0 = ngh_l + 2*(gi - (ngh_l + sub_x*half));
                          int fj0 = ngh_l + 2*(gj - (ngh_l + sub_y*half));
                          Real favg = 0.25*(u(dm,v,fk,fj0,fi0)+u(dm,v,fk,fj0,fi0+1)
                                           +u(dm,v,fk,fj0+1,fi0)+u(dm,v,fk,fj0+1,fi0+1));
                          u(m,v,gks,gj,gi) = ot*(4.0*favg - u(m,v,gks+ok,gj+oj,gi+oi));
                        }
                      }
                    }
                  }

                } else if (nlev < m_lev) {
                  int gis, gie, gjs, gje, gks, gke;
                  if (ox1 < 0) {
                    gis = 0; gie = ngh_l - 1;
                  } else if (ox1 > 0) {
                    gis = ngh_l + ncells_l;
                    gie = ngh_l + ncells_l + ngh_l - 1;
                  } else {
                    gis = ngh_l;
                    gie = ngh_l + ncells_l - 1;
                  }
                  if (ox2 < 0) {
                    gjs = 0; gje = ngh_l - 1;
                  } else if (ox2 > 0) {
                    gjs = ngh_l + ncells_l;
                    gje = ngh_l + ncells_l + ngh_l - 1;
                  } else {
                    gjs = ngh_l;
                    gje = ngh_l + ncells_l - 1;
                  }
                  if (ox3 < 0) {
                    gks = 0; gke = ngh_l - 1;
                  } else if (ox3 > 0) {
                    gks = ngh_l + ncells_l;
                    gke = ngh_l + ncells_l + ngh_l - 1;
                  } else {
                    gks = ngh_l;
                    gke = ngh_l + ncells_l - 1;
                  }

                  for (int v = 0; v < nvar_l; ++v) {
                    for (int gk = gks; gk <= gke; ++gk) {
                      for (int gj = gjs; gj <= gje; ++gj) {
                        for (int gi = gis; gi <= gie; ++gi) {
                          int si, sj, sk;
                          if (ox1 < 0)      si = ngh_l + ncells_l - 1;
                          else if (ox1 > 0) si = ngh_l;
                          else
                            si = ngh_l + child_x*half + (gi - ngh_l)/2;
                          if (ox2 < 0)      sj = ngh_l + ncells_l - 1;
                          else if (ox2 > 0) sj = ngh_l;
                          else
                            sj = ngh_l + child_y*half + (gj - ngh_l)/2;
                          if (ox3 < 0)      sk = ngh_l + ncells_l - 1;
                          else if (ox3 > 0) sk = ngh_l;
                          else
                            sk = ngh_l + child_z*half + (gk - ngh_l)/2;

                          u(m, v, gk, gj, gi) = u(dm, v, sk, sj, si);
                        }
                      }
                    }
                  }

                } else {
                  // Finer neighbor, edge/corner: simple restriction
                  int sub_x = 0, sub_y = 0, sub_z = 0;
                  if (nface == 2) {
                    if (ox1 == 0) sub_x = f1;
                    if (ox2 == 0) sub_y = f1;
                    if (ox3 == 0) sub_z = f1;
                  }
                  int gis, gie, gjs, gje, gks, gke;
                  if (ox1 < 0) {
                    gis = 0; gie = ngh_l - 1;
                  } else if (ox1 > 0) {
                    gis = ngh_l + ncells_l;
                    gie = ngh_l + ncells_l + ngh_l - 1;
                  } else {
                    gis = ngh_l + sub_x*half;
                    gie = ngh_l + sub_x*half + half - 1;
                  }
                  if (ox2 < 0) {
                    gjs = 0; gje = ngh_l - 1;
                  } else if (ox2 > 0) {
                    gjs = ngh_l + ncells_l;
                    gje = ngh_l + ncells_l + ngh_l - 1;
                  } else {
                    gjs = ngh_l + sub_y*half;
                    gje = ngh_l + sub_y*half + half - 1;
                  }
                  if (ox3 < 0) {
                    gks = 0; gke = ngh_l - 1;
                  } else if (ox3 > 0) {
                    gks = ngh_l + ncells_l;
                    gke = ngh_l + ncells_l + ngh_l - 1;
                  } else {
                    gks = ngh_l + sub_z*half;
                    gke = ngh_l + sub_z*half + half - 1;
                  }

                  for (int v = 0; v < nvar_l; ++v) {
                    for (int gk = gks; gk <= gke; ++gk) {
                      for (int gj = gjs; gj <= gje; ++gj) {
                        for (int gi = gis; gi <= gie; ++gi) {
                          int fi0, fi1, fj0, fj1, fk0, fk1;
                          if (ox1 < 0) {
                            fi0 = ngh_l+ncells_l-2; fi1 = ngh_l+ncells_l-1;
                          } else if (ox1 > 0) {
                            fi0 = ngh_l; fi1 = ngh_l + 1;
                          } else {
                            fi0 = ngh_l+2*(gi-(ngh_l+sub_x*half)); fi1 = fi0+1;
                          }
                          if (ox2 < 0) {
                            fj0 = ngh_l+ncells_l-2; fj1 = ngh_l+ncells_l-1;
                          } else if (ox2 > 0) {
                            fj0 = ngh_l; fj1 = ngh_l + 1;
                          } else {
                            fj0 = ngh_l+2*(gj-(ngh_l+sub_y*half)); fj1 = fj0+1;
                          }
                          if (ox3 < 0) {
                            fk0 = ngh_l+ncells_l-2; fk1 = ngh_l+ncells_l-1;
                          } else if (ox3 > 0) {
                            fk0 = ngh_l; fk1 = ngh_l + 1;
                          } else {
                            fk0 = ngh_l+2*(gk-(ngh_l+sub_z*half)); fk1 = fk0+1;
                          }
                          u(m, v, gk, gj, gi) = 0.125 * (
                            u(dm,v,fk0,fj0,fi0) + u(dm,v,fk0,fj0,fi1) +
                            u(dm,v,fk0,fj1,fi0) + u(dm,v,fk0,fj1,fi1) +
                            u(dm,v,fk1,fj0,fi0) + u(dm,v,fk1,fj0,fi1) +
                            u(dm,v,fk1,fj1,fi0) + u(dm,v,fk1,fj1,fi1));
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
  });

#if MPI_PARALLEL_ENABLED
  // Cross-rank fine-coarse ghost fill via synchronous MPI.
  // Each rank sends its block's data at the current MG level for every cross-rank
  // FC neighbor pair, then applies prolongation/restriction using the received data.
  {
    auto &nghbr_h = pmy_pack->pmb->nghbr;
    auto &mblev_h = pmy_pack->pmb->mb_lev;

    struct FCPair {
      int m, n, ox1, ox2, ox3, f1, f2, nface;
      int remote_rank, remote_gid, nlev, m_lev;
    };
    std::vector<FCPair> pairs;
    for (int m = 0; m < nmb; ++m) {
      for (int ox3 = -1; ox3 <= 1; ++ox3) {
        for (int ox2 = -1; ox2 <= 1; ++ox2) {
          for (int ox1 = -1; ox1 <= 1; ++ox1) {
            if (ox1 == 0 && ox2 == 0 && ox3 == 0) continue;
            int nf_ = (ox1 != 0 ? 1 : 0)
                      + (ox2 != 0 ? 1 : 0)
                      + (ox3 != 0 ? 1 : 0);
            int f2_max = (nf_ == 1) ? 1 : 0;
            int f1_max = (nf_ <= 2) ? 1 : 0;
            for (int f2_ = 0; f2_ <= f2_max; ++f2_) {
              for (int f1_ = 0; f1_ <= f1_max; ++f1_) {
                int nn = NeighborIndex(ox1, ox2, ox3, f1_, f2_);
                if (nn < 0 || nn >= nnghbr) continue;
                if (nghbr_h.h_view(m,nn).gid < 0) continue;
                int nlev_n = nghbr_h.h_view(m,nn).lev;
                if (nlev_n == mblev_h.h_view(m)) continue;
                if (nghbr_h.h_view(m,nn).rank == my_rank) continue;
                FCPair p;
                p.m = m; p.n = nn;
                p.ox1 = ox1; p.ox2 = ox2; p.ox3 = ox3;
                p.f1 = f1_; p.f2 = f2_;
                p.nface = (ox1 != 0 ? 1 : 0)
                          + (ox2 != 0 ? 1 : 0)
                          + (ox3 != 0 ? 1 : 0);
                p.remote_rank = nghbr_h.h_view(m,nn).rank;
                p.remote_gid = nghbr_h.h_view(m,nn).gid;
                p.nlev = nlev_n;
                p.m_lev = mblev_h.h_view(m);
                pairs.push_back(p);
              }
            }
          }
        }
      }
    }

    if (!pairs.empty()) {
      Kokkos::fence();
      int ntot = ncells + 2*ngh;
      int blk_sz = nvar * ntot * ntot * ntot;

      int np = static_cast<int>(pairs.size());
      std::vector<std::vector<Real>> sdata(np), rdata(np);
      std::vector<MPI_Request> sreqs(np, MPI_REQUEST_NULL);
      std::vector<MPI_Request> rreqs(np, MPI_REQUEST_NULL);

      for (int p = 0; p < np; ++p) {
        rdata[p].resize(blk_sz);
        int tag = CreateBvals_MPI_Tag(pairs[p].m, pairs[p].n + 64);
        MPI_Irecv(rdata[p].data(), blk_sz, MPI_ATHENA_REAL,
                  pairs[p].remote_rank, tag, comm_vars, &rreqs[p]);
      }
      for (int p = 0; p < np; ++p) {
        sdata[p].resize(blk_sz);
        int bm = pairs[p].m;
        int idx = 0;
        for (int v = 0; v < nvar; ++v)
          for (int k = 0; k < ntot; ++k)
            for (int j = 0; j < ntot; ++j)
              for (int i = 0; i < ntot; ++i)
                sdata[p][idx++] = u(bm, v, k, j, i);
        int dn = nghbr_h.h_view(bm, pairs[p].n).dest;
        int rlid = pairs[p].remote_gid
                   - pmy_pack->pmesh->gids_eachrank[pairs[p].remote_rank];
        int stag = CreateBvals_MPI_Tag(rlid, dn + 64);
        MPI_Isend(sdata[p].data(), blk_sz, MPI_ATHENA_REAL,
                  pairs[p].remote_rank, stag, comm_vars, &sreqs[p]);
      }

      MPI_Waitall(np, rreqs.data(), MPI_STATUSES_IGNORE);

      constexpr Real ot_h = 1.0/3.0;
      for (int p = 0; p < np; ++p) {
        auto &pr = pairs[p];
        int ml = pr.m;
        // Compute child offsets from global LogicalLocation data
        int gid_ml = pmy_pack->pmb->mb_gid.h_view(ml);
        LogicalLocation &loc_ml = pmy_pack->pmesh->lloc_eachmb[gid_ml];
        int root_level = pmy_pack->pmesh->root_level;
        int cx = (loc_ml.level > root_level) ?
                 static_cast<int>(loc_ml.lx1 & 1) : 0;
        int cy = (loc_ml.level > root_level) ?
                 static_cast<int>(loc_ml.lx2 & 1) : 0;
        int cz = (loc_ml.level > root_level) ?
                 static_cast<int>(loc_ml.lx3 & 1) : 0;
        int hl = ncells / 2;
        int nl = pr.nlev, mlev = pr.m_lev;
        int nf = pr.nface;
        int ox1 = pr.ox1, ox2 = pr.ox2, ox3 = pr.ox3;
        int f1_ = pr.f1, f2_ = pr.f2;

        auto R = [&](int v, int k, int j, int i) -> Real {
          return rdata[p][((v*ntot + k)*ntot + j)*ntot + i];
        };

        if (nl < mlev && nf == 1) {
          if (ox1 != 0) {
            int fig = (ox1 < 0) ? ngh - 1 : ngh + ncells;
            int fi = (ox1 < 0) ? ngh : ngh + ncells - 1;
            int si = (ox1 < 0) ? ngh + ncells - 1 : ngh;
            int sj0 = ngh + cy*hl;
            int sk0 = ngh + cz*hl;
            for (int v = 0; v < nvar; ++v) {
              for (int sk = sk0; sk < sk0+hl; ++sk) {
                for (int sj = sj0; sj < sj0+hl; ++sj) {
                  int fj = ngh + 2*(sj - sj0);
                  int fk = ngh + 2*(sk - sk0);
                  Real cc = R(v, sk, sj, si);
                  int sjm = (sj > ngh) ? sj-1 : sj;
                  int sjp = (sj < ngh+ncells-1) ? sj+1 : sj;
                  int skm = (sk > ngh) ? sk-1 : sk;
                  int skp = (sk < ngh+ncells-1) ? sk+1 : sk;
                  Real gy = 0.125 *
                      (R(v,sk,sjp,si) - R(v,sk,sjm,si));
                  Real gz = 0.125 *
                      (R(v,skp,sj,si) - R(v,skm,sj,si));
                  u(ml,v,fk,fj,fig) =
                      ot_h*(2.0*(cc-gy-gz) + u(ml,v,fk,fj,fi));
                  u(ml,v,fk,fj+1,fig) =
                      ot_h*(2.0*(cc+gy-gz) + u(ml,v,fk,fj+1,fi));
                  u(ml,v,fk+1,fj,fig) =
                      ot_h*(2.0*(cc-gy+gz) + u(ml,v,fk+1,fj,fi));
                  u(ml,v,fk+1,fj+1,fig) =
                      ot_h*(2.0*(cc+gy+gz) + u(ml,v,fk+1,fj+1,fi));
                }
              }
            }
          } else if (ox2 != 0) {
            int fjg = (ox2 < 0) ? ngh - 1 : ngh + ncells;
            int fj = (ox2 < 0) ? ngh : ngh + ncells - 1;
            int sj = (ox2 < 0) ? ngh + ncells - 1 : ngh;
            int si0 = ngh + cx*hl;
            int sk0 = ngh + cz*hl;
            for (int v = 0; v < nvar; ++v) {
              for (int sk = sk0; sk < sk0+hl; ++sk) {
                for (int si = si0; si < si0+hl; ++si) {
                  int fi = ngh + 2*(si - si0);
                  int fk = ngh + 2*(sk - sk0);
                  Real cc = R(v, sk, sj, si);
                  int sim = (si > ngh) ? si-1 : si;
                  int sip = (si < ngh+ncells-1) ? si+1 : si;
                  int skm = (sk > ngh) ? sk-1 : sk;
                  int skp = (sk < ngh+ncells-1) ? sk+1 : sk;
                  Real gx = 0.125 *
                      (R(v,sk,sj,sip) - R(v,sk,sj,sim));
                  Real gz = 0.125 *
                      (R(v,skp,sj,si) - R(v,skm,sj,si));
                  u(ml,v,fk,fjg,fi) =
                      ot_h*(2.0*(cc-gx-gz) + u(ml,v,fk,fj,fi));
                  u(ml,v,fk,fjg,fi+1) =
                      ot_h*(2.0*(cc+gx-gz) + u(ml,v,fk,fj,fi+1));
                  u(ml,v,fk+1,fjg,fi) =
                      ot_h*(2.0*(cc-gx+gz) + u(ml,v,fk+1,fj,fi));
                  u(ml,v,fk+1,fjg,fi+1) =
                      ot_h*(2.0*(cc+gx+gz) + u(ml,v,fk+1,fj,fi+1));
                }
              }
            }
          } else {
            int fkg = (ox3 < 0) ? ngh - 1 : ngh + ncells;
            int fk = (ox3 < 0) ? ngh : ngh + ncells - 1;
            int sk = (ox3 < 0) ? ngh + ncells - 1 : ngh;
            int si0 = ngh + cx*hl;
            int sj0 = ngh + cy*hl;
            for (int v = 0; v < nvar; ++v) {
              for (int sj = sj0; sj < sj0+hl; ++sj) {
                for (int si = si0; si < si0+hl; ++si) {
                  int fi = ngh + 2*(si - si0);
                  int fj = ngh + 2*(sj - sj0);
                  Real cc = R(v, sk, sj, si);
                  int sim = (si > ngh) ? si-1 : si;
                  int sip = (si < ngh+ncells-1) ? si+1 : si;
                  int sjm = (sj > ngh) ? sj-1 : sj;
                  int sjp = (sj < ngh+ncells-1) ? sj+1 : sj;
                  Real gx = 0.125 *
                      (R(v,sk,sj,sip) - R(v,sk,sj,sim));
                  Real gy = 0.125 *
                      (R(v,sk,sjp,si) - R(v,sk,sjm,si));
                  u(ml,v,fkg,fj,fi) =
                      ot_h*(2.0*(cc-gx-gy) + u(ml,v,fk,fj,fi));
                  u(ml,v,fkg,fj,fi+1) =
                      ot_h*(2.0*(cc+gx-gy) + u(ml,v,fk,fj,fi+1));
                  u(ml,v,fkg,fj+1,fi) =
                      ot_h*(2.0*(cc-gx+gy) + u(ml,v,fk,fj+1,fi));
                  u(ml,v,fkg,fj+1,fi+1) =
                      ot_h*(2.0*(cc+gx+gy) + u(ml,v,fk,fj+1,fi+1));
                }
              }
            }
          }
        } else if (nl > mlev && nf == 1) {
          int sx = 0, sy = 0, sz = 0;
          if (ox1 != 0) {
            sy = f1_; sz = f2_;
          }
          if (ox2 != 0) {
            sx = f1_; sz = f2_;
          }
          if (ox3 != 0) {
            sx = f1_; sy = f2_;
          }
          int gis, gie, gjs, gje, gks, gke;
          if (ox1 < 0) {
            gis = 0; gie = ngh - 1;
          } else if (ox1 > 0) {
            gis = ngh + ncells;
            gie = ngh + ncells + ngh - 1;
          } else {
            gis = ngh + sx*hl;
            gie = ngh + sx*hl + hl - 1;
          }
          if (ox2 < 0) {
            gjs = 0; gje = ngh - 1;
          } else if (ox2 > 0) {
            gjs = ngh + ncells;
            gje = ngh + ncells + ngh - 1;
          } else {
            gjs = ngh + sy*hl;
            gje = ngh + sy*hl + hl - 1;
          }
          if (ox3 < 0) {
            gks = 0; gke = ngh - 1;
          } else if (ox3 > 0) {
            gks = ngh + ncells;
            gke = ngh + ncells + ngh - 1;
          } else {
            gks = ngh + sz*hl;
            gke = ngh + sz*hl + hl - 1;
          }
          int oi = (ox1 < 0) ? 1 : (ox1 > 0) ? -1 : 0;
          int oj = (ox2 < 0) ? 1 : (ox2 > 0) ? -1 : 0;
          int ok = (ox3 < 0) ? 1 : (ox3 > 0) ? -1 : 0;
          if (ox1 != 0) {
            int fi = (ox1 > 0) ? ngh : ngh + ncells - 1;
            for (int v = 0; v < nvar; ++v) {
              for (int gk = gks; gk <= gke; ++gk) {
                for (int gj = gjs; gj <= gje; ++gj) {
                  int fj0 = ngh + 2*(gj - (ngh + sy*hl));
                  int fk0 = ngh + 2*(gk - (ngh + sz*hl));
                  Real fa = 0.25 *
                      (R(v,fk0,fj0,fi) + R(v,fk0,fj0+1,fi)
                       + R(v,fk0+1,fj0,fi)
                       + R(v,fk0+1,fj0+1,fi));
                  u(ml,v,gk,gj,gis) =
                      ot_h*(4.0*fa
                            - u(ml,v,gk+ok,gj+oj,gis+oi));
                }
              }
            }
          } else if (ox2 != 0) {
            int fj = (ox2 > 0) ? ngh : ngh + ncells - 1;
            for (int v = 0; v < nvar; ++v) {
              for (int gk = gks; gk <= gke; ++gk) {
                for (int gi = gis; gi <= gie; ++gi) {
                  int fi0 = ngh + 2*(gi - (ngh + sx*hl));
                  int fk0 = ngh + 2*(gk - (ngh + sz*hl));
                  Real fa = 0.25 *
                      (R(v,fk0,fj,fi0) + R(v,fk0,fj,fi0+1)
                       + R(v,fk0+1,fj,fi0)
                       + R(v,fk0+1,fj,fi0+1));
                  u(ml,v,gk,gjs,gi) =
                      ot_h*(4.0*fa
                            - u(ml,v,gk+ok,gjs+oj,gi+oi));
                }
              }
            }
          } else {
            int fk = (ox3 > 0) ? ngh : ngh + ncells - 1;
            for (int v = 0; v < nvar; ++v) {
              for (int gj = gjs; gj <= gje; ++gj) {
                for (int gi = gis; gi <= gie; ++gi) {
                  int fi0 = ngh + 2*(gi - (ngh + sx*hl));
                  int fj0 = ngh + 2*(gj - (ngh + sy*hl));
                  Real fa = 0.25 *
                      (R(v,fk,fj0,fi0) + R(v,fk,fj0,fi0+1)
                       + R(v,fk,fj0+1,fi0)
                       + R(v,fk,fj0+1,fi0+1));
                  u(ml,v,gks,gj,gi) =
                      ot_h*(4.0*fa
                            - u(ml,v,gks+ok,gj+oj,gi+oi));
                }
              }
            }
          }
        } else if (nl < mlev) {
          int gis, gie, gjs, gje, gks, gke;
          if (ox1 < 0) {
            gis = 0; gie = ngh - 1;
          } else if (ox1 > 0) {
            gis = ngh + ncells;
            gie = ngh + ncells + ngh - 1;
          } else {
            gis = ngh; gie = ngh + ncells - 1;
          }
          if (ox2 < 0) {
            gjs = 0; gje = ngh - 1;
          } else if (ox2 > 0) {
            gjs = ngh + ncells;
            gje = ngh + ncells + ngh - 1;
          } else {
            gjs = ngh; gje = ngh + ncells - 1;
          }
          if (ox3 < 0) {
            gks = 0; gke = ngh - 1;
          } else if (ox3 > 0) {
            gks = ngh + ncells;
            gke = ngh + ncells + ngh - 1;
          } else {
            gks = ngh; gke = ngh + ncells - 1;
          }
          for (int v = 0; v < nvar; ++v) {
            for (int gk = gks; gk <= gke; ++gk) {
              for (int gj = gjs; gj <= gje; ++gj) {
                for (int gi = gis; gi <= gie; ++gi) {
                  int si, sj, sk;
                  if (ox1 < 0) {
                    si = ngh + ncells - 1;
                  } else if (ox1 > 0) {
                    si = ngh;
                  } else {
                    si = ngh + cx*hl + (gi - ngh)/2;
                  }
                  if (ox2 < 0) {
                    sj = ngh + ncells - 1;
                  } else if (ox2 > 0) {
                    sj = ngh;
                  } else {
                    sj = ngh + cy*hl + (gj - ngh)/2;
                  }
                  if (ox3 < 0) {
                    sk = ngh + ncells - 1;
                  } else if (ox3 > 0) {
                    sk = ngh;
                  } else {
                    sk = ngh + cz*hl + (gk - ngh)/2;
                  }
                  u(ml,v,gk,gj,gi) = R(v,sk,sj,si);
                }
              }
            }
          }
        } else {
          int sx = 0, sy = 0, sz = 0;
          if (nf == 2) {
            if (ox1 == 0) sx = f1_;
            if (ox2 == 0) sy = f1_;
            if (ox3 == 0) sz = f1_;
          }
          int gis, gie, gjs, gje, gks, gke;
          if (ox1 < 0) {
            gis = 0; gie = ngh - 1;
          } else if (ox1 > 0) {
            gis = ngh + ncells;
            gie = ngh + ncells + ngh - 1;
          } else {
            gis = ngh + sx*hl;
            gie = ngh + sx*hl + hl - 1;
          }
          if (ox2 < 0) {
            gjs = 0; gje = ngh - 1;
          } else if (ox2 > 0) {
            gjs = ngh + ncells;
            gje = ngh + ncells + ngh - 1;
          } else {
            gjs = ngh + sy*hl;
            gje = ngh + sy*hl + hl - 1;
          }
          if (ox3 < 0) {
            gks = 0; gke = ngh - 1;
          } else if (ox3 > 0) {
            gks = ngh + ncells;
            gke = ngh + ncells + ngh - 1;
          } else {
            gks = ngh + sz*hl;
            gke = ngh + sz*hl + hl - 1;
          }
          for (int v = 0; v < nvar; ++v) {
            for (int gk = gks; gk <= gke; ++gk) {
              for (int gj = gjs; gj <= gje; ++gj) {
                for (int gi = gis; gi <= gie; ++gi) {
                  int fi0, fi1, fj0, fj1, fk0, fk1;
                  if (ox1 < 0) {
                    fi0 = ngh + ncells - 2;
                    fi1 = ngh + ncells - 1;
                  } else if (ox1 > 0) {
                    fi0 = ngh; fi1 = ngh + 1;
                  } else {
                    fi0 = ngh + 2*(gi - (ngh + sx*hl));
                    fi1 = fi0 + 1;
                  }
                  if (ox2 < 0) {
                    fj0 = ngh + ncells - 2;
                    fj1 = ngh + ncells - 1;
                  } else if (ox2 > 0) {
                    fj0 = ngh; fj1 = ngh + 1;
                  } else {
                    fj0 = ngh + 2*(gj - (ngh + sy*hl));
                    fj1 = fj0 + 1;
                  }
                  if (ox3 < 0) {
                    fk0 = ngh + ncells - 2;
                    fk1 = ngh + ncells - 1;
                  } else if (ox3 > 0) {
                    fk0 = ngh; fk1 = ngh + 1;
                  } else {
                    fk0 = ngh + 2*(gk - (ngh + sz*hl));
                    fk1 = fk0 + 1;
                  }
                  u(ml,v,gk,gj,gi) = 0.125 * (
                      R(v,fk0,fj0,fi0) + R(v,fk0,fj0,fi1)
                      + R(v,fk0,fj1,fi0) + R(v,fk0,fj1,fi1)
                      + R(v,fk1,fj0,fi0) + R(v,fk1,fj0,fi1)
                      + R(v,fk1,fj1,fi0) + R(v,fk1,fj1,fi1));
                }
              }
            }
          }
        }
      }
      MPI_Waitall(np, sreqs.data(), MPI_STATUSES_IGNORE);
    }
  }
#endif

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MultigridBoundaryValues::PackAndSend()
//! \brief Pack restricted fluxes of multigrid variables at fine/coarse
//! boundaries into boundary buffers and send to neighbors.
//! Adapts to different block sizes per level.

//----------------------------------------------------------------------------------------
//! \fn MultigridBoundaryValues::GetPackList()
//! \brief Return this MG level's cached list of active (MeshBlock, neighbour) pairs for
//! PackMG and UnpackMG, rebuilding it if its key changed (src/cfc/DEVELOPMENT.md item
//! 64).
//! The selection, the index boxes and the buffer targets are exactly those of the former
//! per-slot kernels:
//!   - a pair is active if the neighbour exists and is not a fine/coarse neighbour on a
//!     level where fine/coarse exchange is skipped;
//!   - box: isame / icoar / ifine of send_mg_indcs_ (pack) or recv_mg_indcs_ (unpack) for
//!     nlev == / < / > mlev;
//!   - pack target: recvbuf[dest].vars row (gid - mb_gid(0)) if the neighbour is on this
//!     rank, else the aggregate send buffer at mg_send_agg_offset_(m*nnghbr+n) (MPI) or
//!     sendbuf[n].vars row m (no MPI);
//!   - unpack source: the aggregate recv buffer at mg_recv_agg_offset_(m*nnghbr+n) if
//!     that is >= 0 (MPI), else recvbuf[n].vars row m.
//! Pointers are formed as data() + row*stride(0) on the host (no View access);
//! stride(1) == 1 (LayoutRight) makes element bi of a row the same as vars(row, bi).
//! Neighbour data come from device mirrors of nghbr/mb_lev/mb_gid, i.e. what the former
//! kernels read. has_cross_rank is computed by the former host loop (nghbr.h_view) and
//! cached with the list.

MultigridBoundaryValues::MGPackListLevel &
MultigridBoundaryValues::GetPackList(const int nvar, const int lev, const bool skip_fc) {
  if (static_cast<int>(mg_pl_cache_.size()) < kMaxMGLevels) {
    mg_pl_cache_.resize(kMaxMGLevels);
  }
  if (lev < 0 || lev >= kMaxMGLevels) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "MG level " << lev << " outside the pack-list cache" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto &L = mg_pl_cache_[lev];

  const int nmb = pmy_pack->nmb_thispack;
  const int nnghbr = pmy_pack->pmb->nnghbr;
  const int seq = pmy_pack->pmesh->GetAMRLoadBalanceUpdateSeq();
  bool ptrs_ok = true;   // compared in place: no allocation on the per-round path
  for (int n = 0; n < 56 && ptrs_ok; ++n) {
    ptrs_ok = (L.buf_ptrs[n] == sendbuf[n].vars.data()) &&
              (L.buf_ptrs[56 + n] == recvbuf[n].vars.data());
  }
  const bool key_ok = L.built && L.buf_gen == mg_buf_gen_ &&
                      L.seq == seq && L.nmb == nmb && L.nvar == nvar &&
                      L.skip_fc == skip_fc &&
                      L.agg_send == mg_rank_sendbuf_vars_.data() &&
                      L.agg_recv == mg_rank_recvbuf_vars_.data() &&
                      L.soff == mg_send_agg_offset_.data() &&
                      L.roff == mg_recv_agg_offset_.data() &&
                      ptrs_ok;
  if (key_ok) return L;

#if MPI_PARALLEL_ENABLED
  const bool urp = true;
#else
  const bool urp = false;
#endif
  const int my_rank = global_variable::my_rank;
  auto nghbr_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(),
                                                     pmy_pack->pmb->nghbr.d_view);
  auto mblev_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(),
                                                     pmy_pack->pmb->mb_lev.d_view);
  auto mbgid_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(),
                                                     pmy_pack->pmb->mb_gid.d_view);
  auto soff_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), mg_send_agg_offset_);
  auto roff_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), mg_recv_agg_offset_);

  auto row_ptr = [&](const DvceArray2D<Real> &a, const int row,
                     const char *what) -> Real* {
    if (a.extent_int(1) > 1 && a.stride(1) != 1) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "MG " << what
                << " buffer is not LayoutRight (stride(1) = " << a.stride(1) << ")"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    return a.data() + static_cast<std::size_t>(row) * a.stride(0);
  };

  std::vector<MGPackPair> hp, hu;
  hp.reserve(nmb*nnghbr);
  hu.reserve(nmb*nnghbr);
  for (int m = 0; m < nmb; ++m) {
    for (int n = 0; n < nnghbr; ++n) {
      if (nghbr_h(m, n).gid < 0) continue;
      const int nlev = nghbr_h(m, n).lev;
      const int mlev = mblev_h(m);
      if ((nlev != mlev) && skip_fc) continue;
      const int off_rank = (nghbr_h(m, n).rank != my_rank) ? 1 : 0;
      const MeshBufferIndcs &sb = (nlev == mlev) ? send_mg_indcs_[n][lev].isame :
                                  (nlev < mlev) ? send_mg_indcs_[n][lev].icoar :
                                                  send_mg_indcs_[n][lev].ifine;
      const MeshBufferIndcs &rb = (nlev == mlev) ? recv_mg_indcs_[n][lev].isame :
                                  (nlev < mlev) ? recv_mg_indcs_[n][lev].icoar :
                                                  recv_mg_indcs_[n][lev].ifine;
      MGPackPair p;
      p.m = m; p.n = n; p.off_rank = off_rank;
      p.use_cbuf = (nlev < mlev) ? 1 : 0;
      p.il = sb.bis; p.ni = sb.bie - sb.bis + 1;
      p.jl = sb.bjs; p.nj = sb.bje - sb.bjs + 1;
      p.kl = sb.bks; p.nk = sb.bke - sb.bks + 1;
      if (!off_rank) {
        const int dm = nghbr_h(m, n).gid - mbgid_h(0);
        const int dn = nghbr_h(m, n).dest;
        p.buf = row_ptr(recvbuf[dn].vars, dm, "recv");
      } else if (urp) {
        p.buf = mg_rank_sendbuf_vars_.data() + soff_h(m*nnghbr + n);
      } else {
        p.buf = row_ptr(sendbuf[n].vars, m, "send");
      }
      hp.push_back(p);

      MGPackPair q = p;
      q.il = rb.bis; q.ni = rb.bie - rb.bis + 1;
      q.jl = rb.bjs; q.nj = rb.bje - rb.bjs + 1;
      q.kl = rb.bks; q.nk = rb.bke - rb.bks + 1;
      const int base = urp ? roff_h(m*nnghbr + n) : -1;
      q.buf = (base >= 0) ? (mg_rank_recvbuf_vars_.data() + base)
                          : row_ptr(recvbuf[n].vars, m, "recv");
      hu.push_back(q);
    }
  }

  // has_cross_rank: the former per-call host loop of PackAndSendMG, unchanged
  bool has_cross_rank = false;
  {
    auto &nghbr = pmy_pack->pmb->nghbr;
    auto &mblev = pmy_pack->pmb->mb_lev;
    for (int m=0; m<nmb && !has_cross_rank; ++m) {
      for (int n=0; n<nnghbr && !has_cross_rank; ++n) {
        if (nghbr.h_view(m,n).gid >= 0 && nghbr.h_view(m,n).rank != my_rank) {
          int nlev_h = nghbr.h_view(m,n).lev;
          int mlev_h = mblev.h_view(m);
          bool is_fc_h = (nlev_h != mlev_h);
          if (is_fc_h && skip_fc) continue;
          has_cross_rank = true;
        }
      }
    }
  }

  L.npairs = static_cast<int>(hp.size());
  const int nalloc = std::max(L.npairs, 1);
  L.pack = Kokkos::View<MGPackPair*, DevMemSpace>("mg_pack_pairs", nalloc);
  L.unpack = Kokkos::View<MGPackPair*, DevMemSpace>("mg_unpack_pairs", nalloc);
  {
    auto hpv = Kokkos::create_mirror_view(L.pack);
    auto huv = Kokkos::create_mirror_view(L.unpack);
    for (int i = 0; i < L.npairs; ++i) {
      hpv(i) = hp[i];
      huv(i) = hu[i];
    }
    Kokkos::deep_copy(L.pack, hpv);
    Kokkos::deep_copy(L.unpack, huv);
  }
  L.has_cross_rank = has_cross_rank;
  L.buf_gen = mg_buf_gen_;
  L.seq = seq; L.nmb = nmb; L.nvar = nvar; L.skip_fc = skip_fc;
  L.agg_send = mg_rank_sendbuf_vars_.data();
  L.agg_recv = mg_rank_recvbuf_vars_.data();
  L.soff = mg_send_agg_offset_.data();
  L.roff = mg_recv_agg_offset_.data();
  for (int n = 0; n < 56; ++n) {
    L.buf_ptrs[n] = sendbuf[n].vars.data();
    L.buf_ptrs[56 + n] = recvbuf[n].vars.data();
  }
  L.built = true;
  if (mg_packlist_check_ > 0) CheckPackList(L, nvar, lev, skip_fc);
  return L;
}

//----------------------------------------------------------------------------------------
//! \fn MultigridBoundaryValues::CheckPackList()
//! \brief Self-check of GetPackList (<cfc> mg_packlist_check = 1; src/cfc/DEVELOPMENT.md
//! item 64), host only, run after every list (re)build. (a) Re-derives, slot by slot in
//! the former kernels' (m, n) order and with their branch structure, every active pair's
//! box, source/target array and buffer address (full stride arithmetic, no LayoutRight
//! assumption), and compares with the list. (b) Checks that no two pack payloads overlap
//! and that no two unpack boxes of the same MeshBlock and target array overlap.
//! (c) Compares has_cross_rank with the list's off-rank flags. Aborts on the first
//! mismatch; prints one line per level otherwise.

void MultigridBoundaryValues::CheckPackList(const MGPackListLevel &L, const int nvar,
                                            const int lev, const bool skip_fc) {
  const int nmb = pmy_pack->nmb_thispack;
  const int nnghbr = pmy_pack->pmb->nnghbr;
  const int my_rank = global_variable::my_rank;
#if MPI_PARALLEL_ENABLED
  const bool urp = true;
#else
  const bool urp = false;
#endif
  auto fail = [&](const std::string &what, int m, int n) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MGPACKLIST check failed on rank " << my_rank << " level "
              << lev << " (m, n) = (" << m << ", " << n << "): " << what << std::endl;
    std::exit(EXIT_FAILURE);
  };
  auto nghbr_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(),
                                                     pmy_pack->pmb->nghbr.d_view);
  auto mblev_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(),
                                                     pmy_pack->pmb->mb_lev.d_view);
  auto mbgid_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(),
                                                     pmy_pack->pmb->mb_gid.d_view);
  auto soff_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), mg_send_agg_offset_);
  auto roff_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), mg_recv_agg_offset_);
  auto hp = Kokkos::create_mirror_view_and_copy(HostMemSpace(), L.pack);
  auto hu = Kokkos::create_mirror_view_and_copy(HostMemSpace(), L.unpack);
  auto &smgi = send_mg_indcs_;
  auto &rmgi = recv_mg_indcs_;

  // (a) former per-slot selection, literally
  int p = 0;
  for (int m = 0; m < nmb; ++m) {
    for (int n = 0; n < nnghbr; ++n) {
      if (nghbr_h(m, n).gid < 0) continue;
      int nlev = nghbr_h(m, n).lev;
      int mlev = mblev_h(m);
      bool is_fc = (nlev != mlev);
      if (is_fc && skip_fc) continue;
      if (p >= L.npairs) fail("list shorter than the per-slot selection", m, n);
      bool is_coarser = (nlev < mlev);
      int il, iu, jl, ju, kl, ku;
      if (nlev == mlev) {
        il = smgi[n][lev].isame.bis; iu = smgi[n][lev].isame.bie;
        jl = smgi[n][lev].isame.bjs; ju = smgi[n][lev].isame.bje;
        kl = smgi[n][lev].isame.bks; ku = smgi[n][lev].isame.bke;
      } else if (is_coarser) {
        il = smgi[n][lev].icoar.bis; iu = smgi[n][lev].icoar.bie;
        jl = smgi[n][lev].icoar.bjs; ju = smgi[n][lev].icoar.bje;
        kl = smgi[n][lev].icoar.bks; ku = smgi[n][lev].icoar.bke;
      } else {
        il = smgi[n][lev].ifine.bis; iu = smgi[n][lev].ifine.bie;
        jl = smgi[n][lev].ifine.bjs; ju = smgi[n][lev].ifine.bje;
        kl = smgi[n][lev].ifine.bks; ku = smgi[n][lev].ifine.bke;
      }
      int dm = nghbr_h(m, n).gid - mbgid_h(0);
      int dn = nghbr_h(m, n).dest;
      const Real *addr;   // address of payload element bi = 0
      if (nghbr_h(m, n).rank == my_rank) {
        addr = recvbuf[dn].vars.data() + dm*recvbuf[dn].vars.stride(0);
      } else if (urp) {
        if (soff_h(m*nnghbr + n) < 0) {
          fail("off-rank pack pair without a send offset", m, n);
        }
        addr = mg_rank_sendbuf_vars_.data() + soff_h(m*nnghbr + n);
      } else {
        addr = sendbuf[n].vars.data() + m*sendbuf[n].vars.stride(0);
      }
      const MGPackPair &a = hp(p);
      if (a.m != m || a.n != n) fail("pack pair order", m, n);
      if (a.use_cbuf != (is_coarser ? 1 : 0)) fail("pack source array", m, n);
      if (a.il != il || a.ni != iu-il+1 || a.jl != jl || a.nj != ju-jl+1 ||
          a.kl != kl || a.nk != ku-kl+1) fail("pack box", m, n);
      if (a.buf != addr) fail("pack target address", m, n);
      if (a.off_rank != ((nghbr_h(m, n).rank != my_rank) ? 1 : 0)) fail("off_rank", m, n);

      bool from_coarser = (nlev < mlev);
      if (nlev == mlev) {
        il = rmgi[n][lev].isame.bis; iu = rmgi[n][lev].isame.bie;
        jl = rmgi[n][lev].isame.bjs; ju = rmgi[n][lev].isame.bje;
        kl = rmgi[n][lev].isame.bks; ku = rmgi[n][lev].isame.bke;
      } else if (from_coarser) {
        il = rmgi[n][lev].icoar.bis; iu = rmgi[n][lev].icoar.bie;
        jl = rmgi[n][lev].icoar.bjs; ju = rmgi[n][lev].icoar.bje;
        kl = rmgi[n][lev].icoar.bks; ku = rmgi[n][lev].icoar.bke;
      } else {
        il = rmgi[n][lev].ifine.bis; iu = rmgi[n][lev].ifine.bie;
        jl = rmgi[n][lev].ifine.bjs; ju = rmgi[n][lev].ifine.bje;
        kl = rmgi[n][lev].ifine.bks; ku = rmgi[n][lev].ifine.bke;
      }
      const int base = urp ? roff_h(m*nnghbr + n) : -1;
      const Real *raddr = (base >= 0) ? mg_rank_recvbuf_vars_.data() + base
                        : recvbuf[n].vars.data() + m*recvbuf[n].vars.stride(0);
      if (nghbr_h(m, n).rank != my_rank && urp && base < 0)
        fail("off-rank unpack pair without a recv offset", m, n);
      const MGPackPair &b = hu(p);
      if (b.m != m || b.n != n) fail("unpack pair order", m, n);
      if (b.use_cbuf != (from_coarser ? 1 : 0)) fail("unpack target array", m, n);
      if (b.il != il || b.ni != iu-il+1 || b.jl != jl || b.nj != ju-jl+1 ||
          b.kl != kl || b.nk != ku-kl+1) fail("unpack box", m, n);
      if (b.buf != raddr) fail("unpack source address", m, n);
      ++p;
    }
  }
  if (p != L.npairs) fail("list longer than the per-slot selection", -1, -1);

  // (b) no overlapping writes. Pack: payload address ranges of all pairs disjoint.
  {
    std::vector<std::pair<const Real*, const Real*>> r;
    for (int i = 0; i < L.npairs; ++i) {
      const MGPackPair &a = hp(i);
      if (a.ni <= 0 || a.nj <= 0 || a.nk <= 0) fail("empty pack box", a.m, a.n);
      r.emplace_back(a.buf, a.buf + static_cast<std::size_t>(nvar)*a.ni*a.nj*a.nk);
    }
    std::sort(r.begin(), r.end());
    for (std::size_t i = 1; i < r.size(); ++i) {
      if (r[i].first < r[i-1].second) fail("overlapping pack payloads", -1, -1);
    }
  }
  // Unpack: boxes of pairs with the same m and target array (u or coarse_buf_) disjoint.
  for (int i = 0; i < L.npairs; ++i) {
    const MGPackPair &a = hu(i);
    if (a.ni <= 0 || a.nj <= 0 || a.nk <= 0) fail("empty unpack box", a.m, a.n);
    for (int j = i + 1; j < L.npairs && hu(j).m == a.m; ++j) {
      const MGPackPair &b = hu(j);
      if (b.use_cbuf != a.use_cbuf) continue;
      bool ox = a.il < b.il + b.ni && b.il < a.il + a.ni;
      bool oy = a.jl < b.jl + b.nj && b.jl < a.jl + a.nj;
      bool oz = a.kl < b.kl + b.nk && b.kl < a.kl + a.nk;
      if (ox && oy && oz) fail("overlapping unpack boxes with slot " +
                               std::to_string(b.n), a.m, a.n);
    }
  }
  // (c) has_cross_rank
  bool any_off = false;
  for (int i = 0; i < L.npairs; ++i) any_off = any_off || (hp(i).off_rank != 0);
  if (any_off != L.has_cross_rank) fail("has_cross_rank", -1, -1);

  std::cout << "MGPACKLIST rank " << my_rank << " lev " << lev << " nvar " << nvar
            << " skip_fc " << skip_fc << " pairs " << L.npairs << " of " << nmb*nnghbr
            << " slots, cross_rank " << L.has_cross_rank << ": OK" << std::endl;
}

TaskStatus MultigridBoundaryValues::PackAndSendMG(const DvceArray5D<Real> &u) {
  if (pmy_mg == nullptr) return TaskStatus::complete;

  int nvar = u.extent_int(1);
  int lev_ = pmy_mg->GetCurrentLevel();
  int nlev_total = pmy_mg->GetNumberOfLevels();
  int shift_ps = nlev_total - 1 - lev_;
  int ncells_ps = pmy_mg->GetSize() >> shift_ps;
  bool skip_fc_this_level = (ncells_ps < 2);

#if MPI_PARALLEL_ENABLED
  for (std::size_t i = 0; i < mg_send_var_reqs_.size(); ++i) {
    MPI_Wait(&mg_send_var_reqs_[i], MPI_STATUS_IGNORE);
  }
#endif

  // The kernels run over this level's cached list of active (MeshBlock, neighbour) pairs
  // (GetPackList; src/cfc/DEVELOPMENT.md item 64) instead of every nmb*nnghbr*nvar slot,
  // and capture only Views and scalars. Functors of >= 1800 B are launched indirectly by
  // Kokkos-SYCL (functor copied to host USM after a fence on the previous indirect
  // launch, kokkos/core/src/SYCL/Kokkos_SYCL_Instance.hpp:236-320); the static_asserts
  // below keep these under that threshold. Element bi of a pair's payload, and so every
  // buffer entry written, is the same as before: pure copies, no arithmetic.
  auto &L = GetPackList(nvar, lev_, skip_fc_this_level);
  const int npairs = L.npairs;
  auto plist = L.pack;
  auto uu = u;
  auto cbuf = coarse_buf_;
  const int nvar_l = nvar;

  auto pack_team = KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int p = tmember.league_rank() / nvar_l;
    const int v = tmember.league_rank() - p * nvar_l;
    const MGPackPair pp = plist(p);
    const int nkj = pp.nk * pp.nj;
    Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj),
    [&](const int idx) {
      int k = idx / pp.nj;
      int j = (idx - k * pp.nj) + pp.jl;
      k += pp.kl;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember, pp.il, pp.il + pp.ni),
      [&](const int i) {
        const int bi = (i-pp.il + pp.ni*(j-pp.jl + pp.nj*(k-pp.kl + pp.nk*v)));
        pp.buf[bi] = pp.use_cbuf ? cbuf(pp.m, v, k, j, i) : uu(pp.m, v, k, j, i);
      });
    });
  };
  auto pack_flat = KOKKOS_LAMBDA(const int pv) {
    const int p = pv / nvar_l;
    const int v = pv - p * nvar_l;
    const MGPackPair pp = plist(p);
    for (int k = pp.kl; k < pp.kl + pp.nk; ++k) {
      for (int j = pp.jl; j < pp.jl + pp.nj; ++j) {
        for (int i = pp.il; i < pp.il + pp.ni; ++i) {
          const int bi = (i-pp.il + pp.ni*(j-pp.jl + pp.nj*(k-pp.kl + pp.nk*v)));
          pp.buf[bi] = pp.use_cbuf ? cbuf(pp.m, v, k, j, i) : uu(pp.m, v, k, j, i);
        }
      }
    }
  };
  // Kokkos-SYCL indirect-launch threshold: Kokkos_SYCL_Instance.hpp:236-246
  static_assert(sizeof(pack_team) < 1800, "PackMG functor would be launched indirectly");
  static_assert(sizeof(pack_flat) < 1800,
                "PackMGFlat functor would be launched indirectly");
  if (mg_packlist_check_ > 0 && !mg_packlist_sizes_printed_ &&
      global_variable::my_rank == 0) {
    std::cout << "MGPACKLIST functor bytes: PackMG " << sizeof(pack_team)
              << " PackMGFlat " << sizeof(pack_flat) << std::endl;
  }

  if (npairs > 0) {
    if (ncells_ps <= kMGPackFlatMaxCells) {
      Kokkos::parallel_for("PackMGFlat",
                           Kokkos::RangePolicy<>(DevExeSpace(), 0, npairs * nvar_l),
                           pack_flat);
    } else {
      Kokkos::TeamPolicy<> policy(DevExeSpace(), npairs * nvar_l, Kokkos::AUTO);
      Kokkos::parallel_for("PackMG", policy, pack_team);
    }
  }

  #if MPI_PARALLEL_ENABLED
  // former per-call loop, cached per level
  const bool has_cross_rank = L.has_cross_rank;
  if (has_cross_rank) {
    Kokkos::fence();
    unpack_unfenced_ = false;   // also completes any earlier UnpackMG (PostRecvMG)
  }

  bool no_errors = true;
  // PackMG already wrote every off-rank payload directly into
  // mg_rank_sendbuf_vars_ at its per-(m,n) base offset, so the only device
  // work left before MPI is the fence above. No separate aggregate kernel and
  // no header message: the receiver derives the payload layout from its own
  // sorted recv entries (see BuildRankPackedMGMetadata).
  for (std::size_t i = 0; i < mg_send_var_msgs_.size(); ++i) {
    const auto &msg = mg_send_var_msgs_[i];
    int ierr_d = MPI_Isend(mg_rank_sendbuf_vars_.data() + msg.offset, msg.data_size,
                           MPI_ATHENA_REAL, msg.rank, 97, comm_vars,
                           &mg_send_var_reqs_[i]);
    if (ierr_d != MPI_SUCCESS) {
      no_errors = false;
    } else {
      pmy_mg->pmy_driver_->mg_timers_.msg_count += 1;
      pmy_mg->pmy_driver_->mg_timers_.bytes_sent +=
          (msg.data_size * static_cast<int64_t>(sizeof(Real)));
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in posting sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MultigridBoundaryValuesCC::RecvAndUnpackMG()
//! \brief Receive and unpack cell-centered multigrid variables.
//! Handles ghost-cell filling at each multigrid level independently.

TaskStatus MultigridBoundaryValues::RecvAndUnpackMG(DvceArray5D<Real> &u) {
  if (pmy_mg == nullptr) return TaskStatus::complete;
  int shift_ru = pmy_mg->GetNumberOfLevels() - 1 - pmy_mg->GetCurrentLevel();
  int ncells_ru = pmy_mg->GetSize() >> shift_ru;
  bool skip_fc_this_level = (ncells_ru < 2);

  #if MPI_PARALLEL_ENABLED
  //----- STEP 1: check that recv boundary buffer communications have all completed
  bool bflag = false;
  for (std::size_t i = 0; i < mg_recv_var_msgs_.size(); ++i) {
    int test_d = 0;
    int ierr_d = MPI_Test(&mg_recv_var_reqs_[i], &test_d, MPI_STATUS_IGNORE);
    if (ierr_d != MPI_SUCCESS) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "MPI error in testing rank-packed MG receives"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (!(static_cast<bool>(test_d))) {
      bflag = true;
    }
  }
  if (bflag) {
    return TaskStatus::incomplete;
  }
#endif

  //----- STEP 2: buffers have all completed, so unpack
  // Off-rank payloads are read directly out of the rank-packed aggregate recv
  // buffer (fuses the former MGRankUnpackScatter kernel); on-rank payloads sit
  // in their per-neighbour recv buffers, written there directly by the sender.
  // Over the level's cached active-pair list (see PackAndSendMG; item 64); each
  // entry's buf is the aggregate recv buffer at mg_recv_agg_offset_ (MPI, off-rank) or
  // recvbuf[n].vars row m, so element bi is read from the same place as before.
  int nvar = u.extent_int(1);
  int lev_ = pmy_mg->GetCurrentLevel();
  auto &L = GetPackList(nvar, lev_, skip_fc_this_level);
  const int npairs = L.npairs;
  auto plist = L.unpack;
  auto uu = u;
  auto cbuf = coarse_buf_;
  const int nvar_l = nvar;

  auto unpack_team = KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int p = tmember.league_rank() / nvar_l;
    const int v = tmember.league_rank() - p * nvar_l;
    const MGPackPair pp = plist(p);
    const int nkj = pp.nk * pp.nj;
    Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj),
    [&](const int idx) {
      int k = idx / pp.nj;
      int j = (idx - k * pp.nj) + pp.jl;
      k += pp.kl;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember, pp.il, pp.il + pp.ni),
      [&](const int i) {
        const int bi = (i-pp.il + pp.ni*(j-pp.jl + pp.nj*(k-pp.kl + pp.nk*v)));
        if (pp.use_cbuf) {
          cbuf(pp.m, v, k, j, i) = pp.buf[bi];
        } else {
          uu(pp.m, v, k, j, i) = pp.buf[bi];
        }
      });
    });
  };
  auto unpack_flat = KOKKOS_LAMBDA(const int pv) {
    const int p = pv / nvar_l;
    const int v = pv - p * nvar_l;
    const MGPackPair pp = plist(p);
    for (int k = pp.kl; k < pp.kl + pp.nk; ++k) {
      for (int j = pp.jl; j < pp.jl + pp.nj; ++j) {
        for (int i = pp.il; i < pp.il + pp.ni; ++i) {
          const int bi = (i-pp.il + pp.ni*(j-pp.jl + pp.nj*(k-pp.kl + pp.nk*v)));
          if (pp.use_cbuf) {
            cbuf(pp.m, v, k, j, i) = pp.buf[bi];
          } else {
            uu(pp.m, v, k, j, i) = pp.buf[bi];
          }
        }
      }
    }
  };
  // Kokkos-SYCL indirect-launch threshold: Kokkos_SYCL_Instance.hpp:236-246
  static_assert(sizeof(unpack_team) < 1800,
                "UnpackMG functor would be launched indirectly");
  static_assert(sizeof(unpack_flat) < 1800,
                "UnpackMGFlat functor would be launched indirectly");
  if (mg_packlist_check_ > 0 && !mg_packlist_sizes_printed_) {
    if (global_variable::my_rank == 0) {
      std::cout << "MGPACKLIST functor bytes: UnpackMG " << sizeof(unpack_team)
                << " UnpackMGFlat " << sizeof(unpack_flat) << std::endl;
    }
    mg_packlist_sizes_printed_ = true;
  }

  unpack_unfenced_ = true;   // reads aggrbuf asynchronously: see PostRecvMG
  if (npairs > 0) {
    if (ncells_ru <= kMGPackFlatMaxCells) {
      Kokkos::parallel_for("UnpackMGFlat",
                           Kokkos::RangePolicy<>(DevExeSpace(), 0, npairs * nvar_l),
                           unpack_flat);
    } else {
      Kokkos::TeamPolicy<> policy(DevExeSpace(), npairs * nvar_l, Kokkos::AUTO);
      Kokkos::parallel_for("UnpackMG", policy, unpack_team);
    }
  }

  return TaskStatus::complete;
}

#if MPI_PARALLEL_ENABLED
//----------------------------------------------------------------------------------------
//! Save/Load the active rank-packed comm state to/from a per-level cache slot.
//! Views are ref-counted (O(1), no data copy); the vectors are small. The MPI
//! request vectors are saved/loaded so each level's in-flight sends stay paired
//! with that level's buffer across level transitions.

void MultigridBoundaryValues::SaveRankPackState(MGRankPackLevelState &s) {
  s.send_msgs = mg_send_var_msgs_;
  s.recv_msgs = mg_recv_var_msgs_;
  s.send_reqs = mg_send_var_reqs_;
  s.recv_reqs = mg_recv_var_reqs_;
  s.sendbuf   = mg_rank_sendbuf_vars_;
  s.recvbuf   = mg_rank_recvbuf_vars_;
  s.send_off  = mg_send_agg_offset_;
  s.recv_off  = mg_recv_agg_offset_;
}

void MultigridBoundaryValues::LoadRankPackState(const MGRankPackLevelState &s) {
  mg_send_var_msgs_     = s.send_msgs;
  mg_recv_var_msgs_     = s.recv_msgs;
  mg_send_var_reqs_     = s.send_reqs;
  mg_recv_var_reqs_     = s.recv_reqs;
  mg_rank_sendbuf_vars_ = s.sendbuf;
  mg_rank_recvbuf_vars_ = s.recvbuf;
  mg_send_agg_offset_   = s.send_off;
  mg_recv_agg_offset_   = s.recv_off;
}
#endif

//----------------------------------------------------------------------------------------
//! \fn  void MeshBoundaryValues::InitRecv
//! \brief Posts non-blocking receives (with MPI) for boundary communications of vars.

TaskStatus MultigridBoundaryValues::InitRecvMG(const int nvars) {
#if MPI_PARALLEL_ENABLED
  int lev_ = pmy_mg->GetCurrentLevel();
  int shift_ir = pmy_mg->GetNumberOfLevels() - 1 - lev_;
  int ncells_ir = pmy_mg->GetSize() >> shift_ir;
  bool skip_fc_ir = (ncells_ir < 2);
  const int mesh_seq = pmy_pack->pmesh->GetAMRLoadBalanceUpdateSeq();

  // Per-level metadata cache (see header). The V-cycle revisits each level
  // ~12x/cycle; building metadata each time meant ~85 rebuilds/cycle, each
  // doing several device allocations + H2D deep_copies that dominate at small
  // payloads. Build each level once and persist it; write the active level's
  // state back before switching so its in-flight requests + buffers stay paired.
  if (static_cast<int>(mg_rp_cache_.size()) < kMaxMGLevels) {
    mg_rp_cache_.resize(kMaxMGLevels);
  }
  if (mg_rp_cache_seq_ != mesh_seq) {            // AMR/regrid: invalidate all levels
    for (auto &s : mg_rp_cache_) s = MGRankPackLevelState();
    mg_rp_cache_seq_ = mesh_seq;
    mg_rp_active_level_ = -1;
  }
  if (lev_ >= 0 && lev_ < kMaxMGLevels) {
    auto &c = mg_rp_cache_[lev_];
    const bool key_ok = c.built && c.nvars == nvars && c.skip_fc == skip_fc_ir;
    if (lev_ != mg_rp_active_level_ || !key_ok) {
      if (mg_rp_active_level_ >= 0 && mg_rp_active_level_ != lev_) {
        SaveRankPackState(mg_rp_cache_[mg_rp_active_level_]);
      }
      if (!key_ok) {
        BuildRankPackedMGMetadata(nvars, lev_, skip_fc_ir);
        SaveRankPackState(c);
        c.built = true; c.nvars = nvars; c.skip_fc = skip_fc_ir;
      } else {
        LoadRankPackState(c);
      }
      mg_rp_active_level_ = lev_;
    }
  } else {
    // out-of-range level (should not happen): rebuild bare each call
    BuildRankPackedMGMetadata(nvars, lev_, skip_fc_ir);
    mg_rp_active_level_ = -1;
  }

#endif
  return TaskStatus::complete;   // the receives are posted by PostRecvMG
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MultigridBoundaryValues::PostRecvMG()
//! \brief Posts this level's rank-packed receives into the aggregate recv buffer. Called by
//! SendBoundary right after PackAndSendMG, whose pre-send fence has completed the previous
//! round's UnpackMG (which reads that buffer); fences itself if no fence followed the last
//! UnpackMG launch.

TaskStatus MultigridBoundaryValues::PostRecvMG() {
#if MPI_PARALLEL_ENABLED
  if (unpack_unfenced_ && !mg_recv_var_msgs_.empty()) {
    Kokkos::fence();
    unpack_unfenced_ = false;
    if (fallback_fences_++ == 0) {
      std::cout << "### WARNING MultigridBoundaryValues::PostRecvMG: no fence since the last "
                << "UnpackMG on rank " << global_variable::my_rank << "; fenced here (further "
                << "cases not reported)" << std::endl;
    }
  }

  // Payload-only Irecv: the receiver derives the payload layout from its own
  // sorted recv entries (see BuildRankPackedMGMetadata), so no header is sent.
  for (std::size_t i = 0; i < mg_recv_var_msgs_.size(); ++i) {
    auto &msg = mg_recv_var_msgs_[i];
    MPI_Wait(&mg_recv_var_reqs_[i], MPI_STATUS_IGNORE);

    int ierr_d = MPI_Irecv(mg_rank_recvbuf_vars_.data() + msg.offset, msg.data_size,
                           MPI_ATHENA_REAL, msg.rank, 97, comm_vars,
                           &mg_recv_var_reqs_[i]);
    if (ierr_d != MPI_SUCCESS) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
         << std::endl << "MPI error in posting rank-packed MG receives" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
#endif
  return TaskStatus::complete;
}


//----------------------------------------------------------------------------------------
//! \fn TaskStatus MultigridBoundaryValues::ClearRecvMG()
//! \brief Rank-packed-aware clear for MG sends/recvs.

TaskStatus MultigridBoundaryValues::ClearRecvMG() {
#if MPI_PARALLEL_ENABLED
  for (std::size_t i = 0; i < mg_recv_var_reqs_.size(); ++i) {
    MPI_Wait(&mg_recv_var_reqs_[i], MPI_STATUS_IGNORE);
  }
  return TaskStatus::complete;
#else
  return ClearRecv();
#endif
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MultigridBoundaryValues::ClearSendMG()
//! \brief Rank-packed-aware clear for MG sends/recvs.

TaskStatus MultigridBoundaryValues::ClearSendMG() {
#if MPI_PARALLEL_ENABLED
  for (std::size_t i = 0; i < mg_send_var_reqs_.size(); ++i) {
    MPI_Wait(&mg_send_var_reqs_[i], MPI_STATUS_IGNORE);
  }
  return TaskStatus::complete;
#else
  return ClearSend();
#endif
}
