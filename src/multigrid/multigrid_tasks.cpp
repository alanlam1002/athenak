//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mg_task_list.cpp
//! \brief functions for MultigridTaskList class

// C headers

// C++ headers
#include <iomanip>    // setprecision
#include <iostream>   // endl
#include <memory>
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <utility>

// Athena++ headers
#include "../athena.hpp"
#include "../globals.hpp"
#include "../mesh/mesh.hpp"
#include "multigrid.hpp"

//namespace multigrid{ // NOLINT (build/namespace)

//----------------------------------------------------------------------------------------
//! \fn void MultigridTaskList::DoTaskListOneStage(MultigridDriver *pmd)
//! \brief completes all tasks in this list, will not return until all are tasks done
void MultigridDriver::DoTaskListOneStage() {
  return;
}

TaskStatus MultigridDriver::ClearRecv(Driver *pdrive, int stage) {
  mgprof::Timer pt(prof_id_, mgprof::sBndClear);
  TaskStatus tstat;
  tstat = pmg->pbval->ClearRecvMG();
  return tstat;
}

TaskStatus MultigridDriver::ClearSend(Driver *pdrive, int stage) {
  mgprof::Timer pt(prof_id_, mgprof::sBndClear);
  TaskStatus tstat;
  tstat = pmg->pbval->ClearSendMG();
  return tstat;
}

TaskStatus MultigridDriver::StartReceive(Driver *pdrive, int stage) {
  mgprof::Timer pt(prof_id_, mgprof::sBndStart);
  // only switches the level's state; the receives are posted in SendBoundary after the
  // pre-send fence, so the previous round's UnpackMG no longer reads their buffer
  TaskStatus tstat = pmg->pbval->InitRecvMG(pmg->nvar_);
  return tstat;
}

TaskStatus MultigridDriver::SendBoundary(Driver *pdrive, int stage) {
  mgprof::Timer pt(prof_id_, mgprof::sBndSend);
  DvceArray5D<Real> u = pmg->GetCurrentData();
  TaskStatus tstat = pmg->pbval->PackAndSendMG(u);
  if (tstat == TaskStatus::complete) {
    tstat = pmg->pbval->PostRecvMG();
  }
  return tstat;
}

TaskStatus MultigridDriver::RecvBoundary(Driver *pdrive, int stage) {
  mgprof::Timer pt(prof_id_, mgprof::sBndRecv);
  DvceArray5D<Real> u = pmg->GetCurrentData();
  TaskStatus tstat = pmg->pbval->RecvAndUnpackMG(u);
  return tstat;
}

TaskStatus MultigridDriver::SmoothRed(Driver *pdrive, int stage) {
  mgprof::Timer pt(prof_id_, mgprof::sSmooth);
  pmg->SmoothPack(coffset_);
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::SmoothBlack(Driver *pdrive, int stage) {
  mgprof::Timer pt(prof_id_, mgprof::sSmooth);
  pmg->SmoothPack(1-coffset_);
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::Restrict(Driver *pdrive, int stage) {
  mgprof::Timer pt(prof_id_, mgprof::sRestrict);
  pmg->RestrictPack();
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::Prolongate(Driver *pdrive, int stage) {
  mgprof::Timer pt(prof_id_, mgprof::sProlong);
  pmg->ProlongateAndCorrectPack();
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::FMGProlongateTask(Driver *pdrive, int stage) {
  mgprof::Timer pt(prof_id_, mgprof::sProlong);
  pmg->FMGProlongatePack();
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::CalculateFASRHS(Driver *pdrive, int stage) {
  mgprof::Timer pt(prof_id_, mgprof::sFASRHS);
  if (current_level_ < fmglevel_) {
    pmg->StoreOldData();
    pmg->CalculateFASRHSPack();
  }
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::ProlongateBoundary(Driver *pdrive, int stage) {
  //pmg->pmgbval->ProlongateMultigridBoundariesFluxCons();
  return TaskStatus::complete;
}


TaskStatus MultigridDriver::ProlongateBoundaryForProlongation(Driver *pdrive, int stage) {
  //pmg->pmgbval->ProlongateMultigridBoundaries(true, false);
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::FillFCBoundary(Driver *pdrive, int stage) {
  mgprof::Timer pt(prof_id_, mgprof::sBndFill);
  if (nreflevel_ == 0) return TaskStatus::complete;
  DvceArray5D<Real> u = pmg->GetCurrentData();
  pmg->pbval->FillFineCoarseMGGhosts(u);
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::FillCoarseBoundary(Driver *pdrive, int stage) {
  mgprof::Timer pt(prof_id_, mgprof::sBndFill);
  if (nreflevel_ == 0) return TaskStatus::complete;
  DvceArray5D<Real> u = pmg->GetCurrentData();
  pmg->pbval->FillCoarseMG(u);
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::ProlongateFCBoundary(Driver *pdrive, int stage) {
  mgprof::Timer pt(prof_id_, mgprof::sBndProlFC);
  if (nreflevel_ == 0) return TaskStatus::complete;
  DvceArray5D<Real> u = pmg->GetCurrentData();
  pmg->pbval->ProlongateFCMG(u);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::BuildPhysicalFaceLists()
//! \brief V-002 C2: per face (inner_x1 .. outer_x3), the local MeshBlocks whose face is a
//! physical boundary (mb_bcs neither block nor periodic). Rebuilt when the mesh changes.

void MultigridDriver::BuildPhysicalFaceLists() {
  int nmb = pmy_pack_->nmb_thispack;
  int seq = pmy_mesh_->GetAMRLoadBalanceUpdateSeq();
  if (bface_seq_ == seq && bface_nmb_ == nmb) return;
  auto bcs_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(),
                                                   pmy_pack_->pmb->mb_bcs.d_view);
  for (int f = 0; f < 6; ++f) {
    std::vector<int> lst;
    for (int m = 0; m < nmb; ++m) {
      if (bcs_h(m, f) != BoundaryFlag::block && bcs_h(m, f) != BoundaryFlag::periodic)
        lst.push_back(m);
    }
    bface_n_[f] = static_cast<int>(lst.size());
    Kokkos::realloc(bface_list_[f], std::max(bface_n_[f], 1));
    auto h = Kokkos::create_mirror_view(bface_list_[f]);
    for (int i = 0; i < bface_n_[f]; ++i) h(i) = lst[i];
    Kokkos::deep_copy(bface_list_[f], h);
  }
  bface_seq_ = seq;
  bface_nmb_ = nmb;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MultigridDriver::PhysicalBoundary(Driver *pdrive, int stage)
//! \brief Outer (physical) MG boundary conditions: multipole, Robin, zero-fixed/reflect.
//!
//! V-002 C2: one launch per face in the original order (inner_x1, outer_x1, ..., outer_x3),
//! each parallel over (blocks with that physical face) x nvar x face cells, instead of one
//! work-item per MeshBlock looping over everything. The per-face launch order keeps the
//! zero-fixed/reflect branch (which spans the edges and reads ghosts written by earlier
//! faces) in the original order; the multipole/Robin branches have no cross-face reads.
//! The multipole coefficients live in a persistent device view, uploaded only when they
//! change (once per solve). Arithmetic expressions are unchanged.

TaskStatus MultigridDriver::PhysicalBoundary(Driver *pdrive, int stage) {
  mgprof::Timer pt(prof_id_, mgprof::sBndPhys);
  if (pmy_pack_->pmesh->strictly_periodic) return TaskStatus::complete;

  DvceArray5D<Real> u = pmg->GetCurrentData();
  int nvar = u.extent_int(1);
  int shift = pmg->GetLevelShift();
  int ngh = pmg->GetGhostCells();
  int ncells = pmg->GetSize() >> shift;
  if (ncells < 1) return TaskStatus::complete;

  BuildPhysicalFaceLists();

  bool has_multipole = false;
  for (int f = 0; f < 6; ++f) has_multipole |= (mg_mesh_bcs_[f] == BoundaryFlag::mg_multipole);

  bool use_mp = false;
  Real d_xo = 0.0, d_yo = 0.0, d_zo = 0.0;
  int d_order = 0;
  if (has_multipole && mporder_ > 0) {
    const int nc = nvar_*25;
    bool same = (static_cast<int>(mpc_h_cache_.size()) == nc);
    for (int c = 0; same && c < nc; ++c) same = (mpc_h_cache_[c] == mpcoeff_[c]);
    if (!same) {
      mpc_h_cache_.assign(mpcoeff_, mpcoeff_ + nc);
      Kokkos::realloc(mpc_d_, nc);
      auto h_mpc = Kokkos::create_mirror_view(mpc_d_);
      for (int c = 0; c < nc; ++c) h_mpc(c) = mpcoeff_[c];
      Kokkos::deep_copy(mpc_d_, h_mpc);
    }
    use_mp = true;
    d_xo = mpo_[0]; d_yo = mpo_[1]; d_zo = mpo_[2];
    d_order = mporder_;
  }
  auto d_mpc = mpc_d_;
  auto &mb_size = pmy_pack_->pmb->mb_size;

  int nxd[3] = {pmg->indcs_.nx1 >> shift, pmg->indcs_.nx2 >> shift,
                pmg->indcs_.nx3 >> shift};
  Real rorder = static_cast<Real>(robin_order_);

  for (int f = 0; f < 6; ++f) {
    const int nlist = bface_n_[f];
    if (nlist == 0) continue;
    const BoundaryFlag bc = mg_mesh_bcs_[f];
    const int dir = f/2;          // 0: x1, 1: x2, 2: x3
    const bool outer = (f % 2) == 1;
    const bool mp = (bc == BoundaryFlag::mg_multipole) && use_mp;
    const bool rb = (bc == BoundaryFlag::mg_robin);
    const int nt = (mp || rb) ? ncells : ncells + 2*ngh;   // transverse range
    const int t0 = (mp || rb) ? ngh : 0;
    const int ta_dir = (dir == 0) ? 1 : 0;                 // first transverse axis
    const int tb_dir = (dir == 2) ? 1 : 2;                 // second transverse axis
    const Real sign = (bc == BoundaryFlag::mg_zerofixed) ? -1.0 : 1.0;
    const int nd0 = nxd[0], nd1 = nxd[1], nd2 = nxd[2];
    auto lst = bface_list_[f];
    const int nv = nvar;
    const int total = nlist*nv*nt*nt;
    Kokkos::parallel_for("MGPhysicalBoundary",
      Kokkos::RangePolicy<DevExeSpace>(0, total),
      KOKKOS_LAMBDA(const int idx) {
        const int ia = idx % nt;              // first transverse index (fastest)
        const int ib = (idx / nt) % nt;       // second transverse index
        const int v = (idx / (nt*nt)) % nv;
        const int m = lst(idx / (nt*nt*nv));
        const auto &sz = mb_size.d_view(m);
        const Real xmin[3] = {sz.x1min, sz.x2min, sz.x3min};
        const Real xmax[3] = {sz.x1max, sz.x2max, sz.x3max};
        const int nxa[3] = {nd0, nd1, nd2};
        int ijk[3];
        ijk[ta_dir] = t0 + ia;
        ijk[tb_dir] = t0 + ib;
        // ghost/interior index along the face normal for layer n
        auto at = [&](int nrm) -> Real & {
          int c[3] = {ijk[0], ijk[1], ijk[2]};
          c[dir] = nrm;
          return u(m, v, c[2], c[1], c[0]);
        };
        if (mp) {
          const Real o[3] = {d_xo, d_yo, d_zo};
          Real c[3];
          for (int d = 0; d < 3; ++d) {
            if (d == dir) {
              c[d] = (outer ? xmax[d] : xmin[d]) - o[d];
            } else {
              Real dxl = (xmax[d] - xmin[d]) / static_cast<Real>(nxa[d]);
              c[d] = xmin[d] + (ijk[d]-ngh+0.5)*dxl - o[d];
            }
          }
          Real phis = EvalMultipolePhi(c[0], c[1], c[2], d_mpc.data() + v*25, d_order);
          for (int n = 0; n < ngh; ++n) {
            if (outer) at(ngh+ncells+n) = 2.0*phis - at(ngh+ncells-1-n);
            else       at(ngh-1-n) = 2.0*phis - at(ngh+n);
          }
        } else if (rb) {
          Real dxd = (xmax[dir] - xmin[dir]) / static_cast<Real>(nxa[dir]);
          Real c[3];
          for (int d = 0; d < 3; ++d) {
            if (d != dir) {
              Real dxl = (xmax[d] - xmin[d]) / static_cast<Real>(nxa[d]);
              c[d] = xmin[d] + (ijk[d]-ngh+0.5)*dxl;
            }
          }
          c[dir] = outer ? xmax[dir] - 0.5*dxd : xmin[dir] + 0.5*dxd;
          Real r_a = Kokkos::sqrt(SQR(c[0]) + SQR(c[1]) + SQR(c[2]));
          Real u_a = outer ? at(ngh+ncells-1) : at(ngh);
          for (int n = 0; n < ngh; ++n) {
            c[dir] = outer ? xmax[dir] + (0.5+n)*dxd : xmin[dir] - (0.5+n)*dxd;
            Real r_g = Kokkos::sqrt(SQR(c[0]) + SQR(c[1]) + SQR(c[2]));
            if (outer) at(ngh+ncells+n) = u_a * Kokkos::pow(r_a/(r_g+1.0e-30), rorder);
            else       at(ngh-1-n) = u_a * Kokkos::pow(r_a/(r_g+1.0e-30), rorder);
          }
        } else {
          for (int n = 0; n < ngh; ++n) {
            if (outer) at(ngh+ncells+n) = sign * at(ngh+ncells-1-n);
            else       at(ngh-1-n) = sign * at(ngh+n);
          }
        }
      });
  }

  return TaskStatus::complete;
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridTaskList::SetMGTaskListToFiner(int nsmooth, int ngh, int flag)
//! \brief Set the task list for prolongation and post smoothing

void MultigridDriver::SetMGTaskListToFiner(int nsmooth, int ngh, int flag) {
  auto &tl = pmy_pack_->tl_map;
  tl.erase("mg_to_finer");
  tl.emplace(std::make_pair("mg_to_finer", std::make_shared<TaskList>()));
  TaskID none(0);

  if (flag == 1) {
    // First time on meshblock levels: no boundary comm before prolongation
    id.prolongate = tl["mg_to_finer"]->AddTask(&MultigridDriver::Prolongate, this, none);
  } else {
    // Coarse-level boundary comm before prolongation
    id.fill_coarse0 = tl["mg_to_finer"]->AddTask(
                        &MultigridDriver::FillCoarseBoundary, this, none);
    id.ircv0    = tl["mg_to_finer"]->AddTask(&MultigridDriver::StartReceive, this, none);
    id.send0    = tl["mg_to_finer"]->AddTask(
        &MultigridDriver::SendBoundary, this, id.fill_coarse0);
    id.recv0    = tl["mg_to_finer"]->AddTask(
        &MultigridDriver::RecvBoundary, this, id.send0);
    id.physb0   = tl["mg_to_finer"]->AddTask(
        &MultigridDriver::PhysicalBoundary, this, id.recv0);
    id.fc_ghosts0 = tl["mg_to_finer"]->AddTask(
        &MultigridDriver::ProlongateFCBoundary, this, id.physb0);
    id.prolongate = tl["mg_to_finer"]->AddTask(
        &MultigridDriver::Prolongate, this, id.fc_ghosts0);
  }

  TaskID last_smooth = id.prolongate;
  if (nsmooth > 0) {
    // Fine-level boundary comm after prolongation
    id.fill_coarse1 = tl["mg_to_finer"]->AddTask(
                        &MultigridDriver::FillCoarseBoundary, this, id.prolongate);
    id.ircv1    = tl["mg_to_finer"]->AddTask(
        &MultigridDriver::StartReceive, this, id.prolongate);
    id.send1    = tl["mg_to_finer"]->AddTask(
        &MultigridDriver::SendBoundary, this, id.fill_coarse1);
    id.recv1    = tl["mg_to_finer"]->AddTask(
        &MultigridDriver::RecvBoundary, this, id.send1);
    id.physb1   = tl["mg_to_finer"]->AddTask(
        &MultigridDriver::PhysicalBoundary, this, id.recv1);
    id.fc_ghosts_prol = tl["mg_to_finer"]->AddTask(
                      &MultigridDriver::ProlongateFCBoundary, this, id.physb1);

    // Post-smoothing (red-black)
    id.smoothR   = tl["mg_to_finer"]->AddTask(
        &MultigridDriver::SmoothRed, this, id.fc_ghosts_prol);
    id.fill_coarseR = tl["mg_to_finer"]->AddTask(
        &MultigridDriver::FillCoarseBoundary, this, id.smoothR);
    id.ircvR    = tl["mg_to_finer"]->AddTask(
        &MultigridDriver::StartReceive, this, id.fc_ghosts_prol);
    id.sendR    = tl["mg_to_finer"]->AddTask(
        &MultigridDriver::SendBoundary, this, id.fill_coarseR);
    id.recvR    = tl["mg_to_finer"]->AddTask(
        &MultigridDriver::RecvBoundary, this, id.sendR);
    id.physbR   = tl["mg_to_finer"]->AddTask(
        &MultigridDriver::PhysicalBoundary, this, id.recvR);
    id.fc_ghostsR = tl["mg_to_finer"]->AddTask(
                      &MultigridDriver::ProlongateFCBoundary, this, id.physbR);

    id.smoothB   = tl["mg_to_finer"]->AddTask(
        &MultigridDriver::SmoothBlack, this, id.fc_ghostsR);
    last_smooth = id.smoothB;

    if (nsmooth > 1) {
      id.fill_coarseB = tl["mg_to_finer"]->AddTask(
          &MultigridDriver::FillCoarseBoundary, this, id.smoothB);
      id.ircvB    = tl["mg_to_finer"]->AddTask(
          &MultigridDriver::StartReceive, this, id.fc_ghostsR);
      id.sendB    = tl["mg_to_finer"]->AddTask(
          &MultigridDriver::SendBoundary, this, id.fill_coarseB);
      TaskID dep_recvB = id.ircvB | id.sendB;  // receives are posted in SendBoundary
      id.recvB    = tl["mg_to_finer"]->AddTask(
          &MultigridDriver::RecvBoundary, this, dep_recvB);
      id.physbB   = tl["mg_to_finer"]->AddTask(
          &MultigridDriver::PhysicalBoundary, this, id.recvB);
      id.fc_ghostsB = tl["mg_to_finer"]->AddTask(
                        &MultigridDriver::ProlongateFCBoundary, this, id.physbB);

      id.smoothR2   = tl["mg_to_finer"]->AddTask(
          &MultigridDriver::SmoothRed, this, id.fc_ghostsB);

      id.fill_coarseR2 = tl["mg_to_finer"]->AddTask(
          &MultigridDriver::FillCoarseBoundary, this, id.smoothR2);
      id.ircvR2    = tl["mg_to_finer"]->AddTask(
          &MultigridDriver::StartReceive, this, id.fc_ghostsB);
      id.sendR2    = tl["mg_to_finer"]->AddTask(
          &MultigridDriver::SendBoundary, this, id.fill_coarseR2);
      TaskID dep_recvR2 = id.ircvR2 | id.sendR2;  // receives are posted in SendBoundary
      id.recvR2    = tl["mg_to_finer"]->AddTask(
          &MultigridDriver::RecvBoundary, this, dep_recvR2);
      id.physbR2   = tl["mg_to_finer"]->AddTask(
          &MultigridDriver::PhysicalBoundary, this, id.recvR2);
      id.fc_ghostsR2 = tl["mg_to_finer"]->AddTask(
          &MultigridDriver::ProlongateFCBoundary, this, id.physbR2);

      id.smoothB2   = tl["mg_to_finer"]->AddTask(
          &MultigridDriver::SmoothBlack, this, id.fc_ghostsR2);
      last_smooth = id.smoothB2;
    }
  }

  // flag==2: last step to finest level -- add final boundary exchange so ghost cells
  // are up-to-date for CalculateDefectNorm (matching Athena++ MG_STARTRECVL etc.)
  if (flag == 2) {
    TaskID fillcL = tl["mg_to_finer"]->AddTask(
                      &MultigridDriver::FillCoarseBoundary, this, last_smooth);
    TaskID ircvL  = tl["mg_to_finer"]->AddTask(
                      &MultigridDriver::StartReceive, this, last_smooth);
    TaskID sendL  = tl["mg_to_finer"]->AddTask(
                      &MultigridDriver::SendBoundary, this, fillcL);
    TaskID recvL  = tl["mg_to_finer"]->AddTask(
                      &MultigridDriver::RecvBoundary, this, sendL);
    TaskID physL  = tl["mg_to_finer"]->AddTask(
                      &MultigridDriver::PhysicalBoundary, this, recvL);
    TaskID fcL    = tl["mg_to_finer"]->AddTask(
                      &MultigridDriver::ProlongateFCBoundary, this, physL);
    last_smooth = fcL;
  }

  id.clear_send0 = tl["mg_to_finer"]->AddTask(
      &MultigridDriver::ClearSend, this, last_smooth);
  id.clear_recv0 = tl["mg_to_finer"]->AddTask(
      &MultigridDriver::ClearRecv, this, id.clear_send0);
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SetMGTaskListFMGProlongate(int ngh, int flag)
//! \brief Set the task list for FMG prolongation only (no smoothing)

void MultigridDriver::SetMGTaskListFMGProlongate(int ngh, int flag) {
  auto &tl = pmy_pack_->tl_map;
  tl.erase("mg_fmg_prolongate");
  tl.emplace(std::make_pair("mg_fmg_prolongate", std::make_shared<TaskList>()));
  TaskID none(0);

  if (flag == 1) {
    // First time on meshblock levels, right after TransferFromRootToBlocks:
    // block data and ghosts were filled directly from the root/octet grids, and
    // the boundary machinery cannot operate at the 1-cell level (FillCoarseMG /
    // ProlongateFCMG no-op below 2 cells; sends would read a stale coarse_buf_).
    // Prolongate only, matching Athena++ SetMGTaskListFMGProlongate flag==1.
    id.fmg_prolongate = tl["mg_fmg_prolongate"]->AddTask(
                    &MultigridDriver::FMGProlongateTask, this, none);
    return;
  }

  id.fill_coarse0 = tl["mg_fmg_prolongate"]->AddTask(
                  &MultigridDriver::FillCoarseBoundary, this, none);
  id.ircv0    = tl["mg_fmg_prolongate"]->AddTask(
                  &MultigridDriver::StartReceive, this, none);
  id.send0    = tl["mg_fmg_prolongate"]->AddTask(
                  &MultigridDriver::SendBoundary, this, id.fill_coarse0);
  id.recv0    = tl["mg_fmg_prolongate"]->AddTask(
                  &MultigridDriver::RecvBoundary, this, id.send0);
  id.physb0   = tl["mg_fmg_prolongate"]->AddTask(
                  &MultigridDriver::PhysicalBoundary, this, id.recv0);
  id.fc_ghosts0 = tl["mg_fmg_prolongate"]->AddTask(
                  &MultigridDriver::ProlongateFCBoundary, this, id.physb0);

  id.fmg_prolongate = tl["mg_fmg_prolongate"]->AddTask(
                  &MultigridDriver::FMGProlongateTask, this, id.fc_ghosts0);

  id.clear_send0 = tl["mg_fmg_prolongate"]->AddTask(
                  &MultigridDriver::ClearSend, this, id.fmg_prolongate);
  id.clear_recv0 = tl["mg_fmg_prolongate"]->AddTask(
                  &MultigridDriver::ClearRecv, this, id.clear_send0);
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridTaskList::SetMGTaskListToCoarser(int nsmooth, int ngh)
//! \brief Set the task list for pre smoothing and restriction

void MultigridDriver::SetMGTaskListToCoarser(int nsmooth, int cycle) {
  auto &tl = pmy_pack_->tl_map;
  tl.erase("mg_to_coarser");
  tl.emplace(std::make_pair("mg_to_coarser",
      std::make_shared<TaskList>()));
  TaskID none(0);

  id.fill_coarse0 = tl["mg_to_coarser"]->AddTask(
      &MultigridDriver::FillCoarseBoundary, this, none);
  id.ircv0    = tl["mg_to_coarser"]->AddTask(
      &MultigridDriver::StartReceive, this, none);
  id.send0    = tl["mg_to_coarser"]->AddTask(
      &MultigridDriver::SendBoundary, this, id.fill_coarse0);
  id.recv0    = tl["mg_to_coarser"]->AddTask(
      &MultigridDriver::RecvBoundary, this, id.send0);
  id.physb0   = tl["mg_to_coarser"]->AddTask(
      &MultigridDriver::PhysicalBoundary, this, id.recv0);
  id.fc_ghosts0 = tl["mg_to_coarser"]->AddTask(
      &MultigridDriver::ProlongateFCBoundary, this, id.physb0);

  id.calc_rhs = tl["mg_to_coarser"]->AddTask(
      &MultigridDriver::CalculateFASRHS, this, id.fc_ghosts0);

  if (nsmooth > 0) {
    id.ircvR    = tl["mg_to_coarser"]->AddTask(
        &MultigridDriver::StartReceive, this, id.fc_ghosts0);
    id.smoothR  = tl["mg_to_coarser"]->AddTask(
        &MultigridDriver::SmoothRed, this, id.calc_rhs);
    id.fill_coarseR = tl["mg_to_coarser"]->AddTask(
        &MultigridDriver::FillCoarseBoundary, this, id.smoothR);
    id.sendR    = tl["mg_to_coarser"]->AddTask(
        &MultigridDriver::SendBoundary, this, id.fill_coarseR);
    id.recvR    = tl["mg_to_coarser"]->AddTask(
        &MultigridDriver::RecvBoundary, this, id.sendR);
    id.physbR   = tl["mg_to_coarser"]->AddTask(
        &MultigridDriver::PhysicalBoundary, this, id.recvR);
    id.fc_ghostsR = tl["mg_to_coarser"]->AddTask(
        &MultigridDriver::ProlongateFCBoundary, this, id.physbR);

    id.ircvB    = tl["mg_to_coarser"]->AddTask(
        &MultigridDriver::StartReceive, this, id.fc_ghostsR);
    id.smoothB  = tl["mg_to_coarser"]->AddTask(
        &MultigridDriver::SmoothBlack, this, id.fc_ghostsR);
    id.fill_coarseB = tl["mg_to_coarser"]->AddTask(
        &MultigridDriver::FillCoarseBoundary, this, id.smoothB);
    id.sendB    = tl["mg_to_coarser"]->AddTask(
        &MultigridDriver::SendBoundary, this, id.fill_coarseB);
    id.recvB    = tl["mg_to_coarser"]->AddTask(
        &MultigridDriver::RecvBoundary, this, id.sendB);
    id.physbB   = tl["mg_to_coarser"]->AddTask(
        &MultigridDriver::PhysicalBoundary, this, id.recvB);
    id.fc_ghostsB = tl["mg_to_coarser"]->AddTask(
        &MultigridDriver::ProlongateFCBoundary, this, id.physbB);
    if (nsmooth > 1) {
      id.ircvR2  = tl["mg_to_coarser"]->AddTask(
          &MultigridDriver::StartReceive, this, id.fc_ghostsB);
      id.smoothR2 = tl["mg_to_coarser"]->AddTask(
          &MultigridDriver::SmoothRed, this, id.fc_ghostsB);
      id.fill_coarseR2 = tl["mg_to_coarser"]->AddTask(
          &MultigridDriver::FillCoarseBoundary, this, id.smoothR2);
      id.sendR2  = tl["mg_to_coarser"]->AddTask(
          &MultigridDriver::SendBoundary, this, id.fill_coarseR2);
      id.recvR2  = tl["mg_to_coarser"]->AddTask(
          &MultigridDriver::RecvBoundary, this, id.sendR2);
      id.physbR2 = tl["mg_to_coarser"]->AddTask(
          &MultigridDriver::PhysicalBoundary, this, id.recvR2);
      id.fc_ghostsR2 = tl["mg_to_coarser"]->AddTask(
          &MultigridDriver::ProlongateFCBoundary, this, id.physbR2);

      id.ircvB2  = tl["mg_to_coarser"]->AddTask(
          &MultigridDriver::StartReceive, this, id.fc_ghostsR2);
      id.smoothB2 = tl["mg_to_coarser"]->AddTask(
          &MultigridDriver::SmoothBlack, this, id.fc_ghostsR2);
      id.fill_coarseB2 = tl["mg_to_coarser"]->AddTask(
          &MultigridDriver::FillCoarseBoundary, this, id.smoothB2);
      id.sendB2  = tl["mg_to_coarser"]->AddTask(
          &MultigridDriver::SendBoundary, this, id.fill_coarseB2);
      id.recvB2  = tl["mg_to_coarser"]->AddTask(
          &MultigridDriver::RecvBoundary, this, id.sendB2);
      id.physbB2 = tl["mg_to_coarser"]->AddTask(
          &MultigridDriver::PhysicalBoundary, this, id.recvB2);
      id.fc_ghostsB2 = tl["mg_to_coarser"]->AddTask(
          &MultigridDriver::ProlongateFCBoundary, this, id.physbB2);

      id.restrict_ = tl["mg_to_coarser"]->AddTask(
          &MultigridDriver::Restrict, this, id.fc_ghostsB2);
    } else {
      id.restrict_ = tl["mg_to_coarser"]->AddTask(
          &MultigridDriver::Restrict, this, id.fc_ghostsB);
    }
  } else {
    id.restrict_ = tl["mg_to_coarser"]->AddTask(
        &MultigridDriver::Restrict, this, id.calc_rhs);
  }
  id.clear_send0 = tl["mg_to_coarser"]->AddTask(
      &MultigridDriver::ClearSend, this, id.restrict_);
  id.clear_recv0 = tl["mg_to_coarser"]->AddTask(
      &MultigridDriver::ClearRecv, this, id.clear_send0);
}
