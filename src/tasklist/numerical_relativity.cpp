//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file numerical_relativity.cpp
//  \brief implementation of functions for NumericalRelativity
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "numerical_relativity.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "multigrid/multigrid.hpp"
#include "cfc/cfc.hpp"
#include "mhd/mhd.hpp"
#include "coordinates/adm.hpp"
#include "utils/chksum.hpp"

namespace numrel {

NumericalRelativity::NumericalRelativity(MeshBlockPack *ppack, ParameterInput *pin) {
  pmy_pack = ppack;
  chksum::ncycles = pin->GetOrAddInteger("job", "checksum_cycles", 0);
  chksum::mg_level = pin->GetOrAddInteger("job", "checksum_mg", 2);
  // <job> checksum_start: the first cycle to checksum (default: the first cycle run)
  chksum::cycle0 = pin->GetOrAddInteger("job", "checksum_start", -1);
  MultigridDriver::zero_hierarchy_ = pin->GetOrAddBoolean("job", "mg_zero_hierarchy", false);
}

//! <job> checksum_cycles (utils/chksum.hpp): this rank's checksums after a completed task
// debug (C-102): per-MeshBlock checksums (by gid; interior and ghosts separately) of
// u0, w0, adm, temperature under a label
void ChecksumPerBlock(MeshBlockPack *pp, const std::string &label) {
  const int cycle = pp->pmesh->ncycle;
  if (!chksum::Active(cycle) || chksum::fp == nullptr) return;
  auto &indcs = pp->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  auto one = [&](const char *name, const DvceArray5D<Real> &a) {
    if (a.span() == 0) return;
    const int nv = a.extent_int(1), n3 = a.extent_int(2), n2 = a.extent_int(3);
    const int n1 = a.extent_int(4);
    for (int m = 0; m < pp->nmb_thispack; ++m) {
      uint64_t si = 0, sg = 0;
      Kokkos::parallel_reduce("chk_blk", Kokkos::RangePolicy<>(DevExeSpace(), 0,
                              nv*n3*n2*n1),
      KOKKOS_LAMBDA(const int idx, uint64_t &ai, uint64_t &ag) {
        const int v = idx/(n3*n2*n1);
        const int k = (idx/(n2*n1))%n3, j = (idx/n1)%n2, i = idx%n1;
        const uint64_t b = Kokkos::bit_cast<uint64_t>(a(m, v, k, j, i));
        if (k >= ks && k <= ke && j >= js && j <= je && i >= is && i <= ie) ai += b;
        else ag += b;
      }, Kokkos::Sum<uint64_t>(si), Kokkos::Sum<uint64_t>(sg));
      std::fprintf(chksum::fp, "%d -1 %s %d blk_%s gid=%d lev=%d int=%016llx gh=%016llx\n",
                   cycle, label.c_str(), 0, name, pp->pmb->mb_gid.h_view(m),
                   pp->pmb->mb_lev.h_view(m), (unsigned long long)si,
                   (unsigned long long)sg);
    }
  };
  if (pp->pmhd != nullptr) { one("u0", pp->pmhd->u0); one("w0", pp->pmhd->w0); }
  if (pp->padm != nullptr) one("adm", pp->padm->u_adm);
  if (pp->pdyngr != nullptr) one("temp", pp->pdyngr->temperature);
  std::fflush(chksum::fp);
}

void ChecksumAfterTask(MeshBlockPack *pp, const std::string &task, int stage) {
  const int cycle = pp->pmesh->ncycle;
  if (!chksum::Active(cycle)) return;
  chksum::nmb_active = pp->nmb_thispack;
  if (chksum::fp == nullptr) {
    char fn[64];
    std::snprintf(fn, sizeof(fn), "chksum.%d.txt", global_variable::my_rank);
    chksum::fp = std::fopen(fn, "w");
  }
  if (cycle != chksum::last_cycle || stage != chksum::last_stage) {
    chksum::occ.clear();
    chksum::last_cycle = cycle; chksum::last_stage = stage;
  }
  const int k = chksum::occ[task]++;
  auto *fp = chksum::fp;
  std::fprintf(fp, "%d %d %s %d host %016llx %016llx %016llx %016llx %016llx %016llx "
               "%016llx %016llx\n", cycle, stage, task.c_str(), k,
               (unsigned long long)chksum::Bits(pp->pmesh->dt),
               (unsigned long long)chksum::Bits(pp->pmesh->time),
               (unsigned long long)chksum::Bits(pp->padm ? pp->padm->frame_vel_u[0] : 0.0),
               (unsigned long long)chksum::Bits(pp->padm ? pp->padm->frame_vel_u[1] : 0.0),
               (unsigned long long)chksum::Bits(pp->padm ? pp->padm->frame_vel_u[2] : 0.0),
               (unsigned long long)chksum::Bits(pp->padm ? pp->padm->bc_center[0] : 0.0),
               (unsigned long long)chksum::Bits(pp->padm ? pp->padm->bc_center[1] : 0.0),
               (unsigned long long)chksum::Bits(pp->padm ? pp->padm->bc_center[2] : 0.0));
  if (pp->pmhd != nullptr) {
    auto *m = pp->pmhd;
    chksum::Line(cycle, stage, task, k, "u0", m->u0);
    chksum::Line(cycle, stage, task, k, "u1", m->u1);
    chksum::Line(cycle, stage, task, k, "w0", m->w0);
    chksum::Line(cycle, stage, task, k, "flx1", m->uflx.x1f);
    chksum::Line(cycle, stage, task, k, "flx2", m->uflx.x2f);
    chksum::Line(cycle, stage, task, k, "flx3", m->uflx.x3f);
    chksum::Line(cycle, stage, task, k, "fofc", m->fofc);
  }
  if (pp->pdyngr != nullptr) {
    chksum::Line(cycle, stage, task, k, "temp", pp->pdyngr->temperature);
  }
  if (pp->padm != nullptr) chksum::Line(cycle, stage, task, k, "adm", pp->padm->u_adm);
  if (pp->pcfc != nullptr) {
    auto *c = pp->pcfc;
    chksum::Line(cycle, stage, task, k, "dpsi", c->delta_psi);
    chksum::Line(cycle, stage, task, k, "dapsi", c->delta_alpha_psi);
    chksum::Line(cycle, stage, task, k, "cbeta", c->u_beta);
    chksum::Line(cycle, stage, task, k, "adual", c->u_adual);
    chksum::Line(cycle, stage, task, k, "utilde", c->u_tilde);
    chksum::Line(cycle, stage, task, k, "upx", c->u_p_x);
    chksum::Line(cycle, stage, task, k, "upbeta", c->u_p_beta);
    chksum::Line(cycle, stage, task, k, "upsrc", c->u_p_src);
    chksum::Line(cycle, stage, task, k, "a0dual", c->u_a0dual);
    chksum::Line(cycle, stage, task, k, "psi0", c->u_psi0);
    std::fprintf(fp, "%d %d %s %d mbh %016llx\n", cycle, stage, task.c_str(), k,
                 (unsigned long long)chksum::Bits(c->puncture_mass_));
  }
  std::fflush(fp);
}

std::vector<QueuedTask>& NumericalRelativity::SelectQueue(TaskLocation loc) {
  switch(loc) {
    case Task_Start:
      return start_queue;
    case Task_Run:
      return run_queue;
    case Task_End:
      return end_queue;
    default:
      std::cout << "NumericalRelativity: Unknown task queue requested!\n";
      abort();
  }

  return end_queue;
}

PhysicsDependency NumericalRelativity::NeedsPhysics(TaskName task) {
  if (task < MHD_NTASKS) {
    return Phys_MHD;
  } else if (task < Z4c_NTASKS) {
    return Phys_Z4c;
  } else if (task < CFC_NTASKS) {
    return Phys_CFC;
  } else {
    return Phys_None;
  }
}

bool NumericalRelativity::DependencyAvailable(PhysicsDependency dep) {
  switch(dep) {
    case Phys_None:
      return true;
    case Phys_MHD:
      return pmy_pack->pdyngr != nullptr;
    case Phys_Z4c:
      return pmy_pack->pz4c != nullptr;
    case Phys_CFC:
      return pmy_pack->pcfc != nullptr;
    default:
      std::cout << "NumericalRelativity: Unknown dependency\n";
  }
  return false;
}

bool NumericalRelativity::DependenciesMet(std::vector<TaskName>& tasks) {
  for (auto& task : tasks) {
    PhysicsDependency phys = NeedsPhysics(task);
    if (!DependencyAvailable(phys)) {
      return false;
    }
  }

  return true;
}

bool NumericalRelativity::DependenciesMet(QueuedTask& task,
                                          std::vector<QueuedTask>& queue,
                                          TaskID& dependencies) {
  // Loop through all the dependencies, then compare them to each task in the queue.
  // If a dependency hasn't been added to the task list, return false. Otherwise, add its
  // id to the dependency id. If the dependency simply doesn't exist in the list, always
  // return false.
  for (auto& dep : task.dependencies) {
    bool found = false;
    for (auto &test_task : queue) {
      if (test_task.name == dep) {
        found = true;
        if (!test_task.added) {
          return false;
        }
        dependencies = dependencies | test_task.id;
      }
    }
    // This handles the corner case where the dependency isn't in the queue.
    if (!found) {
      return false;
    }
  }

  return true;
}

void NumericalRelativity::AddExtraDependencies(std::vector<TaskName>& required,
                                               std::vector<TaskName>& optional) {
  for (auto& task : optional) {
    PhysicsDependency phys = NeedsPhysics(task);
    if (DependencyAvailable(phys)) {
      required.push_back(task);
    }
  }
}

bool NumericalRelativity::AssembleNumericalRelativityTasks(
    std::shared_ptr<TaskList>& list, std::vector<QueuedTask> &queue) {
  int added = 0;
  int size = queue.size();
  while (added < size) {
    int cycle_added = 0;
    for (auto &task : queue) {
      TaskID dep(0);
      if (DependenciesMet(task, queue, dep) && !task.added) {
        task.added = true;
        if (chksum::ncycles > 0) {
          auto f = task.func_;
          const std::string nm = task.name_string;
          MeshBlockPack *pp = pmy_pack;
          task.id = list->AddTask([f, nm, pp](Driver *d, int s) mutable -> TaskStatus {
            TaskStatus st = f(d, s);
            if (st == TaskStatus::complete) ChecksumAfterTask(pp, nm, s);
            return st;
          }, dep);
        } else {
          task.id = list->AddTask(task.func_, dep);
        }
        cycle_added++;
        added++;
        /*std::cout << "Successfully added " << task.name_string << " to task list!\n"
                  << "  ID: ";
        task.id.PrintID();
        std::cout << "  Dependencies: ";
        dep.PrintID();*/
      }
    }
    if (cycle_added == 0) {
      return false;
    }
  }

  return true;
}

void NumericalRelativity::PrintMissingTasks(std::vector<QueuedTask> &queue) {
  std::cout << "Successfully added the following tasks:\n";
  for (auto& task : queue) {
    if (task.added) {
      std::cout << "  " << task.name_string << "\n";
    }
  }
  std::cout << "Could not add the following tasks:\n";
  for (auto& task : queue) {
    if (!task.added) {
      std::cout << "  " << task.name_string << "\n";
    }
  }
}

void NumericalRelativity::AssembleNumericalRelativityTasks(
       std::map<std::string, std::shared_ptr<TaskList>>& tl) {
  // Assemble the task lists for all physics modules
  if (pmy_pack->pdyngr != nullptr) {
    pmy_pack->pdyngr->QueueDynGRMHDTasks();
  }
  if (pmy_pack->pz4c != nullptr) {
    pmy_pack->pz4c->QueueZ4cTasks();
  }
  if (pmy_pack->pcfc != nullptr) {
    pmy_pack->pcfc->QueueCFCTasks();
  }

  bool success = AssembleNumericalRelativityTasks(tl["before_stagen"], start_queue);
  if (!success) {
    std::cout << "NumericalRelativity: Failed to construct start TaskList!\n"
              << "  Check that there are no cyclical dependencies or missing tasks.\n";
    PrintMissingTasks(start_queue);
    abort();
  }

  success = AssembleNumericalRelativityTasks(tl["stagen"], run_queue);
  if (!success) {
    std::cout << "NumericalRelativity: Failed to construct run TaskList!\n"
              << "  Check that there are no cyclical dependencies or missing tasks.\n";
    PrintMissingTasks(run_queue);
    abort();
  }

  success = AssembleNumericalRelativityTasks(tl["after_stagen"], end_queue);
  if (!success) {
    std::cout << "NumericalRelativity: Failed to construct end TaskList!\n"
              << "  Check that there are no cyclical dependencies or missing tasks.\n";
    PrintMissingTasks(end_queue);
    abort();
  }
}

} // namespace numrel
