#ifndef UTILS_CHKSUM_HPP_
#define UTILS_CHKSUM_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file chksum.hpp
//! \brief <job> checksum_cycles (default 0; research R-084 Q3 / R-087 item 10, a debug
//! diagnostic): for the first N cycles after (re)start, after every completed task of the
//! numerical-relativity task lists (MHD + CFC; NumericalRelativity::Assemble...), each rank
//! appends to chksum.<rank>.txt one line per array: cycle stage task occurrence array sum.
//! The sum is the 64-bit wrap-around sum of the arrays' element bit patterns over the whole
//! allocation (ghosts included): exact and independent of the reduction order, so two
//! bitwise-identical states give identical sums, and the first task whose sums differ
//! between two runs locates the first non-deterministic kernel. No MPI: a collective inside
//! a task would deadlock (tasks complete in different orders on different ranks).
//! Host scalars (dt, time, the comoving frame's velocity and centre) are written as their
//! bit patterns on each line set's "host" row.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>

#include "athena.hpp"

namespace chksum {

inline int ncycles = 0;      // <job> checksum_cycles
// <job> checksum_mg: 0 no MG checksums, 1 MG solve points (start, each V-cycle, level steps,
// coarsest, multipole), 2 also after every MG task-list task (default; each is a fence)
inline int mg_level = 2;
inline int cycle0 = -1;      // the first cycle checksummed (<job> checksum_start, else the
                             // first cycle seen)
inline FILE *fp = nullptr;
inline int last_cycle = -1, last_stage = -1;
inline std::map<std::string, int> occ;   // task occurrences within (cycle, stage)

inline bool Active(int cycle) {
  if (ncycles <= 0) return false;
  if (cycle0 < 0) cycle0 = cycle;
  return cycle >= cycle0 && cycle < cycle0 + ncycles;
}

template <typename T>
inline uint64_t Sum(const T *p, size_t n) {
  uint64_t s = 0;
  Kokkos::parallel_reduce("chksum", Kokkos::RangePolicy<>(DevExeSpace(), 0, n),
  KOKKOS_LAMBDA(const size_t i, uint64_t &a) {
    if constexpr (sizeof(T) == 8) {
      a += Kokkos::bit_cast<uint64_t>(p[i]);
    } else {
      a += static_cast<uint64_t>(p[i]);
    }
  }, Kokkos::Sum<uint64_t>(s));
  return s;
}

// the active MeshBlocks (C-102): arrays whose first extent is the MeshBlock index are
// summed over their first nmb_active blocks only (allocations padded to a historical
// maximum otherwise differ between a continuous and a restarted run); others in full
inline int nmb_active = -1;
template <typename V>
inline void Line(int cycle, int stage, const std::string &task, int k, const char *name,
                 const V &v) {
  if (v.span() == 0) return;
  size_t n = v.span();
  if (nmb_active > 0 && V::rank >= 4 && v.extent(0) >= static_cast<size_t>(nmb_active)) {
    n = (n/v.extent(0))*static_cast<size_t>(nmb_active);
  }
  std::fprintf(fp, "%d %d %s %d %s %016llx\n", cycle, stage, task.c_str(), k, name,
               static_cast<unsigned long long>(Sum(v.data(), n)));
}

inline uint64_t Bits(double x) { uint64_t b; std::memcpy(&b, &x, 8); return b; }

}  // namespace chksum

#endif  // UTILS_CHKSUM_HPP_
