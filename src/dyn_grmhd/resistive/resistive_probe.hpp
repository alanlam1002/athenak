#ifndef DYN_GRMHD_RESISTIVE_RESISTIVE_PROBE_HPP_
#define DYN_GRMHD_RESISTIVE_RESISTIVE_PROBE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file resistive_probe.hpp
//! \brief Test-only full-grid array dumps for the K5-cold-z first-cycles diagnostic
//!        (07 v2.9 §3.2.4(e)). Enabled only by <mhd> resistive_probe_cycles = N > 0
//!        (read without adding the key, so default runs keep identical output headers);
//!        single rank only. Output only: no state is modified.
//!
//! File <basename>.probe.c<cycle>.s<stage>.<tag>.bin (cycle = Mesh::ncycle during the
//! stage): "ATKPROBE", int32 sizeof(Real), then per array: char[16] name, int32 rank,
//! int32 extents[rank], rank-ordered (LayoutRight) Real data.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"

namespace resistive {

// N from <block>/resistive_probe_cycles if present, else 0 (never adds the key).
inline int ProbeCycles(ParameterInput *pin, const std::string &block = "mhd") {
  if (!pin->DoesParameterExist(block, "resistive_probe_cycles")) return 0;
  const int n = pin->GetInteger(block, "resistive_probe_cycles");
  if (n > 0 && global_variable::nranks != 1) {
    std::cout << "### FATAL ERROR: <" << block << "> resistive_probe_cycles needs a "
              << "single MPI rank" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return (n > 0) ? n : 0;
}

class ProbeFile {
 public:
  ProbeFile(const std::string &base, MeshBlockPack *pp, const int stage,
            const std::string &tag) {
    Mesh *pm = pp->pmesh;
    const std::string fn = base + ".probe.c" + std::to_string(pm->ncycle) + ".s" +
                           std::to_string(stage) + "." + tag + ".bin";
    fp_ = std::fopen(fn.c_str(), "wb");
    if (fp_ == nullptr) {
      std::cout << "### FATAL ERROR: cannot open " << fn << std::endl;
      std::exit(EXIT_FAILURE);
    }
    std::fwrite("ATKPROBE", 1, 8, fp_);
    const int32_t rs = static_cast<int32_t>(sizeof(Real));
    std::fwrite(&rs, sizeof(rs), 1, fp_);
    // geometry: time, dt, ng, then (x1min, x1max, x2min, x2max, x3min, x3max) per block
    const int nmb = pp->nmb_thispack;
    HostArray2D<Real> geo("probe_geo", nmb + 1, 6);
    geo(0, 0) = pm->time;
    geo(0, 1) = pm->dt;
    geo(0, 2) = static_cast<Real>(pm->mb_indcs.ng);
    geo(0, 3) = geo(0, 4) = geo(0, 5) = 0.0;
    auto &sz = pp->pmb->mb_size;
    for (int m = 0; m < nmb; ++m) {
      geo(m + 1, 0) = sz.h_view(m).x1min;
      geo(m + 1, 1) = sz.h_view(m).x1max;
      geo(m + 1, 2) = sz.h_view(m).x2min;
      geo(m + 1, 3) = sz.h_view(m).x2max;
      geo(m + 1, 4) = sz.h_view(m).x3min;
      geo(m + 1, 5) = sz.h_view(m).x3max;
    }
    Add("geo", geo);
  }
  ~ProbeFile() { std::fclose(fp_); }

  template <class V>
  void Add(const char *name, const V &v) {
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
    char nm[16];
    std::memset(nm, 0, sizeof(nm));
    std::strncpy(nm, name, sizeof(nm) - 1);
    std::fwrite(nm, 1, sizeof(nm), fp_);
    const int32_t r = static_cast<int32_t>(V::rank);
    std::fwrite(&r, sizeof(r), 1, fp_);
    for (int d = 0; d < r; ++d) {
      const int32_t e = static_cast<int32_t>(h.extent(d));
      std::fwrite(&e, sizeof(e), 1, fp_);
    }
    std::fwrite(h.data(), sizeof(Real), h.size(), fp_);
  }

 private:
  FILE *fp_;
};

}  // namespace resistive

#endif  // DYN_GRMHD_RESISTIVE_RESISTIVE_PROBE_HPP_
