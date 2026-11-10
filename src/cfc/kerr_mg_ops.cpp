//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file kerr_mg_ops.cpp
//! \brief W3a host drivers and device kernels for the Kerr multigrid building blocks
//! (see kerr_mg_ops.hpp).

#include <limits>
#include <vector>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "mesh/nghbr_index.hpp"
#include "kerr_mg_ops.hpp"

namespace cfc {
namespace kmg {


void ComputeBlockEdgeMasks(MeshBlockPack *pmbp, DualArray1D<int> &mask,
                           DualArray1D<int> &phys) {
  const int nmb = pmbp->nmb_thispack;
  Kokkos::realloc(mask, nmb);
  Kokkos::realloc(phys, nmb);
  auto &nghbr = pmbp->pmb->nghbr;
  auto &mblev = pmbp->pmb->mb_lev;
  auto &bcs = pmbp->pmb->mb_bcs;
  for (int m = 0; m < nmb; ++m) {
    int bits = 0, pbits = 0;
    const int own = mblev.h_view(m);
    for (int q = 0; q < 12; ++q) {
      int e, a, b, sa, sb;
      EdgeAxes(q, e, a, b, sa, sb);
      int o[3];
      o[e] = 0;
      o[a] = sa;
      o[b] = sb;
      const int n0 = NeighborIndex(o[0], o[1], o[2], 0, 0);
      const int n1 = NeighborIndex(o[0], o[1], o[2], 1, 0);
      const auto &s0 = nghbr.h_view(m, n0);
      const auto &s1 = nghbr.h_view(m, n1);
      const bool safe = (s0.gid >= 0) && (s0.lev == own) &&
                        ((s1.gid < 0) || (s1.lev == own));
      if (!safe) {
        bits |= (1 << q);
        const BoundaryFlag fa = bcs.h_view(m, 2*a + (sa > 0 ? 1 : 0));
        const BoundaryFlag fb = bcs.h_view(m, 2*b + (sb > 0 ? 1 : 0));
        if ((fa != BoundaryFlag::block && fa != BoundaryFlag::periodic) ||
            (fb != BoundaryFlag::block && fb != BoundaryFlag::periodic)) {
          pbits |= (1 << q);
        }
      }
    }
    mask.h_view(m) = bits;
    phys.h_view(m) = pbits;
  }
  mask.template modify<HostExeSpace>();
  mask.template sync<DevExeSpace>();
  phys.template modify<HostExeSpace>();
  phys.template sync<DevExeSpace>();
}

void ComputeRootEdgeMask(const BoundaryFlag bcs[6], DualArray1D<int> &mask,
                         DualArray1D<int> &phys) {
  Kokkos::realloc(mask, 1);
  Kokkos::realloc(phys, 1);
  int bits = 0;
  for (int q = 0; q < 12; ++q) {
    int e, a, b, sa, sb;
    EdgeAxes(q, e, a, b, sa, sb);
    const BoundaryFlag fa = bcs[2*a + (sa > 0 ? 1 : 0)];
    const BoundaryFlag fb = bcs[2*b + (sb > 0 ? 1 : 0)];
    const bool partial = (fa == BoundaryFlag::mg_robin || fa == BoundaryFlag::mg_multipole ||
                          fb == BoundaryFlag::mg_robin || fb == BoundaryFlag::mg_multipole);
    if (partial) { bits |= (1 << q); }
  }
  mask.h_view(0) = bits;
  phys.h_view(0) = bits;
  mask.template modify<HostExeSpace>();
  mask.template sync<DevExeSpace>();
  phys.template modify<HostExeSpace>();
  phys.template sync<DevExeSpace>();
}

void CountEdges(const DualArray1D<int> &mask, const DualArray1D<int> &phys, int nmb,
                long counts[3]) {
  counts[0] = counts[1] = counts[2] = 0;
  for (int m = 0; m < nmb; ++m) {
    for (int q = 0; q < 12; ++q) {
      if (!((mask.h_view(m) >> q) & 1)) continue;
      ++counts[0];
      if ((phys.h_view(m) >> q) & 1) { ++counts[1]; } else { ++counts[2]; }
    }
  }
}

void ApplyEdgeFill(const DvceArray5D<Real> &u, const DvceArray1D<int> &mask, int nmmb,
                   int nvar, int ngh, int nx, int ny, int nz, EdgeFill mode, bool poison) {
  if (mode == EdgeFill::off && !poison) return;
  const int nmax = (nx > ny) ? ((nx > nz) ? nx : nz) : ((ny > nz) ? ny : nz);
  const Real nan = std::numeric_limits<Real>::quiet_NaN();
  const int imode = static_cast<int>(mode);
  par_for("kmg::ApplyEdgeFill", DevExeSpace(), 0, nmmb-1, 0, nvar-1, 0, 11, 0, nmax-1,
  KOKKOS_LAMBDA(const int m, const int v, const int q, const int t) {
    if (!((mask(m) >> q) & 1)) return;
    int e, a, b, sa, sb;
    EdgeAxes(q, e, a, b, sa, sb);
    const int n[3] = {nx, ny, nz};
    if (t >= n[e]) return;
    const int ca = (sa > 0) ? ngh + n[a] - 1 : ngh;
    const int cb = (sb > 0) ? ngh + n[b] - 1 : ngh;
    const int ga = ca + sa, gb = cb + sb;
    const int da = -sa, db = -sb;
    int I[3];
    I[e] = ngh + t;
    I[a] = ga;
    I[b] = gb;
    if (poison) { u(m, v, I[2], I[1], I[0]) = nan; }
    if (imode == static_cast<int>(EdgeFill::off)) return;
    auto at = [&](int ia, int ib) -> Real {
      int J[3];
      J[e] = I[e];
      J[a] = ia;
      J[b] = ib;
      return u(m, v, J[2], J[1], J[0]);
    };
    const Real u00 = at(ca, cb);
    Real g = at(ga, cb) + at(ca, gb) - u00;
    if (imode == static_cast<int>(EdgeFill::rule_r) && n[a] > 1 && n[b] > 1) {
      g += at(ca + da, cb + db) - at(ca + da, cb) - at(ca, cb + db) + u00;
    }
    u(m, v, I[2], I[1], I[0]) = g;
  });
}

void PoisonCorners(const DvceArray5D<Real> &u, int nmmb, int nvar, int ngh, int nx, int ny,
                   int nz) {
  const Real nan = std::numeric_limits<Real>::quiet_NaN();
  par_for("kmg::PoisonCorners", DevExeSpace(), 0, nmmb-1, 0, nvar-1, 0, 7,
  KOKKOS_LAMBDA(const int m, const int v, const int c) {
    const int i = (c & 1) ? ngh + nx : ngh - 1;
    const int j = (c & 2) ? ngh + ny : ngh - 1;
    const int k = (c & 4) ? ngh + nz : ngh - 1;
    u(m, v, k, j, i) = nan;
  });
}

void FillLevelCoeffs(const DvceArray5D<Real> &coeff, const DualArray1D<RegionSize> &blk,
                     int nmmb, int ngh, int nx, int ny, int nz, const CoefSource &cs,
                     const Manuf &mf) {
  auto bd = blk.d_view;
  const int ncx = nx + 2*ngh, ncy = ny + 2*ngh, ncz = nz + 2*ngh;
  par_for("kmg::FillLevelCoeffs", DevExeSpace(), 0, nmmb-1, 0, ncz-1, 0, ncy-1, 0, ncx-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real dx[3] = {(bd(m).x1max - bd(m).x1min)/static_cast<Real>(nx),
                        (bd(m).x2max - bd(m).x2min)/static_cast<Real>(ny),
                        (bd(m).x3max - bd(m).x3min)/static_cast<Real>(nz)};
    const Real lo[3] = {bd(m).x1min + static_cast<Real>(i - ngh)*dx[0],
                        bd(m).x2min + static_cast<Real>(j - ngh)*dx[1],
                        bd(m).x3min + static_cast<Real>(k - ngh)*dx[2]};
    const Real xc[3] = {lo[0] + 0.5*dx[0], lo[1] + 0.5*dx[1], lo[2] + 0.5*dx[2]};
    for (int d = 0; d < 3; ++d) {
      Real p[3] = {xc[0], xc[1], xc[2]};
      p[d] = lo[d];
      Real gt[6], gtu[6], dgt[3][6];
      PointMetric(cs, p[0], p[1], p[2], gt, gtu, dgt);
      for (int s = 0; s < 6; ++s) { coeff(m, kFace + 6*d + s, k, j, i) = gtu[s]; }
    }
    coeff(m, kChC, k, j, i) = mf.c;
    coeff(m, kChS, k, j, i) = ManufSource(cs, mf, xc[0], xc[1], xc[2]);
  });
}

void FillManufU(const DvceArray5D<Real> &u, const DualArray1D<RegionSize> &blk, int nmmb,
                int ngh, int nx, int ny, int nz, const Manuf &mf) {
  auto bd = blk.d_view;
  const int ncx = nx + 2*ngh, ncy = ny + 2*ngh, ncz = nz + 2*ngh;
  par_for("kmg::FillManufU", DevExeSpace(), 0, nmmb-1, 0, ncz-1, 0, ncy-1, 0, ncx-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real dx1 = (bd(m).x1max - bd(m).x1min)/static_cast<Real>(nx);
    const Real dx2 = (bd(m).x2max - bd(m).x2min)/static_cast<Real>(ny);
    const Real dx3 = (bd(m).x3max - bd(m).x3min)/static_cast<Real>(nz);
    const Real x = bd(m).x1min + (static_cast<Real>(i - ngh) + 0.5)*dx1;
    const Real y = bd(m).x2min + (static_cast<Real>(j - ngh) + 0.5)*dx2;
    const Real z = bd(m).x3min + (static_cast<Real>(k - ngh) + 0.5)*dx3;
    Real uu, du[3], ddu[6];
    ManufEval(mf, x, y, z, uu, du, ddu);
    u(m, 0, k, j, i) = uu;
  });
}

long CountNonFinite(const DvceArray5D<Real> &a, int nmmb, int nvar, int ngh, int nx,
                    int ny, int nz) {
  const long n = static_cast<long>(nmmb)*nvar*nz*ny*nx;
  long cnt = 0;
  Kokkos::parallel_reduce("kmg::CountNonFinite",
                          Kokkos::RangePolicy<DevExeSpace>(0, n),
  KOKKOS_LAMBDA(const long idx, long &c) {
    long r = idx;
    const int i = static_cast<int>(r % nx); r /= nx;
    const int j = static_cast<int>(r % ny); r /= ny;
    const int k = static_cast<int>(r % nz); r /= nz;
    const int v = static_cast<int>(r % nvar); r /= nvar;
    const int m = static_cast<int>(r);
    if (NonFinite(a(m, v, k + ngh, j + ngh, i + ngh))) { c += 1; }
  }, cnt);
  return cnt;
}

}  // namespace kmg
}  // namespace cfc
