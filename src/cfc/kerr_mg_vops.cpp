//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file kerr_mg_vops.cpp
//! \brief W3c device kernels for the Kerr vector multigrid (see kerr_mg_vops.hpp).

#include <algorithm>
#include <cmath>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "kerr_mg_ops.hpp"
#include "kerr_mg_vops.hpp"

namespace cfc {
namespace kmg {

namespace {
//! the cell width and lower corner of cell (k, j, i) (ghost-inclusive indices)
KOKKOS_INLINE_FUNCTION
void CellGeom(const RegionSize &b, int ngh, int nx, int ny, int nz, int k, int j, int i,
              Real dx[3], Real lo[3]) {
  dx[0] = (b.x1max - b.x1min)/static_cast<Real>(nx);
  dx[1] = (b.x2max - b.x2min)/static_cast<Real>(ny);
  dx[2] = (b.x3max - b.x3min)/static_cast<Real>(nz);
  lo[0] = b.x1min + static_cast<Real>(i - ngh)*dx[0];
  lo[1] = b.x2min + static_cast<Real>(j - ngh)*dx[1];
  lo[2] = b.x3min + static_cast<Real>(k - ngh)*dx[2];
}
}  // namespace

long FillLevelVecCoeffs(const DvceArray5D<Real> &coeff, const DualArray1D<RegionSize> &blk,
                        int nmmb, int ngh, int nx, int ny, int nz, const CoefSource &cs) {
  auto bd = blk.d_view;
  const int ncx = nx + 2*ngh, ncy = ny + 2*ngh, ncz = nz + 2*ngh;
  par_for("kmg::FillLevelVecCoeffs", DevExeSpace(), 0, nmmb-1, 0, ncz-1, 0, ncy-1, 0,
          ncx-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real dx[3], lo[3];
    CellGeom(bd(m), ngh, nx, ny, nz, k, j, i, dx, lo);
    const Real xc[3] = {lo[0] + 0.5*dx[0], lo[1] + 0.5*dx[1], lo[2] + 0.5*dx[2]};
    Real gt[6], gtu[6], dgt[3][6], dgtu[3][6];
    for (int d = 0; d < 3; ++d) {
      Real p[3] = {xc[0], xc[1], xc[2]};
      p[d] = lo[d];
      PointMetric(cs, p[0], p[1], p[2], gt, gtu, dgt);
      PointDgtu(gtu, dgt, dgtu);
      const int base = kVFace + kVFaceStride*d;
      for (int q = 0; q < 6; ++q) {
        coeff(m, base + q, k, j, i) = gtu[q];
        for (int t = 0; t < 3; ++t) { coeff(m, base + 6 + 6*t + q, k, j, i) = dgtu[t][q]; }
      }
    }
    PointMetric(cs, xc[0], xc[1], xc[2], gt, gtu, dgt);
    Real gam[3][6];
    ChristoffelT(gtu, dgt, gam);
    for (int a = 0; a < 3; ++a) {
      for (int q = 0; q < 6; ++q) { coeff(m, kVGam + 6*a + q, k, j, i) = gam[a][q]; }
    }
  });
  const long n = static_cast<long>(nmmb)*nz*ny*nx;
  long bad = 0;
  Kokkos::parallel_reduce("kmg::FillLevelVecDinv", Kokkos::RangePolicy<DevExeSpace>(0, n),
  KOKKOS_LAMBDA(const long idx, long &cnt) {
    long r = idx;
    const int i = static_cast<int>(r % nx) + ngh; r /= nx;
    const int j = static_cast<int>(r % ny) + ngh; r /= ny;
    const int k = static_cast<int>(r % nz) + ngh; r /= nz;
    const int m = static_cast<int>(r);
    const Real h = (bd(m).x1max - bd(m).x1min)/static_cast<Real>(nx);
    VViewAcc<DvceArray5D<Real>> A{coeff, coeff, m};
    Real Di[9];
    const Real det = VecDinv(A, k, j, i, h, Di);
    if (!(det > 0.0) || NonFinite(det)) { cnt += 1; }
    for (int q = 0; q < 9; ++q) { coeff(m, kVDinv + q, k, j, i) = Di[q]; }
  }, bad);
  return bad;
}

void FillManufV(const DvceArray5D<Real> &u, const DualArray1D<RegionSize> &blk, int nmmb,
                int ngh, int nx, int ny, int nz, const VManuf &vm) {
  auto bd = blk.d_view;
  const int ncx = nx + 2*ngh, ncy = ny + 2*ngh, ncz = nz + 2*ngh;
  par_for("kmg::FillManufV", DevExeSpace(), 0, nmmb-1, 0, ncz-1, 0, ncy-1, 0, ncx-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real dx[3], lo[3];
    CellGeom(bd(m), ngh, nx, ny, nz, k, j, i, dx, lo);
    for (int c = 0; c < 3; ++c) {
      Real uu, du[3], ddu[6];
      ManufEval(vm.c[c], lo[0] + 0.5*dx[0], lo[1] + 0.5*dx[1], lo[2] + 0.5*dx[2], uu, du,
                ddu);
      u(m, c, k, j, i) = uu;
    }
  });
}

void FillManufVecSrc(const DvceArray5D<Real> &src, const DualArray1D<RegionSize> &blk,
                     int nmmb, int ngh, int nx, int ny, int nz, const CoefSource &cs,
                     const VManuf &vm) {
  auto bd = blk.d_view;
  const int ncx = nx + 2*ngh, ncy = ny + 2*ngh, ncz = nz + 2*ngh;
  par_for("kmg::FillManufVecSrc", DevExeSpace(), 0, nmmb-1, 0, ncz-1, 0, ncy-1, 0, ncx-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real dx[3], lo[3];
    CellGeom(bd(m), ngh, nx, ny, nz, k, j, i, dx, lo);
    Real s[3];
    ManufVecSource(cs, vm, lo[0] + 0.5*dx[0], lo[1] + 0.5*dx[1], lo[2] + 0.5*dx[2], s);
    for (int c = 0; c < 3; ++c) { src(m, c, k, j, i) = s[c]; }
  });
}

void KelvinFaceFill(const DvceArray5D<Real> &u, const DualArray1D<RegionSize> &blk,
                    const DvceArray1D<int> &faces, int nmmb, int ngh, int nx, int ny,
                    int nz, const Real x0[3], const VMoments &mom) {
  auto bd = blk.d_view;
  const int nmax = std::max(nx, std::max(ny, nz));
  const Real ox = x0[0], oy = x0[1], oz = x0[2];
  par_for("kmg::KelvinFaceFill", DevExeSpace(), 0, nmmb-1, 0, 5, 0, nmax-1, 0, nmax-1,
  KOKKOS_LAMBDA(const int m, const int f, const int tb, const int ta) {
    if (!((faces(m) >> f) & 1)) return;
    const int d = f >> 1, side = f & 1;
    const int a = (d + 1) % 3, b = (d + 2) % 3;
    const int n[3] = {nx, ny, nz};
    if (ta >= n[a] || tb >= n[b]) return;
    const RegionSize &r = bd(m);
    const Real mn[3] = {r.x1min, r.x2min, r.x3min};
    const Real mx[3] = {r.x1max, r.x2max, r.x3max};
    Real X[3];
    X[d] = (side == 0) ? mn[d] : mx[d];
    X[a] = mn[a] + (static_cast<Real>(ta) + 0.5)*(mx[a] - mn[a])/static_cast<Real>(n[a]);
    X[b] = mn[b] + (static_cast<Real>(tb) + 0.5)*(mx[b] - mn[b])/static_cast<Real>(n[b]);
    Real V[3];
    KelvinV(mom, X[0] - ox, X[1] - oy, X[2] - oz, V);
    for (int g = 0; g < ngh; ++g) {
      int I[3], J[3];   // ghost and mirror (i, j, k)
      I[a] = J[a] = ngh + ta;
      I[b] = J[b] = ngh + tb;
      if (side == 0) {
        I[d] = ngh - 1 - g;
        J[d] = ngh + g;
      } else {
        I[d] = ngh + n[d] + g;
        J[d] = ngh + n[d] - 1 - g;
      }
      for (int c = 0; c < 3; ++c) {
        u(m, c, I[2], I[1], I[0]) = 2.0*V[c] - u(m, c, J[2], J[1], J[0]);
      }
    }
  });
}

void LocalMoments(const DvceArray5D<Real> &u, const DvceArray5D<Real> &src,
                  const DvceArray5D<Real> &coeff, const DualArray1D<RegionSize> &blk,
                  int nmmb, int ngh, int nx, int ny, int nz, bool self, const Real x0[3],
                  Real mom[kNMom]) {
  auto bd = blk.d_view;
  const Real ox = x0[0], oy = x0[1], oz = x0[2];
  DvceArray2D<Real> part("kmg_mom_part", nmmb*nz, kNMom);
  par_for("kmg::LocalMoments", DevExeSpace(), 0, nmmb-1, 0, nz-1,
  KOKKOS_LAMBDA(const int m, const int kk) {
    const RegionSize &r = bd(m);
    const Real dx1 = (r.x1max - r.x1min)/static_cast<Real>(nx);
    const Real dx2 = (r.x2max - r.x2min)/static_cast<Real>(ny);
    const Real dx3 = (r.x3max - r.x3min)/static_cast<Real>(nz);
    const Real dv = dx1*dx2*dx3, h = dx1, ih2 = 1.0/(h*h);
    Real acc[kNMom];
    for (int q = 0; q < kNMom; ++q) { acc[q] = 0.0; }
    const int k = kk + ngh;
    const Real y2 = r.x3min + (static_cast<Real>(kk) + 0.5)*dx3 - oz;
    VViewAcc<DvceArray5D<Real>> A{u, coeff, m};
    FlatVAcc<VViewAcc<DvceArray5D<Real>>> F{A};
    for (int jj = 0; jj < ny; ++jj) {
      const int j = jj + ngh;
      const Real y1 = r.x2min + (static_cast<Real>(jj) + 0.5)*dx2 - oy;
      for (int ii = 0; ii < nx; ++ii) {
        const int i = ii + ngh;
        const Real y0 = r.x1min + (static_cast<Real>(ii) + 0.5)*dx1 - ox;
        const Real Y[3] = {y0, y1, y2};
        Real q[3];
        for (int c = 0; c < 3; ++c) { q[c] = src(m, c, k, j, i); }
        if (self) {
          Real S[3], Sf[3];
          VecS(A, k, j, i, h, S);
          VecS(F, k, j, i, h, Sf);
          for (int c = 0; c < 3; ++c) { q[c] += (S[c] - Sf[c])*ih2; }
        }
        for (int c = 0; c < 3; ++c) {
          const Real w = q[c]*dv;
          acc[c] += w;
          for (int a = 0; a < 3; ++a) {
            acc[3 + 3*c + a] += w*Y[a];
            for (int b = 0; b < 3; ++b) { acc[12 + 9*c + 3*a + b] += w*Y[a]*Y[b]; }
          }
        }
      }
    }
    for (int q = 0; q < kNMom; ++q) { part(m*nz + kk, q) = acc[q]; }
  });
  auto h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), part);
  for (int q = 0; q < kNMom; ++q) { mom[q] = 0.0; }
  for (int p = 0; p < nmmb*nz; ++p) {
    for (int q = 0; q < kNMom; ++q) { mom[q] += h(p, q); }
  }
}

void MetricSelfCheck(const CoefSource &cs, const Real lo[3], const Real len[3], int npts,
                     Real fd, Real out[4]) {
  DvceArray2D<Real> res("kmg_metric_chk", npts, 4);
  const Real l0 = lo[0], l1 = lo[1], l2 = lo[2];
  const Real s0 = len[0], s1 = len[1], s2 = len[2];
  par_for("kmg::MetricSelfCheck", DevExeSpace(), 0, npts-1,
  KOKKOS_LAMBDA(const int p) {
    using kerr_impl::Sym;
    // a deterministic low-discrepancy point (additive recurrence, irrational steps)
    const Real t = static_cast<Real>(p) + 0.5;
    const Real f0 = t*0.7548776662466927 - Kokkos::floor(t*0.7548776662466927);
    const Real f1 = t*0.5698402909980532 - Kokkos::floor(t*0.5698402909980532);
    const Real f2 = t*0.4301597090019468 - Kokkos::floor(t*0.4301597090019468);
    const Real X[3] = {l0 + f0*s0, l1 + f1*s1, l2 + f2*s2};
    Real gt[6], gtu[6], dgt[3][6], ddgt[6][6];
    PointMetric2(cs, X[0], X[1], X[2], gt, gtu, dgt, ddgt);
    Real gt1[6], gtu1[6], dgt1[3][6];
    PointMetric(cs, X[0], X[1], X[2], gt1, gtu1, dgt1);
    Real e0 = 0.0, e1 = 0.0, sc = 0.0, e3 = 0.0;
    for (int q = 0; q < 6; ++q) {
      e3 = Kokkos::fmax(e3, Kokkos::fabs(gt[q] - gt1[q]));
      for (int k = 0; k < 3; ++k) { e0 = Kokkos::fmax(e0, Kokkos::fabs(dgt[k][q] - dgt1[k][q])); }
    }
    for (int l = 0; l < 3; ++l) {
      Real Xp[3] = {X[0], X[1], X[2]}, Xm[3] = {X[0], X[1], X[2]};
      Xp[l] += fd;
      Xm[l] -= fd;
      Real gp[6], gup[6], dgp[3][6], gm[6], gum[6], dgm[3][6];
      PointMetric(cs, Xp[0], Xp[1], Xp[2], gp, gup, dgp);
      PointMetric(cs, Xm[0], Xm[1], Xm[2], gm, gum, dgm);
      for (int k = 0; k < 3; ++k) {
        for (int q = 0; q < 6; ++q) {
          const Real fdv = (dgp[k][q] - dgm[k][q])/(2.0*fd);
          e1 = Kokkos::fmax(e1, Kokkos::fabs(ddgt[Sym(k, l)][q] - fdv));
          sc = Kokkos::fmax(sc, Kokkos::fabs(ddgt[Sym(k, l)][q]));
        }
      }
    }
    res(p, 0) = e0;
    res(p, 1) = e1;
    res(p, 2) = sc;
    res(p, 3) = e3;
  });
  auto h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), res);
  for (int c = 0; c < 4; ++c) { out[c] = 0.0; }
  for (int p = 0; p < npts; ++p) {
    for (int c = 0; c < 4; ++c) { out[c] = std::max(out[c], h(p, c)); }
  }
}

}  // namespace kmg
}  // namespace cfc
