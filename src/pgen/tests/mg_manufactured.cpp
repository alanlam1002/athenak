//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mg_manufactured.cpp
//  \brief Manufactured-solution test of the multigrid fine/coarse interface (coarser-face
//  prolongation, src/cfc/DEVELOPMENT.md item 67) with the gravity solver on a periodic,
//  statically refined mesh. Regression test:
//  tst/test_suite/multigrid/test_mg_manufactured_*.
//  Expected with the current rule (cfc_opt O-035..O-037): phi second order in L2 and max,
//  the gradient second order in L2 and first order in max, and no kink (second
//  differences O(h^2)) at parent, sibling, mid-face and open block ends. The former rule
//  had an O(1)
//  gradient error (0.126) and a first-order kink at parent-block edges.
//
//  phi_ex = A0 sin(2 pi x + a) sin(2 pi y + b) cos(2 pi z + c)
//         + A1 exp(-|x-c1|^2 / (2 s1^2)) + A2 exp(-|x-c2|^2 / (2 s2^2))
//  The sine part is periodic; the Gaussians (s1 = 0.02, s2 = 0.06 by default) are below
//  e^-34 at the box faces of [-0.5, 0.5]^3 (printed at start). Source: four_pi_G rho =
//  lap(phi_ex) + C, C > 0 keeping rho positive; the periodic solve removes C with the
//  mean. Errors after removing the volume mean of phi - phi_ex. Run with time/nlim = 1
//  and a tiny cfl_number so the hydro step does not change rho measurably.
//
//  Error classes per cell (from the MeshBlock's neighbour levels, as the MG diagnostic):
//    fcend: <= 2 cells from a face whose neighbour is coarser AND <= 2 cells from a
//           transverse block edge whose edge neighbour beyond is coarser
//    fc   : <= 2 cells from a coarser face, not fcend (fine side of the interface)
//    fcc  : <= 2 cells from a finer face (coarse side), not fc/fcend
//    int  : the rest
//  Printed per class (rank 0, "MGMANU" lines): cells, L2 (volume rms) and max of
//  |phi - phi_ex| / max|phi_ex|, the share of the total volume-integrated squared error,
//  and the same for the centred-difference gradient error / max|grad phi_ex|.
//  Kink: on the first fine cell layer at every coarser face, the second difference
//  of e = phi - phi_ex - mean along each transverse direction,
//  |e(t+1) - 2 e(t) + e(t-1)| / max|phi_ex|, at the block-edge cells t = 0 and t = n-1
//  (stencil uses the transverse ghost;
//  only where that transverse neighbour is on the same level), split into "parent" (the
//  neighbour block has another parent: low child at its low end, high child at its high
//  end)
//  and "sibling" edges, and at t = n/2 ("mid", reference). MGKINK line, rank 0.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "mesh/nghbr_index.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "gravity/gravity.hpp"
#include "gravity/mg_gravity.hpp"
#include "pgen/pgen.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {
constexpr Real kPi = 3.14159265358979323846;

struct Manu {
  Real a0, pa, pb, pc;
  Real a1, s1, c1[3];
  Real a2, s2, c2[3];
  KOKKOS_INLINE_FUNCTION Real Phi(Real x, Real y, Real z) const {
    const Real k = 2.0*kPi;
    Real r1 = SQR(x-c1[0]) + SQR(y-c1[1]) + SQR(z-c1[2]);
    Real r2 = SQR(x-c2[0]) + SQR(y-c2[1]) + SQR(z-c2[2]);
    return a0*std::sin(k*x+pa)*std::sin(k*y+pb)*std::cos(k*z+pc)
         + a1*std::exp(-r1/(2.0*s1*s1)) + a2*std::exp(-r2/(2.0*s2*s2));
  }
  KOKKOS_INLINE_FUNCTION Real Lap(Real x, Real y, Real z) const {
    const Real k = 2.0*kPi;
    Real r1 = SQR(x-c1[0]) + SQR(y-c1[1]) + SQR(z-c1[2]);
    Real r2 = SQR(x-c2[0]) + SQR(y-c2[1]) + SQR(z-c2[2]);
    Real g1 = a1*std::exp(-r1/(2.0*s1*s1)), g2 = a2*std::exp(-r2/(2.0*s2*s2));
    return -3.0*k*k*a0*std::sin(k*x+pa)*std::sin(k*y+pb)*std::cos(k*z+pc)
         + g1*(r1/(s1*s1*s1*s1) - 3.0/(s1*s1)) + g2*(r2/(s2*s2*s2*s2) - 3.0/(s2*s2));
  }
  void Grad(Real x, Real y, Real z, Real g[3]) const {
    const Real k = 2.0*kPi;
    Real r1 = SQR(x-c1[0]) + SQR(y-c1[1]) + SQR(z-c1[2]);
    Real r2 = SQR(x-c2[0]) + SQR(y-c2[1]) + SQR(z-c2[2]);
    Real g1 = a1*std::exp(-r1/(2.0*s1*s1)), g2 = a2*std::exp(-r2/(2.0*s2*s2));
    Real sx = std::sin(k*x+pa), sy = std::sin(k*y+pb), cz = std::cos(k*z+pc);
    g[0] = a0*k*std::cos(k*x+pa)*sy*cz - g1*(x-c1[0])/(s1*s1) - g2*(x-c2[0])/(s2*s2);
    g[1] = a0*k*sx*std::cos(k*y+pb)*cz - g1*(y-c1[1])/(s1*s1) - g2*(y-c2[1])/(s2*s2);
    g[2] = -a0*k*sx*sy*std::sin(k*z+pc) - g1*(z-c1[2])/(s1*s1) - g2*(z-c2[2])/(s2*s2);
  }
};

Manu ReadManu(ParameterInput *pin) {
  Manu p;
  p.a0 = pin->GetOrAddReal("problem", "a0", 1.0);
  p.pa = pin->GetOrAddReal("problem", "phase_a", 0.3);
  p.pb = pin->GetOrAddReal("problem", "phase_b", 1.1);
  p.pc = pin->GetOrAddReal("problem", "phase_c", 0.7);
  p.a1 = pin->GetOrAddReal("problem", "a1", 0.02);
  p.s1 = pin->GetOrAddReal("problem", "s1", 0.02);
  p.c1[0] = pin->GetOrAddReal("problem", "c1x", 0.020);
  p.c1[1] = pin->GetOrAddReal("problem", "c1y", 0.010);
  p.c1[2] = pin->GetOrAddReal("problem", "c1z", -0.005);
  p.a2 = pin->GetOrAddReal("problem", "a2", 0.1);
  p.s2 = pin->GetOrAddReal("problem", "s2", 0.06);
  p.c2[0] = pin->GetOrAddReal("problem", "c2x", -0.050);
  p.c2[1] = pin->GetOrAddReal("problem", "c2y", 0.030);
  p.c2[2] = pin->GetOrAddReal("problem", "c2z", 0.020);
  return p;
}

void MGManufacturedErrors(ParameterInput *pin, Mesh *pm);
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::MGManufactured()

void ProblemGenerator::MGManufactured(ParameterInput *pin, const bool restart) {
  pgen_final_func = MGManufacturedErrors;
  if (restart) return;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  Real four_pi_G = pin->GetOrAddReal("gravity", "four_pi_G", 1.0);
  if (pmbp->pgrav != nullptr) {
    pmbp->pgrav->four_pi_G = four_pi_G;
    if (pmbp->pgrav->pmgd != nullptr) pmbp->pgrav->pmgd->SetFourPiG(four_pi_G);
  }
  if (pmbp->phydro == nullptr) return;
  const Manu mp = ReadManu(pin);
  Real rho_c = pin->GetOrAddReal("problem", "rho_const", 0.0);
  if (rho_c <= 0.0) {
    // max |lap| bound: sine part + Gaussian peaks (|r^2/s^4 - 3/s^2| <= 3/s^2 at r = 0)
    Real lmax = 3.0*SQR(2.0*kPi)*std::abs(mp.a0) + 3.0*std::abs(mp.a1)/SQR(mp.s1)
              + 3.0*std::abs(mp.a2)/SQR(mp.s2);
    rho_c = 1.1*lmax/four_pi_G + 1.0;
  }
  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  auto u0 = pmbp->phydro->u0;
  const bool is_ideal = pmbp->phydro->peos->eos_data.is_ideal;
  const Real fpg = four_pi_G, rc = rho_c;
  par_for("mg_manufactured_init", DevExeSpace(), 0, pmbp->nmb_thispack - 1, ks, ke, js,
          je, is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x = CellCenterX(i - is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real y = CellCenterX(j - js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real z = CellCenterX(k - ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    Real rho = mp.Lap(x, y, z)/fpg + rc;
    u0(m, IDN, k, j, i) = rho;
    u0(m, IM1, k, j, i) = 0.0;
    u0(m, IM2, k, j, i) = 0.0;
    u0(m, IM3, k, j, i) = 0.0;
    if (is_ideal) u0(m, IEN, k, j, i) = rho;
  });
  if (global_variable::my_rank == 0) {
    const Real h = 0.5;
    Real g1 = std::abs(mp.a1)*std::exp(-SQR(h - std::abs(mp.c1[0]))/(2.0*SQR(mp.s1)));
    Real g2 = std::abs(mp.a2)*std::exp(-SQR(h - std::abs(mp.c2[0]))/(2.0*SQR(mp.s2)));
    std::cout << std::scientific << std::setprecision(4)
              << "MGMANU setup: rho_const " << rho_c
              << ", Gaussian values at the nearest "
              << "box face (x1): " << g1 << " " << g2 << " (residual non-periodicity)"
              << std::endl;
  }
}

namespace {
//----------------------------------------------------------------------------------------
//! \fn void MGManufacturedErrors()

void MGManufacturedErrors(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->pgrav == nullptr) return;
  const Manu mp = ReadManu(pin);
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx = indcs.nx1, ny = indcs.nx2, nz = indcs.nx3;
  const int nmb = pmbp->nmb_thispack;
  auto phi = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmbp->pgrav->phi);
  auto size = Kokkos::create_mirror_view_and_copy(HostMemSpace(),
                                                  pmbp->pmb->mb_size.d_view);
  auto nghbr = Kokkos::create_mirror_view_and_copy(HostMemSpace(),
                                                   pmbp->pmb->nghbr.d_view);
  auto mblev = Kokkos::create_mirror_view_and_copy(HostMemSpace(),
                                                   pmbp->pmb->mb_lev.d_view);

  // per block and face: -2 none, -1 finer face neighbour, >= 0 coarser (end bits)
  // Neighbour level at offset (a, b, c), -1 if none. A coarser FACE neighbour is stored
  // in the subblock (n1, n2) given by this block's child position (mesh/meshblock.cpp),
  // not in (0, 0): scan all subblocks (subblock 0 alone misses 3/4 of the coarser faces).
  auto lev_at = [&](int m, int a, int b, int c, int) {
    const int nf = (a != 0) + (b != 0) + (c != 0);
    const int n2max = (nf == 1) ? 1 : 0, n1max = (nf <= 2) ? 1 : 0;
    for (int n2 = 0; n2 <= n2max; ++n2) for (int n1 = 0; n1 <= n1max; ++n1) {
      int nn = NeighborIndex(a, b, c, n1, n2);
      if (nn >= 0 && nn < 56 && nghbr(m, nn).gid >= 0) return nghbr(m, nn).lev;
    }
    return -1;
  };
  auto coarser_at = [&](int m, int a, int b, int c) {
    for (int f = 0; f <= 1; ++f) {
      int l = lev_at(m, a, b, c, f);
      if (l >= 0 && l < mblev(m)) return 1;
    }
    return 0;
  };
  std::vector<int> face(6*nmb, -2);
  for (int m = 0; m < nmb; ++m) {
    for (int f = 0; f < 6; ++f) {
      int o[3] = {0, 0, 0};
      o[f/2] = (f % 2) ? 1 : -1;
      int l = lev_at(m, o[0], o[1], o[2], 0);
      if (l < 0) continue;
      if (l > mblev(m)) {
        face[6*m + f] = -1;
        continue;
      }
      if (l == mblev(m)) continue;
      int bits[4];
      if (f/2 == 0) {
        bits[0] = coarser_at(m, o[0], -1, 0); bits[1] = coarser_at(m, o[0], 1, 0);
        bits[2] = coarser_at(m, o[0], 0, -1); bits[3] = coarser_at(m, o[0], 0, 1);
      } else if (f/2 == 1) {
        bits[0] = coarser_at(m, -1, o[1], 0); bits[1] = coarser_at(m, 1, o[1], 0);
        bits[2] = coarser_at(m, 0, o[1], -1); bits[3] = coarser_at(m, 0, o[1], 1);
      } else {
        bits[0] = coarser_at(m, -1, 0, o[2]); bits[1] = coarser_at(m, 1, 0, o[2]);
        bits[2] = coarser_at(m, 0, -1, o[2]); bits[3] = coarser_at(m, 0, 1, o[2]);
      }
      face[6*m + f] = bits[0] | (bits[1] << 1) | (bits[2] << 2) | (bits[3] << 3);
    }
  }
  auto cls_of = [&](int m, int i, int j, int k) {
    int cls = 0;   // 0 int, 1 fcc, 2 fc, 3 fcend
    for (int f = 0; f < 6; ++f) {
      const int e = face[6*m + f];
      if (e == -2) continue;
      const int dir = f/2, hi = f % 2;
      const int c = (dir == 0) ? i : (dir == 1) ? j : k;
      const int n = (dir == 0) ? nx : (dir == 1) ? ny : nz;
      if ((hi ? n - 1 - c : c) >= 2) continue;
      if (e == -1) {
        cls = std::max(cls, 1);
        continue;
      }
      const int ta = (dir == 0) ? j : i, na = (dir == 0) ? ny : nx;
      const int tb = (dir == 2) ? j : k, nb = (dir == 2) ? ny : nz;
      if (((e & 1) && ta < 2) || ((e & 2) && ta > na - 3) ||
          ((e & 4) && tb < 2) || ((e & 8) && tb > nb - 3)) {
        cls = 3;
      } else {
        cls = std::max(cls, 2);
      }
    }
    return cls;
  };
  // "Open" end = end of a coarser face whose edge neighbour beyond it is NOT
  // coarser (concave corner of an L-shaped region); the rule clamps its slope one-sided
  // there.
  // A cell is "open" if <= 2 cells from such a face and <= 2 cells from the open end.
  auto open_of = [&](int m, int i, int j, int k) {
    for (int f = 0; f < 6; ++f) {
      const int e = face[6*m + f];
      if (e < 0) continue;
      const int dir = f/2, hi = f % 2;
      const int c = (dir == 0) ? i : (dir == 1) ? j : k;
      const int n = (dir == 0) ? nx : (dir == 1) ? ny : nz;
      if ((hi ? n - 1 - c : c) >= 2) continue;
      const int ta = (dir == 0) ? j : i, na = (dir == 0) ? ny : nx;
      const int tb = (dir == 2) ? j : k, nb = (dir == 2) ? ny : nz;
      // only the outer (parent-edge) end of each child: a sibling-side end's edge
      // neighbour is the face neighbour itself and is not stored, so its bit is 0 without
      // being open
      const auto &ll = pm->lloc_eachmb[pmbp->pmb->mb_gid.h_view(m)];
      const int px[3] = {static_cast<int>(ll.lx1 & 1), static_cast<int>(ll.lx2 & 1),
                         static_cast<int>(ll.lx3 & 1)};
      const int pa = px[(dir == 0) ? 1 : 0], pb = px[(dir == 2) ? 1 : 2];
      if ((!(e & 1) && pa == 0 && ta < 2) || (!(e & 2) && pa == 1 && ta > na - 3) ||
          (!(e & 4) && pb == 0 && tb < 2) || (!(e & 8) && pb == 1 && tb > nb - 3)) {
        return true;
      }
    }
    return false;
  };
  auto pos = [&](int m, int i, int j, int k, Real x[3]) {
    x[0] = CellCenterX(i, nx, size(m).x1min, size(m).x1max);
    x[1] = CellCenterX(j, ny, size(m).x2min, size(m).x2max);
    x[2] = CellCenterX(k, nz, size(m).x3min, size(m).x3max);
  };
  // pass 1: mean of phi - phi_ex, max |phi_ex|, max |grad phi_ex|
  double s1[5] = {0.0, 0.0, 0.0, 0.0, 0.0};   // sum d*vol, vol, (max phi), (max grad), -
  for (int m = 0; m < nmb; ++m) {
    const Real vol = size(m).dx1*size(m).dx2*size(m).dx3;
    for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
      Real x[3], g[3];
      pos(m, i, j, k, x);
      Real pe = mp.Phi(x[0], x[1], x[2]);
      mp.Grad(x[0], x[1], x[2], g);
      s1[0] += (phi(m, 0, ks+k, js+j, is+i) - pe)*vol;
      s1[1] += vol;
      s1[2] = std::max(s1[2], static_cast<double>(std::abs(pe)));
      const double gn = std::sqrt(g[0]*g[0] + g[1]*g[1] + g[2]*g[2]);
      s1[3] = std::max(s1[3], gn);
    }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, s1, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, s1 + 2, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
  const double shift = s1[0]/s1[1], pmax = s1[2], gmax = s1[3];
  // pass 2: per class (0 int, 1 fcc, 2 fc, 3 fcend; 4 = all) sums and maxima
  // sum: err^2 vol, gerr^2 vol, vol; mx: err, gerr
  double sum[3*5] = {0.0}, mx[2*5] = {0.0};
  double ncl[5] = {0.0};
  // Location of each class's gradient-error maximum, and the same maxima
  // restricted to stencils that use no ghost cell ("act": 1 <= i,j,k <= n-2).
  double amx[5] = {0.0};             // act-only grad max per class
  // open cells: phi max, grad max, act grad max, count
  double omx[4] = {0.0, 0.0, 0.0, 0.0};
  struct Loc {
    double ge = -1.0;
    int m = -1, i = 0, j = 0, k = 0, comp = 0;
    double g[3] = {0, 0, 0};
  };
  Loc loc[5], aloc[5];
  for (int m = 0; m < nmb; ++m) {
    const Real vol = size(m).dx1*size(m).dx2*size(m).dx3;
    const Real dx = size(m).dx1, dy = size(m).dx2, dz = size(m).dx3;
    for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
      Real x[3], g[3];
      pos(m, i, j, k, x);
      mp.Grad(x[0], x[1], x[2], g);
      const int I = is+i, J = js+j, K = ks+k;
      double e = std::abs(phi(m, 0, K, J, I) - mp.Phi(x[0], x[1], x[2]) - shift)/pmax;
      double gx = (phi(m, 0, K, J, I+1) - phi(m, 0, K, J, I-1))/(2.0*dx) - g[0];
      double gy = (phi(m, 0, K, J+1, I) - phi(m, 0, K, J-1, I))/(2.0*dy) - g[1];
      double gz = (phi(m, 0, K+1, J, I) - phi(m, 0, K-1, J, I))/(2.0*dz) - g[2];
      double ge = std::sqrt(gx*gx + gy*gy + gz*gz)/gmax;
      const int c = cls_of(m, i, j, k);
      if (open_of(m, i, j, k)) {
        const bool act0 = (i >= 1 && i <= nx-2 && j >= 1 && j <= ny-2 &&
                           k >= 1 && k <= nz-2);
        omx[0] = std::max(omx[0], e); omx[1] = std::max(omx[1], ge);
        if (act0) omx[2] = std::max(omx[2], ge);
        omx[3] += 1.0;
      }
      const bool act = (i >= 1 && i <= nx-2 && j >= 1 && j <= ny-2 &&
                        k >= 1 && k <= nz-2);
      const double ga[3] = {std::abs(gx), std::abs(gy), std::abs(gz)};
      const int comp = (ga[0] >= ga[1] && ga[0] >= ga[2]) ? 0 : (ga[1] >= ga[2]) ? 1 : 2;
      for (int cc : {c, 4}) {
        sum[3*cc] += e*e*vol; sum[3*cc+1] += ge*ge*vol; sum[3*cc+2] += vol;
        mx[2*cc] = std::max(mx[2*cc], e); mx[2*cc+1] = std::max(mx[2*cc+1], ge);
        ncl[cc] += 1.0;
        if (ge > loc[cc].ge) {
          loc[cc].ge = ge; loc[cc].m = m; loc[cc].i = i; loc[cc].j = j; loc[cc].k = k;
          loc[cc].comp = comp; loc[cc].g[0] = gx; loc[cc].g[1] = gy; loc[cc].g[2] = gz;
        }
        if (act) {
          amx[cc] = std::max(amx[cc], ge);
          if (ge > aloc[cc].ge) {
            aloc[cc].ge = ge; aloc[cc].m = m;
            aloc[cc].i = i; aloc[cc].j = j; aloc[cc].k = k;
            aloc[cc].comp = comp;
            aloc[cc].g[0] = gx; aloc[cc].g[1] = gy; aloc[cc].g[2] = gz;
          }
        }
      }
    }
  }
  // kink: 0 parent edge, 1 sibling edge, 2 mid, 3 open end (edge neighbour not coarser);
  // count of evaluated stencils
  double kmax[4] = {0.0, 0.0, 0.0, 0.0}, kcnt[4] = {0.0, 0.0, 0.0, 0.0};
  // open ends (per face end): edge neighbour same, finer, none
  double oend[3] = {0.0, 0.0, 0.0};
  {
    auto &gidh = pmbp->pmb->mb_gid.h_view;
    // i, j, k interior-relative, may be -1 or n
    auto err = [&](int m, int i, int j, int k) {
      Real x[3];
      pos(m, i, j, k, x);
      return static_cast<double>(phi(m, 0, ks+k, js+j, is+i) - mp.Phi(x[0], x[1], x[2])
                                 - shift);
    };
    for (int m = 0; m < nmb; ++m) {
      const auto &ll = pm->lloc_eachmb[gidh(m)];
      const int par[3] = {static_cast<int>(ll.lx1 & 1), static_cast<int>(ll.lx2 & 1),
                          static_cast<int>(ll.lx3 & 1)};
      const int n = nx;
      for (int f = 0; f < 6; ++f) {
        if (face[6*m + f] < 0) continue;   // not a coarser face
        const int d = f/2, cn = (f % 2) ? n - 1 : 0;
        for (int a = 0; a < 3; ++a) {
          if (a == d) continue;
          const int b = 3 - d - a;
          for (int tt = 0; tt < 3; ++tt) {
            const int t = (tt == 0) ? 0 : (tt == 1) ? n - 1 : n/2;
            int cls = 2;
            if (tt < 2) {
              int o[3] = {0, 0, 0};
              o[a] = (tt == 0) ? -1 : 1;
              if (lev_at(m, o[0], o[1], o[2], 0) != mblev(m)) continue;
              cls = ((tt == 0 && par[a] == 0) || (tt == 1 && par[a] == 1)) ? 0 : 1;
              const int a0 = (d == 0) ? 1 : 0;   // first transverse direction
              const int bi = 2*((a == a0) ? 0 : 1) + tt;
              if (cls == 0 && !(face[6*m + f] & (1 << bi))) {   // outer end only
                cls = 3;
                int oe[3] = {0, 0, 0};
                oe[d] = (f % 2) ? 1 : -1;
                oe[a] = (tt == 0) ? -1 : 1;
                const int le = lev_at(m, oe[0], oe[1], oe[2], 0);
                oend[(le < 0) ? 2 : (le > mblev(m)) ? 1 : 0] += 1.0;
              }
            }
            for (int sb = 0; sb < n; ++sb) {
              int c0[3], cm[3], cp[3];
              c0[d] = cm[d] = cp[d] = cn;
              c0[b] = cm[b] = cp[b] = sb;
              c0[a] = t; cm[a] = t - 1; cp[a] = t + 1;
              double k2 = std::abs(err(m, cp[0], cp[1], cp[2])
                                   - 2.0*err(m, c0[0], c0[1], c0[2])
                                   + err(m, cm[0], cm[1], cm[2]))/pmax;
              kmax[cls] = std::max(kmax[cls], k2);
              kcnt[cls] += 1.0;
            }
          }
        }
      }
    }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, kmax, 4, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, kcnt, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, omx, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, omx + 3, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, oend, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, sum, 15, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, ncl, 5, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, mx, 10, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, amx, 5, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
  {
    // MGMAXLOC lines: printed by the rank that holds each maximum. Face codes for the six
    // faces (x-, x+, y-, y+, z-, z+): S same level, C coarser (+end bits), F finer,
    // P physical.
    auto &gidh = pmbp->pmb->mb_gid.h_view;
    const char *nm[5] = {"int", "fcc", "fc", "fcend", "all"};
    auto pr = [&](const char *tag, int c, const Loc &L) {
      if (L.m < 0) return;
      Real x[3];
      pos(L.m, L.i, L.j, L.k, x);
      const auto &ll = pm->lloc_eachmb[gidh(L.m)];
      std::ostringstream fs;
      for (int f = 0; f < 6; ++f) {
        int o[3] = {0, 0, 0};
        o[f/2] = (f % 2) ? 1 : -1;
        const int l = lev_at(L.m, o[0], o[1], o[2], 0);
        if (l < 0) {
          const int nn0 = NeighborIndex(o[0], o[1], o[2], 0, 0);
          fs << (nghbr(L.m, nn0).gid < 0 ? "P" : "?");
        } else if (l > mblev(L.m)) {
          fs << "F";
        } else if (l == mblev(L.m)) {
          fs << "S";
        } else {
          fs << "C" << face[6*L.m + f];
        }
        if (f < 5) fs << ",";
      }
      // distance (cells) to the nearest block boundary in each direction, and the
      // 26-neighbour levels relative to this block that differ (edge/corner neighbours
      // included)
      std::ostringstream ns;
      int nd = 0;
      for (int c3 = -1; c3 <= 1; ++c3)
      for (int b = -1; b <= 1; ++b)
      for (int a = -1; a <= 1; ++a) {
        if (a == 0 && b == 0 && c3 == 0) continue;
        const int l = lev_at(L.m, a, b, c3, 0);
        if (l >= 0 && l != mblev(L.m)) {
          if (nd++ < 8) ns << "(" << a << b << c3 << ":" << (l - mblev(L.m)) << ")";
        }
      }
      std::cout << std::scientific << std::setprecision(4) << "MGMAXLOC " << tag << " "
                << std::setw(5) << nm[c] << " ge " << L.ge << " gid " << gidh(L.m)
                << " lev " << ll.level << " lloc " << ll.lx1 << " " << ll.lx2 << " "
                << ll.lx3
                << " ijk " << L.i << " " << L.j << " " << L.k << " (n " << nx << ") x "
                << x[0] << " " << x[1] << " " << x[2] << " gerr " << L.g[0] << " "
                << L.g[1]
                << " " << L.g[2] << " faces " << fs.str() << " nbr_dlev " << nd << " "
                << ns.str() << std::endl;
    };
    for (int c : {4, 0, 1, 2, 3}) {
      if (loc[c].m >= 0 && loc[c].ge == mx[2*c+1]) pr("all", c, loc[c]);
      if (aloc[c].m >= 0 && aloc[c].ge == amx[c]) pr("act", c, aloc[c]);
    }
    if (global_variable::my_rank == 0) {
      std::cout << std::scientific << std::setprecision(6)
                << "MGMANU act grad_max (stencil without ghosts) all " << amx[4]
                << " int " << amx[0] << " fcc " << amx[1]
                << " fc " << amx[2] << " fcend " << amx[3] << std::endl;
    }
  }
  if (global_variable::my_rank == 0) {
    const char *nm[5] = {"int", "fcc", "fc", "fcend", "all"};
    std::cout << std::scientific << std::setprecision(6)
              << "MGMANU norm: max|phi_ex| " << pmax << " max|grad phi_ex| " << gmax
              << " mean shift " << shift << std::endl;
    std::cout << "MGKINK parent " << kmax[0] << " (" << static_cast<std::int64_t>(kcnt[0])
              << ") sibling " << kmax[1] << " (" << static_cast<std::int64_t>(kcnt[1])
              << ") mid " << kmax[2] << " (" << static_cast<std::int64_t>(kcnt[2])
              << ") open " << kmax[3] << " (" << static_cast<std::int64_t>(kcnt[3]) << ")"
              << std::endl;
    std::cout << "MGOPEN cells " << static_cast<std::int64_t>(omx[3]) << " phi_max "
              << omx[0]
              << " grad_max " << omx[1] << " act_grad_max " << omx[2]
              << " ends: edge nbr same "
              << static_cast<std::int64_t>(oend[0]) << " finer "
              << static_cast<std::int64_t>(oend[1])
              << " none " << static_cast<std::int64_t>(oend[2]) << std::endl;
    for (int c : {4, 0, 1, 2, 3}) {
      double l2 = (sum[3*c+2] > 0) ? std::sqrt(sum[3*c]/sum[3*c+2]) : 0.0;
      double gl2 = (sum[3*c+2] > 0) ? std::sqrt(sum[3*c+1]/sum[3*c+2]) : 0.0;
      std::cout << "MGMANU " << std::setw(5) << nm[c] << " cells "
                << static_cast<std::int64_t>(ncl[c]) << " phi_l2 " << l2 << " phi_max "
                << mx[2*c]
                << " phi_share " << ((sum[12] > 0) ? sum[3*c]/sum[12] : 0.0)
                << " grad_l2 " << gl2 << " grad_max " << mx[2*c+1] << " grad_share "
                << ((sum[13] > 0) ? sum[3*c+1]/sum[13] : 0.0) << std::endl;
    }
  }
}
}  // namespace
