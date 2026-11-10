//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file resistive_diagnostics.cpp
//! \brief Reductions for resistive GRMHD diagnostics (see resistive_diagnostics.hpp).

#include <cmath>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "dyn_grmhd/resistive/resistive_diagnostics.hpp"

namespace resistive {

namespace {

void AllReduceSumMax(Real *sums, int nsum, Real *maxs, int nmax) {
#if MPI_PARALLEL_ENABLED
  if (nsum > 0) {
    MPI_Allreduce(MPI_IN_PLACE, sums, nsum, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  }
  if (nmax > 0) {
    MPI_Allreduce(MPI_IN_PLACE, maxs, nmax, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  }
#endif
}

}  // namespace

//----------------------------------------------------------------------------------------
ChargeDiag TotalCharge(MeshBlockPack *pmbp, const DvceFaceFld4D<Real> &ef) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, nx1 = indcs.nx1;
  const int js = indcs.js, je = indcs.je, nx2 = indcs.nx2;
  const int ks = indcs.ks, ke = indcs.ke, nx3 = indcs.nx3;
  const bool use2 = pmbp->pmesh->multi_d, use3 = pmbp->pmesh->three_d;
  const int nmkji = pmbp->nmb_thispack*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1, nji = nx2*nx1;
  auto &size = pmbp->pmb->mb_size;
  auto e1 = ef.x1f, e2 = ef.x2f, e3 = ef.x3f;

  Real qtot = 0.0, qbnd = 0.0, qmax = 0.0;
  Kokkos::parallel_reduce("resist_charge", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &qt, Real &qb, Real &qm) {
    int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    const Real dx1 = size.d_view(m).dx1, dx2 = size.d_view(m).dx2;
    const Real dx3 = size.d_view(m).dx3;
    const Real dv = dx1*dx2*dx3;
    Real q = (e1(m,k,j,i+1) - e1(m,k,j,i))/dx1;
    Real b = 0.0;
    if (i == ie) b += e1(m,k,j,i+1)*dx2*dx3;
    if (i == is) b -= e1(m,k,j,i)*dx2*dx3;
    if (use2) {
      q += (e2(m,k,j+1,i) - e2(m,k,j,i))/dx2;
      if (j == je) b += e2(m,k,j+1,i)*dx1*dx3;
      if (j == js) b -= e2(m,k,j,i)*dx1*dx3;
    }
    if (use3) {
      q += (e3(m,k+1,j,i) - e3(m,k,j,i))/dx3;
      if (k == ke) b += e3(m,k+1,j,i)*dx1*dx2;
      if (k == ks) b -= e3(m,k,j,i)*dx1*dx2;
    }
    qt += q*dv;
    qb += b;
    qm = fmax(qm, fabs(q));
  }, Kokkos::Sum<Real>(qtot), Kokkos::Sum<Real>(qbnd), Kokkos::Max<Real>(qmax));

  Real sums[2] = {qtot, qbnd};
  AllReduceSumMax(sums, 2, &qmax, 1);
  return ChargeDiag{sums[0], sums[1], qmax};
}

//----------------------------------------------------------------------------------------
Real MaxDivB(MeshBlockPack *pmbp, const DvceFaceFld4D<Real> &bf) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int is = indcs.is, nx1 = indcs.nx1;
  const int js = indcs.js, nx2 = indcs.nx2;
  const int ks = indcs.ks, nx3 = indcs.nx3;
  const bool use2 = pmbp->pmesh->multi_d, use3 = pmbp->pmesh->three_d;
  const int nmkji = pmbp->nmb_thispack*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1, nji = nx2*nx1;
  auto &size = pmbp->pmb->mb_size;
  auto b1 = bf.x1f, b2 = bf.x2f, b3 = bf.x3f;

  Real dmax = 0.0;
  Kokkos::parallel_reduce("resist_divb", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &dm) {
    int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    Real d = (b1(m,k,j,i+1) - b1(m,k,j,i))/size.d_view(m).dx1;
    if (use2) d += (b2(m,k,j+1,i) - b2(m,k,j,i))/size.d_view(m).dx2;
    if (use3) d += (b3(m,k+1,j,i) - b3(m,k,j,i))/size.d_view(m).dx3;
    dm = fmax(dm, fabs(d));
  }, Kokkos::Max<Real>(dmax));

  AllReduceSumMax(nullptr, 0, &dmax, 1);
  return dmax;
}

//----------------------------------------------------------------------------------------
MismatchDiag ECellFaceMismatch(MeshBlockPack *pmbp, const DvceArray5D<Real> &ec,
                               const DvceFaceFld4D<Real> &ef) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int is = indcs.is, nx1 = indcs.nx1;
  const int js = indcs.js, nx2 = indcs.nx2;
  const int ks = indcs.ks, nx3 = indcs.nx3;
  const int nmkji = pmbp->nmb_thispack*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1, nji = nx2*nx1;
  auto &size = pmbp->pmb->mb_size;
  auto e1 = ef.x1f, e2 = ef.x2f, e3 = ef.x3f;

  Real emax = 0.0, esum = 0.0, vsum = 0.0;
  Kokkos::parallel_reduce("resist_ecmis", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &em, Real &es, Real &vs) {
    int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    const Real dv = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    const Real d1 = fabs(ec(m,0,k,j,i) - 0.5*(e1(m,k,j,i) + e1(m,k,j,i+1)));
    const Real d2 = fabs(ec(m,1,k,j,i) - 0.5*(e2(m,k,j,i) + e2(m,k,j+1,i)));
    const Real d3 = fabs(ec(m,2,k,j,i) - 0.5*(e3(m,k,j,i) + e3(m,k+1,j,i)));
    const Real d = fmax(d1, fmax(d2, d3));
    em = fmax(em, d);
    es += d*dv;
    vs += dv;
  }, Kokkos::Max<Real>(emax), Kokkos::Sum<Real>(esum), Kokkos::Sum<Real>(vsum));

  Real sums[2] = {esum, vsum};
  AllReduceSumMax(sums, 2, &emax, 1);
  return MismatchDiag{emax, (sums[1] > 0.0) ? sums[0]/sums[1] : 0.0};
}

//----------------------------------------------------------------------------------------
void NewtonStats::Read(int64_t out[kNewtonNCounters]) const {
  auto h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), d);
  for (int n = 0; n < kNewtonNCounters; ++n) out[n] = h(n);
#if MPI_PARALLEL_ENABLED
  int64_t sums[4] = {out[kNewtonCells], out[kNewtonIterSum], out[kNewtonFallback],
                     out[kNewtonFail]};
  MPI_Allreduce(MPI_IN_PLACE, sums, 4, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &out[kNewtonIterMax], 1, MPI_INT64_T, MPI_MAX,
                MPI_COMM_WORLD);
  out[kNewtonCells] = sums[0]; out[kNewtonIterSum] = sums[1];
  out[kNewtonFallback] = sums[2]; out[kNewtonFail] = sums[3];
#endif
}

//----------------------------------------------------------------------------------------
void NewtonStats::Load(const int64_t in[kNewtonNCounters]) const {
  auto h = Kokkos::create_mirror_view(d);
  for (int n = 0; n < kNewtonNCounters; ++n) {
    h(n) = (global_variable::my_rank == 0) ? in[n] : 0;
  }
  Kokkos::deep_copy(d, h);
}

//----------------------------------------------------------------------------------------
void SigmaMHist::Read(int64_t out[kHistBinnings][kHistBins][kHNCounters]) const {
  auto h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), d);
  for (int s = 0; s < kHistBinnings; ++s)
    for (int b = 0; b < kHistBins; ++b)
      for (int c = 0; c < kHNCounters; ++c) out[s][b][c] = h(s, b, c);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &out[0][0][0], kHistBinnings*kHistBins*kHNCounters,
                MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
#endif
}

//----------------------------------------------------------------------------------------
void SigmaMHist::Load(const int64_t in[kHistBinnings][kHistBins][kHNCounters]) const {
  auto h = Kokkos::create_mirror_view(d);
  for (int s = 0; s < kHistBinnings; ++s)
    for (int b = 0; b < kHistBins; ++b)
      for (int c = 0; c < kHNCounters; ++c) {
        h(s, b, c) = (global_variable::my_rank == 0) ? in[s][b][c] : 0;
      }
  Kokkos::deep_copy(d, h);
}

//----------------------------------------------------------------------------------------
void ProtoLog::Read(int64_t out[kNLogSets][kHistBins][kLogBins],
                    Real mxo[kNLogSets][kHistBins]) const {
  auto h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), d);
  auto hm = Kokkos::create_mirror_view_and_copy(HostMemSpace(), mx);
  for (int s = 0; s < kNLogSets; ++s) {
    for (int b = 0; b < kHistBins; ++b) {
      mxo[s][b] = hm(s, b);
      for (int c = 0; c < kLogBins; ++c) out[s][b][c] = h(s, b, c);
    }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &out[0][0][0], kNLogSets*kHistBins*kLogBins, MPI_INT64_T,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &mxo[0][0], kNLogSets*kHistBins, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
#endif
}

//----------------------------------------------------------------------------------------
void RFailCount::Read(int64_t *out) const {
  auto h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), d);
  for (int r = 0; r < kRFailReasons; ++r)
    for (int b = 0; b < kRFailRBins; ++b) out[r*kRFailRBins + b] = h(r, b);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, out, kRFailReasons*kRFailRBins, MPI_INT64_T, MPI_SUM,
                MPI_COMM_WORLD);
#endif
}

}  // namespace resistive
