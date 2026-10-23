//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mass_closure.cpp
//! \brief see mass_closure.hpp. Compiled with -fp-model=precise (src/CMakeLists.txt).

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "mhd/mhd.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/mass_closure.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace mclose {
namespace {

//! N exact sums reduced together
template <int N>
struct FixArr {
  Fix v[N];
};

template <int N>
struct FixArrReducer {
  using reducer = FixArrReducer;
  using value_type = FixArr<N>;
  using result_view_type = Kokkos::View<value_type, Kokkos::HostSpace,
                                        Kokkos::MemoryUnmanaged>;
  value_type &ref;
  KOKKOS_INLINE_FUNCTION explicit FixArrReducer(value_type &r) : ref(r) {}
  KOKKOS_INLINE_FUNCTION void join(value_type &dst, const value_type &src) const {
    for (int n = 0; n < N; ++n) dst.v[n].Join(src.v[n]);
  }
  KOKKOS_INLINE_FUNCTION void init(value_type &val) const {
    for (int n = 0; n < N; ++n) { val.v[n].l[0] = 0; val.v[n].l[1] = 0; val.v[n].l[2] = 0; }
  }
  KOKKOS_INLINE_FUNCTION value_type &reference() const { return ref; }
  result_view_type view() const { return result_view_type(&ref); }
  KOKKOS_INLINE_FUNCTION bool references_scalar() const { return true; }
};

//! move each limb's excess into the next (arithmetic shifts: exact for negative limbs too)
void Normalize(Fix &f) {
  for (int q = 0; q < 2; ++q) {
    const int64_t carry = f.l[q] >> 38;
    f.l[q] -= carry*(static_cast<int64_t>(1) << 38);
    f.l[q+1] += carry;
  }
}

//! the exact value as a double-double (s + c)
KSum ToK(Fix f) {
  Normalize(f);
  const __int128 v = (static_cast<__int128>(f.l[2]) << 76) +
                     (static_cast<__int128>(f.l[1]) << 38) + static_cast<__int128>(f.l[0]);
  const Real h = static_cast<Real>(v);
  const __int128 rem = v - static_cast<__int128>(h);
  KSum k;
  k.s = h*0x1p-128;
  k.c = static_cast<Real>(rem)*0x1p-128;
  return k;
}

//! the sum over ranks: exact and order-independent (int64 limbs)
void GlobalSum(const Fix *loc, int n, KSum *out) {
  std::vector<int64_t> buf(3*n);
  for (int q = 0; q < n; ++q) {
    Fix f = loc[q];
    Normalize(f);
    for (int b = 0; b < 3; ++b) buf[3*q + b] = f.l[b];
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, buf.data(), 3*n, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
#endif
  for (int q = 0; q < n; ++q) {
    Fix f;
    for (int b = 0; b < 3; ++b) f.l[b] = buf[3*q + b];
    out[q] = ToK(f);
  }
}

//! this rank's sum of a(m, n, k, j, i) dV over interior cells
Fix LocalMass(MeshBlockPack *pp, const DvceArray5D<Real> &a, int n) {
  auto &indcs = pp->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmb = pp->nmb_thispack, nkji = nx3*nx2*nx1, nji = nx2*nx1;
  auto &size = pp->pmb->mb_size;
  FixArr<1> r;
  Kokkos::parallel_reduce("mclose_mass", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*nkji),
  KOKKOS_LAMBDA(const int idx, FixArr<1> &acc) {
    const int m = idx/nkji;
    const int k = (idx - m*nkji)/nji + ks;
    const int j = (idx - m*nkji - (k - ks)*nji)/nx1 + js;
    const int i = idx - m*nkji - (k - ks)*nji - (j - js)*nx1 + is;
    auto &sz = size.d_view(m);
    acc.v[0].Add(a(m, n, k, j, i)*(sz.dx1*sz.dx2*sz.dx3));
  }, FixArrReducer<1>(r));
  return r.v[0];
}

//! double-double arithmetic for the cycle's RK combination: Y_s = M_pre - gam0 M_end(s-1)
//! - gam1 M(u1) differences masses of ~M_grid whose effects are ~1e-16 M_grid, and the
//! weights w_s = prod gam0 are products of rounded constants (1/3, ...); both must be exact
//! for the identity M_end(ns) - M(u1) = sum_s w_s (Y_s + X_s) to hold beyond eps M_grid
struct DD { Real hi = 0.0, lo = 0.0; };
DD TwoSum(Real a, Real b) {
  const Real s = a + b, bb = s - a;
  return {s, (a - (s - bb)) + (b - bb)};
}
DD Add(const DD &a, const DD &b) {
  DD s = TwoSum(a.hi, b.hi);
  s.lo += a.lo + b.lo;
  return TwoSum(s.hi, s.lo);
}
DD Mul(const DD &a, const DD &b) {
  const Real p = a.hi*b.hi;
  const Real e = std::fma(a.hi, b.hi, -p);
  return TwoSum(p, e + (a.hi*b.lo + a.lo*b.hi));
}
DD FromK(const KSum &k) { return TwoSum(k.s, k.c); }
DD FromR(Real x) { return {x, 0.0}; }
DD Neg(const DD &a) { return {-a.hi, -a.lo}; }

}  // namespace

MassClosure::MassClosure(MeshBlockPack *pp, ParameterInput *pin, int var) :
    pmy_pack(pp), var_(var) {
  on = pin->GetOrAddBoolean("mhd", "mass_closure", false);
  every = pin->GetOrAddInteger("mhd", "mass_closure_every", 0);
  tol = pin->GetOrAddReal("mhd", "mass_closure_tol", 1.0e-10);
  mstar = pin->GetOrAddReal("mhd", "mass_closure_mstar", 0.0);
  m_prev_end = std::numeric_limits<Real>::quiet_NaN();
  if (var_ == IDN) {
    tag_ = "MCLOSE"; key_ = "mass_closure";
    if (on && pin->GetOrAddBoolean("mhd", "mass_closure_full", false)) {
      for (int v : {static_cast<int>(IM1), static_cast<int>(IM2), static_cast<int>(IM3),
                    static_cast<int>(IEN)}) {
        extra_.emplace_back(new MassClosure(pp, pin, v));
      }
    }
  } else if (var_ == IEN) {
    tag_ = "ECLOSE"; key_ = "closure_tau";
  } else {
    tag_ = "SCLOSE" + std::to_string(var_ - IM1 + 1);
    key_ = "closure_s" + std::to_string(var_ - IM1 + 1);
  }
}

KSum MassClosure::Mass() const {
  Fix loc = LocalMass(pmy_pack, pmy_pack->pmhd->u0, var_);
  KSum g;
  GlobalSum(&loc, 1, &g);
  return g;
}

void MassClosure::StagePre(int stage) {
  if (!on || stage < 1 || stage > 4) return;
  for (auto &x : extra_) x->StagePre(stage);
  auto &u0 = pmy_pack->pmhd->u0;
  if (var_ != IDN) {   // the variable before the C2P (the mass closure uses the ledger's d0)
    if (x0_.extent_int(0) != u0.extent_int(0) || x0_.extent_int(1) != u0.extent_int(2) ||
        x0_.extent_int(2) != u0.extent_int(3) || x0_.extent_int(3) != u0.extent_int(4)) {
      x0_ = DvceArray4D<Real>("mclose_x0", u0.extent_int(0), u0.extent_int(2),
                              u0.extent_int(3), u0.extent_int(4));
    }
    Kokkos::deep_copy(DevExeSpace(), x0_, Kokkos::subview(u0, Kokkos::ALL, var_,
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
  }
  st_pre[stage-1] = LocalMass(pmy_pack, u0, var_);
  if (stage == 1) st_u1 = LocalMass(pmy_pack, pmy_pack->pmhd->u1, var_);
}

void MassClosure::StagePost(int stage, int ns, const Real *gam0, const Real *gam1,
                            const Real *beta, Real dt, const DvceArray4D<Real> &d0,
                            const DvceArray4D<Real> &cls, const DvceArray4D<Real> &c2p,
                            int cycle, Real time) {
  if (!on || stage < 1 || stage > ns || ns > 4) return;
  for (auto &x : extra_) {
    x->StagePost(stage, ns, gam0, gam1, beta, dt, d0, cls, c2p, cycle, time);
  }
  MeshBlockPack *pp = pmy_pack;
  const int nv = var_;
  const DvceArray4D<Real> x0 = (nv == IDN) ? d0 : x0_;   // the variable before the C2P
  auto &indcs = pp->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmb = pp->nmb_thispack, nkji = nx3*nx2*nx1, nji = nx2*nx1;
  auto &size = pp->pmb->mb_size;
  auto &u0 = pp->pmhd->u0;
  auto &adm = pp->padm->adm;
  auto &cd = pp->pcoord->coord_data;
  const bool exdyn = cd.bh_excise && cd.excision_dynamic &&
                     cd.excision_scheme == ExcisionScheme::lapse;
  const Real exlap = cd.excise_lapse;
  const Real cx = pp->padm->bc_center[0], cy = pp->padm->bc_center[1];
  const Real cz = pp->padm->bc_center[2];
  // the C2P's change by item (the ledger's per-cell class, mass_ledger_hpp classes: 0/1
  // excision floor, 2 zone, 3 floored, 4 other; bit 256 of the code = cold recovery)
  FixArr<kNStage> r;
  Kokkos::parallel_reduce("mclose_items", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*nkji),
  KOKKOS_LAMBDA(const int idx, FixArr<kNStage> &acc) {
    const int m = idx/nkji;
    const int k = (idx - m*nkji)/nji + ks;
    const int j = (idx - m*nkji - (k - ks)*nji)/nx1 + js;
    const int i = idx - m*nkji - (k - ks)*nji - (j - js)*nx1 + is;
    auto &sz = size.d_view(m);
    const Real dv = sz.dx1*sz.dx2*sz.dx3;
    acc.v[kMend].Add(u0(m, nv, k, j, i)*dv);
    acc.v[kMd0].Add(x0(m, k, j, i)*dv);
    const Real dm = u0(m, nv, k, j, i)*dv - x0(m, k, j, i)*dv;
    if (dm == 0.0) return;
    const Real c = cls(m, k, j, i);
    if (c < 1.5) {          // excision floor (0: and flux mask; 1: floor mask only)
      if (exdyn && !(adm.alpha(m, k, j, i) < exlap)) { acc.v[kRf].Add(dm); return; }
      const Real x = CellCenterX(i - is, nx1, sz.x1min, sz.x1max) - cx;
      const Real y = CellCenterX(j - js, nx2, sz.x2min, sz.x2max) - cy;
      const Real z = CellCenterX(k - ks, nx3, sz.x3min, sz.x3max) - cz;
      const Real r2 = x*x + y*y + z*z;
      const int ir = (r2 >= 0.5917*0.5917) + (r2 >= 0.7793*0.7793) + (r2 >= 1.88*1.88)
                     + (r2 >= 2.92*2.92);
      acc.v[kExc0 + ir].Add(dm);
    } else if (c < 2.5) {
      acc.v[(dm < 0.0) ? kZoneRem : kZoneAdd].Add(dm);
    } else if (c < 3.5) {
      acc.v[kFloor].Add(dm);
    } else if (static_cast<int>(c2p(m, k, j, i)) & 256) {
      acc.v[kCrec].Add(dm);
    } else {
      acc.v[kOther].Add(dm);
    }
  }, FixArrReducer<kNStage>(r));
  for (int q = 0; q < kNStage; ++q) st_c[stage-1][q] = r.v[q];
  // face sums: outflow through the mesh's outer faces, inflow into the excision-flux mask
  const bool exc = cd.bh_excise;
  auto &xf = pp->pcoord->excision_flux;
  auto &flx = pp->pmhd->uflx;
  auto &ms = pp->pmesh->mesh_size;
  const Real mx1 = ms.x1min, Mx1 = ms.x1max, mx2 = ms.x2min, Mx2 = ms.x2max;
  const Real mx3 = ms.x3min, Mx3 = ms.x3max;
  FixArr<kNFace> f;
  Kokkos::parallel_reduce("mclose_faces", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*nkji),
  KOKKOS_LAMBDA(const int idx, FixArr<kNFace> &acc) {
    const int m = idx/nkji;
    const int k = (idx - m*nkji)/nji + ks;
    const int j = (idx - m*nkji - (k - ks)*nji)/nx1 + js;
    const int i = idx - m*nkji - (k - ks)*nji - (j - js)*nx1 + is;
    auto &sz = size.d_view(m);
    const Real a1f = sz.dx2*sz.dx3, a2f = sz.dx1*sz.dx3, a3f = sz.dx1*sz.dx2;
    const Real tol = 1.0e-6*sz.dx1;
    if (i == is && sz.x1min <= mx1 + tol) acc.v[kFout].Add(-flx.x1f(m, nv, k, j, i)*a1f);
    if (i == ie && sz.x1max >= Mx1 - tol) acc.v[kFout].Add(flx.x1f(m, nv, k, j, i+1)*a1f);
    if (j == js && sz.x2min <= mx2 + tol) acc.v[kFout].Add(-flx.x2f(m, nv, k, j, i)*a2f);
    if (j == je && sz.x2max >= Mx2 - tol) acc.v[kFout].Add(flx.x2f(m, nv, k, j+1, i)*a2f);
    if (k == ks && sz.x3min <= mx3 + tol) acc.v[kFout].Add(-flx.x3f(m, nv, k, j, i)*a3f);
    if (k == ke && sz.x3max >= Mx3 - tol) acc.v[kFout].Add(flx.x3f(m, nv, k+1, j, i)*a3f);
    if (exc && xf(m, k, j, i)) {
      if (!xf(m, k, j, i-1)) acc.v[kFexc].Add(flx.x1f(m, nv, k, j, i)*a1f);
      if (!xf(m, k, j, i+1)) acc.v[kFexc].Add(-flx.x1f(m, nv, k, j, i+1)*a1f);
      if (!xf(m, k, j-1, i)) acc.v[kFexc].Add(flx.x2f(m, nv, k, j, i)*a2f);
      if (!xf(m, k, j+1, i)) acc.v[kFexc].Add(-flx.x2f(m, nv, k, j+1, i)*a2f);
      if (!xf(m, k-1, j, i)) acc.v[kFexc].Add(flx.x3f(m, nv, k, j, i)*a3f);
      if (!xf(m, k+1, j, i)) acc.v[kFexc].Add(-flx.x3f(m, nv, k+1, j, i)*a3f);
    }
  }, FixArrReducer<kNFace>(f));
  for (int q = 0; q < kNFace; ++q) st_f[stage-1][q] = f.v[q];
  st_bdt[stage-1] = beta[stage-1]*dt;   // applied after the global sum
  if (stage != ns) return;

  // the cycle: gather every stage's sums in one rank-ordered reduction
  const int nper = 1 + kNStage + kNFace;
  std::vector<Fix> loc(1 + ns*nper);
  std::vector<KSum> glb(1 + ns*nper);
  loc[0] = st_u1;
  for (int q = 0; q < ns; ++q) {
    loc[1 + q*nper] = st_pre[q];
    for (int n = 0; n < kNStage; ++n) loc[2 + q*nper + n] = st_c[q][n];
    for (int n = 0; n < kNFace; ++n) loc[2 + kNStage + q*nper + n] = st_f[q][n];
  }
  GlobalSum(loc.data(), static_cast<int>(loc.size()), glb.data());
  for (int q = 0; q < ns; ++q) {   // the face sums x beta dt (tiny: double is ample)
    for (int n = 0; n < kNFace; ++n) {
      KSum &fk = glb[2 + kNStage + q*nper + n];
      fk.s *= st_bdt[q];
      fk.c *= st_bdt[q];
    }
  }
  DD wd[4];
  wd[ns-1] = FromR(1.0);
  for (int q = ns-2; q >= 0; --q) wd[q] = Mul(FromR(gam0[q+1]), wd[q+1]);
  Real w[4];
  for (int q = 0; q < ns; ++q) w[q] = wd[q].hi + wd[q].lo;
  const DD mu1d = FromK(glb[0]);
  const Real mu1 = glb[0].Val();
  KSum cyc[kNItem];
  DD Yd, Xd, mprevd = mu1d;
  for (int q = 0; q < ns; ++q) {
    const KSum *c = &glb[2 + q*nper];
    const KSum *fq = &glb[2 + kNStage + q*nper];
    // the update's change, in double-double; X_s = M_end - M_pre is partitioned by the
    // class sums below (those changes are small, so plain weights suffice for them)
    DD ys = Add(FromK(glb[1 + q*nper]), Neg(Mul(FromR(gam0[q]), mprevd)));
    ys = Add(ys, Neg(Mul(FromR(gam1[q]), mu1d)));
    Yd = Add(Yd, Mul(wd[q], ys));
    // X_s from the masses (debug: the class partition must equal it)
    Xd = Add(Xd, Mul(wd[q], Add(FromK(c[kMend]), Neg(FromK(glb[1 + q*nper])))));
    cyc[iOut].Add(-w[q]*fq[kFout].Val());
    cyc[iFexc].Add(w[q]*fq[kFexc].Val());
    for (int b = 0; b < 5; ++b) cyc[iExc0 + b].Add(w[q]*c[kExc0 + b].Val());
    cyc[iRf].Add(w[q]*c[kRf].Val());
    cyc[iZoneRem].Add(w[q]*c[kZoneRem].Val());
    cyc[iZoneAdd].Add(w[q]*c[kZoneAdd].Val());
    cyc[iFloorStage].Add(w[q]*c[kFloor].Val());
    cyc[iCrec].Add(w[q]*c[kCrec].Val());
    cyc[iOther].Add(w[q]*c[kOther].Val());
    mprevd = FromK(c[kMend]);
  }
  // the RK coefficients' rounding: M_end(ns) - M(u1) = sum_s w_s (Y_s + X_s) + C M(u1),
  // C = w_0 (gam0_0 + gam1_0) + sum_{s>0} w_s gam1_s - 1 (0 in exact arithmetic)
  DD C = Mul(wd[0], Add(FromR(gam0[0]), FromR(gam1[0])));
  for (int q = 1; q < ns; ++q) C = Add(C, Mul(wd[q], FromR(gam1[q])));
  C = Add(C, FromR(-1.0));
  const DD rk = Mul(C, mu1d);
  cyc[iRkCoef].Add(rk.hi);
  cyc[iRkCoef].Add(rk.lo);
  // item 8: the update's non-conservation, Y + F_out (iOut holds -F_out)
  cyc[iUpdate].Add(Yd.hi);
  cyc[iUpdate].Add(Yd.lo);
  cyc[iUpdate].Add(-cyc[iOut].s);
  cyc[iUpdate].Add(-cyc[iOut].c);
  for (int n = 0; n < kNItem; ++n) { seg[n].Join(cyc[n]); cum[n].Join(cyc[n]); }
  // the gap since the previous cycle's end (0 unless something outside the stages and the
  // booked paths changed the mass; it ends up in R)
  DD gap = Add(mu1d, Neg(FromK(m_last_end)));
  const bool had_end = have_last_end;
  m_last_end = glb[2 + (ns-1)*nper + kMend];
  have_last_end = true;
  ++ncyc;
  if (every > 0 && cycle % every == 0 && global_variable::my_rank == 0) {
    KSum xp;   // the class partition of X
    for (int n : {iExc0, iExc1, iExc2, iExc3, iExc4, iRf, iZoneRem, iZoneAdd, iFloorStage,
                  iCrec, iOther}) xp.Join(cyc[n]);
    std::cout << std::setprecision(17) << tag_ << "_CYC cycle=" << cycle << " time=" << time
              << " Mu1=" << mu1 << " Mend=" << m_last_end.Val()
              << " gap=" << (had_end ? gap.hi + gap.lo : 0.0)
              << " Y=" << Yd.hi + Yd.lo << " X=" << Xd.hi + Xd.lo << " Xpart=" << xp.Val()
              << " Mpre-Md0:";
    for (int q = 0; q < ns; ++q) {
      const DD dd = Add(FromK(glb[1 + q*nper]), Neg(FromK(glb[2 + q*nper + kMd0])));
      std::cout << " " << dd.hi + dd.lo;
    }
    std::cout
              << " dcyc=" << (FromK(m_last_end).hi - mu1d.hi) + (FromK(m_last_end).lo - mu1d.lo)
              << " items:";
    for (int n = 0; n < kNItem; ++n) std::cout << " " << cyc[n].Val();
    std::cout << std::endl;
  }
}

void MassClosure::MarkBefore() {
  if (!on) return;
  for (auto &x : extra_) x->MarkBefore();
  m_mark = Mass();
}

void MassClosure::MarkAfter(int item) {
  if (!on) return;
  for (auto &x : extra_) x->MarkAfter(item);
  const KSum a = Mass();
  KSum d;
  d.Add(a.s); d.Add(a.c); d.Add(-m_mark.s); d.Add(-m_mark.c);
  seg[item].Join(d); cum[item].Join(d);
  // a booked change outside the stages (the F1 unmask, the regrid path's C2P) moves the
  // reference the next cycle's start (and a restart's saved M_end) is compared with
  if (have_last_end) m_last_end = a;
}

void MassClosure::RegridProlongated() {
  if (!on) return;
  for (auto &x : extra_) x->RegridProlongated();
  const KSum a = Mass();
  if (have_last_end) {
    KSum d;
    d.Add(a.s); d.Add(a.c); d.Add(-m_last_end.s); d.Add(-m_last_end.c);
    seg[iRegrid].Join(d); cum[iRegrid].Join(d);
  }
  m_mark = a;   // the regrid path's C2P is booked by MarkAfter(iFloorInit)
  if (have_last_end) m_last_end = a;
}

void MassClosure::SegmentStart(ParameterInput *pin, int cycle) {
  if (!on) return;
  m_start = Mass();
  m_last_end = m_start;
  have_last_end = true;
  cycle0 = cycle;
  // the run's cumulative items and M* (the parameter copy of a restart)
  for (int n = 0; n < kNItem; ++n) {
    const std::string k = key_ + "_cum" + std::to_string(n);
    if (pin->DoesParameterExist("mhd", k + "s")) {
      cum[n].s = pin->GetReal("mhd", k + "s");
      cum[n].c = pin->GetReal("mhd", k + "c");
    }
  }
  if (var_ == IDN) {
    if (!(mstar > 0.0)) {
      mstar = pin->DoesParameterExist("mhd", "mass_closure_m0") ?
              pin->GetReal("mhd", "mass_closure_m0") : m_start.Val();
    }
    pin->SetReal("mhd", "mass_closure_m0", mstar);
    // S_i, tau: the same M* (their tolerance is in mass units too)
    for (auto &x : extra_) { x->mstar = mstar; x->SegmentStart(pin, cycle); }
  }
  if (pin->DoesParameterExist("mhd", key_ + "_mend")) {
    m_prev_end = pin->GetReal("mhd", key_ + "_mend");
    const Real pc = pin->DoesParameterExist("mhd", key_ + "_mend_c") ?
                    pin->GetReal("mhd", key_ + "_mend_c") : 0.0;
    KSum d;
    d.Add(m_start.s); d.Add(m_start.c); d.Add(-m_prev_end); d.Add(-pc);
    seg[iRestart].Join(d); cum[iRestart].Join(d);
  }
  if (global_variable::my_rank == 0) {
    std::cout << std::setprecision(17) << tag_ << "_START cycle=" << cycle << " M_start="
              << m_start.Val() << " M_prev_end=" << m_prev_end << std::endl;
  }
}

void MassClosure::SegmentEnd(int cycle, Real time) {
  if (!on) return;
  for (auto &x : extra_) x->SegmentEnd(cycle, time);
  const KSum me = Mass();
  KSum dm;
  dm.Add(me.s); dm.Add(me.c); dm.Add(-m_start.s); dm.Add(-m_start.c);
  // items 1-8 inside the segment (9, restart, is at its start; Fexc is info)
  KSum sum;
  for (int n = 0; n < kNItem; ++n) {
    if (n == iRestart || n == iFexc) continue;
    sum.Join(seg[n]);
  }
  KSum rr = dm;
  rr.Add(-sum.s); rr.Add(-sum.c);
  const Real R = rr.Val(), mg = me.Val();
  const Real eps = std::numeric_limits<Real>::epsilon();
  // the round-off scale (S_i, tau: their net sum can be ~0 or negative, so at least M*)
  const Real scl = (var_ == IDN) ? mg : std::max(std::fabs(mg), mstar);
  const Real rnd = 10.0*eps*scl*static_cast<Real>(ncyc > 0 ? ncyc : 1);
  std::string flags;
  // S_i, tau: crec, the source term (update) and other (tau: the dual-energy resync) are
  // physical changes, not checks
  const int chk[5] = {iRegrid, iRestart, iCrec, iUpdate, iOther};
  const char *chkn[5] = {"regrid", "restart", "crec", "update", "other"};
  const int nchk = (var_ == IDN) ? 5 : 2;
  for (int q = 0; q < nchk; ++q) {
    if (std::fabs(seg[chk[q]].Val()) > rnd) flags += std::string(flags.empty() ? "" : ",")
                                                     + chkn[q];
  }
  const bool pass = std::fabs(R) <= tol*mstar;
  if (global_variable::my_rank != 0) return;
  const char *nm[kNItem] = {"out", "exc<0.59", "exc0.59-0.78", "exc0.78-1.88",
                            "exc1.88-2.92", "exc>=2.92", "rf", "zone_rem", "zone_add",
                            "unmask", "floor_stage", "floor_init", "regrid", "crec",
                            (var_ == IDN) ? "update" : "source", "rk_coef", "restart",
                            "other", "fexc_info"};
  std::ostringstream o;
  o << std::setprecision(10) << tag_ << " cycles=" << cycle0 << "-" << cycle << " ncyc=" << ncyc
    << " time=" << time << " Mstar=" << mstar << std::setprecision(17) << " M_start="
    << m_start.Val() << " M_end=" << mg << " M_last_end=" << m_last_end.Val()
    << std::setprecision(10) << " dM=" << dm.Val();
  for (int n = 0; n < kNItem; ++n) o << " " << nm[n] << "=" << seg[n].Val();
  o << " R=" << R << " R/M*=" << R/mstar << " R/M_grid=" << R/mg << " roundoff=" << rnd
    << " " << (pass ? "PASS" : "FAIL") << " flags=" << (flags.empty() ? "none" : flags);
  std::cout << o.str() << std::endl;
  std::ostringstream c;
  c << std::setprecision(10) << tag_ << "_RUN cycle=" << cycle << " (cumulative since the run's"
    << " start, restarts included)";
  for (int n = 0; n < kNItem; ++n) c << " " << nm[n] << "=" << cum[n].Val();
  std::cout << c.str() << std::endl;
}

void MassClosure::Save(ParameterInput *pin) const {
  if (!on) return;
  for (auto &x : extra_) x->Save(pin);
  if (have_last_end) {   // both parts: the value is exact beyond a double (Fix sums)
    pin->SetReal("mhd", key_ + "_mend", m_last_end.s);
    pin->SetReal("mhd", key_ + "_mend_c", m_last_end.c);
  }
  for (int n = 0; n < kNItem; ++n) {
    const std::string k = key_ + "_cum" + std::to_string(n);
    pin->SetReal("mhd", k + "s", cum[n].s);
    pin->SetReal("mhd", k + "c", cum[n].c);
  }
  if (var_ == IDN) pin->SetReal("mhd", "mass_closure_m0", mstar);
}

}  // namespace mclose
