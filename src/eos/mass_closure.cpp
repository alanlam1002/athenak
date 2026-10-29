//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mass_closure.cpp
//! \brief see mass_closure.hpp. Compiled with -fp-model=precise (src/CMakeLists.txt).

#include <algorithm>
#include <cstdlib>
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
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "cfc/cfc.hpp"

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
    // tracer_closure: the DY closure is created at the first SegmentStart (the MHD's
    // variable count and the dual-energy diagnostics exist by then)
    ytr_req_ = on && pin->GetOrAddBoolean("mhd", "tracer_closure", false);
    if (on && pin->GetOrAddBoolean("mhd", "mass_closure_full", false)) {
      for (int v : {static_cast<int>(IM1), static_cast<int>(IM2), static_cast<int>(IM3),
                    static_cast<int>(IEN)}) {
        extra_.emplace_back(new MassClosure(pp, pin, v));
      }
    }
  } else if (var_ == IEN) {
    tag_ = "ECLOSE"; key_ = "closure_tau";
    // <mhd> energy_variable (R-155 item 1): close T instead (mass_closure.hpp)
    const std::string ev = pin->DoesParameterExist("mhd", "energy_variable") ?
                           pin->GetString("mhd", "energy_variable") : "valencia";
    kill_ = (ev == "killing_t1") ? 1 : ((ev == "killing_ta") ? 2 : 0);
    // <mhd> killing_closure = t1 | ta (diagnostic, valencia only): the same T closure for a
    // Valencia run, its face flux formed from the stored D, S, tau fluxes (as the killing
    // options' KillingFaceFluxes), so the two formulations' T drifts are booked alike
    if (pin->DoesParameterExist("mhd", "killing_closure")) {
      const std::string kc = pin->GetString("mhd", "killing_closure");
      const int k = (kc == "t1") ? 1 : ((kc == "ta") ? 2 : 0);
      if (k == 0 || (kill_ != 0 && k != kill_)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "<mhd> killing_closure = '" << kc << "' (t1 or ta; with energy_variable "
                  << ev << ")" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      kill_face_ = (kill_ == 0);
      kill_ = k;
    }
    if (kill_) {
      // <mhd> killing_weights = boost (R-158): (alpha0, beta0) = (1, xidot)
      kboost_ = pin->DoesParameterExist("mhd", "killing_weights") &&
                pin->GetString("mhd", "killing_weights") == "boost";
      tag_ = "KCLOSE"; key_ = "closure_kill";
      dyngr::BuildTrumpetTable(ktab_, pin->GetOrAddReal("dyn_grmhd", "metric_table_wmax", 64.0),
                               pin->GetOrAddReal("dyn_grmhd", "metric_table_h", 0.0025));
    }
  } else {
    tag_ = "SCLOSE" + std::to_string(var_ - IM1 + 1);
    key_ = "closure_s" + std::to_string(var_ - IM1 + 1);
  }
}

int MassClosure::Phase(Real t) const {
  int p = 0;
  for (int q = 0; q < kYPh - 1; ++q) p += (t >= y_ph[q]);
  return p;
}

//! tracer_closure: this stage's C2P change of DY per (band, category), on this rank
void MassClosure::YStage(int stage, const DvceArray4D<Real> &cls,
                         const DvceArray4D<Real> &c2p) {
  MeshBlockPack *pp = pmy_pack;
  const int nv = var_;
  auto &indcs = pp->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmb = pp->nmb_thispack, nkji = nx3*nx2*nx1, nji = nx2*nx1;
  if (y_cat_.extent_int(0) != nmb || y_cat_.extent_int(1) != nx3 ||
      y_cat_.extent_int(2) != nx2 || y_cat_.extent_int(3) != nx1) {
    Kokkos::realloc(y_cat_, nmb, nx3, nx2, nx1);
    Kokkos::realloc(y_val_, nmb, nx3, nx2, nx1);
  }
  auto &size = pp->pmb->mb_size;
  auto &u0 = pp->pmhd->u0;
  auto &w0 = pp->pmhd->w0;
  auto &adm = pp->padm->adm;
  auto &dd = pp->pdyngr->de_diag;
  const DvceArray4D<Real> x0 = x0_;
  auto &cd = pp->pcoord->coord_data;
  const bool exdyn = cd.bh_excise && cd.excision_dynamic &&
                     cd.excision_scheme == ExcisionScheme::lapse;
  const Real exlap = cd.excise_lapse;
  const Real irc = 1.0/y_rhoc;
  // |V.n|_coord bin edges (kYV = 7 bins)
  const Real e0 = 0.003, e1 = 0.01, e2 = 0.03, e3 = 0.1, e4 = 0.2, e5 = 0.3;
  auto &cat = y_cat_;
  auto &val = y_val_;
  par_for("yclose_cat", DevExeSpace(), 0, nmb-1, 0, nx3-1, 0, nx2-1, 0, nx1-1,
  KOKKOS_LAMBDA(const int m, const int kk, const int jj, const int ii) {
    const int k = kk + ks, j = jj + js, i = ii + is;
    auto &sz = size.d_view(m);
    const Real dv = sz.dx1*sz.dx2*sz.dx3;
    const Real dm = u0(m, nv, k, j, i)*dv - x0(m, k, j, i)*dv;
    val(m, kk, jj, ii) = dm;
    if (dm == 0.0) { cat(m, kk, jj, ii) = -1; return; }
    const Real rr = w0(m, IDN, k, j, i)*irc;
    const int band = (rr < 1.0e-1) + (rr < 1.0e-2) + (rr < 1.0e-3);
    const Real c = cls(m, k, j, i);
    int q;
    if (c < 1.5) {
      q = (exdyn && !(adm.alpha(m, k, j, i) < exlap)) ? yRf : yExc;
    } else if (c < 2.5) {
      q = (dm < 0.0) ? yZoneRem : yZoneAdd;
    } else if (static_cast<int>(c2p(m, k, j, i)) & 256) {
      // cold recovery before the floor class: ColdRecoveryTry needs prim_floor, so every
      // recovered cell is also class 3 (primitive_solver_hyd.hpp ColdRecoveryTry)
      q = yCrec;
    } else if (c < 3.5) {
      q = yFloor;
    } else {
      const Real dbr = dd(m, 0, k, j, i);
      if (dbr > 3.5 && dbr < 4.5) {
        // the gas's normal grid speed |(alpha v^i - beta^i) n_i|, n = -grad P/|grad P|
        const Real gx = (w0(m, IPR, k, j, i+1) - w0(m, IPR, k, j, i-1))/(2.0*sz.dx1);
        const Real gy = (nx2 > 1) ?
                        (w0(m, IPR, k, j+1, i) - w0(m, IPR, k, j-1, i))/(2.0*sz.dx2) : 0.0;
        const Real gz = (nx3 > 1) ?
                        (w0(m, IPR, k+1, j, i) - w0(m, IPR, k-1, j, i))/(2.0*sz.dx3) : 0.0;
        const Real gn = sqrt(gx*gx + gy*gy + gz*gz);
        const Real ux = w0(m, IVX, k, j, i), uy = w0(m, IVY, k, j, i);
        const Real uz = w0(m, IVZ, k, j, i);
        const Real uu = adm.g_dd(m, 0, 0, k, j, i)*ux*ux + adm.g_dd(m, 1, 1, k, j, i)*uy*uy
                      + adm.g_dd(m, 2, 2, k, j, i)*uz*uz
                      + 2.0*(adm.g_dd(m, 0, 1, k, j, i)*ux*uy + adm.g_dd(m, 0, 2, k, j, i)*ux*uz
                             + adm.g_dd(m, 1, 2, k, j, i)*uy*uz);
        const Real iw = 1.0/sqrt(1.0 + uu);
        const Real al = adm.alpha(m, k, j, i);
        const Real vx = al*ux*iw - adm.beta_u(m, 0, k, j, i);
        const Real vy = al*uy*iw - adm.beta_u(m, 1, k, j, i);
        const Real vz = al*uz*iw - adm.beta_u(m, 2, k, j, i);
        const Real vn = (gn > 0.0) ? fabs(vx*gx + vy*gy + vz*gz)/gn : 0.0;
        const int vb = (vn >= e0) + (vn >= e1) + (vn >= e2) + (vn >= e3) + (vn >= e4)
                       + (vn >= e5);
        q = ((dm > 0.0) ? yUp0 : yDn0) + vb;
      } else if (dbr < 0.5) {
        q = yNonDense;
      } else {
        q = yDenseChk;
      }
    }
    cat(m, kk, jj, ii) = band*kYCat + q;
  });
  for (int g = 0; g < kYNG; ++g) {
    const int q0 = g*kYG;
    FixArr<kYG> r;
    Kokkos::parallel_reduce("yclose_sum", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*nkji),
    KOKKOS_LAMBDA(const int idx, FixArr<kYG> &acc) {
      const int m = idx/nkji;
      const int kk = (idx - m*nkji)/nji;
      const int jj = (idx - m*nkji - kk*nji)/nx1;
      const int ii = idx - m*nkji - kk*nji - jj*nx1;
      const int q = cat(m, kk, jj, ii) - q0;
      if (q >= 0 && q < kYG) acc.v[q].Add(val(m, kk, jj, ii));
    }, FixArrReducer<kYG>(r));
    for (int q = 0; q < kYG && q0 + q < kYN; ++q) y_st[stage-1][q0 + q] = r.v[q];
  }
}

//! energy_variable != valencia: T = alpha0 (tau + D) - beta0^k S_k - w D of u (densitized,
//! every cell) into kv_, with the weights of dyn_grmhd_killing.cpp (the trumpet table at the
//! cell centre, relative to the background centre CFC::bg_xbh_); otherwise u itself
const DvceArray5D<Real> &MassClosure::KSrc(const DvceArray5D<Real> &u) const {
  if (!kill_) return u;
  if (kv_.extent_int(0) != u.extent_int(0) || kv_.extent_int(2) != u.extent_int(2) ||
      kv_.extent_int(3) != u.extent_int(3) || kv_.extent_int(4) != u.extent_int(4)) {
    kv_ = DvceArray5D<Real>("kclose_t", u.extent_int(0), 1, u.extent_int(2),
                            u.extent_int(3), u.extent_int(4));
  }
  auto *pcfc = pmy_pack->pcfc;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  auto &size = pmy_pack->pmb->mb_size;
  const Real cx = pcfc->bg_xbh_[0], cy = pcfc->bg_xbh_[1], cz = pcfc->bg_xbh_[2];
  const Real m_bh = pcfc->puncture_mass_;
  const bool ta = (kill_ == 2), kb = kboost_;
  const Real fv0 = pmy_pack->padm->frame_vel_u[0], fv1 = pmy_pack->padm->frame_vel_u[1];
  const Real fv2 = pmy_pack->padm->frame_vel_u[2];
  auto &td = ktab_.d;
  const Real th = ktab_.h, tih = ktab_.ih;
  const int nt = ktab_.n;
  auto kv = kv_;
  par_for("kclose_t", DevExeSpace(), 0, pmy_pack->nmb_thispack - 1, 0, u.extent_int(2) - 1,
          0, u.extent_int(3) - 1, 0, u.extent_int(4) - 1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const auto &sz = size.d_view(m);
    const Real x = CellCenterX(i - is, nx1, sz.x1min, sz.x1max) - cx;
    const Real y = CellCenterX(j - js, nx2, sz.x2min, sz.x2max) - cy;
    const Real z = CellCenterX(k - ks, nx3, sz.x3min, sz.x3max) - cz;
    Real psi0 = 1.0, ap0 = 1.0, b0[3] = {fv0, fv1, fv2}, dpsi0, dal0, b, db;
    if (!kb) {
      dyngr::TrumpetTableBackground(td, th, tih, nt, m_bh, x, y, z, psi0, ap0, b0, dpsi0,
                                    dal0, b, db);
    }
    const Real al0 = ap0/psi0;
    const Real d = u(m, IDN, k, j, i);
    kv(m, 0, k, j, i) = al0*(u(m, IEN, k, j, i) + d)
                        - (b0[0]*u(m, IM1, k, j, i) + b0[1]*u(m, IM2, k, j, i)
                           + b0[2]*u(m, IM3, k, j, i))
                        - (ta ? al0 : 1.0)*d;
  });
  return kv_;
}

KSum MassClosure::Mass() const {
  Fix loc = LocalMass(pmy_pack, KSrc(pmy_pack->pmhd->u0), KVar());
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
    Kokkos::deep_copy(DevExeSpace(), x0_, Kokkos::subview(KSrc(u0), Kokkos::ALL, KVar(),
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
  }
  st_pre[stage-1] = LocalMass(pmy_pack, KSrc(u0), KVar());
  if (stage == 1) st_u1 = LocalMass(pmy_pack, KSrc(pmy_pack->pmhd->u1), KVar());
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
  const DvceArray5D<Real> u0 = KSrc(pp->pmhd->u0);   // (KCLOSE: T, one variable)
  const int nu = KVar();
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
    acc.v[kMend].Add(u0(m, nu, k, j, i)*dv);
    acc.v[kMd0].Add(x0(m, k, j, i)*dv);
    const Real dm = u0(m, nu, k, j, i)*dv - x0(m, k, j, i)*dv;
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
    } else if (static_cast<int>(c2p(m, k, j, i)) & 256) {   // before class 3 (as YCLOSE)
      acc.v[kCrec].Add(dm);
    } else if (c < 3.5) {
      acc.v[kFloor].Add(dm);
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
  // KCLOSE of a Valencia run: T's face flux from the D, S, tau fluxes (weights at the face)
  const bool kf = kill_face_, kta = (kill_ == 2), kb = kboost_;
  const Real kv0 = pp->padm->frame_vel_u[0], kv1 = pp->padm->frame_vel_u[1];
  const Real kv2 = pp->padm->frame_vel_u[2];
  const Real kcx = kf ? pp->pcfc->bg_xbh_[0] : 0.0, kcy = kf ? pp->pcfc->bg_xbh_[1] : 0.0;
  const Real kcz = kf ? pp->pcfc->bg_xbh_[2] : 0.0, kmbh = kf ? pp->pcfc->puncture_mass_ : 1.0;
  auto &ktd = ktab_.d;
  const Real kth = ktab_.h, ktih = ktab_.ih;
  const int knt = ktab_.n;
  auto fT = [=] KOKKOS_FUNCTION (const DvceArray5D<Real> &fl, const int m, const int k,
                                 const int j, const int i, const Real x, const Real y,
                                 const Real z) {
    if (!kf) return fl(m, nv, k, j, i);
    Real psi0 = 1.0, ap0 = 1.0, b0[3] = {kv0, kv1, kv2}, dpsi0, dal0, b, db;
    if (!kb) {
      dyngr::TrumpetTableBackground(ktd, kth, ktih, knt, kmbh, x - kcx, y - kcy, z - kcz,
                                    psi0, ap0, b0, dpsi0, dal0, b, db);
    }
    const Real al0 = ap0/psi0, fd = fl(m, IDN, k, j, i);
    return al0*(fl(m, IEN, k, j, i) + fd) - (b0[0]*fl(m, IM1, k, j, i)
           + b0[1]*fl(m, IM2, k, j, i) + b0[2]*fl(m, IM3, k, j, i)) - (kta ? al0 : 1.0)*fd;
  };
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
    // the cell's centre and its six faces (KCLOSE of a Valencia run needs the face positions)
    const Real xc = CellCenterX(i - is, nx1, sz.x1min, sz.x1max);
    const Real yc = CellCenterX(j - js, nx2, sz.x2min, sz.x2max);
    const Real zc = CellCenterX(k - ks, nx3, sz.x3min, sz.x3max);
    const Real xl = LeftEdgeX(i - is, nx1, sz.x1min, sz.x1max);
    const Real xr = LeftEdgeX(i + 1 - is, nx1, sz.x1min, sz.x1max);
    const Real yl = LeftEdgeX(j - js, nx2, sz.x2min, sz.x2max);
    const Real yr = LeftEdgeX(j + 1 - js, nx2, sz.x2min, sz.x2max);
    const Real zl = LeftEdgeX(k - ks, nx3, sz.x3min, sz.x3max);
    const Real zr = LeftEdgeX(k + 1 - ks, nx3, sz.x3min, sz.x3max);
    const auto &f1 = flx.x1f, &f2 = flx.x2f, &f3 = flx.x3f;
    if (i == is && sz.x1min <= mx1 + tol) acc.v[kFout].Add(-fT(f1, m, k, j, i, xl, yc, zc)*a1f);
    if (i == ie && sz.x1max >= Mx1 - tol) acc.v[kFout].Add(fT(f1, m, k, j, i+1, xr, yc, zc)*a1f);
    if (j == js && sz.x2min <= mx2 + tol) acc.v[kFout].Add(-fT(f2, m, k, j, i, xc, yl, zc)*a2f);
    if (j == je && sz.x2max >= Mx2 - tol) acc.v[kFout].Add(fT(f2, m, k, j+1, i, xc, yr, zc)*a2f);
    if (k == ks && sz.x3min <= mx3 + tol) acc.v[kFout].Add(-fT(f3, m, k, j, i, xc, yc, zl)*a3f);
    if (k == ke && sz.x3max >= Mx3 - tol) acc.v[kFout].Add(fT(f3, m, k+1, j, i, xc, yc, zr)*a3f);
    if (exc && xf(m, k, j, i)) {
      if (!xf(m, k, j, i-1)) acc.v[kFexc].Add(fT(f1, m, k, j, i, xl, yc, zc)*a1f);
      if (!xf(m, k, j, i+1)) acc.v[kFexc].Add(-fT(f1, m, k, j, i+1, xr, yc, zc)*a1f);
      if (!xf(m, k, j-1, i)) acc.v[kFexc].Add(fT(f2, m, k, j, i, xc, yl, zc)*a2f);
      if (!xf(m, k, j+1, i)) acc.v[kFexc].Add(-fT(f2, m, k, j+1, i, xc, yr, zc)*a2f);
      if (!xf(m, k-1, j, i)) acc.v[kFexc].Add(fT(f3, m, k, j, i, xc, yc, zl)*a3f);
      if (!xf(m, k+1, j, i)) acc.v[kFexc].Add(-fT(f3, m, k+1, j, i, xc, yc, zr)*a3f);
    }
  }, FixArrReducer<kNFace>(f));
  for (int q = 0; q < kNFace; ++q) st_f[stage-1][q] = f.v[q];
  st_bdt[stage-1] = beta[stage-1]*dt;   // applied after the global sum
  if (ytr_) { YStage(stage, cls, c2p); y_tlast = time; }
  if (stage != ns) return;

  // the cycle: gather every stage's sums in one rank-ordered reduction
  const int nper = 1 + kNStage + kNFace;
  const int ny0 = 1 + ns*nper, ny = ytr_ ? ns*kYN : 0;   // tracer_closure: the split
  std::vector<Fix> loc(ny0 + ny);
  std::vector<KSum> glb(ny0 + ny);
  loc[0] = st_u1;
  for (int q = 0; q < ns; ++q) {
    loc[1 + q*nper] = st_pre[q];
    for (int n = 0; n < kNStage; ++n) loc[2 + q*nper + n] = st_c[q][n];
    for (int n = 0; n < kNFace; ++n) loc[2 + kNStage + q*nper + n] = st_f[q][n];
  }
  for (int q = 0; q < ns && ytr_; ++q) {
    for (int n = 0; n < kYN; ++n) loc[ny0 + q*kYN + n] = y_st[q][n];
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
  if (ytr_) {   // the split, RK-weighted like the class items; the cycle's phase
    const int ph = Phase(time);
    for (int n = 0; n < kNItem; ++n) phcum[ph][n].Join(cyc[n]);
    for (int n = 0; n < kYN; ++n) {
      KSum yc;
      for (int q = 0; q < ns; ++q) yc.Add(w[q]*glb[ny0 + q*kYN + n].Val());
      yseg[n].Join(yc); ycum[ph][n].Join(yc);
    }
  }
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
  if (ytr_) phcum[Phase(y_tlast)][item].Join(d);
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
    if (ytr_) phcum[Phase(y_tlast)][iRegrid].Join(d);
  }
  m_mark = a;   // the regrid path's C2P is booked by MarkAfter(iFloorInit)
  if (have_last_end) m_last_end = a;
}

void MassClosure::SegmentStart(ParameterInput *pin, int cycle) {
  if (!on) return;
  if (ytr_req_) {
    ytr_req_ = false;
    auto *pm = pmy_pack->pmhd;
    auto *pdg = pmy_pack->pdyngr;
    if (pdg == nullptr || pm == nullptr || pm->nscalars < 1 || pdg->de_diag.extent_int(1) < 2) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "<mhd> tracer_closure needs dual_energy and dual_energy_diag" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    // the K tracer is the last scalar (PrimitiveSolverHydro: de_idx = nscalars - 1)
    MassClosure *y = new MassClosure(pmy_pack, pin, pm->nmhd + pm->nscalars - 1);
    y->ytr_ = true; y->tag_ = "YCLOSE"; y->key_ = "closure_y";
    y->y_rhoc = pin->GetOrAddReal("mhd", "tracer_closure_rhoc", 3.24e-4);
    std::istringstream ss(pin->GetOrAddString("mhd", "tracer_closure_phase_t",
                                              "125.9,145.9,185.9"));
    std::string tok;
    for (int q = 0; q < kYPh - 1 && std::getline(ss, tok, ','); ++q) y->y_ph[q] = std::stod(tok);
    extra_.emplace_back(y);
  }
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
  y_tlast = pmy_pack->pmesh->time;
  for (int p = 0; p < kYPh && ytr_; ++p) {
    for (int n = 0; n < kNItem; ++n) {
      const std::string k = key_ + "_ph" + std::to_string(p) + "_" + std::to_string(n);
      if (pin->DoesParameterExist("mhd", k + "s")) {
        phcum[p][n].s = pin->GetReal("mhd", k + "s");
        phcum[p][n].c = pin->GetReal("mhd", k + "c");
      }
    }
    for (int n = 0; n < kYN; ++n) {
      const std::string k = key_ + "_yc" + std::to_string(p) + "_" + std::to_string(n);
      if (pin->DoesParameterExist("mhd", k + "s")) {
        ycum[p][n].s = pin->GetReal("mhd", k + "s");
        ycum[p][n].c = pin->GetReal("mhd", k + "c");
      }
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
  if (pin->DoesParameterExist("mhd", key_ + "_embed_cums")) {
    embed_cum_.s = pin->GetReal("mhd", key_ + "_embed_cums");
    embed_cum_.c = pin->GetReal("mhd", key_ + "_embed_cumc");
  }
  if (pin->DoesParameterExist("mhd", key_ + "_mend")) {
    m_prev_end = pin->GetReal("mhd", key_ + "_mend");
    const Real pc = pin->DoesParameterExist("mhd", key_ + "_mend_c") ?
                    pin->GetReal("mhd", key_ + "_mend_c") : 0.0;
    KSum d;
    d.Add(m_start.s); d.Add(m_start.c); d.Add(-m_prev_end); d.Add(-pc);
    if (embed_) {
      // R1: the load changed the grid outside the stages; e0 (old domain, new blocks zero)
      // against the saved M_end is the bitwise check, e1 - e0 the atmosphere added, e2 - e1
      // the load's C2P; their sum (e2 - e0) is regrid_embed, the rest stays the restart item
      auto diff = [](const KSum &a, const KSum &b, Real bc = 0.0) {
        KSum r; r.Add(a.s); r.Add(a.c); r.Add(-b.s); r.Add(-b.c); r.Add(-bc); return r;
      };
      KSum old0;
      old0.Add(e_mark_[0].s); old0.Add(e_mark_[0].c); old0.Add(-m_prev_end); old0.Add(-pc);
      const KSum atm = diff(e_mark_[1], e_mark_[0]), lc2p = diff(e_mark_[2], e_mark_[1]);
      const KSum emb = diff(e_mark_[2], e_mark_[0]);
      d.Add(-emb.s); d.Add(-emb.c);
      embed_cum_.Join(emb);
      if (global_variable::my_rank == 0) {
        std::cout << std::setprecision(17) << tag_ << "_EMBED cycle=" << cycle
                  << " old_domain_minus_prev_end=" << old0.Val() << " atmosphere="
                  << atm.Val() << " load_c2p=" << lc2p.Val() << " regrid_embed="
                  << emb.Val() << " start_minus_e2=" << diff(m_start, e_mark_[2]).Val()
                  << " regrid_embed_cum=" << embed_cum_.Val() << std::endl;
      }
      embed_ = false;
    }
    seg[iRestart].Join(d); cum[iRestart].Join(d);
    if (ytr_) phcum[Phase(y_tlast)][iRestart].Join(d);
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
  if (!ytr_) return;
  // the split's partition checks for this segment: each C2P item against its band/category sum
  const char *ynm[8] = {"exc", "rf", "zone_rem", "zone_add", "floor", "crec", "nondense",
                        "dense_chk"};
  auto ysum = [&](const KSum *a, int q0, int q1) {
    KSum t;
    for (int b = 0; b < kYBand; ++b) for (int q = q0; q < q1; ++q) t.Join(a[b*kYCat + q]);
    return t;
  };
  KSum ex5;
  for (int n = iExc0; n <= iExc4; ++n) ex5.Join(seg[n]);
  KSum oth = ysum(yseg, yNonDense, kYCat);
  std::ostringstream y;
  y << std::setprecision(10) << "YCLOSE_SPLIT cycles=" << cycle0 << "-" << cycle
    << " exc=" << ex5.Val() << " vs " << ysum(yseg, yExc, yExc + 1).Val()
    << " rf=" << seg[iRf].Val() << " vs " << ysum(yseg, yRf, yRf + 1).Val()
    << " zone=" << seg[iZoneRem].Val() + seg[iZoneAdd].Val() << " vs "
    << ysum(yseg, yZoneRem, yZoneAdd + 1).Val()
    << " floor=" << seg[iFloorStage].Val() << " vs " << ysum(yseg, yFloor, yFloor + 1).Val()
    << " crec=" << seg[iCrec].Val() << " vs " << ysum(yseg, yCrec, yCrec + 1).Val()
    << " other=" << seg[iOther].Val() << " vs " << oth.Val()
    << " flagged_up=" << ysum(yseg, yUp0, yUp0 + kYV).Val()
    << " flagged_dn=" << ysum(yseg, yDn0, yDn0 + kYV).Val()
    << " nondense=" << ysum(yseg, yNonDense, yNonDense + 1).Val()
    << " dense_chk=" << ysum(yseg, yDenseChk, yDenseChk + 1).Val();
  std::cout << y.str() << std::endl;
  for (int p = 0; p < kYPh; ++p) {
    std::ostringstream o2;
    o2 << std::setprecision(10) << "YCLOSE_PH_RUN cycle=" << cycle << " ph=" << p << " t=["
       << ((p > 0) ? y_ph[p-1] : -1.0) << "," << ((p < kYPh - 1) ? y_ph[p] : 1.0e30) << ")";
    for (int n = 0; n < kNItem; ++n) o2 << " " << nm[n] << "=" << phcum[p][n].Val();
    std::cout << o2.str() << std::endl;
    for (int b = 0; b < kYBand; ++b) {
      const KSum *a = &ycum[p][b*kYCat];
      std::ostringstream o3;
      o3 << std::setprecision(10) << "YCLOSE_BAND_RUN cycle=" << cycle << " ph=" << p
         << " band=" << b;
      for (int q = 0; q < 8; ++q) o3 << " " << ynm[q] << "=" << a[q].Val();
      o3 << " up=";
      for (int v = 0; v < kYV; ++v) o3 << (v ? "," : "") << a[yUp0 + v].Val();
      o3 << " dn=";
      for (int v = 0; v < kYV; ++v) o3 << (v ? "," : "") << a[yDn0 + v].Val();
      std::cout << o3.str() << std::endl;
    }
  }
}

void MassClosure::EmbedMark(int k) {
  if (!on) return;
  for (auto &x : extra_) x->EmbedMark(k);
  e_mark_[k] = Mass();
  if (k == 2) embed_ = true;
}

void MassClosure::Save(ParameterInput *pin) const {
  if (!on) return;
  for (auto &x : extra_) x->Save(pin);
  if (embed_cum_.s != 0.0 || embed_cum_.c != 0.0) {
    pin->SetReal("mhd", key_ + "_embed_cums", embed_cum_.s);
    pin->SetReal("mhd", key_ + "_embed_cumc", embed_cum_.c);
  }
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
  for (int p = 0; p < kYPh && ytr_; ++p) {
    for (int n = 0; n < kNItem; ++n) {
      const std::string k = key_ + "_ph" + std::to_string(p) + "_" + std::to_string(n);
      pin->SetReal("mhd", k + "s", phcum[p][n].s);
      pin->SetReal("mhd", k + "c", phcum[p][n].c);
    }
    for (int n = 0; n < kYN; ++n) {
      const std::string k = key_ + "_yc" + std::to_string(p) + "_" + std::to_string(n);
      pin->SetReal("mhd", k + "s", ycum[p][n].s);
      pin->SetReal("mhd", k + "c", ycum[p][n].c);
    }
  }
}

}  // namespace mclose
