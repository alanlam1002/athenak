//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mg_kerr_vector.cpp
//! \brief W3c: the Kerr vector multigrid (see mg_kerr_vector.hpp).

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "multigrid/multigrid.hpp"
#include "kerr_mg_ops.hpp"
#include "kerr_mg_vops.hpp"
#include "mg_kerr_vector.hpp"

namespace cfc {

namespace {

struct OctAccV {
  MGOctet *o;
  inline Real U(int v, int k, int j, int i) const { return o->U(v, k, j, i); }
  inline Real C(int q, int k, int j, int i) const { return o->Coeff(q, k, j, i); }
};

[[noreturn]] void Fatal(const std::string &msg) {
  std::cout << "### FATAL ERROR in MGKerrVectorDriver" << std::endl << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

//! the class of interior cell I (as in mg_kerr_scalar.cpp)
int CellClass(int msk, int phy, const int I[3], const int n[3]) {
  int cls = 0;
  for (int q = 0; q < 12; ++q) {
    if (!((msk >> q) & 1)) continue;
    int e, a, b, sa, sb;
    kmg::EdgeAxes(q, e, a, b, sa, sb);
    const int ca = (sa > 0) ? n[a] - 1 : 0, cb = (sb > 0) ? n[b] - 1 : 0;
    if (I[a] == ca && I[b] == cb) { cls = std::max(cls, ((phy >> q) & 1) ? 1 : 2); }
  }
  return cls;
}

//! one point-block update at an octet cell: V += omega h^2 Dinv (src + S/h^2)
void OctPoint(MGOctet &oct, int k, int j, int i, Real h, Real omega, Real nw[3]) {
  OctAccV A{&oct};
  Real S[3], Di[9];
  kmg::VecS(A, k, j, i, h, S);
  kmg::VecDinv(A, k, j, i, h, Di);
  const Real ih2 = 1.0/(h*h);
  Real r[3];
  for (int a = 0; a < 3; ++a) { r[a] = oct.Src(a, k, j, i) + S[a]*ih2; }
  for (int a = 0; a < 3; ++a) {
    nw[a] = oct.U(a, k, j, i)
          + omega*h*h*(Di[3*a]*r[0] + Di[3*a + 1]*r[1] + Di[3*a + 2]*r[2]);
  }
}

}  // namespace

//----------------------------------------------------------------------------------------
// MGKerrVector

MGKerrVector::MGKerrVector(MultigridDriver *pmd, MeshBlockPack *pmbp, int nghost,
                           bool on_host)
    : Multigrid(pmd, pmbp, nghost, on_host) {
  ncoeff_ = kmg::kNCoeffVec;
  for (int l = 0; l < nlevel_; l++) {
    int ll = nlevel_-1-l;
    Kokkos::realloc(coeff_[l], nmmb_, ncoeff_, (indcs_.nx3>>ll)+2*ngh_,
                    (indcs_.nx2>>ll)+2*ngh_, (indcs_.nx1>>ll)+2*ngh_);
  }
  Kokkos::realloc(scratch_, nmmb_, 3, indcs_.nx3+2*ngh_, indcs_.nx2+2*ngh_,
                  indcs_.nx1+2*ngh_);
}

MGKerrVectorDriver *MGKerrVector::Drv() const {
  return static_cast<MGKerrVectorDriver*>(pmy_driver_);
}

Real MGKerrVector::LevelDx(int lev, int m) const {
  const int ll = nlevel_ - 1 - lev;
  return block_rdx_.h_view(m)*static_cast<Real>(1 << ll);
}

void MGKerrVector::BuildExtents() {
  Kokkos::realloc(blk_, nmmb_);
  if (pmy_pack_ != nullptr) {
    auto &mb_size = pmy_pack_->pmb->mb_size;
    for (int m = 0; m < nmmb_; ++m) { blk_.h_view(m) = mb_size.h_view(m); }
  } else {
    blk_.h_view(0) = pmy_mesh_->mesh_size;
  }
  blk_.template modify<HostExeSpace>();
  blk_.template sync<DevExeSpace>();
}

long MGKerrVector::FillAllLevels(const kmg::CoefSource &cs) {
  BuildExtents();
  if (static_cast<int>(scratch_.extent(0)) != nmmb_) {
    Kokkos::realloc(scratch_, nmmb_, 3, indcs_.nx3+2*ngh_, indcs_.nx2+2*ngh_,
                    indcs_.nx1+2*ngh_);
  }
  long bad = 0;
  for (int lev = 0; lev < nlevel_; ++lev) {
    const int nx = LevelN(lev, 0), ny = LevelN(lev, 1), nz = LevelN(lev, 2);
    if (static_cast<int>(coeff_[lev].d_view.extent(0)) != nmmb_) {
      Kokkos::realloc(coeff_[lev], nmmb_, ncoeff_, nz+2*ngh_, ny+2*ngh_, nx+2*ngh_);
    }
    bad += kmg::FillLevelVecCoeffs(coeff_[lev].d_view, blk_, nmmb_, ngh_, nx, ny, nz, cs);
  }
  return bad;
}

void MGKerrVector::Hook(int lev, bool poison_ok) {
  const MGKerrVectorDriver *d = Drv();
  const int nx = LevelN(lev, 0), ny = LevelN(lev, 1), nz = LevelN(lev, 2);
  auto u = u_[lev].d_view;
  if (d->corner_poison_) { kmg::PoisonCorners(u, nmmb_, nvar_, ngh_, nx, ny, nz); }
  if (d->kelvin_) {
    kmg::KelvinFaceFill(u, blk_, kfaces_.d_view, nmmb_, ngh_, nx, ny, nz, d->mpo_,
                        d->mom_);
  }
  kmg::ApplyEdgeFill(u, mask_.d_view, nmmb_, nvar_, ngh_, nx, ny, nz, d->edge_fill_,
                     d->edge_poison_ && poison_ok);
}

void MGKerrVector::SaveGhosts(int lev) {
  if (!Drv()->hook_operator_) return;
  if (static_cast<int>(gsave_.extent(0)) != nmmb_) {
    Kokkos::realloc(gsave_, nmmb_, nvar_, indcs_.nx3+2*ngh_, indcs_.nx2+2*ngh_,
                    indcs_.nx1+2*ngh_);
  }
  kmg::CopyGhostShell(gsave_, u_[lev].d_view, nmmb_, nvar_, ngh_, LevelN(lev, 0),
                      LevelN(lev, 1), LevelN(lev, 2));
}

void MGKerrVector::RestoreGhosts(int lev) {
  if (!Drv()->hook_operator_) return;
  kmg::CopyGhostShell(u_[lev].d_view, gsave_, nmmb_, nvar_, ngh_, LevelN(lev, 0),
                      LevelN(lev, 1), LevelN(lev, 2));
}

void MGKerrVector::SmoothPack(int color) {
  const int lev = current_level_;
  const int ll = nlevel_-1-lev;
  const int nx = LevelN(lev, 0), ny = LevelN(lev, 1), nz = LevelN(lev, 2);
  const int is = ngh_, js = ngh_, ks = ngh_;
  const int c0 = color ^ pmy_driver_->GetCoffset();
  // pbj: one full damped Jacobi sweep per colour pair (the same cell updates as gs8)
  const bool pbj = (Drv()->smoother_ == 2);
  if (pbj && c0 != 0) return;
  const Real omega = pbj ? Drv()->pbj_omega_ : Drv()->omega_;
  const Real fac = static_cast<Real>(1 << ll);
  auto brdx = block_rdx_.d_view;
  auto u = u_[lev].d_view;
  auto src = src_[lev].d_view;
  auto coeff = coeff_[lev].d_view;
  using kmg::kVDinv;
  SaveGhosts(lev);
  Hook(lev, true);
  if (Drv()->smoother_ == 0) {
    for (int sc = 0; sc < 8; ++sc) {
      const int oi = sc & 1, oj = (sc >> 1) & 1, ok = (sc >> 2) & 1;
      if (((oi + oj + ok) & 1) != c0) continue;
      const int ni = (nx - oi + 1)/2, nj = (ny - oj + 1)/2, nk = (nz - ok + 1)/2;
      if (ni <= 0 || nj <= 0 || nk <= 0) continue;
      par_for("MGKerrVector::SmoothGS8", DevExeSpace(), 0, nmmb_-1, 0, nk-1, 0, nj-1,
              0, ni-1,
      KOKKOS_LAMBDA(const int m, const int kk, const int jj, const int ii) {
        const int k = ks + ok + 2*kk, j = js + oj + 2*jj, i = is + oi + 2*ii;
        const Real h = brdx(m)*fac;
        const Real ih2 = 1.0/(h*h);
        kmg::VViewAcc<decltype(u)> A{u, coeff, m};
        Real S[3], r[3];
        kmg::VecS(A, k, j, i, h, S);
        for (int a = 0; a < 3; ++a) { r[a] = src(m, a, k, j, i) + S[a]*ih2; }
        Real dv[3];
        for (int a = 0; a < 3; ++a) {
          dv[a] = omega*h*h*(coeff(m, kVDinv + 3*a, k, j, i)*r[0]
                           + coeff(m, kVDinv + 3*a + 1, k, j, i)*r[1]
                           + coeff(m, kVDinv + 3*a + 2, k, j, i)*r[2]);
        }
        for (int a = 0; a < 3; ++a) { u(m, a, k, j, i) += dv[a]; }
      });
      Hook(lev, false);
    }
  } else {
    auto scr = scratch_;
    const int ie = is + nx - 1, je = js + ny - 1, ke = ks + nz - 1;
    par_for("MGKerrVector::SmoothRBJ", DevExeSpace(), 0, nmmb_-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      if (!pbj && (((i - is) + (j - js) + (k - ks)) & 1) != c0) return;
      const Real h = brdx(m)*fac;
      const Real ih2 = 1.0/(h*h);
      kmg::VViewAcc<decltype(u)> A{u, coeff, m};
      Real S[3], r[3];
      kmg::VecS(A, k, j, i, h, S);
      for (int a = 0; a < 3; ++a) { r[a] = src(m, a, k, j, i) + S[a]*ih2; }
      for (int a = 0; a < 3; ++a) {
        scr(m, a, k, j, i) = u(m, a, k, j, i)
                           + omega*h*h*(coeff(m, kVDinv + 3*a, k, j, i)*r[0]
                                      + coeff(m, kVDinv + 3*a + 1, k, j, i)*r[1]
                                      + coeff(m, kVDinv + 3*a + 2, k, j, i)*r[2]);
      }
    });
    par_for("MGKerrVector::SmoothRBJCopy", DevExeSpace(), 0, nmmb_-1, ks, ke, js, je,
            is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      if (!pbj && (((i - is) + (j - js) + (k - ks)) & 1) != c0) return;
      for (int a = 0; a < 3; ++a) { u(m, a, k, j, i) = scr(m, a, k, j, i); }
    });
    Hook(lev, false);
  }
  RestoreGhosts(lev);
}

void MGKerrVector::CalculateDefectPack() {
  const int lev = current_level_;
  const int ll = nlevel_-1-lev;
  const int nx = LevelN(lev, 0), ny = LevelN(lev, 1), nz = LevelN(lev, 2);
  const int is = ngh_, ie = is + nx - 1, js = ngh_, je = js + ny - 1;
  const int ks = ngh_, ke = ks + nz - 1;
  const Real fac = static_cast<Real>(1 << ll);
  auto brdx = block_rdx_.d_view;
  auto u = u_[lev].d_view;
  auto def = def_[lev].d_view;
  auto src = src_[lev].d_view;
  auto coeff = coeff_[lev].d_view;
  SaveGhosts(lev);
  Hook(lev, true);
  par_for("MGKerrVector::Defect", DevExeSpace(), 0, nmmb_-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real h = brdx(m)*fac;
    const Real ih2 = 1.0/(h*h);
    kmg::VViewAcc<decltype(u)> A{u, coeff, m};
    Real S[3];
    kmg::VecS(A, k, j, i, h, S);
    for (int a = 0; a < 3; ++a) { def(m, a, k, j, i) = src(m, a, k, j, i) + S[a]*ih2; }
  });
  RestoreGhosts(lev);
  if (Drv()->count_nonfinite_) {
    nonfinite_def_ += kmg::CountNonFinite(def, nmmb_, nvar_, ngh_, nx, ny, nz);
  }
}

void MGKerrVector::CalculateFASRHSPack() {
  const int lev = current_level_;
  const int ll = nlevel_-1-lev;
  const int nx = LevelN(lev, 0), ny = LevelN(lev, 1), nz = LevelN(lev, 2);
  const int is = ngh_, ie = is + nx - 1, js = ngh_, je = js + ny - 1;
  const int ks = ngh_, ke = ks + nz - 1;
  const Real fac = static_cast<Real>(1 << ll);
  auto brdx = block_rdx_.d_view;
  auto u = u_[lev].d_view;
  auto src = src_[lev].d_view;
  auto coeff = coeff_[lev].d_view;
  SaveGhosts(lev);
  Hook(lev, true);
  // persist only: uold = u was stored just before, ahead of the hook (see
  // kmg::CopyMaskedEdges): give uold the hooked edges too. With hook_scope = operator u's
  // ghosts (edges and kelvin faces) stay framework-filled, as uold's.
  if (!Drv()->hook_operator_ && Drv()->uold_edge_sync_ != kmg::kSyncNone) {
    kmg::CopyMaskedEdges(uold_[lev].d_view, u, sync_.d_view, nmmb_, nvar_, ngh_, nx, ny,
                         nz);
  }
  par_for("MGKerrVector::FASRHS", DevExeSpace(), 0, nmmb_-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real h = brdx(m)*fac;
    const Real ih2 = 1.0/(h*h);
    kmg::VViewAcc<decltype(u)> A{u, coeff, m};
    Real S[3];
    kmg::VecS(A, k, j, i, h, S);
    for (int a = 0; a < 3; ++a) { src(m, a, k, j, i) -= S[a]*ih2; }
  });
  RestoreGhosts(lev);
}

//----------------------------------------------------------------------------------------
// MGKerrVectorDriver

MGKerrVectorDriver::MGKerrVectorDriver(MeshBlockPack *pmbp, ParameterInput *pin,
                                       const std::string &block,
                                       const kmg::CoefSource &cs, const kmg::VManuf &vm)
    : MultigridDriver(pmbp, 3), cs_(cs), vm_(vm) {
  ncoeff_ = kmg::kNCoeffVec;
  eps_ = pin->GetOrAddReal(block, "mg_threshold", 1.0e-10);
  std::string norm_str = pin->GetOrAddString(block, "mg_norm", "rms");
  // K-005 B1(b): as <cfc>/mg_norm_ref_volume (mg_cfc_conformal_factor.cpp:430); <= 0 =
  // the root-grid volume (multigrid_driver.cpp CalculateDefectNorm)
  norm_ref_vol_ = pin->GetOrAddReal(block, "mg_norm_ref_volume", 0.0);
  if (norm_str == "legacy") {
    norm_mode_ = MGNormScaling::legacy;
  } else if (norm_str == "rms") {
    norm_mode_ = MGNormScaling::rms;
  } else if (norm_str == "max") {
    norm_mode_ = MGNormScaling::max;
  } else {
    Fatal(block + "/mg_norm must be 'legacy', 'rms' or 'max'.");
  }
  rtol_ = pin->GetOrAddReal(block, "mg_rtol", 0.0);
  fshowdef_ = pin->GetOrAddInteger(block, "mg_verbose", 0);
  ReadConvergenceOptions(pin, block);
  min_vcycles_ = pin->GetOrAddInteger(block, "mg_min_vcycles", 0);
  mg_verbose_ = fshowdef_;
  full_multigrid_ = false;
  npresmooth_ = pin->GetOrAddInteger(block, "mg_npresmooth", npresmooth_);
  npostsmooth_ = pin->GetOrAddInteger(block, "mg_npostsmooth", npostsmooth_);
  omega_ = pin->GetOrAddReal(block, "mg_omega", 1.0);

  std::string sm = pin->GetOrAddString(block, "smoother", "gs8");
  if (sm == "gs8") {
    smoother_ = 0;
  } else if (sm == "rb") {
    smoother_ = 1;
  } else if (sm == "pbj") {
    smoother_ = 2;
  } else {
    Fatal(block + "/smoother must be 'gs8', 'rb' or 'pbj'.");
  }
  pbj_omega_ = pin->GetOrAddReal(block, "pbj_omega", 6.0/7.0);
  std::string ef = pin->GetOrAddString(block, "edge_fill", "rule_r");
  if (ef == "rule_r") {
    edge_fill_ = kmg::EdgeFill::rule_r;
  } else if (ef == "bilinear") {
    edge_fill_ = kmg::EdgeFill::bilinear;
  } else if (ef == "off") {
    edge_fill_ = kmg::EdgeFill::off;
  } else {
    Fatal(block + "/edge_fill must be 'rule_r', 'bilinear' or 'off'.");
  }
  edge_poison_ = pin->GetOrAddBoolean(block, "edge_poison", false);
  std::string hs = pin->GetOrAddString(block, "hook_scope", "operator");
  if (hs == "operator") {
    hook_operator_ = true;
  } else if (hs == "persist") {
    hook_operator_ = false;
  } else {
    Fatal(block + "/hook_scope must be 'operator' or 'persist'.");
  }
  std::string us = pin->GetOrAddString(block, "uold_edge_sync", "free");
  if (us == "none") {
    uold_edge_sync_ = kmg::kSyncNone;
  } else if (us == "free") {
    uold_edge_sync_ = kmg::kSyncFree;
  } else if (us == "all") {
    uold_edge_sync_ = kmg::kSyncAll;
  } else {
    Fatal(block + "/uold_edge_sync must be 'none', 'free' or 'all'.");
  }
  count_nonfinite_ = pin->GetOrAddBoolean(block, "count_nonfinite", false);

  std::string ob = pin->GetOrAddString(block, "mg_outer_bc_vec", "kelvin");
  if (ob == "kelvin") {
    kelvin_ = true;
  } else if (ob == "zerofixed") {
    kelvin_ = false;
  } else {
    Fatal(block + "/mg_outer_bc_vec must be 'kelvin' or 'zerofixed'.");
  }
  std::string ms = pin->GetOrAddString(block, "mp_source", "self");
  if (ms == "self") {
    mp_self_ = true;
  } else if (ms == "rhs") {
    mp_self_ = false;
  } else {
    Fatal(block + "/mp_source must be 'self' or 'rhs'.");
  }
  mp_outer_ = pin->GetOrAddInteger(block, "mp_outer", 3);
  coeff_cache_ = kmg::CoeffCacheEnabled();
  cache_check_ = coeff_cache_ && kmg::CoeffCacheCheck();
  const RegionSize &msz = pmbp->pmesh->mesh_size;
  mpo_[0] = pin->GetOrAddReal(block, "mp_x01", 0.5*(msz.x1min + msz.x1max));
  mpo_[1] = pin->GetOrAddReal(block, "mp_x02", 0.5*(msz.x2min + msz.x2max));
  mpo_[2] = pin->GetOrAddReal(block, "mp_x03", 0.5*(msz.x3min + msz.x3max));
  for (int q = 0; q < kmg::kNMom; ++q) { mom_.v[q] = 0.0; }

  // the framework's physical boundary: zerofixed on the outer faces (the kelvin values
  // are written over it by the hook), zerograd on reflecting faces
  for (int f = 0; f < 6; ++f) {
    if (pmbp->pmesh->mesh_bcs[f] == BoundaryFlag::reflect) {
      mg_mesh_bcs_[f] = BoundaryFlag::mg_zerograd;
    } else if (pmbp->pmesh->mesh_bcs[f] != BoundaryFlag::periodic) {
      mg_mesh_bcs_[f] = BoundaryFlag::mg_zerofixed;
    }
  }
  fsubtract_average_ = false;

  int nghost = pin->GetOrAddInteger(block, "mg_nghost", 1);
  if (pin->GetOrAddBoolean(block, "root_on_host", false)) {
    Fatal(block + "/root_on_host = true is not supported (device-only coefficient fills).");
  }
  mgroot_ = new MGKerrVector(this, nullptr, nghost, false);
  mglevels_ = new MGKerrVector(this, pmbp, nghost);
  mglevels_->pbval = new MultigridBoundaryValues(pmbp, pin, false, mglevels_);
  mglevels_->pbval->InitializeBuffers(nvar_);
  mglevels_->pbval->RemapIndicesForMG();
  mglevels_->pbval->ComputePerLevelIndices();
}

MGKerrVectorDriver::~MGKerrVectorDriver() {
  delete mgroot_;
  delete mglevels_;
}

void MGKerrVectorDriver::Prepare() {
  // VecS assumes cubic cells (one h for all directions)
  {
    auto &mbs = pmy_pack_->pmb->mb_size;
    for (int m = 0; m < pmy_pack_->nmb_thispack; ++m) {
      const RegionSize &r = mbs.h_view(m);
      const Real tol = 1.0e-12*r.dx1;
      if (std::abs(r.dx2 - r.dx1) > tol || std::abs(r.dx3 - r.dx1) > tol) {
        Fatal("the Kerr vector multigrid needs cubic cells (dx1 = dx2 = dx3).");
      }
    }
  }
  PrepareForAMR();
  auto *lv = static_cast<MGKerrVector*>(mglevels_);
  auto *rt = static_cast<MGKerrVector*>(mgroot_);
  kmg::ComputeBlockEdgeMasks(pmy_pack_, blk_mask_, blk_phys_);
  // the root's physical edges are unsafe as on the blocks when the faces are kelvin (the
  // hook fills the interior transverse range only): build its mask as for mg_multipole
  BoundaryFlag rb[6];
  for (int f = 0; f < 6; ++f) {
    rb[f] = (kelvin_ && mg_mesh_bcs_[f] == BoundaryFlag::mg_zerofixed)
          ? BoundaryFlag::mg_multipole : mg_mesh_bcs_[f];
  }
  kmg::ComputeRootEdgeMask(rb, root_mask_, root_phys_);
  // kelvin faces: blocks, a physical non-reflecting face; root, every zerofixed face
  const int nmb = pmy_pack_->nmb_thispack;
  Kokkos::realloc(blk_kf_, nmb);
  Kokkos::realloc(root_kf_, 1);
  auto &bcs = pmy_pack_->pmb->mb_bcs;
  for (int m = 0; m < nmb; ++m) {
    int bits = 0;
    for (int f = 0; f < 6; ++f) {
      const BoundaryFlag b = bcs.h_view(m, f);
      if (kelvin_ && b != BoundaryFlag::block && b != BoundaryFlag::periodic &&
          mg_mesh_bcs_[f] == BoundaryFlag::mg_zerofixed) {
        bits |= (1 << f);
      }
    }
    blk_kf_.h_view(m) = bits;
  }
  {
    int bits = 0;
    for (int f = 0; f < 6; ++f) {
      if (kelvin_ && mg_mesh_bcs_[f] == BoundaryFlag::mg_zerofixed) { bits |= (1 << f); }
    }
    root_kf_.h_view(0) = bits;
  }
  blk_kf_.template modify<HostExeSpace>();
  blk_kf_.template sync<DevExeSpace>();
  root_kf_.template modify<HostExeSpace>();
  root_kf_.template sync<DevExeSpace>();
  lv->SetMasks(blk_mask_, blk_phys_, blk_kf_);
  rt->SetMasks(root_mask_, root_phys_, root_kf_);
  kmg::BuildSyncMask(blk_mask_, blk_phys_, nmb, uold_edge_sync_, false, blk_sync_);
  kmg::BuildSyncMask(root_mask_, root_phys_, 1, uold_edge_sync_, true, root_sync_);
  lv->SetSyncMask(blk_sync_);
  rt->SetSyncMask(root_sync_);
  if (coeff_cache_ && CoeffsIntact()) {
    ++coeff_reuses_;
    if (cache_check_) { CheckAgainstRefill(); }
  } else if (coeff_cache_ && CopyCoeffsFrom(coeff_donor_)) {
    ++coeff_copies_;
    if (cache_check_) { CheckAgainstRefill(); }
    StoreKey();
  } else {
    key_.ok = false;
    bad_dinv_[0] = lv->FillAllLevels(cs_);
    bad_dinv_[1] = rt->FillAllLevels(cs_);
    FillOctetCoeffs();
    ++coeff_fills_;
    StoreKey();
  }
  SetupMultigrid(0.0, false);
  Kokkos::fence();
}

namespace {
void CoeffPtrs(Multigrid *lv, Multigrid *rt, std::vector<const Real*> &p) {
  p.clear();
  for (Multigrid *g : {lv, rt}) {
    for (int lev = 0; lev < g->GetNumberOfLevels(); ++lev) {
      p.push_back(g->CoeffAtLevel(lev).d_view.data());
    }
  }
}
}  // namespace

// called after PrepareForAMR (which reallocates the storage and rebuilds the octets,
// zeroing their coefficients, when the mesh has changed)
bool MGKerrVectorDriver::CoeffsIntact() const {
  if (!kmg::KeyMatches(key_, cs_, pmy_mesh_, pmy_pack_, nreflevel_, noctets_)) return false;
  std::vector<const Real*> p;
  CoeffPtrs(mglevels_, mgroot_, p);
  return p == key_ptr_;
}

void MGKerrVectorDriver::StoreKey() {
  kmg::CaptureKey(key_, cs_, pmy_mesh_, pmy_pack_, nreflevel_, noctets_);
  CoeffPtrs(mglevels_, mgroot_, key_ptr_);
}

// KERR_MG_CACHECHECK: refill (as an uncached Prepare would) and count what differs from the
// reused/copied coefficients; the refilled values are kept
void MGKerrVectorDriver::CheckAgainstRefill() {
  auto *lv = static_cast<MGKerrVector*>(mglevels_);
  auto *rt = static_cast<MGKerrVector*>(mgroot_);
  Multigrid *g2[2] = {mglevels_, mgroot_};
  std::vector<DvceArray5D<Real>> save;
  for (Multigrid *g : g2) {
    for (int lev = 0; lev < g->GetNumberOfLevels(); ++lev) {
      save.push_back(kmg::CloneView(g->CoeffAtLevel(lev).d_view));
    }
  }
  std::vector<std::vector<Real>> osave(nreflevel_);
  for (int l = 0; l < nreflevel_; ++l) { osave[l] = oct_coeff_buf_[l]; }
  const long bsave[3] = {bad_dinv_[0], bad_dinv_[1], bad_dinv_[2]};
  const long bosave = boundary_octets_;
  bad_dinv_[0] = lv->FillAllLevels(cs_);
  bad_dinv_[1] = rt->FillAllLevels(cs_);
  FillOctetCoeffs();
  long diff = 0;
  size_t n = 0;
  for (Multigrid *g : g2) {
    for (int lev = 0; lev < g->GetNumberOfLevels(); ++lev, ++n) {
      const long c = kmg::CountDiff(save[n], g->CoeffAtLevel(lev).d_view);
      diff += (c < 0) ? 1 : c;
    }
  }
  for (int l = 0; l < nreflevel_; ++l) {
    if (osave[l].size() != oct_coeff_buf_[l].size()) { ++diff; continue; }
    for (size_t q = 0; q < osave[l].size(); ++q) {
      const Real x = osave[l][q], y = oct_coeff_buf_[l][q];
      if (x != y && !(x != x && y != y)) { ++diff; }
    }
  }
  for (int q = 0; q < 3; ++q) { if (bsave[q] != bad_dinv_[q]) { ++diff; } }
  if (bosave != boundary_octets_) { ++diff; }
  ++check_n_;
  check_diff_ += diff;
}

bool MGKerrVectorDriver::CopyCoeffsFrom(MGKerrVectorDriver *d) {
  if (d == nullptr || d == this || !d->CoeffsIntact()) return false;
  // d's key matches the current state; so must this driver's layout
  if (d->ncoeff_ != ncoeff_ || d->nreflevel_ != nreflevel_) return false;
  for (int l = 0; l < nreflevel_; ++l) {
    if (d->noctets_[l] != noctets_[l] ||
        d->oct_coeff_buf_[l].size() != oct_coeff_buf_[l].size()) {
      return false;
    }
  }
  Multigrid *mine[2] = {mglevels_, mgroot_}, *theirs[2] = {d->mglevels_, d->mgroot_};
  for (int g = 0; g < 2; ++g) {
    if (mine[g]->GetNumberOfLevels() != theirs[g]->GetNumberOfLevels()) return false;
    for (int lev = 0; lev < mine[g]->GetNumberOfLevels(); ++lev) {
      const auto &a = mine[g]->CoeffAtLevel(lev).d_view;
      const auto &b = theirs[g]->CoeffAtLevel(lev).d_view;
      for (int r = 0; r < 5; ++r) {
        if (a.extent(r) != b.extent(r)) return false;
      }
    }
  }
  auto *lv = static_cast<MGKerrVector*>(mglevels_);
  auto *rt = static_cast<MGKerrVector*>(mgroot_);
  lv->BuildExtents();
  rt->BuildExtents();
  for (int g = 0; g < 2; ++g) {
    for (int lev = 0; lev < mine[g]->GetNumberOfLevels(); ++lev) {
      Kokkos::deep_copy(mine[g]->CoeffAtLevel(lev).d_view,
                        theirs[g]->CoeffAtLevel(lev).d_view);
    }
  }
  for (int l = 0; l < nreflevel_; ++l) {
    std::copy(d->oct_coeff_buf_[l].begin(), d->oct_coeff_buf_[l].end(),
              oct_coeff_buf_[l].begin());
  }
  for (int q = 0; q < 3; ++q) { bad_dinv_[q] = d->bad_dinv_[q]; }
  boundary_octets_ = d->boundary_octets_;
  return true;
}

void MGKerrVectorDriver::ComputeMoments() {
  if (!kelvin_) return;
  auto *lv = static_cast<MGKerrVector*>(mglevels_);
  const int lev = lv->GetNumberOfLevels() - 1;
  Real m[kmg::kNMom];
  // self: (S - S_flat) on the cells next to the box and to unsafe edges reads the hooked
  // ghosts (kelvin faces, edge fill), as the operator kernels do (hook_scope = operator
  // leaves framework values there between kernels)
  if (mp_self_) {
    lv->SaveGhosts(lev);
    lv->Hook(lev, false);
  }
  kmg::LocalMoments(lv->UAt(lev).d_view, lv->SrcAt(lev).d_view,
                    lv->CoeffAtLevel(lev).d_view, lv->Extents(), lv->NumMB(),
                    lv->GetGhostCells(), lv->LevelN(lev, 0), lv->LevelN(lev, 1),
                    lv->LevelN(lev, 2), mp_self_, mpo_, m);
  if (mp_self_) { lv->RestoreGhosts(lev); }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, m, kmg::kNMom, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  for (int q = 0; q < kmg::kNMom; ++q) { mom_.v[q] = m[q]; }
}

void MGKerrVectorDriver::Solve(Driver *pdriver, int stage, Real dt) {
  Prepare();
  const int nout = (kelvin_ && mp_self_) ? std::max(mp_outer_, 1) : 1;
  for (int n = 0; n < nout; ++n) {
    ComputeMoments();
    SolveMG(pdriver);
  }
  Kokkos::fence();
}

void MGKerrVectorDriver::LoadSource(const DvceArray5D<Real> &src, int ngh, Real fac) {
  mglevels_->ReallocateForAMR();
  mglevels_->LoadSource(src, 0, ngh, fac);
}

void MGKerrVectorDriver::RetrieveSolution(DvceArray5D<Real> &dst) {
  mglevels_->RetrieveResult(dst, 0, pmy_pack_->pmesh->mb_indcs.ng);
}

void MGKerrVectorDriver::ResetSolution() {
  ZeroHierarchy();
  auto *lv = static_cast<MGKerrVector*>(mglevels_);
  const int nl = lv->GetNumberOfLevels();
  Kokkos::deep_copy(lv->UAt(nl-1).d_view, 0.0);
  Kokkos::deep_copy(lv->UAt(nl-1).h_view, 0.0);
  if (src_blob_) {
    kmg::FillManufV(lv->SrcAt(nl-1).d_view, lv->Extents(), lv->NumMB(),
                    lv->GetGhostCells(), lv->LevelN(nl-1, 0), lv->LevelN(nl-1, 1),
                    lv->LevelN(nl-1, 2), vm_);
  } else {
    kmg::FillManufVecSrc(lv->SrcAt(nl-1).d_view, lv->Extents(), lv->NumMB(),
                         lv->GetGhostCells(), lv->LevelN(nl-1, 0), lv->LevelN(nl-1, 1),
                         lv->LevelN(nl-1, 2), cs_, vm_);
  }
  Kokkos::deep_copy(lv->SrcAt(nl-1).h_view, lv->SrcAt(nl-1).d_view);
  for (int q = 0; q < kmg::kNMom; ++q) { mom_.v[q] = 0.0; }
  lv->nonfinite_def_ = 0;
  static_cast<MGKerrVector*>(mgroot_)->nonfinite_def_ = 0;
  nonfinite_oct_ = 0;
  Kokkos::fence();
}

std::vector<Real> MGKerrVectorDriver::RunCycles(Driver *pdriver, int ncyc) {
  std::vector<Real> d;
  d.push_back(TotalDefectNorm());
  for (int n = 0; n < ncyc; ++n) {
    if (mp_self_ || n == 0) { ComputeMoments(); }
    SolveVCycle(pdriver, npresmooth_, npostsmooth_);
    d.push_back(TotalDefectNorm());
  }
  Kokkos::fence();
  return d;
}

void MGKerrVectorDriver::FillOctetCoeffs() {
  bad_dinv_[2] = 0;
  boundary_octets_ = 0;
  if (nreflevel_ == 0) return;
  const int ngh = mgroot_->GetGhostCells();
  const int nc = 2 + 2*ngh;
  const Real rootdx = mgroot_->GetRootDx();
  const RegionSize &ms = pmy_mesh_->mesh_size;
  for (int l = 0; l < nreflevel_; ++l) {
    const int noct = noctets_[l];
    if (noct == 0) continue;
    DualArray1D<RegionSize> blk;
    Kokkos::realloc(blk, noct);
    const Real dxo = rootdx/static_cast<Real>(1 << (l + 1));
    for (int o = 0; o < noct; ++o) {
      const LogicalLocation &loc = octets_[l][o].loc;
      RegionSize r;
      r.x1min = ms.x1min + static_cast<Real>(loc.lx1)*2.0*dxo;
      r.x2min = ms.x2min + static_cast<Real>(loc.lx2)*2.0*dxo;
      r.x3min = ms.x3min + static_cast<Real>(loc.lx3)*2.0*dxo;
      r.x1max = r.x1min + 2.0*dxo;
      r.x2max = r.x2min + 2.0*dxo;
      r.x3max = r.x3min + 2.0*dxo;
      r.dx1 = r.dx2 = r.dx3 = dxo;
      blk.h_view(o) = r;
      const Real tol = 1.0e-9*dxo;
      const bool lo[3] = {r.x1min < ms.x1min + tol, r.x2min < ms.x2min + tol,
                          r.x3min < ms.x3min + tol};
      const bool hi[3] = {r.x1max > ms.x1max - tol, r.x2max > ms.x2max - tol,
                          r.x3max > ms.x3max - tol};
      bool touch = false;
      for (int d = 0; d < 3; ++d) {
        if (lo[d] && mg_mesh_bcs_[2*d] != BoundaryFlag::periodic) touch = true;
        if (hi[d] && mg_mesh_bcs_[2*d + 1] != BoundaryFlag::periodic) touch = true;
      }
      if (touch) ++boundary_octets_;
    }
    blk.template modify<HostExeSpace>();
    blk.template sync<DevExeSpace>();
    DvceArray5D<Real> tmp("kmg_voctcoeff", noct, ncoeff_, nc, nc, nc);
    bad_dinv_[2] += kmg::FillLevelVecCoeffs(tmp, blk, noct, ngh, 2, 2, 2, cs_);
    auto h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), tmp);
    for (int o = 0; o < noct; ++o) {
      MGOctet &oct = octets_[l][o];
      for (int c = 0; c < ncoeff_; ++c) {
        for (int k = 0; k < nc; ++k) {
          for (int j = 0; j < nc; ++j) {
            for (int i = 0; i < nc; ++i) { oct.Coeff(c, k, j, i) = h(o, c, k, j, i); }
          }
        }
      }
    }
  }
}

// octets: Dinv is formed on the fly from h = root dx / 2^rlev (not the stored channels)
void MGKerrVectorDriver::SmoothOctet(MGOctet &oct, int rlev, int color) {
  const int ngh = mgroot_->GetGhostCells();
  const Real h = mgroot_->GetRootDx()/static_cast<Real>(1 << rlev);
  const int c = color ^ coffset_;
  if (smoother_ == 0) {
    for (int sc = 0; sc < 8; ++sc) {
      const int oi = sc & 1, oj = (sc >> 1) & 1, ok = (sc >> 2) & 1;
      if (((oi + oj + ok) & 1) != c) continue;
      const int k = ngh + ok, j = ngh + oj, i = ngh + oi;
      Real nw[3];
      OctPoint(oct, k, j, i, h, omega_, nw);
      for (int a = 0; a < 3; ++a) { oct.U(a, k, j, i) = nw[a]; }
    }
  } else {
    // rb: Jacobi within the colour; pbj: all 8 cells, damped, on colour 0 only
    const bool pbj = (smoother_ == 2);
    if (pbj && c != 0) return;
    const Real om = pbj ? pbj_omega_ : omega_;
    Real nw[8][3];
    for (int sc = 0; sc < 8; ++sc) {
      const int oi = sc & 1, oj = (sc >> 1) & 1, ok = (sc >> 2) & 1;
      if (!pbj && ((oi + oj + ok) & 1) != c) continue;
      OctPoint(oct, ngh + ok, ngh + oj, ngh + oi, h, om, nw[sc]);
    }
    for (int sc = 0; sc < 8; ++sc) {
      const int oi = sc & 1, oj = (sc >> 1) & 1, ok = (sc >> 2) & 1;
      if (!pbj && ((oi + oj + ok) & 1) != c) continue;
      for (int a = 0; a < 3; ++a) { oct.U(a, ngh + ok, ngh + oj, ngh + oi) = nw[sc][a]; }
    }
  }
}

void MGKerrVectorDriver::CalculateDefectOctet(MGOctet &oct, int rlev) {
  const int ngh = mgroot_->GetGhostCells();
  const Real h = mgroot_->GetRootDx()/static_cast<Real>(1 << rlev);
  const Real ih2 = 1.0/(h*h);
  OctAccV A{&oct};
  for (int k = ngh; k <= ngh+1; ++k) {
    for (int j = ngh; j <= ngh+1; ++j) {
      for (int i = ngh; i <= ngh+1; ++i) {
        Real S[3];
        kmg::VecS(A, k, j, i, h, S);
        for (int a = 0; a < 3; ++a) {
          const Real d = oct.Src(a, k, j, i) + S[a]*ih2;
          oct.Def(a, k, j, i) = d;
          if (count_nonfinite_ && kmg::NonFinite(d)) { ++nonfinite_oct_; }
        }
      }
    }
  }
}

void MGKerrVectorDriver::CalculateFASRHSOctet(MGOctet &oct, int rlev) {
  const int ngh = mgroot_->GetGhostCells();
  const Real h = mgroot_->GetRootDx()/static_cast<Real>(1 << rlev);
  const Real ih2 = 1.0/(h*h);
  OctAccV A{&oct};
  for (int k = ngh; k <= ngh+1; ++k) {
    for (int j = ngh; j <= ngh+1; ++j) {
      for (int i = ngh; i <= ngh+1; ++i) {
        Real S[3];
        kmg::VecS(A, k, j, i, h, S);
        for (int a = 0; a < 3; ++a) { oct.Src(a, k, j, i) -= S[a]*ih2; }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
// test helpers

KerrOpStats MGKerrVectorDriver::OperatorTest(bool exact_edges, bool poison_corners) {
  auto *lv = static_cast<MGKerrVector*>(mglevels_);
  const int lev = lv->GetNumberOfLevels() - 1;
  const int nmb = lv->NumMB();
  const int ngh = lv->GetGhostCells();
  const int nx = lv->LevelN(lev, 0), ny = lv->LevelN(lev, 1), nz = lv->LevelN(lev, 2);
  auto u = lv->UAt(lev).d_view;
  auto coeff = lv->CoeffAtLevel(lev).d_view;
  kmg::FillManufV(u, lv->Extents(), nmb, ngh, nx, ny, nz, vm_);
  if (!exact_edges) {
    kmg::ApplyEdgeFill(u, lv->Mask().d_view, nmb, 3, ngh, nx, ny, nz, edge_fill_,
                       edge_poison_);
  }
  if (poison_corners) { kmg::PoisonCorners(u, nmb, 3, ngh, nx, ny, nz); }
  DvceArray5D<Real> s("kmg_vop_src", nmb, 3, nz + 2*ngh, ny + 2*ngh, nx + 2*ngh);
  kmg::FillManufVecSrc(s, lv->Extents(), nmb, ngh, nx, ny, nz, cs_, vm_);
  DvceArray5D<Real> out("kmg_voptest", nmb, 2, nz, ny, nx);
  DvceArray1D<Real> rdx("kmg_rdx", nmb);
  {
    auto h = Kokkos::create_mirror_view(rdx);
    for (int m = 0; m < nmb; ++m) { h(m) = lv->LevelDx(lev, m); }
    Kokkos::deep_copy(rdx, h);
  }
  const Real qnan = std::numeric_limits<Real>::quiet_NaN();
  par_for("kmg::VOperatorTest", DevExeSpace(), 0, nmb-1, 0, nz-1, 0, ny-1, 0, nx-1,
  KOKKOS_LAMBDA(const int m, const int kk, const int jj, const int ii) {
    const int k = kk + ngh, j = jj + ngh, i = ii + ngh;
    const Real h = rdx(m), ih2 = 1.0/(h*h);
    kmg::VViewAcc<decltype(u)> A{u, coeff, m};
    kmg::FlatVAcc<kmg::VViewAcc<decltype(u)>> F{A};
    Real S[3], Sf[3];
    kmg::VecS(A, k, j, i, h, S);
    kmg::VecS(F, k, j, i, h, Sf);
    Real dmax = 0.0, fmax = 0.0;
    bool bad = false;
    for (int a = 0; a < 3; ++a) {
      const Real d = s(m, a, k, j, i) + S[a]*ih2;
      if (kmg::NonFinite(d)) bad = true;
      dmax = Kokkos::fmax(dmax, Kokkos::fabs(d));
      fmax = Kokkos::fmax(fmax, Kokkos::fabs(S[a] - Sf[a])*ih2);
    }
    out(m, 0, kk, jj, ii) = bad ? qnan : dmax;
    out(m, 1, kk, jj, ii) = fmax;
  });
  auto h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), out);
  KerrOpStats st;
  for (int m = 0; m < nmb; ++m) {
    const int msk = lv->Mask().h_view(m), phy = lv->Phys().h_view(m);
    const Real dx = lv->LevelDx(lev, m), dv = dx*dx*dx;
    const int n[3] = {nx, ny, nz};
    for (int kk = 0; kk < nz; ++kk) {
      for (int jj = 0; jj < ny; ++jj) {
        for (int ii = 0; ii < nx; ++ii) {
          const int I[3] = {ii, jj, kk};
          const int cls = CellClass(msk, phy, I, n);
          const Real d = h(m, 0, kk, jj, ii);
          if (kmg::NonFinite(d)) {
            ++st.nonfinite;
            continue;
          }
          st.maxd[cls] = std::max(st.maxd[cls], d);
          st.l2[cls] += d*d*dv;
          st.vol[cls] += dv;
          ++st.ncell[cls];
          st.max_s7 = std::max(st.max_s7, h(m, 1, kk, jj, ii));
        }
      }
    }
  }
  return st;
}

void MGKerrVectorDriver::ErrorRadial(const Real c[3], const std::vector<Real> &e,
                          std::vector<Real> &emax, std::vector<Real> &s2,
                          std::vector<Real> &vol, std::vector<long> &n) {
  auto *lv = static_cast<MGKerrVector*>(mglevels_);
  const int lev = lv->GetNumberOfLevels() - 1;
  const int nmb = lv->NumMB();
  const int ngh = lv->GetGhostCells();
  const int nx = lv->LevelN(lev, 0), ny = lv->LevelN(lev, 1), nz = lv->LevelN(lev, 2);
  auto h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), lv->UAt(lev).d_view);
  const int ns = static_cast<int>(e.size());
  emax.assign(ns, 0.0);
  s2.assign(ns, 0.0);
  vol.assign(ns, 0.0);
  n.assign(ns, 0);
  for (int m = 0; m < nmb; ++m) {
    const RegionSize &b = lv->Extents().h_view(m);
    const Real dx1 = (b.x1max - b.x1min)/nx, dx2 = (b.x2max - b.x2min)/ny;
    const Real dx3 = (b.x3max - b.x3min)/nz, dv = dx1*dx2*dx3;
    for (int k = 0; k < nz; ++k) {
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          const Real X[3] = {b.x1min + (i + 0.5)*dx1, b.x2min + (j + 0.5)*dx2,
                             b.x3min + (k + 0.5)*dx3};
          const Real r = std::sqrt((X[0] - c[0])*(X[0] - c[0]) + (X[1] - c[1])*(X[1] - c[1])
                                   + (X[2] - c[2])*(X[2] - c[2]));
          int s = -1;
          for (int q = 0; q < ns; ++q) {
            if (r >= e[q]) s = q;
          }
          if (s < 0) continue;
          Real ea = 0.0, e2 = 0.0;
          for (int q = 0; q < 3; ++q) {
            Real us, du[3], ddu[6];
            kmg::ManufEval(vm_.c[q], X[0], X[1], X[2], us, du, ddu);
            const Real e = h(m, q, k + ngh, j + ngh, i + ngh) - us;
            ea = std::max(ea, std::abs(e));
            e2 += e*e;
          }
          emax[s] = std::max(emax[s], ea);
          s2[s] += e2*dv;
          vol[s] += dv;
          ++n[s];
        }
      }
    }
  }
}

void MGKerrVectorDriver::ErrorVsManufByClass(Real emax[3], Real erms[3], long ncell[3]) {
  auto *lv = static_cast<MGKerrVector*>(mglevels_);
  const int lev = lv->GetNumberOfLevels() - 1;
  const int nmb = lv->NumMB();
  const int ngh = lv->GetGhostCells();
  const int nx = lv->LevelN(lev, 0), ny = lv->LevelN(lev, 1), nz = lv->LevelN(lev, 2);
  const int n[3] = {nx, ny, nz};
  auto h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), lv->UAt(lev).d_view);
  Real s2[3] = {0, 0, 0}, vol[3] = {0, 0, 0};
  for (int c = 0; c < 3; ++c) {
    emax[c] = 0.0;
    ncell[c] = 0;
  }
  for (int m = 0; m < nmb; ++m) {
    const RegionSize &b = lv->Extents().h_view(m);
    const int msk = lv->Mask().h_view(m), phy = lv->Phys().h_view(m);
    const Real dx1 = (b.x1max - b.x1min)/nx, dx2 = (b.x2max - b.x2min)/ny;
    const Real dx3 = (b.x3max - b.x3min)/nz, dv = dx1*dx2*dx3;
    for (int k = 0; k < nz; ++k) {
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          const int I[3] = {i, j, k};
          const int cls = CellClass(msk, phy, I, n);
          Real e2 = 0.0, ea = 0.0;
          for (int c = 0; c < 3; ++c) {
            Real us, du[3], ddu[6];
            kmg::ManufEval(vm_.c[c], b.x1min + (i + 0.5)*dx1, b.x2min + (j + 0.5)*dx2,
                           b.x3min + (k + 0.5)*dx3, us, du, ddu);
            const Real e = h(m, c, k + ngh, j + ngh, i + ngh) - us;
            e2 += e*e;
            ea = std::max(ea, std::abs(e));
          }
          emax[cls] = std::max(emax[cls], ea);
          s2[cls] += e2*dv;
          vol[cls] += dv;
          ++ncell[cls];
        }
      }
    }
  }
  for (int c = 0; c < 3; ++c) {
    erms[c] = (vol[c] > 0.0) ? std::sqrt(s2[c]/vol[c]) : 0.0;
  }
}

void MGKerrVectorDriver::Probe(const std::vector<Real> &pts, std::vector<Real> &val,
                               std::vector<Real> &ctr, std::vector<int> &own) {
  auto *lv = static_cast<MGKerrVector*>(mglevels_);
  const int lev = lv->GetNumberOfLevels() - 1;
  const int nmb = lv->NumMB();
  const int ngh = lv->GetGhostCells();
  const int n[3] = {lv->LevelN(lev, 0), lv->LevelN(lev, 1), lv->LevelN(lev, 2)};
  auto h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), lv->UAt(lev).d_view);
  const int np = static_cast<int>(pts.size())/3;
  val.assign(3*np, 0.0);
  ctr.assign(3*np, 0.0);
  own.assign(np, 0);
  for (int p = 0; p < np; ++p) {
    const Real X[3] = {pts[3*p], pts[3*p + 1], pts[3*p + 2]};
    for (int m = 0; m < nmb && !own[p]; ++m) {
      const RegionSize &b = lv->Extents().h_view(m);
      const Real mn[3] = {b.x1min, b.x2min, b.x3min}, mx[3] = {b.x1max, b.x2max, b.x3max};
      bool in = true;
      for (int d = 0; d < 3; ++d) { in = in && X[d] >= mn[d] && X[d] < mx[d]; }
      if (!in) continue;
      int I[3];
      for (int d = 0; d < 3; ++d) {
        const Real dx = (mx[d] - mn[d])/static_cast<Real>(n[d]);
        I[d] = std::min(n[d] - 1, static_cast<int>((X[d] - mn[d])/dx));
        ctr[3*p + d] = mn[d] + (static_cast<Real>(I[d]) + 0.5)*dx;
      }
      for (int c = 0; c < 3; ++c) {
        val[3*p + c] = h(m, c, I[2] + ngh, I[1] + ngh, I[0] + ngh);
      }
      own[p] = 1;
    }
  }
}

void MGKerrVectorDriver::NonFiniteCounts(long c[3]) const {
  c[0] = static_cast<const MGKerrVector*>(mglevels_)->nonfinite_def_;
  c[1] = static_cast<const MGKerrVector*>(mgroot_)->nonfinite_def_;
  c[2] = nonfinite_oct_;
}

void MGKerrVectorDriver::EdgeCounts(long c[3]) {
  kmg::CountEdges(blk_mask_, blk_phys_, pmy_pack_->nmb_thispack, c);
}

}  // namespace cfc
