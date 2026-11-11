//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mg_kerr_scalar.cpp
//! \brief W3a: the Kerr variable-coefficient scalar multigrid (see mg_kerr_scalar.hpp).

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "multigrid/multigrid.hpp"
#include "kerr_mg_ops.hpp"
#include "mg_kerr_scalar.hpp"

namespace cfc {

using kmg::kChPsi0;

namespace {

template <class V>
using VAcc = kmg::ViewAcc<V>;

struct OctAcc {
  MGOctet *o;
  inline Real U(int k, int j, int i) const { return o->U(0, k, j, i); }
  inline Real C(int q, int k, int j, int i) const { return o->Coeff(q, k, j, i); }
};

[[noreturn]] void Fatal(const std::string &msg) {
  std::cout << "### FATAL ERROR in MGKerrScalarDriver" << std::endl << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

//! the class of interior cell I of a block with n cells (KerrOpStats): 0 bulk, 1 next to
//! an unsafe edge on the domain boundary, 2 next to an unsafe interior edge
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

}  // namespace

//----------------------------------------------------------------------------------------
// MGKerrScalar

MGKerrScalar::MGKerrScalar(MultigridDriver *pmd, MeshBlockPack *pmbp, int nghost,
                           int ncoeff, bool on_host)
    : Multigrid(pmd, pmbp, nghost, on_host) {
  ncoeff_ = ncoeff;
  for (int l = 0; l < nlevel_; l++) {
    int ll = nlevel_-1-l;
    Kokkos::realloc(coeff_[l], nmmb_, ncoeff_, (indcs_.nx3>>ll)+2*ngh_,
                    (indcs_.nx2>>ll)+2*ngh_, (indcs_.nx1>>ll)+2*ngh_);
  }
  Kokkos::realloc(scratch_, nmmb_, 1, indcs_.nx3+2*ngh_, indcs_.nx2+2*ngh_,
                  indcs_.nx1+2*ngh_);
}

MGKerrScalarDriver *MGKerrScalar::Drv() const {
  return static_cast<MGKerrScalarDriver*>(pmy_driver_);
}

Real MGKerrScalar::LevelDx(int lev, int m) const {
  const int ll = nlevel_ - 1 - lev;
  return block_rdx_.h_view(m)*static_cast<Real>(1 << ll);
}

void MGKerrScalar::BuildExtents() {
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

void MGKerrScalar::FillAllLevels(const kmg::CoefSource &cs, const kmg::Manuf &mf) {
  BuildExtents();
  if (static_cast<int>(scratch_.extent(0)) != nmmb_) {
    Kokkos::realloc(scratch_, nmmb_, 1, indcs_.nx3+2*ngh_, indcs_.nx2+2*ngh_,
                    indcs_.nx1+2*ngh_);
  }
  for (int lev = 0; lev < nlevel_; ++lev) {
    const int nx = LevelN(lev, 0), ny = LevelN(lev, 1), nz = LevelN(lev, 2);
    if (static_cast<int>(coeff_[lev].d_view.extent(0)) != nmmb_) {
      Kokkos::realloc(coeff_[lev], nmmb_, ncoeff_, nz+2*ngh_, ny+2*ngh_, nx+2*ngh_);
    }
    kmg::FillLevelCoeffs(coeff_[lev].d_view, blk_, nmmb_, ngh_, nx, ny, nz, cs, mf);
  }
}

void MGKerrScalar::FillAllLevelsBg(const kmg::CoefSource &cs, int lev_lo, int lev_hi) {
  BuildExtents();
  if (static_cast<int>(scratch_.extent(0)) != nmmb_) {
    Kokkos::realloc(scratch_, nmmb_, 1, indcs_.nx3+2*ngh_, indcs_.nx2+2*ngh_,
                    indcs_.nx1+2*ngh_);
  }
  if (lev_hi < 0) { lev_hi = nlevel_ - 1; }
  for (int lev = 0; lev < nlevel_; ++lev) {
    const int nx = LevelN(lev, 0), ny = LevelN(lev, 1), nz = LevelN(lev, 2);
    if (static_cast<int>(coeff_[lev].d_view.extent(0)) != nmmb_) {
      Kokkos::realloc(coeff_[lev], nmmb_, ncoeff_, nz+2*ngh_, ny+2*ngh_, nx+2*ngh_);
    }
    if (lev < lev_lo || lev > lev_hi) { continue; }
    kmg::FillLevelBg(coeff_[lev].d_view, blk_, nmmb_, ngh_, nx, ny, nz, cs);
  }
}

void MGKerrScalar::FillFinestMatterManuf(int kind, const kmg::CoefSource &cs,
                                         const kmg::Manuf &mf,
                                         const kmg::MatterManuf &mm) {
  const int lev = nlevel_ - 1;
  kmg::FillLevelMatterManuf(coeff_[lev].d_view, blk_, nmmb_, ngh_, LevelN(lev, 0),
                            LevelN(lev, 1), LevelN(lev, 2), kind, cs, mf, mm);
}

void MGKerrScalar::FillFinestMatterExt(const DvceArray5D<Real> &mt,
                                       const DvceArray5D<Real> &da2,
                                       const DvceArray5D<Real> &dpsi, bool has_dpsi,
                                       int ngh) {
  const int lev = nlevel_ - 1;
  auto cm = coeff_[lev].d_view;
  const int lngh = ngh_, off = ngh - ngh_;
  const int ie = lngh + indcs_.nx1 - 1, je = lngh + indcs_.nx2 - 1;
  const int ke = lngh + indcs_.nx3 - 1;
  par_for("MGKerrScalar::FillFinestMatterExt", DevExeSpace(), 0, nmmb_-1, lngh, ke, lngh,
          je, lngh, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    cm(m, kmg::kChM, k, j, i) = mt(m, 0, k+off, j+off, i+off);
    cm(m, kmg::kChDA2, k, j, i) = da2(m, 0, k+off, j+off, i+off);
    cm(m, kmg::kChDPsi, k, j, i) = has_dpsi ? dpsi(m, 0, k+off, j+off, i+off) : 0.0;
  });
}

void MGKerrScalar::EdgeHook(int lev, bool poison_ok) {
  const MGKerrScalarDriver *d = Drv();
  const int nx = LevelN(lev, 0), ny = LevelN(lev, 1), nz = LevelN(lev, 2);
  if (d->corner_poison_) {
    kmg::PoisonCorners(u_[lev].d_view, nmmb_, nvar_, ngh_, nx, ny, nz);
  }
  kmg::ApplyEdgeFill(u_[lev].d_view, mask_.d_view, nmmb_, nvar_, ngh_, nx, ny, nz,
                     d->edge_fill_, d->edge_poison_ && poison_ok);
}

void MGKerrScalar::SaveGhosts(int lev) {
  if (!Drv()->hook_operator_) return;
  if (static_cast<int>(gsave_.extent(0)) != nmmb_) {
    Kokkos::realloc(gsave_, nmmb_, nvar_, indcs_.nx3+2*ngh_, indcs_.nx2+2*ngh_,
                    indcs_.nx1+2*ngh_);
  }
  kmg::CopyGhostShell(gsave_, u_[lev].d_view, nmmb_, nvar_, ngh_, LevelN(lev, 0),
                      LevelN(lev, 1), LevelN(lev, 2));
}

void MGKerrScalar::RestoreGhosts(int lev) {
  if (!Drv()->hook_operator_) return;
  kmg::CopyGhostShell(u_[lev].d_view, gsave_, nmmb_, nvar_, ngh_, LevelN(lev, 0),
                      LevelN(lev, 1), LevelN(lev, 2));
}

void MGKerrScalar::SmoothPack(int color) {
  const int lev = current_level_;
  const int ll = nlevel_-1-lev;
  const int nx = LevelN(lev, 0), ny = LevelN(lev, 1), nz = LevelN(lev, 2);
  const int is = ngh_, js = ngh_, ks = ngh_;
  const int c0 = color ^ pmy_driver_->GetCoffset();
  const Real omega = Drv()->omega_;
  const int kind = Drv()->kind_;
  const Real pfl = Drv()->psi_floor_;
  const Real fac = static_cast<Real>(1 << ll);
  auto brdx = block_rdx_.d_view;
  auto u = u_[lev].d_view;
  auto src = src_[lev].d_view;
  auto coeff = coeff_[lev].d_view;
  const kmg::EdgeFill fill = Drv()->edge_fill_;
  SaveGhosts(lev);
  EdgeHook(lev, true);
  if (Drv()->smoother_ == 0) {
    for (int sc = 0; sc < 8; ++sc) {
      const int oi = sc & 1, oj = (sc >> 1) & 1, ok = (sc >> 2) & 1;
      if (((oi + oj + ok) & 1) != c0) continue;
      const int ni = (nx - oi + 1)/2, nj = (ny - oj + 1)/2, nk = (nz - ok + 1)/2;
      if (ni <= 0 || nj <= 0 || nk <= 0) continue;
      par_for("MGKerrScalar::SmoothGS8", DevExeSpace(), 0, nmmb_-1, 0, nk-1, 0, nj-1,
              0, ni-1,
      KOKKOS_LAMBDA(const int m, const int kk, const int jj, const int ii) {
        const int k = ks + ok + 2*kk, j = js + oj + 2*jj, i = is + oi + 2*ii;
        const Real dx = brdx(m)*fac;
        const Real ih2 = 1.0/(dx*dx);
        VAcc<decltype(u)> A{u, coeff, m, 0};
        Real dg;
        const Real r = kmg::PointResidual(kind, A, k, j, i, ih2, dg) + src(m, 0, k, j, i);
        const Real p0 = (kind == kmg::kKindPsi) ? coeff(m, kChPsi0, k, j, i) : 0.0;
        u(m, 0, k, j, i) = kmg::FloorPsi(kind, u(m, 0, k, j, i) + omega*r/dg, p0, pfl);
      });
      kmg::ApplyEdgeFill(u, mask_.d_view, nmmb_, nvar_, ngh_, nx, ny, nz, fill, false);
    }
  } else {
    auto scr = scratch_;
    const int ie = is + nx - 1, je = js + ny - 1, ke = ks + nz - 1;
    par_for("MGKerrScalar::SmoothRBJ", DevExeSpace(), 0, nmmb_-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      if ((((i - is) + (j - js) + (k - ks)) & 1) != c0) return;
      const Real dx = brdx(m)*fac;
      const Real ih2 = 1.0/(dx*dx);
      VAcc<decltype(u)> A{u, coeff, m, 0};
      Real dg;
      const Real r = kmg::PointResidual(kind, A, k, j, i, ih2, dg) + src(m, 0, k, j, i);
      const Real p0 = (kind == kmg::kKindPsi) ? coeff(m, kChPsi0, k, j, i) : 0.0;
      scr(m, 0, k, j, i) = kmg::FloorPsi(kind, u(m, 0, k, j, i) + omega*r/dg, p0, pfl);
    });
    par_for("MGKerrScalar::SmoothRBJCopy", DevExeSpace(), 0, nmmb_-1, ks, ke, js, je,
            is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      if ((((i - is) + (j - js) + (k - ks)) & 1) != c0) return;
      u(m, 0, k, j, i) = scr(m, 0, k, j, i);
    });
    kmg::ApplyEdgeFill(u, mask_.d_view, nmmb_, nvar_, ngh_, nx, ny, nz, fill, false);
  }
  RestoreGhosts(lev);
}

void MGKerrScalar::CalculateDefectPack() {
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
  const int kind = Drv()->kind_;
  SaveGhosts(lev);
  EdgeHook(lev, true);
  par_for("MGKerrScalar::Defect", DevExeSpace(), 0, nmmb_-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real dx = brdx(m)*fac;
    const Real ih2 = 1.0/(dx*dx);
    VAcc<decltype(u)> A{u, coeff, m, 0};
    Real dg;
    def(m, 0, k, j, i) = kmg::PointResidual(kind, A, k, j, i, ih2, dg)
                       + src(m, 0, k, j, i);
  });
  RestoreGhosts(lev);
  if (Drv()->count_nonfinite_) {
    nonfinite_def_ += kmg::CountNonFinite(def, nmmb_, nvar_, ngh_, nx, ny, nz);
  }
}

void MGKerrScalar::CalculateFASRHSPack() {
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
  const int kind = Drv()->kind_;
  SaveGhosts(lev);
  EdgeHook(lev, true);
  // persist only: uold = u was stored just before (multigrid_driver.cpp:762-763,
  // :1206-1207, multigrid_tasks.cpp:96-97), ahead of the hook: give uold the hooked edges
  // too. With hook_scope = operator u's ghosts stay framework-filled, as uold's.
  if (!Drv()->hook_operator_ && Drv()->uold_edge_sync_ != kmg::kSyncNone) {
    kmg::CopyMaskedEdges(uold_[lev].d_view, u, sync_.d_view, nmmb_, nvar_, ngh_, nx, ny,
                         nz);
  }
  par_for("MGKerrScalar::FASRHS", DevExeSpace(), 0, nmmb_-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real dx = brdx(m)*fac;
    const Real ih2 = 1.0/(dx*dx);
    VAcc<decltype(u)> A{u, coeff, m, 0};
    Real dg;
    src(m, 0, k, j, i) -= kmg::PointResidual(kind, A, k, j, i, ih2, dg);
  });
  RestoreGhosts(lev);
}

//----------------------------------------------------------------------------------------
// MGKerrScalarDriver

MGKerrScalarDriver::MGKerrScalarDriver(MeshBlockPack *pmbp, ParameterInput *pin,
                                       const std::string &block,
                                       const kmg::CoefSource &cs, const kmg::Manuf &mf)
    : MultigridDriver(pmbp, 1), cs_(cs), mf_(mf) {
  std::string kd = pin->GetOrAddString(block, "kind", "linear");
  if (kd == "linear") {
    kind_ = kmg::kKindLinear;
  } else if (kd == "psi") {
    kind_ = kmg::kKindPsi;
  } else if (kd == "chi") {
    kind_ = kmg::kKindChi;
  } else {
    Fatal(block + "/kind must be 'linear', 'psi' or 'chi'.");
  }
  psi_floor_ = pin->GetOrAddReal(block, "psi_floor", 0.05);
  coeff_cache_ = kmg::CoeffCacheEnabled();
  cache_check_ = coeff_cache_ && kmg::CoeffCacheCheck();
  zero_coarse_ = pin->GetOrAddBoolean(block, "zero_coarse", true);
  ncoeff_ = (kind_ == kmg::kKindLinear) ? kmg::kNCoeffScalar : kmg::kNCoeffKerr;
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
  } else {
    Fatal(block + "/smoother must be 'gs8' or 'rb'.");
  }
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
  std::string hs = pin->GetOrAddString(block, "hook_scope", "operator");
  if (hs == "operator") {
    hook_operator_ = true;
  } else if (hs == "persist") {
    hook_operator_ = false;
  } else {
    Fatal(block + "/hook_scope must be 'operator' or 'persist'.");
  }
  count_nonfinite_ = pin->GetOrAddBoolean(block, "count_nonfinite", false);

  robin_order_ = pin->GetOrAddInteger(block, "mg_robin_order", 1);
  std::string outer_bc_str = pin->GetOrAddString(block, "mg_outer_bc", "robin");
  BoundaryFlag outer_bc;
  if (outer_bc_str == "robin") {
    outer_bc = BoundaryFlag::mg_robin;
  } else if (outer_bc_str == "zerofixed") {
    outer_bc = BoundaryFlag::mg_zerofixed;
  } else {
    Fatal(block + "/mg_outer_bc must be 'robin' or 'zerofixed'.");
  }
  for (int f = 0; f < 6; ++f) {
    if (pmbp->pmesh->mesh_bcs[f] == BoundaryFlag::reflect) {
      mg_mesh_bcs_[f] = BoundaryFlag::mg_zerograd;
    } else if (pmbp->pmesh->mesh_bcs[f] != BoundaryFlag::periodic) {
      mg_mesh_bcs_[f] = outer_bc;
    }
  }
  mporder_ = 4;
  autompo_ = false;
  AllocateMultipoleCoefficients();
  fsubtract_average_ = false;

  int nghost = pin->GetOrAddInteger(block, "mg_nghost", 1);
  if (pin->GetOrAddBoolean(block, "root_on_host", false)) {
    Fatal(block + "/root_on_host = true is not supported (device-only coefficient fills).");
  }
  mgroot_ = new MGKerrScalar(this, nullptr, nghost, ncoeff_, false);
  mglevels_ = new MGKerrScalar(this, pmbp, nghost, ncoeff_);
  mglevels_->pbval = new MultigridBoundaryValues(pmbp, pin, false, mglevels_);
  mglevels_->pbval->InitializeBuffers(nvar_);
  mglevels_->pbval->RemapIndicesForMG();
  mglevels_->pbval->ComputePerLevelIndices();
}

MGKerrScalarDriver::~MGKerrScalarDriver() {
  delete mgroot_;
  delete mglevels_;
}

void MGKerrScalarDriver::Prepare() {
  PrepareForAMR();
  auto *lv = static_cast<MGKerrScalar*>(mglevels_);
  auto *rt = static_cast<MGKerrScalar*>(mgroot_);
  kmg::ComputeBlockEdgeMasks(pmy_pack_, blk_mask_, blk_phys_);
  kmg::ComputeRootEdgeMask(mg_mesh_bcs_, root_mask_, root_phys_);
  lv->SetEdgeMasks(blk_mask_, blk_phys_);
  rt->SetEdgeMasks(root_mask_, root_phys_);
  kmg::BuildSyncMask(blk_mask_, blk_phys_, pmy_pack_->nmb_thispack, uold_edge_sync_, false,
                     blk_sync_);
  kmg::BuildSyncMask(root_mask_, root_phys_, 1, uold_edge_sync_, true, root_sync_);
  lv->SetSyncMask(blk_sync_);
  rt->SetSyncMask(root_sync_);
  if (kind_ == kmg::kKindLinear) {
    lv->FillAllLevels(cs_, mf_);
    rt->FillAllLevels(cs_, mf_);
    FillOctetCoeffs(false);
  } else {
    // matter: finest level -> block levels -> root/octets -> root levels (the order of
    // MGCFCConformalFactorDriver::Solve, mg_cfc_conformal_factor.cpp:547-560); then the
    // background, restricted along with it, is overwritten analytically everywhere.
    // The block levels' finest background is filled once: the matter fill and the
    // restrictions leave its channels alone (Multigrid::RestrictCoefficients writes levels
    // < finest only; TransferCoeffToRoot reads level-0 interiors, which it has written),
    // so only the coarser block levels need the second fill (bitwise as before, K-005 B1)
    const int lfin = lv->GetNumberOfLevels() - 1;
    key_.ok = false;
    lv->FillAllLevelsBg(cs_, 1, 0);   // extents and storage only
    const bool shared = CopyBgFromDonor(lfin, lfin);
    if (!shared) { lv->FillAllLevelsBg(cs_, lfin, lfin); }
    rt->FillAllLevelsBg(cs_);
    if (matter_ext_) {
      lv->FillFinestMatterExt(ext_m_, ext_da2_, ext_dpsi_, kind_ == kmg::kKindChi, ext_ngh_);
    } else {
      lv->FillFinestMatterManuf(kind_, cs_, mf_, mm_);
    }
    lv->RestrictCoefficients();
    TransferCoeffToRoot();
    RestrictCoeffOctets();
    rt->RestrictCoefficients();
    if (lfin > 0 && !(shared && CopyBgFromDonor(0, lfin - 1))) {
      lv->FillAllLevelsBg(cs_, 0, lfin - 1);
    }
    rt->FillAllLevelsBg(cs_);
    FillOctetCoeffs(true);
    if (shared) { ++bg_copies_; } else { ++bg_fills_; }
    if (shared && cache_check_) {
      // KERR_MG_CACHECHECK: refill the copied background and count what differs
      std::vector<DvceArray5D<Real>> save;
      for (int lev = 0; lev <= lfin; ++lev) {
        save.push_back(kmg::CloneView(lv->CoeffAtLevel(lev).d_view));
      }
      lv->FillAllLevelsBg(cs_, 0, lfin);
      for (int lev = 0; lev <= lfin; ++lev) {
        const long c = kmg::CountDiff(save[lev], lv->CoeffAtLevel(lev).d_view);
        check_diff_ += (c < 0) ? 1 : c;
      }
      ++check_n_;
    }
    kmg::CaptureKey(key_, cs_, pmy_mesh_, pmy_pack_, nreflevel_, noctets_);
    key_ptr_.clear();
    for (int lev = 0; lev <= lfin; ++lev) {
      key_ptr_.push_back(lv->CoeffAtLevel(lev).d_view.data());
    }
  }
  SetupMultigrid(0.0, false);
  Kokkos::fence();
}

bool MGKerrScalarDriver::BgIntactFor(const kmg::CoefSource &cs) const {
  if (kind_ == kmg::kKindLinear) return false;
  if (!kmg::KeyMatches(key_, cs, pmy_mesh_, pmy_pack_, nreflevel_, noctets_)) return false;
  auto *lv = static_cast<MGKerrScalar*>(mglevels_);
  if (static_cast<int>(key_ptr_.size()) != lv->GetNumberOfLevels()) return false;
  for (int lev = 0; lev < lv->GetNumberOfLevels(); ++lev) {
    if (lv->CoeffAtLevel(lev).d_view.data() != key_ptr_[lev]) return false;
  }
  return true;
}

bool MGKerrScalarDriver::CopyBgFromDonor(int lo, int hi) {
  MGKerrScalarDriver *d = bg_donor_;
  if (!coeff_cache_ || d == nullptr || d == this || kind_ == kmg::kKindLinear ||
      d->ncoeff_ != ncoeff_ || !d->BgIntactFor(cs_)) {
    return false;
  }
  auto *lv = static_cast<MGKerrScalar*>(mglevels_);
  auto *dl = static_cast<MGKerrScalar*>(d->mglevels_);
  if (dl->GetNumberOfLevels() != lv->GetNumberOfLevels()) return false;
  for (int lev = lo; lev <= hi; ++lev) {
    const auto &a = lv->CoeffAtLevel(lev).d_view, &b = dl->CoeffAtLevel(lev).d_view;
    for (int r = 0; r < 5; ++r) {
      if (a.extent(r) != b.extent(r)) return false;
    }
  }
  for (int lev = lo; lev <= hi; ++lev) {
    kmg::CopyBgChannels(lv->CoeffAtLevel(lev).d_view, dl->CoeffAtLevel(lev).d_view);
  }
  return true;
}

void MGKerrScalarDriver::Solve(Driver *pdriver, int stage, Real dt) {
  Prepare();
  if (zero_coarse_) ZeroHierarchy();
  SolveMG(pdriver);
  Kokkos::fence();
}

void MGKerrScalarDriver::SetMatter(const DvceArray5D<Real> &mt,
                                   const DvceArray5D<Real> &da2,
                                   const DvceArray5D<Real> &dpsi, int ngh) {
  matter_ext_ = true;
  ext_m_ = mt;
  ext_da2_ = da2;
  ext_dpsi_ = dpsi;
  ext_ngh_ = ngh;
}

void MGKerrScalarDriver::SeedInitialGuess(const DvceArray5D<Real> &u, int ngh) {
  mglevels_->ReallocateForAMR();
  mglevels_->LoadFinestData(u, 0, ngh);
}

void MGKerrScalarDriver::RetrieveSolution(DvceArray5D<Real> &dst) {
  mglevels_->RetrieveResult(dst, 0, pmy_pack_->pmesh->mb_indcs.ng);
}

void MGKerrScalarDriver::ResetSolution() {
  ZeroHierarchy();
  auto *lv = static_cast<MGKerrScalar*>(mglevels_);
  const int nl = lv->GetNumberOfLevels();
  Kokkos::deep_copy(lv->UAt(nl-1).d_view, 0.0);
  Kokkos::deep_copy(lv->UAt(nl-1).h_view, 0.0);
  Kokkos::deep_copy(lv->SrcAt(nl-1).d_view, 0.0);
  Kokkos::deep_copy(lv->SrcAt(nl-1).h_view, 0.0);
  if (init_noise_ > 0.0) {
    kmg::AddNoise(lv->UAt(nl-1).d_view, lv->Extents(), lv->NumMB(), 1, lv->GetGhostCells(),
                  lv->LevelN(nl-1, 0), lv->LevelN(nl-1, 1), lv->LevelN(nl-1, 2),
                  init_noise_);
    Kokkos::deep_copy(lv->UAt(nl-1).h_view, lv->UAt(nl-1).d_view);
  }
  lv->nonfinite_def_ = 0;
  static_cast<MGKerrScalar*>(mgroot_)->nonfinite_def_ = 0;
  nonfinite_oct_ = 0;
  Kokkos::fence();
}

std::vector<Real> MGKerrScalarDriver::RunCycles(Driver *pdriver, int ncyc) {
  std::vector<Real> d;
  d.push_back(TotalDefectNorm());
  for (int n = 0; n < ncyc; ++n) {
    SolveVCycle(pdriver, npresmooth_, npostsmooth_);
    d.push_back(TotalDefectNorm());
  }
  Kokkos::fence();
  return d;
}

void MGKerrScalarDriver::FillOctetCoeffs(bool bg_only) {
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
    }
    blk.template modify<HostExeSpace>();
    blk.template sync<DevExeSpace>();
    DvceArray5D<Real> tmp("kmg_octcoeff", noct, ncoeff_, nc, nc, nc);
    if (bg_only) {
      kmg::FillLevelBg(tmp, blk, noct, ngh, 2, 2, 2, cs_);
    } else {
      kmg::FillLevelCoeffs(tmp, blk, noct, ngh, 2, 2, 2, cs_, mf_);
    }
    auto h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), tmp);
    for (int o = 0; o < noct; ++o) {
      MGOctet &oct = octets_[l][o];
      for (int c = 0; c < ncoeff_; ++c) {
        if (bg_only && !kmg::IsBgChannel(c)) continue;
        for (int k = 0; k < nc; ++k) {
          for (int j = 0; j < nc; ++j) {
            for (int i = 0; i < nc; ++i) { oct.Coeff(c, k, j, i) = h(o, c, k, j, i); }
          }
        }
      }
    }
  }
}

void MGKerrScalarDriver::TransferCoeffToRoot() {
  const int nc = ncoeff_;
  auto *lv = static_cast<MGKerrScalar*>(mglevels_);
  auto *rt = static_cast<MGKerrScalar*>(mgroot_);
  auto &coeff_lvl = lv->CoeffAtLevel(0);
  const int ngh_mb = lv->GetGhostCells();
  const int nmmb = pmy_pack_->nmb_thispack - 1;
  const int padding = nslist_[global_variable::my_rank];
  DualArray2D<Real> coeffbuf;
  Kokkos::realloc(coeffbuf, nc, nbtotal_);
  auto cb = coeffbuf.d_view;
  auto cl = coeff_lvl.d_view;
  par_for("MGKerrScalarDriver::SaveCoeffToRoot", DevExeSpace(), 0, nmmb,
  KOKKOS_LAMBDA(const int m) {
    for (int v = 0; v < nc; ++v) {
      cb(v, m + padding) = cl(m, v, ngh_mb, ngh_mb, ngh_mb);
    }
  });
  coeffbuf.template modify<DevExeSpace>();
  coeffbuf.template sync<HostExeSpace>();
#if MPI_PARALLEL_ENABLED
  for (int v = 0; v < nc; ++v) {
    MPI_Allgatherv(MPI_IN_PLACE, nblist_[global_variable::my_rank], MPI_ATHENA_REAL,
                   &coeffbuf.h_view(v, 0), nblist_, nslist_, MPI_ATHENA_REAL,
                   MPI_COMM_WORLD);
  }
#endif
  const auto loc = pmy_mesh_->lloc_eachmb;
  const int ngh = rt->GetGhostCells();
  auto &coeff_root = rt->CoeffAtLevel(rt->GetNumberOfLevels() - 1);
  // the root array may have been written on the device only (FillAllLevelsBg)
  Kokkos::deep_copy(coeff_root.h_view, coeff_root.d_view);
  auto rh = coeff_root.h_view;
  for (int n = 0; n < nbtotal_; ++n) {
    const int i = static_cast<int>(loc[n].lx1);
    const int j = static_cast<int>(loc[n].lx2);
    const int k = static_cast<int>(loc[n].lx3);
    if (loc[n].level == locrootlevel_) {
      for (int v = 0; v < nc; ++v) {
        rh(0, v, k + ngh, j + ngh, i + ngh) = coeffbuf.h_view(v, n);
      }
    } else {
      LogicalLocation oloc;
      oloc.lx1 = (loc[n].lx1 >> 1);
      oloc.lx2 = (loc[n].lx2 >> 1);
      oloc.lx3 = (loc[n].lx3 >> 1);
      oloc.level = loc[n].level - 1;
      const int olev = oloc.level - locrootlevel_;
      const int oid = octetmap_[olev][oloc];
      MGOctet &oct = octets_[olev][oid];
      for (int v = 0; v < nc; ++v) {
        oct.Coeff(v, (k & 1) + ngh, (j & 1) + ngh, (i & 1) + ngh) = coeffbuf.h_view(v, n);
      }
    }
  }
  Kokkos::deep_copy(coeff_root.d_view, coeff_root.h_view);
}

void MGKerrScalarDriver::SmoothOctet(MGOctet &oct, int rlev, int color) {
  const int ngh = mgroot_->GetGhostCells();
  const Real dx = mgroot_->GetRootDx()/static_cast<Real>(1 << rlev);
  const Real ih2 = 1.0/(dx*dx);
  const int c = color ^ coffset_;
  OctAcc A{&oct};
  if (smoother_ == 0) {
    for (int sc = 0; sc < 8; ++sc) {
      const int oi = sc & 1, oj = (sc >> 1) & 1, ok = (sc >> 2) & 1;
      if (((oi + oj + ok) & 1) != c) continue;
      const int k = ngh + ok, j = ngh + oj, i = ngh + oi;
      Real dg;
      const Real r = kmg::PointResidual(kind_, A, k, j, i, ih2, dg) + oct.Src(0, k, j, i);
      const Real p0 = (kind_ == kmg::kKindPsi) ? oct.Coeff(kChPsi0, k, j, i) : 0.0;
      oct.U(0, k, j, i) = kmg::FloorPsi(kind_, oct.U(0, k, j, i) + omega_*r/dg, p0,
                                        psi_floor_);
    }
  } else {
    Real nw[8];
    for (int sc = 0; sc < 8; ++sc) {
      const int oi = sc & 1, oj = (sc >> 1) & 1, ok = (sc >> 2) & 1;
      if (((oi + oj + ok) & 1) != c) continue;
      const int k = ngh + ok, j = ngh + oj, i = ngh + oi;
      Real dg;
      const Real r = kmg::PointResidual(kind_, A, k, j, i, ih2, dg) + oct.Src(0, k, j, i);
      const Real p0 = (kind_ == kmg::kKindPsi) ? oct.Coeff(kChPsi0, k, j, i) : 0.0;
      nw[sc] = kmg::FloorPsi(kind_, oct.U(0, k, j, i) + omega_*r/dg, p0, psi_floor_);
    }
    for (int sc = 0; sc < 8; ++sc) {
      const int oi = sc & 1, oj = (sc >> 1) & 1, ok = (sc >> 2) & 1;
      if (((oi + oj + ok) & 1) != c) continue;
      oct.U(0, ngh + ok, ngh + oj, ngh + oi) = nw[sc];
    }
  }
}

void MGKerrScalarDriver::CalculateDefectOctet(MGOctet &oct, int rlev) {
  const int ngh = mgroot_->GetGhostCells();
  const Real dx = mgroot_->GetRootDx()/static_cast<Real>(1 << rlev);
  const Real ih2 = 1.0/(dx*dx);
  OctAcc A{&oct};
  for (int k = ngh; k <= ngh+1; ++k) {
    for (int j = ngh; j <= ngh+1; ++j) {
      for (int i = ngh; i <= ngh+1; ++i) {
        Real dg;
        const Real d = kmg::PointResidual(kind_, A, k, j, i, ih2, dg)
                     + oct.Src(0, k, j, i);
        oct.Def(0, k, j, i) = d;
        if (count_nonfinite_ && kmg::NonFinite(d)) { ++nonfinite_oct_; }
      }
    }
  }
}

void MGKerrScalarDriver::CalculateFASRHSOctet(MGOctet &oct, int rlev) {
  const int ngh = mgroot_->GetGhostCells();
  const Real dx = mgroot_->GetRootDx()/static_cast<Real>(1 << rlev);
  const Real ih2 = 1.0/(dx*dx);
  OctAcc A{&oct};
  for (int k = ngh; k <= ngh+1; ++k) {
    for (int j = ngh; j <= ngh+1; ++j) {
      for (int i = ngh; i <= ngh+1; ++i) {
        Real dg;
        oct.Src(0, k, j, i) -= kmg::PointResidual(kind_, A, k, j, i, ih2, dg);
      }
    }
  }
}

//----------------------------------------------------------------------------------------
// test helpers

KerrOpStats MGKerrScalarDriver::OperatorTest(bool exact_edges, bool poison_corners) {
  auto *lv = static_cast<MGKerrScalar*>(mglevels_);
  const int lev = lv->GetNumberOfLevels() - 1;
  const int nmb = lv->NumMB();
  const int ngh = lv->GetGhostCells();
  const int nx = lv->LevelN(lev, 0), ny = lv->LevelN(lev, 1), nz = lv->LevelN(lev, 2);
  auto u = lv->UAt(lev).d_view;
  auto coeff = lv->CoeffAtLevel(lev).d_view;
  kmg::FillManufU(u, lv->Extents(), nmb, ngh, nx, ny, nz, mf_);
  if (!exact_edges) {
    kmg::ApplyEdgeFill(u, lv->Mask().d_view, nmb, 1, ngh, nx, ny, nz, edge_fill_,
                       edge_poison_);
  }
  if (poison_corners) { kmg::PoisonCorners(u, nmb, 1, ngh, nx, ny, nz); }
  DvceArray5D<Real> out("kmg_optest", nmb, 2, nz, ny, nx);
  DvceArray1D<Real> rdx("kmg_rdx", nmb);
  {
    auto h = Kokkos::create_mirror_view(rdx);
    for (int m = 0; m < nmb; ++m) { h(m) = lv->LevelDx(lev, m); }
    Kokkos::deep_copy(rdx, h);
  }
  const int kind = kind_;
  par_for("kmg::OperatorTest", DevExeSpace(), 0, nmb-1, 0, nz-1, 0, ny-1, 0, nx-1,
  KOKKOS_LAMBDA(const int m, const int kk, const int jj, const int ii) {
    const int k = kk + ngh, j = jj + ngh, i = ii + ngh;
    const Real ih2 = 1.0/(rdx(m)*rdx(m));
    kmg::ViewAcc<decltype(u)> A{u, coeff, m, 0};
    const Real S = kmg::ScalarS(A, k, j, i);
    const Real S7 = u(m, 0, k, j, i+1) + u(m, 0, k, j, i-1) + u(m, 0, k, j+1, i)
                  + u(m, 0, k, j-1, i) + u(m, 0, k+1, j, i) + u(m, 0, k-1, j, i)
                  - 6.0*u(m, 0, k, j, i);
    Real dg;
    out(m, 0, kk, jj, ii) = kmg::PointResidual(kind, A, k, j, i, ih2, dg);
    out(m, 1, kk, jj, ii) = S - S7;
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
          st.maxd[cls] = std::max(st.maxd[cls], std::abs(d));
          st.l2[cls] += d*d*dv;
          st.vol[cls] += dv;
          ++st.ncell[cls];
          st.max_s7 = std::max(st.max_s7, std::abs(h(m, 1, kk, jj, ii)));
        }
      }
    }
  }
  return st;
}

void MGKerrScalarDriver::ErrorVsManuf(Real &emax, Real &erms) {
  auto *lv = static_cast<MGKerrScalar*>(mglevels_);
  const int lev = lv->GetNumberOfLevels() - 1;
  const int nmb = lv->NumMB();
  const int ngh = lv->GetGhostCells();
  const int nx = lv->LevelN(lev, 0), ny = lv->LevelN(lev, 1), nz = lv->LevelN(lev, 2);
  auto h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), lv->UAt(lev).d_view);
  emax = 0.0;
  Real s2 = 0.0, vol = 0.0;
  for (int m = 0; m < nmb; ++m) {
    const RegionSize &b = lv->Extents().h_view(m);
    const Real dx1 = (b.x1max - b.x1min)/nx, dx2 = (b.x2max - b.x2min)/ny;
    const Real dx3 = (b.x3max - b.x3min)/nz, dv = dx1*dx2*dx3;
    for (int k = 0; k < nz; ++k) {
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          Real us, du[3], ddu[6];
          kmg::ManufEval(mf_, b.x1min + (i + 0.5)*dx1, b.x2min + (j + 0.5)*dx2,
                         b.x3min + (k + 0.5)*dx3, us, du, ddu);
          const Real e = h(m, 0, k + ngh, j + ngh, i + ngh) - us;
          emax = std::max(emax, std::abs(e));
          s2 += e*e*dv;
          vol += dv;
        }
      }
    }
  }
  erms = (vol > 0.0) ? std::sqrt(s2/vol) : 0.0;
}

void MGKerrScalarDriver::ErrorRadial(const Real c[3], const std::vector<Real> &e,
                          std::vector<Real> &emax, std::vector<Real> &s2,
                          std::vector<Real> &vol, std::vector<long> &n) {
  auto *lv = static_cast<MGKerrScalar*>(mglevels_);
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
          {
            Real us, du[3], ddu[6];
            kmg::ManufEval(mf_, X[0], X[1], X[2], us, du, ddu);
            const Real e = h(m, 0, k + ngh, j + ngh, i + ngh) - us;
            ea = std::abs(e);
            e2 = e*e;
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

void MGKerrScalarDriver::ErrorVsManufByClass(Real emax[3], Real erms[3], long ncell[3]) {
  auto *lv = static_cast<MGKerrScalar*>(mglevels_);
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
          Real us, du[3], ddu[6];
          kmg::ManufEval(mf_, b.x1min + (i + 0.5)*dx1, b.x2min + (j + 0.5)*dx2,
                         b.x3min + (k + 0.5)*dx3, us, du, ddu);
          const Real e = h(m, 0, k + ngh, j + ngh, i + ngh) - us;
          emax[cls] = std::max(emax[cls], std::abs(e));
          s2[cls] += e*e*dv;
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

void MGKerrScalarDriver::NonFiniteCounts(long c[3]) const {
  c[0] = static_cast<const MGKerrScalar*>(mglevels_)->nonfinite_def_;
  c[1] = static_cast<const MGKerrScalar*>(mgroot_)->nonfinite_def_;
  c[2] = nonfinite_oct_;
}

void MGKerrScalarDriver::EdgeCounts(long c[3]) {
  kmg::CountEdges(blk_mask_, blk_phys_, pmy_pack_->nmb_thispack, c);
}

}  // namespace cfc
