#ifndef CFC_MG_KERR_VECTOR_HPP_
#define CFC_MG_KERR_VECTOR_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mg_kerr_vector.hpp
//! \brief W3c (K-003): the Kerr vector multigrid for -DL V = s on the conformal metric
//! (operator, channels and outer boundary: kerr_mg_vops.hpp). nvar = 3 (V^x, V^y, V^z),
//! 99 coefficient channels, all analytic per level/root/octet. The right-hand side is the
//! finest level's src (as the flat MGCFCVectorPoisson's LoadSource).
//!
//! Smoothers (per framework colour = one parity of i + j + k):
//!  - gs8: point-block Gauss-Seidel over the 4 parity classes of the colour in turn
//!    (race-free for the 19-point stencil), V_c += omega h^2 Dinv r;
//!  - rb:  point-block Jacobi within the colour (from a snapshot).
//! Hook (before every sweep, defect and FAS right-hand side, and after each gs8 class):
//! [corner poison], the kelvin face fill on the physical faces (<block>/mg_outer_bc_vec =
//! kelvin, the default; or zerofixed), then
//! Rule R / B / off (+ E1 poison) on the unsafe edges (kerr_mg_ops.hpp). The framework's
//! own physical boundary is mg_zerofixed on those faces (it fills the whole face; the hook
//! overwrites the interior transverse range); the root's edge mask is built as if the faces
//! were mg_multipole so its physical edges are unsafe (Rule R) as on the blocks.
//! Octets get no hook: an octet on a physical face keeps the framework's zerofixed ghosts
//! (counted as boundary_octets in Prepare; the tests keep the refinement inside).

#include <string>
#include <vector>

#include "athena.hpp"
#include "multigrid/multigrid.hpp"
#include "kerr_mg_ops.hpp"
#include "kerr_mg_vops.hpp"
#include "mg_kerr_scalar.hpp"

class MeshBlockPack;
class ParameterInput;
class Driver;

namespace cfc {

class MGKerrVectorDriver;

class MGKerrVector : public Multigrid {
 public:
  MGKerrVector(MultigridDriver *pmd, MeshBlockPack *pmbp, int nghost, bool on_host = false);
  ~MGKerrVector() = default;

  void SmoothPack(int color) final;
  void CalculateDefectPack() final;
  void CalculateFASRHSPack() final;

  void SetMasks(const DualArray1D<int> &mask, const DualArray1D<int> &phys,
                const DualArray1D<int> &kfaces) {
    mask_ = mask;
    phys_ = phys;
    kfaces_ = kfaces;
  }
  //! the coefficients on every level; returns the cells with det(-dS/dV_c) <= 0
  long FillAllLevels(const kmg::CoefSource &cs);
  void BuildExtents();
  int NumMB() const { return nmmb_; }
  int LevelN(int lev, int dir) const {
    const int ll = nlevel_ - 1 - lev;
    return ((dir == 0) ? indcs_.nx1 : ((dir == 1) ? indcs_.nx2 : indcs_.nx3)) >> ll;
  }
  DualArray5D<Real> &UAt(int l) { return u_[l]; }
  DualArray5D<Real> &SrcAt(int l) { return src_[l]; }
  const DualArray1D<RegionSize> &Extents() const { return blk_; }
  const DualArray1D<int> &Mask() const { return mask_; }
  const DualArray1D<int> &Phys() const { return phys_; }
  void SetSyncMask(const DualArray1D<int> &sync) { sync_ = sync; }
  //! the hook on level lev (see the file comment); poison_ok: E1 poison allowed
  void Hook(int lev, bool poison_ok);
  //! hook_scope = operator: see MGKerrScalar::SaveGhosts (the kelvin faces too)
  void SaveGhosts(int lev);
  void RestoreGhosts(int lev);
  Real LevelDx(int lev, int m) const;

  long nonfinite_def_ = 0;

 private:
  MGKerrVectorDriver *Drv() const;
  DualArray1D<int> sync_;     // uold edge sync mask (kmg::BuildSyncMask)
  DualArray1D<int> mask_, phys_, kfaces_;
  DualArray1D<RegionSize> blk_;
  DvceArray5D<Real> scratch_;
  DvceArray5D<Real> gsave_;   // hook_scope = operator: u's ghosts (finest-level shape)
};

class MGKerrVectorDriver : public MultigridDriver {
 public:
  MGKerrVectorDriver(MeshBlockPack *pmbp, ParameterInput *pin, const std::string &block,
                     const kmg::CoefSource &cs, const kmg::VManuf &vm);
  ~MGKerrVectorDriver();

  //! Prepare, then mp_outer passes of {moments, SolveMG}
  void Solve(Driver *pdriver, int stage, Real dt = 0.0) override;

  //! masks, coefficients on every level/root/octet, SetupMultigrid
  void Prepare();
  //! u = 0 everywhere, finest src = s = -DL V* (the manufactured right-hand side), or
  //! (src_blob_) src = V*'s components themselves (a source with a monopole; no exact
  //! solution: for the box-doubling probes)
  void ResetSolution();
  //! the moments of q (mp_source) on the finest level, reduced over ranks
  void ComputeMoments();
  //! the defect norm before and after each of ncyc V-cycles; the moments are recomputed
  //! before every cycle for mp_source = self, once for rhs
  std::vector<Real> RunCycles(Driver *pdriver, int ncyc);
  //! def = s + S(V*)/h^2 on the finest block level, max over the components, by class;
  //! KerrOpStats::max_s7 holds max |S - S_flat|/h^2 here (0 on a flat metric)
  KerrOpStats OperatorTest(bool exact_edges, bool poison_corners);
  //! |V - V*| on the finest block level by class (max over components)
  void ErrorVsManufByClass(Real emax[3], Real erms[3], long ncell[3]);
  //! V at points (x, y, z) (3 per point; finest leaf cell containing it, this rank;
  //! zero where not owned) and the cell centre found (3 per point; 0 if not owned)
  void Probe(const std::vector<Real> &pts, std::vector<Real> &val, std::vector<Real> &ctr,
             std::vector<int> &own);
  void EdgeCounts(long c[3]);
  int RootMask() const { return root_mask_.h_view(0); }
  void NonFiniteCounts(long c[3]) const;

  // settings (public for the test harness)
  kmg::EdgeFill edge_fill_ = kmg::EdgeFill::rule_r;
  bool edge_poison_ = false;
  int uold_edge_sync_ = kmg::kSyncFree;   // block/uold_edge_sync = none | free | all
  bool hook_operator_ = true;   // block/hook_scope = operator (default) | persist
  bool corner_poison_ = false;
  bool count_nonfinite_ = false;
  int smoother_ = 0;          // 0 gs8, 1 rb
  Real omega_ = 1.0;
  bool kelvin_ = true;        // block/mg_outer_bc_vec = kelvin | zerofixed
  bool mp_self_ = true;       // block/mp_source = self | rhs
  int mp_outer_ = 3;
  bool src_blob_ = false;
  Real mpo_[3] = {0.0, 0.0, 0.0};   // the multipole origin (block/mp_x01..3)
  kmg::VMoments mom_;
  kmg::CoefSource cs_;
  kmg::VManuf vm_;
  long bad_dinv_[3] = {0, 0, 0};    // blocks, root, octets
  long boundary_octets_ = 0;

  void SmoothOctet(MGOctet &oct, int rlev, int color) override;
  void CalculateDefectOctet(MGOctet &oct, int rlev) override;
  void CalculateFASRHSOctet(MGOctet &oct, int rlev) override;

  long nonfinite_oct_ = 0;

 private:
  void FillOctetCoeffs();
  DualArray1D<int> blk_sync_, root_sync_;
  DualArray1D<int> blk_mask_, blk_phys_, root_mask_, root_phys_, blk_kf_, root_kf_;
};

}  // namespace cfc

#endif  // CFC_MG_KERR_VECTOR_HPP_
