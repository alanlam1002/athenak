#ifndef CFC_MG_KERR_SCALAR_HPP_
#define CFC_MG_KERR_SCALAR_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mg_kerr_scalar.hpp
//! \brief W3a (K-003): the variable-coefficient scalar multigrid on the Kerr conformal
//! metric, for the linear test equation
//!     A u = -Dt_i(gt^ij d_j u) + c u = s
//! (19-point flux stencil, kerr_methods.tex sec. 5.1; Rule R on the unsafe edge ghosts,
//! sec. 5.3). The W3b drivers (eq:cpsi, eq:cchi) reuse its operator, smoother and hooks.
//!
//! Coefficient channels (kerr_mg_ops.hpp): 0-17 gt^ij at the lower faces, 18 c, 19 s; all
//! analytic at every level (blocks, root, octets), never restricted.
//! W3b (kind = psi | chi; eq:cpsi, eq:cchi): 25 channels. The background ones (faces,
//! Rt/8, psi0, Ahat0^2, chi0) are analytic at every level, root and octet; the matter ones
//! (E* or E* + 2S*, Ahat^2 - Ahat0^2, delta psi) are formed on the finest level and
//! restricted through the block levels, the root (TransferCoeffToRoot) and the octets
//! (kerr_methods.tex sec. 4.2). The pointwise terms c, s(u), ds/du come from
//! kmg::PointTerm; smooth: u += omega def/(c - ds/du - SDiag/h^2), with the flat driver's
//! psi floor for kind = psi.
//! Discrete form (h the level's cell width, S = h^2 Dt^2 u from ScalarS):
//!     def = s + src + S/h^2 - c u,  FAS: src += (c u - S/h^2) - s,
//!     smooth: u += omega def/(c - SDiag/h^2).
//! Smoothers: gs8, Gauss-Seidel over the 8 parity classes of (i, j, k) (each framework
//! colour = the 4 classes of its parity, swept in turn; race-free for the 19-point
//! stencil); rb, the framework's red-black split with Jacobi inside each colour (red-black
//! Gauss-Seidel is not race-free for this stencil, so the comparison uses the
//! deterministic variant).
//! Edge hook, before every sweep, defect and FAS right-hand side, and between and after
//! the gs8 sub-sweeps: [poison the unsafe edge ghosts with NaN (E1)], then Rule R
//! (or B, or nothing), from the block's own mask. The root's mask marks the edges beside
//! a Robin/multipole face; the octets get no hook (their edges are composed in full by
//! SetBoundariesOctets, multigrid_driver.cpp:1618ff).

#include <string>
#include <vector>

#include "athena.hpp"
#include "multigrid/multigrid.hpp"
#include "kerr_mg_ops.hpp"

class MeshBlockPack;
class ParameterInput;
class Driver;

namespace cfc {

class MGKerrScalarDriver;

class MGKerrScalar : public Multigrid {
 public:
  MGKerrScalar(MultigridDriver *pmd, MeshBlockPack *pmbp, int nghost, int ncoeff,
               bool on_host = false);
  ~MGKerrScalar() = default;

  void SmoothPack(int color) final;
  void CalculateDefectPack() final;
  void CalculateFASRHSPack() final;

  bool IsRoot() const { return pmy_pack_ == nullptr; }
  void SetEdgeMasks(const DualArray1D<int> &mask, const DualArray1D<int> &phys) {
    mask_ = mask;
    phys_ = phys;
  }
  //! all levels' coefficients from (cs, mf)
  void FillAllLevels(const kmg::CoefSource &cs, const kmg::Manuf &mf);
  //! W3b: the background channels on all levels (see kmg::FillLevelBg)
  void FillAllLevelsBg(const kmg::CoefSource &cs);
  //! W3b test: the manufactured matter channels on the finest level
  //! W4 production: the matter channels on the finest level from mesh arrays (channel 0,
  //! ghost depth ngh): M, DA2 and (has_dpsi) DPsi, interior cells only
  void FillFinestMatterExt(const DvceArray5D<Real> &mt, const DvceArray5D<Real> &da2,
                           const DvceArray5D<Real> &dpsi, bool has_dpsi, int ngh);
  void FillFinestMatterManuf(int kind, const kmg::CoefSource &cs, const kmg::Manuf &mf,
                             const kmg::MatterManuf &mm);
  //! the extent of each "m" (blocks: mb_size; root: the mesh)
  void BuildExtents();
  int NumMB() const { return nmmb_; }
  int LevelN(int lev, int dir) const {
    const int ll = nlevel_ - 1 - lev;
    return ((dir == 0) ? indcs_.nx1 : ((dir == 1) ? indcs_.nx2 : indcs_.nx3)) >> ll;
  }
  DualArray5D<Real> &UAt(int l) { return u_[l]; }
  DualArray5D<Real> &DefAt(int l) { return def_[l]; }
  DualArray5D<Real> &SrcAt(int l) { return src_[l]; }
  const DualArray1D<RegionSize> &Extents() const { return blk_; }
  const DualArray1D<int> &Mask() const { return mask_; }
  const DualArray1D<int> &Phys() const { return phys_; }
  void SetSyncMask(const DualArray1D<int> &sync) { sync_ = sync; }
  //! the edge hook on level lev (see the file comment)
  void EdgeHook(int lev, bool poison_ok);
  //! hook_scope = operator: save u's ghosts before the hook / restore them after the
  //! kernels, so the hooked values never reach the framework's transfer (no-ops if persist)
  void SaveGhosts(int lev);
  void RestoreGhosts(int lev);
  //! level width of block m's cells
  Real LevelDx(int lev, int m) const;

  long nonfinite_def_ = 0;   // non-finite defects seen (count_nonfinite_ only)

 private:
  MGKerrScalarDriver *Drv() const;
  DualArray1D<int> sync_;     // uold edge sync mask (kmg::BuildSyncMask)
  DualArray1D<int> mask_, phys_;
  DualArray1D<RegionSize> blk_;
  DvceArray5D<Real> scratch_;
  DvceArray5D<Real> gsave_;   // hook_scope = operator: u's ghosts (finest-level shape)
};

//! \struct KerrOpStats
//! \brief operator-only check on the finest block level (OperatorTest): max |def| and
//! the volume-weighted sum of def^2 by cell class (0 bulk, 1 next to an unsafe edge on the
//! domain boundary, 2 next to an unsafe interior edge), the cell counts, non-finite
//! defects, and max |S - S7| (the flat 7-point Laplacian; meaningful for a flat metric).
struct KerrOpStats {
  Real maxd[3] = {0, 0, 0};
  Real l2[3] = {0, 0, 0};
  Real vol[3] = {0, 0, 0};
  long ncell[3] = {0, 0, 0};
  long nonfinite = 0;
  Real max_s7 = 0.0;
};

class MGKerrScalarDriver : public MultigridDriver {
 public:
  MGKerrScalarDriver(MeshBlockPack *pmbp, ParameterInput *pin, const std::string &block,
                     const kmg::CoefSource &cs, const kmg::Manuf &mf);
  ~MGKerrScalarDriver();

  void Solve(Driver *pdriver, int stage, Real dt = 0.0) override;

  //! masks, coefficients on every level/root/octet, SetupMultigrid
  void Prepare();
  //! u = 0 and src = 0 everywhere (all levels, root, octets)
  void ResetSolution();
  //! the defect norm before and after each of ncyc V-cycles
  std::vector<Real> RunCycles(Driver *pdriver, int ncyc);
  //! the operator applied to u* on the finest block level (see KerrOpStats);
  //! exact_edges: keep u* in the unsafe edge ghosts, else apply the hook's edge fill
  KerrOpStats OperatorTest(bool exact_edges, bool poison_corners);
  //! max and volume-weighted rms of u - u* on the finest block level
  void ErrorVsManuf(Real &emax, Real &erms);
  //! the same split by cell class (KerrOpStats: 0 bulk, 1 boundary edge, 2 interior edge)
  void ErrorVsManufByClass(Real emax[3], Real erms[3], long ncell[3]);
  //! V2a-e radial profile: max |u - u*| (max over components), sum (u - u*)^2 dV, volume
  //! and cells in the shells [e[s], e[s+1]) of r = |x - c| (the last shell open), finest
  //! block level, this rank
  void ErrorRadial(const Real c[3], const std::vector<Real> &e, std::vector<Real> &emax,
                   std::vector<Real> &s2, std::vector<Real> &vol, std::vector<long> &n);
  //! total, on-boundary and interior unsafe block edges (this rank)
  void EdgeCounts(long c[3]);
  int RootMask() const { return root_mask_.h_view(0); }
  //! the block levels (test harness: the K-004 A5 solution dump)
  MGKerrScalar *Levels() { return static_cast<MGKerrScalar*>(mglevels_); }
  //! non-finite defects counted since ResetSolution: [0] block levels, [1] root, [2] octets
  void NonFiniteCounts(long c[3]) const;

  // settings (public for the test harness)
  kmg::EdgeFill edge_fill_ = kmg::EdgeFill::rule_r;
  bool edge_poison_ = false;
  int uold_edge_sync_ = kmg::kSyncFree;   // block/uold_edge_sync = none | free | all
  bool hook_operator_ = true;   // block/hook_scope = operator (default) | persist
  bool corner_poison_ = false;
  bool count_nonfinite_ = false;
  int smoother_ = 0;          // 0 gs8, 1 rb (Jacobi within colour)
  Real omega_ = 1.0;
  //! V2a-U test: ResetSolution adds init_noise_ * xi (kmg::AddNoise) to the finest u
  Real init_noise_ = 0.0;
  int kind_ = kmg::kKindLinear;   // block/kind = linear | psi | chi
  Real psi_floor_ = 0.05;         // block/psi_floor (kind = psi), as the flat driver
  kmg::CoefSource cs_;
  kmg::Manuf mf_;
  kmg::MatterManuf mm_;           // kinds psi/chi: the manufactured matter

  //! W4 production (cfc_kerr.cpp): Prepare() takes the finest-level matter from these mesh
  //! arrays (mesh ghost depth ngh, channel 0) instead of mm_: M (E* for psi, E* + 2S* for
  //! chi), DA2 = Ahat^2 - Ahat0^2 and, for chi, DPsi = delta psi. The views are kept, so
  //! they are read at every Solve.
  void SetMatter(const DvceArray5D<Real> &mt, const DvceArray5D<Real> &da2,
                 const DvceArray5D<Real> &dpsi, int ngh);
  //! the finest u (interior) from a mesh array of ghost depth ngh (cold start / regrid)
  void SeedInitialGuess(const DvceArray5D<Real> &u, int ngh);
  //! the finest u into a mesh array (mesh ghost depth; the inner min(ngh_, ng) ring too)
  void RetrieveSolution(DvceArray5D<Real> &dst);
  bool matter_ext_ = false;
  DvceArray5D<Real> ext_m_, ext_da2_, ext_dpsi_;
  int ext_ngh_ = 0;

  // octets (host): physics-dependent operations
  void SmoothOctet(MGOctet &oct, int rlev, int color) override;
  void CalculateDefectOctet(MGOctet &oct, int rlev) override;
  void CalculateFASRHSOctet(MGOctet &oct, int rlev) override;

  long nonfinite_oct_ = 0;

 private:
  //! all channels (linear) or the background channels only (psi/chi)
  void FillOctetCoeffs(bool bg_only);
  //! W3b: every channel of the coarsest block cell into the root grid or its parent octet
  //! (a copy of MGCFCConformalFactorDriver::TransferCoeffToRoot,
  //! mg_cfc_conformal_factor.cpp:708-777)
  void TransferCoeffToRoot();
  DualArray1D<int> blk_sync_, root_sync_;
  DualArray1D<int> blk_mask_, blk_phys_, root_mask_, root_phys_;
};

}  // namespace cfc

#endif  // CFC_MG_KERR_SCALAR_HPP_
