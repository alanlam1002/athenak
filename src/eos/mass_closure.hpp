#ifndef EOS_MASS_CLOSURE_HPP_
#define EOS_MASS_CLOSURE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mass_closure.hpp
//! \brief <mhd> mass_closure (default false; research R-114, user-approved): one itemized
//! rest-mass closure line MCLOSE per segment (run), M = sum u0(IDN) dV over interior cells.
//! Items (signed changes of the grid mass, each summed over the segment):
//!   1 outer boundary: -sum_s w_s beta_s dt F_out,s (the RK-weighted face flux)
//!   2 lapse excision: the C2P change in excision-floor cells with alpha < excise_lapse (or
//!     every excision-floor cell without the time-dependent excision), by r_BH bin
//!     (< 0.59, 0.59-0.78, 0.78-1.88, 1.88-2.92, >= 2.92); info: the flux into the mask
//!   3 absorbed at r_f: the same in cells excised only by r < r_f (includes its growth: the
//!     newly excised cells are reset by the next C2P)
//!   4 zone: removed (class 2, dm < 0) and added (dm > 0) by the C2P; added by the F1
//!     unmask between cycles (DynGRMHD::ZoneUnmaskPost); tier 3 is velocity-only
//!   5 floors: the stage C2P's floored cells (class 3); the regrid/init path's C2P
//!   6 regrid: prolongation/restriction (M_prol - M_old), 0 on a static mesh
//!   7 cold recovery: the C2P change in cold-recovered cells (0 by construction; a check)
//!   8 FOFC / flux update: Y + F_out, the RK update's non-conservation (FOFC replaces face
//!     fluxes in flux form, so it can only appear here; a check); and, separately, the
//!     RK coefficients' rounding (gam0 + gam1 = 1 - 1.1e-16 for rk3's 2/3 + 1/3): the
//!     update scales M(u1) by sum_s w_s gam1_s + w_0 gam0_0 != 1, a real ~5e-17 M per cycle
//!   9 restart: M at this segment's start minus M at the previous segment's end (0 with
//!     <job> restart_state; a check)
//!  10 the residual R = dM_grid - sum(1-9), dM_grid = M(end) - M(start of segment)
//! plus a check "other C2P" (class 4 cells that are not cold-recovered: 0 expected).
//! Precision: every cell sum (M itself and every item) is an exact fixed-point sum (Fix
//! below), on each rank and over ranks (an int64 MPI_Allreduce: exact, order-independent);
//! the cycle's RK combination is evaluated in double-double, and the cycles are summed
//! with Neumaier on the host. This file is compiled with -fp-model=precise and
//! -ffp-contract=off (src/CMakeLists.txt) for the host-side compensated arithmetic.
//! Pass: |R| <= mass_closure_tol M*.
//! <mhd> mass_closure_full (default false; R-115 Q5, user-approved): the same closure for
//! S_1, S_2, S_3 and tau (lines SCLOSE1/2/3, ECLOSE, each with _START/_RUN/_CYC), from a
//! copy of each before the C2P. Their item "source" (in place of "update") is Y + F_out,
//! the RK-weighted geometric source term plus the update's rounding; "crec" and "other"
//! (tau: the dual-energy resync) are physical changes there, so only regrid and restart
//! are flagged. Same exact sums, same tolerance (in M*).
//! <mhd> tracer_closure (default false; R-130/R-131 item 3, user-approved; needs mass_closure,
//! dual_energy and dual_energy_diag): the same closure for the dual-energy tracer DY (line
//! YCLOSE; diagnostic only, reads the state), with its item "other" (the C2P's change in the
//! ordinary cells) split per cell by the dual-energy branch of that C2P (de_diag slot 0, dbr:
//! 4 = flagged dense, energy solution + resync; 0 = non-dense resync; 1/2/3/5 = entropy or bad,
//! DY unchanged, a check), the flagged-dense resyncs into up and down and binned by the gas's
//! normal grid speed |V.n|_coord = |(alpha v^i - beta^i) n_i|, n = -grad P/|grad P| (coordinate
//! components, central differences of w0; as scripts/c117_shock_speed.py); every C2P item binned
//! by density band (rho/rho_c, <mhd> tracer_closure_rhoc); and every item accumulated per phase
//! (<mhd> tracer_closure_phase_t = t1,t2,t3: phases t < t1, [t1,t2), [t2,t3), >= t3). Printed
//! per segment (YCLOSE, YCLOSE_RUN) and cumulatively per phase (YCLOSE_PH_RUN, YCLOSE_BAND_RUN).

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "athena.hpp"
#include "dyn_grmhd/metric_faces.hpp"

class MeshBlockPack;
class ParameterInput;

namespace mclose {

//! a Neumaier-compensated sum (value s + c)
struct KSum {
  Real s = 0.0, c = 0.0;
  KOKKOS_INLINE_FUNCTION void Add(const Real x) {
    const Real t = s + x;
    if (Kokkos::fabs(s) >= Kokkos::fabs(x)) {
      c += (s - t) + x;
    } else {
      c += (x - t) + s;
    }
    s = t;
  }
  KOKKOS_INLINE_FUNCTION void Join(const KSum &o) { Add(o.s); Add(o.c); }
  Real Val() const { return s + c; }
};

//! an exact fixed-point accumulator: value = (l[2] 2^76 + l[1] 2^38 + l[0]) 2^-128. Each
//! double splits exactly into the limbs (power-of-two scalings and truncations), and integer
//! sums are exact and order-independent, so a cell sum (and the sum over ranks, an int64
//! MPI_Allreduce) is exact to 2^-128 ~ 3e-39 whatever the reduction tree or fp model.
//! Range: |x| <= ~1e-4 per term (2^38.7 in the top limb), <= 2^24 terms per rank before
//! the per-rank carry normalization (C-101: compensated double sums were only accurate to
//! about an ulp of M on the GPU).
struct Fix {
  int64_t l[3] = {0, 0, 0};
  KOKKOS_INLINE_FUNCTION void Add(const Real x) {
    const Real y = x*0x1p128;
    const Real q2 = Kokkos::trunc(y*0x1p-76);
    const Real r1 = y - q2*0x1p76;
    const Real q1 = Kokkos::trunc(r1*0x1p-38);
    const Real r0 = r1 - q1*0x1p38;
    l[2] += static_cast<int64_t>(q2);
    l[1] += static_cast<int64_t>(q1);
    l[0] += static_cast<int64_t>(Kokkos::trunc(r0));
  }
  KOKKOS_INLINE_FUNCTION void Join(const Fix &o) {
    l[0] += o.l[0]; l[1] += o.l[1]; l[2] += o.l[2];
  }
};

// per-stage cell sums of StageItems, in this order
enum StageIdx {
  kExc0 = 0, kExc1, kExc2, kExc3, kExc4,   // lapse excision by r_BH bin
  kRf,                                     // absorbed at r_f
  kZoneRem, kZoneAdd,                      // zone (class 2)
  kFloor,                                  // class 3
  kCrec,                                   // cold-recovered cells (class 4, bit 256)
  kOther,                                  // other class-4 cells
  kMend,                                   // M after the C2P
  kMd0,                                    // sum d0 dV (debug: must equal M_pre)
  kNStage
};
// tracer_closure: the per-cell categories of the C2P change, per density band
enum YCat {
  yExc = 0, yRf, yZoneRem, yZoneAdd, yFloor, yCrec,
  yNonDense,                    // dbr 0: non-dense resync
  yDenseChk,                    // dbr 1/2/3/5: entropy or bad (DY unchanged; a check)
  yUp0,                         // dbr 4, DY up, |V.n| bins (kYV of them)
  kYV = 7,
  yDn0 = yUp0 + kYV,            // dbr 4, DY down, |V.n| bins
  kYCat = yDn0 + kYV
};
constexpr int kYBand = 4;       // rho/rho_c >= 1e-1, [1e-2, 1e-1), [1e-3, 1e-2), < 1e-3
constexpr int kYPh = 4;         // phases (tracer_closure_phase_t: three edges)
constexpr int kYN = kYBand*kYCat;
constexpr int kYG = 16;         // reduced in groups of 16 sums
constexpr int kYNG = (kYN + kYG - 1)/kYG;

// face sums: F_out (outer faces) and F_exc (into the excision-flux mask)
enum FaceIdx { kFout = 0, kFexc, kNFace };

// the segment items (host accumulators)
enum Item {
  iOut = 0,
  iExc0, iExc1, iExc2, iExc3, iExc4,
  iRf,
  iZoneRem, iZoneAdd, iUnmask,
  iFloorStage, iFloorInit,
  iRegrid,
  iCrec,
  iUpdate,
  iRkCoef,   // the RK coefficients' rounding: gam0 + gam1 != 1 in double
  iRestart,
  iOther,
  iFexc,     // info, not part of the sum
  kNItem
};

class MassClosure {
 public:
  // var: the conserved variable (IDN: the mass closure, which owns the others)
  MassClosure(MeshBlockPack *pp, ParameterInput *pin, int var = IDN);
  bool on = false;
  // before a stage's C2P (u1 is the cycle-start register; read on stage 1)
  void StagePre(int stage);
  // after a stage's C2P; d0 = D before it, cls/c2p = the ledger's per-cell class/code
  void StagePost(int stage, int ns, const Real *gam0, const Real *gam1, const Real *beta,
                 Real dt, const DvceArray4D<Real> &d0, const DvceArray4D<Real> &cls,
                 const DvceArray4D<Real> &c2p, int cycle, Real time);
  // exact before/after mass around a change outside the stages (F1 unmask, regrid path)
  void MarkBefore();
  void MarkAfter(int item);
  // the regrid path: M_prol after prolongation (item 6 against the last M_end)
  void RegridProlongated();
  // after Driver::Initialize (M_start, the restart item) and at the end (the MCLOSE line)
  void SegmentStart(ParameterInput *pin, int cycle);
  void SegmentEnd(int cycle, Real time);
  // into the restart's parameter copy: M_end, the run's cumulative items, M*
  void Save(ParameterInput *pin) const;
  // R1 embedding at load (<job> restart_embed_fill, restart_state::EmbedFill): the grid
  // mass after the restart is staged (k = 0: the old domain only, the new blocks are
  // zero), after the atmosphere fill (1) and after the load's metric re-solve and C2P
  // (2). SegmentStart then reports them (TAG_EMBED: old domain - saved M_end, which must
  // be exactly 0; the atmosphere added; the load C2P's change) and books their sum as
  // regrid_embed instead of in the restart item.
  void EmbedMark(int k);
  // energy_variable = killing_full (dyn_grmhd_killing.cpp KillingFullStage): the T change from
  // the weights' change at fixed u of this stage (stage 1: since the last stage), booked as
  // KCLOSE's "weights" (not in the restart's parameter copy: its _RUN value is this process's)
  void KillWeights(int stage, Real g);
  // killing_weights = comoving: the stage's weight-motion source, beta_dt sum sqrt(g) s_W dV
  void KillMotion(int stage, Real g);

 private:
  MeshBlockPack *pmy_pack;
  int var_ = IDN;             // the conserved variable
  std::string tag_, key_;     // its output tag (MCLOSE, ...) and parameter-key prefix
  DvceArray4D<Real> x0_;      // S_i / tau: the variable before the C2P (IDN: ledger d0)
  std::vector<std::unique_ptr<MassClosure>> extra_;   // mass_closure_full: S_1..3, tau
                                                      // (+ tracer_closure: DY)
  // tracer_closure
  bool ytr_req_ = false;      // (IDN) create the DY closure at the first SegmentStart
  bool ytr_ = false;          // this is the DY closure (YCLOSE)
  Real y_rhoc = 3.24e-4;
  Real y_ph[kYPh-1] = {125.9, 145.9, 185.9};
  Real y_tlast = 0.0;         // the time of the last stage (phase of the booked items)
  DvceArray4D<int> y_cat_;    // per cell: band*kYCat + category, -1 = no change
  DvceArray4D<Real> y_val_;   // per cell: the change dV
  Fix y_st[4][kYN];
  KSum yseg[kYN], ycum[kYPh][kYN], phcum[kYPh][kNItem];
  int Phase(Real t) const;
  void YStage(int stage, const DvceArray4D<Real> &cls, const DvceArray4D<Real> &c2p);
  int every = 0;              // <mhd> mass_closure_every: per-cycle MCLOSE_CYC lines
  Real tol = 1.0e-10;         // <mhd> mass_closure_tol (in M*)
  Real mstar = 0.0;           // <mhd> mass_closure_mstar (0: the grid mass at run start)
  Fix st_pre[4], st_u1;       // per stage (this rank): M_pre, M(u1)
  Fix st_c[4][kNStage];       // per stage (this rank): StageItems
  Fix st_f[4][kNFace];        // per stage (this rank): face sums (x beta dt after the sum)
  Real st_bdt[4] = {0.0, 0.0, 0.0, 0.0};
  KSum seg[kNItem], cum[kNItem];
  KSum m_start, m_last_end, m_mark;
  bool have_last_end = false;
  Real m_prev_end = 0.0;      // the previous segment's M_end (restart), NaN if unknown
  int ncyc = 0, cycle0 = 0;
  KSum e_mark_[3];            // EmbedMark(0..2)
  bool embed_ = false;        // EmbedMark(2) done: SegmentStart books regrid_embed
  KSum embed_cum_;            // the run's cumulative regrid_embed (parameter copy)
  KSum Mass() const;          // global (rank-ordered) M of u0
  // <mhd> energy_variable = killing_t1 | killing_ta (R-155 item 1): the tau closure becomes
  // the closure of the evolved Killing energy T = alpha0 (tau + D) - beta0^k S_k - w D
  // (densitized; w = 1 or alpha0; line KCLOSE), whose face flux is the tau slot's flux there
  // (dyn_grmhd_killing.cpp). Its item "source" is then the RK-weighted residual-order source.
  int kill_ = 0;              // 0: off, 1: killing_t1, 2: killing_ta, 3: killing_full
  Real kw_st_[4] = {0.0, 0.0, 0.0, 0.0};   // killing_full: KillWeights per stage
  KSum kw_seg_, kw_cum_;
  bool kboost_ = false;       // <mhd> killing_weights = boost: (alpha0, beta0) = (1, xidot)
  // <mhd> killing_weights = comoving (R-175): the weights move with the hole (bg_xbh_), so the
  // stage gaps are booked as "weights" (as killing_full); the frozen weights' rate added to the
  // source, xidot^j (E d_j alpha0 - S_k d_j beta_BH^k), is its own item "motion" (out of
  // "source"). kcflat_: no puncture, weights (1, 0).
  bool kcomov_ = false, kcflat_ = false;
  Real km_st_[4] = {0.0, 0.0, 0.0, 0.0};   // KillMotion per stage (the stage's applied amount)
  KSum km_seg_, km_cum_;
  bool KWt() const { return kill_ == 3 || kcomov_; }
  bool kill_face_ = false;    // <mhd> killing_closure on a Valencia run: T's face flux formed
  mutable DvceArray5D<Real> kv_;   // T of the array passed to KSrc (one variable)
  dyngr::TrumpetTable ktab_;
  const DvceArray5D<Real> &KSrc(const DvceArray5D<Real> &u) const;   // u, or T(u) into kv_
  int KVar() const { return kill_ ? 0 : var_; }
};

}  // namespace mclose

#endif  // EOS_MASS_CLOSURE_HPP_
