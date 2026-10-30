#ifndef DYN_GRMHD_DYN_GRMHD_HPP_
#define DYN_GRMHD_DYN_GRMHD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyn_grmhd.hpp
//  \brief definitions for DynGRMHD class

#include "athena.hpp"
#include "parameter_input.hpp"
#include "eos/mass_closure.hpp"
#include "tasklist/task_list.hpp"
#include "driver/driver.hpp"
#include "eos/primitive_solver_hyd.hpp"
#include "dyn_grmhd/metric_faces.hpp"

// Riemann solvers for dynamical GR; hlle_frame_dyngr: <mhd> riemann_frame = boosted (R-168)
enum class DynGRMHD_RSolver {llf_dyngr, hlle_dyngr, hlle_frame_dyngr};
enum class DynGRMHD_EOS {eos_ideal, eos_piecewise_poly,
                      eos_compose, eos_hybrid};        // EOS policies for dynamical GR
enum class DynGRMHD_Error {reset_floor};               // Error policies for dynamical GR

//----------------------------------------------------------------------------------------
//! \struct DynGRMHDTaskIDs
//  \brief container to hold TaskIDs of all dyngr tasks

struct DynGRMHDTaskIDs {
  TaskID irecv;
  TaskID copyu;
  TaskID flux;
  TaskID settmunu;
  TaskID sendf;
  TaskID recvf;
  TaskID expl;
  TaskID restu;
  TaskID sendu;
  TaskID recvu;
  TaskID efld;
  TaskID sende;
  TaskID recve;
  TaskID ct;
  TaskID restb;
  TaskID sendb;
  TaskID recvb;
  TaskID bcs;
  TaskID c2p;
  TaskID newdt;
  TaskID clear;
  TaskID zrecv;
  TaskID zcopyu;
  TaskID zmattersrc;
  TaskID zcrhsdep;
  TaskID zcrhs;
  TaskID zsombc;
  TaskID zexpl;
  TaskID zsendu;
  TaskID zrecvu;
  TaskID znewdt;
  TaskID zbcs;
  TaskID zalgc;
  TaskID z4tad;
  TaskID zadmc;
  TaskID zclear;
  TaskID zrestu;
  TaskID zadep;
  TaskID c2pdep;
  TaskID rkdep;
};

namespace dyngr {

// <mhd> energy_variable (R-155 item 1): the energy equation evolved
// (killing_full, R-165 item 4: the full-metric comoving form, weights (alpha, beta') of the CFC metric)
enum class EnergyVariable {valencia, killing_t1, killing_ta, killing_full};

// <mhd> killing_source_diag (R-165 item 3(a)): r_iso bins (0: r < kKdR0, kKdBins-1: r >= kKdR1,
// geometric in between) x components (sums: dt dV x, cumulative; maxima: over the run)
constexpr int kKdBins = 42, kKdComp = 14, kKdNSum = 8;
constexpr Real kKdR0 = 0.25, kKdR1 = 40.0;
// sums: 0 eff (the effective T source), 1-5 dt dV sqrt(g) (s_K, s_alpha, s_E, s_beta, s_ta),
// 6 eff of the cells left on the Valencia update (alpha0 < 1e-3), 7 dt dV |delta alpha|;
// maxima: 8 |delta alpha|, 9 |delta alpha/alpha0|, 10 |delta beta|, 11 |psi^4 - psi0^4|,
// 12 |d delta alpha|, 13 dt dV (the cell-time weight, a sum)

class DynGRMHD {
 public:
  DynGRMHD(MeshBlockPack *ppack, ParameterInput *pin);
  virtual ~DynGRMHD();

  // container to hold names of TaskIDs
  DynGRMHDTaskIDs id;

  TaskStatus SetTmunu(Driver *d, int stage);
  TaskStatus SetADMVariables(Driver *d, int stage);
  TaskStatus UpdateExcisionMasks(Driver *d, int stage);
  TaskStatus ApplyPhysicalBCs(Driver *d, int stage);

  // <coord> excision_radius_dynamic (R-112 item 3; research TIME_DEP_EXCISION_DESIGN.md):
  // a time-dependent excision sphere r < exc_rf about the moving hole, OR'ed with the lapse
  // mask. exc_rf is one global value per cycle, updated between cycles (the final stage's
  // mask rebuild) by UpdateExcisionRadius:
  //   r_f = min(rmax, d_min - margin_cells*dx - vin*dt), grown by at most dx per cycle,
  //   shrunk at once; d_min = distance to the nearest interior cell with rho > rho_thr.
  // Persisted in the restart file (outputs/restart.cpp), so a restart continues bitwise.
  bool exc_dyn = false;
  Real exc_rmax = 2.5, exc_margin = 4.0, exc_vin = 0.6, exc_rho_thr = 0.0;
  Real exc_rf = 0.0;          // the current radius (0: lapse mask only)
  Real exc_abs_total = 0.0;   // rest mass absorbed at r_f (cells excised only by r < r_f)
  long exc_nshrink = 0, exc_ngrow = 0;
  void UpdateExcisionRadius(Driver *d);
  void SetExcisionGeometry();   // hands exc_rf and the hole position to Coordinates
  virtual Real DrainAbsorbedAtRf() { return 0.0; }

  // functions

  virtual void QueueDynGRMHDTasks() = 0;

  virtual TaskStatus ConToPrim(Driver* pdrive, int stage) = 0;
  virtual void ConToPrimBC(int is, int ie, int js, int je, int ks, int ke) = 0;
  virtual void UpdateShockFlags() {}   // dual_energy_flag_exchange (R-069)
  // the stage-0 mass ledger around the regrid / init path's C2P (R-078 item 3)
  virtual void MassLedgerRegridPre() {}
  virtual void MassLedgerRegridPost(const char *path) {}
  // <mhd> atmosphere_zone_unmask (R-078 item 4 F1), around the post-C2P mask rebuild
  virtual void ZoneUnmaskPre() {}
  virtual void ZoneUnmaskPost() {}
  virtual void PrimToConInit(int is, int ie, int js, int je, int ks, int ke) = 0;
  // restart_state (outputs/restart_state.hpp): per-cell state beyond u0, ghosts included
  virtual int RestartStateVars() { return 0; }
  // restart_state's group layout, "name:n,..." in SaveRestartState's order (R-140 item 4)
  virtual std::string RestartStateLayout() { return ""; }
  virtual void SaveRestartState(DvceArray5D<Real> &dst, int ns) {}
  virtual void LoadRestartState(const DvceArray5D<Real> &src, int ns) {}
  virtual void SaveRestartScalars(ParameterInput *pin) {}   // per-cycle scalars (tier 3)
  // <mhd> mass_closure (eos/mass_closure.hpp): the itemized MCLOSE ledger; nullptr if off
  mclose::MassClosure *pmclose = nullptr;
  // dual energy (item 70): {dense cells, entropy-branch cells, dE of the last full C2P,
  // cumulative dE}; false if dual energy is off.
  virtual bool GetDualEnergyStats(Real out[4]) { return false; }
  virtual void ConvertInternalEnergyToPressure(int is, int ie,
                                               int js, int je, int ks, int ke) = 0;

  virtual void AddCoordTerms(const DvceArray5D<Real> &w0, const DvceArray5D<Real> &bcc0,
                             const Real dt, DvceArray5D<Real> &u0, int nghost) = 0;

  // dyn_grmhd_newdt.cpp: replaces mhd::MHD::NewTimeStep (which hardcodes the speed of
  // light, max_dv=1, for any is_dynamical_relativistic_ run) with a timestep based on
  // the actual GR fast magnetosonic speed -- computed via this module's own
  // primitive-solver EOS (PrimitiveSolverHydro::GetGRFastMagnetosonicSpeeds, not the
  // ideal-gas-only EquationOfState used by plain mhd/hydro) and the current dynamical
  // ADM metric (padm->adm, pointwise -- not a fixed analytic background the way
  // PR #698's is_general_relativistic_/gr_dt path uses ComputeMetricAndInverse for a
  // static Kerr-Schild spacetime). Gated on gr_dt (below): false (default) preserves
  // the old, maximally conservative max_dv=1 behavior.
  virtual TaskStatus NewTimeStep(Driver *pdrive, int stage) = 0;

  // <dyn_grmhd> metric_terms / face_residual_interp (R-140 item 1; metric_faces.hpp).
  // Default cell: none of the members below is allocated or used.
  MetricTerms metric_terms = MetricTerms::cell;
  int face_residual_interp = 4;     // 4: (-1, 9, 9, -1)/16 of the residual; 2: the average
  TrumpetTable ttab;
  DvceArray5D<Real> fm_res;         // the residual (delta psi, delta(alpha psi), beta_res)
  DvceArray5D<Real> fm_x1, fm_x2, fm_x3;   // the face metric per direction (kFm*)
  int fm_probe_cycles = 0, fm_probe_c0 = -1;   // <dyn_grmhd> face_probe_* (ghost/owner)
  Real fm_probe_box[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  void ComputeFaceMetric(int stage);   // once per stage, at the start of CalcFluxes
  void FaceMetricProbe(int stage);
  void TrumpetTableTest(int npts);     // <dyn_grmhd> metric_table_test (host and device)

  // <mhd> energy_variable (R-155 item 1; dyn_grmhd_killing.cpp). Default valencia: none of
  // the members below is used, and no kernel of the default path is changed.
  EnergyVariable energy_variable = EnergyVariable::valencia;
  DvceArray5D<Real> kill_res;       // the residual (delta alpha, delta beta^i, psi^4 - psi0^4)
  bool kill_boost = false;          // <mhd> killing_weights = boost: (alpha0, beta0) = (1, xidot)
  // <mhd> killing_weights = comoving (R-175; KILLING_ENERGY_NOTE Sec. 5): the trumpet weights at
  // the comoving hole (CFC::bg_xbh_, refilled every stage), excluding xidot, or (1, 0) without a
  // puncture; delta beta = beta' - xidot - beta_BH; the frozen weights' rate
  // -xidot^j (E d_j alpha0 - S_k d_j beta_BH^k) added to the T source (KCLOSE item "motion")
  bool kill_comov = false, kill_flat = false;
  bool kill_have_xprev = false;     // KCLOSE: the weights' centre of the previous stage
  Real kill_xprev[3] = {0.0, 0.0, 0.0}, kill_m1u1 = 0.0;
  DvceArray5D<Real> kill_scr;       // per-cell stage values reduced after the source kernel
  void KillingT1Stage(Driver *pdriver, int stage);   // comoving: KCLOSE's weights item
  void KillingFaceFluxes();         // the tau slot's face flux -> the T face flux
  int kill_probe_cycles = 0, kill_probe_c0 = -1, kill_probe_stage = 0;   // ghost/owner probe
  void KillingFaceProbe();          // <dyn_grmhd> killing_probe_cycles, face_probe_box
  // <mhd> killing_source_diag (R-165 item 3(a)): the effective T source of every stage update
  // (killing_t1/ta: dt sqrt(g) s_T; valencia + killing_closure: the Valencia update's T
  // violation) and the residual's size, binned in r_iso, cumulative; KDIAG lines every
  // killing_source_diag_every cycles. Diagnostic only: the update is unchanged.
  bool kill_diag = false, kill_diag_ta = false;
  int kill_diag_every = 25, kill_diag_last = -1;
  DvceArray2D<Real> kdiag;
  void ReportKillingDiag();
  // energy_variable = killing_full (R-165 item 4): weights (alpha, beta') of the current CFC
  // metric, at faces the Riemann solver's face metric (adm::Face?Metric averages).
  // <mhd> killing_full_dtg: false = arm (i), d_t' g = 0; true = arm (ii), the source
  // -1/2 alpha sqrt(gamma) T^{mu nu} d_t' g_{mu nu} and the weights' rate, both from the
  // previous stage's metric (kf_prev: alpha, beta^i, psi^4 at every cell, ghosts included).
  // R-174 (arm (ii) extended): kf_prev also holds the previous stage's CFC residual (delta psi,
  // delta(alpha psi), beta_res: slots 5-9), the source differences the residual only, against
  // the previous residual on the CURRENT background (and xidot), and adds the background's rate
  // analytically (comoving: -xdot_BH . grad g0 from the trumpet table; xi-ddot in beta');
  // kf_prev and the stage bookkeeping are restart state (dyngr.kfprev, <mhd> kf_*).
  bool kf_dtg = false, kf_have_prev = false;
  DvceArray5D<Real> kf_prev;
  Real kf_tprev = 0.0, kf_tstage = 0.0;
  Real kf_m1u1 = 0.0;   // KCLOSE weights item: [M_s - M_1](u1), telescoped (KillingFullStage)
  int kf_key[3] = {-1, -1, -1};   // nmb_created, nmb_deleted, nmb_thispack of kf_prev
  bool kf_rst_loaded = false;     // kf_prev restored: keep it across the restart's new key
  bool kf_pre_rst = false;        // the parameters have no kf_* (fresh start or pre-arm-(ii) rst)
  bool kf_bres_stale = false;     // restarted without dyngr.kfbres: u_beta is the zero fill
  bool kf_bgmove = false;         // <cfc> puncture_moving: the background's analytic rate
  bool kf_have_xd = false;        // xi-ddot: xidot of the last step (frame controller)
  Real kf_xdstep[3] = {0.0, 0.0, 0.0}, kf_xdd[3] = {0.0, 0.0, 0.0};
  // <mhd> killing_full_split_every (R-174 (e)): KFSPLIT lines, each stage's s_T integrals
  // (background / residual part; rho >= killing_full_split_rho and all cells)
  int kf_split_every = 0;
  Real kf_split_rho = 0.0, kf_split_rmin = 0.0;   // the masked ("dense") set
  void KillingFullStage(Driver *pdriver, int stage);   // stage time + KCLOSE weights item
  void ReportKillingStage(int stage, Real dt);   // reduce kill_scr: KFSPLIT / KCLOSE motion
  void SaveKillingScalars(ParameterInput *pin);  // arm (ii)'s stage bookkeeping -> <mhd> kf_*
  void LoadKillingScalars(ParameterInput *pin);  // (the constructor, on a restart)
  bool KfRst() const { return energy_variable == EnergyVariable::killing_full && kf_dtg; }
  int kill_stage = 0;             // the RK stage of the current killing source call

  // <mhd> riemann_frame = boosted (R-168, "level R"; rsolvers/hlle_frame_dyn_grmhd.hpp,
  // dyn_grmhd_riemann_frame.cpp): HLLE in the frame of U at each face. rsolver_method is
  // then hlle_frame_dyngr; the default (eulerian) leaves every kernel unchanged.
  bool rf_on = false;               // riemann_frame = boosted
  bool rf_ugrid = false;            // riemann_frame_U = grid_rest (else eulerian, U = n)
  bool rf_r2 = false;               // riemann_frame_recon = R2 (else R1)
  bool rf_hllc = false;             // riemann_frame_solver = hllc (else hlle; FOFC: hlle)
  bool rf_bchecked = false;         // the B = 0 check done
  DvceArray1D<int> rf_err;          // (0): faces where grid_rest U is not timelike
  DvceArray5D<Real> rf_vsave;       // R2: w0's velocity during the sweeps
  void RFrameCellVelocities(int dir);   // R2: w0 velocity -> u^(i), cell reference tetrad
  void RFrameRestoreVelocities();       // R2: w0 velocity back (bitwise)
  void RFrameStageCheck(bool after);    // before: B = 0 (once), err reset; after: err

  // DynGRMHD policies
  DynGRMHD_RSolver rsolver_method;
  DynGRMHD_RSolver fofc_method;
  DynGRMHD_EOS eos_policy;
  DynGRMHD_Error error_policy;

  // Storage for temperature
  DvceArray5D<Real> temperature;
  // <mhd> dual_energy_diag: per-cell switch maps (PrimitiveSolverHydro::de_diag; output
  // variable mhd_de_diag). 1x1 unless on.
  DvceArray5D<Real> de_diag;

 protected:
  MeshBlockPack *pmy_pack;  // ptr to MeshBlockPack containing this Hydro
  int scratch_level;        // GPU scratch level for flux and source calculations
  bool enforce_maximum;     // enforce local maximum principle during FOFC
  Real dmp_M;               // threshold multiplier for discrete maximum principle.
  bool fixed_evolution;     // Disable mhd evolution
  bool scalar_pplimiter;    // Apply positivity preserving limiter on scalar
  // <mhd>/fofc_scalar_consistent (default false; ported from proj/g-mode 3c39565, R-087
  // item 2): on faces whose mass flux FOFC replaces with the first-order flux (hydro-flagged
  // or excised cell), take each scalar flux as the same first-order Riemann flux of D*Y
  // (same wave speeds and branch as D) instead of the first-order mass flux times the
  // upwind cell Y. Scalar-only faces keep the upwind form, which is the one consistent with
  // their unchanged high-order mass flux.
  bool fofc_scalar_consistent;
  // <time>/gr_dt (default false): opt-in to the real GR fast-magnetosonic-speed
  // timestep in dyn_grmhd_newdt.cpp instead of the conservative max_dv=1 (speed of
  // light) fallback -- same input key as PR #698's analogous flag on Hydro/MHD.
  bool gr_dt;
  // <time>/dt_skip_excised (default false; DEVELOPMENT.md item 73, research H): leave
  // cells inside the excision mask (excision_flux) out of the CFL reduction. Their
  // primitives are reset to the excision floor every step and their face fluxes are
  // first order, so their own (gauge) shift need not set dt.
  bool dt_skip_excised;
  bool dt_floor_keep_velocity = false;   // <time> dt_floor_keep_velocity (R-076 item 3)
  int dt_lim_report = 0, dt_lim_cyc0 = -1;   // <time> dt_limiter_report (debug)
  int dt_lim_every = 0;                      // <time> dt_limiter_report_every (R-115)
  DvceArray4D<Real> dt_lim_cell;
};

template<class EOSPolicy, class ErrorPolicy>
class DynGRMHDPS : public DynGRMHD {
 public:
  DynGRMHDPS(MeshBlockPack *ppack, ParameterInput *pin) :
      DynGRMHD(ppack, pin), eos("mhd", ppack, pin) {
    if (pmclose != nullptr) eos.ml_track = true;   // MCLOSE reads the ledger's cell data
    if (de_diag.extent_int(1) > 1) {
      if (!eos.de_on) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<mhd> dual_energy_diag needs dual_energy = true"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      eos.de_diag = de_diag;
      eos.de_diag_on = true;
    }
  }
  virtual ~DynGRMHDPS() {}

  // Dynamical EOS
  PrimitiveSolverHydro<EOSPolicy, ErrorPolicy> eos;

  // CalculateFluxes function templated over Riemann Solvers; FM: the face metric of
  // <dyn_grmhd> metric_terms != cell (FM = false is the present scheme)
  template<DynGRMHD_RSolver T, bool FM = false>
  TaskStatus CalcFluxes(Driver *d, int stage);

  template<DynGRMHD_RSolver T, bool FM = false>
  void FOFC(Driver *d, int stage);

  // functions
  virtual void QueueDynGRMHDTasks();

  virtual TaskStatus ConToPrim(Driver* pdrive, int stage);
  virtual void ConToPrimBC(int is, int ie, int js, int je, int ks, int ke);
  void RefreshExcisionMasks();   // <coord> excision_mask_refresh
  void UpdateShockFlags() override;
  void MassLedgerRegridPre() override { eos.MassLedgerRegridPre(); }
  void MassLedgerRegridPost(const char *path) override { eos.MassLedgerRegridPost(path); }
  void ZoneUnmaskPre() override { eos.ZoneUnmaskPre(); }
  void ZoneUnmaskPost() override;
  Real DrainAbsorbedAtRf() override {
    const Real a = eos.exc_absorbed_rf;
    eos.exc_absorbed_rf = 0.0;
    return a;
  }
  virtual void PrimToConInit(int is, int ie, int js, int je, int ks, int ke);
  int RestartStateVars() override;
  std::string RestartStateLayout() override;
  void SaveRestartState(DvceArray5D<Real> &dst, int ns) override;
  void LoadRestartState(const DvceArray5D<Real> &src, int ns) override;
  void SaveRestartScalars(ParameterInput *pin) override {
    eos.SaveTier3State(pin);
    eos.SaveDualEnergyTotal(pin);
    SaveKillingScalars(pin);
  }
  virtual bool GetDualEnergyStats(Real out[4]) {
    if (!eos.de_on) return false;
    out[0] = eos.de_ndense; out[1] = eos.de_nent; out[2] = eos.de_dE;
    out[3] = eos.de_dE_total;
    return true;
  }
  virtual void ConvertInternalEnergyToPressure(int is, int ie,
                                               int js, int je, int ks, int ke);

  virtual void AddCoordTerms(const DvceArray5D<Real> &w0, const DvceArray5D<Real> &bcc0,
                             const Real dt, DvceArray5D<Real> &u0, int nghost);

  // dyn_grmhd_newdt.cpp
  virtual TaskStatus NewTimeStep(Driver *pdrive, int stage);
  // riemann_frame = boosted: the CFL from the frame speeds (dyn_grmhd_riemann_frame.cpp)
  TaskStatus NewTimeStepFrame(Driver *pdrive, int stage);

  template<int NGHOST>
  void AddCoordTermsEOS(const DvceArray5D<Real> &w0, const DvceArray5D<Real> &bcc0,
                        const Real dt, DvceArray5D<Real> &u0);
  // metric_terms = analytic_bg | wellbalanced (dyn_grmhd_metric.cpp)
  template<int NGHOST>
  void AddCoordTermsFM(const DvceArray5D<Real> &w0, const DvceArray5D<Real> &bcc0,
                       const Real dt, DvceArray5D<Real> &u0);
  // energy_variable = killing_t1 | killing_ta (dyn_grmhd_killing.cpp): after
  // AddCoordTermsEOS, the tau slot's update replaced by the one implied by the T update
  template<int NGHOST>
  void AddKillingTerms(const DvceArray5D<Real> &w0, const DvceArray5D<Real> &bcc0,
                       const Real dt, DvceArray5D<Real> &u0);
  // energy_variable = killing_full (R-165 item 4; dyn_grmhd_killing.cpp)
  template<int NGHOST>
  void AddKillingFullTerms(const DvceArray5D<Real> &w0, const DvceArray5D<Real> &bcc0,
                           const Real dt, DvceArray5D<Real> &u0);
};

// Factory function for generating DynGRMHD based on parameter input.
// Used to make the MeshBlockPack creation a little bit cleaner.
DynGRMHD* BuildDynGRMHD(MeshBlockPack *ppack, ParameterInput *pin);

} // namespace dyngr

#endif  // DYN_GRMHD_DYN_GRMHD_HPP_
