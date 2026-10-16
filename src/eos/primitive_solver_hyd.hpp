#ifndef EOS_PRIMITIVE_SOLVER_HYD_HPP_
#define EOS_PRIMITIVE_SOLVER_HYD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file primitive_solver_hyd.hpp
//  \brief Contains the template class for PrimitiveSolverHydro, which is independent
//  of the EquationOfState class used elsewhere in AthenaK.

// C headers
#include <float.h>
#include <math.h>

// C++ headers
#include <string>
#include <type_traits>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>

// PrimitiveSolver headers
#include "eos/primitive-solver/eos.hpp"
#include "eos/primitive-solver/primitive_solver.hpp"
#include "eos/primitive-solver/idealgas.hpp"
#include "eos/primitive-solver/piecewise_polytrope.hpp"
#include "eos/primitive-solver/eos_compose.hpp"
#include "eos/primitive-solver/eos_hybrid.hpp"
#include "eos/primitive-solver/reset_floor.hpp"
#include "eos/primitive-solver/logs.hpp"

// AthenaK headers
#include "athena.hpp"
#include "globals.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "coordinates/adm.hpp"
#include "mhd/mhd.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"

template<class EOSPolicy, class ErrorPolicy>
class PrimitiveSolverHydro {
 protected:
  void SetPolicyParams(std::string block, ParameterInput *pin) {
    // Parameters for an ideal gas
    if constexpr(std::is_same_v<Primitive::IdealGas, EOSPolicy>) {
      ps.GetEOSMutable().SetGamma(pin->GetOrAddReal(block, "gamma", 5.0/3.0));
      ps.GetEOSMutable().SetNSpecies(pin->GetOrAddInteger(block, "nscalars", 0));
    }
    // Parameters for a piecewise polytrope
    if constexpr(std::is_same_v<Primitive::PiecewisePolytrope, EOSPolicy>) {
      bool result = ps.GetEOSMutable().ReadParametersFromInput(block, pin);
      if (!result) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "There was an error while constructing the EOS."
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
    // Parameters for CompOSE EoS
    if constexpr (
         std::is_same_v<Primitive::EOSCompOSE<Primitive::NormalLogs>, EOSPolicy> ||
         std::is_same_v<Primitive::EOSCompOSE<Primitive::NQTLogs>, EOSPolicy>) {
      // Get and set number of scalars in table. This will currently fail if not 1.
      ps.GetEOSMutable().SetNSpecies(pin->GetOrAddInteger(block, "nscalars", 1));
      std::string units = pin->GetOrAddString(block, "units", "geometric_solar");
      if (!units.compare("geometric_solar")) {
        ps.GetEOSMutable().SetCodeUnitSystem(Primitive::MakeGeometricSolar());
      } else if (!units.compare("geometric_kilometer")) {
        ps.GetEOSMutable().SetCodeUnitSystem(Primitive::MakeGeometricKilometer());
      } else if (!units.compare("nuclear")) {
        ps.GetEOSMutable().SetCodeUnitSystem(Primitive::MakeNuclear());
      } else if (!units.compare("cgs")) {
        ps.GetEOSMutable().SetCodeUnitSystem(Primitive::MakeCGS());
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Unknown unit system " << units << " requested."
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }

      // Get table filename, then read the table,
      std::string fname = pin->GetString(block, "table");
      ps.GetEOSMutable().ReadTableFromFile(fname);

      // Ensure table was read properly
      assert(ps.GetEOSMutable().IsInitialized());
    }
        // Parameters for Hybrid EoS
    if constexpr (
         std::is_same_v<Primitive::EOSHybrid<Primitive::NormalLogs>, EOSPolicy> ||
         std::is_same_v<Primitive::EOSHybrid<Primitive::NQTLogs>, EOSPolicy>) {
      // Get and set number of scalars in table. This will currently fail if not 0.
      ps.GetEOSMutable().SetThermalGamma(pin->GetOrAddReal(block, "gamma_thermal",
                                         5.0/3.0));
      ps.GetEOSMutable().SetNSpecies(pin->GetOrAddInteger(block, "nscalars", 0));
      std::string units = pin->GetOrAddString(block, "units", "geometric_solar");
      if (!units.compare("geometric_solar")) {
        ps.GetEOSMutable().SetCodeUnitSystem(Primitive::MakeGeometricSolar());
      } else if (!units.compare("geometric_kilometer")) {
        ps.GetEOSMutable().SetCodeUnitSystem(Primitive::MakeGeometricKilometer());
      } else if (!units.compare("nuclear")) {
        ps.GetEOSMutable().SetCodeUnitSystem(Primitive::MakeNuclear());
      } else if (!units.compare("cgs")) {
        ps.GetEOSMutable().SetCodeUnitSystem(Primitive::MakeCGS());
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Unknown unit system " << units << " requested."
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }

      // Get table filename, then read the table,
      std::string fname = pin->GetString(block, "table");
      ps.GetEOSMutable().ReadTableFromFile(fname);

      // Ensure table was read properly
      assert(ps.GetEOSMutable().IsInitialized());
    }
  }

 public:
  Primitive::PrimitiveSolver<EOSPolicy, ErrorPolicy> ps;
  MeshBlockPack* pmy_pack;
  unsigned int nerrs;
  unsigned int errcap;

  // ---- Dual energy (src/cfc/DEVELOPMENT.md item 70; code/ENERGY_FIX_DESIGN.md) ----
  // <mhd> dual_energy = true evolves the adiabat K = P/rho^gamma as the LAST passive
  // scalar, stored as Y = (K/de_kmax)^de_ypow so it stays inside the ideal-gas species
  // limits [0,1] (conserved: sqrt(gamma) D Y; same transport as Ye, no source term).
  // <mhd> dual_energy_variable = k_1g (default; de_ypow = 1/gamma, research R-029 §3):
  // D Y = W P^(1/gamma)/kmax^(1/gamma) is uniform across a pressure-equilibrium contact,
  // so mixing it keeps P; = k (de_ypow = 1) is the original D K. With de_recon_rhoy the
  // flux kernel reconstructs rho Y (prop. to P^(1/gamma)) and forms Y = (rho Y)/rho at
  // faces; with de_flux_hll the tracer flux is the HLLE flux of D Y itself instead of
  // (HLLE mass flux) x Y_upwind, whose mass diffusion breaks the contact property where
  // the HLL fan is open (subsonic contacts; DEVELOPMENT.md item 71). In dense
  // (D > de_rho_switch), unshocked gas the pressure comes from K via EntropyInversion
  // and tau is resynced; elsewhere K is resynced from the energy solution. A
  // primitive-floor hit in dense gas (the silent T < T_atm path of RESULTS §7.1) also
  // takes the entropy branch. Ideal gas, B = 0 cells only.
  bool de_on = false;
  int de_idx = -1;                 // scalar index of the K tracer (nscalars - 1)
  Real de_kmax = 1.0, de_rho_sw = 0.0, de_dp_shock = 0.3, de_eta1 = 0.0, de_gamma = 5./3.;
  Real de_ypow = 1.0;              // Y = (K/kmax)^de_ypow
  // shock sensor (item 72; default vcurv + one hysteresis dilation pass): 0 = jump
  // (|dP|/P > de_dp_shock, item 70), 1 = jameson
  // (psi_P > de_psi_shock), 2 = vcurv (psi_v > de_vcurv_shock, or psi_P > de_psi_jump for
  // discontinuities between states at rest); all of them also need div v <= 0.
  //   psi_P = max_a |P+ - 2P + P-|/(P+ + 2P + P-),  psi_v = max_a |v_a+ - 2v_a + v_a-|/c_s
  int de_sensor = 2;
  // <mhd> dual_energy_div (R-060; default coord): the div v <= 0 test of the shock flag
  // uses the coordinate divergence d_a v^a (coord) or the slice-covariant
  // D_a v^a = d_a v^a + v^a d_a ln sqrt(gamma) (covariant; = d_a v^a + 6 v^a d_a ln psi
  // for a conformally flat metric), which removes the d ln psi infall bias.
  bool de_div_cov = false;
  Real de_psi_shock = 0.25, de_vcurv_shock = 0.05, de_psi_jump = 1.0e30;
  bool de_recon_rhoy = false, de_flux_hll = false;
  DvceArray4D<int> de_shock;       // per-cell shock flag from the previous primitives
  DvceArray4D<int> de_shock2;      // scratch for the dilation (de_dilate)
  int de_dilate = 1;               // also flag the face neighbours of flagged cells
  Real de_dilate_weak = 0.5;       // ... if their own sensor exceeds this x the threshold
  // flag filters against near-threshold flicker (R-043 / C-040 P2; both off by default).
  // In a comoving star the vcurv sensor sits at 0.3-1x threshold over much of the star, so
  // isolated cells flag at random; each takes the energy solution, whose K runs hot there,
  // and the resync ratchets the tracer up (the second-law guard blocks the way back).
  //   de_isolated = true: before the dilation, a flagged cell with no flagged face
  //     neighbour is demoted to a weak candidate (a resolved shock is >= 2-3 cells wide).
  //   de_persist = N > 1: a cell takes the energy branch only once its final flag has
  //     been set in N consecutive C2P calls (de_streak counts them; reset on realloc).
  bool de_isolated = false;
  int de_persist = 1;
  DvceArray4D<int> de_streak;
  // second-law guard (item 73, default on): a dense flagged cell whose energy solution has
  // K < K_tracer takes the entropy solution instead -- a shock can only raise K, so a drop
  // there is the energy equation's numerical loss (V4F's tidal caps, RESULTS §11.5)
  bool de_second_law = true;
  Real de_second_law_tol = -1.0e-3; // entropy solution if K_en <= K_tr (1 + tol): ignore drops < 1e-3
  // Diagnostics of the last full (non-floors_only) C2P call, interior cells only:
  // dense cells, cells on the entropy branch, and the energy the tau resync added
  // (sum of sqrt(gamma) dtau dV); de_dE_total accumulates the latter over all calls.
  Real de_ndense = 0.0, de_nent = 0.0, de_dE = 0.0, de_dE_total = 0.0;
  // <mhd> dual_energy_diag (default false; research R-041 item 1a): per-cell maps of the
  // switch, set by DynGRMHDPS when on (a view of DynGRMHD::de_diag), from the last C2P
  // call (and the last coord_src call for 6-7):
  //   0 branch: 0 not dense, 1 entropy (unflagged), 2 entropy (second-law guard),
  //     3 entropy (floor/C2P failure), 4 energy + K resync, 5 energy, no resync
  //   1 shock sensor / its threshold (div v ignored)   2 final shock flag (0/1)
  //   3 K_energy/K_tracer - 1 (dense B = 0 cells with a good energy solution)
  //   4 K resync on the energy branch, K_new/K_tracer - 1   5 entropy-branch dtau/D
  //   6 alpha K_ij S^ij   7 -S^a d_a alpha   (the tau source terms, undensitized)
  //   8, 9 internal-energy budget of the last RK stage (C-041), as fractional rates
  //     d(W rho eps)/dt / (W rho eps) carried by the flux divergence (8) and by the source
  //     terms (9), from the exact combination W (dE - v^a dS_a) - dD, E = tau + D
  //   10-12 K_energy/K_tracer - 1 (as 3) at C2P stage 1, 2, 3 (R-045 per-stage check;
  //     de_stage is set by DynGRMHDPS::ConToPrim before each call)
  static constexpr int kNDEDiag = 13;
  bool de_diag_on = false;
  int de_stage = 0;
  DvceArray5D<Real> de_diag;
  // <mhd> atmosphere_frame_velocity (default false; research R-028 E1, a test): give
  // floored cells the comoving frame's velocity (Eulerian v = xidot = padm->frame_vel_u)
  // instead of v = 0, so the atmosphere is at rest on a comoving grid.
  bool atm_frame_vel = false;
  // <mhd> atmosphere_frame_gate (default 0 = everywhere): only floored cells farther than
  // this from the puncture (padm->bc_center, x_BH(t) for a moving puncture) get the
  // comoving velocity; near the BH, where infall dominates, the floor stays at v = 0
  // (COMOVING_GAUGE_DESIGN.md §2.4, Phase 2 M2)
  Real atm_frame_gate = 0.0;
  // <mhd> atmosphere_zone (default false; research R-058 §2 / R-059 S1, the single-tier
  // near-BH zone): every C2P call, non-excised cells within atmosphere_zone_radius (r_s)
  // of the puncture (padm->bc_center) with rho < atmosphere_zone_rho (rho_thr) get the
  // grid-rest Eulerian velocity v = beta'/alpha (beta' = adm.beta_u, which carries the
  // comoving frame's xidot), capped at |v| <= atmosphere_zone_vmax along beta', a cold
  // pressure (T = tfloor) and rho -> min(rho, atmosphere_zone_rho_reset) (default
  // 1e-3 rho_thr). The cell is then at rest on the comoving grid, so the infalling floor
  // gas near a moving hole no longer sets the CFL step (C-046 S0). With the zone on the
  // atmosphere_frame_gate is set to r_s (the zone replaces the v = 0 floor inside it).
  // Booking (zone_acc, drained per RK stage by DynGRMHDPS::ConToPrim with the stage
  // weights of CFC::AccreteExcisedMass): the removed rest mass, binned in
  //   r_BH: < 1.88 / 1.88-2.92 / >= 2.92 M (areal 3 M / 4 M on the trumpet slice),
  //   E/dE (BH-frame Killing E = -u_t - 1 of the state before the reset, dE =
  //   atmosphere_zone_de): edges -6 -3 -2 -1 0 1 2 3 6,
  //   |L| = |(x - x_BH) x u_i|: edges 2 4 6 10 M,
  // plus the removed momentum S_i and energy tau + D (densitized x cell volume), the
  // modified-cell count, the v-capped count, max rho of the modified cells before the
  // reset, max rho of all non-excised cells within r_s, and the count of modified cells
  // with rho >= rho_thr (must stay 0). Diagnostic only: the mass is NOT handed to the
  // puncture (it is ~1e-9 M* over a run, C-047).
  bool zone_on = false;
  Real zone_r = 0.0, zone_rho = 0.0, zone_rho_reset = 0.0, zone_vmax = 0.9;
  Real zone_de = 0.677/(16.4*16.4);
  static constexpr int kZoneNE = 10, kZoneNL = 5, kZoneNR = 3;
  static constexpr int kZoneNB = kZoneNE*kZoneNL*kZoneNR;     // 150 mass bins
  // after the bins: dSx dSy dSz d(tau+D) dM  n_mod n_cap  max_rho_mod max_rho_region n_viol
  static constexpr int kZoneSx = kZoneNB, kZoneEn = kZoneNB + 3, kZoneM = kZoneNB + 4;
  static constexpr int kZoneN = kZoneNB + 5, kZoneNcap = kZoneNB + 6;
  static constexpr int kZoneRmod = kZoneNB + 7, kZoneRreg = kZoneNB + 8;
  static constexpr int kZoneViol = kZoneNB + 9;
  // rest mass removed by the excision reset (interior cells; zone block only), so that
  // grid + excision + zone closes in rest mass (the CFC excision tally books E/psi)
  static constexpr int kZoneExcM = kZoneNB + 10, kZoneLen = kZoneNB + 11;
  // <mhd> atmosphere_zone_book (default false): the booking (ATMZONE lines, incl. the
  // excised rest mass) without the zone itself, for zone-off controls
  bool zone_book = false;
  DvceArray1D<Real> zone_acc;
  // <mhd> atmosphere_zone_tier3 (default false; R-058 §3, R-063 §3): velocity-only grid
  // rest (as the zone, but no rho reset) for r_s <= r < 2 r_s and rho < atmosphere_zone_
  // tier3_rho, excluding cells within 3 R_90 of the dense-gas centroid, and only while the
  // dense gas (rho > 1e-3 rho_max; mass-weighted centroid x_c, R_90 = its 90%-mass radius
  // about x_c) is farther than 2 r_s + 3 R_90 from x_BH AND receding, d.ddot > 0 with
  // d = x_c - x_BH and ddot = <alpha v - beta'> + xidot (x_BH = puncture_x - xi). The state
  // is measured once per cycle at the end of the last stage (UpdateTier3) and used during
  // the next cycle. Its tally is a second block of zone_acc with the same layout (the bins
  // hold the modified mass, nothing is removed); printed as ATMZONE_T3 / ATMZONE_T3_BINS.
  bool t3_on = false, t3_active = false;
  Real t3_rho = 0.0, t3_c[3] = {0.0, 0.0, 0.0}, t3_r90 = 0.0, t3_d = 0.0, t3_ddot = 0.0;
  static constexpr int kT3NR = 2048;            // R_90 histogram bins of t3_dr
  Real t3_dr = 1.0/32.0;
  DvceArray1D<Real> t3_hist;
  Real zone_stage[4][2*kZoneLen];   // per-RK-stage drains (host); [zone | tier 3]
  Real zone_cum[2*kZoneLen] = {};   // cumulative since (re)start, all ranks
  int zone_bins_every = 50;         // <mhd> atmosphere_zone_bins_every: cycles per bins line

  //! Drain this C2P call's zone tallies into the stage slot; on the last stage combine the
  //! stages with their propagation weights w_s = prod_{j>s} gam0_j (as
  //! CFC::AccreteExcisedMass), reduce over ranks and print one ATMZONE line (rank 0) and,
  //! every zone_bins_every cycles, the cumulative E/L/r bins (ATMZONE_BINS); likewise
  //! ATMZONE_T3 / ATMZONE_T3_BINS for tier 3 when it is enabled.
  void DrainZone(int stage, int ns, const Real *gam0, Real time, int cycle) {
    if (!zone_book) return;
    if (stage < 1 || stage > ns || ns > 4) {        // not an RK stage: discard
      Kokkos::deep_copy(zone_acc, 0.0);
      return;
    }
    auto h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), zone_acc);
    for (int n = 0; n < 2*kZoneLen; ++n) zone_stage[stage-1][n] = h(n);
    Kokkos::deep_copy(zone_acc, 0.0);
    if (stage != ns) return;
    Real w[4];
    w[ns-1] = 1.0;
    for (int q = ns-2; q >= 0; --q) w[q] = gam0[q+1]*w[q+1];
    Real sum[2*kZoneLen], mx[4] = {0.0, 0.0, 0.0, 0.0};
    for (int n = 0; n < 2*kZoneLen; ++n) sum[n] = 0.0;
    for (int bl = 0; bl < 2; ++bl) {
      const int o = bl*kZoneLen;
      for (int q = 0; q < ns; ++q) {
        for (int n = 0; n <= kZoneM; ++n) sum[o + n] += w[q]*zone_stage[q][o + n];
        sum[o + kZoneExcM] += w[q]*zone_stage[q][o + kZoneExcM];
        sum[o + kZoneViol] += zone_stage[q][o + kZoneViol];
        mx[2*bl] = fmax(mx[2*bl], zone_stage[q][o + kZoneRmod]);
        mx[2*bl + 1] = fmax(mx[2*bl + 1], zone_stage[q][o + kZoneRreg]);
      }
      sum[o + kZoneN] = zone_stage[ns-1][o + kZoneN];   // cells modified in the last stage
      sum[o + kZoneNcap] = zone_stage[ns-1][o + kZoneNcap];
    }
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, sum, 2*kZoneLen, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, mx, 4, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
    for (int bl = 0; bl < 2; ++bl) {
      const int o = bl*kZoneLen;
      for (int n = 0; n < kZoneLen; ++n) {
        if (n != kZoneN && n != kZoneNcap && n != kZoneRmod && n != kZoneRreg) {
          zone_cum[o + n] += sum[o + n];
        }
      }
    }
    if (global_variable::my_rank == 0) {
      const bool bins = zone_bins_every > 0 && cycle % zone_bins_every == 0;
      std::cout << std::setprecision(6) << "ATMZONE cycle=" << cycle << " time=" << time
                << " dM=" << sum[kZoneM] << " n_mod=" << sum[kZoneN] << " n_cap="
                << sum[kZoneNcap] << " max_rho_mod=" << mx[0] << " max_rho_zone=" << mx[1]
                << " n_viol=" << sum[kZoneViol] << " cum_dM=" << zone_cum[kZoneM]
                << " cum_dS=(" << zone_cum[kZoneSx] << "," << zone_cum[kZoneSx+1] << ","
                << zone_cum[kZoneSx+2] << ") cum_dE=" << zone_cum[kZoneEn]
                << " dM_exc=" << sum[kZoneExcM] << " cum_dM_exc=" << zone_cum[kZoneExcM]
                << std::endl;
      if (bins) {
        std::cout << "ATMZONE_BINS cycle=" << cycle << " time=" << time
                  << " [r][E][L] cumulative dM:";
        for (int n = 0; n < kZoneNB; ++n) std::cout << " " << zone_cum[n];
        std::cout << std::endl;
      }
      if (t3_on) {
        const int o = kZoneLen;
        std::cout << "ATMZONE_T3 cycle=" << cycle << " time=" << time << " active="
                  << t3_active << " d=" << t3_d << " R90=" << t3_r90 << " d.ddot="
                  << t3_ddot << " M_mod=" << sum[o + kZoneM] << " n_mod=" << sum[o + kZoneN]
                  << " n_cap=" << sum[o + kZoneNcap] << " max_rho_mod=" << mx[2]
                  << " n_viol=" << sum[o + kZoneViol] << " cum_dS=(" << zone_cum[o + kZoneSx]
                  << "," << zone_cum[o + kZoneSx+1] << "," << zone_cum[o + kZoneSx+2]
                  << ") cum_dE=" << zone_cum[o + kZoneEn] << std::endl;
        if (bins) {
          std::cout << "ATMZONE_T3_BINS cycle=" << cycle << " time=" << time
                    << " [r][E][L] cumulative modified mass:";
          for (int n = 0; n < kZoneNB; ++n) std::cout << " " << zone_cum[o + n];
          std::cout << std::endl;
        }
      }
    }
    UpdateTier3();
  }

  //! Tier-3 activation state from the current primitives (end of a cycle): rho_max, the
  //! dense-gas centroid and coordinate velocity, R_90 (histogram of t3_dr bins about x_c).
  void UpdateTier3() {
    if (!t3_on) return;
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    const int is = indcs.is, js = indcs.js, ks = indcs.ks;
    const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
    const int nmb = pmy_pack->nmb_thispack;
    const int nkji = nx3*nx2*nx1, nji = nx2*nx1;
    auto &size = pmy_pack->pmb->mb_size;
    auto &w0 = pmy_pack->pmhd->w0;
    auto &u0 = pmy_pack->pmhd->u0;
    auto &adm = pmy_pack->padm->adm;
    Real rmax = 0.0;
    Kokkos::parallel_reduce("zone_t3_rhomax", Kokkos::RangePolicy<>(DevExeSpace(), 0,
                            nmb*nkji),
    KOKKOS_LAMBDA(const int idx, Real &mxr) {
      const int m = idx/nkji, k = (idx - m*nkji)/nji + ks;
      const int j = (idx - m*nkji - (k - ks)*nji)/nx1 + js;
      const int i = idx - m*nkji - (k - ks)*nji - (j - js)*nx1 + is;
      mxr = fmax(mxr, w0(m, IDN, k, j, i));
    }, Kokkos::Max<Real>(rmax));
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &rmax, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
    const Real thr = 1.0e-3*rmax;
    Real s[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    Kokkos::parallel_reduce("zone_t3_centroid", Kokkos::RangePolicy<>(DevExeSpace(), 0,
                            nmb*nkji),
    KOKKOS_LAMBDA(const int idx, Real &a0, Real &a1, Real &a2, Real &a3, Real &a4,
                  Real &a5, Real &a6) {
      const int m = idx/nkji, k = (idx - m*nkji)/nji + ks;
      const int j = (idx - m*nkji - (k - ks)*nji)/nx1 + js;
      const int i = idx - m*nkji - (k - ks)*nji - (j - js)*nx1 + is;
      if (!(w0(m, IDN, k, j, i) > thr)) return;
      const Real x = CellCenterX(i - is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      const Real y = CellCenterX(j - js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      const Real z = CellCenterX(k - ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
      const Real ux = w0(m, IVX, k, j, i), uy = w0(m, IVY, k, j, i), uz = w0(m, IVZ, k, j, i);
      const Real gxx = adm.g_dd(m,0,0,k,j,i), gxy = adm.g_dd(m,0,1,k,j,i);
      const Real gxz = adm.g_dd(m,0,2,k,j,i), gyy = adm.g_dd(m,1,1,k,j,i);
      const Real gyz = adm.g_dd(m,1,2,k,j,i), gzz = adm.g_dd(m,2,2,k,j,i);
      const Real W = sqrt(1.0 + ux*(gxx*ux + gxy*uy + gxz*uz) + uy*(gxy*ux + gyy*uy + gyz*uz)
                          + uz*(gxz*ux + gyz*uy + gzz*uz));
      const Real aw = adm.alpha(m,k,j,i)/W;
      const Real dm = u0(m, IDN, k, j, i)*size.d_view(m).dx1*size.d_view(m).dx2*
                      size.d_view(m).dx3;
      a0 += dm; a1 += dm*x; a2 += dm*y; a3 += dm*z;
      a4 += dm*(aw*ux - adm.beta_u(m,0,k,j,i));
      a5 += dm*(aw*uy - adm.beta_u(m,1,k,j,i));
      a6 += dm*(aw*uz - adm.beta_u(m,2,k,j,i));
    }, Kokkos::Sum<Real>(s[0]), Kokkos::Sum<Real>(s[1]), Kokkos::Sum<Real>(s[2]),
       Kokkos::Sum<Real>(s[3]), Kokkos::Sum<Real>(s[4]), Kokkos::Sum<Real>(s[5]),
       Kokkos::Sum<Real>(s[6]));
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, s, 7, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
    if (!(s[0] > 0.0)) { t3_active = false; return; }
    const Real cx = s[1]/s[0], cy = s[2]/s[0], cz = s[3]/s[0];
    Kokkos::deep_copy(t3_hist, 0.0);
    auto &hist = t3_hist;
    const Real idr = 1.0/t3_dr;
    par_for("zone_t3_r90", DevExeSpace(), 0, nmb-1, ks, ks+nx3-1, js, js+nx2-1, is, is+nx1-1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      if (!(w0(m, IDN, k, j, i) > thr)) return;
      const Real x = CellCenterX(i - is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      const Real y = CellCenterX(j - js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      const Real z = CellCenterX(k - ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
      const Real r = sqrt((x - cx)*(x - cx) + (y - cy)*(y - cy) + (z - cz)*(z - cz));
      const int b = static_cast<int>(fmin(r*idr, kT3NR - 1.0));
      Kokkos::atomic_add(&hist(b), u0(m, IDN, k, j, i)*size.d_view(m).dx1*
                         size.d_view(m).dx2*size.d_view(m).dx3);
    });
    auto hh = Kokkos::create_mirror_view_and_copy(HostMemSpace(), t3_hist);
    Real hv[kT3NR];
    for (int b = 0; b < kT3NR; ++b) hv[b] = hh(b);
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, hv, kT3NR, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
    Real cum = 0.0, tot = 0.0;
    for (int b = 0; b < kT3NR; ++b) tot += hv[b];
    int b90 = kT3NR - 1;
    for (int b = 0; b < kT3NR; ++b) {
      cum += hv[b];
      if (cum >= 0.9*tot) { b90 = b; break; }
    }
    t3_r90 = (b90 + 1)*t3_dr;
    const Real *xb = pmy_pack->padm->bc_center, *xd = pmy_pack->padm->frame_vel_u;
    const Real d[3] = {cx - xb[0], cy - xb[1], cz - xb[2]};
    const Real dd[3] = {s[4]/s[0] + xd[0], s[5]/s[0] + xd[1], s[6]/s[0] + xd[2]};
    t3_d = sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
    t3_ddot = d[0]*dd[0] + d[1]*dd[1] + d[2]*dd[2];
    t3_c[0] = cx; t3_c[1] = cy; t3_c[2] = cz;
    t3_active = (t3_d > 2.0*zone_r + 3.0*t3_r90) && (t3_ddot > 0.0);
  }

  // <mhd> mass_ledger (default false; research R-066 item 2, a debug diagnostic): a
  // per-RK-stage rest-mass ledger over the interior cells, to localise the unbooked
  // grid-mass loss of C-051 §5. Per stage s, with M = sum u0(IDN) dV:
  //   M_pre(s) before the C2P, M_end(s) after it, M(u1) (the cycle-start register);
  //   Y_s = M_pre(s) - gam0_s M_end(s-1) - gam1_s M(u1)   (the RK update's own change;
  //         = -beta_s dt F_out,s if the flux update conserves, F_out = net outflow
  //         through the mesh's outer faces);
  //   X_s = M_end(s) - M_pre(s) (the C2P's change), binned by class (0 excision floor
  //         and flux mask, 1 floor mask only, 2 zone/tier 3, 3 C2P-floored, 4 other) x
  //         r_BH (< 1.88, 1.88-2.92, 2.92-6, 6-12, >= 12) x (removed, added);
  //   F_exc,s = the mass flux into excision-flux cells through faces they share with
  //         cells outside that mask; FOFC-flagged cells (outside it) at r < 6 / 6-12.
  // With the propagation weights w_s = prod_{j>s} gam0_j, M_end(ns) - M(u1) =
  // sum_s w_s (Y_s + X_s) holds identically; one MLEDGER (+ MLEDGER_X bins) line per
  // cycle on rank 0. The change between M_end of one cycle and M(u1) of the next is the
  // regrid's (or anything else outside the stages).
  bool ml_on = false;
  static constexpr int kMlNC = 5, kMlNR = 5, kMlNX = kMlNC*kMlNR*2;
  // per-stage slots: [0, kMlNX) X bins, then M_pre, M_end, M(u1), F_exc, F_out,
  // n_fofc(r < 6), n_fofc(6-12), n_cells
  static constexpr int kMlPre = kMlNX, kMlEnd = kMlNX + 1, kMlU1 = kMlNX + 2;
  static constexpr int kMlFexc = kMlNX + 3, kMlFout = kMlNX + 4, kMlFo6 = kMlNX + 5;
  static constexpr int kMlFo12 = kMlNX + 6, kMlNcell = kMlNX + 7, kMlLen = kMlNX + 8;
  DvceArray4D<Real> ml_d0, ml_cls;   // D sqrt(gamma) before the C2P; the cell's class
  Real ml_stage[4][kMlLen] = {};

  //! sum over interior cells of a(m, n, k, j, i) dV
  Real MlMass(const DvceArray5D<Real> &a, int n) {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    const int is = indcs.is, js = indcs.js, ks = indcs.ks;
    const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
    const int nmb = pmy_pack->nmb_thispack, nkji = nx3*nx2*nx1, nji = nx2*nx1;
    auto &size = pmy_pack->pmb->mb_size;
    Real s = 0.0;
    Kokkos::parallel_reduce("ml_mass", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*nkji),
    KOKKOS_LAMBDA(const int idx, Real &a0) {
      const int m = idx/nkji;
      const int k = (idx - m*nkji)/nji + ks;
      const int j = (idx - m*nkji - (k - ks)*nji)/nx1 + js;
      const int i = idx - m*nkji - (k - ks)*nji - (j - js)*nx1 + is;
      a0 += a(m, n, k, j, i)*size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    }, Kokkos::Sum<Real>(s));
    return s;
  }

  //! before the stage's C2P: M_pre, M(u1), and a copy of u0(IDN)
  void MassLedgerPre(int stage) {
    if (!ml_on || stage < 1 || stage > 4) return;
    auto &u0 = pmy_pack->pmhd->u0;
    const int nmb = pmy_pack->nmb_thispack;
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    const int n3 = indcs.nx3 + 2*indcs.ng, n2 = indcs.nx2 + 2*indcs.ng;
    const int n1 = indcs.nx1 + 2*indcs.ng;
    const int nmbu = static_cast<int>(u0.extent(0));  // u0 is sized for the rank's max
    if (static_cast<int>(ml_d0.extent(0)) != nmbu) {  // (re)sized after a regrid
      Kokkos::realloc(ml_d0, nmbu, n3, n2, n1);
      Kokkos::realloc(ml_cls, nmbu, n3, n2, n1);
    }
    Kokkos::deep_copy(ml_d0, Kokkos::subview(u0, Kokkos::ALL, static_cast<int>(IDN),
                                             Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
    Kokkos::deep_copy(ml_cls, 4.0);
    for (int n = 0; n < kMlLen; ++n) ml_stage[stage-1][n] = 0.0;
    ml_stage[stage-1][kMlPre] = MlMass(u0, IDN);
    ml_stage[stage-1][kMlU1] = MlMass(pmy_pack->pmhd->u1, IDN);
  }

  //! after the stage's C2P: the X bins, M_end, the face fluxes and FOFC counts; on the
  //! last stage combine, reduce over ranks and print (rank 0)
  void MassLedgerPost(int stage, int ns, const Real *gam0, const Real *gam1,
                      const Real *beta, Real dt, Real time, int cycle) {
    if (!ml_on || stage < 1 || stage > ns || ns > 4) return;
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    const int is = indcs.is, js = indcs.js, ks = indcs.ks;
    const int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
    const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
    const int nmb = pmy_pack->nmb_thispack, nkji = nx3*nx2*nx1, nji = nx2*nx1;
    auto &size = pmy_pack->pmb->mb_size;
    auto &u0 = pmy_pack->pmhd->u0;
    auto &d0 = ml_d0;
    auto &cls = ml_cls;
    const Real cx = pmy_pack->padm->bc_center[0], cy = pmy_pack->padm->bc_center[1];
    const Real cz = pmy_pack->padm->bc_center[2];
    Real *st = ml_stage[stage-1];
    // X bins: one reduction per class (10 sums: r bin x removed/added)
    for (int c = 0; c < kMlNC; ++c) {
      Real v[10] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      const Real cc = static_cast<Real>(c);
      Kokkos::parallel_reduce("ml_x", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*nkji),
      KOKKOS_LAMBDA(const int idx, Real &a0, Real &a1, Real &a2, Real &a3, Real &a4,
                    Real &a5, Real &a6, Real &a7, Real &a8, Real &a9) {
        const int m = idx/nkji;
        const int k = (idx - m*nkji)/nji + ks;
        const int j = (idx - m*nkji - (k - ks)*nji)/nx1 + js;
        const int i = idx - m*nkji - (k - ks)*nji - (j - js)*nx1 + is;
        if (cls(m, k, j, i) != cc) return;
        const Real dm = (u0(m, IDN, k, j, i) - d0(m, k, j, i))*
                        size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
        if (dm == 0.0) return;
        const Real x = CellCenterX(i - is, nx1, size.d_view(m).x1min,
                                   size.d_view(m).x1max) - cx;
        const Real y = CellCenterX(j - js, nx2, size.d_view(m).x2min,
                                   size.d_view(m).x2max) - cy;
        const Real z = CellCenterX(k - ks, nx3, size.d_view(m).x3min,
                                   size.d_view(m).x3max) - cz;
        const Real r2 = x*x + y*y + z*z;
        const int ir = (r2 >= 1.88*1.88) + (r2 >= 2.92*2.92) + (r2 >= 36.0) + (r2 >= 144.0);
        const Real rem = (dm < 0.0) ? -dm : 0.0, add = (dm > 0.0) ? dm : 0.0;
        if (ir == 0) { a0 += rem; a1 += add; }
        else if (ir == 1) { a2 += rem; a3 += add; }
        else if (ir == 2) { a4 += rem; a5 += add; }
        else if (ir == 3) { a6 += rem; a7 += add; }
        else { a8 += rem; a9 += add; }
      }, Kokkos::Sum<Real>(v[0]), Kokkos::Sum<Real>(v[1]), Kokkos::Sum<Real>(v[2]),
         Kokkos::Sum<Real>(v[3]), Kokkos::Sum<Real>(v[4]), Kokkos::Sum<Real>(v[5]),
         Kokkos::Sum<Real>(v[6]), Kokkos::Sum<Real>(v[7]), Kokkos::Sum<Real>(v[8]),
         Kokkos::Sum<Real>(v[9]));
      for (int q = 0; q < 10; ++q) st[c*10 + q] = v[q];
    }
    st[kMlEnd] = MlMass(u0, IDN);
    // face fluxes (this stage's uflx, after the flux correction) and FOFC counts
    const bool exc = pmy_pack->pcoord->coord_data.bh_excise;
    auto &xf = pmy_pack->pcoord->excision_flux;
    auto &flx = pmy_pack->pmhd->uflx;
    auto &fofc = pmy_pack->pmhd->fofc;
    const bool use_fofc = pmy_pack->pmhd->use_fofc;
    auto &ms = pmy_pack->pmesh->mesh_size;
    const Real mx1 = ms.x1min, Mx1 = ms.x1max, mx2 = ms.x2min, Mx2 = ms.x2max;
    const Real mx3 = ms.x3min, Mx3 = ms.x3max;
    Real fe = 0.0, fo = 0.0, n6 = 0.0, n12 = 0.0, nc = 0.0;
    Kokkos::parallel_reduce("ml_flux", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*nkji),
    KOKKOS_LAMBDA(const int idx, Real &a0, Real &a1, Real &a2, Real &a3, Real &a4) {
      const int m = idx/nkji;
      const int k = (idx - m*nkji)/nji + ks;
      const int j = (idx - m*nkji - (k - ks)*nji)/nx1 + js;
      const int i = idx - m*nkji - (k - ks)*nji - (j - js)*nx1 + is;
      auto &sz = size.d_view(m);
      const Real a1f = sz.dx2*sz.dx3, a2f = sz.dx1*sz.dx3, a3f = sz.dx1*sz.dx2;
      a4 += 1.0;
      const bool xc = exc && xf(m, k, j, i);
      if (xc) {   // inflow through faces shared with cells outside the flux mask
        if (!xf(m, k, j, i-1)) a0 += flx.x1f(m, IDN, k, j, i)*a1f;
        if (!xf(m, k, j, i+1)) a0 -= flx.x1f(m, IDN, k, j, i+1)*a1f;
        if (!xf(m, k, j-1, i)) a0 += flx.x2f(m, IDN, k, j, i)*a2f;
        if (!xf(m, k, j+1, i)) a0 -= flx.x2f(m, IDN, k, j+1, i)*a2f;
        if (!xf(m, k-1, j, i)) a0 += flx.x3f(m, IDN, k, j, i)*a3f;
        if (!xf(m, k+1, j, i)) a0 -= flx.x3f(m, IDN, k+1, j, i)*a3f;
      }
      // outflow through the mesh's outer faces
      const Real tol = 1.0e-6*sz.dx1;
      if (i == is && sz.x1min <= mx1 + tol) a1 -= flx.x1f(m, IDN, k, j, i)*a1f;
      if (i == ie && sz.x1max >= Mx1 - tol) a1 += flx.x1f(m, IDN, k, j, i+1)*a1f;
      if (j == js && sz.x2min <= mx2 + tol) a1 -= flx.x2f(m, IDN, k, j, i)*a2f;
      if (j == je && sz.x2max >= Mx2 - tol) a1 += flx.x2f(m, IDN, k, j+1, i)*a2f;
      if (k == ks && sz.x3min <= mx3 + tol) a1 -= flx.x3f(m, IDN, k, j, i)*a3f;
      if (k == ke && sz.x3max >= Mx3 - tol) a1 += flx.x3f(m, IDN, k+1, j, i)*a3f;
      if (use_fofc && !xc && fofc(m, k, j, i)) {
        const Real x = CellCenterX(i - is, nx1, sz.x1min, sz.x1max) - cx;
        const Real y = CellCenterX(j - js, nx2, sz.x2min, sz.x2max) - cy;
        const Real z = CellCenterX(k - ks, nx3, sz.x3min, sz.x3max) - cz;
        const Real r2 = x*x + y*y + z*z;
        if (r2 < 36.0) a2 += 1.0; else if (r2 < 144.0) a3 += 1.0;
      }
    }, Kokkos::Sum<Real>(fe), Kokkos::Sum<Real>(fo), Kokkos::Sum<Real>(n6),
       Kokkos::Sum<Real>(n12), Kokkos::Sum<Real>(nc));
    st[kMlFexc] = beta[stage-1]*dt*fe;   // mass over the stage's update
    st[kMlFout] = beta[stage-1]*dt*fo;
    st[kMlFo6] = n6; st[kMlFo12] = n12; st[kMlNcell] = nc;
    if (stage != ns) return;
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &ml_stage[0][0], 4*kMlLen, MPI_ATHENA_REAL, MPI_SUM,
                  MPI_COMM_WORLD);
#endif
    Real w[4];
    w[ns-1] = 1.0;
    for (int q = ns-2; q >= 0; --q) w[q] = gam0[q+1]*w[q+1];
    const Real mu1 = ml_stage[0][kMlU1];
    Real Y = 0.0, X = 0.0, Fe = 0.0, Fo = 0.0, f6 = 0.0, f12 = 0.0, xb[kMlNX];
    for (int n = 0; n < kMlNX; ++n) xb[n] = 0.0;
    Real mprev = mu1;
    for (int q = 0; q < ns; ++q) {
      const Real *s = ml_stage[q];
      Y += w[q]*(s[kMlPre] - gam0[q]*mprev - gam1[q]*mu1);
      X += w[q]*(s[kMlEnd] - s[kMlPre]);
      Fe += w[q]*s[kMlFexc];
      Fo += w[q]*s[kMlFout];
      f6 += s[kMlFo6]; f12 += s[kMlFo12];
      for (int n = 0; n < kMlNX; ++n) xb[n] += w[q]*s[n];
      mprev = s[kMlEnd];
    }
    if (global_variable::my_rank == 0) {
      std::cout << std::setprecision(10) << "MLEDGER cycle=" << cycle << " time=" << time
                << " dt=" << dt << " Mu1=" << mu1 << " Mend=" << ml_stage[ns-1][kMlEnd]
                << " Y=" << Y << " Fout=" << Fo << " Fexc=" << Fe << " X=" << X
                << " fofc6=" << f6 << " fofc12=" << f12 << " ncell="
                << ml_stage[0][kMlNcell] << "," << ml_stage[ns-1][kMlNcell] << std::endl;
      std::cout << "MLEDGER_X cycle=" << cycle;
      for (int n = 0; n < kMlNX; ++n) std::cout << " " << xb[n];
      std::cout << std::endl;
    }
  }

  //! Entropy inversion for B = 0: given undensitized D, S^2 = S_i S^i and K, solve
  //! D^2 h^2 (W^2 - 1) = S^2 with rho = D/W, h = 1 + g/(g-1) K rho^(g-1) for W by
  //! bisection on [1, sqrt(1 + S^2/D^2)] (f(1) = -S^2 <= 0, f(Whi) >= 0 since h >= 1).
  KOKKOS_INLINE_FUNCTION
  static bool EntropyInversion(Real D, Real ssq, Real K, Real gam, Real &W, Real &h) {
    if (!(D > 0.0) || !(K > 0.0) || !(ssq >= 0.0)) return false;
    const Real gfac = gam/(gam - 1.0);
    Real wlo = 1.0, whi = sqrt(1.0 + ssq/(D*D));
    for (int it = 0; it < 200 && (whi - wlo) > 1.0e-15*whi; ++it) {
      Real wm = 0.5*(wlo + whi);
      Real hm = 1.0 + gfac*K*pow(D/wm, gam - 1.0);
      if (D*D*hm*hm*(wm*wm - 1.0) - ssq > 0.0) { whi = wm; } else { wlo = wm; }
    }
    W = 0.5*(wlo + whi);
    h = 1.0 + gfac*K*pow(D/W, gam - 1.0);
    return isfinite(W) && isfinite(h);
  }

  //! tracer <-> adiabat for Y = (K/kmax)^p
  KOKKOS_INLINE_FUNCTION
  static Real DualEnergyY(Real K, Real kmax, Real p) {
    Real y = fmin(fmax(K/kmax, 0.0), 1.0);
    return (p == 1.0) ? y : pow(y, p);
  }
  KOKKOS_INLINE_FUNCTION
  static Real DualEnergyK(Real Y, Real kmax, Real p) {
    return (p == 1.0) ? kmax*Y : kmax*pow(fmax(Y, 0.0), 1.0/p);
  }

  PrimitiveSolverHydro(std::string block, MeshBlockPack *pp, ParameterInput *pin) :
//        pmy_pack(pp), ps{&eos} {
        pmy_pack(pp), nerrs(0) {
    SetPolicyParams(block, pin);
    atm_frame_vel = pin->GetOrAddBoolean(block, "atmosphere_frame_velocity", false);
    atm_frame_gate = pin->GetOrAddReal(block, "atmosphere_frame_gate", 0.0);
    zone_on = pin->GetOrAddBoolean(block, "atmosphere_zone", false);
    zone_book = zone_on || pin->GetOrAddBoolean(block, "atmosphere_zone_book", false);
    ml_on = pin->GetOrAddBoolean(block, "mass_ledger", false);
    if (zone_book && !zone_on) {
      Kokkos::realloc(zone_acc, 2*kZoneLen);
      Kokkos::deep_copy(zone_acc, 0.0);
    }
    if (zone_on) {
      zone_r = pin->GetReal(block, "atmosphere_zone_radius");
      zone_rho = pin->GetReal(block, "atmosphere_zone_rho");
      zone_rho_reset = pin->GetOrAddReal(block, "atmosphere_zone_rho_reset", 1.0e-3*zone_rho);
      zone_vmax = pin->GetOrAddReal(block, "atmosphere_zone_vmax", 0.9);
      zone_de = pin->GetOrAddReal(block, "atmosphere_zone_de", zone_de);
      if (!(zone_r > 0.0) || !(zone_rho > 0.0) || zone_rho_reset > zone_rho ||
          !(zone_vmax > 0.0 && zone_vmax < 1.0)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<" << block << "> atmosphere_zone needs radius > 0, "
                  << "rho > 0, rho_reset <= rho and 0 < vmax < 1" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (atm_frame_vel && atm_frame_gate != zone_r) {
        if (global_variable::my_rank == 0) {
          std::cout << "<" << block << "> atmosphere_zone: atmosphere_frame_gate "
                    << atm_frame_gate << " -> " << zone_r << " (the zone radius)"
                    << std::endl;
        }
        atm_frame_gate = zone_r;
      }
      zone_bins_every = pin->GetOrAddInteger(block, "atmosphere_zone_bins_every", 50);
      t3_on = pin->GetOrAddBoolean(block, "atmosphere_zone_tier3", false);
      t3_rho = pin->GetOrAddReal(block, "atmosphere_zone_tier3_rho", 1.0e-6*zone_rho);
      t3_dr = pin->GetOrAddReal(block, "atmosphere_zone_tier3_dr", 1.0/32.0);
      if (t3_on) {
        Kokkos::realloc(t3_hist, kT3NR);
      }
      Kokkos::realloc(zone_acc, 2*kZoneLen);
      Kokkos::deep_copy(zone_acc, 0.0);
    }
    de_on = pin->GetOrAddBoolean(block, "dual_energy", false);
    if (de_on) {
      int nsc = pin->GetOrAddInteger(block, "nscalars", 0);
      if (!std::is_same_v<Primitive::IdealGas, EOSPolicy> || nsc < 1) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<" << block << "> dual_energy needs the ideal-gas EOS"
                  << " and nscalars >= 1 (the last scalar carries K)" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      de_idx = nsc - 1;
      de_gamma = pin->GetOrAddReal(block, "gamma", 5.0/3.0);
      de_kmax = pin->GetReal(block, "dual_energy_kmax");
      de_rho_sw = pin->GetReal(block, "dual_energy_rho_switch");
      de_dp_shock = pin->GetOrAddReal(block, "dual_energy_dp_shock", 0.3);
      std::string sen = pin->GetOrAddString(block, "dual_energy_shock_sensor", "vcurv");
      if (sen == "jump") {
        de_sensor = 0;
      } else if (sen == "jameson") {
        de_sensor = 1;
      } else if (sen == "vcurv") {
        de_sensor = 2;
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<" << block << "> dual_energy_shock_sensor = '" << sen
                  << "' (use jump, jameson or vcurv)" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      std::string dv = pin->GetOrAddString(block, "dual_energy_div", "coord");
      if (dv != "coord" && dv != "covariant") {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<" << block << "> dual_energy_div = '" << dv
                  << "' (use coord or covariant)" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      de_div_cov = (dv == "covariant");
      de_psi_shock = pin->GetOrAddReal(block, "dual_energy_psi_shock", 0.25);
      de_vcurv_shock = pin->GetOrAddReal(block, "dual_energy_vcurv_shock", 0.05);
      de_psi_jump = pin->GetOrAddReal(block, "dual_energy_psi_jump", 1.0e30);  // off
      de_dilate = pin->GetOrAddInteger(block, "dual_energy_shock_dilate", 1);
      de_dilate_weak = pin->GetOrAddReal(block, "dual_energy_shock_dilate_weak", 0.5);
      de_isolated = pin->GetOrAddBoolean(block, "dual_energy_shock_isolated_off", false);
      de_persist = pin->GetOrAddInteger(block, "dual_energy_shock_persist", 1);
      de_second_law = pin->GetOrAddBoolean(block, "dual_energy_second_law", true);
      de_second_law_tol = pin->GetOrAddReal(block, "dual_energy_second_law_tol", -1.0e-3);
      de_eta1 = pin->GetOrAddReal(block, "dual_energy_eta1", 0.0);
      std::string var = pin->GetOrAddString(block, "dual_energy_variable", "k_1g");
      if (var == "k_1g") {
        de_ypow = 1.0/de_gamma;
      } else if (var == "k") {
        de_ypow = 1.0;
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<" << block << "> dual_energy_variable = '" << var
                  << "' (use k_1g or k)" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      de_recon_rhoy = pin->GetOrAddBoolean(block, "dual_energy_recon_rhoy",
                                           de_ypow != 1.0);
      // hll is only implemented in the HLLE solver (hlle_dyn_grmhd.hpp)
      bool hlle = pin->GetOrAddString(block, "rsolver", "llf") == "hlle";
      std::string fl = pin->GetOrAddString(block, "dual_energy_flux",
                                           (de_ypow != 1.0 && hlle) ? "hll" : "mass");
      if ((fl != "mass" && fl != "hll") || (fl == "hll" && !hlle)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<" << block << "> dual_energy_flux = '" << fl
                  << "' (use mass or hll; hll needs rsolver = hlle)" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      de_flux_hll = (fl == "hll");
    }
    Real mb = ps.GetEOS().GetBaryonMass();
    ps.GetEOSMutable().SetDensityFloor(pin->GetOrAddReal(block, "dfloor", (FLT_MIN))/mb);
    ps.GetEOSMutable().SetTemperatureFloor(pin->GetOrAddReal(block, "tfloor", (FLT_MIN)));
    ps.GetEOSMutable().SetThreshold(pin->GetOrAddReal(block, "dthreshold", 1.0));
    ps.tol = pin->GetOrAddReal(block, "c2p_tol", 1e-15);
    ps.GetRootSolverMutable().iterations = pin->GetOrAddInteger(block, "c2p_iter", 50);
    errcap = pin->GetOrAddInteger(block, "c2perrs", 1000);

    // Calculate maximum allowed velocity, for the root-find's internal estimate
    // (SetMaxVelocity/v_max, primitive_solver.hpp's RootFunction) -- this does NOT
    // by itself bound the converged output; SetMaxLorentzFactor below does that.
    Real Wmax = pin->GetOrAddReal(block, "gamma_max", 50.0);
    Real vmax = sqrt(1.0 - 1.0/(Wmax*Wmax));
    ps.GetEOSMutable().SetMaxVelocity(vmax);
    ps.GetEOSMutable().SetMaxLorentzFactor(Wmax);

    // Set maximum B^2/D
    ps.GetEOSMutable().SetMaximumMagnetization(pin->GetOrAddReal(block, "max_bsq", 1e6));

    for (int n = 0; n < ps.GetEOS().GetNSpecies(); n++) {
      std::stringstream spec_name;
      spec_name << "s" << n << "_atmosphere";
      ps.GetEOSMutable().SetSpeciesAtmosphere(
          pin->GetOrAddReal(block, spec_name.str(), 0.0), n);
    }
  }

  // The prim to con function used on the reconstructed states inside the Riemann solver.
  // It also extracts the primitives into a form usable by PrimitiveSolver.
  KOKKOS_INLINE_FUNCTION
  void PrimToConsPt(const ScrArray2D<Real> &w, const ScrArray2D<Real> &brc,
                    const DvceArray4D<Real> &bx,
                    Real prim_pt[NPRIM], Real cons_pt[NCONS], Real b[NMAG],
                    Real g3d[NSPMETRIC], Real sdetg,
                    const int m, const int k, const int j, const int i,
                    const int &nhyd, const int &nscal,
                    const int ibx, const int iby, const int ibz) const {
    auto &eos = ps.GetEOS();
    Real mb = eos.GetBaryonMass();
    // The magnetic field is densitized, but the PrimToCon call
    // needs undensitized variables.
    Real isdetg = 1.0/sdetg;
    Real bin[NMAG];
    bin[ibx] = bx(m, k, j, i)*isdetg;
    bin[iby] = brc(iby, i)*isdetg;
    bin[ibz] = brc(ibz, i)*isdetg;
    Real prim_pt_old[NPRIM];
    prim_pt[PRH] = prim_pt_old[PRH] = w(IDN, i)/mb;
    prim_pt[PVX] = prim_pt_old[PVX] = w(IVX, i);
    prim_pt[PVY] = prim_pt_old[PVY] = w(IVY, i);
    prim_pt[PVZ] = prim_pt_old[PVZ] = w(IVZ, i);
    for (int n = 0; n < nscal; n++) {
      prim_pt[PYF + n] = prim_pt_old[PYF + n] = w(nhyd + n, i);
    }
    prim_pt[PPR] = prim_pt_old[PPR] = w(IPR, i);

    // Apply the floor to make sure these values are physical.
    // FIXME(JF): Is this needed if the first-order flux correction is enabled?
    prim_pt[PTM] = prim_pt_old[PTM] = eos.GetTemperatureFromP(prim_pt[PRH],
                                        prim_pt[PPR], &prim_pt[PYF]);
    ps.GetEOS().ApplyPrimitiveFloor(prim_pt[PRH], &prim_pt[PVX],
                                    prim_pt[PPR], prim_pt[PTM], &prim_pt[PYF]);

    ps.PrimToCon(prim_pt, cons_pt, bin, g3d);

    // Check for NaNs
    /*if (CheckForConservedNaNs(cons_pt)) {
      printf("Location: PrimToConsPt\n");
      DumpPrimitiveVars(prim_pt);
    }*/

    // Densitize the variables
    for (int n = 0; n < nhyd + nscal; n++) {
      cons_pt[n] *= sdetg;
    }
    b[ibx] = bx(m, k, j, i);
    b[iby] = brc(iby, i);
    b[ibz] = brc(ibz, i);

    // Previously we checked if the floor was applied and copied these variables back
    // into the original array. However, this is pointless because only the extracted
    // variables in the C-style array are used from this point forward.
  }

  void PrimToCons(DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                  DvceArray5D<Real> &cons,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku) {
    //int &is = indcs.is, &js = indcs.js, &ks = indcs.ks;
    //auto &size = pmy_pack->pmb->mb_size;
    //auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
    auto &eos_ = ps.GetEOS();
    auto &ps_  = ps;

    auto &adm = pmy_pack->padm->adm;

    int &nhyd = pmy_pack->pmhd->nmhd;
    int &nscal = pmy_pack->pmhd->nscalars;
    int &nmb = pmy_pack->nmb_thispack;

    Real mb = eos_.GetBaryonMass();
    const bool de_on_ = de_on;
    const int de_idx_ = de_idx;
    const Real de_kmax_ = de_kmax, de_gamma_ = de_gamma, de_ypow_ = de_ypow;


    par_for("pshyd_prim2cons", DevExeSpace(), 0, (nmb-1), kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // Extract metric at a single point
      Real g3d[NSPMETRIC];
      g3d[S11] = adm.g_dd(m, 0, 0, k, j, i);
      g3d[S12] = adm.g_dd(m, 0, 1, k, j, i);
      g3d[S13] = adm.g_dd(m, 0, 2, k, j, i);
      g3d[S22] = adm.g_dd(m, 1, 1, k, j, i);
      g3d[S23] = adm.g_dd(m, 1, 2, k, j, i);
      g3d[S33] = adm.g_dd(m, 2, 2, k, j, i);
      Real sdetg = sqrt(Primitive::GetDeterminant(g3d));

      // The magnetic field is densitized, but the PrimToCon calculation is
      // done with undensitized variables.
      Real b[NMAG] = {bcc(m, IBX, k, j, i)/sdetg,
                      bcc(m, IBY, k, j, i)/sdetg,
                      bcc(m, IBZ, k, j, i)/sdetg};

      // Extract primitive variables at a single point
      Real prim_pt[NPRIM], cons_pt[NCONS];
      prim_pt[PRH] = prim(m, IDN, k, j, i)/mb;
      prim_pt[PVX] = prim(m, IVX, k, j, i);
      prim_pt[PVY] = prim(m, IVY, k, j, i);
      prim_pt[PVZ] = prim(m, IVZ, k, j, i);
      for (int n = 0; n < nscal; n++) {
        prim_pt[PYF + n] = prim(m, nhyd + n, k, j, i);
      }
      // FIXME: Debug only! Use specific energy to validate other
      // hydro functions before breaking things.
      //Real e = prim(m, IDN, k, j, i) + prim(m, IEN, k, j, i);
      //prim_pt[PTM] = eos_.GetTemperatureFromE(prim_pt[PRH], e, &prim_pt[PYF]);
      //prim_pt[PPR] = eos_.GetPressure(prim_pt[PRH], prim_pt[PTM], &prim_pt[PYF]);
      prim_pt[PPR] = prim(m, IPR, k, j, i);

      // Apply the floor to make sure these values are physical.
      prim_pt[PTM] = eos_.GetTemperatureFromP(prim_pt[PRH], prim_pt[PPR], &prim_pt[PYF]);
      bool floor = eos_.ApplyPrimitiveFloor(prim_pt[PRH], &prim_pt[PVX],
                                           prim_pt[PPR], prim_pt[PTM], &prim_pt[PYF]);
      if (de_on_) {
        // the K tracer is always derived from P and rho at prim->cons time
        Real rho_ = prim_pt[PRH]*mb;
        prim_pt[PYF + de_idx_] = DualEnergyY(prim_pt[PPR]/pow(rho_, de_gamma_), de_kmax_,
                                             de_ypow_);
        prim(m, nhyd + de_idx_, k, j, i) = prim_pt[PYF + de_idx_];
      }

      ps_.PrimToCon(prim_pt, cons_pt, b, g3d);

      // Check for NaNs
      if (CheckForConservedNaNs(cons_pt)) {
        Kokkos::printf("Error occurred in PrimToCons at (%d, %d, %d, %d)\n", m, k, j, i);
        DumpPrimitiveVars(prim_pt);
      }

      // Save the densitized conserved variables.
      cons(m, IDN, k, j, i) = cons_pt[CDN]*sdetg;
      cons(m, IM1, k, j, i) = cons_pt[CSX]*sdetg;
      cons(m, IM2, k, j, i) = cons_pt[CSY]*sdetg;
      cons(m, IM3, k, j, i) = cons_pt[CSZ]*sdetg;
      cons(m, IEN, k, j, i) = cons_pt[CTA]*sdetg;
      for (int n = 0; n < nscal; n++) {
        cons(m, nhyd + n, k, j, i) = cons_pt[CYD + n]*sdetg;
      }

      // If we floored the primitive variables, we need to adjust those, too.
      if (floor) {
        prim(m, IDN, k, j, i) = prim_pt[PRH]*mb;
        prim(m, IVX, k, j, i) = prim_pt[PVX];
        prim(m, IVY, k, j, i) = prim_pt[PVY];
        prim(m, IVZ, k, j, i) = prim_pt[PVZ];
        prim(m, IPR, k, j, i) = prim_pt[PPR];
        for (int n = 0; n < nscal; n++) {
          prim(m, nhyd + n, k, j, i) = prim_pt[PYF + n];
        }
      }
    });

    return;
  }

  void ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &bfc,
                  DvceArray5D<Real> &bcc0, DvceArray5D<Real> &prim,
                  DvceArray5D<Real> &temperature,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku, bool floors_only=false) {
    int &nhyd = pmy_pack->pmhd->nmhd;
    int &nscal = pmy_pack->pmhd->nscalars;
    int &nmb = pmy_pack->nmb_thispack;
    auto &fofc_ = pmy_pack->pmhd->fofc;

    // Some problem-specific parameters
    auto &excise = pmy_pack->pcoord->coord_data.bh_excise;
    auto &smoothing = pmy_pack->pcoord->coord_data.smooth_excision;
    auto &excision_floor_ = pmy_pack->pcoord->excision_floor;
    auto &excision_flux_ = pmy_pack->pcoord->excision_flux;
    auto &dexcise_ = pmy_pack->pcoord->coord_data.dexcise;
    auto &texcise_ = pmy_pack->pcoord->coord_data.texcise;

    auto &adm  = pmy_pack->padm->adm;
    auto &eos_ = ps.GetEOS();
    auto &ps_  = ps;

    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int &is = indcs.is;
    int &js = indcs.js;
    int &ks = indcs.ks;
    auto &size = pmy_pack->pmb->mb_size;

    // Captured by value for the excised-mass tally below (interior-cell guard).
    // ie/je/ke are copies because indcs members cannot be captured by reference into
    // a device lambda. The tally itself is accumulated via this call's own
    // Kokkos::parallel_reduce below (see the reducer arguments and coordinates.hpp's
    // excised_tally comment for why -- deterministic, unlike the atomic-add scatter
    // this replaced).
    const int ie_ = indcs.ie, je_ = indcs.je, ke_ = indcs.ke;
    const bool floors_only_ = floors_only;

    const int ni = (iu - il + 1);
    const int nji = (ju - jl + 1)*ni;
    const int nkji = (ku - kl + 1)*nji;
    const int nmkji = nmb*nkji;

    const int rank = global_variable::my_rank;
    const int nerrs_ = nerrs;
    const int errcap_ = errcap;

    Real mb = eos_.GetBaryonMass();

    // FIXME: This only works for a flooring policy that has these functions!
    bool prim_failure=false, cons_failure=false;
    if (floors_only) {
      prim_failure = ps.GetEOSMutable().IsPrimitiveFlooringFailure();
      cons_failure = ps.GetEOSMutable().IsConservedFlooringFailure();
      ps.GetEOSMutable().SetPrimitiveFloorFailure(true);
      ps.GetEOSMutable().SetConservedFloorFailure(true);
    }

    // FIXME(JMF): We can short-circuit the primitive solve if FOFC is already enabled
    // due to a maximum principle violation.
    // ---- dual energy: shock flag from the previous primitives (before they are
    // overwritten below): compressive (div v <= 0) and a pressure jump > de_dp_shock
    // to any face neighbour (neighbours clamped to this call's index range).
    const bool de_on_ = de_on && !floors_only;
    const int de_idx_ = de_idx;
    const Real de_kmax_ = de_kmax, de_rho_sw_ = de_rho_sw, de_eta1_ = de_eta1;
    const Real de_gamma_ = de_gamma, de_ypow_ = de_ypow;
    const bool de_2law_ = de_second_law;
    const Real de_2tol_ = de_second_law_tol;
    const bool ddon_ = de_on_ && de_diag_on;
    const int dstg_ = de_stage;
    auto &dd_ = de_diag;
    if (de_on_) {
      int e1 = prim.extent_int(4), e2 = prim.extent_int(3), e3 = prim.extent_int(2);
      if (de_shock.extent_int(0) != prim.extent_int(0) || de_shock.extent_int(1) != e3 ||
          de_shock.extent_int(2) != e2 || de_shock.extent_int(3) != e1) {
        Kokkos::realloc(de_shock, prim.extent_int(0), e3, e2, e1);
      }
      auto &shk = de_shock;
      const Real dpsh = de_dp_shock;
      const int sensor = de_sensor;
      const Real psish = de_psi_shock, vcsh = de_vcurv_shock, psijp = de_psi_jump;
      const Real gam_ = de_gamma;
      const Real wk = de_dilate_weak;
      const bool divcov = de_div_cov;
      par_for("de_shock_flag", DevExeSpace(), 0, nmb-1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        auto vel = [&](int a, int kk, int jj, int ii) {
          Real u0 = prim(m, IVX, kk, jj, ii), u1 = prim(m, IVY, kk, jj, ii);
          Real u2 = prim(m, IVZ, kk, jj, ii);
          Real usq = adm.g_dd(m,0,0,kk,jj,ii)*u0*u0 + adm.g_dd(m,1,1,kk,jj,ii)*u1*u1
                   + adm.g_dd(m,2,2,kk,jj,ii)*u2*u2
                   + 2.0*(adm.g_dd(m,0,1,kk,jj,ii)*u0*u1 + adm.g_dd(m,0,2,kk,jj,ii)*u0*u2
                          + adm.g_dd(m,1,2,kk,jj,ii)*u1*u2);
          Real ua = (a == 0) ? u0 : ((a == 1) ? u1 : u2);
          return ua/sqrt(1.0 + usq);
        };
        auto lnsg = [&](int kk, int jj, int ii) {      // ln sqrt(gamma)
          return 0.5*log(adm::SpatialDet(adm.g_dd(m,0,0,kk,jj,ii), adm.g_dd(m,0,1,kk,jj,ii),
                                         adm.g_dd(m,0,2,kk,jj,ii), adm.g_dd(m,1,1,kk,jj,ii),
                                         adm.g_dd(m,1,2,kk,jj,ii), adm.g_dd(m,2,2,kk,jj,ii)));
        };
        Real p0 = prim(m, IPR, k, j, i);
        Real divv = 0.0, dpmax = 0.0, psip = 0.0, psiv = 0.0;
        for (int a = 0; a < 3; ++a) {
          int ip = i, im = i, jp = j, jm = j, kp = k, km = k;
          Real dx;
          if (a == 0) { ip = (i < iu) ? i+1 : i; im = (i > il) ? i-1 : i;
                        dx = size.d_view(m).dx1*(ip - im); }
          else if (a == 1) { jp = (j < ju) ? j+1 : j; jm = (j > jl) ? j-1 : j;
                             dx = size.d_view(m).dx2*(jp - jm); }
          else { kp = (k < ku) ? k+1 : k; km = (k > kl) ? k-1 : k;
                 dx = size.d_view(m).dx3*(kp - km); }
          if (dx <= 0.0) continue;
          divv += (vel(a, kp, jp, ip) - vel(a, km, jm, im))/dx;
          if (divcov) divv += vel(a, k, j, i)*(lnsg(kp, jp, ip) - lnsg(km, jm, im))/dx;
          Real pp = prim(m, IPR, kp, jp, ip), pm = prim(m, IPR, km, jm, im);
          dpmax = fmax(dpmax, fabs(pp - p0)/fmax(fmin(pp, p0), 1.0e-300));
          dpmax = fmax(dpmax, fabs(pm - p0)/fmax(fmin(pm, p0), 1.0e-300));
          // second differences only where both neighbours exist (a clamped edge would
          // turn them into one-sided first differences)
          if ((ip - im) + (jp - jm) + (kp - km) == 2) {
            psip = fmax(psip, fabs(pp - 2.0*p0 + pm)/fmax(pp + 2.0*p0 + pm, 1.0e-300));
            psiv = fmax(psiv, fabs(vel(a, kp, jp, ip) - 2.0*vel(a, k, j, i)
                                   + vel(a, km, jm, im)));
          }
        }
        // <= so a discontinuity between states at rest (t = 0 of a Riemann problem)
        // counts as a shock: with < it ran entropy-conserving and never formed one (V1).
        // 2 = flagged; 1 = weak candidate (sensor > wk x threshold), joins a flagged
        // face neighbour in the dilation pass; 0 = not flagged
        bool sh, weak;
        if (sensor == 0) {
          sh = dpmax > dpsh; weak = dpmax > wk*dpsh;
        } else if (sensor == 1) {
          sh = psip > psish; weak = psip > wk*psish;
        } else {
          Real rho0 = prim(m, IDN, k, j, i);
          Real h0 = 1.0 + gam_/(gam_ - 1.0)*p0/fmax(rho0, 1.0e-300);
          Real cs0 = sqrt(fmax(gam_*p0/(fmax(rho0, 1.0e-300)*h0), 1.0e-300));
          sh = (psiv > vcsh*cs0) || (psip > psijp);
          weak = (psiv > wk*vcsh*cs0) || (psip > wk*psijp);
        }
        shk(m, k, j, i) = (divv <= 0.0) ? (sh ? 2 : (weak ? 1 : 0)) : 0;
        if (ddon_) {
          Real ratio;
          if (sensor == 0) {
            ratio = dpmax/dpsh;
          } else if (sensor == 1) {
            ratio = psip/psish;
          } else {
            Real rho0 = prim(m, IDN, k, j, i);
            Real h0 = 1.0 + gam_/(gam_ - 1.0)*p0/fmax(rho0, 1.0e-300);
            Real cs0 = sqrt(fmax(gam_*p0/(fmax(rho0, 1.0e-300)*h0), 1.0e-300));
            ratio = fmax(psiv/(vcsh*cs0), psip/psijp);
          }
          dd_(m, 1, k, j, i) = ratio;
        }
      });
      // optional isolation filter (de_isolated): demote a flagged cell (2) with no flagged
      // face neighbour to a weak candidate (1), so the dilation cannot re-flag it either
      if (de_isolated) {
        if (de_shock2.extent_int(0) != de_shock.extent_int(0) ||
            de_shock2.extent_int(1) != e3 || de_shock2.extent_int(2) != e2 ||
            de_shock2.extent_int(3) != e1) {
          Kokkos::realloc(de_shock2, de_shock.extent_int(0), e3, e2, e1);
        }
        auto &sh2 = de_shock2;
        par_for("de_shock_isolated", DevExeSpace(), 0, nmb-1, kl, ku, jl, ju, il, iu,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
          int s0 = shk(m, k, j, i), nb = 0;
          if (i > il) nb = (shk(m, k, j, i-1) == 2) ? 1 : nb;
          if (i < iu) nb = (shk(m, k, j, i+1) == 2) ? 1 : nb;
          if (j > jl) nb = (shk(m, k, j-1, i) == 2) ? 1 : nb;
          if (j < ju) nb = (shk(m, k, j+1, i) == 2) ? 1 : nb;
          if (k > kl) nb = (shk(m, k-1, j, i) == 2) ? 1 : nb;
          if (k < ku) nb = (shk(m, k+1, j, i) == 2) ? 1 : nb;
          sh2(m, k, j, i) = (s0 == 2 && !nb) ? 1 : s0;
        });
        Kokkos::deep_copy(DevExeSpace(), de_shock, de_shock2);
      }
      // optional dilation: a shock is smeared over ~3 cells but the sensor peaks on the
      // central ones (item 72), so also put the face neighbours on the energy branch
      for (int pass = 0; pass < de_dilate; ++pass) {
        if (de_shock2.extent_int(0) != de_shock.extent_int(0) ||
            de_shock2.extent_int(1) != e3 || de_shock2.extent_int(2) != e2 ||
            de_shock2.extent_int(3) != e1) {
          Kokkos::realloc(de_shock2, de_shock.extent_int(0), e3, e2, e1);
        }
        auto &sh2 = de_shock2;
        par_for("de_shock_dilate", DevExeSpace(), 0, nmb-1, kl, ku, jl, ju, il, iu,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
          int s0 = shk(m, k, j, i), nb = 0;
          if (i > il) nb = (shk(m, k, j, i-1) == 2) ? 1 : nb;
          if (i < iu) nb = (shk(m, k, j, i+1) == 2) ? 1 : nb;
          if (j > jl) nb = (shk(m, k, j-1, i) == 2) ? 1 : nb;
          if (j < ju) nb = (shk(m, k, j+1, i) == 2) ? 1 : nb;
          if (k > kl) nb = (shk(m, k-1, j, i) == 2) ? 1 : nb;
          if (k < ku) nb = (shk(m, k+1, j, i) == 2) ? 1 : nb;
          sh2(m, k, j, i) = (s0 == 2 || (s0 == 1 && nb)) ? 2 : s0;
        });
        Kokkos::deep_copy(DevExeSpace(), de_shock, de_shock2);
      }
      // final flag: 1 for shocked (2), else 0; with de_persist > 1 only once the cell has
      // been flagged in de_persist consecutive calls
      const int npers = de_persist;
      if (npers > 1 && (de_streak.extent_int(0) != de_shock.extent_int(0) ||
                        de_streak.extent_int(1) != e3 || de_streak.extent_int(2) != e2 ||
                        de_streak.extent_int(3) != e1)) {
        Kokkos::realloc(de_streak, de_shock.extent_int(0), e3, e2, e1);
      }
      auto &stk = de_streak;
      par_for("de_shock_final", DevExeSpace(), 0, nmb-1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        int f = (shk(m, k, j, i) == 2) ? 1 : 0;
        if (npers > 1) {
          int n = f ? stk(m, k, j, i) + 1 : 0;
          stk(m, k, j, i) = (n < npers) ? n : npers;
          f = (n >= npers) ? 1 : 0;
        }
        shk(m, k, j, i) = f;
      });
    }
    auto &de_shock_ = de_shock;
    const bool atm_fv_ = atm_frame_vel && !floors_only;
    const Real fv0 = pmy_pack->padm->frame_vel_u[0], fv1 = pmy_pack->padm->frame_vel_u[1];
    const Real fv2 = pmy_pack->padm->frame_vel_u[2];
    const Real fg2_ = atm_frame_gate*atm_frame_gate;
    const Real fgx_ = pmy_pack->padm->bc_center[0], fgy_ = pmy_pack->padm->bc_center[1];
    const Real fgz_ = pmy_pack->padm->bc_center[2];
    const bool zon_ = zone_on && !floors_only;
    const Real zr2_ = zone_r*zone_r, zthr_ = zone_rho, zrst_ = zone_rho_reset;
    const Real zvmax_ = zone_vmax, zde_ = zone_de;
    auto &zacc_ = zone_acc;
    const bool zbook_ = zone_book && !floors_only;
    // only once MassLedgerPre has sized ml_cls for this pack (C2P also runs outside the
    // RK stages, e.g. at restart/regrid initialisation, before any stage)
    const bool mlon_ = ml_on && !floors_only &&
                       static_cast<int>(ml_cls.extent(0)) >= pmy_pack->nmb_thispack;
    auto &mlcls_ = ml_cls;
    const bool t3a_ = zon_ && t3_on && t3_active;
    const Real t3rho_ = t3_rho, t3c0_ = t3_c[0], t3c1_ = t3_c[1], t3c2_ = t3_c[2];
    const Real t3ex2_ = 9.0*t3_r90*t3_r90;

    int count_errs=0;
    // Excised-mass tally totals for THIS call, reduced deterministically alongside
    // count_errs (see coordinates.hpp's excised_tally comment) rather than scattered
    // via Kokkos::atomic_add into a per-MeshBlock array.
    Real dE_tally = 0.0, dPx_tally = 0.0, dPy_tally = 0.0, dPz_tally = 0.0;
    Real dEraw_tally = 0.0;
    Real de_nden = 0.0, de_nent_c = 0.0, de_dE_c = 0.0;
    Kokkos::parallel_reduce("pshyd_c2p",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, int &sumerrs, Real &dE_r, Real &dPx_r, Real &dPy_r,
                  Real &dPz_r, Real &dEraw_r, Real &nden_r, Real &nent_r, Real &dede_r) {
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/ni;
      int i = (idx - m*nkji - k*nji - j*ni) + il;
      j += jl;
      k += kl;

      // Add in a short circuit where FOFC is guaranteed.
      if (floors_only && fofc_(m, k, j, i)) {
        return;
      }
      if (floors_only && excise) {
        if (excision_flux_(m,k,j,i)) {
          return;
        }
      }

      // Extract the metric
      Real g3d[NSPMETRIC], g3u[NSPMETRIC], detg, sdetg;
      g3d[S11] = adm.g_dd(m, 0, 0, k, j, i);
      g3d[S12] = adm.g_dd(m, 0, 1, k, j, i);
      g3d[S13] = adm.g_dd(m, 0, 2, k, j, i);
      g3d[S22] = adm.g_dd(m, 1, 1, k, j, i);
      g3d[S23] = adm.g_dd(m, 1, 2, k, j, i);
      g3d[S33] = adm.g_dd(m, 2, 2, k, j, i);
      detg = Primitive::GetDeterminant(g3d);
      sdetg = sqrt(detg);
      Real isdetg = 1.0/sdetg;
      adm::SpatialInv(1.0/detg,
                  g3d[S11], g3d[S12], g3d[S13], g3d[S22], g3d[S23], g3d[S33],
                 &g3u[S11], &g3u[S12], &g3u[S13], &g3u[S22], &g3u[S23], &g3u[S33]);

      // Extract the conserved variables
      Real cons_pt[NCONS], cons_pt_old[NCONS], prim_pt[NPRIM];
      cons_pt[CDN] = cons_pt_old[CDN] = cons(m, IDN, k, j, i)*isdetg;
      cons_pt[CSX] = cons_pt_old[CSX] = cons(m, IM1, k, j, i)*isdetg;
      cons_pt[CSY] = cons_pt_old[CSY] = cons(m, IM2, k, j, i)*isdetg;
      cons_pt[CSZ] = cons_pt_old[CSZ] = cons(m, IM3, k, j, i)*isdetg;
      cons_pt[CTA] = cons_pt_old[CTA] = cons(m, IEN, k, j, i)*isdetg;
      for (int n = 0; n < nscal; n++) {
        cons_pt[CYD + n] = cons_pt_old[CYD + n] = cons(m, nhyd + n, k, j, i)*isdetg;
      }
      // If we're only testing the floors, we can use the CC fields.
      Real b3u[NMAG];
      if (floors_only) {
        b3u[IBX] = bcc0(m, IBX, k, j, i)*isdetg;
        b3u[IBY] = bcc0(m, IBY, k, j, i)*isdetg;
        b3u[IBZ] = bcc0(m, IBZ, k, j, i)*isdetg;
      } else {
        // Otherwise we don't have the correct CC fields yet, so use
        // the FC fields.
        bcc0(m, IBX, k, j, i) = 0.5*(bfc.x1f(m,k,j,i) + bfc.x1f(m,k,j,i+1));
        bcc0(m, IBY, k, j, i) = 0.5*(bfc.x2f(m,k,j,i) + bfc.x2f(m,k,j+1,i));
        bcc0(m, IBZ, k, j, i) = 0.5*(bfc.x3f(m,k,j,i) + bfc.x3f(m,k+1,j,i));
        b3u[IBX] = bcc0(m, IBX, k, j, i)*isdetg;
        b3u[IBY] = bcc0(m, IBY, k, j, i)*isdetg;
        b3u[IBZ] = bcc0(m, IBZ, k, j, i)*isdetg;
      }

      // If we're in an excised region, set the primitives to some default value.
      Primitive::SolverResult result;
      if (excise) {
        // If smooth excision is enabled, do C2P everywhere.
        if (smoothing) {
          result = ps_.ConToPrim(prim_pt, cons_pt, b3u, g3d, g3u);
        } else {
          if (excision_floor_(m,k,j,i)) {
            prim_pt[PRH] = dexcise_/mb;
            prim_pt[PVX] = 0.0;
            prim_pt[PVY] = 0.0;
            prim_pt[PVZ] = 0.0;
            for (int n = 0; n < nscal; n++) {
              // FIXME: Particle abundances should probably be set to a
              // default inside an excised region.
              prim_pt[PYF + n] = cons_pt[CYD]/cons_pt[CDN];
            }
            prim_pt[PPR] = eos_.GetPressure(prim_pt[PRH], texcise_, &prim_pt[PYF]);
            prim_pt[PTM] = texcise_;
            result.error = Primitive::Error::SUCCESS;
            result.iterations = 0;
            result.cons_floor = false;
            result.prim_floor = false;
            result.cons_adjusted = true;
            ps_.PrimToCon(prim_pt, cons_pt, b3u, g3d);

            // Bank what this reset just removed, so CFC can hand it to the puncture.
            // Without this the hole loses the mass it swallowed: CFC's metric comes
            // from elliptic constraints on the instantaneous matter distribution, so
            // deleting matter deletes its gravity in the same step.
            //
            // Interior cells only.  This kernel is also run over ghost zones (the
            // caller sweeps 0..n1m1) and re-entered by ConToPrimBC on boundary
            // strips, so an unguarded sum would multiply-count every excised cell.
            // Also skipped for floors_only, where FOFC probes the state and writes
            // nothing back.
            if (!floors_only_ && i >= is && i <= ie_ && j >= js && j <= je_ &&
                k >= ks && k <= ke_) {
              // psi from sqrt(det g) == psi^6 (conformal flatness).  The energy is
              // weighted by 1/psi to convert the densitized conserved energy
              // psi^6*E into the ADM-mass contribution psi^5*E; the momenta are
              // NOT, since psi^6*S_i is already the momentum-constraint source.
              Real psi_c = Kokkos::pow(sdetg, 1.0/6.0);
              Real dvol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
              Real dE = ((cons_pt_old[CTA] + cons_pt_old[CDN])
                        - (cons_pt[CTA] + cons_pt[CDN]))*sdetg*dvol;
              // Contributed to this call's own reduction (deterministic), not
              // atomic-added into a shared accumulator -- see coordinates.hpp.
              dE_r   += dE/psi_c;
              dPx_r  += (cons_pt_old[CSX] - cons_pt[CSX])*sdetg*dvol;
              dPy_r  += (cons_pt_old[CSY] - cons_pt[CSY])*sdetg*dvol;
              dPz_r  += (cons_pt_old[CSZ] - cons_pt[CSZ])*sdetg*dvol;
              dEraw_r += dE;   // raw psi^6 E, validation only
              if (zbook_) {    // rest mass, for the zone booking's closure
                Kokkos::atomic_add(&zacc_(kZoneExcM), (cons_pt_old[CDN] - cons_pt[CDN])*sdetg*dvol);
              }
            }
          } else {
            result = ps_.ConToPrim(prim_pt, cons_pt, b3u, g3d, g3u);
          }
        }
      } else {
        result = ps_.ConToPrim(prim_pt, cons_pt, b3u, g3d, g3u);
      }

      if (result.error != Primitive::Error::SUCCESS && floors_only) {
        fofc_(m,k,j,i) = true;
      } else if (!floors_only) {
        if (result.error != Primitive::Error::SUCCESS && (nerrs_ + sumerrs < errcap_)) {
          sumerrs++;
          // Find out where the point went bad and report a bunch of information about it.
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

          Kokkos::printf("An error occurred during the primitive solve: %s\n"
                 "  Location: (%d, %d, %d, %d)\n"
                 "            (%.17g, %.17g, %.17g)\n"
                 "  Conserved vars: \n"
                 "    D   = %.17g\n"
                 "    Sx  = %.17g\n"
                 "    Sy  = %.17g\n"
                 "    Sz  = %.17g\n"
                 "    tau = %.17g\n"
                 "    Dye = %.17g\n"
                 "    Bx  = %.17g\n"
                 "    By  = %.17g\n"
                 "    Bz  = %.17g\n"
                 "  Metric vars: \n"
                 "    detg = %.17g\n"
                 "    g_dd = {%.17g, %.17g, %.17g, %.17g, %.17g, %.17g}\n"
                 "    alp  = %.17g\n"
                 "    beta = {%.17g, %.17g, %.17g}\n"
                 "    psi4 = %.17g\n"
                 "    K_dd = {%.17g, %.17g, %.17g, %.17g, %.17g, %.17g}\n",
                 ErrorToString(result.error),
                 m, k, j, i,
                 x1v, x2v, x3v,
                 cons_pt_old[CDN], cons_pt_old[CSX], cons_pt_old[CSY], cons_pt_old[CSZ],
                 cons_pt_old[CTA], cons_pt_old[CYD], b3u[IBX], b3u[IBY], b3u[IBZ], detg,
                 g3d[S11], g3d[S12], g3d[S13], g3d[S22], g3d[S23], g3d[S33],
                 adm.alpha(m, k, j, i),
                 adm.beta_u(m, 0, k, j, i),
                 adm.beta_u(m, 1, k, j, i), adm.beta_u(m, 2, k, j, i),
                 adm.psi4(m, k, j, i),
                 adm.vK_dd(m, 0, 0, k, j, i), adm.vK_dd(m, 0, 1, k, j, i),
                 adm.vK_dd(m, 0, 2, k, j, i),
                 adm.vK_dd(m, 1, 1, k, j, i), adm.vK_dd(m, 1, 2, k, j, i),
                 adm.vK_dd(m, 2, 2, k, j, i));
          if (nerrs_ + sumerrs == errcap_) {
            Kokkos::printf("%d C2P errors have been detected on rank %d."
                   "All future C2P errors\n"
                   "on this rank will be suppressed. Fix your code!\n",
                   nerrs_ + sumerrs,rank);
          }
        }
        // ---- comoving atmosphere (see atm_frame_vel): floored cells move with the frame
        bool de_write = false;
        bool fv_here = atm_fv_ && (result.prim_floor || result.cons_floor) &&
                       (fv0 != 0.0 || fv1 != 0.0 || fv2 != 0.0);
        if (fv_here && fg2_ > 0.0) {
          Real xg = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min,
                                size.d_view(m).x1max) - fgx_;
          Real yg = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min,
                                size.d_view(m).x2max) - fgy_;
          Real zg = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min,
                                size.d_view(m).x3max) - fgz_;
          fv_here = (xg*xg + yg*yg + zg*zg > fg2_);
        }
        if (fv_here) {
          Real vsq = g3d[S11]*fv0*fv0 + g3d[S22]*fv1*fv1 + g3d[S33]*fv2*fv2
                   + 2.0*(g3d[S12]*fv0*fv1 + g3d[S13]*fv0*fv2 + g3d[S23]*fv1*fv2);
          Real wl = 1.0/sqrt(1.0 - fmin(vsq, 0.99));
          prim_pt[PVX] = wl*fv0; prim_pt[PVY] = wl*fv1; prim_pt[PVZ] = wl*fv2;
          ps_.PrimToCon(prim_pt, cons_pt, b3u, g3d);
          de_write = true;
        }

        int zml = 0;   // mass_ledger class 2 (zone / tier 3)
        // ---- near-BH atmosphere zone (see zone_on)
        if (zon_ && !(excise && !smoothing && excision_floor_(m,k,j,i))) {
          const Real xz = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min,
                                      size.d_view(m).x1max) - fgx_;
          const Real yz = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min,
                                      size.d_view(m).x2max) - fgy_;
          const Real zz = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min,
                                      size.d_view(m).x3max) - fgz_;
          const Real r2 = xz*xz + yz*yz + zz*zz;
          const bool interior = (i >= is && i <= ie_ && j >= js && j <= je_ &&
                                 k >= ks && k <= ke_);
          const Real rho0 = prim_pt[PRH]*mb;
          if (r2 < zr2_ && interior) Kokkos::atomic_max(&zacc_(kZoneRreg), rho0);
          // mode 1: the zone (rho reset, mass removed); mode 2: tier 3 (velocity only)
          int zmode = (r2 < zr2_ && rho0 < zthr_) ? 1 : 0;
          if (!zmode && t3a_ && r2 >= zr2_ && r2 < 4.0*zr2_ && rho0 < t3rho_) {
            const Real cx = xz + fgx_ - t3c0_, cy = yz + fgy_ - t3c1_, cz = zz + fgz_ - t3c2_;
            if (cx*cx + cy*cy + cz*cz > t3ex2_) zmode = 2;
          }
          zml = zmode;
          if (zmode) {
            const int zb = (zmode == 1) ? 0 : kZoneLen;     // tally block
            const Real al = adm.alpha(m, k, j, i);
            const Real bu[3] = {adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
                                adm.beta_u(m, 2, k, j, i)};
            // the state before the reset: BH-frame Killing E and |L| (CFC::MeasureBound)
            const Real uu[3] = {prim_pt[PVX], prim_pt[PVY], prim_pt[PVZ]};
            Real ud[3];
            Primitive::LowerVector(ud, uu, g3d);
            const Real W0 = sqrt(1.0 + Primitive::Contract(uu, ud));
            const Real ekill = al*W0 - ((bu[0] - fv0)*ud[0] + (bu[1] - fv1)*ud[1] +
                                        (bu[2] - fv2)*ud[2]) - 1.0;
            const Real lx = yz*ud[2] - zz*ud[1], ly = zz*ud[0] - xz*ud[2];
            const Real lz = xz*ud[1] - yz*ud[0];
            const Real lmag = sqrt(lx*lx + ly*ly + lz*lz);
            // grid rest v = beta'/alpha, capped at zvmax along beta'
            Real bd[3];
            Primitive::LowerVector(bd, bu, g3d);
            const Real vb = sqrt(fmax(Primitive::Contract(bu, bd), 0.0))/fmax(al, 1.0e-300);
            const Real f = (vb > zvmax_) ? zvmax_/vb : 1.0;
            const Real vsq = f*f*vb*vb;
            const Real wl = 1.0/sqrt(1.0 - vsq);
            Real cpre[NCONS];
            for (int n = 0; n < NCONS; ++n) cpre[n] = cons_pt[n];
            if (zmode == 1) prim_pt[PRH] = fmin(rho0, zrst_)/mb;
            prim_pt[PVX] = wl*f*bu[0]/fmax(al, 1.0e-300);
            prim_pt[PVY] = wl*f*bu[1]/fmax(al, 1.0e-300);
            prim_pt[PVZ] = wl*f*bu[2]/fmax(al, 1.0e-300);
            prim_pt[PTM] = eos_.GetTemperatureFloor();
            prim_pt[PPR] = eos_.GetPressure(prim_pt[PRH], prim_pt[PTM], &prim_pt[PYF]);
            ps_.PrimToCon(prim_pt, cons_pt, b3u, g3d);
            de_write = true;
            if (interior) {
              const Real dv = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3*sdetg;
              // zone: removed mass; tier 3: the modified cell's mass (none is removed)
              const Real dM = (zmode == 1) ? (cpre[CDN] - cons_pt[CDN])*dv : cpre[CDN]*dv;
              const Real ee = ekill/zde_;
              int ie = (ee >= -6.0) + (ee >= -3.0) + (ee >= -2.0) + (ee >= -1.0) + (ee >= 0.0)
                       + (ee >= 1.0) + (ee >= 2.0) + (ee >= 3.0) + (ee >= 6.0);
              int il = (lmag >= 2.0) + (lmag >= 4.0) + (lmag >= 6.0) + (lmag >= 10.0);
              int ir = (r2 >= 1.88*1.88) + (r2 >= 2.92*2.92);
              Kokkos::atomic_add(&zacc_(zb + (ir*kZoneNE + ie)*kZoneNL + il), dM);
              Kokkos::atomic_add(&zacc_(zb + kZoneSx), (cpre[CSX] - cons_pt[CSX])*dv);
              Kokkos::atomic_add(&zacc_(zb + kZoneSx + 1), (cpre[CSY] - cons_pt[CSY])*dv);
              Kokkos::atomic_add(&zacc_(zb + kZoneSx + 2), (cpre[CSZ] - cons_pt[CSZ])*dv);
              Kokkos::atomic_add(&zacc_(zb + kZoneEn), (cpre[CTA] + cpre[CDN] - cons_pt[CTA]
                                                   - cons_pt[CDN])*dv);
              Kokkos::atomic_add(&zacc_(zb + kZoneM), dM);
              Kokkos::atomic_add(&zacc_(zb + kZoneN), 1.0);
              if (f < 1.0) Kokkos::atomic_add(&zacc_(zb + kZoneNcap), 1.0);
              Kokkos::atomic_max(&zacc_(zb + kZoneRmod), rho0);
              if (rho0 >= ((zmode == 1) ? zthr_ : t3rho_))
                Kokkos::atomic_add(&zacc_(zb + kZoneViol), 1.0);
            }
          }
        }

        // ---- dual energy (see de_on): choose the energy or the entropy solution.
        if (de_on_) {
          const bool interior = (i >= is && i <= ie_ && j >= js && j <= je_ &&
                                 k >= ks && k <= ke_);
          const Real D = cons_pt_old[CDN];
          const bool dense = D > de_rho_sw_;
          const bool bzero = (b3u[IBX] == 0.0 && b3u[IBY] == 0.0 && b3u[IBZ] == 0.0);
          const bool bad = result.prim_floor ||
                           (result.error != Primitive::Error::SUCCESS);
          bool use_ent = dense && bzero && (de_shock_(m,k,j,i) == 0 || bad);
          int dbr = !dense ? 0 : (use_ent ? (bad ? 3 : 1) : 4);
          Real dken = 0.0, ddk = 0.0, ddtau = 0.0;
          if (ddon_ && dense && bzero && !bad) {
            Real rho_e = prim_pt[PRH]*mb;
            Real k_tr = DualEnergyK(cons_pt_old[CYD + de_idx_]/D, de_kmax_, de_ypow_);
            dken = prim_pt[PPR]/pow(rho_e, de_gamma_)/fmax(k_tr, 1.0e-300) - 1.0;
          }
          if (use_ent && !bad && de_eta1_ > 0.0) {
            // optional energy-ratio gate: entropy only where eps is a small part of tau
            Real eps = prim_pt[PPR]/((de_gamma_ - 1.0)*prim_pt[PRH]*mb);
            if (eps > de_eta1_*cons_pt_old[CTA]/D) use_ent = false;
          }
          if (!use_ent && de_2law_ && dense && bzero && !bad) {
            // second-law guard: the energy solution may raise K (a shock), never lower it
            Real rho_e = prim_pt[PRH]*mb;
            Real k_en = prim_pt[PPR]/pow(rho_e, de_gamma_);
            Real k_tr = DualEnergyK(cons_pt_old[CYD + de_idx_]/D, de_kmax_, de_ypow_);
            if (k_en <= k_tr*(1.0 + de_2tol_)) { use_ent = true; dbr = 2; }
          }
          if (use_ent) {
            Real K = DualEnergyK(cons_pt_old[CYD + de_idx_]/D, de_kmax_, de_ypow_);
            Real S_d[3] = {cons_pt_old[CSX], cons_pt_old[CSY], cons_pt_old[CSZ]};
            Real S_u[3] = {g3u[S11]*S_d[0] + g3u[S12]*S_d[1] + g3u[S13]*S_d[2],
                           g3u[S12]*S_d[0] + g3u[S22]*S_d[1] + g3u[S23]*S_d[2],
                           g3u[S13]*S_d[0] + g3u[S23]*S_d[1] + g3u[S33]*S_d[2]};
            Real ssq = S_u[0]*S_d[0] + S_u[1]*S_d[1] + S_u[2]*S_d[2];
            Real W, h;
            if (EntropyInversion(D, ssq, K, de_gamma_, W, h)) {
              Real rho_ = D/W;
              Real P_ = K*pow(rho_, de_gamma_);
              for (int n = 0; n < NCONS; ++n) { cons_pt[n] = cons_pt_old[n]; }
              prim_pt[PRH] = rho_/mb;
              prim_pt[PPR] = P_;
              prim_pt[PVX] = S_u[0]/(D*h);
              prim_pt[PVY] = S_u[1]/(D*h);
              prim_pt[PVZ] = S_u[2]/(D*h);
              for (int n = 0; n < nscal; n++) {
                prim_pt[PYF + n] = cons_pt_old[CYD + n]/D;
              }
              prim_pt[PTM] = eos_.GetTemperatureFromP(prim_pt[PRH], P_, &prim_pt[PYF]);
              Real tau_new = D*h*W - P_ - D;
              if (interior) {
                Real dvol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
                dede_r += (tau_new - cons_pt_old[CTA])*sdetg*dvol;
                nent_r += 1.0;
              }
              ddtau = (tau_new - cons_pt_old[CTA])/D;
              cons_pt[CTA] = tau_new;
              de_write = true;
            } else {
              use_ent = false;
              dbr = dense ? 4 : 0;
            }
          }
          if (!use_ent && !(dense && bad)) {
            // resync K from the energy solution (never from a floored dense state)
            Real rho_ = prim_pt[PRH]*mb;
            Real yk = (rho_ > 0.0) ?
                      DualEnergyY(prim_pt[PPR]/pow(rho_, de_gamma_), de_kmax_, de_ypow_) : 0.0;
            if (dense) { dbr = 4; }   // also covers the de_eta1 gate
            if (ddon_ && dense) {
              Real k_tr = DualEnergyK(cons_pt_old[CYD + de_idx_]/D, de_kmax_, de_ypow_);
              ddk = DualEnergyK(yk, de_kmax_, de_ypow_)/fmax(k_tr, 1.0e-300) - 1.0;
            }
            prim_pt[PYF + de_idx_] = yk;
            cons_pt[CYD + de_idx_] = cons_pt[CDN]*yk;
            de_write = true;
          } else if (!use_ent && dense) {
            dbr = 5;
          }
          if (interior && dense) { nden_r += 1.0; }
          if (ddon_) {
            dd_(m, 0, k, j, i) = dbr;
            dd_(m, 2, k, j, i) = de_shock_(m, k, j, i);
            dd_(m, 3, k, j, i) = dken;
            dd_(m, 4, k, j, i) = ddk;
            dd_(m, 5, k, j, i) = ddtau;
            if (dstg_ >= 1 && dstg_ <= 3) dd_(m, 9 + dstg_, k, j, i) = dken;
          }
        }

        // Regardless of failure, we need to copy the primitives.
        prim(m, IDN, k, j, i) = prim_pt[PRH]*mb;
        prim(m, IVX, k, j, i) = prim_pt[PVX];
        prim(m, IVY, k, j, i) = prim_pt[PVY];
        prim(m, IVZ, k, j, i) = prim_pt[PVZ];
        prim(m, IPR, k, j, i) = prim_pt[PPR];
        for (int n = 0; n < nscal; n++) {
          prim(m, nhyd + n, k, j, i) = prim_pt[PYF + n];
        }

        temperature(m,0,k,j,i) = prim_pt[PTM];

        // If the conservative variables were floored or adjusted for consistency,
        // we need to copy the conserved variables, too.
        if (result.cons_floor || result.cons_adjusted || de_write) {
          /*if (fabs((cons_pt[CDN] - cons_pt_old[CDN])/cons_pt_old[CDN]) > 1e-12) {
            Real &x1min = size.d_view(m).x1min;
            Real &x1max = size.d_view(m).x1max;
            Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

            Real &x2min = size.d_view(m).x2min;
            Real &x2max = size.d_view(m).x2max;
            Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

            Real &x3min = size.d_view(m).x3min;
            Real &x3max = size.d_view(m).x3max;
            Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
            bool is_ghost = (i < is) || (i > ie) ||
                            (j < js) || (j > je) ||
                            (k < ks) || (k > ke);

            printf("Density was nontrivially adjusted on MeshBlock %d!\n"
                   "  Grid index: (i=%d, j=%d, k=%d)\n"
                   "  Physical position: (%g, %g, %g)\n"
                   "  D (old): %.17g\n"
                   "  D (new): %.17g\n"
                   "  Ghost zone? %s\n",
                   m, i, j, k,
                   x1v, x2v, x3v, cons_pt_old[CDN], cons_pt[CDN],
                   is_ghost ? "true" : "false");
          }*/
          cons(m, IDN, k, j, i) = cons_pt[CDN]*sdetg;
          cons(m, IM1, k, j, i) = cons_pt[CSX]*sdetg;
          cons(m, IM2, k, j, i) = cons_pt[CSY]*sdetg;
          cons(m, IM3, k, j, i) = cons_pt[CSZ]*sdetg;
          cons(m, IEN, k, j, i) = cons_pt[CTA]*sdetg;
          for (int n = 0; n < nscal; n++) {
            cons(m, nhyd + n, k, j, i) = cons_pt[CYD + n]*sdetg;
          }
        }
        if (mlon_ && i >= is && i <= ie_ && j >= js && j <= je_ && k >= ks && k <= ke_) {
          Real c = 4.0;
          if (excise && !smoothing && excision_floor_(m,k,j,i)) {
            c = excision_flux_(m,k,j,i) ? 0.0 : 1.0;
          } else if (zml) {
            c = 2.0;
          } else if (result.prim_floor || result.cons_floor) {
            c = 3.0;
          }
          mlcls_(m,k,j,i) = c;
        }
      }
    }, Kokkos::Sum<int>(count_errs), Kokkos::Sum<Real>(dE_tally), Kokkos::Sum<Real>(dPx_tally),
       Kokkos::Sum<Real>(dPy_tally), Kokkos::Sum<Real>(dPz_tally),
       Kokkos::Sum<Real>(dEraw_tally), Kokkos::Sum<Real>(de_nden),
       Kokkos::Sum<Real>(de_nent_c), Kokkos::Sum<Real>(de_dE_c));
    if (de_on_) {
      de_dE_total += de_dE_c;
      if (il == 0 && jl == 0 && kl == 0) {   // the full-array call, not a BC strip
        de_ndense = de_nden; de_nent = de_nent_c; de_dE = de_dE_c;
      }
    }

    if (floors_only) {
      ps.GetEOSMutable().SetPrimitiveFloorFailure(prim_failure);
      ps.GetEOSMutable().SetConservedFloorFailure(cons_failure);
    } else {
      nerrs += count_errs;
      // Ordinary (non-atomic) accumulation on this rank's single controlling host
      // thread: deterministic, since each call's own reduction above is deterministic
      // and this simply adds that one already-combined result once, in the fixed
      // order calls happen in. See coordinates.hpp's excised_tally comment.
      auto &tally = pmy_pack->pcoord->excised_tally;
      tally[0] += dE_tally;
      tally[1] += dPx_tally;
      tally[2] += dPy_tally;
      tally[3] += dPz_tally;
      tally[4] += dEraw_tally;
    }
  }

  //! V0 self-test of the dual-energy entropy inversion on real data: for every dense
  //! interior cell, recover (rho, P, Wv) from (D, S_i, K) of the just-built conserved
  //! state and return the max relative error against the primitives (0 if off).
  Real DualEnergySelfTest(const DvceArray5D<Real> &cons, const DvceArray5D<Real> &prim) {
    if (!de_on) return 0.0;
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
    int ks = indcs.ks, ke = indcs.ke;
    int nmb = pmy_pack->nmb_thispack;
    auto &adm = pmy_pack->padm->adm;
    int nhyd = pmy_pack->pmhd->nmhd;
    const int idx_ = de_idx;
    const Real kmax_ = de_kmax, rsw_ = de_rho_sw, gam_ = de_gamma, ypow_ = de_ypow;
    Real errmax = 0.0;
    const int ni = ie - is + 1, nji = (je - js + 1)*ni, nkji = (ke - ks + 1)*nji;
    Kokkos::parallel_reduce("de_selftest", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*nkji),
    KOKKOS_LAMBDA(const int &n, Real &emax) {
      int m = n/nkji, k = (n - m*nkji)/nji, j = (n - m*nkji - k*nji)/ni;
      int i = n - m*nkji - k*nji - j*ni + is; j += js; k += ks;
      Real g3d[NSPMETRIC], g3u[NSPMETRIC];
      g3d[S11] = adm.g_dd(m,0,0,k,j,i); g3d[S12] = adm.g_dd(m,0,1,k,j,i);
      g3d[S13] = adm.g_dd(m,0,2,k,j,i); g3d[S22] = adm.g_dd(m,1,1,k,j,i);
      g3d[S23] = adm.g_dd(m,1,2,k,j,i); g3d[S33] = adm.g_dd(m,2,2,k,j,i);
      Real detg = Primitive::GetDeterminant(g3d), sdetg = sqrt(detg);
      adm::SpatialInv(1.0/detg, g3d[S11], g3d[S12], g3d[S13], g3d[S22], g3d[S23],
                      g3d[S33], &g3u[S11], &g3u[S12], &g3u[S13], &g3u[S22], &g3u[S23],
                      &g3u[S33]);
      Real D = cons(m, IDN, k, j, i)/sdetg;
      if (!(D > rsw_)) return;
      Real S_d[3] = {cons(m, IM1, k, j, i)/sdetg, cons(m, IM2, k, j, i)/sdetg,
                     cons(m, IM3, k, j, i)/sdetg};
      Real S_u[3] = {g3u[S11]*S_d[0] + g3u[S12]*S_d[1] + g3u[S13]*S_d[2],
                     g3u[S12]*S_d[0] + g3u[S22]*S_d[1] + g3u[S23]*S_d[2],
                     g3u[S13]*S_d[0] + g3u[S23]*S_d[1] + g3u[S33]*S_d[2]};
      Real ssq = S_u[0]*S_d[0] + S_u[1]*S_d[1] + S_u[2]*S_d[2];
      Real K = DualEnergyK(cons(m, nhyd + idx_, k, j, i)/sdetg/D, kmax_, ypow_);
      Real W, h;
      if (!EntropyInversion(D, ssq, K, gam_, W, h)) { emax = fmax(emax, 1.0); return; }
      Real rho = D/W, P = K*pow(rho, gam_);
      Real e = fabs(rho/prim(m, IDN, k, j, i) - 1.0);
      e = fmax(e, fabs(P/prim(m, IPR, k, j, i) - 1.0));
      Real u0 = prim(m, IVX, k, j, i), wv = S_u[0]/(D*h);
      e = fmax(e, fabs(wv - u0)/fmax(fabs(u0), 1.0e-3));
      emax = fmax(emax, e);
    }, Kokkos::Max<Real>(errmax));
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &errmax, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
    return errmax;
  }

  // Get the transformed magnetosonic speeds at a point in a given direction.
  KOKKOS_INLINE_FUNCTION
  void GetGRFastMagnetosonicSpeeds(Real& lambda_p, Real& lambda_m,
                                   Real prim[NPRIM], Real bsq, Real g3d[NSPMETRIC],
                                   Real beta_u[3], Real alpha, Real gii,
                                   int pvx) const {
    Real uu[3] = {prim[PVX], prim[PVY], prim[PVZ]};
    Real usq = Primitive::SquareVector(uu, g3d);
    int index = pvx - PVX;

    // Get spacetime quantities
    Real Wsq = 1.0 + usq;
    Real ialpha = 1.0/Primitive::FloorLapse(alpha);
    Real W = sqrt(Wsq);
    Real u0 = W*ialpha;
    Real u1 = uu[index] - u0*beta_u[index];
    Real g00 = -ialpha*ialpha;
    Real g01 = -g00*beta_u[index];
    Real g11 = gii - g01*beta_u[index];

    // Calculate the sound speed and the Alfven speed
    Real cs = ps.GetEOS().GetSoundSpeed(prim[PRH], prim[PTM], &prim[PYF]);
    Real csq = cs*cs;
    Real H = ps.GetEOS().GetBaryonMass()*prim[PRH]*
             ps.GetEOS().GetEnthalpy(prim[PRH], prim[PTM], &prim[PYF]);
    Real vasq = bsq/(bsq + H);
    Real cmsq = csq + vasq - csq*vasq;

    // Set fast magnetosonic speed in appropriate coordinates
    Real a = u0*u0 - (g00 + u0*u0)*cmsq;
    Real b = -2.0 * (u0 * u1 - (g01 + u0 * u1) *cmsq);
    Real c = u1*u1 - (g11 + u1*u1)*cmsq;
    Real a1 = b / a;
    Real a0 = c / a;
    Real s = fmax(a1*a1 - 4.0 * a0, 0.0);
    s = sqrt(s);
    lambda_p = (a1 >= 0.0) ? -2.0 * a0 / (a1 + s) : (-a1 + s) / 2.0;
    lambda_m = (a1 >= 0.0) ? (-a1 - s) / 2.0 : -2.0 * a0 / (a1 - s);
  }

  // A function for converting PrimitiveSolver errors to strings
  KOKKOS_INLINE_FUNCTION
  static const char * ErrorToString(Primitive::Error e) {
    switch(e) {
      case Primitive::Error::SUCCESS:
        return "SUCCESS";
        break;
      case Primitive::Error::RHO_TOO_BIG:
        return "RHO_TOO_BIG";
        break;
      case Primitive::Error::RHO_TOO_SMALL:
        return "RHO_TOO_SMALL";
        break;
      case Primitive::Error::NANS_IN_CONS:
        return "NANS_IN_CONS";
        break;
      case Primitive::Error::MAG_TOO_BIG:
        return "MAG_TOO_BIG";
        break;
      case Primitive::Error::BRACKETING_FAILED:
        return "BRACKETING_FAILED";
        break;
      case Primitive::Error::NO_SOLUTION:
        return "NO_SOLUTION";
        break;
      default:
        return "OTHER";
        break;
    }
  }

  // A function for checking for NaNs in the conserved variables.
  KOKKOS_INLINE_FUNCTION
  static int CheckForConservedNaNs(const Real cons_pt[NCONS]) {
    int nans = 0;
    if (!isfinite(cons_pt[CDN])) {
      Kokkos::printf("D is NaN!\n"); // NOLINT
      nans = 1;
    }
    if (!isfinite(cons_pt[CSX])) {
      Kokkos::printf("Sx is NaN!\n"); // NOLINT
      nans = 1;
    }
    if (!isfinite(cons_pt[CSY])) {
      Kokkos::printf("Sy is NaN!\n"); // NOLINT
      nans = 1;
    }
    if (!isfinite(cons_pt[CSZ])) {
      Kokkos::printf("Sz is NaN!\n"); // NOLINT
      nans = 1;
    }
    if (!isfinite(cons_pt[CTA])) {
      Kokkos::printf("Tau is NaN!\n"); // NOLINT
      nans = 1;
    }

    return nans;
  }

  KOKKOS_INLINE_FUNCTION
  static void DumpPrimitiveVars(const Real prim_pt[NPRIM]) {
    Kokkos::printf("Primitive vars: \n" // NOLINT
           "  rho = %.17g\n"
           "  ux  = %.17g\n"
           "  uy  = %.17g\n"
           "  uz  = %.17g\n"
           "  P   = %.17g\n"
           "  T   = %.17g\n",
           prim_pt[PRH], prim_pt[PVX], prim_pt[PVY],
           prim_pt[PVZ], prim_pt[PPR], prim_pt[PTM]);
  }
};
#endif  // EOS_PRIMITIVE_SOLVER_HYD_HPP_
