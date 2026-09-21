//========================================================================================
// PrimitiveSolver equation-of-state framework
// Copyright(C) 2023 Jacob M. Fields <jmf6719@psu.edu>
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos_zla_bag.cpp
//  \brief Implementation of EOS ZLA Phase Transition

#include <math.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <iostream>
#include <cstddef>
#include <string>

#include <Kokkos_Core.hpp>

#include "../../parameter_input.hpp"
#include "athena.hpp"
#include "eos_zla_bag.hpp"
#include "utils/tr_table.hpp"
#include "logs.hpp"
#include "globals.hpp"

namespace Primitive {

template<typename LogPolicy>
bool EOSZlaBag<LogPolicy>::ReadParametersFromInput(std::string block,
                                                   ParameterInput * pin) {
  Real nscal = pin->GetOrAddInteger(block, "nscalars", 0);
  // Number of scalars has to be 4
  assert(nscal==4);

  // Set Reduce Planck constant
  //h_bar = 1.0545718e-27 * eos_units.EnergyConversion(CGS)
  //                      * eos_units.TimeConversion(CGS);
  // CODATA hbar*c in MeV fm. The previous default, 197.327, is a truncation of this
  // value; because the Fermi pressure scales as hbar^2, the 1.0e-7 relative
  // truncation showed up as a systematic 1.99e-7 relative offset between the analytic
  // cold pressure and the tabulated pressure across the whole table.
  h_bar = pin->GetOrAddReal(block, "h_bar", 197.3269804);
  pi2hbar3 = Kokkos::numbers::pi*Kokkos::numbers::pi * h_bar*h_bar*h_bar;

  ZL_eta = pin->GetOrAddReal(block, "ZL_eta", 1.0);

  // Phase fractions closer than f_snap to 0 or 1 are snapped onto the single-phase
  // limit; see EOSZlaBag::SnapPhaseFraction() for why. The default sits ~3 decades
  // above the round-off band that corrupts the bag term at atmosphere densities and
  // ~6 decades below the smallest genuine mixed-phase fraction in a ZLA table.
  f_snap = pin->GetOrAddReal(block, "f_snap", 1.0e-8);

  // Resolvability tolerance on the nucleon fraction. y_lQ = Y[3]/(1-yn) is a ratio of
  // two independently advected small quantities; below this tolerance the denominator
  // is round-off and the quark phase is treated as absent rather than divided by. Same
  // default as f_snap, which guards the identical hazard for the volume fraction.
  yn_snap = pin->GetOrAddReal(block, "yn_snap", 1.0e-8);

  // MIGRATION GUARD. The advected scalars are the decoupled set
  // (f, Y_N, y_lN, y_lQ); the nested products (f, yn, yn*y_lN, (1-yn)*y_lQ) this EOS
  // once supported are gone, along with the ResetFloorZlaBag cascade limiter that
  // enforced their coupled bands. A parfile written for the nested packing carries no
  // zla_flat_scalars key at all, so silently defaulting would have the pgen write one
  // packing while the EOS reads the other -- a wrong star that still looks physical.
  // Require the key, and require it true. Remove this guard once no nested parfiles
  // remain in circulation.
  if (!pin->GetOrAddBoolean(block, "zla_flat_scalars", false)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "EOSZlaBag advects the decoupled scalar set "
              << "(f, Y_N, y_lN, y_lQ), so <" << block << ">/zla_flat_scalars must be "
              << "set to true." << std::endl
              << "A parfile without it was written for the nested packing "
              << "(f, yn, yn*y_lN, (1-yn)*y_lQ), which this build no longer supports; "
              << "its initial data would be mis-ordered." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Pressure floor on the phase fraction. Quark matter is unbound below the transition
  // density -- ColdPressureQuarks tends to -Bag_B -- so a frozen composition holding a
  // quark phase there gives a negative total pressure. Measured on 400 real failing
  // states from a 600 M TOV run: P < 0 in 400/400. See SnapPhaseFraction() for the
  // mechanism and for why snapping on density instead would be wrong.
  // Default false, so an input that does not mention it is bit-identical to before.
  pfloor_on = pin->GetOrAddBoolean(block, "zla_pfloor", false);
  // Floor expressed as a fraction of the single-phase nucleonic pressure at the same
  // density. 0 requires only P >= 0, which is the physical statement.
  pfloor_frac = pin->GetOrAddReal(block, "zla_pfloor_frac", 0.0);
  // Keep the composition-independent fast path? Default true = historical behaviour.
  // False makes the pressure-floor fixed point the single criterion at every density,
  // removing the jump at n_pfloor_force. See SnapPhaseFraction().
  pfloor_fastpath = pin->GetOrAddBoolean(block, "zla_pfloor_fastpath", true);

  // Bisection switch for the m_min_h = fmin(m_min_h, mb) floor applied after the
  // table scan (see ReadTableFromFile). True reproduces the current behaviour;
  // false restores the pure beta-equilibrium scan that HEAD uses.
  min_h_floor_mb = pin->GetOrAddBoolean(block, "eos_min_h_floor_mb", true);

  // Print every node of the table self-test, not just the summary. Off by default:
  // the full dump is ~53k lines per run for the production table and is only useful
  // when a specific node is under investigation.
  selftest_verbose = pin->GetOrAddBoolean(block, "zla_selftest_verbose", false);

  // Fermi-integral series/closed-form crossover. HEAD used 1.0e-2; 3.0e-2 is the
  // current default. Bisectable because the band between them is the
  // non-relativistic tail, where the failures are.
  x_series_max = pin->GetOrAddReal(block, "x_series_max", 3.0e-2);

  // Temperature ceiling. The constructor leaves max_T at DBL_MAX and the table (which
  // is a 1D cold slice plus a gamma-law thermal part) supplies no upper bound, so
  // ApplyTemperatureLimits has nothing to clamp against: a conserved state implying an
  // arbitrarily large tau/D is accepted rather than limited. Opt in by setting
  // <mhd>/tceiling. The default preserves the previous unbounded behaviour.
  //
  // Note this bounds the temperature, not the energy: RootFunctor's nu_b is built from
  // the conserved q, so fmax(nu_a, nu_b) still tracks the full q. A ceiling makes the
  // recovered state finite and floorable; it does not reject an unphysical q.
  max_T = pin->GetOrAddReal(block, "tceiling",
                            std::numeric_limits<Real>::max());

  m_electron = pin->GetOrAddReal(block, "m_electron", 0.5109989499961642);
  m_muon     = pin->GetOrAddReal(block, "m_muon"    , 105.65837549724458);
  m_u_quark  = pin->GetOrAddReal(block, "m_u_quark" , 5.0);
  m_d_quark  = pin->GetOrAddReal(block, "m_d_quark" , 7.0);
  m_s_quark  = pin->GetOrAddReal(block, "m_s_quark" , 150.0);

  Real dm = Kokkos::sqrt(SQR(m_muon) - SQR(m_electron));
  m_idiff_mu_e = 3.0 * pi2hbar3 / (dm*dm*dm);
  dm = Kokkos::sqrt(SQR(m_s_quark) - SQR(m_d_quark));
  m_idiff_s_d = 3.0 * pi2hbar3 / (dm*dm*dm);

  ZL_a0   = pin->GetOrAddReal(block, "ZL_a0"  , -96.64);
  ZL_b0   = pin->GetOrAddReal(block, "ZL_b0"  , 58.85);
  ZL_gam0 = pin->GetOrAddReal(block, "ZL_gam0", 1.40);
  ZL_a1   = pin->GetOrAddReal(block, "ZL_a1"  , -26.06);
  ZL_b1   = pin->GetOrAddReal(block, "ZL_b1"  , 7.34);
  ZL_gam1 = pin->GetOrAddReal(block, "ZL_gam1", 2.45);
  n_sat   = pin->GetOrAddReal(block, "n_sat"  , 0.16);

  Bag_a4 = pin->GetOrAddReal(block, "Bag_a4", 1.0);
  Bag_av = pin->GetOrAddReal(block, "Bag_av", 0.20);
  Bag_B  = pin->GetOrAddReal(block, "Bag_B" , 160.0);
  Bag_B = Bag_B*Bag_B*Bag_B*Bag_B / (h_bar*h_bar*h_bar);
  Bag_av = Bag_av * h_bar;

  return true;
}

template<typename LogPolicy>
void EOSZlaBag<LogPolicy>::ReadTableFromFile(std::string fname) {
  if (m_initialized==false) {
    TableReader::Table table;
    auto read_result = table.ReadTable(fname);
    if (read_result.error != TableReader::ReadResult::SUCCESS) {
      std::cout << "Table could not be read.\n" << std::flush;
      abort();
    }
    // Make sure table has correct dimensions
    assert(table.GetNDimensions()==1);
    // TODO(PH) check that required fields are present?

    // Read baryon (neutron) mass
    auto& table_scalars = table.GetScalars();
    mb = table_scalars.at("mn");
    m_neutron = mb;
    m_proton  = table_scalars.at("mp");

    // Get table dimensions
    auto& point_info = table.GetPointInfo();
    m_nn = point_info[0].second;

    // (Re)Allocate device storage
    Kokkos::realloc(m_log_nb, m_nn);
    Kokkos::realloc(m_table, ECNVARS, m_nn);

    // Create host storage to read into
    HostArray1D<Real>::HostMirror host_log_nb = create_mirror_view(m_log_nb);
    HostArray2D<Real>::HostMirror host_table =  create_mirror_view(m_table);

    { // read nb
      Real * table_nb = table["nb"];

      for (size_t in=0; in<m_nn; ++in) {
        host_log_nb(in) = log2_(table_nb[in]);
      }

      m_id_log_nb = 1.0/(host_log_nb(1) - host_log_nb(0));
      min_n = table_nb[0]*(1 + 1e-15);
      max_n = table_nb[m_nn-1]*(1 - 1e-15);
    }

    { // Read Q1 -> log(P)
      Real * table_Q1 = table["Q1"];
      for (size_t in=0; in<m_nn; ++in) {
        Real p_current = table_Q1[in]*exp2_(host_log_nb(in));
        host_table(ECLOGP,in) = log2_(p_current);
      }
    }

    { // Read Q2 -> S
      Real * table_Q2 = table["Q2"];
      for (size_t in=0; in<m_nn; ++in) {
        host_table(ECENT,in) = table_Q2[in];
      }
    }

    { // Read Q3-> mu_b
      Real * table_Q3 = table["Q3"];
      for (size_t in=0; in<m_nn; ++in) {
        host_table(ECMUB,in) = (table_Q3[in]+1)*mb;
      }
    }

    { // Read Q4-> mu_q
      Real * table_Q4 = table["Q4"];
      for (size_t in=0; in<m_nn; ++in) {
        host_table(ECMUQ,in) = table_Q4[in]*mb;
      }
    }

    { // Read Q5-> mu_le
      Real * table_Q5 = table["Q5"];
      for (size_t in=0; in<m_nn; ++in) {
        host_table(ECMUL,in) = table_Q5[in]*mb;
      }
    }

    { // Read Q7-> log(e)
      Real * table_Q7 = table["Q7"];
      for (size_t in=0; in<m_nn; ++in) {
        Real e_current = mb*(table_Q7[in] + 1)*exp2_(host_log_nb(in));
        host_table(ECLOGE,in) = log2_(e_current);
      }
    }

    { // Read cs2-> cs
      Real * table_cs2 = table["cs2"];
      for (size_t in=0; in<m_nn; ++in) {
        host_table(ECCS,in) = sqrt(table_cs2[in]);
      }
    }

    { // Read f
      Real * table_f = table["Q8"];
      for (size_t in=0; in<m_nn; ++in) {
        host_table(ECFVOL,in) = table_f[in];
      }
    }

    { // Read Y_N
      Real * table_yn = table["Q9"];
      for (size_t in=0; in<m_nn; ++in) {
        host_table(ECYN,in) = table_yn[in];
      }
    }

    { // Read Y_EG
      Real * table_yln = table["Q10"];
      for (size_t in=0; in<m_nn; ++in) {
        host_table(ECYLN,in) = table_yln[in];
      }
    }

    { // Read Y_EN
      Real * table_ylq = table["Q11"];
      for (size_t in=0; in<m_nn; ++in) {
        host_table(ECYLQ,in) = table_ylq[in];
      }
    }

    // Worst-case accumulators for the table self-test. The scan is cheap and its
    // result is a real correctness check on the analytic EOS, so it always runs and
    // always reports; only the per-node dump is optional (zla_selftest_verbose).
    Real st_on_p = 0.0, st_on_e = 0.0, st_on_h = 0.0;
    Real st_off_p = 0.0, st_off_e = 0.0, st_off_h = 0.0;
    if (global_variable::my_rank == 0) {
    {
      for (size_t in=0; in<m_nn; ++in) {
        Real n = exp2_(host_log_nb[in]);
        Real f = host_table(ECFVOL, in);
        Real yn = host_table(ECYN, in);
        Real yln = host_table(ECYLN, in);
        Real ylq = host_table(ECYLQ, in);
        // Pack exactly as the pgen does, so the self-test also checks that
        // ConvertPrimitive's recovery is the inverse of the packing written by the
        // initial data.
        Real y[4] = {0.0};
        y[0] = f;
        y[1] = (f > 0.0) ? yn/f : 0.0;
        y[2] = yln;
        y[3] = ylq;
        Real nY[6] = {0.0};
        Real nY_Q[5] = {0.0};
        ConvertPrimitive(n, y, nY);
        if (nY[4] < 1.0) {
          QuarksFractionFromYq(nY[1], nY[3], nY_Q);
        }
        Real p_tab = exp2_(host_table(ECLOGP, in));
        Real p_cold = ColdPressure(n, y);
        Real p_n = ColdPressureNucleons(nY[0], nY[2]);
        Real p_q = ColdPressureQuarks(nY[1], nY[3]);
        Real e_tab = exp2_(host_table(ECLOGE, in));
        Real e_cold = ColdEnergy(n, y);
        Real e_n = ColdEnergyNucleons(nY[0], nY[2]);
        Real e_ln = ColdEnergyLeptons(nY[0], nY[2]);
        Real e_q = ColdEnergyQuarks(nY[1], nY[3]);
        Real e_lq = ColdEnergyLeptons(nY[1], nY[3]);
        // Enthalpy: computed directly from the analytic building blocks (same pattern
        // as p_n/p_q, e_n/e_q above) and checked against the thermodynamic identity
        // h = (P+E)/n using the tabulated P, E.
        Real h_n = (nY[4] > 0.0) ? ColdEnthalpyNucleons(nY[0], nY[2]) : 0.0;
        Real h_q = (nY[4] < 1.0) ? ColdEnthalpyQuarks(nY[1], nY[3]) : 0.0;
        Real h_g = (ZL_eta < 1.0) ? ColdEnthalpyLeptons(n, nY[5]) : 0.0;
        Real h_analytic = (h_n * nY[4] + h_q * (1.0-nY[4]) + (1.0-ZL_eta) * h_g) / n;
        Real h_tab = (p_tab + e_tab) / n;
        Real h_reldiff = h_analytic/h_tab - 1.0;
        st_on_p = fmax(st_on_p, fabs(p_cold/p_tab - 1.0));
        st_on_e = fmax(st_on_e, fabs(e_cold/e_tab - 1.0));
        st_on_h = fmax(st_on_h, fabs(h_reldiff));
        // Everything below is dump-only. test_dPdn/test_dEdn are numerical
        // derivatives, so skipping them is the bulk of the saving when quiet.
        if (!selftest_verbose) { continue; }
        Real e_mu = GetHeavyLeptonFraction(nY[0] * fabs(nY[2]), m_idiff_mu_e);
        Real chp_e = ChemPoFermion(nY[0] * fabs(nY[2]) * (1.0-e_mu), m_electron);
        Real chp_mu = ChemPoFermion(nY[0] * fabs(nY[2]) * (e_mu), m_muon);
        Real q_mu = GetHeavyLeptonFraction(nY[1] * fabs(nY[3]), m_idiff_mu_e);
        Real cs_tab = host_table(ECCS, in);
        Real cs_cold = Kokkos::sqrt(ColdSoundSpeed2(n, y));
        Real cs_dPdn = test_dPdn(n, y);
        Real cs_dEdn = test_dEdn(n, y);
        Real cs_dPdn_N = test_dPdn_N(n, y);
        Real cs_dEdn_N = test_dEdn_N(n, y);
        Real cs_dPdn_Q = test_dPdn_Q(n, y);
        Real cs_dEdn_Q = test_dEdn_Q(n, y);
        Real cs_dPdn_G = test_dPdn_G(n, y);
        Real cs_dEdn_G = test_dEdn_G(n, y);
        std::cout << "Test table " << in << std::endl
                  << in << " Prim0 = [ "
                  << n << ", "
                  << f << ", "
                  << yn << ", "
                  << yln << ", "
                  << ylq << " ], [ "
                  << nY[0] << ", "
                  << nY[1] << ", "
                  << nY[2] << ", "
                  << nY[3] << ", "
                  << nY[4] << ", "
                  << nY[5] << " ]"
                  << std::endl
                  << in << ", cs = [ "
                  << cs_tab << ", "
                  << cs_cold << ", "
                  << cs_dPdn << ", "
                  << cs_dEdn << ", "
                  << cs_dPdn_N << ", "
                  << cs_dEdn_N << ", "
                  << cs_dPdn_Q << ", "
                  << cs_dEdn_Q << ", "
                  << cs_dPdn_G << ", "
                  << cs_dEdn_G << ", "
                  << nY_Q[0] << ", "
                  << nY_Q[1] << ", "
                  << nY_Q[2] << ", "
                  << nY_Q[3] << ", "
                  << nY_Q[4]
                  << " ]" << std::endl
                  << in << ", P = [ "
                  << p_tab << ", "
                  << p_cold << ", "
                  << p_n << ", "
                  << p_q << ", "
                  << code_units.PressureConversion(eos_units) << ", "
                  << eos_units.TemperatureConversion(code_units)
                  << " ], E = [ "
                  << e_tab << ", "
                  << e_cold << ", "
                  << e_n << ", "
                  << e_q << ", "
                  << e_ln << ", "
                  << e_lq << ", "
                  << e_mu << ", "
                  << chp_e << ", "
                  << chp_mu
                  << " ]" << std::endl
                  << in << ", H = [ tab=" << h_tab
                  << ", analytic=" << h_analytic
                  << ", h_n=" << h_n
                  << ", h_q=" << h_q
                  << ", h_g=" << h_g
                  << ", reldiff=" << h_reldiff
                  << " ]" << std::endl;
      }
    }
    }

    // Copy from host to device
    Kokkos::deep_copy(m_log_nb, host_log_nb);
    Kokkos::deep_copy(m_table,  host_table);

    // Check consistency at OFF-GRID points, not just at table nodes (the loop above
    // only checks n exactly on the table, where ColdPressure/ColdEnergy(n, Y_table(n))
    // matches the table by construction). During evolution, only the analytic EOS is
    // ever evaluated, fed by a composition Y that is itself interpolated (via eval_at_n,
    // linear in log2 n) from the table -- independently of how P/E are interpolated.
    // This checks whether ColdPressure/ColdEnergy(n, Y_interp(n)) still agrees with the
    // table's own (log-log-linear) interpolation of P/E at the same off-grid n. It uses
    // eval_at_n_host below (a host-side copy of eval_at_n/weight_idx_ln's math run
    // against host_table/host_log_nb) rather than eval_at_n itself, since eval_at_n reads
    // the device-space m_table/m_log_nb directly and is only safe inside a parallel_for
    // (device execution spaces such as SYCL are not host-accessible).
    if (global_variable::my_rank == 0) {
    {
      auto eval_at_n_host = [&](int vi, Real n) -> Real {
        Real log_n = log2_(n);
        int i = static_cast<int>((log_n - host_log_nb(0))*m_id_log_nb);
        i = (i < 0) ? 0 : ((i > static_cast<int>(m_nn)-2) ? static_cast<int>(m_nn)-2 : i);
        Real w1 = (log_n - host_log_nb(i))*m_id_log_nb;
        Real w0 = 1.0 - w1;
        return w0 * host_table(vi, i+0) + w1 * host_table(vi, i+1);
      };
      for (size_t in=0; in+1<m_nn; ++in) {
        for (Real t : {0.25, 0.5, 0.75}) {
          Real log_nb_test = (1.0-t)*host_log_nb(in) + t*host_log_nb(in+1);
          Real n_test = exp2_(log_nb_test);

          Real f   = eval_at_n_host(ECFVOL, n_test);
          Real yn  = eval_at_n_host(ECYN,   n_test);
          Real yln = eval_at_n_host(ECYLN,  n_test);
          Real ylq = eval_at_n_host(ECYLQ,  n_test);
          Real y[4];
          y[0] = f;
          y[1] = (f > 0.0) ? yn/f : 0.0;
          y[2] = yln;
          y[3] = ylq;

          Real p_tab = exp2_(eval_at_n_host(ECLOGP, n_test));
          Real e_tab = exp2_(eval_at_n_host(ECLOGE, n_test));

          Real p_cold = ColdPressure(n_test, y);
          Real e_cold = ColdEnergy(n_test, y);

          Real p_reldiff = p_cold/p_tab - 1.0;
          Real e_reldiff = e_cold/e_tab - 1.0;

          // Enthalpy: same analytic building blocks as the on-grid loop above (matches
          // ColdEnthalpy()'s own combination, computed directly here to also report the
          // h_n/h_q/h_g breakdown).
          Real nY_h[6] = {0.0};
          ConvertPrimitive(n_test, y, nY_h);
          Real h_n = (nY_h[4] > 0.0) ? ColdEnthalpyNucleons(nY_h[0], nY_h[2]) : 0.0;
          Real h_q = (nY_h[4] < 1.0) ? ColdEnthalpyQuarks(nY_h[1], nY_h[3]) : 0.0;
          Real h_g = (ZL_eta < 1.0) ? ColdEnthalpyLeptons(n_test, nY_h[5]) : 0.0;
          Real h_analytic = (h_n*nY_h[4] + h_q*(1.0-nY_h[4]) + (1.0-ZL_eta)*h_g) / n_test;
          Real h_tab = (p_tab + e_tab) / n_test;
          Real h_reldiff = h_analytic/h_tab - 1.0;

          st_off_p = fmax(st_off_p, fabs(p_reldiff));
          st_off_e = fmax(st_off_e, fabs(e_reldiff));
          st_off_h = fmax(st_off_h, fabs(h_reldiff));
          if (!selftest_verbose) { continue; }

          std::cout << "Test offgrid " << in << " t=" << t
                    << " n = " << n_test
                    << " Y = [ " << f << ", " << yn << ", " << yln << ", " << ylq << " ]"
                    << " P = [ tab=" << p_tab << ", cold=" << p_cold
                    << ", reldiff=" << p_reldiff << " ]"
                    << " E = [ tab=" << e_tab << ", cold=" << e_cold
                    << ", reldiff=" << e_reldiff << " ]"
                    << " H = [ tab=" << h_tab << ", analytic=" << h_analytic
                    << ", reldiff=" << h_reldiff << " ]"
                    << std::endl;
        }
      }
    }
    }

    if (global_variable::my_rank == 0) {
      std::cout << "### EOSZlaBag table self-test over " << m_nn
                << " nodes: max |analytic/table - 1|" << std::endl
                << "###   on-grid  P = " << st_on_p
                << ", E = " << st_on_e << ", H = " << st_on_h << std::endl
                << "###   off-grid P = " << st_off_p
                << ", E = " << st_off_e << ", H = " << st_off_h << std::endl;
      if (!selftest_verbose) {
        std::cout << "###   (set <mhd>/zla_selftest_verbose = true for the per-node dump)"
                  << std::endl;
      }
    }

    m_initialized = true;

    min_Y[0] = 0.0;
    max_Y[0] = 1.0;
    min_Y[1] = 0.0;
    max_Y[1] = 1.0;
    min_Y[2] = 0.0;
    max_Y[2] = 1.0;
    min_Y[3] = -1.0;
    max_Y[3] =  2.0;

    m_min_h = std::numeric_limits<Real>::max();
    // Compute minimum enthalpy
    for (int in = 0; in < m_nn; ++in) {
      Real const nb = exp2_(host_log_nb(in));
      // This would use GPU memory, and we are currently on the CPU, so Enthalpy is
      // hardcoded
      Real e = exp2_(host_table(ECLOGE,in));
      Real p = exp2_(host_table(ECLOGP,in));
      Real h = (e + p) / nb;
      m_min_h = fmin(m_min_h, h);
    }

    // m_min_h is scanned along the table's beta-equilibrium composition, but the C2P
    // solver admits *frozen* compositions, whose enthalpy can sit below the
    // equilibrium minimum. PrimitiveSolver::ConToPrim uses 1/min_h as the upper
    // bracket for mu = 1/(h W) (primitive_solver.hpp:456-459), so an m_min_h that is
    // not a genuine lower bound over the admissible compositions puts the root
    // outside the bracket and FalsePosition reports NO_SOLUTION without iterating.
    // The rest mass per baryon is a valid floor for any cold, non-negative-pressure
    // state, and is exactly what the analytic EOS policies return
    // (cf. idealgas.hpp:92, piecewise_polytrope.hpp:121).
    if (min_h_floor_mb) {
      m_min_h = fmin(m_min_h, mb);
    }

    // Guard against a table that produced a nonsensical bound, as eos_compose.cpp
    // does. Without this the bracket silently becomes [0, inf) or [0, negative).
    if (!(m_min_h > 0.0)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "EOSZlaBag: minimum enthalpy is not positive ("
                << m_min_h << "); the table in " << fname << " is unusable."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // Locate the density below which no resolvable quark phase can be supported, i.e.
    // the largest n whose maximum admissible (1-f) is still under f_snap. Below it
    // SnapPhaseFraction can collapse straight to f = 1 with no EOS evaluation, which
    // is the path every atmosphere cell takes -- and the atmosphere is where the
    // failures live, so this keeps the floor essentially free where it matters most.
    //
    // The bound is 1-f <= P_N/(Bag_B + P_N) from P_N*f - Bag_B*(1-f) >= 0. P_N rises
    // monotonically with n, so bisect. Use yq = 0.5 (symmetric matter), which
    // maximises the nucleonic pressure and therefore *over*estimates what is
    // admissible -- making n_pfloor_force a deliberate underestimate, so the fast path
    // can never fire where a quark phase would actually have been supportable.
    if (pfloor_on) {
      Real nlo = min_n;
      Real nhi = max_n;
      const Real yq_max_P = 0.5;
      auto omf_max = [&](Real nn) {
        const Real PN = ColdPressureNucleons(nn, yq_max_P);
        return (PN > 0.0) ? PN/(Bag_B + PN) : 0.0;
      };
      if (omf_max(nlo) >= f_snap) {
        n_pfloor_force = 0.0;            // even the floor density supports a phase
      } else if (omf_max(nhi) < f_snap) {
        n_pfloor_force = nhi;            // nothing in the table supports one
      } else {
        for (int it = 0; it < 200; ++it) {
          const Real nmid = sqrt(nlo*nhi);
          if (omf_max(nmid) < f_snap) { nlo = nmid; } else { nhi = nmid; }
          if (nhi <= nlo*(1.0 + 1.0e-12)) { break; }
        }
        n_pfloor_force = nlo;
      }
      std::cout << "### EOSZlaBag: pressure floor on the phase fraction is ENABLED"
                << std::endl
                << "###   zla_pfloor_frac = " << pfloor_frac
                << ",  n_pfloor_force = " << n_pfloor_force << " fm^-3"
                << "  (n_sat = " << n_sat << " fm^-3)" << std::endl;
    }
  } // if (m_initialized==false)
}

template class EOSZlaBag<NormalLogs>;
template class EOSZlaBag<NQTLogs>;

} // namespace Primitive
