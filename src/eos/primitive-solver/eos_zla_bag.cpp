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
  h_bar = pin->GetOrAddReal(block, "h_bar", 197.327);
  pi2hbar3 = Kokkos::numbers::pi*Kokkos::numbers::pi * h_bar*h_bar*h_bar;

  ZL_eta = pin->GetOrAddReal(block, "ZL_eta", 1.0);

  enforce_eos_equilibrium = pin->GetOrAddBoolean(block, "enforce_eos_equilibrium", false);

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

    if (global_variable::my_rank == 0) {
    {
      for (size_t in=0; in<m_nn; ++in) {
        Real n = exp2_(host_log_nb[in]);
        Real f = host_table(ECFVOL, in);
        Real yn = host_table(ECYN, in);
        Real yln = host_table(ECYLN, in);
        Real ylq = host_table(ECYLQ, in);
        Real y[4] = {0.0};
        y[0] = f;
        y[1] = yn;
        y[2] = yn * yln;
        y[3] = (1.0-yn) * ylq;
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
    // table's own (log-log-linear) interpolation of P/E at the same off-grid n, using
    // eval_at_n itself so this exercises the exact same interpolation code used
    // elsewhere (e.g. by the GetXFromRho accessors used to set up TOV/BNS initial data).
    // Note: eval_at_n reads m_table/m_log_nb directly, so this assumes a host-accessible
    // execution space (true for this project's current CPU-only OpenMP/Serial Kokkos
    // build; would need a parallel_for wrapper on a GPU build).
    if (global_variable::my_rank == 0) {
    {
      for (size_t in=0; in+1<m_nn; ++in) {
        for (Real t : {0.25, 0.5, 0.75}) {
          Real log_nb_test = (1.0-t)*host_log_nb(in) + t*host_log_nb(in+1);
          Real n_test = exp2_(log_nb_test);

          Real f   = eval_at_n(ECFVOL, n_test);
          Real yn  = eval_at_n(ECYN,   n_test);
          Real yln = eval_at_n(ECYLN,  n_test);
          Real ylq = eval_at_n(ECYLQ,  n_test);
          Real y[4] = {f, yn, yn*yln, (1.0-yn)*ylq};

          Real p_tab = exp2_(eval_at_n(ECLOGP, n_test));
          Real e_tab = exp2_(eval_at_n(ECLOGE, n_test));

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
  } // if (m_initialized==false)
}

template class EOSZlaBag<NormalLogs>;
template class EOSZlaBag<NQTLogs>;

} // namespace Primitive
