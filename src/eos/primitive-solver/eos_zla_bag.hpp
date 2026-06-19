#ifndef EOS_PRIMITIVE_SOLVER_EOS_ZLA_BAG_HPP_
#define EOS_PRIMITIVE_SOLVER_EOS_ZLA_BAG_HPP_
//========================================================================================
// PrimitiveSolver equation-of-state framework
// Copyright(C) 2023 Jacob M. Fields <jmf6719@psu.edu>
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos_zla_bag.hpp
//  \brief Defines EOSTable, which stores information from a 1D tabulated
//         equation of state in CompOSE format, with a thermal Gamma-Law component.
//
//  Tables should be generated using
//  <a href="https://bitbucket.org/dradice/pycompose">PyCompOSE</a>

///  \warning This code assumes the table to be uniformly spaced in
///           log nb

#include <string>
#include <limits>

#include <Kokkos_Core.hpp>

#include "../../athena.hpp"
#include "ps_types.hpp"
#include "eos_policy_interface.hpp"
#include "unit_system.hpp"
#include "logs.hpp"

#define ZL_CUBE(x) ((x)*(x)*(x))
#define ZL_POW4(x) ((x)*(x)*(x)*(x))
#define ZL_POW6(x) ((x)*(x)*(x)*(x)*(x)*(x))
#define ZL_POW8(x) ((x)*(x)*(x)*(x)*(x)*(x)*(x)*(x))
#define ZL_POW12(x) ((x)*(x)*(x)*(x)*(x)*(x)*(x)*(x)*(x)*(x)*(x)*(x))

namespace Primitive {

template<typename LogPolicy>
class EOSZlaBag : public EOSPolicyInterface, public LogPolicy {
 private:
  using LogPolicy::log2_;
  using LogPolicy::exp2_;

 public:
  enum TableVariables {
    ECLOGP  = 0,  //! log (pressure / 1 MeV fm^-3)
    ECENT   = 1,  //! entropy per baryon [kb]
    ECMUB   = 2,  //! baryon chemical potential [MeV]
    ECMUQ   = 3,  //! charge chemical potential [MeV]
    ECMUL   = 4,  //! lepton chemical potential [MeV]
    ECLOGE  = 5,  //! log (total energy density / 1 MeV fm^-3)
    ECCS    = 6,  //! sound speed [c]
    ECFVOL  = 7,  //! volume fraction in equilibrium [f]
    ECYN    = 8,  //! nucleons fraction in equilibrium [f n_B,N / n_B]
    ECYLN   = 9,  //! leptons fraction for nucleons in equilibrium [y_q,N]
    ECYLQ   = 10, //! leptons fraction for quarks in equilibrium [y_q,Q]
    ECNVARS = 11
  };

 protected:
  /// Constructor
  EOSZlaBag() :
      m_log_nb("log nb",1),
      m_table("EoS table",1,1) {
    n_species = 0;
    eos_units = MakeNuclear();
    m_initialized = false;

    // These will be set properly when the table is read
    m_id_log_nb = std::numeric_limits<Real>::quiet_NaN();
    m_nn = std::numeric_limits<int>::quiet_NaN();
    m_min_h = std::numeric_limits<Real>::max();
    mb =    std::numeric_limits<Real>::quiet_NaN();

    h_bar = std::numeric_limits<Real>::quiet_NaN();

    min_n = std::numeric_limits<Real>::quiet_NaN();
    max_n = std::numeric_limits<Real>::quiet_NaN();
    min_T = 0.0;
    max_T = std::numeric_limits<Real>::max();
    for (int i = 0; i < MAX_SPECIES; i++) {
      min_Y[i] = std::numeric_limits<Real>::quiet_NaN();
      max_Y[i] = std::numeric_limits<Real>::quiet_NaN();
    }
  }

/*
  /// Destructor
  ~EOSZlaBag();
*/

  /// Temperature from energy density.
  KOKKOS_INLINE_FUNCTION Real TemperatureFromE(Real n, Real e, Real *Y) const {
    assert (m_initialized);
    if (n < min_n) {
      // If density is OOB then return minimum temperature
      return min_T;
    } else if (e <= MinimumEnergy(n, Y)) {
      // If energy is OOB then return minimum temperature
      return min_T;
    }

    Real e_cold = ColdEnergy(n, Y);
    Real T = gamma_th_m1*(e-e_cold)/n;
    return Kokkos::fmax(T,min_T);
  }

  /// Calculate the temperature using.
  KOKKOS_INLINE_FUNCTION Real TemperatureFromP(Real n, Real p, Real *Y) const {
    assert (m_initialized);
    if (n < min_n) {
      // If density is OOB then return minimum temperature
      return min_T;
    } else if (p <= MinimumPressure(n, Y)) {
      // If pressure is OOB then return minimum temperature
      return min_T;
    }

    Real p_cold = ColdPressure(n, Y);
    Real T = (p-p_cold)/n;
    return Kokkos::fmax(T,min_T);
  }

  /// Calculate the energy density using.
  KOKKOS_INLINE_FUNCTION Real Energy(Real n, Real T, const Real *Y) const {
    Real e_cold = ColdEnergy(n, Y);
    Real e_th   = n*T/gamma_th_m1;
    return e_cold + e_th;
  }

  /// Calculate the pressure using.
  KOKKOS_INLINE_FUNCTION Real Pressure(Real n, Real T, Real *Y) const {
    Real p_cold = ColdPressure(n, Y);
    Real p_th   = n*T;
    return p_cold + p_th;
  }

  /// Calculate the enthalpy per baryon using.
  KOKKOS_INLINE_FUNCTION Real Enthalpy(Real n, Real T, Real *Y) const {
    Real H_cold = ColdEnthalpy(n, Y);
    Real H_th   = (gamma_th*T)/(gamma_th_m1);
    return H_cold + H_th;
  }

  /// Calculate the sound speed.
  KOKKOS_INLINE_FUNCTION Real SoundSpeed(Real n, Real T, Real *Y) const {
    Real H_cold = ColdEnthalpy(n, Y);
    Real H_th   = (gamma_th*T)/(gamma_th_m1);

    Real Hcs2_cold = pow(ColdSoundSpeed(n, Y),2.0)*H_cold;
    Real Hcs2_th   = gamma_th*T;

    return sqrt((Hcs2_cold + Hcs2_th)/(H_cold + H_th));
  }

  /// Calculate the specific internal energy per unit mass.
  KOKKOS_INLINE_FUNCTION Real SpecificInternalEnergy(Real n, Real T, Real *Y) const {
    return Energy(n, T, Y)/(mb*n) - 1;
  }

  /// Calculate Energy for the cold part.
  KOKKOS_INLINE_FUNCTION Real ColdEnergy(Real n, const Real *Y) const {
    assert (m_initialized);
    Real nY[6] = {0.0};
    ConvertPrimitive(n, Y, nY);
    Real E_N = 0.0;
    Real E_Q = 0.0;
    Real E_G = 0.0;
    if (nY[4] > 0.0) E_N = ColdEnergyNucleons(nY[0], nY[2]);
    if (nY[4] < 1.0) E_Q = ColdEnergyQuarks(nY[1], nY[3]);
    if (ZL_eta < 1.0) E_G = ColdEnergyLeptons(n, nY[5]);
    return E_N * nY[4] + E_Q * (1.0-nY[4]) + (1.0-ZL_eta) * E_G;
  }

  /// Calculate pressure for the cold part.
  KOKKOS_INLINE_FUNCTION Real ColdPressure(Real n, Real *Y) const {
    assert (m_initialized);
    Real nY[6] = {0.0};
    ConvertPrimitive(n, Y, nY);
    Real P_N = 0.0;
    Real P_Q = 0.0;
    Real P_G = 0.0;
    if (nY[4] > 0.0) P_N = ColdPressureNucleons(nY[0], nY[2]);
    if (nY[4] < 1.0) P_Q = ColdPressureQuarks(nY[1], nY[3]);
    if (ZL_eta < 1.0) P_G = ColdPressureLeptons(n, nY[5]);
    return P_N * nY[4] + P_Q * (1.0-nY[4]) + (1.0-ZL_eta) * P_G;
  }

  /// Calculate enthalpy for the cold part.
  KOKKOS_INLINE_FUNCTION Real ColdEnthalpy(Real n, Real *Y) const {
    assert (m_initialized);
    Real nY[6] = {0.0};
    ConvertPrimitive(n, Y, nY);
    Real h_N = 0.0;
    Real h_Q = 0.0;
    Real h_G = 0.0;
    if (nY[4] > 0.0) h_N = ColdEnthalpyNucleons(nY[0], nY[2]);
    if (nY[4] < 1.0) h_Q = ColdEnthalpyQuarks(nY[1], nY[3]);
    if (ZL_eta < 1.0) h_G = ColdEnthalpyLeptons(n, nY[5]);
    return ( h_N * nY[4] + h_Q * (1.0-nY[4]) + (1.0-ZL_eta) * h_G ) / n;
  }

  /// Calculate sound speed for the cold part.
  KOKKOS_INLINE_FUNCTION Real ColdSoundSpeed(Real n, Real *Y) const {
    assert (m_initialized);
    Real nY[6] = {0.0};
    ConvertPrimitive(n, Y, nY);
    Real dEdn_N = 0.0;
    Real dEdn_Q = 0.0;
    Real dEdn_G = 0.0;
    Real dPdn_N = 0.0;
    Real dPdn_Q = 0.0;
    Real dPdn_G = 0.0;
    if (nY[4] > 0.0) {
      Real Y_N[3] = {0.0};
      NucleonsFractionFromYq(nY[0], nY[2], Y_N);
      dEdn_N = dEdn_Nucleons(nY[0], Y_N);
      dPdn_N = dPdn_Nucleons(nY[0], Y_N);
    }
    if (nY[4] < 1.0) {
      Real Y_Q[5] = {0.0};
      QuarksFractionFromYq(nY[1], nY[3], Y_Q);
      dEdn_Q = dEdn_Quarks(nY[1], Y_Q);
      dPdn_Q = dPdn_Quarks(nY[1], Y_Q);
    }
    if (ZL_eta < 1.0) {
      Real yf = GetHeavyLeptonFraction(n * nY[5], m_electron, m_muon);
      Real y_e  = nY[5] * (1.0 - yf);
      Real y_mu = nY[5] * yf;
      dEdn_G = dEdn_Leptons(n, y_e, y_mu);
      dPdn_G = dPdn_Leptons(n, y_e, y_mu);
    }
    return ( dPdn_N * nY[4] + dPdn_Q * (1.0-nY[4]) + (1.0-ZL_eta) * dPdn_G )
          /( dEdn_N * nY[4] + dEdn_Q * (1.0-nY[4]) + (1.0-ZL_eta) * dEdn_G );
  }

  /// Convert Primitive Variables to the mass fraction
  /// (n_N, n_Q, Y_qN, Y_qQ, f, Y_qG)
  KOKKOS_INLINE_FUNCTION void ConvertPrimitive(Real n, const Real *Y, Real nY[6]) const{
    nY[4] = Y[0];                                 // Volume fraction
    if ( Y[0] > 0.0 ) {
      nY[0] = Y[1] * n / Y[0];                    // Nucleon number density
      if ( Y[1] > 0.0 ) {
        nY[2] = Y[2] / Y[1];                      // Charge Fraction for Nucleons
      }
    }
    if ( Y[0] < 1.0 ) {
      nY[1] = (1.0-Y[1]) * n / (1.0-Y[0]);        // Quark number density
      if ( Y[1] < 1.0 ) {
        nY[3] = Y[3] / (1.0-Y[1]);                // Charge Fraction for Quarks
      }
    }
    nY[5] = Y[2] + Y[3];                          // Total Charge Fraction
    //nY[0] = Y[0] * n;                             // Nucleon number density
    //nY[1] = Y[1] * n;                             // Quark number density
    //nY[2] = Y[2] / Y[0];                          // Charge Fraction for Nucleons
    //nY[3] = Y[3] / Y[1];                          // Charge Fraction for Quarks
    //nY[4] = (nY[1] - n) / (nY[1] - nY[0]);        // Volume fraction
    //nY[5] = nY[4] * nY[2] * (1.0-nY[4]) * nY[3];  // Total Charge Fraction
  }

  /// Leptons Cold Energy (electron and muon)
  KOKKOS_INLINE_FUNCTION Real ColdEnergyLeptons(Real n, Real yq) const{
    // Muon and Electron Fraction Assuming Equilibrium
    Real abs_yq = Kokkos::fabs(yq);
    Real yf = GetHeavyLeptonFraction(n * abs_yq, m_electron, m_muon);
    Real y_e  = abs_yq * (1.0 - yf);
    Real y_mu = abs_yq * yf;
    return EnergyFermion(n * y_e , m_electron)
         + EnergyFermion(n * y_mu, m_muon);
  }

  /// Leptons Cold Energy (electron and muon)
  KOKKOS_INLINE_FUNCTION Real ColdPressureLeptons(Real n, Real yq) const{
    // Muon and Electron Fraction Assuming Equilibrium
    Real abs_yq = Kokkos::fabs(yq);
    Real yf = GetHeavyLeptonFraction(n * abs_yq, m_electron, m_muon);
    Real y_e  = abs_yq * (1.0 - yf);
    Real y_mu = abs_yq * yf;
    return PressureFermion(n * y_e , m_electron) 
         + PressureFermion(n * y_mu, m_muon);
  }

  /// Leptons Cold Energy (electron and muon)
  KOKKOS_INLINE_FUNCTION Real ColdEnthalpyLeptons(Real n, Real yq) const{
    // Muon and Electron Fraction Assuming Equilibrium
    Real abs_yq = Kokkos::fabs(yq);
    Real yf = GetHeavyLeptonFraction(n * abs_yq, m_electron, m_muon);
    Real y_e  = abs_yq * (1.0 - yf);
    Real y_mu = abs_yq * yf;
    return EnthalpyFermion(n * y_e , m_electron) 
         + EnthalpyFermion(n * y_mu, m_muon);
  }

  /// Leptons Fraction
  KOKKOS_INLINE_FUNCTION void LeptonsFractionFromYq(Real n, Real yq, Real Y[2]) const{
    // Muon and Electron Fraction Assuming Equilibrium
    Real abs_yq = Kokkos::fabs(yq);
    Real yf = GetHeavyLeptonFraction(n * abs_yq, m_electron, m_muon);
    Y[0]  = yq * (1.0 - yf);
    Y[1]  = yq * yf;
  }

  /// Leptons dE/dn
  KOKKOS_INLINE_FUNCTION Real dEdn_Leptons(Real n, Real y_e, Real y_mu) const{
    return dEdn_Fermion(n, y_e , m_electron)
         + dEdn_Fermion(n, y_mu, m_muon);
  }

  /// Leptons dP/dn
  KOKKOS_INLINE_FUNCTION Real dPdn_Leptons(Real n, Real y_e, Real y_mu) const{
    return dPdn_Fermion(n, y_e , m_electron)
         + dPdn_Fermion(n, y_mu, m_muon);
  }

  /// Nucleons Cold Energy
  KOKKOS_INLINE_FUNCTION Real ColdEnergyNucleons(Real n, Real yq) const{
    return EnergyFermion(n *      yq , m_proton)
         + EnergyFermion(n * (1.0-yq), m_neutron)
         + ZLattimer_Energy(n, yq)
         + ZL_eta * ColdEnergyLeptons(n, yq);
  }

  /// Nucleons Cold Pressure
  KOKKOS_INLINE_FUNCTION Real ColdPressureNucleons(Real n, Real yq) const{
    return PressureFermion(n *      yq , m_proton)
         + PressureFermion(n * (1.0-yq), m_neutron)
         + ZLattimer_Pressure(n, yq)
         + ZL_eta * ColdPressureLeptons(n, yq);
  }

  /// Nucleons Cold Enthalpy
  KOKKOS_INLINE_FUNCTION Real ColdEnthalpyNucleons(Real n, Real yq) const{
    return n * ( ChemPoNucleon(n,     yq, m_proton ) * yq
               + ChemPoNucleon(n, 1.0-yq, m_neutron) * (1.0-yq) )
         + ZL_eta * ColdEnthalpyLeptons(n, yq);
  }

  /// Nucleons Fraction from charge fraction
  KOKKOS_INLINE_FUNCTION void NucleonsFractionFromYq(Real n, Real yq, Real Y[3]) const{
    Real abs_yq = Kokkos::fabs(yq);
    Y[0] = yq;
    // Muon and Electron Fraction Assuming Equilibrium
    Real yf = GetHeavyLeptonFraction(n * abs_yq, m_electron, m_muon);
    Y[1] = yq * (1.0 - yf);
    Y[2] = yq * yf;
  }

  /// Nucleons dEdn
  KOKKOS_INLINE_FUNCTION Real dEdn_Nucleons(Real n, Real *Y) const{
    return dEdn_Fermion(n,     Y[0], m_proton)
         + dEdn_Fermion(n, 1.0-Y[0], m_neutron)
         + ZLattimer_dEdn(n, Y[0])
         + ZL_eta * dEdn_Leptons(n, Y[1], Y[2]);
  }

  /// Nucleons dPdn
  KOKKOS_INLINE_FUNCTION Real dPdn_Nucleons(Real n, Real *Y) const{
    return dPdn_Fermion(n,     Y[0], m_proton)
         + dPdn_Fermion(n, 1.0-Y[0], m_neutron)
         + ZLattimer_dPdn(n, Y[0])
         + ZL_eta * dPdn_Leptons(n, Y[1], Y[2]);
  }

  /// Quarks Cold Energy
  KOKKOS_INLINE_FUNCTION Real ColdEnergyQuarks(Real n, Real yq) const{
    Real nQ = n / Bag_a4;
    Real y_u = 1.0 + yq;      // Up Quark
    Real y_ds = 2.0 - yq;     // Down + Strange Quarks
    Real n_ds = y_ds * nQ;
    // Down and Strange Quarks Fraction assuming equilibrium
    Real yf = GetHeavyLeptonFraction(n_ds * ONE_3RD, m_d_quark, m_s_quark);
    Real y_d = y_ds * (1.0 - yf);
    Real y_s = y_ds * yf;
    return Bag_a4 * 3.0 *
        ( EnergyFermion(nQ * y_u * ONE_3RD, m_u_quark)
        + EnergyFermion(nQ * y_d * ONE_3RD, m_d_quark)
        + EnergyFermion(nQ * y_s * ONE_3RD, m_s_quark) )
        + Bag_B + 0.5 * Bag_av * SQR(3.0 * nQ)
        + ZL_eta * ColdEnergyLeptons(n, yq);
  }

  /// Quarks Cold Pressure
  KOKKOS_INLINE_FUNCTION Real ColdPressureQuarks(Real n, Real yq) const{
    Real nQ = n / Bag_a4;
    Real y_u = 1.0 + yq;      // Up Quark
    Real y_ds = 2.0 - yq;     // Down + Strange Quarks
    Real n_ds = y_ds * nQ;
    // Down and Strange Quarks Fraction assuming equilibrium
    Real yf = GetHeavyLeptonFraction(n_ds * ONE_3RD, m_d_quark, m_s_quark);
    Real y_d = y_ds * (1.0 - yf);
    Real y_s = y_ds * yf;
    return Bag_a4 * 3.0 * 
         ( PressureFermion(nQ * y_u * ONE_3RD, m_u_quark)
         + PressureFermion(nQ * y_d * ONE_3RD, m_d_quark)
         + PressureFermion(nQ * y_s * ONE_3RD, m_s_quark) )
         - Bag_B + 0.5 * Bag_av * SQR(3.0 * nQ)
         + ZL_eta * ColdPressureLeptons(n, yq);
  }

  /// Quarks Cold Enthalpy
  KOKKOS_INLINE_FUNCTION Real ColdEnthalpyQuarks(Real n, Real yq) const{
    Real nQ = n / Bag_a4;
    Real y_u = 1.0 + yq;      // Up Quark
    Real y_ds = 2.0 - yq;     // Down + Strange Quarks
    Real n_ds = y_ds * nQ;
    // Down and Strange Quarks Fraction assuming equilibrium
    Real yf = GetHeavyLeptonFraction(n_ds * ONE_3RD, m_d_quark, m_s_quark);
    Real y_d = y_ds * (1.0 - yf);
    Real y_s = y_ds * yf;
    return Bag_a4 * 3.0 *
         ( EnthalpyFermion(nQ * y_u * ONE_3RD, m_u_quark)
         + EnthalpyFermion(nQ * y_d * ONE_3RD, m_d_quark)
         + EnthalpyFermion(nQ * y_s * ONE_3RD, m_s_quark) )
         + ZL_eta * ColdEnthalpyLeptons(n, yq);
  }

  /// Quarks Fraction from charge fraction
  KOKKOS_INLINE_FUNCTION void QuarksFractionFromYq(Real n, Real yq, Real Y[5]) const{
    Real nQ = n / Bag_a4;
    Y[0] = 1.0 + yq;          // Up Quark
    Real y_ds = 2.0 - yq;     // Down + Strange Quarks
    Real n_ds = y_ds * nQ;
    // Down and Strange Quarks Fraction assuming equilibrium
    Real yf = GetHeavyLeptonFraction(n_ds * ONE_3RD, m_d_quark, m_s_quark);
    Y[1] = y_ds * (1.0 - yf);
    Y[2] = y_ds * yf;
    yf = GetHeavyLeptonFraction(n * yq, m_electron, m_muon);
    Y[3] = yq * (1.0 - yf);
    Y[4] = yq * yf;
  }

  /// Quarks dEdn
  KOKKOS_INLINE_FUNCTION Real dEdn_Quarks(Real n, Real *Y) const{
    Real nQ = n / Bag_a4;
    return Bag_a4 * 3.0 *
           ( dEdn_Fermion(nQ, Y[0] * ONE_3RD, m_u_quark)
           + dEdn_Fermion(nQ, Y[1] * ONE_3RD, m_d_quark)
           + dEdn_Fermion(nQ, Y[2] * ONE_3RD, m_s_quark) )
           + 9.0 * Bag_av * nQ / Bag_a4
           + ZL_eta * dEdn_Leptons(n, Y[3], Y[4]);
  }

  /// Quarks dPdn
  KOKKOS_INLINE_FUNCTION Real dPdn_Quarks(Real n, Real *Y) const{
    Real nQ = n / Bag_a4;
    return Bag_a4 * 3.0 *
           ( dPdn_Fermion(nQ, Y[0] * ONE_3RD, m_u_quark)
           + dPdn_Fermion(nQ, Y[1] * ONE_3RD, m_d_quark)
           + dPdn_Fermion(nQ, Y[2] * ONE_3RD, m_s_quark) )
           + 9.0 * Bag_av * nQ / Bag_a4
           + ZL_eta * dPdn_Leptons(n, Y[3], Y[4]);
  }

  /// Chemical Potential for Nucleon (Proton or Neutron)
  KOKKOS_INLINE_FUNCTION Real ChemPoNucleon(Real n, Real y, Real m) const{
    return ChemPoFermion(n*y, m) + ZLattimer_ChemPo(n, y);
  }

  /// Chemical Potential for Quark
  KOKKOS_INLINE_FUNCTION Real ChemPoQuark(Real n, Real y, Real m) const{
    return 3.0 * ChemPoFermion(n*y * ONE_3RD, m) + Bag_av * 3.0 * n / Bag_a4;
  }

  /// Zhao-Lattimer Energy
  KOKKOS_INLINE_FUNCTION Real ZLattimer_Energy(Real n, Real y) const{
    Real u = n / n_sat;
    return n *
      ( 4.0 * y * (1.0-y) 
        * ( ZL_a0 * u + ZL_b0 * Kokkos::pow(u, ZL_gam0) ) 
      + SQR(1.0 - 2.0 * y) 
        * ( ZL_a1 * u + ZL_b1 * Kokkos::pow(u, ZL_gam1) ) );
  }

  /// Zhao-Lattimer Chemical Potential
  KOKKOS_INLINE_FUNCTION Real ZLattimer_ChemPo(Real n, Real y) const{
    Real u = n / n_sat;
    return - 2.0 * u * (ZL_a1 * (1.0-2.0*y) - 2.0 * ZL_a0 * (1.0-y))
      + 4.0 * ZL_b0 * Kokkos::pow(u, ZL_gam0)
      * (1.0-y) * (1.0 + (ZL_gam0 - 1.0) * y)
      - ZL_b1 * Kokkos::pow(u, ZL_gam1) 
      * (1.0 - 2.0 * y) * (3.0 - 2.0 * y - ZL_gam1 * (1.0 - 2.0 * y));
  }

  /// Zhao-Lattimer Pressure
  KOKKOS_INLINE_FUNCTION Real ZLattimer_Pressure(Real n, Real y) const{
    Real u = n / n_sat;
    return n * ( SQR(1.0-2.0*y)
        * (ZL_a1 * u + ZL_b1 * ZL_gam1 * Kokkos::pow(u, ZL_gam1))
        + 4.0 * y * (1.0-y) 
        * (ZL_a0 * u + ZL_b0 * ZL_gam0 * Kokkos::pow(u, ZL_gam0)) );
  }

  /// Zhao-Lattimer dEdn
  KOKKOS_INLINE_FUNCTION Real ZLattimer_dEdn(Real n, Real y) const{
    Real u = n / n_sat;
    return SQR(1.0-2.0*y) * (2.0 * ZL_a1 * u 
      + ZL_b1 * (1.0+ZL_gam1) * Kokkos::pow(u, ZL_gam1) )
      + 4.0 * y * (1.0-y) * (2.0 * ZL_a0 * u
      + ZL_b0 * (1.0+ZL_gam0) * Kokkos::pow(u, ZL_gam0) );
  }

  /// Zhao-Lattimer Potential dPdn
  KOKKOS_INLINE_FUNCTION Real ZLattimer_dPdn(Real n, Real y) const{
    Real u = n / n_sat;
    return SQR(1.0-2.0*y)
      * (2.0 * ZL_a1 * u + ZL_b1 * ZL_gam1 * (1.0+ZL_gam1) * Kokkos::pow(u, ZL_gam1))
      + 4.0 * y * (1.0-y) 
      * (2.0 * ZL_a0 * u + ZL_b0 * ZL_gam0 * (1.0+ZL_gam0) * Kokkos::pow(u, ZL_gam0));
  }

  /// Fermion Chemical Potential
  KOKKOS_INLINE_FUNCTION Real ChemPoFermion(Real n, Real m) const{
    Real x = FermiMomentum(n) / m;
    return m * ChemPoFermion_FromX(x);
  }

  /// Fermion Energy
  KOKKOS_INLINE_FUNCTION Real EnergyFermion(Real n, Real m) const{
    Real x = FermiMomentum(n) / m;
    return ZL_POW4(m) * EnergyFermion_FromX(x);
  }

  /// Fermion Pressure
  KOKKOS_INLINE_FUNCTION Real PressureFermion(Real n, Real m) const{
    Real x = FermiMomentum(n) / m;
    return ZL_POW4(m) * PressureFermion_FromX(x);
  }

  /// Fermion Enthalpy
  KOKKOS_INLINE_FUNCTION Real EnthalpyFermion(Real n, Real m) const{
    return n * ChemPoFermion(n, m);
  }

  /// Fermion dE / dn
  KOKKOS_INLINE_FUNCTION Real dEdn_Fermion(Real n, Real y, Real m) const{
    return y * ChemPoFermion(n*y, m);
  }

  /// Fermion dP / dn
  KOKKOS_INLINE_FUNCTION Real dPdn_Fermion(Real n, Real y, Real m) const{
    Real x = FermiMomentum(n*y) / m;
    return ONE_3RD * m * y * SQR(x) / ChemPoFermion_FromX(x);
  }

  /// Fermion Chemical Potential / m
  KOKKOS_INLINE_FUNCTION Real ChemPoFermion_FromX(Real x) const{
    return Kokkos::sqrt(x*x+1.0);
  }

  /// Fermion Energy / m^4
  KOKKOS_INLINE_FUNCTION Real EnergyFermion_FromX(Real x) const{
    if ( x > 1.e-2 ) {
      return 0.125 / pi2hbar3 
          * (x * Kokkos::sqrt(SQR(x) + 1.0) * (2.0*SQR(x) + 1.0)
          - Kokkos::log1p(x + SQR(x) / (1.0 + Kokkos::sqrt(SQR(x) + 1.0))));
    } else {
      Real x3 = x*x*x;
      Real x2 = x*x;
      return 0.125 / pi2hbar3 * x3
          * ( ONE_3RD * 8.0 + 0.8 * x2 - SQR(x2) / 7.0 );
    }
  }

  /// Fermion Pressure / m^4
  KOKKOS_INLINE_FUNCTION Real PressureFermion_FromX(Real x) const{
    if ( x > 1.e-2 ) {
      return 0.125 * ONE_3RD / pi2hbar3 
          * (x * Kokkos::sqrt(SQR(x) + 1.0) * (2.0*SQR(x) - 3.0)
          + 3.0 * Kokkos::log1p(x + SQR(x) / (1.0 + Kokkos::sqrt(SQR(x) + 1.0))));
    } else {
      Real x5 = x*x*x*x*x;
      Real x2 = x*x;
      return 0.125 * ONE_3RD / pi2hbar3 * x5
          * (1.6 - 4.0 * x2 / 7.0 + ONE_3RD * SQR(x2) );
    }
  }

  /// Fermion dEdx / m^4
  KOKKOS_INLINE_FUNCTION Real dEdx_Fermion(Real x) const{
    return SQR(x) * Kokkos::sqrt(SQR(x) + 1.0) / pi2hbar3;
  }

  /// Fermi Momentum
  KOKKOS_INLINE_FUNCTION Real FermiMomentum(Real n) const{
    return Kokkos::pow(3.0 * pi2hbar3 * n, ONE_3RD);
  }

  /// Calcalate the lepton fractions assuming equilibrium (chem1 = chem2)
  KOKKOS_INLINE_FUNCTION Real GetHeavyLeptonFraction(Real n, Real m1, Real m2) const {
    Real a = Kokkos::sqrt(SQR(m2) - SQR(m1));
    Real a3 = ZL_CUBE(a);
    Real b = 3.0 * pi2hbar3 * n;
    if (b > a3) {
      Real a2 = SQR(a);
      Real a6 = ZL_POW6(a);
      Real b2 = SQR(b);
      Real c_cubic =- 11.0 * SQR(a6)
                    + 14.0 * a6 * b2
                    -  2.0 * SQR(b2)
                    +  2.0 * Kokkos::sqrt((b-a3) * (b+a3)
                    * ZL_CUBE(a6 + b2)); // b^4
      Real c = SIGN(c_cubic) * Kokkos::cbrt(Kokkos::fabs(c_cubic)); // b^(4/3)
      Real d = (5.0*ZL_POW8(a) - 4.0*a2*b2 + SQR(c))/(3.0*a2*c); // b^(2/3)
      Real f = (-6.0*a6 + b2)/(9.0*SQR(a2)); //b^(2/3)
      Real g = (2.0*b*(9.0 - b2/a6))/27.0; // b
      Real kF = ( -(b/a2) - SIGN(b-3.0*a3)*3.0*Kokkos::sqrt(d+f) 
              + 3.0*Kokkos::sqrt(-d + 2.0*f - SIGN(b-3.0*a3)*g
              / Kokkos::sqrt(d+f)) ) / 6.0; // b^(1/3)
      return ZL_CUBE(kF) / b;
    } else {
      return 0.0;
    }
  }

  /// Get the minimum enthalpy per baryon.
  KOKKOS_INLINE_FUNCTION Real MinimumEnthalpy() const {
    assert (m_initialized);
    return m_min_h;
  }

  /// Get the minimum pressure at a given density and composition.
  KOKKOS_INLINE_FUNCTION Real MinimumPressure(Real n, Real *Y) const {
    return Pressure(n, min_T, Y);
  }

  /// Get the maximum pressure at a given density and composition.
  KOKKOS_INLINE_FUNCTION Real MaximumPressure(Real n, Real *Y) const {
    // Note that max_T is already set to numeric_limits<Real>::max!
    return max_T;
  }

  /// Get the minimum energy at a given density and composition.
  KOKKOS_INLINE_FUNCTION Real MinimumEnergy(Real n, Real *Y) const {
    return Energy(n, min_T, Y);
  }

  /// Get the maximum energy at a given density and composition.
  KOKKOS_INLINE_FUNCTION Real MaximumEnergy(Real n, Real *Y) const {
    // Note that max_T is already set to numeric_limits<Real>::max!
    return max_T;
  }

 public:
  //! \brief Load the EOS parameters from the input file
  bool ReadParametersFromInput(std::string block, ParameterInput * pin);

  /// Reads the table file.
  void ReadTableFromFile(std::string fname);

  /// Get the raw number density.
  KOKKOS_INLINE_FUNCTION DvceArray1D<Real> const GetRawLogNumberDensity() const {
    return m_log_nb;
  }

  /// Get the raw table data.
  KOKKOS_INLINE_FUNCTION DvceArray2D<Real> const GetRawTable() const {
    return m_table;
  }

  // Indexing used to access the data.
  KOKKOS_INLINE_FUNCTION ptrdiff_t index(int iv, int in) const {
    return in + m_nn*iv;
  }

  /// Check if the EOS has been initialized properly.
  KOKKOS_INLINE_FUNCTION bool IsInitialized() const {
    return m_initialized;
  }

  /// Set the number of species. Throw an exception if
  /// the number of species is invalid. TODO
  KOKKOS_INLINE_FUNCTION void SetNSpecies(int n) {
    // Number of species must be within limits
    assert (n<=MAX_SPECIES && n>=0);

    n_species = n;
    return;
  }

  /// Set the adiabatic constant for the thermal part.
  /// Gamma is limited to the range 1.00001 <= g <= 2.0.
  KOKKOS_INLINE_FUNCTION void SetThermalGamma(Real g) {
    gamma_th = (g <= 1.00001) ? 1.00001 : ((g >= 2.0) ? 2.0 : g);
    gamma_th_m1 = gamma_th - 1.0;
  }

  /// Get the adiabatic constant for the thermal part.
  KOKKOS_INLINE_FUNCTION Real GetThermalGamma() const {
    return gamma_th;
  }

  /// Set the EOS unit system.
  KOKKOS_INLINE_FUNCTION void SetEOSUnitSystem(UnitSystem units) {
    eos_units = units;
  }

 private:
  /// Low level evaluation function, not intended for outside use.
  KOKKOS_INLINE_FUNCTION Real eval_at_n(int vi, Real n) const {
    Real log_n = log2_(n);
    return eval_at_ln(vi, log_n);
  }

  /// Low level evaluation function, not intended for outside use.
  KOKKOS_INLINE_FUNCTION Real eval_at_ln(int iv, Real log_n)
      const {
    int in;
    Real wn0, wn1;

    weight_idx_ln(&wn0, &wn1, &in, log_n);

    return
      wn0 * m_table(iv, in+0) +
      wn1 * m_table(iv, in+1);
  }

  /// Evaluate interpolation weight for density.
  KOKKOS_INLINE_FUNCTION void weight_idx_ln(Real *w0, Real *w1, int *in, Real log_n)
      const {
    *in = (log_n - m_log_nb(0))*m_id_log_nb;
    *w1 = (log_n - m_log_nb(*in))*m_id_log_nb;
    *w0 = 1.0 - (*w1);
    return;
  }

 private:
  // Inverse of table spacing
  Real m_id_log_nb;
  // Table size
  int m_nn;
  // Minimum enthalpy per baryon
  Real m_min_h;
  // Reduced Planck Constant
  Real h_bar;
  // Mass
  Real m_electron;
  Real m_muon;
  Real m_proton;
  Real m_neutron;
  Real m_u_quark;
  Real m_d_quark;
  Real m_s_quark;
  // Parameters for ZLA EOS
  Real ZL_a0;
  Real ZL_b0;
  Real ZL_gam0;
  Real ZL_a1;
  Real ZL_b1;
  Real ZL_gam1;
  Real n_sat;
  // Parameters for MIT Bag models
  Real Bag_a4;
  Real Bag_av;
  Real Bag_B;
  // Parameters for Phase Transition
  Real ZL_eta;

  // bool to protect against access of uninitialized table and prevent repeated reading
  // of table
  bool m_initialized;

  // Table storage on DEVICE.
  DvceArray1D<Real> m_log_nb;
  DvceArray2D<Real> m_table;

  // Thermal Gamma
  Real gamma_th;
  Real gamma_th_m1;

  // Pi^2 hbar^3
  Real pi2hbar3;
};

}; // namespace Primitive

#undef ZL_CUBE
#undef ZL_POW4
#undef ZL_POW6
#undef ZL_POW8
#undef ZL_POW12

#endif //EOS_PRIMITIVE_SOLVER_EOS_ZLA_BAG_HPP_
