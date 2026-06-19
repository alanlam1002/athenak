#ifndef UTILS_TOV_TOV_ZLABAG_HPP_
#define UTILS_TOV_TOV_ZLABAG_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file tov_zla_bag.hpp
//  \brief Tabulated ZLA BAG EOS in equilibrium for use with TOVStar
#include <iostream>
#include <string>
#include <sstream>
#include <stdexcept>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "utils/tr_table.hpp"
#include "tov_utils.hpp"
#include "eos/primitive-solver/unit_system.hpp"

namespace tov {

class ZlaBagEOS {
 private:
  DualArray1D<Real> m_log_rho;
  DualArray1D<Real> m_log_p;
  DualArray1D<Real> m_log_e;
  DualArray1D<Real> m_fvol;
  DualArray1D<Real> m_yn;
  DualArray1D<Real> m_yln;
  DualArray1D<Real> m_ylq;

  Real dlrho;
  Real lrho_min;
  Real lrho_max;
  Real lP_min;
  Real lP_max;
  Real le_min;
  Real le_max;

  Real y0_atmosphere;
  Real y1_atmosphere;
  Real y2_atmosphere;
  Real y3_atmosphere;

  std::string fname;
  size_t m_nn;

  //static const Real fm_to_Msun = 6.771781959609192e-19
  //static const Real MeV_to_Msun = 8.962968324680417e-61
  static constexpr Real ener_to_geo = 2.8863099290608455e-6;

 public:
  explicit ZlaBagEOS(ParameterInput* pin) {
    fname = pin->GetString("problem", "table");

    TableReader::Table table;

    auto read_result = table.ReadTable(fname);
    if (read_result.error != TableReader::ReadResult::SUCCESS) {
      std::cout << read_result.message << std::endl
                << "TOV EOS table could not be read.\n";
      std::exit(EXIT_FAILURE);
    }

    // Unit conversions
    Primitive::UnitSystem unit_geo = Primitive::MakeGeometricSolar();
    Primitive::UnitSystem unit_nuc = Primitive::MakeNuclear();

    auto test_field = [](bool test, const std::string name) -> void {
      if (test) {
        return;
      } else {
        std::stringstream ss;
        ss << "Table is missing key '" << name << "'\n";
        throw std::runtime_error(ss.str());
      }
    };

    // TODO(JMF) Check that table has right fields and dimensions
    auto& table_scalars = table.GetScalars();
    test_field(table_scalars.count("mn") > 0, "mn");
    Real mb = table_scalars.at("mn");

    // Get table dimensions
    auto& point_info = table.GetPointInfo();
    m_nn = point_info[0].second;

    // Allocate storage
    Kokkos::realloc(m_log_rho, m_nn);
    Kokkos::realloc(m_log_p, m_nn);
    Kokkos::realloc(m_log_e, m_nn);
    Kokkos::realloc(m_fvol, m_nn);
    Kokkos::realloc(m_yn, m_nn);
    Kokkos::realloc(m_yln, m_nn);
    Kokkos::realloc(m_ylq, m_nn);

    // Read rho
    test_field(table.HasField("nb"), "nb");
    Real * table_nb = table["nb"];
    for (size_t in = 0; in < m_nn; in++) {
      //m_log_rho.h_view(in) = log(table_nb[in]*mb*ener_to_geo);
      m_log_rho.h_view(in) = log(table_nb[in]*mb*
                                 unit_nuc.MassDensityConversion(unit_geo));
    }
    dlrho = m_log_rho.h_view(1)-m_log_rho.h_view(0);
    lrho_min = m_log_rho.h_view(0);
    lrho_max = m_log_rho.h_view(m_nn-1);

    // Read pressure
    test_field(table.HasField("Q1"), "Q1");
    Real * table_Q1 = table["Q1"];
    for (size_t in = 0; in < m_nn; in++) {
      m_log_p.h_view(in) = log(table_Q1[in]*table_nb[in]*
                                unit_nuc.EnergyDensityConversion(unit_geo));
    }
    lP_min = m_log_p.h_view(0);
    lP_max = m_log_p.h_view(m_nn-1);

    // Read energy
    test_field(table.HasField("Q7"), "Q7");
    Real * table_Q7 = table["Q7"];
    for (size_t in = 0; in < m_nn; in++) {
      m_log_e.h_view(in) = log(mb*(table_Q7[in] + 1.)*table_nb[in]*
                                    unit_nuc.EnergyDensityConversion(unit_geo));
    }
    le_min = m_log_e.h_view(0);
    le_max = m_log_e.h_view(m_nn-1);

    // Read volume fraction
    test_field(table.HasField("Q8"), "Q8");
    Real * table_Q8 = table["Q8"];
    for (size_t in = 0; in < m_nn; in++) {
      m_fvol.h_view(in) = table_Q8[in];
    }
    // Read Nucleons fraction
    test_field(table.HasField("Q9"), "Q9");
    Real * table_Q9 = table["Q9"];
    for (size_t in = 0; in < m_nn; in++) {
      m_yn.h_view(in) = table_Q9[in];
    }
    // Read Nucleons electron fraction
    test_field(table.HasField("Q10"), "Q10");
    Real * table_Q10 = table["Q10"];
    for (size_t in = 0; in < m_nn; in++) {
      m_yln.h_view(in) = table_Q10[in];
    }
    // Read Quarks electron fraction
    test_field(table.HasField("Q11"), "Q11");
    Real * table_Q11 = table["Q11"];
    for (size_t in = 0; in < m_nn; in++) {
      m_ylq.h_view(in) = table_Q11[in];
    }

    std::cout << "Loaded table " << fname << std::endl
              << "  rho = [" << exp(lrho_min) << ", " << exp(lrho_max) << "]" << std::endl
              << "  P = [" << exp(lP_min) << ", " << exp(lP_max) << "]" << std::endl;

    y0_atmosphere = pin->GetOrAddReal("mhd", "s0_atmosphere",1.0);
    y1_atmosphere = pin->GetOrAddReal("mhd", "s1_atmosphere",1.0);
    y2_atmosphere = pin->GetOrAddReal("mhd", "s2_atmosphere",0.0);
    y3_atmosphere = pin->GetOrAddReal("mhd", "s3_atmosphere",0.0);

    // Sync the views to the GPU
    m_log_rho.template modify<HostMemSpace>();
    m_log_p.template modify<HostMemSpace>();
    m_log_e.template modify<HostMemSpace>();
    m_fvol.template modify<HostMemSpace>();
    m_yn.template modify<HostMemSpace>();
    m_yln.template modify<HostMemSpace>();
    m_ylq.template modify<HostMemSpace>();

    m_log_rho.template sync<DevExeSpace>();
    m_log_p.template sync<DevExeSpace>();
    m_log_e.template sync<DevExeSpace>();
    m_fvol.template sync<DevExeSpace>();
    m_yn.template sync<DevExeSpace>();
    m_yln.template sync<DevExeSpace>();
    m_ylq.template sync<DevExeSpace>();
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetPFromRho(Real rho) const {
    Real lrho = log(rho);
    if (lrho < lrho_min) {
      return 0.0;
    }
    int lb = fmin(static_cast<int>((lrho-lrho_min)/dlrho), static_cast<int>(m_nn)-2);
    int ub = lb + 1;
    auto& lrho_view = GetView<loc>(m_log_rho);
    auto& lp_view = GetView<loc>(m_log_p);
    return exp(Interpolate(lrho, lrho_view(lb), lrho_view(ub), lp_view(lb), lp_view(ub)));
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetEFromRho(Real rho) const {
    Real lrho = log(rho);
    if (lrho < lrho_min) {
      return 0.0;
    }
    int lb = fmin(static_cast<int>((lrho-lrho_min)/dlrho), static_cast<int>(m_nn)-2);
    int ub = lb + 1;
    auto& lrho_view = GetView<loc>(m_log_rho);
    auto& le_view = GetView<loc>(m_log_e);
    return exp(Interpolate(lrho, lrho_view(lb), lrho_view(ub), le_view(lb), le_view(ub)));
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetRhoFromE(Real e) const {
    Real le = log(e);

    // Guard against negative total energy
    // densities and garbage values.
    if (le < le_min || e < 0.0) {
      return 0.0;
    }

    return GetRhoFromVar<loc>(le, m_log_e);
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetFvolFromRho(Real rho) const {
    Real lrho = log(rho);
    if (lrho < lrho_min) {
      return y0_atmosphere;
    }
    int lb = fmin(static_cast<int>((lrho-lrho_min)/dlrho), static_cast<int>(m_nn)-2);
    int ub = lb + 1;
    auto& lrho_view = GetView<loc>(m_log_rho);
    auto& y0_view = GetView<loc>(m_fvol);
    return Interpolate(lrho, lrho_view(lb), lrho_view(ub), y0_view(lb), y0_view(ub));
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetYnFromRho(Real rho) const {
    Real lrho = log(rho);
    if (lrho < lrho_min) {
      return y1_atmosphere;
    }
    int lb = fmin(static_cast<int>((lrho-lrho_min)/dlrho), static_cast<int>(m_nn)-2);
    int ub = lb + 1;
    auto& lrho_view = GetView<loc>(m_log_rho);
    auto& y1_view = GetView<loc>(m_yn);
    return Interpolate(lrho, lrho_view(lb), lrho_view(ub), y1_view(lb), y1_view(ub));
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetYlnFromRho(Real rho) const {
    Real lrho = log(rho);
    if (lrho < lrho_min) {
      return y2_atmosphere;
    }
    int lb = fmin(static_cast<int>((lrho-lrho_min)/dlrho), static_cast<int>(m_nn)-2);
    int ub = lb + 1;
    auto& lrho_view = GetView<loc>(m_log_rho);
    auto& y2_view = GetView<loc>(m_yln);
    return Interpolate(lrho, lrho_view(lb), lrho_view(ub), y2_view(lb), y2_view(ub));
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetYlqFromRho(Real rho) const {
    Real lrho = log(rho);
    if (lrho < lrho_min) {
      return y3_atmosphere;
    }
    int lb = fmin(static_cast<int>((lrho-lrho_min)/dlrho), static_cast<int>(m_nn)-2);
    int ub = lb + 1;
    auto& lrho_view = GetView<loc>(m_log_rho);
    auto& y3_view = GetView<loc>(m_ylq);
    return Interpolate(lrho, lrho_view(lb), lrho_view(ub), y3_view(lb), y3_view(ub));
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetRhoFromP(Real P) const {
    Real lP = log(P);
    // If the pressure is below the minimum of the table, we return zero density.
    if (lP < lP_min) {
      return 0.0;
    }
    return GetRhoFromVar<loc>(lP, m_log_p);
  }

 private:
  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  auto& GetView(const DualArray1D<Real>& arr) const {
    if constexpr (loc == LocationTag::Host) {
      return arr.h_view;
    } else {
      return arr.d_view;
    }
  }

  // Use bisection on a specified variable to find the corresponding density.
  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetRhoFromVar(Real var, const DualArray1D<Real>& arr_var) const {
    int lb = 0;
    int ub = m_nn-1;
    auto& v_view = GetView<loc>(arr_var);
    auto& lrho_view = GetView<loc>(m_log_rho);
    // Do a binary search for the lower and upper indices of arr_var
    while (ub - lb > 1) {
      int idx = (lb + ub)/2;
      if (v_view(idx) > var) {
        ub = idx;
      } else {
        lb = idx;
      }
    }
    return exp(Interpolate(var, v_view(lb), v_view(ub), lrho_view(lb), lrho_view(ub)));
  }
};


} // namespace tov

#endif // UTILS_TOV_TOV_ZLABAG_HPP_
