//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file c2p_zla_bag.cpp
//  \brief Diagnostic unit test for the ZLA-bag EOS conserved-to-primitive solve at
//         atmosphere densities.
//
//  Motivation: a dyn-GR MHD BNS run with dyn_eos = zla_bag emits Error::NO_SOLUTION in
//  essentially every atmosphere cell, for the trivially invertible state v = 0, B = 0,
//  D = dfloor, Y = (1,1,0,0). This test instruments that exact state.
//
//  With rsqr = bsqr = 0 the Kastaun root function reduces to the straight line
//      f(mu) = mu - 1/nu,   nu = max(h/mb, (D*(1+q) + P)/D)   (independent of mu)
//  bracketed on [mul, muh] = [0, 1/min_h] (primitive_solver.hpp:456-459). Since
//  f(0) = -1/nu < 0 always, the solve can only fail when f(1/min_h) is also negative,
//  i.e. when nu < min_h -- FalsePosition then returns false at the flb*fub > 0 test
//  (numtools_root.hpp:62) before performing a single iteration. This test computes nu
//  and the two bracket residuals directly so that failure mode is visible.
//
//  Four sweeps are run:
//    A) f = Y[0] stepped down from exactly 1.0. The bag constant is stored as
//       Bag_B^4/hbar^3 ~ 85.3 MeV/fm^3, while n*mb at the atmosphere is only ~1.7e-15
//       MeV/fm^3, and the quark branch is taken for *any* f < 1, weighted by (1-f). The
//       bag term enters the energy as +(1-f)*Bag_B and the pressure as -(1-f)*Bag_B, so
//       it cancels in h = (e+P)/(n mb) -- but only to round-off. The surviving residue
//       is ~(1-f)*Bag_B*eps_mach/(n*mb), which exceeds the true enthalpy excess
//       (h/mb - 1 = 3.17e-13) once (1-f) > ~3e-14. Beyond that, h/mb lands below 1
//       erratically, nu < min_h, and the solve fails at the bracket sign test.
//    B) density stepped up from the bottom of the table with f held at 1 - 1 ulp.
//    C) PrimToCon -> ConToPrim round trip at exactly f = 1, to measure how accurately
//       rho, T and P are recovered (the temperature is recovered from
//       ColdEnergy - n*mb, so it is round-off noise at these densities).
//    D) tau scaled about its floor value at exactly f = 1. Included to show that tau
//       excursions are *harmless* here: nu is pinned to nu_a = h/mb, which does not
//       depend on tau, so even tau -> 0 still solves.
//    E) explicit conserved states read from <problem>/states_file, one per line, so
//       that real failing cells lifted verbatim out of a production log can be
//       inverted offline. Each line carries the 18 numbers a C2P failure report
//       prints, in the order they are printed:
//           D Sx Sy Sz tau Dy0 Dy1 Dy2 Dy3 Bx By Bz g11 g12 g13 g22 g23 g33
//       Blank lines and lines beginning with '#' are skipped. The printed D/S/tau/B
//       are already divided by sqrt(detg) (primitive_solver_hyd.hpp:442-464), which is
//       exactly the normalisation ConToPrim expects, so the numbers go straight in.

#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"

template<class LogPolicy>
void PerformC2PTests(Mesh* pmesh, ParameterInput *pin);

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::C2PZlaBag()
//! \brief Runs the ZLA-bag C2P atmosphere diagnostics.

void ProblemGenerator::C2PZlaBag(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (pmbp->pdyngr == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "The zla_bag C2P unit test only works for DynGRMHD!" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string eos_string = pin->GetString("mhd", "dyn_eos");
  if (eos_string.compare("zla_bag") != 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "The zla_bag C2P unit test needs <mhd> dyn_eos = zla_bag!" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (pin->GetOrAddInteger("mhd", "nscalars", 0) != 4) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "The zla_bag C2P unit test needs <mhd> nscalars = 4!" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (pin->GetOrAddBoolean("mhd", "use_NQT", false)) {
    PerformC2PTests<Primitive::NQTLogs>(pmy_mesh_, pin);
  } else {
    PerformC2PTests<Primitive::NormalLogs>(pmy_mesh_, pin);
  }

  // Initialize the ADM variables to Minkowski, otherwise the pgen finishes with a pile
  // of unrelated C2P failures (cf. pgen/unit_tests/eos_compose.cpp).
  pmbp->padm->SetADMVariables(pmbp);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void PerformC2PTests()
//! \brief Instruments ConToPrim at atmosphere densities for the ZLA-bag EOS.

template<class LogPolicy>
void PerformC2PTests(Mesh *pmesh, ParameterInput *pin) {
  MeshBlockPack *pmbp = pmesh->pmb_pack;

  using EOSPolicy = Primitive::EOSZlaBag<LogPolicy>;
  using ErrorPolicy = Primitive::ResetFloorZlaBag;

  // Same downcast as pgen/unit_tests/eos_compose.cpp, but we need the PrimitiveSolver
  // itself rather than just the EOS, so that ConToPrim can be exercised.
  auto& ps = static_cast<dyngr::DynGRMHDPS<EOSPolicy, ErrorPolicy>*>(pmbp->pdyngr)
             ->eos.ps;

  const int n_species = ps.GetEOS().GetNSpecies();
  if (n_species != 4) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Expected 4 species from the ZLA-bag EOS, got " << n_species
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // Report the scalar parameters that set the scale of the problem. h_bar and Bag_B are
  // re-read from the input because the EOS stores them in derived combinations.
  const Real h_bar = pin->GetOrAddReal("mhd", "h_bar", 197.327);
  const Real bag_b_in = pin->GetOrAddReal("mhd", "Bag_B", 160.0);
  const Real bag_b = bag_b_in*bag_b_in*bag_b_in*bag_b_in/(h_bar*h_bar*h_bar);

  const Real n_atm = ps.GetEOS().GetDensityFloor();
  const Real T_atm = ps.GetEOS().GetTemperatureFloor();
  const Real mb = ps.GetEOS().GetBaryonMass();
  const Real min_h = ps.GetEOS().GetMinimumEnthalpy();
  const Real min_n = ps.GetEOS().GetMinimumDensity();
  const Real max_n = ps.GetEOS().GetMaximumDensity();

  std::cout << std::endl
            << "==================== zla_bag C2P atmosphere diagnostics ==============="
            << std::endl
            << "  log policy   = " << (std::is_same_v<LogPolicy, Primitive::NQTLogs>
                                       ? "NQTLogs" : "NormalLogs") << std::endl
            << "  c2p_tol      = " << ps.tol << std::endl
            << "  c2p_iter     = " << ps.GetRootSolver().iterations << std::endl
            << "  mb           = " << mb << std::endl
            << "  min_n, max_n = " << min_n << ", " << max_n << std::endl
            << "  n_atm        = " << n_atm << "   (n_atm/min_n = "
            << n_atm/min_n << ")" << std::endl
            << "  T_atm        = " << T_atm << std::endl
            << "  min_h - 1    = " << min_h - 1.0 << std::endl
            << "  h_bar        = " << h_bar << "   (CODATA hbar*c = 197.3269804)"
            << std::endl
            << "  Bag_B^4/hbar^3 = " << bag_b << " MeV/fm^3" << std::endl
            << "  n_atm*mb     = " << n_atm*mb << " (code units)" << std::endl
            << "======================================================================"
            << std::endl << std::endl;

  Real Y_atm[MAX_SPECIES] = {0.0};
  for (int i = 0; i < n_species; ++i) {
    Y_atm[i] = ps.GetEOS().GetSpeciesAtmosphere(i);
  }
  std::cout << "  Y_atm = (" << Y_atm[0] << ", " << Y_atm[1] << ", "
            << Y_atm[2] << ", " << Y_atm[3] << ")" << std::endl << std::endl;

  const int nf = pin->GetOrAddInteger("problem", "nf", 7);
  const int nn = pin->GetOrAddInteger("problem", "nn", 7);

  // Test E input: explicit conserved states, 18 numbers per line. Read on the host and
  // pushed to the device so the same kernel serves CPU and GPU builds.
  const int NSTF = 18;
  std::string states_file = pin->GetOrAddString("problem", "states_file", "");
  std::vector<Real> flat;
  int nst = 0;
  if (!states_file.empty()) {
    std::ifstream fh(states_file);
    if (!fh) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Could not open <problem>/states_file = '"
                << states_file << "'" << std::endl;
      exit(EXIT_FAILURE);
    }
    std::string line;
    int lineno = 0;
    while (std::getline(fh, line)) {
      lineno++;
      std::size_t first = line.find_first_not_of(" \t\r");
      if (first == std::string::npos || line[first] == '#') { continue; }
      std::istringstream iss(line);
      Real v[NSTF];
      int k = 0;
      while (k < NSTF && (iss >> v[k])) { k++; }
      if (k != NSTF) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << states_file << ":" << lineno << ": expected " << NSTF
                  << " numbers, read " << k << std::endl;
        exit(EXIT_FAILURE);
      }
      for (int i = 0; i < NSTF; ++i) { flat.push_back(v[i]); }
      nst++;
    }
    std::cout << "  Test E: read " << nst << " conserved state(s) from " << states_file
              << std::endl << std::endl;
  }
  DualArray2D<Real> states("c2p_states", (nst > 0) ? nst : 1, NSTF);
  for (int m = 0; m < nst; ++m) {
    for (int i = 0; i < NSTF; ++i) { states.h_view(m, i) = flat[m*NSTF + i]; }
  }
  states.template modify<HostMemSpace>();
  states.template sync<DevExeSpace>();

  // Everything below runs in a single-element kernel on the default execution space so
  // that it works identically on CPU and GPU builds and the output stays ordered.
  auto& ps_ = ps;
  const int nf_ = nf;
  const int nn_ = nn;
  const int nst_ = nst;
  auto st_ = states;
  Kokkos::parallel_for("c2p_zla_bag", Kokkos::RangePolicy<>(DevExeSpace(), 0, 1),
  KOKKOS_LAMBDA(const int /*dummy*/) {
    const auto& eos = ps_.GetEOS();
    const Real mb_ = eos.GetBaryonMass();
    const Real min_h_ = eos.GetMinimumEnthalpy();
    const Real n_a = eos.GetDensityFloor();
    const Real T_a = eos.GetTemperatureFloor();

    // Flat spatial metric.
    Real g3d[NSPMETRIC] = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0};
    Real g3u[NSPMETRIC] = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0};

    Real Ya[MAX_SPECIES] = {0.0};
    for (int i = 0; i < 4; ++i) {
      Ya[i] = eos.GetSpeciesAtmosphere(i);
    }

    // Replicates RootFunctor (primitive_solver.hpp:88-163) for rsqr = bsqr = 0, where
    // the root function is mu - 1/nu with nu independent of mu.
    auto compute_nu = [&](Real n, Real *Y, Real *nu_a_out, Real *nu_b_out,
                          Real *T_out, Real *P_out) {
      const Real D = n*mb_;
      const Real tau = eos.GetTauFloor(D, Y, 0.0);
      const Real q = tau/D;
      const Real eoverD = 1.0 + q;
      // Mirror the shipped RootFunctor: recover T from the internal energy, which for
      // a cell at rest is exactly tau, rather than from the total energy density.
      Real That = eos.GetTemperatureFromInternalE(n, tau, Y);
      eos.ApplyTemperatureLimits(That);
      const Real ehat = eos.GetEnergy(n, That, Y);
      const Real Phat = eos.GetPressure(n, That, Y);
      *nu_a_out = (ehat + Phat)/(mb_*n);
      *nu_b_out = (D*eoverD + Phat)/D;
      *T_out = That;
      *P_out = Phat;
      return q;
    };

    // ------------------------------------------------------------------ Test A
    Kokkos::printf("--- Test A: f = Y[0] stepped below 1, n = n_atm = %.17g\n", n_a);
    Kokkos::printf("%-24s %-13s %-13s %-14s %-14s %-14s %s\n",
                   "1-f", "P_cold", "P(T_solver)", "h/mb-1", "nu-1", "fub", "ConToPrim");
    const Real eps_mach = 2.220446049250313e-16;
    for (int k = 0; k < nf_; ++k) {
      // 0, 1 ulp, 2 ulp, then a decade sweep 1e-15 ... 1e-2.
      Real one_minus_f = 0.0;
      if (k == 0) {
        one_minus_f = 0.0;
      } else if (k == 1) {
        one_minus_f = eps_mach;
      } else if (k == 2) {
        one_minus_f = 2*eps_mach;
      } else {
        one_minus_f = 1.0e-15;
        for (int j = 0; j < k - 3; ++j) { one_minus_f *= 10.0; }
      }
      Real Y[MAX_SPECIES] = {0.0};
      Y[0] = 1.0 - one_minus_f;
      // SpeciesLimits ties Y[1] to Y[0]; mirror that so the state is self-consistent.
      Y[1] = Y[0];
      Y[2] = Ya[2];
      Y[3] = Ya[3];

      Real nu_a, nu_b, T, P;
      compute_nu(n_a, Y, &nu_a, &nu_b, &T, &P);
      const Real nu = Kokkos::fmax(nu_a, nu_b);
      const Real flb = 0.0 - 1.0/nu;
      const Real fub = 1.0/min_h_ - 1.0/nu;

      // Build a self-consistent conserved state and try to invert it.
      Real prim[NPRIM] = {0.0};
      Real cons[NCONS] = {0.0};
      Real bu[NMAG] = {0.0, 0.0, 0.0};
      prim[PRH] = n_a;
      prim[PVX] = prim[PVY] = prim[PVZ] = 0.0;
      prim[PTM] = T_a;
      prim[PPR] = eos.GetPressure(n_a, T_a, Y);
      for (int i = 0; i < 4; ++i) { prim[PYF + i] = Y[i]; }
      ps_.PrimToCon(prim, cons, bu, g3d);

      Real prim_out[NPRIM] = {0.0};
      auto res = ps_.ConToPrim(prim_out, cons, bu, g3d, g3u);

      // P_cold is GetPressure at T_atm; P(T_solver) is the same call at the temperature
      // the root function derives from the conserved state. They agree once the
      // temperature is recovered from the internal energy; a gap between them means
      // a spurious thermal pressure has crept in from a rest-mass cancellation.
      Kokkos::printf("%-24.17g %-13.5e %-13.5e %-14.6e %-14.6e %-14.6e %s\n",
                     one_minus_f, prim[PPR], P, nu_a - 1.0, nu - 1.0, fub,
                     (res.error == Primitive::Error::SUCCESS) ? "SUCCESS"
                     : ((res.error == Primitive::Error::NO_SOLUTION) ? "NO_SOLUTION"
                                                                    : "OTHER"));
    }

    // ------------------------------------------------------------------ Test B
    Kokkos::printf("\n--- Test B: f = 1 - 1ulp, density swept upward from min_n\n");
    Kokkos::printf("%-14s %-14s %-14s %-14s %-14s %-14s %s\n",
                   "n", "n*mb", "(1-f)*BagB/(n*mb)", "h/mb-1", "nu-1", "fub",
                   "ConToPrim");
    for (int k = 0; k < nn_; ++k) {
      Real fac = 1.0;
      for (int j = 0; j < k; ++j) { fac *= 100.0; }
      Real n = eos.GetMinimumDensity()*fac;
      if (n > eos.GetMaximumDensity()) { break; }

      Real Y[MAX_SPECIES] = {0.0};
      Y[0] = 1.0 - eps_mach;
      Y[1] = Y[0];
      Y[2] = Ya[2];
      Y[3] = Ya[3];

      Real nu_a, nu_b, T, P;
      compute_nu(n, Y, &nu_a, &nu_b, &T, &P);
      const Real nu = Kokkos::fmax(nu_a, nu_b);
      const Real fub = 1.0/min_h_ - 1.0/nu;

      Real prim[NPRIM] = {0.0};
      Real cons[NCONS] = {0.0};
      Real bu[NMAG] = {0.0, 0.0, 0.0};
      prim[PRH] = n;
      prim[PTM] = T_a;
      prim[PPR] = eos.GetPressure(n, T_a, Y);
      for (int i = 0; i < 4; ++i) { prim[PYF + i] = Y[i]; }
      ps_.PrimToCon(prim, cons, bu, g3d);

      Real prim_out[NPRIM] = {0.0};
      auto res = ps_.ConToPrim(prim_out, cons, bu, g3d, g3u);

      Kokkos::printf("%-14.5e %-14.5e %-14.6e %-14.6e %-14.6e %-14.6e %s\n",
                     n, n*mb_, eps_mach*8.5294419e+01/(n*mb_), nu_a - 1.0, nu - 1.0,
                     fub,
                     (res.error == Primitive::Error::SUCCESS) ? "SUCCESS"
                     : ((res.error == Primitive::Error::NO_SOLUTION) ? "NO_SOLUTION"
                                                                    : "OTHER"));
    }

    // ------------------------------------------------------------------ Test C
    Kokkos::printf("\n--- Test C: PrimToCon -> ConToPrim round trip at exactly f = 1\n");
    Kokkos::printf("%-14s %-14s %-14s %-14s %-14s %s\n",
                   "n", "rel err rho", "rel err T", "rel err P", "nu-1", "ConToPrim");
    for (int k = 0; k < nn_; ++k) {
      Real fac = 1.0;
      for (int j = 0; j < k; ++j) { fac *= 100.0; }
      Real n = eos.GetMinimumDensity()*fac;
      if (n > eos.GetMaximumDensity()) { break; }

      Real Y[MAX_SPECIES] = {0.0};
      Y[0] = 1.0;
      Y[1] = 1.0;
      Y[2] = Ya[2];
      Y[3] = Ya[3];

      Real nu_a, nu_b, T, P;
      compute_nu(n, Y, &nu_a, &nu_b, &T, &P);
      const Real nu = Kokkos::fmax(nu_a, nu_b);

      Real prim[NPRIM] = {0.0};
      Real cons[NCONS] = {0.0};
      Real bu[NMAG] = {0.0, 0.0, 0.0};
      prim[PRH] = n;
      prim[PTM] = T_a;
      prim[PPR] = eos.GetPressure(n, T_a, Y);
      for (int i = 0; i < 4; ++i) { prim[PYF + i] = Y[i]; }
      ps_.PrimToCon(prim, cons, bu, g3d);

      Real prim_out[NPRIM] = {0.0};
      auto res = ps_.ConToPrim(prim_out, cons, bu, g3d, g3u);

      const Real err_rho = prim_out[PRH]/prim[PRH] - 1.0;
      const Real err_T = (prim[PTM] != 0.0) ? (prim_out[PTM]/prim[PTM] - 1.0)
                                            : prim_out[PTM];
      const Real err_P = (prim[PPR] != 0.0) ? (prim_out[PPR]/prim[PPR] - 1.0)
                                            : prim_out[PPR];

      Kokkos::printf("%-14.5e %-14.6e %-14.6e %-14.6e %-14.6e %s\n",
                     n, err_rho, err_T, err_P, nu - 1.0,
                     (res.error == Primitive::Error::SUCCESS) ? "SUCCESS"
                     : ((res.error == Primitive::Error::NO_SOLUTION) ? "NO_SOLUTION"
                                                                    : "OTHER"));
    }
    // ------------------------------------------------------------------ Test D
    // The production failures do not sit exactly on the floor: D and tau carry small
    // excursions from advection/reconstruction. Perturb tau about its floor value at
    // exactly f = 1 and see how much of a *downward* excursion the solve tolerates.
    Kokkos::printf("\n--- Test D: f = 1 exactly, n = n_atm, tau scaled by (1+delta)\n");
    Kokkos::printf("%-14s %-16s %-14s %-14s %-14s %-14s %s\n",
                   "delta", "tau", "q", "nu-1", "fub", "|fub|/ub vs tol", "ConToPrim");
    {
      Real Y[MAX_SPECIES] = {0.0};
      Y[0] = 1.0; Y[1] = 1.0; Y[2] = Ya[2]; Y[3] = Ya[3];
      const Real D = n_a*mb_;
      const Real tau0 = eos.GetTauFloor(D, Y, 0.0);
      for (int k = 0; k < 11; ++k) {
        Real delta = 0.0;
        switch (k) {
          case 0:  delta =  1.0e-2;  break;
          case 1:  delta =  1.0e-4;  break;
          case 2:  delta =  1.0e-6;  break;
          case 3:  delta =  0.0;     break;
          case 4:  delta = -1.0e-6;  break;
          case 5:  delta = -1.0e-4;  break;
          case 6:  delta = -1.0e-3;  break;
          case 7:  delta = -1.0e-2;  break;
          case 8:  delta = -1.0e-1;  break;
          case 9:  delta = -0.5;     break;
          default: delta = -1.0;     break;
        }
        const Real tau = tau0*(1.0 + delta);
        const Real q = tau/D;

        // Replicate the root function at this q.
        const Real eoverD = 1.0 + q;
        Real That = eos.GetTemperatureFromInternalE(n_a, tau, Y);
        eos.ApplyTemperatureLimits(That);
        const Real ehat = eos.GetEnergy(n_a, That, Y);
        const Real Phat = eos.GetPressure(n_a, That, Y);
        const Real nu_a2 = (ehat + Phat)/(mb_*n_a);
        const Real nu_b2 = (D*eoverD + Phat)/D;
        const Real nu = Kokkos::fmax(nu_a2, nu_b2);
        const Real ub = 1.0/min_h_;
        const Real fub = ub - 1.0/nu;

        Real cons[NCONS] = {0.0};
        Real bu[NMAG] = {0.0, 0.0, 0.0};
        cons[CDN] = D;
        cons[CSX] = cons[CSY] = cons[CSZ] = 0.0;
        cons[CTA] = tau;
        for (int i = 0; i < 4; ++i) { cons[CYD + i] = D*Y[i]; }

        Real prim_out[NPRIM] = {0.0};
        auto res = ps_.ConToPrim(prim_out, cons, bu, g3d, g3u);

        Kokkos::printf("%-14.2e %-16.9e %-14.6e %-14.6e %-14.6e %-14.6e %s\n",
                       delta, tau, q, nu - 1.0, fub, Kokkos::fabs(fub)/ub,
                       (res.error == Primitive::Error::SUCCESS) ? "SUCCESS"
                       : ((res.error == Primitive::Error::NO_SOLUTION) ? "NO_SOLUTION"
                                                                      : "OTHER"));
      }
    }

    // ------------------------------------------------------------------ Test E
    // Explicit conserved states from <problem>/states_file -- real failing cells
    // lifted out of a production log. This is the only test that exercises a general
    // metric and a nonzero momentum, so it is the one that speaks to the production
    // failures directly.
    if (nst_ > 0) {
      Kokkos::printf("\n--- Test E: explicit conserved states from states_file\n");
      Kokkos::printf("%-4s %-13s %-12s %-12s %-12s %-13s %-13s %-5s %-13s %s\n",
                     "idx", "D", "1-Y0", "q=tau/D", "rsqr", "min_h-1", "h/mb-1",
                     "iter", "rho_out", "ConToPrim");
      for (int m = 0; m < nst_; ++m) {
        Real cons[NCONS] = {0.0};
        Real bu[NMAG] = {0.0, 0.0, 0.0};
        cons[CDN] = st_.d_view(m, 0);
        cons[CSX] = st_.d_view(m, 1);
        cons[CSY] = st_.d_view(m, 2);
        cons[CSZ] = st_.d_view(m, 3);
        cons[CTA] = st_.d_view(m, 4);
        for (int i = 0; i < 4; ++i) { cons[CYD + i] = st_.d_view(m, 5 + i); }
        bu[IBX] = st_.d_view(m, 9);
        bu[IBY] = st_.d_view(m, 10);
        bu[IBZ] = st_.d_view(m, 11);

        Real ge[NSPMETRIC], gi[NSPMETRIC];
        ge[S11] = st_.d_view(m, 12);
        ge[S12] = st_.d_view(m, 13);
        ge[S13] = st_.d_view(m, 14);
        ge[S22] = st_.d_view(m, 15);
        ge[S23] = st_.d_view(m, 16);
        ge[S33] = st_.d_view(m, 17);
        const Real dete = Primitive::GetDeterminant(ge);
        adm::SpatialInv(1.0/dete,
                        ge[S11], ge[S12], ge[S13], ge[S22], ge[S23], ge[S33],
                        &gi[S11], &gi[S12], &gi[S13], &gi[S22], &gi[S23], &gi[S33]);

        // Diagnostics the bracket depends on, formed the same way ConToPrim forms
        // them (primitive_solver.hpp:389-420): r_d = S_d/D, rsqr = g^ij r_i r_j.
        const Real Din = cons[CDN];
        const Real r_d[3] = {cons[CSX]/Din, cons[CSY]/Din, cons[CSZ]/Din};
        const Real rsqr =
            gi[S11]*r_d[0]*r_d[0] + 2.0*gi[S12]*r_d[0]*r_d[1]
          + 2.0*gi[S13]*r_d[0]*r_d[2] +   gi[S22]*r_d[1]*r_d[1]
          + 2.0*gi[S23]*r_d[1]*r_d[2] +   gi[S33]*r_d[2]*r_d[2];
        const Real q = cons[CTA]/Din;
        const Real Y0 = cons[CYD]/Din;

        // The enthalpy of this cell's own (frozen, generally non-equilibrium)
        // composition at n = D/mb. With bsqr = 0 the bracket sign test fails exactly
        // when this drops below min_h, so printing both side by side localises the
        // failure to the m_min_h bound rather than to the iteration.
        Real Ye[MAX_SPECIES] = {0.0};
        for (int i = 0; i < 4; ++i) { Ye[i] = cons[CYD + i]/Din; }
        const Real ne = Din/mb_;
        Real Te = eos.GetTemperatureFromInternalE(ne, cons[CTA], Ye);
        eos.ApplyTemperatureLimits(Te);
        const Real hmb = (eos.GetEnergy(ne, Te, Ye) + eos.GetPressure(ne, Te, Ye))
                         /(mb_*ne);

        Real prim_out[NPRIM] = {0.0};
        auto res = ps_.ConToPrim(prim_out, cons, bu, ge, gi);
        const char *tag = "OTHER";
        switch (res.error) {
          case Primitive::Error::SUCCESS:           tag = "SUCCESS";      break;
          case Primitive::Error::NO_SOLUTION:       tag = "NO_SOLUTION";  break;
          case Primitive::Error::NANS_IN_CONS:      tag = "NANS_IN_CONS"; break;
          case Primitive::Error::BRACKETING_FAILED: tag = "BRACKETING";   break;
          case Primitive::Error::CONS_FLOOR:        tag = "CONS_FLOOR";   break;
          case Primitive::Error::PRIM_FLOOR:        tag = "PRIM_FLOOR";   break;
          case Primitive::Error::MAG_TOO_BIG:       tag = "MAG_TOO_BIG";  break;
          default:                                  tag = "OTHER";        break;
        }
        Kokkos::printf("%-4d %-13.6e %-12.4e %-12.4e %-12.4e %-13.6e %-13.6e %-5d "
                       "%-13.6e %s\n",
                       m, Din, 1.0 - Y0, q, rsqr, min_h_ - 1.0, hmb - 1.0,
                       res.iterations, prim_out[PRH]*mb_, tag);
      }
    }

    // ------------------------------------------------------------------ Test F
    // Bag_B is private to EOSZlaBag, so measure it from the EOS itself rather than
    // reconstructing it from parfile values: in the pure-quark limit at vanishing
    // density every term of ColdPressureQuarks tends to zero except -Bag_B, so
    // -P(n->0, f=0) *is* Bag_B in whatever units the code actually uses. That also
    // sidesteps any unit/normalisation ambiguity in comparing against 160^4/hbar^3.
    Real Yq_pure[MAX_SPECIES] = {0.0, 0.0, Ya[2], Ya[3]};
    const Real bagB_ = -eos.GetPressure(n_a*1.0e-6, T_a, Yq_pure);

    // Is the total pressure negative at the real failing states, and if so, is there
    // a well-defined minimum phase fraction that recovers a positive pressure?
    //
    // The mechanism under test: ColdPressureQuarks carries the *absolute* constant
    // -Bag_B (eos_zla_bag.hpp), so as the quark-phase density falls the quark pressure
    // tends to -Bag_B rather than to zero. ColdPressure then mixes the phases linearly
    // in the volume fraction, P = P_N*f + P_Q*(1-f), so at atmosphere densities -- where
    // P_N is ~30 decades below Bag_B -- any resolvable (1-f) drives the total pressure
    // negative. A frozen composition can legitimately sit there, which is why snapping
    // on density alone would be wrong.
    if (nst_ > 0) {
      Kokkos::printf("\n--- Test F1: pressure decomposition at the failing states\n");
      Kokkos::printf("%-4s %-12s %-12s %-12s %-12s %-13s %-13s %-13s %s\n",
                     "idx", "n", "1-f", "1-yn", "nQ/n", "P_N", "P_Q", "P_total", "sign");
      int nneg = 0;
      for (int m = 0; m < nst_; ++m) {
        const Real Din = st_.d_view(m, 0);
        Real Yf[MAX_SPECIES] = {0.0};
        for (int i = 0; i < 4; ++i) { Yf[i] = st_.d_view(m, 5 + i)/Din; }
        const Real nf = Din/mb_;
        const Real f = Yf[0];
        const Real yn = Yf[1];
        const Real omf = 1.0 - f;
        const Real nQ_over_n = (omf > 0.0) ? (1.0 - yn)/omf : 0.0;

        // Single-phase pressures at this cell's own phase densities, so the two
        // contributions to ColdPressure can be read separately.
        Real Yn1[MAX_SPECIES] = {1.0, 1.0, Yf[2], Yf[3]};   // pure nucleonic
        Real Yq0[MAX_SPECIES] = {0.0, 0.0, Yf[2], Yf[3]};   // pure quark
        const Real P_N = eos.GetPressure(nf, T_a, Yn1);
        const Real P_Q = eos.GetPressure(nf*fmax(nQ_over_n, 1e-300), T_a, Yq0);
        const Real P_t = eos.GetPressure(nf, T_a, Yf);
        if (P_t < 0.0) { nneg++; }
        if (m < 12) {
          Kokkos::printf("%-4d %-12.4e %-12.4e %-12.4e %-12.4e %-13.5e %-13.5e "
                         "%-13.5e %s\n", m, nf, omf, 1.0 - yn, nQ_over_n,
                         P_N, P_Q, P_t, (P_t < 0.0) ? "NEGATIVE" : "positive");
        }
      }
      Kokkos::printf("  P_total < 0 in %d of %d states (%.2f%%)\n",
                     nneg, nst_, 100.0*static_cast<Real>(nneg)/nst_);

      // Robustness to the unknown Lorentz factor: C2P failed at these cells, so n is
      // only known up to W. Sweep n by decades about D/mb and re-count.
      Kokkos::printf("  n-sensitivity (n scaled about D/mb):\n");
      for (int s = -2; s <= 2; ++s) {
        Real scale = 1.0;
        for (int j = 0; j < (s < 0 ? -s : s); ++j) { scale *= (s < 0) ? 0.1 : 10.0; }
        int nn = 0;
        for (int m = 0; m < nst_; ++m) {
          const Real Din = st_.d_view(m, 0);
          Real Yf[MAX_SPECIES] = {0.0};
          for (int i = 0; i < 4; ++i) { Yf[i] = st_.d_view(m, 5 + i)/Din; }
          if (eos.GetPressure(Din/mb_*scale, T_a, Yf) < 0.0) { nn++; }
        }
        Kokkos::printf("    n x %-8.1e : P<0 in %d/%d (%.2f%%)\n",
                       scale, nn, nst_, 100.0*static_cast<Real>(nn)/nst_);
      }
    }

    // Test F2: map P(n, f) and locate the minimum f that gives P > 0.
    // Monotonicity of P in f is not obvious -- ConvertPrimitive sets
    // nY[0] = yn*n/f and nY[1] = (1-yn)*n/(1-f), so both phase densities move with f.
    // Convention (a) holds nQ/n fixed at the observed value; (b) holds yn fixed.
    Kokkos::printf("\n--- Test F2: P(n, 1-f) map and f_min, two conventions\n");
    Kokkos::printf("%-13s %-14s %-14s %-9s %-14s %s\n",
                   "n", "(a)1-f_cross", "(b)1-f_cross", "P_N(n)", "P_N/Bag_B",
                   "(a)monotone");
    for (int in = 0; in < 15; ++in) {
      Real nn = n_a;
      for (int j = 0; j < in; ++j) { nn *= 10.0; }
      Real Yn1[MAX_SPECIES] = {1.0, 1.0, Ya[2], Ya[3]};
      const Real PN = eos.GetPressure(nn, T_a, Yn1);
      Real cross_a = -1.0, cross_b = -1.0;
      bool mono_a = true;
      Real prev_a = -1e300;
      // sweep 1-f downward from 0.5 to 1e-12, 8 points/decade
      for (int k = 0; k <= 96; ++k) {
        Real omf = 0.5;
        for (int j = 0; j < k; ++j) { omf *= 0.75; }
        if (omf < 1e-12) { break; }
        // (a) hold nQ/n = 1.2579 (the median of the failing population)
        Real Ya_[MAX_SPECIES] = {1.0 - omf, 1.0 - 1.2579*omf, Ya[2], Ya[3]};
        const Real Pa = eos.GetPressure(nn, T_a, Ya_);
        // (b) hold yn fixed at the same distance from 1 as the largest omf
        Real Yb_[MAX_SPECIES] = {1.0 - omf, 1.0 - 0.5*1.2579, Ya[2], Ya[3]};
        const Real Pb = eos.GetPressure(nn, T_a, Yb_);
        if (Pa < prev_a) { mono_a = false; }
        prev_a = Pa;
        if (cross_a < 0.0 && Pa > 0.0) { cross_a = omf; }
        if (cross_b < 0.0 && Pb > 0.0) { cross_b = omf; }
      }
      Kokkos::printf("%-13.4e %-14.4e %-14.4e %-9.2e %-14.4e %s\n",
                     nn, cross_a, cross_b, PN, PN/bagB_,
                     mono_a ? "yes" : "NO");
    }
    Kokkos::printf("  Bag_B = %.9e (code units)\n", bagB_);
    Kokkos::printf("  closed-form prediction 1-f_min ~ P_N/(Bag_B + P_N); compare col 2\n");

    // Test F3: SAFETY -- would a pressure floor eat the star's genuine mixed phase?
    // rev600 held f-min = 0.7597 through 600 M, i.e. a real quark fraction of ~0.24 at
    // stellar densities. If P > 0 there, the floor never fires on physical material and
    // the mixed phase is untouched. This is the check the density-gate proposal failed.
    Kokkos::printf("\n--- Test F3: is the genuine mixed phase safe? (P at f = f-min obs.)\n");
    Kokkos::printf("%-13s %-12s %-14s %-14s %s\n",
                   "n", "1-f", "P_total", "1-f_max(n)", "verdict");
    for (int in = 8; in < 16; ++in) {
      Real nn = n_a;
      for (int j = 0; j < in; ++j) { nn *= 10.0; }
      Real Ym[MAX_SPECIES] = {0.7597, 1.0 - 1.2579*0.2403, Ya[2], Ya[3]};
      const Real Pm = eos.GetPressure(nn, T_a, Ym);
      Real Yn1[MAX_SPECIES] = {1.0, 1.0, Ya[2], Ya[3]};
      const Real PN = eos.GetPressure(nn, T_a, Yn1);
      const Real omf_max = PN/(bagB_ + PN);
      Kokkos::printf("%-13.4e %-12.4e %-14.5e %-14.4e %s\n",
                     nn, 0.2403, Pm, omf_max,
                     (Pm > 0.0) ? "SAFE (P>0)" : "floor would fire");
    }

    // Test F4: with the floor active, verify on the real failing states that
    //   (i)  the corrected composition still reconstructs the total density,
    //        n == f*nY[0] + (1-f)*nQ, and
    //   (ii) the resulting pressure is at or above the floor.
    // (i) is the invariant that a naive "hold both phase densities fixed" correction
    // would break; checking it here rather than trusting the algebra is the point.
    if (nst_ > 0) {
      Kokkos::printf("\n--- Test F4: post-floor consistency on the failing states\n");
      Real worst_n = 0.0, worst_P = 1.0e300;
      int nneg = 0;
      for (int m = 0; m < nst_; ++m) {
        const Real Din = st_.d_view(m, 0);
        Real Yf[MAX_SPECIES] = {0.0};
        for (int i = 0; i < 4; ++i) { Yf[i] = st_.d_view(m, 5 + i)/Din; }
        const Real nf = Din/mb_;
        Real nY[6] = {0.0};
        eos.GetPhaseDecomposition(nf, Yf, nY);
        // n reconstructed from the *corrected* phase densities and volume fraction.
        const Real n_rec = nY[4]*nY[0] + (1.0 - nY[4])*nY[1];
        const Real relerr = fabs(n_rec - nf)/nf;
        if (relerr > worst_n) { worst_n = relerr; }
        const Real P = eos.GetPressure(nf, T_a, Yf);
        if (P < worst_P) { worst_P = P; }
        if (P < 0.0) { nneg++; }
      }
      Kokkos::printf("  worst |n_rec/n - 1| over %d states : %.3e\n", nst_, worst_n);
      Kokkos::printf("  minimum P over %d states          : %.6e\n", nst_, worst_P);
      Kokkos::printf("  states with P < 0                  : %d of %d\n", nneg, nst_);
    }

    // ------------------------------------------------------------------ Test G
    // C2P convergence across the first-order quark-hadron transition plateau.
    //
    // A *different* failure from the atmosphere modes above. The bracket is healthy --
    // min_h - 1 = 0 and h/mb - 1 is a comfortable 4e-02 .. 8e-02 -- and the solve fails
    // only because FalsePosition runs out of iterations. Across the transition the EOS
    // has a near-plateau where dP/dn is very small, so the root function is nearly flat,
    // false position converges linearly at best, and |x-xold|/x <= tol is not reached
    // within c2p_iter steps.
    //
    // Measured on 300 real failing cells from a BNS run (m1.4_g0_B160_frozen_v2,
    // segments 0001-0002, t = 33 .. 2329):
    //     c2p_iter =  50 -> 248 NO_SOLUTION        c2p_iter = 300 -> 5
    //     c2p_iter = 500 ->   3 NO_SOLUTION        c2p_iter = 800 -> 1
    // Every recovered state returned the same density, 4.325724e-04 -- the plateau
    // itself. The failing cells occupied a density band only 0.7% wide; perturbing D by
    // 0.1% moved all of them off it, while perturbing tau by 0.1% did not, identifying
    // D -- i.e. where the root lands relative to the flat region -- as the controlling
    // variable. Zeroing the momentum also cures them, so the mode needs bulk motion:
    // a *static* star never triggers it, which is why a TOV test stays silent no matter
    // how long it runs even though it spans the plateau density.
    //
    // The states below are real failing cells copied verbatim from that run, spanning
    // the observed 1-f range, so this test is self-contained and reproduces the mode
    // deterministically with no external state file. Difficulty tracks 1-f: at
    // c2p_iter = 50 the two cells with 1-f <= 2.5e-03 still converge and the four with
    // 1-f >= 5.6e-03 do not, consistent with a larger quark fraction flattening the
    // root function further.
    {
      Kokkos::printf("\n--- Test G: C2P across the transition plateau (embedded cells)\n");
      Kokkos::printf("  c2p_iter = %d, c2p_tol = %.3g\n",
                     ps_.GetRootSolver().iterations, ps_.tol);
      const Real gstates[6][18] = {
        {0.00043475771251898167, 2.6009584932404285e-05, 5.3512949404931102e-05, -2.4834079778579053e-06, 9.468640619370202e-06, 0.00043450583922652114, 0.00043445354482703603, 1.9261263464817445e-05, -2.7240357366356378e-07, 0, 0, 0, 1.7146247032735964, -0.0057249745572215778, -0.016154103865512581, 1.7510860764961051, -0.036268727254053661, 1.709912250604622},   // 1-f = 5.793e-04
        {0.00043472572878742316, 5.3123510303486221e-05, -2.6009804483774832e-05, 1.0141593647907223e-06, 1.1865375302391179e-05, 0.00043365532129966864, 0.00043338858319889271, 2.3315747930320253e-05, -1.1783856912121293e-06, 0, 0, 0, 1.683273638150907, -0.017212724915685575, -0.039816724924499877, 1.8060567882448626, 0.021359859969107432, 1.7404813866550435},   // 1-f = 2.462e-03
        {0.00043531721273746101, 1.51812243186332e-05, -6.659609472174653e-05, 2.7053198107580913e-06, 1.6393258709163403e-05, 0.00043287265482917943, 0.00043247177435936577, 2.1915211063800995e-05, -2.8277634358198168e-06, 0, 0, -0, 1.7605606289719247, 0.095500348557899323, 0.00087265682488049114, 1.7925172472519832, -0.0045404556711991654, 1.7725654442390744},   // 1-f = 5.616e-03
        {0.00043484468783775088, -3.6553941596465079e-06, 5.9346720203812824e-05, -4.6121771440021529e-06, 1.3044830384508817e-05, 0.00043113405648466092, 0.00043041395777376144, 2.4802041417166189e-05, -4.0039453079399974e-06, 0, 0, 0, 1.7646795470035141, -0.010221492054417837, -0.001192290989202484, 1.6311158266641332, -0.036175279535092406, 1.7377426180506876},   // 1-f = 8.533e-03
        {0.00043447862474202284, 4.9041700008720206e-05, 2.9192451289629175e-05, 2.3616200245098488e-06, 1.3896042460211634e-05, 0.00042908452160823034, 0.00042806470170674547, 2.6725299307927038e-05, -6.1543149668035717e-06, 0, 0, 0, 1.808854537083262, 0.003675721592967801, -0.025132102248305529, 1.736789748074522, -0.011505899854487123, 1.7296424658906417},   // 1-f = 1.242e-02
        {0.00043503379897151111, 6.1627771771761853e-05, 6.1791616074533441e-06, -7.651782579729833e-06, 1.5044132392034945e-05, 0.00042679896234994798, 0.00042520045638980218, 2.8669065563611428e-05, -7.234241372423834e-06, 0, 0, 0, 1.6349851490100573, 0.0097208265555862924, -0.036485326300134791, 1.7747140643096959, 0.0017974665640484312, 1.7452510237704277},   // 1-f = 1.893e-02
      };
      Kokkos::printf("%-4s %-13s %-12s %-13s %-13s %-6s %s\n",
                     "idx", "D", "1-f", "h/mb-1", "rho_out", "iter", "ConToPrim");
      int nfail = 0;
      for (int m = 0; m < 6; ++m) {
        Real cons[NCONS] = {0.0};
        Real bu[NMAG] = {0.0, 0.0, 0.0};
        cons[CDN] = gstates[m][0];
        cons[CSX] = gstates[m][1];
        cons[CSY] = gstates[m][2];
        cons[CSZ] = gstates[m][3];
        cons[CTA] = gstates[m][4];
        for (int i = 0; i < 4; ++i) { cons[CYD + i] = gstates[m][5 + i]; }
        Real ge[NSPMETRIC], gi[NSPMETRIC];
        ge[S11] = gstates[m][12]; ge[S12] = gstates[m][13]; ge[S13] = gstates[m][14];
        ge[S22] = gstates[m][15]; ge[S23] = gstates[m][16]; ge[S33] = gstates[m][17];
        const Real dete = Primitive::GetDeterminant(ge);
        adm::SpatialInv(1.0/dete,
                        ge[S11], ge[S12], ge[S13], ge[S22], ge[S23], ge[S33],
                        &gi[S11], &gi[S12], &gi[S13], &gi[S22], &gi[S23], &gi[S33]);
        const Real Din = cons[CDN];
        Real Ye[MAX_SPECIES] = {0.0};
        for (int i = 0; i < 4; ++i) { Ye[i] = cons[CYD + i]/Din; }
        const Real ne = Din/mb_;
        Real Te = eos.GetTemperatureFromInternalE(ne, cons[CTA], Ye);
        eos.ApplyTemperatureLimits(Te);
        const Real hmb = (eos.GetEnergy(ne, Te, Ye) + eos.GetPressure(ne, Te, Ye))
                         /(mb_*ne);
        Real prim_out[NPRIM] = {0.0};
        auto res = ps_.ConToPrim(prim_out, cons, bu, ge, gi);
        const bool ok = (res.error == Primitive::Error::SUCCESS);
        if (!ok) { nfail++; }
        Kokkos::printf("%-4d %-13.6e %-12.4e %-13.6e %-13.6e %-6d %s\n",
                       m, Din, 1.0 - Ye[0], hmb - 1.0, prim_out[PRH]*mb_,
                       res.iterations, ok ? "SUCCESS" : "NO_SOLUTION");
      }
      Kokkos::printf("  Test G: %d of 6 failed to invert "
                     "(expect 4 at c2p_iter=50, 0 at c2p_iter>=300)\n", nfail);
    }

    // ------------------------------------------------------------------ Test H
    // y_lQ conditioning, and conservation of int(D*Y_3) at the noise floor.
    //
    // y_lQ is never advected. It is derived as Y[3]/(1 - yn), a ratio of two
    // independently advected conserved quantities, so once (1 - yn) reaches the
    // round-off floor the quotient is amplified noise. In a production BNS run it
    // saturated at exactly the -1 species bound on 1.09e9 cell-visits with (1 - Y_N)
    // down to 1e-9, and the clamp that produced it -- written back into the conserved
    // variables -- drained 72% of int(D*(1-Y_N)*y_lQ).
    //
    // H1: the conditioning contract. Faithful recovery where (1 - yn) is resolved,
    //     quark phase declared absent below yn_snap, genuine mixed phase untouched.
    // H2: the conservation contract, which is deliberately asymmetric:
    //       (a) at the NOISE FLOOR an out-of-band y_lQ must not move D*Y_3 at all --
    //           this is the measured sink, and yn_snap is what closes it;
    //       (b) in RESOLVED material an in-band y_lQ must not move D*Y_3 either;
    //       (c) in RESOLVED material an out-of-band y_lQ SHOULD be clamped -- that is
    //           the species limiter doing its job -- and the clamp may reach cons via
    //           PrimToCon when a floor also fires. Asserted as a documented allowance,
    //           not a pass.
    {
      Kokkos::printf("\n--- Test H1: y_lQ = Y[3]/(1-yn) conditioning across yn_snap\n");
      Kokkos::printf("%-12s %-14s %-15s %-15s %s\n",
                     "1-yn", "Y[3]", "y_lQ recovered", "y_lQ wanted", "verdict");
      const Real ylq_want = -0.5;      // inside the table range [-0.679, 0]
      // Well inside the mixed phase: n_tr = 0.2538, n_c = 0.4114 (table units).
      const Real n_h = 0.45;
      const Real snap = eos.GetYnSnap();
      int h1bad = 0;
      for (int e = 2; e <= 12; ++e) {
        Real omyn = 1.0;
        for (int q = 0; q < e; ++q) { omyn *= 0.1; }
        Real Yh[MAX_SPECIES] = {0.0};
        Yh[0] = 0.75;                          // f, comfortably mixed
        Yh[1] = 1.0 - omyn;                    // Y_N
        Yh[2] = 0.10*Yh[1];                    // Y_N * y_lN
        Yh[3] = ylq_want*omyn;                 // (1-Y_N) * y_lQ
        Real nY[6] = {0.0};
        eos.GetPhaseDecomposition(n_h, Yh, nY);
        const Real got = nY[3];
        // Forming Yh[1] = 1 - omyn and recovering (1 - Yh[1]) is a catastrophic
        // cancellation: the recovered denominator carries an absolute error ~1e-16,
        // hence a relative error ~1e-16/omyn. Tolerate exactly that much and no more --
        // this IS the amplification the guard exists to bound.
        const Real tol = 1.0e-15/omyn + 1.0e-12;
        // Within a factor of 2 of the threshold the branch taken depends on that same
        // round-off, so do not predict it; report it instead of scoring it.
        const bool boundary = (omyn > 0.5*snap) && (omyn < 2.0*snap);
        const bool resolved = (omyn > snap);
        bool ok = true;
        const char *verdict;
        if (boundary) {
          verdict = (got == 0.0) ? "boundary (absent)" : "boundary (resolved)";
        } else if (resolved) {
          ok = (fabs(got - ylq_want) < tol);
          verdict = ok ? "ok (resolved)" : "FAIL (not recovered)";
        } else {
          ok = (got == 0.0);
          verdict = ok ? "ok (absent)" : "FAIL (divided by noise)";
        }
        if (!ok) { h1bad++; }
        Kokkos::printf("%-12.1e %-14.6e %-15.8e %-15.8e %s\n",
                       omyn, Yh[3], got, ylq_want, verdict);
      }
      // Out-of-range y_lQ at the noise floor must not reach the bag term.
      {
        Real Yh[MAX_SPECIES] = {0.0};
        const Real omyn = 1.0e-11;
        Yh[0] = 0.75; Yh[1] = 1.0 - omyn; Yh[2] = 0.10*Yh[1];
        Yh[3] = -5.0*omyn;                     // y_lQ = -5, far outside [-1, 2]
        Real nY[6] = {0.0};
        eos.GetPhaseDecomposition(n_h, Yh, nY);
        const bool ok = (nY[3] == 0.0);
        if (!ok) { h1bad++; }
        Kokkos::printf("  y_lQ = -5 at 1-yn = 1e-11 -> nY[3] = %.8e  %s\n",
                       nY[3], ok ? "ok (absent)" : "FAIL (noise reached the bag term)");
      }
      // The genuine mixed phase must be completely untouched by the guard.
      {
        Real Yh[MAX_SPECIES] = {0.0};
        const Real omyn = 0.2403;              // the observed f-min mixed phase
        Yh[0] = 1.0 - omyn; Yh[1] = 1.0 - omyn; Yh[2] = 0.10*Yh[1];
        Yh[3] = ylq_want*omyn;
        Real nY[6] = {0.0};
        eos.GetPhaseDecomposition(n_h, Yh, nY);
        const bool ok = (fabs(nY[3] - ylq_want) < 1.0e-12);
        if (!ok) { h1bad++; }
        Kokkos::printf("  genuine mixed phase (1-yn = 0.2403) -> y_lQ = %.8e  %s\n",
                       nY[3], ok ? "ok (untouched)" : "FAIL (guard ate real material)");
      }
      Kokkos::printf("  Test H1: %d failure(s) (expect 0)\n", h1bad);

      Kokkos::printf("\n--- Test H2: does a species clamp move the conserved D*Y_3?\n");
      Kokkos::printf("%-10s %-9s %-16s %-16s %-11s %s\n",
                     "1-Y_N", "y_lQ", "D*Y_3 before", "D*Y_3 after", "rel change",
                     "verdict");
      int h2bad = 0;
      const Real n_h2 = 0.45;
      for (int e = 0; e < 7; ++e) {
        const Real omyn_c[7] = {1.0e-11, 1.0e-11, 0.25, 0.25, 1.0e-11, 0.25, 0.25};
        const Real ylq_c[7]  = {-3.0,    -0.5,    -0.5, -3.0, -1.0e11, -0.5, -0.5};
        // rows 5,6 sit BELOW the density floors, so the atmosphere reset (Y := Y_atm,
        // and s3_atmosphere = 0) must delete D*Y_3 outright -- the floor channel.
        // rows 0,1 : noise floor, |Y[3]| inside the FLOORED band -> must never move
        //            (this is the measured sink)
        // row  2   : resolved, in band -> must never move
        // row  3   : resolved, out of band -> clamp is CORRECT; cons may follow
        // row  4   : noise floor with a RUNAWAY Y[3] (y_lQ = -1e11, i.e. Y[3] = -1) ->
        //            must be CLAMPED and BOUNDED. Skipping the test here is what let
        //            v7 inflate the atmosphere pressure and go NaN.
        const Real omyn = omyn_c[e];
        // The state must satisfy Y_N <= f, which ResetFloorZlaBag enforces first: its
        // cascade clamps Y[1] to Y_max[1]*Y[0] BEFORE evaluating Y[3]'s band. Violating
        // it means Y[1] is repaired to f and (1 - Y[1]) becomes O(1 - f), so the row
        // never reaches the noise floor it is meant to probe. Keep 1 - f just under
        // 1 - Y_N so the quark BARYON fraction is the small quantity.
        Real Yh[MAX_SPECIES] = {0.0};
        Yh[0] = 1.0 - 0.5*omyn;                // f
        Yh[1] = 1.0 - omyn;                    // Y_N <= f
        Yh[2] = 0.10*Yh[1];
        Yh[3] = ylq_c[e]*omyn;
        Real prim_in[NPRIM] = {0.0};
        prim_in[PRH] = (e >= 5) ? ((e == 5) ? n_a*0.1 : n_a*0.5) : n_h2;
        prim_in[PVX] = 0.0; prim_in[PVY] = 0.0; prim_in[PVZ] = 0.0;
        prim_in[PTM] = T_a;
        for (int i = 0; i < 4; ++i) { prim_in[PYF + i] = Yh[i]; }
        prim_in[PPR] = eos.GetPressure(prim_in[PRH], prim_in[PTM], &prim_in[PYF]);
        Real gflat[NSPMETRIC]  = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0};
        Real giflat[NSPMETRIC] = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0};
        Real bu[NMAG] = {0.0, 0.0, 0.0};
        Real cons[NCONS] = {0.0};
        ps_.PrimToCon(prim_in, cons, bu, gflat);
        const Real before = cons[CYD + 3];
        Real prim_out[NPRIM] = {0.0};
        auto res = ps_.ConToPrim(prim_out, cons, bu, gflat, giflat);
        const Real after = cons[CYD + 3];
        const Real rel = (before != 0.0) ? fabs(after - before)/fabs(before) : 0.0;
        bool ok;
        const char *verdict;
        if (e == 3) {
          // Resolved material, genuinely out of band: clamping is the limiter working.
          ok = (res.error == Primitive::Error::SUCCESS);
          verdict = ok ? "allowed (real clamp)" : "FAIL (no solution)";
        } else if (e == 4) {
          // Runaway at the noise floor: cons MUST move, and the result must land inside
          // the floored band, |Y[3]| <= max(|Y_min[3]|, Y_max[3]) * q_snap.
          const Real Dn = cons[CDN];
          const Real y3_out = (Dn != 0.0) ? after/Dn : 0.0;
          const Real cap = 2.0*eos.GetYnSnap()*1.000001;
          ok = isfinite(y3_out) && (fabs(y3_out) <= cap) &&
               (res.error == Primitive::Error::SUCCESS);
          verdict = ok ? "ok (bounded)" : "FAIL (Y[3] unbounded -> NaN risk)";
        } else if (e >= 5) {
          // Below the floor the atmosphere reset is INTENDED; report the channel and
          // the size of the deletion rather than scoring it as a failure.
          ok = (res.error == Primitive::Error::SUCCESS);
          verdict = ok ? "floor reset (expected)" : "FAIL (no solution)";
        } else {
          ok = (rel < 1.0e-14) && (res.error == Primitive::Error::SUCCESS);
          verdict = ok ? "ok (conserved)" : "FAIL (sink still open)";
        }
        if (!ok) { h2bad++; }
        Kokkos::printf("%-10.1e %-9.2g %-16.9e %-16.9e %-11.3e %s "
                       "[1-f=%.1e cfl=%d pfl=%d spadj=%d chan=%d]\n",
                       omyn, ylq_c[e], before, after, rel, verdict, 1.0 - Yh[0],
                       static_cast<int>(res.cons_floor),
                       static_cast<int>(res.prim_floor),
                       static_cast<int>(res.species_adjusted),
                       res.y3_chan);
      }
      Kokkos::printf("  Test H2: %d failure(s) (expect 0)\n", h2bad);
    }

    // ------------------------------------------------------------------ Test I
    // Is an UNBOUNDED Y[3] safe once the quark phase is declared absent?
    //
    // yn_snap stops nY[3] = Y[3]/(1-yn) being derived from noise, and the matching
    // skip in SpeciesLimits stops Y[3] being clamped there. But ConvertPrimitive still
    // computes nY[5] = Y[2] + Y[3] UNCONDITIONALLY, so the total charge fraction keeps
    // consuming the raw Y[3]. If nothing bounds Y[3], nY[5] inherits whatever advection
    // put there. Measured in v7: |y_lQ| up to 1.06e+03 in cells the clamp still reached,
    // and the run produced 411,155 NANS_IN_CONS at t = 779.25, 29 M after restart.
    // This test asks the EOS directly whether a large Y[3] is survivable.
    {
      Kokkos::printf("\n--- Test I: EOS response to an unbounded Y[3] "
                     "(quark phase declared absent)\n");
      Kokkos::printf("%-10s %-12s %-15s %-15s %-15s %s\n",
                     "1-yn", "Y[3]", "nY[5]=Y2+Y3", "P", "e", "finite?");
      int nbad = 0;
      const Real omyn_i = 1.0e-9;          // below yn_snap: phase declared absent
      for (int e = 0; e < 6; ++e) {
        const Real y3v[6] = {1.0e-9, 1.0e-3, 1.0, 1.0e1, 1.0e3, 1.0e6};
        Real Yi[MAX_SPECIES] = {0.0};
        Yi[0] = 1.0 - 1.0e-9;              // f -> 1, the envelope
        Yi[1] = 1.0 - omyn_i;              // Y_N -> 1
        Yi[2] = 0.10*Yi[1];
        Yi[3] = -y3v[e];                   // unbounded, negative as in the run
        Real nY[6] = {0.0};
        eos.GetPhaseDecomposition(n_a, Yi, nY);
        const Real P = eos.GetPressure(n_a, T_a, Yi);
        const Real en = eos.GetEnergy(n_a, T_a, Yi);
        const bool fin = isfinite(P) && isfinite(en) && isfinite(nY[5]);
        if (!fin) { nbad++; }
        Kokkos::printf("%-10.1e %-12.1e %-15.6e %-15.6e %-15.6e %s\n",
                       omyn_i, Yi[3], nY[5], P, en, fin ? "yes" : "NON-FINITE");
      }
      Kokkos::printf("  Test I: %d of 6 non-finite  "
                     "(any > 0 means nY[5] must exclude an absent phase)\n", nbad);
    }

    // ------------------------------------------------------------------ Test K
    // Does zla_pfloor's yn rescaling amplify y_lQ?
    //
    // The floor raises f and rescales yn so that nQ is held fixed and the baryon
    // budget balances: (1-yn_new) = (1-yn_old)*(1-f_new)/(1-f_old). But Y[2] and Y[3]
    // are const, so ConvertPrimitive then forms y_lQ = Y[3]/(1-yn_new) with the NEW
    // denominator and the OLD numerator, giving
    //     y_lQ_new = y_lQ_old * (1-f_old)/(1-f_new).
    // Physically the leptons move with the baryons that were transferred Q->N, which
    // leaves y_lQ INVARIANT. Predicted amplification A = (1-f_old)/(1-f_new); observed
    // A is printed alongside so the two can be compared directly.
    {
      Kokkos::printf("\n--- Test K: does the pressure floor amplify y_lQ?\n");
      Kokkos::printf("%-12s %-10s %-10s %-13s %-13s %-9s %-9s %s\n",
                     "1-f_in", "f_in", "f_out", "y_lQ_in", "y_lQ_out", "A_obs",
                     "A_pred", "verdict");
      const Real ylq0 = -0.30;          // inside the table range [-0.679, 0]
      const Real omyn0 = 1.0e-2;        // measured median (1-Y_N) in the injection band
      int kbad = 0;
      for (int e = 0; e < 8; ++e) {
        // Fixed n just above n_pfloor_force, sweeping the QUARK VOLUME fraction (1-f)
        // upward. With pfloor_frac = 0 the floor engages only once the mixture
        // pressure would go negative, i.e. once (1-f)*|P_Q| overcomes f*P_N, so the
        // quark fraction is the control parameter, not n.
        const Real omf_k[8] = {1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 2e-1, 3e-1, 5e-1};
        const Real nk_fixed = 0.200;
        const Real nk[8] = {nk_fixed,nk_fixed,nk_fixed,nk_fixed,
                            nk_fixed,nk_fixed,nk_fixed,nk_fixed};
        Real Yk[MAX_SPECIES] = {0.0};
        Yk[0] = 1.0 - omf_k[e];         // f
        Yk[1] = 1.0 - 0.5*omf_k[e];     // Y_N <= f, so (1-Y_N) = 0.5*(1-f)
        Yk[2] = 0.10*Yk[1];
        Yk[3] = ylq0*(1.0 - Yk[1]);     // (1-Y_N) * y_lQ
        Real nY[6] = {0.0};
        eos.GetPhaseDecomposition(nk[e], Yk, nY);
        const Real f_out = nY[4];
        const Real ylq_out = nY[3];
        const Real A_obs  = (ylq0 != 0.0) ? ylq_out/ylq0 : 0.0;
        const Real A_pred = (1.0 - f_out > 0.0) ? (1.0 - Yk[0])/(1.0 - f_out) : 0.0;
        const bool moved = (f_out > Yk[0]*(1.0 + 1.0e-12));
        // CONTRACT (post-fix): raising f must leave y_lQ INVARIANT, because the
        // leptons move with the baryons transferred Q->N. A_pred is the amplification
        // the OLD code produced (y_lQ * (1-f_old)/(1-f_new)); it is printed only to
        // show how large the bias was, and must NOT be matched.
        const bool ok = (fabs(A_obs - 1.0) < 1.0e-9);
        if (!ok) { kbad++; }
        Kokkos::printf("%-12.2e %-10.6f %-10.6f %-13.6e %-13.6e %-9.3f %-9.3f %s\n",
                       1.0 - Yk[0], Yk[0], f_out, ylq0, ylq_out, A_obs, A_pred,
                       ok ? (moved ? "ok (f moved, y_lQ invariant)" : "ok (floor inactive)")
                          : "FAIL (y_lQ amplified)");
      }
      Kokkos::printf("  Test K: %d failure(s) -- y_lQ must be invariant under the "
                     "floor (expect 0)\n", kbad);
      Kokkos::printf("  A_pred is the OLD bias, shown for scale; it must no longer occur.\n");
      Kokkos::printf("  (physical y_lQ range is [-0.679, 0]; species bound is [-1, 2])\n");
    }

    // ------------------------------------------------------------------ Test L
    // Is a NEGATIVE quark density reachable, and is it fatal?
    //
    // SnapPhaseFraction guards yqQ against omyn0 = 1-yn at/below the noise floor but
    // computes nQ = omyn0*n/omf with no guard, while the error policy guards the same
    // quantity as fmax(1-Y[1], q_snap). Y[1] > 1 cannot reach C2P (SpeciesLimits clamps
    // first) but CAN reach the EOS via reconstructed w0 states in the flux path, where
    // WENOZ overshoots at extrema. Then nQ < 0 flows into ColdPressureQuarks.
    {
      Kokkos::printf("\n--- Test L: EOS response to yn > 1 (negative quark density)\n");
      Kokkos::printf("%-12s %-13s %-13s %-14s %-14s %s\n",
                     "1-yn", "nQ", "y_lQ(nY3)", "P", "e", "finite?");
      int lbad = 0;
      const Real nL = 0.200;
      for (int e = 0; e < 6; ++e) {
        const Real omyn[6] = {1.0e-2, 1.0e-6, 0.0, -1.0e-12, -1.0e-6, -1.0e-2};
        Real Yl[MAX_SPECIES] = {0.0};
        Yl[0] = 0.90;
        Yl[1] = 1.0 - omyn[e];
        Yl[2] = 0.10;
        Yl[3] = -0.30*omyn[e];
        Real nY[6] = {0.0};
        eos.GetPhaseDecomposition(nL, Yl, nY);
        const Real P  = eos.GetPressure(nL, T_a, Yl);
        const Real en = eos.GetEnergy(nL, T_a, Yl);
        const bool fin = isfinite(P) && isfinite(en) && isfinite(nY[1]) && isfinite(nY[3]);
        if (!fin) { lbad++; }
        Kokkos::printf("%-12.1e %-13.5e %-13.5e %-14.6e %-14.6e %s\n",
                       omyn[e], nY[1], nY[3], P, en, fin ? "yes" : "NON-FINITE");
      }
      Kokkos::printf("  Test L: %d of 6 non-finite\n", lbad);
      Kokkos::printf("  (NON-FINITE at omyn <= 0 means nQ needs the guard the error "
                     "policy already applies)\n");
    }

    // ------------------------------------------------------------------ Test M
    // WHY does ConToPrim fail at n_pfloor_force?
    //
    // Production (b160-v21, segments 0000/0001, 27666 reports) puts 100.00 % of
    // NO_SOLUTION failures within 1 % of n_pfloor_force, all JUST ABOVE it:
    // n/n_pfloor_force in [1.00299, 1.00864], mean composition f = 0.942282,
    // Y_N = 0.936501, Y_2 = 0.084680, Y_3 = -0.063302 (y_lQ = -0.9969, i.e. pinned at
    // the -1 species bound). That is the hard branch in SnapPhaseFraction which, for
    // n < n_pfloor_force, collapses f -> 1, yn -> 1, Y3e -> 0 with no EOS evaluation.
    //
    // FalsePosition has exactly two failure exits (numtools_root.hpp): the sign test
    // returns IMMEDIATELY with iters == 0 (root outside the bracket), and
    // non-convergence returns after the full iteration count. So res.iterations == 0
    // discriminates "bad bracket" from "too few iterations", and this test reports it
    // rather than assuming. c2p_iter is already 500 in production.
    //
    // This test documents the branch; it asserts no fix. What it pins is (a) whether
    // the production failure band reproduces here, (b) the jump in f/y_lQ/P/e across
    // the branch, and (c) which FalsePosition exit is taken.
    {
      const Real npf = eos.GetPfloorForce();
      Kokkos::printf("\n--- Test M: C2P across the pressure-floor branch\n");
      Kokkos::printf("  n_pfloor_force = %.9e fm^-3   c2p_iter = %d\n",
                     npf, ps_.GetRootSolver().iterations);
      Kokkos::printf("  composition from the production failures: "
                     "f=%.6f Y_N=%.6f Y_2=%.6f Y_3=%.6f (y_lQ=%.4f)\n",
                     0.942282, 0.936501, 0.084680, -0.063302, -0.063302/(1.0-0.936501));
      Kokkos::printf("%-10s %-13s %-13s %-13s %-14s %-14s %-7s %s\n",
                     "n/npf", "f_out", "y_lQ_out", "nQ_out", "P", "e", "iters",
                     "ConToPrim");

      Real Ym[MAX_SPECIES] = {0.0};
      Ym[0] = 0.942282;
      Ym[1] = 0.936501;
      Ym[2] = 0.084680;
      Ym[3] = -0.063302;

      int mfail = 0, mzero = 0, mband = 0;
      const int NM = 41;
      for (int q = 0; q < NM; ++q) {
        const Real ratio = 0.995 + (1.015 - 0.995)*q/(NM - 1.0);
        const Real nM = ratio*npf;

        Real nY[6] = {0.0};
        eos.GetPhaseDecomposition(nM, Ym, nY);
        const Real P  = eos.GetPressure(nM, T_a, Ym);
        const Real en = eos.GetEnergy(nM, T_a, Ym);

        Real prim[NPRIM] = {0.0};
        Real cons[NCONS] = {0.0};
        Real bu[NMAG] = {0.0, 0.0, 0.0};
        prim[PRH] = nM;
        prim[PVX] = prim[PVY] = prim[PVZ] = 0.0;
        prim[PTM] = T_a;
        prim[PPR] = P;
        for (int i = 0; i < 4; ++i) { prim[PYF + i] = Ym[i]; }
        ps_.PrimToCon(prim, cons, bu, g3d);
        Real prim_out[NPRIM] = {0.0};
        auto res = ps_.ConToPrim(prim_out, cons, bu, g3d, g3u);

        const bool ok = (res.error == Primitive::Error::SUCCESS);
        if (!ok) {
          mfail++;
          if (res.iterations == 0) { mzero++; }
          if (ratio >= 1.00299 && ratio <= 1.00864) { mband++; }
        }
        Kokkos::printf("%-10.5f %-13.6e %-13.6e %-13.6e %-14.6e %-14.6e %-7d %s\n",
                       ratio, nY[4], nY[3], nY[1], P, en, res.iterations,
                       ok ? "SUCCESS"
                          : ((res.error == Primitive::Error::NO_SOLUTION)
                               ? "NO_SOLUTION" : "OTHER"));
      }
      Kokkos::printf("  Test M: %d of %d failed", mfail, NM);
      if (mfail > 0) {
        Kokkos::printf("  (%d with iterations == 0 => BAD BRACKET; "
                       "%d inside the production band [1.00299, 1.00864])",
                       mzero, mband);
      }
      Kokkos::printf("\n");
      Kokkos::printf("  iterations == 0 means FalsePosition's sign test rejected the "
                     "bracket, i.e. raising c2p_iter cannot help.\n");

      // Does raising c2p_iter clear them? If the failures are a bad bracket this is a
      // no-op; if they are slow convergence the count drops. Restore afterwards so the
      // rest of the harness is unaffected.
      // ps_ is const here, so sweep on a local copy (PrimitiveSolver holds Kokkos
      // Views by value; copying is shallow and leaves the harness's solver untouched).
      auto ps_m = ps_;
      Kokkos::printf("  c2p_iter sweep at n/npf = 1.005 (mid production band):\n");
      for (int w = 0; w < 3; ++w) {
        const int itw[3] = {50, 500, 5000};
        ps_m.GetRootSolverMutable().iterations = itw[w];
        const Real nM = 1.005*npf;
        Real prim[NPRIM] = {0.0};
        Real cons[NCONS] = {0.0};
        Real bu[NMAG] = {0.0, 0.0, 0.0};
        prim[PRH] = nM;
        prim[PTM] = T_a;
        prim[PPR] = eos.GetPressure(nM, T_a, Ym);
        for (int i = 0; i < 4; ++i) { prim[PYF + i] = Ym[i]; }
        ps_m.PrimToCon(prim, cons, bu, g3d);
        Real prim_out[NPRIM] = {0.0};
        auto res = ps_m.ConToPrim(prim_out, cons, bu, g3d, g3u);
        Kokkos::printf("    c2p_iter=%-6d iters=%-7d %s\n", itw[w], res.iterations,
                       (res.error == Primitive::Error::SUCCESS) ? "SUCCESS"
                       : ((res.error == Primitive::Error::NO_SOLUTION) ? "NO_SOLUTION"
                                                                      : "OTHER"));
      }
    }

    // ------------------------------------------------------------------ Test N
    // FOOTPRINT of the pressure-floor change: over WHICH densities does turning the
    // legacy fast path off actually alter the composition, and by how much?
    //
    // Test M only sweeps +-1.5% around n_pfloor_force, because that is where the C2P
    // failures are. That coverage passed a change which then destroyed 6% of the star
    // in production -- the unit tests said nothing about the rest of the density range.
    // This sweeps the whole table so the blast radius is visible in seconds instead of
    // after a 45-minute run. Diff this block between zla_pfloor_fastpath = true/false.
    {
      Kokkos::printf("\n--- Test N: f and P across the FULL density range\n");
      Kokkos::printf("  n_pfloor_force = %.6e   n_tr ~ 0.2538   n_sat = 0.16\n",
                     eos.GetPfloorForce());
      Kokkos::printf("%-14s %-10s %-13s %-13s %-14s %s\n",
                     "n", "n/npf", "f_out", "y_lQ_out", "P", "e");
      // Two compositions: (A) the production failure population, barely any quark
      // phase and y_lQ pinned at -1; (B) a genuinely mixed phase with 30% quarks and a
      // physical y_lQ. A footprint measured at only one composition is what let the
      // previous two attempts through -- the changed band sits near n_tr, where a real
      // mixed phase behaves differently from the nearly-pure-nucleon case.
      const int NN = 48;
      const Real nlo = Kokkos::fmax(min_n, 1.0e-3);
      const Real nhi = Kokkos::fmin(max_n, 1.0);
      for (int cc = 0; cc < 2; ++cc) {
        Real Yn[MAX_SPECIES] = {0.0};
        if (cc == 0) {
          Yn[0] = 0.942282; Yn[1] = 0.936501; Yn[2] = 0.084680; Yn[3] = -0.063302;
          Kokkos::printf("  [comp A] f=%.6f Y_N=%.6f y_lQ=%.4f (production failures)\n",
                         Yn[0], Yn[1], Yn[3]/(1.0 - Yn[1]));
        } else {
          Yn[0] = 0.700000; Yn[1] = 0.650000; Yn[2] = 0.126400; Yn[3] = -0.105000;
          Kokkos::printf("  [comp B] f=%.6f Y_N=%.6f y_lQ=%.4f (mixed phase)\n",
                         Yn[0], Yn[1], Yn[3]/(1.0 - Yn[1]));
        }
        for (int q = 0; q < NN; ++q) {
          const Real nN2 = nlo*Kokkos::pow(nhi/nlo, q/(NN - 1.0));
          Real nY[6] = {0.0};
          eos.GetPhaseDecomposition(nN2, Yn, nY);
          Kokkos::printf("%-14.6e %-10.4f %-13.9e %-13.6e %-14.6e %.6e\n",
                         nN2, nN2/eos.GetPfloorForce(), nY[4], nY[3],
                         eos.GetPressure(nN2, T_a, Yn), eos.GetEnergy(nN2, T_a, Yn));
        }
      }
    }

    Kokkos::printf("\n");
  });
  Kokkos::fence();

  return;
}
