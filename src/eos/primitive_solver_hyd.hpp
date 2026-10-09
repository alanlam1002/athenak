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
  // scalar, stored as Y = K/de_kmax so it stays inside the ideal-gas species limits
  // [0,1] (conserved: sqrt(gamma) D Y; same transport as Ye, no source term). In dense
  // (D > de_rho_switch), unshocked gas the pressure comes from K via EntropyInversion
  // and tau is resynced; elsewhere K is resynced from the energy solution. A
  // primitive-floor hit in dense gas (the silent T < T_atm path of RESULTS §7.1) also
  // takes the entropy branch. Ideal gas, B = 0 cells only.
  bool de_on = false;
  int de_idx = -1;                 // scalar index of the K tracer (nscalars - 1)
  Real de_kmax = 1.0, de_rho_sw = 0.0, de_dp_shock = 0.3, de_eta1 = 0.0, de_gamma = 5./3.;
  DvceArray4D<int> de_shock;       // per-cell shock flag from the previous primitives
  // Diagnostics of the last full (non-floors_only) C2P call, interior cells only:
  // dense cells, cells on the entropy branch, and the energy the tau resync added
  // (sum of sqrt(gamma) dtau dV); de_dE_total accumulates the latter over all calls.
  Real de_ndense = 0.0, de_nent = 0.0, de_dE = 0.0, de_dE_total = 0.0;

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

  PrimitiveSolverHydro(std::string block, MeshBlockPack *pp, ParameterInput *pin) :
//        pmy_pack(pp), ps{&eos} {
        pmy_pack(pp), nerrs(0) {
    SetPolicyParams(block, pin);
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
      de_eta1 = pin->GetOrAddReal(block, "dual_energy_eta1", 0.0);
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
    const Real de_kmax_ = de_kmax, de_gamma_ = de_gamma;


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
        Real yk = prim_pt[PPR]/pow(rho_, de_gamma_)/de_kmax_;
        prim_pt[PYF + de_idx_] = fmin(fmax(yk, 0.0), 1.0);
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
    const Real de_gamma_ = de_gamma;
    if (de_on_) {
      int e1 = prim.extent_int(4), e2 = prim.extent_int(3), e3 = prim.extent_int(2);
      if (de_shock.extent_int(0) != prim.extent_int(0) || de_shock.extent_int(1) != e3 ||
          de_shock.extent_int(2) != e2 || de_shock.extent_int(3) != e1) {
        Kokkos::realloc(de_shock, prim.extent_int(0), e3, e2, e1);
      }
      auto &shk = de_shock;
      const Real dpsh = de_dp_shock;
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
        Real p0 = prim(m, IPR, k, j, i);
        Real divv = 0.0, dpmax = 0.0;
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
          Real pp = prim(m, IPR, kp, jp, ip), pm = prim(m, IPR, km, jm, im);
          dpmax = fmax(dpmax, fabs(pp - p0)/fmax(fmin(pp, p0), 1.0e-300));
          dpmax = fmax(dpmax, fabs(pm - p0)/fmax(fmin(pm, p0), 1.0e-300));
        }
        // <= so a discontinuity between states at rest (t = 0 of a Riemann problem)
        // counts as a shock: with < it ran entropy-conserving and never formed one (V1).
        shk(m, k, j, i) = (divv <= 0.0 && dpmax > dpsh) ? 1 : 0;
      });
    }
    auto &de_shock_ = de_shock;

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
        // ---- dual energy (see de_on): choose the energy or the entropy solution.
        bool de_write = false;
        if (de_on_) {
          const bool interior = (i >= is && i <= ie_ && j >= js && j <= je_ &&
                                 k >= ks && k <= ke_);
          const Real D = cons_pt_old[CDN];
          const bool dense = D > de_rho_sw_;
          const bool bzero = (b3u[IBX] == 0.0 && b3u[IBY] == 0.0 && b3u[IBZ] == 0.0);
          const bool bad = result.prim_floor ||
                           (result.error != Primitive::Error::SUCCESS);
          bool use_ent = dense && bzero && (de_shock_(m,k,j,i) == 0 || bad);
          if (use_ent && !bad && de_eta1_ > 0.0) {
            // optional energy-ratio gate: entropy only where eps is a small part of tau
            Real eps = prim_pt[PPR]/((de_gamma_ - 1.0)*prim_pt[PRH]*mb);
            if (eps > de_eta1_*cons_pt_old[CTA]/D) use_ent = false;
          }
          if (use_ent) {
            Real K = cons_pt_old[CYD + de_idx_]/D*de_kmax_;
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
              cons_pt[CTA] = tau_new;
              de_write = true;
            } else {
              use_ent = false;
            }
          }
          if (!use_ent && !(dense && bad)) {
            // resync K from the energy solution (never from a floored dense state)
            Real rho_ = prim_pt[PRH]*mb;
            Real yk = (rho_ > 0.0) ? prim_pt[PPR]/pow(rho_, de_gamma_)/de_kmax_ : 0.0;
            yk = fmin(fmax(yk, 0.0), 1.0);
            prim_pt[PYF + de_idx_] = yk;
            cons_pt[CYD + de_idx_] = cons_pt[CDN]*yk;
            de_write = true;
          }
          if (interior && dense) { nden_r += 1.0; }
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
    const Real kmax_ = de_kmax, rsw_ = de_rho_sw, gam_ = de_gamma;
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
      Real K = cons(m, nhyd + idx_, k, j, i)/sdetg/D*kmax_;
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
