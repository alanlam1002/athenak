#ifndef COORDINATES_COORDINATES_HPP_
#define COORDINATES_COORDINATES_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file coordinates.hpp
//! \brief implemention of light-weight coordinates class.  Provides data structure that
//! stores array of RegionSizes over (# of MeshBlocks), and inline functions for
//! computing positions.  In GR, also provides inline metric functions (currently only
//! Cartesian Kerr-Schild)

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"

// forward declarations
struct EOS_Data;

// Enumerator for the excision method
enum class ExcisionScheme {
  fixed,
  lapse,
  horizon
};

//----------------------------------------------------------------------------------------
//! \struct CoordData
//! \brief container for Coordinate variables and functions needed inside kernels. Storing
//! everything in a container makes them easier to capture, and pass to inline functions,
//! inside kernels.

struct CoordData {
  // following data is only used in GR calculations to compute metric
  bool is_minkowski;               // flag to specify Minkowski (flat) space
  Real bh_spin;                    // needed for GR metric
  bool bh_excise;                  // flag to specify excision
  Real rexcise;                    // excision radius (SKS)
  Real dexcise;                    // rest-mass density inside excised region
  Real pexcise;                    // pressure inside excised region
  Real texcise;                    // temperature inside excised region (for smooth exc.)
  Real flux_excise_r;              // reduce to first-order inside this radius
  ExcisionScheme excision_scheme;  // excision method
  Real excise_lapse;               // if excision_scheme = lapse, excise under this lapse
  bool smooth_excision = false;    // flag to specify smooth excision (fastflow)
  Real horizon_factor;             // factor to muliply the horizon factor by (fastflow)
  Real tdamp;                      // damping time (needed for smooth excision)
};

//----------------------------------------------------------------------------------------
//! \class Coordinates
//! \brief data and functions for coordinates

class Coordinates {
 public:
  explicit Coordinates(ParameterInput *pin, MeshBlockPack *ppack);
  ~Coordinates() {}

  // flags to denote relativistic dynamics in these coordinates
  bool is_special_relativistic = false;
  bool is_general_relativistic = false;
  bool is_dynamical_relativistic = false;

  // data needed to compute metric in GR
  CoordData coord_data;

  // excision masks
  DvceArray4D<bool> excision_floor;  // cell-centered mask for C2P flooring about horizon
  DvceArray4D<bool> excision_flux;   // cell-centered mask for FOFC about horizon

  // Running tally of what excision has REMOVED from the grid, THIS RANK, since the
  // last drain: index 0 = energy (ADM-weighted, see below), 1-3 = momentum P_i.
  // Accumulated by DynGRMHDPS::ConsToPrim at the reset site and drained once per
  // cycle.  This exists because CFC recovers the metric from ELLIPTIC constraints
  // sourced by the instantaneous matter distribution -- it has no memory -- so
  // deleting matter also deletes its gravity in the same step and the hole would
  // lose the mass it just swallowed.  (Z4c does not need this: its hyperbolic
  // evolution retains the information in the metric.)  The energy column is
  // weighted by 1/psi so that it is the matter's ADM-mass contribution
  // int(psi^5 E)dV, NOT the raw conserved energy int(psi^6 E)dV -- see
  // Coordinates::DrainExcisedTally's comment for the derivation.
  // index 4 additionally banks the RAW conserved energy int(psi^6 E)dV (no 1/psi).
  // It is never fed to the puncture; it exists so the accounting can be validated
  // like-for-like against the history file's own conserved sums, and so the ratio
  // col0/col4 = <1/psi> reports the effective psi at which matter is being absorbed
  // -- i.e. exactly the size of the psi^5-vs-psi^6 weighting difference.
  //
  // Plain host scalars, NOT a per-MeshBlock device array: the per-cell contributions
  // are combined by DynGRMHDPS::ConsToPrim's own Kokkos::parallel_reduce (deterministic
  // by construction -- Kokkos's reduction tree shape is fixed by the kernel launch
  // configuration, not by runtime thread-scheduling order) into five reduction
  // results, which this rank's single controlling host thread then adds in here with
  // ordinary (non-atomic) `+=`. This REPLACES an earlier version that scattered
  // per-cell contributions into a per-MeshBlock DvceArray2D via Kokkos::atomic_add:
  // atomic float addition is exact for each individual add, but the ORDER in which
  // concurrent device threads perform theirs is not reproducible run-to-run, and
  // float addition is not associative, so the accumulated total differed slightly
  // between bitwise-identical runs (measured: first-accretion dM_adm 5.6429e-03 vs
  // 5.6735e-03, diverging to a 335-cycle spread in failure onset --
  // NANCASCADE_HANDOFF.md Sec 14.6/15.2/17.5/19.4/21.6). Never use onset cycle as an
  // A/B metric on the excision path regardless; compare signatures (Sec 17.2).
  Real excised_tally[5];

  // functions
  void CoordSrcTerms(const DvceArray5D<Real> &w0, const EOS_Data &eos, const Real dt,
                     DvceArray5D<Real> &u0);
  void CoordSrcTerms(const DvceArray5D<Real> &w0, const DvceArray5D<Real> &bcc,
                     const EOS_Data &eos, const Real dt, DvceArray5D<Real> &u0);
  void SetExcisionMasks(DvceArray4D<bool> &floor, DvceArray4D<bool> &flux);

  // Sum excised_tally over MeshBlocks and MPI ranks, ZERO it, and return the totals
  // in tot[4] = {dM, dPx, dPy, dPz}.  Collective: every rank gets the same answer
  // (MPI_Allreduce, not Reduce), because every rank must then apply the same updated
  // puncture mass.  Call once per cycle.
  void DrainExcisedTally(Real tot[5]);

  void UpdateExcisionMasks();

 private:
  MeshBlockPack* pmy_pack;
};

#endif // COORDINATES_COORDINATES_HPP_
