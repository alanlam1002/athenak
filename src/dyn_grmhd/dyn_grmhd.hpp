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
#include "tasklist/task_list.hpp"
#include "driver/driver.hpp"
#include "eos/primitive_solver_hyd.hpp"

enum class DynGRMHD_RSolver {llf_dyngr, hlle_dyngr};   // Riemann solvers for dynamical GR
// EOS policies for dynamical GR
enum class DynGRMHD_EOS {eos_ideal, eos_piecewise_poly,
                         eos_compose, eos_hybrid, eos_zla_bag};
enum class DynGRMHD_Error {reset_floor};        // Error policies for dynamical GR

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

  // functions

  virtual void QueueDynGRMHDTasks() = 0;

  virtual TaskStatus ConToPrim(Driver* pdrive, int stage) = 0;
  virtual void ConToPrimBC(int is, int ie, int js, int je, int ks, int ke) = 0;
  virtual void PrimToConInit(int is, int ie, int js, int je, int ks, int ke) = 0;
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

  //! \brief Restore per-species bounds on the conserved passive scalars after AMR
  //! prolongation.
  //
  //  MeshRefinement::RefineCC reconstructs every component of u0 independently, each
  //  with its own min-mod slope, so sqrt(gamma)*D (slot IDN) and sqrt(gamma)*D*Y_i
  //  (slot nmhd+i) are prolongated separately. The implied primitive
  //  Y_i = u0(nmhd+i)/u0(IDN) is then a ratio of two separately limited
  //  reconstructions: min-mod bounds each numerator and denominator against its own
  //  coarse neighbours, but places no bound on the quotient, and where D is small the
  //  overshoot is unbounded. Prolongation is the only writer of the scalars with no
  //  bound enforcement -- reconstruction (the Riemann solvers), the flux update
  //  (scalar_pplimiter) and C2P all clamp Y already.
  //
  //  This is declared on the non-templated base but implemented in DynGRMHDPS, because
  //  the admissible set comes from the ErrorPolicy via EOS::ApplySpeciesLimits, which
  //  only the templated class can reach from inside a device kernel.
  virtual void EnforceScalarBoundsAfterRefinement(DualArray1D<int> &new_to_old,
                                                  DualArray1D<int> &refine_flag,
                                                  int new_nmb, int ngids) = 0;

  // DynGRMHD policies
  DynGRMHD_RSolver rsolver_method;
  DynGRMHD_RSolver fofc_method;
  DynGRMHD_EOS eos_policy;
  DynGRMHD_Error error_policy;

  // Storage for temperature
  DvceArray5D<Real> temperature;

 protected:
  MeshBlockPack *pmy_pack;  // ptr to MeshBlockPack containing this Hydro
  int scratch_level;        // GPU scratch level for flux and source calculations
  bool enforce_maximum;     // enforce local maximum principle during FOFC
  Real dmp_M;               // threshold multiplier for discrete maximum principle.
  bool fixed_evolution;     // Disable mhd evolution
  bool scalar_pplimiter;    // Apply positivity preserving limiter on scalar
  // Gate for EnforceScalarBoundsAfterRefinement (<mhd>/amr_scalar_repair). Default
  // false: the pass is under investigation as the cause of a conserved-scalar
  // corruption and mass-loss regression, so it must be opted into explicitly.
  bool amr_scalar_repair;
  // <time>/gr_dt (default false): opt-in to the real GR fast-magnetosonic-speed
  // timestep in dyn_grmhd_newdt.cpp instead of the conservative max_dv=1 (speed of
  // light) fallback -- same input key as PR #698's analogous flag on Hydro/MHD.
  bool gr_dt;
};

template<class EOSPolicy, class ErrorPolicy>
class DynGRMHDPS : public DynGRMHD {
 public:
  DynGRMHDPS(MeshBlockPack *ppack, ParameterInput *pin) :
      DynGRMHD(ppack, pin), eos("mhd", ppack, pin) {}
  virtual ~DynGRMHDPS() {}

  // Dynamical EOS
  PrimitiveSolverHydro<EOSPolicy, ErrorPolicy> eos;

  // CalculateFluxes function templated over Riemann Solvers
  template<DynGRMHD_RSolver T>
  TaskStatus CalcFluxes(Driver *d, int stage);

  template<DynGRMHD_RSolver T>
  void FOFC(Driver *d, int stage);

  // functions
  virtual void QueueDynGRMHDTasks();

  virtual TaskStatus ConToPrim(Driver* pdrive, int stage);
  virtual void ConToPrimBC(int is, int ie, int js, int je, int ks, int ke);
  virtual void PrimToConInit(int is, int ie, int js, int je, int ks, int ke);
  virtual void ConvertInternalEnergyToPressure(int is, int ie,
                                               int js, int je, int ks, int ke);

  virtual void AddCoordTerms(const DvceArray5D<Real> &w0, const DvceArray5D<Real> &bcc0,
                             const Real dt, DvceArray5D<Real> &u0, int nghost);

  // dyn_grmhd_newdt.cpp
  virtual TaskStatus NewTimeStep(Driver *pdrive, int stage);

  virtual void EnforceScalarBoundsAfterRefinement(DualArray1D<int> &new_to_old,
                                                  DualArray1D<int> &refine_flag,
                                                  int new_nmb, int ngids);

  template<int NGHOST>
  void AddCoordTermsEOS(const DvceArray5D<Real> &w0, const DvceArray5D<Real> &bcc0,
                        const Real dt, DvceArray5D<Real> &u0);
};

// Factory function for generating DynGRMHD based on parameter input.
// Used to make the MeshBlockPack creation a little bit cleaner.
DynGRMHD* BuildDynGRMHD(MeshBlockPack *ppack, ParameterInput *pin);

} // namespace dyngr

#endif  // DYN_GRMHD_DYN_GRMHD_HPP_
