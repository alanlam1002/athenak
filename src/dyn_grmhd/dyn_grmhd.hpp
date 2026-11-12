#ifndef DYN_GRMHD_DYN_GRMHD_HPP_
#define DYN_GRMHD_DYN_GRMHD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyn_grmhd.hpp
//  \brief definitions for DynGRMHD class

#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "driver/driver.hpp"
#include "eos/primitive_solver_hyd.hpp"

enum class DynGRMHD_RSolver {llf_dyngr, hlle_dyngr};   // Riemann solvers for dynamical GR
enum class DynGRMHD_EOS {eos_ideal, eos_piecewise_poly,
                      eos_compose, eos_hybrid};        // EOS policies for dynamical GR
enum class DynGRMHD_Error {reset_floor};               // Error policies for dynamical GR

namespace resistive { class ResistiveDynGR; }

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

  // DynGRMHD policies
  DynGRMHD_RSolver rsolver_method;
  DynGRMHD_RSolver fofc_method;
  DynGRMHD_EOS eos_policy;
  DynGRMHD_Error error_policy;

  // Storage for temperature
  DvceArray5D<Real> temperature;

  // Resistive GRMHD state and EOS-independent tasks (nullptr unless <mhd> resistive)
  resistive::ResistiveDynGR *presist = nullptr;

 protected:
  MeshBlockPack *pmy_pack;  // ptr to MeshBlockPack containing this Hydro
  int scratch_level;        // GPU scratch level for flux and source calculations
  bool enforce_maximum;     // enforce local maximum principle during FOFC
  Real dmp_M;               // threshold multiplier for discrete maximum principle.
  bool fixed_evolution;     // Disable mhd evolution
  bool scalar_pplimiter;    // Apply positivity preserving limiter on scalar
  // test-only resistive_probe_cycles (07 v2.9 §3.2.4(e)): ideal-path c2p dumps for the
  // K5-cold-z comparison; 0 (key absent) = off, output only
  int probe_cycles = 0;
  std::string probe_base;
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

  template<int NGHOST>
  void AddCoordTermsEOS(const DvceArray5D<Real> &w0, const DvceArray5D<Real> &bcc0,
                        const Real dt, DvceArray5D<Real> &u0);

  // Resistive GRMHD (07_scoping_v2 §3.3), defined in resistive/resistive_dyngr_tasks.cpp
  void QueueResistiveTasks();
  TaskStatus ResistiveFluxes(Driver *d, int stage);
  TaskStatus ResistivePre1(Driver *d, int stage);
  TaskStatus ResistivePre2(Driver *d, int stage);
  TaskStatus ResistiveC2PPre(Driver *d, int stage);
  TaskStatus ResistiveImplicit(Driver *d, int stage);
  TaskStatus ResistiveConToPrim(Driver *d, int stage);
  void ResistiveImplicitSolve(Driver *d, int estage);
  void ResistiveC2P(bool record = false, int pstage = -1);
  void ResistivePrimToCon(int is, int ie, int js, int je, int ks, int ke);
};

// Factory function for generating DynGRMHD based on parameter input.
// Used to make the MeshBlockPack creation a little bit cleaner.
DynGRMHD* BuildDynGRMHD(MeshBlockPack *ppack, ParameterInput *pin);

} // namespace dyngr

#endif  // DYN_GRMHD_DYN_GRMHD_HPP_
