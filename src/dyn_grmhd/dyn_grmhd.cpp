//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyngr.cpp
//! \brief implementation of functions for DynGRMHD and DynGRMHDPS controlling the task
//! list

#include <math.h>

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "bvals/bvals.hpp"
#include "mhd/mhd.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "z4c/tmunu.hpp"
#include "dyn_grmhd.hpp"
#include "tasklist/numerical_relativity.hpp"

#include "eos/primitive_solver_hyd.hpp"
#include "eos/primitive-solver/idealgas.hpp"
#include "eos/primitive-solver/eos_compose.hpp"
#include "eos/primitive-solver/eos_hybrid.hpp"
#include "eos/primitive-solver/piecewise_polytrope.hpp"
#include "eos/primitive-solver/reset_floor.hpp"

namespace dyngr {

// A dumb template function containing the switch statement needed to select an EOS.
template<class ErrorPolicy>
DynGRMHD* SelectDynGRMHDEOS(MeshBlockPack *ppack, ParameterInput *pin,
                            DynGRMHD_EOS eos_policy) {
  DynGRMHD* dyn_gr = nullptr;
  bool use_NQT = false;
  switch(eos_policy) {
    case DynGRMHD_EOS::eos_ideal:
      dyn_gr = new DynGRMHDPS<Primitive::IdealGas, ErrorPolicy>(ppack, pin);
      break;
    case DynGRMHD_EOS::eos_piecewise_poly:
      dyn_gr = new DynGRMHDPS<Primitive::PiecewisePolytrope, ErrorPolicy>(ppack, pin);
      break;
    case DynGRMHD_EOS::eos_compose:
      use_NQT = pin->GetOrAddBoolean("mhd", "use_NQT",false);
      if (use_NQT) {
        dyn_gr = new DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                                ErrorPolicy>(ppack, pin);
      } else {
        dyn_gr = new DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                                ErrorPolicy>(ppack, pin);
      }
      break;
    case DynGRMHD_EOS::eos_hybrid:
      use_NQT = pin->GetOrAddBoolean("mhd", "use_NQT",false);
      if (use_NQT) {
        dyn_gr = new DynGRMHDPS<Primitive::EOSHybrid<Primitive::NQTLogs>,
                                ErrorPolicy>(ppack, pin);
      } else {
        dyn_gr = new DynGRMHDPS<Primitive::EOSHybrid<Primitive::NormalLogs>,
                                ErrorPolicy>(ppack, pin);
      }
      break;
  }
  return dyn_gr;
}

DynGRMHD* BuildDynGRMHD(MeshBlockPack *ppack, ParameterInput *pin) {
  std::string eos_string = pin->GetString("mhd", "dyn_eos");
  std::string error_string = pin->GetString("mhd", "dyn_error");
  DynGRMHD_EOS eos_policy;
  DynGRMHD_Error error_policy;

  if (eos_string.compare("ideal") == 0) {
    eos_policy = DynGRMHD_EOS::eos_ideal;
  } else if (eos_string.compare("piecewise_poly") == 0) {
    eos_policy = DynGRMHD_EOS::eos_piecewise_poly;
  } else if (eos_string.compare("compose") == 0) {
    eos_policy = DynGRMHD_EOS::eos_compose;
  } else if (eos_string.compare("hybrid") == 0) {
    eos_policy = DynGRMHD_EOS::eos_hybrid;
  } else {
    std::cout << "### FATAL ERROR in " <<__FILE__ << " at line " << __LINE__
              << std::endl << "<mhd> dyn_eos = '" << eos_string
              << "' not implemented for GR dynamics" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (error_string.compare("reset_floor") == 0) {
    error_policy = DynGRMHD_Error::reset_floor;
  } else {
    std::cout << "### FATAL ERROR in " <<__FILE__ << " at line " << __LINE__
              << std::endl << "<mhd> dyn_error = '" << error_string
              << "' not implemented for GR dynamics" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  DynGRMHD* dyn_gr = nullptr;

  switch (error_policy) {
    case DynGRMHD_Error::reset_floor:
      dyn_gr = SelectDynGRMHDEOS<Primitive::ResetFloor>(ppack, pin, eos_policy);
      break;
  }

  dyn_gr->eos_policy = eos_policy;
  dyn_gr->error_policy = error_policy;

  return dyn_gr;
}

DynGRMHD::DynGRMHD(MeshBlockPack *pp, ParameterInput *pin) :
    temperature("temperature",1,1,1,1,1),
    de_diag("de_diag",1,1,1,1,1),
    pmy_pack(pp) {
  std::string rsolver = pin->GetString("mhd", "rsolver");
  if (rsolver.compare("llf") == 0) {
    rsolver_method = DynGRMHD_RSolver::llf_dyngr;
  } else if (rsolver.compare("hlle") == 0) {
    rsolver_method = DynGRMHD_RSolver::hlle_dyngr;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<mhd> rsolver = '" << rsolver
              << "' not implemented for GR dynamics" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::string fofc = pin->GetOrAddString("mhd", "fofc_method", "llf");
  if (fofc.compare("llf") == 0) {
    fofc_method = DynGRMHD_RSolver::llf_dyngr;
  } else if (fofc.compare("hlle") == 0) {
    fofc_method = DynGRMHD_RSolver::hlle_dyngr;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<mhd> fofc_method = '" << fofc
              << "' not implemented for FOFC" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  scratch_level = pin->GetOrAddInteger("mhd", "dyn_scratch", 0);
  enforce_maximum = pin->GetOrAddBoolean("mhd", "enforce_maximum", true);
  dmp_M = pin->GetOrAddReal("mhd", "dmp_M", 1.2);
  scalar_pplimiter = pin->GetOrAddBoolean("mhd", "scalar_pplimiter", true);
  fofc_scalar_consistent = pin->GetOrAddBoolean("mhd", "fofc_scalar_consistent", false);

  fixed_evolution = pin->GetOrAddBoolean("mhd", "fixed", false);
  gr_dt = pin->GetOrAddBoolean("time", "gr_dt", false);
  dt_skip_excised = pin->GetOrAddBoolean("time", "dt_skip_excised", false);
  dt_floor_keep_velocity = pin->GetOrAddBoolean("time", "dt_floor_keep_velocity", false);
  dt_lim_report = pin->GetOrAddInteger("time", "dt_limiter_report", 0);
  dt_lim_every = pin->GetOrAddInteger("time", "dt_limiter_report_every", 0);

  // <coord> excision_radius_dynamic (R-112 item 3): only with the lapse excision scheme
  exc_dyn = pin->GetOrAddBoolean("coord", "excision_radius_dynamic", false);
  if (exc_dyn) {
    if (!pin->GetOrAddBoolean("coord", "excise", false) ||
        pin->GetOrAddString("coord", "excision_scheme", "fixed") != "lapse") {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "<coord> excision_radius_dynamic needs excise = true and "
                << "excision_scheme = lapse" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    exc_rmax = pin->GetOrAddReal("coord", "excision_rmax", 2.5);
    exc_margin = pin->GetOrAddReal("coord", "excision_margin_cells", 4.0);
    exc_vin = pin->GetOrAddReal("coord", "excision_vin", 0.6);
    // the guard's density: the zone's threshold unless set explicitly
    const Real zr = pin->DoesParameterExist("mhd", "atmosphere_zone_rho") ?
                    pin->GetReal("mhd", "atmosphere_zone_rho") : 0.0;
    exc_rho_thr = pin->GetOrAddReal("coord", "excision_rho_thr", zr);
    if (!(exc_rho_thr > 0.0)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "<coord> excision_radius_dynamic needs excision_rho_thr (or <mhd> "
                << "atmosphere_zone_rho) > 0" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    // the state written by outputs/restart.cpp (absent on a fresh start: r_f grows from 0)
    exc_rf = pin->GetOrAddReal("coord", "excision_rf_state", 0.0);
    exc_abs_total = pin->GetOrAddReal("coord", "excision_absorbed_total", 0.0);
  }

  // allocate memory for temperature
  {
    int nmb = std::max((pmy_pack->nmb_thispack), (pmy_pack->pmesh->nmb_maxperrank));
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(temperature, nmb, 1, ncells3, ncells2, ncells1);
    if (pin->GetOrAddBoolean("mhd", "dual_energy_diag", false)) {
      // <mhd> shell_counters: 13 = the flag sensor's compression theta, 14/15 = the sensor
      // ratio and theta that the last C2P's flag came from (mass_closure.hpp)
      const bool shc = pin->GetOrAddBoolean("mhd", "shell_counters", false);
      Kokkos::realloc(de_diag, nmb, shc ? 16 : 13, ncells3, ncells2, ncells1);
    }
  }
  pmclose = new mclose::MassClosure(pp, pin);
  if (!pmclose->on) { delete pmclose; pmclose = nullptr; }

  // <dyn_grmhd> metric_terms (R-140 item 1; metric_faces.hpp). Default cell: nothing below.
  std::string mt = pin->GetOrAddString("dyn_grmhd", "metric_terms", "cell");
  if (mt == "cell") {
    metric_terms = MetricTerms::cell;
  } else if (mt == "analytic_bg") {
    metric_terms = MetricTerms::analytic_bg;
  } else if (mt == "wellbalanced") {
    metric_terms = MetricTerms::wellbalanced;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<dyn_grmhd> metric_terms = '" << mt << "' (cell, analytic_bg or "
              << "wellbalanced)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (metric_terms != MetricTerms::cell) {
    face_residual_interp = pin->GetOrAddInteger("dyn_grmhd", "face_residual_interp", 4);
    if (face_residual_interp != 2 && face_residual_interp != 4) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "<dyn_grmhd> face_residual_interp must be 2 or 4" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (!pin->GetOrAddBoolean("cfc", "puncture_enabled", false)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "<dyn_grmhd> metric_terms = " << mt << " needs the CFC trumpet "
                << "background (<cfc> puncture_enabled = true)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    if (indcs.ng < 2) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "<dyn_grmhd> metric_terms needs nghost >= 2" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    int nmb = std::max((pmy_pack->nmb_thispack), (pmy_pack->pmesh->nmb_maxperrank));
    int n1 = indcs.nx1 + 2*(indcs.ng);
    int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(fm_res, nmb, 5, n3, n2, n1);
    Kokkos::realloc(fm_x1, nmb, kFmN, n3, n2, n1 + 1);
    if (n2 > 1) Kokkos::realloc(fm_x2, nmb, kFmN, n3, n2 + 1, n1);
    if (n3 > 1) Kokkos::realloc(fm_x3, nmb, kFmN, n3 + 1, n2, n1);
    // the table: w = sqrt(r_iso/M) in [0, 64] (r_iso <= 4096 M), h = 0.0025 (quintic
    // Hermite; 2e-16 relative in rho and rho - 3/2, mpmath check in c122)
    BuildTrumpetTable(ttab, pin->GetOrAddReal("dyn_grmhd", "metric_table_wmax", 64.0),
                      pin->GetOrAddReal("dyn_grmhd", "metric_table_h", 0.0025));
    fm_probe_cycles = pin->GetOrAddInteger("dyn_grmhd", "face_probe_cycles", 0);
    const char *bk[6] = {"face_probe_xmin", "face_probe_xmax", "face_probe_ymin",
                         "face_probe_ymax", "face_probe_zmin", "face_probe_zmax"};
    for (int q = 0; q < 6; ++q) {
      fm_probe_box[q] = pin->GetOrAddReal("dyn_grmhd", bk[q], (q % 2) ? 1.0e30 : -1.0e30);
    }
    if (global_variable::my_rank == 0) {
      std::cout << "<dyn_grmhd> metric_terms = " << mt << ", face_residual_interp = "
                << face_residual_interp << ", trumpet table n = " << ttab.n << std::endl;
    }
    const int nt = pin->GetOrAddInteger("dyn_grmhd", "metric_table_test", 0);
    if (nt > 0) TrumpetTableTest(nt);
  }
}

DynGRMHD::~DynGRMHD() {
  delete pmclose;
}

template<class EOSPolicy, class ErrorPolicy>
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::QueueDynGRMHDTasks() {
  using namespace mhd;  // NOLINT(build/namespaces)
  using namespace z4c;  // NOLINT(build/namespaces)
  using namespace numrel; // NOLINT(build/namespaces))
  Z4c *pz4c = pmy_pack->pz4c;
  adm::ADM *padm = pmy_pack->padm;
  MHD *pmhd = pmy_pack->pmhd;
  NumericalRelativity *pnr = pmy_pack->pnr;

  // Start task list
  pnr->QueueTask(&MHD::InitRecv, pmhd, MHD_Recv, "MHD_Recv", Task_Start);

  // Run task list
  pnr->QueueTask(&MHD::CopyCons, pmhd, MHD_CopyU, "MHD_CopyU", Task_Run);

  // Select which CalculateFlux function to add based on rsolver_method.
  // CalcFlux requires metric in flux - must happen before z4ctoadm updates the metric
  if (metric_terms != MetricTerms::cell) {   // the face metric (metric_faces.hpp)
    if (rsolver_method == DynGRMHD_RSolver::llf_dyngr) {
      pnr->QueueTask(
        &DynGRMHDPS<EOSPolicy, ErrorPolicy>::CalcFluxes<DynGRMHD_RSolver::llf_dyngr, true>,
        this, MHD_Flux, "MHD_Flux", Task_Run, {MHD_CopyU});
    } else if (rsolver_method == DynGRMHD_RSolver::hlle_dyngr) {
      pnr->QueueTask(
        &DynGRMHDPS<EOSPolicy, ErrorPolicy>::CalcFluxes<DynGRMHD_RSolver::hlle_dyngr, true>,
        this, MHD_Flux, "MHD_Flux", Task_Run, {MHD_CopyU});
    } else {
      abort();
    }
  } else if (rsolver_method == DynGRMHD_RSolver::llf_dyngr) {
    pnr->QueueTask(
           &DynGRMHDPS<EOSPolicy, ErrorPolicy>::CalcFluxes<DynGRMHD_RSolver::llf_dyngr>,
           this, MHD_Flux, "MHD_Flux", Task_Run, {MHD_CopyU});
  } else if (rsolver_method == DynGRMHD_RSolver::hlle_dyngr) {
    pnr->QueueTask(
           &DynGRMHDPS<EOSPolicy, ErrorPolicy>::CalcFluxes<DynGRMHD_RSolver::hlle_dyngr>,
           this, MHD_Flux, "MHD_Flux", Task_Run, {MHD_CopyU});
  } else { // put more rsolvers here
    abort();
  }

  // Now the rest of the MHD run tasks
  if (pz4c != nullptr) {
    pnr->QueueTask(&DynGRMHD::SetTmunu, this, MHD_SetTmunu, "MHD_SetTmunu",
                   Task_Run, {MHD_CopyU});
  }
  pnr->QueueTask(&MHD::SendFlux, pmhd, MHD_SendFlux, "MHD_SendFlux",
                 Task_Run, {MHD_Flux});
  pnr->QueueTask(&MHD::RecvFlux, pmhd, MHD_RecvFlux, "MHD_RecvFlux",
                 Task_Run, {MHD_SendFlux});
  if (pz4c != nullptr) {
    pnr->QueueTask(&MHD::RKUpdate, pmhd, MHD_ExplRK, "MHD_ExplRK", Task_Run,
                   {MHD_RecvFlux, MHD_SetTmunu});
  } else {
    pnr->QueueTask(&MHD::RKUpdate, pmhd, MHD_ExplRK, "MHD_ExplRK", Task_Run,
                   {MHD_RecvFlux});
  }
  pnr->QueueTask(&MHD::MHDSrcTerms, pmhd, MHD_AddSrc, "MHD_AddSrc", Task_Run,
                 {MHD_ExplRK});
  pnr->QueueTask(&MHD::RestrictU, pmhd, MHD_RestU, "MHD_RestU", Task_Run, {MHD_AddSrc});
  pnr->QueueTask(&MHD::SendU, pmhd, MHD_SendU, "MHD_SendU", Task_Run, {MHD_RestU});
  pnr->QueueTask(&MHD::RecvU, pmhd, MHD_RecvU, "MHD_RecvU", Task_Run, {MHD_SendU});
  pnr->QueueTask(&MHD::CornerE, pmhd, MHD_EField, "MHD_EField", Task_Run, {MHD_RecvU});
  pnr->QueueTask(&MHD::SendE, pmhd, MHD_SendE, "MHD_SendE", Task_Run, {MHD_EField});
  pnr->QueueTask(&MHD::RecvE, pmhd, MHD_RecvE, "MHD_RecvE", Task_Run, {MHD_SendE});
  pnr->QueueTask(&MHD::CT, pmhd, MHD_CT, "MHD_CT", Task_Run, {MHD_RecvE});
  pnr->QueueTask(&MHD::RestrictB, pmhd, MHD_RestB, "MHD_RestB", Task_Run, {MHD_CT});
  pnr->QueueTask(&MHD::SendB, pmhd, MHD_SendB, "MHD_SendB", Task_Run, {MHD_RestB});
  pnr->QueueTask(&MHD::RecvB, pmhd, MHD_RecvB, "MHD_RecvB", Task_Run, {MHD_SendB});
  pnr->QueueTask(&MHD::Prolongate, pmhd, MHD_Prolong, "MHD_Prolong", Task_Run,
                 {MHD_RecvB});
  pnr->QueueTask(&MHD::ApplyPhysicalBCs, pmhd, MHD_BCS, "MHD_BCS", Task_Run,
                 {MHD_Prolong});
  //pnr->QueueTask(&DynGRMHD::ApplyPhysicalBCs, this, MHD_BCS, "MHD_BCS", Task_Run,
  //                 {MHD_RecvB});
  if (pz4c == nullptr && padm->is_dynamic == true) {
    pnr->QueueTask(&DynGRMHD::SetADMVariables, this, MHD_SetADM, "MHD_SetADM", Task_Run,
                    {MHD_ExplRK});
    // CFC_AssemblePsiG is optional: when a <cfc> block is present, con2prim must wait for
    // the freshly-solved psi/g_dd (see cfc::CFC::QueueCFCTasks) so it inverts against
    // the current stage's metric rather than the previous one -- in the ghost cells too
    // (R-072 M1: psi exchanged and g_dd reassembled there first, so a ghost's metric is
    // its owner's bitwise; waiting on CFC_SolvePsi alone left ghosts one solve stale,
    // C-055); a no-cfc run is
    // unaffected since AddExtraDependencies() only promotes optional deps whose
    // physics module actually exists.
    pnr->QueueTask(&DynGRMHDPS<EOSPolicy, ErrorPolicy>::ConToPrim, this, MHD_C2P,
                   "MHD_C2P", Task_Run, {MHD_BCS, MHD_SetADM},
                   {Z4c_Excise, CFC_AssemblePsiG});
    pnr->QueueTask(&DynGRMHD::UpdateExcisionMasks, this, MHD_Excise, "MHD_Excise",
                   Task_Run, {MHD_SetADM});
    if (eos.zone_unmask) {   // needs the rebuild after the C2P (the branch below)
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<mhd> atmosphere_zone_unmask needs <adm> dynamic = false"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  } else {
    pnr->QueueTask(&DynGRMHDPS<EOSPolicy, ErrorPolicy>::ConToPrim, this, MHD_C2P,
                   "MHD_C2P", Task_Run, {MHD_BCS}, {Z4c_Excise, CFC_AssemblePsiG});

    // The dynamic excision schemes (lapse/horizon) rebuild excision_floor/_flux from
    // the CURRENT lapse every cycle; only "fixed" is filled once at construction
    // (coordinates.cpp:111).  The branch above queues that refresh, but it is gated
    // on <adm> dynamic=true -- which no CFC fixture sets (all leave <adm> empty, so
    // adm.cpp:34 defaults it false).  The result was that excision_scheme=lapse
    // allocated its masks, never wrote them, and silently excised nothing at all in
    // every CFC run: verified in a TDE run where density inside the nominal excision
    // radius sat at 1.9e-9 instead of dexcise=1e-24, velocities there were nonzero,
    // and no mass ever left the grid.  So queue the refresh here too.
    // Dependencies mirror MHD_Newdt just below: MHD_C2P required, CFC_AssembleFinal
    // optional so that with <cfc> active the mask is built from the fully assembled
    // lapse of this step (and the optional dep is simply dropped when CFC is absent).
    if (pmy_pack->pcoord->coord_data.bh_excise &&
        pmy_pack->pcoord->coord_data.excision_scheme != ExcisionScheme::fixed) {
      // CFC_BCSADM (optional, implies CFC_AssembleFinal): AssembleLapseShiftK writes alpha on
    // the interior only, so the rebuild must wait for the ADM ghost exchange; otherwise the
    // ghost masks (and F1's ghost decisions, atmosphere_zone_unmask) use the previous solve's
    // lapse and differ from their owners at the next cycle's first flux call (R-093 Sec. 1;
    // measured by the C-099 probe at the excision boundary, all ghost depths)
    pnr->QueueTask(&DynGRMHD::UpdateExcisionMasks, this, MHD_Excise, "MHD_Excise",
                     Task_Run, {MHD_C2P}, {CFC_BCSADM});
    }
  }
  // CFC_AssembleFinal is optional for the same reason: MHD_Newdt should see the final
  // lapse/shift when cfc is active, but is unaffected when it isn't.
  // dyn_grmhd_newdt.cpp's NewTimeStep replaces mhd::MHD::NewTimeStep here -- the
  // latter hardcodes max_dv=1 (speed of light) for any is_dynamical_relativistic_ run,
  // this module's own version can instead use the real GR fast magnetosonic speed
  // (gated on <time>/gr_dt) via this module's primitive-solver EOS.
  // With <mhd> atmosphere_zone_unmask the mask rebuild also sets the state of the cells
  // it un-excises, so NewTimeStep must come after it.
  if (eos.zone_unmask && pmy_pack->pcoord->coord_data.bh_excise &&
      pmy_pack->pcoord->coord_data.excision_scheme != ExcisionScheme::fixed) {
    pnr->QueueTask(&DynGRMHDPS<EOSPolicy, ErrorPolicy>::NewTimeStep, this, MHD_Newdt,
                   "MHD_Newdt", Task_Run, {MHD_C2P, MHD_Excise}, {CFC_AssembleFinal});
  } else {
    pnr->QueueTask(&DynGRMHDPS<EOSPolicy, ErrorPolicy>::NewTimeStep, this, MHD_Newdt,
                   "MHD_Newdt", Task_Run, {MHD_C2P}, {CFC_AssembleFinal});
  }

  // End task list
  pnr->QueueTask(&MHD::ClearSend, pmhd, MHD_ClearS, "MHD_ClearS", Task_End);
  pnr->QueueTask(&MHD::ClearRecv, pmhd, MHD_ClearR, "MHD_ClearR", Task_End);
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus DynGRMHD::ADMMatterSource_(Driver *pdrive, int stage) {
//  \brief
template<class EOSPolicy, class ErrorPolicy>
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::PrimToConInit(int is, int ie, int js, int je,
                                                    int ks, int ke) {
  eos.PrimToCons(pmy_pack->pmhd->w0, pmy_pack->pmhd->bcc0, pmy_pack->pmhd->u0,
                 is, ie, js, je, ks, ke);
  if (eos.de_on) {
    Real err = eos.DualEnergySelfTest(pmy_pack->pmhd->u0, pmy_pack->pmhd->w0);
    if (global_variable::my_rank == 0) {
      std::cout << "dual energy V0 self-test: max relative error of the entropy inversion"
                << " over dense cells = " << err << std::endl;
      std::cout << "dual energy: Y = (K/kmax)^" << eos.de_ypow << ", reconstruct "
                << (eos.de_recon_rhoy ? "rho*Y" : "Y") << ", tracer flux "
                << (eos.de_flux_hll ? "hll" : "mass") << ", shock sensor "
                << (eos.de_sensor == 0 ? "jump" : (eos.de_sensor == 1 ? "jameson" : "vcurv"))
                << ", dilate " << eos.de_dilate << ", second-law guard "
                << (eos.de_second_law ? "on" : "off") << std::endl;
    }
  }
  if (pmy_pack->ptmunu != nullptr) {
    bool fixed = fixed_evolution;
    fixed_evolution = false;
    SetTmunu(nullptr, 0);
    fixed_evolution = fixed;
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void DynGRMHD::ConvertInternalEnergyToPressure
//  \brief
template<class EOSPolicy, class ErrorPolicy>
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::ConvertInternalEnergyToPressure(int is, int ie,
    int js, int je, int ks, int ke) {
  int nmb = pmy_pack->nmb_thispack;
  auto &prim = pmy_pack->pmhd->w0;
  auto &eos_ = eos.ps.GetEOS();
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;

  const Real mb = eos_.GetBaryonMass();

  par_for("coord_src", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real n = prim(m, IDN, k, j, i) / mb;
    Real egas = mb*n + prim(m, IEN, k, j, i);
    Real Y[MAX_SPECIES] = {0.};
    for (int s = 0; s < nscal; s++) {
      Y[s] = prim(m, nmhd + s, k, j, i);
    }
    Real T;
    // Note that this is done explicitly rather than with a flooring policy because we
    // don't have the temperature yet, and it's probable that the energy is bunk if the
    // density is. There may be a cleaner way to do this elsewhere.
    if (n < eos_.GetMinimumDensity()) {
      n = eos_.GetMinimumDensity();
      T = eos_.GetMinimumTemperature();
    } else {
      T = eos_.GetTemperatureFromE(n, egas, Y);
    }
    prim(m, IPR, k, j, i) = eos_.GetPressure(n, T, Y);
  });
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus DynGRMHD::ADMMatterSource_(Driver *pdrive, int stage) {
//  \brief
template<class EOSPolicy, class ErrorPolicy>
TaskStatus DynGRMHDPS<EOSPolicy, ErrorPolicy>::ConToPrim(Driver *pdrive, int stage) {
  if (fixed_evolution) {
    return TaskStatus::complete;
  }

  // Extract the indices
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1m1 = indcs.nx1 + 2*ng - 1;
  int n2m1 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng - 1) : 0;
  int n3m1 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng - 1) : 0;
  RefreshExcisionMasks();
  eos.de_stage = stage;   // dual_energy_diag per-stage slots only
  eos.MetricProbe(0, stage);   // R-072 item 1 (debug; off unless the probe is set)
  eos.MassLedgerPre(stage);
  if (pmclose != nullptr) pmclose->StagePre(stage);
  eos.ConsToPrim(pmy_pack->pmhd->u0, pmy_pack->pmhd->b0, pmy_pack->pmhd->bcc0,
                 pmy_pack->pmhd->w0, temperature, 0, n1m1, 0, n2m1, 0, n3m1, false);
  eos.DrainZone(stage, pdrive->nexp_stages, pdrive->gam0, pmy_pack->pmesh->time,
                pmy_pack->pmesh->ncycle);
  eos.ShockCounters(stage, pdrive->nexp_stages, pmy_pack->pmesh->ncycle);   // R-112 item 1
  eos.UpdateShockFlags();   // dual_energy_flag_exchange: the owner decides (R-069)
  eos.MassLedgerPost(stage, pdrive->nexp_stages, pdrive->gam0, pdrive->gam1, pdrive->beta,
                     pmy_pack->pmesh->dt, pmy_pack->pmesh->time, pmy_pack->pmesh->ncycle);
  if (pmclose != nullptr) {
    pmclose->StagePost(stage, pdrive->nexp_stages, pdrive->gam0, pdrive->gam1, pdrive->beta,
                       pmy_pack->pmesh->dt, eos.ml_d0, eos.ml_cls, eos.ml_c2p,
                       pmy_pack->pmesh->ncycle, pmy_pack->pmesh->time);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void DynGRMHDPS::ConToPrimBC(int is, int ie, int js, int je, int ks, int ke)
//  \brief
template<class EOSPolicy, class ErrorPolicy>
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::ConToPrimBC(int is, int ie, int js, int je,
                                                int ks, int ke) {
  if (fixed_evolution) {
    return;
  }
  RefreshExcisionMasks();
  eos.ConsToPrim(pmy_pack->pmhd->u0, pmy_pack->pmhd->b0, pmy_pack->pmhd->bcc0,
                 pmy_pack->pmhd->w0, temperature, is, ie, js, je, ks, ke, false);
}

//----------------------------------------------------------------------------------------
//! \fn void DynGRMHDPS::UpdateShockFlags()
//! \brief dual_energy_flag_exchange: the owner's flags after a C2P outside the RK stages
//! (the regrid path, CFC::ReinitializeMetricForAMR)
template<class EOSPolicy, class ErrorPolicy>
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::UpdateShockFlags() {
  eos.UpdateShockFlags();
}

//----------------------------------------------------------------------------------------
//! \fn int DynGRMHDPS::RestartStateVars()
//! \brief restart_state (outputs/restart_state.hpp): w0, u0's flag slot(s), bcc0,
//! temperature, and de_streak when dual_energy_shock_persist > 1
template<class EOSPolicy, class ErrorPolicy>
int DynGRMHDPS<EOSPolicy, ErrorPolicy>::RestartStateVars() {
  auto *pmhd = pmy_pack->pmhd;
  const int nw = pmhd->w0.extent_int(1);
  const int nflag = pmhd->u0.extent_int(1) - (pmhd->nmhd + pmhd->nscalars);
  return nw + nflag + 3 + 1 + ((eos.de_persist > 1) ? 1 : 0);
}

//! the same variables as named groups (restart_state.hpp: the file's layout, so a later
//! binary maps them by name)
template<class EOSPolicy, class ErrorPolicy>
std::string DynGRMHDPS<EOSPolicy, ErrorPolicy>::RestartStateLayout() {
  auto *pmhd = pmy_pack->pmhd;
  const int nw = pmhd->w0.extent_int(1);
  const int nflag = pmhd->u0.extent_int(1) - (pmhd->nmhd + pmhd->nscalars);
  std::string s = "dyngr.w0:" + std::to_string(nw);
  if (nflag > 0) s += ",dyngr.flag:" + std::to_string(nflag);
  s += ",dyngr.bcc0:3,dyngr.temp:1";
  if (eos.de_persist > 1) s += ",dyngr.streak:1";
  return s;
}

template<class EOSPolicy, class ErrorPolicy>
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::SaveRestartState(DvceArray5D<Real> &dst,
                                                          int ns) {
  auto *pmhd = pmy_pack->pmhd;
  const int nmb = pmy_pack->nmb_thispack;
  const int nw = pmhd->w0.extent_int(1), nu = pmhd->nmhd + pmhd->nscalars;
  const int nflag = pmhd->u0.extent_int(1) - nu;
  auto mb = std::make_pair(0, nmb);
  auto all = Kokkos::ALL;
  int n = ns;
  Kokkos::deep_copy(Kokkos::subview(dst, all, std::make_pair(n, n + nw), all, all, all),
                    Kokkos::subview(pmhd->w0, mb, all, all, all, all));
  n += nw;
  if (nflag > 0) {
    Kokkos::deep_copy(Kokkos::subview(dst, all, std::make_pair(n, n + nflag), all, all, all),
                      Kokkos::subview(pmhd->u0, mb, std::make_pair(nu, nu + nflag), all, all,
                                      all));
    n += nflag;
  }
  Kokkos::deep_copy(Kokkos::subview(dst, all, std::make_pair(n, n + 3), all, all, all),
                    Kokkos::subview(pmhd->bcc0, mb, all, all, all, all));
  n += 3;
  Kokkos::deep_copy(Kokkos::subview(dst, all, std::make_pair(n, n + 1), all, all, all),
                    Kokkos::subview(temperature, mb, all, all, all, all));
  n += 1;
  if (eos.de_persist > 1) {
    auto &stk = eos.de_streak;
    const bool have = stk.extent_int(0) >= nmb && stk.extent_int(1) == dst.extent_int(2) &&
                      stk.extent_int(2) == dst.extent_int(3) &&
                      stk.extent_int(3) == dst.extent_int(4);
    const int nn = n;
    par_for("rst_save_streak", DevExeSpace(), 0, nmb-1, 0, dst.extent_int(2)-1,
            0, dst.extent_int(3)-1, 0, dst.extent_int(4)-1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      dst(m, nn, k, j, i) = have ? static_cast<Real>(stk(m, k, j, i)) : 0.0;
    });
  }
}

template<class EOSPolicy, class ErrorPolicy>
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::LoadRestartState(const DvceArray5D<Real> &src,
                                                          int ns) {
  auto *pmhd = pmy_pack->pmhd;
  const int nmb = pmy_pack->nmb_thispack;
  const int nw = pmhd->w0.extent_int(1), nu = pmhd->nmhd + pmhd->nscalars;
  const int nflag = pmhd->u0.extent_int(1) - nu;
  auto mb = std::make_pair(0, nmb);
  auto all = Kokkos::ALL;
  int n = ns;
  Kokkos::deep_copy(Kokkos::subview(pmhd->w0, mb, all, all, all, all),
                    Kokkos::subview(src, all, std::make_pair(n, n + nw), all, all, all));
  n += nw;
  if (nflag > 0) {
    Kokkos::deep_copy(Kokkos::subview(pmhd->u0, mb, std::make_pair(nu, nu + nflag), all, all,
                                      all),
                      Kokkos::subview(src, all, std::make_pair(n, n + nflag), all, all, all));
    n += nflag;
  }
  Kokkos::deep_copy(Kokkos::subview(pmhd->bcc0, mb, all, all, all, all),
                    Kokkos::subview(src, all, std::make_pair(n, n + 3), all, all, all));
  n += 3;
  Kokkos::deep_copy(Kokkos::subview(temperature, mb, all, all, all, all),
                    Kokkos::subview(src, all, std::make_pair(n, n + 1), all, all, all));
  n += 1;
  if (eos.de_persist > 1) {
    // sized as ComputeShockFlags sizes it (de_shock: w0's extents), so it is kept
    Kokkos::realloc(eos.de_streak, pmhd->w0.extent_int(0), src.extent_int(2),
                    src.extent_int(3), src.extent_int(4));
    auto &stk = eos.de_streak;
    const int nn = n;
    par_for("rst_load_streak", DevExeSpace(), 0, nmb-1, 0, src.extent_int(2)-1,
            0, src.extent_int(3)-1, 0, src.extent_int(4)-1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      stk(m, k, j, i) = static_cast<int>(src(m, nn, k, j, i));
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void DynGRMHDPS::ZoneUnmaskPost()
//! \brief <mhd> atmosphere_zone_unmask: zone state for cells the post-C2P mask rebuild
//! un-excised (see PrimitiveSolverHydro::ZoneUnmaskPost)
template<class EOSPolicy, class ErrorPolicy>
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::ZoneUnmaskPost() {
  const bool mc = (pmclose != nullptr) && eos.zone_unmask;
  if (mc) pmclose->MarkBefore();
  eos.ZoneUnmaskPost(pmy_pack->pmhd->u0, pmy_pack->pmhd->bcc0, pmy_pack->pmhd->w0,
                     temperature, pmy_pack->pmesh->time, pmy_pack->pmesh->ncycle);
  if (mc) pmclose->MarkAfter(mclose::iUnmask);   // MCLOSE item 4, the F1 unmask
}

//----------------------------------------------------------------------------------------
//! \fn void DynGRMHDPS::RefreshExcisionMasks()
//! \brief <coord> excision_mask_refresh: rebuild the lapse-scheme masks from the current
//! lapse before every C2P (see coordinates.cpp).
template<class EOSPolicy, class ErrorPolicy>
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::RefreshExcisionMasks() {
  auto &cd = pmy_pack->pcoord->coord_data;
  if (cd.bh_excise && cd.excision_mask_refresh &&
      cd.excision_scheme == ExcisionScheme::lapse) {
    SetExcisionGeometry();
    pmy_pack->pcoord->UpdateExcisionMasks();
  }
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus DynGRMHD::ADMMatterSource_(Driver *pdrive, int stage) {
//  \brief
template<class EOSPolicy, class ErrorPolicy>
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddCoordTerms(const DvceArray5D<Real> &prim,
    const DvceArray5D<Real> &bcc,
    const Real dt, DvceArray5D<Real> &rhs, int nghost) {
  if (metric_terms != MetricTerms::cell) {   // dyn_grmhd_metric.cpp
    switch (nghost) {
      case 2: AddCoordTermsFM<2>(prim, bcc, dt, rhs);
              break;
      case 3: AddCoordTermsFM<3>(prim, bcc, dt, rhs);
              break;
      case 4: AddCoordTermsFM<4>(prim, bcc, dt, rhs);
              break;
    }
    return;
  }
  switch (nghost) {
    case 2: AddCoordTermsEOS<2>(prim, bcc, dt, rhs);
            break;
    case 3: AddCoordTermsEOS<3>(prim, bcc, dt, rhs);
            break;
    case 4: AddCoordTermsEOS<4>(prim, bcc, dt, rhs);
            break;
  }
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus DynGRMHD::ApplyPhysicalBCs(Driver *pdrive, int stage)
//  \brief
TaskStatus DynGRMHD::ApplyPhysicalBCs(Driver *pdrive, int stage) {
  // do not apply BCs if domain is strictly periodic
  if (pmy_pack->pmesh->strictly_periodic) return TaskStatus::complete;

  // We need the first physical point on all the boundaries in order to calculate
  // the boundaries. So, we need to perform a ConToPrim at these points.
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto pm = pmy_pack->pmesh;
  auto pmhd = pmy_pack->pmhd;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  // X1-boundary
  ConToPrimBC(is-ng, is+ng, 0, (n2-1), 0, (n3-1));
  ConToPrimBC(ie-ng, ie+ng, 0, (n2-1), 0, (n3-1));
  // X2-boundary
  if (pm->multi_d) {
    ConToPrimBC(0, (n1-1), js-ng, js+ng, 0, (n3-1));
    ConToPrimBC(0, (n1-1), je-ng, je+ng, 0, (n3-1));
  }
  // X3-boundary
  if (pm->three_d) {
    ConToPrimBC(0, (n1-1), 0, (n2-1), ks-ng, ks+ng);
    ConToPrimBC(0, (n1-1), 0, (n2-1), ke-ng, ke+ng);
  }

  // Physical boundaries
  pmhd->pbval_u->HydroBCs((pmy_pack), (pmhd->pbval_u->u_in), pmhd->w0);
  pmhd->pbval_b->BFieldBCs((pmy_pack), (pmhd->pbval_b->b_in), pmhd->b0);

  // User BCs
  if (pmy_pack->pmesh->pgen->user_bcs) {
    (pmy_pack->pmesh->pgen->user_bcs_func)(pmy_pack->pmesh);
  }

  // We now need to do a PrimToCon on all these boundary points.
  // X1-boundary
  PrimToConInit(is-ng, is, 0, (n2-1), 0, (n3-1));
  PrimToConInit(ie, ie+ng, 0, (n2-1), 0, (n3-1));
  // X2-boundary
  if (pm->multi_d) {
    PrimToConInit(0, (n1-1), js-ng, js, 0, (n3-1));
    PrimToConInit(0, (n1-1), je, je+ng, 0, (n3-1));
  }
  // X3-boundary
  if (pm->three_d) {
    PrimToConInit(0, (n1-1), 0, (n2-1), ks-ng, ks);
    PrimToConInit(0, (n1-1), 0, (n2-1), ke, ke+ng);
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus DynGRMHD::SetTmunu(Driver *pdrive, int stage)
//! \brief Add the perfect fluid contribution to the stress-energy tensor. This is assumed
//!  to be the first contribution, so it sets the values rather than adding.
TaskStatus DynGRMHD::SetTmunu(Driver *pdrive, int stage) {
  if (fixed_evolution) {
    return TaskStatus::complete;
  }
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  //auto &size  = pmy_pack->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  int nmb = pmy_pack->nmb_thispack;

  auto &adm = pmy_pack->padm->adm;
  auto &tmunu = pmy_pack->ptmunu->tmunu;
  //auto &nhyd = pmy_pack->pmhd->nmhd;
  //int &nscal = pmy_pack->pmhd->nscalars;
  auto &prim = pmy_pack->pmhd->w0;
  // TODO(JMF): double-check that this needs to be u1, not u0!
  auto &cons = pmy_pack->pmhd->u0;
  auto &bcc = pmy_pack->pmhd->bcc0;

  par_for("dyngr_tmunu_loop",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Calculate the determinant/volume form
    Real detg = adm::SpatialDet(adm.g_dd(m,0,0,k,j,i),adm.g_dd(m,0,1,k,j,i),
                                adm.g_dd(m,0,2,k,j,i),adm.g_dd(m,1,1,k,j,i),
                                adm.g_dd(m,1,2,k,j,i),adm.g_dd(m,2,2,k,j,i));
    Real ivol = 1.0/sqrt(detg);

    // Calculate the lower velocity components
    Real v_d[3] = {0.0};
    Real iW = 0.;
    Real B_d[3] = {0.0};
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        v_d[a] += prim(m, IVX + b, k, j, i)*adm.g_dd(m, a, b, k, j, i);
        iW += prim(m, IVX + a, k, j, i)*prim(m, IVX + b, k, j, i) *
                adm.g_dd(m, a, b, k, j, i);
        B_d[a] += bcc(m, b, k, j, i)*adm.g_dd(m, a, b, k, j, i)*ivol;
      }
    }
    iW = 1.0/sqrt(1. + iW);
    Real Bv = 0.;
    Real Bsq = 0.;
    for (int a = 0; a < 3; ++a) {
      Bv += bcc(m, a, k, j, i) * v_d[a]*ivol;
      Bsq += bcc(m, a, k, j, i) * B_d[a]*ivol;
    }
    Real bsq = (Bsq + Bv*Bv)*(iW*iW);

    tmunu.E(m, k, j, i) = (cons(m, IEN, k, j, i) + cons(m, IDN, k, j, i))*ivol;
    for (int a = 0; a < 3; ++a) {
      tmunu.S_d(m, a, k, j, i) = cons(m, IM1 + a, k, j, i)*ivol;
      for (int b = a; b < 3; ++b) {
        tmunu.S_dd(m, a, b, k, j, i) =
              cons(m, IM1 + a, k, j, i)*ivol*v_d[b]*iW
              - (B_d[a] + Bv*v_d[a])*SQR(iW)*B_d[b]
              + (prim(m, IPR, k, j, i) + 0.5*bsq)*adm.g_dd(m, a, b, k, j, i);
      }
    }
  });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void DynGRMHD::SetADMVariables
//! \brief

TaskStatus DynGRMHD::SetADMVariables(Driver *pdrive, int stage) {
  pmy_pack->padm->SetADMVariables(pmy_pack);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void Z4c::UpdateExcisionMasks
//! \brief

TaskStatus DynGRMHD::UpdateExcisionMasks(Driver *pdrive, int stage) {
  if (pmy_pack->pcoord->coord_data.bh_excise && stage == pdrive->nexp_stages) {
    // the final stage's rebuild is the between-cycles point: the next cycle's r_f is set
    // here and held through all its stages (the per-C2P refresh reuses it)
    UpdateExcisionRadius(pdrive);
    ZoneUnmaskPre();
    SetExcisionGeometry();
    pmy_pack->pcoord->UpdateExcisionMasks();
    ZoneUnmaskPost();
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void DynGRMHD::SetExcisionGeometry()
//! \brief hand the time-dependent excision radius and the hole's position (padm->bc_center,
//! the zone's centre) to Coordinates before a lapse-mask rebuild. No-op unless
//! <coord> excision_radius_dynamic.
void DynGRMHD::SetExcisionGeometry() {
  auto &cd = pmy_pack->pcoord->coord_data;
  cd.excision_dynamic = exc_dyn;
  cd.excision_rf = exc_dyn ? exc_rf : 0.0;
  cd.exc_cx = pmy_pack->padm->bc_center[0];
  cd.exc_cy = pmy_pack->padm->bc_center[1];
  cd.exc_cz = pmy_pack->padm->bc_center[2];
}

//----------------------------------------------------------------------------------------
//! \fn void DynGRMHD::UpdateExcisionRadius(Driver *pdrive)
//! \brief the R-106 rule for the next cycle's r_f (research TIME_DEP_EXCISION_DESIGN.md
//! sec. 2, 6): d_min = distance from the hole to the nearest interior cell with
//! rho > exc_rho_thr (a global min, order-independent), dx = the cell size of the block
//! holding the hole (the finest level there), r_target = min(rmax, d_min - margin dx -
//! vin dt) (>= 0); grow by at most dx per cycle, shrink at once. Aborts if dense gas lies
//! within r_f + 2 dx (cannot happen with margin >= 2; a safety check). Books the rest mass
//! the C2P removed in cells excised only by r < r_f, and prints one EXCDYN line per cycle.
void DynGRMHD::UpdateExcisionRadius(Driver *pdrive) {
  if (!exc_dyn) return;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmb = pmy_pack->nmb_thispack;
  auto &size = pmy_pack->pmb->mb_size;
  auto &w0 = pmy_pack->pmhd->w0;
  const Real cx = pmy_pack->padm->bc_center[0], cy = pmy_pack->padm->bc_center[1];
  const Real cz = pmy_pack->padm->bc_center[2];
  const Real thr = exc_rho_thr;
  const int nkji = nx3*nx2*nx1, nji = nx2*nx1;
  Real d2 = std::numeric_limits<Real>::max();
  Kokkos::parallel_reduce("exc_dmin", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*nkji),
  KOKKOS_LAMBDA(const int idx, Real &dmin2) {
    const int m = idx/nkji;
    const int k = (idx - m*nkji)/nji + ks;
    const int j = (idx - m*nkji - (k - ks)*nji)/nx1 + js;
    const int i = idx - m*nkji - (k - ks)*nji - (j - js)*nx1 + is;
    if (!(w0(m, IDN, k, j, i) > thr)) return;
    const Real x = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max) - cx;
    const Real y = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max) - cy;
    const Real z = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max) - cz;
    const Real r2 = x*x + y*y + z*z;
    if (r2 < dmin2) dmin2 = r2;
  }, Kokkos::Min<Real>(d2));
  Real dxh = std::numeric_limits<Real>::max();
  for (int m = 0; m < nmb; ++m) {
    auto &b = size.h_view(m);
    if (cx >= b.x1min && cx < b.x1max && cy >= b.x2min && cy < b.x2max &&
        cz >= b.x3min && cz < b.x3max) {
      dxh = std::min(dxh, b.dx1);
    }
  }
  Real absorbed = DrainAbsorbedAtRf();
#if MPI_PARALLEL_ENABLED
  Real mn[2] = {d2, dxh}, mx[2];
  MPI_Allreduce(mn, mx, 2, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
  d2 = mx[0]; dxh = mx[1];
  MPI_Allreduce(MPI_IN_PLACE, &absorbed, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  exc_abs_total += absorbed;
  const Real dmin = std::sqrt(d2);
  if (!(dxh < std::numeric_limits<Real>::max())) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "excision_radius_dynamic: no MeshBlock contains the hole position" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  Real target = std::min(exc_rmax, dmin - exc_margin*dxh - exc_vin*pmy_pack->pmesh->dt);
  if (!(target > 0.0)) target = 0.0;
  const Real old = exc_rf;
  bool shrink = false, grow = false;
  if (target < old) {
    exc_rf = target; shrink = true; ++exc_nshrink;
  } else {
    exc_rf = std::min(target, old + dxh);
    if (exc_rf > old) { grow = true; ++exc_ngrow; }
  }
  if (exc_rf > 0.0 && dmin < exc_rf + 2.0*dxh) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "excision_radius_dynamic guard: gas with rho > " << thr << " at r = " << dmin
              << " within r_f + 2 dx = " << exc_rf + 2.0*dxh << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (global_variable::my_rank == 0) {
    std::cout << std::setprecision(10) << "EXCDYN cycle=" << pmy_pack->pmesh->ncycle
              << " time=" << pmy_pack->pmesh->time + pmy_pack->pmesh->dt << " r_f=" << exc_rf
              << " r_f_old=" << old << " d_min=" << dmin << " dx=" << dxh << " margin="
              << dmin - exc_rf << " absorbed=" << absorbed << " absorbed_total="
              << exc_abs_total << " shrink=" << shrink << " grow=" << grow << " nshrink="
              << exc_nshrink << " ngrow=" << exc_ngrow << std::endl;
  }
}

template<class EOSPolicy, class ErrorPolicy> template<int NGHOST>
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddCoordTermsEOS(const DvceArray5D<Real> &prim,
    const DvceArray5D<Real> &bcc,
    const Real dt, DvceArray5D<Real> &rhs) {
  if (fixed_evolution) {
    return;
  }
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size  = pmy_pack->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  int nmb = pmy_pack->nmb_thispack;

  auto &adm = pmy_pack->padm->adm;
  auto &eos_ = eos.ps.GetEOS();
  //auto &tmunu = pmy_pack->ptmunu->tmunu;

  // fetch flag for smooth excision and
  // excision mask, and target values
  bool smoothing = pmy_pack->pcoord->coord_data.smooth_excision;
  auto &floor = pmy_pack->pcoord->excision_floor;
  Real &dexcise = pmy_pack->pcoord->coord_data.dexcise;
  // Real &pexcise = pmy_pack->pcoord->coord_data.pexcise;
  Real &texcise = pmy_pack->pcoord->coord_data.texcise;
  Real &tdamp = pmy_pack->pcoord->coord_data.tdamp;

  int &nhyd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;

  const Real mb = eos.ps.GetEOS().GetBaryonMass();
  const int imap[3][3] = {
    {S11, S12, S13},
    {S12, S22, S23},
    {S13, S23, S33}
  };

  // Check the number of dimensions to determine which derivatives we need.
  int ndim;
  if (pmy_pack->pmesh->one_d) {
    ndim = 1;
  } else if (pmy_pack->pmesh->two_d) {
    ndim = 2;
  } else {
    ndim = 3;
  }

  const bool ddon_ = eos.de_diag_on;
  auto &dd_ = de_diag;
  // stage fluxes (densitized; already applied to rhs by MHD::Update before this task)
  auto &fx1_ = pmy_pack->pmhd->uflx.x1f;
  auto &fx2_ = pmy_pack->pmhd->uflx.x2f;
  auto &fx3_ = pmy_pack->pmhd->uflx.x3f;
  const bool md_ = pmy_pack->pmesh->multi_d, d3_ = pmy_pack->pmesh->three_d;
  par_for("coord_src", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Extract the metric and coordinate quantities.
    Real g3d[NSPMETRIC] = {adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
                           adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
                           adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i)};
    const Real& alpha = adm.alpha(m, k, j, i);
    Real detg = adm::SpatialDet(g3d[S11], g3d[S12], g3d[S13],
                                g3d[S22], g3d[S23], g3d[S33]);
    Real vol = sqrt(detg);
    Real g3u[NSPMETRIC] = {0.};
    adm::SpatialInv(1.0/detg, g3d[S11], g3d[S12], g3d[S13], g3d[S22], g3d[S23], g3d[S33],
                    &g3u[S11], &g3u[S12], &g3u[S13], &g3u[S22], &g3u[S23], &g3u[S33]);

    // Calculate the metric derivatives
    Real idx[] = {1./size.d_view(m).dx1, 1./size.d_view(m).dx2, 1./size.d_view(m).dx3};
    Real dalpha_d[3] = {0.};
    for (int a = 0; a < ndim; a++) {
      dalpha_d[a] = Dx<NGHOST>(a, idx, adm.alpha, m, k, j, i);
    }
    Real dbeta_du[3][3] = {};
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < ndim; b++) {
        dbeta_du[b][a] = Dx<NGHOST>(b, idx, adm.beta_u, m, a, k, j, i);
      }
    }
    Real dg_ddd[3][3][3] = {};
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        for (int c = 0; c < ndim; ++c) {
          dg_ddd[c][a][b] = Dx<NGHOST>(c, idx, adm.g_dd, m, a, b, k, j, i);
        }
      }
    }

    // Fluid quantities
    Real prim_pt[NPRIM] = {0.0};
    prim_pt[PRH] = prim(m, IDN, k, j, i)/mb;
    prim_pt[PVX] = prim(m, IVX, k, j, i);
    prim_pt[PVY] = prim(m, IVY, k, j, i);
    prim_pt[PVZ] = prim(m, IVZ, k, j, i);
    for (int s = 0; s < nscal; s++) {
      prim_pt[PYF + s] = prim(m, nhyd + s, k, j, i);
    }
    prim_pt[PPR] = prim(m, IPR, k, j, i);
    prim_pt[PTM] = eos_.GetTemperatureFromP(prim_pt[PRH], prim_pt[PPR], &prim_pt[PYF]);

    // Get the conserved variables. Note that we don't use PrimitiveSolver here --
    // that's because we would need to recalculate quantities used in E and S_d in order
    // to get S_dd.
    Real H =
      prim(m, IDN, k, j, i)*eos_.GetEnthalpy(prim_pt[PRH], prim_pt[PTM], &prim_pt[PYF]);
    Real usq = Primitive::SquareVector(&prim_pt[PVX], g3d);
    Real const Wsq = 1.0 + usq;
    Real const W = sqrt(Wsq);
    Real B_u[NMAG] = {bcc(m, IBX, k, j, i)/vol,
                      bcc(m, IBY, k, j, i)/vol,
                      bcc(m, IBZ, k, j, i)/vol};
    Real Bv = 0.0;
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3; b++) {
        Bv += adm.g_dd(m,a,b,k,j,i)*prim_pt[PVX + a]*B_u[b];
      }
    }
    Real Bsq = Primitive::SquareVector(B_u, g3d);
    Bv = Bv/W;
    Real bsq = Bv*Bv + Bsq/Wsq;

    Real E = (H*Wsq + Bsq) - prim_pt[PPR] - 0.5*bsq;
    //Real E = tmunu.E(m,k,j,i);

    Real S_d[3] = {0.0};
    for (int a = 0; a < 3; a++) {
      //S_d[a] = tmunu.S_d(m,a,k,j,i);
      for (int b = 0; b < 3; b++) {
        S_d[a] += ((H*Wsq + Bsq)*prim_pt[PVX + b]/W - Bv*B_u[b])*g3d[imap[a][b]];
      }
    }

    Real S_uu[3][3];
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b <= a; b++) {
        S_uu[a][b] = (H + Bsq/Wsq)*prim_pt[PVX + a]*prim_pt[PVX + b]
                      - B_u[a]*B_u[b]/Wsq
                      - Bv*(B_u[a]*prim_pt[PVX + b] + B_u[b]*prim_pt[PVX + a])/W
                      + (prim_pt[PPR] + 0.5*bsq)*g3u[imap[a][b]];
        /*S_uu[a][b] = 0.0;
        for (int c = 0; c < 3; c++) {
          for (int d = 0; d < 3; d++) {
            S_uu[a][b] += tmunu.S_dd(m,c,d,k,j,i)*g3u[imap[a][c]]*g3u[imap[b][d]];
          }
        }*/
        S_uu[b][a] = S_uu[a][b];
      }
    }

    // Assemble energy RHS
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3; b++) {
        rhs(m, IEN, k, j, i) += dt*vol*(alpha*adm.vK_dd(m, a, b, k, j, i)*S_uu[a][b] -
            g3u[imap[a][b]] * S_d[a]*dalpha_d[b]);
      }
    }
    if (ddon_) {
      Real src_ks = 0.0, src_da = 0.0;
      for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
          src_ks += alpha*adm.vK_dd(m, a, b, k, j, i)*S_uu[a][b];
          src_da -= g3u[imap[a][b]] * S_d[a]*dalpha_d[b];
        }
      }
      dd_(m, 6, k, j, i) = src_ks;
      dd_(m, 7, k, j, i) = src_da;
      // internal-energy budget: W (dE - v^a dS_a) - dD (E = tau + D) for the flux
      // divergence and for the sources (D has none); E - v.S = rho (1 + eps) exactly
      Real srcS[3];
      for (int a = 0; a < 3; a++) {
        srcS[a] = -E*dalpha_d[a];
        for (int b = 0; b < 3; b++) {
          srcS[a] += S_d[b]*dbeta_du[a][b];
          for (int c = 0; c < 3; c++) {
            srcS[a] += 0.5*alpha*S_uu[b][c]*dg_ddd[a][b][c];
          }
        }
      }
      auto divf = [&](int n) {
        Real d = (fx1_(m, n, k, j, i+1) - fx1_(m, n, k, j, i))/size.d_view(m).dx1;
        if (md_) d += (fx2_(m, n, k, j+1, i) - fx2_(m, n, k, j, i))/size.d_view(m).dx2;
        if (d3_) d += (fx3_(m, n, k+1, j, i) - fx3_(m, n, k, j, i))/size.d_view(m).dx3;
        return d/vol;
      };
      Real dD = -divf(IDN), dE = -divf(IEN) + dD;
      Real vdS_f = 0.0, vdS_s = 0.0;
      for (int a = 0; a < 3; a++) {
        Real va = prim_pt[PVX + a]/W;
        vdS_f -= va*divf(IM1 + a);
        vdS_s += va*srcS[a];
      }
      Real rho_ = prim(m, IDN, k, j, i);
      Real e_int = W*fmax(H - rho_ - prim_pt[PPR], 1.0e-300);
      dd_(m, 8, k, j, i) = (W*(dE - vdS_f) - dD)/e_int;
      dd_(m, 9, k, j, i) = W*(src_ks + src_da - vdS_s)/e_int;
    }

    // Assemble momentum RHS
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3; b++) {
        for (int c = 0; c < 3; c++) {
          rhs(m,IM1+a, k, j, i) += 0.5*dt*alpha*vol*S_uu[b][c]*dg_ddd[a][b][c];
        }
        rhs(m, IM1+a, k, j, i) += dt*vol*S_d[b]*dbeta_du[a][b];
      }
      rhs(m, IM1+a, k, j, i) -= dt*vol*E*dalpha_d[a];
    }

    // Assemble damping source terms
    if (smoothing) {
      // D = rho*W and tau = E-D are needed for the smooth damping terms
      // inside excised regions, if existing.
      Real D = prim(m, IDN, k, j, i) * W;
      Real tau = E - D;

      // Compute the excised value for the energy.
      // Real tau_ex = (dexcise*eos_.GetEnthalpy(dexcise/mb, texcise, &prim_pt[PYF]))
      //               + Bsq - pexcise - 0.5*bsq - dexcise;
      Real tau_ex = eos_.GetEnergy(dexcise/mb, texcise, &prim_pt[PYF])
                                    + 0.5*Bsq - dexcise;

      rhs(m, IDN, k, j, i) -= (dt*vol*floor(m,k,j,i)*(D-dexcise))/tdamp;
      rhs(m, IM1, k, j, i) -= (dt*floor(m,k,j,i)*vol*S_d[0])/tdamp;
      rhs(m, IM2, k, j, i) -= (dt*floor(m,k,j,i)*vol*S_d[1])/tdamp;
      rhs(m, IM3, k, j, i) -= (dt*floor(m,k,j,i)*vol*S_d[2])/tdamp;
      rhs(m, IEN, k, j, i) -= dt*vol*floor(m,k,j,i)*(tau - tau_ex)/tdamp;
      for (int s = 0; s < nscal; s++) {
        rhs(m, IYF+s, k, j, i) -= (dt*vol*floor(m,k,j,i)*(D-dexcise)*prim_pt[PYF+s])
                                      /tdamp;
      }
    }
  });
}

// Instantiated templates
template class DynGRMHDPS<Primitive::IdealGas, Primitive::ResetFloor>;
template class DynGRMHDPS<Primitive::PiecewisePolytrope, Primitive::ResetFloor>;
template class DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                          Primitive::ResetFloor>;
template class DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                          Primitive::ResetFloor>;
template class DynGRMHDPS<Primitive::EOSHybrid<Primitive::NormalLogs>,
                          Primitive::ResetFloor>;
template class DynGRMHDPS<Primitive::EOSHybrid<Primitive::NQTLogs>,
                          Primitive::ResetFloor>;

// Macro for defining CoordTerms templates
#define INSTANTIATE_COORD_TERMS(EOSPolicy, ErrorPolicy) \
template \
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddCoordTermsEOS<2>( \
      const DvceArray5D<Real> &prim, \
      const DvceArray5D<Real> &bcc, const Real dt, DvceArray5D<Real> &rhs); \
template \
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddCoordTermsEOS<3>( \
      const DvceArray5D<Real> &prim, \
      const DvceArray5D<Real> &bcc, const Real dt, DvceArray5D<Real> &rhs); \
template \
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddCoordTermsEOS<4>( \
      const DvceArray5D<Real> &prim, \
      const DvceArray5D<Real> &bcc, const Real dt, DvceArray5D<Real> &rhs);

INSTANTIATE_COORD_TERMS(Primitive::IdealGas, Primitive::ResetFloor);
INSTANTIATE_COORD_TERMS(Primitive::PiecewisePolytrope, Primitive::ResetFloor);
INSTANTIATE_COORD_TERMS(Primitive::EOSCompOSE<Primitive::NormalLogs>,
                        Primitive::ResetFloor);
INSTANTIATE_COORD_TERMS(Primitive::EOSCompOSE<Primitive::NQTLogs>, Primitive::ResetFloor);
INSTANTIATE_COORD_TERMS(Primitive::EOSHybrid<Primitive::NormalLogs>,
                        Primitive::ResetFloor);
INSTANTIATE_COORD_TERMS(Primitive::EOSHybrid<Primitive::NQTLogs>, Primitive::ResetFloor);

#undef INSTANTIATE_COORD_TERMS

} // namespace dyngr
