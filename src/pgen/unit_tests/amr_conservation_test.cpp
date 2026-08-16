//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file amr_conservation_test.cpp
//! \brief Acceptance gate for curvilinear SMR/AMR Phase 1 (cell-centered): total mass
//! sum(rho*Vol) must be conserved to roundoff across many refine AND derefine cycles.
//!
//! WHY THIS EXISTS RATHER THAN AN .hst-BASED TEST. The obvious approach -- run with
//! adaptive refinement and check the mass column of the history file -- was tried first
//! and is too weak to be a gate, for two independent reasons:
//!
//!   1. The history file is written with ~6 significant digits, so relative drift below
//!      about 1e-6 is invisible. Deliberately breaking the prolongation weighting
//!      produced a drift of 1.07e-5 -- detectable only just, and only in one of the three
//!      coordinate systems.
//!   2. A density-threshold criterion on an outgoing shock only ever ADDS blocks
//!      ("2 MeshBlocks created, 0 deleted"). Restriction changes the total only during
//!      DEREFINEMENT, where 2/4/8 fine cells collapse into one coarse cell, so a run that
//!      never derefines cannot detect a broken RestrictCC at all -- verified by
//!      reverting RestrictCC to unweighted averaging and watching the .hst test still
//!      pass.
//!
//! This pgen fixes both: it computes the mass in double precision inside the code, and it
//! drives a refinement pattern that deliberately OSCILLATES so that blocks are repeatedly
//! refined and then derefined.
//!
//! What each ingredient is actually testing:
//!   * derefinement  -> RestrictCC          (volume-weighted average into a coarse cell)
//!   * refinement    -> ProlongCC           (conservative centroid-based reconstruction)
//!   * level bndries -> flux correction     (area-weighted, every stage)
//! A non-uniform density profile is essential: prolongating a constant is exact for any
//! weighting, so a uniform IC would silently test nothing.

#include <cmath>
#include <iostream>
#include <limits>
#include <iomanip>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/meshblock_pack.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/mesh_geometry.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"

namespace {

// mass at t=0, and the running worst relative drift
Real mass_initial = -1.0;
Real max_rel_drift = 0.0;
Real drift_tol = 1.0e-11;
// refinement pattern control
Real ref_period = 0.02;
int  target_level = 1;

//----------------------------------------------------------------------------------------
//! \fn TotalMass()
//! \brief sum(rho*Vol) over active cells, in double precision on the device.

Real TotalMass(MeshBlockPack *pmbp) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  auto &u0 = pmbp->phydro->u0;
  auto &geom = pmbp->pgeom->geom_data;

  const int nkji = (ke - ks + 1)*(je - js + 1)*(ie - is + 1);
  const int nji  = (je - js + 1)*(ie - is + 1);
  const int ni   = (ie - is + 1);
  Real mass = 0.0;
  Kokkos::parallel_reduce("amr_cons_mass", Kokkos::RangePolicy<>(DevExeSpace(), 0,
                          nmb*nkji),
  KOKKOS_LAMBDA(const int idx, Real &sum) {
    int m = idx/nkji;
    int r = idx - m*nkji;
    int k = r/nji;
    int j = (r - k*nji)/ni;
    int i = (r - k*nji - j*ni) + is;
    j += js;
    k += ks;
    sum += u0(m,IDN,k,j,i)*geom.Vol(m,k,j,i);
  }, Kokkos::Sum<Real>(mass));
  return mass;
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn AMRConservationRefine()
//! \brief forced refinement pattern that OSCILLATES in time, so that blocks are both
//! created and destroyed. Refines the inner half of the x1 range during one half of each
//! period and the outer half during the other, which guarantees a steady stream of
//! derefinements -- the only way RestrictCC ever affects the total.

void AMRConservationRefine(MeshBlockPack *pmbp) {
  Mesh *pm = pmbp->pmesh;
  auto &refine_flag = pm->pmr->refine_flag;
  auto &mblev = pmbp->pmb->mb_lev;
  auto &mb_size = pmbp->pmb->mb_size;
  int nmb = pmbp->nmb_thispack;
  int mbs = pm->gids_eachrank[global_variable::my_rank];
  int root_level = pm->root_level;

  // which half is refined right now
  Real phase = pm->time/ref_period;
  bool inner_half = ((static_cast<int>(phase)) % 2) == 0;
  Real x1mid = 0.5*(pm->mesh_size.x1min + pm->mesh_size.x1max);
  int tgt = target_level;

  par_for("amr_cons_refine", DevExeSpace(), 0, nmb-1, KOKKOS_LAMBDA(const int m) {
    Real xc = 0.5*(mb_size.d_view(m).x1min + mb_size.d_view(m).x1max);
    bool in_region = inner_half ? (xc < x1mid) : (xc >= x1mid);
    int level = mblev.d_view(m);
    if (in_region && (level < root_level + tgt)) {
      refine_flag.d_view(m + mbs) = 1;
    } else if ((!in_region) && (level > root_level)) {
      refine_flag.d_view(m + mbs) = -1;
    }
  });
  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();
}

//----------------------------------------------------------------------------------------
//! \fn AMRConservationFinal()
//! \brief compares the final mass against the value recorded at t=0 and fatals on drift.

void AMRConservationFinal(ParameterInput *pin, Mesh *pm) {
  Real mass = TotalMass(pm->pmb_pack);
  Real rel = std::abs(mass - mass_initial)/std::abs(mass_initial);
  max_rel_drift = std::max(max_rel_drift, rel);
  std::cout << "amr_conservation_test: mass_initial=" << std::scientific
            << std::setprecision(16) << mass_initial << " mass_final=" << mass
            << " rel_drift=" << rel << std::endl;
  if (!(rel < drift_tol)) {
    std::cout << "### FATAL ERROR amr_conservation_test: total mass drifted by "
              << rel << " (tolerance " << drift_tol << ") across refine/derefine "
              << "cycles. Restriction, prolongation or flux correction is not "
              << "correctly geometry-weighted at level boundaries." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "amr_conservation_test: PASSED" << std::endl;
}

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::AMRConservationTest()
//! \brief non-uniform density profile + oscillating forced refinement.

void ProblemGenerator::AMRConservationTest(ParameterInput *pin, const bool restart) {
  user_ref_func = AMRConservationRefine;
  pgen_final_func = AMRConservationFinal;
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR amr_conservation_test requires <hydro>" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  drift_tol = pin->GetOrAddReal("problem", "drift_tol", 1.0e-11);
  ref_period = pin->GetOrAddReal("problem", "ref_period", 0.02);
  target_level = pin->GetOrAddInteger("problem", "target_level", 1);
  Real d0 = pin->GetOrAddReal("problem", "d0", 1.0);
  Real amp = pin->GetOrAddReal("problem", "amp", 0.5);
  Real p0 = pin->GetOrAddReal("problem", "p0", 1.0);

  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  auto &u0 = pmbp->phydro->u0;
  auto &size = pmbp->pmb->mb_size;
  Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;
  int nx1 = indcs.nx1;
  Real x1min_mesh = pmy_mesh_->mesh_size.x1min;
  Real x1max_mesh = pmy_mesh_->mesh_size.x1max;

  // Smooth, strongly NON-uniform density: a constant would make prolongation exact for
  // any weighting and the test vacuous. Velocity is zero so the only way mass can change
  // is through the refinement machinery itself.
  par_for("amr_cons_init", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real s = (x1v - x1min_mesh)/(x1max_mesh - x1min_mesh);
    Real d = d0*(1.0 + amp*std::sin(2.0*M_PI*s));
    u0(m,IDN,k,j,i) = d;
    u0(m,IM1,k,j,i) = 0.0;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;
    u0(m,IEN,k,j,i) = p0/gm1;
  });

  mass_initial = TotalMass(pmbp);
  return;
}
