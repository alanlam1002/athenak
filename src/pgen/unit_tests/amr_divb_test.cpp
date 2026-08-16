//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file amr_divb_test.cpp
//! \brief Acceptance gate for curvilinear SMR/AMR Phase 2 (face-centered / MHD):
//! div(B) must stay at roundoff, and total mass must be conserved, across many refine
//! AND derefine cycles on a curvilinear grid.
//!
//! This is the face-centered counterpart of amr_conservation_test.cpp, and it inherits
//! that test's two hard-won design rules:
//!
//!   1. THE CHECK LIVES IN THE PGEN, not in a Python wrapper reading the history file.
//!      The .hst file carries ~6 significant digits, which cannot support a
//!      roundoff-level gate.
//!   2. THE REFINEMENT PATTERN MUST OSCILLATE. A criterion that only ever ADDS blocks
//!      never exercises restriction, which is the half of the machinery that only matters
//!      during DEREFINEMENT. The pattern below alternates between the inner and outer
//!      half of x1 so that blocks are continuously created and destroyed.
//!
//! WHAT EACH INGREDIENT TESTS
//!   * derefinement   -> RestrictFC             (area-weighted, sum(A_f B_f)/sum(A_f))
//!   * refinement     -> ProlongFCShared*       (area-weighted transverse centroids)
//!                       ProlongFCInternal      (Toth-Roe recast onto fluxes A*B)
//!   * every RK stage -> flux_correct_fc.cpp    (Len-weighted EMF correction)
//! The cell-centered Phase 1 machinery is re-checked at the same time via the mass sum.
//!
//! WHY div(B) IS THE RIGHT INVARIANT. It is a topological identity: given ANY set of face
//! fields, the Stokes-form divergence below telescopes to zero if and only if the
//! prolongation/restriction operators moved the FLUXES A*B consistently. It therefore
//! cannot be satisfied by accident on a curvilinear grid, where the unweighted operators
//! move the pointwise B instead. The initial field is built as a discrete curl so that
//! div(B) starts at roundoff and any growth is attributable to the refinement machinery.
//!
//! The divergence is measured in the SAME area-weighted form the CT update integrates
//! (mhd_ct.cpp), which is what makes "div(B) at roundoff" meaningful rather than a
//! statement about a mismatched discretization.

#include <cmath>
#include <iostream>
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
#include "mhd/mhd.hpp"

namespace {

Real mass_initial = -1.0;
Real divb_tol = 1.0e-10;
Real mass_tol = 1.0e-11;
int  ref_ncycle = 20;
int  target_level = 1;

//----------------------------------------------------------------------------------------
//! \fn MaxRelDivB()
//! \brief max |sum(+/- Area*B_n)| over active cells, normalized by a per-cell face-flux
//! scale. Identical form to ct_divb_test.cpp so the two are directly comparable.

Real MaxRelDivB(MeshBlockPack *pmbp) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  auto &geom = pmbp->pgeom->geom_data;
  auto &b0 = pmbp->pmhd->b0;
  int nmb = pmbp->nmb_thispack;
  int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
  int nkji = nx3*nx2*nx1, nji = nx2*nx1;

  Real max_divb = 0.0, max_scale = 0.0;
  Kokkos::parallel_reduce("amr_divb_check",
                          Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*nkji),
  KOKKOS_LAMBDA(const int &idx, Real &mx, Real &mscale) {
    int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    Real divb = (geom.Area1(m,k,j,i+1)*b0.x1f(m,k,j,i+1)
                 - geom.Area1(m,k,j,i)*b0.x1f(m,k,j,i))
              + (geom.Area2(m,k,j+1,i)*b0.x2f(m,k,j+1,i)
                 - geom.Area2(m,k,j,i)*b0.x2f(m,k,j,i))
              + (geom.Area3(m,k+1,j,i)*b0.x3f(m,k+1,j,i)
                 - geom.Area3(m,k,j,i)*b0.x3f(m,k,j,i));
    mx = fmax(mx, fabs(divb));
    Real scale = geom.Area1(m,k,j,i)*fabs(b0.x1f(m,k,j,i))
               + geom.Area2(m,k,j,i)*fabs(b0.x2f(m,k,j,i))
               + geom.Area3(m,k,j,i)*fabs(b0.x3f(m,k,j,i));
    mscale = fmax(mscale, scale);
  }, Kokkos::Max<Real>(max_divb), Kokkos::Max<Real>(max_scale));

  return max_divb/std::fmax(max_scale, 1.0e-300);
}

//----------------------------------------------------------------------------------------
//! \fn TotalMass()
//! \brief sum(rho*Vol) in double precision, re-checking the Phase 1 cell-centered path.

Real TotalMass(MeshBlockPack *pmbp) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  auto &u0 = pmbp->pmhd->u0;
  auto &geom = pmbp->pgeom->geom_data;

  const int nkji = (ke-ks+1)*(je-js+1)*(ie-is+1);
  const int nji  = (je-js+1)*(ie-is+1);
  const int ni   = (ie-is+1);
  Real mass = 0.0;
  Kokkos::parallel_reduce("amr_divb_mass",
                          Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*nkji),
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

//----------------------------------------------------------------------------------------
//! \fn A3Fn()
//! \brief the edge-centred vector potential B is built from.
//!
//! A free KOKKOS_INLINE_FUNCTION rather than a KOKKOS_LAMBDA captured into the kernels
//! below: nvcc restricts extended lambdas nested inside other extended lambdas, and this
//! branch has not been through a device compiler yet. Matches the convention upstream's
//! own pgen/tests/divb_amr.cpp uses for exactly this job.
KOKKOS_INLINE_FUNCTION
Real A3Fn(const Real xa, const Real xb, const Real amp,
          const Real x1min, const Real dx1, const Real x2min, const Real dx2) {
  Real u = (xa - x1min)/dx1;
  Real v = (xb - x2min)/dx2;
  return amp*sin(2.0*M_PI*u)*sin(2.0*M_PI*v);
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn AMRDivBRefine()
//! \brief oscillating forced refinement, so blocks are both created and destroyed. Copied
//! in structure from amr_conservation_test.cpp -- see the note there on why a
//! monotonically-refining criterion makes this whole test vacuous.

void AMRDivBRefine(MeshBlockPack *pmbp) {
  Mesh *pm = pmbp->pmesh;
  auto &refine_flag = pm->pmr->refine_flag;
  auto &mblev = pmbp->pmb->mb_lev;
  auto &mb_size = pmbp->pmb->mb_size;
  int nmb = pmbp->nmb_thispack;
  int mbs = pm->gids_eachrank[global_variable::my_rank];
  int root_level = pm->root_level;

  // Driven by CYCLE, not time: the timestep varies by orders of magnitude between
  // coordinate systems and resolutions, so a time-based period would silently stop
  // oscillating (and therefore stop testing derefinement) in some configurations.
  bool inner_half = ((pm->ncycle/ref_ncycle) % 2) == 0;
  Real x1mid = 0.5*(pm->mesh_size.x1min + pm->mesh_size.x1max);
  int tgt = target_level;

  par_for("amr_divb_refine", DevExeSpace(), 0, nmb-1, KOKKOS_LAMBDA(const int m) {
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
//! \fn AMRDivBFinal()
//! \brief the gate: div(B) at roundoff and mass conserved, after the refine/derefine run.

void AMRDivBFinal(ParameterInput *pin, Mesh *pm) {
  Real rel_divb = MaxRelDivB(pm->pmb_pack);
  Real mass = TotalMass(pm->pmb_pack);
  Real rel_mass = std::abs(mass - mass_initial)/std::abs(mass_initial);

  std::cout << "amr_divb_test: rel_divb=" << std::scientific << std::setprecision(6)
            << rel_divb << " (tol " << divb_tol << ")  rel_mass_drift=" << rel_mass
            << " (tol " << mass_tol << ")" << std::endl;

  bool failed = false;
  if (!(rel_divb < divb_tol)) {
    std::cout << "### FATAL ERROR amr_divb_test: div(B) grew to " << rel_divb
              << " across refine/derefine cycles. Face-centered restriction, "
              << "prolongation (shared or internal) or EMF correction is not correctly "
              << "geometry-weighted at level boundaries." << std::endl;
    failed = true;
  }
  if (!(rel_mass < mass_tol)) {
    std::cout << "### FATAL ERROR amr_divb_test: total mass drifted by " << rel_mass
              << " across refine/derefine cycles." << std::endl;
    failed = true;
  }
  if (failed) { std::exit(EXIT_FAILURE); }
  std::cout << "amr_divb_test: PASSED" << std::endl;
}

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::AMRDivBTest()
//! \brief B initialized as the discrete Stokes-form curl of an edge-centered vector
//! potential, using the SAME Area/Len tables mhd_ct.cpp reads, so div(B)=0 holds to
//! roundoff at t=0 by construction in every coordinate system.

void ProblemGenerator::AMRDivBTest(ParameterInput *pin, const bool restart) {
  user_ref_func = AMRDivBRefine;
  pgen_final_func = AMRDivBFinal;
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR amr_divb_test requires <mhd>" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  divb_tol = pin->GetOrAddReal("problem", "divb_tol", 1.0e-10);
  mass_tol = pin->GetOrAddReal("problem", "mass_tol", 1.0e-11);
  ref_ncycle = pin->GetOrAddInteger("problem", "ref_ncycle", 20);
  target_level = pin->GetOrAddInteger("problem", "target_level", 1);
  Real d0 = pin->GetOrAddReal("problem", "d0", 1.0);
  Real amp = pin->GetOrAddReal("problem", "amp", 0.3);
  Real p0 = pin->GetOrAddReal("problem", "p0", 1.0);
  Real bz0 = pin->GetOrAddReal("problem", "bz0", 0.2);

  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  auto &u0 = pmbp->pmhd->u0;
  auto &b0 = pmbp->pmhd->b0;
  auto &geom = pmbp->pgeom->geom_data;
  auto &size = pmbp->pmb->mb_size;
  Real gm1 = pmbp->pmhd->peos->eos_data.gamma - 1.0;
  int nx1 = indcs.nx1, nx2 = indcs.nx2;
  Real x1min_mesh = pmy_mesh_->mesh_size.x1min;
  Real x1max_mesh = pmy_mesh_->mesh_size.x1max;
  Real x2min_mesh = pmy_mesh_->mesh_size.x2min;
  Real x2max_mesh = pmy_mesh_->mesh_size.x2max;
  bool multi_d = pmy_mesh_->multi_d;

  // Edge-centered vector potential A3, smooth and non-uniform. A constant field would be
  // reproduced exactly by ANY weighting and would make the test vacuous, so the profile
  // must vary in both resolved directions.
  Real dx1 = (x1max_mesh - x1min_mesh);
  Real dx2 = (x2max_mesh - x2min_mesh);

  // B = curl(A) in Stokes form, built from the same geom tables the CT update uses:
  //   B1*Area1 = +d(A3*Len3)/dx2 ,  B2*Area2 = -d(A3*Len3)/dx1
  // which makes sum(+/- Area*B) telescope to exactly zero cell by cell.
  //
  // THREE SEPARATE KERNELS, one per component, each over that component's own face
  // range. A single kernel over (ks..ke+1, js..je+1, is..ie+1) is wrong: in 2D the x1f
  // and x2f arrays have only ncells3 == 1 in k, so writing k == ke+1 runs off the end of
  // them. (Found the hard way -- it silently corrupted x3f and left div(B) at 2.5e-01.)
  par_for("amr_divb_b1", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real ar1 = geom.Area1(m,k,j,i);
    if (multi_d) {
      Real x1f = LeftEdgeX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      Real x2f = LeftEdgeX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      Real x2fp = LeftEdgeX(j+1-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      // circulation of A3 along the two x3-edges bounding this x1-face
      Real p_lo = A3Fn(x1f, x2f , amp, x1min_mesh, dx1, x2min_mesh, dx2);
      Real p_hi = A3Fn(x1f, x2fp, amp, x1min_mesh, dx1, x2min_mesh, dx2);
      Real a_lo = p_lo*geom.Len3(m,k,j  ,i);
      Real a_hi = p_hi*geom.Len3(m,k,j+1,i);
      b0.x1f(m,k,j,i) = (ar1 > 0.0) ? (a_hi - a_lo)/ar1 : 0.0;
    } else {
      // 1D: the only divergence-free radial field has Area1*B1 = const
      b0.x1f(m,k,j,i) = (ar1 > 0.0) ? amp/ar1 : 0.0;
    }
  });

  par_for("amr_divb_b2", DevExeSpace(), 0, nmb-1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real ar2 = geom.Area2(m,k,j,i);
    if (multi_d) {
      Real x1f = LeftEdgeX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      Real x1fp = LeftEdgeX(i+1-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      Real x2f = LeftEdgeX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      Real p_lo = A3Fn(x1f , x2f, amp, x1min_mesh, dx1, x2min_mesh, dx2);
      Real p_hi = A3Fn(x1fp, x2f, amp, x1min_mesh, dx1, x2min_mesh, dx2);
      Real b_lo = p_lo*geom.Len3(m,k,j,i  );
      Real b_hi = p_hi*geom.Len3(m,k,j,i+1);
      b0.x2f(m,k,j,i) = (ar2 > 0.0) ? -(b_hi - b_lo)/ar2 : 0.0;
    } else {
      b0.x2f(m,k,j,i) = 0.0;
    }
  });

  // uniform-flux x3 component: a constant Phi3 = bz0 keeps d(Area3*B3)/dx3 = 0 exactly
  par_for("amr_divb_b3", DevExeSpace(), 0, nmb-1, ks, ke+1, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real ar3 = geom.Area3(m,k,j,i);
    b0.x3f(m,k,j,i) = (ar3 > 0.0) ? bz0/ar3 : 0.0;
  });

  // Conserved hydro variables, with a non-uniform density so the mass check has teeth.
  par_for("amr_divb_cons", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real s = (x1v - x1min_mesh)/(x1max_mesh - x1min_mesh);
    Real d = d0*(1.0 + 0.5*sin(2.0*M_PI*s));
    u0(m,IDN,k,j,i) = d;
    u0(m,IM1,k,j,i) = 0.0;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;
    Real b1 = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
    Real b2 = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
    Real b3 = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
    u0(m,IEN,k,j,i) = p0/gm1 + 0.5*(b1*b1 + b2*b2 + b3*b3);
  });

  mass_initial = TotalMass(pmbp);
  std::cout << "amr_divb_test: initial rel_divb=" << std::scientific
            << std::setprecision(6) << MaxRelDivB(pmbp) << std::endl;
  return;
}
