//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ct_monopole_stationarity_test.cpp
//! \brief Unit test for Task D3 (second half): 1D-radial spherical stationarity of
//! B_r*r^2 = const. In the required 1D-radial layout (nx2=nx3=1, multi_d=false), CT's B1
//! update is unconditionally skipped (src/mhd/mhd_ct.cpp: `if (multi_d) {...B1...}`) --
//! there is no transverse direction to curl, so a purely radial field has no mechanism to
//! evolve at all under CT specifically. This test therefore checks something adjacent but
//! still load-bearing: that NOTHING ELSE in the code path (flux divergence's magnetic
//! pressure/tension terms, the geometric source terms of Task C1/C2, floors, FOFC) reads
//! or perturbs B1 in a way that breaks div(B)=0 (i.e. B_r*Area1 = B_r*r^2 = const) even
//! while hydrodynamic quantities (density, radial velocity, pressure) genuinely evolve
//! under a radial wind -- a regression/non-interference check complementing the D2
//! topological check and the field-loop test above (see DEVELOPMENT.md Task D3 log for
//! why this is a distinct, still-meaningful check rather than a trivial no-op).

#include <cmath>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/meshblock_pack.hpp"
#include "coordinates/mesh_geometry.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"

namespace {
constexpr Real kRelTol = 1.0e-12;

//----------------------------------------------------------------------------------------
//! \fn MonopoleStationarityCheck()
//! \brief pgen_final_func: verifies Bx1f(i)*Area1(i) is still exactly the t=0 constant
//! (read back from the input file, not recomputed from a stored copy) everywhere,
//! including ghost-adjacent active faces.
void MonopoleStationarityCheck(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie, &js = indcs.js, &ks = indcs.ks;
  auto &geom = pmbp->pgeom->geom_data;
  auto &b0 = pmbp->pmhd->b0;
  Real flux0 = pin->GetOrAddReal("problem", "br_flux", 1.0);
  int nmb1 = pmbp->nmb_thispack - 1;
  int nx1 = ie - is + 2;  // number of x1 FACES

  Real max_err = 0.0;
  Kokkos::parallel_reduce("monopole_check", Kokkos::RangePolicy<>(DevExeSpace(), 0,
                           (nmb1+1)*nx1),
  KOKKOS_LAMBDA(const int &idx, Real &mx) {
    int m = idx/nx1;
    int i = (idx - m*nx1) + is;
    Real flux = geom.Area1(m,ks,js,i)*b0.x1f(m,ks,js,i);
    mx = fmax(mx, fabs(flux - flux0));
  }, Kokkos::Max<Real>(max_err));

  Real rel = max_err/std::fmax(std::fabs(flux0), 1.0e-300);
  if (rel > kRelTol) {
    std::cout << "CT Monopole Stationarity Test FAILED: max|B_r*Area1 - const| = "
              << max_err << ", relative " << rel << " (tolerance " << kRelTol << ")"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout << "CT Monopole Stationarity Test Passed (rel=" << rel << ")" << std::endl;
}
} // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::CTMonopoleStationarityTest()

void ProblemGenerator::CTMonopoleStationarityTest(ParameterInput *pin,
                                                    const bool restart) {
  if (restart) return;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "CTMonopoleStationarityTest requires a <mhd> block"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie, &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  auto &geom = pmbp->pgeom->geom_data;
  auto &u0 = pmbp->pmhd->u0;
  auto &b0 = pmbp->pmhd->b0;
  EOS_Data &eos = pmbp->pmhd->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;

  Real d0    = pin->GetOrAddReal("problem", "dens", 1.0);
  Real p0    = pin->GetOrAddReal("problem", "pgas", 1.0);
  Real v0    = pin->GetOrAddReal("problem", "vr", 0.05);
  Real flux0 = pin->GetOrAddReal("problem", "br_flux", 1.0);

  int nmb1 = pmbp->nmb_thispack - 1;
  par_for("monopole_ic", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u0(m,IDN,k,j,i) = d0;
    u0(m,IM1,k,j,i) = d0*v0;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;

    // B_r*Area1 = const (div(B)=0 for a purely radial field with no transverse part)
    b0.x1f(m,k,j,i) = flux0/geom.Area1(m,k,j,i);
    b0.x2f(m,k,j,i) = 0.0;
    b0.x3f(m,k,j,i) = 0.0;
    if (i==ie) { b0.x1f(m,k,j,i+1) = flux0/geom.Area1(m,k,j,i+1); }
    if (j==je) { b0.x2f(m,k,j+1,i) = 0.0; }
    if (k==ke) { b0.x3f(m,k+1,j,i) = 0.0; }
  });

  par_for("monopole_ie", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u0(m,IEN,k,j,i) = p0/gm1 + (0.5/u0(m,IDN,k,j,i))*SQR(u0(m,IM1,k,j,i)) +
        0.5*SQR(0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1)));
  });

  pgen_final_func = MonopoleStationarityCheck;
  return;
}
