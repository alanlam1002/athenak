//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ct_divb_test.cpp
//! \brief Unit test for Task D2: div(B)=0 preservation under the area/edge-length-
//! weighted CT curl (src/mhd/mhd_ct.cpp, Task D1), for cartesian (control),
//! cylindrical, cylindrical_axisym, and spherical_polar.
//!
//! This is a TOPOLOGICAL check, not a physical one (see v2 plan Correction C6): B is
//! initialized as the discrete curl of an edge-centered vector potential A3, using the
//! SAME geom.Area1/2/3 and geom.Len1/2/3 tables that CT itself reads (mirroring
//! src/pgen/tests/orszag_tang.cpp's flat curl-of-A construction, generalized to
//! curvilinear via the Stokes-form B = curl_stokes(A)/Area identity). A uniform x1
//! velocity is added so CT's curl terms actually fire during the run (not just at
//! t=0 -- a pure "is the IC divergence-free" check would be a strictly weaker test,
//! already implied by B4's geometry-construction tests). div(B)=0 for a field built
//! this way, and evolved by a CT scheme that reads the identical Area/Len tables for
//! both the update AND the check below, is an EXACT discrete identity (each edge is
//! shared by exactly two faces with opposite circulation orientation, so the telescoping
//! sum is zero for ANY E-field/edge-length/area values) -- so it holds to roundoff
//! regardless of whether the CT curl is *physically* correct (a CT implementation with
//! wrong edge lengths, wrong handedness, or a flipped EMF sign could still pass this to
//! roundoff, which is exactly why Task D3's physical induction tests exist as the real
//! correctness gate; this test's job is only to catch an INCONSISTENCY between the
//! Area/Len tables used by the update vs. the divergence check, e.g. a copy-paste index
//! bug).

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
constexpr Real kRelTol = 1.0e-10;

KOKKOS_INLINE_FUNCTION
Real A3Fn(Real x1, Real x2, Real amp,
          Real x1min, Real x1max, Real x2min, Real x2max) {
  Real u = (x1 - x1min)/(x1max - x1min);
  Real v = (x2 - x2min)/(x2max - x2min);
  return amp*std::sin(2.0*M_PI*u)*std::sin(2.0*M_PI*v);
}

//----------------------------------------------------------------------------------------
//! \fn CTDivBCheck()
//! \brief pgen_final_func: computes max|div(B)*Vol| over the active domain, normalized
//! by a typical face-flux scale, and fails if it exceeds roundoff.
void CTDivBCheck(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie, &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  auto &geom = pmbp->pgeom->geom_data;
  auto &b0 = pmbp->pmhd->b0;
  int nmb = pmbp->nmb_thispack;
  int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
  int nkji = nx3*nx2*nx1, nji = nx2*nx1;
  int nmkji = nmb*nkji;

  Real max_divb = 0.0, max_scale = 0.0;
  Kokkos::parallel_reduce("ct_divb_check", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
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

  Real rel = max_divb/std::fmax(max_scale, 1.0e-300);
  if (rel > kRelTol) {
    std::cout << "CT div(B) Test FAILED: max|div(B)*Vol| = " << max_divb
              << ", relative to face-flux scale " << max_scale
              << " -> " << rel << " (tolerance " << kRelTol << ")" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout << "CT div(B) Test Passed (rel=" << rel << ")" << std::endl;
}
} // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::CTDivBTest()

void ProblemGenerator::CTDivBTest(ParameterInput *pin, const bool restart) {
  if (restart) return;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "CTDivBTest requires a <mhd> block" << std::endl;
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

  Real amp  = pin->GetOrAddReal("problem", "amp", 0.3);
  Real d0   = pin->GetOrAddReal("problem", "dens", 1.0);
  Real p0   = pin->GetOrAddReal("problem", "pgas", 1.0);
  Real v0   = pin->GetOrAddReal("problem", "v0", 0.1);
  Real x1min = pmy_mesh_->mesh_size.x1min, x1max = pmy_mesh_->mesh_size.x1max;
  Real x2min = pmy_mesh_->mesh_size.x2min, x2max = pmy_mesh_->mesh_size.x2max;

  int nmb1 = pmbp->nmb_thispack - 1;
  par_for("ct_divb_ic", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u0(m,IDN,k,j,i) = d0;
    u0(m,IM1,k,j,i) = d0*v0;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;

    Real xf_i = geom.xf1(m,i), xf_ip1 = geom.xf1(m,i+1);
    Real xf_j = geom.xf2(m,j), xf_jp1 = geom.xf2(m,j+1);
    Real a_ij   = A3Fn(xf_i,   xf_j,   amp, x1min,x1max,x2min,x2max);
    Real a_ip1j = A3Fn(xf_ip1, xf_j,   amp, x1min,x1max,x2min,x2max);
    Real a_ijp1 = A3Fn(xf_i,   xf_jp1, amp, x1min,x1max,x2min,x2max);

    b0.x1f(m,k,j,i) = (a_ijp1*geom.Len3(m,k,j+1,i) - a_ij*geom.Len3(m,k,j,i))
                      / geom.Area1(m,k,j,i);
    b0.x2f(m,k,j,i) = -(a_ip1j*geom.Len3(m,k,j,i+1) - a_ij*geom.Len3(m,k,j,i))
                      / geom.Area2(m,k,j,i);
    b0.x3f(m,k,j,i) = 0.0;

    if (i==ie) {
      Real a_ip1jp1 = A3Fn(xf_ip1, xf_jp1, amp, x1min,x1max,x2min,x2max);
      b0.x1f(m,k,j,i+1) = (a_ip1jp1*geom.Len3(m,k,j+1,i+1) - a_ip1j*geom.Len3(m,k,j,i+1))
                          / geom.Area1(m,k,j,i+1);
    }
    if (j==je) {
      Real a_ip1jp1 = A3Fn(xf_ip1, xf_jp1, amp, x1min,x1max,x2min,x2max);
      b0.x2f(m,k,j+1,i) = -(a_ip1jp1*geom.Len3(m,k,j+1,i+1) - a_ijp1*geom.Len3(m,k,j+1,i))
                          / geom.Area2(m,k,j+1,i);
    }
    if (k==ke) {
      b0.x3f(m,k+1,j,i) = 0.0;
    }
  });

  par_for("ct_divb_ie", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u0(m,IEN,k,j,i) = p0/gm1 + (0.5/u0(m,IDN,k,j,i))*
        (SQR(u0(m,IM1,k,j,i)) + SQR(u0(m,IM2,k,j,i)) + SQR(u0(m,IM3,k,j,i))) +
        0.5*(SQR(0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1))) +
             SQR(0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i))) +
             SQR(0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i))));
  });

  pgen_final_func = CTDivBCheck;
  return;
}
