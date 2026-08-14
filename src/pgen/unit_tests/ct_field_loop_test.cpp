//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ct_field_loop_test.cpp
//! \brief Unit test for Task D3: the REAL correctness gate for the area/edge-length-
//! weighted CT curl (Task D1) -- a physical induction test, not merely the topological
//! div(B)=0 identity of Task D2 (see v2 plan Correction C6: a CT implementation with a
//! wrong edge length, wrong handedness, or a flipped EMF sign can still satisfy D2's
//! check to roundoff, since that check is a telescoping-sum identity true for ANY E/A/L
//! values).
//!
//! Classic field-loop advection (Gardiner & Stone 2005 etc.), adapted to
//! cylindrical_axisym (R,z) -- the required 2D curvilinear layout. A circular loop of
//! poloidal field (B_R, B_z, built as the curl of a "tent" vector potential A_phi) is
//! advected by a UNIFORM z-velocity on a domain periodic in z (z-translation is an exact
//! symmetry of the axisym geometry, unlike R-translation, which is not -- Area1/2/3 and
//! Len1/2/3 all depend on R but not z, so advecting purely in z is the curvilinear
//! analogue of the flat-Cartesian test's diagonal advection). After exactly one z-period,
//! the field must return to its initial shape to TRUNCATION error (not roundoff -- finite
//! advection with a real reconstruction scheme has genuine numerical diffusion) -- a
//! coordinate SIGN or HANDEDNESS error, or a wrong edge length in the CT curl, would
//! either destroy the loop's circular shape, advect it at the wrong rate, or grow/damp
//! its amplitude in a way this test WOULD catch (unlike D2's topological check).
//!
//! Uses the same reused-`pgen_final_func`-plus-`OutputErrors()` mechanism as
//! geom_equilibrium_test.cpp: since advection is by an exact integer number of periods,
//! the reference solution at t_final is IDENTICAL to the t=0 IC (same helper function,
//! called a second time to fill the u1/b1 registers instead of u0/b0).

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
//! \brief A_phi(R,z) = amp*(loop_radius - dist) for dist < loop_radius, else 0, with
//! dist = sqrt((R-R0)^2+(z-z0)^2) -- the classic compactly-supported "tent" potential.
KOKKOS_INLINE_FUNCTION
Real ALoop(Real R, Real z, Real amp, Real loop_radius, Real R0, Real z0) {
  Real dist = std::sqrt(SQR(R-R0) + SQR(z-z0));
  return (dist < loop_radius) ? amp*(loop_radius - dist) : 0.0;
}

//----------------------------------------------------------------------------------------
//! \fn FillFieldLoop()
//! \brief writes the analytic field-loop state (uniform density/pressure, uniform
//! z-velocity, poloidal B from curl(A_phi)) into whichever (u,b) register pair is passed.
void FillFieldLoop(MeshBlockPack *pmbp, ParameterInput *pin,
                    DvceArray5D<Real> &u_target, DvceFaceFld4D<Real> &b_target) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie, &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int nmb1 = pmbp->nmb_thispack - 1;
  auto &geom = pmbp->pgeom->geom_data;

  Real d0    = pin->GetOrAddReal("problem", "dens", 1.0);
  Real p0    = pin->GetOrAddReal("problem", "pgas", 1.0);
  Real w0    = pin->GetOrAddReal("problem", "vz", 0.5);
  Real amp   = pin->GetOrAddReal("problem", "amp", 1.0e-3);
  Real rad   = pin->GetOrAddReal("problem", "loop_radius", 0.3);
  Real R0    = pin->GetOrAddReal("problem", "R0", 1.5);
  Real z0    = pin->GetOrAddReal("problem", "z0", 0.0);
  Real gm1 = pmbp->pmhd->peos->eos_data.gamma - 1.0;

  auto bx1f = b_target.x1f;
  auto bx2f = b_target.x2f;
  auto bx3f = b_target.x3f;

  par_for("field_loop_ic", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u_target(m,IDN,k,j,i) = d0;
    u_target(m,IM1,k,j,i) = 0.0;
    u_target(m,IM2,k,j,i) = d0*w0;
    u_target(m,IM3,k,j,i) = 0.0;

    Real xf_i = geom.xf1(m,i), xf_ip1 = geom.xf1(m,i+1);
    Real xf_j = geom.xf2(m,j), xf_jp1 = geom.xf2(m,j+1);
    Real a_ij   = ALoop(xf_i,   xf_j,   amp, rad, R0, z0);
    Real a_ip1j = ALoop(xf_ip1, xf_j,   amp, rad, R0, z0);
    Real a_ijp1 = ALoop(xf_i,   xf_jp1, amp, rad, R0, z0);

    bx1f(m,k,j,i) = (a_ijp1*geom.Len3(m,k,j+1,i) - a_ij*geom.Len3(m,k,j,i))
                    / geom.Area1(m,k,j,i);
    bx2f(m,k,j,i) = -(a_ip1j*geom.Len3(m,k,j,i+1) - a_ij*geom.Len3(m,k,j,i))
                    / geom.Area2(m,k,j,i);
    bx3f(m,k,j,i) = 0.0;

    if (i==ie) {
      Real a_ip1jp1 = ALoop(xf_ip1, xf_jp1, amp, rad, R0, z0);
      bx1f(m,k,j,i+1) = (a_ip1jp1*geom.Len3(m,k,j+1,i+1) - a_ip1j*geom.Len3(m,k,j,i+1))
                        / geom.Area1(m,k,j,i+1);
    }
    if (j==je) {
      Real a_ip1jp1 = ALoop(xf_ip1, xf_jp1, amp, rad, R0, z0);
      bx2f(m,k,j+1,i) = -(a_ip1jp1*geom.Len3(m,k,j+1,i+1) - a_ijp1*geom.Len3(m,k,j+1,i))
                        / geom.Area2(m,k,j+1,i);
    }
    if (k==ke) {
      bx3f(m,k+1,j,i) = 0.0;
    }
  });

  par_for("field_loop_ie", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u_target(m,IEN,k,j,i) = p0/gm1 + (0.5/u_target(m,IDN,k,j,i))*
        (SQR(u_target(m,IM1,k,j,i)) + SQR(u_target(m,IM2,k,j,i))
         + SQR(u_target(m,IM3,k,j,i))) +
        0.5*(SQR(0.5*(bx1f(m,k,j,i) + bx1f(m,k,j,i+1))) +
             SQR(0.5*(bx2f(m,k,j,i) + bx2f(m,k,j+1,i))) +
             SQR(0.5*(bx3f(m,k,j,i) + bx3f(m,k+1,j,i))));
  });
}

//----------------------------------------------------------------------------------------
//! \fn FieldLoopErrors()
//! \brief pgen_final_func: since advection is by an exact integer number of z-periods,
//! the reference solution equals the t=0 IC exactly -- fill u1/b1 with it and diff.
void FieldLoopErrors(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  FillFieldLoop(pmbp, pin, pmbp->pmhd->u1, pmbp->pmhd->b1);
  pm->pgen->OutputErrors(pin, pm);
}
} // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::CTFieldLoopTest()

void ProblemGenerator::CTFieldLoopTest(ParameterInput *pin, const bool restart) {
  if (restart) return;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "CTFieldLoopTest requires a <mhd> block" << std::endl;
    exit(EXIT_FAILURE);
  }
  FillFieldLoop(pmbp, pin, pmbp->pmhd->u0, pmbp->pmhd->b0);
  pgen_final_func = FieldLoopErrors;
  return;
}
