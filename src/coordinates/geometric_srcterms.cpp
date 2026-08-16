//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file geometric_srcterms.cpp
//! \brief Host-side dispatch (Task C1/C2): selects, once per call (no per-cell branch),
//! the compile-time-templated kernel matching Mesh::coord_general, mirroring the
//! dispatch pattern already used by MeshGeometry's constructor and ReconDispatch.

#include <iostream>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "mesh_geometry.hpp"
#include "geometric_srcterms.hpp"

void AddCoordGeomSrcTermsHydro(MeshBlockPack *pmbp, const DvceArray5D<Real> &w0,
                                const EOS_Data &eos, const Real beta_dt,
                                DvceArray5D<Real> &u0, const DvceFaceFld5D<Real> &uflx) {
  auto &geom = pmbp->pgeom->geom_data;
  DvceArray5D<Real> no_bcc;  // never dereferenced: IsMHD=false branches are compiled out
  // Task G1: SR hydro uses the rho*h-and-u^i (not rho-and-v^i) generalization of the
  // centrifugal/pressure terms -- see AddCylindricalSrcTerms's doc comment. Mutually
  // exclusive with GR/dynamical-GR already, by construction (this function is only
  // ever called from the `else` branch of that check in hydro_tasks.cpp).
  bool is_sr = pmbp->pcoord->is_special_relativistic;
  switch (pmbp->pmesh->coord_general) {
    case CoordinateGeneral::cartesian:
      break;
    case CoordinateGeneral::cylindrical:
      if (is_sr) {
        AddCylindricalSrcTerms<false, false, true>(pmbp, w0, no_bcc, eos, beta_dt, geom,
                                                     uflx, u0);
      } else {
        AddCylindricalSrcTerms<false, false, false>(pmbp, w0, no_bcc, eos, beta_dt, geom,
                                                      uflx, u0);
      }
      break;
    case CoordinateGeneral::cylindrical_axisym:
      if (is_sr) {
        AddCylindricalSrcTerms<true, false, true>(pmbp, w0, no_bcc, eos, beta_dt, geom,
                                                    uflx, u0);
      } else {
        AddCylindricalSrcTerms<true, false, false>(pmbp, w0, no_bcc, eos, beta_dt, geom,
                                                     uflx, u0);
      }
      break;
    case CoordinateGeneral::spherical_polar:
      if (is_sr) {
        AddSphericalSrcTerms<false, true>(pmbp, w0, no_bcc, eos, beta_dt, geom, uflx, u0);
      } else {
        AddSphericalSrcTerms<false, false>(pmbp, w0, no_bcc, eos, beta_dt, geom,
                                          uflx, u0);
      }
      break;
  }
}

void AddCoordGeomSrcTermsMHD(MeshBlockPack *pmbp, const DvceArray5D<Real> &w0,
                              const DvceArray5D<Real> &bcc0, const EOS_Data &eos,
                              const Real beta_dt, DvceArray5D<Real> &u0,
                              const DvceFaceFld5D<Real> &uflx) {
  auto &geom = pmbp->pgeom->geom_data;
  // Task G1: SR+MHD geometric source terms are not implemented (would need the
  // comoving-frame field strength, not simply bcc0) -- fatal rather than silently
  // wrong, but ONLY when curvilinear is actually in play: plain SR+MHD+cartesian is a
  // pre-existing, fully-supported combination (this function is a no-op for cartesian
  // regardless -- see the switch below) and must keep working unaffected, since this
  // function is called unconditionally from the `else` branch of the GR check in
  // mhd_tasks.cpp for EVERY non-GR run, not just curvilinear ones.
  if (pmbp->pcoord->is_special_relativistic &&
      pmbp->pmesh->coord_general != CoordinateGeneral::cartesian) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "SR+MHD geometric source terms (curvilinear) are not "
              << "implemented" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  switch (pmbp->pmesh->coord_general) {
    case CoordinateGeneral::cartesian:
      break;
    case CoordinateGeneral::cylindrical:
      AddCylindricalSrcTerms<false, true, false>(pmbp, w0, bcc0, eos, beta_dt, geom,
                                                   uflx, u0);
      break;
    case CoordinateGeneral::cylindrical_axisym:
      AddCylindricalSrcTerms<true, true, false>(pmbp, w0, bcc0, eos, beta_dt, geom,
                                                  uflx, u0);
      break;
    case CoordinateGeneral::spherical_polar:
      AddSphericalSrcTerms<true, false>(pmbp, w0, bcc0, eos, beta_dt, geom, uflx, u0);
      break;
  }
}
