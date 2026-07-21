//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file scalar_field_calcrhs.cpp
//! \brief RHS for the scalar-field sector.
//!
//! Phase 0 (current): this is a no-op placeholder -- the RHS is identically zero, so
//! the evolved (sphi, Pi) state stays frozen at its initial data. Phase 1 will replace
//! this with the scalar's own Klein-Gordon-like RHS (see DEVELOPMENT_NOTES.md for the
//! target equations and references into bssn_st.f90 / arXiv:2406.05211).

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "scalar_field/scalar_field.hpp"

namespace scalarfield {
//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ScalarField::CalcRHS
//! \brief Phase 0 placeholder: zeroes the scalar-field RHS.
template <int NGHOST>
TaskStatus ScalarField::CalcRHS(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  auto &rhs = pmy_pack->pscalarfield->rhs;

  par_for("sf_rhs_noop", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    rhs.sphi(m,k,j,i) = 0.0;
    rhs.vpi(m,k,j,i)  = 0.0;
  });

  return TaskStatus::complete;
}

template TaskStatus ScalarField::CalcRHS<2>(Driver *pdriver, int stage);
template TaskStatus ScalarField::CalcRHS<3>(Driver *pdriver, int stage);
template TaskStatus ScalarField::CalcRHS<4>(Driver *pdriver, int stage);

} // namespace scalarfield
