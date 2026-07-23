//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_hohlraum.cpp
//  \brief 1D/2D Hohlraum tests for radiation

// C++ headers

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "radiation/radiation.hpp"
#include "pgen/pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::UserProblem(ParameterInput *pin)
//  \brief Sets initial conditions for GR radiation hohlraum test

// [Stage 11] Was previously named ProblemGenerator::Hohlraum, which is not the
// entry point CallProblemGenerator() looks for under either dispatch path (not
// in the runtime pgen_name table in pgen.cpp, and not the required UserProblem
// name for the -D PROBLEM=rad_hohlraum dedicated-build path used by every other
// file living directly in pgen/ -- e.g. rad_relax.cpp, rad_diffusion.cpp). This
// meant the file compiled but was never actually reachable from any input file
// -- true dead code. Renaming to UserProblem is the fix; see
// build_hohlraum_sakura.sh for the dedicated build this now enables.
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  // capture variables for kernel
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int nmb1 = (pmbp->nmb_thispack-1);
  int nang1 = (pmbp->prad->prgeo->nangles-1);

  // set inflow state in BoundaryValues, sync to device
  auto &i_in = pmbp->prad->pbval_i->i_in;
  for (int n=0; n<=nang1; ++n) {
    i_in.h_view(n,BoundaryFace::inner_x1) = (-1.0/(4.0*M_PI));
  }
  if (n2 > 1) {
    for (int n=0; n<=nang1; ++n) {
      i_in.h_view(n,BoundaryFace::inner_x2) = (-1.0/(4.0*M_PI));
    }
  }
  i_in.template modify<HostMemSpace>();
  i_in.template sync<DevExeSpace>();

  // return if restart
  if (restart) return;

  auto &i0 = pmbp->prad->i0;
  par_for("rad_hohlraum",DevExeSpace(),0,nmb1,0,nang1,0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    i0(m,n,k,j,i) = 0.0;
  });

  return;
}
