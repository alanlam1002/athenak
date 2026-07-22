#ifndef SCALAR_FIELD_SCALAR_FIELD_HPP_
#define SCALAR_FIELD_SCALAR_FIELD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file scalar_field.hpp
//! \brief definitions for ScalarField class: the massive scalar-tensor (Damour-Esposito-
//! Farese-type) scalar-field sector, coupled to the Z4c/ADM spacetime sector.
//!
//! Through Phase 3: the scalar's own Klein-Gordon RHS, its back-reaction on the Z4c
//! equations, and coupling to the dyn_grmhd fluid (via RescaleTmunu and matter-trace
//! terms) are all implemented; the mass term and its implicit solver are still Phase 4.
//! See src/scalar_field/DEVELOPMENT_NOTES.md for the full phased rollout plan and status.

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"
#include "athena_tensor.hpp"

// forward declarations
class Driver;

namespace scalarfield {

//----------------------------------------------------------------------------------------
//! \class ScalarField

class ScalarField {
 public:
  ScalarField(MeshBlockPack *ppack, ParameterInput *pin);
  ~ScalarField();

  // Indices of evolved variables
  enum {
    I_SF_SPHI,   // canonical scalar field, \tilde\varphi (SACRA: sphi)
    I_SF_PI,     // "momentum" Pi := -n^a nabla_a sphi (SACRA: Pi)
    nscalarfield
  };
  // Names of scalar-field variables
  static char const * const ScalarField_names[nscalarfield];

  // data
  DvceArray5D<Real> u0;         // scalar-field solution
  DvceArray5D<Real> u1;         // solution at intermediate stage
  DvceArray5D<Real> u_rhs;      // rhs storage
  DvceArray5D<Real> coarse_u0;  // coarse representation of solution

  struct ScalarField_vars {
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> sphi;  // scalar field
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> vpi;   // momentum Pi
  };
  ScalarField_vars sf;
  ScalarField_vars rhs;

  struct Options {
    Real omega_c;         // Brans-Dicke-like coupling constant (SACRA omega_c/paper's B)
    Real beta0;           // DEF coupling exponent: A(sphi) = exp(0.5*beta0*sphi^2)
    Real mass2;            // scalar mass squared (SACRA pmass2); 0 = massless
    Real sphi0;            // asymptotic value of the scalar field at infinity
    Real diss;             // Kreiss-Oliger dissipation amplitude
    Real newton_tol;       // Phase 4: implicit mass-term Newton solve tolerance
    int  newton_maxiter;   // Phase 4: implicit mass-term Newton solve max iterations
  };
  Options opt;
  Real diss;   // dissipation parameter, scaled as in z4c.cpp

  // Boundary communication buffers and functions for u
  MeshBoundaryValuesCC *pbval_u;

  // functions
  void QueueScalarFieldTasks();
  TaskStatus InitRecv(Driver *d, int stage);
  TaskStatus ClearRecv(Driver *d, int stage);
  TaskStatus ClearSend(Driver *d, int stage);
  TaskStatus CopyU(Driver *d, int stage);
  TaskStatus SendU(Driver *d, int stage);
  TaskStatus RecvU(Driver *d, int stage);
  TaskStatus RestrictU(Driver *d, int stage);
  TaskStatus Prolongate(Driver *d, int stage);
  TaskStatus ApplyPhysicalBCs(Driver *d, int stage);
  TaskStatus ExpRKUpdate(Driver *d, int stage);
  TaskStatus RescaleTmunu(Driver *d, int stage);

  template <int NGHOST>
  TaskStatus CalcRHS(Driver *d, int stage);

 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this ScalarField
};

} // namespace scalarfield
#endif // SCALAR_FIELD_SCALAR_FIELD_HPP_
