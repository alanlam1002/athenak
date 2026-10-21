#ifndef EOS_COLD_RECOVERY_HPP_
#define EOS_COLD_RECOVERY_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cold_recovery.hpp
//! \brief <mhd> cold_recovery (research R-083): the closed form behind the cold-gas
//! fallback of PrimitiveSolverHydro::ConsToPrim. Kept free of the solver so the host
//! unit test (tst/cold_recovery_test.cpp) checks the same code the kernel runs.
//!
//! At a fixed specific enthalpy hf (ideal gas at the floor temperature T_f:
//! hf = 1 + Gamma/(Gamma - 1) T_f/m_b, independent of the density) and B = 0, the
//! primitives follow from D and S_i alone, with no iteration:
//!   W = sqrt(1 + S^2/(D hf)^2),  S^2 = gamma^ij S_i S_j,
//!   rho = D/W,  u^i = W v^i = S^i/(D hf).
//! The caller takes P = P(rho, T_f) from the EOS and rebuilds tau = D hf W - P - D.

#include <math.h>

//! returns W; u_u gets u^i = W v^i. S_u = gamma^ij S_j, ssq = S_u^i S_i.
KOKKOS_INLINE_FUNCTION
Real ColdRecoveryW(const Real D, const Real S_u[3], const Real ssq, const Real hf,
                   Real u_u[3]) {
  const Real dh = D*hf;
  u_u[0] = S_u[0]/dh;
  u_u[1] = S_u[1]/dh;
  u_u[2] = S_u[2]/dh;
  return sqrt(1.0 + ssq/(dh*dh));
}

#endif  // EOS_COLD_RECOVERY_HPP_
