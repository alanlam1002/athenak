#ifndef Z4C_Z4C_ALGCONSTR_HPP_
#define Z4C_Z4C_ALGCONSTR_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_algconstr.hpp
//! \brief pointwise algebraic-constraint projection (det g-tilde = 1, tr A-tilde = 0),
//! shared by Z4c::AlgConstr (on u0) and the Cartoon fill (on the u_cart ghost planes).

#include <cmath>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "z4c/z4c.hpp"

namespace z4c {

KOKKOS_INLINE_FUNCTION
void AlgConstrPoint(const Z4c::Z4c_vars &z4c, const int m, const int k, const int j,
                    const int i) {
  Real detg = adm::SpatialDet(z4c.g_dd(m,0,0,k,j,i), z4c.g_dd(m,0,1,k,j,i),
                            z4c.g_dd(m,0,2,k,j,i),z4c.g_dd(m,1,1,k,j,i),
                            z4c.g_dd(m,1,2,k,j,i), z4c.g_dd(m,2,2,k,j,i));
  detg = detg > 0. ? detg : 1.;
  // Real eps = detg - 1.;
  // Real oopsi4 = (eps < opt.eps_floor) ? (1. - opt.eps_floor/3.) :
  //             (std::pow(1./detg, 1./3.));
  Real oopsi4 = std::cbrt(1./detg);

  for(int a = 0; a < 3; ++a)
  for(int b = a; b < 3; ++b) {
    z4c.g_dd(m,a,b,k,j,i) *= oopsi4;
  }

  // compute trace of A
  // note: here we are assuming that det g = 1, which we enforced above
  Real A = adm::Trace(1.0,
            z4c.g_dd(m,0,0,k,j,i), z4c.g_dd(m,0,1,k,j,i), z4c.g_dd(m,0,2,k,j,i),
            z4c.g_dd(m,1,1,k,j,i), z4c.g_dd(m,1,2,k,j,i), z4c.g_dd(m,2,2,k,j,i),
            z4c.vA_dd(m,0,0,k,j,i), z4c.vA_dd(m,0,1,k,j,i), z4c.vA_dd(m,0,2,k,j,i),
            z4c.vA_dd(m,1,1,k,j,i), z4c.vA_dd(m,1,2,k,j,i), z4c.vA_dd(m,2,2,k,j,i));

  // enforce trace of A to be zero
  for(int a = 0; a < 3; ++a)
  for(int b = a; b < 3; ++b) {
    z4c.vA_dd(m,a,b,k,j,i) -= (1.0/3.0) * A * z4c.g_dd(m,a,b,k,j,i);
  }
}

} // namespace z4c
#endif // Z4C_Z4C_ALGCONSTR_HPP_
