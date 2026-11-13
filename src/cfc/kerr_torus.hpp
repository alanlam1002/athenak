#ifndef CFC_KERR_TORUS_HPP_
#define CFC_KERR_TORUS_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file kerr_torus.hpp
//! \brief K-007 W (RK-023 W1-W5), <cfc> kerr_background = true only:
//!  - KerrFMTorusInit: <problem> test_flow = fm_torus. A Fishbone-Moncrief torus about the
//!    fixed hole, over the star_enabled = false atmosphere: (r_f, th_f) -> (r_BL, th_BL) by
//!    bicubic Lagrange interpolation in F1 (kc24 a2 blmap, <problem> fm_blmap), ln h and
//!    ell from gr_torus.cpp (CalculateLFromRPeak :853-860, LogHAux :876-904), and
//!    ut^i = u^t (Omega xi^i + beta0^i) with beta0 from W1 (kerr_dev_note "FM initial data").
//!  - KerrV4Diag: <cfc> kerr_v4_dt > 0 (default 0 = off). The V4 shell diagnostics (M_D, T,
//!    J_z, sigma_eff, sigma_X, the Killing-T sources S_K, S_alpha, S_E, S_beta, alpha0
//!    Delta_X and the J_z sources), every kerr_v4_dt in time, from the history output.

#include "athena.hpp"

class Mesh;
class MeshBlockPack;
class ParameterInput;

namespace cfc {

// per-cell V4 diagnostic (KerrV4Cell): the inputs a host recomputation needs, then the
// outputs. Indices into V4Cell::v.
enum V4In {
  kV4Rho = 0, kV4Pre, kV4Ux, kV4Uy, kV4Uz,                // prim (rho, p, ut^i)
  kV4Gdd = 5,                                             // + 6: gamma_ij (adm)
  kV4Alp = 11, kV4Kx = 12,                                // alpha; + 6: K_X = adm.vK_dd
  kV4Psi0 = 18, kV4Alp0, kV4B0 = 20,                      // psi0, alpha0, + 3: beta0
  kV4Gt = 23, kV4Dgt = 29, kV4Da0 = 47, kV4Db0 = 50,      // gt(6), d gt(18), d alpha0, d beta0
  kV4Dl0 = 59, kV4K0 = 62,                                // d ln psi0(3), K0(6)
  kV4Br = 68, kV4Dda = 71, kV4Dbr = 74, kV4Dell = 83,     // b_res(3), Dx dalpha, Dx b_res, Dx ell
  kV4Xi = 86, kV4Gam = 88, kV4Ell = 89, kV4Dal = 90,      // xi^x, xi^y; Gamma; ell; delta alpha
  kV4NIn = 91
};
enum V4Out {
  kV4Vol = kV4NIn, kV4E, kV4SK, kV4SA, kV4SE, kV4SB,      // sqrt(gamma); E; S_K S_alpha S_E S_beta
  kV4Sig, kV4SigX, kV4T, kV4Jz, kV4J1, kV4J2, kV4J3,      // per unit sqrt(gamma) except Vol
  kV4N
};

void KerrFMTorusInit(ParameterInput *pin, MeshBlockPack *pmbp, bool restart);
void KerrV4Setup(ParameterInput *pin, MeshBlockPack *pmbp);
void KerrV4Diag(Mesh *pm);
void KerrTorusFinalize();

}  // namespace cfc

#endif  // CFC_KERR_TORUS_HPP_
