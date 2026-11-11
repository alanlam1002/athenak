#ifndef DYN_GRMHD_KERR_SOURCES_HPP_
#define DYN_GRMHD_KERR_SOURCES_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file kerr_sources.hpp
//! \brief <cfc> kerr_background (K-004 B3): the per-cell parts of the Valencia sources on
//! gamma = psi^4 gt (kerr_methods eq:smom, eq:sen), for AddCoordTermsKerr and the host
//! pairing test. The background's derivatives come from W1 (cfc/kerr_background.hpp:67-82),
//! the residuals' from Dx<NGHOST>; nothing here differences anything. Needs Real and
//! KOKKOS_INLINE_FUNCTION only (the host test defines both).

namespace kerrsrc {

//! symmetric pair (a, b) -> xx xy xz yy yz zz
KOKKOS_INLINE_FUNCTION int Sym(const int a, const int b) {
  const int lo = (a < b) ? a : b, hi = (a < b) ? b : a;
  return (lo == 0) ? hi : ((lo == 1) ? 2 + hi : 5);
}

//! d_k alpha = d_k alpha0 + d_k delta alpha, d_k beta'^i = d_k beta0^i + d_k b_res^i (d xidot
//! = 0), d_k gamma_ij = 4 gamma_ij (d_k ln psi0 + d_k ln(psi/psi0)) + psi^4 d_k gt_ij.
//! db0[3k + i] = d_k beta0^i, dgt[6k + ij] = d_k gt_ij (kbg_'s layouts); dbr[k][i] = d_k b_res^i
KOKKOS_INLINE_FUNCTION void ComposeDerivs(const Real g3d[6], const Real p4, const Real dgt[18],
                                          const Real da0[3], const Real dda[3],
                                          const Real db0[9], const Real dbr[3][3],
                                          const Real dl0[3], const Real dell[3],
                                          Real dalpha[3], Real dbeta[3][3],
                                          Real dg[3][3][3]) {
  for (int k = 0; k < 3; ++k) {
    dalpha[k] = da0[k] + dda[k];
    const Real dl = dl0[k] + dell[k];
    for (int i = 0; i < 3; ++i) { dbeta[k][i] = db0[3*k + i] + dbr[k][i]; }
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        const int q = Sym(a, b);
        dg[k][a][b] = 4.0*g3d[q]*dl + p4*dgt[6*k + q];
      }
    }
  }
}

//! alpha S^ij Keff_ij of eq:kcfcbg (trace dropped):
//!   1/2 psi^4 S^ij (Lt b_res)_ij + alpha0 (psi/psi0)^4 S^ij K0_ij,
//!   (Lt b)_ij = gt_jq d_i b^q + gt_iq d_j b^q - 2/3 gt_ij d_k b^k + b^k d_k gt_ij  (eq:E2, lowered)
//! p4r = (psi/psi0)^4
KOKKOS_INLINE_FUNCTION Real EnergyK(const Real p4, const Real p4r, const Real alpha0,
                                    const Real gt[6], const Real dgt[18], const Real br[3],
                                    const Real dbr[3][3], const Real k0[6],
                                    const Real S_uu[3][3]) {
  const Real dv = dbr[0][0] + dbr[1][1] + dbr[2][2];
  Real sl = 0.0, sk = 0.0;
  for (int a = 0; a < 3; ++a) {
    for (int b = a; b < 3; ++b) {
      const int q = Sym(a, b);
      const Real w = (a == b) ? 1.0 : 2.0;
      Real l = -(2.0/3.0)*gt[q]*dv;
      for (int c = 0; c < 3; ++c) {
        l += gt[Sym(b, c)]*dbr[a][c] + gt[Sym(a, c)]*dbr[b][c] + br[c]*dgt[6*c + q];
      }
      sl += w*S_uu[a][b]*l;
      sk += w*S_uu[a][b]*k0[q];
    }
  }
  return 0.5*p4*sl + alpha0*p4r*sk;
}

//! s_k = -E d_k alpha + S_j d_k beta'^j + 1/2 alpha S^ij d_k gamma_ij (AddCoordTermsEOS's form)
KOKKOS_INLINE_FUNCTION void Momentum(const Real alpha, const Real E, const Real S_d[3],
                                     const Real S_uu[3][3], const Real dalpha[3],
                                     const Real dbeta[3][3], const Real dg[3][3][3],
                                     Real s_k[3]) {
  for (int a = 0; a < 3; ++a) {
    Real s = 0.0;
    for (int b = 0; b < 3; ++b) {
      for (int c = 0; c < 3; ++c) { s += 0.5*alpha*S_uu[b][c]*dg[a][b][c]; }
      s += S_d[b]*dbeta[a][b];
    }
    s_k[a] = s - E*dalpha[a];
  }
}

//! -S^j d_j alpha, S^j = gamma^jb S_b (g3u packed)
KOKKOS_INLINE_FUNCTION Real EnergyAlpha(const Real g3u[6], const Real S_d[3],
                                        const Real dalpha[3]) {
  Real s = 0.0;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) { s -= g3u[Sym(a, b)]*S_d[a]*dalpha[b]; }
  }
  return s;
}

}  // namespace kerrsrc

#endif  // DYN_GRMHD_KERR_SOURCES_HPP_
