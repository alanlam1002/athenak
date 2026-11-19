#ifndef CFC_KERR_SHIBATA_HPP_
#define CFC_KERR_SHIBATA_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file kerr_shibata.hpp
//! \brief K-019 (RK-041; kerr_methods.tex sec. 5, "Option C: Shibata's split in covariant
//! form"): the Kerr vector equation DL V = s (s^i = u_p_src, the sign of kerr_methods; the
//! vector drivers solve -DL V = -u_p_src, cfc_kerr.cpp:427) as four scalar Delta_s solves,
//!     V_j = P_j - (1/8) d_j Phi,  Phi = eta + x^k P_k,  V^i = gt^ij V_j,
//!     Delta_s P_k = s_k - Q_k[P] + (1/4) Rt_k^l d_l Phi,
//!     Q_k = -2 d_i(gt^ij Gt^l_jk P_l) + gt^ab (d_k Gt^l_ab) P_l,
//!     Delta_s eta = -x^k s_k - 2 d_i(x^k gt^ij Gt^l_jk P_l) - (d_j gt^jl) P_l
//!                   + x^k gt^ab (d_k Gt^l_ab) P_l - (1/4) x^k Rt_k^l d_l Phi,
//! s_k = gt_kl s^l, Delta_s f = d_i(gt^ij d_j f) (det gt = 1), x^k = x - xbh (any constant
//! origin is equivalent: eta -> eta - c^k P_k). For gt = delta: Delta P_k = s_k,
//! Delta eta = -x^k s_k, the flat path's Shibata system.
//!
//! Discretisation: the operator is MGKerrScalarDriver's (kind linear, c = s = 0: -Delta_s u
//! = src, 19-point flux, Rule R, robin outer boundary), one instance shared by the X and
//! beta solvers (KerrShibCommon), Prepare()d only when the background key changes. The
//! couplings are lagged (outer Picard passes): the P_k sources use the previous pass
//! (Jacobi), eta's uses this pass's P (Gauss-Seidel) and the previous eta. Face fluxes
//! F_d = C_d[k][l] (P_l(c) + P_l(c - e_d))/2 with C_d[k][l] = gt^dj Gt^l_jk evaluated
//! analytically at the lower d-face; the trace, Ricci and d gt^jl terms at the cell centre
//! (analytic, PointMetric2); d Phi by centred differences. The ghosts these stencils read
//! are the MG finest level's one-cell ring (RetrieveResult). Passes stop when
//! max |V_new - V_old| <= shib_tol (default: the block's mg_threshold) or at
//! shib_max_pass (default 10).
//!
//! <block> keys (block = cfc_kerr_x or cfc_kerr_beta; read only when vec_solver = shibata):
//!   vec_solver     direct (default) | shibata
//!   shib_tol       Picard stop on max |dV| per pass (default mg_threshold)
//!   shib_max_pass  (default 10)
//!   shib_log       0 none | 1 one KSHIB line per solve (default) | 2 one per pass

#include <string>

#include "athena.hpp"
#include "kerr_mg_ops.hpp"
#include "mg_kerr_scalar.hpp"

class MeshBlockPack;
class ParameterInput;
class Driver;

namespace cfc {
namespace kshib {
// geometry channels (KerrShibCommon::geo), every cell incl. ghosts
inline constexpr int kCF = 0;    // + 9 d + 3 k + l: gt^dj Gt^l_jk at the lower d-face
inline constexpr int kT = 27;    // + 3 k + l: gt^ab d_k Gt^l_ab (centre)
inline constexpr int kR = 36;    // + 3 k + l: Rt_k^l = Rt_kj gt^jl (centre)
inline constexpr int kDG = 45;   // + l: d_j gt^jl (centre)
inline constexpr int kGd = 48;   // + q: gt_kl, packed xx xy xz yy yz zz (centre)
inline constexpr int kGu = 54;   // + q: gt^kl (centre)
inline constexpr int kN = 60;
}  // namespace kshib

//! the scalar driver with access to the protected solve entry points
class KerrShibScalar : public MGKerrScalarDriver {
 public:
  using MGKerrScalarDriver::MGKerrScalarDriver;
  //! the finest u and src are already loaded: (zero the coarse hierarchy,) SolveMG
  void SolveLoaded(Driver *pdriver) {
    if (zero_coarse_) ZeroHierarchy();
    SolveMG(pdriver);
  }
  Real Eps() const { return eps_; }
};

//! \class KerrShibCommon
//! \brief what the X and beta Option C solvers share: the scalar driver and the geometry
class KerrShibCommon {
 public:
  KerrShibCommon(MeshBlockPack *pmbp, ParameterInput *pin, const std::string &block,
                 const kmg::CoefSource &cs);
  ~KerrShibCommon();
  //! Prepare the scalar driver and fill geo for cs (nothing while the key still matches)
  void Update(const kmg::CoefSource &cs);
  MeshBlockPack *pmbp;
  KerrShibScalar *sc = nullptr;
  DvceArray5D<Real> geo;
  kmg::CoefSource cs_now;
  long fills = 0, reuses = 0;
 private:
  kmg::CoeffKey key_;
};

//! \class KerrShibataSolver
//! \brief one vector equation (X or beta_res) by Option C
class KerrShibataSolver {
 public:
  KerrShibataSolver(MeshBlockPack *pmbp, ParameterInput *pin, const std::string &block,
                    KerrShibCommon *com, const std::string &name);
  //! DL V = s with s^i = src(m, 0..2) (ghost depth src_ngh, interior read); V^i into
  //! dst(m, 0..2) interior (mesh ghost depth)
  void Solve(Driver *pdriver, int stage, const kmg::CoefSource &cs,
             const DvceArray5D<Real> &src, int src_ngh, DvceArray5D<Real> &dst);
  //! restart_state: P_k, eta (4 variables, mesh shape)
  int NumVars() const { return 4; }
  void SaveState(DvceArray5D<Real> &dst, int ns, int nmb) const;
  void LoadState(const DvceArray5D<Real> &src, int ns, int nmb);
  //! the last solve: passes, final max |dV|
  int last_pass_ = 0;
  Real last_dv_ = 0.0;

 private:
  void EnsureState();
  Real ComputeV(const DvceArray5D<Real> &v);   // V^i from pe_ into v (interior)
  MeshBlockPack *pmbp_;
  KerrShibCommon *com_;
  std::string name_;
  Real tol_ = 0.0;
  int max_pass_ = 10;
  int log_ = 1;
  int pe_seq_ = -1, pe_nmb_ = -1;
  DvceArray5D<Real> pe_;    // P_0..2, eta (persistent warm start)
  DvceArray5D<Real> pnew_;  // this pass's P (Jacobi)
  DvceArray5D<Real> sk_;    // s_k = gt_kl s^l
  DvceArray5D<Real> rhs_;   // the loaded src = -rhs
  DvceArray5D<Real> vold_, vnew_;
};

}  // namespace cfc

#endif  // CFC_KERR_SHIBATA_HPP_
