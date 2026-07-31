#ifndef CFC_CFC_PUNCTURE_HPP_
#define CFC_CFC_PUNCTURE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cfc_puncture.hpp
//! \brief Stationary, non-spinning, maximal-slicing BH "trumpet" background for the
//! CFC + BH-puncture TDE extension (see CFC_PUNCTURE_TDE_PLAN.md, Sec 3.2/3.6). Ported
//! from ~/SACRA_2D/SACRA_MPI/mod_bh.f90 (tbh_get_var/tbh_areal_to_iso/tbh_iso_to_areal),
//! the vetted source for the Baumgarte & Naculich 2007 (arXiv:0709.0299) maximal trumpet
//! solution. The puncture is fixed at the coordinate origin and time-independent, so
//! every function here is a pure, stateless function of position -- none of them touch
//! mesh/task-graph state; the per-cell fill this feeds is a separate, later piece
//! (CFC_PUNCTURE_TDE_PLAN.md Sec 5 Phase A item 1).
//!
//! Units: TrumpetArealToIso/TrumpetIsoToAreal work in M_BH=1-normalized areal radius
//! (rho = R_sch/M_BH), mirroring mod_bh.f90's own convention -- tbh_get_var itself is
//! generic in the physical mass m_bh, but tbh_areal_to_iso/tbh_iso_to_areal are not, so
//! callers rescale by m_bh on the way in/out (see TrumpetBackground's own doc comment).

#include <cmath>

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "eos/primitive-solver/numtools_root.hpp"

// forward declarations
class MeshBlockPack;

namespace cfc {

// Areal radius of the trumpet throat, in M_BH=1 units (mod_bh.f90:79,80,83,108 et al.).
inline constexpr Real kTrumpetThroat = 1.5;

//----------------------------------------------------------------------------------------
//! \fn Real TrumpetArealToIso(Real rho)
//! \brief Closed-form forward map areal->isotropic radius chi(rho), M_BH=1 units
//! (mod_bh.f90:56-65). Valid for rho >= kTrumpetThroat.
KOKKOS_INLINE_FUNCTION
Real TrumpetArealToIso(Real rho) {
  Real s1 = std::sqrt(4.0*rho*rho + 4.0*rho + 3.0);
  Real s2 = std::sqrt(8.0*rho*rho + 8.0*rho + 6.0);
  Real numer = (4.0 + 3.0*std::sqrt(2.0)) * (2.0*rho - 3.0);
  Real denom = 8.0*rho + 6.0 + 3.0*s2;
  return 0.25*(2.0*rho + 1.0 + s1) * std::pow(numer/denom, 1.0/std::sqrt(2.0));
}

namespace impl {

// Residual functor for TrumpetIsoToAreal's bracketed root-find: chi(rho)/chi_cell - 1.
struct TrumpetArealResidual {
  KOKKOS_INLINE_FUNCTION
  Real operator()(Real rho, Real chi_cell) const {
    return TrumpetArealToIso(rho)/chi_cell - 1.0;
  }
};

}  // namespace impl

//----------------------------------------------------------------------------------------
//! \fn Real TrumpetIsoToAreal(Real chi_cell)
//! \brief Bracketed inverse of TrumpetArealToIso (r_iso -> R_sch, M_BH=1 units),
//! CFC_PUNCTURE_TDE_PLAN.md Sec 3.6 / mod_bh.f90:67-122. Built on NumTools::Root::
//! FalsePosition (an Anderson-Bjorck/Illinois false-position variant, already used the
//! same way in eos/primitive-solver/ideal_c2p_hyd.hpp / ideal_c2p_mhd.hpp) rather than
//! hand-porting mod_bh.f90's own rootfinding_illinois loop -- same algorithm family,
//! reuses vetted in-repo code instead of a parallel implementation.
KOKKOS_INLINE_FUNCTION
Real TrumpetIsoToAreal(Real chi_cell) {
  constexpr Real tol = 1.0e-15;
  // mod_bh.f90:83-84: an exactly-zero isotropic radius maps to the throat; special-case
  // it directly rather than relying on FalsePosition's own near-degenerate-bracket
  // handling right at this single point.
  if (chi_cell <= kTrumpetThroat*tol) {
    return kTrumpetThroat;
  }
  Real lb = std::fmax(kTrumpetThroat, chi_cell);
  Real ub = std::fmax(kTrumpetThroat, chi_cell + 1.5);
  Real rho;
  NumTools::Root root;
  root.iterations = 100;  // ample margin over FalsePosition's superlinear convergence
  impl::TrumpetArealResidual residual;
  root.FalsePosition(residual, lb, ub, rho, tol, chi_cell);
  // mod_bh.f90:108: clamp guards throat-adjacent roundoff undershoot.
  return std::fmax(rho, kTrumpetThroat);
}

//----------------------------------------------------------------------------------------
//! \fn void TrumpetBackground(...)
//! \brief Stationary maximal-slicing trumpet psi0/alpha0/beta0^i/Ahat0_ij/Ahat0^2 at
//! Cartesian position x=(x1,x2,x3), given BH mass m_bh and the areal radius r_sch
//! (= m_bh*TrumpetIsoToAreal(|x|/m_bh), left to the caller to supply -- kept separate
//! since CFC_PUNCTURE_TDE_PLAN.md Sec 3.8 requires this be callable at arbitrary
//! multigrid-level coordinates with r_sch precomputed once per level, not recomputed via
//! a root-find on every call). Port of mod_bh.f90:11-54 (tbh_get_var); Aij0 is packed
//! (xx,yy,zz,xy,xz,yz), matching that Fortran's own component order. dpsi0/dalpha0
//! (radial derivatives) are optional out-params (nullptr to skip) -- used by
//! FillPunctureBackground below to assemble grad_ap6_0 = D_j(alpha0*psi0^-6) (Sec 3.9's
//! shift-source ingredient, Sec 5 Phase A item 7).
KOKKOS_INLINE_FUNCTION
void TrumpetBackground(Real m_bh, Real x1, Real x2, Real x3, Real r_sch,
                       Real *psi0, Real *alpha0, Real beta0[3], Real Aij0[6], Real *a2,
                       Real *dpsi0 = nullptr, Real *dalpha0 = nullptr) {
  Real r = std::sqrt(x1*x1 + x2*x2 + x3*x3);
  Real rrs = r_sch/m_bh;
  Real rrs3 = rrs*rrs*rrs;
  Real psi = std::sqrt(r_sch/r);
  Real alpha = std::sqrt(1.0 - 2.0/rrs + 1.6875/(rrs3*rrs));
  *psi0 = psi;
  *alpha0 = alpha;
  if (dpsi0 != nullptr) {
    *dpsi0 = -psi*(1.0 - 0.84375/rrs3) / (rrs*r*(1.0 + alpha));
  }
  if (dalpha0 != nullptr) {
    *dalpha0 = (1.0 - 3.375/rrs3) / (rrs*r);
  }

  const Real fac = 0.75*std::sqrt(3.0);
  Real bmag = fac / (r_sch*rrs*rrs);
  beta0[0] = bmag*x1;
  beta0[1] = bmag*x2;
  beta0[2] = bmag*x3;

  Real r2 = r*r;
  Real aij_amp = fac*m_bh*m_bh / (r_sch*r_sch*r_sch);
  Aij0[0] = aij_amp*(1.0 - 3.0*x1*x1/r2);
  Aij0[1] = aij_amp*(1.0 - 3.0*x2*x2/r2);
  Aij0[2] = aij_amp*(1.0 - 3.0*x3*x3/r2);
  Aij0[3] = aij_amp*(-3.0*x1*x2/r2);
  Aij0[4] = aij_amp*(-3.0*x1*x3/r2);
  Aij0[5] = aij_amp*(-3.0*x2*x3/r2);

  *a2 = 10.125*m_bh*m_bh*m_bh*m_bh / (r_sch*r_sch*r_sch*r_sch*r_sch*r_sch);
}

//----------------------------------------------------------------------------------------
//! \fn void WormholeBackground(...)
//! \brief Conformally-flat, time-symmetric ("wormhole") Brill-Lindquist single-puncture
//! background -- Ahat0=beta0=0 identically. NOT derived from mod_bh.f90; kept only as a
//! cheap debug fallback (CFC_PUNCTURE_TDE_PLAN.md Sec 3.2/Sec 5 Phase A item 2),
//! mirroring the formula already in production use at
//! src/pgen/z4c/z4c_one_puncture.cpp's ADMOnePuncture. Also the exact M_BH->0 limit of
//! TrumpetBackground (Sec 5 Phase A item 7's bring-up check).
KOKKOS_INLINE_FUNCTION
void WormholeBackground(Real m_bh, Real x1, Real x2, Real x3,
                        Real *psi0, Real *alpha0, Real beta0[3], Real Aij0[6], Real *a2) {
  Real r = std::sqrt(x1*x1 + x2*x2 + x3*x3);
  Real psi = 1.0 + 0.5*m_bh/r;
  *psi0 = psi;
  *alpha0 = (1.0 - 0.5*m_bh/r) / psi;
  for (int a = 0; a < 3; ++a) { beta0[a] = 0.0; }
  for (int a = 0; a < 6; ++a) { Aij0[a] = 0.0; }
  *a2 = 0.0;
}

//----------------------------------------------------------------------------------------
//! \fn void FillPunctureBackground(MeshBlockPack *pmbp, Real m_bh,
//!            DvceArray5D<Real> &u_psi0, DvceArray5D<Real> &u_alpha0_psi0,
//!            AthenaTensor<Real, TensorSymm::NONE, 3, 1> &beta0_u,
//!            AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &a0_dd,
//!            DvceArray5D<Real> &a0_sq,
//!            AthenaTensor<Real, TensorSymm::NONE, 3, 1> &grad_ap6_0)
//! \brief One-time (per construction / per AMR regrid) ghost-inclusive fill of the
//! trumpet background arrays, mirroring src/pgen/z4c/z4c_one_puncture.cpp's
//! ADMOnePuncture -- CellCenterX for coordinates, is-ng..ie+ng bounds. u_alpha0_psi0
//! stores alpha0*psi0 (the product, matching delta_alpha_psi's own convention).
//! grad_ap6_0 = D_j(alpha0*psi0^-6) (Sec 3.9's closed-form ingredient, the *bare*
//! gradient 3-vector -- NOT contracted with Ahat0^ij) is assembled here from
//! TrumpetBackground's dpsi0/dalpha0 via D_j(alpha0*psi0^-6) =
//! (alpha0*psi0^-6)*[dalpha0/alpha0 - 6*dpsi0/psi0]*x_j/r. Consumed by
//! cfc.cpp::BuildShiftSourceImpl's "2*Ahats^ij*D_j(alpha0*psi0^-6)" term (Sec 5 Phase A
//! item 7) -- the bare form is what's needed there (dotted against the matter-only
//! Ahats^ij = Ahat^ij-Ahat0^ij at the call site), not a pre-contracted-with-Ahat0
//! quantity, which cancels out of the final residual entirely (Sec 3.9).
void FillPunctureBackground(MeshBlockPack *pmbp, Real m_bh,
                            DvceArray5D<Real> &u_psi0, DvceArray5D<Real> &u_alpha0_psi0,
                            AthenaTensor<Real, TensorSymm::NONE, 3, 1> &beta0_u,
                            AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &a0_dd,
                            DvceArray5D<Real> &a0_sq,
                            AthenaTensor<Real, TensorSymm::NONE, 3, 1> &grad_ap6_0);

}  // namespace cfc

#endif  // CFC_CFC_PUNCTURE_HPP_
