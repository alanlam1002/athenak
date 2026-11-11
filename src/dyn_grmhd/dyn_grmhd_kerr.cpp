//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyn_grmhd_kerr.cpp
//! \brief <cfc> kerr_background = true (K-004 B3; kerr_methods sec:valencia, sec:keff):
//! AddCoordTermsEOS's sources on gamma = psi^4 gt with the background's parts analytic.
//!   s_k   = -E d_k alpha + S_j d_k beta'^j + 1/2 alpha S^ij d_k gamma_ij            (eq:smom)
//!   s_tau = 1/2 psi^4 S^ij (Lt b_res)_ij + alpha0 (psi/psi0)^4 S^ij K0_ij
//!           - S^j d_j alpha                                                       (eq:sen)
//! with d alpha = d alpha0 + Dx(delta alpha), d beta' = d beta0 + Dx(b_res) (d xidot = 0),
//! d gamma_ij = 4 gamma_ij (d ln psi0 + Dx(ln(psi/psi0))) + psi^4 d gt_ij: d alpha0, d beta0,
//! d ln psi0, d gt and K0 from W1 (CFC::KerrBgForSources, kbg_ about bg_xbh_, the background
//! the stage's metric was assembled on: SolveVecXTask refills it after MHD_AddSrc,
//! cfc.cpp:1818-1827). Dx<NGHOST> is applied only to the residuals delta alpha = alpha -
//! alpha0, b_res = beta' - beta0 - xidot and ln(psi/psi0) (kill_res 0-3 and kbg_ell, every
//! cell), and the one Dx(b_res) array enters both s_k and s_tau (B_E = B_S, sec:pair; the
//! host test kerr_work/tests/kerr_pair_test.cpp). The trace of Keff is dropped (eq:kcfcbg).
//! The dual-energy diagnostic and the smooth-excision damping are AddCoordTermsEOS's.

#include <math.h>

#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/coordinates.hpp"
#include "cfc/cfc.hpp"
#include "cfc/kerr_background.hpp"
#include "dyn_grmhd.hpp"
#include "dyn_grmhd/kerr_sources.hpp"
#include "utils/finite_diff.hpp"

#include "eos/primitive_solver_hyd.hpp"
#include "eos/primitive-solver/idealgas.hpp"
#include "eos/primitive-solver/eos_compose.hpp"
#include "eos/primitive-solver/eos_hybrid.hpp"
#include "eos/primitive-solver/piecewise_polytrope.hpp"
#include "eos/primitive-solver/reset_floor.hpp"

namespace dyngr {

//----------------------------------------------------------------------------------------
//! \fn void DynGRMHD::KerrBgResidual
//! \brief the Kerr form of KillingBgResidual (dyn_grmhd_killing.cpp:426): on every cell
//! (ghosts included, the ADM exchange's), from kbg_ (the background the metric was built on)
//!   kill_res 0 = alpha - alpha0, 1-3 = beta'^i - beta0^i - xidot^i,
//!   kbg_ell = ln(psi/psi0) = ln(det gamma)/12 - ln psi0   (det gt = 1)
void DynGRMHD::KerrBgResidual() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int nmb = pmy_pack->nmb_thispack;
  const int n3 = indcs.nx3 + 2*indcs.ng, n2 = indcs.nx2 + 2*indcs.ng;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  if (kill_res.extent_int(0) < nmb || kill_res.extent_int(1) != 4) {
    Kokkos::realloc(kill_res, nmb, 4, n3, n2, n1);
  }
  if (kbg_ell.extent_int(0) < nmb) Kokkos::realloc(kbg_ell, nmb, n3, n2, n1);
  auto bg = pmy_pack->pcfc->KerrBgForSources();
  auto &adm = pmy_pack->padm->adm;
  const Real gx0 = pmy_pack->padm->frame_vel_u[0], gx1 = pmy_pack->padm->frame_vel_u[1];
  const Real gx2 = pmy_pack->padm->frame_vel_u[2];
  auto &res = kill_res;
  auto &el = kbg_ell;
  par_for("kerr_bg_res", DevExeSpace(), 0, nmb-1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    res(m, 0, k, j, i) = adm.alpha(m, k, j, i) - bg(m, cfc::kKbgAlpha0, k, j, i);
    res(m, 1, k, j, i) = adm.beta_u(m, 0, k, j, i) - (bg(m, cfc::kKbgBeta0, k, j, i) + gx0);
    res(m, 2, k, j, i) = adm.beta_u(m, 1, k, j, i) - (bg(m, cfc::kKbgBeta0 + 1, k, j, i) + gx1);
    res(m, 3, k, j, i) = adm.beta_u(m, 2, k, j, i) - (bg(m, cfc::kKbgBeta0 + 2, k, j, i) + gx2);
    const Real detg = adm::SpatialDet(adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
                                      adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
                                      adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i));
    el(m, k, j, i) = log(detg)/12.0 - log(bg(m, cfc::kKbgPsi0, k, j, i));
  });
}

//----------------------------------------------------------------------------------------
//! \fn void DynGRMHDPS::AddCoordTermsKerr
//! \brief AddCoordTermsEOS on the Kerr path (above); called from its top when
//! pcfc->KerrOn(), so the default kernel is not changed

template<class EOSPolicy, class ErrorPolicy> template<int NGHOST>
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddCoordTermsKerr(const DvceArray5D<Real> &prim,
    const DvceArray5D<Real> &bcc,
    const Real dt, DvceArray5D<Real> &rhs) {
  if (!pmy_pack->pmesh->three_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<cfc> kerr_background needs a 3D mesh" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  KerrBgResidual();
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size  = pmy_pack->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nmb = pmy_pack->nmb_thispack;

  auto &adm = pmy_pack->padm->adm;
  auto &eos_ = eos.ps.GetEOS();
  auto bg = pmy_pack->pcfc->KerrBgForSources();
  auto &res = kill_res;
  auto &ell = kbg_ell;

  bool smoothing = pmy_pack->pcoord->coord_data.smooth_excision;
  auto &floor = pmy_pack->pcoord->excision_floor;
  Real &dexcise = pmy_pack->pcoord->coord_data.dexcise;
  Real &texcise = pmy_pack->pcoord->coord_data.texcise;
  Real &tdamp = pmy_pack->pcoord->coord_data.tdamp;

  int &nhyd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;

  const Real mb = eos.ps.GetEOS().GetBaryonMass();
  const int imap[3][3] = {
    {S11, S12, S13},
    {S12, S22, S23},
    {S13, S23, S33}
  };

  const bool ddon_ = eos.de_diag_on;
  auto &dd_ = de_diag;
  auto &fx1_ = pmy_pack->pmhd->uflx.x1f;
  auto &fx2_ = pmy_pack->pmhd->uflx.x2f;
  auto &fx3_ = pmy_pack->pmhd->uflx.x3f;
  par_for("coord_src_kerr", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real g3d[NSPMETRIC] = {adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
                           adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
                           adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i)};
    const Real& alpha = adm.alpha(m, k, j, i);
    Real detg = adm::SpatialDet(g3d[S11], g3d[S12], g3d[S13],
                                g3d[S22], g3d[S23], g3d[S33]);
    Real vol = sqrt(detg);
    Real g3u[NSPMETRIC] = {0.};
    adm::SpatialInv(1.0/detg, g3d[S11], g3d[S12], g3d[S13], g3d[S22], g3d[S23], g3d[S33],
                    &g3u[S11], &g3u[S12], &g3u[S13], &g3u[S22], &g3u[S23], &g3u[S33]);
    const Real p4 = cbrt(detg);   // psi^4 (det gt = 1)

    // the background (analytic, W1) and the residuals' stencil values
    Real gt[6], k0[6], dgt[18], da0[3], db0[9], dl0[3], br[3];
    for (int q = 0; q < 6; ++q) {
      gt[q] = bg(m, cfc::kKbgGt + q, k, j, i);
      k0[q] = bg(m, cfc::kKbgK0 + q, k, j, i);
    }
    for (int q = 0; q < 18; ++q) dgt[q] = bg(m, cfc::kKbgDGt + q, k, j, i);
    for (int q = 0; q < 9; ++q) db0[q] = bg(m, cfc::kKbgDBeta0 + q, k, j, i);
    for (int a = 0; a < 3; ++a) {
      da0[a] = bg(m, cfc::kKbgDAlpha0 + a, k, j, i);
      dl0[a] = bg(m, cfc::kKbgDLnPsi0 + a, k, j, i);
      br[a] = res(m, 1 + a, k, j, i);
    }
    const Real psi0 = bg(m, cfc::kKbgPsi0, k, j, i);
    const Real alpha0 = bg(m, cfc::kKbgAlpha0, k, j, i);
    const Real p40 = psi0*psi0*psi0*psi0;
    Real idx[] = {1./size.d_view(m).dx1, 1./size.d_view(m).dx2, 1./size.d_view(m).dx3};
    Real dda[3], dell[3], dbr[3][3];
    for (int a = 0; a < 3; ++a) {
      dda[a] = Dx<NGHOST>(a, idx, res, m, 0, k, j, i);
      dell[a] = Dx<NGHOST>(a, idx, ell, m, k, j, i);
      for (int c = 0; c < 3; ++c) dbr[a][c] = Dx<NGHOST>(a, idx, res, m, 1 + c, k, j, i);
    }
    Real dalpha_d[3], dbeta_du[3][3], dg_ddd[3][3][3];
    kerrsrc::ComposeDerivs(g3d, p4, dgt, da0, dda, db0, dbr, dl0, dell,
                           dalpha_d, dbeta_du, dg_ddd);

    // Fluid quantities (as AddCoordTermsEOS)
    Real prim_pt[NPRIM] = {0.0};
    prim_pt[PRH] = prim(m, IDN, k, j, i)/mb;
    prim_pt[PVX] = prim(m, IVX, k, j, i);
    prim_pt[PVY] = prim(m, IVY, k, j, i);
    prim_pt[PVZ] = prim(m, IVZ, k, j, i);
    for (int s = 0; s < nscal; s++) {
      prim_pt[PYF + s] = prim(m, nhyd + s, k, j, i);
    }
    prim_pt[PPR] = prim(m, IPR, k, j, i);
    prim_pt[PTM] = eos_.GetTemperatureFromP(prim_pt[PRH], prim_pt[PPR], &prim_pt[PYF]);
    Real H =
      prim(m, IDN, k, j, i)*eos_.GetEnthalpy(prim_pt[PRH], prim_pt[PTM], &prim_pt[PYF]);
    Real usq = Primitive::SquareVector(&prim_pt[PVX], g3d);
    Real const Wsq = 1.0 + usq;
    Real const W = sqrt(Wsq);
    Real B_u[NMAG] = {bcc(m, IBX, k, j, i)/vol,
                      bcc(m, IBY, k, j, i)/vol,
                      bcc(m, IBZ, k, j, i)/vol};
    Real Bv = 0.0;
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3; b++) {
        Bv += adm.g_dd(m,a,b,k,j,i)*prim_pt[PVX + a]*B_u[b];
      }
    }
    Real Bsq = Primitive::SquareVector(B_u, g3d);
    Bv = Bv/W;
    Real bsq = Bv*Bv + Bsq/Wsq;
    Real E = (H*Wsq + Bsq) - prim_pt[PPR] - 0.5*bsq;
    Real S_d[3] = {0.0};
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3; b++) {
        S_d[a] += ((H*Wsq + Bsq)*prim_pt[PVX + b]/W - Bv*B_u[b])*g3d[imap[a][b]];
      }
    }
    Real S_uu[3][3];
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b <= a; b++) {
        S_uu[a][b] = (H + Bsq/Wsq)*prim_pt[PVX + a]*prim_pt[PVX + b]
                      - B_u[a]*B_u[b]/Wsq
                      - Bv*(B_u[a]*prim_pt[PVX + b] + B_u[b]*prim_pt[PVX + a])/W
                      + (prim_pt[PPR] + 0.5*bsq)*g3u[imap[a][b]];
        S_uu[b][a] = S_uu[a][b];
      }
    }

    // energy (eq:sen) and momentum (eq:smom)
    const Real src_ks = kerrsrc::EnergyK(p4, p4/p40, alpha0, gt, dgt, br, dbr, k0, S_uu);
    const Real src_da = kerrsrc::EnergyAlpha(g3u, S_d, dalpha_d);
    rhs(m, IEN, k, j, i) += dt*vol*(src_ks + src_da);
    Real s_k[3];
    kerrsrc::Momentum(alpha, E, S_d, S_uu, dalpha_d, dbeta_du, dg_ddd, s_k);
    for (int a = 0; a < 3; a++) rhs(m, IM1 + a, k, j, i) += dt*vol*s_k[a];

    if (ddon_) {   // AddCoordTermsEOS's internal-energy budget, with this path's sources
      dd_(m, 6, k, j, i) = src_ks;
      dd_(m, 7, k, j, i) = src_da;
      auto divf = [&](int n) {
        Real d = (fx1_(m, n, k, j, i+1) - fx1_(m, n, k, j, i))/size.d_view(m).dx1;
        d += (fx2_(m, n, k, j+1, i) - fx2_(m, n, k, j, i))/size.d_view(m).dx2;
        d += (fx3_(m, n, k+1, j, i) - fx3_(m, n, k, j, i))/size.d_view(m).dx3;
        return d/vol;
      };
      Real dD = -divf(IDN), dE = -divf(IEN) + dD;
      Real vdS_f = 0.0, vdS_s = 0.0;
      for (int a = 0; a < 3; a++) {
        Real va = prim_pt[PVX + a]/W;
        vdS_f -= va*divf(IM1 + a);
        vdS_s += va*s_k[a];
      }
      Real rho_ = prim(m, IDN, k, j, i);
      Real e_int = W*fmax(H - rho_ - prim_pt[PPR], 1.0e-300);
      dd_(m, 8, k, j, i) = (W*(dE - vdS_f) - dD)/e_int;
      dd_(m, 9, k, j, i) = W*(src_ks + src_da - vdS_s)/e_int;
    }

    if (smoothing) {   // as AddCoordTermsEOS
      Real D = prim(m, IDN, k, j, i) * W;
      Real tau = E - D;
      Real tau_ex = eos_.GetEnergy(dexcise/mb, texcise, &prim_pt[PYF])
                                    + 0.5*Bsq - dexcise;
      rhs(m, IDN, k, j, i) -= (dt*vol*floor(m,k,j,i)*(D-dexcise))/tdamp;
      rhs(m, IM1, k, j, i) -= (dt*floor(m,k,j,i)*vol*S_d[0])/tdamp;
      rhs(m, IM2, k, j, i) -= (dt*floor(m,k,j,i)*vol*S_d[1])/tdamp;
      rhs(m, IM3, k, j, i) -= (dt*floor(m,k,j,i)*vol*S_d[2])/tdamp;
      rhs(m, IEN, k, j, i) -= dt*vol*floor(m,k,j,i)*(tau - tau_ex)/tdamp;
      for (int s = 0; s < nscal; s++) {
        rhs(m, IYF+s, k, j, i) -= (dt*vol*floor(m,k,j,i)*(D-dexcise)*prim_pt[PYF+s])
                                      /tdamp;
      }
    }
  });
}

#define INSTANTIATE_KERR_TERMS(EOSPolicy, ErrorPolicy) \
template void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddCoordTermsKerr<2>( \
    const DvceArray5D<Real> &, const DvceArray5D<Real> &, const Real, DvceArray5D<Real> &); \
template void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddCoordTermsKerr<3>( \
    const DvceArray5D<Real> &, const DvceArray5D<Real> &, const Real, DvceArray5D<Real> &); \
template void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddCoordTermsKerr<4>( \
    const DvceArray5D<Real> &, const DvceArray5D<Real> &, const Real, DvceArray5D<Real> &);

INSTANTIATE_KERR_TERMS(Primitive::IdealGas, Primitive::ResetFloor);
INSTANTIATE_KERR_TERMS(Primitive::PiecewisePolytrope, Primitive::ResetFloor);
INSTANTIATE_KERR_TERMS(Primitive::EOSCompOSE<Primitive::NormalLogs>, Primitive::ResetFloor);
INSTANTIATE_KERR_TERMS(Primitive::EOSCompOSE<Primitive::NQTLogs>, Primitive::ResetFloor);
INSTANTIATE_KERR_TERMS(Primitive::EOSHybrid<Primitive::NormalLogs>, Primitive::ResetFloor);
INSTANTIATE_KERR_TERMS(Primitive::EOSHybrid<Primitive::NQTLogs>, Primitive::ResetFloor);

#undef INSTANTIATE_KERR_TERMS

}  // namespace dyngr
