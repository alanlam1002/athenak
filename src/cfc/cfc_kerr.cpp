//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cfc_kerr.cpp
//! \brief W4 (K-004 Part B; kerr_methods.tex sec. 8): the Kerr path of the CFC step.
//!
//! With <cfc> kerr_background = true every flat-path solver function of cfc.cpp branches
//! here at its top; the task graph, its ghost exchanges and the floors are unchanged.
//!   X:     DL X = 8 pi gt^ij S*_j (eq:cX), MGKerrVectorDriver <cfc_kerr_x>;
//!   Ahat:  Ahat^ij = Ahat0^ij + (Lt X)^ij (eq:mom, eq:G2; Dx<NGHOST> of the exchanged X,
//!          the background's d gt analytic);
//!   psi:   eq:cpsi, MGKerrScalarDriver <cfc_kerr_psi> (kind psi), g_ij = psi^4 gt_ij;
//!   chi:   eq:cchi, MGKerrScalarDriver <cfc_kerr_chi> (kind chi), chi = alpha psi;
//!   beta:  eq:cbeta, MGKerrVectorDriver <cfc_kerr_beta>; beta' = beta0 + b + xidot.
//! The background (kbg_, W1) is refilled about x_BH(t) once per solve (SolveVecXTask), and
//! the flat-named background arrays CFC's shared code reads (u_psi0, u_alpha0_psi0,
//! beta0_u, a0_dd, a0_sq, grad_ap6_0) are derived from it, with a0_dd = Ahat0^ij (upper,
//! like a_dd on this path) and grad_ap6_0 = d(alpha0 psi0^-6) analytic, so that
//! BuildShiftSource's cross terms 2[Ahat^ij d_j dap6 + (Ahat - Ahat0)^ij grad_ap6_0_j] are
//! eq:cbeta's.

#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "parameter_input.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "utils/finite_diff.hpp"
#include "mhd/mhd.hpp"
#include "driver/driver.hpp"
#include "cfc.hpp"
#include "kerr_background.hpp"
#include "kerr_mg_ops.hpp"
#include "mg_kerr_scalar.hpp"
#include "mg_kerr_vector.hpp"

namespace cfc {

namespace {

[[noreturn]] void KerrFatal(const std::string &msg) {
  std::cout << "### FATAL ERROR in CFC (kerr_background): " << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

KOKKOS_INLINE_FUNCTION int KSym(int a, int b) {
  // xx 0, xy 1, xz 2, yy 3, yz 4, zz 5
  const int i = (a < b) ? a : b, j = (a < b) ? b : a;
  return (i == 0) ? j : ((i == 1) ? 2 + j : 5);
}

//! Ahat^ij = Ahat0^ij + (Lt X)^ij (eq:G2), with d_k gt^ij = -gt^ia gt^jb d_k gt_ab from the
//! background (no finite differences of it) and d_k X^j = Dx<NGHOST> of the exchanged X
//! (u_p_x ch. 0-2). Also Ahat^2 = gt_ik gt_jl Ahat^ij Ahat^kl and, cancellation-free,
//! Ahat^2 - Ahat0^2 = gt_ik gt_jl (Ahat - Ahat0)^ij (Ahat + Ahat0)^kl. Interior cells.
template <int NGHOST>
void KerrAhatImpl(MeshBlockPack *pmbp, const DvceArray5D<Real> &kbg,
                  const DvceArray5D<Real> &ux,
                  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &a_dd, DvceArray5D<Real> &a_sq,
                  DvceArray5D<Real> &da2) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  par_for("cfc_kerr_ahat", DevExeSpace(), 0, pmbp->nmb_thispack-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real idx[] = {1.0/size.d_view(m).dx1, 1.0/size.d_view(m).dx2,
                        1.0/size.d_view(m).dx3};
    Real dX[3][3];   // d_k X^c
    for (int kk = 0; kk < 3; ++kk) {
      for (int c = 0; c < 3; ++c) { dX[kk][c] = Dx<NGHOST>(kk, idx, ux, m, c, k, j, i); }
    }
    const Real divX = dX[0][0] + dX[1][1] + dX[2][2];
    Real X[3];
    for (int c = 0; c < 3; ++c) { X[c] = ux(m, c, k, j, i); }
    Real gu[3][3], gd[3][3];
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        gu[a][b] = kbg(m, kKbgGtu + KSym(a, b), k, j, i);
        gd[a][b] = kbg(m, kKbgGt + KSym(a, b), k, j, i);
      }
    }
    // X^k d_k gt_ab
    Real xdg[3][3];
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        Real s = 0.0;
        for (int kk = 0; kk < 3; ++kk) {
          s += X[kk]*kbg(m, kKbgDGt + 6*kk + KSym(a, b), k, j, i);
        }
        xdg[a][b] = s;
      }
    }
    Real ah[3][3], a0[3][3];
    for (int p = 0; p < 3; ++p) {
      for (int q = p; q < 3; ++q) {
        Real lx = -2.0/3.0*gu[p][q]*divX;
        for (int kk = 0; kk < 3; ++kk) { lx += gu[p][kk]*dX[kk][q] + gu[q][kk]*dX[kk][p]; }
        // -X^k d_k gt^pq = +gt^pa gt^qb X^k d_k gt_ab
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) { lx += gu[p][a]*gu[q][b]*xdg[a][b]; }
        }
        const Real b0 = kbg(m, kKbgAhu + KSym(p, q), k, j, i);
        a0[p][q] = a0[q][p] = b0;
        ah[p][q] = ah[q][p] = b0 + lx;
        a_dd(m, p, q, k, j, i) = b0 + lx;
      }
    }
    // lowered: gt_ia gt_jb A^ab
    Real sq = 0.0, dsq = 0.0;
    for (int p = 0; p < 3; ++p) {
      for (int q = 0; q < 3; ++q) {
        Real lo = 0.0, lsum = 0.0;
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            lo += gd[p][a]*gd[q][b]*ah[a][b];
            lsum += gd[p][a]*gd[q][b]*(ah[a][b] + a0[a][b]);
          }
        }
        sq += lo*ah[p][q];
        dsq += lsum*(ah[p][q] - a0[p][q]);
      }
    }
    a_sq(m, 0, k, j, i) = sq;
    da2(m, 0, k, j, i) = dsq;
  });
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void CFC::KerrSetup(ParameterInput *pin)
//! \brief the constructor's Kerr part (after the table load and the D3/D4 radii): the
//! option checks, kbg_, the four Kerr drivers, the first refill

void CFC::KerrSetup(ParameterInput *pin) {
  // ---- options the Kerr path does not support (they assume the flat storage convention
  // or the trumpet's closed forms)
  if (!puncture_enabled_) {
    KerrFatal("needs <cfc> puncture_enabled = true (x_BH(t), M_BH and the split).");
  }
  if (rs_on_) { KerrFatal("<cfc> residual_scale is not supported."); }
  if (pin->DoesParameterExist("dyn_grmhd", "metric_terms") &&
      pin->GetString("dyn_grmhd", "metric_terms") != "cell") {
    KerrFatal("<dyn_grmhd> metric_terms must be cell (the face terms are trumpet-only).");
  }
  if (pin->DoesParameterExist("dyn_grmhd", "trace_diag_ahat") &&
      pin->GetString("dyn_grmhd", "trace_diag_ahat") != "off") {
    KerrFatal("<dyn_grmhd> trace_diag_ahat is not supported.");
  }
  if (pin->DoesParameterExist("dyn_grmhd", "killing_momentum_bg")) {
    KerrFatal("<dyn_grmhd> killing_momentum_bg is not supported.");
  }
  if (pin->DoesParameterExist("mhd", "energy_variable") &&
      pin->GetString("mhd", "energy_variable") != "valencia") {
    KerrFatal("<mhd> energy_variable must be valencia (the Killing variants read the "
              "trumpet background).");
  }
  // K-004 B3: these read the trumpet background or overwrite kill_res/kbg_ell, which
  // AddCoordTermsKerr (dyn_grmhd_kerr.cpp) fills with the Kerr residuals
  if (pin->DoesParameterExist("mhd", "killing_source_diag") &&
      pin->GetBoolean("mhd", "killing_source_diag")) {
    KerrFatal("<mhd> killing_source_diag is not supported.");
  }
  if (pin->DoesParameterExist("dyn_grmhd", "tau_source_K") &&
      pin->GetString("dyn_grmhd", "tau_source_K") != "xcfc") {
    KerrFatal("<dyn_grmhd> tau_source_K must be xcfc.");
  }
  if (pin->DoesParameterExist("dyn_grmhd", "trace_diag") &&
      pin->GetBoolean("dyn_grmhd", "trace_diag")) {
    KerrFatal("<dyn_grmhd> trace_diag is not supported.");
  }
  if (pin->DoesParameterExist("mhd", "tau_budget_every") &&
      pin->GetInteger("mhd", "tau_budget_every") > 0) {
    KerrFatal("<mhd> tau_budget_every is not supported.");
  }
  if (!pmy_pack->pmesh->three_d) { KerrFatal("needs a 3D mesh."); }
  // D4 (K-004 B4): for a > 0 the mask is the sphere r < max(r_D4, r_f) about bc_center with
  // r_D4 = 0.75 min_theta r_AH, and no lapse test (the Kerr lapse need not fall below
  // excise_lapse inside r_AH). a = 0 keeps the lapse scheme unchanged.
  if (kerr_table_->a > 0.0) {
    auto &cd = pmy_pack->pcoord->coord_data;
    if (!cd.bh_excise || cd.excision_scheme != ExcisionScheme::lapse) {
      KerrFatal("a > 0 needs <coord> excise = true and excision_scheme = lapse (D4).");
    }
    cd.excision_rfloor = 0.75*kerr_r_ah_min_;
    cd.excision_lapse_off = true;
    // also into pin: a regrid rebuilds Coordinates from pin (coordinates.cpp reads these),
    // which otherwise reverted the mask to the lapse test after the first regrid (K-005)
    pin->SetBoolean("coord", "kerr_excision_lapse_off", true);
    pin->SetReal("coord", "kerr_excision_rfloor", cd.excision_rfloor);
    if (global_variable::my_rank == 0) {
      std::cout << "[kerr] D4 excision: r < max(" << cd.excision_rfloor
                << ", r_f) about the hole, lapse mask off" << std::endl;
    }
  } else if (pin->DoesParameterExist("coord", "kerr_excision_lapse_off")) {
    // a = 0 restarted from an a > 0 restart file: drop its D4 keys
    auto &cd = pmy_pack->pcoord->coord_data;
    cd.excision_rfloor = 0.0;
    cd.excision_lapse_off = false;
    pin->SetBoolean("coord", "kerr_excision_lapse_off", false);
    pin->SetReal("coord", "kerr_excision_rfloor", 0.0);
  }
  // M_BH is fixed on this path (the tables are per a, in units of M): accretion is a
  // tally only, as freeze_puncture_mass (R-155)
  if (accrete_to_puncture_ || !freeze_puncture_mass_) {
    accrete_to_puncture_ = false;
    freeze_puncture_mass_ = true;
    pin->SetBoolean("cfc", "accrete_to_puncture", false);
    pin->SetBoolean("cfc", "freeze_puncture_mass", true);
    if (global_variable::my_rank == 0) {
      std::cout << "[kerr] M_BH fixed: accretion is tally-only (freeze_puncture_mass)"
                << std::endl;
    }
  }
  kerr_log_ = pin->GetOrAddInteger("cfc", "kerr_log", 0);

  // ---- storage
  const int nmb = std::max(pmy_pack->nmb_thispack, pmy_pack->pmesh->nmb_maxperrank);
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*indcs.ng) : 1;
  const int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*indcs.ng) : 1;
  Kokkos::realloc(kbg_, nmb, kKbgN, n3, n2, n1);
  Kokkos::realloc(kerr_da2_, nmb, 1, n3, n2, n1);
  // X and b live in u_p_x/u_p_beta channels 0-2; the eta channel stays 0
  Kokkos::deep_copy(u_p_x, 0.0);
  Kokkos::deep_copy(u_p_beta, 0.0);

  // ---- the drivers (kind defaults so a deck need not repeat them)
  kmg::CoefSource cs;
  cs.mode = 2;
  cs.tab = *kerr_table_;
  cs.m_bh = puncture_mass_;
  for (int a = 0; a < 3; ++a) { cs.xbh[a] = bg_xbh_[a]; }
  cs.blend_r1 = kerr_blend_r1_;
  cs.blend_r2 = kerr_blend_r2_;
  kmg::Manuf mf;
  kmg::VManuf vm;
  pin->GetOrAddString("cfc_kerr_psi", "kind", "psi");
  pin->GetOrAddString("cfc_kerr_chi", "kind", "chi");
  pkx_ = new MGKerrVectorDriver(pmy_pack, pin, "cfc_kerr_x", cs, vm);
  pkbeta_ = new MGKerrVectorDriver(pmy_pack, pin, "cfc_kerr_beta", cs, vm);
  pkpsi_ = new MGKerrScalarDriver(pmy_pack, pin, "cfc_kerr_psi", cs, mf);
  pkchi_ = new MGKerrScalarDriver(pmy_pack, pin, "cfc_kerr_chi", cs, mf);
  // K-005 B1: same background each stage (KerrRefill before the X solve) -> share it
  pkbeta_->coeff_donor_ = pkx_;
  pkchi_->bg_donor_ = pkpsi_;
  if (pkpsi_->kind_ != kmg::kKindPsi || pkchi_->kind_ != kmg::kKindChi) {
    KerrFatal("<cfc_kerr_psi> kind must be psi and <cfc_kerr_chi> kind chi.");
  }
  if (pkx_->src_blob_ || pkbeta_->src_blob_) {
    KerrFatal("<cfc_kerr_x/beta> ssrc = blob is a test option.");
  }
  pkx_->SetLogName("kX");
  pkbeta_->SetLogName("kbeta");
  pkpsi_->SetLogName("kpsi");
  pkchi_->SetLogName("kchi");
  pkpsi_->SetMatter(u_tilde, kerr_da2_, delta_psi, indcs.ng);
  pkchi_->SetMatter(u_plus_2s, kerr_da2_, delta_psi, indcs.ng);
  if (cfc_init_use_psi5_ && global_variable::my_rank == 0) {
    std::cout << "[kerr] <cfc> init_use_psi5_source ignored: the Kerr psi equation takes "
              << "E* (u_tilde) at initialisation too" << std::endl;
  }

  KerrRefill(bg_xbh_);
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::KerrRefill(const Real xbh[3])
//! \brief kbg_ about xbh (every cell of the pack, ghosts included), the flat-named
//! background arrays from it, and the drivers' hole position

void CFC::KerrRefill(const Real xbh[3]) {
  Kokkos::Timer tm;
  {
    // CFC's arrays are sized max(nmb_thispack, nmb_maxperrank); follow them after a regrid
    const int nmb = u_psi0.extent_int(0);
    if (kbg_.extent_int(0) != nmb) {
      Kokkos::realloc(kbg_, nmb, kKbgN, kbg_.extent_int(2), kbg_.extent_int(3),
                      kbg_.extent_int(4));
      Kokkos::realloc(kerr_da2_, nmb, 1, kbg_.extent_int(2), kbg_.extent_int(3),
                      kbg_.extent_int(4));
      pkpsi_->SetMatter(u_tilde, kerr_da2_, delta_psi, pmy_pack->pmesh->mb_indcs.ng);
      pkchi_->SetMatter(u_plus_2s, kerr_da2_, delta_psi, pmy_pack->pmesh->mb_indcs.ng);
    }
  }
  FillKerrBackground(pmy_pack, *kerr_table_, puncture_mass_, xbh, kerr_blend_r1_,
                     kerr_blend_r2_, kbg_);
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*indcs.ng) : 1;
  const int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*indcs.ng) : 1;
  auto bg = kbg_;
  auto psi0 = u_psi0;
  auto ap0 = u_alpha0_psi0;
  auto b0 = beta0_u;
  auto a0 = a0_dd;
  auto a0s = a0_sq;
  auto g6 = grad_ap6_0;
  par_for("cfc_kerr_refill", DevExeSpace(), 0, pmy_pack->nmb_thispack-1, 0, n3-1, 0, n2-1,
          0, n1-1, KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real p = bg(m, kKbgPsi0, k, j, i);
    const Real al = bg(m, kKbgAlpha0, k, j, i);
    psi0(m, 0, k, j, i) = p;
    ap0(m, 0, k, j, i) = al*p;
    for (int a = 0; a < 3; ++a) { b0(m, a, k, j, i) = bg(m, kKbgBeta0 + a, k, j, i); }
    Real s = 0.0;
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        const int q = KSym(a, b);
        const Real au = bg(m, kKbgAhu + q, k, j, i);
        a0(m, a, b, k, j, i) = au;
        s += ((a == b) ? 1.0 : 2.0)*au*bg(m, kKbgAhd + q, k, j, i);
      }
    }
    a0s(m, 0, k, j, i) = s;
    const Real p2 = p*p, ip6 = 1.0/(p2*p2*p2);
    for (int a = 0; a < 3; ++a) {
      g6(m, a, k, j, i) = ip6*(bg(m, kKbgDAlpha0 + a, k, j, i)
                               - 6.0*al*bg(m, kKbgDLnPsi0 + a, k, j, i));
    }
  });
  for (int a = 0; a < 3; ++a) {
    pkx_->cs_.xbh[a] = xbh[a];
    pkbeta_->cs_.xbh[a] = xbh[a];
    pkpsi_->cs_.xbh[a] = xbh[a];
    pkchi_->cs_.xbh[a] = xbh[a];
    kbg_xbh_[a] = xbh[a];
    bg_xbh_[a] = xbh[a];
  }
  kbg_valid_ = true;
  Kokkos::fence();
  kerr_fill_sec_ += tm.seconds();
  ++kerr_fill_n_;
}

const DvceArray5D<Real> &CFC::KerrBgForSources() {
  if (!kbg_valid_ || kbg_.extent_int(0) < pmy_pack->nmb_thispack) { KerrRefill(bg_xbh_); }
  return kbg_;
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::KerrVectorSourceX()
//! \brief eq:cX's right-hand side 8 pi gt^ij S*_j into p_src (interior; the eta channel
//! 0), and U~ = E*, u_raw, S*_i as the flat step 1 builds them (cons is densitized by
//! sqrt(det g) = psi^6, det gt = 1)

void CFC::KerrVectorSourceX() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int mg = mg_nghost_;
  auto &adm = pmy_pack->padm->adm;
  auto &cons = pmy_pack->pmhd->u0;
  auto ut = u_tilde;
  auto ur = u_raw;
  auto sd = s_tilde_d;
  auto ps = u_p_src;
  auto bg = kbg_;
  par_for("cfc_kerr_vecX_src", DevExeSpace(), 0, pmy_pack->nmb_thispack-1, ks, ke, js, je,
          is, ie, KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const int mk = k - ks + mg, mj = j - js + mg, mi = i - is + mg;
    ut(m, 0, k, j, i) = cons(m, IEN, k, j, i) + cons(m, IDN, k, j, i);
    const Real detg = adm::SpatialDet(adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
                                      adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
                                      adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i));
    ur(m, 0, k, j, i) = ut(m, 0, k, j, i)/Kokkos::sqrt(detg);
    Real s[3];
    for (int a = 0; a < 3; ++a) {
      s[a] = cons(m, IM1 + a, k, j, i);
      sd(m, a, k, j, i) = s[a];
    }
    for (int a = 0; a < 3; ++a) {
      Real v = 0.0;
      for (int b = 0; b < 3; ++b) { v += bg(m, kKbgGtu + KSym(a, b), k, j, i)*s[b]; }
      ps(m, a, mk, mj, mi) = 8.0*M_PI*v;
    }
    ps(m, 3, mk, mj, mi) = 0.0;
  });
}

//! eq:cbeta's pointwise part 16 pi alpha psi^-6 gt^ij S*_j into p_src (interior); the
//! cross terms follow in BuildShiftSource (AssembleVectorSource)
void CFC::KerrShiftPointSource() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int mg = mg_nghost_;
  auto dps = delta_psi;
  auto dap = delta_alpha_psi;
  auto psi0 = u_psi0;
  auto ap0 = u_alpha0_psi0;
  auto sd = s_tilde_d;
  auto ps = u_p_src;
  auto bg = kbg_;
  par_for("cfc_kerr_shift_src", DevExeSpace(), 0, pmy_pack->nmb_thispack-1, ks, ke, js, je,
          is, ie, KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const int mk = k - ks + mg, mj = j - js + mg, mi = i - is + mg;
    const Real psi = dps(m, 0, k, j, i) + psi0(m, 0, k, j, i);
    const Real p2 = psi*psi, p7 = p2*p2*p2*psi;
    const Real ap6 = (dap(m, 0, k, j, i) + ap0(m, 0, k, j, i))/p7;
    for (int a = 0; a < 3; ++a) {
      Real v = 0.0;
      for (int b = 0; b < 3; ++b) { v += bg(m, kKbgGtu + KSym(a, b), k, j, i)*sd(m, b, k, j, i); }
      ps(m, a, mk, mj, mi) = 16.0*M_PI*ap6*v;
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::KerrSolveX(Driver *pdriver, int stage)

void CFC::KerrSolveX(Driver *pdriver, int stage) {
  ComputeVectorCentroidX();
  for (int a = 0; a < 3; ++a) { pkx_->mpo_[a] = r_com_X_[a]; }
  AssembleVectorSource(/*for_shift=*/false);
  pkx_->LoadSource(u_p_src, mg_nghost_, -1.0);   // the driver solves -DL V = s
  {
    MGScopedTimer pt(MultigridDriver::phase_on, MultigridDriver::phase_solve_sec);
    pkx_->Solve(pdriver, stage);
  }
  pkx_->RetrieveSolution(u_p_x);
  if (kerr_log_ > 0 && global_variable::my_rank == 0 && stage <= 1 &&
      pmy_pack->pmesh->ncycle % kerr_log_ == 0) {
    std::printf("KERR FILL cycle= %d n= %ld total_s= %.4e per_fill_ms= %.4f "
                "vcoef fill/reuse/copy X= %ld/%ld/%ld beta= %ld/%ld/%ld sbg fill/copy/self "
                "psi= %ld/%ld/%ld chi= %ld/%ld/%ld\n",
                pmy_pack->pmesh->ncycle, kerr_fill_n_, kerr_fill_sec_,
                1.0e3*kerr_fill_sec_/static_cast<double>(std::max(kerr_fill_n_, 1L)),
                pkx_->coeff_fills_, pkx_->coeff_reuses_, pkx_->coeff_copies_,
                pkbeta_->coeff_fills_, pkbeta_->coeff_reuses_, pkbeta_->coeff_copies_,
                pkpsi_->bg_fills_, pkpsi_->bg_copies_, pkpsi_->bg_self_, pkchi_->bg_fills_,
                pkchi_->bg_copies_, pkchi_->bg_self_);
    if (pkx_->cache_check_ || pkchi_->cache_check_) {
      std::printf("KERR CACHECHECK cycle= %d X= %ld/%ld beta= %ld/%ld psi= %ld/%ld "
                  "chi= %ld/%ld (checks/diffs, rank 0)\n", pmy_pack->pmesh->ncycle,
                  pkx_->check_n_, pkx_->check_diff_, pkbeta_->check_n_,
                  pkbeta_->check_diff_, pkpsi_->check_n_, pkpsi_->check_diff_,
                  pkchi_->check_n_, pkchi_->check_diff_);
    }
  }
}

void CFC::KerrComputeAhat() {
  switch (pmy_pack->pmesh->mb_indcs.ng) {
    case 2: KerrAhatImpl<2>(pmy_pack, kbg_, u_p_x, a_dd, a_sq, kerr_da2_); break;
    case 3: KerrAhatImpl<3>(pmy_pack, kbg_, u_p_x, a_dd, a_sq, kerr_da2_); break;
    case 4: KerrAhatImpl<4>(pmy_pack, kbg_, u_p_x, a_dd, a_sq, kerr_da2_); break;
    default: KerrFatal("needs nghost 2, 3 or 4.");
  }
}

//----------------------------------------------------------------------------------------
//! \fn void CFC::KerrSolvePsi(Driver *pdriver, int stage)
//! \brief eq:cpsi. The first call (and the first after a regrid, psi_seeded_ = false)
//! seeds delta psi = det(g)^(1/12) - psi0 from adm (= psi - psi0 for g = psi^4 gt); a
//! restart's first call loads the restored delta_psi into the driver (kerr_psi_warm_).

void CFC::KerrSolvePsi(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  if (!psi_seeded_) {
    psi_seeded_ = true;
    kerr_psi_warm_ = false;
    const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
    const int ks = indcs.ks, ke = indcs.ke;
    auto &adm = pmy_pack->padm->adm;
    auto dps = delta_psi;
    auto psi0 = u_psi0;
    // fresh start: the pgen's psi_* - 1 on top of psi0 (differencing against psi0 seeds
    // delta psi ~ 1 - psi0 ~ -4 at the hole, which the psi^-7 FAS cannot recover, k004d)
    const bool pg = kerr_pgen_psi_;
    par_for("cfc_kerr_seed_psi", DevExeSpace(), 0, pmy_pack->nmb_thispack-1, ks, ke, js, je,
            is, ie, KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      const Real detg = adm::SpatialDet(adm.g_dd(m, 0, 0, k, j, i),
                                        adm.g_dd(m, 0, 1, k, j, i),
                                        adm.g_dd(m, 0, 2, k, j, i),
                                        adm.g_dd(m, 1, 1, k, j, i),
                                        adm.g_dd(m, 1, 2, k, j, i),
                                        adm.g_dd(m, 2, 2, k, j, i));
      dps(m, 0, k, j, i) = Kokkos::pow(detg, 1.0/12.0)
                           - (pg ? 1.0 : psi0(m, 0, k, j, i));
    });
    kerr_pgen_psi_ = false;
  }
  if (!kerr_psi_warm_) {
    pkpsi_->SeedInitialGuess(delta_psi, indcs.ng);
    kerr_psi_warm_ = true;
  }
  {
    MGScopedTimer pt(MultigridDriver::phase_on, MultigridDriver::phase_solve_sec);
    pkpsi_->Solve(pdriver, stage);
  }
  pkpsi_->RetrieveSolution(delta_psi);
  ApplyPsiFloor();
  KerrAssemblePsiMetric(false);
}

//! eq:cchi; seeded like the flat SolveLapse: chi = alpha (adm) x this stage's psi
void CFC::KerrSolveLapse(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int nmb = pmy_pack->nmb_thispack;
  const int n1 = u_tilde.extent_int(4), n2 = u_tilde.extent_int(3);
  const int n3 = u_tilde.extent_int(2);
  auto ut = u_tilde;
  auto st = s_tilde;
  auto u2 = u_plus_2s;
  par_for("cfc_kerr_u_plus_2s", DevExeSpace(), 0, nmb-1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    u2(m, 0, k, j, i) = ut(m, 0, k, j, i) + 2.0*st(m, 0, k, j, i);
  });
  if (!alpha_psi_seeded_) {
    alpha_psi_seeded_ = true;
    kerr_chi_warm_ = false;
    const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
    const int ks = indcs.ks, ke = indcs.ke;
    auto &adm = pmy_pack->padm->adm;
    auto dps = delta_psi;
    auto dap = delta_alpha_psi;
    auto psi0 = u_psi0;
    auto ap0 = u_alpha0_psi0;
    // fresh start: alpha = alpha0 x the pgen's alpha_* (adm.alpha ~ 1 at the hole)
    const bool pg = kerr_pgen_chi_;
    par_for("cfc_kerr_seed_chi", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      const Real p0 = psi0(m, 0, k, j, i);
      const Real al = pg ? adm.alpha(m, k, j, i)*ap0(m, 0, k, j, i)/p0 : adm.alpha(m, k, j, i);
      dap(m, 0, k, j, i) = al*(dps(m, 0, k, j, i) + p0) - ap0(m, 0, k, j, i);
    });
    kerr_pgen_chi_ = false;
  }
  if (!kerr_chi_warm_) {
    pkchi_->SeedInitialGuess(delta_alpha_psi, indcs.ng);
    kerr_chi_warm_ = true;
  }
  {
    MGScopedTimer pt(MultigridDriver::phase_on, MultigridDriver::phase_solve_sec);
    pkchi_->Solve(pdriver, stage);
  }
  pkchi_->RetrieveSolution(delta_alpha_psi);
  ApplyAlphaFloor();
}

//! eq:cbeta (AssembleVectorSource(true) branches to KerrShiftPointSource, then the shared
//! BuildShiftSource cross terms and r_com_beta_)
void CFC::KerrSolveShift(Driver *pdriver, int stage) {
  AssembleVectorSource(/*for_shift=*/true);
  for (int a = 0; a < 3; ++a) { pkbeta_->mpo_[a] = r_com_beta_[a]; }
  pkbeta_->LoadSource(u_p_src, mg_nghost_, -1.0);
  {
    MGScopedTimer pt(MultigridDriver::phase_on, MultigridDriver::phase_solve_sec);
    pkbeta_->Solve(pdriver, stage);
  }
  pkbeta_->RetrieveSolution(u_p_beta);
  if (mg_log_vcycles_) LogVcycles(stage);
}

//! beta_u = b (u_p_beta ch. 0-2, exchanged), every cell
void CFC::KerrReconstructShift() {
  auto ub = u_p_beta;
  auto bu = beta_u;
  const int n1 = ub.extent_int(4), n2 = ub.extent_int(3), n3 = ub.extent_int(2);
  par_for("cfc_kerr_reconstruct_shift", DevExeSpace(), 0, pmy_pack->nmb_thispack-1, 0, n3-1,
          0, n2-1, 0, n1-1, KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    for (int a = 0; a < 3; ++a) { bu(m, a, k, j, i) = ub(m, a, k, j, i); }
  });
}

//! psi4 = psi^4 and g_ij = psi^4 gt_ij (gt from kbg_, ghost-inclusive)
void CFC::KerrAssemblePsiMetric(bool with_ghosts) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int gi = with_ghosts ? indcs.ng : 0;
  const int gj = (with_ghosts && indcs.nx2 > 1) ? indcs.ng : 0;
  const int gk = (with_ghosts && indcs.nx3 > 1) ? indcs.ng : 0;
  const int is = indcs.is - gi, ie = indcs.ie + gi;
  const int js = indcs.js - gj, je = indcs.je + gj;
  const int ks = indcs.ks - gk, ke = indcs.ke + gk;
  auto &adm = pmy_pack->padm->adm;
  auto dps = delta_psi;
  auto psi0 = u_psi0;
  auto bg = kbg_;
  par_for("cfc_kerr_assemble_metric", DevExeSpace(), 0, pmy_pack->nmb_thispack-1, ks, ke,
          js, je, is, ie, KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real psi = dps(m, 0, k, j, i) + psi0(m, 0, k, j, i);
    const Real p4 = psi*psi*psi*psi;
    adm.psi4(m, k, j, i) = p4;
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        adm.g_dd(m, a, b, k, j, i) = p4*bg(m, kKbgGt + KSym(a, b), k, j, i);
      }
    }
  });
}

//! K_ij = psi^-2 gt_ia gt_jb Ahat^ab, alpha = chi/psi (alpha_floor), beta' = beta0 + b +
//! xidot; interior cells, as AssembleLapseShiftK
void CFC::KerrAssembleADM(Real time) {
  Real xd[3] = {0.0, 0.0, 0.0};
  if (frame_enabled_) { FrameVel(time, xd); }
  const Real x0 = xd[0], x1 = xd[1], x2 = xd[2];
  const Real afl = alpha_floor_;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  auto &adm = pmy_pack->padm->adm;
  auto dps = delta_psi;
  auto dap = delta_alpha_psi;
  auto psi0 = u_psi0;
  auto ap0 = u_alpha0_psi0;
  auto ad = a_dd;
  auto bu = beta_u;
  auto b0 = beta0_u;
  auto bg = kbg_;
  par_for("cfc_kerr_assemble_adm", DevExeSpace(), 0, pmy_pack->nmb_thispack-1, ks, ke, js,
          je, is, ie, KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real psi = dps(m, 0, k, j, i) + psi0(m, 0, k, j, i);
    const Real ip2 = 1.0/(psi*psi);
    Real gd[3][3], ah[3][3];
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        gd[a][b] = bg(m, kKbgGt + KSym(a, b), k, j, i);
        ah[a][b] = ad(m, a, b, k, j, i);
      }
    }
    for (int p = 0; p < 3; ++p) {
      for (int q = p; q < 3; ++q) {
        Real lo = 0.0;
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) { lo += gd[p][a]*gd[q][b]*ah[a][b]; }
        }
        adm.vK_dd(m, p, q, k, j, i) = lo*ip2;
      }
    }
    const Real al = (dap(m, 0, k, j, i) + ap0(m, 0, k, j, i))/psi;
    adm.alpha(m, k, j, i) = (al < afl) ? afl : al;
    adm.beta_u(m, 0, k, j, i) = bu(m, 0, k, j, i) + b0(m, 0, k, j, i) + x0;
    adm.beta_u(m, 1, k, j, i) = bu(m, 1, k, j, i) + b0(m, 1, k, j, i) + x1;
    adm.beta_u(m, 2, k, j, i) = bu(m, 2, k, j, i) + b0(m, 2, k, j, i) + x2;
  });
  for (int a = 0; a < 3; ++a) { pmy_pack->padm->frame_vel_u[a] = xd[a]; }
  frame_assembled_ = true;
  KerrLogCorrections();
}

//! <cfc> kerr_log = n: once per n-th cycle, max over interior cells of the corrections
//! |delta psi/psi0|, |delta chi/chi0| and |beta_res| (V2: zero to round-off without a star)
void CFC::KerrLogCorrections() {
  const int nc = pmy_pack->pmesh->ncycle;
  if (kerr_log_ <= 0 || nc % kerr_log_ != 0 || nc == kerr_corr_cycle_) return;
  kerr_corr_cycle_ = nc;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nmb = pmy_pack->nmb_thispack;
  auto dps = delta_psi;
  auto dap = delta_alpha_psi;
  auto psi0 = u_psi0;
  auto ap0 = u_alpha0_psi0;
  auto bu = beta_u;
  Real mx[3] = {0.0, 0.0, 0.0};
  Kokkos::parallel_reduce("cfc_kerr_corr",
    Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>({0, ks, js, is},
                                       {nmb, indcs.ke + 1, indcs.je + 1, indcs.ie + 1}),
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, Real &a, Real &b,
                  Real &c) {
      a = Kokkos::fmax(a, Kokkos::fabs(dps(m, 0, k, j, i)/psi0(m, 0, k, j, i)));
      b = Kokkos::fmax(b, Kokkos::fabs(dap(m, 0, k, j, i)/ap0(m, 0, k, j, i)));
      const Real b2 = bu(m, 0, k, j, i)*bu(m, 0, k, j, i) + bu(m, 1, k, j, i)*bu(m, 1, k, j, i)
                    + bu(m, 2, k, j, i)*bu(m, 2, k, j, i);
      c = Kokkos::fmax(c, Kokkos::sqrt(b2));
    }, Kokkos::Max<Real>(mx[0]), Kokkos::Max<Real>(mx[1]), Kokkos::Max<Real>(mx[2]));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, mx, 3, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  if (global_variable::my_rank == 0) {
    std::printf("KERR CORR cycle= %d time= %.10e max|dpsi/psi0|= %.6e max|dchi/chi0|= %.6e "
                "max|beta_res|= %.6e\n", nc, pmy_pack->pmesh->time, mx[0], mx[1], mx[2]);
  }
}

//! FillEmbedADM's Kerr form: psi = psi0, g = psi0^4 gt, alpha = alpha0, beta = beta0 +
//! xidot, K = K0 (kbg_ refilled first: the blocks are new)
void CFC::KerrFillEmbedADM(const DvceArray1D<int> &newmb, Real time) {
  Real xbh[3];
  BHPosition(time, xbh);
  KerrRefill(xbh);
  Real xd[3] = {0.0, 0.0, 0.0};
  if (frame_enabled_) { FrameVel(time, xd); }
  const Real x0 = xd[0], x1 = xd[1], x2 = xd[2];
  const Real afl = alpha_floor_;
  auto &adm = pmy_pack->padm->adm;
  auto bg = kbg_;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*indcs.ng) : 1;
  const int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*indcs.ng) : 1;
  par_for("cfc_kerr_embed_adm", DevExeSpace(), 0, pmy_pack->nmb_thispack-1, 0, n3-1, 0,
          n2-1, 0, n1-1, KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    if (newmb(m) == 0) return;
    const Real p = bg(m, kKbgPsi0, k, j, i);
    const Real p4 = p*p*p*p;
    adm.psi4(m, k, j, i) = p4;
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        const int q = KSym(a, b);
        adm.g_dd(m, a, b, k, j, i) = p4*bg(m, kKbgGt + q, k, j, i);
        adm.vK_dd(m, a, b, k, j, i) = bg(m, kKbgK0 + q, k, j, i);
      }
    }
    const Real al = bg(m, kKbgAlpha0, k, j, i);
    adm.alpha(m, k, j, i) = (al < afl) ? afl : al;
    adm.beta_u(m, 0, k, j, i) = bg(m, kKbgBeta0 + 0, k, j, i) + x0;
    adm.beta_u(m, 1, k, j, i) = bg(m, kKbgBeta0 + 1, k, j, i) + x1;
    adm.beta_u(m, 2, k, j, i) = bg(m, kKbgBeta0 + 2, k, j, i) + x2;
  });
  for (int a = 0; a < 3; ++a) { pmy_pack->padm->frame_vel_u[a] = xd[a]; }
}

}  // namespace cfc
