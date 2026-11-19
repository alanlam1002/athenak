//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file kerr_shibata.cpp
//! \brief K-019: Option C (kerr_shibata.hpp).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "driver/driver.hpp"
#include "kerr_background.hpp"
#include "kerr_mg_ops.hpp"
#include "kerr_mg_vops.hpp"
#include "kerr_shibata.hpp"

namespace cfc {

namespace {

using kerr_impl::Sym;
using kerr_impl::IdxI;
using kerr_impl::IdxJ;

[[noreturn]] void ShibFatal(const std::string &msg) {
  std::cout << "### FATAL ERROR in KerrShibata: " << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

//! the centre channels (T, R, DG, gt, gtu) at x from PointMetric2 (every CoefSource mode)
KOKKOS_INLINE_FUNCTION
void CentreGeo(const kmg::CoefSource &cs, Real x, Real y, Real z, Real T[3][3],
               Real R[3][3], Real DG[3], Real gt[6], Real gtu[6]) {
  Real dgt[3][6], ddgt[6][6];
  kmg::PointMetric2(cs, x, y, z, gt, gtu, dgt, ddgt);
  Real gu[3][3];
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) { gu[a][b] = gtu[Sym(a, b)]; }
  }
  // Gamma_{l ij}, Gt^k_ij
  Real Gl[3][6], G[3][6];
  for (int l = 0; l < 3; ++l) {
    for (int q = 0; q < 6; ++q) {
      const int i = IdxI(q), j = IdxJ(q);
      Gl[l][q] = 0.5*(dgt[i][Sym(l, j)] + dgt[j][Sym(i, l)] - dgt[l][q]);
    }
  }
  for (int k = 0; k < 3; ++k) {
    for (int q = 0; q < 6; ++q) {
      G[k][q] = gu[k][0]*Gl[0][q] + gu[k][1]*Gl[1][q] + gu[k][2]*Gl[2][q];
    }
  }
  // d_m gt^kl and d_m Gt^k_ij
  Real dgu[3][3][3];
  for (int m = 0; m < 3; ++m) {
    for (int k = 0; k < 3; ++k) {
      for (int l = 0; l < 3; ++l) {
        Real a = 0.0;
        for (int aa = 0; aa < 3; ++aa) {
          for (int bb = 0; bb < 3; ++bb) { a -= gu[k][aa]*dgt[m][Sym(aa, bb)]*gu[bb][l]; }
        }
        dgu[m][k][l] = a;
      }
    }
  }
  Real dG[3][3][6];   // [m][k][q]
  for (int m = 0; m < 3; ++m) {
    for (int q = 0; q < 6; ++q) {
      const int i = IdxI(q), j = IdxJ(q);
      Real dGl[3];
      for (int l = 0; l < 3; ++l) {
        dGl[l] = 0.5*(ddgt[Sym(m, i)][Sym(l, j)] + ddgt[Sym(m, j)][Sym(i, l)]
                      - ddgt[Sym(m, l)][q]);
      }
      for (int k = 0; k < 3; ++k) {
        Real a = 0.0;
        for (int l = 0; l < 3; ++l) { a += dgu[m][k][l]*Gl[l][q] + gu[k][l]*dGl[l]; }
        dG[m][k][q] = a;
      }
    }
  }
  // Rt_ij = d_k Gt^k_ij - d_j Gt^k_ik + Gt^k_kl Gt^l_ij - Gt^k_jl Gt^l_ik
  Real rt[3][3];
  for (int q = 0; q < 6; ++q) {
    const int i = IdxI(q), j = IdxJ(q);
    Real a = 0.0;
    for (int k = 0; k < 3; ++k) {
      a += dG[k][k][q] - dG[j][k][Sym(i, k)];
      for (int l = 0; l < 3; ++l) {
        a += G[k][Sym(k, l)]*G[l][q] - G[k][Sym(j, l)]*G[l][Sym(i, k)];
      }
    }
    rt[i][j] = a;
    rt[j][i] = a;
  }
  for (int k = 0; k < 3; ++k) {
    for (int l = 0; l < 3; ++l) {
      R[k][l] = rt[k][0]*gu[0][l] + rt[k][1]*gu[1][l] + rt[k][2]*gu[2][l];
      // T_k^l = gt^ab d_k Gt^l_ab (packed: off-diagonal twice)
      Real a = 0.0;
      for (int q = 0; q < 6; ++q) {
        const Real w = (IdxI(q) == IdxJ(q)) ? 1.0 : 2.0;
        a += w*gtu[q]*dG[k][l][q];
      }
      T[k][l] = a;
    }
  }
  for (int l = 0; l < 3; ++l) { DG[l] = dgu[0][0][l] + dgu[1][1][l] + dgu[2][2][l]; }
}

}  // namespace

//----------------------------------------------------------------------------------------
// KerrShibCommon

KerrShibCommon::KerrShibCommon(MeshBlockPack *pmbp_, ParameterInput *pin,
                               const std::string &block, const kmg::CoefSource &cs)
    : pmbp(pmbp_), cs_now(cs) {
  kmg::Manuf mf;
  mf.amp = 0.0;   // linear kind: s = 0 (and c = mf.c = 0), the source is the loaded src
  mf.c = 0.0;
  // the scalar driver reads its MG options from <block>; its kind is linear whatever the
  // block says, and a vector-only smoother name falls back to gs8
  if (pin->DoesParameterExist(block, "smoother") &&
      pin->GetString(block, "smoother") != "gs8" && pin->GetString(block, "smoother") != "rb") {
    ShibFatal("<" + block + "> smoother must be gs8 or rb with vec_solver = shibata.");
  }
  if (pin->DoesParameterExist(block, "kind") && pin->GetString(block, "kind") != "linear") {
    ShibFatal("<" + block + "> kind must be linear (or absent) with vec_solver = shibata.");
  }
  sc = new KerrShibScalar(pmbp, pin, block, cs, mf);
  sc->SetLogName("kshib");
  const int nmb = std::max(pmbp->nmb_thispack, pmbp->pmesh->nmb_maxperrank);
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*indcs.ng) : 1;
  const int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*indcs.ng) : 1;
  Kokkos::realloc(geo, nmb, kshib::kN, n3, n2, n1);
}

KerrShibCommon::~KerrShibCommon() { delete sc; }

void KerrShibCommon::Update(const kmg::CoefSource &cs) {
  Mesh *pm = pmbp->pmesh;
  if (kmg::KeyMatches(key_, cs, pm, pmbp, 0, nullptr) &&
      geo.extent_int(0) >= pmbp->nmb_thispack) {
    ++reuses;
    return;
  }
  cs_now = cs;
  sc->cs_ = cs;
  sc->Prepare();
  const int nmb = std::max(pmbp->nmb_thispack, pm->nmb_maxperrank);
  auto &indcs = pm->mb_indcs;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*indcs.ng) : 1;
  const int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*indcs.ng) : 1;
  if (geo.extent_int(0) < nmb) { Kokkos::realloc(geo, nmb, kshib::kN, n3, n2, n1); }
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  auto &size = pmbp->pmb->mb_size;
  auto g = geo;
  const kmg::CoefSource c = cs;
  par_for("kshib_geo", DevExeSpace(), 0, pmbp->nmb_thispack-1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real dx[3] = {size.d_view(m).dx1, size.d_view(m).dx2, size.d_view(m).dx3};
    const Real xc[3] = {size.d_view(m).x1min + (i - is + 0.5)*dx[0],
                        size.d_view(m).x2min + (j - js + 0.5)*dx[1],
                        size.d_view(m).x3min + (k - ks + 0.5)*dx[2]};
    Real T[3][3], R[3][3], DG[3], gt[6], gtu[6];
    CentreGeo(c, xc[0], xc[1], xc[2], T, R, DG, gt, gtu);
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        g(m, kshib::kT + 3*a + b, k, j, i) = T[a][b];
        g(m, kshib::kR + 3*a + b, k, j, i) = R[a][b];
      }
      g(m, kshib::kDG + a, k, j, i) = DG[a];
    }
    for (int q = 0; q < 6; ++q) {
      g(m, kshib::kGd + q, k, j, i) = gt[q];
      g(m, kshib::kGu + q, k, j, i) = gtu[q];
    }
    for (int d = 0; d < 3; ++d) {
      Real xf[3] = {xc[0], xc[1], xc[2]};
      xf[d] -= 0.5*dx[d];
      Real fgt[6], fgtu[6], fdgt[3][6], gam[3][6];
      kmg::PointMetric(c, xf[0], xf[1], xf[2], fgt, fgtu, fdgt);
      kmg::ChristoffelT(fgtu, fdgt, gam);
      for (int kk = 0; kk < 3; ++kk) {
        for (int l = 0; l < 3; ++l) {
          Real a = 0.0;
          for (int jj = 0; jj < 3; ++jj) { a += fgtu[Sym(d, jj)]*gam[l][Sym(jj, kk)]; }
          g(m, kshib::kCF + 9*d + 3*kk + l, k, j, i) = a;
        }
      }
    }
  });
  Kokkos::fence();
  kmg::CaptureKey(key_, cs, pm, pmbp, 0, nullptr);
  ++fills;
}

//----------------------------------------------------------------------------------------
// KerrShibataSolver

KerrShibataSolver::KerrShibataSolver(MeshBlockPack *pmbp, ParameterInput *pin,
                                     const std::string &block, KerrShibCommon *com,
                                     const std::string &name)
    : pmbp_(pmbp), com_(com), name_(name) {
  const Real eps = pin->DoesParameterExist(block, "mg_threshold") ?
                   pin->GetReal(block, "mg_threshold") : com->sc->Eps();
  tol_ = pin->DoesParameterExist(block, "shib_tol") ? pin->GetReal(block, "shib_tol") : eps;
  max_pass_ = pin->DoesParameterExist(block, "shib_max_pass") ?
              pin->GetInteger(block, "shib_max_pass") : 10;
  log_ = pin->DoesParameterExist(block, "shib_log") ? pin->GetInteger(block, "shib_log") : 1;
  if (max_pass_ < 1) ShibFatal("<" + block + "> shib_max_pass must be >= 1.");
  EnsureState();
}

void KerrShibataSolver::EnsureState() {
  Mesh *pm = pmbp_->pmesh;
  const int seq = pm->GetAMRLoadBalanceUpdateSeq();
  const int nmb = std::max(pmbp_->nmb_thispack, pm->nmb_maxperrank);
  if (seq == pe_seq_ && pmbp_->nmb_thispack == pe_nmb_ && pe_.extent_int(0) >= nmb) return;
  auto &indcs = pm->mb_indcs;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*indcs.ng) : 1;
  const int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*indcs.ng) : 1;
  // a new mesh (start, regrid, load balance): cold start from zero
  Kokkos::realloc(pe_, nmb, 4, n3, n2, n1);
  Kokkos::realloc(pnew_, nmb, 3, n3, n2, n1);
  Kokkos::realloc(sk_, nmb, 3, n3, n2, n1);
  Kokkos::realloc(rhs_, nmb, 1, n3, n2, n1);
  Kokkos::realloc(vold_, nmb, 3, n3, n2, n1);
  Kokkos::realloc(vnew_, nmb, 3, n3, n2, n1);
  Kokkos::deep_copy(pe_, 0.0);
  Kokkos::deep_copy(pnew_, 0.0);
  Kokkos::deep_copy(rhs_, 0.0);
  pe_seq_ = seq;
  pe_nmb_ = pmbp_->nmb_thispack;
}

void KerrShibataSolver::SaveState(DvceArray5D<Real> &dst, int ns, int nmb) const {
  auto all = Kokkos::ALL;
  Kokkos::deep_copy(Kokkos::subview(dst, all, std::make_pair(ns, ns + 4), all, all, all),
                    Kokkos::subview(pe_, std::make_pair(0, nmb), all, all, all, all));
}

void KerrShibataSolver::LoadState(const DvceArray5D<Real> &src, int ns, int nmb) {
  EnsureState();
  auto all = Kokkos::ALL;
  Kokkos::deep_copy(Kokkos::subview(pe_, std::make_pair(0, nmb), all, all, all, all),
                    Kokkos::subview(src, all, std::make_pair(ns, ns + 4), all, all, all));
}

//! V^i = gt^ij [P_j - (1/8) d_j Phi] (interior) from pe_; returns 0
Real KerrShibataSolver::ComputeV(const DvceArray5D<Real> &v) {
  auto &indcs = pmbp_->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  auto &size = pmbp_->pmb->mb_size;
  auto pe = pe_;
  auto g = com_->geo;
  const Real xb0 = com_->cs_now.xbh[0], xb1 = com_->cs_now.xbh[1];
  const Real xb2 = com_->cs_now.xbh[2];
  par_for("kshib_v", DevExeSpace(), 0, pmbp_->nmb_thispack-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real dx[3] = {size.d_view(m).dx1, size.d_view(m).dx2, size.d_view(m).dx3};
    const Real x0 = size.d_view(m).x1min - xb0, y0 = size.d_view(m).x2min - xb1;
    const Real z0 = size.d_view(m).x3min - xb2;
    auto phi = [&](int kk, int jj, int ii) {
      const Real x = x0 + (ii - is + 0.5)*dx[0], y = y0 + (jj - js + 0.5)*dx[1];
      const Real z = z0 + (kk - ks + 0.5)*dx[2];
      return pe(m, 3, kk, jj, ii) + x*pe(m, 0, kk, jj, ii) + y*pe(m, 1, kk, jj, ii)
             + z*pe(m, 2, kk, jj, ii);
    };
    Real vl[3];
    vl[0] = pe(m, 0, k, j, i) - 0.125*(phi(k, j, i+1) - phi(k, j, i-1))/(2.0*dx[0]);
    vl[1] = pe(m, 1, k, j, i) - 0.125*(phi(k, j+1, i) - phi(k, j-1, i))/(2.0*dx[1]);
    vl[2] = pe(m, 2, k, j, i) - 0.125*(phi(k+1, j, i) - phi(k-1, j, i))/(2.0*dx[2]);
    for (int a = 0; a < 3; ++a) {
      Real s = 0.0;
      for (int b = 0; b < 3; ++b) { s += g(m, kshib::kGu + Sym(a, b), k, j, i)*vl[b]; }
      v(m, a, k, j, i) = s;
    }
  });
  return 0.0;
}

void KerrShibataSolver::Solve(Driver *pdriver, int stage, const kmg::CoefSource &cs,
                              const DvceArray5D<Real> &src, int src_ngh,
                              DvceArray5D<Real> &dst) {
  com_->Update(cs);
  EnsureState();
  KerrShibScalar *sc = com_->sc;
  auto &indcs = pmbp_->pmesh->mb_indcs;
  const int ng = indcs.ng;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pmbp_->nmb_thispack;
  auto &size = pmbp_->pmb->mb_size;
  auto g = com_->geo;
  auto pe = pe_;
  auto pn = pnew_;
  auto sk = sk_;
  auto rhs = rhs_;
  const Real xb0 = com_->cs_now.xbh[0], xb1 = com_->cs_now.xbh[1];
  const Real xb2 = com_->cs_now.xbh[2];
  // s_k = gt_kl s^l
  par_for("kshib_sk", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const int mk = k - ks + src_ngh, mj = j - js + src_ngh, mi = i - is + src_ngh;
    for (int a = 0; a < 3; ++a) {
      Real s = 0.0;
      for (int b = 0; b < 3; ++b) {
        s += g(m, kshib::kGd + Sym(a, b), k, j, i)*src(m, b, mk, mj, mi);
      }
      sk(m, a, k, j, i) = s;
    }
  });
  ComputeV(vold_);
  Real dv_prev = -1.0, dv = 0.0;
  int pass = 0;
  int it[4] = {0, 0, 0, 0};
  Real rms[4] = {0.0, 0.0, 0.0, 0.0};
  static const char *cname[4] = {"P1", "P2", "P3", "eta"};
  for (pass = 1; pass <= max_pass_; ++pass) {
    // ---- the four scalar solves (c = 0: P_0..2 from the previous pass, then eta)
    for (int c = 0; c < 4; ++c) {
      par_for("kshib_rhs", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        const Real dx[3] = {size.d_view(m).dx1, size.d_view(m).dx2, size.d_view(m).dx3};
        const Real x0 = size.d_view(m).x1min - xb0, y0 = size.d_view(m).x2min - xb1;
        const Real z0 = size.d_view(m).x3min - xb2;
        // P_l: the previous pass for the P solves, this pass's for eta (pe holds it then)
        auto P = [&](int l, int kk, int jj, int ii) { return pe(m, l, kk, jj, ii); };
        auto phi = [&](int kk, int jj, int ii) {
          const Real x = x0 + (ii - is + 0.5)*dx[0], y = y0 + (jj - js + 0.5)*dx[1];
          const Real z = z0 + (kk - ks + 0.5)*dx[2];
          return pe(m, 3, kk, jj, ii) + x*P(0, kk, jj, ii) + y*P(1, kk, jj, ii)
                 + z*P(2, kk, jj, ii);
        };
        const Real xc[3] = {x0 + (i - is + 0.5)*dx[0], y0 + (j - js + 0.5)*dx[1],
                            z0 + (k - ks + 0.5)*dx[2]};
        Real dphi[3];
        dphi[0] = (phi(k, j, i+1) - phi(k, j, i-1))/(2.0*dx[0]);
        dphi[1] = (phi(k, j+1, i) - phi(k, j-1, i))/(2.0*dx[1]);
        dphi[2] = (phi(k+1, j, i) - phi(k-1, j, i))/(2.0*dx[2]);
        // the face flux of cell (kk, jj, ii)'s lower d-face, weighted by w[kq]
        // (P solve: w = e_c; eta: w = x_face)
        auto flux = [&](int d, int kk, int jj, int ii) {
          const int km = kk - (d == 2), jm = jj - (d == 1), im = ii - (d == 0);
          Real w[3];
          if (c < 3) {
            w[0] = (c == 0); w[1] = (c == 1); w[2] = (c == 2);
          } else {
            w[0] = x0 + (ii - is + 0.5)*dx[0] - ((d == 0) ? 0.5*dx[0] : 0.0);
            w[1] = y0 + (jj - js + 0.5)*dx[1] - ((d == 1) ? 0.5*dx[1] : 0.0);
            w[2] = z0 + (kk - ks + 0.5)*dx[2] - ((d == 2) ? 0.5*dx[2] : 0.0);
          }
          Real f = 0.0;
          for (int l = 0; l < 3; ++l) {
            const Real pf = 0.5*(P(l, kk, jj, ii) + P(l, km, jm, im));
            Real cw = 0.0;
            for (int q = 0; q < 3; ++q) {
              cw += w[q]*g(m, kshib::kCF + 9*d + 3*q + l, kk, jj, ii);
            }
            f += cw*pf;
          }
          return f;
        };
        const Real div = (flux(0, k, j, i+1) - flux(0, k, j, i))/dx[0]
                       + (flux(1, k, j+1, i) - flux(1, k, j, i))/dx[1]
                       + (flux(2, k+1, j, i) - flux(2, k, j, i))/dx[2];
        Real r;
        if (c < 3) {
          // Delta_s P_c = s_c + 2 div(gt Gt P) - T_c^l P_l + 1/4 R_c^l d_l Phi
          Real tr = 0.0, rr = 0.0;
          for (int l = 0; l < 3; ++l) {
            tr += g(m, kshib::kT + 3*c + l, k, j, i)*P(l, k, j, i);
            rr += g(m, kshib::kR + 3*c + l, k, j, i)*dphi[l];
          }
          r = sk(m, c, k, j, i) + 2.0*div - tr + 0.25*rr;
        } else {
          // Delta_s eta = -x^k s_k - 2 div(x gt Gt P) - DG_l P_l + x^k T_k^l P_l
          //               - 1/4 x^k R_k^l d_l Phi
          Real xs = 0.0, dg = 0.0, xt = 0.0, xr = 0.0;
          for (int kk = 0; kk < 3; ++kk) {
            xs += xc[kk]*sk(m, kk, k, j, i);
            dg += g(m, kshib::kDG + kk, k, j, i)*P(kk, k, j, i);
            for (int l = 0; l < 3; ++l) {
              xt += xc[kk]*g(m, kshib::kT + 3*kk + l, k, j, i)*P(l, k, j, i);
              xr += xc[kk]*g(m, kshib::kR + 3*kk + l, k, j, i)*dphi[l];
            }
          }
          r = -xs - 2.0*div - dg + xt - 0.25*xr;
        }
        rhs(m, 0, k, j, i) = -r;   // the scalar driver solves -Delta_s u = src
      });
      sc->Levels()->LoadFinestData(pe_, c, ng);
      sc->Levels()->LoadSource(rhs_, 0, ng, 1.0);
      sc->SetLogName(name_ + "." + cname[c]);
      sc->SolveLoaded(pdriver);
      it[c] = sc->LastNVcycle();
      rms[c] = sc->LastDef();
      if (c < 3) {
        // Jacobi: this pass's P_c into pnew_ (interior + the MG ring); pe keeps the
        // previous pass's P until all three are solved
        sc->Levels()->RetrieveResult(pnew_, c, ng);
        if (c == 2) {
          par_for("kshib_newp", DevExeSpace(), 0, nmb-1, 0, pe.extent_int(2)-1, 0,
                  pe.extent_int(3)-1, 0, pe.extent_int(4)-1,
          KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
            for (int a = 0; a < 3; ++a) { pe(m, a, k, j, i) = pn(m, a, k, j, i); }
          });
        }
      } else {
        sc->Levels()->RetrieveResult(pe_, 3, ng);
      }
    }
    // ---- convergence: max |V_new - V_old|
    ComputeV(vnew_);
    auto vn = vnew_;
    auto vo = vold_;
    Real mx = 0.0;
    Kokkos::parallel_reduce("kshib_dv",
      Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>({0, ks, js, is},
                                                         {nmb, ke + 1, je + 1, ie + 1}),
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, Real &a) {
        for (int c = 0; c < 3; ++c) {
          a = Kokkos::fmax(a, Kokkos::fabs(vn(m, c, k, j, i) - vo(m, c, k, j, i)));
        }
      }, Kokkos::Max<Real>(mx));
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &mx, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
    dv = mx;
    if (log_ >= 2 && global_variable::my_rank == 0) {
      std::printf("KSHIB %s cyc %d t %.10e stage %d pass %d dV %.4e contraction %.4e "
                  "it %d %d %d %d def %.4e %.4e %.4e %.4e\n", name_.c_str(),
                  pmbp_->pmesh->ncycle, pmbp_->pmesh->time, stage, pass, dv,
                  (dv_prev > 0.0) ? dv/dv_prev : 0.0, it[0], it[1], it[2], it[3], rms[0],
                  rms[1], rms[2], rms[3]);
    }
    Kokkos::deep_copy(vold_, vnew_);
    if (dv <= tol_) break;
    dv_prev = dv;
  }
  if (pass > max_pass_) pass = max_pass_;
  last_pass_ = pass;
  last_dv_ = dv;
  if (log_ == 1 && global_variable::my_rank == 0) {
    std::printf("KSHIB %s cyc %d t %.10e stage %d passes %d dV %.4e it %d %d %d %d def "
                "%.4e %.4e %.4e %.4e\n", name_.c_str(), pmbp_->pmesh->ncycle,
                pmbp_->pmesh->time, stage, pass, dv, it[0], it[1], it[2], it[3], rms[0],
                rms[1], rms[2], rms[3]);
  }
  // V^i into dst (interior; the task list's exchange fills the ghosts as for the direct
  // driver)
  auto vf = vold_;
  par_for("kshib_out", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    for (int a = 0; a < 3; ++a) { dst(m, a, k, j, i) = vf(m, a, k, j, i); }
  });
  Kokkos::fence();
}

}  // namespace cfc
