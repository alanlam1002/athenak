//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_spinning_puncture.cpp
//! \brief A single spinning puncture: a maximal Kerr slice through the Einstein-Rosen
//! bridge, ported from SACRA (sacra-aei/mod_sbh.f90, input.f90) for M2f / T4 (chat/0011).
//! Two radial coordinates (problem/spin_coord):
//!   les: Liu-Etienne-Shapiro, r_BL = r (1 + r_+/4r)^2, horizon at r_h = r_+/4 =
//!        (M + sqrt(M^2 - a^2))/4 (D-019, 2502.03223 Eqs. 68-70); SACRA sbh_get_var /
//!        input_sbh (alpha = W, beta = 0).
//!   qi:  Brandt-Seidel quasi-isotropic, r_BL = r (1 + (M+a)/2r)(1 + (M-a)/2r), horizon
//!        at r_h = sqrt(M^2 - a^2)/2; SACRA sbh_get_var_quasi / input_sbh_quasi
//!        (alpha = W, beta = Kerr shift).
//! Both are K = 0 with only K_{R phi}, K_{z phi} non-zero in the cylindrical frame,
//! and the horizon is the coordinate sphere r = r_h.
//!
//! The spin is along PHYSICAL +z. On a 3D Cartesian mesh physical = mesh. Under 2D
//! Cartoon (D-017) physical (x, y, z) = mesh (x1, -x3, x2): the spin is along mesh +x2,
//! the symmetry axis, and the centre must lie on that axis.
//!
//! <problem> keys: punc_ADM_mass (M), punc_chi (a/M, |chi| < 1), punc_center_x1..x3
//! (mesh coordinates), spin_coord = les | qi, spin_lapse = w | kerr (w: alpha = W =
//! gamma^{-1/6}, SACRA's choice; kerr: the Kerr lapse), spin_shift = zero | kerr | sacra
//! (sacra: as SACRA's input routine for that coordinate). The defaults les, w, zero are
//! the T4 initial data (D-023). ahf_at_init (run the
//! <fastflow> finders once on the initial data, for unit tests with nlim = 0).

#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "z4c/fastflow.hpp"
#include "coordinates/adm.hpp"
#include "z4c/z4c_pos.hpp"
#include "pgen/pgen.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {

//----------------------------------------------------------------------------------------
// SACRA sbh_get_var_quasi at cylindrical (R >= 0, z) (only the outputs used here):
//   wk = psi^-2, qk = e^{2q}, qk1 = e^{2q} - 1, ek1 / ek2 = W^2 e^{2q/3} K_{R phi-hat},
//   K_{z phi-hat} (conformal, see below), alpk = Kerr lapse, betk = beta^{phi-hat}.
// Arguments as in SACRA: m = mass, a = spin parameter (a = chi m).
KOKKOS_INLINE_FUNCTION
void KerrQuasi(const Real x, const Real z, const Real m, const Real a, Real &wk, Real &qk,
               Real &qk1, Real &ek1, Real &ek2, Real &alpk, Real &betk) {
  const Real aa = a/m;
  const Real rs = 0.5*Kokkos::sqrt(1.0 - aa*aa);
  const Real a2 = aa*aa/(rs*rs);
  const Real xx = x/m/rs, zz = z/m/rs;
  const Real rr = Kokkos::sqrt(xx*xx + zz*zz);
  Real rk;   // rk := 1/r_BL in units of m*rs
  if (rr >= 1.0) {
    rk = 1.0/(rr + 1.0/rr + 1.0/rs);
  } else {
    rk = rr/(1.0 + rr/rs + rr*rr);
  }
  Real sint, cost;
  if (rr > 0.0) {
    sint = xx/rr; cost = zz/rr;
  } else {
    sint = 1.0; cost = 0.0;
  }
  const Real sint2 = sint*sint, cost2 = cost*cost;
  const Real rk2 = rk*rk;
  const Real sigk = 1.0 + a2*rk2*cost2;
  const Real xik = (1.0 + a2*rk2)*sigk + 2.0*a2*rk2*rk*sint2/rs;
  qk = sigk*sigk/xik;
  qk1 = -a2*rk2*sint2*(sigk + 2.0*rk/rs)/xik;
  const Real he = ((1.0 - a2*rk2)*sigk + 2.0*(1.0 + a2*rk2))*sint/(sigk*sigk);
  const Real hf = -2.0*a2*rk2*Kokkos::sqrt(1.0 - 2.0*rk/rs + a2*rk2)*sint2*cost
                  /(sigk*sigk);
  wk = rr*rk*Kokkos::sqrt(sigk/xik);
  const Real sx3 = Kokkos::sqrt(sigk/xik)*sigk/xik;   // (sigk/xik)^{3/2}
  ek1 = a*(he*sint + hf*cost)*rk2*rk*sx3/(rs*rs*rs)/(m*m);
  ek2 = a*(he*cost - hf*sint)*rk2*rk*sx3/(rs*rs*rs)/(m*m);
  alpk = Kokkos::sqrt(sigk*Kokkos::sqrt(1.0 - 2.0*rk/rs + a2*rk2)/xik);
  betk = -2.0*a*rk2*rk/xik/(rs*rs)*sint;
}

//----------------------------------------------------------------------------------------
// SACRA sbh_get_var (Liu-Etienne-Shapiro radial coordinate) at physical (x, y, z)
// relative to the centre, mass m, chi = a/m. Returns W (gamma = (delta + h)/W^2), the
// conformal h_ij and A-tilde_ij (= W^2 K_ij, K = 0) in the order xx, xy, xz, yy, yz, zz,
// the Kerr lapse and beta^{phi-hat}.
KOKKOS_INLINE_FUNCTION
void KerrLES(const Real x, const Real y, const Real z, const Real m, const Real chi,
             Real &W, Real h[6], Real at[6], Real &alp, Real &bet) {
  const Real a2 = Kokkos::fmin(1.0, chi*chi);
  const Real sq = Kokkos::sqrt(1.0 - a2);
  const Real rp = 1.0 + sq, rm = 1.0 - sq, rh = 0.25*rp;
  const Real r1 = x/m, r2 = y/m, r3 = z/m;
  const Real rs = Kokkos::sqrt(r1*r1 + r2*r2 + r3*r3);
  const Real rcyl = Kokkos::sqrt(r1*r1 + r2*r2);
  Real ct, st, cp, sp;
  if (rs > 0.0) { ct = r3/rs; st = rcyl/rs; } else { ct = 0.0; st = 1.0; }
  if (rcyl > 0.0) { cp = r1/rcyl; sp = r2/rcyl; } else { cp = 1.0; sp = 0.0; }
  const Real ct2 = ct*ct, st2 = st*st, c2p = cp*cp - sp*sp, s2p = 2.0*sp*cp;
  const Real rrs = rs/rh;
  const Real irBL = (rrs <= 1.0) ? rrs/((1.0 + rrs)*(1.0 + rrs))/rh
                                 : 1.0/((1.0 + 1.0/rrs)*(1.0 + 1.0/rrs))/rrs/rh;
  const Real ia = chi*irBL, ia2 = a2*irBL*irBL, irm = rm*irBL, rbr = 1.0 - irm;
  const Real sig = 1.0 + ia2*ct2;
  const Real del = 1.0 - 2.0*irBL + ia2;
  const Real abr = (1.0 + ia2)*(1.0 + ia2) - del*ia2*st2;
  alp = Kokkos::sqrt(Kokkos::fmax(del, 0.0)*sig/abr);
  W = rs*irBL/Kokkos::pow(sig*abr/rbr, 1.0/6.0);
  bet = -2.0*ia/abr*irBL*irBL*rs*st;
  const Real f1 = Kokkos::cbrt(sig*sig/abr/(rbr*rbr));
  const Real ht1 = f1 - 1.0;
  const Real ht2 = Kokkos::cbrt(sig*sig*rbr/abr) - 1.0;
  const Real ht3 = Kokkos::cbrt(abr*abr*rbr/(sig*sig*sig*sig)) - 1.0;
  // local frame (R, phi-hat, z), then the rotation by phi (SACRA order xx yy zz xy xz yz)
  Real txx = st2*ht1 + ct2*ht2, tyy = ht3, tzz = ct2*ht1 + st2*ht2;
  Real txy = 0.0, txz = st*ct*irm*f1, tyz = 0.0;
  h[0] = txx*cp*cp + tyy*sp*sp - s2p*txy;          // xx
  h[3] = txx*sp*sp + tyy*cp*cp + s2p*txy;          // yy
  h[5] = tzz;                                      // zz
  h[1] = txy*c2p + (txx - tyy)*s2p*0.5;            // xy
  h[2] = txz*cp - tyz*sp;                          // xz
  h[4] = txz*sp + tyz*cp;                          // yz
  const Real s116 = Kokkos::pow(sig, 11.0/6.0), a56 = Kokkos::pow(abr, 5.0/6.0);
  const Real arp = ia*st/s116/a56/Kokkos::pow(rbr, 1.0/6.0)
                 *(3.0 + 2.0*ia2 - ia2*ia2 - ia2*(1.0 - ia2)*st2)*irBL*irBL;
  const Real atp = -2.0*ia*ia*ia*ct*st2/s116/a56*Kokkos::pow(rbr, 5.0/6.0)
                 *(1.0 - 0.25*rp/rs)*Kokkos::sqrt(rs)*Kokkos::pow(irBL, 2.5);
  txx = 0.0; tyy = 0.0; tzz = 0.0;
  txy = st*arp + ct*atp; txz = 0.0; tyz = ct*arp - st*atp;
  at[0] = (txx*cp*cp + tyy*sp*sp - s2p*txy)/m;
  at[3] = (txx*sp*sp + tyy*cp*cp + s2p*txy)/m;
  at[5] = tzz/m;
  at[1] = (txy*c2p + (txx - tyy)*s2p*0.5)/m;
  at[2] = (txz*cp - tyz*sp)/m;
  at[4] = (txz*sp + tyz*cp)/m;
}

//----------------------------------------------------------------------------------------
// Physical-frame ADM data at physical position (x, y, z) relative to the centre, spin
// along +z. Symmetric tensors in the order xx, xy, xz, yy, yz, zz.
KOKKOS_INLINE_FUNCTION
void SpinningPuncturePhys(const Real x, const Real y, const Real z, const Real m,
                          const Real a, const bool les, const bool kerr_lapse,
                          const bool kerr_shift, Real g[6], Real K[6], Real &alpha,
                          Real beta[3]) {
  const Real R = Kokkos::sqrt(x*x + y*y);
  if (les) {
    Real W, h[6], at[6], alp, bet;
    KerrLES(x, y, z, m, a/m, W, h, at, alp, bet);
    const Real iW2 = 1.0/(W*W);
    const int dg[6] = {1, 0, 0, 1, 0, 1};
    for (int n = 0; n < 6; ++n) {
      g[n] = (dg[n] + h[n])*iW2;
      K[n] = at[n]*iW2;
    }
    alpha = kerr_lapse ? alp : W;
    const Real c = (R > 0.0) ? x/R : 1.0, s = (R > 0.0) ? y/R : 0.0;
    const Real bp = kerr_shift ? bet : 0.0;
    beta[0] = -bp*s; beta[1] = bp*c; beta[2] = 0.0;
    return;
  }
  Real wk, qk, qk1, ek1, ek2, alpk, betk;
  KerrQuasi(R, z, m, a, wk, qk, qk1, ek1, ek2, alpk, betk);
  const Real q13 = Kokkos::cbrt(qk), q23 = q13*q13;
  const Real W = wk/q13;                       // conformal factor, gamma = gt / W^2
  const Real iW2 = 1.0/(W*W);
  // conformal metric in the cylindrical frame (R, phi-hat, z) on the meridian (SACRA:
  // GXX = GZZ = e^{2q/3} - 1, GYY = e^{-4q/3} - 1), so det = 1
  const Real hRR = qk1/(q23 + q13 + 1.0);
  const Real hPP = -qk1*(q13 + 1.0)/q23/(q23 + q13 + 1.0);
  const Real gRR = (1.0 + hRR)*iW2, gPP = (1.0 + hPP)*iW2, gZZ = (1.0 + hRR)*iW2;
  // K = A (maximal); SACRA's AKXY = ek1/q23 is W^2 K_{R phi}, so K_{R phi} = ek1/wk^2
  const Real iwk2 = 1.0/(wk*wk);
  const Real KRP = ek1*iwk2, KZP = ek2*iwk2;
  // frame vectors e_R = (c, s, 0), e_phi = (-s, c, 0), e_z
  const Real c = (R > 0.0) ? x/R : 1.0, s = (R > 0.0) ? y/R : 0.0;
  // gamma = gRR e_R e_R + gPP e_phi e_phi + gZZ e_z e_z
  g[0] = gRR*c*c + gPP*s*s;
  g[1] = (gRR - gPP)*c*s;
  g[2] = 0.0;
  g[3] = gRR*s*s + gPP*c*c;
  g[4] = 0.0;
  g[5] = gZZ;
  // K = KRP (e_R e_phi + e_phi e_R) + KZP (e_z e_phi + e_phi e_z)
  K[0] = -2.0*KRP*c*s;
  K[1] = KRP*(c*c - s*s);
  K[2] = -KZP*s;
  K[3] = 2.0*KRP*c*s;
  K[4] = KZP*c;
  K[5] = 0.0;
  alpha = kerr_lapse ? alpk : W;
  const Real bp = kerr_shift ? betk : 0.0;   // beta = bp e_phi
  beta[0] = -bp*s;
  beta[1] = bp*c;
  beta[2] = 0.0;
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::Z4cSpinningPuncture()
//! \brief single spinning puncture in quasi-isotropic coordinates

void ProblemGenerator::Z4cSpinningPuncture(ParameterInput *pin, const bool restart) {
  if (restart) return;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;
  if (pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "z4c_spinning_puncture needs a <z4c> block" << std::endl;
    exit(EXIT_FAILURE);
  }
  const Real m0 = pin->GetOrAddReal("problem", "punc_ADM_mass", 1.0);
  const Real chi = pin->GetOrAddReal("problem", "punc_chi", 0.0);
  const Real cx1 = pin->GetOrAddReal("problem", "punc_center_x1", 0.0);
  const Real cx2 = pin->GetOrAddReal("problem", "punc_center_x2", 0.0);
  const Real cx3 = pin->GetOrAddReal("problem", "punc_center_x3", 0.0);
  const std::string coord = pin->GetOrAddString("problem", "spin_coord", "les");
  const std::string lapse = pin->GetOrAddString("problem", "spin_lapse", "w");
  const std::string shift = pin->GetOrAddString("problem", "spin_shift", "zero");
  const bool ahf_at_init = pin->GetOrAddBoolean("problem", "ahf_at_init", false);
  auto fatal = [](const std::string &msg) {
    std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
              << "z4c_spinning_puncture: " << msg << std::endl;
    std::exit(EXIT_FAILURE);
  };
  if (!(std::abs(chi) < 1.0)) fatal("punc_chi must satisfy |chi| < 1");
  if (lapse != "w" && lapse != "kerr") fatal("spin_lapse must be w or kerr");
  if (coord != "les" && coord != "qi") fatal("spin_coord must be les or qi");
  if (shift != "sacra" && shift != "kerr" && shift != "zero") {
    fatal("spin_shift must be sacra, kerr or zero");
  }
  const z4c::Z4c::CartoonMode cmode = pmbp->pz4c->opt.cartoon;
  const bool cartoon = pmbp->pz4c->IsCartoon();
  if (cartoon && (cx1 != 0.0 || cx3 != 0.0)) {
    fatal("under 2D Cartoon the puncture must lie on the symmetry axis (mesh x2): "
          "punc_center_x1 = punc_center_x3 = 0 (D-017)");
  }
  const bool les = (coord == "les");
  const bool kerr_lapse = (lapse == "kerr");
  // SACRA: input_sbh (les) has beta = 0, input_sbh_quasi (qi) the Kerr shift
  const bool kerr_shift = (shift == "kerr") || (shift == "sacra" && !les);
  const Real a = chi*m0;

  auto &size = pmbp->pmb->mb_size;
  const int ng2 = (indcs.nx2 > 1) ? indcs.ng : 0;
  const int ng3 = (indcs.nx3 > 1) ? indcs.ng : 0;
  const int isg = indcs.is - indcs.ng, ieg = indcs.ie + indcs.ng;
  const int jsg = indcs.js - ng2, jeg = indcs.je + ng2;
  const int ksg = indcs.ks - ng3, keg = indcs.ke + ng3;
  const int nmb = pmbp->nmb_thispack;
  adm::ADM::ADM_vars &adm = pmbp->padm->adm;
  const RegionIndcs indcs_ = indcs;
  par_for("pgen spinning puncture", DevExeSpace(), 0, nmb-1, ksg, keg, jsg, jeg, isg, ieg,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // mesh position (D-017 helper, mesh order), relative to the centre
    Real X[3];
    z4c::Z4cCartesianPos(cmode, size.d_view(m), indcs_, k, j, i, X[0], X[1], X[2]);
    X[0] -= cx1; X[1] -= cx2; X[2] -= cx3;
    // physical position: Cartoon (x, y, z) = (X1, -X3, X2); 3D physical = mesh
    const Real xp = X[0];
    const Real yp = cartoon ? -X[2] : X[1];
    const Real zp = cartoon ? X[1] : X[2];
    Real g[6], K[6], alpha, beta[3];
    SpinningPuncturePhys(xp, yp, zp, m0, a, les, kerr_lapse, kerr_shift, g, K, alpha,
                         beta);
    // physical -> mesh components. Cartoon: mesh (1, 2, 3) = physical (x, z, -y), so a
    // component with an odd number of mesh-x3 indices is minus the physical one.
    // Physical order xx xy xz yy yz zz = 0..5.
    Real gm[6], Km[6], bm[3];
    if (cartoon) {
      // mesh order 00 01 02 11 12 22 = xx, xz, -xy, zz, -zy, yy
      gm[0] = g[0]; gm[1] = g[2]; gm[2] = -g[1];
      gm[3] = g[5]; gm[4] = -g[4]; gm[5] = g[3];
      Km[0] = K[0]; Km[1] = K[2]; Km[2] = -K[1];
      Km[3] = K[5]; Km[4] = -K[4]; Km[5] = K[3];
      bm[0] = beta[0]; bm[1] = beta[2]; bm[2] = -beta[1];
    } else {
      for (int n = 0; n < 6; ++n) { gm[n] = g[n]; Km[n] = K[n]; }
      for (int n = 0; n < 3; ++n) bm[n] = beta[n];
    }
    int n = 0;
    for (int p = 0; p < 3; ++p) {
      for (int q = p; q < 3; ++q) {
        adm.g_dd(m,p,q,k,j,i) = gm[n];
        adm.vK_dd(m,p,q,k,j,i) = Km[n];
        ++n;
      }
    }
    adm.alpha(m,k,j,i) = alpha;
    for (int p = 0; p < 3; ++p) adm.beta_u(m,p,k,j,i) = bm[p];
  });

  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin); break;
    case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin); break;
    case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin); break;
  }
  pmbp->pz4c->Z4cToADM(pmbp);
  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMConstraints<2>(pmbp); break;
    case 3: pmbp->pz4c->ADMConstraints<3>(pmbp); break;
    case 4: pmbp->pz4c->ADMConstraints<4>(pmbp); break;
  }

  // Optional diagnostic for the unit tests: max |H| and max |M_i| over interior cells in
  // the shell 1.5 r_h < r < report_rmax (physical radius about the centre), all ranks.
  if (pin->GetOrAddBoolean("problem", "report_constraints", false)) {
    const Real chi2 = chi*chi;
    const Real rh = les ? 0.25*m0*(1.0 + std::sqrt(1.0 - chi2))
                        : 0.5*m0*std::sqrt(1.0 - chi2);
    const Real rmin = 1.5*rh;
    const Real rmax = pin->GetOrAddReal("problem", "report_rmax", 0.8);
    auto &con = pmbp->pz4c->u_con;
    const int iH = z4c::Z4c::I_CON_H;
    const int iMx = z4c::Z4c::I_CON_MX;
    Real eh = 0.0, em = 0.0;
    Kokkos::parallel_reduce("spin_punc_con",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(DevExeSpace(),
                                             {0, indcs.ks, indcs.js, indcs.is},
                                             {nmb, indcs.ke+1, indcs.je+1, indcs.ie+1}),
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, Real &h, Real &mm) {
      Real X[3];
      z4c::Z4cCartesianPos(cmode, size.d_view(m), indcs_, k, j, i, X[0], X[1], X[2]);
      X[0] -= cx1; X[1] -= cx2; X[2] -= cx3;
      const Real r = Kokkos::sqrt(X[0]*X[0] + X[1]*X[1] + X[2]*X[2]);
      if (r <= rmin || r >= rmax) return;
      h = Kokkos::fmax(h, Kokkos::fabs(con(m,iH,k,j,i)));
      for (int a = 0; a < 3; ++a) mm = Kokkos::fmax(mm, Kokkos::fabs(con(m,iMx+a,k,j,i)));
    }, Kokkos::Max<Real>(eh), Kokkos::Max<Real>(em));
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &eh, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &em, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
    if (global_variable::my_rank == 0) {
      std::printf("SPIN_PUNC_CON maxH %.17e maxM %.17e\n", eh, em);
    }
  }

  if (ahf_at_init) {
    for (auto &pahf : pmbp->pz4c->pfastflow) {
      pahf->Find(0, pmy_mesh_->time);
      pahf->Write(0, pmy_mesh_->time);
    }
  }
  if (global_variable::my_rank == 0) {
    std::cout << "z4c_spinning_puncture (" << coord << "): M = " << m0
              << ", chi = " << chi
              << " (spin along physical +z"
              << (cartoon ? " = mesh +x2, 2D Cartoon" : " = mesh +x3") << ")"
              << std::endl;
  }
}
