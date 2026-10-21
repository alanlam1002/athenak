//========================================================================================
// Host test of the <mhd> cold_recovery closed form (src/eos/cold_recovery.hpp; research
// R-083 tests 1 and 2), against the real PrimitiveSolver<IdealGas, ResetFloor>.
//
//  (a) unit check, generic non-diagonal metric, W = 1.0001 .. 5:
//      PrimToCon of the cold-recovered primitives returns D and S_i (to round-off) and
//      the rebuilt tau; a direct ConToPrim of (D, S_i, tau_cr) returns the same rho, u^i
//      and T = T_f (at a well-conditioned T_f; at the deck's T_f = 1e-22 only rho, u^i:
//      T is then below the inversion's resolution, which is the point of the fallback);
//  (b) the logged cycle-2890 cell of c079_c2p (C-082, MLEDGER0_MAX): the densitised cons
//      rebuilt from the logged rho, W, P and psi4_before, inverted against psi4_before and
//      psi4_after (R-082 1a), then the old floor path and the cold recovery on the new one.
//
// Build/run: bash tst/unit/run_test_cold_recovery.sh [cpu_build_dir]
//========================================================================================
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "eos/primitive-solver/primitive_solver.hpp"
#include "eos/primitive-solver/idealgas.hpp"
#include "eos/primitive-solver/reset_floor.hpp"
#include "eos/cold_recovery.hpp"

using PS = Primitive::PrimitiveSolver<Primitive::IdealGas, Primitive::ResetFloor>;

static int nfail = 0;
static void Check(bool ok, const char *what, double v) {
  std::printf("  %-58s %-4s %.3e\n", what, ok ? "ok" : "FAIL", v);
  if (!ok) ++nfail;
}

static void Setup(PS &ps, double tfloor) {
  auto &eos = ps.GetEOSMutable();
  eos.SetGamma(5.0/3.0);
  eos.SetNSpecies(1);
  eos.SetDensityFloor(1.0e-24);   // the R6 decks: dfloor, tfloor, dthreshold
  eos.SetTemperatureFloor(tfloor);
  eos.SetThreshold(1.02);
}

// the kernel's branch: primitives from (D, S_i) at T_f; returns tau
static double ColdRecover(const PS &ps, const double cons[NCONS], const double g3u[6],
                          double tf, double prim[NPRIM], double *Wout) {
  auto &eos = ps.GetEOS();
  const double D = cons[CDN];
  const double S_d[3] = {cons[CSX], cons[CSY], cons[CSZ]};
  double S_u[3];
  Primitive::RaiseForm(S_u, S_d, g3u);
  const double ssq = Primitive::Contract(S_u, S_d);
  double yc[MAX_SPECIES] = {cons[CYD]/D, 0.0, 0.0};
  const double hf = eos.GetEnthalpy(1.0, tf, yc);
  double uc[3];
  const double W = ColdRecoveryW(D, S_u, ssq, hf, uc);
  const double nc = D/(W*eos.GetBaryonMass());
  prim[PRH] = nc; prim[PVX] = uc[0]; prim[PVY] = uc[1]; prim[PVZ] = uc[2];
  prim[PTM] = tf; prim[PYF] = yc[0];
  prim[PPR] = eos.GetPressure(nc, tf, yc);
  *Wout = W;
  return D*hf*W - prim[PPR] - D;
}

static void Inv(const double g[6], double gu[6]) {
  const double det = Primitive::GetDeterminant(g);
  gu[S11] = (g[S22]*g[S33] - g[S23]*g[S23])/det;
  gu[S12] = (g[S13]*g[S23] - g[S12]*g[S33])/det;
  gu[S13] = (g[S12]*g[S23] - g[S13]*g[S22])/det;
  gu[S22] = (g[S11]*g[S33] - g[S13]*g[S13])/det;
  gu[S23] = (g[S12]*g[S13] - g[S11]*g[S23])/det;
  gu[S33] = (g[S11]*g[S22] - g[S12]*g[S12])/det;
}

int main() {
  double bu[NMAG] = {0.0, 0.0, 0.0};
  // ---------------------------------------------------------------- (a) unit check
  std::printf("(a) unit check: generic metric, closed form vs PrimToCon / ConToPrim\n");
  const double g[6] = {1.21, 0.03, -0.02, 1.25, 0.01, 1.19};
  double gu[6];
  Inv(g, gu);
  const double Ws[4] = {1.0001, 1.106, 2.0, 5.0};
  const double tfs[3] = {1.0e-3, 1.0e-6, 1.0e-22};
  for (double tf : tfs) {
    PS ps, pd;
    Setup(ps, tf);
    Setup(pd, tf*1.0e-3);   // direct solve: floor out of the way
    for (double Wt : Ws) {
      // a reference state (any T): D, S_i from it; the closed form then only sees D, S_i
      const double ut = std::sqrt(Wt*Wt - 1.0);
      double dir[3] = {0.6, -0.48, 0.64}, dd[3];
      Primitive::LowerVector(dd, dir, g);
      const double nrm = std::sqrt(Primitive::Contract(dir, dd));
      double pr[NPRIM] = {0.0}, cons[NCONS] = {0.0};
      pr[PRH] = 4.0e-9; pr[PTM] = 7.0e-8; pr[PYF] = 0.3;
      for (int a = 0; a < 3; ++a) pr[PVX + a] = ut*dir[a]/nrm;
      pr[PPR] = ps.GetEOS().GetPressure(pr[PRH], pr[PTM], &pr[PYF]);
      double gg[6] = {g[0], g[1], g[2], g[3], g[4], g[5]};
      ps.PrimToCon(pr, cons, bu, gg);
      double pc[NPRIM] = {0.0}, W;
      const double tau_cr = ColdRecover(ps, cons, gu, tf, pc, &W);
      double c2[NCONS] = {0.0};
      ps.PrimToCon(pc, c2, bu, gg);
      const double smag = std::fabs(cons[CSX]) + std::fabs(cons[CSY]) + std::fabs(cons[CSZ]);
      const double eD = std::fabs(c2[CDN]/cons[CDN] - 1.0);
      double eS = 0.0;
      for (int a = 0; a < 3; ++a) eS = std::fmax(eS, std::fabs(c2[CSX + a] - cons[CSX + a])/smag);
      const double eT = std::fabs(c2[CTA] - tau_cr)/(cons[CDN]*Wt);
      std::printf(" T_f=%.0e W=%.4f  (W_cr=%.6f)\n", tf, Wt, W);
      Check(eD < 4e-16*Wt*Wt, "PrimToCon(cold prims): D recovered, rel err", eD);
      Check(eS < 4e-16*Wt*Wt, "PrimToCon(cold prims): S_i recovered, err/|S|", eS);
      Check(eT < 4e-16*Wt*Wt, "tau consistent with the primitives, err/(D W)", eT);
      // direct solve of (D, S_i, tau_cr)
      double c3[NCONS], p3[NPRIM] = {0.0}, gu3[6] = {gu[0], gu[1], gu[2], gu[3], gu[4], gu[5]};
      for (int n = 0; n < NCONS; ++n) c3[n] = cons[n];
      c3[CTA] = tau_cr;
      auto r = pd.ConToPrim(p3, c3, bu, gg, gu3);
      double eu = 0.0;
      for (int a = 0; a < 3; ++a) eu = std::fmax(eu, std::fabs(p3[PVX + a] - pc[PVX + a])/ut);
      const double en = std::fabs(p3[PRH]/pc[PRH] - 1.0);
      // at T_f = 1e-22 the direct solve cannot resolve T (it may floor it): rho, u only
      Check(r.error == Primitive::Error::SUCCESS && (tf < 1.0e-6 || !r.prim_floor) &&
            en < 1e-9,
            "direct ConToPrim: rho matches, rel err", en);
      Check(eu < 1e-9, "direct ConToPrim: u^i matches, err/|u|", eu);
      if (tf >= 1.0e-6) {
        const double eTm = std::fabs(p3[PTM]/tf - 1.0);
        Check(eTm < 1e-4, "direct ConToPrim: T = T_f, rel err", eTm);
      }
    }
  }

  // ---------------------------------------------------------------- (b) the 2890 cell
  std::printf("\n(b) c079_c2p cycle 2890 cell (MLEDGER0_MAX, 10-digit log)\n");
  const double Dt_log = 5.929558005e-09, rho0 = 4.019971242e-09, W0 = 1.106109371;
  const double P0 = 2.936215486e-16, p4o = 1.211530084, p4n = 1.211528078;
  const double Dafter_log = 5.833497306e-09, Wafter_log = 1.088190253;
  PS ps;
  Setup(ps, 1.0e-22);
  const double T0 = P0/rho0;
  std::printf("  logged state: rho=%.10g W=%.10g P/rho=%.4g, dpsi4/psi4=%.3e\n", rho0, W0,
              T0, p4n/p4o - 1.0);
  // old metric: conformally flat psi4_before; u along x (isotropic metric: direction-free)
  double go[6] = {p4o, 0.0, 0.0, p4o, 0.0, p4o}, guo[6];
  double gn[6] = {p4n, 0.0, 0.0, p4n, 0.0, p4n}, gun[6];
  Inv(go, guo);
  Inv(gn, gun);
  double pr[NPRIM] = {0.0}, c_und[NCONS] = {0.0};
  pr[PRH] = rho0; pr[PTM] = T0; pr[PYF] = 0.0;
  pr[PVX] = std::sqrt((W0*W0 - 1.0)/p4o);
  pr[PPR] = ps.GetEOS().GetPressure(rho0, T0, &pr[PYF]);
  ps.PrimToCon(pr, c_und, bu, go);
  const double s6o = std::pow(p4o, 1.5), s6n = std::pow(p4n, 1.5);
  double ct[NCONS];   // densitised: the u0 the regrid carries unchanged
  for (int n = 0; n < NCONS; ++n) ct[n] = c_und[n]*s6o;
  std::printf("  densitised D rebuilt %.10g vs logged Dbefore %.10g (rel %.1e)\n", ct[CDN],
              Dt_log, ct[CDN]/Dt_log - 1.0);
  std::printf("  tau/D - (W-1) = %.4e, eps = %.4e: amplification (tau/D)/eps = %.3g\n",
              c_und[CTA]/c_und[CDN] - (W0 - 1.0), 1.5*T0, (c_und[CTA]/c_und[CDN])/(1.5*T0));
  for (int pass = 0; pass < 2; ++pass) {
    const double *gm = pass ? gn : go, *gmu = pass ? gun : guo, s6 = pass ? s6n : s6o;
    double gg[6], ggu[6], c[NCONS], p[NPRIM] = {0.0};
    for (int a = 0; a < 6; ++a) { gg[a] = gm[a]; ggu[a] = gmu[a]; }
    for (int n = 0; n < NCONS; ++n) c[n] = ct[n]/s6;
    PS pfree;   // no T floor in the way: the inversion's own temperature
    Setup(pfree, 1.0e-300);
    double cf[NCONS], pf[NPRIM] = {0.0};
    for (int n = 0; n < NCONS; ++n) cf[n] = c[n];
    auto rf = pfree.ConToPrim(pf, cf, bu, gg, ggu);
    std::printf("  %s metric: inversion T = %.4e (T_f = 1e-22), P = %.4e, err=%d\n",
                pass ? "NEW" : "OLD", pf[PTM], pf[PPR], static_cast<int>(rf.error));
    auto r = ps.ConToPrim(p, c, bu, gg, ggu);
    std::printf("    with the deck floor: prim_floor=%d cons_floor=%d cons_adjusted=%d "
                "error=%d\n", r.prim_floor, r.cons_floor, r.cons_adjusted,
                static_cast<int>(r.error));
    if (!pass) {
      Check(!r.prim_floor && pf[PTM] > 1.0e-22,
            "OLD metric: P above the floor (mechanism, R-082 1a)", pf[PTM]);
    } else {
      Check(r.prim_floor && pf[PTM] < 1.0e-22,
            "NEW metric: T below the floor -> code-208 class", pf[PTM]);
      // the old treatment then sets v = xidot; cold recovery instead:
      double cc[NCONS], pc[NPRIM] = {0.0}, W;
      for (int n = 0; n < NCONS; ++n) cc[n] = ct[n]/s6;
      const double tau_cr = ColdRecover(ps, cc, ggu, 1.0e-22, pc, &W);
      const double dtau = tau_cr - cc[CTA];
      std::printf("    cold recovery: W=%.10g (logged before %.10g, old path after %.10g)\n",
                  W, W0, Wafter_log);
      std::printf("    rho=%.10g  dtau=%.4e = %.3e D (densitised dtau dV-free %.4e)\n",
                  pc[PRH], dtau, dtau/cc[CDN], dtau*s6);
      std::printf("    old path: D %.10g -> %.10g (%.2f%%); cold recovery: D unchanged "
                  "(not written back)\n", Dt_log, Dafter_log,
                  100.0*(Dafter_log/Dt_log - 1.0));
      Check(std::fabs(W/W0 - 1.0) < 1e-5, "cold-recovery W = W_before (from S)",
            W/W0 - 1.0);
      Check(std::fabs(dtau/cc[CDN]) < 1e-6, "|dtau|/D ~ 1e-7 (R-083 expectation)",
            std::fabs(dtau/cc[CDN]));
    }
  }
  std::printf("\n%s (%d failure%s)\n", nfail ? "FAILED" : "ALL PASS", nfail,
              nfail == 1 ? "" : "s");
  return nfail ? 1 : 0;
}
