//========================================================================================
// K-003 step 0: host-side checks from the KSTAB tables (no GPU, no AthenaK objects).
//   (a) V2a-U (kerr_methods eq:hardy): min over (r_f, th) of 4 r_f^2 c0, with
//       c0 = Rt/8 + 7/8 psi0^-8 Ahat0_ij Ahat0^ij = Rt/8 + 7/8 psi0^4 K_ij K^ij.
//   (b) D3/D4 geometry: r_AH(th) on the slice, from the horizon generator
//       chi = d_T + Omega_H d_phi being null: N = -alpha^2 + g_ij chi^i chi^j = 0, with
//       chi^i = beta^i + Omega_H (-y, x, 0)^i and Omega_H = a/(r_+^2 + a^2). (The slice is
//       stationary and its azimuth is phi~ + g(r_f, th), so d_T and d_phi at fixed table
//       coordinates are the Killing fields.) Reported: max_th r_AH / min_th r_AH.
// The conformal quantities are formed independently of kerr_background.hpp, from
// kstab::Table::eval_d (exact spectral derivatives), and are checked against F4.
//
//   g++ -O3 -std=c++17 -fopenmp -I$KSTAB_INC kerr_host_checks.cpp -o kerr_host_checks
//   kerr_host_checks f4  KSTAB_a0.60 F4_FILE
//   kerr_host_checks run KSTAB A [NR NTH]
//   kerr_host_checks blend KSTAB A RAHMAX F1 F2 [NRT NTH]   (K-004 A2: D3-blended gt)
//   kerr_host_checks dt KSTAB RCUT ACUT RMAX RBOX [XI1 XI2 XI3 NR NTH NPH]   (K-004 A6)
//========================================================================================
#include <omp.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "kstab.hpp"

namespace {
const int II[6] = {0, 0, 0, 1, 1, 2}, JJ[6] = {0, 1, 2, 1, 2, 2};
int S(int i, int j) {
  static const int m[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
  return m[i][j];
}

struct Conf {
  double psi, rt, ah2, c0, detm1, gtrace;  // gtrace: max_i |Gt^k_ki|
  double rtij[6], ahu[6];
};

// d1[k][q], d2[m][q] with m = Sym(k,l) as in kstab::eval_d.
// D3 blend (K-004 A2), as BlendNode (kerr_background.hpp) then KerrConformal do it, but
// written here independently at the conformal level: gt -> gb = M det(M)^(-1/3),
// M = delta + w(r) (gt - delta), w the quintic smootherstep in r = |x| between r1 and r2;
// det g (psi0) and K_ij are kept, so K is raised with g_b^-1 = gb^-1 det(g)^(-1/3).
struct Blend { bool on = false; double x[3] = {0, 0, 0}, r1 = 0, r2 = 0; };

// gt = e g (e = det(g)^(-1/3)) and its derivatives from g's (ddg[k][l] symmetric)
void Conformal3(const double g[3][3], const double dg[3][3][3], const double ddg[3][3][3][3],
                double gt[3][3], double gi[3][3], double dgt[3][3][3],
                double ddgt[3][3][3][3], double *det) {
  const double D = g[0][0]*(g[1][1]*g[2][2] - g[1][2]*g[2][1])
                 - g[0][1]*(g[1][0]*g[2][2] - g[1][2]*g[2][0])
                 + g[0][2]*(g[1][0]*g[2][1] - g[1][1]*g[2][0]);
  gi[0][0] = (g[1][1]*g[2][2] - g[1][2]*g[2][1])/D;
  gi[0][1] = (g[0][2]*g[2][1] - g[0][1]*g[2][2])/D;
  gi[0][2] = (g[0][1]*g[1][2] - g[0][2]*g[1][1])/D;
  gi[1][1] = (g[0][0]*g[2][2] - g[0][2]*g[2][0])/D;
  gi[1][2] = (g[0][2]*g[1][0] - g[0][0]*g[1][2])/D;
  gi[2][2] = (g[0][0]*g[1][1] - g[0][1]*g[1][0])/D;
  gi[1][0] = gi[0][1]; gi[2][0] = gi[0][2]; gi[2][1] = gi[1][2];
  const double e = std::pow(D, -1.0/3.0);
  double L[3], LL[3][3];
  for (int k = 0; k < 3; ++k) {
    L[k] = 0;
    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) L[k] += gi[a][b]*dg[k][a][b];
  }
  for (int k = 0; k < 3; ++k) for (int l = 0; l < 3; ++l) {
    double s = 0;
    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) {
      s += gi[a][b]*ddg[k][l][a][b];
      for (int c = 0; c < 3; ++c) for (int d = 0; d < 3; ++d)
        s -= gi[a][c]*gi[b][d]*dg[l][c][d]*dg[k][a][b];
    }
    LL[k][l] = s;
  }
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
    gt[i][j] = e*g[i][j];
    for (int k = 0; k < 3; ++k) {
      dgt[k][i][j] = e*(dg[k][i][j] - L[k]*g[i][j]/3);
      for (int l = 0; l < 3; ++l)
        ddgt[k][l][i][j] = e*(ddg[k][l][i][j] - L[l]*dg[k][i][j]/3 - L[k]*dg[l][i][j]/3
                              - LL[k][l]*g[i][j]/3 + L[k]*L[l]*g[i][j]/9);
    }
  }
  *det = D;
}

Conf Form(const double f[16], const double d1[3][16], const double d2[6][16],
          const Blend &bl = Blend()) {
  Conf o{};
  double g[3][3], gi[3][3], dg[3][3][3], ddg[3][3][3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      g[i][j] = f[S(i, j)];
      for (int k = 0; k < 3; ++k) {
        dg[k][i][j] = d1[k][S(i, j)];
        for (int l = 0; l < 3; ++l) ddg[k][l][i][j] = d2[S(k, l)][S(i, j)];
      }
    }
  const double D = g[0][0]*(g[1][1]*g[2][2] - g[1][2]*g[2][1])
                 - g[0][1]*(g[1][0]*g[2][2] - g[1][2]*g[2][0])
                 + g[0][2]*(g[1][0]*g[2][1] - g[1][1]*g[2][0]);
  gi[0][0] = (g[1][1]*g[2][2] - g[1][2]*g[2][1])/D;
  gi[0][1] = (g[0][2]*g[2][1] - g[0][1]*g[2][2])/D;
  gi[0][2] = (g[0][1]*g[1][2] - g[0][2]*g[1][1])/D;
  gi[1][1] = (g[0][0]*g[2][2] - g[0][2]*g[2][0])/D;
  gi[1][2] = (g[0][2]*g[1][0] - g[0][0]*g[1][2])/D;
  gi[2][2] = (g[0][0]*g[1][1] - g[0][1]*g[1][0])/D;
  gi[1][0] = gi[0][1]; gi[2][0] = gi[0][2]; gi[2][1] = gi[1][2];
  const double psi = std::pow(D, 1.0/12.0), e = std::pow(D, -1.0/3.0);
  double L[3], LL[3][3];
  for (int k = 0; k < 3; ++k) {
    L[k] = 0;
    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) L[k] += gi[a][b]*dg[k][a][b];
  }
  for (int k = 0; k < 3; ++k) for (int l = 0; l < 3; ++l) {
    double s = 0;
    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) {
      s += gi[a][b]*ddg[k][l][a][b];
      for (int c = 0; c < 3; ++c) for (int d = 0; d < 3; ++d)
        s -= gi[a][c]*gi[b][d]*dg[l][c][d]*dg[k][a][b];
    }
    LL[k][l] = s;
  }
  double gt[3][3], gtu[3][3], dgt[3][3][3], ddgt[3][3][3][3];
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
    gt[i][j] = e*g[i][j];
    gtu[i][j] = gi[i][j]/e;
    for (int k = 0; k < 3; ++k) {
      dgt[k][i][j] = e*(dg[k][i][j] - L[k]*g[i][j]/3);
      for (int l = 0; l < 3; ++l)
        ddgt[k][l][i][j] = e*(ddg[k][l][i][j] - L[l]*dg[k][i][j]/3 - L[k]*dg[l][i][j]/3
                              - LL[k][l]*g[i][j]/3 + L[k]*L[l]*g[i][j]/9);
    }
  }
  if (bl.on) {
    const double r = std::sqrt(bl.x[0]*bl.x[0] + bl.x[1]*bl.x[1] + bl.x[2]*bl.x[2]);
    if (r < bl.r2) {
      double w = 0, w1 = 0, w2 = 0;
      if (r > bl.r1) {
        const double Lw = bl.r2 - bl.r1, q = (r - bl.r1)/Lw;
        w = q*q*q*(10 + q*(-15 + 6*q));
        w1 = 30*q*q*(1 - q)*(1 - q)/Lw;
        w2 = 60*q*(1 - q)*(1 - 2*q)/(Lw*Lw);
      }
      double dw[3] = {0, 0, 0}, ddw[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
      if (r > bl.r1) {
        for (int k = 0; k < 3; ++k) {
          const double nk = bl.x[k]/r;
          dw[k] = w1*nk;
          for (int l = 0; l < 3; ++l) {
            const double nl = bl.x[l]/r;
            ddw[k][l] = w2*nk*nl + w1*((k == l ? 1.0 : 0.0) - nk*nl)/r;
          }
        }
      }
      double M[3][3], dM[3][3][3], ddM[3][3][3][3];
      for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
        const double h = gt[i][j] - (i == j ? 1.0 : 0.0);
        M[i][j] = (i == j ? 1.0 : 0.0) + w*h;
        for (int k = 0; k < 3; ++k) {
          dM[k][i][j] = dw[k]*h + w*dgt[k][i][j];
          for (int l = 0; l < 3; ++l)
            ddM[k][l][i][j] = ddw[k][l]*h + dw[k]*dgt[l][i][j] + dw[l]*dgt[k][i][j]
                              + w*ddgt[k][l][i][j];
        }
      }
      double Mi[3][3], dum;
      Conformal3(M, dM, ddM, gt, Mi, dgt, ddgt, &dum);
      // gtu = gb^-1 = Mi det(M)^(1/3) ... recompute from gt directly
      const double Dg = gt[0][0]*(gt[1][1]*gt[2][2] - gt[1][2]*gt[2][1])
                      - gt[0][1]*(gt[1][0]*gt[2][2] - gt[1][2]*gt[2][0])
                      + gt[0][2]*(gt[1][0]*gt[2][1] - gt[1][1]*gt[2][0]);
      gtu[0][0] = (gt[1][1]*gt[2][2] - gt[1][2]*gt[2][1])/Dg;
      gtu[0][1] = (gt[0][2]*gt[2][1] - gt[0][1]*gt[2][2])/Dg;
      gtu[0][2] = (gt[0][1]*gt[1][2] - gt[0][2]*gt[1][1])/Dg;
      gtu[1][1] = (gt[0][0]*gt[2][2] - gt[0][2]*gt[2][0])/Dg;
      gtu[1][2] = (gt[0][2]*gt[1][0] - gt[0][0]*gt[1][2])/Dg;
      gtu[2][2] = (gt[0][0]*gt[1][1] - gt[0][1]*gt[1][0])/Dg;
      gtu[1][0] = gtu[0][1]; gtu[2][0] = gtu[0][2]; gtu[2][1] = gtu[1][2];
      for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) gi[i][j] = gtu[i][j]*e;
    }
  }
  // Gamma_lij (lowered) and its derivative; Gt^k_ij and d_m Gt^k_ij
  double Gl[3][3][3], dGl[3][3][3][3], G[3][3][3], dG[3][3][3][3], dgtu[3][3][3];
  for (int l = 0; l < 3; ++l) for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
    Gl[l][i][j] = 0.5*(dgt[i][l][j] + dgt[j][i][l] - dgt[l][i][j]);
    for (int m = 0; m < 3; ++m)
      dGl[m][l][i][j] = 0.5*(ddgt[m][i][l][j] + ddgt[m][j][i][l] - ddgt[m][l][i][j]);
  }
  for (int m = 0; m < 3; ++m) for (int k = 0; k < 3; ++k) for (int l = 0; l < 3; ++l) {
    double s = 0;
    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) s -= gtu[k][a]*dgt[m][a][b]*gtu[b][l];
    dgtu[m][k][l] = s;
  }
  for (int k = 0; k < 3; ++k) for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
    double s = 0;
    for (int l = 0; l < 3; ++l) s += gtu[k][l]*Gl[l][i][j];
    G[k][i][j] = s;
    for (int m = 0; m < 3; ++m) {
      double t = 0;
      for (int l = 0; l < 3; ++l) t += dgtu[m][k][l]*Gl[l][i][j] + gtu[k][l]*dGl[m][l][i][j];
      dG[m][k][i][j] = t;
    }
  }
  double rt = 0;
  for (int q = 0; q < 6; ++q) {
    const int i = II[q], j = JJ[q];
    double s = 0;
    for (int k = 0; k < 3; ++k) {
      s += dG[k][k][i][j] - dG[j][k][i][k];
      for (int l = 0; l < 3; ++l) s += G[k][k][l]*G[l][i][j] - G[k][j][l]*G[l][i][k];
    }
    o.rtij[q] = s;
  }
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) rt += gtu[i][j]*o.rtij[S(i, j)];
  // K^ij (physical), Ahat0^ij = psi^10 K^ij, Ahat0_ij Ahat0^ij = psi^12 K_ij K^ij
  double K[3][3], Ku[3][3], kk = 0;
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) K[i][j] = f[6 + S(i, j)];
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
    double s = 0;
    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) s += gi[i][a]*gi[j][b]*K[a][b];
    Ku[i][j] = s;
  }
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) kk += K[i][j]*Ku[i][j];
  for (int q = 0; q < 6; ++q) o.ahu[q] = std::pow(psi, 10)*Ku[II[q]][JJ[q]];
  o.psi = psi;
  o.rt = rt;
  o.ah2 = std::pow(psi, 12)*kk;
  o.c0 = rt/8 + 0.875*std::pow(psi, 4)*kk;
  double dt = gt[0][0]*(gt[1][1]*gt[2][2] - gt[1][2]*gt[2][1])
            - gt[0][1]*(gt[1][0]*gt[2][2] - gt[1][2]*gt[2][0])
            + gt[0][2]*(gt[1][0]*gt[2][1] - gt[1][1]*gt[2][0]);
  o.detm1 = std::fabs(dt - 1);
  for (int i = 0; i < 3; ++i) {
    double s = 0;
    for (int k = 0; k < 3; ++k) s += G[k][k][i];
    o.gtrace = std::fmax(o.gtrace, std::fabs(s));
  }
  return o;
}

Conf AtPolar(const kstab::Table &t, double rf, double th, double ph = 0, double br1 = 0,
             double br2 = 0) {
  double f[16], d1[3][16], d2[6][16];
  Blend bl;
  bl.x[0] = rf*std::sin(th)*std::cos(ph);
  bl.x[1] = rf*std::sin(th)*std::sin(ph);
  bl.x[2] = rf*std::cos(th);
  bl.on = br2 > 0;
  bl.r1 = br1;
  bl.r2 = br2;
  t.eval_d(bl.x[0], bl.x[1], bl.x[2], f, d1, d2);
  return Form(f, d1, d2, bl);
}

// N = -alpha^2 + g_ij chi^i chi^j on phi = 0 (y = 0): chi = beta + Omega_H (0, x, 0)
double HorizonN(const kstab::Table &t, double om, double rf, double th) {
  double f[16];
  const double x = rf*std::sin(th), z = rf*std::cos(th);
  t.eval(x, 0.0, z, f);
  double chi[3] = {f[13], f[14] + om*x, f[15]};
  double n = -f[12]*f[12];
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) n += f[S(i, j)]*chi[i]*chi[j];
  return n;
}

int F4Check(const char *tab, const char *f4) {
  kstab::Table t(tab);
  std::ifstream in(f4);
  std::map<std::string, std::map<std::string, double>> ref;
  std::map<std::string, double> prf, pth, pph;
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream ss(line);
    std::string p, k;
    if (!(ss >> p >> k) || p.size() != 2 || p[0] != 'P') continue;
    if (k == "r_f") {
      std::string a, b;
      double r, th, ph;
      ss >> r >> a >> th >> b >> ph;
      prf[p] = r; pth[p] = th; pph[p] = ph;
    } else {
      double v;
      if (ss >> v) ref[p][k] = v;
    }
  }
  const char *nm[6] = {"xx", "xy", "xz", "yy", "yz", "zz"};
  double worst = 0;
  for (auto &kv : prf) {
    const std::string &p = kv.first;
    Conf c = AtPolar(t, prf[p], pth[p], pph[p]);
    auto rel = [](double a, double b, double s) { return std::fabs(a - b)/s; };
    double sR = 0, sA = 0, eR = 0, eA = 0;
    for (int q = 0; q < 6; ++q) {
      sR = std::fmax(sR, std::fabs(ref[p][std::string("Rt_") + nm[q]]));
      sA = std::fmax(sA, std::fabs(ref[p][std::string("Ahat^") + nm[q]]));
    }
    for (int q = 0; q < 6; ++q) {
      eR = std::fmax(eR, rel(c.rtij[q], ref[p][std::string("Rt_") + nm[q]], sR));
      eA = std::fmax(eA, rel(c.ahu[q], ref[p][std::string("Ahat^") + nm[q]], sA));
    }
    // c0 from F4's own values
    double a2 = 0;
    for (int q = 0; q < 6; ++q) {
      const double w = (II[q] == JJ[q]) ? 1 : 2;
      a2 += w*ref[p][std::string("Ahat^") + nm[q]]*ref[p][std::string("Ahat_") + nm[q]];
    }
    const double psi = ref[p]["psi"];
    const double c0ref = ref[p]["Rt"]/8 + 0.875*std::pow(psi, -8)*a2;
    const double e1 = rel(c.psi, psi, psi), e2 = rel(c.rt, ref[p]["Rt"], std::fabs(ref[p]["Rt"]));
    const double e3 = rel(c.c0, c0ref, std::fabs(c0ref));
    std::printf("F4 %s r_f %.3f th %.3f: psi %.1e Rt_ij %.1e Rt %.1e Ahat^ij %.1e | c0 %+.6e "
                "(F4 %+.6e, rel %.1e) 4r^2c0 %+.4e\n", p.c_str(), prf[p], pth[p], e1, eR, e2,
                eA, c.c0, c0ref, e3, 4*prf[p]*prf[p]*c.c0);
    worst = std::fmax(worst, std::fmax(std::fmax(e1, eR), std::fmax(eA, e3)));
  }
  std::printf("F4 worst relative %.2e -> %s\n", worst, worst <= 1e-10 ? "PASS" : "FAIL");
  return worst <= 1e-10 ? 0 : 1;
}

int Run(const char *tab, double a, int nr, int nth) {
  kstab::Table t(tab);
  const double rmin = t.rmin, rmax = 1e3;
  const double rp = 1 + std::sqrt(1 - a*a), om = a/(rp*rp + a*a);
  // (b) horizon: bracket N's sign change in r_f by a scan, then bisection
  const int nah = 257;
  std::vector<double> rah(nah);
  int nbad = 0;
  for (int j = 0; j < nah; ++j) {
    const double th = M_PI*j/(nah - 1);
    // chi is spacelike (N > 0) between r_- and r_+, where the trumpet's throat lies, timelike
    // just outside r_+, and spacelike again beyond the light surface: the horizon is the
    // FIRST sign change scanning outward from r_min.
    double rlo = -1, rhi = -1;
    const int ns = 4000;
    const double r0 = rmin*1.0001;
    double rprev = r0, nprev = HorizonN(t, om, rprev, th);
    for (int s = 1; s <= ns; ++s) {
      const double r = r0*std::exp(std::log(10/r0)*s/ns);
      const double n = HorizonN(t, om, r, th);
      if ((n > 0) != (nprev > 0)) { rlo = rprev; rhi = r; break; }
      rprev = r; nprev = n;
    }
    if (rlo > 0 && !(HorizonN(t, om, r0, th) > 0)) rlo = -1;  // must start spacelike
    if (rlo < 0) { rah[j] = NAN; ++nbad; continue; }
    double nlo = HorizonN(t, om, rlo, th);
    for (int it = 0; it < 200 && rhi - rlo > 1e-15*rhi; ++it) {
      const double rm = 0.5*(rlo + rhi), nm = HorizonN(t, om, rm, th);
      if ((nm > 0) == (nlo > 0)) { rlo = rm; nlo = nm; } else { rhi = rm; }
    }
    rah[j] = 0.5*(rlo + rhi);
  }
  double amin = 1e300, amax = 0, thmin = 0, thmax = 0;
  for (int j = 0; j < nah; ++j) {
    if (!(rah[j] > 0)) continue;
    if (rah[j] < amin) { amin = rah[j]; thmin = M_PI*j/(nah - 1); }
    if (rah[j] > amax) { amax = rah[j]; thmax = M_PI*j/(nah - 1); }
  }
  const double rexc = 0.75*amin;
  std::printf("AH a %.3f r_+ %.15f Omega_H %.12f: r_f pole %.12f equator %.12f; min %.12f "
              "(th %.4f) max %.12f (th %.4f); max/min %.9f; unresolved %d of %d -> %s\n",
              a, rp, om, rah[0], rah[(nah - 1)/2], amin, thmin, amax, thmax, amax/amin, nbad,
              nah, (nbad == 0 && amax/amin < 1.5) ? "PASS" : "FAIL");
  std::printf("AH a %.3f blend 0.5 max r_AH %.6f vs excision 0.75 min r_AH %.6f\n", a,
              0.5*amax, rexc);
  // KHC_AH_ONLY (K-006 D5): print r_AH at every th and skip (a)
  const char *aho = std::getenv("KHC_AH_ONLY");
  if (aho != nullptr && std::atoi(aho) != 0) {
    for (int j = 0; j < nah; ++j)
      std::printf("AHPROF a %.3f j %d th %.17e r_AH %.17e\n", a, j, M_PI*j/(nah - 1), rah[j]);
    return nbad == 0 ? 0 : 1;
  }
  // (a) V2a-U on a (ln r_f, th) grid over [r_min, 1e3] x [0, pi], then refine at the minima
  struct Best { double v = 1e300, r = 0, th = 0, c0 = 0, rt = 0, ah = 0, psi = 0; };
  auto scan = [&](double lr0, double lr1, double t0, double t1, int n1, int n2, double rcut,
                  Best &b, double &cmin_any, int &nneg) {
    std::vector<Best> bs(omp_get_max_threads());
    std::vector<double> cm(bs.size(), 1e300);
    std::vector<int> ng(bs.size(), 0);
#pragma omp parallel for collapse(2) schedule(dynamic, 16)
    for (int i = 0; i < n1; ++i) for (int j = 0; j < n2; ++j) {
      const double r = std::exp(lr0 + (lr1 - lr0)*i/(n1 - 1));
      const double th = t0 + (t1 - t0)*j/(n2 - 1);
      if (r < rcut) continue;
      Conf c = AtPolar(t, r, th);
      const int me = omp_get_thread_num();
      const double v = 4*r*r*c.c0;
      if (c.c0 < 0) ng[me]++;
      cm[me] = std::fmin(cm[me], c.c0);
      if (v < bs[me].v) bs[me] = {v, r, th, c.c0, c.rt, c.ah2, c.psi};
    }
    for (size_t m = 0; m < bs.size(); ++m) {
      if (bs[m].v < b.v) b = bs[m];
      cmin_any = std::fmin(cmin_any, cm[m]);
      nneg += ng[m];
    }
  };
  const double lr0 = std::log(rmin*(1 + 1e-12)), lr1 = std::log(rmax);
  const double rcuts[3] = {0, rexc, amax};
  const char *rname[3] = {"all r_f       ", "r_f>=0.75minAH", "r_f>=maxAH    "};
  for (int reg = 0; reg < 3; ++reg) {
    Best b;
    double cmin = 1e300;
    int nneg = 0;
    scan(lr0, lr1, 0, M_PI, nr, nth, rcuts[reg], b, cmin, nneg);
    const int ncoarse = nneg;
    // refine around the minimum: +-2 coarse cells, 4x finer per direction... twice
    double dl = (lr1 - lr0)/(nr - 1), dt = M_PI/(nth - 1);
    for (int pass = 0; pass < 2; ++pass) {
      const double l = std::log(b.r), th = b.th;
      int dummy = 0;
      double cd = 1e300;
      scan(std::fmax(lr0, l - 2*dl), std::fmin(lr1, l + 2*dl), std::fmax(0.0, th - 2*dt),
           std::fmin(M_PI, th + 2*dt), 33, 33, rcuts[reg], b, cd, dummy);
      dl /= 8; dt /= 8;
    }
    std::printf("HARDY a %.3f %s: min 4 r_f^2 c0 = %+.6e at r_f %.6e th %.6f (c0 %+.4e, Rt/8 "
                "%+.4e, 7/8 psi^-8 A^2 %+.4e, psi0 %.4f); margin to -1: %.3g; "
                "c0 < 0 at %d of %d grid points; min c0 %+.4e -> %s\n",
                a, rname[reg], b.v, b.r, b.th, b.c0, b.rt/8,
                b.c0 - b.rt/8, b.psi, b.v < 0 ? -1/b.v : INFINITY, ncoarse, nr*nth, cmin,
                b.v > -1 ? "PASS" : "FAIL");
  }
  // radial profile: min over th of 4 r_f^2 c0, per decade-quarter
  for (int i = 0; i <= 4*7; ++i) {
    const double r = std::pow(10.0, -4 + 0.25*i);
    if (r < rmin) continue;
    double w = 1e300, wth = 0;
    for (int j = 0; j < nth; ++j) {
      const double th = M_PI*j/(nth - 1);
      Conf c = AtPolar(t, r, th);
      if (4*r*r*c.c0 < w) { w = 4*r*r*c.c0; wth = th; }
    }
    std::printf("PROF a %.3f r_f %.4e (r_f/minAH %.3e): min_th 4 r_f^2 c0 = %+.4e (th %.4f)\n",
                a, r, r/amin, w, wth);
  }
  // beyond F2's edge (the spectral tail; Hardy over R^3 needs all r): decades to 1e6
  for (double r : {1e3, 1e4, 1e5, 1e6}) {
    double w = 1e300, wth = 0;
    for (int j = 0; j < nth; ++j) {
      const double th = M_PI*j/(nth - 1);
      Conf c = AtPolar(t, r, th);
      if (4*r*r*c.c0 < w) { w = 4*r*r*c.c0; wth = th; }
    }
    std::printf("TAIL a %.3f r_f %.0e: min_th 4 r_f^2 c0 = %+.4e (th %.4f)\n", a, r, w, wth);
  }
  if (t.fallback_count() > 0) std::printf("WARNING fallback used %d times\n",
                                          static_cast<int>(t.fallback_count()));
  return 0;
}
// K-004 A2: V2a-U on the D3-blended gt. r1, r2 = (f1, f2) max_th r_AH (F1), table units.
// Regions: r_f < r1 (flat gt: Rt = 0), [r1, r2] (linear in r_f, >= 50 points), r_f >= r2;
// each scanned on a deterministic (r_f, th) grid, then refined twice about the minimum.
int RunBlend(const char *tab, double a, double rahmax, double f1, double f2, int nrt,
             int nth) {
  kstab::Table t(tab);
  const double rmin = t.rmin, rmax = 1e3, r1 = f1*rahmax, r2 = f2*rahmax;
  struct Best { double v = 1e300, r = 0, th = 0, c0 = 0, rt = 0, psi = 0; };
  // wst/umin: the largest drop of 4 r_f^2 c0 caused by the blend (unblended minus blended at
  // the same point; > 0 where the blend lowers it) and the unblended minimum, over the scan
  auto scan = [&](bool logr, double ra, double rb, double t0, double t1, int n1, int n2,
                  Best &b, int &nneg, double &dmax, double &gmax, Best &wst, double &umin) {
    std::vector<Best> bs(omp_get_max_threads()), ws(bs.size());
    std::vector<double> um(bs.size(), 1e300);
    for (Best &x : ws) x.v = -1e300;
    std::vector<int> ng(bs.size(), 0);
    std::vector<double> dm(bs.size(), 0), gm(bs.size(), 0);
#pragma omp parallel for collapse(2) schedule(dynamic, 16)
    for (int i = 0; i < n1; ++i) for (int j = 0; j < n2; ++j) {
      const double q = static_cast<double>(i)/(n1 - 1);
      const double r = logr ? std::exp(std::log(ra) + (std::log(rb) - std::log(ra))*q)
                            : ra + (rb - ra)*q;
      const double th = t0 + (t1 - t0)*j/(n2 - 1);
      Conf c = AtPolar(t, r, th, 0.0, r1, r2);
      const int me = omp_get_thread_num();
      const double v = 4*r*r*c.c0;
      if (c.c0 < 0) ng[me]++;
      dm[me] = std::fmax(dm[me], c.detm1);
      gm[me] = std::fmax(gm[me], c.gtrace);
      if (v < bs[me].v) bs[me] = {v, r, th, c.c0, c.rt, c.psi};
      const double vu = (r < r2) ? 4*r*r*AtPolar(t, r, th).c0 : v;
      um[me] = std::fmin(um[me], vu);
      if (vu - v > ws[me].v) ws[me] = {vu - v, r, th, 0, 0, 0};
    }
    for (size_t m = 0; m < bs.size(); ++m) {
      if (bs[m].v < b.v) b = bs[m];
      nneg += ng[m];
      dmax = std::fmax(dmax, dm[m]);
      gmax = std::fmax(gmax, gm[m]);
      if (ws[m].v > wst.v) wst = ws[m];
      umin = std::fmin(umin, um[m]);
    }
  };
  struct Reg { const char *name; bool logr; double ra, rb; int n1; };
  const Reg regs[3] = {{"r_f<r1   ", true, rmin*(1 + 1e-12), r1, 64},
                       {"r1<=r_f<r2", false, r1, r2, nrt},
                       {"r_f>=r2  ", true, r2, rmax, 1024}};
  std::printf("BLEND a %.3f r_AH,max %.6f f1 %.3f f2 %.3f: r1 %.6f r2 %.6f; D4 floor needs "
              "r_AH,min from F1\n", a, rahmax, f1, f2, r1, r2);
  for (const Reg &g : regs) {
    Best b, wst;
    wst.v = -1e300;
    int nneg = 0;
    double dmax = 0, gmax = 0, umin = 1e300;
    scan(g.logr, g.ra, g.rb, 0, M_PI, g.n1, nth, b, nneg, dmax, gmax, wst, umin);
    const int ncoarse = nneg;
    double dq = 1.0/(g.n1 - 1), dt = M_PI/(nth - 1);
    for (int pass = 0; pass < 2; ++pass) {
      int dummy = 0;
      double d1 = 0, d2 = 0, u1 = 1e300;
      Best w1;
      w1.v = -1e300;
      double lo, hi;
      if (g.logr) {
        const double L = std::log(g.rb/g.ra);
        lo = std::fmax(g.ra, b.r*std::exp(-2*dq*L));
        hi = std::fmin(g.rb, b.r*std::exp(2*dq*L));
      } else {
        lo = std::fmax(g.ra, b.r - 2*dq*(g.rb - g.ra));
        hi = std::fmin(g.rb, b.r + 2*dq*(g.rb - g.ra));
      }
      scan(g.logr, lo, hi, std::fmax(0.0, b.th - 2*dt), std::fmin(M_PI, b.th + 2*dt), 33, 33,
           b, dummy, d1, d2, w1, u1);
      dq /= 8; dt /= 8;
    }
    std::printf("BHARDY a %.3f %s: min 4 r_f^2 c0 = %+.6e at r_f %.6e (r_f/r_AH,max %.4f) "
                "th %.6f (c0 %+.4e, Rt/8 %+.4e, 7/8 psi^-8 A^2 %+.4e); c0 < 0 at %d of %d; "
                "max|det gt - 1| %.2e max|Gt^k_ki| %.2e; unblended min %+.6e; largest drop "
                "by the blend (unblended - blended) %+.3e at r_f/r_AH,max %.4f th %.4f [info; gate is A3]\n", a,
                g.name, b.v, b.r, b.r/rahmax, b.th, b.c0, b.rt/8, b.c0 - b.rt/8, ncoarse,
                g.n1*nth, dmax, gmax, umin, wst.v, wst.r/rahmax, wst.th);
  }
  if (t.fallback_count() > 0) std::printf("WARNING fallback used %d times\n",
                                          static_cast<int>(t.fallback_count()));
  return 0;
}
}  // namespace

// K-004 A6: light-cone speed bound per direction, lambda_i = |beta^i + xi^i| + alpha sqrt(g^ii)
// (the bound RK-013 asks newdt to use), on evolved cells: r >= RCUT and alpha >= ACUT.
// The D3 blend acts only at r < r2 < 0.75 r_AH,min, inside the D4 excision, so the physical
// metric of every evolved cell is the table's own (no blend needed here).
// Reports lambda_max over RCUT <= r <= RMAX, its location, and max over shells of a log grid.
int RunDt(const char *tab, double rcut, double acut, double rmax, double rbox,
          const double xi[3], int nr, int nth, int nph) {
  kstab::Table t(tab);
  std::vector<double> shell(nr, 0.0), shr(nr);
  struct Best { double lam = 0, r = 0, th = 0, ph = 0, al = 0, b = 0; int dir = -1; };
  std::vector<Best> best(nr);
  for (int ir = 0; ir < nr; ++ir)
    shr[ir] = rcut*std::pow(rmax/rcut, nr > 1 ? double(ir)/(nr - 1) : 0.0);
#pragma omp parallel for schedule(dynamic)
  for (int ir = 0; ir < nr; ++ir) {
    const double r = shr[ir];
    Best bb;
    for (int it = 0; it < nth; ++it) {
      const double th = M_PI*it/(nth - 1);
      for (int ip = 0; ip < nph; ++ip) {
        const double ph = 2*M_PI*ip/nph;
        double f[16];
        t.eval(r*std::sin(th)*std::cos(ph), r*std::sin(th)*std::sin(ph), r*std::cos(th), f);
        if (f[12] < acut) continue;
        double g[3][3];
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) g[i][j] = f[S(i, j)];
        const double D = g[0][0]*(g[1][1]*g[2][2] - g[1][2]*g[2][1])
                       - g[0][1]*(g[1][0]*g[2][2] - g[1][2]*g[2][0])
                       + g[0][2]*(g[1][0]*g[2][1] - g[1][1]*g[2][0]);
        const double gii[3] = {(g[1][1]*g[2][2] - g[1][2]*g[2][1])/D,
                               (g[0][0]*g[2][2] - g[0][2]*g[2][0])/D,
                               (g[0][0]*g[1][1] - g[0][1]*g[1][0])/D};
        for (int i = 0; i < 3; ++i) {
          const double lam = std::fabs(f[13 + i] + xi[i]) + f[12]*std::sqrt(gii[i]);
          if (lam > bb.lam) {
            bb.lam = lam; bb.r = r; bb.th = th; bb.ph = ph; bb.dir = i;
            bb.al = f[12]*std::sqrt(gii[i]); bb.b = std::fabs(f[13 + i] + xi[i]);
          }
        }
      }
    }
    best[ir] = bb;
    shell[ir] = bb.lam;
  }
  Best all, out;
  for (int ir = 0; ir < nr; ++ir) {
    if (best[ir].lam > all.lam) all = best[ir];
    if (shr[ir] >= rbox && best[ir].lam > out.lam) out = best[ir];
  }
  std::printf("DT tab=%s rcut=%.6f acut=%.3f rmax=%.3f xi=(%.3f,%.3f,%.3f)\n", tab, rcut, acut,
              rmax, xi[0], xi[1], xi[2]);
  std::printf("DT lam_max=%.6f at r=%.6f th=%.4f ph=%.4f dir=%d (|b+xi|=%.4f a*sqrt(g^ii)=%.4f)\n",
              all.lam, all.r, all.th, all.ph, all.dir, all.b, all.al);
  std::printf("DT lam_max(r>=rbox=%.4f)=%.6f at r=%.6f th=%.4f dir=%d\n", rbox, out.lam, out.r,
              out.th, out.dir);
  const int step = std::getenv("DT_ALL") ? 1 : (nr > 24 ? nr/24 : 1);
  for (int ir = 0; ir < nr; ir += step)
    std::printf("DTPROF r=%.6f lam_shell=%.6f\n", shr[ir], shell[ir]);
  return 0;
}

int main(int argc, char **argv) {
  if (argc >= 7 && std::string(argv[1]) == "blend")
    return RunBlend(argv[2], std::atof(argv[3]), std::atof(argv[4]), std::atof(argv[5]),
                    std::atof(argv[6]), argc > 7 ? std::atoi(argv[7]) : 200,
                    argc > 8 ? std::atoi(argv[8]) : 257);
  if (argc >= 7 && std::string(argv[1]) == "dt") {
    const double xi[3] = {argc > 7 ? std::atof(argv[7]) : 0.0, argc > 8 ? std::atof(argv[8]) : 0.0,
                          argc > 9 ? std::atof(argv[9]) : 0.0};
    return RunDt(argv[2], std::atof(argv[3]), std::atof(argv[4]), std::atof(argv[5]),
                 std::atof(argv[6]), xi, argc > 10 ? std::atoi(argv[10]) : 240,
                 argc > 11 ? std::atoi(argv[11]) : 65, argc > 12 ? std::atoi(argv[12]) : 64);
  }
  if (argc >= 4 && std::string(argv[1]) == "f4") return F4Check(argv[2], argv[3]);
  if (argc >= 4 && std::string(argv[1]) == "run")
    return Run(argv[2], std::atof(argv[3]), argc > 4 ? std::atoi(argv[4]) : 1024,
               argc > 5 ? std::atoi(argv[5]) : 257);
  std::fprintf(stderr, "usage: %s f4 KSTAB F4 | run KSTAB A [NR NTH] | blend KSTAB A RAHMAX "
               "F1 F2 [NRT NTH] | dt KSTAB RCUT ACUT RMAX RBOX [XI1 XI2 XI3 NR NTH NPH]\n", argv[0]);
  return 2;
}
