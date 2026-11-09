//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file test_kerr_bg_device.cpp
//! \brief V0 (K-001): the Kerr background port (src/cfc/kerr_background.hpp, evaluated in
//! Kokkos kernels on the device) against the KSTAB table's own spectral evaluation
//! (kslice/kstab/kstab.hpp eval_d on the host: the exact reference). No evolution.
//!   (a) host eval_d -> KerrConformal at P1-P4 against F4 and the refvalues; the device's
//!       spectral path against host eval_d;
//!   (b) the device interpolation at P1-P4 against F4, per family;
//!   (c) random points per radial band: interpolation errors per family (incl. ddg, dK,
//!       d alpha, d beta, Rt, Gt), device interp against host eval_d;
//!   (d) per band: CTS a1, d_k beta^k + 6 beta.grad ln psi, Dt_j Ahat^ij, det gt - 1,
//!       Gt^k_ki, on the port and on the exact values;
//!   (e) axisymmetry (rotated points, rotated back);
//!   (f) r_f > 1e3 (F2's edge): the jump across it, and the device spectral path far out;
//!   (g) GPU cost per cell;
//!   (s) a = 0 only: against the Schwarzschild trumpet of cfc_puncture.hpp.
//! kstab.hpp is NOT part of AthenaK (private spinning_trumpet repository): it is found
//! through -I at compile time (run_test_kerr_bg_device.sh). Not a CMake target.
//!
//!   test_kerr_bg_device F2_PREFIX KSTAB_TXT [F4_TXT|-] [REFVALUES_TXT|-] [NRAND]
//! Exit code = number of failed checks (thresholds documented at each check).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <typeinfo>
#include <vector>

#include <Kokkos_Core.hpp>

#include "athena.hpp"
#include "cfc/kerr_background.hpp"
#include "cfc/cfc_puncture.hpp"
#include "kstab.hpp"

using cfc::KerrTable;
using cfc::KerrBgPoint;
namespace ki = cfc::kerr_impl;

namespace {

constexpr int NV = cfc::kKnNV;
constexpr int NB = cfc::kKbgN;
int g_fail = 0;

void Check(bool ok, const char *what, double val, double tol) {
  if (!ok) {
    ++g_fail;
    std::printf("  FAIL %s: %.3e > %.1e\n", what, val, tol);
  }
}

// ---- families
struct Fam { const char *name; int lo, hi, slo, shi; };
// node families (scale = the family itself)
const Fam kNodeFam[9] = {{"g", 0, 6, 0, 6}, {"K", 6, 12, 6, 12}, {"alpha", 12, 13, 12, 13},
                         {"beta", 13, 16, 13, 16}, {"dg", 16, 34, 16, 34},
                         {"ddg", 34, 70, 34, 70}, {"dK", 70, 88, 70, 88},
                         {"dalpha", 88, 91, 88, 91}, {"dbeta", 91, 100, 91, 100}};
// conformal families (packed kKbg* offsets); Rt is scaled by max|Rt_ij|
const Fam kConfFam[14] = {{"psi0", 0, 1, 0, 1}, {"alpha0", 1, 2, 1, 2},
                          {"beta0", 2, 5, 2, 5}, {"gt", 5, 11, 5, 11},
                          {"gtu", 11, 17, 11, 17}, {"dgt", 17, 35, 17, 35},
                          {"Gt", 35, 53, 35, 53}, {"Rt_ij", 53, 59, 53, 59},
                          {"Rt", 59, 60, 53, 59}, {"dalpha0", 60, 63, 60, 63},
                          {"dbeta0", 63, 72, 63, 72}, {"dlnpsi0", 72, 75, 72, 75},
                          {"Ahat^ij", 81, 87, 81, 87}, {"Ahat_ij", 87, 93, 87, 93}};

double FamRel(const double *a, const double *b, const Fam &f) {
  double d = 0, s = 0;
  for (int q = f.lo; q < f.hi; ++q) d = std::max(d, std::fabs(a[q] - b[q]));
  for (int q = f.slo; q < f.shi; ++q) s = std::max(s, std::fabs(b[q]));
  return d/std::max(s, 1e-300);
}

KOKKOS_INLINE_FUNCTION
void Pack(const KerrBgPoint &b, Real *o) {
  o[cfc::kKbgPsi0] = b.psi0;
  o[cfc::kKbgAlpha0] = b.alpha0;
  o[cfc::kKbgRs] = b.rs;
  for (int a = 0; a < 3; ++a) {
    o[cfc::kKbgBeta0 + a] = b.beta0[a];
    o[cfc::kKbgDAlpha0 + a] = b.dalpha0[a];
    o[cfc::kKbgDLnPsi0 + a] = b.dlnpsi0[a];
    for (int c = 0; c < 3; ++c) o[cfc::kKbgDBeta0 + 3*a + c] = b.dbeta0[a][c];
    for (int q = 0; q < 6; ++q) {
      o[cfc::kKbgDGt + 6*a + q] = b.dgt[a][q];
      o[cfc::kKbgGam + 6*a + q] = b.gam[a][q];
    }
  }
  for (int q = 0; q < 6; ++q) {
    o[cfc::kKbgGt + q] = b.gt[q];
    o[cfc::kKbgGtu + q] = b.gtu[q];
    o[cfc::kKbgRt + q] = b.rt[q];
    o[cfc::kKbgK0 + q] = b.k0[q];
    o[cfc::kKbgAhu + q] = b.ahu[q];
    o[cfc::kKbgAhd + q] = b.ahd[q];
  }
}

// ---- the exact reference: kstab eval_d -> the node layout
void RefNode(const kstab::Table &T, double x, double y, double z, double *v) {
  double out[16], d1[3][16], d2[6][16];
  T.eval_d(x, y, z, out, d1, d2);
  for (int q = 0; q < 16; ++q) v[q] = out[q];
  for (int k = 0; k < 3; ++k) {
    for (int q = 0; q < 6; ++q) {
      v[cfc::kKnDG + 6*k + q] = d1[k][q];
      v[cfc::kKnDK + 6*k + q] = d1[k][6 + q];
    }
    v[cfc::kKnDA + k] = d1[k][12];
    for (int i = 0; i < 3; ++i) v[cfc::kKnDB + 3*k + i] = d1[k][13 + i];
  }
  for (int m = 0; m < 6; ++m)
    for (int q = 0; q < 6; ++q) v[cfc::kKnDDG + 6*m + q] = d2[m][q];
}

void HostConf(const double *v, double *o) {
  KerrBgPoint b;
  cfc::KerrConformal(v, b);
  Pack(b, o);
}

// ---- device evaluation of n points: nodes (n x NV) and packed conformal (n x NB)
using V2 = Kokkos::View<Real**, Kokkos::LayoutRight, DevMemSpace>;
void DeviceEval(const KerrTable &kt, const std::vector<double> &pts, bool exact,
                std::vector<double> &node, std::vector<double> &conf, Real m_bh = 1.0) {
  const int n = static_cast<int>(pts.size()/3);
  V2 p("p", n, 3), vn("vn", n, NV), vc("vc", n, NB);
  auto hp = Kokkos::create_mirror_view(p);
  for (int i = 0; i < n; ++i)
    for (int a = 0; a < 3; ++a) hp(i, a) = pts[3*i + a];
  Kokkos::deep_copy(p, hp);
  const KerrTable t = kt;
  Kokkos::parallel_for("kerr_v0_eval", Kokkos::RangePolicy<DevExeSpace>(0, n),
  KOKKOS_LAMBDA(const int i) {
    Real v[NV];
    cfc::KerrNodeCartesian(t, m_bh, p(i, 0), p(i, 1), p(i, 2), v, exact);
    for (int q = 0; q < NV; ++q) vn(i, q) = v[q];
    KerrBgPoint b;
    cfc::KerrConformal(v, b);
    Real o[NB];
    Pack(b, o);
    for (int q = 0; q < NB; ++q) vc(i, q) = o[q];
  });
  auto hn = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vn);
  auto hc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vc);
  node.assign(static_cast<size_t>(n)*NV, 0.0);
  conf.assign(static_cast<size_t>(n)*NB, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int q = 0; q < NV; ++q) node[static_cast<size_t>(i)*NV + q] = hn(i, q);
    for (int q = 0; q < NB; ++q) conf[static_cast<size_t>(i)*NB + q] = hc(i, q);
  }
}

// ---- (d) identities of a node (host), all relative
struct Ident { double cts, divb, momc, det, gtr; };
Ident Identities(const double *v) {
  KerrBgPoint b;
  cfc::KerrConformal(v, b);
  double gtu[3][3], gt[3][3], G[3][3][3], Db[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      gtu[i][j] = b.gtu[ki::Sym(i, j)];
      gt[i][j] = b.gt[ki::Sym(i, j)];
      for (int k = 0; k < 3; ++k) G[k][i][j] = b.gam[k][ki::Sym(i, j)];
    }
  for (int k = 0; k < 3; ++k)
    for (int j = 0; j < 3; ++j) {
      double s = b.dbeta0[k][j];
      for (int l = 0; l < 3; ++l) s += G[j][k][l]*b.beta0[l];
      Db[k][j] = s;
    }
  const double divDb = Db[0][0] + Db[1][1] + Db[2][2];
  const double ip6 = std::pow(b.psi0, -6.0);
  double dmax = 0, rmax = 0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      double l = -2.0/3.0*gtu[i][j]*divDb;
      for (int k = 0; k < 3; ++k) l += gtu[i][k]*Db[k][j] + gtu[j][k]*Db[k][i];
      const double r = 2.0*b.alpha0*ip6*b.ahu[ki::Sym(i, j)];
      dmax = std::max(dmax, std::fabs(l - r));
      rmax = std::max(rmax, std::fabs(r));
    }
  Ident id;
  id.cts = dmax/std::max(rmax, 1e-300);
  double res = 0, sc = 0;
  for (int k = 0; k < 3; ++k) {
    res += b.dbeta0[k][k] + 6.0*b.beta0[k]*b.dlnpsi0[k];
    sc += std::fabs(b.dbeta0[k][k]) + 6.0*std::fabs(b.beta0[k]*b.dlnpsi0[k]);
  }
  id.divb = std::fabs(res)/std::max(sc, 1e-300);
  // momentum constraint: d_m Ahat^ij from dg, dK and d ln psi (as conf_spectral)
  double g[3][3], gu[3][3], K[3][3], dg[3][3][3], dK[3][3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      g[i][j] = v[ki::Sym(i, j)];
      K[i][j] = v[6 + ki::Sym(i, j)];
      for (int m = 0; m < 3; ++m) {
        dg[m][i][j] = v[cfc::kKnDG + 6*m + ki::Sym(i, j)];
        dK[m][i][j] = v[cfc::kKnDK + 6*m + ki::Sym(i, j)];
      }
    }
  const double det = g[0][0]*(g[1][1]*g[2][2] - g[1][2]*g[2][1])
                   - g[0][1]*(g[1][0]*g[2][2] - g[1][2]*g[2][0])
                   + g[0][2]*(g[1][0]*g[2][1] - g[1][1]*g[2][0]);
  gu[0][0] = (g[1][1]*g[2][2] - g[1][2]*g[2][1])/det;
  gu[0][1] = (g[0][2]*g[2][1] - g[0][1]*g[2][2])/det;
  gu[0][2] = (g[0][1]*g[1][2] - g[0][2]*g[1][1])/det;
  gu[1][1] = (g[0][0]*g[2][2] - g[0][2]*g[2][0])/det;
  gu[1][2] = (g[0][2]*g[1][0] - g[0][0]*g[1][2])/det;
  gu[2][2] = (g[0][0]*g[1][1] - g[0][1]*g[1][0])/det;
  gu[1][0] = gu[0][1]; gu[2][0] = gu[0][2]; gu[2][1] = gu[1][2];
  double Ku[3][3], dgu[3][3][3], dA[3][3][3], A[3][3];
  const double p10 = std::pow(b.psi0, 10.0);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      double s = 0;
      for (int a = 0; a < 3; ++a)
        for (int c = 0; c < 3; ++c) s += gu[i][a]*gu[j][c]*K[a][c];
      Ku[i][j] = s;
      A[i][j] = p10*s;
      for (int m = 0; m < 3; ++m) {
        double t = 0;
        for (int a = 0; a < 3; ++a)
          for (int c = 0; c < 3; ++c) t -= gu[i][a]*dg[m][a][c]*gu[c][j];
        dgu[m][i][j] = t;
      }
    }
  double dAmax = 0;
  for (int m = 0; m < 3; ++m)
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        double s = 0;
        for (int a = 0; a < 3; ++a)
          for (int c = 0; c < 3; ++c)
            s += dgu[m][i][a]*gu[j][c]*K[a][c] + gu[i][a]*dgu[m][j][c]*K[a][c]
               + gu[i][a]*gu[j][c]*dK[m][a][c];
        dA[m][i][j] = p10*(10.0*b.dlnpsi0[m]*Ku[i][j] + s);
        dAmax = std::max(dAmax, std::fabs(dA[m][i][j]));
      }
  double mc = 0;
  for (int i = 0; i < 3; ++i) {
    double s = 0;
    for (int j = 0; j < 3; ++j) {
      s += dA[j][i][j];
      for (int k = 0; k < 3; ++k) s += G[i][j][k]*A[k][j] + G[j][j][k]*A[i][k];
    }
    mc = std::max(mc, std::fabs(s));
  }
  id.momc = mc/std::max(dAmax, 1e-300);
  const double dt = gt[0][0]*(gt[1][1]*gt[2][2] - gt[1][2]*gt[2][1])
                  - gt[0][1]*(gt[1][0]*gt[2][2] - gt[1][2]*gt[2][0])
                  + gt[0][2]*(gt[1][0]*gt[2][1] - gt[1][1]*gt[2][0]);
  id.det = std::fabs(dt - 1.0);
  double gmax = 0, tr = 0;
  for (int i = 0; i < 3; ++i) {
    double s = 0;
    for (int k = 0; k < 3; ++k) s += G[k][k][i];
    tr = std::max(tr, std::fabs(s));
    for (int k = 0; k < 3; ++k)
      for (int j = 0; j < 3; ++j) gmax = std::max(gmax, std::fabs(G[k][i][j]));
  }
  id.gtr = tr/std::max(gmax, 1e-300);
  return id;
}

// ---- "Pn name value" files (F4, refvalues); lines not starting with P1..P4 are skipped
struct PointFile { std::map<std::string, double> val[4]; double x[4][3]; bool ok = false; };
PointFile ReadPoints(const std::string &path) {
  PointFile pf;
  std::ifstream in(path);
  if (!in) return pf;
  std::string line;
  while (std::getline(in, line)) {
    if (line.size() < 3 || line[0] != 'P' || line[1] < '1' || line[1] > '4' || line[2] != ' ')
      continue;
    const int p = line[1] - '1';
    std::istringstream ss(line.substr(3));
    std::string name;
    ss >> name;
    if (name == "r_f" || name == "x") {
      std::string k1, k2; double a, b, c;
      ss >> a >> k1 >> b >> k2 >> c;
      pf.val[p][name == "r_f" ? "r_f" : "x"] = a;
      pf.val[p][name == "r_f" ? "th" : "y"] = b;
      pf.val[p][name == "r_f" ? "phi" : "z"] = c;
    } else {
      double a;
      if (ss >> a) pf.val[p][name] = a;
    }
  }
  for (int p = 0; p < 4; ++p) {
    const double r = pf.val[p]["r_f"], th = pf.val[p]["th"], ph = pf.val[p]["phi"];
    pf.x[p][0] = r*std::sin(th)*std::cos(ph);
    pf.x[p][1] = r*std::sin(th)*std::sin(ph);
    pf.x[p][2] = r*std::cos(th);
  }
  pf.ok = true;
  return pf;
}

const char *kXYZ[3] = {"x", "y", "z"};
std::string PairName(int q) {
  return std::string(kXYZ[ki::IdxI(q)]) + kXYZ[ki::IdxJ(q)];
}

// the F4 quantities, as a vector in the order (psi, gt 6, Gt 18, Rt_ij 6, Rt, Ahat^ 6,
// Ahat_ 6), from a packed conformal point or the F4 file
const char *kF4Fam[7] = {"psi", "gt", "Gt", "Rt_ij", "Rt", "Ahat^ij", "Ahat_ij"};
const int kF4Lo[8] = {0, 1, 7, 25, 31, 32, 38, 44};
void F4FromConf(const double *c, double *o) {
  o[0] = c[cfc::kKbgPsi0];
  for (int q = 0; q < 6; ++q) {
    o[1 + q] = c[cfc::kKbgGt + q];
    o[25 + q] = c[cfc::kKbgRt + q];
    o[32 + q] = c[cfc::kKbgAhu + q];
    o[38 + q] = c[cfc::kKbgAhd + q];
  }
  for (int k = 0; k < 3; ++k)
    for (int q = 0; q < 6; ++q) o[7 + 6*k + q] = c[cfc::kKbgGam + 6*k + q];
  o[31] = c[cfc::kKbgRs];
}
bool F4FromFile(std::map<std::string, double> &m, double *o) {
  auto get = [&](const std::string &k, double *d) {
    auto it = m.find(k);
    if (it == m.end()) return false;
    *d = it->second;
    return true;
  };
  bool ok = get("psi", &o[0]) && get("Rt", &o[31]);
  for (int q = 0; q < 6; ++q) {
    const std::string pn = PairName(q);
    ok = ok && get("gt_" + pn, &o[1 + q]) && get("Rt_" + pn, &o[25 + q])
            && get("Ahat^" + pn, &o[32 + q]) && get("Ahat_" + pn, &o[38 + q]);
    for (int k = 0; k < 3; ++k)
      ok = ok && get(std::string("Gt^") + kXYZ[k] + "_" + pn, &o[7 + 6*k + q]);
  }
  return ok;
}
double F4FamRel(const double *a, const double *b, int f) {
  // Rt (f = 4) is scaled by max|Rt_ij| (as F4's own checks)
  const int lo = kF4Lo[f], hi = kF4Lo[f + 1];
  const int slo = (f == 4) ? kF4Lo[3] : lo, shi = (f == 4) ? kF4Lo[4] : hi;
  double d = 0, s = 0;
  for (int q = lo; q < hi; ++q) d = std::max(d, std::fabs(a[q] - b[q]));
  for (int q = slo; q < shi; ++q) s = std::max(s, std::fabs(b[q]));
  return d/std::max(s, 1e-300);
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s F2_PREFIX KSTAB_TXT [F4|-] [REFVALUES|-] [NRAND]\n",
                 argv[0]);
    return 2;
  }
  const std::string f2 = argv[1], ks = argv[2];
  const std::string f4 = (argc > 3) ? argv[3] : "-", rv = (argc > 4) ? argv[4] : "-";
  const int nrand = (argc > 5) ? std::atoi(argv[5]) : 4000;
  Kokkos::initialize(argc, argv);
  {
    kstab::Table T(ks);
    KerrTable kt;
    cfc::LoadKerrTable(f2, ks, kt);
    std::printf("# V0 (K-001): F2 %s\n#   KSTAB %s\n#   a = %.6g r_min = %.6e r_max = %.6g "
                "nr = %d nth = %d nseg = %d kmax = %d p = (%g %g %g %g)\n", f2.c_str(),
                ks.c_str(), kt.a, kt.rmin, kt.rmax, kt.nr, kt.nth, kt.nseg, kt.kmax,
                kt.p[0], kt.p[1], kt.p[2], kt.p[3]);
    std::printf("# device: %s\n", typeid(DevExeSpace).name());
    std::vector<double> dn, dc, en, ec;

    // ---------------- (a), (b): P1-P4
    PointFile F4 = (f4 != "-") ? ReadPoints(f4) : PointFile();
    PointFile RV = (rv != "-") ? ReadPoints(rv) : PointFile();
    PointFile &PP = RV.ok ? RV : F4;
    if (PP.ok) {
      std::printf("\n## (a)/(b) P1-P4  [famrel = max|diff| over the family / max|ref|]\n");
      std::vector<double> pts;
      for (int p = 0; p < 4; ++p)
        for (int a = 0; a < 3; ++a) pts.push_back(PP.x[p][a]);
      DeviceEval(kt, pts, false, dn, dc);   // interpolated port
      DeviceEval(kt, pts, true, en, ec);    // device spectral
      for (int p = 0; p < 4; ++p) {
        double vref[NV], cref[NB];
        RefNode(T, PP.x[p][0], PP.x[p][1], PP.x[p][2], vref);
        HostConf(vref, cref);
        std::printf("P%d r_f %.3g th %.4f phi %.3f\n", p + 1, PP.val[p]["r_f"],
                    PP.val[p]["th"], PP.val[p]["phi"]);
        if (RV.ok) {   // (a) host eval_d vs the refvalues (fields + psi, gt)
          static const char *fn[16] = {"gxx", "gxy", "gxz", "gyy", "gyz", "gzz", "Kxx",
                                       "Kxy", "Kxz", "Kyy", "Kyz", "Kzz", "alpha", "beta^x",
                                       "beta^y", "beta^z"};
          double r16[NV] = {0}, ok = 1;
          for (int q = 0; q < 16; ++q) {
            auto it = RV.val[p].find(fn[q]);
            if (it == RV.val[p].end()) ok = 0; else r16[q] = it->second;
          }
          double w = 0;
          for (int f = 0; f < 4; ++f) w = std::max(w, FamRel(vref, r16, kNodeFam[f]));
          double ps = std::fabs(cref[0] - RV.val[p]["psi"])/RV.val[p]["psi"];
          std::printf("  (a) host eval_d vs refvalues: fields max famrel %.2e, psi %.2e%s\n",
                      w, ps, ok ? "" : " (missing names)");
          Check(ok && w <= 1e-12 && ps <= 1e-12, "(a) refvalues", std::max(w, ps), 1e-12);
        }
        double fhost[45], fdev[45], fex[45], ffile[45];
        F4FromConf(cref, fhost);
        F4FromConf(&dc[static_cast<size_t>(p)*NB], fdev);
        F4FromConf(&ec[static_cast<size_t>(p)*NB], fex);
        if (F4.ok && F4FromFile(F4.val[p], ffile)) {
          std::printf("  %-8s %10s %10s %10s\n", "family", "(a)host", "(b)interp", "devspec");
          for (int f = 0; f < 7; ++f) {
            const double ea = F4FamRel(fhost, ffile, f), eb = F4FamRel(fdev, ffile, f),
                         es = F4FamRel(fex, ffile, f);
            std::printf("  %-8s %10.2e %10.2e %10.2e\n", kF4Fam[f], ea, eb, es);
            Check(ea <= 1e-12, "(a) host conformal vs F4", ea, 1e-12);
            Check(es <= 1e-12, "(a) device spectral vs F4", es, 1e-12);
            Check(eb <= 1e-6, "(b) interp vs F4 (loose bound; budgets in K-001)", eb, 1e-6);
          }
        }
        double wn = 0;
        for (int f = 0; f < 9; ++f)
          wn = std::max(wn, FamRel(&en[static_cast<size_t>(p)*NV], vref, kNodeFam[f]));
        std::printf("  device spectral node vs host eval_d: max famrel %.2e\n", wn);
        Check(wn <= 1e-12, "(a) device spectral node vs host", wn, 1e-12);
      }
    }

    // ---------------- (c), (d): random points per band
    const double rmin = kt.rmin;
    std::vector<double> edges = {rmin, 1e-3, 1e-2, 0.1, 0.5, 1, 3, 10, 100, 1e3};
    std::mt19937_64 rng(20261109);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    auto rand_pts = [&](double r0, double r1, int n, std::vector<double> &pts) {
      pts.clear();
      for (int i = 0; i < n; ++i) {
        const double r = std::exp(std::log(r0) + (std::log(r1) - std::log(r0))*U(rng));
        const double ct = 2*U(rng) - 1, st = std::sqrt(std::max(0.0, 1 - ct*ct));
        const double ph = 2*M_PI*U(rng);
        pts.push_back(r*st*std::cos(ph)); pts.push_back(r*st*std::sin(ph));
        pts.push_back(r*ct);
      }
    };
    std::printf("\n## (c) interpolation (device) vs host eval_d, %d random points per band "
                "(log r, uniform cos th, phi); per-point famrel: max / mean\n", nrand);
    std::printf("%-21s", "band");
    for (int f = 0; f < 9; ++f) std::printf(" %9s", kNodeFam[f].name);
    for (int f : {3, 6, 7, 8, 11, 12}) std::printf(" %9s", kConfFam[f].name);
    std::printf("\n");
    std::vector<std::vector<double>> bandmax(edges.size() - 1);
    std::vector<std::array<double, 5>> idp(edges.size() - 1), ide(edges.size() - 1);
    for (size_t b = 0; b + 1 < edges.size(); ++b) {
      std::vector<double> pts, vref(NV), cref(NB);
      rand_pts(edges[b], edges[b + 1], nrand, pts);
      DeviceEval(kt, pts, false, dn, dc);
      std::vector<double> mx(15, 0), mean(15, 0);
      std::array<double, 5> ip = {0, 0, 0, 0, 0}, ie = {0, 0, 0, 0, 0};
      for (int i = 0; i < nrand; ++i) {
        RefNode(T, pts[3*i], pts[3*i + 1], pts[3*i + 2], vref.data());
        HostConf(vref.data(), cref.data());
        const double *vd = &dn[static_cast<size_t>(i)*NV];
        const double *cd = &dc[static_cast<size_t>(i)*NB];
        int c = 0;
        for (int f = 0; f < 9; ++f, ++c) {
          const double e = FamRel(vd, vref.data(), kNodeFam[f]);
          mx[c] = std::max(mx[c], e); mean[c] += e/nrand;
        }
        for (int f : {3, 6, 7, 8, 11, 12}) {
          const double e = FamRel(cd, cref.data(), kConfFam[f]);
          mx[c] = std::max(mx[c], e); mean[c] += e/nrand; ++c;
        }
        const Ident a = Identities(vd), e = Identities(vref.data());
        const double av[5] = {a.cts, a.divb, a.momc, a.det, a.gtr};
        const double ev[5] = {e.cts, e.divb, e.momc, e.det, e.gtr};
        for (int k = 0; k < 5; ++k) { ip[k] = std::max(ip[k], av[k]); ie[k] = std::max(ie[k], ev[k]); }
      }
      bandmax[b] = mx;
      idp[b] = ip; ide[b] = ie;
      char lab[64];
      std::snprintf(lab, sizeof(lab), "[%.1e,%.0e]", edges[b], edges[b + 1]);
      std::printf("%-21s", lab);
      for (int c = 0; c < 15; ++c) std::printf(" %9.2e", mx[c]);
      std::printf("\n%-21s", "   mean");
      for (int c = 0; c < 15; ++c) std::printf(" %9.2e", mean[c]);
      std::printf("\n");
    }
    std::printf("\n## (d) identities per band, max over the band's points: port (interp) | "
                "exact (eval_d)\n#  cts = max|Lt beta - 2 alpha psi^-6 Ahat| / max|rhs|; divb = "
                "|d_k beta^k + 6 beta.dlnpsi| / sum|terms|;\n#  momc = max|Dt_j Ahat^ij| / "
                "max|d Ahat|; det = |det gt - 1|; gtr = max|Gt^k_ki| / max|Gt|\n");
    std::printf("%-21s %9s %9s %9s %9s %9s | %9s %9s %9s %9s %9s\n", "band", "cts", "divb",
                "momc", "det", "gtr", "cts", "divb", "momc", "det", "gtr");
    for (size_t b = 0; b + 1 < edges.size(); ++b) {
      char lab[64];
      std::snprintf(lab, sizeof(lab), "[%.1e,%.0e]", edges[b], edges[b + 1]);
      std::printf("%-21s", lab);
      for (int k = 0; k < 5; ++k) std::printf(" %9.2e", idp[b][k]);
      std::printf(" |");
      for (int k = 0; k < 5; ++k) std::printf(" %9.2e", ide[b][k]);
      std::printf("\n");
      Check(idp[b][3] <= 1e-12, "(d) det gt - 1 (port)", idp[b][3], 1e-12);
    }

    // ---------------- (e) axisymmetry
    {
      const int n = 500, nph = 8;
      std::vector<double> pts0, pts;
      rand_pts(1e-3, 1e3, n, pts0);
      for (int i = 0; i < n; ++i) {   // put the base points on phi = 0
        const double rho = std::hypot(pts0[3*i], pts0[3*i + 1]);
        pts0[3*i] = rho; pts0[3*i + 1] = 0.0;
      }
      std::vector<double> d0n, d0c;
      DeviceEval(kt, pts0, false, d0n, d0c);
      double w[9] = {0}, ws = 0;
      for (int k = 1; k < nph; ++k) {
        const double ph = 2*M_PI*k/nph + 0.1, c = std::cos(ph), s = std::sin(ph);
        pts.clear();
        for (int i = 0; i < n; ++i) {
          pts.push_back(c*pts0[3*i]); pts.push_back(s*pts0[3*i]); pts.push_back(pts0[3*i + 2]);
        }
        DeviceEval(kt, pts, false, dn, dc);
        for (int i = 0; i < n; ++i) {
          double vb[NV];
          cfc::KerrRotateNode(c, -s, &dn[static_cast<size_t>(i)*NV], vb);
          for (int f = 0; f < 9; ++f)
            w[f] = std::max(w[f], FamRel(vb, &d0n[static_cast<size_t>(i)*NV], kNodeFam[f]));
          for (int q : {cfc::kKbgPsi0, cfc::kKbgAlpha0, cfc::kKbgRs}) {
            const double a0 = d0c[static_cast<size_t>(i)*NB + q];
            ws = std::max(ws, std::fabs(dc[static_cast<size_t>(i)*NB + q] - a0)
                              /std::max(std::fabs(a0), 1e-300));
          }
        }
      }
      std::printf("\n## (e) axisymmetry: %d points x %d azimuths, R_z(-phi) f(R_z(phi) x) vs "
                  "f(x), famrel max\n ", n, nph - 1);
      double wm = 0;
      for (int f = 0; f < 9; ++f) { std::printf(" %s %.1e", kNodeFam[f].name, w[f]); wm = std::max(wm, w[f]); }
      std::printf("\n  scalars psi0, alpha0, Rt: max rel %.1e\n", ws);
      Check(wm <= 1e-12, "(e) axisymmetry", wm, 1e-12);
    }

    // ---------------- (f) beyond F2's edge
    {
      const int n = 400;
      std::vector<double> pin, pout, base;
      rand_pts(1.0, 1.0, n, base);   // unit directions
      for (int i = 0; i < n; ++i)
        for (int a = 0; a < 3; ++a) {
          pin.push_back(base[3*i + a]*kt.rmax*(1 - 1e-13));
          pout.push_back(base[3*i + a]*kt.rmax*(1 + 1e-13));
        }
      std::vector<double> inn, inc, outn, outc;
      DeviceEval(kt, pin, false, inn, inc);
      DeviceEval(kt, pout, false, outn, outc);
      std::printf("\n## (f) r_f > r_max = %g: interp (1-1e-13) vs spectral (1+1e-13) jump, "
                  "famrel max over %d directions\n ", kt.rmax, n);
      for (int f = 0; f < 9; ++f) {
        double w = 0;
        for (int i = 0; i < n; ++i)
          w = std::max(w, FamRel(&inn[static_cast<size_t>(i)*NV],
                                 &outn[static_cast<size_t>(i)*NV], kNodeFam[f]));
        std::printf(" %s %.1e", kNodeFam[f].name, w);
      }
      std::printf("\n");
      for (auto rr : {std::make_pair(1e3, 1e4), std::make_pair(1e4, 1e5),
                      std::make_pair(1e5, 1e7)}) {
        std::vector<double> pts, vref(NV);
        rand_pts(rr.first, rr.second, n, pts);
        DeviceEval(kt, pts, false, dn, dc);
        double w = 0, psimax = 0;
        for (int i = 0; i < n; ++i) {
          RefNode(T, pts[3*i], pts[3*i + 1], pts[3*i + 2], vref.data());
          for (int f = 0; f < 9; ++f)
            w = std::max(w, FamRel(&dn[static_cast<size_t>(i)*NV], vref.data(), kNodeFam[f]));
          const double r = std::sqrt(pts[3*i]*pts[3*i] + pts[3*i + 1]*pts[3*i + 1]
                                     + pts[3*i + 2]*pts[3*i + 2]);
          psimax = std::max(psimax, std::fabs(dc[static_cast<size_t>(i)*NB] - 1.0)*r);
        }
        std::printf("  r_f in [%.0e, %.0e]: device (spectral) vs host eval_d, all families "
                    "max famrel %.2e; max |psi0 - 1| r_f = %.4f\n", rr.first, rr.second, w,
                    psimax);
        Check(w <= 1e-12, "(f) device spectral far field", w, 1e-12);
      }
    }

    // ---------------- (r) below r_min: the port vs kstab's fallback
    {
      std::vector<double> pts, vref(NV);
      rand_pts(1e-3*rmin, rmin*(1 - 1e-12), 500, pts);
      DeviceEval(kt, pts, false, dn, dc);
      std::printf("\n## (r) r_f < r_min (fallback): device vs host eval_d, famrel max\n ");
      for (int f = 0; f < 9; ++f) {
        double w = 0;
        for (int i = 0; i < 500; ++i) {
          RefNode(T, pts[3*i], pts[3*i + 1], pts[3*i + 2], vref.data());
          w = std::max(w, FamRel(&dn[static_cast<size_t>(i)*NV], vref.data(), kNodeFam[f]));
        }
        std::printf(" %s %.1e", kNodeFam[f].name, w);
      }
      std::printf("\n");
    }

    // ---------------- (s) a = 0: the Schwarzschild trumpet of cfc_puncture.hpp
    if (kt.a == 0.0) {
      std::vector<double> pts;
      std::printf("\n## (s) a = 0: device interp vs TrumpetBackground (cfc_puncture.hpp), "
                  "max rel (psi0, alpha0) and |beta0 diff|/|beta0|\n");
      for (size_t b = 1; b + 1 < edges.size(); ++b) {
        rand_pts(edges[b], edges[b + 1], 1000, pts);
        DeviceEval(kt, pts, false, dn, dc);
        double ep = 0, ea = 0, eb = 0;
        for (int i = 0; i < 1000; ++i) {
          const double x = pts[3*i], y = pts[3*i + 1], z = pts[3*i + 2];
          const double r = std::sqrt(x*x + y*y + z*z);
          double psi, alp, beta[3], A[6], a2;
          cfc::TrumpetBackground(1.0, x, y, z, cfc::TrumpetIsoToAreal(r), &psi, &alp, beta,
                                 A, &a2);
          const double *c = &dc[static_cast<size_t>(i)*NB];
          ep = std::max(ep, std::fabs(c[cfc::kKbgPsi0] - psi)/psi);
          ea = std::max(ea, std::fabs(c[cfc::kKbgAlpha0] - alp)/alp);
          const double bn = std::sqrt(beta[0]*beta[0] + beta[1]*beta[1] + beta[2]*beta[2]);
          double bd = 0;
          for (int a = 0; a < 3; ++a) bd += std::pow(c[cfc::kKbgBeta0 + a] - beta[a], 2);
          eb = std::max(eb, std::sqrt(bd)/std::max(bn, 1e-300));
        }
        std::printf("  [%.0e,%.0e]  psi0 %.2e  alpha0 %.2e  beta0 %.2e\n", edges[b],
                    edges[b + 1], ep, ea, eb);
      }
    }

    // ---------------- (g) cost per cell
    {
      const int n1 = 128, n = n1*n1*n1;
      V2 out("out", n, NB);
      Kokkos::View<Real*, DevMemSpace> o1("o1", n);
      const KerrTable t = kt;
      auto xyz = KOKKOS_LAMBDA(const int i, Real &x, Real &y, Real &z) {
        const int a = i % n1, bb = (i/n1) % n1, c = i/(n1*n1);
        x = -40.0 + 80.0*(a + 0.5)/n1; y = -40.0 + 80.0*(bb + 0.5)/n1;
        z = -40.0 + 80.0*(c + 0.5)/n1;
      };
      auto run = [&](int mode) {
        Kokkos::fence();
        Kokkos::Timer tm;
        if (mode == 0) {
          Kokkos::parallel_for("kerr_v0_node", Kokkos::RangePolicy<DevExeSpace>(0, n),
          KOKKOS_LAMBDA(const int i) {
            Real x, y, z, v[NV];
            xyz(i, x, y, z);
            cfc::KerrNodeCartesian(t, 1.0, x, y, z, v, false);
            Real s = 0;
            for (int q = 0; q < NV; ++q) s += v[q];
            o1(i) = s;
          });
        } else {
          const bool ex = (mode == 2);
          Kokkos::parallel_for("kerr_v0_full", Kokkos::RangePolicy<DevExeSpace>(0, n),
          KOKKOS_LAMBDA(const int i) {
            Real x, y, z, v[NV];
            xyz(i, x, y, z);
            cfc::KerrNodeCartesian(t, 1.0, x, y, z, v, ex);
            KerrBgPoint b;
            cfc::KerrConformal(v, b);
            Real o[NB];
            Pack(b, o);
            for (int q = 0; q < NB; ++q) out(i, q) = o[q];
          });
        }
        Kokkos::fence();
        return tm.seconds();
      };
      std::printf("\n## (g) GPU cost, %d^3 cells in [-40, 40]^3 M (one tile), ns per cell "
                  "(best of 3, after a warm-up)\n", n1);
      const char *lab[3] = {"interp nodes only", "interp + conformal + store 93",
                            "spectral + conformal + store 93"};
      for (int mode = 0; mode < 3; ++mode) {
        run(mode);
        double best = 1e30;
        for (int r = 0; r < 3; ++r) best = std::min(best, run(mode));
        std::printf("  %-32s %8.1f ns/cell  (%.3f s)\n", lab[mode], 1e9*best/n, best);
      }
    }
  }
  Kokkos::finalize();
  std::printf("\n# V0 checks failed: %d\n", g_fail);
  return g_fail;
}
