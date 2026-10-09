#pragma once
/*
 * kstab.hpp -- header-only reader/evaluator for KSTAB v1 tables (analysis_kslice/KSTAB_v1.md, section 5;
 * kslice KC-8). Kerr's stationary maximal trumpet slice as 3+1 data in Cartesian components; M = 1.
 *
 * Vendored verbatim from alanlam1002/spinning_trumpet, kslice/kstab/kstab.hpp (KC-8 a300e35; KC-14 adds the
 * amendment-A3 "azimuth" header line), for the z4c_kerr_trumpet problem generator (KC-9, KC-14).
 *
 * Coordinates: (x, y, z) = r_f (sin th cos phi, sin th sin phi, cos th), (r_f, th) the isothermal coordinates,
 * phi the azimuth (ingoing phi~, or A3 Gamt^phi = 0: the "azimuth" header line); spin along +z.  Fields (index order of KSTAB::eval):
 *   0..5  gxx gxy gxz gyy gyz gzz      6..11 Kxx Kxy Kxz Kyy Kyz Kzz      12 alp      13..15 betax betay betaz
 * On the half-plane phi = 0 each field is sum_n T_n(x) sum_k A_{n,k} cos k th (even family) or B_{n,k} sin k th
 * (odd family: gxz gyz Kxz Kyz betax betay), per radial segment ("log": x = 2 (ln r - ln lo)/(ln hi - ln lo) - 1;
 * "inv": x = 2 r_1/r_f - 1).  Coefficients absent from the file are zero.  At azimuth phi: tensors
 * R_z(phi) T R_z(phi)^T, vectors R_z(phi) v, scalars unchanged.  r_f < r_min: f(r_min, th) (r_f/r_min)^p per
 * family (asymp line), with a warning on the first use (counted in fallback_count()).
 *
 * Usage:  kstab::Table t("kstab_a0.60.txt");  double f[16];  t.eval(x, y, z, f);
 * No dependencies beyond the C++17 standard library.  Not thread-safe for the fallback counter only.
 */
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace kstab
{

inline const char* const FIELD_NAMES[16] = {"gxx", "gxy", "gxz", "gyy", "gyz", "gzz", "Kxx", "Kxy",
                                             "Kxz", "Kyy", "Kyz", "Kzz", "alp", "betax", "betay", "betaz"};
/** true: sin k th family (odd under R_z(pi)); false: cos k th */
inline bool odd_field(int f) { return f == 2 || f == 4 || f == 8 || f == 10 || f == 13 || f == 14; }
/** 0: g, 1: K, 2: alp, 3: beta */
inline int family(int f) { return f < 6 ? 0 : (f < 12 ? 1 : (f == 12 ? 2 : 3)); }

struct Segment {
    bool inv = false;
    double lo = 0, hi = 0;
    int N = 0;
};

class Table
{
  public:
    double a = 0, M = 1, C = 0, J = 0;
    double p[4] = {-2, 0, 0, 1}; // fallback exponents: g, K, alp, beta
    double rmin = 0, r1 = 0;
    int Kmax = 0;
    std::vector<Segment> seg;
    std::string azimuth = "ingoing"; // amendment A3: "gamt_phi0" when the header says so (KC-14)
    double azimuth_c = 0;            // the A3 winding c

    explicit Table(const std::string& path)
    {
        std::ifstream in(path);
        if (!in) throw std::runtime_error("kstab: cannot open " + path);
        std::string line;
        int stage = 0, nseg = 0;
        while (std::getline(in, line)) {
            const size_t h = line.find('#');
            if (h != std::string::npos) line.erase(h);
            std::istringstream ss(line);
            std::string tag;
            if (!(ss >> tag)) continue;
            if (stage == 0) {
                int ver = 0;
                ss >> ver;
                if (tag != "KSTAB" || ver != 1) throw std::runtime_error("kstab: not a KSTAB v1 file");
                stage = 1;
            } else if (tag == "a") {
                std::string k;
                ss >> a >> k >> M >> k >> C >> k >> J;
            } else if (tag == "asymp") {
                std::string k;
                ss >> k >> p[0] >> k >> p[1] >> k >> p[2] >> k >> p[3];
            } else if (tag == "azimuth") { // amendment A3: azimuth <name> c <c>
                std::string k;
                ss >> azimuth >> k >> azimuth_c;
            } else if (tag == "rmin") {
                std::string k;
                ss >> rmin >> k >> r1 >> k >> Kmax >> k >> nseg;
                seg.resize(nseg);
                coef_.assign(16, std::vector<std::vector<double>>(nseg));
            } else if (tag == "seg") {
                int i = 0;
                std::string type, hi;
                Segment s;
                ss >> i >> type >> s.lo >> hi >> s.N;
                s.inv = type == "inv";
                s.hi = s.inv ? INFINITY : std::stod(hi);
                if (i < 0 || i >= nseg) throw std::runtime_error("kstab: bad segment index");
                seg[i] = s;
                for (int f = 0; f < 16; f++) coef_[f][i].assign(static_cast<size_t>(s.N) * (Kmax + 1), 0.0);
            } else {
                int f = -1;
                for (int q = 0; q < 16; q++)
                    if (tag == FIELD_NAMES[q]) f = q;
                if (f < 0) throw std::runtime_error("kstab: unknown line '" + tag + "'");
                int i = 0, n = 0, k = 0;
                std::string cs;
                double v = 0;
                ss >> i >> n >> cs >> k >> v;
                if (ss.fail() || i < 0 || i >= nseg || n < 0 || n >= seg[i].N || k < 0 || k > Kmax)
                    throw std::runtime_error("kstab: bad coefficient line '" + line + "'");
                if ((cs == "s") != odd_field(f)) throw std::runtime_error("kstab: wrong Fourier family on '" + line + "'");
                coef_[f][i][static_cast<size_t>(n) * (Kmax + 1) + k] = v;
            }
        }
        if (seg.empty() || rmin <= 0) throw std::runtime_error("kstab: incomplete header");
    }

    /** the 16 fields at (r_f, th) on the half-plane phi = 0, th in [0, pi] */
    void eval_polar(double rf, double th, double out[16]) const
    {
        if (rf < rmin) {
            if (fallback_++ == 0)
                std::fprintf(stderr, "kstab WARNING: r_f = %.3e < r_min = %.3e: cylinder fallback f(r_min) (r_f/r_min)^p used\n", rf,
                             rmin);
            eval_polar(rmin, th, out);
            for (int f = 0; f < 16; f++) out[f] *= std::pow(rf / rmin, p[family(f)]);
            return;
        }
        int s = static_cast<int>(seg.size()) - 1;
        for (int i = 0; i < static_cast<int>(seg.size()); i++)
            if (!seg[i].inv && rf <= seg[i].hi) { s = i; break; }
        const Segment& S = seg[s];
        const double x = S.inv ? 2 * r1 / rf - 1 : 2 * (std::log(rf) - std::log(S.lo)) / (std::log(S.hi) - std::log(S.lo)) - 1;
        std::vector<double> T(S.N), ck(Kmax + 1), sk(Kmax + 1);
        T[0] = 1;
        if (S.N > 1) T[1] = x;
        for (int n = 2; n < S.N; n++) T[n] = 2 * x * T[n - 1] - T[n - 2];
        const double c1 = std::cos(th), s1 = std::sin(th);
        ck[0] = 1;
        sk[0] = 0;
        for (int k = 1; k <= Kmax; k++) {
            ck[k] = ck[k - 1] * c1 - sk[k - 1] * s1;
            sk[k] = sk[k - 1] * c1 + ck[k - 1] * s1;
        }
        for (int f = 0; f < 16; f++) {
            const std::vector<double>& c = coef_[f][s];
            const std::vector<double>& tr = odd_field(f) ? sk : ck;
            double v = 0;
            for (int n = 0; n < S.N; n++) {
                double w = 0;
                const double* row = &c[static_cast<size_t>(n) * (Kmax + 1)];
                for (int k = 0; k <= Kmax; k++) w += row[k] * tr[k];
                v += T[n] * w;
            }
            out[f] = v;
        }
    }

    /** the 16 fields at Cartesian (x, y, z) (rotated from phi = 0) */
    void eval(double x, double y, double z, double out[16]) const
    {
        const double rho = std::hypot(x, y), rf = std::hypot(rho, z);
        const double th = std::atan2(rho, z), ph = std::atan2(y, x);
        double f0[16];
        eval_polar(rf, th, f0);
        const double c = std::cos(ph), s = std::sin(ph);
        const double R[3][3] = {{c, -s, 0}, {s, c, 0}, {0, 0, 1}};
        const int ii[6] = {0, 0, 0, 1, 1, 2}, jj[6] = {0, 1, 2, 1, 2, 2};
        for (int t = 0; t < 2; t++) { // g, K
            double T0[3][3];
            for (int q = 0; q < 6; q++) T0[ii[q]][jj[q]] = T0[jj[q]][ii[q]] = f0[6 * t + q];
            for (int q = 0; q < 6; q++) {
                double v = 0;
                for (int m = 0; m < 3; m++)
                    for (int n = 0; n < 3; n++) v += R[ii[q]][m] * R[jj[q]][n] * T0[m][n];
                out[6 * t + q] = v;
            }
        }
        out[12] = f0[12];
        for (int i = 0; i < 3; i++) out[13 + i] = R[i][0] * f0[13] + R[i][1] * f0[14] + R[i][2] * f0[15];
    }

    long fallback_count() const { return fallback_; }

  private:
    std::vector<std::vector<std::vector<double>>> coef_; // [field][segment][n * (Kmax + 1) + k]
    mutable long fallback_ = 0;
};

} // namespace kstab
