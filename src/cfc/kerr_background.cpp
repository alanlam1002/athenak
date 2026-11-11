//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file kerr_background.cpp
//! \brief LoadKerrTable and FillKerrBackground (see kerr_background.hpp).

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "coordinates/cell_locations.hpp"
#include "kerr_background.hpp"

namespace cfc {

namespace {

const char *const kFieldNames[16] = {"gxx", "gxy", "gxz", "gyy", "gyz", "gzz", "Kxx",
                                     "Kxy", "Kxz", "Kyy", "Kyz", "Kzz", "alp", "betax",
                                     "betay", "betaz"};

// strip a '#' comment; true if anything but white space is left
bool Clean(std::string *line) {
  const size_t h = line->find('#');
  if (h != std::string::npos) line->erase(h);
  return line->find_first_not_of(" \t\r\n") != std::string::npos;
}

struct HostSeg { bool inv = false; double lo = 0, hi = 0; int N = 0; };

}  // namespace

void LoadKerrTable(const std::string &f2_prefix, const std::string &kstab_path,
                   KerrTable &t) {
  // ---- the KSTAB v1 table (text): header, segments, coefficients
  std::ifstream in(kstab_path);
  if (!in) throw std::runtime_error("kerr_background: cannot open " + kstab_path);
  std::string line;
  bool have_magic = false;
  double ka = 0, rmin = 0, r1 = 0, azc = 0, p[4] = {-2, 0, 0, 1};
  int kmax = -1, nseg = 0;
  std::vector<HostSeg> seg;
  std::vector<std::vector<double>> coef;  // [seg][(f*N + n)*(kmax+1) + k]
  while (std::getline(in, line)) {
    if (!Clean(&line)) continue;
    std::istringstream ss(line);
    std::string tag, kw;
    ss >> tag;
    if (!have_magic) {
      int ver = 0;
      ss >> ver;
      if (tag != "KSTAB" || ver != 1) {
        throw std::runtime_error("kerr_background: not a KSTAB v1 file: " + kstab_path);
      }
      have_magic = true;
    } else if (tag == "a") {
      ss >> ka;
    } else if (tag == "asymp") {
      ss >> kw >> p[0] >> kw >> p[1] >> kw >> p[2] >> kw >> p[3];
    } else if (tag == "azimuth") {
      ss >> kw >> kw >> azc;
    } else if (tag == "rmin") {
      ss >> rmin >> kw >> r1 >> kw >> kmax >> kw >> nseg;
      if (nseg <= 0 || kmax < 0 || kmax + 1 > kKMaxK) {
        throw std::runtime_error("kerr_background: bad rmin line in " + kstab_path);
      }
      seg.assign(nseg, HostSeg());
      coef.assign(nseg, std::vector<double>());
    } else if (tag == "seg") {
      int i = -1;
      std::string type, hi;
      HostSeg s;
      ss >> i >> type >> s.lo >> hi >> s.N;
      if (i < 0 || i >= nseg || s.N < 1 || s.N > kKMaxN) {
        throw std::runtime_error("kerr_background: bad segment line '" + line + "'");
      }
      s.inv = (type == "inv");
      s.hi = s.inv ? 0.0 : std::stod(hi);
      seg[i] = s;
      coef[i].assign(static_cast<size_t>(16)*s.N*(kmax + 1), 0.0);
    } else {
      int f = -1;
      for (int q = 0; q < 16; ++q) {
        if (tag == kFieldNames[q]) f = q;
      }
      if (f < 0) throw std::runtime_error("kerr_background: unknown line '" + line + "'");
      int i = -1, n = -1, k = -1;
      std::string cs;
      double v = 0;
      ss >> i >> n >> cs >> k >> v;
      if (ss.fail() || i < 0 || i >= nseg || n < 0 || n >= seg[i].N || k < 0 || k > kmax ||
          (cs == "s") != kerr_impl::OddField(f)) {
        throw std::runtime_error("kerr_background: bad coefficient line '" + line + "'");
      }
      coef[i][(static_cast<size_t>(f)*seg[i].N + n)*(kmax + 1) + k] = v;
    }
  }
  if (!have_magic || nseg == 0 || rmin <= 0) {
    throw std::runtime_error("kerr_background: incomplete KSTAB header in " + kstab_path);
  }

  // ---- F2 header
  std::ifstream ht(f2_prefix + ".txt");
  if (!ht) throw std::runtime_error("kerr_background: cannot open " + f2_prefix + ".txt");
  double fa = -1, fazc = 0, frmin = -1, frmax = -1;
  int nr = 0, nth = 0, nv = 0;
  while (std::getline(ht, line)) {
    if (!Clean(&line)) continue;
    std::istringstream ss(line);
    std::string tag;
    ss >> tag;
    if (tag == "table_a") ss >> fa;
    else if (tag == "azimuth_c") ss >> fazc;
    else if (tag == "r_min") ss >> frmin;
    else if (tag == "r_max") ss >> frmax;
    else if (tag == "nr") ss >> nr;
    else if (tag == "nth") ss >> nth;
    else if (tag == "nv") ss >> nv;
  }
  if (nv != kKnNV || nr < 6 || nth < 6 || frmax <= frmin || frmin <= 0) {
    throw std::runtime_error("kerr_background: bad F2 header " + f2_prefix + ".txt");
  }
  if (fa != ka || frmin != rmin || fazc != azc) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "kerr_background: F2 (a %.17g, r_min %.17g, c %.17g) "
                  "does not match the KSTAB table (a %.17g, r_min %.17g, c %.17g)",
                  fa, frmin, fazc, ka, rmin, azc);
    throw std::runtime_error(buf);
  }
  const size_t ntot = static_cast<size_t>(nr)*nth*kKnNV;
  HostArray1D<Real> hf2("kerr_f2_h", ntot);
  {
    std::ifstream bin(f2_prefix + ".bin", std::ios::binary | std::ios::ate);
    if (!bin) throw std::runtime_error("kerr_background: cannot open " + f2_prefix + ".bin");
    if (static_cast<size_t>(bin.tellg()) != ntot*sizeof(double)) {
      throw std::runtime_error("kerr_background: " + f2_prefix + ".bin has the wrong size");
    }
    bin.seekg(0);
    std::vector<double> buf(ntot);
    bin.read(reinterpret_cast<char *>(buf.data()), ntot*sizeof(double));
    if (!bin) throw std::runtime_error("kerr_background: short read of " + f2_prefix);
    for (size_t q = 0; q < ntot; ++q) hf2(q) = static_cast<Real>(buf[q]);
  }
  t.f2 = DvceArray1D<Real>("kerr_f2", ntot);
  Kokkos::deep_copy(t.f2, hf2);
  t.nr = nr;
  t.nth = nth;
  t.rmin = rmin;
  t.rmax = frmax;
  t.u0 = std::log(rmin);
  t.hu = (std::log(frmax) - t.u0)/(nr - 1);
  t.ht = M_PI/(nth - 1);
  t.a = ka;
  t.azimuth_c = azc;
  for (int f = 0; f < 4; ++f) t.p[f] = p[f];

  // ---- spectral coefficients: one flat array, segment s at seg_off(s), [16][N][kmax+1]
  t.nseg = nseg;
  t.kmax = kmax;
  t.r1 = r1;
  HostArray1D<Real> hlo("kerr_seg_lo_h", nseg), hhi("kerr_seg_hi_h", nseg);
  HostArray1D<int> hn("kerr_seg_n_h", nseg), hinv("kerr_seg_inv_h", nseg),
                   hoff("kerr_seg_off_h", nseg);
  size_t ncoef = 0;
  for (int s = 0; s < nseg; ++s) {
    if (seg[s].N < 1) throw std::runtime_error("kerr_background: missing segment line");
    hlo(s) = seg[s].lo;
    hhi(s) = seg[s].hi;
    hn(s) = seg[s].N;
    hinv(s) = seg[s].inv ? 1 : 0;
    hoff(s) = static_cast<int>(ncoef);
    ncoef += coef[s].size();
  }
  HostArray1D<Real> hc("kerr_coef_h", ncoef);
  for (int s = 0; s < nseg; ++s) {
    for (size_t q = 0; q < coef[s].size(); ++q) hc(hoff(s) + q) = coef[s][q];
  }
  t.seg_lo = DvceArray1D<Real>("kerr_seg_lo", nseg);
  t.seg_hi = DvceArray1D<Real>("kerr_seg_hi", nseg);
  t.seg_n = DvceArray1D<int>("kerr_seg_n", nseg);
  t.seg_inv = DvceArray1D<int>("kerr_seg_inv", nseg);
  t.seg_off = DvceArray1D<int>("kerr_seg_off", nseg);
  t.coef = DvceArray1D<Real>("kerr_coef", ncoef);
  Kokkos::deep_copy(t.seg_lo, hlo);
  Kokkos::deep_copy(t.seg_hi, hhi);
  Kokkos::deep_copy(t.seg_n, hn);
  Kokkos::deep_copy(t.seg_inv, hinv);
  Kokkos::deep_copy(t.seg_off, hoff);
  Kokkos::deep_copy(t.coef, hc);
}

void FillKerrBackground(MeshBlockPack *pmbp, const KerrTable &t, Real m_bh,
                        const Real xbh[3], Real blend_r1, Real blend_r2,
                        DvceArray5D<Real> &bg) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is; int &js = indcs.js; int &ks = indcs.ks;
  int isg = is - indcs.ng, ieg = indcs.ie + indcs.ng;
  int jsg = js - indcs.ng, jeg = indcs.je + indcs.ng;
  int ksg = ks - indcs.ng, keg = indcs.ke + indcs.ng;
  int nmb = pmbp->nmb_thispack;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const Real xb0 = xbh[0], xb1 = xbh[1], xb2 = xbh[2];
  const KerrTable tt = t;
  par_for("cfc_fill_kerr_background", DevExeSpace(), 0, nmb-1, ksg, keg, jsg, jeg, isg, ieg,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max) - xb0;
    Real x2v = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max) - xb1;
    Real x3v = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max) - xb2;
    Real v[kKnNV];
    KerrNodeCartesian(tt, m_bh, x1v, x2v, x3v, v, false);
    if (blend_r2 > 0.0) { BlendNode(x1v, x2v, x3v, blend_r1, blend_r2, v); }
    KerrBgPoint b;
    KerrConformal(v, b);
    bg(m, kKbgPsi0, k, j, i) = b.psi0;
    bg(m, kKbgAlpha0, k, j, i) = b.alpha0;
    bg(m, kKbgRs, k, j, i) = b.rs;
    for (int a = 0; a < 3; ++a) {
      bg(m, kKbgBeta0 + a, k, j, i) = b.beta0[a];
      bg(m, kKbgDAlpha0 + a, k, j, i) = b.dalpha0[a];
      bg(m, kKbgDLnPsi0 + a, k, j, i) = b.dlnpsi0[a];
      for (int c = 0; c < 3; ++c) { bg(m, kKbgDBeta0 + 3*a + c, k, j, i) = b.dbeta0[a][c]; }
      for (int q = 0; q < 6; ++q) {
        bg(m, kKbgDGt + 6*a + q, k, j, i) = b.dgt[a][q];
        bg(m, kKbgGam + 6*a + q, k, j, i) = b.gam[a][q];
      }
    }
    for (int q = 0; q < 6; ++q) {
      bg(m, kKbgGt + q, k, j, i) = b.gt[q];
      bg(m, kKbgGtu + q, k, j, i) = b.gtu[q];
      bg(m, kKbgRt + q, k, j, i) = b.rt[q];
      bg(m, kKbgK0 + q, k, j, i) = b.k0[q];
      bg(m, kKbgAhu + q, k, j, i) = b.ahu[q];
      bg(m, kKbgAhd + q, k, j, i) = b.ahd[q];
    }
  });
}

void KerrSetRow0Exact(KerrTable &t) {
  const KerrTable tt = t;
  par_for("cfc_kerr_row0_exact", DevExeSpace(), 0, t.nth - 1, KOKKOS_LAMBDA(const int j) {
    Real v[kKnNV];
    KerrNodeSpectral(tt, tt.rmin, j*tt.ht, v);
    for (int q = 0; q < kKnNV; ++q) { tt.f2(static_cast<size_t>(j)*kKnNV + q) = v[q]; }
  });
  Kokkos::fence();
}

}  // namespace cfc
