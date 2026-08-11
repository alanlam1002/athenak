//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyngr_grass.cpp
//  \brief Problem generator for a (possibly differentially rotating) neutron star built
//  from GRASS (an RNS-family rotating-NS equilibrium code) restart-binary initial data.
//  Works under EITHER <z4c> (full dynamical Z4c) or <cfc>/<adm> (Conformally Flat
//  Condition metric solver), selected at runtime by which block is present in the
//  .athinput file -- mirrors the dual-mode pattern used by
//  dyn_grmhd/celephais/celephais_{bns,ns}.cpp. Unlike the sibling scalar-tensor
//  dyngr_rns_st.cpp, no <scalarfield> block is needed or used -- GRASS's restart
//  carries a generic scalar slot for a different (scalarized) solver mode, ignored
//  here (see grass/grass_reader.hpp). GRASS's own (pressure, energy) pair from its
//  equilibrium solve is reused directly, so (unlike dyngr_tov.cpp) no AthenaK EOS call
//  constructs P(rho) for the initial data -- only a small dedicated GRASS-EOS-table
//  lookup (grass/grass_eos_table.hpp) recovers rest-mass density from the restart's
//  `energy` field.
//  If <mhd> nscalars > 0, the passive scalar Y[e] (composition) is additionally seeded
//  from a separate 1D <problem> table (tov::TabulatedEOS, the same reader dyngr_tov.cpp/
//  celephais_bns.cpp use) -- built to carry Yq(nb) along the SAME (nb,T,Yl) trajectory
//  that produced GRASS's own e(n0),p(n0), so the star starts on the trajectory it was
//  built on when evolved with a genuine 3D tabulated <mhd> dyn_eos=compose EOS. This is
//  purely a composition seed -- temperature is never read from this table; PrimToConInit
//  recovers T internally from (rho0,P,Yq) via whatever 3D EOS is active.
//
//  GRASS's own metric is the CST (Cook-Shapiro-Teukolsky) stationary-axisymmetric form
//  (see grass/grass_reader.hpp's own header derivation) -- NOT conformally flat in
//  general. Under <z4c>, the true g_dd/K_dd are used directly (z4c evolves a general
//  metric natively). Under <cfc>, which can only represent a conformally-flat 3-metric
//  (g_dd = psi^4*delta_ij; CFC::InitializeMetric()'s own AssembleConformalMetric
//  overwrites g_dd from its solved psi on every pass regardless of what is fed in here),
//  the per-point metric is isotropized into a conformal-factor guess
//  psi4_guess = cbrt(det(g_dd)) (matches the true metric's local volume element), with
//  vK_dd zeroed -- matching the convention every existing CFC pgen already uses
//  (xns_rotstar.cpp/dyngr_tov.cpp: the extrinsic-curvature initial guess is always
//  discarded by CFC's own elliptic solve for the lapse/shift/K assembly, so it need not
//  be exact, only a reasonable starting point). alpha/beta_u are used as-is from GRASS's
//  own point values in both modes.
//  Compile with '-D PROBLEM=dyn_grmhd/dyngr_grass' to enroll as user-specific pgen.

#include <math.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "z4c/z4c.hpp"
#include "cfc/cfc.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "grass/grass_units.hpp"
#include "grass/grass_eos_table.hpp"
#include "grass/grass_reader.hpp"
#include "grass/grass_magnetic_web.hpp"
#include "utils/tov/tov_tabulated.hpp"

// Prototype for user-defined history function
void GrassHistory(HistoryData *pdata, Mesh *pm);

// Prototype for the magnetic-web + dipole initial B-field builder (see its own
// definition below for the full design; magnetic_web_id_athenak.md, repo root,
// is the physics/implementation spec this follows). Reads only padm->adm
// (g_dd/mode-agnostic by construction, see file header above) and pmhd
// w0/b0/bcc0 -- no z4c/CFC-specific state, so this ported over unmodified.
void BuildMagneticField(ParameterInput *pin, Mesh *pmy_mesh_,
                         const grass::GrassData &data, const grass::GrassEosTable &eos_table);

// Gauss -> AthenaK code-unit (<mhd> units=geometric_solar) conversion constant,
// reused as-is from src/pgen/dyn_grmhd/lorene/lorene_bns.cpp (its athenaB/1e9
// derivation) -- GrassUnits::code is ALSO Primitive::MakeGeometricSolar(), the
// same G=c=Msun=1 system Lorene's own athenaL/athenaM construction targets, so
// this carries over directly (see grass/NOTES.md for the sub-permille caveat).
static constexpr Real kGaussToCode = 1.0 / 8.3519664583273e+19;

//----------------------------------------------------------------------------------------
//! \fn void SetupGrass
//  \brief Loads a GRASS restart binary and fills ADM/(Z4c or CFC)/hydro initial data.
//  Host-only: the reader's interpolation (grass::GrassData) is plain-CPU code, so this
//  mirrors the host-fill-then-deep_copy pattern used by the other external-ID pgens
//  (elliptica/lorene/sgrid/celephais/rns_st), not dyngr_tov.cpp's device-side par_for
//  (which works directly from a closed-form/ODE solution instead). Non-templated (unlike
//  the TOV-family pgens' Setup<EOS>) -- see file header, no AthenaK EOS call is needed.

void SetupGrass(ParameterInput *pin, Mesh *pmy_mesh_) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  std::string id_file = pin->GetString("problem", "id_file");
  std::string eos_table_file = pin->GetString("problem", "grass_eos_table");

  grass::GrassUnits units;
  grass::GrassEosTable eos_table(eos_table_file, units);
  grass::GrassData data(id_file, units);
  // 1D DD2_hot_slice table: Y[e]=Yl(nb) composition seed only (see file header) --
  // <problem> table = ... , same input key tov::TabulatedEOS's other callers
  // (dyngr_tov.cpp, celephais_bns.cpp) already use.
  tov::TabulatedEOS slice_eos(pin);
  const bool read_ye = pin->GetOrAddInteger("mhd", "nscalars", 0) > 0;

  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;

  auto &u_adm = pmbp->padm->u_adm;
  auto &w0 = pmbp->pmhd->w0;
  const bool is_z4c = (pmbp->pz4c != nullptr);

  // Host-side fill, then move to device -- see file header.
  HostArray5D<Real>::HostMirror host_u_adm = Kokkos::create_mirror_view(u_adm);
  HostArray5D<Real>::HostMirror host_w0 = Kokkos::create_mirror_view(w0);
  HostArray5D<Real>::HostMirror host_u_z4c;

  // Gauge variables (alpha, beta_u) live in Z4c's own evolved state under <z4c> (ADMToZ4c
  // only derives the geometric/conformal part from g_dd/vK_dd, it never touches the
  // gauge), but in u_adm itself under <cfc>/bare <adm> -- matches adm::ADM's own
  // constructor logic (adm.cpp). g_dd/vK_dd/psi4 always live in u_adm regardless.
  adm::ADM::ADMhost_vars host_adm;
  if (is_z4c) {
    host_u_z4c = Kokkos::create_mirror_view(pmbp->pz4c->u0);
    host_adm.alpha.InitWithShallowSlice(host_u_z4c, z4c::Z4c::I_Z4C_ALPHA);
    host_adm.beta_u.InitWithShallowSlice(host_u_z4c,
        z4c::Z4c::I_Z4C_BETAX, z4c::Z4c::I_Z4C_BETAZ);
  } else {
    host_adm.alpha.InitWithShallowSlice(host_u_adm, adm::ADM::I_ADM_ALPHA);
    host_adm.beta_u.InitWithShallowSlice(host_u_adm,
        adm::ADM::I_ADM_BETAX, adm::ADM::I_ADM_BETAZ);
  }
  host_adm.psi4.InitWithShallowSlice(host_u_adm, adm::ADM::I_ADM_PSI4);
  host_adm.g_dd.InitWithShallowSlice(host_u_adm,
      adm::ADM::I_ADM_GXX, adm::ADM::I_ADM_GZZ);
  host_adm.vK_dd.InitWithShallowSlice(host_u_adm,
      adm::ADM::I_ADM_KXX, adm::ADM::I_ADM_KZZ);

  for (int m = 0; m < nmb; ++m) {
    Real &x1min = size.h_view(m).x1min;
    Real &x1max = size.h_view(m).x1max;
    Real &x2min = size.h_view(m).x2min;
    Real &x2max = size.h_view(m).x2max;
    Real &x3min = size.h_view(m).x3min;
    Real &x3max = size.h_view(m).x3max;
    for (int k = 0; k < n3; ++k) {
      Real x3v = CellCenterX(k - indcs.ks, indcs.nx3, x3min, x3max);
      for (int j = 0; j < n2; ++j) {
        Real x2v = CellCenterX(j - indcs.js, indcs.nx2, x2min, x2max);
        for (int i = 0; i < n1; ++i) {
          Real x1v = CellCenterX(i - indcs.is, indcs.nx1, x1min, x1max);

          grass::GrassData::Point pt;
          data.Interpolate(x1v, x2v, x3v, eos_table, slice_eos, &pt);

          host_adm.alpha(m, k, j, i) = pt.alpha;
          host_adm.beta_u(m, 0, k, j, i) = pt.beta_u[0];
          host_adm.beta_u(m, 1, k, j, i) = pt.beta_u[1];
          host_adm.beta_u(m, 2, k, j, i) = pt.beta_u[2];

          if (is_z4c) {
            // True (generally non-conformally-flat) CST metric/extrinsic curvature --
            // z4c evolves a general metric natively, no isotropization needed.
            host_adm.g_dd(m, 0, 0, k, j, i) = pt.g_dd[0];
            host_adm.g_dd(m, 0, 1, k, j, i) = pt.g_dd[1];
            host_adm.g_dd(m, 0, 2, k, j, i) = pt.g_dd[2];
            host_adm.g_dd(m, 1, 1, k, j, i) = pt.g_dd[3];
            host_adm.g_dd(m, 1, 2, k, j, i) = pt.g_dd[4];
            host_adm.g_dd(m, 2, 2, k, j, i) = pt.g_dd[5];

            host_adm.vK_dd(m, 0, 0, k, j, i) = pt.K_dd[0];
            host_adm.vK_dd(m, 0, 1, k, j, i) = pt.K_dd[1];
            host_adm.vK_dd(m, 0, 2, k, j, i) = pt.K_dd[2];
            host_adm.vK_dd(m, 1, 1, k, j, i) = pt.K_dd[3];
            host_adm.vK_dd(m, 1, 2, k, j, i) = pt.K_dd[4];
            host_adm.vK_dd(m, 2, 2, k, j, i) = pt.K_dd[5];
          } else {
            // CFC can only represent a conformally-flat 3-metric; isotropize GRASS's
            // true (anisotropic) metric into a conformal-factor guess that matches its
            // local volume element (det g_dd). vK_dd is zeroed -- CFC::InitializeMetric()
            // always overwrites both from its own elliptic solve (see file header).
            Real gxx = pt.g_dd[0], gxy = pt.g_dd[1], gxz = pt.g_dd[2];
            Real gyy = pt.g_dd[3], gyz = pt.g_dd[4], gzz = pt.g_dd[5];
            Real det_g = gxx*(gyy*gzz - gyz*gyz) - gxy*(gxy*gzz - gyz*gxz)
                         + gxz*(gxy*gyz - gyy*gxz);
            Real psi4_guess = std::cbrt(std::max(det_g, 1.0e-300));

            host_adm.g_dd(m, 0, 0, k, j, i) = psi4_guess;
            host_adm.g_dd(m, 0, 1, k, j, i) = 0.0;
            host_adm.g_dd(m, 0, 2, k, j, i) = 0.0;
            host_adm.g_dd(m, 1, 1, k, j, i) = psi4_guess;
            host_adm.g_dd(m, 1, 2, k, j, i) = 0.0;
            host_adm.g_dd(m, 2, 2, k, j, i) = psi4_guess;
            host_adm.psi4(m, k, j, i) = psi4_guess;

            host_adm.vK_dd(m, 0, 0, k, j, i) = 0.0;
            host_adm.vK_dd(m, 0, 1, k, j, i) = 0.0;
            host_adm.vK_dd(m, 0, 2, k, j, i) = 0.0;
            host_adm.vK_dd(m, 1, 1, k, j, i) = 0.0;
            host_adm.vK_dd(m, 1, 2, k, j, i) = 0.0;
            host_adm.vK_dd(m, 2, 2, k, j, i) = 0.0;
          }

          Real rho = (pt.rho0 > 0.0) ? pt.rho0 : 0.0;
          Real pres = (pt.rho0 > 0.0) ? pt.pres : 0.0;
          Real vu[3] = {0.0, 0.0, 0.0};
          if (pt.rho0 > 0.0) {
            vu[0] = pt.vu[0]; vu[1] = pt.vu[1]; vu[2] = pt.vu[2];
          }

          host_w0(m, IDN, k, j, i) = rho;
          host_w0(m, IPR, k, j, i) = pres;
          host_w0(m, IVX, k, j, i) = vu[0];
          host_w0(m, IVY, k, j, i) = vu[1];
          host_w0(m, IVZ, k, j, i) = vu[2];
          if (read_ye) {
            host_w0(m, IYF, k, j, i) = pt.Yq;
          }
        }
      }
    }
  }

  Kokkos::deep_copy(u_adm, host_u_adm);
  Kokkos::deep_copy(w0, host_w0);
  if (is_z4c) {
    Kokkos::deep_copy(pmbp->pz4c->u0, host_u_z4c);
  }

  // GRASS's ID carries no B-field data. AthenaK does not zero-initialize device
  // memory by default, so b0/bcc0 must be set explicitly or they're left as garbage,
  // which otherwise poisons the primitive solver with NaN from the very first step.
  // BuildMagneticField() installs the magnetic-web + dipole field (see its own
  // definition below) when <problem> web_enable/dipole_enable request it; it
  // zero-fills b0/bcc0 itself (both default to 0, i.e. unmagnetized) otherwise --
  // old input files that never mention these keys get exactly the previous
  // unconditional-zero behavior.
  BuildMagneticField(pin, pmy_mesh_, data, eos_table);
}

//----------------------------------------------------------------------------------------
//! \fn void BuildMagneticField
//! \brief Builds and installs the "magnetic web" (Skoutnev & Beloborodov 2025,
//  arXiv:2504.07223) + optional dipole initial magnetic field, per
//  magnetic_web_id_athenak.md (repo root) -- see grass/grass_magnetic_web.hpp for the
//  field-defining math and grass/NOTES.md for the full design rationale (host-side
//  construction, units/densitization convention, the derived a3 AMR correction).
//  <problem> web_enable/dipole_enable both default to 0 (off) -- input files that never
//  mention them get the exact same zero-B behavior as before this feature existed.

void BuildMagneticField(ParameterInput *pin, Mesh *pmy_mesh_,
                         const grass::GrassData &data, const grass::GrassEosTable &eos_table) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int n1 = indcs.nx1 + 2*indcs.ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*indcs.ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*indcs.ng) : 1;
  int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;
  auto &nghbr = pmbp->pmb->nghbr;
  auto &mblev = pmbp->pmb->mb_lev;
  auto &b0 = pmbp->pmhd->b0;
  auto &bcc0 = pmbp->pmhd->bcc0;

  bool web_enable = pin->GetOrAddInteger("problem", "web_enable", 0) != 0;
  bool dipole_enable = pin->GetOrAddInteger("problem", "dipole_enable", 0) != 0;

  if (!web_enable && !dipole_enable) {
    Kokkos::deep_copy(b0.x1f, 0.0);
    Kokkos::deep_copy(b0.x2f, 0.0);
    Kokkos::deep_copy(b0.x3f, 0.0);
    Kokkos::deep_copy(bcc0, 0.0);
    return;
  }

  // ---- <problem> keys ---------------------------------------------------------------
  bool dipole_confine = pin->GetOrAddInteger("problem", "dipole_confine", 0) != 0;
  bool need_h = web_enable || (dipole_enable && dipole_confine);
  Real rho_lo = 0.0, rho_hi = 0.0;
  if (need_h) {
    rho_lo = pin->GetReal("problem", "web_rho_lo");
    rho_hi = pin->GetReal("problem", "web_rho_hi");
  }
  Real mu_star = 2.0, web_bmax_code = 0.0, web_tor_pol = 4.0, web_kmin = 0.0, web_kmax = 0.0;
  if (web_enable) {
    mu_star = pin->GetOrAddReal("problem", "web_mu_star", 2.0);
    web_bmax_code = pin->GetOrAddReal("problem", "web_bmax_gauss", 1.0e16) * kGaussToCode;
    web_tor_pol = pin->GetOrAddReal("problem", "web_tor_pol", 4.0);
    web_kmin = pin->GetReal("problem", "web_kmin");
    web_kmax = pin->GetReal("problem", "web_kmax");
  }
  Real dipole_bmax_code = 0.0, dipole_r0 = 5.0;
  if (dipole_enable) {
    dipole_bmax_code = pin->GetOrAddReal("problem", "dipole_bmax_gauss", 1.0e16) * kGaussToCode;
    dipole_r0 = pin->GetOrAddReal("problem", "dipole_r0", 5.0);
  }

  // ---- Resolution guard (web only): 2*pi/k_max must resolve on >=8 finest cells -----
  if (web_enable) {
    Real dx_finest = pmy_mesh_->mesh_size.dx1 /
                      std::pow(2.0, pmy_mesh_->max_level - pmy_mesh_->root_level);
    if (2.0*M_PI/web_kmax < 8.0*dx_finest) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "web_kmax=" << web_kmax << " under-resolved: 2*pi/web_kmax="
                << 2.0*M_PI/web_kmax << " < 8*dx_finest=" << 8.0*dx_finest << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  // ---- rho_max validation + shell -> R_cyl startup diagnostic -----------------------
  if (need_h) {
    auto &w0d = pmbp->pmhd->w0;
    int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
    const int nmkji = nmb*nx3*nx2*nx1;
    const int nkji = nx3*nx2*nx1, nji = nx2*nx1;
    Real rho_max = 0.0;
    Kokkos::parallel_reduce("GrassWebRhoMax", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &mmax) {
      int m = idx/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks; j += js;
      mmax = fmax(mmax, w0d(m, IDN, k, j, i));
    }, Kokkos::Max<Real>(rho_max));
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &rho_max, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
    if (rho_hi > rho_max) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "web_rho_hi=" << rho_hi << " exceeds this star's rho_max=" << rho_max
                << " -- the mass-shell window must lie within the star." << std::endl;
      exit(EXIT_FAILURE);
    }
    if (global_variable::my_rank == 0) {
      Real r_inner = -1.0, r_outer = -1.0;
      Real x1max = pmy_mesh_->mesh_size.x1max;
      for (int s = 1; s <= 2000; ++s) {
        Real r = s * (x1max / 2000.0);
        Real rho_here = data.InterpolateRho(r, 0.0, 0.0, eos_table);
        if (rho_here < rho_hi && r_inner < 0.0) { r_inner = r; }
        if (rho_here < rho_lo && r_outer < 0.0) { r_outer = r; break; }
      }
      if (r_inner > 0.0 && r_outer > r_inner) {
        Real shell_width = r_outer - r_inner;
        std::cout << "GRASS magnetic web: rho shell [" << rho_lo << "," << rho_hi
                  << "] (code units) maps to equatorial R_cyl in [" << r_inner << ","
                  << r_outer << "] (width=" << shell_width << "), rho_max=" << rho_max
                  << std::endl;
        if (web_enable && 2.0*M_PI/web_kmin > shell_width) {
          std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__ << std::endl
                    << "web_kmin=" << web_kmin << " gives the longest mode wavelength "
                    << "2*pi/web_kmin=" << 2.0*M_PI/web_kmin << " exceeding the shell "
                    << "width=" << shell_width << " -- consider raising web_kmin."
                    << std::endl;
        }
      } else {
        std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "Could not bracket the rho shell along the equatorial sample line -- "
                  << "check web_rho_lo/web_rho_hi against this star's density profile."
                  << std::endl;
      }
    }
  }

  // ---- Generic edge-staggered builder + AMR fine/coarse correction ------------------
  // Shared by the web's poloidal pass, toroidal pass, AND the dipole -- the exact
  // boundary-correction logic (derived this session for a3, verified against
  // src/mesh/nghbr_index.hpp's neighbor-index blocks) lives in exactly ONE place here,
  // parameterized by per-component evaluator closures (each closure already captures
  // whatever h(rho)/mode-table/lambda/current it needs). Mirrors
  // src/pgen/dyn_grmhd/lorene/lorene_bns.cpp's a1/a2 pattern structurally; a3 (new,
  // does not exist there) follows by direct analogy: x1-face correction unconditional
  // (matching a2's own x1 treatment), x2-face/x1x2-edge correction gated by nx2>1
  // (matching a1's own x2/x3 gating), averaged over x3v+-0.25*dx3 (a3 is cell-centered
  // in x3, same idiom as a1/a2 averaging over their own centering direction).
  auto BuildAndCorrect = [&](HostArray4D<Real> &arr1, HostArray4D<Real> &arr2,
                              HostArray4D<Real> &arr3, bool has_a3,
                              const std::function<Real(Real, Real, Real)> &f1,
                              const std::function<Real(Real, Real, Real)> &f2,
                              const std::function<Real(Real, Real, Real)> &f3) {
    for (int m = 0; m < nmb; ++m) {
      Real x1min = size.h_view(m).x1min, x1max = size.h_view(m).x1max;
      Real x2min = size.h_view(m).x2min, x2max = size.h_view(m).x2max;
      Real x3min = size.h_view(m).x3min, x3max = size.h_view(m).x3max;
      for (int k = ks; k <= ke+1; ++k) {
        Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
        Real x3f = LeftEdgeX(k-ks, indcs.nx3, x3min, x3max);
        for (int j = js; j <= je+1; ++j) {
          Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
          Real x2f = LeftEdgeX(j-js, indcs.nx2, x2min, x2max);
          for (int i = is; i <= ie+1; ++i) {
            Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
            Real x1f = LeftEdgeX(i-is, indcs.nx1, x1min, x1max);
            arr1(m, k, j, i) = f1(x1v, x2f, x3f);
            arr2(m, k, j, i) = f2(x1f, x2v, x3f);
            arr3(m, k, j, i) = has_a3 ? f3(x1f, x2f, x3v) : 0.0;
          }
        }
      }
    }
    for (int m = 0; m < nmb; ++m) {
      Real x1min = size.h_view(m).x1min, x1max = size.h_view(m).x1max;
      Real x2min = size.h_view(m).x2min, x2max = size.h_view(m).x2max;
      Real x3min = size.h_view(m).x3min, x3max = size.h_view(m).x3max;
      Real dx1 = size.h_view(m).dx1, dx2 = size.h_view(m).dx2, dx3 = size.h_view(m).dx3;
      for (int k = ks; k <= ke+1; ++k) {
        for (int j = js; j <= je+1; ++j) {
          for (int i = is; i <= ie+1; ++i) {
            // --- a1: x2-face/x3-face/x2x3-edge neighbors (verbatim lorene_bns.cpp) ---
            bool corr1 = (indcs.nx2 > 1 && (
                (j==js && (nghbr.h_view(m,8).lev>mblev.h_view(m) ||
                           nghbr.h_view(m,9).lev>mblev.h_view(m) ||
                           nghbr.h_view(m,10).lev>mblev.h_view(m) ||
                           nghbr.h_view(m,11).lev>mblev.h_view(m))) ||
                (j==je+1 && (nghbr.h_view(m,12).lev>mblev.h_view(m) ||
                             nghbr.h_view(m,13).lev>mblev.h_view(m) ||
                             nghbr.h_view(m,14).lev>mblev.h_view(m) ||
                             nghbr.h_view(m,15).lev>mblev.h_view(m))))) ||
                (indcs.nx3 > 1 && (
                (k==ks && (nghbr.h_view(m,24).lev>mblev.h_view(m) ||
                           nghbr.h_view(m,25).lev>mblev.h_view(m) ||
                           nghbr.h_view(m,26).lev>mblev.h_view(m) ||
                           nghbr.h_view(m,27).lev>mblev.h_view(m))) ||
                (k==ke+1 && (nghbr.h_view(m,28).lev>mblev.h_view(m) ||
                             nghbr.h_view(m,29).lev>mblev.h_view(m) ||
                             nghbr.h_view(m,30).lev>mblev.h_view(m) ||
                             nghbr.h_view(m,31).lev>mblev.h_view(m))) ||
                (j==js && k==ks && (nghbr.h_view(m,40).lev>mblev.h_view(m) ||
                                    nghbr.h_view(m,41).lev>mblev.h_view(m))) ||
                (j==je+1 && k==ks && (nghbr.h_view(m,42).lev>mblev.h_view(m) ||
                                      nghbr.h_view(m,43).lev>mblev.h_view(m))) ||
                (j==js && k==ke+1 && (nghbr.h_view(m,44).lev>mblev.h_view(m) ||
                                      nghbr.h_view(m,45).lev>mblev.h_view(m))) ||
                (j==je+1 && k==ke+1 && (nghbr.h_view(m,46).lev>mblev.h_view(m) ||
                                        nghbr.h_view(m,47).lev>mblev.h_view(m)))));
            if (corr1) {
              Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
              Real x2f = LeftEdgeX(j-js, indcs.nx2, x2min, x2max);
              Real x3f = LeftEdgeX(k-ks, indcs.nx3, x3min, x3max);
              Real xl = x1v + 0.25*dx1, xr = x1v - 0.25*dx1;
              arr1(m, k, j, i) = 0.5*(f1(xl, x2f, x3f) + f1(xr, x2f, x3f));
            }

            // --- a2: x1-face (unconditional) + x3-face/x1x3-edge (verbatim) ---
            bool corr2 =
                (nghbr.h_view(m,0).lev>mblev.h_view(m) && i==is) ||
                (nghbr.h_view(m,1).lev>mblev.h_view(m) && i==is) ||
                (nghbr.h_view(m,2).lev>mblev.h_view(m) && i==is) ||
                (nghbr.h_view(m,3).lev>mblev.h_view(m) && i==is) ||
                (nghbr.h_view(m,4).lev>mblev.h_view(m) && i==ie+1) ||
                (nghbr.h_view(m,5).lev>mblev.h_view(m) && i==ie+1) ||
                (nghbr.h_view(m,6).lev>mblev.h_view(m) && i==ie+1) ||
                (nghbr.h_view(m,7).lev>mblev.h_view(m) && i==ie+1) ||
                (indcs.nx3 > 1 && (
                  (nghbr.h_view(m,24).lev>mblev.h_view(m) && k==ks) ||
                  (nghbr.h_view(m,25).lev>mblev.h_view(m) && k==ks) ||
                  (nghbr.h_view(m,26).lev>mblev.h_view(m) && k==ks) ||
                  (nghbr.h_view(m,27).lev>mblev.h_view(m) && k==ks) ||
                  (nghbr.h_view(m,28).lev>mblev.h_view(m) && k==ke+1) ||
                  (nghbr.h_view(m,29).lev>mblev.h_view(m) && k==ke+1) ||
                  (nghbr.h_view(m,30).lev>mblev.h_view(m) && k==ke+1) ||
                  (nghbr.h_view(m,31).lev>mblev.h_view(m) && k==ke+1) ||
                  (nghbr.h_view(m,32).lev>mblev.h_view(m) && i==is && k==ks) ||
                  (nghbr.h_view(m,33).lev>mblev.h_view(m) && i==is && k==ks) ||
                  (nghbr.h_view(m,34).lev>mblev.h_view(m) && i==ie+1 && k==ks) ||
                  (nghbr.h_view(m,35).lev>mblev.h_view(m) && i==ie+1 && k==ks) ||
                  (nghbr.h_view(m,36).lev>mblev.h_view(m) && i==is && k==ke+1) ||
                  (nghbr.h_view(m,37).lev>mblev.h_view(m) && i==is && k==ke+1) ||
                  (nghbr.h_view(m,38).lev>mblev.h_view(m) && i==ie+1 && k==ke+1) ||
                  (nghbr.h_view(m,39).lev>mblev.h_view(m) && i==ie+1 && k==ke+1)));
            if (corr2) {
              Real x1f = LeftEdgeX(i-is, indcs.nx1, x1min, x1max);
              Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
              Real x3f = LeftEdgeX(k-ks, indcs.nx3, x3min, x3max);
              Real xl = x2v + 0.25*dx2, xr = x2v - 0.25*dx2;
              arr2(m, k, j, i) = 0.5*(f2(x1f, xl, x3f) + f2(x1f, xr, x3f));
            }

            // --- a3 (new, derived this session): x1-face (unconditional, matching a2's
            // own x1 treatment) + x2-face/x1x2-edge (matching a1's own nx2 gating) ---
            if (has_a3) {
              bool corr3 =
                  (nghbr.h_view(m,0).lev>mblev.h_view(m) && i==is) ||
                  (nghbr.h_view(m,1).lev>mblev.h_view(m) && i==is) ||
                  (nghbr.h_view(m,2).lev>mblev.h_view(m) && i==is) ||
                  (nghbr.h_view(m,3).lev>mblev.h_view(m) && i==is) ||
                  (nghbr.h_view(m,4).lev>mblev.h_view(m) && i==ie+1) ||
                  (nghbr.h_view(m,5).lev>mblev.h_view(m) && i==ie+1) ||
                  (nghbr.h_view(m,6).lev>mblev.h_view(m) && i==ie+1) ||
                  (nghbr.h_view(m,7).lev>mblev.h_view(m) && i==ie+1) ||
                  (indcs.nx2 > 1 && (
                    (nghbr.h_view(m,8).lev>mblev.h_view(m) && j==js) ||
                    (nghbr.h_view(m,9).lev>mblev.h_view(m) && j==js) ||
                    (nghbr.h_view(m,10).lev>mblev.h_view(m) && j==js) ||
                    (nghbr.h_view(m,11).lev>mblev.h_view(m) && j==js) ||
                    (nghbr.h_view(m,12).lev>mblev.h_view(m) && j==je+1) ||
                    (nghbr.h_view(m,13).lev>mblev.h_view(m) && j==je+1) ||
                    (nghbr.h_view(m,14).lev>mblev.h_view(m) && j==je+1) ||
                    (nghbr.h_view(m,15).lev>mblev.h_view(m) && j==je+1) ||
                    (nghbr.h_view(m,16).lev>mblev.h_view(m) && i==is && j==js) ||
                    (nghbr.h_view(m,17).lev>mblev.h_view(m) && i==is && j==js) ||
                    (nghbr.h_view(m,18).lev>mblev.h_view(m) && i==ie+1 && j==js) ||
                    (nghbr.h_view(m,19).lev>mblev.h_view(m) && i==ie+1 && j==js) ||
                    (nghbr.h_view(m,20).lev>mblev.h_view(m) && i==is && j==je+1) ||
                    (nghbr.h_view(m,21).lev>mblev.h_view(m) && i==is && j==je+1) ||
                    (nghbr.h_view(m,22).lev>mblev.h_view(m) && i==ie+1 && j==je+1) ||
                    (nghbr.h_view(m,23).lev>mblev.h_view(m) && i==ie+1 && j==je+1)));
              if (corr3) {
                Real x1f = LeftEdgeX(i-is, indcs.nx1, x1min, x1max);
                Real x2f = LeftEdgeX(j-js, indcs.nx2, x2min, x2max);
                Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
                Real xl = x3v + 0.25*dx3, xr = x3v - 0.25*dx3;
                arr3(m, k, j, i) = 0.5*(f3(x1f, x2f, xl) + f3(x1f, x2f, xr));
              }
            }
          }
        }
      }
    }
  };

  // Device curl (verbatim lorene_bns.cpp pgen_Bfc pattern) -> temporary cell-centered
  // Cartesian B components, used for the web's normalization/rescale passes (which need
  // B, not just A). Allocates its own temporary face arrays each call -- called only a
  // handful of times total (P pass, T pass, web rescale check, dipole rescale check),
  // so the extra allocations are a negligible one-time startup cost.
  auto CurlToBcc = [&](DvceArray4D<Real> &da1, DvceArray4D<Real> &da2, DvceArray4D<Real> &da3,
                        DvceArray4D<Real> &out_bx, DvceArray4D<Real> &out_by,
                        DvceArray4D<Real> &out_bz) {
    DvceArray4D<Real> f1, f2, f3;
    Kokkos::realloc(f1, nmb, n3, n2, n1);
    Kokkos::realloc(f2, nmb, n3, n2, n1);
    Kokkos::realloc(f3, nmb, n3, n2, n1);
    par_for("grass_web_curl", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real dx1 = size.d_view(m).dx1, dx2 = size.d_view(m).dx2, dx3 = size.d_view(m).dx3;
      f1(m,k,j,i) = (da3(m,k,j+1,i)-da3(m,k,j,i))/dx2 - (da2(m,k+1,j,i)-da2(m,k,j,i))/dx3;
      f2(m,k,j,i) = (da1(m,k+1,j,i)-da1(m,k,j,i))/dx3 - (da3(m,k,j,i+1)-da3(m,k,j,i))/dx1;
      f3(m,k,j,i) = (da2(m,k,j,i+1)-da2(m,k,j,i))/dx1 - (da1(m,k,j+1,i)-da1(m,k,j,i))/dx2;
      if (i==ie) {
        f1(m,k,j,i+1) = (da3(m,k,j+1,i+1)-da3(m,k,j,i+1))/dx2 - (da2(m,k+1,j,i+1)-da2(m,k,j,i+1))/dx3;
      }
      if (j==je) {
        f2(m,k,j+1,i) = (da1(m,k+1,j+1,i)-da1(m,k,j+1,i))/dx3 - (da3(m,k,j+1,i+1)-da3(m,k,j+1,i))/dx1;
      }
      if (k==ke) {
        f3(m,k+1,j,i) = (da2(m,k+1,j,i+1)-da2(m,k+1,j,i))/dx1 - (da1(m,k+1,j+1,i)-da1(m,k+1,j,i))/dx2;
      }
    });
    par_for("grass_web_bcc_tmp", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      out_bx(m,k,j,i) = 0.5*(f1(m,k,j,i) + f1(m,k,j,i+1));
      out_by(m,k,j,i) = 0.5*(f2(m,k,j,i) + f2(m,k,j+1,i));
      out_bz(m,k,j,i) = 0.5*(f3(m,k,j,i) + f3(m,k+1,j,i));
    });
  };

  HostArray4D<Real> final_a1, final_a2, final_a3;
  Kokkos::realloc(final_a1, nmb, n3, n2, n1);
  Kokkos::realloc(final_a2, nmb, n3, n2, n1);
  Kokkos::realloc(final_a3, nmb, n3, n2, n1);
  Kokkos::deep_copy(final_a1, 0.0);
  Kokkos::deep_copy(final_a2, 0.0);
  Kokkos::deep_copy(final_a3, 0.0);

  // ==== Web: build P/T decomposition, enforce E_tor/E_pol exactly, rescale, combine ==
  if (web_enable) {
    grass::WebModeTable mode_table;
    grass::GenerateWebModeTable(pin, &mode_table);

    std::function<Real(Real, Real, Real)> f1P = [&](Real x, Real y, Real z) -> Real {
      Real rho = data.InterpolateRho(x, y, z, eos_table);
      Real h = grass::WebEnvelope(rho, rho_lo, rho_hi);
      if (h == 0.0) { return 0.0; }
      Real ax, ay, az;
      grass::WebA(x, y, z, mode_table, mu_star, 0.0, 1.0, &ax, &ay, &az);
      return h*ax;
    };
    std::function<Real(Real, Real, Real)> f2P = [&](Real x, Real y, Real z) -> Real {
      Real rho = data.InterpolateRho(x, y, z, eos_table);
      Real h = grass::WebEnvelope(rho, rho_lo, rho_hi);
      if (h == 0.0) { return 0.0; }
      Real ax, ay, az;
      grass::WebA(x, y, z, mode_table, mu_star, 0.0, 1.0, &ax, &ay, &az);
      return h*ay;
    };
    std::function<Real(Real, Real, Real)> f3P = [&](Real x, Real y, Real z) -> Real {
      Real rho = data.InterpolateRho(x, y, z, eos_table);
      Real h = grass::WebEnvelope(rho, rho_lo, rho_hi);
      if (h == 0.0) { return 0.0; }
      Real ax, ay, az;
      grass::WebA(x, y, z, mode_table, mu_star, 0.0, 1.0, &ax, &ay, &az);
      return h*az;
    };
    std::function<Real(Real, Real, Real)> f1T = [&](Real x, Real y, Real z) -> Real {
      Real rho = data.InterpolateRho(x, y, z, eos_table);
      Real h = grass::WebEnvelope(rho, rho_lo, rho_hi);
      if (h == 0.0) { return 0.0; }
      Real ax, ay, az;
      grass::WebA(x, y, z, mode_table, mu_star, 1.0, 0.0, &ax, &ay, &az);
      return h*ax;
    };
    std::function<Real(Real, Real, Real)> f2T = [&](Real x, Real y, Real z) -> Real {
      Real rho = data.InterpolateRho(x, y, z, eos_table);
      Real h = grass::WebEnvelope(rho, rho_lo, rho_hi);
      if (h == 0.0) { return 0.0; }
      Real ax, ay, az;
      grass::WebA(x, y, z, mode_table, mu_star, 1.0, 0.0, &ax, &ay, &az);
      return h*ay;
    };
    std::function<Real(Real, Real, Real)> f3T = [&](Real x, Real y, Real z) -> Real {
      Real rho = data.InterpolateRho(x, y, z, eos_table);
      Real h = grass::WebEnvelope(rho, rho_lo, rho_hi);
      if (h == 0.0) { return 0.0; }
      Real ax, ay, az;
      grass::WebA(x, y, z, mode_table, mu_star, 1.0, 0.0, &ax, &ay, &az);
      return h*az;
    };

    HostArray4D<Real> a1P, a2P, a3P, a1T, a2T, a3T;
    Kokkos::realloc(a1P, nmb, n3, n2, n1); Kokkos::realloc(a2P, nmb, n3, n2, n1);
    Kokkos::realloc(a3P, nmb, n3, n2, n1); Kokkos::realloc(a1T, nmb, n3, n2, n1);
    Kokkos::realloc(a2T, nmb, n3, n2, n1); Kokkos::realloc(a3T, nmb, n3, n2, n1);
    BuildAndCorrect(a1P, a2P, a3P, true, f1P, f2P, f3P);
    BuildAndCorrect(a1T, a2T, a3T, true, f1T, f2T, f3T);

    DvceArray4D<Real> d_a1P, d_a2P, d_a3P, d_a1T, d_a2T, d_a3T;
    Kokkos::realloc(d_a1P, nmb, n3, n2, n1); Kokkos::realloc(d_a2P, nmb, n3, n2, n1);
    Kokkos::realloc(d_a3P, nmb, n3, n2, n1); Kokkos::realloc(d_a1T, nmb, n3, n2, n1);
    Kokkos::realloc(d_a2T, nmb, n3, n2, n1); Kokkos::realloc(d_a3T, nmb, n3, n2, n1);
    Kokkos::deep_copy(d_a1P, a1P); Kokkos::deep_copy(d_a2P, a2P); Kokkos::deep_copy(d_a3P, a3P);
    Kokkos::deep_copy(d_a1T, a1T); Kokkos::deep_copy(d_a2T, a2T); Kokkos::deep_copy(d_a3T, a3T);

    DvceArray4D<Real> bxP, byP, bzP, bxT, byT, bzT;
    Kokkos::realloc(bxP, nmb, n3, n2, n1); Kokkos::realloc(byP, nmb, n3, n2, n1);
    Kokkos::realloc(bzP, nmb, n3, n2, n1); Kokkos::realloc(bxT, nmb, n3, n2, n1);
    Kokkos::realloc(byT, nmb, n3, n2, n1); Kokkos::realloc(bzT, nmb, n3, n2, n1);
    CurlToBcc(d_a1P, d_a2P, d_a3P, bxP, byP, bzP);
    CurlToBcc(d_a1T, d_a2T, d_a3T, bxT, byT, bzT);

    // ---- Six-integral reduction (restricted to rho>rho_lo), quadratic solve --------
    auto &w0d = pmbp->pmhd->w0;
    auto &admd = pmbp->padm->adm;
    int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
    const int nmkji = nmb*nx3*nx2*nx1;
    const int nkji = nx3*nx2*nx1, nji = nx2*nx1;
    Real sa=0.0, sb=0.0, sc=0.0, sd=0.0, se=0.0, sf=0.0;
    Kokkos::parallel_reduce("GrassWebRatioSums", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &ra, Real &rb, Real &rc, Real &rd, Real &re, Real &rf) {
      int m = idx/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks; j += js;
      if (w0d(m,IDN,k,j,i) <= rho_lo) { return; }
      Real x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      Real x2v = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      Real rcyl = sqrt(x1v*x1v + x2v*x2v);
      Real dx1 = size.d_view(m).dx1, dx2 = size.d_view(m).dx2, dx3 = size.d_view(m).dx3;
      if (rcyl < 1.0e-6*dx1) { return; }
      Real phix = -x2v/rcyl, phiy = x1v/rcyl;
      Real rhx = x1v/rcyl, rhy = x2v/rcyl;
      Real btorP = bxP(m,k,j,i)*phix + byP(m,k,j,i)*phiy;
      Real btorT = bxT(m,k,j,i)*phix + byT(m,k,j,i)*phiy;
      Real bRP = bxP(m,k,j,i)*rhx + byP(m,k,j,i)*rhy, bZP = bzP(m,k,j,i);
      Real bRT = bxT(m,k,j,i)*rhx + byT(m,k,j,i)*rhy, bZT = bzT(m,k,j,i);
      Real detg = adm::SpatialDet(admd.g_dd(m,0,0,k,j,i), admd.g_dd(m,0,1,k,j,i),
                                   admd.g_dd(m,0,2,k,j,i), admd.g_dd(m,1,1,k,j,i),
                                   admd.g_dd(m,1,2,k,j,i), admd.g_dd(m,2,2,k,j,i));
      Real dV = sqrt(fmax(detg, 0.0)) * dx1*dx2*dx3;
      ra += btorP*btorP*dV;
      rb += 2.0*btorP*btorT*dV;
      rc += btorT*btorT*dV;
      rd += (bRP*bRP + bZP*bZP)*dV;
      re += 2.0*(bRP*bRT + bZP*bZT)*dV;
      rf += (bRT*bRT + bZT*bZT)*dV;
    }, Kokkos::Sum<Real>(sa), Kokkos::Sum<Real>(sb), Kokkos::Sum<Real>(sc),
       Kokkos::Sum<Real>(sd), Kokkos::Sum<Real>(se), Kokkos::Sum<Real>(sf));
#if MPI_PARALLEL_ENABLED
    {
      Real local6[6] = {sa, sb, sc, sd, se, sf};
      Real global6[6];
      MPI_Allreduce(local6, global6, 6, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
      sa=global6[0]; sb=global6[1]; sc=global6[2]; sd=global6[3]; se=global6[4]; sf=global6[5];
    }
#endif
    Real qA = sc - web_tor_pol*sf, qB = sb - web_tor_pol*se, qC = sa - web_tor_pol*sd;
    Real lam_T = 0.0;
    bool have_root = false;
    if (std::abs(qA) < 1.0e-12*std::max({std::abs(qB), std::abs(qC), 1.0})) {
      if (std::abs(qB) > 0.0) { lam_T = -qC/qB; have_root = (lam_T > 0.0); }
    } else {
      Real disc = qB*qB - 4.0*qA*qC;
      if (disc >= 0.0) {
        Real sq = std::sqrt(disc);
        Real x1r = (-qB+sq)/(2.0*qA), x2r = (-qB-sq)/(2.0*qA);
        bool p1 = x1r > 0.0, p2 = x2r > 0.0;
        if (p1 && p2) {
          lam_T = (std::abs(x1r-mu_star) < std::abs(x2r-mu_star)) ? x1r : x2r;
          have_root = true;
        } else if (p1) {
          lam_T = x1r; have_root = true;
        } else if (p2) {
          lam_T = x2r; have_root = true;
        }
      }
    }
    if (!have_root) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "No positive real root for lambda_T/lambda_P in the web tor/pol "
                << "normalization quadratic (A=" << qA << " B=" << qB << " C=" << qC << ")."
                << std::endl;
      exit(EXIT_FAILURE);
    }
    if (global_variable::my_rank == 0) {
      Real achieved = (sa + sb*lam_T + sc*lam_T*lam_T) /
                       std::max(sd + se*lam_T + sf*lam_T*lam_T, 1.0e-300);
      std::cout << "GRASS magnetic web: lambda_P=1, lambda_T=" << lam_T
                << ", achieved E_tor/E_pol=" << achieved << " (target=" << web_tor_pol << ")"
                << std::endl;
    }

    // ---- Combine aP + lam_T*aT (host), rescale to web_bmax_code, add into final -----
    HostArray4D<Real> web_a1, web_a2, web_a3;
    Kokkos::realloc(web_a1, nmb, n3, n2, n1); Kokkos::realloc(web_a2, nmb, n3, n2, n1);
    Kokkos::realloc(web_a3, nmb, n3, n2, n1);
    for (int m = 0; m < nmb; ++m) {
      for (int k = ks; k <= ke+1; ++k) {
        for (int j = js; j <= je+1; ++j) {
          for (int i = is; i <= ie+1; ++i) {
            web_a1(m,k,j,i) = a1P(m,k,j,i) + lam_T*a1T(m,k,j,i);
            web_a2(m,k,j,i) = a2P(m,k,j,i) + lam_T*a2T(m,k,j,i);
            web_a3(m,k,j,i) = a3P(m,k,j,i) + lam_T*a3T(m,k,j,i);
          }
        }
      }
    }
    DvceArray4D<Real> d_wa1, d_wa2, d_wa3, wbx, wby, wbz;
    Kokkos::realloc(d_wa1, nmb, n3, n2, n1); Kokkos::realloc(d_wa2, nmb, n3, n2, n1);
    Kokkos::realloc(d_wa3, nmb, n3, n2, n1);
    Kokkos::deep_copy(d_wa1, web_a1); Kokkos::deep_copy(d_wa2, web_a2);
    Kokkos::deep_copy(d_wa3, web_a3);
    Kokkos::realloc(wbx, nmb, n3, n2, n1); Kokkos::realloc(wby, nmb, n3, n2, n1);
    Kokkos::realloc(wbz, nmb, n3, n2, n1);
    CurlToBcc(d_wa1, d_wa2, d_wa3, wbx, wby, wbz);

    Real bmax_raw = 0.0;
    {
      const int nmkji2 = nmb*nx3*nx2*nx1;
      Kokkos::parallel_reduce("GrassWebBmax", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji2),
      KOKKOS_LAMBDA(const int &idx, Real &bm) {
        int m = idx/nkji;
        int k = (idx - m*nkji)/nji;
        int j = (idx - m*nkji - k*nji)/nx1;
        int i = (idx - m*nkji - k*nji - j*nx1) + is;
        k += ks; j += js;
        Real bsq = wbx(m,k,j,i)*wbx(m,k,j,i) + wby(m,k,j,i)*wby(m,k,j,i) +
                   wbz(m,k,j,i)*wbz(m,k,j,i);
        bm = fmax(bm, sqrt(bsq));
      }, Kokkos::Max<Real>(bmax_raw));
#if MPI_PARALLEL_ENABLED
      MPI_Allreduce(MPI_IN_PLACE, &bmax_raw, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
    }
    Real web_scale = (bmax_raw > 0.0) ? (web_bmax_code/bmax_raw) : 0.0;
    for (int m = 0; m < nmb; ++m) {
      for (int k = ks; k <= ke+1; ++k) {
        for (int j = js; j <= je+1; ++j) {
          for (int i = is; i <= ie+1; ++i) {
            final_a1(m,k,j,i) += web_scale*web_a1(m,k,j,i);
            final_a2(m,k,j,i) += web_scale*web_a2(m,k,j,i);
            final_a3(m,k,j,i) += web_scale*web_a3(m,k,j,i);
          }
        }
      }
    }
    if (global_variable::my_rank == 0) {
      std::cout << "GRASS magnetic web: max|B|(raw)=" << bmax_raw << ", scale factor="
                << web_scale << " -> web_bmax_gauss target reached." << std::endl;
    }
  }

  // ==== Dipole: current-loop A1/A2 (ported from lorene_bns.cpp), own normalization ===
  if (dipole_enable) {
    Real I_0 = 4.0*dipole_r0*dipole_bmax_code/(23.0*M_PI);
    std::function<Real(Real, Real, Real)> fd1 = [&](Real x, Real y, Real z) -> Real {
      Real a = grass::DipoleA1(x, y, z, I_0, dipole_r0);
      if (dipole_confine) {
        Real rho = data.InterpolateRho(x, y, z, eos_table);
        a *= grass::WebEnvelope(rho, rho_lo, rho_hi);
      }
      return a;
    };
    std::function<Real(Real, Real, Real)> fd2 = [&](Real x, Real y, Real z) -> Real {
      Real a = grass::DipoleA2(x, y, z, I_0, dipole_r0);
      if (dipole_confine) {
        Real rho = data.InterpolateRho(x, y, z, eos_table);
        a *= grass::WebEnvelope(rho, rho_lo, rho_hi);
      }
      return a;
    };
    std::function<Real(Real, Real, Real)> fd3 = [](Real, Real, Real) -> Real { return 0.0; };

    HostArray4D<Real> dip_a1, dip_a2, dip_a3;
    Kokkos::realloc(dip_a1, nmb, n3, n2, n1); Kokkos::realloc(dip_a2, nmb, n3, n2, n1);
    Kokkos::realloc(dip_a3, nmb, n3, n2, n1);
    BuildAndCorrect(dip_a1, dip_a2, dip_a3, false, fd1, fd2, fd3);

    DvceArray4D<Real> d_dip_a1, d_dip_a2, d_dip_a3, dbx, dby, dbz;
    Kokkos::realloc(d_dip_a1, nmb, n3, n2, n1); Kokkos::realloc(d_dip_a2, nmb, n3, n2, n1);
    Kokkos::realloc(d_dip_a3, nmb, n3, n2, n1);
    Kokkos::deep_copy(d_dip_a1, dip_a1); Kokkos::deep_copy(d_dip_a2, dip_a2);
    Kokkos::deep_copy(d_dip_a3, dip_a3);
    Kokkos::realloc(dbx, nmb, n3, n2, n1); Kokkos::realloc(dby, nmb, n3, n2, n1);
    Kokkos::realloc(dbz, nmb, n3, n2, n1);
    CurlToBcc(d_dip_a1, d_dip_a2, d_dip_a3, dbx, dby, dbz);

    int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
    const int nkji = nx3*nx2*nx1, nji = nx2*nx1;
    Real dip_bmax_raw = 0.0;
    {
      const int nmkji2 = nmb*nx3*nx2*nx1;
      Kokkos::parallel_reduce("GrassDipoleBmax", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji2),
      KOKKOS_LAMBDA(const int &idx, Real &bm) {
        int m = idx/nkji;
        int k = (idx - m*nkji)/nji;
        int j = (idx - m*nkji - k*nji)/nx1;
        int i = (idx - m*nkji - k*nji - j*nx1) + is;
        k += ks; j += js;
        Real bsq = dbx(m,k,j,i)*dbx(m,k,j,i) + dby(m,k,j,i)*dby(m,k,j,i) +
                   dbz(m,k,j,i)*dbz(m,k,j,i);
        bm = fmax(bm, sqrt(bsq));
      }, Kokkos::Max<Real>(dip_bmax_raw));
#if MPI_PARALLEL_ENABLED
      MPI_Allreduce(MPI_IN_PLACE, &dip_bmax_raw, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
    }
    Real dip_scale = (dip_bmax_raw > 0.0) ? (dipole_bmax_code/dip_bmax_raw) : 0.0;
    for (int m = 0; m < nmb; ++m) {
      for (int k = ks; k <= ke+1; ++k) {
        for (int j = js; j <= je+1; ++j) {
          for (int i = is; i <= ie+1; ++i) {
            final_a1(m,k,j,i) += dip_scale*dip_a1(m,k,j,i);
            final_a2(m,k,j,i) += dip_scale*dip_a2(m,k,j,i);
            // final_a3 untouched -- dipole's own a3 is identically zero.
          }
        }
      }
    }
    if (global_variable::my_rank == 0) {
      std::cout << "GRASS dipole: max|B|(raw)=" << dip_bmax_raw << ", scale factor="
                << dip_scale << " -> dipole_bmax_gauss target reached (confine="
                << dipole_confine << ")." << std::endl;
    }
  }

  // ==== Final curl -> b0, reconstruct bcc0 (verbatim lorene_bns.cpp pattern) =========
  DvceArray4D<Real> d_fa1, d_fa2, d_fa3;
  Kokkos::realloc(d_fa1, nmb, n3, n2, n1); Kokkos::realloc(d_fa2, nmb, n3, n2, n1);
  Kokkos::realloc(d_fa3, nmb, n3, n2, n1);
  Kokkos::deep_copy(d_fa1, final_a1); Kokkos::deep_copy(d_fa2, final_a2);
  Kokkos::deep_copy(d_fa3, final_a3);

  par_for("grass_final_Bfc", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real dx1 = size.d_view(m).dx1, dx2 = size.d_view(m).dx2, dx3 = size.d_view(m).dx3;
    b0.x1f(m,k,j,i) = (d_fa3(m,k,j+1,i)-d_fa3(m,k,j,i))/dx2 - (d_fa2(m,k+1,j,i)-d_fa2(m,k,j,i))/dx3;
    b0.x2f(m,k,j,i) = (d_fa1(m,k+1,j,i)-d_fa1(m,k,j,i))/dx3 - (d_fa3(m,k,j,i+1)-d_fa3(m,k,j,i))/dx1;
    b0.x3f(m,k,j,i) = (d_fa2(m,k,j,i+1)-d_fa2(m,k,j,i))/dx1 - (d_fa1(m,k,j+1,i)-d_fa1(m,k,j,i))/dx2;
    if (i==ie) {
      b0.x1f(m,k,j,i+1) = (d_fa3(m,k,j+1,i+1)-d_fa3(m,k,j,i+1))/dx2 -
                          (d_fa2(m,k+1,j,i+1)-d_fa2(m,k,j,i+1))/dx3;
    }
    if (j==je) {
      b0.x2f(m,k,j+1,i) = (d_fa1(m,k+1,j+1,i)-d_fa1(m,k,j+1,i))/dx3 -
                          (d_fa3(m,k,j+1,i+1)-d_fa3(m,k,j+1,i))/dx1;
    }
    if (k==ke) {
      b0.x3f(m,k+1,j,i) = (d_fa2(m,k+1,j,i+1)-d_fa2(m,k+1,j,i))/dx1 -
                          (d_fa1(m,k+1,j+1,i)-d_fa1(m,k+1,j,i))/dx2;
    }
  });
  par_for("grass_final_bcc", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    bcc0(m,IBX,k,j,i) = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
    bcc0(m,IBY,k,j,i) = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
    bcc0(m,IBZ,k,j,i) = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
  });

  // ==== Startup diagnostics: E_mag, E_mag/M_rest (proxy for |W|), max|div B| ==========
  {
    auto &admd = pmbp->padm->adm;
    auto &w0d = pmbp->pmhd->w0;
    int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
    const int nmkji = nmb*nx3*nx2*nx1;
    const int nkji = nx3*nx2*nx1, nji = nx2*nx1;
    Real e_mag = 0.0, m_rest = 0.0;
    Kokkos::parallel_reduce("GrassWebEmag", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &em, Real &mr) {
      int m = idx/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks; j += js;
      Real dx1 = size.d_view(m).dx1, dx2 = size.d_view(m).dx2, dx3 = size.d_view(m).dx3;
      Real detg = adm::SpatialDet(admd.g_dd(m,0,0,k,j,i), admd.g_dd(m,0,1,k,j,i),
                                   admd.g_dd(m,0,2,k,j,i), admd.g_dd(m,1,1,k,j,i),
                                   admd.g_dd(m,1,2,k,j,i), admd.g_dd(m,2,2,k,j,i));
      Real dV = sqrt(fmax(detg, 0.0)) * dx1*dx2*dx3;
      Real bsq = bcc0(m,IBX,k,j,i)*bcc0(m,IBX,k,j,i) + bcc0(m,IBY,k,j,i)*bcc0(m,IBY,k,j,i) +
                 bcc0(m,IBZ,k,j,i)*bcc0(m,IBZ,k,j,i);
      em += 0.5*bsq*dV;
      mr += w0d(m,IDN,k,j,i)*dV;
    }, Kokkos::Sum<Real>(e_mag), Kokkos::Sum<Real>(m_rest));

    Real divb_max = 0.0;
    Kokkos::parallel_reduce("GrassWebDivB", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &dm) {
      int m = idx/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks; j += js;
      Real dx1 = size.d_view(m).dx1, dx2 = size.d_view(m).dx2, dx3 = size.d_view(m).dx3;
      Real divb = (b0.x1f(m,k,j,i+1)-b0.x1f(m,k,j,i))/dx1 +
                  (b0.x2f(m,k,j+1,i)-b0.x2f(m,k,j,i))/dx2 +
                  (b0.x3f(m,k+1,j,i)-b0.x3f(m,k,j,i))/dx3;
      Real bmag = sqrt(bcc0(m,IBX,k,j,i)*bcc0(m,IBX,k,j,i) + bcc0(m,IBY,k,j,i)*bcc0(m,IBY,k,j,i) +
                        bcc0(m,IBZ,k,j,i)*bcc0(m,IBZ,k,j,i));
      Real dx_local = std::cbrt(dx1*dx2*dx3);
      dm = fmax(dm, (bmag > 1.0e-300) ? std::abs(divb)*dx_local/bmag : 0.0);
    }, Kokkos::Max<Real>(divb_max));

#if MPI_PARALLEL_ENABLED
    {
      Real local2[2] = {e_mag, m_rest};
      Real global2[2];
      MPI_Allreduce(local2, global2, 2, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
      e_mag = global2[0]; m_rest = global2[1];
      MPI_Allreduce(MPI_IN_PLACE, &divb_max, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
    }
#endif
    if (global_variable::my_rank == 0) {
      Real ratio = (m_rest > 0.0) ? (e_mag/m_rest) : 0.0;
      std::cout << "GRASS magnetic field: E_mag=" << e_mag
                << ", E_mag/M_rest=" << ratio << " (M_rest=int(rho0)dV, a rest-mass-"
                << "energy PROXY for |W| -- not a true binding-energy calculation; warn "
                << "above 1e-3), max|div(B)|*dx/|B|=" << divb_max << std::endl;
      if (ratio > 1.0e-3) {
        std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "E_mag/M_rest=" << ratio << " exceeds 1e-3 -- the field may be "
                  << "dynamically significant enough to need a constraint re-solve."
                  << std::endl;
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//  \brief Sets initial conditions for a GRASS-built (possibly differentially rotating)
//  neutron star, under either <z4c> or <cfc>/<adm>.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!pmbp->pcoord->is_dynamical_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "GRASS-initial-data problem requires a <z4c> or <cfc>/<adm> block "
              << "(dynamical-relativistic spacetime evolution)" << std::endl;
    exit(EXIT_FAILURE);
  }

  user_hist_func = &GrassHistory;

  // No restart-path special-casing: this pgen only ever sets initial data once --
  // evolution afterward proceeds from checkpointed Z4c/CFC/hydro state like any other
  // dynamical run (same reasoning as dyngr_rns_st.cpp).
  if (restart) { return; }

  // Note: GRASS's own (pressure, energy) pair is reused directly -- no eos_policy
  // dispatch/template parameter is needed here, unlike the TOV-family pgens.
  SetupGrass(pin, pmy_mesh_);

  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;

  pmbp->pdyngr->PrimToConInit(0, (n1-1), 0, (n2-1), 0, (n3-1));

  if (pmbp->pz4c != nullptr) {
    switch (indcs.ng) {
      case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
              pmbp->pz4c->ADMConstraints<2>(pmbp);
              break;
      case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
              pmbp->pz4c->ADMConstraints<3>(pmbp);
              break;
      case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
              pmbp->pz4c->ADMConstraints<4>(pmbp);
              break;
    }
  }

  return;
}

// History function: rho-max, alpha-min (same diagnostics as dyngr_tov.cpp's TOVHistory
// / dyngr_rns_st.cpp's RnsStHistory, minus the scalar-field entries -- no scalar field
// here) plus, per magnetic_web_id_athenak.md section 7, magnetic diagnostics: E_mag,
// E_tor, E_pol (WHOLE-domain, not restricted to the initial rho_lo shell -- unlike the
// one-time normalization pass in BuildMagneticField, restricting an ONGOING diagnostic
// to the field's initial confinement boundary would become physically meaningless once
// the star evolves and the field moves), Bmax, max|div(B)|*dx/|B|. Helicity and total
// angular momentum J are NOT implemented here (see grass/VALIDATION.md) -- helicity
// would need the vector potential A, which isn't retained after BuildMagneticField
// returns, and J needs a separate integral not otherwise used by this pgen; both are
// scoped out of this first implementation pass, not silently forgotten.

void GrassHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 7;
  pdata->label[0] = "rho-max";
  pdata->label[1] = "alpha-min";
  pdata->label[2] = "E_mag";
  pdata->label[3] = "E_tor";
  pdata->label[4] = "E_pol";
  pdata->label[5] = "Bmax";
  pdata->label[6] = "divBmax";

  auto &w0_ = pm->pmb_pack->pmhd->w0;
  auto &adm = pm->pmb_pack->padm->adm;
  auto &bcc0_ = pm->pmb_pack->pmhd->bcc0;
  auto &b0_ = pm->pmb_pack->pmhd->b0;
  auto &size_ = pm->pmb_pack->pmb->mb_size;

  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is; int nx1 = indcs.nx1;
  int js = indcs.js; int nx2 = indcs.nx2;
  int ks = indcs.ks; int nx3 = indcs.nx3;
  const int nmkji = (pm->pmb_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  Real rho_max = std::numeric_limits<Real>::max();
  Real alpha_min = -rho_max;
  Real bmax = 0.0;
  Kokkos::parallel_reduce("GrassHistSums", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &mb_max, Real &mb_alp_min, Real &mb_bmax) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    mb_max = fmax(mb_max, w0_(m, IDN, k, j, i));
    mb_alp_min = fmin(mb_alp_min, adm.alpha(m, k, j, i));
    Real bsq = bcc0_(m,IBX,k,j,i)*bcc0_(m,IBX,k,j,i) + bcc0_(m,IBY,k,j,i)*bcc0_(m,IBY,k,j,i) +
               bcc0_(m,IBZ,k,j,i)*bcc0_(m,IBZ,k,j,i);
    mb_bmax = fmax(mb_bmax, sqrt(bsq));
  }, Kokkos::Max<Real>(rho_max), Kokkos::Min<Real>(alpha_min), Kokkos::Max<Real>(bmax));

  Real e_mag = 0.0, e_tor = 0.0, e_pol = 0.0, divb_max = 0.0;
  Kokkos::parallel_reduce("GrassHistMag", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &em, Real &etor, Real &epol, Real &dbm) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    Real dx1 = size_.d_view(m).dx1, dx2 = size_.d_view(m).dx2, dx3 = size_.d_view(m).dx3;
    Real detg = adm::SpatialDet(adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
                                 adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
                                 adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i));
    Real dV = sqrt(fmax(detg, 0.0)) * dx1*dx2*dx3;
    Real x1v = CellCenterX(i-is, nx1, size_.d_view(m).x1min, size_.d_view(m).x1max);
    Real x2v = CellCenterX(j-js, nx2, size_.d_view(m).x2min, size_.d_view(m).x2max);
    Real rcyl = sqrt(x1v*x1v + x2v*x2v);
    Real bx = bcc0_(m,IBX,k,j,i), by = bcc0_(m,IBY,k,j,i), bz = bcc0_(m,IBZ,k,j,i);
    em += 0.5*(bx*bx + by*by + bz*bz)*dV;
    if (rcyl > 1.0e-6*dx1) {
      Real phix = -x2v/rcyl, phiy = x1v/rcyl, rhx = x1v/rcyl, rhy = x2v/rcyl;
      Real btor = bx*phix + by*phiy;
      Real bR = bx*rhx + by*rhy;
      etor += 0.5*btor*btor*dV;
      epol += 0.5*(bR*bR + bz*bz)*dV;
    }
    Real divb = (b0_.x1f(m,k,j,i+1)-b0_.x1f(m,k,j,i))/dx1 +
                (b0_.x2f(m,k,j+1,i)-b0_.x2f(m,k,j,i))/dx2 +
                (b0_.x3f(m,k+1,j,i)-b0_.x3f(m,k,j,i))/dx3;
    Real bmag = sqrt(bx*bx + by*by + bz*bz);
    Real dx_local = cbrt(dx1*dx2*dx3);
    dbm = fmax(dbm, (bmag > 1.0e-300) ? fabs(divb)*dx_local/bmag : 0.0);
  }, Kokkos::Sum<Real>(e_mag), Kokkos::Sum<Real>(e_tor), Kokkos::Sum<Real>(e_pol),
     Kokkos::Max<Real>(divb_max));

#if MPI_PARALLEL_ENABLED
  if (global_variable::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &rho_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &alpha_min, 1, MPI_ATHENA_REAL, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &bmax, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &divb_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &e_mag, 1, MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &e_tor, 1, MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &e_pol, 1, MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&rho_max, &rho_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&alpha_min, &alpha_min, 1, MPI_ATHENA_REAL, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&bmax, &bmax, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&divb_max, &divb_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&e_mag, &e_mag, 1, MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&e_tor, &e_tor, 1, MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&e_pol, &e_pol, 1, MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
    rho_max = 0.;
    alpha_min = 0.;
    bmax = 0.;
    divb_max = 0.;
    e_mag = 0.;
    e_tor = 0.;
    e_pol = 0.;
  }
#endif

  pdata->hdata[0] = rho_max;
  pdata->hdata[1] = alpha_min;
  pdata->hdata[2] = e_mag;
  pdata->hdata[3] = e_tor;
  pdata->hdata[4] = e_pol;
  pdata->hdata[5] = bmax;
  pdata->hdata[6] = divb_max;
}
