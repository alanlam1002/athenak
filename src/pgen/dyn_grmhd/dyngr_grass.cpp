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
#include <cstdio>
#include <functional>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

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

// b0/bcc0 hold the DENSITIZED field Btilde^i = sqrt(gamma)*B^i: PrimToCons and
// ConsToPrim divide bcc0 by sqrt(gamma) (eos/primitive_solver_hyd.hpp), and the CT
// update evolves the face values as fluxes. The discrete curl of the edge potential
// A_i therefore yields Btilde directly (same as lorene_bns.cpp). All magnetic
// diagnostics below convert to the physical Eulerian field B^i = Btilde^i/sqrt(gamma)
// with the cell's own g_dd: |B|^2 = g_ij B^i B^j, energy density |B|^2/2 (code
// units, Heaviside-Lorentz), volume element sqrt(gamma) d^3x. Components along the
// flat unit directions e (varpi-hat, phi-hat, z-hat) are g(B,e)/sqrt(g(e,e)); the
// poloidal energy is |B|^2 - B_phi^2.
struct GrassPhysB {
  Real bvarpi, bphi, bz;   // physical components (code units)
  Real bsq;                // g_ij B^i B^j
  Real sdetg;              // sqrt(gamma)
};

KOKKOS_INLINE_FUNCTION
GrassPhysB GrassPhysicalB(Real gxx, Real gxy, Real gxz, Real gyy, Real gyz, Real gzz,
                          Real btx, Real bty, Real btz, Real x, Real y) {
  GrassPhysB out;
  Real detg = adm::SpatialDet(gxx, gxy, gxz, gyy, gyz, gzz);
  out.sdetg = Kokkos::sqrt(fmax(detg, 1.0e-300));
  Real bx = btx/out.sdetg, by = bty/out.sdetg, bz = btz/out.sdetg;
  Real bdx = gxx*bx + gxy*by + gxz*bz;
  Real bdy = gxy*bx + gyy*by + gyz*bz;
  Real bdz = gxz*bx + gyz*by + gzz*bz;
  out.bsq = bdx*bx + bdy*by + bdz*bz;
  out.bz = bdz/Kokkos::sqrt(gzz);
  Real w = Kokkos::sqrt(x*x + y*y);
  if (w > 0.0) {
    Real ex = x/w, ey = y/w;   // varpi-hat; phi-hat = (-ey, ex)
    out.bvarpi = (bdx*ex + bdy*ey)/Kokkos::sqrt(gxx*ex*ex + 2.0*gxy*ex*ey + gyy*ey*ey);
    out.bphi = (-bdx*ey + bdy*ex)/Kokkos::sqrt(gxx*ey*ey - 2.0*gxy*ex*ey + gyy*ex*ex);
  } else {
    out.bvarpi = 0.0;
    out.bphi = 0.0;
  }
  return out;
}

// Prototype for user-defined history function
void GrassHistory(HistoryData *pdata, Mesh *pm);

// Prototype for the magnetic-web + dipole initial B-field builder (see its own
// definition below for the full design; magnetic_web_id_athenak.md, repo root,
// is the physics/implementation spec this follows). Reads only padm->adm
// (g_dd/mode-agnostic by construction, see file header above) and pmhd
// w0/b0/bcc0 -- no z4c/CFC-specific state, so this ported over unmodified.
void BuildMagneticField(ParameterInput *pin, Mesh *pmy_mesh_,
                         const grass::GrassData &data, const grass::GrassEosTable &eos_table);

// Startup report on the installed field that needs the conserved variables (defined
// below; called from UserProblem after PrimToConInit).
void GrassMagneticReport(ParameterInput *pin, Mesh *pm);

// Web-run diagnostics (<basename>.webdiag.txt and azimuthal maps), called from
// GrassHistory; configured in UserProblem (defined below).
void GrassWebDiagnostics(Mesh *pm);

// Gauss -> AthenaK code-unit conversion factor, honoring <mhd> units (same key/
// options as PrimitiveSolverHydro::SetPolicyParams). B^2/2 is the code's energy
// density (Heaviside-Lorentz convention, no 4*pi -- see cfc.cpp's own bsq usage);
// Gaussian-cgs B^2/(8*pi) is the cgs energy density, so
// B_code = B_cgs * sqrt(EnergyDensityConversion / (4*pi)).
Real GaussToCode(ParameterInput *pin) {
  std::string units_str = pin->GetOrAddString("mhd", "units", "geometric_solar");
  Primitive::UnitSystem code_units;
  if (units_str == "geometric_solar") {
    code_units = Primitive::MakeGeometricSolar();
  } else if (units_str == "geometric_kilometer") {
    code_units = Primitive::MakeGeometricKilometer();
  } else if (units_str == "nuclear") {
    code_units = Primitive::MakeNuclear();
  } else if (units_str == "cgs") {
    code_units = Primitive::MakeCGS();
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Unknown <mhd> units '" << units_str << "' requested." << std::endl;
    exit(EXIT_FAILURE);
  }
  Real ed_conv = Primitive::MakeCGS().EnergyDensityConversion(code_units);
  return Kokkos::sqrt(ed_conv / (4.0*M_PI));
}

//----------------------------------------------------------------------------------------
//! \fn void SetupGrass
//  \brief Loads a GRASS restart binary and fills ADM/(Z4c or CFC)/hydro initial data.
//  Host-only, mirrors the host-fill-then-deep_copy pattern used by the other
//  external-ID pgens (elliptica/lorene/sgrid/celephais/rns_st). Templated on
//  IsZ4c/UseYe so the per-point metric-write and Y[e]-write branches are resolved
//  once per instantiation instead of per grid point; dispatched in UserProblem from
//  a runtime pz4c/nscalars check. UseYe's guard is also a bounds guard: host_w0's
//  width is nmhd+nscalars, so writing IYF when nscalars==0 would be out of bounds.

template<bool IsZ4c, bool UseYe>
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

  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;

  auto &u_adm = pmbp->padm->u_adm;
  auto &w0 = pmbp->pmhd->w0;

  // Host-side fill, then move to device -- see file header.
  HostArray5D<Real>::HostMirror host_u_adm = Kokkos::create_mirror_view(u_adm);
  HostArray5D<Real>::HostMirror host_w0 = Kokkos::create_mirror_view(w0);
  HostArray5D<Real>::HostMirror host_u_z4c;

  // Gauge variables (alpha, beta_u) live in Z4c's own evolved state under <z4c> (ADMToZ4c
  // only derives the geometric/conformal part from g_dd/vK_dd, it never touches the
  // gauge), but in u_adm itself under <cfc>/bare <adm> -- matches adm::ADM's own
  // constructor logic (adm.cpp). g_dd/vK_dd/psi4 always live in u_adm regardless.
  adm::ADM::ADMhost_vars host_adm;
  if constexpr (IsZ4c) {
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

          Real vu[3] = {0.0, 0.0, 0.0};
          if constexpr (IsZ4c) {
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

            if (pt.rho0 > 0.0) {
              vu[0] = pt.vu[0]; vu[1] = pt.vu[1]; vu[2] = pt.vu[2];
            }
          } else {
            // CFC can only represent a conformally-flat 3-metric; isotropize GRASS's
            // true (anisotropic) metric into a conformal-factor guess that matches its
            // local volume element (det g_dd). vK_dd is zeroed -- CFC::InitializeMetric()
            // always overwrites both from its own elliptic solve (see file header).
            Real gxx = pt.g_dd[0], gxy = pt.g_dd[1], gxz = pt.g_dd[2];
            Real gyy = pt.g_dd[3], gyz = pt.g_dd[4], gzz = pt.g_dd[5];
            Real det_g = gxx*(gyy*gzz - gyz*gyz) - gxy*(gxy*gzz - gyz*gxz)
                         + gxz*(gxy*gyz - gyy*gxz);
            Real psi4_guess = Kokkos::cbrt(Kokkos::max(det_g, 1.0e-300));

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

            if (pt.rho0 > 0.0) {
              Real vx = pt.vu[0], vy = pt.vu[1], vz = pt.vu[2];
              // Preserve the true Lorentz factor exactly: rescale the whole velocity
              // vector by one scalar so gamma_iso*(s*v)^2 matches the true, full
              // (cross-term-including) v^2 = gamma_true_ij*v^i*v^j -- tighter than
              // rescaling each component independently, which only matches the
              // diagonal contributions and silently drops the off-diagonal terms
              // an isotropized metric can't represent anyway.
              Real v2_true = gxx*vx*vx + gyy*vy*vy + gzz*vz*vz
                             + 2.0*(gxy*vx*vy + gxz*vx*vz + gyz*vy*vz);
              Real v2_iso = vx*vx + vy*vy + vz*vz;
              Real vscale = (v2_iso > 0.0) ?
                  Kokkos::sqrt(Kokkos::max(v2_true, 0.0) / (psi4_guess*v2_iso)) : 0.0;
              vu[0] = vx*vscale; vu[1] = vy*vscale; vu[2] = vz*vscale;
            }
          }

          Real rho = (pt.rho0 > 0.0) ? pt.rho0 : 0.0;
          Real pres = (pt.rho0 > 0.0) ? pt.pres : 0.0;

          host_w0(m, IDN, k, j, i) = rho;
          host_w0(m, IPR, k, j, i) = pres;
          host_w0(m, IVX, k, j, i) = vu[0];
          host_w0(m, IVY, k, j, i) = vu[1];
          host_w0(m, IVZ, k, j, i) = vu[2];
          if constexpr (UseYe) {
            host_w0(m, IYF, k, j, i) = pt.Yq;
          }
        }
      }
    }
  }

  Kokkos::deep_copy(u_adm, host_u_adm);
  Kokkos::deep_copy(w0, host_w0);
  if constexpr (IsZ4c) {
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

  const Real gauss_to_code = GaussToCode(pin);
  grass::GrassUnits units;
  const Real rho_to_cgs = 1.0/units.MassDensityCgsToCode(1.0);
  const Real len_to_km = 1.0/units.LengthCgsToCode(1.0e5);
  const Real dx_finest = pmy_mesh_->mesh_size.dx1 /
                         std::pow(2.0, pmy_mesh_->max_level - pmy_mesh_->root_level);

  // ---- <problem> keys ---------------------------------------------------------------
  // Normalizations (web_field_design_P_CFC.md §3-4):
  //   web_norm = bmax : max|B_web| = web_bmax_gauss (spec default)
  //            = rms  : rms |B_web| over cells with h(rho) >= 0.5 = web_brms_gauss
  //   dipole_norm = bmax       : max|B_dip| = dipole_bmax_gauss (spec default)
  //               = bvarpi_rms : rms over h >= 0.5 of the azimuthal-mean cylindrical-
  //                              radial component Bbar_varpi = dipole_bvarpi_rms_gauss
  // Both rms forms use the physical field and the proper volume sqrt(gamma) d^3x.
  bool dipole_confine = pin->GetOrAddInteger("problem", "dipole_confine", 0) != 0;
  // envelope shape (web_envelope = bump|poly3, see grass::WebEnvelope) and the tor/pol
  // definition (web_tor_pol_def = cylindrical|chandrasekhar, see the quadratic solve)
  const std::string env_name = pin->GetOrAddString("problem", "web_envelope", "bump");
  const int env_type = (env_name == "poly3") ? 1 : 0;
  const std::string tor_pol_def = pin->GetOrAddString("problem", "web_tor_pol_def",
                                                      "cylindrical");
  if ((env_name != "bump" && env_name != "poly3") ||
      (tor_pol_def != "cylindrical" && tor_pol_def != "chandrasekhar")) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "web_envelope must be bump|poly3 and web_tor_pol_def "
              << "cylindrical|chandrasekhar" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::string web_norm = "bmax", dipole_norm = "bmax";
  if (web_enable) { web_norm = pin->GetOrAddString("problem", "web_norm", "bmax"); }
  if (dipole_enable) { dipole_norm = pin->GetOrAddString("problem", "dipole_norm", "bmax"); }
  if ((web_norm != "bmax" && web_norm != "rms") ||
      (dipole_norm != "bmax" && dipole_norm != "bvarpi_rms")) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "web_norm must be bmax|rms and dipole_norm bmax|bvarpi_rms (got '"
              << web_norm << "', '" << dipole_norm << "')" << std::endl;
    exit(EXIT_FAILURE);
  }
  bool need_h = web_enable || (dipole_enable && (dipole_confine || dipole_norm != "bmax"));
  Real rho_lo = 0.0, rho_hi = 0.0;
  if (need_h) {
    rho_lo = pin->GetReal("problem", "web_rho_lo");
    rho_hi = pin->GetReal("problem", "web_rho_hi");
  }
  Real mu_star = 2.0, web_target = 0.0, web_tor_pol = 4.0, web_kmin = 0.0, web_kmax = 0.0;
  if (web_enable) {
    mu_star = pin->GetOrAddReal("problem", "web_mu_star", 2.0);
    web_target = (web_norm == "rms") ? pin->GetReal("problem", "web_brms_gauss")
                                     : pin->GetOrAddReal("problem", "web_bmax_gauss", 1.0e16);
    web_target *= gauss_to_code;
    web_tor_pol = pin->GetOrAddReal("problem", "web_tor_pol", 4.0);
    web_kmin = pin->GetReal("problem", "web_kmin");
    web_kmax = pin->GetReal("problem", "web_kmax");
  }
  // Test-only switches for the spec's V3/V4 checks: h == 1 everywhere (no envelope on
  // the web), and lambda_T = lambda_P = 1 without the tor/pol solve.
  const bool web_test_h_one = pin->GetOrAddInteger("problem", "web_test_h_one", 0) != 0;
  const bool web_test_fixed_lambda =
      pin->GetOrAddInteger("problem", "web_test_fixed_lambda", 0) != 0;
  Real dipole_target = 0.0, dipole_r0 = 5.0;
  if (dipole_enable) {
    dipole_target = (dipole_norm == "bvarpi_rms")
        ? pin->GetReal("problem", "dipole_bvarpi_rms_gauss")
        : pin->GetOrAddReal("problem", "dipole_bmax_gauss", 1.0e16);
    dipole_target *= gauss_to_code;
    dipole_r0 = pin->GetOrAddReal("problem", "dipole_r0", 5.0);
  }

  // ---- Resolution guard (web only): 2*pi/k_max must resolve on >=8 finest cells -----
  if (web_enable) {
    if (2.0*M_PI/web_kmax < 8.0*dx_finest) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "web_kmax=" << web_kmax << " under-resolved: 2*pi/web_kmax="
                << 2.0*M_PI/web_kmax << " < 8*dx_finest=" << 8.0*dx_finest << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  // ---- rho_max validation + shell -> R_cyl startup diagnostic -----------------------
  Real r_outer_eq = -1.0, r_outer_pole = -1.0;
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
    // ---- Density sweeps along the equator (x axis) and the pole (z axis): the
    // rho_lo/rho_hi -> radius mapping, the h peak, plus a fragmentation guard. The
    // rho_hi<=rho_max check above only guards against rho_hi missing the star
    // entirely -- it does NOT guard against rho_hi sitting strictly between this
    // star's on-axis (r=0) density and its true (possibly off-center) peak, a real
    // risk for differentially rotating remnants whose equatorial density profile can
    // be quasi-toroidal. If rho_hi falls in that band, the excluded (rho>=rho_hi)
    // region is an ANNULUS, so the shell silently fragments into a spurious inner
    // disk plus the intended outer shell (confirmed on grass_diff_DD2_hot_v2,
    // VALIDATION.md 2026-08-14). Detected on the equatorial ray: once the profile has
    // dropped below rho_hi it must never rise back above rho_hi before dropping below
    // rho_lo. InterpolateRho is a deterministic host evaluation, so every rank aborts
    // together.
    struct Sweep { Real r_inner = -1.0, r_outer = -1.0, r_reentry = -1.0, r_hpeak = -1.0; };
    auto sweep = [&](Real ex, Real ez) {
      Sweep sw;
      Real rmax = pmy_mesh_->mesh_size.x1max, hbest = 0.0;
      bool entered_shell = false;
      for (int s = 1; s <= 20000; ++s) {
        Real r = s*(rmax/20000.0);
        Real rho_here = data.InterpolateRho(r*ex, 0.0, r*ez, eos_table);
        Real hh = grass::WebEnvelope(rho_here, rho_lo, rho_hi, env_type);
        if (hh > hbest) { hbest = hh; sw.r_hpeak = r; }
        if (rho_here < rho_hi) {
          if (sw.r_inner < 0.0) { sw.r_inner = r; }
          entered_shell = true;
        } else if (entered_shell && sw.r_reentry < 0.0) {
          sw.r_reentry = r;
        }
        if (rho_here < rho_lo && sw.r_outer < 0.0) { sw.r_outer = r; break; }
      }
      return sw;
    };
    Sweep eq = sweep(1.0, 0.0), pole = sweep(0.0, 1.0);
    r_outer_eq = eq.r_outer;
    r_outer_pole = pole.r_outer;
    if (eq.r_reentry > 0.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "web_rho_hi=" << rho_hi << " sits between this star's on-axis (r=0) "
                << "density and its true (possibly off-center) peak: the equatorial "
                << "density profile drops below rho_hi at r=" << eq.r_inner << " but rises "
                << "back above it at r=" << eq.r_reentry << ". The high-density region "
                << "excluded from the web is therefore an annulus, not a disk containing "
                << "the axis, so the confinement shell fragments into a spurious inner "
                << "disk plus the intended outer shell. Lower web_rho_hi below this "
                << "star's on-axis (r=0) density." << std::endl;
      exit(EXIT_FAILURE);
    }
    if (global_variable::my_rank == 0) {
      std::cout << "GRASS magnetic web: window rho [" << rho_lo << ", " << rho_hi
                << "] code = [" << rho_lo*rho_to_cgs << ", " << rho_hi*rho_to_cgs
                << "] g/cm^3; rho_max = " << rho_max << " code = " << rho_max*rho_to_cgs
                << " g/cm^3" << std::endl;
      std::cout << "GRASS magnetic web: equator (x axis): rho=rho_hi at r=" << eq.r_inner
                << " M (" << eq.r_inner*len_to_km << " km), h peak at " << eq.r_hpeak
                << " M (" << eq.r_hpeak*len_to_km << " km), rho=rho_lo at " << eq.r_outer
                << " M (" << eq.r_outer*len_to_km << " km)" << std::endl;
      std::cout << "GRASS magnetic web: pole (z axis): rho=rho_hi at r=" << pole.r_inner
                << " M (" << pole.r_inner*len_to_km << " km), h peak at " << pole.r_hpeak
                << " M (" << pole.r_hpeak*len_to_km << " km), rho=rho_lo at "
                << pole.r_outer << " M (" << pole.r_outer*len_to_km << " km)" << std::endl;
      if (eq.r_inner > 0.0 && eq.r_outer > eq.r_inner) {
        Real shell_width = eq.r_outer - eq.r_inner;
        if (web_enable) {
          std::cout << "GRASS magnetic web: k in [" << web_kmin << ", " << web_kmax
                    << "] /M = [" << web_kmin/len_to_km << ", " << web_kmax/len_to_km
                    << "] /km; k*R_w in [" << web_kmin*eq.r_outer << ", "
                    << web_kmax*eq.r_outer << "] with R_w = " << eq.r_outer
                    << " M (equatorial rho_lo radius); wavelengths "
                    << 2.0*M_PI/web_kmax*len_to_km << "-" << 2.0*M_PI/web_kmin*len_to_km
                    << " km = " << 2.0*M_PI/web_kmax/dx_finest << "-"
                    << 2.0*M_PI/web_kmin/dx_finest << " cells at dx_finest="
                    << dx_finest << " M (" << dx_finest*len_to_km*1.0e3 << " m)"
                    << std::endl;
        }
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

  // ---- Cell reductions shared by the normalization passes and the startup report ----
  auto &w0d = pmbp->pmhd->w0;
  auto &admd = pmbp->padm->adm;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmkji = nmb*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1, nji = nx2*nx1;

  // max physical |B| over all cells, and rms physical |B| over cells with h >= 0.5,
  // of a densitized cell-centred field
  auto BmaxRms = [&](DvceArray4D<Real> &bx, DvceArray4D<Real> &by, DvceArray4D<Real> &bz,
                     Real *bmax, Real *brms) {
    Real bm = 0.0, s2 = 0.0, sv = 0.0;
    Kokkos::parallel_reduce("GrassWebBmaxRms", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &rbm, Real &rs2, Real &rsv) {
      int m = idx/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks; j += js;
      Real x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      Real x2v = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      GrassPhysB pb = GrassPhysicalB(admd.g_dd(m,0,0,k,j,i), admd.g_dd(m,0,1,k,j,i),
                                     admd.g_dd(m,0,2,k,j,i), admd.g_dd(m,1,1,k,j,i),
                                     admd.g_dd(m,1,2,k,j,i), admd.g_dd(m,2,2,k,j,i),
                                     bx(m,k,j,i), by(m,k,j,i), bz(m,k,j,i), x1v, x2v);
      rbm = fmax(rbm, sqrt(pb.bsq));
      if (grass::WebEnvelope(w0d(m,IDN,k,j,i), rho_lo, rho_hi, env_type) >= 0.5) {
        Real dV = pb.sdetg*size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
        rs2 += pb.bsq*dV;
        rsv += dV;
      }
    }, Kokkos::Max<Real>(bm), Kokkos::Sum<Real>(s2), Kokkos::Sum<Real>(sv));
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &bm, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
    Real l2[2] = {s2, sv};
    MPI_Allreduce(MPI_IN_PLACE, l2, 2, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    s2 = l2[0]; sv = l2[1];
#endif
    *bmax = bm;
    *brms = (sv > 0.0) ? std::sqrt(s2/sv) : 0.0;
  };

  // Azimuthal averages on (varpi, z) bins of width dx_finest (proper-volume weighted):
  // per bin 0: dV, 1: B_varpi dV, 2: B_phi dV, 3: B_z dV, 4: h dV, 5: rho0 dV.
  const Real az_db = dx_finest;
  const Real az_rmax = 1.25*std::max({r_outer_eq, r_outer_pole, 4.0*dx_finest});
  const int az_nw = static_cast<int>(std::ceil(az_rmax/az_db));
  const int az_nz = 2*az_nw;
  const Real az_zmin = -az_nw*az_db;
  constexpr int kAzN = 6;
  using AzHost = typename DvceArray3D<Real>::HostMirror;
  auto AzimuthalSums = [&](DvceArray4D<Real> &bx, DvceArray4D<Real> &by,
                           DvceArray4D<Real> &bz, AzHost *out) {
    DvceArray3D<Real> sums("grass_az_sums", az_nw, az_nz, kAzN);
    par_for("grass_az_sums", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      Real x2v = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      Real x3v = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
      int iw = static_cast<int>(sqrt(x1v*x1v + x2v*x2v)/az_db);
      int iz = static_cast<int>(Kokkos::floor((x3v - az_zmin)/az_db));
      if (iw >= az_nw || iz < 0 || iz >= az_nz) { return; }
      GrassPhysB pb = GrassPhysicalB(admd.g_dd(m,0,0,k,j,i), admd.g_dd(m,0,1,k,j,i),
                                     admd.g_dd(m,0,2,k,j,i), admd.g_dd(m,1,1,k,j,i),
                                     admd.g_dd(m,1,2,k,j,i), admd.g_dd(m,2,2,k,j,i),
                                     bx(m,k,j,i), by(m,k,j,i), bz(m,k,j,i), x1v, x2v);
      Real dV = pb.sdetg*size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
      Real rho = w0d(m,IDN,k,j,i);
      Kokkos::atomic_add(&sums(iw,iz,0), dV);
      Kokkos::atomic_add(&sums(iw,iz,1), pb.bvarpi*dV);
      Kokkos::atomic_add(&sums(iw,iz,2), pb.bphi*dV);
      Kokkos::atomic_add(&sums(iw,iz,3), pb.bz*dV);
      Kokkos::atomic_add(&sums(iw,iz,4), grass::WebEnvelope(rho, rho_lo, rho_hi, env_type)*dV);
      Kokkos::atomic_add(&sums(iw,iz,5), rho*dV);
    });
    *out = Kokkos::create_mirror_view(sums);
    Kokkos::deep_copy(*out, sums);
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, out->data(), static_cast<int>(out->size()),
                  MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  };

  // rms over h >= 0.5 of the azimuthal mean Bbar_varpi (from AzimuthalSums) and of the
  // non-axisymmetric remainder B_varpi - Bbar_varpi
  auto BvarpiMeanRms = [&](DvceArray4D<Real> &bx, DvceArray4D<Real> &by,
                           DvceArray4D<Real> &bz, const AzHost &sums,
                           Real *rms_mean, Real *rms_fluct) {
    DvceArray2D<Real> mean("grass_az_mean", az_nw, az_nz);
    auto mean_h = Kokkos::create_mirror_view(mean);
    for (int iw = 0; iw < az_nw; ++iw) {
      for (int iz = 0; iz < az_nz; ++iz) {
        mean_h(iw,iz) = (sums(iw,iz,0) > 0.0) ? sums(iw,iz,1)/sums(iw,iz,0) : 0.0;
      }
    }
    Kokkos::deep_copy(mean, mean_h);
    Real s2m = 0.0, s2f = 0.0, sv = 0.0;
    Kokkos::parallel_reduce("GrassAzRms", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &rm, Real &rf, Real &rv) {
      int m = idx/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks; j += js;
      if (grass::WebEnvelope(w0d(m,IDN,k,j,i), rho_lo, rho_hi, env_type) < 0.5) { return; }
      Real x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      Real x2v = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      Real x3v = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
      int iw = static_cast<int>(sqrt(x1v*x1v + x2v*x2v)/az_db);
      int iz = static_cast<int>(Kokkos::floor((x3v - az_zmin)/az_db));
      if (iw >= az_nw || iz < 0 || iz >= az_nz) { return; }
      GrassPhysB pb = GrassPhysicalB(admd.g_dd(m,0,0,k,j,i), admd.g_dd(m,0,1,k,j,i),
                                     admd.g_dd(m,0,2,k,j,i), admd.g_dd(m,1,1,k,j,i),
                                     admd.g_dd(m,1,2,k,j,i), admd.g_dd(m,2,2,k,j,i),
                                     bx(m,k,j,i), by(m,k,j,i), bz(m,k,j,i), x1v, x2v);
      Real dV = pb.sdetg*size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
      rm += SQR(mean(iw,iz))*dV;
      rf += SQR(pb.bvarpi - mean(iw,iz))*dV;
      rv += dV;
    }, Kokkos::Sum<Real>(s2m), Kokkos::Sum<Real>(s2f), Kokkos::Sum<Real>(sv));
#if MPI_PARALLEL_ENABLED
    Real l3[3] = {s2m, s2f, sv};
    MPI_Allreduce(MPI_IN_PLACE, l3, 3, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    s2m = l3[0]; s2f = l3[1]; sv = l3[2];
#endif
    *rms_mean = (sv > 0.0) ? std::sqrt(s2m/sv) : 0.0;
    *rms_fluct = (sv > 0.0) ? std::sqrt(s2f/sv) : 0.0;
  };

  // Magnetic helicity H = integral A_i Btilde^i d^3x (= integral A.B sqrt(gamma) d^3x):
  // edge potentials averaged to the cell centre, cell-centred densitized B. Gauge
  // invariant when B.n = 0 on the boundary of the support (the h-confined fields).
  auto Helicity = [&](DvceArray4D<Real> &a1, DvceArray4D<Real> &a2, DvceArray4D<Real> &a3,
                      DvceArray4D<Real> &bx, DvceArray4D<Real> &by, DvceArray4D<Real> &bz) {
    Real hel = 0.0;
    Kokkos::parallel_reduce("GrassHelicity", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &rh) {
      int m = idx/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks; j += js;
      Real ac1 = 0.25*(a1(m,k,j,i) + a1(m,k,j+1,i) + a1(m,k+1,j,i) + a1(m,k+1,j+1,i));
      Real ac2 = 0.25*(a2(m,k,j,i) + a2(m,k,j,i+1) + a2(m,k+1,j,i) + a2(m,k+1,j,i+1));
      Real ac3 = 0.25*(a3(m,k,j,i) + a3(m,k,j,i+1) + a3(m,k,j+1,i) + a3(m,k,j+1,i+1));
      rh += (ac1*bx(m,k,j,i) + ac2*by(m,k,j,i) + ac3*bz(m,k,j,i))*
            size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    }, Kokkos::Sum<Real>(hel));
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &hel, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
    return hel;
  };

  HostArray4D<Real> final_a1, final_a2, final_a3;
  Kokkos::realloc(final_a1, nmb, n3, n2, n1);
  Kokkos::realloc(final_a2, nmb, n3, n2, n1);
  Kokkos::realloc(final_a3, nmb, n3, n2, n1);
  Kokkos::deep_copy(final_a1, 0.0);
  Kokkos::deep_copy(final_a2, 0.0);
  Kokkos::deep_copy(final_a3, 0.0);

  // ==== Web: build P/T decomposition, enforce E_tor/E_pol exactly, rescale, combine ==
  Real web_scale = 0.0, web_bmax_raw = 0.0, web_brms_raw = 0.0, hel_web = 0.0;
  if (web_enable) {
    grass::WebModeTable mode_table;
    grass::GenerateWebModeTable(pin, &mode_table);
    if (global_variable::my_rank == 0 && mode_table.nmodes <= 4) {
      for (int p = 0; p < mode_table.nmodes; ++p) {
        std::printf("GRASS magnetic web: mode %d k=(%.17g, %.17g, %.17g) amp=%.17g "
                    "phase=%.17g\n", p, mode_table.kx[p], mode_table.ky[p], mode_table.kz[p],
                    mode_table.amp[p], mode_table.phase[p]);
      }
    }

    std::function<Real(Real, Real, Real)> f1P = [&](Real x, Real y, Real z) -> Real {
      Real rho = data.InterpolateRho(x, y, z, eos_table);
      Real h = web_test_h_one ? 1.0 : grass::WebEnvelope(rho, rho_lo, rho_hi, env_type);
      if (h == 0.0) { return 0.0; }
      Real ax, ay, az;
      grass::WebA(x, y, z, mode_table, mu_star, 0.0, 1.0, &ax, &ay, &az);
      return h*ax;
    };
    std::function<Real(Real, Real, Real)> f2P = [&](Real x, Real y, Real z) -> Real {
      Real rho = data.InterpolateRho(x, y, z, eos_table);
      Real h = web_test_h_one ? 1.0 : grass::WebEnvelope(rho, rho_lo, rho_hi, env_type);
      if (h == 0.0) { return 0.0; }
      Real ax, ay, az;
      grass::WebA(x, y, z, mode_table, mu_star, 0.0, 1.0, &ax, &ay, &az);
      return h*ay;
    };
    std::function<Real(Real, Real, Real)> f3P = [&](Real x, Real y, Real z) -> Real {
      Real rho = data.InterpolateRho(x, y, z, eos_table);
      Real h = web_test_h_one ? 1.0 : grass::WebEnvelope(rho, rho_lo, rho_hi, env_type);
      if (h == 0.0) { return 0.0; }
      Real ax, ay, az;
      grass::WebA(x, y, z, mode_table, mu_star, 0.0, 1.0, &ax, &ay, &az);
      return h*az;
    };
    std::function<Real(Real, Real, Real)> f1T = [&](Real x, Real y, Real z) -> Real {
      Real rho = data.InterpolateRho(x, y, z, eos_table);
      Real h = web_test_h_one ? 1.0 : grass::WebEnvelope(rho, rho_lo, rho_hi, env_type);
      if (h == 0.0) { return 0.0; }
      Real ax, ay, az;
      grass::WebA(x, y, z, mode_table, mu_star, 1.0, 0.0, &ax, &ay, &az);
      return h*ax;
    };
    std::function<Real(Real, Real, Real)> f2T = [&](Real x, Real y, Real z) -> Real {
      Real rho = data.InterpolateRho(x, y, z, eos_table);
      Real h = web_test_h_one ? 1.0 : grass::WebEnvelope(rho, rho_lo, rho_hi, env_type);
      if (h == 0.0) { return 0.0; }
      Real ax, ay, az;
      grass::WebA(x, y, z, mode_table, mu_star, 1.0, 0.0, &ax, &ay, &az);
      return h*ay;
    };
    std::function<Real(Real, Real, Real)> f3T = [&](Real x, Real y, Real z) -> Real {
      Real rho = data.InterpolateRho(x, y, z, eos_table);
      Real h = web_test_h_one ? 1.0 : grass::WebEnvelope(rho, rho_lo, rho_hi, env_type);
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
    // Physical field, proper volume; E_tor = int B_phi^2, E_pol = int (|B|^2 - B_phi^2).
    // The P-T cross term g(B^P,B^T) comes from polarization of |B^P + B^T|^2.
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
      Real g0 = admd.g_dd(m,0,0,k,j,i), g1 = admd.g_dd(m,0,1,k,j,i);
      Real g2 = admd.g_dd(m,0,2,k,j,i), g3 = admd.g_dd(m,1,1,k,j,i);
      Real g4 = admd.g_dd(m,1,2,k,j,i), g5 = admd.g_dd(m,2,2,k,j,i);
      GrassPhysB pP = GrassPhysicalB(g0, g1, g2, g3, g4, g5, bxP(m,k,j,i), byP(m,k,j,i),
                                     bzP(m,k,j,i), x1v, x2v);
      GrassPhysB pT = GrassPhysicalB(g0, g1, g2, g3, g4, g5, bxT(m,k,j,i), byT(m,k,j,i),
                                     bzT(m,k,j,i), x1v, x2v);
      GrassPhysB pS = GrassPhysicalB(g0, g1, g2, g3, g4, g5, bxP(m,k,j,i) + bxT(m,k,j,i),
                                     byP(m,k,j,i) + byT(m,k,j,i),
                                     bzP(m,k,j,i) + bzT(m,k,j,i), x1v, x2v);
      Real cross = 0.5*(pS.bsq - pP.bsq - pT.bsq);
      Real dV = pP.sdetg*dx1*dx2*dx3;
      ra += pP.bphi*pP.bphi*dV;
      rb += 2.0*pP.bphi*pT.bphi*dV;
      rc += pT.bphi*pT.bphi*dV;
      rd += (pP.bsq - pP.bphi*pP.bphi)*dV;
      re += 2.0*(cross - pP.bphi*pT.bphi)*dV;
      rf += (pT.bsq - pT.bphi*pT.bphi)*dV;
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
    // Range of E_tor/E_pol reachable with lambda_T/lambda_P >= 0, for the report.
    auto ratio_at = [&](Real x) {
      return (sa + sb*x + sc*x*x)/std::max(sd + se*x + sf*x*x, 1.0e-300);
    };
    Real tp_min = std::min(ratio_at(0.0), sc/std::max(sf, 1.0e-300));
    Real tp_max = std::max(ratio_at(0.0), sc/std::max(sf, 1.0e-300));
    for (int s = 0; s <= 4000; ++s) {
      Real r = ratio_at(std::pow(10.0, -6.0 + 12.0*s/4000.0));
      tp_min = std::min(tp_min, r);
      tp_max = std::max(tp_max, r);
    }
    if (global_variable::my_rank == 0) {
      std::cout << "GRASS magnetic web: E_tor/E_pol of the P pass alone " << sa/sd
                << ", of the T pass alone " << sc/sf << "; reachable range for "
                << "lambda_T/lambda_P >= 0: [" << tp_min << ", " << tp_max << "]; "
                << "E(T pass)/E(P pass) = " << (sc + sf)/std::max(sa + sd, 1.0e-300)
                << " (Chandrasekhar toroidal/poloidal, ~mu_star^2)" << std::endl;
    }

    Real qA = sc - web_tor_pol*sf, qB = sb - web_tor_pol*se, qC = sa - web_tor_pol*sd;
    Real lam_T = 0.0;
    bool have_root = false;
    if (web_test_fixed_lambda) {
      lam_T = 1.0;
      have_root = true;
    } else if (tor_pol_def == "chandrasekhar") {
      // Chandrasekhar (spherical) toroidal/poloidal: energy of the lambda_T*T generator
      // field over that of the P generator field, cross term excluded --
      // lambda_T^2 (c+f)/(a+d) = web_tor_pol (spec §2.2's mu_star^2 estimate; V4, C-025)
      lam_T = std::sqrt(web_tor_pol*(sa + sd)/std::max(sc + sf, 1.0e-300));
      have_root = (lam_T > 0.0);
    } else if (std::abs(qA) < 1.0e-12*std::max({std::abs(qB), std::abs(qC), 1.0})) {
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
                << "normalization quadratic (A=" << qA << " B=" << qB << " C=" << qC << "): "
                << "web_tor_pol=" << web_tor_pol << " lies outside the reachable range ["
                << tp_min << ", " << tp_max << "]." << std::endl;
      exit(EXIT_FAILURE);
    }
    if (global_variable::my_rank == 0) {
      std::cout << "GRASS magnetic web: lambda_P=1, lambda_T=" << lam_T
                << ", achieved cylindrical E_tor/E_pol=" << ratio_at(lam_T)
                << ", Chandrasekhar E_T/E_P=" << lam_T*lam_T*(sc + sf)/std::max(sa + sd, 1.0e-300)
                << " (target " << web_tor_pol << " " << tor_pol_def << ", mu_star="
                << mu_star << ", envelope " << env_name << ")" << std::endl;
    }

    // ---- Combine aP + lam_T*aT (host), normalize, add into final --------------------
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

    BmaxRms(wbx, wby, wbz, &web_bmax_raw, &web_brms_raw);
    Real web_norm_raw = (web_norm == "rms") ? web_brms_raw : web_bmax_raw;
    web_scale = (web_norm_raw > 0.0) ? (web_target/web_norm_raw) : 0.0;
    hel_web = SQR(web_scale)*Helicity(d_wa1, d_wa2, d_wa3, wbx, wby, wbz);
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
      std::cout << "GRASS magnetic web: normalized by " << web_norm << " -> rms|B| (h>=0.5) = "
                << web_scale*web_brms_raw/gauss_to_code << " G, max|B| = "
                << web_scale*web_bmax_raw/gauss_to_code << " G; helicity H_web = " << hel_web
                << " (code units; sign " << ((hel_web >= 0.0) ? "+" : "-") << ")"
                << std::endl;
    }
  }

  // ==== Dipole: current-loop A1/A2 (ported from lorene_bns.cpp), own normalization ===
  AzHost dip_az;
  Real dip_scale = 0.0;
  if (dipole_enable) {
    // I_0 > 0 gives a dipole moment along +z (lorene_bns.cpp convention); the overall
    // scale is fixed by the normalization below, which keeps that sign.
    Real I_0 = 4.0*dipole_r0*dipole_target/(23.0*M_PI);
    std::function<Real(Real, Real, Real)> fd1 = [&](Real x, Real y, Real z) -> Real {
      Real a = grass::DipoleA1(x, y, z, I_0, dipole_r0);
      if (dipole_confine) {
        Real rho = data.InterpolateRho(x, y, z, eos_table);
        a *= grass::WebEnvelope(rho, rho_lo, rho_hi, env_type);
      }
      return a;
    };
    std::function<Real(Real, Real, Real)> fd2 = [&](Real x, Real y, Real z) -> Real {
      Real a = grass::DipoleA2(x, y, z, I_0, dipole_r0);
      if (dipole_confine) {
        Real rho = data.InterpolateRho(x, y, z, eos_table);
        a *= grass::WebEnvelope(rho, rho_lo, rho_hi, env_type);
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

    Real dip_bmax_raw = 0.0, dip_brms_raw = 0.0, dip_bvm_raw = 0.0, dip_bvf_raw = 0.0;
    BmaxRms(dbx, dby, dbz, &dip_bmax_raw, &dip_brms_raw);
    AzimuthalSums(dbx, dby, dbz, &dip_az);
    if (need_h) { BvarpiMeanRms(dbx, dby, dbz, dip_az, &dip_bvm_raw, &dip_bvf_raw); }
    Real dip_norm_raw = (dipole_norm == "bvarpi_rms") ? dip_bvm_raw : dip_bmax_raw;
    dip_scale = (dip_norm_raw > 0.0) ? (dipole_target/dip_norm_raw) : 0.0;
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
      std::cout << "GRASS dipole: normalized by " << dipole_norm << " (confine="
                << dipole_confine << ", r0=" << dipole_r0 << " M) -> rms Bbar_varpi "
                << "(h>=0.5) = " << dip_scale*dip_bvm_raw/gauss_to_code
                << " G, rms of its non-axisymmetric part = "
                << dip_scale*dip_bvf_raw/gauss_to_code << " G, rms|B| (h>=0.5) = "
                << dip_scale*dip_brms_raw/gauss_to_code << " G, max|B| = "
                << dip_scale*dip_bmax_raw/gauss_to_code << " G; moment along "
                << ((I_0*dip_scale > 0.0) ? "+z" : "-z") << std::endl;
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

  // ==== Startup report on the total field ============================================
  // (E_mag/|W|, the h >= 0.5 mass and T need the conserved variables and are printed
  // by GrassMagneticReport() after PrimToConInit.)
  DvceArray4D<Real> tbx, tby, tbz;
  Kokkos::realloc(tbx, nmb, n3, n2, n1); Kokkos::realloc(tby, nmb, n3, n2, n1);
  Kokkos::realloc(tbz, nmb, n3, n2, n1);
  CurlToBcc(d_fa1, d_fa2, d_fa3, tbx, tby, tbz);
  Real tot_bmax = 0.0, tot_brms = 0.0, tot_bvm = 0.0, tot_bvf = 0.0;
  BmaxRms(tbx, tby, tbz, &tot_bmax, &tot_brms);
  AzHost tot_az;
  AzimuthalSums(tbx, tby, tbz, &tot_az);
  if (need_h) { BvarpiMeanRms(tbx, tby, tbz, tot_az, &tot_bvm, &tot_bvf); }
  Real hel_tot = Helicity(d_fa1, d_fa2, d_fa3, tbx, tby, tbz);

  {
    // E_mag, E_tor, E_pol (physical, proper volume); max|div B|*dx/|B|; confinement
    // (max|B| on cells whose own rho lies outside the window: |u| >= 1 is the one-cell
    // skin where edge-staggered h is still non-zero, |u| >= 1.6 must be exactly zero);
    // max comoving b^2/(2P) where h > 0 and where h >= 0.5.
    Real e_mag = 0.0, e_tor = 0.0, e_pol = 0.0;
    Real divb_max = 0.0, bmax_skin = 0.0, bmax_out = 0.0, pm_max = 0.0, pm_max05 = 0.0;
    Kokkos::parallel_reduce("GrassWebReport", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &em, Real &et, Real &ep, Real &dm, Real &bs,
                  Real &bo, Real &pm, Real &pm5) {
      int m = idx/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks; j += js;
      Real dx1 = size.d_view(m).dx1, dx2 = size.d_view(m).dx2, dx3 = size.d_view(m).dx3;
      Real x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      Real x2v = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      Real g0 = admd.g_dd(m,0,0,k,j,i), g1 = admd.g_dd(m,0,1,k,j,i);
      Real g2 = admd.g_dd(m,0,2,k,j,i), g3 = admd.g_dd(m,1,1,k,j,i);
      Real g4 = admd.g_dd(m,1,2,k,j,i), g5 = admd.g_dd(m,2,2,k,j,i);
      GrassPhysB pb = GrassPhysicalB(g0, g1, g2, g3, g4, g5, bcc0(m,IBX,k,j,i),
                                     bcc0(m,IBY,k,j,i), bcc0(m,IBZ,k,j,i), x1v, x2v);
      Real dV = pb.sdetg*dx1*dx2*dx3;
      em += 0.5*pb.bsq*dV;
      et += 0.5*pb.bphi*pb.bphi*dV;
      ep += 0.5*(pb.bsq - pb.bphi*pb.bphi)*dV;
      Real divb = (b0.x1f(m,k,j,i+1)-b0.x1f(m,k,j,i))/dx1 +
                  (b0.x2f(m,k,j+1,i)-b0.x2f(m,k,j,i))/dx2 +
                  (b0.x3f(m,k+1,j,i)-b0.x3f(m,k,j,i))/dx3;
      Real btmag = sqrt(SQR(bcc0(m,IBX,k,j,i)) + SQR(bcc0(m,IBY,k,j,i)) +
                        SQR(bcc0(m,IBZ,k,j,i)));
      dm = fmax(dm, (btmag > 1.0e-300) ? fabs(divb)*cbrt(dx1*dx2*dx3)/btmag : 0.0);
      Real rho = w0d(m,IDN,k,j,i);
      Real u = grass::WebWindowU(rho, rho_lo, rho_hi);
      if (fabs(u) >= 1.0) { bs = fmax(bs, sqrt(pb.bsq)); }
      if (fabs(u) >= 1.6) { bo = fmax(bo, sqrt(pb.bsq)); }
      Real h = grass::WebEnvelope(rho, rho_lo, rho_hi, env_type);
      if (h > 0.0 && w0d(m,IPR,k,j,i) > 0.0) {
        // comoving b^2 = (|B|^2 + (B_i u~^i)^2)/W^2, u~^i = primitive velocity
        Real ux = w0d(m,IVX,k,j,i), uy = w0d(m,IVY,k,j,i), uz = w0d(m,IVZ,k,j,i);
        Real usq = g0*ux*ux + g3*uy*uy + g5*uz*uz + 2.0*(g1*ux*uy + g2*ux*uz + g4*uy*uz);
        Real bx = bcc0(m,IBX,k,j,i)/pb.sdetg, by = bcc0(m,IBY,k,j,i)/pb.sdetg;
        Real bz = bcc0(m,IBZ,k,j,i)/pb.sdetg;
        Real bu = (g0*bx + g1*by + g2*bz)*ux + (g1*bx + g3*by + g4*bz)*uy +
                  (g2*bx + g4*by + g5*bz)*uz;
        Real ratio = 0.5*(pb.bsq + bu*bu)/(1.0 + usq)/w0d(m,IPR,k,j,i);
        pm = fmax(pm, ratio);
        if (h >= 0.5) { pm5 = fmax(pm5, ratio); }
      }
    }, Kokkos::Sum<Real>(e_mag), Kokkos::Sum<Real>(e_tor), Kokkos::Sum<Real>(e_pol),
       Kokkos::Max<Real>(divb_max), Kokkos::Max<Real>(bmax_skin), Kokkos::Max<Real>(bmax_out),
       Kokkos::Max<Real>(pm_max), Kokkos::Max<Real>(pm_max05));
#if MPI_PARALLEL_ENABLED
    {
      Real l3[3] = {e_mag, e_tor, e_pol};
      MPI_Allreduce(MPI_IN_PLACE, l3, 3, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
      e_mag = l3[0]; e_tor = l3[1]; e_pol = l3[2];
      Real l5[5] = {divb_max, bmax_skin, bmax_out, pm_max, pm_max05};
      MPI_Allreduce(MPI_IN_PLACE, l5, 5, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
      divb_max = l5[0]; bmax_skin = l5[1]; bmax_out = l5[2]; pm_max = l5[3];
      pm_max05 = l5[4];
    }
#endif
    if (global_variable::my_rank == 0) {
      const Real g2c = gauss_to_code;
      std::cout << "GRASS magnetic field (total): E_mag=" << e_mag << " (E_tor/E_pol="
                << e_tor/std::max(e_pol, 1.0e-300) << "), max|B| = " << tot_bmax/g2c
                << " G, rms|B| (h>=0.5) = " << tot_brms/g2c << " G, rms Bbar_varpi "
                << "(h>=0.5) = " << tot_bvm/g2c << " G, rms of the non-axisymmetric "
                << "B_varpi = " << tot_bvf/g2c << " G, helicity H = " << hel_tot
                << std::endl;
      std::cout << "GRASS magnetic field: max|div(B)|*dx/|B| = " << divb_max
                << "; confinement max|B| on cells outside the window: one-cell skin "
                << "(|u|>=1) " << bmax_skin/std::max(tot_bmax, 1.0e-300)
                << " of max|B|, beyond it (|u|>=1.6) " << bmax_out << " (code units)"
                << std::endl;
      std::cout << "GRASS magnetic field: max (b^2/2)/P = (B^2/8pi)/P where h>0: "
                << pm_max << ", where h>=0.5: " << pm_max05 << std::endl;
      // t=0 (varpi, z) map of the azimuthal means (design §4: check that the confined
      // dipole's Bbar_varpi covers the window)
      std::string fname = pin->GetString("job", "basename") + ".web_map_t0.txt";
      FILE *fp = std::fopen(fname.c_str(), "w");
      if (fp != nullptr) {
        std::fprintf(fp, "# t=0 azimuthal averages on (varpi, z) bins of %.6g M "
                     "(proper-volume weighted)\n# varpi[M] z[M] h rho0[g/cm^3] "
                     "Bvarpi[G] Bphi[G] Bz[G] Bvarpi_dipole[G]\n", az_db);
        for (int iw = 0; iw < az_nw; ++iw) {
          for (int iz = 0; iz < az_nz; ++iz) {
            Real dv = tot_az(iw,iz,0);
            if (dv <= 0.0) { continue; }
            Real bvd = (dipole_enable && dip_az(iw,iz,0) > 0.0)
                       ? dip_scale*dip_az(iw,iz,1)/dip_az(iw,iz,0) : 0.0;
            std::fprintf(fp, "%.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n",
                         (iw + 0.5)*az_db, az_zmin + (iz + 0.5)*az_db, tot_az(iw,iz,4)/dv,
                         tot_az(iw,iz,5)/dv*rho_to_cgs, tot_az(iw,iz,1)/dv/g2c,
                         tot_az(iw,iz,2)/dv/g2c, tot_az(iw,iz,3)/dv/g2c, bvd/g2c);
          }
        }
        std::fclose(fp);
        std::cout << "GRASS magnetic field: wrote " << fname << std::endl;
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void GrassMagneticReport
//! \brief Startup report items that need the conserved variables, so they run after
//  PrimToConInit: E_mag/|W|, with |W| = T/(T/|W|)_GRASS (<problem> grass_t_over_w, from
//  the GRASS solve; T = (1/2) int Omega dJ evaluated on the AthenaK grid), and the proper
//  volume and rest mass of the region with h(rho) >= 0.5.

void GrassMagneticReport(ParameterInput *pin, Mesh *pm) {
  bool web_enable = pin->GetOrAddInteger("problem", "web_enable", 0) != 0;
  bool dipole_enable = pin->GetOrAddInteger("problem", "dipole_enable", 0) != 0;
  if (!web_enable && !dipole_enable) { return; }
  MeshBlockPack *pmbp = pm->pmb_pack;
  Real rho_lo = pin->GetOrAddReal("problem", "web_rho_lo", 0.0);
  Real rho_hi = pin->GetOrAddReal("problem", "web_rho_hi", 0.0);
  Real t_over_w = pin->GetOrAddReal("problem", "grass_t_over_w", 0.0);
  const int env_type = (pin->GetOrAddString("problem", "web_envelope", "bump") == "poly3")
                       ? 1 : 0;

  auto &indcs = pm->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  const int nmkji = pmbp->nmb_thispack*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1, nji = nx2*nx1;
  auto &w0 = pmbp->pmhd->w0;
  auto &u0 = pmbp->pmhd->u0;
  auto &bcc0 = pmbp->pmhd->bcc0;
  auto &adm = pmbp->padm->adm;
  auto &size = pmbp->pmb->mb_size;
  // the gauge lives in the z4c state under <z4c>, in u_adm under <cfc>
  auto alpha = (pmbp->pz4c != nullptr) ? pmbp->pz4c->z4c.alpha : adm.alpha;
  auto beta_u = (pmbp->pz4c != nullptr) ? pmbp->pz4c->z4c.beta_u : adm.beta_u;

  Real e_mag = 0.0, t_kin = 0.0, vol05 = 0.0, mass05 = 0.0, mass_tot = 0.0;
  Kokkos::parallel_reduce("GrassMagReport", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &em, Real &tk, Real &v5, Real &m5, Real &mt) {
    int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    Real x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real x2v = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real g0 = adm.g_dd(m,0,0,k,j,i), g1 = adm.g_dd(m,0,1,k,j,i);
    Real g2 = adm.g_dd(m,0,2,k,j,i), g3 = adm.g_dd(m,1,1,k,j,i);
    Real g4 = adm.g_dd(m,1,2,k,j,i), g5 = adm.g_dd(m,2,2,k,j,i);
    GrassPhysB pb = GrassPhysicalB(g0, g1, g2, g3, g4, g5, bcc0(m,IBX,k,j,i),
                                   bcc0(m,IBY,k,j,i), bcc0(m,IBZ,k,j,i), x1v, x2v);
    em += 0.5*pb.bsq*pb.sdetg*vol;
    // T = (1/2) int Omega dJ; dJ = (x S_y - y S_x) d^3x with S_i densitized (u0)
    Real ux = w0(m,IVX,k,j,i), uy = w0(m,IVY,k,j,i), uz = w0(m,IVZ,k,j,i);
    Real wlor = sqrt(1.0 + g0*ux*ux + g3*uy*uy + g5*uz*uz +
                     2.0*(g1*ux*uy + g2*ux*uz + g4*uy*uz));
    Real a = alpha(m,k,j,i);
    Real vx = a*ux/wlor - beta_u(m,0,k,j,i), vy = a*uy/wlor - beta_u(m,1,k,j,i);
    Real w2 = x1v*x1v + x2v*x2v;
    if (w2 > 0.0) {
      Real omega = (x1v*vy - x2v*vx)/w2;
      tk += 0.5*omega*(x1v*u0(m,IM2,k,j,i) - x2v*u0(m,IM1,k,j,i))*vol;
    }
    mt += u0(m,IDN,k,j,i)*vol;
    if (grass::WebEnvelope(w0(m,IDN,k,j,i), rho_lo, rho_hi, env_type) >= 0.5) {
      v5 += pb.sdetg*vol;
      m5 += u0(m,IDN,k,j,i)*vol;
    }
  }, Kokkos::Sum<Real>(e_mag), Kokkos::Sum<Real>(t_kin), Kokkos::Sum<Real>(vol05),
     Kokkos::Sum<Real>(mass05), Kokkos::Sum<Real>(mass_tot));
#if MPI_PARALLEL_ENABLED
  Real l5[5] = {e_mag, t_kin, vol05, mass05, mass_tot};
  MPI_Allreduce(MPI_IN_PLACE, l5, 5, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  e_mag = l5[0]; t_kin = l5[1]; vol05 = l5[2]; mass05 = l5[3]; mass_tot = l5[4];
#endif
  if (global_variable::my_rank == 0) {
    grass::GrassUnits units;
    Real km = units.LengthCgsToCode(1.0e5);
    std::cout << "GRASS magnetic field: region h>=0.5: proper volume " << vol05 << " M^3 = "
              << vol05/(km*km*km) << " km^3, rest mass " << mass05 << " Msun = "
              << mass05/std::max(mass_tot, 1.0e-300) << " of the total " << mass_tot
              << std::endl;
    if (t_over_w > 0.0) {
      Real w_abs = t_kin/t_over_w;
      Real ratio = e_mag/w_abs;
      std::cout << "GRASS magnetic field: E_mag=" << e_mag << ", T=" << t_kin
                << ", |W|=T/(T/|W|)=" << w_abs << " (grass_t_over_w=" << t_over_w
                << "), E_mag/|W|=" << ratio << ", E_mag/T=" << e_mag/t_kin << std::endl;
      if (ratio > 1.0e-3) {
        std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "E_mag/|W|=" << ratio << " exceeds 1e-3." << std::endl;
      }
    } else {
      std::cout << "GRASS magnetic field: E_mag=" << e_mag << ", T=" << t_kin
                << "; set <problem> grass_t_over_w (GRASS T/|W|) to report E_mag/|W|"
                << std::endl;
    }
  }
}
//----------------------------------------------------------------------------------------
//! \fn void GrassWebDiagnostics
//! \brief Magnetic-web run diagnostics (C-026; R-033 (c)), called from GrassHistory on
//  every rank, so the history cadence sets theirs (0.05 ms for the web runs). Rank 0
//  appends one line to <basename>.webdiag.txt, and every web_map_dt writes the (varpi, z)
//  azimuthal-average map <basename>.azmap.NNNNN.txt. User history output is capped at
//  NHISTORY_VARIABLES = 20 columns, hence the separate file.
//  All fields are physical (B^i = bcc0^i/sqrt(gamma), components along flat unit
//  vectors), volumes proper (sqrt(gamma) d^3x). Azimuthal means are proper-volume
//  weighted over (varpi, z) bins of width dx_finest out to grass_diag.rmax.
//  Columns:
//   t, E_mag, E_m0 (energy of the azimuthal-mean field), E_mnz (rest, inside the bins),
//   E_out (rho0 outside [rho_lo, rho_hi] now), rms Bbar_varpi, rms bbar_phi,
//   alpha = rms bbar_phi/rms Bbar_varpi, rms Bbar_R (spherical radial), alpha_R (rms
//   over the window bins, rho0-weighted), W1/W2/W3 (SACRA 2D: Omega averaged over
//   0.5 km in varpi, Omega_max/Omega_min over window bins with varpi >= 1.5 km),
//   W3_eq (equatorial bins), mono (rho0 > 1e12 fraction with dlnOmega/dlnvarpi < -1e-3),
//   torque through cylinders varpi = 2, 4, 6 km, Maxwell (-alpha sqrt(gamma) b^varpi
//   b_phi) and Reynolds (sqrt(gamma) S_phi alpha u^varpi/W = alpha sqrt(gamma) rho h
//   u^varpi u_phi, from the conserved S_phi) parts, each int F^varpi varpi dphi dz from
//   a one-cell shell, C1, C2 (rho0-weighted, varpi < 8 M), rho_c, Mb.

namespace {
struct GrassDiagConfig {
  bool on = false;
  Real rho_lo = 0.0, rho_hi = 0.0;
  Real db = 0.0, rmax = 0.0;
  Real map_dt = 0.0, next_map = -1.0;
  int map_count = 0;
  std::string basename;
  bool header_done = false;
};
GrassDiagConfig grass_diag;
}  // namespace

void GrassWebDiagnostics(Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmkji = pmbp->nmb_thispack*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1, nji = nx2*nx1;
  auto &w0 = pmbp->pmhd->w0;
  auto &u0 = pmbp->pmhd->u0;
  auto &bcc0 = pmbp->pmhd->bcc0;
  auto &adm = pmbp->padm->adm;
  auto &size = pmbp->pmb->mb_size;
  const Real rho_lo = grass_diag.rho_lo, rho_hi = grass_diag.rho_hi;
  const Real db = grass_diag.db;
  const int nw = static_cast<int>(std::ceil(grass_diag.rmax/db));
  const int nz = 2*nw;
  const Real zmin = -nw*db;
  grass::GrassUnits units;
  const Real km = units.LengthCgsToCode(1.0e5);
  const Real rho_cgs = 1.0/units.MassDensityCgsToCode(1.0);
  const Real wc0 = 2.0*km, wc1 = 4.0*km, wc2 = 6.0*km;   // cylinder radii
  const Real dxf = db;                                    // shell thickness

  // per bin: 0 dV, 1 rho dV, 2 Bvarpi dV, 3 Bphi dV, 4 Bz dV, 5 BR dV, 6 Omega dV, 7 E dV
  constexpr int kN = 8;
  DvceArray3D<Real> bins("grass_diag_bins", nw, nz, kN);
  par_for("grass_diag_bins", DevExeSpace(), 0, pmbp->nmb_thispack-1, ks, ks+nx3-1,
          js, js+nx2-1, is, is+nx1-1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real y = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real z = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    Real w = sqrt(x*x + y*y);
    int iw = static_cast<int>(w/db);
    int iz = static_cast<int>(Kokkos::floor((z - zmin)/db));
    if (iw >= nw || iz < 0 || iz >= nz) { return; }
    Real g0 = adm.g_dd(m,0,0,k,j,i), g1 = adm.g_dd(m,0,1,k,j,i), g2 = adm.g_dd(m,0,2,k,j,i);
    Real g3 = adm.g_dd(m,1,1,k,j,i), g4 = adm.g_dd(m,1,2,k,j,i), g5 = adm.g_dd(m,2,2,k,j,i);
    GrassPhysB pb = GrassPhysicalB(g0, g1, g2, g3, g4, g5, bcc0(m,IBX,k,j,i),
                                   bcc0(m,IBY,k,j,i), bcc0(m,IBZ,k,j,i), x, y);
    Real dV = pb.sdetg*size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    Real r = sqrt(w*w + z*z);
    Real bR = (r > 0.0) ? (pb.bvarpi*w + pb.bz*z)/r : 0.0;
    Real ux = w0(m,IVX,k,j,i), uy = w0(m,IVY,k,j,i), uz = w0(m,IVZ,k,j,i);
    Real wl = sqrt(1.0 + g0*ux*ux + g3*uy*uy + g5*uz*uz + 2.0*(g1*ux*uy + g2*ux*uz + g4*uy*uz));
    Real a = adm.alpha(m,k,j,i);
    Real vx = a*ux/wl - adm.beta_u(m,0,k,j,i), vy = a*uy/wl - adm.beta_u(m,1,k,j,i);
    Real om = (w > 0.0) ? (x*vy - y*vx)/(w*w) : 0.0;
    Real rho = w0(m,IDN,k,j,i);
    Kokkos::atomic_add(&bins(iw,iz,0), dV);
    Kokkos::atomic_add(&bins(iw,iz,1), rho*dV);
    Kokkos::atomic_add(&bins(iw,iz,2), pb.bvarpi*dV);
    Kokkos::atomic_add(&bins(iw,iz,3), pb.bphi*dV);
    Kokkos::atomic_add(&bins(iw,iz,4), pb.bz*dV);
    Kokkos::atomic_add(&bins(iw,iz,5), bR*dV);
    Kokkos::atomic_add(&bins(iw,iz,6), om*dV);
    Kokkos::atomic_add(&bins(iw,iz,7), 0.5*pb.bsq*dV);
  });

  // cell sums: 0 E_mag, 1 E_out, 2-4 Maxwell torque, 5-7 Reynolds torque,
  // 8 rho cos phi dV, 9 rho sin phi dV, 10 rho cos 2phi dV, 11 rho sin 2phi dV, 12 rho dV
  // (varpi < 8 M), 13 rho_c sum, 14 rho_c count, 15 Mb
  constexpr int kS = 16;
  DvceArray1D<Real> sums("grass_diag_sums", kS);
  par_for("grass_diag_sums", DevExeSpace(), 0, pmbp->nmb_thispack-1, ks, ks+nx3-1,
          js, js+nx2-1, is, is+nx1-1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real dx1 = size.d_view(m).dx1, dx2 = size.d_view(m).dx2, dx3 = size.d_view(m).dx3;
    Real vol = dx1*dx2*dx3;
    Real x = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real y = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real z = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    Real w = sqrt(x*x + y*y);
    Real g0 = adm.g_dd(m,0,0,k,j,i), g1 = adm.g_dd(m,0,1,k,j,i), g2 = adm.g_dd(m,0,2,k,j,i);
    Real g3 = adm.g_dd(m,1,1,k,j,i), g4 = adm.g_dd(m,1,2,k,j,i), g5 = adm.g_dd(m,2,2,k,j,i);
    GrassPhysB pb = GrassPhysicalB(g0, g1, g2, g3, g4, g5, bcc0(m,IBX,k,j,i),
                                   bcc0(m,IBY,k,j,i), bcc0(m,IBZ,k,j,i), x, y);
    Real rho = w0(m,IDN,k,j,i);
    Kokkos::atomic_add(&sums(0), 0.5*pb.bsq*pb.sdetg*vol);
    if (rho < rho_lo || rho > rho_hi) {
      Kokkos::atomic_add(&sums(1), 0.5*pb.bsq*pb.sdetg*vol);
    }
    Kokkos::atomic_add(&sums(15), u0(m,IDN,k,j,i)*vol);
    if (fabs(x) < dx1 && fabs(y) < dx2 && fabs(z) < dx3) {
      Kokkos::atomic_add(&sums(13), rho);
      Kokkos::atomic_add(&sums(14), 1.0);
    }
    if (w < 8.0 && w > 0.0) {
      Real c1 = x/w, s1 = y/w;
      Real dm = rho*pb.sdetg*vol;
      Kokkos::atomic_add(&sums(8), dm*c1);
      Kokkos::atomic_add(&sums(9), dm*s1);
      Kokkos::atomic_add(&sums(10), dm*(c1*c1 - s1*s1));
      Kokkos::atomic_add(&sums(11), dm*2.0*s1*c1);
      Kokkos::atomic_add(&sums(12), dm);
    }
    // angular-momentum flux through the cylinders: F^varpi = alpha sqrt(gamma) T^varpi_phi
    Real wcs[3] = {wc0, wc1, wc2};
    for (int c = 0; c < 3; ++c) {
      if (fabs(w - wcs[c]) >= 0.5*dxf) { continue; }
      Real ux = w0(m,IVX,k,j,i), uy = w0(m,IVY,k,j,i), uz = w0(m,IVZ,k,j,i);
      Real uxd = g0*ux + g1*uy + g2*uz, uyd = g1*ux + g3*uy + g4*uz, uzd = g2*ux + g4*uy + g5*uz;
      Real wl = sqrt(1.0 + uxd*ux + uyd*uy + uzd*uz);
      Real a = adm.alpha(m,k,j,i);
      Real bx = adm.beta_u(m,0,k,j,i), by = adm.beta_u(m,1,k,j,i);
      // u^i = u~^i - W beta^i/alpha; u_i = u~_i (spatial)
      Real upx = ux - wl*bx/a, upy = uy - wl*by/a;
      Real u_var = (x*upx + y*upy)/w;            // u^varpi
      // b^mu from B^i: b^0 = W (B.v)/alpha, b^i = B^i/W + b^0 u^i/u^t... in the form
      // b^i = (B^i + alpha b^0 u^i)/W, b_i = (B_i + alpha b^0 u_i)/W
      Real sd = pb.sdetg;
      Real Bx = bcc0(m,IBX,k,j,i)/sd, By = bcc0(m,IBY,k,j,i)/sd, Bz = bcc0(m,IBZ,k,j,i)/sd;
      Real Bxd = g0*Bx + g1*By + g2*Bz, Byd = g1*Bx + g3*By + g4*Bz, Bzd = g2*Bx + g4*By + g5*Bz;
      Real Bu = Bxd*ux + Byd*uy + Bzd*uz;        // B_i u~^i = W B.v
      Real ab0 = Bu;                             // alpha b^0
      Real bvx = (Bx + ab0*upx)/wl, bvy = (By + ab0*upy)/wl;
      Real b_var = (x*bvx + y*bvy)/w;            // b^varpi
      Real b_phi = (x*(Byd + ab0*uyd) - y*(Bxd + ab0*uxd))/wl;   // b_phi
      Real fac = vol/dxf;                        // int F^varpi varpi dphi dz from the shell
      Kokkos::atomic_add(&sums(2 + c), -fac*a*sd*b_var*b_phi);
      // Reynolds: alpha sqrt(gamma) rho h u^varpi u_phi = (sqrt(gamma) S_phi)(alpha v^varpi
      // - beta^varpi), S_phi from the conserved momentum (its small Poynting part included)
      Real s_phi = x*u0(m,IM2,k,j,i) - y*u0(m,IM1,k,j,i);
      Kokkos::atomic_add(&sums(5 + c), fac*s_phi*a*u_var/wl);
    }
  });

  auto bins_h = Kokkos::create_mirror_view(bins);
  Kokkos::deep_copy(bins_h, bins);
  auto sums_h = Kokkos::create_mirror_view(sums);
  Kokkos::deep_copy(sums_h, sums);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, bins_h.data(), static_cast<int>(bins_h.size()),
                MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, sums_h.data(), kS, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (global_variable::my_rank != 0) { return; }

  // ---- host post-processing of the bins (rank 0) ----
  auto mean = [&](int iw, int iz, int q) {
    return (bins_h(iw,iz,0) > 0.0) ? bins_h(iw,iz,q)/bins_h(iw,iz,0) : 0.0;
  };
  Real e_m0 = 0.0, e_bins = 0.0;
  Real s_bv = 0.0, s_bp = 0.0, s_br = 0.0, s_rho = 0.0;
  for (int iw = 0; iw < nw; ++iw) {
    for (int iz = 0; iz < nz; ++iz) {
      Real dv = bins_h(iw,iz,0);
      if (dv <= 0.0) { continue; }
      Real bv = mean(iw,iz,2), bp = mean(iw,iz,3), bz = mean(iw,iz,4), br = mean(iw,iz,5);
      e_m0 += 0.5*(bv*bv + bp*bp + bz*bz)*dv;
      e_bins += bins_h(iw,iz,7);
      Real rho = mean(iw,iz,1);
      if (rho >= rho_lo && rho <= rho_hi) {
        s_bv += rho*bv*bv*dv; s_bp += rho*bp*bp*dv; s_br += rho*br*br*dv; s_rho += rho*dv;
      }
    }
  }
  Real rms_bv = (s_rho > 0.0) ? std::sqrt(s_bv/s_rho) : 0.0;
  Real rms_bp = (s_rho > 0.0) ? std::sqrt(s_bp/s_rho) : 0.0;
  Real rms_br = (s_rho > 0.0) ? std::sqrt(s_br/s_rho) : 0.0;

  // Omega windows, SACRA 2D definition: |Omega| (sign-agnostic), z folded, averaged over
  // 0.5 km along varpi (running mean over the populated bins), varpi >= 1.5 km
  const int nzh = nw;                         // folded |z| bins
  std::vector<Real> om(nw*nzh, 0.0), rh(nw*nzh, 0.0), vv(nw*nzh, 0.0);
  for (int iw = 0; iw < nw; ++iw) {
    for (int iz = 0; iz < nz; ++iz) {
      int jz = (iz >= nw) ? (iz - nw) : (nw - 1 - iz);
      int q = iw*nzh + jz;
      om[q] += bins_h(iw,iz,6); rh[q] += bins_h(iw,iz,1); vv[q] += bins_h(iw,iz,0);
    }
  }
  const int nsm = std::max(1, static_cast<int>(std::lround(0.5*km/db)));
  std::vector<Real> oms(nw*nzh, 0.0);
  Real rho_max = 0.0;
  for (int q = 0; q < nw*nzh; ++q) {
    if (vv[q] > 0.0) { rho_max = std::max(rho_max, rh[q]/vv[q]); }
  }
  for (int jz = 0; jz < nzh; ++jz) {
    for (int iw = 0; iw < nw; ++iw) {
      Real s = 0.0; int c = 0; bool full = true;
      for (int d = -(nsm/2); d < nsm - nsm/2; ++d) {
        int iv = iw + d;
        if (iv < 0 || iv >= nw || vv[iv*nzh + jz] <= 0.0) { full = false; continue; }
        s += std::abs(om[iv*nzh + jz]/vv[iv*nzh + jz]); ++c;
      }
      oms[iw*nzh + jz] = (full && c > 0) ? s/c : -1.0;
    }
  }
  auto window_ratio = [&](Real lo, Real hi, bool eq_only) {
    Real omax = 0.0, omin = 1.0e300;
    for (int iw = 0; iw < nw; ++iw) {
      if ((iw + 0.5)*db < 1.5*km) { continue; }
      for (int jz = 0; jz < (eq_only ? 1 : nzh); ++jz) {
        int q = iw*nzh + jz;
        if (vv[q] <= 0.0 || oms[q] < 0.0) { continue; }
        Real rho = rh[q]/vv[q]*rho_cgs;
        if (rho < lo || rho > hi) { continue; }
        omax = std::max(omax, oms[q]); omin = std::min(omin, oms[q]);
      }
    }
    return (omax > 0.0 && omin < 1.0e300) ? omax/omin : 0.0;
  };
  Real w1 = window_ratio(1.0e13, 5.0e14, false), w2 = window_ratio(1.0e12, 1.0e14, false);
  Real w3 = window_ratio(1.0e14, 0.9*rho_max*rho_cgs, false);
  Real w3eq = window_ratio(1.0e14, 0.9*rho_max*rho_cgs, true);
  Real mono_bad = 0.0, mono_all = 0.0;
  for (int jz = 0; jz < nzh; ++jz) {
    for (int iw = 1; iw < nw - 1; ++iw) {
      int q = iw*nzh + jz, qm = (iw - 1)*nzh + jz, qp = (iw + 1)*nzh + jz;
      Real wv = (iw + 0.5)*db;
      if (wv < 1.5*km || vv[q] <= 0.0 || oms[q] <= 0.0 || oms[qm] <= 0.0 || oms[qp] <= 0.0) {
        continue;
      }
      Real rho = rh[q]/vv[q];
      if (rho*rho_cgs <= 1.0e12) { continue; }
      Real dl = (oms[qp] - oms[qm])/(2.0*db)*wv/oms[q];
      mono_all += rho*wv;
      if (dl < -1.0e-3) { mono_bad += rho*wv; }
    }
  }

  Real e_mag = sums_h(0);
  Real c1 = (sums_h(12) > 0.0) ? std::hypot(sums_h(8), sums_h(9))/sums_h(12) : 0.0;
  Real c2 = (sums_h(12) > 0.0) ? std::hypot(sums_h(10), sums_h(11))/sums_h(12) : 0.0;
  Real rho_c = (sums_h(14) > 0.0) ? sums_h(13)/sums_h(14) : 0.0;
  std::string fname = grass_diag.basename + ".webdiag.txt";
  FILE *fp = std::fopen(fname.c_str(), "a");
  if (fp != nullptr) {
    if (!grass_diag.header_done) {
      std::fprintf(fp, "# GrassWebDiagnostics (dyngr_grass.cpp; C-026). Code units, physical "
                   "field; window rho0 in [%.6e, %.6e] code\n# 1:t 2:E_mag 3:E_m0 4:E_mnz "
                   "5:E_out 6:Bbar_varpi_rms 7:bbar_phi_rms 8:alpha 9:Bbar_R_rms 10:alpha_R "
                   "11:W1_2d 12:W2_2d 13:W3_2d 14:W3_eq 15:mono_2d 16:tqM_2km 17:tqM_4km "
                   "18:tqM_6km 19:tqR_2km 20:tqR_4km 21:tqR_6km 22:C1 23:C2 24:rho_c 25:Mb\n",
                   rho_lo, rho_hi);
      grass_diag.header_done = true;
    }
    std::fprintf(fp, "%.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e "
                 "%.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e "
                 "%.10e %.10e\n", pm->time, e_mag, e_m0, e_bins - e_m0, sums_h(1), rms_bv,
                 rms_bp, (rms_bv > 0.0) ? rms_bp/rms_bv : 0.0, rms_br,
                 (rms_br > 0.0) ? rms_bp/rms_br : 0.0, w1, w2, w3, w3eq,
                 (mono_all > 0.0) ? mono_bad/mono_all : 0.0, sums_h(2), sums_h(3), sums_h(4),
                 sums_h(5), sums_h(6), sums_h(7), c1, c2, rho_c, sums_h(15));
    std::fclose(fp);
  }

  // ---- (varpi, z) map every web_map_dt ----
  if (grass_diag.map_dt > 0.0) {
    if (grass_diag.next_map < 0.0) {
      grass_diag.next_map = std::ceil(pm->time/grass_diag.map_dt - 1.0e-9)*grass_diag.map_dt;
      grass_diag.map_count = static_cast<int>(std::lround(grass_diag.next_map/grass_diag.map_dt));
    }
    if (pm->time >= grass_diag.next_map - 1.0e-9) {
      char buf[64];
      std::snprintf(buf, sizeof(buf), ".azmap.%05d.txt", grass_diag.map_count);
      FILE *fm = std::fopen((grass_diag.basename + buf).c_str(), "w");
      if (fm != nullptr) {
        std::fprintf(fm, "# t = %.10e; azimuthal averages on (varpi, z) bins of %.6g M "
                     "(proper-volume weighted), code units\n# varpi z rho0 Omega Bvarpi "
                     "Bphi Bz BR\n", pm->time, db);
        for (int iw = 0; iw < nw; ++iw) {
          for (int iz = 0; iz < nz; ++iz) {
            if (bins_h(iw,iz,0) <= 0.0 || mean(iw,iz,1)*rho_cgs < 1.0e10) { continue; }
            std::fprintf(fm, "%.5e %.5e %.6e %.6e %.6e %.6e %.6e %.6e\n", (iw + 0.5)*db,
                         zmin + (iz + 0.5)*db, mean(iw,iz,1), mean(iw,iz,6), mean(iw,iz,2),
                         mean(iw,iz,3), mean(iw,iz,4), mean(iw,iz,5));
          }
        }
        std::fclose(fm);
      }
      grass_diag.map_count += 1;
      grass_diag.next_map += grass_diag.map_dt;
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

  // Web-run diagnostics (GrassWebDiagnostics): on by default when a field is installed,
  // web_diag = 1 also turns them on for a B = 0 twin. Set here, before the restart
  // return, so restarted segments keep them.
  {
    bool field = pin->GetOrAddInteger("problem", "web_enable", 0) != 0 ||
                 pin->GetOrAddInteger("problem", "dipole_enable", 0) != 0;
    grass_diag.on = pin->GetOrAddInteger("problem", "web_diag", field ? 1 : 0) != 0;
    grass_diag.rho_lo = pin->GetOrAddReal("problem", "web_rho_lo", 0.0);
    grass_diag.rho_hi = pin->GetOrAddReal("problem", "web_rho_hi", 0.0);
    grass_diag.db = pmy_mesh_->mesh_size.dx1 /
                    std::pow(2.0, pmy_mesh_->max_level - pmy_mesh_->root_level);
    grass_diag.rmax = pin->GetOrAddReal("problem", "web_diag_rmax", 9.0);
    grass_diag.map_dt = pin->GetOrAddReal("problem", "web_map_dt", 0.0);
    grass_diag.basename = pin->GetString("job", "basename");
  }

  // No restart-path special-casing: this pgen only ever sets initial data once --
  // evolution afterward proceeds from checkpointed Z4c/CFC/hydro state like any other
  // dynamical run (same reasoning as dyngr_rns_st.cpp).
  if (restart) { return; }

  // Dispatch pz4c/nscalars to SetupGrass's IsZ4c/UseYe template args once, here.
  const bool read_ye = pin->GetOrAddInteger("mhd", "nscalars", 0) > 0;
  if (pmbp->pz4c != nullptr) {
    if (read_ye) {
      SetupGrass<true, true>(pin, pmy_mesh_);
    } else {
      SetupGrass<true, false>(pin, pmy_mesh_);
    }
  } else {
    if (read_ye) {
      SetupGrass<false, true>(pin, pmy_mesh_);
    } else {
      SetupGrass<false, false>(pin, pmy_mesh_);
    }
  }

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

  GrassMagneticReport(pin, pmy_mesh_);

  return;
}

// History function: rho-max, alpha-min (same diagnostics as dyngr_tov.cpp's TOVHistory
// / dyngr_rns_st.cpp's RnsStHistory, minus the scalar-field entries -- no scalar field
// here) plus, per magnetic_web_id_athenak.md section 7, magnetic diagnostics: E_mag,
// E_tor, E_pol (WHOLE-domain, not restricted to the initial rho_lo shell -- unlike the
// one-time normalization pass in BuildMagneticField, restricting an ONGOING diagnostic
// to the field's initial confinement boundary would become physically meaningless once
// the star evolves and the field moves), Bmax, max|div(B)|*dx/|B|, total angular
// momentum ang-mom (J_z = integral of vol*(x*S_y - y*S_x), reusing xns_rotstar.cpp's
// XNSRotStarHistory formula verbatim -- mode-agnostic since it only reads the
// pre-densitized conserved momentum u0(IM1/IM2), not adm.g_dd/vK_dd directly), and the
// volume-integrated Maxwell stress int B_varpi B_phi dV. Magnetic quantities use the
// physical field B^i = bcc0^i/sqrt(gamma) (bcc0 is densitized; see GrassPhysicalB):
// energies int |B|^2/2 sqrt(gamma) d^3x, Bmax = max sqrt(g_ij B^i B^j), code units.
// Helicity is reported once at t=0 by BuildMagneticField (it needs A, not retained).

void GrassHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 9;
  pdata->label[0] = "rho-max";
  pdata->label[1] = "alpha-min";
  pdata->label[2] = "E_mag";
  pdata->label[3] = "E_tor";
  pdata->label[4] = "E_pol";
  pdata->label[5] = "Bmax";
  pdata->label[6] = "divBmax";
  pdata->label[7] = "ang-mom";
  pdata->label[8] = "Maxwell";

  auto &w0_ = pm->pmb_pack->pmhd->w0;
  auto &u0_ = pm->pmb_pack->pmhd->u0;
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
  Real e_mag = 0.0, e_tor = 0.0, e_pol = 0.0, divb_max = 0.0, ang_mom = 0.0;
  Real maxwell = 0.0;
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
    Real x1v = CellCenterX(i-is, nx1, size_.d_view(m).x1min, size_.d_view(m).x1max);
    Real x2v = CellCenterX(j-js, nx2, size_.d_view(m).x2min, size_.d_view(m).x2max);
    GrassPhysB pb = GrassPhysicalB(adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
                                   adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
                                   adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i),
                                   bcc0_(m,IBX,k,j,i), bcc0_(m,IBY,k,j,i),
                                   bcc0_(m,IBZ,k,j,i), x1v, x2v);
    mb_bmax = fmax(mb_bmax, sqrt(pb.bsq));
  }, Kokkos::Max<Real>(rho_max), Kokkos::Min<Real>(alpha_min), Kokkos::Max<Real>(bmax));

  Kokkos::parallel_reduce("GrassHistMag", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &em, Real &etor, Real &epol, Real &dbm, Real &jz,
                Real &mxw) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    Real dx1 = size_.d_view(m).dx1, dx2 = size_.d_view(m).dx2, dx3 = size_.d_view(m).dx3;
    Real x1v = CellCenterX(i-is, nx1, size_.d_view(m).x1min, size_.d_view(m).x1max);
    Real x2v = CellCenterX(j-js, nx2, size_.d_view(m).x2min, size_.d_view(m).x2max);
    // J_z = integral of vol*(x*S_y - y*S_x); u0(IM1/IM2) is already the densitized
    // ADM momentum density sqrt(gamma)*S_a (see dyn_grmhd.cpp's SetTmunu), so plain
    // vol=dx1*dx2*dx3 is correct here -- no extra sqrt(gamma)/dV factor.
    jz += dx1*dx2*dx3*(x1v*u0_(m,IM2,k,j,i) - x2v*u0_(m,IM1,k,j,i));
    GrassPhysB pb = GrassPhysicalB(adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
                                   adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
                                   adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i),
                                   bcc0_(m,IBX,k,j,i), bcc0_(m,IBY,k,j,i),
                                   bcc0_(m,IBZ,k,j,i), x1v, x2v);
    Real dV = pb.sdetg*dx1*dx2*dx3;
    em += 0.5*pb.bsq*dV;
    etor += 0.5*pb.bphi*pb.bphi*dV;
    epol += 0.5*(pb.bsq - pb.bphi*pb.bphi)*dV;
    mxw += pb.bvarpi*pb.bphi*dV;
    Real divb = (b0_.x1f(m,k,j,i+1)-b0_.x1f(m,k,j,i))/dx1 +
                (b0_.x2f(m,k,j+1,i)-b0_.x2f(m,k,j,i))/dx2 +
                (b0_.x3f(m,k+1,j,i)-b0_.x3f(m,k,j,i))/dx3;
    Real btmag = sqrt(SQR(bcc0_(m,IBX,k,j,i)) + SQR(bcc0_(m,IBY,k,j,i)) +
                      SQR(bcc0_(m,IBZ,k,j,i)));
    Real dx_local = cbrt(dx1*dx2*dx3);
    dbm = fmax(dbm, (btmag > 1.0e-300) ? fabs(divb)*dx_local/btmag : 0.0);
  }, Kokkos::Sum<Real>(e_mag), Kokkos::Sum<Real>(e_tor), Kokkos::Sum<Real>(e_pol),
     Kokkos::Max<Real>(divb_max), Kokkos::Sum<Real>(ang_mom), Kokkos::Sum<Real>(maxwell));

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
  // ang_mom and maxwell are left out of the manual reduction above: it's a genuine per-rank
  // partial volume-integral sum, and the framework's generic MPI_SUM reduction
  // over hdata (HistoryOutput::WriteOutputFile) combines per-rank partial sums
  // into the correct global total with no manual reduction needed here --
  // matches xns_rotstar.cpp's XNSRotStarHistory treatment of the same quantity.

  pdata->hdata[0] = rho_max;
  pdata->hdata[1] = alpha_min;
  pdata->hdata[2] = e_mag;
  pdata->hdata[3] = e_tor;
  pdata->hdata[4] = e_pol;
  pdata->hdata[5] = bmax;
  pdata->hdata[6] = divb_max;
  pdata->hdata[7] = ang_mom;
  pdata->hdata[8] = maxwell;

  if (grass_diag.on) { GrassWebDiagnostics(pm); }
}
