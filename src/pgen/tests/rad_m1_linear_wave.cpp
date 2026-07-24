//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_m1_linear_wave.cpp
//! \brief grey M1 radiation-hydrodynamic linear wave test (arXiv:2302.04283 Section 3.9)
//!
//! Sets up the same 1D, x1-propagating, radiation-modified hydrodynamic sound wave as
//! the DO module's own rad_linear_wave.cpp -- same background state and complex
//! eigenvalue/eigenvector, read verbatim from the <problem> block (see
//! inputs/tests/rad_linwave.athinput's <problem> values, themselves matching the
//! paper's own Appendix A Table 2 "H1" gas-dominated case to ~13 significant figures).
//! Unlike DO, M1 has no angular grid: its own (E, F_d) *are* the fluid-frame radiation
//! moments (Ē, F̄_a) the paper's eigenvector is expressed in, up to a Lorentz boost into
//! the lab (Eulerian-observer) frame M1 actually evolves. That boost -- fluid-frame
//! Eddington stress tensor -> lab-frame (E, F_d) -- is adapted directly from DO's own
//! rad_linear_wave.cpp (its u_wave/rf_wave/lambda_c_f_wave/r_wave sequence, lines
//! ~279-344), simplified because this wave only ever propagates along x1: DO's
//! wave-direction rotation (its cos_a2/cos_a3/sin_a2/sin_a3 machinery, for waves at an
//! angle to the grid) is the identity transform here, so the boosted "r_wave" tensor
//! *is* the lab-frame tensor with no further rotation needed.
//!
//! Only the t=0 initial condition is set here -- unlike DO's pgen, this does not also
//! recompute a reference solution at the final time via a pgen_final_func hook; the
//! companion check script (tst/scripts/radiation_m1/check_rad_m1_photon_linear_wave.py)
//! independently reconstructs the analytic solution at whatever time the saved output
//! actually reports, matching the convention already used by every other stage's
//! comparison script in this project.

#include <cmath>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/eos.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "radiation_m1/radiation_m1.hpp"
#include "radiation_m1/radiation_m1_helpers.hpp"

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::RadiationM1LinearWave(ParameterInput *pin)
//! \brief Sets initial conditions for the grey M1 radiation linear wave test

void ProblemGenerator::RadiationM1LinearWave(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pradm1 == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "The M1 linear wave problem generator requires a <radiation_m1> block"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->pradm1->params.opacity_type != radiationm1::Photons) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "The M1 linear wave problem generator requires opacity_type=photons "
                 "(a real fluid is needed for the radiation-matter-coupled background)"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  auto *ptest_ideal =
      dynamic_cast<dyngr::DynGRMHDPS<Primitive::IdealGas, Primitive::ResetFloor> *>(
          pmbp->pdyngr);
  if (ptest_ideal == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "The M1 linear wave problem generator requires <mhd> dyn_eos = ideal"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  int isg = is - indcs.ng;
  int ieg = indcs.ie + indcs.ng;
  int jsg = (indcs.nx2 > 1) ? js - indcs.ng : js;
  int jeg = (indcs.nx2 > 1) ? indcs.je + indcs.ng : indcs.je;
  int ksg = (indcs.nx3 > 1) ? ks - indcs.ng : ks;
  int keg = (indcs.nx3 > 1) ? indcs.ke + indcs.ng : indcs.ke;
  int nmb = pmbp->nmb_thispack;
  adm::ADM::ADM_vars &adm = pmbp->padm->adm;
  auto &size = pmbp->pmb->mb_size;
  DvceArray5D<Real> w0_ = pmbp->pmhd->w0;

  // flat, static Minkowski background metric -- same as every other M1 test
  par_for(
      "pgen_m1_linwave_metric", DevExeSpace(), 0, nmb - 1, ksg, keg, jsg, jeg, isg, ieg,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        for (int a = 0; a < 3; ++a)
          for (int b = a; b < 3; ++b) {
            adm.g_dd(m, a, b, k, j, i) = (a == b ? 1. : 0.);
          }
        adm.psi4(m, k, j, i) = 1.;
        adm.alpha(m, k, j, i) = 1.;
        adm.beta_u(m, 0, k, j, i) = 0.;
        adm.beta_u(m, 1, k, j, i) = 0.;
        adm.beta_u(m, 2, k, j, i) = 0.;
      });

  // read the (precomputed offline, matching the paper's Appendix A Table 2 "H1" case)
  // background state and complex eigenvalue/eigenvector -- same keys/values as
  // rad_linwave.athinput's <problem> block, read verbatim so both codes solve the
  // literal same problem
  Real rho0 = pin->GetReal("problem", "rho");
  Real pgas0 = pin->GetReal("problem", "pgas");
  Real erad0 = pin->GetReal("problem", "erad");
  Real fxrad0 = pin->GetOrAddReal("problem", "fxrad", 0.0);
  Real delta = pin->GetReal("problem", "delta");
  Real drho_real = pin->GetReal("problem", "drho_real");
  Real drho_imag = pin->GetOrAddReal("problem", "drho_imag", 0.0);
  Real dpgas_real = pin->GetReal("problem", "dpgas_real");
  Real dpgas_imag = pin->GetOrAddReal("problem", "dpgas_imag", 0.0);
  Real dux_real = pin->GetReal("problem", "dux_real");
  Real dux_imag = pin->GetOrAddReal("problem", "dux_imag", 0.0);
  Real derad_real = pin->GetReal("problem", "derad_real");
  Real derad_imag = pin->GetOrAddReal("problem", "derad_imag", 0.0);
  Real dfxrad_real = pin->GetReal("problem", "dfxrad_real");
  Real dfxrad_imag = pin->GetOrAddReal("problem", "dfxrad_imag", 0.0);
  Real k_par = 2.0 * M_PI / (pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min);

  // set primitive variables -- wave-space perturbation evaluated at t=0 (sn=sin(k x),
  // cn=cos(k x); DO's own general damped-in-time formula reduces to exactly this at
  // t=0, see rad_linear_wave.cpp:226-234)
  par_for(
      "pgen_m1_linwave_prim", DevExeSpace(), 0, nmb - 1, ksg, keg, jsg, jeg, isg, ieg,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        Real x1v = CellCenterX(i - is, indcs.nx1, x1min, x1max);
        Real sn = sin(k_par * x1v);
        Real cn = cos(k_par * x1v);

        w0_(m, IDN, k, j, i) = rho0 + delta * (drho_real * cn - drho_imag * sn);
        w0_(m, IPR, k, j, i) = pgas0 + delta * (dpgas_real * cn - dpgas_imag * sn);
        w0_(m, IVX, k, j, i) = delta * (dux_real * cn - dux_imag * sn);
        w0_(m, IVY, k, j, i) = 0.0;
        w0_(m, IVZ, k, j, i) = 0.0;
      });
  pmbp->pdyngr->PrimToConInit(0, (indcs.nx1 + 2 * indcs.ng - 1),
                              0, (indcs.nx2 > 1 ? indcs.nx2 + 2 * indcs.ng - 1 : 0),
                              0, (indcs.nx3 > 1 ? indcs.nx3 + 2 * indcs.ng - 1 : 0));

  // Set the M1 radiation field: boost the fluid-frame (Eddington-closure) radiation
  // moments into the lab (Eulerian-observer) frame M1 evolves, following DO's own
  // rad_linear_wave.cpp boost sequence (lines 279-344) -- simplified since the wave
  // only ever propagates along x1 here, so no further wave-direction rotation is
  // needed (DO's own rotation collapses to the identity for along_x1).
  auto &u0_ = pmbp->pradm1->u0;
  auto &m1_params_ = pmbp->pradm1->params;
  par_for(
      "pgen_m1_linwave_rad", DevExeSpace(), 0, nmb - 1, ksg, keg, jsg, jeg, isg, ieg,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        Real x1v = CellCenterX(i - is, indcs.nx1, x1min, x1max);
        Real sn = sin(k_par * x1v);
        Real cn = cos(k_par * x1v);

        // fluid 4-velocity (contravariant, wave-aligned == coordinate-aligned since
        // the wave propagates along x1): u^x perturbed, u^y=u^z=0
        Real ux = delta * (dux_real * cn - dux_imag * sn);
        Real u0 = sqrt(1.0 + SQR(ux));

        // fluid-frame radiation moments, Eddington closure (paper's Appendix A Eq. A6)
        Real erad = erad0 + delta * (derad_real * cn - derad_imag * sn);
        Real fxrad = fxrad0 + delta * (dfxrad_real * cn - dfxrad_imag * sn);
        Real rf00 = erad;
        Real rf01 = fxrad;
        Real rf11 = (1.0 / 3.0) * erad;

        // fluid-frame -> lab-frame boost (flat metric, boost purely along x1;
        // matches DO's lambda_c_f_wave/r_wave construction exactly for this
        // special case)
        Real lam00 = u0, lam01 = ux;
        Real lam11 = 1.0 + SQR(ux) / (1.0 + u0);
        // r_lab[a][b] = lam[a][0]*lam[b][0]*rf00 + lam[a][0]*lam[b][1]*rf01
        //             + lam[a][1]*lam[b][0]*rf01 + lam[a][1]*lam[b][1]*rf11
        Real E_lab = lam00 * lam00 * rf00 + 2.0 * lam00 * lam01 * rf01 +
                     lam01 * lam01 * rf11;
        Real Fx_lab = lam00 * lam01 * rf00 + (lam00 * lam11 + lam01 * lam01) * rf01 +
                      lam01 * lam11 * rf11;

        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
        AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_uu{};
        g_uu(0, 0) = -1;
        g_uu(1, 1) = g_uu(2, 2) = g_uu(3, 3) = 1;
        pack_F_d(0, 0, 0, Fx_lab, 0.0, 0.0, F_d);
        radiationm1::apply_floor(g_uu, E_lab, F_d, m1_params_);

        u0_(m, M1_E_IDX, k, j, i) = E_lab;
        u0_(m, M1_FX_IDX, k, j, i) = F_d(1);
        u0_(m, M1_FY_IDX, k, j, i) = F_d(2);
        u0_(m, M1_FZ_IDX, k, j, i) = F_d(3);
      });

  return;
}
