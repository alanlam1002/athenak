//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_m1_diffusiontest.cpp
//  \brief 1D diffusion test in a moving medium for grey M1

// C++ headers

// Athena++ headers
#include <coordinates/cell_locations.hpp>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/eos.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "radiation_m1/radiation_m1.hpp"
#include "radiation_m1/radiation_m1_helpers.hpp"

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::UserProblem(ParameterInput *pin)
//  \brief Sets initial conditions for radiation M1 beams test

void ProblemGenerator::RadiationM1DiffusionTest(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (pmbp->pradm1 == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "The 2d lattice problem generator can only be run with "
                 "radiation-m1, but no "
              << "<radiation_m1> block in input file" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!pmbp->pmesh->one_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "The 1d diffusion test problem generator can only be run with one "
                 "dimension, but parfile"
              << "grid setup is not in 1d" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->pradm1->nspecies != 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "The 1d diffusion test problem generator can only be run with "
                 "one neutrino species only!"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  if (pmbp->pradm1->params.opacity_type == radiationm1::Toy) {
    if (pmbp->pradm1->params.src_update == radiationm1::Explicit) {
      pmbp->pradm1->toy_opacity_fn = radiationm1::ToyOpacity{radiationm1::ToyOpacityModel::DiffusionExplicit};
    }
    if (pmbp->pradm1->params.src_update == radiationm1::Implicit) {
      pmbp->pradm1->toy_opacity_fn = radiationm1::ToyOpacity{radiationm1::ToyOpacityModel::DiffusionImplicit};
    }
  }

  // opacity_type=photons needs real density/pressure to compute opacities from
  // (CalcOpacityPhotons_IdealGas_ reads w0_(IDN)/w0_(IEN)), so it uses the real
  // <mhd> fluid instead of the placeholder pradm1->w0 the toy model gets by with.
  bool use_mhd = (pmbp->pradm1->params.opacity_type == radiationm1::Photons);
  dyngr::DynGRMHDPS<Primitive::IdealGas, Primitive::ResetFloor> *ptest_ideal = nullptr;
  Real mb = 1.0;
  if (use_mhd) {
    ptest_ideal =
        dynamic_cast<dyngr::DynGRMHDPS<Primitive::IdealGas, Primitive::ResetFloor> *>(
            pmbp->pdyngr);
    if (ptest_ideal == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "The diffusion test problem generator requires <mhd> "
                   "dyn_eos = ideal when opacity_type = photons"
                << std::endl;
      exit(EXIT_FAILURE);
    }
    mb = ptest_ideal->eos.ps.GetEOSMutable().GetBaryonMass();
  }

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is;
  int &ie = indcs.ie;
  int &js = indcs.js;
  int &je = indcs.je;
  int &ks = indcs.ks;
  int &ke = indcs.ke;

  int isg = is - indcs.ng;
  int ieg = ie + indcs.ng;
  int jsg = (indcs.nx2 > 1) ? js - indcs.ng : js;
  int jeg = (indcs.nx2 > 1) ? je + indcs.ng : je;
  int ksg = (indcs.nx3 > 1) ? ks - indcs.ng : ks;
  int keg = (indcs.nx3 > 1) ? ke + indcs.ng : ke;
  int nmb = pmbp->nmb_thispack;
  DvceArray5D<Real> w0_ = pmbp->pradm1->w0;
  if (use_mhd) {
    w0_ = pmbp->pmhd->w0;
  }
  auto &u0_ = pmbp->pradm1->u0;
  adm::ADM::ADM_vars &adm = pmbp->padm->adm;
  auto &params_ = pmbp->pradm1->params;

  auto initial_data = pin->GetOrAddString("problem", "initial_data", "gaussian");
  if (initial_data != "gaussian" && initial_data != "step") {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "Unknown problem/initial_data: " << initial_data
              << " (expected gaussian or step)" << std::endl;
    exit(EXIT_FAILURE);
  }
  bool erf = (initial_data == "gaussian");
  auto vx = pin->GetOrAddReal("problem", "fluid_velocity", 0.0);
  auto lorentz_w = 1. / Kokkos::sqrt(1. - vx * vx);
  Real rho = pin->GetOrAddReal("problem", "rho", 1.0);
  Real temp = pin->GetOrAddReal("problem", "temp", 1.0);
  Real nb = rho / mb;

  // Diffusion coefficient for the Gaussian IC's flux, below (photon path only;
  // the toy path prescribes scat_1 directly, not through photon_op_params).
  Real dd = 0.0;
  if (use_mhd) {
    dd = 1.0 / (3.0 * pmbp->pradm1->photon_op_params.kappa_s * rho);
  }

  // set metric to minkowski, initialize velocity to zero
  par_for(
      "pgen_diffusiontest_initialize", DevExeSpace(), 0, nmb - 1, ksg, keg, jsg, jeg, isg,
      ieg, KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        for (int a = 0; a < 3; ++a)
          for (int b = a; b < 3; ++b) {
            adm.g_dd(m, a, b, k, j, i) = (a == b ? 1. : 0.);
          }

        adm.psi4(m, k, j, i) = 1.;  // adm.psi4

        adm.alpha(m, k, j, i) = 1.;

        w0_(m, IVX, k, j, i) = lorentz_w * vx;
        w0_(m, IVY, k, j, i) = 0.;
        w0_(m, IVZ, k, j, i) = 0.;

        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        int nx1 = indcs.nx1;
        Real x1 = CellCenterX(i - is, nx1, x1min, x1max);

        Real E{};
        if (erf) {
          E = Kokkos::exp(-9. * x1 * x1);
        } else {
          E = (x1 < 0);
        }

        Real J = 3. * E / (4. * lorentz_w * lorentz_w - 1.);
        Real Fx = (4. / 3.) * J * lorentz_w * lorentz_w * vx;
        if (erf && use_mhd) {
          // Add the static diffusive flux F = -D*dE/dx = D*2*nu^2*x*E (nu^2=9,
          // matching the exp(-9x^2) Gaussian above), i.e. the flux consistent
          // with the exact solution of the diffusion equation at t=0 (see
          // src/pgen/rad_diffusion.cpp's `fr`, simplified to v1=0). Without
          // this, the M1 solver starts from a "non-equilibrated" F=0 IC and
          // needs a spin-up transient (~1/kscat) before flux and gradient are
          // correctly related, which contaminates early-time measurements of
          // the spreading rate (found via a cross-check against the
          // discrete-ordinate rad_diffusion.cpp module).
          Fx += dd * 2. * 9. * x1 * E;
        }

        AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_uu{};
        for (int a = 0; a < 4; ++a) {
          for (int b = 0; b < 4; ++b) {
            g_uu(a, b) = 0;
          }
        }
        g_uu(0, 0) = -1;
        g_uu(1, 1) = 1;
        g_uu(2, 2) = 1;
        g_uu(3, 3) = 1;
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
        pack_F_d(adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
                 adm.beta_u(m, 2, k, j, i), Fx, 0, 0, F_d);
        radiationm1::apply_floor(g_uu, E, F_d, params_);
        u0_(m, M1_E_IDX, k, j, i) = E;
        u0_(m, M1_FX_IDX, k, j, i) = F_d(M1_FX_IDX);
        u0_(m, M1_FY_IDX, k, j, i) = F_d(M1_FY_IDX);
        u0_(m, M1_FZ_IDX, k, j, i) = F_d(M1_FZ_IDX);
      });

  // opacity_type=photons: fill in the uniform density/pressure CalcOpacityPhotons_
  // needs, and convert to MHD conserved variables. Kept as a second kernel (rather
  // than folded into the one above) so the EOS object -- only obtainable once
  // ptest_ideal is known non-null -- is never captured by a Kokkos lambda unless
  // it's actually valid to do so.
  if (use_mhd) {
    Primitive::EOS<Primitive::IdealGas, Primitive::ResetFloor> &eos =
        ptest_ideal->eos.ps.GetEOSMutable();
    par_for(
        "pgen_diffusiontest_mhd_state", DevExeSpace(), 0, nmb - 1, ksg, keg, jsg, jeg,
        isg, ieg, KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
          Real dummy_y{};
          w0_(m, IDN, k, j, i) = rho;
          w0_(m, IPR, k, j, i) = eos.GetPressure(nb, temp, &dummy_y);
        });
    pmbp->pdyngr->PrimToConInit(0, (indcs.nx1 + 2 * indcs.ng - 1), jsg, jeg, ksg, keg);
  }
  return;
}
