//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file adm_z4c.cpp
//! \brief implementation of functions in the Z4c class related to ADM decomposition

// C standard headers
#include <math.h> // pow
#include <cstdlib>

// C++ standard headers
#include <iostream>
#include <fstream>

// Athena++ headers
#include "parameter_input.hpp"
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "z4c/z4c.hpp"
#include "z4c/tmunu.hpp"
#include "coordinates/cell_locations.hpp"

namespace z4c {

//! \fn void Z4c::ADMToZ4c(MeshBlockPack *pmbp, ParameterInput *pin)
//! \brief Compute Z4c variables from ADM variables
//
// p  = detgbar^(-1/3)
// p0 = psi^(-4)
//
// gtilde_ij = p gbar_ij
// Ktilde_ij = p p0 K_ij
//
// phi = - log(p) / 4
// K   = gtildeinv^ij Ktilde_ij
// Atilde_ij = Ktilde_ij - gtilde_ij K / 3
//
// G^i = - del_j gtildeinv^ji
//
// BAM: Z4c_init()
// https://git.tpi.uni-jena.de/bamdev/z4
// https://git.tpi.uni-jena.de/bamdev/z4/blob/master/z4_init.m
//
// The Z4c variables will be set on the whole MeshBlock with the exception of
// the Gamma's that can only be set in the interior of the MeshBlock.
//
// 2D Cartoon (M2a): the pointwise part runs on the plane as in 3D. Then the Z4c set is
// filled into u_cart (K12, K3), g-tilde^ij is inverted on all 1+2ng cache planes, and
// Gamma-tilde^i = -d_j g-tilde^ji is taken on the y = 0 cache plane with the same FD
// operator as CalcRHS (y spacing dx1, D-001), written back to u0.
template <int NGHOST>
void Z4c::ADMToZ4c(MeshBlockPack *pmbp, ParameterInput *pin) {
  // capture variables for the kernel
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int isg = is-indcs.ng; int ieg = ie+indcs.ng;
  // A direction with one cell (nx = 1) has no ghosts: u0 has extent 1 there.
  int ng2 = (indcs.nx2 > 1) ? indcs.ng : 0;
  int ng3 = (indcs.nx3 > 1) ? indcs.ng : 0;
  int jsg = js-ng2; int jeg = je+ng2;
  int ksg = ks-ng3; int keg = ke+ng3;

  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = indcs.nx2 + 2*(indcs.ng);
  int ncells3 = indcs.nx3 + 2*(indcs.ng);
  int nmb = pmbp->nmb_thispack;

  auto &z4c = pmbp->pz4c->z4c;
  auto &adm = pmbp->padm->adm;
  auto &opt = pmbp->pz4c->opt;
  // 2 1D scratch array and 1 2D scratch array
  par_for("initialize z4c fields",DevExeSpace(),
  0,nmb-1,ksg,keg,jsg,jeg,isg,ieg,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> Kt_dd;
    Real detg = adm::SpatialDet(adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
                                adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
                                adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i));
    Real oopsi4 = pow(detg, -1./3.);
    z4c.chi(m,k,j,i) = pow(detg, 1./12.*opt.chi_psi_power);

    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      z4c.g_dd(m,a,b,k,j,i) = oopsi4 * adm.g_dd(m,a,b,k,j,i);
      Kt_dd(a,b)            = oopsi4 * adm.vK_dd(m,a,b,k,j,i);
    }

    detg = adm::SpatialDet(z4c.g_dd(m,0,0,k,j,i), z4c.g_dd(m,0,1,k,j,i),
                           z4c.g_dd(m,0,2,k,j,i), z4c.g_dd(m,1,1,k,j,i),
                           z4c.g_dd(m,1,2,k,j,i), z4c.g_dd(m,2,2,k,j,i));
    z4c.vKhat(m,k,j,i) = adm::Trace(1.0/detg,
                              z4c.g_dd(m,0,0,k,j,i), z4c.g_dd(m,0,1,k,j,i),
                              z4c.g_dd(m,0,2,k,j,i), z4c.g_dd(m,1,1,k,j,i),
                              z4c.g_dd(m,1,2,k,j,i), z4c.g_dd(m,2,2,k,j,i),
                              Kt_dd(0,0), Kt_dd(0,1), Kt_dd(0,2),
                              Kt_dd(1,1), Kt_dd(1,2), Kt_dd(2,2));

    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      z4c.vA_dd(m,a,b,k,j,i) = Kt_dd(a,b) - (1./3.) *
                                z4c.vKhat(m,k,j,i) * z4c.g_dd(m,a,b,k,j,i);
    }
  });
  Kokkos::fence();

  // Cartoon: the metric is read from the filled cache (planes 0..2ng; ncells3 = 1+2ng on
  // a 2D mesh), Gamma is evaluated on cache plane k and written to u0 at kw = k - koff.
  const bool cartoon = pmbp->pz4c->IsCartoon();
  if (cartoon) pmbp->pz4c->CartoonFill(pmbp);
  const int koff = cartoon ? indcs.ng - ks : 0;
  auto &z4cr = cartoon ? pmbp->pz4c->z4c_cart : pmbp->pz4c->z4c;
  const int kinvs = cartoon ? 0 : ksg, kinve = cartoon ? 2*indcs.ng : keg;

  DvceArray5D<Real> g_uu("g_uu", nmb, 6, ncells3, ncells2, ncells1);
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> g3u;
  g3u.InitWithShallowSlice(g_uu, 0, 5);
  // GLOOP
  par_for("invert z4c metric",DevExeSpace(),
  0,nmb-1,kinvs,kinve,jsg,jeg,isg,ieg,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i){
    Real detg = adm::SpatialDet(z4cr.g_dd(m,0,0,k,j,i), z4cr.g_dd(m,0,1,k,j,i),
                                z4cr.g_dd(m,0,2,k,j,i), z4cr.g_dd(m,1,1,k,j,i),
                                z4cr.g_dd(m,1,2,k,j,i), z4cr.g_dd(m,2,2,k,j,i));
    adm::SpatialInv(1.0/detg,
              z4cr.g_dd(m,0,0,k,j,i), z4cr.g_dd(m,0,1,k,j,i), z4cr.g_dd(m,0,2,k,j,i),
              z4cr.g_dd(m,1,1,k,j,i), z4cr.g_dd(m,1,2,k,j,i), z4cr.g_dd(m,2,2,k,j,i),
              &g3u(m,0,0,k,j,i), &g3u(m,0,1,k,j,i), &g3u(m,0,2,k,j,i),
              &g3u(m,1,1,k,j,i), &g3u(m,1,2,k,j,i), &g3u(m,2,2,k,j,i));
  });
  Kokkos::fence();

  // Compute Gammas
  // Compute only for internal points
  // ILOOP
  /*int const &IZ4CGAMX = pmbp->pz4c->I_Z4C_GAMX;
  int const &IZ4CGAMY = pmbp->pz4c->I_Z4C_GAMY;
  int const &IZ4CGAMZ = pmbp->pz4c->I_Z4C_GAMZ;
  auto              &u0 = pmbp->pz4c->u0;
  sub_DvceArray5D_0D g_00 = Kokkos::subview(g_uu, Kokkos::ALL, 0,
                            Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  sub_DvceArray5D_0D g_01 = Kokkos::subview(g_uu, Kokkos::ALL, 1,
                            Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  sub_DvceArray5D_0D g_02 = Kokkos::subview(g_uu, Kokkos::ALL, 2,
                            Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  sub_DvceArray5D_0D g_11 = Kokkos::subview(g_uu, Kokkos::ALL, 3,
                            Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  sub_DvceArray5D_0D g_12 = Kokkos::subview(g_uu, Kokkos::ALL, 4,
                            Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  sub_DvceArray5D_0D g_22 = Kokkos::subview(g_uu, Kokkos::ALL, 5,
                            Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);*/
  par_for("initialize Gamma",DevExeSpace(),0,nmb-1,ks+koff,ke+koff,js,je,is,ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // u0 (write) index; k reads g3u (cache planes under Cartoon)
    const int kw = k - koff;
    // Usage of Dx: pmbp->pz4c->Dx(blockn, posvar, k,j,i, dir, nghost, dx, quantity);
    Real idx[] = {1/size.d_view(m).dx1, 1/size.d_view(m).dx2,
                  cartoon ? 1/size.d_view(m).dx1 : 1/size.d_view(m).dx3};
    /*AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> g_uu;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_udd;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> dg_ddd;
    Real detg = adm::SpatialDet(z4c.g_dd(m,0,0,k,j,i), z4c.g_dd(m,0,1,k,j,i),
                                z4c.g_dd(m,0,2,k,j,i), z4c.g_dd(m,1,1,k,j,i),
                                z4c.g_dd(m,1,2,k,j,i), z4c.g_dd(m,2,2,k,j,i));
    adm::SpatialInv(1.0/detg,
              z4c.g_dd(m,0,0,k,j,i), z4c.g_dd(m,0,1,k,j,i), z4c.g_dd(m,0,2,k,j,i),
              z4c.g_dd(m,1,1,k,j,i), z4c.g_dd(m,1,2,k,j,i), z4c.g_dd(m,2,2,k,j,i),
              &g_uu(0,0), &g_uu(0,1), &g_uu(0,2),
              &g_uu(1,1), &g_uu(1,2), &g_uu(2,2));
    for (int a = 0; a < 3; ++a)
    for (int b = 0; b < a; ++b)
    for (int c = 0; c < 3; ++c) {
      dg_ddd(c,a,b) = Dx<NGHOST>(c, idx, z4c.g_dd, m, a, b, k, j, i);
    }*/
    /*u0(m,IZ4CGAMX,k,j,i) = -Dx<NGHOST>(0, idx, g_00, m, k, j, i)  // d/dx g00
                           -Dx<NGHOST>(1, idx, g_01, m, k, j, i)  // d/dy g01
                           -Dx<NGHOST>(2, idx, g_02, m, k, j, i); // d/dz g02
    u0(m,IZ4CGAMY,k,j,i) = -Dx<NGHOST>(0, idx, g_01, m, k, j, i)  // d/dx g01
                           -Dx<NGHOST>(1, idx, g_11, m, k, j, i)  // d/dy g11
                           -Dx<NGHOST>(2, idx, g_12, m, k, j, i); // d/dz g12
    u0(m,IZ4CGAMZ,k,j,i) = -Dx<NGHOST>(0, idx, g_02, m, k, j, i)  // d/dx g01
                           -Dx<NGHOST>(1, idx, g_12, m, k, j, i)  // d/dy g11
                           -Dx<NGHOST>(2, idx, g_22, m, k, j, i); // d/dz g12*/
    /*for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
    for (int c = 0; c < b; ++c) {
      Gamma_udd(a, b, c) = 0.0;
      for (int d = 0; d < 3; ++d) {
        Gamma_udd(a, b, c) += 0.5*g_uu(a, d)*
          (-dg_ddd(d, b, c) + dg_ddd(b, d, c) + dg_ddd(c, b, d));
      }
    }
    for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
    for (int c = 0; c < 3; ++c) {
      z4c.vGam_u(m, a, k, j, i) += g_uu(b, c)*Gamma_udd(a, b, c);
    }*/
    for (int a = 0; a < 3; ++a) {
      z4c.vGam_u(m, a, kw, j, i) = 0.0;
      for (int b = 0; b < 3; ++b) {
        z4c.vGam_u(m, a, kw, j, i) -= Dx<NGHOST>(b, idx, g3u, m, b, a, k, j, i);
      }
    }
  });
  AlgConstr(pmbp);
  return;
}
template void Z4c::ADMToZ4c<2>(MeshBlockPack *pmbp, ParameterInput *pin);
template void Z4c::ADMToZ4c<3>(MeshBlockPack *pmbp, ParameterInput *pin);
template void Z4c::ADMToZ4c<4>(MeshBlockPack *pmbp, ParameterInput *pin);
//----------------------------------------------------------------------------------------
//! \fn void Z4c::Z4cToADM(MeshBlockPack *pmbp)
//! \brief Compute ADM Psi4, g_ij, and K_ij from Z4c variables
//
// This sets the ADM variables everywhere in the MeshBlock
void Z4c::Z4cToADM(MeshBlockPack *pmbp) {
  // capture variables for the kernel
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  //For GLOOPS
  int isg = is-indcs.ng; int ieg = ie+indcs.ng;
  // A direction with one cell (nx = 1) has no ghosts: u0 has extent 1 there.
  int ng2 = (indcs.nx2 > 1) ? indcs.ng : 0;
  int ng3 = (indcs.nx3 > 1) ? indcs.ng : 0;
  int jsg = js-ng2; int jeg = je+ng2;
  int ksg = ks-ng3; int keg = ke+ng3;

  int nmb = pmbp->nmb_thispack;

  auto &z4c = pmbp->pz4c->z4c;
  auto &adm = pmbp->padm->adm;
  auto &opt = pmbp->pz4c->opt;
  par_for("initialize z4c fields",DevExeSpace(),
  0,nmb-1,ksg,keg,jsg,jeg,isg,ieg,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    adm.psi4(m,k,j,i) = pow(z4c.chi(m,k,j,i), 4./opt.chi_psi_power);

    // g_ab
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      adm.g_dd(m,a,b,k,j,i) = adm.psi4(m,k,j,i) * z4c.g_dd(m,a,b,k,j,i);
    }

    // K_ab
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      adm.vK_dd(m,a,b,k,j,i) = adm.psi4(m,k,j,i) * z4c.vA_dd(m,a,b,k,j,i) +
        (1./3.) * (z4c.vKhat(m,k,j,i) + 2.*z4c.vTheta(m,k,j,i)) * adm.g_dd(m,a,b,k,j,i);
    }
  });
  return;
}
//----------------------------------------------------------------------------------------
//! \fn void Z4c::ADMConstraints(AthenaArray<Real> & u_adm, AthenaArray<Real> & u_mat)
//! \brief compute constraints ADM vars
//
// Note: we are assuming that u_adm has been initialized with the correct
// metric and matter quantities
//
// BAM: adm_constraints_N()
// https://git.tpi.uni-jena.de/bamdev/adm
// https://git.tpi.uni-jena.de/bamdev/adm/blob/master/adm_constraints_N.m
//
// The constraints are set only in the MeshBlock interior, because derivatives
// of the ADM quantities are needed to compute them.
//
// 2D Cartoon (M2a): the Z4c and ADM sets are filled into their caches (u_cart,
// u_adm_cart) and every field is read there on plane k (y spacing dx1, D-001); the
// constraints are written at the mesh index kw = k - koff, as in CalcRHS.
template <int NGHOST>
void Z4c::ADMConstraints(MeshBlockPack *pmbp) {
  // capture variables for the kernel
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  //For GLOOPS

  int nmb = pmbp->nmb_thispack;

  const bool cartoon = pmbp->pz4c->IsCartoon();
  if (cartoon) {
    pmbp->pz4c->CartoonFill(pmbp);
    pmbp->pz4c->CartoonFillADM(pmbp);
  }
  const int koff = cartoon ? indcs.ng - ks : 0;
  auto &z4c = cartoon ? pmbp->pz4c->z4c_cart : pmbp->pz4c->z4c;
  // the ADM fields read below (psi4, g_ij, K_ij): the mesh arrays, or the Cartoon cache
  ADM_vars adm;
  if (cartoon) {
    adm = pmbp->pz4c->adm_cart;
  } else {
    adm.psi4 = pmbp->padm->adm.psi4;
    adm.g_dd = pmbp->padm->adm.g_dd;
    adm.vK_dd = pmbp->padm->adm.vK_dd;
  }
  auto &u_con = pmbp->pz4c->u_con;

  // vacuum or with matter?
  bool is_vacuum = (pmy_pack->ptmunu == nullptr) ? true : false;
  Tmunu::Tmunu_vars tmunu;
  if (!is_vacuum) tmunu = pmy_pack->ptmunu->tmunu;

  Kokkos::deep_copy(u_con, 0.);
  auto &con = pmbp->pz4c->con;
  par_for("ADM constraints loop",DevExeSpace(),
  0,nmb-1,ks+koff,ke+koff,js,je,is,ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // mesh (write) index; k reads z4c/adm (caches under Cartoon)
    const int kw = k - koff;
    AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> Gamma_u;
    AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> Gamma_u_z4c;
    AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> M_u;
    AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dpsi4_d;

    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> g_uu;
    //AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> g_uu_z4c;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> R_dd;
    AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> K_ud;

    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> dg_ddd;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> dg_ddd_z4c;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> dK_ddd;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_ddd;
    //AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_ddd_z4c;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_udd;
    //AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_udd_z4c;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> DK_ddd;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> DK_udd;

    AthenaPointTensor<Real, TensorSymm::SYM22, 3, 4> ddg_dddd;

    // Cartoon: the ghost planes in the x3 index are spaced dx1 (D-001)
    Real idx[] = {1/size.d_view(m).dx1, 1/size.d_view(m).dx2,
                  cartoon ? 1/size.d_view(m).dx1 : 1/size.d_view(m).dx3};

    // -----------------------------------------------------------------------------------
    // derivatives
    //
    // first derivatives of g and K
    for(int c = 0; c < 3; ++c)
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      dg_ddd(c,a,b) = Dx<NGHOST>(c, idx, adm.g_dd, m,a,b,k,j,i);
      dg_ddd_z4c(c,a,b) = Dx<NGHOST>(c, idx, z4c.g_dd, m,a,b,k,j,i);
      dK_ddd(c,a,b) = Dx<NGHOST>(c, idx, adm.vK_dd, m,a,b,k,j,i);
    }

    // first derivative of psi4
    for (int a =0; a < 3; ++a) {
      dpsi4_d(a) = Dx<NGHOST>(a, idx, adm.psi4, m, k, j, i);
    }

    // second derivatives of g
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b)
    for(int c = 0; c < 3; ++c)
    for(int d = c; d < 3; ++d) {
      if(a == b) {
        ddg_dddd(a,a,c,d) = Dxx<NGHOST>(a, idx, adm.g_dd, m,c,d,k,j,i);
      } else {
        ddg_dddd(a,b,c,d) = Dxy<NGHOST>(a, b, idx, adm.g_dd, m,c,d,k,j,i);
      }
    }

    // -----------------------------------------------------------------------------------
    // inverse metric
    //
    Real detg = adm::SpatialDet(adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
                                adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
                                adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i));
    adm::SpatialInv(1./detg,
               adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i), adm.g_dd(m,0,2,k,j,i),
               adm.g_dd(m,1,1,k,j,i), adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i),
               &g_uu(0,0), &g_uu(0,1), &g_uu(0,2),
               &g_uu(1,1), &g_uu(1,2), &g_uu(2,2));

    /*Real detg_z4c = adm::SpatialDet(z4c.g_dd(m,0,0,k,j,i), z4c.g_dd(m,0,1,k,j,i),
                                z4c.g_dd(m,0,2,k,j,i), z4c.g_dd(m,1,1,k,j,i),
                                z4c.g_dd(m,1,2,k,j,i), z4c.g_dd(m,2,2,k,j,i));
    adm::SpatialInv(1./detg_z4c,
               z4c.g_dd(m,0,0,k,j,i), z4c.g_dd(m,0,1,k,j,i), z4c.g_dd(m,0,2,k,j,i),
               z4c.g_dd(m,1,1,k,j,i), z4c.g_dd(m,1,2,k,j,i), z4c.g_dd(m,2,2,k,j,i),
               &g_uu_z4c(0,0), &g_uu_z4c(0,1), &g_uu_z4c(0,2),
               &g_uu_z4c(1,1), &g_uu_z4c(1,2), &g_uu_z4c(2,2));*/

    // -----------------------------------------------------------------------------------
    // Christoffel symbols
    //
    for(int c = 0; c < 3; ++c)
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      Gamma_ddd(c,a,b) = 0.5*(dg_ddd(a,b,c) + dg_ddd(b,a,c) - dg_ddd(c,a,b));
      Gamma_udd(c,a,b) = 0.0;
    }

    for(int c = 0; c < 3; ++c)
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b)
    for(int d = 0; d < 3; ++d) {
      Gamma_udd(c,a,b) += g_uu(c,d)*Gamma_ddd(d,a,b);
    }

    for(int a = 0; a < 3; ++a) {
      Gamma_u(a) = 0.0;
      for(int b = 0; b < 3; ++b)
      for(int c = 0; c < 3; ++c) {
        Gamma_u(a) += g_uu(b,c)*Gamma_udd(a,b,c);
      }
    }

    // same but for z4c metric
    /*for(int c = 0; c < 3; ++c)
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      Gamma_ddd_z4c(c,a,b) = 0.5*(dg_ddd_z4c(a,b,c)
                          + dg_ddd_z4c(b,a,c) - dg_ddd_z4c(c,a,b));
      Gamma_udd_z4c(c,a,b) = 0.0;
    }

    for(int c = 0; c < 3; ++c)
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b)
    for(int d = 0; d < 3; ++d) {
      Gamma_udd_z4c(c,a,b) += g_uu_z4c(c,d)*Gamma_ddd_z4c(d,a,b);
    }

    for(int a = 0; a < 3; ++a) {
      Gamma_u_z4c(a) = 0.0;
      for(int b = 0; b < 3; ++b)
      for(int c = 0; c < 3; ++c) {
        Gamma_u_z4c(a) += g_uu_z4c(b,c)*Gamma_udd_z4c(a,b,c);
      }
    }*/
    // Find the contracted conformal Christoffel symbol
    for (int a = 0; a < 3; ++a) {
      Gamma_u_z4c(a) = adm.psi4(m,k,j,i)*Gamma_u(a);
      for (int b = 0; b < 3; ++b) {
        Gamma_u_z4c(a) += 0.5*g_uu(a,b)*dpsi4_d(b);
      }
    }

    // -----------------------------------------------------------------------------------
    // Ricci tensor and Ricci scalar
    //
    Real R = 0.0;
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      R_dd(a,b) = 0.0;
      for(int c = 0; c < 3; ++c)
      for(int d = 0; d < 3; ++d) {
        // Part with the Christoffel symbols
        for(int e = 0; e < 3; ++e) {
          R_dd(a,b) += g_uu(c,d) * Gamma_udd(e,a,c) * Gamma_ddd(e,b,d);
          R_dd(a,b) -= g_uu(c,d) * Gamma_udd(e,a,b) * Gamma_ddd(e,c,d);
        }
        // Wave operator part of the Ricci
        R_dd(a,b) += 0.5*g_uu(c,d)*(
            - ddg_dddd(c,d,a,b) - ddg_dddd(a,b,c,d) +
              ddg_dddd(a,c,b,d) + ddg_dddd(b,c,a,d));
      }
    }

    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b) {
      R += g_uu(a,b) * R_dd(a,b);
    }

    // -----------------------------------------------------------------------------------
    // Extrinsic curvature: traces and derivatives
    //
    Real K = 0.0;
    for(int a = 0; a < 3; ++a) {
      for(int b = 0; b < 3; ++b) {
        K_ud(a,b) = 0.0;
        for(int c = 0; c < 3; ++c) {
          K_ud(a,b) += g_uu(a,c) * adm.vK_dd(m,c,b,k,j,i);
        }
      }
      K += K_ud(a,a);
    }

    // K^a_b K^b_a
    Real KK = 0.0;
    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b) {
      KK += K_ud(a,b) * K_ud(b,a);
    }

    // Covariant derivative of K
    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b)
    for(int c = b; c < 3; ++c) {
      DK_ddd(a,b,c) = dK_ddd(a,b,c);
      for(int d = 0; d < 3; ++d) {
        DK_ddd(a,b,c) -= Gamma_udd(d,a,b) * adm.vK_dd(m,d,c,k,j,i);
        DK_ddd(a,b,c) -= Gamma_udd(d,a,c) * adm.vK_dd(m,b,d,k,j,i);
      }
    }

    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b)
    for(int c = b; c < 3; ++c) {
      DK_udd(a,b,c) = 0.0;
      for(int d = 0; d < 3; ++d) {
        DK_udd(a,b,c) += g_uu(a,d) * DK_ddd(d,b,c);
      }
    }

    // -----------------------------------------------------------------------------------
    // Actual constraints
    //
    // Hamiltonian constraint
    //
    con.H(m,kw,j,i) = R + SQR(K) - KK;
    if(!is_vacuum) {
      con.H(m,kw,j,i) -= 16*M_PI * tmunu.E(m,kw,j,i);
    }
    // Momentum constraint (contravariant)
    //
    for(int a = 0; a < 3; ++a) {
      M_u(a) = 0.0;
      for(int b = 0; b < 3; ++b) {
        if(!is_vacuum) {
          M_u(a) -= 8*M_PI * g_uu(a,b) * tmunu.S_d(m,b,kw,j,i);
        }
        for(int c = 0; c < 3; ++c) {
          M_u(a) += g_uu(a,b) * DK_udd(c,b,c);
          M_u(a) -= g_uu(b,c) * DK_udd(a,b,c);
        }
      }
    }

    // Momentum constraint (covariant)
    for(int a = 0; a < 3; ++a) {
      for(int b = 0; b < 3; ++b) {
        con.M_d(m,a,kw,j,i) += adm.g_dd(m,a,b,k,j,i) * M_u(b);
      }
    }

    // Momentum constraint (norm squared)
    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b) {
      con.M(m,kw,j,i) += adm.g_dd(m,a,b,k,j,i) * M_u(a) * M_u(b);
    }

    // Constraint violation Z (norm squared)
    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b) {
      con.Z(m,kw,j,i) += 0.25*z4c.g_dd(m,a,b,k,j,i)
                        *(z4c.vGam_u(m,a,k,j,i) - Gamma_u_z4c(a))
                        *(z4c.vGam_u(m,b,k,j,i) - Gamma_u_z4c(b));
    }

    // Constraint violation monitor C^2
    con.C(m,kw,j,i) = SQR(con.H(m,kw,j,i)) + con.M(m,kw,j,i) +
                     SQR(z4c.vTheta(m,k,j,i)) + 4.0*con.Z(m,kw,j,i);
});
}
template void Z4c::ADMConstraints<2>(MeshBlockPack *pmbp);
template void Z4c::ADMConstraints<3>(MeshBlockPack *pmbp);
template void Z4c::ADMConstraints<4>(MeshBlockPack *pmbp);
} // namespace z4c
