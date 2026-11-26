//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_cartoon.cpp
//! \brief Cartoon method for axisymmetric vacuum Z4c on a 2D Cartesian mesh.
//!
//! The mesh holds the plane y = 0 with x = x1 >= 0 (the axis is the face x1 = 0) and
//! z = x2. Derivatives in the missing direction y come from ng ghost planes y = l*dx1,
//! l = -ng..ng, held in the Z4c-local cache u_cart(m, n, 1+2ng, j, i) with local k = ng
//! the y = 0 plane (D-001, D-006). A field value at (x, y) is the value at
//! (rho, 0), rho = sqrt(x^2 + y^2), rotated by the angle with c = x/rho, s = y/rho.
//!
//! Frame (D-017): physical (x, y, z) = mesh (x1, -x3, x2); symmetry axis = mesh x2.
//! In this file "x, z, y" in comments and the rotation below name the MESH directions
//! x1, x2, x3 (so "y" is mesh x3 = physical -y). The fill is a rotation about mesh x2
//! and is symmetric in +-x3, so it is the same in either labelling.
//!
//! The fill (chat/0004 GPU design), run immediately before each consumer (D-007). It acts
//! on any variable set given by a CartoonVarSet descriptor (M2a): the Z4c set
//! (u0 -> u_cart, with the algebraic constraints) and the ADM set (u_adm -> u_adm_cart,
//! without). Two kernels (M2c, D-015: K1 folded into K2, K4 applied in registers):
//!   K12 per (m, l = 0..ng, j, i). l = 0 copies the y = 0 plane from u0 (x1/x2 ghosts
//!       included). l > 0, cells with x_i > 0: Lagrange-interpolate every variable at
//!       rho along x1 (npt points, weights computed in registers), rotate, apply the
//!       algebraic constraints (det g = 1, tr A = 0; Z4c set only, D-007), and write
//!       both planes +l and -l (s -> -s)
//!   K3  cells with x_i < 0 in planes l != 0: pi-rotation copy
//!       Q(x, y) = P_Q Q(-x, -y) with the D-002 parity table (D-008). The copy keeps
//!       det g and tr A (every term has an even number of parity-odd factors), so the
//!       copied values are already constrained.
//! The plane l = 0 at x < 0 is the D-002 axis boundary on u0 (Z4cBCs, CartoonAxisBC).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_pos.hpp"

namespace z4c {

namespace {

// largest supported stencil: npt = 2*ng + 1 with ng <= 4
constexpr int kMaxNpt = 9;
constexpr int kMaxVar = Z4c::kCartoonMaxVar;

// Physical label of each Z4c variable on the Cartoon mesh (D-017): physical (x, y, z) =
// mesh (x1, -x3, x2), so a component with an odd number of x3 indices is minus the
// physical one.
const char *const kPhysName[Z4c::nz4c] = {
  "chi",
  "g_xx", "g_xz", "-g_xy", "g_zz", "-g_zy", "g_yy",
  "Khat",
  "A_xx", "A_xz", "-A_xy", "A_zz", "-A_zy", "A_yy",
  "Gam^x", "Gam^z", "-Gam^y",
  "Theta",
  "alpha",
  "beta^x", "beta^z", "-beta^y",
};

// Rotate one interpolated state T (all vs.nvar variables at (rho, 0)) to (x, y) with
// c = x/rho, s = y/rho:  V' = R V,  G' = R T R^T,  R = [[c, -s], [s, c]] in (x, y).
// Written for the mesh component order: x = x1 (0), z = x2 (1), y = x3 (2).
KOKKOS_INLINE_FUNCTION
void CartoonRotate(const Z4c::CartoonVarSet &vs, const Real *T, const Real c,
                   const Real s, Real *G) {
  const Real c2 = c*c, s2 = s*s, cs = c*s;
  for (int n = 0; n < vs.nvar; ++n) G[n] = T[n];   // scalars, z components
  // symmetric tensors: offsets of the xx entry
  for (int t = 0; t < vs.nsym; ++t) {
    const int o = vs.sym[t];
    // mesh order within a tensor: 0 xx, 1 "xy"=xz, 2 "xz"=xy, 3 "yy"=zz, 4 "yz"=zy,
    // 5 "zz"=yy (physical labels on the right)
    const Real Txx = T[o+0], Txz = T[o+1], Txy = T[o+2];
    const Real Tzy = T[o+4], Tyy = T[o+5];
    G[o+0] = c2*Txx + s2*Tyy - 2.0*cs*Txy;         // xx
    G[o+5] = s2*Txx + c2*Tyy + 2.0*cs*Txy;         // yy
    G[o+2] = (c2 - s2)*Txy + cs*(Txx - Tyy);       // xy
    G[o+1] = c*Txz - s*Tzy;                        // xz
    G[o+4] = s*Txz + c*Tzy;                        // yz
    // G[o+3] = T zz unchanged
  }
  // vectors: (x, z, y) in mesh order
  for (int t = 0; t < vs.nvec; ++t) {
    const int o = vs.vec[t];
    const Real Vx = T[o+0], Vy = T[o+2];
    G[o+0] = c*Vx - s*Vy;
    G[o+2] = s*Vx + c*Vy;
  }
}

// The algebraic constraints (D-007) on a state held in registers: g-tilde at
// T[og..og+5], A-tilde at T[oA..oA+5] (mesh order 00 01 02 11 12 22). Same operations
// as AlgConstrPoint (z4c_algconstr.hpp), which acts on arrays.
KOKKOS_INLINE_FUNCTION
void AlgConstrRegs(Real *T, const int og, const int oA) {
  Real detg = adm::SpatialDet(T[og+0], T[og+1], T[og+2], T[og+3], T[og+4], T[og+5]);
  detg = detg > 0. ? detg : 1.;
  Real oopsi4 = std::cbrt(1./detg);
  for (int c = 0; c < 6; ++c) T[og+c] *= oopsi4;
  Real A = adm::Trace(1.0, T[og+0], T[og+1], T[og+2], T[og+3], T[og+4], T[og+5],
                      T[oA+0], T[oA+1], T[oA+2], T[oA+3], T[oA+4], T[oA+5]);
  for (int c = 0; c < 6; ++c) T[oA+c] -= (1.0/3.0) * A * T[og+c];
}

// Lagrange weights for nodes 0..npt-1 at fractional position t (product form).
KOKKOS_INLINE_FUNCTION
void LagrangeWeights(const int npt, const Real t, Real *w) {
  for (int q = 0; q < npt; ++q) {
    Real wq = 1.0;
    for (int r = 0; r < npt; ++r) {
      if (r != q) wq *= (t - r)/static_cast<Real>(q - r);
    }
    w[q] = wq;
  }
}

// Stencil start for interpolation at rho: centred on the nearest node, then clamped.
// The nodes are the x1 midpoints, x(i) = x0 + i*dx with x0 the midpoint of index 0.
// Returns the unclamped start through i0raw so callers can count clamps.
KOKKOS_INLINE_FUNCTION
int StencilStart(const Real rho, const Real x0, const Real dx,
                 const int npt, const int n1, Real &f, int &i0raw) {
  f = (rho - x0)/dx;                               // fractional cell index of rho
  i0raw = static_cast<int>(Kokkos::floor(f + 0.5)) - (npt - 1)/2;
  return Kokkos::min(Kokkos::max(i0raw, 0), n1 - npt);
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn Z4c::CartoonParity Z4c::CartoonParityTable()
//! \brief THE parity table (D-002): sign of each variable under the pi rotation about the
//! z axis. Odd (reflect -ve): Q^x, Q^y, Q_xz, Q_yz. Even (mirror): everything else.
//! Same table as SACRA's boundary_eqsym_cartoon. The one place it is defined.

Z4c::CartoonParity Z4c::CartoonParityTable() {
  CartoonParity t;
  for (int n = 0; n < nz4c; ++n) t.p[n] = 1;
  t.p[I_Z4C_GXY]   = -1;   // g_xz
  t.p[I_Z4C_GYZ]   = -1;   // g_zy
  t.p[I_Z4C_AXY]   = -1;   // A_xz
  t.p[I_Z4C_AYZ]   = -1;   // A_zy
  t.p[I_Z4C_GAMX]  = -1;   // Gam^x
  t.p[I_Z4C_GAMZ]  = -1;   // Gam^y
  t.p[I_Z4C_BETAX] = -1;   // beta^x
  t.p[I_Z4C_BETAZ] = -1;   // beta^y
  return t;
}

//----------------------------------------------------------------------------------------
//! \fn Z4c::CartoonZ4cSet(), Z4c::CartoonADMSet()
//! \brief the variable-set descriptors of the two Cartoon caches (M2a)

Z4c::CartoonVarSet Z4c::CartoonZ4cSet() {
  CartoonVarSet vs;
  vs.nvar = nz4c;
  vs.nsym = 2; vs.sym[0] = I_Z4C_GXX;  vs.sym[1] = I_Z4C_AXX;
  vs.nvec = 2; vs.vec[0] = I_Z4C_GAMX; vs.vec[1] = I_Z4C_BETAX;
  return vs;
}

Z4c::CartoonVarSet Z4c::CartoonADMSet() {
  // with <z4c>, u_adm holds g_ij, K_ij and psi4 only (alpha, beta^i alias u0)
  CartoonVarSet vs;
  vs.nvar = adm::ADM::I_ADM_PSI4 + 1;
  vs.nsym = 2; vs.sym[0] = adm::ADM::I_ADM_GXX; vs.sym[1] = adm::ADM::I_ADM_KXX;
  vs.nvec = 0; vs.vec[0] = 0; vs.vec[1] = 0;
  return vs;
}

//----------------------------------------------------------------------------------------
//! \fn Z4c::CartoonParity Z4c::CartoonParityOf(const CartoonVarSet &vs)
//! \brief D-002 parity derived from a descriptor: the pi rotation (x,y,z) -> (-x,-y,z)
//! flips mesh components 0 (x) and 2 (y), so a component has the sign of the product of
//! its index signs. Scalars are even.

Z4c::CartoonParity Z4c::CartoonParityOf(const CartoonVarSet &vs) {
  const int sv[3] = {-1, +1, -1};                        // mesh x, z, y
  const int ss[6] = {+1, -1, +1, +1, -1, +1};            // 00 01 02 11 12 22
  CartoonParity t;
  for (int n = 0; n < kCartoonMaxVar; ++n) t.p[n] = 1;
  for (int a = 0; a < vs.nsym; ++a)
    for (int c = 0; c < 6; ++c) t.p[vs.sym[a] + c] = ss[c];
  for (int a = 0; a < vs.nvec; ++a)
    for (int c = 0; c < 3; ++c) t.p[vs.vec[a] + c] = sv[c];
  return t;
}

//----------------------------------------------------------------------------------------
//! \fn void Z4c::CartoonUnsupported(const std::string &what) const
//! \brief fatal exit under 2D Cartoon for a problem generator that is not axisymmetric
//! (or not ported yet); a no-op otherwise.

void Z4c::CartoonUnsupported(const std::string &what) const {
  if (!IsCartoon()) return;
  std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
            << what << " is not supported under 2D Cartoon (D-017)" << std::endl;
  std::exit(EXIT_FAILURE);
}

//----------------------------------------------------------------------------------------
//! \fn void Z4c::CartoonSetup(ParameterInput *pin)
//! \brief read <z4c> cartoon, check the mesh, print the parity table, allocate u_cart.

void Z4c::CartoonSetup(ParameterInput *pin) {
  // D-014: the mode follows the coord and the mesh dimensionality; there is no switch.
  Mesh *pm = pmy_pack->pmesh;
  auto &mi = pm->mesh_indcs;
  auto fatal = [](const std::string &msg) {
    std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
              << "Z4c: " << msg << std::endl;
    std::exit(EXIT_FAILURE);
  };
  switch (pm->coord_general) {
    case CoordinateGeneral::cartesian:
      // Standard Z4c, unchanged (1D/2D Cartesian Z4c is left exactly as it was).
      opt.cartoon = CartoonMode::none;
      return;
    case CoordinateGeneral::cylindrical_axisym:
      if (!pm->two_d) fatal("coord = cylindrical_axisym needs a 2D mesh for Cartoon");
      opt.cartoon = CartoonMode::axisym;
      break;
    case CoordinateGeneral::spherical_polar:
      if (pm->one_d) fatal("1D Cartoon (coord = spherical_polar) is not implemented yet "
                           "(M4)");
      fatal("coord = spherical_polar with <z4c> is only meaningful as 1D Cartoon (M4)");
      break;
    default:
      fatal("<z4c> supports coord = cartesian, or cylindrical_axisym (2D Cartoon)");
  }

  // axisym validity checks (R2). The coord itself already requires x1min >= 0 and a
  // reflecting inner x1 face at x1min = 0 (Mesh::ValidateCoordGeneral).
  if (!(mi.nx2 > 1 && mi.nx3 == 1)) {
    fatal("2D Cartoon requires a 2D mesh (nx2 > 1, nx3 = 1)");
  }
  if (pm->mesh_size.x1min != 0.0) {
    fatal("2D Cartoon requires <mesh> x1min = 0 exactly (the axis)");
  }
  if (pm->mesh_bcs[BoundaryFace::inner_x1] != BoundaryFlag::reflect) {
    fatal("2D Cartoon requires <mesh> ix1_bc = reflect; Z4c replaces it by the "
          "parity (pi-rotation) axis boundary (D-002)");
  }
  if (pin->DoesBlockExist("hydro") || pin->DoesBlockExist("mhd") ||
      pin->GetOrAddBoolean("coord", "general_rel", false)) {
    fatal("2D Cartoon is vacuum only until M5 (no <hydro>/<mhd>, general_rel = false)");
  }
  if (pm->multilevel) {
    fatal("2D Cartoon: mesh refinement is not supported yet (uniform grid only; SMR "
          "comes in M3, D-004)");
  }
  if (pm->mb_indcs.nx1 < pm->mb_indcs.ng) {
    fatal("2D Cartoon: <meshblock> nx1 must be >= nghost");
  }
  // Consumers that would read y-derivatives or Cartesian positions they cannot map yet
  // (Cartoon-aware versions come in M2).
  if (pin->GetOrAddReal("z4c", "nrad_wave_extraction", 0) > 0) {
    fatal("2D Cartoon: wave extraction (Weyl scalars) is not supported yet (M2)");
  }
  if (pin->DoesParameterExist("z4c", "co_0_type")) {
    fatal("2D Cartoon: compact-object trackers are not supported yet (M2)");
  }
  const BoundaryFace somm_faces[3] = {BoundaryFace::outer_x1, BoundaryFace::inner_x2,
                                      BoundaryFace::outer_x2};
  // M2b: outflow/diode/vacuum on the outer x1 face and both x2 faces get the
  // Cartoon-aware Sommerfeld RHS (z4c_Sbc.cpp: radial direction from Z4cCartesianPos,
  // no x3 terms). A user boundary runs a problem-specific function that nothing has
  // checked under Cartoon, so it stays fatal.
  for (auto f : somm_faces) {
    if (pm->mesh_bcs[f] == BoundaryFlag::user) {
      fatal("2D Cartoon: user outer boundaries are not supported (M2b allows outflow, "
            "diode, vacuum and periodic/reflect)");
    }
  }
  // Uniform spacing (R2/R4): AthenaK has no mesh-stretching option (no x1rat/x2rat), so
  // every MeshBlock has uniform dx1 and dx2 by construction; the interpolation relies on
  // it. The Cartoon y spacing is dx1, set internally (D-014).

  {
    // the descriptor-derived parity must reproduce THE table (D-002)
    const CartoonParity t = CartoonParityTable(), d = CartoonParityOf(CartoonZ4cSet());
    for (int n = 0; n < nz4c; ++n) {
      if (t.p[n] != d.p[n]) fatal("Cartoon parity table and Z4c descriptor disagree");
    }
  }
  if (global_variable::my_rank == 0) {
    CartoonParity t = CartoonParityTable();
    std::cout << "Z4c Cartoon axisym: ng = " << pm->mb_indcs.ng
              << ", npt = " << opt.cartoon_npt
              << ", cartoon_ydiss = " << (opt.cartoon_ydiss ? "true" : "false")
              << std::endl
              << "  physical (x, y, z) = mesh (x1, -x3, x2); symmetry axis = mesh x2"
              << std::endl
              << "  axis parity table (D-002):"
              << std::endl;
    for (int n = 0; n < nz4c; ++n) {
      std::printf("    %-10s (%-7s) %s\n", Z4c_names[n], kPhysName[n],
                  t.p[n] > 0 ? "+1 even (mirror)" : "-1 odd  (reflect -ve)");
    }
  }
  CartoonAlloc();
}

//----------------------------------------------------------------------------------------
//! \fn void Z4c::CartoonAlloc()
//! \brief (re)allocate u_cart to match u0 and slice the z4c_cart accessors.

void Z4c::CartoonAlloc() {
  const int ng = pmy_pack->pmesh->mb_indcs.ng;
  Kokkos::realloc(u_cart, u0.extent(0), nz4c, 1 + 2*ng, u0.extent(3), u0.extent(4));
  z4c_cart.alpha.InitWithShallowSlice (u_cart, I_Z4C_ALPHA);
  z4c_cart.beta_u.InitWithShallowSlice(u_cart, I_Z4C_BETAX, I_Z4C_BETAZ);
  z4c_cart.chi.InitWithShallowSlice   (u_cart, I_Z4C_CHI);
  z4c_cart.vKhat.InitWithShallowSlice (u_cart, I_Z4C_KHAT);
  z4c_cart.vTheta.InitWithShallowSlice(u_cart, I_Z4C_THETA);
  z4c_cart.vGam_u.InitWithShallowSlice(u_cart, I_Z4C_GAMX, I_Z4C_GAMZ);
  z4c_cart.g_dd.InitWithShallowSlice  (u_cart, I_Z4C_GXX, I_Z4C_GZZ);
  z4c_cart.vA_dd.InitWithShallowSlice (u_cart, I_Z4C_AXX, I_Z4C_AZZ);
}

//----------------------------------------------------------------------------------------
//! \fn void Z4c::CartoonAxisBC()
//! \brief D-002 axis boundary on a fine array: i < is from i >= is with parity signs, on
//! MeshBlocks whose inner x1 face is the (reflecting) mesh boundary. The production path
//! is the same rule inside MeshBoundaryValues::Z4cBCs; this entry point lets the unit
//! test apply it alone, with a deliberately altered table.

void Z4c::CartoonAxisBC(MeshBlockPack *pmbp, DvceArray5D<Real> u,
                        const CartoonParity &par) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int ng = indcs.ng, is = indcs.is;
  const int n2 = u.extent_int(3), n3 = u.extent_int(2);
  const int nmb = pmbp->nmb_thispack;
  auto &mb_bcs = pmbp->pmb->mb_bcs;
  par_for("cartoon_axis_bc", DevExeSpace(), 0, nmb-1, 0, nz4c-1, 0, n3-1, 0, n2-1,
  KOKKOS_LAMBDA(const int m, const int n, const int k, const int j) {
    if (mb_bcs.d_view(m, BoundaryFace::inner_x1) != BoundaryFlag::reflect) return;
    for (int g = 0; g < ng; ++g) {
      u(m,n,k,j,is-g-1) = par.p[n]*u(m,n,k,j,is+g);
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn void Z4c::CartoonFill()
//! \brief u0 -> u_cart with the production settings.

void Z4c::CartoonFill(MeshBlockPack *pmbp) {
  CartoonFill(pmbp, opt.cartoon_npt, CartoonParityTable(), false);
}

//----------------------------------------------------------------------------------------
//! \fn void Z4c::CartoonFill(pmbp, npt, par, direct_negx)
//! \brief K12 + K3 on the Z4c set. With direct_negx (test only) the x < 0 cells are
//! filled by K12's direct interpolation at rho with c = x/rho < 0 instead of the K3
//! pi-copy.

void Z4c::CartoonFill(MeshBlockPack *pmbp, int npt, const CartoonParity &par,
                      bool direct_negx) {
  const int ng = pmbp->pmesh->mb_indcs.ng;
  if (u_cart.extent(0) != u0.extent(0) || u_cart.extent(3) != u0.extent(3) ||
      u_cart.extent(4) != u0.extent(4) ||
      u_cart.extent_int(2) != 1 + 2*ng) {
    CartoonAlloc();
  }
  // with the algebraic constraints on the ghost planes (D-007), applied inside K12
  CartoonFillSet(pmbp, CartoonZ4cSet(), u0, u_cart, npt, par, direct_negx, true);
}

//----------------------------------------------------------------------------------------
//! \fn void Z4c::CartoonFillADM(MeshBlockPack *pmbp)
//! \brief u_adm (g_ij, K_ij, psi4) -> u_adm_cart with the production stencil (M2a).

void Z4c::CartoonFillADM(MeshBlockPack *pmbp) {
  const int ng = pmbp->pmesh->mb_indcs.ng;
  const CartoonVarSet vs = CartoonADMSet();
  auto &u_adm = pmbp->padm->u_adm;
  if (u_adm.extent_int(1) != vs.nvar) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Cartoon ADM fill expects u_adm with " << vs.nvar << " variables (g, K, "
              << "psi4), found " << u_adm.extent_int(1) << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (u_adm_cart.extent(0) != u_adm.extent(0) ||
      u_adm_cart.extent(3) != u_adm.extent(3) ||
      u_adm_cart.extent(4) != u_adm.extent(4) ||
      u_adm_cart.extent_int(2) != 1 + 2*ng) {
    Kokkos::realloc(u_adm_cart, u_adm.extent(0), vs.nvar, 1 + 2*ng, u_adm.extent(3),
                    u_adm.extent(4));
    adm_cart.psi4.InitWithShallowSlice (u_adm_cart, adm::ADM::I_ADM_PSI4);
    adm_cart.g_dd.InitWithShallowSlice (u_adm_cart, adm::ADM::I_ADM_GXX,
                                        adm::ADM::I_ADM_GZZ);
    adm_cart.vK_dd.InitWithShallowSlice(u_adm_cart, adm::ADM::I_ADM_KXX,
                                        adm::ADM::I_ADM_KZZ);
  }
  CartoonFillSet(pmbp, vs, u_adm, u_adm_cart, opt.cartoon_npt, CartoonParityOf(vs),
                 false, false);
}

//----------------------------------------------------------------------------------------
//! \fn void Z4c::CartoonFillSet(pmbp, vs, src, dst, npt, par, direct_negx)
//! \brief K12 and K3 for the variable set vs: src(m, n, ks, j, i) -> dst(m, n, kc, j, i).

void Z4c::CartoonFillSet(MeshBlockPack *pmbp, const CartoonVarSet &vs,
                         DvceArray5D<Real> src, DvceArray5D<Real> dst, int npt,
                         const CartoonParity &par, bool direct_negx, bool algc) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int ng = indcs.ng, ks = indcs.ks;
  if (npt < 2 || npt > kMaxNpt) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Cartoon npt = " << npt << " outside [2, " << kMaxNpt << "]"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (vs.nvar > kMaxVar) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Cartoon variable set larger than " << kMaxVar << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const int nmb = pmbp->nmb_thispack;
  const int n1 = src.extent_int(4), n2 = src.extent_int(3);
  const int nvar = vs.nvar;
  auto u0_ = src;
  auto uc = dst;
  auto &size = pmbp->pmb->mb_size;
  const RegionIndcs indcs_ = indcs;
  const CartoonMode mode = opt.cartoon;

  if (algc && vs.nsym < 2) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Cartoon fill: algebraic constraints need g-tilde and A-tilde"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const int og = vs.sym[0], oA = vs.sym[1];

  // K12: l = 0 copies the y = 0 plane; l > 0 interpolates at rho, rotates, constrains
  // (algc), and writes planes +l and -l. One interpolation serves both planes.
  par_for("cartoon_K12", DevExeSpace(), 0, nmb-1, 0, ng, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int l, const int j, const int i) {
    if (l == 0) {
      for (int n = 0; n < nvar; ++n) uc(m,n,ng,j,i) = u0_(m,n,ks,j,i);
      return;
    }
    const Real dx = size.d_view(m).dx1;
    Real x, x2p, x3p, x0;
    // mesh (R, z, 0)
    Z4cCartesianPos(mode, size.d_view(m), indcs_, ks, j, i, x, x2p, x3p);
    if (x < 0.0 && !direct_negx) return;
    Z4cCartesianPos(mode, size.d_view(m), indcs_, ks, j, 0, x0, x2p, x3p);
    const Real y = l*dx;
    const Real rho = Kokkos::sqrt(x*x + y*y);
    Real f; int i0raw;
    const int i0 = StencilStart(rho, x0, dx, npt, n1, f, i0raw);
    Real w[kMaxNpt];
    LagrangeWeights(npt, f - i0, w);
    Real T[kMaxVar], G[kMaxVar];
    for (int n = 0; n < nvar; ++n) {
      Real v = 0.0;
      for (int q = 0; q < npt; ++q) v += w[q]*u0_(m,n,ks,j,i0+q);
      T[n] = v;
    }
    const Real c = x/rho, s = y/rho;
    CartoonRotate(vs, T, c, s, G);
    if (algc) AlgConstrRegs(G, og, oA);
    for (int n = 0; n < nvar; ++n) uc(m,n,ng+l,j,i) = G[n];
    CartoonRotate(vs, T, c, -s, G);
    if (algc) AlgConstrRegs(G, og, oA);
    for (int n = 0; n < nvar; ++n) uc(m,n,ng-l,j,i) = G[n];
  });

  // K3: x < 0 cells of planes l != 0 by the pi rotation (x,y) -> (-x,-y) (D-008)
  if (!direct_negx) {
    par_for("cartoon_K3", DevExeSpace(), 0, nmb-1, 0, 2*ng, 0, n2-1, 0, n1-1,
    KOKKOS_LAMBDA(const int m, const int kc, const int j, const int i) {
      if (kc == ng) return;
      const Real dx = size.d_view(m).dx1;
      Real x, x2p, x3p, x0;
      Z4cCartesianPos(mode, size.d_view(m), indcs_, ks, j, i, x, x2p, x3p);
      if (x >= 0.0) return;
      // cell whose midpoint is at -x
      Z4cCartesianPos(mode, size.d_view(m), indcs_, ks, j, 0, x0, x2p, x3p);
      const int im = static_cast<int>(Kokkos::round((-x - x0)/dx));
      for (int n = 0; n < nvar; ++n) {
        uc(m,n,kc,j,i) = par.p[n]*uc(m,n,2*ng-kc,j,im);
      }
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void Z4c::CartoonClampStats()
//! \brief Count K12 stencils that had to be clamped (one-sided), per ghost plane l, on
//! the host. Diagnostic for the unit test only; the production kernels do not count.
//! per_l[l-1] = {l, n_clamped_low, n_clamped_high, n_total} over (m, j, i) with x > 0.

void Z4c::CartoonClampStats(MeshBlockPack *pmbp, int npt,
                            std::vector<std::array<int, 4>> &per_l) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int ng = indcs.ng;
  const int n1 = indcs.nx1 + 2*ng, n2 = indcs.nx2 + 2*ng;
  const int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;
  per_l.assign(ng, {0, 0, 0, 0});
  for (int m = 0; m < nmb; ++m) {
    const Real dx = size.h_view(m).dx1;
    Real x0, x2p, x3p;
    Z4cCartesianPos(opt.cartoon, size.h_view(m), indcs, indcs.ks, indcs.js, 0, x0, x2p,
                    x3p);
    for (int l = 1; l <= ng; ++l) {
      per_l[l-1][0] = l;
      for (int i = 0; i < n1; ++i) {
        Real x;
        Z4cCartesianPos(opt.cartoon, size.h_view(m), indcs, indcs.ks, indcs.js, i, x, x2p,
                        x3p);
        if (x < 0.0) continue;
        const Real rho = std::sqrt(x*x + (l*dx)*(l*dx));
        Real f; int i0raw;
        StencilStart(rho, x0, dx, npt, n1, f, i0raw);
        if (i0raw < 0) per_l[l-1][1] += n2;
        if (i0raw > n1 - npt) per_l[l-1][2] += n2;
        per_l[l-1][3] += n2;
      }
    }
  }
}

}  // namespace z4c
