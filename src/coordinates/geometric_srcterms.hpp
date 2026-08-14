#ifndef COORDINATES_GEOMETRIC_SRCTERMS_HPP_
#define COORDINATES_GEOMETRIC_SRCTERMS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file geometric_srcterms.hpp
//! \brief Newtonian geometric (curvature) source terms for cylindrical/cylindrical_
//! axisym (Task C1) and spherical_polar (Task C2). Ported (math only) from old
//! Athena++'s Cylindrical::AddCoordTermsDivergence / SphericalPolar::
//! AddCoordTermsDivergence, ported onto this project's Delta-A/Delta-V GeomData
//! coefficients (v2 plan Correction C2) instead of the naive-but-wrong 1/x1v form.
//! Dispatched host-side by coord_general (geometric_srcterms.cpp), exactly one kernel
//! launch per call -- no per-cell coordinate branch, per the confinement principle.
//!
//! GR/dynamical-GR are mutually exclusive with this (guarded at the call site in
//! hydro_tasks.cpp/mhd_tasks.cpp: this only ever runs when !is_general_relativistic &&
//! !is_dynamical_relativistic).

#include "athena.hpp"
// mesh/mesh.hpp must be included before meshblock_pack.hpp/eos.hpp in any translation
// unit that hasn't already included it: mesh.hpp <-> meshblock.hpp <-> meshblock_pack.hpp
// <-> coordinates/coordinates.hpp form a header cycle, and eos.hpp includes
// meshblock.hpp directly -- whichever of these is entered FIRST in a TU determines
// whether Mesh::FindMeshBlockIndex() sees a complete MeshBlock/MeshBlockPack (matches
// the working order already used by mesh_geometry.cpp and every hydro/mhd_tasks.cpp).
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "eos/eos.hpp"
#include "mesh_geometry.hpp"

// host-side dispatch entry points (geometric_srcterms.cpp), called from
// Hydro::HydroSrcTerms / MHD::MHDSrcTerms as a sibling of the existing GR CoordSrcTerms
// call; a no-op for coord_general==cartesian.
void AddCoordGeomSrcTermsHydro(MeshBlockPack *pmbp, const DvceArray5D<Real> &w0,
                                const EOS_Data &eos, const Real beta_dt,
                                DvceArray5D<Real> &u0, const DvceFaceFld5D<Real> &uflx);
void AddCoordGeomSrcTermsMHD(MeshBlockPack *pmbp, const DvceArray5D<Real> &w0,
                              const DvceArray5D<Real> &bcc0, const EOS_Data &eos,
                              const Real beta_dt, DvceArray5D<Real> &u0,
                              const DvceFaceFld5D<Real> &uflx);

//----------------------------------------------------------------------------------------
//! \fn AddCylindricalSrcTerms<PhiInIM3,IsMHD>()
//! \brief centrifugal/pressure term added to R-momentum (IM1), and the Ju-thesis
//! angular-momentum-conserving flux-average correction added to the phi-momentum slot.
//! PhiInIM3=false: general cylindrical (R,phi,z), phi resolved as grid x2, in IM2.
//! PhiInIM3=true: cylindrical_axisym (R,z), phi carried as a non-grid rotational
//! component in IM3 (see geometry_cylindrical_axisym.cpp's handedness note). Both the
//! centrifugal term (quadratic in v_phi/B_phi) and the flux-average correction (linear,
//! but self-consistently applied to whatever's actually stored in the phi slot via the
//! flux the generic Riemann solver already computed for it) are correct verbatim with
//! IM2->IM3 substituted, with NO extra sign flip -- verified algebraically in
//! DEVELOPMENT.md's Task C1 log. z-momentum (the OTHER slot) is untouched: z is flat,
//! no curvature source term applies there, matching old Athena++ exactly.
//! IsSR=true (Task G1): SR hydro. AthenaK stores u^i = Gamma*v^i (the spatial 4-velocity
//! components, NOT the 3-velocity) in the w0(IVX/IVY/IVZ) slots for relativistic runs
//! (verified against src/hydro/rsolvers/llf_hyd_singlestate.hpp's SingleStateLLF_SRHyd,
//! which reads wl.vx/vy/vz directly from the same primitive array and documents them as
//! "u^i = Gamma*v^i"). The Newtonian centrifugal term rho*vphi^2 generalizes to the
//! SR momentum-flux form rho*h*u_phi^2 (rho*h = total enthalpy density = d + gamma*e,
//! the exact "wgas" quantity llf_hyd_singlestate.hpp computes for the Riemann flux) --
//! only this replacement is needed; the flux-average correction term is UNCHANGED since
//! it uses whatever momentum flux the (already SR-correct) Riemann solver produced,
//! exactly as for the Newtonian/MHD cases. SR+MHD is not implemented (see
//! geometric_srcterms.cpp's dispatch guard) -- the SR momentum flux's magnetic
//! contribution needs the comoving-frame field strength, not simply bcc0, which is a
//! larger undertaking deferred alongside the rest of curvilinear+MHD+SR.
template <bool PhiInIM3, bool IsMHD, bool IsSR>
inline void AddCylindricalSrcTerms(MeshBlockPack *pmbp, const DvceArray5D<Real> &w0,
                                    const DvceArray5D<Real> &bcc0, const EOS_Data &eos,
                                    const Real beta_dt, const GeomData &geom,
                                    const DvceFaceFld5D<Real> &uflx,
                                    DvceArray5D<Real> &u0) {
  static_assert(!(IsMHD && IsSR), "SR+MHD geometric source terms not implemented");
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie, &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int nmb1 = pmbp->nmb_thispack - 1;
  constexpr int IPHI = PhiInIM3 ? IM3 : IM2;
  constexpr int IBPHI = PhiInIM3 ? IBZ : IBY;
  constexpr int IBOTHER = PhiInIM3 ? IBY : IBZ;  // z-component slot in bcc0
  auto flx1 = uflx.x1f;
  bool is_ideal = eos.is_ideal;
  Real gamma_ = eos.gamma;
  Real iso_cs2 = eos.iso_cs*eos.iso_cs;

  par_for("cyl_geom_src", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real d = w0(m,IDN,k,j,i);
    Real vphi = w0(m,IPHI,k,j,i);
    Real m_pp;
    if constexpr (IsSR) {
      Real e = w0(m,IEN,k,j,i);
      Real pgas = eos.IdealGasPressure(e);
      Real wgas = d + gamma_*e;  // total enthalpy density (rho*h), matches llf_srhyd.hpp
      m_pp = wgas*vphi*vphi + pgas;
    } else {
      // w0(IEN) stores internal-energy DENSITY for the ideal-gas EOS (not pressure --
      // see EOS_Data::IdealGasPressure()/hydro_newdt.cpp's identical usage).
      Real pgas = is_ideal ? eos.IdealGasPressure(w0(m,IEN,k,j,i)) : iso_cs2*d;
      m_pp = d*vphi*vphi + pgas;
    }
    if constexpr (IsMHD) {
      Real bR = bcc0(m,IBX,k,j,i);
      Real bphi = bcc0(m,IBPHI,k,j,i);
      Real bz = bcc0(m,IBOTHER,k,j,i);
      m_pp += 0.5*(bR*bR - bphi*bphi + bz*bz);
    }
    u0(m,IM1,k,j,i) += beta_dt*geom.src1(m,i)*m_pp;

    u0(m,IPHI,k,j,i) -= beta_dt*geom.src2(m,i)*
        (geom.xf1(m,i)*flx1(m,IPHI,k,j,i) + geom.xf1(m,i+1)*flx1(m,IPHI,k,j,i+1));
  });
}

//----------------------------------------------------------------------------------------
//! \fn AddSphericalSrcTerms<IsMHD>()
//! \brief full (r,theta,phi) geometric source terms, ported from SphericalPolar::
//! AddCoordTermsDivergence: radial centrifugal/pressure term (IM1), the flux-average
//! correction for both IM2 and IM3 (theta/phi momentum advected through r-faces), the
//! theta-momentum centrifugal term from phi-rotation (automatically inert for the
//! required 1D-radial layout since src1_j=0 there -- see geometry_spherical.cpp), and
//! the phi-momentum term from theta-rotation, which uses the flux-average form when
//! theta is resolved (multi_d) or the direct local-product form otherwise (matching old
//! Athena++'s use_x2_fluxes branch exactly).
//! IsSR=true (Task G1): see AddCylindricalSrcTerms's doc comment for the u^i-vs-v^i and
//! rho*h-vs-rho rationale; SR+MHD is not implemented (static_assert below).
template <bool IsMHD, bool IsSR>
inline void AddSphericalSrcTerms(MeshBlockPack *pmbp, const DvceArray5D<Real> &w0,
                                  const DvceArray5D<Real> &bcc0, const EOS_Data &eos,
                                  const Real beta_dt, const GeomData &geom,
                                  const DvceFaceFld5D<Real> &uflx,
                                  DvceArray5D<Real> &u0) {
  static_assert(!(IsMHD && IsSR), "SR+MHD geometric source terms not implemented");
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie, &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int nmb1 = pmbp->nmb_thispack - 1;
  auto flx1 = uflx.x1f;
  auto flx2 = uflx.x2f;
  bool is_ideal = eos.is_ideal;
  Real gamma_ = eos.gamma;
  Real iso_cs2 = eos.iso_cs*eos.iso_cs;
  bool use_x2_fluxes = pmbp->pmesh->multi_d;

  par_for("sph_geom_src", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real d = w0(m,IDN,k,j,i);
    Real vth = w0(m,IM2,k,j,i);
    Real vph = w0(m,IM3,k,j,i);
    Real mass_or_wgas = d;
    Real pgas;
    if constexpr (IsSR) {
      Real e = w0(m,IEN,k,j,i);
      pgas = eos.IdealGasPressure(e);
      mass_or_wgas = d + gamma_*e;  // total enthalpy density (rho*h)
    } else {
      // w0(IEN) stores internal-energy DENSITY for the ideal-gas EOS (not pressure);
      // must convert via IdealGasPressure(), matching hydro_newdt.cpp's identical usage.
      pgas = is_ideal ? eos.IdealGasPressure(w0(m,IEN,k,j,i)) : iso_cs2*d;
    }

    // src_1 = < M_thth + M_phph > <1/r>
    Real m_ii = mass_or_wgas*(vth*vth + vph*vph) + 2.0*pgas;
    if constexpr (IsMHD) { m_ii += SQR(bcc0(m,IBX,k,j,i)); }
    u0(m,IM1,k,j,i) += beta_dt*geom.src1(m,i)*m_ii;

    // src_2, src_3 = -< M_thr >, -< M_phr > <1/r>  (r-flux-average correction)
    u0(m,IM2,k,j,i) -= beta_dt*geom.src2(m,i)*
        (geom.a1i(m,i)*flx1(m,IM2,k,j,i) + geom.a1i(m,i+1)*flx1(m,IM2,k,j,i+1));
    u0(m,IM3,k,j,i) -= beta_dt*geom.src2(m,i)*
        (geom.a1i(m,i)*flx1(m,IM3,k,j,i) + geom.a1i(m,i+1)*flx1(m,IM3,k,j,i+1));

    // src_2 = < M_phph > <cot theta / r>  (automatically inert when src1_j=0, i.e. the
    // required 1D-radial layout with x2min=0,x2max=pi -- see geometry_spherical.cpp)
    Real m_pp = mass_or_wgas*vph*vph + pgas;
    if constexpr (IsMHD) {
      Real br = bcc0(m,IBX,k,j,i), bth = bcc0(m,IBY,k,j,i), bph = bcc0(m,IBZ,k,j,i);
      m_pp += 0.5*(br*br + bth*bth - bph*bph);
    }
    u0(m,IM2,k,j,i) += beta_dt*geom.src1(m,i)*geom.src1_j(m,j)*m_pp;

    // src_3 = -< M_phth > <cot theta / r>
    if (use_x2_fluxes) {
      u0(m,IM3,k,j,i) -= beta_dt*geom.src1(m,i)*geom.src2_j(m,j)*
          (geom.a2j(m,j)*flx2(m,IM3,k,j,i) + geom.a2j(m,j+1)*flx2(m,IM3,k,j+1,i));
    } else {
      Real m_ph = mass_or_wgas*vph*vth;
      if constexpr (IsMHD) { m_ph -= bcc0(m,IBY,k,j,i)*bcc0(m,IBZ,k,j,i); }
      u0(m,IM3,k,j,i) -= beta_dt*geom.src1(m,i)*geom.src2_j(m,j)*m_ph;
    }
  });
}

#endif  // COORDINATES_GEOMETRIC_SRCTERMS_HPP_
