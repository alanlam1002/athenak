//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ideal_grmhd.cpp
//! \brief derived class that implements ideal gas EOS in general relativistic mhd

#include <float.h>

#include <cstdio>

#include "athena.hpp"
#include "mhd/mhd.hpp"
#include "eos.hpp"
#include "eos/ideal_c2p_mhd.hpp"

#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor

IdealGRMHD::IdealGRMHD(MeshBlockPack *pp, ParameterInput *pin) :
    EquationOfState("mhd", pp, pin) {
  eos_data.is_ideal = true;
  eos_data.gamma = pin->GetReal("mhd","gamma");
  eos_data.iso_cs = 0.0;
  eos_data.gamma_max = pin->GetOrAddReal("mhd","gamma_max",(FLT_MAX));  // gamma ceiling
  eos_data.sigma_max = pin->GetOrAddReal("mhd","sigma_max",(FLT_MAX));  // sigma ceiling
  // bhstar diagnostic (numerics-013 D-a); -1 = off
  c2p_dump_cycle_ = pin->GetOrAddInteger("mhd","c2p_dump_cycle",-1);
  if (c2p_dump_cycle_ >= 0) {
    Kokkos::realloc(c2p_dump_, kDumpRec*kDumpNF);
    Kokkos::realloc(c2p_dump_n_, 1);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ConsToPrim()
//! \brief Converts conserved into primitive variables.
//! Operates over range of cells given in argument list.

void IdealGRMHD::ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                            DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                            const bool only_testfloors,
                            const int il, const int iu, const int jl, const int ju,
                            const int kl, const int ku) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &js = indcs.js, &ks = indcs.ks;
  auto &size = pmy_pack->pmb->mb_size;
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  auto &fofc_ = pmy_pack->pmhd->fofc;
  auto eos = eos_data;
  Real gm1 = eos_data.gamma - 1.0;

  auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmy_pack->pcoord->coord_data.bh_spin;
  auto &use_excise = pmy_pack->pcoord->coord_data.bh_excise;
  auto &excision_floor_ = pmy_pack->pcoord->excision_floor;
  auto &excision_flux_ = pmy_pack->pcoord->excision_flux;
  auto &dexcise_ = pmy_pack->pcoord->coord_data.dexcise;
  auto &pexcise_ = pmy_pack->pcoord->coord_data.pexcise;

  // bhstar diagnostic (numerics-013 D-a): record the first density-floor / velocity-ceiling /
  // failure events at or after cycle c2p_dump_cycle_ (energy-floor-only events are not
  // recorded: in the cold free fall they occur in most cells every call)
  const int ncyc = pmy_pack->pmesh->ncycle;
  if (ncyc != c2p_lastcycle_) {c2p_ncall_ = 0; c2p_lastcycle_ = ncyc;}
  c2p_ncall_++;
  const bool do_dump = (!only_testfloors) && (c2p_dump_cycle_ >= 0) && (!c2p_dumped_)
                       && (ncyc >= c2p_dump_cycle_);
  if (do_dump) Kokkos::deep_copy(c2p_dump_n_, 0);
  auto dump_ = c2p_dump_;
  auto dump_n_ = c2p_dump_n_;
  auto &gid_ = pmy_pack->pmb->mb_gid;
  auto &lev_ = pmy_pack->pmb->mb_lev;
  const int ie_ = indcs.ie, je_ = indcs.je, ke_ = indcs.ke;
  const int ncall_ = c2p_ncall_;

  const int ni   = (iu - il + 1);
  const int nji  = (ju - jl + 1)*ni;
  const int nkji = (ku - kl + 1)*nji;
  const int nmkji = nmb*nkji;

  int nfloord_=0, nfloore_=0, nceilv_=0, nfail_=0, maxit_=0;
  Kokkos::parallel_reduce("grmhd_c2p",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, int &sumd, int &sume, int &sumv, int &sumf, int &max_it) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/ni;
    int i = (idx - m*nkji - k*nji - j*ni) + il;
    j += jl;
    k += kl;

    // load single state conserved variables
    MHDCons1D u;
    u.d  = cons(m,IDN,k,j,i);
    u.mx = cons(m,IM1,k,j,i);
    u.my = cons(m,IM2,k,j,i);
    u.mz = cons(m,IM3,k,j,i);
    u.e  = cons(m,IEN,k,j,i);

    // load cell-centered fields into conserved state
    // use input CC fields if only testing floors with FOFC
    if (only_testfloors) {
      u.bx = bcc(m,IBX,k,j,i);
      u.by = bcc(m,IBY,k,j,i);
      u.bz = bcc(m,IBZ,k,j,i);
    // else use simple linear average of face-centered fields
    } else {
      u.bx = 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));
      u.by = 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i));
      u.bz = 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i));
    }

    // Extract components of metric
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);

    HydPrim1D w;
    bool dfloor_used=false, efloor_used=false;
    bool vceiling_used=false, c2p_failure=false;
    int iter_used=0;

    // Only execute cons2prim if outside excised region
    bool excised = false;
    if (use_excise) {
      if (excision_floor_(m,k,j,i)) {
        w.d = dexcise_;
        w.vx = 0.0;
        w.vy = 0.0;
        w.vz = 0.0;
        w.e = pexcise_/gm1;
        excised = true;
      }
      if (only_testfloors) {
        if (excision_flux_(m,k,j,i)) {
          excised = true;
        }
      }
    }

    if (!(excised)) {
      // calculate SR conserved quantities
      MHDCons1D u_sr;
      Real s2, b2, rpar;
      TransformToSRMHD(u,glower,gupper,s2,b2,rpar,u_sr);

      // call c2p function
      // (inline function in ideal_c2p_mhd.hpp file)
      SingleC2P_IdealSRMHD(u_sr, eos, s2, b2, rpar, w,
                           dfloor_used, efloor_used, c2p_failure, iter_used);

      if (do_dump && (dfloor_used || c2p_failure)) {
        int slot = Kokkos::atomic_fetch_add(&dump_n_(0), 1);
        if (slot < kDumpRec) {
          Real *r = &dump_(slot*kDumpNF);
          r[0] = ncyc; r[1] = ncall_; r[2] = gid_.d_view(m); r[3] = lev_.d_view(m);
          r[4] = i; r[5] = j; r[6] = k;
          r[7] = (i < is || i > ie_ || j < js || j > je_ || k < ks || k > ke_) ? 1.0 : 0.0;
          r[8] = x1v; r[9] = x2v; r[10] = x3v;
          r[11] = u.d; r[12] = u.mx; r[13] = u.my; r[14] = u.mz; r[15] = u.e;  // GR cons
          r[16] = u.bx; r[17] = u.by; r[18] = u.bz;
          r[19] = u_sr.d; r[20] = u_sr.e; r[21] = s2; r[22] = b2; r[23] = rpar;
          r[24] = iter_used; r[25] = dfloor_used; r[26] = efloor_used; r[27] = c2p_failure;
          r[28] = prim(m,IDN,k,j,i); r[29] = prim(m,IVX,k,j,i); r[30] = prim(m,IVY,k,j,i);
          r[31] = prim(m,IVZ,k,j,i); r[32] = prim(m,IEN,k,j,i);   // previous primitives
          r[33] = w.d; r[34] = w.vx; r[35] = w.vy; r[36] = w.vz; r[37] = w.e;  // result
          r[38] = 0.0; r[39] = 0.0;
        }
      }

      // apply velocity ceiling if necessary
      Real tmp = glower[1][1]*SQR(w.vx)
               + glower[2][2]*SQR(w.vy)
               + glower[3][3]*SQR(w.vz)
               + 2.0*glower[1][2]*w.vx*w.vy + 2.0*glower[1][3]*w.vx*w.vz
               + 2.0*glower[2][3]*w.vy*w.vz;
      Real lor = sqrt(1.0+tmp);
      if (lor > eos.gamma_max) {
        vceiling_used = true;
        Real factor = sqrt((SQR(eos.gamma_max)-1.0)/(SQR(lor)-1.0));
        w.vx *= factor;
        w.vy *= factor;
        w.vz *= factor;
      }
    }

    // set FOFC flag and quit loop if this function called only to check floors
    if (only_testfloors) {
      if (dfloor_used || efloor_used || vceiling_used || c2p_failure) {
        fofc_(m,k,j,i) = true;
        sumd++;  // use dfloor as counter for when either is true
      }
    } else {
      if (dfloor_used) {sumd++;}
      if (efloor_used) {sume++;}
      if (vceiling_used) {sumv++;}
      if (c2p_failure) {sumf++;}
      max_it = (iter_used > max_it) ? iter_used : max_it;

      // store primitive state in 3D array
      prim(m,IDN,k,j,i) = w.d;
      prim(m,IVX,k,j,i) = w.vx;
      prim(m,IVY,k,j,i) = w.vy;
      prim(m,IVZ,k,j,i) = w.vz;
      prim(m,IEN,k,j,i) = w.e;

      // store cell-centered fields in 3D array
      bcc(m,IBX,k,j,i) = u.bx;
      bcc(m,IBY,k,j,i) = u.by;
      bcc(m,IBZ,k,j,i) = u.bz;

      // reset conserved variables if floor, ceiling, failure, or excision encountered
      if (dfloor_used || efloor_used || vceiling_used || c2p_failure || excised) {
        MHDPrim1D w_in;
        w_in.d  = w.d;
        w_in.vx = w.vx;
        w_in.vy = w.vy;
        w_in.vz = w.vz;
        w_in.e  = w.e;
        w_in.bx = u.bx;
        w_in.by = u.by;
        w_in.bz = u.bz;

        HydCons1D u_out;
        SingleP2C_IdealGRMHD(glower, gupper, w_in, eos.gamma, u_out);
        cons(m,IDN,k,j,i) = u_out.d;
        cons(m,IM1,k,j,i) = u_out.mx;
        cons(m,IM2,k,j,i) = u_out.my;
        cons(m,IM3,k,j,i) = u_out.mz;
        cons(m,IEN,k,j,i) = u_out.e;
        u.d = u_out.d;  // (needed if there are scalars below)
      }

      // convert scalars (if any)
      for (int n=nmhd; n<(nmhd+nscal); ++n) {
        prim(m,n,k,j,i) = cons(m,n,k,j,i)/u.d;
      }
    }
  }, Kokkos::Sum<int>(nfloord_), Kokkos::Sum<int>(nfloore_), Kokkos::Sum<int>(nceilv_),
     Kokkos::Sum<int>(nfail_), Kokkos::Max<int>(maxit_));

  if (do_dump) {
    auto hn = Kokkos::create_mirror_view_and_copy(HostMemSpace(), c2p_dump_n_);
    if (hn(0) > 0) {
      PrintC2PDump(hn(0), il, iu, jl, ju, kl, ku);
      c2p_dumped_ = true;
    }
  }

  // store appropriate counters
  if (only_testfloors) {
    pmy_pack->pmesh->ecounter.nfofc += nfloord_;
  } else {
    pmy_pack->pmesh->ecounter.neos_dfloor += nfloord_;
    pmy_pack->pmesh->ecounter.neos_efloor += nfloore_;
    pmy_pack->pmesh->ecounter.neos_vceil  += nceilv_;
    pmy_pack->pmesh->ecounter.neos_fail   += nfail_;
    pmy_pack->pmesh->ecounter.maxit_c2p = maxit_;
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void PrimToCons()
//! \brief Converts primitive into conserved variables.  Operates over range of cells
//! given in argument list.

void IdealGRMHD::PrimToCons(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc,
                            DvceArray5D<Real> &cons, const int il, const int iu,
                            const int jl, const int ju, const int kl, const int ku) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &js = indcs.js, &ks = indcs.ks;
  auto &size = pmy_pack->pmb->mb_size;
  auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmy_pack->pcoord->coord_data.bh_spin;
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  Real &gamma = eos_data.gamma;

  par_for("grmhd_p2c", DevExeSpace(), 0, (nmb-1), kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // Extract components of metric
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);

    // Load single state of primitive variables
    MHDPrim1D w;
    w.d  = prim(m,IDN,k,j,i);
    w.vx = prim(m,IVX,k,j,i);
    w.vy = prim(m,IVY,k,j,i);
    w.vz = prim(m,IVZ,k,j,i);
    w.e  = prim(m,IEN,k,j,i);

    // load cell-centered fields into primitive state
    w.bx = bcc(m,IBX,k,j,i);
    w.by = bcc(m,IBY,k,j,i);
    w.bz = bcc(m,IBZ,k,j,i);

    // call p2c function
    HydCons1D u;
    SingleP2C_IdealGRMHD(glower, gupper, w, gamma, u);

    // store conserved quantities in 3D array
    cons(m,IDN,k,j,i) = u.d;
    cons(m,IM1,k,j,i) = u.mx;
    cons(m,IM2,k,j,i) = u.my;
    cons(m,IM3,k,j,i) = u.mz;
    cons(m,IEN,k,j,i) = u.e;

    // convert scalars (if any)
    for (int n=nmhd; n<(nmhd+nscal); ++n) {
      cons(m,n,k,j,i) = u.d*prim(m,n,k,j,i);
    }
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void IdealGRMHD::PrintC2PDump()
//! \brief bhstar diagnostic (numerics-013 D-a): print the recorded c2p events of this rank

void IdealGRMHD::PrintC2PDump(int nrec, int il, int iu, int jl, int ju, int kl, int ku) {
  auto h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), c2p_dump_);
  int n = (nrec < kDumpRec) ? nrec : kDumpRec;
  std::printf("C2PDUMP rank %d: %d events (%d recorded) in call %d of cycle %d, "
              "range i %d-%d j %d-%d k %d-%d\n", global_variable::my_rank, nrec, n,
              c2p_ncall_, pmy_pack->pmesh->ncycle, il, iu, jl, ju, kl, ku);
  std::printf("C2PDUMP cols: cycle call gid lev i j k ghost x y z D Sx Sy Sz tau+D Bx By Bz "
              "Dsr Esr s2 b2 rpar iter dfl efl fail rho0 vx0 vy0 vz0 e0 rho vx vy vz e\n");
  for (int q = 0; q < n; ++q) {
    std::printf("C2PDUMP %d", global_variable::my_rank);
    for (int f = 0; f < 38; ++f) std::printf(" %.17g", h(q*kDumpNF + f));
    std::printf("\n");
  }
  std::fflush(stdout);
}
