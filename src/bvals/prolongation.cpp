//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file prolongation.cpp
//! \brief functions to prolongate data at boundaries for cell-centered and face-centered
//! variables. Functions are members of MeshBoundaryValuesCC or MeshBoundaryValuesFC
//! classes.

#include <cstdlib>
#include <iostream>
#include <iomanip>    // std::setprecision()

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/nghbr_index.hpp"
#include "bvals.hpp"
#include "mesh/prolongation.hpp" // implements prolongation operators
#include "mesh/restriction.hpp" // implements restriction operators

#include "coordinates/cell_locations.hpp"
#include "coordinates/mesh_geometry.hpp"

namespace {

KOKKOS_INLINE_FUNCTION
void NeighborOffsetFromIndex(const int n, int &ox1, int &ox2, int &ox3) {
  ox1 = 0;
  ox2 = 0;
  ox3 = 0;
  for (int iz = -1; iz <= 1; ++iz) {
    for (int iy = -1; iy <= 1; ++iy) {
      for (int ix = -1; ix <= 1; ++ix) {
        if ((ix == 0) && (iy == 0) && (iz == 0)) continue;
        for (int n1 = 0; n1 <= 1; ++n1) {
          for (int n2 = 0; n2 <= 1; ++n2) {
            if (NeighborIndex(ix, iy, iz, n1, n2) == n) {
              ox1 = ix;
              ox2 = iy;
              ox3 = iz;
              return;
            }
          }
        }
      }
    }
  }
}

KOKKOS_INLINE_FUNCTION
int MaxNeighborLevelAtOffset(const int m, const int nnghbr, const int ox1,
                             const int ox2, const int ox3,
                             const DualArray2D<NeighborBlock> &nghbr) {
  int max_lev = -1;
  for (int n1 = 0; n1 <= 1; ++n1) {
    for (int n2 = 0; n2 <= 1; ++n2) {
      const int idx = NeighborIndex(ox1, ox2, ox3, n1, n2);
      if ((idx >= 0) && (idx < nnghbr) && (nghbr.d_view(m, idx).gid >= 0)) {
        max_lev =
            (nghbr.d_view(m, idx).lev > max_lev) ? nghbr.d_view(m, idx).lev : max_lev;
      }
    }
  }
  return max_lev;
}

KOKKOS_INLINE_FUNCTION
bool IsActiveFCFace(const int v, const int k, const int j, const int i,
                    const RegionIndcs &indcs) {
  if (v == 0) {
    return (i >= indcs.is) && (i <= indcs.ie + 1) &&
           (j >= indcs.js) && (j <= indcs.je) &&
           (k >= indcs.ks) && (k <= indcs.ke);
  } else if (v == 1) {
    return (i >= indcs.is) && (i <= indcs.ie) &&
           (j >= indcs.js) && (j <= indcs.je + 1) &&
           (k >= indcs.ks) && (k <= indcs.ke);
  } else {
    return (i >= indcs.is) && (i <= indcs.ie) &&
           (j >= indcs.js) && (j <= indcs.je) &&
           (k >= indcs.ks) && (k <= indcs.ke + 1);
  }
}

KOKKOS_INLINE_FUNCTION
bool CanProlongateFCFace(const int m, const int nnghbr, const int v,
                         const int k, const int j, const int i,
                         const int ox1, const int ox2, const int ox3,
                         const int my_lev, const RegionIndcs &indcs,
                         const DualArray2D<NeighborBlock> &nghbr) {
  if (!IsActiveFCFace(v, k, j, i, indcs)) {
    return true;
  }

  // Coarse-neighbor prolongation may write active faces only at the physical
  // fine/coarse interface normal to the face component. Active interior faces and active
  // boundary faces owned by same-level or finer neighbors are left untouched.
  if (v == 0) {
    int normal_ox = 0;
    if (i == indcs.is) {
      normal_ox = -1;
    } else if (i == indcs.ie + 1) {
      normal_ox = 1;
    } else {
      return false;
    }
    return (ox1 == normal_ox) && (ox2 == 0) && (ox3 == 0) &&
           (MaxNeighborLevelAtOffset(m, nnghbr, normal_ox, 0, 0, nghbr) < my_lev);
  } else if (v == 1) {
    int normal_ox = 0;
    if (j == indcs.js) {
      normal_ox = -1;
    } else if (j == indcs.je + 1) {
      normal_ox = 1;
    } else {
      return false;
    }
    return (ox1 == 0) && (ox2 == normal_ox) && (ox3 == 0) &&
           (MaxNeighborLevelAtOffset(m, nnghbr, 0, normal_ox, 0, nghbr) < my_lev);
  } else {
    int normal_ox = 0;
    if (k == indcs.ks) {
      normal_ox = -1;
    } else if (k == indcs.ke + 1) {
      normal_ox = 1;
    } else {
      return false;
    }
    return (ox1 == 0) && (ox2 == 0) && (ox3 == normal_ox) &&
           (MaxNeighborLevelAtOffset(m, nnghbr, 0, 0, normal_ox, nghbr) < my_lev);
  }
}

KOKKOS_INLINE_FUNCTION
void StoreProlongatedFCFace(const int m, const int nnghbr, const int v,
                            const int k, const int j, const int i,
                            const int ox1, const int ox2, const int ox3,
                            const int my_lev, const RegionIndcs &indcs,
                            const DualArray2D<NeighborBlock> &nghbr,
                            const Real value, const DvceArray4D<Real> &bf) {
  if (CanProlongateFCFace(m, nnghbr, v, k, j, i, ox1, ox2, ox3,
                          my_lev, indcs, nghbr)) {
    bf(m,k,j,i) = value;
  }
}

KOKKOS_INLINE_FUNCTION
void ProlongFCSharedX1FaceOwned(const int m, const int nnghbr,
                                const int k, const int j, const int i,
                                const int fk, const int fj, const int fi,
                                const int ox1, const int ox2, const int ox3,
                                const int my_lev, const bool multi_d,
                                const bool three_d, const RegionIndcs &indcs,
                                const DualArray2D<NeighborBlock> &nghbr,
                                const DvceArray2D<Real> &ct1,
                                const DvceArray2D<Real> &ct2,
                                const DvceArray2D<Real> &t1,
                                const DvceArray2D<Real> &t2,
                                const bool uniform,
                                const DvceArray4D<Real> &cbx1f,
                                const DvceArray4D<Real> &bx1f) {
  // increments shared with the regrid-path kernel in mesh/prolongation.hpp -- see
  // FCSharedIncrements() there for the derivation and for why the two must not diverge
  Real d2m = 0.0, d2p = 0.0;
  if (multi_d) {
    FCSharedIncrements(cbx1f(m,k,j-1,i), cbx1f(m,k,j,i), cbx1f(m,k,j+1,i),
                       ct1, t1, m, j, fj, uniform, d2m, d2p);
  }
  Real d3m = 0.0, d3p = 0.0;
  if (three_d) {
    FCSharedIncrements(cbx1f(m,k-1,j,i), cbx1f(m,k,j,i), cbx1f(m,k+1,j,i),
                       ct2, t2, m, k, fk, uniform, d3m, d3p);
  }

  StoreProlongatedFCFace(m, nnghbr, 0, fk, fj, fi, ox1, ox2, ox3,
                         my_lev, indcs, nghbr, cbx1f(m,k,j,i) + d2m + d3m, bx1f);
  if (multi_d) {
    StoreProlongatedFCFace(m, nnghbr, 0, fk, fj+1, fi, ox1, ox2, ox3,
                           my_lev, indcs, nghbr, cbx1f(m,k,j,i) + d2p + d3m, bx1f);
  }
  if (three_d) {
    StoreProlongatedFCFace(m, nnghbr, 0, fk+1, fj, fi, ox1, ox2, ox3,
                           my_lev, indcs, nghbr, cbx1f(m,k,j,i) + d2m + d3p, bx1f);
    StoreProlongatedFCFace(m, nnghbr, 0, fk+1, fj+1, fi, ox1, ox2, ox3,
                           my_lev, indcs, nghbr, cbx1f(m,k,j,i) + d2p + d3p, bx1f);
  }
}

KOKKOS_INLINE_FUNCTION
void ProlongFCSharedX2FaceOwned(const int m, const int nnghbr,
                                const int k, const int j, const int i,
                                const int fk, const int fj, const int fi,
                                const int ox1, const int ox2, const int ox3,
                                const int my_lev, const bool three_d,
                                const RegionIndcs &indcs,
                                const DualArray2D<NeighborBlock> &nghbr,
                                const DvceArray2D<Real> &ct1,
                                const DvceArray2D<Real> &ct2,
                                const DvceArray2D<Real> &t1,
                                const DvceArray2D<Real> &t2,
                                const bool uniform,
                                const DvceArray4D<Real> &cbx2f,
                                const DvceArray4D<Real> &bx2f) {
  Real d1m, d1p;
  FCSharedIncrements(cbx2f(m,k,j,i-1), cbx2f(m,k,j,i), cbx2f(m,k,j,i+1),
                     ct1, t1, m, i, fi, uniform, d1m, d1p);
  Real d3m = 0.0, d3p = 0.0;
  if (three_d) {
    FCSharedIncrements(cbx2f(m,k-1,j,i), cbx2f(m,k,j,i), cbx2f(m,k+1,j,i),
                       ct2, t2, m, k, fk, uniform, d3m, d3p);
  }

  StoreProlongatedFCFace(m, nnghbr, 1, fk, fj, fi, ox1, ox2, ox3,
                         my_lev, indcs, nghbr, cbx2f(m,k,j,i) + d1m + d3m, bx2f);
  StoreProlongatedFCFace(m, nnghbr, 1, fk, fj, fi+1, ox1, ox2, ox3,
                         my_lev, indcs, nghbr, cbx2f(m,k,j,i) + d1p + d3m, bx2f);
  if (three_d) {
    StoreProlongatedFCFace(m, nnghbr, 1, fk+1, fj, fi, ox1, ox2, ox3,
                           my_lev, indcs, nghbr, cbx2f(m,k,j,i) + d1m + d3p, bx2f);
    StoreProlongatedFCFace(m, nnghbr, 1, fk+1, fj, fi+1, ox1, ox2, ox3,
                           my_lev, indcs, nghbr, cbx2f(m,k,j,i) + d1p + d3p, bx2f);
  }
}

KOKKOS_INLINE_FUNCTION
void ProlongFCSharedX3FaceOwned(const int m, const int nnghbr,
                                const int k, const int j, const int i,
                                const int fk, const int fj, const int fi,
                                const int ox1, const int ox2, const int ox3,
                                const int my_lev, const bool multi_d,
                                const RegionIndcs &indcs,
                                const DualArray2D<NeighborBlock> &nghbr,
                                const DvceArray2D<Real> &ct1,
                                const DvceArray2D<Real> &ct2,
                                const DvceArray2D<Real> &t1,
                                const DvceArray2D<Real> &t2,
                                const bool uniform,
                                const DvceArray4D<Real> &cbx3f,
                                const DvceArray4D<Real> &bx3f) {
  Real d1m, d1p;
  FCSharedIncrements(cbx3f(m,k,j,i-1), cbx3f(m,k,j,i), cbx3f(m,k,j,i+1),
                     ct1, t1, m, i, fi, uniform, d1m, d1p);
  Real d2m = 0.0, d2p = 0.0;
  if (multi_d) {
    FCSharedIncrements(cbx3f(m,k,j-1,i), cbx3f(m,k,j,i), cbx3f(m,k,j+1,i),
                       ct2, t2, m, j, fj, uniform, d2m, d2p);
  }

  StoreProlongatedFCFace(m, nnghbr, 2, fk, fj, fi, ox1, ox2, ox3,
                         my_lev, indcs, nghbr, cbx3f(m,k,j,i) + d1m + d2m, bx3f);
  StoreProlongatedFCFace(m, nnghbr, 2, fk, fj, fi+1, ox1, ox2, ox3,
                         my_lev, indcs, nghbr, cbx3f(m,k,j,i) + d1p + d2m, bx3f);
  if (multi_d) {
    StoreProlongatedFCFace(m, nnghbr, 2, fk, fj+1, fi, ox1, ox2, ox3,
                           my_lev, indcs, nghbr, cbx3f(m,k,j,i) + d1m + d2p, bx3f);
    StoreProlongatedFCFace(m, nnghbr, 2, fk, fj+1, fi+1, ox1, ox2, ox3,
                           my_lev, indcs, nghbr, cbx3f(m,k,j,i) + d1p + d2p, bx3f);
  }
}

KOKKOS_INLINE_FUNCTION
void ProlongFCInternalOwned(const int m, const int nnghbr, const int fk, const int fj,
                            const int fi, const int ox1, const int ox2, const int ox3,
                            const int my_lev, const bool three_d,
                            const RegionIndcs &indcs,
                            const DualArray2D<NeighborBlock> &nghbr,
                            const GeomData &geom, const bool cubic,
                            const DvceFaceFld4D<Real> &b) {
  // Boundary-path twin of ProlongFCInternal: identical values, but each store is routed
  // through the write-ownership check. The Toth-Roe arithmetic itself now lives in ONE
  // place (ProlongFCInternalValues, mesh/prolongation.hpp) rather than being duplicated
  // here, which is how the regrid and boundary paths previously came to risk diverging.
  Real v1[4], v2[4], v3[4];
  ProlongFCInternalValues(m, fk, fj, fi, three_d, geom, cubic, b, v1, v2, v3);

  StoreProlongatedFCFace(m, nnghbr, 0, fk, fj  , fi+1, ox1, ox2, ox3, my_lev,
                         indcs, nghbr, v1[0], b.x1f);
  StoreProlongatedFCFace(m, nnghbr, 0, fk, fj+1, fi+1, ox1, ox2, ox3, my_lev,
                         indcs, nghbr, v1[1], b.x1f);
  StoreProlongatedFCFace(m, nnghbr, 1, fk, fj+1, fi  , ox1, ox2, ox3, my_lev,
                         indcs, nghbr, v2[0], b.x2f);
  StoreProlongatedFCFace(m, nnghbr, 1, fk, fj+1, fi+1, ox1, ox2, ox3, my_lev,
                         indcs, nghbr, v2[1], b.x2f);
  if (three_d) {
    StoreProlongatedFCFace(m, nnghbr, 0, fk+1, fj  , fi+1, ox1, ox2, ox3, my_lev,
                           indcs, nghbr, v1[2], b.x1f);
    StoreProlongatedFCFace(m, nnghbr, 0, fk+1, fj+1, fi+1, ox1, ox2, ox3, my_lev,
                           indcs, nghbr, v1[3], b.x1f);
    StoreProlongatedFCFace(m, nnghbr, 1, fk+1, fj+1, fi  , ox1, ox2, ox3, my_lev,
                           indcs, nghbr, v2[2], b.x2f);
    StoreProlongatedFCFace(m, nnghbr, 1, fk+1, fj+1, fi+1, ox1, ox2, ox3, my_lev,
                           indcs, nghbr, v2[3], b.x2f);
    StoreProlongatedFCFace(m, nnghbr, 2, fk+1, fj  , fi  , ox1, ox2, ox3, my_lev,
                           indcs, nghbr, v3[0], b.x3f);
    StoreProlongatedFCFace(m, nnghbr, 2, fk+1, fj  , fi+1, ox1, ox2, ox3, my_lev,
                           indcs, nghbr, v3[1], b.x3f);
    StoreProlongatedFCFace(m, nnghbr, 2, fk+1, fj+1, fi  , ox1, ox2, ox3, my_lev,
                           indcs, nghbr, v3[2], b.x3f);
    StoreProlongatedFCFace(m, nnghbr, 2, fk+1, fj+1, fi+1, ox1, ox2, ox3, my_lev,
                           indcs, nghbr, v3[3], b.x3f);
  }
}

} // namespace
//----------------------------------------------------------------------------------------
//! \fn void FillCoarseInBndryCC()
//! \brief To ensure that the coarse array is up-to-date in all neighboring cells touched
//! by the prolongation interpolation stencil, data is restricted to coarse array in
//! boundaries between MeshBlocks at the same level.

void MeshBoundaryValuesCC::FillCoarseInBndryCC(DvceArray5D<Real> &a,
                                               DvceArray5D<Real> &ca,
                                               bool is_z4c) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  //bool not_z4c = (pmbp->pz4c == nullptr)? true : false;

  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR
  int nmnv = nmb*nnghbr*nvar;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &rbuf = recvbuf;
  auto &indcs  = pmy_pack->pmesh->mb_indcs;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  auto &nx1 = pmy_pack->pmesh->mb_indcs.nx1;
  auto &nx2 = pmy_pack->pmesh->mb_indcs.nx2;
  auto &nx3 = pmy_pack->pmesh->mb_indcs.nx3;
  auto& restrict_2nd = pmy_pack->pmesh->pmr->weights.restrict_2nd;
  auto& restrict_4th = pmy_pack->pmesh->pmr->weights.restrict_4th;
  auto& restrict_4th_edge = pmy_pack->pmesh->pmr->weights.restrict_4th_edge;
  auto &geom = pmy_pack->pgeom->geom_data;
  const bool uni = geom.cells_uniform;

  // Restrict data into coarse array in any boundary filled with data from the same
  // level.  This ensures data in the coarse array at corners where one direction is a
  // coarser level and the other the same level is filled properly.
  // (Only needed in multidimensions)

  if (multi_d) {
    auto &cis = indcs.cis;
    auto &cjs = indcs.cjs;
    auto &cks = indcs.cks;
    // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
    Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
    Kokkos::parallel_for("ProlCCSame", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
      const int m = (tmember.league_rank())/(nnghbr*nvar);
      const int n = (tmember.league_rank() - m*(nnghbr*nvar))/nvar;
      const int v = (tmember.league_rank() - m*(nnghbr*nvar) - n*nvar);

      // only restrict when neighbor exists and is at SAME level
      if ((nghbr.d_view(m,n).gid >= 0) && (nghbr.d_view(m,n).lev == mblev.d_view(m))) {
        // loop over indices for receives at same level, but convert loop limits to
        // coarse array
        int il = (rbuf[n].isame[0].bis + cis)/2;
        int iu = (rbuf[n].isame[0].bie + cis)/2;
        int jl = (rbuf[n].isame[0].bjs + cjs)/2;
        int ju = (rbuf[n].isame[0].bje + cjs)/2;
        int kl = (rbuf[n].isame[0].bks + cks)/2;
        int ku = (rbuf[n].isame[0].bke + cks)/2;

        const int ni = iu - il + 1;
        const int nj = ju - jl + 1;
        const int nk = ku - kl + 1;
        const int nkji = nk*nj*ni;
        const int nji  = nj*ni;

        // Middle loop over k,j,i
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji),[&](const int idx) {
          int k = idx/nji;
          int j = (idx - k*nji)/ni;
          int i = (idx - k*nji - j*ni) + il;
          j += jl;
          k += kl;

          // indices refer to coarse array.  So must compute indices for fine array
          int finei = (i - indcs.cis)*2 + indcs.is;
          int finej = (j - indcs.cjs)*2 + indcs.js;
          int finek = (k - indcs.cks)*2 + indcs.ks;

          // Volume-weighted, matching MeshRefinement::RestrictCC -- see the doc comment
          // there. Only fine geometry is needed since V_coarse == sum(V_fine).
          // restrict in 2D
          if (!(three_d)) {
            if (uni) {
              ca(m,v,kl,j,i) = 0.25*(a(m,v,kl,finej  ,finei) + a(m,v,kl,finej  ,finei+1)
                                   + a(m,v,kl,finej+1,finei) + a(m,v,kl,finej+1,finei+1));
              return;
            }
            Real v00 = geom.Vol(m,kl,finej  ,finei  );
            Real v01 = geom.Vol(m,kl,finej  ,finei+1);
            Real v10 = geom.Vol(m,kl,finej+1,finei  );
            Real v11 = geom.Vol(m,kl,finej+1,finei+1);
            ca(m,v,kl,j,i) = (v00*a(m,v,kl,finej  ,finei) + v01*a(m,v,kl,finej  ,finei+1)
                            + v10*a(m,v,kl,finej+1,finei) + v11*a(m,v,kl,finej+1,finei+1))
                             /(v00 + v01 + v10 + v11);
          // restrict in 3D
          } else {
            if (!is_z4c) {
              if (uni) {
                ca(m,v,k,j,i) = 0.125*(
                    a(m,v,finek  ,finej  ,finei) + a(m,v,finek  ,finej  ,finei+1)
                  + a(m,v,finek  ,finej+1,finei) + a(m,v,finek  ,finej+1,finei+1)
                  + a(m,v,finek+1,finej,  finei) + a(m,v,finek+1,finej,  finei+1)
                  + a(m,v,finek+1,finej+1,finei) + a(m,v,finek+1,finej+1,finei+1));
                return;
              }
              Real v000 = geom.Vol(m,finek  ,finej  ,finei  );
              Real v001 = geom.Vol(m,finek  ,finej  ,finei+1);
              Real v010 = geom.Vol(m,finek  ,finej+1,finei  );
              Real v011 = geom.Vol(m,finek  ,finej+1,finei+1);
              Real v100 = geom.Vol(m,finek+1,finej  ,finei  );
              Real v101 = geom.Vol(m,finek+1,finej  ,finei+1);
              Real v110 = geom.Vol(m,finek+1,finej+1,finei  );
              Real v111 = geom.Vol(m,finek+1,finej+1,finei+1);
              ca(m,v,k,j,i) = (
                  v000*a(m,v,finek  ,finej  ,finei) + v001*a(m,v,finek  ,finej  ,finei+1)
                + v010*a(m,v,finek  ,finej+1,finei) + v011*a(m,v,finek  ,finej+1,finei+1)
                + v100*a(m,v,finek+1,finej,  finei) + v101*a(m,v,finek+1,finej,  finei+1)
                + v110*a(m,v,finek+1,finej+1,finei) + v111*a(m,v,finek+1,finej+1,finei+1))
                 /(v000 + v001 + v010 + v011 + v100 + v101 + v110 + v111);
            } else {
                switch (indcs.ng) {
                  case 2: ca(m,v,k,j,i) = RestrictInterpolation<2>(m,v,finek,finej,finei,
                              nx1,nx2,nx3,a,restrict_2nd,restrict_4th,restrict_4th_edge);
                          break;
                  case 4: ca(m,v,k,j,i) = RestrictInterpolation<4>(m,v,finek,finej,finei,
                              nx1,nx2,nx3,a,restrict_2nd,restrict_4th,restrict_4th_edge);
                          break;
                }
            }
          }
        });
      }
      tmember.team_barrier();
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProlongateCC()
//! \brief Prolongate data at boundaries for cell-centered data.
//! Code here is based on MeshRefinement::ProlongateCellCenteredValues() in C++ version

void MeshBoundaryValuesCC::ProlongateCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca,
    bool is_z4c) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;

  // ptr to z4c, which requires different prolongation/restriction scheme
  //bool not_z4c = (pmbp->pz4c == nullptr)? true : false;

  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR
  int nmnv = nmb*nnghbr*nvar;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &rbuf = recvbuf;
  auto &indcs  = pmy_pack->pmesh->mb_indcs;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;

  // Fine and coarse volumetric centroids for the conservative prolongation (see the
  // ProlongCC doc comment). Only the six centroid arrays are pulled out, so the kernel
  // closure grows by 6 View handles rather than by two whole GeomData structs.
  auto &cx1v = pmy_pack->pgeom->coarse_geom_data.x1v;
  auto &cx2v = pmy_pack->pgeom->coarse_geom_data.x2v;
  auto &cx3v = pmy_pack->pgeom->coarse_geom_data.x3v;
  auto &x1v  = pmy_pack->pgeom->geom_data.x1v;
  auto &x2v  = pmy_pack->pgeom->geom_data.x2v;
  auto &x3v  = pmy_pack->pgeom->geom_data.x3v;
  const bool geom_uni = pmy_pack->pgeom->geom_data.cells_uniform;
  auto &nx1 = indcs.nx1;
  auto &nx2 = indcs.nx2;
  auto &nx3 = indcs.nx3;
  auto& prolong_2nd = pmy_pack->pmesh->pmr->weights.prolong_2nd;
  auto& prolong_4th = pmy_pack->pmesh->pmr->weights.prolong_4th;

  // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("ProlCC", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(nnghbr*nvar);
    const int n = (tmember.league_rank() - m*(nnghbr*nvar))/nvar;
    const int v = (tmember.league_rank() - m*(nnghbr*nvar) - n*nvar);

    // only prolongate when neighbor exists and is at coarser level
    if ((nghbr.d_view(m,n).gid >= 0) && (nghbr.d_view(m,n).lev < mblev.d_view(m))) {
      // loop over indices for prolongation on this buffer
      int il = rbuf[n].iprol[0].bis;
      int iu = rbuf[n].iprol[0].bie;
      int jl = rbuf[n].iprol[0].bjs;
      int ju = rbuf[n].iprol[0].bje;
      int kl = rbuf[n].iprol[0].bks;
      int ku = rbuf[n].iprol[0].bke;
      const int ni = iu - il + 1;
      const int nj = ju - jl + 1;
      const int nk = ku - kl + 1;
      const int nkji = nk*nj*ni;
      const int nji  = nj*ni;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int k = idx/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + il;
        j += jl;
        k += kl;

        // indices for prolongation refer to coarse array.  So must compute
        // indices for fine array
        int fi = (i - indcs.cis)*2 + indcs.is;
        int fj = (j - indcs.cjs)*2 + indcs.js;
        int fk = (k - indcs.cks)*2 + indcs.ks;
        // call inlined prolongation operator for CC variables
        if (!is_z4c) {
          ProlongCC(m,v,k,j,i,fk,fj,fi,multi_d,three_d,
                    cx1v,cx2v,cx3v,x1v,x2v,x3v,
                    geom_uni,ca,a);
        } else {
          switch (indcs.ng) {
            case 2: HighOrderProlongCC<2>(m,v,k,j,i,fk,fj,fi,nx1,nx2,nx3,
                                          ca,a,prolong_2nd);
                    break;
            case 4: HighOrderProlongCC<4>(m,v,k,j,i,fk,fj,fi,nx1,nx2,nx3,
                                          ca,a,prolong_4th);
                    break;
          }
        }
      });
    }
    tmember.team_barrier();
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void FillCoarseInBndryFC()
//! \brief As in the case of cell-centered variables, to ensure that the coarse field is
//! up-to-date in all neighboring cells touched by the prolongation interpolation stencil,
//! data is also restricted to coarse array in boundaries between MeshBlocks at the same
//! level.

void MeshBoundaryValuesFC::FillCoarseInBndryFC(DvceFaceFld4D<Real> &b,
                                           DvceFaceFld4D<Real> &cb) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;

  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &indcs  = pmy_pack->pmesh->mb_indcs;
  auto &mblev = pmy_pack->pmb->mb_lev;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  // Restrict data into coarse array in any boundary filled with data from the same
  // level. (Only needed in multidimensions)

  // Area-weighted, for the same reason as MeshRefinement::RestrictFC -- this is the
  // second, independent copy of face-centered restriction (the per-stage boundary path
  // rather than the regrid path), and the two must agree or the prolongation stencil
  // reads coarse data that disagrees with what a regrid would have produced.
  auto &geom = pmy_pack->pgeom->geom_data;
  const bool uni = geom.cells_uniform;

  if (multi_d) {
    int nmnv = 3*nmb*nnghbr;
    auto &rbuf = recvbuf;
    auto &cis = indcs.cis;
    auto &cjs = indcs.cjs;
    auto &cks = indcs.cks;
    // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
    Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
    Kokkos::parallel_for("ProlFCSame", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
      const int m = (tmember.league_rank())/(3*nnghbr);
      const int n = (tmember.league_rank() - m*(3*nnghbr))/3;
      const int v = (tmember.league_rank() - m*(3*nnghbr) - 3*n);

      // only restrict when neighbor exists and is at SAME level
      if ((nghbr.d_view(m,n).gid >= 0) && (nghbr.d_view(m,n).lev == mblev.d_view(m))) {
        // loop over indices for receives at same level, but convert loop limits to
        // coarse array
        int il = (rbuf[n].isame[v].bis + cis)/2;
        int iu = (rbuf[n].isame[v].bie + cis)/2;
        int jl = (rbuf[n].isame[v].bjs + cjs)/2;
        int ju = (rbuf[n].isame[v].bje + cjs)/2;
        int kl = (rbuf[n].isame[v].bks + cks)/2;
        int ku = (rbuf[n].isame[v].bke + cks)/2;

        const int ni = iu - il + 1;
        const int nj = ju - jl + 1;
        const int nk = ku - kl + 1;
        const int nkji = nk*nj*ni;
        const int nji  = nj*ni;

        // Middle loop over k,j,i
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji),[&](const int idx) {
          int k = idx/nji;
          int j = (idx - k*nji)/ni;
          int i = (idx - k*nji - j*ni) + il;
          j += jl;
          k += kl;

          // indices refer to coarse array.  So must compute indices for fine array
          int fk = (k - indcs.cks)*2 + indcs.ks;
          int fj = (j - indcs.cjs)*2 + indcs.js;
          int fi = (i - indcs.cis)*2 + indcs.is;

          // restrict in 2D
          if (!(three_d)) {
            if (v==0) {
              cb.x1f(m,kl,j,i) = uni ?
                0.5*(b.x1f(m,kl,fj,fi) + b.x1f(m,kl,fj+1,fi)) :
                WeightedMean(geom.Area1(m,kl,fj  ,fi), b.x1f(m,kl,fj  ,fi),
                             geom.Area1(m,kl,fj+1,fi), b.x1f(m,kl,fj+1,fi));
            } else if (v==1) {
              cb.x2f(m,kl,j,i) = uni ?
                0.5*(b.x2f(m,kl,fj,fi) + b.x2f(m,kl,fj,fi+1)) :
                WeightedMean(geom.Area2(m,kl,fj,fi  ), b.x2f(m,kl,fj,fi  ),
                             geom.Area2(m,kl,fj,fi+1), b.x2f(m,kl,fj,fi+1));
            } else {
              Real b3c = uni ?
                0.25*(b.x3f(m,kl,fj  ,fi) + b.x3f(m,kl,fj  ,fi+1)
                    + b.x3f(m,kl,fj+1,fi) + b.x3f(m,kl,fj+1,fi+1)) :
                WeightedMean(geom.Area3(m,kl,fj  ,fi  ), b.x3f(m,kl,fj  ,fi  ),
                             geom.Area3(m,kl,fj  ,fi+1), b.x3f(m,kl,fj  ,fi+1),
                             geom.Area3(m,kl,fj+1,fi  ), b.x3f(m,kl,fj+1,fi  ),
                             geom.Area3(m,kl,fj+1,fi+1), b.x3f(m,kl,fj+1,fi+1));
              cb.x3f(m,kl  ,j,i) = b3c;
              cb.x3f(m,kl+1,j,i) = b3c;
            }

          // restrict in 3D
          } else {
            if (v==0) {
              cb.x1f(m,k,j,i) = uni ?
                0.25*(b.x1f(m,fk  ,fj,fi) + b.x1f(m,fk  ,fj+1,fi)
                    + b.x1f(m,fk+1,fj,fi) + b.x1f(m,fk+1,fj+1,fi)) :
                WeightedMean(geom.Area1(m,fk  ,fj  ,fi), b.x1f(m,fk  ,fj  ,fi),
                             geom.Area1(m,fk  ,fj+1,fi), b.x1f(m,fk  ,fj+1,fi),
                             geom.Area1(m,fk+1,fj  ,fi), b.x1f(m,fk+1,fj  ,fi),
                             geom.Area1(m,fk+1,fj+1,fi), b.x1f(m,fk+1,fj+1,fi));
            } else if (v==1) {
              cb.x2f(m,k,j,i) = uni ?
                0.25*(b.x2f(m,fk  ,fj,fi) + b.x2f(m,fk  ,fj,fi+1)
                    + b.x2f(m,fk+1,fj,fi) + b.x2f(m,fk+1,fj,fi+1)) :
                WeightedMean(geom.Area2(m,fk  ,fj,fi  ), b.x2f(m,fk  ,fj,fi  ),
                             geom.Area2(m,fk  ,fj,fi+1), b.x2f(m,fk  ,fj,fi+1),
                             geom.Area2(m,fk+1,fj,fi  ), b.x2f(m,fk+1,fj,fi  ),
                             geom.Area2(m,fk+1,fj,fi+1), b.x2f(m,fk+1,fj,fi+1));
            } else {
              cb.x3f(m,k,j,i) = uni ?
                0.25*(b.x3f(m,fk,fj  ,fi) + b.x3f(m,fk,fj  ,fi+1)
                    + b.x3f(m,fk,fj+1,fi) + b.x3f(m,fk,fj+1,fi+1)) :
                WeightedMean(geom.Area3(m,fk,fj  ,fi  ), b.x3f(m,fk,fj  ,fi  ),
                             geom.Area3(m,fk,fj  ,fi+1), b.x3f(m,fk,fj  ,fi+1),
                             geom.Area3(m,fk,fj+1,fi  ), b.x3f(m,fk,fj+1,fi  ),
                             geom.Area3(m,fk,fj+1,fi+1), b.x3f(m,fk,fj+1,fi+1));
            }
          }
        });
      }
      tmember.team_barrier();
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProlongateFC()
//! \brief Prolongate data at boundaries for face-centered data (e.g. magnetic fields).

void MeshBoundaryValuesFC::ProlongateFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;

  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &indcs  = pmy_pack->pmesh->mb_indcs;
  auto &mblev = pmy_pack->pmb->mb_lev;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  // Prolongate b.x1f/b.x2f/b.x3f at all shared coarse/fine cell edges
  // Code here is based on MeshRefinement::ProlongateSharedFieldX1/2/3() and
  // MeshRefinement::ProlongateInternalField() in C++ version

  // Area-weighted transverse face centroids for the conservative prolongation. Must
  // match what MeshRefinement::RefineFC passes on the regrid path, or the per-stage
  // boundary values and the post-regrid values disagree at a level boundary.
  auto &geom = pmy_pack->pgeom->geom_data;
  auto &cgeom = pmy_pack->pgeom->coarse_geom_data;
  const bool uni = geom.cells_uniform;
  // the internal-face (Toth-Roe) operator keys on cubic_cells instead -- see
  // ProlongFCInternal in mesh/prolongation.hpp
  const bool cubic = geom.cubic_cells;

  // Outer loop over (# of MeshBlocks)*(# of buffers)*(three field components)
  {int nmnv = 3*nmb*nnghbr;
  auto &rbuf = recvbuf;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("ProFC-2d-shared", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(3*nnghbr);
    const int n = (tmember.league_rank() - m*(3*nnghbr))/3;
    const int v = (tmember.league_rank() - m*(3*nnghbr) - 3*n);

    // only prolongate when neighbor exists and is at coarser level
    if ((nghbr.d_view(m,n).gid >= 0) && (nghbr.d_view(m,n).lev < mblev.d_view(m))) {
      int ox1, ox2, ox3;
      NeighborOffsetFromIndex(n, ox1, ox2, ox3);
      const int my_lev = mblev.d_view(m);

      int il = rbuf[n].iprol[v].bis;
      int iu = rbuf[n].iprol[v].bie;
      int jl = rbuf[n].iprol[v].bjs;
      int ju = rbuf[n].iprol[v].bje;
      int kl = rbuf[n].iprol[v].bks;
      int ku = rbuf[n].iprol[v].bke;
      const int ni = iu - il + 1;
      const int nj = ju - jl + 1;
      const int nk = ku - kl + 1;
      const int nkji = nk*nj*ni;
      const int nji  = nj*ni;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember,nkji),[&](const int idx) {
        int k = idx/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + il;
        j += jl;
        k += kl;

        int fi = (i - indcs.cis)*2 + indcs.is;                   // fine i
        int fj = (multi_d)? ((j - indcs.cjs)*2 + indcs.js) : j;  // fine j
        int fk = (three_d)? ((k - indcs.cks)*2 + indcs.ks) : k;  // fine k

        // Prolongate face-centered fields at shared faces betwen fine and coarse cells
        // by calling inlined prolongation operator for FC variables
        if (v==0) {
          ProlongFCSharedX1FaceOwned(m,nnghbr,k,j,i,fk,fj,fi,ox1,ox2,ox3,my_lev,
                                     multi_d,three_d,indcs,nghbr,
                                     cgeom.fc1_2,cgeom.fc1_3,geom.fc1_2,geom.fc1_3,uni,
                                     cb.x1f,b.x1f);
        } else if (v==1) {
          ProlongFCSharedX2FaceOwned(m,nnghbr,k,j,i,fk,fj,fi,ox1,ox2,ox3,my_lev,
                                     three_d,indcs,nghbr,
                                     cgeom.fc2_1,cgeom.fc2_3,geom.fc2_1,geom.fc2_3,uni,
                                     cb.x2f,b.x2f);
        } else {
          ProlongFCSharedX3FaceOwned(m,nnghbr,k,j,i,fk,fj,fi,ox1,ox2,ox3,my_lev,
                                     multi_d,indcs,nghbr,
                                     cgeom.fc3_1,cgeom.fc3_2,geom.fc3_1,geom.fc3_2,uni,
                                     cb.x3f,b.x3f);
        }
      });
    }
    tmember.team_barrier();
  });}

  // Now prolongate b.x1f/b.x2f/b.x3f at interior fine cells using the 2nd-order
  // divergence-preserving interpolation scheme of Toth & Roe, JCP 180, 736 (2002).
  // Note prolongation at shared coarse/fine cell edges must be completed first as
  // interpolation formulae use these values.

  // Outer loop over (# of MeshBlocks)*(# of buffers)
  {int nmn = nmb*nnghbr;
  bool &one_d = pmy_pack->pmesh->one_d;
  auto &rbuf = recvbuf;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmn, Kokkos::AUTO);
  Kokkos::parallel_for("ProFC-2d-int", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(nnghbr);
    const int n = (tmember.league_rank() - m*(nnghbr));

    // only prolongate when neighbor exists and is at coarser level
    if ((nghbr.d_view(m,n).gid >= 0) && (nghbr.d_view(m,n).lev < mblev.d_view(m))) {
      int ox1, ox2, ox3;
      NeighborOffsetFromIndex(n, ox1, ox2, ox3);
      const int my_lev = mblev.d_view(m);

      // use prolongation indices of different field components for interior fine cells
      int il = rbuf[n].iprol[2].bis;
      int iu = rbuf[n].iprol[2].bie;
      int jl = rbuf[n].iprol[0].bjs;
      int ju = rbuf[n].iprol[0].bje;
      int kl = rbuf[n].iprol[1].bks;
      int ku = rbuf[n].iprol[1].bke;
      const int ni = iu - il + 1;
      const int nj = ju - jl + 1;
      const int nk = ku - kl + 1;
      const int nkji = nk*nj*ni;
      const int nji  = nj*ni;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember,nkji),[&](const int idx) {
        int k = idx/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + il;
        j += jl;
        k += kl;

        int fi = (i - indcs.cis)*2 + indcs.is;   // fine i
        int fj = (j - indcs.cjs)*2 + indcs.js;   // fine j
        int fk = (k - indcs.cks)*2 + indcs.ks;   // fine k

        if (one_d) {
          // In 1D, interior face field is trivial. Averaging the two bracketing FLUXES
          // rather than the two fields -- see ProlongFCInternal1D.
          Real a_l = geom.Area1(m,fk,fj,fi  );
          Real a_r = geom.Area1(m,fk,fj,fi+2);
          Real a_c = geom.Area1(m,fk,fj,fi+1);
          Real val = (cubic || !(a_c > 0.0)) ?
              0.5*(b.x1f(m,fk,fj,fi) + b.x1f(m,fk,fj,fi+2)) :
              0.5*(a_l*b.x1f(m,fk,fj,fi) + a_r*b.x1f(m,fk,fj,fi+2))/a_c;
          StoreProlongatedFCFace(m, nnghbr, 0, fk, fj, fi+1, ox1, ox2, ox3,
                                 my_lev, indcs, nghbr, val, b.x1f);
        } else {
          // in multi-D call inlined prolongation operator for FC fields at internal faces
          ProlongFCInternalOwned(m,nnghbr,fk,fj,fi,ox1,ox2,ox3,my_lev,
                                 three_d,indcs,nghbr,geom,cubic,b);
        }
      });
    }
    tmember.team_barrier();
  });}

  return;
}
