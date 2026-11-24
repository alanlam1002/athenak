//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_budget.cpp
//! \brief Optional energy-budget accounting for radiation (M)HD, see radiation_budget.hpp

#include <cstdio>
#include <cstdlib>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "radiation/radiation.hpp"
#include "radiation/radiation_budget.hpp"

namespace radiation {

RadBudget::RadBudget(MeshBlockPack *pp, ParameterInput *pin) :
    eg_base(0.0), er_base(0.0), nbadcell(0.0), acc("rad_budget_acc", 4), pmy_pack(pp) {
  for (int x=0; x<kNBudget; ++x) {c[x] = 0.0; c_base[x] = 0.0;}
  // resume the counters on restart (keys present only in restart parameter blocks)
  if (global_variable::my_rank == 0) {
    for (int x=0; x<kNBudget; ++x) {
      std::string key = "budget_c" + std::to_string(x) + "_hex";
      if (pin->DoesParameterExist("radiation", key)) {
        c[x] = std::strtod(pin->GetString("radiation", key).c_str(), nullptr);
      }
    }
    if (pin->DoesParameterExist("radiation", "budget_nbad_hex")) {
      nbadcell = std::strtod(pin->GetString("radiation", "budget_nbad_hex").c_str(), nullptr);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void RadBudget::SaveToPin()
//! \brief store the global counter sums in the parameter block (called by every rank just
//! before a restart file writes its parameters; collective with MPI)

void RadBudget::SaveToPin(ParameterInput *pin) {
  Real tot[kNBudget+1];
  for (int x=0; x<kNBudget; ++x) tot[x] = c[x];
  tot[kNBudget] = nbadcell;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, tot, kNBudget+1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  char buf[64];
  for (int x=0; x<kNBudget; ++x) {
    std::snprintf(buf, sizeof(buf), "%a", tot[x]);  // hex float: exact round trip
    pin->SetString("radiation", "budget_c" + std::to_string(x) + "_hex", std::string(buf));
  }
  std::snprintf(buf, sizeof(buf), "%a", tot[kNBudget]);
  pin->SetString("radiation", "budget_nbad_hex", std::string(buf));
}

//----------------------------------------------------------------------------------------
//! \fn Real RadBudget::GasEnergy()
//! \brief E_gas = -sum u0(IEN) dV over active cells of this rank

Real RadBudget::GasEnergy() {
  DvceArray5D<Real> u0_;
  if (pmy_pack->pmhd != nullptr) {
    u0_ = pmy_pack->pmhd->u0;
  } else if (pmy_pack->phydro != nullptr) {
    u0_ = pmy_pack->phydro->u0;
  } else {
    return 0.0;
  }
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1, nji = nx2*nx1;
  const int nmkji = (pmy_pack->nmb_thispack)*nkji;
  auto &size = pmy_pack->pmb->mb_size;
  Real sum = 0.0;
  Kokkos::parallel_reduce("budget_egas", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &s) {
    int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    Real dv = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    s -= u0_(m,IEN,k,j,i)*dv;
  }, Kokkos::Sum<Real>(sum));
  return sum;
}

//----------------------------------------------------------------------------------------
//! \fn Real RadBudget::RadEnergy()
//! \brief E_rad = -sum_n w_n i0_n dV over active cells of this rank

Real RadBudget::RadEnergy() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int nang = pmy_pack->prad->prgeo->nangles;
  const int nkji = nx3*nx2*nx1, nji = nx2*nx1, nnkji = nang*nkji;
  const int nmnkji = (pmy_pack->nmb_thispack)*nnkji;
  auto &size = pmy_pack->pmb->mb_size;
  auto &i0_ = pmy_pack->prad->i0;
  auto &w_ = pmy_pack->prad->prgeo->solid_angles;
  Real sum = 0.0;
  Kokkos::parallel_reduce("budget_erad", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmnkji),
  KOKKOS_LAMBDA(const int &idx, Real &s) {
    int m = idx/nnkji;
    int n = (idx - m*nnkji)/nkji;
    int k = (idx - m*nnkji - n*nkji)/nji;
    int j = (idx - m*nnkji - n*nkji - k*nji)/nx1;
    int i = (idx - m*nnkji - n*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    Real dv = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    s -= w_.d_view(n)*i0_(m,n,k,j,i)*dv;
  }, Kokkos::Sum<Real>(sum));
  return sum;
}

//----------------------------------------------------------------------------------------
//! \fn void RadBudget::StageBase()
//! \brief Store counters and energies of the RK base state (start of the step)

void RadBudget::StageBase(int stage) {
  if (stage != 1) return;
  for (int x=0; x<kNBudget; ++x) {c_base[x] = c[x];}
  eg_base = GasEnergy();
  er_base = RadEnergy();
}

//----------------------------------------------------------------------------------------
//! \fn void RadBudget::Combine()
//! \brief Combine counters with the RK weights of this stage (u = gam0*u + gam1*u_base)

void RadBudget::Combine(Real gam0, Real gam1) {
  for (int x=0; x<kNBudget; ++x) {c[x] = gam0*c[x] + gam1*c_base[x];}
}

//----------------------------------------------------------------------------------------
//! \fn Real RadBudget::RadFaceFlux()
//! \brief Radiation energy that enters this rank's part of the domain through physical
//! (non-block) faces during the RK update, beta_dt * sum_n w_n F_n dA (inward positive)

Real RadBudget::RadFaceFlux(Real beta_dt) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je, ks = indcs.ks;
  int ke = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int nmb = pmy_pack->nmb_thispack;
  int nang = pmy_pack->prad->prgeo->nangles;
  auto &size = pmy_pack->pmb->mb_size;
  auto &bcs = pmy_pack->pmb->mb_bcs;
  auto &flx = pmy_pack->prad->iflx;
  auto &w_ = pmy_pack->prad->prgeo->solid_angles;
  Real sum = 0.0, part = 0.0;
  // x1 faces: loop over (m,n,k,j)
  {
    const int nkj = nx3*nx2, nnkj = nang*nkj, ntot = nmb*nnkj;
    Kokkos::parallel_reduce("budget_rflx1", Kokkos::RangePolicy<>(DevExeSpace(), 0, ntot),
    KOKKOS_LAMBDA(const int &idx, Real &s) {
      int m = idx/nnkj;
      int n = (idx - m*nnkj)/nkj;
      int k = (idx - m*nnkj - n*nkj)/nx2 + ks;
      int j = (idx - m*nnkj - n*nkj - (k-ks)*nx2) + js;
      Real da = size.d_view(m).dx2*size.d_view(m).dx3*w_.d_view(n);
      // E = -sum w i0 dV and i0 -= beta_dt*(F(i+1)-F(i))/dx1, so
      // dE = -beta_dt*F(is)*dA at the inner face and +beta_dt*F(ie+1)*dA at the outer
      if (bcs.d_view(m,BoundaryFace::inner_x1) != BoundaryFlag::block) {
        s -= flx.x1f(m,n,k,j,is)*da;
      }
      if (bcs.d_view(m,BoundaryFace::outer_x1) != BoundaryFlag::block) {
        s += flx.x1f(m,n,k,j,ie+1)*da;
      }
    }, Kokkos::Sum<Real>(part));
    sum += part;
  }
  if (pmy_pack->pmesh->multi_d) {
    const int nki = nx3*nx1, nnki = nang*nki, ntot = nmb*nnki;
    Kokkos::parallel_reduce("budget_rflx2", Kokkos::RangePolicy<>(DevExeSpace(), 0, ntot),
    KOKKOS_LAMBDA(const int &idx, Real &s) {
      int m = idx/nnki;
      int n = (idx - m*nnki)/nki;
      int k = (idx - m*nnki - n*nki)/nx1 + ks;
      int i = (idx - m*nnki - n*nki - (k-ks)*nx1) + is;
      Real da = size.d_view(m).dx1*size.d_view(m).dx3*w_.d_view(n);
      if (bcs.d_view(m,BoundaryFace::inner_x2) != BoundaryFlag::block) {
        s -= flx.x2f(m,n,k,js,i)*da;
      }
      if (bcs.d_view(m,BoundaryFace::outer_x2) != BoundaryFlag::block) {
        s += flx.x2f(m,n,k,je+1,i)*da;
      }
    }, Kokkos::Sum<Real>(part));
    sum += part;
  }
  if (pmy_pack->pmesh->three_d) {
    const int nji = nx2*nx1, nnji = nang*nji, ntot = nmb*nnji;
    Kokkos::parallel_reduce("budget_rflx3", Kokkos::RangePolicy<>(DevExeSpace(), 0, ntot),
    KOKKOS_LAMBDA(const int &idx, Real &s) {
      int m = idx/nnji;
      int n = (idx - m*nnji)/nji;
      int j = (idx - m*nnji - n*nji)/nx1 + js;
      int i = (idx - m*nnji - n*nji - (j-js)*nx1) + is;
      Real da = size.d_view(m).dx1*size.d_view(m).dx2*w_.d_view(n);
      if (bcs.d_view(m,BoundaryFace::inner_x3) != BoundaryFlag::block) {
        s -= flx.x3f(m,n,ks,j,i)*da;
      }
      if (bcs.d_view(m,BoundaryFace::outer_x3) != BoundaryFlag::block) {
        s += flx.x3f(m,n,ke+1,j,i)*da;
      }
    }, Kokkos::Sum<Real>(part));
    sum += part;
  }
  return beta_dt*sum;
}

//----------------------------------------------------------------------------------------
//! \fn Real RadBudget::GasFaceFlux()
//! \brief Gas energy that enters through physical faces during the (M)HD RK update

Real RadBudget::GasFaceFlux(Real beta_dt) {
  if (pmy_pack->pmhd != nullptr) {
    return GasFaceFluxOf(pmy_pack->pmhd->uflx, beta_dt);
  } else if (pmy_pack->phydro != nullptr) {
    return GasFaceFluxOf(pmy_pack->phydro->uflx, beta_dt);
  }
  return 0.0;
}

Real RadBudget::GasFaceFluxOf(const DvceFaceFld5D<Real> &flx, Real beta_dt) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je, ks = indcs.ks;
  int ke = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int nmb = pmy_pack->nmb_thispack;
  auto &size = pmy_pack->pmb->mb_size;
  auto &bcs = pmy_pack->pmb->mb_bcs;
  Real sum = 0.0, part = 0.0;
  {
    const int nkj = nx3*nx2, ntot = nmb*nkj;
    Kokkos::parallel_reduce("budget_gflx1", Kokkos::RangePolicy<>(DevExeSpace(), 0, ntot),
    KOKKOS_LAMBDA(const int &idx, Real &s) {
      int m = idx/nkj;
      int k = (idx - m*nkj)/nx2 + ks;
      int j = (idx - m*nkj - (k-ks)*nx2) + js;
      Real da = size.d_view(m).dx2*size.d_view(m).dx3;
      if (bcs.d_view(m,BoundaryFace::inner_x1) != BoundaryFlag::block) {
        s -= flx.x1f(m,IEN,k,j,is)*da;
      }
      if (bcs.d_view(m,BoundaryFace::outer_x1) != BoundaryFlag::block) {
        s += flx.x1f(m,IEN,k,j,ie+1)*da;
      }
    }, Kokkos::Sum<Real>(part));
    sum += part;
  }
  if (pmy_pack->pmesh->multi_d) {
    const int nki = nx3*nx1, ntot = nmb*nki;
    Kokkos::parallel_reduce("budget_gflx2", Kokkos::RangePolicy<>(DevExeSpace(), 0, ntot),
    KOKKOS_LAMBDA(const int &idx, Real &s) {
      int m = idx/nki;
      int k = (idx - m*nki)/nx1 + ks;
      int i = (idx - m*nki - (k-ks)*nx1) + is;
      Real da = size.d_view(m).dx1*size.d_view(m).dx3;
      if (bcs.d_view(m,BoundaryFace::inner_x2) != BoundaryFlag::block) {
        s -= flx.x2f(m,IEN,k,js,i)*da;
      }
      if (bcs.d_view(m,BoundaryFace::outer_x2) != BoundaryFlag::block) {
        s += flx.x2f(m,IEN,k,je+1,i)*da;
      }
    }, Kokkos::Sum<Real>(part));
    sum += part;
  }
  if (pmy_pack->pmesh->three_d) {
    const int nji = nx2*nx1, ntot = nmb*nji;
    Kokkos::parallel_reduce("budget_gflx3", Kokkos::RangePolicy<>(DevExeSpace(), 0, ntot),
    KOKKOS_LAMBDA(const int &idx, Real &s) {
      int m = idx/nji;
      int j = (idx - m*nji)/nx1 + js;
      int i = (idx - m*nji - (j-js)*nx1) + is;
      Real da = size.d_view(m).dx1*size.d_view(m).dx2;
      if (bcs.d_view(m,BoundaryFace::inner_x3) != BoundaryFlag::block) {
        s -= flx.x3f(m,IEN,ks,j,i)*da;
      }
      if (bcs.d_view(m,BoundaryFace::outer_x3) != BoundaryFlag::block) {
        s += flx.x3f(m,IEN,ke+1,j,i)*da;
      }
    }, Kokkos::Sum<Real>(part));
    sum += part;
  }
  return beta_dt*sum;
}

//----------------------------------------------------------------------------------------
//! \fn RadBudget::ZeroAcc / ReadAcc
//! \brief device accumulators used with Kokkos::atomic_add inside kernels

void RadBudget::ZeroAcc() {
  Kokkos::deep_copy(acc, 0.0);
}

void RadBudget::ReadAcc(Real *a, int n) {
  auto h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), acc);
  for (int x=0; x<n; ++x) {a[x] = h(x);}
}

} // namespace radiation
