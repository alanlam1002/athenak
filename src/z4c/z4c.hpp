#ifndef Z4C_Z4C_HPP_
#define Z4C_Z4C_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c.hpp
//! \brief definitions for Z4c class

#include <array>
#include <map>
#include <memory>    // make_unique, unique_ptr
#include <list>
#include <string>
#include <vector>
#include "athena.hpp"
#include "utils/finite_diff.hpp"
#include "utils/cart_grid.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"
#include "athena_tensor.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "geodesic-grid/spherical_grid.hpp"

// forward declarations
class Coordinates;
class Driver;
class CompactObjectTracker;
class FastFlow;
class HorizonDump;
class DriftControl;

namespace z4c {
class Z4c_AMR;

// Shift needed for derivatives
//----------------------------------------------------------------------------------------
//! \class Z4c

class Z4c {
 public:
  Z4c(MeshBlockPack *ppack, ParameterInput *pin);
  ~Z4c();

  // Indices of evolved variables
  enum {
    I_Z4C_CHI,
    I_Z4C_GXX, I_Z4C_GXY, I_Z4C_GXZ, I_Z4C_GYY, I_Z4C_GYZ, I_Z4C_GZZ,
    I_Z4C_KHAT,
    I_Z4C_AXX, I_Z4C_AXY, I_Z4C_AXZ, I_Z4C_AYY, I_Z4C_AYZ, I_Z4C_AZZ,
    I_Z4C_GAMX, I_Z4C_GAMY, I_Z4C_GAMZ,
    I_Z4C_THETA,
    I_Z4C_ALPHA,
    I_Z4C_BETAX, I_Z4C_BETAY, I_Z4C_BETAZ,
    nz4c
  };
  // Names of Z4c variables
  static char const * const Z4c_names[nz4c];

  // Cartoon (axisymmetry via ghost planes in the missing direction, D-001..D-011).
  // On a 2D Cartesian mesh the evolved plane is y = 0 with x = x1 >= 0 and z = x2; the
  // Cartoon direction y is the x3 INDEX. Component indices follow the mesh axes, so the
  // physical labels map as  x -> 0,  z -> 1,  y -> 2.  E.g. physical beta^y is
  // I_Z4C_BETAZ and physical g_xz is I_Z4C_GXY. CartoonParity holds the sign of each
  // variable under the pi rotation about the z axis, (x,y,z) -> (-x,-y,z) (D-002).
  // The mode follows the coord (D-014): cartesian -> none, cylindrical_axisym (2D) ->
  // axisym, spherical_polar (1D) -> spherical (M4; fatal until then).
  enum class CartoonMode {none, axisym, spherical};
  // A Cartoon fill acts on a variable set described by where its symmetric tensors
  // (6 components, mesh order xx, xz, xy, zz, zy, yy in physical labels) and vectors
  // (3 components, mesh order x, z, y) start; every other variable is a scalar (M2a).
  static constexpr int kCartoonMaxVar = nz4c;
  struct CartoonVarSet {
    int nvar;
    int nsym, sym[2];
    int nvec, vec[2];
  };
  static CartoonVarSet CartoonZ4cSet();  // u0 / u_cart
  static CartoonVarSet CartoonADMSet();  // u_adm (g, K, psi4) / u_adm_cart
  struct CartoonParity {
    int p[kCartoonMaxVar];
  };
  static CartoonParity CartoonParityTable();
  // the D-002 parity of any set, derived from its descriptor (the pi rotation flips the
  // mesh x1 and x3 components); equals CartoonParityTable() for the Z4c set
  static CartoonParity CartoonParityOf(const CartoonVarSet &vs);
  // Indices of Constraint variables
  enum {
    I_CON_C,
    I_CON_H,
    I_CON_M,
    I_CON_Z,
    I_CON_MX, I_CON_MY, I_CON_MZ,
    ncon,
  };
  // Names of constraint variables
  static char const * const Constraint_names[ncon];
  // Indices of matter fields
  /*enum {
    I_MAT_RHO,
    I_MAT_SX, I_MAT_SY, I_MAT_SZ,
    I_MAT_SXX, I_MAT_SXY, I_MAT_SXZ, I_MAT_SYY, I_MAT_SYZ, I_MAT_SZZ,
    nmat
  };
  // Names of matter variables
  static char const * const Matter_names[nmat];*/

  // data
  // flags to denote relativistic dynamics
  DvceArray5D<Real> u_con;     // constraints fields
  DvceArray5D<Real> u_mat;
  DvceArray5D<Real> u0;        // z4c solution
  DvceArray5D<Real> u1;        // z4c solution at intermediate timestep
  DvceArray5D<Real> u_rhs;     // z4c rhs storage
  DvceArray5D<Real> coarse_u0; // coarse representation of z4c solution
  DvceArray5D<Real> u_weyl; // weyl scalars
  DvceArray5D<Real> coarse_u_weyl; // coarse representation of weyl scalars
  // Cartoon working array (D-006): u_cart(m, n, 1+2ng, j, i), local k = ng is the y = 0
  // plane. A cache refilled from u0 before each consumer; never communicated, restarted,
  // load balanced or prolongated. Only allocated when cartoon != none.
  DvceArray5D<Real> u_cart;
  // The same for the ADM set (g_ij, K_ij, psi4 of u_adm), used by the ADM constraints
  // (M2a). Allocated on first use.
  DvceArray5D<Real> u_adm_cart;

  struct ADM_vars {
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> psi4;
    AthenaTensor<Real, TensorSymm::SYM2, 3, 2> g_dd;
    AthenaTensor<Real, TensorSymm::SYM2, 3, 2> vK_dd;
  };
  ADM_vars adm;
  ADM_vars adm_cart;  // accessors sliced from u_adm_cart (valid only with Cartoon)

  struct ADMhost_vars {
    AthenaHostTensor<Real, TensorSymm::NONE, 3, 0> psi4;
    AthenaHostTensor<Real, TensorSymm::SYM2, 3, 2> g_dd;
    AthenaHostTensor<Real, TensorSymm::SYM2, 3, 2> vK_dd;
  };

  struct Wave_Extr_vars {
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> rpsi4;
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> ipsi4;
  };
  Wave_Extr_vars weyl;

  struct Z4c_vars {
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> chi;     // conf. factor
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> vKhat;   // trace extr. curvature
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> vTheta;  // Theta var in Z4c
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> alpha;   // lapse
    AthenaTensor<Real, TensorSymm::NONE, 3, 1> vGam_u;  // Gamma functions (BSSN)
    AthenaTensor<Real, TensorSymm::NONE, 3, 1> beta_u;  // shift
    AthenaTensor<Real, TensorSymm::SYM2, 3, 2> g_dd;    // conf. 3-metric
    AthenaTensor<Real, TensorSymm::SYM2, 3, 2> vA_dd;   // conf. traceless extr. curvature
  };
  Z4c_vars z4c;
  Z4c_vars rhs;
  Z4c_vars z4c_cart;  // same accessors, sliced from u_cart (valid only with Cartoon)

  // aliases for the constraints
  struct Constraint_vars {
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> C;         // Z constraint monitor
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> H;         // hamiltonian constraint
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> M;         // norm squared of M_d
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> Z;         // Z constraint violation
    AthenaTensor<Real, TensorSymm::NONE, 3, 1> M_d;       // momentum constraint
  };
  Constraint_vars con;

  // aliases for the matter variables
  /*struct Matter_vars {
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> rho;       // matter energy density
    AthenaTensor<Real, TensorSymm::NONE, 3, 1> vS_d;       // matter momentum density
    AthenaTensor<Real, TensorSymm::SYM2, 3, 2> vS_dd;      // matter stress tensor
  };
  Matter_vars mat;*/

  struct Options {
    Real chi_psi_power;   // chi = psi^N, N = chi_psi_power
    // puncture's floor value for chi, use max(chi, chi_div_floor)
    // in non-differentiated chi
    Real chi_div_floor;
    Real chi_min_floor;   // minimum of chi, only used in slow-start-lapse
    bool floor_chi;       // used as a safe guard after RK update
    // where a square root is necessary.
    Real diss;            // amount of numerical dissipation
    Real eps_floor;       // a small number O(10^-12)
    // Constraint damping parameters
    Real damp_kappa1;
    Real damp_kappa2;
    // Gauge conditions for the lapse
    Real lapse_oplog;
    Real lapse_harmonicf;
    Real lapse_harmonic;
    Real lapse_advect;
    // slow start lapse condition
    bool slow_start_lapse;
    Real ssl_damping_amp;
    Real ssl_damping_time;
    Real ssl_damping_index;
    // Gauge condition for the shift
    Real shift_ggamma;
    Real shift_alpha2ggamma;
    Real shift_hh;
    Real shift_advect;
    Real shift_eta;
    // turn on shift damping smoothly
    bool slow_roll_eta;
    Real turn_on_time;
    // Enable BSSN if false (disable theta)
    bool use_z4c;
    // Apply the Sommerfeld condition for user BCs.
    bool user_Sbc;
    // Boundary extrapolation order
    int extrap_order;
    // Value of chi to specify the excision region for constraint evaluation
    Real excise_chi;
    // Cartoon: mode (derived from the coord, D-014), Kreiss-Oliger dissipation in the
    // Cartoon direction (<z4c> cartoon_ydiss, default false as in SACRA-2D; M2d T6 found
    // true changes the results by ~0.2x the truncation error, D-027), and the number of
    // Lagrange interpolation points (2ng+1, chat/0004).
    CartoonMode cartoon;
    bool cartoon_ydiss;
    int cartoon_npt;
    // D-025: run the ADM-constraint task only on cycles that write an output (default
    // false: every cycle, as before)
    bool constraints_on_output_only;

    // Flag if drift control for COT should be used.
    bool enable_driftcontrol;
    int dc_variety;
    int dc_tracker_index;
    Real dc_fixed_x, dc_fixed_y, dc_fixed_z;
    Real dc_damping_time, dc_damping_scale, dc_damping_coeff;
    Real dc_Kp, dc_Ki, dc_Kd;
    Real dc_omega_c, dc_omega_o, dc_zeta;
    Real dc_relaxation_time, dc_kappa;
    Real dc_gamma_suppress;
    Real dc_gain_x, dc_gain_y, dc_gain_z;
    int dc_gaussian_center;
  };
  Options opt;
  Real diss;              // Dissipation parameter

  // Boundary communication buffers and functions for u
  MeshBoundaryValuesCC *pbval_u;

  // Boundary communication buffers for the weyl scalar
  MeshBoundaryValuesCC *pbval_weyl;

  // following only used for time-evolving flow
  Real dtnew;

  // geodesic grid for wave extr
  std::vector<std::unique_ptr<SphericalGrid>> spherical_grids;
  // array storing waveform at each radii
  Real * psi_out;
  Real waveform_dt;
  Real last_output_time;
  int nrad; // number of radii to perform wave extraction

  // CCE
  Real cce_dump_dt;
  Real cce_dump_last_output_time;
  // dump data cube at horizon

  // functions
  void QueueZ4cTasks();
  TaskStatus InitRecv(Driver *d, int stage);
  TaskStatus ClearRecv(Driver *d, int stage);
  TaskStatus ClearSend(Driver *d, int stage);
  TaskStatus InitRecvWeyl(Driver *d, int stage);
  TaskStatus ClearRecvWeyl(Driver *d, int stage);
  TaskStatus ClearSendWeyl(Driver *d, int stage);
  TaskStatus CopyU(Driver *d, int stage);
  TaskStatus SendU(Driver *d, int stage);
  TaskStatus RecvU(Driver *d, int stage);
  TaskStatus SendWeyl(Driver *d, int stage);
  TaskStatus RecvWeyl(Driver *d, int stage);
  TaskStatus Prolongate(Driver *pdrive, int stage);
  TaskStatus ProlongateWeyl(Driver *pdrive, int stage);
  TaskStatus ExpRKUpdate(Driver *d, int stage);
  TaskStatus Z4cFloorChi(Driver *pdrive, int stage);
  TaskStatus NewTimeStep(Driver *d, int stage);
  TaskStatus ApplyPhysicalBCs(Driver *d, int stage);
  TaskStatus EnforceAlgConstr(Driver *d, int stage);

  TaskStatus ConvertZ4cToADM(Driver *d, int stage);
  TaskStatus UpdateExcisionMasks(Driver *d, int stage);
  TaskStatus ADMConstraints_(Driver *d, int stage);
  // D-025: u_con is stale after a skipped cycle; RefreshConstraints() recomputes it then
  bool con_stale = false;
  void RefreshConstraints();
  TaskStatus Z4cBoundaryRHS(Driver *d, int stage);
  TaskStatus RestrictU(Driver *d, int stage);
  TaskStatus RestrictWeyl(Driver *d, int stage);
  TaskStatus CCEDump(Driver *pdrive, int stage);
  TaskStatus TrackCompactObjects(Driver *d, int stage);
  TaskStatus FindHorizon(Driver *d, int stage);
  TaskStatus CalcWeylScalar(Driver *d, int stage);
  TaskStatus CalcWaveForm(Driver *d, int stage);
  TaskStatus DumpHorizons(Driver *d, int stage);

  template <int NGHOST>
  TaskStatus CalcRHS(Driver *d, int stage);
  template <int NGHOST>
  void ADMToZ4c(MeshBlockPack *pmbp, ParameterInput *pin);
  void GaugePreCollapsedLapse(MeshBlockPack *pmbp, ParameterInput *pin);
  void Z4cToADM(MeshBlockPack *pmbp);
  template <int NGHOST>
  void ADMConstraints(MeshBlockPack *pmbp);
  template <int NGHOST>
  void Z4cWeyl(MeshBlockPack *pmbp);
  void WaveExtr(MeshBlockPack *pmbp);
  void AlgConstr(MeshBlockPack *pmbp);
  // Cartoon fill u0 -> u_cart (K12, K3). The overload with arguments exists for the unit
  // test (npt scan, deliberately wrong parity, direct-interpolation cross-check at x<0).
  void CartoonFill(MeshBlockPack *pmbp);
  void CartoonFill(MeshBlockPack *pmbp, int npt, const CartoonParity &par,
                   bool direct_negx);
  // K12, K3 on any variable set: src(m, n, ks, j, i) -> dst(m, n, 1+2ng, j, i), n < nvar.
  // algc: apply the algebraic constraints (det g-tilde = 1, tr A-tilde = 0) to the
  // interpolated ghost-plane values; the set's first two symmetric tensors are then
  // g-tilde and A-tilde (the Z4c set).
  void CartoonFillSet(MeshBlockPack *pmbp, const CartoonVarSet &vs,
                      DvceArray5D<Real> src, DvceArray5D<Real> dst, int npt,
                      const CartoonParity &par, bool direct_negx, bool algc);
  // u_adm -> u_adm_cart (no algebraic constraints: the ADM set has none)
  void CartoonFillADM(MeshBlockPack *pmbp);
  // Fatal exit for a problem generator that cannot run under 2D Cartoon (no-op in 3D)
  void CartoonUnsupported(const std::string &what) const;
  // D-002 axis boundary on an arbitrary (fine) array: fill i < is from i >= is with the
  // parity signs, on MeshBlocks whose inner x1 face is the mesh boundary.
  void CartoonAxisBC(MeshBlockPack *pmbp, DvceArray5D<Real> u,
                     const CartoonParity &par);
  // K12 stencil clamps per ghost plane l: {l, n_low, n_high, n_total}; test/diagnostic
  // only (host loop; the production kernels do not count).
  void CartoonClampStats(MeshBlockPack *pmbp, int npt,
                         std::vector<std::array<int, 4>> &per_l);
  bool IsCartoon() const { return opt.cartoon != CartoonMode::none; }

  Z4c_AMR *pamr;
  std::vector<std::unique_ptr<CompactObjectTracker>> ptracker;
  std::vector<std::unique_ptr<FastFlow>> pfastflow;
  std::vector<std::unique_ptr<HorizonDump>> phorizon_dump;
  std::unique_ptr<DriftControl> pdrift_control;

  /*
  std::list<CartesianGrid> horizon_dump;
  Real horizon_dt;
  Real horizon_last_output_time;
  std::vector<Real> horizon_extent; // radius for dumping data in a cube
  std::vector<int> horizon_nx;  // number of points in each direction
  */
  // TODO(@hzhu): think about how to automatically trigger common horizon
  // maybe have a horizon dump object to save all the space here
  // same for the waveform.
 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Z4c
  void CartoonSetup(ParameterInput *pin);
  void CartoonAlloc();
};

} // namespace z4c
#endif //Z4C_Z4C_HPP_
