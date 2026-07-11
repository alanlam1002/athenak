# CFC module development notes

This file tracks the design and implementation status of `src/cfc/`, the
conformally-flat-condition (CFC/XCFC) metric solver for AthenaK. It is meant to be
kept up to date as the module evolves — treat it as the running log of *why* the
code looks the way it does, not just *what* it does.

## Goal

Solve the Einstein field equations under the conformal-flatness approximation
(algebraic constraint solve every step, replacing free evolution of the spacetime)
sourced from the matter stress-energy tensor computed by `dyn_grmhd`, and feed the
result into the ADM metric container (`adm::ADM`) that `dyn_grmhd` already reads for
its Riemann solver and conserved-to-primitive conversion.

## References

- Cheong et al. 2021, "Gmunu" (arXiv:2012.07322), sec. 2.6 — the exact XCFC equation
  set (eqs. 71-76), the 6-step solve order, and the boundary conditions this module
  is modeled after.
- Shibata 1999 (arXiv:gr-qc/9905058), sec. 3 — the scalar decomposition that turns a
  "vector Laplacian + 1/3 grad-div" equation into independent flat scalar Poisson
  equations. Both CFC vector equations (for `X^i` and `beta^i`) are exactly this
  form.
- `src/gravity/{gravity,mg_gravity}.{hpp,cpp}` — the structural template this module
  mirrors (a thin physics orchestrator + Multigrid/MultigridDriver subclass pairs).

## Status: skeleton, plus `cfc_reconstruct.cpp` and all four multigrid solvers
## (`MGCFCVectorPoisson`/`MGCFCScalarPoisson`/`MGCFCConformalFactor`/`MGCFCLapse`)
## implemented

All classes, member variables, and function signatures exist and the module builds
into the project (registered in `src/CMakeLists.txt`, wired into `MeshBlockPack` and
the shared `NumericalRelativity` task graph — see the task-graph design-decision
bullet below). `src/cfc/cfc_reconstruct.cpp`'s 4 free functions
(`ComputeADualFromX`, `ReconstructVectorFromPotentials`, `AssembleConformalMetric`,
`AssembleLapseShiftK`), `MGCFCVectorPoisson[Driver]`/`MGCFCScalarPoisson[Driver]`
(the two linear/`nvar_`-decoupled elliptic solvers, `P_i` and `eta`), and
`MGCFCConformalFactor[Driver]`/`MGCFCLapse[Driver]` (the two nonlinear/screened-affine
solvers, `psi` and `alpha*psi`) are now implemented (open items 1-3, below) —
everything else (`cfc.cpp`'s own bodies: constructor, `AssembleVectorSource`,
`RescaleMatterSources`) is still a `// TODO(cfc): ...` stub. `cpplint` was run
against all files; no new warning categories beyond what's already accepted in
`gravity`'s own files (include-path style, `public:`/`private:` indent). **Not
compiled**: this sandbox's system GCC (7.5.0) is below the bundled Kokkos 4.7.2's
minimum (8.2.0), so all of this has only been verified by cpplint and careful manual
cross-checking against
`z4c_calcrhs.cpp`/`z4c_adm.cpp`/`adm.cpp`/`mg_gravity.hpp`/`.cpp`'s equivalent
patterns — a real compile against a supported toolchain is still owed before
trusting any of it further, and is *especially* owed for items 3's `RHS(u)`
sign/scaling derivation and its new `TransferCoeffToRoot` MPI path (see item 3's
findings below), neither of which reuses an already-exercised code path the way
items 1-2 did. The pure-virtual method requirements on `Multigrid`/`MultigridDriver`
do give a strong, free structural check once a real compile is possible: miss one and
the class stays abstract.

## The XCFC equations and solve order (Gmunu eqs. 71-76)

Conformal decomposition `gamma_ij = psi^4 f_ij` (`f_ij` = flat metric = `delta_ij` in
Cartesian), maximal slicing `K = 0`. Matter sources `U`, `S_i`, `S_ij`, rescaled as
`U-tilde = psi^6 U`, `S-tilde_i = psi^6 S_i`, `S-tilde = psi^6 S`.

**`U`/`S_i` and `S` (the trace of `S_ij`) are not equally "free," though.** Mirroring
`dyn_grmhd.cpp`'s `DynGRMHD::SetTmunu` (lines 461/463/464-468): `E = (tau+D)/
sqrt(gamma)` and `S_i = S_i_cons/sqrt(gamma)` are algebraically exact functions of
the *evolved conserved state alone* (`pmy_pack->pmhd->u0`) — no primitives needed,
so `U-tilde`/`S-tilde_i` can be built directly from `u0` and rescaled by the new
`psi^6` as soon as `psi` is available. But the trace of `S_ij` needs velocity and
pressure — primitives, which require a conserved-to-primitive (con2prim) recovery
using the metric. **CFC builds `U`/`S_i` from `pmy_pack->pmhd->u0` directly, NOT
from `MeshBlockPack::ptmunu`**: `Tmunu` is only populated by `MHD_SetTmunu`, which
`dyn_grmhd.cpp`'s `QueueDynGRMHDTasks()` only queues `if (pz4c != nullptr)` — but
CFC's primary use case has no z4c free evolution, so `ptmunu` may not exist at all
in that configuration. See the design-decision bullets below for how the single
con2prim needed for the trace term is obtained without duplicating dyn_grmhd's own
per-stage con2prim.

1. **Vector potential `X^i`**: `Delta X^i + (1/3) D^i(D_j X^j) = 8 pi f^ij S-tilde_j`
   (source built directly from `pmy_pack->pmhd->u0`, right after this stage's hydro
   flux+source update).
2. **`Adual^ij`** (algebraic): `Adual^ij ~= D^i X^j + D^j X^i - (2/3) D_k X^k f^ij`
3. **Conformal factor `psi`** (nonlinear): `Delta psi = -2 pi U-tilde psi^-1 - (1/8) Ahat^2 psi^-7`.
   Immediately followed by an early, partial write of `psi4`/`g_dd` into
   `padm->u_adm` (`cfc::AssembleConformalMetric`) — the shared con2prim in step 4
   needs it.
4. **Rebuild the trace source, using the con2prim dyn_grmhd already runs.**
   `dyn_grmhd`'s own per-stage `MHD_C2P` task is made to depend on step 3 (see the
   task-graph design-decision bullet below), so by the time this step runs,
   `pmy_pack->pmhd->w0` (density, pressure, velocity) is already fresh against the
   `g_dd` step 3 just wrote — no second con2prim call. `S-tilde` (the trace of
   `S_ij`) is rebuilt from that `w0` (`rho*h*W^2*v^2 + 3*P`, densitized by the new
   `psi^6`); `U-tilde`/`S-tilde_i` don't need rebuilding (see above).
5. **Lapse x psi `alpha*psi`** (nonlinear): depends on `psi`, `Ahat^2`, and the
   post-con2prim rescaled sources (especially `S-tilde`, the trace term).
6. **Shift `beta^i`**: same vector-equation form as step 1, different source.

Final step: write `vK_dd`/`alpha`/`beta_u` into `padm->u_adm`
(`cfc::AssembleLapseShiftK`) — `psi4`/`g_dd` were already written after step 3.

Steps 1 and 6 each decompose (Shibata sec. 3) into a vector potential `P_i`
(`Delta P_i = S_i`, 3 fully independent components) plus a scalar potential `eta`
(`Delta eta = -S_i x^i`), reconstructed as
`V^j = (7/8) P_j - (1/8)(eta,_j + P_k,_j x^k)`.

## File layout

```
src/cfc/
  cfc.hpp / cfc.cpp                     # orchestrator; owns all 6 multigrid drivers
                                         # and the 6-step control flow, queued into
                                         # the NumericalRelativity task graph via
                                         # QueueCFCTasks() (not a direct Solve() call)
  mg_cfc_vector_poisson.hpp/.cpp        # P_i solver (linear, nvar=3), shared by
                                         # both X^i and beta^i (2 instances)
  mg_cfc_scalar_poisson.hpp/.cpp        # eta solver (linear, nvar=1), shared by
                                         # both X^i and beta^i (2 instances)
  mg_cfc_conformal_factor.hpp/.cpp      # psi solver (nonlinear scalar, nvar=1)
  mg_cfc_lapse.hpp/.cpp                 # alpha*psi solver (nonlinear scalar, nvar=1)
  cfc_reconstruct.hpp/.cpp              # free-function Kokkos kernels: Adual^ij from
                                         # X^i, vector reconstruction from (P_i, eta),
                                         # final ADM assembly into padm->u_adm
```

6 multigrid driver instances total: `pmgd_px`, `pmgd_etax` (for `X^i`), `pmgd_pbeta`,
`pmgd_etabeta` (for `beta^i`), `pmgd_psi`, `pmgd_alpha`.

## Key design decisions (and why)

- **Reuses `src/multigrid/` unchanged.** The linear equations (`P_i`, `eta`) reuse
  the existing generic templated `Smooth`/`CalculateDefect`/`CalculateFASRHS`
  helpers with a plain flat 7-point Laplacian stencil (same form as
  `gravity::GravityStencil`). The two nonlinear equations (`psi`, `alpha*psi`,
  self-coupled through `psi^-7`/`psi^-8` terms) override `SmoothPack`/
  `CalculateDefectPack`/`CalculateFASRHSPack` directly with hand-written
  Newton-Gauss-Seidel point relaxation instead, since the shared template assumes a
  constant-diagonal linear operator. No changes needed to the multigrid module
  itself either way.

- **Vector/tensor quantities use `AthenaTensor`, not raw `DvceArray5D`.** Mirrors
  `adm::ADM`/`z4c::Z4c`: a flat `DvceArray5D<Real>` owns the actual storage, and an
  `AthenaTensor` view (`InitWithShallowSlice`) provides tensor-index access on top of
  it. Applies to `x_u`, `beta_u` (rank-1, `TensorSymm::NONE`), `a_dd` (rank-2,
  `TensorSymm::SYM2`), `s_tilde_d` (rank-1), and the decomposed vector potentials
  `p_x`/`p_beta` (rank-1). Genuine scalars (`psi`, `alpha_psi`, `a_sq`, `u_tilde`,
  `s_tilde`, `eta_x`, `eta_beta`) stay plain `DvceArray5D<Real>`, same as
  `gravity::Gravity::phi`.

- **`P_i` and `eta` are solved as two separate sequential multigrid solves, not one
  combined 4-channel solve.** `Delta P_i = S_i` (3 components) and
  `Delta eta = -S_i x^i` are mathematically independent of each other, but the
  workflow solves `P_i` to completion first (`MGCFCVectorPoissonDriver`, `nvar_=3`),
  then builds and solves `eta`'s equation second (`MGCFCScalarPoissonDriver`,
  `nvar_=1`) using the same known source `S_i`. This is a deliberate implementation
  choice (explicit user direction), not a correctness requirement — it keeps the
  vector quantity `P_i` cleanly represented as an `AthenaTensor` throughout, rather
  than packed into an opaque 4-component array alongside the unrelated scalar `eta`.

- **Boundary conditions reuse existing multigrid BCs, no new BC code.**
  `X^i`/`beta^i` (and their decomposed `P_i`/`eta`) use `BoundaryFlag::mg_zerofixed`
  (simple Dirichlet-zero, Gmunu eqs. 79-80). `psi`/`alpha*psi` are solved as
  deviations from 1 (`delta_psi = psi - 1`, etc.) using `BoundaryFlag::mg_multipole`
  (already-implemented isolated/asymptotically-flat 1/r falloff, Gmunu eqs. 77-78).

- **`CFC` queues its steps into the shared `NumericalRelativity` task graph
  (`QueueCFCTasks()`), it does NOT call `Driver::Execute()` directly the way
  `gravity::Gravity` does.** This was the original design (mirroring
  `gravity::Gravity::pmgd->Solve()`, called once per stage right before
  `ExecuteTaskList(pmesh, "stagen", stage)`), but it can't express the ordering XCFC
  actually needs: CFC's vector-potential/`psi` steps must run *after* this stage's
  hydro flux+update, and the trace-source rebuild (step 4) must share dyn_grmhd's
  own per-stage con2prim rather than duplicating it. Since `"stagen"` is one opaque
  `TaskList` from `driver.cpp`'s point of view, nothing can be interleaved inside it
  from outside — exactly the problem `dyn_grmhd`/`z4c` solve by queuing individual,
  dependency-tracked tasks into `NumericalRelativity` instead (see `tasklist/
  numerical_relativity.hpp`'s `CFC_*` `TaskName` values and `Phys_CFC`). The
  resulting per-stage order, all inside `"stagen"`:
  ```
  MHD_CopyU -> MHD_Flux -> ... -> MHD_ExplRK -> MHD_AddSrc
    -> CFC_SolveVecX (steps 1-2) -> CFC_SolvePsi (step 3, writes psi4/g_dd)
    -> [B-field CT/restrict/send/recv/BCS/Prolong, unchanged, running in parallel]
    -> MHD_C2P (single con2prim; required dep {MHD_Prolong}, optional dep
       {Z4c_Excise, CFC_SolvePsi} -- see dyn_grmhd.cpp)
    -> CFC_RescaleSrc (step 4, no con2prim call -- just reads the w0 MHD_C2P wrote)
    -> CFC_SolveLapse (step 5) -> CFC_SolveShift (step 6) -> CFC_AssembleFinal
    -> MHD_Newdt (optional dep on CFC_AssembleFinal)
  ```
  `MHD_C2P`/`MHD_Newdt`'s CFC dependencies are *optional*
  (`NumericalRelativity::AddExtraDependencies`), so a run without a `<cfc>` block
  produces the exact same task graph as before this change. Because
  `NumericalRelativity::AssembleNumericalRelativityTasks(tl_map)` (called from
  `MeshBlockPack::AddPhysics`) is what invokes `QueueCFCTasks()`, `pcfc` must be
  constructed *before* that call — `meshblock_pack.cpp` was reordered so the `<cfc>`
  block now sits right after `ptmunu`'s construction, ahead of
  `pnr = new NumericalRelativity(...)`. Runs every stage for now — no `\Delta n`-cycle
  cadence control (a pure performance optimization described in Gmunu sec. 2.6.2)
  has been added yet.

- **Exactly one con2prim call per stage, shared between `dyn_grmhd` and `CFC`.**
  An earlier version of this design had `CFC::RescaleMatterSources` call
  `pmy_pack->pdyngr->ConToPrim(pdriver, stage)` itself to get fresh primitives for
  the trace source — but `dyn_grmhd`'s own per-stage `MHD_C2P` task already does a
  con2prim, so that duplicated the (expensive, nonlinear) inversion. Fixed by making
  `MHD_C2P` itself depend on `CFC_SolvePsi` (optionally, see above) and having
  `CFC_RescaleSrc` depend on `MHD_C2P` in turn: CFC no longer runs its own con2prim
  at all, it just waits for dyn_grmhd's and reads the `pmy_pack->pmhd->w0` that call
  already populated. This is also why `AssembleADM` is split into two calls
  (`cfc::AssembleConformalMetric` right after step 3, `cfc::AssembleLapseShiftK` at
  the very end): `PrimitiveSolverHydro::ConsToPrim`
  (`src/eos/primitive_solver_hyd.hpp:308`) reads `padm->adm.g_dd` directly, so `g_dd`
  must be written *before* `MHD_C2P` runs, not deferred to the final assembly step.

## Integration points outside `src/cfc/`

- `src/mesh/meshblock_pack.hpp` / `.cpp`: `MeshBlockPack::pcfc`, constructed when a
  `<cfc>` input block exists (requires `padm` and `ptmunu` to already exist, i.e. a
  `<z4c>` or `<adm>` block plus an `<mhd>` block), and constructed *before*
  `pnr = new numrel::NumericalRelativity(...)` / `AssembleNumericalRelativityTasks`
  runs (moved from its original spot after `gravity`'s construction), since that
  call is what invokes `pcfc->QueueCFCTasks()`.
- `src/tasklist/numerical_relativity.hpp`/`.cpp`: `CFC_SolveVecX`, `CFC_SolvePsi`,
  `CFC_RescaleSrc`, `CFC_SolveLapse`, `CFC_SolveShift`, `CFC_AssembleFinal`
  `TaskName` values (appended after `Z4c_NTASKS`) and a `Phys_CFC`
  `PhysicsDependency`; `AssembleNumericalRelativityTasks(tl_map)` calls
  `pmy_pack->pcfc->QueueCFCTasks()` alongside `pdyngr`/`pz4c`'s equivalents.
- `src/dyn_grmhd/dyn_grmhd.cpp`: `MHD_C2P`'s and `MHD_Newdt`'s `QueueTask` calls
  take `CFC_SolvePsi`/`CFC_AssembleFinal` as *optional* dependencies (alongside the
  existing `Z4c_Excise`), so con2prim/new-dt wait on CFC's outputs only when a
  `<cfc>` block is present.
- `src/driver/driver.cpp`: no longer calls into `cfc` directly — `CFC`'s tasks now
  run entirely inside `ExecuteTaskList(pmesh, "stagen", stage)`.
- `src/CMakeLists.txt`: the 6 new `.cpp` files are registered.
- `pmy_pack->pmhd->u0`/`w0`/`bcc0`: `u0` (conserved state) is read directly by
  `AssembleVectorSource`/matter-source building; `w0` (primitives) is read by
  `RescaleMatterSources` after `MHD_C2P` has populated it. `cfc` never calls
  `ConToPrim` itself.

## Open items / next steps

1. ~~Implement `cfc_reconstruct.cpp`'s kernels first (pure Kokkos, no multigrid
   dependency)~~ **Done.** `ComputeADualFromX`/`ReconstructVectorFromPotentials`
   are each a `template <int NGHOST>` file-local implementation (`ComputeADual-
   FromXImpl`/`ReconstructVectorFromPotentialsImpl`, anonymous namespace) behind a
   `switch(indcs.ng)` dispatch in the public entry point, mirroring
   `Z4c::ADMToZ4c`'s shape. `eta` (a plain `DvceArray5D<Real>`, not an
   `AthenaTensor`) is locally wrapped via `AthenaTensor<...,0>::
   InitWithShallowSlice(eta, 0)` so it can go through the same generic
   `Dx<NGHOST>` scalar overload as everything else, rather than hand-rolling a
   separate finite-difference path for it.
   `AssembleConformalMetric`/`AssembleLapseShiftK` are plain untemplated
   `par_for`s over `is..ie` (no derivatives, so no `NGHOST` needed).
   **`vK_dd` convention locked in**: `vK_dd = Adual_ij / psi^2` exactly as
   already stated in `cfc_reconstruct.hpp`'s docstring (K=0 maximal slicing) —
   implemented as written there, not re-derived from the Gmunu paper during this
   pass (see the "Not compiled" caveat above for what verification this did and
   didn't get).
2. ~~Implement `MGCFCVectorPoisson`/`MGCFCScalarPoisson`'s `SmoothPack`/
   `CalculateDefectPack`/`CalculateFASRHSPack`~~ **Done.** Turned out to be more
   than "copy `MGGravity`'s pattern" — four real findings, all recorded in the
   plan file's addendum and worth restating here so item 3 doesn't rediscover
   (or misapply) them:
   - `Multigrid::Smooth`/`CalculateDefect`/`CalculateFASRHS` (`multigrid.hpp`)
     hardcode variable index `0`. `MGCFCVectorPoisson` (`nvar_=3`: `P_x,P_y,P_z`,
     mutually independent) calls each once per channel via a rank-preserving
     `Kokkos::subview(..., std::make_pair(v,v+1), ...)` — the same idiom
     `AthenaTensor<...,1>::InitWithShallowSlice` already uses. Factored into 3
     file-local template helpers (`SmoothChannels`/`CalculateDefectChannels`/
     `CalculateFASRHSChannels`, anonymous namespace, `mg_cfc_vector_poisson.cpp`)
     so the 3 `*Pack` methods stay short. **`MGCFCScalarPoisson` (`nvar_=1`,
     i.e. eta) needs none of this** — it's a literal, unmodified port of
     `MGGravity`'s 3 `*Pack` methods.
   - The Octet functions (`SmoothOctet`/`CalculateDefectOctet`/
     `CalculateFASRHSOctet`) have the identical hardcoded-index-0 problem in
     `MGGravityDriver`'s reference implementation; `MGCFCVectorPoissonDriver`'s
     versions loop `v=0..2` over `oct.U(v,...)`/`Src`/`Def` (those accessors
     already take `v`). `MGCFCScalarPoissonDriver`'s stay `v=0`-only.
   - `ProlongateOctetBoundariesFluxCons` was dropped entirely from
     `MGCFCVectorPoisson` (removed from `mg_cfc_vector_poisson.hpp`, not just
     left unimplemented) — `MultigridDriver`'s base default is already
     `nvar_`-generic trilinear prolongation; gravity's override exists only for
     exact flux conservation, which these auxiliary potentials don't need.
   - `LoadPoissonSource`/`RetrieveSolution` were declared taking `AthenaTensor<
     ...>&`, but `Multigrid::LoadSource`/`RetrieveResult` require a genuine
     `DvceArray5D<Real>&` (non-templated) — an `AthenaTensor`'s
     `Kokkos::subview`-backed storage doesn't type-check against that. Both
     signatures now take the raw backing array instead (`mg_cfc_vector_poisson.
     hpp`); `cfc.hpp` grew a `u_p_src`/`p_src` member pair so
     `AssembleVectorSource`'s output has real storage to hand
     `LoadPoissonSource` (see item 4's updated description below).
   - Also discovered along the way: `MultigridDriver`'s *base* constructor
     already defaults every non-periodic `mg_mesh_bcs_[f]` to
     `BoundaryFlag::mg_zerofixed` (`multigrid_driver.cpp`) — exactly what `P_i`
     needs, so neither driver constructor sets boundary flags explicitly at
     all (unlike `MGGravityDriver`, which overrides this only for its
     configurable `mg_bc` input option that CFC has no equivalent of).
   - Not yet done: reusing the previous stage's converged solution as the next
     stage's initial guess (`LoadFinestData`, matching gravity's iterative-mode
     branch) — skipped for this pass; `Solve()` currently just runs a cold
     V-cycle from whatever `u_[nlevel_-1]` already holds (which may itself
     retain the prior stage's state, since nothing explicitly zeros it between
     calls — not verified either way).
3. ~~Implement the two nonlinear Newton-Gauss-Seidel solvers
   (`MGCFCConformalFactor`, `MGCFCLapse`) last — the highest-risk piece.~~ **Done.**
   Both are `nvar_=1` scalars, so none of item 2's per-channel subview treatment
   applied. Four real findings, all worth restating here so item 4 doesn't
   rediscover (or misapply) them:
   - **Finding A**: eq. 74 (lapse) is *not* actually nonlinear once `psi`/`Ahat^2`
     are already fixed (steps 2-3 have already converged them by the time this
     solve runs) — the bracketed factor `K(x)` depends only on those known fields,
     not on the unknown `alpha*psi` itself, so `Delta(u+1) - K(x)*(u+1) = 0` is
     affine in `u`. It still can't reuse the generic `Smooth<StencilOp>` template
     (the per-point diagonal `6 + dx^2*K(x)` isn't constant), but the "Newton" step
     is an *exact* one-step Gauss-Seidel solve, not an approximate linearization —
     `mg_cfc_lapse.hpp`'s header docstring was corrected to reflect this (it
     previously said "nonlinear...self-coupled", copied from the conformal-factor
     case). `MGCFCConformalFactor` (eq. 73) *is* genuinely nonlinear (self-coupled
     through `psi^-7`) and needed a true per-point Newton linearization, with a
     `psi_floor_` clamp (new `<cfc>` param, default 0.05) so a bad step can't drive
     `psi <= 0` (where `psi^-7` is ill-defined), plus an `mg_omega_psi_` damping
     factor (default 1.0) separate from the linear solvers' own `mg_omega`.
   - **Finding B**: `Multigrid::coeff_`/`ncoeff_` is real (threaded through
     `Smooth`/`CalculateDefect`/`CalculateFASRHS`'s signatures) but was completely
     unwired before this — grepping confirmed gravity (the only prior caller) never
     allocates it (`ncoeff_` stays 0, `coeff_` stays default-constructed/zero-sized)
     and its `GravityStencil::Apply` ignores the `coeff` parameter entirely. Fixed
     entirely inside `MGCFCConformalFactor`/`MGCFCLapse`'s own constructors (no
     `src/multigrid/` changes): each sets its own `ncoeff_` (2 and 3 respectively)
     and manually loops `l = 0..nlevel_-1` doing the same `Kokkos::realloc` the base
     ctor already does for `u_`/`src_`/`def_`. Both `Utilde`/`Ahat^2` (psi) and
     `Utilde+2*Stilde`/`psi`/`Ahat^2` (lapse) live in `coeff_`, *not* `src_` via
     `LoadSource` — `src_` is exactly what the V-cycle's FAS machinery restricts and
     tau-corrects, which would corrupt these physically-fixed fields at coarser
     levels. `RestrictCoefficients()` (unlike `src_`, restricted automatically every
     V-cycle descent) had to be called explicitly, once, right after loading, before
     `SetupMultigrid()`.
   - **Finding C**: the distributed root grid (`mgroot_`) never received `coeff_`
     either, and this is *not* an AMR-only edge case — `MultigridDriver::
     TransferFromBlocksToRoot` aggregates every rank's coarsest per-block cell into
     `mgroot_` via `MPI_Allgatherv` for **any** multi-meshblock mesh, but only
     transfers `src_`/`u_`, never `coeff_`. Fixed with a new CFC-local
     `TransferCoeffToRoot()` helper on each driver, duplicating the relevant
     non-octet-parented slice of that MPI logic for `coeff_`'s channels instead —
     zero changes to `src/multigrid/`, at the cost of ~50 lines of duplicated
     bookkeeping that must be kept in sync by hand if `TransferFromBlocksToRoot`
     itself is ever restructured again. **This is new MPI code that can only be
     validated on an actual multi-rank, multi-meshblock run** — flagged for the
     first real-compiler pass.
   - **Finding D**: `MGOctet` has no coefficient storage at all — genuinely
     AMR-specific (octets only exist `if (nreflevel_ > 0)`), and extending it would
     be a materially bigger core change than Finding C's, not required for the
     near-term goal (BU0/BU8 verification, uniform-resolution test cases). Guarded
     rather than implemented: both drivers' `Solve()` fatal-errors if
     `nreflevel_ > 0` right after `PrepareForAMR()`; the `SmoothOctet`/
     `CalculateDefectOctet`/`CalculateFASRHSOctet` overrides exist only as
     `std::exit(EXIT_FAILURE)` stubs to satisfy the pure-virtual contract, since the
     `Solve()`-level guard means they should never actually be reached. See new open
     item 3b below.
3b. Extend `MGOctet` (`src/multigrid/multigrid.hpp`) with its own `coeff_`-style
    storage and restriction/boundary-exchange path so `MGCFCConformalFactor`/
    `MGCFCLapse` can support AMR-refined meshes (currently fatal-errors if
    `nreflevel_ > 0`, see item 3 Finding D above). Touches `InitializeOctets`, the
    per-octet restriction/boundary paths, and the already-suspiciously-pre-sized
    `cbuf_`/`cbufold_` (`multigrid_driver.cpp:431`, sized to
    `max(nvar_, max(ncoeff_,1))` — possibly anticipated by an earlier author but
    never finished). Not required for BU0/BU8 (item 5, uniform-resolution test
    cases) — defer until AMR+CFC is an actual need.
4. Fill in `cfc.cpp`'s constructor (sizing all arrays including the new
   `u_p_src`/`eta_src`, wiring `InitWithShallowSlice`, constructing the 6
   driver instances), `AssembleVectorSource(bool for_shift)` (signature no
   longer takes `p_src`/`eta_src` as parameters — writes directly into the
   `p_src`/`eta_src` members now that they exist; building `S-tilde_i` directly
   from `pmy_pack->pmhd->u0`, mirroring `dyn_grmhd.cpp`'s `SetTmunu` lines
   461/463 but without going through `Tmunu`), and `RescaleMatterSources`'s
   trace-source recomputation from the `w0` `MHD_C2P` already populated
   (reference `SetTmunu`'s `S_dd` formula, lines 464-468, for the exact trace
   to mirror with fresh primitives). `SolveVectorPotential`/`SolveShift`'s
   `LoadPoissonSource`/`RetrieveSolution` calls take the raw `u_p_x`/`u_p_beta`/
   `u_p_src` arrays, not the `p_x`/`p_beta`/`p_src` `AthenaTensor` views (item
   2's Finding 4) — the TODO comments in `cfc.cpp` already reflect this.
5. Verify against the Gmunu paper's BU0/BU8 test cases once the above is complete —
   out of scope until equation bodies exist.
6. Consider the `\Delta n`-cycle solve cadence (Gmunu sec. 2.6.2) as a later
   performance optimization.
