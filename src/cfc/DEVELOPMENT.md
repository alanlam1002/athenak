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

## Status: feature-complete (all equation bodies implemented); first full run of a
## CFC test case completes (10 cycles, dyngr_tov/isotropic TOV star) but is not
## yet correct -- NaNs at the x3=0 reflecting boundary under active investigation

All classes, member variables, and function signatures exist and the module builds
into the project (registered in `src/CMakeLists.txt`, wired into `MeshBlockPack` and
the shared `NumericalRelativity` task graph — see the task-graph design-decision
bullet below). `src/cfc/cfc_reconstruct.cpp`'s 4 free functions
(`ComputeADualFromX`, `ReconstructVectorFromPotentials`, `AssembleConformalMetric`,
`AssembleLapseShiftK`), `MGCFCVectorPoisson[Driver]`/`MGCFCScalarPoisson[Driver]`
(the two linear/`nvar_`-decoupled elliptic solvers, `P_i` and `eta`),
`MGCFCConformalFactor[Driver]`/`MGCFCLapse[Driver]` (the two nonlinear/screened-affine
solvers, `psi` and `alpha*psi`), and `cfc::CFC` itself (constructor,
`AssembleVectorSource`, `RescaleMatterSources`, the full ghost-exchange task graph)
are now all implemented (open items 1-4, below) — no `// TODO(cfc): ...` stubs
remain anywhere in `src/cfc/`. `cpplint` was run against all files; no new warning
categories beyond what's already accepted in `gravity`'s own files (include-path
style, `public:`/`private:` indent).

**First real compile attempted** (`~/athenak_cfc/cfc_sakura.sh`, Sakura cluster,
`PROBLEM=dyn_grmhd/dyngr_tov`, Intel oneAPI `mpiicpx`/`icx` 2024.0 over a
`gcc/13`-toolchain, Kokkos 4.7.2 — this sandbox's own system GCC (7.5.0) is still
below Kokkos's minimum, but a working module-based toolchain exists on Sakura),
surfaced two compile errors — full detail, root cause, and fix in open item 7
below. **Both are now fixed**: all six `cfc/*.cpp` object files compile cleanly,
and no error touching `src/cfc` or `src/multigrid/multigrid.hpp` appears anywhere
in the rebuild log (76 files built past that point). Everything below the "Not
compiled" caveats this status line replaces has thus cleared its first real
compile: item 3's `RHS(u)` sign/scaling derivation and its `TransferCoeffToRoot`
MPI path, and item 4's matter-source algebra (`RescaleMatterSources`' `S_dd`/trace
contraction, Finding E) and its 32-node ghost-exchange task graph are now known to
at least type-check and link against the rest of the project; they still haven't
been *run*. The pure-virtual method requirements on `Multigrid`/`MultigridDriver`
gave one free structural check along the way: nothing reported a class staying
abstract from a missed pure-virtual override.

**Full project build initially still failed**, but on something unrelated to CFC
or `multigrid.hpp`: the rebuild crashed (exit 139, not a normal diagnostic) inside
Intel `icx` 2024.0.2's optimizer while compiling `src/multigrid/multigrid_driver.cpp`
— a file untouched by either fix above, and not reached by the two earlier
(pre-fix) build attempts. Isolated by hand (recompiling just that file with
different flags outside the normal `make` invocation): reproduced at `-O3` both
with and without `-march=native -mtune=native`; did **not** reproduce at `-O2` or
`-O1`. Diagnosed as a pre-existing Intel `icx` 2024.0.2 optimizer bug on that
file, independent of this session's other changes.

**Fixed by switching toolchain, no source changes**: reran the same isolated
compile of `multigrid_driver.cpp` with `intel/2025.3` + `impi/2021.17` (both
available as Sakura modules) instead of `intel/2024.0` + `impi/2021.11` — compiled
cleanly at `-O3 -march=native -mtune=native`, same as every other flag
combination tried. `cfc_sakura.sh` was updated to load `intel/2025.3`/
`impi/2021.17` (and the corresponding `INTEL_ROOT`/`IMPI_ROOT` paths) instead of
the 2024.0 versions. **A full rebuild from clean then succeeded end to end**:
`athena` links (15MB executable, `~/athenak_cfc/build_cfc/src/athena`) with
`PROBLEM=dyn_grmhd/dyngr_tov`, exercising every line of `src/cfc/` and the fixed
`multigrid.hpp` through to a real linked binary — the first time this has
happened. This was a genuine upstream Intel compiler bug fixed between 2024.0.2
and 2025.3, not anything in this project's code; worth mentioning to
`multigrid.hpp`'s owner (or whoever maintains the Sakura build scripts) in case
other in-progress work is still pinned to `intel/2024.0` for this project.

Full physics verification (BU0/BU8, open item 5) is the next milestone — it was
blocked only on getting a working binary, which now exists.

**First actual run attempted**, `inputs/dyn_grmhd/cfc_tov.athinput` (new file,
based on `whisky_tov.athinput`: `isotropic=true` so `dyngr_tov.cpp` sets up the
exact analytic conformally-flat/K=0 TOV solution rather than the non-CFC-compatible
Schwarzschild-coordinate branch, `v_pert=0` for a static-equilibrium first check,
empty `<cfc>` block, `nlim=10` for a fast smoke test). The initial-data
bootstrapping question that prompted this test (primitives-only initial data needs
psi before it can build `Ũ=ψ⁶U`, but psi isn't solved yet) turned out to be a
non-issue for TOV specifically: `dyngr_tov.cpp`'s `isotropic=true` branch sets
`adm.psi4` from the exact closed-form isotropic TOV mapping *before* the first
`Prim2Con` ever runs, so the very first conserved state already has the correct
psi baked in — no Picard iteration needed here (deferred to whenever non-analytic
initial data, e.g. a LORENE/KADATH binary, is attempted — see open item 5's
note). Three more bugs surfaced and were fixed getting to a completing run — full
detail in open item 8 below. The run now completes all 10 cycles without crashing,
but is not yet correct: `NANS_IN_CONS` C2P errors appear at cycle 0, localized to
grid points at `k=4` (the first physical cell above the `x3=0` reflecting
boundary) where `g_dd`/`psi4` come out `-nan`.

**Still unresolved as of this writing** — see open item 9 for the full,
multi-round investigation, including one claim that turned out to be wrong
(`MultigridDriver::PhysicalBoundary` was reported dead code; it is not — a bad
grep exclusion hid its actual call sites) and one real fix that landed but did
not resolve the crash (`mg_mesh_bcs_[face]` must hold a multigrid-internal
`BoundaryFlag` value — `mg_zerograd`, not the ordinary mesh `BoundaryFlag::
reflect` — fixed in all 4 CFC driver constructors). Isolated with temporary
instrumentation (added, used, then removed — not left in the tree) to the root
grid specifically: even with clean, sane inputs (`u_tilde`/`a_sq` both NaN-free
going in), `psi` at the root grid's single interior cell is already partly NaN
immediately after the very first V-cycle solve, before anything else in CFC
touches it. Root cause not yet found. Full detail in open item 9.

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
   `S_ij`) is rebuilt from that `w0`, densitized by the new `psi^6`. **Not** the
   pure-fluid closed form `rho*h*W^2*v^2 + 3*P` this bullet originally said (that
   drops the magnetic-field contribution) — `RescaleMatterSources` instead mirrors
   `SetTmunu`'s full `S_dd` formula (including the `B_d`/`Bv`/`bsq` magnetic terms,
   lines 476-479) inline, then contracts to the trace via the existing
   `adm::Trace(detginv, g_dd components, S_dd components)` utility (`coordinates/
   adm.hpp`) rather than a hand-derived closed form (see item 4's Finding E).
   `U-tilde`/`S-tilde_i` don't need rebuilding (see above).
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
    -> CFC_BuildSrcX (step 1: S_i from pmhd->u0; solve P_i/eta)
    -> CFC_Rest/Send/Recv/ProlongPX, ...EtaX (ghost-exchange p_x, eta_x, parallel)
    -> CFC_ReconstructX (Shibata recon -> x_u)
    -> CFC_Rest/Send/Recv/ProlongX (ghost-exchange x_u)
    -> CFC_ComputeADual (step 2: Adual^ij/Ahat^2)
    -> CFC_SolvePsi (step 3, writes psi4/g_dd)
    -> [B-field CT/restrict/send/recv/BCS/Prolong, unchanged, running in parallel]
    -> MHD_C2P (single con2prim; required dep {MHD_Prolong}, optional dep
       {Z4c_Excise, CFC_SolvePsi} -- see dyn_grmhd.cpp)
    -> CFC_RescaleSrc (step 4, no con2prim call -- just reads the w0 MHD_C2P wrote)
    -> CFC_SolveLapse (step 5)
    -> CFC_Rest/Send/Recv/ProlongPsi, ...AlphaPsi (ghost-exchange psi/alpha_psi)
    -> CFC_BuildSrcBeta (step 6: eq. 75 source; solve P_i/eta for beta^i)
    -> CFC_Rest/Send/Recv/ProlongPBeta, ...EtaBeta (ghost-exchange, parallel)
    -> CFC_ReconstructBeta (Shibata recon -> beta_u)
    -> CFC_AssembleFinal
    -> MHD_Newdt (optional dep on CFC_AssembleFinal)
  ```
  32 `CFC_*` `TaskName` entries total (see the enum in `numerical_relativity.hpp` for
  the exact list) -- expanded from an original 6-node sketch once it became clear
  each multigrid solve's *output* needs its own post-retrieve ghost-exchange round
  before `cfc_reconstruct.cpp` can safely finite-difference it (see item 4's "NGHOST-
  deep ghost exchange" design, folded into item 4's implementation). Each `Rest*`/
  `Send*`/`Recv*`/`Prolong*` quartet is a thin one-liner on `cfc::CFC` mirroring
  `z4c::Z4c::RestrictU`/`SendU`/`RecvU`/`Prolongate`'s exact shape, using one
  `MeshBoundaryValuesCC` + `coarse_*` pair per field (`pbval_px`/`coarse_u_px`, etc.,
  `cfc.hpp`) -- `is_z4c=false` throughout since CFC is not z4c.
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
- `src/tasklist/numerical_relativity.hpp`/`.cpp`: 32 `CFC_*` `TaskName` values
  (appended after `Z4c_NTASKS`, see the task-graph design-decision bullet above for
  the full list/order) and a `Phys_CFC` `PhysicsDependency`; both `NeedsPhysics`/
  `DependencyAvailable` are purely ordinal against `Z4c_NTASKS`/`CFC_NTASKS`, so no
  `.cpp` changes were needed there when the list grew from 6 to 32 entries.
  `AssembleNumericalRelativityTasks(tl_map)` calls `pmy_pack->pcfc->QueueCFCTasks()`
  alongside `pdyngr`/`pz4c`'s equivalents.
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
4. ~~Fill in `cfc.cpp`'s constructor, `AssembleVectorSource`, `RescaleMatterSources`,
   and the rest of the orchestrator~~ **Done.** Grew well beyond the original
   sketch once the "NGHOST-deep ghost exchange" design (this file, addendum-era
   text above/below) was folded in as part of the same pass, plus two follow-up
   fixes to already-committed item 2/3 code. Four more findings (E-H), recorded in
   the plan file's addendum #4 and restated here:
   - **Finding E**: `RescaleMatterSources`' trace source must mirror `SetTmunu`'s
     full `S_dd` (fluid + magnetic), contracted via `adm::Trace`, not the pure-fluid
     closed form this file originally sketched in the solve-order section above
     (now corrected there too) — dropping the magnetic terms would be silently
     wrong for any magnetized run.
   - **Finding F**: multigrid solver *outputs* that get finite-differenced
     afterward (`p_x`/`eta_x`/`p_beta`/`eta_beta`, alongside `x_u`/`psi`/
     `alpha_psi` which were already mesh-`NGHOST`-deep) must be sized at
     mesh-`NGHOST` depth, not this solver's own `ngh_` — confirmed from
     `Multigrid::RetrieveResult`'s actual body (`multigrid.cpp:420-449`), which
     already accepts an arbitrary caller depth via its `ngh` parameter and offsets
     correctly. **Required a small fix to already-committed item-2 code**:
     `MGCFCVectorPoissonDriver::RetrieveSolution`/`MGCFCScalarPoissonDriver::
     RetrieveSolution` (`mg_cfc_vector_poisson.cpp`/`mg_cfc_scalar_poisson.cpp`)
     hardcoded `mglevels_->GetGhostCells()` as that depth; both now pass
     `pmy_pack_->pmesh->mb_indcs.ng` instead.
   - **Finding G**: the `delta_psi -> psi` (`+1`) offset is `cfc.cpp`'s own
     responsibility (`RetrieveSolution` hands back the raw deviation), applied as
     one pointwise pass over the full array — safe because the field's own
     ghost-exchange round always overwrites the untouched outer ring before
     anything differentiates it.
   - **Finding H**: the load-side sibling of Finding F — `LoadMatterSource`/
     `LoadNonlinearCoefficient` (`mg_cfc_conformal_factor.cpp`) and
     `LoadMatterSource`/`LoadKnownFields` (`mg_cfc_lapse.cpp`) had the identical
     hardcoded-depth bug, discovered only once `psi` needed to flow *both*
     directions (solved, then read back in by `LoadKnownFields` for the lapse
     solve, while also needing mesh-`NGHOST` depth for the eq. 75 finite
     difference). **Required a second fix to already-committed item-3 code**: all
     four functions gained an explicit `int ngh` parameter and now use the same
     offset-aware indexing `Multigrid::LoadCoefficients` (the generic base-class
     version, `multigrid.cpp:327`) already established, instead of assuming their
     argument matches this driver's own `ngh_` exactly. Every "physical" CFC field
     (everything except `u_p_src`/`eta_src`, which stay `ngh_`-deep — pure
     `LoadSource` inputs, never differentiated or ghost-exchanged) is now
     uniformly sized at mesh-`NGHOST` depth, with every `Load*`/`Retrieve*` call
     site in `cfc.cpp` passing `indcs.ng`.

   The ghost-exchange design itself (7 `MeshBoundaryValuesCC`/`coarse_*` pairs, the
   32-node task graph) is exactly as scoped in this file's task-graph
   design-decision bullet above — see that bullet and the plan file's addendum #4
   for the full per-field rationale (which fields need exchange and why).
5. Verify against the Gmunu paper's BU0/BU8 test cases now that all equation bodies
   exist — the next real milestone, and now unblocked: `~/athenak_cfc/cfc_sakura.sh`
   produces a working `athena` binary (see "Status" above and item 7). Not started
   yet. The findings E/H above (new, unexercised numerics) are the highest-risk
   places to re-check by hand first, since a clean compile+link doesn't validate
   the physics. `inputs/dyn_grmhd/whisky_tov.athinput` (ADM-only, no `<z4c>` block,
   `<mhd>` present) is the natural base input file to add a `<cfc>` block to for a
   first single-star test.
6. Consider the `\Delta n`-cycle solve cadence (Gmunu sec. 2.6.2) as a later
   performance optimization.
7. First real-compiler pass (`cfc_sakura.sh`, Sakura, Intel `mpiicpx`/`icx` 2024.0
   over `gcc/13`, `PROBLEM=dyn_grmhd/dyngr_tov`) found two compile errors. **Both
   now fixed** (rebuild confirms: all six `cfc/*.cpp` object files compile, and no
   error touching `src/cfc` or `src/multigrid/multigrid.hpp` appears anywhere in
   the log — see "Status" above for what's still outstanding, an unrelated
   compiler crash in a different file):
   - **Own bug — root cause in `src/cfc/`, fix landed in `multigrid.hpp` since the
     missing piece was a shared accessor other physics modules could reuse**:
     `mg_cfc_conformal_factor.cpp`'s `LoadMatterSource`/`LoadNonlinearCoefficient`
     and `mg_cfc_lapse.cpp`'s equivalents call
     `mglevels_->CoeffAtLevel(mglevels_->GetNumberOfLevels()-1)` — an accessor that
     was never actually added to `Multigrid`. `Multigrid::coeff_` is a protected
     member array only reachable via `Multigrid`'s own methods or its two declared
     friends (`MultigridDriver`, `MultigridBoundaryValues`); neither CFC driver
     class is either. The existing public `LoadCoefficients(coeff, ngh)` can't
     substitute — it copies **all** `ncoeff_` channels from one combined array in a
     single call, but `u_tilde`/`a_sq` (and `MGCFCLapse`'s three known-field inputs)
     are loaded via separate calls into separate channels of the same `coeff_`
     tensor. **Fix applied**: added a one-line accessor to `Multigrid`, right next
     to the existing `GetCurrentCoefficient()`:
     ```cpp
     // multigrid.hpp, in class Multigrid's public section, after GetCurrentCoefficient()
     DualArray5D<Real>& CoeffAtLevel(int lev) { return coeff_[lev]; }
     ```
     No other changes needed — all eight call sites (`mg_cfc_conformal_factor.cpp:
     282, 303, 340, 366`, `mg_cfc_lapse.cpp:249, 270, 302, 328`) already assumed
     exactly this signature.
   - **Pre-existing bug in `multigrid.hpp`, not introduced by CFC — reported to,
     and now fixed with, the module owner's sign-off**: `Multigrid::Smooth`
     (was `multigrid.hpp:305-329`, templated on `<ViewType, StencilOp>`, defined
     inline inside the `Multigrid` class body) calls `pmy_driver_->GetCoffset()`
     (`pmy_driver_` is `MultigridDriver*`). `class MultigridDriver`'s full
     definition (where `GetCoffset()` lives) appears *later* in the same header —
     only forward-declared at the point `Smooth` was defined. Because
     `pmy_driver_`'s type doesn't depend on `Smooth`'s template parameters,
     two-phase lookup requires `MultigridDriver` to be complete at `Smooth`'s
     definition point, which it wasn't — Clang/IntelLLVM (`icx`) diagnosed this as
     `error: member access into incomplete type 'MultigridDriver'`; GCC is known to
     be more lenient about exactly this case (untested here, so unconfirmed whether
     GCC builds were silently relying on that leniency). `git blame` traces both
     the `GetCoffset()`/`coffset_` AMR-coloring mechanism and this call site to
     `davidvelasco07`, Feb 2026 — no evidence anyone had compiled this file with a
     strict/Clang-based toolchain before this session. `mg_gravity.{hpp,cpp}` also
     includes `multigrid.hpp` and would have hit the identical error if ever
     compiled the same way (it doesn't currently call `Smooth` from outside
     `multigrid.hpp`, so it was never exposed to this).
     **Fix applied** (the "proper fix" option, not the GCC-workaround option — see
     recommendation below for why): declared `Smooth` in-class as before but with
     no body, then defined it out-of-line further down `multigrid.hpp`, after
     `class MultigridDriver` closes, so `MultigridDriver` is complete by the time
     the body is parsed. `CalculateDefect`/`CalculateFASRHS` (the two templates
     right after `Smooth` in the class body) were left untouched — they don't
     touch `pmy_driver_`, so they never had this problem.
     ```cpp
     // In class Multigrid's public section (was a full inline definition, now a
     // declaration only):
     template <typename ViewType, typename StencilOp>
     void Smooth(ViewType &u, const ViewType &src, const ViewType &coeff,
                 const ViewType &matrix, const StencilOp &stencil, int rlev,
                 int il, int iu, int jl, int ju, int kl, int ku, int color, bool th);

     // ... class Multigrid { ... }; ends, then class MultigridDriver { ... }; ends ...

     // New: out-of-line definition, placed immediately after MultigridDriver's
     // closing brace, body byte-for-byte identical to the original inline one:
     template <typename ViewType, typename StencilOp>
     void Multigrid::Smooth(ViewType &u, const ViewType &src, const ViewType &coeff,
                             const ViewType &matrix, const StencilOp &stencil,
                             int rlev, int il, int iu, int jl, int ju, int kl,
                             int ku, int color, bool th) {
       // ... unchanged body, including the pmy_driver_->GetCoffset() call ...
     }
     ```
     Net effect: identical generated code, identical public API (still a public
     template member of `Multigrid`, called the same way from every existing call
     site) — this is purely a reordering to satisfy two-phase lookup, not a
     behavior change. Verified by rebuild: the incomplete-type error is gone and
     no new error appears in its place.
   **For reporting to `multigrid.hpp`'s owner**: both changes are additive/
   reordering only — no existing method signature, behavior, or call site changed.
   The `Smooth` reorder is the one worth their explicit review, since it's their
   code being restructured: confirm the moved body is indeed unchanged (it is,
   verified during the edit — no lines inside the function body differ from the
   original), and confirm no other `.cpp` in the tree defines its own
   out-of-line specialization of `Multigrid::Smooth` that would now conflict with
   the new location (none found via `grep -rn "Multigrid::Smooth"` across `src/`
   before this change — only the one definition existed). Recommend they also
   audit whether any GCC-only build of this project was implicitly relying on
   GCC's leniency here, in case other code near `GetCoffset()`/AMR-coloring has
   the same latent pattern elsewhere. Also found while verifying, unrelated to
   either fix: a full project build initially crashed inside Intel `icx`
   2024.0.2's optimizer while compiling `multigrid_driver.cpp` at `-O3` — resolved
   by moving to `intel/2025.3`/`impi/2021.17` on Sakura, no source change needed
   (see "Status" above). Worth flagging to the owner in case other in-progress
   work on this project is still pinned to `intel/2024.0`.
8. First actual run (`inputs/dyn_grmhd/cfc_tov.athinput`, new file, see "Status"
   above) surfaced three more bugs, all now fixed, each caught by a successively
   deeper stage of the same run:
   - **`<cfc>` missing from `parameter_input.cpp`'s block-name whitelist**
     (`ParameterInput::CheckBlockNames`, `src/parameter_input.cpp`) — the input
     file was rejected outright before the mesh was even built (`FATAL ERROR ...
     Invalid <block_name> in input file`). One-line fix: added `"cfc"` to the
     `valid_name` list alongside `"z4c"`/`"cce"`/etc. Purely a missed registration,
     not a design issue — every other `<cfc>`-reading code path already worked.
   - **`AssembleVectorSource`/`BuildShiftSource` (`cfc.cpp`) wrote `p_src`/
     `eta_src` at the wrong depth**: both are allocated at this solver's own
     shallower `mg_nghost` ghost width (cfc.hpp's `u_p_src`/`eta_src` comment,
     item 4/Finding H's "stays ngh_-deep" carve-out), but every write to them
     reused the surrounding loop's mesh-`NGHOST`-indexed `(k,j,i)` directly, with
     no translation between the two index spaces. With `nghost=4` (mesh) vs.
     `mg_nghost=1` (default), this overran the destination array by exactly the
     gap between the two depths — caught immediately by
     `Kokkos_ENABLE_DEBUG_BOUNDS_CHECK` (`Kokkos::View ERROR: out of bounds
     access label=("cfc_u_p_src") ...`) on the very first `AssembleVectorSource`
     call at cycle 0. Fixed by adding a cached `mg_nghost_` member to `CFC` (set
     once in the constructor, alongside the existing local `mg_nghost` the
     constructor already reads to size `u_p_src`/`eta_src`) and computing
     translated indices `mk = k - ks + mg_nghost_` (and `mj`/`mi` likewise) at
     every `p_src_`/`eta_src_` read or write site — four sites total, all within
     `AssembleVectorSource`'s two branches and `BuildShiftSourceImpl` (which
     needed `mg_nghost` threaded through as a new parameter from its one call
     site). `u_tilde_`/`s_tilde_d_` (mesh-`NGHOST`-deep, correctly indexed
     already) were left untouched. `src_`'s own ghost ring is never actually
     read by `Multigrid::Smooth`/`CalculateDefect`/`CalculateFASRHS` (confirmed
     by inspection — both only index `src(m,0,k,j,i)` at the exact stencil
     point, never a neighbor), so leaving `u_p_src`/`eta_src`'s own ghost ring at
     its zero-initialized default (rather than also populating it) is safe, not
     just expedient.
   - **`MGCFCVectorPoisson`'s per-channel `Smooth`/`CalculateDefect`/
     `CalculateFASRHS` helpers subview an array nobody allocates**: item 2's
     `SmoothChannels`/`CalculateDefectChannels`/`CalculateFASRHSChannels`
     (`mg_cfc_vector_poisson.cpp`) sliced `coeff_[level]`/`matrix_[level]` per
     channel with `Kokkos::subview(..., std::make_pair(v,v+1), ...)`, alongside
     the same per-channel treatment of `u_`/`src_`/`def_`. But
     `CFCVectorPoissonStencil::Apply()` (this file's flat 7-point Laplacian)
     never reads `coeff`/`matrix` at all, and neither is actually allocated for
     this solver: `ncoeff_` stays at `MultigridDriver`'s default 0 (this solver
     never sets it, unlike `MGCFCConformalFactor`/`MGCFCLapse`'s item-3 fix), and
     `matrix_`/`nmatrix_` turn out to be dead code across the *entire* codebase —
     grepping confirms no `Kokkos::realloc(matrix_...)` call exists anywhere,
     including in `gravity`. Subviewing a channel range on an unallocated
     (all-zero-extent) array is a bounds violation regardless of which channel,
     caught by Kokkos on the very first V-cycle smooth
     (`Kokkos::subview bounds error (...)`, traced via `addr2line` through
     `MGCFCVectorPoisson::CalculateDefectPack` → `Multigrid::CalculateDefectNorm`
     → `MultigridDriver::SolveIterative` → `CFC::SolveVecXTask`). Tried first:
     widening the `coeff`/`matrix` subview's channel dimension to `Kokkos::ALL`
     instead of `vr` — compiles only if `ViewType` still matches `u`/`src`/`def`'s
     subview type, which it doesn't (`ALL_t` vs. `std::pair<int,int>` are
     different template arguments to `Kokkos::subview`, confirmed by the
     resulting "deduced conflicting types for parameter 'ViewType'" compile
     error). **Fix actually applied**: stopped threading `coeff_lv`/`matrix_lv`
     through these three helpers at all — `u`'s (or `src`'s) own already-valid,
     already-correctly-typed per-channel subview is passed in the `coeff`/
     `matrix` argument slots instead, since `Smooth`/`CalculateDefect`/
     `CalculateFASRHS`'s generic signature requires *some* same-typed 4th/5th
     argument but this stencil never dereferences it. Removed the now-dead
     `coeff_[current_level_]`/`matrix_[current_level_]` arguments from all three
     call sites in `SmoothPack`/`CalculateDefectPack`/`CalculateFASRHSPack`
     rather than leaving unused parameters around.
   With all three fixed, `cfc_tov.athinput` runs to completion (`nlim=10`, exit
   0) for the first time — see "Status" above and item 9 for what's still wrong.
9. **Investigation, in three rounds** — the completing run from item 8 throws
   `NANS_IN_CONS` C2P errors at cycle 0, localized to grid points at `k=4` (the
   first physical cell above the `x3=0` reflecting boundary, i.e. `ix3_bc=reflect`
   in `cfc_tov.athinput`) where `g_dd`/`psi4` come out `-nan` (`alp`/`beta`/`K_dd`
   are fine at the same points — turns out to be because `MHD_C2P` runs between
   `CFC_SolvePsi` and `CFC_SolveLapse`, so it sees `adm.alpha` still holding the
   pgen's original analytic value at this point, only `adm.g_dd`/`psi4` having
   been freshly overwritten by `SolveConformalFactor`'s `AssembleConformalMetric`
   call — a useful clue, not a bug itself). All 1000 reported error locations (the
   cap before AthenaK suppresses further C2P error printouts) are at `k=4`
   specifically, confirming this is boundary-localized, not domain-wide.
   - **Round 1 (found and fixed, real bug, insufficient alone)**: CFC had no
     equivalent of `Z4c::ApplyPhysicalBCs`/`MeshBoundaryValues::Z4cBCs` — its 7
     `RecvAndUnpackCC` ghost-exchange calls (`CFC::QueueCFCTasks`) only handle
     inter-MeshBlock/periodic communication, exactly like Hydro/Z4c/radiation, and
     *all three* of those have a dedicated physical-BC pass the mesh-level
     `MeshBoundaryValuesCC` machinery does not provide automatically. Fixed by
     adding `src/bvals/physics/cfc_bcs.cpp` (new file, registered in
     `src/CMakeLists.txt`, `MeshBoundaryValues::CFCScalarBCs`/`CFCVectorBCs`
     declared in `bvals.hpp` alongside `HydroBCs`/`Z4cBCs`) and calling the
     appropriate one from each of the 7 `Recv*Task` functions in `cfc.cpp`, gated
     on `tstat == TaskStatus::complete && !strictly_periodic` (matching
     `Z4c::ApplyPhysicalBCs`'s own periodicity guard; the completeness check
     specifically guards against acting on a still-in-flight async MPI recv,
     though it's moot for the current single-meshblock test). Deliberately
     simpler than `HydroBCs`: no inflow table (no fluid-state analog for these
     elliptic-equation potentials) and no diode-specific velocity clamping (no
     flux/flow concept either) — `outflow`/`diode`/`inflow`/`user` all reduce to
     a plain zero-gradient copy; only `reflect` (even parity for the 4 scalar
     fields `eta_x`/`psi`/`alpha_psi`/`eta_beta`, odd parity on the
     face-aligned channel only for the 3 vector fields `p_x`/`x_u`/`p_beta`) and
     `vacuum` need special handling. **Rebuilt and reran: byte-for-byte identical
     failure** (same location, same `alp` value) — this fix is real (these ghost
     cells do feed `cfc_reconstruct.cpp`'s `Dx<NGHOST>` and were genuinely never
     filled before), but provably not what's causing *this* NaN.
   - **Round 2 (found and fixed, real bug; wrong flag value used at first, see
     round 3)**: the `MultigridDriver` base constructor blanket-converts *every*
     non-periodic mesh face to `BoundaryFlag::mg_zerofixed` (or, for
     `MGCFCConformalFactorDriver`/`MGCFCLapseDriver`, their own constructors then
     blanket-convert every non-periodic face again to `BoundaryFlag::mg_multipole`
     — `multigrid_driver.cpp:82-91`, `mg_cfc_conformal_factor.cpp`/
     `mg_cfc_lapse.cpp`'s old constructor comments). Both select the odd/
     antisymmetric mirror at that face — correct for the true outer/
     asymptotically-flat boundary (`X^i|_rmax=0`, Gmunu eq. 77/78's 1/r falloff),
     wrong for a reflecting symmetry plane, which needs an even/symmetric mirror
     instead. Fixed in all 4 CFC driver constructors: after the base
     constructor's blanket default (and, for the two nonlinear solvers, their own
     multipole-default loop) runs, a small loop restores the even-mirror
     treatment wherever `pmbp->pmesh->mesh_bcs[f] == BoundaryFlag::reflect`.
     **First attempt used `mg_mesh_bcs_[f] = BoundaryFlag::reflect`** (the
     ordinary mesh-level flag) — rebuilt and reran: byte-for-byte identical
     failure, which led to round 3.
   - **Round 3 (found the real flag-value bug; also one claim below turned out
     wrong, see the correction inline)**: traced why round 2 had zero effect.
     `mg_mesh_bcs_[face]` is multigrid-internal state, read only by
     `MultigridDriver`'s own boundary code — and that code recognizes exactly
     four values: `periodic`, `mg_zerofixed`, `mg_zerograd`, `mg_multipole`.
     `BoundaryFlag::reflect` (the ordinary mesh flag round 2 used) isn't one of
     them. Confirmed directly in `MultigridDriver::MGRootBoundary`'s device path
     (`multigrid_driver.cpp:1856-1920`): explicit `if`/`else if` chains on
     `periodic`/`mg_zerofixed`/`mg_zerograd` only, no `else` fallback — a face
     set to plain `reflect` falls through every branch silently and that ghost
     cell is simply never written, leaking whatever was there before (0.0, from
     allocation, on the first call). **Fix**: changed all 4 constructors to set
     `mg_mesh_bcs_[f] = BoundaryFlag::mg_zerograd` instead of `reflect` on
     reflecting mesh faces — `mg_zerograd`'s `ghost = interior` formula is
     exactly the even mirror needed, and it's a value `MGRootBoundary` already
     handles. **A claim made and then retracted in this same round**: initially
     (wrongly) concluded `MultigridDriver::PhysicalBoundary`
     (`multigrid_tasks.cpp:129`) was dead code, never called anywhere, based on
     `grep -rn "PhysicalBoundary" src/ | grep -v "multigrid_tasks.cpp|multigrid.
     hpp"` — an exclusion that hid the very call sites it was looking for, since
     `SetMGTaskListToFiner`/`SetMGTaskListToCoarser`/`SetMGTaskListFMGProlongate`
     (all in `multigrid_tasks.cpp`, the file excluded) queue `PhysicalBoundary`
     seven times combined via `AddTask(&MultigridDriver::PhysicalBoundary, ...)`.
     It is not dead code and needed no wiring; this was communicated to the user
     as a "root cause" before being caught and corrected — flagged here so the
     mistake isn't silently lost. **Rebuilt and reran with the `mg_zerograd`
     fix: still fails**, and by a NaN-scan diagnostic added in round 4, the root
     grid's own defect count got *worse* (4 NaN cells out of 27 before this fix,
     20 after) — the fix is more correct in principle but is not what's causing
     this crash, or isn't the whole story.
   - **Round 4 (bisection, temporary instrumentation, not left in the tree)**:
     added a temporary host-side NaN scan (`Kokkos::create_mirror_view` +
     `std::isnan` loop, removed after use — no debug code remains in `src/cfc/`)
     at three points in `CFC::SolveConformalFactor`/
     `MGCFCConformalFactorDriver::Solve`: (1) `u_tilde`/`a_sq` right before the
     solve — **clean**, zero NaN, sane magnitudes (max ~1.3e-3 and ~2.2e-7
     respectively) on the very first call; (2) the root grid's `coeff_` right
     after `TransferCoeffToRoot()` — **clean**, zero NaN; (3) the root grid's
     `u_` (its single interior cell plus ghost ring, 3x3x3 with `ngh=1`) right
     after `SolveMG` returns — **already 4 (later 20) of 27 cells NaN**, with
     every remaining cell exactly `0.0`. So the corruption is seeded at the root
     grid, during the very first V-cycle, from otherwise-clean inputs — not
     something a later stage's stale data drags in, and not domain-wide in the
     sense of "everywhere at once": `psi` after the full solve showed 287,300 of
     373,248 mesh-depth cells NaN (~77%), but that's downstream contamination
     spreading from this one degenerate root grid outward once the corrupted
     root values get transferred back to the meshblock level
     (`TransferFromRootToBlocks`) and iterated on.
   - **Round 5 (narrowed further, root cause still not found)**: the root grid
     for this test is a single interior cell (`nlevel=1`, aggregating the one
     meshblock down to one point), with 3 faces on `mg_zerograd` (the fixed
     reflecting faces) and 3 on `mg_multipole` (the true outer/diode faces,
     `mporder=4` by default). Tried `mporder=2` (quadrupole instead of
     hexadecapole) via the input file, no rebuild needed: identical result (20
     NaN). Could not test with multipole disabled entirely —
     `MGCFCConformalFactorDriver`'s constructor fatal-errors unless
     `mporder` is 2 or 4, there is no "off" setting. Current best guess, **not
     confirmed**: something in the multipole boundary evaluation
     (`EvalMultipolePhi`, or the `CalculateCenterOfMass`/
     `CalculateMultipoleCoefficients` calculation feeding it) misbehaves for
     this degenerate single-cell root grid specifically, independent of the
     `mg_zerograd` faces (which were confirmed correct in isolation by
     `MGRootBoundary`'s code, round 3) — but this has not been verified by
     directly instrumenting the multipole coefficients/evaluation the way
     rounds 3-4 instrumented the transfer and root `u_`. **Stopped here** to
     check in rather than keep iterating unilaterally; next step is almost
     certainly to scan `mpcoeff_`/`d_mpcoeff_` for NaN right after
     `CalculateMultipoleCoefficients()`/`SyncMultipoleToDevice()`, and/or to
     manually trace `EvalMultipolePhi`'s arithmetic for a degenerate
     single-source-point, single-evaluation-point geometry.
