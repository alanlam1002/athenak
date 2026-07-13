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

## Status: feature-complete (all equation bodies implemented); the original
## boundary NaN is fixed (item 9, rounds 6-7). Both the `psi` O(1) corner-error
## bug (item 9, rounds 8-15) and the `alpha`/lapse non-convergence bug (rounds
## 16-18) are FIXED and verified at two resolutions. Four bugs total, all fixed
## and committed:
## 1. (round 15) `MGCFCConformalFactor::SmoothPack`/`CalculateDefectPack` silently
##    dropped the FAS `src_` coarse-grid-correction term.
## 2. (round 15) `RetrieveSolution` (both `psi` and `alpha` drivers) passed the
##    wrong ghost depth to `Multigrid::RetrieveResult` (`mglevels_->
##    GetGhostCells()`, this solver's own shallow internal depth, instead of the
##    mesh's true `NGHOST`), silently mis-aligning the solved-interior copy-out
##    by 3 cells and leaving the outermost interior cells near every domain face
##    never written at all -- the actual `psi` corner-catastrophe root cause
##    (`psi_maxerr` `0.996` -> `~0.025`).
## 3. (round 17, found by user code review) `MGCFCLapse`'s `K(x)` reaction
##    coefficient (`Utilde+2*Stilde`, `psi`, `Ahat^2`) was restricted to coarser
##    V-cycle levels as three separate channels, then recombined nonlinearly
##    (`psi^-2`, `psi^-8`) at every level -- inconsistent with FAS, since
##    restriction doesn't commute with a nonlinear recombination. Fixed by
##    precomputing `K(x)` once at the finest level and restricting that single
##    coefficient directly (`ncoeff_` 3->1).
## 4. (round 18, found by user code review) `MGCFCLapse::SmoothPack`/
##    `CalculateDefectPack` never read `src_` at all (same bug class as #1, just
##    in the lapse solver) -- the FAS coarse-grid correction `CalculateFASRHSPack`
##    computed was silently discarded every V-cycle.
## Verification: an iterated `DebugCFCSolveAtT0` t=0 replay (rounds 9/11/13-18,
## always added then fully reverted after use) confirms both `psi` and the final
## assembled `alpha` converge to a small, resolution-stable residual (`psi
## ~0.68%`, `alpha ~1.35%`) once self-consistently iterated, at both 128^3 and
## 256^3, with **zero** `"Failed to converge"` V-cycle messages post-round-18 (a
## single cold-start isolated solve shows a much larger, non-resolution-
## convergent error at the star's core for both fields -- this is expected
## "lagged psi^6" cold-start behavior inherent to the diagnostic, not a bug; see
## round 16). No outstanding open follow-ups from this investigation thread.

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

**The original boundary NaN (item 9, rounds 1-6) is fixed and confirmed by
rebuild+rerun; a second, more serious bug has since been found and is
unresolved** — see open item 9 for the full, multi-round investigation,
including one claim that turned out to be wrong
(`MultigridDriver::PhysicalBoundary` was reported dead code; it is not — a bad
grep exclusion hid its actual call sites) and one real fix that landed but did
not by itself resolve the original crash (`mg_mesh_bcs_[face]` must hold a
multigrid-internal `BoundaryFlag` value — `mg_zerograd`, not the ordinary mesh
`BoundaryFlag::reflect` — fixed in all 4 CFC driver constructors). Round 6
tracked the actual NaN source to `CalculateCenterOfMass()`
(`multigrid_driver.cpp`) dividing by an always-zero `src_` for these two solvers
(their matter fields live in `coeff_`, not `src_`, by design — Finding B), which
poisons the multipole expansion origin/coefficients with NaN and gets evaluated
directly into the outer ghost cells via `mg_multipole`. Fixed by switching
`psi`/`alpha_psi`'s outer boundary from `mg_multipole` to `mg_zerofixed`
(matching `P_i`/`eta`'s existing treatment) instead of fixing the multipole
integration itself. **Round 7 confirmed by rebuild+rerun that this actually
clears the NaN** (cycle 0 completes, no `NANS_IN_CONS` anywhere), but surfaced a
`SolveIterative` convergence-plateau warning in its place, hypothesized (at the
time) to be caused by the `mg_zerofixed` outer boundary's `O(M/r)` mismatch.
**Round 8 falsified that hypothesis**: spatial-defect and analytic-comparison
diagnostics (both temporary, since reverted) show the stalled defect is
concentrated at the single interior cell closest to the star's center — the
corner where the three reflecting (`mg_zerograd`) faces meet — not near the
outer boundary at all, and the solved `psi`/`alpha` are wrong by `O(1)`
(`psi` off by up to `2.996`, `alpha` by `0.180`) after just one solve, not
merely short of a tight tolerance. This is a real, unresolved bug, most likely
in how the reflecting-corner ghost cells are filled, not a slow-convergence
tuning question. **Round 9 (per user request) isolated it further**: running a
single CFC solve directly against the pristine, exactly-analytic t=0 initial
data — bypassing all hydro/con2prim tasks entirely — already produces
`psi_maxerr=0.996`, `alpha_maxerr=0.248`. So the bug is not introduced or
amplified by fluid evolution at all; it's present in CFC's very first solve on
known-correct input. **Round 10 (per user request) then tested a full,
non-octant domain with no reflecting boundaries anywhere**, to check whether
the corner-localized error was specific to the reflecting/`mg_zerograd`
symmetry or something more general — and found what now looks like **two
separate bugs**: `psi`'s error (`0.992`) is still corner-localized, but now at
a plain `mg_zerofixed`-only corner (ruling out anything reflecting-specific,
pointing instead at `MGRootBoundary`'s sequential per-axis ghost fill not
composing correctly at any multi-face corner, regardless of `BoundaryFlag`);
`alpha`'s error (`0.254`) moved away from any corner entirely and onto the
star's core, suggesting a second, independent problem in the lapse solve
itself, unrelated to boundary handling. **Neither is yet root-caused**, see
item 9 rounds 8-10 for the full evidence and the two separate next angles this
split implies.

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

- **CFC-only (no `<z4c>`) runs currently still use the speed-of-light CFL bound for
  `dt`, not the true fluid characteristic speed -- known, deliberately deferred.**
  `MHD::NewTimeStep` (`src/mhd/mhd_newdt.cpp:87-91`) hardcodes the characteristic
  speed to `1.0` whenever `is_general_relativistic_` or `is_dynamical_relativistic_`
  is true, instead of computing the true GRMHD fast-magnetosonic speed the way the
  Newtonian/SR branches just below it do. `is_dynamical_relativistic` is true for
  any CFC run (`coordinates.cpp:31`: true whenever `<adm>` or `<z4c>` exists), so
  this branch always fires here. For z4c-based runs this is harmless -- `Z4c::
  NewTimeStep` (`z4c_newdt.cpp:50-52`) independently imposes `dt ~ dx/1` anyway
  (light-crossing CFL for the hyperbolic z4c system), so it was never the binding
  constraint. **For CFC (no z4c), nothing else imposes a light-crossing
  restriction, so `dt` ends up needlessly capped at the light-crossing scale**
  rather than the physically-appropriate sound/fast-magnetosonic-crossing scale --
  confirmed empirically (`dt` is bit-for-bit constant every cycle in a CFC TOV-star
  run, `= cfl_number*dx/1.0`, never tracking density/pressure). **Decision (per
  user direction): leave as-is for now** -- explicitly keep using the
  `is_general_relativistic_`-style conservative bound for CFC runs too; computing
  the actual GRMHD fast-magnetosonic characteristic speed is deferred to future
  work, not part of the current CFC investigation. Revisit here first if CFC
  performance ever needs improving (likely a large speedup, since fluid sound speed
  is generally well below `c`).

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
     item 3b below (now being implemented -- see item 12's investigation and plan).
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
   - **Round 6 (root cause confirmed, fixed by removing multipole use rather than
     fixing the multipole path itself)**: round 5's guess was on the right track but
     not quite located correctly — it isn't about the root grid's degenerate size,
     it's that `CalculateCenterOfMass()`/`CalculateMultipoleCoefficients()`
     (`multigrid_driver.cpp:2204/2365`, both written for and only previously
     exercised by `gravity`) unconditionally integrate `mglevels_->src_[nlevel_-1]`
     as "the density." That's correct for `gravity`, which populates `src_` via
     `LoadSource(u0, IDN, ng, -four_pi_G_)` — but item 3's Finding B deliberately
     put `Utilde`/`Ahat^2` (and the lapse equation's `Utilde+2*Stilde`/`psi`/
     `Ahat^2`) in `coeff_` instead, specifically so the V-cycle's automatic FAS
     tau-correction into `src_` can't corrupt them. `src_` is therefore always zero
     for both nonlinear CFC solvers at the point `Solve()` calls the multipole
     setup (before any V-cycle work has run). `CalculateCenterOfMass()`'s
     `Real im = 1.0 / totals[0];` divides by that zero total, producing `inf`, and
     `mpo_[0..2] = im * totals[1..3]` (`inf * 0.0`) comes out **NaN** —
     `CalculateMultipoleCoefficients()` then uses this NaN `mpo_` as the expansion
     origin for every higher moment (`x = ... - xorigin`, etc.), so even though the
     per-point integrand `s = src(...)*vol` is exactly `0`, `s * x = 0 * NaN = NaN`
     poisons `mpcoeff_[1..24]` (only the origin-independent monopole
     `mpcoeff_[0]` survives as clean `0`). `SyncMultipoleToDevice()` ships this to
     the device, and `MGRootBoundary`'s `mg_multipole` branch evaluates
     `EvalMultipolePhi` with it directly into the outer ghost cells during the very
     first V-cycle — exactly reproducing every earlier round's evidence: `coeff_`
     clean (round 4, it was never the culprit), `u_` already NaN immediately after
     `SolveMG` returns (corruption enters via the boundary before any interior
     Newton iteration), and `mporder` 2 vs. 4 making no difference (round 5, both
     orders consume the same NaN `mpo_`). This is also **not specific to the
     octant/reflecting-boundary test setup** — the `mg_multipole` outer faces
     exist in every CFC configuration that uses these two solvers, since that was
     the only outer-BC option they had.
     **Fix**: rather than fixing `CalculateCenterOfMass`/
     `CalculateMultipoleCoefficients` to read `coeff_` instead of `src_` (which
     would also need a from-scratch re-derivation of the multipole integrand's
     normalization, since `ScaleMultipoleCoefficients()`'s constants are calibrated
     for gravity's specific `src = -4*pi*G*rho` convention, not eq. 73/74's `2*pi`
     coefficient), per user direction the outer-BC choice was changed instead:
     `MGCFCConformalFactorDriver`/`MGCFCLapseDriver`'s constructors no longer set
     `mg_mesh_bcs_[f] = BoundaryFlag::mg_multipole` on any face — non-periodic,
     non-reflecting faces now fall through to `MultigridDriver`'s own base-
     constructor default, `BoundaryFlag::mg_zerofixed` (`multigrid_driver.cpp:
     83-90`), same as `P_i`/`eta`'s existing treatment. `mg_zerofixed`'s actual
     ghost-cell formula (`ghost = -interior`, `multigrid_driver.cpp:1858-1859`) is
     Dirichlet `u=0` at the face, i.e. `psi=1`/`alpha*psi=1` exactly at the domain
     boundary — the leading-order (monopole-zero) truncation of Gmunu eq. 77/78's
     true isolated `1/r` falloff. This is a real accuracy tradeoff (`O(M/r_boundary)`
     error at the boundary instead of the paper's higher-order asymptotic
     condition), accepted for now since `cfc_tov.athinput`'s boundary (`x1max=
     102.4`) sits well outside the star; **not a permanent abandonment of
     multipole support** — `mporder_`/`autompo_`/`nodipole_`/
     `AllocateMultipoleCoefficients()`/the "mporder must be 2 or 4" input parsing
     are left in place (inert), and `Solve()`'s call to `CalculateCenterOfMass()`/
     `CalculateMultipoleCoefficients()`/`SyncMultipoleToDevice()` was removed
     entirely (not just left unused) since it would otherwise still compute the
     `1.0/totals[0]` division — and therefore still produce an internal inf/NaN —
     every single stage regardless of whether any face reads the result, which is
     both wasted work and a landmine under any future FPE-trapping build. If
     multipole boundaries are wanted later (e.g. to allow a smaller box for the
     same accuracy, per Gmunu sec. 2.6.2), the actual fix is a CFC-local
     replacement for these two functions that integrates `coeff_` channel 0 instead
     of `src_`, with the normalization re-derived — deliberately not implemented
     now, to keep this fix minimal and avoid touching `src/multigrid/` at all.
   - **Round 7 (fix verified by rebuild + rerun; new, much smaller issue found)**:
     rebuilt (`cfc_sakura.sh`'s toolchain, `intel/2025.3`/`impi/2021.17`) and reran
     `cfc_tov.athinput` (`run_cfc_tov/run10.log`). **The boundary NaN is gone**: no
     `NANS_IN_CONS`, no `-nan` anywhere in the log; cycle 0 completes all RK stages
     and cycle 1 begins. A different, far less severe issue surfaced in its place:
     `MultigridDriver::SolveIterative` (`multigrid_driver.cpp:781-820`, generic,
     untouched by this fix) logs "Failed to converge after 40 iterations" 6 times
     during cycle 0 -- defect plateaus around `2e-6`-`7e-6`, short of the
     `<cfc> mg_threshold` default of `1e-10`. Not a crash: hitting the 40-iteration
     cap is a soft stop in that function (sets `pdriver->nlim = pmy_mesh_->ncycle`
     rather than aborting), so the run just quietly ends after cycle 1 starts
     instead of reaching `nlim=10`. Six failures in one cycle matches 2 nonlinear
     solvers (`psi`, `alpha_psi`) x 3 RK stages -- consistent with this being
     specific to the two solvers whose outer BC round 6 just changed, not the
     linear `P_i`/`eta` solves (which use `mg_zerofixed` already and were
     unaffected by this fix).
     **Leading hypothesis, not yet confirmed**: `mg_zerofixed` forces
     `psi=1`/`alpha*psi=1` exactly at the domain boundary (`x1max=102.4`), but the
     true solution's boundary value is the small nonzero isolated `~M/(2r)`
     falloff -- a fixed mismatch between the imposed Dirichlet data and the actual
     solution that V-cycle smoothing can reduce but not eliminate, so the defect
     plateaus at a floor set by that mismatch rather than continuing to shrink
     toward `1e-10` (which was tuned assuming the more accurate multipole BC).
     **Before picking a new `mg_threshold` value, next step (per user direction) is
     to verify this hypothesis and validate the solver itself, not just tune around
     the symptom**: (1) instrument the defect array's spatial distribution at the
     40-iteration cutoff to confirm it's actually concentrated near the outer
     boundary rather than spread through the domain (would falsify the boundary-
     mismatch hypothesis and point at a different bug instead); (2) compare the
     converged `psi`/`alpha=alpha_psi/psi` against `dyngr_tov.cpp`'s own analytic
     isotropic-TOV profile (available since `isotropic=true`, `v_pert=0.0` is a
     static-equilibrium configuration -- the matter distribution barely changes
     step to step, so the CFC-solved metric should reproduce the pgen's original
     analytic profile closely if the solver is working correctly, independent of
     whatever `mg_threshold` is chosen). Both diagnostics to be added as temporary
     instrumentation (removed once used, matching rounds 1-6's pattern) -- in
     progress.
   - **Round 8 (both diagnostics run; round 7's boundary-mismatch hypothesis
     falsified -- this is a real wrong-answer bug, not a slow-convergence
     tolerance issue)**: temporary instrumentation added to both
     `MGCFCConformalFactor`/`MGCFCLapse` (a `DebugPrintDefectStats()` method
     printing the finest-level `def_` array's largest magnitude and its location
     as a fraction along each axis, called right after `SolveMG()` returns in each
     driver's `Solve()`) and to `dyngr_tov.cpp`'s `TOVHistory` (two new history
     columns, `psi-maxerr`/`alpha-maxerr`, comparing `padm->adm.psi4^0.25`/`alpha`
     against the pgen's own analytic isotropic-TOV profile, reusing
     `tov::TOVStar::GetMandAlphaIso`/`FindSchwarzschildR` the same way
     `SetADMVariablesToTOV` does). Getting the second diagnostic working surfaced
     two more real (if narrow) bugs along the way, both already reverted:
     - `ptov_params` (the persistent TOV solution `TOVHistory` needs to read) is
       only ever populated `if (pmbp->padm->is_dynamic || pmy_mesh_->adaptive)`
       (`SolveTOV`/`SetupTOV`, `dyngr_tov.cpp`) -- both false for this test, so it
       stays permanently `nullptr` for the whole run. A first attempt at the
       diagnostic crashed with a null-pointer segfault inside `TOVHistory` at
       startup (confirmed via `gdb -batch -ex "bt full"` on the resulting core
       dump, using a `-g` recompile of just that one file to get line numbers) --
       traced to `Driver::Initialize -> HistoryOutput::LoadOutputData` calling
       `user_hist_func` once before `ProblemGenerator` has run at all. Worked
       around for the diagnostic by temporarily removing the `is_dynamic`/
       `adaptive` guard (unconditionally populating `ptov_params`) -- confirmed
       safe (nothing else conditions on its null-ness except `FinalizeTOV`'s
       cleanup and `SetADMVariablesToTOV`, the latter separately gated on
       `is_dynamic` in `dyn_grmhd.cpp`'s own task queueing, unaffected) -- but
       reverted afterward rather than kept, since it's a real behavior change to
       shared pgen code that wasn't asked for.
     - Own bug in the comparison formula itself: first pass set
       `psi_analytic = fmet` (`fmet = r_schw/r`), but `dyngr_tov.cpp`'s own pgen
       code sets `psi4 = fmet^2` (`g_dd = psi4*delta_ij`), and `tov.hpp`'s exterior
       formula (`FindSchwarzschildR`: `r_schw = r_iso*psi^2`) confirms
       `fmet = psi^2`, not `psi` -- so `psi_analytic` should be `sqrt(fmet)`. Caught
       immediately by a sanity check built into the diagnostic itself: at `t=0`
       (before CFC's first solve ever runs, metric still exactly the pgen's raw
       analytic initial data), `psi-maxerr` should read machine epsilon, not a
       real number -- the first (buggy) formula gave `0.222` at `t=0`, an obvious
       tell; fixed to `sqrt(fmet)`, rerun gave `2.22e-16` at `t=0` as expected,
       confirming the corrected formula and the rest of the diagnostic
       infrastructure are both trustworthy before trusting its `t=0.64` output.
     - **Results, `run_cfc_tov/run15.log` + `cfc_tov.user.hst`** (both diagnostics
       reverted after this run, per the established add-use-remove pattern; not
       left in the tree):
       - Defect location: **identical for every one of the 6 failed solves across
         both `psi` and `alpha_psi`** -- `(k,j,i)=(1,1,1)`,
         `frac_along_axis=(0,0,0)`, i.e. the single innermost interior cell at the
         corner where all three reflecting faces (`x1=x2=x3=0`) meet, closest to
         the star's center (highest density, `rhoc=1.28e-3`) -- **not** near the
         outer `mg_zerofixed` boundary at all. This directly falsifies round 7's
         "Dirichlet-mismatch-at-the-outer-boundary" hypothesis: whatever is
         stalling convergence is happening at the opposite end of the domain,
         at/near the coordinate origin where the three `mg_zerograd` reflecting
         faces meet in a corner, not at any `mg_zerofixed` face.
       - Analytic comparison: `psi-maxerr` goes from `2.22e-16` at `t=0` (confirms
         the diagnostic is measuring correctly, and confirms the pgen's initial
         data is exactly analytic as expected) to **`2.996`** at `t=0.64` (after
         cycle 0's single CFC solve) -- `psi` itself is `O(1-1.5)` for this star,
         so an absolute error of `~3.0` is not "close but short of a tight
         tolerance," it is grossly wrong. `alpha-maxerr` goes from `0.0` to
         `0.180` over the same step (`alpha` ranges roughly `0.68`-`1.0` here) --
         also a large, not-nearly-converged error.
     - **Conclusion**: this is not the "boundary Dirichlet-mismatch creates an
       irreducible defect floor" issue round 7 hypothesized -- that framing is
       retired. The real, still-unexplained problem is localized at/near the
       reflecting-boundary corner nearest the star's center, and is severe enough
       to produce an `O(1)` error in `psi`/`alpha` after just one solve, not a
       slow-but-basically-correct convergence tail. **Not yet root-caused.**
       Plausible next angles (not yet tried): (1) whether `cfc_bcs.cpp`'s
       `CFCScalarBCs`/`CFCVectorBCs` (or `MGRootBoundary`'s `mg_zerograd` fill,
       `multigrid_driver.cpp:1858-1920`) correctly fill the actual *corner* ghost
       cells where two or three reflecting faces meet -- the per-face sequential
       x1-then-x2-then-x3 fill order in `MGRootBoundary` means a true corner cell
       needs contributions this fill order may not compose correctly, and this is
       a genuinely different code path from the single-flat-face case already
       exercised/confirmed correct in round 3; (2) whether the Newton
       linearization (`ConformalFactorRHS`, `mg_cfc_conformal_factor.cpp`) is
       well-behaved at the highest-density point in the domain, independent of
       any boundary-fill question, given this cell is also where `rhoc` peaks;
       (3) re-running with a non-octant (full, non-reflecting-boundary) domain to
       see if the same corner-adjacent failure mode persists without any
       reflecting faces at all, which would cleanly separate "corner-fill bug"
       from "high-density-point solver bug." Stopped here to report back rather
       than keep iterating unilaterally, per the same pattern as round 5.
   - **Round 9 (per user request: isolate the CFC solve from fluid evolution
     entirely, by running it once against the pristine t=0 initial data)**: added
     a temporary `DebugCFCSolveAtT0(Driver*, Mesh*)` function (in `dyngr_tov.cpp`,
     since it needed direct access to `ptov_params`/`tov::TOVStar` for the
     analytic comparison), hooked into the very top of `Driver::Execute()` --
     before `ExecuteTaskList(pmesh, "before_timeintegrator", 0)` or anything else
     runs -- so it sees `pmbp->pmhd->u0`/`w0`/`padm->adm` exactly as
     `ProblemGenerator` left them. It manually calls all ~35 of `cfc::CFC`'s
     public task methods once, by hand, in the exact order `QueueCFCTasks()`
     queues them (`cfc.cpp:271-361`, a call sequence that is already a valid
     topological order of the dependency graph, so replaying it literally in
     source order is correct), bypassing `MHD_C2P` entirely (`RescaleSrcTask`'s
     real dependency) since `w0` at this point is already exactly the pgen's
     analytic primitives -- no con2prim inversion needed or wanted. After the
     chain completes it compares the resulting `psi`/`alpha` against the same
     analytic isotropic-TOV formula used in round 8, then calls `std::exit(0)` so
     the program never proceeds into the real simulation loop. Needed the same
     `ptov_params`-unconditional-populate change as round 8 (reapplied, then
     re-reverted afterward, exact same rationale).
     - **Result** (`run_cfc_tov/run16.log`): `global psi_maxerr=0.996`,
       `alpha_maxerr=0.248` after a *single* CFC solve against data that is
       *exactly* the analytic fixed point going in -- no fluid evolution, no RK
       substepping, no repeated cycles involved at all. This is smaller than
       round 8's after-one-full-cycle numbers (`2.996`/`0.180`) but is equally
       definitive: **the bug is entirely inside CFC's own solve pipeline, not
       something introduced or amplified by hydro/con2prim coupling.** Only 2
       `SolveIterative` "Failed to converge" messages this run (vs. 6 in round 7/8,
       which included 3 RK-stage repeats) -- exactly matches the 2 nonlinear
       solves (`psi`, `alpha_psi`) invoked, confirming the manual chain really did
       run exactly once as intended, not by accident replaying stale state.
       At the specific corner cell flagged in round 8 (mesh-indexed `(ks,js,is)`,
       `r=1.386`): `psi_solved=1.032` vs `psi_analytic=1.187` (off by `0.155` --
       notably *not* the domain's worst point in this pure-t=0 test, unlike round
       8's after-hydro run) but `alpha_solved=0.928` vs `alpha_analytic=0.680`
       (off by `0.248`, exactly matching the printed global `alpha_maxerr` -- so
       the worst `alpha` error *is* at this corner, even in the pure t=0 case).
     - **Narrows the search substantially**: no need to look at `MHD_C2P`/hydro
       coupling, RK sub-stepping, or repeated-cycle accumulation at all -- the
       first-ever CFC solve on known-exact input already fails. The bug is
       somewhere in the ~35-task chain itself: `AssembleVectorSource`/the `P_i`/
       `eta` linear solves, `ComputeADualFromX`, `SolveConformalFactor`'s Newton
       iteration, `RescaleMatterSources`, `SolveLapse`'s Gauss-Seidel solve, the
       ghost-exchange/`cfc_bcs.cpp` physical-BC rounds between them, or
       `AssembleLapseShiftK`'s final pointwise assembly. Round 8's corner-adjacent
       defect concentration is still the strongest lead (now further supported by
       `alpha`'s worst error recurring at that exact corner even at `t=0`) --
       round 8's three next-angle suggestions (corner ghost-cell fill correctness,
       Newton behavior at the high-density point, non-octant domain to isolate
       corner-fill from high-density-point causes) remain the natural next steps,
       now with added confidence they're looking in the right general area since
       fluid coupling has been ruled out as a contributing factor. Not yet
       root-caused. Temporary instrumentation removed after use, as in every
       prior round.
   - **Round 10 (per user request: rule out the *reflecting*-corner hypothesis
     specifically, by testing a full non-octant domain with no reflecting
     boundaries anywhere)**: new input file
     `inputs/dyn_grmhd/cfc_tov_full.athinput` -- same star (`rhoc=1.28e-3`,
     `kappa=100`), same resolution (`dx=1.6`, 64 cells per axis, so identical
     per-solve cost to the octant test), but centered in a full box
     (`x1,x2,x3 in [-51.2,51.2]`) with `diode` (-> `mg_zerofixed`) on all six
     faces and no `reflect` anywhere -- so `mg_mesh_bcs_[f]` is `mg_zerofixed` on
     every face for both nonlinear solvers, with no `mg_zerograd` faces and no
     reflecting-symmetry corner at all. Reran round 9's `DebugCFCSolveAtT0`
     diagnostic (reapplied then re-reverted, identical mechanism), extended to
     also report the domain-relative location (fraction along each axis) of the
     worst `psi`/`alpha` error, not just the corner-cell value, since this domain
     has no reflecting corner to specifically check.
     - **Result** (`run_cfc_tov_full/run1.log`): `psi_maxerr=0.992` at
       `frac_along_axis=(0,0,1)` -- a genuine corner of the box (where `x3min`,
       `x2min`, `x1max` meet), even though every face there is a plain
       `mg_zerofixed` face with no reflecting symmetry involved at all.
       `alpha_maxerr=0.254` at `frac_along_axis=(0.508,0.508,0.508)` -- the
       domain's geometric *center*, i.e. right at the star's core, nowhere near
       any boundary.
     - **Conclusion: this looks like two separate bugs, not one.**
       - `psi`'s error is still large and still corner-concentrated, but the
         corner is now a plain `mg_zerofixed`-`mg_zerofixed`-`mg_zerofixed`
         meeting point, not a reflecting/`mg_zerograd` one. This rules out
         anything specific to reflecting symmetry or `mg_zerograd`'s even-mirror
         formula as the cause -- the real issue is more general: `MGRootBoundary`
         filling ghost cells one axis at a time in sequence (x1, then x2 using
         the just-filled x1 ghosts, then x3 using both) may simply not compose
         correctly at any point where 2 or 3 faces' fills overlap, regardless of
         which `BoundaryFlag` those faces carry. Round 8's suspicion of the
         reflecting corner specifically is superseded by this more general
         "any multi-face corner" framing.
       - `alpha`'s error moved entirely away from any corner and onto the star's
         core instead -- a location with no boundary-fill explanation at all,
         pointing at something in the lapse solve itself (`SolveLapseTask`/
         `MGCFCLapse`'s Gauss-Seidel kernel, or the `RescaleSrcTask` source it
         consumes) misbehaving specifically where the matter source is largest,
         independent of whatever's wrong with `psi` at the corners. That
         `alpha`'s worst error happened to sit at the same corner cell in
         rounds 8-9's octant tests now looks like it may have been coincidence
         (or a secondary/downstream effect of `psi`'s corner error feeding into
         `alpha`'s own equation, since `K(x)` in eq. 74 depends on `psi`) rather
         than `alpha` having its own independent corner problem.
     - **Two separate next angles, not one**: (1) for `psi`, inspect
       `MGRootBoundary`'s per-axis-sequential ghost fill
       (`multigrid_driver.cpp:1847-1922`) directly at a corner cell to see
       whether the x2/x3 passes correctly incorporate the x1 pass's results (or
       clobber/ignore them) -- this is now suspected to be a general multigrid
       ghost-fill bug potentially affecting `gravity` too if it ever runs a
       single-meshblock, all-`mg_zerofixed` configuration, not CFC-specific,
       worth flagging carefully if confirmed; (2) for `alpha`, examine
       `SolveLapseTask`/`RescaleSrcTask`/`MGCFCLapse`'s Gauss-Seidel kernel
       independently of any corner question, focusing on behavior at/near the
       highest-density point. Not yet root-caused for either. Temporary
       instrumentation removed after use; `cfc_tov_full.athinput` and
       `run_cfc_tov_full/` are kept (unlike the driver.cpp/dyngr_tov.cpp
       instrumentation) since they're reusable test scaffolding, not debug code.
   - **Round 11 (per user request: focus on `psi` only for now -- `alpha`'s error
     may just be a downstream effect of `psi`'s, so it's set aside until `psi` is
     understood -- and test whether round 10's corner error is a real bug or an
     artifact of the outer boundary being too close, by doubling the domain
     half-width)**: before running anything, a back-of-envelope check using the
     star's known mass (`Mass: 1.40016` from the pgen's own log output) against the
     analytic exterior falloff `psi = 1 + M/(2r)`: at round 10's corner
     (`r = 51.2*sqrt(3) = 88.7`), the true deviation from `psi=1` out there is only
     `1.40016/(2*88.7) = 0.008` -- two orders of magnitude below the observed
     `psi_maxerr=0.992`, suggesting the corner error is not simple finite-domain
     Dirichlet truncation. Tested this directly rather than trusting the estimate
     alone: new input `inputs/dyn_grmhd/cfc_tov_full_2x.athinput` (kept, reusable
     scaffolding like round 10's `cfc_tov_full.athinput`) -- same star, same
     `dx=1.6` resolution, but half-width doubled `51.2 -> 102.4` (`nx` doubled
     `64 -> 128` per axis to hold resolution fixed while only the boundary moves,
     isolating domain-size from resolution effects), still all `diode`
     (-> `mg_zerofixed`) on every face. Reran round 9-10's `DebugCFCSolveAtT0`
     diagnostic unchanged (reapplied to `driver.cpp`/`dyngr_tov.cpp`, including the
     `ptov_params` unconditional-populate patch, then fully reverted after use --
     confirmed via `git diff --stat` showing empty output on both files, and a
     clean rebuild afterward). Built and ran on `/sakura/ptmp/tlam/` (scratch
     space) rather than the home-directory `~/athenak_cfc/build_cfc` used in
     earlier rounds: the home-directory build hit `Disk quota exceeded` mid-compile,
     traced to ~9.3 GB of accumulated `core.sakura01.*` crash dumps in `~/tlam`
     left over from earlier debugging sessions (some pre-dating this investigation
     entirely) -- removed, and a fresh build tree set up under
     `/sakura/ptmp/tlam/athenak_cfc_build` (source still read from
     `~/athenak_cfc`) to keep future large runs off the quota-limited home
     filesystem. `run_cfc_tov_full_2x/` (also on scratch,
     `/sakura/ptmp/tlam/run_cfc_tov_full_2x/`) holds `run1.log`.
     - **Result**: `psi_maxerr=0.996022` at `frac_along_axis=(0.996,0.004,0.004)`,
       `r=175.976` -- essentially unchanged from round 10's `0.992` at the
       half-width-51.2 domain's corner (if anything, very slightly larger), even
       though the boundary moved twice as far away and the back-of-envelope
       truncation estimate at this new, farther corner is smaller still
       (`1.40016/(2*175.976) = 0.004`). The error stayed pinned to a domain corner
       in both cases (`frac_along_axis` component values sit at/near `0` or `1` on
       all three axes in both round 10 and round 11 -- the exact corner identity
       shifted since round 10 used `ix1_bc=ix2_bc=ix3_bc=diode` symmetric bounds
       and this run's data-dependent worst point simply landed on a different one
       of the box's 8 corners, not evidence against the "any multi-face corner"
       framing). `alpha_maxerr=0.247404` at `frac_along_axis=(0.504,0.504,0.504)`
       (the domain center/star's core again, essentially unchanged from round 10's
       `0.254`), consistent with `alpha`'s error being independent of domain size
       too, as expected for an issue unrelated to any boundary.
     - **Conclusion: this rules out finite-domain Dirichlet truncation as the
       (or a significant) cause of `psi`'s corner error.** Doubling the domain --
       which should have roughly halved a true truncation-driven error, per the
       `M/(2r)` scaling -- left `psi_maxerr` unchanged to 3 significant figures.
       Combined with round 10's result (reflecting vs. plain `mg_zerofixed`
       corners both fail identically) and the original back-of-envelope estimate
       (predicted truncation error two orders of magnitude too small to explain
       the observed one even before running anything), the `psi` corner error is
       now on strong footing as a genuine bug, most likely in `MGRootBoundary`'s
       sequential per-axis (x1, then x2, then x3) ghost-cell fill
       (`multigrid_driver.cpp:1847-1922`) not composing multiple faces' Dirichlet
       data correctly at a true corner cell -- not a symptom of the domain being
       "too small" for the physics. **Next step (per user direction to focus on
       `psi` only): read `MGRootBoundary` directly and check what happens at a
       corner cell where 2 or 3 faces are filled in sequence** -- does the x2 pass
       see/preserve the x1 pass's result, does x3 see both, or does a later pass
       overwrite an earlier one's corner contribution? `alpha` remains set aside
       per this round's framing (its error is unchanged in both magnitude and
       location by this test, still consistent with it being either a downstream
       effect of `psi`'s own error via eq. 74's `K(x)` term, or an independent
       lapse-solve issue -- not distinguished by this test, deliberately not
       investigated further this round).
   - **Round 12 (per user direction: read `MGRootBoundary` and check the corner-fill
     composition directly)**: before touching code, hand-traced both
     `MultigridDriver::MGRootBoundary` (`multigrid_driver.cpp:1826-2020`, the
     distributed-root-grid fill) and `MultigridDriver::PhysicalBoundary`
     (`multigrid_tasks.cpp:129-350`, the per-MeshBlock finest-level fill that
     actually governs the Newton-smoothed `psi` solution in these single-MeshBlock
     tests) -- both share the same sequential x1-then-x2-then-x3 structure, each
     pass looping over the *full* (ghost-inclusive) extent of the other two axes.
     Worked a concrete 3D example by hand (`ngh=1`, all faces `mg_zerofixed`,
     interior corner value `A`): pass 1 (x) writes wrong garbage at the true
     corner (reads an as-yet-unfilled ghost row), but pass 2 (y) *overwrites* the
     x1-x2 edge at the interior-k row correctly (`= +A`, using pass 1's already-
     valid face fill), and pass 3 (z), last, overwrites the true 3-way corner using
     that now-valid edge value (`= -A`) -- exactly the correct triple-reflection
     result. The scheme is self-correcting by construction for any BC formula that
     only mirrors along its own axis while holding the other two indices fixed
     (`mg_zerofixed`/`mg_zerograd`/`periodic`, everything used in these tests) --
     the *last* pass to touch a given ghost cell always wins, and by induction its
     source is already validated by the earlier passes in the sequence. This
     directly contradicts round 10-11's leading hypothesis.
     - **Verified empirically, not just on paper**: added a small, surgical
       temporary probe (`MGCFCConformalFactorDriver::DebugPrintCornerGhosts()`,
       `mg_cfc_conformal_factor.hpp`/`.cpp`, called once right after `SolveMG()`
       returns in `Solve()`) that reads back the finest-level `delta_psi` array
       and prints the interior cell adjacent to the round-11 corner
       (`ox1`/`ix2`/`ix3`, all `mg_zerofixed` for `cfc_tov_full_2x.athinput`)
       alongside its 2-face-edge and 3-face-corner ghost mirrors, comparing
       against the hand-derived expected values (`edge = +A`, `corner = -A`).
       Rebuilt (scratch space) and ran the normal `cfc_tov_full_2x.athinput`
       task graph directly (no need to reapply the `DebugCFCSolveAtT0` replay
       machinery this time -- this check is a pointwise algebraic identity on the
       multigrid's own ghost fill, true regardless of convergence state, so the
       ordinary per-stage `CFC_SolvePsi` task exercises it for free).
     - **Result** (`/sakura/ptmp/tlam/run_cfc_tov_full_2x_probe/run1.log`):
       `edge_diff=0.000000e+00` and `corner_diff=0.000000e+00` **exactly**, on
       every RK stage sampled. The multigrid's internal ghost fill composes
       corners perfectly correctly -- round 10-11's leading hypothesis
       (`MGRootBoundary`/`PhysicalBoundary`'s sequential fill not composing
       multi-face corners) is **retracted**. This was a real, reasoned hypothesis
       that looked structurally suspicious, but both the hand trace and the
       empirical probe now rule it out conclusively.
     - **Conclusion / where this leaves the `psi` corner error**: the O(1) error
       at corner-adjacent interior cells is real (rounds 8-11) but is *not* a
       ghost-fill composition bug. Originally logged two remaining candidates
       here; **candidate (2), the CFC-level post-solve `MeshBoundaryValuesCC`
       exchange for `psi` (`pbval_psi`), is retracted** (caught by the user while
       reviewing this round -- see round 13's note below for the full reasoning):
       `SolveConformalFactor` (`cfc.cpp`, called by `SolvePsiTask`) ends with
       `AssembleConformalMetric(pmy_pack, psi)`, which takes `psi` as
       `const DvceArray5D<Real>&` (`cfc_reconstruct.cpp:137`) and never writes
       back into it -- so `psi` is fully determined once `SolvePsiTask` returns,
       strictly *before* `CFC_RestPsi`/`SendPsi`/`RecvPsi`/`ProlongPsi` (later,
       separate nodes in `QueueCFCTasks()`, `cfc.cpp:318-325`) ever run. Every
       `psi_maxerr` measurement in rounds 9-13 was taken by replaying only up
       through `SolvePsiTask` and reading `psi` back immediately after, so that
       post-solve exchange was never even exercised by any of these diagnostics
       -- and even if it had run, it only touches mesh-level *ghost* cells, never
       the interior cells (`i=ie`, etc.) these rounds have been tracking. The
       sole remaining candidate is **(1) `MGCFCConformalFactor`'s own
       Newton-Gauss-Seidel kernel** (`SmoothPack`/`ConformalFactorRHS`,
       `mg_cfc_conformal_factor.cpp`) or another step inside
       `SolveConformalFactor`'s own solve path itself (`LoadMatterSource`,
       `LoadNonlinearCoefficient`, `RetrieveSolution`, the `+1` offset pass --
       all in-scope, all executed, all upstream of what every round's diagnostic
       has read back) -- possibly misbehaving specifically for interior cells
       adjacent to multiple ghost faces at once (e.g. a coloring/indexing issue,
       or the `psi_floor` positivity clamp firing pathologically in this
       far-field, near-vacuum region where `Utilde`/`Ahat^2` are both ~0).
       Temporary probe fully reverted after use (confirmed via `git diff --stat`:
       `mg_cfc_conformal_factor.hpp`
       shows no diff, `.cpp` matches exactly the permanent round-6/7 fix's diff
       stat), and a clean rebuild confirms the revert compiles.
   - **Round 13 (per user request: double the resolution to check whether this is
     just a convergence problem)**: same domain as round 11-12's
     `cfc_tov_full_2x.athinput` (half-width 102.4, all faces `mg_zerofixed`), but
     `dx` halved (`1.6 -> 0.8`, `nx` doubled `128 -> 256`) -- new input
     `inputs/dyn_grmhd/cfc_tov_full_2x_hires.athinput` (kept, reusable scaffolding
     like the other `cfc_tov_full*` inputs). Needed rounds 9-12's isolated t=0
     `DebugCFCSolveAtT0` replay again (a full normal run's per-cycle multigrid
     solves are too slow to wait out repeatedly at this resolution -- confirmed
     directly this round: a first attempt just running the ordinary task graph
     and hooking the comparison through `TOVHistory` instead took over 7 minutes
     for a single cycle to reach its first history output and was abandoned, see
     below), but the prior rounds' version of that diagnostic was never committed
     and had already been reverted out of the working tree, so it isn't preserved
     anywhere -- **reconstructed from scratch** this round, directly off
     `cfc.cpp`'s `QueueCFCTasks()` ordering (`cfc.cpp:271-361`) rather than
     memory: replays `SolveVecXTask` through `ComputeADualTask` (the full
     X-vector-potential chain `Ahat^2` depends on) then `SolvePsiTask`, skipping
     `MHD_C2P`/`RescaleSrcTask`/`SolveLapseTask`/the shift solve entirely (not
     needed for a `psi`-only check, same reasoning as before: `w0` at t=0 is
     already exactly the pgen's analytic primitives). Comparison target and
     reporting logic also rebuilt from first principles rather than recalled:
     analytic `psi(r) = sqrt(FindSchwarzschildR(r,mass)/r)`, the exact relation
     `dyngr_tov.cpp`'s own `SetADMVariablesToTOV` uses to set the initial guess
     (lines 162/218-219 and 543/577-578) -- confirmed this reconstruction is
     faithful by first rerunning the *unchanged* `cfc_tov_full_2x.athinput` and
     getting back round 11's exact number, `psi_maxerr=0.996022` at the same
     corner, before trusting the new hi-res input's result.
     - **Abandoned approach, noted for the record**: first tried hooking the
       comparison through `TOVHistory` (which already runs every history-output
       dt and has `ptov_params` in scope in the same translation unit) instead of
       replaying the CFC task graph by hand, reasoning that a pointwise
       comparison against the *converged* per-cycle state should be just as valid
       as an isolated t=0 solve and would avoid needing to reconstruct the replay
       machinery. This was correct in principle but impractical: with `nlim=10`
       and `dt`-limited timesteps, a single cycle's worth of real (non-isolated)
       multigrid solves -- 2 nonlinear solves (`psi`, `alpha*psi`) x 3 RK stages,
       each hitting the same 40-iteration non-convergence -- took over 7 minutes
       of wall time before the first history output even fired, confirmed via a
       run that hit a wall-clock timeout mid-cycle-1 with zero diagnostic prints
       despite cycle 0 having fully completed. Reverted (`git diff --stat` clean
       on `dyngr_tov.cpp` before starting the replay reconstruction) in favor of
       the isolated-solve replay, which reproduces a comparable result in
       seconds.
     - **Result**: hi-res (`/sakura/ptmp/tlam/run_cfc_tov_full_2x_hires/run_replay.log`):
       `psi_maxerr=0.996037` at `frac_along_axis=(1,0,0)`, `r=176.669` --
       essentially identical to the baseline-resolution run's `psi_maxerr=0.996022`
       at `r=175.976` (same corner; the small `r` shift is just the finer grid's
       cell-center landing fractionally closer to the true corner). Doubling
       resolution changed the error by 1.5e-5 in absolute terms, a 0.0015%
       relative change -- effectively zero, and nowhere near what halving the
       discretization error of an under-resolved or slowly-converging feature
       should produce.
     - **Conclusion: this rules out under-resolution and slow/incomplete Newton
       convergence as the (or a significant) cause of `psi`'s corner error.**
       Combined with round 11 (domain size doesn't matter) and round 12
       (ghost-fill composition is provably correct), three independent axes --
       domain size, resolution, and ghost-fill algorithm -- have now all been
       ruled out. The error is not a truncation, convergence, or boundary-fill
       artifact; whatever is wrong lives specifically in the per-point physics/
       numerics of the corner-adjacent interior cells themselves. (Round 12's
       other candidate, the CFC-level post-solve `MeshBoundaryValuesCC` psi
       exchange, was separately retracted after this round -- see the note added
       to round 12's writeup above -- since every `psi_maxerr` measurement,
       rounds 9-13 alike, is read back before that exchange ever runs.) This
       leaves `SolveConformalFactor`'s own solve path (`cfc.cpp`) -- most likely
       the Newton-Gauss-Seidel kernel itself
       (`SmoothPack`/`ConformalFactorRHS`, `mg_cfc_conformal_factor.cpp`) -- as
       the sole remaining candidate; next round should instrument it directly
       rather than continue testing external variables.
       Temporary diagnostic fully reverted after use (confirmed via
       `git diff --stat` showing empty output on `dyngr_tov.cpp` and
       `driver.cpp`), and a clean rebuild confirms the revert compiles.
   - **Round 14 (per user suggestion: substitute the analytic TOV solution into the
     discretized equation directly and check the residual, to test whether the
     equation itself is implemented correctly independent of whether the Newton-
     Gauss-Seidel solver converges to it)**: for a static (`v_pert=0`), non-rotating
     TOV star, `Ahat^2 = 0` identically everywhere (no extrinsic curvature in a
     static conformally-flat spacetime) and `psi_analytic = 1+M/(2r)` is the *exact*
     solution of the vacuum Laplace equation eq. 73 reduces to away from the star --
     so injecting the analytic profile and evaluating the discrete residual should
     give ~roundoff, not O(1), if the discretization is correct.
     - **Implementation**: reused rounds 9-13's `SolveVecXTask`-through-
       `ComputeADualTask` replay unchanged (populates the real `u_tilde`/`a_sq`),
       but instead of calling `SolvePsiTask` (the real Newton solve), built the
       analytic `delta_psi = sqrt(FindSchwarzschildR(r,mass)/r) - 1` array on the
       host over the *entire* domain (interior and ghost alike, valid inside the
       star too via `FindSchwarzschildR`'s interior interpolation-table branch, not
       just the `r > 2*R_edge_iso` restriction rounds 9-13 used) and injected it
       directly into the multigrid's finest-level `u_` via one new temporary method,
       `MGCFCConformalFactorDriver::DebugCheckAnalyticResidual` (mirrors
       `LoadMatterSource`/`LoadNonlinearCoefficient`'s existing offset-aware
       ngh-vs-lngh copy pattern, `mg_cfc_conformal_factor.cpp`), which then calls
       the solver's own *unmodified* `Multigrid::CalculateDefectPack()`
       (`def(m,0,k,j,i) = RHS(u) - Laplacian(u)/dx^2`, `mg_cfc_conformal_factor.cpp:
       133-155`) and reads the result back via one new temporary accessor,
       `MGCFCConformalFactor::DefAtLevel(int l)` (same public-accessor-for-
       cross-hierarchy-access shape as the existing, permanent `CoeffAtLevel`).
       Reusing the solver's own residual code (rather than re-deriving the RHS
       formula by hand in the diagnostic) was deliberate: it means this check has
       zero risk of the diagnostic itself introducing a formula mismatch that could
       be mistaken for a solver bug. Reports both the global max `|residual|` and,
       separately, the residual specifically at the round 8-13 corner (max-x1,
       min-x2, min-x3) -- added after a first pass showed the global max landing at
       the star's core, not the corner (see below), so a corner-specific readout
       was needed to actually test the hypothesis at hand rather than an unrelated
       one.
     - **Result** (`/sakura/ptmp/tlam/run_cfc_tov_full_2x_r14/run_residual2.log`,
       `/sakura/ptmp/tlam/run_cfc_tov_full_2x_hires_r14/run_residual.log`):
       at the corner, `residual_at_corner=6.0105e-10` (baseline, 128^3) and
       `6.19764e-10` (hi-res, 256^3) -- **essentially machine precision, unchanged
       across resolutions**. The global max `|residual|` (`0.0118535` baseline,
       `0.0136989` hi-res) lands at the star's *core* (`(k,j,i)=(64,64,64)` /
       `(128,128,128)`, the domain center), not the corner -- ordinary, expected
       truncation error where the density/RHS is steepest, unrelated to the corner
       investigation (not chased further this round; noted for completeness only).
     - **Conclusion: the discretized equation (stencil, `ConformalFactorRHS`, and
       the `Utilde`/`Ahat^2` coefficients feeding it) is correct at the corner.**
       This directly confirms -- not just by elimination, but by positive proof
       that the true fixed point exists and satisfies the discrete equations to
       roundoff -- that round 12-13's remaining candidate is right: the bug is in
       the *iterative* Newton-Gauss-Seidel solve process itself, not the equation
       it's solving. This sharpens the picture further: a correct fixed point
       exists and is locally self-consistent (near-zero residual) at the corner,
       yet the solver converges to a value ~0.99 away from it. That combination --
       small *local* residual but wrong *global* value, in a region where the
       source term is exactly zero (pure vacuum Laplace's equation, which is only
       satisfied by the *correct* smooth 1/r-falloff solution given the right
       long-range boundary information, not by just any locally-flat
       configuration) -- is the classic signature of a multigrid V-cycle whose
       *coarse-grid correction* isn't propagating long-wavelength information
       correctly, leaving point relaxation to satisfy the local stencil in a
       self-consistent but globally-wrong way. Also consistent with rounds 11/13's
       observation that `SolveIterative` needs far more than the expected ~10-20
       V-cycles for grid-size-independent multigrid convergence (`"Failed to
       converge after 40 iterations"`, defect still 4-5 orders above threshold) --
       true multigrid convergence should not degrade with iteration count this way
       regardless of domain size. **Next round's leading hypothesis: check this
       solver's V-cycle coarse-grid correction specifically** -- `RestrictCoefficients()`
       (this solver's own hand-written `Utilde`/`Ahat^2` restriction, since generic
       `Multigrid::LoadCoefficients()` can't be reused here, see this file's header
       docstring) and `CalculateFASRHSPack()`'s tau-correction are the two most
       likely places a nonlinear-solver-specific restriction bug could silently
       corrupt what the coarse levels correct against, without affecting the
       fine-level residual check just performed here at all. A useful complementary
       check: compare the defect-reduction factor per V-cycle iteration against
       what a correctly functioning multigrid should show (roughly constant,
       ~0.1-0.3x per cycle, independent of resolution) rather than the apparent
       slow/stalling behavior observed so far.
     - Temporary diagnostic fully reverted after use (confirmed via `git diff
       --stat`: `dyngr_tov.cpp`/`driver.cpp`/`mg_cfc_conformal_factor.hpp` show no
       diff, `mg_cfc_conformal_factor.cpp` matches exactly the permanent round-6/7
       fix's diff stat), and a clean rebuild confirms the revert compiles. Decided
       against leaving this round's specific injection/residual-check scaffolding
       in place for reuse (despite it being a live option) since round 15's new
       leading hypothesis (coarse-grid correction / convergence-rate) needs
       different instrumentation entirely -- this round's code already did its job
       and gave a clean, conclusive answer.
   - **Round 15 (per user direction: dig into the coarse-grid correction) -- TWO
     bugs found and fixed, one masking the other**:
     - **Bug 1 (real, but not the corner root cause): `MGCFCConformalFactor::
       SmoothPack`/`CalculateDefectPack` silently dropped the FAS `src_`
       tau-correction.** Comparing against the generic linear pattern
       (`Multigrid::Smooth`/`CalculateDefect`, `multigrid.hpp:315-360`, used
       correctly by `gravity/mg_gravity.cpp`) and against this file's own
       (correct) `CalculateFASRHSPack`, which properly accumulates
       `src_ += lap(ubar)/dx^2 - RHS(ubar)`: `SmoothPack`'s Newton step
       (`u_new = u_old - omega*(lap - rhs*dx2)/fprime`) and `CalculateDefectPack`'s
       residual (`def = rhs - lap*idx2`) both never read `src_` at all, even though
       this file's own header docstring explicitly says `src_`'s "entire role here
       is the FAS correction accumulator." Traced the V-cycle sequencing
       (`multigrid_driver.cpp:677-693`) and confirmed it's otherwise a
       textbook-correct FAS down-sweep -- this was the only broken link. Since
       `src_` is always exactly zero at the finest level (nothing restricts into
       it there), this bug has zero effect at the finest level -- consistent with
       round 14's residual check (evaluated at the finest level) showing
       machine-precision agreement despite this bug already being present at the
       time. **Fixed**: added `src(m,0,k,j,i)` into both formulas (`mg_cfc_
       conformal_factor.cpp`, `SmoothPack`/`CalculateDefectPack`). Verified via
       `mg_verbose=2`: defect went from `"Failed to converge after 40 iterations,
       defect=2.7e-6"` to a textbook multigrid trajectory (`~5x` reduction per
       iteration, `5.9e-11` in 8 iterations) -- a real, substantial improvement,
       kept. **But `psi_maxerr` was completely unchanged (`0.996022`, bit-for-bit)
       after this fix alone** -- the actual corner catastrophe was untouched.
     - **Bug 2 (the actual corner root cause): `RetrieveSolution` passed the
       wrong ghost depth to `Multigrid::RetrieveResult`.** Investigated the
       "beautiful convergence, unchanged wrong answer" puzzle by printing the raw
       solved `psi` (not just the error) plus a full radial trace: `psi_num=2`
       exactly, uniformly, at the corner cell *and* its immediate neighbors (both
       interior and ghost) -- a suspiciously round number. Prompted by the user's
       question about the initial guess for `u`: confirmed `psi` is correctly
       initialized to `1.0` at construction (`cfc.cpp:178`, with a comment
       explicitly flagging this exact concern), so `u=1.0` wasn't simply an
       untouched initial guess -- something moved it there. The radial trace
       (`i=4` to `i=128`, stepping by 4) showed a smooth profile tracking the
       analytic one reasonably (small, explainable lag -- see below) all the way
       out to `i=128`, then a *sharp* jump to `psi=2` at `i=131` (`=ie`, the true
       last interior cell) -- a 3-cell-wide anomaly, not a gradual departure.
       Comparing all four multigrid solvers' `RetrieveResult` calls:
       `gravity/mg_gravity.cpp:242` and `cfc/mg_cfc_vector_poisson.cpp:242` both
       correctly pass the *mesh's* `NGHOST` (`indcs.ng`, e.g. 4); `cfc/mg_cfc_
       conformal_factor.cpp:358` and `cfc/mg_cfc_lapse.cpp:296` (alpha's
       identical bug) both instead passed `mglevels_->GetGhostCells()` -- this
       solver's own, generally much shallower, internal ghost depth (`ngh_=1` for
       this test). `RetrieveResult`'s copy offset is `dst_off = ngh - ngh_`;
       passing `ngh_` in place of the mesh's true `NGHOST` collapses this to `0`
       instead of the correct `3` (`4-1`), so the fully-solved interior gets
       copied into `dst` (`psi`, mesh-`NGHOST`-deep) **unshifted** -- landing 3
       cells too low relative to where `dst`'s own indexing expects it. A 3-cell
       position error is nearly invisible where the profile is smooth and slowly
       varying (matching the small, uniform-looking lag seen across most of the
       radial trace), but the outermost mesh-interior cells (`ie-2` through `ie`)
       fall completely outside the (mis-aligned) copy loop's write range and are
       *never written at all* -- retaining their pre-solve `1.0` default, which
       the unconditional `+1.0` post-pass in `SolveConformalFactor` then turns
       into exactly `2.0`, matching the observed value precisely. **Fixed**:
       changed both `RetrieveResult(dst, 0, mglevels_->GetGhostCells())` calls
       (`mg_cfc_conformal_factor.cpp`, `mg_cfc_lapse.cpp`) to
       `RetrieveResult(dst, 0, pmy_pack_->pmesh->mb_indcs.ng)`, matching gravity's
       and `MGCFCVectorPoissonDriver`'s already-correct pattern.
     - **Result, both fixes together** (`/sakura/ptmp/tlam/run_cfc_tov_full_2x_r15/
       run_verify3.log`, `/sakura/ptmp/tlam/run_cfc_tov_full_2x_hires_r15/
       run_verify.log`): `psi_maxerr` dropped from `0.996022` to `0.0247179`
       (baseline, 128^3) / `0.0248592` (hi-res, 256^3) -- a ~40x reduction, and
       critically the worst-error location **moved from the domain corner
       (`r=176`) to just outside the star's surface (`r≈16.4`, `R_edge_iso≈8.1`)**,
       a far more ordinary place for residual solver error to concentrate. The
       corner cells are now smoothly consistent with the interior profile
       (`psi_at_ie_plus1=1.019`, not `2.0`). The remaining `~2.5%` discrepancy
       near the star is consistent with the *expected* single-solve "lagged
       Utilde" effect inherent to the XCFC scheme (`AssembleVectorSource`'s
       `psi^6` factor uses the *previous* stage's converged psi, standard
       lagged-coefficient structure per this file's own comments) -- this
       diagnostic only ever runs one isolated solve, not the multi-stage/
       multi-cycle iteration a real run would use to refine it further, so this
       residual isn't itself surprising and wasn't chased further this round.
     - **This round's corner-catastrophe investigation (rounds 8-15) is
       concluded**: the O(1) `psi` error was a genuine, confirmed, two-part bug
       (a masked-but-real FAS correction gap, plus a silent result-copy
       misalignment that was the actual dominant cause), now fixed and verified
       at two resolutions.
     - **Two open follow-ups noted, not yet investigated**: (1) `alpha` has the
       identical `RetrieveSolution` bug (`mg_cfc_lapse.cpp`, now also fixed) and
       almost certainly shares the same root cause as its own O(0.1-0.25) core
       error from rounds 8-10 -- not directly re-verified this round (would need
       extending the replay chain through `MHD_C2P`/`RescaleSrcTask`/
       `SolveLapseTask`, not just `SolvePsiTask`); (2) a full (non-isolated,
       ordinary task-graph) run of `cfc_tov_full_2x.athinput` still shows 3
       `"Failed to converge"` messages in cycle 0 (down from 6 before this
       round's fixes, but not zero) -- not yet root-caused; could be `alpha`'s own
       solve, or the vector-potential (`p_x`/`eta_x`/`p_beta`/`eta_beta`) Poisson
       solves, which were spot-checked this round (`mg_cfc_vector_poisson.cpp`'s
       `SmoothPack`/`CalculateDefectPack` correctly thread `src_` through their
       shared `SmoothChannels`/`CalculateDefectChannels` helpers, unlike bug 1
       above) but not fully ruled out.
     - Verification diagnostic (`DebugCFCSolveAtT0`, reconstructed fresh again
       this round from `cfc.cpp`'s `QueueCFCTasks()` ordering, same as round 13)
       fully reverted after use (confirmed via `git diff --stat`: `dyngr_tov.cpp`/
       `driver.cpp` show no diff), leaving only the two permanent fixes in
       `mg_cfc_conformal_factor.cpp`/`mg_cfc_lapse.cpp`. Clean rebuild confirmed
       both with and without the temporary diagnostic present.
   - **Round 16 (per user request: verify the full CFC solver for both psi and alpha
     at t=0)**: reconstructed `DebugCFCSolveAtT0` again (same mechanism as rounds
     9/11/13/14/15), this time extended all the way through `RescaleSrcTask`,
     `SolveLapseTask`, `SolveShiftTask`, and `AssembleFinalTask` (not stopping at
     `SolvePsiTask`), so both `psi` (the member array) and the *final assembled*
     `alpha` (`pmy_pack->padm->u_adm`'s `I_ADM_ALPHA` channel -- the actual field
     downstream consumers read, not a hand-computed `alpha_psi/psi`) could be
     checked against the analytic isotropic TOV solution in one pass. Bypassed
     `MHD_C2P` as before (`w0` is already exact at t=0).
     - **First result, single isolated solve** (`cfc_tov_full_2x.athinput`,
       128^3): `psi_maxerr=0.1021` and `alpha_maxerr=0.1262`, both located at the
       star's *core* (`r=1.386`) -- much larger than round 15's reported
       `psi_maxerr=0.0247` near the star's surface, and initially looked like a
       possible new, unfixed bug. Repeated at hi-res (256^3,
       `cfc_tov_full_2x_hires.athinput`): `psi_maxerr=0.1059`,
       `alpha_maxerr=0.1298`, essentially unchanged -- **not resolution-convergent**,
       which briefly looked concerning (a genuine truncation effect should shrink
       substantially between 128^3 and 256^3).
     - **Diagnosed via a targeted intermediate check**: added interior-only vs.
       full-domain (ghost-inclusive) error reporting after each task, plus raw
       `psi_num`/`psi_analytic` printouts at the worst point. This showed the
       *interior* error was already `0.1021` immediately after `SolvePsiTask`
       itself (unchanged by every later task, confirming `psi` is never modified
       after `SolveConformalFactor` returns, as expected) -- so this is not a
       ghost-exchange-timing artifact, and not related to round 15's fix. Root
       cause: `AssembleVectorSource`'s `u_tilde = psi^6 * U` deliberately uses the
       *previous stage's converged* `psi` (cfc.cpp's own documented lagged-
       coefficient convention, matching the paper's XCFC iteration structure) --
       but in this *isolated, single-solve* diagnostic starting cold from the
       pgen's uniform `psi=1` initial guess, "previous stage" is literally `psi=1`
       everywhere, not the true converged profile. At the star's core
       `psi_analytic≈1.19`, so `psi_analytic^6≈4.0` -- using `1.0` instead
       under-sources eq. 73 by a factor of ~4 exactly where the density is
       highest, which is large enough to explain the observed core error and
       is *not* resolution-dependent (it's a wrong-input-coefficient effect, not
       a discretization-error effect), matching what was observed.
     - **Verified by iterating the isolated solve**: reran the exact same full
       task-chain replay a second, third, and fourth time in the same process
       (cheap: `psi`/`x_u`/etc. are persistent member arrays, so a second replay
       naturally uses the just-solved `psi` as the next "previous stage" input,
       exactly like consecutive RK stages would in a real run). Result (both
       resolutions): `psi_maxerr` and `alpha_maxerr` both drop by more than an
       order of magnitude after just one extra iteration (128^3:
       `psi 0.1021->0.00681`, `alpha 0.1262->0.0180`; 256^3: `psi 0.1059->0.00682`,
       `alpha 0.1298->0.0167`) and then stay **bit-for-bit stable** for two further
       iterations -- a genuine, resolution-stable fixed point. **Conclusion: the
       large core error was the expected single-solve "lagged psi^6" cold-start
       artifact, not a new bug** -- a real run never starts every stage from a
       flat `psi=1` guess, so this doesn't apply outside this isolated diagnostic.
       This also re-confirms round 15's fixes are working correctly: self-
       consistently iterated, both `psi` and `alpha` converge to a small,
       resolution-stable residual (`~0.7%` / `~1.7-2%`).
     - **Genuinely new finding, not previously confirmed**: every single call to
       `SolveLapseTask` (all 4 iterations, both resolutions) printed
       `### FATAL ERROR in MultigridDriver::SolveIterative -- Failed to converge
       after 40 iterations`, with the stalled defect *increasing* with resolution
       (128^3: `4-8e-6`; 256^3: `1.3e-5`-`2.6e-5`) -- atypical for correctly
       functioning multigrid, which should converge in a roughly resolution-
       independent number of V-cycles. No such message was ever printed for
       `psi`'s own solve or for any of the four vector-potential Poisson solves
       (`p_x`/`eta_x`/`p_beta`/`eta_beta`) in these runs. **This directly confirms
       and narrows down the round-15 open follow-up (2)** ("3 `Failed to converge`
       messages in a full run... could be alpha's own solve or the vector-potential
       Poisson solves") -- it is specifically the lapse (`alpha`) solve. The
       resulting `alpha` error is still small and slightly *improves* with
       resolution (`0.0180` -> `0.0167`), so this does not look like a correctness
       bug, but the non-convergence itself is unexplained and worth investigating
       (Finding A in the item-3 addendum above already notes the lapse equation is
       a linear Helmholtz solve, not Newton -- a different code path from `psi`'s
       nonlinear solver but sharing the same FAS/coarse-grid-correction
       machinery that round 15 found and fixed one bug in for `psi`; an analogous
       bug specific to the lapse solve's own `SmoothPack`/`CalculateDefectPack`/
       `CalculateFASRHSPack` has not yet been ruled out).
     - Temporary diagnostic fully reverted after use (confirmed via `git diff
       --stat`: `driver.cpp`/`dyngr_tov.cpp` show no diff), clean rebuild confirmed.
     - **Not yet investigated**: root-causing the lapse solve's non-convergence
       (this round's new finding); the pre-existing round-15 follow-up (2)'s other
       half (whether the vector-potential Poisson solves ever contribute to a full
       run's "Failed to converge" count) is now less likely given this round's
       observation that they never triggered the message in any of these isolated-
       solve tests, but a full (non-isolated) run hasn't been re-checked since
       round 15's fixes to see if its count is now fully explained by the lapse
       solve alone.
   - **Round 17 (per user's own code review of `LapseReactionCoeff`)**: the user
     independently spotted a real bug while reading `mg_cfc_lapse.cpp`:
     `LapseReactionCoeff` takes three coefficients (`Utilde+2*Stilde`, `psi`,
     `Ahat^2`) as arguments, none of which depend on the equation's own unknown
     `alpha*psi` -- and the pre-round-17 code stored all three as separate `coeff_`
     channels (`ncoeff_=3`), restricted independently to every coarser V-cycle
     level via the generic (linear, per-channel) `Multigrid::RestrictCoefficients`,
     then recombined them nonlinearly (via `psi^-2`, `psi^-8`) *inside* `SmoothPack`/
     `CalculateDefectPack`/`CalculateFASRHSPack` at *every* level, including coarse
     ones. The user's suggested fix: since `K(x) = LapseReactionCoeff(...)` doesn't
     depend on the unknown, compute it once at the finest level and restrict *that*
     single coefficient down through the hierarchy, instead of restricting its raw
     ingredients separately and recombining them at each level.
     - **Why this is a real bug, confirmed by re-deriving the FAS math**: for a
       linear restriction operator `R`, `R(f(a,b)) != f(R(a), R(b))` whenever `f`
       is nonlinear -- true here (`psi^-2`/`psi^-8`). This is different from `psi`'s
       own nonlinear solver (`mg_cfc_conformal_factor.cpp`), which also mixes
       `Utilde`/`Ahat^2` (restricted independently, but each appears *linearly* in
       that equation's RHS) with `psi` itself -- but there, `psi = u+1` is the
       *local unknown being solved for*, correctly carried through the standard FAS
       `u_` restriction (which *is* the FAS-correct treatment for a genuinely
       nonlinear unknown), not a separately-restricted "coefficient". In the lapse
       equation, `psi` and `Ahat^2` are both already-converged, externally-fixed
       fields from earlier steps -- genuine coefficients, not this equation's
       unknown -- so restricting them independently and recombining nonlinearly at
       each coarse level gives every level below the finest a systematically wrong
       `K(x)`, i.e. an inconsistent coarse-grid operator. This is exactly the kind
       of defect that produces round 16's observed symptom (`SolveIterative`
       failing to converge, with the stalled defect *growing* rather than shrinking
       at higher resolution -- deeper V-cycles exercise more, and more divergent,
       coarse levels).
     - **Fix implemented**: `MGCFCLapseDriver::LoadMatterSource`/`LoadKnownFields`
       (two separate loaders, three `coeff_` channels) replaced with a single
       `LoadReactionCoefficient(u_plus_2s_tilde, psi, a_sq, ngh)` that evaluates
       `LapseReactionCoeff` once per point, at the finest level only, and writes
       the resulting `K(x)` into `coeff_` channel 0 (`ncoeff_` dropped `3` -> `1`
       in both `MGCFCLapse`'s and `MGCFCLapseDriver`'s constructors).
       `SmoothPack`/`CalculateDefectPack`/`CalculateFASRHSPack` now read `K(x)`
       directly from `coeff(m,0,k,j,i)` instead of recomputing it from three
       channels. `TransferCoeffToRoot` needed no code change (already generic over
       `ncoeff_`). `cfc.cpp`'s `SolveLapse` updated to call the new combined
       loader. `mg_cfc_lapse.cpp`'s file-header comment and `mg_cfc_lapse.hpp`'s
       loader docstring rewritten to document the fix and the FAS reasoning above.
     - **Verified via the same iterated `DebugCFCSolveAtT0` replay as round 16**
       (reapplied, then reverted again after use -- `git diff --stat` confirms
       `driver.cpp`/`dyngr_tov.cpp` show no diff). Result (`cfc_tov_full_2x.athinput`,
       128^3, 4 iterations): `alpha_maxerr` improved from `0.0180` to `0.01351` at
       the stable fixed point, and its worst-error location moved from the star's
       core (`r=1.386`, independent of `psi`'s own worst point) to exactly match
       `psi`'s (`r=101.6`) -- both now dominated by the same general residual
       source rather than `alpha` having its own independent core-localized
       problem. A real, worthwhile improvement, kept.
     - **However, the fix did NOT resolve the "Failed to converge" message**:
       `SolveLapseTask` still triggers `### FATAL ERROR in MultigridDriver::
       SolveIterative -- Failed to converge after 40 iterations` on every single
       call, post-fix, with a similar stalled defect (`3.6e-6`-`6.3e-6`, vs.
       `4.1e-6`-`8.4e-6` pre-fix -- marginally better but still failing the
       `1e-10` threshold). **Conclusion: the nonlinear-coefficient-restriction bug
       was real and worth fixing (it measurably improved `alpha`'s accuracy and
       eliminated its independent core-error localization), but it is not the sole
       cause of the lapse solve's non-convergence** -- something else is still
       preventing `SolveIterative` from reaching its threshold. Not yet
       root-caused; the search should continue elsewhere (candidates not yet
       checked: `mgroot_`'s coarsest-grid direct solve reading `K(x)` correctly;
       whether `TransferCoeffToRoot`'s single-cell-per-block copy is consistent
       with `RestrictCoefficients()` already having been called first; whether the
       screened/Helmholtz operator's coarse-grid smoothing factor is fundamentally
       different from the plain-Laplacian case `SolveIterative`'s convergence
       expectations were tuned against).
   - **Round 18 (per user's own code review, again -- found by reading
     `mg_cfc_lapse.cpp` directly): the actual remaining cause of the "Failed to
     converge" non-convergence, and it's the exact same bug class as round 15's
     Bug 1, just in the lapse solver instead of the conformal-factor solver.**
     The user noticed `MGCFCLapse::SmoothPack` never reads `src_` at all -- and
     confirmed `CalculateDefectPack` doesn't either, even though
     `CalculateFASRHSPack` (unchanged, already correct) faithfully accumulates the
     FAS tau-correction into it every V-cycle descent. Exactly like round 15's
     original finding for `MGCFCConformalFactor`: the coarse-grid correction was
     computed and stored, then silently discarded, leaving every level below the
     finest smoothing/computing its defect against its own homogeneous equation,
     decoupled from the fine grid's actual defect -- a textbook explanation for
     "V-cycle relaxes locally fine but never actually converges."
     - **Fix**: added `src(m,0,k,j,i)` into both `SmoothPack`'s Newton/GS update
       (`fval = lap + dx2*kx*(u_old+1.0) - dx2*src(m,0,k,j,i)`) and
       `CalculateDefectPack`'s residual (`def = (-kx*(u+1) + src) - lap*idx2`),
       mirroring `MGCFCConformalFactor::SmoothPack`/`CalculateDefectPack`'s round-15
       fix exactly (same sign convention, re-derived from `CalculateFASRHSPack`'s
       existing, already-correct `src += lap*idx2 + kx*(u+1)` accumulation to keep
       the three functions consistent).
     - **Verified via the same iterated `DebugCFCSolveAtT0` replay as rounds 16-17**
       (reapplied, then reverted -- `git diff --stat` confirms `driver.cpp`/
       `dyngr_tov.cpp` show no diff), both resolutions: **zero** `"Failed to
       converge"` messages across all 4 iterations, at both 128^3 and 256^3 --
       completely resolved, not just improved. `alpha_maxerr` at the stable fixed
       point: `0.01352` (128^3) / `0.01355` (256^3) -- resolution-stable, and
       essentially unchanged from round 17's post-fix-but-still-stalling value
       (`0.01351`/`0.01355`), confirming the round-17 coefficient-restriction fix
       had already gotten `alpha`'s *accuracy* right; round 18 was purely about
       actually reaching genuine multigrid convergence rather than stalling near a
       similar-looking answer.
     - **Both the psi (rounds 8-15) and alpha (rounds 16-18) corner/convergence
       investigations under item 9 are now concluded.** All four fixes (FAS `src_`
       for psi, `RetrieveResult` ghost depth for both psi and alpha, FAS-consistent
       `K(x)` coefficient restriction for alpha, FAS `src_` for alpha) are
       committed. **Confirmed**: a full (non-isolated), ordinary task-graph-driven
       run of `cfc_tov_full_2x.athinput` (all 10 cycles, `128^3`) completes cleanly
       with **zero** `"Failed to converge"` messages anywhere in the log -- the
       fix holds under the real per-stage pipeline, not just the isolated
       `DebugCFCSolveAtT0` replay. No outstanding open follow-ups remain from this
       thread; item 9 is closed.

10. ~~Seed the multigrid initial guess for `psi`/`alpha_psi` from the problem
    generator's own ADM data, instead of always cold-starting from flat space.~~
    **Done for `psi`/`alpha_psi`; deliberately skipped for `beta` -- see below.**
    Confirmed (round 19 investigation) that `Driver::Initialize()` writes the t=0
    output using the pgen's raw analytic metric (e.g. `dyngr_tov.cpp`'s
    isotropic-gauge TOV solution, written directly into `padm->adm.alpha`/
    `g_dd`/`psi4`/`beta_u`/`vK_dd`) -- CFC's task graph (`"stagen"`) hasn't run
    even once at that point, so the *first* V-cycle CFC ever runs used to start
    from a cold Kokkos-zero guess, ignoring that a much better guess (the pgen's
    own `psi4^(1/4)`, `alpha`) was already sitting in `padm`.
    - **Mechanism**: `Multigrid::LoadFinestData` (already used by
      `gravity::MGGravityDriver::Solve()` for its own warm start, but never
      wired up for CFC) copies a caller-supplied field straight into the
      V-cycle's finest-level solution array. Added a one-line public wrapper,
      `SeedInitialGuess(guess, ngh)`, to both `MGCFCConformalFactorDriver` and
      `MGCFCLapseDriver` (`mg_cfc_conformal_factor.hpp/.cpp`,
      `mg_cfc_lapse.hpp/.cpp`).
    - **Where the seeding happens**: two new one-shot `bool` flags on `CFC`
      (`psi_seeded_`, `alpha_psi_seeded_`, `cfc.hpp`), checked at the top of
      `SolveConformalFactor`/`SolveLapse` (`cfc.cpp`) and never reset -- every
      call after the very first is left alone (its own natural multigrid warm
      start from whatever the previous stage/cycle converged to). Seeding can't
      happen in `CFC`'s constructor: `pcfc` is constructed *before* the
      problem generator runs (`meshblock_pack.cpp` vs. `main.cpp`'s
      `new ProblemGenerator(...)` call), so `padm->adm` would still hold its own
      pre-pgen default, not the real initial data. Waiting for the first actual
      `SolveConformalFactor`/`SolveLapse` call (which only happens once the main
      loop's `"stagen"` list runs, well after the pgen -- or after a restart
      finishes loading its checkpoint) sidesteps that ordering problem for both
      a fresh run and a restart alike, with no special-casing needed.
    - **What's actually copied**: `psi`'s own backing array is reused as scratch
      -- written with `delta_psi = pow(padm->adm.psi4, 0.25) - 1` over the
      interior, handed to `SeedInitialGuess`, then immediately overwritten by
      `RetrieveSolution`'s genuinely-converged answer a few lines later (safe,
      since nothing reads the scratch value in between). `alpha_psi` mirrors
      this with `delta_(alpha*psi) = padm->adm.alpha * psi - 1`, using
      `padm->adm.alpha` (still the pgen's/restart's raw value -- 
      `AssembleLapseShiftK`, the task that overwrites it, runs much later this
      same stage) times `psi` (this *stage's* own just-converged value from
      `SolveConformalFactor`, a strictly better multiplier than re-deriving a
      guess from `padm->adm.psi4`, which `AssembleConformalMetric` already
      overwrote earlier the same stage).
    - **`beta` deliberately not seeded**: the multigrid unknowns for the shift
      are `u_p_beta`/`eta_beta` (Shibata's `P_i`/`eta` potentials), not `beta^i`
      itself -- there is no cheap pointwise inversion from a given `beta^i` back
      to `(P_i, eta)` (that would require solving another elliptic-like problem,
      defeating the purpose of a free warm start). For the TOV pgen this is moot
      anyway: the initial data has `beta^i = 0` identically, which is already
      exactly what `u_p_beta`/`eta_beta`'s existing cold-zero default produces
      after reconstruction -- so there is currently nothing to gain here. Left
      as a known gap for a future pgen with genuinely nonzero initial shift.
    - **Verified**: rebuilt cleanly; a single-rank smoke test
      (`cfc_tov.athinput`, `mg_verbose=2`, 3 cycles) shows every solve block
      (including the now-seeded `psi`/`alpha_psi` ones) converging smoothly
      with no `"Failed to converge"`/`FATAL` messages, e.g. the conformal-factor
      solve's first call: initial defect `8.25e-04` -> `9.7e-11` in 11
      iterations. Not yet A/B-benchmarked against the pre-seed iteration count
      (would require reverting and rebuilding a second time); the change is
      narrow enough (only the initial guess of an already-correct, already-
      converging solve) that this wasn't treated as blocking.

11. ~~Converge `padm->adm` to the (fixed) initial primitives at t=0, instead of
    using the problem generator's raw analytic metric guess as-is.~~ **Done.**
    Item 10 established that CFC's task graph never runs before the t=0 output,
    so the pgen's raw metric guess (e.g. a 1D TOV profile interpolated onto the
    3D grid) is what the first timestep actually starts from -- not the true
    self-consistent CFC solution for that matter distribution, causing a
    mismatch that only gets corrected (lagged by one step) once evolution
    begins. Fixed with a genuine fixed-point iteration, holding the primitives
    (`pmhd->w0`) exactly as the pgen set them and iterating the *metric*:
    - **Why iteration, not one pass**: conserved variables are metric-dependent
      functions of the (fixed) primitives (`D = sqrt(gamma)*rho*W`,
      `S_i = sqrt(gamma)*rho*h*W^2*v_i`), so refreshing `u0` from `w0` after a
      metric update changes `u0`, which changes `X^i`'s source, which changes
      `psi`, which changes the metric again.
    - **Scope-narrowing insight**: only `{X^i, psi}` are mutually coupled (via
      `cons`'s metric-dependence and `Adual^ij`); lapse and shift don't feed
      back into either equation, so they're solved once, after `{X^i, psi}`
      converge, exactly like the existing per-stage pipeline's ordering. Also,
      since the recent `AssembleVectorSource` simplification, `U-tilde`/
      `S-tilde_i` don't read `psi` at all anymore, so `psi` itself needs no
      ghost exchange *during* the iteration -- only `p_x`/`eta_x` (before
      `ReconstructVectorFromPotentials`) and `x_u` (before `ComputeADualFromX`)
      do, since those are the only fields actually finite-differenced inside
      the `{X^i, psi}` subsystem.
    - **Mechanism**: `dyngr::DynGRMHD::PrimToConInit(is,ie,js,je,ks,ke)`
      (`dyn_grmhd.cpp`, already existed for the pgen's own first conserved-
      variable fill) is called once per iteration -- it reads
      `padm->adm.g_dd` directly, so calling it again after each metric update
      is exactly "update conservative variables instead (PrimToCons)". No
      con2prim ever runs during this procedure, so it's immune to the
      `cons_floor`/`cons_adjusted` staleness discussed earlier in round 19 --
      primitives genuinely never change.
    - **New `CFC::InitializeMetric(Driver *pdriver)`** (public; takes a real
      `Driver*`, not `nullptr` -- the per-field multigrid solves' own internal
      iteration-cap failure path writes `pdriver->nlim = ...`, which would
      segfault on null): a hand-written loop calling CFC's existing private
      step methods (`SolveVectorPotential`, `ReconstructVectorPotential`,
      `ComputeADual`, `SolveConformalFactor`) and `Rest*/Send*/Recv*/Prolong*`
      task methods directly, entirely outside the `NumericalRelativity` task
      graph -- reusing `"stagen"` isn't possible here since a pass through it
      also flux-updates/RK-evolves the hydro state, which must not happen
      during this one-time initialization. `Recv*Task` calls are spun in a
      `while (... != TaskStatus::complete) {}` busy-wait, since nothing else is
      driving the retry the way the real task-list scheduler normally would.
    - **A real deadlock hazard, designed around explicitly**: the existing
      `CFC_InitRecv`/`ClearSend`/`ClearRecv` (round 19's earlier fix) post/wait
      on all 8 `MeshBoundaryValuesCC` instances at once. `InitializeMetric`'s
      loop only sends/receives 3 of them (`p_x`, `eta_x`, `x_u`) per iteration
      -- reusing the all-8 versions would post `MPI_Irecv`s for the other 5
      that never get a matching send that iteration, and `ClearRecv`'s
      `MPI_Wait` on those would hang. Added two new scoped pairs instead:
      `InitRecvXFields`/`ClearXFields` (3 fields, called once per iteration)
      and `InitRecvTailFields`/`ClearTailFields` (the remaining 5, called once
      for the one-shot lapse/shift/final tail).
    - **Convergence check**: `max|psi_new - psi_old|` over the interior,
      `MPI_Allreduce(..., MPI_MAX)`-reduced across ranks (mirrors
      `MultigridDriver::CalculateDefectNorm`'s own reduction pattern) --
      tracking `psi` alone was confirmed sufficient per discussion (it's the
      one field both matter-coupling paths feed into). New `<cfc>` params:
      `init_iter_max` (default 50 -- see below), `init_tol` (default `1e-10`),
      `init_verbose` (default false, prints `max|delta psi|` per iteration).
    - **Non-convergence is a warning, not fatal** (per discussion): prints a
      `### WARNING` and proceeds with the current (non-converged) metric rather
      than aborting the run.
    - **Hooked into `Driver::Initialize()`** as a new "Step 1b", right after
      `InitBoundaryValuesAndPrimitives` and before the timestep calculation
      (so `NewTimeStep`'s light-crossing estimate sees the converged `alpha`,
      not the raw guess) -- guarded by `!res_flag && pmb_pack->pcfc != nullptr`.
      Restarts are untouched: a restart's checkpointed metric is already
      self-consistent with its checkpointed conserved variables (built up by
      ordinary evolution, not this procedure), so re-deriving it from
      primitives at restart time would discard that consistency, not just
      redundantly recompute it.
    - **Verified**: rebuilt cleanly. Single-rank smoke test (`cfc_tov.athinput`,
      `init_verbose=true`, `init_tol=1e-10`) converges to `max|delta psi| = 0`
      in 24 iterations; a 2-rank pure-face-neighbor isolation test (mirroring
      the one that originally exposed the round-19 ghost-exchange bug)
      converges in 21 iterations with no deadlock, no `FATAL`, no `NaN`, and a
      finite, sane t=0 mass. The initial default `init_iter_max=20` was in fact
      too tight for this test case (residual `5.7e-7` at iteration 20, short of
      `1e-10`) -- exercised the warn-and-continue path correctly, but prompted
      raising the default to `50` for headroom. Convergence is geometric/linear
      (ratio roughly `0.6-0.7` per iteration for this test), not quadratic --
      consistent with a Picard-style (not Newton-style) fixed-point coupling
      between `PrimToCons` and the CFC solve; matches the O(10-100)-iteration
      initial-data convergence behavior reported by other XCFC codes, not a
      sign of a bug.

12. **AMR support for CFC (supersedes open item 3b).** In progress. Investigation
    (re-reading the full octet/AMR machinery in `src/multigrid/` end to end, and
    every CFC multigrid driver, not just re-stating item 3b's original note) found
    the gap is narrower than 3b assumed, plus two additional bugs in *shared*
    (non-CFC) code that item 3b's investigation hadn't surfaced.
    - **Already AMR-capable, confirmed by reading the actual code, not assumed**:
      `MGCFCVectorPoisson`/`MGCFCScalarPoisson` (the linear solvers backing `P_i`/
      `eta` for both `X^i` and `beta^i`) already have real `SmoothOctet`/
      `CalculateDefectOctet`/`CalculateFASRHSOctet` bodies (`mg_cfc_vector_
      poisson.cpp`/`mg_cfc_scalar_poisson.cpp`) -- a direct port of gravity's
      `OctLaplacian` pattern, generalized to 3 channels for the vector case, no
      `nreflevel_` guard on `Solve()`. These equations (`Delta P^i = S^i`) have no
      point-varying coefficient, so they never needed `coeff_` and were never
      blocked by Finding D. CFC's own mesh-level ghost exchange for every field
      (`p_x`/`eta_x`/`x_u`/`psi`/`alpha_psi`/`p_beta`/`eta_beta`/`u_adm`, via
      `MeshBoundaryValuesCC`/`RestrictCC`/`ProlongateCC`) also already rides on the
      same generic, already-AMR-capable machinery hydro/MHD/z4c use, with
      `coarse_*` shadow arrays already gated on `multilevel`.
    - **The real gap is isolated to the two nonlinear solvers** (`MGCFCConformal-
      Factor` for `psi`, `MGCFCLapse` for `alpha*psi`) -- their equations have
      genuinely point-varying coefficients (`Utilde`/`Ahat^2`/`K(x)`, in `coeff_`),
      and `MGOctet` carries no coefficient storage at all (confirmed by reading its
      full definition, `multigrid.hpp` -- only `u`/`def`/`src`/`uold` pointers).
      Six concrete pieces, traced through the actual V-cycle call chain:
      1. `MGOctet` has no `coeff` pointer/`Coeff()` accessor/`ncoeff` member.
      2. `MultigridDriver::InitializeOctets()` allocates/wires `oct_u_buf_`/
         `oct_def_buf_`/`oct_src_buf_`/`oct_uold_buf_` but nothing analogous for
         coefficients.
      3. `TransferCoeffToRoot()` (the CFC-local helper item 3's Finding C already
         added, mirroring `MultigridDriver::TransferFromBlocksToRoot`) explicitly
         has no octet-parented branch -- its own comment says so verbatim
         (`// nreflevel_ == 0 is already guaranteed by Solve()'s guard ... no
         octet-parented branch to handle`). `TransferFromBlocksToRoot` itself
         *does* have this branch already (writes `oct.Src`/`oct.U` for blocks
         refined past the root level) -- `TransferCoeffToRoot` needs the
         `oct.Coeff(...)` equivalent.
      4. No octet-level coefficient restriction exists. `u`/`src` restrict
         octet-to-octet via `PreRestrictOctetU`/`RestrictOctetsBeforeTransfer`/
         `RestrictOctets`, all built on free functions `RestrictOne`/
         `RestrictOneSrc`/`RestrictOneDef` (simple 8-child averages,
         `multigrid.hpp`). Since `coeff_` is static for the whole solve (loaded
         once before `Solve()`, never touched by smoothing), it needs its own
         **one-time** restriction pass (new `RestrictOneCoeff` + a loop mirroring
         `PreRestrictOctetU`'s structure), run once right after
         `TransferCoeffToRoot()`, not every V-cycle iteration. Confirmed `coeff_`
         needs no boundary/ghost exchange at the octet level at all: every
         existing `SmoothPack`/octet-`Smooth` reads `coeff`/`Src` only at the
         exact point being updated, never at a neighbor offset (unlike `u`, the
         only reason `cbuf_`/`SetBoundariesOctets` exist).
      5. `SmoothOctet`/`CalculateDefectOctet`/`CalculateFASRHSOctet` are
         `std::exit(EXIT_FAILURE)` stubs. The Newton math itself doesn't need
         re-deriving: `ConformalFactorRHS(u, u_tilde, ahat_sq, &rhs, &drhs_du)`
         and `LapseReactionCoeff(...)` are already plain `KOKKOS_INLINE_FUNCTION`s
         taking scalars, not views -- reusable as-is at octet scale. Only
         `ConformalFactorLap`/`LapseLap` (templated on a 5D view, `u(m,0,k,j,i)`)
         need octet-indexed counterparts (`oct.U(0,k,j,i)`), mechanically
         identical in shape to gravity's own `OctLaplacian`.
      6. Remove the `nreflevel_ > 0` `FATAL` guards in both `Solve()`s once 1-5
         land.
    - **Two more gaps found, both in *shared* (non-CFC) code, not previously
      recorded anywhere**:
      - `Multigrid::ncoeff_` is never initialized in the base class constructor
        (`multigrid.cpp:36-39`'s init list sets `nvar_` but not `ncoeff_`) -- for
        gravity (which also never sets it) this is a genuinely uninitialized
        `int`, implicitly relied on to "happen to be 0." Worth fixing regardless
        of AMR.
      - `Multigrid::ReallocateForAMR()` (the generic per-block-count-change
        handler, called from `PrepareForAMR()` whenever AMR creates/destroys
        MeshBlocks on a rank) resizes `u_`/`src_`/`def_`/`uold_` per level but
        never `coeff_`. This is a real bug independent of octets entirely: if AMR
        ever changes `nmmb_` while a CFC nonlinear solver is live, `coeff_`
        silently stays the old size, and the next `LoadMatterSource`/
        `LoadNonlinearCoefficient` call reads/writes out of bounds or stale data.
        Two-line fix (`if (ncoeff_ > 0) Kokkos::realloc(coeff_[l], ...)` in the
        existing per-level loop), safe for gravity since `ncoeff_` stays `0`
        there -- but depends on the previous bullet's fix first, so the guard is
        well-defined rather than reading garbage.
    - **Plan / implementation order** (chosen so the one piece with blast radius
      beyond CFC is validated in isolation first):
      1. The two shared base-class fixes above, verified against a gravity AMR
         smoke test before touching CFC at all.
      2. `MGOctet`/`InitializeOctets` coefficient plumbing (points 1-2).
      3. `TransferCoeffToRoot`'s octet branch + new `RestrictCoeffOctets`
         (points 3-4).
      4. Real `SmoothOctet`/`CalculateDefectOctet`/`CalculateFASRHSOctet` bodies +
         guard removal for `MGCFCConformalFactor`, verified in isolation.
      5. Repeat for `MGCFCLapse`.
      6. An end-to-end AMR TOV test (no existing CFC `.athinput` sets
         `multilevel`/refinement -- a new one is needed to exercise any of this).
    - **Files**: `src/multigrid/multigrid.hpp`/`.cpp`/`multigrid_driver.cpp`
      (shared -- `MGOctet`, `RestrictOneCoeff`, `oct_coeff_buf_`, `ncoeff_` init,
      `ReallocateForAMR`); `src/cfc/mg_cfc_conformal_factor.hpp`/`.cpp`,
      `src/cfc/mg_cfc_lapse.hpp`/`.cpp` (octet bodies, `TransferCoeffToRoot`,
      `RestrictCoeffOctets`, guard removal); a new AMR-enabled `.athinput`.

    - **Implemented, all 6 steps above.** Additional finding made partway through
      step 4's verification, not anticipated in the original plan: `TransferCoeff-
      ToRoot` only ever populated `mgroot_`'s own *finest* internal V-cycle level
      (the root grid can itself span multiple levels, e.g. 4x4x4 -> 2x2x2 -> 1x1x1,
      whenever there are enough root-level blocks/octets) -- every coarser root
      level's `coeff_` was left at its post-construction default (0), corrupting
      the FAS coarse-grid correction from those levels. Fixed by calling
      `mgc_root->RestrictCoefficients()` (already-existing, generic `Multigrid`
      method -- the same one `mglevels_->RestrictCoefficients()` already applies
      to the per-block hierarchy) at the end of `TransferCoeffToRoot`, for both
      drivers. Also added `<cfc>` `mg_npresmooth`/`mg_npostsmooth` (default 1,
      matching the base class): AMR-refined meshes need more smoothing per level
      to fully converge than a uniform mesh does, exactly matching
      `binary_gravity.athinput`'s own top-of-file advice ("more smoothing
      (npresmooth/npostsmooth = 2 or 3), or refinement = none") -- a known,
      pre-existing characteristic of this multigrid implementation at refinement
      boundaries, confirmed by observing gravity's own AMR test plateau
      similarly (~3.9e-6) well before reaching a tight tolerance, using only the
      base-class default of 1.
    - **Open, not yet root-caused**: with `<cfc> init_tol=1e-10`/`mg_threshold=
      1e-10` (both tuned for the uniform-resolution case), a small single/few-
      octet AMR test's inner V-cycle solve (`psi`) plateaus at a *stable,
      deterministic* residual (~1.4e-4, reproducibly the same value run to run,
      not noise) regardless of `mg_npresmooth`/`mg_npostsmooth` (tried 1, 3, 6 --
      partial improvement 1->3, none 3->6) -- suggestive of a small, bounded,
      not-yet-understood discretization inconsistency at the refinement boundary,
      separate from the `mgroot_` restriction bug above (still present after that
      fix). Not chased further this round. Important context that *does*
      distinguish "real bug" from "just needs a looser tolerance," though:
      the **outer** `CFC::InitializeMetric` fixed-point loop (item 11) converges
      smoothly through this the entire time (`max|delta psi|` geometric, ratio
      ~0.577/iteration, matching the non-AMR case's ~0.6-0.7 almost exactly) --
      i.e. the physics is still converging correctly overall, via an inexact-
      Newton/Picard-style tolerance to the inner solve's own imprecision; the
      residual floor is a property of the *inner* V-cycle only. Practical
      consequence fixed regardless of root cause: `MultigridDriver::
      SolveIterative`'s hard-coded 40-iteration soft-failure path sets
      `pdriver->nlim = pmesh->ncycle`, meant to gracefully truncate the *main*
      evolution loop -- but `InitializeMetric` runs *before* that loop starts, so
      repeated inner failures there were silently zeroing `nlim` and terminating
      the whole run after 0 cycles even though the metric itself converged fine.
      Fixed with a save/restore of `nlim` around the `InitializeMetric` call in
      `Driver::Initialize()` (`driver.cpp`), independent of whether the residual
      floor itself ever gets root-caused.
    - **Verified**: rebuilt cleanly. Gravity AMR regression (`binary_gravity.
      athinput`, separate build/pgen, 232 MeshBlocks, 5 refinement levels) shows
      identical defect norms before and after the two shared base-class fixes --
      confirms no regression from touching `src/multigrid/`. New CFC AMR test
      (`cfc_tov_amr*.athinput`, not yet committed to `inputs/`): 32^3 root mesh,
      8^3 meshblock (4x4x4=64 root blocks), static refinement around the stellar
      core --
      - **1 refinement level** (1-8 octets depending on region size, with
        `mg_threshold=5e-4`/`mg_npresmooth=mg_npostsmooth=2` to work around the
        open residual-floor item above): completes all 3 requested cycles
        cleanly, `nlim` intact, zero NaN/FATAL, only transient warnings, mass
        conserved to ~0.06-0.09% (reasonable for this loose a tolerance on a
        short, coarse test).
      - **2 refinement levels** (`num_levels=3`, nested `refined_region1`/
        `refined_region2`, "Octet level 0: 8 octets" + "Octet level 1: 8
        octets"): the first real exercise of the octet-to-octet coefficient
        restriction path (`RestrictCoeffOctets`'s actual loop body is a no-op
        whenever `nreflevel_<=1`, so the 1-level test above never touched it).
        Also completes all 3 cycles cleanly, no NaN/FATAL, mass sane.
      - Not yet verified: the user's actual production configuration
        (`inputs/.../cfc_tov_stability`-style setup via
        `/sakura/ptmp/tlam/athenak_run/cfc_stability`, `num_levels=5`,
        `refined_region1` at `level=4`) is far deeper than either test above
        (4 refinement levels vs. 1-2) -- that specific run was observed to be
        progressing cleanly (cycle 1295/tlim=100, mass drift ~6e-8 over 40 time
        units, zero warnings in its log) but almost certainly under an
        intermediate build predating some of the fixes in this item (job started
        at 14:23:19, shared build directory last rebuilt 14:27:36), so it isn't
        by itself confirmation of the current code.
13. **Store `delta_psi`/`delta_alpha_psi` (`psi - 1`, `alpha*psi - 1`), not the
    physical `psi`/`alpha*psi`, as `cfc::CFC`'s persistent members.** Done.
    Both fields asymptote to 1 far from the star, so a `Real` holding the
    physical value loses precision exactly where the *deviation* is most
    interesting (e.g. far-field metric perturbations many orders of magnitude
    below 1) -- storing the deviation directly keeps those significant digits.
    This is also exactly the unknown the multigrid solve already iterates on
    internally (`RetrieveSolution` always handed back the raw deviation, see
    item 3's Finding A / addendum #3), so this change also **removes** the
    per-solve `+1.0` pointwise pass in `SolveConformalFactor`/`SolveLapse` that
    used to convert it, rather than adding new work.
    - Renamed `cfc::CFC::psi`/`alpha_psi` -> `delta_psi`/`delta_alpha_psi`
      throughout `cfc.hpp`/`cfc.cpp` (constructor init list, task-graph
      `Rest/Send/Recv/ProlongPsi`/`AlphaPsi` methods, `InitializeMetric`'s
      convergence check -- a difference of two same-convention values, so
      unaffected either way -- and the two one-shot seeding blocks in
      `SolveConformalFactor`/`SolveLapse`, which already computed exactly this
      delta and needed no formula change beyond the rename, except the
      `alpha_psi` seed's `adm.alpha * psi` multiply, which now reconstructs
      the physical `psi` with `+1.0` first). Constructor's flat-space initial
      value changed from `deep_copy(psi, 1.0)` to `deep_copy(delta_psi, 0.0)`
      to match.
    - Every consumer that needs the physical field now reconstructs it inline
      (`+1.0`) at the point of use -- a small, fixed set of call sites, found
      by grepping every read of the old `psi`/`alpha_psi` members:
      `AssembleConformalMetric`/`AssembleLapseShiftK` (`cfc_reconstruct.cpp`,
      `psi^4`/`alpha=alpha_psi/psi`/`vK_dd=Adual/psi^2`), `AssembleVectorSource`'s
      `for_shift=true` branch and `BuildShiftSource` (`cfc.cpp`, `alpha*psi^-6`
      for the eq. 75 source), and `MGCFCLapseDriver::LoadReactionCoefficient`
      (`mg_cfc_lapse.cpp`/`.hpp`, `K(x) = 2pi(Ũ+2S̃)psi^-2 + (7/8)Ahat^2 psi^-8`
      needs the physical `psi`). `RescaleMatterSources` needed no change --
      it already avoids `psi` entirely via the `psi^6 == sqrt(detg)` identity
      (item from the previous simplification pass). `MGCFCConformalFactor`'s
      own solve needed no change either -- it never took an external `psi`
      argument; its nonlinear `psi^-7` terms are built from the local FAS
      unknown `u+1` inside `mg_cfc_conformal_factor.cpp`, already exactly this
      same delta convention.
    - **Side effect, not the goal but worth noting**: `MeshBoundaryValues::
      CFCScalarBCs`' `vacuum` physical-BC case hardcodes ghost cells to `0.0`
      (`cfc_bcs.cpp`) -- correct for `eta_x`/`eta_beta` (Dirichlet-zero
      potentials) but was silently wrong for `psi`/`alpha_psi` under the old
      physical-value convention (should have been `~1`, not `0`, at a true
      vacuum boundary) if that BC flag were ever actually selected there. This
      refactor makes that case correct for `delta_psi`/`delta_alpha_psi` too,
      as a side effect -- not otherwise exercised by any test in this
      investigation (all use `reflect`/`diode`, whose zero-gradient copy is
      convention-independent either way).
    - **Verified**: rebuilt cleanly; grepped for every remaining bare `psi(`/
      `alpha_psi(` member read across `src/cfc/` to confirm none were missed.
      No behavior change intended for any already-passing test (the physical
      values read back out via `+1.0` are bit-for-bit the same physics, just
      relocated); not separately re-run against the AMR/stability tests from
      item 12 as of this writing.
