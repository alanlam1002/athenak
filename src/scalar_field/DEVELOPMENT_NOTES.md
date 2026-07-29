# Scalar-Tensor Extension: Development Notes

Status: **Phases 0-6 done.** Scalarized-star initial data (Phase 5, an external-reader
pgen rather than the in-repo shooting solve originally planned) and physical-boundary
ghost-zone filling for the scalar field (Phase 6) are both built and verified. Still
open: whether the MPA1 star's evolution instability (Phase 5) is a genuine physical
instability of that configuration or a numerical/coupling issue (unresolved even after
Phase 6's fix, see below); a quantitative Yukawa `exp(-m*r)/r` falloff check against
SACRA (Phase 4's still-missing verification item); and `Z4c::ADMConstraints` scalar-
awareness (see the dedicated gap note below). This directory implements a massive
scalar-tensor (Damour-Esposito-Farese-type) gravity sector for AthenaK, ported from
`~/SACRA_2D/SACRA_MPI/bssn_st.f90` and cross-checked against arXiv:2406.05211 ("Binary
neutron star mergers in massive scalar-tensor theory", Lam, Kuan, Shibata, Van Aelst,
Kiuchi). The full architecture decision record, physics equations, and 5-phase roadmap
are in `PLAN.md` in this same directory (copied in-repo so it survives independently of
any external planning tool). This note tracks what's actually been done, so future work
(and future sessions) can pick up context without re-reading the whole plan.

## What Phase 0 actually did

New module `scalarfield::ScalarField`, architected as a sibling to `z4c::Z4c`/`adm::ADM`/
`Tmunu`, wired into `MeshBlockPack` and the `NumericalRelativity` task graph exactly like
those classes are. **The scalar field is completely inert right now**: `CalcRHS` zeroes
the RHS unconditionally, so `(sphi, Pi)` stay frozen at whatever the problem generator's
initial data set them to. Zero lines were touched in `z4c/z4c_calcrhs.cpp` or
`z4c/z4c_tasks.cpp` -- the Z4c sector is provably untouched when `<scalarfield>` is
absent, and provably non-back-reacting even when present.

New files (`src/scalar_field/`):
- `scalar_field.hpp/.cpp` -- class definition + constructor. 2-variable state
  `{I_SF_SPHI, I_SF_PI}`, own `u0/u1/u_rhs/coarse_u0` arrays, `Options` struct parsed from
  a new `<scalarfield>` input block (`omega_c`, `beta0`, `mass2`, `sphi0`, `diss`). Phase 4
  adds `user_Sbc`; the `newton_tol`/`newton_maxiter` fields that used to live here
  (anticipating an implicit mass-term solve) were removed in Phase 4 once that plan
  changed to a fully explicit treatment -- see "What Phase 4 actually did" below.
- `scalar_field_calcrhs.cpp` -- `CalcRHS<NGHOST>`, currently a no-op (RHS = 0).
- `scalar_field_update.cpp` -- `ExpRKUpdate`, a byte-for-byte copy of `z4c_update.cpp`'s
  generic weighted-average RK update with `nvar = nscalarfield`.
- `scalar_field_tasks.cpp` -- `QueueScalarFieldTasks()` plus all the boilerplate task
  bodies (`InitRecv`, `CopyU`, `SendU`/`RecvU`, `RestrictU`, `Prolongate`,
  `ApplyPhysicalBCs`, `ClearSend`/`ClearRecv`), mirroring the corresponding `Z4c::*`
  functions in `z4c_tasks.cpp`.

Edits to existing files (all additive/gated, no behavior change when `<scalarfield>` is
absent from the input file):
- `src/tasklist/numerical_relativity.hpp/.cpp`: new `TaskName` values (`SF_Recv` ...
  `SF_NTASKS`) appended *after* `Z4c_NTASKS`, a new `Phys_ScalarField` dependency class,
  and the corresponding `NeedsPhysics`/`DependencyAvailable` dispatch arms. This was a
  **required correctness fix**, not polish: `NeedsPhysics` classifies task names purely by
  their numeric position in the enum, so any new task name placed after `Z4c_NTASKS`
  without also extending this dispatch chain would silently be classified `Phys_None`
  ("always available"), which is wrong and can hang task-list assembly if such a task is
  ever used as an optional dependency by another module.
- `src/mesh/meshblock_pack.hpp/.cpp`: `pscalarfield` pointer, destructor entry (placed
  between `ptmunu` and `padm` deletion, respecting the existing
  roughly-reverse-of-construction order), and a new gated construction block in
  `AddPhysics()` right after the Z4c/ADM block. Hard-errors if `<scalarfield>` is present
  without `<z4c>` (a static `<adm>`-only background isn't enough for a dynamical scalar
  sector).
- `src/CMakeLists.txt`: new unconditional source group for `scalar_field/*.cpp`, same
  style as the `z4c/*.cpp` group.
- `src/parameter_input.cpp` (`ParameterInput::CheckBlockNames`, ~line 102): added
  `"scalarfield"` to the `valid_name` allow-list of recognized `<input_block>` names.
  Easy to miss -- without this, any input file with a `<scalarfield>` block hard-aborts
  at startup with "Invalid <block_name> in input file" regardless of everything else
  being wired correctly. Found this the first time an actual `<scalarfield>` block was
  tried at runtime (smoke test, see below), not from reading the Z4c/ADM/Tmunu code
  paths, since none of those needed a new block name.

## Design decisions made during Phase 0 (revisit if they turn out wrong)

- **Boundary-buffer treatment (superseded during Phase 1, re-reverted in Phase 5, see
  below)**: originally `ScalarField`'s own `MeshBoundaryValuesCC` was constructed with
  `is_z4c=true`, i.e. it opted into the same 4th-order-safe prolongation/restriction
  path Z4c's own fields use. During Phase 1 debugging this was changed to
  `is_z4c=false` (the plain path Hydro/MHD use) while chasing the
  `InitBoundaryValuesAndPrimitives` bug below -- it turned out not to be the cause, but
  `false` was kept anyway as the more standard/correct choice for a generic 2-variable
  field (see an Explore agent's analysis at the time), with an explicit note to
  "revisit if `sphi` differentiability at refinement boundaries ever becomes suspect
  once AMR is actually exercised."

  **That revisit happened in Phase 5**: the first `<scalarfield>` run to actually use
  `<mesh_refinement>/refinement=static` (`rns_st_gam2k100_test_v3`) prompted reverting
  all 3 call sites back to `is_z4c=true` (`scalar_field.cpp`'s constructor, and the
  matching `RestrictCC`/`ProlongateCC` calls in `scalar_field_tasks.cpp`). Traced
  through end-to-end to confirm this is correct and complete, not just plausible:
  `is_z4c=true` (a) makes the boundary-buffer pack/unpack also exchange restricted/
  coarse data between *same-level* neighbors (`bvals_cc.cpp`, sized in
  `bvals.hpp::AllocateBuffers`) so a high-order stencil always has valid coarse-side
  data even at same-level corners, (b) switches `RestrictCC`
  (`mesh_refinement.cpp:1159`) from a plain unweighted 8-point average to the same
  weighted `RestrictInterpolation<NGHOST>` operator Z4c uses, and (c) switches
  `ProlongateCC` from 2nd-order min-mod-*limited* prolongation (built for Hydro/MHD
  conserved variables that can have shocks) to unlimited 4th-order polynomial
  prolongation, matching Z4c. All 3 sites were changed together and consistently, and
  the `isame_z4c`/`isame_z4c_ndat` buffer-index metadata is computed unconditionally
  for every module's boundary-value object regardless of the flag
  (`buffs_cc.cpp:48-59`), so no other wiring is missing. Physically justified: the
  scalar's own RHS (`scalar_field_calcrhs.cpp`) computes `D_iD_j(sphi)` via finite
  differences exactly the way `z4c_calcrhs.cpp` builds `Ddalpha_dd` -- a min-mod
  limiter at a refinement boundary would clip `sphi`'s smoothness there and degrade
  that FD derivative right at the block interface, the same reasoning Z4c itself
  relies on.

  Two things noticed while tracing this, neither caused by nor requiring a change to
  the flag flip above, flagged for whoever hits them next:
  - `HighOrderProlongCC`/`RestrictInterpolation`'s dispatch is
    `switch(indcs.ng) { case 2: ...; case 4: ...; }` with **no default case** -- an
    `nghost=3` run (e.g. the Phase-1 `scalar_field_linear_wave` convergence test's own
    convention) combined with `is_z4c=true` would silently skip prolongation/
    restriction for that cell, leaving it stale, with no warning. Not active for any
    current run (all use `nghost=4`), but would silently corrupt data in a future
    `nghost=3` AMR scalar-field test.
  - `MeshRefinement::AdaptiveMeshRefinement`'s per-physics `RefineCC`/
    `DerefineCCSameRank` dispatch list (`mesh_refinement.cpp:508-600`, used to
    transfer field data to/from newly created/destroyed MeshBlocks) includes
    Hydro/MHD/Radiation/Z4c but never `pscalarfield` -- structurally the same class of
    bug as the Phase-1 "`ScalarField` never registered in
    `Driver::InitBoundaryValuesAndPrimitives`" fix above. Confirmed dormant right now:
    this function only runs when `pmesh->adaptive` is true (`mesh.cpp:175`, i.e.
    `<mesh_refinement>/refinement=adaptive`), and every `<scalarfield>` test so far
    (including the static-AMR one that prompted this section) uses
    `refinement=static` (`adaptive=false`), so it's never called. Will need fixing
    before this module is exercised under genuine dynamic (`adaptive`) AMR, or
    `sphi`/`Pi` will simply not exist on any MeshBlock created/destroyed by a real
    regridding event.
- **`ApplyPhysicalBCs` is a no-op (superseded in Phase 6, see below)**: beyond the
  generic user-BC hook every physics module calls, there was no `ScalarFieldBCs`
  formula -- ghost zones at the edge of the whole domain held whatever the problem
  generator set them to, forever, regardless of BC type. This was originally assumed
  harmless for Phase 0/1's periodic/reflecting test domains, and conflated with the
  Yukawa-falloff Sommerfeld condition added in Phase 4 (`SF_SomBC`) -- but that task is
  an RHS/source-term correction, not a ghost-zone fill, and does not substitute for
  one. **This turned out to matter in practice**: every `<scalarfield>` run through
  Phase 5 (RNS-ST, `reflect`/`diode` domain-edge BCs) silently left `sphi`/`Pi` ghost
  zones frozen at their `t=0` values for the whole run. Fixed in Phase 6 with a
  dedicated `ScalarFieldBCs`, mirroring `Z4cBCs`.
- **No `SF_Newdt` task**: the scalar field's own characteristic speed is the speed of
  light, already reflected in Z4c's existing CFL-based timestep. (Originally this bullet
  anticipated Phase 4's mass term needing a separate stiff-source timestep/IMEX
  treatment; that plan changed -- see "What Phase 4 actually did" below -- the mass term
  is explicit and needs no separate timestep constraint either.)
- **No `z4c_tasks.cpp` dependency wiring yet (partly superseded, see below)**:
  `Z4c_CalcRHS`'s optional-dependency list was untouched at Phase 0. What actually
  happened later was *not* exactly what this note predicted: `SF_RescaleT` was indeed
  added to `Z4c_CalcRHS`'s optional deps (Phase 3, matches the prediction), but
  `SF_CalcRHS` was **not** -- it was added to `Z4c_ExplRK`'s optional deps instead
  (Phase 2), because the actual hazard that needed fixing was a cross-module RK race
  (see "What Phase 2 actually did" below), not "`Z4c_CalcRHS` reading stale scalar-field
  RHS data" as this note assumed. `Z4c_CalcRHS` reads `sf.sphi`/`sf.vpi` directly (the
  scalar's *state*, not its RHS), so it never actually needed a dependency on
  `SF_CalcRHS` at all.

## What Phase 1 actually did

Implemented the scalar's own massless Klein-Gordon-like RHS in `scalar_field_calcrhs.cpp`
(still **no** back-reaction on geometry -- `z4c_calcrhs.cpp` remains untouched):
```
dt(sphi) = -alpha * Pi                                    (+ shift advection)
dt(Pi)   = alpha*(K+2*Theta)*Pi - alpha*D2(sphi) - g^ij di(alpha) dj(sphi)
           - alpha*sphi*(|D(sphi)|^2 - Pi^2)
```
(the matter-trace and mass-term pieces come in Phase 3/4). Derivatives use the
conformal-variable convention established in `z4c_calcrhs.cpp` (differentiate
`pz4c->z4c.g_dd`/`chi` plus `oopsi4`, not `padm->adm.g_dd` directly) -- the scalar's
physical-metric covariant Hessian `D_iD_j(sphi)` is built exactly the way
`z4c_calcrhs.cpp` builds the lapse's `Ddalpha_dd` (conformal-covariant Hessian plus the
two chi-derivative correction terms). Added K-O dissipation for the scalar sector's own
channels, mirroring `z4c_calcrhs.cpp`'s dissipation loop.

Added a new test pgen, `src/pgen/tests/scalar_field_linear_wave.cpp` (`pgen_name =
scalar_field_linear_wave`), modeled on `z4c_linear_wave.cpp`: a massless scalar plane wave
on a fixed flat (Minkowski) Z4c background (an exact static vacuum-Z4c solution, so it's a
clean decoupling-limit test of the scalar sector alone). Sample input at
`inputs/scalar_field/scalar_field_linear_wave.athinput`.

**Validated by convergence, not just by running**: RMS-L1 error at `nx=16/32/64`
(diagonal wave, `kx1=kx2=kx3=1`) is `1.241e-10 / 5.414e-12 / 4.665e-13`, giving measured
convergence orders **4.52** (16->32) and **3.54** (32->64) -- consistent with the nominal
4th-order (`nghost=3`) finite-difference + RK4 scheme. Confirms `sphi` propagates at the
correct (light) speed and `Pi = -dt(sphi)/alpha` self-consistently.

### A real bug found and fixed along the way

The first version of this test showed an error that did **not** converge with resolution
(flat/slowly growing ~2-3e-9 regardless of grid spacing) despite the RHS math being
correct. Root cause, found by process of elimination (ruled out: a `dt(sphi)=+alpha*Pi`
sign error, found and fixed first; K-O dissipation; the `is_z4c` boundary-buffer flag;
`rk4`-specific `CopyU` bookkeeping via an `rk2` comparison; Dxy/off-diagonal terms via a
pure-1D-wave test; a genuine growing instability via a 5-period run) --

**`ScalarField` was never registered in `Driver::InitBoundaryValuesAndPrimitives`**
(`src/driver/driver.cpp`). This function is a hand-written, per-module opt-in list (one
hardcoded `if (pmodule != nullptr) {...}` block per physics module -- Z4c, Hydro, MHD,
Radiation) that runs **once**, after the problem generator sets initial data but before
the main evolution loop starts, to populate every evolved field's ghost zones (periodic
wrap, prolongation, etc.) for the very first `CalcRHS` call. There is no generic
"fill ghost zones for whatever fields exist" mechanism -- it is not automatic, and Z4c's
own registration doesn't cover modules added later. Missing this meant the very first
`CalcRHS` of the very first stage read stale/uninitialized ghost-zone data near the
domain edges; that contamination then persisted (bounded, non-growing, resolution-
independent) for the rest of the run, exactly matching the observed symptom. Fixed by
adding a `pscalarfield` block in `driver.cpp` mirroring the existing Z4c block (see the
diff in this commit). **Any future new evolved-field module must add its own block
here, or it will silently carry the same bug.**

## What Phase 2 actually did

Enabled the scalar's back-reaction on the geometry in `z4c/z4c_calcrhs.cpp` (still
vacuum/massless -- the matter-trace and mass-term pieces are Phase 3/4), guarded
end-to-end by `bool has_scalar = (pmy_pack->pscalarfield != nullptr)`, mirroring the
existing `is_vacuum`/matter-term pattern exactly. Five RHS spots gained an extra
guarded term: the lapse gauge (`-3*alpha*sphi*Pi`), the Theta/Hamiltonian-like
combination, the K/Khat RHS, the (already-trace-free) Aij RHS, and the Gamma-tilde^i
RHS. All five were derived by reading `bssn_st.f90` directly (not by trusting the
condensed equations in `PLAN.md`'s "Physics being ported" section, which were a lossy
compression written before implementation) -- **and this caught a real error in that
plan**: the plan's shorthand for the Gamma-tilde^i term was
`-8*pi*gamma^ij*momsca_j` (an assumed matter-source-like coefficient/metric
convention); the source shows it's actually `-2*alpha*g_tilde^ij*momsca_j`, using the
*conformal* inverse metric with coefficient `-2`, structured identically to how the
pre-existing matter term in that same RHS line is written. Implemented the corrected
version; see `PLAN.md`'s "Critical files" section for the exact statement.

A second, non-obvious point requiring care: SACRA's own covariant-Hessian-like
quantities (e.g. `sphi_cdxx`) already have the `oopsi4` conformal factor baked in,
whereas this module's `Ddsphi_dd` (Phase 1, and reused here) does not -- it's built
exactly like Z4c's own `Ddalpha_dd`, which is *not* itself the physical Hessian, only
becomes physical when contracted with `oopsi4*g_uu`. Missing this distinction would
have silently dropped a factor of `oopsi4` from the Aij term specifically (the term
that uses `Ddsphi_dd` as a full tensor, not just its trace). Confirmed correct by
checking that the added Aij tensor is exactly physical-trace-free once `strtr_sf` is
added back via `g_dd(a,b)`, i.e. that the whole addition doesn't corrupt Aij's
trace-free invariant (verified algebraically, not just numerically).

**Task-list wiring: a second, distinct cross-module race** (beyond Phase 1's
`InitBoundaryValuesAndPrimitives` gotcha, which was about the *first* RHS evaluation
only). Once both Z4c and ScalarField read *each other's* raw evolved state every
stage (not just once at t=0), a new hazard appears that didn't exist while the scalar
was non-back-reacting: `Z4c_ExplRK` (which overwrites Z4c's `u0`) could in principle
run before `SF_CalcRHS` has read that stage's `z4c.g_dd/alpha/...`, and symmetrically
`SF_ExplRK` (which overwrites the scalar's `u0`) could run before `Z4c_CalcRHS` has
read `sf.sphi/sf.vpi`. This is *not* the same situation as `Z4c_CalcRHS`'s existing
`{MHD_SetTmunu}` dependency, because `Tmunu` is recomputed fresh every stage from
already-boundary-filled primitives (not itself RK-evolved), so there's no "future
value" it could race against; both Z4c and the scalar sector *are* independently
RK-evolved, so this real hazard needed an explicit fix: `SF_ExplRK` now has an
optional dependency on `Z4c_CalcRHS`, and `Z4c_ExplRK` now has an optional dependency
on `SF_CalcRHS` (both edges use the existing `AddExtraDependencies` optional-dependency
mechanism, so they're silently dropped when `<scalarfield>` is absent). These two
edges plus each sector's own internal chain form a valid DAG (a diamond: both
`CalcRHS`s are independent sources; each `ExplRK` depends on the *other* sector's
`CalcRHS` plus its own chain) -- no deadlock. **Any future pair of mutually-coupled,
independently-RK-evolved sectors in AthenaK needs this same kind of pairwise
`ExplRK`-depends-on-the-other's-`CalcRHS` edge**, not just a one-directional
dependency like the Tmunu case.

### A debugging detour: a self-inflicted test-harness bug, not a code bug

While validating Phase 2 with a real (non-flat) Z4c background (the existing
`z4c_linear_wave` gravitational-wave test, `inputs/tests/linear_wave_z4c.athinput`,
with a `<scalarfield>` block added but the scalar initialized to exactly zero
everywhere -- an even stronger regression check than Phase 0/1's flat-background
tests, since it exercises the `has_scalar` guards against genuinely time-varying
`z4c.g_dd/alpha/chi/...`), the very first attempt showed the Z4c evolution
**diverging from the no-`<scalarfield>` case starting at the first RK substage**,
by a few parts in 10^4 -- despite the scalar field being provably, bit-exactly zero
everywhere at every stage (every added term is proportional to `sphi` and/or `Pi`; this
was confirmed directly with an instrumented build that flagged any nonzero occurrence
of any added term across the whole domain -- there were none). Chased this for a long
time through several wrong hypotheses -- the new Z4c_ExplRK/SF_ExplRK task-race edges
above (reverting them didn't change anything), a shared/aliased boundary-communication
buffer between Z4c's and ScalarField's separate `MeshBoundaryValuesCC` instances
(constructing a second, unused `MeshBoundaryValuesCC` in isolation didn't reproduce it
either), and generic memory-layout/allocation-order sensitivity (a same-sized dummy
array allocation also didn't reproduce it) -- before finding the actual cause: **the
quick-and-dirty test input file was hand-edited with `sed` to insert the
`<scalarfield>` block immediately after the `<z4c>` header line, which landed it
*before* `<z4c>`'s own `diss = 1` parameter line**. AthenaK's input-block parser
assigns a `key = value` line to whichever block header it most recently saw, so that
`diss = 1` silently became `<scalarfield>`'s `diss` instead of `<z4c>`'s -- leaving
Z4c's own Kreiss-Oliger dissipation coefficient different between the two test
configurations, which is what actually diverged. Rebuilding the same comparison with
the `<scalarfield>` block appended safely at the end of the file (or anywhere not
immediately following another block's own header) gives byte-identical output, as
expected. **Lesson for future test-input construction (not a code issue): never
insert a new block immediately after another block's header via blind text
insertion -- append new blocks at the end of the file, or insert immediately before
a block header, never between one and its first parameter line.**

Also ran a second check with a genuinely nonzero coupling (the `scalar_field_linear_wave`
pgen's flat background, `amp = 1e-3` instead of Phase 1's `1e-8`, `nx=16`, 20 cycles):
ran stably (no NaN/blowup), and `chi`/`alpha` departed from their flat initial value of 1
by ~1e-6 -- consistent with the expected leading order `O(amp^2)` scaling of the
back-reaction (every added term is quadratic in `sphi`/`Pi`), a sensible-magnitude check
that the new terms are firing with roughly the right size, not a sign/coefficient error
that would show up as O(amp) or blow up outright.

## Known gap: `Z4c::ADMConstraints` is not yet scalar-aware

`Z4c::ADMConstraints` (`z4c_adm.cpp`) computes the Hamiltonian/momentum constraint
monitors purely from `padm` (physical-frame ADM data) plus the fluid's `Tmunu` -- it has
no scalar-field awareness at all. With back-reaction now enabled, this diagnostic will
under-report true constraint violation whenever the scalar carries nonzero
stress-energy, since it's missing the scalar's own Hamiltonian/momentum source terms
(the same physics as the `z4c_calcrhs.cpp` additions above, but re-derived in the
physical-frame convention `z4c_adm.cpp` already uses -- actually *simpler* there, since
no chi/conformal-correction terms are needed, everything is already physical-frame).
This should be added before relying on `Z4c::ADMConstraints` for rigorous
constraint-convergence testing in Phase 2/3; deliberately deferred here to keep this
pass's scope bounded to the RHS physics itself. See `PLAN.md` for the same note.

## What Phase 3 actually did

Coupled the scalar sector to the `dyn_grmhd` fluid, still massless (`mass2=0`, Phase 4).
Two pieces:

**`ScalarField::RescaleTmunu`** (new task `SF_RescaleT`, in `scalar_field_tasks.cpp`):
multiplies the fluid's `Tmunu` (`E`, `S_d`, `S_dd`, all set fresh each stage by
`DynGRMHD::SetTmunu` from the MHD primitives, in the Jordan frame) by `1/A(sphi) =
exp(-0.5*sphi^2)` in place, matching SACRA's `tabfac = 1/A(sphi)` factor -- confirmed by
reading how SACRA computes `tnn`/`txx` (`bssn_st.f90:1083-1085`: `tnn = hd_tmp(...)*wa_p3
*tabfac`, i.e. the *same* rescale factor SACRA applies to every fluid stress-energy
component before it's used anywhere). No-op when no fluid module exists (task returns
immediately), and mathematically a no-op when `sphi==0` everywhere (`A(0)=1`). Runs after
`MHD_SetTmunu` and before anything that reads `Tmunu` as a geometry-equation source --
both `Z4c_CalcRHS` (which already had a pre-existing, is-vacuum-gated matter term for
ordinary GR) and the scalar's own `CalcRHS`, both extended via the optional-dependency
mechanism as usual.

**Matter-trace `T` terms**, added in exactly two places (verified by reading
`bssn_st.f90` for every appearance of `ttrace`; it does *not* appear in the Theta/Ricci or
Gamma-tilde^i RHS lines, contrary to what might be guessed from "matter couples
everywhere"):
- the scalar's own Pi-RHS (`scalar_field_calcrhs.cpp`): `+ 2*pi*alpha*omega_c*T*sphi`
- the K/Khat RHS (`z4c_calcrhs.cpp`, `dk_sf`): `- 3*pi*omega_c*sphi^2*T`

where `T = -E + gamma^ij*S_ij` (`bssn_st.f90`'s `ttrace = -tnn + wa_p2*g^ij*t_ij`) is the
fluid's full 4D stress-energy trace, computed from the *already-rescaled* (Einstein-frame)
`Tmunu`. In `z4c_calcrhs.cpp` this reuses the `S` variable already computed by the
pre-existing (pre-Phase-2) matter-term code (`S = oopsi4*g_uu(a,b)*tmunu.S_dd(a,b)`), so
no new tensor contraction was needed there, only `T = -tmunu.E + S`. In
`scalar_field_calcrhs.cpp` the same trace has to be computed fresh (that file didn't
previously read `Tmunu` at all), using the `g_uu`/`oopsi4` already in scope from the
scalar's own derivative computations.

### Validation

Since `dyngr_tov.cpp` (the only in-repo star-initial-data pgen) has no scalar-field
awareness, `sphi` stays at exactly `0.0` (Kokkos default-zero-init) for any TOV-star run
regardless of `<scalarfield>` block content — this makes the TOV star an excellent
regression test (real, nonzero fluid stress-energy exercising `RescaleTmunu` and both new
`T`-terms for the first time, unlike Phase 2's vacuum-only tests) but *not* a test of a
genuinely nonzero-coupling scenario (see the open gap below).

Built a dynamical-Z4c TOV star input (`<z4c>` instead of the existing
`inputs/dyn_grmhd/whisky_tov.athinput`'s static `<adm>` background; `dyngr_tov.cpp`
already supports this via its `pmbp->pz4c != nullptr` -> `ADMToZ4c` conversion path) and
ran it two ways, with and without a (content-identical, safely-appended-at-file-end this
time -- see Phase 2's `sed`-pitfall note) `<scalarfield>` block:
- Serial, single-MeshBlock, `nx=32`, 20 cycles: byte-identical.
- MPI, production resolution (`nx=64`, `meshblock=32` -> 2x2x2=8 MeshBlocks, 8 ranks
  across 4 nodes), 30 cycles: byte-identical across all 32 `.tab` dumps and all 3 `.hst`
  history files (submitted via `/sakura/ptmp/tlam/athenak_run/test_stt`, following this
  account's existing MPCDF module conventions: `intel/2025.3 + impi/2021.17 + gcc/13 +
  gsl/2.4`, matching `~/athenak_cfc`'s established build). **Correction found in Phase 4**:
  the diff script used here (`diff -rq dirA dirB`) compared directories whose files are
  named with different `basename`s (`tov_z4c_mpi_noscalar.*` vs `tov_z4c_mpi_withscalar.*`)
  -- `diff -rq` never matches mismatched filenames, so it printed a full "Only in ..."
  listing for *every* file regardless of content and never actually performed a byte
  comparison. The "byte-identical" conclusion above happened to still be true (reverified
  in Phase 4 with a corrected script that matches files by their common cycle-number
  suffix, see below), but this specific run's log did not actually prove it at the time.
  **Lesson: when comparing two runs with different `basename`s, diff matching pairs of
  files explicitly (e.g. by stripping/renaming the differing prefix) -- never
  `diff -rq` two directories whose filenames aren't identical.** Confirms the new task
  (`SF_RescaleT`) and its dependency wiring behave correctly under real MPI domain
  decomposition and ghost-zone exchange too, not just the single-block case Phase 0-2
  were checked under.

### Open gap: no self-consistent scalarized initial data (not resolved this pass)

`dyngr_tov.cpp` was **not** extended with a coupled scalarized-TOV solve. This means
there is currently no way to actually exercise the new `T`-matter-coupling terms with
`sphi` genuinely nonzero and constraint-satisfying -- only the "no-op when zero" side has
been validated. Building this requires a shooting-method radial ODE integration for the
coupled TOV+scalar stellar-structure equations (extending `src/utils/tov/tov.cpp`'s
existing pure-GR solve) and a decision on the initial-data pipeline (in-repo solver vs.
external, see `PLAN.md`'s "Open decision" note) -- a substantial, independent
numerical-methods task, deliberately not attempted in this pass to avoid rushing it.
**Do not claim Phase 3 is physics-validated against SACRA's scalarized-star results until
this is built** -- what's validated so far is "the new code doesn't break existing GRMHD
physics," not "the new code correctly reproduces scalarized stars."

## What Phase 4 actually did

Added the mass term, fully **explicitly** -- no Newton solve, no IMEX, no new
`impl_src`-style array, contrary to this module's original plan (see PLAN.md's
"Mass-term treatment" section for the full reasoning): AthenaK uses one global timestep
across all refinement levels, set by the finest level's CFL condition -- unlike SACRA's
per-level-subcycled timestepping, which is what actually forces SACRA's implicit
treatment. At AthenaK's timestep this makes the mass term no stiffer than every other
explicit RHS term, so it's just added alongside them. Three pieces:

- **Mass term in the scalar's own RHS** (`scalar_field_calcrhs.cpp`'s Pi-RHS):
  `+ mass2*alpha*sphi*A(sphi)`, `A(sphi) = exp(0.5*sphi^2)` (beta0=1 hardcoded, matching
  every other Phase 2/3 coefficient's implicit beta0=1).
- **Mass term in the Z4c-side RHS** (`z4c_calcrhs.cpp`): an extra
  `-2*(mass2/omega_c)*sphi^2*A(sphi)` piece in the Theta/Hamiltonian-like combination
  (`dtheta_sf`), and an extra `-1.5*mass2*sphi^2*A(sphi) - (mass2/omega_c)*sphi^2*A(sphi)`
  piece in the K/Khat RHS (`dk_sf`) -- both exactly as given in PLAN.md's physics
  equations, both identically zero when `mass2=0` (regression-safe by construction: adding
  `0.0` to an existing sum cannot change any bit of the result).
- **Yukawa Sommerfeld outer BC** (new `scalar_field_Sbc.cpp`, new task `SF_SomBC`,
  queued between `SF_CalcRHS` and `SF_ExplRK` exactly mirroring `Z4c_SomBC`'s placement
  between `Z4c_CalcRHS` and `Z4c_ExplRK`): overwrites `rhs.sphi`/`rhs.vpi` at
  outflow/diode/vacuum/user-flagged faces with
  `rhs = -(f-f_inf)/r - m*(f-f_inf) - s^i*di(f-f_inf)`, `m = sqrt(mass2)`, `f_inf = sphi0`
  for `sphi` and `f_inf = 0` for `Pi` (a static/quasi-static configuration has
  `dt(sphi)->0` at infinity). Reduces exactly to `Z4c`'s own massless Sommerfeld form
  when `mass2=0` (`f_inf` is a constant, so `di(f_inf)=0` always). Cross-checked against
  `~/SACRA_2D/SACRA_MPI/boundary.f90`'s own outer-boundary treatment: it extrapolates
  toward `asym` (`sphi00` for sphi, `0` for Pi) with exponential rate `exp_drop = pmass`
  (i.e. `m`, not `m^2`) for *both* components -- the same `(f_inf, m)` pairing used here,
  even though SACRA implements it via buffer-zone extrapolation rather than an
  RHS-replacement ODE (AthenaK's own existing `Z4c_SomBC` convention, used here for
  consistency with this codebase rather than copying SACRA's mechanism literally).
  Added a new `Options::user_Sbc` flag (mirrors `z4c::Options::user_Sbc`) to let this BC
  also apply on `user`-flagged faces, same convention as Z4c.
- Removed the now-unused `Options::newton_tol`/`newton_maxiter` fields (added in Phase 0
  in anticipation of a Newton solve that never got built) rather than leaving them as
  dead/misleading options.

### Validation

- **Compiles clean**: both a custom-pgen MPI build (`-DPROBLEM=dyn_grmhd/dyngr_tov`) and
  the default `built_in_pgens` serial build (which is what `scalar_field_linear_wave`
  actually needs -- see the pitfall below) succeed with no warnings from the new code.
- **`mass2=0` regression, quantitative**: reran Phase 1's `scalar_field_linear_wave` test
  (`nx=64`, same input file, unchanged) against the Phase 4 code. RMS-L1 error:
  `4.665155e-13` -- an *exact* match, digit-for-digit, to the RMS-L1 value recorded when
  Phase 1 was first validated. Proves the new (structurally zero) mass-term code paths
  and the new `SF_SomBC` task (a no-op here: this test uses periodic BCs, so it never
  engages) perturb nothing.
- **TOV MPI byte-identical regression, corrected methodology**: reran the Phase 3 TOV
  `noscalar`/`withscalar` (`mass2=0`) comparison against the Phase 4 code, this time with
  a per-file diff that matches files by their common cycle-number suffix instead of
  `diff -rq` (see the Phase 3 "Correction found in Phase 4" note above for why the
  original script couldn't have actually caught a regression). Genuinely byte-identical:
  all 32 `.tab` dumps, all 3 `.hst` files. This TOV input uses `diode` outer BCs, so
  `SF_SomBC` *does* engage here (unlike the linear-wave test) -- and still produces
  exactly zero RHS contribution, because `sphi`/`Pi` are bit-exactly zero everywhere in
  this run (no scalarized initial data yet, see the open gap above), so
  `f - f_inf = 0 - 0 = 0` at every boundary point.
- **Massive-field stability sanity check** (not a quantitative Yukawa-falloff
  validation): `scalar_field_linear_wave`'s flat background with `mass2=100` (vs. the
  decoupling-limit test's `mass2=0`) runs stably to the same `tlim`, no NaN/blowup. Its
  built-in error diagnostic reports a larger RMS-L1 (`1.35e-7` vs `4.67e-13`) -- expected
  and *not* a bug: that diagnostic compares against the *massless* plane-wave analytic
  solution, which is the wrong dispersion relation once `mass2>0`. A real quantitative
  check of the Yukawa `exp(-m*r)/r` falloff needs an actual asymptotically-flat,
  spherically-symmetric massive-scalar-star setup -- not available until scalarized
  initial data exists (see the open gap above), so this remains **not yet done**.
- **Build pitfall found while testing**: `scalar_field_linear_wave` is a *built-in* pgen
  (dispatched at runtime by its `pgen_name` string in `pgen.cpp`, and already
  unconditionally compiled via `src/CMakeLists.txt`'s source list) -- it does **not**
  define `ProblemGenerator::UserProblem`. Building it with a custom
  `-DPROBLEM=tests/scalar_field_linear_wave` flag (as opposed to the default
  `PROBLEM=built_in_pgens`) sets `USER_PROBLEM_ENABLED`, which makes `pgen.cpp` call a
  `UserProblem()` this file never defines -- a link error
  (`undefined reference to ProblemGenerator::UserProblem`), not a code bug. The custom
  `-DPROBLEM=` flag is only for standalone pgens that *do* define `UserProblem` (like
  `dyngr_tov.cpp`). Build `scalar_field_linear_wave` (and any other `pgen/tests/*.cpp`
  pgen dispatched by `pgen_name`) with the default `PROBLEM=built_in_pgens` instead.

## Next step (as of Phase 4 -- superseded in part by Phase 5/6, see below)

Phase 4's code (explicit mass term, Yukawa Sommerfeld BC) is done and structurally
verified (regression + stability), but -- same as Phase 3 -- not yet *physics*-validated
against SACRA, because that requires a genuinely nonzero, constraint-satisfying `sphi`,
which still doesn't exist in this repo. The remaining open items, in the order they
naturally unblock each other:
1. ~~Build the scalarized-TOV shooting solve in-repo (extend `src/utils/tov/tov.cpp` +
   `dyngr_tov.cpp`)~~ -- **this path was not taken.** Phase 5 instead ported an
   external RNS-ST equilibrium solver's own reader/interpolator into a new pgen
   (`dyngr_rns_st.cpp`), which already produces genuinely nonzero, constraint-
   satisfying `sphi` initial data (validated against two real stars) -- see Phase 5
   below for what that closed and what it left open (an unexplained MPA1 instability).
2. Verify the `exp(-m*r)/r` far-field falloff over several e-foldings against SACRA's
   `pmass2>0` single-star output using the now-available RNS-ST data (both tested
   stars already have `mass2>0`) -- this is Phase 4's still-missing quantitative
   verification step (the mass-term *code* is done and stable, but this specific
   physics check has not been performed). **Still open.**
3. Close the `Z4c::ADMConstraints` scalar-awareness gap above before relying on
   constraint convergence as a verification method for either of the above. **Still
   open.**

## Phase 5 -- RNS-ST scalarized-star initial data (new pgen, closes the open gap above)

The scalarized-star initial-data gap flagged throughout Phase 3/4 (no genuinely
nonzero, constraint-satisfying `sphi` in this repo) is now closed via a new problem
generator, `src/pgen/dyn_grmhd/dyngr_rns_st.cpp` + `rns_st_reader.hpp`, rather than an
in-repo shooting solve: the user already has an external RNS-ST equilibrium solver
(legacy Fortran, `~/SACRA_2D/SACRA_MPI/read_grass_st.f90` +
`input_rns_st`/`input.f90:1214-1297`) that produces exactly this data. The new pgen is
a direct C++ port of that Fortran reader/interpolator (7-point Lagrange scheme on an
axisymmetric `(s,cos-theta)` grid), not a new independent derivation -- see the plan
file this was built from (`/u/tlam/.claude/plans/zippy-foraging-scroll.md` at time of
writing) for the full port rationale and the conformal-to-physical-ADM /
covariant-to-contravariant-velocity conversions it performs on top of the literal port.

**Reader validated against real data**: standalone (no-Kokkos) test against
`/sakura/ptmp/tlam/initial/rns_new/MPA1_J0.00_..._rhoc9.356E+14_sphim1.714.dat`
reproduces the filename's own central density (`9.356e14` after the code's
`rho_uni` conversion) and central `sphi` (`1.711` vs. filename's `1.714`) to good
precision, and correctly recovers exact spherical symmetry (equator/pole identical,
zero shift, zero extrinsic curvature) for this `J=0` (non-rotating) model. A real
format bug was found and fixed here: the ID file's header line has extra trailing
padding values beyond `nr,np,nfac,R_ref` that Fortran's record-based list-directed
read silently discards but a naive C++ `ifstream >>` tokenizer would not -- fixed by
parsing the header line in isolation via `getline` before switching to token-based
reads for the data section.

**Build/run validated, evolution NOT yet stable at the resolution tried**: built
cleanly (`build_rns_st_sakura.sh`, `-DPROBLEM=dyn_grmhd/dyngr_rns_st`) and runs
end-to-end without crashing on the MPA1 ID file above (`omega_c=153`, `mass2=0.01`,
piecewise-polytrope EOS parameters derived from `/sakura/ptmp/tlam/EOS_2D/MPA1_v2.d`,
see `inputs/dyn_grmhd/rns_st_mpa1.athinput`). One real bug found and fixed along the
way: `pmhd->b0`/`bcc0` (this star is unmagnetized, so the pgen never touches them)
must be explicitly zeroed -- AthenaK does not zero-initialize device memory by
default, and leaving them untouched produced immediate `NANS_IN_CONS` (`Bx=nan`)
errors from the very first primitive solve. After that fix, the star holds up for
about 13-14 cycles (`t~8` in code units, central density/lapse/`sphi` all only
drifting slowly) before the lapse goes negative and the primitive solver starts
failing at ghost cells (`dyn_error=reset_floor` recovers gracefully rather than
crashing, but by `t~11` the star has been floored away entirely). Tested whether this
was a reflecting-boundary-condition artifact (this is the first-ever test of the
scalar-tensor module with a genuinely nonzero, spatially-varying `sphi` through a
reflect BC -- every earlier Phase 0-4 check used `sphi=0` identically, where a BC
parity bug would be invisible) by rerunning on a full (non-octant, diode-everywhere)
domain: it also fails, at cycle 0 rather than cycle 13, but at a different (coarser,
confounded) resolution, so this did NOT cleanly rule reflect-BC-at-the-corner in or
out.

**Resolution-convergence check done -- under-resolution hypothesis REFUTED.** Reran
at 2x resolution (`inputs/dyn_grmhd/rns_st_mpa1_hires.athinput`: `nx1=nx2=nx3=128`,
`dx=0.8` vs. `1.6`, ~16 cells across the star instead of ~8; same domain/BC/gauge/EOS,
resolution the only change, `nlim=100` to run well past the low-res failure point).
Result is the OPPOSITE of what under-resolution would predict: the instability
triggers **sooner** (`alpha-min` starts dropping steeply at `t~4.8` vs. `t~7.7`) and
is far more violent (`rho-max` overshoots to `2.47` -- ~1600x the initial central
density -- and `alpha-min` reaches `-1.6e15` at `t~6.1`, vs. the low-res run's much
milder `alpha-min~-5.7` before flooring). Both runs eventually floor the star away
entirely and settle into a smooth residual decay. A genuine numerical instability that
*worsens* under refinement is not consistent with plain under-resolution -- that
pattern instead suggests either (a) a real physical instability of this specific
star (this `MPA1`/`omega_c=153`/`mass2=0.01`/`rhoc~9.36e14` configuration may
genuinely sit past a stability threshold in this scalar-tensor theory -- e.g.
spontaneous-(de)scalarization-adjacent parameter territory, which the user is far
better positioned to judge than a numerical experiment), or (b) a resolution-amplified
coupling/gauge issue in the new back-reaction terms, possibly related to `omega_c=153`
being far outside every previously-tested config (`omega_c=12` default) combined with
a nonzero mass term -- similar in spirit to (but a structurally different mechanism
from) the already-documented Stage 8 `SrcThin` stiffness finding in the sibling M1
project. **Next step for whoever picks this up**: before touching any code, check
whether this star is expected to be stable at these parameters (e.g. against a known
stability curve/turning-point criterion for this theory) -- only chase a numerical
root cause if the star is expected to be stable and isn't.

**Second, independent star tested -- stable.** Ran the same pgen/pipeline (no code
changes; `dyngr_rns_st.cpp`'s EOS dispatch already had a `tov::PolytropeEOS` branch for
`dyn_eos=ideal`) against a structurally different star:
`Gam2_K100_J.00_..._B1.20E+01_mphi1.00E-02_rhoc7.168E+14_sphim.316.dat` -- single
polytrope EOS (`K=100`, `Gamma=2`, code units, not MPA1's piecewise polytrope),
`omega_c=12` (the module's previously-validated/default coupling, vs. MPA1's untested
`omega_c=153`), `mass2=0.0001` (`mphi=0.01`, vs. MPA1's `0.01`). Interpolated central
density (`~6.7e14` after unit conversion) matches the filename's `rhoc=7.168e14` to
similar precision as the MPA1 check. Input:
`inputs/dyn_grmhd/rns_st_gam2k100.athinput` (same `102.4`/`nx=64` domain/BC/gauge as
the low-res MPA1 run). Result: **clean for all 30 cycles (`t~19.2`), zero NaNs**,
`rho-max`/`alpha-min`/`sphi-max` all show small (~1-2%), smooth, non-secular
oscillation about the initial equilibrium values -- no sign of the MPA1 run's runaway
instability at either resolution. This doesn't settle which of hypotheses (a)/(b)
above is right for MPA1 (EOS, `omega_c`, and `mass2` all differ between the two
stars, so nothing is cleanly isolated), but it does show the pgen/pipeline itself
handles a genuinely nonzero, spatially-varying `sphi` correctly through a full
evolution when the star is otherwise well-behaved -- the MPA1 instability is
specific to that configuration, not a generic bug in the new pgen or the
reflect-BC/back-reaction wiring.

Central `sphi` sanity check: `sphi-max=0.305` at `t=0` vs. the filename's central
value `sphim=.316` -- a ~3.5% difference, the same order of match quality already
accepted for central density in both stars (attributable to the ID-grid-to-Cartesian
interpolation, not a new discrepancy).

`RnsStHistory` was extended to report true signed `sphi-max`/`sphi-min` (was
previously a single `|sphi|`-max column) -- useful now that a rotating or
sign-changing scalar profile could in principle be tested later; for both stars
tested so far `sphi` is positive everywhere so `sphi-max` is unchanged from before,
and `sphi-min` is a small positive value near the outer domain boundary where the
field decays toward zero far from the star.

Also added `variable = scalarfield` (and per-component `sf_sphi`/`sf_pi`) support to
`outputs.hpp`/`basetype_output.cpp`, modeled on `Tmunu`'s dispatch pattern -- works for
`bin`/`tab`/`vtk` output alike, since they all share one variable-dispatch path.
Verified against a real run via `vis/python/bin_convert.py`: `sf_sphi` max/min in the
`.bin` output match the `.hst` diagnostics exactly, `sf_pi=0` everywhere (correct for
a `J=0` star).

## Phase 6 -- ScalarFieldBCs: physical-boundary ghost-zone fill (closes a Phase-0 gap)

Found and fixed a real, live gap while investigating `ScalarField::ApplyPhysicalBCs`
(`scalar_field_tasks.cpp`): it was a true no-op beyond the generic `user_bcs_func`
hook (which never fires for any RNS-ST input, since those use `reflect`/`diode`, not
`user`). Every other physics module (Hydro, MHD, Radiation, Z4c) has its own
dedicated `<Module>BCs` function in `src/bvals/physics/` that fills ghost zones at
true physical (domain-edge) boundaries every stage; `ScalarField` never got one.
Confirmed this is a *separate* mechanism from inter-MeshBlock communication (which
only exchanges data with real neighbor blocks and never touches true domain-edge
ghost cells, regardless of BC type) -- so this was not a redundant safety net, it was
the only mechanism that could have filled these cells, and it was entirely absent.

**Practical consequence, retroactive**: every `<scalarfield>` run to date (all of
Phase 5, both MPA1 and Gam2_K100) left `sphi`/`Pi` ghost zones at every
`reflect`/`diode` domain-edge face frozen at whatever the problem generator set at
`t=0`, for the entire run -- they were never updated as the interior solution
evolved. This means the Phase 5 conclusion above ("not a generic bug in the new pgen
or the reflect-BC/back-reaction wiring") should be read narrowly: there was, in fact,
no reflect-BC ghost-zone wiring at all for the scalar field at that point, for either
star. The Gam2_K100 stability result itself doesn't change (that run's `sphi` ghost
zones were simply stale for its whole duration and it was still stable), but the
open MPA1 instability question has NOT been re-examined with this fix in place -- it
remains exactly as open as before, now with one more variable (frozen vs.
correctly-filled ghost zones) worth trying if/when someone returns to it.

Fix: new `src/bvals/physics/scalar_field_bcs.cpp`, `MeshBoundaryValues::ScalarFieldBCs`,
a direct structural port of `Z4cBCs`/`BCHelper` (same per-face `switch` over
`reflect`/`outflow`/`diode`/`vacuum`/`inflow`, same coarse-array second pass for
`multilevel`). The one simplification vs. `Z4cBCs`: `sphi` and `Pi` are true
3-scalars (even parity under every reflection), unlike Z4c's tensor components
(which need a hardcoded odd-parity sign flip for 6 of 25 variables) -- so every
`reflect` case here is a plain mirror copy, no parity branch at all. Reuses
`pz4c->opt.extrap_order` for the `outflow`/`diode`/`vacuum` extrapolation order
rather than adding an independent `<scalarfield>` input option, since `ScalarField`
already hard-requires `<z4c>` and its RHS already piggybacks on Z4c's derivative
conventions. Wired via `ApplyPhysicalBCs` (mirroring `Z4c::ApplyPhysicalBCs`) and a
new `bvals.hpp` declaration + `CMakeLists.txt` entry, same pattern as the other three
`*_bcs.cpp` files.

**Verified directly, not just by absence of crashes**: reran `rns_st_gam2k100.athinput`
with `ghost_zones=true` on the `scalarfield` bin output and inspected raw ghost-cell
values at the meshblock touching the inner (reflecting) domain corner. At `t=0` the
mirror property `u0(is-i-1)==u0(is+i)` holds (expected -- the problem generator fills
ghost cells directly from the same interpolator). At `t=3.2` (cycle 5) it *still*
holds exactly, but the interior values themselves have changed slightly (e.g.
`0.0217578448 -> 0.0217578653`) -- confirming the ghost cells are being actively
recomputed from the evolving interior every step, not left stale. Separately checked
the outer `diode` face (a different MeshBlock): ghost-cell values there also shift
between `t=0` and `t=3.2` in a manner consistent with linear extrapolation from the
(also slightly shifted) interior edge value, not a frozen constant. Also reran the
plain `rns_st_gam2k100.athinput` regression (no `ghost_zones` override, `nlim=30`):
still clean, zero NaNs -- this fix doesn't destabilize the one star that was already
working.

## Sample input block

```
<scalarfield>
omega_c   = 12.0
beta0     = 1.0
mass2     = 0.0
sphi0     = 0.0
diss      = 0.0
user_Sbc  = false   # Phase 4: also apply the Yukawa Sommerfeld BC on "user"-flagged faces
```
Requires a `<z4c>` block also be present.
