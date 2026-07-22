# Scalar-Tensor Extension: Development Notes

Status: **Phase 2 (full DEF coupling, vacuum) complete.** This directory implements a massive
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
  a new `<scalarfield>` input block (`omega_c`, `beta0`, `mass2`, `sphi0`, `diss`,
  `newton_tol`, `newton_maxiter` -- the last two unused until Phase 4).
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

- **Boundary-buffer treatment**: `ScalarField`'s own `MeshBoundaryValuesCC` is constructed
  with `is_z4c=true` (`scalar_field.cpp`'s constructor), i.e. it opts into the same
  4th-order-safe prolongation/restriction path Z4c's own fields use, rather than the
  plain 2nd-order path Hydro/MHD use. Reasoning: once Phase 1+ back-reaction is enabled,
  `sphi` is differentiated twice and feeds directly into the Z4c RHS, so it needs the same
  refinement-boundary smoothness Z4c's own metric fields need. This has no effect unless
  `pmesh->multilevel` (SMR/AMR) is on. If this turns out to be unnecessary overhead or
  wrong for some other reason, it's a one-line flip back to `false`.
- **`ApplyPhysicalBCs` is a no-op** (beyond the generic user-BC hook every physics module
  calls). There is no `ScalarFieldBCs` formula yet -- ghost zones at the edge of the whole
  domain will hold whatever the problem generator set them to, or whatever a pgen's
  `user_bcs_func` sets. This is fine for Phase 0/1 (periodic or reflecting test domains);
  Phase 4 is where the Yukawa-falloff Sommerfeld condition gets added (see the plan file).
- **No `SF_Newdt` task**: the scalar field's own characteristic speed is the speed of
  light, already reflected in Z4c's existing CFL-based timestep; no separate timestep
  constraint is needed until Phase 4's stiff mass term requires one (handled through the
  IMEX machinery instead, not a CFL limiter).
- **No `z4c_tasks.cpp` dependency wiring yet**: `Z4c_CalcRHS`'s optional-dependency list
  is untouched. Phase 2/3 will need to add `SF_CalcRHS`/`SF_RescaleT` there once the
  Z4c-side back-reaction terms are actually implemented (see `PLAN.md`'s "Task-list
  wiring" section) -- don't forget this when Phase 2 starts, or the back-reaction terms
  will silently read stale/uninitialized scalar-field data.

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

## Next step: Phase 3

Couple to the `dyn_grmhd` fluid: add the `RescaleTmunu` task (rescale the fluid's
Jordan-frame `Tmunu` by `1/A(phi)` before it enters any geometry equation) and the
matter-trace `T` terms in both the scalar's own RHS and `z4c_calcrhs.cpp`'s K/Khat and
Gamma-tilde^i RHS (currently zero since Phase 2 is vacuum-only). See `PLAN.md`'s
"Fluid stress-energy rescaling" and "Phased rollout" sections for the full plan,
including the open scalarized-TOV initial-data decision. The `Z4c::ADMConstraints` gap
above should ideally be closed alongside this, since Phase 3's own verification plan
(constraint convergence with matter) depends on it being scalar-aware too.

## Sample input block

```
<scalarfield>
omega_c   = 12.0
beta0     = 1.0
mass2     = 0.0
sphi0     = 0.0
diss      = 0.0
```
Requires a `<z4c>` block also be present.
