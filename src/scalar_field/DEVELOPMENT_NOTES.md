# Scalar-Tensor Extension: Development Notes

Status: **Phase 0 (scaffolding) complete.** This directory implements a massive
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

## Next step: Phase 1

Massless (`mass2=0`), scalar's own RHS only, **no** back-reaction on geometry yet (i.e.
`z4c_calcrhs.cpp` still untouched). Implement the actual Klein-Gordon-like RHS in
`scalar_field_calcrhs.cpp`:
```
dt(sphi) = alpha * Pi                                     (+ shift advection)
dt(Pi)   = alpha*(K+2*Theta)*Pi - alpha*D2(sphi) - g^ij di(alpha) dj(sphi)
           - alpha*sphi*(|D(sphi)|^2 - Pi^2)
```
(the matter-trace and mass-term pieces come in Phase 3/4). Use the conformal-variable
convention already established in `z4c_calcrhs.cpp` (differentiate `pz4c->z4c.g_dd`/`chi`
plus `oopsi4`, not `padm->adm.g_dd` directly) for consistency with the rest of the Z4c
sector's finite-difference machinery. Validate with a pgen modeled on
`src/pgen/tests/z4c_linear_wave.cpp`: check `sphi` propagates at light speed, `Pi =
-dt(sphi)/alpha` to machine precision, and convergence order matches the finite-difference
scheme order.

Full physics reference (all modified BSSN/Z4c RHS terms for Phases 2-3, Newton-solve
structure for Phase 4, boundary-condition formula, and the phased test/verification plan)
is in `PLAN.md` in this directory -- read that before starting Phase 1. The equations are
also fully derived and line-cited in `~/SACRA_2D/SACRA_MPI/bssn_st.f90` and
arXiv:2406.05211 Eqs. 2-13.

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
