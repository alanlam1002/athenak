# Massive Scalar-Tensor Gravity Extension for AthenaK -- Full Plan

This is the full architecture/physics plan this module was built from, copied into the
repo so it survives independently of any external planning tool. See
`DEVELOPMENT_NOTES.md` in this directory for what has actually been implemented so far
and the current status; this file is the (mostly static) reference plan/roadmap.

## Context

`~/athenak_st` (git branch `STT`) is a Kokkos-based numerical-relativity code
implementing Z4c/BSSN + dynamical-GRMHD (paper: arXiv:2409.10383). The goal is to extend
it to massive scalar-tensor (Damour-Esposito-Farese-type) gravity, matching the
formulation already implemented in a legacy Fortran SACRA-family code at
`~/SACRA_2D/SACRA_MPI/bssn_st.f90` and published in arXiv:2406.05211 ("Binary neutron
star mergers in massive scalar-tensor theory", Lam, Kuan, Shibata, Van Aelst, Kiuchi).
SACRA already produced physics results for this theory; the motivation for porting to
AthenaK is to take advantage of AthenaK's GPU/Kokkos performance and AMR for future
scalar-tensor BNS-merger production runs.

Both codebases were read in full detail and the physics was cross-checked against the
published paper's equations, so the formulation below is high-confidence.

## Physics being ported

Two new evolved fields, Einstein frame: `phi` (canonical scalar, SACRA `sphi`) and
`Pi := -n^a nabla_a phi` (its "momentum", SACRA `Pi`). Coupling function
`A(phi) = exp(beta0*phi^2/2)` (SACRA hardcodes `beta0=1`; exposed here as a free
parameter -- zero extra cost). Coupling constant `omega_c` (SACRA `omega_c`, default 12)
matches the paper's `B`. Scalar mass^2 `m^2` (SACRA `pmass2`, default 0, massless limit
recovers plain GR).

```
dt(phi) = -alpha*Pi                                                   (+ shift advection)
dt(Pi)  = alpha*(K+2*Theta)*Pi - alpha*D2(phi) - g^ij di(alpha) dj(phi)
          - alpha*phi*(|D(phi)|^2 - Pi^2)
          + 2*pi*alpha*omega_c*T*phi + m^2*alpha*phi*A(phi)
```
`T` = trace of the **fluid** stress-energy (matter only). D2, |D.|^2 use the physical
3-metric.

Modified Z4c geometry RHS (extra terms only, added on top of vacuum BSSN/Z4c):
```
omega_sphi2 = 2/omega_c - 1.5*phi^2
strtr = (1/3)*[omega_sphi2*|D(phi)|^2 + phi*D2(phi) + |D(phi)|^2*(1+phi^2)]
momsca_i = omega_sphi2*Pi*di(phi) + phi*di(Pi) + Pi*di(phi)*(1+phi^2)
           - Atilde_i^j*dj(phi)*phi - (K+2*Theta)*di(phi)*phi/3

dt(alpha) += -3*alpha*phi*Pi          (1+log lapse uses K+1.5*phi*Pi instead of bare K)
dt(Theta) += (alpha/2)*{ -omega_sphi2*(Pi^2+|D(phi)|^2)
             - 2*[-(K+2*Theta)*Pi*phi + phi*D2(phi) + |D(phi)|^2*(1+phi^2)]
             - 2*(m^2/omega_c)*phi^2*A(phi) }
dt(K)     += alpha*omega_sphi2*Pi^2
             + alpha*[phi*D2(phi) - (K+2*Theta)*Pi*phi + |D(phi)|^2*(1+phi^2)
                       - 3*pi*omega_c*phi^2*T + 1.5*(Pi^2-|D(phi)|^2)]
             - 1.5*m^2*phi^2*A(phi)*alpha - (m^2/omega_c)*alpha*phi^2*A(phi)
dt(Atilde_ij) += -alpha*[W^2*omega_sphi2*di(phi)*dj(phi) + W^2*(1+phi^2)*di(phi)*dj(phi)
                          + phi*(DiDj(phi))_TF]
                 + alpha*Atilde_ij*Pi*phi + alpha*gamma_ij*strtr
dt(Gamtilde^i) += -8*pi*gamma^ij*momsca_j
```
Fluid stress-energy (`tmunu.E/S_d/S_dd`) must be rescaled by `1/A(phi)` (Jordan->Einstein)
before it enters any geometry equation; the scalar's own stress is already Einstein-frame,
untouched.

Source line references (ground truth, re-check against these when implementing):
`bssn_st.f90:868` (lapse), `1712-1748` (Theta RHS, scalar RHS), `1810-1927` (Atilde_ij, K
RHS), `1929-2048` (Gamtilde^i RHS / `momsca_i`), `2096-2312` (implicit mass-term Newton
solve), `scalar.f90:1-35` (parameters), `boundary.f90` (Yukawa outer BC, cartoon parity).
Paper arXiv:2406.05211 Eqs. 2-13 independently confirm this structure (fetched and
cross-checked during planning).

## Architecture decision: new sibling module, not a Z4c state-vector extension

AthenaK already separates spacetime geometry (`z4c::Z4c`, 25-var state), ADM 3+1
quantities (`adm::ADM`), and matter stress-energy (`Tmunu`) into distinct classes wired
together through `MeshBlockPack` pointers (`pz4c`, `padm`, `ptmunu`), gated by
`pin->DoesBlockExist(...)` in `src/mesh/meshblock_pack.cpp`. This is the established seam
for optional physics.

**A new `scalarfield::ScalarField` class**, directory `src/scalar_field/`, shaped like
`Z4c` itself (it needs its own evolved state + RHS + RK-update + BCs, not just a passive
data container like `Tmunu`):
- `scalar_field.hpp/.cpp` -- 2-variable state `{I_SF_SPHI, I_SF_PI}`, own
  `DvceArray5D<Real> u0,u1,u_rhs,coarse_u0`, `Options{omega_c, beta0, mass2, sphi0, diss,
  newton_tol, newton_maxiter}` parsed from a new `<scalarfield>` input block via
  `pin->GetOrAddReal`, mirroring `z4c.cpp`'s constructor idiom.
- `scalar_field_calcrhs.cpp` -- the scalar's own `dt(phi)/dt(Pi)` RHS.
- `scalar_field_update.cpp` -- generic RK update, literally `z4c_update.cpp` with
  `nvar = nscalarfield`.
- `scalar_field_tasks.cpp` -- `QueueScalarFieldTasks()` (task registration) + (Phase 3)
  `RescaleTmunu` (the `1/A(phi)` fluid rescale task).
- `scalar_field_geom.hpp` (Phase 2+) -- shared `KOKKOS_INLINE_FUNCTION` helpers for
  `strtr`, `momsca_i`, the covariant Hessian, etc., called from **both**
  `scalar_field_calcrhs.cpp` and the new edits inside `z4c_calcrhs.cpp`, to avoid
  duplicating the tensor algebra.
- `scalar_field_imex.cpp` (Phase 4) -- Newton-solve implicit mass-term update.
- `scalar_field_Sbc.cpp` (Phase 4) -- Yukawa outer boundary condition.

Convention to follow for derivatives: mirror `z4c_calcrhs.cpp`'s established idiom of
working in **conformal** variables (`z4c.g_dd`, `chi`) plus `oopsi4` rather than
differentiating `padm->adm.g_dd` directly (that path -- used only in
`Z4c::ADMConstraints` -- is a once-per-step diagnostic, a different
performance/consistency regime). This keeps the scalar module using the exact same
finite-difference/Christoffel machinery already resident in `z4c_calcrhs.cpp`'s kernel
and avoids introducing a second, physical-metric-based Christoffel code path via `padm`.

## Where each RHS piece is computed

- **Scalar's own RHS** -> `scalar_field_calcrhs.cpp`, using the conformal-variable
  convention above. Needs `D2(phi)`, `|D(phi)|^2`, `gamma^ij di(alpha) dj(phi)`,
  `K = tr(K_ij)` (trace `adm.vK_dd`, equivalently Z4c's `Khat+2*Theta`), and (if a fluid
  is present) the matter trace `T`, read from `pmy_pack->ptmunu->tmunu` **after** it has
  been rescaled by `RescaleTmunu` (ordering matters -- see task graph below). Guarded the
  same way vacuum/matter is guarded elsewhere (`bool is_vacuum =
  (pmy_pack->ptmunu == nullptr)`).
- **Z4c-side extra terms** -> direct edits inside the existing single big kernel in
  `src/z4c/z4c_calcrhs.cpp`, because they need Z4c's own already-computed local
  quantities (`z4c.vA_dd`, `K`, `Theta`, conformal `g_uu`, `oopsi4`) at the same grid
  point:
  - lapse gauge (add `+1.5*alpha*phi*Pi` alongside the existing `f*alpha*vKhat` term)
  - Theta/Hamiltonian RHS
  - K/vKhat RHS
  - Atilde_ij/vA_dd RHS
  - Gamtilde^i/vGam_u RHS
  Guard every addition with a new `bool has_scalar = (pmy_pack->pscalarfield !=
  nullptr)`, exactly mirroring the existing `if (!is_vacuum) {...}` matter-term blocks --
  this keeps pure-GR runs byte-identical (one extra null check per grid point) when
  `<scalarfield>` is absent. (Verified true for Phase 0: with zero back-reaction wired in
  yet, all Z4c outputs are byte-identical with/without `<scalarfield>` present.)

## Fluid stress-energy rescaling by 1/A(phi)

`DynGRMHD::SetTmunu` (`dyn_grmhd.cpp`) **sets** (not adds) `tmunu.E/S_d/S_dd` from MHD
variables, with zero knowledge of any scalar field. Do not edit that function. Instead
add a follow-on task `ScalarField::RescaleTmunu` in `scalar_field_tasks.cpp` that
multiplies `tmunu.E`, `S_d`, `S_dd` in place by `1/A(phi)`, queued with dependency
`{MHD_SetTmunu}` and itself required before `Z4c_CalcRHS` and the scalar's own `CalcRHS`.
This is non-invasive (zero edits to `dyn_grmhd.cpp`), trivially skipped when
`<scalarfield>` is absent, and keeps the "only fluid T_munu gets rescaled, scalar's own
stress never does" invariant automatically satisfied.

## Implicit mass-term treatment (deferred to Phase 4)

SACRA's mass term is **not** naively explicit: it does a per-gridpoint Newton iteration
each RK substage (`bssn_st.f90:2096-2312`, tol `1e-10`) because `A(phi)=exp(beta0*phi^2/2)`
is transcendental, so no closed-form implicit solve exists (unlike AthenaK's only existing
stiff-source precedent, `src/ion-neutral/ion-neutral_tasks.cpp`, whose drag term is
analytically invertible). Recommendation: **require `mass2=0` through Phases 1-3**, and
only in Phase 4 add IMEX support (`imex2`/`imex2+`/`imex3`, already implemented
generically in `src/driver/driver.cpp`) plus a `ScalarField::ImpRKUpdate` task doing a
capped (~20 iteration), tolerance-based Newton solve for `(phi,Pi)`, mirroring
`ion-neutral_tasks.cpp`'s control flow (cache explicit RHS, solve implicit piece, cache
`R(U^n)` for later stages). Give `ScalarField` its **own** `impl_src_sf` array rather than
reusing/resizing `Driver::impl_src` (currently hardcoded to 8 slots and gated only on
ion-neutral, and ion-neutral/z4c are mutually exclusive today) -- avoids touching `Driver`
at all.

## Task-list wiring (required correctness fix, not optional)

`numrel::NumericalRelativity::NeedsPhysics` classifies any `TaskName` **by its numeric
position** in the enum (`< MHD_NTASKS` -> MHD, `< Z4c_NTASKS` -> Z4c,
`< SF_NTASKS` -> ScalarField (added in Phase 0), else -> `Phys_None`, always
"available"). New scalar-field task names must be added *and* explicitly classified, or
`DependenciesMet` will loop forever searching for a task that was never queued when
`<scalarfield>` is absent, aborting task-list construction. (Done for Phase 0's own task
set; Phase 2/3 will need to extend `Z4c_CalcRHS`'s optional-dependency list in
`z4c_tasks.cpp` from `{MHD_SetTmunu}` to `{MHD_SetTmunu, SF_RescaleT, SF_CalcRHS}` once
the back-reaction terms exist -- the existing `AddExtraDependencies` mechanism only pulls
in tasks whose physics module actually exists, so this is safe when `<scalarfield>` is
absent.)

`SF_CalcRHS` should read the *previous stage's* converted ADM data (same lagged-metric
convention already used by `MHD_SetTmunu`) -- no new synchronization primitive needed,
just copy the existing pattern.

## MeshBlockPack wiring

`pscalarfield` pointer + destructor entry in `meshblock_pack.hpp/.cpp`; in `AddPhysics`
(right after the existing z4c/adm block), construction gated on
`pin->DoesBlockExist("scalarfield")`, hard-error if `pz4c == nullptr` (scalar-tensor
requires full Z4c evolution, not static `<adm>`-only backgrounds). Done in Phase 0.

## Boundary conditions (Phase 4)

`src/z4c/z4c_Sbc.cpp` implements a massless Sommerfeld condition
(`rhs.f = -f/r - s^i*di(f)`) per variable, no generic asymptotic-falloff mechanism. Add
`ScalarFieldSommerfeld` in `scalar_field_Sbc.cpp` mirroring that structure but with the
Yukawa-consistent radial ODE `rhs = -(f-f_inf)/r - m*(f-f_inf) - s^i*di(f-f_inf)`; reduces
exactly to the massless case when `mass2=0` (good regression check).

## CMakeLists

`scalar_field/*.cpp` added to the unconditional source list in `src/CMakeLists.txt`, same
style as the existing `z4c/*.cpp` group. Done in Phase 0. No new `-DPROBLEM=` gating
needed for the module itself (only for specific new test pgens, same convention as
existing z4c pgens).

## Phased rollout and verification

| Phase | Scope | Verification | Status |
|---|---|---|---|
| 0 -- scaffolding | `ScalarField` class/arrays/tasks wired in, RHS is a no-op, `has_scalar` guards all false-by-default | Existing `tst/` Z4c + dyn_grmhd regression suite byte-identical with `<scalarfield>` absent; compiles/runs as no-op when present | **Done** -- verified via `z4c_linear_wave` smoke test, outputs byte-identical with/without `<scalarfield>` |
| 1 -- decoupling-limit sanity | Massless (`mass2=0`), scalar's own RHS only, **no** back-reaction on geometry (`z4c_calcrhs.cpp` edits not yet enabled) | New pgen modeled on `src/pgen/tests/z4c_linear_wave.cpp`; check `phi` propagates at light speed, `Pi = -dt(phi)/alpha` to machine precision, convergence order matches scheme order | **Done** -- `scalar_field_linear_wave` pgen; measured convergence order 4.52 (n=16->32), 3.54 (n=32->64), matching the nominal 4th-order scheme. Also fixed a real bug: `ScalarField` was missing from `Driver::InitBoundaryValuesAndPrimitives` (per-module opt-in initial ghost-zone fill), see DEVELOPMENT_NOTES.md |
| 2 -- full DEF coupling, vacuum | Enable `z4c_calcrhs.cpp` extra terms + modified lapse gauge, no fluid | `sphi==0` exact recovery (see note below on why this replaces the plan's original `beta0->0` idea); stable, sensible-magnitude evolution with nonzero coupling; full `Z4c::ADMConstraints` convergence deferred | **Done** (back-reaction terms + task-race fix); constraint-convergence diagnostic extension deferred, see DEVELOPMENT_NOTES.md |
| 3 -- coupling to `dyn_grmhd` fluid | `RescaleTmunu` task + matter-trace terms (`T`) in both scalar and Z4c RHS | Regression: real (nonzero) fluid stress-energy + `sphi==0` exact recovery, at production resolution with real MPI decomposition. Scalarized-TOV shooting solve + quantitative SACRA comparison deferred (see note below) | **Done** (code); regression verified, self-consistent scalarized-star ID **not built** -- see DEVELOPMENT_NOTES.md |
| 4 -- massive field + implicit solver | `ImpRKUpdate` Newton solve, Yukawa BC, require `imex2/imex2+/imex3` | Static massive scalar star: verify `exp(-m*r)/r` far-field falloff over several e-foldings; Newton solve converges in a few iterations; compare against SACRA `pmass2>0` single-star run | Not started |
| 5 -- full BNS merger validation | Integration/perf tuning only, no new files expected | Reproduce arXiv:2406.05211 diagnostics (scalar-charge growth, GW dephasing, post-merger remnant scalarization) against SACRA_MPI outputs for matched binary parameters | Not started |

**Open decision, still not resolved**: scalarized-star initial data. SACRA currently
reads an externally pre-computed scalar profile (`input.f90:1186`). AthenaK already has a
pure-GR TOV pgen (`dyngr_tov.cpp`) that is the natural extension point for an in-repo
scalarized-TOV solver, but reusing an external ID pipeline instead (as SACRA does) is also
an option. Phase 3's code (RescaleTmunu + matter-trace terms) is done and regression-
tested, but this decision was **not** made or acted on -- `dyngr_tov.cpp` still has zero
scalar-field awareness, so there is currently no way to produce a genuinely
constraint-satisfying nonzero-`sphi` star. Whichever path is chosen, it's a substantial,
independent numerical-methods task (radial ODE integration + a shooting method for the
scalar-charge boundary condition) deserving its own dedicated session -- see
DEVELOPMENT_NOTES.md for what was validated instead in its absence.

**Note on the `beta0->0` recovery check (superseded)**: the plan originally proposed
comparing a `beta0->0` run against unmodified GR as a Phase 2 sanity check. This doesn't
apply to what was actually implemented: following SACRA line-for-line, every Phase 2 RHS
term below hardcodes `beta0=1` implicitly (e.g. `omega_sphi2 = 2/omega_c - 1.5*sphi^2`,
the `(1+sphi^2)` factors) exactly as SACRA does -- `beta0` is not actually threaded through
these coefficients, so there is no `beta0` to dial to zero. The check that actually matters
and was used instead: every added term is proportional to `sphi` and/or `Pi` (never a bare
constant), so `sphi==Pi==0` identically reproduces unmodified GR to machine precision --
this was verified directly (see DEVELOPMENT_NOTES.md). Generalizing to a free `beta0`
would require re-deriving which coefficients carry an implicit `beta0` dependence from
first principles rather than copying SACRA -- deferred, not needed for matching SACRA's
published results.

**Constraint-convergence diagnostic gap (deferred)**: `Z4c::ADMConstraints`
(`z4c_adm.cpp`) computes the Hamiltonian/momentum constraint monitors from `padm` (the
physical-frame ADM data) and currently has no scalar-field awareness at all -- it only
knows about the fluid's `Tmunu`. With scalar back-reaction now enabled, this diagnostic
under-reports the true constraint violation whenever the scalar carries nonzero
stress-energy (it's missing the scalar's own Hamiltonian/momentum source, an analogous
addition to what's in `z4c_calcrhs.cpp`, but rebuilt in the physical-frame convention
`z4c_adm.cpp` already uses -- notably *simpler* there since no chi/conformal correction
terms are needed). This should be done before relying on `Z4c::ADMConstraints` for
Phase 2/3 verification; it wasn't done as part of this pass to keep scope bounded (see
DEVELOPMENT_NOTES.md).

## Critical files

- New: `src/scalar_field/{scalar_field.hpp,.cpp, scalar_field_calcrhs.cpp,
  scalar_field_update.cpp, scalar_field_tasks.cpp}` -- done (Phase 0-3);
  `scalar_field_geom.hpp` -- **deliberately not created**: Phase 2's covariant-Hessian
  construction duplicates ~10 lines already present in `scalar_field_calcrhs.cpp`
  (Phase 1) rather than factoring it into a shared header, matching the codebase's own
  existing style (Z4c's `Ddalpha_dd` construction isn't factored out either, despite
  `Rphi_dd` needing an analogous chi-correction). Revisit if a third call site appears.
  `scalar_field_imex.cpp, scalar_field_Sbc.cpp` -- not yet created (Phase 4)
- New: `src/pgen/tests/scalar_field_linear_wave.cpp` -- done (Phase 1 validation pgen)
- Edit: `src/z4c/z4c_calcrhs.cpp` (guarded scalar-source insertions, 5 spots: lapse,
  Theta, K/Khat, Aij, Gamma-tilde, plus Phase 3's matter-trace `T` term in K/Khat) --
  **done (Phase 2-3)**. Every coefficient was cross-checked directly against
  `bssn_st.f90` line-by-line, not just the condensed plan equations above -- the plan's
  `Gamma-tilde^i` RHS coefficient (`-8*pi*gamma^ij*momsca_j`) turned out to be wrong; the
  correct term (verified against the source) is `-2*alpha*g_tilde^ij*momsca_j` using the
  *conformal* inverse metric, matching how the pre-existing matter term in the same RHS
  is structured. See DEVELOPMENT_NOTES.md.
- Edit: `src/scalar_field/scalar_field_tasks.cpp` (`ScalarField::RescaleTmunu`, new
  `SF_RescaleT` task) -- **done (Phase 3)**: rescales the fluid's Jordan-frame `Tmunu`
  (`E`, `S_d`, `S_dd`) by `1/A(sphi)` in place, matching SACRA's `tabfac` factor, before
  anything reads it as a geometry-equation source.
- Edit: `src/tasklist/numerical_relativity.{hpp,cpp}` (task enum + dispatch + queueing) --
  done for Phase 0's task set; `SF_RescaleT` added to the enum in Phase 3
- Edit: `src/z4c/z4c_tasks.cpp` / `src/scalar_field/scalar_field_tasks.cpp` (cross-module
  task-race fix: `Z4c_ExplRK` now optionally depends on `SF_CalcRHS`, and `SF_ExplRK` now
  optionally depends on `Z4c_CalcRHS`) -- **done (Phase 2)**, required once both sectors
  read each other's raw evolved state; see DEVELOPMENT_NOTES.md for why this is a real
  correctness issue and not just belt-and-suspenders. Phase 3 additionally makes
  `Z4c_CalcRHS` optionally depend on `SF_RescaleT` (must read the *rescaled* Tmunu).
- Edit: `src/mesh/meshblock_pack.{hpp,cpp}` (pointer, destructor, `AddPhysics` gating) --
  done
- Edit: `src/CMakeLists.txt` (add new source groups) -- done
- Edit: `src/parameter_input.cpp` (add `"scalarfield"` to the block-name allow-list) --
  done (found during Phase 0 smoke testing, not anticipated during planning)
- Edit: `src/driver/driver.cpp` (`Driver::InitBoundaryValuesAndPrimitives` --
  per-module-opt-in initial ghost-zone fill) -- done (Phase 1; this was a real bug found
  via a non-converging validation test, not anticipated during planning -- see
  DEVELOPMENT_NOTES.md for the full debugging account). **Any future new evolved-field
  module must add its own block here.**
- Edit: `src/pgen/pgen.hpp`/`pgen.cpp` (new `ScalarFieldLinearWave` declaration +
  `pgen_name` dispatch entry) -- done (Phase 1)
- Reference only (do not edit): `src/z4c/z4c_adm.cpp`, `src/utils/finite_diff.hpp`,
  `src/dyn_grmhd/dyn_grmhd.cpp`, `src/ion-neutral/ion-neutral_tasks.cpp`,
  `src/driver/driver.hpp`, `src/pgen/tests/z4c_linear_wave.cpp`,
  `src/pgen/dyn_grmhd/dyngr_tov.cpp`
- Physics ground truth: `~/SACRA_2D/SACRA_MPI/bssn_st.f90`, `scalar.f90`, `boundary.f90`,
  `dissipation.f90`; paper arXiv:2406.05211 Eqs. 2-13.
