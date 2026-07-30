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
##
## Status update (2026-07-23): the paragraph above describes item 9's own
## investigation thread specifically (as of that point) -- it predates and does
## not reflect items 10-23 below. Since then: AMR support landed (item 12);
## multipole/Robin outer BCs replaced the original zerofixed truncation (items
## 16-17); P_i/eta were merged into one solve (item 18); psi/alpha_psi's
## convergence check switched to relative solution change (item 20); two more
## genuine bugs were found and fixed in the shared coarse-fine/prolongation
## machinery (item 21, `ProlongateFCMG`'s child-parity clamp and
## `RestrictCoeffOctets`'s missing octet-to-root branch); a second, rotating-
## star pgen was added (item 23) and, after five rounds of user-driven bug
## hunting (an oblate-surface/atmosphere bug, a missing Lorentz factor in the
## velocity primitive, a sign error in the X^i/beta^i vector-Poisson source,
## psi/alpha_psi's own convergence threshold being too loose for this star's
## accuracy, and an unnecessary/counterproductive under-relaxation
## (`init_omega=0.5`) stalling the outer Picard loop short of convergence),
## now reproduces the initial data's own `psi`/mass/angular momentum to
## ~0.001%-0.02% (mass/ang-mom) with the outer loop reaching exact convergence
## well inside its iteration cap -- full production run with all five fixes in
## place completed successfully (jobs 248477+248639, see item 23's own final
## bullet for the numbers). **Two threads are currently open**: item 22
## (migration-test per-cycle convergence plateau at a refined-region
## boundary/the R=40 mesh-explosion divergence) and item 24 (a separate,
## deeper problem on the *same* migration-test star, found 2026-07-24:
## `CFC::InitializeMetric()`'s outer Picard loop converges cleanly but to the
## *wrong* `psi` -- ~2.7x off at the star's center -- for this compact star;
## conclusively ruled out as an AMR/discretization effect via a uniform-grid
## control run, root cause still open, likely in the outer iteration's own
## fixed-point dynamics). Both explicitly paused mid-investigation, not fixed.
##
## Status update (2026-07-25): items 25-26 closed a real, independent gap found
## while investigating item 24 (though ultimately not its cause): `Multigrid
## Driver::ApplyPhysicalBoundariesOctet` never actually implemented `mg_robin`/
## `mg_multipole` at the AMR-octet level despite claiming to in its own doc
## comment, silently falling back to a cruder `zerograd`-like reflection
## instead. Both are now implemented and verified (no NaN/regression, `beta^i`
## correctly stays exactly `0` for a static star); `mg_multipole` was live for
## `P_i`/`eta`'s vector-Poisson solve (`MGCFCVectorPoissonDriver`'s own
## default), `mg_robin` was inert for every current test (`psi`/`alpha_psi`
## never select it, and no test's refined region reaches the domain boundary
## anyway) but is a real correctness fix for any future one that does. Read
## the numbered items below for the current, authoritative state -- don't
## rely on this header alone.
##
## Status update (2026-07-25, #2): items 27-28 tried a self-consistent
## `U_raw*psi^5` alternative to `InitializeMetric()`'s psi Newton solve
## (compile-time-templated via `<cfc> init_use_psi5_source`) as another
## attempt at item 24. Diverges to NaN on the migration test (extreme
## compactness, `2M/R~0.5`) even under heavy Newton damping -- that input now
## explicitly sets `init_use_psi5_source=false`. But it converges cleanly and
## accurately -- and much faster (2-3 outer iterations vs. 25-80+) -- for both
## the mild "BU0" star and the rotating "BU8" star, so as of 2026-07-25 it is
## `InitializeMetric()`'s new DEFAULT for every other star (item 28).
## Per-stage `CFC_SolvePsi` is unaffected either way.
##
## Status update (2026-07-25, #3): item 29 (same day, different mechanism)
## appears to actually RESOLVE item 24 for the migration test: a new
## `<cfc> init_freeze_conserved` one-shot mode, freezing Utilde/S-tilde_i at
## the pgen's own initial metric guess and solving X^i/psi exactly ONCE (no
## outer Picard loop at all), reproduces the analytic TOV solution to
## `~0.01-0.03%` on the migration test's own compact star -- vs. the default
## iterative mode's known-wrong `~0.376-0.71` ratio -- and is equally accurate
## on the mild "BU0" star. Opt-in, default `false` (unlike item 28's default
## flip, this one hasn't been made default anywhere yet). Item 24 is very
## likely closeable via this item; see item 29 for the full result and what's
## still untested (longer dynamical runs, combining with `init_use_psi5_
## source=true`).
##
## Status update (2026-07-26): both of item 29's own follow-ups landed.
## `init_freeze_conserved`/`init_use_psi5_source` are now mutually exclusive
## by construction (item 29's own new bullet) -- combining them was never
## meaningful and can no longer happen silently. A `tlim=200` dynamical run
## (item 30) confirmed excellent mass conservation through the star's first
## oscillation/bounce (`t=0` to `~0.5ms`), so item 24/29's own `t=0`-
## initialization result stands -- but surfaced a SEPARATE, still-OPEN
## concern: the density field collapses to near-floor everywhere by
## `t~0.55ms` and an unexplained `~14%` mass jump appears around the same
## time, most likely tied to the migration test's *static* refined region
## not tracking the star's post-bounce expansion (not yet root-caused, not
## attributed to item 29). Also recorded: a performance fix hoisting two
## per-call scratch allocations into persistent members (item 31), and an
## open (pre-existing, currently harmless) gap in CFC's own AMR-regrid
## support that would matter if `refinement=adaptive` were ever used with
## CFC (item 32, found in a separate session).
##
## Status update (2026-07-26, later): item 32's dynamic-AMR gap is now
## RESOLVED and verified working end-to-end (item 33) -- CFC's own arrays
## get AMR headroom, a new `CFC::ReinitializeMetricForAMR` regrid hook
## re-solves the metric after every regrid, and a second, deeper pre-existing
## bug (stale `coeff_` sizing in `LoadMatterSource`/`LoadNonlinearCoefficient`/
## `LoadReactionCoefficient`, found while verifying the above) is also fixed.
## `refinement=adaptive` is now a supported (if not yet performance-tuned)
## configuration for CFC -- see item 33 for the full design and verification.
##
## Status update (2026-07-26, same day, final): two corrections to the above.
## (1) item 33's own design was itself corrected the same day: the regrid
## re-solve must NOT rebuild conserved variables from primitives (a real
## mass-conservation bug in the original `ReinitializeMetricForAMR`, caught
## by user review before this ever shipped to a real physics run) -- fixed by
## holding conserved variables fixed and doing exactly one plain CFC step,
## recovering primitives via `ConToPrim` afterward. (2) that same fix (and
## item 33's own field-remap warm-start attempt, since reverted) surfaced item
## 34: a separate, pre-existing gap in AthenaK's generic AMR load-balancing
## MPI transfer (`padm`/`pcfc` are missing from `load_balance.cpp`'s packed-
## variable list), which silently corrupts the metric for any block that
## moves to a different MPI rank during a regrid. `refinement=adaptive` is
## therefore verified correct for CFC only when regrid events stay on the
## same rank (item 33's own single-rank test) -- NOT yet safe for a real
## multi-rank production run whose load balancer moves blocks across ranks,
## until item 34 is fixed. **(Item 34 is now fixed -- see item 37, same day.)**
## (3) item 29 also gained a related primitive-
## recovery bug fix, independent of AMR, which reopened item 30's own
## still-unresolved `tlim=200` anomaly as a question -- RE-TESTED and
## CONFIRMED (2026-07-26): the same `tlim=200` run with the fix shows clean,
## physically sensible migrate-and-ring-down behavior throughout, with no
## density collapse and no mass jump. Item 30 is now CLOSED (see item 29's
## and item 30's own updated text for the full result).
##
## Status update (2026-07-26, later still): item 35 simplified `RunXPsiSolvePass`/
## `RunLapseShiftAssemblePass`'s signatures (the `bool` parameters items 29/33
## added are now decided at each call site instead) and swept every comment in
## `src/cfc/` down to concise, non-obvious-WHY-only notes -- no logic change,
## verified via non-comment-token diffing, committed `9f3ae81a`. Item 36 then
## found and fixed a real, independent bug the user spotted by inspection: after
## `InitializeMetric()`/`ReinitializeMetricForAMR` reconcile primitives vs.
## conserved variables (whichever direction that solve's mode requires), the
## OTHER quantity's hydro ghost cells were left stale (or, for `ConToPrim`,
## computed against a not-yet-final metric) with nothing refreshing them before
## the next step reads them -- fixed by re-running `Driver::
## InitBoundaryValuesAndPrimitives()` once more at the end of each function.
## Verified via a 16-rank/288-MeshBlock/5-level smoke test: no deadlock (the one
## real risk an earlier hand-rolled draft of this fix would have hit), no
## FATAL/NaN. See items 35-36 for full detail.
##
## Status update (2026-07-26, later still, #2): item 34's cross-rank AMR-transfer
## gap is now CLOSED (item 37) -- final design: `adm::ADM` owns `u_adm`
## directly (unchanged from before this session; an earlier same-day attempt
## had `pcfc` own it instead with `adm::ADM` aliasing on, since superseded per
## user direction for structural clarity), with a new `coarse_u_adm` member and
## `padm` wired into the generic AMR pipeline (`mesh_refinement.cpp`/
## `load_balance.cpp`) as its own independent physics-module block -- gated on
## `pcfc != nullptr` (not just `padm != nullptr`), since z4c/stationary ADM
## already refresh the metric before/without needing this transfer (verified by
## reading `Driver::InitBoundaryValuesAndPrimitives` directly), so only CFC
## needs it. Re-running the exact 8-rank scenario that produced item 34's
## original confirmed `-nan` metric (job 249526) confirms the bug is gone: zero
## ranks report any metric NaN, versus every rank going to `-nan` before. A
## separate, narrower, non-fatal residual was found and documented: a
## cross-rank-transferred new block's ghost cells briefly read as zero (never
## NaN) during the *first* of two post-regrid `InitBoundaryValuesAndPrimitives`
## calls, before CFC's own metric refresh has run. Item 38 (same day) closes
## most of this by folding that refresh directly into
## `InitBoundaryValuesAndPrimitives`, mirroring `z4c::Z4c::ConvertZ4cToADM`'s
## placement -- confirmed via re-test to cut the residual substantially
## (3584->896 `NANS_IN_CONS` reports/rank, 2->5 clean ranks of 8), though a
## narrower, distinct residual remains (likely a separate, pre-existing gap in
## `u_adm`'s own neighbor ghost exchange for a coarse-fine interface case, not
## yet investigated). See items 37-38 for full detail.

## Status update (2026-07-27, later still, #3): item 38's residual
## investigated much further, across three sessions. One real, scoped fix
## applied and kept (item 39a, `FillCoarseInBndryCC`). The residual's TRUE
## root cause is confirmed directly (item 39b) -- `MeshBlock::SetNeighbors`'s
## octant-parity guard for coarser edge/corner neighbors -- but fixing it
## (removing the guard) trades one non-fatal residual for a genuine
## `dest`-slot collision (item 39d, root-caused directly). Two candidate
## local fixes were designed and empirically tested (item 39e): one fixes
## 39d but regresses item 38; the other fixes both but introduces a NEW,
## worse collision elsewhere. This proves the bug is NOT fixable with a
## per-relationship guard/condition in `SetNeighbors` -- AthenaK's edge/
## corner neighbor-slot scheme has genuinely insufficient capacity for
## irregular AMR topology (more than 2 distinct contributors can need a
## single diagonal-direction slot). A real fix needs a structural change
## (extended slot capacity, or a global post-registration reconciliation
## pass) -- see item 39e for the two options. A follow-up attempt (item 39f,
## same session) found the CONFIRMED, empirically-derived rule distinguishing
## needed from spurious registrations (`recip == nullptr`, i.e. register only
## when the target's own direct diagonal query finds nothing) -- this fixed
## 39d's original case and preserved item 38's fix, but exposed a second,
## distinct collision (two independent finer blocks landing on the same
## slot), and a deterministic tie-break for THAT collision still left an
## unidentified 1408-NaN regression on the 1-rank test, likely a formula bug
## in the non-empirically-validated edge/corner sites. 39c (cross-rank MPI
## abort) is confirmed a SEPARATE, still-unexplained bug under every variant
## tried, not the same mechanism as 39d. The guard-removal fix is NOT
## applied -- reverted, left for a future session.
##
## Status update (2026-07-28, later): a minimal, CFC-independent reproducer
## for this whole bug now exists (item 39g) -- plain hydro, no CFC/GR at
## all, `inputs/tests/lwave_hydro_diag_collision.athinput` -- with BOTH the
## structural (`nghbr`-registration) and physics-level (actual corrupted
## field values, channel-by-channel) evidence checked into the repo. Useful
## for whoever picks up the structural fix next, and for reporting upstream
## to AthenaK maintainers if desired, since this is generic mesh code, not
## anything CFC-specific. See item 39 for the full investigation; 39f has
## the confirmed registration rule and the precise remaining gap; 39g has
## the reproducer and the real corrupted-field numbers.
##
## Status update (2026-07-28, later still): upstream PR #748 ("Fix
## Refinement at Boundary for Z4c/MHD/Hydro/Radiation") cherry-picked onto
## this branch and checked against item 39 -- confirmed, both by static
## analysis of its diff and by an empirical rerun of the 39g reproducer
## post-patch, that it is a real but ORTHOGONAL fix (physical-boundary BC
## ordering relative to prolongation) that does not touch and does not fix
## item 39's `nghbr` `dest`-slot collision. See item 40 for the full
## writeup; item 39's own open problem (the tie-break formula bug) is
## unchanged by this.

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

**Corrected 2026-07-22** — this section described the module's pre-item-18 layout
(separate `P_i`/`eta` solvers) long after item 18 merged them; see that item for the
merge itself and its rationale.

```
src/cfc/
  cfc.hpp / cfc.cpp                     # orchestrator; owns all 4 multigrid drivers
                                         # and the 6-step control flow, queued into
                                         # the NumericalRelativity task graph via
                                         # QueueCFCTasks() (not a direct Solve() call)
  mg_cfc_vector_poisson.hpp/.cpp        # P_i+eta solver (linear, nvar=4 -- P_i at
                                         # channels 0-2, its paired eta at channel 3,
                                         # item 18), shared by both X^i and beta^i
                                         # (2 instances). mg_cfc_scalar_poisson.{hpp,
                                         # cpp} (the old separate eta-only solver)
                                         # was deleted by item 18, not just emptied.
  mg_cfc_conformal_factor.hpp/.cpp      # psi solver (nonlinear scalar, nvar=1)
  mg_cfc_lapse.hpp/.cpp                 # alpha*psi solver (nonlinear scalar, nvar=1)
  cfc_reconstruct.hpp/.cpp              # free-function Kokkos kernels: Adual^ij from
                                         # X^i, vector reconstruction from (P_i, eta),
                                         # final ADM assembly into padm->u_adm
```

4 multigrid driver instances total (was 6 before item 18's merge): `pmgd_pietax`
(`X^i`'s `P_i`+`eta`, packed), `pmgd_pietabeta` (`beta^i`'s `P_i`+`eta`, packed),
`pmgd_psi`, `pmgd_alpha`.

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
  `p_x`/`p_beta` (rank-1, now channels 0-2 of the merged `u_p_x`/`u_p_beta` storage,
  item 18 — `eta`'s own value lives at channel 3 of that same array, no separate
  `eta_x`/`eta_beta` member exists anymore). Genuine scalars (`psi`, `alpha_psi`,
  `a_sq`, `u_tilde`, `s_tilde`) stay plain `DvceArray5D<Real>`, same as
  `gravity::Gravity::phi`.

- **Corrected 2026-07-22 (was stale since item 18): `P_i` and `eta` are now solved
  as one combined `nvar_=4` multigrid solve, not two separate sequential ones.**
  This bullet originally documented the opposite as a deliberate, permanent
  choice — item 18 (2026-07-14) reversed it per later user direction: `Delta P_i
  = S_i` (channels 0-2) and `Delta eta = -S_i x^i` (channel 3) are loaded into one
  merged source array and solved together by `MGCFCVectorPoissonDriver` (`nvar_=4`
  since the merge, was 3), retiring the separate `MGCFCScalarPoissonDriver`
  entirely. See item 18 for why a genuine storage merge (not just "call Solve()
  once") was required — `Multigrid::LoadSource`/`RetrieveResult` always populate
  the driver's *entire* channel range in one call, so two separate 3-/1-channel
  calls could never target one merged driver's internal storage.

- **Boundary conditions reuse existing multigrid BCs (mostly) — corrected
  2026-07-22, was stale since items 16/17.** This bullet originally said
  `X^i`/`beta^i` use `mg_zerofixed` and `psi`/`alpha*psi` use `mg_multipole`, and
  claimed no new BC code was needed — both the specifics and that claim are now
  wrong. Current defaults (each overridable via its own `<cfc>` input, see items
  16/17 for the full derivation): `psi`/`alpha*psi` (solved as deviations
  `delta_psi = psi - 1` etc.) default to `BoundaryFlag::mg_robin` (`<cfc>
  mg_outer_bc`, item 16) — a local `ghost = interior*(r_anchor/r_ghost)^order`
  extrapolation that sidesteps the `mg_multipole` bug class entirely (no moments,
  no MPI reduction) rather than fixing it; `mg_multipole` was tried first for
  these two fields and found to divide by zero, since their matter source lives
  in `coeff_`, not `src_`, which `CalculateCenterOfMass`/
  `CalculateMultipoleCoefficients` only ever integrate (item 9, rounds 1-5).
  `X^i`/`beta^i` (`P_i`/`eta`, merged per item 18 above) default instead to
  `BoundaryFlag::mg_multipole` (`<cfc> mg_poisson_outer_bc`, item 17) — safe for
  these because their source genuinely lives in `src_`, so the existing
  gravity-proven multipole machinery applies without the bug class above; `Delta
  eta = -S_i x^i` predates `x^i`'s own `1/r` falloff and needed `multigrid.hpp`
  generalized to per-channel multipole moments to support this 4-channel driver
  (item 17's Option A) — genuinely new BC-adjacent code, not a same-BC reuse.

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
    -> CFC_BuildSrcX (step 1: S_i from pmhd->u0; solve merged P_i+eta, item 18)
    -> CFC_Rest/Send/Recv/ProlongPiEtaX (ghost-exchange the merged u_p_x)
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
    -> CFC_BuildSrcBeta (step 6: eq. 75 source; solve merged P_i+eta for beta^i)
    -> CFC_Rest/Send/Recv/ProlongPiEtaBeta (ghost-exchange the merged u_p_beta)
    -> CFC_ReconstructBeta (Shibata recon -> beta_u)
    -> CFC_AssembleFinal
    -> [CFC_Rest/Send/Recv/ProlongADM: item 14's own u_adm ghost-exchange round]
    -> MHD_Newdt (optional dep on CFC_AssembleFinal)
  ```
  **Node count corrected 2026-07-22**: this bullet originally said "32 `CFC_*`
  `TaskName` entries total" — stale on two counts since: item 18's P_i/eta merge
  *removed* 8 entries (`...EtaX`/`...EtaBeta` quartets folded into
  `...PiEtaX`/`...PiEtaBeta`), and item 14 *added* `CFC_InitRecv`, 4
  `CFC_Rest/Send/Recv/ProlongADM` entries, and `CFC_ClearSend`/`CFC_ClearRecv` for
  `padm->u_adm`'s own ghost exchange (not shown in the diagram at all as of this
  bullet's original writing). Rather than hardcode a count here that will drift
  again, treat `numerical_relativity.hpp`'s `CFC_*` enum (ending at `CFC_NTASKS`)
  as the single source of truth for the current exact list/count. Each `Rest*`/
  `Send*`/`Recv*`/`Prolong*` quartet is a thin one-liner on `cfc::CFC` mirroring
  `z4c::Z4c::RestrictU`/`SendU`/`RecvU`/`Prolongate`'s exact shape, using one
  `MeshBoundaryValuesCC` + `coarse_*` pair per field (`pbval_pietax`/
  `coarse_u_pietax`, etc. -- `pbval_px`/`pbval_etax` before item 18's merge,
  `cfc.hpp`) -- `is_z4c=false` throughout since CFC is not z4c (as of this
  writing; item 44 later switches all five fields to `is_z4c=true` to reuse
  z4c's higher-order restrict/prolong path).
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
- `src/tasklist/numerical_relativity.hpp`/`.cpp`: `CFC_*` `TaskName` values
  (appended after `Z4c_NTASKS`, count has drifted since this was first written --
  see the task-graph design-decision bullet above, corrected 2026-07-22) and a
  `Phys_CFC` `PhysicsDependency`; both `NeedsPhysics`/
  `DependencyAvailable` are purely ordinal against `Z4c_NTASKS`/`CFC_NTASKS`, so no
  `.cpp` changes were needed there as the list grew (originally 6 entries, then
  32, then adjusted again by items 14/18 -- see above).
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
      **Update (item 23, 2026-07-23)**: that future pgen now exists
      (`xns_rotstar.cpp`, a rotating star with genuinely nonzero `beta^phi`) --
      the gap described above is confirmed still present (`u_p_beta`/`eta_beta`
      still cold-start from zero every run, the pgen has no way to seed them),
      but turned out not to matter in practice: item 23's run converged and
      evolved stably regardless. Still an open, un-implemented optimization if
      warm-starting the shift solve is ever needed for a harder configuration.
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

12. **AMR support for CFC (supersedes open item 3b).** **Status corrected
    2026-07-22: effectively Done, not "In progress"** — this item's own last
    verification note below (2-level AMR, 3 cycles) predates items 20/21 and
    this file's own item 22, which have since exercised `nreflevel_ > 0`
    configurations far more deeply (up to 5 refinement levels, real multi-cycle
    dynamical evolution, the full migration-test setup) without ever hitting the
    `nreflevel_ > 0` guards this item describes removing — confirming they were
    in fact removed and the octet-coefficient machinery below works as
    implemented. What item 21 found is a separate, secondary discretization
    property (a resolution-order truncation floor at the coarse-fine interface),
    not a gap in AMR support itself. Original investigation follows unchanged:
    (re-reading the full octet/AMR machinery in `src/multigrid/` end to end, and
    every CFC multigrid driver, not just re-stating item 3b's original note) found
    the gap is narrower than 3b assumed, plus two additional bugs in *shared*
    (non-CFC) code that item 3b's investigation hadn't surfaced.
    - **Already AMR-capable, confirmed by reading the actual code, not assumed**:
      `MGCFCVectorPoisson`/`MGCFCScalarPoisson` (the linear solvers backing `P_i`/
      `eta` for both `X^i` and `beta^i`, since merged into one `nvar_=4` driver by
      item 18 -- `mg_cfc_scalar_poisson.{hpp,cpp}`, named here, no longer exists)
      already have real `SmoothOctet`/`CalculateDefectOctet`/`CalculateFASRHSOctet`
      bodies (`mg_cfc_vector_poisson.cpp`, and the now-deleted `mg_cfc_scalar_
      poisson.cpp` at the time this was written) -- a direct port of gravity's
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
14. **Physical (non-block, non-periodic) boundary condition for `padm->u_adm`.**
    Done. Closes a real gap surfaced by inspection (not a test failure): unlike
    every other CC field in the codebase (`HydroBCs`, `Z4cBCs`, and CFC's own
    `CFCScalarBCs`/`CFCVectorBCs`), `CFC::RecvADMTask` called
    `RecvAndUnpackCC` with no follow-up physical-BC pass. `RecvAndUnpackCC` is a
    no-op at a genuine physical domain edge (no neighbor block to exchange
    with), and `AssembleConformalMetric`/`AssembleLapseShiftK` only ever write
    the interior (`is..ie`), so `u_adm`'s physical-boundary ghost cells were
    silently frozen at whatever the problem generator wrote there at t=0,
    forever -- even as the interior metric evolved. Since `dyn_grmhd`'s
    geometric source terms differentiate `alpha`/`g_dd`/`beta_u` right up to
    the domain edge, a stale ghost value there contaminates those derivatives
    with a spurious kink between the (evolving) interior and the (frozen)
    ghost region.
    - User's initial suggestion was to reuse z4c's Sommerfeld BC
      (`z4c_Sbc.cpp`) directly. On inspection this doesn't port as-is:
      `Z4cSommerfeld` injects an outgoing-radiation *RHS* at the
      boundary-adjacent *interior* point (`rhs.vTheta`/`vKhat`/`vGam_u`/
      `vA_dd`), consumed by the RK time-integrator -- it presupposes a
      `du/dt = RHS` evolution equation. `u_adm` under CFC has none: every
      channel is a pure algebraic output of the elliptic solve, rebuilt from
      scratch each stage. There is no `rhs.u_adm` to inject into.
    - **Final approach (per user)**: a direct ghost-cell power-law
      extrapolation instead -- for a ghost cell at depth `n` past the last
      interior (domain-boundary) cell, `f_ghost = f_flat + (f_interior -
      f_flat)*(r_interior/r_ghost)^n`, using the true 3D coordinate radius
      from the origin (`r = sqrt(x1v^2+x2v^2+x3v^2)`, the same pseudo-radial
      convention `Z4cSommerfeld` itself uses) -- **not** chained ghost-to-
      ghost; every ghost depth references the same single nearest interior
      cell. `f_flat` and the falloff order `n` are channel-specific: `alpha`,
      `psi4`, `g_dd` -> `f_flat=1` (`0` for `g_dd`'s off-diagonal
      components), `n=1` (mass monopole, `~M/r`); `vK_dd`, `beta_u` ->
      `f_flat=0`, `n=2` (`~1/r^2`, the next order for extrinsic
      curvature/shift around a non-boosted, asymptotically-flat source).
    - New `MeshBoundaryValues::ADMBCs(MeshBlockPack*, DvceArray5D<Real>)`
      (`src/bvals/physics/adm_bcs.cpp`, registered in `CMakeLists.txt`,
      declared in `bvals.hpp` alongside `CFCScalarBCs`/`CFCVectorBCs`), wired
      into `CFC::RecvADMTask` (`cfc.cpp`) exactly as `CFCScalarBCs` is wired
      into `RecvPsiTask`. `reflect` faces use the same explicit per-channel
      parity enumeration `z4c_bcs.cpp`'s `Z4cBCs` already established: a
      rank-2 tensor component (`g_dd`, `vK_dd`) flips sign iff exactly one of
      its two indices is aligned with the reflected axis (`g_xy` odd under an
      x1 reflection, `g_xx`/`g_yy`/`g_zz` even); a vector component
      (`beta_u`) flips iff its own index is; scalars (`alpha`, `psi4`) never
      flip -- implemented generically via a small per-channel lookup
      (`GetADMChannelInfo`/`ChannelFlipsAtAxis`, anonymous namespace) rather
      than z4c's hand-enumerated `if` chain, since `u_adm` has 17 channels
      (vs. z4c's own conformal set) but the same 3 structural cases (scalar/
      vector/rank-2-tensor) cover all of them. `outflow`/`diode`/`vacuum`/
      `inflow`/`user` all get the same falloff extrapolation (no ADM
      "inflow table" concept exists, matching `CFCScalarBCs`/`CFCVectorBCs`'s
      own precedent of folding those four together).
    - **Verified**: rebuilt cleanly (including the `CMakeLists.txt`
      registration). Re-ran the existing single-level octant TOV control
      input (`ix1/2/3_bc=reflect`, `ox1/2/3_bc=diode` -- exercises both new
      branches on all three axes): completes all 3 cycles cleanly, zero NaN/
      FATAL, and the `InitializeMetric` convergence trace is bit-for-bit
      identical to the pre-change run (expected: `ADMBCs` only touches
      `u_adm`'s ghosts, which `InitializeMetric`'s own X^i/psi loop never
      reads). Not yet separately checked against a run where the boundary
      ghost region's *value* actually visibly departs from its t=0
      initialization (would need a longer run or a dedicated diagnostic dump
      to confirm quantitatively; the qualitative check here is that the
      mechanism runs, compiles against the right enum values, and doesn't
      break anything already passing).
15. **Relax the timestep's speed-of-light restriction for dynamical-GR MHD (the
    path CFC uses).** Done. `mhd::MHD::NewTimeStep` (`mhd_newdt.cpp`) hardcodes
    `max_dv=1` (the speed of light) for every `is_dynamical_relativistic_` run --
    always correct but far more conservative than necessary once a real fast
    magnetosonic speed can be computed. Upstream PR #698
    (github.com/IAS-Astrophysics/athenak/pull/698) closes this gap for the
    *static*-background GR path only (`is_general_relativistic_`, e.g. a fixed
    Kerr-Schild metric) behind a new `<time>/gr_dt` flag, using `mhd`'s
    ideal-gas-only `EquationOfState::IdealGRMHDFastSpeeds` and a closed-form
    background metric (`ComputeMetricAndInverse`) -- it does not touch
    `is_dynamical_relativistic_` at all.
    - Couldn't reuse that PR's approach verbatim: dyn_grmhd's metric is the
      actual solved/evolved pointwise ADM data (`padm->adm.g_dd/beta_u/alpha`),
      not a fixed analytic background -- there's no closed-form metric to
      evaluate at an arbitrary point the way `ComputeMetricAndInverse` does.
      dyn_grmhd also uses the primitive-solver EOS infrastructure
      (`PrimitiveSolverHydro<EOSPolicy, ErrorPolicy>`, supporting piecewise-
      polytrope/tabulated/hybrid EOS, not just ideal gas), so the wavespeed
      calculation must go through `PrimitiveSolverHydro::
      GetGRFastMagnetosonicSpeeds` -- already used identically by this
      module's own Riemann solvers (`rsolvers/{llf,hlle}_dyn_grmhd.hpp`) --
      rather than `EquationOfState::IdealGRMHDFastSpeeds`.
    - New `dyngr::DynGRMHD::NewTimeStep` (pure virtual, `dyn_grmhd.hpp`),
      implemented in `DynGRMHDPS<EOSPolicy, ErrorPolicy>`
      (`src/dyn_grmhd/dyn_grmhd_newdt.cpp`, registered in `CMakeLists.txt`),
      replacing `&MHD::NewTimeStep` for the `MHD_Newdt` task
      (`dyn_grmhd.cpp`'s `QueueDynGRMHDTasks()`). Gated on a new `gr_dt` member
      read from `<time>/gr_dt` (same input key/default=false as PR #698, for a
      single consistent knob across both GR paths) -- `false` preserves the
      old `max_dv=1` behavior exactly; `true` computes the real per-direction
      fast magnetosonic speed pointwise from `w0`/`bcc0`/`padm->adm` (undensitizing
      `bcc0` via `isdetg`, computing the comoving-frame `b^2` the same way
      `SingleStateFlux` does, and the per-direction inverse-metric diagonal
      `gii` the same way `{llf,hlle}_dyn_grmhd.hpp` do) via
      `PrimitiveSolverHydro::GetGRFastMagnetosonicSpeeds`.
    - **Bug found while wiring this up (before any run)**: a direct
      `#include "eos/primitive_solver_hyd.hpp"` before `#include "dyn_grmhd.hpp"`
      failed to compile (`PrimitiveSolverHydro` reported as an incomplete/wrong
      type) -- `dyn_grmhd.hpp` and `eos/primitive_solver_hyd.hpp` include each
      other (the latter needs `DynGRMHDPS`, the former needs
      `PrimitiveSolverHydro` for `DynGRMHDPS::eos`'s member declaration), so
      whichever is included *first* in a translation unit determines whether
      the other sees a fully-defined `PrimitiveSolverHydro` by the time it's
      needed. `dyn_grmhd_fluxes.cpp` already gets this right (includes
      `dyn_grmhd.hpp` before anything primitive-solver-related, and doesn't
      include `eos/primitive_solver_hyd.hpp` directly at all, relying on the
      transitive include); `dyn_grmhd_newdt.cpp` now follows the same order.
    - **Verified**: rebuilt cleanly. Ran the existing octant TOV control input
      unmodified (`gr_dt` unset, defaults false): bit-for-bit identical
      `dt`/`time` trace to every prior run of this test (`dt=6.4e-1`,
      `time=1.92` after 3 cycles) -- confirms zero behavior change when the
      flag is off. Same input with `<time> gr_dt = true` added: completes all
      3 cycles cleanly, zero NaN/FATAL/warnings, and `dt` grows to `2.313`
      (`time=4.377` after 3 cycles) -- roughly 3.6x larger than the
      speed-of-light bound, as physically expected for a fast magnetosonic
      speed well below `c` in this non-relativistic-velocity static star.
    - Renamed to `dyn_grmhd_newdt.cpp` (was `dyngr_mhd_newdt.cpp`) to match
      this directory's `dyn_grmhd_*.cpp` convention (`dyn_grmhd_fluxes.cpp`,
      `dyn_grmhd_fofc.cpp`) rather than introducing a new `dyngr_*` prefix.
    - **Bug found by the user, cycle-0-only**: a real production `gr_dt=true`
      run (`/sakura/ptmp/tlam/athenak_run/cfc_stability_v2`) still showed the
      old speed-of-light `dt` at cycle 0. Root cause: `Driver::Initialize()`
      (`driver.cpp`) primes `pmesh->dt` *before* the main evolution loop (and
      hence before the task graph's own `MHD_Newdt` task ever runs) by calling
      `pmesh->pmb_pack->pmhd->NewTimeStep(this, nexp_stages)` directly --
      `mhd::MHD::NewTimeStep` is **not virtual**, so this call always ran the
      base (hardcoded `max_dv=1`) version through the plain `pmhd` pointer,
      completely bypassing `dyngr::DynGRMHDPS::NewTimeStep` regardless of
      `gr_dt`. `MeshRefinement::AdaptiveMeshRefinement()` (`mesh_refinement.cpp`)
      has the exact same priming call, run after every AMR regrid event, with
      the identical bug. Both fixed by dispatching through
      `pmy_pack->pdyngr->NewTimeStep(...)` instead of `pmhd->NewTimeStep(...)`
      whenever `pdyngr != nullptr` (dyn_grmhd active) -- the task graph's own
      `MHD_Newdt` task was never affected (it's queued directly against
      `&DynGRMHDPS<...>::NewTimeStep`, see above), so every cycle *after* the
      first (or after the first post-refinement cycle) already had the
      correct, larger `dt`; only these two one-time priming calls were stuck
      on the conservative fallback.
    - **Verified**: rebuilt cleanly. Re-ran the octant TOV control input with
      `<time> gr_dt = true`: cycle 0 now shows `dt=2.749`, consistent with
      cycles 1-3's `dt~2.3-2.6` -- confirms the priming call now picks up the
      real fast-magnetosonic speed from the very first cycle instead of the
      old hardcoded `max_dv=1`. `gr_dt` unset still reproduces the unchanged
      `dt=6.4e-1` at every cycle including cycle 0, confirming zero regression
      for the default (off) path.
16. **Robin outer boundary condition for `psi`/`alpha_psi`'s multigrid solves,
    replacing the `mg_zerofixed` truncation (item 9 rounds 6-7).** Done (root grid
    and per-MeshBlock V-cycle levels; AMR octets explicitly out of scope, see
    below). `MGCFCConformalFactorDriver`/`MGCFCLapseDriver` had been left on
    `BoundaryFlag::mg_zerofixed` (Dirichlet `delta_psi=0`/`delta_(alpha*psi)=0`
    exactly at the domain edge) ever since item 9 round 6 found `mg_multipole`
    divides by zero for these two solvers (`CalculateCenterOfMass`'s
    `1.0/totals[0]`, since `Utilde`/`Ahat^2` live in `coeff_`, not `src_`, and
    `CalculateMultipoleCoefficients()` only ever integrates `src_`). That was
    always a known, explicitly-accepted accuracy tradeoff, not a fix: the true
    asymptotic behavior (Gmunu eq. 77/78) is an isolated `~M/(2r)` falloff, not
    zero, so `mg_zerofixed` is only the leading-order (monopole-zero) truncation
    of the correct condition. (Round 7's suspicion that this mismatch was also
    causing the "Failed to converge" messages was later falsified by round 8 --
    those were four unrelated bugs, fixed by round 18 -- so this item is a pure
    accuracy improvement to an already-converging solver, not a bugfix.)
    - **User's request**: replace `mg_zerofixed` with a genuine Robin (mixed
      value/derivative) condition, in the spirit of the `1/r^n` extrapolation
      technique already implemented for `padm->u_adm`'s outer ghost cells (item
      14, `src/bvals/physics/adm_bcs.cpp`). That technique *is* a discretized
      Robin condition: for any field `u ~ C/r^n` asymptotically, `du/dr + n*u/r
      = 0` holds for *any* `C`, so filling a ghost cell as `u_ghost = u_anchor *
      (r_anchor/r_ghost)^n` enforces the correct falloff without ever needing to
      know `C` -- no matter integral, no MPI reduction, no `src_`/`coeff_`
      dependency at all. This sidesteps `mg_multipole`'s entire bug class here
      rather than fixing it (fixing `CalculateCenterOfMass`/
      `CalculateMultipoleCoefficients` to read `coeff_` instead of `src_` would
      still need a from-scratch renormalization re-derivation, per item 9 round
      6's own note -- not attempted here).
    - **Scope decision (confirmed with user)**: the V-cycle touches the outer
      boundary through three separate ghost-cell fillers --
      `MultigridDriver::MGRootBoundary` (root grid), `MultigridDriver::
      PhysicalBoundary` (per-MeshBlock levels, `multigrid_tasks.cpp`), and
      `MultigridDriver::ApplyPhysicalBoundariesOctet` (AMR octets, item 12).
      `ApplyPhysicalBoundariesOctet` has no physical-position math at all today
      (pure index-space `sign=+-1` reflection; even `mg_multipole` skips real
      position computation there, relying on inheriting values from the coarser
      root grid) -- deriving an octet's physical position from its
      `LogicalLocation` would be genuinely new arithmetic with nothing existing
      to mirror. Every current CFC test input places mesh refinement near the
      star, never at the domain edge, so no octet ever actually touches the
      outer physical boundary. This pass implements Robin **only for the root
      grid and per-MeshBlock levels**; `ApplyPhysicalBoundariesOctet` is
      unchanged, with a comment (next to its existing multipole note)
      documenting that any face marked `mg_robin` there silently falls back to
      the same `sign=+1` (zerograd-like) reflection -- an accepted, narrowly
      scoped gap, matching this codebase's precedent of flagging rather than
      silently ignoring deferred octet work (item 3b, before item 12 closed
      it). Revisit if a future input ever places refinement at the domain edge.
      **Closed (2026-07-25, item 25)**: the position derivation turned out
      simpler than assumed here -- `ApplyPhysicalBoundariesOctet` now
      implements `mg_robin` properly, see item 25.
    - New `BoundaryFlag::mg_robin` (`bvals.hpp`). New `int robin_order_` on the
      base `MultigridDriver` (default `1`, next to `mporder_`) -- the falloff
      power `n`, physically `1` for both solvers (Gmunu eq. 77/78's leading
      monopole term) but exposed rather than hardcoded, matching `mporder_`/
      `mg_omega_psi`/`psi_floor`'s existing precedent. Ghost fill uses raw mesh
      coordinates (`pmy_mesh_->mesh_size` / `mb_size.d_view(m)`), deliberately
      **not** the multipole origin `mpo_` -- fully decoupled from that separate
      (still-inert) machinery, recomputed at whichever V-cycle level is
      currently being smoothed (all three fillers are re-invoked every
      red/black half-sweep at every level, confirmed by reading
      `OneStepToFiner`/`OneStepToCoarser`/`SolveCoarsestGrid`/the
      `mg_to_finer`/`mg_to_coarser` task-list wiring -- not just once at the
      finest level).
    - New `<cfc>` inputs (both drivers): `mg_outer_bc` (string, `"robin"`
      default, `"zerofixed"` for A/B rollback -- mirrors gravity's own `mg_bc`
      string precedent, `mg_gravity.cpp`) and `mg_robin_order` (int, default
      `1`, sets `robin_order_`). The reflect-face -> `mg_zerograd` loop and
      `mporder_`/`autompo_`/`nodipole_`/`AllocateMultipoleCoefficients()` (still
      inert, still available for a future real multipole fix) are unchanged.
    - **Files**: `src/bvals/bvals.hpp` (enum); `src/multigrid/multigrid.hpp`
      (`robin_order_`); `src/multigrid/multigrid_driver.cpp`
      (`MGRootBoundary`'s device+host Robin blocks, `ApplyPhysicalBoundariesOctet`'s
      doc comment, constructor init list); `src/multigrid/multigrid_tasks.cpp`
      (`PhysicalBoundary`'s 6 new `else if` branches); `src/cfc/mg_cfc_
      conformal_factor.cpp`, `src/cfc/mg_cfc_lapse.cpp` (constructors).
    - **Verification, no compiler in this sandbox (same recurring constraint as
      every other CFC step)**: `cpplint`-clean plus manual cross-check against
      the existing, battle-tested `mg_multipole` code at each of the two call
      sites (same per-level `ncx/ncy/ncz`/`dx` pattern, same cell-center-
      coordinate formula, same device+host duality). **Not yet run against a
      real build** -- next step once built on Sakura: rerun `cfc_stability_v2`/
      `cfc_tov.athinput` with the new default (`mg_outer_bc=robin`) against an
      explicit `mg_outer_bc=zerofixed` control run of the same setup, confirm
      no NaN/inf, `mg_threshold=1e-10` still converges with zero "Failed to
      converge" messages (matching the zerofixed baseline item 9 established),
      `psi`/`alpha_psi`'s boundary-adjacent values now differ smoothly from `1`
      instead of being pinned exactly at it, and spot-check the converged
      `psi`/`alpha` against `dyngr_tov.cpp`'s analytic isotropic-TOV profile
      (the same diagnostic item 9 rounds 7-8 used).
17. **Multipole outer boundary condition for `P_i`/`eta`'s linear Poisson solves
    (`X^i`/`beta^i`'s own decomposed vector-potential/scalar equations).** Done.
    Following item 16 (Robin for `psi`/`alpha_psi`), checked whether the *linear*
    Poisson solves that produce `P_i`/`eta` (`MGCFCVectorPoissonDriver`/
    `MGCFCScalarPoissonDriver`) could use `BoundaryFlag::mg_multipole` instead of
    their `mg_zerofixed` default, since (unlike `psi`/`alpha_psi`) these equations'
    matter source is a real, unmodified `src_` (`LoadPoissonSource`'s `fac=1.0`,
    loaded via the generic `Multigrid::LoadSource`, no `coeff_`, no custom Newton
    relaxation -- both drivers reuse the *generic* `Smooth`/`CalculateDefect`
    templates unmodified). Confirmed `ScaleMultipoleCoefficients()`'s normalization
    constants (`c0=0.25/pi`, etc.) are the generic Green's-function/solid-harmonic
    constants for *any* `-Delta u = src` equation (not a gravity-specific
    `-4*pi*G*rho` calibration, despite the comment above them name-checking
    gravity's convention) -- confirmed by reading `LoadSource`'s body (`src_ = fac *
    argument`, `multigrid.cpp:306`) -- so no re-derivation was needed, unlike the
    `psi`/`alpha_psi` case that motivated item 16's Robin approach instead.
    - **Real gap found and fixed**: every current multipole user (gravity) is
      `nvar_=1`; `CalculateMultipoleCoefficients`/`CalculateCenterOfMass`/both
      ghost-fill sites (`MGRootBoundary`, `PhysicalBoundary`) all hardcoded channel
      `0`. `P_i` (`nvar_=3`: `P_x,P_y,P_z`, fully decoupled) would have silently
      gotten wrong/missing boundary data on channels 1-2 (`P_y`/`P_z`) without a
      fix. `eta` (`nvar_=1`) had no such problem. Per user direction, generalized
      the shared machinery to be per-channel (rather than splitting `P_i` into 3
      independent `nvar_=1` scalar solves, which would have reused the already-
      correct single-channel path unmodified but required restructuring `cfc.cpp`/
      `cfc.hpp`'s existing, verified `P_i` ghost-exchange task chain instead) --
      confirmed zero blast radius on gravity (`grep` shows `mg_gravity.{hpp,cpp}`
      never reads `mpcoeff_`/`d_mpcoeff_` directly, and every new per-channel loop
      below reduces to its original single iteration at `nvar_=1`).
    - `multigrid.hpp`: `Real mpcoeff_[25]` -> `Real mpcoeff_[kMaxMultipoleChannels*25]`
      (new `static constexpr int kMaxMultipoleChannels = 4`), flat-indexed
      `mpcoeff_[v*25+c]` (fixed per-channel stride of 25 regardless of `nmpcoeff_`'s
      actual value -- already the pre-existing convention for the single-channel
      case). `d_mpcoeff_` kept as a flat `DvceArray1D<Real>` (not switched to 2D) so
      a raw `v*25` pointer offset into it is always contiguous regardless of Kokkos
      layout, mirroring `MGOctet`'s own manual flat indexing for the same reason;
      just sized `nvar_*25` instead of the hardcoded `25`.
    - `multigrid_driver.cpp`: `AllocateMultipoleCoefficients()` gained a
      `nvar_ <= kMaxMultipoleChannels` fatal-error guard (the natural "multipole is
      being activated" hook). `CalculateMultipoleCoefficients()` wrapped in a
      `for (v=0; v<nvar_; ++v)` loop, `src(m,0,...)` -> `src(m,v,...)`,
      `mpcoeff_[c]` -> `mpcoeff_[v*25+c]` -- **the `memset`/`MPI_Allreduce` extents
      had to become `nvar_*25`, not `nvar_*nmpcoeff_`**, a bug caught and fixed
      mid-implementation: since the storage stride is a fixed 25 regardless of
      `nmpcoeff_` (9 when `mporder_=2`), a narrower `nvar_*nmpcoeff_` extent would
      leave channel 1+'s region partly stale/uninitialized across stages whenever
      `mporder_=2`. `ScaleMultipoleCoefficients()` similarly wrapped in a per-channel
      loop (identical constants, `mc = &mpcoeff_[v*25]`). `SyncMultipoleToDevice()`
      resizes/copies `nvar_*25` elements (was fixed `25`). `MGRootBoundary()`'s
      multipole block (device *and* host paths) gained a `for (v=0; v<nvar; ++v)`
      loop around the existing per-face logic, `u(0,0,...)` -> `u(0,v,...)`,
      `EvalMultipolePhi(..., d_mpc.data(), order)` -> `(..., mc, order)` with
      `mc = d_mpc.data() + v*25` (device) / `&mpcoeff_[v*25]` (host).
      `CalculateCenterOfMass()` deliberately **not** touched -- see below.
    - `multigrid_tasks.cpp` -- `PhysicalBoundary()`'s multipole block already looped
      `for (v=0; v<nvar; ++v)` (finest-level per-MeshBlock case) but reused the
      *same* single-channel `d_mpc.data()` for every `v`; fixed by sizing
      `d_mpc`/copying `nvar_*25` elements (was `25`) and passing
      `d_mpc.data() + v*25` into each `EvalMultipolePhi` call.
    - `mg_cfc_vector_poisson.cpp`/`mg_cfc_scalar_poisson.cpp`: both constructors
      gained new `<cfc>` inputs **distinct from** `psi`/`alpha_psi`'s
      `mg_outer_bc`/`mg_robin_order` (item 16) -- a shared key can't express two
      different defaults for two different call sites that both read it during the
      same construction sequence (`GetOrAdd*` locks in whichever constructor reads
      it first): `mg_poisson_outer_bc` (string, default `"multipole"`, also accepts
      `"zerofixed"`/`"robin"` for rollback/comparison -- `"robin"` reuses item 16's
      `robin_order_` machinery, already generic over `nvar_` in `PhysicalBoundary`,
      needing only the same `MGRootBoundary` per-channel loop this item added
      anyway) and `mg_poisson_mporder` (int, default `4`, independent of
      `psi`/`alpha_psi`'s own `mporder`). Unlike `psi`/`alpha_psi`, `autompo_` is
      forced `false` unconditionally (no `auto_mporigin` input read at all) --
      every current CFC test star sits at the coordinate origin, so the base
      constructor's own `mpo_=(0,0,0)` default is already correct, and skipping
      `CalculateCenterOfMass()` (gated by `if(autompo_)`, mirroring gravity's own
      `Solve()`) avoids generalizing a *third* channel-0-only function that isn't
      actually needed. `Solve()` in both drivers gained
      `if (mporder_>0) { CalculateMultipoleCoefficients(); SyncMultipoleToDevice(); }`
      (mirroring `MGGravityDriver::Solve`), inserted after `SetupMultigrid`, before
      `SolveMG` -- previously absent entirely (neither driver computed multipole
      moments at all, since neither had ever set `mg_multipole` on any face before
      this item).
    - `bvals.hpp`/`cfc_bcs.cpp`: this is a *separate* BC layer from the
      multigrid-internal one above (see item 16's own note on the same distinction
      for `delta_psi`/`delta_alpha_psi`) -- the mesh-level ghost fill applied once
      per stage to CFC's own field arrays after inter-block exchange, read by
      `ComputeADualFromX`/`ReconstructVectorFromPotentials`'s finite differences.
      Once `P_i`/`eta`'s multigrid solve gives them a real `~1/r` falloff at the
      domain edge, their matching mesh-level ghost cells should too, rather than
      staying at the previous crude zero-gradient copy (an artificial kink between
      the correctly-falling-off interior and a flat ghost region right at the
      boundary the finite differences read). `CFCScalarBCs`/`CFCVectorBCs` gained a
      new `int order = 0` trailing parameter (default preserves every pre-existing
      call site's exact behavior with no edits needed there): `order==0` keeps the
      exact old zero-gradient copy; `order>0` uses the same `f_ghost =
      f_interior_anchor * (r_anchor/r_ghost)^order` extrapolation `adm_bcs.cpp`'s
      `ADMBCs` and item 16's Robin BC both already use (`flat=0` unconditionally,
      unlike ADM's per-channel `(flat,order)` table -- none of these CFC potential
      fields have a nonzero flat-space reference value). **Scope: only the 4 fields
      that are themselves direct Poisson-solve outputs** -- `u_p_x`/`eta_x`/
      `u_p_beta`/`eta_beta` (`cfc.cpp`'s `RecvPXTask`/`RecvEtaXTask`/`RecvPBetaTask`/
      `RecvEtaBetaTask`, now passing `order=1`). **Not** `u_x` (algebraically
      reconstructed from `p_x`/`eta_x`, not itself a solve output) or
      `delta_psi`/`delta_alpha_psi` (already has its own dedicated Robin treatment
      at the multigrid level from item 16; its mesh-level BC was deliberately left
      alone there too -- same precedent, not revisited here).
    - **Octet BC scope**: same decision as item 16, carried forward unchanged --
      `ApplyPhysicalBoundariesOctet` still has no physical-position math for any BC
      kind, no current CFC input refines mesh at the domain edge, and multipole at
      octet granularity keeps whatever fallback it already had before this item
      (unchanged by this item). **Closed (2026-07-25, item 26)**: unlike `psi`/
      `alpha_psi`'s Robin gap (inert for every current test), this one was live
      for this driver's own default (`mg_poisson_outer_bc=multipole`) -- see item
      26 for the fix and verification.
    - **Verification, no compiler in this sandbox**: `cpplint`-clean (line length,
      brace balance, checked directly) plus manual cross-check against gravity's
      own already-proven `Solve()`/multipole sequence (`mg_gravity.cpp`) as the
      reference for the per-channel generalization. **Not yet run against a real
      build.** Once built on Sakura: rerun `cfc_stability_v2`/`cfc_tov.athinput`
      with the new defaults (`mg_poisson_outer_bc=multipole` for `P_i`/`eta`,
      `mg_outer_bc=robin` for `psi`/`alpha_psi`, unchanged from item 16) against a
      `mg_poisson_outer_bc=zerofixed` control run -- no NaN/inf, `mg_threshold=
      1e-10` convergence unaffected, and (the concrete signature multipole should
      produce that `zerofixed`/`robin` can't) `P_i`/`eta`'s boundary values should
      carry real non-spherically-symmetric angular structure (dipole/quadrupole,
      per `mg_poisson_mporder`) rather than being pinned to zero or forced into a
      purely isotropic `1/r` falloff -- **explicitly re-verify gravity's own
      behavior is bit-for-bit unchanged** after these shared-code edits, since this
      is the one part of this item with blast radius outside `src/cfc/`.
18. **Merge `P_i`/`eta` into one `nvar_=4` Poisson solve per Shibata pair
    (`X^i`/`beta^i`), retiring `MGCFCScalarPoissonDriver`.** Done. User observation:
    since `P_i` (`nvar_=3`) and `eta` (`nvar_=1`) are independent flat Poisson
    equations with already-independent source terms (both built in the same
    `AssembleVectorSource` pass), the two separate sequential multigrid solves per
    Shibata pair could be combined into one. Made cheap by two pieces of
    infrastructure item 17 already built: the per-channel multipole generalization
    (`kMaxMultipoleChannels=4`, already exactly `nvar_`'s new value) and the shared
    `mg_poisson_outer_bc`/`mg_poisson_mporder` `<cfc>` keys (already read
    identically by both drivers, so merging loses no configurability).
    - **Why a real storage merge was required, not just "call `Solve()` once"**:
      `Multigrid::LoadSource(src, ns, ngh, fac)`/`RetrieveResult(dst, ns, ngh)`
      always populate the driver's *entire* internal channel range `v=0..nvar_-1`
      in one call -- `ns` only offsets which channels of the *caller's* array are
      read from/written to, it does not let two separate calls each fill a
      different sub-range of the driver's own internal storage. So a merged
      `nvar_=4` driver cannot be fed by two independent 3-channel/1-channel calls;
      `p_src`/`p_x`/`p_beta` and their paired `eta` arrays had to become genuinely
      merged 4-channel arrays (`P_i` at channels 0-2, `eta` at channel 3), reusing
      the existing names (`u_p_src`/`u_p_x`/`u_p_beta`, just resized 3->4) rather
      than inventing new ones. The separate `eta_x`/`eta_beta`/`eta_src`
      `DvceArray5D<Real>` members are gone entirely -- every place that read or
      wrote them now reads/writes channel 3 of the merged array directly (no
      dedicated view needed, since nothing besides raw indexing and two functions
      below ever needed a distinct "eta view").
    - `mg_cfc_vector_poisson.{hpp,cpp}`: base-class ctor arg `MultigridDriver(pmbp,
      3)` -> `(pmbp, 4)`. The stencil (`CFCVectorPoissonStencil`) was already
      channel-agnostic (plain decoupled 7-point Laplacian per channel) -- `eta`
      obeys the identical flat Laplacian as `P_i`, just with its own RHS, so
      extending to 4 channels needed no stencil change. The 3 file-local
      `*Channels` helper templates (`SmoothChannels`/`CalculateDefectChannels`/
      `CalculateFASRHSChannels`) hardcoded `for (v=0; v<3; ++v)` -- changed to an
      explicit `int nchan` parameter, with the 3 `MGCFCVectorPoisson::*Pack`
      methods passing their own `nvar_` (a `Multigrid`-protected member already
      directly accessible). The 3 `Octet` methods (`MGCFCVectorPoissonDriver`
      members) hardcoded the same `v<3` loop -- changed to `v<nvar_` directly (no
      parameter needed, already a member). `LoadPoissonSource`/`RetrieveSolution`
      needed **no signature change** -- they already took a single
      `DvceArray5D<Real>&` and passed `ns=0`, which now naturally means "all 4
      channels of the caller's merged array."
    - `mg_cfc_scalar_poisson.{hpp,cpp}`: deleted entirely (confirmed via `grep` --
      no other user in the tree besides `cfc.{hpp,cpp}`/this file). Removed from
      `src/CMakeLists.txt`'s source list.
    - `cfc_reconstruct.{hpp,cpp}`: `ReconstructVectorFromPotentials` took `eta` as
      `const DvceArray5D<Real>&` and internally did `eta_view.InitWithShallowSlice
      (eta, 0)` -- hardcoded channel 0. Added a trailing `int eta_chan = 0`
      parameter (default preserves old behavior), threaded through to
      `InitWithShallowSlice(eta, eta_chan)`. Call sites now pass the merged array
      itself plus `eta_chan=3` -- e.g. `ReconstructVectorFromPotentials(pmy_pack,
      p_x, u_p_x, x_u, 3)` -- no separate `eta_x`/`eta_beta` array needed.
    - `bvals.hpp`/`cfc_bcs.cpp`: `CFCScalarBCs`/`CFCVectorBCs` hardcoded channel 0
      (`u0(m,0,...)`, ~28 call sites) / looped `n=0..nvar-1` with `u0(m,n,...)`
      (`constexpr int nvar=3`, axis-parity checks on `n` itself). Added a trailing
      `int chan0 = 0` parameter to both; mechanically offset every array access
      (`u0(m,0,` -> `u0(m,chan0,` in `CFCScalarBCs`, scripted substitution over
      exactly the function's line range, verified 30/30 occurrences moved and zero
      un-offset survivors afterward; `u0(m,n,` -> `u0(m,chan0+n,` in
      `CFCVectorBCs`, same verification, 30/30) -- **critically, the loop variable
      `n` itself stays 0..2** so the existing `n==0`/`n==1`/`n==2` axis-alignment
      parity checks stay correct relative to the vector's own local components,
      not the absolute channel index (confirmed unchanged via `grep` after the
      substitution). `cfc.cpp`'s merged `RecvPiEtaXTask`/`RecvPiEtaBetaTask` call
      both functions against the same merged array: `CFCVectorBCs(pmy_pack,
      u_p_x, 1)` (implicit `chan0=0`) and `CFCScalarBCs(pmy_pack, u_p_x, 1, 3)`.
    - `cfc.hpp`/`cfc.cpp`: collapsed each `(vector driver, scalar driver)` and
      `(vector pbval, scalar pbval)` pair into one -- `pmgd_px`+`pmgd_etax` ->
      `pmgd_pietax` (and `pmgd_pbeta`+`pmgd_etabeta` -> `pmgd_pietabeta`);
      `pbval_px`+`pbval_etax` -> `pbval_pietax`/`coarse_u_pietax`
      (`InitializeBuffers(4)`, and the beta equivalent) -- drops the `pbval_*`
      instance count from 8 to 6 (`pietax`, `x`, `psi`, `alpha_psi`, `pietabeta`,
      `adm`). `SolveVectorPotential`/`SolveShift` shrank from 6 lines (two
      Load/Solve/Retrieve sequences) to 3 (one). `AssembleVectorSource`'s two
      `eta_src_(m,0,...)` writes became `u_p_src_(m,3,...)` writes directly (no
      alias needed for a removed member); `BuildShiftSource`/`BuildShiftSourceImpl`
      needed no change (only ever touch `p_src`, the 3-channel view, never `eta`).
      Task graph (`QueueCFCTasks()`): collapsed the 8-task `PX`+`EtaX` group
      (previously both depending only on `{CFC_BuildSrcX}` and running in
      parallel, joined at `CFC_ReconstructX`'s 2-dependency wait) into a single
      4-task `CFC_RestPiEtaX -> CFC_SendPiEtaX -> CFC_RecvPiEtaX ->
      CFC_ProlongPiEtaX` chain, `CFC_ReconstructX` now depending on just
      `{CFC_ProlongPiEtaX}` -- identical collapse for the beta side. This removed
      8 `TaskName` enumerators from `numerical_relativity.hpp` (`CFC_NTASKS`
      shrinks by 8 -- confirmed via `grep` that nothing outside
      `numerical_relativity.{hpp,cpp}` depends on its specific numeric value, only
      ordinal comparison `task < CFC_NTASKS`/`Phys_CFC` classification).
      `InitRecvXFields`/`ClearXFields`/`InitRecvTailFields`/`ClearTailFields` (used
      by `InitializeMetric`, item 11) and `InitRecvTask`/`ClearSendTask`/
      `ClearRecvTask` (the task-graph versions) all collapsed their paired
      `InitRecv(3)`/`InitRecv(1)` calls into one `InitRecv(4)` call per merged
      field. **`InitializeMetric`'s hand-rolled task sequence** (`cfc.cpp`,
      duplicates `QueueCFCTasks()`'s wiring outside the normal task graph, easy to
      miss since it's not driven by the same code) was updated in the same pass --
      flagged explicitly here since it's the one place a merge like this is easy
      to apply in only one of the two places it's needed.
    - **Verification, no compiler in this sandbox**: `cpplint`-clean (line length,
      brace balance -- both checked directly via `awk`/Python after every edit)
      plus careful manual cross-checking, since this touches already-implemented,
      already-committed code (item 4's task graph, item 17's multipole work) more
      than any prior single item in this log. Explicitly re-checked: (1) the
      `cfc_bcs.cpp` channel-offset substitution is exhaustive (scripted, then
      grep-verified zero un-offset survivors within the modified functions); (2)
      `InitializeMetric`'s hand-rolled sequence was updated, not just
      `QueueCFCTasks()`; (3) `AllocateMultipoleCoefficients()`'s `nvar_ >
      kMaxMultipoleChannels` guard still passes at exactly `nvar_=4` (`4 > 4` is
      false, confirmed already true before this item since `kMaxMultipoleChannels`
      was set to 4 specifically anticipating this merge). **Not yet run against a
      real build.** Once built on Sakura: rerun `cfc_tov.athinput`/
      `cfc_stability_v2` and confirm `P_i`/`eta`'s converged values are
      bit-for-bit (or numerically identical within roundoff) to a pre-merge run --
      this is a pure refactor with no intended physics change, so any difference
      beyond roundoff indicates a bug in the merge, not an expected side effect.
19. **Merge `CFCScalarBCs`/`CFCVectorBCs` (`bvals/physics/cfc_bcs.cpp`) into one
    shared implementation; fix `vacuum` to use the `order>0` Robin falloff, matching
    `user`, instead of a hard zero.** Done at the time (both functions kept as
    separate public wrappers around one shared `CFCBCsImpl`). **Superseded by item
    41**, which merges the two public wrappers themselves into a single `CFCBCs`/
    `CFCBCsCoarse` entry point -- `CFCScalarBCs`/`CFCVectorBCs` no longer exist as
    of item 41; every reference to them below (and in item 14/40) describes the
    state at the time it was written, not current code. User observation: the two functions
    were structurally near-identical (same 6-face zero-gradient/falloff logic, same
    `CellCenterX`/`r_i`/`r_g` setup) apart from whether a per-channel loop existed at
    all and `reflect`'s parity flip (`CFCScalarBCs`: never; `CFCVectorBCs`: only the
    channel aligned with the face's own axis). Also flagged: `vacuum` was hard-
    zeroed in both, unlike `outflow`/`diode`/`inflow`/`user`, which already use the
    `order>0` falloff.
    - **Precedent confirms the `vacuum` fix**: `adm_bcs.cpp`'s `ADMBCs` -- the
      function this file's own doc comment cites as the model for its falloff
      technique -- already groups `vacuum` with `outflow`/`diode`/`inflow`/`user`
      under *one* case block, unconditionally (confirmed by reading
      `adm_bcs.cpp:127-140`; no separate hard-zero path for `vacuum` exists there
      at all). `cfc_bcs.cpp`'s hard-zero-for-`vacuum` was therefore a real
      inconsistency relative to its own cited model, not a recorded design choice --
      and the same class of bug the `order>0` falloff itself exists to fix (a
      hard-zero ghost cell creates an artificial kink between a smoothly
      falling-off interior and a flat-clamped ghost region, right where
      `cfc_reconstruct.cpp`'s finite differences read across the boundary).
      `vacuum` is a common outer-boundary choice for GRMHD runs, so this is a real,
      reachable case.
    - **Behavior change, stated explicitly**: merging `vacuum` into the shared
      case block makes its behavior order-dependent everywhere, matching `user`
      exactly -- at `order==0` (`u_x`/`delta_psi`/`delta_alpha_psi`'s calls),
      `vacuum` changes from hard `0.0` to the same zero-gradient copy `user`
      already gets at `order==0`; at `order>0` (`u_p_x`/`u_p_beta`'s calls,
      `order=1`), `vacuum` changes from hard `0.0` to the same `1/r^order`
      extrapolation. This is the correct, consistent choice (matches `ADMBCs`'
      own unconditional treatment), not a partial fix gated on `order>0` only.
    - **Merge implementation**: one file-local `CFCBCsImpl(ppack, u0, order, chan0,
      nvar)` (anonymous namespace, alongside the existing `ReflectedValue` helper)
      contains the full 6-face body, with `reflect`'s parity flip generalized to
      `bool flip = (nvar==3) && (n==axis)` (`axis` = 0/1/2 for x1/x2/x3 faces) --
      for `nvar=1` this is always `false` (matches `CFCScalarBCs`'s exact previous
      behavior, since `(nvar==3)` is false regardless of `n`/axis), for `nvar=3`
      with axis alignment it matches `CFCVectorBCs`'s exact previous behavior.
      `CFCScalarBCs`/`CFCVectorBCs` stay as public 3-line wrappers
      (`CFCBCsImpl(ppack, u0, order, chan0, 1)` / `(..., 3)`) with **unchanged
      signatures**, so all 7 existing call sites in `cfc.cpp` needed zero edits --
      confirmed via `grep` after the merge. Net: ~430 lines of near-duplicated
      per-face bodies collapsed to one ~230-line shared implementation plus two
      3-line wrappers.
    - **Verification, no compiler in this sandbox**: `cpplint`-clean (line length,
      brace balance, checked via `awk`/Python) plus manual re-derivation of the
      `flip = (nvar==3) && (n==axis)` formula against both original functions'
      exact behavior, and a `grep` confirming all 6 faces' `vacuum` case is now
      inside the shared falloff block (not a separate hard-zero case) and all 7
      `cfc.cpp` call sites are unaffected. **Not yet run against a real build.**
      Once built on Sakura: exercise a run with `vacuum` set on an outer mesh face
      and confirm no NaN/inf and no visible kink in `P_i`/`eta`/`psi`/`alpha_psi`
      near that boundary, compared to the previous hard-zero behavior.
20. **`psi`/`alpha_psi`'s multigrid convergence check switched to relative
    solution change, not defect norm.** Implemented, then **reverted** (user:
    no measurable improvement found in practice) -- `Solve()` in both drivers
    calls the base-class `SolveMG()` again, exactly as before this item. The
    tried implementation (described in full below) is kept commented out
    in-place immediately after each `SolveMG(pdriver); Kokkos::fence();` call
    (`mg_cfc_conformal_factor.cpp`/`mg_cfc_lapse.cpp`), and the `u_prev_`
    member/its doc comment are commented out in both `.hpp` files, rather than
    deleted outright, in case the approach is worth revisiting later (e.g. with
    a different norm or a genuine test case where it matters) -- avoids
    re-deriving the `Kokkos::parallel_reduce`/`MPI_Allreduce` shape from
    scratch. The `eps_ = pin->GetOrAddReal(...)` lines in both constructors are
    back to their original (pre-item-20) wording, since `eps_` is once again
    only the base class's defect-norm threshold.
    `MGCFCConformalFactorDriver`/
    `MGCFCLapseDriver::Solve()` previously called the shared base-class
    `MultigridDriver::SolveMG()`, which loops V-cycles until the L2 defect/
    residual norm (`CalculateDefectNorm`) drops below `eps_` (`<cfc>`
    `mg_threshold`). User request: switch these two solvers specifically to
    `max_i |u_i - u_old_i| / |u_old_i|` between successive V-cycles instead,
    evaluated pointwise (with a `+1.0e-30` denominator floor, matching the same
    idiom already used for the Robin BC's `r_i/(r_g+1.0e-30)`). Physical
    motivation: `delta_psi = psi - 1` falls off as `~M/r` at large radius (Gmunu
    eq. 77, `M` the ADM mass), so a pointwise relative-change check directly
    controls the accuracy of that leading-order coefficient, which a defect-norm
    threshold doesn't guarantee as tightly. **Scope: only these two nonlinear
    drivers** -- gravity and `MGCFCVectorPoissonDriver` (`P_i`/`eta`) keep their
    existing defect-norm convergence check completely unchanged.
    - **No changes to `src/multigrid/`.** `SolveMG`/`SolveIterative`/
      `SolveVCycle` (`multigrid_driver.cpp`) are `protected`, non-virtual
      `MultigridDriver` members (confirmed via `multigrid.hpp`'s access-specifier
      layout) -- both drivers' `Solve()` now call `SolveVCycle` directly in a
      self-contained loop, bypassing `SolveMG`/`SolveIterative` entirely, with
      zero blast radius on gravity or the P_i/eta solver.
      `Multigrid::GetCurrentData()` (`return u_[current_level_].d_view;`,
      already a public one-liner) exposes the finest level's raw internal
      solution array directly -- no need to go through `RetrieveSolution`'s
      mesh-NGHOST-depth copy-out machinery, since this check only needs the
      multigrid's own internal representation.
    - **New storage**: one new `private:` member per driver,
      `DvceArray5D<Real> u_prev_;` (`mg_cfc_conformal_factor.hpp`/
      `mg_cfc_lapse.hpp`), lazily (re)sized on first use (handles AMR-triggered
      resizing automatically, same lazy-realloc idiom
      `MultigridDriver::SyncMultipoleToDevice()` already uses). Deliberately
      **not** the base class's own `uold_` (used internally for FMG's
      coarse-grid-correction bookkeeping across every level, mutated at points
      not aligned with "start vs. end of one full V-cycle at the finest level" --
      reusing it would risk interfering with semantics not worth reasoning
      through in depth for this change; a dedicated new array is trivially safe
      to reason about instead).
    - **Loop shape**: mirrors `SolveIterative`'s existing robustness (a
      `fshowdef_ >= 2`-gated per-iteration log, a 40-iteration cap with the same
      fatal-error/`pdriver->nlim` truncation behavior on non-convergence) so
      switching criteria doesn't regress observability or safety. The
      `Kokkos::parallel_reduce`/`MPI_Allreduce(..., MPI_MAX, ...)` shape mirrors
      `CFC::InitializeMetric`'s existing `dpsi` convergence check (`cfc.cpp`)
      exactly, just computing a ratio instead of an absolute difference, over
      the multigrid's own internal array (channel `v=0` only, `nvar_=1` for
      both drivers) rather than `delta_psi`/`delta_alpha_psi`'s mesh-shaped
      storage. No new `#include`s needed (`globals.hpp`/`athena.hpp`, which
      supply `global_variable::my_rank`/`MPI_ATHENA_REAL`/
      `MPI_PARALLEL_ENABLED`, were already included in both files).
    - **Reuses the existing `mg_threshold` `<cfc>` key** (already read into
      `eps_` in both constructors) rather than adding a new input parameter --
      same "how tightly converged do you want this solve" semantic label, now
      measuring relative solution change instead of defect norm for these two
      drivers specifically (doc comment on the `eps_ = ...` line updated in both
      constructors to note this; the line itself is otherwise unchanged).
    - **No shared helper factored out**: the convergence loop is duplicated,
      nearly verbatim, between `mg_cfc_conformal_factor.cpp` and
      `mg_cfc_lapse.cpp` -- matches this module's existing precedent (each
      nonlinear solver already has its own fully independent hand-written
      Newton-Gauss-Seidel kernels, no shared base beyond `MultigridDriver`
      itself) rather than introducing a new shared utility file for two ~25-line
      call sites.
    - **Verification, no compiler in this sandbox**: `cpplint`-clean (line
      length, brace balance, checked via `awk`/Python) plus manual re-derivation
      of the `Kokkos::MDRangePolicy` bounds and the `MPI_Allreduce` reduction
      shape against the `InitializeMetric` precedent it mirrors; confirmed
      `SolveVCycle`/`GetCurrentData` are reachable (protected/public
      respectively) from both driver subclasses via `multigrid.hpp`'s
      access-specifier layout. **Not yet run against a real build.** Once built
      on Sakura: rerun `cfc_tov.athinput`/`cfc_stability_v2` and confirm both
      solvers still converge (no "Failed to converge" messages) within the
      40-iteration cap; compare the converged `psi`/`alpha_psi` fields against a
      previous defect-norm-based run (expect small differences from the
      different stopping criterion, not a qualitative change) and specifically
      check the near-star region (where the `M/r` coefficient this change
      targets is best resolved) against the analytic isotropic-TOV profile (the
      same diagnostic DEVELOPMENT.md item 9 already used).

21. **Temporary diagnostic added to `psi`'s solver to root-cause item 12's
    AMR refinement-boundary residual floor** (2026-07-20). Not yet resolved --
    in progress. A user-run investigation (TOV star, low-resolution static-
    refined corner block, dx=0.8/0.4 and dx=0.4/0.2 variants, both compared
    against a matching uniform-resolution control) found `psi`'s multigrid
    solve converges 10-50x worse (defect norm) when a refinement boundary is
    present than the identical uniform-resolution control, and the field-level
    `psi4`/`alpha` jump across the coarse/fine interface does not fully resolve
    even with `mg_npresmooth`/`mg_npostsmooth` raised to 3 -- directly
    reproducing this item's residual-floor note with concrete field/log
    evidence, not just the original round's single stalled-defect number.
    - An extensive read-through of every piece of AMR/octet machinery specific
      to the nonlinear solvers' `coeff_` handling (`MGOctet` storage,
      `SmoothOctet`/`CalculateDefectOctet`/`CalculateFASRHSOctet` in both
      `mg_cfc_conformal_factor.cpp`/`mg_cfc_lapse.cpp`, `TransferCoeffToRoot`'s
      octet-parented branch, `RestrictCoeffOctets`, the `root_flat_buf_stale_`
      cache invalidation sites, and the composite-grid FAS relax/restrict
      sequence in `multigrid_driver.cpp`'s `OneStepToCoarser`/`OneStepToFiner`)
      found nothing provably wrong -- every piece traced correctly against
      either a proven shared pattern (gravity's own octet code) or an
      internally-consistent duplicate of `TransferFromBlocksToRoot`'s existing
      logic. Leading hypothesis: genuine nonlinear-FAS-at-a-resolution-
      transition sensitivity (a Newton relaxation's coarse-grid correction
      isn't exact the way a linear equation's is, unlike gravity/`P_i`/`eta`),
      not a wiring bug -- but not yet confirmed.
    - **Added, to distinguish the two**: `Multigrid::GetCurrentDefect()`/
      `GetCurrentDefect_h()` (`multigrid.hpp`, 2-line pure-additive accessors
      mirroring the existing `GetCurrentData()`/`GetCurrentData_h()` pair
      exactly -- zero behavior change for any existing caller) and
      `MGCFCConformalFactorDriver::DebugReportDefectByLevel()`
      (`mg_cfc_conformal_factor.{hpp,cpp}`, private, called once at the end of
      `Solve()`, gated on a new `<cfc>` `mg_debug_defect_by_level` boolean
      input, default `false`/zero cost). Recomputes `def_` at the finest level
      (`mglevels_->CalculateDefectPack()`, the same call `CalculateDefectNorm`
      already makes internally), then reports this rank's own worst |defect|
      cell and its physical `(x1,x2,x3)` location, split by whether the owning
      MeshBlock is at the mesh's root level or a refined level (via
      `pmy_mesh_->lloc_eachmb[gid].level` vs. `locrootlevel_`, the same
      comparison `TransferCoeffToRoot` already uses). Deliberately no
      `MPI_Allreduce` -- reports only this rank's local worst cell per
      category (skipping a category entirely if this rank owns no MeshBlocks
      of that kind), since the question is *where* the residual concentrates,
      not a single global number.
    - **Explicitly temporary**: unlike item 20's reverted-but-kept-commented-out
      code (which has future value), this diagnostic is meant to be deleted
      outright once the root-cause question below is answered, not preserved.
    - **Run (2026-07-20, job 248058, `cfc_tov_amr_ghosttest_v2_debug`)**: the
      REFINED-block worst cell is *not* diffuse -- it is pinned persistently to
      the same two octet cells across dozens of solves (`gid=1` at
      `(6.2,0.6,0.2)`, `gid=4` at `(0.6,0.2,6.2)`), plateauing at `~0.044-0.045`
      (ticking up slightly, not down) while ROOT-block worst defects are
      `1e3-1e4`x smaller. This single cell dominates the global L2 defect norm,
      which is exactly why `mg_verbose` reports `"defect ratio = 1"` (no
      progress) for dozens of consecutive V-cycles. This reverses the leading
      hypothesis above: the residual is concentrated right at a specific
      location, not diffuse FAS sensitivity. Both worst cells sit at the
      intersection of the coarse/fine interface (the `6.2`-vs-6.4 axis) *and*
      a reflecting-boundary-adjacent corner (the other two axes are `0.2`/`0.6`,
      i.e. the octet's own corner nearest a reflecting wall) -- a real,
      localized defect, still not root-caused to a specific line.
    - **Code paths checked this pass and NOT the bug** (static reading, no
      compiler): `InitializeOctets`'s per-axis neighbor classification correctly
      marks any edge/corner direction touching a non-periodic domain boundary as
      `{-2,-2}` (physical boundary) before ever considering a coarse-neighbor
      lookup, so there's no periodic-wrap contamination at these
      reflecting-adjacent corners. `ApplyPhysicalBoundariesOctet`'s reflect
      handling uses the correct `sign=+1` (zerograd) for CFC's reflecting faces.
      `SetOctetBoundaryFromCoarser`/`ProlongateOctetBoundaries` (root->octet u_
      fill) and `ProlongateFCMG` (the regular per-MeshBlock coarse-fine ghost
      fill, shared with gravity) are both refreshed every red-black half-sweep,
      not stale. One real-but-likely-inconsequential quirk found:
      `ApplyPhysicalBoundariesOctet`'s sequential x1->x2->x3 reflection order can
      leave stale data in the octet's own *diagonal/corner* ghost cells (x1's
      pass reads edge-ghost rows before x2's pass has refreshed them this call)
      -- but the 7-point stencil never reads diagonal ghost cells directly, so
      this doesn't explain the observed stall by itself.
    - **Follow-up diagnostic added and run**: `DebugReportDefectByLevel` also
      dumps, at the single worst REFINED cell only, its own `U`/`Coeff(0)`/
      `Coeff(1)` and its 6 face-neighbor `U` values. Result: hand-computing the
      discrete Laplacian from the printed neighbor values (`sum_neighbors -
      6*U ~ -0.0075`, `/dx^2 ~ -0.047`) against the physical RHS
      (`2*pi*Utilde/psi ~ -0.0028`, `Ahat^2=0` since this is a static
      configuration) reproduces the printed defect almost exactly -- so the
      reported residual is numerically self-consistent, not a diagnostic
      artifact, and none of the raw neighbor `U` values are anomalous (no
      NaN/zero/wrong sign; the one place a neighbor exactly equals the cell's
      own `U` is the *correct* zero-gradient reflecting-BC behavior for a
      1-cell-wide multigrid ghost, not a bug).
    - **Decisive follow-up experiment**: temporarily bumped the hardcoded
      40-iteration `SolveIterative` cap (`multigrid_driver.cpp`, then reverted)
      to 400 with `mg_verbose=2` (per-iteration defect printing) to test
      "genuinely stuck" vs. "just needs more iterations." Result: the L2 defect
      norm (and the worst-REFINED-cell defect) drops for the first ~5
      iterations, then becomes **bit-for-bit identical for the remaining ~380
      iterations** (`0.00204167` repeated exactly). This rules out "slow but
      real convergence" -- Newton-Gauss-Seidel has reached an exact numerical
      fixed point that is *not* a root of the discrete equation at this cell
      (the cell's own defect, computed by the same `ConformalFactorRHS`/
      `ConformalFactorLap` the smoother itself uses, would have to be ~0 if the
      smoother's own Newton step there were genuinely converged). This is now
      strong evidence of a real bug -- most likely something that freezes this
      specific cell's update (or its ghost input) after the first few
      iterations, rather than a "stiff but working" relaxation.
    - **Decisive analytic-ghost test (2026-07-21, job 248069,
      `cfc_tov_amr_analytic_test`)**: added `DebugAnalyticResidualTest()` (same
      files), gated on `mg_debug_analytic_residual_test`, run *before* `SolveMG`
      so no smoother iterations occur in either measurement. Seeds `delta_psi`
      at every cell -- including every ghost cell at the refinement boundary --
      from the exact analytic isotropic TOV solution
      (`tov::TOVStar::ConstructTOV`/`PolytropeEOS`, the same machinery
      `dyngr_tov.cpp` itself uses), measures the residual, then overwrites just
      the ghost cells with one real ghost-communication round (`FillCoarseBoundary`
      -> `StartReceive` -> `SendBoundary` -> `RecvBoundary` -> `PhysicalBoundary`
      -> `ProlongateFCBoundary`, called directly -- the same sequence
      `SetMGTaskListToFiner`'s flag==2 block uses) and measures again. Result:
      worst REFINED-block defect goes from `~0.00036` (pure analytic, matching
      ordinary O(dx^2) truncation error at dx=0.4) to `~0.0028-0.0038` (**~8-10x
      larger**) after just one ghost-comm round -- with zero smoother
      involvement in either number. ROOT-block defects barely move (`~1e-7` to
      `~1e-6` in both cases). The worst point after ghost-comm lands in the same
      "coarse-fine interface (x1~6.2), small x2/x3" neighborhood the actual
      stuck Newton solve pins to. **This conclusively shifts the root cause from
      the nonlinear relaxation to the ghost-communication/coarse-fine
      prolongation machinery itself** (`FillCoarseMG`/`ProlongateFCMG` in
      `multigrid_bvals.cpp`, or the ghost depth `ngh_` used there) -- the
      smoother was never able to converge because it's being fed bad ghost data
      every iteration, not a nonlinear-solver-specific issue. Not yet narrowed
      to a specific line within that ghost-fill code -- next step would be
      comparing, cell by cell, the analytic ghost value against what
      `ProlongateFCMG`'s flux-conserving formula actually produces there.
    - **Root cause found and FIXED (2026-07-21, `multigrid_bvals.cpp`,
      `ProlongateFCMG`)**: `fc_childx_`/`fc_childy_`/`fc_childz_` (each fine
      MeshBlock's octant parity relative to its refined parent) are computed and
      loaded at the top of `ProlongateFCMG`'s kernel but were never used in the
      coarse-face prolongation branches. Traced via `ComputePerLevelIndices`'
      `compute_recv` `icoar` logic: for a face message, a "low" child (parity
      bit 0) receives one extra transverse overlap cell on the HIGH edge of its
      window; a "high" child (parity bit 1) receives it on the LOW edge instead
      -- the reverse. But the transverse-gradient clamp bounds in all three face
      branches (`ox1!=0`/`ox2!=0`/`ox3!=0`) were hardcoded to the low-child
      layout regardless of parity, so high children never read their own valid
      low-edge overlap cell (using a degraded one-sided difference there
      instead). Confirmed directly (not just by index arithmetic) with a
      ghost-vs-analytic dump (`DebugDumpCoarseBuf`, added this session, gated
      inside `DebugAnalyticResidualTest`): for a high child (`gid=3`,
      `child_y=1`), the low-edge fine-cell pair showed diffs of `4-5e-4` against
      the true analytic ghost value, 5-15x the `3-8e-5` seen everywhere else.
      **First fix attempt made things worse**: narrowing the HIGH bound too
      (mirroring the low-bound shift) broke a previously-fine high edge --
      confirmed by the SAME ghost-vs-analytic check moving its error to the
      other corner (`fi=7,8`, diff `~4e-4`) after that attempt. The high edge
      apparently already reads something usable regardless of child parity
      (likely `FillCoarseMG`'s own leftover self-restriction at that exact
      slot, which is close enough given how smooth the field is) and must not
      be clamped away. **Final fix**: only the LOW bound is now child-parity-
      dependent (`ngh_l-1` for a high child in that axis, else `ngh_l`
      unchanged); the HIGH bound is left unconditionally at `ngh_l+half` in all
      three branches, matching the original code. Verified via the same
      ghost-vs-analytic dump: both edges now show uniformly small errors
      (`~3e-5` to `1e-4`) for high children on both the `+x1` and `+x2`
      directions. The analytic-test worst-REFINED-defect (after one ghost-comm
      round, no smoother) dropped from `~0.0028-0.0038` to `~0.0009` -- close to
      the `~0.00036` pure-truncation baseline.
    - **This fix is real and confirmed, but does NOT explain the REAL solve's
      non-convergence** -- rerunning the actual iterated solve
      (`cfc_tov_amr_fix_verify_realsolve`) still shows "Failed to converge"
      every stage, defect stuck at `~0.0022` (essentially unchanged from
      `~0.00204` before the fix). The worst-defect location/value from
      `DebugReportDefectByLevel` is *also* unchanged: still exactly `gid=1`
      (a LOW child, `child_y=0,child_z=0`) at `(6.2,0.6,0.2)`, same stencil
      values to 4-5 significant figures as every prior run, before or after
      this fix. This makes sense in hindsight: low children's clamp bounds were
      never touched by the fix (they already matched the confirmed-valid
      window), so nothing about their ghost fill could have changed. Directly
      checked with the SAME ghost-vs-analytic dump, extended to cover low
      children too: `gid=1`'s own `+x1` ghost fill is uniformly accurate
      (`~1-1.4e-4` diff, no outlier) both before and after the fix -- so the
      stuck residual at `gid=1` is **not** a bad ghost value at all. The actual
      cause of the low-child plateau (and hence of the real solve's
      non-convergence) is still open -- candidates not yet checked: the
      Newton-Gauss-Seidel smoother itself at this specific cell, the RHS/coeff_
      values it reads, or a different piece of the multigrid machinery (e.g.
      octet-level code in `multigrid_driver.cpp`, though that shouldn't be in
      play for a plain MeshBlock-to-MeshBlock coarse-fine pair at `nreflevel_=1`
      as used here). The `ProlongateFCMG` fix should be kept regardless (it is
      a genuine, verified correctness improvement for high children), but the
      investigation is not finished.
    - **Second root cause found and FIXED (2026-07-21, `multigrid_driver.cpp`'s
      `MultigridDriver::RestrictCoeffOctets()`)**: this function is meant to be
      the one-time (`coeff_` is static for the whole solve) analog of the
      generic, per-V-cycle-sweep `RestrictOctets()` -- but `RestrictOctets()`
      actually has *two* branches (fine-octet-to-coarser-octet, and, for the
      coarsest octet level, an explicit "octets to root grid" branch that
      writes `root_src_h`/`root_u_h` directly). `RestrictCoeffOctets()` only
      ever mirrored the first branch (`for l = nreflevel_-1; l >= 1; --l`);
      nothing mirrored the second. At `nreflevel_=1` (this test's geometry)
      that loop condition (`0 >= 1`) is never true, so the function was a
      **complete no-op**: the root-level cell under the refined patch was never
      written by anything (`TransferCoeffToRoot()` only writes `root_coeff_h`
      directly for blocks *at* root level, and the octet's own `Coeff()` for
      refined blocks -- never both), and stayed at `coeff_`'s post-construction
      default (`0.0`) for the entire solve. Confirmed empirically (not just by
      code reading) with a new temporary diagnostic,
      `MGCFCConformalFactorDriver::DebugDumpRootCoeffUnderOctet()`
      (`mg_cfc_conformal_factor.{hpp,cpp}`, gated on the existing
      `mg_debug_analytic_residual_test` input, called from `Solve()` right
      after `RestrictCoeffOctets()`): before the fix, `mgroot_`'s actual `Utilde`
      at the cell under the refined octet read back as exactly `0.0`, while
      `RestrictOneCoeff`'s volume-averaged expectation from that same octet's
      own (independently correct) `Coeff()` was `~7.4e-4` -- a real, non-trivial
      value, not some edge-case zero. **Fix**: added the missing "octet level 0
      -> root grid" branch to `RestrictCoeffOctets()`, using `Multigrid`'s
      already-public `CoeffAtLevel()` accessor (no new friend/downcast needed).
      This also required a small ordering fix: `TransferCoeffToRoot()` (both
      `MGCFCConformalFactorDriver` and `MGCFCLapseDriver`'s copies) used to call
      `mgc_root->RestrictCoefficients()` (propagates `mgroot_`'s own finest
      level down through its coarser internal levels) at its own end -- but
      that ran *before* `RestrictCoeffOctets()` even executed, so it was always
      one step too early for any refined patch's root cell. Moved that call out
      of `TransferCoeffToRoot()` into `Solve()`, right after
      `RestrictCoeffOctets()`, in both drivers. Verified via rerun: `actual`
      now matches `expected` exactly (`diff=0`) for every rank.
    - **This fix is also real and confirmed, but likewise does NOT explain the
      REAL solve's non-convergence**: rerunning the actual iterated solve with
      both fixes in place still shows `"Failed to converge"` every stage, at
      essentially the same defect (`0.00204301`, vs. `~0.00204`/`~0.0022`
      before either fix) and the same worst-REFINED-cell location (`gid=1`,
      `(6.2,0.6,0.6)`, value `~9.04e-4` -- consistent with every prior run to
      3-4 significant figures). The `coeff_` restriction gap was real, but its
      effect on this particular residual is evidently negligible compared to
      whatever actually dominates it.
    - **Re-ran the analytic-ghost-vs-defect comparison with both fixes applied**
      (per explicit user request, to directly check whether prolongation/
      restriction at the refinement boundary is now correct): with **no**
      smoother involved, seeding `delta_psi` everywhere from the exact analytic
      TOV solution and measuring the defect before vs. after one real MG
      ghost-communication round (`FillCoarseBoundary` -> ... ->
      `ProlongateFCBoundary`, i.e. the real coarse-fine `ProlongateFCMG` path):
      worst REFINED-block defect goes from `~3.6e-4` (analytic ghosts, pure
      `O(dx^2)` truncation baseline, at the high children `gid=3`/`gid=7`) to
      `~9.04e-4` (**~2.5x larger**) at `gid=1`/`gid=4` (low children) after the
      real ghost-comm round -- essentially unchanged from every pre-fix run of
      this same test. The worst ROOT-block defect *also* increases post-ghost-
      comm (`~1.8e-4` -> `~4.5e-4`, at `gid=8`, the coarse neighbor bordering
      the refined patch), so the degradation isn't confined to the fine side's
      own prolongation formula. Directly comparing ghost *values* (not defect)
      against the analytic solution at `gid=1`/`gid=3`/`gid=5`/`gid=7`'s `+x1`
      (and `gid=3`'s `+x2`) faces: all read back accurate to `~4e-5`-`1.4e-4`,
      matching every previous measurement, with no outlier cell. **So the
      ghost-fill machinery is correctly reproducing point *values* to
      `~1e-4`, but the discrete *residual* (which depends on second-derivative/
      curvature information built from those ghost values via the 7-point
      Laplacian stencil) is still ~2.5x worse right at the coarse-fine interface
      than the pure-truncation baseline, on both sides of the interface.** This
      rules out a simple "wrong ghost value" bug (both of the two fixes so far
      were exactly that class, and both are now closed) -- what's left is either
      a genuine, inherent discretization effect of a 2:1 resolution jump feeding
      into a second-derivative stencil (not obviously a "bug" to fix), or a
      more subtle issue in exactly how the prolongation formula's gradient/
      curvature terms are constructed that a pointwise value check can't catch.
      Not yet root-caused -- still open.
    - **Reflecting-BC control test (2026-07-21, `cfc_tov_amr_noreflect_test`)**:
      per explicit user request, ruled out whether the octant-reduced domain's
      inner `reflect` BC (`ix1_bc=ix2_bc=ix3_bc=reflect` in every prior run of
      this investigation) is a necessary ingredient in the observed defect
      amplification. Built a control input doubling the domain to the full
      range (`x1min=-25.6` instead of `0.0`, `nx1` doubled `32->64` to keep the
      same `dx=0.8`, same for x2/x3) with **all 6 faces** set to the original's
      non-reflecting outer BC (`diode`) -- no `reflect` anywhere -- and
      recentered the refined region on the origin (`[-6.4,6.4]` instead of
      `[0,6.4]` on all 3 axes), which spans exactly 2 root MeshBlocks per axis
      (8 root blocks total, each still refined to 8 children -- 64 children
      total) and reproduces the original single-octant's "one coarse block
      refined to 2x2x2 children" structure 8 times over, once per octant
      around the origin, with every refined child now bordering only same-level
      siblings or genuine coarse-fine interfaces -- never a reflecting wall.
      Re-ran the same no-smoother analytic-residual test: worst REFINED |defect|
      goes from `~3.77e-4` (analytic ghosts, matching the octant test's
      `~3.6e-4` baseline almost exactly -- same `dx`, same star, same class of
      truncation error) to **`~2.03e-3`** after one real ghost-comm round --
      **~5.4x the baseline, and ~2.2x *larger* than the octant-reduced test's
      post-ghost-comm value (`~9.04e-4`)**, not smaller. This pattern repeats at
      comparable magnitude (`~9e-4` to `~2e-3`) simultaneously at *several*
      different octants' coarse-fine corners in the same run (e.g. `gid=56` at
      `(-12.6,-12.6,-12.6)`, `gid=88` at `(-12.6,-12.6,-6.2)`, `gid=296` at
      `(-12.6,7,-12.6)`, and others), none of which border any reflecting
      boundary in this domain at all. **Conclusion: the reflecting BC is
      definitively ruled out** as a necessary or contributing cause -- removing
      it entirely did not shrink the effect, it grew somewhat larger. The
      amplification is an intrinsic property of the coarse-fine ghost-fill/
      prolongation machinery at a 2:1 refinement jump, independent of what
      boundary condition (if any) sits elsewhere in the domain. (Side note,
      not the focus of this test: this particular run's *real* iterated solve
      actually converged with no `"Failed to converge"` messages at all, unlike
      every octant-reduced run -- consistent with `mg_threshold`'s convergence
      check being an L2-style norm over many more cells here, not a per-cell
      max, so one still-bad corner cell at `~2e-3` can hide under a
      norm-based threshold even though the same localized residual floor this
      whole item is chasing is still plainly present at the cell level.)
    - **Curvature/gradient-reconstruction investigation (2026-07-21)**: per
      explicit user request, investigated whether `ProlongateFCMG`'s quadratic
      (value + transverse-gradient) coarse-face formula has a further, still-
      unfound bug beyond the two already fixed, or whether the residual
      amplification at the interface is an inherent discretization property.
      **First attempt used the wrong metric**: comparing the domain-wide "worst
      REFINED cell" defect between this test's `dx=0.8/0.4` and a pre-existing
      `dx=0.4/0.2` companion input (`cfc_tov_amr_ghosttest_2x_v2.athinput`) showed
      the *pre-ghost-comm* (pure analytic, zero communication) baseline
      **increasing** at finer resolution (`~3.6e-4` -> `~1.33e-3`) instead of
      shrinking `~4x` like ordinary `O(dx^2)` truncation -- the opposite of what
      a well-behaved discretization should do. Root cause: `DebugReportWorst
      Defect`'s "worst cell in the whole refined patch" search isn't a fixed
      physical location -- the TOV star's own density profile (rhoc/kappa) has
      reduced smoothness (a kink or rapid falloff) somewhere near the stellar
      surface, and doubling resolution moved *which* cell is domain-wide-worst
      to a different physical point entirely (confirmed: the reported gid/
      position changed between the two resolutions). Comparing "worst-of-N-
      samples" between two different N's (and, worse, from two different
      underlying features) can't isolate the ghost-fill-specific error at all.
    - **Fix: added a resolution-fair, fixed-location metric.** New temporary
      diagnostic `DebugDumpInterfaceDefect(label)` (`mg_cfc_conformal_factor.
      {hpp,cpp}`, called twice from `DebugAnalyticResidualTest`, before and
      after the real ghost-comm round) restricts the search to just the single
      layer of fine cells immediately adjacent to a real coarse-fine `+x1`
      interface (same block-selection logic as `DebugDumpCoarseBuf`) and
      reports **both** the max and the **RMS** over that fixed layer -- RMS
      because even *max-over-the-fixed-layer* still grows with the number of
      transverse cells in the layer (extreme-value statistics: doubling
      resolution quadruples the transverse cell count, so the max of an
      unchanged underlying per-cell error distribution would grow on its own),
      confirmed as a real, separate confound this pass, distinct from the
      density-profile issue above. RMS is per-point and doesn't share that bias.
    - **Result (RMS, resolution-fair)**: at `dx=0.8/0.4` (`gid=3`): before
      `9.30e-5`, after `5.83e-4` (`6.3x`); at `dx=0.4/0.2` (`gid=3`): before
      `2.73e-4`, after `6.35e-4` (`2.3x`). Same pattern at `gid=5`: `1.14e-4`
      ->`3.18e-4` (`2.8x`) at `dx=0.8/0.4`, `3.20e-4`->`4.17e-4` (`1.3x`) at
      `dx=0.4/0.2`. Two consistent signals, both pointing the same direction:
      (1) the *relative* extra defect the ghost-comm round adds (the
      after/before ratio) **shrinks toward 1** as resolution increases (not
      flat, not growing) -- the signature of a genuine, convergent discretization
      effect, not a fixed indexing/formula bug (a real bug's effect size doesn't
      generally track resolution this cleanly); (2) the *absolute* extra defect
      the ghost-comm round adds (after minus before) also shrinks under 2x
      resolution, but only by `~1.4x`-`2.1x`, not the `~4x` the bulk interior's
      `O(dx^2)` truncation gets -- i.e. *some*where between `O(dx)` and
      `O(dx^2)`, roughly one truncation order lower than the smooth interior.
    - **Conclusion**: this is the textbook, generally-accepted characteristic of
      a 2:1 AMR refinement jump for a second-derivative (Laplacian) elliptic
      operator -- matching a coarse cell's *value and gradient* (what
      `ProlongateFCMG`'s quadratic formula does) is not sufficient to keep the
      *curvature* (what the 7-point Laplacian stencil actually needs) accurate
      to the same order as the smooth interior; achieving that would require a
      higher-order (matching second derivatives too) reconstruction. This is
      exactly why AMR multigrid codes conventionally compensate with extra
      smoothing sweeps near refinement boundaries -- already this code's
      existing workaround (`mg_npresmooth`/`mg_npostsmooth=3`, item 12's own
      note). **No further bug found in the curvature/gradient reconstruction
      itself** -- the two real bugs already fixed this session (`ProlongateFCMG`
      child-parity clamp, `RestrictCoeffOctets`'s missing octet-to-root step)
      remain the correctness fixes this investigation produced; the remaining
      residual floor is now understood to be an inherent, convergent (if
      imperfect) discretization limitation rather than a third hidden bug. A
      genuine improvement (if ever wanted) would mean a higher-order/curvature-
      matching prolongation formula at the coarse-fine boundary, a real (if
      involved) numerical-methods project, not a bug fix -- noted here as a
      possible future direction, not undertaken in this investigation.
    - **Higher-resolution/deeper-AMR check (2026-07-21, `cfc_stability_v4`'s
      production input, adapted for a 1-cycle diagnostic run)**: per explicit
      user request, reran the analytic-residual test on a much more realistic
      setup than any prior test in this item -- full domain (`[-160,160]^3`,
      `dx=1.6` root), **5** refinement levels (not 1), innermost refined region
      `[-10,10]^3` down to `dx=0.1`, 32 MPI ranks. Extended
      `DebugAnalyticResidualTest` with a new phase (needs `Driver*`, now
      forwarded from `Solve()`): after the existing before/after-ghost-comm
      defect measurements, runs exactly **one** real `SolveVCycle` (normal
      `npresmooth_`/`npostsmooth_` counts) from the analytically-seeded state,
      then reports the defect again *and* (new function,
      `DebugReportWorstSolutionError`, mirrors `DebugReportWorstDefect`'s root/
      refined classification but compares `u` directly against the analytic
      value instead of the residual) how far the actual **solution** has moved
      from the exact answer.
      - Global-max results (across all 32 ranks): pure analytic baseline (no
        comm, no smoother) worst REFINED defect `8.58e-3`; after one real
        ghost-comm round `1.72e-2` (`~2x` worse, same qualitative pattern as
        every smaller test in this item); after just **one** V-cycle
        `1.24e-3` -- **~14x smaller than the post-ghost-comm value, and ~7x
        smaller than even the pure-truncation baseline**. The Newton-GS
        smoother is clearly effective here, even in a single sweep.
      - Solution-value error after that one V-cycle: worst `|u-analytic|` =
        `2.73e-6` (ROOT) / `4.35e-5` (REFINED) -- excellent absolute accuracy
        (recall `delta_psi` itself is `O(0.01-1)` for this star, so this is a
        relative error of order `1e-4`-`1e-5` after a single iteration).
      - The real iterated solve for this same stage (full `SolveMG`, not just
        the one diagnostic V-cycle) converged to a final worst |defect| of
        `~1e-9`-`1e-10` (both ROOT and REFINED) by the end, with only 3 total
        `"Failed to converge"`/`"Slow convergence"` messages across every
        multigrid solve this stage runs (consistent with early transients
        before convergence, not a stuck residual floor).
      - **Conclusion**: at this much higher resolution and deeper (5-level)
        AMR -- closer to how this solver would actually be used in production
        -- the residual-floor behavior characterized above (items 21's earlier
        entries) does not appear to be a practical problem: the real solve
        converges to near machine precision, and even a single V-cycle from a
        cold analytic seed reproduces the exact solution to `~1e-5` relative
        accuracy. This doesn't contradict the earlier findings (the coarse-fine
        interface truncation-order reduction is still real, and still visible
        in the `~2x` post-ghost-comm defect bump above) -- it suggests that
        reduction's *practical* impact shrinks enough at realistic production
        resolution/AMR depth to not matter, consistent with it being an
        inherent, convergent discretization effect rather than a fixed-size bug.
      - One data point did not print (`DebugReportWorstSolutionError`'s "AFTER
        ghost-comm (no smoother yet)" call, called right before the one
        V-cycle) -- likely lost to stdout interleaving from 32 concurrent MPI
        ranks each producing a very large volume of other diagnostic output
        around the same point, not a logic error (the identically-implemented
        call immediately afterward, "AFTER one V-cycle", printed correctly on
        every rank). Not chased further since the data obtained already answers
        the question this test was run for.
22. **TODO, open — "Migration of an unstable neutron star" test (Gmunu sec.
    3.3.5) not yet running cleanly.** New input file
    `inputs/dyn_grmhd/cfc_tov_migration.athinput` (untracked in git as of this
    writing) attempts to reproduce this test: polytropic `K=100`, `Gamma=2`
    star on the *unstable* branch, `rhoc=8.00e-3` (the paper's "SU" model,
    Cordero-Carrion et al. 2009) -- `dyngr_tov.cpp` already supports an
    arbitrary `rhoc` with no code changes needed for the physics itself. The
    paper's own setup is 2D axisymmetric cylindrical (`R,z`); this CFC port has
    no cylindrical/axisymmetric support at all (confirmed: `dyngr_tov.cpp`'s
    whole pgen and every `cfc_reconstruct.cpp` kernel assume flat 3D Cartesian
    `x1v/x2v/x3v`), so the input reproduces it in full 3D Cartesian with octant
    symmetry instead, matching the paper's root/finest resolution to the digit
    where the coordinate change allows.
    - **Root cause of every failure mode below: this star is meaningfully more
      compact/relativistic (`2M/R~0.34`, `Mass=1.44729`, `R_schw=5.83677`) than
      any star previously exercised anywhere in this module** (prior stars sit
      around `2M/R~0.29`). Two separate numerical consequences, found in this
      order:
    - **(1) FIXED, in code but NOT YET COMMITTED (`src/cfc/cfc.hpp`/`cfc.cpp`,
      modified, uncommitted as of this writing)**: `CFC::InitializeMetric()`'s
      one-time `t=0` Picard iteration (item 11) diverged outright for this star
      under its default unrelaxed update (`max|delta psi|` growing from `0.011`
      at iteration 1 to `2.1e52` by iteration 18, then NaN). Added a new
      `<cfc>` `init_omega` under-relaxation parameter (new private member
      `cfc_init_omega_`, default `1.0` -- byte-identical behavior for every
      existing test/input, since `(1-1)*old + 1*new = new`): after
      `SolveConformalFactor()`'s unrelaxed solve, blend
      `delta_psi = (1-omega)*psi_old + omega*delta_psi_new` (interior-only, matching
      `AssembleConformalMetric`'s own no-ghost-dependency implementation) and
      re-run `AssembleConformalMetric()` so `padm->adm.g_dd`/`psi4` reflect the
      relaxed value before the next iteration's `PrimToConInit`. At
      `init_omega=0.5` this star converges cleanly (geometric decay, typically
      ratio `~0.5`-`0.86`/iteration depending on the AMR configuration below, to
      well under the default `init_tol=1e-10`).
    - **(2) STILL OPEN**: even with (1) fixed, the routine *per-cycle* metric
      solves (`psi`/lapse, every RK stage of every cycle -- not just the one-time
      initialization) land at or above the default `mg_threshold=1e-10`,
      tripping `MultigridDriver::SolveIterative`'s existing safety valve (any
      *single* non-converged solve anywhere sets `pdriver->nlim =
      pmy_mesh_->ncycle`, force-truncating the whole run -- this is
      `multigrid_driver.cpp:821-828`, shared/unmodified code, not new). Diagnosed
      with `mg_debug_defect_by_level` (item 12's existing diagnostic, reused
      unmodified): the plateau is pinned exactly at the edges/corners of
      `refined_region1`, where the reported defect (`~0.01`-`0.04`) tracks
      `delta_psi`'s own local value there (consistent with `~M/(2r)`, Gmunu
      eq. 77) rather than shrinking toward zero -- i.e. the same 2:1-jump
      truncation-order effect item 21 already characterized as an inherent,
      convergent (not buggy) discretization property, just landing somewhere
      the field's curvature is still significant for this specific box
      placement, unlike item 21's own higher-resolution check where it turned
      out not to matter in practice.
      - `R=20` (root MeshBlocks are 30-wide, so this snaps to `R=22.5`,
        confined to 1 of 8 root blocks, 309 total MeshBlocks): stable, zero
        NaN, but plateau (`~5e-4`-`2e-3`) trips the clamp after 1 cycle even
        with `mg_threshold=5e-4`/`mg_npresmooth=mg_npostsmooth=3` (item 12's
        own established workaround).
      - `R=40` (spills into all 8 root blocks since `40 > 30`, 2024 total
        MeshBlocks -- a genuinely different, much larger AMR footprint, not
        just a bigger box): **fixes `InitializeMetric` completely** (clean
        `~0.5`/iteration decay to `7e-11`, no cap hit at all) but introduces a
        **new, more severe, uninvestigated failure**: early in cycle 1, the
        psi defect explodes `6.5e-4 -> 2.5e13 -> 1.7e22 -> 4.7e27 -> 2.5e30` in
        a handful of solves, traced to a matter-source coefficient
        (`Coeff0(Utilde)`) already at `~8.7e9` (should be `O(1e-2)` or
        smaller) by that point -- i.e. the *hydro* state went bad first,
        likely at one of the many new coarse-fine interfaces this much larger
        AMR footprint introduces, and that then detonates psi's nonlinear
        solve. **Not root-caused** -- flagged here as a real, separate
        robustness question worth its own investigation if AMR footprints this
        large (spanning multiple root MeshBlocks) are needed again, CFC or not.
      - `R=30` (exactly matches the root-MeshBlock boundary, stays confined to
        1 of 8 root blocks like `R=20`, 701 total MeshBlocks -- the safe
        middle ground): stable, zero NaN, modestly better plateau
        (`~5e-4`-`1.5e-3`, worst seen `~2.6e-3`) than `R=20`, but still trips
        the clamp.
      - **Current best configuration**: `R=30` + `mg_threshold=3e-3` (loosened
        from the `5e-4` tried first) + `mg_npresmooth=mg_npostsmooth=3` +
        `init_omega=0.5` runs **5 clean cycles** (up from dying after cycle 1),
        zero NaN throughout, `InitializeMetric` converging fully (no cap hit).
        Trips the same clamp again at cycle 4->5 (`defect=7.1e-3`), notably
        *higher* than the `t=0` baseline (`~2.6e-3`) -- expected in isolation,
        since this is an *unstable* star that's actively beginning to migrate/
        oscillate by this point, so per-cycle solve difficulty appears to rise
        somewhat as the dynamics develop, not just sit at a fixed floor.
    - **Not yet decided**: whether to keep loosening `mg_threshold` (next
      candidate discussed: `1e-2`) to give headroom for that rising per-cycle
      difficulty and push further into the run, or to investigate the
      per-cycle plateau/the `R=40` mesh-explosion divergence more fundamentally
      first. Paused here per explicit user request (testing a different
      problem next) -- pick this back up by rerunning
      `cfc_tov_migration.athinput` from `/sakura/ptmp/tlam/athenak_run/
      cfc_tov_migration_test/` (parfile/sub.sh already staged there) with
      `mg_threshold` adjusted, or by digging into either open sub-question
      above.
    - **The "different problem next" this was paused for is item 23** (BU8
      rotating-star stability test) -- which hit its own, related-but-distinct
      AMR-refinement-boundary divergence, resolved there by raising resolution
      and dropping an equatorial reflecting BC. This migration test's own two
      open sub-questions above are still unresolved and untouched.
    - **Note (item 23, 2026-07-23)**: BU8 later found `mg_threshold=3e-3`
      (inherited from this item's own conclusion) too loose for psi/
      alpha_psi's *accuracy* on a rotating star specifically (not a robustness/
      convergence-failure concern like this item's own migration-test context)
      -- BU8's production input now uses `1e-10` instead. This is a different
      test's own tuning, not a correction to this item's conclusion for the
      migration test, whose own robustness tradeoff (loosen threshold to avoid
      hitting the divergence clamp) is still the relevant one there.
    - **Ruled out (2026-07-23): the `LoadPoissonSource` sign fix (item 23) does
      NOT fix this item's per-cycle clamp trip.** BU8's investigation also
      found (and fixed) a real sign error in `MGCFCVectorPoissonDriver::
      LoadPoissonSource` (`fac` was `+1.0`, should be `-1.0` per the same
      stencil-sign convention `gravity::MGGravityDriver` establishes) -- since
      this migration-test star also has nonzero velocity (hence nonzero
      `S_i`/`X^i`, not just BU8's rotation), it was worth checking whether this
      same bug was contributing to this item's own still-open clamp-trip
      problem. Rebuilt `build_cfc` (the TOV pgen's own build dir, separate from
      BU8's `build_cfc_xns`) with the fix and reran the exact "current best
      configuration" (`R=30`, `mg_threshold=mg_poisson_threshold=3e-3`,
      `mg_npresmooth=mg_npostsmooth=3`, `init_omega=0.5` -- `mg_poisson_
      threshold` explicitly set to match the old shared-key value, isolating
      just the sign fix as the one change) for a quick capped (`nlim=20`) test
      (job 248484). **Result: no improvement, if anything very slightly
      worse** -- `FATAL ERROR ... Failed to converge` now trips at the cycle
      3->4 transition (`defect=6.03e-3`) instead of cycle 4->5
      (`defect=7.11e-3` previously) -- same order of magnitude above
      threshold either way. Consistent with this item's own existing diagnosis
      (`mg_debug_defect_by_level`: the plateau sits at `refined_region1`'s
      edge/corner and tracks `delta_psi`'s own local `~M/(2r)` value there,
      i.e. an inherent 2:1-refinement-jump truncation-order effect per item
      21, not a matter-source sign/normalization bug) -- the sign fix is a
      real, independent bugfix (matters for BU8's `beta^i`/frame-dragging
      sign) but is not relevant to this item's own open problem. This item's
      own two open sub-questions (further loosen `mg_threshold`, or
      investigate the plateau/`R=40` divergence more fundamentally) remain
      exactly as they were.
    - **Also ruled out (2026-07-23): tightening `mg_threshold`/
      `mg_poisson_threshold` (BU8's other fix) to `1e-8` makes this item's
      problem WORSE, not better.** Same `R=30`/`init_omega=0.5`/`mg_npresmooth=
      mg_npostsmooth=3` configuration, `mg_threshold=mg_poisson_threshold=
      1.0e-8` (from `3.0e-3`), quick capped test (job 248548, 4 nodes -- the
      16-node BU8 production job was using most of the account's `136`-node
      QOS cap at the time, see below). **Result**: during `t=0`
      `InitializeMetric()`, every single inner psi V-cycle solve now "fails to
      converge" (the per-solve defect plateaus around `5.1e-4`, essentially
      unchanged regardless of the target threshold -- confirms this is a real
      floor, not merely "hasn't finished iterating yet"), though the *outer*
      Picard loop's own `dpsi` still decreased smoothly throughout (reaching
      `5e-6` by iteration 49, hitting the `init_iter_max=50` cap). Once
      evolution started, the per-cycle solve immediately failed at cycle 0's
      very first stage and the run terminated after cycle 1 (`nlim=0`) -- markedly
      *worse* than the `3e-3` baseline's cycle-4/5 trip, not better. This is the
      expected, consistent result given the plateau's already-diagnosed nature
      (an inherent, non-vanishing truncation-order defect at the coarse-fine
      interface, not a tunable convergence-tolerance problem) -- unlike BU8's
      *outer* Picard-loop stall (which genuinely could reach a true fixed
      point, just needed more iterations/no under-relaxation), this item's
      *inner*, per-solve defect has a hard floor that no threshold change can
      cross. This item's own two open sub-questions remain unresolved; no new
      avenue found by either of BU8's two fixes.
    - **Aside, resource note**: this quick test needed to run on 4 nodes
      instead of the usual 16 -- the account's SLURM QOS node cap (`136`) was
      fully saturated by other concurrently-running jobs (BU8's own 16-node
      production run plus several unrelated jobs from other work), so a
      16-node request sat pending (`QOSMaxNodePerUserLimit`) instead of
      starting.
    - **See also item 24 (2026-07-24)**: a *separate*, deeper problem found on
      this same migration-test star -- `CFC::InitializeMetric()`'s own t=0
      metric initialization converges to the wrong `psi`, independent of this
      item's per-cycle dynamical-evolution plateau. The two are related (same
      star, same test) but distinct symptoms/investigations; item 24's own
      control tests (more smoothing, wider refined region, uniform grid with
      no AMR at all) conclusively ruled out this item's own AMR-boundary
      mechanism as the cause of *that* problem, so the two remain open,
      unresolved questions, not the same bug.

23. **RESOLVED (2026-07-23) -- "Stability of a rapidly rotating neutron star"
    test (Gmunu sec. 3.3.2, model "BU8") diverged after ~100 stable cycles, via
    a new (rotation-specific) coefficient blowup at a refined-region CORNER;
    fixed by higher resolution + dropping the equatorial reflect BC.** New
    files: `src/utils/xns/xns_rotator.hpp` (reads/bilinearly interpolates the
    external XNS code's 2D `(r,theta)` tabulated equilibrium) and
    `src/pgen/dyn_grmhd/xns_rotstar.cpp` (the pgen -- sets `padm->adm`/`pmhd->w0`
    initial guess from that table, converted to Cartesian via the flat-space
    identity `V^x=-V^phi*y, V^y=V^phi*x` for both `beta^phi` and `v^phi`; also
    registers `SetADMVariablesToXNS` as the `padm->SetADMVariables` callback,
    confirmed mandatory since `MeshRefinement::AdaptiveMeshRefinement()`
    unconditionally calls it on every regrid for any CFC-only run, independent
    of `is_dynamic`). **2026-07-26 update (item 33)**: this regrid-time call
    is now SKIPPED for CFC runs specifically (`mesh_refinement.cpp`'s Step 11
    gate gained a `pcfc == nullptr` condition) -- `CFC::ReinitializeMetricForAMR`
    replaces its role for CFC, since re-deriving the metric from XNS's static
    t=0 table on every regrid would discard any dynamical evolution since
    then. The registration itself is still needed (t=0/restart setup still
    calls it directly, unaffected by this), just no longer for the regrid
    path described here -- see item 33 for the full story. Both build cleanly
    (`-D PROBLEM=dyn_grmhd/xns_rotstar`).
    Initial data: XNS's bundled `RotBU8UnMagPol2` reference config
    (`RHOINI=1.28e-3, K1=100, GAMMA=2, OMG=2.633e-2`, uniform rotation,
    unmagnetized) -- converged XNS result matches its own reference
    `LogFile.dat` to 10+ significant figures (Komar mass `1.6911876652765`,
    circumferential radius `19.2496` km, axis ratio `0.594306`), run staged at
    `/sakura/ptmp/tlam/XNS_runs/rot_bu8_gr_pol2/`.
    - New input `inputs/dyn_grmhd/cfc_bu8_stability.athinput`: unlike every
      prior `cfc_tov*` input, a rotating star has only axisymmetry-about-z plus
      equatorial (`z->-z`) reflection symmetry, NOT full octant symmetry
      (`v^x=-v^phi*y`/`v^y=v^phi*x` do not obey the usual mirror parity through
      `x=0`/`y=0` individually) -- so `x1`/`x2` span the full `+/-45` range (no
      reflect), only `x3` gets `ix3_bc=reflect`. Root grid `3x3x1` MeshBlocks
      (`16^3` cells each, `dx=1.875`, same finest-`dx=0.234375`/meshblock-size
      convention as `cfc_tov_migration.athinput` -- MeshBlock cell counts must
      be a power of 2, a hard multigrid requirement, confirmed by a `FATAL
      ERROR in Multigrid::Multigrid` hit during initial local testing with a
      non-power-of-2 size). `refined_region1` deliberately confined to the
      single CENTRAL root MeshBlock (which straddles the origin, spanning
      exactly `x1,x2` in `[-15,15]`) -- a rotating star centered at the origin
      can't be confined to a single *corner* root block the way the
      octant-symmetric migration test's star could, but a central block that
      straddles the origin works the same way. `init_omega=0.5`,
      `mg_threshold=3e-3`, `mg_npresmooth=mg_npostsmooth=3`,
      `mg_debug_defect_by_level=true` all set proactively from the start,
      carrying forward item 22's precedent rather than discovering the need
      after a stalled run. 632 total MeshBlocks.
    - **Result**: `CFC::InitializeMetric()` converged cleanly and quickly on
      this genuinely new (rotating, near-mass-shedding) matter distribution --
      geometric decay `max|delta psi|` `0.110 -> 5.1e-11` over 32 iterations,
      no divergence, confirmed both in a local single-rank smoke test and at
      the start of the full cluster run (job 248441, 4 nodes/8 ranks). The
      subsequent evolution then ran **stably for ~100 cycles** (`t=0` to
      `t~29.25`): the `.mhd.hst` `mass` column stayed constant to 10
      significant figures (`0.8990139291...`) the entire time, i.e. this
      rotating equilibrium is being evolved essentially perfectly for a
      substantial stretch -- then, within the next 1-2 cycles (`t~29.9` to
      `t=30.097`, cycle 100->101), it **diverged catastrophically**: the `.hst`
      `mass`/`tot-E` columns jump to `~1.49e92`/`~1.49e82` (obviously
      unphysical, effectively NaN-adjacent), and
      `MultigridDriver::SolveIterative` (the `psi`/conformal-factor solve --
      the only driver `mg_debug_defect_by_level`'s diagnostic is wired into,
      per item 21's scope decision) threw its "Failed to converge" clamp
      repeatedly with the reported defect escalating from `O(1-10)` to
      `O(1e17)-O(1e21)` across consecutive stage-solves within that same
      cycle -- a genuine, fast runaway, not a slow drift.
    - **Diagnostic (`mg_debug_defect_by_level`, reused unmodified)**: the
      worst-defect cells at the moment of blowup cluster tightly at
      `x1,x2 ~ +/-14.88` (just inside the refined-region/root-MeshBlock
      boundary at `+/-15`) **and** `x3 ~ 0.12-0.23` (the first cell or two
      above the equatorial `z=0` reflecting boundary) simultaneously -- i.e.
      right at the CORNERS where `refined_region1`'s horizontal edge meets the
      equatorial-symmetry boundary, not a generic interior or flat-boundary
      location. The specific quantity that blows up first is `Coeff1(Ahat2)`
      (`Ahat^ij Ahat_ij`, the shift/extrinsic-curvature source term in eq. 73)
      -- reaching `~1e39`-`1e42` by the time it's caught -- which is the
      rotation-specific analog of item 22's `Coeff0(Utilde)` blowup signature
      for the `R=40` migration-test case: that star had no rotation/shift, so
      `Ahat^2 = 0` identically there and `Utilde` (matter density) was the
      coefficient that went bad first; this is the first CFC test where
      `Ahat^2` is genuinely large and nonzero, and it is *this* coefficient
      that fails first here.
    - **Not root-caused.** Leading hypothesis (not yet confirmed): the same
      general class of AMR-refinement-boundary robustness issue already
      flagged as open in item 22 (a coarse-fine-interface truncation effect
      feeding back destructively into a nonlinear coefficient under the right
      conditions), but (a) manifesting through the curvature/shift coefficient
      rather than the matter coefficient, since this is the first rotating
      configuration tested, and (b) possibly compounded by occurring at a
      genuine CORNER (two boundary types meeting: the static-refinement edge
      and the equatorial-reflection edge) rather than a single flat boundary
      plane, a combination no prior CFC test (all non-rotating, all either
      unrefined-boundary-only or octant-symmetric) has exercised at once.
      Notably this run's own refined-region footprint was deliberately kept
      confined to a single root MeshBlock (learning directly from item 22's
      `R=40` multi-root-block finding) -- and it still diverged, meaning
      "confined to one root block" alone is not sufficient here; the
      equatorial-boundary corner interaction appears to be a distinct
      contributing factor item 22's own (non-rotating, non-equatorial-only)
      test geometry never encountered.
    - **Not yet decided**: whether to (a) loosen `mg_threshold` further /
      increase smoothing more (the item-22-style band-aid, cheapest to try
      first), (b) enlarge the refined region or move its horizontal edge
      further from the star so the corner sits somewhere the field's curvature
      is smaller (mirrors item 22's own `R=20/30` exploration), (c) rerun with
      AMR entirely disabled (uniform resolution) to confirm whether the
      refinement boundary specifically is required for the blowup or whether
      it's something else entirely, or (d) investigate the corner-specific
      interaction between the static-refinement and equatorial-reflection
      boundaries directly. Paused here pending direction -- rerun from
      `/sakura/ptmp/tlam/athenak_run/cfc_bu8_stability_test/` (parfile/sub.sh
      staged there, points at `build_cfc_xns/src/athena`).
    - **Fix applied and confirmed (job 248446, 2026-07-23), per user
      direction**: two changes together, not isolated individually (so which
      one was necessary/sufficient on its own is not yet known -- see below):
      (1) resolution increased one more AMR level (`num_levels` 4->5,
      `refined_region1 level` 3->4), finest `dx` `0.234375 -> 0.1171875` --
      motivated by the star's rotation being close to mass-shedding and the
      previous finest `dx` giving only `~28.5` cells across the polar radius
      (`R_pole=R_eq*axis_ratio=11.24*0.594306=6.68`), below a judged floor of
      "at least 30 cells across the polar radius" (now `~57`, `~96` across
      `R_eq`); (2) the equatorial `ix3_bc=reflect` boundary was dropped
      entirely -- `x3` now spans the same full `+/-45` range as `x1`/`x2` with
      a plain outer BC, and `refined_region1` was correspondingly widened to
      the symmetric cube `[-15,15]` in all three directions (still confined to
      exactly the one central root MeshBlock, now `3x3x3=27` root blocks
      instead of `3x3x1=9`, `5760` total MeshBlocks, run on 16 nodes/32 ranks).
      **Result**: `CFC::InitializeMetric()` converged cleanly again on the
      finer grid; the full `tlim=100` run (1118 cycles) completed with **zero**
      `FATAL ERROR`s -- no recurrence of the corner defect blowup at all.
      `.mhd.hst` mass conserved to `~1.4e-6` relative drift end-to-end
      (`1.377580224...` at `t=0` to `1.377578328...` at `t=100`); kinetic
      energy terms show only small bounded oscillations, no runaway. Confirms
      the hypothesis above (this was the same class of AMR-refinement-boundary
      robustness issue as item 22, sensitive to both resolution and the
      reflecting-boundary corner) without yet isolating which of the two
      changes was doing the work -- if that distinction matters for a future,
      cheaper-resolution run, it would need an isolated A/B (higher-res with
      reflect still on, or full-domain at the original resolution) to
      separate them; not done here since the user's direction was to apply
      both together and this fully resolved the failure on the first retry.
    - **Second bug found and fixed (2026-07-23, user code review): the
      exterior/atmosphere test used a spherical radius (`r > xns.rmax()`),
      which is wrong for an oblate star, and separately relied on a bare
      density threshold that XNS's own non-vacuum exterior solution can
      exceed.** `xns_rotstar.cpp`'s original atmosphere test was `rho <=
      dfloor` on the *interpolated* Hydroeq.dat density -- but XNS's tabulated
      solution outside the actual stellar surface is not vacuum (it's the
      solver's own smooth continuation, with nonzero, non-monotonic density
      that can sit above a typical evolution-code `dfloor`, e.g. `1e-14`).
      Confirmed visually: the pre-fix density slices showed a distinct
      "halo" -- a ring of matter-like density (`~1e-8`-`1e-10`) extending all
      the way out to `r=20` (the table's own `rmax()`, used at the time as
      the only exterior cutoff), well beyond the star's actual oblate surface
      (`R_eq~11.3`, `R_pole~6.8`-`6.9`) -- and that ring was being assigned a
      nonzero rotational velocity from the table's `v^phi`, i.e. spurious
      "rotating atmosphere" matter that shouldn't exist.
      - **Fix**: `xns::XNSRotator` (`xns_rotator.hpp`) now also reads XNS's
        `Surf.dat` (confirmed format: `NTH` values, no header, the actual
        stellar surface radius `R_surf(theta)` at each of `Grid.dat`'s own
        theta cell centers -- `XNSMAIN.f90:838-840`,
        `WRITE(13,*) R(WSURF(IX)+1)`) into a new `rsurf_` array, exposed via a
        new `template<Loc> Real SurfaceRadius(Real theta) const` method
        (bilinear-interpolated in theta only, since `Surf.dat` is 1D). The
        theta-bracketing index logic (previously inlined in `Interpolate()`)
        was factored into a shared private `ThetaBracket()` helper used by
        both methods, to avoid duplicating it.
      - `xns_rotstar.cpp`'s `XNSInterpToADMAndPrim` now computes
        `r_surf = xns_star.SurfaceRadius(theta)` and uses `atmosphere = (r >
        r_surf)` -- a physically correct, shape-aware test -- instead of the
        density threshold. Inside that radius, `rho`/`p` are still floored as
        a safety net (`fmax(rho,dfloor)`), but that's no longer what decides
        atmosphere-vs-star. The `r > xns.rmax()` branch is unchanged and still
        needed (points genuinely outside the table's own solved domain, where
        there is no data at all regardless of the star's shape).
      - **Simplified input**: `<problem>` now takes one `id_dir` parameter
        (the directory containing `Grid.dat`/`Hydroeq.dat`/`Surf.dat`) instead
        of separate `grid_file`/`hydro_file` paths, since all three files
        always come from the same XNS run directory.
      - **Verified**: rebuilt cleanly; a dedicated 1-cycle check job (248449,
        `/sakura/ptmp/tlam/athenak_run/cfc_bu8_surffix_check/`) ran with
        **zero** `FATAL ERROR`s, and `t=0` density slices (with
        `plot_slice.py --grid` MeshBlock overlays) confirm the halo is gone --
        density now cuts cleanly to `dfloor` exactly at the star's actual
        oblate surface, matching `Surf.dat`'s `R_eq~11.3`/`R_pole~6.8`-`6.9`
        with no extended halo region at all.
    - **Follow-up production run (job 248450, 2026-07-23)**: with the
      Surf.dat fix in place, resubmitted the full test at `tlim=200` (double
      the first corrected-ID run) and `mg_debug_defect_by_level=false` (the
      per-solve debug screen output served its purpose for this
      investigation and is no longer needed) -- 16 nodes, same
      `cfc_bu8_stability.athinput`/`cfc_bu8_stability_test/` otherwise
      unchanged. The previous (pre-Surf.dat-fix) diagnostics/movies were
      moved to `old_run_presurffix/` in that directory rather than deleted,
      since they were generated from the halo-contaminated initial data and
      are superseded, not just stale.
      **Result: completed cleanly.** Reached `tlim=200.0` exactly at
      cycle 2321 (~75 min wall time on 16 nodes), zero `FATAL ERROR`/NaN/inf
      messages anywhere in the log across the entire run. Baryon mass
      conservation from `cfc_bu8_stability.mhd.hst`: `M0=1.377519325298131`,
      `M(t=200)=1.377487860418838`, max `|M(t)-M0|/M0` over the whole run
      `= 2.284e-5` -- tighter than the first corrected-ID run's shorter
      duration, confirming the Surf.dat-fixed initial data plus the
      higher-resolution/full-z-domain configuration (from the earlier
      corner-defect fix) is stable well beyond the point the original,
      buggy-ID run diverged (t~30). This is the first fully clean, full-
      duration BU8 rotating-star run in this investigation. **However**, see
      the next bullet -- the user inspected the `rho_max(t)` diagnostic and
      correctly judged the star not actually stable (a large-amplitude
      transient, not the small perturbation response a converged equilibrium
      should show); a third, more serious ID bug (missing Lorentz factor) was
      found and fixed as a result.
    - **Third bug found and fixed: missing Lorentz factor in the velocity
      primitive (2026-07-23, user diagnostic review)**. `XNSInterpToADMAndPrim`
      set `w0(IVX/IVY/IVZ)` directly to the Eulerian-observer coordinate-basis
      3-velocity `v^i` (`vx=-vphi*x2; vy=vphi*x1;`, `vphi` = `Hydroeq.dat`
      column 4, confirmed via `XNSMAIN.f90`/`HYDROEQ.f90` to be
      `v^phi = u^phi/(alpha u^0) + beta^phi/alpha`, the standard 3+1 Eulerian
      velocity). But AthenaK's dyn_grmhd primitives do **not** store `v^i`
      itself -- confirmed via `src/eos/primitive-solver/primitive_solver.hpp:
      508-535,570` (`ConToPrim` builds/stores `Wv_u`; comment: "Athena passes
      in Wv, not v") and cross-checked against every other ID-import pgen that
      sets a genuine nonzero velocity (`pgen/dyn_grmhd/lorene/lorene_bns.cpp:
      263-277`, `kadath/kadath_bns.cpp:406-408`, `sgrid/sgrid_bns.cpp:331-333`,
      `elliptica/elliptica.cpp:399-401`), all of which compute
      `vsq = gamma_ij v^i v^j`, `W=1/sqrt(1-vsq)`, and write `W*v^i`. This pgen
      was missing that factor entirely -- the star started with too little
      physical angular momentum, hence the large adjustment transient.
      **Fix**: in `XNSInterpToADMAndPrim`'s star-interior branch, after
      building `vx,vy,vz` from `vphi`, compute
      `vsq = psi4*(SQR(vx)+SQR(vy)+SQR(vz))` (`gamma_ij v^i v^j` with
      `gamma_ij=psi4*delta_ij`, CFC's conformally-flat Cartesian metric;
      mirrors `lorene_bns.cpp`'s `Primitive::SquareVector(vu,g3d)` call,
      inlined since our metric is trivially diagonal), clamp
      `vsq=fmin(vsq,0.9999)` for numerical safety near the surface/mass-
      shedding limit, `lorentz_w=1/sqrt(1-vsq)`, then multiply `vx,vy,vz` by
      `lorentz_w` before returning. Verified via a 1-cycle check (job 248459):
      zero FATAL errors, `rho_max(t=0)=1.279e-3` (matches the model's central
      density), `max|W*v_phi|~0.347` (physically sane for a near-mass-shedding
      rotator: naive `Omega*R_eq=0.02633*11.24=0.296`, Lorentz-boosted to
      ~0.35), and a clean, artifact-free t=0 density slice.
    - **New history diagnostics (2026-07-23, user request via `/btw`)**: added
      `XNSRotStarHistory` (`xns_rotstar.cpp`), enrolled via
      `user_hist_func = &XNSRotStarHistory;` in `UserProblem()` and
      `<problem> user_hist = true` in the athinput. Mirrors `TOVHistory` in
      `dyngr_tov.cpp` for `rho-max`/`alpha-min` (same `Kokkos::Max`/`Min`
      reduction, same manual `MPI_MAX`/`MPI_MIN`-then-zero-non-root-copies
      hack needed since `history.cpp`'s own post-reduction step is always
      `MPI_SUM`), plus a new `ang-mom` field: the total angular momentum about
      the z-axis, `integral sqrt(gamma)*(x*S_y - y*S_x) d^3x` (`S_i` the
      undensitized ADM momentum density). Confirmed via `dyn_grmhd.cpp`'s own
      `SetTmunu` (`tmunu.S_d(a) = cons(IM1+a)*ivol`, `ivol=1/sqrt(gamma)`) that
      the evolved conserved variable `u0(IM1+a)` is already `sqrt(gamma)*S_a`
      -- so the integrand needs **no explicit `psi^6`/`sqrt(detg)` factor in
      code**, just `vol=dx1*dx2*dx3` times `(x*u0(IM2) - y*u0(IM1))`, exactly
      mirroring `history.cpp`'s own `LoadMHDHistoryData` mass-sum idiom
      (`vol=dx1*dx2*dx3`, `hvars[IDN]=vol*u0_(IDN)`, no metric factor) --
      contrast `LoadZ4cHistoryData`'s `vol=dx1*dx2*dx3*sqrt(|detg|)`, needed
      there only because *that* function's fields are not pre-densitized.
      **Implementation note**: `HistoryOutput` writes each `PhysicsModule`
      block to its own separate file (`history.cpp:401-416`) -- these three
      new fields land in `<basename>.user.hst`, not `<basename>.mhd.hst`
      (initially overlooked when first verifying this; the values were there
      all along, just in the sibling file). Verified via job 248461 (1-cycle,
      run together with the Lorentz-factor fix above): zero FATAL errors,
      `cfc_bu8_surffix_check.user.hst` shows `rho-max=1.2794e-3` (matches job
      248459's independent `bu8_bin_reader.py` cross-check to 4 sig figs),
      `alpha-min=0.7107` (sane lapse). **Correction (see next bullet)**:
      `ang-mom=1.2125` was *not* actually a sane match -- it (and `mass`) were
      both substantially wrong, discovered once compared against the initial
      data's own reported values, not just checked for plausibility.
    - **Fourth bug found and fixed: `mg_threshold` too loose for psi/alpha_psi's
      own accuracy on this configuration, plus a real sign error in the X^i/
      beta^i vector-Poisson source (2026-07-23, user diagnostic review)**.
      Job 248462 (first full-run attempt with the Lorentz-factor fix) was
      cancelled by the user mid-run after they cross-checked the new
      `ang-mom`/`mass` history columns against XNS's own `LogFile.dat`
      (`ANGUL. MOMENT. (E) = 1.8128063487726112`, `REST MASS = 1.8255846452192752`)
      and found both substantially off (`mass=1.379`, 24.5% low; `ang-mom=1.2125`,
      33% low) -- not the "right order of magnitude" match the previous bullet
      assumed. The user correctly identified the diagnostic to isolate this:
      compare `psi` after `CFC::InitializeMetric()` against the initial data's
      own `psi` -- these should match for a genuine equilibrium. They did not:
      AthenaK's converged `psi` was up to ~6-7% low near the star's center
      relative to the XNS table (extracted via `plot_slice.py --dump-npz` on
      the `adm_psi4` field vs. `Hydroeq.dat`'s own `psi` column).
      - **Ruled out first** (each independently verified correct, to make sure
        the search stayed on the real cause): re-integrating XNS's own
        `Grid.dat`/`Hydroeq.dat`/`Surf.dat` tables directly in Python (same
        atmosphere cut, same Lorentz-factor formula as `xns_rotstar.cpp`)
        reproduced `REST MASS`/`ANGUL. MOMENT.` to 7 significant figures --
        confirming the Surf.dat atmosphere cut and the Lorentz-factor fix
        (previous bullet) are both correct, and the bug is specifically in
        AthenaK's own metric *re-solve*, not the pgen's initial-data
        transcription. `CFC::InitializeMetric()`'s outer Picard loop itself
        was also confirmed tightly convergent (`max|delta psi|` decaying
        geometrically to `~1e-10`), ruling out "just hasn't finished
        converging yet" at the outer-loop level.
      - **Bug A (found first, real but not the dominant cause): wrong sign in
        `MGCFCVectorPoissonDriver::LoadPoissonSource`** (`mg_cfc_vector_poisson.cpp`).
        `CFCVectorPoissonStencil::Apply` is identical in form to
        `gravity::GravityStencil::Apply`; gravity's own precedent
        (`mg_gravity.cpp`: `LoadSource(u0, IDN, ng, -four_pi_G_)`) establishes
        that solving `Delta(u) = C` via this stencil requires
        `LoadSource(..., fac=-1.0)` when the array passed in already equals the
        full intended RHS `C` (as `p_src`/`eta_src` do here, already carrying
        their own `8*pi`/`16*pi` factors). The code had `fac=+1.0` -- solving
        the sign-flipped equation. **Fixed**: changed to `fac=-1.0`.
        Mathematically this does *not* change `psi`/`mass`/`ang-mom` (a global
        sign flip of `X^i` flips `Adual^ij` linearly, but `Ahat^2 = sum
        (Adual^ab)^2`, feeding `psi`'s own equation, is invariant to that flip)
        -- confirmed empirically too (job 248470: `mass`/`ang-mom` unchanged
        from before the sign fix). It *does* matter for `beta^i`'s own source
        (Gmunu eq. 75's `2*Adual^ij*D_j(alpha*psi^-6)` term uses `Adual^ij`
        linearly, not squared) and hence `vK_dd`'s physical sign/frame-dragging
        direction -- a real, independent bug, fixed regardless of it not being
        this particular symptom's cause.
      - **Bug B (the actual dominant cause): `<cfc> mg_threshold=3e-3`
        (item 22's AMR-robustness loosening) was too loose for psi/alpha_psi's
        own accuracy on this star.** Diagnosed by adding one-off `DEBUG
        <SolverName>(this=...) eps_=...` prints (temporary, since removed) to
        all three `Solve()` methods (`mg_cfc_vector_poisson.cpp`,
        `mg_cfc_conformal_factor.cpp`, `mg_cfc_lapse.cpp`) to unambiguously
        identify which trace belonged to which solver in the interleaved
        `mg_verbose=2` log. This revealed `ConformalFactor`'s own solve
        stopping after only 2-3 V-cycles once its defect fell just under
        `3e-3` (e.g. `0.0029537 < 0.003`) -- correct per its own (deliberately
        loose) threshold, but nowhere near machine precision, and evidently
        not accurate enough for this star's own equilibrium to be preserved.
        A first attempt narrowed this down incorrectly: giving the X^i/beta^i
        vector-Poisson solves their own tight, dedicated
        `mg_poisson_threshold` (new `<cfc>` key, mirroring the existing
        `mg_poisson_outer_bc`/`mg_poisson_mporder` precedent) while leaving
        `mg_threshold` (psi/alpha_psi) at the loose `3e-3` reproduced the
        *exact same* `mass=1.379`/`ang-mom=1.2125` deficit (job 248468/9/70) --
        proving the vector-Poisson threshold was NOT the culprit (consistent
        with Bug A's sign-invariance argument: `Ahat^2` doesn't care how
        tightly `X^i` itself converges once it's converged at all, since the
        earlier `mg_poisson_threshold` fix already had it converging to
        `~1e-8`-`1e-9`, plenty tight). **Fixed**: tightened `mg_threshold`
        itself to `1e-10` (job 248471, later corroborated by an earlier
        coarser test at `1e-8`, job 248467) -- the per-call V-cycles
        themselves converge cleanly at this tolerance every time (3-5
        V-cycles per Picard iteration after the first, each warm-started); see
        the next bullet, though, for a genuine (separate) convergence-rate
        issue this surfaced at the *outer* Picard-loop level.
      - **Final verification (job 248471, mg_threshold=mg_poisson_threshold=
        1e-8)**: `psi` now matches the XNS initial data's own `psi` to
        **~0.01% or better**, from the stellar center out to `r~19` (was up to
        6-7% low at the center before) -- direct point-by-point comparison via
        `plot_slice.py --dump-npz` on `adm_psi4` vs. `Hydroeq.dat`'s `psi`
        column. `mass=1.824618089` vs. XNS's `1.8255846452192752` (0.05% off,
        was 24.5%); `ang-mom=1.811048179` vs. XNS's `1.8128063487726112`
        (0.10% off, was 33%). **Per user direction, production uses
        `mg_threshold=mg_poisson_threshold=1e-10`** (tighter than the `1e-8`
        used in this last verification run, since no per-call convergence cost
        was found at `1e-8` and `1e-10` is the base class's own original
        default before item 22 ever loosened it).
      - **New finding (2026-07-23, discovered while starting the job 248475
        production run): the outer `CFC::InitializeMetric()` Picard loop's own
        contraction rate slowed substantially once X^i is genuinely resolved
        each iteration, and this "final verification" run (248471) was itself
        silently hitting the 50-iteration cap (`<cfc>/init_iter_max`, default),
        not truly converging to `init_tol=1e-10`.** Re-examining job 248471's
        full log (not just the final `psi`/mass/ang-mom comparison) shows the
        exact same `### WARNING in CFC::InitializeMetric ... did not converge
        after 50 iterations` message, stalled at `max|delta psi|=2.09333e-05`
        -- i.e. the excellent `~0.01%` psi match documented above was itself
        obtained from a *non-fully-converged* outer state, not a truly
        converged one. Before the threshold fixes (loose `mg_threshold=3e-3`),
        this same outer loop showed a clean, exact `0.5x`-per-iteration decay
        (exactly matching `init_omega=0.5`) all the way to `~1e-10` within 29
        iterations (job 248461) -- consistent with X^i's own under-convergence
        effectively *decoupling* the outer iteration (a frozen/inaccurate X^i
        each step behaves, for the psi-update alone, like simple linear
        relaxation). With X^i properly resolved every step (post-fix), the
        outer loop is now iterating the *genuinely coupled* (psi, X^i)
        fixed-point map, whose own natural contraction rate for this star
        turns out to be much slower (~0.88x/iteration, not `0.5x`) -- at that
        rate, reaching `1e-10` from the post-iteration-1 value needs roughly
        150 iterations, not 50. The per-call multigrid V-cycles are not at
        fault (each one converges to its own `eps_` correctly every time;
        occasional "Slow convergence: defect ratio=..." lines are the
        standard, benign multigrid diagnostic, not failures). Job 248475 (the
        `tlim=200` production run) hit the identical cap/warning at
        essentially the same `~2.09e-5` value. Per user direction, job 248475
        was initially left running to investigate the stall separately (see
        below) -- it was later cancelled and resubmitted once the actual
        cause was found (next bullet).
      - **Root cause found and fixed: `init_omega=0.5` (the under-relaxation
        itself) was causing the stall, not helping it.** `init_omega=0.5` was
        applied proactively to BU8 per item 22's precedent (a *different*,
        non-rotating migration-test star that genuinely needed under-
        relaxation for robustness) but was never actually verified as
        necessary for BU8 itself. Direct A/B test in
        `cfc_bu8_surffix_check/` (job 248476, `init_omega=1.0`, otherwise
        identical to the stalled configuration): the outer loop converges
        almost immediately -- `max|delta psi|` drops from `0.223754` (iteration
        0) to `1.28373e-06` (iteration 1, a single step!), then decays
        smoothly to *exactly* `0` by iteration 25 of the 50 available, no
        warning at all. Compare the `omega=0.5` trace's own iteration-0/1
        values (`0.111877 -> 0.017713`, only a `~6x` drop) -- the
        under-relaxation was directly responsible for both the slow ~0.88x/
        iteration asymptotic decay *and* the iteration-1 bottleneck. Removing
        it also **improved accuracy**, not just speed: `mass=1.825610933` vs.
        XNS's `1.8255846452192752` (`0.0014%` off, vs. `0.05%` with the
        stalled `omega=0.5` run) and `ang-mom=1.812464081` vs. XNS's
        `1.8128063487726112` (`0.019%` off, vs. `0.10%`) -- consistent with
        letting the Picard loop actually reach its fixed point instead of
        stopping partway. **Fixed**: production `cfc_bu8_stability.athinput`
        now uses `init_omega=1.0` (no under-relaxation). `mg_npresmooth`/
        `mg_npostsmooth=3` (also inherited from item 22's migration-test
        precedent) were left unchanged -- no evidence yet that they're
        similarly unnecessary for BU8, and changing multiple things at once
        would have muddied this A/B test.
    - **Final production run (jobs 248477 + 248639, 2026-07-23): SUCCESS.**
      Full `tlim=200` run with all five fixes together (Surf.dat surface cut,
      Lorentz factor, vector-Poisson sign, `mg_threshold`/
      `mg_poisson_threshold=1e-10`, `init_omega=1.0`) -- supersedes job 248450
      (missing Lorentz factor), job 248460 (Lorentz-fixed but missing history
      diagnostics, cancelled before completion), job 248462 (Lorentz+history-
      fixed but still using the too-loose `mg_threshold=3e-3`, cancelled once
      the `ang-mom`/`mass` mismatch against `LogFile.dat` was found), and job
      248475 (threshold-fixed but still under-relaxed and hence non-fully-
      converged at `t=0`, cancelled once the under-relaxation was identified
      as the cause). Job 248477 ran to `t=169.49` (cycle 1825) before hitting
      its wall-clock limit with zero FATAL/NaN errors; job 248639 restarted
      from the last checkpoint and completed the remaining `~30.5` time units
      cleanly, reaching `tlim=200.0` exactly at cycle 2121 -- zero FATAL/NaN
      across the entire combined run.
      **Diagnostics** (`cfc_bu8_stability.mhd.hst`/`.user.hst`, both segments
      concatenated continuously): baryon mass conserved to
      `max|M(t)-M0|/M0 = 1.19e-8` over the *entire* `tlim=200` run (vs.
      `2.28e-5` for the earlier, still-buggy "successful" run, item 23's first
      pass) -- essentially machine-precision-level conservation. Angular
      momentum conserved to `1.32e-5` relative drift. `M0=1.825610932873623`
      vs. XNS's `REST MASS=1.8255846452192752` (`0.0014%` off);
      `ang-mom(t=0)=1.812464080932092` vs. XNS's `ANGUL. MOMENT.=
      1.8128063487726112` (`0.019%` off) -- both matching the dedicated
      verification run (job 248476) that first established these fixes work.
      **`rho_max(t)` now shows only a small-amplitude oscillation
      (`~0.0012793`-`0.0012800`, `~0.02%` peak-to-peak variation) around the
      model's central density, with no large transient at all** -- a
      qualitatively different, physically correct result compared to the
      first "successful" run (item 23's earlier pass) that the user correctly
      judged as *not actually stable* from exactly this diagnostic (a `~30%`
      adjustment transient before settling into a persistent oscillation).
      This is the genuine, validated confirmation that the BU8 rotating-star
      equilibrium is preserved by this module to high accuracy. Diagnostic
      plots/movies regenerated in `cfc_bu8_stability_test/` (superseding
      `old_run_underrelaxed/`, `old_run_loosethreshold/`,
      `old_run_novelocityfix/`, `old_run_presurffix/`, each preserved for
      comparison, not deleted).

24. **Baryon mass added to the TOV solver (done); migration test's (item 22)
    `CFC::InitializeMetric()` found to converge cleanly to the WRONG psi for
    its compact star -- root cause narrowed to the outer Picard iteration
    scheme itself, not AMR (2026-07-24).**
    - **Baryon mass in `TOVStar` (`src/utils/tov/tov.hpp`)**: added
      `dMb/dr = 4*pi*r^2*sqrt(A)*rho` (`A = 1/(1-2m/r) = g_rr`, the proper-
      volume rest-mass integral in Schwarzschild-like coordinates) alongside
      the existing `P`/`m`/`alpha`/`R_iso` RK4 integration -- a direct parallel
      of the existing `dm = 4*pi*r^2*e` line, just using `rho` (rest-mass
      density, already computed locally) instead of `e` (total energy
      density) and the extra `sqrt(A)` proper-volume factor. New `Mb`/
      `Mb_edge` members, printed as `Baryon mass: <value>` right after the
      existing `Mass: <value>` line. Verified independently: an from-scratch
      Python re-integration of the identical RK4 scheme (rhoc=8e-3, kappa=100,
      Gamma=2) reproduces the C++ output to 5+ significant figures
      (`Mass=1.447294`, `Baryon mass=1.534986`).
    - **Metric-initialization check (the actual point of this investigation)**:
      compared `CFC::InitializeMetric()`'s converged `psi4` for the migration
      test's compact star (`rhoc=8e-3`, `2M/R_schw~0.5`) against the analytic
      isotropic-TOV solution via `plot_slice.py --dump-npz` + the independent
      Python re-integration above. Found a large, systematic discrepancy:
      `psi4` at the star's center converges to `~2.22` vs. the analytic
      `~5.91` (a factor `~2.7x` too small), decreasing to `~1.4x` off near the
      edge. Density matches the analytic profile to `<0.3%` everywhere,
      ruling out a pgen/primitive bug -- this is specifically the solved
      metric that's wrong. Confirmed this is the direct cause of the
      simulation's own `.mhd.hst` `mass` column reading `~0.597` at `t=0`
      instead of the true `Mb_edge=1.535` (reconstructing the baryon mass
      integral from the grid's own psi4+rho reproduces `~0.59`; substituting
      the *analytic* psi4 with the same grid rho reproduces `~1.51`, close to
      the true value) -- the low reported mass is a correct consequence of
      the wrong metric, not a separate history-diagnostic bug.
    - **Ruled out: `init_omega=1.0`** (removing the outer Picard loop's
      under-relaxation, mirroring item 23's BU8 precedent) -- unlike BU8, this
      genuinely diverges exponentially for this more compact star (`max|delta
      psi|` grows `0.28 -> 1.29 -> 172 -> 7.6e24 -> NaN` over ~25 iterations,
      confirmed via a real run then cancelled before it filled the disk with
      NaN-cascade error spam). The existing `init_omega=0.5` workaround is
      load-bearing for this star, unlike BU8's case.
    - **Ruled out: AMR/refinement entirely.** Using the already-built
      `mg_debug_analytic_residual_test` diagnostic (`mg_cfc_conformal_
      factor.cpp`, dormant since item 21), isolated to exactly the outer
      loop's very first iteration (`init_iter_max=1`, guaranteeing `Utilde`/
      `Ahat^2` are built only from the pgen's own pristine analytic seed, not
      a drifted later-iteration state): one V-cycle from the exact analytic
      seed reproduces the star's *center* to `~0.00045` (excellent) -- the
      discretization/coefficients are fine in isolation. Two cheap AMR-
      configuration variations (more smoothing, `mg_npresmooth/postsmooth`
      3->8; wider refined region, `[-7.5,7.5]`->`[-10,10]`) left the
      converged center-ψ4 unchanged to 4 significant figures
      (`2.219235`/`2.219067`/`2.219067`). **Decisive test (user-requested)**:
      a uniform-resolution, single-level, no-`<mesh_refinement>`-at-all
      control run (octant domain, half-width `21` (`~5x R_edge_iso~4.267`),
      `nx1=128` -> `dx~0.164`, entirely different resolution/domain/shape from
      every AMR-based run) reproduced essentially the *same* wrong answer
      (`psi4` ratio to analytic `~0.383` vs. `~0.376` for every AMR variant --
      a ~2% difference consistent with ordinary resolution effects, not a
      qualitative fix). **This conclusively rules out AMR/refinement as the
      cause, at any level or configuration.**
    - **Current understanding, not yet resolved**: the wrong answer is a
      robust, reproducible feature of `CFC::InitializeMetric()`'s outer Picard
      iteration itself (rebuilding `Utilde`/`Ahat^2` from the current `psi`
      every iteration, a long-range-coupled Poisson-type fixed point),
      independent of spatial discretization entirely. Working hypothesis: this
      compact star's coupled nonlinear fixed-point map may have multiple
      self-consistent solutions, and the damped (`omega=0.5`) iteration is
      being captured by the wrong one rather than staying near the correct
      one it was seeded from -- a genuinely different class of problem than
      anything in items 9/12/21 (which are all about the multigrid V-cycle's
      own spatial-discretization accuracy, not the outer iteration's own
      dynamics/uniqueness). Not yet root-caused further; the next
      investigative step (not yet taken) would need to probe the outer
      iteration's own map (e.g. perturbing the seed, or tracking the
      map's local linearization/eigenvalues across iterations) rather than
      anything spatial-discretization-related.
    - **Also implemented this pass (separate from the above, per user
      request), independent of whether it fixes the migration star**: a new
      `<cfc> mg_correction_omega` damping knob for the FAS coarse-grid
      correction (`u -= uold`, prolongated back onto a finer level), gated via
      a new `virtual Real MultigridDriver::CorrectionOmega() const { return
      1.0; }` hook (`multigrid.hpp`, default no-op -- zero behavior change for
      gravity/vector-Poisson, which never override it), applied in both
      `Multigrid::ComputeCorrection()` (`multigrid.cpp`) and
      `MultigridDriver::ProlongateAndCorrectOctets()` (`multigrid_driver.cpp`,
      the octet-level correction, computed inline there rather than through
      `ComputeCorrection()`). Empirically found **not** to change the
      migration star's converged answer at all (`mg_correction_omega=0.3`
      gave the same `~0.376` ratio, just slower/noisier convergence with new
      per-cycle "Failed to converge" messages) -- consistent with the
      AMR-is-ruled-out finding above (damping a correction can't fix a
      systematically wrong fixed point, only its convergence path). Kept in
      the code as a generically useful, zero-risk-when-unused knob, not
      reverted.
    - **See also items 27-29 (2026-07-25)**: three different follow-up
      attempts at this item's own "not yet resolved" state above. Items
      27-28 (`psi^5` Newton formulation) diverge for this exact star, ruled
      out here specifically (though made the new default for milder stars).
      Item 29 (`init_freeze_conserved`, a one-shot mode that sidesteps the
      outer Picard loop described above entirely) reproduces the analytic
      solution to `~0.01-0.03%` on this same star -- appears to actually
      RESOLVE this item, pending the user's own confirmation before marking
      it closed outright. Note (2026-07-26): a longer `tlim=200` dynamical
      run under `init_freeze_conserved=true` surfaced a SEPARATE, still-open
      issue downstream of `t=0` (item 30) -- that finding does not reopen
      this item's own `t=0`-initialization result, which remains excellent.

25. **Robin BC implemented at the octet (AMR-refined) level of
    `MultigridDriver::ApplyPhysicalBoundariesOctet` (2026-07-24) -- closes
    the gap flagged (but deliberately deferred) in item 16's Robin BC entry.**
    - **The gap**: `ApplyPhysicalBoundariesOctet` had no physical-position math
      at all -- any face marked `BoundaryFlag::mg_robin` silently fell back to
      the same `sign=+1` (zerograd-like) reflection every other non-
      `mg_zerofixed` flag got. Item 16 deferred fixing this because deriving
      an octet's physical position from its `LogicalLocation` was assumed to
      need "genuinely new arithmetic with nothing existing to mirror," and no
      current CFC test placed refinement at the actual domain boundary (item
      22's migration test's own refined region is deep in the interior,
      confirmed this session -- this fix was **not** expected to change that
      test's own outcome, and empirically did not; see item 24 above).
    - **Turned out simpler than assumed**: the needed building block,
      `maxlx1 = nrbx1_ << lev` (number of octets spanning x1 at this
      refinement level), was already computed and used in this exact
      function's own outer-x1-face check. The octet's physical extent follows
      directly: `octet_width = (mesh_size.x1max-mesh_size.x1min) / maxlx1`,
      `octet_x1min = mesh_size.x1min + loc.lx1*octet_width` (same for x2/x3
      via `nrbx2_/nrbx3_`/`loc.lx2/lx3`). Each `MGOctet` has a fixed 2-cell
      core (`nc = 2+2*ngh`), so the per-axis cell width is `octet_width/2`
      for the normal (`oct.u`) case or the whole `octet_width` for the
      coarse-buffer (`fcbuf=true`, single-cell) case -- mirrors the existing
      `l=ngh,r=ngh+1` vs. `r=ngh` distinction already coded for exactly this
      purpose. The Robin fill formula itself (`u_ghost = u_anchor *
      (r_anchor/(r_ghost+1e-30))^robin_order_`, true 3D radius, no
      multipole-origin subtraction) is identical to the already-working
      root-level (`MGRootBoundary`) and per-MeshBlock (`PhysicalBoundary`)
      implementations -- just evaluated at this octet's own derived extent
      instead of a MeshBlock's `mb_size` or the whole root grid.
    - **Implementation**: `multigrid_driver.cpp`'s `ApplyPhysicalBoundariesOctet`
      only -- computes the octet's own extent/effective-dx once per call, adds
      an `if (... == BoundaryFlag::mg_robin)` branch (using the formula above)
      ahead of the existing sign-based fallback in each of the 6 face blocks,
      leaving every other flag's behavior (zerofixed/zerograd/multipole)
      completely unchanged. No gating flag needed: `mg_robin` is only ever
      written into `mg_mesh_bcs_` by `MGCFCConformalFactorDriver`/
      `MGCFCLapseDriver`'s own constructors, so this is inherently scoped to
      those two solvers with zero risk to gravity or the CFC vector-Poisson
      solvers.
    - **Verified via a dedicated test** (since no existing input exercises
      this path): a small octant TOV1 (`rhoc=1.28e-3`) input adapted from
      `cfc_tov_amr_ghosttest.athinput`, with `refined_region1` moved from the
      innermost corner to the *outermost* one (`[19.2,25.6]^3`, touching
      `x1max=x2max=x3max=25.6` on all three axes at once -- a harder stress
      case than a single-face touch). Confirmed via direct A/B (git-stashing
      the fix, rebuilding, rerunning the identical input): both pre- and
      post-fix runs hit the *same* pre-existing "Failed to converge"/"Slow
      convergence" (defect stuck at `~1.46e-6`) for this specific new
      configuration -- confirming that symptom is unrelated to this change,
      not a regression it introduces (no NaN/Inf in either case). The fix
      itself produces a small, expected shift in the near-boundary `psi4`
      profile (`1.081141` post-fix vs. `1.081263` pre-fix at the last real
      cell before the boundary) with the profile remaining smooth and
      monotonically decreasing toward the boundary in both cases (no kink,
      no blowup) -- consistent with a correct, working Robin implementation
      whose effect is modest at this star's distance/compactness scale, as
      expected (item 24's own note that Robin at the octet level was never
      expected to be large for a mild star this far from a compact source).

26. **Multipole BC implemented at the octet (AMR-refined) level of
    `ApplyPhysicalBoundariesOctet` (2026-07-25) -- direct counterpart to item
    25's Robin fix, closing the analogous, previously-documented gap.**
    - **The gap**: identical in shape to item 25's -- `ApplyPhysicalBoundaries
      Octet`'s own top-of-function comment already described the intended
      formula ("ghost = 2*phi_mp - interior, linear extrapolation") but the
      `apply_bc` lambda never implemented it; any face marked `BoundaryFlag::
      mg_multipole` fell through to the same `sign=+1` (zerograd-like)
      fallback. **Unlike Robin, this one is live, not inert**: `psi`/
      `alpha_psi`'s `mg_outer_bc` only ever accepts `"robin"`/`"zerofixed"`
      (confirmed by re-reading `mg_cfc_conformal_factor.cpp`'s constructor --
      multipole is never actually selected for them), but
      `MGCFCVectorPoissonDriver` (the merged `P_i`/`eta` solve for both `X^i`
      and `beta^i`, `nvar_=4`) defaults `mg_poisson_outer_bc` to `"multipole"`
      (`mg_cfc_vector_poisson.cpp:210-216`, `autompo_=false`, fixed origin at
      the coordinate origin) -- this driver's octets, created whenever *any*
      CFC AMR run refines at all, hit this exact gap whenever refinement
      reaches the domain boundary.
    - **Formula, mirrored exactly from the already-working `MGRootBoundary`
      host-path multipole block** (`multigrid_driver.cpp`, the `"Multipole
      expansion boundaries on host"` block inside `MGRootBoundary` -- line
      numbers not cited here since item 25's own edits to the earlier
      `ApplyPhysicalBoundariesOctet` shifted everything after it in this same
      file; search by comment text instead of a line range), the one to
      mirror since octets are host-side code: per channel (`v=0..nvar_-1`,
      reading `mpcoeff_[v*25]`, the existing per-channel flat-indexed
      convention from item 18), evaluate `phis = EvalMultipolePhi(x-mpo_[0],
      y-mpo_[1], z-mpo_[2], mc, mporder_)` **once** at the face coordinate
      itself (the octet's own `ox1min`/`ox1max` etc., derived via item 25's
      already-built octet-extent machinery -- reused as-is, no new position
      derivation needed), then fill **every** ghost depth `n` with the same
      `phis`: `ghost(ngh-1-n) = 2*phis - interior(ngh+n)`. Two shape
      differences from Robin worth flagging for a future reader: (1) one
      `phis` evaluation reused for all `n` (not re-evaluated per depth like
      Robin's `r`-dependent ratio), and (2) the ghost/interior index pairing
      is symmetric per-depth (`ngh-1-n` paired with `ngh+n`, matching
      `zerofixed`/`zerograd`'s own existing convention) rather than Robin's
      fixed-single-anchor convention -- using the wrong pairing would have
      been an easy, subtle mistake to carry over from the Robin code shape.
    - **Implementation**: `multigrid_driver.cpp`'s `ApplyPhysicalBoundariesOctet`
      only -- one new `else if (... == BoundaryFlag::mg_multipole && mporder_
      > 0)` branch per face (6 total), inserted between the existing `mg_robin`
      branch (item 25) and the sign-based fallback, reusing the same octet-
      extent/`pos1`/`pos2`/`pos3` helpers item 25 already added. No new gating:
      `mg_multipole` is only ever written into `mg_mesh_bcs_` by
      `MGCFCVectorPoissonDriver`'s own constructor, so zero risk elsewhere
      (gravity and `psi`/`alpha_psi` never select it).
    - **Verified**: rebuilt cleanly; reran item 25's own boundary-touching test
      (`cfc_octet_robin_check`'s input, refined corner touching
      `x1max=x2max=x3max` at once -- exercises `P_i`/`eta`'s octets there too,
      not just `psi`/`alpha_psi`'s). No NaN/Inf (confirmed via explicit grep).
      Same 63 "Failed to converge" FATAL count as pre-fix (this test's
      pre-existing, unrelated convergence quirk, per item 25's own A/B
      finding) -- consistent, not a regression. The stuck defect floor shifted
      slightly (`~1.67e-6` vs. `~1.46e-6` pre-fix), consistent with the fix
      genuinely perturbing `P_i`/`eta`'s own solve rather than being a no-op.
      `beta^i` (built from `P_i`/`eta`) confirmed to remain **exactly** `0.0`
      everywhere in this slice, as physically required for a static,
      non-rotating star (`S_i=0` exactly, no momentum source to produce any
      real multipole moment) -- a clean sanity check that the fix introduces
      no spurious nonzero content where none should exist. As with item 25,
      not expected to (and does not) change the still-open migration-test
      investigation (item 24), whose own refined region doesn't reach the
      domain boundary either -- this is a standalone correctness improvement
      for any future test that does place AMR refinement near the domain edge.

27. **Self-consistent `U_raw*psi^5` Newton formulation for
    `CFC::InitializeMetric()`'s psi solve (item 24 follow-up, 2026-07-25):
    diverges to NaN for the migration test's compact/unstable star, but
    converges cleanly (matching analytic TOV to ~0.02%) for the mild/stable
    "BU0" star -- the instability appears tied to compactness, not the
    formulation itself.**
    - **Motivation**: item 24 traced `CFC::InitializeMetric()`'s wrong-psi
      convergence to the outer Picard loop itself (AMR/discretization already
      ruled out). `SolveConformalFactor`'s Newton kernel evaluates the U-term
      as `Utilde*psi^-1`, where `Utilde = psi_prev^6*U` was built from the
      *previous* outer iteration's metric and held FIXED for the whole
      nonlinear V-cycle, while `psi^-1` tracks the LIVE Newton iterate --
      since `Utilde*psi^-1 = psi_prev^6*U*psi_current^-1`, this is only
      exactly the true equation's `U*psi_current^5` when `psi_prev ==
      psi_current`, which is never true mid-solve and may differ a lot
      across outer iterations for a compact star. Proposed fix (user):
      replace the source with the algebraically-identical `U_raw*psi^5`
      (`U_raw = Utilde/sqrt(detg)`, the raw undensitized energy density,
      computed exactly via the g_dd that built `Utilde`), using the SAME live
      Newton iterate for the whole `psi^5` power -- removes the outer-loop
      staleness entirely, folding the full nonlinearity into one
      self-consistent Newton solve.
    - **Design**: `ConformalFactorRHS` (`mg_cfc_conformal_factor.cpp`) is now
      a `template <bool UsePsi5>` function (`if constexpr`-branched, C++17)
      instead of a single formula -- per user direction, compiled as two
      genuinely separate code paths (not a runtime flag read inside the
      per-point hot loop) for performance. All 6 call sites
      (`SmoothPack`/`CalculateDefectPack`/`CalculateFASRHSPack` on
      `MGCFCConformalFactor`, and their Octet counterparts on
      `MGCFCConformalFactorDriver`) became thin one-line dispatchers (read
      `use_psi5_source_` once, call the matching `*Impl<true/false>`
      instantiation) wrapping the original bodies, now templated and calling
      `ConformalFactorRHS<UsePsi5>`. New `cfc::CFC` member `u_raw` (computed
      alongside `u_tilde` in `AssembleVectorSource`, an exact
      un-densitization via `adm::SpatialDet`, cheap so always computed). New
      `<cfc> init_use_psi5_source` input (default `false` as originally
      implemented in this item -- **flipped to `true` by item 28**, see
      below) read into `cfc_init_use_psi5_`, threaded only through
      `SolveConformalFactor`'s new `use_psi5_source` parameter and only ever
      set `true` by `InitializeMetric()`; the per-stage `CFC_SolvePsi` task
      always passes the default `false` regardless of the `<cfc>` input.
      Zero behavior change for every existing input at the time this item
      landed (confirmed: rebuild + the flag defaulted off everywhere it
      wasn't explicitly set) -- no longer true after item 28's default flip,
      see that item's own "Safety follow-up" bullet for which inputs needed
      an explicit override as a result.
    - **Tested on the migration test** (same `cfc_tov_migration_baryon_check`
      setup as item 24's original discovery,
      `cfc_tov_migration_psi5_check/`, job 249208, `init_use_psi5_source =
      true`, `init_omega = 0.5` unchanged): **diverges to NaN within
      `InitializeMetric()`'s very first outer iteration** (iteration 0's own
      `max|delta psi|` report is already `-1.79769e+308`, i.e. `Kokkos::Max`'s
      identity value surviving untouched because every comparison against a
      NaN `delta_psi` silently fails) -- ends in a wall of `NANS_IN_CONS`
      con2prim errors at cycle 1. Re-tested with heavy Newton damping
      (`cfc_tov_migration_psi5_damped_check/`, job 249209, `mg_omega_psi =
      0.2`) to check whether this was merely a Newton step-size problem:
      **same immediate NaN divergence**, ruling that out. The likely
      mechanism (not confirmed via further diagnostics, given the clear
      negative result): unlike the original `psi^-1` term (bounded,
      monotonically DEcreasing in `psi`, self-limiting), the `psi^5` term is
      unbounded and monotonically INcreasing -- `psi_floor_` only clamps
      Newton's step from below, so nothing prevents an overshoot to a large
      `psi` on a bad step (or from an inaccurate FAS coarse-grid correction
      inherited from a coarser V-cycle level, per item 24's own already-ruled-
      out `CorrectionOmega` hypothesis, now revisited under a much steeper
      nonlinearity), after which `psi^5`/`psi^4` growth compounds every
      subsequent iteration until overflow.
    - **Retested on the stable "BU0" star** (2026-07-25, user request): the
      migration star's divergence left open whether the `psi^5` formulation
      is fundamentally unstable, or only for that star's high compactness
      (`2M/R~0.5`, unstable branch). Re-ran on the milder, stable TOV1 star
      already used throughout item 9/item 5's own "BU0/BU8 stand-in"
      (`rhoc=1.28e-3`, `kappa=100`, `gamma=2`, `2M/R~0.29` -- same star as
      `cfc_stability_v4`/`inputs/dyn_grmhd/cfc_tov_stability.athinput`,
      `256^3` root + AMR to level 4), same setup, `nlim=1` (metric-init-only
      check), `init_use_psi5_source=true`
      (`cfc_bu0_psi5_check/`, job 249210): **converges cleanly in just 2
      outer iterations** (`iteration 0: max|delta psi|=0.194`, `iteration 1:
      max|delta psi|=0`), no NaN, no FATAL. Pointwise `psi^4`-vs-analytic
      comparison (same `plot_slice.py --dump-npz` + independent Python
      TOV re-integration pipeline used throughout this investigation, rhoc
      changed to `1.28e-3` in a new `tov_reintegrate_bu0.py`) matches to
      **~0.02%** (ratio `0.9998-0.9999`) from the star's center out through
      the vacuum region -- as good as or better than item 9's own original
      `~0.68%` validation of the DEFAULT formulation for this same star, and
      dramatically faster (2 iterations vs. the migration star's ~82 with the
      default formulation).
    - **Conclusion**: `U_raw*psi^5` is NOT fundamentally broken -- it works
      cleanly, accurately, and fast for a star well inside its numerically
      well-behaved regime. The NaN divergence is specific to the migration
      star's extreme compactness (unstable branch, `2M/R~0.5`), consistent
      with the mechanism guessed above (the term's unboundedness above makes
      Newton's linearization overshoot badly when the true solution's `psi`
      is far from 1, which happens for a very compact star but not a mild
      one). Item 24 (the migration test's own wrong-psi problem) remains
      OPEN -- this formulation isn't a fix for that specific star without
      additional stabilization (an upper `psi` ceiling, or clamping the
      Newton step itself) that's out of scope here. See item 28 for the
      follow-up BU8 test and the resulting decision to make this formulation
      `InitializeMetric()`'s new default for every OTHER star.

28. **BU8 (rotating star) tested with the psi^5 formulation; made the
    default for `InitializeMetric()` (2026-07-25).**
    - **Motivation**: item 27 showed `U_raw*psi^5` works cleanly and
      accurately for the mild, non-rotating "BU0" star, but is unusable as-is
      for the migration test's extreme, unstable-branch star. Before deciding
      on a default, the user asked to test the intermediate case: BU8, the
      rotating star (`rhoc=1.28e-3`, `Omega=2.633e-2`, mass ~1.69, moderately
      but not extremely compact, near mass-shedding) already used for item 22/
      23's own stability investigation -- a genuinely different regime (2D
      structure, off-center matter distribution, XNS-tabulated initial data)
      from both BU0 and the migration star.
    - **Test**: reused `cfc_bu8_stability_test/`'s own already-verified setup
      (`build_cfc_xns`, `xns_rotstar` pgen, `256`-ish AMR grid, `init_omega=
      1.0`, `mg_threshold=mg_poisson_threshold=1.0e-10`), `nlim=1`
      (metric-init-only check), `init_use_psi5_source=true`
      (`cfc_bu8_psi5_check/`, job 249213, rebuilt `build_cfc_xns` first since
      it shares the modified CFC source). Result: **converges in 3 outer
      iterations** (`0.224` -> `3.15e-7` -> `1.83e-8` -> `0`) vs. the default
      formulation's already-good ~25 iterations for this same star (item 23) --
      a large speedup. One non-fatal `### FATAL ERROR in MultigridDriver::
      SolveIterative` message appears ONCE, on the very first (cold-started)
      inner V-cycle solve, before the outer loop's own iteration-0 report
      (`defect=5.04e-10` vs. the very tight `threshold=1e-10` -- essentially
      converged, just short of the strict threshold within the 40-iteration
      cap); this message does NOT appear anywhere in the baseline run's own
      `.out` file, so it is specific to `UsePsi5`'s first cold-start solve,
      but is non-fatal (execution continues, no NaN, and every subsequent
      outer iteration converges cleanly) and does not recur. **Physical
      accuracy**: the converged baryon mass (`.mhd.hst`'s `mass` column at
      `t=0`) is `1.825610965013746` vs. the baseline (default formulation)
      run's `1.825610932873623` -- agree to `~2e-8` relative, i.e.
      effectively the same converged solution, not just "no NaN."
    - **Decision** (per user direction, "if it works, keep psi5 true by
      default since it speeds up initialization for well-behaved regime"):
      `<cfc> init_use_psi5_source`'s default flipped from `false` to `true`
      in `cfc::CFC`'s constructor (`cfc.cpp`) -- `InitializeMetric()` now uses
      the faster, self-consistent formulation unless an input explicitly
      opts out. Per-stage `CFC_SolvePsi` is completely unaffected either way
      (always calls `SolveConformalFactor` with the 2-argument, hardcoded-
      `false` form -- this default only touches the one-time metric
      initialization pass).
    - **Safety follow-up (required, not optional)**: since this default now
      has a demonstrated, non-graceful failure mode (NaN, not a warning) for
      the migration test's compact/unstable star, that star's own inputs
      needed an explicit opt-out added BEFORE the default flip could safely
      land: `inputs/dyn_grmhd/cfc_tov_migration.athinput` (the canonical
      source template) and `/sakura/ptmp/tlam/athenak_run/
      cfc_tov_migration_test/parfile.par` (the main production run directory,
      most likely to be resubmitted/restarted later) both gained
      `init_use_psi5_source = false` with a comment pointing at this item.
      The many now-historical, already-completed diagnostic-only run
      directories from items 22/24 (`cfc_tov_migration_baryon_check/`,
      `_moresmooth_check/`, `_widerregion_check/`, `_uniform_check/`,
      `_corromega_check/`, `_omega1_check/`, `_residualtest*/`) were
      deliberately NOT touched -- their results are already recorded and
      analyzed, and they are not expected to be blindly resubmitted; if any
      of them ever IS rerun, it would now hit the same NaN divergence unless
      manually given the same override, so treat any future reuse of those
      directories' `parfile.par` files as needing the same fix first.
    - **No other current CFC input is known to need this override**: BU0
      (item 27), BU8 (this item), and every octet/ghost-exchange smoke test
      this session (`cfc_octet_robin_check` etc., much milder stars, `v_pert=
      0`) either already worked or were never observed to diverge under the
      default (pre-flip) formulation, and none of them hit anything close to
      the migration star's `2M/R~0.5` compactness -- but this is not an
      exhaustive guarantee for every possible future star, only a record of
      what was actually tested. A future user hitting an unexpected NaN
      during `InitializeMetric()` for a new, very compact star should try
      `init_use_psi5_source=false` first, per this item.

29. **"Frozen-conserved" one-shot `InitializeMetric()` mode (`<cfc>
    init_freeze_conserved`): appears to RESOLVE item 24's wrong-psi problem
    for the migration test (2026-07-25).**
    - **Motivation**: item 24 traced the migration test's wrong-psi
      convergence to the outer Picard loop's own repeated re-freezing of
      `Ũ = psi^6*U` against an evolving metric (not the discretized equation
      itself -- that same investigation already found a single V-cycle seeded
      from the exact analytic solution reproduces the star's center to
      `~0.00045`). Items 27/28 tried fixing this via the Newton kernel's own
      formulation (`psi^5`) -- that diverges for this star. The user's
      alternative: stop rebuilding `Ũ`/`S-tilde_i` from fixed primitives
      against an evolving metric every outer iteration; instead call
      `PrimToConInit` ONCE, using whatever metric the pgen's own initial data
      already provides, and hold the resulting `Ũ`/`S-tilde_i` fixed while
      solving `X^i -> Adual^ij -> psi` exactly once -- "the same as doing one
      CFC step," no outer loop, no convergence check. The implied primitives
      may not exactly match the pgen's original ones afterward -- an accepted
      tradeoff, the same one every per-stage `CFC_SolvePsi` call already lives
      with (it never re-derives primitives from the converged metric either).
    - **Design**: `InitializeMetric()`'s Picard-loop body (`PrimToConInit`,
      `InitRecvXFields`, `SolveVectorPotential` + ghost exchange,
      `ReconstructVectorPotential` + ghost exchange, `ClearXFields`,
      `ComputeADual`, `SolveConformalFactor`) was extracted verbatim into a new
      private method, `CFC::RunXPsiSolvePass(Driver*)` (`cfc.cpp`/`cfc.hpp`) --
      a pure extract-method refactor, zero behavior change for the existing
      iterative path (confirmed: nothing in the extracted block reads/writes
      state that isn't already re-derived fresh each call). New `<cfc>`
      boolean `init_freeze_conserved` (default `false`) selects a top-level
      branch in `InitializeMetric()`: `true` calls `RunXPsiSolvePass` exactly
      once (plus an informational `max|delta_psi - initial guess|` print if
      `init_verbose`, no convergence gate -- there's nothing to converge);
      `false` runs the pre-existing `for` loop (`psi_old`/`omega`-blend/
      `dpsi`-check/`converged` bookkeeping, all unchanged, just now calling
      `RunXPsiSolvePass` once per iteration instead of the inlined sequence).
      Both branches fall through into the completely unmodified tail section
      (final refresh `PrimToConInit`, `RescaleMatterSources`, `SolveLapse`,
      ghost-exchange, `SolveShift`, `ReconstructShift`, `AssembleADM`,
      ghost-exchange `ADM`) -- that section already runs exactly once
      regardless of how many (if any) outer iterations happened, so it needed
      no changes at all. **Interaction with `cfc_init_use_psi5_` (items
      27/28)**: `RunXPsiSolvePass` calls `SolveConformalFactor` with whatever
      that flag is set to -- but per user direction, the two are mutually
      incompatible (see the dedicated bullet below), so `CFC::CFC`'s
      constructor forces `cfc_init_use_psi5_` false whenever
      `cfc_init_freeze_conserved_` is true, rather than leaving them free to
      combine.
    - **Verified**: rebuilt `build_cfc`/`build_cfc_xns` cleanly.
      **Regression check** (`cfc_migration_freezecons_regression/`, job
      249216, `init_freeze_conserved=false` explicit, `init_use_psi5_source=
      false` explicit): converges in ~82 iterations with the exact same
      geometric-decay pattern as item 24's own original finding, ending at the
      same known-wrong `psi^4` ratio (`~0.376` at center, `~0.71` at the edge)
      -- confirms the extract-method refactor is byte-identical to the
      pre-refactor iterative path. **New-mode test** (`cfc_migration_
      freezecons_check/`, job 249217, `init_freeze_conserved=true`,
      `init_use_psi5_source=false`): no NaN, no FATAL, one-shot pass reports
      `max|delta psi - initial guess| = 0.558946`. Pointwise `psi^4`-vs-
      analytic comparison (same `plot_slice.py --dump-npz` +
      `tov_reintegrate.py` pipeline used throughout this investigation):
      **ratio 0.9997-1.0000 from the star's center out to the edge** -- i.e.
      this one-shot pass is as accurate as (matching or slightly exceeding)
      item 24's own "exact analytic seed + 1 V-cycle" reference result,
      dramatically better than the iterative default's `~0.376-0.71`.
      **BU0 sanity check** (`cfc_bu0_freezecons_check/`, job 249219,
      `init_freeze_conserved=true`, `init_use_psi5_source=false`, the same
      mild TOV1 star from items 27/28): also clean, `ratio 0.9999-1.0000`
      throughout -- confirms the new mode doesn't regress an already-working
      case, and is at least as accurate as (if not marginally better than)
      both the iterative default and the `psi^5` formulation for this star.
    - **Mutual-exclusivity safeguard with `cfc_init_use_psi5_` (2026-07-25,
      per user direction)**: the two modes encode contradictory assumptions
      about what's held fixed during the single Newton solve.
      `init_freeze_conserved` holds `Utilde = psi^6*U` (the WEIGHTED/
      densitized conserved source) fixed across `RunXPsiSolvePass`'s solve --
      exactly what the default (`psi^-1`) Newton kernel already assumes.
      `init_use_psi5_source` instead holds `U_raw = Utilde/psi^6` (i.e.
      implicitly the PRIMITIVES) fixed and lets `Utilde` vary self-
      consistently as the Newton iterate itself changes -- the opposite
      assumption. Combining both would mean `init_freeze_conserved`'s
      one-shot `Utilde` snapshot gets silently re-interpreted as "primitives
      fixed" inside the Newton solve, defeating its own purpose. `CFC::CFC`'s
      constructor now forces `cfc_init_use_psi5_` false whenever
      `cfc_init_freeze_conserved_` is true (printing a `WARNING` when it does
      so), covering both an explicit `init_use_psi5_source=true` and the
      more likely silent case: `init_use_psi5_source`'s own default-`true`
      (item 28) combining unintentionally with a newly-added
      `init_freeze_conserved=true`.
    - **Conclusion**: this appears to genuinely RESOLVE the migration test's
      wrong-psi problem -- unlike items 27/28's `psi^5` attempt (which fixed
      the same root cause conceptually but proved numerically unstable for
      this exact star), the frozen-conserved one-shot approach sidesteps the
      outer Picard loop ENTIRELY rather than trying to make one Newton solve
      more self-consistent, and empirically works cleanly for both the
      compact/unstable star that broke everything else tried so far AND the
      mild star. Item 24 is very likely CLOSEABLE via this item, pending the
      user's own confirmation/decision (not marked closed unilaterally here --
      unlike items 27/28, no default was changed by this item; `<cfc>
      init_freeze_conserved` stays opt-in, default `false`, so every existing
      input's behavior is unaffected unless explicitly set).
    - **Update (2026-07-25/26)**: both open questions from this bullet's
      original text are now resolved/answered. Combining `init_freeze_
      conserved=true` with `init_use_psi5_source=true` is no longer possible
      to do accidentally -- see the mutual-exclusivity safeguard bullet
      above. Whether the one-shot pass's implied primitive mismatch stays
      small enough through real dynamical evolution was tested via a
      `tlim=200` run -- see item 30: mass conservation is excellent through
      the star's first oscillation/bounce (`t=0` to `~0.5ms`), but a
      separate, NOT-yet-root-caused issue appears downstream of that (item
      30's own "Conclusion"), unrelated to this item's own `t=0`
      initialization accuracy (still validated as excellent above).
    - **Diagnostic-print bugfix (2026-07-25, found by user re-running the
      check and asking why `max|delta psi - initial guess|` still printed
      `~0.559` -- the exact same value as the very first test above)**: this
      print's own "initial guess" snapshot (`Kokkos::deep_copy(psi_before,
      delta_psi)`, taken at the very top of `InitializeMetric()`'s
      `init_freeze_conserved` branch) was silently wrong -- `delta_psi` is
      still at its CONSTRUCTOR value (zero) at that point on a fresh run;
      the actual pgen-seeded initial guess (`psi4^0.25-1`, from `psi_
      seeded_`'s one-time block in `SolveConformalFactor`) only gets written
      into `delta_psi` *inside* `RunXPsiSolvePass` -- i.e., AFTER the
      snapshot. So the print was really reporting "the converged delta_psi's
      own magnitude vs. zero" (which is `~0.559` at the star's center simply
      because that's the correct physical value of `psi-1` there -- NOT a
      sign of any mismatch with the analytic solution; the actual accuracy
      check, `psi^4` vs. analytic via `plot_slice.py`, already separately
      confirmed `ratio 0.9997-1.0000` for this exact run). Fixed by
      recomputing the true initial guess directly from `adm.psi4` (the same
      formula `psi_seeded_`'s block uses) instead of snapshotting `delta_
      psi` itself. Purely a diagnostic-print fix -- no change to `Run
      XPsiSolvePass`, `SolveConformalFactor`, or any actual solve logic, so
      the already-recorded accuracy numbers above are unaffected.
    - **Primitive-recovery bug fix (2026-07-26, found by user while reviewing
      item 33's regrid design, but present here independently of AMR)**:
      `InitializeMetric()`'s tail always called `pdyngr->PrimToConInit(...)`
      ("final refresh") before `RescaleMatterSources`/`SolveLapse` read
      `pmhd->w0`/`u0` -- correct for the iterative Picard branch (primitives
      ARE the fixed quantity there, so re-deriving `u0` from them against the
      final metric is right), but backwards for `init_freeze_conserved=true`:
      that mode's entire premise is holding CONSERVED variables (`Utilde`)
      fixed through the Newton solve, so the physically correct step
      afterward is to recover PRIMITIVES from that fixed conserved state via
      `ConToPrim` (con2prim) -- not rebuild `u0` again from stale primitives.
      The old code silently (a) never updated `w0` at all, staying at
      whatever the pgen set (or whatever it last was), and (b) *overwrote*
      `u0` -- the very quantity this mode is supposed to hold fixed -- with a
      value inconsistent with what `psi`'s own Newton solve actually assumed,
      using those stale primitives against the new metric.
      `RescaleMatterSources`/`SolveLapse` then ran against this corrupted,
      internally-inconsistent triple. Confirmed the normal per-stage task
      graph already gets this right: `dyn_grmhd.cpp`'s `MHD_C2P` task
      (`ConToPrim`, a plain virtual method on the `DynGRMHD` base class) runs
      *after* `CFC_SolvePsi` and *before* `CFC_RescaleSrc` every stage,
      providing exactly this reconciliation -- `InitializeMetric()`'s own
      hand-rolled sequence, which deliberately bypasses that task graph, had
      no equivalent call.
      **Fix**: extracted the tail into a new method,
      `CFC::RunLapseShiftAssemblePass(Driver*, bool primitives_are_fixed)` --
      `true` (iterative branch) keeps the existing `PrimToConInit` call;
      `false` (`init_freeze_conserved=true`, and now also used by item 33's
      regrid re-solve) calls `ConToPrim` instead. `InitializeMetric()` now
      calls `RunLapseShiftAssemblePass(pdriver, !cfc_init_freeze_conserved_)`.
      (This signature was simplified shortly after -- see item 35 -- the
      `bool` parameter was removed and each caller now calls `PrimToConInit`/
      `ConToPrim` itself before invoking the now-parameterless
      `RunLapseShiftAssemblePass`/`RunXPsiSolvePass`; the reconciliation
      logic described here is unchanged, only its packaging.)
      **Verified**: rebuilt cleanly; reran a `nlim=1`, `init_freeze_
      conserved=true` t=0 check (`cfc_migration_freezecons_check_v2/`, job
      249528) -- no FATAL, no NaN, one-shot pass reports `max|delta psi -
      initial guess| = 0.000492109` (small, physically reasonable for this
      compact star), and `.mhd.hst`'s baryon mass at `t=0` (`1.534989`)
      matches the TOV solver's own printed `Baryon mass: 1.53499` to 6
      significant figures, with no discontinuity into the first evolved
      cycle -- consistent with the fix, though a direct "did `w0` actually
      change relative to the pgen's raw primitives" comparison was not done
      (would need an added diagnostic; the structural correctness of
      `ConToPrim` replacing `PrimToConInit` is clear from code review, and
      this run's clean, sane behavior is consistent with it working).
      **Connection to item 30's mystery -- CONFIRMED (2026-07-26, re-tested):**
      item 30's `tlim=200` migration-test run used `init_freeze_conserved=
      true` at `t=0` and found an unexplained post-bounce density collapse /
      mass jump, never root-caused. This bug (stale primitives + a silently-
      altered `t=0` conserved state) IS the root cause -- see item 30's own
      updated "Update" bullet for the full re-test result
      (`cfc_migration_freezecons_tlim200_fixed/`, job 249531): with the fix,
      the exact same `tlim=200` setup shows clean, physically sensible
      "migrate and ring down" oscillation throughout, with baryon mass
      conserved to `~7e-7` relative error the entire run -- no collapse, no
      mass jump. Item 30's own anomaly is resolved.

30. **`tlim=200` dynamical-evolution test of `init_freeze_conserved=true` on
    the migration test: excellent through the first bounce, but a separate,
    unresolved issue appears afterward (2026-07-25/26). RESOLVED 2026-07-26,
    see this item's own "Re-test result" bullet below -- root cause was
    item 29's primitive-recovery bug, now fixed.**
    - **Motivation**: item 29 only validated the one-shot mode's `t=0`
      metric-initialization accuracy (`nlim=1`). The user asked for a real
      dynamical run (`tlim=200`, `cfc_migration_freezecons_tlim200/`) to see
      whether the mode's accepted primitive/metric mismatch stays small
      enough in practice once the star actually evolves, plus rho-max/
      density diagnostics reusing the approach already established for BU8
      (`plot_bu8_diagnostics.py`/`make_bu8_density_movies.py`).
    - **Setup notes**: this run needed two fixes unrelated to item 29 itself
      before it would even complete: (1) `mg_threshold=mg_poisson_
      threshold=1.0e-8` (inherited from an unrelated, much earlier diagnostic
      thread in this same input file) tripped `MultigridDriver::
      SolveIterative`'s 40-iteration cap during dynamical evolution around
      `t~71.7` -- the driver's own safety fallback (`pdriver->nlim = current
      cycle`) then gets misreported by `Driver::Finalize`'s end-of-run print
      as `"Terminating on wall clock limit"` regardless of the true cause
      (that printout is a simple `else` fallback after checking `ncycle==
      nlim`/`time>=tlim`, with no dedicated "convergence failure" case) --
      reverted to `3.0e-3`, item 22's own already-established working value
      for this test's dynamical evolution, which resolved it; (2) `sub.sh`'s
      `#SBATCH -o` output file is OVERWRITTEN (not appended) by each new
      `sbatch` submission, unlike the `.hst` files themselves (which the
      *program* opens in append mode across restarts) -- worth remembering
      for future multi-restart runs in this same style, since it means the
      `.out` file alone is NOT a reliable full run history once a job has
      been restarted more than once.
    - **Diagnostics used**: `plot_migration_diagnostics.py`/
      `make_density_movie.py` (this test's own pre-existing, purpose-built
      scripts in `cfc_tov_migration_test/`, reused directly rather than the
      BU8 scripts verbatim -- this star has full spherical symmetry, so one
      slice plane suffices unlike BU8's rotation-induced axisymmetry, and
      `dyngr_tov.cpp`'s own `TOVHistory` already provides a native `rho-max`
      `.user.hst` column, unlike BU8's XNS-sourced pgen which needed a
      bin-file-scanning fallback at the time those scripts were written).
      Needed one further fix: `athena_read.hst` (the shared plotting
      utility) silently keeps only the MOST RECENT header/data segment when
      a `.hst` file has been appended across multiple restarts (of which
      this run had several, from the threshold-trip fix above), which would
      have discarded everything before the last restart -- added a local
      `read_hst_concat()` helper (in the copied script, not the shared
      utility) that stitches all segments together, dropping any row that
      doesn't strictly exceed the running max time (handles the small
      overlap a restart's own resumed checkpoint can produce).
    - **Result, `t=0` to `~0.5ms` (the star's first oscillation/bounce)**:
      excellent. Baryon mass agrees with the analytic value to `~0.03%` at
      `t=0` and drifts by only another `~0.03%` through the bounce;
      `rho_c(t)/rho_c(0)` rises smoothly to a peak of `~1.86` around
      `t~0.31ms` then drops sharply as the star bounces -- exactly the
      qualitative migration signature the paper describes, with no NaN, no
      FATAL (after the threshold fix), and only the one already-explained
      convergence-cap trip (fixed, not a recurrence).
    - **Result, `t~0.5ms` onward -- OPEN, unresolved concern**: the density
      field visually collapses to a near-uniform atmosphere-floor value
      everywhere in the domain by `t~0.55ms` and stays that way through
      `t=200` (no visible star structure in the 2D slices at all), even
      though `rho-max` itself retains a real, non-floor residual (`~6e-5` at
      `t=200`, `~130x` below the initial central density -- far more than
      the paper's own factor-of-`~6` migration-to-stable-branch prediction).
      Coincident with this, the (densitized, domain-integrated) baryon mass
      JUMPS `~14%` (`1.535->1.757`) around `t~102` (code units, `~0.5ms`),
      tracking a sharp `dt` crash-then-recovery in the same window, then
      stays essentially perfectly constant (to 6 significant figures) for
      the remaining `t~104` to `200` -- i.e. not an ongoing instability,
      but a one-time, unexplained jump. `alpha-min` also drops to an
      extreme `~2.2e-3` by `t=200`. No `NANS_IN_CONS`/con2prim-error
      messages appear in this run's own log around that time (`dyn_error=
      reset_floor` may reset silently without printing, so this doesn't
      rule out a floor-related cause).
    - **Conclusion -- explicitly NOT attributed to item 29**: this pattern
      (mass jumping then becoming suspiciously exactly constant, density
      visually vanishing, lapse going extreme) does not look like a clean
      "migrate to the stable branch and oscillate" outcome, and is most
      likely a separate, pre-existing numerical/resolution issue -- leading
      hypothesis: the star's post-bounce expansion outrunning the *static*
      refined region (fixed at `[-7.5,7.5]`, sized for the ORIGINAL compact
      star, never adjusted for an expanded/disrupted state), possibly
      combined with a floor/con2prim-reset consistency issue during the
      violent bounce itself. This is downstream of, and independent from,
      item 29's own `t=0` initialization accuracy (still validated as
      excellent, see item 29's own numbers) -- item 29 is not implicated,
      but this remains a genuinely OPEN question for follow-up (not yet
      root-caused): does widening/deepening the static refined region
      change this outcome; is there a silent floor-reset injecting mass;
      does the same issue appear under the default iterative
      initialization mode too (not yet checked, would help isolate whether
      this is initialization-related at all).
    - **Artifacts**: `cfc_migration_freezecons_tlim200/
      migration_freezecons_diagnostics.png` (mass/rho_c vs. time, full
      `t=0`-`200` history), `migration_freezecons_density_xy.mp4` (density
      animation).
    - **Update (2026-07-26): RESOLVED.** item 29 gained a new bullet
      documenting a real primitive-recovery bug in `InitializeMetric()`'s
      tail, specific to `init_freeze_conserved=true` (the mode this run
      used) -- `w0` was never updated post-solve and `u0` was silently
      corrupted relative to what `psi`'s Newton solve actually assumed. This
      reopened the "not attributed to item 29" conclusion above as a
      question worth re-testing.
    - **Re-test result (2026-07-26): CONFIRMED as the root cause.** Re-ran
      this exact `tlim=200` setup (`cfc_migration_freezecons_tlim200_fixed/`,
      same `parfile.par` as the original run -- `init_freeze_conserved=true`,
      `mg_threshold=mg_poisson_threshold=3.0e-3`, static refinement -- job
      249531, 16 ranks, completed cleanly to `t=200` on "Terminating on time
      limit," zero `FATAL`/NaN messages anywhere in the log) with the
      `ConToPrim` fix applied. Result: **the anomaly is gone.** `rho-max`
      stays within `1.0`-`1.0067` of its `t=0` value for the ENTIRE run (was:
      collapse to `~6e-5`, `~130x` below initial, by `t~0.55ms`) --
      `plot_migration_diagnostics.py`'s own diagnostic plot
      (`migration_freezecons_fixed_diagnostics.png`) shows a clean,
      physically sensible oscillatory "migrate to the stable branch and ring
      down" pattern for the full `t=0`-`~1ms` physical duration, matching the
      paper's own expected qualitative behavior -- not the visually-vanished-
      star catastrophe the buggy run showed. Baryon mass relative error
      stays at `~6.8e-7` throughout, several orders of magnitude tighter than
      before, with **no `~14%` jump** anywhere (was: `1.535->1.757` around
      `t~102`). `alpha-min` stays at a physically sane `~0.274` minimum (was:
      an extreme `~2.2e-3`). This confirms the primitive-recovery bug (item
      29's own fix) was indeed the root cause of this item's entire anomaly
      -- not the static-refined-region-not-tracking-expansion hypothesis
      floated in the original "Conclusion" above, which turns out not to
      have been needed as an explanation after all. Item 30 is now CLOSED.

31. **Persistent scratch buffers for `u_plus_2s`/`u_alpha_psi6`
    (performance fix, 2026-07-25).**
    - `SolveLapse` (`u_tilde + 2*s_tilde`) and `BuildShiftSource`
      (`alpha*psi^-6`) each used to construct a fresh, full-mesh-extent
      `DvceArray5D<Real>` scratch array -- a real device allocation -- on
      EVERY call (i.e. every RK stage, every cycle). Both are now persistent
      `cfc::CFC` members (`u_plus_2s`, `u_alpha_psi6`, declared in `cfc.hpp`
      alongside every other intermediate field this class owns), allocated
      once in the constructor (`Kokkos::realloc`, same mesh-`NGHOST`-depth
      sizing as `u_tilde`) and fully overwritten (not read-before-written)
      at the top of each call, exactly like every other persistent
      scratch/intermediate array in this class already works. `BuildShift
      Source`/`BuildShiftSourceImpl` (file-local free functions in
      `cfc.cpp`) gained an explicit `DvceArray5D<Real> &ap6` parameter so
      the caller (`AssembleVectorSource`) can pass `u_alpha_psi6` in,
      instead of the function allocating its own local `ap6`.
    - Pure performance change (avoids repeated device-memory allocation/
      deallocation churn every stage/cycle) -- no intended difference in
      numerical behavior, since both arrays are always fully overwritten
      before being read either way.

32. **CFC's own member arrays are not wired into AthenaK's dynamic-AMR
    regrid path -- open, pre-existing gap, safe under every current test
    (2026-07-25, found in a separate session, recorded here for
    completeness). RESOLVED 2026-07-26, see item 33 -- the fix actually
    implemented there is NOT the field-by-field remap this item's own
    "How to apply" bullet suggested; see item 33 for the design that was
    used instead and why.**
    - `cfc::CFC`'s persistent member arrays (`u_x`, `u_beta`, `delta_psi`,
      `u_tilde`, `u_p_x`, `coarse_u_x`, etc.) are sized once in the
      constructor directly against `nmb_thispack`, with no `std::max(
      nmb_thispack, pmesh->nmb_maxperrank)` headroom the way `hydro.cpp`/
      `z4c.cpp` size their own arrays. `RedistAndRefineMeshBlocks`
      (`src/mesh/mesh_refinement.cpp`) -- the function AthenaK's dynamic-AMR
      path calls to remap `phydro`/`pmhd`/`pz4c`/`padm` arrays across a
      block-count change -- never references `pcfc` anywhere (confirmed via
      direct grep: zero `pcfc`/`cfc::` matches in that file).
    - The four `MultigridDriver`-based solvers CFC owns (`pmgd_pietax`,
      `pmgd_pietabeta`, `pmgd_psi`, `pmgd_alpha`) already have a working
      per-block-count AMR resize hook (`Multigrid::ReallocateForAMR()`/
      `MultigridDriver::PrepareForAMR()`) -- it's specifically `cfc::CFC`'s
      OWN physics-facing arrays (declared directly in `cfc.hpp`, not owned
      by a `Multigrid`/`MultigridDriver` subobject) that have no analogous
      hook.
    - **Why this hasn't bitten anyone**: every existing CFC test input uses
      `refinement = static` (block count fixed at problem setup,
      `AdaptiveMeshRefinement`/`RedistAndRefineMeshBlocks` never invoked
      during the run) -- `cfc_bu8_stability.athinput` itself already flags a
      `refinement=adaptive` test as unfinished follow-up work, so this isn't
      a surprise regression, just a genuinely open item nobody has picked up
      yet.
    - **How to apply**: static/fixed mesh refinement (SMR set once at
      problem start) is completely unaffected and safe to keep using as-is
      -- this only matters before ever running CFC with true `refinement =
      adaptive` (block count changing mid-run). Before attempting that: (1)
      mirror hydro/z4c's `std::max(nmb_thispack, pmesh->nmb_maxperrank)`
      constructor sizing for CFC's own arrays, and (2) hook `pcfc` into
      `RedistAndRefineMeshBlocks`'s remap path the same way `phydro`/`pz4c`/
      `padm` already are (see those blocks for the pattern to follow:
      `DerefineCCSameRank`, `CopyCC`/`CopyForRefinementCC`/`RefineCC`, gated
      on `!= nullptr`).

33. **Dynamic-AMR support for `cfc::CFC` -- implemented and verified working
    (2026-07-26), closing item 32.**

    Item 32's own suggested fix ("hook `pcfc` into `RedistAndRefineMeshBlocks`'s
    remap path the same way `phydro`/`pz4c`/`padm` already are") turned out to
    be the wrong shape for CFC specifically -- CFC's ~14 top-level arrays
    (`delta_psi`, `u_x`, `u_p_x`, etc.) are all Newton-solve outputs / linear-
    solve outputs, not genuinely-evolved conserved state the way `phydro->u0`/
    `pz4c->u0` are, so field-by-field `DerefineCCSameRank`/`CopyCC`/
    `CopyForRefinementCC`/`RefineCC` wiring for all 14 of them would have been
    substantial, never-before-exercised new code for comparatively little
    benefit. The design actually implemented instead mirrors `adm::ADM::
    SetADMVariables` -- the existing escape hatch for metric fields not worth
    restricting/prolonging field-by-field: re-derive the metric from a cheaper,
    already-correctly-remapped source (here, `padm->adm.psi4`/`adm.alpha` plus
    the primitives) rather than trying to carry CFC's own stale arrays across
    the regrid.

    **Design** (mechanical part, confirmed via 3 parallel Explore agents plus
    direct reads of `driver.cpp`/`mesh_refinement.cpp`):
    - `src/cfc/cfc.cpp`'s constructor: `int nmb = pmy_pack->nmb_thispack;`
      became `int nmb = std::max(pmy_pack->nmb_thispack, pmy_pack->pmesh->
      nmb_maxperrank);` -- a one-line fix (mirrors `hydro.cpp`/`z4c.cpp`/
      `adm.cpp`'s own idiom) that gives every one of CFC's ~20 constructor-
      allocated arrays (14 primary + 6 `coarse_*` + the `mg_nghost`-depth
      `u_p_src`) AMR headroom in one shot, since they all key off this single
      local variable.
    - New public method `CFC::ReinitializeMetricForAMR(Driver *pdriver)`
      (`cfc.hpp`/`cfc.cpp`): originally reset the one-time `psi_seeded_`/
      `alpha_psi_seeded_` flags to `false`, then called `InitializeMetric(
      pdriver)` verbatim (same Picard loop / `init_freeze_conserved` one-shot
      mode, whichever `<cfc>` already selects). **This was corrected shortly
      after (same day) -- see the "mass-conservation correction" bullet
      below; `ReinitializeMetricForAMR` no longer calls `InitializeMetric()`
      at all.** Resetting the seed flags (still done, in the corrected
      design) is the crux of the warm-start half of this design: CFC's
      linear `X^i`/`beta^i` (`P_i`/`eta`) solves don't care about their
      initial guess for correctness (a multigrid V-cycle for a well-posed
      linear elliptic PDE with `mg_zerofixed` BCs converges to the same
      unique solution regardless of starting point -- a stale guess only
      costs a few extra V-cycles), but the two NONLINEAR Newton solves
      (`psi`, `alpha*psi`) genuinely do -- a bad starting point risks Newton
      overshoot into `psi<=0` (`psi_floor_` is a last-resort catch, not a
      substitute for a sane start). Forcing the reseed makes
      `SolveConformalFactor`/`SolveLapse` re-seed from `adm.psi4`/`adm.alpha`,
      which -- unlike CFC's own `delta_psi`/`delta_alpha_psi` -- IS properly
      `CopyCC`'d across a regrid for blocks staying on the same rank
      (confirmed: `RedistAndRefineMeshBlocks`'s `else if (padm != nullptr) {
      CopyCC(padm->u_adm); }` branch at Step 6 fires for every CFC run, since
      CFC never constructs `pz4c`) -- see item 34 for a real limitation of
      this found later (cross-rank moves).
    - `src/mesh/mesh_refinement.cpp`: `MeshRefinement::AdaptiveMeshRefinement`
      calls `pmbp->pcfc->ReinitializeMetricForAMR(pdriver)` right after
      `pdriver->InitBoundaryValuesAndPrimitives(pmy_mesh)` and before any
      `NewTimeStep` priming -- mirroring `Driver::Initialize()`'s own t=0
      ordering exactly (primitives must be valid before the metric solve reads
      them; the metric solve must run before the post-regrid dt estimate,
      since GR timestep bounds read the lapse).
    - Also in `mesh_refinement.cpp`: Step 11's existing `padm->
      SetADMVariables(pm->pmb_pack)` call -- which already fires
      unconditionally on every regrid for any `pz4c==nullptr && padm!=nullptr`
      run, i.e. every CFC run today -- is now gated with an added
      `&& (pm->pmb_pack->pcfc == nullptr)`. This closes a real, previously
      undiscovered hazard found while implementing this item: CFC pgens
      (`dyngr_tov.cpp`/`xns_rotstar.cpp`) set `SetADMVariables` to a callback
      that re-derives the metric from the pgen's **static t=0** analytic/
      tabulated initial data -- correct for the genuinely time-independent
      backgrounds this callback mechanism was built for (Kerr-Schild, FLRW,
      etc.), but for a CFC run it would silently discard all of CFC's
      dynamical metric evolution since t=0 the moment AMR ever regridded. This
      was latent (never triggered, since no CFC test used `refinement=
      adaptive` before this item), not something introduced by this change --
      but it had to be fixed as part of enabling dynamic AMR at all, since
      `ReinitializeMetricForAMR` is what replaces its role for CFC runs.

    **A second, deeper bug found and fixed while verifying the above** (not
    anticipated by the design; found via bisection with temporary `std::cerr`
    checkpoints plus `addr2line` against the actual crash backtrace, not
    guesswork): the very first smoke test (small octant TOV setup, 8 root
    MeshBlocks, `<mesh_refinement> refinement=adaptive`, a `min_max` density
    criterion tuned to refine the block containing the star's core) crashed
    with `Kokkos::View ERROR: out of bounds access ... extents [1,2,10,10,10]`
    the moment the regrid-triggered `ReinitializeMetricForAMR` ran.
    `addr2line` against the actual binary resolved the crash to
    `MGCFCConformalFactorDriver::LoadMatterSource`'s host-side `par_for`
    lambda. Root cause: `LoadMatterSource`/`LoadNonlinearCoefficient`
    (`mg_cfc_conformal_factor.cpp`) and `LoadReactionCoefficient`
    (`mg_cfc_lapse.cpp`) all use `pmy_pack_->nmb_thispack` directly (the
    CURRENT, live block count) as their `par_for` loop bound when writing into
    `mglevels_`'s own finest-level `coeff_` array -- but `coeff_`'s actual
    allocated size is only kept in sync with the current block count inside
    `Multigrid::ReallocateForAMR()`, which is called from `Solve()` (via
    `PrepareForAMR()`). `cfc.cpp` calls `LoadMatterSource`/
    `LoadNonlinearCoefficient`/`LoadReactionCoefficient` **before** calling
    `Solve()` (`SolveConformalFactor`/`SolveLapse`'s existing call order,
    unrelated to this item) -- so the very first call to these functions after
    a regrid writes into `coeff_`'s still-stale (pre-regrid) allocation,
    reading/writing out of bounds. This is a genuinely pre-existing latent bug
    (not introduced by this item's own design above), just never triggered
    before since no CFC test previously changed `nmb_thispack` mid-run.
    `MGCFCVectorPoissonDriver::LoadPoissonSource` has the same "called before
    Solve()" shape, but its underlying `Multigrid::LoadSource()` reads its own
    cached `nmmb_` (not `pmy_pack_->nmb_thispack` directly) for both the loop
    bound and the array's own size, so it stays internally self-consistent --
    no crash, but a real correctness gap (newly-created blocks after a regrid
    would silently keep a zero source instead of the freshly-loaded one, until
    whatever next call happens to follow `Solve()`).

    **Fix**: added `mglevels_->ReallocateForAMR();` as the first line of all
    four functions (`LoadMatterSource`, `LoadNonlinearCoefficient`,
    `LoadReactionCoefficient`, `LoadPoissonSource`). `ReallocateForAMR()` is
    already idempotent (an early-return no-op whenever its cached `nmmb_`
    already matches `pmy_pack_->nmb_thispack`), so this is safe and cheap
    regardless of whether `Solve()` already ran this cycle.

    **Verification**: a dedicated new test,
    `/sakura/ptmp/tlam/athenak_run/cfc_amr_dynamic_check/` (octant-symmetric
    TOV star, `rhoc=1.28e-3`, 8 root MeshBlocks at very coarse resolution
    (`dx=1.6`), `<mesh_refinement> refinement=adaptive` with a `min_max`
    density criterion tuned to refine the star's core block on the very first
    check, `refinement_interval=0` to allow immediate refinement, `nlim=4`) --
    no existing CFC input exercises `refinement=adaptive` at all, so this is a
    from-scratch smoke test, not a resolution/accuracy study. After the fix:
    the run completes cleanly through `nlim=4` with a real regrid event (`7
    MeshBlocks created, 0 deleted by AMR`, `Current number of MeshBlocks =
    15`), `CFC::InitializeMetric` visibly re-fires and re-converges right after
    the regrid (`iteration 0: max|delta psi| = 5.3e-2` -> `iteration 2:
    0.0`, confirming `ReinitializeMetricForAMR` actually ran), and the
    reported baryon mass is continuous across the regrid event to ~7
    significant figures (`0.19141739284` immediately before -> `0.19141739271`
    immediately after) -- no discontinuity/glitch from the metric re-solve.
    Per-rank output (`srun --output=perrank/rank_%t.out`) confirmed zero
    `### FATAL ERROR` and zero crash signals on any of the 8 ranks; the only
    warnings present (`Error occurred in PrimToCons`/NaN-then-reset messages,
    `dyn_error=reset_floor` recovering) are pre-existing floor/atmosphere
    noise from this smoke test's deliberately tiny `dfloor=1e-10` at very
    coarse resolution -- already present in the very first pre-fix test run,
    unrelated to AMR, and non-fatal by design.

    **Mass-conservation correction (2026-07-26, same day, found by user
    review): `ReinitializeMetricForAMR` must NOT rebuild conserved variables
    from primitives.** The original design above called `InitializeMetric()`
    verbatim, which -- via `RunXPsiSolvePass`'s leading `PrimToConInit` call
    and the tail's own "final refresh" `PrimToConInit` -- rebuilds `pmhd->u0`
    from primitives at least twice. That's correct at true t=0 (primitives
    are the only data the pgen provides), but wrong at a regrid: `pmhd->u0`
    already holds genuinely evolved, mass-conserving conserved data, correctly
    remapped onto the new block layout by the standard hydro/mhd AMR machinery
    (`DerefineCCSameRank`/`CopyCC`/`CopyForRefinementCC`/`RefineCC`, already
    wired for `phydro`/`pmhd`). Rebuilding it from primitives would silently
    discard that -- the user's direction: hold conserved variables fixed, and
    do exactly one ordinary CFC step (the same sequence a normal per-stage
    evolution step already runs, which reads `pmhd->u0` directly and never
    touches `PrimToConInit`) to produce a self-consistent metric for the new
    mesh.
    - **Design**: `CFC::RunXPsiSolvePass` gained `bool refresh_cons_from_
      primitives = true` (skip its own leading `PrimToConInit` when false).
      `InitializeMetric()`'s tail was extracted into a new method,
      `CFC::RunLapseShiftAssemblePass(Driver*, bool primitives_are_fixed)` --
      the SAME method also created to fix item 29's `ConToPrim`-vs-
      `PrimToConInit` bug (see item 29's own bullet; both fixes share one
      refactor since they're the same underlying issue -- "which quantity was
      held fixed during this solve, and how to reconcile primitives/conserved
      afterward"). `ReinitializeMetricForAMR` at the time read:
      ```cpp
      void CFC::ReinitializeMetricForAMR(Driver *pdriver) {
        psi_seeded_ = false;
        alpha_psi_seeded_ = false;
        RunXPsiSolvePass(pdriver, /*refresh_cons_from_primitives=*/false);
        RunLapseShiftAssemblePass(pdriver, /*primitives_are_fixed=*/false);
      }
      ```
      (Superseded by item 35's signature simplification -- the two `bool`
      parameters were removed, with `ConToPrim`/the reseed logic called
      directly at this site instead -- and by item 36, which appends a final
      `pdriver->InitBoundaryValuesAndPrimitives(pmy_pack->pmesh);` call. See
      those items for the current, accurate form of this function.)
      No Picard loop, no `PrimToConInit` anywhere in this path -- `pmhd->u0`
      is read as-is, and primitives are recovered via `ConToPrim` afterward,
      exactly matching a normal per-stage `QueueCFCTasks()` step plus its own
      `MHD_C2P`. A `cfc_init_verbose_`-gated diagnostic print reports
      `max|delta psi - initial guess|` (the reseed formula recomputed
      directly from `adm.psi4`, same as `psi_seeded_`'s own block, per item
      29's diagnostic-print-bugfix precedent), since `InitializeMetric`'s own
      iteration print no longer fires for this path.
    - **A field-remap warm-start alternative was tried and reverted.** The
      original plan (per the user's own suggestion) was to ALSO directly
      `DerefineCCSameRank`/`CopyCC`/`CopyForRefinementCC`/`RefineCC` CFC's own
      `u_x`/`delta_psi`/`delta_alpha_psi` across a regrid, mirroring
      `z4c::Z4c::u0`/`coarse_u0`, instead of relying solely on the
      `psi_seeded_`/`alpha_psi_seeded_` reseed-from-`adm.psi4` mechanism.
      Implemented and smoke-tested -- this is what surfaced item 34's
      discovery (a real, reproducible NaN, root-caused to a SEPARATE,
      pre-existing gap in AthenaK's generic AMR load-balancing MPI transfer,
      unrelated to anything in this item's own design). Reverted entirely
      (mesh_refinement.cpp has zero net diff from before this addendum) in
      favor of the simpler `psi_seeded_`/`alpha_psi_seeded_` reseed, which
      -- despite depending on `padm->u_adm`, which carries the exact same
      cross-rank gap -- is safe in practice because CFC's own re-solve
      unconditionally overwrites every field in `padm->u_adm` every call
      anyway (`AssembleConformalMetric`/`AssembleLapseShiftK`/`AssembleADM`),
      so whatever it holds beforehand, right or wrong, is only ever used as a
      best-effort Newton starting point -- never doubled down on as this
      pass's own persistent state the way directly remapping CFC's own
      arrays would have been. See item 34 for the full writeup of the
      cross-rank gap itself.
    - **Verification**: rebuilt cleanly. The original multi-rank smoke test
      (`cfc_amr_dynamic_check/`, 8 ranks, 1 MeshBlock/rank) hit item 34's
      cross-rank NaN on its SECOND regrid event (the first, refine-only,
      happened to keep all children on the same rank and was clean; the
      second involved derefinement and cross-rank load-balancing moves). To
      verify this item's OWN logic in isolation from that separate bug, a
      single-MPI-rank variant (`cfc_amr_dynamic_check_1rank/`, `-N 1
      --ntasks-per-node=1`, same parfile) was run instead -- with only one
      rank, every regrid event is trivially "same-rank," sidestepping item
      34's gap entirely. Result: clean throughout `nlim=4`, baryon mass
      continuous across the (single) regrid event to 6 significant figures
      (`0.191417392843` -> `0.191459448143`, the small shift reflecting the
      star's own dynamics, not a discontinuity), `ReinitializeMetricForAMR`'s
      diagnostic print reports a sane `max|delta psi - initial guess| =
      4.055271e-02`, and the run terminates cleanly on the cycle limit with
      no FATAL/NaN anywhere. This confirms the mass-conservation redesign
      itself (no `PrimToConInit`, single CFC step, `ConToPrim` recovery) is
      correct; item 34's cross-rank gap is a genuinely separate, pre-existing
      problem, not something this item's own logic introduced or can fix.
    - **Known limitation (superseded)**: an earlier draft of this item noted
      "`ReinitializeMetricForAMR` reruns `InitializeMetric`'s full Picard
      loop on every regrid" as a performance concern with a suggested future
      fix (a cheaper, always-one-shot regrid mode). This is now moot: the
      corrected design above already does exactly one CFC step per regrid,
      unconditionally, with no Picard loop at all.

    Still open (at the time this item was written): whether `refinement=
    adaptive` is production-worthy for a real (non-smoke-test) CFC physics
    run given item 34's cross-rank gap -- this item establishes that CFC's
    own regrid logic is correct for same-rank regrid events, but a run whose
    load balancing moves blocks across ranks (an ordinary, expected
    occurrence for any long-running adaptive-AMR simulation, not a corner
    case) would silently hit item 34's NaN until that separate gap was
    fixed. **That gap is now closed, see item 37 (same day).**

34. **RESOLVED, see item 37: AthenaK's generic AMR load-balancing MPI
    transfer has no entry for `padm`/`pcfc` -- a block moving to a different
    rank during a regrid never has its ADM/CFC field data transferred at all
    (found 2026-07-26; fixed the same day -- see item 37, which supersedes
    this item's own "scope of a proper fix" bullet below with a different,
    user-directed design).**
    - **How it was found**: item 33's original field-remap warm-start attempt
      (directly `DerefineCCSameRank`/`CopyCC`/`CopyForRefinementCC`/`RefineCC`
      -ing `u_x`/`delta_psi`/`delta_alpha_psi` across a regrid, mirroring
      `z4c::Z4c::u0`) crashed with a real, reproducible `-nan` baryon mass on
      the SECOND regrid event of the `cfc_amr_dynamic_check` smoke test (8
      ranks, 1 MeshBlock/rank) -- the first regrid (refine-only, all children
      stayed on the same rank) was clean; the second (involving derefinement
      and load-balancing-driven cross-rank moves) was not. Reverting the
      field-remap code did NOT fix the NaN (confirmed empirically, job
      249526) -- proving the root cause is independent of that code and
      pre-existing.
    - **Root cause**: `src/mesh/load_balance.cpp`'s `MeshRefinement::
      PackAndSendAMR`/`ClearRecvAndUnpackAMR` -- the generic MPI pack/send/
      unpack mechanism used whenever a regrid moves a MeshBlock to a
      different rank -- compute how many cell-centered variables to transfer
      (`ncc_tosend`/equivalent on the unpack side) by explicitly summing
      `phydro->nhydro+nscalars`, `pmhd->nmhd+nscalars`, `prad->prgeo->
      nangles`, `pz4c->nz4c`. **Neither `padm` nor `pcfc` appears anywhere in
      this list.** Confirmed by reading both the pack side (`load_balance.
      cpp:390-405`) and the unpack side (`ClearRecvAndUnpackAMR`, same file --
      only `phydro`/`pmhd`/`prad`/`pz4c` unpacked). This means: for ANY
      `pz4c==nullptr, padm!=nullptr` run (i.e. every CFC run), a block that
      moves to a different rank during a regrid has its `padm->u_adm`
      (`psi4`/`g_dd`/`alpha`/`beta_u`/`vK_dd`) -- and, if wired in the future,
      any CFC-owned field -- left at whatever uninitialized/stale memory
      happens to occupy that array slot on the receiving rank, since the
      actual data is never sent. A Newton/multigrid solve cannot self-correct
      away from a NaN or wildly-out-of-range starting value (floating-point
      arithmetic doesn't "forget" NaN through further computation), so this
      silently poisons the solve rather than just slowing convergence the
      way a merely-stale-but-finite guess would.
    - **Why this has never been noticed before**: this MPI transfer path only
      ever runs when `RedistAndRefineMeshBlocks` moves a block across ranks
      during a regrid. Every CFC test before item 33 used `refinement=
      static` (regrid never happens at all). `padm->u_adm`'s own gap
      (same-rank-only `CopyCC`, confirmed in `mesh_refinement.cpp`'s Step 6 --
      no `Derefine`/`CopyForRefinement`/`Refine` treatment for `padm->u_adm`
      at all, unlike `pz4c->u0`) is even more exposed than CFC's own fields:
      it's missing BOTH the cross-rank MPI path AND the same-rank
      derefine/refine-prolongation treatment that `pz4c->u0` gets. It hasn't
      caused visible problems in dynGRMHD-without-CFC contexts either,
      presumably because those configurations use `padm->SetADMVariables` to
      fully re-derive the metric from an analytic/tabulated formula on every
      regrid (Step 11, `mesh_refinement.cpp`) -- completely overwriting
      whatever `CopyCC` (or the lack of any cross-rank transfer) left behind,
      making the gap harmless THERE. For CFC specifically, that same Step-11
      callback is now deliberately skipped (item 33's own `pcfc == nullptr`
      gate), and CFC's `ReinitializeMetricForAMR` only re-seeds `psi`/
      `alpha*psi`'s Newton starting point from `adm.psi4`/`adm.alpha` -- it
      does not touch `g_dd`/`beta_u` at all before the next per-stage
      `MHD_C2P`/flux calculation might read them, so a garbage cross-rank
      value CAN leak through to places item 33's own re-solve doesn't
      overwrite.
    - **Scope of a proper fix (not attempted here)**: extending
      `load_balance.cpp`'s packing/unpacking to include `padm->u_adm` (and,
      if ever wired, CFC's own fields) -- this is core, shared AMR
      infrastructure used by every physics module, not something scoped to
      `src/cfc/`. A minimal fix would need: (1) add `padm`'s own variable
      count to `ncc_tosend`/the unpack side's equivalent counter, (2) pack/
      unpack `padm->u_adm` alongside `phydro`/`pmhd`/`prad`/`pz4c` in both
      functions, (3) decide whether `padm->u_adm` also needs the same-rank
      `Derefine`/`CopyForRefinement`/`Refine` treatment `pz4c->u0` gets
      (currently missing entirely, Step 6 only does `CopyCC`) -- for CFC runs
      specifically this may be lower priority than the MPI fix, given
      `ReinitializeMetricForAMR`'s own re-solve overwrites `psi4`/`alpha` (but
      not `g_dd`/`beta_u`, `vK_dd`/other unwritten fields) every regrid
      regardless.
      **(Superseded -- see item 37: rather than adding `padm`-specific
      blocks to the generic machinery as sketched here, `pcfc` was instead
      given its own independent `u_adm` array, mirroring `pz4c->u0`, with
      `adm::ADM` aliasing onto it -- so the actual fix touches all three of
      the bullets above, just via a `pcfc` block instead of a `padm` one.)**
    - **How to apply**: static refinement (every current CFC test) is
      completely unaffected -- this only matters for `refinement=adaptive`
      once a regrid actually triggers a cross-rank block move, which will
      happen for any sufficiently long adaptive-AMR run (load balancing is
      the whole point of the mechanism). Item 33's own dynamic-AMR work is
      verified correct and safe for same-rank regrid events only (see item
      33's own single-rank verification). **(Resolved by item 37 -- see that
      item for verification that a cross-rank regrid no longer produces a
      NaN metric.)**

35. **`RunXPsiSolvePass`/`RunLapseShiftAssemblePass` simplified (their `bool`
    parameters removed, moved to call sites); full-module comment sweep
    (2026-07-26). Committed `9f3ae81a`.**
    - **Signature simplification**: both methods (introduced by items 29/33
      above to share the primitives-vs-conserved reconciliation logic between
      `InitializeMetric()`'s two modes and `ReinitializeMetricForAMR`) lost
      their `bool refresh_cons_from_primitives`/`bool primitives_are_fixed`
      parameters. The decision each parameter used to encode -- whether to
      call `PrimToConInit` before `RunXPsiSolvePass`, and whether to call
      `PrimToConInit` or `ConToPrim` before `RunLapseShiftAssemblePass` -- is
      now made explicitly at each of the three call sites instead (the
      iterative Picard loop, the `init_freeze_conserved` one-shot branch, and
      `ReinitializeMetricForAMR`), rather than threaded through as an argument
      to the shared helper. Verified functionally equivalent by tracing all
      three call sites against the original branch semantics: no logic change,
      pure refactor. (Items 29/33's own text above, including their inline
      code blocks showing the old signatures, were annotated to point here
      rather than silently rewritten -- see the parenthetical notes added to
      each.)
    - **Comment sweep**: every file in `src/cfc/` (`cfc.{hpp,cpp}`,
      `cfc_reconstruct.{hpp,cpp}`, `mg_cfc_conformal_factor.{hpp,cpp}`,
      `mg_cfc_lapse.{hpp,cpp}`, `mg_cfc_vector_poisson.{hpp,cpp}`) had its
      comments condensed to concise, non-obvious-WHY-only notes -- net
      -879 lines (1716 removed, 837 added as condensed replacements) across
      the module. Removed: dated investigation narrative ("2026-07-21",
      "Round 16 fix", "Finding C", "Item 12", "plan addendum #4" -- that
      history lives in git log/this file's own numbered items already, not
      redundantly in source comments), ~130 lines of dead fully-commented-out
      code (a reverted relative-change convergence experiment, duplicated in
      both nonlinear solvers -- see item 20's own note that this was tried
      and reverted), restated-signature `\fn` lines, and verbose blow-by-blow
      debugging narrative in the temporary diagnostic functions (`Debug*` in
      `mg_cfc_conformal_factor.cpp`, trimmed to just what each one does).
      Kept: physics/equation references, sizing/indexing invariants that
      would silently break code if violated, sign/formula derivations, and
      the one genuinely load-bearing safety warning (item 34's
      `load_balance.cpp` cross-rank AMR gap, at the time still called out in
      `ReinitializeMetricForAMR`'s own doc comment -- that CAUTION note was
      since removed as part of item 37's fix, once the gap it warned about
      was actually closed). Verified zero logic
      changes by diffing every file's non-comment tokens against the prior
      commit (`git diff` stripped of `//`-comments and whitespace) -- confirmed
      byte-identical except for the signature simplification above. Both
      `build_cfc`/`build_cfc_xns` rebuilt cleanly.
    - **This file (`DEVELOPMENT.md`) is explicitly out of scope for that
      sweep** -- it's a deliberate running journal of this module's
      development history, a different kind of document from source
      comments, not something the comment-cleanup convention applies to.

36. **Hydro ghost cells left stale after CFC metric (re)initialization --
    fixed (2026-07-26).**
    - **Motivation**: the user asked whether hydro variables get ghost-
      exchanged after CFC's metric initialization, and correctly predicted a
      real bug. `RunLapseShiftAssemblePass`'s two callers each reconcile
      hydro state right before calling it (see items 29/33/35): the default
      (Picard) mode calls `PrimToConInit(is, ie, js, je, ks, ke)` --
      **interior-only** (confirmed via `dyn_grmhd.cpp`: a thin wrapper around
      `eos.PrimToCons` with whatever range is passed) -- so `pmhd->u0`'s
      ghost cells are simply never refreshed after CFC's solve.
      `init_freeze_conserved`/regrid mode calls `ConToPrim(pdriver, 0)` --
      full array **including ghosts** (confirmed via `dyn_grmhd.cpp`) -- but
      this runs *before* `RunLapseShiftAssemblePass`'s own tail finishes
      ghost-exchanging `padm->u_adm` (`pbval_adm`'s `Rest/Send/Recv/Prolong`
      sequence, the last thing that function does), so it computes
      primitives in ghost cells using metric-ghost values that are stale
      relative to the solve that just converged. Neither `Driver::
      Initialize()` nor `MeshRefinement::AdaptiveMeshRefinement` re-invokes
      any hydro Send/Recv/ConToPrim between CFC's call returning and the next
      step (main loop / `NewTimeStep` priming) -- confirmed via direct reads
      of both.
    - **Why this is scoped to init/regrid, not the per-stage path**: the
      normal per-stage task graph has the same "one-stage-lag" shape
      (`MHD_C2P` depends on `{MHD_Prolong}`/optionally `{CFC_SolvePsi}`, not
      on the later `CFC_ProlongADM`) -- but that's an accepted, designed
      tradeoff during ongoing evolution (last stage's ghost metric is a
      reasonable fallback, refreshed by *this* stage's own `CFC_ProlongADM`
      before next use). At t=0/regrid there is no such valid prior state to
      fall back on (t=0: the pgen's raw guess; regrid: a possibly wrong
      pre-regrid block layout), so only `InitializeMetric()`/
      `ReinitializeMetricForAMR` needed this fix.
    - **Fix**: both functions now call
      `pdriver->InitBoundaryValuesAndPrimitives(pmy_pack->pmesh);`
      immediately after their own `RunLapseShiftAssemblePass(pdriver);` call
      -- the exact same function AthenaK already calls once *before* CFC's
      solve (`Driver::Initialize()`/`MeshRefinement::AdaptiveMeshRefinement`,
      both call it right before invoking CFC) to establish a consistent
      hydro state, reused here to re-establish it against the now-final
      metric. Confirmed safe to call a second time: it does not call
      `padm->SetADMVariables` (that lives in a different function,
      `MeshRefinement::RedistAndRefineMeshBlocks`, which runs *before*
      `InitBoundaryValuesAndPrimitives` is even invoked during a regrid, so
      it's unreachable from here) -- important, since item 33 deliberately
      gates that callback off for CFC runs (it would discard CFC's dynamical
      metric evolution otherwise). Placed in each of the two top-level
      functions rather than inside the shared `RunLapseShiftAssemblePass`
      itself, per user direction: keeps that helper's responsibility scoped
      to the metric-solve tail only (matching its existing name/doc comment),
      and both callers already handle their own separate post-processing
      individually (e.g. each has its own `cfc_init_verbose_` diagnostic
      print) rather than pushing everything into the shared tail.
    - **A rejected first draft**: hand-assembling the equivalent
      `MHD::InitRecv`/`RestrictU`/`SendU`/`RecvU`/`ApplyPhysicalBCs`/
      `Prolongate`/`ConToPrim`/`ClearSend`/`ClearRecv` sequence directly
      (mirroring dyn_grmhd's own per-stage task graph) was designed first,
      then caught by a design review before implementation: calling
      `InitRecv`/`ClearSend`/`ClearRecv` with `stage=0` would **deadlock** in
      a multi-rank run -- those three unconditionally post/wait on
      flux-buffer MPI requests whenever `stage>=0`, with no matching send
      ever issued outside a real RK sub-stage (`Driver::
      InitBoundaryValuesAndPrimitives` itself uses `stage=-1` specifically to
      suppress this). Reusing `InitBoundaryValuesAndPrimitives` wholesale
      sidesteps the whole bug class rather than re-deriving its own correct
      `stage` handling by hand.
    - **Verified**: rebuilt `build_cfc`/`build_cfc_xns` cleanly. Ran a
      16-rank, 288-MeshBlock, 5-AMR-level smoke test
      (`cfc_migration_freezecons_check_v3/`, job 249562, reusing the
      established `init_freeze_conserved=true` migration-test check --
      exercises exactly the `ConToPrim` path this fix targets): no hang/
      deadlock (completed in ~13s wall-clock, matching prior runs of this
      same check), no FATAL/NaN, `InitializeMetric`'s own diagnostic print
      unaffected (`max|delta psi - initial guess| = 0.000492109`, consistent
      with prior runs). This clears the one real risk the design review
      flagged. A direct before/after diff of a MeshBlock-boundary-adjacent
      `w0`/`u0` value (to demonstrate the actual ghost-cell content change,
      not just "no crash") was not done -- the mechanism is understood from
      code reading and the fix is low-risk (reuses an already-proven
      function verbatim), but this would be a worthwhile follow-up
      confirmation if the anomaly this fix targets is ever suspected of
      causing a visible symptom in a production run.
    - **Scoping note**: for the regrid path specifically, this fix was
      necessary but not sufficient for a fully-correct multi-rank result --
      item 34's cross-rank AMR-transfer gap meant `padm->u_adm` itself could
      still be wrong on a cross-rank regrid, independent of this fix. That
      gap is now closed by item 37 (same day); this fix still separately
      closes the hydro-ghost-staleness gap for same-rank regrids and for the
      t=0 `InitializeMetric()` path (which has no such cross-rank caveat).

37. **Item 34's cross-rank AMR-transfer gap closed -- `adm::ADM` gained its
    own `coarse_u_adm`, and `padm->u_adm` is now wired directly into the
    generic AMR pipeline as its own independent physics-module block
    (2026-07-26).**
    - **Design history**: a first attempt at this fix (same day, since
      superseded) had `cfc::CFC` own an independent `u_adm` array mirroring
      `pz4c->u0`, with `adm::ADM` aliasing onto it via a `Kokkos::View`
      assignment -- requiring `pcfc`'s construction to move ahead of
      `padm`/`pz4c` in `MeshBlockPack::AddPhysics()`. The user asked to keep
      `u_adm` under `padm` instead, for structural clarity: `adm::ADM` is
      the single spacetime-agnostic container hydro/dyn_grmhd always read
      through regardless of which module drives the metric, so having it
      actually *own* the array (rather than alias `pcfc`'s) is more legible,
      and avoids the construction-reorder machinery entirely. This item
      describes the final, `padm`-owns-it design that replaced it.
    - **What stayed unchanged throughout**: `AssembleConformalMetric`/
      `AssembleLapseShiftK` (`cfc_reconstruct.cpp`) -- the only places CFC
      writes the solved metric -- write exclusively through
      `pmy_pack->padm->adm.{psi4,g_dd,vK_dd,alpha,beta_u}`, the
      `AthenaTensor` shallow-slice views, never through `u_adm` raw indices
      directly. That write path never changed in either design attempt; only
      where the *backing* array lives, and how the generic AMR machinery
      transfers it, changed.
    - **`adm::ADM` gains `coarse_u_adm`** (`coordinates/adm.hpp`/`adm.cpp`),
      mirroring `hydro::Hydro::coarse_u0`/`z4c::Z4c::coarse_u0`'s existing
      declaration/allocation idiom exactly: a `multilevel`-gated
      `Kokkos::realloc(coarse_u_adm, nmb, u_adm.extent_int(1), ...)`, using
      the same coarse-cell (`cnx1`/`cnx2`/`cnx3`) sizing convention as those
      two. Sized from `u_adm.extent_int(1)` (its own actual channel count --
      `nadm`=17, or `nadm-4`=13 when `pz4c` aliases `alpha`/`beta_u` out of
      it), not a hardcoded constant, so it stays correct in the z4c+ADM
      combination too, not just CFC's own `pz4c==nullptr` case. `ADM::ADM()`
      itself needed no other changes -- still the original two-branch
      `pz4c == nullptr` / `else` structure, unchanged from before this
      session's work.
    - **Generic AMR pipeline wiring**: added an independent
      `if (padm != nullptr) {...}` block, mirroring how `phydro`/`pmhd`/
      `prad`/`pz4c` are each their own independent block (not attached to
      any other module's conditional), at: `mesh_refinement.cpp`'s
      constructor buffer pre-sizing; Steps 5 (`DerefineCCSameRank`), 7
      (`CopyForRefinementCC`), and 9 (`RefineCC(..., true)` -- same
      `HighOrderProlongCC` interpolation `pz4c->u0` gets, structurally the
      right choice for metric-like data) of `RedistAndRefineMeshBlocks`.
      Step 6's existing same-rank index-copy fallback chain
      (`if (pz4c != nullptr) {...} else if (padm != nullptr) {...}`) needed
      no change at all -- it already had a `padm` branch from before this
      session. And in `load_balance.cpp`: `InitRecvAMR`'s and
      `PackAndSendAMR`'s variable counting, `PackAndSendAMR`'s pack calls,
      and `ClearRecvAndUnpackAMR`'s unpack calls -- all using
      `padm->u_adm.extent_int(1)` for the variable count, matching
      `coarse_u_adm`'s own sizing (see below) -- this code path is now
      `pcfc`-gated (see the narrowing bullet below) so it never actually
      runs in the 13-channel z4c+ADM configuration in practice, but there's
      no reason to hardcode `17` in place of the already-correct runtime
      lookup. `#include "coordinates/adm.hpp"` (previously missing entirely)
      added to `load_balance.cpp`.
    - **Narrowed to `pcfc != nullptr`, not left unconditional on
      `padm != nullptr`** (a same-day follow-up correction, prompted by the
      user asking whether this transfer is actually needed when z4c is
      active). Read `Driver::InitBoundaryValuesAndPrimitives`
      (`driver.cpp:603-687`) directly to check -- it contains, at lines
      662-669:
      ```cpp
      if (pdyngr == nullptr) {
        (void) pmhd->ConToPrim(this, 0);
      } else {
        if (pz4c != nullptr) {
          (void) pz4c->ConvertZ4cToADM(this, 0);
        }
        (void) pdyngr->ConToPrim(this, 0);
      }
      ```
      (This snippet shows the code as it read at the time this bullet was
      written -- superseded by item 38, which restructures this same
      dispatch further to also fold `CFC::ReinitializeMetricForAMR` in here;
      see that item for the current form.) This function is called once by
      `AdaptiveMeshRefinement()` right after every regrid, before any
      per-stage task graph runs again. When `pz4c != nullptr`, it already
      calls `ConvertZ4cToADM` itself,
      immediately before `ConToPrim` -- so there is no window at all where a
      stale/untransferred `padm->u_adm` could be read for a z4c run: the
      metric is refreshed synchronously, inside this same function, using
      `pz4c->u0` (which has its own, already-established AMR transfer,
      confirmed via `Z4c::Z4cToADM`, `z4c_adm.cpp:201-236`, to be a purely
      pointwise conversion covering the *full* ghost-inclusive array, no
      stencil/derivative needing neighbor data). For bare stationary ADM
      (`pz4c == nullptr && pcfc == nullptr`), `padm->SetADMVariables` already
      ran inside `RedistAndRefineMeshBlocks`'s own Step 11 -- *before*
      `InitBoundaryValuesAndPrimitives` is even called -- and that callback
      is a pure function of grid coordinates (analytic/tabulated, e.g.
      Kerr-Schild), independent of whatever `u_adm` held pre-regrid. **CFC is
      the only configuration where this transfer is load-bearing**: it falls
      into the same `pdyngr != nullptr` branch above (CFC also builds
      `pdyngr`) but `pz4c == nullptr`, so nothing in
      `InitBoundaryValuesAndPrimitives` refreshes the metric for it --
      `CFC::ReinitializeMetricForAMR` (the function that does) is
      deliberately called *after* `InitBoundaryValuesAndPrimitives` returns
      (it needs primitives already valid on the new mesh first), so CFC's
      one post-regrid `ConToPrim` call has no earlier refresh to fall back
      on. Every `padm`-gated block above was accordingly narrowed from
      `if (padm != nullptr)` to `if ((padm != nullptr) &&
      (pcfc != nullptr))` -- mirroring Step 11's own complementary gate
      (`(pz4c==nullptr) && (padm!=nullptr) && (pcfc==nullptr)`). This
      supersedes the "side effect" this item originally claimed (that the
      unconditional wiring would also incidentally fix the z4c+ADM/bare-ADM
      configurations' own pre-existing gap) -- those configurations were
      never actually broken in the first place, so there was no gap for the
      unconditional version to have fixed; the transfer for them would have
      been pure overhead with zero effect on correctness, since it's
      unconditionally overwritten (z4c) or never read before being
      overwritten (stationary) regardless.
    - **`pcfc`'s own module-specific arrays stay deliberately unmapped**
      (`u_x`, `u_beta`, `u_tilde`, `u_p_x`, `u_p_beta`, etc.), per item 33's
      own design -- `ReinitializeMetricForAMR`'s full Poisson re-solve every
      regrid is what stands in for that; only `padm->u_adm` needs to cross
      correctly, since it's the one thing every other module and CFC's own
      next solve pass read as a genuine algebraic input.
    - **Verified**: rebuilt `build_cfc` cleanly (no warnings; touches
      `adm.hpp`, a widely-included header, so triggers a broad rebuild) both
      before and after the `pcfc`-gating narrowing above. Single-rank
      re-run of the existing same-rank AMR smoke test
      (`cfc_amr_dynamic_check_1rank`-style; unconditional version, fresh copy
      `cfc_amr_dynamic_check_1rank_padmfix`, job 249676; `pcfc`-gated version,
      fresh copy `cfc_amr_dynamic_check_1rank_gated`, job 249694): both
      clean, no regression, identical diagnostic (`max|delta psi - initial
      guess| = 9.844341e-02`). Re-run of the exact 8-rank/4-node scenario
      that produced item 34's original confirmed NaN
      (`cfc_amr_dynamic_check`-style; unconditional version, fresh copy
      `cfc_amr_dynamic_check_padmfix`, job 249677; `pcfc`-gated version,
      fresh copy `cfc_amr_dynamic_check_gated`, job 249695): both produce the
      identical result -- confirmed bug gone (zero metric-NaN anywhere), and
      the same narrower, non-fatal residual (zero, never NaN, ghost cells on
      a cross-rank-transferred new block during the first of two post-regrid
      `InitBoundaryValuesAndPrimitives` calls, self-healing by the second
      call per item 36) reproduces identically across all three variants
      (`pcfc`-owned, `padm`-owned unconditional, `padm`-owned `pcfc`-gated)
      -- confirming both that it's a pipeline-ordering artifact independent
      of ownership/gating design, and that narrowing the gate to
      `pcfc != nullptr` is a true no-op for CFC's own path (byte-identical
      diagnostics and NaN counts across all three verification rounds).
    - **Not investigated further here**: whether the same ghost-cell
      transient (a *different* concern from the `ConvertZ4cToADM`-ordering
      question resolved above -- this one is about whether `pz4c->u0` itself
      might have not-yet-exchanged ghosts the first time something reads it
      after a regrid, the same category item 36 documents for hydro fields)
      also affects z4c runs -- out of scope for a CFC-focused session with
      no z4c+AMR test input on hand, flagged here since the underlying
      pattern is generic, not CFC-specific.
    - **This residual is substantially (not fully) closed by item 38** --
      see that item for the fold-in fix that eliminates the pipeline-ordering
      transient described above, and for the narrower, distinct residual it
      exposed once that larger issue was out of the way.

38. **`CFC::ReinitializeMetricForAMR` folded into `Driver::
    InitBoundaryValuesAndPrimitives`, mirroring `z4c::Z4c::ConvertZ4cToADM`'s
    placement -- closes most of item 37's residual transient (2026-07-26).**
    - **User-proposed redesign**: item 37's residual (a cross-rank-
      transferred new block's ghost cells reading a zero/stale metric during
      the *first* of two post-regrid `InitBoundaryValuesAndPrimitives`
      calls) exists because CFC's regrid refresh was called as a separate
      step entirely *after* `InitBoundaryValuesAndPrimitives` returns --
      unlike z4c's `ConvertZ4cToADM`, called *inside* that same function,
      right before `ConToPrim`. The user proposed folding CFC's refresh into
      the same slot, splitting its own con2prim need into an interior-only
      pass (right after solving X^i/psi) and a ghost-only pass (right after
      solving lapse/shift, once `padm->u_adm`'s own ghost exchange -- already
      happening at that pass's own tail -- completes), removing
      `ReinitializeMetricForAMR`'s own recursive call to
      `InitBoundaryValuesAndPrimitives` entirely (previously needed to avoid:
      calling the new fold-in location from *inside*
      `InitBoundaryValuesAndPrimitives` while `ReinitializeMetricForAMR`
      itself still called that same function at its tail would be
      unconditional infinite recursion).
    - **Verified against the actual code before implementing** (not assumed):
      `RunXPsiSolvePass`'s only conserved-variable read
      (`AssembleVectorSource(false)`, `cfc.cpp:1025`) and
      `RunLapseShiftAssemblePass`'s only primitive read
      (`RescaleMatterSources`, `cfc.cpp:1194`) are both confirmed
      **interior-only** (`is..ie,js..je,ks..ke`), confirming an interior-only
      con2prim pass between them is sufficient. `padm->u_adm`'s own ghost
      exchange (`RestADMTask`/`SendADMTask`/`RecvADMTask`/`ProlongADMTask`
      quartet) is confirmed the **last data-affecting step** of
      `RunLapseShiftAssemblePass` (`ClearTailFields()` right after it only
      tears down send/recv bookkeeping). The range-restricted con2prim
      primitive needed for both passes **already existed and was already
      public/virtual**: `DynGRMHD::ConToPrimBC(is,ie,js,je,ks,ke)`
      (`dyn_grmhd.hpp:89`), already used today by `DynGRMHD::
      ApplyPhysicalBCs` with restricted (thin-band) bounds -- so **no
      changes were needed to `dyn_grmhd.hpp`/`.cpp` at all**; both the
      interior-only call and a new 6-slab full-ghost-shell decomposition
      (mirroring `ApplyPhysicalBCs`'s own boundary-strip pattern but with
      full ghost width instead of a thin band) are implemented entirely
      within `cfc.cpp`, reusing the existing primitive. One genuine gap
      found and fixed as part of this: nothing let
      `Driver::InitBoundaryValuesAndPrimitives(Mesh*)` distinguish being
      called from `Driver::Initialize()` (t=0, where `CFC::InitializeMetric`'s
      different Picard-iteration algorithm is needed instead) from being
      called during a regrid -- fixed by adding a `bool is_amr_regrid =
      false` parameter, defaulted `false` at the t=0 call site
      (`driver.cpp:316`, unchanged) and passed `true` at the regrid call site
      (`mesh_refinement.cpp`).
    - **Implementation**: `driver.cpp`'s MHD/dyn_grmhd dispatch
      (`driver.cpp:662-669` before this item) restructured to
      `if (pz4c != nullptr) {...} else if ((pcfc != nullptr) &&
      is_amr_regrid) { pcfc->ReinitializeMetricForAMR(this); } else {
      pdyngr->ConToPrim(this, 0); }` -- the final `else` covers both
      CFC-at-t=0 (`is_amr_regrid` false there) and any other
      `pdyngr`-without-`pz4c`-without-`pcfc` case, identical behavior to
      before for both. `mesh_refinement.cpp`'s separate
      `pcfc->ReinitializeMetricForAMR(pdriver)` call (previously right after
      its own `InitBoundaryValuesAndPrimitives(pmesh)` call) removed --
      folded into that same call now, via `is_amr_regrid=true`.
      `CFC::ReinitializeMetricForAMR` restructured: kept
      `RunXPsiSolvePass`/`RunLapseShiftAssemblePass` unchanged; replaced the
      old full-array mid-solve `pdyngr->ConToPrim(pdriver,0)` with
      `pdyngr->ConToPrimBC(is,ie,js,je,ks,ke)` (interior-only); replaced the
      old tail's `pdriver->InitBoundaryValuesAndPrimitives(...)` call with
      the new 6-slab ghost-shell decomposition (6 `ConToPrimBC` calls with
      full-ghost-width bounds -- verified by hand to sum to exactly
      interior + full ghost shell, no gaps/double-counting, matching
      `n1*n2*n3` for the test geometry used below).
    - **Verified**: rebuilt cleanly (no warnings; touches `driver.hpp`, a
      widely-included header). t=0 `InitializeMetric` convergence
      (`cfc_amr_dynamic_check_1rank`/`cfc_amr_dynamic_check`-style, fresh
      copies `..._1rank_foldin`/`..._foldin`, jobs 249696/249697) is
      byte-identical to every prior run (`0.192087` -> `0`, confirming
      `is_amr_regrid` correctly stays `false` there and this fold-in doesn't
      touch t=0 at all). The regrid diagnostic
      (`max|delta psi - initial guess|`) is also byte-identical to every
      prior verification round on both tests (`9.844341e-02` 1-rank,
      `5.261418e-02` 8-rank) -- confirms the con2prim-call restructuring
      doesn't change CFC's solve result, only when/how primitives get
      recomputed around it. The 8-rank cross-rank scenario's residual
      `NANS_IN_CONS` count dropped from 3584/rank (6 ranks affected, item
      37's baseline) to **896/rank (only 3 ranks affected)** -- clean ranks
      went from 2-of-8 to 5-of-8. A real, substantial improvement, not a
      regression.
    - **Not fully eliminated -- a narrower, distinct residual remains**: the
      remaining 896 errors per affected rank are all confined to a single
      4-cell-wide ghost slab (one face) on the affected block, still
      `detg=0`. Checked against the parfile's domain (`x2 in [0,25.6]`) --
      the affected coordinates (`y~14.8`) are nowhere near a physical
      boundary, ruling out a BC-application gap. This is very likely a
      **separate, pre-existing gap in `padm->u_adm`'s own ghost-exchange
      completeness** (the `RestADMTask`/`SendADMTask`/`RecvADMTask`/
      `ProlongADMTask` quartet, untouched by this item) for a specific
      neighbor-topology case -- plausibly a coarse-fine interface on one
      face of a newly-refined child block, since this run's single regrid
      event does create such children. **Not investigated further this
      session** (explicit user decision: document and stop here) -- this
      item's own fold-in is confirmed working exactly as designed and
      verified; the remaining residual is a distinct bug in a different
      mechanism (u_adm's neighbor ghost exchange itself, not the
      pipeline-ordering issue this item targeted) and would need its own
      separate investigation (start by checking whether the affected
      block's j-high neighbor is on a different refinement level, and
      whether `ProlongateCC`'s coarse-fine treatment of `u_adm` -- as
      opposed to same-level `RecvAndUnpackCC` -- is where the gap lives).
      **Correction (item 42): this "ruling out a BC-application gap" check
      only verified the y-coordinate against the domain -- it never checked
      x1. The actual corrupted cell sits at `i=0`, the block's own physical
      `x1=0` reflect boundary. See item 42 for the real mechanism (`u_adm`
      had no physical-BC pass at all before item 41) and the direct
      cell-by-cell trace confirming it.

39. **(2026-07-27, updated 2026-07-28) Item 38's residual: root cause
    confirmed, fix identified, but not yet safe to apply.** A deep,
    multi-session-length investigation (extensive runtime instrumentation --
    targeted `printf` probes bracketing every stage of the ghost-exchange
    pipeline, added and then fully removed each round -- not static reading
    alone) into item 38's residual (896/rank `NANS_IN_CONS`, 3 of 8 ranks,
    `detg=0`, confined to one ghost slab per affected block). **Current
    read, superseding the original framing below**: this is not "one fix
    blocked by two separate pre-existing bugs" -- 39d is a direct structural
    consequence of the fix itself (not pre-existing), and the actual
    registration rule that separates needed from spurious cases is now
    confirmed (39f: `recip == nullptr`), just not yet fully, correctly
    implemented (a residual formula bug in 3 of 4 diagonal sites, still
    unlocated). 39c (the MPI abort) remains a genuinely separate, unrelated,
    still-unexplained bug. See 39f/39g for the current state and a minimal,
    CFC-independent reproducer with confirmed physics-level evidence.

    - **39a. Real, scoped fix applied and kept** (superseded for `is_z4c`/
      `FillCoarseInBndryCC` specifically by item 43 -- `u_x` removed -- and
      item 44 -- fields switched to `is_z4c=true` and `FillCoarseInBndryCC`
      removed entirely; the same-level corner-padding gap this bullet fixes
      is still real and still needed for the `is_z4c=false` era it
      describes): CFC's six ghost-exchanged fields (`u_p_x`, `u_x`,
      `delta_psi`, `delta_alpha_psi`, `u_p_beta`, `padm->u_adm`) all use
      `is_z4c=false` (like Hydro/MHD), which means
      `ProlongateCC`'s stencil needs a `FillCoarseInBndryCC` call
      immediately beforehand to correctly populate the coarse scratch
      array's transverse padding at same-level-neighbor-adjacent corners --
      Hydro/MHD both call it (`hydro_tasks.cpp:378`, `mhd_tasks.cpp:527-528`),
      z4c doesn't need to (its `is_z4c=true` path fills the same role via an
      extra same-level payload baked into the send/recv step itself,
      `z4c_tasks.cpp:273`'s call is explicitly commented out for this
      reason). CFC was missing this call entirely on all six fields --
      added at all six `Prolong*Task` call sites in `cfc.cpp`, mirroring
      Hydro/MHD's exact placement. This is real, correct, and independent
      of everything below -- it fixes a same-level corner-padding gap, not
      item 38's residual (confirmed: it did not change the residual's
      count or locations when tested alone).

    - **39b. Item 38's residual root cause, confirmed directly**: the
      residual is exactly the `MeshBlock::SetNeighbors` octant-parity guard
      first suspected back in items 34-37, now confirmed via direct runtime
      tracing rather than inference. For every EDGE (2 of 3 offsets
      nonzero) and CORNER (all 3 nonzero) neighbor direction,
      `SetNeighbors` (`meshblock.cpp`, four sites: x1x2/x3x1/x2x3 edges and
      corners) only registers a **coarser** neighbor into the shared
      `nghbr` array when the local block's own octant parity is the
      "exterior" one for that diagonal -- an "interior octant" block's
      coarser diagonal neighbor is left at `gid=-1`, and every downstream
      consumer (`PackAndSendCC`/`RecvAndUnpackCC`/`ProlongateCC`, gated on
      `gid>=0`) silently no-ops, leaving that ghost region permanently at
      its zero-initialized value. Traced on the exact 8-rank scenario that
      produces the residual: the affected block (local index `m=1` on rank
      1, global id reassigned post-regrid to `gid=2`, refinement level 2 --
      *not* an original unrefined root block, gid numbering is fully
      reassigned across the whole tree after every regrid, do not assume
      gid identity survives a regrid) has its failing ghost region (`k=14,
      j=14`, both hi-side ghosts for `ng=4`) at the `(ox2=+1,ox3=+1)`
      diagonal. `FindNeighbor` correctly finds the true neighbor there
      (`gid=9`, level 1 -- genuinely coarser than `gid=2`'s level 2) and
      computes the correct target slot (`NeighborIndex(0,1,1,0,0)=46`), but
      the guard's condition (`nt->lloc_.level >= lloc.level ||
      (myox2==ox2 && myox3==ox3)`) evaluates false (`1>=2` is false;
      `gid=2`'s own octant parity `myox3=-1` doesn't match the needed
      `ox3=+1`) -- confirmed by printing the guard's own inputs and boolean
      result directly at the write site, and confirming no
      `nghbr.h_view(m,46)` write ever executes for this block. The doc
      comment's claim (`meshblock.cpp:136-137`) that interior-octant
      edge/corner neighbors are "redundant" with face neighbors and
      therefore skippable is false for this case: a face neighbor's own
      ghost fill only extends `ng` cells toward the shared *interior* seam,
      never reaching the block's *opposite* exterior ghost region a true
      diagonal neighbor is needed for.

      The fix (verified correct, matches the unconditional structure every
      FACE direction already uses, no octant-parity gate at all): delete
      the `if (...) {...}` wrapper at all four edge/corner sites in
      `SetNeighbors` and dedent the 4-line assignment body inside each, plus
      fix the false "redundant" doc comment. Confirmed via `bvals.cpp`/
      `buffs_cc.cpp` reading that no companion change is needed anywhere
      else -- index-range computation (`InitSendIndices`/`InitRecvIndices`)
      and buffer sizing (`BuildRankPackedVarMetadata`) are already
      unconditional across all 56 neighbor slots regardless of whether a
      slot is populated; only the registration gate itself is the bug.

    - **39c. Applying 39b's fix causes an MPI `internal_Waitall` abort on
      the 8-rank cross-rank scenario** (job aborts in ~6s, never completes a
      single cycle -- `Abort(17) ... Fatal error in internal_Waitall`,
      ranks 4-7). Not yet root-caused to the same rigor as 39b. Leading
      hypothesis (from static reading of `bvals.cpp`'s
      `BuildRankPackedVarMetadata`, not yet confirmed by direct tracing):
      `nghbr` tables are built completely independently per rank with *no*
      cross-rank consistency check before the one-shot header handshake
      (`bvals.cpp:264-286`) -- if rank A's and rank B's independent tree
      searches disagree even slightly about a newly-registered edge/corner
      neighbor pair's level classification (same vs. coarser vs. finer,
      which determines message size via `isame`/`icoar`/`ifine` index
      ranges), the resulting message-size mismatch would produce exactly
      this failure mode, and would plausibly only affect the subset of rank
      pairs where such a mismatch actually occurs (matching that only 4 of
      8 ranks abort, not all 8).

    - **39d. Applying 39b's fix ALSO causes a distinct, silent data-
      corruption bug on the 1-rank scenario** (never crashes; 1152 NEW
      `NANS_IN_CONS` reports appear where the test was previously always
      clean). **Root-caused directly** (2026-07-27, follow-up session):
      this is **not** a pack/unpack arithmetic bug and **not** a
      pre-existing bug independent of 39b -- it is a direct structural
      consequence of 39b's `dest`-slot computation, exposed only once the
      guard is removed.

      Symptom, confirmed by synchronized send-side/recv-side instrumentation
      (dumping every channel of `padm->u_adm`, `nvar=17` with z4c active,
      at both the `PackAndSendCC` write and the matching `RecvAndUnpackCC`
      read for one specific neighbor pair): block `m=11`'s same-level
      x2x3-edge neighbor `gid=9` (slot 42) packs a uniform, conformally-flat
      value (`1.06099595`) into every metric channel, but block 11 receives
      that correct value in only *some* channels (`PSI4`=v12, `ALPHA`=v13)
      while others (`GXX`=v0, `KXX`=v6) read back `1.12117774` -- a value
      `gid=9` never sent at all -- and still others (`GYY`=v3, `GZZ`=v5)
      read back zero.

      Root cause: a **second, unrelated sender is writing into the same
      receive slot.** `nghbr` topology dump (added to `SetNeighbors` itself,
      host-side, trivial to reproduce) shows a *third* block, `gid=6`
      (level 2, i.e. one level finer than block 11), registers block 11 as
      its own coarser x2x3-edge neighbor at its own slot `n=44` -- exactly
      the registration 39b's guard removal newly enables (`gid=6`'s own
      octant position, `myfx1=0, myfx2=1, myfx3=1`, was the "interior"
      octant the old guard used to reject). The `dest` field this
      registration computes, `NeighborIndex(0,-m,-l,myfx1,0)` with
      `myfx1=0`, evaluates to **42** -- landing on the *exact same slot*
      block 11 already legitimately uses for its unrelated same-level
      neighbor `gid=9`. Verified two ways: (1) hand-computed from the
      `NeighborIndex` formula directly (`nghbr_index.hpp`) -- `gid=6`'s own
      slot 44 decodes to offset `(m,l)=(-1,+1)`, so its `dest` is
      `NeighborIndex(0,+1,-1,0,0) = 42`, matching the runtime dump exactly;
      (2) the topology dump shows block 11 has **no** entry anywhere in
      slots 40-47 for `gid=6` via its *own* (unconditional, unaffected by
      39b) finer-neighbor registration branch -- meaning this relationship
      only exists because `gid=6` computed it unilaterally, with no
      cross-check against what block 11 already has claimed for that slot.
      `PackAndSendCC` then genuinely has two independent senders
      (`gid=9`'s same-level write and `gid=6`'s coarser-branch write, the
      latter using `ca`/`coarse_u_adm` data and a smaller box) both
      targeting `rbuf[42]` on block 11, racing/overwriting each other
      per-channel -- exactly reproducing the observed per-channel pattern
      (channels `gid=6`'s smaller box happens to overwrite come out wrong;
      channels it doesn't touch survive with `gid=9`'s correct value).

      **39c is NOT the same mechanism as 39d** -- ruled out empirically
      this session (see 39e below): a fix that fully resolves 39d and
      preserves item 38's fix still leaves 39c's MPI abort completely
      unaffected. 39c needs independent root-causing; the two are
      superficially similar (both appear once 39b's fix is applied) but
      are demonstrably different bugs.

    - **39e. Two candidate fixes designed and empirically tested this
      session (2026-07-27, later still) against three simultaneous
      requirements: fix 39d, don't regress item 38's original fix, don't
      introduce new corruption. Neither fully succeeds -- the results
      themselves prove the slot-collision problem is more fundamental than
      a local guard/condition can solve.** Both replace the coarser-neighbor
      branch's registration condition with a check computed via
      `ptree->FindNeighbor` from the *target* block's own logical location
      (fully local -- the block tree is globally replicated per rank, no
      MPI needed, so this works identically for same-rank and cross-rank
      pairs).

      - **Attempt 1, strict reciprocity**: only register if the target's
        own reciprocal search, enumerated the same way its *existing*
        unconditional finer-neighbor branch already would, actually
        recovers this exact block as one of its children. Result: **fixes
        39d completely** (1-rank test: 1152 -> 0 `NANS_IN_CONS`) but
        **reintroduces item 38's original residual in full** (8-rank test:
        back to 896/rank on 3 ranks). This proves item 38's originally-
        needed registration is *itself* non-reciprocal by this same
        criterion -- it is structurally the same kind of "finer block
        reached at an angle" relationship as the one causing 39d's
        collision, not a distinguishable "genuine diagonal" case. Reciprocity
        cannot separate "spurious" from "needed" because both are the same
        shape of relationship.

      - **Attempt 2, narrower collision-only skip**: register
        unconditionally (restoring 39b's coverage) *except* skip when the
        target already has a **different, competing same-level neighbor**
        at the reciprocal direction (the precise, narrow condition that
        actually caused 39d's corruption). Result: **fixes 39d** (1-rank,
        first sub-test: 0 `NANS_IN_CONS`) **and preserves item 38's fix**
        (8-rank: 0/rank, all 8 ranks clean) -- but a second, independent
        1-rank run with this fix produced **1448 NEW `NANS_IN_CONS`**,
        worse than 39d's original 1152. This means the collision space is
        bigger than "one degenerate registration vs. one same-level
        neighbor": **two or more independently-computed, non-reciprocal
        "finer block reached at an angle" relationships (like `gid=6`'s)
        can alias onto the same target slot with *neither* side being a
        same-level neighbor**, and this narrower check has no way to detect
        that case at all.

      **Conclusion**: this is not fixable with a per-relationship
      guard/condition in `SetNeighbors`, however it's phrased. The root
      issue is that AthenaK's edge/corner `NeighborIndex` scheme provides
      only 2 slots per diagonal direction (meant for "same-level, OR up to
      2 finer children split along the free axis"), but irregular AMR
      topology can genuinely produce *more than 2* distinct, simultaneously-
      needed contributors to a single coarser block's diagonal ghost region
      (a same-level neighbor filling most of it, plus one or more
      differently-positioned finer blocks reached indirectly through a
      face-neighbor's own refinement, filling the true corner-of-corner
      cells the same-level neighbor's own `ng`-wide box doesn't reach). No
      local, per-block check can safely arbitrate between multiple
      legitimate claimants on the same slot number. A real fix needs one of:
      (a) **extend slot capacity** -- give edge/corner directions enough
      slots to represent every genuinely-distinct contributor (a structural
      change to `nghbr_index.hpp`'s layout, `MeshBoundaryBuffer` allocation,
      and every place that assumes the fixed 56-slot/4-per-edge-direction
      layout); or (b) **global reconciliation pass** -- after every block's
      local registration is built, detect any slot with more than one
      legitimate claimant and explicitly resolve it (e.g. split the ghost
      region's index range between claimants by actual physical footprint,
      rather than letting both write the same flat buffer offset). Both are
      substantially bigger than the 4-guard-removal originally scoped for
      39b -- this needs dedicated design work, not a quick follow-up.

    - **39f. A fourth attempt (2026-07-28, follow-up session), aimed
      directly at implementing 39e's structural fix, found the precise rule
      distinguishing item 38's needed case from 39d's spurious one --
      confirmed empirically, not guessed -- but a full, robust
      implementation is still not achieved.** Reapplied 39b, instrumented
      every coarser-branch registration (all 4 sites: x1x2/x3x1/x2x3 edges,
      corners) to print, for each candidate, whether the *target* block's
      own direct diagonal query at the reciprocal direction resolves to
      `NULL` (nothing found), a same-level `LEAF`, or a `FINER` node.
      Re-ran the exact 8-rank scenario that originally produced item 38's
      residual and captured this data directly (not inferred) for the first
      time this session -- see `/sakura/ptmp/tlam/athenak_run/
      cfc_item38_topology_8rank`. Result: **every genuinely-needed
      registration had `recip=NULL`; every case matching 39d's known-
      redundant shape had `recip=LEAF`** (target already has an unrelated
      same-level neighbor there) **or `recip=FINER`** (target's own
      existing, unconditional finer-neighbor code already discovers and
      registers this exact relationship, making the coarser-branch
      registration purely duplicative). This gives a clean, confirmed rule:
      `do_register = (recip == nullptr)`.

      Implementing *just* this rule (all 4 sites) fixed 39d's exact original
      collision (confirmed directly: block 11's slot 43, where `gid=6`
      previously landed, dumped cleanly as unregistered) and preserved item
      38's fix (0/rank on the 8-rank test). But the 1-rank test regressed to
      **1344 new `NANS_IN_CONS`** -- worse than 39d's original 1152, at a
      *different* location than block 11 (confirmed: block 11's own slot 42
      dump was clean). Root cause identified from the SAME 8-rank topology
      dump: **two different finer blocks, reached via a target's two
      different face-neighbors, can independently compute `recip=NULL`
      simultaneously and land on the identical `dest` slot** -- confirmed a
      live instance in the captured data (`b_gid=1` and `b_gid=3`, both
      registering `target_gid=8`'s x3x1-edge slot with `recip=NULL`). Unlike
      the same-level case, these two candidates are **not** redundant with
      each other (two distinct, non-overlapping physical blocks cannot
      cover the same territory) -- dropping either loses real coverage.

      Added a deterministic tie-break (lower gid wins the natural slot, the
      higher gid's registration is dropped -- a real, narrower, honestly
      worse-than-ideal but non-corrupting compromise, pending the actual
      extra-slot-capacity work from 39e) to all 4 sites, with the geometric
      formulas for the "sibling candidate via the target's other
      face/edge-neighbor" derived and implemented per site. This did **not**
      resolve the 1-rank regression (1408 `NANS_IN_CONS`, same order of
      magnitude, after the tie-break) -- and the block-11 dump confirmed the
      corruption is **not** the original gid=6 case or the diagnosed P-vs-P'
      case at that location; it comes from an as-yet-unidentified third
      source, likely a sign/axis error in the x1x2-edge, x3x1-edge, or
      corner tie-break formulas (only the x2x3-edge formulas were directly
      empirically validated against the confirmed `gid=6` example -- the
      other 3 sites' formulas were derived by analogy and never independently
      checked against a known-good case). Reverted after this second
      regression rather than guess further; `meshblock.cpp` is clean,
      matches `HEAD`.

    - **39g. A minimal, CFC-independent test problem illustrating the bug
      mechanism directly.** `inputs/tests/lwave_hydro_diag_collision.athinput`
      -- plain Newtonian hydro (`linear_wave` pgen), no CFC/GR/elliptic
      solve at all, confirming this is generic AthenaK mesh code, not
      anything CFC-specific. Requires a build configured with
      `-DPROBLEM=built_in_pgens` (not the TOV-locked `build_cfc`) -- a new
      `build_generic` directory was created for this and any future
      non-CFC-locked test runs (mirrors the `build_generic` convention
      noted from the `proj/g-mode` branch's PR #748 work, recreated here
      since it didn't exist in this checkout).

      Uses `<refined_region1>` (parsed in `src/mesh/build_tree.cpp:80-133`,
      independent of any AMR criterion -- gives a fully deterministic tree,
      no adaptive-criterion guessing) to refine exactly one root block of a
      4x4x4 root grid one level finer. Confirmed directly via temporary
      `SetNeighbors` instrumentation (same style as 39b/39f's diagnostics,
      removed after use -- not a permanent code change) that this produces
      the exact bug mechanism: block `C` (root block `(1,1,1)`) has a
      genuine same-level x2x3-edge diagonal neighbor (slot 40, gid 1) via
      its own direct query; separately, 4 different children of the
      refined block (gids 7/8/9/10, reached via x1x2-edge, x3x1-edge,
      x2x3-edge, and corner directions respectively) each independently
      compute a coarser-neighbor registration targeting `C`, and **all 4
      are currently guard-blocked** (`guard_pass=0` for every one, dumped
      directly) -- this is item 38's original under-registration bug,
      concretely exhibited. The x2x3-edge case (gid 7) was confirmed to
      compute `dest` landing on slot 40 -- the exact slot `C`'s legitimate
      same-level neighbor (gid 1) already occupies -- confirming that
      removing the guard (39b) reproduces 39d's collision at this same,
      minimal, reproducible location.

      **Follow-up (2026-07-28, later): the physics-level symptom, with real
      numbers.** Added a temporary per-cell field probe (env-var-gated
      `printf` in `PackAndSendCC`/`RecvAndUnpackCC`, same technique as
      every other probe this investigation used, removed after) targeting
      `Hydro::u0` at `C`'s slot-40 ghost cell `(m=14, n=40, k=0, j=0, i=6)`,
      all 5 conserved-variable channels (`IDN=0, IM1=1, IM2=2, IM3=3,
      IEN=4`, `src/athena.hpp:65`). Ran the identical input twice, `HEAD`
      vs. 39b's guard-removal fix reapplied, same cell, same cycle (0):

      | channel | `HEAD` (correct) | 39b applied (corrupted) |
      |---|---|---|
      | IDN | 9.99043e-01 | 9.99121e-01 |
      | IM1 | 5.51961e-04 | 5.06893e-04 |
      | IM2 | 5.51961e-04 | **8.98682e-01** (~1630x larger) |
      | IM3 | 5.51961e-04 | 5.51961e-04 (frozen at `HEAD`'s own value) |
      | IEN | 8.98565e-01 | 8.98565e-01 |

      This is the exact channel-scrambling signature found throughout this
      whole investigation's original CFC work (`gid=6`/`gid=9`): some
      channels drift by a small, plausible-looking amount (IDN, IM1 --
      consistent with two legitimately-different physical sources blending
      via undefined write order), one channel takes on a wildly wrong
      value that looks like it belongs to a *different* channel entirely
      (IM2 landing near IEN's own magnitude), and one channel reads back
      exactly its `HEAD`-run value, i.e. never touched by this cycle's
      unpack at all. Confirmed via a parallel `PackAndSendCC` probe that
      `gid=7` (the colliding block) is genuinely sending real, different
      data at this cycle -- so this is a live collision artifact, not
      noise. This closes the gap flagged when 39g was first built:
      `inputs/tests/lwave_hydro_diag_collision.athinput` now has
      documented, reproducible evidence of the bug's effect on the actual
      physical solution, not just on `nghbr` bookkeeping.

      The probe instrumentation itself was temporary and has been removed
      (`git diff` of `bvals_cc.cpp`/`meshblock.cpp` is clean, matches
      `HEAD`) -- re-add it the same way (search this repo's history/this
      doc for the exact `printf` lines) if reproducing these numbers
      directly rather than trusting the table above. Re-verify the specific
      gid numbers (14, 40, 7, cell `(0,0,6)`) with a fresh topology dump
      before trusting them blindly if the input file, AthenaK version, or
      build configuration changes -- they depend on Z-order block
      numbering.

    - **Current state**: only 39a (`FillCoarseInBndryCC`) is applied and
      committed-ready. 39b's fix (the `SetNeighbors` guard removal, in any
      of the now 4 variants tried across two sessions) is *not* applied --
      `git diff` of `src/mesh/meshblock.cpp` is clean (matches `HEAD`). All
      SLURM job outputs proving each attempt's result are preserved under
      `/sakura/ptmp/tlam/athenak_run/cfc_recipfix*`, `cfc_item38_topology_
      8rank`, `cfc_recipnull_*`, `cfc_tiebreak_*` for reference. **Next
      session's highest-value next step**: the `recip==nullptr` rule (39f)
      is confirmed correct and should be kept as the foundation -- do NOT
      revisit 39e's reciprocity/collision-only variants, that space is
      closed. What remains is (1) find and fix the bug in the x1x2/x3x1/
      corner tie-break formulas (compare each site's derivation line-by-line
      against the x2x3-edge one that's confirmed correct, or instrument each
      site's own `sib_gid` computation the same way block 11 was dumped, on
      whichever block is producing the 1408-NaN residual -- not yet
      identified which site/location it is), and (2) once the tie-break is
      fully correct, still implement the actual extra-slot-capacity fallback
      (39e's design) rather than the "drop the loser" compromise, since the
      tie-break as implemented trades corruption for a real, silent
      coverage gap on the dropped candidate -- better than NaNs, but still
      not a complete fix. 39c remains completely unexplained and
      independent (confirmed again this session: it still aborts under
      every 39b variant tried, including 39f); needs its own root-causing
      from scratch. This is a real, generic AthenaK mesh-connectivity bug
      (not CFC-specific), so it is also latent for z4c's own puncture/BNS
      AMR tests once fixed and applied, per item 38's earlier z4c-exposure
      analysis. **A minimal, CFC-independent reproducer with both structural
      and physics-level evidence now exists (item 39g,
      `inputs/tests/lwave_hydro_diag_collision.athinput` +
      `build_generic`)** -- use it directly to verify the eventual structural
      fix, rather than reconstructing CFC's own TOV-star scenario each time.
      **Update (item 42)**: the CFC 8-rank TOV scenario's own visible
      `NANS_IN_CONS` symptom (used above to track this item) is now gone as
      of item 41 -- but for an unrelated reason (item 41 gave `padm->u_adm`
      its first-ever physical-BC pass, fixing a real but separate bug). This
      item's own registration bug is confirmed still present and unfixed;
      item 39g's reproducer remains the reliable way to verify (1)/(2) above,
      since the CFC TOV scenario no longer surfaces this item's residual on
      its own.

40. **(2026-07-28) Cherry-picked upstream (unmerged) PR
    [IAS-Astrophysics/athenak#748](https://github.com/IAS-Astrophysics/athenak/pull/748)
    ("Fix Refinement at Boundary for Z4c/MHD/Hydro/Radiation") -- confirmed,
    both by static analysis and by empirical rerun, that it is orthogonal to
    item 39 and does not fix it.** This PR was already cherry-picked onto a
    *different* repo/branch (`~/athenak`'s `proj/g-mode`, commit `e2a16544`,
    unrelated project) in an earlier session; this is the first time it's
    applied to `athenak_cfc`.

    **What the PR actually fixes**: physical boundary conditions were
    applied to a refined block's fine array *before* coarse-to-fine
    prolongation ran, so when an AMR fine/coarse junction coincided with a
    *physical* domain boundary, the fine ghost zone feeding prolongation's
    interpolation stencil was stale/unfilled, corrupting C2P there. Fix:
    new `HydroBCsCoarse`/`BFieldBCsCoarse`/`RadiationBCsCoarse`/
    `Z4cBCsCoarse` helpers apply BCs to the coarse array first, then
    prolongate, then BC the fine array -- reordered consistently across
    `Driver::InitBoundaryValuesAndPrimitives` (`driver.cpp`), the four
    physics modules' own task lists (`{hydro,mhd,radiation,z4c}_tasks.cpp`),
    and `dyn_grmhd.cpp`'s GR-MHD task queue.

    **Why it can't fix item 39, confirmed two ways**:
    - *Static*: the PR's diff (16 files) never touches
      `src/mesh/meshblock.cpp`, `nghbr_index.hpp`, `SetNeighbors`,
      `NeighborIndex`, or `FindNeighbor` -- the exact machinery item 39's
      `dest`-slot collision lives in. Item 39's own reproducer
      (`inputs/tests/lwave_hydro_diag_collision.athinput`, item 39g) uses
      `periodic` BCs on all 6 faces -- there is no physical boundary
      anywhere in that setup for this PR's fix to even engage.
    - *Empirical*: reran `lwave_hydro_diag_collision.athinput` via
      `build_generic` (rebuilt with this PR applied) with the same
      temporary `CFC_DEBUG_NGHBR`-gated `SetNeighbors` dump used in 39f/39g
      (added, used, then fully reverted -- `git diff` of `meshblock.cpp` is
      clean). Result, byte-for-byte identical to the pre-PR topology: block
      `C` (gid 14) still has its genuine same-level neighbor at slot 40
      (`b_gid=14 slot40 gid=1 lev=2 dest=46`), and `gid=7`'s x2x3-edge
      candidate still computes `guard_pass=0 inghbr=46 idest=40` -- the
      identical collision destination documented in 39g. The PR is a no-op
      for this bug, exactly as the static analysis predicted.

    **Applying it**: `git apply --3way` against this diff; 9 of 12 touched
    files applied cleanly, 3 conflicted with CFC's own prior work in the
    same files and were resolved by hand:
    - `src/bvals/bvals.hpp`: the PR's new `*BCsCoarse` declarations and the
      `Z4cBCs()` signature change (drops the now-redundant `coarse_u0`
      param, confirmed by checking `z4c_bcs.cpp`/`z4c_tasks.cpp`'s own
      already-cleanly-patched call site) simply coexist with CFC's existing
      `CFCScalarBCs`/`CFCVectorBCs`/`ADMBCs` declarations -- concatenated,
      no semantic conflict. (This PR does not add `*BCsCoarse`/reordering
      support for CFC's own fields, only Hydro/MHD/Radiation/Z4c -- item 41
      closes that gap with a hand-rolled parallel implementation, and along
      the way replaces `CFCScalarBCs`/`CFCVectorBCs` with a single merged
      `CFCBCs`/`CFCBCsCoarse`, so this bullet's function names are stale as
      of item 41.)
    - `src/driver/driver.cpp`: `Driver::InitBoundaryValuesAndPrimitives`'s
      Z4c/Hydro/Radiation blocks all had the PR's fix (`Prolongate` then
      `ApplyPhysicalBCs`) applied cleanly by `git apply`; only the MHD block
      conflicted, because CFC's own `cfc::CFC *pcfc = pm->pmb_pack->pcfc;`
      line sits immediately after this exact pair. Confirmed against
      pristine `HEAD` (`git show HEAD:...`) that MHD's block had the *same*
      buggy BC-then-Prolongate order the other three blocks did pre-patch
      -- resolved by applying the identical reordering to MHD too, then
      keeping the `pcfc` line unchanged immediately after.
    - `src/dyn_grmhd/dyn_grmhd.cpp`: the PR renames `MHD_C2P`'s required
      task dependency from `MHD_Prolong` to `MHD_BCS` (so C2P now correctly
      waits for the fine-array BC step, not just prolongation) in both the
      z4c-active and z4c-absent branches of `QueueDynGRMHDTasks`. CFC's own
      optional-dependency insertion (`CFC_SolvePsi` alongside `Z4c_Excise`)
      is orthogonal to which *required* dependency C2P waits on -- kept
      unchanged, just re-attached to the PR's corrected required-dep list.

    **Regressions checked, all clean**:
    - The PR's own new regression inputs (`tst/inputs/lwave_{mhd,z4c}_bc.athinput`,
      outflow BCs + AMR, exactly the scenario this PR targets), run directly
      via `build_generic`: MHD -- `eos_fail=0` across all 145 logged cycles
      (run stopped early on an internal wall-clock default since no `-t` was
      passed; sim time reached 0.68 of tlim=1.0, plenty of cycles to
      exercise the boundary); Z4c -- L-infinity error `2.36e-14` (vs. the
      `~1e-9` no-fix reference noted in the PR, tolerance `1e-12`) --
      matches the `~/athenak` cherry-pick's own result almost exactly.
    - CFC's own dynamic-AMR smoke test (`cfc_amr_dynamic_check_1rank_setneighbors`'s
      parfile -- octant-symmetric TOV, reflect BCs at the domain corner,
      `min_max`-triggered adaptive AMR), run via `build_cfc`: completed
      cleanly to `nlim=4` cycles, `ReinitializeMetricForAMR` fired
      post-regrid as expected, block count grew 8->15, no NaN/crash. This is
      the one case in this repo where CFC's own regrid hook and the PR's
      reordering are both genuinely active at once -- no interaction issue.
    - CFC's non-AMR TOV smoke input (`inputs/dyn_grmhd/cfc_tov.athinput`)
      segfaulted on `build_cfc` with this PR applied. **Confirmed
      pre-existing, unrelated to this cherry-pick**: built a throwaway
      pristine-`HEAD` baseline (`build_cfc_baseline`, since deleted) via
      `git stash`, reran the identical input -- byte-for-byte identical
      output (same repeated `MultigridDriver::SolveIterative "Failed to
      converge"` messages, same segfault at the same point after
      `Terminating on wall clock limit`, only wall-clock timing numbers
      differ). This matches the long history of "Failed to converge"
      churn already documented earlier in this file (items 15-29 area) --
      not something this PR introduced or need fix.

    **Current state**: applied and verified, staged for commit alongside
    this writeup. `git diff` of `src/mesh/meshblock.cpp` and
    `src/bvals/bvals_cc.cpp` are clean (all verification instrumentation
    reverted). Both `build_cfc` and `build_generic` rebuilt successfully
    with this PR applied. **Conclusion for future readers**: this PR is
    worth having on this branch (matters whenever a refined block borders a
    physical boundary -- relevant to future CFC AMR/BNS work), but it is
    *not* progress on item 39/40's own open problem (the tie-break formula
    bug in 3 of 4 diagonal sites, item 39f/39g's "Current state" bullet) --
    that remains exactly as open as before this cherry-pick.

41. **(2026-07-29) Reorder CFC's own ghost-exchange pipeline to match z4c's
    `RestrictU -> SendU -> RecvU -> Prolongate -> ApplyPhysicalBCs`; add
    `*BCsCoarse` helpers; merge `CFCScalarBCs`/`CFCVectorBCs` into one
    `CFCBCs`/`CFCBCsCoarse`; fix `delta_psi`/`delta_alpha_psi`/`u_x`'s
    physical-boundary falloff order.** Done, closes the gap item 40 flagged:
    upstream PR #748 reordered physical BCs relative to prolongation for
    Hydro/MHD/Radiation/Z4c, but never touched CFC's own 6
    `MeshBoundaryValuesCC` fields (`u_p_x`, `u_x`, `delta_psi`,
    `delta_alpha_psi`, `u_p_beta`, `padm->u_adm`) -- all 6 still applied
    physical BCs *inside* `Recv*Task`, i.e. *before* `Prolong*Task` ran, the
    same bug class PR #748 fixed for the other 4 modules. User asked for CFC
    to mirror z4c's task order exactly.
    - **Task reorder** (`src/tasklist/numerical_relativity.hpp`,
      `src/cfc/cfc.hpp`, `src/cfc/cfc.cpp`): added 6 new graph-visible tasks,
      `CFC_BCSPiEtaX`/`CFC_BCSX`/`CFC_BCSPsi`/`CFC_BCSAlphaPsi`/
      `CFC_BCSPiEtaBeta`/`CFC_BCSADM`, each depending on that field's own
      `CFC_Prolong*` (mirroring `z4c_tasks.cpp`'s `Z4c_BCS` depending on
      `{Z4c_Prolong}`) -- not folded into the tail of `Prolong*Task`, per
      user's explicit choice. Each field's `Recv*Task` lost its inline
      `if (tstat==complete && !periodic) { CFC*BCs(...); }` block (the
      `tstat` check is now redundant regardless, since the task-graph
      dependency chain `BCS* -> Prolong* -> Recv*` already guarantees `Recv`
      completed by the time `BCS` runs); the removed call moved verbatim
      into the new `BCS*Task`, gated only on `!strictly_periodic`. Four
      downstream task dependencies that used to point at `CFC_Prolong*`
      (`CFC_ReconstructX`, `CFC_ComputeADual`, `CFC_BuildSrcBeta`,
      `CFC_ReconstructBeta`) were rewired to point at the new `CFC_BCS*`
      instead, since they must now wait for BC'd-*and*-prolongated data, not
      just prolongated data -- a deliberate, intended behavior change (the
      converged value of `x_u`/`Adual^ij`/`beta_u` at a fine/coarse boundary
      near a physical edge can shift slightly, more correct under AMR).
      `padm->u_adm`'s stale doc comment (previously claiming no physical-BC
      pass existed for it at all, already contradicted by the code 20 lines
      below calling `ADMBCs`) was corrected in the same pass. The two manual
      (non-task-graph) call sequences, `RunXPsiSolvePass`/
      `RunLapseShiftAssemblePass` (used by `InitializeMetric`/
      `ReinitializeMetricForAMR`), got a `BCS*Task(pdriver, 0)` call inserted
      right after each corresponding `Prolong*Task` call, in the same order.
    - **Coarse-array BC pass** (`src/bvals/bvals.hpp`,
      `src/bvals/physics/cfc_bcs.cpp`, `src/bvals/physics/adm_bcs.cpp`):
      z4c's own `Prolongate` task applies physical BCs to the *coarse* array
      first (`Z4cBCsCoarse`) so the prolongation stencil doesn't read
      stale/never-written coarse ghost data at a physical boundary -- CFC
      had no equivalent for any of its 6 fields. Added
      `CFCScalarBCsCoarse`/`CFCVectorBCsCoarse` (later merged into
      `CFCBCsCoarse`, see below) and `ADMBCsCoarse`, following the existing
      `HydroBCsCoarse`/`BFieldBCsCoarse`/`RadiationBCsCoarse`/`Z4cBCsCoarse`
      naming precedent, called inside each field's `Prolong*Task` right
      before the existing `FillCoarseInBndryCC`/`ProlongateCC` pair (kept
      unchanged as of this writing -- unlike z4c, which skips
      `FillCoarseInBndryCC` entirely since its same-level-neighbor coarse
      data arrives via a dedicated MPI buffer CFC's own `MeshBoundaryValuesCC`
      objects don't use yet. **Superseded by item 44**: CFC's objects switch
      to `is_z4c=true` -- gaining that same MPI buffer -- and
      `FillCoarseInBndryCC` is removed from all five `Prolong*Task`s, exactly
      because keeping it turned out to be unsafe, not merely un-mirrored.
      `CFCBCsCoarse`/`ADMBCsCoarse` themselves are untouched by that change.)
      **Non-obvious correctness point**: `CFCScalarBCs`/`CFCVectorBCs`/
      `ADMBCs`'s `order>0` falloff branch computes a real physical radius via
      `CellCenterX(idx, indcs.nx1, x1min, x1max)` -- hardcoded to the *fine*
      cell count. A coarse-array call needs `indcs.cnx1/cnx2/cnx3` and
      `indcs.cis/cie/cjs/cje/cks/cke` instead, or the computed radius is
      wrong by the refinement factor (the coarse array covers the same
      `[x1min,x1max]` extent with half as many cells) -- confirmed by
      reading `CellCenterX`'s signature (`cell_locations.hpp:36-39`: second
      argument is the total interior cell count the extent is divided into).
      Both `cfc_bcs.cpp`'s `CFCBCsImpl` and `adm_bcs.cpp`'s (new)
      `ADMBCsImpl` were widened to take `is/ie/js/je/ks/ke/nx1/nx2/nx3/
      n1/n2/n3` as explicit parameters (mirroring z4c_bcs.cpp's
      `BCHelper<order>`, which didn't need this since it's coordinate-free),
      with thin fine/coarse public wrappers computing the right index/extent
      set. Confirmed via direct reads of `bvals/prolongation.cpp` that
      `FillCoarseInBndryCC`/`ProlongateCC` never touch physical-boundary
      ghost cells themselves (both gated on an actual neighbor `gid>=0`), so
      the new coarse-BC calls are neither redundant nor mistimed.
    - **Merge `CFCScalarBCs`/`CFCVectorBCs` into `CFCBCs`(`nvar`)**
      (`src/bvals/bvals.hpp`, `src/bvals/physics/cfc_bcs.cpp`,
      `src/cfc/cfc.cpp`) -- user's follow-up request, superseding item 19's
      earlier (still-two-function) merge. `u_p_x`/`u_p_beta` pack a 3-channel
      vector (`P_i`) and a trailing scalar (`eta`, channel 3) into one array,
      so every call site touching them called both functions back-to-back
      with the same `order` (once per part). Key insight enabling a true
      single call, not just a rename: the reflect-parity flip condition,
      `bool flip = (nvar==3) && (n==axis)` (`axis` = 0/1/2 per face), widens
      cleanly to `(nvar>=3) && (n==axis)` -- since `axis` is always 0/1/2,
      local channel index `n==3` (the packed `eta` channel) can never match
      it, so it falls through to "never flip" automatically, without a
      separate "how many leading channels are vector" argument. `nvar` alone
      (1 for a lone scalar, 3 for a lone vector, 4 for vector+packed-scalar)
      is therefore sufficient. `CFCScalarBCs`/`CFCVectorBCs`/
      `CFCScalarBCsCoarse`/`CFCVectorBCsCoarse` (4 functions) were replaced
      by `CFCBCs`/`CFCBCsCoarse` (2 functions, `nvar` as a required leading
      parameter); every call site updated, collapsing 2 calls into 1 at the
      `u_p_x`/`u_p_beta` sites.
    - **`order` consistency fix, also user-requested**: `delta_psi`/
      `delta_alpha_psi` (and, in a follow-up, `u_x`) previously called with
      the implicit default `order=0` (plain zero-gradient copy) at their
      outer (diode) physical boundary, while `P_i`/`eta` used `order=1`
      (`1/r^1` falloff). Since `psi`/`alpha_psi`'s ghost values are also
      outputs of an isolated-system elliptic solve (same category the file's
      own doc comment already calls out for the `order>0` treatment), and
      `X^i` is likewise a vector-Poisson-solve output, `order=0` for these
      was an inconsistency, not an intentional distinction -- all 3 fields
      (`u_p_x`, `u_x`, `delta_psi`/`delta_alpha_psi`, `u_p_beta`) now
      consistently pass `order=1`.
    - **Verified**: rebuilt `build_cfc` cleanly after each of the 3 passes
      above. Reran the existing octant-symmetric TOV dynamic-AMR smoke test
      (`cfc_amr_dynamic_check`-style parfile, reflect/diode BCs, `min_max`-
      triggered adaptive AMR) at both 1-rank and 8-rank, fresh each time
      (restart files cleared to force a real regrid rather than resuming
      past `nlim`): no crashes, NaNs, or `Kokkos_ENABLE_DEBUG_BOUNDS_CHECK`
      assertions in any of the 3 rebuild/rerun cycles. The 1-rank
      `max|delta psi - initial guess|` diagnostic
      (`CFC::ReinitializeMetricForAMR`) stayed bit-for-bit `9.844341e-02`
      and the 8-rank equivalent stayed `5.261418e-02` across all 3 passes --
      the reorder and the `CFCBCs` merge are confirmed behavior-preserving
      as designed, and the `order=0->1` fix did not detectably shift this
      particular coarse smoke test (plausible: its outer diode boundary
      sits far enough from the star that `psi`/`alpha_psi`/`X^i` are already
      near-zero there regardless of falloff order -- not evidence the fix is
      inert, just that this test doesn't probe it strongly). The 8-rank
      test's item 39 residual (the unrelated `SetNeighbors`/`nghbr` tie-break
      bug, `gid=14`'s corner slots 52-55 staying `UNREGISTERED` per the
      `FIELD-PROBE-FINAL` diagnostic) persisted unchanged across all 3
      passes, as expected -- this item is entirely orthogonal to item 39's
      open problem, exactly as item 40's PR #748 was for the 4 modules it
      touched.
    - **Not yet done**: no test was run with a genuine reflecting-*and*-
      diode corner where an AMR refinement boundary also meets the physical
      edge in a way that would visibly exercise the new coarse-BC pass's
      actual numerical content (the smoke tests above confirm no regression,
      not that the new coarse-BC step changes anything measurable in this
      particular topology) -- a dedicated test isolating that corner case,
      similar in spirit to item 40's own verification, would be the natural
      next check if this becomes load-bearing for future work.

42. **(2026-07-29 -- 2026-07-30) Item 38's 896/rank residual explained: it was
    never item 38's own bug -- `padm->u_adm` simply had no physical-BC pass
    at all before item 41, and item 41's `BCSADMTask` addition fixed that
    real, separate bug by coincidence. Item 38's own `SetNeighbors`
    registration bug remains open and unfixed.** A direct, empirical A/B
    test and cell-by-cell trace, prompted by re-running the CFC 8-rank
    TOV/AMR scenario (`cfc_item38_topology_8rank`) with the item 39
    diagnostic instrumentation (`CFC_DEBUG_NGHBR`) still in place and
    noticing its `NANS_IN_CONS` count had silently dropped to 0.

    - **A/B test**: built a clean git worktree at `baf91fcf` (item 40, one
      commit before item 41), ported the same debug instrumentation, and
      reran the identical scenario. Result: **896/rank `NANS_IN_CONS` on
      3-of-8 ranks reappeared**, byte-for-byte matching the originally
      documented signature (same regrid event, same `5.261418e-02`
      convergence diagnostic, so a true apples-to-apples comparison) --
      confirmed item 41 is what made the symptom disappear on current
      `HEAD` (`11fed7a8`). Worktree/build removed after the comparison;
      `git worktree` was used specifically so `HEAD`/the `cfc` branch were
      never touched.
    - **Cell-by-cell trace of *why***: the corrupted cell (`m=1` on rank 1,
      later identified as `gid=2`, location `k=6, j=12, i=0`, `nghost=4`,
      meshblock `8^3`) was first suspected to be explained by a masking
      mechanism in `src/bvals/buffs_cc.cpp` -- `InitRecvIndices`'s `icoar`/
      `iprol` index blocks widen the transverse (perpendicular-to-face)
      range by `ng`/`ng/2`, gated by the `f1`/`f2` sub-block selector, when
      a same-axis FACE neighbor is coarser (`isame`, same-level, has no
      such widening). This is real code and does mean a single coarser face
      neighbor can incidentally cover territory a missing diagonal neighbor
      would otherwise be needed for. **This hypothesis was directly tested
      and disproven for this specific cell**: a temporary print of the
      actual computed `icoar` bounds for the relevant coarser face-neighbor
      slot (slot 12, `gid=9`, one level coarser) gave `i=[4,11] j=[8,11]
      k=[4,11]` in coarse-index units -- nowhere near coarse `i=0-3`, which
      is where the corrupted fine-index `i=0` cell actually lives. So the
      widening mechanism is real but was not what fixed this cell.
    - **The actual mechanism, confirmed directly**: `gid=2`'s full 56-slot
      neighbor table shows **all four x1-inner face slots (0-3)
      `UNREGISTERED`** -- face registration in `SetNeighbors` is always
      unconditional (no octant-parity guard, unlike edges/corners), so
      finding nothing there means `gid=2` genuinely has no neighbor in the
      `-x1` direction. Its logical location (`lx=(0,1,0)`) confirms `lx1=0`,
      i.e. it sits at the mesh's actual `x1=0` edge, exactly where
      `ix1_bc=reflect` applies. Before item 41, `padm->u_adm` had *no*
      physical-BC call anywhere in its own ghost-exchange pipeline (item
      41's own writeup already noted this: "stale doc comment... previously
      claiming no physical-BC pass existed for u_adm" -- this was not
      stale, it was literally true). At a genuine physical boundary, no
      same-level/coarser/finer neighbor exchange and no prolongation touch
      that ghost region (all gated on an actual registered neighbor
      existing) -- so pre-item-41, this ghost slab was simply never written
      by *anything*, left at whatever zero/garbage state was allocated,
      producing `detg=0`. Item 41's new `BCSADMTask` (the first-ever call
      to `ADMBCs` on `u_adm`'s fine array) is what actually fixes it.
    - **This corrects item 38's own residual analysis** (see the forward-
      pointer added there): its "ruling out a BC-application gap" check only
      verified the y-coordinate against the domain extent -- it never
      checked x1, where this cell actually sits at a physical edge.
    - **Net implication**: item 41 fixed a real, previously-undocumented bug
      (`u_adm` had zero physical-BC treatment, at any physical boundary, in
      any CFC run with non-periodic BCs) -- independent of and unrelated to
      item 38/39's `SetNeighbors` diagonal-registration bug, which remains
      **confirmed present and unfixed** (`git diff` of `meshblock.cpp` is
      clean; no guard-removal fix has been reapplied). It happened to also
      make the CFC TOV scenario's visible symptom disappear, which is a
      coincidence of this particular test's topology (the residual's
      specific cell turned out to be a physical-boundary case, not a
      genuine interior AMR corner) -- not evidence that item 38/39's actual
      open problems (tie-break formula bug, extra-slot capacity, item 39c's
      MPI abort) are resolved. Item 39g's CFC-independent minimal
      reproducer (`inputs/tests/lwave_hydro_diag_collision.athinput`,
      periodic BCs, no physical boundary anywhere) is unaffected by any of
      this and remains the reliable way to verify future work on items
      38/39's actual registration bug.
    - **Diagnostic instrumentation**: all temporary `CFC_DEBUG_NGHBR`/
      `CFC_DEBUG_ICOAR`-gated instrumentation used for this investigation
      (`src/mesh/meshblock.cpp`, `src/bvals/bvals_cc.cpp`,
      `src/bvals/buffs_cc.cpp`) has been moved to a dedicated branch,
      `item39-debug-instrumentation`, rather than committed on `cfc` --
      `git diff` of all three files on `cfc` is clean (matches `HEAD`).
      Check out that branch to reuse the probes for future item 38/39 work
      instead of re-deriving them.

43. **(2026-07-30) Skip reconstructing X^i entirely -- compute Adual^ij
    directly from P_i/eta, eliminating X^i's own ghost-exchange pipeline
    (1 fewer MPI communication per stage).** User-proposed: `X^i` (the
    vector potential reconstructed from the packed `P_i`/`eta` solve,
    Shibata 1999 eq. 3.9) was used for exactly one thing --
    `ComputeADualFromX` (Gmunu eq. 76) differentiates it to get `Adual^ij`.
    Confirmed via grep before touching anything: `x_u`/`u_x` had zero other
    call sites anywhere in `cfc.cpp`/`cfc.hpp`. `X^i` needed its own full
    `RestX -> SendX -> RecvX -> ProlongX -> BCSX` ghost-exchange quintet
    (mirroring `P_i`/`eta`'s own) purely so that differentiation could read
    valid ghost cells -- unlike `beta^i` (reconstructed the same way from
    `u_p_beta`), which never gets its own ghost exchange because
    `AssembleADM` only reads it at interior points, never differentiates it.
    - **The math**: substituting `X^j`'s own definition into `Adual`'s
      formula and expanding gives `D_i X^j = 0.875*D_i P^j -
      0.125*[D_i D_j eta + sum_k (D_i D_j P^k) x^k + D_j P^i]` (worked out
      by hand from the code's own two formulas, not copied from the paper --
      Gmunu/Shibata don't give this combined form). `D_i D_j` is a genuine
      second derivative (`Dxx(i,...)` when `i==j`, `Dxy(i,j,...)` otherwise)
      of `P_i`/`eta` directly -- `X^i` is never materialized at all. Trace/
      symmetrization into `Adual^ab` is otherwise identical to the old
      `ComputeADualFromXImpl`.
    - **Confirmed feasible before implementing, not assumed**: checked
      `src/utils/finite_diff.hpp` directly -- `Dxx<NGHOST>`/`Dxy<NGHOST>`
      already exist as proper single-application second-derivative stencils
      with the *same* radius as `Dx<NGHOST>` (not "apply `Dx` twice"), so
      the direct computation needs no wider ghost region than `P_i`/`eta`'s
      existing exchange already provides.
    - **Mandatory empirical cross-check before removing anything** (given
      this is a from-scratch derivation, not trusted on algebra alone):
      added the new `cfc::ComputeADualFromPotentials` function
      (`cfc_reconstruct.hpp`/`.cpp`) alongside the old `X`-based path,
      temporarily computed *both* in `CFC::ComputeADual` gated behind
      `CFC_DEBUG_ADUAL_CHECK`, and diffed. Rebuilt, reran the 1-rank
      `cfc_amr_dynamic_check`-style sanity test: **`max|Adual_direct -
      Adual_via_X| = 0.000000e+00`** at every single call (multiple per
      stage, multiple stages/cycles) -- exact bit-for-bit agreement, not
      just close. This is the actual correctness gate; only removed the old
      path afterward.
    - **Removal**: deleted `X^i`'s entire ghost-exchange pipeline --
      `CFC_ReconstructX`/`CFC_RestX`/`CFC_SendX`/`CFC_RecvX`/
      `CFC_ProlongX`/`CFC_BCSX` (task methods, `QueueCFCTasks()`
      registrations, and `TaskName` enum entries in
      `numerical_relativity.hpp`), the `u_x`/`x_u`/`coarse_u_x`/`pbval_x`
      members, `ReconstructVectorPotential()`, the temporary cross-check
      code, and the now-dead `ComputeADualFromX`/`ComputeADualFromXImpl`
      (0 remaining call sites after the switch). `CFC_ComputeADual`'s
      task-graph dependency changed from `{CFC_BCSX}` to `{CFC_BCSPiEtaX}`
      directly. `RunXPsiSolvePass`'s manual sequence (used by
      `InitializeMetric`/`ReinitializeMetricForAMR`) updated the same way.
      `ReconstructVectorFromPotentials` itself (the shared low-level
      helper) is unchanged and still used for `beta^i`
      (`CFC::ReconstructShift`).
    - **Verified**: rebuilt `build_cfc` cleanly (no dangling references --
      confirmed via grep across `src/` for every removed symbol before and
      after). Reran both established sanity tests fresh (1-rank/8-rank
      `cfc_amr_dynamic_check`-style, `rst`/`perrank` cleared): no
      crash/NaN/fatal, and `max|delta psi - initial guess|` came back
      **byte-identical** to every prior run this session (`9.844341e-02`
      1-rank, `5.261418e-02` 8-rank) -- confirms zero behavior change, a
      pure optimization. (This was tested concurrently with an unrelated,
      already-running long physics test job on the same `build_cfc` binary
      -- confirmed rebuilding mid-run doesn't disturb an already-executing
      job, since the linker replaces the executable via a new inode/rename,
      not an in-place truncate.)
    - **Net effect**: 1 fewer full ghost-exchange round trip
      (`PackAndSendCC`/`RecvAndUnpackCC`, plus the restrict/prolongate/BC
      work around it) per stage, with the removed code's correctness
      confirmed empirically rather than assumed from the derivation alone.

44. **(2026-07-30) Switch CFC's ghost-exchanged fields (and `padm->u_adm`) to
    z4c's higher-order (Lagrange) restrict/prolong path -- closes out item
    39's own "future direction" note and item 39a's deliberate `is_z4c=false`
    choice.** User-requested: CFC's five ghost-exchanged fields (`u_p_x`,
    `delta_psi`, `delta_alpha_psi`, `u_p_beta`, `padm->u_adm` -- `u_x` no
    longer exists after item 43) all used the plain (non-z4c) path: simple
    cell-averaging restriction and a min-mod-limited piecewise-linear
    prolongation. `z4c::Z4c` has always had a second path for its own metric
    fields (`u0`, `u_weyl`): unlimited Lagrange-polynomial restrict/prolong
    (`RestrictInterpolation<NGHOST>`/`HighOrderProlongCC<NGHOST>`, 2nd/4th
    order keyed off `mesh/nghost`), selected by a `bool is_z4c` argument
    threaded through `RestrictCC`/`FillCoarseInBndryCC`/`ProlongateCC` and a
    matching `bool z4c` constructor argument on `MeshBoundaryValuesCC`. This
    is exactly the improvement item 39's own AMR analytic-residual
    investigation flagged but did not undertake (see the note ~line
    3382-3395: the coarse-fine boundary residual for CFC's elliptic solve is
    limited by prolongation order, and "a genuine improvement... would mean
    a higher-order/curvature-matching prolongation formula").
    - **Confirmed feasible with zero new plumbing**, by reading (not
      assuming) the actual generic implementations before touching
      anything: `PackAndSendCC`/`RecvAndUnpackCC` (`bvals_cc.cpp`) gate the
      extra same-level `isame_z4c` coarse-data payload purely on the
      `is_z4c_` member set at construction -- no per-module code needed
      beyond passing `true`. `InitSendIndices`/`InitRecvIndices`
      (`buffs_cc.cpp`) compute `isame_z4c` bounds unconditionally for every
      `MeshBoundaryValuesCC`, so no buffer-init changes were needed either.
      CFC already had the Z4c-mirroring physical-BC-before-prolongation
      pattern fully wired from item 41's BC-reorder work (`CFCBCsCoarse`/
      `ADMBCsCoarse` before `ProlongateCC`; `CFCBCs`/`ADMBCs` after) -- built
      for exactly this purpose, so no task-graph or BC-ordering changes were
      needed. All of CFC's own test inputs (`inputs/dyn_grmhd/cfc_tov*
      .athinput`) use `nghost = 4`, one of the only two cases (`ng==2`/`4`)
      the Lagrange-weight switch handles.
    - **The change itself**: in `CFC::CFC`, all five
      `new MeshBoundaryValuesCC(pmbp, pin, false)` -> `true`
      (`pbval_pietax`/`pbval_psi`/`pbval_alpha_psi`/`pbval_pietabeta`/
      `pbval_adm`). In each field's `Rest*Task`/`Prolong*Task`: the trailing
      `false` on `RestrictCC`/`ProlongateCC` -> `true`.
    - **A crash, not a redundancy -- `FillCoarseInBndryCC` had to be removed,
      not just flipped to `true`.** The original plan (mirroring the literal
      request) kept the existing `FillCoarseInBndryCC` call in each
      `Prolong*Task` and passed it `is_z4c=true` too. Rebuilding and rerunning
      the 1-rank/8-rank sanity tests immediately crashed with a Kokkos
      bounds-check failure: `out of bounds access label=("cfc_u_p_x") with
      indices [6,0,-1,4,4] but extents [128,4,16,16,16]` (and several more,
      including one overflowing the *high* side too). Traced to the actual
      cause rather than guessed at: `FillCoarseInBndryCC`'s `is_z4c` branch
      restricts *this block's own* just-received same-level ghost strip
      (`rbuf[n].isame[0]`, which for a `-x1` neighbor sits at fine indices
      `0..ng-1` -- already the very edge of the array) down into the coarse
      array. For CFC's TOV test (`MeshBlock size: 8 x 8 x 8`, `nghost=4` --
      an unusually high ghost-to-interior ratio), `RestrictInterpolation<4>`'s
      5-point-wide Lagrange stencil, applied to a strip that already sits at
      the fine array's outer edge, needs to read one cell *beyond* that edge
      (index `-1`) -- data that simply doesn't exist. This is exactly why
      z4c's own `Prolongate()` (`z4c_tasks.cpp:281-284`) skips calling
      `FillCoarseInBndryCC` when `is_z4c=true` -- not a perf-only redundancy
      as first assumed, but a genuine correctness/safety requirement: the
      same same-level coarse data is instead delivered via the `isame_z4c`
      payload in `Send`/`Recv`, computed by the *neighbor* from its own
      interior data (which has proper margin), sidestepping the problem
      entirely. **Fix**: removed all five `FillCoarseInBndryCC` calls from
      CFC's `Prolong*Task` functions (matching z4c's pattern exactly), each
      replaced with a comment explaining why.
    - **Verified**: rebuilt `build_cfc` cleanly. Reran the established
      1-rank/8-rank `cfc_amr_dynamic_check`-style sanity tests (fresh
      `rst`/`perrank`): no crash/NaN/`NANS_IN_CONS`, and
      `max|delta psi - initial guess|` came back **byte-identical** to the
      long-standing baseline (`9.844341e-02` 1-rank, `5.261418e-02` 8-rank)
      -- this particular diagnostic measures the regrid re-solve's Newton
      convergence against its own initial guess, which is dominated by the
      coarse-level guess quality rather than ghost-exchange interpolation
      order, so an unchanged value here is expected and not a sign the
      switch had no effect.
