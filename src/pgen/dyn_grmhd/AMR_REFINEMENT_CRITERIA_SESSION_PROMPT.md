Repo: ~/athenak_tde, branch proj/tde, currently at commit bdc42eb1 (pushed). This
branch just closed out a long investigation chain: the SetNeighbors/nghbr-symmetry
bug (SETNEIGHBORS_HANDOFF.md), the two-cause NANS_IN_CONS cascade
(src/cfc/NANCASCADE_HANDOFF.md), an unguarded-lapse-division bug
(src/dyn_grmhd/GRWAVESPEED_ALPHA_HANDOFF.md, commit 1dca0eff), and four follow-up
items from the cascade handoff (commit bdc42eb1) -- including a rewrite of the TDE
setup's AMR refinement criterion, `TDERefineTracker` in
src/pgen/dyn_grmhd/dyngr_tov.cpp (read its doc comment, ~line 691, before anything
else -- it has the full design rationale and known limitations already written
down). Two full production runs (`tde_prod_fixed_v3`, `tde_prod_fixed_v4`) have
completed cleanly to t=500 with 0 NaN, and the new refinement criterion was
validated at full 16-node production scale reaching t=145 (its own fixture's
documented full-disruption target) with 0 NaN.

None of that is your task and none of it should be re-litigated -- it's closed,
documented, and pushed. Your task is the NEXT open question: whether/how to improve
the adaptive refinement criterion further. This file is a prior session's
investigation and PROPOSED plan, not a validated conclusion. **Verify every claim
below yourself before acting on it** -- re-read the source, re-derive the numbers
where practical, and treat anything you can't independently confirm as unconfirmed,
the same discipline the linked handoff docs used throughout (several "obvious"
conclusions earlier in this same investigation turned out to be wrong on closer
inspection -- see SETNEIGHBORS_HANDOFF.md Sec 8 for the most dramatic example).

## Current design (verify by reading the code, not just this summary)

`TDERefineTracker`: a fixed puncture box UNIONed with a relative-density region
(`rho > amr_density_frac * rho_max`, default 1e-2), plus a guard (tracks
`last_good_rho_max`, disables the region half if `rho_max` drops >1e6x in one call)
against acting on a corrupted density field. Supersedes an earlier single
tracked-box-on-the-density-maximum design that thrashed once the star disrupted
(multiple clumps of comparable density, the "max" jumping between them).

## Four things this session's investigation flagged -- verify each independently

### 1. The 2D-vs-3D MeshBlock cost estimate gap

The design's own doc comment (dyngr_tov.cpp ~line 710) justifies
`amr_density_frac=1e-2` from a **2D, z=0-slice** projection: 566 MeshBlocks,
against "~1200-1275 total" for the old design -- with its own caveat, "multiply by
the debris' vertical extent in blocks for 3D," left unquantified.

**Claimed finding (VERIFY THIS FIRST, it's the most load-bearing number)**: the
actual full-3D validation run that reached t=145
(`/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/tde_tracker_nexteval_longrun_v1/`,
segments `output-0000` job 8857444 and `output-0001` job 8858653 -- check the tail
of `output-0001`'s `.out` for the final "Current number of MeshBlocks" line) used
**2672 total MeshBlocks**, ~4.7x the 2D estimate, and roughly 2x what the OLD
design typically used (`tde_prod_fixed_v3`/`v4`,
`/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/tde_prod_fixed_v{3,4}/`,
still running the OLD tracked-box design since both were built before/without
commit bdc42eb1 -- check their own `.out` files' MeshBlock-count lines at a
MATCHED PHYSICAL TIME, not matched cycle count, since dt differs; see
NANCASCADE_HANDOFF.md Sec 9.6 for why a common-time-window comparison matters and
how easy it is to get this wrong).

**If confirmed**: the cost argument that justified the current default threshold
undersells real 3D cost substantially. This doesn't mean the design is wrong --
the run was clean and evidently affordable -- but the specific number used to pick
`amr_density_frac=1e-2` needs redoing properly, apples-to-apples, at full 3D scale
and a common time, not projected from a 2D slice.

### 2. AthenaK has built-in gradient/curvature refinement criteria, unused here

`src/mesh/refinement_criteria.cpp` implements `slope` (normalized dq/q) and
`second_deriv` (normalized d2q/q) as built-in Löhner-style criteria, dispatched
from `src/mesh/mesh_refinement.cpp`'s `CheckForRefinement` (~line 230). **Verify
this yourself**: multiple `<amr_criterion>` blocks in one input file compose by
UNION -- `refine_flag` is zeroed once per check, then each criterion's Check*
function is called in turn without re-zeroing in between (read
`CheckForRefinement` and at least one of `CheckSlope`/`CheckSecondDeriv` in full to
confirm this, not just the dispatch switch). If true, a curvature-based criterion
on density (or velocity/Lorentz factor, for tidal-stream edges specifically) could
be added as an ADDITIONAL `<amr_criterion>` block alongside the existing
`user`-method tracker, with no changes to `TDERefineTracker` itself needed for a
first experiment.

**Untested hypothesis, not a finding**: such a criterion might target actual sharp
features (stream edges, shocks) rather than the whole bulk debris cloud above a
flat density cut, potentially addressing both the cost problem (1) and giving a
more truncation-error-motivated refinement signal than an arbitrary density
fraction. This needs an actual experiment, not just the architectural feasibility
check above.

### 3. The resolution-non-convergence finding predates the real bug fixes

NANCASCADE_HANDOFF.md Sec 9.6 measured the disrupted debris' energy growth rate
(`d(ln tot-E)/dt`) as NOT resolution-convergent (apparent order 0.022 to 0.241,
successive differences GROWING under refinement rather than shrinking) across a
4-level bracket. Read that section in full, including its own methodological
correction about needing a common TIME window rather than common cycle count.

**Open question, not yet re-checked**: that study ran during the buggy era, before
the alpha-guard fix (1dca0eff) and the two-cause cascade fix existed. It is not
established whether the non-convergence was a genuine physical/numerical property
of the debris, or was partly an artifact of the since-fixed pathologies (unexcised
puncture pileup, unguarded 1/alpha divisions -- both of which specifically corrupt
near-puncture/near-throat cells, which could plausibly leak into a global energy
growth-rate diagnostic even for a debris-focused resolution study). Worth a fresh,
cheap re-run of the same bracket (or a subset of it) with current code before
trusting the old conclusion for any new design decision.

### 4. No level cap for a more aggressive density fraction

The doc comment (dyngr_tov.cpp ~line 712) notes `amr_density_frac=1e-4` would need
"a level cap, not implemented here." Verify whether `<mesh_refinement>` or the mesh
machinery already has a generic per-region/per-criterion level-cap mechanism you
could reuse, or whether one needs to be added to `TDERefineTracker` specifically.

## Proposed plan (this session's proposal -- re-prioritize freely once you've
verified the above; nothing here is a commitment)

1. Redo the MeshBlock-cost comparison properly: matched physical time, full 3D,
   old design vs. new, using data already on disk (see file paths in item 1) before
   generating any new runs.
2. Prototype a `second_deriv`/`slope` criterion alongside the existing tracker
   (cheap: one new `<amr_criterion>` block, no code change) and compare cost/
   resolution against the pure region-criterion baseline.
3. Re-run a resolution-convergence check on debris energetics with current code,
   matching Sec 9.6's common-time-window methodology, to see if the non-convergence
   finding still holds post-fix.
4. If 1-3 suggest it's warranted: add a level cap and revisit lowering
   `amr_density_frac`.
5. Lower priority: stress-test the corrupted-`rho_max` guard's `1e-6` threshold
   (dyngr_tov.cpp ~line 802) against a legitimate sharp physical feature
   (shock/detonation-like discontinuity) to check it can't false-positive on real
   physics, only on genuine corruption.

## Constraints and traps (carried over from this whole investigation)

- Never modify `tde_elliptic_tracker_nexteval_v2/` or its checkpoints --
  `tde_prod_fixed_v3`/`v4` and most of this project's diagnostic runs restart from
  its `.rst` files via symlink (`output-0000/parent/rst/... -> ...`); deleting or
  altering it breaks their provenance.
- Build hosts: `aurora-uan-0007`/`-0008` carry the newer 2026-09 next-eval preview
  OS image and are required ONLY for next-eval-targeted builds (PBS there exports
  the right MODULEPATH for the pe2626 route). Building an ordinary debug/
  capacity-queue CPU binary there produces a GLIBC ABI mismatch against normal
  compute nodes -- use any standard login node for non-next-eval work instead (see
  memory note `project-aurora-nexteval-kokkos-blocker` if available to you, or
  re-derive this from scratch: it cost real debugging time to find once already).
- `MPIR_CVAR_CH4_IPC_GPU_MAX_CACHE_ENTRIES=1024` for GPU jobs. `nlim` is an
  absolute cycle target, not "N more from restart."
- `<cfc> src_nan_check=1` is cheap and directly useful; `<cfc> solve_interval`
  defaults to 0 (= checked every cycle), no need to touch it for a short run.
- Auto-chained debug-queue jobs can hit a benign post-completion teardown crash
  (signal 11 in MPI/Kokkos finalize, AFTER the physics and checkpoint are already
  written cleanly -- confirmed via `Driver::Finalize`, src/driver/driver.cpp
  ~line 532-597). The batch-chaining scripts already in use on this project's runs
  (see `tde_tracker_cpu_longrun_v1`/`tde_tracker_nexteval_longrun_v1`'s
  `BATCH/batch.t` for a working example) have a `completed_ok` check for this --
  reuse that pattern rather than re-discovering it.
- Do not re-litigate the SetNeighbors, NaN-cascade, or alpha-guard diagnoses --
  all three are closed. This work is additional to, not a reopening of, any of
  them.
