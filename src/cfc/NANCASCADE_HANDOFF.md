# TDE run: the energy runaway that ends in `NANS_IN_CONS` — handoff note

**Status**: undiagnosed, and it is the only thing now blocking the
`tde_elliptic_tracker_nexteval_v2` production run. Written 2026-09-16 for a
fresh session.

**How to read this file.** Everything in §1–§3 is measured, with the command or
log it came from. §4 is leads — explicitly unproven, and none of it should be
treated as a cause until the symptom moves when you change it. That distinction
is not pedantry here: this run's previous investigation spent weeks on a
confidently-asserted cause that turned out to be wrong (see §7), and the single
thing that would have prevented it was refusing to promote a lead to a fact.

---

## 1. The symptom, precisely

The run dies in a `NANS_IN_CONS` cascade with a runaway timestep. Most recently
observed in job 8832259 (8 nodes / 96 ranks, next-eval), restarting from cycle
10000:

```
cycle=10597 time=4.441874e+02 dt=3.078349e-02     <- clean
cycle=10598 time=4.442182e+02 dt=3.078349e-02     <- clean
cycle=10599 time=4.442490e+02 dt=3.078349e-02     <- clean, last clean cycle
cycle=10600 time=4.442798e+02 dt=6.156698e-02     <- 2x
cycle=10601 time=4.443413e+02 dt=1.231340e-01     <- 2x
...                                                   exactly 2x every cycle
cycle=10610 time=5.000000e+02 dt=4.851898e+01
974640 NANS_IN_CONS ; "Terminating on time limit"
```

Facts about the terminal phase, all from that run's `.out`:

- **`dt` doubles exactly 2x per cycle** once it starts. Exactly 2x, not
  approximately — which is what a CFL computed from an already-degenerate state
  looks like, not physical steepening. Worth taking seriously as a hint that the
  corruption precedes the first `NANS_IN_CONS` by at least one step.
- The "terminating on time limit" is **spurious**: `tlim=500` is reached by the
  runaway `dt` overshooting, not by the simulation integrating there.
- **The elliptic solver converged throughout** — `grep -c "Failed to converge"`
  is **0** across all 600 cycles. Whatever this is, the multigrid solve was not
  failing to converge.
- **NaN sites are not localized.** Of the first 3000: 285 at r<5, 730 at
  5<=r<15, 948 at 15<=r<40, 1037 at r>=40. This is a global blowup, not a
  puncture or excision artifact (`excise = false` in this run;
  `puncture_mass = 1.0`, constant across every checkpoint — verified, it is not
  growing).
- It is **reproducible and not a restart artifact**. The original v2 run hit the
  same cascade at cycle 10660 with **974,603** occurrences; the restart hit it at
  10600 with **974,640**. Same cascade, same magnitude, ~60 cycles apart (a
  checkpoint restart is not bitwise identical to continuous running).

## 2. What is ruled out

**`MeshBlock::SetNeighbors` / the neighbour table.** This was the previous
investigation's entire focus and it is not the cause. The cascade reproduces at
the same cycle, to the same magnitude, against a neighbour table proven
symmetric and ghost-complete, re-verified on all 51 AMR remeshes during the run.
See `SETNEIGHBORS_HANDOFF.md` §8. Do not re-open this; if you want to confirm it
cheaply, `ATHENAK_CHECK_NGHBR_SYMMETRY=1` audits the table at every mesh build.

**Elliptic-solver convergence failure.** 0 occurrences (§1). `SETNEIGHBORS_HANDOFF.md`
§5 discusses a convergence failure, but that was a *different input* (the
wide-domain smoke test), not this run.

**Matter being created.** Mass is conserved to 11 significant digits through the
entire clean phase (§3 table). Whatever grows, it is not rest mass.

## 3. The actual shape of the problem: an exponential energy runaway from periapsis

**This is the most important thing in this file, and it reframes the question.**
The cascade at cycle 10600 is not the beginning of anything — it is the terminal
phase of a runaway that has been building for hundreds of M.

From the original run's own history file
(`tde_elliptic_tracker_nexteval_v2/output-0000/cfc_tde_wd_imbh_elliptic_tracker.mhd.hst`,
column 7 = `tot-E`, column 3 = `mass`, output cadence `dt=0.1`):

| t | tot-E | mass | note |
|---|---|---|---|
| 0 | 8.345e-07 | 8.331e-05 | |
| 108 | 1.069e-06 | 8.331e-05 | slow secular growth, ~2x per 180 M |
| 252 | 3.272e-06 | 8.326e-05 | |
| 324 | 8.611e-06 | 8.160e-05 | mass dropping — star disrupting, physical |
| 342 | 1.083e-05 | 8.093e-05 | |
| **360** | **2.187e-05** | 8.051e-05 | **inflection** |
| **378** | **3.073e-04** | 8.049e-05 | **14x in 18 M** |
| 396 | 2.413e-03 | 8.049e-05 | 8x |
| 414 | 5.611e-03 | 8.047e-05 | |
| 432 | 1.396e-02 | 8.048e-05 | |
| 446 | 2.086e-01 | — | last clean row, then NaN |

Read that as: a slow secular energy rise from t=0 that **sharply accelerates
around t≈360–378** into a near-exponential runaway (e-folding ~2 M by the end),
terminating in the cascade ~100 M later. Total growth over the run: **~2.5e5x**.

Two facts that pin the inflection:

- **Periapsis is at t≈349** (the parfile's own comment). The acceleration begins
  immediately after it. The parfile also notes the fixture "explicitly left the
  post-periapsis debris-spreading phase untested", and that this run's "first
  production attempt DID hit it, past periapsis."
- **Mass is flat across the acceleration** (8.051e-05 at t=360 to 8.048e-05 at
  t=432, a 0.04% drift). The earlier mass drop (t≈250–360, -3.4%) is the star
  being disrupted and is physical. So the energy growth is not tracking mass
  loss.

During the 600 "clean" cycles of the validation run, `tot-E` grows **16.4x**
(9.163e-03 -> 1.505e-01) while `dt` *decreases* smoothly (3.51e-02 -> 3.08e-02)
right up to the jump. Those cycles are clean only in the sense of producing no
NaNs; the run is already deep in the runaway throughout.

**Consequence for how you work on this**: restarting from cycle 10000 (t=422) is
starting ~60 M into the blowup and is nearly useless for finding a cause. Start
from **checkpoint 00008, cycle 8000, t=347.3** — at periapsis, before the
inflection. See §5.

**A caveat to hold honestly**: it has NOT been established that any particular
part of this growth is unphysical. A tidally disrupted star being compressed and
shocked at periapsis is a violent event and some energy growth is expected. What
is suspicious is the near-exponential character and the termination in NaN, not
the existence of growth. Establishing which part is numerical is itself a task —
a resolution or configuration study is the honest way to do it, not assertion.

## 4. Leads — unproven, in rough order of how testable they are

**4a. `solve_interval=10` is used far outside the regime it was validated in.**
The run sets `<cfc> solve_interval = 10` (verified in every checkpoint's embedded
parfile), so the CFC elliptic metric is re-solved only every 10th cycle and held
fixed in between. DEVELOPMENT.md item 61's addendum validated `N=10` on this
fixture and found "no detectable accuracy cost" — but that measurement was
**`nlim=40`, at t=1.7689**, i.e. 40 cycles from t=0 with the star far from
periapsis and the metric evolving slowly. Post-periapsis the matter distribution
changes fast, and a metric held fixed for 10 cycles is a much stronger
approximation there. The onset of the energy acceleration coincides with exactly
that regime change.

This is the most directly testable lead in this file, and it has a clean
quantitative criterion. **Test**: restart from checkpoint 00008 (t=347.3) twice,
identically except `solve_interval = 1` vs `10`, run a few hundred cycles, and
compare the `tot-E` growth rate in the `.hst`. If the growth rate is materially
lower at `N=1`, that is the symptom moving when you change the thing. If it is
unchanged, `solve_interval` is exonerated and you have spent one cheap job.

Note this cuts against a performance decision: `N=10` buys 3.65x on this fixture
(item 61 addendum). If it turns out to be the cause, that tradeoff needs
revisiting with the user, not silently reverting.

**4b. The tracked/moving AMR refinement region.** The run uses star-tracking AMR
(`RefineTOVTracker`, DEVELOPMENT.md `## 63`) following a disrupting star. Post-
periapsis the debris spreads and the tracked region moves and remeshes fast — 51
remeshes in 600 cycles in the validation run. Repeated prolongation/restriction
cycling is a plausible energy-injection mechanism. Untested. A cheap probe:
correlate remesh events in the `.out` against the `tot-E` series; if the growth
is stepwise at remeshes rather than smooth, that is a strong signal.

**4c. `alpha_floor`.** Item 61's measurements repeatedly note `alpha-min` sitting
*exactly* at the floor (`1.0e-6`). A lapse pinned at a floor is a modified
equation; whether that matters here is unexamined.

**4d. The exact-2x `dt` doubling.** Worth understanding mechanically even though
it is terminal-phase: find what in `NewTimeStep` produces exactly 2x, since that
identifies which quantity has gone degenerate (and probably one cycle before the
first NaN).

## 5. Assets — what exists, so you do not regenerate it

Base: `/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/`

**Original production run** (`tde_elliptic_tracker_nexteval_v2/output-0000/`) —
**read-only, do not modify or delete; it is the only record of the original
failure**:
- `rst/*.rst`, cycle 0 to 10670. Mapped:

  | file | cycle | t | nmb |
  |---|---|---|---|
  | 00008 | 8000 | **347.309** | 1212 | <- **start here: periapsis, pre-inflection**
  | 00009 | 9000 | 386.440 | 1240 | already accelerating |
  | 00010 | 10000 | 422.239 | 1268 | deep in the runaway |
  | 00011 | 10670 | 500.000 | 848 | post-cascade, garbage |

  (00000–00007 are cycle 0–7000, t=0–308.5, in 1000-cycle steps.)
- `cfc_tde_wd_imbh_elliptic_tracker.mhd.hst` — 4471 rows at `dt=0.1` from t=0 to
  the cascade. This is the single most useful file here; §3's table is from it.
- `.user.hst`, and `bin/` dumps (`adm` and `mhd_w_bcc`, `dt=1.0`).

**Clean 600-cycle validation run** (`tde_elliptic_tracker_nexteval_v2_parityfix/output-0000/`,
job 8832259) — the current-code baseline, with the neighbour-table fix and the
audit enabled:
- `.out` with per-cycle lines, 51 audit lines (all 0 asymmetric), and the cascade.
- `.mhd.hst` (223 rows), 58 `bin/` dumps, one `rst/` at cycle 10610 (post-cascade).

**Build**: `~/athenak_tde/build_gpu_pe2626/src/athena` — PE 26.26.0 SYCL/PVC,
`PROBLEM=dyn_grmhd/dyngr_tov`, branch `proj/tde`. Rebuild with
`bash ~/athenak_tde/build_aurora_gpu_nexteval_pe2626.sh`, **which must be run on
aurora-uan-0007 or -0008** (only UANs with the next-eval image).

## 6. Tooling that already exists

- `ATHENAK_CHECK_NGHBR_SYMMETRY=1` — audits the neighbour table at every mesh
  build (`=2` aborts). Cheap; leave it on, it rules out a whole class of
  suspicion for free.
- `scripts/nghbr_symmetry_offline.py <file.rst>` — the same audit straight from a
  restart file, no build, no MPI, no allocation.
- Restart files carry the full tree and the complete parameter deck. Parsing one
  is ~20 lines (see that script) and answers most "what was the configuration /
  topology at time X" questions in seconds without a job.
- `<cfc> mg_verbose=1` for multigrid diagnostics (DEVELOPMENT.md item 62).
- The `athenak-plot` agent for the `.hst`/`bin` time series.
- **`mpiexec` does not work on Aurora UANs.** For anything runnable serially,
  build with `-DAthena_ENABLE_MPI=OFF` (as `~/athenak_tde/build_serial_audit`
  was) and run directly on the login node.

## 7. How this investigation has gone wrong before — please read

The previous investigation into this run's failure spent weeks and ~10 commits
on `MeshBlock::SetNeighbors`, and the diagnosis was wrong from the start. The
chain is documented in `SETNEIGHBORS_HANDOFF.md` §8 and DEVELOPMENT.md item 39b.
Three specific failure modes, all worth actively guarding against here:

1. **A theory outlived its evidence.** Item 42 found that the symptom motivating
   the whole `SetNeighbors` theory belonged to a different bug entirely ("it was
   never item 38's own bug") — and in the same breath kept the theory alive on
   the strength of a static code reading. Everything downstream was built on
   that. If the observation that motivated your hypothesis gets explained by
   something else, the hypothesis does not survive on elegance.
2. **A traced observation is not a traced conclusion.** Item 39b did real runtime
   instrumentation and got a true result. The inference it drew from that result
   was never instrumented, and was false. Check the step you did not measure.
3. **Reproducers that cannot fail.** Two tests were cited for years as
   reproducing the bug; one ran a topology that behaves identically under every
   rule, the other asserted nothing and skipped by default. Before trusting a
   test here, confirm it fails when the thing it tests is broken.

The corresponding discipline for this problem: **do not accept a cause until the
symptom moves when you change it.** §3 gives you a continuous, quantitative
symptom (the `tot-E` growth rate) rather than a binary crash, which makes that
much easier than it was for the previous investigation — use it.

## 8. Constraints

- **Never modify or delete `tde_elliptic_tracker_nexteval_v2/`** or its
  checkpoints. Restart via a fresh case directory with the parent `rst` as a
  **symlink** — see `tde_elliptic_tracker_nexteval_v2_parityfix/` for the exact
  pattern (`batchtools init` / `makesegment --no-parent`, then create
  `output-0000/parent/rst/<name>.rst` as a symlink).
- **Do not restart production** until there is validated evidence the cause is
  addressed. Restarting from cycle 10000 today buys ~600 cycles and hits the same
  wall.
- next-eval: walltime <= 06:00:00, 20 queued/user, FIFO. Submit from
  aurora-uan-0007/0008 (PBS exports the submitting shell's MODULEPATH).
- Set `MPIR_CVAR_CH4_IPC_GPU_MAX_CACHE_ENTRIES=1024`, **not 128**. 128 leaves
  CFC-heavy 8-node runs 3.21x slower than capacity. The two jobs above ran at
  128, so their timings are not a valid performance baseline (their correctness
  results are unaffected — same value both sides). DEVELOPMENT.md item 62 in
  `~/athenak_cfc`.
- Node count must match the checkpoint's original decomposition (8 nodes / 96
  ranks) unless you intend to study that: restarting at a different rank count
  runs ~60% slower per cycle.
- `nlim` is an **absolute** cycle target, not "N more from the restart".
