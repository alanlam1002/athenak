# TDE run: the energy runaway that ends in `NANS_IN_CONS` — handoff note

**PRODUCTION COMPLETE (2026-09-19): `tde_prod_fixed_v3` reached `tlim=500`.**
Cycle 12109, `dt`=3.20e-02, **0 `NANS_IN_CONS`, 0 `CFC NONFINITE`, 0 convergence
failures**, and the physics is correct: `tau/M` peaks at **0.1390 at t=358.7**
(periapsis shock heating) then declines monotonically to 0.0333, and `|S|/D`
peaks at 0.615 at periapsis then falls to 0.056 -- the debris turnover. The
original run read `tau/M` = **2592** at t=446 before its cascade. **This problem
is closed.** See §21.

**RESOLVED (2026-09-18) -- READ §19 FIRST. The run completes.** With **excision ON**
plus the **§15 `grad_ap6_0` fix**, the case ran to **`tlim=500`** with 0
`NANS_IN_CONS`, 0 `CFC NONFINITE`, healthy `dt`, and `tau/M` **declining**
monotonically to 0.0256 -- against 2592 in the original run. The §9.7 production
gate is met. §19 has the recommendation and what remains open.

**TWO CAUSES, BOTH IDENTIFIED, BOTH ADDRESSED -- §17.**
The production failure and the excision-path failure are *different bugs with
different fingerprints*, which is why every single-cause theory in §§1-15 came
back partly right and partly wrong. Production needs **excision ON** *and* the
**§15 `grad_ap6_0` fix**; neither alone suffices. §17 has the evidence, §16 the
working, §15 the code fix.

**(superseded header, kept for the reasoning trail) ONE CAUSE FOUND AND FIXED, ONE STILL OPEN -- §15 AND §16.**
`cfc_puncture.cpp`'s `grad_ap6_0` formed `dalpha0/alpha0`, which is 0/0 at the
maximal-slicing trumpet throat. Fixed (§15), and **validated on the excision
path**: `tde_fixval_exc` reached cycle 10900 clean. But the **production
(no-excision) configuration still fails at ~10604**, essentially where it always
did -- so §15 explains the *excision* runs' early failures, NOT the original one.
A second, independent cause remains; §16 has what is known and what is running.
§§1-14 are the investigation, left intact including the wrong turns.

**READ §11 NEXT (2026-09-17).** §3's central claim -- that the energy runaway is
a global numerical blowup -- is **WRONG**, and §11 corrects it with measurements.
The debris is healthy. The runaway is entirely inside r<2, at the **unexcised
puncture**, where the lapse sits pinned at `alpha_floor`. Most of §4 and §9's
negative results are explained by their all having tested the *debris*, which was
never the problem. Sections 1-10 are left as written, with pointers added, so the
reasoning chain stays reconstructible.

**Status**: undiagnosed, and it is the only thing now blocking the
`tde_elliptic_tracker_nexteval_v2` production run. Written 2026-09-16 for a
fresh session; **updated 2026-09-16 (session 2)** -- see §9 for what that
session added. In short: **§4a, §4b and §4d are now all closed**, §4c is argued
down, and the numerical onset is pinned to periapsis (t~350) rather than §3's
t~360-378. **All four of the original note's leads are now spent, and no cause
is established.** §9.5 says where to go next.

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

**The `.hst` `1-KE`/`2-KE`/`3-KE` columns as an energy diagnostic.** Not a
ruled-out *cause* -- a ruled-out *instrument*. `history.cpp:345` computes them as
`vol*0.5*SQR(u0(IM_i))/u0(IDN)`, i.e. the Newtonian `S_i^2/(2 D)` applied to the
GR **conserved** variables. In the low-density atmosphere `D` sits at its floor
while `S_i` does not, so the column diverges by construction. It reaches 7.9e+06
against a rest mass of 8.0e-05 -- eleven orders of magnitude above the total
energy, which is impossible for a kinetic energy. **Do not read these columns as
an energy, and do not treat their blowup as a finding.** Use `tot-E` (column 7),
which is a legitimate `vol*u0(IEN)` integral of the conserved `tau`.

**AMR remesh events as the energy-injection site (the §4b mechanism).** Ruled out
as the *injection event*; see §9.2 for the measurement. The growth is smooth and
volumetric, not stepwise at regrids. This does **not** rule out the refinement
*configuration* (i.e. insufficient resolution on the spreading debris), which is
a different claim and remains open.

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

**Which part of the growth is numerical -- ANSWERED (2026-09-16, session 2).**
The original note left this open and explicitly warned against assuming it. It is
now settled for the post-periapsis part, from the `.hst` alone, and without
relying on the broken `KE` columns (§2):

- **`tau/M` (= `tot-E`/`mass`), the specific energy in `c=1` units**: 0.0100 at
  t=0, 0.0384 at t=250, 0.153 at periapsis (t=349), then **crosses 1.0 at
  t=370.5** and reaches **2592** by t=446. A specific energy of 2592 c^2 for
  tidal debris is impossible -- 2592x more energy than there is rest mass to
  carry it.
- **`|S|/D` (= `sqrt(1-mom^2 + 2-mom^2 + 3-mom^2)`/`mass`), the mass-weighted
  bulk `h*W*v`**: 0.146 at t=0, rising smoothly and monotonically to 0.712 at
  periapsis -- exactly the profile of free-fall onto a relativistic periapsis --
  then **1.9e+04** by t=446, a bulk Lorentz factor of order 1e4.

The discriminator is **the turnover that never happens**. As the debris recedes
from periapsis it must decelerate: both `tau` and `|S|` have to turn over and
fall. Neither does; both keep growing near-exponentially. Therefore:

- **Pre-periapsis (t=0 to ~349) growth is consistent with physical infall** and
  is not anomalous in character. Not *proven* physical, but nothing about it
  demands explanation.
- **Post-periapsis growth is numerical.** Not "suspicious" -- numerical, on the
  energy-exceeds-rest-mass bound above.

**This moves the onset earlier than this section originally claimed.** t~360-378
is where the runaway became *visible* in `tot-E`; the physics above says it
begins at or immediately after **periapsis, t~350**. Checkpoint 00008 (t=347.3)
therefore straddles the onset almost exactly, which is what makes it the right
restart point -- better than the original note knew.

**Still open**: whether the slow pre-periapsis rise (e-folding ~130 M) is itself
a small numerical source that periapsis merely amplifies, or is entirely orbital.
A resolution study is still the honest way to settle that; it has not been done.

**e-folding time of `tot-E`, by window.** The growth is exponential *throughout*
with a shortening timescale -- it is not "flat, then a knee":

| window (M) | e-folding time |
|---|---|
| 100-250 | 133 M |
| 250-320 | 75 M |
| 320-350 | 69 M |
| **350-370** | **11 M** |
| **370-400** | **8 M** |
| 400-440 | 17 M |

## 4. Leads — unproven, in rough order of how testable they are

**4a. `solve_interval=10` is used far outside the regime it was validated in.**
**[CLOSED 2026-09-16 -- EXONERATED. The A/B was run (jobs 8832465 / 8832467) and
the symptom did not move: `d(ln tot-E)/dt` = 0.0893 /M at `solve_interval=10` vs
0.0899 /M at `solve_interval=1`, a 0.65% difference, with the si=1 arm very
slightly *higher*. Full result and its validity argument in §9.4. The lead below
is left intact as written because the reasoning was sound and only the answer
was no.]**
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

**A mechanism for 4a, found in session 2 -- still a lead, not a cause.** The CFC
task order (DEVELOPMENT.md lines 461-484) is: step 3 solve `psi` ->
`AssembleConformalMetric` writes `psi4`/`g_dd` -> **step 4 con2prim** -> steps
5-6 solve lapse/shift -> `AssembleLapseShiftK` writes `alpha`/`beta`/`vK_dd`. So
the con2prim at step 4 always runs against a **freshly solved `psi^4` paired with
an `alpha`/`vK_dd` left over from the previous solve**. That inconsistency is one
cycle old at `solve_interval=1` and **ten cycles old at `solve_interval=10`** --
it scales with exactly the parameter under test. It is also the right *character*
to explain the symptom: a smooth, per-cycle, volumetric source, which is what
§9.2 measured the growth to be. This mechanism was confirmed to exist by reading
the task order; it has **not** been shown to inject energy, let alone enough.
The A/B is what decides that.

**4b. The tracked/moving AMR refinement region.** **[CLOSED as the injection
event -- see §9.2. The proposed probe was run and came back negative. The
refinement *configuration*/resolution remains open; that is a different claim.]** The run uses star-tracking AMR
(`RefineTOVTracker`, DEVELOPMENT.md `## 63`) following a disrupting star. Post-
periapsis the debris spreads and the tracked region moves and remeshes fast — 51
remeshes in 600 cycles in the validation run. Repeated prolongation/restriction
cycling is a plausible energy-injection mechanism. Untested. A cheap probe:
correlate remesh events in the `.out` against the `tot-E` series; if the growth
is stepwise at remeshes rather than smooth, that is a strong signal.

**4c. `alpha_floor`.** Item 61's measurements repeatedly note `alpha-min` sitting
*exactly* at the floor (`1.0e-6`). A lapse pinned at a floor is a modified
equation; whether that matters here is unexamined.

**Evidence against it at the failure site (session 2, §9.3):** at the *first*
`NANS_IN_CONS` cells, `alp = 0.8657` -- five orders of magnitude off the
`1.0e-6` floor, and healthy. Whatever is pinned at the floor elsewhere in the
domain, the lapse is not degenerate where the NaN first appears. This lowers 4c
considerably but does not close it: a floor-pinned lapse somewhere else could
still be a slow energy source feeding the global runaway.

**4d. The exact-2x `dt` doubling. -- RESOLVED (session 2), and it is a pure
downstream symptom.** `Mesh::NewTimeStep` (`src/mesh/mesh.cpp:583`) opens with
`dt = 2.0*dt` as a **growth limiter**, then takes `min` with `cfl*dtnew`. Exact
2x per cycle therefore means only that the limiter is binding, i.e. `dtnew` has
become enormous and stays so.

Why `dtnew` becomes enormous: it is a `Kokkos::Min` reduction of `dx/max_dv`
(`dyn_grmhd_newdt.cpp:195-197`), and **`fmin(NaN, x)` returns `x`** per IEEE 754.
NaN cells are therefore *silently dropped out of* the reduction rather than
poisoning it. As NaN spreads, the cells that were constraining `dt` stop being
counted and `dtnew` shoots up.

Checked against the log rather than asserted: in the parityfix `.out` the first
`NANS_IN_CONS` prints **between** the `cycle=10599` line and the `cycle=10600`
line -- i.e. NaN appears in the same step that produces the first doubled `dt`,
which is what the dropout mechanism predicts.

**Consequence: this closes 4d and it identifies nothing about the cause.** It
also retires §1's inference that the doubling implies corruption "at least one
step" before the first `NANS_IN_CONS` -- the doubling is fully explained by NaN
arriving in that same step. (The *real* corruption does precede it, by ~100 M,
but that is §3's argument, not this one.)

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

---

## 9. Session 2 (2026-09-16): what was measured, and what is running

Everything in 9.1-9.3 is measured or read straight from the source, with the
command or file. 9.4 is an experiment in flight with no result yet. Nothing here
is a cause.

### 9.1 The `KE` columns are a broken instrument (and cost the first hour)

Reading the `.hst` past column 7 shows `1-KE+2-KE+3-KE` reaching 7.9e+06 against
a rest mass of 8.0e-05, and departing from sanity around t~300 -- ~60 M *before*
§3's inflection. That looks like a major finding and it is not one.
`src/outputs/history.cpp:345` computes the column as

```
vol*0.5*SQR(u0(m,IM1,k,j,i))/u0(m,IDN,k,j,i)
```

-- Newtonian `S_i^2/(2D)` on the GR **conserved** variables. In the atmosphere
`D` is at its floor and `S_i` is not, so it diverges by construction and carries
no information about the energy. Recorded here because it is exactly §7 trap 2
(an un-instrumented inference from a real observation), and the next reader will
open the same file and see the same tempting numbers. Use `tot-E` only.

### 9.2 AMR remeshes are not the injection event (§4b's own proposed probe, run)

§4b proposed: "correlate remesh events in the `.out` against the `tot-E` series;
if the growth is stepwise at remeshes rather than smooth, that is a strong
signal." Done, on the original v2 run.

Method: 502 remesh events extracted from the 624 MB `.out` by the
`CFC::ReinitializeMetricForAMR` line, timestamped from the preceding `cycle=`
line; `tot-E` from the `.hst` at its native `dt=0.1`, which is finer than the
~0.6 M mean remesh interval, so the two are resolvable against each other.

**Result 1 -- remesh rate does not track growth rate:**

| window | remeshes/M | d(ln tot-E)/dt |
|---|---|---|
| 325-350 | 1.56 | 0.0145 |
| 350-375 | 1.64 | 0.1074 |
| 400-425 | **2.64** | **0.0455** |

The growth rate jumps 7.4x across periapsis with the remesh rate essentially
unchanged (1.56 -> 1.64), and the *highest* remesh rate in the run coincides with
a *lower* growth rate.

**Result 2 -- growth is not stepwise at remeshes.** Comparing `.hst` intervals
that contain a remesh against those that do not:

| window | with remesh | without remesh | share of growth in remesh intervals |
|---|---|---|---|
| t=300-350 | +0.00142 | +0.00143 | 14.5% (of 14.6% of intervals) |
| t=350-400 | +0.00778 | +0.01161 | 10.0% (of 14.2% of intervals) |

Statistically identical before periapsis; *slower* in remesh intervals after it.

**Conclusion**: the energy growth is smooth and volumetric -- a per-cycle,
per-cell source -- not an injection at regrid events. Prolongation/restriction
cycling is closed as the mechanism. Note carefully what this does **not** close:
whether the refinement *configuration* leaves the spreading debris underresolved
is a separate question and is untouched by this.

### 9.3 Forensics on the first NaN cells

From the parityfix `.out`, the first `NANS_IN_CONS` blocks (the printer is
`src/eos/primitive_solver_hyd.hpp:509`; index order is `(m,k,j,i)`, confirmed
against the arg list at line 529):

```
  Location: (8, 9, 4, 4)   (-11.96875, 7.03125, 3.34375)     <- r ~ 14.3
    D, Sx, Sy, Sz, tau, Bx, By, Bz  = nan
    detg = nan
    g_dd = {nan, 0, 0, nan, 0, nan}          <- psi^4 * delta_ij, CFC form
    alp  = 0.86567484364035896               <- FINITE, healthy
    beta = {0.069, -0.095, -0.025}           <- FINITE
    psi4 = nan
    K_dd = {0.0033, -0.0069, ...}            <- FINITE
```

**Fact**: the NaN enters the metric *entirely through `psi`*. `g_dd` is NaN only
on its diagonal (the off-diagonals are exactly 0 -- the conformally-flat form),
`detg` and `psi4` are NaN, and `alpha`, `beta`, `vK_dd` are all finite and
healthy. So **the conformal factor came back NaN from the multigrid solve in that
cycle**, and con2prim reported it.

**A trap that was walked into and backed out of, recorded so the next reader does
not repeat it.** `psi4`, `alpha` and `vK_dd` are all derived from the same
`psi_val` in `cfc_reconstruct.cpp` (lines 188, 216, 225). If `psi_val` were NaN,
`alpha = (...)/psi_val` and `vK_dd = a_dd/psi_val^2` would be NaN too. They are
not. That looks like a genuine inconsistency -- two views of the same `psi`
disagreeing -- and it is not: it is ordinary task ordering. `psi4`/`g_dd` are
written at step 3 and `alpha`/`vK_dd` at the final step (DEVELOPMENT.md 461-484),
with con2prim in between at step 4, so at the moment of the error `alpha` is
simply the **previous** solve's value. No bug. (The *scaling* of that staleness
with `solve_interval` is a separate and live lead -- see §4a.)

**Also**: `alp = 0.8657` at the first NaN site is nowhere near `alpha_floor =
1.0e-6`. See §4c.

### 9.4 The §4a A/B -- RUN, and `solve_interval` is exonerated

Two restarts from checkpoint **00008** (cycle 8000, t=347.309), which §3 now
shows straddles the numerical onset almost exactly. Both ran to `nlim=8600`
(600 cycles, to t~372) and **both completed cleanly** -- 0 `NANS_IN_CONS`, 0
elliptic convergence failures, 0 asymmetric neighbour registrations.

> **"Clean" here means "no NaN in this window". It does NOT mean healthy, and it
> is NOT evidence the problem is fixed -- do not read it as production
> readiness.** The parent v2 run is *also* NaN-free at cycle 8600: its first
> `NANS_IN_CONS` is at **cycle 10659, t=446**, which is 2,059 cycles and 74 M
> past where these runs stop. A clean log over this window is what a sick run
> produces, so it discriminates nothing.
>
> Both arms in fact **end already unphysical**: `tau/M` = **1.337** (si=10) and
> **1.350** (si=1) at t~372, against the parent's 1.383 -- i.e. the fluid
> finishes carrying 34-35% more energy than its own rest mass, past §3's
> impossibility threshold (`tau/M` crosses 1.0 at t=370.5), with zero NaNs
> logged. The runaway is fully present and running at the parent's rate; only
> the terminal symptom is absent, because the runs stop before it.
>
> This is exactly §7's trap 3 (a test that cannot fail) wearing a different hat:
> mistaking a symptom-free window for a cured run. The load-bearing output of
> these jobs is the *growth rate*, not the exit status.

```
/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/
  tde_si_ab_ckpt8_si10/   job 8832465   solve_interval = 10   (control)  654 s
  tde_si_ab_ckpt8_si1/    job 8832467   solve_interval = 1    (test)     752 s
```

**RESULT -- the symptom did not move:**

| run | `tot-E` at t=347.44 | at t=372.0 | growth | `d(ln tot-E)/dt` |
|---|---|---|---|---|
| v2 parent (same window) | 1.1938e-05 | 1.1128e-04 | 9.32x | **0.09081 /M** |
| si=10 (control) | 1.1963e-05 | 1.0755e-04 | 8.99x | **0.08929 /M** |
| si=1 (test) | 1.1977e-05 | 1.0895e-04 | 9.10x | **0.08987 /M** |

The two arms track each other to within 1-2% at every output row, and the si=1
arm's growth rate is **0.65% HIGHER**, not lower -- well inside the row-to-row
scatter (the ratio wanders over 0.982-1.017). **`solve_interval` is not the
driver of the energy runaway.** It also means the §4a mechanism below (con2prim
at step 4 seeing a `psi^4`/`alpha` staleness that scales with `solve_interval`)
is not injecting energy at any magnitude that matters here.

**Keep `solve_interval=10`.** The 3.65x performance win stands and there is no
tradeoff to raise -- the measurement removed the reason to give it up.

**Why this result is trustworthy (the §7 trap-3 argument).**

1. *The parameter demonstrably took effect.* The restart deck inside each run's
   own `rst/*.00009.rst` records `solve_interval = 10` and `= 1` respectively
   -- the value each run actually parsed, not the one in the parent deck (which
   says 10 for both). This confirms empirically what `src/main.cpp:267` says by
   reading: with both `-r` and `-i`, the input file overrides the restart deck.
2. *The two runs are not the same run.* They diverge -- different `dt` traces
   from early on (e.g. at cycle 8598, 3.463841e-02 vs 3.475902e-02). Identical
   parameters on an identical decomposition would have been bit-identical.
3. *The reproducer actually reproduces.* Both arms recover the parent v2 run's
   growth rate over the same window to within 1.7%. This test could have shown
   a difference; it was not a test that cannot fail.

**A correction to this note's own earlier validity check, recorded because it
nearly caused a misreading.** §9.4 originally asserted "the si=1 arm MUST run
~3.65x slower per cycle, or the parameter did not take effect." **That was
wrong.** Item 61's 3.65x is `N=0` vs `N=10` (its cases are named `{N0,N10}`),
and `N=0` solves every *substage*, so it does `nexp_stages`x more solves than
`N=1`. The measured si=1/si=10 cost ratio here is **1.15x** (752 s vs 654 s),
which is not evidence of anything either way. The likely reason the ratio is so
small: the multigrid takes the previous solution as its initial guess, so
solving every cycle is a tiny, cheap correction, while the once-per-10-cycles
solve is a large, expensive one -- item 61 itself measured exactly this
bimodality (a 15-17 s spike every 10th cycle against 1.0-1.3 s in between).
Timing was therefore the wrong instrument; the restart-deck check (1 above) is
the right one.

### 9.5 Where this leaves the problem

**No cause is established, and all four of the original note's leads are now
spent.** 4a exonerated by measurement (9.4), 4b closed as the injection event
(9.2), 4d resolved as a downstream symptom (§4d), 4c argued down at the failure
site (9.3). The honest summary is that session 2 removed wrong answers rather
than finding the right one -- but it removed them *by measurement*, which is
what §7 asks for, and it did so cheaply.

What is now known that constrains the next hypothesis. The energy source is:

- **smooth, per-cycle and volumetric** -- not an event at regrids (9.2);
- **switched on at periapsis**, t~350, not at §3's t~360-378 (§3);
- **insensitive to elliptic-solve cadence** over a 10x range (9.4), which argues
  the metric being stale is not the problem -- and by extension points away from
  the CFC solve as the source and toward the **hydro/con2prim side**;
- **first visibly breaking `psi`** only at the very end (9.3), which is
  downstream: `psi`'s source is built from the matter, so NaN matter poisons the
  elliptic source, not the other way round;
- **reproducible in 600 cycles / ~11 minutes on 8 nodes** from checkpoint 00008
  (9.4). This is the single most useful thing session 2 produced: there is now a
  cheap, clean, quantitative reproducer of the runaway, with a matched control
  pair already on disk. Any future hypothesis can be tested against
  `d(ln tot-E)/dt ~ 0.090 /M` over cycles 8000-8600 for the cost of one short
  job.

Suggested next moves, in order:

1. **A resolution study on the post-periapsis debris. -- SET UP AND RUNNING, see
   §9.6.** The one remaining reading of 4b, and the only thing that cleanly
   separates "underresolved shocked debris" (a configuration problem, fixable)
   from "a code bug".
2. **Instrument the per-cycle `tau` budget.** With the source now localized to
   the hydro side, find which term adds energy: fluxes, the GR source terms, the
   atmosphere/floor resets in con2prim, or FOFC. The `ResetFloor` error policy
   and `dyn_grmhd_fofc.cpp` are the obvious first places to look -- a floor or
   fallback that fires on the low-density, high-velocity debris every cycle is
   exactly a smooth volumetric per-cell energy source, which is the shape 9.2
   measured. **Untested, and offered only as where to point the instrument.**
3. Only then consider whether the pre-periapsis secular rise needs its own
   explanation.

**Do not restart production on the strength of anything in this section.** §8's
constraint stands: the cause is not addressed, and a restart from cycle 10000
still hits the same wall.


### 9.6 Resolution study -- RUN. Resolution is a weak modifier, NOT the cause.

Three arms against the baseline `tde_si_ab_ckpt8_si10`, all from checkpoint
00008, `nlim=8600`, 8 nodes / 96 ranks, `solve_interval=10`, same binary; each
parfile differing from the baseline in only the line(s) named (diff-verified).
**All four completed cleanly: 0 `NANS_IN_CONS`, 0 FATAL/Abort, 0 convergence
failures. `lev5` did NOT hang** at the new refinement level (the multigrid
multi-level concern in project memory did not materialise here).

| case | job | change | MeshBlocks (start->end) | dt at cycle 8600 |
|---|---|---|---|---|
| `tde_si_ab_ckpt8_si10` | 8832465 | baseline | 1212 -> 1219 | 3.48e-02 |
| `tde_res_ckpt8_lev3` | 8832921 | `amr_level_star` 4->3 | 1212 -> **820** | 4.41e-02 |
| `tde_res_ckpt8_rad6` | 8832922 | `amr_radius_star` 2->6 | 1212 -> **3424** | 3.40e-02 |
| `tde_res_ckpt8_lev5` | 8832923 | `amr_level_star` 4->5, `num_levels` 5->6 | 1212 -> **2514** (8 logical levels, vs 7) | 2.27e-02 |

The knobs demonstrably took effect: the MeshBlock counts move the right way and
by large factors, `lev5` gains a logical level, and `dt` moves the right way in
every arm (the star sits where `alpha~1` and sets the global timestep, so
coarsening it *raises* `dt` -- `dyngr_tov.cpp`'s own comment predicts exactly
this). This is not a test that could not fail.

**A methodological correction to this section's own stated criterion.** §9.6
originally said to compare `d(ln tot-E)/dt` "over cycles 8000-8600". **That is
wrong and would have biased the result.** Because `dt` differs per arm, 600
cycles covers a *different amount of simulation time* in each (t=373.8 for
`lev3` down to t=361.5 for `lev5`), and the growth rate varies strongly across
this window (e-folding 11 M at t=350-370 vs 8 M at t=370-400). Arms must be
compared over a **common time window**, here **t = 347.442 to 361.528** (14.09
M, the overlap set by `lev5`'s shorter reach).

**RESULT (common window t=347.4-361.5):**

| arm | `dx` on debris | `d(ln tot-E)/dt` | vs baseline | `tau/M` at t=361.5 |
|---|---|---|---|---|
| `lev3` (coarser) | 0.125 | 0.057260 | **+1.6%** | 0.3325 |
| baseline | 0.0625 | 0.056377 | -- | 0.3284 |
| parent v2 | 0.0625 | 0.055991 | -0.7% | 0.3259 |
| `rad6` (3x coverage) | 0.0625 | 0.052768 | **-6.4%** | 0.3124 |
| `lev5` (finer) | 0.03125 | 0.047715 | **-15.4%** | 0.2911 |

The ordering **is** monotonic in resolution, so the growth is not entirely
resolution-independent. But the magnitudes say it is a weak modifier, not the
driver, on three independent readings:

1. **Coarsening the debris a full level moved the growth rate by 1.6%.** This is
   the single most diagnostic number in the table. If underresolved debris were
   injecting the energy, halving the debris resolution should have made it
   dramatically worse. It did essentially nothing.
2. **The sequence is not convergent.** Apparent order is p=+0.022
   (lev3->baseline) and p=+0.241 (baseline->lev5) -- against p~2 for a
   2nd-order-convergent truncation error, which would cut the rate ~4x per
   refinement. Worse, the successive differences **grow** on refinement
   (0.000883 then 0.008662, ratio 0.102 < 1) instead of shrinking. **There is no
   zero-growth limit to extrapolate to**, so "refine more" is not a fix.
3. **The best arm buys almost nothing in wall-clock terms.** Projecting each arm
   forward to `tau/M` = 1 (the §3 impossibility threshold): baseline t~381,
   `rad6` t~384, `lev5` t~387. These are *upper* bounds, since the rate
   accelerates -- the parent actually crossed at t=370.5. So a full extra
   refinement level buys **~6 M of headroom against the ~130 M** needed to carry
   the run from t=372 to `tlim=500`, at 2.1x the MeshBlock count.

**Caveats held honestly.** This is not a textbook convergence family: `lev3` and
`lev5` change the star box's level while the puncture box stays at 4, `rad6`
changes coverage rather than cell size, and each arm carries a different `dt` so
temporal and spatial truncation vary together. Three points is also few. And
`lev5` only reached t=361.5, so a benefit that only switches on later would be
invisible here. What the data *does* support is the negative: nothing in this
study behaves like a numerical error converging away under refinement.

**Conclusion: §4b is now closed in both of its readings.** Not the regrid events
(9.2), and not the refinement configuration either (this section). Underresolved
debris is not the cause, and **there is no configuration change that unblocks
production.** The remaining hypothesis is a code bug, and §9.5 item 2 is the next
move.

**Production: still blocked.** Nothing has been fixed; all leads are now spent
and no cause is established. See §9.7.

### 9.7 Production readiness -- NO, as of 2026-09-17

Asked and answered explicitly, because "the jobs ran clean" keeps looking like a
green light and is not one.

- **Nothing has been changed, let alone fixed.** `solve_interval` (9.4) and
  resolution (9.6) were both *hypotheses that failed*. The code and the
  production configuration are byte-for-byte what they were when the run died at
  cycle 10659. A restart from cycle 10000 today reproduces the same cascade in
  the same place -- that is the one outcome already predictable.
- **Every clean run in this investigation is clean only because it stops early.**
  The parent's first NaN is at cycle **10659, t=446**. All five 600-cycle
  reproducer runs stop at cycle 8600, t~362-374. The parent is NaN-free there
  too. A clean log over this window discriminates nothing.
- **Every arm, including the best, ends already unphysical.** Final `tau/M`:
  1.337 (baseline, at t=372), and 0.29-0.33 for all arms at t=361.5 and still
  growing exponentially at 0.048-0.057 /M. §3's bound is `tau/M` < ~O(0.1) for
  this system; > 1 is impossible.

**The gate for production**, unchanged and now with one lead fewer: a change that
materially lowers `d(ln tot-E)/dt` from the baseline's 0.0564 /M on the 9.4/9.6
reproducer, *and* whose run ends with `tau/M` well below 1 rather than merely
NaN-free. No change tested so far does either. Do not restart production until
one does.


---

## 10. Regrid timing vs the NaN, and the static-refinement test (2026-09-17)

Raised by the user, and the first part is now measured.

### 10.1 FACT: regridding is NOT the proximate trigger of the NaN

The question "is a regrid triggered right before the NaN?" is distinct from
§9.2's (which measured that the *energy growth* is not stepwise at remeshes).
Answered directly from both runs' `.out`:

| run | first `NANS_IN_CONS` | last regrid before it | gap |
|---|---|---|---|
| v2 | step after cycle 10659 | after cycle 10635 | **24 cycles, no regrid** |
| parityfix | step after cycle 10599 | after cycle 10543 | **56 cycles, no regrid** |

And in the failing step itself, `NANS_IN_CONS` prints **before** that step's
`ReinitializeMetricForAMR` line -- the regrid follows the NaN. The regrid burst
at cycles 10660-10663 in v2 is a *consequence* (AMR responding to a corrupted
field), not a cause.

Two independent reproductions, two different gaps, both large. **AMR regridding
does not trigger the NaN.**

### 10.2 FACT: the density-maximum tracker IS degenerate post-disruption, and it thrashes

The user's concern was correct, and it is now measured rather than argued.

**Method** (no job required, ~2 s): the `bin/` `prim_xy` dumps are z=0 slices of
`mhd_w_bcc` at `dt=1.0`, read with the in-tree `vis/python/bin_convert.py`
(`read_binary`; `h5py` only gates the athdf *writer*, so it can be stubbed).
For each dump: locate the max of `dens`, its position, the frame-to-frame jump,
and the density of the highest *competing* clump -- defined as the highest cell
further than 4.0 from the max, i.e. outside any possible overlap with the
`amr_radius_star = 2.0` ball. Script: `scratchpad/rhomax_track.py`.

**Caveat**: these are z=0 slices while `TDERefineTracker` uses the true 3D
global max. The orbit is in the xy-plane (`star_vel_x2`, `x3=0`), so the slice
should be representative, but an off-plane max would not be seen here.

**Result 1 -- the criterion becomes degenerate.** The second clump's density
climbs monotonically toward parity with the first:

| t | `rho_max` | `rho2/rho1` |
|---|---|---|
| 372 | 2.30e-04 | 0.010 |
| 390 | 3.46e-05 | 0.240 |
| 401 | 9.36e-06 | 0.626 |
| 417 | 4.92e-06 | **0.955** |
| 425 | 3.74e-06 | **0.986** |
| 435 | 3.47e-06 | **0.995** |
| 446 | 8.36e-07 | **0.994** |

`rho_max` itself falls **526x** from t=340 to t=446 as the debris rarefies. By
t>417 the top two clumps agree to within 5%, and by t=435 to within 0.5% --
**which of them is "the" maximum is then decided by noise.**

**Result 2 -- the refined box consequently teleports.** Frame-to-frame jump of
the max position (box radius is 2.0, so a jump > 4.0 relocates the ball with
*zero* overlap):

| window | max jump | jumps >1 | >2 | >4 |
|---|---|---|---|---|
| t=360-402 | 0.380 | 0 | 0 | 0 |
| t=402-446 | **6.147** | 4 | 3 | **3** |

The individual events:

```
  t=402.0  jump 1.119   ( 1.156, 7.406) -> ( 2.062, 8.062)   ratio before 0.626
  t=408.0  jump 0.981   ( 1.031, 8.969) -> ( 0.062, 8.812)   ratio before 0.510
  t=426.0  jump 6.147   (-5.688,10.062) -> (-8.094, 4.406)   ratio before 0.986
  t=436.0  jump 4.133   (-10.062,3.938) -> (-9.812, 8.062)   ratio before 0.995
  t=437.0  jump 4.640   (-9.812, 8.062) -> (-10.469,3.469)   ratio before 0.999
  t=442.0  jump 0.993   (-11.281,2.656) -> (-11.688,3.562)   ratio before 0.963
```

Note t=436 -> t=437: the max **oscillates back and forth between two clumps
~4.6 apart in consecutive samples**. The fine region is thrashing between them,
fully derefining and re-refining ~1200 cells' worth of mesh each time.

Before t=402 the motion is smooth (<=0.380 per M, i.e. well inside the box).
**So the tracker works exactly as designed while the star is intact, and breaks
down once the debris splits into comparable clumps -- precisely the failure mode
predicted.**

**What this does NOT show, stated plainly.** The thrashing is real but it is
**not the cause of the energy runaway**:

- **Wrong time.** Jumping starts at t~402. The runaway's numerical onset is
  t~350 (§3) and `tau/M` already crosses 1.0 at t=370.5 -- 30 M *before* the
  first jump. The energy problem is well established before the tracker
  misbehaves.
- **Wrong direction.** The measured growth rate during the jumping phase
  (t=400-425: **0.0455 /M**) is *lower* than during the smooth-tracking phase
  (t=350-400: **0.11 /M**). If thrashing were injecting energy, the rate should
  rise when it starts. It falls.

So this is a **second, independent defect**: the refinement criterion is invalid
post-disruption and is wasting/misplacing resolution, and it should be fixed on
its own merits for physics fidelity. It is not the NaN's cause, and fixing it
should not be expected to unblock production by itself. Whether the thrashing at
t=426-437 contributes to the *final* NaN at t=446 remains open -- the worst
events are 9-20 M before it, and §10.1 already showed no regrid in the 24
cycles immediately preceding it.

### 10.3 The static-refinement test -- RUN. AMR is fully exonerated.

Prediction recorded before the result (§10.1 showed a 24/56-cycle gap between
the last regrid and the first NaN): *the static arm should NaN anyway*. **It
did.**

| case | job | refinement | regrids | `NANS_IN_CONS` | first NaN | reached |
|---|---|---|---|---|---|---|
| `tde_frz_ckpt10_static` | 8833302 | **static** | **0** | **974,677** | cycle 10830 | t=500 (dt runaway) |
| `tde_frz_ckpt10_adaptive` | 8833303 | adaptive | 49 | 974,667 | cycle 10596 | t=500 (dt runaway) |
| `tde_frz_ckpt8_static` | 8833304 | static | 0 | 0 | -- | cycle 8600 |

**With literally zero regrids the cascade still happens, with the same magnitude
to within 10 occurrences out of ~974,670** (974,677 static vs 974,667 adaptive).
The identical `dt`-doubling runaway to `tlim` appears in both.

**And freezing the mesh does not change the growth rate either.** On the ckpt8
reproducer, over the common window t=347.4-371.4:

| arm | `d(ln tot-E)/dt` | `tau/M` at end |
|---|---|---|
| baseline (adaptive) | 0.087842 /M | 1.2217 |
| `ckpt8_static` (frozen, 0 regrids) | 0.086817 /M | 1.1863 |

A 1.2% difference. So neither proximate regridding (§10.1), nor accumulated
regridding, nor the energy growth rate, depends on AMR at all.

**Conclusion: AMR is exonerated in every reading tested** -- regrid events as
injection site (§9.2), regrid as proximate NaN trigger (§10.1), refinement
level/coverage (§9.6), and now regridding in aggregate (this section). Combined
with §11, this is expected rather than surprising: every one of these tests
varies the *debris* mesh, and §11 shows the failure is at the puncture.

The static arm's NaN arriving ~234 cycles later than the adaptive one (10830 vs
10596) is the only difference, and it is **not** evidence that AMR matters: a
frozen mesh is a different (coarser, more diffusive) discretisation of the debris
and the two runs diverge immediately, so the onset cycle is not expected to
match. The magnitude and character do.

---

## 11. CORRECTION (2026-09-17): the runaway is at the puncture, not in the debris

Prompted by the user doubting that the NaN follows from the energy growth, on the
grounds that post-disruption accretion onto the black hole would raise `tau`
naturally. That doubt was right to raise, and testing it overturned §3.

### 11.1 What was wrong, and why it survived so long

**§3 claimed**: post-periapsis `tau` and `|S|` never turn over as the debris
recedes, and `tau/M` reaching 2592 is impossible, therefore the growth is
numerical and global.

**The error**: `tot-E` is a whole-domain integral. It was read as a statement
about the debris. It is not -- it is dominated by a tiny region around the
puncture. The turnover §3 said "never happens" **does happen** in the debris; it
was masked by the puncture contribution.

This is §7 trap 2 again (an un-instrumented inference from a real observation),
and it is the single most expensive mistake in this investigation: it aimed §4a,
§4b, §9.4 and §9.6 at the debris, which was never the problem. Their negative
results are all consistent with that.

### 11.2 FACT: the energy is at r<2 and carries almost no mass

Method: `bin/` `prim_xy` + `adm_xy` z=0 slices (`dt=1.0`), reconstructing the
conserved variables from primitives. `velx` is the spatial 4-velocity `u^i`
(confirmed from `primitive_solver_hyd.hpp`'s `iWsq = 1/(1+Contract(uu,ud))`), the
CFC 3-metric is `psi4*delta_ij`, `Gamma=5/3`:
`W=sqrt(1+psi4|u|^2)`, `sqrt(g)=psi4^1.5`, `D=sqrt(g)rho W`,
`tau=sqrt(g)((rho+2.5p)W^2-p)-D`. Script: `scripts/tau_radial.py`.

**Fraction of slice-integrated `tau` inside r<2**: 0.06 (t=340) -> 0.37 (t=360)
-> 0.73 (t=370) -> 0.96 (t=380) -> **0.997+ from t=390 onward**.

**Absolute, and this is the decisive table:**

| t | `tau(r<2)` | `tau(r>2)` | `tau/D` (r<2) | `tau/D` (r>2) |
|---|---|---|---|---|
| 340 | 2.87e-06 | 4.51e-05 | 1.2e+06 | 0.126 |
| 370 | 2.03e-04 | 7.37e-05 | 5.1e+05 | **0.145** |
| 400 | 4.70e-02 | 2.88e-05 | 5.1e+06 | 0.099 |
| 446 | 6.52e-02 | 4.63e-06 | 1.1e+06 | 0.048 |

- **`tau` inside r<2 grows 22,675x.** Global `tot-E` grew 19,823x over the same
  span, so the global diagnostic *is* the r<2 component (2D slice vs 3D volume
  accounts for the rest).
- **`tau` outside r<2 DECREASES 10x** (4.51e-05 -> 4.63e-06). There is no energy
  runaway in the debris.
- **The debris specific energy turns over exactly as physics requires**:
  0.126 -> peak **0.145 at t=370** (periapsis shock heating) -> 0.048. Rise
  through periapsis, decline as the debris expands and recedes. This is the
  turnover §3 asserted never happens.
- **The r<2 region holds essentially no rest mass.** `D(r<2)/D_total` is ~0
  throughout (2e-12 at t=340, rising only to ~2e-3 at t=440). Meanwhile the
  debris mass moves *outward*: `M(5-10)` 0.997 -> 0.037, `M(>10)` 0.003 -> 0.96.

**So it is not accretion either.** The user's proposed mechanism predicts mass
*and* energy arriving together at the hole; the measurement shows energy with
essentially no mass. `tau/D` at r<2 is ~1e6, i.e. a specific energy of ~10^6
c^2 carried by a vanishing amount of material. `Wmax` reaches **1697 at r=0.22**
(t=440). This is low-density atmosphere being accelerated at the puncture, not
matter being swallowed.

### 11.3 FACT: the lapse is pinned at `alpha_floor` exactly where this happens

| t | `alpha_min` | at r | `rho` there |
|---|---|---|---|
| 340 | 1.00000e-06 | 1.458 | 6.25e-17 |
| 400 | 1.00000e-06 | 1.458 | 7.38e-11 |
| 446 | 1.00000e-06 | 1.458 | 2.31e-09 |

`alpha` sits at **exactly** `alpha_floor = 1.0e-6` at every time sampled, at
r~1.46, and the density there climbs **8 orders of magnitude**. `excise = false`
in this run, so there is no sink: the floor is the only thing preventing the
`1/alpha` singularity (parfile's own note, item 59).

**This reopens §4c, and retracts the reason it was argued down.** §9.3 dismissed
`alpha_floor` because `alp = 0.8657` at the first `NANS_IN_CONS` print. That was
the wrong place to look -- it is where the NaN *surfaces*, not where it
originates. The lapse is floored precisely where the `tau` runaway lives.

### 11.4 LEAD (not measured): why the NaN is global and simultaneous

A chain consistent with everything in §9-§11, offered as a hypothesis, with the
measured and unmeasured steps marked:

1. *(measured)* `alpha` pinned at the floor around r~1.5 with no excision.
2. *(measured)* Atmosphere there accelerates (W up to 1697) and its density rises
   8 orders; `tau(r<2)` grows 22,675x.
3. *(NOT measured)* That `tau` enters the elliptic source for `psi`. Once it is
   large enough to overflow, the source goes non-finite.
4. *(NOT measured, but strongly consistent)* **The `psi` solve is a global
   elliptic solve**, so a bad source anywhere poisons `psi` *everywhere in one
   step*. This would explain three otherwise-awkward facts at once: the NaN
   appears globally and simultaneously (§1: NaN sites at all radii); `psi4` is
   NaN while `alpha`/`vK_dd` are finite (§9.3 -- they are the previous solve's
   values, written before the bad solve); and the elliptic solver reports
   convergence right up to the failure (§2).

**Test it before believing it.** The cheap version: instrument the `psi` source
(or just `max|tau|` and its location) each solve and watch it against the NaN.

### 11.5 What this means for the open work

- **The debris-side results stand but were aimed at the wrong target.**
  `solve_interval` (§9.4), debris resolution (§9.6), regrid events (§9.2, §10.1)
  are all genuinely exonerated -- and their uniform null results are now
  *explained*: none of them touches r<2.
- **§10.2's tracker degeneracy is still a real defect** and still worth fixing,
  but it is now clearly a fidelity issue, not the cause. It is in the debris.
- **The three static-refinement jobs in §10.3 are now near-irrelevant to the
  cause** (they vary debris meshing). Left running; they will still answer §10.3's
  narrow question.
- **Next move is the puncture, not the debris.** In rough order: instrument
  `tau`/the elliptic source at r<2 per solve (§11.4 step 3-4); then examine
  `excise=false` and the `alpha_floor=1e-6` band-aid, which is where the
  modified equation lives.
- **Production: still blocked**, and §9.7's gate is now known to have been
  measuring the wrong quantity. A corrected gate should track `tau(r<2)` and
  `alpha_min`, not global `tot-E`.

### 11.6 Density-threshold refinement cost (for the §10.2 fix, when it is taken up)

In-plane level-4 MeshBlock footprints covering `{rho > f*rho_max}`. **z=0 slice,
so this is the IN-PLANE count**; the true 3D count is this times the debris'
vertical extent in blocks.

| t | `rho_max` | f=1e-1 | f=1e-2 | f=1e-3 | f=1e-4 |
|---|---|---|---|---|---|
| 350 | 3.62e-04 | 4 | 16 | 30 | 50 |
| 390 | 3.46e-05 | 27 | 74 | 167 | 266 |
| 410 | 6.66e-06 | 73 | 194 | 383 | 547 |
| 446 | 8.36e-07 | 216 | 566 | 1020 | 1410 |

Current mesh is ~1212-1275 blocks total (3D); the `rad6` arm reached 3424 and ran
fine on 8 nodes. So f=1e-2 is affordable even allowing several blocks of vertical
extent; f=1e-3 is borderline late in the run; f=1e-4 is not affordable without a
level cap.


---

## 12. The Lorentz factor at the puncture, and the ceilings that exist (2026-09-17)

Prompted by the user asking (a) whether W is very high there, and (b) whether a
Lorentz-factor or temperature limiter exists that could prevent the NaN. Both
answered; (b) turns up a configured-but-ineffective ceiling.

### 12.1 FACT: W reaches 1e7 at the puncture, and `gamma_max` is 50

Same slice method as §11.2 (`W = sqrt(1 + psi4|u|^2)`; `velx` is `u^i`, confirmed
again at `primitive_solver.hpp`'s `prim[PVX] = Wv_u[0]`).

| t | `W_max` | at r | n(W>10) | n(W>50) | n(W>500) | max W at r>2 |
|---|---|---|---|---|---|---|
| 340 | **1.00e+07** | 0.044 | 4 | 4 | 3 | 1.81 |
| 400 | **1.01e+07** | 0.099 | 9 | 6 | 6 | 2.48 |
| 430 | **1.03e+07** | 0.133 | 13 | 9 | 9 | 2.50 |
| 440 | **1.07e+07** | 0.238 | 26 | 24 | 22 | 2.47 |
| 446 | **1.00e+07** | 0.633 | 94 | **84** | 78 | 4.99 |

- **W ~ 1e7** in a handful of cells at r = 0.04-0.6, i.e. right at the puncture.
  W=1e7 means `v = 1 - 5e-15`, at the edge of double precision -- `1-v^2` and
  `1/W^2` there carry essentially no significant digits.
- **The debris is fine**: max W outside r>2 is 1.8-5.0 throughout.
- **The affected cell count grows sharply into the cascade**: 4 -> 9 -> 13 -> 24
  -> **84** at t=446, the last clean output before the NaN.

**A configured ceiling exists and is not holding.**
`primitive_solver_hyd.hpp:144` reads `<mhd> gamma_max` (**default 50**, and it is
NOT set in this run's parfile, so 50 applies) and converts it to `v_max` via
`SetMaxVelocity`. Measured output is `W = 1e7`, i.e. **2e5 times the nominal
ceiling**. Tracing where `v_max` is applied:

- It is used in exactly one place: `primitive_solver.hpp:113`, capping `vsq_max`
  (hence `vhatsq`) -- the velocity *estimate inside the c2p root-find iteration*.
- `ResetFloor::PrimitiveFloor` (`reset_floor.hpp:35`) has **no velocity clamp**.
  It zeroes velocity only when `n < n_atm*n_threshold`, and otherwise only floors
  temperature.
- `CheckDensityValid`'s `W_max = sqrt(1 + rsq/h_min^2)` is computed from the
  conserved state and is unrelated to `gamma_max`.
- `eos_data.gamma_max` in `ideal_grhyd.cpp:140`/`ideal_grmhd.cpp:153` *does*
  rescale velocity when `lor > gamma_max` -- but those are the **non-dyn_grmhd**
  ideal-gas C2P paths, not the PrimitiveSolver path this run uses.

So the converged `u^i` written to `w0` is never clamped to `gamma_max` on this
code path. **Whether that is a bug or an accepted design limitation is not
established here** -- what is established is that the parameter does not bound
the output, so *tightening `gamma_max` alone should not be expected to fix this.*
That is worth one cheap test rather than assumption (see 12.3).

### 12.2 FACT: the runaway cells are COLD and tenuous -- a temperature ceiling cannot help

Properties of the `W>50` cells (`dfloor=1e-24`, `tfloor=1e-22`, `dthreshold=1.02`):

| t | n(W>50) | `rho` range | `T` (all cells) | `rho_min`/`dfloor` |
|---|---|---|---|---|
| 340 | 4 | 1.1e-23 .. 6.7e-20 | **1.000e-22 = tfloor** | 10.8 |
| 400 | 6 | 1.9e-20 .. 1.8e-17 | **1.000e-22 = tfloor** | 1.9e+04 |
| 440 | 24 | 6.9e-19 .. 2.8e-12 | **1.000e-22 = tfloor** | 6.9e+05 |
| 446 | 84 | 4.2e-24 .. 2.4e-12 | **1.000e-22 = tfloor** | 4.2 |

Two consequences, both directly answering the user's question:

1. **`T` is pinned exactly at `tfloor` in every one of these cells at every
   time.** The material is *cold*, not hot. A temperature **ceiling** would never
   engage; it is the floor that is active. So a `T_max` cannot help here.
2. **The atmosphere reset never fires on them.** `PrimitiveFloor` resets
   (`v -> 0`) only when `n < n_atm*n_threshold`, i.e. `rho < 1.02e-24`, but these
   cells sit **10x to 7e5x above `dfloor`**. They are too dense to be treated as
   atmosphere and too tenuous to be real fluid -- they fall straight through the
   gap between the two.

This is the mechanism in one sentence: **cold, tenuous material accumulates at
the unexcised puncture where the lapse is pinned at `alpha_floor`, is accelerated
to W~1e7, is never caught by the density floor or the velocity ceiling, and its
`tau` grows 22,675x until it destroys the global elliptic solve.**

### 12.3 Excision exists, is OFF in this run, and has a documented silent-failure mode

The user's framing -- that such material "should be masked very quickly after
falling in" -- is exactly excision, and this code has it:
`<coord> excise` (**defaults to true**; this run sets it **false**), plus
`rexcise`, `dexcise`, `texcise`, `excision_scheme` (`fixed`/lapse/horizon),
`tdamp`, with damping wired into the RHS (`dyn_grmhd.cpp:697-713`) and into c2p
(`primitive_solver_hyd.hpp:319`).

**A trap recorded in the source, which any excision attempt must clear.**
`dyn_grmhd.cpp:248-261`: the dynamic (`lapse`/`horizon`) schemes rebuild their
masks from the current lapse each cycle, but that refresh was gated on
`<adm> dynamic=true`, **which no CFC fixture sets**. The result was that
`excision_scheme=lapse` "allocated its masks, never wrote them, and silently
excised nothing at all in every CFC run" -- verified in a TDE run where density
inside the nominal excision radius sat at 1.9e-9 instead of `dexcise=1e-24`. The
fix (queueing the refresh unconditionally) is in the tree, but the lesson stands:
**an excision run must be verified to have actually excised** -- check `rho`
inside `rexcise` against `dexcise`, and check the `excised_tally`. Do not accept
"it ran clean" as evidence excision was active. This is §7 trap 3 with a specific
known instance.

### 12.4 Candidate fixes, in the order they should be tried

All untested. Each is cheap to test on the §9.4 reproducer, but note the
**diagnostic gate must change**: track `W_max`, `n(W>50)`, `tau(r<2)` and
`alpha_min`, NOT global `tot-E` (§9.7's gate measured the wrong quantity, §11.5).

1. **Turn excision on** (`excise=true` with a sensible `rexcise`, `dexcise`,
   `texcise`). Most physical, matches the user's reasoning, and removes the
   pile-up region entirely rather than patching its symptoms. Must be verified
   per 12.3. Main risk: `rexcise` interacting with `puncture_mass`/`alpha_floor`,
   and this fixture has never run with excision on.
2. **Raise `dfloor`** so the puncture material is caught by the atmosphere reset.
   To catch `rho ~ 2.4e-12` needs `dfloor` ~1e-12, a 1e12 raise. Debris `rho_max`
   at t=446 is 8.4e-7, and the f=1e-4 debris contour is ~8.4e-11, so ~1e-12 sits
   just below the debris -- plausible but it is a domain-wide change with obvious
   potential to corrupt the debris tail. Cheap to test, needs a debris check.
3. **Enforce `gamma_max` on the c2p output** (a code change: clamp `u^i` after the
   root find, as `ideal_grmhd.cpp:153` already does on its own path). Most
   targeted, and arguably what the existing parameter was meant to do. But it
   treats the symptom; the material would still accumulate.
4. **Revisit `alpha_floor=1e-6`**. It is the modified equation at the heart of
   this (§11.3) and is only in place because there is no excision (item 59). If
   (1) works, this may become unnecessary.

**Do not expect (3) alone to be sufficient** -- capping W stops the precision
loss but not the accumulation, and `tau` would likely keep growing more slowly.
(1) is the one that addresses the cause.


---

## 13. Excision test -- RUNNING (job 8833552), and a correction on the scheme

### 13.1 Correction: `excision_scheme` must be `lapse` for CFC, not `fixed`

This session first configured the excision test with `excision_scheme=fixed`,
reasoning from `dyn_grmhd.cpp:248-261` that the dynamic schemes had a
silent-no-op history and `fixed` was therefore the safe choice. **That was
wrong** (user correction; DEVELOPMENT.md (cfc) item 57 is the reference), for
three reasons, two of them measurable:

1. **`fixed` cannot cover the pathology.** `rexcise` is **not a parfile
   parameter** -- `coordinates.cpp:69` hardcodes it to **1.0** absent a
   `<radiation>` block. Measured: at t=446 the `W>50` region reaches r=**1.609**,
   and **5 cells with W>50 sit outside r=1.0**, i.e. exactly the cells that begin
   the cascade would have been left un-excised. The `lapse` criterion
   (`alpha < excise_lapse = 0.25`) covers r<**1.495** and is rebuilt every cycle.
2. **CFC accretes the excised mass to the puncture, so M_BH grows.**
   `CFC::AccreteExcisedMass` (`<cfc> accrete_to_puncture`, default **true**,
   gated only on `bh_excise`) hands the hole `(D+tau)/psi` -- not `(D+tau)`;
   item 57 measured `<1/psi>` = 0.4550, so the raw conserved energy would
   over-credit by 2.20x. This exists because CFC's elliptic metric has *no
   memory*: deleting matter deletes its gravity in the same step, so excised
   matter must be handed to `M_BH` or the hole loses what it swallowed. As M_BH
   grows the strong-field region grows with it, and **only a lapse-based surface
   tracks that; a hardcoded radius cannot.**
3. **The historical worry is already fixed.** Item 57's no-op arose because
   `UpdateExcisionMasks` was queued only under `<adm> dynamic=true`, which no CFC
   fixture sets. `dyn_grmhd.cpp:261` now queues the refresh whenever excision is
   on with a non-fixed scheme. The comment recording the old bug was read here
   and mistaken for a live hazard.

**Method note for the future reader**: the failure mode was reading a source
comment about a *past* bug and treating it as a current constraint, without
checking whether the fix was in the tree -- the comment's own last line
("So queue the refresh here too") said it was. A near-miss of §7 trap 1, a theory
outliving its evidence.

### 13.2 The run

`tde_exc_ckpt10_16n`, job **8833552**, **16 nodes / 192 ranks** (user's choice;
this does NOT match the checkpoint's 8/96, so per-cycle *timing* is not
comparable to earlier arms -- correctness is unaffected). From ckpt 00010
(cycle 10000, t=422.239) to `nlim=10900`, past both observed onsets
(10659 v2, 10599 parityfix, 10830/10596 in §10.3) with margin.

Settings: `excise=true`, `excision_scheme=lapse`, `excise_lapse=0.25`,
`dexcise=1.0e-24`, `texcise=1.0e-22` (matched to this run's `dfloor`/`tfloor`),
`accrete_to_puncture` default true, plus `<cfc> adm_mass_r0/r1/r2 = 20/30/40`.

### 13.3 Verification: excision IS working (all four criteria pass)

Item 57's checks, run before interpreting anything -- because item 57 records
that a previous "excision validated" claim was retroactively invalidated when
the masks turned out never to have been written.

| t | run | `rho(r<1)` | `W_max(r<1)` | `tau(r<2)` | n(W>50) |
|---|---|---|---|---|---|
| 423 | v2 (no excision) | 1.51e-09 | **1.03e+07** | 1.49e-01 | 11 |
| 423 | **excision** | **1.000e-24** | **1.000** | **3.13e-09** | **0** |
| 433 | v2 | 2.14e-08 | 1.04e+07 | 2.32e-01 | 16 |
| 433 | **excision** | **1.000e-24** | **1.000** | **4.65e-09** | **0** |

1. **`rho` inside r<1 sits at exactly `dexcise = 1e-24`** -- the precise check
   that caught the old no-op (which showed 1.9e-9 instead).
2. **`W` there is exactly 1.000**, against 1.03e7 in v2. Zero `W>50` cells
   anywhere in the domain.
3. **`puncture_mass` grew**: `M_BH` 1.0 -> 1.00564292 on the first accretion
   drain (cycle 10010), `dM_adm=5.64e-3` from `dM_raw=1.89e-2` at
   `<1/psi>=0.298` -- i.e. it swallowed the accumulated pile-up at once, then
   settled to `dM ~ 1e-10` per drain.
4. **The independent ADM surface integral agrees**: `M_total = 1.00572` at
   r=20/30/40, consistent with `M_BH + M_res`.

**`tau(r<2)` fell by ~8 orders of magnitude.** The puncture pile-up that §11 and
§12 identified is completely eliminated.

### 13.4 RESULT: the cascade happens anyway, EARLIER -- §11.4 is refuted

| run | refinement | first NaN | `NANS_IN_CONS` |
|---|---|---|---|
| v2 original | adaptive | cycle 10659 (t=446.0) | 974,603 |
| `ckpt10_adaptive` control | adaptive | cycle 10596 (t=444.2) | 974,667 |
| **`tde_exc_ckpt10_16n`** | adaptive + **excision** | **cycle 10351 (t=436.0)** | **1,949,347** |

With the puncture pathology entirely removed, the cascade arrives **245 cycles
earlier** and with **twice** the occurrences.

**This refutes §11.4.** That section hypothesised: puncture `tau` runaway ->
elliptic `psi` source overflows -> global NaN. The `tau` runaway is gone
(13.3) and the NaN is not. It was explicitly marked "NOT measured"; this is the
measurement, and it says no.

**What §11 still gets right**: the debris-vs-puncture localisation of the
*energy* (§11.2) is unaffected -- it was measured directly, and excision
confirmed it by removing exactly that energy. What falls is only the causal link
from that energy to the NaN.

`puncture_mass = nan` appears in the output restart deck, but that is a
*consequence*, not a cause: accretion was healthy at cycle 10350
(`dM=2.09e-9`, `<1/psi>=0.993`), the first `NANS_IN_CONS` is at 10351, and the
accretion diagnostics only go NaN at the next drain, 10360.

### 13.5 FACT: it is the VECTOR POISSON solve that returns NaN, in BOTH runs

The excision run gives a far cleaner diagnostic than the original ever did,
because its first failing `bin` dump is only one output after a fully clean one.

**Excision run, per-field NaN counts over the whole z=0 slice:**

| field | t=435 | t=436 |
|---|---|---|
| `adm_psi4` | 0 | **0** |
| `adm_alpha` | 0 | **0** |
| `adm_Kxx` | 0 | **0** |
| `adm_betax/y/z` | 0 | **52480 / 52480** |
| `dens`, `velx`, `press` | 0 | **0** |

**Only the shift is NaN, in 100% of cells, in a single step, from inputs that are
all clean** -- and with `grep -c "Failed to converge"` = **0**, i.e. the multigrid
reported success while returning an entirely NaN field.

**The original v2 run shows the same thing** (t=447.9, post-cascade): `adm_betax`
and `adm_Kxx` 100% NaN (28672/28672), `adm_psi4` and `adm_alpha` **0%** NaN.

The common factor: `beta^i` (step 6) and `Adual^ij`/`K_dd` (via `X^i`, steps 1-2)
are exactly the quantities produced by the **vector** Poisson solves
(`MGCFCVectorPoisson`, `mg_cfc_vector_poisson.cpp`). The **scalar** solves
(`psi`, `alpha*psi`) stay clean in both runs. Shibata's decomposition gives
steps 1 and 6 "the same vector-equation form, different source".

**Weight of evidence, stated honestly**: for the excision run this is strong --
first failing dump, one output after clean, single field, 100% of cells, clean
inputs. For v2 it is suggestive rather than equally strong, because the only
available dump is post-cascade (t=447.9), by which point the state has evolved
under the `dt` runaway; v2's own first-NaN *print* (§9.3) showed `psi4=nan` at
one cell, which does not match, and the two cannot be reconciled from the data on
hand. **The v2 first-NaN print site and the excision run's should both be treated
as "first block to print", not "where it started"** -- MeshBlock 0 at the domain
corner is simply rank 0's first block.

### 13.6 OBSERVATION (unexplained): a second atmosphere pathology at the OUTER boundary

Present in **both** runs, essentially identically, so not caused by excision:

| t | `rho_max` at chebyshev r>56 | `|u|_max` there |
|---|---|---|
| 423 | 4.12e-22 | 2.18 |
| 428 | 7.50e-22 | 2.20 |
| 432 | 1.29e-20 | 2.23 |
| 435 | 2.73e-20 | 2.23 |

Density at the outer boundary rises **66x in 12 M** while `|u| ~ 2.2` (W~2.4).
The boundaries are `diode`. Not implicated in anything yet -- recorded because it
is a second place where the atmosphere is being accelerated, and because the
vector Poisson solve is the one whose outer boundary condition would be most
exposed to it.

### 13.7 Where this leaves it

**Production: still blocked.** Excision is a real fix for a real, separately
verified defect (the puncture pile-up, §11/§12, now demonstrably removed) and
should probably be kept on its own merits -- but it does not fix the cascade, and
it makes the onset earlier.

**Everything the investigation has excluded so far**: `SetNeighbors`/the
neighbour table; elliptic *convergence failure*; matter creation; `solve_interval`
cadence; debris refinement level and coverage; AMR regridding in every reading
(events, proximate trigger, aggregate, frozen mesh); the density-max tracker
(degenerate, §10.2, but a fidelity defect not the cause); the debris energy
(healthy, §11.2); and now the puncture energy pile-up (§13.3-13.4).

**Next move**: `mg_cfc_vector_poisson.cpp`. The question is no longer "what
physics blows up" but "why does a multigrid vector solve return an all-NaN field
from clean inputs while reporting convergence". Concretely: instrument the vector
solve's source and residual per solve (`<cfc> mg_verbose=1` exists), and check its
outer boundary condition against 13.6. The excision run is the better reproducer
for this -- it fails 245 cycles sooner and its failing dump is one output after a
clean one.

**Do NOT re-test debris-side or matter-side hypotheses without a reason that
survives §11.2 and §13.4.** Six independent debris/matter hypotheses have now
come back null, and the one measurement that isolates the failing object points
at the solver.


---

## 14. Elliptic-source instrumentation (2026-09-17) -- job 8835173

User's line of reasoning, which set this up: *the beta NaN probably comes from
K_dd -- but if K_dd were NaN it would have to show in psi4 and alpha immediately,
because Ahat^2 enters both scalar sources.* That is a genuine constraint, and it
is what makes the measurement worth taking.

### 14.1 The constraint rules out the K_dd route on data already in hand

Task order (`cfc.cpp:447-472`) is `CFC_BuildSrcX` -> `CFC_ComputeADual` ->
`CFC_SolvePsi` -> `CFC_RescaleSrc` -> `CFC_SolveLapse` -> ... -> shift. So
`Adual^ij` is built **before both scalar solves**, and `Ahat^2` appears in both
their sources. Measured (§13.5): `adm_psi4` and `adm_alpha` are **0% NaN** in the
step where `adm_betax/y/z` are **100% NaN**. Therefore `Adual` was finite when the
scalar solves consumed it, and **the corruption is downstream of `Adual`, not in
it.** `K_dd = a_dd*psi^-2` is written at the very end (`AssembleLapseShiftK`), so
a NaN appearing there is a symptom, not the origin.

### 14.2 FACT: `psi` does not approach zero -- a tempting route, closed

Before instrumenting, one hypothesis was cheap enough to kill from the dumps. The
shift source is the only place `psi^-7` appears
(`ap6 = alpha_psi/psi7`, and `dap6`'s `psi_inv7`/`onepx_inv = psi0*psi_inv`), so a
`psi` zero-crossing would blow it up. Critically, **that would have been invisible
in both diagnostics used so far**: `psi4 = psi^4` is sign-blind and stays positive,
and `alpha = alpha_psi/psi` would go negative and then be clamped by
`fmax(alpha, alpha_floor)` to exactly the floor -- which is what `alpha` already
reads.

Measured over t=423-436 in both runs: `min(psi4) ~ 0.99`, i.e.
**`min(psi) ~ 0.997`, stable, nowhere near zero**. The hypothesis is dead. Recorded
because it fits every prior observation and the next reader will think of it too.

### 14.3 What was added

`<cfc> src_nan_check` (default **0** = off, so the checks sit behind one branch and
the instrumented binary stays valid for production) plus `CFC::CheckFinite`
(`cfc.cpp`, declared in `cfc.hpp`): a Kokkos reduction that counts NaN/Inf in an
array and reports the globally-first offender via `MPI_MINLOC` -- rank, MeshBlock,
channel, index, coordinates, value. Call sites, ordered as the data flows:

| label | what it is |
|---|---|
| `X.src[u_p_src]` / `X.out[u_p_x]` | the X^i solve's source and output -- the **control**: same `MGCFCVectorPoisson` machinery, far simpler source (`8*pi*S_i`, no `psi^-7`, no derivatives) |
| `adual[u_adual]` | `Adual^ij`, tests §14.1 directly |
| `psi.out[delta_psi]` | conformal factor output |
| `lapse.out[delta_alpha_psi]` | lapse output |
| `shift.dap6[u_alpha_psi6]` | `Delta(alpha*psi^-6)` -- the one place `psi^-7` and a `psi0*(1/psi)` product appear |
| `shift.src[u_p_src]` | the assembled shift source |
| `shift.out[u_p_beta]` | the shift solve **output** |

### 14.4 The run, and how to read it

`tde_ellipdiag_ckpt10_16n`, job **8835173**: the excision case (§13) unchanged --
excision on, `lapse` scheme, ckpt 00010, `nlim=10900`, 16 nodes -- plus
`mg_verbose=1` and `src_nan_check=1`. Excision is deliberately kept **on** because
that run is the better reproducer: it fails at cycle 10351 instead of 10596, with
a clean `bin` dump one output before.

Binary rebuilt on aurora-uan-0007, md5 **f4f2227ed6590a8fb8216fc5a1fc7d21** -- NOT
the `890a0a29...` binary every earlier arm used.

**The decision this makes:**

- `shift.dap6` or `shift.src` reports **before** `shift.out` => the source handed
  to the multigrid was already non-finite. Then the question is which factor, and
  §14.2 says it is not a `psi` zero-crossing.
- `shift.out` reports while `shift.dap6`/`shift.src` stay clean => **the multigrid
  produced NaN from a finite source**, which is what §13.5 infers but has never
  measured. `mg_verbose=1`'s residual trace then becomes the primary evidence,
  especially against the 0 reported convergence failures.
- `X.src`/`X.out` staying clean while the shift pair reports is the control
  passing: same solver, different source.

`mg_verbose=1` covers all three solvers (`mg_cfc_vector_poisson.cpp:251`,
`mg_cfc_conformal_factor.cpp:442`, `mg_cfc_lapse.cpp:411`).


### 14.5 Round 1 RESULT (job 8835173): the instrumentation fired too late

**The run did not answer the question, and the reason is a flaw in the
instrumentation, not in the physics.** Recorded in full because the failure mode
is reusable.

Timeline from the `.out`:

```
cycle 10010   CFC accretion: dM_adm=5.6735e-03  dPx=-1.2404e-01   (pile-up dump)
cycle 10016   *** first NANS_IN_CONS ***
cycle 10020   ### CFC NONFINITE [X.src[u_p_src]]   count=16508044
              ### CFC NONFINITE [adual[u_adual]]   count=22388736
              ### CFC NONFINITE [shift.src[u_p_src]] count=16508044
```

Only **3** reports, all at cycle 10020 -- **four cycles after** the matter had
already gone NaN. **Every CFC task, and therefore every `CheckFinite` call, is
gated by `DoSolveThisStage`**, so at `solve_interval=10` the checks run only on
every 10th cycle. They measured the aftermath. The three arrays that reported are
exactly the ones built *from the matter*, which by then was NaN.

Note also what did NOT report, which is informative in itself: `X.out[u_p_x]`,
`psi.out`, `lapse.out`, `shift.dap6` and `shift.out[u_p_beta]` were all **clean at
cycle 10020** even though their sources were NaN. That is consistent with the
multigrid's outer boundary condition dominating a source that has gone bad, but
it is not evidence for anything and should not be read as such.

**The failure signature is reproducible**, and matches §13.5 exactly: first
`NANS_IN_CONS` at cycle 10016 shows `beta = {-nan,-nan,-nan}` with `detg`,
`g_dd`, `alp = 0.9896`, `psi4 = 1.0211` and `K_dd` all finite, at the outer
boundary. The last solve before it was cycle 10010, so **that** solve produced the
NaN `beta`.

### 14.6 FACT: excision runs are NOT reproducible -- onset cycle is not a metric

Two runs of the identical configuration (same checkpoint, same parfile except
diagnostics):

| run | first accretion `dM_adm` | first NaN cycle |
|---|---|---|
| `tde_exc_ckpt10_16n` (8833552) | 5.6429201165e-03 | **10351** |
| `tde_ellipdiag_ckpt10_16n` (8835173) | 5.6735464477e-03 | **10016** |

A 0.54% difference in the first accretion event, and a **335-cycle** spread in
onset. Item 57 records that `excised_tally` is filled by **atomic adds**, whose
GPU summation order is nondeterministic; the rebuilt binary is a second candidate
and the two cannot be separated from these runs alone.

**Consequences, which bind any future work here:**
- **The onset cycle cannot be used as a metric for excision runs**, and no A/B may
  be run on it. §13.4's "excision made it 245 cycles earlier" must accordingly be
  **downgraded**: that difference is within the observed run-to-run spread and is
  not evidence that excision hurts.
- What *is* reproducible is the **signature** -- `beta` 100% NaN with
  `psi4`/`alpha`/`K_dd` and every primitive clean -- seen identically in both runs
  and in v2. Signature, not timing, is the thing to test against.

### 14.7 Round 2 -- job 8835347, the two fixes

`tde_ellipdiag2_ckpt10_16n`. Same case, with:

1. **`solve_interval = 1`** so the gated checks run **every cycle** (production
   value stays 10; §9.4 exonerated it, and this is a diagnostic run only).
2. **`CheckFinite("matter.in[pmhd->u0]")` added as the first call of the cycle**,
   before any elliptic work -- the ordering test round 1 could not make.
3. `CheckFinite` now **skips ghost zones** (`u_p_src` is pure solver input, never
   ghost-exchanged, so its ghosts are never written).

**The decision it makes:**

- **`matter.in` reports first at a cycle** => the conserved state was already bad
  when the solves began; every elliptic NaN that cycle is downstream, and the
  elliptic solver is exonerated. The question then moves to what corrupts the
  matter -- and note §13.6's outer-boundary atmosphere, since both this run's and
  the excision run's first NaN sit at the outer boundary.
- **Elliptic labels report while `matter.in` is clean** => the solve originated
  the NaN, with `X.src`/`X.out` as the control (same `MGCFCVectorPoisson`, simple
  source, no `psi^-7`, no derivatives).


### 14.8 ROUND 2 RESULT (job 8835347): the NaN is born in the SHIFT SOURCE

**This is the first measurement that isolates the origin.** With
`solve_interval=1` the checks run every cycle, and the progression is unambiguous:

| ncycle | labels reporting non-finite |
|---|---|
| **10010** | **`shift.src` -- and nothing else** |
| 10011 | `matter.in`, `X.src`, `adual`, `shift.src` |
| 10012 | + `shift.dap6` |
| 10013 | + `psi.out`, `lapse.out` |

At **ncycle=10010**, `shift.src[u_p_src]` is non-finite in 5,369,744 cells while
**`matter.in[pmhd->u0]`, `X.src`, `adual[u_adual]`, `psi.out[delta_psi]`,
`lapse.out[delta_alpha_psi]` and `shift.dap6` (interior) are ALL clean.** One
cycle later the matter follows, then the rest.

**Consequences:**

1. **The matter is exonerated as the origin.** The conserved state is clean when
   the failing cycle's solves begin, and goes bad only *after* the shift source
   does. Every matter-side and debris-side hypothesis in §4/§9/§11 is now not just
   null but *explained*: the corruption enters upstream of the fluid.
2. **The multigrid solver is exonerated as the origin.** §13.5 inferred "the
   solver produced NaN from a finite source"; that was wrong. The **source handed
   to it was already NaN**. `X.src`/`X.out` (the control: same
   `MGCFCVectorPoisson`, simple source) stay clean at 10010, so the machinery
   itself is fine.
3. **The origin is `CFC::AssembleVectorSource(for_shift=true)` /
   `BuildShiftSource`.** Nothing else.

**The causal chain, now measured rather than inferred:**
shift source NaN (10010) -> shift solve returns NaN `beta` -> the fluid update
consumes NaN `beta` -> matter NaN (10011) -> everything.

This also explains the §13.5 signature that looked so strange -- `beta` 100% NaN
with `psi4`/`alpha`/`K_dd` and every primitive clean. The shift is solved **last**
(step 6), so a source built only for it corrupts only it, and the con2prim that
reports the failure is the *next* cycle's.

### 14.9 What is left, and round 3 (job 8835402)

At ncycle=10010 every *checked* input of the shift source was clean. The unchecked
inputs of `BuildShiftSource` are a short, complete list:

| candidate | why it is a candidate |
|---|---|
| `dap6` **ghosts** | `Dx<NGHOST>` differences `dap6` across its ghost cells; round 2's check was interior-only, so ghost NaN would poison the interior source invisibly |
| **`grad_ap6_0`** | `grad(alpha0*psi0^-6)`. **Used ONLY in the shift source and nowhere else in CFC** -- which alone would explain why the shift source is the sole object to go bad |
| `a0_dd`, `u_psi0`, `u_alpha0_psi0` | the analytic **backgrounds**, which `AccreteExcisedMass` **refills** whenever the puncture mass changes (item 57). The big accretion drain landed at cycle **10001**, nine cycles before the failure |

Round 3 (`tde_ellipdiag3_ckpt10_16n`, job **8835402**) checks exactly these, with
`ngh=0` so ghosts are included. `nlim=10030` -- the answer arrives by cycle 10011,
so there is no reason to re-run the whole cascade.

**A caveat that bounds whatever round 3 finds**: the background-refill route needs
a *changing* puncture mass, which only happens with excision on. The original v2
run has `puncture_mass` constant at 1.0 and no accretion at all, yet **also** ends
with `beta` 100% NaN. So either there are two routes to the same failure, or the
cause lies in a part of the shift source that does not depend on the refill --
`grad_ap6_0` and the `dap6` ghosts are present either way. Do not over-claim from
the excision runs alone; the same instrumentation should be run once on a
**no-excision** restart to confirm the origin is identical there.


---

## 15. ROOT CAUSE (2026-09-18): `grad_ap6_0` forms 0/0 at the trumpet throat

### 15.1 The defect

`src/cfc/cfc_puncture.cpp`, `FillPunctureBackground`, as written:

```cpp
Real ap6  = alpha0*psi0_inv6;                   // -> 0  as alpha0 -> 0
Real dfac = dalpha0/alpha0 - 6.0*dpsi0/psi0;    // -> 0/0 at the throat
grad_ap6_0(m,0,k,j,i) = ap6*dfac*x1v/r;
```

The background lapse is the stationary maximal-slicing trumpet
(`cfc_puncture.hpp`, `TrumpetBackground`):

```
alpha0^2 = 1 - 2/rrs + 1.6875/rrs^4 ,   rrs = r_sch/m_bh
dalpha0  = (1 - 3.375/rrs^3)/(rrs*r)
```

`alpha0^2` has a **double root at rrs = 3/2** -- the trumpet throat, R = 1.5 M.
And `dalpha0`'s numerator `1 - 3.375/rrs^3` vanishes at **the same point**
(3.375/1.5^3 = 1). So at the throat `dalpha0/alpha0` is **0/0 = NaN**.

`alpha0` **cancels analytically**:

```
ap6*dfac = (alpha0*psi0^-6)*(dalpha0/alpha0 - 6*dpsi0/psi0)
         =  psi0^-6*(dalpha0 - 6*alpha0*dpsi0/psi0)
```

so the quotient never had to be formed. Because the root is a *double* root,
`alpha0^2` also loses all significance to cancellation nearby. Measured, sampling
`rrs` uniformly within 1e-7 of 3/2 (2e6 samples):

| outcome | fraction | consequence |
|---|---|---|
| `alpha0^2 == 0` exactly | **2.8%** | `alpha0 = 0`, `dalpha0 ~ 0` -> **0/0 = NaN** |
| `alpha0^2 < 0` | **2.4%** | `sqrt(negative)` -> **alpha0 = NaN** directly |

### 15.2 Why this produced exactly the symptoms observed

**`grad_ap6_0` is consumed by `BuildShiftSource` and by nothing else in CFC.**
That single fact explains the signature that made this so hard to localise:

- **Only `beta^i` goes NaN**, 100% of cells, while `psi4`, `alpha`, `K_dd` and
  every primitive stay clean (§13.5) -- the corrupted input feeds only the shift
  source, and the shift is solved **last** (step 6).
- **Zero elliptic convergence failures** (§2) -- the solve is fine; its *source*
  is NaN.
- **The NaN appears globally and simultaneously** -- an elliptic solve couples the
  whole domain in one step.
- **`K_dd` NaN in v2 but not in the excision run** -- `K_dd = a_dd*psi^-2` is
  written last by `AssembleLapseShiftK`, so it is a late symptom, not the origin.
- **The matter goes NaN one cycle later** (§14.8) -- it consumes the NaN `beta`.

**CORRECTION (2026-09-18, after validation).** The two bullets below were written
before the validation runs and **the first of them is WRONG**. `tde_fixval_noexc`
-- the production deck with this fix applied -- still fails at cycle ~10604,
against v2's 10659 and parityfix's 10599. So this defect does **not** explain the
original no-excision failure; it explains the excision runs' *early* failures
(10016-10351), which is why `tde_fixval_exc` now passes. The second bullet (the
335-cycle spread between excision runs) stands. See §16.

It also explains the two timing puzzles:

- **v2 (no excision) survived to cycle 10659, the excision runs failed at
  10016-10351.** The throat sits at `r_sch = 1.5*m_bh`. With `puncture_mass`
  pinned at 1.0 the singular surface is *fixed*, and only AMR remeshes resample
  cell positions against it (51 remeshes in v2). With excision on,
  `AccreteExcisedMass` changes `m_bh` every drain, **sweeping the throat across
  the grid** and sampling far more positions against it.
- **The 335-cycle spread between identical excision runs** (§14.6) -- the drain
  uses atomic adds, so `dM` differs run to run, so the throat lands differently.
  Not mysterious once the cause is known.

### 15.3 How it was found, in one line each

`solve_interval` (§9.4), debris resolution (§9.6), AMR in four separate readings
(§9.2, §10.1, §10.3), the density-max tracker (§10.2), the debris energy (§11.2)
and the puncture energy pile-up (§13.3-13.4) were all excluded by measurement.
`CheckFinite` instrumentation at `solve_interval=1` (§14.8) then showed the NaN
appearing in `shift.src` **alone**, one cycle before the matter; round 3 (§14.9)
narrowed it to `grad_ap6_0` alone, with `psi0`, `alpha0*psi0`, `a0_dd` and `dap6`
all clean -- which is only consistent with `alpha0 = 0` (finite, so the background
checks pass) feeding a division.

### 15.4 The fix

`cfc_puncture.cpp`: compute the analytically-cancelled form, which never divides
by `alpha0`:

```cpp
Real amp = psi0_inv6*(dalpha0 - 6.0*alpha0*dpsi0/psi0);
grad_ap6_0(m,0,k,j,i) = amp*x1v/r;   // and x2v, x3v
```

`cfc_puncture.hpp`: clamp before the square root. `alpha0^2 >= 0` analytically
everywhere, so this restores the mathematical value rather than masking a real
negative:

```cpp
Real alpha_sq = 1.0 - 2.0/rrs + 1.6875/(rrs3*rrs);
Real alpha = std::sqrt(alpha_sq > 0.0 ? alpha_sq : 0.0);
```

`psi0 = sqrt(r_sch/r) > 0` always, so `dpsi0/psi0` is safe. `r` carries a
`+1e-30` guard already.

### 15.5 Validation -- RUNNING

| case | job | config | nodes |
|---|---|---|---|
| `tde_fixval_noexc` | **8836063** | **production deck**, `excise=false`, `solve_interval=10`, adaptive AMR | 8 (matches ckpt) |
| `tde_fixval_exc` | **8836064** | excision on (`lapse`) | 16 |

Both restart from ckpt 00010 (cycle 10000) to `nlim=10900`, with
`src_nan_check=1` as a safety net (`mg_verbose` off -- it produced 1.3 GB logs).

**PASS = reach cycle 10900 with 0 `NANS_IN_CONS` and 0 `CFC NONFINITE`.** v2 died
at 10659 and the parityfix restart at 10599, so clearing 10659 by ~240 cycles in
the *same* configuration is the result that matters. The no-excision arm is the
one that speaks to production.

**RESULT (2026-09-18):**

| arm | outcome |
|---|---|
| `tde_fixval_exc` (excision) | **PASS** -- cycle 10900, t=459.6, **0 `NANS_IN_CONS`, 0 `CFC NONFINITE`**, `dt` healthy at 4.45e-02 |
| `tde_fixval_noexc` (**production**) | **FAIL** -- cascade at ~10604, 974,644 NaN |

The fix is **real and validated on the excision path**, and `grad_ap6_0` does not
appear in the failing arm's reports at all. But it does **not** fix production.
§9.7's rule stands: do not restart production. §16 continues.

### 15.6 If it passes

- The excision work (§13) is a **separate** real fix for a separate real defect
  (the puncture pile-up: `W ~ 1e7`, `tau(r<2)` growing 22,675x) and should be
  considered on its own merits, not as part of this.
- §10.2's tracker degeneracy (density max jumps up to 6.1 with two clumps within
  0.5%) remains an **open fidelity defect**, independent of this cause.
- The `gamma_max` ceiling not binding the c2p output (§12.1) is also still open.
- A regression test belongs here: evaluate `grad_ap6_0` at `rrs` within ~1e-9 of
  1.5 and assert finiteness. The bug was invisible for the run's first 10,000
  cycles precisely because it needs a cell to land near a measure-zero surface,
  so an ordinary smoke test will not catch it.


---

## 16. The remaining no-excision failure (2026-09-18) -- OPEN

### 16.1 What the validation established

§15's fix is validated on the excision path and is not the production cause:

- `tde_fixval_exc`: cycle 10900, **0 NaN, 0 NONFINITE**. Previously this
  configuration failed at 10016-10351.
- `tde_fixval_noexc` (production deck, `excise=false`, `solve_interval=10`,
  adaptive, 8 nodes): **fails at ~10604**, against v2's 10659 and parityfix's
  10599. Unchanged.

So there are **two distinct causes**, and §15 is only one of them.

### 16.2 What the failing arm's instrumentation shows

All 8 reports land at `ncycle=10610` -- too late to order, because the production
deck runs `solve_interval=10` and the matter had already gone NaN at **10604**.
The same trap as §14.5. What is still usable:

- **`grad_ap6_0` does NOT appear.** The §15 fix holds.
- **`psi.out[delta_psi]`, `lapse.out[delta_alpha_psi]` and
  `shift.dap6[u_alpha_psi6]` all have their first offender at EXACTLY
  `x=(0,0,0)`, MeshBlock 9** -- a cell sitting *on* the puncture. (`matter.in`,
  `X.src`, `adual`, `shift.src` report at the same cycle with their first offender
  at the domain corner, which is just the lowest flat index.)

A cell exactly at the origin is notable in itself: `FillPunctureBackground` guards
with `r = sqrt(x^2+y^2+z^2 + 1e-30)`, i.e. `r = 1e-15` there, and
`psi0 = sqrt(r_sch/r)` is then ~4e7 with `psi0^6 ~ 3e45`.

### 16.3 LEAD: `alpha_floor` is applied after ALL solvers, so it protects nothing upstream

Raised by the user, and the code confirms it. `alpha_floor` is applied at
`cfc_reconstruct.cpp:225`, inside `AssembleLapseShiftK`, which is called by
`CFC::AssembleADM()` from **`AssembleFinalTask`** -- the last task in the graph
(`Task_Run, {CFC_ReconstructBeta}`). Therefore:

- the lapse solve's own output `delta_alpha_psi` is **never floored**;
- the **shift source** consumes the unfloored `alpha*psi` via
  `ap6 = (delta_alpha_psi + u_alpha0_psi0)/psi7`;
- the **next cycle's** psi and lapse solves consume it unfloored;
- only `adm.alpha`, the value handed to con2prim/hydro, is clamped.

`alpha_floor` is a cosmetic clamp on the ADM output. It does not protect the
elliptic solves or the shift source.

**This invalidates a check made earlier in this file.** §14.2 concluded "`psi`
does not approach zero" partly from `min(alpha)` read out of the `bin` dumps --
but `adm_alpha` is the **floored** field and reads `1.0e-6` by construction
whatever the raw `alpha*psi` is doing. The `psi4` half of that check is still
sound (`psi4` is written from `psi_val` directly and would go to 0 with it). **The
alpha half is worthless and must not be relied on.** `lapse.out[delta_alpha_psi]`
is the unfloored quantity, and it is now instrumented.

This is a **lead, not a cause**: it is established that the floor cannot protect
upstream consumers, and that `delta_alpha_psi` goes NaN at the puncture cell. It
is **not** established that an unfloored `alpha*psi` crossing zero is what starts
the cascade.

### 16.4 Running: job 8836351

`tde_fixdiag_noexc_si1` -- the failing production arm, identical except
`solve_interval=1` so the gated checks run every cycle and can order matter vs
`psi.out` vs `lapse.out` vs `shift.src`, exactly as §14.8 did for the excision
path.

**Read it the same way**: whichever label reports *alone* at the first failing
cycle names the origin. If `lapse.out` or `psi.out` reports while `matter.in` is
clean, the cause is in the scalar solves at the puncture cell, and §16.3 is the
first thing to test. If `matter.in` reports first, the origin is on the fluid side
and §11/§12's puncture pile-up (`W ~ 1e7`, which excision removes and which the
no-excision deck still has) becomes the prime suspect -- note that this would also
explain why the excision arm passes and this one does not.


---

## 17. RESOLUTION (2026-09-18): two causes, two fixes, both needed

### 17.1 The decisive measurement

**The no-excision failure is locked to physical TIME, not to a cycle count:**

| run | code | first `NANS_IN_CONS` |
|---|---|---|
| v2 original | pre-fix | cycle 10659, **t=446.01** |
| parityfix | pre-fix | cycle 10599, **t=444.25** |
| `tde_fixval_noexc` | **post-§15-fix** | cycle 10604, **t=444.44** |
| `tde_fixdiag_noexc_si1` | post-fix, `solve_interval=1` | cycle 10684, **t=446.80** |

Four runs spanning two code versions, two `solve_interval` values and different
diagnostics, all failing inside a **2.6 M window, t=444.2-446.8**. A physical
clock, not a numerical accident -- and exactly when §11/§12 measured the puncture
pile-up becoming extreme (`tau(r<2)` x22,675, `W -> 1e7`, count of `W>50` cells
going 4 -> 24 -> 84).

**The excision arm with the §15 fix ran clean to t=459.6** (`tde_fixval_exc`,
cycle 10900, 0 `NANS_IN_CONS`, 0 `CFC NONFINITE`, `dt` = 4.45e-02) -- **13 M past
the entire failure window.**

### 17.2 Two causes, distinguishable by fingerprint

| | **production (no excision)** | **excision path** |
|---|---|---|
| NaN signature | **`psi4` NaN**, `alpha`/`beta`/`K_dd` finite | **`beta` NaN**, `psi4`/`alpha`/`K_dd` clean |
| first bad object | the `psi` (conformal factor) solve | `grad_ap6_0` -> shift source |
| cause | unexcised puncture pile-up feeding the `psi` source | `dalpha0/alpha0` = 0/0 at the trumpet throat |
| when | t ~ 445, reproducibly | cycle 10016-10351, nondeterministically |
| fix | **excision ON** (`lapse`) | **§15 code fix** |
| status | validated empirically (17.1) | validated (arm passes) |

The two signatures are mutually exclusive and were observed cleanly in both
directions, which is what makes this a two-cause problem rather than one cause
with variable presentation.

### 17.3 CORRECTION: §13.4's refutation of §11.4 was confounded

§13.4 concluded: *"excision removed the `tau` runaway and the NaN persisted,
therefore §11.4 is refuted."* **That inference was wrong, and the reason is
instructive.** Excision did remove the pile-up -- but it simultaneously *exposed a
second, unrelated bug* (`grad_ap6_0`), because a changing `m_bh` sweeps the
trumpet throat across the grid. So the NaN "persisted" for a completely different
reason, with a completely different signature, and the refutation attributed that
persistence to the original hypothesis.

**§11.4 is reinstated**: the puncture pile-up feeding the `psi` source is the
production cause. The mechanism's final step (the `psi` source overflowing) is
still not *directly* instrumented -- what is measured is that the no-excision deck
fails reproducibly at t~445 with a `psi`-NaN signature, and that removing the
pile-up carries the run 13 M past it.

**The general lesson, and it is the same trap as §7.1 in a new costume**: when a
fix changes the system enough to expose a *different* bug, "the symptom persisted"
does not refute the original hypothesis. The signature has to be checked, not just
the presence of a failure. Had §13.4 compared *which field* went NaN rather than
*whether* one did, this would have been caught three sections earlier.

### 17.4 Recommendation

**Run production with BOTH fixes. Neither alone is sufficient**, and this is
measured, not argued:

- Without the §15 fix, the excision runs died at cycles 10016-10351.
- Without excision, the production deck still dies at t~445 -- `tde_fixval_noexc`
  and `tde_fixdiag_noexc_si1` both carry the §15 fix and both still fail.

`tde_fixcont_exc` (job **8836484**) continues the validated configuration from its
cycle-10900 checkpoint with `nlim=-1` so `tlim=500` governs, to establish that it
completes the science target rather than merely clearing the old failure point.

**Production remains blocked until that lands.** The gate: reach `tlim=500` with
0 `NANS_IN_CONS` and 0 `CFC NONFINITE`.

### 17.5 Still open, and independent of the above

- **§10.2 tracker degeneracy** -- the density-max refinement target jumps up to
  6.1 code units with two clumps within 0.5% of each other. A real fidelity defect
  in the debris; unrelated to either cause here. The `f = 1e-2` relative-density
  criterion and its measured MeshBlock cost are in §11.6.
- **§12.1 `gamma_max` does not bind the c2p output** -- configured at 50 (default),
  measured `W = 1e7`. Applied only to the root-find's internal velocity estimate.
- **§14.6 excision accretion is nondeterministic** (atomic-add tally), so excision
  runs are not bitwise reproducible and onset cycle is not a valid A/B metric.
- A **regression test** for §15: evaluate `grad_ap6_0` at `rrs` within ~1e-9 of
  3/2 and assert finiteness. The bug hid for 10,000 cycles because it needs a cell
  to land near a measure-zero surface.


---

## 18. Solve-output floors for `psi` and `alpha` (2026-09-18) -- jobs 8836515 / 8836516

User's change. The motivation is §16.3: `alpha_floor` was applied only in
`AssembleLapseShiftK`, i.e. in `AssembleFinalTask`, the **last task of the step**.

### 18.1 What was wrong with a floor applied at the end

The late clamp protects only `adm.alpha`, the copy handed to con2prim/hydro. It
does **not** protect:

- **the shift source**, which consumes the RAW `alpha*psi` and `psi` via
  `ap6 = (delta_alpha_psi + u_alpha0_psi0)/psi^7` (`AssembleVectorSource`, step 6);
- **the next cycle's psi and lapse solves**, which re-consume the raw residuals.

So a zero or negative `alpha` (or `psi`) could enter an elliptic source while every
ADM-side diagnostic still read a healthy floored `1.0e-6`. That is also why §14.2's
"alpha looks fine" reading was worthless: it sampled the floored field.

### 18.2 What was added

`CFC::ApplyPsiFloor()` and `CFC::ApplyAlphaFloor()` (`cfc.cpp`), plus
`<cfc> psi_floor` (default **0.0**, disabled, so no existing fixture changes).

**Placement, chosen against the task graph:**

- `ApplyPsiFloor` runs after `pmgd_psi->RetrieveSolution`, **before
  `AssembleConformalMetric`** -- so `psi4`/`g_dd`, which the shared con2prim at
  step 4 consumes, are built from the floored `psi` -- and before the psi ghost
  exchange (`CFC_RestPsi` -> ... -> `CFC_BCSPsi`, all downstream of
  `CFC_SolveLapse`).
- `ApplyAlphaFloor` runs after `pmgd_alpha->RetrieveSolution`, before the
  `alpha*psi` exchange and before `CFC_BuildSrcBeta` (which depends on
  `CFC_BCSPsi` **and** `CFC_BCSAlphaPsi`).

**Residual arithmetic.** Both arrays are residuals, so the clamp is on the sum and
rearranged onto the residual:

```
psi   = delta_psi + psi0        >= psi_floor    ->  delta_psi       >= psi_floor - psi0
alpha = alpha_psi/psi           >= alpha_floor  ->  delta_alpha_psi >= alpha_floor*psi
                                                                       - alpha0*psi0
```

The alpha clamp is on **alpha**, not on `alpha*psi`, to match the existing
`AssembleLapseShiftK` semantics exactly. It uses the already-floored `psi`, since
`ApplyPsiFloor` ran earlier in the graph.

**Ghost cells.** Both loops sweep the FULL array extent, not just the interior:
that costs nothing and also clears stale sub-floor values left in ghosts from a
previous step. Because they run before the exchange, the exchange then propagates
already-floored data. *Caveat recorded for a future reader*: `ProlongPsi`
interpolates fine-level ghosts from coarse data, and a non-monotone prolongation
stencil could in principle undershoot the floor again; if that is ever observed
the fix is a second clamp after `CFC_BCSPsi`, not a change to these routines.

### 18.3 DELIBERATE: the clamps are not `fmax`, and do not suppress NaN

IEEE `fmax(NaN, x)` returns `x`. A floor written with `fmax` would therefore
**silently absorb a NaN solve output into the floor value** -- masking exactly the
failures this file exists to diagnose, and yielding a quietly wrong spacetime
instead of a loud one. Both routines instead use an explicit `(v < lo) ? lo : v`,
which is false for NaN, so **NaN passes through untouched** and is still caught by
`src_nan_check` and `NANS_IN_CONS`.

These floors are for non-positive **finite** values, which is what they were asked
to prevent. They are **not** a NaN suppressor, and must not be treated as one.

### 18.4 Validation -- RUNNING

| case | job | config |
|---|---|---|
| `tde_floor_noexc` | **8836515** | production deck, `excise=false`, `psi_floor=1e-6` | 
| `tde_floor_exc` | **8836516** | excision on, `psi_floor=1e-6` (regression check) |

Both from ckpt 00010 to `nlim=10900`, `alpha_floor=1e-6`, `src_nan_check=1`.

**The criterion for the `noexc` arm is unambiguous because the failure window is a
physical clock**: four independent no-excision runs died at **t=444.2-446.8** with
a `psi4`-NaN signature (§17.1). Clearing that window is the result.

**Expectation, recorded before the result**: these floors address §16.3's *blind
spot*, and it is NOT established that a sub-floor finite `alpha` or `psi` is what
starts the production cascade -- §17 attributes that to the puncture pile-up
feeding the `psi` source, for which excision is the demonstrated fix. If
`tde_floor_noexc` clears t=447, that attribution needs revisiting. If it fails at
t~445 like its four predecessors, the floors are a correct hardening that does not
address the production cause, and §17.4's recommendation (excision + §15) stands.
Either outcome is informative; neither is assumed.


---

## 19. FINAL STATE (2026-09-18): the run completes, and the production recipe

### 19.1 The result

`tde_fixcont_exc` (job 8836484), continuing the validated configuration:

```
cycle=12155 time=4.999935e+02 dt=6.462867e-03
cycle=12156 time=5.000000e+02 dt=1.292573e-02
Terminating on time limit
```

**0 `NANS_IN_CONS`, 0 `CFC NONFINITE`, 0 convergence failures.** `dt` sits at
~3.26e-02 throughout (the final two short steps are AthenaK trimming to land
exactly on `tlim`). This "Terminating on time limit" is **genuine**, unlike §1's --
the original reached the message via `dt` running away to 48.5 with 974,603 NaNs.

**And the physics is right, not merely finite.** Combined history across the two
segments (t=422.3 -> 500.0, 778 rows, 0 NaN rows):

| t | mass | `tot-E` | **`tau/M`** |
|---|---|---|---|
| 422.3 | 8.0483e-05 | 5.239e-06 | **0.0651** |
| 446.0 | 8.0461e-05 | 3.693e-06 | **0.0459** |
| 470.0 | 8.0411e-05 | 2.753e-06 | **0.0342** |
| 500.0 | 8.0360e-05 | 2.061e-06 | **0.0256** |

`tau/M` **declines monotonically** -- the debris expanding and receding, which is
the turnover §11.2 showed the physics requires and which the original run's
whole-domain diagnostic was masking. At t=446, where all four no-excision runs
die, this run reads 0.0459; the original read **2592**. Mass drift is -0.153% over
78 M, monotonic, consistent with excision plus outflow through the `diode`
boundaries.

### 19.2 PRODUCTION RECIPE

```
<coord>  excise = true            # addresses cause A (§11/§12/§17)
         excision_scheme = lapse  # NOT fixed -- §13.1
         dexcise = 1.0e-24
         texcise = 1.0e-22
```
plus the **§15 `cfc_puncture.cpp`/`cfc_puncture.hpp` code fix** (cause B).

**Both are required, and this is measured:**
- without §15, the excision runs died at cycles 10016-10351;
- without excision, the production deck dies at t~445 *with* §15 applied
  (`tde_fixval_noexc`, `tde_fixdiag_noexc_si1`, `tde_floor_noexc` -- three runs).

Optional hardening, validated as non-regressing: `<cfc> psi_floor = 1.0e-6` with
`alpha_floor = 1.0e-6` (§18). Recommended on general grounds, but it is **not**
what fixes production -- see 19.3.

`<cfc> src_nan_check = 1` is cheap (one branch when clean) and turns any future
silent cascade into a named first offender. Recommended for production.

### 19.3 The §18 floors: correct hardening, NOT the production fix

Prediction recorded in §18.4 before the runs; it held.

| arm | result |
|---|---|
| `tde_floor_exc` (8836516) | **PASS** -- cycle 10900, 0 NaN. No regression from the floors. |
| `tde_floor_noexc` (8836515) | **FAIL** at cycle 10654, **t=446.01** -- inside the same t=444.2-446.8 window as its four predecessors. |

And the **signature is unchanged**: `psi4` NaN with `alp = 0.8356`, `beta` and
`K_dd` finite -- identical to the original v2 failure. So the production cascade
is the `psi` **solve** returning NaN, not a sub-floor finite value propagating.
The floors pass NaN through by design (§18.3), so they neither fix nor mask it.

They remain worth keeping: §16.3's blind spot was real (the late `alpha_floor`
protected only `adm.alpha` while the shift source and the next cycle's solves
consumed the raw residuals), and closing it costs nothing measurable.

### 19.4 Still open

None of these block production.

- **§10.2 tracker degeneracy.** `TDERefineTracker` targets the global density
  maximum, which post-disruption jumps up to 6.1 code units between clumps within
  0.5% of each other, teleporting a radius-2 refinement ball with zero overlap. A
  real fidelity defect in the debris. §11.6 has the measured MeshBlock cost of an
  `f = 1e-2` relative-density criterion (affordable; `f = 1e-4` is not without a
  level cap).
- **§12.1 `gamma_max` does not bind the c2p output.** Configured at 50 (default),
  measured `W = 1e7`. It is applied only to the root-find's internal velocity
  estimate; `ResetFloor::PrimitiveFloor` has no velocity clamp. Excision removes
  the region where this mattered here, but the gap is still there.
- **§14.6 excision accretion is nondeterministic** (atomic-add `excised_tally`),
  so excision runs are not bitwise reproducible and **onset cycle is not a valid
  A/B metric** on this path. Signature, not timing, is what to compare.
- **Regression test for §15**: evaluate `grad_ap6_0` at `rrs` within ~1e-9 of 3/2
  and assert finiteness. The bug hid for 10,000 cycles because it needs a cell to
  land near a measure-zero surface, so no ordinary smoke test catches it.
- The **mechanism of cause A** is attributed, not directly instrumented: it is
  measured that the no-excision deck fails reproducibly at t~445 with a
  `psi`-NaN signature, and that removing the pile-up carries the run to `tlim`.
  The `psi`-source overflow itself was never caught in the act.


---

## 20. Production restart (2026-09-18): `tde_prod_fixed_v3`, job 8837395

Supersedes `tde_elliptic_tracker_nexteval_v2`.

### 20.1 Why ckpt 00007 and not 00010

The validation runs started from ckpt 00010 (t=422) because it was closest to the
failure. **That is the wrong place to restart production from**, and the reason is
measured: the fraction of the `tau` integral sitting inside r<2 -- i.e. how much of
the `psi` elliptic source is the puncture-pile-up artifact rather than matter --
reads

| checkpoint | t | fraction of `tau` inside r<2 |
|---|---|---|
| (t<=300) | <=300 | **0.0000** |
| -- | 320 | 0.0002 |
| **00007** | **308.5** | **~0.0000 (clean)** |
| 00008 | 347.3 | 0.196 |
| 00010 | 422.2 | **0.9999** |

Restarting from 00010 would inherit 422 M of evolution whose `psi` source was
99.99% artifact -- and the `psi` solve is global, so that contaminates the whole
domain, not just r<2. Even 00008 is 20% contaminated.

**Ckpt 00007 is clean AND pre-periapsis** (periapsis is t~349), so the entire
disruption and post-periapsis debris-spreading phase -- the part of the science
this run exists for, and the part the original fixture explicitly left untested --
is recomputed with a correct spacetime. Turning excision on at t=308.5 is smooth:
essentially no matter is inside r<1 there.

Going further back to t=0 buys nothing: the contamination is 0.0000 for all
t<=300, so the pre-308 trajectory is already uncontaminated.

### 20.2 Configuration

```
<coord>  excise = true            # cause A (§11/§12/§17)
         excision_scheme = lapse  # NOT fixed -- §13.1
         dexcise = 1.0e-24
         texcise = 1.0e-22
<cfc>    solve_interval = 10      # production value, exonerated §9.4
         alpha_floor = 1.0e-6
         psi_floor   = 1.0e-6     # §18 hardening, validated non-regressing
         src_nan_check = 1        # safety net, one branch when clean
<time>   nlim = -1                # tlim = 500 governs
```
plus the **§15 `grad_ap6_0` code fix** (cause B), in the binary
(md5 `037951e9bd9fe38429d5eee2d03f7731`, built on aurora-uan-0007).

16 nodes / 192 ranks, matching the **validated** configuration
(`tde_fixval_exc` -> `tde_fixcont_exc`, which reached `tlim=500` with 0 NaN and
`tau/M` declining to 0.0256). This does not match the checkpoint's original 8/96,
so per-cycle timing is not comparable to the pre-fix runs; correctness is
unaffected. `MPIR_CVAR_CH4_IPC_GPU_MAX_CACHE_ENTRIES=1024`.

Expected ~5,000-6,000 cycles from cycle 7000 at ~0.58 s/cycle = ~1 hour; walltime
05:40:00, and `batch.t` chains a segment if that is not enough.

**v2 is untouched**: reached through a symlink in `parent/rst`; `grep -c` finds no
write path to it in `batch.sub`, and `find -newermt` over `v2/output-0000` returns
nothing.

### 20.3 What to check when it lands

1. `grep -c NANS_IN_CONS` = **0** and `grep -c "CFC NONFINITE"` = **0**.
2. It must clear **t=444.2-446.8**, where four no-excision runs died. That window
   is a physical clock, so clearing it is unambiguous.
3. `Terminating on time limit` with a **healthy `dt`** (~3e-02), not the runaway
   `dt`~48 that produced the same message in v2 (§1).
4. **`tau/M` declining** through the post-periapsis phase. In the validated run it
   fell 0.0651 -> 0.0256 over t=422->500. A rising `tau/M` means something is
   wrong even if no NaN appears.
5. `puncture_mass` grows from 1.0 via `AccreteExcisedMass`, and the independent
   `<cfc> adm_mass_r*` surface integral tracks it (§13.3). Note this run does not
   set `adm_mass_r*`; add it if the accretion bookkeeping needs auditing.
6. Because this restarts pre-periapsis with a corrected spacetime, **the
   trajectory will differ from v2's** downstream of t=308.5. That is expected and
   is the point; do not treat divergence from v2 as an error.


---

## 21. PRODUCTION COMPLETE (2026-09-19) -- `tde_prod_fixed_v3`, job 8837395

### 21.1 Result

```
cycle=12109 time=5.000000e+02 dt=3.202152e-02
Terminating on time limit
```

**0 `NANS_IN_CONS`, 0 `CFC NONFINITE`, 0 convergence failures, 0 NaN rows in the
history.** `dt` healthy at 3.2e-02 -- this "Terminating on time limit" is genuine,
unlike §1's, which was reached by `dt` running away to 48.5 with 974,603 NaNs.

### 21.2 The physics is right, which matters more than the absence of NaN

History over t=308.6 -> 500.0 (1915 rows), restarted from the clean ckpt 00007:

| t | mass | `tot-E` | **`tau/M`** | **`\|S\|/D`** |
|---|---|---|---|---|
| 308.6 | 8.2094e-05 | 6.921e-06 | 0.0843 | 0.4657 |
| 349.0 (periapsis) | 8.0596e-05 | 1.0925e-05 | 0.1356 | **0.6151** |
| **358.7** | -- | -- | **0.1390 (peak)** | -- |
| 400.0 | 8.0369e-05 | 7.670e-06 | 0.0954 | 0.3829 |
| **445.0** | 8.0261e-05 | 4.241e-06 | **0.0528** | 0.1369 |
| 500.0 | 7.9082e-05 | 2.633e-06 | **0.0333** | **0.0559** |

- **`tau/M` peaks at 0.1390 just after periapsis, then declines monotonically.**
  That is periapsis shock heating followed by the debris expanding and cooling.
- **`|S|/D` peaks at 0.615 at periapsis, then falls to 0.056.** This is the
  deceleration turnover that §3 asserted "never happens" and used as its central
  evidence that the growth was numerical. With the causes fixed it happens exactly
  as the physics requires -- confirming §11.2's finding that the original `tot-E`
  signal was a puncture artifact, not debris physics.
- **The critical window t=444.2-446.8 is cleared cleanly**: 30 history rows,
  `tau/M` 0.0518-0.0534, no NaN. Four no-excision runs died in that window; the
  original read `tau/M` = 2592 there.

### 21.3 Mass accounting

`puncture_mass` grew **1.0 -> 1.00001**, matching `M_accreted = 5.3345e-06` from
`AccreteExcisedMass`. Grid rest mass fell 3.012e-06 (-3.67% over 191 M).

The ratio `M_accreted / grid_mass_lost = 1.77` is expected, not an inconsistency:
`dM_adm` is `(D+tau)/psi`, i.e. total ADM **energy**, while the grid figure is
rest mass, and accreted material near the hole carries energy well above its rest
mass. **v2 could not do this at all** -- with `excise=false` nothing ever left the
grid (item 57: "no mass ever left the grid"), so its mass drift was pure numerics.

### 21.4 The answer, in full

Two independent causes, each with its own NaN fingerprint:

| | cause A | cause B |
|---|---|---|
| what | unexcised puncture pile-up feeding the `psi` elliptic source | `grad_ap6_0` forms `dalpha0/alpha0` = 0/0 at the maximal-slicing trumpet throat (`rrs`=3/2), where both vanish; `alpha0` cancels analytically |
| fingerprint | `psi4` NaN, `alpha`/`beta`/`K_dd` finite | `beta` 100% NaN, `psi4`/`alpha`/`K_dd` clean |
| killed | v2, parityfix, and every no-excision run, at t=444.2-446.8 | the excision runs, cycles 10016-10351 |
| fix | `excise=true`, `excision_scheme=lapse` | §15 code fix in `cfc_puncture.cpp`/`.hpp` |

Neither fix alone is sufficient, measured in both directions. Production config is
§20.2; the run above is its validation.

### 21.5 What this investigation got wrong, for the next reader

Kept deliberately, because the wrong turns cost far more than the right ones.

1. **§3 read a whole-domain integral as a statement about the debris.** `tot-E`
   was 99.99% puncture artifact by t=422. This aimed §4a, §4b, §9.4 and §9.6 at
   the debris, which was never the problem -- four null results, all explained
   only in §11.
2. **§13.4 refuted §11.4 on a confounded comparison.** Excision removed the
   pile-up but simultaneously exposed cause B, so "the NaN persisted" was true and
   the inference from it was false. **Comparing *which field* went NaN rather than
   *whether* one did would have caught it three sections earlier.**
3. **§9.4's validity check was mis-specified** (3.65x is `N=0` vs `N=10`, not
   `N=1` vs `N=10`), nearly voiding a good result.
4. **§9.6's criterion compared arms at fixed cycle count** when `dt` differed per
   arm, which would have manufactured a win for the finest arm.
5. **§14.5's instrumentation fired on solve cycles only**, so at
   `solve_interval=10` it measured the aftermath, not the origin.
6. **§14.2 read `alpha` from a field that is floored by construction**, and was
   blind by design (§16.3).
7. The `.hst` `KE` columns are a broken instrument (§9.1) and cost the first hour.

The common thread: **every one of these was a measurement whose limitations were
not checked before its result was believed.** §7's three traps were all present,
and all three were walked into at least once.

### 21.6 Still open (none blocking)

- **§10.2 tracker degeneracy** -- density max teleports up to 6.1 between clumps
  within 0.5%. Fidelity defect in the debris. `f=1e-2` criterion costed in §11.6.
- **§12.1 `gamma_max` does not bind the c2p output** (set to 50, measured `W`=1e7).
- **§14.6 excision accretion is nondeterministic** (atomic-add tally) -- compare
  signature, never onset cycle.
- **Regression test for §15**: evaluate `grad_ap6_0` at `rrs` within ~1e-9 of 3/2
  and assert finiteness.
- **Cause A's mechanism is attributed, not instrumented**: the `psi`-source
  overflow was never caught in the act. The fix is validated empirically.
