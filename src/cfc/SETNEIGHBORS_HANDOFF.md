# `MeshBlock::SetNeighbors` diagonal-registration bug — handoff note

> **READ SECTION 8 FIRST (added 2026-09-16).** Sections 1 and 2c argue that the
> `nghbr` table has a genuine slot-*capacity* problem, and section 4 sketches an
> overflow redesign on that basis. **That premise is wrong**, and section 8 shows
> why, with measurements on the exact production topology that is currently
> failing: the pre-`fee50981` octant-parity rule is exactly symmetric with no
> unfilled ghost region, and the two commits on `proj/tde` are what introduce the
> 1172 asymmetric registrations that abort the restart (now fixed, 8.7). The rest of this file is
> kept unedited as the record of how the investigation got here -- read it for
> history, not for direction.


**Purpose of this file**: a self-contained account of everything confirmed
about this bug, for starting a *fresh* session dedicated to it (recommended
over continuing in the session that produced this note — see the "Why a new
session" note at the end). `src/cfc/DEVELOPMENT.md` items 38/39(a-g)/42/63/64
cover the same ground in far more narrative/chronological detail, with more
job IDs and false starts; this file is the distilled, current-state version.
Read this first; go to DEVELOPMENT.md only for a specific historical detail.

Everything here concerns `src/mesh/meshblock.cpp`'s `MeshBlock::SetNeighbors`
and the shared `nghbr` table it populates (`src/mesh/mesh.hpp`'s
`NeighborBlock`, `src/mesh/meshblock.hpp`'s `DualArray2D<NeighborBlock>
nghbr`) — this is core AthenaK code, used identically by every physics module
(hydro, MHD, Z4c, dyn_grmhd, CFC). **It is not CFC-specific.** CFC is simply
the first module in this codebase's history to expose it in production,
plausibly because (a) this project's tracked, continuously-moving AMR region
following a disrupting star produces far more irregular diagonal-neighbor
topology than the static/symmetric nested refinement typical of BH/NS-only
z4c/dyn_grmhd runs, and (b) CFC's elliptic solve may read corner/edge ghost
data that hydro/MHD/Z4c's own stencils never touch, making a silently-missing
registration visible as corruption where it would otherwise be inert. Neither
half of that is proven — it's the best explanation given what's below.

## 1. The data structure and why it has a genuine capacity problem

`nghbr(m, n)` — one `NeighborBlock{gid,lev,rank,dest}` per `(MeshBlock, slot)`
— has **exactly one candidate's worth of storage per slot**
(`src/mesh/mesh.hpp:47-52`). `src/mesh/nghbr_index.hpp`'s `NeighborIndex`
function maps a direction to a slot number:

```
x1faces:    [0-3],  [4-7]        (4 subblocks per face x 2 faces)
x2faces:    [8-11], [12-15]
x1x2edges:  [16-23]              (2 subblocks per edge-direction x 4 dirs)
x3faces:    [24-27], [28-31]
x3x1edges:  [32-39]
x2x3edges:  [40-47]
corners:    [48-55]              (1 slot per corner x 8 corners — NO subdivision)
```

Faces are a true, collision-free bijection: 4 disjoint subblock slots per
face direction, both branches of `MeshBlock::SetNeighbors`' face sites
register unconditionally, and there is no known problem there. Each edge
direction has exactly 2 subslots — but both are already committed to a
*different*, pre-existing geometric meaning (which of the 2 children along
the free perpendicular axis), confirmed by direct reading of the formula
and every site's own subblock-index computation — **not spare capacity**.
Corners have **zero** subdivision at all (the corner branch of
`NeighborIndex` takes no free-axis argument).

`MeshBlockTree::GetLeaf(ox1,ox2,ox3)` (`src/mesh/meshblock_tree.hpp:36-37`)
is a single, **non-recursive** array index (`pleaf_[ox1+2*ox2+4*ox3]`) — one
immediate child, not an enumeration. `MeshBlockTree::FindNeighbor`
(`src/mesh/meshblock_tree.cpp:360-460`) computes a target's shifted logical
location **at the caller's own level** and walks the tree down to it,
returning the first leaf it hits (same-level or coarser) or, if the shifted
address is still internal at the caller's level, the internal node itself
(the caller's own code then calls `GetLeaf` once to pick a specific child
for "finer" cases). This is the concrete basis for "corners have no way to
discover a second sibling via a single query" and "edges' finer-branch loop
already covers both free-axis children, but that says nothing about a
sibling reached via an entirely different tree path."

Given all this, under **legitimate, standard-enforced** AMR (2:1 balance
across all 26 directions including edges/corners is genuinely enforced —
confirmed via `MeshBlockTree::Refine`/`Derefine`, `meshblock_tree.cpp:152-306`,
which loop `ox,oy,oz` over all of `{-1,0,1}^3` when forcing same-level
neighbors to exist), it is nonetheless possible for **two genuinely
distinct, simultaneously-needed diagonal relationships to want the exact
same slot** on a target block, reached via *different* tree paths (e.g.
through two different face-neighbors of the target) that a single direct
query cannot both see.

## 2. Two confirmed, distinct failure modes

### 2a. Item 38's original under-registration (fixed, see §3)
The historical guard (pre this investigation) was a pure octant-parity check
(`myox1==n && myox2==m [&& myox3==l]`) that silently dropped genuinely-needed
coarser diagonal registrations under irregular AMR. This is what caused the
actual production crash that started this whole investigation: the TDE run
`tde_elliptic_tracker_nexteval_v2` hit a `NANS_IN_CONS` cascade starting at
cycle 10660/t≈446 (974,603 occurrences by the point it was caught), traced
back to silently-stale ghost data at under-registered diagonal slots.

### 2b. The orphan/asymmetric-registration mechanism (item 39c)
A **different** bug, found while trying to fix 2a with a "defer to the other
side" strategy (see §3): when block A registers block B as a coarser
diagonal neighbor but *defers* (chooses not to write its own row, trusting
B's own unconditional branch to "cover" the relationship from B's side), A's
own row is left with **nothing** at that slot — but A's own row is what A's
own send/recv bookkeeping (`MeshBoundaryValues::BuildRankPackedVarMetadata`,
`src/bvals/bvals.cpp`) is built from, with zero cross-rank coordination.
B registering A does nothing for A's own bookkeeping. The result: an
orphaned, one-sided registration — B's rank expects to exchange with A, A's
rank never built a matching entry — that aborts real multi-rank runs with
`MPI internal_Wait: Message truncated`, reliably, before cycle 0, on the CFC
+ BH-puncture + TOV wide-domain smoke test (jobs 8827734/8828453/8828484/
8828511/8828576/8828663, zero exceptions across ~9 attempts).

**Key realization, easy to miss**: "B will register A" (true, if B's own
query finds A) does **not** give A anything — A always needs its own row
entry regardless of what B independently does. The old "defer" logic
answered a different, real question (does registering A's relationship
also risk a target-side slot collision?) by giving up something A always
needed (A's own bookkeeping) — conflating two separate concerns.

### 2c. The genuine N-to-1 tie-break collision (item 39f/63, still open, not addressed by the current patch)
Separately, two *different* source blocks can each independently discover a
diagonal relationship with the *same* target slot, reached via *different*
face-neighbor lineages of the target — the target's own single query can
only resolve one of them. Two live, fully-traced examples exist:
- item 39f: `gid=1`/`gid=3` both compute a genuine, non-redundant coarser
  x3x1-edge relationship with target `gid=8`, colliding on one slot.
- item 63 (DEVELOPMENT.md's numbering, not this file's): `gid=31`/`gid=59`
  colliding on `nt_gid=70`'s corner slot 50.

This is a genuine capacity problem (see §4) — **not fixed by anything landed
so far**. **Update (2026-09-16, production restart attempt)**: this almost
certainly *does* occur in the real production topology, not just deliberately
-constructed test cases. Restarting the actual TDE production run
(`tde_elliptic_tracker_nexteval_v2`, from its last clean checkpoint, cycle
10000) at its real scale (96 ranks, 1268 MeshBlocks across 5 irregular AMR
levels) hit an `internal_Waitall` abort on 78 of 96 ranks, before a single
post-restart cycle completed (job 8830813) — never reproduced at the small
scale (≤512 blocks, ≤8 ranks) this session's earlier validation used.

Worse: the corner-site patch as originally landed in commit `fee50981` had a
**real bug**, found only by this failure — it registered a corner candidate
unconditionally whenever the reciprocal query was FINER, *regardless* of
whether the target's own single-child resolution actually matched this
candidate (the code comment claimed this distinction "does not matter", which
is wrong). That is *worse* than not fixing §2c: in the non-matching case, the
target's own row only ever expects to hear from the sibling it actually
resolves to, so having the non-matching candidate ALSO register and send
produces an orphaned message with no matching recv — precisely an
`internal_Waitall`-style abort, actively caused by the patch rather than
merely left unaddressed. **This has been corrected** (still uncommitted as of
this note): the corner site now restores the original `recip_leaf`-vs-`b`
gid comparison, registering unconditionally only in the matching sub-case
(fixing §2b's orphan mechanism, as intended) and leaving the non-matching
sub-case unregistered (preserving, not fixing, §2c — the same behavior the
already-validated `recip==nullptr`+corner-`GetLeaf` state had before this
session's regression, so no new bug, but the underlying capacity problem is
still open). Edges were checked and do NOT need the same correction: an
edge's finer-branch loop is unconditional over *both* free-axis children (no
single-child guess involved), so `recip==FINER` for edges always implies the
target's own branch will find this exact candidate — there is no
non-matching case to worry about there.

**Re-validated (job 8831986, same restart checkpoint/scale as 8830813): the
corner-site correction alone does NOT clear this abort.** Identical failure
signature, same timing: `internal_Waitall`/`Abort(17)` on dozens of ranks,
immediately after "Multigrid root grid levels... Number of MeshBlocks in the
pack... MeshBlock size..." prints, before a single `cycle=` line. The
corner-site correction was still necessary (it fixed a real, separate
regression it had introduced — see above) but it was never going to fix
*this* — the evidence now points squarely at **`src/multigrid/multigrid_bvals.cpp`**,
not `bvals.cpp`:
- It is a separate, independent reimplementation of the same slot/offset
  boundary-exchange pattern, for the elliptic solver specifically (grep
  confirms it reads `pmy_pack->pmb->nghbr` directly, `for n in 0..nnghbr`,
  `if (nghbr...gid<0) continue` — a naive consumer with no gating/deferring
  logic of its own, i.e. no analogue of item 2b's bug, but also no protection
  against §2c).
- It runs its own boundary exchange during multigrid root-grid setup, which
  happens *before* the first evolution cycle — meaning if the still-open
  §2c tie-break collision (a target's own single query missing a second,
  genuinely-needed diagonal neighbor, so one side never gets a matching
  registration) occurs anywhere in this run's real topology, multigrid's own
  exchange is the FIRST subsystem to attempt cross-rank communication using
  the resulting asymmetric `nghbr`, and hits an orphaned message before
  `bvals.cpp`'s own (already-fixed) path ever gets a chance to run.
- The smaller-scale smoke test (job 8830513, 512 MeshBlocks) also uses CFC's
  multigrid solver and did NOT hit this — but §2c was always understood to
  be topology-dependent (found via a *deliberately constructed* collision,
  not guaranteed to occur at every scale). **This is now the first direct
  evidence that §2c occurs in the real production topology** (1268
  MeshBlocks, 5 irregular AMR levels), not just synthetic test cases.

**Net conclusion**: item 2b (the orphan mechanism, this session's actual
target) is fixed and validated. Item 2c (the tie-break collision) is
confirmed to occur in real production and is now the thing actually blocking
this production run's restart — via `multigrid_bvals.cpp` specifically, at
minimum, though `bvals.cpp` remains equally exposed in principle if the same
collision were to occur on a boundary exchange it handles instead. Fixing
this for real needs the capacity-extension redesign (§4) applied to BOTH
`bvals.cpp`'s path and `multigrid_bvals.cpp`'s independent one — i.e. the
harder problem this note was written to hand off, now with concrete
production evidence that it is not optional.

## 3. What was tried this session, in order, and outcomes

1. **`recip==nullptr` rule** at all 4 diagonal sites (replacing the octant-
   parity guard): register unless the target's own reciprocal query
   (`FindNeighbor` from the target's location, reverse direction) is
   non-null. Validated: 72 new, correct additive registrations on real CFC
   physics (job 8827750), zero regressions, and clean on the hydro
   reproducer (job 8827681). This is a real, confirmed fix for 2a.
2. **Corner-specific asymmetry fix**: corners have no free axis, so
   `recip==FINER` doesn't prove the target's own single `GetLeaf` resolves
   to *this* candidate specifically — added a `recip_leaf->gid_ ==
   mb_gid.h_view(b)` comparison, registering only when it does NOT match
   (meaning the target's own branch won't cover it). Validated: 184 more
   additive registrations, zero regressions.
3. **Item 39c investigation**: the above (1+2), while individually
   validated, reliably hits the `internal_Wait: Message truncated` abort at
   8-rank CFC+TOV scale. A first hypothesis (a cross-rank `nvars` cache
   desync in `BuildRankPackedVarMetadata`) was chased and **refuted** — it
   was a diagnostic artifact from not disambiguating which of ~9 separate
   `MeshBoundaryValues` instances (Hydro/MHD/Z4c/CFC each have their own) a
   debug line came from; adding an `instance_seq_` disambiguator eliminated
   the apparent mismatch. A decisive control test (job 8828760, **unmodified
   HEAD code**) confirmed 39c is a genuine side effect of the fix above (not
   present in unmodified code — but unmodified code isn't "fine" either: it
   produces 7964 real `NANS_IN_CONS` and an eventual SIGBUS on the same
   input). The actual mechanism (§2b, the orphan/defer problem) was then
   found by direct, fully-traced tracking of one concrete example (block 57
   registering block 132 at slot 55/dest=48; block 132 correctly defers per
   step 2's own logic, leaving its own slot 48 completely empty).
4. **Slot-capacity redesign attempt (this session, reverted)**: added a
   same-shape `nghbr_ovfl` companion array (K=2 candidates per slot), removed
   *all* recip-based gating in favor of "always register unconditionally on
   both sides, plus actively search for a second candidate via a composed
   face-then-residual-direction query" (`FindAltDiagonalLeaves`). This
   **compiled and linked cleanly**, but job 8829761 showed it **regresses**:
   even the small, simple 2-rank hydro reproducer now dies before cycle 0 on
   a new self-check —
   ```
   Rank-packed recv header from peer 1 names src_gid=0 for this rank's own
   (lid=0,dn=0), but neither this slot's primary (gid=16) nor its overflow
   (gid=-1) match -- the two ranks' SetNeighbors registrations disagree
   ```
   — and the new overflow slot (`nghbr_ovfl`) **never fired** in either test
   (`topology_ovfl`: 0 lines in both runs). Diagnosis: removing the
   `recip==LEAF` case (which correctly says "this candidate is not really
   your neighbor, the target has a real same-level neighbor there instead")
   along with the defer logic caused genuinely **invalid** relationships to
   be registered — a worse bug than either 2a or 2b, not a fix. **This whole
   direction was reverted** (`git checkout HEAD -- src/bvals/bvals.cpp
   src/bvals/bvals.hpp src/bvals/bvals_cc.cpp src/bvals/bvals_fc.cpp
   src/mesh/meshblock.hpp src/mesh/meshblock.cpp`, then step 5 reapplied by
   hand) — there is **no `nghbr_ovfl`, no capacity extension, no alt-
   discovery** in the tree as of this note.
5. **Minimal patch actually landed** (see §5) — narrower than either 1+2 or
   4: keeps the `recip==LEAF ⇒ invalid, don't register` check (this is what
   step 4 wrongly discarded), and fixes *only* the confirmed orphan mechanism
   (§2b) by having a block register its own row even in the case it used to
   defer (recip is FINER — whether or not it resolves to exactly this
   candidate). Does **not** attempt the tie-break collision (§2c) at all.

## 4. If picking this back up: where the real fix likely lives

The capacity argument in §1 is real, and the regression in step 4 was a
**bug in that specific implementation** (conflating "needs capacity" with
"drop all validity gating"), not evidence the capacity idea itself is wrong.
A corrected design would need to, at minimum:
- **Keep** the `recip==LEAF ⇒ invalid` rejection intact — it is load-bearing,
  confirmed by the regression.
- Give edge/corner slots real headroom (a same-shape overflow companion
  array is one reasonable shape; `src/geodesic-grid/geodesic_grid.hpp`'s
  `num_neighbors`+`ind_neighbors` count-array pattern is a working, already-
  used-elsewhere precedent for "variable count, small fixed cap" storage).
- Solve the *discovery* problem correctly this time: a target's own single
  query genuinely cannot see a sibling reached via a different face-neighbor
  lineage — this needs a validated (not just compiled) composed-query or
  reconciliation mechanism, tested directly against a concrete collision
  case (see §6, no such case currently exists in the test suite).
- Extend `BuildRankPackedVarMetadata` + `bvals_cc.cpp`/`bvals_fc.cpp`'s
  pack/unpack kernels to transmit a second candidate — and **resolve an
  unresolved correctness question found but never tested this session**:
  the unpack step writes incoming data into the destination ghost-cell array
  using an index range keyed only by slot number, identical for primary and
  overflow — meaning, as of anything built this session, two candidates'
  data would race into the *same* destination cells unless their true
  geometric footprints are actually disjoint (unproven either way).
- `prolongation.cpp`'s `FillCoarseInBndryCC`/`ProlongateCC` (and
  `prolong_prims.cpp`) also read the full slot range including corners/edges
  (confirmed via direct reading, their own comment cites corner correctness
  explicitly) — a capacity fix must extend there too, not just the MPI path.
- `src/bvals/bvals_part.cpp` relies on slot **contiguity** (probes forward
  `indx++` from a starting `NeighborIndex(...)` until it finds a populated
  slot) — any redesign changing per-category contiguity, not just widening
  capacity alongside it, would silently break particle relocation.
- `src/multigrid/multigrid_bvals.cpp` is a separate, independent
  reimplementation of the same slot/offset pattern for the elliptic solver's
  own boundary exchange — not touched by anything in this investigation,
  would need the identical treatment in its own right.

## 5. Current state of the tree (as of this note)

- `src/bvals/bvals.{cpp,hpp}`, `bvals_cc.cpp`, `bvals_fc.cpp`,
  `src/mesh/meshblock.hpp`: byte-identical to committed HEAD — no
  `nghbr_ovfl`, no capacity extension, nothing from step 4 above remains.
- `src/mesh/meshblock.cpp`: HEAD plus a **minimal** patch at all 3 edge sites
  and the corner site's coarser-branch: `do_register = (recip == nullptr) ||
  (recip->pleaf_ != nullptr)` (previously `do_register = (recip ==
  nullptr)` for edges, or the `recip_leaf`-matching comparison for corners).
  In words: register this block's own row unless the reciprocal query finds
  a genuine same-level LEAF neighbor instead (the only case that means this
  candidacy is invalid); register in *both* FINER sub-cases (whether or not
  the target's own resolution happens to match this specific candidate),
  since this block's own row needs the entry regardless of what the target
  independently does. This directly targets §2b without touching the
  §2c tie-break case at all, and does not reintroduce the step-4 regression
  (the `recip==LEAF` rejection is untouched).
- **Validation status of this patch: item 39c confirmed fixed (job 8830513).**
  Run 1 (2-rank hydro reproducer): clean pass, exit 0, reaches `cycle=10`
  (nlim), zero `NANS_IN_CONS`, no FATAL. Run 2 (8-rank CFC+puncture+TOV
  wide-domain smoke test): **zero occurrences of `internal_Wait`/
  `internal_Waitall`/`Message truncated`/`Abort(` anywhere in the log** --
  this is the first SetNeighbors fix attempt across this entire
  investigation to get the wide-domain 8-rank test past cycle 0 at all, let
  alone all the way to `nlim` (reaches `cycle=20` cleanly, "Terminating on
  cycle limit" printed, full cycle stats). Two SEPARATE problems remain in
  run 2, both matching item 64's own documented control-test signature for
  *unmodified* `meshblock.cpp` almost exactly (7987 NaNs here vs. 7964
  there) -- i.e. they read as pre-existing and independent of this patch,
  not caused by it, though no same-job A/B was run to prove that:
  (a) the CFC elliptic solver (`MultigridDriver::SolveIterative`) fails to
  converge at cycle-0 metric initialization ("Failed to converge after 40
  iterations, defect=253.384, threshold=1e-10"), producing widespread
  `NANS_IN_CONS` (7987) downstream of that; (b) a late SIGBUS on one rank
  *after* AthenaK's own clean `nlim`-termination print (exit 135). Neither
  looks related to `SetNeighbors`/boundary-exchange bookkeeping -- both
  present as a separate, likely pre-existing initial-data/elliptic-solver
  issue with this specific input, worth its own investigation but out of
  scope for this note.
- `src/cfc/DEVELOPMENT.md` still has items 63/64 (this session's own numbers,
  **not to be confused** with the pre-existing, unrelated `## 63.
  Star-tracking AMR refinement region` header earlier in the same file —
  the document has a genuine, pre-existing numbering collision between a
  `##`-header sequence and a separately-numbered list-item sequence that
  predates this session; do not try to fix it while doing something else,
  it's an orthogonal, lower-stakes cleanup).

## 6. Existing reproducers and tests — what they do and don't cover

- `inputs/tests/lwave_hydro_diag_collision.athinput` /
  `tst/inputs/lwave_hydro_diag_collision.athinput` — plain Newtonian hydro,
  no CFC/GR, 4x4x4 root MeshBlocks, one root block statically refined one
  level, periodic BCs. **Confirmed too symmetric to exercise a genuine
  2-candidate collision** (every coarser diagonal candidate here is
  redundant with an existing registration) — it's a clean-registration
  regression check, not a collision reproducer.
- `inputs/dyn_grmhd/cfc_puncture_offcenter_tov_amr_setneighbors_smoke_wide.
  athinput` — real CFC+BH-puncture+TOV physics, 8x8x8 root MeshBlock domain,
  wide enough to avoid domain-edge artifacts. This is what actually hits
  item 39c; it has never been run long enough to confirm or deny whether the
  §2c tie-break case occurs in it specifically (found via `CFC_DEBUG_NGHBR`
  topology dumps on this exact input, but not proven necessary for this
  input to run correctly — only necessary for a hypothetical more-adversarial
  topology).
- `tst/test_suite/unit_tests/test_ut_setneighbors_diag_collision_cpu.py` —
  runs the hydro reproducer, checks exit code + absence of `NANS_IN_CONS` in
  the newly-appended log region. Passes today (`1 passed in 0.31s`,
  confirmed this session).
- `tst/test_suite/dyngrmhd/test_dyngrmhd_setneighbors_cfc_amr_mpicpu.py` —
  runs the CFC wide-domain smoke test at 8 ranks (its input,
  `tst/inputs/cfc_setneighbors_amr_smoke.athinput`, confirmed byte-for-byte
  identical in physics parameters to the `_wide.athinput` fixture job 8830513
  validated — comments/output blocks differ only). Docstring and `xfail`
  reason corrected (were attributing item 39c to the refuted cache-desync
  theory) to state the actual, now-fixed orphan mechanism (§2b). **Still
  `xfail`**: item 39c itself is fixed, but the same input separately hits the
  elliptic-solver-convergence/SIGBUS issue described in §5's validation
  writeup — unrelated to `SetNeighbors`, not yet investigated on its own.
- **No test anywhere exercises §2c (the genuine tie-break collision)**.
  Building one is a concrete, well-specified first task for a redesign
  session: two *different*, asymmetrically-placed `<refined_region N>`
  blocks (not mirror-symmetric siblings under one parent) refining two
  different face-neighbors of a common coarser target block, so the
  target's diagonal direction is genuinely touched by two distinct finer
  leaves reached through different lineages (the shape item 39f's own live
  example has). A paired unit test should assert clean exit AND (via a
  `CFC_DEBUG_NGHBR`-style topology dump, already established instrumentation
  convention) that the collision is actually represented in the dump, not
  silently dropped or silently resolved to one candidate.

## 7. Why a new session for this, per the user's own question

Answered directly in-session: this session's history includes several
refuted theories (cache-desync) and one implemented-then-reverted redesign
attempt, all of which are context weight a fresh session doesn't need to
carry. The two goals are also different in kind and urgency — a small patch
to unblock production vs. a real architectural fix needing iterative Aurora
validation (slow, queue-gated) — and mixing them risks the small patch
waiting on the hard problem. This file is the intended full substitute for
re-deriving any of the above from scratch.

## 8. 2026-09-16, follow-up session: the capacity premise is wrong

**Summary**: sections 1 and 2c are incorrect. There is no slot-capacity
problem. The neighbor table produced by the *pre-`fee50981`* octant-parity
guard is exactly symmetric, with no unfilled ghost region, on the real
production topology and on every random AMR tree tried. The two commits on
`proj/tde` are what introduce the asymmetry that aborts the restart. Nothing
has been changed about the rule yet -- this session added only a self-check,
a reproducer and this writeup -- but the evidence below should be read before
any further work on section 4's redesign, which would be solving a problem
that does not exist.

### 8.1 The invariant that actually matters

Neither `bvals.cpp` nor `multigrid_bvals.cpp` negotiates anything across
ranks: each builds its MPI messages purely from its own `nghbr` rows. A send
for `(m,n)` carries this block's data to slot `dest` of block `gid`, and the
peer's matching recv exists only if the peer's own row at that slot names us
back. So the whole scheme is correct if and only if

```
nghbr(b,n) = {gid=T, dest=d}   =>   nghbr(T,d) = {gid=b, dest=n}
```

holds for every populated slot. Violate it and the aggregated per-peer message
sizes disagree -- which is exactly `internal_Waitall` / `Message truncated`,
surfacing in whichever subsystem exchanges first (multigrid, since its root-grid
setup runs before cycle 0) with nothing pointing back at `SetNeighbors`.

This is a *property of the tree alone*. It needs no MPI, no physics and no
Aurora job to evaluate.

### 8.2 Measured, on the exact topology that is failing

`MeshBlock::CheckNeighborSymmetry` (new, section 8.5) rebuilds the rows of
**every** gid and checks the invariant. Run on the real blocked restart's own
checkpoint -- `tde_elliptic_tracker_nexteval_v2/output-0000/rst/
cfc_tde_wd_imbh_elliptic_tracker.00010.rst`, cycle 10000, 1268 MeshBlocks,
levels 3-7, `diode` BCs:

| rule | registrations | asymmetric | ghost-coverage holes |
|---|---|---|---|
| pre-`fee50981` octant-parity guard | 26834 | **0** | **0** |
| HEAD (`fee50981` + `8641e00f`)     | 28006 | **1172** (all "stolen") | -- |

HEAD adds exactly 1172 entries over the old rule, removes none, and alters
none. **All 1172 added entries are asymmetric, and every asymmetry is one of
the added entries.** 586 distinct source blocks are involved, which across 96
ranks makes cross-rank orphans a certainty -- consistent with job 8830813
aborting on 78 of 96 ranks.

Independently reproduced two ways that share no code: an offline Python model
of `FindNeighbor`/`SetNeighbors`/`NeighborIndex` reading the same restart file,
and the in-tree C++ audit. Same totals, same first offending pairs (`gid=63`
slot 16 -> `gid=60` dest 22, whose target slot is held by `gid=70`; `gid=64`
slot 16 -> `gid=55` dest 22, held by `gid=242`; ...).

Fuzzing 40 random 2:1-balanced AMR trees (periodic and outflow, 4 levels,
~400-900 blocks): parity rule 0 asymmetries and 0 ghost holes every time;
HEAD 26,244 asymmetries in total.

### 8.3 Why the octant-parity guard is right, and complete

Let `B` be at level `L`, direction `d` a diagonal, and let `FindNeighbor`
return a coarser leaf `T` at level `L-1`. With `A = B.lloc + d`, the node
`FindNeighbor` descends to gives `T.lloc = A >> 1`. Comparing that against
`B`'s parent, `(A - d) >> 1`, component by component:

* on an axis with `d_i = 0`, they always agree;
* on an axis with `d_i = +1`, they agree iff `b_i` is odd;
* on an axis with `d_i = -1`, they agree iff `b_i` is even.

That is precisely `myox_i == d_i` -- the old guard. So:

**Parity matches on every diagonal axis.** `T.lloc - d` *is* `B`'s parent, `T`'s
own reciprocal query descends into it, and its `GetLeaf(ff...)` resolves to
exactly `B` (the `ff` indices are `1-(d_i+1)/2`, which equal `B`'s child indices
under exactly this parity condition). Both sides register, slots and `dest`
agree. Symmetric by construction.

**Parity mismatches on a non-empty axis set `S`.** Then `|S| = 3` is impossible
(it would make `T` equal `B`'s parent, which is internal, not a leaf, so
`FindNeighbor` descends past it and returns a same-level-or-finer node instead).
For `|S| = 1` or `2`, `T` is `B`'s *face or edge* neighbour in the reduced
direction `d'` (`d` with the axes in `S` zeroed) -- not a genuinely separate
diagonal block at all. And that lower-order relationship is itself always
registered: faces register unconditionally, and the reduced edge case has an
empty mismatch set, so the parity rule registers it too.

**No ghost region is left unfilled.** `InitRecvIndices`' `icoar` range for the
reduced slot extends `ng` cells along each free axis `i`: toward `+i` when
`f[i] == 0` and toward `-i` when `f[i] == 1` (`buffs_cc.cpp`, the `ox1 == 0` /
`ox2 == 0` / `ox3 == 0` branches). A mismatch on axis `i` means `d_i = +1` with
`b_i` even (`f[i] = 0`), or `d_i = -1` with `b_i` odd (`f[i] = 1`) -- i.e. the
extension direction *equals* `d_i`, every time. The coarse face/edge buffer from
the very same block `T` already covers the region the diagonal slot would have
covered. (The `(f1,f2)` -> axis mapping was checked against each of the six
registration sites individually; they all agree with the `icoar` branches.)

So a parity-mismatched coarser diagonal is redundant, and registering it is
worse than redundant: the `dest` it computes, `NeighborIndex(-d, ...)`, names a
slot on `T` that `T`'s own query fills from a *different* block entirely. That
is the "stolen" case, and it is the whole of the 1172.

### 8.4 What this means for the earlier items

* **Section 1's capacity argument does not hold.** Under the parity rule each
  `(target, slot)` pair has exactly one claimant; the finer, same-level and
  coarser branches are provably reciprocal. Corners need no subdivision because
  a corner slot has exactly one geometric owner.
* **Section 2c's "N-to-1 tie-break collision" is not two genuine neighbours.**
  The second candidate in items 39f / 63 is a parity-mismatched one: redundant,
  already covered by a face or edge buffer. There is nothing to arbitrate.
* **Section 3 step 4's regression is consistent with this.** That attempt
  removed the gating *and* added capacity; it registered invalid relationships
  because there was never a shortage of slots, only candidates that should not
  have been registered.
* **Section 2a's attribution is doubtful.** The production `NANS_IN_CONS`
  cascade was read as under-registration by the parity guard, but the guard
  drops nothing that is needed (8.3). Section 5's own validation is consistent
  with that: the patched wide-domain run produced 7987 NaNs downstream of a
  *cycle-0 elliptic-solver convergence failure*, against 7964 in the unmodified
  control -- i.e. the NaNs track the elliptic solver, not the registration rule.
  That remains a separate, open question; it is not evidence for the rule change.

### 8.5 What this session added (no rule change)

* `MeshBlock::BuildNeighborRow` (`src/mesh/meshblock.cpp`) -- the per-gid
  registration logic, factored out verbatim from the body of `SetNeighbors`'
  block loop. Now the single authoritative copy: `SetNeighbors` fills this
  rank's rows with it, and the audit rebuilds *all* gids' rows with it, so the
  audit can never drift from the rule it audits. Depends only on globally
  replicated state (tree, `lloc_eachmb`, `ranklist`), which is what makes a
  global check legal from any rank.
* `MeshBlock::CheckNeighborSymmetry` -- opt-in audit of the 8.1 invariant,
  `O(nmb_total * nnghbr)` tree walks once per mesh build (so it also covers
  post-AMR remesh, via `mesh_refinement.cpp`'s own `SetNeighbors` call).
  `ATHENAK_CHECK_NGHBR_SYMMETRY=1` reports and continues (the useful mode when
  diagnosing an existing abort -- audit output and real abort land in one log);
  `=2` reports and aborts. Unset, it costs nothing.
* `inputs/tests/amr_hydro_nghbr_asymmetry.athinput` (+ a copy under
  `tst/inputs/`) -- the small reproducer section 6 asked for, and an answer to
  why the old one never worked. **Two refined regions at the same level is not
  enough, and neither is one region: what is required is three levels with the
  finest region placed asymmetrically inside the coarser one.** 127 MeshBlocks,
  serial, seconds. Reports `124 asymmetric (orphan=44 stolen=80)` under HEAD;
  the offline model puts the parity rule at 0 on the identical tree.
* `scripts/nghbr_symmetry_offline.py` -- the same audit done straight from a
  restart file, with no build, no MPI and no allocation: a `.rst` carries the
  complete tree plus the parameter deck, which is all `SetNeighbors` depends on,
  so a large failing run's table can be rebuilt and checked in seconds. It
  implements *both* rules (`parity` and `head`) so they can be compared on
  identical input, and additionally checks the parity rule for ghost-coverage
  holes -- symmetry alone is not sufficient, a rule could be symmetric and still
  wrong by dropping needed data. This is what produced 8.2's table; it and the
  in-tree audit were written independently and agree exactly.
* `tst/test_suite/unit_tests/test_ut_nghbr_symmetry_cpu.py` -- runs it and
  asserts a clean audit. Currently `xfail(strict=True)`: it flips to a failure
  the moment the rule is fixed, which is the signal to drop the marker.

### 8.6 In-situ confirmation at production scale (job 8832179)

Predictions in 8.2 were made offline, from two independent implementations, and
written into the case's `BATCH/CONFIG` **before** submission. Case
`tde_elliptic_tracker_nexteval_v2_nghbraudit`, 8 nodes / 96 ranks -- identical
to the blocked restart in every respect except `ATHENAK_CHECK_NGHBR_SYMMETRY=1`,
with the registration rule deliberately left unchanged so the run had to fail
the same way jobs 8830813 and 8831986 did. It did:

```
Restarting from parent: parent/rst/cfc_tde_wd_imbh_elliptic_tracker.00010.rst
### nghbr symmetry audit: 1268 MeshBlocks, 28006 registrations, 1172 asymmetric
    (orphan=0 stolen=1172 destmismatch=0), 847 of them cross-rank
  STOLEN  gid=63 (lev 4 @ 5,6,6, rank 4) slot=16 -> gid=60 (lev 3 @ 2,2,3, rank 4)
          dest=22 ; that slot holds gid=70 dest=16
  ...
Multigrid root grid levels: 4
Number of MeshBlocks in the pack: 13
MeshBlock size: 16 x 16 x 16          [x4]
Abort(17) ... Fatal error in internal_Waitall                [64 of 96 ranks]
```

Every offline number reproduced exactly at 96 ranks (1268 / 28006 / 1172, all
"stolen", same offending pairs). The one genuinely new number is **847 of the
1172 are cross-rank** -- 847 sends with no matching recv, spread over most of
the job. The abort then lands 6 s in, immediately after multigrid's root-grid
setup prints and before a single `cycle=` line: byte-for-byte the signature of
8830813 and 8831986, and precisely where 2c predicted it (multigrid's own
exchange is the first cross-rank communication after the mesh is built).

What this does and does not establish. It establishes that the asymmetries are
real, present at production scale, overwhelmingly cross-rank, and that the abort
follows them at the predicted place -- an observation-only run cannot do better
than that. It does **not** by itself prove causation; that needs the A/B against
the parity rule, which was deliberately not attempted here.

### 8.7 The fix, and what it was checked against

The octant-parity condition is restored at all four diagonal sites, as a forward
commit -- `fee50981` and `8641e00f` stay in the log. The resulting rule is
identical to the pre-`fee50981` one: the only differences from that code are the
`do_register` refactor (`level >= lloc.level || parity` split into a branch) and
`myox_i = myfx_i*2 - 1` in place of the open-coded `((lloc.lx_i & 1) == 1)*2 - 1`,
which is the same expression. Verified by normalised diff against
`fee50981^:src/mesh/meshblock.cpp`.

Each site now carries the derivation (8.3) in a comment, so the next reader does
not have to rediscover why the condition is what it is -- the absence of that
explanation is a large part of how this went wrong: the guard looked like an
unexplained heuristic, so it was replaced by one.

Checked, all three matching the offline predictions exactly:

| topology | before | after |
|---|---|---|
| `amr_hydro_nghbr_asymmetry.athinput` (127 blocks) | 2288 reg, 124 asym | **2164 reg, 0 asym** |
| `lwave_hydro_diag_collision.athinput` (71 blocks) | 1780 reg, 0 asym | 1780 reg, 0 asym |
| TDE production checkpoint (1268 blocks) | 28006 reg, 1172 asym | **26834 reg, 0 asym** |

plus 0 ghost-coverage holes on all three, and the `xfail` dropped from
`test_ut_nghbr_symmetry_cpu.py`.

### 8.8 Still open, and NOT explained by any of this

Section 2a attributed the original production `NANS_IN_CONS` cascade (cycle
10660, t~446) to `SetNeighbors` under-registration. 8.3 and 8.4 make that
attribution doubtful: the parity rule drops nothing that is needed. Section 5's
own validation points the same way -- the patched wide-domain run produced 7987
NaNs downstream of a *cycle-0 elliptic-solver convergence failure*, against 7964
in the unmodified control, i.e. the NaNs track the elliptic solver, not the
registration rule. So the cause of the cascade that started this whole
investigation may never have been found, and the restart validated here could
still run into it again around the same cycle. That is a separate investigation;
it should not be folded back into this one.


