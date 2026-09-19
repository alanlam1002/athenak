# `MeshBlock::SetNeighbors` octant-parity guard — correction to item 39b

Ported 2026-09-19 from `~/athenak_tde`'s `src/cfc/SETNEIGHBORS_HANDOFF.md`
section 8 (commits `dd31ea6b`..`69918bda` on `proj/tde`), because this repo's
own `DEVELOPMENT.md` item 39b (and everything descending from it — 39c
through the "Current state" roadmap at the end of item 39) drew the opposite,
incorrect conclusion and recommends a fix that was tried on that sibling
branch and found to break multi-rank AMR runs at production scale. See the
2026-09-16 correction block inserted into this repo's `DEVELOPMENT.md`
directly under item 39b for the short version and the pointer here.

This file is that branch's own section 8, kept close to verbatim so the full
derivation and evidence trail survive independently of that other checkout
being available. Job IDs, SLURM paths and specific input names below are
from `~/athenak_tde` (a different repo/production topology), not this one —
kept as citations, not claims about this repo's own runs. This repo's own
`SetNeighbors` (`src/mesh/meshblock.cpp`) carries the same octant-parity
guard, unmodified, at the same four diagonal sites (`git diff` against `HEAD`
is clean — no version of item 39b's fix was ever committed here either).

---

**Summary**: there is no slot-capacity problem. The neighbor table produced
by the octant-parity guard is exactly symmetric, with no unfilled ghost
region, on the real production topology and on every random AMR tree tried.
The two commits that replaced it on `proj/tde` (`fee50981`, `8641e00f`) are
what introduce the asymmetry that aborts the restart. The evidence below
should be read before any further work on this repo's own item 39e/63
"extra-slot-capacity" redesign, which would be solving a problem that does
not exist.

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
sizes disagree — which is exactly `internal_Waitall` / `Message truncated`,
surfacing in whichever subsystem exchanges first (multigrid, since its root-grid
setup runs before cycle 0) with nothing pointing back at `SetNeighbors`.

This is a *property of the tree alone*. It needs no MPI, no physics and no
Aurora job to evaluate.

### 8.2 Measured, on the exact topology that was failing (TDE production)

`MeshBlock::CheckNeighborSymmetry` (does not exist in this repo yet — see
8.5) rebuilds the rows of **every** gid and checks the invariant. Run on the
real blocked restart's own checkpoint (TDE's `tde_elliptic_tracker_nexteval_v2`,
cycle 10000, 1268 MeshBlocks, levels 3-7, `diode` BCs):

| rule | registrations | asymmetric | ghost-coverage holes |
|---|---|---|---|
| pre-`fee50981` octant-parity guard | 26834 | **0** | **0** |
| HEAD (`fee50981` + `8641e00f`)     | 28006 | **1172** (all "stolen") | — |

HEAD adds exactly 1172 entries over the old rule, removes none, and alters
none. **All 1172 added entries are asymmetric, and every asymmetry is one of
the added entries.** 586 distinct source blocks are involved, which across 96
ranks makes cross-rank orphans a certainty — consistent with a job aborting
on 78 of 96 ranks.

Independently reproduced two ways that share no code: an offline Python model
of `FindNeighbor`/`SetNeighbors`/`NeighborIndex` reading the same restart file,
and an in-tree C++ audit. Same totals, same first offending pairs.

Fuzzing 40 random 2:1-balanced AMR trees (periodic and outflow, 4 levels,
~400-900 blocks): parity rule 0 asymmetries and 0 ghost holes every time;
the replacement rule 26,244 asymmetries in total.

### 8.3 Why the octant-parity guard is right, and complete

Let `B` be at level `L`, direction `d` a diagonal, and let `FindNeighbor`
return a coarser leaf `T` at level `L-1`. With `A = B.lloc + d`, the node
`FindNeighbor` descends to gives `T.lloc = A >> 1`. Comparing that against
`B`'s parent, `(A - d) >> 1`, component by component:

* on an axis with `d_i = 0`, they always agree;
* on an axis with `d_i = +1`, they agree iff `b_i` is odd;
* on an axis with `d_i = -1`, they agree iff `b_i` is even.

That is precisely `myox_i == d_i` — the guard this repo's `SetNeighbors`
carries at all four diagonal sites (`myox1==n && myox2==m`, etc.). So:

**Parity matches on every diagonal axis.** `T.lloc - d` *is* `B`'s parent, `T`'s
own reciprocal query descends into it, and its `GetLeaf(ff...)` resolves to
exactly `B` (the `ff` indices are `1-(d_i+1)/2`, which equal `B`'s child indices
under exactly this parity condition). Both sides register, slots and `dest`
agree. Symmetric by construction.

**Parity mismatches on a non-empty axis set `S`.** Then `|S| = 3` is impossible
(it would make `T` equal `B`'s parent, which is internal, not a leaf, so
`FindNeighbor` descends past it and returns a same-level-or-finer node instead).
For `|S| = 1` or `2`, `T` is `B`'s *face or edge* neighbour in the reduced
direction `d'` (`d` with the axes in `S` zeroed) — not a genuinely separate
diagonal block at all. And that lower-order relationship is itself always
registered: faces register unconditionally, and the reduced edge case has an
empty mismatch set, so the parity rule registers it too.

**No ghost region is left unfilled.** `InitRecvIndices`'s `icoar` range for the
reduced slot extends `ng` cells along each free axis `i`: toward `+i` when
`f[i] == 0` and toward `-i` when `f[i] == 1` (`buffs_cc.cpp`, the `ox1 == 0` /
`ox2 == 0` / `ox3 == 0` branches). A mismatch on axis `i` means `d_i = +1` with
`b_i` even (`f[i] = 0`), or `d_i = -1` with `b_i` odd (`f[i] = 1`) — i.e. the
extension direction *equals* `d_i`, every time. The coarse face/edge buffer from
the very same block `T` already covers the region the diagonal slot would have
covered.

So a parity-mismatched coarser diagonal is redundant, and registering it is
worse than redundant: the `dest` it computes, `NeighborIndex(-d, ...)`, names a
slot on `T` that `T`'s own query fills from a *different* block entirely. That
is the "stolen" case, and it is the whole of the 1172.

### 8.4 What this means for this repo's own item 39/39e/63 reasoning

* **A slot-capacity argument does not hold.** Under the parity rule each
  `(target, slot)` pair has exactly one claimant; the finer, same-level and
  coarser branches are provably reciprocal. Corners need no subdivision because
  a corner slot has exactly one geometric owner.
* **An "N-to-1 tie-break collision" (this repo's items 39e/39f/63) is not two
  genuine neighbours.** The second candidate is a parity-mismatched one:
  redundant, already covered by a face or edge buffer. There is nothing to
  arbitrate — do not build the extra-slot-capacity fallback.
* **item 38's original `NANS_IN_CONS` residual is not this.** This repo's own
  item 42 already found that residual's real cause (`u_adm` missing a
  physical-BC pass, fixed by item 41) — the guard drops nothing that is
  needed (8.3), so it was never a plausible cause in the first place.

### 8.5 In-tree tooling added on the sibling branch (not yet ported here)

* `MeshBlock::BuildNeighborRow` (`src/mesh/meshblock.cpp`) — the per-gid
  registration logic, factored out verbatim from the body of `SetNeighbors`'
  block loop, so an audit can never drift from the rule it audits.
* `MeshBlock::CheckNeighborSymmetry` — opt-in audit of the 8.1 invariant.
  `ATHENAK_CHECK_NGHBR_SYMMETRY=1` reports and continues; `=2` reports and
  aborts. Unset, it costs nothing.
* `inputs/tests/amr_hydro_nghbr_asymmetry.athinput` — small reproducer: three
  levels with the finest `<refined_regionN>` placed asymmetrically inside the
  coarser one (two regions at the same level, or one region, gives an
  octant-symmetric tree with zero asymmetries either way — that is why this
  repo's `lwave_hydro_diag_collision.athinput` never caught this). 127
  MeshBlocks, serial, seconds.
* `scripts/nghbr_symmetry_offline.py` — the same audit straight from a `.rst`,
  no build, no MPI, no allocation. Implements both rules (`parity` and
  `head`/replacement) so they can be compared on identical input, and checks
  the parity rule for ghost-coverage holes too (symmetry alone isn't
  sufficient — a rule could be symmetric and still wrong by dropping needed
  data).

None of this is CFC-specific — if useful here, port `BuildNeighborRow` +
`CheckNeighborSymmetry` + the offline script directly; they are independent
of any CFC/elliptic-solve code.

### 8.6 In-situ confirmation at production scale (TDE job 8832179)

Case `tde_elliptic_tracker_nexteval_v2_nghbraudit`, 8 nodes / 96 ranks —
identical to the blocked restart in every respect except
`ATHENAK_CHECK_NGHBR_SYMMETRY=1`, registration rule deliberately left
unchanged so the run had to fail the same way earlier jobs did. It did:

```
Restarting from parent: parent/rst/cfc_tde_wd_imbh_elliptic_tracker.00010.rst
### nghbr symmetry audit: 1268 MeshBlocks, 28006 registrations, 1172 asymmetric
    (orphan=0 stolen=1172 destmismatch=0), 847 of them cross-rank
  STOLEN  gid=63 (lev 4 @ 5,6,6, rank 4) slot=16 -> gid=60 (lev 3 @ 2,2,3, rank 4)
          dest=22 ; that slot holds gid=70 dest=16
  ...
Abort(17) ... Fatal error in internal_Waitall                [64 of 96 ranks]
```

Every offline number reproduced exactly at 96 ranks. 847 of the 1172 asymmetric
entries are cross-rank — sends with no matching recv. The abort lands 6s in,
immediately after multigrid's root-grid setup, before a single `cycle=` line.

### 8.7 The fix, and what it was checked against

The octant-parity condition, restored at all four diagonal sites, is identical
to the pre-`fee50981` rule (verified by normalised diff). `origin/main`
carries this same guard unchanged too — this is not a novel rule argued from
first principles, it is upstream AthenaK's own, exercised by every other user
of the code.

| topology | before | after |
|---|---|---|
| `amr_hydro_nghbr_asymmetry.athinput` (127 blocks) | 2288 reg, 124 asym | **2164 reg, 0 asym** |
| `lwave_hydro_diag_collision.athinput` (71 blocks) | 1780 reg, 0 asym | 1780 reg, 0 asym |
| TDE production checkpoint (1268 blocks) | 28006 reg, 1172 asym | **26834 reg, 0 asym** |

plus 0 ghost-coverage holes on all three.

### 8.8 Validation run (TDE job 8832259): fix confirmed

The restart that had never survived to cycle 0 ran **600 clean cycles**
(10000 → 10599), no MPI abort, no NaN. AMR remeshed **51 times** during those
cycles; all 51 reported **0 asymmetric**, over MeshBlock counts from 757 to
1345 across 20 distinct tree sizes — validated on 51 independently generated,
dynamically evolving AMR topologies, not just the one frozen checkpoint tree.

(That run then hit an unrelated cycle-10600 NaN-cascade problem — a separate,
already root-caused issue in the TDE repo, documented there in
`NANCASCADE_HANDOFF.md`; not a neighbour-table problem, and out of scope
here.)

### 8.9 The tests that were said to reproduce this (on the sibling branch)

Worth stating plainly, because it is the reason the wrong diagnosis survived so
long there: **no test in that repository ever demonstrated the upstream rule
failing.** Both of that branch's SetNeighbors tests were added by the bad
commit itself (validating it, not exposing a pre-existing defect), and one of
them used a two-level, octant-symmetric topology that passes identically on
both rules and cannot distinguish them — a caution for this repo's own
`lwave_hydro_diag_collision.athinput`, which is likewise too symmetric to
distinguish the rules (0 asymmetric under either rule, checked directly).

### 8.10 Scope of the correctness claim

What has been established is that the upstream octant-parity rule yields a
**symmetric, ghost-complete** neighbour table on every topology obtainable —
production checkpoints, dynamic AMR remeshes, random 2:1-balanced trees, and
small reproducers. That is exactly the property `internal_Waitall` depends on.

It is **not** a proof that `SetNeighbors` is correct in general. Not audited:
buffer-size agreement between paired slots, the 1D/2D paths, flux correction,
and `bvals_part.cpp`'s reliance on per-category slot contiguity.

### 8.11 If picking this back up in this repo

The guard in this repo's `src/mesh/meshblock.cpp` is already correct and
already matches upstream `origin/main` — there is nothing to fix here. If a
future investigation (in this repo, under whatever item number is current by
then) again concludes "the octant-parity guard is dropping a needed diagonal
neighbour," that conclusion should be treated as suspect by default: re-derive
against 8.3's parity argument and check `buffs_cc.cpp`'s `icoar`/`iprol`
extension on the reduced face/edge slot before acting on it, and run
`scripts/nghbr_symmetry_offline.py` (port it if not already present) on a real
restart before touching `SetNeighbors` at all.
