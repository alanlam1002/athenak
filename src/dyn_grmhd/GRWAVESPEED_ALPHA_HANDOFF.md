# Handoff: unguarded 1/alpha divisions in the shared dyn_grmhd Riemann-solver /
# timestep code

Repo: ~/athenak_tde, branch proj/tde. This is core `src/dyn_grmhd/` + `src/eos/`
code, used by BOTH Z4c-based and CFC-based dyn_grmhd evolution -- NOT a
CFC-specific bug, even though it was found while debugging a CFC/TDE run. Read
this whole file before touching anything; it is the complete, closed diagnosis.
No source code has been changed for this bug yet -- diagnosis only.

For provenance/context (you should not need to re-derive any of this, but it's
here if something doesn't add up): `src/cfc/NANCASCADE_HANDOFF.md` (the
separate, already-resolved t~445 NANS_IN_CONS cascade) and
`src/cfc/SETNEIGHBORS_HANDOFF.md` (the separate, already-resolved neighbour-
table symmetry bug). This bug is neither of those. It was found while
smoke-testing an unrelated AMR-refinement-criterion change
(`src/cfc/NANCASCADE_HANDOFF.md` Sec 22.6, "Finding A" / "item 5"), and this
file supersedes that section's open question -- Finding A's root cause is now
identified, below.

## 1. The bug

`src/eos/primitive_solver_hyd.hpp:644`, inside `GetGRFastMagnetosonicSpeeds`:

```cpp
Real ialpha = 1.0/alpha;
```

`alpha` here is the ADM lapse at a point, passed in by the caller with no
floor or guard. If `alpha` is zero or extremely small, `ialpha` is `Inf`, which
poisons `u0 = W*ialpha`, `g00 = -ialpha*ialpha`, and from there the quadratic
`a`/`b`/`c` -> `a1 = b/a` / `a0 = c/a` -> `lambda_p`/`lambda_m` (the fast
magnetosonic wave speeds) computed by this function. `Inf - Inf` / `Inf * 0`
patterns in that chain readily produce NaN, not just Inf.

**This is not an isolated line.** The same unguarded pattern recurs at every
site that needs `1/alpha` or `X/alpha` in this part of the code (found via
`grep -rn "1\.0/alpha\|/ *alpha\b\|/alpha)" src/dyn_grmhd/ src/eos/primitive_solver_hyd.hpp`,
not yet exhaustively audited beyond this):

- `src/eos/primitive_solver_hyd.hpp:644` -- `GetGRFastMagnetosonicSpeeds`'s own `ialpha`.
- `src/dyn_grmhd/dyn_grmhd_newdt.cpp:171` -- a *separate* `ialpha = 1.0/alpha`
  in the CFL timestep calculation, computing `bu0` (comoving-frame field
  component), independent of the function above (which `dyn_grmhd_newdt.cpp`
  also calls, at lines ~180/185/190, for the wave speeds themselves).
- `src/dyn_grmhd/rsolvers/hlle_dyn_grmhd.hpp:74` and `:224` -- two call sites
  (likely the two flux-calculation entry points -- single-state and the
  main loop, or L/R -- not yet distinguished) computing
  `Real qa = lambda_r*lambda_l/alpha;` directly.
- `src/dyn_grmhd/rsolvers/flux_dyn_grmhd.hpp:41` -- another
  `const Real ialpha = 1.0/alpha;`.

`llf_dyn_grmhd.hpp` calls `GetGRFastMagnetosonicSpeeds` too (so it inherits
that one site's exposure) but does not appear to have its own independent
`/alpha` beyond that, per the same grep -- **not fully verified**, worth
re-checking directly before considering the audit complete.

**Scope**: `GetGRFastMagnetosonicSpeeds` and `dyn_grmhd_newdt.cpp` are called
by the general dyn_grmhd task graph regardless of whether Z4c or CFC supplies
the metric (`src/dyn_grmhd/dyn_grmhd.cpp`'s task list is shared; CFC is the
`else` branch at dyn_grmhd.cpp:245, Z4c the branch above it, both go through
the same `MHD_C2P`/Riemann-solver/`MHD_Newdt` tasks). The Riemann solvers
(`hlle_dyn_grmhd.hpp`, `llf_dyn_grmhd.hpp`) are likewise shared. **This is a
general dyn_grmhd numerics bug, not something in CFC's own code.**

## 2. Why alpha can legitimately be ~0 here, and why that's not itself a bug

The trigger, in this instance: `src/cfc/cfc_puncture.hpp`'s maximal-slicing
trumpet background has `alpha0^2 = 1 - 2/rrs + 1.6875/rrs^4`, a double root at
the trumpet throat `rrs = kTrumpetThroat = 1.5` (in `m_bh=1` units). A cell
whose isotropic radius `r` maps, via `TrumpetIsoToAreal(r/m_bh)`, to an areal
radius `rrs` very close to 1.5 legitimately has `alpha0` very close to (or, at
the clamp, exactly) zero. `cfc_puncture.hpp:118-125` already has a deliberate,
correct clamp for this (`alpha_sq > 0.0 ? alpha_sq : 0.0`, a fix from earlier
in this investigation, companion to the `grad_ap6_0` fix in
`NANCASCADE_HANDOFF.md` Sec 15) -- **that part of the code is fine and not
the bug.** The bug is downstream: nothing that later *consumes* a near-zero
`alpha` in a division is protected.

Physically, this is also just what maximal-slicing / moving-puncture data
does everywhere near a horizon-ish region -- lapse collapse is the point of
the slicing. Z4c-based BH evolutions presumably never trip this because their
excision is tied to the lapse from the very first cycle (see Sec 3), so an
active, non-excised grid cell essentially never gets this close to zero lapse
in practice. CFC's TDE setup is (as far as this investigation found) the
first configuration to actually place a live cell there.

## 3. Why excision does NOT protect against this (checked directly, not assumed)

`src/coordinates/coordinates.cpp:112-114`: `excision_floor`/`excision_flux`
are populated at construction ONLY for `excision_scheme=fixed`. For `lapse`/
`horizon` schemes they are `Kokkos::realloc`'d (arbitrary/unspecified content)
and are only ever filled by `DynGRMHD::UpdateExcisionMasks`
(`src/dyn_grmhd/dyn_grmhd.cpp:524-527`), which is a **per-cycle task graph
entry** (`MHD_Excise`, `src/dyn_grmhd/dyn_grmhd.cpp:243/263`) -- it is never
called during problem-generator setup. The initial primitive-to-conserved
conversion, `DynGRMHDPS::PrimToConInit`
(`src/dyn_grmhd/dyn_grmhd.cpp:285-294`), is called directly and only once
from the problem generator (`src/pgen/dyn_grmhd/dyngr_tov.cpp:551`), before
the task graph -- and hence before `UpdateExcisionMasks` -- has ever run.

So even with `<coord> excise=true, excision_scheme=lapse` set (the full
production recipe, `NANCASCADE_HANDOFF.md` Sec 19.2), the very first flux
evaluation and timestep calculation at cycle 0 have **zero** excision
protection, regardless of setting. This was verified empirically, not just
argued from the code: see Sec 5.

## 4. The empirical trace that pins this down (not inference alone)

Reproducer: `inputs/dyn_grmhd/cfc_tde_wd_imbh_amr.athinput` (as it exists in
the current proj/tde working tree, including this session's uncommitted items
1-4 -- irrelevant to this bug, just noting the state it was run in), 1 node /
12 ranks, CPU, debug queue, `max_nmb_per_rank=400`, `<cfc> src_nan_check=1`.
Job 8845399, run dir
`/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/tde_tracker_cpu_excise_v1/`,
with the FULL production excision override applied on top of the fixture's
own (inherited, never-actually-used-in-production) `excise=false` default:

```
<coord>  excise = true
         excision_scheme = lapse
         dexcise = 1.0e-24
         texcise = 1.0e-22
```

Result: **identical failure** to the no-excision run (job 8843221,
`tde_tracker_cpu_bracket_v1/`) -- same 11,989 `NANS_IN_CONS`, same collapse at
t=0.119 (cycle 1), same all-conserved-fields-NaN signature. Excision made
zero measurable difference, consistent with Sec 3's code-level argument.

`src_nan_check`'s ordered fingerprint (`tde_tracker_cpu_excise_v1.out`) is the
decisive piece:

```
### CFC NONFINITE [matter.in[pmhd->u0]] stage=1 ncycle=0 count=2317000
    first: rank=8 mb=1 chan=0 (k,j,i)=(4,4,6)
    x=(-0.421875, 1.01562, 1.01562) val=-nan
```

-- reported at **ncycle=0, stage=1**, i.e. before even the first RK substage
of the first cycle has properly completed, at a *real* grid location (not the
domain-corner default index that some of this run's other labels report,
which is a "first flat-array NaN found" artifact, not a physical location --
see `NANCASCADE_HANDOFF.md` Sec 16.2's parenthetical on this). `psi.out` and
`lapse.out` (the CFC elliptic-solve diagnostics) do not report ANYTHING until
**ncycle=2**, strictly after `matter.in` is already corrupted. This is the
*reverse* order from the already-resolved production cascade's cause A
(`NANCASCADE_HANDOFF.md` Sec 17.2: there, `psi4`/the metric solve fails
first, matter follows) -- confirming this is a genuinely different mechanism,
not a recurrence of cause A.

The flagged cell's position: `x = (-0.421875, 1.01562, 1.01562)`, isotropic
radius `r = sqrt(0.421875^2 + 1.01562^2 + 1.01562^2) ~= 1.497`, in `m_bh=1`
units (the puncture mass is 1 in these code units; the domain and `x=30 r_g`
star placement in this project's other docs use the same convention) --
suspiciously close to `kTrumpetThroat=1.5`. This is fully consistent with:
`TrumpetIsoToAreal(1.497)` root-finding to an areal radius barely above 1.5
(the areal-to-iso map `TrumpetArealToIso(1.5)=0` exactly, so small isotropic
radii map close to, but strictly above, the throat) -- i.e. `alpha0` at this
cell is legitimately very close to zero, and something in the flux/timestep
chain divides by it unprotected.

**Not yet directly confirmed** (do this first in the fixing session, it's
cheap): instrument or hand-evaluate `GetGRFastMagnetosonicSpeeds` /
`dyn_grmhd_newdt.cpp`'s `bu0` calc at exactly this cell's `alpha` value to
show the `Inf`/`NaN` appearing there specifically, rather than relying on the
"this is the only plausible unguarded division reachable this early" argument
above. The evidence is strong but this last link (cell's actual alpha value,
still not read out) has not been directly observed.

## 5. Why this is genuinely new, not a recurrence of anything already fixed

- Ruled out rank-count dependence: 12-rank CPU (job 8843221) and 24-rank GPU
  (job 8841249, the original repro) fail identically.
- Ruled out CPU-vs-GPU: same failure on both platforms, same magnitude.
- Ruled out cause A (unexcised puncture pile-up feeding the psi solve,
  `NANCASCADE_HANDOFF.md` Sec 17): that mechanism fails metric-first,
  develops over ~445 M of dynamical time, and IS fixed by `excise=true`. This
  bug fails matter-first, at cycle 1, from pure grid geometry present at
  t=0, and is NOT fixed by `excise=true` (Sec 4).
- Ruled out `grad_ap6_0` (`NANCASCADE_HANDOFF.md` Sec 15, already fixed) and
  the `alpha_sq` clamp (Sec 2 above, already fixed and correct): both are in
  `cfc_puncture.cpp`/`cfc_puncture.hpp`, upstream of this bug, and were
  re-checked directly in this investigation -- they are not implicated.
- Ruled out the SetNeighbors/nghbr asymmetry bug entirely (`
  SETNEIGHBORS_HANDOFF.md`): unrelated subsystem, already fixed and pushed to
  `origin/proj/tde` (`177cac40`).

## 6. Why this probably does NOT explain the ORIGINAL production cascade

The original t~445 cascade (`NANCASCADE_HANDOFF.md` Sec 17) has a `psi4`-NaN,
metric-first fingerprint, is time-locked (not cycle-locked) across multiple
code versions and `solve_interval` settings, and is fully resolved by
`excise=true` + the Sec 15 fix -- `tde_prod_fixed_v3` ran clean to `tlim=500`
with that recipe. This new bug's fingerprint (matter-first, cycle-locked, NOT
fixed by excision) is different enough that it's very unlikely to be a
mislabeled recurrence. It's presented here as its own, separate, real bug --
just one that so far has only been OBSERVED in a smoke-test fixture that
happens to place a cell unusually close to the trumpet throat at t=0, not (as
far as this investigation checked) in the validated production run itself.
**Not yet checked**: whether `tde_prod_fixed_v3`'s own grid, at any point in
its evolution (including near periapsis, where the puncture-relative geometry
changes a lot under AMR), ever puts a live cell this close to zero lapse. If
it does and simply got lucky, this bug could be latent in the validated
production result too -- worth a deliberate check, not an assumption either
way.

## 7. Your task

1. **Confirm the mechanism directly.** Instrument `GetGRFastMagnetosonicSpeeds`
   (or a standalone unit test, matching the pattern of the existing
   `grad_ap6_0` regression test in `tst/unit/test_cfc_puncture.cpp`) to show
   `ialpha`/`lambda_p`/`lambda_m` actually going non-finite at the flagged
   cell's `alpha` value. Don't skip this even though the case above is
   strong -- this investigation's own history (Sec 5's exclusions) is full of
   plausible-looking mechanisms that turned out wrong on direct check.
2. **Audit exhaustively for every unguarded `/alpha` in the reachable code
   path**, not just the four/five sites found by one grep pass here (Sec 1).
   Check `llf_dyn_grmhd.hpp` directly instead of relying on the incomplete
   grep result noted there.
3. **Design and apply a fix.** Candidates, evaluate rather than pick blindly:
   floor `alpha` to some small positive epsilon before any division (mirrors
   `alpha_floor`'s existing intent, but note `NANCASCADE_HANDOFF.md` Sec 16.3
   already found that floors applied in the wrong place/time can be
   worthless -- make sure this floor actually runs before the division it's
   meant to protect, unlike that prior mistake); or special-case near-zero
   `alpha` to return a causal/light-speed wave-speed bound directly (common
   in other GRMHD codes for exactly this degenerate-lapse situation); or
   something else. Check whether the codebase has an existing convention
   (grep for other `alpha_floor`-style guards near metric consumption) before
   inventing a new one.
4. **Validate the fix**, in this order (cheap first):
   - New/extended unit test asserting finiteness near `alpha=0` (Sec 7.1).
   - Rerun the exact failing reproducer (`cfc_tde_wd_imbh_amr.athinput`, 1
     node/12 ranks CPU debug queue -- reuse the `tde_tracker_cpu_bracket_v1`/
     `tde_tracker_cpu_excise_v1` batch.t/parfile.t templates, cheap and fast)
     with BOTH `excise=false` and `excise=true` -- both should now run past
     cycle 1 cleanly.
   - Confirm no regression on `tde_prod_fixed_v3`'s configuration: rerun it
     (or at least enough cycles to cover its periapsis passage) and confirm
     bit-identical or negligibly-different results, not just "still 0 NaN" --
     a wave-speed-formula change that never actually triggers differently
     should reproduce the same trajectory/timestep history exactly.
   - Since this touches shared Z4c-reachable code: run at least one existing
     Z4c dyn_grmhd smoke/regression test (check `tst/test_suite/dyngrmhd/`
     for what already exists) to confirm no behavioural change there either.
5. Once validated, this can go into `proj/tde` alongside (or ahead of, your
   call) the four already-implemented-but-uncommitted follow-up items from
   `src/cfc/NANCASCADE_FOLLOWUP_SESSION_PROMPT.md` -- item 2 in particular
   (the AMR refinement-criterion fix) is still blocked on this bug being
   fixed, since its own smoke test can't get past cycle 1 without it.

## 8. Constraints and traps (carried over from this whole investigation)

- Never modify `tde_elliptic_tracker_nexteval_v2/` or its checkpoints --
  `tde_prod_fixed_v3` and most of this project's diagnostic runs restart from
  its `.rst` files via symlink; deleting or altering it breaks their
  provenance. Restart via a fresh case dir with the parent `.rst` symlinked,
  same pattern as every run cited above.
- Build on aurora-uan-0007/0008 only. `MPIR_CVAR_CH4_IPC_GPU_MAX_CACHE_ENTRIES=1024`
  if GPU is involved. `nlim` is an absolute cycle target, not "N more from
  restart."
- `<cfc> src_nan_check=1` is cheap and was the tool that actually cracked
  this -- use it on any new diagnostic run. `<cfc> solve_interval` defaults
  to 0 (= every cycle checked), no need to touch it for a short debug-queue
  run.
- CPU debug-queue jobs are cheap and fast (minutes, not a queue slot on
  next-eval) -- prefer them for this kind of unit-level numerics bug over
  GPU/production-scale runs, until the fix needs production-scale
  confirmation.
- Do not re-litigate the SetNeighbors or original-cascade diagnoses --
  both are closed (`SETNEIGHBORS_HANDOFF.md`, `NANCASCADE_HANDOFF.md`
  Sec 17-21). This bug is additional to, not a reopening of, either.
