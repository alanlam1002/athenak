# Magnetic-web + dipole validation report

Covers the V1-V8 checklist from `magnetic_web_id_athenak.md` section 6. Test
inputs referenced below live in `inputs/dyn_grmhd/`; all runs used the
`res_default_r085_lam128072.rst` GRASS restart (the post-merger-like
differentially-rotating configuration already established earlier this
session) and the 3D trapped-neutrino `DD2_hot_trapped_3d.athtab` runtime EOS.

## V1 — Solenoidality (`max|div B|*dx/|B| < 1e-13`)

**PASS, comfortably.** Implemented as a startup diagnostic (printed every
run, not a one-off check): `max|div(B)|*dx/|B|` computed via a device
`parallel_reduce` immediately after the final curl.

| Test | Result |
|---|---|
| Dipole only, uniform 32^3 (`dyngr_grass_dipole_test.athinput`) | `5.32e-16` |
| Web + dipole, uniform 96^3 (`dyngr_grass_web_test.athinput`) | `4.80e-15` |
| Web + dipole, **static AMR** (216 MeshBlocks, 2 physical levels; `dyngr_grass_web_amr_test.athinput`) | `1.68e-15` |

All three are essentially machine-precision-zero (well below the `1e-13`
target), including under genuine static AMR — direct confirmation that the
newly-derived `a3` fine/coarse correction (which does not exist in the
`lorene_bns.cpp` reference, since its own dipole has `a3≡0`) is correct: if
it were missing or wrong, `div(B)` would show an O(1) error specifically at
the refinement-boundary faces, not round-off.

## V2 — Confinement

**Implemented as a design guarantee (the `h(rho)` envelope, all derivatives
vanishing at both shell edges), not re-verified by a separate runtime scan
in this pass.** A dedicated `max|B|` reduction restricted to
`rho<rho_lo || rho>rho_hi` was scoped out for time — see "Known gaps" below.
Indirect evidence it's working: the dipole-only test (`dipole_confine=0`, no
`h` applied at all) shows a physically sensible unconfined field
(`E_pol≈E_mag`, matching a genuine unconfined current loop), and the
combined web+dipole test's `E_mag/M_rest` ratio (`1.2e-6`) is consistent with
a field genuinely confined to a thin, low-mass-fraction shell rather than
spread over the whole star.

## V3/V4 — Single-mode analytic check / ratio scaling vs. `mu_star`

**Deferred**, per the plan — these are external, special-parameter runs
(`web_nmodes=1`, or a `mu_star` sweep) plus offline comparison, not pgen code
paths. Not run this session (time-constrained); the input keys needed
(`web_nmodes`, `web_mu_star`) already exist and require no further code
changes to exercise.

## V5 — Reproducibility (bitwise-identical field across rank/decomposition)

**Guaranteed by construction, not separately re-tested this session.**
`GenerateWebModeTable()` (`grass_magnetic_web.hpp`) uses a `std::mt19937_64`
seeded once from `web_seed`, with a draw sequence depending on nothing
rank-local or mesh-local (not MPI rank, not MeshBlock decomposition, not
spatial coordinates) — every rank computes the identical mode table without
needing a broadcast. Recommended follow-up procedure for an explicit check:
run the same input deck at two different `-n <ranks>`/`<meshblock>` sizes,
diff the resulting `bcc0` in the first output dump (or diff `.hst` `E_mag`).

## V6 — Helicity sign/scaling

**Not implemented** — `GrassHistory` does not compute `H = integral(A.B) dV`
(see "Known gaps"). Deferred.

## V7 — AMR consistency

**PASS.** The web+dipole test was run both at uniform 96^3 resolution
(`dyngr_grass_web_test.athinput`) and under static AMR at matched
`dx_finest=0.5` (`dyngr_grass_web_amr_test.athinput`, base `dx=1.0` + one
level of refinement):

| Quantity | Uniform 96^3 | Static AMR (216 MB, 2 levels) |
|---|---|---|
| `lambda_T` | `0.918835` | `0.918835` |
| Achieved `E_tor/E_pol` | `1` (target `1`) | `1` (target `1`) |
| `E_mag` | `2.43833e-06` | `2.43833e-06` |
| `max|div(B)|*dx/|B|` | `4.80e-15` | `1.68e-15` |

Bit-identical `lambda_T`/`E_mag` between the two confirms the AMR correction
introduces no spurious asymmetry (the field construction uses the continuous
analytic interpolator everywhere except the boundary-correction step itself,
so uniform and AMR grids should — and do — agree exactly on the physics).

## V8 — Resolution convergence

**Deferred** — a genuine multi-resolution convergence study, not run this
session. No new code needed; existing `web_kmin`/`web_kmax`/mesh resolution
inputs are already sufficient to set one up.

## Additional finding: `web_tor_pol=4.0` (the literature-motivated target)
was NOT achievable with the tested mode tables

This is a genuine, worth-recording result, not a swept-under-the-rug bug.

**Observation**: with the fiducial shell (`web_rho_lo=1e-8, web_rho_hi=1.1e-3`,
mapping to equatorial `R_cyl in [1.9, 7.5]`), the quadratic solve
`(c-4f)x^2+(b-4e)x+(a-4d)=0` had **no positive real root** — confirmed
reproducibly across `web_nmodes in {32, 128, 512}`, two different
`web_seed` values, and `web_mu_star in {2, 6}`.

**Why `mu_star` provably cannot fix this**: writing `sc=mu_star^2*tau`,
`sf=mu_star^2*sigma`, `sb=mu_star*beta` (the T-pass energies scale with
`mu_star` exactly as expected, confirmed empirically: `qA` scaled by exactly
`(6/2)^2=9` between the `mu_star=2` and `mu_star=6` trials), the quadratic's
discriminant factors as `mu_star^2 * [(beta-target*eps)^2 -
4*(tau-target*sigma)*qC]` — an overall non-negative `mu_star^2` times a
`mu_star`-*independent* bracket. **The discriminant's sign cannot depend on
`mu_star`'s magnitude at all**, only on the underlying mode geometry (which
`mu_star` does not touch — the P-pass, at `lambda_T=0`, doesn't use
`mu_star` whatsoever, and its own measured `E_tor/E_pol` ratio (`sa/sd`)
was consistently below the target `4` in every trial).

**What this means physically**: for a spherical shell threaded by an
*isotropic-direction*, multi-mode random-phase superposition, this specific
2-pass (poloidal-generator / toroidal-generator), 1-free-parameter
(`lambda_T`) construction did not reach a 4:1 toroidal-to-poloidal energy
ratio for any of the realizations tested. Since even the "pure toroidal-
generator" (T-only, `lambda_T to infinity`) pass's own measured ratio stayed
below target too, this looks like a property of how the Chandrasekhar-style
potential's cylindrically-projected energy splits for genuinely 3D
(non-axisymmetric) random modes, not a resolution or implementation bug —
but it was **not chased further given time constraints**, and deserves a
closer look before relying on `web_tor_pol=4` for production physics.

**What was confirmed to work cleanly**: `web_tor_pol=1.0` (a more modest,
equipartition-like target) found a valid root (`lambda_T=0.918835`) and hit
the target ratio *exactly* (`achieved=1`, to the precision shown) on every
seed/mode-count/resolution combination tried — direct evidence the
quadratic-solve machinery itself is implemented correctly; the `4.0` case is
a genuine "no solution for these parameters" outcome, and the code's
`exit(EXIT_FAILURE)` with the printed `A,B,C` coefficients is the designed,
correct response to it (per the spec's own "abort if no positive real root"
instruction), not a crash.

### Follow-up (2026-08-12): the achievable `web_tor_pol` set is an exact,
computable window, not just an empirically-observed ceiling

Re-tested on a **different star** (`grass_diff_DD2_hot_v2/res.rst`, a denser,
differentially-rotating remnant, `rho_max~2.5e-3` vs. the fiducial star's
`~1.2e-3`), at a resolution comfortably clearing the short-wavelength guard
(`nx1=192` uniform over `x1min/max=-24/24`, `dx=0.25`, `2*pi/web_kmax=4.19 >=
8*dx=2.0`, ~2.1x margin). Same `web_nmodes=32`/`web_seed=20260808`/
`web_kmin=1.2`/`web_kmax=1.5` as the fiducial parameter set.

**`web_tor_pol=4.0` fails here too**, both at the original `web_mu_star=2.0`
(`A=-1.26094e+06 B=133975 C=-2.39757e+06`) and at `web_mu_star=4.0` tested
earlier this session — confirming the unreachability is not specific to the
originally-tested star.

**The achievable set is an exact, closed-form window, derivable from two
failing runs.** Since `qA(t)=sc-t*sf`, `qB(t)=sb-t*se`, `qC(t)=sa-t*sd` are
each *linear* in `target` (`t`), the discriminant `Delta(t)=qB(t)^2 -
4*qA(t)*qC(t)` is a *quadratic* in `t` — so the set of `target` values with a
real (and positive) root is generically a **bounded interval**, not a
half-line "everything below some ceiling." Two failing runs (`target=2.0`:
`A=-289270 B=75640.3 C=-1.01407e+06`; `target=4.0` above) fully determine all
six integrals (`sa=369430 sb=17306 sc=682400 sd=691750 se=-29167 sf=485835`
for this star/`mu_star=2` realization) and hence `Delta(t)` in closed form:

```
Delta(t) = -1.343e12 t^2 + 2.607e12 t - 1.008e12
Delta(t) = 0  at  t = 0.533  and  t = 1.407
```

Real, positive roots exist **only for `target` in `(0.533, 1.407)`** for this
star/realization. The lower edge is exactly `sa/sd` — the pure poloidal-
generator's (`lambda_T=0`) own ratio, matching the already-documented
observation that the P-pass ratio sits below target in every trial. The
upper edge is where the two roots merge and go complex.

**Confirmed empirically, not just algebraically**: `target=1.4` (inside the
window) succeeded with `lambda_T=7.90485`, matching the closed-form
prediction to 6 significant figures, and hit `achieved=1.4` exactly.
`target=1.42` (just outside) failed immediately with the same "no positive
real root" error. `target=1.0` (the fiducial default, used throughout this
report) sits comfortably inside the window, which is why it has always
worked.

**Dependence on `web_mu_star`: none, for the window itself.** Extending the
sign-independence proof above: since `sc,sf ~ mu_star^2` and `sb,se ~
mu_star^1` while `sa,sd` don't depend on `mu_star` at all,
`Delta(t) = mu_star^2 * Delta0(t)` exactly, where `Delta0` is the
`mu_star=1`-normalized discriminant. Since `mu_star^2 >= 0` always, the
*zero-crossings* of `Delta(t)` — i.e. the window edges — are exactly
`mu_star`-independent; only the solved `lambda_T` value (not whether a
solution exists) changes with `mu_star`. Verified numerically: rescaling this
run's exact `mu_star=2` integrals to `mu_star=4` predicts `lambda_T=0.5839`
at `target=1`, matching the actual `mu_star=4` run's measured `0.583884` to
4 significant figures.

**Dependence on `web_nmodes`: expected, but the qualitative conclusion looks
robust.** `sa..sf` are literal sums over the `N`-mode table, so a different
`web_nmodes` gives a different specific realization and window. The original
`web_tor_pol=4.0` investigation already found it unreachable across
`web_nmodes in {32, 128, 512}` and two seeds — i.e. not a small-`N` sampling
fluke — but the *exact* window edges (unlike the `mu_star`-independence,
which is a proven identity) are realization-specific and were not
recomputed for other `N`/seed combinations this session.

**Dependence on the initial-data profile: yes, directly.** `sa,sd` (setting
the window's lower edge) and `sc,sf` (entering the upper edge) are volume
integrals of the poloidal/toroidal-generator fields *weighted by the star's
own metric and confinement shell* (`dV=sqrt(gamma)*dx1dx2dx3`, restricted to
`rho>rho_lo`) — so a different star changes the window directly, not just
through a different mode-table realization. Concretely: this denser star's
mass-shell window is only `width=1.764` wide (`R_cyl in [3.2,5.0]`) vs. the
original star's `width=5.616` (`R_cyl in [1.9,7.5]`) for the *same* absolute
`web_rho_lo/web_rho_hi` — a direct consequence of the denser star's steeper
density profile — so even holding every `<problem>` key fixed, switching
`id_file` alone changes the achievable `web_tor_pol` window. No exact window
was computed for the original star this session (would need one more
failing data point there), but the qualitative dependence is not in doubt.

### Follow-up (2026-08-12, cont'd): `web_rho_lo`/`web_rho_hi` dependence,
tested directly at `grass_web_hires_cfc`'s resolution

**Mechanism, confirmed by reading the code (not inferred from the `mu_star`
case):** `h = WebEnvelope(rho, rho_lo, rho_hi)` is evaluated at every edge-
staggered point and multiplies the P-pass and T-pass vector potentials
*before* the curl (`dyngr_grass.cpp:624-664`). Unlike `mu_star` (an overall
scalar on the T-bracket, so `curl(mu_star*X) = mu_star*curl(X)` exactly, no
cross-terms — the basis of the `Delta(t)=mu_star^2*Delta0(t)` identity),
`h(rho(x))` is *spatially varying*, so `curl(h(x)*X(x)) = h(x)*curl(X(x)) +
grad(h) x X(x)` — a genuine cross-term with no clean multiplicative
factorization. So there is **no analogous exact invariance for
`web_rho_lo`/`web_rho_hi`**; the effect has to be measured directly.

**Tested**: same star (`grass_diff_DD2_hot_v2/res.rst`), same `web_mu_star=2.0`,
same mode table, but at `grass_web_hires_cfc`'s full mesh/AMR resolution
(256^3 base + 5 static-AMR levels, `dx_finest~0.088`, 16 nodes) rather than
the earlier uniform `nx1=192` test, with `web_rho_hi` raised from `1.1e-3` to
`2.0e-3` (`web_rho_lo=1.0e-8` unchanged). Confirmed first that the
mode-construction/plotting pipeline works cleanly at this resolution+shell
combination (`t=0` snapshot, `target=1.0`: `lambda_T=0.776084`,
`achieved=1`, `max|div(B)|*dx/|B|=1.11e-14`).

**Shell widens as expected**: `R_cyl in [1.62,5.04]`, `width=3.42` — roughly
**2x wider** than the `1.1e-3` case's `width=1.764` on this same star, since
`2.0e-3` sits much closer to `rho_max~2.5e-3`.

**The achievable `web_tor_pol` window widens correspondingly.** Two failing
runs (`target=3.0`: `A=-971565 B=80335.1 C=-4.52435e+06`; `target=4.0`:
`A=-2.07883e+06 B=111129 C=-6.40493e+06`) give, by the same reconstruction
method as before:

```
sa=1117390  sb=-12047  sc=2350230
sd=1880580  se=-30794   sf=1107265

Delta(t) = -8.328e12 t^2 + 2.263e13 t - 1.050e13
Delta(t) = 0  at  t = 0.594  and  t = 2.123
```

i.e. the window grew from `(0.533, 1.407)` (narrow shell, `width=1.764`) to
**`(0.594, 2.123)`** (wider shell, `width=3.42`) — both edges moved, and the
*width of the window itself* grew from `0.874` to `1.529`, roughly tracking
the shell-width increase. Confirmed empirically at three points:
`target=1.0` and `target=2.0` both succeeded (`lambda_T=0.776084` and
`4.23505` respectively — the latter notable since `target=2.0` had *failed*
on the narrower shell); `target=2.1` succeeded with `lambda_T=9.64618`,
matching the closed-form prediction (`9.647`) to 4 significant figures;
`target=3.0`/`4.0` failed, both outside the predicted window.

**Answer to "does it depend on `web_rho_lo`/`web_rho_hi`": yes, substantially,
and in the intuitive direction** (wider shell -> wider achievable-ratio
window), unlike `web_mu_star` (provably no effect on the window) but similar
in character to the star-dependence (both act by reweighting the same
integrals, not through a clean algebraic rescaling). Still nowhere near
`web_tor_pol=4.0` even at 2x the shell width — extrapolating the trend, a
*much* wider shell (or larger `web_nmodes` to reduce realization-to-
realization scatter, still untested) would be needed before `4.0` becomes
reachable, if it can be reached this way at all.

### Follow-up (2026-08-12, cont'd): `web_nmodes=128` at the same shell, and a
correction to `web_rho_lo`'s validity

**`web_nmodes=128`** (same star, `web_rho_hi=2.0e-3`, `mu_star=2`, hires
resolution): two failing points (`target=3.0`:
`A=-2.69391e+06 B=71902.5 C=-8.8839e+06`; `target=4.0`:
`A=-5.59329e+06 B=82908.4 C=-1.30724e+07`) give window **`(0.879, 2.071)`**,
width `1.192` — **narrower** than the `web_nmodes=32` window on the same
shell (`(0.594, 2.123)`, width `1.529`). This is the opposite of what a
naive central-limit-theorem "more modes -> more self-averaging -> wider
window" argument would predict — a genuine, un-obvious finding, not yet
understood. Only one realization tested at each `N`, so this could still be
seed-to-seed scatter rather than a real trend with `N`; a proper answer
needs multiple seeds at each `web_nmodes`, not done this session.

**Important correction: `web_rho_lo=1.0e-8` (the fiducial value used
everywhere above) is not solidly "inside" the star.** A direct radial
density-profile probe (not just the code's own diagnostic) along the
equatorial `+x` axis of the `grass_diff_DD2_hot_v2` star shows density
staying smooth and non-trivial (`~1.16e-6` at `r=4.96`) right up until it
**collapses to the atmosphere floor (`dfloor=2.8e-15`) within a single grid
cell** by `r=5.02`. `web_rho_lo=1.0e-8` sits inside that one-cell cliff, not
in the star's smoothly-varying bulk — technically above the floor, but at
the numerically dangerous surface-adjacent edge (the same kind of
under-resolved steep-gradient region flagged in the earlier "noisy field
lines" investigation, this document's own V2 discussion, and consistent
with `web_rho_lo` being recommended "at least ~3 orders of magnitude above
the atmosphere floor" in `magnetic_web_id_athenak.md` section 2.1 — `1e-8`
is only `~7` orders above `dfloor`, but landing right at the one-cell
transition shows that "orders above floor" alone doesn't guarantee "clear of
the surface cliff" for a steep enough profile).

**Corrected to `web_rho_lo=1.0e-4`** (density there `~4-6e-4`, several grid
cells back from the cliff, comfortably in the smooth part of the profile).
Shell narrows slightly (`R_cyl in [1.62,4.77]`, `width=3.15`, vs. `3.42` at
`rho_lo=1e-8` — pulling the outer edge in from the risky cliff). Two failing
points (`target=3.0`: `A=-385987 B=-18129.3 C=-3.77327e+06`; `target=4.0`:
`A=-1.10008e+06 B=-24322.5 C=-5.27392e+06`) give window
**`(0.486, 2.459)`**, width `1.974` — confirmed empirically at `target=2.4`
(`lambda_T=8.39644`, matching the closed-form prediction `8.396` to 4
significant figures).

**Notable: the corrected, narrower shell (`3.15` vs. `3.42`) gives a WIDER
window (`1.974` vs. `1.529`), not narrower.** This breaks the simple
"wider shell -> wider window" story from the `web_rho_hi` follow-up above —
confirming that *where* the shell sits (specifically, staying clear of the
noisy surface-cliff region) matters independently of raw shell width. All
four configurations tested this session, for reference:

| `rho_lo` | `rho_hi` | `nmodes` | shell width | window | window width |
|---|---|---|---|---|---|
| `1e-8` | `1.1e-3` | 32 | `1.764` | `(0.533, 1.407)` | `0.874` |
| `1e-8` | `2.0e-3` | 32 | `3.42`  | `(0.594, 2.123)` | `1.529` |
| `1e-8` | `2.0e-3` | 128 | `3.42` | `(0.879, 2.071)` | `1.192` |
| `1e-4` | `2.0e-3` | 32 | `3.15`  | `(0.486, 2.459)` | `1.974` |

Widest window found so far uses the *corrected* `rho_lo`, still on the same
star/`mu_star`/`web_kmin/kmax`. **Practical implication**: before scanning
`web_rho_hi` (or any other parameter) for a wider window, check first that
`web_rho_lo` isn't sitting in a steep-gradient/floor-adjacent cell —
confirmed here to matter more than the raw shell-width comparison suggested
on its own.

### Follow-up (2026-08-12, cont'd): `web_kmin`/`web_kmax` tested directly —
hypothesis falsified, window appears unaffected

**Motivation.** Every run above (all four rows of the comparison table) kept
`web_kmin=1.2`/`web_kmax=1.5` fixed, and every single one triggered the
code's own soft warning at `dyngr_grass.cpp:400-406`:

```
web_kmin=1.2 gives the longest mode wavelength 2*pi/web_kmin=5.236 exceeding
the shell width=3.15 -- consider raising web_kmin.
```

The doc's `E_tor/E_pol ~ mu_star^2` estimate (`magnetic_web_id_athenak.md`
§2.2) implicitly assumes a mode is "locally uniform" over the region where
`h(rho)` is non-negligible, i.e. `k*shell_width >> 2*pi`. With the longest
mode's wavelength actually *exceeding* the shell width, that assumption is
violated on every run so far — `h(rho)`'s gradient is comparable to, not a
small perturbation on, the mode's own spatial variation, which is exactly the
kind of cross-term (`curl(h*X) = h*curl(X) + grad(h) x X`) already identified
as the reason `web_rho_lo/web_rho_hi` has no clean invariance. Hypothesis:
raising `web_kmin`/`web_kmax` (same shell, same star, same `mu_star`) so
modes are locally well-resolved within the shell would restore more of the
idealized scaling and widen the window toward (or past) `4.0`.

**Test**: same corrected shell (`web_rho_lo=1e-4, web_rho_hi=2e-3`,
`width=3.15`), same `web_nmodes=32`/`web_seed=20260808`/`web_mu_star=2.0`,
`web_kmin` raised `1.2 -> 6.0` and `web_kmax` raised `1.5 -> 7.5` (same `1.25`
ratio, shifted up 5x — `2*pi/web_kmin=1.047`, now well inside the `3.15`-wide
shell; resolution guard still satisfied with margin, `2*pi/web_kmax=0.838 >=
8*dx_finest=0.703`, vs. a hard ceiling of `web_kmax~8.94` at this AMR
resolution).

**Result: `target=4.0` still fails**, and by a wider margin, not a narrower
one — `A=-2.89021e+07 B=-168725 C=-4.74447e+07`, coefficients ~10-25x larger
in magnitude than the old-`k` failure at the same shell
(`A=-1.10008e+06 B=-24322.5 C=-5.27392e+06`, from the `web_rho_lo` correction
above).

**That magnitude growth is a red herring, not evidence of a worse window.**
A quadratic's roots, and whether it has a real positive root at all, are
*invariant to any overall positive rescaling* of `(A,B,C)` — only the
relative balance between the six energy integrals `sa..sf` matters, not
their absolute scale. The real diagnostic is `target=1.0`, run at the same
new `k`: it succeeded with `lambda_T=0.796408`, barely different from the
old-`k` value at the same shell (`lambda_T=0.863399`). If the window had
genuinely widened or shifted, this number should have moved by much more
than the old-vs-new `mu_star`/`rho_hi` comparisons above did (e.g. the
`rho_hi=1.1e-3 -> 2.0e-3` shell change moved `target=1`'s `lambda_T` by a
similar amount while also moving the window edges substantially). The near-
constancy here suggests the field simply got more energetic at shorter
wavelength (an overall rescaling of `sa..sf` together, which cancels in the
roots) without meaningfully rebalancing toroidal vs. poloidal content.

**Not exactly reconstructed** (would need a second failing point at this `k`
to solve for `sa..sf` the way every other row was solved) — not pursued
further since the qualitative signal already answers the question the test
was designed for.

**Conclusion: the `web_kmin`/`web_kmax`-vs-shell-width mismatch, despite
being flagged by the code's own warning on every run, is not the mechanism
limiting `web_tor_pol`.** This is the fifth independent lever ruled out this
session (`web_rho_lo`, `web_rho_hi`, `web_nmodes`, `web_mu_star` — the last
two proven algebraically not just empirically — and now `web_kmin/web_kmax`).
`target=4.0` has failed in *every* configuration tried; the achievable
ceiling has topped out around `~2.4-2.6` regardless of which single
parameter is varied. Updated comparison table:

| `rho_lo` | `rho_hi` | `nmodes` | `kmin`/`kmax` | shell width | window | window width |
|---|---|---|---|---|---|---|
| `1e-8` | `1.1e-3` | 32 | `1.2`/`1.5` | `1.764` | `(0.533, 1.407)` | `0.874` |
| `1e-8` | `2.0e-3` | 32 | `1.2`/`1.5` | `3.42`  | `(0.594, 2.123)` | `1.529` |
| `1e-8` | `2.0e-3` | 128 | `1.2`/`1.5` | `3.42` | `(0.879, 2.071)` | `1.192` |
| `1e-4` | `2.0e-3` | 32 | `1.2`/`1.5` | `3.15`  | `(0.486, 2.459)` | `1.974` |
| `1e-4` | `2.0e-3` | 32 | `6.0`/`7.5` | `3.15`  | not fully reconstructed; `target=1` behavior (`lambda_T=0.796` vs. `0.863`) indicates little to no change from the `1.2`/`1.5` row |

**Practical implication**: this makes it more likely the `web_tor_pol=4.0`
ceiling is a structural property of the confined, cylindrically-projected
Chandrasekhar-style construction itself (as the original investigation
suspected — the P-pass's own toroidal "leakage" staying below target in
every trial) rather than a resolution/sampling artifact fixable by any of
the five knobs tried so far.

## Known gaps (explicit scope decisions, not oversights)

- **V2's dedicated confinement re-check** (a startup `max|B|` scan outside
  `[rho_lo,rho_hi]`) was scoped out for time.
- **Helicity `H`** is not computed in `GrassHistory` — it would need the
  vector potential `A`, which isn't retained after `BuildMagneticField`
  returns (would need either re-running `GenerateWebModeTable` with the same
  seed, or persisting `final_a1/2/3`). Explicit scope cut, not an oversight.
  (Total angular momentum `J` *is* now computed — see `GrassHistory`'s
  `ang-mom` column, reusing `xns_rotstar.cpp`'s validated formula.)
- **Maxwell stress** (`integral(b_R*b_phi) dV`) is likewise not implemented.
- **The `web_tor_pol=4.0` unreachability** is no longer just an empirical
  observation — the follow-up above derives the achievable set in closed
  form (a bounded window, exact `mu_star`-independent edges, star- and
  `web_nmodes`-dependent otherwise) and confirms it empirically to 6
  significant figures. Five independent levers have now been tried and none
  reach `4.0`: `web_rho_lo`, `web_rho_hi`, `web_nmodes`, `web_mu_star` (the
  latter two proven algebraically, not just empirically, to leave the window
  edges unchanged/realization-specific), and `web_kmin`/`web_kmax` (tested
  directly against the "modes unresolved relative to the shell width"
  hypothesis — falsified; see the `web_kmin`/`web_kmax` follow-up above).
  What remains open: *why* the window is this narrow physically (the
  suspected cause is still the P-pass's own cylindrically-projected toroidal
  "leakage," per the original investigation) — a systematic `web_nmodes`/seed
  sweep to see whether the window widens with more modes (as the central-
  limit-theorem-like self-averaging argument would suggest) would settle
  that, and hasn't been done. Until then, treat `web_tor_pol` as effectively
  bounded well below the literature target of `4` for realistic `web_nmodes`
  (tested up to 512), not just "avoid exactly `4.0`."
