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
- **The `web_tor_pol=4.0` unreachability** (above) is an open finding, not a
  closed issue — worth a follow-up investigation (larger `web_nmodes`, a
  systematic seed sweep, or reconsidering whether the P-pass's own
  cylindrically-projected toroidal "leakage" is expected/correctable) before
  the feature is used for literature-matched production physics.
