# Grey photon M1 development notes

This file tracks the design and implementation status of the grey (single-group)
photon path through `src/radiation_m1/` — the two-moment M1 radiation-transport
module originally built for neutrino transport (bns-nurates tables) on branch
`photon_m1`. It is meant to be kept up to date as the module evolves — treat it as
the running log of *why* the code looks the way it does, not just *what* it does.

## Goal

Extend the existing GR two-moment M1 infrastructure (fluxes, closures, implicit
source solver, GR sources, AMR+MPI, `T_dd` backreaction — all already working for
neutrinos) with a grey photon opacity path, coupled to GRMHD, for eventual use in
radiation-pressure-supported systems (the `dyngr_bhstar.cpp` quasi-star/BH-star pgen
is the target application).

## Status (as of this note)

**Implemented and tested:**
- Grey photon opacities for `Primitive::IdealGas` (fast, code-units-only path,
  `CalcOpacityPhotons_IdealGas_`) and `Primitive::EOSCompOSE` (cgs-unit-aware,
  supports constant or Kramers/power-law opacities via the shared
  `OpacityFunction` helper also used by the non-M1 `radiation` module,
  `CalcOpacityPhotons_<EOSPolicy,ErrorPolicy>`), both in
  `radiation_m1_calc_opacities_photons.cpp`.
- Compton energy exchange, **`Primitive::IdealGas` path only** (scope decision —
  see "Compton implementation" below). The EOSCompOSE path hard-errors if
  `compton=true` rather than silently ignoring it.
- Four single-zone (homogeneous, 0-D-equivalent) tests, all passing against
  closed-form analytic solutions — see "The single-zone test" below: LTE
  relaxation via true Planck absorption/emission (`rad_m1_photon_singlezone`,
  pre-existing), via Compton alone (`rad_m1_photon_compton_singlezone`), a
  scattering-only null test confirming `E` stays pinned at its floor with no
  energy-exchange channel (`rad_m1_photon_scattering_singlezone`), and a
  finite-`v_x` boosted-LTE test validating the lab-frame↔comoving-frame boost
  in the source solver against an independently-derived closed-form
  equilibrium (`rad_m1_photon_vx_singlezone`).

**In progress, crash fixed, one accuracy question still open**: the
optically-thick diffusion test (item 2 below) works well at `kappa_s=5`
(sub-1% error, cross-validated against the reference discrete-ordinate module)
after fixing a "zero-flux spin-up transient" IC issue. That same corrected IC
used to make `E` collapse to the floor everywhere at `kappa_s=200` (the
stiff/implicit source-solver regime); this has been root-caused and fixed (a
latent Newton-solver/floor interaction — see `apply_floor()` in
`radiation_m1_helpers.hpp` and item 2's writeup). What remains open: `E` at
`kappa_s=200` still spreads ~27-34% faster than the analytic diffusion law
predicts, the same discrepancy that originally motivated the cross-check
against the discrete-ordinate module — not yet explained.

**Not yet done** (see "Stage 2 plan" below): free-streaming and
radiation-pressure-backreaction transport tests, and CI wiring. Kramers/
`power_opacity` and the EOSCompOSE branch have no dedicated test yet either
(deprioritized alongside EOSCompOSE Compton — see below).

**Scope decision — IdealGas only, for now:** Compton and the newer single-zone
tests all target `Primitive::IdealGas`, not `Primitive::EOSCompOSE`. This was an
explicit simplification during Stage 1 to keep the physics-derivation and
verification loop fast; EOSCompOSE support (including Compton, Kramers-opacity
testing, and cgs-unit round-tripping) is deferred, not abandoned.

## Bugs found and fixed along the way

Two pre-existing, unrelated-to-photons bugs blocked this work and were fixed:

1. **`radiation_m1.hpp:25` used `#ifdef ENABLE_NURATES` instead of
   `#if ENABLE_NURATES`.** `config.hpp` always `#define`s `ENABLE_NURATES` (to `0`
   or `1`), so `#ifdef` was always true regardless of `Athena_ENABLE_NURATES`,
   making the `bns_nurates` submodule a hard compile-time requirement even when
   nurates was off. Every other `ENABLE_NURATES` guard in the codebase correctly
   uses `#if`; this one was the sole outlier. Fixed to match.
2. **`athena_read.tab()` (vis/python) misparses `.tab` output from any run with a
   degenerate mesh dimension** (`nx2=1` and/or `nx3=1` — i.e. any single-zone-style
   test by construction). AthenaK's tab writer lists the full generic coordinate
   column set (`gid i x1v j x2v k x3v`) in the header comment but omits the
   columns for inactive dimensions from the actual data rows; `athena_read.tab()`
   assumes header and data columns line up 1:1 and throws an `IndexError`. This is
   a general AthenaK tab-writer/reader inconsistency (`src/outputs/`, not
   photon-M1-specific) — out of scope to fix at the writer level here. Worked
   around locally with `inputs/tests/m1_tab_utils.py`, a minimal reader that
   matches variable columns by counting from the *end* of each row (physical
   variables are always the trailing columns; only leading coordinate columns get
   dropped). `check_rad_m1_photon_singlezone.py` — which, notably, had never
   actually been run before this — was fixed to use it.

## Compton implementation

**Physics.** Grey (Kompaneets zeroth-order) Compton energy exchange: electron
scattering shifts photon energy by a fraction `~4 k_B T_gas / (m_e c^2)` per
scattering, driving the comoving-frame radiation energy density `J` toward
`a_rad T_gas^4` on a rate `kappa_s * 4 k_B T_gas / (m_e c^2)` — the same
equilibrium blackbody target as true absorption, just mediated by a different
microphysical channel.

**Why it slots into the existing solver for free.** M1's implicit source term
(`radiation_m1_helpers.hpp::calc_rad_sources`) is
`S_a = (eta - kabs*J)*u_a - (kabs+kscat)*H_a` — **linear in `J`**. The Compton rate
above is *also* linear in `J` (for `J`/`T_gas` frozen at the start of the
timestep, exactly like the existing Planck channel already is — opacities are
computed once per timestep, not re-evaluated per RK substage or per Newton
iteration). So Compton was added as an **extra effective (`eta`, `kabs`) pair**,
folded directly into `eta_1_`/`abs_1_` in `CalcOpacityPhotons_IdealGas_`, on top of
the existing Planck channel:
```
sigma_compton = kappa_s * rho * 4 * T_gas * inv_t_electron   [inv_t_electron = k_B/(m_e c^2), in code units]
eta_1  += sigma_compton * a_rad * T_gas^4
abs_1  += sigma_compton
scat_1  = kappa_s + kappa_a   (unchanged — kappa_s still contributes its ordinary elastic/flux-damping role too)
```
This required **zero changes** to the Newton solver, its hand-derived analytic
Jacobian, the closure, or `T_dd` backreaction — they all just see a larger
`eta_1_`/`abs_1_` and treat it identically to ordinary Planck absorption.

**Alternative considered and rejected:** the sibling (non-M1, discrete-ordinate)
`radiation` module's Compton implementation (`radiation_source.cpp`) instead
*jointly* solves for `T_gas` and `T_rad` within a single sub-step via a
closed-form quartic (`FourthPolyRoot`), because it's coupling per-angle
intensities directly. Embedding that level of self-consistency into M1 would mean
extending `source_jacobian`/the residual functor in `radiation_m1_roots_fns.hpp`
so `eta`/`kabs` depend on the current Newton iterate — a much more invasive change
to an already-delicate, hand-coded-Jacobian solver, for a nonlinearity that (given
opacities are already only updated once per timestep) isn't actually being
exploited by the rest of the module. Deferred; see `radiation_source.cpp:290-330`
if a future need for that level of fidelity arises.

**Validation:** `rad_m1_photon_compton_singlezone` (κ_p=κ_a=0, κ_s=1, so the
*only* coupling channel is Compton) reproduces the closed-form relaxation to
5×10⁻⁵ relative error at all times, shrinking as `t` grows (consistent with RK2
truncation error on a smooth exponential, not a mismatched equilibrium). See
"The single-zone test" below for why this is a meaningful check and not just a
fitted trend.

## The single-zone test: what it is, and why the result is trustworthy

Both single-zone tests use `nx1=5, nx2=1, nx3=1`, periodic BCs, and spatially
uniform initial data. Since every cell starts identical and periodic BCs cannot
introduce asymmetry, every cell stays identical for all time — no spatial gradient
ever appears. `backreact=false` additionally freezes `T_gas`. This collapses the
*entire* PDE system (fluxes, Eddington-factor closure, GR source terms) down to a
single, local ODE in time.

With the fluid static (`v=0`) and, by the same homogeneity argument, the flux
`F_i` staying exactly zero for all time, `J = E` and `H_a = 0` identically, so
`calc_rad_sources`'s `S_a = (eta - kabs*J)*u_a - (kabs+kscat)*H_a` reduces to:
```
dE/dt = eta - kabs*E
```
a linear ODE with the closed-form solution
```
E(t) = J_eq + (E0 - J_eq)*exp(-kabs*t),   J_eq = eta/kabs
```
This is checked *pointwise against the full trajectory*, not just the final
value, by re-deriving `J_eq` and the rate independently (in Python, replicating
`units.cpp`'s cgs↔code conversion by hand for the Compton test) and comparing
against every output time in `E(t)`. That the relative error *shrinks* as `t`
grows (rather than plateauing at some nonzero floor) is itself diagnostic: a wrong
equilibrium value would show up as a non-decaying offset, while what's observed is
exactly the RK2 O(dt²) truncation-error signature on a correctly-targeted
exponential.

**What this test does *not* validate**: fluxes, the Eddington-factor closure,
curvature/GR source terms, AMR, or MPI — all trivially zero/inactive here. It is
deliberately the smallest possible harness that isolates the matter-radiation
coupling term from the transport machinery. Stage 2 (below) is precisely about
exercising the parts this test cannot reach.

## Stage 1 — complete

- [x] Investigate Compton approach; user chose "operator-split, fold into existing
  eta/kabs" over "fully implicit, embedded in the Newton solve" (see above).
- [x] Implement Compton for the `IdealGas` opacity path; hard-error it on the
  EOSCompOSE path instead of silently ignoring `compton=true`.
- [x] Fix the `ENABLE_NURATES` `#ifdef`/`#if` bug blocking any nurates-free build.
- [x] Build on Sakura (`build_m1_sakura.sh`, modeled on `~/athenak_cfc/cfc_sakura.sh`).
- [x] Fix `athena_read.tab()` breakage for degenerate-dimension tab output
  (`inputs/tests/m1_tab_utils.py`); confirm the pre-existing single-zone LTE test
  still passes (no regression) with the fixed reader.
- [x] Add + validate `rad_m1_photon_compton_singlezone` (5×10⁻⁵ final relative
  error against the closed-form Compton relaxation).

## Stage 2 plan — exercise transport + real photon opacities together

Everything in Stage 1 is 0-D (homogeneous, source-term-only). Stage 2 is the
validation ladder that turns on fluxes, the closure, motion, and backreaction one
at a time, each isolating one new piece of physics, still scoped to
`Primitive::IdealGas` (per the Stage 1 scope decision above) unless noted.
Recommended order (cheapest / fewest new mechanisms first):

1. **Single-zone extensions (cheap, no new transport) — done**
   - [x] *Scattering-only null test* (`rad_m1_photon_scattering_singlezone`):
     `kappa_s=1, kappa_p=kappa_a=0, compton=false`. Verified `E(t)` stays
     **bit-for-bit identical** to `E(0)` (its floor value) for the entire run
     (max relative deviation exactly `0.0`) — confirms `kscat` never leaks into
     the `kabs*J` emission-driving term. Analytically expected: with
     `eta=kabs=0` exactly, `source_update_ll`'s non-stiff explicit branch
     (`radiation_m1_sources.hpp:154`) gives `Edot=0` identically (verified via
     Explore-agent code read before implementing — no division by `kabs`/`eta`
     anywhere in this path, so no risk of NaN/inf at exactly-zero opacities
     either).
   - [x] *Finite `v_x`* (`rad_m1_photon_vx_singlezone`, `v_x=0.3`, `W≈1.048`):
     still homogeneous (no reconstruction/Riemann solver involved) but now
     exercises the lab-frame↔comoving-frame boost inside the source solver
     (`u_u`, `W`, `proj_ud`). Checked the **final-time equilibrium** only (not
     the full trajectory, unlike the static tests — the transient would need
     re-deriving the intermediate-time boost algebra by hand, which wasn't
     independently verified) against a closed-form prediction derived from,
     and cross-checked line-by-line against,
     `assemble_rT`/`calc_J_from_rT`/`calc_H_from_rT`
     (`radiation_m1_helpers.hpp`) and the Eddington closure pinning `chi=1/3`
     exactly (`radiation_m1_closure.hpp:22-23`, which zeroes the anisotropic
     `F_aF_b/|F|^2` closure term completely — `helpers.hpp:216-229`):
     `E_eq = J_eq*(4W²-1)/3`, `F_x,eq = J_eq*(4/3)*W²*v_x`. Measured final
     relative error: `1.7×10⁻⁴` (`E`), `3.2×10⁻⁴` (`F_x`) — both within the
     `10⁻³` tolerance and consistent with RK2 truncation error at this `cfl`.
     First M1 test in the repo to output `rad_m1_F` (columns `Fx:0`/`Fy:0`/`Fz:0`,
     `basetype_output.cpp:736-753`).

2. **Optically-thick diffusion test (real opacities, first genuine transport
   test) — IN PROGRESS, crash fixed, one accuracy question open.** Real-physics analogue of the
   existing toy-opacity `rad_m1_diffusiontest.cpp` (which prescribes `D`
   directly via `ToyOpacityModel::Diffusion{Explicit,Implicit}`); extended
   that same pgen (rather than writing a new one) to also support
   `opacity_type=photons` via a real `<mhd>`+`dyn_grmhd` fluid (guarded by
   `params.opacity_type`, so the toy path is untouched) — see
   `rad_m1_photon_diffusion.athinput`, `check_rad_m1_photon_diffusion.py`.
   Physics setup: `kappa_s>0` alone (`kappa_p=kappa_a=0, compton=false` — the
   exact configuration already validated by item 1's scattering-only null
   test, so `abs_1=0` exactly and there's no competing local-relaxation
   physics), giving pure diffusion `dE/dt=D*d²E/dx²`,
   `D=1/(3*kappa_s*rho)`. Checked via the E-weighted variance of the
   `E(x)` profile against `sigma²(t)=sigma0²+2Dt` (`m1_tab_utils.py`'s new
   `read_profile()`).
   - **Cross-checked against the reference discrete-ordinate module**
     (`src/pgen/rad_diffusion.cpp`/`inputs/radiation/rad_diffusion.athinput`,
     from arXiv:2302.04283, already existing/unmodified in this repo) with
     matched parameters (`inputs/radiation/rad_diffusion_m1crosscheck.athinput`):
     it reproduces the analytic diffusion law to ~0.6-1.2%, confirming the
     target formula is right and isolating the M1 scheme as the source of an
     initially much larger (3-34%) M1 discrepancy.
   - **Root cause found (partially fixed)**: the M1 pgen initialized `F_x=0`
     everywhere, but the reference module's *exact* diffusion-equation
     solution has a nonzero flux even in the static case
     (`F(x,0)=-D*dE/dx`, Fick's law — `rad_diffusion.cpp:112-116`). Starting
     from `F=0` forces a "spin-up" transient (~`1/kscat`) before flux and
     gradient reach the right relation, contaminating early-time spreading-rate
     measurements. Fixed by initializing `F_x` from the diffusive relation
     instead (only in the Gaussian branch, only for the photon path).
     **Result at `kappa_s=5`** (non-stiff/explicit source-solver branch,
     `cdt*kscat≈0.19`): error dropped to sub-1% by mid-run (was 3-8% before
     the fix), consistent with a decaying transient, not a bug — matches the
     DO module's own precision.
   - **Crash at `kappa_s=200` — ROOT-CAUSED AND FIXED.** With the corrected
     nonzero-flux IC, at `kappa_s=200` (stiff/implicit source-solver branch,
     `cdt*kscat≈7.5`) `E` used to collapse to the floor value **everywhere**
     (including the former peak) by the first output time, with total energy
     vanishing (`integral/integral(0)≈0`). Root-caused via a debug build
     (`CMAKE_BUILD_TYPE=Debug` enables the existing `DEBUG_BUILD`
     solver-failure printfs) plus temporary instrumentation (not committed):
     the failure starts at the domain-edge cells, where `E` has decayed into
     the Gaussian tail and gets clamped to `rad_E_floor=1e-30` while the
     diffusive-flux IC still assigns a correspondingly tiny nonzero `F_x`
     there. With `E` pinned at the floor, the Hybridsj Newton solve's
     residual becomes essentially insensitive to `F_x` at these scales:
     instead of converging, it chases `F_x` chaotically across 100+ orders of
     magnitude in the subnormal range (captured directly, e.g.
     `F_x` bouncing between `-6e-154`, `-1.3e-100`, `+5e-24` within a single
     failing solve) until an intermediate value underflows to exact `0.0`, a
     subsequent division produces `Inf`, and that becomes `NaN`
     (`x=[1e-30,-nan,-nan,-nan]` appears already at Newton iteration 1 of the
     failing call). `source_update_ll` returns `SrcFail` but leaves
     `Enew`/`Fnew_d` at that NaN-contaminated value (the clean assignment
     only happens on the success path,
     `radiation_m1_sources.hpp::source_update_ll`); the caller never checks
     the returned signal (`radiation_m1_update.cpp`); and the old
     `apply_floor()` didn't sanitize it either, since it only rescaled `F_d`
     `if (F2 > lim)` and a NaN comparison is always `false` in IEEE-754, so
     the NaN passed straight through into the conserved state and spread via
     the ordinary hyperbolic flux update (~8 more contaminated cells/cycle),
     eventually corrupting the whole domain. This was a **latent fragility**
     in the Newton-solver/floor interaction for near-floor states with tiny
     nonzero flux — never triggered before because every earlier M1 test used
     an exact `F=0` IC (a trivial fixed point needing zero iterations); the
     diffusive-flux IC fix above is what first seeds nonzero flux at
     floor-level cells.
     **Fix** (`apply_floor()`, `radiation_m1_helpers.hpp`): reset `(E,F_d)` to
     `(rad_E_floor, 0)` whenever either is non-finite, and zero `F_d`
     whenever `E` was at or below the floor before clamping — physically
     correct (no meaningful flux at floor density) and removes the
     ill-conditioned Newton hunt at its source, not just its symptom.
     Verified: the full `kappa_s=200` run (`tlim=200`) now completes with no
     NaNs and exact energy conservation (`integral/integral(0)=1.0` at every
     output); all four single-zone tests and the `kappa_s=5` diffusion test
     re-ran clean with unchanged results (no regression).
   - **Still open, NOT the crash — a genuine accuracy question**: with the
     crash fixed, `kappa_s=200` now runs to completion but its `sigma²(t)`
     still spreads ~27-34% faster than `sigma0²+2Dt` predicts, matching the
     *original* discrepancy that motivated cross-checking against the
     discrete-ordinate module in the first place (the diffusive-flux IC fix
     resolved `kappa_s=5` but evidently not this regime). Not yet
     investigated — possibly a genuine breakdown of strict diffusion-limit
     validity at this optical depth per cell (`kappa_s*dx≈18.75`, with the
     pulse width `sigma0` only ~2.5 cells wide), or a separate scheme issue
     specific to the stiff/implicit branch.

3. **Free-streaming/beam test with photon opacities.** Reuse
   `rad_m1_beams.cpp`/`radiation_m1_beams.cpp` with `opacity_type=photons` and
   `kappa`s near zero, confirming a beam propagates at exactly `c` without
   spurious spreading or damping — the opposite limit from the diffusion test,
   and a classic M1 failure mode (implicit solvers can over-damp free-streaming
   radiation even at `kappa~0`). **Open question, not yet resolved**: whether
   `rad_m1_beams.cpp` currently assumes a specific `opacity_type` (needs
   inspection before implementation).

4. **Radiation-pressure backreaction test (`backreact=true`,
   `backreact_tmunu=true`).** Start with the cheapest meaningful case rather than
   a full hydrostatic atmosphere: repeat the single-zone LTE setup but let
   `T_gas` respond too, so `E` and `T_gas` jointly relax to an equilibrium set by
   **total energy conservation** (`rho*cv*T + E` constant), not just `E` chasing a
   fixed external `T`. Check gets two things essentially for free: (a) energy
   conservation to near machine precision (a strong, cheap invariant-based check
   independent of knowing the transient in closed form), and (b) the correct
   final joint equilibrium `T_final`/`E_final` from the conservation constraint —
   optionally cross-check the transient itself against a `scipy`-integrated
   reference solution of the coupled 2-ODE system. A full spatial hydrostatic
   (Eddington) atmosphere test is a natural follow-on once this passes, but is
   more ambitious (needs a steady-state spatial profile, not just a 0-D
   equilibrium) — deliberately deferred past this first step.

5. **Wire into `tst/test_suite/` CI (pytest, AMR+MPI).** Follow the
   `test_rad_beam_gpu.py`/`test_rad_lwave*.py` pattern once 2-4 are stable. Open
   decision to make at that point: the CI harness's tab-reading path goes through
   `athena_read.tab()`/`vis/python`, which has the degenerate-dimension bug noted
   above. Options: (a) fix the general AthenaK tab writer/reader (broadest fix,
   most blast radius, benefits everything not just M1), (b) design the CI-facing
   test variants with genuinely non-degenerate geometry so the existing reader
   works unmodified, or (c) relocate/adapt `m1_tab_utils.py` into the `tst/`
   Python path for CI's use. Not resolved yet — revisit once we're actually
   wiring this in.

## Stage 3 (later) — capstone application

Bring `dyngr_bhstar.cpp` (quasi-star/BH-star Bondi accretion + LTE radiation
initial data) to a validated run: confirm the Bondi profile is recovered in
hydro-only mode first, then turn on radiation and confirm the LTE profile /
accretion luminosity are physically sane.

## Stage 4 (later) — hardening

Input validation for unimplemented combinations, and keeping this note current as
the module evolves.
