# CFC + BH-puncture for tidal disruption events — equations & implementation plan

**Status: planning only. No code changes yet.**
**Branch:** `proj/tde` in `~/athenak_cfc_tde`.
**Author context:** follow-up to Lam, Shibata & Kiuchi 2023 (arXiv:2212.10891),
*"Numerical-relativity simulation for tidal disruption of white dwarfs by a
supermassive black hole"*, re-implemented on top of the AthenaK **CFC module**
(`src/cfc/`, see `src/cfc/DEVELOPMENT.md`) instead of the full-GR
error-correction BSSN evolution used there.

---

## 1. Goal and the core idea

We want to model a white dwarf (WD) tidally disrupted by a non-spinning
supermassive black hole (SMBH, `M_BH ~ 1e5 M_sun`). The mass ratio is extreme
(`M_WD/M_BH ~ 1e-5`), so the spacetime is *almost exactly* the stationary SMBH
plus a tiny, dynamical perturbation from the debris.

The previous paper handled this with a **background split** of the full Einstein
equations:

- Every metric field `Q = Q0 + Qs`, where `Q0` is the *analytic, stationary,
  maximal-slicing trumpet puncture* of the SMBH and `Qs` is the small residual.
- The evolution used an **error-correction** RHS,
  `∂t Qs = F(Q0 + Qs) − F(Q0)`,
  which analytically subtracts the truncation error of the strong background so
  that the tiny stellar self-gravity is not swamped. BSSN + moving-puncture gauge
  (1+log lapse, Gamma-driver shift). Initial data from CTS/IWM elliptic solves.

**This time** we replace the free BSSN evolution with the CFC module. The key
observations that make this clean:

1. **CFC (Isenberg–Wilson–Mathews) is an elliptic constraint system solved every
   step**, not a hyperbolic free evolution. The metric is `γ_ij = ψ⁴ f_ij`
   (conformally flat), maximal slicing `K = 0`. There is *no* gauge evolution and
   *no* GW content — both are excellent approximations for a TDE (GW luminosity
   is dynamically negligible, and the field is quasi-stationary strong-field).

2. **A single non-spinning BH admits an *exact* conformally-flat, maximally-sliced
   slice** — the stationary maximal "trumpet". So CFC is *exact* for the SMBH
   background `Q0`; conformal flatness is only an approximation for the (tiny,
   slowly moving) debris. This is precisely the regime CFC is good in.

3. The **error-correction trick becomes trivial in an elliptic solve**: instead
   of a subtracted RHS, we simply solve the CFC elliptic equations *for the
   residual field* `Qs = Q − Q0` directly, with the analytic background moved to
   the source side. The discretization error then lives on the small residual,
   not on the huge `M/r` background — the elliptic analog of `F(Q0+Qs)−F(Q0)`.

4. The **puncture** is carried entirely in the analytic `Q0` (which is singular
   at `r → 0`). The residual `Qs` is regular everywhere, so **no excision of the
   metric is needed** — same virtue as moving-puncture, obtained for free.

So the project is essentially: **generalize the CFC solver's asymptotic
background from "flat space" (`ψ=1, αψ=1, β=0`) to "analytic trumpet puncture"
(`ψ0, α0ψ0, β0^i, Â0^ij`), and solve the CFC elliptic equations for the residual
sourced by the WD debris.**

---

## 2. What the CFC module already does (baseline)

From `src/cfc/DEVELOPMENT.md` and the code, the module solves the **XCFC** system
(Cheong et al. 2021 "Gmunu", arXiv:2012.07322, eqs. 71–76) every RK stage,
interleaved into the `NumericalRelativity` task graph, sourced from
`dyn_grmhd`'s conserved state. Decomposition `γ_ij = ψ⁴ f_ij`, `K = 0`.

Matter sources (`U ≡ n_μ n_ν T^{μν}`, `S_i ≡ −n_μ γ^ν_i T^{μν}`,
`S ≡ γ_ij S^{ij}`) are densitized as `Ũ = ψ⁶ U`, `S̃_i = ψ⁶ S_i`, `S̃ = ψ⁶ S`.

**6-step solve order (current):**

1. Vector potential `X^i`:  `Δ̃ X^i + (1/3) ∇̃^i(∇̃_j X^j) = 8π f^{ij} S̃_j`.
2. `Â^ij` (algebraic):  `Â^ij = ∇̃^i X^j + ∇̃^j X^i − (2/3) f^{ij} ∇̃_k X^k`.
3. Conformal factor `ψ` (nonlinear):
   `Δ̃ ψ = −2π Ũ ψ⁻¹ − (1/8) Â_ij Â^ij ψ⁻⁷`.
4. Rebuild trace source `S̃` from the shared con2prim.
5. Lapse `αψ` (nonlinear/affine):
   `Δ̃(αψ) = (αψ)[ 2π(Ũ + 2S̃) ψ⁻² + (7/8) Â_ij Â^ij ψ⁻⁸ ]`.
6. Shift `β^i`:
   `Δ̃ β^i + (1/3) ∇̃^i(∇̃_j β^j) = 16π α ψ⁻⁶ f^{ij} S̃_j + 2 Â^ij ∇̃_j(α ψ⁻⁶)`.

Each vector equation (steps 1, 6) is split à la Shibata (1999) into a flat
vector-Poisson `Δ P_i = S_i` plus scalar `Δ η = −S_i x^i`, reconstructed as
`V^j = (7/8) P_j − (1/8)(η_{,j} + P_{k,j} x^k)`.

**Current background convention:** the code solves *deviations from flat space*.
`ψ = 1 + δψ`, `αψ = 1 + δ(αψ)`, `β^i` about 0. The scalar solvers store
`delta_psi = ψ − 1`, `delta_alpha_psi = αψ − 1` (`cfc.hpp:93-94`). This hardcoded
"`−1`"/"about 0" background is exactly the single thing we generalize. **Outer BC
(corrected — see §3.10 for the verified, code-checked current state, which
supersedes an earlier, wrong draft of this line):** `ψ/αψ` actually default to
`BoundaryFlag::mg_robin` (a local extrapolation, not a multipole expansion), and
the vector potentials (`P_i`/`η` for `X^i`/`β^i`) actually default to
`BoundaryFlag::mg_multipole` — the reverse of what an earlier pass of this note
said. Both currently assume the source sits at the coordinate origin — §3.10
explains why that assumption breaks for this project and what changes.

Verified working (uniform grid, single/multi-MeshBlock): TOV star converges to
small resolution-stable residual. **AMR is NOT yet supported** — the two
nonlinear solvers fatal-error if `nreflevel_ > 0` (DEVELOPMENT.md item 3/3b).

---

## 3. The new equation set: CFC + non-spinning trumpet puncture

**Decided (per user):** stationary **maximal-slicing trumpet** background,
**fixed at the origin `x=0`**, **non-spinning** (⇒ spatial slice is exactly
conformally flat, consistent with CFC).

### 3.1 Field split

Write each CFC field as **analytic stationary background + regular residual**:

```
ψ      = ψ0            + u
αψ     = α0 ψ0         + v
β^i    = β0^i          + b^i
Â^ij   = Â0^ij         + Âs^ij
```

`{ψ0, α0, β0^i, Â0^ij}` is the SMBH's stationary maximal-slicing trumpet
solution. It satisfies the **vacuum** CFC equations exactly (`U = S_i = S = 0`),
so with no WD present all residuals vanish identically. The puncture singularity
lives entirely in `ψ0` (and `α0 → 0` at the throat); every residual `{u, v, b^i,
Âs}` is smooth ⇒ no metric excision needed.

> **Important trumpet caveat (changed from the wormhole case).** The trumpet is
> *not* a moment-of-time-symmetry slice: it has a **nonzero stationary extrinsic
> curvature `Â0^ij ≠ 0`** and a **nonzero stationary shift `β0^i ≠ 0`**. That
> `Â0^ij` is *not* generated by the matter-sourced vector potential `X^i` — it is
> a vacuum property of the slice. So we must **supply `Â0^ij` and `β0^i`
> analytically** as background fields (not regenerate them from `X`), and the
> matter-driven `Xs^i` solve produces only the *residual* curvature `Âs^ij` on top
> of `Â0^ij`. (With the wormhole slice both would be zero — that is the one real
> simplification we lose by choosing the trumpet, and it is worth it: finite
> proper volume, single asymptotic end, matches the 2022 paper.)

### 3.2 Background `Q0` — the maximal trumpet

**Notation (per user):** `r_iso ≡ |x|` is the grid ("isotropic-like" trumpet)
radial coordinate; `R_sch` is the areal (Schwarzschild-like) radius. Spherically
symmetric about `x = 0`, non-spinning.

**Source of truth: `~/SACRA_2D/SACRA_MPI/mod_bh.f90`** (`tbh_get_var`,
`tbh_areal_to_iso`, `tbh_iso_to_areal`) — a working, already-used-in-production
transcription of the stationary maximal-slicing trumpet. Plan directly ports
this, not a fresh derivation from Baumgarte & Naculich 2007 (arXiv:0709.0299;
the underlying reference the Fortran implements). Non-dimensionalize by `M_BH`:
`ρ ≡ R_sch/M_BH`, `χ ≡ r_iso/M_BH`, `n^i ≡ x^i/r_iso`. Exact formulas (`fac ≡
(3√3/4) M_BH²`):

```
ψ0        = sqrt( R_sch / r_iso )                                  (conformal factor)
α0        = sqrt( 1 − 2/ρ + 1.6875/ρ⁴ )       [1.6875 = 27/16]      (lapse, → 0 at throat)
β0^i      = fac · x^i / ( R_sch · ρ² ) = fac · x^i / R_sch³         (stationary radial shift)
Â0_ij     = (fac/R_sch³) · ( δ_ij − 3 n_i n_j )                     (traceless, spherically symm.)
Â0_ij Â0^ij = 10.125 · M_BH⁴ / R_sch⁶       [10.125 = 81/8]         (= 6·fac²/R_sch⁶, verified)
R_sch,0   = 1.5 M_BH                                                (throat)
```

`mod_bh.f90` also carries analytic radial derivatives `dψ0/dr_iso`, `dα0/dr_iso`,
`∂β0^i/∂x^j` — useful as an independent cross-check of whatever finite-difference
or reconstructed `Â0` the AthenaK port produces, but not required as an input
(§3.3's `Â0` is used directly, not re-derived from `β0` via `(Lβ0)^ij`).

**`r_iso(R_sch)` forward map** (`tbh_areal_to_iso`, exact, closed form, units
`M_BH=1`):
```
χ(ρ) = (1/4)·[ 2ρ+1+sqrt(4ρ²+4ρ+3) ] · { (4+3√2)(2ρ−3) / [8ρ+6+3·sqrt(8ρ²+8ρ+6)] }^(1/√2)
```
**Numerically verified** (this session, not hand-derived — see below) against
the expected asymptotics: `χ(ρ) → ρ − 1` as `ρ → ∞` (⇒ `ψ0 → 1 + M_BH/(2 r_iso)`,
the standard puncture falloff with the correct *positive* ADM mass — an earlier
draft of this note hedged on this sign/leading-term and that hedge is now
resolved) and `χ(ρ) → 0` as `ρ → 1.5⁺` (throat maps to the coordinate origin).
Both confirm the transcription is correct; no independent re-derivation needed.

Everything above depends only on `r_iso = |x|` and is **time-independent** (fixed
puncture at `x=0`) ⇒ the whole background is computed **once** at
initialization and stored (§3.6, §5) — never touched in the per-step solve.

### 3.3 Residual equations (what CFC actually solves each step)

Subtract the background's own (vacuum) equation from each XCFC equation. Because
the flat Laplacian is linear, `Δ̃ Q = Δ̃ Q0 + Δ̃ Qs` and the background terms
cancel against `Δ̃ Q0`, leaving:

**Step 1 — vector potential residual** `Xs^i` (linear):
```
Δ̃ Xs^i + (1/3) ∇̃^i(∇̃_j Xs^j) = 8π f^{ij} S̃_j
```
Matter-sourced only ⇒ *unchanged* from the current code. The trumpet's own vacuum
curvature `Â0^ij` is **not** produced here; it is supplied analytically in step 2.
(`Xs` here is the full solved `X` in the code, since we never introduce an `X0`.)

**Step 2 — algebraic curvature (add the analytic background):**
```
Â^ij = Â0^ij + [ ∇̃^i Xs^j + ∇̃^j Xs^i − (2/3) f^{ij} ∇̃_k Xs^k ]
```
i.e. add the analytic background `Â0^ij` to the matter-sourced part. `Â_ij Â^ij`
(the `a_sq` field) is then formed from the **total** `Â^ij`.

**Step 3 — conformal-factor residual** `u = ψ − ψ0` (nonlinear):
```
Δ̃ u = −2π Ũ ψ⁻¹ − (1/8) Â_ij Â^ij ψ⁻⁷  +  (1/8) Â0_ij Â0^ij ψ0⁻⁷
```
with `u → 0` at infinity (multipole/`1/r` BC), where `ψ = ψ0 + u` on the RHS.
The last term is the background's own source (`Δ̃ ψ0 = −(1/8) Â0² ψ0⁻⁷`), moved to
the RHS so `u` is sourced *only* by matter and by the `Â`-vs-`Â0` difference.
**For the trumpet `Â0 ≠ 0`, so this correction term is present and must be
included** (it would only drop if we used the wormhole slice).

**Step 5 — lapse residual** `v = αψ − α0ψ0` (affine once `ψ, Â` are fixed):
```
Δ̃ v = (αψ)[ 2π(Ũ + 2S̃) ψ⁻² + (7/8) Â_ij Â^ij ψ⁻⁸ ]
       − (α0 ψ0)(7/8) Â0_ij Â0^ij ψ0⁻⁸
```
with `v → 0` at infinity; `αψ = α0ψ0 + v`. The second line is the background
subtraction; **nonzero for the trumpet (`Â0 ≠ 0`) — keep it.**

**Step 6 — shift residual** `b^i = β^i − β0^i` (linear):
```
Δ̃ b^i + (1/3) ∇̃^i(∇̃_j b^j)
   = 16π α ψ⁻⁶ f^{ij} S̃_j + 2 Â^ij ∇̃_j(α ψ⁻⁶)
     − [ 16π α0 ψ0⁻⁶ f^{ij} S̃0_j + 2 Â0^ij ∇̃_j(α0 ψ0⁻⁶) ]
```
with `b^i → 0`. `S̃0_j = 0` in vacuum, so the subtracted term is just the
background's `Â0`-driven part; **nonzero for the trumpet (`Â0, β0 ≠ 0`) — keep
it.**

### 3.4 Boundary conditions

- Residuals `u, v, Xs, b` all carry only the **star's** small contribution (the
  BH's own `M_BH/r_iso`-type falloff is already fully absorbed into `ψ0, α0, β0,
  Â0`) ⇒ each falls off `~1/r_iso` at large distance. **This is why solving the
  residual is numerically superior**: the outer-boundary and discretization error
  act on an `O(M_WD)` quantity, not `O(M_BH)`.
- **This does *not* mean the existing BCs are correct unmodified.** The existing
  `mg_robin` (ψ/αψ) and `mg_multipole` (`P_i`/`η`) implementations both assume the
  *residual's own* leading multipole content is centered at the coordinate
  origin — true in every existing CFC test (star at `x=0`), false here (the BH,
  not the star, is at `x=0`). **See §3.10** for the verified current-code BC
  state and the recentering this requires — do not rely on the (superseded, now
  corrected) BC assignment this note originally had here.

### 3.5 Final ADM assembly

`γ_ij = ψ⁴ f_ij`, `K_ij = Â_ij/ψ² · (…)` with `K = 0`, `α = (αψ)/ψ`,
`β^i` — all built from the **total** fields `ψ = ψ0+u`, `αψ = α0ψ0+v`,
`β = β0+b`, `Â = Â0+Âs`, exactly as `cfc_reconstruct.cpp` already does, once the
background arrays are added in.

### 3.6 Trumpet radius conversion `r_iso → R_sch` — port SACRA's bracket method

Every background quantity is a function of the **areal** radius `R_sch`, but the
grid gives `r_iso = |x|`. `χ(ρ)` (§3.2) is closed-form and monotonic but has no
closed-form inverse — `mod_bh.f90::tbh_iso_to_areal` already solves this
robustly with a **bracketed root-find (Illinois method, a bisection/false-position
hybrid)**, not Newton. **Port this directly rather than the table-seeded-Newton
scheme this note originally proposed** — a bracketing method needs no derivative,
has no stiffness failure mode near the throat (unlike Newton, which is exactly
where a cold Newton is most likely to misbehave), and this is a one-time,
off-hot-path cost, so there is no performance reason to prefer Newton's faster
local convergence. Per `tbh_iso_to_areal`, in units `M_BH=1`, given target `χ_cell
= r_iso,cell/M_BH`:

```
func(ρ) ≡ χ(ρ)/χ_cell − 1                          (χ(ρ) = §3.2's forward map)

bracket:  ρ_lo = max(1.5, χ_cell),   ρ_hi = max(1.5, χ_cell + 1.5)
special case: χ_cell ≲ 1.5·tol  ⇒  ρ = 1.5  (the puncture itself, skip solve)
else: Illinois method on [ρ_lo, ρ_hi] to func(ρ)=0, tol = 1e-15, iter_max = 1e5
R_sch = M_BH · max(ρ, 1.5)                          (clamp guards throat roundoff)
```

The bracket is not arbitrary: since `χ(ρ) → ρ−1` for `ρ ≳ few`, `ρ ≈ χ_cell+1`
solves `func=0` approximately and always lies inside `[χ_cell, χ_cell+1.5]`
except very near the throat, where the `χ_cell ≲ 1.5·tol` special case (a
degenerate bracket, `ρ_lo=ρ_hi=1.5`) takes over instead. Confirmed numerically
this session (e.g. `χ_cell=10 → ρ_true≈11`, inside `[10, 11.5]`).

**Per-cell procedure:**
1. `r_iso = |x|`; if `r_iso ≲ 1.5 M_BH · tol`, set `R_sch = 1.5 M_BH` (throat) —
   this is purely a **numerical safeguard against the degenerate root-find at the
   single exact `r_iso=0` point** (avoiding a divide-by-zero/degenerate bracket in
   the Illinois solve), not a physical excision decision. **Do not conflate this
   with the fluid excision region** (§5 Phase B #9), which is a separate,
   *physically* motivated, much larger region (order the horizon scale, not a
   single grid point) — see that item for the resolved criterion/radius.
2. Else run the Illinois bracket above to get `R_sch`.
3. Assemble `ψ0, α0, α0ψ0, β0^i, Â0_ij, Â0²` from `R_sch, x^i` per §3.2.

**Fixed puncture (`x=0`, time-independent) ⇒ this whole procedure runs once**, in
a single `par_for` over the grid at CFC construction/initialization (rebuilt only
on an AMR regrid) — never in the per-stage hot loop, which only reads the stored
background arrays.

Implementation: new `cfc_puncture.{hpp,cpp}`. `tbh_areal_to_iso`'s `χ(ρ)` and
`tbh_iso_to_areal`'s Illinois solve port essentially line-for-line from
`mod_bh.f90` (device-callable free functions, no table/interpolation needed —
drop that part of this note's earlier draft along with the table-Newton idea).
`tbh_get_var`'s `ψ0/α0/β0/Â0/Â0²` assembly ports the same way. Keep the existing
wormhole formulas (earlier §3.2 draft) behind a compile/runtime flag purely as a
cheap debug fallback (`M_BH→0` sanity check, §5 Phase D #12) — not derived from
SACRA, so keep it clearly marked as such.

### 3.7 Puncture regularization near `r_iso = 0` — catastrophic cancellation

**The problem.** Near the puncture `ψ0 → ∞`, so both `ψ⁻⁷` and `ψ0⁻⁷` are tiny,
and their difference `ψ⁻⁷ − ψ0⁻⁷` (which appears in the ψ residual, §3.3 step 3)
is a *difference of two tiny nearly-equal numbers* — catastrophic cancellation, no
significant digits survive in double precision. The same happens for `ψ⁻⁸ − ψ0⁻⁸`
in the lapse residual (step 5). Computing these two terms as written is numerically
fatal at the cells nearest `x=0`.

**The fix (from `~/NSWD/octree-mg/src/m_cfc_psi_tbh.f90::box_lpsi_tbh`).** Never form
the difference directly. Introduce `x ≡ δψ/ψ0` (the *ratio* of the residual to the
background) and factor the difference analytically. Using
`ψ = ψ0(1+x)` ⇒ `ψ⁻ⁿ − ψ0⁻ⁿ = ψ0⁻ⁿ[(1+x)⁻ⁿ − 1]` and `(1+x)ⁿ − 1 = x·Pₙ(x)`:

```
ψ⁻⁷ − ψ0⁻⁷ = − ψ0⁻⁷ · x·P7(x)/(1+x)⁷ ,
    P7(x) = 7 + 21x + 35x² + 35x³ + 21x⁴ + 7x⁵ + x⁶      [ = ((1+x)⁷−1)/x , C(7,k) ]

ψ⁻⁸ − ψ0⁻⁸ = − ψ0⁻⁸ · x·P8(x)/(1+x)⁸ ,
    P8(x) = 8 + 28x + 56x² + 70x³ + 56x⁴ + 28x⁵ + 8x⁶ + x⁷    [ = ((1+x)⁸−1)/x , C(8,k) ]
```

This is *exact*, not an approximation. For small `x` (which is exactly the regime
near the puncture, since `δψ` is a bounded regular correction while `ψ0 → ∞`) it is
a well-conditioned polynomial — leading behavior `ψ⁻⁷−ψ0⁻⁷ ≈ −7 ψ0⁻⁸ δψ` — with no
cancellation, and it → 0 smoothly as `ψ0 → ∞`.

**Where it plugs into the CFC residual equations.** Split `Â² = Â0² + ΔÂ²`
(`ΔÂ² = 2 Â0·Âs + Âs²`, the matter/residual part, which is `O(matter)` and regular).
Only the *pure-background* `Â0²` piece multiplies a `(ψ⁻ⁿ − ψ0⁻ⁿ)` difference; the
`ΔÂ²` and matter (`Ũ ψ⁻¹`) pieces multiply `ψ⁻ⁿ` alone (tiny but single-valued near
the puncture — no cancellation). So §3.3 step 3 becomes, in regularized form:

```
Δ(δψ) = −2π Ũ ψ⁻¹ − (1/8) ΔÂ² ψ⁻⁷
        + (1/8) Â0² ψ0⁻⁷ · x·P7(x)/(1+x)⁷        ← the regularized −(1/8)Â0²(ψ⁻⁷−ψ0⁻⁷)
```
Note `x·Pₙ(x)/(1+x)ⁿ = 1 − (1+x)⁻ⁿ`, a convenient equivalent for coding.

**ψ solver — nonlinear, regularize in the operator + Newton Jacobian.** The ψ
equation is genuinely nonlinear in `δψ` (self-coupled through `ψ⁻⁷`), so
`MGCFCConformalFactor` does per-point Newton–Gauss–Seidel (DEVELOPMENT.md item 3,
Finding A). Put the `P7` form directly in the operator/defect, and use the exact
cancellation-free derivative (from `box_gs_lpsi_tbh`, line 164) in `dLdu`:
```
d/d(δψ) [ (1/8) Â0² ψ0⁻⁷ (1−(1+x)⁻⁷) ] = (7/8) Â0² ψ0⁻⁸ (1+x)⁻⁸
```

**Lapse solver — LINEAR, regularize in the coefficient/RHS assembly (verified
against `m_cfc_alp_tbh.f90` + `cfc_solve_alp`).** The lapse operator
`box_lalp_tbh` is exactly `L(v) = Δv + K·v` and its smoother `box_gs_lalp_tbh`
(lines 135-142) is a *direct linear division* — **no Newton step at all** (once
`ψ, Â` from step 3 are fixed, `K` is a known field; Finding A, now confirmed at the
operator level). So the `ψ⁻⁸−ψ0⁻⁸` cancellation is handled entirely when the
coefficient `K` and RHS are *built*, not in the smoother. Writing §3.3 step 5 as
`Δv + K·v = rhs` with `v = αψ − α0ψ0`:
```
K   = −[ 2π(Ũ+2S̃) ψ⁻² + (7/8) Â² ψ⁻⁸ ]                       (known; no subtraction)
rhs = α0ψ0 [ 2π(Ũ+2S̃) ψ⁻² + (7/8) ΔÂ² ψ⁻⁸ ]
      + α0ψ0 (7/8) Â0² ψ0⁻⁸ · x·P8(x)/(1+x)⁸                 ← regularized (ψ⁻⁸−ψ0⁻⁸)
    P8(x) = 8 + 28x + 56x² + 70x³ + 56x⁴ + 28x⁵ + 8x⁶ + x⁷    [ ((1+x)⁸−1)/x , C(8,k) ]
```
**Note the init code does this differently and slightly less accurately:**
`cfc_solve_alp` (lines 2017-2033) keeps ratio-powers of `(1+x)=ψ/ψ0` and forms
`A2p − A2·(1+x)⁴`, whose background piece reduces to `Â0²[1−(1+x)⁻⁸]` — a *mild*
cancellation for the small `x` near the puncture that the explicit `P8` form above
removes entirely. **Recommendation: use the explicit `P8` polynomial (as the ψ
solver already does with `P7`), not the `A2p − A2(1+x)⁴` regrouping** — strictly
better conditioned, at no extra cost.

**Consequence for the field convention.** This *requires* the residual to be stored
and iterated as `δψ = ψ − ψ0` (deviation from the analytic background), exactly the
convention §5 Phase A #3 already sets up — and `ψ0` must be available accurately at
every point the operator/smoother touches, which is what §3.8 guarantees. The
Laplacian is applied to `δψ` **only**; `ψ0` is never finite-differenced (it is
sharply peaked and its `Δψ0` contribution is folded analytically into the source
terms — differencing it numerically near the puncture would reintroduce the very
error we are avoiding).

### 3.8 Analytic puncture coefficients at *every* multigrid level (not restricted)

**The principle (from `~/NSWD/init_TBHWD_headon.f90::cfc_solve_psi`, the per-level
loop `do lvl = lowest_lvl, highest_lvl`).** The multigrid coefficient fields that
depend on the analytic puncture — `ψ0`, `α0`, `Â0²`, and the stored areal radius
`R_sch` — are **evaluated directly from the closed-form trumpet solution at each
MG level's own grid points** (`tbh_get_var(...)` called per level), *not* computed
on the finest grid and restricted down the hierarchy.

**Why this matters (accuracy + convergence).** `ψ0 = √(R_sch/r_iso)` is sharply
peaked (diverges) at the puncture. Restriction is an *averaging* operation:
restricting a sharply-peaked `ψ0` to a coarse level smooths the peak, so the
coarse-grid operator no longer represents the true fine-grid physics, and FAS
coarse-grid correction degrades or stalls — precisely the kind of corner/center
convergence failure the CFC module already fought once (DEVELOPMENT.md item 9,
rounds 8-15, though that was a different root cause). Evaluating `ψ0` analytically
per level keeps the coarse operators exact and the V-cycle converging.

**How it maps onto the AthenaK CFC module.** DEVELOPMENT.md Findings B/C established
that the CFC solvers keep coefficients in `coeff_` (not `src_`, which FAS would
tau-correct), and currently push them to coarse levels via `RestrictCoefficients()`
plus `TransferCoeffToRoot()`. The refinement:

- **Background channels (`ψ0`, `α0`, `Â0²`, `R_sch`) — fill analytically per level,
  do *not* restrict.** Replace the `RestrictCoefficients()` step *for these channels*
  with a per-level `par_for` calling the `cfc_puncture` generator (§3.6) at that
  level's cell coordinates. This requires the generator be callable at arbitrary MG
  level coordinates, and the multigrid hierarchy to expose per-level cell centers.
  **Verify AthenaK's `Multigrid` exposes per-level coordinates** (octree-mg uses
  `boxes(id)%r_min + ([IJK]−0.5)·dr(:,lvl)`); if not, this is a small addition.
- **Matter channels (`Ũ`, `S̃`, `ΔÂ²`) — keep as they are (finest-grid + restriction /
  FAS).** These come from the fluid, which only lives on the finest grid, so they
  must still be restricted (or handled by FAS) as the module already does. Only the
  *analytic* puncture part is evaluated per level.
- **Precompute and store `R_sch` per level** as its own coefficient channel: the
  Illinois root-find (§3.6) runs once per level at setup, and every subsequent
  `tbh_get_var` on that level reuses it (mirrors `cc(IJK, i_rs)` in the Fortran).
- **AMR caveat.** Per-level analytic coefficients on refined *octets* intersect the
  still-open `MGOctet` coefficient-storage gap (DEVELOPMENT.md item 3b / Finding D).
  For the uniform-grid bring-up this is not needed; for production AMR it must be
  solved together with item 3b. Flagged in §5 Phase C.

### 3.9 Puncture regularization for the shift `β^i` (§3.3 step 6) — the *easy* one

**The CFC shift is far simpler than the CTS shift — and this is not an accident.**
I checked the reference (`m_cts_beta_boost_v2.f90` + `cfc_solve_betam_v2`), and the
reason is structural:

- In **CTS**, `Â^ij` is built from the shift itself (`Â^ij ∝ (Lβ)^ij`), so the source
  term `2 Â^ij ∇_j(αψ⁻⁶)` becomes an *operator acting on the unknown `β`* — an
  advection-like `f_j·(Lβ)^ij` term that the CTS code folds into the elliptic
  operator with **sign-upwinded first-derivative stencils** (`box_lbeta_cts_boost_v2`
  lines 218-262, `dx1_p`/`dx1_m` chosen by `sign(f)`), plus a boost velocity. That is
  the complexity the user warned about.
- In **XCFC/CFC**, `Â^ij` comes from the *separately solved* vector potential `X^i`
  (§3.3 steps 1-2), so `2 Â^ij ∇_j(αψ⁻⁶)` is a **fully explicit, known source**. The
  shift operator stays the **plain flat vector-Laplacian** `Δb^i + (1/3)∇^i∇_jb^j`,
  i.e. the module's existing Shibata `P_i`/`η` machinery **unchanged** — no
  operator edits, no upwinding, no boost, no `β`-in-the-operator coupling.

So **all shift regularization lives in building the explicit source**, in
`AssembleVectorSource(for_shift=true)`/`BuildShiftSource` (`cfc.cpp`). And most of
the source is already benign at the puncture:

- **Matter momentum term** `16π α ψ⁻⁶ S̃_i`: since `S̃_i = ψ⁶ S_i`, the `ψ⁻⁶` cancels
  → `= 16π α S_i`. No `ψ` divergence, and `S_i = 0` in the vacuum near the BH. Nothing
  to regularize.
- **`Â^ij`** = `Â0^ij + Âs^ij`: `Â0 ∝ C/R_sch³` is finite at the throat and `Âs → 0`
  in vacuum. Finite. Nothing to regularize.

**`∇_j(αψ⁻⁶)` — the puncture-sharp part cancels out of the residual equation
entirely; it never appears as a standalone additive term.** `αψ⁻⁶ → 0` at the
puncture as a product of the *divergent* `ψ0` and *vanishing* `α0`, which is why
naively finite-differencing the assembled *total* field there would be
inaccurate. **Re-derived here from §3.3 step 6 to be unambiguous about what
actually survives, since an earlier draft of this note described this in a way
that could be misread as "add the background term, then also subtract it" — two
separate actions that must exactly cancel, not two terms that both appear in the
final answer.** Start from step 6's exact (unregularized) residual form,
`Δ̃b^i+... = 16παψ⁻⁶S̃_i + [2Â^ij∇_j(αψ⁻⁶)](total) − [2Â0^ij∇_j(α0ψ0⁻⁶)](background)`.
Expand the bracketed *total* term via `Â=Â0+Âs`, `αψ⁻⁶ = α0ψ0⁻⁶ + Δ(αψ⁻⁶)`
(Leibniz product rule):
```
[2Â^ij∇_j(αψ⁻⁶)](total)
  = 2Â0^ij∇_j(α0ψ0⁻⁶)                                        ← identical to the
  + 2Â0^ij∇_j[Δ(αψ⁻⁶)] + 2Âs^ij∇_j(α0ψ0⁻⁶) + 2Âs^ij∇_j[Δ(αψ⁻⁶)]    subtracted background
```
The first piece is, term for term, **exactly** the `[background]` piece already
being subtracted in the residual equation above — **it cancels identically,
verified by direct substitution.** So the shift residual's `Â`-gradient
contribution, in full, is:
```
2Â^ij∇_j(αψ⁻⁶) − 2Â0^ij∇_j(α0ψ0⁻⁶)  =  2Â0^ij∇_j[Δ(αψ⁻⁶)] + 2Âs^ij∇_j(α0ψ0⁻⁶) + 2Âs^ij∇_j[Δ(αψ⁻⁶)]
```
**— three deviation cross-terms, full stop. The analytic background piece
`2Â0^ij∇_j(α0ψ0⁻⁶)` never appears as a term in the final sum**; it is only an
*ingredient* two of the three surviving terms need (`Â0` itself, and the
closed-form `∇_j(α0ψ0⁻⁶)` factor):
- `∇_j(α0ψ0⁻⁶) = (α0ψ0⁻⁶)[dα0/α0 − 6 dψ0/ψ0]·n_j`, `n_j=x_j/r_iso` — closed-form
  (SACRA `tbh_get_var` already returns `dpsi`,`dalp`), computed once per level
  (§3.8) alongside `ψ0/α0/β0/Â0`. Used *only* inside the `2Âs^ij∇_j(α0ψ0⁻⁶)` term
  above (multiplied by the matter-sourced `Âs`, itself smooth/bounded) — **not
  added to the residual RHS on its own.**
- `Δ(αψ⁻⁶) = αψ⁻⁶ − α0ψ0⁻⁶`, needed inside the other two terms, is itself built
  cancellation-free via §3.7's background factor
  (`αψ⁻⁶ = (α0ψ0+δφ)ψ0⁻⁷(1+x)⁻⁷`, `x=δψ/ψ0`), then ordinarily finite-differenced
  (it is smooth/bounded, an `O(matter)` quantity — no puncture divergence).

All three surviving terms are smooth, bounded, `O(matter)` quantities — genuinely
**no regularization needed** for the shift, beyond building `Δ(αψ⁻⁶)`
cancellation-free (which just reuses §3.7's machinery, already required for
ψ/lapse).

**This drops the CTS `dalppsi`/`dalppsi0` log-gradient machinery entirely** — it was
needed in the CTS code because `Â^ij` there was `(Lβ)`-coupled into the operator; in
CFC the background piece is just an analytic source field, so the elaborate split is
unnecessary.

**Background subtraction (residual `b^i = β^i − β0^i`).** Unlike ψ/α — where the split
is *mandatory* because `ψ0`/`α0ψ0⁻ⁿ` diverge — the shift background `β0^i = C x^i/R_sch³`
is **regular (finite, → 0 at the puncture)**, so a full-`β` solve is also viable.
Recommend still solving the residual `b^i` for consistency with the error-correction
framework (discretization error on the small residual): the source is exactly the
three deviation cross-terms derived above (no separate subtraction step needed at
implementation time — the cancellation is already built into that formula); add
`β0^i` back at reconstruction (§5 Phase A #3).

**MG-level treatment of this source (a gap this review closes).** §3.8 mandates
"analytic per level, never restricted" specifically because `ψ0` *diverges* at the
puncture, which restriction (an averaging operation) handles badly. `Â0`, `β0`, and
`α0ψ0⁻⁶` are all **bounded** everywhere (including at the throat — `Â0² ∝ M⁴/R_sch⁶`
is finite at `R_sch=1.5M`; `α0ψ0⁻⁶ → 0` smoothly, no divergence) — restricting the
shift's `src_` (which is how the vector-Poisson solve already works: a genuinely
linear equation using the generic `src_`/FAS machinery, no `coeff_` at all, per
Finding B) is therefore a much milder approximation than restricting `ψ0` would be.
**Recommendation: do not build new per-level/`coeff_`-like infrastructure for the
vector-Poisson solver — let the shift's analytic source term restrict via the
existing generic mechanism, same as the matter part.** This is a deliberate,
reasoned choice (not an oversight), justified by the boundedness above; revisit
only if Phase D validation shows it matters.

**Net for the plan:** the shift needs *no* new multigrid operator/smoother work
(contrast psi §3.7-nonlinear and lapse §3.7-linear-coefficient) and *no* special
regularization — only a source assembly in `cfc.cpp` that computes the three
deviation cross-terms derived above (`2Â0^ij∇_j[Δ(αψ⁻⁶)] + 2Âs^ij∇_j(α0ψ0⁻⁶) +
2Âs^ij∇_j[Δ(αψ⁻⁶)]`), each using the analytic `Â0`/`∇_j(α0ψ0⁻⁶)` only as an
ingredient — **the residual source has no separate additive background term to
remember to subtract.**

### 3.10 Outer boundary condition: recentering on the star, not the origin

**Why this matters now (it didn't before).** Every existing CFC outer-BC choice
was built assuming *the matter sits at the coordinate origin* — literally stated
in `mg_cfc_vector_poisson.cpp`'s constructor comment: "`autompo_` is deliberately
left false (fixed origin (0,0,0)) ... every current CFC test problem's star sits
at the coordinate origin." **That assumption is now false by construction**: the
BH puncture is fixed at `x=0` (§3.2), and the WD orbits *around* it, so the
residual fields `u,v,b^i` — sourced almost entirely by the WD — are centered
somewhere that moves, generally far from the origin. An outer BC that (implicitly
or explicitly) assumes the source is at `x=0` will be systematically biased,
exactly the way an off-center multipole expansion is until you center it on the
source. The user's proposal is the right fix; here is what I verified against the
actual code and what I'd refine.

**Current BC state (verified in code, not from memory):**
- **ψ, αψ** (`MGCFCConformalFactor`/`MGCFCLapse`) default to `BoundaryFlag::mg_robin`
  (`mg_cfc_conformal_factor.cpp:243-287`, `mg_cfc_lapse.cpp:229-257`) — **not**
  `mg_multipole`. This supersedes this note's own earlier draft, which still said
  `mg_multipole`/DEVELOPMENT.md's older narrative text (`DEVELOPMENT.md:295-299`,
  stale). `mg_robin` is a **purely local** extrapolation, `ghost = interior_anchor
  · (r_anchor/r_ghost)^robin_order` (`multigrid_driver.cpp:2125-2230`,
  `2433-2513`), where **`r_a`,`r_g` are literal distances from the mesh/coordinate
  origin** (`x1min_v`, etc. — absolute mesh coordinates, no origin subtraction at
  all, confirmed by direct read). `mg_multipole` was deliberately *not* used here
  because of a real historical bug (`CalculateCenterOfMass()` integrates `src_`,
  but ψ/αψ's matter lives in `coeff_` by design — Finding B — so it silently
  divided by zero; `mg_cfc_conformal_factor.cpp:252-265`). Robin sidesteps that bug
  class entirely rather than fixing it.
- **X^i, β^i's `P_i`/`η`** (`MGCFCVectorPoisson`) default to `BoundaryFlag::mg_multipole`
  (`mg_cfc_vector_poisson.cpp:211-223`), with real **per-channel multipole
  machinery already in place** (`nvar_` up to `kMaxMultipoleChannels=4`, exactly
  matching the packed `P_i`(3)+`η`(1) — DEVELOPMENT.md item 17), origin `mpo_[0..2]`
  settable via `autompo_=false` (currently just left at the base-class default
  `(0,0,0)`, never actually computed).

**Recommendation — two different mechanisms, matched to what already exists:**

1. **ψ, αψ (mass-weighted center, Robin BC).** Compute
   `r_com^i = Σ x^i·D·ΔV / Σ D·ΔV` where **`D` is `pmy_pack->pmhd->u0(IDN)`
   directly** — this is *already* the fully densitized conserved rest-mass density
   `sqrt(γ)·ρ·W` (confirmed: `primitive_solver_hyd.hpp:205,283`, `cons(IDN) =
   cons_pt[CDN]*sdetg`). **Do not multiply by an extra `sqrt(γ)`/`ψ⁶`** — the user's
   written formula `∫x·D·√γ` double-counts if `D` denotes this same conserved
   variable; use `u0(IDN)` alone, unweighted further. This mirrors
   `CalculateCenterOfMass()`'s own reduction pattern exactly (`multigrid_driver.
   cpp:2733-2792`: per-MeshBlock `par_for` building `Σs·vol`/`Σs·x·vol`/etc.,
   `Kokkos::create_mirror_view_and_copy` to host, `MPI_Allreduce`) — reuse that
   *shape*, but as a **new CFC-local reduction reading `u0` directly**, not a fix to
   `CalculateCenterOfMass()` itself (which reads `src_`, permanently wrong for
   these two solvers' `coeff_`-based storage — no need to touch it).
   **This requires an actual code change to `mg_robin`**, since it has *no* settable
   center today: add a center point (analogous to `mpo_`, e.g. `robin_center_[3]`)
   to `MultigridDriver`/the two ghost-fill kernel sites (root `MGRootBoundary`,
   `multigrid_driver.cpp:2125-2230`; non-root `PhysicalBoundary`, `2433-2513`), and
   subtract it before computing `r_a`,`r_g`. This is the right minimal fix — it
   gives Robin's existing order-1 (pure `1/r`) assumption the same leading-order
   benefit a centered monopole term gives a full multipole expansion, at Robin's
   existing (cheap, local) cost; no need to revive `mg_multipole` for ψ/αψ to get
   this (that remains a possible *future* upgrade if higher-order accuracy is later
   needed, now that we know how to bypass `CalculateCenterOfMass()`'s `coeff_`
   mismatch by injecting an externally-computed center directly — not needed for
   the first milestone).

2. **X^i, β^i (multipole BC) — `S_jS^j` means each equation's *own* source, not
   momentum specifically (user correction, verified against the actual code —
   the two equations' sources are genuinely different arrays, not proportional).**
   `r_com^i = Σx^i·(S_jS^j)·ΔV / Σ(S_jS^j)·ΔV`, same point for all 3 vector
   components of a *given* solve (one elliptic solve, one expansion center) —
   but **computed separately for `X^i` and `β^i`**, each from that equation's own
   assembled `p_src_` (the actual channel-0-2 array `cfc.cpp` builds right before
   the Shibata `P_i`/`η` decomposition), not a single shared "momentum" proxy:
   - **`X^i`'s own source** (`cfc.cpp`'s step-1 branch, confirmed by direct read):
     `p_src_(a) = 8π·S̃_a`, where `S̃_a = cons(IM1+a)` is the **raw densitized**
     conserved momentum (`= ψ⁶S_a`, no compensating prefactor — this literally
     *is* Gmunu eq. 72's RHS, `8π f^{ij}S̃_j`). Squaring this raw quantity for the
     weight turns the `ψ⁶` densitization into `ψ¹²`, which **can spuriously
     amplify any small nonzero value near the puncture** (e.g. atmosphere-floor
     momentum just outside the excision mask), biasing the centroid toward the BH.
     **Divide out `ψ¹²` before squaring** (equivalently: weight by the physical
     `S_aS^a`, not the raw `p_src·p_src`) — this is still "the general source term
     of the `X^i` equation," just corrected for *that equation's own* known
     densitization convention.
   - **`β^i`'s own source** (`cfc.cpp`'s step-6 branch, confirmed by direct read,
     *after* `BuildShiftSource` has added the derivative term):
     `p_src_(a) = 16πα·S_a + 2Â^{aj}∇_j(αψ⁻⁶)` — the matter piece's `αψ⁻⁶`
     prefactor **already exactly cancels** `S̃_a`'s `ψ⁶` (confirmed:
     `ap6·s_tilde_d = (αψ⁻⁶)(ψ⁶S_a) = αS_a`), and the `Â`-gradient piece is built
     directly from `α,ψ,Â` (no bare conserved density at all, and it vanishes
     smoothly at the puncture since `α0→0` there). **No `ψ¹²`-type correction
     needed** — square the actual, as-assembled `p_src_` directly. This term is
     *not* proportional to `X^i`'s own source (the `Â`-gradient piece has a
     genuinely different spatial profile), which is exactly why it needs its own
     `r_com`, not a shared one.
   - **Do not share a single `r_com` between the two solves.** For a coherently
     orbiting, undisrupted star `r_com_X` and `r_com_β` should numerically nearly
     coincide (both track "where the star is," and both are legitimate proxies)
     — log their agreement as a cheap sanity check (§5 Phase D #14) — but don't
     force them equal or reuse one for the other; once the `Â`-gradient
     contribution becomes non-negligible (e.g. during tidal distortion) there's
     no reason to expect exact coincidence, and each solve should be centered on
     its own actual source.
   - **Task-order consequence:** `r_com_X` can be computed directly from `u0`
     as soon as it's available (early in the stage, same dependency as
     `CFC_BuildSrcX`); `r_com_β` can only be computed *after* `BuildShiftSource`
     has finished assembling β's `p_src_` (i.e. after the point in `cfc.cpp`
     marked "using the now-complete `p_src`," right before η's source is built).
   No such risk/subtlety for the mass-centroid above — `u0(IDN)` integrates
   directly to the total rest mass, a bounded physical quantity with no
   puncture-adjacent blow-up, so it needs no correction and is shared as-is
   between ψ and αψ (both genuinely have the same mass-like leading source).
   **Mechanism: no shared-module change needed.** Unlike Robin, `mg_multipole`
   *already* supports an externally-set, non-auto origin (`autompo_=false`, `mpo_`
   is `protected` not `private`) — `MGCFCVectorPoissonDriver` just needs one new
   public setter (`SetMultipoleOrigin(x,y,z)` assigning its own inherited `mpo_`)
   that `cfc.cpp` calls once per solve (twice per stage — once with `r_com_X`
   before the `X^i` solve, once with `r_com_β` before the `β^i` solve) instead of
   the current hardcoded `(0,0,0)`.

3. **Consistency requirement the proposal implies but doesn't state explicitly —
   verified, not assumed.** The user's suggestion to replace `η`'s source
   `-S_i x^i` with `-S_i(x^i-r_com^i)` is exactly right, but I checked the
   Shibata-decomposition identity by direct re-derivation: `Δ̃V_i + (1/3)∇̃_i∇̃_jV^j
   = S_i` is solved by `V_i = (7/8)W_i - (1/8)(∂_iχ + y^k∂_iW_k)` with `ΔW_i=S_i`,
   `Δχ=-y^jS_j`, for **any constant shift `y≡x-x_0`** — the identity's derivation
   only uses `∂_iy^j=δ_i^j` (true for any constant `x_0`), so it holds exactly, not
   approximately, *provided `y` is used consistently in both places*. **This means
   the reconstruction step must shift too, not just `η`'s source.** I confirmed in
   code that both currently use the *raw* origin-based `x^k`:
   - `η`'s source: `cfc.cpp:879-888` (X^i) and the equivalent block further down
     for `β^i` (`eta_val -= p_a*xk[a]`, raw `xk`).
   - The reconstruction cross-term: `cfc_reconstruct.cpp:91,97`
     (`ReconstructVectorFromPotentialsImpl`, `sum += ∂_jP_k · xk[kdir]`, raw `xk`,
     shared by both the `X^i` and `β^i` reconstructions).
   **Shifting only the source (as literally written in the user's message) while
   leaving the reconstruction's `xk` unshifted would silently reconstruct the
   WRONG `V^i`** — not just a worse-conditioned one. Both sites must subtract the
   same `r_com` used for that solve's multipole origin — and since `X^i`/`β^i` now
   use *different* `r_com`'s (item 2 above), each reconstruction call
   (`ReconstructVectorFromPotentials` for `X^i`, and separately for `β^i`) must be
   passed *its own* solve's `r_com` (`r_com_X` for `X^i`'s reconstruction,
   `r_com_β` for `β^i`'s) — sharing one would reintroduce exactly the inconsistency
   this item warns against, just between the two solves instead of within one.

4. **Where this fits in the per-stage task graph.** The mass-weighted `r_com`
   (item 1) and `r_com_X` (item 2) only need `pmy_pack->pmhd->u0`, available from
   the start of the stage — compute both in one early task (e.g. `CFC_ComputeCOM`,
   depending on the same post-flux-update `u0` that `CFC_BuildSrcX` already depends
   on), before `X^i`'s solve/BC/reconstruction run. `r_com_β` is computed later in
   the same stage, right after `BuildShiftSource` finishes assembling β's `p_src_`
   (item 2's task-order note), before β's own solve/BC/reconstruction. Recompute
   every stage (matches the module's existing per-stage re-assembly pattern; cheap
   relative to the multigrid solves themselves; revisit only if profiling ever
   shows otherwise). No special bootstrap issue for `t=0`/`InitializeMetric` — the
   pgen's initial primitives already populate `u0` before the first CFC solve.

---

## 4. How this differs from the CTS initial data (do not conflate)

The elliptic equations look almost identical, but they are **different systems**:

| | CTS / XCTS (initial data, once) | CFC / IWM (this project, every step) |
|---|---|---|
| Conformal metric | freely specifiable `γ̃_ij` (can be non-flat) | forced flat `f_ij` (`γ_ij = ψ⁴ f_ij`) |
| Time derivatives | `∂t γ̃_ij`, `∂t K` are free data | dropped (`∂t γ̃ = 0`, `K = 0` imposed) |
| What is solved | Hamiltonian + momentum constraints + lapse (XCTS) on **one** slice | same-form elliptic system re-imposed as the **full dynamics** at every step |
| Role of `Â` | fixed by `(Lβ)` + free `Â_TT` | entirely from matter `X^i` + analytic `Â0` |

So the CTS puncture solver from the 2022 work is **only** used (conceptually) to
seed *consistent initial matter + background*; the running approximation here is
CFC, which is stricter (conformal flatness + `K=0` at all times). For a single
non-spinning BH the two coincide exactly on the initial slice, which is why the
trumpet background is a fixed point of the CFC solve.

**Initial data in practice:** we do *not* need to port the full CTS/KADATH
puncture solver. Set (i) the analytic trumpet `Q0`, (ii) the WD matter
(equilibrium profile boosted onto its orbit), then run the module's existing
`CFC::InitializeMetric()` fixed-point (`cfc.hpp:277`), which already iterates
`X^i`/`ψ` ↔ con2prim to make the metric self-consistent with the matter. That
loop just needs to iterate the *residual* about `Q0` instead of about flat space.

---

## 5. Implementation plan

Ordered, each item scoped against existing code. Nothing below is started.

### Phase A — background infrastructure (the crux)
1. **Add background storage to `cfc::CFC`**: `psi0`, `alpha0_psi0` (scalars),
   `beta0_u` (rank-1), `a0_dd`/`a0_sq` (rank-2 + scalar), and the **analytic shift
   background source** `s0_beta_u ≡ 2Â0^ij∇_j(α0ψ0⁻⁶)` (rank-1, §3.9) arrays, sized
   like the existing residual fields. **Time-independent (fixed puncture at `x=0`) ⇒
   populated once** at construction (and re-derived, not evolved, on restart /
   AMR regrid); never touched in the per-step solve, which only reads them.
2. **Trumpet radius conversion + background generator** (new `cfc_puncture.{hpp,
   cpp}`, §3.6): port `mod_bh.f90`'s `tbh_areal_to_iso`/`tbh_iso_to_areal`
   (Illinois bracket root-find, no table/Newton) and `tbh_get_var`'s
   `ψ0, α0, β0^i, Â0_ij, Â0²` assembly as device-callable free functions.
   **Must accept arbitrary cell coordinates (any MG level), not just the finest
   mesh** — §3.8 calls it per level. Hardcode `M_BH` and origin `x=0`; no
   puncture-position argument, no tracker. Provide the wormhole slice (§3.2's
   earlier draft, kept only as a debug fallback — not from SACRA) behind a flag.
   **Unit-test against §3.2's asymptotic limits (already confirmed numerically for
   `χ(ρ)` itself this session) before wiring in.**
3. **Generalize the "deviation" convention** from `−1` / `about 0` to
   `− Q0`. **Required by §3.7's `x = δψ/ψ0` regularization**, which only makes
   sense if the iterated field is the deviation from `ψ0`. Concretely:
   - `delta_psi` now means `ψ − ψ0`, `delta_alpha_psi` means `αψ − α0ψ0`
     (rename to `u`/`v` or document clearly). Every `+1.0` reconstruction site
     (`AssembleConformalMetric`, `AssembleLapseShiftK`, `AssembleVectorSource`'s
     shift branch, `MGCFCLapseDriver::LoadReactionCoefficient`) becomes `+ ψ0`
     / `+ α0ψ0` (read from the new arrays). These sites are enumerated in
     `cfc.hpp:88-92`.
   - Shift stored/solved as `b^i = β^i − β0^i`; add `+ β0^i` at reconstruction.
4. **Per-level analytic puncture coefficients (§3.8) — the key MG-accuracy item.**
   In `MGCFCConformalFactor`/`MGCFCLapse`, store `ψ0`, `α0`, `Â0²`, `R_sch` as
   `coeff_` channels and **fill them by calling `cfc_puncture` at each level's own
   grid points**, replacing `RestrictCoefficients()` *for these channels* (matter
   channels `Ũ`, `S̃`, `ΔÂ²` still restrict as today). Precompute `R_sch` per level
   once via the Illinois solve; reuse for all `tbh_get_var` on that level.
   **Prereq: confirm AthenaK `Multigrid` exposes per-level cell coordinates**
   (add if missing). Mirrors `cfc_solve_psi`'s `do lvl = lowest_lvl, highest_lvl`
   loop.
5. **Rewrite the two nonlinear solvers with the §3.7 regularized form** — the
   highest-numerical-risk item; validate at the puncture cells specifically
   (Phase D). Port the exact expressions from `m_cfc_psi_tbh.f90::box_lpsi_tbh`
   / `box_gs_lpsi_tbh`:
   - **ψ solver** (genuinely nonlinear in `δψ`): in
     `MGCFCConformalFactor::{SmoothPack,CalculateDefectPack}` replace the `Â²ψ⁻⁷`
     handling with `−(1/8)ΔÂ²ψ⁻⁷ + (1/8)Â0²ψ0⁻⁷·x·P7(x)/(1+x)⁷` (`x=δψ/ψ0`), and
     the per-point Newton `dLdu` with the closed-form `(7/8)Â0²ψ0⁻⁸(1+x)⁻⁸`.
   - **Lapse solver** (LINEAR `Δv + K·v = rhs`; `box_gs_lalp_tbh` is a direct
     division, *no Newton* — verified): the `ψ⁻⁸−ψ0⁻⁸` cancellation is handled when
     `K` and `rhs` are *built*, not in the smoother. In `MGCFCLapse`'s coefficient/
     RHS assembly (and `cfc.cpp`'s `RescaleMatterSources`/`LoadReactionCoefficient`),
     put the regularized `+α0ψ0(7/8)Â0²ψ0⁻⁸·x·P8(x)/(1+x)⁸` background term into the
     RHS. **Use the explicit `P8` polynomial, not the init code's `A2p−A2(1+x)⁴`
     regrouping** (§3.7 — the polynomial fully removes a mild cancellation the
     regrouping leaves in). The smoother's diagonal is just `d2f + K`, unchanged in
     structure from `box_gs_lalp_tbh`.
6. **Add `Â0^ij` into step 2** (`ComputeADualFromX` in `cfc_reconstruct.cpp`):
   `Â = Â0 + (matter part)`; form the three `Â²` pieces (`Â0²` analytic, cross
   `2Â0·Âs`, self `Âs²`) so **both steps 3 (ψ) and 5 (lapse)** can treat only `Â0²`
   with the cancellation regularization and lump the rest into `ΔÂ²` (§3.7) — this
   split is shared by both nonlinear solvers, not lapse-only.
7. **Shift solver — source assembly only (§3.9), NO operator changes and NO special
   regularization** (the easy one). The flat vector-Laplacian (`MGCFCVectorPoisson`,
   Shibata `P_i`/`η`) is reused as-is, including its existing generic (non-`coeff_`)
   restriction — deliberately not given per-level analytic treatment (§3.9's new
   MG-level note: `Â0`/`β0`/`α0ψ0⁻⁶` are bounded, unlike `ψ0`, so this is acceptable).
   In `AssembleVectorSource(for_shift=true)`/`BuildShiftSource` (`cfc.cpp`), compute
   the residual `Â`-gradient source as the **three deviation cross-terms directly**
   (`2Â0^ij∇_j[Δ(αψ⁻⁶)] + 2Âs^ij∇_j(α0ψ0⁻⁶) + 2Âs^ij∇_j[Δ(αψ⁻⁶)]`, §3.9) — **not**
   "add an analytic background field, then separately subtract it": that framing
   invites a double-counting bug; the background piece is only an ingredient
   (`Â0` itself, and the closed-form `∇_j(α0ψ0⁻⁶)` factor via `dα0`,`dψ0`) inside two
   of the three terms, never a standalone additive term. Build `Δ(αψ⁻⁶)`
   cancellation-free via §3.7's `ψ0⁻⁷(1+x)⁻⁷` factoring, then FD it ordinarily (it's
   smooth/bounded). Add `β0^i` at reconstruction. Matter term
   `16πα ψ⁻⁶S̃_i = 16πα S_i` needs nothing. **No `dalppsi`/`dalppsi0` log-gradient
   machinery** (that was CTS-only).
   *Bring-up check:* **the actual trumpet formulas themselves** — not just the
   wormhole fallback — must degenerate correctly as `M_BH→0`: verified this session
   that `R_sch→r_iso`, hence `ψ0,α0→1` and `Â0²,β0→0` (§3.2's forward map already
   confirms `R_sch→r_iso`); this collapses every term in items 4-7 to the current
   flat-background code, reproducing the validated TOV result. The wormhole mode
   (`Â0=β0=0` identically, at *any* `M_BH`, not just `M_BH→0`) is a useful
   *additional*, simpler independent cross-check — but the primary bring-up gate
   should exercise the real trumpet code path, since that's what actually ships.
8. **Outer BC recentering (§3.10) — two separate changes, different scope.**
   (a) **Shared-module change**: add a settable center to `mg_robin`
   (`multigrid_driver.cpp`'s two ghost-fill sites, root + `PhysicalBoundary`) and
   subtract it from `r_a,r_g`; wire it to a new CFC-local, per-stage reduction over
   `pmy_pack->pmhd->u0(IDN)` (mirroring `CalculateCenterOfMass()`'s reduction shape,
   §3.10 item 1 — not a fix to that function itself), shared by ψ and αψ.
   (b) **CFC-local-only change**: add `MGCFCVectorPoissonDriver::SetMultipoleOrigin
   (x,y,z)` (assigns the inherited `mpo_`, already-existing machinery) and call it
   **twice per stage, with two independently-computed centers** — `r_com_X`
   (weighted by `X^i`'s own raw `p_src` = `S̃²`, undensitized by dividing out `ψ¹²`
   before squaring) before the `X^i` solve, and `r_com_β` (weighted by `β^i`'s own
   fully-assembled `p_src` — matter + `Â`-gradient piece, no `ψ¹²` correction
   needed since it carries no bare conserved-density factor) before the `β^i` solve
   — **not** a single shared value (§3.10 item 2, corrected per user: the two
   equations' actual source arrays are not proportional, so their natural centers
   are computed, and may differ, independently). `r_com_β` can only be computed
   *after* `BuildShiftSource` has finished assembling that stage's `p_src`.
   (c) **Correctness-critical**: shift `η`'s source (`cfc.cpp:879-888` and the
   `β^i` equivalent) *and* `ReconstructVectorFromPotentialsImpl`'s cross-term
   (`cfc_reconstruct.cpp:91,97`) by *that same solve's* `r_com` — shifting only
   one, or using the wrong solve's `r_com` in the other, silently reconstructs the
   wrong `V^i` (§3.10 item 3, verified by re-deriving the Shibata identity's
   shift-invariance).

### Phase B — matter & BH interaction near the puncture
9. **Excision / atmosphere at the puncture — concrete mechanism identified, not
   just "reuse existing excision" (this review's resolution of a real gap: an
   earlier draft said "mask a small sphere" with no size/criterion given).**
   `src/coordinates/coordinates.cpp`/`excision.cpp` already implement **three**
   selectable schemes via `<coord> excision_scheme`: `fixed` (a coordinate radius
   `rexcise`, compared against the Kerr-Schild-like `r_ks = KSRX(x1,x2,x3,bh_spin)`),
   `lapse` (excise wherever `padm->adm.alpha < excise_lapse`, default `0.25`), and
   `horizon` (apparent-horizon-tracker-based, needs `<fastflow>`). All three work for
   a CFC-only run (`is_dynamical_relativistic` is true whenever `<adm>`/`<z4c>`
   exists, which gates this code path; `excise_lapse` reads the generic ADM `alpha`
   array CFC's `AssembleLapseShiftK` already writes).
   - **Two findings that resolve the choice:**
     (a) For **non-spinning** (`bh_spin=0`, this project's locked decision, §6),
     `KSRX` reduces *exactly* to `sqrt(x1²+x2²+x3²) = r_iso` (verified by direct
     substitution: the `a→0` limit of `KSRX`'s formula collapses to the bare
     Euclidean radius) — so the `fixed` scheme's `rexcise` is directly usable in
     our own `r_iso` convention with zero coordinate-conversion risk, *if* a fixed
     radius is wanted. (b) **Better fit: `lapse`**. A maximal-slicing trumpet's
     `α0(r_iso)` is *already* the natural "how deep inside the horizon" indicator
     (monotonic, `→0` at the throat) — recommend `excision_scheme=lapse` over a
     hand-picked fixed radius, since it needs no bespoke tuning to our specific
     background and adapts automatically to the (small) residual `v`.
   - **Numerically checked** (this session) that the existing default
     `excise_lapse=0.25` is already a sensible, safely-inside-the-horizon choice
     for our trumpet: `α0=0.25` at `ρ≈1.85` (`r_iso≈0.59 M_BH`), comfortably inside
     the horizon `ρ=2` (`α0≈0.32`, `r_iso≈0.78 M_BH`) — no urgent need to retune,
     though this should be revisited once real runs exist.
   - **Ties back to §3.10's COM integrals**: the mass/momentum-weighted centroids
     integrate `u0(IDN)`/`u0`'s momentum components over the *whole* domain,
     including near-excised cells. This is fine *because* the excision floor values
     (`dexcise`/`texcise`/`pexcise`) are deliberately set far below the WD's own
     density — their contribution to those integrals is negligible by construction
     of the floor, not because of any special handling in the COM computation
     itself. Worth an explicit check in Phase D, not just an assumption.
   - Remaining item: with §3.7 the metric-solver terms near `x=0` are themselves
     now well-conditioned (that was a separate, already-resolved concern) — the
     excision scheme above is purely about the *fluid/con2prim* treatment near the
     horizon, a distinct concern from the metric regularization.
10. **Initial data**: extend `InitializeMetric()` to iterate the residual about
   `Q0` (the analog of `solve_cfc`'s outer fixed-point loop over ψ/α/β, iterated
   against con2prim); place the WD (existing TOV/WD equilibrium pgen) with an
   orbital boost. New pgen `src/pgen/tests/tde_wd.cpp` (or similar) setting matter
   + calling the trumpet generator.

### Phase C — dynamic range (needed for a *realistic* TDE, not the first test)
11. **AMR support in the nonlinear CFC solvers** — currently fatal-errors for
   `nreflevel_ > 0` (DEVELOPMENT.md item 3b: extend `MGOctet` with `coeff_`
   storage + restriction/boundary paths). The WD-to-BH scale separation makes
   this effectively mandatory for production runs; the first proof-of-concept can
   use a uniform grid with a small `M_BH` / wide WD to keep ratios modest.
   **Couples to §3.8**: the per-level *analytic* puncture coefficients must also be
   filled on refined octets (call `cfc_puncture` at octet cell coordinates), so
   solve this together with item 3b — the analytic-per-level fill is actually
   *easier* on octets than restriction would be (no inter-level averaging of the
   peaked `ψ0`), once octet coefficient storage exists.

### Phase D — validation
12. `M_BH → 0` (wormhole fallback) reduces to the validated flat-background TOV
    result (Phase A #7 bring-up check).
13. **Puncture regularization check (§3.7/§3.9):** single static BH + no matter ⇒
    residuals stay `~0` (background is a CFC fixed point); check `α, ψ, β` reproduce
    the analytic trumpet to solver tolerance, and **specifically inspect the cells
    nearest `x=0`** — this is where the naive `ψ⁻⁷−ψ0⁻⁷` would have produced garbage,
    so it is the decisive test that the `x=δψ/ψ0` factoring works (the shift source's
    sharp part is supplied analytically, so it should be clean there by construction).
    Also confirm V-cycle convergence does not degrade with resolution (tests §3.8's
    per-level analytic coefficients).
14. **Outer BC recentering check (§3.10):** move the WD to a static offset from the
    BH (no orbit yet) and confirm the solved `u,v,b^i` fall off cleanly around the
    *WD's* position rather than showing a spurious origin-centered asymmetry;
    compare boundary-error convergence with and without recentering at fixed domain
    size, and confirm the mass-weighted `r_com` (ψ/αψ) and the two independently
    computed `r_com_X`/`r_com_β` (§3.10 item 2, corrected: separate per-equation
    sources, not a shared momentum proxy) all agree to good approximation for a
    non-disrupted star — a three-way cross-check, not just two.
15. BH + static low-mass star at large separation ⇒ compare to Cowling/Newtonian
    tidal field; then a full disruption vs the 2022 paper's debris/fallback
    curves.

---

## 6. Decisions locked / remaining questions

**Locked (per user):**
- **Trumpet background**, maximal slicing (§3.2). Wormhole kept only as a debug
  fallback.
- **Puncture fixed at `x = 0`** for all time ⇒ background is static + spherically
  symmetric, computed once (§3.6). No moving puncture, no tracker, no per-step
  `Q0` regeneration.
- **Non-spinning** ⇒ spatial slice exactly conformally flat, consistent with CFC.
  No Bowen–York needed.
- **Exact background formulae + radius conversion**: ported directly from
  `~/SACRA_2D/SACRA_MPI/mod_bh.f90` (`tbh_get_var`, `tbh_areal_to_iso`,
  `tbh_iso_to_areal`), not re-derived from the paper. This resolves what was the
  highest-risk piece (silent sign/constant errors); the forward map was
  additionally numerically re-verified against the expected asymptotics this
  session (§3.2).
- **Fluid excision mechanism resolved** (this review pass): use the already-
  implemented `<coord> excision_scheme=lapse` (§5 Phase B #9) rather than a
  bespoke radius — verified `bh_spin=0` makes the existing `fixed` scheme's `r_ks`
  coincide exactly with our own `r_iso` as a fallback option, and confirmed the
  default `excise_lapse=0.25` already sits safely inside the horizon for the
  trumpet background.
- **Shift-source algebra double-checked** (this review pass, §3.9): re-derived
  from the exact (unregularized) residual equation that the analytic background
  term `2Â0^ij∇_j(α0ψ0⁻⁶)` cancels identically against itself — the final
  residual source is exactly three deviation cross-terms, no separate additive
  background term. The document's earlier phrasing risked being misread as two
  separate, non-canceling steps; rewritten to remove that ambiguity.

**Still worth confirming:**
- **Does CFC's dropped GW / non-flat conformal metric matter for the science
  target?** For fallback rate / debris dynamics it should be negligible; state as
  an explicit caveat in the paper.
- **Timestep**: CFC-only runs currently use the light-crossing CFL bound
  (deliberately, see DEVELOPMENT.md + memory note); with a `1e5 M_sun` SMBH light
  radius this may be expensive — flagged, not changed.

---

## 7. References
- **`~/SACRA_2D/SACRA_MPI/mod_bh.f90`** — the primary, vetted source for §3.2/§3.6:
  `tbh_get_var` (`ψ0, α0, β0^i, Â0_ij, Â0²` and their radial derivatives),
  `tbh_areal_to_iso` (forward map `χ(ρ)`), `tbh_iso_to_areal` (Illinois-bracket
  inverse). Ported directly rather than re-derived.
- **`~/NSWD/octree-mg/src/m_cfc_psi_tbh.f90`** — vetted source for §3.7 (ψ): the
  `x=δψ/ψ0` catastrophic-cancellation regularization of `ψ⁻⁷−ψ0⁻⁷`
  (`box_lpsi_tbh`, the `P7`/`(1+x)⁷` form) and its Newton–Gauss–Seidel Jacobian
  (`box_gs_lpsi_tbh`, the `(1+x)⁻⁸` form).
- **`~/NSWD/octree-mg/src/m_cfc_alp_tbh.f90`** — the lapse operator/smoother:
  confirms the lapse is *linear* (`Δv + K·v`, `box_lalp_tbh`) with a direct-division
  GS smoother (`box_gs_lalp_tbh`, no Newton). Its `ψ⁻⁸−ψ0⁻⁸` regularization is built
  in `cfc_solve_alp` (init file) via the `A2p−A2(1+x)⁴` regrouping — §3.7
  recommends the explicit `P8` form instead.
- **`~/NSWD/octree-mg/src/m_cts_beta_boost_v2.f90`** + `cfc_solve_betam_v2` (init
  file) — reference for §3.9 (shift). Shows the CTS operator folds `2Â^ij∇_j(αψ⁻⁶)`
  into the elliptic operator (upwinded, `β`-coupled, boosted) — which CFC avoids
  entirely since `Â` comes from `X^i` (explicit source). CFC also does *not* need the
  CTS `dalppsi`/`dalppsi0` log-gradient machinery: the sharp background piece
  `2Â0^ij∇_j(α0ψ0⁻⁶)` is supplied analytically (regular), and the deviation is smooth.
- **`~/NSWD/init_TBHWD_headon.f90`**, `solve_cfc` and the `cfc_solve_*` routines it
  calls — vetted source for §3.8: the per-level (`do lvl = lowest_lvl,
  highest_lvl`) analytic evaluation of `ψ0` and the puncture coefficients via
  `tbh_get_var` at each MG level (`cfc_solve_psi`), instead of restriction; also
  the outer ψ/α/β fixed-point loop pattern for initial data (§5 Phase B #10).
- **AthenaK's own `src/multigrid/multigrid_driver.cpp`** — for §3.10 (outer BC
  recentering): `CalculateCenterOfMass()` (lines 2733-2792, the reduction-pattern
  template to mirror, not fix) and the `mg_robin` ghost-fill kernels (lines
  2125-2230, 2433-2513, confirming no settable center exists today). `src/cfc/
  mg_cfc_vector_poisson.cpp:184-194` documents the "every current CFC test problem's
  star sits at the coordinate origin" assumption this project invalidates.
  `src/cfc/cfc_reconstruct.cpp:91,97` and `src/cfc/cfc.cpp:879-888` confirm both the
  reconstruction cross-term and `η`'s source use the unshifted origin today.
- Lam, Shibata, Kiuchi 2023 (arXiv:2212.10891) — the TDE background-split /
  puncture / error-correction formulation this re-implements.
- Cheong et al. 2021, "Gmunu" (arXiv:2012.07322) sec. 2.6, eqs. 71–76 — the XCFC
  equation set the module already solves.
- Shibata 1999 (arXiv:gr-qc/9905058) sec. 3 — vector-Laplacian decomposition.
- `src/cfc/DEVELOPMENT.md` — module status, solve order, known limits (esp. the
  AMR/`MGOctet` gap, item 3b).
- Baumgarte & Naculich 2007 (arXiv:0709.0299) — underlying stationary
  maximal-slicing trumpet Schwarzschild theory that `mod_bh.f90` implements.
- **AthenaK's own `src/coordinates/{coordinates,excision}.cpp`** — for §5 Phase B
  #9 (fluid excision): the three existing `excision_scheme` options (`fixed`,
  `lapse`, `horizon`; `coordinates.cpp:56-101`), `KSRX`'s `a→0` reduction to the
  bare Euclidean radius (`excision.cpp:20-25`), and `lapse`'s direct read of
  `padm->adm.alpha` (`excision.cpp:182`), confirming it works for a CFC-only
  (no `<z4c>`) run.
