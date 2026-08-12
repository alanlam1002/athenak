# Magnetic-web + dipole implementation notes

Answers to `magnetic_web_id_athenak.md` (repo root) section 1's prerequisite
questions, as investigated and confirmed this session before writing any code.

## 1. How is rho obtained?

`grass::GrassData::Interpolate()` (`grass_reader.hpp`) is the density source —
a host-only Lagrange-stencil interpolator over GRASS's restart binary, callable
at arbitrary Cartesian `(x,y,z)`, including exactly the edge-staggered
coordinates the vector-potential construction needs (no cell-centered-only
limitation). It returns the full ADM+hydro `Point` struct.

A slimmed variant, `GrassData::InterpolateRho(x,y,z,eos_table)`, was added
specifically for this feature — it shares the stencil-*location* logic with
`Interpolate()` via a factored-out `LocateStencil()` helper (that logic is
exactly where two real indexing bugs were found and fixed earlier this
session; keeping it in one place matters), but evaluates only the two fields
(`energy`, `enthalpy`) needed to recover `rho0`, skipping the metric/Jacobian/
velocity construction `Interpolate()` also does. This matters because the
web/dipole construction calls it *far* more often than the ADM/hydro fill loop
calls `Interpolate()` — three separate edge-staggered grids (`a1`, `a2`, `a3`),
each evaluated at least twice (poloidal pass, toroidal pass), plus AMR
boundary corrections.

## 2. `rho_max` / atmosphere floor, in code units

No existing accessor exposed `rho_max` — `BuildMagneticField()` (`dyngr_grass.cpp`)
adds a one-time device `parallel_reduce` + genuine `MPI_Allreduce(MAX)` (not
`GrassHistory`'s rank-0-only hack, since `web_rho_hi <= rho_max` must be
validated identically on every rank) over the already-populated `w0(IDN)`
array. This is diagnostic/validation-only, not load-bearing to the field
construction itself (`web_rho_hi` is a direct user input).

Atmosphere floor: `<mhd> dfloor` (e.g. `2.8e-15` code units in the test inputs,
derived from the runtime EOS table's own native `nb` floor — see
`/u/tlam/GRASS/docs/DD2_hot_equal_pinned_investigation_notes.md`) is applied
post-hoc by AthenaK's primitive-solver flooring, **not** by `SetupGrass`'s host
loop itself, which writes literal `rho0=0.0` for exterior cells
(`dyngr_grass.cpp`, gated on `pt.rho0>0.0`). `WebEnvelope()`/`h(rho)` guards
`rho<=0` explicitly (never calls `log()` of a non-positive value) before any
floor is applied, so this distinction doesn't matter for the field's
correctness, but is worth recording since it's easy to assume `dfloor` is
already reflected in `w0` at the point `BuildMagneticField` reads it (it is,
since `Kokkos::deep_copy(w0, host_w0)` runs before `BuildMagneticField` is
called, and AthenaK's own C2P/flooring machinery, not this pgen, is what
would enforce `dfloor` on any *evolved* state — the pgen's own initial `w0`
fill is unaffected either way since GRASS's own zero-outside-star convention
already sits well below any sane `dfloor`).

## 3. `b0` densitization convention

Confirmed directly from `src/pgen/dyn_grmhd/lorene/lorene_bns.cpp`'s own curl
kernel (`pgen_Bfc`, lines ~447-475): plain Cartesian curl, `dx1/dx2/dx3` are
bare coordinate cell widths (`size.d_view(m).dx1` etc.) — **no** `sqrt(gamma)`
or any other metric factor appears anywhere in the curl. `b0` therefore stores
the plain (non-densitized) Eulerian `B^i`, not `sqrt(gamma)*B^i`. This
implementation matches that exactly — `sqrt(gamma)` (via `adm::SpatialDet`)
is used **only** for the diagnostic-integral volume element
`dV = sqrt(gamma)*dx1*dx2*dx3` (the tor/pol normalization reduction, `E_mag`/
`E_tor`/`E_pol` in both `BuildMagneticField` and `GrassHistory`), never fed
back into the evolved `b0`/`bcc0` arrays.

## 4. SMR/AMR status

Confirmed this session: production GRASS runs use **static AMR**
(`<mesh_refinement> refinement=static`, up to 5 levels, nested
`<refined_regionN>` blocks — the exact configuration that originally crashed
this pgen twice before those bugs were fixed). The `a3` fine/coarse
vector-potential correction (below) is therefore **mandatory**, not optional,
for this pgen — confirmed by an actual static-AMR test
(`dyngr_grass_web_amr_test.athinput`, 216 MeshBlocks, 2 physical levels)
achieving `max|div(B)|*dx/|B| = 1.68e-15` (see `VALIDATION.md`, V1/V7).

`lorene_bns.cpp`'s own dipole never needed an `a3` correction because its
dipole's vector potential has `a3 ≡ 0` identically (a pure current loop in the
x-y plane has no z-component of `A`) — the web's Chandrasekhar poloidal-
toroidal potential does not share that property (`Az` is generically
nonzero), so the correction had to be derived from scratch this session, by
direct analogy to `lorene_bns.cpp`'s existing `a1`/`a2` corrections and
verified against `src/mesh/nghbr_index.hpp`'s documented neighbor-index
blocks (`x1faces:[0-7], x2faces:[8-15], x1x2edges:[16-23]` for the `a3` case)
before being written.

## 5. `nscalars` / Y[e] status

`<mhd> nscalars=1`, `dyn_eos=compose` (3D trapped-neutrino DD2 table) were
already wired in an earlier phase of this pgen's development, seeding the
passive scalar `Y[e]` from a separate 1D slice table. Entirely unrelated to,
and unaffected by, this magnetic-field work.

## 6. `lorene_bns.cpp` pattern reused/extended

Read in full this session (`src/pgen/dyn_grmhd/lorene/lorene_bns.cpp`).
Reused/extended:
- Edge-staggered `DvceArray4D<Real> a1,a2,a3` allocation and `CellCenterX`/
  `LeftEdgeX` staggering convention (`a1` at `(x1v,x2f,x3f)`, `a2` at
  `(x1f,x2v,x3f)`, `a3` at `(x1f,x2f,x3v)`), filled over the staggered range
  `is..ie+1, js..je+1, ks..ke+1`.
- The discrete-curl `pgen_Bfc`/cell-centered-average `pgen_bcc` kernel pair,
  including the "extra face at edge of block" terms — reused essentially
  verbatim (`CurlToBcc` lambda / the final curl block in `BuildMagneticField`).
- The AMR fine/coarse vector-potential correction pattern (average over
  `x_center ± 0.25*dx` on the correction axis, gated by neighbor-level checks
  via `nghbr`/`mb_lev`) — reused for `a1`/`a2` (same neighbor-index blocks),
  extended (new, this session) for `a3`.
- The Gauss-to-code-unit conversion: initially reused `lorene_bns.cpp`'s
  hardcoded `8.3519664583273e+19` constant, which silently assumed `<mhd>
  units=geometric_solar`. Replaced with `GaussToCode(pin)` in
  `dyngr_grass.cpp`, derived from `Primitive::UnitSystem::
  EnergyDensityConversion` and honoring the actual `<mhd> units` setting, the
  same key/options `PrimitiveSolverHydro::SetPolicyParams` already reads.
  Matches the old constant to ~4-5 significant figures under the default
  `geometric_solar` (the sub-permille residual is `lorene_bns.cpp`'s own SI
  constants vs. this codebase's CGS ones, not a bug).

**Not reused**: `lorene_bns.cpp`'s current-loop `A1`/`A2` functions themselves
*were* ported verbatim into `grass_magnetic_web.hpp` as `DipoleA1`/`DipoleA2`
(with `center=0`, one star at the origin, vs. Lorene's two-star `center_m`/
`center_p`).

## Design choices made this session (not spelled out verbatim in the spec)

- **Host-side construction throughout.** `GrassData::Interpolate`/
  `InterpolateRho` are host-only; porting them to device would be a real
  re-architecture risk for a one-time startup cost that doesn't need device
  speed. Vector potentials are built as host arrays, `deep_copy`'d to device
  only for the curl/reduction steps (which *are* pure arithmetic, safe on
  device, and reused directly from `lorene_bns.cpp`'s own device kernels).
- **Unified `BuildAndCorrect` helper**, parameterized by per-component
  evaluator closures (`std::function<Real(Real,Real,Real)>`), shared by the
  web's poloidal pass, toroidal pass, *and* the dipole. The alternative
  (three separate hand-written build+correct blocks) would have tripled the
  surface area for exactly the kind of boundary-indexing bug this session
  already hit twice elsewhere in this pgen.
- **`E_mag/|W|` uses a rest-mass-energy proxy** (`M_rest = integral(rho0) dV`),
  not a true gravitational binding-energy calculation — flagged explicitly in
  the startup print, not silently presented as the real thing.
- **Helicity (`H`) is NOT implemented** in `GrassHistory` — see
  `VALIDATION.md` for the explicit scope decision.

## Port from a parallel (CFC) development line

`GaussToCode(pin)`'s `<mhd> units`-aware conversion, the `SetupGrass<UseYe>`
compile-time dispatch, the `Kokkos::` (vs. `std::`) math swaps, and
`GrassHistory`'s `ang-mom` (`J_z`) column were all developed on this repo's
`cfc` branch (which extended this same pgen to *also* run under a CFC/`<adm>`
metric solver) and ported back here. Only the CFC-agnostic pieces were
ported — the `cfc` branch's actual dual z4c/CFC dispatch (metric
isotropization, Lorentz-factor-preserving velocity rescale, the
`is_dynamical_relativistic`-based guard) was deliberately **not** brought
over: this branch has no `cfc` module compiled in, and `<z4c>` remains the
only supported metric evolution here (`UserProblem`'s `pz4c==nullptr` guard
is unchanged). The `Kokkos::` math swap inside `GrassHistory`'s/
`BuildMagneticField`'s device `KOKKOS_LAMBDA` bodies (`max|div B|` reduction)
is more than style — `std::cbrt`/`std::abs` are not device-callable under a
GPU (CUDA/HIP) Kokkos backend, so the pre-port code had a latent GPU-build
bug that never showed up under the CPU serial/OpenMP backends used so far.
