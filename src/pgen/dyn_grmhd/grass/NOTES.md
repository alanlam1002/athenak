# Magnetic-web + dipole implementation notes

Answers to the prerequisite questions of the web spec §1
(`Athenak_magnetic-web/research/spec/magnetic_web_id_athenak.md`) and to the design's
§8 questions (`research/spec/web_field_design_P_CFC.md`), for star P
(uryu_inc p1 λ2, Mb 3.0, J 1.5, DD2-hc, `res.rst` md5 e8cba651).

**Revised 2026-10-26.** The earlier version (2026-08) said that `b0` is *not*
densitized. That was wrong (§3 below); it was based on the curl kernel having no
metric factors, which says nothing about what the evolution treats `b0` as. All
magnetic diagnostics and normalizations were re-written accordingly in the same
commit.

## 1. How is rho obtained? Can it be evaluated at the staggered edges?

Preference 1 of spec §4.3 is available. `grass::GrassData::Interpolate()`
(`grass_reader.hpp`) is a host-side Lagrange-stencil interpolator over GRASS's
(s, μ) restart grid, callable at any Cartesian (x, y, z). `InterpolateRho()` is the
rho-only variant: it interpolates GRASS's `energy` (and `enthalpy`, which gates the
surface) and maps energy to n0 through GRASS's own EOS table (`grass_eos_table.hpp`),
rho0 = n0 m_b. The cell-centre `w0(IDN)` is filled through the same path, so the edge
and cell-centre densities are the same function.

`BuildMagneticField()` calls `InterpolateRho()` at each of the three edge staggerings
(`a1` at (x1v, x2f, x3f), `a2` at (x1f, x2v, x3f), `a3` at (x1f, x2f, x3v)) and at
the x±dx/4 half-cell points of the AMR correction. No cell-centre averaging or
trilinear interpolation of `w0` is used anywhere.

## 2. rho_max, surface density and atmosphere floor for P (code units: G = c = Msun = 1, rho unit 6.1760e17 g/cm^3)

| quantity | code units | cgs |
|---|---|---|
| rho0_c = rho_max (GRASS) | 1.9430e-3 | 1.19998e15 g/cm^3 |
| AthenaK cell-centre rho_max at t=0 (C-023) | 1.9422e-3 | 1.1995e15 |
| window rho_lo / rho_hi (design §2) | 4.8575e-4 / 1.7487e-3 | 3e14 / 1.08e15 |
| h peak (u = 0), sqrt(rho_lo rho_hi) | 9.217e-4 | 5.69e14 |
| GRASS surface (enthalpy cut) | rho0 → 0 | equator 11.725 km = 7.94 M (= GRASS r_e, so r_e is a coordinate radius); pole 10.48 km |
| rho0 = 1e10 g/cm^3 | 1.62e-8 | equator 9.79 km (6.63 M), pole 9.14 km |
| atmosphere floor `<mhd> dfloor` | 2.8e-15 | 1.73e3 g/cm^3 |

- rho_lo / dfloor = 1.7e11, far above the spec's "≥ 3 orders".
- The pgen writes rho0 = 0 outside GRASS's surface. The floor is applied by
  AthenaK's primitive-solver flooring at the first C2P, not by the pgen.
- `h(rho)` returns 0 for rho ≤ 0, so it never takes the log of zero.
- The window radii on P (from the reference profiles; the pgen prints its own sweep):
  - rho_hi at 1.85 km (equator) and 1.82 km (pole);
  - rho_lo at 7.47 km = 5.05 M (equator) and 7.17 km (pole).

## 3. What does `b0` store? Densitized: Btilde^i = sqrt(gamma) B^i

Evidence (`src/eos/primitive_solver_hyd.hpp`):
- `PrimToCons` builds the conserved variables with `b = bcc/sdetg` (lines 249-251).
- `ConsToPrim` uses `b3u = bcc0*isdetg` (lines 406-417). It rebuilds `bcc0` from the
  face values `bfc` = `b0` (lines 412-414) and then divides by sqrt(gamma) again.
- `dyn_grmhd.cpp` (SetTmunu, line 467ff) and `cfc.cpp`'s `RescaleMatterSources`
  (line 1320ff) both use `bcc(...)*ivol` with ivol = 1/sqrt(gamma).

So `b0`/`bcc0` hold the densitized field. This is also what CT preserves: the discrete
∂_i Btilde^i = 0. The plain Cartesian curl of the edge potential A_i (no metric
factors, as in `lorene_bns.cpp`) therefore produces Btilde directly, with div = 0 to
round-off. The pgen matches `lorene_bns.cpp` exactly and never inserts sqrt(gamma)
into the evolved arrays.

Consequences:
- **Diagnostics** convert to the physical Eulerian field B^i = Btilde^i/sqrt(gamma)
  with the cell's own g_dd. Energy density is g_ij B^i B^j / 2 (code units are
  Heaviside-Lorentz), and the volume element is sqrt(gamma) d^3x.
- **Components** along the flat unit directions e are g(B, e)/sqrt(g(e, e)).
- **Where this applies:** `GrassPhysicalB()` in `dyngr_grass.cpp` is used by the
  normalization reductions, the startup report and `GrassHistory`.
- **What the old code got wrong:** before this revision, all of these treated `bcc0` as
  B. In a conformally flat metric the physical |B| = psi^-4 |Btilde|, so the old
  "max|Btilde| = target" left the physical field a factor psi^4 below the target
  (psi^4 ≈ 2-3 in P's core).
- **Helicity:** H = ∫ A_i Btilde^i d^3x (= ∫ A·B sqrt(gamma) d^3x). It needs no metric
  factor.

## 4. Mesh layout and the a3 correction

The production grid (`grass_P_B0_cfc`, C-022) is:
- root 256^3 over ±180 M (Δx = 1.40625 M), 32^3 meshblocks;
- static refinement with `num_levels = 5` and one `<refined_region1>` at level 4 over
  ±10 M. The finest level has Δx = 0.08789 M = 130 m, with the intermediate levels for
  2:1 nesting.

The window (r ≤ 5.05 M on the equator) lies wholly inside the level-4 box. The field
therefore never reaches a fine/coarse boundary on the production grid, but the
correction is still required for generality, and for the 260 m comparison grid, whose
box is the same in M.

`BuildAndCorrect()` applies the `lorene_bns.cpp` a1/a2 corrections verbatim. It adds the
a3 correction: an average over x3 ± dx3/4 on x1-faces (neighbours 0-7), x2-faces (8-15)
and x1x2-edges (16-23), from `mesh/nghbr_index.hpp`. This was verified at the time:
div B = 1.7e-15 under 2-level SMR (VALIDATION.md V1/V7).

## 5. Is Y_l advected?

Yes:
- `<mhd> nscalars = 1` with `dyn_eos = compose` on the 3D table `DD2_hot_trapped_3d.athtab`.
- The pgen seeds the passive scalar from the 1D slice table's Y_l(nb) (`tov_tabulated`;
  the seeding bug was fixed in d5e09cb4, C-009).
- The evolved EOS is not barotropic: P(rho, T, Y_l) from the 3D table, with Y_l
  advected and T recovered from (rho, P, Y_l) at t=0. The composition stratification
  is therefore carried by the evolution.

## 6. `lorene_bns.cpp`, read end to end

What is reused:
- the edge allocation and staggering;
- `CellCenterX`/`LeftEdgeX`;
- the discrete curl with the extra faces at ie/je/ke;
- the face-average `bcc0`;
- the a1/a2 AMR corrections;
- the current-loop `A1`/`A2` (as `DipoleA1`/`DipoleA2`, center = 0, a3 ≡ 0).

What differs:
- **Gauss conversion:** `lorene_bns.cpp` hard-codes `b_max/8.3519664583273e19`.
  `GaussToCode()` derives the same factor from the `<mhd> units` system: sqrt(energy
  density conversion/4π) = 1/8.3520e19 for geometric_solar.
- **Normalization:** `lorene_bns.cpp` uses max|B| on its own (non-densitized-aware)
  bcc. Here every normalization uses the physical field (§3).
- **Ordering:** after the field, `PrimToConInit` builds the conserved variables
  (including B); under `<z4c>` `ADMToZ4c` follows, as in the reference.

## 7. Design §8: does the CFC metric source include the EM terms?

Yes, all three:
- **E** (Hamiltonian/psi source): `u_tilde = cons(IEN) + cons(IDN)` (cfc.cpp:1140) is
  sqrt(gamma) (tau + D) = sqrt(gamma) E. `PrimToCons` builds tau with the magnetic
  energy.
- **S_i** (shift source): `cons(IM1+a)` (cfc.cpp:1162) is the full conserved momentum,
  Poynting term included.
- **S** (lapse source): `RescaleMatterSources` (cfc.cpp:1280-1350) rebuilds S_ij with
  the magnetic stress, -(B_i + (B·v) v_i)B_j/W^2 + (P + b^2/2) g_ij, and takes the trace.

## 8. Normalization options (design §3-4)

| key | values | meaning |
|---|---|---|
| `web_norm` | `bmax` (spec default, `web_bmax_gauss`) / `rms` (`web_brms_gauss`) | rms of physical \|B_web\| over cells with h(rho_cell) ≥ 0.5, proper-volume weighted |
| `dipole_norm` | `bmax` (`dipole_bmax_gauss`) / `bvarpi_rms` (`dipole_bvarpi_rms_gauss`) | rms over h ≥ 0.5 of the azimuthal mean of the physical cylindrical-radial component, B̄_varpi |
| `grass_t_over_w` | GRASS T/\|W\| | lets the report give E_mag/\|W\| with \|W\| = T/(T/\|W\|), T = ½∫Ω dJ on the AthenaK grid |

How B̄_varpi is computed:
- a proper-volume-weighted mean over (varpi, z) bins of width Δx_finest, covering
  1.25× the window's outer radius;
- the same bins give the t=0 map `<basename>.web_map_t0.txt`.

The tor/pol quadratic (spec §3) uses the physical B_phi and |B|^2 - B_phi^2.

## Design choices (unchanged from 2026-08)

- **Host-side construction:** the GRASS interpolator is host-only. The potentials are
  built on the host and copied to the device for the curl and the reductions.
- **One builder:** a single `BuildAndCorrect` serves the web P pass, the T pass and the
  dipole, so the boundary-correction logic exists once.
