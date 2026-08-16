# Curvilinear Coordinates — Development Notes

This file tracks implementation progress against the approved plan for adding
runtime-selectable curvilinear coordinate support to AthenaK. It is a living
document: update the status line for a task as soon as it lands, and
**audit the whole file (not just the new section) for cross-stage
contradictions and stale `file:line` citations before every commit** — line
numbers drift as earlier phases land.

Plan file (frozen record of what was approved, **v2**):
`/u/tlam/.claude/plans/go-to-athenak-curvilinear-right-sparkling-pizza.md`

## Current state (as of 2026-08-16)

**Done and verified**: the ENTIRE v2 plan, Phase 0 through Phase G
(T0, A1-A5, B1-B7, C1-C2, D1-D3, E1-E2, F1-F3, G1). All architecture,
reconstruction, geometric source terms, constrained transport, origin
boundary handling, end-to-end problem generators, and the SR extension are
implemented and tested. **254/254 tests passing, 0 failures, 15 skipped**
— full suite re-run solo on a dedicated compute node (see "Testing on a
shared cluster" below) after G1 landed and its regression was fixed.

NOTE on those 15 skips: an earlier version of this file described them as
"GPU/MPI-only, correctly excluded on this no-CUDA node". That was wrong, and
it mattered, so it is corrected here rather than silently edited. `pytest -k
_cpu` DESELECTS the 18 `_gpu` test files; deselected and skipped are different
pytest categories and the deselected count was never reported at all. The 15
actual skips are in-test `pytest.skip()` calls (e.g.
`gr/test_gr_shocktube_cpu.py:78` "Can't compare reference against reference",
`multigrid/test_mg_binary_gravity_cpu.py:99` "Prerequisite tests did not run").
So the 15 is NOT a measure of GPU coverage, and nothing in the suite has ever
exercised a GPU — see "Performance and GPU status" below.

Bugs found and fixed along the way (see each task's log for full detail):
`history.cpp`'s volume diagnostic and `derived_variables.cpp`'s `mhd_divb`
diagnostic (both used flat `dx1*dx2*dx3`/`dx1/dx2/dx3` instead of
`geom.Vol()`/`geom.Area1/2/3()`); missing ghost-zone mirroring for
PLM/PPM reconstruction data (B6/B7) at non-origin reflecting walls;
`geometric_srcterms.hpp` reading `w0(IEN)` (internal-energy density) as if
it were pressure directly (C1/C2); a missing GR+curvilinear mutual-
exclusion guard (D1); and a SR+MHD guard that incorrectly fired for plain
SR+MHD+Cartesian runs too, not just curvilinear ones (G1, caught by the
session-wide full-suite re-run, not the coordinates suite alone).

**Remaining work**: none from the approved v2 plan. The "Deferred" section
below (SMR/AMR for curvilinear, WENOZ/TENO curvilinear reconstruction, full
3D polar-axis handling, non-separable coordinate mappings) was explicitly
out of scope from the start and remains so unless the user asks to revisit
it. A follow-on effort covering performance, GPU-readiness and upstream
separation is tracked in "Performance and GPU status" below.

**Working tree state**: the v2 plan landed on branch `curvilinear_coordinate`
in three commits — `5d5f81d4` (core architecture, geometry, physics kernels),
`202c67f8` (test suite), `0fa5ac13` (these development notes). Before
committing anything further, audit this whole file per the note at the top,
and get explicit user sign-off per the global git commit instruction.

**How to resume verification**: `cd tst && python3 run_test_suite.py --cpu
"-DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_BUILD_TYPE=Release
-DKokkos_ARCH_NATIVE=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_SERIAL=ON"`
after `module load cmake/3.26 gcc/13` (see "Build environment" section
below for the full recipe and its gotchas) — and run it via `srun -p
p.sakura -N 1 -n 1 -c 40 --time=<T> bash -c '...'` for a DEDICATED node,
not directly on the login node (see "Testing on a shared cluster" below;
the login node's load can exceed 30 from unrelated users, which looks
exactly like a hang).

## Performance and GPU status (as of 2026-08-16)

### Measured CPU throughput

Run with `tst/perf_benchmark_compare.sh <ref_commit>` on a dedicated node
(`OMP_NUM_THREADS=8`, 3 reps, mean). The script measures two DIFFERENT things
and it is important not to conflate them:

**Axis A — regression.** Did adding curvilinear support slow down the
*Cartesian* path? Same `coord=cartesian` orszag_tang input on a pristine build
of the pre-feature base `190d482e` vs. the current tree. Run-to-run scatter on
this machine is roughly +/-1 point, so two independent runs are quoted for the
final state:

| state | plm | ppm4 |
|-------|-----|------|
| as originally committed (`0fa5ac13`) | **23.46%** slower | 4.90% slower |
| + PLM factors precomputed            | 18.45%            | 4.25%        |
| + update-kernel invariant hoist      | 18.30%            | 3.88%        |
| + uniform-spacing PLM path           | 5.81%             | 3.91%        |
| + uniform-spacing PPM path           | 6.89% / 7.06%     | 1.53% / 1.07% |
| **after `upstream/main` merge (`f84c65d0`, current)** | **7.75%** | **1.92%** |

**Axis B — coordinate cost.** How much slower is curvilinear than Cartesian?
Four inputs (`tst/inputs/perf_coord_*.athinput`, `ct_divb_test`, 400x400) that
are byte-identical apart from the `coord =` line, all on the current build.
Pre-merge values in parentheses:

| recon | cartesian | cylindrical | cylindrical_axisym | spherical_polar |
|-------|-----------|-------------|--------------------|-----------------|
| plm   | baseline  | 9.94% (9.67) | 9.68% (9.75)      | 16.26% (16.47)  |
| ppm4  | baseline  | 4.62% (4.24) | 4.27% (4.35)      | 5.08% (5.16)    |

**The merge cost nothing measurable.** Upstream rewrote `Mesh::NewTimeStep` and
split `mhd_newdt.cpp`, either of which could in principle have moved these, so
the whole matrix was re-run afterwards. Axis B is unchanged to within a few
tenths of a point everywhere. Axis A's plm figure sits at the top of its scatter
band rather than the middle — but there are three pre-merge plm samples spanning
5.81-7.06 on this machine, so 7.75 is consistent with run-to-run noise rather
than a regression. Anyone who needs that distinction settled properly should
raise the repetition count (`NREPS=10 ./perf_benchmark_compare.sh 190d482e`)
instead of reading one more single run; the default `NREPS=3` is tuned for a
quick answer, not for resolving a one-point difference.

### How to read these numbers

Axis B was 2-3% before the optimization work and is 4-16% after. **That is not
a regression.** Both sides of the comparison used to be equally slow, which
flattered the ratio; Cartesian is now ~18% faster in absolute terms
(3.37e6 -> 4.08e6 zc/cpu_s with plm) and the genuine cost of curvilinear
geometry is finally visible. Axis A is the number that measures whether
anything got worse, and it improved by a factor of ~3 (23.46% -> 7.75%).

The original 23.5% was almost entirely the generalized PLM limiter (Task B6):
`PLMGeom()` did 7 divisions per cell per variable per direction where upstream
`PLM()` does 1, in all three directions. ppm4's much smaller 4.9% was
consistent — B7 traded 2 divisions for 10 cache-resident coefficient loads and
roughly broke even. Three changes fixed it, in decreasing order of effect:

1. **Uniform-spacing fast path** (the big one). Both limiters now take
   upstream's original expression whenever a direction's coefficients are the
   flat ones. Deliberately keyed on a per-direction *spacing* flag
   (`GeomData::plm_uniform1/2/3`, `ppm_uniform1`) rather than on
   `CoordinateGeneral`: the kernels still never branch on coordinate system, and
   cylindrical/axisym get the fast path in their genuinely flat phi/z
   directions too, which a "cartesian-only" path would not have given them. The
   coefficients are snapped to the exact flat constants at construction, so the
   fast path is **bitwise identical** to a pre-curvilinear build, not merely
   equivalent. The branch is constant across a launch, hence predicted on CPU
   and warp-coherent on GPU.
2. **Precomputing the PLM factors** into `GeomData::plm_c1/c2/c3` (23.5 -> 18.5).
   Every divisor in the limiter depends only on position, exactly as for B7's
   PPM coefficients. This is what makes the *curvilinear* path fast; the fast
   path in (1) never runs there.
3. **Hoisting loop-invariant geometry** out of the update kernels' inner loop.
   Measured as a wash (18.45 -> 18.30, within noise) — those factor arrays are
   small enough to stay in L1, so the reload was nearly free. Kept because it
   is strictly less work per cell and matters more on a GPU, where the
   arithmetic is cheaper relative to memory. **Recorded here as a negative
   result so nobody re-derives it.**

The residual ~7-8% (plm) / ~2% (ppm4) is the area/volume-weighted flux divergence,
the Stokes-form CT curl and the geometry-based CFL widths, none of which have a
uniform fast path. Adding one would mean templating those kernels, which grows
the upstream merge surface in exactly the files upstream edits most — not worth
it for ~7% until there is a GPU measurement to justify it.

### GPU status: NEVER COMPILED FOR A DEVICE

This is the single biggest gap. To be explicit, because earlier versions of
this file implied otherwise: **no part of this project has ever been through
nvcc (or any device compiler).** Sakura has no GPU — `sinfo` reports
`GRES=(null)` on all 237 nodes — and no `cuda` module exists (`find-module
cuda` → "No module matching 'cuda' found"). There are also **no curvilinear
`_gpu` tests**: all 16 modules under `tst/test_suite/coordinates/` are
`_cpu`-named, so `run_test_suite.py --gpu` would today build with CUDA and run
zero curvilinear cells.

What a device-portability audit found in the committed code:

*Structurally sound.* All `GeomData` arrays are `DvceArray2D<Real>` (device
`Kokkos::View`) filled via `create_mirror_view` + `deep_copy`; nothing captures
`this` or a host pointer (`pmy_pack` is isolated in `MeshGeometry`, deliberately
kept out of `GeomData`); all nine accessors are `KOKKOS_INLINE_FUNCTION`; the
`(m, idx)` LayoutRight indexing is `i`-fastest and therefore coalesced, with
`a1j(m,j)`/`a1k(m,k)` uniform across a warp; and coordinate dispatch is fully
compile-time (`geometric_srcterms.cpp` switches host-side into distinct template
instantiations, and `cartesian` launches no source-term kernel at all).

*Two real defects were found.* Both are described below with their status.

1. **Host-side calls to device accessors — FIXED.**
   `pgen/unit_tests/geometry_curvilinear_test.cpp` called `geom.Area1(...)` /
   `CenterWidth2/3(...)` from plain host `for` loops in nine places. Those
   accessors are `KOKKOS_INLINE_FUNCTION`, i.e. `__host__ __device__`, so the
   calls compile cleanly and then dereference a device pointer at runtime.
   Invisible on a CPU-only build, fatal on CUDA.
   Fixed structurally rather than site-by-site: `GeomDataHost` +
   `MirrorGeomData()` (mesh_geometry.hpp) provide a host mirror with the same
   nine accessors, deliberately NOT annotated, so a device-side call to them is
   a compile error. **Any future host-side geometry code should use that**, and
   the nine call sites now do.
2. **Oversized kernel capture — NOT FIXED, still open.** `GeomData` is ~46
   `View` handles (~1.8 kB) and is captured whole into every geometry-touching
   kernel. Kokkos' `ConstantMemoryUseThreshold` is 512 B
   (`kokkos/core/src/Cuda/Kokkos_Cuda_Instance.hpp`), so all of these kernels
   are pushed off the fast kernel-argument launch path onto the `__constant__`
   path, which costs an extra `cudaMemcpyToSymbolAsync` plus a serializing
   semaphore per launch. Nothing breaks (the limit that would break is
   `KernelArgumentLimit`, 4096), but it is per-launch overhead a Cartesian-only
   build did not pay. The intended fix is to split `GeomData` into per-consumer
   sub-structs (flux / edge / recon / cfl / srcterm) so each kernel captures
   only what it uses — `hydro_update` needs 12 of the handles, `mhd_ct` 15.
   Deliberately deferred: it is pure launch-overhead tuning, it cannot be
   measured on this machine, and the rank-3 `plm_c1/c2/c3` layout chosen in the
   performance work above already avoided the worst of it (3 handles instead of
   the 21 that seven rank-2 arrays per direction would have cost).

### Curvilinear GPU tests now exist

`tst/test_suite/coordinates/` gained `test_geom_curvilinear_construction_gpu.py`,
`test_ct_divb_gpu.py` and `test_recon_exact_gpu.py` (14 tests). They cover the
three distinct groups of `GeomData` arrays — position tables, area/volume/length
tables, and reconstruction coefficient tables (including the rank-3 `plm_c*`,
whose shape is unlike anything else in `GeomData`). Before these, every
coordinates test was `_cpu`-named and `--gpu` would have exercised zero
curvilinear cells.

**These have never been executed** — there is no GPU here. They are collected
correctly (`pytest --collect-only -k _gpu` finds 14) and their inputs and
resolution overrides are valid, but the first real run of them will be the first
run of this code on a device. Expect to fix things.

### Device build recipe (for whoever has GPU access)

```bash
cd tst && python3 run_test_suite.py --gpu "-DKokkos_ARCH_<GPUARCH>=On"
```
`--gpu` already forces `-DKokkos_ENABLE_CUDA=On` (`run_test_suite.py`), so a
non-NVIDIA target (e.g. Aurora's Intel PVC, which needs the Kokkos SYCL backend)
needs that hardcoding relaxed first. Note also `CMakeLists.txt` sets
`Kokkos_ENABLE_CUDA_LAMBDA` automatically when CUDA is on.

One hazard to watch for specifically: at `x1min=0` the spherical factory sets
`a1i(is)` to exactly `0`, and `mhd_ct.cpp` divides by `Area1`. Correctness relies
on the numerator vanishing there too. If it ever does not, the result is a
silent NaN rather than a trap — so a `div(B)` check that comes back NaN (rather
than merely large) points at that division, not at the CT stencil.

## Upstream separation and merge surface

This branch is meant to stay easy to merge with `upstream/main`
(`https://github.com/IAS-Astrophysics/athenak.git`, added as remote `upstream`;
`git config rerere.enabled true` is set so repeated conflict resolutions are
replayed automatically). The guiding rule: **new logic goes in new files;
upstream-owned files carry the smallest possible diff.**

Two extractions were done on 2026-08-16 specifically to shrink the surface:

- `src/reconstruct/recon_geom.hpp` (new) now holds `PLMGeom`/`PPM4Geom`/
  `PPMXGeom`. These were previously extra overloads bolted into upstream's
  `plm.hpp` (+65 lines) and `ppm.hpp` (+161 lines). **Both files are now at
  ZERO diff against upstream.** Distinct names (not overloads) are used so
  resolution never depends on which headers are in scope.
- `src/coordinates/coord_general.{hpp,cpp}` (new) now holds the
  `CoordinateGeneral` enum, `ParseCoordGeneral()` and the definition of
  `Mesh::ValidateCoordGeneral()`. `mesh.cpp` went **99 -> 6 lines** and
  `mesh.hpp` **32 -> 3**.

Net effect: 27 modified upstream files / 641 added lines -> **25 files / 311
added lines**.

### Touchpoint inventory (current)

Regenerate with:
`git diff --stat --diff-filter=M $(git merge-base HEAD upstream/main) -- src/`
(note `--diff-filter` must come BEFORE `--`, or it is silently parsed as a
pathspec and every file is listed).

| Category | Files | Lines | Conflict risk |
|---|---|---|---|
| Hot-loop physics | `recon.hpp` (87), `mhd_ct.cpp` (38), `hydro_update.cpp` (29), `mhd_update.cpp` (24), `{hydro,mhd}_fluxes.cpp` (20), `{hydro,mhd}_newdt.cpp` (18), `dyn_grmhd_fluxes.cpp` (14) | 230 | **Medium-high** — the same loops upstream actively develops. Irreducible: the area/volume-weighted divergence, the Stokes-form CT curl and the geometry-aware reconstruction dispatch have to live inside these kernels. |
| Registration / plumbing | `CMakeLists.txt` (19), `pgen.cpp` (24), `pgen.hpp` (12), `meshblock_pack.{cpp,hpp}` (17), `coordinates.cpp` (13), `{hydro,mhd}.cpp` (22), `{hydro,mhd}_tasks.cpp` (8), `build_tree.cpp` (8), `mesh_refinement.cpp` (8), `mesh.{cpp,hpp}` (9) | 140 | **Low** — mostly additive lines (a new member, a factory call, a guard, a list entry). Conflicts here are mechanical. |
| Independent bug fixes | `history.cpp` (8), `derived_variables.cpp` (15) | 23 | **None, and these should leave.** Both fix pre-existing upstream bugs (flat `dx` weighting in the history mass/energy output and in the `mhd_divb` derived variable) that are wrong on Cartesian too. They are separately PR-able and should be upstreamed on their own so they stop being part of this branch's merge surface. |

### Merge procedure

`git fetch upstream && git merge upstream/main`, then rebuild and run the full
CPU suite. Do this often — a small, frequent merge is far cheaper than a large,
rare one, and `rerere` only helps if conflicts recur. **Do not push to
`upstream`.**

### Merge log

**2026-08-16, `upstream/main` @ `24dd5275`** (3 commits: RKL2 super-time-stepping
#779, pgen neighbor-read guards #778, MHD ghost refresh #777). Two conflicts,
both "keep both sides" in an include block or a member list
(`mesh/mesh.hpp`: our `coord_general.hpp` vs upstream's `diffusion/sts_types.hpp`;
`mesh/meshblock_pack.hpp`: our `mesh_geometry.hpp`/`AddGeometry()` vs upstream's
`parabolic_process.hpp`/`RegisterParabolicProcess()`).

Upstream touched **neither `src/reconstruct/` nor `src/coordinates/`**, so the
extraction described above did its job — `plm.hpp`/`ppm.hpp` stayed at zero diff
and every curvilinear file merged untouched. `mhd_newdt.cpp` is worth noting as
the good case: upstream split `NewTimeStep()` into a new
`RecomputeTimeStepFromCurrentState()`, and because our change was confined to the
function *body*, the `CenterWidth2/3` substitutions moved along with it
automatically.

**The important finding was not the merge but what it brought in.** Upstream's
RKL2 adds `src/hydro/hydro_sts.cpp` and `src/mhd/mhd_sts.cpp`, which contain a
SECOND copy of the kernels this project made curvilinear-aware:
`Hydro::STSUpdate()` and `MHD::STSUpdateU()` each have a flux-divergence loop
dividing by `mbsize.d_view(m).dx1/dx2/dx3`, and `MHD::STSUpdateB()` is a complete
three-kernel CT curl, also flat-`dx`. These are byte-for-byte the pre-curvilinear
forms that `hydro_update.cpp`/`mhd_update.cpp`/`mhd_ct.cpp` replaced.

This also exposed a **pre-existing gap of our own**: nothing guarded diffusion +
curvilinear, and all four diffusion modules compute gradients with flat cell
widths (`viscosity.cpp` 24 `dx` uses, `resistivity.cpp` 30, `conduction.cpp` 30,
`ambipolar.cpp` 7). Viscosity in cylindrical coordinates was already silently
wrong before this merge.

Resolved by **guarding, not porting** (`hydro.cpp`, `mhd.cpp`, next to the
existing WENOZ/TENO + curvilinear guards). Porting the area/volume weighting into
the STS kernels is mechanical, but it would be actively misleading: STS exists
only to accelerate diffusion, and the fluxes it advances are themselves flat-`dx`.
A curvilinear-correct update applied to curvilinear-incorrect fluxes is still
wrong, while looking supported. Because STS is reachable only *through* a
diffusion process, the one guard covers the RKL2 path too. Verified both ways:
curvilinear + `nu_iso` fatals with a clear message, Cartesian + `nu_iso` still
runs.

## History

v1 was approved and implementation of Task A1 began. Verification against
both codebases during that work surfaced 8 substantive problems, the most
important being that v1's `(R,z)` layout (cylindrical with φ dropped via
`nx2=1`) is **impossible** to build in AthenaK — its dimensionality flags are
strictly nested (`nx3>1` forces `multi_d`, and `nx2<4 && multi_d` is fatal),
unlike old Athena++'s independent `f2`/`f3` flags. v2 replaces it with a
dedicated `cylindrical_axisym` coordinate system (x1=R, x2=z, nx3=1 enforced).
See the plan file's "Corrections to v1" section (C1-C8) for the full list;
do not re-derive these — they are settled decisions, not open questions.

## Scope recap (v2)

- Coordinate systems: `cartesian`, `spherical_polar` (r,θ,φ — 1D radial is the
  required layout), `cylindrical` (R,φ,z — general 3D), `cylindrical_axisym`
  (R,z — **the required (R,z) layout**, φ carried as a rotational component,
  `nx3=1` enforced, left-handed).
- Physics: Newtonian first, then flat-spacetime SR. **z4c / dynamical GR is
  untouched — nothing under `src/z4c/` is modified, ever, in this project.**
- Mesh: **single-level uniform only.** SMR and AMR are both deferred (v1
  wrongly kept SMR in scope while deferring curvilinear restriction, which is
  a contradiction — `Hydro::RestrictU`/`MHD::RestrictU`/`RestrictB` all fire
  on `multilevel`, which SMR sets). Enforced by a fatal-error guard.
- Geometry storage: **factored per-direction arrays**, not full `(m,k,j,i)`
  arrays — all four coordinate systems have separable metric factors, so a
  `GeomData` POD struct of small `(m,i)`/`(m,j)`/`(m,k)` tables with inline
  accessors (`Vol`, `Area1/2/3`, `Len1/2/3`) is both cheaper and simpler than
  v1's full 4D arrays.
- Hard requirements every relevant task must demonstrate with a test:
  1. Finite-volume conservation (hydro) exact under new area/volume weighting.
  2. Discrete div(B)=0 (MHD CT) exact under new edge-length/area weighting —
     **but this alone is insufficient** (div(B)=0 is a topological identity
     that holds regardless of edge-length/handedness/sign errors); physical
     induction tests (field-loop advection, `B_r∝1/r²` stationarity) are the
     real correctness gate.
  3. No virtual dispatch / no per-cell coordinate branching in any hot loop.
  4. Cartesian behavior unchanged (bit-for-bit or roundoff) after every
     kernel rewrite.
  5. Geometric source-term coefficients are **ΔA/ΔV ratios** (ported directly
     from `coord_src1_i_`/`coord_src2_i_` in old Athena++), not `1/x1v` as
     v1 incorrectly specified — this is what makes the scheme well-balanced.
  6. CFL timestep must use physical cell widths (`Vol/max(A±)` per direction),
     not raw `dx2`/`dx3`, which are angles (not lengths) in curvilinear
     systems — inert for the two required layouts but a correctness trap for
     any future 2D-spherical/3D use if left unfixed.
  7. Ghost-zone geometry at `x1min=0` is built by mirroring about r=0 (using
     `|r|`), since `LeftEdgeX` extrapolates linearly and would otherwise give
     negative face areas/volumes in the ghost zone.

## Incidents

**2026-08-04, manual-verification `rm -rf` near-miss.** While manually inspecting
`.hst` output for Task A3, a cleanup command `rm -rf /tmp/manual_verify_build
/u/tlam/athenak_curvilinear/inputs/../inputs` was run. `X/../inputs` resolves to
exactly `X` when `X` ends in `inputs` — so the second path resolved to the repo's
real `inputs/` directory, not a no-op, and it was deleted from disk. **No data was
actually lost**: this repo is a git checkout (`git status`/`git log` confirm it,
despite the outer `/sakura/u/tlam` not being one), the deletion was an unstaged
working-tree change, and `git restore inputs/` recovered all ~103 files exactly.
Lesson: never concatenate a variable/relative path onto a destructive command
without first echoing/resolving it (or just `ls`-ing it) to see the literal target;
`X/../Y` where `X` and `Y` share a name is a classic way to silently target `Y`
itself instead of a sibling. Also confirms this repo IS git-tracked, which was not
obvious from the session's initial environment info (only the parent
`/sakura/u/tlam` was reported as "not a git repository") — check `git status`
directly in a subdirectory rather than trusting that report for nested repos.

## Build environment (Phase 0 / Task T0) — DONE, 2026-08-04

Default `/usr/bin/g++` is 7.5.0 — too old for bundled Kokkos 4.7.2 (needs
≥8.2.0). `/u/tlam/athenak_m1/build_m1_sakura.sh`'s icpx+`--gcc-toolchain=`
recipe works but is unnecessarily complex for this project (no MPI needed,
and multi-word `-DCMAKE_CXX_FLAGS="... --gcc-toolchain=..."` values don't
survive `run_test_suite.py`'s flag parsing — see gotcha below). **Simpler
working recipe, verified via the full test harness**: plain `gcc/13` module
directly as the compiler, no MPI, no icpx:
```bash
source /etc/profile.d/modules.sh
module purge
module load cmake/3.26 gcc/13
cd tst
python3 run_test_suite.py --cpu "-DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_BUILD_TYPE=Release -DKokkos_ARCH_NATIVE=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_SERIAL=ON"
```
**Verified baseline (before any curvilinear source changes beyond the
in-progress A1 mesh.{hpp,cpp} edits, which are backward-compatible since
`coord` defaults to `cartesian`): 211 passed, 15 skipped (GPU/MPI-only
tests, correctly excluded by `--cpu`), 0 failed, build 42.75s.** (The log
for this run, `tst/run_suite_baseline.log`, was a scratch file removed
during later cleanup — not committed, so not reproducible verbatim; rerun
the recipe above against a clean `git stash` if the exact baseline output
is needed again.)

**Gotcha**: `module load` does not persist across separate shell
invocations in this environment — always `source /etc/profile.d/modules.sh
&& module purge && module load ...` in the *same* command as the
build/test invocation.

**Gotcha**: `run_test_suite.py --cpu`'s argparse uses `nargs="*"` for cmake
flags, which stops consuming as soon as it sees a bare `-D...` token
(looks like a new option to argparse). Extra cmake flags must be passed as
**one single quoted string** (space-joined) — the script re-splits it
internally (`cmake_flags()` → `arg.split(" ")`). Passing them as separate
shell-quoted `-D...` arguments causes an "unrecognized arguments" error.

No CUDA/nvcc on this node — GPU builds/tests are not possible here (Task A5
is a CPU throughput check instead; document the GPU run for later on a GPU
machine). Test harness: `cd tst && python run_test_suite.py --cpu` (also
`--style`, `--test test_suite/<dir> --cpu` for a subset). `testutils.py`
builds into `tst/build/`, binary at `tst/build/src/athena`, with
`tst/build/inputs` symlinked to the repo-root `inputs/`. `clean_make()`
always does a full clean + reconfigure + rebuild, so the compiler flags
above must be re-supplied on every invocation.

## Testing on a shared cluster (discovered during Tasks D1/D2, 2026-08-14)

The interactive/login node this project builds on is SHARED — `uptime` has
shown load averages of 30-38 on a 40-core node from OTHER users' unrelated
jobs, at times completely unrelated to anything in this session. A test
that should take ~2 seconds can then appear to hang for 5+ minutes under a
naive timeout, with NO error and no indication anything is wrong — this is
NOT a bug in the code being tested, just CPU starvation. `squeue -u
<user>` may also show jobs unrelated to this work (e.g. a separate
long-running allocation, or an unrelated batch job under the same
account) — leave those alone, they are not part of this project.

**Standing convention from here on**: run all builds and test suites
(anything heavier than a single quick smoke-test invocation) via a
dedicated SLURM allocation:
```bash
srun -p p.sakura -N 1 -n 1 -c 40 --time=00:30:00 bash -c '
module purge && module load cmake/3.26 gcc/13
cd tst && python3 run_test_suite.py --cpu "<flags>"
'
```
`squeue -u <user>` confirms the job landed on an otherwise-idle node before
trusting timing; a genuinely-hanging test on a dedicated node (no other
load) is a real signal worth investigating, unlike on the shared login
node where it usually isn't. Confirmed the exact same "hung" test (30+
minutes on the login node) completed in ~2-5 seconds on a dedicated node.

**Also learned (re-confirming the B7/A3 lesson from earlier)**: never run
two `run_test_suite.py`/build invocations against the same `tst/build`
directory concurrently, INCLUDING across a foreground `srun` and a
background one — `clean_make()`'s full clean+reconfigure+rebuild from one
invocation will corrupt/race with the other, producing spurious
`FileNotFoundError`s and unrelated-looking test failures that have nothing
to do with the actual code changes being tested.

## Task status

Legend: `[ ]` not started, `[~]` in progress, `[x]` done (with commit ref),
`[!]` blocked/needs revisit.

### Phase 0
- [x] T0 — Establish working build, confirm baseline `--cpu` suite green —
      DONE 2026-08-04

### Phase A — Foundation
- [x] A1 — Coordinate enum + input parsing + mesh validation (v2) — DONE 2026-08-04
- [x] A2 — `MeshGeometry`/`GeomData` (factored) + Cartesian factory + wiring — DONE 2026-08-04
- [x] A3 — Generic area/volume-weighted flux divergence (Hydro) — DONE 2026-08-04
- [x] A4 — Generic area/volume-weighted flux divergence (MHD) — DONE 2026-08-04
- [x] A5 — CPU performance regression check — DONE 2026-08-04

### Phase B — Curvilinear geometry + reconstruction
- [x] B1 — Cylindrical (R,φ,z) geometry factory — DONE 2026-08-04
- [x] B2 — Cylindrical axisymmetric (R,z) geometry factory [NEW in v2] — DONE 2026-08-04
- [x] B3 — Spherical-polar (r,θ,φ) geometry factory — DONE 2026-08-04
- [x] B4 — Geometry-array construction + conservation unit tests — DONE
      2026-08-05 (both halves; the conservation half surfaced and fixed a
      real bug in `src/outputs/history.cpp` — see log entry below).
- [x] B5 — CFL timestep fix for curvilinear angular directions [NEW in v2] —
      DONE 2026-08-06
- [x] B6 — Non-uniform-grid PLM reconstruction — DONE 2026-08-07
- [x] B7 — Non-uniform-grid PPM reconstruction + WENOZ/TENO guard — DONE 2026-08-07

### Phase C — Geometric source terms (Newtonian, corrected ΔA/ΔV coefficients)
- [x] C1 — Cylindrical/axisym Newtonian geometric source term — DONE 2026-08-07
- [x] C2 — Spherical Newtonian geometric source term — DONE 2026-08-07

### Phase D — Constrained transport
- [x] D1 — Area/edge-length-weighted CT curl — DONE 2026-08-14
- [x] D2 — div(B)=0 preservation test suite (topological check, necessary but insufficient) — DONE 2026-08-14
- [x] D3 — Physical induction tests: field-loop advection + B_r∝1/r² [NEW in v2] — DONE 2026-08-14

### Phase E — Origin boundary
- [x] E1 — Validate reflect BC at x1min=0 + input guardrails — DONE 2026-08-14
- [x] E2 — Sign-flip and origin-conservation correctness tests — DONE 2026-08-14

### Phase F — Problem generators + end-to-end validation
- [x] F1 — Spherical radial Sod shock tube (hydro) — DONE 2026-08-14
- [x] F2 — Axisymmetric (R,z) magnetized rotating-disk equilibrium (MHD) — DONE 2026-08-14
- [x] F3 — Formal convergence-order verification — DONE 2026-08-14

### Phase G — SR (flat-spacetime) extension
- [x] G1 — SR geometric source term generalization — DONE 2026-08-14

### Deferred (not implemented in this project)
SMR/AMR for curvilinear (volume/area-weighted restriction/prolongation).
WENOZ/TENO curvilinear generalization. Full-3D polar-axis handling beyond the
required layouts. Non-separable coordinate mappings. SR+MHD geometric source
terms.

**Diffusion + curvilinear** (added 2026-08-16, see the merge log above). Every
gradient/flux in `src/diffusion/` uses flat cell widths, so viscosity,
conduction, resistivity and ambipolar diffusion are all rejected under
`coord != cartesian` by a guard in `hydro.cpp`/`mhd.cpp`. Lifting that guard is a
two-part job and both parts are required — doing either alone still gives wrong
answers:
1. Make `src/diffusion/` curvilinear-aware (gradients, EMFs, and the parabolic
   timestep limiters, which use `SQR(dx)` and would need `CenterWidth2/3`).
2. Port the area/volume weighting into upstream's RKL2 STS kernels
   (`hydro_sts.cpp` `STSUpdate`, `mhd_sts.cpp` `STSUpdateU`/`STSUpdateB`) — the
   existing `hydro_update.cpp` and `mhd_ct.cpp` treatment transfers directly.
   Note also that the STS path never calls `*SrcTerms()`, so
   `AddCoordGeomSrcTermsHydro/MHD` would not be applied during STS sub-stages;
   that needs handling too.

**GPU verification** is not on this list because it is not a feature decision —
it is undone work. See "Performance and GPU status".

## Per-task implementation log

Add one dated entry per task as it's implemented: files touched, key design
choices made during implementation (especially any deviation from the plan
and why), test results, and any `file:line` references later tasks should be
aware of. Keep entries terse — this is a working log, not a report.

### A1 — 2026-08-04 — DONE
Files: `src/mesh/mesh.hpp`, `src/mesh/mesh.cpp`, `src/mesh/build_tree.cpp`,
`tst/inputs/ut_coord_validation.athinput`,
`tst/test_suite/coordinates/{__init__.py,test_coord_input_validation_cpu.py}`
- `enum class CoordinateGeneral {cartesian, cylindrical, cylindrical_axisym,
  spherical_polar}` in mesh.hpp (next to `RegionSize`); `Mesh::coord_general`
  member (initializer-list order matches header declaration order).
- `<mesh>/coord` parsing (default `"cartesian"`), right after `nx1/nx2/nx3`.
- All validation consolidated into `Mesh::ValidateCoordGeneral()` (private
  method, declared in mesh.hpp, defined in mesh.cpp right before `~Mesh()`):
  `x1min>=0` for all 3 curvilinear systems; `0<=x2min<x2max<=pi` for
  spherical_polar when `multi_d`; `nx3==1` required for cylindrical_axisym;
  `multilevel && coord_general != cartesian` fatal (C3 fix). Called once from
  the constructor and again from `BuildTreeFromRestart()`
  (`build_tree.cpp`, right after `mesh_size`/`mesh_indcs` are memcpy'd from
  the restart binary header) — see the method's doc comment for why:
  `coord_general` itself is NOT part of the restart binary payload (adding it
  would require a restart-format version bump, judged disproportionate for
  this), so it always comes from `ParameterInput`; the second call catches
  the case where a `-r restart.rst -i override.athinput` override leaves
  `coord_general` inconsistent with the actual restart grid extents.
- **Gotcha discovered while writing the test**: `ParameterInput::
  ModifyFromCmdline` (`parameter_input.cpp:396-403`) only *overrides*
  parameter keys that already exist in the input file block — it fatals
  with "Parameter 'X' in block 'Y' on command line not found" if the key
  isn't already declared. So `tst/inputs/*.athinput` files used as a test
  base for `block/param=value` command-line overrides must explicitly
  declare every key any test intends to override (e.g. `coord = cartesian`
  in `<mesh>`), even ones that would otherwise rely on a `GetOrAddString`
  default. Cost ~10 min of debugging (2 tests failed with an unrelated
  parameter_input.cpp:399 error; fixed by declaring `coord` in the base
  input file). Worth remembering for every subsequent task's test input.
- Verified: `tst/test_suite/coordinates/test_coord_input_validation_cpu.py`
  (10 cases: 3 expected-success, 7 expected-fatal) — 10/10 pass, and the 7
  fatal cases were grep-verified against `tst/test_log.txt` to be firing for
  the *intended* reason (not a coincidental unrelated failure) before being
  trusted. Full suite re-run after: 221 passed (211 baseline + 10 new), 15
  skipped, 0 failed.

### A2 — 2026-08-04 — DONE
Files: `src/coordinates/mesh_geometry.{hpp,cpp}`,
`src/coordinates/geometry_cartesian.cpp`, `src/mesh/meshblock_pack.{hpp,cpp}`,
`src/mesh/build_tree.cpp` (2 call sites: fresh + restart), `src/mesh/mesh_refinement.cpp`
(regrid path), `src/CMakeLists.txt`, `src/pgen/pgen.{hpp,cpp}`,
`src/pgen/unit_tests/geometry_cartesian_test.cpp`,
`tst/inputs/ut_geometry_cartesian.athinput`,
`tst/test_suite/coordinates/test_geom_cartesian_construction_cpu.py`.
- **`GeomData` final array layout** (see mesh_geometry.hpp docstring for full
  derivation): 21 factor arrays (a1i/a1j/a1k, a2i/a2j/a2k, a3i/a3j/a3k,
  vi/vj/vk, l1i/l1j/l1k, l2i/l2j/l2k, l3i/l3j/l3k) + 3 centroid arrays
  (x1v/x2v/x3v) + 2 source-coefficient arrays (src1/src2) = 26 total, all
  `DvceArray2D<Real>` indexed `(m, index)`. This is more arrays than the plan
  document's illustrative `ia1,ia2,ia3,iv / ja1,...` sketch implied, because
  that sketch turned out not to generalize: e.g. cylindrical's Area3
  i-factor (`0.5*ΔR²`, a *weighted* cell integral) is a genuinely different
  quantity from Area2's i-factor (`ΔR`, a *plain* width) even though both
  are "the i-factor" for their respective accessors. Verified this by
  reading old Athena++'s actual `Face*Area`/`Edge*Length` formulas for
  cylindrical and spherical_polar (not re-deriving from memory) before
  finalizing the layout — every accessor (`Area1/2/3`, `Vol`, `Len1/2/3`)
  was checked against the reference formula for both systems. The layout
  is designed so the SAME 7 accessor formulas (`a1i*a1j*a1k` etc.) work
  unchanged for all 4 coordinate systems; only the array VALUES differ per
  factory. Array shapes mirror `DvceFaceFld4D`/`DvceEdgeFld4D` exactly
  (own-direction accessor for a face quantity gets `n+1`; cell/width
  quantities get `n`) — verified this alignment lets `Area1(m,k,j,i)` be
  called with literally the same `(m,k,j,i)` tuple already used to index
  `flx1(m,n,k,j,i)`, and `Len1(m,k,j,i)` with the same tuple as
  `efld.x1e(m,k,j,i)` — no reindexing needed anywhere in future A3/A4/D1 work.
- `MeshGeometry` constructor dispatches via `switch(coord_general)` to one
  factory function per system; only `cartesian` is implemented (Task A2),
  the other 3 branches fatal with a clear "not yet implemented, see Phase B"
  message (`mesh_geometry.cpp:36`) — this is intentional, not a bug: Phase B
  fills them in one task at a time.
- `pgeom` wired into `MeshBlockPack` exactly like `pcoord`: declared next to
  it, built via `AddGeometry()` called immediately after `AddMeshBlocks()`
  at all 3 places `AddMeshBlocks` is called (`build_tree.cpp` fresh-build
  and restart paths, `mesh_refinement.cpp` regrid path), destroyed in
  `~MeshBlockPack()` before `pcoord`. The regrid path also needed
  `delete pgeom` added (mirroring the existing `delete pcoord` there) since
  `mb_size` changes across a regrid even for `coord=cartesian` (the only
  case that currently reaches that path, curvilinear+multilevel being
  fatally guarded in Task A1).
- Setup-time-only host fill pattern (not a hot loop, so simplicity over
  cleverness): `BuildFactor()` helper in `geometry_cartesian.cpp` allocates
  a `DvceArray2D`, fills a `Kokkos::create_mirror_view` on the host via a
  small lambda, `deep_copy`s to device. Same pattern will be reused by every
  Phase B factory.
- **Regression caught while testing**: A1's two "success" test cases
  (`test_spherical_polar_positive_x1min_succeeds`,
  `test_cylindrical_axisym_nx3_one_succeeds`) started failing once
  `AddGeometry` was wired in — correctly, since those coordinate systems now
  reach `MeshGeometry`'s "not yet implemented" fatal error further down the
  pipeline than `Mesh::ValidateCoordGeneral()` alone. This is expected, not
  a regression: updated both to `_run_fails()` with a comment to flip back
  to `_run_ok()` once Task B2/B3 land, and verified (via `tst/test_log.txt`)
  that the failure is specifically the "Geometry factory ... not yet
  implemented" message, not something else. **Reminder for Task B1/B2/B3**:
  flip these two test cases back to expect success once their factories
  exist.
- Verified: `test_geom_cartesian_construction_cpu.py` — pgen compares all 21
  factor arrays + 3 centroids + 2 source coefficients against independently
  recomputed values (`mb_size.dx1/2/3`, `CellCenterX()`) to `1e-13`, on a 3D
  grid split into 2 MeshBlocks in x1 (so per-block x1min differences are
  actually exercised) — passes, "Geometry Cartesian Test Passed" confirmed
  in log (not just exit code). Full suite re-run: 222 passed (was 221), 15
  skipped, 0 failed — including AMR/z4c tests, confirming the
  `mesh_refinement.cpp` regrid path's new `pgeom` rebuild is correct too.

### A3 — 2026-08-04 — DONE
Files: `src/hydro/hydro_update.cpp`, `tst/inputs/ut_hydro_mass_conservation.athinput`,
`tst/test_suite/coordinates/test_hydro_mass_conservation_cpu.py`.
- Replaced the 3 raw `(flx[i+1]-flx[i])/dxN` accumulations in `Hydro::RKUpdate`
  with `geom.AreaD(...)*flx - geom.AreaD(...)*flx` (no per-term division),
  then a single `/geom.Vol(m,k,j,i)` at the final `u0_ = gam0*u0_+gam1*u1_-
  beta_dt*divf/Vol` combine step. `geom` is `pmy_pack->pgeom->geom_data`,
  captured into the `KOKKOS_LAMBDA` by value exactly like the pre-existing
  `mbsize` capture it replaced (`GeomData` is a POD of cheap `DvceArray2D`
  view handles, same cost profile as the `CoordData`-by-value pattern already
  used elsewhere — verified this is a correct/idiomatic capture, not a
  performance or correctness risk, before relying on it).
- **Chose not to write a separate bit-for-bit "before/after diff" Cartesian
  regression test** as the plan's phrasing suggested, since (a) exact
  bit-for-bit equality isn't actually achievable — reordering
  `(f2-f1)/dx` into `(A*f2-A*f1)/V` changes floating-point rounding even
  though it's mathematically identical for Cartesian (`A=dx2*dx3` constant),
  and (b) a much stronger, more direct signal already exists: the full
  existing test suite (`test_nr_lwave1d_cpu.py` etc.) enforces tight L1-error
  thresholds against known analytic solutions, so any real regression in the
  divergence operator would show up there. Documenting this reasoning rather
  than leaving it implicit, since it's a deliberate scope call, not an
  oversight.
- **Conservation test, and how it avoids needing a synthetic flux field**:
  rather than manually injecting a hand-constructed periodic flux array
  (`Σ_cells Vol·divf == 0` as a standalone kernel-level check), ran a real
  smooth periodic linear-wave problem (existing `linear_wave` pgen, fully
  periodic BCs, single-level/no-AMR) for a few cycles and checked the
  existing `<output> file_type=hst` "mass" column via `athena_read.hst()`.
  This is mathematically the same telescoping identity (periodic domain ⇒
  `Σ Vol·divf` = zero boundary residual ⇒ mass exactly conserved) but
  exercises the real reconstruction+Riemann-solve+divergence+RK-update
  pipeline end-to-end, and needed no new pgen at all (reused the built-in
  `linear_wave` generator + AthenaK's existing history-output machinery).
- **Directly inspected the actual `.hst` values** (not just the automated
  pass/fail) by building a standalone copy and running the input manually:
  mass = `6.75000e+00` at all 3 output cycles (t=0, 0.056, 0.100), identical
  to displayed precision; total energy `6.07500e+00` identical at all 3;
  even the 3 momentum components were bit-for-bit identical across all 3
  rows. Stronger confirmation than the automated `1e-11` relative-tolerance
  check alone.
- **Incident during this verification step** (see "Incidents" section near
  top of this file): a careless `rm -rf` with a `X/../inputs`-style path
  deleted the repo's real `inputs/` directory. No data lost — this repo is
  git-tracked (confirmed via `git status`/`git log`, on branch
  `curvilinear_coordinate`) and `git restore inputs/` recovered all ~103
  files exactly. Full suite re-run after restore to be sure: 223 passed
  (was 222), 15 skipped, 0 failed.
- **Note for future commits**: this repo has real git history — the user's
  global instruction to always summarize changes and ask permission before
  `git commit` applies here. No commits have been made yet; all work so far
  is uncommitted working-tree changes (10 modified + 17 new untracked files
  as of this task).

### A4 — 2026-08-04 — DONE
Files: `src/mhd/mhd_update.cpp`, `tst/inputs/ut_mhd_mass_conservation.athinput`,
`tst/test_suite/coordinates/test_mhd_mass_conservation_cpu.py`.
Identical transformation to A3, applied to `MHD::RKUpdate` (structurally
identical function). Same conservation-test pattern: periodic MHD linear-wave
problem (existing `linear_wave` pgen, `<mhd>` block), checked via the
`{basename}.mhd.hst` "mass" column (note the `.mhd.hst` vs `.hydro.hst`
filename suffix difference from A3 -- confirmed via
`test_sbox_mhdshwave_mpicpu.py`'s existing usage before relying on it).
Verified: 13/13 coordinates-suite tests pass; full suite 224 passed (was
223), 15 skipped, 0 failed.

### A5 — 2026-08-04 — DONE
Files: `tst/inputs/perf_cartesian_benchmark.athinput`,
`tst/perf_benchmark_compare.sh`.
- No CUDA/nvcc on this node (C7), so this is a CPU/OpenMP throughput check,
  not GPU — a GPU rerun (with `-DKokkos_ENABLE_CUDA=On`) is still needed
  before treating this as evidence of GPU performance; documented as a
  follow-up for whoever has GPU access.
- Used `git worktree` to build a pristine copy of HEAD (pre-any-curvilinear-
  change) in `/tmp`, alongside a build of the current working tree, and
  compared throughput on a fixed-cycle (50), no-I/O, 400x400 2D Orszag-Tang
  MHD benchmark (`perf_cartesian_benchmark.athinput` — this is exactly the
  kind of large, compute-bound Cartesian problem the "no hot-loop
  regression" requirement is about). `git worktree` needed
  `git submodule update --init --recursive` since `kokkos/` is a submodule
  (not automatic with `worktree add`).
- **Gotcha caught while writing the benchmark input**: `tlim=-1` does *not*
  mean "unlimited" the way `nlim=-1` does — the run terminated after cycle 0
  with "Terminating on time limit" since `time=0.0 >= tlim=-1.0`. Fixed by
  setting `tlim=1.0e10` (effectively unlimited) and relying on `nlim=50` as
  the actual stop condition. First benchmark attempt silently reported
  `zone-cycles/cpu_second = 0.000000e+00` for both before and after — caught
  by re-running once without `2>/dev/null` and reading the actual program
  output rather than trusting the metric line alone.
- **Result** (3 reps each, `OMP_NUM_THREADS=8`): before avg
  4.4797e6 zone-cycles/cpu_second, after avg 4.4551e6 — **0.55% slower**,
  well within noise and far under any reasonable tolerance (the plan's
  illustrative 5%). Confirms the `GeomData` factored-array accessors
  (2 extra multiplies per `Area`/`Vol` access vs. the previous single
  division) cost nothing measurable relative to the full
  reconstruction+Riemann-solve+divergence pipeline, consistent with the
  factor arrays being tiny and cache-resident (Architecture Decisions,
  "Geometry storage" section).
- **!! THIS 0.55% RESULT IS STALE AND WAS ALSO PROBABLY NEVER VALID. DO NOT
  CITE IT. !!** Two independent problems, both found on 2026-08-16:
  1. *Stale*: it was measured on 2026-08-04, i.e. after A2-A4 only. B5 (CFL),
     B6 (PLM), B7 (PPM), C1/C2 (source terms), D1 (CT) and G1 all landed
     afterwards and were never re-measured. B6 in particular is where nearly
     all of the real cost is. The remeasured figure for the same Cartesian
     benchmark is **23.46% slower** with plm, not 0.55%.
  2. *Probably never valid*: the script did all its arithmetic in `bc`, which
     cannot parse the scientific notation AthenaK prints
     (`zone-cycles/cpu_second = 4.479700e+06`). `bc` emits a syntax error on
     stderr and an empty result on stdout, so sums/averages silently collapse
     to 0. This was rediscovered the hard way when the rewritten matrix
     version of the script printed `0.0000e+00` in every cell. How the
     4.4797e6/4.4551e6 pair above was obtained is therefore unclear; it may
     have been read off individual run output rather than computed. All
     arithmetic in the script is now awk-based.
  See "Performance and GPU status" near the top of this file for the current
  numbers and methodology.
- Packaged the procedure as `tst/perf_benchmark_compare.sh` (parameterized
  by reference commit, defaults to `HEAD`) rather than a pytest, since
  automating a diff-build-and-compare inside pytest is a materially
  different kind of test infrastructure than anything else in this
  codebase's `tst/` suite — this is a manually-invoked regression check,
  documented here, not part of `run_test_suite.py`.

**Phase A complete** (T0, A1-A5). All Cartesian behavior preserved
(224/224 tests passing, exact conservation verified directly, throughput
unchanged); the `MeshGeometry`/`GeomData` foundation is in place and ready
for Phase B's cylindrical/cylindrical_axisym/spherical factories.

### B1/B2/B3 — 2026-08-04 — DONE
Files: `src/coordinates/geometry_{cylindrical,cylindrical_axisym,spherical}.cpp`,
`src/coordinates/mesh_geometry.cpp` (dispatch), `src/CMakeLists.txt`,
3 new `_run_ok` smoke-test cases in `test_coord_input_validation_cpu.py`
(flipped back from the `_run_fails` placeholders A2 introduced).
- **Every formula verified line-by-line against the actual old Athena++
  source** (`~/athena/src/coordinates/{cylindrical,spherical_polar}.cpp`,
  `coordinates.cpp` for inherited/non-overridden base-class formulas) before
  being written — not re-derived from memory or from the plan document's
  illustrative sketch, which turned out to need real correction in one place
  (see B2 below).
- **B1 (cylindrical, R-φ-z, right-handed)**: `Area1=R_f·Δφ·Δz`,
  `Area2=ΔR·Δz` (no φ dependence — this is the *inherited, non-overridden*
  Cartesian-base-class formula in the reference; a naive
  "cylindrical must weight everything by R" assumption would have gotten
  this wrong), `Area3=0.5(R_f,+²−R_f,−²)·Δφ`, `Vol=Area3-factor·Δz`,
  `Edge1=ΔR` (also inherited, unweighted), `Edge2=R_f·Δφ`, `Edge3=Δz`
  (inherited). Ghost-zone geometry: **no explicit mirroring was
  implemented**, contrary to what the v2 plan's Corrections-C5 section
  said would be needed — see "Ghost-zone reasoning" below for why, worked
  out carefully before writing code, not left as a guess.
- **B2 (cylindrical_axisym, R-z, x3 unused)**: this is *not* a relabeling of
  B1's formulas — x2 means z here, not φ, so every formula was re-derived
  from physical first principles for the actual (R,z) meaning (e.g.
  `Area3`, the "virtual φ-face", is a genuinely new flat `ΔR·Δz`
  cross-section formula that doesn't correspond to any face in the B1
  system). **Handedness note, load-bearing for later tasks**: (R,z,φ) is
  left-handed (R̂×ẑ=−φ̂, since the standard right-handed order is R,φ,z with
  ẑ×R̂=φ̂), so AthenaK's generic right-handed x3/e3/IM3/IB3 slot corresponds
  to *minus* the physical φ-component for this system. Geometry itself
  (areas/volumes, orientation-independent positive scalars) is unaffected,
  but Task C1 (source terms) and D1 (CT) must apply this sign, and per plan
  Correction C6, this can *only* be verified by Task D3's physical
  induction tests — div(B)=0 (D2) is topological and cannot catch a sign
  error here. Documented prominently in the file's docstring so C1/D1/F2
  don't have to rediscover this.
- **B3 (spherical, r-θ-φ, right-handed)**: `Area1=r_f²·|cosθ_j−cosθ_j+1|·Δφ`,
  `Area2=0.5(r_f,+²−r_f,−²)·sinθ_f·Δφ`, `Area3=0.5(r_f,+²−r_f,−²)·Δθ`,
  `Vol=(1/3)(r_f,+³−r_f,−³)·|cosθ_j−cosθ_j+1|·Δφ`, `Edge1=Δr` (inherited),
  `Edge2=r_f·Δθ`, `Edge3=r_f·sinθ_f·Δφ`. `x2v` (θ centroid) uses the exact
  reference trig formula
  `[(sinθ_f,+−θ_f,+cosθ_f,+)−(sinθ_f,−−θ_f,−cosθ_f,−)]/(cosθ_f,−−cosθ_f,+)`
  for `nx2>1`, falling back to the reference's own simple-midpoint special
  case for `nx2==1` (which is what the required 1D-radial layout always
  hits) — the reference does *not* use the trig formula in that case
  either, so this isn't a simplification on my part, it's matching upstream
  exactly. **`src1`/`src2` here are radial-only**; the θ-momentum term needs
  a third, j-indexed coefficient (reference's `coord_src1_j_`/
  `coord_src3_j_`, confirmed numerically identical to each other in the
  reference despite being separate arrays) — deliberately deferred to
  Task C2, not added to `GeomData` here, per the struct's existing docstring
  note.
- **Ghost-zone reasoning (revises the v2 plan's C5 "mirror about r=0"
  decision)**: worked through what the reference implementation actually
  does at the origin before writing any mirroring code, since old
  Athena++'s `x1v` loop runs over the *full* ghost range using raw
  (possibly negative) `x1f` values with no `abs()`/mirroring anywhere in
  `cylindrical.cpp`/`spherical_polar.cpp`. Traced through by hand: (1) the
  plain-width factors (`a2i`/`l1i`, i.e. `ΔR`) are `face(i+1)-face(i)`,
  always positive regardless of sign since `LeftEdgeX` is linear in index;
  (2) the *weighted* factors (`a3i`/`vi`/`src1`/`src2`, i.e.
  `0.5(r_+²−r_−²)`-type quantities) can look "negative" in the deep ghost
  region when raw signed values are used naively, **but are only ever read
  at active cell indices (`is..ie`) by the flux-divergence/CT kernels** —
  never in the ghost region — so this is inert, unused dead data, not a
  bug; (3) `x1v` (the one quantity genuinely read into the first ghost
  cells, by reconstruction, Task B6) gives the *physically correct negative
  centroid* there with the raw-signed formula, which is exactly what's
  needed for a correct gradient across the reflecting boundary — mirroring
  it would be *wrong*. Net effect: implemented all three factories with the
  direct (unmirrored) formulas, matching the reference exactly, simpler
  than what the plan called for, and verified as safe rather than assumed.
  Updated here rather than silently diverging from the recorded plan
  decision.
- Verified (smoke tests only — full analytic-value verification is Task
  B4, immediately following): one new `_run_ok` test case per system in
  `test_coord_input_validation_cpu.py` (build+run to `nlim=0` without
  crashing). 14/14 coordinates-suite tests pass after each of B1/B2/B3;
  full suite after all three: 225 passed (was 224), 15 skipped, 0 failed.

### B4 — 2026-08-04/05 — DONE (both halves; see "Conservation half" sub-entry below)
Files: `src/pgen/unit_tests/geometry_curvilinear_test.cpp`, `src/pgen/pgen.{hpp,cpp}`,
`src/CMakeLists.txt`,
`tst/inputs/ut_geometry_{cylindrical,cylindrical_axisym,spherical}.athinput`,
`tst/test_suite/coordinates/test_geom_curvilinear_construction_cpu.py`.
- One pgen (`geometry_curvilinear_test`, dispatches on `Mesh::coord_general`)
  covers all three systems, driven by three separate input files. Every
  check re-derives the expected value **fresh**, from the same reference
  formulas verified against old Athena++ in the B1-B3 log entries above, but
  written independently in the test file rather than copied from
  `geometry_{cylindrical,cylindrical_axisym,spherical}.cpp` — this is what
  makes the test capable of catching a transcription bug in the production
  code (a copy-pasted formula would pass even if the shared formula itself
  were wrong).
- Checked per system: `a1i`/`a2i`(or `a3i` for axisym)/`vi`/`l1i`(or `l3i`)
  factor arrays across the full index range on a multi-MeshBlock grid (so
  per-block x1min differences are exercised, same discipline as A2's test);
  `x1v`, `src1`, `src2` at active indices; one full `Area1` (and, for
  axisym, `Area3`) spot-check via the actual `GeomData::Area1()`/`Area3()`
  accessors (not just the raw factor arrays) to catch a
  factor-assignment/accessor-wiring bug that per-factor checks alone
  couldn't. **Spherical also gets an independent physical cross-check**: on
  a full-solid-angle grid (θ:[0,π], φ:[0,2π]), `Area1` at any radius must
  equal the textbook `4πr²` sphere-shell formula — this doesn't share any
  code path with the geometry factory's own formula derivation, so it's a
  genuinely independent sanity check, not just the same math checked twice.
- Verified: all three pass ("Geometry Curvilinear Test Passed" confirmed in
  `tst/test_log.txt`, no FAILED lines for any of the ~20+ individual
  `CheckClose` assertions per system, including the 4πr² check). Full suite
  after: 228 passed (was 225), 15 skipped, 0 failed.
- **Conservation half — 2026-08-05 — DONE, and it found a real bug.**
  Files: `tst/inputs/ut_{cylindrical,cylindrical_axisym,spherical}_mass_conservation.athinput`,
  `tst/test_suite/coordinates/test_curvilinear_mass_conservation_cpu.py`,
  `src/outputs/history.cpp`.
  - Design: unlike A3's Cartesian *periodic*-domain test, these use
    *reflecting* boundaries — at `R=0`/`r=0` for the axisym/spherical cases
    (the actual required layouts, so this simultaneously exercises the
    `Area1(is)=0` origin-regularity claim from the B1-B3 log), and at the
    outer radial boundary and (for cylindrical) periodic in φ. A reflecting
    wall gives an exactly-zero mass flux by construction (LLF flux of two
    mirror-image states cancels exactly:
    `F=0.5(F_L+F_R)-0.5·s·(U_R-U_L)`, and mirroring negates the normal
    velocity/momentum while leaving density unchanged, so both the
    average and the dissipation term vanish for the mass component
    specifically), and mass conservation is provably independent of
    reconstruction accuracy (Task B6 curvilinear-aware PLM isn't done yet)
    or of geometric source terms (Task C1/C2 isn't done yet, but source
    terms only ever modify momentum, never mass) — so this is legitimate
    to test now, ahead of those later tasks. Reused the `shock_tube` pgen
    (not physically meaningful in curvilinear coordinates, but that's
    irrelevant — only a smooth-vs-discontinuous *dynamics driver* is
    needed to stress-test the divergence operator, not a physically
    correct wave).
  - **First run: all three FAILED, mass visibly growing** (e.g. axisym:
    `0.8 → 0.808 → 0.819 → 0.832 → 0.834` over 4 output cycles — a ~4-8%
    drift, nowhere near roundoff, i.e. clearly a real bug, not a tolerance
    problem). Traced it to `src/outputs/history.cpp`, NOT to the A3/B1-B3
    kernels: the hst "mass" diagnostic computed volume via
    `size.d_view(m).dx1*dx2*dx3` (flat, mbsize-based) independently of the
    new `MeshGeometry`/`GeomData`, at 3 call sites — `LoadHydroHistoryData`
    (was line 132), `LoadMHDHistoryData` (was line 333), and
    `LoadZ4cHistoryData` (line ~223, **left untouched** — z4c is always
    Cartesian, where `dx1*dx2*dx3` is exactly correct, and z4c is
    explicitly out of scope for this project). For Cartesian grids
    `geom.Vol(m,k,j,i) == dx1*dx2*dx3` exactly, so this bug was invisible
    everywhere the diagnostic had been used before (hence why A3/A4's
    Cartesian mass-conservation tests never caught it) — it only manifests
    once cell volumes genuinely vary with position, which curvilinear
    coordinates are the first thing in this codebase to introduce. Fixed
    by replacing both call sites with `geom.Vol(m,k,j,i)` (capturing
    `auto &geom = pm->pmb_pack->pgeom->geom_data;` alongside the existing
    `u0_` capture) and removing the now-unused `size` capture in each
    function (would otherwise be a dead/unused-variable warning).
  - **This means the A3/A4/B1-B3 kernels were correct all along** — the
    "leak" was purely a measurement artifact in the diagnostic, not a
    physics bug in the flux-divergence rewrite. Worth remembering:
    *diagnostics/output code needs the same geometry-awareness audit as
    the hot-loop kernels* — anything in `src/outputs/` (and, later,
    anywhere else that independently recomputes a cell volume/area rather
    than reading `pgeom`) is a candidate for this same class of bug. No
    other `size.d_view(m).dx1*dx2*dx3`-style volume computation was found
    outside `history.cpp` during this fix (searched
    `grep -rn "dx1.*dx2.*dx3"` — not exhaustive, worth re-checking in
    Phase F when more end-to-end diagnostics get exercised).
  - Verified: all 3 conservation tests pass after the fix (mass constant
    to `<1e-11` relative deviation across all output cycles, same
    tolerance as A3/A4's Cartesian tests). Full suite re-run: 231 passed
    (was 228), 15 skipped, 0 failed — confirming the `history.cpp` fix is
    also a no-op for every existing Cartesian test (as expected, since
    `geom.Vol()` reduces to the old formula there).

### B5 — 2026-08-06 — DONE
Files: `src/coordinates/mesh_geometry.hpp` (new `cw2i/cw2j/cw3i/cw3j/cw3k`
fields + `CenterWidth2()/CenterWidth3()` accessors), all 4
`src/coordinates/geometry_*.cpp` factories (revisited to fill the new
fields), `src/hydro/hydro_newdt.cpp`, `src/mhd/mhd_newdt.cpp`,
`src/pgen/unit_tests/geometry_curvilinear_test.cpp` (extended),
`tst/inputs/ut_geometry_spherical_2d.athinput`,
`tst/test_suite/coordinates/test_geom_curvilinear_construction_cpu.py`
(extended).
- **Resolved the uncertainty flagged in the previous session's stopping
  note**: re-derived which directions are actually "angular" (need
  r-weighting) per system before touching any code, rather than trusting
  the earlier note. Result — `x1` is *never* weighted (always a real
  physical length, in every system); `cylindrical`'s `x2=φ` *is* angular
  (needs `R`-weighting); `cylindrical_axisym`'s `x2=z` is flat, *not*
  angular (moving along z at fixed R doesn't stretch/scale — confirmed by
  literally checking old Athena++'s `CenterWidth2` for cylindrical, which
  uses `x1v(i)*dx2f(j)`, vs. what axisym's z-direction physically needs,
  which is just `dx2f(j)` with no `x1v` factor at all); `spherical`'s `x2=θ`
  and `x3=φ` are *both* angular. **Net effect: the CFL bug is completely
  inert for both required layouts** (cylindrical_axisym R-z: x1 unweighted,
  x2=z flat, x3 unused; spherical 1D-radial: only x1 ever contributes since
  `multi_d=three_d=false`) — it only manifests for general
  multi-dimensional cylindrical/spherical runs with an angular direction
  actually resolved, which this project does support (B1/B3) even though
  the two required layouts don't exercise it.
- Old Athena++ distinguishes `Edge*Length` (face-valued, used by CT) from
  `CenterWidth*` (centroid-valued, used by CFL) — confirmed by reading
  `spherical_polar.cpp:319-335`/`cylindrical.cpp:195-202` directly rather
  than assuming `Len2`/`Len3` (already in `GeomData` from B1-B3) were
  reusable here. They are *not*: `Len2` for cylindrical uses the face value
  `R_f(i)`, while `CenterWidth2` uses the centroid `x1v(i)` — same
  structural form, different underlying quantity. This is why new fields
  (`cw2i/cw2j`, `cw3i/cw3j/cw3k`) were added rather than reusing `Len2/Len3`.
- Formulas, factored the same way as `Area`/`Vol`/`Len` (verified this
  2-and-3-factor structure covers all 4 systems without needing a 3-factor
  `CenterWidth2` anywhere): `cartesian`: `CW2=Δy`, `CW3=Δz` (both flat,
  unchanged from today's behavior). `cylindrical`: `CW2=R_v(i)·Δφ(j)`,
  `CW3=Δz(k)` (flat). `cylindrical_axisym`: `CW2=Δz(j)` (flat — **not**
  `R_v(i)·Δz(j)`, since x2 is z not φ here), `CW3`=unused placeholder
  (`three_d` is always false for this system, dead code path, filled
  trivially for consistency with the rest of the file's convention).
  `spherical`: `CW2=r_v(i)·Δθ(j)`, `CW3=r_v(i)·sin(θ_v(j))·Δφ(k)` (note:
  uses the CENTROID `sin(θ_v(j))`, a new lambda, distinct from the
  FACE-valued `sintheta_face_of` already used by `Area2`/`Len3`).
- `hydro_newdt.cpp`/`mhd_newdt.cpp`: replaced
  `mbsize.d_view(m).dx2/dx3` with `geom.CenterWidth2/3(m,k,j,i)` at all 4
  call sites (both the kinematic-advection and hydrodynamic/MHD branches in
  each file); `dx1` left untouched (never weighted in any system).
- Test: extended `geometry_curvilinear_test.cpp` with direct
  `CenterWidth2`/`CenterWidth3` accessor checks (spot-checked against
  independently re-derived formulas, same discipline as B4) for cylindrical
  and spherical. Added a new 2D spherical input
  (`ut_geometry_spherical_2d.athinput`, θ resolved, `nx2=8`) specifically
  because the required 1D-radial layout only exercises the *degenerate*
  simple-midpoint θ-centroid branch, not the non-trivial trig formula —
  without this, the trig branch would have gone completely untested.
  Deliberately did **not** write a separate full end-to-end
  "run and check the reported `dt` value" test (the plan's literal
  suggestion) — decided the direct accessor check is more precise and
  easier to get exactly right, and the full test suite (232 tests, many
  CFL-timestep-sensitive) already provides strong evidence that reading
  `CenterWidth2/3` into the actual `NewTimeStep` kernels didn't break
  anything for Cartesian.
- Verified: 4/4 curvilinear geometry-construction tests pass (cylindrical,
  cylindrical_axisym, spherical 1D, spherical 2D — new), confirmed via
  `tst/test_log.txt` ("Geometry Curvilinear Test Passed" x4, no FAILED
  lines). Full suite: 232 passed (was 231), 15 skipped, 0 failed —
  confirming Cartesian `dt` is unchanged (since `CenterWidth2/3` reduce
  exactly to `dx2/dx3` there).

### B6 — 2026-08-07 — DONE (largest single task so far; found and fixed a real bug)
Files: `src/coordinates/mesh_geometry.{hpp,cpp}` (new `xf1/xf2/xf3` fields +
`MirrorReflectingGhostGeometry()`), all 4 `src/coordinates/geometry_*.cpp`
factories (revisited again, to fill `xf1/xf2/xf3`), `src/reconstruct/
plm.hpp` (rewritten + new uniform-spacing overload), `src/reconstruct/
recon.hpp` (`ReconCellT`/`ReconDispatch` gain a `GeomData` parameter),
`src/hydro/hydro_fluxes.cpp`, `src/mhd/mhd_fluxes.cpp`, `src/dyn_grmhd/
dyn_grmhd_fluxes.cpp` (all 15 `ReconDispatch` call sites updated),
`src/pgen/unit_tests/recon_exact_gradient_test.cpp`, 3 new input files,
`tst/test_suite/coordinates/test_recon_exact_gradient_cpu.py`.

- **Formula verified against the actual reference, not re-derived from
  memory.** Read `~/athena/src/reconstruct/plm.cpp:60-119` directly before
  writing anything. It needs *five* position values per cell (not just the
  three centroids `x1v(i-1/i/i+1)` the plan document anticipated): also the
  two FACE positions `xf(i)`/`xf(i+1)` bounding cell `i`, since the
  centroid-to-face offset is generally not half the cell width on a
  non-uniform/curvilinear grid. This meant `GeomData` needed new `xf1/xf2/
  xf3` face-position fields (filled via the same `LeftEdgeX()` formula
  already used everywhere — coordinate-independent, since face position in
  index space doesn't depend on the metric, only the volumetric centroid
  does) — required going back to touch all 4 already-completed geometry
  factories again.
- **Derived and verified algebraically (not just trusted) that the ported
  formula is exact for linear data**: for `q = a + b*x`, the limiter's
  `dqF`/`dqB` both collapse to `b*dx1f` regardless of spacing, giving
  `dqm = b*dx1f` exactly, and the final `ql_ip1 = q_i +
  ((xf_ip1-x_i)/dx1f)*dqm = a + b*xf_ip1` — the SAME linear function
  evaluated exactly AT THE FACE, not just at the centroid. This is a
  stronger exactness property than the plan asked for, and is what
  motivated keeping face positions rather than trying to avoid them.
- **Also verified the new formula's uniform-Cartesian limit reduces
  exactly to the old formula**, by hand: `cf=cb=2`, `dqF=dqB=dwr/dwl`,
  giving `dqm=2*dwl*dwr/(dwl+dwr)`, and with the `(xf_ip1-x_i)/dx1f=0.5`
  geometric factor this gives `ql_ip1 = q_i + dwl*dwr/(dwl+dwr)` — matching
  today's `PLM()` exactly (old Athena++'s `dwm` convention differs from
  AthenaK's by a factor of 2, compensated by an explicit `0.5` multiplier
  old Athena++ applies and AthenaK's old code baked into `dqm` directly;
  this rewrite follows old Athena++'s convention, not AthenaK's old one,
  since the geometric multiplier is unavoidable once face positions are
  genuinely used).
- **`PLM()` needed a second overload, not a signature change**: a build
  failure surfaced `src/radiation/radiation_fluxes.cpp`, which calls `PLM()`
  directly (not through `ReconCellT`) for reconstruction across ANGULAR
  bins on a geodesic grid — not a spatial direction at all, so `GeomData`
  involvement would be conceptually wrong there, not just inconvenient.
  Added the old 5-argument uniform-spacing formula back as an overload
  (same name, different signature) specifically so that caller needed zero
  changes. Grepped for every other direct `PLM(` caller before rebuilding,
  found none.
- **Threading `GeomData` through 15 call sites**: `ReconDispatch<ivx>` and
  `ReconCellT<recon,ivx>` both gained a `const GeomData&` parameter; updated
  all 3 hydro, 6 MHD (`_w`/`_b` variants x1/x2/x3), and 6 dyn-GRMHD call
  sites. dyn-GRMHD is a dynamical-GR module (couples to z4c) that is always
  Cartesian in practice (z4c is Cartesian-only, per the hard project
  requirement) — passing `pmy_pack->pgeom->geom_data` there is safe and a
  guaranteed no-op (Cartesian `xf`/`x1v` reduce to the old formula exactly),
  not a violation of "don't touch z4c" (no file under `src/z4c/` was
  touched; this is the shared reconstruction utility dyn-GRMHD already used
  before this task).
- **Found and fixed a real, subtle correctness bug via the B4 conservation
  tests** (rebuilt and reran them as an integration check on this change,
  not just the new B6-specific tests): all 3 curvilinear conservation tests
  started failing with mass drifting at the ~1e-6 relative level (small,
  but far above the `1e-11` roundoff bar, and NOT the ~1e-2 signature of
  the earlier `history.cpp` bug — clearly a different, new problem).
  Root cause: ghost-zone `x1v`/`xf1` were filled by the geometry factories
  via the same linear-extrapolation formula used for active cells. That
  formula happens to be exactly mirror-symmetric at a genuine coordinate
  singularity like `r=0` (verified algebraically in the B1-B3 logs — e.g.
  `x1v(is-1) = -x1v(is)` exactly there), which is why Area/Vol/Len needed
  no special ghost handling (B1-B3's conclusion, still correct for THOSE
  arrays, which are never read in the ghost region). But it is **not**
  mirror-symmetric at a generic reflecting wall away from any such
  symmetry point (e.g. an *outer* reflecting boundary at `R=2.0`): linear
  extrapolation just keeps going outward there, while the reflecting BC
  mirrors the physics DATA inward. That geometry-vs-data mismatch breaks
  the "reflecting wall ⟹ exactly zero flux" property PLM's reconstruction
  needs for the boundary-closed conservation tests to hold exactly. B1-B3's
  ghost-zone analysis was correct but incomplete: it didn't anticipate that
  a LATER task (PLM, B6) would start reading ghost-zone centroids/faces at
  all.
  - Fix: `MirrorReflectingGhostGeometry()` in `mesh_geometry.cpp`, called
    once per direction after the coordinate-specific factory runs,
    overwrites ghost-zone `x1v`/`xf1` (and `x2v`/`xf2`, `x3v`/`xf3` when
    `multi_d`/`three_d`) with the TRUE geometric mirror of the
    corresponding active cell, for every face whose boundary condition is
    `reflect` — checked via the PER-MESHBLOCK `mb_bcs` array (not the
    global `mesh_bcs`), so this is correct for multi-block decompositions
    too (an interior block's "ghost" data comes from a neighbor via
    MPI/local copy, not a reflecting BC, and must not be touched).
  - Verified the mirror formula by hand for both `g=1` and `g=2` ghost
    depths (matters since PLM's 2-ghost-cell-deep stencil needs both) by
    mirroring interval endpoints about the wall position and checking
    self-consistency between adjacent ghost cells' shared face.
  - For Cartesian, this fix is an exact no-op (uniform spacing is already
    mirror-symmetric under linear extrapolation, confirmed by the full
    suite re-run showing zero change in any existing Cartesian test).
- Tests: `recon_exact_gradient_test.cpp` calls `PLM()` directly (not
  through the full RK pipeline) with a manufactured `q = a + b*x1v(i)`
  field, checking `ql`/`qr` against `a + b*xf(face)` to `1e-12` relative,
  on Cartesian (regression), cylindrical, and spherical — all pass, and
  confirmed via `tst/test_log.txt` that all three printed "Recon Exact
  Gradient Test Passed" with no FAILED lines (not just exit code 0).
  Re-ran the B4 conservation tests as an integration-level check on this
  task's ghost-mirroring fix (see above) — now pass again after the fix.
  Full suite: 235 passed (was 232), 15 skipped, 0 failed, including every
  existing tight-tolerance linear-wave convergence test (`test_nr_lwave1d_
  cpu.py` etc.) — exactly the kind of quantitative check that would catch
  a subtle PLM regression, not just "did it run."

### Task B7 — Non-uniform-grid PPM reconstruction + WENOZ/TENO guard (2026-08-07)

User explicitly chose "full port, same rigor as PLM" when asked how to scope
this given PPM's much greater complexity than PLM's B6 rewrite.

- **WENOZ/TENO guard**: fatal-error check in `hydro.cpp`/`mhd.cpp` — if
  `recon_method` is `wenoz` or `teno` AND `coord_general != cartesian`,
  fatal at setup (documented limitation, not a silent wrong answer).
  Per-plan deferred item, not implemented for curvilinear.
- **Key analytic realization that bounded the scope**: AthenaK has no
  mesh-stretching feature at all — grid spacing in INDEX space is always
  uniform, even in curvilinear coordinates. Only the metric/Jacobian
  weighting varies with position. This means old Athena++'s general
  non-uniform-grid PPM machinery (LUP-solve for a `beta`-matrix, needed
  when *physical* spacing is non-uniform) is unnecessary; only the
  uniform-Δr *curvilinear* analytic polynomials (Mignone 2014 eq. B.9
  cylindrical/axisym `m_coord=1`, eq. B.14 spherical `m_coord=2`) are
  needed, evaluated at `io = xf1(face)/dx1` — the SIGNED, continuous,
  position-based local radius in grid-spacing units (deliberately NOT old
  Athena++'s index-offset `io=abs(i-is)`, which implicitly assumes
  `x1min=0`; the continuous-position form is what makes an annulus domain
  with `x1min>0` work with no special-casing, and is what makes the r=0
  ghost mirror-symmetry below fall out for free).
- **`GeomData` gained**: `ppm_c1i..c4i` (face-indexed, size `ncells1+1`,
  the 4-point interpolation weight for the face AT index i, using points
  `i-2,i-1,i,i+1`) and `ppm_hpi`/`ppm_hmi` (cell-indexed, size `ncells1`,
  the Mignone eq. 48 generalized CW-monotonicity-clamp ratios, replacing
  the flat formula's hardcoded `2.0`). Filled in all 4 geometry factories;
  Cartesian gets the exact flat constants `(-1/12,7/12,7/12,-1/12)` and
  `2.0` (verified: this is what B.4/the flat limit of B.9/B.14 reduce to).
- **Verified algebraically before writing any test**: (a) `io→∞` limit of
  both the cylindrical and spherical polynomials reduces exactly to the
  flat weights; (b) `c1(io)=c4(-io)`, `c2(io)=c3(-io)` exactly for both
  systems — meaning the r=0-origin ghost region is automatically correctly
  mirrored with NO special-casing (unlike old Athena++'s index-based `io`,
  which needs explicit reversal logic there); this symmetry is specific to
  `io_wall=0` and does NOT hold at a non-origin reflecting wall, exactly
  analogous to B6's `x1v`/`xf1` finding.
- **`ppm_hpi`/`ppm_hmi` are the FINAL ratios, not the raw Mignone `h_plus`/
  `h_minus`**: computed internally as `h_plus=3±dx/(2*xv)` (cylindrical) or
  the more involved `20*xv²+dx²`-denominator form (spherical), then stored
  as `(h_plus+1)/(h_minus-1)` and `(h_minus+1)/(h_plus-1)` respectively —
  these are what directly replace PPM4/PPMX's hardcoded `2.0` in the
  monotonicity clamp. Verified the flat limit: cylindrical/spherical's
  `h_plus,h_minus→3` as `dx/xv→0`, giving ratio `(3+1)/(3-1)=2.0`, matching
  the Cartesian factory's literal `2.0` constant — self-consistency check
  that the two independently-derived formulas (one direct, one via a
  limit) agree.
- **Ghost-mirroring extension for reflecting walls**: added
  `MirrorReflectingGhostPpmCoeffs()` in `mesh_geometry.cpp`, x1-only
  (matches the x1-only scope of the whole curvilinear-PPM generalization —
  see the `GeomData` doc comment), called after the existing
  `MirrorReflectingGhostGeometry()` calls. Reversal rule for the
  face-indexed coefficients (ghost face's `(c1,c2,c3,c4)` = mirror-partner
  active face's `(c4,c3,c2,c1)`, reversed order) and a swap rule for the
  cell-indexed ratios (ghost `hp`↔mirror `hm`). No aliasing hazard between
  reads and writes: ghost-region indices and their mirror-partner active
  indices never overlap for either boundary, by construction of the `g`
  loop bounds (checked explicitly, not just assumed).
- **`ppm.hpp` gained non-uniform overloads** of `PPM4()`/`PPMX()` (9 extra
  parameters: 4+4 face coefficients for faces i and i+1, plus `hp_i`/
  `hm_i`), keeping the original flat-weight overloads untouched (still used
  for x2/x3, and by the `PLM()`-style dual-overload pattern from B6). Only
  TWO pieces of each function are geometry-dependent and needed changing:
  the initial 4-point interpolation (both PPM4 and PPMX use the identical
  formula) and the final CW-eqn-1.10 "away from extrema" monotonicity
  clamp (`hp_i`/`hm_i` replacing the hardcoded `2.0`). PPMX's
  Colella-Sekora extremum-preserving second-derivative logic in between
  (PH eqns 3.35-3.39) was copied UNCHANGED — it operates purely on
  cell-centered VALUES with an implicit uniform-INDEX-spacing assumption
  that remains exactly valid (no mesh-stretching, ever, in this codebase).
- **`recon.hpp`'s PPM4/PPMX branches** gained the same `if constexpr
  (ivx==IVX)` compile-time split already used for PLM in B6: IVX selects
  the geometry-based overload (reading `geom.ppm_c1i(m,i)`, `geom.ppm_c1i
  (m,i+1)`, etc.); IVY/IVZ keep calling the original flat overload
  unchanged.
- **New exactness unit test**: `recon_exact_cubic_test.cpp` — the direct
  PPM analogue of B6's `recon_exact_gradient_test.cpp`. Manufactures a
  cubic polynomial `c0+c1*R+c2*R²+c3*R³`, computes the EXACT
  metric-weighted cell average analytically (unweighted for Cartesian,
  `R`-weighted for cylindrical/axisym, `R²`-weighted for spherical — i.e.
  the correct finite-volume cell average for each system, not a naive
  point sample), and checks `PPM4()`/`PPMX()` reconstruct the exact face
  POINT value of the cubic to `1e-10` relative. Verified by hand (before
  writing the test) that the flat 4-point Lagrange-type interpolant built
  from exact weighted-moment cell averages is exact for any cubic (checked
  concretely: cell averages of `x³` over unit Cartesian cells, `avg =
  i³+i/4`, reconstruct `f(-0.5)=-0.125` exactly via the flat weights) —
  this resolved an initial uncertainty about whether PPM should be
  "exact for cubics" or merely "correct to design order"; it is exact,
  given the correct weighted cell average as input, provided the
  monotonicity clamp doesn't engage (ensured here by choosing manufactured
  coefficients with no interior extremum). Passes on cartesian/
  cylindrical/spherical.
- **Verification**: coordinates-suite (27 tests at this point, all
  passing) and full suite both green. One apparent regression
  (`test_nr_lwave1d_cpu.py::test_run[mhd-plm-rk2]`) turned out to be a
  false alarm from running two `run_test_suite.py` invocations
  concurrently against the SAME `tst/build` directory (each does a full
  clean+rebuild there) — re-ran that one test file alone afterward and it
  passed; lesson re-confirmed from the A3 incident: never run more than one
  `run_test_suite.py`/build invocation against this repo at a time.

### Tasks C1/C2 — Newtonian geometric source terms (2026-08-07)

`src/coordinates/geometric_srcterms.{hpp,cpp}` (new), ported (math only)
from old Athena++'s `Cylindrical::AddCoordTermsDivergence` / `SphericalPolar
::AddCoordTermsDivergence`, onto this project's ΔA/ΔV `GeomData`
coefficients (the whole point of v2 plan Correction C2 — see below for what
went wrong when this distinction was briefly lost during implementation).
Wired into `Hydro::HydroSrcTerms`/`MHD::MHDSrcTerms` as a sibling of the
existing GR `CoordSrcTerms` call (mutually exclusive with it — the `else`
branch of the existing `is_general_relativistic`/`is_dynamical_relativistic`
checks), host-dispatched by `coord_general` exactly like `MeshGeometry`'s
constructor and `ReconDispatch` (one `switch`, no per-cell branch; a no-op
for `cartesian`).

- **Cylindrical/axisym (C1)**: `AddCylindricalSrcTerms<PhiInIM3,IsMHD>`, one
  templated kernel serving BOTH general cylindrical (`PhiInIM3=false`, phi
  resolved as grid x2, momentum in IM2) and axisym (`PhiInIM3=true`, phi
  carried as a non-grid rotational component in IM3 — see
  `geometry_cylindrical_axisym.cpp`'s handedness note). Two terms per old
  Athena++: `u(IM1) += dt*src1(i)*(rho*vphi²+P [+ 0.5*(BR²-Bphi²+Bz²) for
  MHD])` (centrifugal/pressure), and the Ju-thesis angular-momentum-
  conserving flux-average correction `u(IPHI) -= dt*src2(i)*(xf1(i)*
  flx1(IPHI,i) + xf1(i+1)*flx1(IPHI,i+1))`.
- **Proved algebraically (before writing any code) that IM2→IM3 needs NO
  extra sign flip for axisym**, despite IM3 physically storing `-v_phi`
  (left-handed system): both terms are applied to WHATEVER's actually
  stored in the slot, using fluxes the generic Riemann solver ALREADY
  computed FOR that stored value — the centrifugal term is quadratic
  (`vphi²` is sign-invariant under `Q=-vphi`), and the flux-average
  correction is linear but self-consistent: if the true equation for
  `q=rho*vphi` is `dq/dt+div(F(q))=S(q)` with `S` linear, then for
  `Q=-q`, the GENERICALLY-computed flux `F(Q)=-F(q)` (flux is itself
  linear in the transported variable), giving `dQ/dt+div(F(Q))=S(Q)` —
  the identical formula `S()` applies verbatim to `Q`. z-momentum (the
  OTHER slot, whichever it is) is untouched in both layouts: z is flat,
  matching old Athena++ exactly.
- **Spherical (C2)**: `AddSphericalSrcTerms<IsMHD>`, full (r,θ,φ) port —
  radial centrifugal/pressure term (`m_ii = rho*(vθ²+vφ²)+2P [+Br² MHD]`
  added to IM1), the r-flux-average correction for BOTH IM2 and IM3, the
  θ-momentum centrifugal term from φ-rotation (`m_pp = rho*vφ²+P [+
  magnetic]`, added to IM2 via `src1(i)*src1_j(j)`), and the φ-momentum
  term from θ-rotation, which uses the θ-flux-average form when `multi_d`
  or the direct local-product form otherwise (`use_x2_fluxes` branch,
  matching old Athena++ verbatim). `GeomData` gained `src1_j`/`src2_j`
  (j-indexed, spherical-only — old Athena++'s `coord_src1_j_`==
  `coord_src3_j_`, numerically identical, so only ONE array needed instead
  of two), filled in `geometry_spherical.cpp`.
- **The required 1D-radial layout's θ-term "inertness" falls out for
  free**: for `x2min=0,x2max=π` (the convention every 1D-radial input file
  in this project already uses), `src1_j = (sin(π)-sin(0))/|cos(0)-cos(π)|
  = 0/2 = 0` exactly — no `multi_d` special-casing needed for that specific
  term (unlike the θ-flux-vs-local-product branch for the φ-momentum term,
  which DOES need the explicit `use_x2_fluxes` check, since that's a choice
  of WHICH FORMULA, not a coefficient going to zero).
- **Real bug found and fixed via the well-balancedness equilibrium tests**
  (not caught by the B4 mass-conservation tests, which don't exercise the
  source term's own `m_pp`/`m_ii` pressure term at all in a way that would
  reveal a magnitude error — only the well-balancedness test does): a
  rotating-equilibrium test (uniform density, constant rotation speed,
  analytic `P(R)=P0+rho*v0²*ln(R/R0)` solving `dP/dR=rho*v0²/R`) showed an
  O(1) force imbalance from the FIRST timestep, not a slowly-growing
  truncation-error drift — the tell that this was a real bug, not
  numerics. Root cause: `w0(m,IEN,...)` in AthenaK's primitive array
  stores INTERNAL-ENERGY DENSITY for the ideal-gas EOS (`e`, with
  `pressure=(gamma-1)*e` via `EOS_Data::IdealGasPressure()`), NOT pressure
  directly — unlike old Athena++, where `prim(IEN,...)` IS pressure. Old
  Athena++'s formulas were ported with `w0(m,IEN,...)` used AS pressure
  verbatim, silently reading `e=P/(gamma-1)` (2.5x too large for
  `gamma=1.4`) wherever pressure was needed. Confirmed the fix is exactly
  right by first setting the test's rotation speed to zero (making the
  analytic pressure profile flat, `P=const`): with the bug, this "trivial"
  case ALSO showed the same O(1) error (proving the bug was in the P-only
  piece, not the rotation piece); after the fix (`eos.IdealGasPressure
  (w0(m,IEN,...))`), the `v0=0` case matches to roundoff (`~1e-19`) exactly
  as the ΔA/ΔV cancellation predicts analytically, and the `v0≠0` case
  gives `RMS-L1≈2e-4` after one step and stays bounded at `≈4e-4` (not
  growing) after ~2400 steps (`tlim=50`, many rotation periods) on both
  cylindrical_axisym and spherical_polar. Independently hand-verified the
  discrete ΔA/ΔV cancellation for the `v=0`, uniform-P case algebraically
  (not just numerically): for cylindrical, `Area1=R_f` (linear, not
  squared, unlike spherical's `Area1=r_f²`), so flux-divergence gives
  exactly `+P*src1(i)` for uniform P (ONE factor of `src1`, not two,
  since `ΔArea1 = dR = src1(i)*Vol(i)` by definition), matching the
  source term's `+src1(i)*(0+P)` exactly — confirming `m_pp` should
  contain ONE power of P for cylindrical, vs. spherical's `m_ii=...+2P`
  (TWO transverse curved directions, θ AND φ, each contributing a P), and
  that this distinction (already present in old Athena++, and preserved
  correctly in the port) was never the actual bug.
- **`OutputErrors()`-based test harness** (reused, not reinvented): the
  established `pgen_final_func` mechanism (see `linear_wave.cpp`) —
  `GeomEquilibriumTest()` writes the analytic equilibrium directly into
  conserved-variable register `u0` at t=0 (bypassing `PrimToCons`, since
  the closed-form conserved values are trivial: `IDN=rho0`, `IPHI=rho0*v0`,
  `IEN=P/(gamma-1)+0.5*rho0*v0²`), converts once to get a consistent `w0`,
  and registers a `pgen_final_func` that writes the SAME (time-invariant)
  analytic solution into register `u1` at end-of-run and calls the generic
  `ProblemGenerator::OutputErrors()`, which diffs `u0` vs `u1` and appends
  an RMS-L1/L∞ row to `<basename>-errs.dat` — identical mechanism to the
  linear-wave convergence tests, just with a static rather than
  time-evolving reference solution. New pgen: `geom_equilibrium_test.cpp`;
  new inputs: `ut_geom_equilibrium_{axisym,spherical}.athinput`; new test:
  `test_geom_equilibrium_cpu.py` (tolerance `1e-2`, generous relative to
  the observed `~4e-4`, specifically to still catch an O(1)-class bug like
  the one found above without being sensitive to resolution/parameter
  choices).
- **Header-include-order fragility discovered and fixed** (new failure
  mode, not previously documented): `mesh.hpp` <-> `meshblock.hpp` <->
  `meshblock_pack.hpp` <-> `coordinates/coordinates.hpp` form a header
  cycle broken only by include ORDER (whichever of these is entered FIRST
  in a translation unit determines whether `Mesh::FindMeshBlockIndex()`
  sees a complete `MeshBlock`/`MeshBlockPack`), and `eos/eos.hpp` includes
  `meshblock.hpp` directly, entering the same cycle from a third door. The
  new standalone `geometric_srcterms.cpp` hit BOTH failure modes (in
  succession, as the include order was first fixed one way then the
  other) before settling on the one working order: `mesh/mesh.hpp` literally
  first, before anything else in the group. Documented with a comment in
  `geometric_srcterms.hpp` so the next new coordinates-adjacent `.cpp` file
  doesn't rediscover this the hard way.
- **Verification**: coordinates-suite (29 tests, all passing, including the
  2 new equilibrium tests) and full suite green (see below for the
  full-suite count, confirmed via a solo, non-concurrent
  `run_test_suite.py` invocation per the B7 lesson above).

### Task D1 — Area/edge-length-weighted CT curl (2026-08-14)

Rewrote `src/mhd/mhd_ct.cpp`'s three update blocks (B1/B2/B3) from flat
`mbsize.dx2`/`dx3` division into Stokes form: `d(B1*Area1)/dt = -[e3*Len3]_
{j+1}+[e3*Len3]_j+[e2*Len2]_{k+1}-[e2*Len2]_k` (and cyclic for B2/B3),
reading the SAME `geom.Area1/2/3`/`Len1/2/3` accessors already used by the
flux divergence (A3/A4) and CFL (B5) kernels — no new `GeomData` fields
needed. Verified by hand that the flat-Cartesian reduction (`Area1=dx2*dx3`,
`Len3=dx3`, `Len2=dx2`, etc.) reproduces the old formula exactly for all
three B-components before touching any test.

- **Added a GR/dynamical-GR + curvilinear mutual-exclusion guard** in
  `Coordinates`'s constructor (`coordinates.cpp`) — a gap noticed while
  auditing D1: nothing previously stopped a user from combining
  `coord=spherical_polar` with GR, which would silently use curvilinear
  geometry factors alongside metric machinery that assumes Cartesian
  throughout. Fatal now, matching the plan's "Not touched" invariant.
- **Discovered and fixed a second, independent instance of the `history.cpp`-
  class bug**: `src/outputs/derived_variables.cpp`'s `mhd_divb` diagnostic
  used flat `dx1/dx2/dx3` division, not `geom.Area1/2/3`/`Vol` — found while
  scoping Task D2 (needed a *correct* div(B) diagnostic to build the test
  around) rather than by a failing test, since nothing in the existing
  suite exercises this specific output variable for curvilinear coords.
  Fixed to the same area-weighted flux-form divergence CT itself uses.
- **Verification**: full suite re-run on a dedicated compute node (see
  "Build environment" below for why) — 244 passed (was 240), 15 skipped, 0
  failed, including every existing Cartesian MHD linear-wave/shock-tube
  test at unchanged tolerances (confirms the Stokes-form rewrite is
  roundoff-identical on Cartesian, as the by-hand reduction predicted).

### Task D2 — div(B)=0 preservation test suite (2026-08-14)

`ct_divb_test.cpp`: B initialized as the discrete curl of an edge-centered
vector potential `A3(x1,x2)` (a smooth periodic-ish bump, generalizing
`orszag_tang.cpp`'s flat curl-of-A construction to curvilinear via
`B = curl_stokes(A)/Area`, using the identical `geom.Area1/2/Len3`
accessors CT itself reads), with a uniform x1 velocity added so CT's curl
terms actually fire during the run (not just checking the IC, which would
be a strictly weaker test already implied by B4's geometry-construction
tests). `pgen_final_func` computes `max|div(B)*Vol|` (using the SAME fixed
formula as the `mhd_divb` diagnostic above) normalized by a face-flux
scale, and fails above `1e-10` relative.

- **This is explicitly a topological check, not a physical one** (v2 plan
  Correction C6): since B is built from the SAME Area/Len tables the CT
  update reads, div(B)=0 is an exact discrete identity regardless of
  whether the CT curl is physically correct (a wrong edge length, wrong
  handedness, or a flipped EMF sign could still telescope to zero) — its
  actual job is catching an INCONSISTENCY between the tables used for
  construction vs. the divergence check (e.g. a copy-paste index bug), not
  verifying CT's physics. Task D3 is the real correctness gate.
- Tests on cartesian (control), cylindrical (R,φ,z), cylindrical_axisym
  (R,z), and spherical_polar (r,θ, avoiding the pole) — all 4 pass.
- **Incident**: an early attempt to verify this (and D1) via a *second*,
  concurrent `run_test_suite.py`/pytest invocation on the shared login node
  looked like a hang (30+ minute timeout on what should be a 2-second
  test). Root cause was NOT a bug: `uptime` showed load average 29-38 on a
  40-core shared interactive node from OTHER users' unrelated jobs — the
  test was just badly starved for CPU, not stuck. Fix (and now the
  standing convention for this project going forward): request a dedicated
  node via `srun -p p.sakura -N 1 -n 1 -c 40 --time=<T> bash -c '...'`
  before running anything but the smallest smoke test; confirmed the exact
  same "hung" test completes in ~5 seconds on a dedicated node. `squeue`
  also surfaced unrelated jobs already running under this account
  (`grass_rh`, `athenak_regr_cpu`, `run_comp...`) — left entirely alone,
  not associated with this work.

### Task D3 — Physical induction tests (2026-08-14)

The REAL correctness gate for Task D1's CT curl, per plan Correction C6.

- **Field-loop advection** (`ct_field_loop_test.cpp`): a weak (`amp=1e-3`)
  circular loop of poloidal field (curl of a compactly-supported "tent"
  `A_phi`, the classic Gardiner & Stone construction) in cylindrical_axisym
  (R,z) — the required 2D curvilinear layout. Advected by a UNIFORM
  z-velocity rather than R (z-translation is an exact symmetry of the
  axisym metric — `Area1/2/3`/`Len1/2/3` depend on R but not z — whereas
  R-translation is not, since the geometry itself changes with R; this is
  the curvilinear analogue of the flat test's usual diagonal advection).
  Domain periodic in z; after exactly one z-period the exact solution is
  IDENTICAL to the t=0 IC, so the same reused `pgen_final_func`-plus-
  `OutputErrors()` mechanism from `geom_equilibrium_test.cpp` applies
  directly (fill `u1`/`b1` with the SAME IC-setting function). RMS-L1
  tolerance `5e-3` (truncation-error scale, not roundoff — real advection
  with a real reconstruction scheme has genuine numerical diffusion).
- **1D-radial spherical stationarity** (`ct_monopole_stationarity_test.cpp`):
  in the required 1D-radial layout (`nx2=nx3=1`), CT's B1 update is
  unconditionally gated by `if (multi_d)`, which is false in 1D — there is
  no transverse direction to curl, so a purely radial field literally
  cannot evolve under CT there. Re-scoped this test (documented explicitly
  in the pgen's file docstring, not silently) to what's actually
  load-bearing in 1D: that NOTHING ELSE (flux-divergence magnetic-pressure
  terms, Task C1/C2 geometric source terms, floors) perturbs `B_r*Area1 =
  B_r*r^2` away from its exact constant while density/velocity/pressure
  genuinely evolve under a radial wind. Checked to `1e-12` relative
  directly (no truncation-error tolerance needed — nothing should touch it
  at all).
- **Verification**: both pass; full coordinates-suite re-run confirms no
  regression in D1/D2.

### Task E1 — Origin boundary-condition guardrail (2026-08-14)

Added to `Mesh::ValidateCoordGeneral()`: for any curvilinear system with
`x1min==0`, `ix1_bc` must be `reflect` (outflow/periodic/inflow at a
coordinate SINGULARITY, not an ordinary boundary, would be silently
unphysical — e.g. outflow lets mass "leak" through a point that isn't
actually a domain edge). No new BC code: the existing (unmodified) reflect
BC in `hydro_bcs.cpp` already flips only the normal component and mirrors
via the exact index convention (`ghost is-g` <-> `active is+g-1`) already
matched by Task B6/B7's ghost-geometry mirroring. Checked no existing
curvilinear input file in this project uses `x1min=0` with a non-reflect
BC before adding the guard (all of B4's origin-touching mass-conservation
inputs already used `reflect`). New validation tests: origin+outflow fails,
origin+reflect succeeds.

### Task E2 — Sign-flip and origin-conservation tests (2026-08-14)

`origin_conservation_test.cpp`, for cylindrical_axisym and spherical (1D
radial), both with `x1min=0`:
1. `geom.Area1` at the innermost face (`is`) is exactly `0.0` — checked
   directly (not just assumed from the geometry factories' formulas).
2. Ghost-zone data mirroring: for a manufactured, non-symmetric velocity
   field (all 3 components nonzero and mutually distinct, so the checks
   below can't be masked by an accidental symmetry), verifies the radial
   velocity is exactly negated and the two tangential velocities/density
   are exactly copied between each ghost cell and its mirror-index active
   partner, after boundary conditions have run.
3. No NaN/Inf anywhere in the active domain after 5 steps.

Total-mass-conservation-to-roundoff at the origin is NOT re-implemented
here — Task B4's cylindrical_axisym/spherical mass-conservation tests
already exercise `x1min=0` with `reflect` and already establish this;
duplicating it under a new name would add maintenance surface without new
coverage.

### Task F1 — Spherical radial Sod shock tube (2026-08-14)

No new pgen needed: the existing `shock_tube.cpp` classifies cells as
left/right of `xshock` using the flat Cartesian `CellCenterX` (arithmetic
midpoint) rather than `geom.x1v` (volumetric centroid) — but since the
initial condition is a pure step function (not a smooth profile), any
reasonable within-cell position estimate gives the identical left/right
classification everywhere except cells straddling the shock exactly (an
ambiguity every shock-tube implementation has, coordinate-agnostic). New
input `ut_f1_spherical_sod.athinput`: `coord=spherical_polar`, reflect at
BOTH `r=0` and the outer boundary (a fully closed domain), standard Sod
L/R states, `tlim=0.3` chosen so neither the shock nor rarefaction reaches
either wall (sound speed ~1.2 in the left state, 0.5 to each wall from the
shock). Checked total mass AND total energy conserved to roundoff (`1e-11`
relative) via the `.hst` output — the same closed-domain conservation
argument as Task B4's tests, now exercising a genuine discontinuous/
nonlinear Riemann problem instead of a smooth profile.

**Scope note**: the plan's phrase "shock-front L1 convergence under
refinement" is NOT implemented as a separate check here — Task F3 is the
dedicated, more rigorous convergence-order test (using a smooth pulse,
where a clean asymptotic rate is actually measurable; a shock's ~1st-order
convergence at a moving discontinuity is a fundamentally noisier
measurement to do well) and duplicating a weaker version of it under F1
was judged not worth the added test-suite runtime given everything else in
scope for this session.

### Task F2 — Magnetized rotating-disk equilibrium (2026-08-14)

`mhd_disk_equilibrium_test.cpp` extends Task C1/C2's pure-hydro rotating
equilibrium with a spatially UNIFORM toroidal field `B_phi=B0` (`B_R=B_z=
0`, carried in the same IM3/IB3 slot as `v_phi` for axisym). Worked out by
hand (see the pgen's file docstring for the full derivation): for
`B_R=B_z=0`, the radial MHS momentum flux `T_RR = P+0.5*B_phi^2` (magnetic
pressure adds to thermal pressure isotropically, since a purely toroidal
field is fully transverse to R) while the geometric source term's `m_pp =
rho*vphi^2+P-0.5*Bphi^2` (Task C1's `-Bphi^2` sign, i.e. magnetic TENSION
acts oppositely to magnetic pressure here) — combining these, the
equilibrium condition generalizes from the pure-hydro `dP/dR=rho*v0^2/R` to
`dP/dR = (rho0*v0^2 - B0^2)/R`: the SAME log profile with an effective
"centrifugal" term reduced by the magnetic tension of the uniform toroidal
field (a uniform-strength azimuthal field's hoop stress pulls inward,
partially or fully offsetting rotation's outward push — the correct,
physically-expected sign). Checked via the same `OutputErrors()` L1
mechanism (RMS-L1 tolerance `1e-2`, matching Task C1/C2's), PLUS an
explicit div(B) check after the run (should stay at roundoff, since a
uniform toroidal field with zero R/z components is trivially divergence-
free and nothing should perturb that). Both pass — this is the
integration-level check on Task C1's φ-slot sign conventions the plan asks
for, now exercised together with Task D1's CT update (both must be
consistent for this to hold over many orbits, not just at t=0).

### Task F3 — Formal convergence-order verification (2026-08-14)

`smooth_pulse_convergence_test.cpp`: a Gaussian density pulse at UNIFORM
background pressure and velocity is an EXACT (not just linearized) entropy-
mode perturbation — passively advected at the background velocity with
zero shape change, to full nonlinear accuracy, since it carries no
pressure or velocity perturbation to source any dynamics (entropy is a
materially-conserved Lagrangian quantity; nothing couples its growth to
the background pressure PROFILE, only to the local flow velocity). Used
the SAME z-periodic uniform-advection setup as Task D3's field-loop test
(cylindrical_axisym, translation in z being an exact symmetry, simpler
than exploiting cylindrical's rigid-rotation symmetry which was considered
and rejected: a `v_phi=const` background — the C1/C2 equilibrium profile —
shears a feature localized in both R and phi at different rates per
R-ring, so only a `v_phi=Omega*R` **rigid**-rotation profile would advect
without shape change there, an unnecessary extra layer of derivation when
axisym's z-translation already gives an exact, simple invariance for free).

- Ran at 3 resolutions (24²,48²,96²) via the same reused-`pgen_final_func`
  mechanism, reading consecutive RMS-L1 values from the (single, appended-
  to) `-errs.dat` file and checking the observed order (`log2` of the
  error ratio) falls in `[1.3, 2.8]` — a generous band around PLM's design
  order (2), since this is a genuine end-to-end integration measurement
  (reconstruction + flux divergence + time integration together), not the
  pristine asymptotic setup Tasks B6/B7's exact-reconstruction unit tests
  are.
- **First attempt failed**: at `sigma=0.15` (pulse width) and resolutions
  (16,32,64), the pulse was only ~1.2 cells wide at the lowest resolution
  (severely under-resolved), giving an observed order of 1.14 there —
  correctly reflecting that PLM's design order is an ASYMPTOTIC statement,
  not something guaranteed at any arbitrary coarse resolution. Fixed by
  widening the pulse (`sigma=0.25`) and shifting resolutions up (24,48,96)
  so even the coarsest run resolves it by several cells; re-ran and passed
  cleanly. Documented here since this was a test-TUNING issue, not a code
  bug — worth distinguishing for anyone re-reading this log later.

### Task G1 — SR geometric source term generalization (2026-08-14)

Added an `IsSR` compile-time bool to `AddCylindricalSrcTerms`/
`AddSphericalSrcTerms` (`geometric_srcterms.hpp`), dispatched in
`geometric_srcterms.cpp` from `pmbp->pcoord->is_special_relativistic`
(hydro only — see below). Confirmed from `src/hydro/rsolvers/
llf_hyd_singlestate.hpp`'s `SingleStateLLF_SRHyd` (which reads `wl.vx/vy/
vz` directly from the SAME primitive array `ReconCellT` fills) that
AthenaK stores `u^i = Gamma*v^i` — the spatial 4-velocity components, NOT
the 3-velocity — in the `IVX/IVY/IVZ` primitive slots whenever a run is
relativistic, and that `wgas = d + gamma*e` (`e` = internal energy
density) is the total enthalpy density `rho*h` the SR Riemann flux itself
uses. The generalization is then mechanical: every `rho*v_component^2`
term becomes `rho*h*u_component^2` (`rho`->`wgas`, `v`->`u`); the
flux-average correction terms are UNCHANGED, since they use whatever
momentum flux the (already SR-correct) Riemann solver produced, exactly as
for the Newtonian and MHD cases already handled.
- **SR+MHD is explicitly NOT implemented**: the SR momentum flux's
  magnetic contribution needs the comoving-frame field strength, not
  simply `bcc0` — a materially larger undertaking. `AddCoordGeomSrcTermsMHD`
  fatals if `is_special_relativistic` is set, rather than silently
  producing a wrong answer; `static_assert(!(IsMHD && IsSR))` in both
  kernel templates backs this up at compile time too.
  - **Real regression caught by the session-wide full-suite re-run below**:
    the FIRST version of this guard fired on `is_special_relativistic`
    ALONE, with no `coord_general != cartesian` check — but
    `AddCoordGeomSrcTermsMHD` is called unconditionally from the `else`
    branch of the GR check in `mhd_tasks.cpp` for EVERY non-GR MHD run,
    not just curvilinear ones, so this fataled every pre-existing,
    fully-supported plain SR+MHD+Cartesian run too (19 failures, all in
    `test_sr_lwave1d_cpu.py`/`test_sr_shocktube_cpu.py`'s `mhd-*`
    parametrizations — every SR MHD test in the suite). Fixed by adding
    the missing `coord_general != cartesian` condition. A reminder that
    "fatal if SR+MHD" and "fatal if SR+MHD+curvilinear" are NOT the same
    guard, and that a function reused across the cartesian/curvilinear
    split (this one is a no-op for cartesian via the switch below) needs
    its guards scoped to the SAME split, not just the new feature's
    precondition. Full suite re-run clean after the fix (see below).
- **Test** (`sr_geom_equilibrium_test.cpp`): the v/c->0 cross-check the
  plan calls out explicitly. At small `v0` (0.05c), the SR equilibrium
  reduces to the SAME leading-order log-pressure profile as the Newtonian
  case (`geom_equilibrium_test.cpp`) — relativistic corrections enter at
  `O(v0^2/c^2)` ON TOP of that same leading-order balance, not as a
  different leading-order physics, so reusing the identical profile
  formula at small `v0` is the correct test, not an approximation of one.
  Used `PrimToCons()` (not a hand-rolled SR conserved-energy formula) for
  both the t=0 IC and the `pgen_final_func` reference solution (built in a
  locally-allocated scratch primitive array, since there is no persistent
  SR-aware `w1` register) — deliberately reusing the EOS class's own,
  already-tested SR energy formula rather than re-deriving it inside a
  test meant to verify a DIFFERENT piece of physics. Passes at `RMS-L1 <
  1e-2`, the same tolerance as the Newtonian equilibrium test.

### Session-wide verification (Tasks D1-G1, 2026-08-14)

Full suite (`run_test_suite.py --cpu`, dedicated node): **254 passed (was
240 at the end of Task C2), 15 skipped, 0 failed.** All coordinates-suite
tests (43 total, up from 29) plus the full non-coordinates suite pass. New
coordinates-suite tests added this session: `test_ct_divb_cpu.py` (4),
`test_ct_induction_cpu.py` (2), `test_coord_input_validation_cpu.py` (+2),
`test_origin_conservation_cpu.py` (2), `test_f1_spherical_sod_cpu.py` (1),
`test_mhd_disk_equilibrium_cpu.py` (1), `test_f3_convergence_cpu.py` (1),
`test_sr_geom_equilibrium_cpu.py` (1) — 14 net new tests (29+14=43,
matches); the +10 vs. the 240->254 full-suite delta is Task D1's 4
`test_ct_divb_cpu.py` tests, already counted at that task's own
intermediate full-suite run, plus these 10. All test artifacts (`.hst`/
`.dat`/`.tab` files, and a manually-created `tst/athena` convenience
symlink used only for interactive debugging outside `run_test_suite.py`)
cleaned up afterward; nothing left behind beyond the source/input files
themselves.

**First full-suite run after G1 caught a real regression** (19 failures,
all SR+MHD tests) from the SR+MHD guard bug described in the Task G1 log
above — fixed, and this final, clean 254-passed run is AFTER that fix,
not before it. This is exactly the kind of thing the "always re-run the
full suite, not just the new task's own tests" discipline exists to catch:
the broken guard would never have shown up in the coordinates-suite alone
(it only affects a purely Cartesian, non-curvilinear code path).

**Testing-infrastructure lesson carried forward from the D1/D2 incident**:
this login node is shared and can have load average 30+ from unrelated
users/jobs at any time, which is indistinguishable from a hang without
checking `uptime`/`squeue` first. Every build and test run in this section
was done via `srun -p p.sakura -N 1 -n 1 -c 40 --time=<T> bash -c '...'`
for a dedicated node — this is now the standing convention for this
project, not just a one-off fix.
