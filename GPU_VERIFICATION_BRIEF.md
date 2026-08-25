# Brief: verifying the curvilinear branch on a GPU cluster

You are picking up a task that was prepared on a machine with **no GPU**. Everything
below has been done; none of it has ever been compiled for or run on a device.

Read this file first, then `DEVELOPMENT.md` -> "Stage 5 transfer checklist" for the
technical detail. Do not re-derive the design from the code; it is written up already.

## State you are inheriting

| | |
|---|---|
| branch | `curvilinear_coordinate` |
| HEAD | `8a9fcbec` "Prepare for GPU verification on a device cluster" |
| kokkos submodule | `6739bc62` (matches upstream; `git submodule update --init` if empty) |
| CPU suite | 275 passed / 15 skipped / 5 failed — the 5 are explained below |
| device compiles | **zero, ever** |

The branch extends AthenaK from Cartesian-only to cylindrical, cylindrical_axisym (R,z)
and spherical_polar, selected by `<mesh>/coord`, including both SMR/AMR phases
(cell-centered and face-centered/MHD).

## The objective

Get the **21 curvilinear `_gpu` tests** to pass on a device, and establish that the
curvilinear code paths are actually correct there — not merely that they compile.

```
tst/test_suite/coordinates/
  test_geom_curvilinear_construction_gpu.py   4    geometry tables
  test_recon_exact_gpu.py                     6    PLM/PPM coefficient tables
  test_ct_divb_gpu.py                         4    CT div(B) preservation
  test_amr_conservation_gpu.py                3    cell-centered SMR/AMR (1D radial)
  test_amr_divb_gpu.py                        4    face-centered SMR/AMR (2D)
```

## Passing those 21 tests is NOT sufficient

This is the most important thing in this file. `GeomData::cells_uniform` and
`GeomData::cubic_cells` are **host-computed booleans captured into device kernels**. They
select between upstream's original arithmetic and the curvilinear generalization. If
either is mis-set on device, curvilinear runs still take the curvilinear branch and pass,
while **Cartesian runs silently take the wrong branch and nobody notices**.

"Verified" therefore requires all three:

1. the 21 curvilinear `_gpu` tests pass;
2. the **Cartesian** AMR `_gpu` tests still pass on device — `test_nr_lwave3d_amr_gpu.py`
   and its `sr/`, `gr/`, `rad/`, `z4c/`, `dyngrmhd/` siblings. Note that the *sharpest*
   Cartesian AMR MHD div(B) test, `tst/test_suite/nr/test_nr_divb_amr_mpicpu.py`, is
   **MPI-CPU only and will not run under `--gpu`**. Check it by hand instead: run
   `tst/inputs/divb_amr_2d.athinput` with the device binary and confirm
   `max(max_ndiv)` in the `.user.hst` stays ~`4e-16` (add
   `mesh_refinement/max_nmb_per_rank=4096` if running on one rank). The cartesian cases
   inside `test_amr_divb_gpu.py` / `test_amr_conservation_gpu.py` also serve as fast-path
   controls, since Cartesian is exactly what takes the `cells_uniform` / `cubic_cells`
   branch;
3. one curvilinear AMR MHD run agrees **host vs device** to reduction-ordering tolerance
   — e.g. run `tst/inputs/ut_amr_divb_spherical.athinput` on both and compare the
   reported `rel_divb` and `rel_mass_drift`.

## Start here

```bash
cd tst
# subset first -- a device compile of the whole tree is slow, and these are the
# kernels that are actually unverified
python3 run_test_suite.py --gpu "<backend flags>" --test test_suite/coordinates
```

`--gpu` defaults to CUDA but no longer forces it: it injects `-DKokkos_ENABLE_CUDA=On`
only if you have not already named a Kokkos backend. So:

```bash
--gpu "-DKokkos_ARCH_<ARCH>=On"                                    # NVIDIA
--gpu "-DKokkos_ENABLE_SYCL=On -DKokkos_ARCH_INTEL_PVC=ON"         # Intel PVC / Aurora
--gpu "-DKokkos_ENABLE_HIP=On -DKokkos_ARCH_<AMDARCH>=ON"          # AMD
```

Only once that subset is green, run the full suite.

## This is not the machine the work was done on

The previous machine was `sakura` (MPCDF). **Do not copy its conventions.** Discover the
local ones instead: scheduler and partition names, module names, whether compute nodes
see `/tmp` (on sakura they did not — `/tmp` is node-local there, which silently broke
several runs), and where scratch lives. Build and test on a compute/GPU node, not a login
node.

## Ranked risks, and what each looks like

1. **Compile failure in a pgen.** Device compilers are far stricter than g++ about what
   may appear in a `KOKKOS_LAMBDA`. The convention here is upstream's: a free
   `KOKKOS_INLINE_FUNCTION` taking a POD, never a `KOKKOS_LAMBDA` captured into another
   one (nvcc restricts nested extended lambdas). One instance was already rewritten for
   this reason (`A3Fn` in `amr_divb_test.cpp`). If something fails to compile, look for
   that pattern first, and for `std::` math inside device code.
2. **`GeomData` capture size — measured at 2048 bytes**, exactly 4x CUDA's 512 B
   `ConstantMemoryUseThreshold` (51 `Kokkos::View` handles at 40 B). Every geometry kernel
   passes its closure through global rather than constant memory. **Correctness is
   unaffected**; this is the first thing to look at if curvilinear kernels are
   disproportionately slow relative to their ~7% (plm) / ~2% (ppm4) CPU cost. The fix is
   per-consumer sub-structs, described in `DEVELOPMENT.md`. Do not attempt it before you
   have a measurement showing it matters.
3. **Division by a legitimately zero area.** At `x1min=0` the spherical factory sets
   `a1i(is)` to exactly `0` and `mhd_ct.cpp` divides by `Area1`; correctness relies on the
   numerator vanishing too. A `div(B)` check returning **NaN** (rather than merely large)
   points at that division, not at the CT stencil. The Phase 2 kernels guard their own
   divides (`WeightedMean`, `ToB`), so a NaN out of those is a genuine surprise.
4. **Reductions.** The acceptance gates use `Kokkos::parallel_reduce` with `Sum<Real>` /
   `Max<Real>`. Different reduction ordering on device changes the last digits. A gate
   missing by an ulp or two is expected, not a defect. The div(B) gates carry six orders
   of margin and should be insensitive; the mass gates are tighter.

## Expected differences vs. real bugs

- **The 5 CPU failures are NOT ours.** They are upstream's RKL2 controller test calling
  `math.nextafter`, which needs Python 3.9+; sakura had 3.8. On a newer Python they should
  **pass**. If they fail on your machine, read the actual error — it is something new.
- Reference numbers to compare against, from CPU (`rel_divb`, tolerance 1e-10):
  cartesian `2.13e-16`, cylindrical `2.04e-16`, axisym `4.04e-16`, spherical `2.96e-16`.
  Device values should be the same order. Anything at 1e-3 or worse is a real failure, not
  reduction noise.
- `mass_tol` in the `ut_amr_divb_*` inputs is deliberately `1e-9`, not `1e-11`. A
  pre-existing reflecting-wall effect floors single-level drift at ~9.4e-11. That is
  documented in the input files; do not "fix" it by tightening the tolerance.

## Constraints

- **Do not push anything**, to any remote.
- **Summarize and ask before every `git commit`.** Approval for one commit does not carry
  to the next. This is a standing rule from the user's global config.
- Do not refactor the curvilinear design, the fast-path predicates, or the upstream merge
  surface. If a device fix seems to require it, say so and stop.

## Useful background, if you need it

- `DEVELOPMENT.md` — full technical log. Relevant sections: "Performance and GPU status",
  "Stage 5 transfer checklist", "Deferred (not implemented in this project)" for the two
  SMR/AMR phases.
- A **bounds-checked build** is worth running once on device:
  `-DKokkos_ENABLE_DEBUG=On -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=On`. On CPU it caught a
  real out-of-bounds read that a Release-only suite had passed through review and commit.
  Device backends have their own indexing paths, so it is worth repeating there.
- There is an unfiled upstream bug report about `ProlongFCInternal` being cubic-cells-only.
  It is **not** a problem with this branch (this branch fixes it) and is not your task.
