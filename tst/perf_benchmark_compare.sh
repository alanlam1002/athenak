#!/bin/bash
# CPU zone-cycles/sec throughput benchmark for the curvilinear-coordinate work.
#
# Runs TWO independent measurement axes, which answer two different questions:
#
#   AXIS A -- REGRESSION.  "Did adding curvilinear support slow down the Cartesian path?"
#     Same input, same coord=cartesian, run against a pristine git-worktree build at a
#     reference commit and against the current working tree. Uses
#     tst/inputs/perf_cartesian_benchmark.athinput (orszag_tang), which is the only
#     benchmark pgen that exists on BOTH sides of the feature.
#
#   AXIS B -- COORDINATE COST.  "How much slower is curvilinear than Cartesian?"
#     Current build only (the pgen does not exist at the pre-feature base), sweeping
#     tst/inputs/perf_coord_{cartesian,cylindrical,axisym,spherical}.athinput, which are
#     byte-identical apart from the "coord =" line. Reported as a ratio against the
#     cartesian control from the same build.
#
# Both axes sweep reconstruction {plm, ppm4}, because the two methods exercise very
# different amounts of the curvilinear machinery: plm goes through the generalized
# non-uniform limiter in ALL THREE directions, while ppm4's curvilinear coefficient path
# is x1-only. ppm4 needs >=3 ghost zones, so it is run with mesh/nghost=4 overridden from
# the command line; plm runs at each input file's native nghost. Do not compare a plm
# number against a ppm4 number -- compare across a row (coordinate systems), or across
# builds (before/after).
#
# CPU/OpenMP only: this development cluster has no GPU and no CUDA module (sinfo reports
# GRES=(null) on every node). The GPU equivalent of this matrix must be run on a GPU
# machine before any claim is made about device performance -- see DEVELOPMENT.md.
#
# Builds are cached under $PERF_ROOT (default /sakura/ptmp/tlam/athenak_perf) rather than
# a temp dir, so re-running the measurement after a code change only rebuilds what changed
# and the reference build is done exactly once. Pass --rebuild to force a clean rebuild.
#
# Usage:
#   ./perf_benchmark_compare.sh [reference_commit]   # both axes (default ref: HEAD)
#   ./perf_benchmark_compare.sh --coord-only         # axis B only, skips the ref build
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NREPS=${NREPS:-3}
PERF_ROOT="${PERF_ROOT:-/sakura/ptmp/tlam/athenak_perf}"
COORD_ONLY=0
REBUILD=0
REF_COMMIT="HEAD"
for arg in "$@"; do
  case "$arg" in
    --coord-only) COORD_ONLY=1 ;;
    --rebuild)    REBUILD=1 ;;
    *)            REF_COMMIT="$arg" ;;
  esac
done

CMAKE_FLAGS=(-DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_BUILD_TYPE=Release
             -DKokkos_ARCH_NATIVE=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_SERIAL=ON)

# ---------------------------------------------------------------------------- builds ---
AFTER_DIR="$PERF_ROOT/after"
[ "$REBUILD" -eq 1 ] && rm -rf "$AFTER_DIR"
mkdir -p "$AFTER_DIR"
echo "Building current working tree in $AFTER_DIR ..."
cd "$AFTER_DIR"
cmake "${CMAKE_FLAGS[@]}" "$REPO_ROOT" > cmake.log 2>&1
make -j"$(nproc)" > make.log 2>&1 || { echo "BUILD FAILED, see $AFTER_DIR/make.log"; exit 1; }
AFTER_BIN="$AFTER_DIR/src/athena"

if [ "$COORD_ONLY" -eq 0 ]; then
  WORKTREE_DIR="$PERF_ROOT/ref_${REF_COMMIT}"
  [ "$REBUILD" -eq 1 ] && { cd "$REPO_ROOT"; git worktree remove --force "$WORKTREE_DIR" \
      2>/dev/null || rm -rf "$WORKTREE_DIR"; }
  if [ ! -d "$WORKTREE_DIR" ]; then
    echo "Creating reference worktree (${REF_COMMIT}) ..."
    cd "$REPO_ROOT"
    git worktree add "$WORKTREE_DIR" "$REF_COMMIT" > /dev/null
    cd "$WORKTREE_DIR"
    # kokkos/ is a submodule; `git worktree add` does not populate it.
    git submodule update --init --recursive > /dev/null
  fi
  echo "Building reference (${REF_COMMIT}) ..."
  mkdir -p "$WORKTREE_DIR/build_perf" && cd "$WORKTREE_DIR/build_perf"
  cmake "${CMAKE_FLAGS[@]}" .. > cmake.log 2>&1
  make -j"$(nproc)" > make.log 2>&1 || {
    echo "REFERENCE BUILD FAILED, see $WORKTREE_DIR/build_perf/make.log"; exit 1; }
  BEFORE_BIN="$WORKTREE_DIR/build_perf/src/athena"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# Runs the benchmark NREPS times and echoes the MEAN zone-cycles/cpu_second.
# $1 = binary, $2 = input file, remaining args = block/par=value overrides.
# NOTE: all arithmetic here goes through awk, NOT bc. AthenaK prints the metric in
# scientific notation ("4.479700e+06"), which bc cannot parse at all -- it emits
# "(standard_in): syntax error" to stderr and an empty result to stdout, so every number
# silently becomes 0.00e+00 and every comparison reads as "no difference". That is exactly
# the failure mode that made the first version of this script report all zeros.
run_bench() {
  local bin="$1" input="$2"; shift 2
  local dir vals v
  dir=$(dirname "$bin")
  cd "$dir"
  vals=""
  for _ in $(seq 1 "$NREPS"); do
    rm -f ./*.hst ./*.dat
    v=$("$bin" -i "$input" "$@" 2>/dev/null \
        | grep "zone-cycles/cpu_second" | awk '{print $3}')
    if [ -z "$v" ]; then
      echo "ERROR: no zone-cycles line from: $bin -i $input $*" >&2
      exit 1
    fi
    vals="$vals $v"
  done
  echo "$vals" | awk '{s=0; for(i=1;i<=NF;i++) s+=$i; printf "%.6e", s/NF}'
}

fmt() { awk -v x="$1" 'BEGIN{printf "%.4e", x}'; }
# $1=baseline $2=measured -> percent slower (negative means faster)
pct_slower() { awk -v b="$1" -v m="$2" 'BEGIN{printf "%.2f", (1 - m/b)*100}'; }

# --------------------------------------------------------------- AXIS A: regression ---
if [ "$COORD_ONLY" -eq 0 ]; then
  echo ""
  echo "==========================================================================="
  echo "AXIS A -- Cartesian regression vs ${REF_COMMIT}"
  echo "  input: tst/inputs/perf_cartesian_benchmark.athinput (orszag_tang, 400x400)"
  echo "==========================================================================="
  printf "%-8s %14s %14s %10s\n" "recon" "before" "after" "slower"
  INPUT_A="$REPO_ROOT/tst/inputs/perf_cartesian_benchmark.athinput"
  for recon in plm ppm4; do
    ov=(mhd/reconstruct=$recon)
    [ "$recon" == "ppm4" ] && ov+=(mesh/nghost=4)
    b=$(run_bench "$BEFORE_BIN" "$INPUT_A" "${ov[@]}")
    a=$(run_bench "$AFTER_BIN"  "$INPUT_A" "${ov[@]}")
    printf "%-8s %14s %14s %9.2f%%\n" "$recon" "$(fmt "$b")" "$(fmt "$a")" \
           "$(pct_slower "$b" "$a")"
  done
fi

# ---------------------------------------------------------- AXIS B: coordinate cost ---
echo ""
echo "==========================================================================="
echo "AXIS B -- coordinate-system cost (current build)"
echo "  inputs: tst/inputs/perf_coord_*.athinput (ct_divb_test, 400x400,"
echo "          identical apart from the 'coord =' line)"
echo "==========================================================================="
printf "%-8s %-22s %14s %10s\n" "recon" "coord" "zc/cpu_s" "vs cart"
for recon in plm ppm4; do
  ov=(mhd/reconstruct=$recon)
  [ "$recon" == "ppm4" ] && ov+=(mesh/nghost=4)
  cart=""
  for cfg in cartesian:cartesian cylindrical:cylindrical \
             axisym:cylindrical_axisym spherical:spherical_polar; do
    f="${cfg%%:*}"; label="${cfg#*:}"
    v=$(run_bench "$AFTER_BIN" "$REPO_ROOT/tst/inputs/perf_coord_${f}.athinput" "${ov[@]}")
    if [ -z "$cart" ]; then
      cart="$v"
      printf "%-8s %-22s %14s %10s\n" "$recon" "$label" "$(fmt "$v")" "(baseline)"
    else
      printf "%-8s %-22s %14s %9.2f%%\n" "$recon" "$label" "$(fmt "$v")" \
             "$(pct_slower "$cart" "$v")"
    fi
  done
done
echo ""
