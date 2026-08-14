#!/bin/bash
# Task A5: CPU zone-cycles/sec throughput regression check.
#
# Compares the current working tree's throughput against a pristine git-worktree build
# at a given reference commit (default: current HEAD), using the fixed-cycle-count
# Orszag-Tang benchmark at tst/inputs/perf_cartesian_benchmark.athinput (400x400 2D MHD,
# single MeshBlock, 50 cycles, no I/O). No CUDA/nvcc is available on this development
# node (see DEVELOPMENT.md), so this is a CPU/OpenMP throughput check, not GPU -- run the
# equivalent comparison on a GPU machine (with -DKokkos_ENABLE_CUDA=On) before relying on
# this as evidence of GPU performance, per the plan's Task A5 note.
#
# Usage: ./perf_benchmark_compare.sh [reference_commit]
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF_COMMIT="${1:-HEAD}"
WORKTREE_DIR=$(mktemp -d /tmp/athenak_perf_before.XXXXXX)
NREPS=3

cleanup() {
  cd "$REPO_ROOT"
  git worktree remove --force "$WORKTREE_DIR" 2>/dev/null || rm -rf "$WORKTREE_DIR"
}
trap cleanup EXIT

echo "Building reference (${REF_COMMIT}) in ${WORKTREE_DIR} ..."
cd "$REPO_ROOT"
git worktree add "$WORKTREE_DIR" "$REF_COMMIT" > /dev/null
cd "$WORKTREE_DIR"
git submodule update --init --recursive > /dev/null
mkdir -p build_perf && cd build_perf
cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_BUILD_TYPE=Release \
  -DKokkos_ARCH_NATIVE=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_SERIAL=ON \
  .. > cmake.log 2>&1
make -j"$(nproc)" > make.log 2>&1
BEFORE_BIN="$WORKTREE_DIR/build_perf/src/athena"

echo "Building current working tree in a scratch dir ..."
AFTER_DIR=$(mktemp -d /tmp/athenak_perf_after.XXXXXX)
cd "$AFTER_DIR"
cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_BUILD_TYPE=Release \
  -DKokkos_ARCH_NATIVE=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_SERIAL=ON \
  "$REPO_ROOT" > cmake.log 2>&1
make -j"$(nproc)" > make.log 2>&1
AFTER_BIN="$AFTER_DIR/src/athena"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
INPUT="$REPO_ROOT/tst/inputs/perf_cartesian_benchmark.athinput"

run_bench() {
  local bin="$1" dir
  dir=$(dirname "$bin")
  cd "$dir"
  rm -f PerfBenchmark.hst *.dat
  "$bin" -i "$INPUT" 2>/dev/null | grep "zone-cycles/cpu_second" | awk '{print $3}'
}

echo "=== BEFORE (${REF_COMMIT}) ==="
before_sum=0
for i in $(seq 1 $NREPS); do
  v=$(run_bench "$BEFORE_BIN")
  echo "  run $i: $v"
  before_sum=$(echo "$before_sum + $v" | bc)
done
before_avg=$(echo "$before_sum / $NREPS" | bc -l)

echo "=== AFTER (current working tree) ==="
after_sum=0
for i in $(seq 1 $NREPS); do
  v=$(run_bench "$AFTER_BIN")
  echo "  run $i: $v"
  after_sum=$(echo "$after_sum + $v" | bc)
done
after_avg=$(echo "$after_sum / $NREPS" | bc -l)

ratio=$(echo "$after_avg / $before_avg" | bc -l)
pct=$(echo "(1 - $ratio) * 100" | bc -l)
echo ""
echo "before avg: $before_avg zone-cycles/cpu_second"
echo "after  avg: $after_avg zone-cycles/cpu_second"
echo "slowdown:   ${pct}%"

rm -rf "$AFTER_DIR"
