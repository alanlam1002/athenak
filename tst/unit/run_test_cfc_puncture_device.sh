#!/bin/bash
# Build (and optionally run) tst/unit/test_cfc_puncture_device.cpp against an existing athena
# GPU build directory, with that build's own test_cfc_puncture compile flags and Kokkos
# libraries (C-094 P1). Deliberately not a CMake target: adding one would re-run cmake on
# the production build tree.
#
#   run_test_cfc_puncture_device.sh BUILD_DIR [OUT_EXE] [--run]
#
# --run executes it (needs a GPU node; run on a compute node of the debug queue).
set -euo pipefail
B=$(readlink -f "$1")
OUT=${2:-$B/test_cfc_puncture_device}
SRC=$(cd "$(dirname "$0")/../.." && pwd)
F=$B/CMakeFiles/test_cfc_puncture.dir/flags.make
CXX=$(sed -n 's/^# compile CXX with //p' "$F")
DEFS=$(sed -n 's/^CXX_DEFINES = //p' "$F")
INCS=$(sed -n 's/^CXX_INCLUDES = //p' "$F")
FLAGS=$(sed -n 's/^CXX_FLAGS = //p' "$F")
cd "$B"
eval "$CXX $DEFS $INCS $FLAGS -c $SRC/tst/unit/test_cfc_puncture_device.cpp -o ${OUT}.o"
eval "$CXX -O3 -DNDEBUG -fsycl -fno-sycl-id-queries-fit-in-int -fsycl-dead-args-optimization \
  -DDESUL_SYCL_DEVICE_GLOBAL_SUPPORTED ${OUT}.o -o $OUT \
  kokkos/containers/src/libkokkoscontainers.a kokkos/algorithms/src/libkokkosalgorithms.a \
  kokkos/core/src/libkokkoscore.a -ldl kokkos/simd/src/libkokkossimd.a"
echo "built $OUT"
if [[ "${3:-}" == "--run" ]]; then "$OUT"; fi
