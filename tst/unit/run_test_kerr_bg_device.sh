#!/bin/bash
# Build (and optionally run) tst/unit/test_kerr_bg_device.cpp (V0, K-001) against an existing
# athena GPU build directory: that build's test_cfc_puncture compile flags, its Kokkos
# libraries and its own compiled src/cfc/kerr_background.cpp object (the loader under test).
# KSTAB_INC = the directory of kstab.hpp (kslice/kstab in the PRIVATE spinning_trumpet
# repository; not part of AthenaK). Not a CMake target (see run_test_cfc_puncture_device.sh).
#
#   KSTAB_INC=DIR run_test_kerr_bg_device.sh BUILD_DIR [OUT_EXE] [--run F2_PREFIX KSTAB F4 REF]
set -euo pipefail
B=$(readlink -f "$1")
OUT=${2:-$B/test_kerr_bg_device}
SRC=$(cd "$(dirname "$0")/../.." && pwd)
: "${KSTAB_INC:?set KSTAB_INC to the directory holding kstab.hpp}"
F=$B/CMakeFiles/test_cfc_puncture.dir/flags.make
[ -f "$F" ] || F=$B/src/CMakeFiles/athena.dir/flags.make
OBJ=$B/src/CMakeFiles/athena.dir/cfc/kerr_background.cpp.o
[ -f "$OBJ" ] || { echo "missing $OBJ (build athena first)"; exit 1; }
CXX=$(sed -n 's/^# compile CXX with //p' "$F")
DEFS=$(sed -n 's/^CXX_DEFINES = //p' "$F")
INCS=$(sed -n 's/^CXX_INCLUDES = //p' "$F")
FLAGS="$(sed -n 's/^CXX_FLAGS = //p' "$F") -fp-model=precise ${EXTRA_CXXFLAGS:-}"
cd "$B"
eval "$CXX $DEFS $INCS -I$KSTAB_INC $FLAGS -c $SRC/tst/unit/test_kerr_bg_device.cpp -o ${OUT}.o"
eval "$CXX -O3 -DNDEBUG -fsycl -fno-sycl-id-queries-fit-in-int -fsycl-dead-args-optimization \
  -DDESUL_SYCL_DEVICE_GLOBAL_SUPPORTED ${OUT}.o $OBJ -o $OUT \
  kokkos/containers/src/libkokkoscontainers.a kokkos/algorithms/src/libkokkosalgorithms.a \
  kokkos/core/src/libkokkoscore.a -ldl kokkos/simd/src/libkokkossimd.a"
echo "built $OUT"
if [[ "${3:-}" == "--run" ]]; then shift 3; "$OUT" "$@"; fi
