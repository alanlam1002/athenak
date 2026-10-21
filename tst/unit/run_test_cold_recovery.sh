#!/bin/bash
# Build and run test_cold_recovery.cpp (R-083 tests 1-2) with an existing CPU build's
# compiler, flags and Kokkos. Header-only use of the PrimitiveSolver, so it needs only
# unit_system.cpp.o and the Kokkos libraries from that build (no `athena` rebuild).
# Usage: bash tst/unit/run_test_cold_recovery.sh [cpu_build_dir]   (default build_cpu)
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD="${1:-${SRC}/build_cpu}"
BUILD="$(cd "$BUILD" && pwd)"
FLAGS_MAKE="$BUILD/src/CMakeFiles/athena.dir/flags.make"
LINK_TXT="$BUILD/src/CMakeFiles/athena.dir/link.txt"
CXX=$(sed -n 's/^# compile CXX with //p' "$FLAGS_MAKE")
CXX_DEFINES=$(sed -n 's/^CXX_DEFINES = //p' "$FLAGS_MAKE")
CXX_INCLUDES=$(sed -n 's/^CXX_INCLUDES = //p' "$FLAGS_MAKE")
CXX_FLAGS=$(sed -n 's/^CXX_FLAGS = //p' "$FLAGS_MAKE")
LINK_LIBS=""
for tok in $(cat "$LINK_TXT"); do
  case "$tok" in
    *kokkos*.a|*kokkos*.so|-l*|-L*|-Wl,*) LINK_LIBS="$LINK_LIBS $tok" ;;
  esac
done
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
$CXX $CXX_DEFINES $CXX_INCLUDES $CXX_FLAGS -c "${SRC}/tst/unit/test_cold_recovery.cpp" \
  -o "$WORK/t.o"
(cd "$BUILD/src" && $CXX $CXX_FLAGS "$WORK/t.o" \
  CMakeFiles/athena.dir/eos/primitive-solver/unit_system.cpp.o -o "$WORK/t" $LINK_LIBS)
"$WORK/t"
