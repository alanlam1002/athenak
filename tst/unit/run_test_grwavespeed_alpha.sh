#!/bin/bash
# Build and run test_grwavespeed_alpha.cpp (the GRWAVESPEED_ALPHA_HANDOFF.md regression
# test) against an EXISTING `athena` CPU build, without adding a new CMake target.
#
# Why not a CMake target (unlike test_cfc_puncture): this test needs the real
# PrimitiveSolverHydro::GetGRFastMagnetosonicSpeeds, which (via primitive_solver_hyd.hpp
# -> dyn_grmhd.hpp -> mesh.hpp/driver.hpp/...) pulls in most of the codebase and
# ParameterInput, unlike cfc_puncture.hpp's Mesh-independent math. Reusing the `athena`
# target's own object files (Makefile CXX_INCLUDES/CXX_FLAGS, everything but main.cpp.o)
# gets this for free without restructuring the monolithic `athena` add_executable() --
# which risks the GPU/SYCL relocatable-device-code build (see AthenaK GPU build notes)
# for a test-only change. This script is the reviewable, reproducible substitute.
#
# Usage: bash tst/unit/run_test_grwavespeed_alpha.sh [build_dir]
#   build_dir defaults to build_cpu (relative to the repo root). Must be a CPU
#   (non-SYCL/CUDA) build -- this test runs its assertions on the host.
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD="${1:-${SRC}/build_cpu}"
[ -d "$BUILD" ] || { echo "error: build dir '$BUILD' not found (build it first, e.g. bash build_aurora.sh cpu)"; exit 1; }
BUILD="$(cd "$BUILD" && pwd)"

echo "==> Rebuilding athena in $BUILD (picks up any source changes) ..."
make -C "$BUILD" athena -j 16

FLAGS_MAKE="$BUILD/src/CMakeFiles/athena.dir/flags.make"
LINK_TXT="$BUILD/src/CMakeFiles/athena.dir/link.txt"
[ -f "$FLAGS_MAKE" ] && [ -f "$LINK_TXT" ] || { echo "error: $FLAGS_MAKE / $LINK_TXT not found -- unexpected CMake generator?"; exit 1; }

CXX=$(sed -n 's/^# compile CXX with //p' "$FLAGS_MAKE")
CXX_DEFINES=$(sed -n 's/^CXX_DEFINES = //p' "$FLAGS_MAKE")
CXX_INCLUDES=$(sed -n 's/^CXX_INCLUDES = //p' "$FLAGS_MAKE")
CXX_FLAGS=$(sed -n 's/^CXX_FLAGS = //p' "$FLAGS_MAKE")
# Every linker-relevant token on athena's own link line -- libraries (exact .a/.so
# suffix; a naive substring grep also matches e.g. "...aurora_test..." in an rpath
# directory name) plus -l/-L/-Wl, flags. Whatever this build actually needs.
LINK_LIBS=""
for tok in $(cat "$LINK_TXT"); do
  case "$tok" in
    *.a|*.so|-l*|-L*|-Wl,*) LINK_LIBS="$LINK_LIBS $tok" ;;
  esac
done

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

echo "==> Compiling test_grwavespeed_alpha.cpp ..."
$CXX $CXX_DEFINES $CXX_INCLUDES $CXX_FLAGS \
  -c "${SRC}/tst/unit/test_grwavespeed_alpha.cpp" -o "$WORK/test_grwavespeed_alpha.o"

echo "==> Linking against athena's own objects (all but main.cpp.o) ..."
# link.txt's library paths (e.g. ../kokkos/.../libkokkoscore.a) are relative to
# $BUILD/src, the directory CMake itself runs this link command from -- match that.
OBJS=$(find "$BUILD/src/CMakeFiles/athena.dir" -name "*.cpp.o" ! -name "main.cpp.o")
(cd "$BUILD/src" && $CXX $CXX_FLAGS "$WORK/test_grwavespeed_alpha.o" $OBJS \
  -o "$WORK/test_grwavespeed_alpha" $LINK_LIBS)

echo "==> Running:"
"$WORK/test_grwavespeed_alpha"
