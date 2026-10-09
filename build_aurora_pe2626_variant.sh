#!/bin/bash
# Variant builds on the 2026-09 Aurora image with PE 26.26.0 (same PE logic and
# Kokkos flags as build_aurora_gpu_nexteval_pe2626.sh). Build trees live under
# /lus/flare/.../home_builds/athenak_tde/<name> with a symlink here; never wiped.
#   bash build_aurora_pe2626_variant.sh gpu_gauge   # dyngr_tov, SYCL/PVC -> build_gpu_pe2626_gauge
#   bash build_aurora_pe2626_variant.sh cpu_tests   # built-in pgens, Serial, no MPI -> build_cpu_pe2626_tests
set -e
[ -d /opt/aurora/26.181.0 ] || { echo "ERROR: not on the 2026-09 image ($(hostname))"; exit 1; }
module load oneapi/release/2025.3.1
hash -r
module load cmake
case "$(type -P icpx)" in /opt/aurora/26.26.0/*) ;; *) echo "ERROR: PE 26.26.0 not selected"; exit 1 ;; esac
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH=/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde
COMMON=( -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx -DGSL_ROOT_DIR="${HOME}/local"
         -DKokkos_ENABLE_SERIAL=ON )
case "$1" in
  gpu_gauge) NAME=build_gpu_pe2626_gauge
    EXTRA=( -DAthena_ENABLE_MPI=ON -DPROBLEM=dyn_grmhd/dyngr_tov -DKokkos_ENABLE_SYCL=ON
            -DKokkos_ENABLE_SYCL_RELOCATABLE_DEVICE_CODE=OFF -DKokkos_ARCH_INTEL_PVC=ON ) ;;
  cpu_tests) NAME=build_cpu_pe2626_tests
    EXTRA=( -DAthena_ENABLE_MPI=OFF -DKokkos_ARCH_SPR=ON ) ;;
  *) echo "usage: $0 gpu_gauge|cpu_tests"; exit 1 ;;
esac
mkdir -p "${SCRATCH}/${NAME}"
[ -e "${SRC}/${NAME}" ] || ln -s "${SCRATCH}/${NAME}" "${SRC}/${NAME}"
cmake -S "$SRC" -B "${SCRATCH}/${NAME}" "${COMMON[@]}" "${EXTRA[@]}"
make -C "${SCRATCH}/${NAME}" -j${NJOBS:-16}
ls -lh "${SCRATCH}/${NAME}/src/athena"
