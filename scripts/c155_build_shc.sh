#!/bin/bash
# R-177 item 3: the per-shell remnant counters (wip/shell-counters, from wip/r1-embed 516fee6e) and
# the 516fee6e reference, PROBLEM dyngr_tov (as build_r1_embed.sh).
#   build_shc.sh cpu|gpu new|ref [strict]   (cpu: serial, MPI off; strict: -fp-model=precise)
set -e
module load oneapi/release/2025.3.1 >/dev/null 2>&1; hash -r; module load cmake >/dev/null 2>&1
S=/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde
T=/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees
case "$2" in new) W=$T/shell_counters ;; ref) W=$T/r1_embed ;; *) echo "new|ref"; exit 1 ;; esac
if [ "$1" = cpu ]; then
  B=build_cpu_pe2626_shc_$2${3:+_$3}
  FPF=""; [ "${3:-}" = strict ] && FPF="-fp-model=precise"
  cmake -S $W -B $S/$B -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx -DGSL_ROOT_DIR=$HOME/local \
    -DCMAKE_CXX_FLAGS="$FPF" \
    -DKokkos_ENABLE_SERIAL=ON -DAthena_ENABLE_MPI=OFF -DKokkos_ARCH_SPR=ON \
    -DPROBLEM=dyn_grmhd/dyngr_tov > $S/$B.cmake.log 2>&1
else
  [ -d /opt/aurora/26.181.0 ] || { echo "ERROR: not on the 2026-09 image"; exit 1; }
  case "$(type -P icpx)" in /opt/aurora/26.26.0/*) ;; *) echo "ERROR: PE 26.26.0 not selected"; exit 1 ;; esac
  B=build_gpu_pe2626_shc_$2
  cmake -S $W -B $S/$B -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx -DGSL_ROOT_DIR="${HOME}/local" \
    -DKokkos_ENABLE_SERIAL=ON -DAthena_ENABLE_MPI=ON -DPROBLEM=dyn_grmhd/dyngr_tov -DKokkos_ENABLE_SYCL=ON \
    -DKokkos_ENABLE_SYCL_RELOCATABLE_DEVICE_CODE=OFF -DKokkos_ARCH_INTEL_PVC=ON > $S/$B.cmake.log 2>&1
fi
make -C $S/$B -j${NJ:-16} athena > $S/$B.make.log 2>&1
echo "$B rc=$?"
