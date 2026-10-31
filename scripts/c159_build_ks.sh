#!/bin/bash
# R-188 item 3 (Step B): wip/ks-fixed (metric_terms / killing_weights = kerr_schild) with the built-in pgens
# (no PROBLEM: <problem> pgen_name = gr_bondi).   build_ks.sh cpu [strict] | gpu
#   cpu: serial, MPI off, SPR (strict: -fp-model=precise, the logic gates without PVC codegen); gpu: PVC
set -e
module load oneapi/release/2025.3.1 >/dev/null 2>&1; hash -r; module load cmake >/dev/null 2>&1
S=/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde
W=/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/ks_fixed
if [ "$1" = cpu ]; then
  B=build_cpu_pe2626_ks${2:+_$2}
  FPF=""; [ "${2:-}" = strict ] && FPF="-fp-model=precise"
  cmake -S $W -B $S/$B -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx -DGSL_ROOT_DIR=$HOME/local \
    -DCMAKE_CXX_FLAGS="$FPF" -DKokkos_ENABLE_SERIAL=ON -DAthena_ENABLE_MPI=OFF -DKokkos_ARCH_SPR=ON > $S/$B.cmake.log 2>&1
else
  [ -d /opt/aurora/26.181.0 ] || { echo "ERROR: not on the 2026-09 image"; exit 1; }
  case "$(type -P icpx)" in /opt/aurora/26.26.0/*) ;; *) echo "ERROR: PE 26.26.0 not selected"; exit 1 ;; esac
  B=build_gpu_pe2626_ks
  cmake -S $W -B $S/$B -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx -DGSL_ROOT_DIR="${HOME}/local" \
    -DKokkos_ENABLE_SERIAL=ON -DAthena_ENABLE_MPI=ON -DKokkos_ENABLE_SYCL=ON \
    -DKokkos_ENABLE_SYCL_RELOCATABLE_DEVICE_CODE=OFF -DKokkos_ARCH_INTEL_PVC=ON > $S/$B.cmake.log 2>&1
fi
make -C $S/$B -j${NJ:-16} athena > $S/$B.make.log 2>&1
echo "$B rc=$?"
