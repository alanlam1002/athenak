#!/bin/bash
# R-186 item 2: sanitizer builds of wip/killing-dtg (the Mm moving-bed deck's post-regrid failure), PROBLEM dyngr_tov.
#   build_kdtg_san.sh cpu asan          serial, MPI off, host AddressSanitizer (-O1 -g)
#   build_kdtg_san.sh gpu dasan|dmsan   PVC, device AddressSanitizer / MemorySanitizer (Intel's memcheck / initcheck), -g
set -e
module load oneapi/release/2025.3.1 >/dev/null 2>&1; hash -r; module load cmake >/dev/null 2>&1
S=/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde
W=/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/killing_dtg
if [ "$1" = cpu ]; then
  B=build_cpu_pe2626_kdtg_new_asan; F="-fsanitize=address -fno-omit-frame-pointer -g -O1"
  cmake -S $W -B $S/$B -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx -DGSL_ROOT_DIR=$HOME/local \
    -DCMAKE_CXX_FLAGS="$F" -DCMAKE_CXX_FLAGS_RELEASE="-O1 -g" -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address" \
    -DKokkos_ENABLE_SERIAL=ON -DAthena_ENABLE_MPI=OFF -DKokkos_ARCH_SPR=ON \
    -DPROBLEM=dyn_grmhd/dyngr_tov > $S/$B.cmake.log 2>&1
else
  case "$2" in dasan) F="-Xarch_device -fsanitize=address" ;; dmsan) F="-Xarch_device -fsanitize=memory" ;;
    *) echo "dasan|dmsan"; exit 1 ;; esac
  [ -d /opt/aurora/26.181.0 ] || { echo "ERROR: not on the 2026-09 image"; exit 1; }
  case "$(type -P icpx)" in /opt/aurora/26.26.0/*) ;; *) echo "ERROR: PE 26.26.0 not selected"; exit 1 ;; esac
  B=build_gpu_pe2626_kdtg_new_$2
  cmake -S $W -B $S/$B -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx -DGSL_ROOT_DIR="${HOME}/local" \
    -DCMAKE_CXX_FLAGS="$F -g" -DCMAKE_EXE_LINKER_FLAGS="$F" \
    -DKokkos_ENABLE_SERIAL=ON -DAthena_ENABLE_MPI=ON -DPROBLEM=dyn_grmhd/dyngr_tov -DKokkos_ENABLE_SYCL=ON \
    -DKokkos_ENABLE_SYCL_RELOCATABLE_DEVICE_CODE=OFF -DKokkos_ARCH_INTEL_PVC=ON > $S/$B.cmake.log 2>&1
fi
make -C $S/$B -j${NJ:-16} athena > $S/$B.make.log 2>&1
echo "$B rc=$?"
