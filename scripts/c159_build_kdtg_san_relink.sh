#!/bin/bash
# c159: the device-sanitizer binaries exceed the 2 GB small-code-model limit at link (relocation truncated);
# relink with -flink-huge-device-code (device images in a separate section). Objects are reused.
#   build_kdtg_san_relink.sh dasan|dmsan
set -e
module load oneapi/release/2025.3.1 >/dev/null 2>&1; hash -r; module load cmake >/dev/null 2>&1
S=/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde
case "$1" in dasan) F="-Xarch_device -fsanitize=address" ;; dmsan) F="-Xarch_device -fsanitize=memory" ;; *) exit 1 ;; esac
case "$(type -P icpx)" in /opt/aurora/26.26.0/*) ;; *) echo "ERROR: PE 26.26.0 not selected"; exit 1 ;; esac
B=build_gpu_pe2626_kdtg_new_$1
cmake $S/$B -DCMAKE_EXE_LINKER_FLAGS="$F -flink-huge-device-code" >> $S/$B.cmake.log 2>&1
make -C $S/$B -j${NJ:-16} athena > $S/$B.relink.log 2>&1
echo "$B relink rc=$?"
