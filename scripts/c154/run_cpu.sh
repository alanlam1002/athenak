#!/bin/bash
# c154 CPU gates (serial, MPI off, no PVC codegen): ref = e7ac4704 (wip/riemann-frame), new = wip/killing-dtg.
#   run_cpu.sh CASE BIN(ref|new) DECK [extra athena args]
set -u
E=$(cd $(dirname $0) && pwd)
c=$1; b=$2; p=$3; shift 3
X=/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde/build_cpu_pe2626_kdtg_$b/src/athena
mkdir -p $E/$c && cd $E/$c
cp $X ./athena_$b 2>/dev/null; md5sum ./athena_$b > bin.md5
echo "$c start $(date -u +%FT%TZ) bin $b $(cut -c1-8 bin.md5) deck $p $*" >> $E/runs.log
OMP_NUM_THREADS=1 ./athena_$b -i $E/$p -d $E/$c "$@" > out.log 2> err.log
echo "$c rc=$? $(grep -o 'Terminating on [a-z -]*' out.log | tail -1) last=$(grep -o 'cycle=[0-9]* time=[0-9.e+-]*' out.log | tail -1) end $(date -u +%FT%TZ)" >> $E/runs.log
