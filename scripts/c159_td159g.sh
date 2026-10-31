#!/bin/bash -l
#PBS -A CompactBinaryMerger
#PBS -q debug-scaling
#PBS -l select=9
#PBS -l walltime=01:00:00
#PBS -l filesystems=flare:home
#PBS -l place=scatter
#PBS -k doe
#PBS -o /lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/c159_cap/td159g.out
#PBS -e /lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/c159_cap/td159g.err
# c159 rerun of the KF_z / KF_CFC flat gates (td159all's gZz/gCc decks carried killing_weights = boost and killing_closure = t1
# from the Valencia deck, fatal with killing_fz: mass_closure.cpp:184, dyn_grmhd.cpp:387; both keys removed), then B1Z on 8 nodes.
# A gc pass also writes PASS_GATE_gc.9001813.rerun.<job> so td159c's (9002125) waiting slot 2 runs B1C.
set -uo pipefail
D=/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc
Z=$D/c159_kfz; E=$D/c159_cap; X0=$Z/athena_tb
module load fftw/3.3.10
source /lus/flare/projects/CompactBinaryMerger/tlam/ensembles/c041_rerun67/aurora_guard.sh
aurora_image_guard
aurora_select_pe $X0
unset FI_LOG_LEVEL
source $D/tde_mpi_env.sh
CPU_BIND="list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100"
NF=$E/nodes.$PBS_JOBID; sort -u $PBS_NODEFILE > $NF
T0=$(date +%s)
run() {   # dir deckdir arm n0 nn
  local W=$1 DD=$2 a=$3 n0=$4 nn=$5 d=$1/$3 r left
  [ -e $W/DONE_$a ] && return 0
  left=$(( 3420 - ($(date +%s) - T0) ))
  [ $left -lt 300 ] && { echo "$a deferred (no time left) $PBS_JOBID" >> $W/runs.log; return 1; }
  mkdir -p $d && cd $d
  sed -n "${n0},$((n0 + nn - 1))p" $NF > $W/nodes.$a
  r=$(ls $d/rst/*.rst 2>/dev/null | LC_ALL=C sort | tail -1)
  [ -z "$r" ] && [ -e $W/FROM_$a ] && r=$(cat $W/FROM_$a)
  echo "$a start $(date -u +%FT%TZ) job $PBS_JOBID from ${r:-t=0} bin $(sha256sum $X0 | cut -c1-8) ranks $((12*nn))" >> $W/runs.log
  mpiexec -np $((12*nn)) -ppn 12 --hostfile $W/nodes.$a --cpu-bind="$CPU_BIND" /soft/tools/mpi_wrapper_utils/gpu_tile_compact.sh \
    $X0 -d $d -t $(printf '%02d:%02d:00' $(( (left - 120)/3600 )) $(( ((left - 120)%3600)/60 ))) ${r:+-r $r} -i $DD/$a.par >> out.log 2>> err.log
  echo "$a rc=$? $(grep -o 'Terminating on [a-z -]*' out.log | tail -1) last=$(grep -o 'cycle=[0-9]* time=[0-9.e+]*' out.log | tail -1) segv=$(grep -c SEGV_BACKTRACE err.log) end $(date -u +%FT%TZ)" >> $W/runs.log
  case "$(grep -o 'Terminating on [a-z -]*' out.log | tail -1)" in *"time limit"*|*"cycle limit"*) touch $W/DONE_$a ;; esac
}
gate() {   # g
  if python3 $Z/gate_$1.py $Z >> $Z/runs.log 2>&1; then
    touch $Z/PASS_GATE_$1.$PBS_JOBID $Z/PASS_GATE_$1.9001813.rerun.$PBS_JOBID; echo "GATE_$1 (rerun) passed $(date -u +%FT%TZ) $PBS_JOBID" >> $Z/runs.log
  else touch $Z/FAILED_GATE_$1.rerun.$PBS_JOBID; echo "GATE_$1 (rerun) failed $(date -u +%FT%TZ) $PBS_JOBID" >> $Z/runs.log; fi
}
( run $Z $Z gZz1 1 1; run $Z $Z gZz20 1 1; gate gz; run $Z $Z gCc1 1 1; run $Z $Z gCc20 1 1; gate gc ) &
( while [ ! -e $Z/PASS_GATE_gz.$PBS_JOBID ] && [ ! -e $Z/FAILED_GATE_gz.rerun.$PBS_JOBID ] && [ $(( $(date +%s) - T0 )) -lt 900 ]; do sleep 10; done
  if [ -e $Z/PASS_GATE_gz.$PBS_JOBID ]; then run $E $E B1Z 2 8; else echo "B1Z: gz did not pass in $PBS_JOBID, not run $(date -u +%FT%TZ)" >> $E/runs.log; fi ) &
wait
