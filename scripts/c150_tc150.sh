#!/bin/bash -l
#PBS -A CompactBinaryMerger
#PBS -q debug
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -l filesystems=flare:home
#PBS -l place=scatter
#PBS -k doe
#PBS -o /lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/c150_rframe/tc150.out
#PBS -e /lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/c150_rframe/tc150.err
# c150 (R-168, user-approved 2026-10-30): level R (riemann_frame) tests on wip/riemann-frame, plain debug.
# An arm named GATE runs u1_gate.py, GATE_<name> runs gate_<name>.py; if a gate fails, the slot stops.
# WAIT_<gate> waits for a gate of the other slot (PASS_/FAILED_<gate>.<jobid> markers).
# SLOT1 / SLOT2 (qsub -v): space-separated arm lists, one per node (12 ranks); each slot runs its arms in turn, each
# resuming from its latest restart (LC_ALL=C) or t = 0, with -t from the job's remaining walltime.
# X (optional, qsub -v): the binary (default: the killing build copied here).
set -uo pipefail
D=/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc
E=$D/c150_rframe
X=${X:-$E/athena_rframe}
module load fftw/3.3.10
source /lus/flare/projects/CompactBinaryMerger/tlam/ensembles/c041_rerun67/aurora_guard.sh
aurora_image_guard
aurora_select_pe $X
unset FI_LOG_LEVEL
source $D/tde_mpi_env.sh
CPU_BIND="list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100"
sort -u $PBS_NODEFILE > $E/nodes.$PBS_JOBID
T0=$(date +%s)
slot() {   # node-line arms...
  local n=$1; shift
  for a in "$@"; do
    local d=$E/$a r left
    case $a in GATE*)   # GATE: u1_gate.py; GATE_<name>: gate_<name>.py; a failed gate stops the slot
      local gs=$E/u1_gate.py; [ "$a" != GATE ] && gs=$E/gate_${a#GATE_}.py
      python3 $gs $E >> $E/runs.log 2>&1 || { echo "$a failed: slot stops $(date -u +%FT%TZ)" >> $E/runs.log; touch $E/FAILED_$a.$PBS_JOBID; return; }
      touch $E/PASS_$a.$PBS_JOBID
      continue ;;
    WAIT_*)   # WAIT_<gate>: wait (<= 20 min) for the other slot's gate of this job; stop if it failed
      local gw=${a#WAIT_} n=0
      while [ ! -e $E/PASS_$gw.$PBS_JOBID ] && [ ! -e $E/FAILED_$gw.$PBS_JOBID ] && [ $n -lt 120 ]; do sleep 10; n=$((n + 1)); done
      [ -e $E/PASS_$gw.$PBS_JOBID ] || { echo "$a: no pass, slot stops $(date -u +%FT%TZ)" >> $E/runs.log; return; }
      continue ;;
    esac
    [ -e $E/DONE_$a ] && continue
    left=$(( 3420 - ($(date +%s) - T0) ))   # 57 min of the hour
    [ $left -lt 300 ] && { echo "$a deferred (no time left) $PBS_JOBID" >> $E/runs.log; continue; }
    mkdir -p $d && cd $d
    sed -n "${n}p" $E/nodes.$PBS_JOBID > $E/nodes.$a
    r=$(ls $d/rst/*.rst 2>/dev/null | LC_ALL=C sort | tail -1)
    local x=$X; [ -e $E/BIN_$a ] && x=$(cat $E/BIN_$a)   # c149: a per-arm binary
    echo "$a start $(date -u +%FT%TZ) job $PBS_JOBID from ${r:-t=0} bin $(sha256sum $x | cut -c1-8)" >> $E/runs.log
    mpiexec -np 12 -ppn 12 --hostfile $E/nodes.$a --cpu-bind="$CPU_BIND" /soft/tools/mpi_wrapper_utils/gpu_tile_compact.sh \
      $x -d $d -t $(printf '%02d:%02d:00' $(( (left - 120)/3600 )) $(( ((left - 120)%3600)/60 ))) ${r:+-r $r} -i $E/$a.par >> out.log 2>> err.log
    local rc=$?
    echo "$a rc=$rc $(grep -o 'Terminating on [a-z -]*' out.log | tail -1) last=$(grep -o 'cycle=[0-9]* time=[0-9.e+]*' out.log | tail -1) segv=$(grep -c SEGV_BACKTRACE err.log) end $(date -u +%FT%TZ)" >> $E/runs.log
    case "$(grep -o 'Terminating on [a-z -]*' out.log | tail -1)" in *"time limit"*) touch $E/DONE_$a ;; esac
  done
}
( slot 1 ${SLOT1:-} ) &
( slot 2 ${SLOT2:-} ) &
wait
