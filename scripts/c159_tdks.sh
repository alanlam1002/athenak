#!/bin/bash -l
#PBS -A CompactBinaryMerger
#PBS -q debug-scaling
#PBS -l select=9
#PBS -l walltime=01:00:00
#PBS -l filesystems=flare:home
#PBS -l place=scatter
#PBS -k doe
#PBS -o /lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/c159_ks/td159k.out
#PBS -e /lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/c159_ks/td159k.err
# c159 (R-188 item 3, Step B, user-approved 2026-10-31): dyn_grmhd on the FIXED analytic Kerr-Schild metric, Michel (Step A's
# B.2 mesh/params, ppmx + HLLE + FOFC, RK3), t = 0 -> 120; arms V (valencia), T (killing_t1), A (killing_ta) x L2/L3/L4, KS weights;
# tau budget (shells) t = 100-120; debug-scaling 9 nodes x 1 h; binary athena_ks = wip/ks-fixed 35287253 (built-in pgens):
#   L4 arms on 2 nodes each; L3 then L2 per arm on 1 node
set -uo pipefail
D=/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc
E=$D/c159_ks
X0=$E/athena_ks
module load fftw/3.3.10
source /lus/flare/projects/CompactBinaryMerger/tlam/ensembles/c041_rerun67/aurora_guard.sh
aurora_image_guard
aurora_select_pe $X0
unset FI_LOG_LEVEL
source $D/tde_mpi_env.sh
CPU_BIND="list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100"
sort -u $PBS_NODEFILE > $E/nodes.$PBS_JOBID
T0=$(date +%s)
slot() {   # first-node-line nnodes arms...
  local n0=$1 nn=$2; shift 2
  for a in "$@"; do
    local d=$E/$a r left
    case $a in GATE_*)
      python3 $E/gate_${a#GATE_}.py $E >> $E/runs.log 2>&1 || { echo "$a failed: slot stops $(date -u +%FT%TZ)" >> $E/runs.log; touch $E/FAILED_$a.$PBS_JOBID; return; }
      touch $E/PASS_$a.$PBS_JOBID; echo "$a passed $(date -u +%FT%TZ)" >> $E/runs.log
      continue ;;
    WAIT_*)
      local gw=GATE_${a#WAIT_} wn=0
      while [ ! -e $E/PASS_$gw.$PBS_JOBID ] && [ ! -e $E/FAILED_$gw.$PBS_JOBID ] && [ $wn -lt 150 ]; do sleep 10; wn=$((wn + 1)); done
      [ -e $E/PASS_$gw.$PBS_JOBID ] || { echo "$a: no pass, slot stops $(date -u +%FT%TZ)" >> $E/runs.log; return; }
      continue ;;
    esac
    [ -e $E/DONE_$a ] && continue
    local X=$X0; [ -e $E/BIN_$a ] && X=$(cat $E/BIN_$a)
    left=$(( 3420 - ($(date +%s) - T0) ))
    [ $left -lt 300 ] && { echo "$a deferred (no time left) $PBS_JOBID" >> $E/runs.log; continue; }
    mkdir -p $d && cd $d
    sed -n "${n0},$((n0 + nn - 1))p" $E/nodes.$PBS_JOBID > $E/nodes.$a
    r=$(ls $d/rst/*.rst 2>/dev/null | LC_ALL=C sort | tail -1)
    if [ -z "$r" ] && [ -e $E/FROM_$a ]; then
      local f; f=$(cat $E/FROM_$a)
      case $f in *.rst) r=$f ;; *) r=$(ls $E/$f/rst/*.rst 2>/dev/null | LC_ALL=C sort | tail -1) ;; esac
      [ -n "$r" ] || { echo "$a: FROM_$a ($f) has no restart, skipped $(date -u +%FT%TZ)" >> $E/runs.log; continue; }
    fi
    echo "$a start $(date -u +%FT%TZ) job $PBS_JOBID from ${r:-t=0} bin $(sha256sum $X | cut -c1-8) ranks $((12*nn))" >> $E/runs.log
    mpiexec -np $((12*nn)) -ppn 12 --hostfile $E/nodes.$a --cpu-bind="$CPU_BIND" /soft/tools/mpi_wrapper_utils/gpu_tile_compact.sh \
      $X -d $d -t $(printf '%02d:%02d:00' $(( (left - 120)/3600 )) $(( ((left - 120)%3600)/60 ))) ${r:+-r $r} -i $E/decks/$a.par >> out.log 2>> err.log
    local rc=$?
    echo "$a rc=$rc $(grep -o 'Terminating on [a-z -]*' out.log | tail -1) last=$(grep -o 'cycle=[0-9]* time=[0-9.e+]*' out.log | tail -1) segv=$(grep -c SEGV_BACKTRACE err.log) end $(date -u +%FT%TZ)" >> $E/runs.log
    case "$(grep -o 'Terminating on [a-z -]*' out.log | tail -1)" in *"time limit"*|*"cycle limit"*) touch $E/DONE_$a ;; esac
  done
}
( slot 1 2 SBV_L4 ) &
( slot 3 2 SBT_L4 ) &
( slot 5 2 SBA_L4 ) &
( slot 7 1 SBV_L3 SBV_L2 ) &
( slot 8 1 SBT_L3 SBT_L2 ) &
( slot 9 1 SBA_L3 SBA_L2 ) &
wait
